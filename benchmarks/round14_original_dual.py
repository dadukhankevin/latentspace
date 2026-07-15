"""Round 14: the ORIGINAL algorithm, plus dual repulsive decoders. Nothing else.

This is a retrial of the original genespace/latentspace idea on its own
terms: the package's unmodified `Evolver` — its latent GA (rank selection,
crossover, fixed-sigma mutation), its population lifecycle, its
self-supervised decoder training — with NO pretraining, NO practice
instances, NO PCA, NO CMA-ES, and none of the campaign's epoch/harvest
scaffolding. The single new ingredient is the round-13 mechanism, expressed
in the original vocabulary as one `decoder_update` layer:

  TWO decoders. The population evolves through one (the active). Every
  `refine_every` generations, the OTHER decoder is trained to reproduce the
  active decoder's phenotypes for the best genes — original-style
  self-supervision, except the targets come from a decoder that is not the
  student. Their weights then step 2% apart (the repulsion), and the
  freshly trained decoder becomes the active one. The package's own
  version-stale machinery re-scores the population after every hand-off.

Arms:

  * direct_ga       — the traditional-GA bar (reference);
  * latent_fixed    — original claim: frozen random decoder + latent GA;
  * latent_gradient — original co-evolution: SELF_DISTILL refits every
                      `refine_every` generations;
  * dual            — two decoders, cross-distillation hand-offs, no
                      repulsion (isolates the training rule);
  * dual_repel      — the full round-13 mechanism.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from benchmarks.compare import (
    BenchmarkConfig,
    Objective,
    TrackedFitness,
    _finish_result,
    _require_mps,
    _seed_everything,
    print_summary,
    run_direct_ga,
    summarize,
)
from benchmarks.round3_structure import SmoothTarget
from benchmarks.round8_mlp_pretrain import BlobImage2D
from latentspace.core import Layer
from latentspace.decoder import MLPDecoder
from latentspace.evolver import Evolver

OBJECTIVES: dict[str, Callable[[], Objective]] = {
    "smooth1d_256": SmoothTarget,
    "blob2d_1024": BlobImage2D,
}

REPEL = 0.02
VERSION_OFFSET = 1_000_000      # keeps the two decoders' version counters
                                # disjoint so hand-offs always read as stale


class DualRepelDecoders(Layer):
    """Two co-evolving decoders; neither ever trains on its own outputs.

    Placed as the Evolver's `decoder_update` layer, so the original
    pipeline's trailing DecodeAndEvaluate + Sort re-score the population
    through whichever decoder is active after the hand-off.
    """

    def __init__(self, other: MLPDecoder, every: int = 5, percent: float = 0.4,
                 batch_size: int = 32, repel: float = 0.02):
        super().__init__()
        self.other = other
        self.every, self.percent = every, percent
        self.batch_size, self.repel = batch_size, repel
        self._calls = 0
        self.hand_offs = 0

    def __call__(self, pop):
        self._calls += 1
        if len(pop) < 2 or self._calls % self.every:
            return pop
        teacher = self.env.decoder          # produced this population
        student = self.other
        genes = np.stack([ind.genes for ind in pop]).astype(np.float32)
        k = max(1, int(len(pop) * self.percent))
        inputs = torch.as_tensor(genes[:k], device=student.device)
        with torch.no_grad():
            targets = teacher.forward(genes[:k]).detach()
        for i in range(0, k, self.batch_size):
            xb, yb = inputs[i:i + self.batch_size], targets[i:i + self.batch_size]
            student.opt.zero_grad()
            loss = student.loss_fn(student.forward(xb), yb)
            loss.backward()
            student.opt.step()
        student.mark_updated()
        if self.repel:
            with torch.no_grad():
                for pa, pb in zip(teacher.net.parameters(),
                                  student.net.parameters()):
                    delta = self.repel * (pa - pb)
                    pa.add_(delta)
                    pb.sub_(delta)
            teacher.mark_updated()
        self.env.decoder = student
        self.other = teacher
        self.hand_offs += 1
        return pop


def run_arm(objective, seed, config, arm):
    _require_mps()
    _seed_everything(seed)

    if arm == "direct_ga":
        return run_direct_ga(objective, seed, config)

    tracker = TrackedFitness(objective)
    common = dict(
        output_shape=(objective.dimension,), device="mps",
        latent=config.latent, population=config.population,
        hidden_size=config.hidden_size, num_layers=config.num_layers,
        lr=config.decoder_lr, mutation_rate=config.mutation_rate,
        mutation_sigma=config.mutation_sigma, pressure=1.8, scheme="linear",
        families=max(1, config.offspring // 4), children=4, n_points=4,
        offspring_only_mutation=config.offspring_only_mutation,
    )
    if arm == "latent_fixed":
        evolver = Evolver(tracker, refine_every=None, **common)
    elif arm == "latent_gradient":
        evolver = Evolver(tracker, refine_every=config.refine_every,
                          refine_percent=config.refine_percent, **common)
    elif arm in ("dual", "dual_repel"):
        decoder_b = MLPDecoder(
            input_length=config.latent, output_shape=(objective.dimension,),
            hidden_size=config.hidden_size, num_layers=config.num_layers,
            lr=config.decoder_lr, device="mps")
        decoder_b.version = VERSION_OFFSET
        layer = DualRepelDecoders(
            decoder_b, every=config.refine_every,
            percent=config.refine_percent,
            repel=REPEL if arm == "dual_repel" else 0.0)
        evolver = Evolver(tracker, refine_every=None, decoder_update=layer,
                          **common)
    else:
        raise ValueError(arm)

    started = time.perf_counter()
    while tracker.evaluations < config.evaluation_budget:
        evolver.solve(1, verbose_every=0)
    torch.mps.synchronize()
    result = _finish_result(objective, arm, seed, config, tracker, started,
                            generations=evolver.env.generation,
                            neural_device="mps")
    torch.mps.empty_cache()
    return result


STRATEGIES = ("direct_ga", "latent_fixed", "latent_gradient", "dual",
              "dual_repel")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--objectives", nargs="+", choices=OBJECTIVES, default=list(OBJECTIVES)
    )
    parser.add_argument("--strategies", nargs="+", choices=STRATEGIES,
                        default=list(STRATEGIES))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(3)))
    parser.add_argument("--budget", type=int, default=5_000)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    config = BenchmarkConfig(evaluation_budget=args.budget)
    _require_mps()
    results = []
    for objective_name in args.objectives:
        for arm in args.strategies:
            for seed in args.seeds:
                objective = OBJECTIVES[objective_name]()
                print(
                    f"run objective={objective_name:<14} arm={arm:<15} "
                    f"seed={seed} budget={config.evaluation_budget}",
                    flush=True,
                )
                result = run_arm(objective, seed, config, arm)
                print(f"  {result.metric}={result.metric_at_budget:.6g}",
                      flush=True)
                results.append(result)
    summary = summarize(results)
    print_summary(summary)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": asdict(config),
            "repel": REPEL,
            "search": "original Evolver latent GA (no CMA-ES, no pretraining)",
            "torch_version": torch.__version__,
            "runs": [asdict(r) for r in results],
            "summary": summary,
        }
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
