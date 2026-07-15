"""Evaluation-budgeted comparison of decoder learning strategies on MPS.

This isolates the decoder-training rule while keeping the latent GA, network,
objectives, and evaluation budget fixed. REINFORCE's sampled policy evaluations
are recorded by the same tracker and therefore consume the same budget as all
other objective calls.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from latentspace import (
    AdaptiveMixtureTrainer,
    AdvantageWeightedTrainer,
    ContrastiveTrainer,
    DistillationTrainer,
    Evolver,
    FrozenTrainer,
    GuardedTrainer,
    MixtureTrainer,
    PolicyGradientTrainer,
    TrainMode,
)

from .compare import (
    OBJECTIVES,
    BenchmarkConfig,
    TrackedFitness,
    _finish_result,
    _require_mps,
    _seed_everything,
    _warm_mps,
    print_summary,
    summarize,
)


def make_atomic_trainers(config: BenchmarkConfig):
    return {
        "bottom_to_top": DistillationTrainer(
            mode=TrainMode.SELF_DISTILL,
            percent=config.refine_percent,
        ),
        "good_to_best": DistillationTrainer(
            mode=TrainMode.GOOD_TO_BEST,
            percent=config.refine_percent,
        ),
        "each_to_next": DistillationTrainer(
            mode=TrainMode.EACH_TO_NEXT,
            percent=config.refine_percent,
        ),
        "contrastive_worst": ContrastiveTrainer(
            percent=config.refine_percent,
            margin=0.05,
            negative_weight=0.5,
        ),
        "reinforce": PolicyGradientTrainer(
            percent=0.25,
            samples_per_gene=4,
            exploration_std=0.1,
        ),
        "advantage_weighted": AdvantageWeightedTrainer(
            percent=0.25,
            samples_per_gene=4,
            exploration_std=0.1,
            temperature=0.5,
        ),
    }


def make_trainer(name: str, config: BenchmarkConfig, seed: int = 0):
    if name == "frozen":
        return FrozenTrainer()
    atomic = make_atomic_trainers(config)
    if name in atomic:
        return atomic[name]
    if name == "random_all":
        return MixtureTrainer(atomic, strategy="random", seed=seed)
    if name == "round_robin_all":
        return MixtureTrainer(atomic, strategy="round_robin", seed=seed)
    if name == "shuffled_cycle_all":
        return MixtureTrainer(atomic, strategy="shuffled_cycle", seed=seed)
    if name == "random_non_rl":
        non_rl = {
            key: atomic[key]
            for key in (
                "bottom_to_top",
                "good_to_best",
                "each_to_next",
                "contrastive_worst",
            )
        }
        return MixtureTrainer(non_rl, strategy="random", seed=seed)
    if name == "random_three_all":
        return MixtureTrainer(
            atomic, strategy="random", steps_per_call=3, seed=seed
        )
    if name == "shuffled_three_all":
        return MixtureTrainer(
            atomic, strategy="shuffled_cycle", steps_per_call=3, seed=seed
        )
    if name == "bottom_to_top_three":
        return DistillationTrainer(
            mode=TrainMode.SELF_DISTILL,
            percent=config.refine_percent,
            epochs=3,
        )
    if name == "each_to_next_three":
        return DistillationTrainer(
            mode=TrainMode.EACH_TO_NEXT,
            percent=config.refine_percent,
            epochs=3,
        )
    if name == "contrastive_worst_three":
        return ContrastiveTrainer(
            percent=config.refine_percent,
            margin=0.05,
            negative_weight=0.5,
            epochs=3,
        )
    if name == "reinforce_three":
        return PolicyGradientTrainer(
            percent=0.25,
            samples_per_gene=4,
            exploration_std=0.1,
            epochs=3,
        )
    if name == "advantage_weighted_three":
        return AdvantageWeightedTrainer(
            percent=0.25,
            samples_per_gene=4,
            exploration_std=0.1,
            temperature=0.5,
            epochs=3,
        )
    if name == "guarded_random_all":
        return GuardedTrainer(
            MixtureTrainer(atomic, strategy="random", seed=seed),
            probe_percent=0.25,
        )
    if name == "guarded_shuffled_all":
        return GuardedTrainer(
            MixtureTrainer(atomic, strategy="shuffled_cycle", seed=seed),
            probe_percent=0.25,
        )
    if name == "guarded_random_non_rl":
        non_rl = {
            key: atomic[key]
            for key in (
                "bottom_to_top",
                "good_to_best",
                "each_to_next",
                "contrastive_worst",
            )
        }
        return GuardedTrainer(
            MixtureTrainer(non_rl, strategy="random", seed=seed),
            probe_percent=0.25,
        )
    if name == "guarded_random_three_all":
        return GuardedTrainer(
            MixtureTrainer(
                atomic, strategy="random", steps_per_call=3, seed=seed
            ),
            probe_percent=0.25,
        )
    if name == "guarded_cost_aware_all":
        weights = {
            "bottom_to_top": 1.0,
            "good_to_best": 1.0,
            "each_to_next": 1.0,
            "contrastive_worst": 1.0,
            "reinforce": 0.25,
            "advantage_weighted": 0.25,
        }
        return GuardedTrainer(
            MixtureTrainer(
                atomic,
                strategy="random",
                seed=seed,
                weights=weights,
            ),
            probe_percent=0.25,
        )
    if name in {"adaptive_all_one", "adaptive_all_three"}:
        return AdaptiveMixtureTrainer(
            atomic,
            probe_percent=0.25,
            steps_per_call=1 if name.endswith("one") else 3,
            seed=seed,
        )
    if name == "adaptive_non_rl_three":
        non_rl = {
            key: atomic[key]
            for key in (
                "bottom_to_top",
                "good_to_best",
                "each_to_next",
                "contrastive_worst",
            )
        }
        return AdaptiveMixtureTrainer(
            non_rl,
            probe_percent=0.25,
            steps_per_call=3,
            seed=seed,
        )
    if name == "adaptive_cost_aware_three":
        priors = {
            "bottom_to_top": 1.0,
            "good_to_best": 1.0,
            "each_to_next": 1.0,
            "contrastive_worst": 1.0,
            "reinforce": 0.25,
            "advantage_weighted": 0.25,
        }
        return AdaptiveMixtureTrainer(
            atomic,
            probe_percent=0.25,
            steps_per_call=3,
            seed=seed,
            priors=priors,
        )
    if name == "adaptive_no_warmup_three":
        return AdaptiveMixtureTrainer(
            atomic,
            probe_percent=0.25,
            steps_per_call=3,
            seed=seed,
            warmup=False,
        )
    if name == "adaptive_freeze_three":
        with_frozen = dict(atomic)
        with_frozen["frozen"] = FrozenTrainer()
        return AdaptiveMixtureTrainer(
            with_frozen,
            probe_percent=0.25,
            steps_per_call=3,
            seed=seed,
            warmup=False,
        )
    raise ValueError(name)


TRAINERS = (
    "frozen",
    "bottom_to_top",
    "good_to_best",
    "each_to_next",
    "contrastive_worst",
    "reinforce",
    "advantage_weighted",
    "random_all",
    "round_robin_all",
    "shuffled_cycle_all",
    "random_non_rl",
    "random_three_all",
    "shuffled_three_all",
    "bottom_to_top_three",
    "each_to_next_three",
    "contrastive_worst_three",
    "reinforce_three",
    "advantage_weighted_three",
    "guarded_random_all",
    "guarded_shuffled_all",
    "guarded_random_non_rl",
    "guarded_random_three_all",
    "guarded_cost_aware_all",
    "adaptive_all_one",
    "adaptive_all_three",
    "adaptive_non_rl_three",
    "adaptive_cost_aware_three",
    "adaptive_no_warmup_three",
    "adaptive_freeze_three",
)


def run_trainer(objective, seed, config, trainer_name):
    _require_mps()
    _seed_everything(seed)
    tracker = TrackedFitness(objective)
    trainer = make_trainer(trainer_name, config, seed=seed)
    families = max(1, config.offspring // 4)
    evolver = Evolver(
        tracker,
        output_shape=(objective.dimension,),
        device="mps",
        latent=config.latent,
        population=config.population,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        lr=config.decoder_lr,
        mutation_rate=config.mutation_rate,
        mutation_sigma=config.mutation_sigma,
        refine_every=config.refine_every,
        refine_percent=config.refine_percent,
        pressure=1.8,
        scheme="linear",
        families=families,
        children=4,
        n_points=4,
        offspring_only_mutation=config.offspring_only_mutation,
        trainer=trainer,
    )
    parameter_devices = {
        parameter.device.type for parameter in evolver.decoder.parameters()
    }
    if parameter_devices != {"mps"}:
        raise RuntimeError(
            f"decoder parameters are not exclusively on MPS: {parameter_devices}"
        )

    started = time.perf_counter()
    while tracker.evaluations < config.evaluation_budget:
        evolver.solve(1, verbose_every=0)
    torch.mps.synchronize()
    result = _finish_result(
        objective,
        trainer_name,
        seed,
        config,
        tracker,
        started,
        generations=evolver.env.generation,
        neural_device="mps",
        trainer_choices=(tuple(trainer.history) if hasattr(trainer, "history") else None),
        trainer_acceptance=(
            tuple(trainer.acceptance_history)
            if isinstance(trainer, GuardedTrainer)
            else None
        ),
        trainer_rewards=(
            tuple(trainer.reward_history)
            if isinstance(trainer, AdaptiveMixtureTrainer)
            else None
        ),
        trainer_final_probabilities=(
            trainer.probabilities
            if isinstance(trainer, AdaptiveMixtureTrainer)
            else None
        ),
    )
    torch.mps.empty_cache()
    return result


def run_suite(objective_names, trainer_names, seeds, config):
    _require_mps()
    results = []
    for objective_name in objective_names:
        _warm_mps(OBJECTIVES[objective_name](), config)
        for trainer_name in trainer_names:
            for seed in seeds:
                objective = OBJECTIVES[objective_name]()
                print(
                    f"run objective={objective_name:<12} "
                    f"trainer={trainer_name:<20} seed={seed} "
                    f"budget={config.evaluation_budget}"
                )
                result = run_trainer(objective, seed, config, trainer_name)
                print(
                    f"  {result.metric}={result.metric_at_budget:.6g} "
                    f"evals_run={result.evaluations_run} "
                    f"generations={result.generations} device={result.neural_device}"
                )
                results.append(result)
    return results


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--objectives", nargs="+", choices=OBJECTIVES, default=list(OBJECTIVES)
    )
    parser.add_argument(
        "--trainers", nargs="+", choices=TRAINERS, default=list(TRAINERS)
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--budget", type=int, default=5_000)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    config = BenchmarkConfig(evaluation_budget=args.budget)
    results = run_suite(args.objectives, args.trainers, args.seeds, config)
    summary = summarize(results)
    print_summary(summary)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "study": "decoder_training",
            "config": asdict(config),
            "torch_version": torch.__version__,
            "mps_built": torch.backends.mps.is_built(),
            "mps_available": torch.backends.mps.is_available(),
            "trainers": list(args.trainers),
            "runs": [asdict(result) for result in results],
            "summary": summary,
        }
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
