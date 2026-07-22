# latentspace

*One evolutionary algorithm for any problem.*

A universal genetic algorithm never mutates the solution itself — because
"add noise to the pixels" only exists when the solution is pixels. Instead,
every individual carries a small **genome** plus its own **decoder
network**, and evolution operates only on those (they're just tensors, so
the same operators work for any problem). The decoder translates genome →
solution. The only problem-specific things you supply are the **fitness
function**, the **output shape**, and optionally the **decoder
architecture** — the same things you'd have to supply anyway.

```python
from latentspace import solve

def fitness(phenotypes):           # torch tensor (B, 32, 32), values in [0, 1]
    return -((phenotypes - target) ** 2).flatten(1).mean(dim=1)

result = solve(fitness, output_shape=(32, 32), budget=5_000)
result.best_phenotype              # (32, 32) array — the best solution found

# Many problems? Same call — pass a list and they share one population
# (measured: beats separate runs at every equal-per-problem budget):
result = solve([fit_a, fit_b, fit_c], output_shape=(32, 32), budget=30_000)
result.problems[1].best_phenotype
```

**The evidence** (paired seeds, identical evaluation budgets; full
campaign in [FINDINGS.md](FINDINGS.md)): given nothing but a fitness
function, pure decoder evolution beats a traditional GA with hand-matched
mutation on hidden-image benchmarks (10 runs to 0), beats a
50-city-tour GA with segment-reversal mutation (9 runs to 1, +33% at 100
cities), and — with the self-tuning mutation step that is now the
default — beats CMA-ES on a 256-dim smooth curve, CMA-ES's own home
terrain, plus every hybrid that uses CMA-ES internally. Sixty-plus
falsified ideas along the way are documented next to the wins.

## How it works

`solve` runs three phases against one evaluation budget:

```
EXPLORE   Population of 32 individuals, each = genome + private decoder
          weights. Children are born by crossover, then both tensors
          mutate. 32 independent lineages make 32 independent kinds of
          mistake — on purpose. Mutation size steers itself by
          Rechenberg's 1/5th rule: grow the step while >20% of children
          beat their parents, shrink it otherwise. One run spans a ~150x
          step range, sledgehammer to chisel, with no problem-specific
          constants. Restarts fresh when improvement stalls, until the
          budget is spent.

DISTILL   (optional, exploit="ga") Compress the run's best few hundred
          fitness-vetted solutions into a small linear latent space
          (PCA in logit space). Independent errors cancel; shared
          structure survives.

EXPLOIT   (optional, exploit="ga") A latent-space GA — truncation
          selection, uniform crossover, win-rate mutation — evolves
          genotypes feeding the distilled decoder with the remaining
          budget. CMA-ES appears in this project ONLY as a baseline to
          beat; it is not part of the solver.
```

At no point in any phase is a phenotype mutated, crossed, or otherwise
operated on — solutions are only ever *computed* from a genotype through a
decoder, then scored. Exploration searches (genome, decoder-weight) pairs;
exploitation searches genomes feeding the distilled decoder. That invariant
is what makes the algorithm universal.

Pure decoder evolution end-to-end is the measured robust default: the
exploit ablation found the distilled space itself fails on ~3/10 image
seeds (any optimizer confined to it is trapped — GA-exploit and
CMA-exploit failures correlate 0.97), while pure evolution passed every
seed, at negligible cost on smooth signals.

## Decoder architectures

The decoder's *shape* may match the output modality — that's where problem
structure legally lives. An untrained convolutional network already tends
to produce smooth, locally-coherent outputs, which is why the image
benchmark result holds; the architecture prior alone was worth 23%.

```python
solve(fitness, output_shape=(32, 32), architecture="conv2d")  # images
solve(fitness, output_shape=(256,),   architecture="conv1d")  # signals
solve(fitness, output_shape=(10,),    architecture="mlp")     # fallback
solve(fitness, output_shape=(32, 32))                         # "auto": by shape
```

Register your own (transformers, GRUs, procedural decoders — anything
mapping a `(B, latent)` tensor to `(B, prod(output_shape))` logits):

```python
from latentspace.universal import register_architecture

register_architecture("my-arch", lambda latent, shape: MyNet(latent, shape))
solve(fitness, output_shape=(64, 64), architecture="my-arch")
```

## Tuning knobs (all optional)

```python
solve(
    fitness, output_shape=(32, 32),
    budget=5_000,             # exact number of fitness evaluations
    latent=64,                # genome + distilled search-space dimension.
                              #   Benchmarked: cliff below ~32, broad
                              #   plateau 32-128, best means/variance at 64.
                              #   Scale with solution variety, not output
                              #   size; keep distill_top >= ~3x latent.
    explore_fraction="auto",  # stall-based switch (benchmarked better than
                              #   any fixed split); or a float like 0.6
    distill_top=200,          # solutions compressed into the latent space
    device="auto",            # mps / cuda / cpu
    seed=0,                   # full determinism per seed
)
```

## When you have related problems (pretraining)

Given many instances of a problem *family*, a decoder pretrained on pooled
solutions from cheap practice runs beats direct search 1.3–3.3× on fresh
instances, improving lawfully with practice count (parity at ~16–32
instances). The building blocks are exposed:

```python
from latentspace.universal import distill, cma_minimize

space = distill(pooled_solutions, latent=32, output_shape=(256,))
# freeze; then per new instance: CMA-ES over space.decode(z)
```

The two regimes — single-run and lifetime accumulation — are one system at
two timescales; unifying them is open problem #4 in FINDINGS.md.

## Run it

```bash
pip install -e ".[dev]"
python3 tests/test_universal.py          # CPU, seconds
python -m benchmarks.round18_adaptive --budget 5000   # the headline result (MPS)
```

The benchmark suite (`benchmarks/`) is the seventeen-round campaign that
produced this design: every claim above has a numbered round, exact
budgets, paired seeds, and raw per-seed JSON in `benchmark_results/`.
Start with [FINDINGS.md](FINDINGS.md) for the narrative — including what
was falsified, which is most of the original idea.

## The research API (legacy)

The original co-evolving `Evolver` — a latent GA whose single decoder
trains during the run — remains available (`from latentspace import
Evolver`) along with its trainer strategies and layer pipeline. The
campaign found its central mechanism doesn't work: a decoder trained on
solutions found by searching through itself cannot learn new structure
(FINDINGS.md, "the self-referentiality principle"). It is kept as the
research spine for studying exactly that failure mode, not as the
recommended solver.

## Open problems

1. Variance on smooth-signal problems (the tie should be a win).
2. More architectures: transformers, GRUs, permutation-aware decoders —
   the last may finally crack discrete problems.
3. Lifetime memory: bank vetted solutions across real solves, one shared
   decoder, without the echo-chamber failure (FINDINGS.md, the
   error-independence law).
