# latentspace

*One genetic algorithm for any problem.*

A universal GA never mutates a solution directly — "add noise to the
pixels" only exists when the solution is pixels. Here, a solution is
always **computed**: every individual is a set of **genes** (the input a
shared decoder network reads) plus **latents** (a vector that bends that
network's behavior for this individual alone), and the decoder turns them
into the phenotype. Evolution operates only on genes and latents — they
are just numbers, so the same operators work for any modality. You supply
the **fitness function(s)**, the **output shape**, and optionally a
**decoder architecture**; nothing else is problem-specific.

```python
from latentspace.universal import solve

def fitness(phenotypes):        # torch tensor (B, 32, 32, 3), values in [0, 1]
    return -((phenotypes - target) ** 2).flatten(1).mean(dim=1)

result = solve(fitness, output_shape=(32, 32, 3), epochs=1_500)
result.best_phenotype           # the best solution found

# Many problems? Same call — each fitness function becomes a species
# sharing one population and one decoder:
result = solve([fit_a, fit_b, fit_c], output_shape=(32, 32, 3),
               epochs=5_000)
result.problems[1].best_phenotype
```

## How it works

Sixteen random founders per fitness function (`founders=` — the founding
count is the run's entire coverage of the space, since every later
individual descends from it; two was measured too few on plateau
objectives), then a classical GA loop with two twists that make it
universal:

```
SPECIES    Each fitness function is a species. Fitness is organized as
           SHARES: the whole environment's fitness mass is always 1,
           every species owns an equal slice, and individuals split
           their species' slice by rank. Selection and survival both run
           on shares — no species can take over, and a species down to
           one struggling member concentrates its whole slice there.
           Extinction is allowed.

BREEDING   Parents are drawn by share; the partner comes from the SAME
           species, except a rare (5%) outcross draws it from anywhere —
           measured to rescue stuck lineages while staying rare enough
           not to disrupt working ones. Genes cross by one-point cut;
           latents are inherited WHOLE from one parent (half a bending
           is not half as useful). Genes and latents mutate through
           separate operators with independent self-tuning step dials.

FOLD       Evolution computes update directions: every 32 epochs the
           population votes, coordinate by coordinate and weighted by
           fitness share, on which DIRECTION each decoder dimension
           should move — a majority vote that a single unrepresentative
           individual cannot drag (measured: same mean as absorbing the
           champion's latents, half the run-to-run spread). The step
           passes through an Adam accumulator and is applied DIRECTLY
           into the shared decoder's weights — exact arithmetic, no
           training, no gradients of the fitness function. The decoder
           itself learns what everyone keeps discovering.
```

No phase ever touches a phenotype, no operator treats the genes and
latents as one string of numbers, and every step above — selection, both
crossovers, both mutations, speciation, fold selection — is a replaceable
function you can pass into `solve()`.

**How latents modify the decoder** is itself a choice, because there is
always exactly one decoder and only the form of the small per-individual
modifier changes:

```python
solve(fitness, output_shape=(96, 96, 3), epochs=9_000,
      directions="sparse", latents=2048)   # sparse weight patches
```

`"frozen"` (default) gates a fixed set of random low-rank directions —
cheap, but every individual and every fold is confined to that subspace
forever. `"sparse"` instead gives each individual K freely-placed weight
coordinates (its seed picks them, its latents are the values added
there), so folds can reach any weight. Measured on the apple photo at
matched budget, three paired seeds: sparse at K=2048 beat frozen on all
three, mean MSE 0.0076 vs 0.0133. It is not yet the default — the
control runs separating "more evolvable numbers" from "free placement"
are single-seed so far, and the multi-function check is unrun.

**The evidence** (paired seeds, identical evaluation counts; the full
falsification-heavy campaign is in [FINDINGS.md](FINDINGS.md)): this
design, one day old, statistically ties the heavily tuned predecessor
engine that beat a traditional GA 10 runs to 0 on hidden images, beat a
tour GA from 100 cities up with widening margins, and beat CMA-ES on its
own smooth-curve home terrain — at roughly a tenth of the predecessor's
conceptual complexity. The predecessor engines live on in
[benchmarks/legacy_engines/](benchmarks/legacy_engines/) as the bars this
design has to clear outright: the single-fitness records (apple photo MSE
0.00178 at 150k evaluations) still stand and are the target. CMA-ES
appears in this project only as an opponent.

## Decoder architectures

```python
from latentspace.universal import register_architecture, solve

register_architecture("my-arch", lambda genes, shape: MyDecoder(genes, shape))
solve(fitness, output_shape=(64, 64), architecture="my-arch", epochs=2_000)
```

`architecture="auto"` picks a convolutional decoder for image shapes and a
generic network otherwise; both carry the shared low-rank directions that
latents gate. One measured trap: every decoder is born emitting a
near-constant phenotype. On images that is a good prior (widening it loses,
3/3 paired seeds); when the phenotype is GEOMETRY — coordinates, a weight
vector, anything whose meaning lives in the spread of its values — a
constant output is a degenerate solution and evolution has nothing to
select on. Register an architecture with its final layer scaled up (~10x)
for such problems; `benchmarks/probe_walker.py` is the worked example.

## Tuning (all measured, all replaceable)

```python
solve(
    fitness_fns, output_shape=(32, 32, 3),
    epochs=1_500,             # loop iterations; evaluations are counted and
                              #   reported (result.evaluations)
    genes=64, latents=64,     # sizes of the two spaces
    children=16,              # children per epoch
    founders=16,              # founders per function — the run's coverage
                              #   of the space; pass 2 for pre-2026-07-27
                              #   benchmark reproduction
    population_cap=32,        # raised automatically to hold all founders
    fold_every=32,            # Adam-fold cadence; fold_optimizer="raw"/"off"
    fold_selection=...,       # default sign-vote; champion/rank variants
                              #   in ga.py
    init_decoder=...,         # warm-start from a prior run's result.decoder
    selection=...,            # swap any operator; see ga.py for signatures
)
```

Environment: neural runs use Apple MPS when available. Reproduction
commands for every claim are in [FINDINGS.md](FINDINGS.md); the historical
benchmark scripts under `benchmarks/` run against the legacy engines.
