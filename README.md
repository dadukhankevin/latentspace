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

Two random founders per fitness function, then a classical GA loop with
two twists that make it universal:

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
           best individual of the largest species donates its latents,
           which pass through an Adam accumulator and are applied
           DIRECTLY into the shared decoder's weights — exact
           arithmetic, no training, no gradients of the fitness
           function. Momentum keeps the cross-species consensus; the
           second moment damps dimensions species disagree on. The
           decoder itself learns what everyone keeps discovering.
```

No phase ever touches a phenotype, no operator treats the genes and
latents as one string of numbers, and every step above — selection, both
crossovers, both mutations, speciation, fold selection — is a replaceable
function you can pass into `solve()`.

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
latents gate.

## Tuning (all measured, all replaceable)

```python
solve(
    fitness_fns, output_shape=(32, 32, 3),
    epochs=1_500,             # loop iterations; evaluations are counted and
                              #   reported (result.evaluations)
    genes=64, latents=64,     # sizes of the two spaces
    children=16,              # children per epoch
    population_cap=32,        # raised automatically to hold 2 per function
    fold_every=32,            # Adam-fold cadence; fold_optimizer="raw"/"off"
    selection=...,            # swap any operator; see ga.py for signatures
)
```

Environment: neural runs use Apple MPS when available. Reproduction
commands for every claim are in [FINDINGS.md](FINDINGS.md); the historical
benchmark scripts under `benchmarks/` run against the legacy engines.
