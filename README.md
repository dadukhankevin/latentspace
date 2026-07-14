# latentspace

*One evolutionary algorithm for any problem.* (Working name — rename freely.)

Evolve a universal **latent vector**; a single **co-evolving neural decoder** maps
it to a phenotype of *any* shape. To solve a new problem you change two things —
the fitness function and the output shape — and nothing else.

```python
from latentspace import Evolver

ev = Evolver(fitness_fn, output_shape=(H, W, 3), device="cuda")
ev.solve(400)
phenotype = ev.decode_best()
```

## Why this is one algorithm

This is a synthesis of three earlier projects:

- **Finch** contributes the Keras-style **layer pipeline** and the idea that any
  hyperparameter can be a schedule (a callable).
- **GeneSpace** contributes the core bet: a **universal latent genotype** plus a
  **learned decoder** that produces any output shape. Because every genotype is
  just a fixed latent vector, almost all problem-specificity migrates into the
  fitness function and the decoder — so the genetic operators collapse to
  *"cross and mutate a vector."* That collapse is why it's simpler than Finch,
  and why it's a single algorithm.
- **Aulë** contributes the ontology: **everything is an `Individual`.** Solutions,
  the decoder, layers and the environment share one root type. The load-bearing
  payoff is that the **decoder is an `Individual` too**, so its two improvement
  channels (gradient descent and evolution strategies) are just first-class
  parts of the same system rather than a bolted-on branch.

## The pipeline

```
Populate           # top up the latent population
Crossover          # n-point crossover on the latent vector
Mutate             # gaussian (float) or bit-flip (binary), cache-invalidating
DecodeAndEvaluate  # batched: latent -> phenotype -> fitness
Sort
RefineDecoder      # every N gens: improve the decoder, bump its version
Cap
```

`Evolver` just compiles this. You can rebuild it by hand for full control.

## The decoder co-evolves (two channels)

- **Gradient (`RefineDecoder`, default `SELF_DISTILL`)** — trains the decoder so
  the *worst* individuals' latents decode toward the *best* individuals'
  phenotypes. Targets are the decoder's own outputs, so **no differentiable
  objective is required**; this warps the output manifold toward high-fitness
  regions over time. Also available: `GOOD_TO_BEST`, `EACH_TO_NEXT`.
- **Evolution strategy (`decoder.evolve_step`)** — random weight perturbations
  kept only if they raise population fitness, for when you want no
  self-supervision at all.

## The one subtlety worth knowing

A co-evolving decoder is a **non-stationary landscape**: when the decoder
changes, every previously computed fitness is measured under an old mapping.
The decoder carries a `version` counter that bumps on every weight update; each
individual records the version it was scored under; `DecodeAndEvaluate` only
re-scores individuals whose version is stale. That single counter handles both
mutated genes and a refreshed decoder, and it removes a whole class of
stale-fitness bugs for free.

## Design choices (v1)

- **Float latents** by default (`binary=True` available). Floats give the decoder
  a smoother signal to train against.
- **One decoder**, not a population. An MLP is a universal approximator, so one
  decoder is expressive enough; a population would only add exploration diversity
  of mappings, at the cost of a hard credit-assignment problem. Deferred.

## Run it

```bash
pip install -e .
python examples/quickstart.py   # matches a target vector AND solves TSP, same 2 lines each
```

## Not done yet

This is the spine — correct and general, not tuned. Next: elitism at the mutation
layer, proper benchmarks vs. representation-specific GAs, a refine-cadence /
learning-rate sweep (the decoder LR is the main stability knob), and only *then*
revisiting a population of competing decoders.
