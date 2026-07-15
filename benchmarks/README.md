# Strategy comparison

This suite compares the latent framing against ordinary direct-search methods:

- uniform random search;
- a direct elitist genetic algorithm;
- a bounded direct `(mu + lambda)` evolution strategy with intermediate
  recombination and one-fifth-style step-size adaptation;
- SciPy differential evolution (`best1bin`);
- latent evolution through a fixed random MLP decoder;
- latent evolution with gradient self-distillation;
- latent evolution with perturb-and-select ES updates to decoder weights.

All strategies are compared at the **exact same number of objective-function
evaluations**. A run may finish its current generation after crossing the budget,
but its reported metric is the best value at the exact requested evaluation.
Wall-clock time and the actual number of evaluations executed are also retained.

The direct methods search bounded phenotypes in `[0, 1]`. The latent methods
search fixed-length genotypes whose MLP decoder emits `[0, 1]`. This intentionally
tests whether the learned representation earns its additional machinery; it is
not an assertion that the algorithms have equal compute cost.

## MPS requirement

Every neural strategy constructs and verifies its decoder on `mps`. The command
fails instead of silently falling back to CPU when MPS is unavailable. Direct
NumPy baselines remain on CPU because they contain no neural network.

```bash
pip install -e ".[bench]"
python -m benchmarks.compare \
  --budget 5000 \
  --seeds 0 1 2 3 4 \
  --output benchmark_results/mps.json
```

To isolate decoder learning rules while keeping the latent GA fixed:

```bash
python -m benchmarks.decoder_training \
  --budget 5000 \
  --seeds 0 1 2 3 4 \
  --output benchmark_results/decoder_training.json
```

That study includes a frozen control, bottom-to-top, good-to-best,
each-to-next, a contrastive loss that treats the worst phenotypes as negatives,
black-box REINFORCE, and lower-variance advantage-weighted regression. RL policy
samples consume objective evaluations, so neither method receives free rollouts.

Mixture names in the same runner combine those atomic trainers. `random_all`,
`round_robin_all`, and `shuffled_cycle_all` choose one objective at each decoder
update. `*_three_all` performs three sequential micro-batches. `guarded_*`
variants evaluate the proposed mapping on an elite probe and roll it back when
it improves neither probe mean nor probe best. Guard probes and rejected RL
rollouts still count toward the objective budget.

`adaptive_*` variants never roll back. They update recency-weighted objective
scores from normalized probe improvement, then use softmax allocation with an
exploration floor. `adaptive_freeze_three` also gives the allocator a no-op arm.

The initial objective set covers a smooth target match, multimodal Rastrigin,
and a random-key traveling-salesperson problem. Lower metrics are always better.

TSP size scaling uses the same nested Euclidean instance family, five algorithm
seeds, and a fixed evaluation budget:

```bash
python -m benchmarks.tsp_scaling \
  --sizes 8 12 16 24 32 \
  --budget 5000 \
  --seeds 0 1 2 3 4 \
  --output benchmark_results/mps_tsp_scaling_5000.json
```

Add `--offspring-only-mutation` to run the elitist `(mu + lambda)` ablation that
preserves incumbents and mutates only fresh children. The default retains the
whole-population mutation used by the existing benchmark because the elitist
variant reduced diversity and performed worse at most TSP sizes.

The GeneSpace recipe ablation freezes the decoder while varying binary latent
codes, shallow decoder width, and the original population/selection settings:

```bash
python -m benchmarks.tsp_genespace_ablation \
  --sizes 8 12 16 24 32 \
  --budget 5000 \
  --seeds 0 1 2 3 4 \
  --output benchmark_results/mps_tsp_genespace_ablation_5000.json
```

`compact_guarded20` and `binary250_w2000_guarded20` can be selected explicitly
with `--variants` to test rare guarded decoder updates every 20 generations.

For the controlled float/binary and operator-schedule study:

```bash
python -m benchmarks.tsp_operator_ablation \
  --sizes 8 12 16 24 32 \
  --budget 5000 \
  --seeds 0 1 2 3 4 \
  --output benchmark_results/mps_tsp_operator_ablation_5000.json
```

This runner freezes the decoder and matches network width/depth while varying
float versus binary latents, latent length, mutation strength, four/eight-point
crossover, whole-population versus offspring-only mutation, and the clean
two-stage GeneSpace schedule. The latter evaluates and caps crossover children
before generating independent mutation children. Both replacement modes are
included, alongside a diagnostic that faithfully reproduces GeneSpace's old
in-place mutation and duplicate object references.
