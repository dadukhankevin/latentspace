"""The transfer test — the actual thesis on trial (2026-07-22).

A universal decoder's whole reason to exist is that training it on a family
of related problems makes a FRESH problem solve faster than a cold decoder.
The distillation loop just proved gradient distillation lifts co-resident
problems (+16pt); this asks the deeper question: does the KNOWLEDGE persist
in the decoder's weights and transfer to held-out problems?

Protocol:
  1. WARM the shared decoder by running the distillation loop on a TRAINING
     set of images (its weights absorb the family's structure).
  2. On a HELD-OUT set never seen in training, run the loop twice at matched
     budget: COLD (fresh random decoder) vs WARM (decoder = trained weights).
  3. Metric is ABSOLUTE best MSE on the held-out targets over the budget —
     not error-removed-vs-founder, because a warm decoder starts with better
     founders and that metric would hide its advantage. Transfer shows as
     warm reaching lower MSE FASTER (a head start), per the record's law.

Two relatedness levels, because the record found transfer is PIXEL-level:
  related   — held-out targets are jitter variants of the training family
  unrelated — held-out targets are different CIFAR images (Daniel's claim
              that even these share low-level statistics)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from benchmarks.experimental_distill_loop import Config, run


def load_cifar(n, offset=0):
    from PIL import Image
    files = sorted(Path("/tmp/latentspace_cifar100_scaling_1024")
                   .glob("*.png"))[offset:offset + n]
    return [np.asarray(Image.open(f), dtype=np.float32) / 255.0 for f in files]


def jitter(img, rng):
    """A pixel-related variant: small roll + brightness/colour shift."""
    out = np.roll(img, rng.integers(-3, 4), axis=0)
    out = np.roll(out, rng.integers(-3, 4), axis=1)
    out = np.clip(out * rng.uniform(0.85, 1.15)
                  + rng.uniform(-0.08, 0.08, 3), 0, 1)
    return out.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-epochs", type=int, default=1200)
    parser.add_argument("--test-epochs", type=int, default=700)
    parser.add_argument("--seed", type=int, default=3)
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    rng = np.random.default_rng(args.seed)
    shape = (32, 32, 3)

    # training family: 4 base images x several jitter variants
    bases = load_cifar(4, offset=0)
    train = [jitter(b, rng) for b in bases for _ in range(4)]     # 16
    held_related = [jitter(b, rng) for b in bases for _ in range(2)]  # 8, same family
    held_unrelated = load_cifar(8, offset=100)                    # 8, different images

    print(f"WARMING decoder on {len(train)} training variants "
          f"({args.train_epochs} epochs)...", flush=True)
    cfg_train = Config(epochs=args.train_epochs)
    _, _, gen, _, _ = run(train, shape, cfg_train, seed=args.seed, device=device,
                       return_full=True, log=max(1, args.train_epochs // 4))
    warm_state = {k: v.clone() for k, v in gen.net.state_dict().items()}

    cfg_test = Config(epochs=args.test_epochs)
    for label, held in (("RELATED (jitter variants of the trained family)",
                         held_related),
                        ("UNRELATED (different CIFAR images)",
                         held_unrelated)):
        print(f"\n=== held-out: {label} ===", flush=True)
        traces = {}
        for arm, init in (("cold", None), ("warm", warm_state)):
            _, _, _, trace, _ = run(held, shape, cfg_test, seed=args.seed + 1,
                                 device=device, init_state=init,
                                 return_full=True)
            traces[arm] = dict(trace)
        epochs = sorted(traces["cold"])
        print(f"  {'epoch':>6} {'cold MSE':>10} {'warm MSE':>10} "
              f"{'warm/cold':>10}")
        for e in epochs:
            c, w = traces["cold"][e], traces["warm"][e]
            print(f"  {e:>6} {c:>10.5f} {w:>10.5f} {w / c:>10.2f}", flush=True)


if __name__ == "__main__":
    main()
