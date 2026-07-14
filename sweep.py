"""Ablation sweep: which GeneSpace lever actually moves the needle?

Problem: decode -> (16,), match linspace(0,1,16). Metric = true MSE of the
decoded best individual (lower is better). Each config runs from the same seed.
"""
import time
import numpy as np
import torch

from latentspace import Evolver, TrainMode

TARGET = torch.linspace(0, 1, 16)
GENS = 200


def match_target(phenotypes):
    err = ((phenotypes - TARGET.to(phenotypes.device)) ** 2).mean(dim=1)
    return (1.0 / (err + 1e-6)).tolist()


def run(name, **kw):
    np.random.seed(0)
    torch.manual_seed(0)
    t0 = time.time()
    ev = Evolver(match_target, output_shape=(16,), **kw)
    ev.solve(GENS, verbose_every=0)
    decoded = ev.decode_best().cpu().numpy()
    mse = float(np.mean((decoded - TARGET.numpy()) ** 2))
    print(f"{name:<34} MSE={mse:.5f}   ({time.time()-t0:4.1f}s)")
    return mse


# Baseline = the conservative spine defaults.
BASE = dict(latent=32, population=150, hidden_size=256, num_layers=2, lr=1e-4,
            pressure=1.8, scheme="linear", children=2, n_points=4,
            mode=TrainMode.SELF_DISTILL, refine_every=10)

print(f"target-match, {GENS} gens, MSE lower=better\n" + "-" * 52)
run("baseline (spine)", **BASE)

# One lever at a time toward GeneSpace's config.
run("+ wider decoder (2000)",        **{**BASE, "hidden_size": 2000})
run("+ shallow decoder (1 layer)",   **{**BASE, "num_layers": 1})
run("+ longer latent (250)",         **{**BASE, "latent": 250})
run("+ lower lr (1e-5)",             **{**BASE, "lr": 1e-5})
run("+ higher lr (1e-3)",            **{**BASE, "lr": 1e-3})
run("+ exp pressure (20)",           **{**BASE, "pressure": 20, "scheme": "exp"})
run("+ more children (4), pts (8)",  **{**BASE, "children": 4, "n_points": 8})
run("+ GOOD_TO_BEST mode",           **{**BASE, "mode": TrainMode.GOOD_TO_BEST})
run("+ refine every 5",              **{**BASE, "refine_every": 5})

print("-" * 52)
# Full GeneSpace-style config (float latent kept per our v1 decision).
run("ALL (GeneSpace-style, float)",
    latent=250, population=200, hidden_size=2000, num_layers=1, lr=1e-5,
    pressure=20, scheme="exp", children=4, n_points=8,
    mode=TrainMode.GOOD_TO_BEST, refine_every=10)
