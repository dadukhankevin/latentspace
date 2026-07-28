"""SUPER-DATASET — evolve 10 images that teach a network MNIST (Daniel,
2026-07-27).

The idea: a dataset so distilled it looks like static, yet training on it
produces a working classifier. The phenotype is TEN synthetic 28x28 images
plus ten evolved SOFT labels (both halves evolved, per Daniel's spec). The
fitness function trains a small MLP from fixed initializations on those ten
examples and returns its accuracy on real MNIST digits it has never
configured — the train-then-eval loop is a BLACK BOX to the GA (nothing is
backpropagated through training), which is exactly this library's niche:
the standard dataset-distillation methods differentiate through the whole
training procedure and so cannot handle non-differentiable training at all.

Feedback is DENSE (validation accuracy moves smoothly as the images
improve), the manifold is continuous (pixels and label logits), and
determinism is enforced the same way Dogfight's was: fixed model
initializations, full-batch training (no data order), fixed evaluation
set. A noisy fitness silently breaks selection.

    DYLD_LIBRARY_PATH=/opt/homebrew/opt/expat/lib \
        python3 -m benchmarks.demo_super_dataset --live
"""
from __future__ import annotations

import argparse
import gzip
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from latentspace.universal import solve, register_architecture
from latentspace.universal.architectures import build_mlp

MNIST = Path.home() / "zor" / "data" / "MNIST" / "raw"
N_IMAGES = 10
SIDE = 28
N_CLASSES = 10
DIM = N_IMAGES * SIDE * SIDE + N_IMAGES * N_CLASSES     # images | labels
HIDDEN = 64
TRAIN_STEPS = 150
TRAIN_LR = 0.02
INIT_SEEDS = (0, 1)             # student inits averaged over
EVAL_N = 1024
STORE = Path(__file__).resolve().parent.parent / "demo" / "super_dataset.pt"


def load_mnist(split):
    stem = "train" if split == "train" else "t10k"
    with gzip.open(MNIST / f"{stem}-images-idx3-ubyte.gz") as f:
        x = np.frombuffer(f.read(), dtype=np.uint8, offset=16)
    with gzip.open(MNIST / f"{stem}-labels-idx1-ubyte.gz") as f:
        y = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
    return (x.reshape(-1, SIDE * SIDE).astype(np.float32) / 255.0,
            y.astype(np.int64))


def unpack(pheno):
    """(DIM,) in [0,1] -> images (10, 784) and soft labels (10, 10)."""
    images = pheno[:N_IMAGES * SIDE * SIDE].reshape(N_IMAGES, SIDE * SIDE)
    logits = pheno[N_IMAGES * SIDE * SIDE:].reshape(N_IMAGES, N_CLASSES)
    labels = torch.softmax(torch.as_tensor(logits) * 8.0, dim=1)  # sharpen
    return torch.as_tensor(images), labels


class Student(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(SIDE * SIDE, HIDDEN), nn.ReLU(),
                                 nn.Linear(HIDDEN, N_CLASSES))

    def forward(self, x):
        return self.net(x)


def make_fitness(eval_x, eval_y, device):
    ex = torch.as_tensor(eval_x, device=device)
    ey = torch.as_tensor(eval_y, device=device)

    def teach_and_test(pheno):
        """Train fresh students on the 10 evolved examples; mean accuracy
        on real digits. Deterministic: fixed inits, full batch."""
        images, labels = unpack(pheno)
        images, labels = images.to(device), labels.to(device)
        accs = []
        for seed in INIT_SEEDS:
            torch.manual_seed(seed)
            student = Student().to(device)
            opt = torch.optim.Adam(student.parameters(), lr=TRAIN_LR)
            for _ in range(TRAIN_STEPS):
                opt.zero_grad()
                out = torch.log_softmax(student(images), dim=1)
                loss = -(labels * out).sum(dim=1).mean()   # soft-label CE
                loss.backward()
                opt.step()
            with torch.no_grad():
                accs.append(float((student(ex).argmax(1) == ey)
                                  .float().mean()))
        return float(np.mean(accs))

    def fitness(phenotypes):
        flat = phenotypes.reshape(len(phenotypes), -1).cpu().numpy()
        return torch.tensor([teach_and_test(p) for p in flat],
                            dtype=torch.float32)
    return fitness, teach_and_test


def baselines(train_x, train_y, teach):
    """What must be beaten. Random static, and 10 REAL digits (one per
    class) pushed through the same train-then-test pipeline."""
    rng = np.random.default_rng(0)
    noise = rng.random(DIM).astype(np.float32)
    real = np.zeros(DIM, dtype=np.float32)
    for c in range(N_CLASSES):
        idx = np.flatnonzero(train_y == c)[0]
        real[c * SIDE * SIDE:(c + 1) * SIDE * SIDE] = train_x[idx]
        logits = np.full(N_CLASSES, 0.0); logits[c] = 1.0
        real[N_IMAGES * SIDE * SIDE + c * N_CLASSES:
             N_IMAGES * SIDE * SIDE + (c + 1) * N_CLASSES] = logits
    return teach(noise), teach(real), real


def register_decoder(gain=10.0):
    def build(latent, output_shape):
        net = build_mlp(latent, output_shape, hidden=256)
        with torch.no_grad():
            net[-1].weight.mul_(gain)     # a near-constant phenotype is ten
            net[-1].bias.mul_(gain)       # IDENTICAL images (FINDINGS 14)
        return net
    register_architecture("dataset", build)
    return "dataset"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()

    device = "cpu"          # students are tiny; MPS launch overhead loses
    train_x, train_y = load_mnist("train")
    rng = np.random.default_rng(0)
    eval_idx = rng.choice(len(train_x), EVAL_N, replace=False)
    fitness, teach = make_fitness(train_x[eval_idx], train_y[eval_idx],
                                  device)

    noise_acc, real_acc, real_vec = baselines(train_x, train_y, teach)
    print(f"baselines through the same pipeline: random static "
          f"{noise_acc:.1%}, ten real digits {real_acc:.1%}")

    view = None
    if args.live:
        import matplotlib
        matplotlib.use("MacOSX")
        import matplotlib.pyplot as plt
        plt.ion()
        fig = plt.figure(figsize=(11, 6.2))
        fig.canvas.manager.set_window_title("super-dataset")
        axes = [fig.add_subplot(3, 5, i + 1) for i in range(N_IMAGES)]
        curve_ax = fig.add_subplot(3, 1, 3)
        view = (plt, fig, axes, curve_ax, [], [])

    def progress(epoch, total, spent, phenos, scores):
        acc = scores[0]
        print(f"  epoch {epoch:>5}/{total}  {spent:>6} student-trainings  "
              f"best accuracy {acc:.1%}", flush=True)
        if view is None:
            return
        plt, fig, axes, curve_ax, xs, ys = view
        images, labels = unpack(np.asarray(phenos[0]).reshape(-1))
        xs.append(epoch); ys.append(acc * 100)
        for i, ax in enumerate(axes):
            ax.clear()
            ax.imshow(images[i].reshape(SIDE, SIDE), cmap="gray",
                      vmin=0, vmax=1)
            top = int(labels[i].argmax())
            ax.set_title(f"→ {top} ({labels[i][top]:.0%})", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
        curve_ax.clear()
        curve_ax.plot(xs, ys, color="#c2703a", lw=1.6)
        curve_ax.axhline(noise_acc * 100, color="#999", lw=1, ls=":",
                         label=f"random static {noise_acc:.0%}")
        curve_ax.axhline(real_acc * 100, color="#4d7fa3", lw=1, ls="--",
                         label=f"ten real digits {real_acc:.0%}")
        curve_ax.set_ylabel("accuracy on real MNIST (%)")
        curve_ax.set_xlabel("epoch")
        curve_ax.legend(loc="lower right", fontsize=8)
        fig.suptitle("ten evolved images + evolved labels — a network "
                     "trained ONLY on these is scored on real digits",
                     fontsize=11)
        plt.pause(0.001)

    began = time.time()
    result = solve(fitness, output_shape=(DIM,), epochs=args.epochs,
                   architecture=register_decoder(), seed=args.seed,
                   device="cpu", population_cap=96, children=24,
                   progress=progress, progress_every=10)
    print(f"\nevolved: {result.best_fitness:.1%} on the fixed eval set "
          f"({result.evaluations} student trainings, "
          f"{time.time() - began:.0f}s)")

    # the honest number: untouched test split
    test_x, test_y = load_mnist("test")
    tf, tt = make_fitness(test_x, test_y, device)
    final = tt(np.asarray(result.best_phenotype).reshape(-1))
    real_final = tt(real_vec)
    print(f"HELD-OUT TEST (10k digits never seen by anything): "
          f"evolved {final:.1%} vs ten real digits {real_final:.1%} "
          f"vs random static {noise_acc:.1%}")
    STORE.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"phenotype": np.asarray(result.best_phenotype),
                "eval_acc": result.best_fitness, "test_acc": final}, STORE)
    print(f"saved {STORE}")
    if view is not None:
        view[0].ioff(); view[0].show(block=True)


if __name__ == "__main__":
    main()
