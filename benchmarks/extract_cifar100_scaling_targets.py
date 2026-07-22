"""Extract a deterministic, class-balanced nested CIFAR-100 target sequence."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
from PIL import Image


def _read_pickle(path: Path) -> dict:
    with path.open("rb") as source:
        return pickle.load(source, encoding="bytes")


def nested_indices(labels: np.ndarray, count: int, seed: int) -> np.ndarray:
    """Take one example per shuffled class before taking a second, and so on."""
    classes = np.unique(labels)
    rng = np.random.default_rng(seed)
    class_order = rng.permutation(classes)
    per_class = {
        int(label): rng.permutation(np.flatnonzero(labels == label)).tolist()
        for label in classes
    }
    selected: list[int] = []
    depth = 0
    while len(selected) < count:
        added = False
        for label in class_order:
            candidates = per_class[int(label)]
            if depth < len(candidates):
                selected.append(int(candidates[depth]))
                added = True
                if len(selected) == count:
                    break
        if not added:
            raise ValueError(f"dataset has fewer than {count} examples")
        depth += 1
    return np.asarray(selected, dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path,
                        help="extracted cifar-100-python directory")
    parser.add_argument("output", type=Path)
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20_260_719)
    args = parser.parse_args()

    if args.count < 2:
        raise ValueError("count must be at least two")
    test = _read_pickle(args.dataset / "test")
    meta = _read_pickle(args.dataset / "meta")
    data = np.asarray(test[b"data"], dtype=np.uint8)
    labels = np.asarray(test[b"fine_labels"], dtype=np.int64)
    label_names = [value.decode("utf-8") for value in meta[b"fine_label_names"]]
    indices = nested_indices(labels, args.count, args.seed)

    args.output.mkdir(parents=True, exist_ok=True)
    for order, source_index in enumerate(indices):
        pixels = data[source_index].reshape(3, 32, 32).transpose(1, 2, 0)
        label = int(labels[source_index])
        name = label_names[label].replace(" ", "_")
        path = args.output / f"{order:04d}_{name}_{source_index:05d}.png"
        Image.fromarray(pixels, mode="RGB").save(path)
    print(f"wrote {len(indices)} nested targets to {args.output}")


if __name__ == "__main__":
    main()
