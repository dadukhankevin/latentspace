import numpy as np
import pytest

from benchmarks.tsp_scaling import held_karp_optimum


def test_held_karp_finds_square_perimeter():
    points = np.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    )
    delta = points[:, None, :] - points[None, :, :]
    distances = np.sqrt(np.sum(delta**2, axis=-1))

    assert held_karp_optimum(distances) == pytest.approx(4.0)


def test_held_karp_validates_input():
    with pytest.raises(ValueError, match="square"):
        held_karp_optimum(np.ones((2, 3)))
