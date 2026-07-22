"""Batched per-individual decoding via torch.func.vmap.

The explorer's bottleneck (~96% of wall clock) has always been decoding the
population ONE INDIVIDUAL AT A TIME: vector_to_parameters loads that
individual's weights into the template network, one forward pass runs, and
the next individual repeats it — 32 tiny sequential GPU dispatches per
generation, each paying launch overhead.

This decodes the whole population in ONE call: the stacked weight vectors
(B, n_params) are sliced into per-parameter tensors of shape (B, *param_shape)
and vmap maps functional_call over the batch dimension. Same math, one kernel
launch per op instead of per individual.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.func import functional_call, vmap

from benchmarks.legacy_engines.explorer import _Template


class BatchedTemplate(_Template):
    """A _Template whose decode_batch runs every individual in one vmap call."""

    def __init__(self, builder, device: str):
        super().__init__(builder, device)
        self._names = [n for n, _ in self.net.named_parameters()]
        self._shapes = [tuple(p.shape) for _, p in self.net.named_parameters()]
        self._numels = [int(np.prod(s)) if s else 1 for s in self._shapes]
        self._offsets = np.concatenate([[0], np.cumsum(self._numels)])

        def _forward(params: dict, z: torch.Tensor) -> torch.Tensor:
            return torch.sigmoid(
                functional_call(self.net, params, (z[None],)))[0]

        self._vforward = vmap(_forward)

    def decode_batch(self, thetas: np.ndarray, zs: np.ndarray) -> torch.Tensor:
        """thetas: (B, n_params) float32; zs: (B, latent) float32."""
        flat = torch.as_tensor(np.ascontiguousarray(thetas),
                               device=self.device)
        params = {
            name: flat[:, self._offsets[i]:self._offsets[i + 1]]
            .reshape(len(flat), *self._shapes[i])
            for i, name in enumerate(self._names)}
        genes = torch.as_tensor(np.ascontiguousarray(zs), device=self.device)
        return self._vforward(params, genes)


def self_test(builder, latent: int, output_shape, device: str = "mps",
              batch: int = 8, atol: float = 1e-5) -> float:
    """Max abs difference between sequential and batched decode."""
    rng = np.random.default_rng(0)
    t = BatchedTemplate(builder, device)
    thetas = np.stack([t.init_theta(i) for i in range(batch)])
    zs = rng.standard_normal((batch, latent)).astype(np.float32)
    sequential = torch.stack([t.decode(th, z) for z, th in zip(zs, thetas)])
    batched = t.decode_batch(thetas, zs)
    return float((sequential - batched).abs().max())
