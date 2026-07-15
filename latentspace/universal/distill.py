"""The distillation phase: compress vetted solutions into a search space.

Closed-form PCA in logit space — the campaign's repeated finding is that
this simple linear compression is remarkably hard to beat, and it is exact,
fast, and tuning-free. The compression's power comes from its inputs:
solutions whose errors are independent (different exploration lineages)
cancel, leaving the shared structure of what scored well.
"""
from __future__ import annotations

import numpy as np
import torch


class LatentSpace:
    """decode(z) = sigmoid(mean + z @ basis): a linear generative model."""

    def __init__(self, mean: np.ndarray, basis: np.ndarray,
                 output_shape: tuple, device: str = "cpu"):
        self.latent = basis.shape[0]
        self.output_shape = tuple(output_shape)
        self.device = device
        self.mean = torch.as_tensor(mean.astype(np.float32), device=device)
        self.basis_t = torch.as_tensor(basis.astype(np.float32), device=device)

    def decode(self, genes_batch) -> torch.Tensor:
        genes = torch.as_tensor(
            np.asarray(genes_batch, dtype=np.float32), device=self.device
        )
        with torch.no_grad():
            out = torch.sigmoid(self.mean + genes @ self.basis_t)
        return out.view(-1, *self.output_shape)


def distill(phenotypes: np.ndarray, latent: int, output_shape: tuple,
            device: str = "cpu") -> LatentSpace:
    """PCA of solutions in logit space; latent axes scaled to unit variance."""
    elites = np.clip(np.asarray(phenotypes, dtype=np.float64), 1e-3, 1 - 1e-3)
    elites = elites.reshape(len(elites), -1)
    logits = np.log(elites / (1 - elites))
    mean = logits.mean(axis=0)
    centered = logits - mean
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    k = min(latent, len(singular))
    scale = singular[:k] / np.sqrt(max(len(elites) - 1, 1))
    basis = (scale[:, None] * vt[:k]).astype(np.float32)
    if k < latent:
        basis = np.vstack([basis, np.zeros((latent - k, mean.size), np.float32)])
    return LatentSpace(mean, basis, output_shape, device=device)
