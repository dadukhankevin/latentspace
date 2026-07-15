"""The exploration phase: per-individual decoder evolution.

Every individual carries its own genome AND its own private decoder
weights; children receive noisy copies of both. The population searches
over genotype-to-phenotype maps, not just genotypes, and no operator ever
touches the phenotype — mutation and crossover exist only for tensors, so
the same explorer runs for any output modality.

The 32 independent lineages produce solutions whose errors point in
different directions — the property the distillation phase depends on
(independent errors cancel under compression; shared biases persist).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn


@dataclass
class ExplorerConfig:
    population: int = 32
    elite: int = 16
    genome_mutation_rate: float = 0.1
    genome_mutation_sigma: float = 0.12
    # Weight-noise scales measured viable during the benchmark campaign
    # (43% of mutant outputs beat their parent at 0.003; ~0% at 0.03+).
    weight_sigma_low: float = 0.003
    weight_sigma_high: float = 0.02
    # Adaptive stop: exploration ends when the best loss has improved less
    # than stall_tol (relative) over the last stall_window generations.
    stall_window: int = 10
    stall_tol: float = 0.01


@dataclass
class Archive:
    """Every phenotype the explorer evaluated, with loss and lineage."""
    phenotypes: np.ndarray
    losses: np.ndarray
    lineages: np.ndarray

    def select(self, top: int, lineage_cap: int | None = None) -> np.ndarray:
        """Indices of the `top` best solutions; with `lineage_cap`, no
        lineage contributes more than that many (shortfall topped up with
        the global best) — preserving error independence for distillation."""
        order = np.argsort(self.losses)
        if lineage_cap is None:
            return order[:top]
        counts: dict[int, int] = {}
        chosen: list[int] = []
        for i in order:
            lineage = int(self.lineages[i])
            if counts.get(lineage, 0) >= lineage_cap:
                continue
            counts[lineage] = counts.get(lineage, 0) + 1
            chosen.append(int(i))
            if len(chosen) == top:
                return np.asarray(chosen)
        taken = set(chosen)
        for i in order:
            if int(i) not in taken:
                chosen.append(int(i))
                if len(chosen) == top:
                    break
        return np.asarray(chosen)


class _Template:
    """One reusable network; per-individual weights load in as flat vectors."""

    def __init__(self, builder, device: str):
        self.builder = builder
        self.net = builder().to(device)
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.device = device
        self.n_params = sum(p.numel() for p in self.net.parameters())

    def init_theta(self, seed: int) -> np.ndarray:
        torch.manual_seed(seed)
        fresh = self.builder()
        return nn.utils.parameters_to_vector(
            fresh.parameters()).detach().numpy().astype(np.float32)

    def decode(self, theta: np.ndarray, z: np.ndarray) -> torch.Tensor:
        nn.utils.vector_to_parameters(
            torch.as_tensor(theta, device=self.device),
            self.net.parameters())
        genes = torch.as_tensor(z[None].astype(np.float32), device=self.device)
        return torch.sigmoid(self.net(genes))[0]


class PerIndividualExplorer:
    def __init__(self, builder, latent: int, device: str,
                 config: ExplorerConfig | None = None):
        self.template = _Template(builder, device)
        self.latent = latent
        self.config = config or ExplorerConfig()

    def _mutate_genome(self, z: np.ndarray, rng) -> np.ndarray:
        c = self.config
        mask = rng.random(z.shape) < c.genome_mutation_rate
        if not mask.any():
            mask[rng.integers(0, len(z))] = True
        return (z + mask * rng.normal(0, c.genome_mutation_sigma, z.shape)
                ).astype(np.float32)

    def _mutate_weights(self, theta: np.ndarray, rng) -> np.ndarray:
        c = self.config
        sigma = float(np.exp(rng.uniform(
            np.log(c.weight_sigma_low), np.log(c.weight_sigma_high))))
        scale = max(float(theta.std()), 1e-3)
        return (theta + rng.normal(0, sigma * scale, theta.shape)
                ).astype(np.float32)

    def run(self, evaluate_losses, rng, stop_after,
            adaptive: bool = True, reserve: int = 0) -> Archive:
        """Evolve until the budget boundary `stop_after(evals_spent)` is hit
        or (if `adaptive`) the stall rule fires with `reserve` evals left.

        `evaluate_losses(phenotypes_tensor) -> losses` must count its own
        evaluations; `stop_after(spent)` reports remaining budget.
        """
        c = self.config
        archive_x: list[np.ndarray] = []
        archive_loss: list[float] = []
        archive_lineage: list[int] = []

        def evaluate(z_batch, theta_batch, lineage_batch) -> np.ndarray:
            phenotypes = torch.stack([
                self.template.decode(t, z)
                for z, t in zip(z_batch, theta_batch)])
            losses = np.asarray(evaluate_losses(phenotypes), dtype=np.float64)
            archive_x.extend(phenotypes.detach().cpu().numpy())
            archive_loss.extend(losses.tolist())
            archive_lineage.extend(int(l) for l in lineage_batch)
            return losses

        zs = rng.standard_normal((c.population, self.latent)).astype(np.float32)
        thetas = np.stack([
            self.template.init_theta(int(rng.integers(0, 2**31)))
            for _ in range(c.population)])
        lineages = np.arange(c.population)
        n = min(c.population, stop_after(0))
        loss = evaluate(zs[:n], thetas[:n], lineages[:n])
        zs, thetas, lineages = zs[:n], thetas[:n], lineages[:n]
        best_history = [float(loss.min())]

        while True:
            remaining = stop_after(len(archive_loss))
            if remaining <= 0:
                break
            if adaptive:
                if remaining <= reserve:
                    break
                if len(best_history) > c.stall_window:
                    then = best_history[-c.stall_window - 1]
                    if then - best_history[-1] < c.stall_tol * abs(then):
                        break
            order = np.argsort(loss)[:c.elite]
            zs, thetas, loss = zs[order], thetas[order], loss[order]
            lineages = lineages[order]
            n = min(c.population, remaining)
            parent = rng.integers(0, len(zs), n)
            child_z = np.stack([self._mutate_genome(zs[p], rng)
                                for p in parent])
            child_theta = np.stack([self._mutate_weights(thetas[p], rng)
                                    for p in parent])
            child_loss = evaluate(child_z, child_theta, lineages[parent])
            zs = np.concatenate([zs, child_z])
            thetas = np.concatenate([thetas, child_theta])
            lineages = np.concatenate([lineages, lineages[parent]])
            loss = np.concatenate([loss, child_loss])
            best_history.append(float(loss.min()))

        return Archive(np.asarray(archive_x), np.asarray(archive_loss),
                       np.asarray(archive_lineage))
