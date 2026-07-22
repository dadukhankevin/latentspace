"""The exploration phase: per-individual decoder evolution.

Every individual carries its own genome AND its own private decoder
weights. Children are born by crossover — one cut recombines two parents'
genomes, the parents' decoders are averaged (rounds 42-45) — then both
tensors mutate. No operator ever touches the phenotype, so the same
explorer runs for any output modality.

(Until round 42 this docstring claimed crossover and the code had none:
every earlier result was a mutation-only (mu + lambda) evolution strategy.
Measured at 10 seeds, adding crossover is significant on smooth problems
— image t=5.6, curve t=5.3 — and its TSP benefit GROWS with problem size,
crossing significance at 400 cities. Averaging the parents' decoders is
safe because a run's survivors are lineage-collapsed near-clones, so the
mean cancels mutation noise instead of breaking function; cross-RUN
averaging would be catastrophic — round 37 measured 18-28x worse than
flat gray.)

The population is decoded in ONE batched vmap call per generation —
per-individual sequential decoding was ~7x slower and was ~96% of wall
clock.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.func import functional_call, vmap


@dataclass
class ExplorerConfig:
    population: int = 32
    # Survivors kept per generation (mu). With crossover on, 8 was best or
    # near-best on image, curve and TSP at once (rounds 42-43); without it
    # the best value was problem-dependent (round 38: smooth wants 1, TSP
    # wants 16) and four attempts to steer it closed-loop all failed
    # (rounds 39-41) — the evidence for keeping diversity only becomes
    # legible after the cull is irreversible.
    elite: int = 8
    # Crossover (round 42; the explorer was mutation-only before then).
    # One cut recombines the parents' genomes — the child keeps the fitter
    # parent's genome as its base and grafts in the other's segment. The
    # decoders are averaged: significant wins on image (t=5.6) and
    # TSP-400 (t=4.0) vs mutation-only, never a significant loss, and the
    # best arm on 4 of 5 problems (round 45). "average", "fitter" (the
    # fitter parent donates its decoder whole), or "off".
    crossover: str = "average"
    # Fraction of children that cross over; the rest are pure calibrated
    # mutations. 1.0 = the round-42 always-cross behavior. In the multi-family
    # solve_many, forcing every child to cross with a random stranger was
    # catastrophic (3% vs 81% error removed); here survivors solve one shared
    # problem so crossover is among co-adapted elites, but rarity may still
    # protect the win-rate step controller — under test on TSP.
    crossover_rate: float = 1.0
    # Fitness-signed mutation memory (round 50, Daniel's design). Every
    # child's weight mutation, signed and scaled by its fitness change vs
    # its parent — FAILURES INCLUDED; a step that hurt is a measured bad
    # direction — feeds one shared Adam-style accumulator (the population
    # as a distributed gradient sensor; legal within a run because the
    # lineage-collapsed decoders are near-clones). New mutations add a
    # drift term along the accumulated direction. Confirmed 1.38x on the
    # image at 10 seeds (t=3.9) and 1.54x on the apple photo; not
    # significant either way on curve or TSP-100. "shared" or "off".
    mutation_memory: str = "shared"
    memory_beta1: float = 0.9
    memory_beta2: float = 0.999
    memory_drift: float = 0.5   # drift as a fraction of the mutation step
    genome_mutation_rate: float = 0.1
    genome_mutation_sigma: float = 0.12
    # Nominal weight-noise scales; the step controller below rescales both
    # channels, so these only set the starting point and their ratio.
    weight_sigma_low: float = 0.003
    weight_sigma_high: float = 0.02
    # Step-size control (rounds 29-31): a single gain multiplies both
    # channels' sigmas, steered by the fraction of children that beat their
    # parent — Rechenberg's 1/5th rule. Right-sized steps win ~1 in 5;
    # more winners means the steps are too timid, fewer means too wild.
    # Fixed sigmas were 10-95x worse at short budgets and lost the apple
    # record at 150k; win-rate control spans a 150x step range in one run
    # with no problem-scale constants. "success" (default) or "off".
    step_control: str = "success"
    win_target: float = 0.2
    gain_step: float = 1.15
    gain_limits: tuple = (1e-2, 1e4)
    # Where the win-rate controller starts. 1.0 is conservative: round 31
    # measured the early-run gain climbing to ~19x while the canvas is
    # blank and any big move wins, and the controller spends dozens of
    # generations getting there at 1.15x per step. Setting this higher
    # skips the warm-up on problems known to start far from the answer.
    # Measured (2026-07-21): with THIS explorer's 32-child generations the
    # ramp costs so few evaluations that 1.0 is already right — 512 was
    # 20x WORSE on the blob benchmark at 5k. Hot starts only pay in
    # harnesses with large generations (192-child multi-fitness runs saw
    # 10x+ gains from 512): warmup cost in evaluations scales with
    # generation size, so tune this to the harness, not the problem.
    initial_gain: float = 1.0
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
        self._names = [n for n, _ in self.net.named_parameters()]
        self._shapes = [tuple(p.shape) for _, p in self.net.named_parameters()]
        numels = [int(np.prod(s)) if s else 1 for s in self._shapes]
        self._offsets = np.concatenate([[0], np.cumsum(numels)])

        def _forward(params: dict, z: torch.Tensor) -> torch.Tensor:
            return torch.sigmoid(functional_call(self.net, params, (z[None],)))[0]

        self._vforward = vmap(_forward)

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

    def decode_batch(self, thetas: np.ndarray, zs: np.ndarray) -> torch.Tensor:
        """Decode the whole population in one vmap call: (B, n_params) weight
        vectors are sliced into per-parameter tensors and functional_call is
        mapped over the batch. Bit-identical to sequential `decode` and ~7x
        faster — one kernel launch per op instead of per individual."""
        flat = torch.as_tensor(np.ascontiguousarray(thetas), device=self.device)
        params = {
            name: flat[:, self._offsets[i]:self._offsets[i + 1]]
            .reshape(len(flat), *self._shapes[i])
            for i, name in enumerate(self._names)}
        genes = torch.as_tensor(
            np.ascontiguousarray(zs.astype(np.float32)), device=self.device)
        return self._vforward(params, genes)


class PerIndividualExplorer:
    def __init__(self, builder, latent: int, device: str,
                 config: ExplorerConfig | None = None):
        self.template = _Template(builder, device)
        self.latent = latent
        self.config = config or ExplorerConfig()

    def _mutate_genome(self, z: np.ndarray, rng, gain: float = 1.0) -> np.ndarray:
        c = self.config
        mask = rng.random(z.shape) < c.genome_mutation_rate
        if not mask.any():
            mask[rng.integers(0, len(z))] = True
        return (z + mask * rng.normal(0, c.genome_mutation_sigma * gain,
                                      z.shape)).astype(np.float32)

    def _mutate_weights(self, theta: np.ndarray, rng,
                        gain: float = 1.0) -> np.ndarray:
        c = self.config
        sigma = float(np.exp(rng.uniform(
            np.log(c.weight_sigma_low), np.log(c.weight_sigma_high)))) * gain
        scale = max(float(theta.std()), 1e-3)
        return (theta + rng.normal(0, sigma * scale, theta.shape)
                ).astype(np.float32)

    def _cross_genomes(self, base: np.ndarray, donor: np.ndarray,
                       rng) -> np.ndarray:
        """One-point crossover: `base` is the fitter parent's genome (the one
        co-adapted to the decoder the child inherits); `donor` grafts in one
        contiguous segment."""
        cut = int(rng.integers(1, len(base)))
        child = base.copy()
        child[cut:] = donor[cut:]
        return child.astype(np.float32)

    def run(self, evaluate_losses, rng, stop_after,
            adaptive: bool = True, reserve: int = 0,
            warm_theta: np.ndarray | None = None,
            warm_fraction: float = 0.5,
            warm_sigma: float = 0.01) -> Archive:
        """Evolve until the budget boundary `stop_after(evals_spent)` is hit
        or (if `adaptive`) the stall rule fires with `reserve` evals left.

        `evaluate_losses(phenotypes_tensor) -> losses` must count its own
        evaluations; `stop_after(spent)` reports remaining budget.

        With `warm_theta` set (re-entering exploration after an exploit
        phase), `warm_fraction` of the population starts from those weights
        plus noise — the off-manifold escape channel around the current
        knowledge — and the rest start fresh, keeping the archive's errors
        diverse.
        """
        c = self.config
        gain = float(c.initial_gain)
        memory_m = np.zeros(self.template.n_params, np.float32)
        memory_v = np.zeros(self.template.n_params, np.float32)
        memory_steps, df_scale = 0, None
        archive_x: list[np.ndarray] = []
        archive_loss: list[float] = []
        archive_lineage: list[int] = []

        def evaluate(z_batch, theta_batch, lineage_batch) -> np.ndarray:
            phenotypes = self.template.decode_batch(
                np.asarray(theta_batch), np.asarray(z_batch))
            losses = np.asarray(evaluate_losses(phenotypes), dtype=np.float64)
            archive_x.extend(phenotypes.detach().cpu().numpy())
            archive_loss.extend(losses.tolist())
            archive_lineage.extend(int(l) for l in lineage_batch)
            return losses

        zs = rng.standard_normal((c.population, self.latent)).astype(np.float32)
        thetas = np.stack([
            self.template.init_theta(int(rng.integers(0, 2**31)))
            for _ in range(c.population)])
        if warm_theta is not None:
            scale = max(float(warm_theta.std()), 1e-3)
            n_warm = int(c.population * warm_fraction)
            for i in range(n_warm):
                thetas[i] = warm_theta + rng.normal(
                    0, warm_sigma * scale, warm_theta.shape
                ).astype(np.float32)
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
            if c.crossover != "off" and len(zs) > 1:
                # Survivors are sorted by loss, so the lower index of the two
                # draws IS the fitter parent. It supplies the base genome and
                # the child's lineage; mate selection stays uniform — rank
                # pressure inside the survivor set helped smooth problems but
                # hurt TSP (round 43), and uniform was never significantly
                # beaten anywhere.
                mate = rng.integers(0, len(zs), n)
                parent, mate = np.minimum(parent, mate), np.maximum(parent, mate)
                cross = (np.ones(n, dtype=bool) if c.crossover_rate >= 1.0
                         else rng.random(n) < c.crossover_rate)
                child_z = np.stack([
                    self._mutate_genome(
                        self._cross_genomes(zs[w], zs[l], rng) if x else zs[w],
                        rng, gain)
                    for w, l, x in zip(parent, mate, cross)])
                if c.crossover == "average":
                    avg = (thetas[parent] + thetas[mate]) / 2.0
                    base_theta = np.where(cross[:, None], avg, thetas[parent])
                else:
                    base_theta = thetas[parent]
            else:
                child_z = np.stack([self._mutate_genome(zs[p], rng, gain)
                                    for p in parent])
                base_theta = thetas[parent]
            # weight channel: one log-uniform sigma per child, noise scaled by
            # each base's own spread, plus the memory's drift term when on
            sigmas = np.exp(rng.uniform(np.log(c.weight_sigma_low),
                                        np.log(c.weight_sigma_high), n)) * gain
            scales = np.maximum(base_theta.std(axis=1), 1e-3)
            step_size = (sigmas * scales)[:, None].astype(np.float32)
            noise = rng.standard_normal(
                (n, self.template.n_params)).astype(np.float32)
            if c.mutation_memory == "shared" and memory_steps > 0:
                direction = ((memory_m / (1 - c.memory_beta1 ** memory_steps))
                             / (np.sqrt(memory_v / (1 - c.memory_beta2 ** memory_steps))
                                + 1e-8))
                child_theta = (base_theta + c.memory_drift * step_size * direction
                               + step_size * noise).astype(np.float32)
            else:
                child_theta = (base_theta + step_size * noise).astype(np.float32)
            child_loss = evaluate(child_z, child_theta, lineages[parent])
            if c.mutation_memory == "shared":
                # Every child is a gradient sample — failures included: its
                # birth noise, signed and scaled by the fitness change.
                df = loss[parent] - child_loss
                mag = float(np.abs(df).mean())
                df_scale = (mag if df_scale is None
                            else 0.9 * df_scale + 0.1 * mag)
                g = ((df / max(df_scale, 1e-12))[:, None].astype(np.float32)
                     * noise).mean(axis=0)
                memory_m = c.memory_beta1 * memory_m + (1 - c.memory_beta1) * g
                memory_v = c.memory_beta2 * memory_v + (1 - c.memory_beta2) * g ** 2
                memory_steps += 1
            if c.step_control == "success":
                # A tie means the mutation changed NOTHING — on discrete
                # phenotypes (a tour is an argsort of priorities; small
                # priority edits reorder no cities) that is evidence the step
                # is too SMALL. Counting ties as failures inverts the signal
                # and death-spirals: smaller steps make more ties, more ties
                # make fewer wins. Measured on 100-city TSP, strict wins drive
                # the gain to its floor by generation 45 with 75% ties and
                # learning frozen; counting ties as successes holds the win
                # rate at 16-25% and keeps climbing. Continuous phenotypes
                # never tie, so this is a no-op for them.
                wins = float((child_loss <= loss[parent] + 1e-12).mean())
                gain *= c.gain_step if wins > c.win_target else 1 / c.gain_step
                gain = float(np.clip(gain, *c.gain_limits))
            zs = np.concatenate([zs, child_z])
            thetas = np.concatenate([thetas, child_theta])
            lineages = np.concatenate([lineages, lineages[parent]])
            loss = np.concatenate([loss, child_loss])
            best_history.append(float(loss.min()))

        return Archive(np.asarray(archive_x), np.asarray(archive_loss),
                       np.asarray(archive_lineage))
