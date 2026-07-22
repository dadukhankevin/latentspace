"""Test one shared conditional decoder after multi-objective development.

The first phase is the private-decoder species-vector algorithm from
``demo_image_species_vector``.  At ``--transition-at`` its current survivors
become functional teachers for one of two shared conditional decoders:

``lora``
    Every linear/convolutional layer is

        base(x) + up(coefficients * down(x))

    so the shared down/up matrices are learned low-rank directions and each
    individual evolves only one aligned coefficient vector.  This is a
    crossover-friendly conditional LoRA dictionary, not round 36's fixed
    random flat projection.

``latent``
    The same number of extra individual values are appended to the ordinary
    decoder input.  This controls for merely adding more evolvable state.

``mixed``
    Half of the conditional values are appended to the decoder input and half
    gate learned LoRA directions.  This tests whether the complementary target
    strengths of the two 64-value arms combine at a fixed individual-state
    budget.

The transition is trained only on phenotypes already discovered by evolution;
real target images remain fitness functions, never distillation labels.
Periodic consolidation freezes the current adapted phenotypes as teachers,
shrinks individual coefficients, and trains both adapted and coefficient-zero
outputs toward those teachers.  The coefficient-zero score is recorded as the
amount of knowledge that has reached the shared backbone.

With ``--start-shared --dynamic-assimilation``, the mixed decoder is the only
decoder from the founders onward.  After each complete generation, the current
population is snapshotted, personal conditional values are shrunk by an amount
derived from their coefficient-zero phenotype gap, and the shared decoder takes
small replay steps to preserve those phenotypes.  There is no fixed transition
or consolidation schedule in this mode.

``--assimilation-method retirement_fold`` instead treats persistent parents
that leave the survivor population as retirements. Retirees are weighted inside
their assigned target niche by relative quality, lost coverage, and lifetime
offspring success; represented niches receive equal weight. Their one batched
LoRA legacy is folded exactly into the backbone, and one reproducible retired
champion per target is retained in a compact legacy bank. Rejected newborns
never contribute.

Crossover eligibility is measured either in ``z`` alone or in the complete
direct decoder input (``z`` plus any extra latent). LoRA gates are inherited
and mutated, but are not part of either distance. The shared decoder is also
never part of that distance. Fitness vectors still assign target niches and
control mate preference, relative success, and target-covered survival,
keeping ecological role separate from reproductive compatibility.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from benchmarks.compare import _require_mps, _seed_everything
from benchmarks.demo_clip_species_vector import (
    choose_compatible_mates,
    compatibility_graph,
    connected_components,
    graph_diagnostics,
    normalize_species_vectors,
)
from benchmarks.demo_image_species_vector import (
    ReferenceSpeciesView,
    load_targets,
    pairwise_negative_mse,
    select_target_covered_survivors,
    update_individual_step_state,
)
from benchmarks.round28_anchor_conv import ConvRGB
from latentspace.universal.architectures import resolve
from latentspace.universal.explorer import ExplorerConfig, _Template


LATENT = 64
SHAPE = (3, 96, 96)


class ConditionalLoRAConvRGB(nn.Module):
    """One ConvRGB backbone with shared directions gated by individual state.

    The same coefficient vector gates aligned directions at every layer.  The
    down/up factors are ordinary shared decoder parameters and are trained at
    factorization/consolidation events; evolution touches only coefficients.
    """

    def __init__(self, coefficient_dim: int, base: int = 6,
                 channels: int = 16, extra_latent_dim: int = 0):
        super().__init__()
        if coefficient_dim < 1:
            raise ValueError("coefficient_dim must be positive")
        if extra_latent_dim < 0:
            raise ValueError("extra_latent_dim must be non-negative")
        self.lora_dim = coefficient_dim
        self.extra_latent_dim = extra_latent_dim
        self.coefficient_dim = coefficient_dim + extra_latent_dim
        self.base = base
        self.channels = channels
        decoder_input = LATENT + extra_latent_dim
        self.base_fc = nn.Linear(decoder_input, channels * base * base)
        self.fc_down = nn.Linear(decoder_input, coefficient_dim, bias=False)
        self.fc_up = nn.Linear(
            coefficient_dim, channels * base * base, bias=False)

        doublings = int(np.log2(SHAPE[1] // base))
        self.base_convs = nn.ModuleList()
        self.conv_down = nn.ModuleList()
        self.conv_up = nn.ModuleList()
        in_channels = channels
        for _ in range(doublings):
            self.base_convs.append(nn.Conv2d(
                in_channels, channels, 3, padding=1))
            self.conv_down.append(nn.Conv2d(
                in_channels, coefficient_dim, 3, padding=1, bias=False))
            self.conv_up.append(nn.Conv2d(
                coefficient_dim, channels, 1, bias=False))
            in_channels = channels
        self.output_base = nn.Conv2d(channels, SHAPE[0], 3, padding=1)
        self.output_down = nn.Conv2d(
            channels, coefficient_dim, 3, padding=1, bias=False)
        self.output_up = nn.Conv2d(
            coefficient_dim, SHAPE[0], 1, bias=False)
        self._reset_adapter_parameters()

    def _reset_adapter_parameters(self) -> None:
        # Both factors are nonzero so zero coefficients preserve the backbone
        # exactly while receiving a first-order gradient during factorization.
        modules = [self.fc_down, self.fc_up, *self.conv_down, *self.conv_up,
                   self.output_down, self.output_up]
        for module in modules:
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @property
    def adapter_scale(self) -> float:
        return float(self.lora_dim ** -0.5)

    def initialize_backbone(self, theta: np.ndarray) -> None:
        """Copy a normal 64-gene ConvRGB into the coefficient-zero path."""
        source = ConvRGB(LATENT, SHAPE)
        nn.utils.vector_to_parameters(
            torch.as_tensor(theta, dtype=torch.float32), source.parameters())
        source_convs = [
            module for module in source.convs if isinstance(module, nn.Conv2d)
        ]
        if len(source_convs) != len(self.base_convs) + 1:
            raise RuntimeError("ConvRGB layout changed")
        with torch.no_grad():
            self.base_fc.weight[:, :LATENT].copy_(source.fc.weight)
            if self.extra_latent_dim:
                nn.init.normal_(
                    self.base_fc.weight[:, LATENT:], mean=0.0, std=0.02)
            self.base_fc.bias.copy_(source.fc.bias)
            for target, original in zip(self.base_convs, source_convs[:-1]):
                target.weight.copy_(original.weight)
                target.bias.copy_(original.bias)
            self.output_base.weight.copy_(source_convs[-1].weight)
            self.output_base.bias.copy_(source_convs[-1].bias)

    def _linear(self, decoder_input: torch.Tensor,
                lora_coefficients: torch.Tensor) -> torch.Tensor:
        residual = self.fc_up(
            self.fc_down(decoder_input) * lora_coefficients)
        return self.base_fc(decoder_input) + self.adapter_scale * residual

    def _conv(self, x: torch.Tensor, coefficients: torch.Tensor,
              index: int) -> torch.Tensor:
        gates = coefficients[:, :, None, None]
        residual = self.conv_up[index](self.conv_down[index](x) * gates)
        return self.base_convs[index](x) + self.adapter_scale * residual

    def forward(self, z: torch.Tensor,
                coefficients: torch.Tensor) -> torch.Tensor:
        if self.extra_latent_dim:
            extra = coefficients[:, :self.extra_latent_dim]
            lora_coefficients = coefficients[:, self.extra_latent_dim:]
            decoder_input = torch.cat([z, extra], dim=1)
        else:
            decoder_input = z
            lora_coefficients = coefficients
        if lora_coefficients.shape[1] != self.lora_dim:
            raise ValueError("conditional coefficient width does not match")
        x = self._linear(decoder_input, lora_coefficients).view(
            -1, self.channels, self.base, self.base)
        for index in range(len(self.base_convs)):
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            x = F.leaky_relu(self._conv(
                x, lora_coefficients, index))
        gates = lora_coefficients[:, :, None, None]
        residual = self.output_up(self.output_down(x) * gates)
        x = self.output_base(x) + self.adapter_scale * residual
        return x.flatten(1)


class ExtraLatentConvRGB(nn.Module):
    """A normal ConvRGB receiving additional individual decoder inputs."""

    def __init__(self, extra_dim: int):
        super().__init__()
        if extra_dim < 1:
            raise ValueError("extra_dim must be positive")
        self.extra_dim = extra_dim
        self.net = ConvRGB(LATENT + extra_dim, SHAPE)

    def initialize_backbone(self, theta: np.ndarray) -> None:
        source = ConvRGB(LATENT, SHAPE)
        nn.utils.vector_to_parameters(
            torch.as_tensor(theta, dtype=torch.float32), source.parameters())
        source_convs = [
            module for module in source.convs if isinstance(module, nn.Conv2d)
        ]
        target_convs = [
            module for module in self.net.convs if isinstance(module, nn.Conv2d)
        ]
        with torch.no_grad():
            self.net.fc.weight[:, :LATENT].copy_(source.fc.weight)
            # Zero extra inputs must preserve the source function, but the
            # corresponding columns must be nonzero so both the inputs and the
            # columns receive a first-order factorization signal.
            nn.init.normal_(
                self.net.fc.weight[:, LATENT:], mean=0.0, std=0.02)
            self.net.fc.bias.copy_(source.fc.bias)
            for target, original in zip(target_convs, source_convs):
                target.weight.copy_(original.weight)
                target.bias.copy_(original.bias)

    def forward(self, z: torch.Tensor,
                extra: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, extra], dim=1))


def initialize_conditional_decoder(
        mode: str,
        coefficient_dim: int,
        theta: np.ndarray,
        device: str,
        ) -> nn.Module:
    if mode == "lora":
        model: nn.Module = ConditionalLoRAConvRGB(coefficient_dim)
    elif mode == "latent":
        model = ExtraLatentConvRGB(coefficient_dim)
    elif mode == "mixed":
        if coefficient_dim < 2 or coefficient_dim % 2:
            raise ValueError("mixed coefficient_dim must be a positive even number")
        half = coefficient_dim // 2
        model = ConditionalLoRAConvRGB(
            half, extra_latent_dim=half)
    else:
        raise ValueError(f"unknown conditional mode: {mode}")
    model.initialize_backbone(theta)
    return model.to(device)


def decode_conditional(model: nn.Module, z: np.ndarray,
                       coefficients: np.ndarray, device: str) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        z_t = torch.as_tensor(
            np.ascontiguousarray(z.astype(np.float32)), device=device)
        c_t = torch.as_tensor(
            np.ascontiguousarray(coefficients.astype(np.float32)),
            device=device)
        return torch.sigmoid(model(z_t, c_t)).reshape(len(z), *SHAPE)


def fit_conditional_decoder(
        model: nn.Module,
        teacher_z: np.ndarray,
        teacher_images: np.ndarray,
        coefficient_dim: int,
        steps: int,
        learning_rate: float,
        code_learning_rate: float,
        base_only_weight: float,
        device: str,
        ) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Functionally factor current private teachers into one shared decoder."""
    if steps < 1:
        raise ValueError("factorization steps must be positive")
    if learning_rate <= 0 or code_learning_rate <= 0:
        raise ValueError("learning rates must be positive")
    if base_only_weight < 0:
        raise ValueError("base_only_weight must be non-negative")
    z = nn.Parameter(torch.as_tensor(teacher_z, device=device).clone())
    coefficients = nn.Parameter(torch.zeros(
        (len(teacher_z), coefficient_dim), device=device))
    targets = torch.as_tensor(teacher_images, device=device)
    optimizer = torch.optim.Adam([
        {"params": model.parameters(), "lr": learning_rate},
        {"params": [z, coefficients], "lr": code_learning_rate},
    ])
    trace: list[dict] = []
    report_every = max(1, steps // 10)
    model.train(True)
    for step in range(1, steps + 1):
        adapted = torch.sigmoid(model(z, coefficients)).reshape(
            len(z), *SHAPE)
        zeros = torch.zeros_like(coefficients)
        base = torch.sigmoid(model(z, zeros)).reshape(len(z), *SHAPE)
        adapted_mse = (adapted - targets).square().mean()
        base_mse = (base - targets).square().mean()
        loss = (
            adapted_mse
            + base_only_weight * base_mse
            + 1e-6 * z.square().mean()
            + 1e-6 * coefficients.square().mean()
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == 1 or step % report_every == 0 or step == steps:
            row = {
                "step": step,
                "adapted_phenotype_mse": float(adapted_mse.detach().cpu()),
                "base_phenotype_mse": float(base_mse.detach().cpu()),
                "coefficient_rms": float(
                    coefficients.detach().square().mean().sqrt().cpu()),
            }
            trace.append(row)
            print(
                f"    factor {step:>5}/{steps}  "
                f"adapted {row['adapted_phenotype_mse']:.7f}  "
                f"base {row['base_phenotype_mse']:.7f}  "
                f"coeff {row['coefficient_rms']:.3f}",
                flush=True,
            )
    return (
        z.detach().cpu().numpy().astype(np.float32),
        coefficients.detach().cpu().numpy().astype(np.float32),
        trace,
    )


def consolidate_conditional_decoder(
        model: nn.Module,
        z: np.ndarray,
        coefficients: np.ndarray,
        teacher_images: np.ndarray,
        shrink: float,
        steps: int,
        learning_rate: float,
        base_only_weight: float,
        device: str,
        ) -> tuple[np.ndarray, dict]:
    """Absorb current adapted functions while reducing private coefficients."""
    if not 0 <= shrink <= 1:
        raise ValueError("shrink must be in [0, 1]")
    if steps < 1:
        raise ValueError("consolidation steps must be positive")
    z_t = torch.as_tensor(z, device=device)
    coefficients_t = torch.as_tensor(
        coefficients * shrink, device=device)
    targets = torch.as_tensor(teacher_images, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train(True)
    adapted_mse = base_mse = None
    for _ in range(steps):
        adapted = torch.sigmoid(model(z_t, coefficients_t)).reshape(
            len(z), *SHAPE)
        base = torch.sigmoid(
            model(z_t, torch.zeros_like(coefficients_t))).reshape(
                len(z), *SHAPE)
        adapted_mse = (adapted - targets).square().mean()
        base_mse = (base - targets).square().mean()
        loss = adapted_mse + base_only_weight * base_mse
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert adapted_mse is not None and base_mse is not None
    return coefficients_t.cpu().numpy().astype(np.float32), {
        "adapted_phenotype_mse": float(adapted_mse.detach().cpu()),
        "base_phenotype_mse": float(base_mse.detach().cpu()),
        "coefficient_rms": float(
            coefficients_t.square().mean().sqrt().cpu()),
    }


def assimilation_fraction(phenotype_debt: float, maximum: float,
                          debt_scale: float) -> float:
    """Smoothly increase private-state shrinkage as representation debt grows."""
    if phenotype_debt < 0:
        raise ValueError("phenotype_debt must be non-negative")
    if not 0 <= maximum < 1:
        raise ValueError("maximum assimilation fraction must be in [0, 1)")
    if debt_scale <= 0:
        raise ValueError("assimilation debt scale must be positive")
    return float(maximum * phenotype_debt / (phenotype_debt + debt_scale))


def assimilate_conditional_decoder(
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        z: np.ndarray,
        coefficients: np.ndarray,
        teacher_images: np.ndarray,
        maximum_fraction: float,
        debt_scale: float,
        steps: int,
        base_only_weight: float,
        device: str,
        ) -> tuple[np.ndarray, dict]:
    """Take one gentle population-wide replay step into the shared decoder."""
    if steps < 1:
        raise ValueError("assimilation steps must be positive")
    if base_only_weight < 0:
        raise ValueError("assimilation base-only weight must be non-negative")
    z_t = torch.as_tensor(z, device=device)
    coefficients_t = torch.as_tensor(coefficients, device=device)
    targets = torch.as_tensor(teacher_images, device=device)
    with torch.no_grad():
        base_before = torch.sigmoid(
            model(z_t, torch.zeros_like(coefficients_t))).reshape(
                len(z), *SHAPE)
        phenotype_debt = float(
            (base_before - targets).square().mean().detach().cpu())
    fraction = assimilation_fraction(
        phenotype_debt, maximum_fraction, debt_scale)
    shrunken = (coefficients_t * (1.0 - fraction)).detach()
    model.train(True)
    for _ in range(steps):
        adapted = torch.sigmoid(model(z_t, shrunken)).reshape(
            len(z), *SHAPE)
        base = torch.sigmoid(
            model(z_t, torch.zeros_like(shrunken))).reshape(
                len(z), *SHAPE)
        loss = (
            (adapted - targets).square().mean()
            + base_only_weight * (base - targets).square().mean()
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        adapted = torch.sigmoid(model(z_t, shrunken)).reshape(
            len(z), *SHAPE)
        base = torch.sigmoid(
            model(z_t, torch.zeros_like(shrunken))).reshape(
                len(z), *SHAPE)
        adapted_mse = float(
            (adapted - targets).square().mean().detach().cpu())
        base_mse = float((base - targets).square().mean().detach().cpu())
    return shrunken.cpu().numpy().astype(np.float32), {
        "phenotype_debt_before": phenotype_debt,
        "assimilation_fraction": fraction,
        "adapted_phenotype_mse": adapted_mse,
        "base_phenotype_mse": base_mse,
        "coefficient_rms": float(
            shrunken.square().mean().sqrt().cpu()),
    }


def fold_lora_delta(
        model: ConditionalLoRAConvRGB,
        coefficients: np.ndarray,
        delta: np.ndarray,
        ) -> tuple[np.ndarray, dict]:
    """Exactly move one aligned LoRA-gate offset into the backbone."""
    if coefficients.ndim != 2:
        raise ValueError("coefficients must be a matrix")
    if coefficients.shape[1] != model.coefficient_dim:
        raise ValueError("conditional coefficient width does not match model")
    delta_np = np.asarray(delta, dtype=np.float32)
    if delta_np.shape != (model.lora_dim,):
        raise ValueError("LoRA fold delta width does not match model")
    delta_t = torch.as_tensor(
        delta_np, device=model.base_fc.weight.device,
        dtype=model.base_fc.weight.dtype)
    scale = model.adapter_scale
    with torch.no_grad():
        model.base_fc.weight.add_(
            scale * (model.fc_up.weight * delta_t[None, :])
            @ model.fc_down.weight)
        for base, down, up in zip(
                model.base_convs, model.conv_down, model.conv_up):
            folded = torch.einsum(
                "or,r,rihw->oihw",
                up.weight[:, :, 0, 0], delta_t, down.weight)
            base.weight.add_(scale * folded)
        folded_output = torch.einsum(
            "or,r,rihw->oihw",
            model.output_up.weight[:, :, 0, 0],
            delta_t,
            model.output_down.weight,
        )
        model.output_base.weight.add_(scale * folded_output)
    assimilated = coefficients.copy()
    assimilated[:, model.extra_latent_dim:] -= delta_np[None, :]
    return assimilated.astype(np.float32), {
        "folded_lora_mean_rms": float(np.sqrt(np.mean(delta_np ** 2))),
        "coefficient_rms": float(np.sqrt(np.mean(assimilated ** 2))),
    }


def fold_population_mean_lora(
        model: ConditionalLoRAConvRGB,
        coefficients: np.ndarray,
        fraction: float = 1.0,
        ) -> tuple[np.ndarray, dict]:
    """Exactly move the population-common LoRA gate into the backbone.

    For each layer, ``base(x) + U diag(c) D(x)`` is unchanged by replacing
    ``c`` with ``c - delta`` and adding ``U diag(delta) D`` to the base
    weights.  Choosing ``delta`` from the population mean continuously puts
    shared adaptation in the one decoder while leaving personal deviations in
    individual state.  The extra-latent half remains personal.
    """
    if not 0 <= fraction <= 1:
        raise ValueError("mean-fold fraction must be in [0, 1]")
    if coefficients.ndim != 2:
        raise ValueError("coefficients must be a matrix")
    if coefficients.shape[1] != model.coefficient_dim:
        raise ValueError("conditional coefficient width does not match model")
    start = model.extra_latent_dim
    lora = coefficients[:, start:]
    delta_np = (fraction * lora.mean(axis=0)).astype(np.float32)
    assimilated, diagnostics = fold_lora_delta(
        model, coefficients, delta_np)
    diagnostics.update({
        "assimilation_fraction": float(fraction),
    })
    return assimilated, diagnostics


def balanced_retirement_lora(
        coefficients: np.ndarray,
        goals: np.ndarray,
        relative_fitness: np.ndarray,
        coverage_margin: np.ndarray,
        lifetime_success: np.ndarray,
        lora_start: int,
        target_count: int,
        merge_fraction: float,
        temperature: float,
        relative_weight: float = 1.0,
        coverage_weight: float = 1.0,
        success_weight: float = 1.0,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Reduce all persistent retirements to one niche-balanced LoRA legacy.

    Retirees compete only with others assigned to the same ecological target.
    Each represented target then contributes one equally weighted legacy,
    preventing a populous lineage from dominating the shared decoder.
    """
    values = np.asarray(coefficients, dtype=np.float32)
    goals = np.asarray(goals, dtype=np.int64)
    relative = np.asarray(relative_fitness, dtype=np.float64)
    coverage = np.asarray(coverage_margin, dtype=np.float64)
    success = np.asarray(lifetime_success, dtype=np.float64)
    count = len(values)
    if values.ndim != 2 or not 0 <= lora_start < values.shape[1]:
        raise ValueError("invalid retirement coefficient matrix")
    if not all(array.shape == (count,) for array in (
            goals, relative, coverage, success)):
        raise ValueError("retirement statistics must align")
    if count and (goals.min() < 0 or goals.max() >= target_count):
        raise ValueError("retirement goal is outside target range")
    if not 0 <= merge_fraction <= 1:
        raise ValueError("retirement merge fraction must be in [0, 1]")
    if temperature <= 0:
        raise ValueError("retirement temperature must be positive")
    lora_dim = values.shape[1] - lora_start
    if not count:
        return (
            np.zeros(lora_dim, dtype=np.float32),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            {"retired_parents": 0, "legacy_niches": 0},
        )
    utility = (
        relative_weight * relative
        + coverage_weight * np.maximum(coverage, 0.0)
        + success_weight * success
    )
    scaled = utility / temperature
    maxima = np.full(target_count, -np.inf, dtype=np.float64)
    np.maximum.at(maxima, goals, scaled)
    unnormalized = np.exp(scaled - maxima[goals])
    denominators = np.bincount(
        goals, weights=unnormalized, minlength=target_count)
    weights = unnormalized / denominators[goals]
    niche_gates = np.zeros((target_count, lora_dim), dtype=np.float64)
    np.add.at(
        niche_gates,
        goals,
        weights[:, None] * values[:, lora_start:],
    )
    active = np.bincount(goals, minlength=target_count) > 0
    raw_legacy = niche_gates[active].mean(axis=0)
    delta = (merge_fraction * raw_legacy).astype(np.float32)
    return delta, weights, utility, {
        "retired_parents": int(count),
        "legacy_niches": int(active.sum()),
        "mean_retirement_utility": float(utility.mean()),
        "max_retirement_utility": float(utility.max()),
        "mean_lifetime_success": float(success.mean()),
        "coverage_legacy_count": int((coverage > 0).sum()),
        "raw_legacy_rms": float(np.sqrt(np.mean(raw_legacy ** 2))),
    }


def target_records(scores: np.ndarray, names: list[str]) -> dict[str, float]:
    best = scores.max(axis=0)
    return {name: -float(value) for name, value in zip(names, best)}


def aggregate_records(records: dict[str, float]) -> dict[str, float]:
    values = np.asarray(list(records.values()), dtype=np.float64)
    return {
        "mean_mse": float(values.mean()),
        "median_mse": float(np.median(values)),
        "worst_mse": float(values.max()),
    }


def distinct_target_representatives(
        scores: np.ndarray,
        target_order: np.ndarray | None = None,
        ) -> np.ndarray:
    """Choose one distinct current champion per target."""
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("scores must be a matrix")
    if len(values) < values.shape[1]:
        raise ValueError("population must cover every target distinctly")
    if target_order is None:
        order = np.arange(values.shape[1], dtype=np.int64)
    else:
        order = np.asarray(target_order, dtype=np.int64)
        if sorted(order.tolist()) != list(range(values.shape[1])):
            raise ValueError("target_order must be a target permutation")
    available = np.ones(len(values), dtype=bool)
    representatives: list[int] = []
    for target in order:
        winner = int(np.argmax(np.where(
            available, values[:, target], -np.inf)))
        representatives.append(winner)
        available[winner] = False
    return np.asarray(representatives, dtype=np.int64)


def lineage_succession_selection_scores(
        parent_scores: np.ndarray,
        child_scores: np.ndarray,
        parent_goals: np.ndarray,
        parent_age: np.ndarray,
        child_parents: np.ndarray,
        child_mates: np.ndarray,
        retirement_age: int,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Retire old adults only through descendants competing for their role.

    A retiring adult's target seat is temporarily restricted to children for
    which that adult was either primary parent or mate. If no such child was
    evaluated this generation, the adult receives a reprieve. Actual scores
    are unchanged outside survivor selection, and the external archive keeps
    the retired phenotype reproducible.
    """
    parents = np.asarray(parent_scores, dtype=np.float64)
    children = np.asarray(child_scores, dtype=np.float64)
    goals = np.asarray(parent_goals, dtype=np.int64)
    ages = np.asarray(parent_age, dtype=np.int64)
    child_parents = np.asarray(child_parents, dtype=np.int64)
    child_mates = np.asarray(child_mates, dtype=np.int64)
    if parents.ndim != 2 or children.ndim != 2:
        raise ValueError("parent_scores and child_scores must be matrices")
    if parents.shape[1] != children.shape[1]:
        raise ValueError("parent and child target counts must match")
    if goals.shape != (len(parents),) or ages.shape != (len(parents),):
        raise ValueError("parent goals and ages must match parent count")
    if (child_parents.shape != (len(children),)
            or child_mates.shape != (len(children),)):
        raise ValueError("child parent arrays must match child count")
    if retirement_age < 1:
        raise ValueError("retirement_age must be positive")

    parent_selection = parents.copy()
    child_selection = children.copy()
    expired = np.zeros(len(parents), dtype=bool)
    allowed_by_target = np.zeros(
        (parents.shape[1], len(children)), dtype=bool)
    reprieves = 0
    for adult in np.flatnonzero(ages >= retirement_age):
        descendants = ((child_parents == adult) | (child_mates == adult))
        if descendants.any():
            expired[adult] = True
            allowed_by_target[int(goals[adult])] |= descendants
        else:
            reprieves += 1

    parent_selection[expired] = -np.inf
    succession_targets = np.flatnonzero(allowed_by_target.any(axis=1))
    for target in succession_targets:
        parent_selection[:, target] = -np.inf
        child_selection[~allowed_by_target[target], target] = -np.inf
    return parent_selection, child_selection, expired, {
        "lineage_succession_targets": int(len(succession_targets)),
        "lineage_retirements": int(expired.sum()),
        "lineage_reprieves": int(reprieves),
    }


def decoder_input_vectors(
        mode: str,
        z: np.ndarray,
        coefficients: np.ndarray | None = None,
        ) -> np.ndarray:
    """Return the values fed directly into the shared decoder.

    ``z`` is always a decoder input. In ``latent`` mode every conditional
    coefficient is an additional decoder input; in ``mixed`` mode only the
    first half is. The remaining mixed coefficients, and every coefficient in
    ``lora`` mode, are LoRA gates. They are inherited and mutated, but are not
    decoder inputs and therefore never appear in this vector.
    """
    latent = np.asarray(z, dtype=np.float32)
    if latent.ndim != 2 or latent.shape[1] != LATENT:
        raise ValueError(f"z must have shape (individuals, {LATENT})")
    if mode == "lora" or coefficients is None:
        return np.ascontiguousarray(latent)
    conditional = np.asarray(coefficients, dtype=np.float32)
    if conditional.ndim != 2 or conditional.shape[0] != latent.shape[0]:
        raise ValueError(
            "coefficients must be a matrix with the same individuals as z")
    if mode == "mixed":
        if conditional.shape[1] < 2 or conditional.shape[1] % 2:
            raise ValueError("mixed coefficients must have positive even width")
        conditional = conditional[:, :conditional.shape[1] // 2]
    elif mode != "latent":
        raise ValueError(f"unknown conditional mode: {mode}")
    return np.ascontiguousarray(
        np.concatenate([latent, conditional], axis=1))


def mating_compatibility_vectors(
        space: str,
        mode: str,
        z: np.ndarray,
        coefficients: np.ndarray | None,
        scores: np.ndarray,
        ) -> np.ndarray:
    """Select the reproductive distance space without changing fitness roles."""
    if space == "z_only":
        return decoder_input_vectors(mode, z)
    if space == "decoder_input":
        return decoder_input_vectors(mode, z, coefficients)
    if space == "fitness":
        return normalize_species_vectors(scores)
    raise ValueError(f"unknown compatibility space: {space}")


def choose_ecological_mates_within_input_species(
        compatibility_vectors: np.ndarray,
        fitness_vectors: np.ndarray,
        parents: np.ndarray,
        compatibility_radius: float,
        fitness_radius: float,
        rng: np.random.Generator,
        mate_weights: np.ndarray | None = None,
        ) -> tuple[np.ndarray, np.ndarray]:
    """Choose fitness-similar mates inside transitive input-space species.

    Decoder-input distance alone builds the reproductive species graph.
    Membership is transitive: if A connects to B and B to C, A and C are in
    the same species even when their direct distance exceeds the radius.
    Fitness supplies the ecological preference used to choose a mate inside
    that species; it never changes the input-space graph itself.
    """
    compatibility_adjacency, compatibility_distance = compatibility_graph(
        compatibility_vectors, compatibility_radius)
    components = connected_components(compatibility_adjacency)
    species = np.empty(len(compatibility_vectors), dtype=np.int64)
    for index, component in enumerate(components):
        species[component] = index
    ecological_adjacency, _ = compatibility_graph(
        fitness_vectors, fitness_radius)
    if mate_weights is None:
        weights = np.ones(len(compatibility_vectors), dtype=np.float64)
    else:
        weights = np.asarray(mate_weights, dtype=np.float64)
        if weights.shape != (len(compatibility_vectors),):
            raise ValueError("mate_weights must match the population")
        if np.any(weights < 0):
            raise ValueError("mate_weights must be non-negative")
    parent_indices = np.asarray(parents, dtype=np.int64)
    mates = np.full(len(parent_indices), -1, dtype=np.int64)
    mate_distances = np.full(len(parent_indices), np.nan, dtype=np.float64)
    for row, parent in enumerate(parent_indices):
        candidates = np.flatnonzero(
            ecological_adjacency[parent]
            & (species == species[parent])
            & (weights > 0))
        if len(candidates):
            probabilities = weights[candidates]
            if np.all(probabilities == probabilities[0]):
                mate = int(rng.choice(candidates))
            else:
                probabilities = probabilities / probabilities.sum()
                mate = int(rng.choice(candidates, p=probabilities))
            mates[row] = mate
            mate_distances[row] = compatibility_distance[parent, mate]
    return mates, mate_distances


def select_species_local_survivors(
        parent_scores: np.ndarray,
        child_scores: np.ndarray,
        child_priority: np.ndarray,
        parent_goals: np.ndarray,
        child_goals: np.ndarray,
        parent_compatibility: np.ndarray,
        child_compatibility: np.ndarray,
        compatibility_radius: float,
        survivor_count: int,
        target_order: np.ndarray | None = None,
        ) -> tuple[np.ndarray, np.ndarray, int, dict]:
    """Admit children through protected, species-local replacement.

    One distinct best representative for every fitness target is protected
    across parents and children. Each other child can enter only by beating
    the closest unprotected adult with the same role in its transitive
    compatibility component. Protected children may found or rescue a
    lineage, but still evict only an unprotected seat.
    """
    parents = np.asarray(parent_scores, dtype=np.float64)
    children = np.asarray(child_scores, dtype=np.float64)
    priority = np.asarray(child_priority, dtype=np.float64)
    parent_roles = np.asarray(parent_goals, dtype=np.int64)
    child_roles = np.asarray(child_goals, dtype=np.int64)
    parent_vectors = np.asarray(parent_compatibility, dtype=np.float64)
    child_vectors = np.asarray(child_compatibility, dtype=np.float64)
    if parents.ndim != 2 or children.ndim != 2:
        raise ValueError("parent_scores and child_scores must be matrices")
    if parents.shape[1] != children.shape[1]:
        raise ValueError("parent and child target counts must match")
    if len(parents) != survivor_count:
        raise ValueError("local replacement requires one seat per parent")
    if survivor_count < parents.shape[1]:
        raise ValueError("survivors must cover every target")
    if priority.shape != (len(children),):
        raise ValueError("child_priority must match child count")
    if parent_roles.shape != (len(parents),):
        raise ValueError("parent_goals must match parent count")
    if child_roles.shape != (len(children),):
        raise ValueError("child_goals must match child count")
    if (parent_vectors.ndim != 2 or child_vectors.ndim != 2
            or parent_vectors.shape[1] != child_vectors.shape[1]
            or len(parent_vectors) != len(parents)
            or len(child_vectors) != len(children)):
        raise ValueError("compatibility matrices must align with scores")

    target_count = parents.shape[1]
    if target_order is None:
        order = np.arange(target_count, dtype=np.int64)
    else:
        order = np.asarray(target_order, dtype=np.int64)
        if sorted(order.tolist()) != list(range(target_count)):
            raise ValueError("target_order must be a target permutation")

    combined_scores = np.concatenate([parents, children], axis=0)
    combined_vectors = np.concatenate(
        [parent_vectors, child_vectors], axis=0)
    parent_count = len(parents)
    adjacency, distances = compatibility_graph(
        combined_vectors, compatibility_radius)
    components = connected_components(adjacency)
    species = np.empty(len(combined_vectors), dtype=np.int64)
    for index, component in enumerate(components):
        species[component] = index

    available = np.ones(len(combined_scores), dtype=bool)
    protected: list[int] = []
    protected_roles: dict[int, int] = {}
    for target in order:
        values = np.where(
            available, combined_scores[:, target], -np.inf)
        winner = int(np.argmax(values))
        protected.append(winner)
        protected_roles[winner] = int(target)
        available[winner] = False
    protected_set = set(protected)

    selected = set(range(parent_count))
    roles = {index: int(parent_roles[index])
             for index in range(parent_count)}
    for index, role in protected_roles.items():
        if index < parent_count:
            roles[index] = role

    forced_nonlocal = 0
    replacement_distances: list[float] = []

    def victim_for(child: int, role: int, allow_fallback: bool) -> int | None:
        candidates = np.asarray(sorted(
            selected - protected_set), dtype=np.int64)
        if not len(candidates):
            return None
        same_species = candidates[species[candidates] == species[child]]
        same_role = same_species[
            np.asarray([roles[int(value)] == role
                        for value in same_species], dtype=bool)]
        pool = same_role
        if not len(pool) and allow_fallback:
            pool = same_species
        if not len(pool) and allow_fallback:
            global_role = candidates[np.asarray([
                roles[int(value)] == role for value in candidates
            ], dtype=bool)]
            pool = global_role if len(global_role) else candidates
        if not len(pool):
            return None
        return int(pool[np.argmin(distances[child, pool])])

    # Protected children enter first. Their target contribution is the reason
    # for admission, so they may found a disconnected species if necessary.
    for child in protected:
        if child < parent_count:
            continue
        role = protected_roles[child]
        victim = victim_for(child, role, allow_fallback=True)
        if victim is None:
            raise RuntimeError("no redundant seat for protected child")
        forced_nonlocal += int(species[victim] != species[child])
        replacement_distances.append(float(distances[child, victim]))
        selected.remove(victim)
        roles.pop(victim)
        selected.add(child)
        roles[child] = role

    local_replacements = 0
    ranked_children = sorted(
        (parent_count + index for index in range(len(children))
         if parent_count + index not in protected_set),
        key=lambda index: (-priority[index - parent_count], index),
    )
    for child in ranked_children:
        role = int(child_roles[child - parent_count])
        victim = victim_for(child, role, allow_fallback=False)
        if victim is None:
            continue
        if combined_scores[child, role] <= combined_scores[victim, role]:
            continue
        replacement_distances.append(float(distances[child, victim]))
        selected.remove(victim)
        roles.pop(victim)
        selected.add(child)
        roles[child] = role
        local_replacements += 1

    if len(selected) != survivor_count or not protected_set <= selected:
        raise RuntimeError("local replacement violated protected population size")
    keep = np.asarray(
        protected + sorted(selected - protected_set), dtype=np.int64)
    selected_roles = np.asarray([roles[int(index)] for index in keep],
                                dtype=np.int64)
    target_children = sum(index >= parent_count for index in protected)
    diagnostics = {
        "local_replacements": int(local_replacements),
        "protected_child_replacements": int(target_children),
        "forced_nonlocal_replacements": int(forced_nonlocal),
        "rejected_children": int(
            len(children) - target_children - local_replacements),
        "mean_replacement_distance": (
            float(np.mean(replacement_distances))
            if replacement_distances else None),
    }
    return keep, selected_roles, int(target_children), diagnostics


def run(args: argparse.Namespace) -> dict:
    names, target_arrays = load_targets(args.targets)
    target_count = len(names)
    if args.survivors < target_count:
        raise ValueError("survivors must cover every target")
    if args.children < args.survivors:
        raise ValueError("children must be at least survivors")
    if (not args.start_shared
            and not args.survivors <= args.transition_at < args.budget):
        raise ValueError("transition must follow founders and precede budget")
    if args.dynamic_assimilation and not args.start_shared:
        raise ValueError("dynamic assimilation currently requires --start-shared")
    if args.start_shared and args.mode != "mixed":
        raise ValueError("the always-shared experiment currently requires mixed mode")
    if not 0 <= args.retirement_merge_fraction <= 1:
        raise ValueError("retirement merge fraction must be in [0, 1]")
    if args.retirement_temperature <= 0:
        raise ValueError("retirement temperature must be positive")
    if args.mating_radius is None:
        args.mating_radius = {
            "z_only": 30.0,
            "decoder_input": 50.0,
            "fitness": 0.3,
        }[args.compatibility_space]
    if args.mating_radius < 0:
        raise ValueError("mating radius must be non-negative")
    if args.ecological_mating_radius < 0:
        raise ValueError("ecological mating radius must be non-negative")
    if (args.max_reproductive_age is not None
            and args.max_reproductive_age < 1):
        raise ValueError("max reproductive age must be positive")
    if not 0 <= args.senescent_reproduction_weight <= 1:
        raise ValueError("senescent reproduction weight must be in [0, 1]")
    if args.senescent_mutation_multiplier <= 0:
        raise ValueError("senescent mutation multiplier must be positive")
    _require_mps()
    device = "mps"
    targets = torch.as_tensor(target_arrays, device=device)
    config = ExplorerConfig()
    _seed_everything(args.seed)
    rng = np.random.default_rng(args.seed)
    template = _Template(resolve(
        lambda latent, shape: ConvRGB(latent, shape), LATENT, SHAPE), device)

    def score(phenotypes: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            values = pairwise_negative_mse(
                phenotypes.reshape(len(phenotypes), *SHAPE), targets)
        return values.cpu().numpy().astype(np.float32)

    founder_theta = template.init_theta(int(rng.integers(0, 2**31)))
    population_z = rng.standard_normal(
        (args.survivors, LATENT)).astype(np.float32)
    population_theta = np.repeat(
        founder_theta[None], args.survivors, axis=0)
    population_coefficients: np.ndarray | None = None
    model: nn.Module | None = None
    assimilation_optimizer: torch.optim.Optimizer | None = None
    transitioned = bool(args.start_shared)
    if args.start_shared:
        model = initialize_conditional_decoder(
            args.mode, args.coefficient_dim, founder_theta, device)
        population_coefficients = np.zeros(
            (args.survivors, args.coefficient_dim), dtype=np.float32)
        population_phenotypes = decode_conditional(
            model, population_z, population_coefficients, device)
        if (args.dynamic_assimilation
                and args.assimilation_method == "replay"):
            assimilation_optimizer = torch.optim.Adam(
                model.parameters(), lr=args.assimilation_learning_rate)
    else:
        population_phenotypes = template.decode_batch(
            population_theta, population_z)
    population_scores = score(population_phenotypes)
    population_goals = normalize_species_vectors(
        population_scores).argmax(axis=1)
    population_age = np.zeros(args.survivors, dtype=np.int64)
    population_step_gain = np.ones(args.survivors, dtype=np.float64)
    population_success_wins = np.zeros(args.survivors, dtype=np.int64)
    population_success_attempts = np.zeros(args.survivors, dtype=np.int64)
    population_stagnation_attempts = np.zeros(args.survivors, dtype=np.int64)
    population_lifetime_wins = np.zeros(args.survivors, dtype=np.int64)
    population_lifetime_attempts = np.zeros(args.survivors, dtype=np.int64)
    spent = args.survivors
    generation = 0
    gain = float(args.start_gain)
    global_stall = 0
    hall_scores = population_scores.max(axis=0).astype(np.float64)
    factorization_trace: list[dict] = []
    consolidations: list[dict] = []
    assimilations: list[dict] = []
    legacy_bank: list[dict | None] = [None] * target_count
    trace: list[dict] = []
    next_report = 0
    report_interval = max(1, args.budget // max(args.reports, 1))
    next_consolidation = args.transition_at + args.consolidate_every
    view = (ReferenceSpeciesView(names, target_arrays, args.budget)
            if args.live else None)

    def display_hall() -> list[dict]:
        images = population_phenotypes.detach().cpu().numpy().reshape(
            len(population_z), *SHAPE)
        return [
            {
                "score": float(population_scores[
                    int(np.argmax(population_scores[:, target])), target]),
                "image": images[int(np.argmax(
                    population_scores[:, target]))],
            }
            for target in range(target_count)
        ]

    if view is not None:
        view.update(spent, display_hall())

    if args.start_shared:
        trace.append({
            "e": spent,
            "event": "always_shared_initialization",
            "mode": args.mode,
            "coefficient_dim": args.coefficient_dim,
        })

    def mutate(values: np.ndarray, multiplier: float = 1.0) -> np.ndarray:
        mask = rng.random(values.shape) < config.genome_mutation_rate
        if not mask.any():
            mask[rng.integers(0, len(values))] = True
        return (values + mask * rng.normal(
            0, config.genome_mutation_sigma * gain * multiplier,
            values.shape)).astype(np.float32)

    def crossover(base: np.ndarray, donor: np.ndarray) -> np.ndarray:
        cut = int(rng.integers(1, len(base)))
        child = base.copy()
        child[cut:] = donor[cut:]
        return child.astype(np.float32)

    def crossover_conditional(base: np.ndarray,
                              donor: np.ndarray) -> np.ndarray:
        if args.mode != "mixed":
            return crossover(base, donor)
        half = len(base) // 2
        return np.concatenate([
            crossover(base[:half], donor[:half]),
            crossover(base[half:], donor[half:]),
        ]).astype(np.float32)

    while spent < args.budget:
        if not transitioned and spent >= args.transition_at:
            theta_mean = population_theta.mean(axis=0)
            medoid = int(np.argmin(np.mean(
                (population_theta - theta_mean[None]) ** 2, axis=1)))
            model = initialize_conditional_decoder(
                args.mode, args.coefficient_dim,
                population_theta[medoid], device)
            teachers = population_phenotypes.detach().cpu().numpy().reshape(
                len(population_z), *SHAPE)
            population_z, population_coefficients, factorization_trace = (
                fit_conditional_decoder(
                    model, population_z, teachers, args.coefficient_dim,
                    args.factor_steps, args.learning_rate,
                    args.code_learning_rate, args.factor_base_weight, device,
                ))
            population_phenotypes = decode_conditional(
                model, population_z, population_coefficients, device)
            population_scores = score(population_phenotypes)
            population_goals = normalize_species_vectors(
                population_scores).argmax(axis=1)
            population_age.fill(0)
            population_step_gain.fill(1.0)
            population_success_wins.fill(0)
            population_success_attempts.fill(0)
            population_stagnation_attempts.fill(0)
            population_lifetime_wins.fill(0)
            population_lifetime_attempts.fill(0)
            hall_scores = population_scores.max(axis=0).astype(np.float64)
            transitioned = True
            base_phenotypes = decode_conditional(
                model, population_z, np.zeros_like(population_coefficients),
                device)
            base_records = target_records(score(base_phenotypes), names)
            adapted_records = target_records(population_scores, names)
            transition_row = {
                "e": spent,
                "event": "conditional_factorization",
                "mode": args.mode,
                "coefficient_dim": args.coefficient_dim,
                **{f"adapted_{key}": value for key, value in
                   aggregate_records(adapted_records).items()},
                **{f"base_{key}": value for key, value in
                   aggregate_records(base_records).items()},
            }
            trace.append(transition_row)
            print(
                f"\nCONDITIONAL {args.mode} at {spent}: one shared decoder; "
                f"adapted mean {transition_row['adapted_mean_mse']:.6f}; "
                f"base mean {transition_row['base_mean_mse']:.6f}",
                flush=True,
            )
            if view is not None:
                view.update(spent, display_hall())
            continue

        generation += 1
        count = min(args.children, args.budget - spent)
        if not transitioned and spent < args.transition_at:
            count = min(count, args.transition_at - spent)
        senescent = np.zeros(len(population_z), dtype=bool)
        reproduction_weights = np.ones(len(population_z), dtype=np.float64)
        if transitioned and args.max_reproductive_age is not None:
            senescent = population_age >= args.max_reproductive_age
            reproduction_weights[senescent] = (
                args.senescent_reproduction_weight)
            if not np.any(reproduction_weights > 0):
                # A stalled all-senescent population falls back to its
                # youngest cohort without deleting protected champions.
                reproduction_weights[
                    population_age == population_age.min()] = 1.0
        reproductive_indices = np.flatnonzero(reproduction_weights > 0)
        active_weights = reproduction_weights[reproductive_indices]
        if len(reproductive_indices) == len(population_z) and np.all(
                active_weights == 1.0):
            parent = rng.integers(0, len(population_z), count)
        elif np.all(active_weights == active_weights[0]):
            parent = rng.choice(reproductive_indices, count, replace=True)
        else:
            parent = rng.choice(
                np.arange(len(population_z)), count, replace=True,
                p=reproduction_weights / reproduction_weights.sum())
        parent_fitness_vectors = normalize_species_vectors(population_scores)
        parent_compatibility_vectors = mating_compatibility_vectors(
            args.compatibility_space,
            args.mode,
            population_z,
            population_coefficients if transitioned else None,
            population_scores,
        )
        if args.compatibility_space != "fitness":
            mates, mate_distances = (
                choose_ecological_mates_within_input_species(
                    parent_compatibility_vectors,
                    parent_fitness_vectors,
                    parent,
                    args.mating_radius,
                    args.ecological_mating_radius,
                    rng,
                    reproduction_weights,
                ))
        else:
            mates, mate_distances = choose_compatible_mates(
                parent_compatibility_vectors,
                parent,
                args.mating_radius,
                rng,
            )
        sexual = mates >= 0
        inherited_goals = population_goals[parent]
        individual_active = transitioned
        child_multipliers = (
            population_step_gain[parent].copy()
            if individual_active else np.ones(count, dtype=np.float64)
        )
        child_multipliers *= np.where(
            senescent[parent], args.senescent_mutation_multiplier, 1.0)

        child_z = np.empty((count, LATENT), dtype=np.float32)
        child_coefficients = (
            np.empty((count, args.coefficient_dim), dtype=np.float32)
            if transitioned else None)
        base_theta = population_theta[parent].copy()
        for i, (base, mate) in enumerate(zip(parent, mates)):
            if mate >= 0:
                child_z[i] = mutate(crossover(
                    population_z[base], population_z[mate]),
                    child_multipliers[i])
                if transitioned:
                    assert population_coefficients is not None
                    assert child_coefficients is not None
                    child_coefficients[i] = mutate(crossover_conditional(
                        population_coefficients[base],
                        population_coefficients[mate]),
                        child_multipliers[i])
                else:
                    base_theta[i] = (
                        population_theta[base] + population_theta[mate]) / 2.0
            else:
                child_z[i] = mutate(
                    population_z[base], child_multipliers[i])
                if transitioned:
                    assert population_coefficients is not None
                    assert child_coefficients is not None
                    child_coefficients[i] = mutate(
                        population_coefficients[base], child_multipliers[i])

        if transitioned:
            assert model is not None and child_coefficients is not None
            child_phenotypes = decode_conditional(
                model, child_z, child_coefficients, device)
            child_theta = np.repeat(
                np.zeros((1, 1), dtype=np.float32), count, axis=0)
        else:
            sigmas = np.exp(rng.uniform(
                np.log(config.weight_sigma_low),
                np.log(config.weight_sigma_high), count)) * gain
            scales = np.maximum(base_theta.std(axis=1), 1e-3)
            child_theta = (
                base_theta
                + (sigmas * scales)[:, None]
                * rng.standard_normal((count, template.n_params))
            ).astype(np.float32)
            child_phenotypes = template.decode_batch(child_theta, child_z)

        child_scores = score(child_phenotypes)
        spent += count
        child_vectors = normalize_species_vectors(child_scores)
        goals = child_vectors.argmax(axis=1)
        relative_fitness = child_vectors[np.arange(count), goals]
        child_quality = child_scores[np.arange(count), inherited_goals]
        parent_quality = population_scores[parent, inherited_goals]
        target_scale = np.maximum(population_scores.std(axis=0), 1e-4)
        parent_progress = np.clip(
            (child_quality - parent_quality) / target_scale[inherited_goals],
            -5.0, 5.0)
        selection_priority = relative_fitness + args.progress_weight * parent_progress
        child_wins = child_quality >= parent_quality - 1e-12
        birth_attempts = np.bincount(
            parent, minlength=len(population_z)).astype(np.int64)
        birth_wins = np.bincount(
            parent,
            weights=child_wins.astype(np.int64),
            minlength=len(population_z),
        ).astype(np.int64)
        updated_lifetime_wins = population_lifetime_wins + birth_wins
        updated_lifetime_attempts = (
            population_lifetime_attempts + birth_attempts)
        win_rate = float(child_wins.mean())
        gain *= (config.gain_step if win_rate > config.win_target
                 else 1 / config.gain_step)
        gain = float(np.clip(gain, 0.3, config.gain_limits[1]))
        improved = bool(np.any(child_scores.max(axis=0) > hall_scores + 1e-12))
        hall_scores = np.maximum(hall_scores, child_scores.max(axis=0))
        global_stall = 0 if improved else global_stall + 1
        if global_stall >= 25:
            gain = min(gain * 3.0, config.gain_limits[1])
            global_stall = 0

        (updated_gain, updated_wins, updated_attempts, updated_stagnation,
         _, stagnation_kicks) = update_individual_step_state(
            population_step_gain,
            population_success_wins,
            population_success_attempts,
            population_stagnation_attempts,
            parent, child_wins,
            "stagnation" if transitioned else "off",
            config.win_target, config.gain_step,
            args.individual_success_window,
            args.individual_stagnation_attempts,
            args.individual_stagnation_kick,
            (args.individual_gain_min, args.individual_gain_max),
        )
        child_gain = updated_gain[parent]
        child_success_wins = updated_wins[parent]
        child_success_attempts = updated_attempts[parent]
        child_stagnation = updated_stagnation[parent]

        expired_count = 0
        protected_old_adults = 0
        lineage_diagnostics = {
            "lineage_succession_targets": 0,
            "lineage_retirements": 0,
            "lineage_reprieves": 0,
        }
        selection_diagnostics = {
            "local_replacements": 0,
            "protected_child_replacements": 0,
            "forced_nonlocal_replacements": 0,
            "rejected_children": 0,
            "mean_replacement_distance": None,
        }
        retirement_payload = None
        if count < args.survivors:
            # Match the reference runner: a final undersized evaluation batch
            # updates step statistics but cannot form a complete generation.
            target_children = 0
            population_step_gain = updated_gain
            population_success_wins = updated_wins
            population_success_attempts = updated_attempts
            population_stagnation_attempts = updated_stagnation
            population_lifetime_wins = updated_lifetime_wins
            population_lifetime_attempts = updated_lifetime_attempts
            if transitioned:
                hall_scores = population_scores.max(axis=0).astype(np.float64)
        else:
            target_order = np.argsort(hall_scores)
            if args.death_policy == "species_local":
                parent_compatibility = mating_compatibility_vectors(
                    args.compatibility_space,
                    args.mode,
                    population_z,
                    population_coefficients if transitioned else None,
                    population_scores,
                )
                child_compatibility = mating_compatibility_vectors(
                    args.compatibility_space,
                    args.mode,
                    child_z,
                    child_coefficients if transitioned else None,
                    child_scores,
                )
                (keep, selected_goals, target_children,
                 selection_diagnostics) = select_species_local_survivors(
                    population_scores,
                    child_scores,
                    selection_priority,
                    population_goals,
                    goals,
                    parent_compatibility,
                    child_compatibility,
                    args.mating_radius,
                    args.survivors,
                    target_order,
                )
            else:
                parent_scores_for_selection = population_scores
                child_scores_for_selection = child_scores
                if (transitioned and args.max_age is not None
                        and args.death_policy == "lineage_succession"):
                    (parent_scores_for_selection,
                     child_scores_for_selection,
                     expired,
                     lineage_diagnostics) = (
                        lineage_succession_selection_scores(
                            population_scores,
                            child_scores,
                            population_goals,
                            population_age,
                            parent,
                            mates,
                            args.max_age,
                        ))
                    expired_count = int(expired.sum())
                elif transitioned and args.max_age is not None:
                    expired = population_age >= args.max_age
                    if args.death_policy == "protected_age":
                        protected_parents = distinct_target_representatives(
                            population_scores, target_order)
                        protected_old_adults = int(
                            expired[protected_parents].sum())
                        expired[protected_parents] = False
                    expired_count = int(expired.sum())
                    if expired_count:
                        parent_scores_for_selection = population_scores.copy()
                        parent_scores_for_selection[expired] = -np.inf
                keep, selected_goals, target_children = (
                    select_target_covered_survivors(
                        parent_scores_for_selection,
                        child_scores_for_selection,
                        selection_priority, goals, args.survivors,
                        target_order))
                if args.death_policy == "lineage_succession":
                    selection_matrix = np.concatenate([
                        parent_scores_for_selection,
                        child_scores_for_selection,
                    ], axis=0)
                    target_values = selection_matrix[
                        keep[:target_count], selected_goals[:target_count]]
                    if not np.isfinite(target_values).all():
                        raise RuntimeError(
                            "lineage succession exhausted a target candidate")
            combined_scores = np.concatenate(
                [population_scores, child_scores], axis=0)
            if (transitioned and args.dynamic_assimilation
                    and args.assimilation_method in {
                        "retirement_fold", "retirement_archive"}):
                assert population_coefficients is not None
                parent_count = len(population_z)
                kept_parents = keep[keep < parent_count]
                retired_mask = np.ones(parent_count, dtype=bool)
                retired_mask[kept_parents] = False
                retired = np.flatnonzero(retired_mask)
                if len(retired):
                    roles = population_goals[retired]
                    replacement_best = combined_scores[keep].max(axis=0)
                    coverage_margin = (
                        population_scores[retired, roles]
                        - replacement_best[roles]
                    ) / target_scale[roles]
                    lifetime_success = (
                        updated_lifetime_wins[retired]
                        / np.maximum(updated_lifetime_attempts[retired], 1)
                    )
                    retirement_payload = {
                        "indices": retired,
                        "z": population_z[retired].copy(),
                        "coefficients": population_coefficients[retired].copy(),
                        "goals": roles.copy(),
                        "relative_fitness": parent_fitness_vectors[
                            retired, roles].copy(),
                        "coverage_margin": coverage_margin.astype(np.float64),
                        "lifetime_success": lifetime_success.astype(np.float64),
                        "role_scores": population_scores[retired, roles].copy(),
                        "ages": population_age[retired].copy(),
                        "lifetime_wins": updated_lifetime_wins[retired].copy(),
                        "lifetime_attempts": (
                            updated_lifetime_attempts[retired].copy()),
                    }
            population_goals = selected_goals
            population_z = np.concatenate(
                [population_z, child_z], axis=0)[keep]
            population_scores = combined_scores[keep]
            combined_phenotypes = torch.cat(
                [population_phenotypes, child_phenotypes], dim=0)
            population_phenotypes = combined_phenotypes[
                torch.as_tensor(keep, device=device)]
            population_age = np.concatenate([
                population_age + 1, np.zeros(count, dtype=np.int64)])[keep]
            population_step_gain = np.concatenate(
                [updated_gain, child_gain])[keep]
            population_success_wins = np.concatenate(
                [updated_wins, child_success_wins])[keep]
            population_success_attempts = np.concatenate(
                [updated_attempts, child_success_attempts])[keep]
            population_stagnation_attempts = np.concatenate(
                [updated_stagnation, child_stagnation])[keep]
            population_lifetime_wins = np.concatenate([
                updated_lifetime_wins,
                np.zeros(count, dtype=np.int64),
            ])[keep]
            population_lifetime_attempts = np.concatenate([
                updated_lifetime_attempts,
                np.zeros(count, dtype=np.int64),
            ])[keep]
            if transitioned:
                assert population_coefficients is not None
                assert child_coefficients is not None
                population_coefficients = np.concatenate(
                    [population_coefficients, child_coefficients], axis=0)[keep]
                # Only currently reproducible shared-decoder records survive.
                hall_scores = population_scores.max(axis=0).astype(np.float64)
            else:
                population_theta = np.concatenate(
                    [population_theta, child_theta], axis=0)[keep]

        assimilation = None
        if (transitioned and args.dynamic_assimilation
                and (spent < args.budget
                     or args.assimilation_method == "retirement_archive")):
            assert model is not None and population_coefficients is not None
            teacher_images = population_phenotypes.detach().cpu().numpy().reshape(
                len(population_z), *SHAPE)
            if args.assimilation_method == "mean_fold":
                if not isinstance(model, ConditionalLoRAConvRGB):
                    raise TypeError("mean-fold assimilation requires LoRA")
                base_before = decode_conditional(
                    model, population_z,
                    np.zeros_like(population_coefficients), device)
                population_coefficients, assimilation = (
                    fold_population_mean_lora(
                        model, population_coefficients,
                        args.assimilation_mean_fraction,
                    ))
            elif args.assimilation_method in {
                    "retirement_fold", "retirement_archive"}:
                if not isinstance(model, ConditionalLoRAConvRGB):
                    raise TypeError("retirement handling requires LoRA")
                if retirement_payload is None:
                    assimilation = None
                else:
                    fold_retirement = (
                        args.assimilation_method == "retirement_fold")
                    if fold_retirement:
                        base_before = decode_conditional(
                            model, population_z,
                            np.zeros_like(population_coefficients), device)
                    delta, legacy_weights, legacy_utility, retirement = (
                        balanced_retirement_lora(
                            retirement_payload["coefficients"],
                            retirement_payload["goals"],
                            retirement_payload["relative_fitness"],
                            retirement_payload["coverage_margin"],
                            retirement_payload["lifetime_success"],
                            model.extra_latent_dim,
                            target_count,
                            (args.retirement_merge_fraction
                             if fold_retirement else 0.0),
                            args.retirement_temperature,
                            args.retirement_relative_weight,
                            args.retirement_coverage_weight,
                            args.retirement_success_weight,
                        ))
                    for local_index, target in enumerate(
                            retirement_payload["goals"]):
                        target = int(target)
                        role_score = float(
                            retirement_payload["role_scores"][local_index])
                        incumbent = legacy_bank[target]
                        if (incumbent is None
                                or role_score > incumbent["role_score"]):
                            legacy_bank[target] = {
                                "z": retirement_payload["z"][local_index].copy(),
                                "coefficients": retirement_payload[
                                    "coefficients"][local_index].copy(),
                                "role_score": role_score,
                                "utility": float(legacy_utility[local_index]),
                                "retirement_weight": float(
                                    legacy_weights[local_index]),
                                "age": int(
                                    retirement_payload["ages"][local_index]),
                                "lifetime_wins": int(retirement_payload[
                                    "lifetime_wins"][local_index]),
                                "lifetime_attempts": int(retirement_payload[
                                    "lifetime_attempts"][local_index]),
                                "retired_at": int(spent),
                            }
                    if fold_retirement:
                        population_coefficients, assimilation = fold_lora_delta(
                            model, population_coefficients, delta)
                        for legacy in legacy_bank:
                            if legacy is not None:
                                legacy["coefficients"][
                                    model.extra_latent_dim:] -= delta
                    else:
                        assimilation = {
                            "folded_lora_mean_rms": 0.0,
                            "coefficient_rms": float(np.sqrt(np.mean(
                                population_coefficients ** 2))),
                            "archive_only": True,
                        }
                    assimilation.update(retirement)
                    assimilation.update({
                        "assimilation_fraction": (
                            float(args.retirement_merge_fraction)
                            if fold_retirement else 0.0),
                    })
            else:
                assert assimilation_optimizer is not None
                population_coefficients, assimilation = (
                    assimilate_conditional_decoder(
                        model, assimilation_optimizer, population_z,
                        population_coefficients, teacher_images,
                        args.assimilation_max_fraction,
                        args.assimilation_debt_scale,
                        args.assimilation_steps,
                        args.assimilation_base_weight,
                        device,
                    ))
            if (assimilation is not None
                    and args.assimilation_method != "retirement_archive"):
                population_phenotypes = decode_conditional(
                    model, population_z, population_coefficients, device)
                population_scores = score(population_phenotypes)
                hall_scores = population_scores.max(axis=0).astype(np.float64)
            if (assimilation is not None
                    and args.assimilation_method in {
                        "mean_fold", "retirement_fold"}):
                phenotype_after = population_phenotypes.detach().cpu().numpy().reshape(
                    len(population_z), *SHAPE)
                assimilation.update({
                    "phenotype_debt_before": float(np.mean(
                        (base_before.detach().cpu().numpy().reshape(
                            len(population_z), *SHAPE) - teacher_images) ** 2)),
                    "adapted_phenotype_mse": float(np.mean(
                        (phenotype_after - teacher_images) ** 2)),
                })
            if assimilation is not None:
                assimilation.update({"e": spent})
                assimilations.append(assimilation)
        elif (transitioned and args.consolidate_every > 0
                and spent >= next_consolidation
                and spent < args.budget):
            assert model is not None and population_coefficients is not None
            teacher_images = population_phenotypes.detach().cpu().numpy().reshape(
                len(population_z), *SHAPE)
            population_coefficients, consolidation = (
                consolidate_conditional_decoder(
                    model, population_z, population_coefficients,
                    teacher_images, args.consolidation_shrink,
                    args.consolidation_steps, args.learning_rate,
                    args.consolidation_base_weight, device,
                ))
            population_phenotypes = decode_conditional(
                model, population_z, population_coefficients, device)
            population_scores = score(population_phenotypes)
            hall_scores = population_scores.max(axis=0).astype(np.float64)
            consolidation.update({"e": spent})
            consolidations.append(consolidation)
            while next_consolidation <= spent:
                next_consolidation += args.consolidate_every
            print(
                f"    consolidate at {spent}: adapted "
                f"{consolidation['adapted_phenotype_mse']:.7f}, base "
                f"{consolidation['base_phenotype_mse']:.7f}",
                flush=True,
            )

        current_records = target_records(population_scores, names)
        metrics = aggregate_records(current_records)
        survivor_compatibility_vectors = mating_compatibility_vectors(
            args.compatibility_space,
            args.mode,
            population_z,
            population_coefficients if transitioned else None,
            population_scores,
        )
        graph = graph_diagnostics(
            survivor_compatibility_vectors, args.mating_radius)
        _, survivor_pair_distances = compatibility_graph(
            survivor_compatibility_vectors, float("inf"))
        survivor_pair_distances = survivor_pair_distances[
            np.triu_indices(len(survivor_compatibility_vectors), 1)]
        ecological_graph = graph_diagnostics(
            normalize_species_vectors(population_scores),
            args.ecological_mating_radius,
        )
        fitness_niche_sizes = np.bincount(
            population_goals, minlength=target_count)
        row = {
            "e": spent,
            "generation": generation,
            "transitioned": transitioned,
            "gain": gain,
            "win_rate": win_rate,
            "mean_mse": metrics["mean_mse"],
            "median_mse": metrics["median_mse"],
            "worst_mse": metrics["worst_mse"],
            "sexual_fraction": float(sexual.mean()),
            "fertile_adults": int((reproduction_weights > 0).sum()),
            "senescent_adults": int(senescent.sum()),
            "effective_breeding_adults": float(
                reproduction_weights.sum()),
            "mean_individual_age": float(population_age.mean()),
            "max_individual_age": int(population_age.max()),
            "mean_mate_distance": (
                float(np.nanmean(mate_distances)) if sexual.any() else None),
            "compatibility_space": args.compatibility_space,
            "fitness_species": int((fitness_niche_sizes > 0).sum()),
            "fitness_species_sizes": fitness_niche_sizes.tolist(),
            "compatibility_distance_p10": float(np.percentile(
                survivor_pair_distances, 10)),
            "compatibility_distance_median": float(np.median(
                survivor_pair_distances)),
            "compatibility_distance_p90": float(np.percentile(
                survivor_pair_distances, 90)),
            **{f"ecological_{key}": value
               for key, value in ecological_graph.items()},
            "target_elites_from_children": target_children,
            "expired_parents": expired_count,
            "protected_old_adults": protected_old_adults,
            **lineage_diagnostics,
            **selection_diagnostics,
            "individual_stagnation_kicks": stagnation_kicks,
            "mean_individual_gain": float(population_step_gain.mean()),
            "max_individual_gain": float(population_step_gain.max()),
            "assimilation_fraction": (
                assimilation.get("assimilation_fraction", 0.0)
                if assimilation is not None else 0.0),
            "assimilation_debt": (
                assimilation.get("phenotype_debt_before", 0.0)
                if assimilation is not None else 0.0),
            "retired_parents": (
                assimilation.get("retired_parents", 0)
                if assimilation is not None else 0),
            "legacy_niches": (
                assimilation.get("legacy_niches", 0)
                if assimilation is not None else 0),
            "persistent_decoders": 1 if transitioned else args.survivors,
            **graph,
        }
        trace.append(row)
        if view is not None and generation % 5 == 0:
            view.update(spent, display_hall())
        if spent >= next_report or spent >= args.budget:
            event_text = ""
            if assimilation is not None:
                if assimilation.get("archive_only"):
                    event_text = (
                        f"  archive/{assimilation['retired_parents']}r")
                elif "folded_lora_mean_rms" in assimilation:
                    event_text = (
                        f"  fold {assimilation['folded_lora_mean_rms']:.3f}"
                        + (f"/{assimilation['retired_parents']}r"
                           if "retired_parents" in assimilation else ""))
                else:
                    event_text = (
                        f"  assimilate "
                        f"{100 * assimilation['assimilation_fraction']:.3f}%")
            print(
                f"  {spent:>7} evals  {args.mode:<6}  gain {gain:.2f}  "
                f"mean {metrics['mean_mse']:.5f}  "
                f"worst {metrics['worst_mse']:.5f}  "
                f"decoders {row['persistent_decoders']}  "
                f"components {graph['components']}  "
                f"sexual {100 * sexual.mean():.0f}%"
                + event_text,
                flush=True,
            )
            while next_report <= spent:
                next_report += report_interval

    if view is not None:
        view.update(spent, display_hall())
    records = target_records(population_scores, names)
    final_metrics = aggregate_records(records)
    if transitioned:
        assert model is not None and population_coefficients is not None
        base_phenotypes = decode_conditional(
            model, population_z, np.zeros_like(population_coefficients), device)
        base_records = target_records(score(base_phenotypes), names)
        base_metrics = aggregate_records(base_records)
        coefficient_rms = float(np.sqrt(np.mean(population_coefficients ** 2)))
    else:
        base_records = records.copy()
        base_metrics = final_metrics.copy()
        coefficient_rms = 0.0
    legacy_records: list[dict] = []
    valid_legacy = [
        (target, legacy) for target, legacy in enumerate(legacy_bank)
        if legacy is not None
    ]
    if valid_legacy:
        assert model is not None
        legacy_z = np.stack([
            legacy["z"] for _, legacy in valid_legacy]).astype(np.float32)
        legacy_coefficients = np.stack([
            legacy["coefficients"] for _, legacy in valid_legacy
        ]).astype(np.float32)
        legacy_scores = score(decode_conditional(
            model, legacy_z, legacy_coefficients, device))
        for row_index, (target, legacy) in enumerate(valid_legacy):
            legacy_records.append({
                "target": names[target],
                "target_index": int(target),
                "stored_mse": -float(legacy["role_score"]),
                "current_mse": -float(legacy_scores[row_index, target]),
                "utility": float(legacy["utility"]),
                "retirement_weight": float(legacy["retirement_weight"]),
                "age": int(legacy["age"]),
                "lifetime_wins": int(legacy["lifetime_wins"]),
                "lifetime_attempts": int(legacy["lifetime_attempts"]),
                "retired_at": int(legacy["retired_at"]),
                "z": legacy["z"].tolist(),
                "coefficients": legacy["coefficients"].tolist(),
            })
    legacy_mses = [row["current_mse"] for row in legacy_records]
    legacy_metrics = {
        "filled_targets": len(legacy_records),
        "mean_mse": float(np.mean(legacy_mses)) if legacy_mses else None,
        "worst_mse": float(np.max(legacy_mses)) if legacy_mses else None,
    }
    legacy_lookup = {
        row["target"]: row["current_mse"] for row in legacy_records
    }
    memory_records = {
        name: min(value, legacy_lookup.get(name, float("inf")))
        for name, value in records.items()
    }
    memory_metrics = aggregate_records(memory_records)
    result = {
        "method": "multi_fitness_conditional_decoder",
        "mode": args.mode,
        "coefficient_dim": args.coefficient_dim,
        "latent_conditional_dim": (
            args.coefficient_dim if args.mode == "latent" else
            args.coefficient_dim // 2 if args.mode == "mixed" else 0),
        "lora_conditional_dim": (
            args.coefficient_dim if args.mode == "lora" else
            args.coefficient_dim // 2 if args.mode == "mixed" else 0),
        "targets": [str(path) for path in args.targets],
        "budget": args.budget,
        "transition_at": None if args.start_shared else args.transition_at,
        "start_shared": args.start_shared,
        "dynamic_assimilation": args.dynamic_assimilation,
        "seed": args.seed,
        "torch_version": torch.__version__,
        "survivors": args.survivors,
        "children": args.children,
        "death_policy": args.death_policy,
        "max_age": args.max_age,
        "max_reproductive_age": args.max_reproductive_age,
        "senescent_reproduction_weight": args.senescent_reproduction_weight,
        "senescent_mutation_multiplier": args.senescent_mutation_multiplier,
        "compatibility_space": args.compatibility_space,
        "mating_radius": args.mating_radius,
        "ecological_mating_radius": args.ecological_mating_radius,
        "progress_weight": args.progress_weight,
        "factor_steps": args.factor_steps,
        "factor_base_weight": args.factor_base_weight,
        "consolidate_every": args.consolidate_every,
        "consolidation_steps": args.consolidation_steps,
        "consolidation_shrink": args.consolidation_shrink,
        "consolidation_base_weight": args.consolidation_base_weight,
        "assimilation_learning_rate": args.assimilation_learning_rate,
        "assimilation_method": args.assimilation_method,
        "assimilation_mean_fraction": args.assimilation_mean_fraction,
        "assimilation_steps": args.assimilation_steps,
        "assimilation_max_fraction": args.assimilation_max_fraction,
        "assimilation_debt_scale": args.assimilation_debt_scale,
        "assimilation_base_weight": args.assimilation_base_weight,
        "retirement_merge_fraction": args.retirement_merge_fraction,
        "retirement_temperature": args.retirement_temperature,
        "retirement_relative_weight": args.retirement_relative_weight,
        "retirement_coverage_weight": args.retirement_coverage_weight,
        "retirement_success_weight": args.retirement_success_weight,
        "final_persistent_decoders": 1,
        "shared_decoder_parameters": sum(
            parameter.numel() for parameter in model.parameters()),
        "coefficient_rms": coefficient_rms,
        "records_mse": records,
        "base_only_records_mse": base_records,
        "final_metrics": final_metrics,
        "base_only_metrics": base_metrics,
        "base_only_mean_gap": (
            base_metrics["mean_mse"] - final_metrics["mean_mse"]),
        "factorization_trace": factorization_trace,
        "consolidations": consolidations,
        "assimilations": assimilations,
        "legacy_bank": legacy_records,
        "legacy_bank_metrics": legacy_metrics,
        "memory_records_mse": memory_records,
        "memory_metrics": memory_metrics,
        "trace": trace,
    }
    print("\nFINAL:")
    print(json.dumps({
        "mode": args.mode,
        "coefficient_dim": args.coefficient_dim,
        **final_metrics,
        "base_only_mean_mse": base_metrics["mean_mse"],
        "base_only_worst_mse": base_metrics["worst_mse"],
        "base_only_mean_gap": result["base_only_mean_gap"],
        "coefficient_rms": coefficient_rms,
        "memory_mean_mse": memory_metrics["mean_mse"],
        "memory_worst_mse": memory_metrics["worst_mse"],
    }, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("targets", nargs="+", type=Path)
    parser.add_argument(
        "--mode", choices=("lora", "latent", "mixed"), required=True)
    parser.add_argument("--coefficient-dim", type=int, required=True)
    parser.add_argument("--survivors", type=int, default=48)
    parser.add_argument("--children", type=int, default=192)
    parser.add_argument(
        "--death-policy",
        choices=(
            "global", "protected_age", "lineage_succession", "species_local"),
        default="global",
        help=("global replacement; protected target champions; descendant-"
              "restricted retirement succession; or one-for-one replacement "
              "inside input-space species and fitness roles"),
    )
    parser.add_argument(
        "--compatibility-space",
        choices=("z_only", "decoder_input", "fitness"),
        default="z_only",
        help=("space used only for crossover eligibility: the original z; "
              "every direct decoder input (z plus any extra latent); or the "
              "normalized fitness profile. LoRA gates are never included"),
    )
    parser.add_argument(
        "--mating-radius", type=float,
        help=("maximum RMS compatibility distance (default: 30 for z_only, "
              "50 for decoder_input, 0.3 for fitness)"),
    )
    parser.add_argument(
        "--ecological-mating-radius", type=float, default=0.3,
        help=("maximum normalized fitness-profile distance when choosing "
              "mates inside a decoder-input-defined species"),
    )
    parser.add_argument("--progress-weight", type=float, default=1.0)
    parser.add_argument("--budget", type=int, default=60_000)
    parser.add_argument("--transition-at", type=int, default=20_000)
    parser.add_argument(
        "--start-shared", action="store_true",
        help="use the mixed conditional decoder from the founders onward")
    parser.add_argument(
        "--dynamic-assimilation", action="store_true",
        help=("replace scheduled consolidation with generation-wise exact "
              "LoRA folding or replay steps"))
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--start-gain", type=float, default=1.0)
    parser.add_argument("--factor-steps", type=int, default=1_000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--code-learning-rate", type=float, default=1e-2)
    parser.add_argument("--factor-base-weight", type=float, default=0.1)
    parser.add_argument("--consolidate-every", type=int, default=10_000)
    parser.add_argument("--consolidation-steps", type=int, default=200)
    parser.add_argument("--consolidation-shrink", type=float, default=0.5)
    parser.add_argument("--consolidation-base-weight", type=float, default=1.0)
    parser.add_argument("--assimilation-learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--assimilation-method",
        choices=(
            "mean_fold", "retirement_fold", "retirement_archive", "replay"),
        default="mean_fold")
    parser.add_argument("--assimilation-mean-fraction", type=float, default=1.0)
    parser.add_argument("--assimilation-steps", type=int, default=1)
    parser.add_argument("--assimilation-max-fraction", type=float, default=0.01)
    parser.add_argument("--assimilation-debt-scale", type=float, default=1e-4)
    parser.add_argument("--assimilation-base-weight", type=float, default=1.0)
    parser.add_argument("--retirement-merge-fraction", type=float, default=0.25)
    parser.add_argument("--retirement-temperature", type=float, default=1.0)
    parser.add_argument("--retirement-relative-weight", type=float, default=1.0)
    parser.add_argument("--retirement-coverage-weight", type=float, default=1.0)
    parser.add_argument("--retirement-success-weight", type=float, default=1.0)
    parser.add_argument("--max-age", type=int, default=10)
    parser.add_argument(
        "--max-reproductive-age", type=int,
        help=("age at which reduced reproduction begins without removing "
              "protected population seats"),
    )
    parser.add_argument(
        "--senescent-reproduction-weight", type=float, default=0.0,
        help=("relative parent/mate sampling weight after reproductive age; "
              "zero is complete senescence"),
    )
    parser.add_argument(
        "--senescent-mutation-multiplier", type=float, default=1.0,
        help="mutation multiplier for offspring of a senescent primary parent",
    )
    parser.add_argument("--individual-success-window", type=int, default=20)
    parser.add_argument("--individual-gain-min", type=float, default=0.25)
    parser.add_argument("--individual-gain-max", type=float, default=4.0)
    parser.add_argument("--individual-stagnation-attempts", type=int, default=32)
    parser.add_argument("--individual-stagnation-kick", type=float, default=2.0)
    parser.add_argument("--reports", type=int, default=40)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(args)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
