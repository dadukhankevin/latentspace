"""Evolve one ecosystem toward many real reference images.

This is the target-image counterpart to ``demo_clip_species_vector``.  Every
individual is scored against every reference image using negative pixel MSE,
so its species vector describes which targets it reconstructs unusually well.
There are no prompt-owned islands and no permanent species labels: mating is
allowed only inside the same fixed local radius in normalized species-vector
space, and connected chains provide transitive gene flow.

Unlike the CLIP experiment, this runner has no text model and no negative
prompts.  Its purpose is to isolate the evolutionary question: can one shared
ancestral decoder diversify into recognizable specialists for many concrete
visual targets while useful genes continue to move locally?  With
``--merge-at``, private decoders are temporary teachers: their target-vetted
phenotypes are distilled into one decoder, then every surviving and future
individual uses that exact shared decoder while genomes alone keep evolving.
With ``--reference-sweep-at``, no decoder is constructed: current private
decoders become immutable identities, children reference the locally fitter
parent's identity. When ``--max-reference-age`` adds individual senescence,
each decoder lineage must leave a successor until it transfers its knowledge
through a transactional merger.
With ``--decoder-probe-every``, referenced decoders can also test sparse
mirrored temporary weight perturbations. The temporary models are discarded;
only a fractional nudge toward the fitter side is retained when current target
coverage improves. Probe learning is experimental and disabled by default.
With ``--start-shared``, the founder decoder remains the only decoder from the
first evaluation. Combining it with ``--merge-at`` and
``--distill-init=random`` tests whether a multi-target corpus can make
self-distillation into a fresh representation useful.

Example:

    python3 -m benchmarks.demo_image_species_vector targets/*.png \
      --budget 600000 --live --gif reconstructions.gif --output run.json
"""

from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps

from benchmarks.compare import _require_mps, _seed_everything
from benchmarks.demo_clip_species_vector import (
    choose_compatible_mates,
    graph_diagnostics,
    normalize_species_vectors,
)
from benchmarks.round28_anchor_conv import ConvRGB
from benchmarks.round31_apple_animated import _png
from latentspace.universal.architectures import resolve
from latentspace.universal.explorer import ExplorerConfig, _Template

LATENT = 64
SHAPE = (3, 96, 96)


def pairwise_negative_mse(phenotypes: torch.Tensor,
                          targets: torch.Tensor) -> torch.Tensor:
    """Return ``-MSE`` for every phenotype/target pair without a BxTxD tensor.

    Expanding all pairwise pixel differences would use hundreds of megabytes
    for the normal 192-child, 32-target batch.  The squared-distance identity
    turns it into one matrix multiplication instead.
    """
    pixels = phenotypes.reshape(len(phenotypes), -1).float()
    references = targets.reshape(len(targets), -1).float()
    if pixels.shape[1] != references.shape[1]:
        raise ValueError("phenotypes and targets must have the same size")
    dimension = pixels.shape[1]
    mse = (
        pixels.square().mean(dim=1, keepdim=True)
        + references.square().mean(dim=1)[None]
        - 2.0 * (pixels @ references.T) / dimension
    )
    # Roundoff can make an exact match microscopically negative.
    return -mse.clamp_min(0.0)


def select_target_covered_survivors(
        parent_scores: np.ndarray,
        child_scores: np.ndarray,
        child_priority: np.ndarray,
        child_goals: np.ndarray,
        survivor_count: int,
        target_order: np.ndarray | None = None,
        ) -> tuple[np.ndarray, np.ndarray, int]:
    """Choose one distinct absolute-quality representative per target.

    Target representatives come from parents plus children, so a useful
    lineage cannot disappear merely because all of its mutations were worse
    in one comma-selection generation. Roles are assigned afresh on every
    call; they are ecological seats, not permanent individual/species labels.
    Remaining seats go to children by the caller's rarity/progress priority.

    Returns combined parent+child indices, their one-generation target roles,
    and the number of target seats won by children.
    """
    parents = np.asarray(parent_scores, dtype=np.float64)
    children = np.asarray(child_scores, dtype=np.float64)
    priority = np.asarray(child_priority, dtype=np.float64)
    goals = np.asarray(child_goals, dtype=np.int64)
    if parents.ndim != 2 or children.ndim != 2:
        raise ValueError("parent_scores and child_scores must be matrices")
    if parents.shape[1] != children.shape[1]:
        raise ValueError("parent and child target counts must match")
    if priority.shape != (len(children),) or goals.shape != (len(children),):
        raise ValueError("child priority and goals must match child count")
    target_count = parents.shape[1]
    if survivor_count < target_count:
        raise ValueError("survivors must cover every target")
    if len(children) < survivor_count:
        raise ValueError("child count must be at least survivor count")
    if target_order is None:
        order = np.arange(target_count)
    else:
        order = np.asarray(target_order, dtype=np.int64)
        if sorted(order.tolist()) != list(range(target_count)):
            raise ValueError("target_order must be a target permutation")

    combined = np.concatenate([parents, children], axis=0)
    parent_count = len(parents)
    available = np.ones(len(combined), dtype=bool)
    selected: list[int] = []
    assigned: list[int] = []
    target_children = 0

    # Hard targets can be placed first by the caller. This matters when one
    # generalist is currently best for several targets: each target still gets
    # a distinct representative, with the hardest receiving first choice.
    for target in order:
        values = np.where(available, combined[:, target], -np.inf)
        winner = int(np.argmax(values))
        selected.append(winner)
        assigned.append(int(target))
        available[winner] = False
        target_children += int(winner >= parent_count)

    remaining = survivor_count - target_count
    if remaining:
        child_combined = parent_count + np.arange(len(children))
        eligible = child_combined[available[child_combined]]
        ranked = sorted(
            eligible.tolist(),
            key=lambda index: (-priority[index - parent_count], index),
        )
        if len(ranked) < remaining:
            raise RuntimeError("not enough unselected children to fill survivors")
        for winner in ranked[:remaining]:
            selected.append(winner)
            assigned.append(int(goals[winner - parent_count]))

    return (np.asarray(selected, dtype=np.int64),
            np.asarray(assigned, dtype=np.int64), target_children)


def distill_superdecoder(
        teacher_z: np.ndarray,
        teacher_images: np.ndarray,
        initial_theta: np.ndarray,
        steps: int,
        learning_rate: float,
        code_learning_rate: float,
        device: str,
        report: bool = True,
        ) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Combine private decoder functions into one decoder and new codes.

    This is supervised only by phenotypes already discovered and vetted by
    evolution. The real target images are never used as training labels here.
    Codes are allowed to move because two private decoders may have attached
    different meanings to nearby genomes; the shared decoder needs one
    internally consistent codebook.
    """
    if steps < 1:
        raise ValueError("distillation steps must be positive")
    if learning_rate <= 0 or code_learning_rate <= 0:
        raise ValueError("distillation learning rates must be positive")
    teacher_z = np.asarray(teacher_z, dtype=np.float32)
    teacher_images = np.asarray(teacher_images, dtype=np.float32)
    if teacher_z.ndim != 2 or teacher_z.shape[1] != LATENT:
        raise ValueError(f"teacher_z must have shape (N, {LATENT})")
    if teacher_images.shape != (len(teacher_z), *SHAPE):
        raise ValueError(f"teacher_images must have shape (N, {SHAPE})")

    net = ConvRGB(LATENT, SHAPE).to(device)
    torch.nn.utils.vector_to_parameters(
        torch.as_tensor(initial_theta, device=device), net.parameters())
    for parameter in net.parameters():
        parameter.requires_grad_(True)
    codes = torch.nn.Parameter(torch.as_tensor(teacher_z, device=device))
    targets = torch.as_tensor(teacher_images, device=device)
    optimizer = torch.optim.Adam([
        {"params": net.parameters(), "lr": learning_rate},
        {"params": [codes], "lr": code_learning_rate},
    ])
    trace: list[dict] = []
    report_every = max(1, steps // 10)
    net.train(True)
    for step in range(1, steps + 1):
        predicted = torch.sigmoid(net(codes)).reshape(len(codes), *SHAPE)
        reconstruction = (predicted - targets).square().mean()
        # A tiny code prior prevents gratuitous scale growth while allowing
        # meanings to move enough to resolve cross-decoder code conflicts.
        loss = reconstruction + 1e-6 * codes.square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == 1 or step % report_every == 0 or step == steps:
            row = {
                "step": step,
                "phenotype_mse": float(reconstruction.detach().cpu()),
                "code_rms": float(codes.detach().square().mean().sqrt().cpu()),
            }
            trace.append(row)
            if report:
                print(
                    f"    distill {step:>5}/{steps}  "
                    f"phenotype mse {row['phenotype_mse']:.7f}  "
                    f"code rms {row['code_rms']:.3f}",
                    flush=True,
                )

    theta = torch.nn.utils.parameters_to_vector(
        net.parameters()).detach().cpu().numpy().astype(np.float32)
    learned_codes = codes.detach().cpu().numpy().astype(np.float32)
    return theta, learned_codes, trace


def shared_population_from_codes(
        codes: np.ndarray,
        theta: np.ndarray,
        survivor_count: int,
        rng: np.random.Generator,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a covered population whose individuals share one exact decoder."""
    codes = np.asarray(codes, dtype=np.float32)
    target_count = len(codes)
    if survivor_count < target_count:
        raise ValueError("survivor count must cover all distilled codes")
    extras = survivor_count - target_count
    if extras:
        sources = np.arange(extras) % target_count
        extra_codes = codes[sources] + rng.normal(
            0, 0.01, (extras, codes.shape[1])).astype(np.float32)
        population_z = np.concatenate([codes, extra_codes], axis=0)
        goals = np.concatenate(
            [np.arange(target_count), sources]).astype(np.int64)
    else:
        population_z = codes.copy()
        goals = np.arange(target_count, dtype=np.int64)
    population_theta = np.repeat(
        np.asarray(theta, dtype=np.float32)[None], survivor_count, axis=0)
    return population_z, population_theta, goals


def inherit_decoder_references(
        parents: np.ndarray,
        mates: np.ndarray,
        inherited_goals: np.ndarray,
        population_scores: np.ndarray,
        population_decoder_ids: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray]:
    """Inherit the locally fitter parent's immutable decoder identity."""
    parents = np.asarray(parents, dtype=np.int64)
    mates = np.asarray(mates, dtype=np.int64)
    goals = np.asarray(inherited_goals, dtype=np.int64)
    decoder_ids = np.asarray(population_decoder_ids, dtype=np.int64)
    if parents.shape != mates.shape or parents.shape != goals.shape:
        raise ValueError("parents, mates, and inherited_goals must align")
    donors = parents.copy()
    sexual = mates >= 0
    rows = np.flatnonzero(sexual)
    if len(rows):
        base_quality = population_scores[parents[rows], goals[rows]]
        mate_quality = population_scores[mates[rows], goals[rows]]
        mate_wins = mate_quality > base_quality
        donors[rows[mate_wins]] = mates[rows[mate_wins]]
    inherited = decoder_ids[donors]
    from_mate = sexual & (donors == mates)
    return inherited, from_mate


def preserve_decoder_lineages(
        keep: np.ndarray,
        roles: np.ndarray,
        parent_scores: np.ndarray,
        child_scores: np.ndarray,
        child_priority: np.ndarray,
        parent_goals: np.ndarray,
        child_goals: np.ndarray,
        parent_decoder_ids: np.ndarray,
        child_decoder_ids: np.ndarray,
        target_count: int,
        ) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Ensure survivor selection cannot erase an unmerged decoder lineage.

    A missing lineage first takes a low-priority non-target seat with its best
    child. If every non-target seat is carrying a unique decoder, it replaces
    the target representative for which that child loses the least quality.
    When no child inherited the decoder, its best current carrier receives a
    one-generation reprieve. This separates individual senescence from
    decoder extinction: only an explicit merger may reduce decoder count.

    Returns the adjusted survivor indices and roles, followed by counts of
    child successors and reprieved current carriers.
    """
    keep = np.asarray(keep, dtype=np.int64).copy()
    roles = np.asarray(roles, dtype=np.int64).copy()
    parents = np.asarray(parent_scores, dtype=np.float64)
    children = np.asarray(child_scores, dtype=np.float64)
    priority = np.asarray(child_priority, dtype=np.float64)
    parent_goals = np.asarray(parent_goals, dtype=np.int64)
    child_goals = np.asarray(child_goals, dtype=np.int64)
    parent_ids = np.asarray(parent_decoder_ids, dtype=np.int64)
    child_ids = np.asarray(child_decoder_ids, dtype=np.int64)
    parent_count = len(parents)
    combined_scores = np.concatenate([parents, children], axis=0)
    combined_ids = np.concatenate([parent_ids, child_ids])

    if keep.shape != roles.shape:
        raise ValueError("keep and roles must align")
    if len(keep) < len(np.unique(parent_ids)):
        raise ValueError("survivors cannot preserve every decoder lineage")
    if roles.shape[0] < target_count:
        raise ValueError("target_count exceeds survivor roles")
    if priority.shape != (len(children),):
        raise ValueError("child priority must match child count")

    selected = set(keep.tolist())
    counts = {
        int(decoder_id): int(np.count_nonzero(combined_ids[keep] == decoder_id))
        for decoder_id in np.unique(parent_ids)
    }
    missing = sorted(
        int(decoder_id) for decoder_id in np.unique(parent_ids)
        if counts[int(decoder_id)] == 0
    )
    successors = 0
    reprieved = 0

    for decoder_id in missing:
        available = np.flatnonzero(combined_ids == decoder_id)
        available = np.asarray(
            [index for index in available if int(index) not in selected],
            dtype=np.int64,
        )
        child_candidates = available[available >= parent_count]
        candidates = child_candidates if len(child_candidates) else available
        if not len(candidates):
            raise RuntimeError("decoder lineage has no successor candidate")

        removable = [
            position for position, index in enumerate(keep)
            if counts[int(combined_ids[index])] > 1
        ]
        if not removable:
            raise RuntimeError("no duplicate decoder seat available")

        fill_positions = [
            position for position in removable if position >= target_count
        ]
        if fill_positions:
            # Fill seats are already sorted by descending child priority.
            # Replace the last duplicate-bearing seat, and choose the most
            # promising inheriting child for the new lineage.
            position = fill_positions[-1]
            candidate = int(max(
                candidates.tolist(),
                key=lambda index: (
                    priority[index - parent_count]
                    if index >= parent_count
                    else combined_scores[index, parent_goals[index]],
                    -index,
                ),
            ))
        else:
            # Every removable seat currently represents a target. Choose the
            # candidate/seat pairing with the smallest immediate quality loss.
            position, candidate = max(
                (
                    (position, int(candidate))
                    for position in removable
                    for candidate in candidates
                ),
                key=lambda item: (
                    combined_scores[item[1], roles[item[0]]]
                    - combined_scores[keep[item[0]], roles[item[0]]],
                    -item[0],
                    -item[1],
                ),
            )

        removed = int(keep[position])
        removed_id = int(combined_ids[removed])
        selected.remove(removed)
        selected.add(candidate)
        counts[removed_id] -= 1
        counts[decoder_id] += 1
        keep[position] = candidate
        if position >= target_count:
            roles[position] = int(
                child_goals[candidate - parent_count]
                if candidate >= parent_count else parent_goals[candidate]
            )
        successors += int(candidate >= parent_count)
        reprieved += int(candidate < parent_count)

    if set(np.unique(parent_ids).tolist()) - set(combined_ids[keep].tolist()):
        raise RuntimeError("failed to preserve every decoder lineage")
    return keep, roles, successors, reprieved


def update_individual_step_state(
        gains: np.ndarray,
        success_wins: np.ndarray,
        success_attempts: np.ndarray,
        stagnation_attempts: np.ndarray,
        parents: np.ndarray,
        child_wins: np.ndarray,
        mode: str,
        win_target: float,
        gain_step: float,
        success_window: int,
        stagnation_limit: int,
        stagnation_kick: float,
        gain_limits: tuple[float, float],
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Update inherited mutation multipliers from each parent's offspring.

    ``success`` applies the same one-fifth rule as the global controller, but
    only after one individual has accumulated a sufficiently large window of
    reproductive evidence. ``stagnation`` is deliberately separate: a parent
    whose recent children never improve receives a temporary exploration
    kick. The distinction matters because low success normally tells the
    one-fifth controller to shrink an over-large step, while prolonged lack of
    progress can still justify an occasional escape attempt.
    """
    valid_modes = {"off", "success", "stagnation", "hybrid"}
    if mode not in valid_modes:
        raise ValueError(f"unknown individual step-control mode: {mode}")
    gains = np.asarray(gains, dtype=np.float64).copy()
    success_wins = np.asarray(success_wins, dtype=np.int64).copy()
    success_attempts = np.asarray(success_attempts, dtype=np.int64).copy()
    stagnation_attempts = np.asarray(
        stagnation_attempts, dtype=np.int64).copy()
    parents = np.asarray(parents, dtype=np.int64)
    child_wins = np.asarray(child_wins, dtype=bool)
    size = len(gains)
    if not all(array.shape == (size,) for array in (
            success_wins, success_attempts, stagnation_attempts)):
        raise ValueError("individual step-state arrays must align")
    if parents.shape != child_wins.shape:
        raise ValueError("parents and child wins must align")
    if len(parents) and (parents.min() < 0 or parents.max() >= size):
        raise ValueError("parent index is outside the population")
    if success_window < 1 or stagnation_limit < 1:
        raise ValueError("individual adaptation windows must be positive")
    if not 0 <= win_target <= 1 or gain_step <= 1:
        raise ValueError("invalid one-fifth controller parameters")
    if stagnation_kick <= 1:
        raise ValueError("stagnation kick must exceed one")
    low, high = gain_limits
    if low <= 0 or high < low:
        raise ValueError("invalid individual gain limits")
    if mode == "off" or not len(parents):
        return (gains, success_wins, success_attempts,
                stagnation_attempts, 0, 0)

    attempts = np.bincount(parents, minlength=size).astype(np.int64)
    wins = np.bincount(
        parents, weights=child_wins.astype(np.int64), minlength=size,
    ).astype(np.int64)
    active = attempts > 0
    success_updates = 0
    kicks = 0

    if mode in {"success", "hybrid"}:
        success_wins += wins
        success_attempts += attempts
        ready = success_attempts >= success_window
        if ready.any():
            rates = success_wins[ready] / success_attempts[ready]
            gains[ready] *= np.where(
                rates > win_target, gain_step, 1.0 / gain_step)
            success_updates = int(ready.sum())
            success_wins[ready] = 0
            success_attempts[ready] = 0

    if mode in {"stagnation", "hybrid"}:
        improving = active & (wins > 0)
        stalled = active & ~improving
        stagnation_attempts[improving] = 0
        stagnation_attempts[stalled] += attempts[stalled]
        kick = stagnation_attempts >= stagnation_limit
        if kick.any():
            gains[kick] *= stagnation_kick
            stagnation_attempts[kick] = 0
            kicks = int(kick.sum())

    gains = np.clip(gains, low, high)
    return (gains, success_wins, success_attempts,
            stagnation_attempts, success_updates, kicks)


def sparse_decoder_perturbation(
        theta: np.ndarray,
        fraction: float,
        sigma: float,
        rng: np.random.Generator,
        ) -> np.ndarray:
    """Sample a sparse mirrored-probe direction scaled to decoder weights."""
    theta = np.asarray(theta, dtype=np.float32)
    if theta.ndim != 1 or not len(theta):
        raise ValueError("decoder parameters must be a non-empty vector")
    if not 0 < fraction <= 1:
        raise ValueError("decoder probe fraction must be in (0, 1]")
    if sigma <= 0:
        raise ValueError("decoder probe sigma must be positive")
    mask = rng.random(theta.shape) < fraction
    if not mask.any():
        mask[int(rng.integers(0, len(theta)))] = True
    scale = max(float(theta.std()), 1e-3)
    perturbation = np.zeros_like(theta)
    perturbation[mask] = (
        rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), mask.sum())
        * sigma * scale
    )
    return perturbation


def mirrored_decoder_candidate(
        theta: np.ndarray,
        perturbation: np.ndarray,
        plus_fitness: float,
        minus_fitness: float,
        step_fraction: float,
        ) -> np.ndarray:
    """Nudge decoder weights toward the fitter side of a mirrored probe.

    The temporary ``theta +/- perturbation`` models are discarded.  Only a
    fractional step along their measured directional derivative is returned.
    This is a sign-normalized zeroth-order gradient step rather than ordinary
    offspring selection.
    """
    theta = np.asarray(theta, dtype=np.float32)
    perturbation = np.asarray(perturbation, dtype=np.float32)
    if theta.shape != perturbation.shape:
        raise ValueError("decoder parameters and perturbation must align")
    if not np.isfinite(plus_fitness) or not np.isfinite(minus_fitness):
        raise ValueError("decoder probe fitness must be finite")
    if not 0 < step_fraction <= 1:
        raise ValueError("decoder probe step fraction must be in (0, 1]")
    direction = float(np.sign(plus_fitness - minus_fitness))
    return (theta + direction * step_fraction * perturbation).astype(
        np.float32)


def assigned_role_fitness(
        scores: np.ndarray,
        roles: np.ndarray,
        ) -> tuple[float, float]:
    """Return mean and worst fitness on each carrier's ecological role."""
    scores = np.asarray(scores, dtype=np.float64)
    roles = np.asarray(roles, dtype=np.int64)
    if scores.ndim != 2 or roles.shape != (len(scores),):
        raise ValueError("decoder scores and carrier roles must align")
    if len(scores) == 0:
        raise ValueError("decoder probe requires at least one carrier")
    if roles.min() < 0 or roles.max() >= scores.shape[1]:
        raise ValueError("carrier role is outside the score matrix")
    values = scores[np.arange(len(scores)), roles]
    return float(values.mean()), float(values.min())


def target_coverage_fitness(scores: np.ndarray) -> tuple[float, float]:
    """Return mean and worst best-available target fitness."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2 or not len(scores) or scores.shape[1] == 0:
        raise ValueError("target coverage requires a non-empty score matrix")
    best = scores.max(axis=0)
    return float(best.mean()), float(best.min())


def local_behavior_density(
        vectors: np.ndarray,
        radius: float,
        ) -> np.ndarray:
    """Count each individual's neighbors inside the fixed mating radius."""
    vectors = np.asarray(vectors, dtype=np.float64)
    if vectors.ndim != 2 or not len(vectors):
        raise ValueError("behavior vectors must be a non-empty matrix")
    if radius < 0:
        raise ValueError("behavior radius must be non-negative")
    distances = np.sqrt(np.mean(
        (vectors[:, None, :] - vectors[None, :, :]) ** 2, axis=2))
    return np.count_nonzero(distances <= radius, axis=1).astype(np.int64)


def update_decoder_stagnation(
        scores: np.ndarray,
        goals: np.ndarray,
        decoder_ids: np.ndarray,
        best_fitness: dict[int, float],
        stagnation: dict[int, int],
        min_improvement: float = 1e-8,
        ) -> tuple[dict[int, float], dict[int, int]]:
    """Update NEAT-style stagnation ages for active decoder lineages."""
    scores = np.asarray(scores, dtype=np.float64)
    goals = np.asarray(goals, dtype=np.int64)
    decoder_ids = np.asarray(decoder_ids, dtype=np.int64)
    if scores.ndim != 2 or goals.shape != (len(scores),):
        raise ValueError("decoder scores and roles must align")
    if decoder_ids.shape != (len(scores),):
        raise ValueError("decoder IDs must align with scores")
    if min_improvement < 0:
        raise ValueError("minimum improvement must be non-negative")
    active = set(decoder_ids.astype(int).tolist())
    updated_best = {
        int(key): float(value) for key, value in best_fitness.items()
        if int(key) in active
    }
    updated_stagnation = {
        int(key): int(value) for key, value in stagnation.items()
        if int(key) in active
    }
    for decoder_id in sorted(active):
        carriers = np.flatnonzero(decoder_ids == decoder_id)
        current = float(np.mean(scores[
            carriers, goals[carriers]]))
        previous = updated_best.get(decoder_id, -np.inf)
        if current > previous + min_improvement:
            updated_best[decoder_id] = current
            updated_stagnation[decoder_id] = 0
        else:
            updated_best.setdefault(decoder_id, current)
            updated_stagnation[decoder_id] = (
                updated_stagnation.get(decoder_id, 0) + 1)
    return updated_best, updated_stagnation


def most_encountered_decoder_pair(
        parents: np.ndarray,
        mates: np.ndarray,
        population_decoder_ids: np.ndarray,
        active_decoder_ids: set[int],
        decoder_stagnation: dict[int, int] | None = None,
        stagnation_weight: float = 0.0,
        stagnation_grace: int = 0,
        ) -> tuple[tuple[int, int] | None, int]:
    """Choose an active local pair by encounters plus lineage stagnation."""
    if stagnation_weight < 0:
        raise ValueError("decoder stagnation weight must be non-negative")
    if stagnation_grace < 0:
        raise ValueError("decoder stagnation grace must be non-negative")
    decoder_stagnation = decoder_stagnation or {}
    counts: dict[tuple[int, int], int] = {}
    for parent, mate in zip(parents, mates):
        if mate < 0:
            continue
        a = int(population_decoder_ids[parent])
        b = int(population_decoder_ids[mate])
        if a == b or a not in active_decoder_ids or b not in active_decoder_ids:
            continue
        pair = (a, b) if a < b else (b, a)
        counts[pair] = counts.get(pair, 0) + 1
    if not counts:
        return None, 0
    pair = min(counts, key=lambda item: (
        -(
            counts[item]
            + stagnation_weight * (
                max(0, decoder_stagnation.get(item[0], 0)
                    - stagnation_grace)
                + max(0, decoder_stagnation.get(item[1], 0)
                      - stagnation_grace)
            )
        ),
        -counts[item],
        item,
    ))
    return pair, counts[pair]


def load_targets(paths: list[Path]) -> tuple[list[str], np.ndarray]:
    """Center-crop real images to the decoder's 96x96 RGB output shape."""
    if len(paths) < 2:
        raise ValueError("at least two target images are required")
    names: list[str] = []
    arrays: list[np.ndarray] = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as source:
            image = ImageOps.fit(
                source.convert("RGB"), (SHAPE[2], SHAPE[1]),
                method=Image.Resampling.LANCZOS,
            )
        array = np.asarray(image, dtype=np.float32) / 255.0
        arrays.append(array.transpose(2, 0, 1))
        names.append(path.stem)
    if len(set(names)) != len(names):
        raise ValueError("target filenames must have unique stems")
    return names, np.stack(arrays)


class ReferenceSpeciesView:
    """Live target|reconstruction grid plus per-target MSE histories."""

    def __init__(self, names: list[str], targets: np.ndarray, budget: int):
        import matplotlib
        try:
            matplotlib.use("MacOSX")
        except Exception:
            matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        from matplotlib import colormaps

        self.plt = plt
        plt.ion()
        self.names = names
        self.targets = targets.transpose(0, 2, 3, 1)
        count = len(names)
        cols = min(8, count)
        rows = int(np.ceil(count / cols))
        self.fig = plt.figure(figsize=(2.0 * cols, 2.15 * rows + 4.0))
        grid = (rows + 2, cols)
        self.ims, self.axes = [], []
        for i, name in enumerate(names):
            ax = plt.subplot2grid(grid, (i // cols, i % cols), fig=self.fig)
            pair = np.concatenate(
                [self.targets[i], np.zeros_like(self.targets[i])], axis=1)
            self.ims.append(ax.imshow(pair))
            ax.set_title(f"{name}\ntarget | evolved", fontsize=7)
            ax.axis("off")
            self.axes.append(ax)

        self.ax = plt.subplot2grid(
            grid, (rows, 0), colspan=cols, rowspan=2, fig=self.fig)
        cmap = colormaps["tab20"]
        self.lines = []
        for i, name in enumerate(names):
            (line,) = self.ax.plot(
                [], [], lw=1.2, color=cmap(i % 20), label=name)
            self.lines.append(line)
        self.ax.set_xlim(0, budget)
        self.ax.set_xlabel("evaluations", fontsize=9)
        self.ax.set_ylabel("best reconstruction MSE (lower is better)",
                           fontsize=9)
        self.ax.legend(fontsize=5, loc="upper right", ncols=4)
        self.ax.grid(alpha=0.25)
        self.es: list[int] = []
        self.hist = [[] for _ in names]
        self.fig.tight_layout()
        plt.show(block=False)

    def update(self, evaluations: int, hall: list[dict]) -> None:
        self.es.append(evaluations)
        for i, entry in enumerate(hall):
            reconstruction = entry["image"].transpose(1, 2, 0)
            self.ims[i].set_data(np.concatenate(
                [self.targets[i], reconstruction], axis=1))
            mse = -float(entry["score"])
            self.axes[i].set_title(
                f"{self.names[i]}  mse {mse:.4f}\ntarget | evolved",
                fontsize=7,
            )
            self.hist[i].append(mse)
            self.lines[i].set_data(self.es, self.hist[i])
        values = np.asarray(self.hist, dtype=np.float64)
        if values.size:
            lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
            pad = max((hi - lo) * 0.08, 1e-4)
            self.ax.set_ylim(max(0.0, lo - pad), hi + pad)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        self.plt.pause(0.001)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("targets", nargs="+", type=Path,
                        help="two or more PNG/JPEG reference images")
    parser.add_argument("--survivors", type=int, default=48)
    parser.add_argument("--children", type=int, default=192)
    parser.add_argument("--mating-radius", type=float, default=0.3,
                        help="maximum RMS normalized species-vector distance")
    parser.add_argument("--budget", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start-gain", type=float, default=1.0)
    parser.add_argument(
        "--start-shared", action="store_true",
        help="keep one immutable shared decoder from the first evaluation",
    )
    parser.add_argument(
        "--progress-weight", type=float, default=1.0,
        help="weight of normalized parent-target improvement in fill seats",
    )
    decoder_transition = parser.add_mutually_exclusive_group()
    decoder_transition.add_argument(
        "--merge-at", type=int,
        help=("evaluation count at which private decoders are distilled and "
              "permanently collapsed into one shared decoder"),
    )
    decoder_transition.add_argument(
        "--reference-sweep-at", type=int,
        help=("evaluation count at which current private decoders become "
              "immutable shared references inherited from the locally "
              "fitter parent"),
    )
    parser.add_argument(
        "--transactional-merges", action="store_true",
        help=("allow behaviorally compatible referenced decoder lineages to "
              "attempt fitness-gated phenotype distillation mergers"),
    )
    parser.add_argument("--local-merge-steps", type=int, default=200)
    parser.add_argument("--local-merge-tolerance", type=float, default=2e-4)
    parser.add_argument(
        "--local-merge-policy", choices=("gated", "always"),
        default="gated",
        help="reject harmful local mergers or always commit them after scoring",
    )
    parser.add_argument(
        "--fitness-sharing-weight", type=float, default=0.0,
        help=("NEAT-style local-density penalty on non-target fill-seat "
              "selection; zero disables sharing"),
    )
    parser.add_argument(
        "--merge-stagnation-weight", type=float, default=0.0,
        help=("priority bonus per stagnant generation when choosing among "
              "locally mating decoder pairs"),
    )
    parser.add_argument(
        "--merge-stagnation-grace", type=int, default=0,
        help=("generations without lineage improvement before stagnation "
              "can influence merger priority"),
    )
    parser.add_argument(
        "--max-reference-age", type=int,
        help=("maximum survivor age in generations after reference inheritance; "
              "expired parents cannot retain target seats, while decoder "
              "lineages receive successors until explicitly merged"),
    )
    parser.add_argument(
        "--individual-step-control",
        choices=("off", "success", "stagnation", "hybrid"),
        default="off",
        help=("inherited per-individual mutation multiplier controlled by "
              "offspring success, personal stagnation, both, or neither"),
    )
    parser.add_argument(
        "--individual-step-start", choices=("founder", "reference"),
        default="founder",
        help=("activate individual mutation control immediately or only "
              "after the decoder reference transition"),
    )
    parser.add_argument("--individual-success-window", type=int, default=20)
    parser.add_argument("--individual-gain-min", type=float, default=0.25)
    parser.add_argument("--individual-gain-max", type=float, default=4.0)
    parser.add_argument(
        "--individual-stagnation-attempts", type=int, default=32)
    parser.add_argument("--individual-stagnation-kick", type=float, default=3.0)
    parser.add_argument(
        "--no-global-step-control", action="store_true",
        help="hold the global gain fixed and rely on individual control",
    )
    parser.add_argument(
        "--decoder-probe-every", type=int, default=0,
        help=("evaluation interval for mirrored temporary weight probes on "
              "referenced decoders; zero disables probe learning"),
    )
    parser.add_argument(
        "--decoder-probe-delay", type=int, default=0,
        help="evaluations to wait after the reference sweep before probing",
    )
    parser.add_argument(
        "--decoder-probe-min-carriers", type=int, default=2,
        help="minimum shared carriers required to update one decoder",
    )
    parser.add_argument(
        "--decoder-probe-objective", choices=("coverage", "roles"),
        default="coverage",
        help=("measure temporary probes by ecosystem-wide target coverage or "
              "only by each carrier's assigned ecological role"),
    )
    parser.add_argument(
        "--decoder-probe-fraction", type=float, default=0.01,
        help="fraction of decoder weights perturbed in each mirrored probe",
    )
    parser.add_argument(
        "--decoder-probe-sigma", type=float, default=0.02,
        help="probe noise as a fraction of the decoder parameter RMS scale",
    )
    parser.add_argument(
        "--decoder-probe-step-fraction", type=float, default=0.25,
        help="fraction of the fitter temporary perturbation to retain",
    )
    parser.add_argument(
        "--decoder-probe-worst-tolerance", type=float, default=0.0,
        help=("maximum allowed decrease in worst assigned-role fitness when "
              "committing a probe nudge"),
    )
    parser.add_argument("--distill-steps", type=int, default=3000)
    parser.add_argument("--distill-lr", type=float, default=1e-3)
    parser.add_argument("--distill-code-lr", type=float, default=1e-2)
    parser.add_argument(
        "--distill-init", choices=("medoid", "random"), default="medoid",
        help="initialize the distilled decoder from a teacher or from scratch",
    )
    parser.add_argument("--frames", type=int, default=80)
    parser.add_argument("--live", action="store_true")
    parser.add_argument(
        "--hold-open", action="store_true",
        help="keep the live progress window open after the run finishes",
    )
    parser.add_argument("--gif", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.survivors < 2:
        raise ValueError("--survivors must be at least 2")
    if args.children < args.survivors:
        raise ValueError("--children must be at least --survivors")
    if args.budget < args.survivors:
        raise ValueError("--budget must cover the founder population")
    if args.merge_at is not None:
        if args.merge_at < args.survivors:
            raise ValueError("--merge-at must follow the founder population")
        if args.merge_at + args.survivors > args.budget:
            raise ValueError(
                "--budget must leave one survivor-population evaluation "
                "for the decoder merge")
    if args.reference_sweep_at is not None:
        if args.reference_sweep_at < args.survivors:
            raise ValueError(
                "--reference-sweep-at must follow the founder population")
        if args.reference_sweep_at >= args.budget:
            raise ValueError("--reference-sweep-at must precede --budget")
    if args.start_shared and args.reference_sweep_at is not None:
        raise ValueError(
            "--start-shared already has one decoder; it cannot begin a "
            "private-decoder reference sweep")
    if args.transactional_merges and args.reference_sweep_at is None:
        raise ValueError(
            "--transactional-merges requires --reference-sweep-at")
    if args.local_merge_steps < 1:
        raise ValueError("--local-merge-steps must be positive")
    if args.local_merge_tolerance < 0:
        raise ValueError("--local-merge-tolerance must be non-negative")
    if args.fitness_sharing_weight < 0:
        raise ValueError("--fitness-sharing-weight must be non-negative")
    if args.merge_stagnation_weight < 0:
        raise ValueError("--merge-stagnation-weight must be non-negative")
    if args.merge_stagnation_grace < 0:
        raise ValueError("--merge-stagnation-grace must be non-negative")
    if args.merge_stagnation_weight and not args.transactional_merges:
        raise ValueError(
            "--merge-stagnation-weight requires --transactional-merges")
    if args.max_reference_age is not None:
        if args.reference_sweep_at is None:
            raise ValueError(
                "--max-reference-age requires --reference-sweep-at")
        if args.max_reference_age < 1:
            raise ValueError("--max-reference-age must be positive")
    if args.individual_success_window < 1:
        raise ValueError("--individual-success-window must be positive")
    if not (0 < args.individual_gain_min <= 1
            <= args.individual_gain_max):
        raise ValueError(
            "individual gain limits must be positive and contain one")
    if args.individual_stagnation_attempts < 1:
        raise ValueError(
            "--individual-stagnation-attempts must be positive")
    if args.individual_stagnation_kick <= 1:
        raise ValueError("--individual-stagnation-kick must exceed one")
    if (args.individual_step_control != "off"
            and args.individual_step_start == "reference"
            and args.reference_sweep_at is None):
        raise ValueError(
            "reference-start individual control requires --reference-sweep-at")
    if args.decoder_probe_every < 0 or args.decoder_probe_delay < 0:
        raise ValueError("decoder probe timing must be non-negative")
    if args.decoder_probe_every and args.reference_sweep_at is None:
        raise ValueError("decoder probes require --reference-sweep-at")
    if args.decoder_probe_min_carriers < 1:
        raise ValueError("--decoder-probe-min-carriers must be positive")
    if not 0 < args.decoder_probe_fraction <= 1:
        raise ValueError("--decoder-probe-fraction must be in (0, 1]")
    if args.decoder_probe_sigma <= 0:
        raise ValueError("--decoder-probe-sigma must be positive")
    if not 0 < args.decoder_probe_step_fraction <= 1:
        raise ValueError("--decoder-probe-step-fraction must be in (0, 1]")
    if args.decoder_probe_worst_tolerance < 0:
        raise ValueError(
            "--decoder-probe-worst-tolerance must be non-negative")

    names, target_arrays = load_targets(args.targets)
    target_count = len(names)
    if args.survivors < target_count:
        raise ValueError("--survivors must be at least the number of targets")
    if args.progress_weight < 0:
        raise ValueError("--progress-weight must be non-negative")
    _require_mps()
    device = "mps"
    targets = torch.as_tensor(target_arrays, device=device)
    config = ExplorerConfig()

    def score(phenotypes: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            values = pairwise_negative_mse(
                phenotypes.reshape(len(phenotypes), *SHAPE), targets)
        return values.cpu().numpy().astype(np.float32)

    _seed_everything(args.seed)
    rng = np.random.default_rng(args.seed)
    # Probe sampling must not alter the ordinary reproduction/mating random
    # stream, so paired runs remain identical until a learned decoder changes
    # their fitness landscape.
    probe_rng = np.random.default_rng(args.seed ^ 0x5EEDC0DE)
    template = _Template(resolve(lambda latent, shape: ConvRGB(latent, shape),
                                 LATENT, SHAPE), device)

    # One ancestral decoder and many genomes, matching the CLIP experiment.
    founder_theta = template.init_theta(int(rng.integers(0, 2**31)))
    population_z = rng.standard_normal(
        (args.survivors, LATENT)).astype(np.float32)
    population_theta = np.repeat(
        founder_theta[None], args.survivors, axis=0)
    founder_phenotypes = template.decode_batch(population_theta, population_z)
    population_phenotypes = founder_phenotypes
    population_scores = score(founder_phenotypes)
    population_goals = normalize_species_vectors(
        population_scores).argmax(axis=1)
    population_age = np.zeros(args.survivors, dtype=np.int64)
    population_step_gain = np.ones(args.survivors, dtype=np.float64)
    population_success_wins = np.zeros(args.survivors, dtype=np.int64)
    population_success_attempts = np.zeros(args.survivors, dtype=np.int64)
    population_stagnation_attempts = np.zeros(
        args.survivors, dtype=np.int64)
    spent = args.survivors

    hall = [
        {"score": -np.inf, "image": None, "z": None, "theta": None}
        for _ in names
    ]

    def update_hall(phenotypes: torch.Tensor,
                    scores: np.ndarray,
                    genomes: np.ndarray,
                    decoders: np.ndarray) -> bool:
        improved = False
        for target in range(target_count):
            best = int(np.argmax(scores[:, target]))
            value = float(scores[best, target])
            if value > hall[target]["score"] + 1e-12:
                hall[target] = {
                    "score": value,
                    "image": phenotypes[best].detach().cpu().numpy()
                    .reshape(*SHAPE),
                    "z": genomes[best].copy(),
                    "theta": decoders[best].copy(),
                }
                improved = True
        return improved

    update_hall(
        founder_phenotypes, population_scores, population_z, population_theta)
    gain = float(args.start_gain)
    generation = 0
    global_stall = 0
    shared_decoder = bool(args.start_shared)
    shared_theta: np.ndarray | None = (
        founder_theta.copy() if args.start_shared else None)
    merge_info: dict | None = None
    reference_sweep = False
    population_decoder_ids: np.ndarray | None = None
    decoder_bank: dict[int, np.ndarray] = {}
    decoder_best_fitness: dict[int, float] = {}
    decoder_stagnation: dict[int, int] = {}
    next_decoder_id = 0
    reference_sweep_info: dict | None = None
    cumulative_individual_step_updates = 0
    cumulative_individual_stagnation_kicks = 0
    cumulative_decoder_probe_attempts = 0
    cumulative_decoder_probe_accepts = 0
    next_decoder_probe_at: int | None = None
    frames: list[dict] = []
    trace: list[dict] = []
    frame_interval = max(1, args.budget // max(args.frames, 1))
    next_frame = 0
    view = (ReferenceSpeciesView(names, target_arrays, args.budget)
            if args.live else None)
    if view is not None:
        view.update(spent, hall)

    def mutate_genome(genome: np.ndarray,
                      step_multiplier: float = 1.0) -> np.ndarray:
        mask = rng.random(genome.shape) < config.genome_mutation_rate
        if not mask.any():
            mask[rng.integers(0, len(genome))] = True
        return (genome + mask * rng.normal(
            0, config.genome_mutation_sigma * gain * step_multiplier,
            genome.shape)
                ).astype(np.float32)

    def crossover(base: np.ndarray, donor: np.ndarray) -> np.ndarray:
        cut = int(rng.integers(1, LATENT))
        child = base.copy()
        child[cut:] = donor[cut:]
        return child.astype(np.float32)

    def hall_mses() -> list[float]:
        return [-float(entry["score"]) for entry in hall]

    def capture(row: dict) -> None:
        frame = {**row, "mses": hall_mses()}
        for i, entry in enumerate(hall):
            frame[f"p{i}"] = _png(entry["image"].reshape(-1))
        frames.append(frame)

    def decoder_lineage_diagnostics() -> dict:
        if shared_decoder:
            counts = np.asarray([len(population_z)], dtype=np.int64)
        elif reference_sweep:
            assert population_decoder_ids is not None
            _, counts = np.unique(population_decoder_ids, return_counts=True)
        else:
            counts = np.ones(len(population_z), dtype=np.int64)
        frequencies = counts / counts.sum()
        entropy = -float(np.sum(frequencies * np.log(frequencies)))
        return {
            "persistent_decoders": int(len(counts)),
            "effective_decoders": float(np.exp(entropy)),
            "largest_decoder_share": float(frequencies.max()),
            "decoder_carrier_counts": sorted(
                counts.astype(int).tolist(), reverse=True),
        }

    while spent < args.budget:
        if (merge_info is None and args.merge_at is not None
                and spent >= args.merge_at):
            teacher_z = np.stack([entry["z"] for entry in hall])
            teacher_images = np.stack([entry["image"] for entry in hall])
            teacher_theta = np.stack([entry["theta"] for entry in hall])
            theta_mean = teacher_theta.mean(axis=0)
            # Start from a real private function near the center, rather than
            # assuming that averaging distant weights averages their behavior.
            medoid = int(np.argmin(np.mean(
                (teacher_theta - theta_mean[None]) ** 2, axis=1)))
            if args.distill_init == "random":
                distill_initial_theta = template.init_theta(
                    int(rng.integers(0, 2**31)))
            else:
                distill_initial_theta = teacher_theta[medoid]
            premerge_records = {
                name: mse for name, mse in zip(names, hall_mses())}
            print(
                f"\nMERGE at {spent} evaluations: distilling "
                f"{target_count} target-vetted evolved phenotypes into one "
                "decoder",
                flush=True,
            )
            shared_theta, distilled_codes, distillation_trace = (
                distill_superdecoder(
                    teacher_z, teacher_images, distill_initial_theta,
                    args.distill_steps, args.distill_lr,
                    args.distill_code_lr, device,
                ))
            population_z, population_theta, population_goals = (
                shared_population_from_codes(
                    distilled_codes, shared_theta, args.survivors, rng))
            population_age = np.zeros(args.survivors, dtype=np.int64)
            population_step_gain = np.ones(
                args.survivors, dtype=np.float64)
            population_success_wins = np.zeros(
                args.survivors, dtype=np.int64)
            population_success_attempts = np.zeros(
                args.survivors, dtype=np.int64)
            population_stagnation_attempts = np.zeros(
                args.survivors, dtype=np.int64)
            merged_phenotypes = template.decode_batch(
                population_theta, population_z)
            population_phenotypes = merged_phenotypes
            population_scores = score(merged_phenotypes)
            spent += args.survivors

            # Keep only records that the shared decoder can actually produce.
            # Private teachers cannot linger after their parameters are gone.
            hall = [
                {"score": -np.inf, "image": None, "z": None, "theta": None}
                for _ in names
            ]
            update_hall(
                merged_phenotypes, population_scores,
                population_z, population_theta)
            shared_decoder = True
            global_stall = 0
            postmerge_records = {
                name: mse for name, mse in zip(names, hall_mses())}
            merge_info = {
                "private_evaluations": spent - args.survivors,
                "shared_evaluations_begin": spent,
                "teacher_count": target_count,
                "initial_private_medoid": medoid,
                "distill_init": args.distill_init,
                "premerge_records_mse": premerge_records,
                "immediate_shared_records_mse": postmerge_records,
                "premerge_mean_mse": float(np.mean(
                    list(premerge_records.values()))),
                "immediate_shared_mean_mse": float(np.mean(
                    list(postmerge_records.values()))),
                "distillation_trace": distillation_trace,
            }
            survivor_vectors = normalize_species_vectors(population_scores)
            merge_graph = graph_diagnostics(
                survivor_vectors, args.mating_radius)
            merge_row = {
                "e": spent,
                "generation": generation,
                "event": "decoder_merge",
                "gain": gain,
                "decoder_spread": 0.0,
                "shared_decoder": True,
                "reference_sweep": False,
                **decoder_lineage_diagnostics(),
                **merge_graph,
            }
            trace.append(merge_row)
            capture(merge_row)
            if view is not None:
                view.update(spent, hall)
            print(
                f"MERGED: persistent decoders 1, mean MSE "
                f"{merge_info['premerge_mean_mse']:.6f} -> "
                f"{merge_info['immediate_shared_mean_mse']:.6f}",
                flush=True,
            )
            continue

        if (not reference_sweep and args.reference_sweep_at is not None
                and spent >= args.reference_sweep_at):
            archived_records = {
                name: mse for name, mse in zip(names, hall_mses())}
            population_decoder_ids = np.arange(
                len(population_z), dtype=np.int64)
            decoder_bank = {
                int(index): population_theta[index].copy()
                for index in population_decoder_ids
            }
            decoder_best_fitness, decoder_stagnation = (
                update_decoder_stagnation(
                    population_scores, population_goals,
                    population_decoder_ids, {}, {}))
            next_decoder_id = len(decoder_bank)
            reference_sweep = True
            population_age = np.zeros(args.survivors, dtype=np.int64)
            if args.individual_step_start == "reference":
                population_step_gain = np.ones(
                    args.survivors, dtype=np.float64)
                population_success_wins = np.zeros(
                    args.survivors, dtype=np.int64)
                population_success_attempts = np.zeros(
                    args.survivors, dtype=np.int64)
                population_stagnation_attempts = np.zeros(
                    args.survivors, dtype=np.int64)

            # The lifetime archive may depend on decoder lineages that are no
            # longer in the population. From this point the displayed records
            # are rebuilt every generation from the currently reproducible
            # genome/decoder references only.
            hall = [
                {"score": -np.inf, "image": None, "z": None, "theta": None}
                for _ in names
            ]
            update_hall(
                population_phenotypes, population_scores,
                population_z, population_theta)
            current_records = {
                name: mse for name, mse in zip(names, hall_mses())}
            reference_sweep_info = {
                "private_evaluations": spent,
                "initial_decoders": len(decoder_bank),
                "archived_pre_sweep_records_mse": archived_records,
                "initial_current_records_mse": current_records,
                "archived_pre_sweep_mean_mse": float(np.mean(
                    list(archived_records.values()))),
                "initial_current_mean_mse": float(np.mean(
                    list(current_records.values()))),
                "fixation_evaluation": None,
                "extinction_events": [],
                "transactional_merge_attempts": [],
                "decoder_probe_attempts": [],
            }
            if args.decoder_probe_every:
                next_decoder_probe_at = (
                    spent + args.decoder_probe_delay)
            survivor_vectors = normalize_species_vectors(population_scores)
            sweep_graph = graph_diagnostics(
                survivor_vectors, args.mating_radius)
            sweep_row = {
                "e": spent,
                "generation": generation,
                "event": "decoder_reference_sweep",
                "gain": gain,
                "decoder_spread": float(
                    np.sqrt(np.mean((
                        population_theta
                        - population_theta.mean(axis=0, keepdims=True)
                    ) ** 2))
                    / max(np.sqrt(np.mean(population_theta ** 2)), 1e-12)),
                "shared_decoder": False,
                "reference_sweep": True,
                **decoder_lineage_diagnostics(),
                **sweep_graph,
            }
            trace.append(sweep_row)
            capture(sweep_row)
            if view is not None:
                view.update(spent, hall)
            print(
                f"\nREFERENCE SWEEP at {spent}: "
                f"{len(decoder_bank)} immutable decoder lineages; "
                f"active-current mean MSE "
                f"{reference_sweep_info['initial_current_mean_mse']:.6f}",
                flush=True,
            )
            continue

        generation += 1
        decoder_probe_attempts = 0
        decoder_probe_accepts = 0
        count = min(args.children, args.budget - spent)
        transition_at = (args.merge_at if args.merge_at is not None
                         else args.reference_sweep_at)
        transition_pending = (
            (args.merge_at is not None and merge_info is None)
            or (args.reference_sweep_at is not None and not reference_sweep)
        )
        if (transition_pending and transition_at is not None
                and spent < transition_at):
            count = min(count, transition_at - spent)
        parent = rng.integers(0, len(population_z), count)
        parent_vectors = normalize_species_vectors(population_scores)
        mates, mate_distances = choose_compatible_mates(
            parent_vectors, parent, args.mating_radius, rng)
        sexual = mates >= 0
        inherited_goals = population_goals[parent]
        mating_population_decoder_ids = (
            population_decoder_ids.copy() if reference_sweep else None)
        individual_control_active = (
            args.individual_step_control != "off"
            and (
                args.individual_step_start == "founder"
                or reference_sweep
            )
        )
        child_step_multipliers = (
            population_step_gain[parent].copy()
            if individual_control_active else np.ones(count, dtype=np.float64)
        )

        child_z = np.empty((count, LATENT), dtype=np.float32)
        if reference_sweep:
            child_decoder_ids, decoder_from_mate = (
                inherit_decoder_references(
                    parent, mates, inherited_goals, population_scores,
                    population_decoder_ids))
        else:
            child_decoder_ids = None
            decoder_from_mate = np.zeros(count, dtype=bool)
        base_theta = population_theta[parent].copy()
        for i, (base, mate) in enumerate(zip(parent, mates)):
            if mate >= 0:
                child_z[i] = mutate_genome(crossover(
                    population_z[base], population_z[mate]),
                    child_step_multipliers[i])
                if not reference_sweep and not shared_decoder:
                    base_theta[i] = (
                        population_theta[base]
                        + population_theta[mate]) / 2.0
            else:
                child_z[i] = mutate_genome(
                    population_z[base], child_step_multipliers[i])

        if shared_decoder:
            # Literal convergence: every individual references one exact
            # parameter vector, and no mutation can fork a private decoder.
            child_theta = np.repeat(shared_theta[None], count, axis=0)
        elif reference_sweep:
            assert child_decoder_ids is not None
            child_theta = np.stack([
                decoder_bank[int(decoder_id)]
                for decoder_id in child_decoder_ids
            ])
        else:
            sigmas = np.exp(rng.uniform(
                np.log(config.weight_sigma_low),
                np.log(config.weight_sigma_high), count)
            ) * gain * child_step_multipliers
            scales = np.maximum(base_theta.std(axis=1), 1e-3)
            child_theta = (
                base_theta
                + (sigmas * scales)[:, None]
                * rng.standard_normal((count, template.n_params))
            ).astype(np.float32)

        child_phenotypes = template.decode_batch(child_theta, child_z)
        child_scores = score(child_phenotypes)
        spent += count
        improved = update_hall(
            child_phenotypes, child_scores, child_z, child_theta)

        child_vectors = normalize_species_vectors(child_scores)
        goals = child_vectors.argmax(axis=1)
        relative_fitness = child_vectors[np.arange(count), goals]
        child_behavior_density = (
            local_behavior_density(child_vectors, args.mating_radius)
            if args.fitness_sharing_weight
            else np.ones(count, dtype=np.int64)
        )

        # Compare on the parent's inherited target so children cannot call a
        # mutation successful merely by cherry-picking a new easiest target.
        child_quality = child_scores[np.arange(count), inherited_goals]
        parent_quality = population_scores[parent, inherited_goals]
        target_scale = np.maximum(population_scores.std(axis=0), 1e-4)
        parent_progress = np.clip(
            (child_quality - parent_quality) / target_scale[inherited_goals],
            -5.0, 5.0,
        )
        selection_priority = (
            relative_fitness
            + args.progress_weight * parent_progress
            - args.fitness_sharing_weight
            * np.log(child_behavior_density.astype(np.float64))
        )
        child_wins = child_quality >= parent_quality - 1e-12
        win_rate = float(child_wins.mean())
        if not args.no_global_step_control:
            gain *= (config.gain_step if win_rate > config.win_target
                     else 1 / config.gain_step)
            gain = float(np.clip(gain, 0.3, config.gain_limits[1]))
            global_stall = 0 if improved else global_stall + 1
            if global_stall >= 25:
                gain = min(gain * 3.0, config.gain_limits[1])
                global_stall = 0

        (updated_step_gain,
         updated_success_wins,
         updated_success_attempts,
         updated_stagnation_attempts,
         individual_step_updates,
         individual_stagnation_kicks) = update_individual_step_state(
            population_step_gain,
            population_success_wins,
            population_success_attempts,
            population_stagnation_attempts,
            parent,
            child_wins,
            (args.individual_step_control
             if individual_control_active else "off"),
            config.win_target,
            config.gain_step,
            args.individual_success_window,
            args.individual_stagnation_attempts,
            args.individual_stagnation_kick,
            (args.individual_gain_min, args.individual_gain_max),
        )
        cumulative_individual_step_updates += individual_step_updates
        cumulative_individual_stagnation_kicks += individual_stagnation_kicks
        child_step_gain = updated_step_gain[parent]
        child_success_wins = updated_success_wins[parent]
        child_success_attempts = updated_success_attempts[parent]
        child_stagnation_attempts = updated_stagnation_attempts[parent]

        expired_parent_count = 0
        lineage_successors = 0
        lineage_reprieves = 0
        parent_scores_for_selection = population_scores
        if (reference_sweep and args.max_reference_age is not None):
            expired = population_age >= args.max_reference_age
            expired_parent_count = int(expired.sum())
            if expired_parent_count:
                parent_scores_for_selection = population_scores.copy()
                parent_scores_for_selection[expired] = -np.inf

        if count >= args.survivors:
            # Hard targets choose first when related targets share the same
            # generalist. Roles are reassigned from scratch every generation.
            target_order = np.argsort(-np.asarray(hall_mses()))
            keep, selected_goals, target_children = (
                select_target_covered_survivors(
                    parent_scores_for_selection, child_scores,
                    selection_priority,
                    goals, args.survivors, target_order,
                ))
            if (reference_sweep
                    and args.max_reference_age is not None):
                assert population_decoder_ids is not None
                assert child_decoder_ids is not None
                keep, selected_goals, lineage_successors, lineage_reprieves = (
                    preserve_decoder_lineages(
                        keep, selected_goals,
                        population_scores, child_scores, selection_priority,
                        population_goals, goals,
                        population_decoder_ids, child_decoder_ids,
                        target_count,
                    ))
                target_children = int(np.count_nonzero(
                    keep[:target_count] >= len(population_z)))
            population_goals = selected_goals
            combined_z = np.concatenate([population_z, child_z], axis=0)
            combined_theta = np.concatenate(
                [population_theta, child_theta], axis=0)
            combined_scores = np.concatenate(
                [population_scores, child_scores], axis=0)
            combined_phenotypes = torch.cat(
                [population_phenotypes, child_phenotypes], dim=0)
            combined_age = np.concatenate([
                population_age + 1,
                np.zeros(count, dtype=np.int64),
            ])
            combined_step_gain = np.concatenate([
                updated_step_gain, child_step_gain])
            combined_success_wins = np.concatenate([
                updated_success_wins, child_success_wins])
            combined_success_attempts = np.concatenate([
                updated_success_attempts, child_success_attempts])
            combined_stagnation_attempts = np.concatenate([
                updated_stagnation_attempts, child_stagnation_attempts])
            population_z = combined_z[keep]
            population_theta = combined_theta[keep]
            population_scores = combined_scores[keep]
            population_phenotypes = combined_phenotypes[
                torch.as_tensor(keep, device=device)]
            population_age = combined_age[keep]
            population_step_gain = combined_step_gain[keep]
            population_success_wins = combined_success_wins[keep]
            population_success_attempts = combined_success_attempts[keep]
            population_stagnation_attempts = (
                combined_stagnation_attempts[keep])
            if reference_sweep:
                assert population_decoder_ids is not None
                assert child_decoder_ids is not None
                combined_decoder_ids = np.concatenate(
                    [population_decoder_ids, child_decoder_ids])
                population_decoder_ids = combined_decoder_ids[keep]
                active_ids = set(population_decoder_ids.astype(int).tolist())
                extinct_ids = sorted(set(decoder_bank) - active_ids)
                if extinct_ids:
                    for decoder_id in extinct_ids:
                        del decoder_bank[decoder_id]
                    reference_sweep_info["extinction_events"].append({
                        "e": spent,
                        "extinct_decoder_ids": extinct_ids,
                        "remaining_decoders": len(decoder_bank),
                    })
                population_theta = np.stack([
                    decoder_bank[int(decoder_id)]
                    for decoder_id in population_decoder_ids
                ])
                if (len(decoder_bank) == 1
                        and reference_sweep_info["fixation_evaluation"] is None):
                    reference_sweep_info["fixation_evaluation"] = spent

                hall = [
                    {"score": -np.inf, "image": None,
                     "z": None, "theta": None}
                    for _ in names
                ]
                update_hall(
                    population_phenotypes, population_scores,
                    population_z, population_theta)
        else:
            target_children = 0
            population_step_gain = updated_step_gain
            population_success_wins = updated_success_wins
            population_success_attempts = updated_success_attempts
            population_stagnation_attempts = updated_stagnation_attempts
            if reference_sweep:
                hall = [
                    {"score": -np.inf, "image": None,
                     "z": None, "theta": None}
                    for _ in names
                ]
                update_hall(
                    population_phenotypes, population_scores,
                    population_z, population_theta)

        if reference_sweep:
            assert population_decoder_ids is not None
            decoder_best_fitness, decoder_stagnation = (
                update_decoder_stagnation(
                    population_scores, population_goals,
                    population_decoder_ids,
                    decoder_best_fitness, decoder_stagnation))

        if (reference_sweep and args.transactional_merges
                and len(decoder_bank) > 1
                and mating_population_decoder_ids is not None):
            active_ids = set(decoder_bank)
            merge_pair, encounters = most_encountered_decoder_pair(
                parent, mates, mating_population_decoder_ids, active_ids,
                decoder_stagnation, args.merge_stagnation_weight,
                args.merge_stagnation_grace)
            if merge_pair is not None:
                a, b = merge_pair
                pair_stagnation = (
                    decoder_stagnation.get(a, 0)
                    + decoder_stagnation.get(b, 0))
                carriers = np.flatnonzero(
                    (population_decoder_ids == a)
                    | (population_decoder_ids == b))
                if len(carriers) and spent + len(carriers) <= args.budget:
                    carrier_index = torch.as_tensor(carriers, device=device)
                    teacher_images = (
                        population_phenotypes[carrier_index]
                        .detach().cpu().numpy().reshape(len(carriers), *SHAPE)
                    )
                    teacher_z = population_z[carriers]
                    a_carriers = np.flatnonzero(population_decoder_ids == a)
                    b_carriers = np.flatnonzero(population_decoder_ids == b)
                    a_quality = float(np.mean(population_scores[
                        a_carriers, population_goals[a_carriers]]))
                    b_quality = float(np.mean(population_scores[
                        b_carriers, population_goals[b_carriers]]))
                    fitter = a if a_quality >= b_quality else b
                    candidate_theta, candidate_codes, local_trace = (
                        distill_superdecoder(
                            teacher_z, teacher_images, decoder_bank[fitter],
                            args.local_merge_steps, args.distill_lr,
                            args.distill_code_lr, device, report=False,
                        ))
                    candidate_thetas = np.repeat(
                        candidate_theta[None], len(carriers), axis=0)
                    candidate_phenotypes = template.decode_batch(
                        candidate_thetas, candidate_codes)
                    candidate_scores = score(candidate_phenotypes)
                    spent += len(carriers)

                    old_target_mse = -population_scores.max(axis=0)
                    proposed_scores = population_scores.copy()
                    proposed_scores[carriers] = candidate_scores
                    new_target_mse = -proposed_scores.max(axis=0)
                    old_mean = float(old_target_mse.mean())
                    new_mean = float(new_target_mse.mean())
                    old_worst = float(old_target_mse.max())
                    new_worst = float(new_target_mse.max())
                    accepted = (
                        args.local_merge_policy == "always"
                        or (
                            new_mean
                            <= old_mean + args.local_merge_tolerance
                            and new_worst
                            <= old_worst + args.local_merge_tolerance
                        )
                    )
                    remaining = len(decoder_bank)
                    new_id = None
                    if accepted:
                        new_id = next_decoder_id
                        next_decoder_id += 1
                        del decoder_bank[a]
                        del decoder_bank[b]
                        decoder_bank[new_id] = candidate_theta
                        decoder_best_fitness.pop(a, None)
                        decoder_best_fitness.pop(b, None)
                        decoder_stagnation.pop(a, None)
                        decoder_stagnation.pop(b, None)
                        population_decoder_ids[carriers] = new_id
                        population_z[carriers] = candidate_codes
                        population_age[carriers] = 0
                        population_step_gain[carriers] = 1.0
                        population_success_wins[carriers] = 0
                        population_success_attempts[carriers] = 0
                        population_stagnation_attempts[carriers] = 0
                        population_scores[carriers] = candidate_scores
                        decoder_best_fitness[new_id] = float(np.mean(
                            candidate_scores[
                                np.arange(len(carriers)),
                                population_goals[carriers],
                            ]))
                        decoder_stagnation[new_id] = 0
                        population_phenotypes = population_phenotypes.clone()
                        population_phenotypes[carrier_index] = (
                            candidate_phenotypes)
                        population_theta = np.stack([
                            decoder_bank[int(decoder_id)]
                            for decoder_id in population_decoder_ids
                        ])
                        remaining = len(decoder_bank)
                        hall = [
                            {"score": -np.inf, "image": None,
                             "z": None, "theta": None}
                            for _ in names
                        ]
                        update_hall(
                            population_phenotypes, population_scores,
                            population_z, population_theta)
                        if (remaining == 1 and reference_sweep_info[
                                "fixation_evaluation"] is None):
                            reference_sweep_info[
                                "fixation_evaluation"] = spent
                    attempt = {
                        "e": spent,
                        "pair": [a, b],
                        "new_decoder_id": new_id,
                        "encounters": encounters,
                        "pair_stagnation": pair_stagnation,
                        "carriers": len(carriers),
                        "fitter_parent_decoder": fitter,
                        "accepted": accepted,
                        "policy": args.local_merge_policy,
                        "old_mean_mse": old_mean,
                        "new_mean_mse": new_mean,
                        "old_worst_mse": old_worst,
                        "new_worst_mse": new_worst,
                        "phenotype_mse": local_trace[-1]["phenotype_mse"],
                        "remaining_decoders": remaining,
                    }
                    reference_sweep_info[
                        "transactional_merge_attempts"].append(attempt)
                    print(
                        f"    local merge {a}+{b} "
                        f"{'accepted' if accepted else 'rejected'}  "
                        f"carriers {len(carriers)}  "
                        f"mean {old_mean:.6f}->{new_mean:.6f}  "
                        f"decoders {remaining}",
                        flush=True,
                    )

        if (reference_sweep
                and args.decoder_probe_every
                and next_decoder_probe_at is not None
                and spent >= next_decoder_probe_at
                and spent < args.budget):
            assert population_decoder_ids is not None
            assert reference_sweep_info is not None
            accepted_any = False
            for decoder_id in sorted(decoder_bank):
                carriers = np.flatnonzero(
                    population_decoder_ids == decoder_id)
                if len(carriers) < args.decoder_probe_min_carriers:
                    continue
                # Plus, minus, and the fractional candidate are all charged
                # against the same evaluation budget as ordinary offspring.
                if spent + 3 * len(carriers) > args.budget:
                    break
                theta = decoder_bank[decoder_id]
                perturbation = sparse_decoder_perturbation(
                    theta, args.decoder_probe_fraction,
                    args.decoder_probe_sigma, probe_rng)
                codes = population_z[carriers]
                roles = population_goals[carriers]
                plus_theta = theta + perturbation
                minus_theta = theta - perturbation
                probe_thetas = np.concatenate([
                    np.repeat(plus_theta[None], len(carriers), axis=0),
                    np.repeat(minus_theta[None], len(carriers), axis=0),
                ])
                probe_codes = np.concatenate([codes, codes], axis=0)
                probe_phenotypes = template.decode_batch(
                    probe_thetas, probe_codes)
                probe_scores = score(probe_phenotypes)
                spent += 2 * len(carriers)
                plus_fitness, plus_worst = assigned_role_fitness(
                    probe_scores[:len(carriers)], roles)
                minus_fitness, minus_worst = assigned_role_fitness(
                    probe_scores[len(carriers):], roles)
                if args.decoder_probe_objective == "coverage":
                    plus_population_scores = population_scores.copy()
                    minus_population_scores = population_scores.copy()
                    plus_population_scores[carriers] = (
                        probe_scores[:len(carriers)])
                    minus_population_scores[carriers] = (
                        probe_scores[len(carriers):])
                    plus_fitness, plus_worst = target_coverage_fitness(
                        plus_population_scores)
                    minus_fitness, minus_worst = target_coverage_fitness(
                        minus_population_scores)

                candidate_theta = mirrored_decoder_candidate(
                    theta, perturbation, plus_fitness, minus_fitness,
                    args.decoder_probe_step_fraction)
                candidate_thetas = np.repeat(
                    candidate_theta[None], len(carriers), axis=0)
                candidate_phenotypes = template.decode_batch(
                    candidate_thetas, codes)
                candidate_scores = score(candidate_phenotypes)
                spent += len(carriers)
                old_fitness, old_worst = assigned_role_fitness(
                    population_scores[carriers], roles)
                candidate_fitness, candidate_worst = assigned_role_fitness(
                    candidate_scores, roles)
                if args.decoder_probe_objective == "coverage":
                    old_fitness, old_worst = target_coverage_fitness(
                        population_scores)
                    candidate_population_scores = population_scores.copy()
                    candidate_population_scores[carriers] = candidate_scores
                    candidate_fitness, candidate_worst = (
                        target_coverage_fitness(candidate_population_scores))
                accepted = (
                    candidate_fitness > old_fitness + 1e-12
                    and candidate_worst
                    >= old_worst - args.decoder_probe_worst_tolerance
                )
                if accepted:
                    decoder_bank[decoder_id] = candidate_theta
                    population_theta[carriers] = candidate_theta
                    population_scores[carriers] = candidate_scores
                    population_phenotypes = population_phenotypes.clone()
                    carrier_index = torch.as_tensor(carriers, device=device)
                    population_phenotypes[carrier_index] = (
                        candidate_phenotypes)
                    population_stagnation_attempts[carriers] = 0
                    accepted_any = True
                    decoder_probe_accepts += 1
                decoder_probe_attempts += 1
                cumulative_decoder_probe_attempts += 1
                cumulative_decoder_probe_accepts += int(accepted)
                reference_sweep_info["decoder_probe_attempts"].append({
                    "e": spent,
                    "decoder_id": int(decoder_id),
                    "carriers": int(len(carriers)),
                    "perturbed_weights": int(np.count_nonzero(perturbation)),
                    "plus_fitness": plus_fitness,
                    "minus_fitness": minus_fitness,
                    "plus_worst_fitness": plus_worst,
                    "minus_worst_fitness": minus_worst,
                    "old_fitness": old_fitness,
                    "candidate_fitness": candidate_fitness,
                    "old_worst_fitness": old_worst,
                    "candidate_worst_fitness": candidate_worst,
                    "accepted": accepted,
                })
            if accepted_any:
                # Historical images produced by the old weights are no longer
                # reproducible by an updated shared reference.
                hall = [
                    {"score": -np.inf, "image": None,
                     "z": None, "theta": None}
                    for _ in names
                ]
                update_hall(
                    population_phenotypes, population_scores,
                    population_z, population_theta)
            while next_decoder_probe_at <= spent:
                next_decoder_probe_at += args.decoder_probe_every
            if decoder_probe_attempts:
                print(
                    f"    decoder probes {decoder_probe_accepts}/"
                    f"{decoder_probe_attempts} accepted  "
                    f"total {cumulative_decoder_probe_accepts}/"
                    f"{cumulative_decoder_probe_attempts}",
                    flush=True,
                )

        survivor_vectors = normalize_species_vectors(population_scores)
        graph = graph_diagnostics(survivor_vectors, args.mating_radius)
        survivor_goals = survivor_vectors.argmax(axis=1)
        goal_counts = np.bincount(
            population_goals, minlength=target_count).astype(int).tolist()
        emergent_goal_counts = np.bincount(
            survivor_goals, minlength=target_count).astype(int).tolist()
        if shared_decoder:
            decoder_spread = 0.0
        else:
            theta_center = population_theta.mean(axis=0, keepdims=True)
            decoder_spread = float(
                np.sqrt(np.mean((population_theta - theta_center) ** 2))
                / max(np.sqrt(np.mean(population_theta ** 2)), 1e-12))
        lineage = decoder_lineage_diagnostics()
        decoder_stagnation_values = (
            list(decoder_stagnation.values()) if reference_sweep else [])
        row = {
            "e": spent,
            "generation": generation,
            "gain": gain,
            "win_rate": win_rate,
            "sexual_fraction": float(sexual.mean()),
            "mean_mating_distance": (float(np.nanmean(mate_distances))
                                     if sexual.any() else None),
            "decoder_spread": decoder_spread,
            "shared_decoder": shared_decoder,
            "reference_sweep": reference_sweep,
            "decoder_from_mate_fraction": (
                float(decoder_from_mate[sexual].mean())
                if reference_sweep and sexual.any() else None),
            "mean_individual_age": float(population_age.mean()),
            "max_individual_age": int(population_age.max()),
            "mean_child_behavior_density": float(
                child_behavior_density.mean()),
            "max_child_behavior_density": int(
                child_behavior_density.max()),
            "mean_decoder_stagnation": (
                float(np.mean(decoder_stagnation_values))
                if decoder_stagnation_values else 0.0),
            "max_decoder_stagnation": (
                int(max(decoder_stagnation_values))
                if decoder_stagnation_values else 0),
            "expired_parents": expired_parent_count,
            "lineage_successors": lineage_successors,
            "lineage_reprieves": lineage_reprieves,
            "mean_individual_step_gain": float(
                population_step_gain.mean()),
            "min_individual_step_gain": float(
                population_step_gain.min()),
            "max_individual_step_gain": float(
                population_step_gain.max()),
            "individual_step_updates": individual_step_updates,
            "individual_stagnation_kicks": individual_stagnation_kicks,
            "cumulative_individual_step_updates": (
                cumulative_individual_step_updates),
            "cumulative_individual_stagnation_kicks": (
                cumulative_individual_stagnation_kicks),
            "decoder_probe_attempts": decoder_probe_attempts,
            "decoder_probe_accepts": decoder_probe_accepts,
            "cumulative_decoder_probe_attempts": (
                cumulative_decoder_probe_attempts),
            "cumulative_decoder_probe_accepts": (
                cumulative_decoder_probe_accepts),
            **lineage,
            "goal_counts": goal_counts,
            "emergent_goal_counts": emergent_goal_counts,
            "mean_parent_progress": float(parent_progress.mean()),
            "improving_fraction": float((parent_progress > 0).mean()),
            "target_elites_from_children": target_children,
            **graph,
        }
        trace.append(row)

        if view is not None and generation % 5 == 0:
            view.update(spent, hall)

        if spent >= next_frame or spent >= args.budget:
            capture(row)
            while next_frame <= spent:
                next_frame += frame_interval
            mses = np.asarray(hall_mses())
            easiest = int(np.argmin(mses))
            emergent = int(np.count_nonzero(emergent_goal_counts))
            print(
                f"  {spent:>7} evals  gain {gain:.2f}  "
                f"best {names[easiest]} {mses[easiest]:.5f}  "
                f"mean {mses.mean():.5f}  worst {mses.max():.5f}  "
                f"roles {np.count_nonzero(goal_counts)}  "
                f"emergent {emergent}  components {graph['components']} "
                f"{graph['component_sizes'][:5]}  "
                f"sexual {100 * sexual.mean():.0f}%  "
                f"target-child {target_children}  "
                f"decoders {row['persistent_decoders']}  "
                f"largest {100 * row['largest_decoder_share']:.0f}%  "
                f"age {row['max_individual_age']}",
                flush=True,
            )

    if view is not None:
        view.update(spent, hall)
    if not frames or frames[-1]["e"] != spent:
        capture(trace[-1])

    records = {name: mse for name, mse in zip(names, hall_mses())}
    final_vectors = normalize_species_vectors(population_scores)
    final_graph = graph_diagnostics(final_vectors, args.mating_radius)
    print("\nFINAL target reconstruction MSE:")
    for name, value in sorted(records.items(), key=lambda item: item[1]):
        print(f"  {name:<28} {value:.6f}")
    print(f"FINAL compatibility graph: {final_graph}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "method": (
                "real_image_shared_self_distill"
                if args.start_shared and args.merge_at is not None else
                "real_image_shared_frozen" if args.start_shared else
                "real_image_superdecoder" if args.merge_at is not None else
                "real_image_decoder_transactional_merges"
                if args.transactional_merges else
                "real_image_decoder_reference_sweep"
                if args.reference_sweep_at is not None else
                "real_image_species_vector_covered_progress"
            ),
            "targets": [str(path) for path in args.targets],
            "target_images": {
                name: _png(array.reshape(-1))
                for name, array in zip(names, target_arrays)
            },
            "budget": args.budget,
            "seed": args.seed,
            "survivors": args.survivors,
            "children": args.children,
            "mating_radius": args.mating_radius,
            "start_gain": args.start_gain,
            "start_shared": args.start_shared,
            "progress_weight": args.progress_weight,
            "merge_at": args.merge_at,
            "reference_sweep_at": args.reference_sweep_at,
            "transactional_merges": args.transactional_merges,
            "local_merge_steps": args.local_merge_steps,
            "local_merge_tolerance": args.local_merge_tolerance,
            "local_merge_policy": args.local_merge_policy,
            "fitness_sharing_weight": args.fitness_sharing_weight,
            "merge_stagnation_weight": args.merge_stagnation_weight,
            "merge_stagnation_grace": args.merge_stagnation_grace,
            "max_reference_age": args.max_reference_age,
            "lineage_succession": args.max_reference_age is not None,
            "individual_step_control": args.individual_step_control,
            "individual_step_start": args.individual_step_start,
            "individual_success_window": args.individual_success_window,
            "individual_gain_limits": [
                args.individual_gain_min, args.individual_gain_max],
            "individual_stagnation_attempts": (
                args.individual_stagnation_attempts),
            "individual_stagnation_kick": args.individual_stagnation_kick,
            "global_step_control": not args.no_global_step_control,
            "decoder_probe_every": args.decoder_probe_every,
            "decoder_probe_delay": args.decoder_probe_delay,
            "decoder_probe_min_carriers": args.decoder_probe_min_carriers,
            "decoder_probe_objective": args.decoder_probe_objective,
            "decoder_probe_fraction": args.decoder_probe_fraction,
            "decoder_probe_sigma": args.decoder_probe_sigma,
            "decoder_probe_step_fraction": (
                args.decoder_probe_step_fraction),
            "decoder_probe_worst_tolerance": (
                args.decoder_probe_worst_tolerance),
            "distill_steps": args.distill_steps,
            "distill_lr": args.distill_lr,
            "distill_code_lr": args.distill_code_lr,
            "distill_init": args.distill_init,
            "initial_persistent_decoders": 1,
            "final_persistent_decoders": (
                1 if shared_decoder else len(decoder_bank)
                if reference_sweep else args.survivors),
            "merge": merge_info,
            "reference_sweep": reference_sweep_info,
            "torch_version": torch.__version__,
            "records_mse": records,
            "final_graph": final_graph,
            "trace": trace,
            "frames": frames,
        }, indent=2) + "\n")
        print(f"wrote {args.output}")

    if args.gif and frames:
        columns = min(8, target_count)
        rows = int(np.ceil(target_count / columns))
        tile = 64
        images = []
        for frame in frames:
            sheet = Image.new(
                "RGB", (tile * 2 * columns, tile * rows), (20, 20, 20))
            for target in range(target_count):
                evolved = Image.open(io.BytesIO(base64.b64decode(
                    frame[f"p{target}"].split(",", 1)[1]))).convert("RGB")
                evolved = evolved.resize((tile, tile), Image.Resampling.NEAREST)
                reference = Image.fromarray(
                    (target_arrays[target].transpose(1, 2, 0) * 255)
                    .astype(np.uint8)).resize((tile, tile), Image.Resampling.LANCZOS)
                pair = Image.new("RGB", (tile * 2, tile))
                pair.paste(reference, (0, 0))
                pair.paste(evolved, (tile, 0))
                draw = ImageDraw.Draw(pair)
                draw.rectangle([0, tile - 11, tile * 2, tile], fill=(0, 0, 0))
                draw.text(
                    (2, tile - 10),
                    f"{names[target]} {frame['mses'][target]:.3f}",
                    fill=(255, 255, 255),
                )
                sheet.paste(
                    pair, (tile * 2 * (target % columns),
                           tile * (target // columns)))
            images.append(sheet)
        images.append(images[-1])
        args.gif.parent.mkdir(parents=True, exist_ok=True)
        images[0].save(
            args.gif, save_all=True, append_images=images[1:],
            duration=[80] * (len(images) - 1) + [2500], loop=0,
            optimize=True,
        )
        print(f"wrote {args.gif}")

    if view is not None:
        view.plt.ioff()
        if args.hold_open:
            view.plt.show()
        else:
            view.plt.close(view.fig)


if __name__ == "__main__":
    main()
