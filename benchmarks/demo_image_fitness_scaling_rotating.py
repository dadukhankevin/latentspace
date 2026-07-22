"""Fixed-compute image-fitness scaling with rotating objective panels.

The living population remains 48 individuals and evaluates at most 32 image
fitness functions per generation, even when the full target corpus is much
larger. A deterministic balanced scheduler rotates targets. The best state
ever found for every target is stored in a global archive and is eligible to
re-enter the living population whenever that target becomes active again.

This isolates the question "does access to more objectives improve search?"
without increasing decoder count, survivor count, child count, or the number
of active phenotype/target comparisons per ordinary generation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from benchmarks.compare import _require_mps, _seed_everything
from benchmarks.demo_clip_species_vector import (
    graph_diagnostics,
    normalize_species_vectors,
)
from benchmarks.demo_image_species_conditional_lora import (
    LATENT,
    SHAPE,
    aggregate_records,
    choose_ecological_mates_within_input_species,
    decode_conditional,
    initialize_conditional_decoder,
    lineage_succession_selection_scores,
    mating_compatibility_vectors,
)
from benchmarks.demo_image_species_vector import (
    load_targets,
    pairwise_negative_mse,
    select_target_covered_survivors,
    update_individual_step_state,
)
from benchmarks.round28_anchor_conv import ConvRGB
from latentspace.universal.architectures import resolve
from benchmarks.legacy_engines.explorer import ExplorerConfig, _Template


def balanced_target_panel(
        appearances: np.ndarray,
        width: int,
        rng: np.random.Generator,
        ) -> np.ndarray:
    """Choose a unique least-seen panel, randomizing only equal-count ties."""
    counts = np.asarray(appearances)
    if counts.ndim != 1 or not np.issubdtype(counts.dtype, np.integer):
        raise ValueError("appearances must be a one-dimensional integer array")
    if not 1 <= width <= len(counts):
        raise ValueError("panel width must be between one and target count")
    tie_break = rng.random(len(counts))
    panel = np.lexsort((tie_break, counts))[:width].astype(np.int64)
    appearances[panel] += 1
    return panel


def update_target_archive(
        archive_scores: np.ndarray,
        archive_z: np.ndarray,
        archive_coefficients: np.ndarray,
        active_targets: np.ndarray,
        candidate_scores: np.ndarray,
        candidate_z: np.ndarray,
        candidate_coefficients: np.ndarray,
        ) -> int:
    """Update global target champions from one active score matrix."""
    active = np.asarray(active_targets, dtype=np.int64)
    scores = np.asarray(candidate_scores, dtype=np.float64)
    z = np.asarray(candidate_z, dtype=np.float32)
    coefficients = np.asarray(candidate_coefficients, dtype=np.float32)
    if scores.shape != (len(z), len(active)):
        raise ValueError("candidate scores must match candidates and panel")
    if len(coefficients) != len(z):
        raise ValueError("candidate state arrays must align")
    if len(np.unique(active)) != len(active):
        raise ValueError("active targets must be unique")
    updates = 0
    for local, target in enumerate(active):
        winner = int(np.argmax(scores[:, local]))
        value = float(scores[winner, local])
        if value > archive_scores[target]:
            archive_scores[target] = value
            archive_z[target] = z[winner]
            archive_coefficients[target] = coefficients[winner]
            updates += 1
    return updates


def unique_archive_entries(
        population_z: np.ndarray,
        population_coefficients: np.ndarray,
        archive_z: np.ndarray,
        archive_coefficients: np.ndarray,
        archive_scores: np.ndarray,
        active_targets: np.ndarray,
        ) -> np.ndarray:
    """Return active archived states not already present in the population."""
    keys = {
        (z.tobytes(), coefficients.tobytes())
        for z, coefficients in zip(population_z, population_coefficients)
    }
    selected: list[int] = []
    for target in np.asarray(active_targets, dtype=np.int64):
        if not np.isfinite(archive_scores[target]):
            continue
        key = (archive_z[target].tobytes(),
               archive_coefficients[target].tobytes())
        if key in keys:
            continue
        keys.add(key)
        selected.append(int(target))
    return np.asarray(selected, dtype=np.int64)


def _metrics(values: np.ndarray) -> dict[str, float | int | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return {
            "filled_targets": 0,
            "mean_mse": None,
            "median_mse": None,
            "worst_mse": None,
        }
    return {
        "filled_targets": int(len(finite)),
        "mean_mse": float(np.mean(finite)),
        "median_mse": float(np.median(finite)),
        "worst_mse": float(np.max(finite)),
    }


def record_quality_milestones(
        histories: list[list[list[float | int]]],
        active_targets: np.ndarray,
        exposures: np.ndarray,
        candidate_scores: np.ndarray,
        archive_scores: np.ndarray,
        step: int,
        limit: int,
        ) -> None:
    """Record exact best-so-far quality at target-local exposure counts.

    Candidate rows are treated in evaluation order. If a batch crosses one or
    more milestones, the score stored for each milestone uses only the prefix
    of candidates that target had seen by that exact exposure count.
    """
    if step < 1 or limit < step:
        raise ValueError("quality exposure milestones must be positive")
    active = np.asarray(active_targets, dtype=np.int64)
    counts = np.asarray(exposures, dtype=np.int64)
    scores = np.asarray(candidate_scores, dtype=np.float64)
    archive = np.asarray(archive_scores, dtype=np.float64)
    if scores.shape != (len(scores), len(active)):
        raise ValueError("candidate scores must match active targets")
    if len(histories) != len(counts) or len(archive) != len(counts):
        raise ValueError("target milestone arrays must align")

    batch = len(scores)
    for local, target in enumerate(active):
        start = int(counts[target])
        end = start + batch
        milestone = ((start // step) + 1) * step
        while milestone <= min(end, limit):
            prefix = milestone - start
            best = max(
                float(archive[target]),
                float(np.max(scores[:prefix, local])),
            )
            histories[target].append([milestone, -best])
            milestone += step


def run(args: argparse.Namespace) -> dict:
    names, target_arrays = load_targets(args.targets)
    target_count = len(names)
    active_count = min(args.active_targets, target_count)
    if active_count > args.survivors:
        raise ValueError("active targets cannot exceed survivors")
    if args.children < args.survivors:
        raise ValueError("children must be at least survivors")
    if args.panel_generations < 1:
        raise ValueError("panel generations must be positive")
    if args.max_age < 1:
        raise ValueError("max age must be positive")
    if (args.quality_exposure_step < 1
            or args.quality_exposure_limit < args.quality_exposure_step):
        raise ValueError("quality exposure milestone range is invalid")

    _require_mps()
    device = "mps"
    targets = torch.as_tensor(target_arrays, device=device)
    config = ExplorerConfig()
    _seed_everything(args.seed)
    rng = np.random.default_rng(args.seed)
    panel_rng = np.random.default_rng(args.panel_seed)
    template = _Template(resolve(
        lambda latent, shape: ConvRGB(latent, shape), LATENT, SHAPE), device)

    def score(phenotypes: torch.Tensor,
              active: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            values = pairwise_negative_mse(
                phenotypes.reshape(len(phenotypes), *SHAPE),
                targets[torch.as_tensor(active, device=device)],
            )
        return values.cpu().numpy().astype(np.float32)

    founder_theta = template.init_theta(int(rng.integers(0, 2**31)))
    model = initialize_conditional_decoder(
        "mixed", args.coefficient_dim, founder_theta, device)
    population_z = rng.standard_normal(
        (args.survivors, LATENT)).astype(np.float32)
    population_coefficients = np.zeros(
        (args.survivors, args.coefficient_dim), dtype=np.float32)
    population_phenotypes = decode_conditional(
        model, population_z, population_coefficients, device)

    all_targets = np.arange(target_count, dtype=np.int64)
    initial_all_scores = score(population_phenotypes, all_targets)
    initial_mse = -initial_all_scores.max(axis=0)
    quality_histories: list[list[list[float | int]]] = [
        [[0, float(value)]] for value in initial_mse
    ]
    analysis_score_comparisons = int(
        args.survivors * max(0, target_count - active_count))

    panel_appearances = np.zeros(target_count, dtype=np.int64)
    target_search_exposures = np.zeros(target_count, dtype=np.int64)
    target_fitness_exposures = np.zeros(target_count, dtype=np.int64)
    active_targets = balanced_target_panel(
        panel_appearances, active_count, panel_rng)
    target_search_exposures[active_targets] += args.survivors
    population_scores = initial_all_scores[:, active_targets]
    score_comparisons = int(args.survivors * active_count)
    refresh_decodes = 0

    population_goals = normalize_species_vectors(
        population_scores).argmax(axis=1)
    population_age = np.zeros(args.survivors, dtype=np.int64)
    population_step_gain = np.ones(args.survivors, dtype=np.float64)
    population_success_wins = np.zeros(args.survivors, dtype=np.int64)
    population_success_attempts = np.zeros(args.survivors, dtype=np.int64)
    population_stagnation_attempts = np.zeros(
        args.survivors, dtype=np.int64)

    archive_scores = np.full(target_count, -np.inf, dtype=np.float64)
    archive_z = np.zeros((target_count, LATENT), dtype=np.float32)
    archive_coefficients = np.zeros(
        (target_count, args.coefficient_dim), dtype=np.float32)
    record_quality_milestones(
        quality_histories,
        active_targets,
        target_fitness_exposures,
        population_scores,
        archive_scores,
        args.quality_exposure_step,
        args.quality_exposure_limit,
    )
    target_fitness_exposures[active_targets] += args.survivors
    update_target_archive(
        archive_scores, archive_z, archive_coefficients, active_targets,
        population_scores, population_z, population_coefficients)

    spent = args.survivors
    generation = 0
    gain = float(args.start_gain)
    global_stall = 0
    hall_scores = population_scores.max(axis=0).astype(np.float64)
    panel_history = [{
        "generation": 0,
        "e": spent,
        "targets": active_targets.tolist(),
    }]
    trace: list[dict] = []
    next_report = 0
    report_interval = max(1, args.budget // max(args.reports, 1))

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

    def crossover_coefficients(base: np.ndarray,
                               donor: np.ndarray) -> np.ndarray:
        half = len(base) // 2
        return np.concatenate([
            crossover(base[:half], donor[:half]),
            crossover(base[half:], donor[half:]),
        ]).astype(np.float32)

    def activate_panel() -> None:
        nonlocal active_targets, population_z, population_coefficients
        nonlocal population_phenotypes, population_scores, population_goals
        nonlocal population_age, population_step_gain
        nonlocal population_success_wins, population_success_attempts
        nonlocal population_stagnation_attempts, hall_scores
        nonlocal score_comparisons, refresh_decodes

        active_targets = balanced_target_panel(
            panel_appearances, active_count, panel_rng)
        archived = unique_archive_entries(
            population_z, population_coefficients,
            archive_z, archive_coefficients, archive_scores, active_targets)
        candidate_z = np.concatenate(
            [population_z, archive_z[archived]], axis=0)
        candidate_coefficients = np.concatenate(
            [population_coefficients, archive_coefficients[archived]], axis=0)
        if len(archived):
            archived_phenotypes = decode_conditional(
                model, archive_z[archived], archive_coefficients[archived],
                device)
            candidate_phenotypes = torch.cat(
                [population_phenotypes, archived_phenotypes], dim=0)
        else:
            candidate_phenotypes = population_phenotypes
        candidate_scores = score(candidate_phenotypes, active_targets)
        refresh_decodes += len(archived)
        score_comparisons += int(len(candidate_z) * active_count)
        record_quality_milestones(
            quality_histories,
            active_targets,
            target_fitness_exposures,
            candidate_scores,
            archive_scores,
            args.quality_exposure_step,
            args.quality_exposure_limit,
        )
        target_fitness_exposures[active_targets] += len(candidate_z)
        update_target_archive(
            archive_scores, archive_z, archive_coefficients, active_targets,
            candidate_scores, candidate_z, candidate_coefficients)

        vectors = normalize_species_vectors(candidate_scores)
        goals = vectors.argmax(axis=1)
        priority = vectors[np.arange(len(vectors)), goals]
        target_order = np.argsort(candidate_scores.max(axis=0))
        keep, selected_goals, _ = select_target_covered_survivors(
            np.empty((0, active_count), dtype=np.float32),
            candidate_scores,
            priority,
            goals,
            args.survivors,
            target_order,
        )
        current_count = len(population_z)
        candidate_age = np.concatenate([
            population_age,
            np.zeros(len(archived), dtype=np.int64),
        ])
        candidate_gain = np.concatenate([
            population_step_gain,
            np.ones(len(archived), dtype=np.float64),
        ])
        candidate_wins = np.concatenate([
            population_success_wins,
            np.zeros(len(archived), dtype=np.int64),
        ])
        candidate_attempts = np.concatenate([
            population_success_attempts,
            np.zeros(len(archived), dtype=np.int64),
        ])
        candidate_stagnation = np.concatenate([
            population_stagnation_attempts,
            np.zeros(len(archived), dtype=np.int64),
        ])
        population_z = candidate_z[keep]
        population_coefficients = candidate_coefficients[keep]
        population_phenotypes = candidate_phenotypes[
            torch.as_tensor(keep, device=device)]
        population_scores = candidate_scores[keep]
        population_goals = selected_goals
        population_age = candidate_age[keep]
        population_step_gain = candidate_gain[keep]
        population_success_wins = candidate_wins[keep]
        population_success_attempts = candidate_attempts[keep]
        population_stagnation_attempts = candidate_stagnation[keep]
        hall_scores = population_scores.max(axis=0).astype(np.float64)
        panel_history.append({
            "generation": generation,
            "e": spent,
            "targets": active_targets.tolist(),
            "archive_reentries": int(len(archived)),
            "retained_living_candidates": int(np.sum(keep < current_count)),
        })

    while spent < args.budget:
        generation += 1
        if (target_count > active_count and generation > 1
                and (generation - 1) % args.panel_generations == 0):
            activate_panel()

        count = min(args.children, args.budget - spent)
        parent = rng.integers(0, len(population_z), count)
        fitness_vectors = normalize_species_vectors(population_scores)
        compatibility_vectors = mating_compatibility_vectors(
            "z_only", "mixed", population_z, population_coefficients,
            population_scores)
        mates, mate_distances = choose_ecological_mates_within_input_species(
            compatibility_vectors,
            fitness_vectors,
            parent,
            args.mating_radius,
            args.ecological_mating_radius,
            rng,
        )
        sexual = mates >= 0
        inherited_goals = population_goals[parent]
        child_multipliers = population_step_gain[parent].copy()

        child_z = np.empty((count, LATENT), dtype=np.float32)
        child_coefficients = np.empty(
            (count, args.coefficient_dim), dtype=np.float32)
        for index, (base, mate) in enumerate(zip(parent, mates)):
            if mate >= 0:
                child_z[index] = mutate(crossover(
                    population_z[base], population_z[mate]),
                    child_multipliers[index])
                child_coefficients[index] = mutate(
                    crossover_coefficients(
                        population_coefficients[base],
                        population_coefficients[mate]),
                    child_multipliers[index])
            else:
                child_z[index] = mutate(
                    population_z[base], child_multipliers[index])
                child_coefficients[index] = mutate(
                    population_coefficients[base], child_multipliers[index])

        child_phenotypes = decode_conditional(
            model, child_z, child_coefficients, device)
        child_scores = score(child_phenotypes, active_targets)
        spent += count
        target_search_exposures[active_targets] += count
        record_quality_milestones(
            quality_histories,
            active_targets,
            target_fitness_exposures,
            child_scores,
            archive_scores,
            args.quality_exposure_step,
            args.quality_exposure_limit,
        )
        target_fitness_exposures[active_targets] += count
        score_comparisons += int(count * active_count)

        child_vectors = normalize_species_vectors(child_scores)
        goals = child_vectors.argmax(axis=1)
        relative_fitness = child_vectors[np.arange(count), goals]
        child_quality = child_scores[np.arange(count), inherited_goals]
        parent_quality = population_scores[parent, inherited_goals]
        target_scale = np.maximum(population_scores.std(axis=0), 1e-4)
        parent_progress = np.clip(
            (child_quality - parent_quality) / target_scale[inherited_goals],
            -5.0, 5.0)
        priority = relative_fitness + args.progress_weight * parent_progress
        child_wins = child_quality >= parent_quality - 1e-12

        gain *= (config.gain_step if child_wins.mean() > config.win_target
                 else 1 / config.gain_step)
        gain = float(np.clip(gain, 0.3, config.gain_limits[1]))
        improved = bool(np.any(
            child_scores.max(axis=0) > hall_scores + 1e-12))
        hall_scores = np.maximum(hall_scores, child_scores.max(axis=0))
        global_stall = 0 if improved else global_stall + 1
        if global_stall >= 25:
            gain = min(gain * 3.0, config.gain_limits[1])
            global_stall = 0

        birth_attempts = np.bincount(
            parent, minlength=len(population_z)).astype(np.int64)
        birth_wins = np.bincount(
            parent, weights=child_wins.astype(np.int64),
            minlength=len(population_z)).astype(np.int64)
        (updated_gain, updated_wins, updated_attempts, updated_stagnation,
         _, stagnation_kicks) = update_individual_step_state(
            population_step_gain,
            population_success_wins,
            population_success_attempts,
            population_stagnation_attempts,
            parent,
            child_wins,
            "stagnation",
            config.win_target,
            config.gain_step,
            args.individual_success_window,
            args.individual_stagnation_attempts,
            args.individual_stagnation_kick,
            (args.individual_gain_min, args.individual_gain_max),
        )
        child_gain = updated_gain[parent]
        child_success_wins = updated_wins[parent]
        child_success_attempts = updated_attempts[parent]
        child_stagnation = updated_stagnation[parent]

        combined_z = np.concatenate([population_z, child_z], axis=0)
        combined_coefficients = np.concatenate(
            [population_coefficients, child_coefficients], axis=0)
        combined_scores = np.concatenate(
            [population_scores, child_scores], axis=0)
        archive_updates = update_target_archive(
            archive_scores, archive_z, archive_coefficients, active_targets,
            combined_scores, combined_z, combined_coefficients)

        if count >= args.survivors:
            parent_selection, child_selection, _, succession = (
                lineage_succession_selection_scores(
                    population_scores,
                    child_scores,
                    population_goals,
                    population_age,
                    parent,
                    mates,
                    args.max_age,
                ))
            target_order = np.argsort(hall_scores)
            keep, selected_goals, target_children = (
                select_target_covered_survivors(
                    parent_selection,
                    child_selection,
                    priority,
                    goals,
                    args.survivors,
                    target_order,
                ))
            population_z = combined_z[keep]
            population_coefficients = combined_coefficients[keep]
            population_scores = combined_scores[keep]
            combined_phenotypes = torch.cat(
                [population_phenotypes, child_phenotypes], dim=0)
            population_phenotypes = combined_phenotypes[
                torch.as_tensor(keep, device=device)]
            population_goals = selected_goals
            population_age = np.concatenate([
                population_age + 1,
                np.zeros(count, dtype=np.int64),
            ])[keep]
            population_step_gain = np.concatenate(
                [updated_gain, child_gain])[keep]
            population_success_wins = np.concatenate(
                [updated_wins, child_success_wins])[keep]
            population_success_attempts = np.concatenate(
                [updated_attempts, child_success_attempts])[keep]
            population_stagnation_attempts = np.concatenate(
                [updated_stagnation, child_stagnation])[keep]
            hall_scores = population_scores.max(axis=0).astype(np.float64)
        else:
            succession = {
                "lineage_succession_targets": 0,
                "lineage_retirements": 0,
                "lineage_reprieves": 0,
            }
            target_children = 0
            population_step_gain = updated_gain
            population_success_wins = updated_wins
            population_success_attempts = updated_attempts
            population_stagnation_attempts = updated_stagnation

        if spent >= next_report or spent >= args.budget:
            active_mse = -population_scores.max(axis=0)
            archive_mse = np.where(
                np.isfinite(archive_scores), -archive_scores, np.nan)
            active_metrics = _metrics(active_mse)
            archive_metrics = _metrics(archive_mse)
            graph = graph_diagnostics(
                mating_compatibility_vectors(
                    "z_only", "mixed", population_z,
                    population_coefficients, population_scores),
                args.mating_radius,
            )
            row = {
                "e": spent,
                "generation": generation,
                "gain": gain,
                "active_targets": active_targets.tolist(),
                "archive_filled": archive_metrics["filled_targets"],
                "archive_mean_mse": archive_metrics["mean_mse"],
                "archive_worst_mse": archive_metrics["worst_mse"],
                "active_mean_mse": active_metrics["mean_mse"],
                "active_worst_mse": active_metrics["worst_mse"],
                "sexual_fraction": float(sexual.mean()),
                "archive_updates": archive_updates,
                "target_elites_from_children": target_children,
                "individual_stagnation_kicks": stagnation_kicks,
                **succession,
                **graph,
            }
            trace.append(row)
            print(
                f"  {spent:>7} evals  targets {target_count:>3}  "
                f"panel {active_count:>2}  archive "
                f"{archive_metrics['filled_targets']:>3}/{target_count}  "
                f"mean {archive_metrics['mean_mse']:.5f}  "
                f"worst {archive_metrics['worst_mse']:.5f}  "
                f"components {graph['components']}  "
                f"sexual {100 * sexual.mean():.0f}%",
                flush=True,
            )
            while next_report <= spent:
                next_report += report_interval

    archive_mse = np.where(
        np.isfinite(archive_scores), -archive_scores, np.nan)
    archive_metrics = _metrics(archive_mse)
    records = {
        name: float(archive_mse[index])
        for index, name in enumerate(names)
        if np.isfinite(archive_mse[index])
    }
    if len(records) == target_count:
        archive_metrics.update(aggregate_records(records))
        archive_metrics["filled_targets"] = target_count
    result = {
        "method": "fixed_compute_rotating_fitness_panel",
        "target_count": target_count,
        "active_target_count": active_count,
        "panel_generations": args.panel_generations,
        "budget": args.budget,
        "survivors": args.survivors,
        "children": args.children,
        "seed": args.seed,
        "panel_seed": args.panel_seed,
        "coefficient_dim": args.coefficient_dim,
        "compatibility_space": "z_only",
        "mating_radius": args.mating_radius,
        "ecological_mating_radius": args.ecological_mating_radius,
        "max_age": args.max_age,
        "targets": [str(path) for path in args.targets],
        "target_names": names,
        "initial_records_mse": {
            name: float(initial_mse[index])
            for index, name in enumerate(names)
        },
        "target_quality_milestones": {
            name: quality_histories[index]
            for index, name in enumerate(names)
        },
        "quality_exposure_step": args.quality_exposure_step,
        "quality_exposure_limit": args.quality_exposure_limit,
        "archive_records_mse": records,
        "archive_metrics": archive_metrics,
        "full_coverage": len(records) == target_count,
        "target_search_exposures": target_search_exposures.tolist(),
        "target_panel_appearances": panel_appearances.tolist(),
        "search_exposure_min": int(target_search_exposures.min()),
        "search_exposure_mean": float(target_search_exposures.mean()),
        "search_exposure_max": int(target_search_exposures.max()),
        "target_fitness_exposures": target_fitness_exposures.tolist(),
        "fitness_exposure_min": int(target_fitness_exposures.min()),
        "fitness_exposure_mean": float(target_fitness_exposures.mean()),
        "fitness_exposure_max": int(target_fitness_exposures.max()),
        "score_comparisons": score_comparisons,
        "analysis_score_comparisons": analysis_score_comparisons,
        "refresh_phenotype_decodes": refresh_decodes,
        "coefficient_rms": float(np.sqrt(np.mean(
            population_coefficients ** 2))),
        "panel_history": panel_history,
        "trace": trace,
    }
    print("\nFINAL:")
    print(json.dumps({
        "target_count": target_count,
        "active_target_count": active_count,
        **archive_metrics,
        "search_exposure_range": [
            result["search_exposure_min"],
            result["search_exposure_max"],
        ],
        "fitness_exposure_range": [
            result["fitness_exposure_min"],
            result["fitness_exposure_max"],
        ],
        "refresh_phenotype_decodes": refresh_decodes,
    }, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("targets", nargs="+", type=Path)
    parser.add_argument("--active-targets", type=int, default=32)
    parser.add_argument("--panel-generations", type=int, default=8)
    parser.add_argument("--survivors", type=int, default=48)
    parser.add_argument("--children", type=int, default=192)
    parser.add_argument("--budget", type=int, default=60_000)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--panel-seed", type=int, default=20_260_719)
    parser.add_argument("--coefficient-dim", type=int, default=64)
    parser.add_argument("--mating-radius", type=float, default=30.0)
    parser.add_argument("--ecological-mating-radius", type=float, default=0.3)
    parser.add_argument("--progress-weight", type=float, default=1.0)
    parser.add_argument("--max-age", type=int, default=10)
    parser.add_argument("--start-gain", type=float, default=1.0)
    parser.add_argument("--individual-success-window", type=int, default=20)
    parser.add_argument("--individual-gain-min", type=float, default=0.25)
    parser.add_argument("--individual-gain-max", type=float, default=4.0)
    parser.add_argument("--individual-stagnation-attempts", type=int, default=32)
    parser.add_argument("--individual-stagnation-kick", type=float, default=2.0)
    parser.add_argument("--quality-exposure-step", type=int, default=250)
    parser.add_argument("--quality-exposure-limit", type=int, default=3000)
    parser.add_argument("--reports", type=int, default=20)
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
