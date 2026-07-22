"""Structural tests for fixed-compute rotating image objectives."""

from __future__ import annotations

import json

import numpy as np

from benchmarks.analyze_target_exposure_curves import analyze
from benchmarks.demo_image_fitness_scaling_rotating import (
    balanced_target_panel,
    record_quality_milestones,
    unique_archive_entries,
    update_target_archive,
)


def test_balanced_panels_are_unique_deterministic_and_even():
    first_counts = np.zeros(168, dtype=np.int64)
    second_counts = np.zeros(168, dtype=np.int64)
    first_rng = np.random.default_rng(17)
    second_rng = np.random.default_rng(17)

    for _ in range(37):
        first = balanced_target_panel(first_counts, 32, first_rng)
        second = balanced_target_panel(second_counts, 32, second_rng)
        assert len(np.unique(first)) == 32
        assert np.array_equal(first, second)

    assert np.array_equal(first_counts, second_counts)
    assert first_counts.max() - first_counts.min() <= 1


def test_target_archive_keeps_each_targets_best_personal_state():
    archive_scores = np.full(4, -np.inf)
    archive_z = np.zeros((4, 2), dtype=np.float32)
    archive_coefficients = np.zeros((4, 2), dtype=np.float32)
    active = np.asarray([1, 3])
    candidate_z = np.asarray([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
    candidate_coefficients = candidate_z + 10

    updates = update_target_archive(
        archive_scores,
        archive_z,
        archive_coefficients,
        active,
        candidate_scores=np.asarray([[-0.5, -0.1], [-0.2, -0.4]]),
        candidate_z=candidate_z,
        candidate_coefficients=candidate_coefficients,
    )

    assert updates == 2
    assert archive_scores.tolist() == [-np.inf, -0.2, -np.inf, -0.1]
    assert archive_z[1].tolist() == [2.0, 2.0]
    assert archive_z[3].tolist() == [1.0, 1.0]

    updates = update_target_archive(
        archive_scores,
        archive_z,
        archive_coefficients,
        active,
        candidate_scores=np.asarray([[-0.8, -0.7], [-0.6, -0.3]]),
        candidate_z=candidate_z * 100,
        candidate_coefficients=candidate_coefficients * 100,
    )
    assert updates == 0
    assert archive_z[1].tolist() == [2.0, 2.0]
    assert archive_z[3].tolist() == [1.0, 1.0]


def test_archive_reentry_excludes_states_already_living():
    population_z = np.asarray([[1.0, 2.0]], dtype=np.float32)
    population_coefficients = np.asarray([[3.0, 4.0]], dtype=np.float32)
    archive_z = np.asarray([
        [1.0, 2.0],
        [5.0, 6.0],
        [5.0, 6.0],
    ], dtype=np.float32)
    archive_coefficients = np.asarray([
        [3.0, 4.0],
        [7.0, 8.0],
        [7.0, 8.0],
    ], dtype=np.float32)
    archive_scores = np.asarray([-0.1, -0.2, -0.3])

    selected = unique_archive_entries(
        population_z,
        population_coefficients,
        archive_z,
        archive_coefficients,
        archive_scores,
        active_targets=np.asarray([0, 1, 2]),
    )

    assert selected.tolist() == [1]


def test_quality_milestones_use_exact_candidate_prefixes():
    histories = [
        [[0, 0.8]],
        [[0, 0.9]],
    ]
    exposures = np.asarray([100, 240], dtype=np.int64)
    archive_scores = np.asarray([-0.7, -0.8])
    scores = np.full((410, 2), -1.0)
    scores[149, 0] = -0.6
    scores[399, 0] = -0.4
    scores[9, 1] = -0.75
    scores[259, 1] = -0.5

    record_quality_milestones(
        histories,
        active_targets=np.asarray([0, 1]),
        exposures=exposures,
        candidate_scores=scores,
        archive_scores=archive_scores,
        step=250,
        limit=500,
    )

    assert histories[0] == [[0, 0.8], [250, 0.6], [500, 0.4]]
    assert histories[1] == [[0, 0.9], [250, 0.75], [500, 0.5]]


def test_quality_milestones_do_not_use_candidates_after_threshold():
    histories = [[[0, 0.8]]]
    scores = np.full((200, 1), -0.9)
    scores[149, 0] = -0.7
    scores[150, 0] = -0.1

    record_quality_milestones(
        histories,
        active_targets=np.asarray([0]),
        exposures=np.asarray([100]),
        candidate_scores=scores,
        archive_scores=np.asarray([-0.8]),
        step=250,
        limit=500,
    )

    assert histories[0] == [[0, 0.8], [250, 0.7]]


def test_exposure_analysis_pairs_shared_targets_across_conditions(tmp_path):
    paths = []
    names = ["a", "b"]
    final_mse = {
        32: {3: [0.8, 0.7], 4: [0.7, 0.6]},
        64: {3: [0.6, 0.5], 4: [0.5, 0.4]},
    }
    for count in [32, 64]:
        for seed in [3, 4]:
            payload = {
                "target_count": count,
                "seed": seed,
                "target_names": names,
                "initial_records_mse": {name: 1.0 for name in names},
                "target_quality_milestones": {
                    name: [[0, 1.0], [250, 0.9], [500, final_mse[count][seed][index]]]
                    for index, name in enumerate(names)
                },
                "quality_exposure_step": 250,
                "quality_exposure_limit": 500,
            }
            path = tmp_path / f"n{count}_s{seed}.json"
            path.write_text(json.dumps(payload))
            paths.append(path)

    result = analyze(paths, anchors=2, bootstrap=1000, bootstrap_seed=7)

    assert result["target_counts"] == [32, 64]
    assert np.isclose(
        result["curves"]["32"][-1]["mean_improvement_pct"], 30.0)
    assert np.isclose(
        result["curves"]["64"][-1]["mean_improvement_pct"], 50.0)
    assert np.isclose(
        result["paired_at_limit"]["64"][
            "paired_delta_vs_reference_pct_points"],
        20.0,
    )
