"""Pure-NumPy invariants for the emergent-species mating rule."""

from __future__ import annotations

import numpy as np

from benchmarks.demo_clip_species_vector import (
    choose_compatible_mates,
    compatibility_graph,
    connected_components,
    negative_weight_at,
    normalize_species_vectors,
)
from benchmarks.demo_image_species_vector import (
    assigned_role_fitness,
    inherit_decoder_references,
    local_behavior_density,
    mirrored_decoder_candidate,
    most_encountered_decoder_pair,
    pairwise_negative_mse,
    preserve_decoder_lineages,
    select_target_covered_survivors,
    shared_population_from_codes,
    sparse_decoder_perturbation,
    target_coverage_fitness,
    update_decoder_stagnation,
    update_individual_step_state,
)


def test_species_vectors_are_normalized_per_prompt():
    scores = np.array([
        [1.0, 100.0],
        [2.0, 100.0],
        [3.0, 100.0],
    ])
    vectors = normalize_species_vectors(scores)
    assert np.allclose(vectors[:, 0].mean(), 0.0)
    assert np.allclose(vectors[:, 0].std(), 1.0)
    assert np.all(vectors[:, 1] == 0.0)  # no spread = no species signal


def test_gene_flow_is_transitive_without_distant_mating():
    # A-B and B-C are compatible; A-C are not. The graph still has one
    # component, which is the stepping-stone gene-flow behavior we want.
    vectors = np.array([[0.0], [0.6], [1.2]])
    adjacency, _ = compatibility_graph(vectors, radius=0.7)
    assert adjacency[0, 1]
    assert adjacency[1, 2]
    assert not adjacency[0, 2]
    assert connected_components(adjacency) == [[0, 1, 2]]


def test_mates_never_cross_the_radius_and_isolates_are_asexual():
    vectors = np.array([[0.0], [0.5], [3.0]])
    parents = np.array([0, 1, 2])
    mates, distances = choose_compatible_mates(
        vectors, parents, radius=0.75, rng=np.random.default_rng(0))
    assert mates.tolist() == [1, 0, -1]
    assert np.all(distances[:2] <= 0.75)
    assert np.isnan(distances[2])


def test_negative_pressure_ramps_then_stays_full_strength():
    values = [negative_weight_at(e, 1000, 0.1, 1.0, 0.5)
              for e in (0, 250, 500, 1000)]
    assert np.allclose(values, [0.1, 0.55, 1.0, 1.0])


def test_pairwise_negative_mse_matches_explicit_pixel_differences():
    import torch

    phenotypes = torch.tensor([
        [[[0.0, 1.0]]],
        [[[0.5, 0.5]]],
    ])
    targets = torch.tensor([
        [[[0.0, 1.0]]],
        [[[1.0, 0.0]]],
    ])
    actual = pairwise_negative_mse(phenotypes, targets).numpy()
    expected = np.array([[0.0, -1.0], [-0.25, -0.25]])
    assert np.allclose(actual, expected)


def test_target_coverage_uses_distinct_dynamic_representatives():
    parents = np.array([
        [0.8, 0.7, 0.0],
        [0.7, 0.6, 0.1],
        [0.0, 0.0, 0.8],
    ])
    children = np.array([
        [0.9, 0.95, 0.0],  # best for 0 and 1; can fill only one target seat
        [0.85, 0.5, 0.0],
        [0.0, 0.0, 0.9],
        [0.1, 0.1, 0.1],
    ])
    priority = np.array([0.0, 0.0, 0.0, 10.0])
    child_goals = np.array([1, 0, 2, 2])
    keep, roles, target_children = select_target_covered_survivors(
        parents, children, priority, child_goals, survivor_count=4,
        target_order=np.array([1, 0, 2]),
    )

    assert len(set(keep.tolist())) == 4
    assert roles[:3].tolist() == [1, 0, 2]
    assert set(roles[:3]) == {0, 1, 2}
    assert target_children == 3
    # The remaining seat follows progress/rarity priority, not a fixed label.
    assert keep[-1] == len(parents) + 3
    assert roles[-1] == child_goals[3]


def test_shared_population_has_one_exact_decoder_and_covers_codes():
    codes = np.arange(3 * 64, dtype=np.float32).reshape(3, 64)
    theta = np.linspace(-1, 1, 11, dtype=np.float32)
    population_z, population_theta, roles = shared_population_from_codes(
        codes, theta, survivor_count=5, rng=np.random.default_rng(7))

    assert np.array_equal(population_z[:3], codes)
    assert roles.tolist() == [0, 1, 2, 0, 1]
    assert population_theta.shape == (5, 11)
    assert np.array_equal(population_theta, np.repeat(theta[None], 5, axis=0))


def test_decoder_reference_comes_from_locally_fitter_parent_without_copying():
    scores = np.array([
        [0.8, 0.1],
        [0.9, 0.0],
        [0.0, 0.7],
    ])
    parents = np.array([0, 2, 1])
    mates = np.array([1, 0, -1])
    inherited_goals = np.array([0, 1, 0])
    decoder_ids = np.array([101, 202, 303])

    inherited, from_mate = inherit_decoder_references(
        parents, mates, inherited_goals, scores, decoder_ids)

    # Mate 1 is better than parent 0 on target 0. Parent 2 remains better
    # than mate 0 on target 1. The asexual child keeps its parent's ID.
    assert inherited.tolist() == [202, 303, 202]
    assert from_mate.tolist() == [True, False, False]
    assert set(inherited).issubset(set(decoder_ids))


def test_decoder_lineage_selection_prefers_children_and_preserves_every_id():
    parent_scores = np.array([
        [0.9, 0.0],
        [0.8, 0.1],
        [0.0, 0.9],
    ])
    child_scores = np.array([
        [0.95, 0.0],
        [0.85, 0.1],
        [0.0, 0.8],
    ])
    parent_ids = np.array([10, 20, 30])
    child_ids = np.array([10, 20, 30])
    # Ordinary selection kept two copies of decoder 10 and one of decoder 30,
    # dropping decoder 20 even though it has an inheriting child available.
    keep = np.array([3, 5, 0])
    roles = np.array([0, 1, 0])

    adjusted, adjusted_roles, successors, reprieved = (
        preserve_decoder_lineages(
            keep, roles,
            parent_scores, child_scores,
            child_priority=np.array([3.0, 2.0, 1.0]),
            parent_goals=np.array([0, 0, 1]),
            child_goals=np.array([0, 0, 1]),
            parent_decoder_ids=parent_ids,
            child_decoder_ids=child_ids,
            target_count=2,
        ))

    combined_ids = np.concatenate([parent_ids, child_ids])
    assert set(combined_ids[adjusted]) == {10, 20, 30}
    assert adjusted[-1] == 4  # decoder 20's child, not its current carrier
    assert adjusted_roles.tolist() == [0, 1, 0]
    assert successors == 1
    assert reprieved == 0


def test_decoder_lineage_selection_reprieves_parent_without_a_child():
    adjusted, _, successors, reprieved = preserve_decoder_lineages(
        keep=np.array([2, 3]),
        roles=np.array([0, 0]),
        parent_scores=np.array([[0.9], [0.8]]),
        child_scores=np.array([[1.0], [0.95]]),
        child_priority=np.array([2.0, 1.0]),
        parent_goals=np.array([0, 0]),
        child_goals=np.array([0, 0]),
        parent_decoder_ids=np.array([10, 20]),
        child_decoder_ids=np.array([10, 10]),
        target_count=1,
    )

    combined_ids = np.array([10, 20, 10, 10])
    assert set(combined_ids[adjusted]) == {10, 20}
    assert adjusted[-1] == 1
    assert successors == 0
    assert reprieved == 1


def test_individual_success_and_stagnation_controls_are_separate():
    parents = np.repeat(np.arange(2), 10)
    child_wins = np.array([True] * 5 + [False] * 5 + [False] * 10)
    result = update_individual_step_state(
        gains=np.ones(2),
        success_wins=np.zeros(2, dtype=np.int64),
        success_attempts=np.zeros(2, dtype=np.int64),
        stagnation_attempts=np.zeros(2, dtype=np.int64),
        parents=parents,
        child_wins=child_wins,
        mode="hybrid",
        win_target=0.2,
        gain_step=1.15,
        success_window=10,
        stagnation_limit=10,
        stagnation_kick=3.0,
        gain_limits=(0.25, 4.0),
    )
    gains, wins, attempts, stagnation, updates, kicks = result

    assert np.allclose(gains, [1.15, 3.0 / 1.15])
    assert wins.tolist() == [0, 0]
    assert attempts.tolist() == [0, 0]
    assert stagnation.tolist() == [0, 0]
    assert updates == 2
    assert kicks == 1


def test_individual_success_control_waits_for_enough_evidence():
    state = (
        np.ones(1),
        np.zeros(1, dtype=np.int64),
        np.zeros(1, dtype=np.int64),
        np.zeros(1, dtype=np.int64),
    )
    first = update_individual_step_state(
        *state,
        parents=np.zeros(5, dtype=np.int64),
        child_wins=np.array([True, False, False, False, False]),
        mode="success", win_target=0.2, gain_step=1.15,
        success_window=10, stagnation_limit=10, stagnation_kick=3.0,
        gain_limits=(0.25, 4.0),
    )
    assert first[0].tolist() == [1.0]
    assert first[2].tolist() == [5]
    assert first[4] == 0

    second = update_individual_step_state(
        *first[:4],
        parents=np.zeros(5, dtype=np.int64),
        child_wins=np.array([True, True, False, False, False]),
        mode="success", win_target=0.2, gain_step=1.15,
        success_window=10, stagnation_limit=10, stagnation_kick=3.0,
        gain_limits=(0.25, 4.0),
    )
    assert np.allclose(second[0], [1.15])
    assert second[1].tolist() == [0]
    assert second[2].tolist() == [0]
    assert second[4] == 1


def test_sparse_decoder_probe_changes_only_a_subset_at_weight_scale():
    theta = np.linspace(-2.0, 2.0, 1000, dtype=np.float32)
    perturbation = sparse_decoder_perturbation(
        theta, fraction=0.1, sigma=0.02,
        rng=np.random.default_rng(12))

    changed = perturbation != 0
    assert 60 < changed.sum() < 140
    assert np.allclose(
        np.abs(perturbation[changed]), 0.02 * theta.std())


def test_mirrored_decoder_candidate_moves_toward_fitter_temporary_side():
    theta = np.array([1.0, -2.0], dtype=np.float32)
    perturbation = np.array([0.2, 0.0], dtype=np.float32)
    plus_fitness = -np.square(theta + perturbation).sum()
    minus_fitness = -np.square(theta - perturbation).sum()

    candidate = mirrored_decoder_candidate(
        theta, perturbation, plus_fitness, minus_fitness,
        step_fraction=0.25)

    assert np.allclose(candidate, [0.95, -2.0])
    assert -np.square(candidate).sum() > -np.square(theta).sum()


def test_assigned_role_fitness_uses_each_carriers_current_role():
    scores = np.array([
        [0.9, 0.1, 0.2],
        [0.3, 0.8, 0.4],
        [0.5, 0.6, 0.7],
    ])
    mean, worst = assigned_role_fitness(scores, np.array([0, 1, 2]))

    assert np.isclose(mean, 0.8)
    assert np.isclose(worst, 0.7)


def test_target_coverage_fitness_uses_best_available_individual_per_target():
    mean, worst = target_coverage_fitness(np.array([
        [0.9, 0.1, 0.4],
        [0.2, 0.8, 0.5],
        [0.3, 0.6, 0.7],
    ]))

    assert np.isclose(mean, 0.8)
    assert np.isclose(worst, 0.7)


def test_local_behavior_density_counts_transitive_neighbors_not_components():
    densities = local_behavior_density(
        np.array([[0.0], [0.6], [1.2]]), radius=0.7)

    # A-B and B-C are local neighbors, while A-C remain distant.
    assert densities.tolist() == [2, 3, 2]


def test_decoder_stagnation_resets_only_when_lineage_fitness_improves():
    best, stagnant = update_decoder_stagnation(
        scores=np.array([[0.8, 0.1], [0.7, 0.2], [0.1, 0.9]]),
        goals=np.array([0, 0, 1]),
        decoder_ids=np.array([10, 10, 20]),
        best_fitness={10: 0.8, 20: 0.8},
        stagnation={10: 3, 20: 4},
    )

    assert np.isclose(best[10], 0.8)
    assert stagnant[10] == 4
    assert np.isclose(best[20], 0.9)
    assert stagnant[20] == 0


def test_local_decoder_merge_pair_requires_active_cross_lineage_encounters():
    decoder_ids = np.array([10, 20, 20, 30])
    parents = np.array([0, 0, 1, 2, 3])
    mates = np.array([1, 2, 0, 3, -1])

    pair, encounters = most_encountered_decoder_pair(
        parents, mates, decoder_ids, active_decoder_ids={10, 20, 30})
    assert pair == (10, 20)
    assert encounters == 3

    # Once decoder 10 is extinct, its historical encounters cannot trigger
    # a merger; the remaining 20/30 encounter is selected instead.
    pair, encounters = most_encountered_decoder_pair(
        parents, mates, decoder_ids, active_decoder_ids={20, 30})
    assert pair == (20, 30)
    assert encounters == 1


def test_stagnant_decoder_can_win_priority_among_local_merge_pairs():
    decoder_ids = np.array([10, 20, 30])
    parents = np.array([0, 0, 0, 1])
    mates = np.array([1, 1, 1, 2])

    pair, encounters = most_encountered_decoder_pair(
        parents, mates, decoder_ids, active_decoder_ids={10, 20, 30},
        decoder_stagnation={10: 0, 20: 0, 30: 50},
        stagnation_weight=0.1,
    )

    assert pair == (20, 30)
    assert encounters == 1

    pair, encounters = most_encountered_decoder_pair(
        parents, mates, decoder_ids, active_decoder_ids={10, 20, 30},
        decoder_stagnation={10: 0, 20: 0, 30: 50},
        stagnation_weight=0.1, stagnation_grace=50,
    )
    assert pair == (10, 20)
    assert encounters == 3
