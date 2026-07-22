"""Structural tests for the multi-fitness conditional decoder experiment."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from benchmarks.demo_image_species_conditional_lora import (
    ConditionalLoRAConvRGB,
    ExtraLatentConvRGB,
    LATENT,
    SHAPE,
    assimilate_conditional_decoder,
    assimilation_fraction,
    balanced_retirement_lora,
    choose_ecological_mates_within_input_species,
    decoder_input_vectors,
    distinct_target_representatives,
    fold_lora_delta,
    fold_population_mean_lora,
    lineage_succession_selection_scores,
    mating_compatibility_vectors,
    select_species_local_survivors,
)
from benchmarks.round28_anchor_conv import ConvRGB


def _theta(seed: int = 7) -> np.ndarray:
    torch.manual_seed(seed)
    net = ConvRGB(LATENT, SHAPE)
    return torch.nn.utils.parameters_to_vector(
        net.parameters()).detach().numpy().astype(np.float32)


def _source_output(theta: np.ndarray, z: torch.Tensor) -> torch.Tensor:
    source = ConvRGB(LATENT, SHAPE)
    torch.nn.utils.vector_to_parameters(
        torch.as_tensor(theta), source.parameters())
    return source(z)


def _pairwise_rms(values: np.ndarray) -> np.ndarray:
    delta = values[:, None, :] - values[None, :, :]
    return np.sqrt(np.mean(delta * delta, axis=-1))


def test_decoder_input_space_excludes_mixed_lora_gates():
    z = np.arange(3 * LATENT, dtype=np.float32).reshape(3, LATENT)
    coefficients = np.arange(24, dtype=np.float32).reshape(3, 8)

    vectors = decoder_input_vectors("mixed", z, coefficients)

    assert vectors.shape == (3, LATENT + 4)
    assert np.array_equal(vectors[:, :LATENT], z)
    assert np.array_equal(vectors[:, LATENT:], coefficients[:, :4])


def test_z_only_space_excludes_every_conditional_value():
    rng = np.random.default_rng(39)
    z = rng.standard_normal((3, LATENT)).astype(np.float32)
    coefficients = rng.standard_normal((3, 8)).astype(np.float32)
    scores = rng.standard_normal((3, 2)).astype(np.float32)

    vectors = mating_compatibility_vectors(
        "z_only", "mixed", z, coefficients, scores)

    assert vectors.shape == (3, LATENT)
    assert np.array_equal(vectors, z)


def test_decoder_input_distance_ignores_mixed_lora_gate_changes():
    rng = np.random.default_rng(41)
    z = rng.standard_normal((5, LATENT)).astype(np.float32)
    coefficients = rng.standard_normal((5, 8)).astype(np.float32)
    shifted = coefficients.copy()
    shifted[:, 4:] -= np.asarray(
        [20.0, -35.0, 11.0, 42.0], dtype=np.float32)

    before = decoder_input_vectors("mixed", z, coefficients)
    after = decoder_input_vectors("mixed", z, shifted)

    assert np.array_equal(before, after)


def test_fitness_roles_do_not_enter_decoder_input_compatibility():
    rng = np.random.default_rng(43)
    z = rng.standard_normal((4, LATENT)).astype(np.float32)
    coefficients = rng.standard_normal((4, 6)).astype(np.float32)
    scores_a = rng.standard_normal((4, 3)).astype(np.float32)
    scores_b = -10.0 * scores_a[:, ::-1]

    vectors_a = mating_compatibility_vectors(
        "decoder_input", "mixed", z, coefficients, scores_a)
    vectors_b = mating_compatibility_vectors(
        "decoder_input", "mixed", z, coefficients, scores_b)

    assert np.array_equal(vectors_a, vectors_b)
    assert not np.array_equal(
        mating_compatibility_vectors(
            "fitness", "mixed", z, coefficients, scores_a),
        mating_compatibility_vectors(
            "fitness", "mixed", z, coefficients, scores_b),
    )


def test_transitive_input_species_can_mix_distant_ecological_mates():
    # A--B--C is connected even though A and C are not directly
    # compatible. Fitness makes C the only ecological mate available to A.
    inputs = np.asarray([[0.0], [0.9], [1.8]], dtype=np.float32)
    fitness = np.asarray([[0.0], [10.0], [0.1]], dtype=np.float32)

    mates, distances = choose_ecological_mates_within_input_species(
        inputs,
        fitness,
        parents=np.asarray([0]),
        compatibility_radius=1.0,
        fitness_radius=0.2,
        rng=np.random.default_rng(47),
    )

    assert mates.tolist() == [2]
    assert distances[0] == pytest.approx(1.8)


def test_zero_reproduction_weight_excludes_senescent_mate():
    inputs = np.asarray([[0.0], [0.1], [0.2]], dtype=np.float32)
    fitness = np.asarray([[0.0], [0.05], [0.1]], dtype=np.float32)

    mates, _ = choose_ecological_mates_within_input_species(
        inputs,
        fitness,
        parents=np.asarray([0]),
        compatibility_radius=1.0,
        fitness_radius=1.0,
        rng=np.random.default_rng(49),
        mate_weights=np.asarray([1.0, 0.0, 0.25]),
    )

    assert mates.tolist() == [2]


def test_species_local_child_replaces_closest_redundant_role():
    parent_scores = np.asarray([
        [10.0, 0.0],
        [0.0, 10.0],
        [1.0, 0.0],
    ])
    child_scores = np.asarray([[2.0, 0.0]])

    keep, roles, target_children, diagnostics = (
        select_species_local_survivors(
            parent_scores,
            child_scores,
            child_priority=np.asarray([1.0]),
            parent_goals=np.asarray([0, 1, 0]),
            child_goals=np.asarray([0]),
            parent_compatibility=np.asarray([[0.0], [100.0], [0.9]]),
            child_compatibility=np.asarray([[1.0]]),
            compatibility_radius=1.0,
            survivor_count=3,
        ))

    assert set(keep.tolist()) == {0, 1, 3}
    assert roles.tolist() == [0, 1, 0]
    assert target_children == 0
    assert diagnostics["local_replacements"] == 1
    assert diagnostics["forced_nonlocal_replacements"] == 0


def test_distinct_target_representatives_protect_hard_targets_first():
    scores = np.asarray([
        [10.0, 10.0],
        [9.0, 1.0],
        [1.0, 9.0],
    ])

    representatives = distinct_target_representatives(
        scores, target_order=np.asarray([1, 0]))

    assert representatives.tolist() == [0, 1]
    assert len(np.unique(representatives)) == scores.shape[1]


def test_lineage_succession_restricts_retired_role_to_descendants():
    parent_scores = np.asarray([[10.0, 0.0], [0.0, 10.0]])
    child_scores = np.asarray([
        [8.0, 1.0],
        [9.0, 2.0],
        [7.0, 3.0],
    ])

    parents, children, expired, diagnostics = (
        lineage_succession_selection_scores(
            parent_scores,
            child_scores,
            parent_goals=np.asarray([0, 1]),
            parent_age=np.asarray([10, 2]),
            child_parents=np.asarray([0, 1, 1]),
            child_mates=np.asarray([-1, -1, -1]),
            retirement_age=10,
        ))

    assert expired.tolist() == [True, False]
    assert np.isneginf(parents[:, 0]).all()
    assert np.isfinite(children[0, 0])
    assert np.isneginf(children[1:, 0]).all()
    assert np.array_equal(children[:, 1], child_scores[:, 1])
    assert diagnostics["lineage_succession_targets"] == 1
    assert diagnostics["lineage_retirements"] == 1


def test_species_local_child_cannot_kill_across_disconnected_species():
    parent_scores = np.asarray([
        [10.0, 0.0],
        [0.0, 10.0],
        [1.0, 0.0],
    ])

    keep, _, _, diagnostics = select_species_local_survivors(
        parent_scores,
        child_scores=np.asarray([[2.0, 0.0]]),
        child_priority=np.asarray([1.0]),
        parent_goals=np.asarray([0, 1, 0]),
        child_goals=np.asarray([0]),
        parent_compatibility=np.asarray([[0.0], [100.0], [0.9]]),
        child_compatibility=np.asarray([[50.0]]),
        compatibility_radius=1.0,
        survivor_count=3,
    )

    assert set(keep.tolist()) == {0, 1, 2}
    assert diagnostics["local_replacements"] == 0
    assert diagnostics["rejected_children"] == 1


def test_protected_target_child_can_found_a_disconnected_species():
    parent_scores = np.asarray([
        [10.0, 0.0],
        [0.0, 10.0],
        [1.0, 0.0],
    ])

    keep, roles, target_children, diagnostics = (
        select_species_local_survivors(
            parent_scores,
            child_scores=np.asarray([[11.0, 0.0]]),
            child_priority=np.asarray([1.0]),
            parent_goals=np.asarray([0, 1, 0]),
            child_goals=np.asarray([0]),
            parent_compatibility=np.asarray([[0.0], [100.0], [0.9]]),
            child_compatibility=np.asarray([[50.0]]),
            compatibility_radius=1.0,
            survivor_count=3,
        ))

    assert 3 in keep
    assert roles[0] == 0
    assert target_children == 1
    assert diagnostics["protected_child_replacements"] == 1
    assert diagnostics["forced_nonlocal_replacements"] == 1


def test_conditional_lora_zero_coefficients_equal_source_backbone():
    theta = _theta()
    torch.manual_seed(11)
    z = torch.randn(2, LATENT)
    model = ConditionalLoRAConvRGB(8)
    model.initialize_backbone(theta)

    actual = model(z, torch.zeros(2, 8))
    expected = _source_output(theta, z)

    assert torch.equal(actual, expected)


def test_conditional_lora_coefficients_change_the_shared_function():
    theta = _theta()
    torch.manual_seed(13)
    z = torch.randn(2, LATENT)
    model = ConditionalLoRAConvRGB(8)
    model.initialize_backbone(theta)

    base = model(z, torch.zeros(2, 8))
    adapted = model(z, torch.ones(2, 8))

    assert not torch.equal(base, adapted)
    assert float((base - adapted).abs().mean().detach()) > 0.0


def test_extra_latent_zero_values_equal_source_backbone():
    theta = _theta()
    torch.manual_seed(17)
    z = torch.randn(2, LATENT)
    model = ExtraLatentConvRGB(8)
    model.initialize_backbone(theta)

    actual = model(z, torch.zeros(2, 8))
    expected = _source_output(theta, z)

    assert torch.equal(actual, expected)


def test_zero_conditional_values_receive_first_order_signal():
    theta = _theta()
    torch.manual_seed(19)
    z = torch.randn(2, LATENT)
    for model, width in (
            (ConditionalLoRAConvRGB(8), 8),
            (ExtraLatentConvRGB(8), 8)):
        model.initialize_backbone(theta)
        conditional = torch.zeros(2, width, requires_grad=True)
        model(z, conditional).square().mean().backward()

        assert conditional.grad is not None
        assert float(conditional.grad.abs().sum()) > 0.0


def test_mixed_zero_state_matches_backbone_and_both_halves_are_active():
    theta = _theta()
    torch.manual_seed(23)
    z = torch.randn(2, LATENT)
    model = ConditionalLoRAConvRGB(4, extra_latent_dim=4)
    model.initialize_backbone(theta)
    conditional = torch.zeros(2, 8, requires_grad=True)

    actual = model(z, conditional)
    expected = _source_output(theta, z)
    actual.square().mean().backward()

    assert torch.equal(actual, expected)
    assert conditional.grad is not None
    assert float(conditional.grad[:, :4].abs().sum()) > 0.0
    assert float(conditional.grad[:, 4:].abs().sum()) > 0.0
    assert model.extra_latent_dim == model.lora_dim == 4
    assert model.coefficient_dim == 8


def test_conditional_arms_have_equal_individual_state_dimensions():
    coefficient_dim = 32
    lora = ConditionalLoRAConvRGB(coefficient_dim)
    latent = ExtraLatentConvRGB(coefficient_dim)

    assert lora.coefficient_dim == latent.extra_dim == coefficient_dim
    assert LATENT + lora.coefficient_dim == LATENT + latent.extra_dim


def test_assimilation_fraction_is_bounded_and_debt_driven():
    assert assimilation_fraction(0.0, 0.01, 1e-4) == 0.0
    low = assimilation_fraction(1e-5, 0.01, 1e-4)
    high = assimilation_fraction(1e-3, 0.01, 1e-4)

    assert 0.0 < low < high < 0.01
    with pytest.raises(ValueError):
        assimilation_fraction(-1.0, 0.01, 1e-4)
    with pytest.raises(ValueError):
        assimilation_fraction(1.0, 1.0, 1e-4)


def test_dynamic_assimilation_shrinks_personal_state_and_updates_shared_model():
    theta = _theta()
    torch.manual_seed(29)
    z = torch.randn(3, LATENT)
    model = ConditionalLoRAConvRGB(4, extra_latent_dim=4)
    model.initialize_backbone(theta)
    coefficients = torch.randn(3, 8)
    with torch.no_grad():
        teachers = torch.sigmoid(model(z, coefficients)).reshape(
            3, *SHAPE).numpy()
    before = [parameter.detach().clone() for parameter in model.parameters()]
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    assimilated, diagnostics = assimilate_conditional_decoder(
        model,
        optimizer,
        z.numpy(),
        coefficients.numpy(),
        teachers,
        maximum_fraction=0.02,
        debt_scale=1e-8,
        steps=2,
        base_only_weight=1.0,
        device="cpu",
    )

    fraction = diagnostics["assimilation_fraction"]
    assert 0.0 < fraction < 0.02
    assert np.allclose(
        assimilated, coefficients.numpy() * (1.0 - fraction))
    assert diagnostics["phenotype_debt_before"] > 0.0
    assert diagnostics["adapted_phenotype_mse"] >= 0.0
    assert any(not torch.equal(old, new) for old, new in
               zip(before, model.parameters()))


def test_population_mean_lora_fold_is_function_preserving_and_centers_gates():
    theta = _theta()
    torch.manual_seed(31)
    z = torch.randn(5, LATENT)
    model = ConditionalLoRAConvRGB(4, extra_latent_dim=4)
    model.initialize_backbone(theta)
    coefficients = torch.randn(5, 8).numpy().astype(np.float32)
    with torch.no_grad():
        before = model(z, torch.as_tensor(coefficients)).clone()
    extra_before = coefficients[:, :4].copy()

    centered, diagnostics = fold_population_mean_lora(model, coefficients)
    with torch.no_grad():
        after = model(z, torch.as_tensor(centered))

    assert torch.allclose(before, after, atol=2e-5, rtol=2e-5)
    assert np.array_equal(centered[:, :4], extra_before)
    assert np.allclose(centered[:, 4:].mean(axis=0), 0.0, atol=1e-7)
    assert diagnostics["folded_lora_mean_rms"] > 0.0


def test_arbitrary_lora_legacy_fold_preserves_all_living_functions():
    theta = _theta()
    torch.manual_seed(37)
    z = torch.randn(6, LATENT)
    model = ConditionalLoRAConvRGB(4, extra_latent_dim=4)
    model.initialize_backbone(theta)
    coefficients = torch.randn(6, 8).numpy().astype(np.float32)
    delta = np.asarray([0.5, -0.25, 0.75, 0.125], dtype=np.float32)
    with torch.no_grad():
        before = model(z, torch.as_tensor(coefficients)).clone()

    shifted, diagnostics = fold_lora_delta(model, coefficients, delta)
    with torch.no_grad():
        after = model(z, torch.as_tensor(shifted))

    assert torch.allclose(before, after, atol=2e-5, rtol=2e-5)
    assert np.array_equal(shifted[:, :4], coefficients[:, :4])
    assert np.allclose(shifted[:, 4:], coefficients[:, 4:] - delta)
    assert diagnostics["folded_lora_mean_rms"] > 0.0


def test_retirement_legacy_weights_within_niches_then_balances_niches():
    # Niche zero has two retirees; niche one has only one. Equal utilities
    # make niche zero average internally, then the two niches get equal votes.
    coefficients = np.asarray([
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 3.0, 0.0],
        [0.0, 0.0, 0.0, 4.0],
    ], dtype=np.float32)
    delta, weights, utility, diagnostics = balanced_retirement_lora(
        coefficients,
        goals=np.asarray([0, 0, 1]),
        relative_fitness=np.zeros(3),
        coverage_margin=np.zeros(3),
        lifetime_success=np.zeros(3),
        lora_start=2,
        target_count=2,
        merge_fraction=0.5,
        temperature=1.0,
    )

    assert np.allclose(weights, [0.5, 0.5, 1.0])
    assert np.allclose(utility, 0.0)
    assert np.allclose(delta, [0.5, 1.0])
    assert diagnostics["retired_parents"] == 3
    assert diagnostics["legacy_niches"] == 2


def test_retirement_relative_fitness_controls_weight_inside_one_niche():
    coefficients = np.asarray([
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 5.0, 0.0],
    ], dtype=np.float32)
    delta, weights, _, _ = balanced_retirement_lora(
        coefficients,
        goals=np.asarray([0, 0]),
        relative_fitness=np.asarray([0.0, 4.0]),
        coverage_margin=np.zeros(2),
        lifetime_success=np.zeros(2),
        lora_start=2,
        target_count=1,
        merge_fraction=1.0,
        temperature=1.0,
    )

    assert weights[1] > 0.98
    assert delta[0] > 4.9
