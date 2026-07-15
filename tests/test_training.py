import numpy as np
import pytest
import torch

from latentspace import (
    AdaptiveMixtureTrainer,
    AdvantageWeightedTrainer,
    BacktrackingTrainer,
    ContrastiveTrainer,
    Evolver,
    FrozenTrainer,
    GuardedTrainer,
    LatentIndividual,
    MLPDecoder,
    MixtureTrainer,
    PermutationTrainer,
    PolicyGradientTrainer,
    DecoderTrainer,
    Decoder,
)


def ranked_population(size=8, latent=4):
    population = []
    for index in range(size):
        individual = LatentIndividual(
            np.full(latent, index / size, dtype=np.float32)
        )
        individual.fitness = float(size - index)
        population.append(individual)
    return population


def test_frozen_trainer_never_changes_decoder_version():
    evolver = Evolver(
        lambda phenotypes: phenotypes[:, 0],
        output_shape=(1,),
        latent=4,
        population=8,
        hidden_size=8,
        families=1,
        children=1,
        refine_every=1,
        trainer=FrozenTrainer(),
    )

    evolver.solve(3, verbose_every=0)

    assert evolver.decoder.version == 0
    assert {individual.evaluated_at for individual in evolver.env.population} == {0}


def test_contrastive_trainer_updates_mapping_and_version():
    torch.manual_seed(3)
    decoder = MLPDecoder(4, (3,), hidden_size=8, lr=1e-2)
    before = [parameter.detach().clone() for parameter in decoder.parameters()]

    loss = ContrastiveTrainer(percent=0.5).step(
        decoder, ranked_population()
    )

    assert np.isfinite(loss)
    assert decoder.version == 1
    assert any(
        not torch.equal(old, new)
        for old, new in zip(before, decoder.parameters())
    )


def test_policy_gradient_counts_samples_and_updates_mapping():
    torch.manual_seed(4)
    decoder = MLPDecoder(4, (3,), hidden_size=8, lr=1e-2)
    calls = []

    def fitness(phenotypes):
        calls.append(len(phenotypes))
        return -phenotypes.square().mean(dim=1)

    trainer = PolicyGradientTrainer(
        percent=0.5, samples_per_gene=3, exploration_std=0.2
    )
    loss = trainer.step(decoder, ranked_population(), fitness_fn=fitness)

    assert np.isfinite(loss)
    assert calls == [12]
    assert decoder.version == 1


def test_policy_gradient_requires_a_fitness_function():
    decoder = MLPDecoder(4, (3,), hidden_size=8)

    with pytest.raises(ValueError, match="requires fitness_fn"):
        PolicyGradientTrainer().step(decoder, ranked_population())


def test_advantage_weighted_counts_samples_and_updates_mapping():
    torch.manual_seed(5)
    decoder = MLPDecoder(4, (3,), hidden_size=8, lr=1e-2)
    calls = []

    def fitness(phenotypes):
        calls.append(len(phenotypes))
        return -phenotypes.square().mean(dim=1)

    loss = AdvantageWeightedTrainer(
        percent=0.5, samples_per_gene=3, exploration_std=0.2
    ).step(decoder, ranked_population(), fitness_fn=fitness)

    assert np.isfinite(loss)
    assert calls == [12]
    assert decoder.version == 1


def test_permutation_canonicalization_ignores_rotation_and_reversal():
    forward = np.array([0.0, 0.2, 0.4, 0.6], dtype=np.float32)
    rotated = np.array([0.4, 0.6, 0.0, 0.2], dtype=np.float32)
    reversed_route = np.array([0.0, 0.6, 0.4, 0.2], dtype=np.float32)

    canonical = PermutationTrainer.canonical_route(forward)

    # These keys encode the same cycle after translating city labels back to
    # the route itself: 0-1-2-3 and 0-3-2-1.
    assert canonical == (0, 1, 2, 3)
    assert PermutationTrainer.canonical_route(reversed_route) == canonical
    assert PermutationTrainer.canonical_route(rotated) == (0, 1, 2, 3)


def test_permutation_trainer_updates_mapping_and_version():
    torch.manual_seed(6)
    decoder = MLPDecoder(4, (5,), hidden_size=8, lr=1e-2)
    before = [parameter.detach().clone() for parameter in decoder.parameters()]

    loss = PermutationTrainer(
        percent=0.5,
        anchor_weight=1.0,
        anchor_percent=0.25,
    ).step(decoder, ranked_population())

    assert np.isfinite(loss)
    assert decoder.version == 1
    assert any(
        not torch.equal(old, new)
        for old, new in zip(before, decoder.parameters())
    )


class RecordingTrainer(DecoderTrainer):
    def __init__(self, name, calls):
        self.name = name
        self.calls = calls

    def step(self, decoder, sorted_pop, fitness_fn=None):
        self.calls.append(self.name)
        return 1.0


def recording_components(calls):
    return {
        name: RecordingTrainer(name, calls)
        for name in ("first", "second", "third")
    }


def test_shuffled_mixture_uses_every_objective_before_repeating():
    calls = []
    trainer = MixtureTrainer(
        recording_components(calls), strategy="shuffled_cycle", seed=7
    )

    for _ in range(3):
        trainer.step(None, [])

    assert set(calls) == {"first", "second", "third"}
    assert trainer.history == calls
    assert trainer.selection_counts == {"first": 1, "second": 1, "third": 1}


def test_random_mixture_is_reproducible_and_can_run_multiple_microbatches():
    first_calls, second_calls = [], []
    first = MixtureTrainer(
        recording_components(first_calls),
        strategy="random",
        steps_per_call=3,
        seed=11,
    )
    second = MixtureTrainer(
        recording_components(second_calls),
        strategy="random",
        steps_per_call=3,
        seed=11,
    )

    first.step(None, [])
    second.step(None, [])

    assert len(first_calls) == 3
    assert first_calls == second_calls


class SetAllParametersTrainer(DecoderTrainer):
    def __init__(self, value):
        self.value = value

    def step(self, decoder, sorted_pop, fitness_fn=None):
        with torch.no_grad():
            for parameter in decoder.parameters():
                parameter.fill_(self.value)
        decoder.mark_updated()
        return 0.0


def evaluated_population(decoder, size=4):
    population = ranked_population(size=size)
    phenotypes = decoder.decode(np.stack([item.genes for item in population]))
    fitnesses = phenotypes[:, 0].cpu().tolist()
    for individual, fitness in zip(population, fitnesses):
        individual.fitness = fitness
        individual.evaluated_at = decoder.version
    return sorted(population, key=lambda individual: individual.fitness, reverse=True)


def test_guarded_trainer_rolls_back_a_worse_update_and_restamps_cache():
    torch.manual_seed(8)
    decoder = MLPDecoder(4, (1,), hidden_size=8, lr=1e-2)
    population = evaluated_population(decoder)
    before = [parameter.detach().clone() for parameter in decoder.parameters()]
    trainer = GuardedTrainer(SetAllParametersTrainer(-10.0), probe_percent=0.5)

    trainer.step(
        decoder,
        population,
        fitness_fn=lambda phenotypes: phenotypes[:, 0],
    )

    assert trainer.acceptance_history == [False]
    assert all(
        torch.equal(old, new)
        for old, new in zip(before, decoder.parameters())
    )
    assert {individual.evaluated_at for individual in population} == {1}


def test_guarded_trainer_keeps_a_new_best_despite_lower_probe_mean():
    torch.manual_seed(9)
    decoder = MLPDecoder(4, (1,), hidden_size=8, lr=1e-2)
    population = evaluated_population(decoder)
    for individual in population:
        individual.fitness = 0.5
    trainer = GuardedTrainer(SetAllParametersTrainer(-10.0), probe_percent=0.5)

    trainer.step(
        decoder,
        population,
        fitness_fn=lambda phenotypes: torch.tensor(
            [0.9, 0.0], device=phenotypes.device
        ),
    )

    assert trainer.acceptance_history == [True]
    assert max(individual.fitness for individual in population[:2]) == pytest.approx(0.9)


class BiasDecoder(Decoder):
    def __init__(self):
        super().__init__(input_length=4, output_shape=(1,))
        self.bias = torch.nn.Parameter(torch.tensor(0.0))
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.1)
        self.opt = self.optimizer

    def forward(self, inputs):
        amount = len(inputs)
        return self.bias.expand(amount, 1)

    def decode(self, genes_batch):
        with torch.no_grad():
            return self.forward(genes_batch)


class AddBiasTrainer(DecoderTrainer):
    def __init__(self, delta):
        self.delta = delta

    def step(self, decoder, sorted_pop, fitness_fn=None):
        with torch.no_grad():
            decoder.bias.add_(self.delta)
        decoder.mark_updated()
        return 0.0


def test_backtracking_trainer_keeps_a_smaller_helpful_step():
    decoder = BiasDecoder()
    population = ranked_population(size=4)
    for individual in population:
        individual.fitness = -(0.0 - 0.4) ** 2
        individual.evaluated_at = decoder.version
    trainer = BacktrackingTrainer(
        AddBiasTrainer(1.0),
        probe_percent=0.5,
        factors=(1.0, 0.5, 0.25),
    )

    trainer.step(
        decoder,
        population,
        fitness_fn=lambda phenotypes: -(phenotypes[:, 0] - 0.4).square(),
    )

    assert decoder.bias.item() == pytest.approx(0.5)
    assert trainer.factor_history == [0.5]
    assert trainer.probe_evaluations == 4
    assert {individual.evaluated_at for individual in population[:2]} == {1}


def test_adaptive_mixture_allocates_more_steps_to_the_helpful_objective():
    decoder = BiasDecoder()
    population = evaluated_population(decoder)
    trainer = AdaptiveMixtureTrainer(
        {
            "helpful": AddBiasTrainer(1.0),
            "harmful": AddBiasTrainer(-1.0),
        },
        probe_percent=1.0,
        steps_per_call=30,
        exploration=0.1,
        seed=4,
    )

    trainer.step(
        decoder,
        population,
        fitness_fn=lambda phenotypes: phenotypes[:, 0],
    )

    assert trainer.selection_counts["helpful"] > trainer.selection_counts["harmful"]
    assert trainer.probabilities["helpful"] > trainer.probabilities["harmful"]
    assert len(trainer.reward_history) == 30
    assert {individual.evaluated_at for individual in population} == {30}


def test_adaptive_mixture_changes_preference_when_rewards_switch():
    decoder = BiasDecoder()
    population = evaluated_population(decoder)
    trainer = AdaptiveMixtureTrainer(
        {
            "increase": AddBiasTrainer(1.0),
            "decrease": AddBiasTrainer(-1.0),
        },
        probe_percent=1.0,
        steps_per_call=30,
        exploration=0.1,
        seed=4,
    )

    trainer.step(
        decoder, population, fitness_fn=lambda phenotypes: phenotypes[:, 0]
    )
    first_probabilities = trainer.probabilities
    trainer.step(
        decoder, population, fitness_fn=lambda phenotypes: -phenotypes[:, 0]
    )
    second_probabilities = trainer.probabilities

    assert first_probabilities["increase"] > first_probabilities["decrease"]
    assert second_probabilities["decrease"] > second_probabilities["increase"]


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS is unavailable"
)
@pytest.mark.parametrize(
    "trainer",
    [
        PolicyGradientTrainer(percent=0.5, samples_per_gene=2),
        AdvantageWeightedTrainer(percent=0.5, samples_per_gene=2),
    ],
)
def test_trainers_execute_decoder_updates_on_mps(trainer):
    decoder = MLPDecoder(4, (3,), hidden_size=8, lr=1e-2, device="mps")

    trainer.step(
        decoder,
        ranked_population(),
        fitness_fn=lambda phenotypes: -phenotypes.square().mean(dim=1),
    )

    assert {parameter.device.type for parameter in decoder.parameters()} == {"mps"}
    assert decoder.version == 1
