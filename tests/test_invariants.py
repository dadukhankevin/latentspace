import numpy as np
import pytest
import torch

from latentspace import (Decoder, Evolver, LatentIndividual, Mutate,
                         MutationOffspring, Schedule)
from latentspace.layers import Crossover


def scalar_fitness(phenotypes):
    return phenotypes[:, 0]


def small_evolver(**kwargs):
    defaults = dict(
        output_shape=(1,),
        latent=4,
        population=8,
        hidden_size=8,
        num_layers=1,
        lr=0.1,
        mutation_rate=0.0,
        families=1,
        children=1,
        n_points=2,
    )
    defaults.update(kwargs)
    fitness_fn = defaults.pop("fitness_fn", scalar_fitness)
    return Evolver(fitness_fn, **defaults)


def test_schedule_clamps_at_endpoint():
    schedule = Schedule(0.0, 1.0, steps=10)

    values = [schedule() for _ in range(13)]

    assert values == pytest.approx([i / 10 for i in range(11)] + [1.0, 1.0])


def test_zero_percent_mutation_changes_nothing():
    pop = [LatentIndividual(np.zeros(3, dtype=np.float32)) for _ in range(4)]

    Mutate(rate=1.0, percent=0.0, binary=True)(pop)

    assert all(not ind.genes.any() for ind in pop)


def test_selected_individuals_receive_at_least_one_mutation():
    np.random.seed(2)
    pop = [LatentIndividual(np.zeros(3, dtype=np.float32)) for _ in range(4)]

    Mutate(rate=1e-12, sigma=1.0, percent=1.0)(pop)

    assert all(ind.genes.any() for ind in pop)


def test_offspring_only_mutation_preserves_evaluated_parents():
    np.random.seed(5)
    parents = [LatentIndividual(np.zeros(3, dtype=np.float32)) for _ in range(2)]
    children = [LatentIndividual(np.zeros(3, dtype=np.float32)) for _ in range(2)]
    for parent in parents:
        parent.evaluated_at = 0

    class DecoderVersion:
        version = 0

    class Environment:
        decoder = DecoderVersion()

    mutation = Mutate(
        rate=1.0,
        percent=1.0,
        binary=True,
        offspring_only=True,
    )
    mutation.bind(Environment())
    mutation(parents + children)

    assert all(not parent.genes.any() for parent in parents)
    assert all(parent.evaluated_at == 0 for parent in parents)
    assert all(child.genes.all() for child in children)
    assert all(child.evaluated_at == -1 for child in children)


def test_elitist_pipeline_evaluates_only_new_children_after_initialization():
    evaluations = 0

    def counting_fitness(phenotypes):
        nonlocal evaluations
        evaluations += len(phenotypes)
        return phenotypes[:, 0]

    evolver = small_evolver(
        fitness_fn=counting_fitness,
        refine_every=None,
        population=8,
        families=1,
        children=2,
        offspring_only_mutation=True,
    )

    evolver.solve(3, verbose_every=0)

    assert evaluations == 8 + 3 * 2


def test_mutation_offspring_are_independent_copies_with_replacement():
    np.random.seed(6)
    parent = LatentIndividual(np.zeros(3, dtype=np.float32))
    parent.evaluated_at = 0

    result = MutationOffspring(
        amount=4,
        rate=1.0,
        binary=True,
        replace=True,
    )([parent])
    children = result[1:]

    assert not parent.genes.any()
    assert len({id(child) for child in children}) == 4
    assert all(child.genes.all() for child in children)
    assert all(child.evaluated_at == -1 for child in children)


def test_two_stage_pipeline_evaluates_both_offspring_streams():
    evaluations = 0

    def counting_fitness(phenotypes):
        nonlocal evaluations
        evaluations += len(phenotypes)
        return phenotypes[:, 0]

    evolver = small_evolver(
        fitness_fn=counting_fitness,
        refine_every=None,
        population=8,
        families=1,
        children=2,
        operator_schedule="two_stage",
        mutation_children=3,
    )

    evolver.solve(3, verbose_every=0)

    assert evaluations == 8 + 3 * (2 + 3)


def test_selection_only_sees_current_fitness():
    np.random.seed(7)
    torch.manual_seed(7)
    evolver = small_evolver(refine_every=1)
    crossover = next(layer for layer in evolver.env.layers if isinstance(layer, Crossover))
    original_selection = crossover.selection
    observations = []

    class RecordingSelection:
        def __call__(self, pop, k):
            observations.append(
                (evolver.decoder.version, {ind.evaluated_at for ind in pop})
            )
            return original_selection(pop, k)

    crossover.selection = RecordingSelection()
    evolver.solve(2, verbose_every=0)

    assert observations == [(0, {0}), (1, {1})]


def test_generation_ends_fully_evaluated_under_current_decoder():
    np.random.seed(3)
    torch.manual_seed(3)
    evolver = small_evolver(refine_every=1)

    evolver.solve(3, verbose_every=0)

    assert evolver.decoder.version == 3
    assert {ind.evaluated_at for ind in evolver.env.population} == {3}
    assert evolver.env.history["decoder_version"] == [1, 2, 3]
    assert evolver.env.history["generation"] == [0, 1, 2]


def test_refinement_can_be_disabled_and_does_not_run_at_generation_zero():
    np.random.seed(4)
    torch.manual_seed(4)
    fixed = small_evolver(refine_every=None)
    delayed = small_evolver(refine_every=3)

    fixed.solve(4, verbose_every=0)
    delayed.solve(2, verbose_every=0)

    assert fixed.decoder.version == 0
    assert delayed.decoder.version == 0
    delayed.solve(1, verbose_every=0)
    assert delayed.decoder.version == 1


def test_best_observed_is_a_reproducible_phenotype_snapshot():
    np.random.seed(9)
    torch.manual_seed(9)
    evolver = small_evolver(refine_every=1)
    evolver.solve(2, verbose_every=0)
    phenotype = evolver.decode_best()
    recorded_fitness = evolver.best.fitness

    with torch.no_grad():
        for parameter in evolver.decoder.parameters():
            parameter.add_(10.0)
    evolver.decoder.mark_updated()

    assert evolver.decode_best() == pytest.approx(phenotype)
    assert recorded_fitness == pytest.approx(float(scalar_fitness(phenotype[None])[0]))
    genes = evolver.best.genes
    genes[:] = 99
    assert not np.all(evolver.best.genes == 99)


class FirstGeneDecoder(Decoder):
    def __init__(self):
        super().__init__(input_length=3, output_shape=(1,))

    def decode(self, genes_batch):
        genes = torch.as_tensor(genes_batch, dtype=torch.float32)
        return torch.sigmoid(genes[:, :1])


def test_custom_decoder_is_the_primary_extension_seam():
    np.random.seed(11)
    decoder = FirstGeneDecoder()
    evolver = Evolver(
        scalar_fitness,
        output_shape=(1,),
        latent=3,
        population=6,
        families=1,
        children=1,
        mutation_rate=0.0,
        refine_every=None,
        decoder=decoder,
    )

    evolver.solve(2, verbose_every=0)

    assert evolver.decoder is decoder
    assert evolver.best is not None
    assert evolver.best.decoder_version == 0


def test_custom_decoder_without_refinement_requires_explicit_disable():
    with pytest.raises(ValueError, match="refine_every=None"):
        Evolver(
            scalar_fitness,
            output_shape=(1,),
            latent=3,
            population=6,
            decoder=FirstGeneDecoder(),
        )


def test_fitness_function_must_return_one_value_per_phenotype():
    evolver = Evolver(
        lambda phenotypes: [1.0],
        output_shape=(1,),
        latent=3,
        population=6,
        families=1,
        children=1,
        refine_every=None,
    )

    with pytest.raises(ValueError, match="fitness_fn returned 1 values for 6 phenotypes"):
        evolver.solve(1, verbose_every=0)
