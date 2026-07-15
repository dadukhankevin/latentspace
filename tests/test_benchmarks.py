import numpy as np
import pytest
import torch

from benchmarks.compare import (
    BenchmarkConfig,
    Rastrigin,
    TargetMatch,
    TrackedFitness,
    TravelingSalesperson,
    run_differential_evolution,
    run_direct_ga,
    run_mu_plus_lambda_es,
    run_random_search,
)
from latentspace import EvolveDecoder, Evolver


@pytest.mark.parametrize(
    "objective",
    [TargetMatch(8), Rastrigin(8), TravelingSalesperson(8)],
)
def test_numpy_and_tensor_objectives_match(objective):
    rng = np.random.default_rng(0)
    phenotypes = rng.random((5, objective.dimension), dtype=np.float32)

    numpy_loss = objective.loss_numpy(phenotypes)
    tensor_loss = objective.loss_tensor(torch.from_numpy(phenotypes)).numpy()

    assert tensor_loss == pytest.approx(numpy_loss, rel=1e-5, abs=1e-5)


def test_tracker_reports_the_exact_requested_evaluation():
    tracker = TrackedFitness(TargetMatch(2))
    tracker.evaluate_numpy(
        np.array([[1.0, 1.0], [0.0, 1.0], [0.0, 0.0]], dtype=np.float32)
    )

    assert tracker.evaluations == 3
    assert tracker.best_at(1) == pytest.approx(0.5)
    assert tracker.best_at(2) == pytest.approx(0.0)
    assert tracker.best_at(3) == pytest.approx(0.0)


@pytest.mark.parametrize(
    "runner",
    [
        run_random_search,
        run_direct_ga,
        run_mu_plus_lambda_es,
        run_differential_evolution,
    ],
)
def test_direct_baselines_honor_evaluation_budget(runner):
    config = BenchmarkConfig(
        evaluation_budget=96,
        population=16,
        offspring=16,
        latent=8,
        hidden_size=16,
    )

    result = runner(TargetMatch(4), seed=0, config=config)

    assert result.evaluations_run == 96
    assert np.isfinite(result.metric_at_budget)
    assert result.neural_device is None


def test_decoder_es_is_a_supported_update_layer():
    def fitness(phenotypes):
        return phenotypes[:, 0]

    update = EvolveDecoder(
        fitness,
        every=1,
        n_candidates=2,
        percent=0.5,
        sigma=1e-2,
    )
    evolver = Evolver(
        fitness,
        output_shape=(1,),
        latent=4,
        population=8,
        hidden_size=8,
        families=1,
        children=1,
        mutation_rate=0.1,
        refine_every=None,
        decoder_update=update,
    )

    evolver.solve(1, verbose_every=0)

    assert update.last_fitness is not None
    assert {
        individual.evaluated_at for individual in evolver.env.population
    } == {evolver.decoder.version}
