"""Interchangeable learning rules for the co-evolving decoder.

The evolutionary pipeline supplies a fitness-ranked population. A trainer may
use that ranking as self-supervision, as contrastive signal, or may query the
black-box fitness function directly. Every trainer that changes the mapping
must bump the decoder version so cached population fitness is invalidated.
"""
from __future__ import annotations

import copy
import random
from abc import ABC, abstractmethod
from typing import Callable, List, Mapping

import numpy as np
import torch
import torch.nn.functional as F

from .decoder import Decoder, TrainMode


class DecoderTrainer(ABC):
    """Strategy interface for a single decoder-training step."""

    @abstractmethod
    def step(
        self,
        decoder: Decoder,
        sorted_pop,
        fitness_fn: Callable | None = None,
    ) -> float:
        """Update ``decoder`` from a best-first population and return a metric."""


class FrozenTrainer(DecoderTrainer):
    """Control strategy: leave the decoder exactly as initialized."""

    def step(self, decoder, sorted_pop, fitness_fn=None) -> float:
        return 0.0


class DistillationTrainer(DecoderTrainer):
    """Wrap the three population-derived self-distillation pairings."""

    def __init__(
        self,
        mode: TrainMode = TrainMode.SELF_DISTILL,
        percent: float = 0.4,
        batch_size: int = 32,
        epochs: int = 1,
    ):
        if not isinstance(mode, TrainMode):
            raise TypeError("mode must be a TrainMode")
        if not 0 < percent <= 1:
            raise ValueError("percent must be in (0, 1]")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if epochs < 1:
            raise ValueError("epochs must be at least 1")
        self.mode = mode
        self.percent = percent
        self.batch_size = batch_size
        self.epochs = epochs

    def step(self, decoder, sorted_pop, fitness_fn=None) -> float:
        if not decoder.supports_refinement:
            raise TypeError("decoder does not implement Decoder.refine")
        return decoder.refine(
            sorted_pop,
            mode=self.mode,
            percent=self.percent,
            batch_size=self.batch_size,
            epochs=self.epochs,
        )


def _optimizer_for(decoder: Decoder):
    optimizer = getattr(decoder, "optimizer", None)
    if optimizer is None:
        optimizer = getattr(decoder, "opt", None)
    if optimizer is None:
        raise TypeError(
            "trainer requires decoder.optimizer (or the legacy decoder.opt alias)"
        )
    return optimizer


def _population_genes(sorted_pop) -> np.ndarray:
    if not sorted_pop:
        raise ValueError("cannot train from an empty population")
    return np.stack([individual.genes for individual in sorted_pop]).astype(np.float32)


class ContrastiveTrainer(DecoderTrainer):
    """Pull promising outputs toward the best and push them from the worst.

    The worst phenotypes are negative examples, not regression targets. For
    each promising latent, the positive term minimizes distance to the current
    best phenotype. A hinge term penalizes outputs that remain within ``margin``
    MSE of a paired worst phenotype.
    """

    def __init__(
        self,
        percent: float = 0.4,
        margin: float = 0.05,
        negative_weight: float = 0.5,
        batch_size: int = 32,
        epochs: int = 1,
    ):
        if not 0 < percent <= 1:
            raise ValueError("percent must be in (0, 1]")
        if margin < 0:
            raise ValueError("margin cannot be negative")
        if negative_weight < 0:
            raise ValueError("negative_weight cannot be negative")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if epochs < 1:
            raise ValueError("epochs must be at least 1")
        self.percent = percent
        self.margin = margin
        self.negative_weight = negative_weight
        self.batch_size = batch_size
        self.epochs = epochs

    def step(self, decoder, sorted_pop, fitness_fn=None) -> float:
        genes = _population_genes(sorted_pop)
        if len(genes) < 2:
            return 0.0
        optimizer = _optimizer_for(decoder)
        k = min(len(genes) - 1, max(1, int(len(genes) * self.percent)))
        inputs = torch.as_tensor(genes[1:k + 1], device=decoder.device)
        worst_inputs = torch.as_tensor(genes[-k:], device=decoder.device)
        best_input = torch.as_tensor(genes[:1], device=decoder.device)
        losses: List[float] = []
        was_training = decoder.training
        decoder.train(True)

        try:
            for _ in range(self.epochs):
                with torch.no_grad():
                    positive_targets = decoder(best_input).detach().expand(
                        k, *decoder.output_shape
                    )
                    negative_targets = decoder(worst_inputs).detach()

                for start in range(0, k, self.batch_size):
                    stop = start + self.batch_size
                    predicted = decoder(inputs[start:stop])
                    positive_mse = (
                        (predicted - positive_targets[start:stop])
                        .flatten(start_dim=1)
                        .square()
                        .mean(dim=1)
                    )
                    negative_mse = (
                        (predicted - negative_targets[start:stop])
                        .flatten(start_dim=1)
                        .square()
                        .mean(dim=1)
                    )
                    loss = (
                        positive_mse
                        + self.negative_weight
                        * F.relu(self.margin - negative_mse)
                    ).mean()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    losses.append(float(loss.detach().cpu()))
        finally:
            decoder.train(was_training)

        if losses:
            decoder.mark_updated()
        return float(np.mean(losses)) if losses else 0.0


class PolicyGradientTrainer(DecoderTrainer):
    """A minimal black-box REINFORCE update around decoder outputs.

    Decoder outputs are treated as the means of fixed-width Gaussian policies.
    Perturbed phenotypes are evaluated by the real fitness function, and a
    per-latent reward baseline supplies the advantage. Fitness queries made here
    are real objective evaluations and therefore count against benchmark budget.
    """

    def __init__(
        self,
        percent: float = 0.25,
        samples_per_gene: int = 4,
        exploration_std: float = 0.1,
        epochs: int = 1,
    ):
        if not 0 < percent <= 1:
            raise ValueError("percent must be in (0, 1]")
        if samples_per_gene < 2:
            raise ValueError("samples_per_gene must be at least 2")
        if exploration_std <= 0:
            raise ValueError("exploration_std must be positive")
        if epochs < 1:
            raise ValueError("epochs must be at least 1")
        self.percent = percent
        self.samples_per_gene = samples_per_gene
        self.exploration_std = exploration_std
        self.epochs = epochs

    def step(self, decoder, sorted_pop, fitness_fn=None) -> float:
        if fitness_fn is None:
            raise ValueError("PolicyGradientTrainer requires fitness_fn")
        genes = _population_genes(sorted_pop)
        optimizer = _optimizer_for(decoder)
        k = min(len(genes), max(1, int(len(genes) * self.percent)))
        inputs = torch.as_tensor(genes[:k], device=decoder.device)
        losses: List[float] = []
        was_training = decoder.training
        decoder.train(True)

        try:
            for _ in range(self.epochs):
                means = decoder(inputs)
                noise = torch.randn(
                    (self.samples_per_gene,) + tuple(means.shape),
                    device=means.device,
                    dtype=means.dtype,
                )
                # Samples are actions, not a differentiable path to the reward.
                raw_actions = (
                    means.detach().unsqueeze(0)
                    + self.exploration_std * noise
                )
                actions = raw_actions.clamp(0.0, 1.0)
                flat_actions = actions.flatten(end_dim=1)
                rewards = fitness_fn(flat_actions)
                rewards = torch.as_tensor(
                    rewards, device=means.device, dtype=means.dtype
                ).reshape(-1)
                expected = self.samples_per_gene * k
                if rewards.numel() != expected:
                    raise ValueError(
                        f"fitness_fn returned {rewards.numel()} values for "
                        f"{expected} sampled phenotypes"
                    )
                rewards = rewards.reshape(self.samples_per_gene, k)
                advantages = rewards - rewards.mean(dim=0, keepdim=True)
                advantages = advantages / advantages.std(unbiased=False).clamp_min(1e-6)

                centered = (
                    raw_actions - means.unsqueeze(0)
                ) / self.exploration_std
                log_prob = -0.5 * centered.square()
                log_prob = log_prob.flatten(start_dim=2).mean(dim=2)
                loss = -(advantages.detach() * log_prob).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
        finally:
            decoder.train(was_training)

        if losses:
            decoder.mark_updated()
        return float(np.mean(losses)) if losses else 0.0


class AdvantageWeightedTrainer(DecoderTrainer):
    """A lower-variance, reward-weighted policy-improvement update.

    Like REINFORCE, this explores Gaussian perturbations around decoder outputs
    and uses only black-box fitness. It then converts normalized advantages into
    softmax weights and regresses the decoder toward the better sampled actions.
    This is an advantage-weighted regression (AWR) style update with a fixed
    behavior policy, not a differentiable-objective shortcut.
    """

    def __init__(
        self,
        percent: float = 0.25,
        samples_per_gene: int = 4,
        exploration_std: float = 0.1,
        temperature: float = 0.5,
        epochs: int = 1,
    ):
        if not 0 < percent <= 1:
            raise ValueError("percent must be in (0, 1]")
        if samples_per_gene < 2:
            raise ValueError("samples_per_gene must be at least 2")
        if exploration_std <= 0:
            raise ValueError("exploration_std must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if epochs < 1:
            raise ValueError("epochs must be at least 1")
        self.percent = percent
        self.samples_per_gene = samples_per_gene
        self.exploration_std = exploration_std
        self.temperature = temperature
        self.epochs = epochs

    def step(self, decoder, sorted_pop, fitness_fn=None) -> float:
        if fitness_fn is None:
            raise ValueError("AdvantageWeightedTrainer requires fitness_fn")
        genes = _population_genes(sorted_pop)
        optimizer = _optimizer_for(decoder)
        k = min(len(genes), max(1, int(len(genes) * self.percent)))
        inputs = torch.as_tensor(genes[:k], device=decoder.device)
        losses: List[float] = []
        was_training = decoder.training
        decoder.train(True)

        try:
            for _ in range(self.epochs):
                means = decoder(inputs)
                noise = torch.randn(
                    (self.samples_per_gene,) + tuple(means.shape),
                    device=means.device,
                    dtype=means.dtype,
                )
                actions = (
                    means.detach().unsqueeze(0)
                    + self.exploration_std * noise
                ).clamp(0.0, 1.0)
                rewards = fitness_fn(actions.flatten(end_dim=1))
                rewards = torch.as_tensor(
                    rewards, device=means.device, dtype=means.dtype
                ).reshape(-1)
                expected = self.samples_per_gene * k
                if rewards.numel() != expected:
                    raise ValueError(
                        f"fitness_fn returned {rewards.numel()} values for "
                        f"{expected} sampled phenotypes"
                    )
                rewards = rewards.reshape(self.samples_per_gene, k)
                advantages = rewards - rewards.mean(dim=0, keepdim=True)
                scale = advantages.std(
                    dim=0, unbiased=False, keepdim=True
                ).clamp_min(1e-6)
                weights = torch.softmax(
                    advantages / scale / self.temperature, dim=0
                ).detach()
                squared_error = (
                    actions - means.unsqueeze(0)
                ).flatten(start_dim=2).square().mean(dim=2)
                loss = (weights * squared_error).sum(dim=0).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
        finally:
            decoder.train(was_training)

        if losses:
            decoder.mark_updated()
        return float(np.mean(losses)) if losses else 0.0


class PermutationTrainer(DecoderTrainer):
    """Distill an elite permutation without regressing on arbitrary raw keys.

    Random-key phenotypes represent a permutation through ``argsort``. Their
    Euclidean values are otherwise meaningless: monotone transforms preserve
    the route, while a tiny key change can swap two cities. This trainer uses a
    pairwise ordering loss against the best current route. The route is
    canonicalized across cyclic rotations and reversal before constructing the
    targets, which is appropriate for undirected cyclic routes such as TSP.

    When ``anchor_weight`` is positive, the best ``anchor_percent`` of the
    population are held near their pre-update decoder outputs. Training then
    acts on the next-best individuals, so the update expands a promising basin
    without directly overwriting the elites that define it.
    """

    def __init__(
        self,
        percent: float = 0.4,
        temperature: float = 0.1,
        anchor_weight: float = 0.0,
        anchor_percent: float = 0.1,
        batch_size: int = 32,
        epochs: int = 1,
    ):
        if not 0 < percent <= 1:
            raise ValueError("percent must be in (0, 1]")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if anchor_weight < 0:
            raise ValueError("anchor_weight cannot be negative")
        if not 0 < anchor_percent <= 1:
            raise ValueError("anchor_percent must be in (0, 1]")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if epochs < 1:
            raise ValueError("epochs must be at least 1")
        self.percent = percent
        self.temperature = temperature
        self.anchor_weight = anchor_weight
        self.anchor_percent = anchor_percent
        self.batch_size = batch_size
        self.epochs = epochs

    @staticmethod
    def canonical_route(keys) -> tuple[int, ...]:
        """Return a rotation/reversal-invariant route beginning at city zero."""
        values = np.asarray(keys, dtype=float).reshape(-1)
        route = np.argsort(values).tolist()
        zero = route.index(0)
        forward = route[zero:] + route[:zero]
        reverse = [forward[0], *reversed(forward[1:])]
        return min(tuple(forward), tuple(reverse))

    @staticmethod
    def _pairwise_targets(keys, device, dtype):
        route = PermutationTrainer.canonical_route(keys)
        ranks = np.empty(len(route), dtype=np.float32)
        for rank, city in enumerate(route):
            ranks[city] = rank
        pairs = torch.triu_indices(
            len(route), len(route), offset=1, device=device
        )
        rank_tensor = torch.as_tensor(ranks, device=device, dtype=dtype)
        targets = (
            rank_tensor[pairs[0]] > rank_tensor[pairs[1]]
        ).to(dtype=dtype)
        return pairs, targets

    def step(self, decoder, sorted_pop, fitness_fn=None) -> float:
        if len(decoder.output_shape) != 1 or decoder.output_shape[0] < 2:
            raise ValueError(
                "PermutationTrainer requires a one-dimensional output with "
                "at least two elements"
            )
        genes = _population_genes(sorted_pop)
        optimizer = _optimizer_for(decoder)
        n = len(genes)
        amount = min(n, max(1, int(n * self.percent)))
        anchor_amount = 0
        if self.anchor_weight > 0:
            anchor_amount = min(
                n - 1, max(1, int(n * self.anchor_percent))
            )
        stop = min(n, anchor_amount + amount)
        inputs = torch.as_tensor(
            genes[anchor_amount:stop], device=decoder.device
        )
        if len(inputs) == 0:
            return 0.0

        best_input = torch.as_tensor(genes[:1], device=decoder.device)
        anchor_inputs = torch.as_tensor(
            genes[:anchor_amount], device=decoder.device
        )
        with torch.no_grad():
            best_output = decoder(best_input).detach()[0]
            anchor_targets = (
                decoder(anchor_inputs).detach()
                if anchor_amount else None
            )
        pairs, pair_targets = self._pairwise_targets(
            best_output.cpu().numpy(), best_output.device, best_output.dtype
        )
        losses: List[float] = []
        was_training = decoder.training
        decoder.train(True)

        try:
            for _ in range(self.epochs):
                for start in range(0, len(inputs), self.batch_size):
                    predicted = decoder(inputs[start:start + self.batch_size])
                    logits = (
                        predicted[:, pairs[0]] - predicted[:, pairs[1]]
                    ) / self.temperature
                    targets = pair_targets.expand(len(predicted), -1)
                    order_loss = F.binary_cross_entropy_with_logits(
                        logits, targets
                    )
                    if anchor_targets is None:
                        loss = order_loss
                    else:
                        anchor_loss = F.mse_loss(
                            decoder(anchor_inputs), anchor_targets
                        )
                        loss = order_loss + self.anchor_weight * anchor_loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    losses.append(float(loss.detach().cpu()))
        finally:
            decoder.train(was_training)

        if losses:
            decoder.mark_updated()
        return float(np.mean(losses)) if losses else 0.0


class BacktrackingTrainer(DecoderTrainer):
    """Reuse a proposed gradient while shrinking updates that hurt a probe.

    The wrapped trainer computes one update. This wrapper then evaluates the
    largest interpolation between the old and proposed parameters that improves
    either mean or best elite-probe fitness. It therefore avoids recomputing the
    gradient and only performs a full rollback when every tested step size is
    harmful. Probe evaluations count as ordinary fitness calls.
    """

    def __init__(
        self,
        trainer: DecoderTrainer,
        probe_percent: float = 0.25,
        factors=(1.0, 0.5, 0.25, 0.125),
        min_improvement: float = 0.0,
    ):
        if not isinstance(trainer, DecoderTrainer):
            raise TypeError("trainer must be a DecoderTrainer")
        if not 0 < probe_percent <= 1:
            raise ValueError("probe_percent must be in (0, 1]")
        factors = tuple(float(factor) for factor in factors)
        if not factors or any(not 0 < factor <= 1 for factor in factors):
            raise ValueError("factors must contain values in (0, 1]")
        if any(first <= second for first, second in zip(factors, factors[1:])):
            raise ValueError("factors must be strictly decreasing")
        self.trainer = trainer
        self.probe_percent = probe_percent
        self.factors = factors
        self.min_improvement = float(min_improvement)
        self.factor_history: List[float] = []
        self.probe_improvements: List[float] = []
        self.probe_evaluations = 0

    def step(self, decoder, sorted_pop, fitness_fn=None) -> float:
        if fitness_fn is None:
            raise ValueError("BacktrackingTrainer requires fitness_fn")
        if not sorted_pop:
            raise ValueError("cannot train from an empty population")
        optimizer = _optimizer_for(decoder)
        old_state = {
            name: value.detach().clone()
            for name, value in decoder.state_dict().items()
        }
        optimizer_state = copy.deepcopy(optimizer.state_dict())
        previous_version = decoder.version
        loss = self.trainer.step(
            decoder, sorted_pop, fitness_fn=fitness_fn
        )
        if decoder.version == previous_version:
            return loss
        proposed_state = {
            name: value.detach().clone()
            for name, value in decoder.state_dict().items()
        }

        amount = min(
            len(sorted_pop),
            max(1, int(len(sorted_pop) * self.probe_percent)),
        )
        probe = sorted_pop[:amount]
        baseline = np.asarray(
            [individual.fitness for individual in probe], dtype=float
        )
        genes = np.stack([individual.genes for individual in probe]).astype(
            np.float32
        )
        selected_factor = 0.0
        selected_fitness = None

        for factor in self.factors:
            interpolated = {}
            for name, old_value in old_state.items():
                proposed_value = proposed_state[name]
                if old_value.is_floating_point():
                    interpolated[name] = old_value + factor * (
                        proposed_value - old_value
                    )
                else:
                    interpolated[name] = proposed_value
            decoder.load_state_dict(interpolated)
            candidate = fitness_fn(decoder.decode(genes))
            if isinstance(candidate, torch.Tensor):
                candidate = candidate.detach().reshape(-1).cpu().numpy()
            else:
                candidate = np.asarray(list(candidate), dtype=float).reshape(-1)
            if len(candidate) != amount:
                raise ValueError(
                    f"fitness_fn returned {len(candidate)} values for "
                    f"{amount} probe phenotypes"
                )
            if np.isnan(candidate).any():
                raise ValueError("fitness_fn returned NaN")
            self.probe_evaluations += amount
            mean_gain = float(candidate.mean() - baseline.mean())
            best_gain = float(candidate.max() - baseline.max())
            if (
                mean_gain >= self.min_improvement
                or best_gain > self.min_improvement
            ):
                selected_factor = factor
                selected_fitness = candidate
                self.probe_improvements.append(mean_gain)
                break

        self.factor_history.append(selected_factor)
        if selected_fitness is None:
            decoder.load_state_dict(old_state)
            optimizer.load_state_dict(optimizer_state)
            self.probe_improvements.append(0.0)
            for individual in sorted_pop:
                individual.evaluated_at = decoder.version
        else:
            if selected_factor < 1.0:
                # Full-step Adam moments do not describe an interpolated step.
                optimizer.load_state_dict(optimizer_state)
            for individual, fitness in zip(probe, selected_fitness):
                individual.fitness = float(fitness)
                individual.evaluated_at = decoder.version
        return loss


class MixtureTrainer(DecoderTrainer):
    """Choose one component training objective at a time.

    ``random`` samples objectives with replacement. ``round_robin`` provides a
    deterministic ordering control. ``shuffled_cycle`` randomizes the order but
    guarantees that every objective is used once before any repeats. One
    component step is treated as one training micro-batch; ``steps_per_call``
    controls how many sequential micro-batches happen at a scheduled decoder
    update. All micro-batches share the population ranking snapshot supplied to
    that update.
    """

    STRATEGIES = {"random", "round_robin", "shuffled_cycle"}

    def __init__(
        self,
        trainers: Mapping[str, DecoderTrainer],
        strategy: str = "random",
        steps_per_call: int = 1,
        seed: int | None = None,
        weights: Mapping[str, float] | None = None,
    ):
        if not trainers:
            raise ValueError("trainers cannot be empty")
        if strategy not in self.STRATEGIES:
            raise ValueError(
                f"strategy must be one of {sorted(self.STRATEGIES)}"
            )
        if steps_per_call < 1:
            raise ValueError("steps_per_call must be at least 1")
        if not all(isinstance(trainer, DecoderTrainer) for trainer in trainers.values()):
            raise TypeError("every mixture component must be a DecoderTrainer")

        self.trainers = dict(trainers)
        self.names = tuple(self.trainers)
        self.strategy = strategy
        self.steps_per_call = steps_per_call
        self._rng = random.Random(seed)
        self._cursor = 0
        self._shuffled: List[str] = []
        self.history: List[str] = []
        self.selection_counts = {name: 0 for name in self.names}

        if weights is None:
            self.weights = None
        else:
            unknown = set(weights) - set(self.names)
            missing = set(self.names) - set(weights)
            if unknown or missing:
                raise ValueError(
                    "weights must contain exactly the mixture trainer names"
                )
            values = tuple(float(weights[name]) for name in self.names)
            if any(value < 0 for value in values) or not any(values):
                raise ValueError("weights must be non-negative with a positive sum")
            self.weights = values

    def _choose(self) -> str:
        if self.strategy == "random":
            return self._rng.choices(self.names, weights=self.weights, k=1)[0]
        if self.strategy == "round_robin":
            name = self.names[self._cursor % len(self.names)]
            self._cursor += 1
            return name
        if not self._shuffled:
            self._shuffled = list(self.names)
            self._rng.shuffle(self._shuffled)
        return self._shuffled.pop()

    def step(self, decoder, sorted_pop, fitness_fn=None) -> float:
        losses = []
        for _ in range(self.steps_per_call):
            name = self._choose()
            self.history.append(name)
            self.selection_counts[name] += 1
            losses.append(
                self.trainers[name].step(
                    decoder, sorted_pop, fitness_fn=fitness_fn
                )
            )
        return float(np.mean(losses))


class AdaptiveMixtureTrainer(DecoderTrainer):
    """Keep every update and learn a changing allocation over objectives.

    Each component is a bandit arm. The trainer first tries every arm once in a
    shuffled warm-up, then samples from softmax probabilities derived from an
    exponentially recency-weighted reward. A probability floor keeps every arm
    explorable, while forgetting lets an objective that helped early lose its
    preference later.

    Reward is the normalized change in mean and best fitness on a fixed elite
    probe. Unlike :class:`GuardedTrainer`, harmful proposals are never restored;
    they only reduce the probability of choosing that objective again.
    """

    def __init__(
        self,
        trainers: Mapping[str, DecoderTrainer],
        probe_percent: float = 0.25,
        steps_per_call: int = 3,
        learning_rate: float = 0.5,
        forgetting: float = 0.9,
        temperature: float = 0.1,
        exploration: float = 0.15,
        best_weight: float = 0.5,
        reward_clip: float = 1.0,
        seed: int | None = None,
        priors: Mapping[str, float] | None = None,
        warmup: bool = True,
    ):
        if not trainers:
            raise ValueError("trainers cannot be empty")
        if not all(isinstance(trainer, DecoderTrainer) for trainer in trainers.values()):
            raise TypeError("every adaptive component must be a DecoderTrainer")
        if not 0 < probe_percent <= 1:
            raise ValueError("probe_percent must be in (0, 1]")
        if steps_per_call < 1:
            raise ValueError("steps_per_call must be at least 1")
        if not 0 < learning_rate <= 1:
            raise ValueError("learning_rate must be in (0, 1]")
        if not 0 <= forgetting <= 1:
            raise ValueError("forgetting must be in [0, 1]")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if not 0 <= exploration <= 1:
            raise ValueError("exploration must be in [0, 1]")
        if not 0 <= best_weight <= 1:
            raise ValueError("best_weight must be in [0, 1]")
        if reward_clip <= 0:
            raise ValueError("reward_clip must be positive")

        self.trainers = dict(trainers)
        self.names = tuple(self.trainers)
        self.probe_percent = probe_percent
        self.steps_per_call = steps_per_call
        self.learning_rate = learning_rate
        self.forgetting = forgetting
        self.temperature = temperature
        self.exploration = exploration
        self.best_weight = best_weight
        self.reward_clip = reward_clip
        self._rng = random.Random(seed)
        self._warmup = list(self.names) if warmup else []
        self._rng.shuffle(self._warmup)
        self.values = {name: 0.0 for name in self.names}
        self.history: List[str] = []
        self.reward_history: List[float] = []
        self.probability_history: List[dict[str, float]] = []
        self.selection_counts = {name: 0 for name in self.names}

        if priors is None:
            self.priors = {name: 1.0 for name in self.names}
        else:
            if set(priors) != set(self.names):
                raise ValueError("priors must contain exactly the trainer names")
            self.priors = {name: float(priors[name]) for name in self.names}
            if any(value <= 0 for value in self.priors.values()):
                raise ValueError("every prior must be positive")

    @property
    def probabilities(self) -> dict[str, float]:
        scores = np.asarray(
            [self.values[name] / self.temperature for name in self.names],
            dtype=float,
        )
        scores -= scores.max()
        weights = np.exp(scores) * np.asarray(
            [self.priors[name] for name in self.names], dtype=float
        )
        probabilities = weights / weights.sum()
        probabilities = (
            (1.0 - self.exploration) * probabilities
            + self.exploration / len(self.names)
        )
        return {
            name: float(probability)
            for name, probability in zip(self.names, probabilities)
        }

    def _choose(self) -> str:
        if self._warmup:
            return self._warmup.pop()
        probabilities = self.probabilities
        return self._rng.choices(
            self.names,
            weights=[probabilities[name] for name in self.names],
            k=1,
        )[0]

    def _update_value(self, name: str, reward: float):
        for objective in self.names:
            self.values[objective] *= self.forgetting
        old = self.values[name]
        self.values[name] = (
            (1.0 - self.learning_rate) * old
            + self.learning_rate * reward
        )

    def step(self, decoder, sorted_pop, fitness_fn=None) -> float:
        if fitness_fn is None:
            raise ValueError("AdaptiveMixtureTrainer requires fitness_fn")
        if not sorted_pop:
            raise ValueError("cannot train from an empty population")
        k = min(
            len(sorted_pop),
            max(1, int(len(sorted_pop) * self.probe_percent)),
        )
        probe = sorted_pop[:k]
        baseline = np.asarray(
            [individual.fitness for individual in probe], dtype=float
        )
        genes = np.stack([individual.genes for individual in probe]).astype(
            np.float32
        )
        losses = []

        for _ in range(self.steps_per_call):
            probabilities = self.probabilities
            name = self._choose()
            self.history.append(name)
            self.probability_history.append(probabilities)
            self.selection_counts[name] += 1
            loss = self.trainers[name].step(
                decoder, sorted_pop, fitness_fn=fitness_fn
            )
            losses.append(loss)

            candidate = fitness_fn(decoder.decode(genes))
            if isinstance(candidate, torch.Tensor):
                candidate = candidate.detach().reshape(-1).cpu().numpy()
            else:
                candidate = np.asarray(list(candidate), dtype=float).reshape(-1)
            if len(candidate) != k:
                raise ValueError(
                    f"fitness_fn returned {len(candidate)} values for "
                    f"{k} probe phenotypes"
                )
            if np.isnan(candidate).any():
                raise ValueError("fitness_fn returned NaN")

            scale = max(float(np.mean(np.abs(baseline))), 1e-6)
            mean_gain = float(candidate.mean() - baseline.mean()) / scale
            best_gain = float(candidate.max() - baseline.max()) / scale
            reward = (
                (1.0 - self.best_weight) * mean_gain
                + self.best_weight * best_gain
            )
            reward = float(np.clip(reward, -self.reward_clip, self.reward_clip))
            self.reward_history.append(reward)
            self._update_value(name, reward)

            for individual, fitness in zip(probe, candidate):
                individual.fitness = float(fitness)
                individual.evaluated_at = decoder.version
            baseline = candidate

        return float(np.mean(losses))


class GuardedTrainer(DecoderTrainer):
    """Accept a proposed update only when it improves a fixed elite probe.

    This wrapper gives mixed objectives local credit assignment. It snapshots
    the decoder and optimizer, lets the wrapped trainer propose an update, and
    evaluates the same top-ranked latent probes before and after. Rejected
    proposals restore both model and optimizer state. Probe evaluations are
    ordinary fitness calls and therefore remain visible to evaluation-budgeted
    benchmarks.
    """

    def __init__(
        self,
        trainer: DecoderTrainer,
        probe_percent: float = 0.25,
        min_improvement: float = 0.0,
    ):
        if not isinstance(trainer, DecoderTrainer):
            raise TypeError("trainer must be a DecoderTrainer")
        if not 0 < probe_percent <= 1:
            raise ValueError("probe_percent must be in (0, 1]")
        self.trainer = trainer
        self.probe_percent = probe_percent
        self.min_improvement = float(min_improvement)
        self.acceptance_history: List[bool] = []
        self.probe_improvements: List[float] = []

    @property
    def history(self):
        return getattr(self.trainer, "history", [])

    @property
    def selection_counts(self):
        return getattr(self.trainer, "selection_counts", {})

    def step(self, decoder, sorted_pop, fitness_fn=None) -> float:
        if fitness_fn is None:
            raise ValueError("GuardedTrainer requires fitness_fn")
        if not sorted_pop:
            raise ValueError("cannot train from an empty population")
        optimizer = _optimizer_for(decoder)
        model_state = {
            name: value.detach().clone()
            for name, value in decoder.state_dict().items()
        }
        optimizer_state = copy.deepcopy(optimizer.state_dict())
        previous_version = decoder.version
        loss = self.trainer.step(
            decoder, sorted_pop, fitness_fn=fitness_fn
        )
        if decoder.version == previous_version:
            return loss

        k = min(
            len(sorted_pop),
            max(1, int(len(sorted_pop) * self.probe_percent)),
        )
        probe = sorted_pop[:k]
        baseline = np.asarray(
            [individual.fitness for individual in probe], dtype=float
        )
        genes = np.stack([individual.genes for individual in probe]).astype(
            np.float32
        )
        candidate = fitness_fn(decoder.decode(genes))
        if isinstance(candidate, torch.Tensor):
            candidate = candidate.detach().reshape(-1).cpu().numpy()
        else:
            candidate = np.asarray(list(candidate), dtype=float).reshape(-1)
        if len(candidate) != k:
            raise ValueError(
                f"fitness_fn returned {len(candidate)} values for {k} probe phenotypes"
            )
        if np.isnan(candidate).any():
            raise ValueError("fitness_fn returned NaN")

        improvement = float(candidate.mean() - baseline.mean())
        accepted = (
            improvement >= self.min_improvement
            or float(candidate.max())
            > float(baseline.max()) + self.min_improvement
        )
        self.acceptance_history.append(accepted)
        self.probe_improvements.append(improvement)

        if accepted:
            for individual, fitness in zip(probe, candidate):
                individual.fitness = float(fitness)
                individual.evaluated_at = decoder.version
        else:
            decoder.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)
            # The restored mapping is exactly the mapping under which these
            # stored fitnesses were measured, so advance their cache stamps.
            for individual in sorted_pop:
                individual.evaluated_at = decoder.version
        return loss
