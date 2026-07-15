"""The single co-evolving decoder: latent vector -> phenotype of any shape.

It is an `Individual` (Aule): its fitness is the quality of the population it
currently supports. It improves along two channels:

  * gradient descent  -> `refine()`, using population-derived self-supervision
                         (GeneSpace's key idea). No differentiable objective is
                         required, because the targets are the decoder's OWN
                         outputs on the best individuals.
  * evolution strategy -> `evolve_step()`, random weight perturbations kept only
                         if they raise population fitness. For settings where
                         even the self-supervision is undesirable.

Every weight update bumps `version`, which invalidates the fitness caches of the
whole population (see DecodeAndEvaluate) so nobody keeps a fitness measured under
an old mapping. That single counter is what keeps a non-stationary,
co-evolving landscape honest.
"""
from __future__ import annotations

import enum
from typing import Callable, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .core import Individual


class TrainMode(enum.Enum):
    SELF_DISTILL = 1   # worst genes -> decoder's own phenotypes for the best genes
    GOOD_TO_BEST = 2   # top genes   -> phenotype of the single best
    EACH_TO_NEXT = 3   # ranked chain: each -> the next-better individual's phenotype


class Decoder(nn.Module, Individual):
    """Extension contract for latent-to-phenotype mappings.

    Custom decoders may use any internal architecture, but they must expose a
    fixed latent input length, an output shape, a monotonically increasing
    ``version``, and a batched ``decode`` method. Decoders that can learn during
    evolution should also override ``refine`` and call ``mark_updated`` after
    changing the mapping.
    """

    def __init__(self, input_length: int, output_shape, device: str = "cpu"):
        nn.Module.__init__(self)
        Individual.__init__(self)
        self.input_length = int(input_length)
        if self.input_length < 1:
            raise ValueError("input_length must be at least 1")
        self.output_shape = tuple(output_shape)
        if any(int(size) < 1 for size in self.output_shape):
            raise ValueError("every output dimension must be at least 1")
        self.device = device
        self.version = 0

    @property
    def supports_refinement(self) -> bool:
        return type(self).refine is not Decoder.refine

    @property
    def supports_evolution(self) -> bool:
        return type(self).evolve_step is not Decoder.evolve_step

    def mark_updated(self) -> int:
        """Invalidate cached population fitness after the mapping changes."""
        self.version += 1
        return self.version

    def decode(self, genes_batch) -> torch.Tensor:
        raise NotImplementedError

    def refine(self, sorted_pop, **kwargs) -> float:
        raise NotImplementedError(
            "this decoder does not implement refinement; set refine_every=None"
        )

    def evolve_step(self, sorted_pop, fitness_fn: Callable, **kwargs) -> float:
        raise NotImplementedError("this decoder does not implement weight evolution")


class MLPDecoder(Decoder):
    def __init__(self, input_length: int, output_shape, hidden_size: int = 256,
                 num_layers: int = 2, lr: float = 1e-4, device: str = "cpu",
                 output_activation=nn.Sigmoid):
        Decoder.__init__(self, input_length, output_shape, device)

        if hidden_size < 1:
            raise ValueError("hidden_size must be at least 1")
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        if lr <= 0:
            raise ValueError("lr must be positive")
        self.output_size = int(np.prod(self.output_shape))

        blocks: List[nn.Module] = [nn.Linear(self.input_length, hidden_size), nn.LeakyReLU()]
        for _ in range(num_layers - 1):
            blocks += [nn.Linear(hidden_size, hidden_size), nn.LeakyReLU()]
        blocks += [nn.Linear(hidden_size, self.output_size)]

        self.net = nn.Sequential(*blocks).to(device)
        self.out_act = output_activation().to(device)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        # ``opt`` was the original public-ish attribute. Keep it as an alias so
        # existing experiments do not break while trainers use the clearer name.
        self.opt = self.optimizer
        self.loss_fn = nn.MSELoss()

    # ---- forward / decode ---------------------------------------------------
    def forward(self, x) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32)
        x = x.to(device=self.device, dtype=torch.float32)
        out = self.out_act(self.net(x))
        return out.view(-1, *self.output_shape)

    def decode(self, genes_batch) -> torch.Tensor:
        """(B, input_length) -> (B, *output_shape), no gradient."""
        with torch.no_grad():
            return self.forward(genes_batch)

    # ---- gradient channel ---------------------------------------------------
    def refine(self, sorted_pop, mode: TrainMode = TrainMode.SELF_DISTILL,
               percent: float = 0.4, batch_size: int = 32, epochs: int = 1) -> float:
        if not sorted_pop:
            raise ValueError("cannot refine from an empty population")
        if not 0 < percent <= 1:
            raise ValueError("percent must be in (0, 1]")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if epochs < 1:
            raise ValueError("epochs must be at least 1")

        genes = np.stack([ind.genes for ind in sorted_pop]).astype(np.float32)
        n = len(sorted_pop)
        minimum = 2 if mode == TrainMode.EACH_TO_NEXT and n >= 2 else 1
        k = min(n, max(minimum, int(n * percent)))
        losses: List[float] = []

        for _ in range(epochs):
            if mode == TrainMode.SELF_DISTILL:
                inputs = torch.tensor(genes[-k:], device=self.device)   # worst k
                source = torch.tensor(genes[:k], device=self.device)    # best  k
                with torch.no_grad():
                    targets = self.forward(source).detach()
            elif mode == TrainMode.GOOD_TO_BEST:
                inputs = torch.tensor(genes[:k], device=self.device)
                with torch.no_grad():
                    targets = self.forward(genes[:1]).detach().expand(
                        len(inputs), *self.output_shape)
            elif mode == TrainMode.EACH_TO_NEXT:
                inputs = torch.tensor(genes[1:k], device=self.device)
                with torch.no_grad():
                    targets = self.forward(genes[:k - 1]).detach()
            else:
                raise ValueError(mode)

            for i in range(0, len(inputs), batch_size):
                xb, yb = inputs[i:i + batch_size], targets[i:i + batch_size]
                self.opt.zero_grad()
                loss = self.loss_fn(self.forward(xb), yb)
                loss.backward()
                self.opt.step()
                losses.append(loss.item())

        if losses:
            self.mark_updated()
        return float(np.mean(losses)) if losses else 0.0

    # ---- evolution-strategy channel ----------------------------------------
    def evolve_step(self, sorted_pop, fitness_fn: Callable, n_candidates: int = 8,
                    percent: float = 0.4, sigma: float | None = None) -> float:
        if not sorted_pop:
            raise ValueError("cannot evolve a decoder from an empty population")
        if not 0 < percent <= 1:
            raise ValueError("percent must be in (0, 1]")
        if n_candidates < 0:
            raise ValueError("n_candidates cannot be negative")
        if sigma is not None and sigma < 0:
            raise ValueError("sigma cannot be negative")

        k = min(len(sorted_pop), max(1, int(len(sorted_pop) * percent)))
        genes = torch.tensor(
            np.stack([ind.genes for ind in sorted_pop[:k]]).astype(np.float32),
            device=self.device,
        )
        sigma = sigma if sigma is not None else self.opt.param_groups[0]["lr"]
        base = [p.detach().clone() for p in self.parameters()]

        def population_fitness() -> float:
            values = fitness_fn(self.decode(genes))
            if isinstance(values, torch.Tensor):
                values = values.detach().cpu().numpy()
            values = np.asarray(values, dtype=float).reshape(-1)
            if len(values) != len(genes):
                raise ValueError(
                    f"fitness_fn returned {len(values)} values for {len(genes)} phenotypes"
                )
            return float(np.mean(values))

        best_fit = population_fitness()
        best_pert = None
        for _ in range(n_candidates):
            pert = [torch.randn_like(p) * sigma for p in self.parameters()]
            with torch.no_grad():
                for p, d in zip(self.parameters(), pert):
                    p.add_(d)
            fit = population_fitness()
            with torch.no_grad():
                for p, b in zip(self.parameters(), base):
                    p.copy_(b)
            if fit > best_fit:
                best_fit, best_pert = fit, pert

        if best_pert is not None:
            with torch.no_grad():
                for p, d in zip(self.parameters(), best_pert):
                    p.add_(d)
            self.mark_updated()
        return best_fit
