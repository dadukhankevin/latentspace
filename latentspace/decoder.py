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


class MLPDecoder(nn.Module, Individual):
    def __init__(self, input_length: int, output_shape, hidden_size: int = 256,
                 num_layers: int = 2, lr: float = 1e-4, device: str = "cpu",
                 output_activation=nn.Sigmoid):
        nn.Module.__init__(self)
        Individual.__init__(self)

        self.input_length = int(input_length)
        self.output_shape = tuple(output_shape)
        self.output_size = int(np.prod(self.output_shape))
        self.device = device
        self.version = 0

        blocks: List[nn.Module] = [nn.Linear(self.input_length, hidden_size), nn.LeakyReLU()]
        for _ in range(num_layers - 1):
            blocks += [nn.Linear(hidden_size, hidden_size), nn.LeakyReLU()]
        blocks += [nn.Linear(hidden_size, self.output_size)]

        self.net = nn.Sequential(*blocks).to(device)
        self.out_act = output_activation().to(device)
        self.opt = optim.Adam(self.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    # ---- forward / decode ---------------------------------------------------
    def forward(self, x) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        elif isinstance(x, list):
            x = torch.tensor(np.asarray(x)).float()
        x = x.to(self.device)
        out = self.out_act(self.net(x))
        return out.view(-1, *self.output_shape)

    def decode(self, genes_batch: np.ndarray) -> torch.Tensor:
        """(B, input_length) -> (B, *output_shape), no gradient."""
        with torch.no_grad():
            return self.forward(genes_batch)

    # ---- gradient channel ---------------------------------------------------
    def refine(self, sorted_pop, mode: TrainMode = TrainMode.SELF_DISTILL,
               percent: float = 0.4, batch_size: int = 32, epochs: int = 1) -> float:
        genes = np.stack([ind.genes for ind in sorted_pop]).astype(np.float32)
        n = len(sorted_pop)
        k = max(1, int(n * percent))
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
                    targets = self.forward(genes[:1]).detach().expand(k, *self.output_shape)
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

        self.version += 1
        return float(np.mean(losses)) if losses else 0.0

    # ---- evolution-strategy channel ----------------------------------------
    def evolve_step(self, sorted_pop, fitness_fn: Callable, n_candidates: int = 8,
                    percent: float = 0.4, sigma: float | None = None) -> float:
        k = max(1, int(len(sorted_pop) * percent))
        genes = torch.tensor(
            np.stack([ind.genes for ind in sorted_pop[:k]]).astype(np.float32),
            device=self.device,
        )
        sigma = sigma if sigma is not None else self.opt.param_groups[0]["lr"]
        base = [p.detach().clone() for p in self.parameters()]

        def population_fitness() -> float:
            return float(np.mean(fitness_fn(self.decode(genes))))

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
            self.version += 1
        return best_fit
