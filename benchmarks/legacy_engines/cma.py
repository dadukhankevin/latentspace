"""A self-contained (mu/mu_w, lambda)-CMA-ES.

Hansen-tutorial implementation, validated against a 16-d sphere during the
benchmark campaign. Used by the universal solver's exploit phase; usable
standalone for any latent search.
"""
from __future__ import annotations

import numpy as np


def cma_minimize(evaluate_batch, dim, budget_evaluations, evaluations_done,
                 rng, mean0, sigma0, lam=None, stall_window=None,
                 stall_tol=0.01):
    """Minimize with CMA-ES; returns the number of generations run.

    `evaluate_batch(X) -> losses` must record evaluations itself; the loop
    stops exactly at the budget, discarding the final partial generation
    for distribution updates. With `stall_window` set, the loop also stops
    early once the best loss has improved less than `stall_tol` (relative)
    over that many generations — the same rule the explorer uses, so the
    two phases can hand the budget back and forth.
    """
    lam = lam if lam is not None else 4 + int(3 * np.log(dim))
    mu = lam // 2
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights /= weights.sum()
    mueff = 1.0 / np.sum(weights**2)
    cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
    cs = (mueff + 2) / (dim + mueff + 5)
    c1 = 2 / ((dim + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
    damps = 1 + 2 * max(0.0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
    chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))

    mean = np.array(mean0, dtype=np.float64)
    sigma = float(sigma0)
    covariance = np.eye(dim)
    ps = np.zeros(dim)
    pc = np.zeros(dim)
    generations = 0
    spent = evaluations_done
    best_history: list[float] = []

    while spent < budget_evaluations:
        covariance = (covariance + covariance.T) / 2
        eigenvalues, basis = np.linalg.eigh(covariance)
        eigenvalues = np.maximum(eigenvalues, 1e-20)
        scales = np.sqrt(eigenvalues)
        inv_sqrt_c = (basis / scales) @ basis.T

        n_sample = min(lam, budget_evaluations - spent)
        z = rng.standard_normal((n_sample, dim))
        y = z @ (basis * scales).T
        x = mean + sigma * y
        losses = evaluate_batch(x.astype(np.float32))
        spent += n_sample
        if n_sample < lam:
            break

        if stall_window is not None:
            gen_best = float(np.min(losses))
            best_history.append(min(gen_best, best_history[-1])
                                if best_history else gen_best)
            if len(best_history) > stall_window:
                then = best_history[-stall_window - 1]
                if then - best_history[-1] < stall_tol * abs(then):
                    break

        order = np.argsort(losses)
        selected = y[order[:mu]]
        y_weighted = weights @ selected
        mean = mean + sigma * y_weighted
        ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (inv_sqrt_c @ y_weighted)
        generations += 1
        hsig = (
            np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * generations)) / chi_n
            < 1.4 + 2 / (dim + 1)
        )
        pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * y_weighted
        covariance = (
            (1 - c1 - cmu) * covariance
            + c1 * (np.outer(pc, pc) + (not hsig) * cc * (2 - cc) * covariance)
            + cmu * (selected.T * weights) @ selected
        )
        sigma = sigma * np.exp((cs / damps) * (np.linalg.norm(ps) / chi_n - 1))

    return generations
