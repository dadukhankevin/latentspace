"""Gym control — the universal GA as a policy optimizer (2026-07-26).

The phenotype IS a policy network's weight vector, so the shared decoder
becomes a network that writes networks. Fitness is episode return: a black
box with no gradient path, which is the framing's first condition, and
unlike the terrain demo the family here is credible — the same task under
different physics is a genuinely related instance, not a different problem
wearing the same name.

Three arms at MATCHED evaluation budget, because "we solved CartPole" is
not a result. Direct search on the weights is what the decoder has to beat
for the indirection to be worth anything:

  decoder  the library: genes + latents -> shared decoder -> policy weights
  es       a (mu/mu, lambda) Gaussian ES straight on the weight vector,
           step size on the 1/5th success rule
  random   uniform weight vectors, same budget, best kept

Fitness is made DETERMINISTIC by scoring every individual on the same fixed
episode seeds — a noisy objective silently breaks selection, since an
individual can win on luck and then anchor the population. Generalization
is then checked on 100 seeds nobody optimized against, which is where an
arm that merely memorized its training seeds gets caught.

    python3 -m benchmarks.gym_control --env CartPole-v1 --arm decoder
    python3 -m benchmarks.gym_control --env Acrobot-v1 --all

Needs gymnasium on the path:
    PYTHONPATH=~/.local/gymlibs python3 -m benchmarks.gym_control ...
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from latentspace.universal import solve, register_architecture
from latentspace.universal.architectures import build_mlp

HIDDEN = 8
WEIGHT_SCALE = 4.0          # phenotype [0,1] -> weight [-2, 2]
TRAIN_EPISODES = 3
TEST_EPISODES = 100
SOLVED = {"CartPole-v1": 475.0, "Acrobot-v1": -100.0,
          "Pendulum-v1": -200.0, "MountainCarContinuous-v0": 90.0}


def env_spec(name):
    import gymnasium as gym
    env = gym.make(name)
    obs_dim = int(np.prod(env.observation_space.shape))
    discrete = hasattr(env.action_space, "n")
    if discrete:
        act_dim, high = int(env.action_space.n), None
    else:
        act_dim = int(np.prod(env.action_space.shape))
        high = float(env.action_space.high[0])
    env.close()
    n_params = obs_dim * HIDDEN + HIDDEN + HIDDEN * act_dim + act_dim
    return obs_dim, act_dim, discrete, high, n_params


def split(weights, obs_dim, act_dim):
    i = 0
    w1 = weights[i:i + obs_dim * HIDDEN].reshape(obs_dim, HIDDEN); i += obs_dim * HIDDEN
    b1 = weights[i:i + HIDDEN]; i += HIDDEN
    w2 = weights[i:i + HIDDEN * act_dim].reshape(HIDDEN, act_dim); i += HIDDEN * act_dim
    b2 = weights[i:i + act_dim]
    return w1, b1, w2, b2


def rollout(weights, env, spec, seeds):
    """Mean return over the given episode seeds. The whole objective."""
    obs_dim, act_dim, discrete, high, _ = spec
    w1, b1, w2, b2 = split(weights, obs_dim, act_dim)
    total = 0.0
    for s in seeds:
        obs, _ = env.reset(seed=int(s))
        done = False
        while not done:
            h = np.tanh(np.asarray(obs, dtype=np.float64) @ w1 + b1)
            out = h @ w2 + b2
            action = int(np.argmax(out)) if discrete \
                else np.clip(np.tanh(out) * high, -high, high)
            obs, reward, term, trunc, _ = env.step(action)
            total += float(reward)
            done = term or trunc
    return total / len(seeds)


def make_scorer(name, spec, seeds):
    import gymnasium as gym
    env = gym.make(name)

    def score(weights):
        return rollout(weights, env, spec, seeds)
    return score, env


def to_weights(pheno):
    return (np.asarray(pheno, dtype=np.float64).reshape(-1) - 0.5) * WEIGHT_SCALE


# ------------------------------------------------------------------ arms
def arm_decoder(name, spec, seeds, epochs, seed, gain=10.0):
    n_params = spec[4]

    def build(latent, output_shape):
        net = build_mlp(latent, output_shape)
        with torch.no_grad():
            net[-1].weight.mul_(gain)      # a near-constant phenotype is a
            net[-1].bias.mul_(gain)        # constant policy (FINDINGS 14)
        return net
    register_architecture("policy", build)
    score, env = make_scorer(name, spec, seeds)

    def fitness(phenotypes):
        flat = phenotypes.reshape(len(phenotypes), -1).cpu().numpy()
        return torch.tensor([score(to_weights(p)) for p in flat],
                            dtype=torch.float32)

    result = solve(fitness, output_shape=(n_params,), epochs=epochs,
                   architecture="policy", seed=seed, device="cpu",
                   population_cap=96, children=48)
    env.close()
    return to_weights(result.best_phenotype), result.best_fitness, result.evaluations


def arm_es(name, spec, seeds, budget, seed):
    """(mu/mu, lambda) Gaussian ES on the weights, 1/5th-rule step size."""
    n_params = spec[4]
    rng = np.random.default_rng(seed)
    score, env = make_scorer(name, spec, seeds)
    lam, mu, sigma = 48, 12, 0.5
    mean = rng.normal(0, 0.5, n_params)
    best, best_score, spent = mean.copy(), score(mean), 1
    while spent + lam <= budget:
        kids = mean + sigma * rng.normal(0, 1, (lam, n_params))
        vals = np.array([score(k) for k in kids]); spent += lam
        order = np.argsort(-vals)
        mean = kids[order[:mu]].mean(axis=0)
        if vals[order[0]] > best_score:
            best_score, best = vals[order[0]], kids[order[0]].copy()
        # 1/5th rule: expand while better than half the children improve
        improved = float((vals > best_score).mean())
        sigma *= 1.15 if improved > 0.2 else 1 / 1.05
        sigma = float(np.clip(sigma, 1e-3, 3.0))
    env.close()
    return best, best_score, spent


def arm_random(name, spec, seeds, budget, seed):
    n_params = spec[4]
    rng = np.random.default_rng(seed)
    score, env = make_scorer(name, spec, seeds)
    best, best_score = None, -np.inf
    for _ in range(budget):
        cand = rng.uniform(-2, 2, n_params)
        val = score(cand)
        if val > best_score:
            best, best_score = cand, val
    env.close()
    return best, best_score, budget


def generalization(name, spec, weights):
    """Return on 100 seeds nobody optimized against."""
    score, env = make_scorer(name, spec, np.arange(1000, 1000 + TEST_EPISODES))
    val = score(weights)
    env.close()
    return val


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="CartPole-v1")
    parser.add_argument("--arm", default="decoder",
                        choices=("decoder", "es", "random"))
    parser.add_argument("--all", action="store_true", help="run all three arms")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--hidden", type=int, default=HIDDEN,
                        help="policy width. The interesting axis: direct "
                             "search degrades as the weight vector grows, "
                             "so this is where an indirect encoding has to "
                             "start paying for itself.")
    parser.add_argument("--seed", type=int, default=3)
    args = parser.parse_args()

    globals()["HIDDEN"] = args.hidden
    spec = env_spec(args.env)
    seeds = np.arange(TRAIN_EPISODES)
    print(f"{args.env}: obs {spec[0]}, actions {spec[1]}, "
          f"{'discrete' if spec[2] else 'continuous'} -> "
          f"{spec[4]}-parameter policy")
    bar = SOLVED.get(args.env)
    if bar is not None:
        print(f"  reference: 'solved' is about {bar}")

    # the decoder arm defines the budget; the others are matched to it
    began = time.time()
    w, train, evals = arm_decoder(args.env, spec, seeds, args.epochs, args.seed)
    rows = [("decoder", train, generalization(args.env, spec, w), evals,
             time.time() - began)]
    if args.all:
        for label, fn in (("es", arm_es), ("random", arm_random)):
            began = time.time()
            w, train, spent = fn(args.env, spec, seeds, evals, args.seed)
            rows.append((label, train, generalization(args.env, spec, w),
                         spent, time.time() - began))

    print(f"\n  {'arm':<9}{'train':>10}{'held-out':>11}{'evals':>9}{'time':>8}")
    for label, train, held, spent, secs in rows:
        print(f"  {label:<9}{train:>10.1f}{held:>11.1f}{spent:>9}{secs:>7.0f}s")


if __name__ == "__main__":
    main()
