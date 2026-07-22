"""CLIP ecosystem with emergent species and local gene flow.

There are no prompt-owned islands and no permanent species labels. Every
individual is scored against every positive prompt, producing its species
vector. Columns are normalized across the current population so the vector
measures what an individual is unusually good at, rather than which prompt has
the largest raw CLIP scale.

Mating uses one fixed rule: two individuals may mate when the RMS distance
between their normalized species vectors is at most ``--mating-radius``. A
parent with no compatible partner reproduces asexually. Species are therefore
only the connected components of this compatibility graph. A and C need not
mate directly for genes to flow between them when both can mate with B.

All founders share one decoder. Local decoder averaging plus crossover can
then diffuse useful structure through chains of compatible descendants while
the distance rule protects early behavioral divergence from global mixing.

This is a potential-check runner, not yet part of the public solver:

    python3 -m benchmarks.demo_clip_species_vector \
      --budget 20000 --seed 0 \
      --output benchmark_results/clip_species_vector_s0.json
"""

from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from benchmarks.compare import _require_mps, _seed_everything
from benchmarks.demo_clip_evolve import CLIP_MEAN, CLIP_STD, load_clip
from benchmarks.demo_clip_species import DEFAULT_POOL, SpeciesView
from benchmarks.round28_anchor_conv import ConvRGB
from benchmarks.round31_apple_animated import _png
from latentspace.universal.architectures import resolve
from benchmarks.legacy_engines.explorer import ExplorerConfig, _Template

LATENT = 64
SHAPE = (3, 96, 96)

EXTENDED_POOL = DEFAULT_POOL + [
    "an apple", "a banana", "a bicycle", "a castle",
    "a bird", "a fish", "a flower", "a chair",
    "a train", "a rocket", "a dog", "a violin",
    "a clock", "a shoe", "a camera", "a teapot",
]

DEFAULT_NEGATIVES = [
    "random visual noise",
    "television static",
    "a meaningless abstract texture",
    "a blurry color field",
    "a blank gray image",
    "text and symbols",
]


def negative_weight_at(evaluations: int, budget: int, start: float,
                       end: float, ramp_fraction: float) -> float:
    """Linear negative-prompt pressure, reaching `end` partway through.

    Starting weak lets descendants leave the common gray founder before a
    strong anti-noise constraint narrows the search. Once `ramp_fraction` of
    the budget is spent, the full penalty remains in force.
    """
    if budget <= 0:
        raise ValueError("budget must be positive")
    if start < 0 or end < start:
        raise ValueError("negative weights require 0 <= start <= end")
    if not 0 < ramp_fraction <= 1:
        raise ValueError("ramp_fraction must be in (0, 1]")
    progress = np.clip(evaluations / (budget * ramp_fraction), 0.0, 1.0)
    return float(start + (end - start) * progress)


def normalize_species_vectors(scores: np.ndarray,
                              epsilon: float = 1e-4) -> np.ndarray:
    """Column-wise z-scores, with uninformative near-constant columns zeroed.

    A prompt whose scores have effectively no spread cannot distinguish
    species. Zeroing it also prevents numerical noise from becoming a huge
    relative-fitness signal.
    """
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("scores must have shape (individuals, prompts)")
    mean = values.mean(axis=0, keepdims=True)
    scale = values.std(axis=0, keepdims=True)
    useful = scale[0] >= epsilon
    normalized = np.zeros_like(values)
    normalized[:, useful] = (
        (values[:, useful] - mean[:, useful]) / scale[:, useful]
    )
    return normalized.astype(np.float32)


def compatibility_graph(vectors: np.ndarray,
                        radius: float) -> tuple[np.ndarray, np.ndarray]:
    """Return adjacency and RMS pairwise distances in species-vector space."""
    values = np.asarray(vectors, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("vectors must have shape (individuals, dimensions)")
    if radius < 0:
        raise ValueError("radius must be non-negative")
    delta = values[:, None, :] - values[None, :, :]
    distance = np.sqrt(np.mean(delta * delta, axis=-1))
    adjacency = distance <= radius
    np.fill_diagonal(adjacency, False)
    return adjacency, distance


def connected_components(adjacency: np.ndarray) -> list[list[int]]:
    """Connected components of an undirected boolean compatibility graph."""
    graph = np.asarray(adjacency, dtype=bool)
    if graph.ndim != 2 or graph.shape[0] != graph.shape[1]:
        raise ValueError("adjacency must be square")
    unseen = set(range(len(graph)))
    components: list[list[int]] = []
    while unseen:
        root = min(unseen)
        unseen.remove(root)
        component, stack = [root], [root]
        while stack:
            node = stack.pop()
            neighbors = np.flatnonzero(graph[node])
            for neighbor in neighbors:
                value = int(neighbor)
                if value in unseen:
                    unseen.remove(value)
                    component.append(value)
                    stack.append(value)
        components.append(sorted(component))
    return sorted(components, key=lambda group: (-len(group), group[0]))


def choose_compatible_mates(vectors: np.ndarray, parents: np.ndarray,
                            radius: float, rng: np.random.Generator
                            ) -> tuple[np.ndarray, np.ndarray]:
    """Uniform compatible mate per parent; -1 means asexual fallback."""
    adjacency, distance = compatibility_graph(vectors, radius)
    parents = np.asarray(parents, dtype=np.int64)
    mates = np.full(len(parents), -1, dtype=np.int64)
    mate_distances = np.full(len(parents), np.nan, dtype=np.float64)
    for i, parent in enumerate(parents):
        candidates = np.flatnonzero(adjacency[parent])
        if len(candidates):
            mate = int(rng.choice(candidates))
            mates[i] = mate
            mate_distances[i] = distance[parent, mate]
    return mates, mate_distances


def graph_diagnostics(vectors: np.ndarray, radius: float) -> dict:
    adjacency, distance = compatibility_graph(vectors, radius)
    components = connected_components(adjacency)
    edges = np.transpose(np.triu(adjacency, 1).nonzero())
    edge_distances = ([float(distance[a, b]) for a, b in edges]
                      if len(edges) else [])
    possible = max(len(vectors) * (len(vectors) - 1) // 2, 1)
    return {
        "components": len(components),
        "component_sizes": [len(group) for group in components],
        "largest_component": len(components[0]) if components else 0,
        "edges": int(len(edges)),
        "edge_density": float(len(edges) / possible),
        "mean_edge_distance": (float(np.mean(edge_distances))
                               if edge_distances else None),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", nargs="*", default=EXTENDED_POOL)
    parser.add_argument("--neg", nargs="*", default=DEFAULT_NEGATIVES)
    parser.add_argument("--neg-start", type=float, default=0.1,
                        help="negative-prompt weight at the beginning")
    parser.add_argument("--neg-end", type=float, default=1.0,
                        help="negative-prompt weight after the ramp")
    parser.add_argument("--neg-ramp-fraction", type=float, default=0.7,
                        help="fraction of budget over which pressure ramps")
    parser.add_argument("--model",
                        default="wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M")
    parser.add_argument("--clip-res", type=int, default=128)
    parser.add_argument("--survivors", type=int, default=48)
    parser.add_argument("--children", type=int, default=192)
    parser.add_argument("--mating-radius", type=float, default=0.3,
                        help="maximum RMS normalized species-vector distance")
    parser.add_argument("--budget", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start-gain", type=float, default=1.0)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--live", action="store_true")
    parser.add_argument(
        "--hold-open", action="store_true",
        help="keep the live progress window open after the run finishes",
    )
    parser.add_argument("--gif", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.survivors < 2:
        raise ValueError("--survivors must be at least 2")
    if args.children < args.survivors:
        raise ValueError("--children must be at least --survivors")
    if args.budget < args.survivors:
        raise ValueError("--budget must cover the founder population")
    # Validate the schedule before loading a model or opening a window.
    negative_weight_at(0, args.budget, args.neg_start, args.neg_end,
                       args.neg_ramp_fraction)

    _require_mps()
    device = "mps"
    prompts = list(args.pool)
    prompt_count = len(prompts)
    if not prompt_count:
        raise ValueError("--pool must contain at least one prompt")
    config = ExplorerConfig()

    print(f"loading CLIP: {args.model}", flush=True)
    model, processor = load_clip(args.model, device)
    dtype = next(model.parameters()).dtype
    resolution = args.clip_res or model.config.vision_config.image_size
    mean = CLIP_MEAN.to(device, dtype)
    std = CLIP_STD.to(device, dtype)

    with torch.no_grad():
        tokens = processor(text=prompts + args.neg, return_tensors="pt",
                           padding=True)
        output = model.text_model(
            **{key: value.to(device) for key, value in tokens.items()})
        text = F.normalize(model.text_projection(output.pooler_output).float(),
                           dim=-1)
    text_positive = text[:prompt_count]
    text_negative = text[prompt_count:]

    def score_components(phenotypes: torch.Tensor
                         ) -> tuple[np.ndarray, np.ndarray]:
        """Positive-prompt matrix and shared negative-prompt score.

        Keeping the components separate lets every survivor and hall champion
        be rescored for free as negative pressure increases.
        """
        with torch.no_grad():
            images = phenotypes.reshape(len(phenotypes), *SHAPE)
            images = F.interpolate(
                images, size=(resolution, resolution), mode="bicubic",
                align_corners=False).clamp(0, 1).to(dtype)
            vision = model.vision_model(
                pixel_values=(images - mean) / std,
                interpolate_pos_encoding=True)
            embeddings = F.normalize(
                model.visual_projection(vision.pooler_output).float(), dim=-1)
            positive = embeddings @ text_positive.T
            if len(text_negative):
                negative = (embeddings @ text_negative.T).mean(dim=1)
            else:
                negative = torch.zeros(len(embeddings), device=device)
        return (positive.cpu().numpy().astype(np.float32),
                negative.cpu().numpy().astype(np.float32))

    def combine_scores(positive: np.ndarray, negative: np.ndarray,
                       weight: float) -> np.ndarray:
        return positive - weight * negative[:, None]

    _seed_everything(args.seed)
    rng = np.random.default_rng(args.seed)
    template = _Template(resolve(lambda latent, shape: ConvRGB(latent, shape),
                                 LATENT, SHAPE), device)

    # One ancestral decoder, many genomes: a single seed species. Decoder
    # diversity is created only by descendant mutation and local averaging.
    founder_theta = template.init_theta(int(rng.integers(0, 2**31)))
    population_z = rng.standard_normal(
        (args.survivors, LATENT)).astype(np.float32)
    population_theta = np.repeat(
        founder_theta[None], args.survivors, axis=0)
    founder_phenotypes = template.decode_batch(population_theta, population_z)
    population_positive, population_negative = score_components(
        founder_phenotypes)
    spent = args.survivors
    weight = negative_weight_at(
        spent, args.budget, args.neg_start, args.neg_end,
        args.neg_ramp_fraction)
    population_scores = combine_scores(
        population_positive, population_negative, weight)

    hall = [{"positive": -np.inf, "negative": 0.0, "image": None}
            for _ in prompts]

    def hall_scores(at_weight: float) -> list[float]:
        return [float(entry["positive"] - at_weight * entry["negative"])
                for entry in hall]

    def update_hall(phenotypes: torch.Tensor, positive: np.ndarray,
                    negative: np.ndarray, at_weight: float) -> bool:
        improved = False
        for prompt in range(prompt_count):
            scores = positive[:, prompt] - at_weight * negative
            best = int(np.argmax(scores))
            value = float(scores[best])
            old_value = float(
                hall[prompt]["positive"]
                - at_weight * hall[prompt]["negative"])
            if value > old_value + 1e-9:
                hall[prompt] = {
                    "positive": float(positive[best, prompt]),
                    "negative": float(negative[best]),
                    "image": phenotypes[best].detach().cpu().numpy()
                    .reshape(*SHAPE),
                }
                improved = True
        return improved

    update_hall(founder_phenotypes, population_positive,
                population_negative, weight)
    gain = float(args.start_gain)
    generation = 0
    global_stall = 0
    frames: list[dict] = []
    trace: list[dict] = []
    frame_interval = max(1, args.budget // max(args.frames, 1))
    next_frame = 0
    view = SpeciesView(prompts, args.budget) if args.live else None

    def mutate_genome(genome: np.ndarray) -> np.ndarray:
        mask = rng.random(genome.shape) < config.genome_mutation_rate
        if not mask.any():
            mask[rng.integers(0, len(genome))] = True
        return (genome + mask * rng.normal(
            0, config.genome_mutation_sigma * gain, genome.shape)
                ).astype(np.float32)

    def crossover(base: np.ndarray, donor: np.ndarray) -> np.ndarray:
        cut = int(rng.integers(1, LATENT))
        child = base.copy()
        child[cut:] = donor[cut:]
        return child.astype(np.float32)

    while spent < args.budget:
        generation += 1
        count = min(args.children, args.budget - spent)
        # Recompute the parents' current fitness without another CLIP pass.
        # The schedule changes the objective, so stale scores are invalid.
        mating_weight = negative_weight_at(
            spent, args.budget, args.neg_start, args.neg_end,
            args.neg_ramp_fraction)
        population_scores = combine_scores(
            population_positive, population_negative, mating_weight)
        parent = rng.integers(0, len(population_z), count)
        parent_vectors = normalize_species_vectors(population_scores)
        mates, mate_distances = choose_compatible_mates(
            parent_vectors, parent, args.mating_radius, rng)
        sexual = mates >= 0

        child_z = np.empty((count, LATENT), dtype=np.float32)
        base_theta = population_theta[parent].copy()
        for i, (base, mate) in enumerate(zip(parent, mates)):
            if mate >= 0:
                child_z[i] = mutate_genome(crossover(
                    population_z[base], population_z[mate]))
                base_theta[i] = (
                    population_theta[base] + population_theta[mate]) / 2.0
            else:
                child_z[i] = mutate_genome(population_z[base])

        sigmas = np.exp(rng.uniform(
            np.log(config.weight_sigma_low),
            np.log(config.weight_sigma_high), count)) * gain
        scales = np.maximum(base_theta.std(axis=1), 1e-3)
        child_theta = (
            base_theta
            + (sigmas * scales)[:, None]
            * rng.standard_normal((count, template.n_params))
        ).astype(np.float32)

        child_phenotypes = template.decode_batch(child_theta, child_z)
        child_positive, child_negative = score_components(child_phenotypes)
        spent += count
        weight = negative_weight_at(
            spent, args.budget, args.neg_start, args.neg_end,
            args.neg_ramp_fraction)
        child_scores = combine_scores(child_positive, child_negative, weight)
        parent_scores_at_weight = combine_scores(
            population_positive, population_negative, weight)
        improved = update_hall(
            child_phenotypes, child_positive, child_negative, weight)

        # Relative fitness chooses the prompt on which this child most exceeds
        # its contemporaries. This is Daniel's diversity pressure: crowded
        # capabilities stop paying merely because their raw score is high.
        child_vectors = normalize_species_vectors(child_scores)
        goals = child_vectors.argmax(axis=1)
        relative_fitness = child_vectors[np.arange(count), goals]

        # Step-size success remains an absolute inherited comparison: did the
        # child improve the niche its PARENT occupied? Using the child's new
        # argmax here is corrupt — every child gets to cherry-pick whichever
        # prompt makes its mutation look successful, which drove the gain to
        # 600x and saturated the first potential-check population by 10k.
        parent_goals_all = parent_vectors.argmax(axis=1)
        inherited_goals = parent_goals_all[parent]
        child_quality = child_scores[np.arange(count), inherited_goals]
        parent_quality = parent_scores_at_weight[parent, inherited_goals]
        win_rate = float((child_quality >= parent_quality - 1e-12).mean())
        gain *= (config.gain_step if win_rate > config.win_target
                 else 1 / config.gain_step)
        gain = float(np.clip(gain, 0.3, config.gain_limits[1]))
        global_stall = 0 if improved else global_stall + 1
        if global_stall >= 25:
            gain = min(gain * 3.0, config.gain_limits[1])
            global_stall = 0

        # The final partial generation may be smaller than the survivor set;
        # its evaluations still count and update the hall, but it cannot form
        # a complete next population.
        if count >= args.survivors:
            keep = np.argsort(-relative_fitness)[:args.survivors]
            population_z = child_z[keep]
            population_theta = child_theta[keep]
            population_positive = child_positive[keep]
            population_negative = child_negative[keep]
            population_scores = child_scores[keep]
        else:
            population_scores = parent_scores_at_weight

        survivor_vectors = normalize_species_vectors(population_scores)
        graph = graph_diagnostics(survivor_vectors, args.mating_radius)
        survivor_goals = survivor_vectors.argmax(axis=1)
        goal_counts = np.bincount(
            survivor_goals, minlength=prompt_count).astype(int).tolist()
        theta_center = population_theta.mean(axis=0, keepdims=True)
        decoder_spread = float(
            np.sqrt(np.mean((population_theta - theta_center) ** 2))
            / max(np.sqrt(np.mean(population_theta ** 2)), 1e-12))
        row = {
            "e": spent,
            "generation": generation,
            "negative_weight": weight,
            "gain": gain,
            "win_rate": win_rate,
            "sexual_fraction": float(sexual.mean()),
            "mean_mating_distance": (float(np.nanmean(mate_distances))
                                     if sexual.any() else None),
            "decoder_spread": decoder_spread,
            "goal_counts": goal_counts,
            **graph,
        }
        trace.append(row)

        if spent >= next_frame:
            scores = hall_scores(weight)
            frame = {**row, "scores": scores}
            for prompt in range(prompt_count):
                if hall[prompt]["image"] is not None:
                    frame[f"p{prompt}"] = _png(
                        hall[prompt]["image"].reshape(-1))
            frames.append(frame)
            next_frame += frame_interval
            leader = int(np.argmax(scores))
            occupied = sum(count > 0 for count in goal_counts)
            print(
                f"  {spent:>7} evals  gain {gain:.2f}  "
                f"neg {weight:.2f}  "
                f"best {prompts[leader]} {scores[leader]:.3f}  "
                f"mean {np.mean(scores):.3f}  niches {occupied}  "
                f"components {graph['components']} "
                f"{graph['component_sizes'][:5]}  "
                f"sexual {sexual.mean():.0%}",
                flush=True,
            )
        if view is not None and generation % 5 == 0:
            view.update(
                spent, prompts, [entry["image"] for entry in hall],
                hall_scores(weight),
                set(int(goal) for goal in survivor_goals),
            )

    final_weight = negative_weight_at(
        args.budget, args.budget, args.neg_start, args.neg_end,
        args.neg_ramp_fraction)
    records = {prompt: score for prompt, score
               in zip(prompts, hall_scores(final_weight))}
    print("\nFINAL prompt records:")
    for prompt in sorted(prompts, key=records.get, reverse=True):
        print(f"  {prompt:<20} {records[prompt]:.4f}")
    final_graph = graph_diagnostics(
        normalize_species_vectors(population_scores), args.mating_radius)
    print(f"FINAL compatibility graph: {final_graph}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "method": "species_vector_local_mating",
            "pool": prompts,
            "neg": args.neg,
            "neg_start": args.neg_start,
            "neg_end": args.neg_end,
            "neg_ramp_fraction": args.neg_ramp_fraction,
            "model": args.model,
            "clip_res": args.clip_res,
            "budget": args.budget,
            "seed": args.seed,
            "survivors": args.survivors,
            "children": args.children,
            "mating_radius": args.mating_radius,
            "start_gain": args.start_gain,
            "records": records,
            "final_graph": final_graph,
            "trace": trace,
            "frames": frames,
            "torch_version": torch.__version__,
        }, indent=2) + "\n")
        print(f"wrote {args.output}")

    if args.gif and frames:
        from PIL import ImageDraw
        columns = min(8, prompt_count)
        rows = int(np.ceil(prompt_count / columns))
        images = []
        for frame in frames:
            sheet = Image.new(
                "RGB", (128 * columns, 128 * rows), (20, 20, 20))
            for prompt in range(prompt_count):
                if f"p{prompt}" not in frame:
                    continue
                tile = Image.open(io.BytesIO(base64.b64decode(
                    frame[f"p{prompt}"].split(",", 1)[1]))).convert("RGB")
                tile = tile.resize((128, 128), Image.NEAREST)
                draw = ImageDraw.Draw(tile)
                draw.rectangle([0, 116, 128, 128], fill=(0, 0, 0))
                draw.text(
                    (3, 117),
                    f"{prompts[prompt]} {frame['scores'][prompt]:.2f}",
                    fill=(255, 255, 255),
                )
                sheet.paste(
                    tile, (128 * (prompt % columns),
                           128 * (prompt // columns)))
            images.append(sheet)
        images.append(images[-1])
        args.gif.parent.mkdir(parents=True, exist_ok=True)
        images[0].save(
            args.gif, save_all=True, append_images=images[1:],
            duration=[80] * (len(images) - 1) + [2500], loop=0,
            optimize=True,
        )
        print(f"wrote {args.gif}")

    if view is not None:
        view.plt.ioff()
        if args.hold_open:
            view.plt.show()
        else:
            view.plt.close(view.fig)


if __name__ == "__main__":
    main()
