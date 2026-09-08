"""Deception-rate proxy analysis with persistent paired statistics.

This script re-scores existing output images with the locked WikiArt movement
classifier. It records per-cell hits, Wilson intervals, paired McNemar tests,
Bonferroni adjustment, and crossed content/style bootstrap intervals for the
paired proportion differences.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image
from scipy.stats import binomtest, chi2
from torchvision import transforms

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "checkpoints/wikiart_movement_r50.pth"
METHODS = ("ccstytr", "stytr2", "adain", "sanet", "cast")
BASELINES = ("stytr2", "adain", "sanet", "cast")
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
EXPECTED_CONTENTS = 40
EXPECTED_STYLES = 20
EXPECTED_CELLS = EXPECTED_CONTENTS * EXPECTED_STYLES
BOOTSTRAP_SEED = 20260901
N_BOOTSTRAP = 10_000


def load_model(device: torch.device):
    checkpoint = torch.load(CKPT, weights_only=False, map_location=device)
    classes = checkpoint["classes"]
    model = timm.create_model(
        "resnet50.a1_in1k",
        pretrained=False,
        num_classes=len(classes),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    print(
        f"classifier val_top1={checkpoint.get('val_top1', float('nan')):.4f}, "
        f"{len(classes)} movements"
    )
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return model, classes, transform


def predict(model, transform, paths, device: torch.device, batch: int = 64):
    predictions = []
    for index in range(0, len(paths), batch):
        chunk = paths[index : index + batch]
        inputs = torch.stack([transform(Image.open(path).convert("RGB")) for path in chunk])
        with torch.no_grad(), torch.autocast(
            device_type=device.type,
            enabled=device.type == "cuda",
        ):
            predictions += model(inputs.to(device)).argmax(1).tolist()
    return predictions


def wilson(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    proportion = hits / n
    denominator = 1 + z * z / n
    center = proportion + z * z / (2 * n)
    distance = z * ((proportion * (1 - proportion) + z * z / (4 * n)) / n) ** 0.5
    return (center - distance) / denominator, (center + distance) / denominator


def _bootstrap_counts(
    rng: np.random.Generator,
    replicates: int,
    levels: int,
) -> np.ndarray:
    draws = rng.integers(0, levels, size=(replicates, levels))
    counts = np.zeros((replicates, levels), dtype=np.int16)
    row_indices = np.repeat(np.arange(replicates), levels)
    np.add.at(counts, (row_indices, draws.ravel()), 1)
    return counts


def crossed_bootstrap_ci(
    rows: list[dict],
    seed: int,
    n_bootstrap: int,
) -> tuple[float, float]:
    contents = sorted({row["content"] for row in rows})
    styles = sorted({row["style"] for row in rows})
    if len(contents) != EXPECTED_CONTENTS or len(styles) != EXPECTED_STYLES:
        raise ValueError("deception bootstrap received an unexpected crossed design")
    content_index = {value: index for index, value in enumerate(contents)}
    style_index = {value: index for index, value in enumerate(styles)}
    matrix = np.full((EXPECTED_CONTENTS, EXPECTED_STYLES), np.nan)
    for row in rows:
        matrix[content_index[row["content"]], style_index[row["style"]]] = row["difference"]
    if np.isnan(matrix).any():
        raise ValueError("deception bootstrap matrix is incomplete")

    rng = np.random.default_rng(seed)
    content_counts = _bootstrap_counts(rng, n_bootstrap, EXPECTED_CONTENTS)
    style_counts = _bootstrap_counts(rng, n_bootstrap, EXPECTED_STYLES)
    replicates = np.einsum(
        "bi,ij,bj->b",
        content_counts,
        matrix,
        style_counts,
        optimize=True,
    ) / EXPECTED_CELLS
    return float(np.quantile(replicates, 0.025)), float(np.quantile(replicates, 0.975))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results/main")
    parser.add_argument("--style_dir", default="data/eval/general/style")
    parser.add_argument("--content_dir", default="data/eval/general/content")
    parser.add_argument("--out", default=None, help="default: <results_dir>/deception.csv")
    parser.add_argument(
        "--pairwise-out",
        default=None,
        help="default: <results_dir>/statistics/deception_pairwise.csv",
    )
    parser.add_argument(
        "--hits-out",
        default=None,
        help="default: <results_dir>/statistics/deception_hits_per_pair.csv",
    )
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--ref", default="ccstytr")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    args = parser.parse_args()

    methods = tuple(method.strip() for method in args.methods.split(",") if method.strip())
    if methods != METHODS or args.ref != "ccstytr":
        raise ValueError(f"locked analysis requires methods={METHODS} and ref=ccstytr")
    if args.bootstrap_seed != BOOTSTRAP_SEED or args.n_bootstrap != N_BOOTSTRAP:
        raise ValueError(
            f"locked analysis requires seed={BOOTSTRAP_SEED}, n_bootstrap={N_BOOTSTRAP}"
        )

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    model, classes, transform = load_model(device)
    class_index = {name: index for index, name in enumerate(classes)}

    style_target = {}
    for style in sorted(Path(args.style_dir).iterdir()):
        if style.suffix.lower() not in IMAGE_EXTS:
            continue
        movement = style.stem.split("__", 1)[0]
        if movement not in class_index:
            raise ValueError(f"style movement {movement!r} is not in classifier classes")
        style_target[style.stem] = (class_index[movement], style)
    if len(style_target) != EXPECTED_STYLES:
        raise ValueError(f"expected {EXPECTED_STYLES} styles, found {len(style_target)}")

    summary_rows = []
    hit_rows: list[dict] = []

    def score_paths(tag, paths, targets):
        predictions = predict(model, transform, paths, device, args.batch)
        hits = [int(prediction == target) for prediction, target in zip(predictions, targets)]
        hit_count, n = sum(hits), len(hits)
        low, high = wilson(hit_count, n)
        summary_rows.append(
            {
                "group": tag,
                "n": n,
                "hits": hit_count,
                "rate": hit_count / n,
                "ci95_lo": low,
                "ci95_hi": high,
            }
        )
        print(f"{tag:<12} deception rate {hit_count / n:.4f} (95% CI {low:.4f}-{high:.4f}, n={n})")
        return hits

    style_stems = sorted(style_target)
    score_paths(
        "STYLES(ref)",
        [style_target[stem][1] for stem in style_stems],
        [style_target[stem][0] for stem in style_stems],
    )

    contents = sorted(
        path
        for path in Path(args.content_dir).iterdir()
        if path.suffix.lower() in IMAGE_EXTS
    )
    if len(contents) != EXPECTED_CONTENTS:
        raise ValueError(f"expected {EXPECTED_CONTENTS} contents, found {len(contents)}")
    rng = random.Random(0)
    score_paths(
        "CONTENTS(ref)",
        contents,
        [rng.randrange(len(classes)) for _ in contents],
    )

    method_cells: dict[str, set[str]] = {}
    for method in methods:
        method_dir = Path(args.results_dir) / method
        files = sorted(
            path
            for path in method_dir.iterdir()
            if path.suffix.lower() in IMAGE_EXTS
        )
        if len(files) != EXPECTED_CELLS:
            raise ValueError(f"{method}: expected {EXPECTED_CELLS} outputs, found {len(files)}")
        paths, targets, metadata = [], [], []
        for output in files:
            content_stem, separator, style_stem = output.stem.partition("__")
            if not separator or style_stem not in style_target:
                raise ValueError(f"invalid or unmatched output cell: {output}")
            paths.append(output)
            targets.append(style_target[style_stem][0])
            metadata.append((output.stem, content_stem, style_stem))
        hits = score_paths(method, paths, targets)
        method_cells[method] = {cell for cell, _, _ in metadata}
        if len(method_cells[method]) != EXPECTED_CELLS:
            raise ValueError(f"{method}: duplicate output cell ids")
        for (cell, content, style), hit in zip(metadata, hits):
            hit_rows.append(
                {
                    "method": method,
                    "cell": cell,
                    "content": content,
                    "style": style,
                    "hit": hit,
                }
            )

    reference_cells = method_cells["ccstytr"]
    for method in methods:
        if method_cells[method] != reference_cells:
            raise ValueError(f"{method}: cell ids differ from ccstytr")

    out = Path(args.out) if args.out else Path(args.results_dir) / "deception.csv"
    pairwise_out = (
        Path(args.pairwise_out)
        if args.pairwise_out
        else Path(args.results_dir) / "statistics" / "deception_pairwise.csv"
    )
    hits_out = (
        Path(args.hits_out)
        if args.hits_out
        else Path(args.results_dir) / "statistics" / "deception_hits_per_pair.csv"
    )
    for path in (out, pairwise_out, hits_out):
        path.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    with hits_out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(hit_rows[0]))
        writer.writeheader()
        writer.writerows(hit_rows)

    hit_lookup = {
        method: {
            row["cell"]: row
            for row in hit_rows
            if row["method"] == method
        }
        for method in methods
    }
    pairwise_rows = []
    for baseline in BASELINES:
        differences = []
        b = 0
        c = 0
        for cell in sorted(reference_cells):
            reference_row = hit_lookup["ccstytr"][cell]
            baseline_row = hit_lookup[baseline][cell]
            reference_hit = reference_row["hit"]
            baseline_hit = baseline_row["hit"]
            b += int(reference_hit == 1 and baseline_hit == 0)
            c += int(reference_hit == 0 and baseline_hit == 1)
            differences.append(
                {
                    "content": reference_row["content"],
                    "style": reference_row["style"],
                    "difference": reference_hit - baseline_hit,
                }
            )
        discordant = b + c
        if discordant == 0:
            raise ValueError(f"{baseline}: no discordant pairs")
        statistic = (abs(b - c) - 1) ** 2 / discordant
        raw_p = float(chi2.sf(statistic, 1))
        exact_p = float(binomtest(b, discordant, 0.5, alternative="two-sided").pvalue)
        proportion_difference = (b - c) / EXPECTED_CELLS
        ci_low, ci_high = crossed_bootstrap_ci(
            differences,
            args.bootstrap_seed,
            args.n_bootstrap,
        )
        correction = 0.5 if b == 0 or c == 0 else 0.0
        matched_odds_ratio = (b + correction) / (c + correction)
        pairwise_rows.append(
            {
                "baseline": baseline,
                "n": EXPECTED_CELLS,
                "b_ref_hit_baseline_miss": b,
                "c_ref_miss_baseline_hit": c,
                "mcnemar_statistic": statistic,
                "raw_p": raw_p,
                "bonferroni_adjusted_p": min(1.0, len(BASELINES) * raw_p),
                "exact_binomial_p_sensitivity": exact_p,
                "paired_proportion_difference": proportion_difference,
                "difference_direction": "CC-StyTr rate - baseline rate",
                "crossed_bootstrap_ci95_low": ci_low,
                "crossed_bootstrap_ci95_high": ci_high,
                "bootstrap_seed": args.bootstrap_seed,
                "n_bootstrap": args.n_bootstrap,
                "matched_odds_ratio_b_over_c": matched_odds_ratio,
                "zero_cell_correction": correction,
                "mcnemar_method": "chi-square with continuity correction",
            }
        )

    with pairwise_out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pairwise_rows[0]))
        writer.writeheader()
        writer.writerows(pairwise_rows)
    print("wrote", out)
    print("wrote", hits_out)
    print("wrote", pairwise_out)


if __name__ == "__main__":
    main()
