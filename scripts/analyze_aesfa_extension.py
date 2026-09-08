"""Analyze the separate CC-StyTr versus AesFA recent-baseline extension.

The original four-baseline, 20-test family remains immutable. This script
forms a new five-test family for AesFA, reports paired Wilcoxon tests,
rank-biserial effects, and a 10,000-replicate two-factor crossed bootstrap.
It accepts either the 40 x 20 generic set or the 120 x 30 poster set and
infers the crossed dimensions after strict completeness checks.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, wilcoxon

METHODS = ("ccstytr", "aesfa")
METRICS = ("ssim", "lpips", "e_ssim", "l_c", "l_s")
HIGHER_IS_BETTER = {"ssim": True, "lpips": False, "e_ssim": True, "l_c": False, "l_s": False}
BOOTSTRAP_SEED = 20260902
N_BOOTSTRAP = 10_000


def bootstrap_counts(rng: np.random.Generator, repeats: int, levels: int) -> np.ndarray:
    draws = rng.integers(0, levels, size=(repeats, levels))
    counts = np.zeros((repeats, levels), dtype=np.int16)
    rows = np.repeat(np.arange(repeats), levels)
    np.add.at(counts, (rows, draws.ravel()), 1)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    with Path(args.metrics_csv).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if {row["method"] for row in rows} != set(METHODS):
        raise RuntimeError("input must contain exactly ccstytr and aesfa")
    keyed = {(row["method"], row["content"], row["style"]): row for row in rows}
    if len(keyed) != len(rows):
        raise RuntimeError("duplicate method/content/style rows")
    contents = sorted({row["content"] for row in rows})
    styles = sorted({row["style"] for row in rows})
    expected_cells = {(content, style) for content in contents for style in styles}
    for method in METHODS:
        actual = {(row["content"], row["style"]) for row in rows if row["method"] == method}
        if actual != expected_cells:
            raise RuntimeError(f"{method} does not form a complete crossed design")

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    content_counts = bootstrap_counts(rng, N_BOOTSTRAP, len(contents))
    style_counts = bootstrap_counts(rng, N_BOOTSTRAP, len(styles))
    pairwise_rows = []
    summary_rows = []
    for metric in METRICS:
        reference = np.empty((len(contents), len(styles)), dtype=np.float64)
        baseline = np.empty_like(reference)
        for i, content in enumerate(contents):
            for j, style in enumerate(styles):
                reference[i, j] = float(keyed[("ccstytr", content, style)][metric])
                baseline[i, j] = float(keyed[("aesfa", content, style)][metric])
        difference = reference - baseline if HIGHER_IS_BETTER[metric] else baseline - reference
        flattened = difference.ravel()
        nonzero = flattened[flattened != 0]
        ranks = rankdata(np.abs(nonzero), method="average")
        w_plus = float(ranks[nonzero > 0].sum())
        w_minus = float(ranks[nonzero < 0].sum())
        test = wilcoxon(
            flattened,
            zero_method="wilcox",
            correction=False,
            alternative="two-sided",
            method="asymptotic",
        )
        replicates = np.einsum(
            "bi,ij,bj->b", content_counts, difference, style_counts, optimize=True
        ) / difference.size
        pairwise_rows.append(
            {
                "baseline": "aesfa",
                "metric": metric,
                "n": difference.size,
                "mean_paired_difference": difference.mean(),
                "median_paired_difference": np.median(difference),
                "w_statistic": test.statistic,
                "raw_p": test.pvalue,
                "bonferroni_adjusted_p": min(1.0, 5 * float(test.pvalue)),
                "rank_biserial_r": (w_plus - w_minus) / (w_plus + w_minus),
                "crossed_bootstrap_ci95_low": np.quantile(replicates, 0.025),
                "crossed_bootstrap_ci95_high": np.quantile(replicates, 0.975),
                "bootstrap_seed": BOOTSTRAP_SEED,
                "n_bootstrap": N_BOOTSTRAP,
                "difference_direction": "positive = CC-StyTr better",
            }
        )
        summary_rows.extend(
            [
                {"method": "ccstytr", "metric": metric, "n": reference.size, "mean": reference.mean()},
                {"method": "aesfa", "metric": metric, "n": baseline.size, "mean": baseline.mean()},
            ]
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, output_rows in (("summary.csv", summary_rows), ("pairwise.csv", pairwise_rows)):
        with (output_dir / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=output_rows[0].keys())
            writer.writeheader()
            writer.writerows(output_rows)
    print(
        f"saved AesFA extension: {len(contents)} contents x {len(styles)} styles -> {output_dir}"
    )


if __name__ == "__main__":
    main()
