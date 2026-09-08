"""Submission-grade statistics for the locked CC-StyTr main experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.stats import norm, rankdata, wilcoxon

METHODS = ("ccstytr", "stytr2", "adain", "sanet", "cast")
BASELINES = ("stytr2", "adain", "sanet", "cast")
METRICS = ("ssim", "lpips", "e_ssim", "l_c", "l_s")
HIGHER_IS_BETTER = {
    "ssim": True,
    "lpips": False,
    "e_ssim": True,
    "l_c": False,
    "l_s": False,
}
EXPECTED_CONTENTS = 40
EXPECTED_STYLES = 20
EXPECTED_CELLS = EXPECTED_CONTENTS * EXPECTED_STYLES
BONFERRONI_FAMILY = len(BASELINES) * len(METRICS)
BOOTSTRAP_SEED = 20260901
N_BOOTSTRAP = 10_000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_and_validate(path: Path) -> pd.DataFrame:
    required = {"method", "output", "cell", "content", "style", *METRICS}
    string_columns = ("method", "output", "cell", "content", "style")
    frame = pd.read_csv(path, dtype={key: "string" for key in string_columns})
    missing_columns = sorted(required - set(frame.columns))
    if missing_columns:
        raise ValueError(f"missing columns: {missing_columns}")
    if len(frame) != len(METHODS) * EXPECTED_CELLS:
        raise ValueError(f"expected {len(METHODS) * EXPECTED_CELLS} rows, found {len(frame)}")
    if frame["method"].isna().any() or frame["cell"].isna().any():
        raise ValueError("method/cell contains missing values")
    if set(frame["method"].unique()) != set(METHODS):
        raise ValueError(f"unexpected methods: {sorted(frame['method'].unique())}")
    if frame.duplicated(["method", "cell"]).any():
        duplicate = frame.loc[
            frame.duplicated(["method", "cell"], keep=False), ["method", "cell"]
        ]
        raise ValueError(f"duplicate method-cell rows: {duplicate.head().to_dict('records')}")

    expected_cell = frame["content"].astype(str) + "__" + frame["style"].astype(str)
    mismatched = frame.loc[
        frame["cell"].astype(str) != expected_cell, ["cell", "content", "style"]
    ]
    if not mismatched.empty:
        raise ValueError(f"cell decomposition mismatch: {mismatched.head().to_dict('records')}")

    for metric in METRICS:
        frame[metric] = pd.to_numeric(frame[metric], errors="raise")
        if not np.isfinite(frame[metric].to_numpy(dtype=float)).all():
            raise ValueError(f"{metric} contains NaN or infinity")

    reference_cells: set[str] | None = None
    for method in METHODS:
        subset = frame.loc[frame["method"] == method]
        cells = set(subset["cell"].astype(str))
        if len(cells) != EXPECTED_CELLS:
            raise ValueError(f"{method}: expected {EXPECTED_CELLS} cells, found {len(cells)}")
        if subset["content"].nunique() != EXPECTED_CONTENTS:
            raise ValueError(f"{method}: unexpected content count {subset['content'].nunique()}")
        if subset["style"].nunique() != EXPECTED_STYLES:
            raise ValueError(f"{method}: unexpected style count {subset['style'].nunique()}")
        if reference_cells is None:
            reference_cells = cells
        elif cells != reference_cells:
            raise ValueError(f"{method}: cell ids differ from ccstytr")
    return frame


def descriptive_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for method in METHODS:
        subset = frame.loc[frame["method"] == method]
        for metric in METRICS:
            values = subset[metric].to_numpy(dtype=float)
            std = float(values.std(ddof=1))
            half_width = 1.96 * std / math.sqrt(len(values))
            rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "n": len(values),
                    "mean": float(values.mean()),
                    "std": std,
                    "median": float(np.median(values)),
                    "q1": float(np.quantile(values, 0.25)),
                    "q3": float(np.quantile(values, 0.75)),
                    "cell_level_ci95_low": float(values.mean() - half_width),
                    "cell_level_ci95_high": float(values.mean() + half_width),
                    "ci_scope": "descriptive cell-level CI",
                }
            )
    return pd.DataFrame(rows)


def paired_differences(frame: pd.DataFrame, baseline: str, metric: str) -> pd.DataFrame:
    reference = (
        frame.loc[frame["method"] == "ccstytr", ["cell", "content", "style", metric]]
        .rename(columns={metric: "reference"})
        .sort_values("cell")
    )
    comparator = (
        frame.loc[frame["method"] == baseline, ["cell", "content", "style", metric]]
        .rename(
            columns={
                "content": "baseline_content",
                "style": "baseline_style",
                metric: "baseline",
            }
        )
        .sort_values("cell")
    )
    paired = reference.merge(comparator, on="cell", validate="one_to_one")
    if len(paired) != EXPECTED_CELLS:
        raise ValueError(f"{baseline}/{metric}: expected {EXPECTED_CELLS} matched cells")
    if not (paired["content"].astype(str) == paired["baseline_content"].astype(str)).all():
        raise ValueError(f"{baseline}/{metric}: content identity mismatch")
    if not (paired["style"].astype(str) == paired["baseline_style"].astype(str)).all():
        raise ValueError(f"{baseline}/{metric}: style identity mismatch")
    if HIGHER_IS_BETTER[metric]:
        paired["difference"] = paired["reference"] - paired["baseline"]
    else:
        paired["difference"] = paired["baseline"] - paired["reference"]
    return paired[["cell", "content", "style", "difference"]]


def wilcoxon_statistics(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[tuple[str, str], pd.DataFrame]]:
    rows: list[dict[str, float | int | str]] = []
    paired_frames: dict[tuple[str, str], pd.DataFrame] = {}
    for baseline in BASELINES:
        for metric in METRICS:
            paired = paired_differences(frame, baseline, metric)
            paired_frames[(baseline, metric)] = paired
            differences = paired["difference"].to_numpy(dtype=float)
            nonzero = differences[differences != 0]
            if nonzero.size == 0:
                raise ValueError(f"{baseline}/{metric}: all paired differences are zero")
            result = wilcoxon(
                differences,
                zero_method="wilcox",
                correction=False,
                alternative="two-sided",
                method="asymptotic",
            )
            ranks = rankdata(np.abs(nonzero), method="average")
            w_plus = float(ranks[nonzero > 0].sum())
            w_minus = float(ranks[nonzero < 0].sum())
            rank_biserial = (w_plus - w_minus) / (w_plus + w_minus)
            raw_p = float(result.pvalue)
            rows.append(
                {
                    "baseline": baseline,
                    "metric": metric,
                    "n": len(differences),
                    "n_nonzero": int(nonzero.size),
                    "mean_paired_difference": float(differences.mean()),
                    "median_paired_difference": float(np.median(differences)),
                    "w_statistic": float(result.statistic),
                    "w_plus": w_plus,
                    "w_minus": w_minus,
                    "z_statistic": float(result.zstatistic),
                    "raw_p": raw_p,
                    "bonferroni_adjusted_p": min(1.0, BONFERRONI_FAMILY * raw_p),
                    "rank_biserial_r": float(rank_biserial),
                    "difference_direction": "positive = CC-StyTr better",
                    "zero_method": "wilcox",
                    "p_method": "asymptotic",
                }
            )
    return pd.DataFrame(rows), paired_frames


def mixed_effects(paired_frames: dict[tuple[str, str], pd.DataFrame]) -> pd.DataFrame:
    import statsmodels
    import statsmodels.api as sm

    rows: list[dict[str, float | int | str | bool]] = []
    for baseline in BASELINES:
        for metric in METRICS:
            data = paired_frames[(baseline, metric)].copy()
            data["all_group"] = 1
            variance_components = {
                "content": "0 + C(content)",
                "style": "0 + C(style)",
            }
            accepted_result = None
            accepted_model = None
            accepted_optimizer = ""
            warning_messages: list[str] = []
            for optimizer in ("lbfgs", "bfgs", "cg"):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    model = sm.MixedLM.from_formula(
                        "difference ~ 1",
                        groups="all_group",
                        vc_formula=variance_components,
                        data=data,
                        use_sparse=True,
                    )
                    result = model.fit(
                        reml=True,
                        method=optimizer,
                        maxiter=2000,
                        disp=False,
                    )
                warning_messages.extend(str(item.message) for item in caught)
                if bool(result.converged):
                    accepted_result = result
                    accepted_model = model
                    accepted_optimizer = optimizer
                    break
            if accepted_result is None or accepted_model is None:
                raise RuntimeError(
                    f"mixed model did not converge for {baseline}/{metric}: "
                    + " | ".join(dict.fromkeys(warning_messages))
                )

            beta = float(accepted_result.fe_params["Intercept"])
            se = float(accepted_result.bse_fe["Intercept"])
            z_statistic = beta / se
            variance_by_name = {
                name: float(value)
                for name, value in zip(accepted_model.exog_vc.names, accepted_result.vcomp)
            }
            rows.append(
                {
                    "baseline": baseline,
                    "metric": metric,
                    "n": len(data),
                    "beta_estimate": beta,
                    "standard_error": se,
                    "ci95_low": beta - 1.96 * se,
                    "ci95_high": beta + 1.96 * se,
                    "z_statistic": z_statistic,
                    "raw_p": float(2 * norm.sf(abs(z_statistic))),
                    "content_random_effect_variance": variance_by_name["content"],
                    "style_random_effect_variance": variance_by_name["style"],
                    "residual_variance": float(accepted_result.scale),
                    "reml_log_likelihood": float(accepted_result.llf),
                    "converged": bool(accepted_result.converged),
                    "optimizer": accepted_optimizer,
                    "warnings": " | ".join(dict.fromkeys(warning_messages)),
                    "difference_direction": "positive = CC-StyTr better",
                    "statsmodels_version": statsmodels.__version__,
                }
            )
    return pd.DataFrame(rows)


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


def crossed_bootstrap(
    paired_frames: dict[tuple[str, str], pd.DataFrame],
    seed: int = BOOTSTRAP_SEED,
    n_bootstrap: int = N_BOOTSTRAP,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    content_counts = _bootstrap_counts(rng, n_bootstrap, EXPECTED_CONTENTS)
    style_counts = _bootstrap_counts(rng, n_bootstrap, EXPECTED_STYLES)
    rows: list[dict[str, float | int | str]] = []
    for baseline in BASELINES:
        for metric in METRICS:
            paired = paired_frames[(baseline, metric)]
            matrix_frame = (
                paired.pivot(index="content", columns="style", values="difference")
                .sort_index()
                .sort_index(axis=1)
            )
            if (
                matrix_frame.shape != (EXPECTED_CONTENTS, EXPECTED_STYLES)
                or matrix_frame.isna().any().any()
            ):
                raise ValueError(
                    f"{baseline}/{metric}: incomplete crossed matrix {matrix_frame.shape}"
                )
            matrix = matrix_frame.to_numpy(dtype=float)
            replicates = np.einsum(
                "bi,ij,bj->b",
                content_counts,
                matrix,
                style_counts,
                optimize=True,
            ) / EXPECTED_CELLS
            rows.append(
                {
                    "baseline": baseline,
                    "metric": metric,
                    "mean_paired_difference": float(matrix.mean()),
                    "bootstrap_mean": float(replicates.mean()),
                    "bootstrap_ci95_low": float(np.quantile(replicates, 0.025)),
                    "bootstrap_ci95_high": float(np.quantile(replicates, 0.975)),
                    "bootstrap_seed": seed,
                    "n_bootstrap": n_bootstrap,
                    "resampling_design": "two-factor crossed content/style bootstrap",
                    "difference_direction": "positive = CC-StyTr better",
                }
            )
    return pd.DataFrame(rows)


def write_outputs(
    source: Path,
    output_dir: Path,
    descriptive: pd.DataFrame,
    pairwise: pd.DataFrame,
    mixed: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> None:
    import statsmodels

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "descriptive_statistics.csv": descriptive,
        "wilcoxon_pairwise.csv": pairwise,
        "mixed_effects.csv": mixed,
        "crossed_bootstrap.csv": bootstrap,
    }
    for name, table in outputs.items():
        temporary = output_dir / f".{name}.tmp"
        table.to_csv(temporary, index=False, float_format="%.17g")
        temporary.replace(output_dir / name)

    deception_path = output_dir / "deception_pairwise.csv"
    deception_hits_path = output_dir / "deception_hits_per_pair.csv"
    deception_summary = {
        "mcnemar_b_c_present": False,
        "statistic_present": False,
        "adjusted_p_present": False,
        "paired_proportion_difference_present": False,
        "crossed_bootstrap_present": False,
        "all_adjusted_p_below_0_05": False,
        "crossed_ci_exclude_zero_count": 0,
        "pairwise_comparisons": 0,
        "per_pair_hits_rows": 0,
    }
    if deception_path.exists():
        deception = pd.read_csv(deception_path)
        required_deception = {
            "baseline",
            "b_ref_hit_baseline_miss",
            "c_ref_miss_baseline_hit",
            "mcnemar_statistic",
            "raw_p",
            "bonferroni_adjusted_p",
            "paired_proportion_difference",
            "crossed_bootstrap_ci95_low",
            "crossed_bootstrap_ci95_high",
        }
        if len(deception) != len(BASELINES) or not required_deception.issubset(deception):
            raise ValueError("deception_pairwise.csv failed its schema or row-count audit")
        hits_rows = 0
        if deception_hits_path.exists():
            hits_rows = len(pd.read_csv(deception_hits_path))
            if hits_rows != len(METHODS) * EXPECTED_CELLS:
                raise ValueError(f"expected 4000 deception hit rows, found {hits_rows}")
        excludes_zero = (
            (deception["crossed_bootstrap_ci95_low"] > 0)
            | (deception["crossed_bootstrap_ci95_high"] < 0)
        )
        deception_summary = {
            "mcnemar_b_c_present": bool(
                deception[
                    ["b_ref_hit_baseline_miss", "c_ref_miss_baseline_hit"]
                ].notna().all().all()
            ),
            "statistic_present": bool(deception["mcnemar_statistic"].notna().all()),
            "adjusted_p_present": bool(
                deception["bonferroni_adjusted_p"].notna().all()
            ),
            "paired_proportion_difference_present": bool(
                deception["paired_proportion_difference"].notna().all()
            ),
            "crossed_bootstrap_present": bool(
                deception[
                    ["crossed_bootstrap_ci95_low", "crossed_bootstrap_ci95_high"]
                ].notna().all().all()
            ),
            "all_adjusted_p_below_0_05": bool(
                (deception["bonferroni_adjusted_p"] < 0.05).all()
            ),
            "crossed_ci_exclude_zero_count": int(excludes_zero.sum()),
            "pairwise_comparisons": len(deception),
            "per_pair_hits_rows": hits_rows,
        }

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {"path": str(source), "sha256": sha256_file(source)},
        "general_data": {
            "methods": len(METHODS),
            "contents": EXPECTED_CONTENTS,
            "styles": EXPECTED_STYLES,
            "cells_per_method": EXPECTED_CELLS,
            "total_observations": len(METHODS) * EXPECTED_CELLS,
        },
        "primary_metrics": list(METRICS),
        "pairwise_comparisons": len(pairwise),
        "wilcoxon": {
            "w_present": bool(pairwise["w_statistic"].notna().all()),
            "raw_p_present": bool(pairwise["raw_p"].notna().all()),
            "adjusted_p_present": bool(pairwise["bonferroni_adjusted_p"].notna().all()),
            "bonferroni_family": BONFERRONI_FAMILY,
            "all_adjusted_p_below_0_05": bool(
                (pairwise["bonferroni_adjusted_p"] < 0.05).all()
            ),
        },
        "effect_size": {
            "rank_biserial_present": bool(pairwise["rank_biserial_r"].notna().all())
        },
        "dependence_aware_analysis": {
            "content_random_effect": bool(
                mixed["content_random_effect_variance"].notna().all()
            ),
            "style_random_effect": bool(
                mixed["style_random_effect_variance"].notna().all()
            ),
            "all_mixed_models_converged": bool(mixed["converged"].all()),
            "crossed_bootstrap": len(bootstrap) == BONFERRONI_FAMILY,
            "all_crossed_ci_exclude_zero": bool(
                (
                    (bootstrap["bootstrap_ci95_low"] > 0)
                    | (bootstrap["bootstrap_ci95_high"] < 0)
                ).all()
            ),
            "bootstrap_seed": BOOTSTRAP_SEED,
            "n_bootstrap": N_BOOTSTRAP,
        },
        "deception": deception_summary,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "statsmodels": statsmodels.__version__,
        },
        "manuscript": {
            "all_reported_numbers_traceable": None,
            "audit_pending": True,
        },
    }
    temporary_json = output_dir / ".statistics_summary.json.tmp"
    temporary_json.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary_json.replace(output_dir / "statistics_summary.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="full-precision per-pair CSV")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    args = parser.parse_args()
    if args.bootstrap_seed != BOOTSTRAP_SEED or args.n_bootstrap != N_BOOTSTRAP:
        raise ValueError(
            f"locked analysis requires seed={BOOTSTRAP_SEED}, n_bootstrap={N_BOOTSTRAP}"
        )

    source = Path(args.input)
    frame = load_and_validate(source)
    descriptive = descriptive_statistics(frame)
    pairwise, paired_frames = wilcoxon_statistics(frame)
    mixed = mixed_effects(paired_frames)
    if not bool(mixed["converged"].all()):
        raise RuntimeError("at least one mixed-effects model did not converge")
    bootstrap = crossed_bootstrap(paired_frames, args.bootstrap_seed, args.n_bootstrap)
    write_outputs(source, Path(args.output_dir), descriptive, pairwise, mixed, bootstrap)
    print("validated observations:", len(frame))
    print("pairwise comparisons:", len(pairwise))
    print("all mixed models converged:", bool(mixed["converged"].all()))
    print("saved statistics to:", args.output_dir)


if __name__ == "__main__":
    main()
