from pathlib import Path

import pandas as pd

from scripts.eval_main import collect_pairs
from scripts.statistical_analysis import (
    BASELINES,
    METRICS,
    crossed_bootstrap,
    descriptive_statistics,
    load_and_validate,
    wilcoxon_statistics,
)


def _synthetic_frame() -> pd.DataFrame:
    rows = []
    method_offsets = {
        "ccstytr": 0.0,
        "stytr2": -0.1,
        "adain": -0.2,
        "sanet": -0.3,
        "cast": -0.4,
    }
    for method, offset in method_offsets.items():
        for content_index in range(40):
            content = f"content_{content_index:02d}"
            for style_index in range(20):
                style = f"style_{style_index:02d}"
                cell = f"{content}__{style}"
                nuisance = content_index * 1e-4 + style_index * 1e-5
                rows.append(
                    {
                        "method": method,
                        "output": f"{cell}.png",
                        "cell": cell,
                        "content": content,
                        "style": style,
                        "ssim": 0.7 + offset + nuisance,
                        "lpips": 0.3 - offset + nuisance,
                        "e_ssim": 0.65 + offset + nuisance,
                        "l_c": 1.0 - offset + nuisance,
                        "l_s": 0.8 - offset + nuisance,
                    }
                )
    return pd.DataFrame(rows)


def test_full_precision_statistics_pipeline(tmp_path: Path) -> None:
    source = tmp_path / "metrics.csv"
    _synthetic_frame().to_csv(source, index=False)
    frame = load_and_validate(source)
    descriptive = descriptive_statistics(frame)
    pairwise, paired = wilcoxon_statistics(frame)
    bootstrap = crossed_bootstrap(paired, seed=20260901, n_bootstrap=100)

    assert len(frame) == 4000
    assert len(descriptive) == 25
    assert len(pairwise) == len(BASELINES) * len(METRICS) == 20
    assert (pairwise["mean_paired_difference"] > 0).all()
    assert (pairwise["rank_biserial_r"] > 0).all()
    assert (bootstrap["bootstrap_ci95_low"] > 0).all()


def test_collect_pairs_matches_cells_by_stem_across_extensions(tmp_path: Path) -> None:
    content_dir = tmp_path / "content"
    style_dir = tmp_path / "style"
    results_dir = tmp_path / "results"
    content_dir.mkdir()
    style_dir.mkdir()
    for stem in ("c0", "c1"):
        (content_dir / f"{stem}.jpg").write_bytes(b"")
    for stem in ("movement__s0", "movement__s1"):
        (style_dir / f"{stem}.png").write_bytes(b"")
    for method, extension in (("first", ".png"), ("second", ".jpg")):
        method_dir = results_dir / method
        method_dir.mkdir(parents=True)
        for content in ("c0", "c1"):
            for style in ("movement__s0", "movement__s1"):
                (method_dir / f"{content}__{style}{extension}").write_bytes(b"")

    pairs = collect_pairs(
        results_dir,
        content_dir,
        style_dir,
        methods=("first", "second"),
        expected_contents=2,
        expected_styles=2,
    )
    assert {method: len(triples) for method, triples in pairs.items()} == {
        "first": 4,
        "second": 4,
    }
    assert pairs["first"][0][0].stem == pairs["second"][0][0].stem
