"""Recompute full-precision metrics for the locked main experiment.

The script evaluates the existing 5 x 800 output images. It never generates
images or trains a model. The final CSV is published only after strict checks
confirm a complete 40-content x 20-style cross for every method.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.evaluate import METRIC_KEYS, Evaluator  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
METHODS = ("ccstytr", "stytr2", "adain", "sanet", "cast")
OPTIONAL_METHODS = ("aesfa",)
EXPECTED_CONTENTS = 40
EXPECTED_STYLES = 20
FULL_FIELDS = ("method", "output", "cell", "content", "style", *METRIC_KEYS)


def _image_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"missing image directory: {directory}")
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )


def _unique_by_stem(paths: list[Path], label: str) -> dict[str, Path]:
    indexed: dict[str, Path] = {}
    for path in paths:
        if path.stem in indexed:
            raise ValueError(
                f"duplicate {label} stem {path.stem!r}: {indexed[path.stem]} and {path}"
            )
        indexed[path.stem] = path
    return indexed


def collect_pairs(
    results: Path,
    content_dir: Path,
    style_dir: Path,
    methods: tuple[str, ...] = METHODS,
    expected_contents: int = EXPECTED_CONTENTS,
    expected_styles: int = EXPECTED_STYLES,
) -> dict[str, list[tuple[Path, Path, Path]]]:
    """Return validated output/content/style triples for every locked method."""
    content_by_stem = _unique_by_stem(_image_files(content_dir), "content")
    style_by_stem = _unique_by_stem(_image_files(style_dir), "style")
    if len(content_by_stem) != expected_contents:
        raise ValueError(f"expected {expected_contents} contents, found {len(content_by_stem)}")
    if len(style_by_stem) != expected_styles:
        raise ValueError(f"expected {expected_styles} styles, found {len(style_by_stem)}")

    expected_cells = {
        f"{content}__{style}" for content in content_by_stem for style in style_by_stem
    }
    expected_per_method = expected_contents * expected_styles
    collected: dict[str, list[tuple[Path, Path, Path]]] = {}

    for method in methods:
        outputs = _image_files(results / method)
        output_by_cell = _unique_by_stem(outputs, f"{method} output")
        actual_cells = set(output_by_cell)
        missing = sorted(expected_cells - actual_cells)
        unexpected = sorted(actual_cells - expected_cells)
        if len(outputs) != expected_per_method or missing or unexpected:
            raise ValueError(
                f"{method}: expected {expected_per_method} cells, found {len(outputs)}; "
                f"missing={missing[:5]}, unexpected={unexpected[:5]}"
            )

        triples: list[tuple[Path, Path, Path]] = []
        for cell in sorted(expected_cells):
            content_stem, separator, style_stem = cell.partition("__")
            if not separator:
                raise ValueError(f"invalid cell id: {cell}")
            triples.append(
                (output_by_cell[cell], content_by_stem[content_stem], style_by_stem[style_stem])
            )
        collected[method] = triples

    if set(collected) != set(methods):
        raise ValueError(f"method identity failure: {sorted(collected)}")
    return collected


def _load_partial(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != FULL_FIELDS:
            raise ValueError(f"partial CSV has unexpected columns: {reader.fieldnames}")
        rows = list(reader)
    keyed = {(row["method"], row["cell"]): row for row in rows}
    if len(keyed) != len(rows):
        raise ValueError("partial CSV contains duplicate method-cell rows")
    return keyed


def _validate_full_csv(path: Path, methods: tuple[str, ...], cells_per_method: int) -> None:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(methods) * cells_per_method:
        raise ValueError(
            f"full CSV has {len(rows)} rows; expected {len(methods) * cells_per_method}"
        )

    keyed = {(row["method"], row["cell"]) for row in rows}
    if len(keyed) != len(rows):
        raise ValueError("full CSV contains duplicate method-cell rows")
    reference_cells: set[str] | None = None
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        cells = {row["cell"] for row in method_rows}
        if len(cells) != cells_per_method:
            raise ValueError(f"{method}: found {len(cells)} unique cells")
        if reference_cells is None:
            reference_cells = cells
        elif cells != reference_cells:
            raise ValueError(f"{method}: cell ids differ from the reference method")
        for row in method_rows:
            expected_cell = f"{row['content']}__{row['style']}"
            if row["cell"] != expected_cell:
                raise ValueError(f"cell decomposition mismatch: {row['cell']} != {expected_cell}")
            for metric in METRIC_KEYS:
                if not math.isfinite(float(row[metric])):
                    raise ValueError(f"non-finite {metric} for {method}/{row['cell']}")


def _write_means(full_csv: Path, destination: Path, methods: tuple[str, ...]) -> None:
    with full_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "n", *[f"{metric}_mean" for metric in METRIC_KEYS]])
        for method in methods:
            method_rows = [row for row in rows if row["method"] == method]
            means = [
                float(np.mean([float(row[metric]) for row in method_rows]))
                for metric in METRIC_KEYS
            ]
            writer.writerow([method, len(method_rows), *means])
            print(method, dict(zip(METRIC_KEYS, means)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--content_dir", required=True)
    parser.add_argument("--style_dir", required=True)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--expected_contents", type=int, default=EXPECTED_CONTENTS)
    parser.add_argument("--expected_styles", type=int, default=EXPECTED_STYLES)
    parser.add_argument(
        "--full_csv",
        default=None,
        help="default: <results_dir>/statistics/metrics_per_pair_full.csv",
    )
    parser.add_argument("--resume", action="store_true", help="resume a validated .partial file")
    parser.add_argument("--force", action="store_true", help="replace prior full/partial outputs")
    args = parser.parse_args()

    methods = tuple(method.strip() for method in args.methods.split(",") if method.strip())
    unknown = sorted(set(methods) - set(METHODS) - set(OPTIONAL_METHODS))
    if not methods or unknown:
        raise ValueError(f"unknown or empty method selection: methods={methods}, unknown={unknown}")

    results = Path(args.results_dir)
    pairs = collect_pairs(
        results,
        Path(args.content_dir),
        Path(args.style_dir),
        methods=methods,
        expected_contents=args.expected_contents,
        expected_styles=args.expected_styles,
    )
    print("validated cells:", {method: len(triples) for method, triples in pairs.items()})

    full_csv = (
        Path(args.full_csv)
        if args.full_csv
        else results / "statistics" / "metrics_per_pair_full.csv"
    )
    full_csv.parent.mkdir(parents=True, exist_ok=True)
    partial_csv = full_csv.with_suffix(full_csv.suffix + ".partial")
    if args.force:
        full_csv.unlink(missing_ok=True)
        partial_csv.unlink(missing_ok=True)
    if full_csv.exists():
        raise FileExistsError(f"refusing to overwrite {full_csv}; use --force")
    if partial_csv.exists() and not args.resume:
        raise FileExistsError(
            f"partial evaluation exists at {partial_csv}; use --resume or --force"
        )

    completed = _load_partial(partial_csv) if args.resume else {}
    evaluator = Evaluator(size=args.size)
    write_header = not partial_csv.exists()
    with partial_csv.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FULL_FIELDS)
        if write_header:
            writer.writeheader()
        for method, triples in pairs.items():
            for index, (output, content, style) in enumerate(triples, start=1):
                cell = output.stem
                if (method, cell) in completed:
                    continue
                metrics = evaluator(str(output), str(content), str(style))
                writer.writerow(
                    {
                        "method": method,
                        "output": output.name,
                        "cell": cell,
                        "content": content.stem,
                        "style": style.stem,
                        **{key: float(value) for key, value in metrics.items()},
                    }
                )
                handle.flush()
                if index % 100 == 0 or index == len(triples):
                    print(f"{method}: {index}/{len(triples)}", flush=True)

    cells_per_method = args.expected_contents * args.expected_styles
    _validate_full_csv(partial_csv, methods, cells_per_method)
    partial_csv.replace(full_csv)
    _write_means(full_csv, full_csv.parent / "evaluation_means_full.csv", methods)
    print("saved:", full_csv)


if __name__ == "__main__":
    main()
