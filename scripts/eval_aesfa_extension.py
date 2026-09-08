"""Recompute the five unified metrics for CC-StyTr and AesFA only.

This intentionally writes a separate extension dataset so the archived
five-method evidence and its primary 20-test family remain unchanged.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.eval_main import FULL_FIELDS, _validate_full_csv, collect_pairs
from scripts.evaluate import Evaluator

METHODS = ("ccstytr", "aesfa")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--content_dir", required=True)
    parser.add_argument("--style_dir", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--expected_contents", type=int, required=True)
    parser.add_argument("--expected_styles", type=int, required=True)
    parser.add_argument("--size", type=int, default=512)
    args = parser.parse_args()

    pairs = collect_pairs(
        Path(args.results_dir),
        Path(args.content_dir),
        Path(args.style_dir),
        methods=METHODS,
        expected_contents=args.expected_contents,
        expected_styles=args.expected_styles,
    )
    destination = Path(args.output_csv)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    completed = set()
    if partial.exists():
        with partial.open(newline="", encoding="utf-8") as handle:
            old_rows = list(csv.DictReader(handle))
        completed = {(row["method"], row["cell"]) for row in old_rows}
    evaluator = Evaluator(size=args.size)
    with partial.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FULL_FIELDS)
        if partial.stat().st_size == 0:
            writer.writeheader()
        for method, triples in pairs.items():
            for index, (output, content, style) in enumerate(triples, start=1):
                if (method, output.stem) not in completed:
                    metrics = evaluator(str(output), str(content), str(style))
                    writer.writerow(
                        {
                            "method": method,
                            "output": output.name,
                            "cell": output.stem,
                            "content": content.stem,
                            "style": style.stem,
                            **metrics,
                        }
                    )
                    handle.flush()
                if index % 100 == 0 or index == len(triples):
                    print(f"{method}: {index}/{len(triples)}", flush=True)
    _validate_full_csv(
        partial, METHODS, args.expected_contents * args.expected_styles
    )
    partial.replace(destination)
    print(f"saved -> {destination}")


if __name__ == "__main__":
    main()
