"""Aggregate results/leakage/leakage.csv into a drift table + curve plot.

Outputs:
  - results/leakage/leakage_summary.csv  (method x round mean LPIPS over pairs)
  - results/leakage/leakage_curve.png    (drift curves, if matplotlib present)
  - prints a markdown table for the paper
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # GBK console vs StyTr²

RES = Path("results/leakage")
ORDER = ["ccstytr", "stytr2", "adain", "sanet", "cast"]
NAMES = {"ccstytr": "CC-StyTr (Main)", "stytr2": "StyTr²", "adain": "AdaIN",
         "sanet": "SANet", "cast": "CAST"}


def main() -> None:
    rows = list(csv.DictReader(open(RES / "leakage.csv", encoding="utf-8")))
    order = [method for method in ORDER if any(r["method"] == method for r in rows)]
    acc = defaultdict(list)  # (method, round) -> [lpips]
    for r in rows:
        acc[(r["method"], int(r["round"]))].append(float(r["lpips"]))
    rounds = sorted({int(r["round"]) for r in rows})
    mean = {k: sum(v) / len(v) for k, v in acc.items()}

    with open(RES / "leakage_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method"] + [f"r{n}" for n in rounds] + ["delta_1_20"])
        for m in order:
            series = [mean[(m, n)] for n in rounds]
            w.writerow([m] + [f"{v:.4f}" for v in series]
                       + [f"{series[-1] - series[0]:.4f}"])

    print("| Method | " + " | ".join(f"r{n}" for n in rounds if n in (1, 5, 10, 15, 20))
          + " | drift r1→r20 |")
    print("|---|" * 7)
    for m in order:
        series = [mean[(m, n)] for n in rounds]
        picks = [series[n - 1] for n in (1, 5, 10, 15, 20)]
        print(f"| {NAMES[m]} | " + " | ".join(f"{v:.4f}" for v in picks)
              + f" | {series[-1] - series[0]:+.4f} |")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skip plot")
        return
    plt.figure(figsize=(7, 4.5))
    for m in order:
        series = [mean[(m, n)] for n in rounds]
        plt.plot(rounds, series, marker="o", ms=3, label=NAMES[m])
    plt.xlabel("re-stylization round")
    plt.ylabel("LPIPS(output, original content)")
    plt.title("20-round content-leakage drift (lower = better structure retention)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RES / "leakage_curve.png", dpi=160)
    print("wrote", RES / "leakage_curve.png")


if __name__ == "__main__":
    main()
