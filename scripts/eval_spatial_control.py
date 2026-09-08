"""Evaluate global-strength monotonicity and spatial locality of alpha control.

Inputs are the locked 60 x 10 protocol and outputs from
``run_spatial_control.py``. Global control is measured with LPIPS-to-content
and CSD target-style similarity across five alpha levels. Spatial control uses
the equal-mean protected/inverted maps to measure region-response direction,
normalized locality contrast, OCR preservation (when ``ocr.csv`` is present),
and boundary residuals relative to the uniform alpha=0.5 control.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import lpips
import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from eval_artfid_csd import bootstrap_mean_ci, load_csd, style_embedding

GLOBAL_CONDITIONS = (
    ("alpha_000", 0.0),
    ("alpha_025", 0.25),
    ("alpha_050", 0.5),
    ("alpha_075", 0.75),
    ("alpha_100", 1.0),
)
SPATIAL_CONDITIONS = ("protected", "inverted")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def indexed(directory: Path) -> dict[str, Path]:
    files = {
        path.stem: path
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    }
    if not files:
        raise FileNotFoundError(f"no images in {directory}")
    return files


class GlobalDataset(Dataset):
    def __init__(self, output_dir: Path, content_dir: Path, style_dir: Path, csd_transform):
        self.outputs = indexed(output_dir)
        self.contents = indexed(content_dir)
        self.styles = indexed(style_dir)
        self.ids = sorted(self.outputs)
        self.csd_transform = csd_transform
        self.lpips_transform = transforms.Compose(
            [transforms.Resize((512, 512)), transforms.ToTensor()]
        )

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int):
        pair_id = self.ids[index]
        content_id, style_id = pair_id.split("__", maxsplit=1)
        with Image.open(self.outputs[pair_id]) as image:
            output_image = image.convert("RGB")
            output_lpips = self.lpips_transform(output_image)
            output_csd = self.csd_transform(output_image)
        with Image.open(self.contents[content_id]) as image:
            content_lpips = self.lpips_transform(image.convert("RGB"))
        with Image.open(self.styles[style_id]) as image:
            style_csd = self.csd_transform(image.convert("RGB"))
        return pair_id, output_lpips, content_lpips, output_csd, style_csd


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB").resize((512, 512), Image.Resampling.LANCZOS), dtype=np.float32) / 255.0


def ring_mask(mask: np.ndarray, radius: int = 3) -> np.ndarray:
    tensor = torch.from_numpy(mask.astype(np.float32))[None, None]
    kernel = radius * 2 + 1
    dilated = functional.max_pool2d(tensor, kernel, stride=1, padding=radius)
    eroded = -functional.max_pool2d(-tensor, kernel, stride=1, padding=radius)
    return ((dilated - eroded)[0, 0].numpy() > 0.5)


def load_ocr(path: Path) -> dict[tuple[str, str], float]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            (row["method"], f"{row['content']}__{row['style']}"): float(row["ocr_sim"])
            for row in csv.DictReader(handle)
        }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir", default="data/eval/spatial_control")
    parser.add_argument("--results_dir", default="results/spatial_control")
    parser.add_argument("--csd_root", default="external/CSD")
    parser.add_argument("--csd_checkpoint", required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    results_dir = Path(args.results_dir)
    protocol = json.loads((eval_dir / "protocol.json").read_text(encoding="utf-8"))
    expected_pairs = protocol["n_posters"] * protocol["n_styles"]
    content_dir = eval_dir / "content"
    style_dir = eval_dir / "style"
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    csd_model, csd_transform = load_csd(Path(args.csd_root), Path(args.csd_checkpoint), device)
    lpips_model = lpips.LPIPS(net="vgg", verbose=False).to(device).eval()

    global_rows = []
    for condition, alpha in GLOBAL_CONDITIONS:
        dataset = GlobalDataset(results_dir / condition, content_dir, style_dir, csd_transform)
        if len(dataset) != expected_pairs:
            raise RuntimeError(f"{condition}: expected {expected_pairs} outputs, found {len(dataset)}")
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
        )
        with torch.inference_mode():
            for ids, outputs, contents, outputs_csd, styles_csd in loader:
                lpips_values = lpips_model(
                    outputs.to(device) * 2 - 1, contents.to(device) * 2 - 1
                ).flatten()
                output_embeddings = style_embedding(csd_model, outputs_csd.to(device))
                style_embeddings = style_embedding(csd_model, styles_csd.to(device))
                csd_values = (output_embeddings * style_embeddings).sum(dim=-1)
                for pair_id, lpips_value, csd_value in zip(
                    ids, lpips_values.cpu().tolist(), csd_values.cpu().tolist(), strict=True
                ):
                    content_id, style_id = pair_id.split("__", maxsplit=1)
                    global_rows.append(
                        {
                            "pair_id": pair_id,
                            "content_id": content_id,
                            "style_id": style_id,
                            "condition": condition,
                            "alpha": alpha,
                            "lpips_content": lpips_value,
                            "csd_style_similarity": csd_value,
                        }
                    )
        print(f"global metrics: {condition}", flush=True)

    by_pair: dict[str, list[dict]] = {}
    for row in global_rows:
        by_pair.setdefault(row["pair_id"], []).append(row)
    monotonic_rows = []
    for pair_id, rows in sorted(by_pair.items()):
        rows.sort(key=lambda row: row["alpha"])
        alpha = np.asarray([row["alpha"] for row in rows])
        lpips_values = np.asarray([row["lpips_content"] for row in rows])
        csd_values = np.asarray([row["csd_style_similarity"] for row in rows])
        lpips_rho = float(spearmanr(alpha, lpips_values).statistic)
        csd_rho = float(spearmanr(alpha, csd_values).statistic)
        monotonic_rows.append(
            {
                "pair_id": pair_id,
                "lpips_spearman": lpips_rho,
                "csd_spearman": csd_rho,
                "lpips_adjacent_violation_rate": float(np.mean(np.diff(lpips_values) < 0)),
                "csd_adjacent_violation_rate": float(np.mean(np.diff(csd_values) < 0)),
            }
        )

    ocr = load_ocr(results_dir / "ocr.csv")
    spatial_rows = []
    content_paths = indexed(content_dir)
    protected_maps = indexed(eval_dir / "alpha_protected")
    for pair_id in sorted(indexed(results_dir / "protected")):
        content_id, _ = pair_id.split("__", maxsplit=1)
        content = read_rgb(content_paths[content_id])
        protected = read_rgb(results_dir / "protected" / f"{pair_id}.png")
        inverted = read_rgb(results_dir / "inverted" / f"{pair_id}.png")
        uniform = read_rgb(results_dir / "alpha_050" / f"{pair_id}.png")
        with Image.open(protected_maps[content_id]) as image:
            alpha_map = np.asarray(image.convert("L").resize((512, 512)), dtype=np.float32) / 255.0
        mask = alpha_map < 0.5
        outside = ~mask
        protected_change = np.abs(protected - content).mean(axis=2)
        inverted_change = np.abs(inverted - content).mean(axis=2)
        inside_direction = float(inverted_change[mask].mean() - protected_change[mask].mean())
        outside_direction = float(protected_change[outside].mean() - inverted_change[outside].mean())
        inside_scale = float(inverted_change[mask].mean() + protected_change[mask].mean())
        outside_scale = float(protected_change[outside].mean() + inverted_change[outside].mean())
        locality = 0.5 * (
            inside_direction / max(inside_scale, 1e-8)
            + outside_direction / max(outside_scale, 1e-8)
        )
        boundary = ring_mask(mask)
        protected_boundary_residual = float(np.abs(protected - uniform).mean(axis=2)[boundary].mean())
        inverted_boundary_residual = float(np.abs(inverted - uniform).mean(axis=2)[boundary].mean())
        spatial_rows.append(
            {
                "pair_id": pair_id,
                "inside_direction": inside_direction,
                "outside_direction": outside_direction,
                "direction_success": int(inside_direction > 0 and outside_direction > 0),
                "locality_contrast": locality,
                "protected_boundary_residual": protected_boundary_residual,
                "inverted_boundary_residual": inverted_boundary_residual,
                "protected_ocr": ocr.get(("protected", pair_id), ""),
                "inverted_ocr": ocr.get(("inverted", pair_id), ""),
                "uniform_050_ocr": ocr.get(("alpha_050", pair_id), ""),
            }
        )

    write_csv(results_dir / "global_per_pair.csv", global_rows)
    write_csv(results_dir / "monotonicity_per_pair.csv", monotonic_rows)
    write_csv(results_dir / "spatial_locality_per_pair.csv", spatial_rows)

    summary_rows = []
    for metric in (
        "lpips_spearman",
        "csd_spearman",
        "lpips_adjacent_violation_rate",
        "csd_adjacent_violation_rate",
    ):
        values = np.asarray([row[metric] for row in monotonic_rows], dtype=np.float64)
        low, high = bootstrap_mean_ci(values, 20260902, 10_000)
        summary_rows.append({"metric": metric, "mean": values.mean(), "ci95_low": low, "ci95_high": high})
    for metric in ("direction_success", "locality_contrast"):
        values = np.asarray([row[metric] for row in spatial_rows], dtype=np.float64)
        low, high = bootstrap_mean_ci(values, 20260903, 10_000)
        summary_rows.append({"metric": metric, "mean": values.mean(), "ci95_low": low, "ci95_high": high})
    write_csv(results_dir / "summary.csv", summary_rows)
    print(f"saved controllability metrics -> {results_dir}")


if __name__ == "__main__":
    main()
