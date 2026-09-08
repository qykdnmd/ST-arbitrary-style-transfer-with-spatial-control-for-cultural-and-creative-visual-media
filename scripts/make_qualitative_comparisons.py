"""Build submission-ready qualitative comparisons with source-only cases.

The generic figure contains three fixed content/style pairs. The poster figure
contains two English-bearing source posters, shown as demonstrations 1 and 3;
each full-image row is followed by one text crop and one logo crop. A crop uses
the identical official PKU box, padding, and magnification for the content and
every method. Selection criteria use source images and annotations and are recorded in
``paper/evidence`` together with the selected source-list indices and crop boxes.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("ccstytr", "aesfa", "stytr2", "cast", "sanet", "adain")
GENERIC_LABELS = ("Content", "Style", "CC-StyTr (Main)", "AesFA", "StyTr²", "CAST", "SANet", "AdaIN")
POSTER_LABELS = ("Style", "Content", "CC-StyTr (Main)", "AesFA", "StyTr²", "CAST", "SANet", "AdaIN")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
GENERIC_INDICES = ((5, 2), (12, 5), (27, 7))
POSTER_CASES = (
    # case, content, style category, text-box index, logo-box index
    (1, "6372", "mural", 1, 0),
    (3, "8962", "western_painting", 0, 0),
)


def indexed(directory: Path) -> dict[str, Path]:
    files = {
        path.stem: path
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    }
    if not files:
        raise FileNotFoundError(f"no images in {directory}")
    return files


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = ["arialbd.ttf" if bold else "arial.ttf", "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def centered_text(image: Image.Image, text: str, text_font, fill="#202020") -> None:
    draw = ImageDraw.Draw(image)
    box = draw.textbbox((0, 0), text, font=text_font)
    draw.text(
        ((image.width - box[2]) / 2, (image.height - box[3]) / 2),
        text,
        font=text_font,
        fill=fill,
    )


def output_path(results_dir: Path, method: str, pair_id: str) -> Path:
    hits = [path for path in (results_dir / method).glob(f"{pair_id}.*") if path.suffix.lower() in IMAGE_EXTS]
    if len(hits) != 1:
        raise FileNotFoundError(f"expected one {method} output for {pair_id}, found {len(hits)}")
    return hits[0]


def full_cell(path: Path, cell: int) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB").resize((cell, cell), Image.Resampling.LANCZOS)


def padded_box(box: list[float], width: int, height: int, padding: float = 0.15) -> tuple[float, ...]:
    x1, y1, x2, y2 = map(float, box)
    pad_x = (x2 - x1) * padding
    pad_y = (y2 - y1) * padding
    return max(0, x1 - pad_x), max(0, y1 - pad_y), min(width, x2 + pad_x), min(height, y2 + pad_y)


def output_box(
    box: tuple[float, ...],
    source_size: tuple[int, int],
    output_size: tuple[int, int],
    resize_mode: str,
) -> tuple[float, ...]:
    source_width, source_height = source_size
    output_width, output_height = output_size
    if resize_mode == "stretch":
        scale_x = output_width / source_width
        scale_y = output_height / source_height
        return (
            box[0] * scale_x,
            box[1] * scale_y,
            box[2] * scale_x,
            box[3] * scale_y,
        )
    if resize_mode != "short-side-center-crop":
        raise ValueError(f"unknown resize mode: {resize_mode}")
    if output_width != output_height:
        raise ValueError("short-side center-crop mapping expects a square output")

    # Match torchvision Resize(size=int) followed by CenterCrop(size), used by
    # AesFA in infer_paired.py. torchvision truncates the resized long side and
    # rounds the centered crop offset.
    side = output_width
    if source_width <= source_height:
        resized_width = side
        resized_height = int(side * source_height / source_width)
    else:
        resized_height = side
        resized_width = int(side * source_width / source_height)
    scale_x = resized_width / source_width
    scale_y = resized_height / source_height
    offset_x = round((resized_width - side) / 2.0)
    offset_y = round((resized_height - side) / 2.0)
    return (
        box[0] * scale_x - offset_x,
        box[1] * scale_y - offset_y,
        box[2] * scale_x - offset_x,
        box[3] * scale_y - offset_y,
    )


def crop_cell(
    path: Path,
    box: tuple[float, ...],
    source_size: tuple[int, int],
    cell: int,
    resize_mode: str = "stretch",
) -> Image.Image:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        scaled = output_box(box, source_size, rgb.size, resize_mode)
        if scaled[0] < 0 or scaled[1] < 0 or scaled[2] > rgb.width or scaled[3] > rgb.height:
            raise RuntimeError(
                f"crop {box} maps outside {path.name} under {resize_mode}: {scaled}"
            )
        return rgb.crop(scaled).resize((cell, cell), Image.Resampling.LANCZOS)


def compose(
    rows: list[list[Image.Image]],
    labels: tuple[str, ...],
    output: Path,
    cell: int,
    header: int,
) -> None:
    canvas = Image.new("RGB", (cell * len(labels), header + cell * len(rows)), "white")
    heading_font = font(max(18, cell // 12), bold=True)
    for column, label in enumerate(labels):
        header_cell = Image.new("RGB", (cell, header), "white")
        centered_text(header_cell, label, heading_font)
        canvas.paste(header_cell, (column * cell, 0))
    for row_index, row in enumerate(rows):
        for column, image in enumerate(row):
            canvas.paste(image, (column * cell, header + row_index * cell))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, dpi=(450, 450))
    print(f"saved {output} {canvas.size}")


def generic_figure(args) -> None:
    contents = indexed(Path(args.generic_content))
    styles = indexed(Path(args.generic_style))
    content_ids = sorted(contents)
    style_ids = sorted(styles)
    pairs = [(content_ids[c], style_ids[s]) for c, s in GENERIC_INDICES]
    evidence = ROOT / "paper/evidence/qualitative_general.csv"
    with evidence.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row", "content_id", "style_id", "selection_basis"])
        for index, (content_id, style_id) in enumerate(pairs, start=1):
            writer.writerow([index, content_id, style_id, "fixed source-list index"])

    rows = []
    for content_id, style_id in pairs:
        pair_id = f"{content_id}__{style_id}"
        paths = [contents[content_id], styles[style_id]] + [
            output_path(Path(args.generic_results), method, pair_id) for method in METHODS
        ]
        rows.append([full_cell(path, args.cell) for path in paths])
    compose(rows, GENERIC_LABELS, Path(args.generic_output), args.cell, args.header)


def poster_selection(manifest_path: Path) -> list[tuple[int, dict, str, int, int]]:
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        by_id = {Path(row["file"]).stem: row for row in csv.DictReader(handle)}
    selected = []
    for case_number, content_id, style_category, text_index, logo_index in POSTER_CASES:
        poster = by_id.get(content_id)
        if poster is None:
            raise RuntimeError(f"poster {content_id} is missing from {manifest_path}")
        text_boxes = json.loads(poster.get("text_boxes") or "[]")
        logo_boxes = json.loads(poster.get("logo_boxes") or "[]")
        if not text_boxes or not logo_boxes:
            raise RuntimeError(f"poster {content_id} needs both text and logo boxes")
        if text_index >= len(text_boxes) or logo_index >= len(logo_boxes):
            raise RuntimeError(
                f"poster {content_id} crop indices are invalid: "
                f"text={text_index}/{len(text_boxes)}, logo={logo_index}/{len(logo_boxes)}"
            )
        selected.append((case_number, poster, style_category, text_index, logo_index))
    return selected


def poster_figure(args) -> None:
    content_dir = Path(args.poster_content)
    contents = indexed(content_dir)
    styles = indexed(Path(args.poster_style))
    selected = poster_selection(Path(args.poster_manifest))
    displayed = []
    for case_number, poster, style_category, text_index, logo_index in selected:
        hits = sorted(
            style_id for style_id in styles if style_id.startswith(f"{style_category}__")
        )
        if not hits:
            raise RuntimeError(f"no style for category {style_category}")
        displayed.append((case_number, poster, hits[0], text_index, logo_index))

    evidence = ROOT / "paper/evidence/qualitative_poster.csv"
    with evidence.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["case", "content_id", "style_id", "text_box", "logo_box", "selection_basis"]
        )
        for case_number, poster, style_id, text_index, logo_index in displayed:
            text_box = json.loads(poster["text_boxes"])[text_index]
            logo_box = json.loads(poster["logo_boxes"])[logo_index]
            writer.writerow(
                [
                    case_number,
                    Path(poster["file"]).stem,
                    style_id,
                    json.dumps(text_box),
                    json.dumps(logo_box),
                    "fixed from source-only OCR/visual audit: English-dominant source and in-frame official crop boxes",
                ]
            )

    rows = []
    for _, poster, style_id, text_index, logo_index in displayed:
        content_id = Path(poster["file"]).stem
        pair_id = f"{content_id}__{style_id}"
        paths = [styles[style_id], contents[content_id]] + [
            output_path(Path(args.poster_results), method, pair_id) for method in METHODS
        ]
        rows.append([full_cell(path, args.cell) for path in paths])
        source_size = (int(poster["width"]), int(poster["height"]))
        for field, box_index in (("text_boxes", text_index), ("logo_boxes", logo_index)):
            selected_box = json.loads(poster[field])[box_index]
            box = padded_box(selected_box, *source_size)
            crop_row = [Image.new("RGB", (args.cell, args.cell), "white")]
            crop_row.append(crop_cell(paths[1], box, source_size, args.cell))
            for method, path in zip(METHODS, paths[2:]):
                resize_mode = "short-side-center-crop" if method == "aesfa" else "stretch"
                crop_row.append(
                    crop_cell(path, box, source_size, args.cell, resize_mode=resize_mode)
                )
            rows.append(crop_row)
    compose(rows, POSTER_LABELS, Path(args.poster_output), args.cell, args.header)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generic_content", default="data/eval/general/content")
    parser.add_argument("--generic_style", default="data/eval/general/style")
    parser.add_argument("--generic_results", default="results/main")
    parser.add_argument("--poster_content", default="data/eval/poster_curated")
    parser.add_argument(
        "--poster_manifest",
        default="data/eval/poster_curated/manifest_annotations.csv",
        help="Enriched manifest produced by export_featured_annotations.py",
    )
    parser.add_argument("--poster_style", default="data/eval/featured/style_flat")
    parser.add_argument("--poster_results", default="results/featured")
    parser.add_argument("--generic_output", default="paper/figures/qualitative_general.png")
    parser.add_argument("--poster_output", default="paper/figures/qualitative_poster_crops.png")
    parser.add_argument("--cell", type=int, default=240)
    parser.add_argument("--header", type=int, default=58)
    args = parser.parse_args()
    generic_figure(args)
    poster_figure(args)


if __name__ == "__main__":
    main()
