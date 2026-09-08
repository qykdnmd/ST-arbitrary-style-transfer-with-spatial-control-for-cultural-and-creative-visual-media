"""OCR text-consistency metric for the featured cultural-creative set:
does the text/layout of the creative asset survive stylization?

Definition (per content x style pair):
  GT  = texts detected in the ORIGINAL content image (conf >= 0.5, len >= 2)
  HYP = texts detected in the stylized output
  score = mean over GT tokens of max_{h in HYP} char_similarity(gt, h)
  char_similarity = 1 - edit_distance / max(len)
Pairs whose content has no detectable text are skipped (n_gt == 0) and reported
separately as n_textless.

Engine: RapidOCR (PP-OCRv4, zh+en), ONNX Runtime CPU. Multiprocess workers
each hold their own engine instance.

Usage:
  python scripts/ocr_consistency.py --content_dir data/eval/poster_curated \
      --style_dir data/eval/featured/style_flat --out_dir results/featured \
      --results results/featured/ocr.csv --workers 6
"""

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # GBK console vs CJK OCR text

METHODS = ["ccstytr", "stytr2", "adain", "sanet", "cast"]

_engine = None  # per-process RapidOCR instance
_gt_cache = {}  # per-process GT cache: content path -> text list


def get_engine():
    global _engine
    if _engine is None:
        # cap ONNX Runtime threads per worker: with N process workers on a
        # 16-core box, default sessions (1 thread-pool per worker sized to the
        # whole machine) oversubscribe the CPU ~6x and throughput collapses.
        import rapidocr_onnxruntime.utils as _ru
        _orig_so = _ru.SessionOptions

        def _limited_so():
            so = _orig_so()
            so.intra_op_num_threads = 2
            so.inter_op_num_threads = 1
            return so

        _ru.SessionOptions = _limited_so
        from rapidocr_onnxruntime import RapidOCR
        _engine = RapidOCR()
        _ru.SessionOptions = _orig_so
    return _engine


def norm(s: str) -> str:
    return "".join(s.split()).lower()


def edit_sim(a: str, b: str) -> float:
    """1 - normalized Levenshtein (pure python, tokens are short)."""
    if a == b:
        return 1.0
    la, lb = len(a), len(b)
    if not la or not lb:
        return 0.0
    prev = list(range(lb + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * lb
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return 1.0 - prev[lb] / max(la, lb)


def ocr_texts(img_path: str):
    result, _ = get_engine()(img_path)
    out = []
    for line in result or []:
        # line = [box, text, conf]
        text, conf = norm(str(line[1])), float(line[2])
        if conf >= 0.5 and len(text) >= 2:
            out.append(text)
    return out


def pair_score(args):
    content_path, out_path = args
    if content_path not in _gt_cache:
        _gt_cache[content_path] = ocr_texts(content_path)
    gt = _gt_cache[content_path]
    if not gt:
        return None  # textless content
    hyp = ocr_texts(out_path)
    score = sum(max((edit_sim(g, h) for h in hyp), default=0.0) for g in gt) / len(gt)
    return score, len(gt)


def collect_jobs(content_dir: Path, style_dir: Path, out_dir: Path, methods=None):
    """{(method, content_stem, style_stem): (content_path, output_path)}"""
    contents = {p.stem: p for p in content_dir.iterdir() if p.suffix.lower() == ".png"}
    styles = {p.stem for p in style_dir.iterdir()}
    jobs = {}
    for m in (methods or METHODS):
        mdir = out_dir / m
        if not mdir.exists():
            continue
        for f in mdir.iterdir():
            stem = f.stem
            hit = next(((c, s) for c in contents for s in styles
                        if stem == f"{c}__{s}"), None)
            if hit and hit[0] in contents:
                jobs[(m, hit[0], hit[1])] = (str(contents[hit[0]]), str(f))
    return jobs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--content_dir", default="data/eval/poster_curated")
    ap.add_argument("--style_dir", default="data/eval/featured/style_flat")
    ap.add_argument("--out_dir", default="results/featured")
    ap.add_argument("--results", default="results/featured/ocr.csv")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--methods", default=None,
                    help="comma-separated method dirs under out_dir (default: built-in METHODS)")
    args = ap.parse_args()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()] if args.methods else None

    jobs = collect_jobs(Path(args.content_dir), Path(args.style_dir), Path(args.out_dir), methods)
    print(f"{len(jobs)} output images to OCR", flush=True)

    rows, textless = [], 0
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(pair_score, v): k for k, v in jobs.items()}
        for fut in as_completed(futs):
            m, c, s = futs[fut]
            r = fut.result()
            done += 1
            if done % 500 == 0:
                print(f"  {done}/{len(jobs)}", flush=True)
            if r is None:
                textless += 1
                continue
            rows.append({"method": m, "content": c, "style": s,
                         "ocr_sim": f"{r[0]:.4f}", "n_gt": r[1]})

    dst = Path(args.results)
    with open(dst, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["method", "content", "style", "ocr_sim", "n_gt"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {dst} ({len(rows)} scored pairs, {textless} textless-skipped)")

    # quick summary
    print("\nmethod        n   mean_ocr_sim")
    for m in (methods or METHODS):
        vals = [float(r["ocr_sim"]) for r in rows if r["method"] == m]
        if vals:
            print(f"{m:<10} {len(vals):>5}   {sum(vals)/len(vals):.4f}")


if __name__ == "__main__":
    main()
