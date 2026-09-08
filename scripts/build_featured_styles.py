"""Build the featured heritage style set (experiment protocol): 30 styles =
10 Dunhuang murals + 10 Chinese landscape paintings + 10 Western paintings.

Selection was made by visual audit of contact sheets:
  - mural:    sanity_check/mural_audit/sheet_*.png  (ranks into manifest_selected.csv)
  - landscape/western: sanity_check/audit_landscape|audit_western/sheet_*.png
    (matched by file-stem prefix)

Output: data/eval/featured/style/{mural,chinese_landscape,western_painting}/
"""

import csv
import shutil
from pathlib import Path

STYLES = Path("data/eval/styles")
OUT = Path("data/eval/featured/style")

# mural: ranks within manifest_selected.csv (== ranks of manifest_ranked.csv)
MURAL_RANKS = [10, 12, 24, 45, 53, 74, 92, 120, 125, 159]

LANDSCAPE_PREFIXES = [
    "130942",   # bold dark ink mountains
    "132834",   # blue-green landscape with inscription
    "132836",   # ink mountains with red seals
    "132840",   # cliff temple, blue-green tint
    "132930",   # misty Mi-style ink
    "137205",   # dramatic dark peaks with waterfall
    "142574",   # dark misty peaks
    "142578",   # hemp-fiber texture strokes
    "144829",   # blue-green with architecture
    "148342",   # willow and bridge, dark ink
]

WESTERN_PREFIXES = [
    "met_437769",  # double portrait, red dress
    "met_437967",  # dark portrait
    "met_439405",  # pastel lady, blue dress
    "met_359775",  # classical landscape with ruins
    "met_383787",  # blue building watercolor
    "met_406721",  # colorful bird print
    "met_787672",  # colorful crowd scene
    "met_788182",  # colored interior scene
    "met_739494",  # domestic interior, colored
    "met_649693",  # architecture view, colored
]


def copy_by_prefix(src_dir: Path, prefixes, dst_dir: Path) -> None:
    files = sorted(p for p in src_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    for pre in prefixes:
        hits = [p for p in files if p.stem.startswith(pre)]
        if len(hits) != 1:
            raise SystemExit(f"prefix {pre!r} matched {len(hits)} files in {src_dir}")
        shutil.copy(hits[0], dst_dir / hits[0].name)


def main() -> None:
    # mural via ranked-selected manifest
    rows = {int(r["rank"]): r["file"] for r in
            csv.DictReader(open(STYLES / "mural/manifest_selected.csv", encoding="utf-8"))}
    m_out = OUT / "mural"
    m_out.mkdir(parents=True, exist_ok=True)
    for r in MURAL_RANKS:
        shutil.copy(STYLES / "mural" / rows[r], m_out / rows[r])

    for sub, prefixes in (("chinese_landscape", LANDSCAPE_PREFIXES),
                          ("western_painting", WESTERN_PREFIXES)):
        d = OUT / sub
        d.mkdir(parents=True, exist_ok=True)
        copy_by_prefix(STYLES / sub, prefixes, d)

    for d in sorted(OUT.iterdir()):
        print(d.name, len(list(d.iterdir())))


if __name__ == "__main__":
    main()
