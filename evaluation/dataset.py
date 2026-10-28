from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


def load_annotation(image_path: Path) -> Dict:
    ann: Dict = {}
    js = image_path.with_suffix(".json")
    if js.is_file():
        ann.update(json.loads(js.read_text(encoding="utf-8")))
    txt = image_path.with_suffix(".txt")
    if "text" not in ann and txt.is_file():
        ann["text"] = txt.read_text(encoding="utf-8")
    return ann


def iter_dataset(folder: str) -> Iterator[Tuple[Path, Dict]]:
    root = Path(folder)
    if not root.is_dir():
        raise SystemExit(f"Dataset folder not found: {folder}")
    for p in sorted(root.iterdir()):
        if p.suffix.lower() in IMAGE_SUFFIXES:
            yield p, load_annotation(p)


def write_csv(path: str, rows: List[Dict]) -> None:
    if not rows:
        return
    keys: List[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def mean(values: List[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None
