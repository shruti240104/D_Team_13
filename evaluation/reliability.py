from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

from backend.ocr import configure_tesseract
from backend.pipeline import PipelineOptions, run_pipeline
from evaluation.dataset import write_csv


def run(folder: str, run_ocr: bool = True):
    configure_tesseract()
    root = Path(folder)
    if not root.is_dir():
        raise SystemExit(f"Folder not found: {folder}")
    rows, outcomes = [], Counter()
    for path in sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() != ".json" and p.suffix.lower() != ".txt"):
        start = time.perf_counter()
        try:
            res = run_pipeline(path.read_bytes(), path.name, PipelineOptions(run_ocr=run_ocr))
            outcome = "success" if res.ok else f"{res.error.stage}/{res.error.code}"
            ocr_status = res.ocr.status if res.ocr else ""
        except Exception as exc:  # would be a bug: the pipeline must never raise
            outcome, ocr_status = f"CRASH/{type(exc).__name__}", ""
        outcomes[outcome] += 1
        rows.append({"file": path.name, "outcome": outcome, "ocr_status": ocr_status,
                     "seconds": round(time.perf_counter() - start, 3)})
    return rows, outcomes


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--csv")
    ap.add_argument("--no-ocr", action="store_true")
    a = ap.parse_args()
    rows, outcomes = run(a.folder, not a.no_ocr)
    for r in rows:
        print(r)
    total = len(rows)
    print(f"RUNS: {total}   OUTCOMES: {dict(outcomes)}")
    if rows:
        print(f"Mean time per file: {sum(r['seconds'] for r in rows) / total:.2f} s")
    if a.csv:
        write_csv(a.csv, rows)
