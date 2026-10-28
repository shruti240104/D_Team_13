"""OCR evaluation: CER and WER  (Member 4).

CER (character error rate) = edit distance between reference and hypothesis
characters / number of reference characters.  WER is the same on word lists.
Edit (Levenshtein) distance = minimum number of insertions, deletions and
substitutions needed to turn one sequence into the other.
0.0 is perfect; values can exceed 1.0 when the output has many extra characters.

The command-line tool compares OCR on several image variants of every image
that has ground-truth ``text``:
    raw_photo        plain grey conversion of the whole photo (no scanning)
    warped_gray      perspective-corrected page, NO enhancement
    color / grayscale / bw   the enhanced variants of this project

    python -m evaluation.ocr_metrics sample_data/synthetic --csv ocr.csv
"""
from __future__ import annotations

import argparse
from typing import Dict, List, Sequence

import cv2

from evaluation.dataset import iter_dataset, mean, write_csv


def normalize_text(text: str, lowercase: bool = False) -> str:
    """Collapse all whitespace runs to single spaces (layout is not what we measure)."""
    out = " ".join((text or "").split())
    return out.lower() if lowercase else out


def edit_distance(a: Sequence, b: Sequence) -> int:
    """Levenshtein distance with two rolling rows (O(len(a)*len(b)) time, O(len(b)) memory)."""
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def cer(reference: str, hypothesis: str, lowercase: bool = False) -> float:
    ref, hyp = normalize_text(reference, lowercase), normalize_text(hypothesis, lowercase)
    if not ref:
        return 0.0 if not hyp else 1.0
    return edit_distance(ref, hyp) / len(ref)


def wer(reference: str, hypothesis: str, lowercase: bool = False) -> float:
    ref, hyp = normalize_text(reference, lowercase).split(), normalize_text(hypothesis, lowercase).split()
    if not ref:
        return 0.0 if not hyp else 1.0
    return edit_distance(ref, hyp) / len(ref)


def evaluate_folder(folder: str, lang: str = "eng", psm: int = 3):
    # imports here so that cer()/wer() can be used without OpenCV/Tesseract
    from backend.enhancement import EnhancementOptions
    from backend.ocr import configure_tesseract, check_tesseract, run_ocr
    from backend.pipeline import PipelineOptions, run_pipeline

    configure_tesseract()
    ok, msg = check_tesseract(lang)
    if not ok:
        raise SystemExit("OCR evaluation needs Tesseract: " + msg)
    rows: List[Dict] = []
    for path, ann in iter_dataset(folder):
        if "text" not in ann:
            continue
        result = run_pipeline(path.read_bytes(), path.name, PipelineOptions(run_ocr=False))
        row: Dict = {"image": path.name}
        if not result.ok:
            row["pipeline_error"] = f"{result.error.stage}/{result.error.code}"
            rows.append(row)
            continue
        variants = {
            "raw_photo": cv2.cvtColor(result.original, cv2.COLOR_BGR2GRAY),
            "warped_gray": cv2.cvtColor(result.warp.image, cv2.COLOR_BGR2GRAY),
            **result.enhancement.variants,
        }
        for name, img in variants.items():
            ocr = run_ocr(img, lang=lang, psm=psm)
            row[f"{name}_cer"] = round(cer(ann["text"], ocr.text), 4) if ocr.status == "ok" else None
            row[f"{name}_wer"] = round(wer(ann["text"], ocr.text), 4) if ocr.status == "ok" else None
        rows.append(row)
    summary = {}
    for key in {k for r in rows for k in r if k.endswith(("_cer", "_wer"))}:
        summary[key] = mean([r[key] for r in rows if r.get(key) is not None])
    return rows, dict(sorted(summary.items()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--lang", default="eng")
    ap.add_argument("--psm", type=int, default=3)
    ap.add_argument("--csv")
    args = ap.parse_args()
    rows, summary = evaluate_folder(args.folder, args.lang, args.psm)
    for r in rows:
        print(r)
    print("MEAN CER/WER", summary)
    if args.csv:
        write_csv(args.csv, rows)
