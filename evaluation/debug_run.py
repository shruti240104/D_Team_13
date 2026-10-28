from __future__ import annotations

import argparse
from pathlib import Path

from backend.export import save_debug_bundle
from backend.ocr import configure_tesseract
from backend.pipeline import PipelineOptions, run_pipeline

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("--out", default="debug_output")
    ap.add_argument("--no-ocr", action="store_true")
    a = ap.parse_args()
    configure_tesseract()
    path = Path(a.image)
    res = run_pipeline(path.read_bytes(), path.name, PipelineOptions(run_ocr=not a.no_ocr))
    files = save_debug_bundle(res, a.out)
    print("OK" if res.ok else f"FAILED at {res.error.stage}/{res.error.code}: {res.error.message}")
    if res.detection is not None:
        d = res.detection
        print(f"detection: success={d.success} method={d.method!r} quality={d.quality:.2f} ambiguous={d.ambiguous}")
    for w in res.warnings:
        print("warning:", w)
    print("timings:", res.timings)
    if res.ocr is not None:
        print("ocr:", res.ocr.status, res.ocr.message or f"mean confidence {res.ocr.mean_confidence:.0f}")
        if res.ocr.status == "ok":
            (Path(a.out) / "text.txt").write_text(res.ocr.text, encoding="utf-8")
    print(f"{len(files)} images written to {a.out}")
