"""Free models on the photo ground truth (the calibration run on the person's photos).

Each calibration photo folder holds the real original (in/gt.png) and its controlled degradation
(in/lr.jpg, x2 or x4). The free models restore the same lr.jpg; every output is scored by
calibrate._score against the original and labelled by calibrate.label
(true identity relative to the best output of the same photo). Result: which free models are faithful,
how they compare with each other, and whether the guard still separates faithful from damaging.

Usage: python -m vidimg.bench_free [--run workspace/calibration/<stamp>]
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
from collections import defaultdict
from pathlib import Path

import cv2
from . import imgfile  # noqa: E402

from . import config, faces, guard
from .adapters import free_sr
from .calibrate import CANDIDATES, Candidate, _score, label

RUN = config.WORKSPACE / "calibration" / "20261114-235009"
FREE = [c for c in CANDIDATES if c.engine == "free"]


def run(run_dir: Path = RUN, log=print) -> dict:
    run_dir = Path(run_dir)
    dirs = sorted(d for d in run_dir.iterdir() if d.is_dir() and (d / "in" / "gt.png").exists())
    stage = run_dir / "_free_inputs"
    stage.mkdir(exist_ok=True)
    by_factor: dict[int, list[Path]] = defaultdict(list)
    for d in dirs:
        f = int((d / "in" / "factor.txt").read_text()) if (d / "in" / "factor.txt").exists() else 4
        p = stage / f"{d.name}.jpg"
        shutil.copyfile(d / "in" / "lr.jpg", p)                 # unique names for grouped calls
        by_factor[f].append(p)
    for c in FREE:
        for factor, inputs in by_factor.items():
            outs = stage / c.name
            if c.kwargs["model"] is None:
                for p in inputs:
                    img = imgfile.imread(str(p))
                    imgfile.imwrite(str(run_dir / p.stem / f"lr_{c.name}.png"),
                                cv2.resize(img, (img.shape[1] * factor, img.shape[0] * factor), interpolation=cv2.INTER_LANCZOS4))
                continue
            res = free_sr.upscale_images(inputs, outs, model=c.kwargs["model"], scale=factor, suffix=f"_{c.name}")
            for p in inputs:
                d = res[str(p)]
                if d.get("ok"):
                    shutil.copyfile(d["output"], run_dir / p.stem / f"lr_{c.name}.png")
                else:
                    log(f"   {c.name} {p.stem}: {d.get('error')}")
        log(f"{c.name}: done")
    records = []
    for d in dirs:
        gt = imgfile.imread(str(d / "in" / "gt.png"))
        lr_p = d / "in" / "lr.jpg"
        bic = cv2.resize(imgfile.imread(str(lr_p)), (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_CUBIC)
        has_faces = bool(faces.available() and faces.detect(gt, score=guard.FACE_SCORE))
        for c in CANDIDATES:
            out_p = d / f"lr_{c.name}.png"
            if out_p.exists():
                r = _score(c, gt, lr_p, out_p, d.name, cv2.PSNR(gt, bic), has_faces)
                r["engine"] = c.engine
                records.append(r)
        log(f"scored {d.name}")
    records = label(records)
    (run_dir / "records_free.json").write_text(json.dumps(records, indent=1), encoding="utf-8")
    summary = summarise(records)
    (run_dir / "summary_free.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


def summarise(records: list[dict]) -> dict:
    by: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by[r["candidate"]].append(r)
    mean = lambda xs: round(statistics.fmean(xs), 4) if xs else None
    out = {}
    for c, rs in by.items():
        lab = [r for r in rs if r.get("label") is not None]
        out[c] = {
            "n": len(rs),
            "gt_identity": mean([r["gt_identity"] for r in rs if r.get("gt_identity") is not None]),
            "gt_psnr": mean([r["gt_psnr"] for r in rs]),
            "psnr_vs_bicubic": mean([r["gt_psnr"] - r["bicubic_psnr"] for r in rs]),
            "guard_pass": f"{sum(r['guard_passed'] for r in rs)}/{len(rs)}",
            "labelled_faithful": f"{sum(1 for r in lab if r['label'])}/{len(lab)}",
            "guard_correct": f"{sum(1 for r in lab if r['label'] == r['guard_passed'])}/{len(lab)}",
        }
    return dict(sorted(out.items(), key=lambda kv: -(kv[1]["gt_identity"] or 0)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=RUN)
    a = ap.parse_args()
    s = run(a.run)
    print(f"{'candidate':<22} {'n':>3} {'gt_id':>7} {'psnr':>7} {'dPSNR':>6} {'guard':>6} {'faithful':>9} {'guard ok':>9}")
    for c, v in s.items():
        print(f"{c:<22} {v['n']:>3} {v['gt_identity']!s:>7} {v['gt_psnr']!s:>7} {v['psnr_vs_bicubic']!s:>6} "
              f"{v['guard_pass']:>6} {v['labelled_faithful']:>9} {v['guard_correct']:>9}")
