"""Guard calibration on the person's own photos (PLAN.md phase 2).

For each photo: take a sharp crop as ground truth (faces kept in frame), degrade it in a controlled
way (x4 smaller + JPEG), restore it with faithful and known-damaging configurations, measure every
guard gate (input-only reference, as in real use) and the true distance to the ground truth, then
choose per-gate thresholds that best separate faithful from damaging outputs.

Labels come from the benchmark's prior (which configurations preserve identity) and are checked
per image against the ground truth; disagreements are reported for human review, never hidden.
Thresholds become active only with --apply and at least MIN_IMAGES_TO_APPLY photos.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from . import config, faces, guard
from .adapters import free_sr, video2x

MIN_IMAGES_TO_APPLY = 20
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
CROP_MAX = 1600
# Degraded faces must stay large enough for identity to be recoverable at all: the first run on the
# person's photos used a fixed x4 and left faces of 3-27 px, where every model (faithful or not) scored
# 0.0-0.5 against the true face, so those photos carried no signal. Faces >= ~40 px gave 0.80-0.90.
MIN_LR_FACE = 40


@dataclass(frozen=True)
class Candidate:
    name: str
    faithful: bool          # prior label from research/bench/RESULTS.md
    engine: str
    kwargs: dict


CANDIDATES = [
    # free open models (the pipeline's candidates) and a classical reference
    Candidate("free_bsrnet_x4", True, "free", {"model": "bsrnet_x4"}),
    Candidate("free_realesrnet_x4", True, "free", {"model": "realesrnet_x4"}),
    Candidate("free_swinir_real_x4", True, "free", {"model": "swinir_real_x4"}),
    Candidate("free_realesrgan_x4", True, "free", {"model": "realesrgan_x4"}),
    Candidate("free_general_x4v3", True, "free", {"model": "general_x4v3"}),
    Candidate("free_lanczos", True, "free", {"model": None}),
    # anime-only processors run on real photos ONLY here, as negative examples for calibration
    Candidate("v2x_esrgan_plus", False, "video2x", {"processor": "realesrgan", "model": "realesrgan-plus"}),
    Candidate("v2x_cugan", False, "video2x", {"processor": "realcugan", "model": "models-se"}),
]

# gate -> (metric key in the record, direction): 'min' means pass when value >= threshold
GATES = {"identity_min": ("identity", "min"), "face_backproj_min": ("face_backprojection", "min"),
         "backproj_ssim_min": ("backprojection", "min"), "delta_e_max": ("color", "max")}


def plan_degradation(img: np.ndarray) -> tuple[np.ndarray | None, int, str]:
    """Ground-truth crop and downscale factor (4 or 2) that keep the smallest face >= MIN_LR_FACE px.

    The crop covers the union of all faces (padded) instead of the image centre; a crop larger than
    CROP_MAX is scaled down to it. Returns (gt, factor, note); gt is None when no factor keeps faces
    identifiable (the photo is then skipped, with the reason in note)."""
    h, w = img.shape[:2]
    found = faces.detect(img, score=guard.FACE_SCORE) if faces.available() else []
    if found:
        box = np.array([f.box for f in found])
        x0, y0 = box[:, 0].min(), box[:, 1].min()
        x1, y1 = (box[:, 0] + box[:, 2]).max(), (box[:, 1] + box[:, 3]).max()
        pad = 0.6 * np.median(box[:, 3])
        x0, y0 = int(max(0, x0 - pad)), int(max(0, y0 - pad))
        x1, y1 = int(min(w, x1 + pad)), int(min(h, y1 + pad))
        gt = img[y0:y1, x0:x1]
    else:
        cw, ch = min(w, CROP_MAX), min(h, round(CROP_MAX * 0.625))
        gt = img[(h - ch) // 2:(h - ch) // 2 + ch, (w - cw) // 2:(w - cw) // 2 + cw]
    k = min(1.0, CROP_MAX / max(gt.shape[:2]))
    if k < 1:
        gt = cv2.resize(gt, (round(gt.shape[1] * k), round(gt.shape[0] * k)), interpolation=cv2.INTER_AREA)
    min_face = min(f.box[3] for f in found) * k if found else None
    if min_face is None:
        factor, note = 4, "no faces: non-face gates only"
    elif min_face >= 4 * MIN_LR_FACE:
        factor, note = 4, f"smallest face {min_face:.0f}px -> {min_face / 4:.0f}px at x4"
    elif min_face >= 2 * MIN_LR_FACE:
        factor, note = 2, f"smallest face {min_face:.0f}px -> {min_face / 2:.0f}px at x2"
    else:
        return None, 0, f"skipped: smallest face {min_face:.0f}px cannot stay >= {MIN_LR_FACE}px after x2"
    hh, ww = gt.shape[:2]
    gt = gt[: hh - hh % factor, : ww - ww % factor]
    if min(gt.shape[:2]) < 32 * factor:
        return None, 0, "skipped: crop too small"
    return gt, factor, note


def _gt_identity(gt: np.ndarray, out: np.ndarray) -> float | None:
    if not faces.available():
        return None
    ref = faces.detect(gt)
    if not ref:
        return None
    found = faces.match(ref, faces.detect(out, score=0.4))
    return min(faces.similarity(faces.embedding(gt, f), faces.embedding(out, g or f)) for f, g in zip(ref, found))


def _score(c: Candidate, gt: np.ndarray, lr_p: Path, out_p: Path, photo: str, bic_psnr: float,
           gt_has_faces: bool, seconds: float = 0.0) -> dict:
    out = cv2.imread(str(out_p))
    if out.shape[:2] != gt.shape[:2]:
        out = cv2.resize(out, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_AREA)
    rep = guard.verify_image(lr_p, out_p)
    gv = {g.name: g.value for g in rep.gates}
    gt_id = _gt_identity(gt, out) if gt_has_faces else None
    # Video2X processors are "damaging" because they distort real faces; on photos without faces
    # their output can be acceptable, so they carry no label there (trial: bird ESRGAN looked fine)
    label = None if (c.engine == "video2x" and not gt_has_faces) else c.faithful
    rec = {"photo": photo, "candidate": c.name, "prior_faithful": label, "guard_passed": rep.passed,
           "failed": rep.failed(),
           **{k: gv.get(k) for k in ("identity", "landmarks", "face_backprojection", "backprojection", "color",
                                     "hallucination")},
           "gt_identity": gt_id, "gt_psnr": round(cv2.PSNR(gt, out), 3), "bicubic_psnr": round(bic_psnr, 3),
           "seconds": seconds}
    rec["gt_conflict"] = bool(label is not None and gt_id is not None and
                              ((label and gt_id < 0.80) or (not label and gt_id >= 0.90)))
    return rec


def rescore(run_dir: Path, log=print) -> dict:
    """Re-evaluate the outputs of a finished calibration run with the current guard (no restoring)."""
    run_dir = Path(run_dir)
    old = json.loads((run_dir / "records.json").read_text(encoding="utf-8")) if (run_dir / "records.json").exists() else []
    photo_of = {Path(r["photo"]).stem: r["photo"] for r in old}
    records = []
    dirs = sorted(d for d in run_dir.iterdir() if d.is_dir() and (d / "in" / "gt.png").exists())
    for d in dirs:
        gt = cv2.imread(str(d / "in" / "gt.png"))
        lr_p = d / "in" / "lr.jpg"
        bic = cv2.resize(cv2.imread(str(lr_p)), (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_CUBIC)
        has_faces = bool(faces.available() and faces.detect(gt, score=guard.FACE_SCORE))
        stem = d.name.split("_", 1)[1]
        for c in CANDIDATES:
            out_p = d / f"lr_{c.name}.png"
            if out_p.exists():
                records.append(_score(c, gt, lr_p, out_p, photo_of.get(stem, stem), cv2.PSNR(gt, bic), has_faces))
        log(f"rescored {d.name}")
    (run_dir / "records_rescored.json").write_text(json.dumps(records, indent=1), encoding="utf-8")
    result = _summarise(records, len(dirs))
    (run_dir / "summary_rescored.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    result["folder"] = str(run_dir)
    return result


def _restore(c: Candidate, lr: Path, work: Path, factor: int = 4) -> Path | None:
    if c.engine == "free":
        if c.kwargs["model"] is None:                       # classical Lanczos reference
            img = cv2.imread(str(lr))
            out = work / f"{lr.stem}_{c.name}.png"
            cv2.imwrite(str(out), cv2.resize(img, (img.shape[1] * factor, img.shape[0] * factor),
                                             interpolation=cv2.INTER_LANCZOS4))
            return out
        r = free_sr.upscale(lr, work, model=c.kwargs["model"], scale=factor, suffix=f"_{c.name}")
    else:
        kw = dict(c.kwargs)
        r = video2x.run(lr, work / f"{lr.stem}_{c.name}.png", kw.pop("processor"), content="anime",
                        scale=factor, codec="png", extra={}, **kw)
    return r.output if r.ok else None


def _best_threshold(values: list[tuple[float, bool]], direction: str) -> tuple[float, float]:
    """Threshold maximising balanced accuracy; returns (threshold, balanced_accuracy)."""
    good = [v for v, f in values if f]
    bad = [v for v, f in values if not f]
    if not good or not bad:
        return float("nan"), float("nan")
    best = (float("nan"), -1.0)
    for t in sorted({v for v, _ in values}):
        if direction == "min":
            tpr = np.mean([v >= t for v in good]); tnr = np.mean([v < t for v in bad])
        else:
            tpr = np.mean([v <= t for v in good]); tnr = np.mean([v > t for v in bad])
        acc = (tpr + tnr) / 2
        if acc > best[1]:
            best = (float(t), float(acc))
    return best


def run(folder: Path, *, limit: int | None = None, apply: bool = False, out_root: Path | None = None,
        log=print) -> dict:
    folder = Path(folder)
    photos = sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXT)[:limit]
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    root = (out_root or config.WORKSPACE / "calibration") / stamp
    root.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    skipped: list[dict] = []
    for n, photo in enumerate(photos, 1):
        img = cv2.imread(str(photo))
        if img is None:
            continue
        gt, factor, note = plan_degradation(img)
        if gt is None:
            skipped.append({"photo": str(photo), "reason": note})
            log(f"[{n}/{len(photos)}] {photo.name}: {note}")
            continue
        work = root / f"{n:03d}_{photo.stem}"
        (work / "in").mkdir(parents=True, exist_ok=True)
        lr_img = cv2.resize(gt, (gt.shape[1] // factor, gt.shape[0] // factor), interpolation=cv2.INTER_AREA)
        gt_p, lr_p = work / "in" / "gt.png", work / "in" / "lr.jpg"
        cv2.imwrite(str(gt_p), gt)
        cv2.imwrite(str(lr_p), lr_img, [cv2.IMWRITE_JPEG_QUALITY, 60])
        bic = cv2.resize(cv2.imread(str(lr_p)), (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_CUBIC)
        bic_psnr = cv2.PSNR(gt, bic)
        has_faces = bool(faces.available() and faces.detect(gt, score=guard.FACE_SCORE))
        log(f"[{n}/{len(photos)}] {photo.name}: gt {gt.shape[1]}x{gt.shape[0]} x{factor} ({note})")
        (work / "in" / "factor.txt").write_text(str(factor))
        for c in CANDIDATES:
            t0 = time.monotonic()
            out_p = _restore(c, lr_p, work, factor)
            if out_p is None:
                log(f"    {c.name:<16} failed to run")
                continue
            rec = _score(c, gt, lr_p, out_p, str(photo), bic_psnr, has_faces, round(time.monotonic() - t0, 1))
            records.append(rec)
            gt_id = rec["gt_identity"]
            log(f"    {c.name:<16} guard={'PASS' if rec['guard_passed'] else 'FAIL'} id={rec['identity']} bp={rec['backprojection']} "
                f"gt_id={None if gt_id is None else round(gt_id, 3)} {'CONFLICT' if rec['gt_conflict'] else ''}")
    result = _summarise(records, len(photos) - len(skipped))
    result["skipped"] = skipped
    (root / "records.json").write_text(json.dumps(records, indent=1), encoding="utf-8")
    (root / "summary.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    result["folder"] = str(root)
    if apply:
        if result["images"] < MIN_IMAGES_TO_APPLY:
            result["applied"] = False
            result["apply_refused"] = f"needs at least {MIN_IMAGES_TO_APPLY} photos, got {result['images']}"
        else:
            new = guard.thresholds()
            for gate, info in result["thresholds"].items():
                if not np.isnan(info["threshold"]):
                    new[gate] = info["threshold"]
            new["calibrated"] = True
            new["calibrated_on"] = {"images": result["images"], "time": stamp, "folder": str(root)}
            config.ensure_workspace()
            guard.THRESHOLDS_FILE.write_text(json.dumps(new, indent=1), encoding="utf-8")
            result["applied"] = True
    return result


GT_FAITHFUL_LOSS = 0.05   # within this of the photo's best true identity -> faithful
GT_DAMAGING_LOSS = 0.10   # more than this below it -> damaging; in between -> unlabelled


def label(records: list[dict]) -> list[dict]:
    """Ground-truth labels where faces exist (true identity relative to the best output of the same
    photo); the benchmark prior only for photos without faces. The first run on the person's photos
    showed the prior is unreliable there: low-creativity Redefine matched the best faithful model's
    true identity (0.747 vs 0.747), while Recovery v2 was genuinely worst."""
    best: dict[str, float] = {}
    for r in records:
        if r.get("gt_identity") is not None:
            best[r["photo"]] = max(best.get(r["photo"], -1.0), r["gt_identity"])
    for r in records:
        if r.get("gt_identity") is not None:
            loss = best[r["photo"]] - r["gt_identity"]
            r["label"] = True if loss <= GT_FAITHFUL_LOSS else (False if loss > GT_DAMAGING_LOSS else None)
            r["label_source"] = "ground_truth"
        else:
            r["label"] = r["prior_faithful"]
            r["label_source"] = "prior"
    return records


def _summarise(records: list[dict], images: int) -> dict:
    current = guard.thresholds()
    label(records)
    usable = [r for r in records if r["label"] is not None]
    out = {"images": images, "outputs": len(records), "labelled": len(usable),
           "label_sources": {s: sum(r["label_source"] == s for r in usable) for s in ("ground_truth", "prior")},
           "thresholds": {}, "conflicts": [r for r in records if r["gt_conflict"]]}
    for gate, (key, direction) in GATES.items():
        vals = [(r[key], r["label"]) for r in usable if r.get(key) is not None]
        t, acc = _best_threshold(vals, direction)
        out["thresholds"][gate] = {"threshold": t, "balanced_accuracy": acc, "current": current.get(gate), "samples": len(vals)}
    if usable:
        tp = sum(r["guard_passed"] and r["label"] for r in usable)
        tn = sum((not r["guard_passed"]) and (not r["label"]) for r in usable)
        good = sum(bool(r["label"]) for r in usable)
        out["current_guard"] = {"faithful_accepted": f"{tp}/{good}", "damaging_rejected": f"{tn}/{len(usable) - good}"}
    return out
