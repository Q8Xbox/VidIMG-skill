"""Video guard calibration on the person's own videos (PLAN.md phase 2, video rows).

From each 720p-class video, short segments with large faces are taken as ground truth, degraded
like a messaging-app copy (x2 smaller + low-bitrate H.264), and restored with faithful and damaging
configurations. Every video gate is measured with the input-only reference (as in real use) and
each output is labelled from its true distance to the original (VMAF and face identity vs GT).
"""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from . import imgfile  # noqa: E402

from . import config, faces, guard
from . import media
from .adapters import free_sr, video2x
from .calibrate import _best_threshold
from .guard.video import _ffmpeg, _frames_by_index, verify_video

SEG_SECONDS = 3.0
MIN_GT_FACE = 80          # face height in the original so that it stays >= 40 px after x2
LR_BITRATE = "450k"

REFERENCE = "general"           # the pipeline's fast reference model: relative VMAF is measured against it
CANDIDATES = [
    ("general", True, lambda: [free_sr.FreeSR(model="general_x4v3", scale=2)]),
    ("realesrnet", True, lambda: [free_sr.FreeSR(model="realesrnet_x4", scale=2)]),
    ("bsrnet", True, lambda: [free_sr.FreeSR(model="bsrnet_x4", scale=2)]),
]


def _run(args: list[str], timeout: int = 1800) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True, encoding="utf-8", stdin=subprocess.DEVNULL, timeout=timeout,
                          errors="replace")


def pick_segments(video: Path, n: int = 3, probes: int = 24) -> list[tuple[float, int]]:
    """Start times of up to n segments whose smallest face is >= MIN_GT_FACE (largest first)."""
    info = media.probe(video)
    dur = info["duration"]
    found = []
    with tempfile.TemporaryDirectory(prefix="vidimg_seg_") as tmp:
        for i in range(probes):
            t = dur * (i + 0.5) / probes
            if t + SEG_SECONDS > dur:
                continue
            p = Path(tmp) / f"p{i}.png"
            _run([_ffmpeg(), "-hide_banner", "-loglevel", "error", "-ss", f"{t:.2f}", "-i", str(video), "-frames:v", "1",
                  "-y", str(p)], 120)
            img = imgfile.imread(str(p))
            if img is None:
                continue
            fs = faces.detect(img, score=guard.FACE_SCORE)
            if fs:
                mn = int(min(f.box[3] for f in fs))
                if mn >= MIN_GT_FACE:
                    found.append((t, mn))
    found.sort(key=lambda x: -x[1])
    chosen: list[tuple[float, int]] = []
    for t, mn in found:          # keep segments apart
        if all(abs(t - c) > 30 for c, _ in chosen):
            chosen.append((t, mn))
        if len(chosen) == n:
            break
    return chosen


def _gt_vmaf(gt: Path, out: Path) -> float | None:
    g = media.probe(gt)
    graph = (f"[1:v]scale={g['width']}:{g['height']}:flags=bicubic,setsar=1,format=yuv420p[d];"
             f"[0:v]setsar=1,format=yuv420p[s];[d][s]libvmaf=n_threads=8")
    r = _run([_ffmpeg(), "-hide_banner", "-nostats", "-i", str(gt), "-i", str(out), "-lavfi", graph, "-f", "null", "-"])
    m = re.findall(r"VMAF score:\s*([0-9.]+)", r.stderr)
    return float(m[-1]) if m else None


def _gt_identity(gt: Path, out: Path, samples: int = 4) -> float | None:
    a, b = media.probe(gt), media.probe(out)
    n = min(a["frames"], b["frames"])
    idx = [int(n * (i + 0.5) / samples) for i in range(samples)]
    with tempfile.TemporaryDirectory(prefix="vidimg_gtid_") as tmp:
        gm = _frames_by_index(gt, idx, Path(tmp), "g")
        om = _frames_by_index(out, idx, Path(tmp), "o")
        sims = []
        for i in idx:
            if i not in gm or i not in om:
                continue
            g, o = imgfile.imread(str(gm[i])), imgfile.imread(str(om[i]))
            o = cv2.resize(o, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_AREA)
            fs = faces.detect(g, score=guard.FACE_SCORE)
            got = faces.match(fs, faces.detect(o, score=0.4))
            sims += [faces.similarity(faces.embedding(g, f), faces.embedding(o, x or f)) for f, x in zip(fs, got)]
    return min(sims) if sims else None


def _boil(src: Path, dst: Path) -> bool:
    """Synthetic temporal damage: per-frame noise 'boiling' plus oversharpening."""
    r = _run([_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", "-i", str(src), "-vf",
              "noise=alls=18:allf=t,unsharp=7:7:2.0", "-c:v", "ffv1", "-c:a", "copy", str(dst)])
    return r.returncode == 0 and dst.exists()


def run(folder: Path, *, max_videos: int | None = None, out_root: Path | None = None, log=print) -> dict:
    vids = [p for p in sorted(Path(folder).glob("*.mp4")) if (media.probe(p)["height"] or 0) >= 700][:max_videos]
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    root = (out_root or config.WORKSPACE / "calibration_video") / stamp
    root.mkdir(parents=True, exist_ok=True)
    records = []
    for vi, v in enumerate(vids, 1):
        segs = pick_segments(v)
        log(f"[{vi}/{len(vids)}] {v.name}: {len(segs)} face segments {[(round(t), mn) for t, mn in segs]}")
        for si, (t, mn) in enumerate(segs):
            work = root / f"{vi:02d}_{si}_{v.stem[:20]}"
            work.mkdir(parents=True, exist_ok=True)
            gt, lr = work / "gt.mkv", work / "lr.mp4"
            _run([_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{t:.2f}", "-t", f"{SEG_SECONDS}", "-i", str(v),
                  "-c:v", "ffv1", "-c:a", "flac", str(gt)])
            _run([_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", "-i", str(gt), "-vf", "scale=trunc(iw/4)*2:trunc(ih/4)*2:flags=area",
                  "-c:v", "libx264", "-b:v", LR_BITRATE, "-pix_fmt", "yuv420p", "-c:a", "aac", str(lr)])
            outs: list[tuple[str, bool, Path]] = []
            res = work / "out"            # never next to the input (adapters refuse that by design)
            res.mkdir(exist_ok=True)
            for name, faithful, stages in CANDIDATES:
                o = res / f"{name}.mkv"
                t0 = time.monotonic()
                r = free_sr.run(lr, o, stages(), "ffv1")
                log(f"    {name:<11} {'ok' if r.ok else 'FAILED ' + str(r.error)} {time.monotonic() - t0:.0f}s")
                if r.ok:
                    outs.append((name, faithful, o))
            ref_out = res / f"{REFERENCE}.mkv"
            if ref_out.exists() and _boil(ref_out, res / "boil.mkv"):
                outs.append(("boil", False, res / "boil.mkv"))
            vx = video2x.run(lr, res / "v2x_anime.mkv", "realesrgan", content="anime", scale=2,
                             model="realesr-animevideov3", codec="ffv1", extra={})
            if vx.ok:
                outs.append(("v2x_anime", False, vx.output))
            for name, faithful, o in outs:
                rep = verify_video(lr, o)
                g = {x.name: x.value for x in rep.gates}
                rec = {"video": v.name, "segment": si, "start": round(t, 2), "candidate": name, "prior_faithful": faithful,
                       "guard_passed": rep.passed, "failed": rep.failed(),
                       **{k: g.get(k) for k in ("identity", "face_backprojection", "backprojection", "flicker", "vmaf", "color")},
                       "gt_vmaf": _gt_vmaf(gt, o), "gt_identity": _gt_identity(gt, o)}
                records.append(rec)
                log(f"      {name:<11} guard={'PASS' if rep.passed else 'FAIL ' + ','.join(rep.failed())} "
                    f"vmaf_in={rec['vmaf']} flicker={rec['flicker']} gt_vmaf={rec['gt_vmaf'] and round(rec['gt_vmaf'], 1)} "
                    f"gt_id={rec['gt_identity'] and round(rec['gt_identity'], 3)}")
            (root / "records.json").write_text(json.dumps(records, indent=1), encoding="utf-8")
    result = summarise(records)
    (root / "summary.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    result["folder"] = str(root)
    return result


def rescore(run_dir: Path, log=print) -> dict:
    """Re-evaluate saved outputs of a finished run with the current video guard (no restoring)."""
    run_dir = Path(run_dir)
    old = json.loads((run_dir / "records.json").read_text(encoding="utf-8"))
    seg_dirs = sorted(x for x in run_dir.iterdir() if x.is_dir())
    records = []
    ref_vmaf: dict[tuple, float | None] = {}
    # the reference model first in each segment: relative VMAF is measured against it
    order = sorted(old, key=lambda r: (r["video"], r["segment"], r["candidate"] != REFERENCE))
    for r in order:
        vi = sorted({x["video"] for x in old}).index(r["video"])
        d = next(x for x in seg_dirs if x.name.startswith(f"{vi + 1:02d}_{r['segment']}_"))
        o = d / "out" / f"{r['candidate']}.mkv"
        if not o.exists():
            continue
        key = (r["video"], r["segment"])
        rep = verify_video(d / "lr.mp4", o, reference_vmaf=None if r["candidate"] == REFERENCE else ref_vmaf.get(key))
        if r["candidate"] == REFERENCE:
            ref_vmaf[key] = next((g.value for g in rep.gates if g.name == "vmaf"), None)
        g = {x.name: x.value for x in rep.gates}
        records.append(r | {"guard_passed": rep.passed, "failed": rep.failed(),
                            **{k: g.get(k) for k in ("identity", "face_backprojection", "backprojection", "flicker", "vmaf",
                                                     "color")}})
        log(f"  {d.name[:24]:<25}{r['candidate']:<11} {'PASS' if rep.passed else 'FAIL ' + ','.join(rep.failed())}")
    (run_dir / "records_rescored.json").write_text(json.dumps(records, indent=1), encoding="utf-8")
    result = summarise(records)
    (run_dir / "summary_rescored.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    return result


def label(records: list[dict]) -> list[dict]:
    """Ground-truth labels per segment: faithful when true VMAF is within 5 of the segment's best and
    true identity within 0.05; damaging when VMAF drops > 10 or identity > 0.10; else unlabelled."""
    seg_best: dict[tuple, tuple[float, float]] = {}
    for r in records:
        k = (r["video"], r["segment"])
        bv, bi = seg_best.get(k, (-1.0, -1.0))
        seg_best[k] = (max(bv, r["gt_vmaf"] or -1), max(bi, r["gt_identity"] if r["gt_identity"] is not None else -1))
    for r in records:
        bv, bi = seg_best[(r["video"], r["segment"])]
        dv = bv - (r["gt_vmaf"] or 0)
        di = (bi - r["gt_identity"]) if r["gt_identity"] is not None and bi >= 0 else 0.0
        r["label"] = True if dv <= 5 and di <= 0.05 else (False if dv > 10 or di > 0.10 else None)
    return records


def summarise(records: list[dict]) -> dict:
    label(records)
    usable = [r for r in records if r["label"] is not None]
    out = {"outputs": len(records), "labelled": len(usable), "thresholds": {}}
    for key, d in (("vmaf", "min"), ("flicker", "max"), ("identity", "min"), ("backprojection", "min")):
        vals = [(r[key], r["label"]) for r in usable if r.get(key) is not None]
        t, acc = _best_threshold(vals, d)
        out["thresholds"][key] = {"threshold": t, "balanced_accuracy": acc, "samples": len(vals)}
    good = [r for r in usable if r["label"]]
    bad = [r for r in usable if not r["label"]]
    out["current_guard"] = {"faithful_accepted": f"{sum(r['guard_passed'] for r in good)}/{len(good)}",
                            "damaging_rejected": f"{sum(not r['guard_passed'] for r in bad)}/{len(bad)}"}
    return out
