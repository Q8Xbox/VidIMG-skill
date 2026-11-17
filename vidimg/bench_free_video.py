"""Free video models on the video ground truth (12 segments of the person's 720p videos:
gt.mkv = original, lr.mp4 = half size at 450 kbps). Scores: VMAF / identity / PSNR against the original,
frames paired by index (bench_metrics._score), plus speed. Usage: python -m vidimg.bench_free_video [--limit N]
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

from . import config
from . import media
from .adapters import free_sr
from .bench_metrics import CAL, _score

MODELS = ["general_x4v3", "bsrnet_x4", "realesrnet_x4"]
ROOT = config.WORKSPACE / "bench_free" / "video"


def run(limit: int | None = None, log=print) -> dict:
    segs = sorted(p for p in CAL.iterdir() if p.is_dir())[:limit]
    records = []
    for seg in segs:
        gt, lr = seg / "gt.mkv", seg / "lr.mp4"
        out = ROOT / seg.name
        out.mkdir(parents=True, exist_ok=True)
        n = media.probe(lr)["frames"]
        for m in MODELS:
            o = out / f"{m}.mkv"
            t0 = time.monotonic()
            r = free_sr.run(lr, o, [free_sr.FreeSR(model=m, scale=2)], "ffv1")
            dt = time.monotonic() - t0
            if not r.ok:
                log(f"   {seg.name} {m}: FAILED {r.error}")
                continue
            rec = {"segment": seg.name, "candidate": f"free_{m}", **_score(gt, o), "fps": round(n / dt, 1)}
            records.append(rec)
            log(f"   {seg.name[:24]:<24} {m:<14} vmaf={rec['vmaf']} id={rec['identity']} psnr={rec['psnr']} {rec['fps']} fps")
        (ROOT / "records.json").write_text(json.dumps(records, indent=1), encoding="utf-8")
    by: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by[r["candidate"]].append(r)
    mean = lambda xs: round(statistics.fmean(xs), 3) if xs else None
    base = {r["segment"]: r for r in records if r["candidate"] == "free_general_x4v3"}   # fast reference
    summary = {}
    for c, rs in by.items():
        dv = [r["vmaf"] - base[r["segment"]]["vmaf"] for r in rs if r["segment"] in base and r.get("vmaf") is not None]
        summary[c] = {"n": len(rs), "vmaf": mean([r["vmaf"] for r in rs if r.get("vmaf") is not None]),
                      "identity": mean([r["identity"] for r in rs if r.get("identity") is not None]),
                      "psnr": mean([r["psnr"] for r in rs if r.get("psnr") is not None]),
                      "d_vmaf_vs_general": mean(dv), "wins_vs_general": sum(1 for x in dv if x > 0),
                      "fps": mean([r["fps"] for r in rs if r.get("fps")])}
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    s = run(ap.parse_args().limit)
    print(json.dumps(s, indent=1))
