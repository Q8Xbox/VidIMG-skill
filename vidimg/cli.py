"""Command line (free tools only): `python -m vidimg <command>`.

doctor [--json]                    environment report (open-model runtime, ffmpeg, classical engines, GPU)
restore PHOTOS_OR_FOLDERS [--out]  photo pipeline: analyse, restore, guard, review page
restore-video VIDEO [--out]        video pipeline: field handling, guarded previews, full run, report
analyze FILE                       measured degradation profile of a photo or video (writes nothing)
verify ORIGINAL RESULT             fidelity guard on any pair (photo or video)
upscale IMG OUTDIR --model M       one open model on one photo (bsrnet_x4, realesrnet_x4, swinir_real_x4, ...)
denoise IMG OUTDIR                 SCUNet real-photo denoise
video2x SRC DST --processor P      Video2X (rife for any content; realesrgan/realcugan need --content anime)
calibrate / calibrate-video        tune the guard on your own photos / videos
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import config, env, media
from .adapters import AdapterResult, free_sr, video2x


def _emit(result: AdapterResult, as_json: bool) -> int:
    payload = {"ok": result.ok, "outputs": [str(p) for p in result.outputs], "error": result.error,
               "seconds": result.run.seconds if result.run else None,
               "log": result.run.log_path if result.run else None, "details": result.details}
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=1, default=str))
    else:
        print(("OK  " if result.ok else "FAIL"), ", ".join(payload["outputs"]) or result.error or "")
    return 0 if result.ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="vidimg", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("doctor")

    rv = sub.add_parser("restore", help="restore photos (analyse, restore, guard, retry) + review page")
    rv.add_argument("inputs", nargs="+", type=Path, help="image files or folders")
    rv.add_argument("--out", type=Path, help="output folder (default workspace/review/<time>)")
    rv.add_argument("--deblur", action="store_true",
                    help="opt-in: deblur photos flagged soft/blurred first (NAFNet-GoPro, guard deblur mode)")

    rvv = sub.add_parser("restore-video", help="restore a video (analyse, preview with guard, full run, verify)")
    rvv.add_argument("video", type=Path)
    rvv.add_argument("--out", type=Path)
    rvv.add_argument("--preview-only", action="store_true")
    rvv.add_argument("--encoder", default="hevc10", choices=sorted(media.ENCODERS))
    rvv.add_argument("--scale", type=float, help="exact output scale, e.g. 2 = twice the height (default: automatic)")
    rvv.add_argument("--content", default="auto", choices=["auto", "anime", "real"],
                     help="anime/cartoon or real footage; auto trusts the probe (conservative: real unless sure)")

    an = sub.add_parser("analyze", help="measured degradation profile of an image or video")
    an.add_argument("path", type=Path)

    vf = sub.add_parser("verify", help="fidelity guard: original vs result (image or video)")
    vf.add_argument("original", type=Path)
    vf.add_argument("result", type=Path)
    vf.add_argument("--interpolated", action="store_true")

    up = sub.add_parser("upscale", help="run one open model on one photo")
    up.add_argument("image", type=Path)
    up.add_argument("outdir", type=Path)
    up.add_argument("--model", default="bsrnet_x4", choices=sorted(k for k, m in free_sr.manifest().items() if m["kind"] == "sr"))
    up.add_argument("--scale", type=float, default=2.0, help="output size relative to the input")

    dn = sub.add_parser("denoise", help="SCUNet real-photo denoise at the original size")
    dn.add_argument("image", type=Path)
    dn.add_argument("outdir", type=Path)

    x = sub.add_parser("video2x")
    x.add_argument("src", type=Path)
    x.add_argument("dst", type=Path)
    x.add_argument("--processor", required=True, choices=("rife", "realesrgan", "realcugan", "libplacebo"))
    x.add_argument("--content", default="real", choices=("real", "anime"))
    x.add_argument("--scale", type=int)
    x.add_argument("--model")
    x.add_argument("--fps-multiplier", type=int)
    x.add_argument("--codec", default="libx265")

    cal = sub.add_parser("calibrate", help="calibrate guard thresholds on your own photos")
    cal.add_argument("folder", type=Path, help="photos folder, or a finished run folder with --rescore")
    cal.add_argument("--rescore", action="store_true", help="re-evaluate a finished run with the current guard")
    cal.add_argument("--limit", type=int)
    cal.add_argument("--apply", action="store_true", help="activate thresholds (needs >= 20 photos)")

    cv = sub.add_parser("calibrate-video", help="calibrate video gates on your own 720p+ videos")
    cv.add_argument("folder", type=Path)
    cv.add_argument("--max-videos", type=int)

    a = ap.parse_args(argv)

    if a.cmd == "restore-video":
        from datetime import datetime
        from . import pipeline_video
        from .report import compare
        out = a.out or (config.OUTPUT_ROOT / "review_video" / datetime.now().strftime("%Y%m%d-%H%M%S"))
        print(f"{a.video.name}", flush=True)

        def progress(done: int, total: int) -> None:
            if total and done % 30 == 0:
                print(f"\r   frame {done}/{total} ({100 * done / total:.0f}%)", end="", flush=True)

        rec = pipeline_video.restore_video(a.video, out, preview_only=a.preview_only, encoder=a.encoder,
                                           on_progress=progress, scale=a.scale, content=a.content)
        print()
        out.mkdir(parents=True, exist_ok=True)
        (out / "record.json").write_text(json.dumps(rec.to_dict(), indent=1, ensure_ascii=False, default=str),
                                         encoding="utf-8")
        print(f"report: {compare.video_page(rec.to_dict(), out / 'report.html')}")
        print(json.dumps({"passed": rec.passed, "chosen": rec.chosen, "delivered": rec.delivered,
                          "final_gates": rec.final_gates, "notes": rec.notes, "sheets": rec.sheets},
                         indent=1, ensure_ascii=False, default=str))
        return 0 if rec.passed or a.preview_only else 1

    if a.cmd == "restore":
        from datetime import datetime
        from . import batch
        from .report import compare
        exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}
        files = [p for i in a.inputs for p in (sorted(i.iterdir()) if i.is_dir() else [i]) if p.suffix.lower() in exts]
        out = a.out or (config.OUTPUT_ROOT / "review" / datetime.now().strftime("%Y%m%d-%H%M%S"))
        out.mkdir(parents=True, exist_ok=True)
        from dataclasses import replace as _replace
        policy = _replace(config.DEFAULT_POLICY, deblur=a.deblur)
        records = batch.restore_images(files, out, policy=policy, log=lambda m: print(m, flush=True))
        sheets = {}
        for d in records:
            f = Path(d["source"])
            shown = d["delivered"] or next((x["output"] for x in d["attempts"] if x.get("output")), None)
            if shown:
                sheets[f.name] = compare.sheet(f, Path(shown), out / "sheets" / f"{f.stem}.jpg")
            print(f"{f.name}: {'PASSED -> ' + d['delivered'] if d['passed'] else 'NOT PASSED: ' + '; '.join(d['notes'])}",
                  flush=True)
        (out / "records.json").write_text(json.dumps(records, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
        page = compare.review_page(records, sheets, out / "sheets" / "index.html")
        print(f"\nreview page: {page}")
        return 0

    if a.cmd == "analyze":
        video_ext = {".mp4", ".mkv", ".mov", ".avi", ".m4v", ".mpg", ".mpeg", ".wmv", ".ts", ".mts", ".m2ts"}
        if a.path.suffix.lower() in video_ext:
            from .probe import video as pv
            from . import pipeline_video
            prof = pv.analyze(a.path)
            sheet = pipeline_video.frames_sheet(a.path, config.OUTPUT_ROOT / "analysis" / f"{a.path.stem}_frames.jpg")
            print(json.dumps({**prof.to_dict(), "frames_sheet": str(sheet) if sheet else None},
                             indent=1, ensure_ascii=False, default=str))
            return 0
        else:
            from .probe import image as pi
            prof = pi.analyze(a.path)
        print(json.dumps(prof.to_dict(), indent=1, ensure_ascii=False, default=str))
        return 0

    if a.cmd == "verify":
        from . import guard
        from .guard.video import verify_video
        if a.original.suffix.lower() in {".mp4", ".mkv", ".mov", ".avi", ".m4v", ".ts"}:
            rep = verify_video(a.original, a.result, interpolated=a.interpolated)
        else:
            rep = guard.verify_image(a.original, a.result)
        if a.json:
            print(json.dumps(rep.to_dict(), indent=1, default=str))
        else:
            for g in rep.gates:
                print(f"[{g.status.upper():<4}] {g.name:<15} {g.value!s:<10} (threshold {g.threshold})  {g.detail}")
            print(f"\npassed: {rep.passed}   calibrated thresholds: {rep.calibrated}")
            for s in rep.suggestions:
                print("  ->", s)
        return 0 if rep.passed else 1

    if a.cmd == "upscale":
        return _emit(free_sr.upscale(a.image, a.outdir, model=a.model, scale=a.scale), a.json)
    if a.cmd == "denoise":
        return _emit(free_sr.denoise(a.image, a.outdir), a.json)
    if a.cmd == "video2x":
        return _emit(video2x.run(a.src, a.dst, a.processor, content=a.content, scale=a.scale, model=a.model,
                                 fps_multiplier=a.fps_multiplier, codec=a.codec), a.json)

    if a.cmd == "calibrate":
        from . import calibrate
        res = calibrate.rescore(a.folder) if a.rescore else calibrate.run(a.folder, limit=a.limit, apply=a.apply)
        print(json.dumps({k: v for k, v in res.items() if k != "conflicts"} | {"conflicts": len(res["conflicts"])},
                         indent=1, default=str))
        return 0
    if a.cmd == "calibrate-video":
        from . import calibrate_video
        print(json.dumps(calibrate_video.run(a.folder, max_videos=a.max_videos), indent=1, default=str))
        return 0

    if a.cmd == "doctor":
        report = env.doctor()
        print(json.dumps(report, indent=1, default=str)) if a.json else env.print_report(report)
        return 0 if report["healthy"] else 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
