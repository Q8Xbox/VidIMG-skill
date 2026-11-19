"""Video fidelity guard (PLAN.md section 6, video rows).

Gates: structure (frames, duration, audio), per-frame image gates on sampled frames (identity,
back-projection, colour), temporal flicker (optical-flow warp error, output vs source) and VMAF of
the output scaled back to the source size (system ffmpeg has libvmaf).
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

from .. import media
from . import Gate, GuardReport, SUGGEST, thresholds, verify_image

VIDEO_THRESHOLDS = {
    "frames_tolerance": 2,          # temporal models may drop/hold the last frame or two
    "duration_tolerance_s": 0.15,
    # Calibrated 2026-11-15 on 12 face segments of the person's videos (72 outputs, ground truth):
    # flicker = detail-normalised warp error ADDED by the output (output minus source). Faithful
    # Proteus/Iris added <= +0.006, temporal boiling +0.057..+0.101. The earlier ratio form failed
    # faithful outputs on nearly static shots (source 0.0001 -> ratio 36).
    "flicker_added_max": 0.02,
    # VMAF of the output scaled back to the source measures how much the output CHANGED, which depends
    # on how soft the source is: on synthetic degradations Proteus scored >= 94.5, but on the person's
    # real SD videos faithful Proteus outputs (visually checked) scored 70-91. So VMAF is judged
    # RELATIVE to the first (reference) candidate on the same clip (calibrated 2026-11-15 when that was
    # Topaz Proteus; now it is the first open model): a candidate may be at
    # most vmaf_drop_max below it (balanced accuracy 0.976; damaging rejected 39/41). Without a
    # reference only gross failure is caught (vmaf_floor); flicker/identity/structure gates do the rest.
    # Re-checked 2026-11-16 with the free candidates only (workspace/calibration_video/20261116-153325,
    # 60 outputs): damaging outputs sit only 0.02-2.3 below the reference, so VMAF barely separates them;
    # the "best" fit (0.0) only tracks a label that penalises softness (non-GAN models, identity 0.90-0.93)
    # and would reject faithful outputs. Kept at 4.0. With identity-based labels the whole guard accepts
    # 31/33 faithful and rejects 9/10 damaging (temporal boiling 9/9 via flicker/face gates).
    "vmaf_drop_max": 4.0,
    "vmaf_floor": 60.0,
    "identity_single_frame_floor": 0.70,   # one failing frame above this is a warning, not a failure
    "backprojection_gross": 0.90,          # overall structure fails only below this in video
}


def _ffmpeg() -> str:
    return media.ffmpeg()


def _frames_by_index(path: Path, indices: list[int], folder: Path, tag: str, pre_vf: str | None = None) -> dict[int, Path]:
    """Extract exact frames by decode index in one pass (timestamp seeking misaligns source and
    output, and image2 output duplicates frames of VFR files unless passthrough is used)."""
    expr = "+".join(f"eq(n\\,{i})" for i in sorted(set(indices)))
    pattern = folder / f"{tag}_%04d.png"
    subprocess.run([_ffmpeg(), "-hide_banner", "-loglevel", "error", "-i", str(path), "-vf", (f"{pre_vf}," if pre_vf else "") + f"select='{expr}'",
                    "-fps_mode", "passthrough", "-y", str(pattern)], stdin=subprocess.DEVNULL, timeout=3600)
    files = sorted(folder.glob(f"{tag}_*.png"))
    return dict(zip(sorted(set(indices)), files))


CUT_SHIFT = 6


def _is_cut(a: np.ndarray, b: np.ndarray) -> bool:
    """Hard cut between two consecutive frames: luma histograms stop correlating."""
    ha = cv2.calcHist([cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)], [0], None, [64], [0, 256])
    hb = cv2.calcHist([cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)], [0], None, [64], [0, 256])
    return cv2.compareHist(ha, hb, cv2.HISTCMP_CORREL) < 0.85


def _cut_near(frames: dict[int, Path], s: int) -> bool:
    """A cut between any of the frames s-1, s, s+1, s+2."""
    imgs = [cv2.imread(str(frames[i])) if i in frames else None for i in (s - 1, s, s + 1, s + 2)]
    return any(x is not None and y is not None and _is_cut(x, y) for x, y in zip(imgs, imgs[1:]))


# "max" (since 2026-11-16): the output's warp residual is normalised by the larger of its own and the source's detail,
# so REMOVING grain/noise (less detail) is not read as flicker. "own" (each side by its own detail, the 2026-11-15
# form) failed a grainy night-sky cartoon shot at +0.031 while the output's frame-to-frame change was LOWER than the
# source's ("max": -0.008). Replayed on every labelled output (72 + 60 real-footage, 108 cartoon): no decision changed,
# all 36 boiling outputs still caught (lowest 0.052).
FLICKER_NORM = "max"
ASPECT_TOL = 0.005        # display-aspect drift allowed (even-size rounding stays far below)


def _warp_parts(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """(warp residual, spatial detail) of the pair a -> b."""
    ga, gb = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM).calc(ga, gb, None)
    h, w = ga.shape
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    warped = cv2.remap(gb, gx + flow[..., 0], gy + flow[..., 1], cv2.INTER_LINEAR)
    residual = float(np.mean(np.abs(warped.astype(np.float32) - ga.astype(np.float32))))
    f = ga.astype(np.float32)
    detail = float(np.mean(np.abs(cv2.Sobel(f, cv2.CV_32F, 1, 0)) + np.abs(cv2.Sobel(f, cv2.CV_32F, 0, 1))))
    return residual, detail


def _warp_error(a: np.ndarray, b: np.ndarray) -> float:
    """Temporal inconsistency normalised by spatial detail: mean |warp(b)->a - a| / mean |grad a|.
    A sharper (restored) frame has larger raw residuals at edges; normalising keeps genuine
    sharpening from reading as flicker while real shimmer/boiling still raises the value."""
    ga, gb = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM).calc(ga, gb, None)
    h, w = ga.shape
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    warped = cv2.remap(gb, gx + flow[..., 0], gy + flow[..., 1], cv2.INTER_LINEAR)
    residual = float(np.mean(np.abs(warped.astype(np.float32) - ga.astype(np.float32))))
    f = ga.astype(np.float32)
    detail = float(np.mean(np.abs(cv2.Sobel(f, cv2.CV_32F, 1, 0)) + np.abs(cv2.Sobel(f, cv2.CV_32F, 0, 1))))
    return residual / max(1e-3, detail)


def _vmaf(src: Path, dst: Path, w: int, h: int, seconds: float, src_vf: str | None = None) -> float | None:
    # frames are paired BY INDEX: libvmaf's framesync pairs by timestamp, which mispairs neighbours
    # when the containers store time differently (NUT exact vs MP4/MKV rounded): a QTGMC 60p pre-pass
    # vs its HEVC result scored 59.8 by timestamp while its previews scored 94.5-95 (2026-11-15).
    # Calibration pairs were MKV vs MKV (same grid, already aligned), so calibrated values are unchanged.
    idx = "settb=1/30,setpts=N"
    graph = (f"[1:v]scale={w}:{h}:flags=bicubic,setsar=1,format=yuv420p,{idx}[d];"
             f"[0:v]{src_vf + ',' if src_vf else ''}setsar=1,format=yuv420p,{idx}[s];[d][s]libvmaf=n_threads=8")
    # bounded by FRAME COUNT on both sides (a time limit trusts timestamps: an output with all-zero timestamps read
    # every frame against 20 s of source and scored 25.8 while its PSNR was 32-37 dB)
    n = max(1, round(seconds * (media.probe(src)["fps"] or 25)))
    graph = graph.replace(f"{idx}[d]", f"trim=end_frame={n},{idx}[d]").replace(f"{idx}[s]", f"trim=end_frame={n},{idx}[s]")
    out = subprocess.run([_ffmpeg(), "-hide_banner", "-nostats", "-i", str(src),
                          "-i", str(dst), "-lavfi", graph, "-f", "null", "-"],
                         capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=3600, errors="replace")
    found = re.findall(r"VMAF score:\s*([0-9.]+)", out.stderr)
    return float(found[-1]) if found else None


def verify_video(src: Path, dst: Path, *, samples: int = 6, vmaf_seconds: float = 20.0,
                 interpolated: bool = False, reference_vmaf: float | None = None,
                 src_vf: str | None = None, src_rate: float = 1.0, drawn_faces: bool = False) -> GuardReport:
    """reference_vmaf: VMAF of the reference candidate's output on the same clip (from a previous
    verify_video call); when given, this output may score at most vmaf_drop_max below it.
    src_vf: filter applied to the source before comparison. Interlaced sources are compared after a
    conventional deinterlace ("bwdif=mode=send_frame"): the raw combed frames are not a valid
    reference for a deinterlaced output (a blend of the combed original otherwise "wins").
    src_rate: output frames per source frame when src_vf changes the rate (2 for a double-rate
    deinterlace checked against "bwdif=mode=send_field"); frame indices then refer to the filtered source."""
    src, dst = Path(src), Path(dst)
    vt = VIDEO_THRESHOLDS
    a, b = media.probe(src, exact=True), media.probe(dst, exact=True)
    gates: list[Gate] = []

    # structure
    src_frames = round(a["frames"] * src_rate)
    if not interpolated:
        diff = abs(b["frames"] - src_frames)
        gates.append(Gate("frames", "pass" if diff <= vt["frames_tolerance"] * max(1.0, src_rate) else "fail",
                          b["frames"], src_frames, f"output {b['frames']} vs source {src_frames}"))
    dd = abs(b["duration"] - a["duration"])
    gates.append(Gate("duration", "pass" if dd <= vt["duration_tolerance_s"] else "fail", round(b["duration"], 3),
                      round(a["duration"], 3), f"difference {dd:.3f}s"))
    gates.append(Gate("audio", "pass" if b["audio_streams"] >= a["audio_streams"] else "fail", b["audio_streams"],
                      a["audio_streams"], "audio streams kept"))
    # timestamps must increase (Video2X wrote all-zero timestamps for an MPEG-4 Part 2 source; frame count and
    # duration still looked right, the file did not play right)
    pts = subprocess.run([media.ffprobe(), "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "packet=pts_time", "-read_intervals", "%+#40", "-of", "csv=p=0", str(dst)],
                         capture_output=True, text=True, stdin=subprocess.DEVNULL).stdout.split()
    vals = sorted(float(x) for x in pts if x not in ("N/A", ""))
    # and in DISPLAY order (decoded frames) they must rise strictly: stamps assigned in packet order to a stream with
    # B-frames were all distinct yet shuffled the frames on playback (Sample03, before --max-b-frames 0)
    fts = subprocess.run([media.ffprobe(), "-v", "error", "-select_streams", "v:0", "-show_entries", "frame=pts_time",
                          "-read_intervals", "%+#40", "-of", "csv=p=0", str(dst)],
                         capture_output=True, text=True, stdin=subprocess.DEVNULL).stdout.split()
    fvals = [float(x) for x in fts if x not in ("N/A", "")]
    if len(vals) >= 3:
        rising = all(b > a for a, b in zip(fvals, fvals[1:])) if len(fvals) >= 3 else True
        ok = len(set(vals)) == len(vals) and rising
        gates.append(Gate("timestamps", "pass" if ok else "fail", len(set(vals)), len(vals),
                          f"{len(set(vals))} distinct of the first {len(vals)} video timestamps; "
                          f"display order {'rising' if rising else 'NOT rising (frames would play shuffled)'}"))
    # lip sync: the video/audio start relation must survive (Video2X 6.4.0 re-timed outputs by 0.08-0.91 s and
    # both engines restart the video clock at 0; frame and duration gates cannot see it)
    oa, ob = media.av_offset(a), media.av_offset(b)
    if oa is not None and ob is not None and not interpolated:
        tol = max(0.02, 0.5 / (b["fps"] or 25))
        gates.append(Gate("av_sync", "pass" if abs(oa - ob) <= tol else "fail", round(ob - oa, 3), round(tol, 3),
                          f"video-audio start offset {ob:+.3f} s vs source {oa:+.3f} s"))
    # geometry: the picture must keep its on-screen shape (storage size x pixel aspect). A lost pixel aspect
    # squeezed every face by 6.7 % on PAL 16:15 sources and nothing else noticed (frames are compared resized)
    da, db = media.display_aspect(a), media.display_aspect(b)
    if da and db:
        rel = db / da - 1
        gates.append(Gate("aspect", "pass" if abs(rel) <= ASPECT_TOL else "fail", round(rel, 4), ASPECT_TOL,
                          f"display aspect {db:.4f} vs source {da:.4f} (pixel aspect {b['sar']} vs {a['sar']})"))

    # per-frame image gates and temporal flicker on sampled frames
    n = min(src_frames, b["frames"]) if not interpolated else src_frames
    starts = [int(n * (i + 0.5) / samples) for i in range(samples)]
    starts = [s for s in starts if s + 1 < n]
    with tempfile.TemporaryDirectory(prefix="vidimg_vg_") as tmp:
        t = Path(tmp)
        # Each sample needs a clean pair (s, s+1) away from scene cuts: at a cut the source and a
        # faithful output legitimately differ (an interlaced cut frame holds one field from each
        # scene; a temporal model blends across it). Candidates s and s+CUT_SHIFT are extracted
        # with a neighbour on each side, and the first one with no cut nearby is used.
        options = {s: [o for o in (s, s + CUT_SHIFT) if o - 1 >= 0 and o + 2 < n] for s in starts}
        want = sorted({i for opts in options.values() for o in opts for i in (o - 1, o, o + 1, o + 2)})
        sf_map = _frames_by_index(src, want, t, "s", src_vf)
        df_map = _frames_by_index(dst, want, t, "d") if not interpolated else {}
        worst: dict[str, Gate] = {}
        fail_frames: dict[str, int] = {}
        se, de = [], []
        chosen, skipped_cuts = [], 0
        for s, opts in options.items():
            ok = next((o for o in opts if not _cut_near(sf_map, o)), None)
            skipped_cuts += ok != (opts[0] if opts else None)
            if ok is not None:
                chosen.append(ok)
        pairs = [(sf_map.get(s), sf_map.get(s + 1), df_map.get(s), df_map.get(s + 1)) for s in chosen]
        sf, df = [], []
        for s0p, s1p, d0p, d1p in pairs:
            if s0p and s1p and d0p and d1p:
                sf += [s0p, s1p]
                df += [d0p, d1p]
        for i in range(0, min(len(sf), len(df)) - 1, 2):
            # Interlaced sources: Dione reconstructs moving frames from both fields with motion
            # compensation; no simple reference (bwdif first field, second field or field mode)
            # reproduces that (a clean, faithful moving face scored 0.905-0.91 against all three
            # while the other frames scored 0.97-0.98), so face structure is advisory there.
            rep = verify_image(sf[i], df[i], face_structure_advisory=src_vf is not None, drawn_faces=drawn_faces)
            for g in rep.gates:
                if g.status == "fail":
                    fail_frames[g.name] = fail_frames.get(g.name, 0) + 1
                prev = worst.get(g.name)
                rank = {"fail": 3, "warn": 2, "pass": 1, "n/a": 0}
                if prev is None or rank[g.status] > rank[prev.status] or (
                        g.status == prev.status and g.value is not None and prev.value is not None and
                        (g.value < prev.value if g.name in ("identity", "backprojection") else g.value > prev.value)):
                    worst[g.name] = g
            s0, s1 = cv2.imread(str(sf[i])), cv2.imread(str(sf[i + 1]))
            d0, d1 = cv2.imread(str(df[i])), cv2.imread(str(df[i + 1]))
            if s0 is not None and d0 is not None and s1 is not None and d1 is not None and not interpolated:
                size = (s0.shape[1], s0.shape[0])
                d0, d1 = (cv2.resize(x, size, interpolation=cv2.INTER_AREA) for x in (d0, d1))
                rs, ds_ = _warp_parts(s0, s1)
                rd, dd = _warp_parts(d0, d1)
                se.append(rs / max(1e-3, ds_))
                de.append(rd / max(1e-3, max(dd, ds_) if FLICKER_NORM == "max" else dd))
        for g in worst.values():
            g.detail = f"worst of {samples} sampled frames ({fail_frames.get(g.name, 0)} failing): {g.detail}"
            # Video-specific relaxations, each backed by the person's footage and the calibration:
            # - identity: a single sampled frame where a face turns to profile or into shadow scores low
            #   (a dark profile face in a cinema: 0.77, visually faithful); model damage persists over
            #   frames, so one failing frame is a warning unless it falls below the hard floor
            # - overall structure: it rejected no damaging output that other gates missed (40/42
            #   either way), while faithful denoising of dark compressed footage scored 0.96; it fails
            #   only for gross change
            if g.status == "fail" and g.name == "identity" and fail_frames.get("identity", 0) < 2 \
                    and g.value is not None and g.value >= vt["identity_single_frame_floor"]:
                g.status = "warn"
            if g.status == "fail" and g.name == "backprojection" and g.value is not None \
                    and g.value >= vt["backprojection_gross"]:
                g.status = "warn"
            gates.append(g)
    if se and de:
        s_med, d_med = float(np.median(se)), float(np.median(de))
        added = d_med - s_med
        gates.append(Gate("flicker", "pass" if added <= vt["flicker_added_max"] else "fail", round(added, 4),
                          vt["flicker_added_max"],
                          f"warp error added by the output (source {s_med:.4f}, output {d_med:.4f})"))

    if not interpolated:
        v = _vmaf(src, dst, a["width"], a["height"], min(vmaf_seconds, a["duration"]), src_vf)
        span = f"first {min(vmaf_seconds, a['duration']):.0f}s, output scaled to source"
        if v is None:
            gates.append(Gate("vmaf", "n/a", None, None, span))
        elif reference_vmaf is not None:
            need = reference_vmaf - vt["vmaf_drop_max"]
            gates.append(Gate("vmaf", "pass" if v >= need else "fail", round(v, 2), round(need, 2),
                              f"{span}; at most {vt['vmaf_drop_max']} below the reference candidate ({reference_vmaf:.2f})"))
        else:
            gates.append(Gate("vmaf", "pass" if v >= vt["vmaf_floor"] else "fail", round(v, 2), vt["vmaf_floor"],
                              f"{span}; no reference: gross-failure floor only"))

    failed = [g.name for g in gates if g.status == "fail"]
    extra = {"flicker": "added temporal flicker: prefer a non-GAN model (RealESRNet/BSRNet) or no enhancement here",
             "av_sync": "video/audio start offset changed: remux with media.fix_av_offset (no re-encode needed)",
             "aspect": "display aspect changed (pixel aspect lost): remux with media.fix_aspect (no re-encode needed)",
             "vmaf": "the output drifted from the source: lower enhancement strength or raise blend",
             "frames": "frame count changed: check interpolation/duplicate removal settings",
             "audio": "audio was dropped: map 0:a and copy it"}
    sugg = [SUGGEST.get(n) or extra.get(n) for n in failed]
    return GuardReport(not failed, gates, [s for s in sugg if s], bool(thresholds().get("calibrated")))


# ----------------------------------------------------------------------------- un-blending (sRestore)
DEBLEND_COPY_PSNR = 45.0        # dB on 8-bit luma at analysis size: a copied frame scores 48-108 (measured)
DEBLEND_COPY_SHARE = 0.98       # share of sampled output frames that must be copies of input frames
DEBLEND_SEARCH = 12             # input frames searched on each side of the time-aligned index (drift seen: -6..+11)


def verify_deblend(src: Path, dst: Path, *, in_fps, out_fps, in_share: float, windows: int = 3,
                   seconds: float = 4.0) -> GuardReport:
    """sRestore output against its input. Structure as for any video (duration, audio, lip sync, shape,
    frame count at the new rate); then the two things that make un-blending safe:
    * selection: every sampled output frame is an exact copy of an input frame near the same time
      (sRestore only chooses pictures; nothing is synthesised, so no identity or structure can change);
    * blends: the share of blended frames must actually drop, otherwise the step is not worth keeping."""
    from fractions import Fraction

    from ..probe import blend

    src, dst = Path(src), Path(dst)
    a, b = media.probe(src, exact=True), media.probe(dst, exact=True)
    fi, fo = Fraction(in_fps), Fraction(out_fps)
    gates: list[Gate] = []
    want = int(a["frames"] * fo / fi)
    diff = abs(b["frames"] - want)
    gates.append(Gate("frames", "pass" if diff <= max(2, want * 0.002) else "fail", b["frames"], want,
                      f"output {b['frames']} vs {want} expected at {float(fo):.3f} fps"))
    dd = abs(b["duration"] - a["duration"])
    gates.append(Gate("duration", "pass" if dd <= max(VIDEO_THRESHOLDS["duration_tolerance_s"], 2 / float(fo)) else "fail",
                      round(b["duration"], 3), round(a["duration"], 3), f"difference {dd:.3f}s"))
    gates.append(Gate("audio", "pass" if b["audio_streams"] >= a["audio_streams"] else "fail", b["audio_streams"],
                      a["audio_streams"], "audio streams kept"))
    oa, ob = media.av_offset(a), media.av_offset(b)
    if oa is not None and ob is not None:
        tol = max(0.02, 0.5 / float(fo))
        gates.append(Gate("av_sync", "pass" if abs(oa - ob) <= tol else "fail", round(ob - oa, 3), round(tol, 3),
                          f"video-audio start offset {ob:+.3f} s vs source {oa:+.3f} s"))
    da, db = media.display_aspect(a), media.display_aspect(b)
    if da and db:
        rel = db / da - 1
        gates.append(Gate("aspect", "pass" if abs(rel) <= ASPECT_TOL else "fail", round(rel, 4), ASPECT_TOL,
                          f"display aspect {db:.4f} vs source {da:.4f}"))

    copies = judged = 0
    worst = None
    span = max(1.0, min(seconds, a["duration"] / max(1, windows) - 1))
    for k in range(windows):
        t = max(0.0, a["duration"] * (k + 0.5) / windows - span / 2)
        pre = min(1.0, t)
        outs = list(blend.read_gray(dst, t, span, width=180))
        ins = list(blend.read_gray(src, t - pre, span + 2.0, width=180))
        if not outs or not ins:
            continue
        for i, o in enumerate(outs):
            j = round((i / fo + pre) * fi)
            lo, hi = max(0, j - DEBLEND_SEARCH), min(len(ins), j + DEBLEND_SEARCH + 1)
            if lo >= hi:
                continue
            o32 = o.astype(np.float32)
            mse = min(float(np.mean((o32 - ins[m]) ** 2)) for m in range(lo, hi))
            psnr = 10 * np.log10(255 ** 2 / max(mse, 1e-6))
            judged += 1
            copies += psnr >= DEBLEND_COPY_PSNR
            worst = psnr if worst is None else min(worst, psnr)
    share = copies / judged if judged else 0.0
    gates.append(Gate("selection", "pass" if judged and share >= DEBLEND_COPY_SHARE else "fail", round(share, 4),
                      DEBLEND_COPY_SHARE, f"{copies}/{judged} sampled output frames are copies of input frames "
                      f"(worst {worst if worst is None else round(worst, 1)} dB)"))
    out = blend.analyze(dst)
    limit = max(0.05, 0.6 * in_share)
    gates.append(Gate("blends", "pass" if out.share <= limit else "fail", out.share, round(limit, 3),
                      f"blended frames {out.share:.1%} after vs {in_share:.1%} before"))
    failed = [g.name for g in gates if g.status == "fail"]
    sugg = {"selection": "sRestore output is not made of input frames: keep the blended input",
            "blends": "sRestore did not remove the blends at this rate: keep the blended input"}
    return GuardReport(not failed, gates, [sugg[n] for n in failed if n in sugg], False)


# ----------------------------------------------------------------------------- chroma shift correction
CHROMA_RESIDUAL_MAX = 0.6       # luma px left beyond the clean baseline after the correction (measured 0.0-0.33)
LUMA_UNCHANGED_PSNR = 60.0      # dB: a chroma-only pass must leave the luma bit-identical (measured 138 dB)


CHROMA_UNTOUCHED_SLACK = 0.3   # luma px a component that was not corrected may drift in the re-measurement


def verify_chroma(src: Path, dst: Path, *, moved: dict, before: dict, luma_must_match: bool,
                  frames: int = 24) -> GuardReport:
    """The chroma-shift correction did what it should and nothing else. Measured again on the output: every
    corrected component now lines up with the line art, and the components left alone (the frames did not
    agree on them) are no worse; when no field handling ran in the same pass, the luma is untouched.
    moved: {"u": [dx, dy], "v": [...]} applied (luma px); before: the profile's excess over the baseline."""
    from ..probe import chroma

    gates: list[Gate] = []
    after = chroma.analyze(Path(dst)).excess()
    worst, detail = 0.0, []
    for k in ("u", "v"):
        for i in (0, 1):
            a, b = abs(after[k][i]), abs(before[k][i])
            bad = a if moved[k][i] else max(0.0, a - b - CHROMA_UNTOUCHED_SLACK)
            worst = max(worst, bad)
            detail.append(f"{'Cb' if k == 'u' else 'Cr'} {'xy'[i]} {before[k][i]:+.2f} -> {after[k][i]:+.2f}"
                          + ("" if moved[k][i] else " (left)"))
    gates.append(Gate("chroma_aligned", "pass" if worst <= CHROMA_RESIDUAL_MAX else "fail", round(worst, 2),
                      CHROMA_RESIDUAL_MAX, "px from the lines, before -> after: " + "; ".join(detail)))
    if luma_must_match:
        a, b = media.probe(Path(src)), media.probe(Path(dst))
        w, h = a["width"], a["height"]

        def luma(path: Path) -> np.ndarray:
            raw = subprocess.run([media.ffmpeg(), "-hide_banner", "-loglevel", "error", "-i", str(path), "-map",
                                  "0:v:0", "-frames:v", str(frames), "-vf", "format=yuv420p16le,extractplanes=y",
                                  "-pix_fmt", "gray16le", "-f", "rawvideo", "-"], capture_output=True).stdout
            n = len(raw) // (w * h * 2)
            return (np.frombuffer(raw[: n * w * h * 2], "<u2").reshape(n, h, w) >> 8).astype(np.float32)

        ya, yb = (luma(Path(src)), luma(Path(dst))) if (a["width"], a["height"]) == (b["width"], b["height"]) \
            else (np.zeros(1), np.ones(1))
        n = min(len(ya), len(yb))
        mse = float(np.mean((ya[:n] - yb[:n]) ** 2)) if n else 1e9
        ps = 10 * np.log10(255 ** 2 / max(mse, 1e-9))
        gates.append(Gate("luma_unchanged", "pass" if ps >= LUMA_UNCHANGED_PSNR else "fail", round(ps, 1),
                          LUMA_UNCHANGED_PSNR, f"luma of the first {n} frames vs the source: {ps:.1f} dB"))
    failed = [g.name for g in gates if g.status == "fail"]
    return GuardReport(not failed, gates, ["chroma correction did not line the colour up: run without it"]
                       if failed else [], False)
