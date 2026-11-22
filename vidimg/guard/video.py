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
from .. import imgfile  # noqa: E402

from .. import media, texture
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
    media.run_quiet([_ffmpeg(), "-hide_banner", "-loglevel", "error", "-i", str(path), "-vf", (f"{pre_vf}," if pre_vf else "") + f"select='{expr}'",
                     "-fps_mode", "passthrough", "-y", str(pattern)], timeout=3600)
    files = sorted(folder.glob(f"{tag}_*.png"))
    return dict(zip(sorted(set(indices)), files))


CUT_SHIFT = 6
INTERP_SEARCH = 60      # source frames searched for one that survives unchanged in an interpolated output


def _is_cut(a: np.ndarray, b: np.ndarray) -> bool:
    """Hard cut between two consecutive frames: luma histograms stop correlating."""
    ha = cv2.calcHist([cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)], [0], None, [64], [0, 256])
    hb = cv2.calcHist([cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)], [0], None, [64], [0, 256])
    return cv2.compareHist(ha, hb, cv2.HISTCMP_CORREL) < 0.85


def _cut_near(frames: dict[int, Path], s: int) -> bool:
    """A cut between any of the frames s-1, s, s+1, s+2."""
    imgs = [imgfile.imread(str(frames[i])) if i in frames else None for i in (s - 1, s, s + 1, s + 2)]
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
                         capture_output=True, text=True, encoding="utf-8", stdin=subprocess.DEVNULL, timeout=3600, errors="replace")
    found = re.findall(r"VMAF score:\s*([0-9.]+)", out.stderr)
    return float(found[-1]) if found else None


def _vmaf_spread(src: Path, dst: Path, w: int, h: int, windows: list[tuple[int, int]],
                 src_vf: str | None = None) -> list[float] | None:
    """VMAF of several frame windows [(first, last), ...] spread over the video, one mean per window, in one
    decode of each file. Frames are paired by index as in _vmaf; the per-frame scores come from libvmaf's log."""
    import json

    sel = "+".join(f"between(n\\,{a}\\,{b})" for a, b in windows)
    idx = "settb=1/30,setpts=N"
    graph = (f"[1:v]select='{sel}',scale={w}:{h}:flags=bicubic,setsar=1,format=yuv420p,{idx}[d];"
             f"[0:v]{src_vf + ',' if src_vf else ''}select='{sel}',setsar=1,format=yuv420p,{idx}[s];"
             "[d][s]libvmaf=n_threads=8:log_fmt=json:log_path=vmaf.json")
    with tempfile.TemporaryDirectory(prefix="vidimg_vmaf_") as tmp:      # relative log path: no drive-colon escaping
        subprocess.run([_ffmpeg(), "-hide_banner", "-nostats", "-i", str(Path(src).resolve()), "-i",
                        str(Path(dst).resolve()), "-lavfi", graph, "-fps_mode", "passthrough", "-f", "null", "-"],
                       capture_output=True, text=True, encoding="utf-8", stdin=subprocess.DEVNULL, timeout=4 * 3600, errors="replace",
                       cwd=tmp)
        log = Path(tmp) / "vmaf.json"
        try:
            scores = [f["metrics"]["vmaf"] for f in json.loads(log.read_text(encoding="utf-8"))["frames"]]
        except (OSError, ValueError, KeyError):
            return None
    out, pos = [], 0
    for a, b in windows:
        part = scores[pos: pos + (b - a + 1)]
        pos += b - a + 1
        if len(part) < (b - a + 1) // 2:                 # a window that mostly did not decode is no evidence
            return None
        out.append(float(np.mean(part)))
    return out


SAMPLE_EVERY_S = 60.0           # one sampled frame pair per minute of video ...
SAMPLES_MIN, SAMPLES_MAX = 6, 40    # ... but never fewer than the 6 the calibration used, nor more than 40
VMAF_WINDOW_EVERY_S = 180.0     # long videos: one VMAF window per three minutes (3 to 8 windows of 10 s)
FLICKER_PAIR_FACTOR = 2.0       # a sampled pair "boils" when it adds more than this many times the limit
FLICKER_PAIR_SHARE = 0.25       # this share of boiling pairs fails the video whatever the median says


TIMING_MIN_S = 0.04             # a frame may be shown this much earlier or later than in the source ...
TIMING_FRAMES = 1.5             # ... or this many frame durations, whichever is larger


def _frame_times(path: Path) -> list[float]:
    """Presentation time of every video frame, in display order (packets only: nothing is decoded)."""
    out = subprocess.run([media.ffprobe(), "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts_time",
                          "-of", "csv=p=0", str(path)], capture_output=True, text=True, encoding="utf-8",
                         errors="replace", stdin=subprocess.DEVNULL).stdout.split()
    vals = []
    for x in out:
        try:
            vals.append(float(x.split(",")[0]))
        except ValueError:
            pass
    return sorted(vals)


def plan_samples(duration: float) -> int:
    """Sampled frame pairs for a video of this length: 6 for clips (as calibrated), one per minute beyond that."""
    return int(max(SAMPLES_MIN, min(SAMPLES_MAX, np.ceil(duration / SAMPLE_EVERY_S))))


def verify_video(src: Path, dst: Path, *, samples: int | None = None, vmaf_seconds: float = 20.0,
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
                         capture_output=True, text=True, encoding="utf-8", errors="replace", stdin=subprocess.DEVNULL).stdout.split()
    # first field only: ffprobe adds empty columns for frames with side data (x265 grain-tune output: "0.000000,")
    pts = [x.split(",")[0] for x in pts]
    vals = sorted(float(x) for x in pts if x not in ("N/A", ""))
    # and in DISPLAY order (decoded frames) they must rise strictly: stamps assigned in packet order to a stream with
    # B-frames were all distinct yet shuffled the frames on playback (Sample03, before --max-b-frames 0)
    fts = subprocess.run([media.ffprobe(), "-v", "error", "-select_streams", "v:0", "-show_entries", "frame=pts_time",
                          "-read_intervals", "%+#40", "-of", "csv=p=0", str(dst)],
                         capture_output=True, text=True, encoding="utf-8", errors="replace", stdin=subprocess.DEVNULL).stdout.split()
    fvals = [float(f) for f in (x.split(",")[0] for x in fts) if f not in ("N/A", "")]
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
    # timing: frame i must be shown when the source showed it. A variable-rate source (phone clips) written at a
    # constant rate keeps its frame count, duration and start, so nothing above notices, while picture and sound
    # drift apart in the middle. Advisory: the limit is not calibrated on footage
    if not interpolated and src_vf is None and src_rate == 1.0:
        ta, tb = _frame_times(src), _frame_times(dst)
        m = min(len(ta), len(tb))
        if m >= 2:
            drift = max(abs((y - tb[0]) - (x - ta[0])) for x, y in zip(ta[:m], tb[:m]))
            limit = max(TIMING_MIN_S, TIMING_FRAMES / (a["fps"] or 25))
            gates.append(Gate("timing", "pass" if drift <= limit else "warn", round(drift, 3), round(limit, 3),
                              f"largest difference between a frame's time in the output and in the source: {drift:.3f} s"
                              + ("" if drift <= limit else " (variable frame rate written as constant? sound drifts)")))
    # geometry: the picture must keep its on-screen shape (storage size x pixel aspect). A lost pixel aspect
    # squeezed every face by 6.7 % on PAL 16:15 sources and nothing else noticed (frames are compared resized)
    da, db = media.display_aspect(a), media.display_aspect(b)
    if da and db:
        rel = db / da - 1
        gates.append(Gate("aspect", "pass" if abs(rel) <= ASPECT_TOL else "fail", round(rel, 4), ASPECT_TOL,
                          f"display aspect {db:.4f} vs source {da:.4f} (pixel aspect {b['sar']} vs {a['sar']})"))

    # per-frame image gates and temporal flicker on sampled frames
    n = min(src_frames, b["frames"]) if not interpolated else src_frames
    # six pairs cover a preview clip; on a whole episode they left minutes between samples, so a damaged stretch
    # was never decoded: the count grows with the length
    samples = samples or plan_samples(a["duration"])
    starts =[int(n * (i + 0.5) / samples) for i in range(samples)]
    starts = [s for s in starts if s + 1 < n]
    ratio = b["frames"] / max(1, src_frames)
    if interpolated:
        # RIFE at a whole multiplier m writes m*n - (m-1) frames: nothing is created after the last one. The ratio
        # is then just under m (358 / 120 = 2.983) while source frame s sits at output frame s*m exactly. Read as a
        # ratio, almost no frame "fell on a whole number" and the one that did was a created frame, compared with
        # an original as if it were kept (face structure 0.23 on a correct result, 2026-11-21)
        whole = round(ratio)
        short = src_frames * whole - b["frames"]
        if whole >= 2 and 0 <= short <= whole * 3:          # up to two source frames lost at the very end as well
            ratio = float(whole)
            gates.append(Gate("frames", "pass" if short <= whole else "fail", b["frames"], src_frames * whole - (whole - 1),
                              f"x{whole}: {b['frames']} frames for {src_frames} originals"
                              + ("" if short <= whole else f"; {short - (whole - 1)} missing at the end")))
    if interpolated:
        # frame interpolation keeps the source's frames and adds new ones between them: source frame s is output
        # frame s x ratio wherever that is a whole number (every frame at 2x, every second one at 24 -> 60).
        # Those are compared like any other result; before, an interpolated file passed with no picture gate
        def kept(s: int) -> int | None:
            return next((c for c in range(s, min(n - 2, s + INTERP_SEARCH)) if abs(c * ratio - round(c * ratio)) < 0.02),
                        None)
        starts = sorted({k for k in map(kept, starts) if k is not None})
    with tempfile.TemporaryDirectory(prefix="vidimg_vg_") as tmp:
        t = Path(tmp)
        # Each sample needs a clean pair (s, s+1) away from scene cuts: at a cut the source and a
        # faithful output legitimately differ (an interlaced cut frame holds one field from each
        # scene; a temporal model blends across it). Candidates s and s+CUT_SHIFT are extracted
        # with a neighbour on each side, and the first one with no cut nearby is used.
        options = {s: [o for o in ((s,) if interpolated else (s, s + CUT_SHIFT)) if o - 1 >= 0 and o + 2 < n]
                   for s in starts}
        want = sorted({i for opts in options.values() for o in opts for i in (o - 1, o, o + 1, o + 2)})
        sf_map = _frames_by_index(src, want, t, "s", src_vf)
        if interpolated:
            picks = sorted({round(o * ratio) for opts in options.values() for o in opts})
            got = _frames_by_index(dst, picks, t, "d")
            # the pair's second frame is not needed (no flicker measure between interpolated frames)
            df_map = {o: got[round(o * ratio)] for opts in options.values() for o in opts if round(o * ratio) in got}
            df_map |= {o + 1: p for o, p in list(df_map.items())}
        else:
            df_map = _frames_by_index(dst, want, t, "d")
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
        flat_up: list[float] = []
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
            s0, s1 = imgfile.imread(str(sf[i])), imgfile.imread(str(sf[i + 1]))
            d0, d1 = imgfile.imread(str(df[i])), imgfile.imread(str(df[i + 1]))
            if s0 is not None and d0 is not None and not interpolated:
                fr, fo = texture.flat_shares(cv2.cvtColor(s0, cv2.COLOR_BGR2GRAY), cv2.cvtColor(d0, cv2.COLOR_BGR2GRAY))
                flat_up.append(fo - fr)
            if s0 is not None and d0 is not None and s1 is not None and d1 is not None and not interpolated:
                size = (s0.shape[1], s0.shape[0])
                d0, d1 = (cv2.resize(x, size, interpolation=cv2.INTER_AREA) for x in (d0, d1))
                rs, ds_ = _warp_parts(s0, s1)
                rd, dd = _warp_parts(d0, d1)
                se.append(rs / max(1e-3, ds_))
                de.append(rd / max(1e-3, max(dd, ds_) if FLICKER_NORM == "max" else dd))
        for g in worst.values():
            g.detail = f"worst of {len(sf) // 2} sampled frames ({fail_frames.get(g.name, 0)} failing): {g.detail}"
            # Video-specific relaxations, each backed by the person's footage and the calibration:
            # - identity: a single sampled frame where a face turns to profile or into shadow scores low
            #   (a dark profile face in a cinema: 0.77, visually faithful); model damage persists over
            #   frames, so one failing frame is a warning unless it falls below the hard floor
            # - overall structure: it rejected no damaging output that other gates missed (40/42
            #   either way), while faithful denoising of dark compressed footage scored 0.96; it fails
            #   only for gross change
            #   (one in six sampled frames, the calibrated proportion, whatever the number of samples)
            if g.status == "fail" and g.name == "identity" \
                    and fail_frames.get("identity", 0) <= max(1, (len(sf) // 2) // 6) \
                    and g.value is not None and g.value >= vt["identity_single_frame_floor"]:
                g.status = "warn"
            if g.status == "fail" and g.name == "backprojection" and g.value is not None \
                    and g.value >= vt["backprojection_gross"]:
                g.status = "warn"
            gates.append(g)
    # the picture gates above only exist when frames were actually compared: without this gate a video whose
    # frames could not be extracted passed on frame count, duration and audio alone
    compared, need = len(sf) // 2, max(1, (len(starts) + 1) // 2)
    gates.append(Gate("coverage", "pass" if compared >= need else "fail", compared, need,
                      f"{compared} of {len(starts)} planned frame pairs compared"
                      + (f" ({skipped_cuts} moved or dropped at scene cuts)" if skipped_cuts else "")
                      + (f"; interpolated x{ratio:.3g}: only frames kept from the source" if interpolated else "")))
    if flat_up:
        # over-smoothing ("denoised to death"): flat area between the lines grew vs a bicubic upscale. Advisory:
        # removing noise is legitimate, but the person should see when texture went with it
        up = float(np.median(flat_up))
        gates.append(Gate("texture", "warn" if up > texture.FLAT_INCREASE_WARN else "pass", round(up, 3),
                          texture.FLAT_INCREASE_WARN,
                          f"flat area between the lines {'+' if up >= 0 else ''}{up:.0%} vs a plain upscale"))
    if se and de:
        s_med, d_med = float(np.median(se)), float(np.median(de))
        added = d_med - s_med
        # the median hides a stretch: boiling in 2 of 6 samples left it untouched. Pairs far over the limit are
        # counted as well; one such pair is a warning (a single hard motion can do it), a quarter of them fails
        hot = sum(1 for x, y in zip(se, de) if y - x > FLICKER_PAIR_FACTOR * vt["flicker_added_max"])
        status = "pass" if added <= vt["flicker_added_max"] else "fail"
        if status == "pass" and hot:
            status = "fail" if hot >= 2 and hot >= FLICKER_PAIR_SHARE * len(se) else "warn"
        gates.append(Gate("flicker", status, round(added, 4), vt["flicker_added_max"],
                          f"warp error added by the output (source {s_med:.4f}, output {d_med:.4f}); "
                          f"{hot} of {len(se)} sampled pairs far over the limit"))

    if not interpolated:
        worst_win = None
        if a["duration"] > 2 * vmaf_seconds and n > 0:
            # a long video: windows spread over its length instead of the first seconds only
            k = int(max(3, min(8, np.ceil(a["duration"] / VMAF_WINDOW_EVERY_S))))
            length = max(1, round(vmaf_seconds / 2 * n / a["duration"]))
            firsts = [max(0, min(n - length, int(n * (i + 0.5) / k) - length // 2)) for i in range(k)]
            wins = _vmaf_spread(src, dst, a["width"], a["height"], [(f, f + length - 1) for f in firsts], src_vf)
            v = float(np.mean(wins)) if wins else None
            worst_win = min(wins) if wins else None
            span = f"{k} windows of {vmaf_seconds / 2:.0f}s spread over the video, output scaled to source" + \
                (f"; lowest window {worst_win:.2f}" if wins else "")
        else:
            v = _vmaf(src, dst, a["width"], a["height"], min(vmaf_seconds, a["duration"]), src_vf)
            span = f"first {min(vmaf_seconds, a['duration']):.0f}s, output scaled to source"
        if v is not None and worst_win is not None and reference_vmaf is None and worst_win < vt["vmaf_floor"]:
            gates.append(Gate("vmaf", "fail", round(v, 2), vt["vmaf_floor"],
                              f"{span}: one stretch is below the gross-failure floor"))
        elif v is None:
            gates.append(Gate("vmaf", "warn", None, None, f"{span}; could not be measured (is libvmaf in ffmpeg?)"))
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
             "coverage": "too few frames could be compared: the result cannot be confirmed, the original stays",
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


# ----------------------------------------------------------------------------- padded repeats removed
DEREPEAT_KEPT_SHARE = 0.97      # share of sampled input frames whose picture is still in the output
DEREPEAT_LEFT = 0.03            # copies left among moving frames after the pass


def _rate_gates(a: dict, b: dict, fi, fo) -> list[Gate]:
    """Structure of a pass that changes the frame rate by choosing frames: count at the new rate, duration,
    audio, lip sync, shape."""
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
    return gates


def _best_psnr(frame: np.ndarray, pool: list[np.ndarray], centre: int, search: int) -> float | None:
    lo, hi = max(0, centre - search), min(len(pool), centre + search + 1)
    if lo >= hi:
        return None
    f32 = frame.astype(np.float32)
    mse = min(float(np.mean((f32 - pool[m]) ** 2)) for m in range(lo, hi))
    return float(10 * np.log10(255 ** 2 / max(mse, 1e-6)))


def verify_derepeat(src: Path, dst: Path, *, in_fps, out_fps, windows: int = 4, seconds: float = 6.0) -> GuardReport:
    """Repeat removal against its input. Structure as for un-blending, then what makes dropping frames safe:
    * selection: every sampled output frame is an exact copy of an input frame near the same time;
    * pictures: every sampled input frame still has its picture in the output (only copies were dropped: a
      dropped drawing would be a picture the person never sees again);
    * repeats: the padding cycle is gone from the output."""
    from fractions import Fraction

    from ..probe import blend, framing

    src, dst = Path(src), Path(dst)
    a, b = media.probe(src, exact=True), media.probe(dst, exact=True)
    fi, fo = Fraction(in_fps), Fraction(out_fps)
    gates = _rate_gates(a, b, fi, fo)
    copies = judged = kept = seen = 0
    worst = worst_in = None
    span = max(1.0, min(seconds, a["duration"] / max(1, windows) - 1))
    for k in range(windows):
        t = max(0.0, a["duration"] * (k + 0.5) / windows - span / 2)
        pre = min(1.0, t)
        outs = list(blend.read_gray(dst, t - pre, span + 2.0, width=180))
        ins = list(blend.read_gray(src, t - pre, span + 2.0, width=180))
        if not outs or not ins:
            continue
        first, last = round(pre * fo), round((pre + span) * fo)
        for i in range(first, min(last, len(outs))):                       # output frames of the inner span
            p = _best_psnr(outs[i], ins, round(i / fo * fi), DEBLEND_SEARCH)
            if p is None:
                continue
            judged += 1
            copies += p >= DEBLEND_COPY_PSNR
            worst = p if worst is None else min(worst, p)
        first, last = round(pre * fi), round((pre + span) * fi)
        for j in range(first, min(last, len(ins))):                        # input frames of the inner span
            p = _best_psnr(ins[j], outs, round(j / fi * fo), DEBLEND_SEARCH)
            if p is None:
                continue
            seen += 1
            kept += p >= DEBLEND_COPY_PSNR
            worst_in = p if worst_in is None else min(worst_in, p)
    share = copies / judged if judged else 0.0
    gates.append(Gate("selection", "pass" if judged and share >= DEBLEND_COPY_SHARE else "fail", round(share, 4),
                      DEBLEND_COPY_SHARE, f"{copies}/{judged} sampled output frames are copies of input frames "
                      f"(worst {worst if worst is None else round(worst, 1)} dB)"))
    held = kept / seen if seen else 0.0
    gates.append(Gate("pictures", "pass" if seen and held >= DEREPEAT_KEPT_SHARE else "fail", round(held, 4),
                      DEREPEAT_KEPT_SHARE, f"{kept}/{seen} sampled input pictures are still in the output "
                      f"(worst {worst_in if worst_in is None else round(worst_in, 1)} dB)"))
    left = framing.repeats(dst)
    gates.append(Gate("repeats", "pass" if left.share <= DEREPEAT_LEFT and not left.is_padded else "fail", left.share,
                      DEREPEAT_LEFT, f"copied frames {left.share:.1%} of moving frames after the pass"
                      + (f"; still a cycle of {left.cycle}" if left.is_padded else "")))
    failed = [g.name for g in gates if g.status == "fail"]
    sugg = {"selection": "the output is not made of input frames: keep the padded input",
            "pictures": "real pictures were dropped, not only copies: keep the padded input",
            "repeats": "the copies are still there at this rate: keep the padded input"}
    return GuardReport(not failed, gates, [sugg[n] for n in failed if n in sugg], False)


# ----------------------------------------------------------------------------- composed clean-up
SPANS_LEFT = 0.25               # share of a span's interlaced frames that may still read interlaced afterwards
SPANS_SAME_PSNR = 50.0          # dB: frames outside the spans must be untouched. Not 60: MPEG-2 decoders round
                                # their inverse transform differently (ffmpeg here, L-SMASH in the pass: 55.5 dB on
                                # an untouched DVD frame, measured); a deinterlaced frame scores about 40 or less


def verify_spans(src: Path, dst: Path, *, spans: list, frames: int = 16) -> GuardReport:
    """Span deinterlacing against its input: same frames, duration, audio, shape; frames outside the spans are
    the input's, untouched; inside the spans the combing is gone."""
    from fractions import Fraction

    from ..probe import combing

    src, dst = Path(src), Path(dst)
    a, b = media.probe(src, exact=True), media.probe(dst, exact=True)
    gates = _rate_gates(a, b, Fraction(1), Fraction(1))
    inside = {i for s, e in spans for i in range(s, e + 1)}
    total = min(a["frames"], b["frames"])
    outside = [i for i in (int(total * (k + 0.5) / (frames * 3)) for k in range(frames * 3)) if i not in inside][:frames]
    worst = None
    with tempfile.TemporaryDirectory(prefix="vidimg_span_") as tmp:
        fa = _frames_by_index(src, outside, Path(tmp), "s", pre_vf="extractplanes=y")
        fb = _frames_by_index(dst, outside, Path(tmp), "d", pre_vf="extractplanes=y")
        for i in outside:
            if i not in fa or i not in fb:
                continue
            x, y = _luma8(fa[i]), _luma8(fb[i])
            if x is None or y is None or x.shape != y.shape:
                continue
            p = float(10 * np.log10(255 ** 2 / max(float(np.mean((x - y) ** 2)), 1e-6)))
            worst = p if worst is None else min(worst, p)
    gates.append(Gate("outside", "pass" if worst is not None and worst >= SPANS_SAME_PSNR else "fail",
                      None if worst is None else round(worst, 1), SPANS_SAME_PSNR,
                      "frames outside the spans against the input (lowest of the sampled frames)"))
    before, after = combing.read_kinds(src), combing.read_kinds(dst)
    hit = lambda kinds: sum(1 for i in inside if i < len(kinds) and kinds[i] in ("tff", "bff"))  # noqa: E731
    nb, na = hit(before), hit(after)
    left = na / nb if nb else 1.0
    gates.append(Gate("combing", "pass" if nb and left <= SPANS_LEFT else "fail", round(left, 3), SPANS_LEFT,
                      f"interlaced frames inside the spans: {nb} before, {na} after"))
    failed = [g.name for g in gates if g.status == "fail"]
    sugg = {"outside": "frames outside the spans changed: keep the input",
            "combing": "the spans still read interlaced: keep the input"}
    return GuardReport(not failed, gates, [sugg[n] for n in failed if n in sugg], False)


def block_level(path: Path, seconds: float = 60.0) -> float | None:
    """ffmpeg blockdetect's mean over the middle of the video (higher = more visible 8x8 block edges)."""
    info = media.probe(path)
    start = max(0.0, (info["duration"] or 0) / 2 - seconds / 2)
    cp = subprocess.run([_ffmpeg(), "-hide_banner", "-nostats", "-nostdin", "-ss", f"{start:.2f}", "-t", f"{seconds:g}",
                         "-i", str(path), "-map", "0:v:0", "-vf", "blockdetect", "-an", "-f", "null", "-"],
                        capture_output=True, text=True, encoding="utf-8", errors="replace", stdin=subprocess.DEVNULL,
                        timeout=900)
    found = re.findall(r"block mean:\s*([0-9.]+)", cp.stderr)
    return float(found[-1]) if found else None


def verify_cleanup(src: Path, dst: Path, *, deblocked: bool = False, drawn_faces: bool = False) -> GuardReport:
    """A clean-up pass (deblock, dehalo) against its input: every gate of the video guard (the result is still
    this video: structure, colour, faces, no added flicker), and a filter that was asked to remove blocks must
    have removed some."""
    rep = verify_video(src, dst, drawn_faces=drawn_faces)
    gates = list(rep.gates)
    if deblocked:
        before, after = block_level(src), block_level(dst)
        ok = before is not None and after is not None and after < before
        gates.append(Gate("blocks", "pass" if ok else "fail", after, before,
                          f"block edges {before} before, {after} after"))
    failed = [g.name for g in gates if g.status == "fail"]
    return GuardReport(not failed, gates, list(rep.suggestions) + (
        ["deblocking removed no block edges: keep the input"] if "blocks" in failed else []), False)


# ----------------------------------------------------------------------------- ghost frames replaced
GHOSTS_LEFT = 0.25              # share of the ghosts that may still be found afterwards
GHOST_SAME_PSNR = 50.0          # dB: a frame of the result is a frame of the input (decoders round differently,
                                # see SPANS_SAME_PSNR; another picture scores far below)


def verify_deghost(src: Path, dst: Path, *, pairs: list, frames: int = 16) -> GuardReport:
    """Ghost replacement against its input: same frames, duration, audio, shape; every sampled replaced frame is
    the neighbour it was to become, every sampled other frame is itself; and the ghosts are gone."""
    from fractions import Fraction

    from ..probe import ghosts

    src, dst = Path(src), Path(dst)
    a, b = media.probe(src, exact=True), media.probe(dst, exact=True)
    gates = _rate_gates(a, b, Fraction(1), Fraction(1))
    shown = {int(i): int(j) for i, j in pairs}
    total = min(a["frames"], b["frames"])
    step = max(1, len(pairs) // frames)
    changed = [int(i) for i, _ in pairs[::step]][:frames]
    kept = [i for i in (int(total * (k + 0.5) / (frames * 2)) for k in range(frames * 2)) if i not in shown][:frames]

    def luma(path: Path, idx: list[int], folder: Path, tag: str) -> dict:
        return {i: _luma8(f) for i, f in _frames_by_index(path, idx, folder, tag, pre_vf="extractplanes=y").items()}

    def psnr(x, y) -> float | None:
        if x is None or y is None or x.shape != y.shape:
            return None
        return float(10 * np.log10(255 ** 2 / max(float(np.mean((x - y) ** 2)), 1e-6)))

    with tempfile.TemporaryDirectory(prefix="vidimg_ghost_") as tmp:
        t = Path(tmp)
        out = luma(dst, sorted(set(changed + kept)), t, "d")
        ref = luma(src, sorted({shown[i] for i in changed} | set(kept)), t, "s")
        rep = [p for p in (psnr(ref.get(shown[i]), out.get(i)) for i in changed) if p is not None]
        same = [p for p in (psnr(ref.get(i), out.get(i)) for i in kept) if p is not None]
    gates.append(Gate("replaced", "pass" if rep and min(rep) >= GHOST_SAME_PSNR else "fail",
                      round(min(rep), 1) if rep else None, GHOST_SAME_PSNR,
                      f"{len(rep)} sampled replaced frames against the neighbour each was to become (lowest)"))
    gates.append(Gate("untouched", "pass" if same and min(same) >= GHOST_SAME_PSNR else "fail",
                      round(min(same), 1) if same else None, GHOST_SAME_PSNR,
                      f"{len(same)} sampled other frames against themselves (lowest)"))
    left = ghosts.scan(dst)
    limit = max(2, round(GHOSTS_LEFT * len(pairs)))
    ok = left.error is None and len(left.ghosts) <= limit
    gates.append(Gate("ghosts", "pass" if ok else "fail", len(left.ghosts), limit,
                      left.error or f"ghost frames: {len(pairs)} before, {len(left.ghosts)} after"))
    failed = [g.name for g in gates if g.status == "fail"]
    sugg = {"replaced": "a replaced frame is not the neighbour it was to become: keep the input",
            "untouched": "frames that were not ghosts changed: keep the input",
            "ghosts": "the ghost frames are still there: keep the input"}
    return GuardReport(not failed, gates, [sugg[n] for n in failed if n in sugg], False)


# ----------------------------------------------------------------------------- ghost frames made again (a copy)
REBUILT_LEFT = 0.25             # share of the sampled frames made again that may be no better than the ghost


def verify_rebuilt(src: Path, dst: Path, *, rebuilt: list, frames: int = 24) -> GuardReport:
    """A copy whose ghost frames were made again from their neighbours (rebuild.py), against the file it was made
    from: same frames, duration, audio, shape; every sampled frame that was NOT made again is the input's; and the
    frames that were made again no longer dip below their neighbours. The made frames themselves are created
    picture: nothing here vouches for them, the record says how many they are."""
    from fractions import Fraction

    from ..probe import ghosts

    src, dst = Path(src), Path(dst)
    a, b = media.probe(src, exact=True), media.probe(dst, exact=True)
    gates = _rate_gates(a, b, Fraction(1), Fraction(1))
    made = {int(i) for i in rebuilt}
    total = min(a["frames"], b["frames"])
    kept = [i for i in (int(total * (k + 0.5) / (frames * 2)) for k in range(frames * 2)) if i not in made][:frames]
    worst = None
    with tempfile.TemporaryDirectory(prefix="vidimg_rebuilt_") as tmp:
        fa = _frames_by_index(src, kept, Path(tmp), "s", pre_vf="extractplanes=y")
        fb = _frames_by_index(dst, kept, Path(tmp), "d", pre_vf="extractplanes=y")
        for i in kept:
            x, y = (_luma8(f[i]) if i in f else None for f in (fa, fb))
            if x is None or y is None or x.shape != y.shape:
                continue
            p = float(10 * np.log10(255 ** 2 / max(float(np.mean((x - y) ** 2)), 1e-6)))
            worst = p if worst is None else min(worst, p)
    gates.append(Gate("untouched", "pass" if worst is not None and worst >= GHOST_SAME_PSNR else "fail",
                      None if worst is None else round(worst, 1), GHOST_SAME_PSNR,
                      "frames that were not made again, against the input (lowest of the sampled frames)"))
    # the frames made again: against both neighbours, their edge energy must have risen (a ghost carries about
    # half of a single picture's; measured on the files themselves, whatever was done to them since the ghosts
    # were found)
    order = sorted(made)
    some = order[::max(1, len(order) // 12)][:12]
    want = sorted({k for i in some for k in (i - 1, i, i + 1) if 0 <= k < total})
    better = judged = 0
    with tempfile.TemporaryDirectory(prefix="vidimg_rebuilt2_") as tmp:
        fa = _frames_by_index(src, want, Path(tmp), "s", pre_vf="extractplanes=y")
        fb = _frames_by_index(dst, want, Path(tmp), "d", pre_vf="extractplanes=y")

        def dip(files: dict, i: int) -> float | None:
            e = [ghosts._energy(x) if x is not None else None for x in (_luma8(files[k]) if k in files else None
                                                                       for k in (i - 1, i, i + 1))]
            return None if None in e or e[0] + e[2] <= 0 else e[1] / ((e[0] + e[2]) / 2)

        for i in some:
            x, y = dip(fa, i), dip(fb, i)
            if x is None or y is None:
                continue
            judged += 1
            better += y > x
    share = better / judged if judged else 0.0
    gates.append(Gate("ghosts", "pass" if judged and share >= 1 - REBUILT_LEFT else "fail", round(share, 3),
                      1 - REBUILT_LEFT, f"{better} of {judged} sampled frames made again carry more edge energy "
                                        "against their neighbours than the ghosts they replace"))
    failed = [g.name for g in gates if g.status == "fail"]
    sugg = {"untouched": "frames that were not to be made again changed: the copy is not kept",
            "ghosts": "the frames made again still look like ghosts: the copy is not kept"}
    return GuardReport(not failed, gates, [sugg[n] for n in failed if n in sugg], False)


# ----------------------------------------------------------------------------- stabilised: moved, not altered
MOVED_PSNR = 40.0               # dB after lining the frames up. Measured on an owner cut with artificial jitter:
                                # a stabilised frame against its input 43.3 at worst (median 52.3); the same frames
                                # not lined up 26.9; a slightly blurred copy 35.3; another frame 21.3
MOVED_STEADIER = 0.8            # frame-to-frame movement must fall to this share of the input's, or nothing was gained
MOVED_MIN_FRAMES = 4            # sampled frames that must be judged (a flat frame gives no position to read)
MOVED_RESPONSE = 0.1            # phase-correlation response below which a frame's position is not trusted


def _shake(path: Path, windows: int = 3, seconds: float = 3.0) -> float | None:
    """Mean change from one frame to the next (8-bit luma at analysis size), over windows spread through the video."""
    from ..probe import blend

    dur = media.probe(path, exact=True)["duration"]
    vals = []
    for k in range(windows):
        fr = [f.astype(np.float32) for f in blend.read_gray(path, max(0.0, dur * (k + 0.5) / windows - seconds / 2), seconds, width=180)]
        vals += [float(np.mean(np.abs(fr[i + 1] - fr[i]))) for i in range(len(fr) - 1)]
    return float(np.mean(vals)) if vals else None


def verify_moved(src: Path, dst: Path, *, max_shift: float, frames: int = 16) -> GuardReport:
    """A stabilised video against its input. The ordinary guard compares each frame with the source frame in
    place, so it reads a picture that was only moved as a picture that was changed. Here each sampled frame is
    first lined up with its source (the translation is measured, by phase correlation), and then:
    * shift: no frame was moved further than the step was allowed to move it;
    * picture: lined up, the frame is the source's, in all three colour channels, away from the edges the move
      uncovered (a stabiliser may shift the picture; it may not blur, redraw, zoom or rotate it);
    * steadier: the frame-to-frame movement fell, otherwise the step is not worth keeping;
    and the structure is unchanged: same frames, duration, audio, lip sync, shape."""
    from fractions import Fraction

    src, dst = Path(src), Path(dst)
    a, b = media.probe(src, exact=True), media.probe(dst, exact=True)
    gates = _rate_gates(a, b, Fraction(1), Fraction(1))
    total = min(a["frames"], b["frames"])
    idx = sorted({int(total * (i + 0.5) / frames) for i in range(frames)})
    margin = int(np.ceil(max_shift)) + 6
    worst = far = None
    judged = 0
    with tempfile.TemporaryDirectory(prefix="vidimg_moved_") as tmp:
        fa = _frames_by_index(src, idx, Path(tmp), "s")
        fb = _frames_by_index(dst, idx, Path(tmp), "d")
        for i in idx:
            if i not in fa or i not in fb:
                continue
            x, y = imgfile.imread(str(fa[i])), imgfile.imread(str(fb[i]))
            if x is None or y is None or x.shape != y.shape or min(x.shape[:2]) <= 4 * margin:
                continue
            gx, gy = (cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in (x, y))
            (dx, dy), response = cv2.phaseCorrelate(gx, gy, cv2.createHanningWindow((gx.shape[1], gx.shape[0]), cv2.CV_32F))
            if response < MOVED_RESPONSE:
                continue                                    # nothing to hold on to in this frame
            lined = cv2.warpAffine(x.astype(np.float32), np.float32([[1, 0, dx], [0, 1, dy]]), (x.shape[1], x.shape[0]),
                                   flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            diff = (lined - y.astype(np.float32))[margin:-margin, margin:-margin]
            p = float(10 * np.log10(255 ** 2 / max(float(np.mean(diff ** 2)), 1e-6)))
            judged += 1
            worst = p if worst is None else min(worst, p)
            far = max(far or 0.0, abs(float(dx)), abs(float(dy)))
    ok = judged >= MOVED_MIN_FRAMES
    gates.append(Gate("shift", "pass" if ok and far <= max_shift + 0.5 else "fail", None if far is None else round(far, 2),
                      max_shift, f"largest move of a sampled frame, in pixels ({judged} frames judged)"))
    gates.append(Gate("picture", "pass" if ok and worst >= MOVED_PSNR else "fail", None if worst is None else round(worst, 1),
                      MOVED_PSNR, "each frame lined up with its source and compared (lowest of the sampled frames)"))
    before, after = _shake(src), _shake(dst)
    steadier = before is not None and after is not None and after <= MOVED_STEADIER * before
    gates.append(Gate("steadier", "pass" if steadier else "fail", None if after is None else round(after, 3),
                      None if before is None else round(MOVED_STEADIER * before, 3),
                      f"frame-to-frame movement {before if before is None else round(before, 3)} before, "
                      f"{after if after is None else round(after, 3)} after"))
    failed = [g.name for g in gates if g.status == "fail"]
    sugg = {"shift": "a frame was moved further than allowed, or too few frames could be judged: keep the input",
            "picture": "lined up, the frames are not the source's: the step changed the picture, not only its position",
            "steadier": "the picture is no steadier than before: keep the input"}
    return GuardReport(not failed, gates, [sugg[n] for n in failed if n in sugg], False)


# ----------------------------------------------------------------------------- black borders cropped
CROP_SAME_PSNR = 60.0           # dB: the picture inside the crop must be untouched
CROP_BLACK_LEVEL = 40           # 8-bit luma: 95 % of what was cut away lies below this ...
CROP_BLACK_SHARE = 0.9          # ... in at least this share of the sampled frames


def _luma8(png: Path) -> np.ndarray | None:
    """A stored luma plane (8- or 16-bit PNG) on the 8-bit scale, as float."""
    img = imgfile.imread(str(png), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 3:
        img = img[..., 0]
    return img.astype(np.float32) / (257.0 if img.dtype == np.uint16 else 1.0)


def verify_crop(src: Path, dst: Path, *, crop: tuple[int, int, int, int], frames: int = 16) -> GuardReport:
    """A border crop against its input: same frames, duration, audio and pixel shape; the kept picture is
    the input's, untouched; and what was cut away really was black, on frames spread over the video."""
    from fractions import Fraction

    src, dst = Path(src), Path(dst)
    w, h, x, y = crop
    a, b = media.probe(src, exact=True), media.probe(dst, exact=True)
    gates = [g for g in _rate_gates(a, b, Fraction(1), Fraction(1)) if g.name != "aspect"]
    gates.append(Gate("size", "pass" if (b["width"], b["height"]) == (w, h) else "fail", f"{b['width']}x{b['height']}",
                      f"{w}x{h}", "the cropped size"))
    sa, sb = a.get("sar") or "1:1", b.get("sar") or "1:1"
    same = Fraction(*map(int, sa.split(":"))) == Fraction(*map(int, sb.split(":")))
    gates.append(Gate("pixel_aspect", "pass" if same else "fail", sb, sa,
                      "pixel shape unchanged (the picture is cut, never stretched)"))
    total = min(a["frames"], b["frames"])
    idx = sorted({int(total * (i + 0.5) / frames) for i in range(frames)})
    worst, black, judged = None, 0, 0
    with tempfile.TemporaryDirectory(prefix="vidimg_crop_") as tmp:
        # the luma plane as stored: through RGB, an 8-bit input and its 10-bit lossless copy round differently
        # (48 dB on identical pictures, measured)
        full = _frames_by_index(src, idx, Path(tmp), "s", pre_vf="extractplanes=y")
        cut = _frames_by_index(dst, idx, Path(tmp), "d", pre_vf="extractplanes=y")
        for i in idx:
            if i not in full or i not in cut:
                continue
            g, c = _luma8(full[i]), _luma8(cut[i])
            if g is None or c is None or c.shape[:2] != (h, w):
                continue
            inner = g[y:y + h, x:x + w]
            mse = float(np.mean((inner - c) ** 2))
            p = float(10 * np.log10(255 ** 2 / max(mse, 1e-6)))
            worst = p if worst is None else min(worst, p)
            mask = np.ones(g.shape, bool)
            mask[y:y + h, x:x + w] = False
            judged += 1
            black += bool(mask.any()) and float(np.percentile(g[mask], 95)) <= CROP_BLACK_LEVEL
    gates.append(Gate("picture", "pass" if worst is not None and worst >= CROP_SAME_PSNR else "fail",
                      None if worst is None else round(worst, 1), CROP_SAME_PSNR,
                      "the kept picture against the same area of the input (lowest of the sampled frames)"))
    share = black / judged if judged else 0.0
    gates.append(Gate("removed", "pass" if judged and share >= CROP_BLACK_SHARE else "fail", round(share, 3),
                      CROP_BLACK_SHARE, f"what was cut away is black in {black}/{judged} sampled frames"))
    failed = [g.name for g in gates if g.status == "fail"]
    sugg = {"picture": "the crop changed the picture it kept: keep the input",
            "removed": "the cut-away edges hold picture, not black: keep the input"}
    return GuardReport(not failed, gates, [sugg[n] for n in failed if n in sugg], False)


# ----------------------------------------------------------------------------- chroma shift correction
CHROMA_RESIDUAL_MAX = 0.6       # luma px left beyond the clean baseline after the correction (measured 0.0-0.33)
LUMA_UNCHANGED_PSNR = 60.0      # dB: a chroma-only pass must leave the luma bit-identical (measured 138 dB)


CHROMA_MIN_FRAMES = 3           # of the 12 sampled: fewer readings confirm nothing
CHROMA_UNTOUCHED_SLACK = 0.3   # luma px a component that was not corrected may drift in the re-measurement


def verify_chroma(src: Path, dst: Path, *, moved: dict, before: dict, luma_must_match: bool,
                  frames: int = 24) -> GuardReport:
    """The chroma-shift correction did what it should and nothing else. Measured again on the output: every
    corrected component now lines up with the line art, and the components left alone (the frames did not
    agree on them) are no worse; when no field handling ran in the same pass, the luma is untouched.
    moved: {"u": [dx, dy], "v": [...]} applied (luma px); before: the profile's excess over the baseline."""
    from ..probe import chroma

    gates: list[Gate] = []
    measured = chroma.analyze(Path(dst))
    after = measured.excess()
    if measured.frames < CHROMA_MIN_FRAMES:
        # nothing could be read back: the default reading is "no shift", which passed without any evidence
        gates.append(Gate("chroma_aligned", "fail", None, CHROMA_RESIDUAL_MAX,
                          f"re-measured on {measured.frames} frame(s), at least {CHROMA_MIN_FRAMES} needed: "
                          "the correction cannot be confirmed"))
        return GuardReport(False, gates, ["chroma correction could not be confirmed: run without it"], False)
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
                                  "-pix_fmt", "gray16le", "-f", "rawvideo", "-"], capture_output=True, stdin=subprocess.DEVNULL).stdout
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


# ----------------------------------------------------------------------------- bottom mask (head switching)
MASK_BLACK_TOL = 2.0            # 8-bit levels around limited black (16) for the masked rows


def verify_mask(src: Path, dst: Path, *, rows: int, luma_must_match: bool, frames: int = 24) -> GuardReport:
    """The head-switching mask did exactly that: the bottom `rows` rows are black and, when no field handling
    ran in the same pass, every row above them is bit-identical to the source."""
    a, b = media.probe(Path(src)), media.probe(Path(dst))
    gates: list[Gate] = []

    def luma(path: Path, w: int, h: int) -> np.ndarray:
        raw = subprocess.run([media.ffmpeg(), "-hide_banner", "-loglevel", "error", "-i", str(path), "-map", "0:v:0",
                              "-frames:v", str(frames), "-vf", "format=yuv420p16le,extractplanes=y", "-pix_fmt",
                              "gray16le", "-f", "rawvideo", "-"], capture_output=True, stdin=subprocess.DEVNULL).stdout
        n = len(raw) // (w * h * 2)
        return (np.frombuffer(raw[: n * w * h * 2], "<u2").reshape(n, h, w) >> 8).astype(np.float32)

    yb = luma(Path(dst), b["width"], b["height"])
    band = yb[:, -rows:] if len(yb) else np.full((1, 1, 1), 255.0)
    off = float(np.max(np.abs(band - 16))) if band.size else 255.0
    gates.append(Gate("mask_black", "pass" if off <= MASK_BLACK_TOL else "fail", round(off, 1), MASK_BLACK_TOL,
                      f"bottom {rows} rows: largest distance from black {off:.1f} levels"))
    if luma_must_match:
        same_size = (a["width"], a["height"]) == (b["width"], b["height"])
        ya = luma(Path(src), a["width"], a["height"]) if same_size else np.zeros(0)
        n = min(len(ya), len(yb))
        mse = float(np.mean((ya[:n, :-rows] - yb[:n, :-rows]) ** 2)) if n and same_size else 1e9
        ps = 10 * np.log10(255 ** 2 / max(mse, 1e-9))
        gates.append(Gate("above_unchanged", "pass" if ps >= LUMA_UNCHANGED_PSNR else "fail", round(ps, 1),
                          LUMA_UNCHANGED_PSNR, f"rows above the mask vs the source: {ps:.1f} dB"))
    failed = [g.name for g in gates if g.status == "fail"]
    return GuardReport(not failed, gates, ["the bottom mask did not come out as planned: run without it"]
                       if failed else [], False)
