"""Close-ups for whoever decides the recipe (Claude, or a person): what a number cannot settle, a look can.

Measured on the owner's samples (2026-11-21): a halo score read white eyes beside dark pupils as ringing on one
video and was right on another; a black bar and a night scene's edge give the same border reading; a padded copy
and a held drawing give the same frame difference. So the survey shows, per video:
  * line close-ups: the busiest line art of three frames, enlarged without smoothing (halos, ringing, blocks,
    stair-steps, colour off the lines);
  * the frame with the measured picture area outlined (are those bars, or picture?);
  * a strip of consecutive frames around a measured copy (is it padding, or a held drawing?).
The same module lays recipe trials side by side (one column per recipe) for the choice between them.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np

from .. import imgfile, media

CELL = (160, 120)               # source pixels shown per close-up ...
ZOOM = 3                        # ... enlarged this much, nearest neighbour (nothing smoothed, nothing invented)


def frames_at(src: Path, t: float, count: int = 1, index: int | None = None) -> list[np.ndarray]:
    """`count` consecutive frames from time t, as stored (BGR). index: the frame with this number instead (decoded
    from the start, for short clips): two files of the same frames are then compared on the same frame, which a
    time is not sure to give (a 10 fps source and its result came out one frame apart, across a scene cut)."""
    from .. import texture
    info = media.probe(src)
    w, h = info["width"], info["height"]
    # the colour matrix named, by the tool's own rule (the file's tag or its sidecar, else by height): left to
    # ffmpeg, an untagged HD source was read as BT.601 beside its BT.709-tagged result, and the source column of a
    # recipe sheet looked lighter than every recipe (owner sample, 2026-11-21)
    tag = (media.colour_hint(src) or {}).get("color_space") or texture._probe_stream(src).get("color_space")
    matrix = texture._matrix(tag, h)
    seek = [] if index is not None else ["-ss", f"{max(0.0, t):.3f}"]
    pick = f"select='gte(n\\,{int(index)})'," if index is not None else ""
    cp = subprocess.run([media.ffmpeg(), "-hide_banner", "-loglevel", "error", "-nostdin", *seek,
                         "-i", str(src), "-map", "0:v:0", "-frames:v", str(count),
                         "-vf", f"{pick}scale=in_color_matrix={matrix},format=bgr24", "-fps_mode", "passthrough",
                         "-f", "rawvideo",
                         "-"], capture_output=True, stdin=subprocess.DEVNULL, timeout=300)
    size = w * h * 3
    return [np.frombuffer(cp.stdout[i * size:(i + 1) * size], np.uint8).reshape(h, w, 3)
            for i in range(len(cp.stdout) // size)] if size else []


STATIC_LEVEL = 12               # 8-bit levels: a pixel that stays within this over distant frames is an overlay


def overlay_mask(frames: list[np.ndarray]) -> np.ndarray | None:
    """Pixels that do not change between frames taken far apart: a channel logo, burned-in bars. Without it every
    close-up showed the logo, the busiest spot of each frame (owner sample, 2026-11-21)."""
    if len(frames) < 2 or len({f.shape for f in frames}) != 1:
        return None
    g = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.int16) for f in frames])
    still = (g.max(0) - g.min(0)) < STATIC_LEVEL
    return cv2.dilate(still.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0


def busiest(frame: np.ndarray, cell: tuple[int, int] = CELL, ignore: np.ndarray | None = None) -> tuple[int, int]:
    """Top-left corner of the cell with the most line detail (mean Laplacian energy); detail under `ignore`
    (an overlay mask) does not count."""
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = g.shape
    cw, ch = min(cell[0], w), min(cell[1], h)
    lap = np.abs(cv2.Laplacian(g, cv2.CV_32F))
    if ignore is not None and ignore.shape == lap.shape and not ignore.all():
        lap[ignore] = 0.0
    e = cv2.blur(lap, (cw, ch))
    inner = e[ch // 2:h - ch // 2 + 1, cw // 2:w - cw // 2 + 1]
    if inner.size == 0:
        return 0, 0
    y, x = np.unravel_index(int(np.argmax(inner)), inner.shape)
    return int(x), int(y)


def _label(img: np.ndarray, text: str, colour=(255, 255, 255)) -> np.ndarray:
    bar = np.full((26, img.shape[1], 3), 30, np.uint8)
    cv2.putText(bar, text.encode("ascii", "replace").decode()[:img.shape[1] // 9], (6, 18), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, colour, 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def _row(cells: list[np.ndarray], width: int) -> np.ndarray:
    h = max(c.shape[0] for c in cells)
    cells = [cv2.copyMakeBorder(c, 0, h - c.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30)) for c in cells]
    row = np.hstack(cells)
    if row.shape[1] < width:
        row = cv2.copyMakeBorder(row, 0, 0, 0, width - row.shape[1], cv2.BORDER_CONSTANT, value=(30, 30, 30))
    return row[:, :width]


def crop_zoom(frame: np.ndarray, x: int, y: int, cell: tuple[int, int] = CELL, zoom: int = ZOOM) -> np.ndarray:
    c = frame[y:y + cell[1], x:x + cell[0]]
    return cv2.resize(c, (c.shape[1] * zoom, c.shape[0] * zoom), interpolation=cv2.INTER_NEAREST)


def comb_zoom(frame: np.ndarray, cell: tuple[int, int] = CELL, zoom: int = ZOOM) -> np.ndarray:
    """The part of a frame with the strongest line-to-line alternation (comb teeth), enlarged."""
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    teeth = np.zeros_like(g)
    teeth[1:-1] = np.abs(g[2:] - 2 * g[1:-1] + g[:-2])
    h, w = g.shape
    cw, ch = min(cell[0], w), min(cell[1], h)
    e = cv2.blur(teeth, (cw, ch))[ch // 2:h - ch // 2 + 1, cw // 2:w - cw // 2 + 1]
    y, x = np.unravel_index(int(np.argmax(e)), e.shape) if e.size else (0, 0)
    return crop_zoom(frame, int(x), int(y), cell, zoom)


def sheet(src: Path, dst: Path, *, borders: dict | None = None, repeats: dict | None = None,
          combing: dict | None = None, ghost: list | None = None, title: str = "") -> Path | None:
    """The diagnosis sheet for one video (see the module text). None when no frame could be read."""
    src, dst = Path(src), Path(dst)
    info = media.probe(src)
    dur, fps = info["duration"] or 0.0, info["fps"] or 25.0
    width = CELL[0] * ZOOM * 3
    rows: list[np.ndarray] = []
    lines = []
    picked = [(k, fr[0]) for k in (0.25, 0.5, 0.75) for fr in [frames_at(src, dur * k)] if fr]
    overlay = overlay_mask([f for _, f in picked])
    for k, f in picked:
        x, y = busiest(f, ignore=overlay)
        lines.append(_label(crop_zoom(f, x, y), f"lines at {dur * k:.0f} s, x{ZOOM} (halos? blocks? stair-steps?)"))
    if not lines:
        return None
    rows.append(_row(lines, width))
    c = combing or {}
    if c.get("spans"):
        spans = sorted(c["spans"], key=lambda s: s[0] - s[1])[:3]          # the longest three
        cells = []
        for a, z in sorted(spans):
            fr = frames_at(src, ((a + z) / 2) / fps)
            if fr:
                cells.append(_label(comb_zoom(fr[0]), f"frames {a}-{z} (comb teeth on moving edges?)"))
        if cells:
            head = np.full((26, width, 3), 30, np.uint8)
            cv2.putText(head, f"measured interlaced stretches: {len(c['spans'])} spans, {c.get('span_frames')} frames "
                              "(interlacing, or thin drawn stripes?)", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)
            rows += [head, _row(cells, width)]
    if ghost:
        i, j = int(ghost[0]), int(ghost[1])
        fr = frames_at(src, 0.0, 3, index=max(0, i - 1))
        if len(fr) == 3:
            cw = width // 3
            cells = []
            for k, f in enumerate(fr):
                small = cv2.resize(f, (cw, round(f.shape[0] * cw / f.shape[1])), interpolation=cv2.INTER_AREA)
                label = f"frame {i - 1 + k}" + (" GHOST?" if k == 1 else (" (would be shown instead)" if i - 1 + k == j else ""))
                cells.append(_label(small, label, (0, 255, 255) if k == 1 else (255, 255, 255)))
            head = np.full((26, width, 3), 30, np.uint8)
            cv2.putText(head, "measured ghost frame, between its neighbours (two drawings at once, shown singly beside it?)",
                        (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            rows += [head, _row(cells, width)]
    b = borders or {}
    if b.get("has_borders"):
        fr = frames_at(src, dur * 0.4)
        if fr:
            f = fr[0].copy()
            h, w = f.shape[:2]
            cv2.rectangle(f, (b["left"], b["top"]), (w - b["right"] - 1, h - b["bottom"] - 1), (0, 0, 255), 1)
            scale = min(1.0, width / w)
            f = cv2.resize(f, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
            rows.append(_row([_label(f, f"measured picture area in red: bars left {b['left']}, right {b['right']}, "
                                        f"top {b['top']}, bottom {b['bottom']} px (bars, or picture?)")], width))
    r = repeats or {}
    if r.get("example_s") is not None and r.get("repeats"):
        fr = frames_at(src, r["example_s"] - 4 / fps, 8)
        if len(fr) >= 3:
            cw = width // len(fr)
            small = [cv2.resize(f, (cw, round(f.shape[0] * cw / f.shape[1])), interpolation=cv2.INTER_AREA) for f in fr]
            gray = [cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (160, 120), interpolation=cv2.INTER_AREA)
                    .astype(np.float32) for f in fr]
            cells = [_label(small[0], "frame 1")]
            steps = [float(np.mean(np.abs(gray[i] - gray[i - 1]))) for i in range(1, len(fr))]
            usual = float(np.median(steps))
            for i in range(1, len(fr)):
                d = steps[i - 1]
                same = d < 0.3 and d < 0.3 * usual           # far below the movement around it
                cells.append(_label(small[i], f"{'COPY' if same else 'new'} {d:.2f}", (0, 255, 255) if same else (255, 255, 255)))
            cyc = f"a copy every {r['cycle']} frames: {r['original_fps']} fps original" if r.get("cycle") else "no regular cycle"
            head = np.full((26, width, 3), 30, np.uint8)
            cv2.putText(head, f"consecutive frames at {r['example_s']:.1f} s ({cyc}; padding, or held drawings?)",
                        (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            rows += [head, _row(cells, width)]
    if title:
        head = np.full((30, width, 3), 60, np.uint8)
        cv2.putText(head, title.encode("ascii", "replace").decode()[:150], (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2, cv2.LINE_AA)
        rows.insert(0, head)
    dst.parent.mkdir(parents=True, exist_ok=True)
    imgfile.imwrite(str(dst), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 92])
    return dst


def frames_numbered(src: Path, numbers: list[int]) -> dict[int, np.ndarray]:
    """The frames with these numbers (counted from the start, as stored), in one pass over the file. A seek by
    time landed one frame off on some files, and here the frame beside the wanted one is the wrong picture."""
    from .. import texture
    want = sorted({int(n) for n in numbers if n >= 0})
    if not want:
        return {}
    info = media.probe(src)
    w, h = info["width"], info["height"]
    tag = (media.colour_hint(src) or {}).get("color_space") or texture._probe_stream(src).get("color_space")
    pick = "+".join(f"eq(n\\,{n})" for n in want)
    cp = subprocess.run([media.ffmpeg(), "-hide_banner", "-loglevel", "error", "-nostdin", "-i", str(src), "-map", "0:v:0",
                         "-vf", f"select='{pick}',scale=in_color_matrix={texture._matrix(tag, h)},format=bgr24",
                         "-fps_mode", "passthrough", "-f", "rawvideo", "-"],
                        capture_output=True, stdin=subprocess.DEVNULL, timeout=3600)
    size = w * h * 3
    got = [np.frombuffer(cp.stdout[i * size:(i + 1) * size], np.uint8).reshape(h, w, 3)
           for i in range(len(cp.stdout) // size)] if size else []
    return dict(zip(want, got))


DAMAGE_SHOWN = 12               # candidates on the sheet (the largest first)


RUNS_SHOWN = 8                  # runs of damaged frames on the sheet (the largest first)


def damage_sheet(src: Path, dst: Path, found: list[dict], *, title: str = "", runs: list[dict] | None = None) -> Path | None:
    """Candidate damaged frames (probe.damage), each between its two neighbours, and candidate runs (frames in a
    row between two frames that agree). Damage, or frames drawn that way (a flash, lightning, an impact frame)?
    Only a look can say; the measurement cannot."""
    src, dst = Path(src), Path(dst)
    # the ones that read like tape damage first (probe.damage, "tape"): those are most likely worth the look
    shown = sorted((d for d in found if d.get("lone")),
                   key=lambda d: (d.get("looks") == "flash", not d.get("tape"), -d["share"]))[:DAMAGE_SHOWN]
    long = sorted(runs or [], key=lambda r: (not r.get("tape"), -r["share"]))[:RUNS_SHOWN]
    frames = frames_numbered(src, [d["frame"] + k for d in shown for k in (-1, 0, 1)]
                             + [n for r in long for n in range(r["first"] - 1, r["last"] + 2)])
    width = CELL[0] * ZOOM * 3
    cw = width // 3
    rows: list[np.ndarray] = []
    for r in sorted(long, key=lambda r: r["first"]):
        numbers = list(range(r["first"] - 1, r["last"] + 2))
        if not all(n in frames for n in numbers):
            continue
        rw = width // len(numbers)
        cells = []
        for n in numbers:
            f = frames[n]
            small = cv2.resize(f, (rw, round(f.shape[0] * rw / f.shape[1])), interpolation=cv2.INTER_AREA)
            inside = r["first"] <= n <= r["last"]
            cells.append(_label(small, f"frame {n}" + (" RUN?" if inside else ""), (0, 255, 255) if inside else (255, 255, 255)))
        head = np.full((26, width, 3), 30, np.uint8)
        cv2.putText(head, f"frames {r['first']}-{r['last']} in a row, between two frames that agree ({r['share']:.0%} stands out)"
                          + (" - reads like TAPE damage" if r.get("tape") else ""),
                    (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        rows += [head, _row(cells, width)]
    for d in sorted(shown, key=lambda d: d["frame"]):
        i = d["frame"]
        if not all(i + k in frames for k in (-1, 0, 1)):
            continue
        cells = []
        for k in (-1, 0, 1):
            f = frames[i + k]
            small = cv2.resize(f, (cw, round(f.shape[0] * cw / f.shape[1])), interpolation=cv2.INTER_AREA)
            label = (f"frame {i}: {d['share']:.0%} of it stands out" + (" (one flat tone: a flash?)" if d.get("looks") == "flash" else "")
                     + (" - reads like TAPE damage" if d.get("tape") else "") if k == 0 else f"frame {i + k}")
            cells.append(_label(small, label, (0, 255, 255) if k == 0 else (255, 255, 255)))
        rows.append(_row(cells, width))
    if not rows:
        return None
    head = np.full((30, width, 3), 60, np.uint8)
    text = title or "candidate damaged frames"
    cv2.putText(head, (text + ": damage, or drawn that way?").encode("ascii", "replace").decode()[:150], (6, 21),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    dst.parent.mkdir(parents=True, exist_ok=True)
    imgfile.imwrite(str(dst), np.vstack([head, *rows]), [cv2.IMWRITE_JPEG_QUALITY, 92])
    return dst


def kinds_sheet(src: Path, dst: Path, kinds: list[dict], *, title: str = "") -> Path | None:
    """The kinds of scene a video is made of (probe.scenes), one band per kind: for each example scene the whole
    frame and its busiest line art enlarged. What a kind IS (a night scene, the credits, an action scene) and what
    it wants differently is read here; the measurements only sorted the scenes."""
    src, dst = Path(src), Path(dst)
    fps = media.probe(src)["fps"] or 25.0
    width = CELL[0] * ZOOM * 3
    cw = width // 4
    got = [(k, [(n, fr[0]) for n in k.get("example_frames") or [] for fr in [frames_at(src, n / fps)] if fr]) for k in kinds]
    overlay = overlay_mask([f for _, shown in got for _, f in shown])
    rows: list[np.ndarray] = []
    for k, shown in got:
        if not shown:
            continue
        head = np.full((26, width, 3), 30, np.uint8)
        text = (f"{k['id']}  {k['name']}: {k['scenes']} scenes, {k['share']:.0%} of the video; brightness {k['light']:g}, "
                f"motion {k['motion']:g}, detail {k['detail']:g}"
                + (f"; double images in {k['dips']:.1%} of frames" if k.get("dips", 0) >= 0.005 else ""))
        cv2.putText(head, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        cells = []
        for n, f in shown:
            small = cv2.resize(f, (cw, round(f.shape[0] * cw / f.shape[1])), interpolation=cv2.INTER_AREA)
            # an eighth of the frame's width, at least 120 px: on an HD frame 120 px shows one eyelash
            part = max(cw // ZOOM, f.shape[1] // 8)
            cell = (min(part, f.shape[1]), min(part * 3 // 4, f.shape[0]))
            x, y = busiest(f, cell, ignore=overlay)
            c = f[y:y + cell[1], x:x + cell[0]]
            close = cv2.resize(c, (cw, cw * 3 // 4), interpolation=cv2.INTER_NEAREST)
            cells += [_label(small, f"frame {n} ({n / fps:.0f} s)"), _label(close, f"its lines, x{cw / cell[0]:.1f}")]
        rows += [head, _row(cells, width)]
    if not rows:
        return None
    if title:
        head = np.full((30, width, 3), 60, np.uint8)
        cv2.putText(head, title.encode("ascii", "replace").decode()[:150], (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2, cv2.LINE_AA)
        rows.insert(0, head)
    dst.parent.mkdir(parents=True, exist_ok=True)
    imgfile.imwrite(str(dst), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 92])
    return dst


def recipe_sheet(columns: list[tuple[str, list[np.ndarray | None]]], dst: Path, *, cell: int = 420) -> Path | None:
    """Recipes side by side: one column per recipe (the first is the source), one row per preview. Every picture
    of a row shows the same part of the same moment; each is fitted to `cell` px wide without smoothing when it
    is enlarged."""
    rows_n = max((len(frames) for _, frames in columns), default=0)
    if not rows_n:
        return None
    width = cell * len(columns)
    head = _row([_label(np.zeros((1, cell, 3), np.uint8), name)[:26] for name, _ in columns], width)
    rows = [head]
    overlay = overlay_mask([f for f in columns[0][1] if f is not None])      # a logo is in every preview
    for r in range(rows_n):
        ref = next((frames[r] for _, frames in columns if r < len(frames) and frames[r] is not None), None)
        if ref is None:
            continue
        # the busiest part of the first available picture, as a share of the frame, so that every column shows it
        rx, ry = busiest(ref, (max(16, ref.shape[1] // 4), max(12, ref.shape[0] // 4)),
                         ignore=overlay if overlay is not None and overlay.shape == ref.shape[:2] else None)
        fx, fy = rx / ref.shape[1], ry / ref.shape[0]
        cells = []
        for _, frames in columns:
            f = frames[r] if r < len(frames) else None
            if f is None:
                cells.append(np.full((cell * 3 // 4, cell, 3), 30, np.uint8))
                continue
            h, w = f.shape[:2]
            c = f[round(fy * h):round(fy * h) + h // 4, round(fx * w):round(fx * w) + w // 4]
            interp = cv2.INTER_NEAREST if c.shape[1] < cell else cv2.INTER_AREA
            cells.append(cv2.resize(c, (cell, max(1, round(c.shape[0] * cell / c.shape[1]))), interpolation=interp))
        rows.append(_row(cells, width))
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    imgfile.imwrite(str(dst), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 92])
    return dst
