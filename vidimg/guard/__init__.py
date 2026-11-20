"""Fidelity guard (PLAN.md section 6): does the result still show the same photo and the same people?

Initial thresholds come from the ground-truth benchmark (research/bench/RESULTS.md A), where they
separate every faithful output from every damaging one:

gate               faithful outputs           damaging outputs                     threshold
identity (SFace)   0.8456 .. 0.8944           0.787 .. 0.838 (and 0.858 Redefine)  >= 0.84
back-projection    0.9552 .. 0.9681           0.916 .. 0.9489                      >= 0.95
colour (dE2000)    2.63 .. 3.16               2.89 .. 3.24 (does not separate)     <= 4.0 (catches real shifts)
hallucination      HF x19 .. x31 of bicubic   x62 (Recovery v2, Redefine)          warn > x45 at 4x

No single gate separates all ten cases; together they classify all ten correctly. Margins are
thin (Real-ESRGAN: identity 0.838, back-projection 0.9489), so the thresholds are provisional
until `python -m vidimg calibrate` runs on the person's own photos.

Revised after the trial calibration (7 photos, 49 outputs): raw back-projection rejected faithful
outputs of detailed scenes (forest, parking meters, bird: 0.91-0.947), so it now compares low-passed
structure (faithful >= 0.979, threshold 0.97: a gross-change gate; invented texture on scenes
without people is prevented by policy, which blocks generative models by default). False faces in
foliage and on a dog (YuNet 0.67-0.70 vs 0.94 for real faces) no longer enter the identity gate.
Reference for identity and colour is the input upscaled with bicubic (no ground truth exists in use).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
from skimage.color import deltaE_ciede2000, rgb2lab
from skimage.metrics import structural_similarity

from .. import config, faces

THRESHOLDS_FILE = config.STATE_DIR / "guard_thresholds.json"
LANDMARK_JITTER_PX = 2.0
# YuNet confidence for faces the guard checks: real faces scored 0.94, false "faces" in tree
# texture and on a dog 0.67-0.70 (trial calibration), which then failed identity spuriously
FACE_SCORE = 0.8
BACKPROJ_SIGMA = 1.0   # compare structure, not the JPEG noise a good restoration removes
# Back-projection on SSIM's contrast-structure term only (no luminance term). SSIM's luminance term is
# relative, so near black a 3-level darkening reads as a structural collapse (WhatsApp title card:
# 0.79 while visually identical); brightness/colour shifts are the colour gate's job (delta E).
# Enabled 2026-11-16 by the owner after measurement: 0 decision changes on 385 labelled photo and 72
# video outputs; dark-scene and lighting stress identical; black-crush stress 1/161 escapes (accepted cost).
BACKPROJ_STRUCTURE_ONLY = True


def _ssim_cs(a: np.ndarray, b: np.ndarray, data_range: float = 255.0, win: int = 7) -> float:
    """Mean SSIM contrast-structure term, same window/constants/cropping as skimage's default SSIM."""
    a, b = a.astype(np.float64), b.astype(np.float64)
    k = (win, win)
    ux, uy = cv2.blur(a, k, borderType=cv2.BORDER_REFLECT), cv2.blur(b, k, borderType=cv2.BORDER_REFLECT)
    n = win * win / (win * win - 1)                        # sample covariance, as skimage
    vx = (cv2.blur(a * a, k, borderType=cv2.BORDER_REFLECT) - ux * ux) * n
    vy = (cv2.blur(b * b, k, borderType=cv2.BORDER_REFLECT) - uy * uy) * n
    vxy = (cv2.blur(a * b, k, borderType=cv2.BORDER_REFLECT) - ux * uy) * n
    c2 = (0.03 * data_range) ** 2
    cs = (2 * vxy + c2) / (vx + vy + c2)
    p = (win - 1) // 2
    return float(cs[p:-p, p:-p].mean())
# Calibrated 2026-11-15 against GROUND-TRUTH labels (an output is faithful when its true identity is
# within 0.05 of the best output for that photo, damaging when more than 0.10 below): 76 outputs
# from 19 photos (12 of the person's own, adaptive x2/x4 so faces stay >= 40 px, plus 7 samples).
# Identity vs bicubic was the strongest single predictor (balanced accuracy 0.87 at 0.79); with
# these values faithful outputs accepted rose from 26/55 to 45/55, damaging rejected stayed 19/21.
# Two-tier identity: SFace is forgiving of visible distortion (Real-ESRGAN melted eyes at identity
# 0.838), so in the borderline band identity_min..identity_strict the face must ALSO pass the strict
# face back-projection. Chosen for protection: GT faithful accepted 40/55 (vs 45/55 single-tier),
# damaging rejected 19/21, benchmark visual damage caught 3/4 (Redefine, opt-in only, passes).
DEFAULT_THRESHOLDS = {
    "identity_min": 0.79,          # SFace cosine per face, vs bicubic-upscaled input: hard floor
    "identity_strict": 0.84,       # below this, the strict face back-projection applies
    "landmark_drift_max": 0.04,    # warn above this (mean 5-point shift / face width; weak predictor)
    "landmark_drift_fail": 0.08,   # fail only above this
    "face_backproj_min": 0.945,    # raw SSIM inside faces when identity >= identity_strict
    "face_backproj_strict": 0.9525,  # ... when identity is in the borderline band
    "backproj_ssim_min": 0.97,     # low-passed luma SSIM, result downscaled vs input: below -> warning
    # below this the whole-image structure gate FAILS. On the photo ground truth (100 labelled outputs from
    # Topaz, open models and generative ones, 2026-11-15) no damaging output was caught by this gate alone
    # (all >= 0.972; damage was always caught by identity / face structure), while faithful open-model
    # results on the person's real JPEGs scored 0.955-0.967 because they remove compression artefacts the
    # input still has. Same evidence pattern as the video gate (gross-only there).
    "backproj_gross_min": 0.93,
    "delta_e_max": 4.0,            # mean CIEDE2000, result downscaled vs input
    "hf_ratio_per_scale_warn": 11.25,  # Laplacian energy vs bicubic, per unit of scale (45 at 4x)
    "calibrated": False,
}


def thresholds() -> dict:
    if THRESHOLDS_FILE.exists():
        return DEFAULT_THRESHOLDS | json.loads(THRESHOLDS_FILE.read_text(encoding="utf-8"))
    return dict(DEFAULT_THRESHOLDS)


@dataclass
class Gate:
    name: str
    status: str            # pass | warn | fail | n/a
    value: float | None
    threshold: float | None
    detail: str = ""


@dataclass
class GuardReport:
    passed: bool
    gates: list[Gate] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    calibrated: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    def failed(self) -> list[str]:
        return [g.name for g in self.gates if g.status == "fail"]

    def warnings(self) -> list[str]:
        return [g.name for g in self.gates if g.status == "warn"]


def _read(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"cannot read {path}")
    return img


def _luma(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)[:, :, 0]


# --- deblur mode: a faithful deblur must be explainable as "result * some blur = input" -------------
BLUR_SIGMAS = (0.6, 0.9, 1.2, 1.6, 2.0, 2.5, 3.0)
BLUR_LENGTHS = (3, 5, 7, 9, 11, 13, 15, 19)
BLUR_ANGLES = tuple(range(0, 180, 15))


def _blur_kernel(kind: str, a: float, b: float = 0.0, scale: float = 1.0) -> np.ndarray | None:
    """kind 'none' | 'gauss' (a = sigma) | 'line' (a = length, b = angle in degrees), scaled by `scale`."""
    if kind == "none":
        return None
    if kind == "gauss":
        s = max(0.3, a * scale)
        n = int(2 * round(3 * s) + 1)
        g = cv2.getGaussianKernel(n, s)
        return g @ g.T
    length = max(1.0, a * scale)
    n = int(2 * round(length / 2) + 3)
    k = np.zeros((n, n), np.float32)
    c, t = (n - 1) / 2, np.deg2rad(b)
    dx, dy = np.cos(t) * (length - 1) / 2, np.sin(t) * (length - 1) / 2
    cv2.line(k, (int(round((c - dx) * 16)), int(round((c - dy) * 16))), (int(round((c + dx) * 16)), int(round((c + dy) * 16))),
             1.0, 1, cv2.LINE_AA, shift=4)
    return k / max(k.sum(), 1e-9)


def _apply_blur(img: np.ndarray, kern: np.ndarray | None) -> np.ndarray:
    return img if kern is None else cv2.filter2D(img, -1, kern, borderType=cv2.BORDER_REFLECT)


def match_blur(src: np.ndarray, back: np.ndarray, max_side: int = 512) -> tuple[str, float, float]:
    """The blur (none, Gaussian or straight motion streak) that best turns `back` (the result at the
    input's size) into `src`, searched on luma at <= max_side px."""
    k = min(1.0, max_side / max(src.shape[:2]))
    a, b = _luma(src).astype(np.float32), _luma(back).astype(np.float32)
    if k < 1:
        size = (round(src.shape[1] * k), round(src.shape[0] * k))
        a, b = cv2.resize(a, size, interpolation=cv2.INTER_AREA), cv2.resize(b, size, interpolation=cv2.INTER_AREA)
    # blur first, then compare at the small size: kernels are applied at full scale on the small image
    # equivalently by scaling them by k
    cands = [("none", 0.0, 0.0)] + [("gauss", s, 0.0) for s in BLUR_SIGMAS] + \
            [("line", L, ang) for L in BLUR_LENGTHS for ang in BLUR_ANGLES]
    def err(c):
        return float(np.mean((a - _apply_blur(b, _blur_kernel(c[0], c[1], c[2], k))) ** 2))

    best, best_err = cands[0], err(cands[0])
    for c in cands[1:]:
        e = err(c)
        if e < best_err:
            best, best_err = c, e
    # refine around the coarse optimum (real blur rarely sits on the coarse grid)
    if best[0] == "gauss":
        fine = [("gauss", best[1] + d, 0.0) for d in (-0.25, -0.15, -0.08, 0.08, 0.15, 0.25) if best[1] + d > 0.3]
    elif best[0] == "line":
        fine = [("line", best[1] + dl, (best[2] + da) % 180) for dl in (-2, -1, 0, 1, 2) for da in range(-8, 9, 2)
                if best[1] + dl >= 2 and (dl or da)]
    else:
        fine = []
    for c in fine:
        e = err(c)
        if e < best_err:
            best, best_err = c, e
    return best


def _delta_e(a: np.ndarray, b: np.ndarray, max_side: int = 768) -> float:
    k = min(1.0, max_side / max(a.shape[:2]))
    if k < 1:
        size = (round(a.shape[1] * k), round(a.shape[0] * k))
        a, b = cv2.resize(a, size, interpolation=cv2.INTER_AREA), cv2.resize(b, size, interpolation=cv2.INTER_AREA)
    la = rgb2lab(cv2.cvtColor(a, cv2.COLOR_BGR2RGB))
    lb = rgb2lab(cv2.cvtColor(b, cv2.COLOR_BGR2RGB))
    return float(np.mean(deltaE_ciede2000(la, lb)))


UNCHANGED_FACE_SSIM = 0.975
# Below this height in the SOURCE a "face" is a blur with no identity to preserve (15 px background
# people in the person's WhatsApp video failed identity at 0.67-0.71 while visibly unchanged). Kept
# well under 32 px: the face that face recovery genuinely warped in thumbnail (20) was 32 px.
IDENTITY_MIN_SOURCE_FACE = 24
# Same idea for the structure gates (landmarks, face_backprojection): source faces below this height
# are left out of them. 0 = off (every detected face counts). Under measurement, owner's call.
STRUCTURE_MIN_SOURCE_FACE = 0


def _face_ssims(src: np.ndarray, back: np.ndarray, ref_faces, k: float) -> list[float | None]:
    """Raw SSIM of each face region (input scale) between input and result-downscaled."""
    ys, yb = _luma(src), _luma(back)
    vals: list[float | None] = []
    for f in ref_faces:
        x, y, w, h = (int(round(v * k)) for v in f.box)
        pad = int(0.15 * w)
        x0, y0 = max(0, x - pad), max(0, y - pad)
        a, b = ys[y0:y + h + pad, x0:x + w + pad], yb[y0:y + h + pad, x0:x + w + pad]
        vals.append(float(structural_similarity(a, b, data_range=255, win_size=7)) if min(a.shape) >= 7 else None)
    return vals


def _face_backprojection(src: np.ndarray, back: np.ndarray, ref_faces, k: float) -> float | None:
    """Worst raw SSIM over face regions between input and result-downscaled. Inside faces, invented
    detail must not be tolerated: faithful outputs >= 0.9536, every generative/Real-ESRGAN/CUGAN output
    <= 0.9515 over 24 face cases (benchmark + trial)."""
    vals = [v for v in _face_ssims(src, back, ref_faces, k) if v is not None]
    return min(vals) if vals else None


EYE_MIN_FACE_PX = 80   # iris patches on smaller faces are too few pixels to measure colour


def _iris_chroma(img: np.ndarray, face) -> np.ndarray | None:
    """Mean (a, b) chroma of the darker half of both eye patches (iris/pupil, not sclera or skin)."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    r = max(3, int(face.box[2] * 0.07))
    patches = []
    for ex, ey in face.landmarks[:2]:
        x, y = int(ex), int(ey)
        p = lab[max(0, y - r):y + r, max(0, x - r):x + r].reshape(-1, 3)
        if len(p):
            patches.append(p)
    if not patches:
        return None
    p = np.vstack(patches)
    p = p[p[:, 0] <= np.percentile(p[:, 0], 50)]
    return p[:, 1:].mean(axis=0) - 128


EYE_GREEN_DROP = 4.0   # a* drop (red->green) in the iris that, ending below +1, means "eyes turned green"


def eye_green_shift(ref: np.ndarray, out: np.ndarray, ref_faces) -> float | None:
    """Largest drop of iris a* (red-green axis) for warm-toned eyes that end up greenish.

    The plain chroma distance is useless as a gate (sharpening a blurred iris shifts it 7-18 even
    for faithful outputs). What looked wrong on the person's photo 28.jpg was the direction: dark
    brown eyes turned teal with Gigapixel 'low' (a* +5.9 -> -0.1, +3.7 -> -2.3) but not with 'hf'.
    Returns the largest qualifying drop, 0.0 when no measurable eye turned green, None if no face
    is large enough to measure."""
    drops, measured = [], False
    for f in ref_faces:
        if f.box[2] < EYE_MIN_FACE_PX:
            continue
        a, b = _iris_chroma(ref, f), _iris_chroma(out, f)
        if a is None or b is None:
            continue
        measured = True
        if a[0] > 2.0 and b[0] < 1.0:
            drops.append(float(a[0] - b[0]))
    if not measured:
        return None
    return max(drops, default=0.0)


HF_REF_FLOOR = 1.0      # Laplacian variance on 8-bit luma; noise of 0.22 levels (sigma) already gives this much


def _hf(img: np.ndarray) -> float:
    return float(cv2.Laplacian(_luma(img), cv2.CV_64F).var())


SUGGEST = {
    "identity": "lower face recovery strength by 0.2 or switch to a faithful model (low/hf/std); raise blend by 0.15",
    "landmarks": "use a faithful model without face recovery; geometry changed",
    "face_backprojection": "detail was invented inside faces: use hf/std/low, lower face strength or detail",
    "backprojection": "use a more conservative model (hf, std, recovery v1) or lower detail/sharpen",
    "color": "turn off lighting/color adjustments; check colour space and bit depth",
    "hallucination": "lower detail/creativity or avoid generative models (Recovery v2, Redefine)",
}


def verify_image(original: Path, result: Path, *, th: dict | None = None,
                 face_structure_advisory: bool = False, deblur: bool = False,
                 drawn_faces: bool = False) -> GuardReport:
    """face_structure_advisory: the original is a reconstructed reference (a conventionally
    deinterlaced frame), so face structure is a warning when identity is strong; it stays a hard
    gate in the borderline identity band.
    deblur: the result was deliberately deblurred, so it is SUPPOSED to differ from the blurry input.
    The blur that best maps the result back onto the input is found (match_blur) and every comparison
    is made on the re-blurred result: a faithful deblur re-blurs into the input, an invented or changed
    face does not (same principle as judging an upscale on its downscaled result)."""
    th = th or thresholds()
    src, out = _read(Path(original)), _read(Path(result))
    h, w = out.shape[:2]
    scale = w / src.shape[1]
    ref = cv2.resize(src, (w, h), interpolation=cv2.INTER_CUBIC) if (h, w) != src.shape[:2] else src
    back = cv2.resize(out, (src.shape[1], src.shape[0]), interpolation=cv2.INTER_AREA) if (h, w) != src.shape[:2] else out
    gates: list[Gate] = []
    if deblur:
        kind, p1, p2 = match_blur(src, back)
        back = _apply_blur(back, _blur_kernel(kind, p1, p2))
        out = _apply_blur(out, _blur_kernel(kind, p1, p2, scale))
        gates.append(Gate("deblur_match", "pass", None, None,
                          f"result re-blurred with {kind}" + (f" {p1:g}" if kind != "none" else "")
                          + (f" at {p2:g} deg" if kind == "line" else "") + " before comparing"))

    # 1-2. identity and landmark geometry, per face found on the reference
    if faces.available():
        ref_faces = faces.detect(ref, score=FACE_SCORE)
        if ref_faces:
            found = faces.match(ref_faces, faces.detect(out, score=0.4))
            sims, drifts, missing = [], [], 0
            for f, g in zip(ref_faces, found):
                g_used = g or f                     # missing face: embed the same region
                missing += g is None
                sims.append(faces.similarity(faces.embedding(ref, f), faces.embedding(out, g_used)))
                if g is not None:
                    # YuNet landmarks jitter about 2 px regardless of face size: subtract that floor so
                    # small faces (video, old photos) are not failed for detector noise
                    shift_px = float(np.mean(np.linalg.norm(g.landmarks - f.landmarks, axis=1)))
                    drifts.append((len(sims) - 1, max(0.0, shift_px - LANDMARK_JITTER_PX) / max(1.0, f.box[2])))
            # A face whose structure the model left essentially unchanged cannot have lost identity
            # *because of the model*: low scores there come from the measurement (heavy motion blur,
            # a face covered except the eyes). Damaging outputs always changed face structure (per-face
            # SSIM <= 0.971 over the calibration set), so tolerating >= UNCHANGED_FACE_SSIM keeps them.
            fss = _face_ssims(src, back, ref_faces, src.shape[1] / w)
            k_src = src.shape[0] / h
            tolerated = [i for i, (s, q) in enumerate(zip(sims, fss))
                         if s < th["identity_min"] and ((q is not None and q >= UNCHANGED_FACE_SSIM)
                                                        or ref_faces[i].box[3] * k_src < IDENTITY_MIN_SOURCE_FACE)]
            counted = [s for i, s in enumerate(sims) if i not in tolerated]
            worst = min(counted) if counted else min(sims)
            ok = (not counted or worst >= th["identity_min"]) and not missing
            note = f"; {len(tolerated)} unchanged low-score face(s) tolerated" if tolerated else ""
            gates.append(Gate("identity", ("warn" if tolerated else "pass") if ok else "fail",
                              round(worst, 4), th["identity_min"],
                              f"{len(ref_faces)} faces, per face {[round(s, 3) for s in sims]}, missing {missing}{note}"))
            # same principle for geometry: landmark jitter on a face the model left structurally
            # unchanged is detector instability (blurred, fast-moving faces in the person's WhatsApp
            # video drifted 0.10 with exact frame alignment while the faces were untouched)
            unchanged = {i for i, q in enumerate(fss) if q is not None and q >= UNCHANGED_FACE_SSIM}
            tiny = {i for i, f in enumerate(ref_faces) if f.box[3] * k_src < STRUCTURE_MIN_SOURCE_FACE}
            unchanged |= tiny
            counted_drift = [d for i, d in drifts if i not in unchanged]
            drift = max(counted_drift) if counted_drift else 0.0
            lm_status = ("pass" if drift <= th["landmark_drift_max"]
                         else "fail" if drift > th["landmark_drift_fail"] else "warn")
            gates.append(Gate("landmarks", lm_status, round(drift, 4), th["landmark_drift_fail"],
                              f"max mean landmark shift / face width (warn > {th['landmark_drift_max']})"
                              + (f"; {len(drifts) - len(counted_drift)} unchanged face(s) ignored" if len(counted_drift) < len(drifts) else "")))
            green = eye_green_shift(ref, out, ref_faces)
            if green is not None:
                gates.append(Gate("eye_color", "warn" if green > EYE_GREEN_DROP else "pass", round(green, 2),
                                  EYE_GREEN_DROP, "iris red->green drop (warm eyes turning teal)"))
            fbp = _face_backprojection(src, back, [f for i, f in enumerate(ref_faces) if i not in tiny],
                                       src.shape[1] / w)
            if fbp is not None:
                borderline = worst < th["identity_strict"]
                need = th["face_backproj_strict"] if borderline else th["face_backproj_min"]
                status = "pass" if fbp >= need else "fail"
                if status == "fail" and face_structure_advisory and not borderline:
                    status = "warn"   # reference cannot pin a moving face exactly (see verify_video)
                gates.append(Gate("face_backprojection", status, round(fbp, 4), need,
                                  "raw SSIM inside face regions (input scale)"
                                  + ("; strict: identity in borderline band" if borderline else "")
                                  + ("; advisory: deinterlaced reference" if face_structure_advisory else "")))
        else:
            gates.append(Gate("identity", "n/a", None, th["identity_min"], "no faces"))
    else:
        gates.append(Gate("identity", "warn", None, th["identity_min"], "face models missing: identity not checked"))

    # 3. back-projection: the result must still explain the input's structure. Low-passed, because a
    # faithful restoration is *meant* to differ from the input's JPEG noise (raw SSIM failed faithful
    # outputs of detailed scenes at 0.91-0.947; low-passed they score >= 0.979)
    lp = lambda im: cv2.GaussianBlur(_luma(im), (0, 0), BACKPROJ_SIGMA)
    ss = (_ssim_cs(lp(src), lp(back)) if BACKPROJ_STRUCTURE_ONLY
          else float(structural_similarity(lp(src), lp(back), data_range=255)))
    raw = float(structural_similarity(_luma(src), _luma(back), data_range=255))
    gross = th.get("backproj_gross_min", 0.93)
    bp_status = "pass" if ss >= th["backproj_ssim_min"] else ("warn" if ss >= gross else "fail")
    gates.append(Gate("backprojection", bp_status, round(ss, 4), th["backproj_ssim_min"],
                      f"low-passed SSIM(input, result downscaled); raw {raw:.4f}; fails below {gross}"))

    # 4. colour stability
    de = _delta_e(src, back)
    gates.append(Gate("color", "pass" if de <= th["delta_e_max"] else "fail", round(de, 3), th["delta_e_max"],
                      "mean CIEDE2000 (input vs result downscaled)"))

    # 5. hallucination: high-frequency energy added beyond what upscaling plausibly restores
    # the reference's energy has a floor: on a black or flat frame (a fade) it is near zero, and the faintest
    # encoder noise in the result read as a ratio in the hundreds and rejected a good episode
    ratio = _hf(out) / max(HF_REF_FLOOR, _hf(ref))
    limit = th["hf_ratio_per_scale_warn"] * max(1.0, scale)
    status = "pass" if ratio <= limit else ("fail" if ratio > 2 * limit else "warn")
    gates.append(Gate("hallucination", status, round(ratio, 2), round(limit, 2),
                      f"Laplacian energy vs bicubic at scale {scale:.2f}"))

    if drawn_faces:
        # cartoon / anime: SFace and YuNet are trained on real faces, so on drawings the face gates measure the
        # detector, not identity (owner's cartoon samples: anime models' identity "failed" on drawn faces while the
        # ground truth rated them like the others). They stay visible as warnings; the image gates still decide.
        for g in gates:
            if g.name in ("identity", "landmarks", "face_backprojection", "eye_color") and g.status == "fail":
                g.status, g.detail = "warn", g.detail + "; drawn faces (anime content): advisory"
    failed = [g.name for g in gates if g.status == "fail"]
    return GuardReport(not failed, gates, [SUGGEST[n] for n in failed if n in SUGGEST], bool(th.get("calibrated")))
