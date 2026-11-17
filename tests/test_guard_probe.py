"""Guard and analysis tests on synthetic images and the benchmark photo (no GPU engines run)."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from vidimg import config, guard
from vidimg.calibrate import _best_threshold
from vidimg.guard.video import _warp_error
from vidimg.probe import image as probe

GT = config.PROJECT_ROOT / "research" / "bench" / "images" / "gt.png"


def _gt_img():
    """The benchmark photo stays private (not in the public repository): those tests skip without it."""
    if not GT.exists():
        pytest.skip("benchmark photo research/bench/images/gt.png is not published")
    return cv2.imread(str(GT))


@pytest.fixture(scope="module")
def pair(tmp_path_factory):
    """Low-res input (400x250, JPEG q60) and a faithful stand-in result (bicubic x4)."""
    d = tmp_path_factory.mktemp("pair")
    gt = _gt_img()
    lr = cv2.resize(gt, (400, 250), interpolation=cv2.INTER_AREA)
    lr_p = d / "lr.jpg"
    cv2.imwrite(str(lr_p), lr, [cv2.IMWRITE_JPEG_QUALITY, 60])
    up = cv2.resize(cv2.imread(str(lr_p)), (1600, 1000), interpolation=cv2.INTER_CUBIC)
    up_p = d / "up.png"
    cv2.imwrite(str(up_p), up)
    return d, lr_p, up_p, up


def _gates(rep):
    return {g.name: g for g in rep.gates}


# ------------------------------------------------------------------ analysis
def test_noise_sigma_estimate_is_close():
    rng = np.random.default_rng(0)
    flat = np.full((512, 512), 128.0)
    for sigma in (3.0, 8.0):
        noisy = np.clip(flat + rng.normal(0, sigma, flat.shape), 0, 255).astype(np.uint8)
        assert abs(probe.noise_sigma(noisy) - sigma) < 0.6


def test_jpeg_quality_estimate(tmp_path):
    img = _gt_img()
    for q in (50, 75, 92):
        p = tmp_path / f"q{q}.jpg"
        cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, q])
        assert abs(probe.jpeg_quality(p) - q) <= 3
    assert probe.jpeg_quality(GT) is None


def test_anime_score_separates_cartoon_from_photo():
    cart = np.full((720, 1280, 3), (235, 206, 135), np.uint8)
    for i, c in enumerate([(60, 60, 200), (40, 180, 90), (200, 120, 40)]):
        cv2.circle(cart, (300 + i * 330, 380), 140, c, -1)
        cv2.circle(cart, (300 + i * 330, 380), 140, (20, 20, 20), 6)
    assert probe.anime_score(cart) >= 0.6
    assert probe.anime_score(_gt_img()) < 0.6


def test_analyze_finds_faces_and_flags(pair):
    _, lr_p, _, _ = pair
    prof = probe.analyze(lr_p)
    assert prof.faces.count == 3 and prof.faces.small
    assert {"low_resolution", "heavy_jpeg", "faces", "small_faces"} <= set(prof.flags)


# ------------------------------------------------------------------ image guard
def test_faithful_stand_in_passes(pair):
    _, lr_p, up_p, _ = pair
    rep = guard.verify_image(lr_p, up_p)
    g = _gates(rep)
    assert rep.passed, rep.failed()
    assert g["identity"].value > 0.95 and g["backprojection"].value > 0.98


def test_color_shift_fails_color_gate(pair, tmp_path):
    _, lr_p, _, up = pair
    lab = cv2.cvtColor(up, cv2.COLOR_BGR2LAB).astype(np.int16)
    lab[:, :, 1] += 14            # strong magenta cast, like an unwanted "balance color"
    shifted = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    p = tmp_path / "cast.png"
    cv2.imwrite(str(p), shifted)
    assert _gates(guard.verify_image(lr_p, p))["color"].status == "fail"


def test_geometric_shift_fails_backprojection(pair, tmp_path):
    _, lr_p, _, up = pair
    moved = cv2.warpAffine(up, np.float32([[1, 0, 12], [0, 1, 8]]), (up.shape[1], up.shape[0]),
                           borderMode=cv2.BORDER_REFLECT)
    p = tmp_path / "moved.png"
    cv2.imwrite(str(p), moved)
    rep = guard.verify_image(lr_p, p)
    assert not rep.passed and "backprojection" in rep.failed()


def test_mirrored_result_fails_identity_or_landmarks(pair, tmp_path):
    _, lr_p, _, up = pair
    p = tmp_path / "mirror.png"
    cv2.imwrite(str(p), cv2.flip(up, 1))
    rep = guard.verify_image(lr_p, p)
    assert not rep.passed and ({"identity", "landmarks", "backprojection"} & set(rep.failed()))


def test_suggestions_follow_failed_gates(pair, tmp_path):
    _, lr_p, _, up = pair
    p = tmp_path / "noisy.png"
    rng = np.random.default_rng(1)
    cv2.imwrite(str(p), np.clip(up + rng.normal(0, 40, up.shape), 0, 255).astype(np.uint8))
    rep = guard.verify_image(lr_p, p)
    assert not rep.passed and len(rep.suggestions) == len([n for n in rep.failed() if n in guard.SUGGEST])


# ------------------------------------------------------------------ video helpers and calibration
def test_warp_error_zero_for_identical_frames():
    f = _gt_img()
    assert _warp_error(f, f) < 1e-3


def test_best_threshold_separates():
    vals = [(0.88, True), (0.86, True), (0.85, True), (0.83, False), (0.79, False)]
    t, acc = _best_threshold(vals, "min")
    assert acc == 1.0 and 0.83 < t <= 0.85
    t2, acc2 = _best_threshold([(2.0, True), (2.5, True), (5.0, False)], "max")
    assert acc2 == 1.0 and 2.5 <= t2 < 5.0


def test_backprojection_structure_term_matches_skimage_and_ignores_dark_level_shift():
    from skimage.metrics import structural_similarity
    rng = np.random.default_rng(0)
    img = cv2.GaussianBlur(rng.uniform(0, 255, (120, 160)), (0, 0), 3)
    noisy = img + rng.normal(0, 4, img.shape)
    # with equal means the full SSIM equals the contrast-structure term (same windows and constants)
    shifted = noisy - noisy.mean() + img.mean()
    assert abs(guard._ssim_cs(img, shifted) - structural_similarity(img, shifted, data_range=255)) < 2e-3
    # a near-black frame darkened by 3 levels: full SSIM collapses, structure is unchanged
    dark = img * 0.02 + 3                                  # levels ~3-8, like the title card
    darker = np.clip(dark - 3, 0, 255)
    assert structural_similarity(dark, darker, data_range=255) < 0.9
    assert guard._ssim_cs(dark, darker) > 0.98
    # real structural change is still caught
    assert guard._ssim_cs(img, np.roll(img, 12, axis=1)) < 0.9


# ------------------------------------------------------------------ deblur mode
def test_match_blur_recovers_the_blur_family():
    rng = np.random.default_rng(1)
    img = cv2.cvtColor(cv2.GaussianBlur(rng.uniform(0, 255, (256, 256)).astype(np.float32), (0, 0), 2).astype(np.uint8),
                       cv2.COLOR_GRAY2BGR)
    g = guard._apply_blur(img, guard._blur_kernel("gauss", 1.6))
    kind, s, _ = guard.match_blur(g, img)
    assert kind == "gauss" and abs(s - 1.6) <= 0.3
    m = guard._apply_blur(img, guard._blur_kernel("line", 11, 60))
    kind, length, ang = guard.match_blur(m, img)
    assert kind == "line" and abs(length - 11) <= 2 and min(abs(ang - 60), 180 - abs(ang - 60)) <= 10
    assert guard.match_blur(img, img)[0] == "none"


def test_deblur_mode_accepts_a_true_deblur_and_still_rejects_a_changed_image(tmp_path):
    sharp = cv2.resize(_gt_img(), (800, 500), interpolation=cv2.INTER_AREA)
    blurry = guard._apply_blur(sharp, guard._blur_kernel("line", 9, 30))
    bp, sp, shifted = tmp_path / "blurry.png", tmp_path / "sharp.png", tmp_path / "shifted.png"
    cv2.imwrite(str(bp), blurry)
    cv2.imwrite(str(sp), sharp)
    cv2.imwrite(str(shifted), np.roll(sharp, 25, axis=1))
    rep = guard.verify_image(bp, sp, deblur=True)          # perfect deblur: re-blurs into the input
    assert rep.passed and "line" in _gates(rep)["deblur_match"].detail
    bad = guard.verify_image(bp, shifted, deblur=True)     # geometry changed: no blur explains it
    assert not bad.passed
