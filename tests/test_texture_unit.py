"""Texture between the lines: the flat-area measure and texture return, on synthetic images (no video, no GPU)."""

import cv2
import numpy as np

from vidimg import config, media, texture
from vidimg.adapters import video2x


def _drawing(h=240, w=320, seed=0) -> tuple[np.ndarray, np.ndarray]:
    """(textured source, the same with its texture wiped) at source size: flat shapes + painted texture + lines."""
    rng = np.random.default_rng(seed)
    base = np.full((h, w), 140.0, np.float32)
    cv2.rectangle(base, (40, 40), (200, 180), 90, -1)
    cv2.circle(base, (240, 120), 50, 190, -1)
    lines = base.copy()
    cv2.rectangle(lines, (40, 40), (200, 180), 20, 2)
    cv2.circle(lines, (240, 120), 50, 20, 2)
    paint = cv2.GaussianBlur(rng.normal(0, 6, (h, w)).astype(np.float32), (0, 0), 0.8)
    return np.clip(lines + paint, 0, 255).astype(np.uint8), lines.astype(np.uint8)


def _up(img: np.ndarray, k=2) -> np.ndarray:
    return cv2.resize(img, (img.shape[1] * k, img.shape[0] * k), interpolation=cv2.INTER_CUBIC)


def test_wiped_texture_reads_as_flat_and_kept_texture_does_not():
    src, wiped = _drawing()
    ref_flat, out_flat = texture.flat_shares(src, _up(wiped))
    assert out_flat - ref_flat > 0.5                        # "denoised to death"
    ref_flat, out_flat = texture.flat_shares(src, _up(src))
    assert abs(out_flat - ref_flat) < 0.02                  # a plain upscale changes nothing


def test_texture_return_brings_the_source_texture_back_between_the_lines():
    src, wiped = _drawing()
    ref, out = _up(src).astype(np.float32), _up(wiped).astype(np.float32)
    back = texture.return_texture(ref, out, 1.0)
    b8 = np.clip(back + 0.5, 0, 255).astype(np.uint8)
    r, wiped_flat = texture.flat_shares(src, out.astype(np.uint8))
    _, o = texture.flat_shares(src, b8)
    assert o - r < 0.5 * (wiped_flat - r)                   # more than half of the lost texture is back
    # the lines themselves are left as the model drew them
    lines = texture.edges(np.clip(ref, 0, 255).astype(np.uint8))
    core = cv2.erode(lines.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    assert np.abs(back - out)[core].max() < 1.0             # under one level (the mask fades over 2 px)
    # nothing beyond the per-pixel limit, and strength 0 is the model output
    assert np.abs(back - out).max() <= texture.GIVE_MAX + 1e-3
    assert np.array_equal(texture.return_texture(ref, out, 0.0), out)


def test_the_models_own_detail_is_never_subtracted():
    src, _ = _drawing(seed=1)
    ref = _up(src).astype(np.float32)
    rng = np.random.default_rng(5)
    out = ref + cv2.GaussianBlur(rng.normal(0, 20, ref.shape).astype(np.float32), (0, 0), 1.0)   # MORE texture
    back = texture.return_texture(ref, out, 1.0)
    bo, bb = np.abs(texture.band(out)), np.abs(texture.band(back))
    assert np.mean(bb) >= 0.9 * np.mean(bo)


def test_sixteen_bit_scale_matches_eight_bit():
    src, wiped = _drawing(seed=2)
    ref, out = _up(src).astype(np.float32), _up(wiped).astype(np.float32)
    a = texture.return_texture(ref, out, 0.7)
    b = texture.return_texture(ref * 257, out * 257, 0.7, scale=257.0) / 257
    assert np.abs(a - b).max() < 0.05


def test_encoders_and_default():
    assert "lossless" in media.ENCODERS and "hevc10-grain" in media.ENCODERS
    assert "-tune" in media.ENCODERS["hevc10-grain"] and "grain" in media.ENCODERS["hevc10-grain"]
    assert media.CPU_FALLBACK["lossless"][media.CPU_FALLBACK["lossless"].index("-qp") + 1] == "0"
    assert "lossless" in video2x.ENCODER_TO_CODEC and "lossless" in video2x.NVENC_CODEC
    assert 0 <= config.TEXTURE_STRENGTH <= 1
