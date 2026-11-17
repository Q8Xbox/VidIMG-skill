"""Pure tests of shared adapter helpers, Video2X command building and policy (no programs are run)."""

from pathlib import Path

import pytest

from vidimg import config
from vidimg.adapters import PolicyError, safe_output_dir
from vidimg.adapters import video2x

IMG = Path("C:/in/photo.jpg")
OUT = Path("C:/out")


# ---------------------------------------------------------------- shared helpers
def test_safe_output_dir_refuses_input_folder(tmp_path):
    src = tmp_path / "a.png"
    src.write_bytes(b"x")
    with pytest.raises(PolicyError):
        safe_output_dir(tmp_path, src)
    assert safe_output_dir(tmp_path / "out", src).is_dir()


# ---------------------------------------------------------------- Video2X
def test_video2x_refuses_anime_processors_on_real_content():
    with pytest.raises(PolicyError):
        video2x.build_args(Path("a.mp4"), Path("b.mkv"), "realesrgan", content="real")
    args = video2x.build_args(Path("a.mp4"), Path("b.mkv"), "rife", content="real")
    assert args[args.index("--rife-model") + 1] == "rife-v4.26" and args[args.index("-m") + 1] == "2"


def test_video2x_cugan_noise_flag_only_for_valid_levels():
    args = video2x.build_args(Path("a.png"), Path("b.png"), "realcugan", content="anime", scale=4)
    assert not any(a.startswith(("--noise-level", "-n")) for a in args)   # default = no denoise
    args3 = video2x.build_args(Path("a.png"), Path("b.png"), "realcugan", content="anime", scale=2, noise=3)
    assert "--noise-level=3" in args3


def test_video2x_rejects_model_unsupported_by_release():
    with pytest.raises(ValueError):
        video2x.build_args(Path("a.mp4"), Path("b.mkv"), "realesrgan", content="anime", model="realesr-generalv3")


def test_policy_defaults_are_conservative_and_free():
    p = config.DEFAULT_POLICY
    assert not p.allow_generative and not p.lighting_and_color and not p.frame_interpolation
    assert p.image_bit_depth == 16 and p.identity_threshold == 0.79
    assert not hasattr(p, "gigapixel_face_strength") and not hasattr(config, "media_engine")
