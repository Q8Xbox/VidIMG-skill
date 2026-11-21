"""Anime routing to Video2X (no engines run)."""

from pathlib import Path

from vidimg import pipeline as pi
from vidimg import pipeline_video as pv
from vidimg.adapters import video2x
from vidimg.probe.video import VideoProfile


class _Faces:
    count, min_height, small = 0, 0, False


class _Img:
    width, height, jpeg_quality, faces = 640, 480, 90, _Faces()

    def __init__(self, flags):
        self.flags = list(flags)


def test_anime_photo_goes_to_video2x_anime_models():
    why, cands = pi.plan_candidates(_Img(["anime_like"]))
    assert [c[1]["processor"] for c in cands] == ["realcugan", "realesrgan"]
    assert all(c[1]["engine"] == "video2x" and c[1]["scale"] == 4 for c in cands)
    assert "anime" in why


def test_anime_never_reaches_video2x_without_content_anime(tmp_path):
    r = video2x.run(tmp_path / "a.png", tmp_path / "o" / "b.png", "realcugan", content="real")
    assert not r.ok and "anime" in r.error


def _vp(h, flags):
    p = VideoProfile(path="v.mp4", width=h * 16 // 9, height=h, fps=24.0, frames=240, duration=10.0, codec="h264",
                     bitrate_kbps=900, bits_per_pixel=0.1, interlaced_ratio=0.0, field_order=None, blockiness=1.0, blur=5.0)
    p.flags = list(flags)
    return p


def test_anime_video_plan():
    why, cands = pv.plan_candidates(_vp(480, ["anime_like"]))
    stages = [c[1]()[0] for c in cands]
    assert all(isinstance(s, video2x.AnimeSR) and s.scale == 2 for s in stages[:2])
    assert [s.processor for s in stages[:2]] == ["realesrgan", "realcugan"]
    assert stages[2].model == "general_x4v3" and stages[2].height == 960   # best VMAF on the cartoon ground truth
    assert not any(s.same_size for s in stages[:2])


def test_a_large_cartoon_is_cleaned_at_its_own_size():
    # no scale asked for: a cartoon above 720p keeps its size (the anime models run at x2 and are scaled back)
    why, cands = pv.plan_candidates(_vp(1080, ["anime_like"]))
    stages = [c[1]()[0] for c in cands]
    assert [c[0] for c in cands] == ["anime esrgan-v3", "anime cugan", "free general-v3"] and "own size" in why
    assert all(isinstance(s, video2x.AnimeSR) and s.scale == 2 and s.same_size for s in stages[:2])
    assert stages[2].model == "general_x4v3" and stages[2].height == 1080
    assert stages[0].describe().endswith(":same-size")
    # asked for on a small cartoon too, and with a named model
    why, cands = pv.plan_candidates(_vp(480, []), scale=1, content="anime", model="cugan-pro")
    assert [c[0] for c in cands] == ["anime cugan-pro"] and cands[0][1]()[0].same_size
    assert pv.plan_candidates(_vp(1080, ["anime_like"]), scale=2)[1][0][1]()[0].same_size is False   # 2x when asked
    assert pv.plan_candidates(_vp(480, []), scale=5, content="anime")[1] == []


def test_content_override_and_exact_scale():
    p = _vp(1080, ["faces"])                       # probe saw "faces" (drawn ones) and no anime flag
    assert not pv.is_anime(p) and pv.is_anime(p, "anime") and not pv.is_anime(_vp(480, ["anime_like"]), "real")
    why, cands = pv.plan_candidates(p, scale=2, content="anime")     # 1080p cartoon forced to x2 -> 2160p
    assert "2160p" in why and cands[0][1]()[0].scale == 2
    assert pv.target_size(_vp(576, []), 2)[0] == 1152               # exactly twice, not "to 1080p"
    import pytest
    with pytest.raises(ValueError):
        pv.is_anime(p, "cartoon")


def test_video2x_hevc10_maps_to_10bit_x265():
    args = video2x.build_args(Path("a.mp4"), Path("b.mp4"), "realesrgan", content="anime", scale=2,
                              codec="libx265", pix_fmt="yuv420p10le", extra={"crf": "16"})
    assert "--pix-fmt" in args and "yuv420p10le" in args and "crf=16" in args


def test_video2x_libplacebo_gets_output_size_and_encodes_without_b_frames():
    args = video2x.build_args(Path("a.mp4"), Path("b.mkv"), "libplacebo", content="anime", scale=2,
                              model="anime4k-v4-a", codec="ffv1", extra={}, out_width=1440, out_height=1152)
    assert args[args.index("-w") + 1] == "1440" and args[args.index("-h") + 1] == "1152"
    import pytest
    with pytest.raises(ValueError):            # Video2X 6.4.0 refuses libplacebo without an output size
        video2x.build_args(Path("a.mp4"), Path("b.mkv"), "libplacebo", content="anime", scale=2, codec="ffv1", extra={})
    assert "--max-b-frames" in video2x.build_args(Path("a.mp4"), Path("b.mkv"), "realesrgan", content="anime", codec="ffv1", extra={}, max_b_frames=0)


def test_av_offset_and_container_choice(monkeypatch):
    from vidimg import media
    assert media.av_offset({"video_start": 0.2002, "audio_start": 0.0}) == 0.2002
    assert media.av_offset({"video_start": None, "audio_start": 0.0}) is None
    fake = {"subtitle_codecs": [], "attachments": 0, "audio_codecs": ["aac"]}
    monkeypatch.setattr(media, "probe", lambda p, exact=False: dict(fake))
    assert pv.output_suffix(Path("a.mp4")) == ".mp4"
    fake["subtitle_codecs"] = ["ass"]                       # ASS breaks the MP4 muxer
    assert pv.output_suffix(Path("a.mkv")) == ".mkv"
    fake.update(subtitle_codecs=[], audio_codecs=["pcm_s16le"])
    assert pv.output_suffix(Path("a.avi")) == ".mkv"


def test_pixel_aspect_is_kept():
    """A lost pixel aspect squeezed PAL 16:15 (4:3 on screen) to 5:4; the aspect gate must see it."""
    from vidimg import media
    assert media._sar({"sample_aspect_ratio": "16:15"}) == "16:15"
    for missing in ({}, {"sample_aspect_ratio": "N/A"}, {"sample_aspect_ratio": "0:1"}):
        assert media._sar(missing) == "1:1"
    pal = {"width": 720, "height": 576, "sar": "16:15"}
    assert abs(media.display_aspect(pal) - 4 / 3) < 1e-9
    lost = {"width": 1440, "height": 1152, "sar": "1:1"}
    assert abs(media.display_aspect(lost) / media.display_aspect(pal) - 1) > 0.05     # 5:4 vs 4:3
    vhs = {"width": 720, "height": 544, "sar": "11696:11565"}                          # owner's Sample06
    assert abs(media.display_aspect({**vhs, "width": 1440, "height": 1088}) - media.display_aspect(vhs)) < 1e-9


def test_named_model_is_the_only_candidate():
    p = _vp(480, ["faces"])
    # default order is unchanged and cugan-pro is not in it
    assert [c[0] for c in pv.plan_candidates(p, scale=2, content="anime")[1]] == [
        "anime esrgan-v3", "anime cugan", "free general-v3"]
    why, cands = pv.plan_candidates(p, scale=2, content="anime", model="cugan-pro")
    assert [c[0] for c in cands] == ["anime cugan-pro"] and "requested" in why
    s = cands[0][1]()[0]
    assert s.processor == "realcugan" and s.model == "models-pro" and s.scale == 2
    # a model that does not apply to the content gives no candidate (the original stays), not a silent fallback
    assert pv.plan_candidates(_vp(480, []), scale=2, content="real", model="cugan-pro")[1] == []
    assert [c[0] for c in pv.plan_candidates(_vp(480, []), scale=2, content="real", model="bsrnet")[1]] == ["free bsrnet"]
    import pytest
    with pytest.raises(ValueError):
        pv.plan_candidates(p, scale=2, content="anime", model="gfpgan")
