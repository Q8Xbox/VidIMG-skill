"""AviSynth+ script generation, lint rules and probe flags (no engines run)."""

from fractions import Fraction
from pathlib import Path

import pytest

from vidimg.adapters import avisynth as avs
from vidimg.probe.video import VideoProfile, _flags


def _script(stages, fps=Fraction(30000, 1001)):
    return avs.Script(Path(r"C:\clips\a b.mp4"), stages, Path(r"C:\ws\a.lwi"), fps, threads=8)


def test_script_shape_and_private_plugins():
    text = _script([avs.Deinterlace()]).text()
    code = [ln for ln in text.splitlines() if ln and not ln.startswith("#")]
    assert code[0] == "ClearAutoloadDirs()"
    assert code[1].startswith("AddAutoloadDir(") and "tools" in code[1] and "avisynth" in code[1]
    assert 'cachefile="C:\\ws\\a.lwi"' in text and "fpsnum=30000, fpsden=1001" in text
    assert code[-3:] == ["ConvertBits(10, dither=1)", "AssumeFPS(60000, 1001)", "Prefetch(8)"]
    assert "SourceMatch=3, Lossless=2" in text and "Sharpness=0.2" in text
    assert avs.lint(text) == []


def test_rates():
    assert _script([avs.Deinterlace()]).out_fps() == Fraction(60000, 1001)
    assert _script([avs.Deinterlace(single_rate=True)]).out_fps() == Fraction(30000, 1001)
    assert _script([avs.IVTC()]).out_fps() == Fraction(24000, 1001)
    assert _script([avs.Degrain(), avs.Deblock()], Fraction(30)).out_fps() == Fraction(30)


def test_order_follows_doom9_consensus():
    names = lambda st: [type(s).__name__ for s in avs.order(st)]
    assert names([avs.Degrain(), avs.Deblock()]) == ["Deblock", "Degrain"]          # deblock on the 8x8 grid first
    # Deblock_QED needs progressive frames: after field handling when there is any
    assert names([avs.Degrain(), avs.Deblock(), avs.Deinterlace()]) == ["Deinterlace", "Deblock", "Degrain"]
    assert names([avs.Repair(), avs.ChromaDenoise(), avs.IVTC()]) == ["IVTC", "ChromaDenoise", "Repair"]


def test_no_sharpening_and_lint_rules():
    assert "contrasharp=false" in avs.Degrain().lines()[0]
    good = _script([avs.Deblock()]).text()
    assert avs.lint(good) == []
    assert any("sharpening" in p for p in avs.lint(good.replace("ConvertBits(16)", "ConvertBits(16)\nLSFmod()")))
    assert any("Sharpness" in p for p in avs.lint(_script([avs.Deinterlace(sharpness=1.0)]).text()))
    assert any("Prefetch" in p for p in avs.lint(good.replace("Prefetch(8)", "Prefetch(8)\nTrim(0, 10)")))
    assert any("first statements" in p for p in avs.lint(good.replace("ClearAutoloadDirs()\n", "")))
    assert any("LoadPlugin" in p for p in avs.lint(good.replace("ConvertBits(16)", 'LoadPlugin("x.dll")\nConvertBits(16)')))
    assert any("cachefile" in p for p in avs.lint(good.replace(", cachefile=", ", cache=")))


def test_string_quoting():
    assert avs.avs_str("C:/a b/c.mp4") == '"C:/a b/c.mp4"'
    assert avs.avs_str('say "hi".mp4') == '"""say "hi".mp4"""'
    with pytest.raises(ValueError):
        avs.avs_str('a"""b')


def _vp(inter, rep, order="tff"):
    return VideoProfile(path="v", width=720, height=480, fps=29.97, frames=100, duration=3.3, codec="h264",
                        bitrate_kbps=5000, bits_per_pixel=0.5, interlaced_ratio=inter, field_order=order,
                        blockiness=1.0, blur=5.0, repeated_field_ratio=rep)


def test_telecine_flag():
    # measured: synthetic 3:2 telecine 0.417-0.458 repeated, true interlace 0.0-0.033, progressive 0
    assert "telecine" in _flags(_vp(1.0, 0.42)) and "interlaced" not in _flags(_vp(1.0, 0.42))
    assert "interlaced" in _flags(_vp(1.0, 0.03)) and "telecine" not in _flags(_vp(1.0, 0.03))
    assert not {"interlaced", "telecine"} & set(_flags(_vp(0.0, 0.0, "progressive")))


def test_colour_sidecar_round_trip(tmp_path):
    from vidimg import media
    nut = tmp_path / "prepass.nut"
    nut.write_bytes(b"")
    kept = media.write_colour_sidecar(nut, {"color_range": "tv", "color_space": "bt709", "color_transfer": "bt709",
                                            "color_primaries": "unknown", "field_order": "tt"})
    assert kept == {"color_range": "tv", "color_space": "bt709", "color_transfer": "bt709"}
    assert media.colour_hint(nut) == kept
    pre, opts = media.colour_args(kept)
    assert pre == "setparams=range=tv:colorspace=bt709:color_trc=bt709,"
    assert opts == ["-color_range", "tv", "-colorspace", "bt709", "-color_trc", "bt709"]
    assert media.colour_args({}) == ("", [])


def test_full_range_is_normalised():
    s = avs.Script(Path("C:/a.mp4"), [avs.Deinterlace()], Path("C:/w/a.lwi"), Fraction(30000, 1001), threads=4,
                   full_range=True).text()
    lines = s.splitlines()
    assert lines[lines.index("ConvertBits(16)") + 1] == 'ColorYUV(levels="PC->TV")'
    assert avs.lint(s) == []
    assert avs.is_full_range({"color_range": "pc"}) and not avs.is_full_range({"color_range": "tv"})


def test_engine_order_and_override(monkeypatch):
    from vidimg import config
    monkeypatch.delenv("VIDIMG_CLASSICAL_ENGINES", raising=False)
    assert config.classical_engines() == ("avisynth", "vapoursynth")
    monkeypatch.setenv("VIDIMG_CLASSICAL_ENGINES", "vapoursynth")
    assert config.classical_engines() == ("vapoursynth",)
    monkeypatch.setenv("VIDIMG_CLASSICAL_ENGINES", "dione")
    with pytest.raises(ValueError):
        config.classical_engines()


def test_vapoursynth_spec_mirrors_stages():
    from vidimg.adapters import vapoursynth as vsb
    assert vsb.supports([avs.Deinterlace(), avs.IVTC()]) and not vsb.supports([avs.Degrain()])
    sp = vsb.spec(Path("C:/a.mp4"), [avs.Deinterlace(tff=False, tr2=2)], Path("C:/w/a.lwi"),
                  {"color_range": "pc"}, Fraction(30000, 1001))
    assert sp["fps"] == [30000, 1001] and sp["full_range"] is True
    st = sp["stages"][0]
    assert st["kind"] == "deinterlace" and st["tff"] is False and st["tr2"] == 2
    assert st["source_match"] == 3 and st["lossless"] == 2 and st["sharpness"] == 0.2
