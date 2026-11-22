"""Comparison clips (report/clip.py): the source beside the result, playing in step, for the person."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from vidimg.report import clip, compare

FFMPEG = shutil.which("ffmpeg")
needs_ffmpeg = pytest.mark.skipif(not (FFMPEG and shutil.which("ffprobe")), reason="ffmpeg is not installed")


def test_the_moments_are_spread_over_the_video_and_stay_inside_it():
    assert clip.moments(1200.0) == [(240.0, 4.0), (600.0, 4.0), (960.0, 4.0)]
    assert clip.moments(5.0) == [(0.0, 5.0)] and clip.moments(0.0)[0][0] == 0.0
    assert all(start + length <= 10.0 for start, length in clip.moments(10.0))


def test_the_places_of_made_frames_are_few_spread_and_start_before_the_frame():
    got = clip.places([250, 251, 252, 1000], 25.0, 60.0)
    assert got == [(9.5, 1.2), (39.5, 1.2)]                              # frames side by side share a place
    many = clip.places(list(range(0, 3000, 100)), 25.0, 200.0)           # thirty places: every fifth is played
    assert len(many) == clip.PLACES_MAX and many[0][0] == 0.0 and many[-1][0] > 90
    assert clip.places([], 25.0, 60.0) == [] and clip.places([5], 0.0, 60.0) == []
    assert clip.places([1490], 25.0, 60.0) == [(59.1, 0.9)] or clip.places([1490], 25.0, 60.0)[0][1] < 1.2


def test_one_moment_is_one_call_with_every_picture_at_the_same_height(monkeypatch):
    monkeypatch.setattr(clip.media, "ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr(clip.media, "display_aspect", lambda info: info["width"] / info["height"])
    monkeypatch.setattr(clip, "_matrix", lambda path, height: "bt601" if height < 720 else "bt709")
    infos = [{"width": 640, "height": 480, "fps": 25.0}, {"width": 1280, "height": 960, "fps": 25.0}]
    args, widths = clip.command([("source", Path("a.mp4")), ("restored", Path("b.mp4"))], infos, Path("h.png"), 12.5, 4.0,
                                Path("out.mp4"), slow=4.0)
    assert widths == [720, 720] and args.count("-ss") == 2 and args[args.index("-ss") + 1] == "12.500"
    graph = args[args.index("-filter_complex") + 1]
    assert "scale=720:540" in graph and "in_color_matrix=bt601" in graph and "in_color_matrix=bt709" in graph
    assert "setpts=4*(PTS-STARTPTS)" in graph and "hstack=inputs=2" in graph and "vstack" in graph
    assert args[args.index("-t", args.index("-loop")) + 1] == "16.000"   # the label bar lasts as long as the slowed clip
    assert "libx264" in args and "yuv420p" in args and "-an" in args     # plays in a browser, no sound
    small = clip.command([("restored", Path("b.mp4"))], [{"width": 320, "height": 240, "fps": 90.0}], Path("h.png"), 0, 1,
                         Path("o.mp4"))
    assert "scale=320:240" in small[0][small[0].index("-filter_complex") + 1] and "fps=60" in small[0][small[0].index("-filter_complex") + 1]


def test_a_clip_that_cannot_be_made_is_simply_missing(tmp_path, monkeypatch):
    assert clip.make([("source", tmp_path / "no.mp4"), ("restored", tmp_path / "none.mp4")], tmp_path / "o.mp4", [(0, 1)]) is None
    (tmp_path / "a.mp4").write_bytes(b"x")
    assert clip.make([("restored", tmp_path / "a.mp4")], tmp_path / "o.mp4", []) is None
    monkeypatch.setattr(clip.media, "probe", lambda p: {})               # a file nothing can be read from
    assert clip.before_after(tmp_path / "a.mp4", tmp_path / "a.mp4", tmp_path / "o.mp4") is None
    assert clip.made_again(tmp_path / "a.mp4", tmp_path / "a.mp4", tmp_path / "o.mp4", [5]) is None


def test_the_report_plays_the_clips_above_the_sheets(tmp_path):
    made = tmp_path / "sheets" / "a_restored_compare.mp4"
    made.parent.mkdir()
    made.write_bytes(b"x")
    page = compare.video_page({"source": "C:/in/a.mp4", "passed": True, "plan": "x2", "chosen": "anime esrgan-v3",
                               "clips": [{"path": str(made), "what": "the source beside the restored video"}],
                               "sheets": []}, tmp_path / "report.html").read_text(encoding="utf-8")
    assert '<video controls loop muted playsinline preload="metadata" src="sheets/a_restored_compare.mp4">' in page
    assert "the source beside the restored video" in page and "Clips to compare" in page
    plain = compare.video_page({"source": "C:/in/a.mp4", "passed": True}, tmp_path / "r2.html").read_text(encoding="utf-8")
    assert "<video" not in plain and "Clips to compare" not in plain


@needs_ffmpeg
def test_two_files_come_out_as_one_clip_side_by_side(tmp_path):
    from vidimg import media

    def write(path, w, h, level):
        frames = [np.full((h, w), level + k, np.uint8) for k in range(50)]
        subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-f", "rawvideo", "-pix_fmt", "gray",
                        "-s", f"{w}x{h}", "-r", "25", "-i", "-", "-c:v", "ffv1", "-pix_fmt", "yuv420p", str(path)],
                       input=b"".join(f.tobytes() for f in frames), check=True, capture_output=True)
        return path

    src, out = write(tmp_path / "src.mkv", 160, 120, 60), write(tmp_path / "out.mkv", 320, 240, 150)
    made = clip.before_after(src, out, tmp_path / "sheets" / "out_compare.mp4")
    assert made and Path(made["path"]).exists() and "source beside" in made["what"]
    info = media.probe(Path(made["path"]))
    assert (info["width"], info["height"]) == (640, 240 + 28) and info["codec"] == "h264" and 1.5 < info["duration"] < 2.5
    slow = clip.made_again(out, out, tmp_path / "sheets" / "again.mp4", [20, 21, 40])
    assert slow and "4 times slower" in slow["what"] and media.probe(Path(slow["path"]))["duration"] > 3.0
