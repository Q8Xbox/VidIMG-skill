"""Ordinary input that used to stop a run: rotated phone photos, apostrophes in names, decision values given as
text, videos without a usable frame rate. No engines, GPU or FFmpeg."""

import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from vidimg import auto, media
from vidimg.adapters import avisynth as avs


def test_model_and_guard_read_a_rotated_photo_the_same_way(tmp_path):
    from PIL import Image

    from vidimg import guard

    src = tmp_path / "phone.jpg"
    exif = Image.Exif()
    exif[0x0112] = 6                                       # stored on its side, shown upright
    Image.fromarray(np.zeros((40, 80, 3), np.uint8)).save(src, exif=exif)
    worker = cv2.imread(str(src), cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)       # the flags sr_runner uses
    assert worker.shape == guard._read(src).shape == (80, 40, 3)
    flags = (Path(__file__).parent.parent / "vidimg" / "sr_runner.py").read_text(encoding="utf-8")
    assert "cv2.imread(inp, cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)" in flags   # the worker cannot be imported here
    deep = tmp_path / "deep.png"
    cv2.imwrite(str(deep), np.full((4, 4, 3), 40000, np.uint16))
    assert cv2.imread(str(deep), cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH).dtype == np.uint16    # 16-bit kept


def test_concat_line_escapes_apostrophes():
    assert media.concat_line(Path("C:/v/Dad's birthday/chunk_000.mkv")) == \
        "file 'C:/v/Dad'\\''s birthday/chunk_000.mkv'\n"
    assert media.concat_line(Path("C:/v/plain/c.mkv")) == "file 'C:/v/plain/c.mkv'\n"


def _ffprobe_says(monkeypatch, module, stream: dict):
    reply = SimpleNamespace(stdout=json.dumps({"streams": [{"codec_type": "video", **stream}], "format": {}}),
                            stderr="", returncode=0)
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k: reply)


def test_unknown_average_frame_rate_falls_back_to_the_nominal_one(monkeypatch):
    monkeypatch.setattr(media, "ffprobe", lambda: "ffprobe")
    _ffprobe_says(monkeypatch, media, {"avg_frame_rate": "0/0", "r_frame_rate": "25/1", "duration": "4.0"})
    info = media.probe(Path("x.avi"))
    assert info["fps"] == 25.0 and info["frames"] == 100
    _ffprobe_says(monkeypatch, media, {"avg_frame_rate": "0/0", "r_frame_rate": "0/0"})
    assert media.probe(Path("x.avi"))["fps"] == 0.0                       # nothing known: no crash
    _ffprobe_says(monkeypatch, avs, {"avg_frame_rate": "0/0", "r_frame_rate": "30000/1001", "nb_read_packets": "90"})
    fps, packets, _ = avs._rate(Path("x.avi"))
    assert (fps.numerator, fps.denominator, packets) == (30000, 1001, 90)
    _ffprobe_says(monkeypatch, avs, {"avg_frame_rate": "0/0", "r_frame_rate": "0/0"})
    assert avs._rate(Path("x.avi"))[0] == 0


def _plan(out: Path) -> None:
    video = {"content": "auto", "model": "realesrnet", "model_strict": False, "scale": None, "texture": None,
             "mask_tape_noise": False, "preview_only": False, "skip": False}
    out.mkdir(parents=True, exist_ok=True)
    auto.save({"version": 1, "created": "now", "out": str(out), "inputs": [], "ignored_files": 0, "contact_sheets": [],
               "items": [{"id": "v1", "type": "video", "source": "C:/in/clip.mp4", "decision": dict(video),
                          "decided_by": "analysis", "reasons": [], "state": "pending"},
                         {"id": "photos", "type": "photos", "sources": [], "count": 0,
                          "decision": {"deblur": False, "skip": False}, "decided_by": "analysis", "reasons": [],
                          "state": "pending"}]})


def test_decisions_given_as_text_are_stored_in_their_types(tmp_path):
    _plan(tmp_path)
    plan = auto.decide(tmp_path, {"v1": {"scale": "2", "texture": "0.5", "skip": "false", "mask_tape_noise": "True"},
                                  "photos": {"deblur": "yes"}})
    d = {it["id"]: it["decision"] for it in plan["items"]}
    assert d["v1"]["scale"] == 2 and isinstance(d["v1"]["scale"], int)
    assert d["v1"]["texture"] == 0.5 and d["v1"]["skip"] is False and d["v1"]["mask_tape_noise"] is True
    assert d["photos"]["deblur"] is True
    assert auto.decide(tmp_path, {"v1": {"scale": 1.5}})["items"][0]["decision"]["scale"] == 1.5
    for bad in ({"scale": "big"}, {"scale": True}, {"skip": "maybe"}, {"scale": "9"}):
        with pytest.raises(ValueError):
            auto.decide(tmp_path, {"v1": bad})
