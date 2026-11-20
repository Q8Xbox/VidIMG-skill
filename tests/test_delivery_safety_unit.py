"""Delivery safety: cases where a wrong picture could be delivered, the guard passed without looking, or ordinary
input stopped a run. Engines, GPU and video files are replaced by stubs."""

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from vidimg import batch, guard, pipeline_video as pv
from vidimg.adapters import AdapterResult, safe_output_dir
from vidimg.guard import video as gv
from vidimg.probe.video import VideoProfile


def _photo(path: Path, value: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.full((8, 8, 3), value, np.uint8))
    return path


def test_same_name_photos_keep_their_own_model_output(tmp_path, monkeypatch):
    out = tmp_path / "out"
    a, b = _photo(tmp_path / "A" / "IMG_1.png", 40), _photo(tmp_path / "B" / "IMG_1.png", 200)

    def fake_upscale(inputs, folder, *, model, scale, suffix):
        for i in inputs:                                    # the worker names outputs by input stem
            Path(folder).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(Path(folder) / f"{Path(i).stem}{suffix}.png"), cv2.imread(str(i)))

    judged = {}
    monkeypatch.setattr(batch.free_sr, "upscale_images", fake_upscale)
    monkeypatch.setattr(batch, "_judge", lambda it, label, params, output, *r: judged.update({it.src: output}) or True)
    for src in (a, b):                                      # same stem: one call each, same round and settings
        it = batch.Item(src, out)
        it.state.update(status="planned", stage_in=str(src), candidates=[["bsrnet", {"model": "m", "scale": 2}]])
        batch._run_group([it], "bsrnet", {"model": "m", "scale": 2}, 0, out, lambda *_: None)
    assert judged[a] != judged[b]
    assert int(cv2.imread(str(judged[a]))[0, 0, 0]) == 40 and int(cv2.imread(str(judged[b]))[0, 0, 0]) == 200


def test_one_unreadable_photo_does_not_stop_the_batch(tmp_path, monkeypatch):
    good, bad = _photo(tmp_path / "in" / "good.png", 90), tmp_path / "in" / "bad.jpg"
    bad.write_bytes(b"not a picture")
    calls = []

    def prepare(it, policy, log):
        calls.append(it.src.name)
        if it.src == bad:
            raise ValueError("cannot read")
        it.state.update(status="skipped", profile={})

    monkeypatch.setattr(batch, "_prepare", prepare)
    recs = batch.restore_images([bad, good], tmp_path / "out", log=lambda *_: None)
    assert len(recs) == 2 and not recs[0]["delivered"] and "cannot read" in recs[0]["notes"][0]
    batch.restore_images([bad, good], tmp_path / "out", log=lambda *_: None)
    assert calls == ["bad.jpg", "good.png", "bad.jpg"]      # the broken one is tried again, the finished one is not


def _stub_video(monkeypatch, frames_for):
    info = {"frames": 600, "duration": 24.0, "audio_streams": 1, "fps": 25.0, "width": 96, "height": 72, "sar": "1:1"}
    monkeypatch.setattr(gv.media, "probe", lambda p, exact=False: dict(info))
    monkeypatch.setattr(gv.media, "av_offset", lambda i: None)
    monkeypatch.setattr(gv.media, "display_aspect", lambda i: None)
    monkeypatch.setattr(gv.media, "ffprobe", lambda: "ffprobe")
    monkeypatch.setattr(gv.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="", stderr="", returncode=0))
    monkeypatch.setattr(gv, "_vmaf", lambda *a, **k: 95.0)
    monkeypatch.setattr(gv, "_frames_by_index", frames_for)


def test_video_guard_fails_when_no_frames_were_compared(tmp_path, monkeypatch):
    _stub_video(monkeypatch, lambda path, indices, folder, tag, pre_vf=None: {})
    rep = gv.verify_video(tmp_path / "a.mkv", tmp_path / "b.mkv")
    assert not rep.passed and rep.failed() == ["coverage"]


def test_video_guard_counts_the_pairs_it_compared(tmp_path, monkeypatch):
    rng = np.random.default_rng(1)
    img = cv2.GaussianBlur(rng.integers(0, 255, (72, 96, 3), dtype=np.uint8), (0, 0), 2)

    def frames(path, indices, folder, tag, pre_vf=None):
        out = {}
        for i in indices:
            out[i] = Path(folder) / f"{tag}_{i}.png"
            cv2.imwrite(str(out[i]), img)
        return out

    _stub_video(monkeypatch, frames)
    rep = gv.verify_video(tmp_path / "a.mkv", tmp_path / "b.mkv")
    cov = next(g for g in rep.gates if g.name == "coverage")
    assert cov.status == "pass" and cov.value == 6


def test_black_frame_does_not_trip_the_hallucination_gate(tmp_path):
    rng = np.random.default_rng(2)
    src = np.full((90, 120, 3), 16, np.uint8)                                  # a fade to black
    out = np.clip(16 + rng.normal(0, 0.6, (180, 240, 3)), 0, 255).round().astype(np.uint8)   # faint encoder noise
    cv2.imwrite(str(tmp_path / "s.png"), src)
    cv2.imwrite(str(tmp_path / "o.png"), out)
    rep = guard.verify_image(tmp_path / "s.png", tmp_path / "o.png")
    assert next(g for g in rep.gates if g.name == "hallucination").status == "pass"
    loud = np.clip(16 + rng.normal(0, 12, (180, 240, 3)), 0, 255).round().astype(np.uint8)   # invented texture
    cv2.imwrite(str(tmp_path / "l.png"), loud)
    rep = guard.verify_image(tmp_path / "s.png", tmp_path / "l.png")
    assert next(g for g in rep.gates if g.name == "hallucination").status == "fail"


def test_model_output_never_sits_next_to_the_prepass(tmp_path, monkeypatch):
    work = tmp_path / "out" / "work" / "clip"
    work.mkdir(parents=True)
    base = work / "prepass.nut"                             # pre-pass without blend removal: directly in `work`
    base.write_bytes(b"x")
    seen = {}

    def engine(src, dst, stages, encoder, **kw):
        safe_output_dir(dst.parent, src)                    # what every engine adapter does first
        seen["dst"] = dst
        return AdapterResult(False, error="stub")

    monkeypatch.setattr(pv, "run_engine", engine)
    monkeypatch.setattr(pv, "_room_for_lossless", lambda *a: True)
    monkeypatch.setattr(pv, "output_suffix", lambda src, enc=None: ".mp4")      # no probe of the (absent) source file
    prof = VideoProfile(str(base), 720, 576, 25.0, 250, 10.0, "mpeg2video", 4000, 0.4, 0.9, "tff", None, None)
    rec = pv.VideoRecord(str(base), {}, "", chosen="realesrnet")
    rec = pv._full_run(tmp_path / "clip.mpg", tmp_path / "out", work, base, prof, rec, lambda: [], "hevc10",
                       None, lambda *_: None)
    assert seen["dst"].parent != base.parent and rec.notes == ["full run failed: stub"]


# ----------------------------------------------------------------------------- full-length videos
def _same_frames(img):
    def frames(path, indices, folder, tag, pre_vf=None):
        out = {}
        for i in indices:
            out[i] = Path(folder) / f"{tag}_{i}.png"
            cv2.imwrite(str(out[i]), img)
        return out
    return frames


def _texture():
    rng = np.random.default_rng(1)
    return cv2.GaussianBlur(rng.integers(0, 255, (72, 96, 3), dtype=np.uint8), (0, 0), 2)


def test_samples_grow_with_the_length_of_the_video():
    assert [gv.plan_samples(d) for d in (5, 20, 360, 1437, 7200)] == [6, 6, 6, 24, 40]


def test_a_bad_stretch_fails_vmaf_on_a_long_video(tmp_path, monkeypatch):
    _stub_video(monkeypatch, _same_frames(_texture()))
    long = {"frames": 15000, "duration": 600.0, "audio_streams": 1, "fps": 25.0, "width": 96, "height": 72, "sar": "1:1"}
    monkeypatch.setattr(gv.media, "probe", lambda p, exact=False: dict(long))
    asked = {}

    def spread(src, dst, w, h, windows, src_vf=None):
        asked["windows"] = windows
        return [95.0] * (len(windows) - 1) + [30.0]

    monkeypatch.setattr(gv, "_vmaf_spread", spread)
    rep = gv.verify_video(tmp_path / "a.mkv", tmp_path / "b.mkv")
    assert rep.failed() == ["vmaf"]                                  # the mean (86.9) alone is over the floor
    assert len(asked["windows"]) == 4 and asked["windows"][-1][0] > 12000        # windows reach the end
    assert next(g for g in rep.gates if g.name == "coverage").value == 10        # one pair per minute
    monkeypatch.setattr(gv, "_vmaf_spread", lambda *a, **k: [95.0, 94.0, 96.0, 93.0])
    assert gv.verify_video(tmp_path / "a.mkv", tmp_path / "b.mkv").passed


def test_flicker_in_some_samples_is_not_hidden_by_the_median(tmp_path, monkeypatch):
    _stub_video(monkeypatch, _same_frames(_texture()))

    def boiling(pairs):
        calls = {"n": 0}

        def parts(a, b):
            calls["n"] += 1
            pair, is_output = (calls["n"] - 1) // 2, calls["n"] % 2 == 0
            return (0.2 if is_output and pair < pairs else 0.01), 1.0
        return parts

    def flicker(pairs):
        monkeypatch.setattr(gv, "_warp_parts", boiling(pairs))
        return next(g for g in gv.verify_video(tmp_path / "a.mkv", tmp_path / "b.mkv").gates if g.name == "flicker")

    assert flicker(0).status == "pass"
    assert flicker(1).status == "warn"                               # one pair: a hard motion can do it
    two = flicker(2)
    assert two.status == "fail" and two.value == 0.0                 # the median saw nothing


def test_cut_off_video2x_output_is_not_a_result(tmp_path, monkeypatch):
    from vidimg import media
    from vidimg.adapters import video2x

    src, dst = tmp_path / "in" / "a.mkv", tmp_path / "out" / "a.mkv"
    frames = {src: 1000, dst: 700}
    monkeypatch.setattr(media, "nvenc", lambda: False)
    monkeypatch.setattr(media, "probe", lambda p, exact=False: {"frames": frames[Path(p)], "codec": "h264", "fps": 25.0})
    monkeypatch.setattr(video2x, "_timestamp_safe_input", lambda s, d: (s, None))
    monkeypatch.setattr(video2x, "_fix_timestamps", lambda s, d: None)
    monkeypatch.setattr(video2x, "_restore_tail", lambda *a: None)
    monkeypatch.setattr(video2x, "run", lambda s, d, proc, **kw: AdapterResult(True, [d]))
    r = video2x.run_stage(src, dst, video2x.AnimeSR())
    assert not r.ok and "700 of 1000" in r.error
    frames[dst] = 998                                                # the two frames Video2X always drops
    assert video2x.run_stage(src, dst, video2x.AnimeSR()).ok


# ----------------------------------------------------------------------------- remaining guard holes
def test_interpolated_video_is_compared_on_the_frames_it_kept(tmp_path, monkeypatch):
    good, other = _texture(), cv2.flip(_texture(), 1)
    asked = {}

    def frames(output_img):
        def get(path, indices, folder, tag, pre_vf=None):
            asked[tag] = list(indices)
            out = {}
            for i in indices:
                out[i] = Path(folder) / f"{tag}_{i}.png"
                cv2.imwrite(str(out[i]), good if tag == "s" else output_img)
            return out
        return get

    def probe(p, exact=False):                              # the output has twice the frames
        return {"frames": 1200 if Path(p).name == "b.mkv" else 600, "duration": 24.0, "audio_streams": 1,
                "fps": 25.0, "width": 96, "height": 72, "sar": "1:1"}

    _stub_video(monkeypatch, frames(good))
    monkeypatch.setattr(gv.media, "probe", probe)
    rep = gv.verify_video(tmp_path / "a.mkv", tmp_path / "b.mkv", interpolated=True)
    assert rep.passed and next(g for g in rep.gates if g.name == "coverage").value == 6
    assert asked["d"] == [2 * s for s in (50, 150, 250, 350, 450, 550)]     # output frame = 2 x source frame
    monkeypatch.setattr(gv, "_frames_by_index", frames(other))             # a different picture in the output
    rep = gv.verify_video(tmp_path / "a.mkv", tmp_path / "b.mkv", interpolated=True)
    assert not rep.passed and "backprojection" in rep.failed()


def test_chroma_check_fails_without_a_reading(tmp_path, monkeypatch):
    from vidimg.probe import chroma

    monkeypatch.setattr(chroma, "analyze", lambda p: chroma.ChromaShift())      # nothing could be read back
    rep = gv.verify_chroma(tmp_path / "a.nut", tmp_path / "b.nut", moved={"u": [0, -4.5], "v": [0, -4.5]},
                           before={"u": [0, 4.5], "v": [0, 4.5]}, luma_must_match=False)
    assert not rep.passed and rep.failed() == ["chroma_aligned"]


def test_photo_result_keeps_the_colour_profile(tmp_path):
    from PIL import Image, ImageCms

    from vidimg.pipeline import carry_color_profile

    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    src, plain, out = tmp_path / "p.jpg", tmp_path / "n.jpg", tmp_path / "p_restored.png"
    Image.new("RGB", (8, 8), (10, 20, 30)).save(src, icc_profile=icc)
    Image.new("RGB", (8, 8), (10, 20, 30)).save(plain)
    px = np.arange(16 * 16 * 3, dtype=np.uint16).reshape(16, 16, 3) * 80
    cv2.imwrite(str(out), px)                               # 16-bit, as the model worker writes it
    assert carry_color_profile(plain, out) is None          # no profile in the source: file untouched
    assert carry_color_profile(src, out) == "colour profile of the original kept"
    with Image.open(out) as im:
        assert im.info.get("icc_profile") == icc
    assert np.array_equal(cv2.imread(str(out), cv2.IMREAD_UNCHANGED), px)      # pixels are not touched
    assert carry_color_profile(src, out) is None            # already there: not added twice
