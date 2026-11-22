"""The clean-up Claude composes per video: the bounded catalogue, interlaced stretches of a progressive file,
the scripts built from a recipe, each pass's own check, and how a recipe travels through the automatic flow."""

import shutil
import subprocess
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest

from vidimg import auto, cleanup
from vidimg.adapters import avisynth as avs
from vidimg.probe import combing

FFMPEG = shutil.which("ffmpeg")
needs_ffmpeg = pytest.mark.skipif(not (FFMPEG and shutil.which("ffprobe")), reason="ffmpeg is not installed")


# ----------------------------------------------------------------------------- the catalogue
def test_a_recipe_is_bounded_by_the_catalogue_and_ordered_by_the_tool():
    got = cleanup.parse(["dehalo:strong", {"name": "Deblock"}, "deinterlace_scenes"])
    assert got == [{"name": "deinterlace_scenes"}, {"name": "deblock", "strength": "medium"},
                   {"name": "dehalo", "strength": "strong"}]
    assert cleanup.parse("deblock:light, dehalo") == [{"name": "deblock", "strength": "light"},
                                                      {"name": "dehalo", "strength": "medium"}]
    assert cleanup.parse(None) == [] and cleanup.parse([]) == [] and cleanup.parse("") == []
    assert cleanup.describe(got) == "deinterlace_scenes, deblock (medium), dehalo (strong)"
    for bad in (["sharpen"], ["deblock:extreme"], ["deblock", "deblock:light"], [{"name": "dehalo", "radius": 9}],
                ["deinterlace_scenes:strong"], [{"strength": "light"}], {"name": "deblock"}, [42]):
        with pytest.raises(ValueError):
            cleanup.parse(bad)
    assert [g["name"] for g in cleanup.guide()] == list(cleanup.ORDER) and all(g["when"] for g in cleanup.guide())


def test_the_script_built_from_a_recipe_passes_the_lint_and_never_sharpens():
    stages = cleanup.picture_stages(cleanup.parse(["dehalo:light", "deblock:strong", "deinterlace_scenes"]))
    assert [type(s).__name__ for s in stages] == ["Deblock", "Dehalo"]           # spans run as their own pass
    text = avs.Script(Path("C:/in/a.mkv"), stages, Path("C:/work/a.lwi"), Fraction(25)).text()
    assert avs.lint(text) == []
    assert text.index("Deblock_QED(quant1=30, quant2=34)") < text.index("DeHalo_alpha(rx=2, ry=2, darkstr=0, brightstr=0.6)")
    assert cleanup.wants_spans(cleanup.parse(["deinterlace_scenes"])) and not cleanup.wants_spans(cleanup.parse(["deblock"]))
    assert cleanup.picture_stages([]) == []


def test_span_deinterlacing_touches_only_the_spans():
    st = avs.DeinterlaceSpans(spans=[[0, 9], [50, 59], [90, 99]], frames=100, tff=False)
    lines = st.lines()
    assert lines[0] == "AssumeBFF()" and "FPSDivisor=2" in lines[2] and st.rate() == 1
    assert lines[3] == ("__fix.Trim(0, end=9) ++ __src.Trim(10, end=49) ++ __fix.Trim(50, end=59) ++ "
                        "__src.Trim(60, end=89) ++ __fix.Trim(90, end=99)")
    mid = avs.DeinterlaceSpans(spans=[[20, 29]], frames=100).lines()[3]
    assert mid == "__src.Trim(0, end=19) ++ __fix.Trim(20, end=29) ++ __src.Trim(30, end=99)"
    text = avs.Script(Path("C:/in/a.mkv"), [st], Path("C:/work/a.lwi"), Fraction(25)).text()
    assert avs.lint(text) == []


# ----------------------------------------------------------------------------- interlaced stretches
def test_interlaced_frames_close_together_form_spans():
    kinds = ["progressive"] * 1000
    for i in list(range(100, 140)) + [150, 152] + list(range(600, 604)) + [800]:    # 800: one odd frame at a cut
        kinds[i] = "tff"
    c = combing.spans_of(kinds)
    assert c.spans == [[98, 154], [598, 605]] and c.span_frames == 65 and c.interlaced == 47
    assert c.is_partly and c.tff and c.to_dict()["is_partly_interlaced"]
    assert not combing.spans_of(["progressive"] * 500 + ["bff"] * 3).is_partly        # too few to be a stretch
    whole = combing.spans_of(["bff"] * 400 + ["progressive"] * 100)
    assert whole.spans and not whole.is_partly and not whole.tff                       # that is an interlaced file
    assert combing.spans_of([]).spans == []
    edge = combing.spans_of(["tff"] * 6 + ["progressive"] * 50 + ["tff"] * 6)
    assert edge.spans == [[0, 7], [54, 61]]                                            # padding stays inside the clip
    touching = combing.spans_of(["tff"] * 5 + ["progressive"] * 14 + ["tff"] * 5 + ["progressive"] * 20)
    assert touching.spans == [[0, 6], [17, 25]]


@needs_ffmpeg
def test_spans_are_found_in_a_real_file_and_the_check_tells_untouched_from_changed(tmp_path):
    from vidimg.guard.video import verify_spans

    # 4 s progressive, then 2 s of true interlace (two moments woven into one frame), then 2 s progressive
    src = tmp_path / "mixed.mkv"
    subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-f", "lavfi", "-i",
                    "smptebars=size=320x240:rate=50:duration=8", "-vf",
                    "scroll=horizontal=0.01,split[a][b];[a]select='not(mod(n\\,2))',setpts=N/25/TB[p];"
                    "[b]tinterlace=mode=interleave_top,setpts=N/25/TB[i];"
                    "[p][i]overlay=enable='between(n\\,100\\,149)'", "-r", "25", "-c:v", "ffv1", "-pix_fmt", "yuv420p",
                    str(src)], check=True, stdin=subprocess.DEVNULL, capture_output=True)
    c = combing.analyze(src)
    assert c.is_partly and len(c.spans) == 1
    a, z = c.spans[0]
    assert a <= 102 and z >= 147
    # the "fix": the same frames, the span replaced by its deinterlaced version
    good = tmp_path / "good.mkv"
    subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-i", str(src), "-vf",
                    f"split[a][b];[b]bwdif=mode=send_frame:deint=all[d];[a][d]overlay=enable='between(n\\,{a}\\,{z})'",
                    "-c:v", "ffv1", "-pix_fmt", "yuv420p", str(good)], check=True, stdin=subprocess.DEVNULL, capture_output=True)
    rep = verify_spans(src, good, spans=c.spans)
    assert rep.passed, [(g.name, g.detail) for g in rep.gates if g.status == "fail"]
    # deinterlacing everything changes the frames outside the span too
    every = tmp_path / "every.mkv"
    subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-i", str(src), "-vf",
                    "bwdif=mode=send_frame:deint=all", "-c:v", "ffv1", "-pix_fmt", "yuv420p", str(every)], check=True,
                   stdin=subprocess.DEVNULL, capture_output=True)
    assert "outside" in verify_spans(src, every, spans=c.spans).failed()
    # and leaving the span as it was does not count as done
    assert "combing" in verify_spans(src, src, spans=c.spans).failed()


def test_a_deblock_that_removes_no_blocks_is_not_kept(monkeypatch, tmp_path):
    from vidimg.guard import GuardReport, video as gv

    monkeypatch.setattr(gv, "verify_video", lambda *a, **k: GuardReport(True, [gv.Gate("color", "pass", 1.0, 4.0, "")]))
    levels = iter([2.3, 1.4, 2.3, 2.3, 2.3, None])
    monkeypatch.setattr(gv, "block_level", lambda *a, **k: next(levels))
    a, b = tmp_path / "a.mkv", tmp_path / "b.mkv"
    assert gv.verify_cleanup(a, b, deblocked=True).passed
    assert gv.verify_cleanup(a, b, deblocked=True).failed() == ["blocks"]       # nothing removed
    assert gv.verify_cleanup(a, b, deblocked=True).failed() == ["blocks"]       # could not be measured
    plain = gv.verify_cleanup(a, b, deblocked=False)                             # dehalo alone: the video guard only
    assert plain.passed and [g.name for g in plain.gates] == ["color"]


# ----------------------------------------------------------------------------- through the automatic flow
def _plan(out: Path) -> dict:
    video = {"content": "anime", "model": "esrgan-v3", "model_strict": False, "scale": 2, "texture": None,
             "mask_tape_noise": False, "preview_only": False, "skip": False, "remove_repeats": False,
             "crop_borders": False, "filters": []}
    plan = {"version": 1, "created": "now", "out": str(out), "inputs": [], "ignored_files": 0, "contact_sheets": [],
            "items": [
                {"id": "v1", "type": "video", "source": "C:/in/clip.mp4", "guide": {}, "decision": dict(video),
                 "decided_by": "analysis", "reasons": [], "state": "pending"},
                {"id": "s1", "type": "series", "source": "C:/in/Show", "folder": "C:/in/Show", "episodes": [1, 2, 3],
                 "analysed_episode": "C:/in/Show/Show - 02.mkv", "guide": {},
                 "decision": {k: v for k, v in video.items() if k != "preview_only"},
                 "decided_by": "analysis", "reasons": [], "state": "pending"}]}
    out.mkdir(parents=True, exist_ok=True)
    auto.save(plan)
    return plan


def test_findings_point_at_the_filter_and_the_closeup():
    prof = SimpleNamespace(flags=["partly_interlaced"], fps=25.0, borders={}, repeats={}, blockiness=2.3,
                           combing={"spans": [[76, 173], [714, 721]], "span_frames": 106, "frames": 16110})
    got = {f["finding"]: f for f in auto._findings(prof)}
    assert "deinterlace_scenes" in got["partly_interlaced"]["decision"] and "2 interlaced stretches" in got["partly_interlaced"]["what"]
    assert "deblock" in got["block_edges"]["decision"] and "2.3" in got["block_edges"]["what"]
    clean = SimpleNamespace(flags=[], fps=25.0, borders={}, repeats={}, blockiness=1.2, combing={})
    assert auto._findings(clean) == []


def test_filters_are_decided_for_videos_and_series_and_reach_the_pipelines(tmp_path, monkeypatch):
    from vidimg import pipeline_video, series

    _plan(tmp_path)
    plan = auto.decide(tmp_path, {"v1": {"filters": ["dehalo:light", "deblock"], "reason": "blocks and a bright rim"},
                                  "s1": {"filters": "deinterlace_scenes"}})
    assert plan["items"][0]["decision"]["filters"] == [{"name": "deblock", "strength": "medium"},
                                                       {"name": "dehalo", "strength": "light"}]
    assert plan["items"][1]["decision"]["filters"] == [{"name": "deinterlace_scenes"}]
    for bad in ({"v1": {"filters": ["sharpen"]}}, {"v1": {"filters": ["deblock:max"]}}, {"s1": {"filters": 7}}):
        with pytest.raises(ValueError):
            auto.decide(tmp_path, bad)
    s = auto.summary(auto.load(tmp_path))
    assert {"deblock", "dehalo", "deinterlace_scenes"} <= {b["block"] for b in s["blocks"]} and "blocks" in s["next"]
    seen = {}

    def video(src, out, **kw):
        seen["video"] = kw["filters"]
        return SimpleNamespace(to_dict=lambda: {"chosen": "anime esrgan-v3", "delivered": None, "passed": False,
                                                "notes": [], "final_gates": {}, "sheets": []})

    def show(folder, out, wanted, **kw):
        seen["series"] = kw["filters"]
        return {}

    monkeypatch.setattr(pipeline_video, "restore_video", video)
    monkeypatch.setattr(series, "restore_series", show)
    monkeypatch.setattr("vidimg.report.compare.video_page", lambda r, p: p)
    plan = auto.run(tmp_path, log=lambda m: None)
    assert seen["video"] == plan["items"][0]["decision"]["filters"] and seen["series"] == [{"name": "deinterlace_scenes"}]
    assert "clean-up: deblock (medium), dehalo (light)" in Path(plan["report"]).read_text(encoding="utf-8")


def test_recipes_can_differ_in_their_filters(tmp_path, monkeypatch):
    from vidimg import pipeline_video
    from vidimg import texture as texture_pass
    from vidimg.report import closeups

    _plan(tmp_path)
    with pytest.raises(ValueError):
        auto.trial(tmp_path, "v1", [{"filters": ["warpsharp"]}, {}], log=lambda m: None)
    seen = []

    def fake(src, out, **kw):
        seen.append(kw["filters"])
        att = SimpleNamespace(label="anime esrgan-v3", previews=3, passed_previews=3, failed=[[], [], []], vmaf=[98.0] * 3)
        return SimpleNamespace(attempts=[att], notes=[], chosen=None)

    monkeypatch.setattr(pipeline_video, "restore_video", fake)
    monkeypatch.setattr(closeups, "frames_at", lambda *a, **k: [])
    monkeypatch.setattr(texture_pass, "apply", lambda *a, **k: SimpleNamespace(ok=False))
    plan = auto.trial(tmp_path, "v1", [{"name": "as it is"}, {"name": "dehalo light", "filters": ["dehalo:light"]},
                                       {"name": "dehalo strong", "filters": "dehalo:strong"}], log=lambda m: None)
    assert seen == [[], [{"name": "dehalo", "strength": "light"}], [{"name": "dehalo", "strength": "strong"}]]
    assert [r["name"] for r in plan["items"][0]["trials"]["recipes"]] == ["as it is", "dehalo light", "dehalo strong"]


@needs_ffmpeg
def test_a_moved_picture_passes_only_when_it_is_the_same_picture_and_steadier(tmp_path):
    import numpy as np
    from vidimg.guard.video import verify_moved, verify_video

    rng = np.random.default_rng(5)
    still = (rng.random((30, 40, 3)) * 255).astype(np.uint8)
    import cv2
    still = cv2.GaussianBlur(cv2.resize(still, (320, 240), interpolation=cv2.INTER_CUBIC), (0, 0), 2)   # smooth detail
    # colourless: the frames are shifted before the 4:2:0 encode, and an odd shift would then change how the colour
    # is sampled (26 dB on saturated random colours), which a stabiliser working on the stored planes does not do
    still = cv2.cvtColor(cv2.cvtColor(still, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    shifts = rng.integers(-3, 4, (50, 2))

    def write(path, frames, vf=None):
        cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s",
               "320x240", "-r", "25", "-i", "-", *(["-vf", vf] if vf else []), "-c:v", "ffv1", "-pix_fmt", "yuv420p", str(path)]
        subprocess.run(cmd, input=b"".join(f.tobytes() for f in frames), check=True, capture_output=True)
        return path

    shaky = write(tmp_path / "shaky.mkv", [np.roll(still, (int(dy), int(dx)), axis=(0, 1)) for dx, dy in shifts])
    steady = write(tmp_path / "steady.mkv", [still] * 50)
    rep = verify_moved(shaky, steady, max_shift=4)
    assert rep.passed, [(g.name, g.value, g.detail) for g in rep.gates if g.status == "fail"]
    assert "shift" in verify_moved(shaky, steady, max_shift=1).failed()                # moved further than allowed
    assert verify_moved(shaky, shaky, max_shift=4).failed() == ["steadier"]            # nothing gained
    soft = write(tmp_path / "soft.mkv", [still] * 50, vf="gblur=sigma=2")
    assert "picture" in verify_moved(shaky, soft, max_shift=4).failed()                # steadier, but not the same picture
    assert not verify_video(shaky, steady).passed                                      # why this needed its own check


@needs_ffmpeg
def test_the_guard_finds_the_original_frames_in_an_interpolated_file(tmp_path):
    import cv2
    import numpy as np
    from vidimg.guard.video import verify_video

    rng = np.random.default_rng(9)
    frames = []
    for k in range(40):                                                    # a changing, colourless picture
        base = cv2.GaussianBlur(cv2.resize((rng.random((12, 16)) * 255).astype(np.uint8), (320, 240),
                                           interpolation=cv2.INTER_CUBIC), (0, 0), 3)
        frames.append(cv2.cvtColor(base, cv2.COLOR_GRAY2BGR))

    def write(path, seq, rate):
        subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                        "-s", "320x240", "-r", str(rate), "-i", "-", "-c:v", "ffv1", "-pix_fmt", "yuv420p", str(path)],
                       input=b"".join(f.tobytes() for f in seq), check=True, capture_output=True)
        return path

    src = write(tmp_path / "src.mkv", frames, 10)
    # x3 the way RIFE writes it: 3n - 2 frames, the originals at every third place, something else in between
    made = []
    for a, b in zip(frames, frames[1:]):
        made += [a, cv2.addWeighted(a, 0.67, b, 0.33, 0), cv2.addWeighted(a, 0.33, b, 0.67, 0)]
    smooth = write(tmp_path / "smooth.mkv", made + [frames[-1]], 30)
    rep = verify_video(src, smooth, interpolated=True, samples=6)
    picture = ("backprojection", "color", "coverage")
    assert not [g.name for g in rep.gates if g.status == "fail" and g.name in picture], \
        [(g.name, g.value, g.detail) for g in rep.gates if g.status == "fail"]
    # the same file one frame late: what sits at the originals' places are created frames, and the guard says so
    shifted = write(tmp_path / "shifted.mkv", made[1:] + [frames[-1], frames[-1]], 30)
    bad = verify_video(src, shifted, interpolated=True, samples=6)
    assert [g.name for g in bad.gates if g.status == "fail" and g.name in picture]


@needs_ffmpeg
def test_the_ghost_check_wants_the_neighbours_in_place_and_the_ghosts_gone(tmp_path):
    import cv2
    import numpy as np
    from vidimg.guard.video import verify_deghost
    from vidimg.probe import ghosts

    rng = np.random.default_rng(4)
    base = cv2.GaussianBlur(cv2.resize((rng.random((9, 12)) * 255).astype(np.uint8), (320, 240), interpolation=cv2.INTER_CUBIC), (0, 0), 2)
    clean = [np.roll(base, 6 * k, axis=1) for k in range(40)]
    mix = lambda x, y: ((x.astype(np.float32) + y.astype(np.float32)) / 2).astype(np.uint8)  # noqa: E731
    ghosted = list(clean)
    for i in (7, 18, 29):
        ghosted[i] = mix(clean[i - 1], clean[i + 1])

    def write(path, seq):
        subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-f", "rawvideo", "-pix_fmt", "gray",
                        "-s", "320x240", "-r", "10", "-i", "-", "-c:v", "ffv1", "-pix_fmt", "yuv420p", str(path)],
                       input=b"".join(f.tobytes() for f in seq), check=True, capture_output=True)
        return path

    src = write(tmp_path / "ghosted.mkv", ghosted)
    found = ghosts.scan(src)
    assert found.error is None and found.frames == 40 and [g[0] for g in found.ghosts] == [7, 18, 29]
    pairs = [[i, j] for i, j, _ in found.ghosts]
    fixed = list(ghosted)
    for i, j in pairs:
        fixed[i] = ghosted[j]
    rep = verify_deghost(src, write(tmp_path / "fixed.mkv", fixed), pairs=pairs)
    assert rep.passed, [(g.name, g.value, g.detail) for g in rep.gates if g.status == "fail"]
    assert "ghosts" in verify_deghost(src, src, pairs=pairs).failed()                 # nothing was replaced
    wrong = list(fixed)
    wrong[7] = clean[20]                                                              # not the neighbour it was to become
    assert "replaced" in verify_deghost(src, write(tmp_path / "wrong.mkv", wrong), pairs=pairs).failed()
    other = list(fixed)
    other[3] = clean[25]                                                              # a frame that was no ghost changed
    bad = verify_deghost(src, write(tmp_path / "other.mkv", other), pairs=pairs, frames=40)
    assert "untouched" in bad.failed()


@needs_ffmpeg
def test_ghosts_are_made_again_in_place_and_every_other_frame_passes_through(tmp_path, monkeypatch):
    import cv2
    import numpy as np
    from vidimg import media, rebuild
    from vidimg.guard.video import verify_rebuilt
    from vidimg.probe import ghosts

    rng = np.random.default_rng(6)
    base = cv2.GaussianBlur(cv2.resize((rng.random((9, 12)) * 255).astype(np.uint8), (320, 240), interpolation=cv2.INTER_CUBIC), (0, 0), 2)
    clean = [np.roll(base, 9 * k, axis=1) for k in range(30)]
    two = lambda x, y: ((x.astype(np.float32) + y.astype(np.float32)) / 2).astype(np.uint8)  # noqa: E731
    ghosted = list(clean)
    for i in (8, 17, 23):
        ghosted[i] = two(np.roll(base, 9 * i - 5, axis=1), np.roll(base, 9 * i + 5, axis=1))

    def write(path, seq):
        subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-f", "rawvideo", "-pix_fmt", "gray",
                        "-s", "320x240", "-r", "10", "-i", "-", "-c:v", "ffv1", "-pix_fmt", "yuv420p10le", str(path)],
                       input=b"".join(f.tobytes() for f in seq), check=True, capture_output=True)
        return path

    def read(path):
        raw = subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-nostdin", "-i", str(path), "-pix_fmt", "gray",
                              "-f", "rawvideo", "-"], capture_output=True, stdin=subprocess.DEVNULL).stdout
        return [np.frombuffer(raw[k * 76800:(k + 1) * 76800], np.uint8).reshape(240, 320) for k in range(len(raw) // 76800)]

    feed = write(tmp_path / "feed.mkv", ghosted)
    found = ghosts.scan_soft(feed)
    assert found.error is None and found.found == [8, 17, 23]

    def rife(src, dst, processor, **kw):                 # stands in for RIFE: the frame between two frames is their mean
        fr = read(src)
        out = []
        for a, b in zip(fr, fr[1:]):
            out += [a, two(a, b)]
        write(dst, out + [fr[-1]])
        return SimpleNamespace(ok=True, error=None)

    monkeypatch.setattr(rebuild.video2x, "run", rife)
    out = tmp_path / "rebuilt" / "feed.mkv"
    info = rebuild.rebuild(feed, out, found.found, tmp_path / "work", log=lambda m: None)
    assert info["ok"] and info["replaced"] == 3 and info["frames"] == 30
    got = read(out)
    assert len(got) == 30
    for i in range(30):
        want = two(ghosted[i - 1], ghosted[i + 1]) if i in (8, 17, 23) else ghosted[i]
        # within the rounding of two grey <-> video-range conversions; another frame is 20 levels and more away
        assert float(np.abs(got[i].astype(int) - want).mean()) < 2.0, i
    rep = verify_rebuilt(feed, out, rebuilt=found.found, frames=12)
    assert rep.failed() == [] or rep.failed() == ["ghosts"]          # a plain mean is still a mix: RIFE's job, not this test's
    assert "untouched" not in rep.failed()
    # a copy in which another frame changed is refused
    spoiled = list(got)
    spoiled[3] = clean[20]
    assert "untouched" in verify_rebuilt(feed, write(tmp_path / "spoiled.mkv", spoiled), rebuilt=found.found, frames=30).failed()
    assert media.probe(out, exact=True)["frames"] == 30
