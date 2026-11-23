"""vs-mlrt engine: engine choice, the pipe reader, the exact-size resize, and the pinned release. No GPU work."""

import io
import json

import numpy as np
import pytest

from vidimg import config, mlrt_install, mlrt_runner
from vidimg.adapters import free_sr


def test_area_shrink_is_a_box_average():
    a = np.arange(4 * 6 * 3, dtype=np.float32).reshape(4, 6, 3)
    out = mlrt_runner.area_shrink(a, 3, 2)
    assert out.shape == (2, 3, 3)
    assert np.allclose(out[0, 0], a[0:2, 0:2].mean(axis=(0, 1)))
    assert mlrt_runner.area_shrink(a, 6, 4) is a                    # factor 1: unchanged
    assert mlrt_runner.area_shrink(a, 4, 3) is None                 # not an integer factor
    assert mlrt_runner.area_shrink(a, 3, 4) is None                 # different factors per axis


def test_pipe_frames_in_any_order_and_stop_at_end():
    w, h = 4, 2
    frames = [np.full((h, w, 3), i, np.uint8) for i in range(3)]
    pipe = mlrt_runner.PipeFrames(io.BytesIO(b"".join(f.tobytes() for f in frames)), w, h)
    assert pipe.get(1)[0, 0, 0] == 1                                 # reads 0 and 1, keeps 0
    assert pipe.get(0)[0, 0, 0] == 0
    assert pipe.get(2)[0, 0, 0] == 2 and pipe.real(2)
    assert pipe.get(5).sum() == 0 and not pipe.real(5)               # past the end: placeholder, not real


@pytest.fixture
def fake_engines(monkeypatch):
    state = {"mlrt": True, "torch_files": True, "cuda": True}
    monkeypatch.setattr(mlrt_install, "available", lambda model=None: (state["mlrt"], "ok" if state["mlrt"] else "vs-mlrt missing"))
    monkeypatch.setattr(free_sr, "_torch_cuda", lambda: state["cuda"])
    monkeypatch.setattr(free_sr, "PYTHON", type("P", (), {"exists": lambda self: state["torch_files"]})())
    monkeypatch.setattr(free_sr, "model_file", lambda key: type("P", (), {"exists": lambda self: state["torch_files"]})())
    monkeypatch.setattr(config, "VIDEO_ENGINE", "auto")
    return state


def test_auto_keeps_pytorch_where_cuda_works(fake_engines):
    assert free_sr.video_engine("realesrnet_x4")[0] == "torch"


def test_auto_uses_mlrt_without_cuda(fake_engines):
    fake_engines["cuda"] = False
    eng, why = free_sr.video_engine("realesrnet_x4")
    assert eng == "mlrt" and "DirectML" in why


def test_auto_falls_back_to_cpu_pytorch_when_nothing_else(fake_engines):
    fake_engines.update(cuda=False, mlrt=False)
    eng, why = free_sr.video_engine("realesrnet_x4")
    assert eng == "torch" and "CPU" in why


def test_forced_engines(fake_engines):
    assert free_sr.video_engine("bsrnet_x4", "mlrt")[0] == "mlrt"
    fake_engines["mlrt"] = False
    assert free_sr.video_engine("bsrnet_x4", "mlrt") == (None, "vs-mlrt missing")
    assert free_sr.video_engine("bsrnet_x4", "torch")[0] == "torch"
    assert free_sr.video_engine("bsrnet_x4", "tensorrt")[0] is None
    fake_engines["torch_files"] = False
    assert free_sr.video_engine("bsrnet_x4", "torch")[0] is None


def test_pinned_release_is_official_and_complete():
    man = mlrt_install.manifest()
    assert set(man["models"]) <= set(free_sr.manifest())             # only models with official weights
    for name, a in man["assets"].items():
        assert a["url"].startswith("https://github.com/AmusementClub/vs-mlrt/releases/download/")
        assert a["url"].endswith(name) and len(a["sha256"]) == 64 and a["size"] > 0
    assert all("==" in p for p in man["pip_pins_sr_venv"])


def test_cli_engine_option_sets_config(monkeypatch):
    from vidimg import cli
    seen = {}
    monkeypatch.setattr(config, "VIDEO_ENGINE", "auto")
    monkeypatch.setattr("vidimg.series.parse_episodes", lambda s: seen.setdefault("engine", config.VIDEO_ENGINE) and None)
    monkeypatch.setattr("vidimg.series.find_episodes", lambda *a, **k: ([], []))
    assert cli.main(["restore-series", "nowhere", "--engine", "mlrt", "--report-only"]) == 2
    assert seen["engine"] == "mlrt"


def test_onnx_export_spec_contract(tmp_path, monkeypatch):
    """The installer hands the exporter official weights and gets checked files back (worker faked)."""
    import subprocess
    calls = {}

    def fake_run(cmd, **kw):
        if "onnx_export.py" in " ".join(map(str, cmd)):
            spec = json.loads(open(cmd[-1], encoding="utf-8").read())
            calls["keys"] = [m["key"] for m in spec["models"]]
            out = "\n".join(json.dumps({"key": m["key"], "ok": True, "MB": 1, "max_diff": 1e-6, "seconds": 0.1})
                            for m in spec["models"])
            return subprocess.CompletedProcess(cmd, 0, out, "")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(mlrt_install, "ROOT", tmp_path)
    monkeypatch.setattr(mlrt_install, "MODELS", tmp_path / "models")
    monkeypatch.setattr(mlrt_install, "VS_PY", tmp_path / "py.exe")
    monkeypatch.setattr(mlrt_install, "SR_PY", tmp_path / "srpy.exe")
    for p in (tmp_path / "py.exe", tmp_path / "srpy.exe"):
        p.write_text("x")
    monkeypatch.setattr("vidimg.setup._fetch", lambda *a, **k: None)
    monkeypatch.setattr(mlrt_install, "available", lambda model=None: (True, "ok"))
    monkeypatch.setattr(mlrt_install, "probe", lambda: {"ok": True})
    monkeypatch.setattr("vidimg.free_install.model_path", lambda key: tmp_path)   # "exists"
    monkeypatch.setattr(subprocess, "run", fake_run)
    r = mlrt_install.install(log=lambda m: None)
    assert r["ok"] and calls["keys"] == mlrt_install.manifest()["models"]
    assert all(r["models"][k]["status"] == "converted" for k in mlrt_install.manifest()["models"])
    assert all(r["models"][k]["status"] == "downloaded" for k in mlrt_install.downloads())   # ONNX-only models: fetched


def test_downloaded_models_are_pinned_and_described():
    """ONNX-only models: a fixed commit, a checksum, their licence, and a block Claude can name."""
    from vidimg import library
    blocks = library.load()[0]
    got = mlrt_install.downloads()
    assert set(got) == {"animejanai_sd_v1beta34", "animejanai_hd_v3_1_balanced"}
    for key, d in got.items():
        assert d["url"].startswith("https://raw.githubusercontent.com/the-database/mpv-AnimeJaNai/")
        assert "/main/" not in d["url"] and d["url"].endswith(".onnx")           # a commit, not a moving branch
        assert len(d["sha256"]) == 64 and d["size"] > 0 and d["scale"] == 2 and "NC" in d["license"]
    for name in ("janai-sd", "janai-hd"):
        b = blocks[name]
        assert b.kind == "model" and b.runner == "onnx" and "drawn_faces" in b.needs
        assert b.spec["model"] in got and b.spec["scales"] == [1, 2] and b.spec["label"] == f"anime {name}"
        assert name in config.VIDEO_MODEL_CHOICES


def test_onnx_only_model_runs_on_mlrt_and_is_fetched_once(tmp_path, monkeypatch):
    monkeypatch.setattr(mlrt_install, "MODELS", tmp_path)
    monkeypatch.setattr(mlrt_install, "VS_PY", tmp_path / "py.exe")
    monkeypatch.setattr(mlrt_install, "PLUGIN", tmp_path / "vsort.dll")
    monkeypatch.setattr(mlrt_install, "SCRIPT", tmp_path / "vsmlrt.py")
    monkeypatch.setattr(mlrt_install, "ROOT", tmp_path)
    (tmp_path / "vsort").mkdir()
    for p in ("py.exe", "vsort.dll", "vsmlrt.py", "vsort/onnxruntime.dll"):
        (tmp_path / p).write_text("x")
    monkeypatch.setattr("shutil.which", lambda name: "ffmpeg")
    fetched = []
    monkeypatch.setattr("vidimg.setup._fetch", lambda url, sha, dst, log: (fetched.append(dst.name), dst.write_text("m")))
    key = "animejanai_sd_v1beta34"
    assert mlrt_install.available(key)[0]                           # not there yet: it is fetched on first use
    assert free_sr.video_engine(key, "torch") == ("mlrt", "an ONNX-only model")      # there are no PyTorch weights
    mlrt_install.ensure(key, log=lambda m: None)
    mlrt_install.ensure(key, log=lambda m: None)
    assert fetched == [f"{key}.onnx"]
    assert not mlrt_install.available("no_such_model")[0]


def test_recipe_names_an_onnx_model(monkeypatch):
    from types import SimpleNamespace
    from vidimg import library, recipe
    from vidimg import pipeline_video as pv
    blocks = library.load()[0]
    p = SimpleNamespace(width=720, height=480, fps=24.0, frames=1000, duration=41.0, flags=[])
    rcp, _ = recipe.validate({"faces": "drawn", "models": ["janai-sd"], "scale": 2})
    why, cands = pv.recipe_candidates(p, rcp, blocks)
    assert [c[0] for c in cands][:2] == ["anime janai-sd", "anime esrgan-v3"] and "janai-sd first" in why
    st = cands[0][1]()[0]
    assert (st.model, st.height, st.engine) == ("animejanai_sd_v1beta34", 960, "mlrt")
    own, _ = recipe.validate({"faces": "drawn", "models": ["janai-hd"], "scale": 1, "model_strict": True})
    assert pv.recipe_candidates(p, own, blocks)[1][0][1]()[0].height == 480          # x2 inside, scaled back
    import pytest
    with pytest.raises(ValueError, match="anime model"):
        recipe.validate({"faces": "real", "models": ["janai-sd"]})
    with pytest.raises(ValueError, match="not x4"):
        recipe.validate({"faces": "drawn", "models": ["janai-sd"], "scale": 4})
    # named for the single-video tool: first, and never part of the order that follows another pick
    _, named = pv.plan_candidates(p, 2, "anime", "janai-sd", strict=False)
    assert named[0][0] == "anime janai-sd" and "anime janai-hd" not in [c[0] for c in named]
    assert not [c for c in pv.plan_candidates(p, 2, "anime")[1] if "janai" in c[0]]


def test_half_float_model_spec(tmp_path, monkeypatch):
    """A file that works in half floats is given half-float frames and is not converted again."""
    from pathlib import Path
    src = Path(mlrt_runner.__file__).read_text(encoding="utf-8")
    assert "vs.RGBH if half else vs.RGBS" in src and "and not half" in src and "astype(np.float32" in src
