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
    assert all(v["status"] == "converted" for v in r["models"].values())
