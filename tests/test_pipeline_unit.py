"""Planning logic of the image pipeline (no engines run)."""

from vidimg import pipeline
from vidimg.probe.image import FaceInfo, ImageProfile


def _profile(w, h, faces=0, min_h=0, jq=80, flags=()):
    p = ImageProfile(path="x.jpg", width=w, height=h, channels=3, bit_depth=8, noise_sigma=0.5, sharpness=200,
                     jpeg_quality=jq, clipped_dark=0, clipped_bright=0, color_cast=1, anime_score=0.5,
                     faces=FaceInfo(faces, min_h, min_h, faces > 0 and min_h < 96))
    p.flags = list(flags)
    return p


def test_scale_follows_size():
    assert pipeline.choose_scale(_profile(612, 816)) == 4
    assert pipeline.choose_scale(_profile(1080, 810)) == 2
    assert pipeline.choose_scale(_profile(4032, 3024)) == 1


def test_small_faces_plan_faithful_models_in_order():
    why, c = pipeline.plan_candidates(_profile(612, 816, faces=5, min_h=39, flags=("low_resolution", "faces", "small_faces")))
    assert [x[0] for x in c] == ["bsrnet", "realesrnet", "swinir"]
    assert all(p["engine"] == "free" and p["scale"] == 4 for _, p in c) and "x4" in why


def test_large_photo_plans_x2_with_the_same_faithful_order():
    _, c = pipeline.plan_candidates(_profile(1600, 1000, jq=55))
    assert [p["model"] for _, p in c] == ["bsrnet_x4", "realesrnet_x4", "swinir_real_x4"]
    assert all(p["scale"] == 2 for _, p in c)          # x4 models, resized to the exact x2 target


def test_deblur_is_opt_in_and_only_for_soft_photos(monkeypatch, tmp_path):
    from dataclasses import replace
    from vidimg import config
    assert config.DEFAULT_POLICY.deblur is False
    calls = []
    monkeypatch.setattr(pipeline.free_sr, "model_available", lambda k: True)
    monkeypatch.setattr(pipeline.free_sr, "upscale", lambda *a, **k: calls.append(k) or None)
    src = tmp_path / "a.jpg"
    # off by default: nothing runs
    assert pipeline.deblur_step(src, src, _profile(600, 400, flags=("soft_or_blurred",)), config.DEFAULT_POLICY, tmp_path) \
        == (src, None, None)
    on = replace(config.DEFAULT_POLICY, deblur=True)
    # requested but the photo is sharp: skipped with a note, no model call
    stage, att, note = pipeline.deblur_step(src, src, _profile(600, 400), on, tmp_path)
    assert stage == src and att is None and "not soft" in note and not calls


def test_mcp_and_cli_pass_the_deblur_opt_in(monkeypatch, tmp_path):
    from vidimg import jobs, mcp_server
    seen = {}
    monkeypatch.setattr(jobs, "start", lambda kind, args, out, label: seen.update(kind=kind, args=args) or {"id": "x"})
    photo = tmp_path / "a.jpg"
    photo.write_bytes(b"x")
    mcp_server.restore_photos([str(photo)], str(tmp_path / "out"), deblur=True)
    assert seen["kind"] == "restore-photos" and seen["args"][-1] == "--deblur"
    mcp_server.restore_photos([str(photo)], str(tmp_path / "out"))
    assert "--deblur" not in seen["args"]
