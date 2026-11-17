"""Phase 5 (batches and long videos): chunk plans, checkpoint identity, job resume rules. No engines run."""

from pathlib import Path

import pytest

from vidimg import pipeline_video as pv


@pytest.mark.parametrize("dur", [10, 125, 150, 300, 301, 1000, 3356])
def test_chunk_plan_covers_the_video_exactly(dur):
    plan = pv.chunk_plan(dur)
    assert plan[0][0] == 0
    assert abs(sum(length for _, length in plan) - dur) < 1e-6
    for (s1, l1), (s2, _) in zip(plan, plan[1:]):
        assert abs(s1 + l1 - s2) < 1e-9                     # contiguous, no gap or overlap
    assert all(length >= pv.CHUNK_SECONDS * 0.25 or len(plan) == 1 for _, length in plan)


def test_long_video_threshold_and_chunk_count():
    assert len(pv.chunk_plan(3356)) == 28
    assert pv.LONG_VIDEO_SECONDS >= pv.CHUNK_SECONDS


def test_batch_item_key_is_stable_and_folder_aware(tmp_path):
    from vidimg import batch
    a, b = tmp_path / "x" / "IMG_1.jpg", tmp_path / "y" / "IMG_1.jpg"
    assert batch._key(a) == batch._key(Path(str(a)))
    assert batch._key(a) != batch._key(b) and batch._key(a).startswith("IMG_1-")


def test_batch_item_checkpoint_roundtrip(tmp_path):
    from vidimg import batch
    src = tmp_path / "p.jpg"
    src.write_bytes(b"x")
    it = batch.Item(src, tmp_path / "out")
    assert it.state["status"] == "new" and not it.done
    it.state.update(status="delivered", delivered=str(tmp_path / "out" / "p_restored.png"))
    it.save()
    again = batch.Item(src, tmp_path / "out")
    assert again.done and again.state["delivered"].endswith("p_restored.png")
    other = batch.Item(tmp_path / "q.jpg", tmp_path / "out")      # another photo never reads p's state
    assert other.state["status"] == "new"


def test_resume_refuses_running_or_finished_jobs(monkeypatch):
    from vidimg import jobs
    monkeypatch.setattr(jobs, "status", lambda j: {"state": "running"})
    with pytest.raises(ValueError):
        jobs.resume("20261115000000-abcdef")
    monkeypatch.setattr(jobs, "status", lambda j: {"state": "done"})
    with pytest.raises(ValueError):
        jobs.resume("20261115000000-abcdef")


def test_chunk_boundaries_snap_to_nearby_scene_cuts():
    plan = pv.chunk_plan(600, cuts=[115.2, 131.0, 250.0, 333.3, 470.0])
    starts = [round(s, 1) for s, _ in plan]
    assert starts == [0.0, 115.2, 250.0, 333.3, 470.0]
    assert abs(sum(length for _, length in plan) - 600) < 1e-6
    far = pv.chunk_plan(600, cuts=[50.0, 300.0])                 # no cut within +-30 s of 120/240/360/480
    assert [s for s, _ in far] == [0, 120, 240, 360, 480]
