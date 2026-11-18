"""Series batches: episode numbers in file names, episode ranges, output folder, report page. No engines run."""

from pathlib import Path

import numpy as np
import pytest

from vidimg import series
from vidimg.report import series_report as sr


@pytest.mark.parametrize("name,n", [
    ("[Group] Animal Tales - 01 [WEB-DL][576p][BD24E3DC].mkv", 1),
    ("[Group] Animal Tales - 26 [WEB-DL][576p][1619CF5A].mkv", 26),
    ("The Knight Show S01E05 1080p AMZN WEBRip x264.mkv", 5),
    ("show.s02e112.720p.mkv", 112),
    ("My Show Ep 07.mp4", 7),
    ("My Show EP12 (1991).mkv", 12),
    ("My Show Episode 3.avi", 3),
    ("Some Title - 14v2 [x265].mkv", 14),
    ("Cartoon 08.mkv", 8),
])
def test_episode_number(name, n):
    assert series.episode_number(name) == n


@pytest.mark.parametrize("name", ["Movie 1080p x264.mkv", "Film 2 Part 3.mkv", "trailer.mp4"])
def test_episode_number_unclear(name):
    assert series.episode_number(name) is None


def test_parse_episodes():
    assert series.parse_episodes(None) is None and series.parse_episodes("") is None
    assert series.parse_episodes("1-3") == {1, 2, 3}
    assert series.parse_episodes("1, 3,5-7") == {1, 3, 5, 6, 7}
    with pytest.raises(ValueError):
        series.parse_episodes("7-5")


def test_find_episodes_and_default_folder(tmp_path):
    for n in (1, 2, 3, 30):
        (tmp_path / f"[G] Show - {n:02d} [576p].mkv").write_bytes(b"x")
    (tmp_path / "[G] Show - 02 [576p] (copy).mp4").write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("x")
    (tmp_path / "extras.mkv").write_bytes(b"x")
    eps, notes = series.find_episodes(tmp_path, {1, 2, 3, 4})
    assert [n for n, _ in eps] == [1, 2, 3]
    assert any("two files for episode 02" in x for x in notes)
    assert any("not found: 04" in x for x in notes)
    assert any("extras.mkv" in x for x in notes)
    assert series.default_out_dir(tmp_path, eps, 2) == tmp_path / "VidIMG x2 - Episodes 01-03"
    assert series.default_out_dir(tmp_path, eps[:1], 2).name == "VidIMG x2 - Episode 01"


def test_series_title_and_times():
    names = ["[Group] Animal Tales - 01 [WEB-DL].mkv", "[Group] Animal Tales - 02 [WEB-DL].mkv"]
    assert sr.series_title(names) == "Animal Tales"
    assert sr.series_title(["Knight.Show.S01E01.mkv", "Knight.Show.S01E02.mkv"]) == "Knight Show"
    t = sr.default_times(1440)
    assert len(t) == 10 and t[0] == 72 and t[-1] == int(1440 * 0.9) and t == sorted(t)


def test_blockiness_sees_a_block_grid():
    flat = np.full((64, 64), 128, np.uint8) + np.random.default_rng(0).integers(0, 3, (64, 64)).astype(np.uint8)
    blocky = flat.copy()
    for x in range(0, 64, 8):
        blocky[:, x:x + 8] += (x // 8 % 2) * 12
    assert sr.blockiness(blocky) > 3 * sr.blockiness(flat)


def _row(ep, chosen="anime esrgan-v3", delivered=True, attempts=None):
    gates = {"frames": {"status": "pass", "value": 100}, "backprojection": {"status": "pass", "value": 0.985},
             "color": {"status": "pass", "value": 2.1}, "vmaf": {"status": "pass", "value": 96.5},
             "flicker": {"status": "pass", "value": -0.004}, "identity": {"status": "warn", "value": 0.5}}
    return {"ep": ep, "source": f"S/Show - {ep} [x].mkv", "name": f"Show - {ep} [x].mkv",
            "delivered": f"O/Show - {ep}_restored.mp4" if delivered else None, "passed": delivered,
            "chosen": chosen if delivered else None, "minutes": 30.0, "notes": [], "error": None,
            "align": [0, 0] if delivered else None, "gates": gates if delivered else {},
            "attempts": attempts or [], "report": f"reports/Show - {ep} [x].html",
            "sharp_x": 5.0, "edge_x": 1.4, "psnr": 33.5, "dE": 2.5, "block_o": 1.01, "block_r": 0.97,
            "orig": {"w": 720, "h": 576, "dar": "4:3", "codec": "h264", "pix": "yuv420p", "duration": 1440, "mib": 400, "mbps": 2.3},
            "rest": {"w": 1440, "h": 1152, "dar": "4:3", "codec": "hevc", "pix": "yuv420p10le", "duration": 1440, "mib": 1100, "mbps": 6.4}
            if delivered else {}}


def test_report_page_full_and_partial(tmp_path):
    fallback = [{"label": "anime esrgan-v3", "passed_previews": 2, "previews": 3, "failed": [[], ["flicker"], []]},
                {"label": "anime cugan", "passed_previews": 3, "previews": 3, "failed": [[], [], []]}]
    rows = [_row("01"), _row("02", "anime cugan", attempts=fallback), _row("03")]
    html = sr.page(rows, tmp_path, {"source": "G:/x", "scale": 2, "content": "anime"})
    assert "All 3 episodes were enhanced" in html and "Show, episodes 01–03" in html
    assert "passed 2 of 3 test clips (flicker); anime cugan passed them all" in html
    assert "advisory on drawn faces" in html and 'lang="en"' in html
    assert "reports/Show%20-%2001%20%5Bx%5D.html" in html            # relative, URL-quoted links
    rows[2] = _row("03", delivered=False)
    html = sr.page(rows, tmp_path, {"source": "G:/x", "scale": 2, "content": "anime"})
    assert "2 of 3 episodes were enhanced" in html and "Not delivered: 03" in html and "kept original" in html


def test_fallback_reason_absent_when_first_model_used():
    assert sr._fallback_reason(_row("01", attempts=[{"label": "anime esrgan-v3"}])) is None


def _moving(n=20, start=0):
    rng = np.random.default_rng(1)
    base = rng.integers(0, 255, (48, 64, 3)).astype(np.uint8)
    return [np.roll(base, 3 * (start + i), axis=1) for i in range(n)]


def test_alignment_offset_detects_shift_and_skips_still_scenes():
    a = _moving()
    assert series.decisive_offset(a, _moving()) == 0
    assert series.decisive_offset(a, _moving(start=1)) == -1         # result runs one frame ahead
    assert series.decisive_offset(a, _moving(start=-2)) == 2         # result runs two frames behind
    still = [a[0].copy() for _ in range(20)]
    assert series.decisive_offset(still, still) is None              # identical frames cannot decide
