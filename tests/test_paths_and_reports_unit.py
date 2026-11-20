"""Image files under any name, probe edge cases, one list of media types, and report pages that survive odd names.
No engines, GPU or FFmpeg."""

from pathlib import Path

import cv2
import numpy as np

from vidimg import auto, imgfile, media, series
from vidimg.probe import image as pi, video as pvid
from vidimg.report import compare


def test_images_are_read_and_written_under_any_name(tmp_path):
    folder = tmp_path / "صور العائلة"                       # OpenCV's own imread/imwrite fail on such a path
    deep = np.arange(8 * 8 * 3, dtype=np.uint16).reshape(8, 8, 3) * 300
    p = folder / "صورة 1.png"
    assert imgfile.imwrite(p, deep) and p.exists()          # the folder is created too
    back = imgfile.imread(p, cv2.IMREAD_UNCHANGED)
    assert back.dtype == np.uint16 and np.array_equal(back, deep)
    assert imgfile.imread(p).dtype == np.uint8              # default flags: 8-bit colour, as cv2.imread
    assert imgfile.imwrite(folder / "صورة.jpg", np.zeros((8, 8, 3), np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 90])
    assert imgfile.imread(folder / "missing.png") is None
    (folder / "empty.png").write_bytes(b"")
    assert imgfile.imread(folder / "empty.png") is None
    assert not imgfile.imwrite(folder / "x.nosuchformat", deep)


def test_no_module_calls_opencv_file_functions_directly():
    root = Path(__file__).parent.parent / "vidimg"
    for f in root.rglob("*.py"):
        text = f.read_text(encoding="utf-8")
        assert "cv2.imread(" not in text and "cv2.imwrite(" not in text, f"{f.name}: use imgfile (any path) instead"


def test_a_black_frame_says_nothing_about_drawn_or_real():
    black = np.full((120, 160, 3), 16, np.uint8)
    assert pi.anime_score(black) == 0.0                     # was 0.85: flat and few colours by definition
    rng = np.random.default_rng(3)
    cartoon = np.zeros((120, 160, 3), np.uint8)
    cartoon[:, :80], cartoon[:, 80:] = (40, 180, 220), (200, 60, 60)
    photo = cv2.GaussianBlur(rng.integers(0, 255, (120, 160, 3), dtype=np.uint8), (0, 0), 1.2)
    assert pi.anime_score(cartoon) > 0.6 > pi.anime_score(photo)
    assert pvid._anime([black, black, cartoon]) == pi.anime_score(cartoon)      # the fades do not vote
    assert pvid._anime([black, black]) == 0.0 and pvid._anime([]) == 0.0


def test_a_handful_of_frames_is_not_called_interlaced(monkeypatch):
    def log(tff, prog):
        return f"Repeated Fields: Neither: {tff + prog} Top: 0 Bottom: 0\nMulti frame detection: TFF: {tff} BFF: 0 Progressive: {prog}"

    monkeypatch.setattr(pvid, "_filter_log", lambda src, vf, seconds: log(3, 1))
    assert pvid._idet(Path("x.mp4"), 1.0)[:2] == (0.0, None)                    # 4 frames: no verdict
    monkeypatch.setattr(pvid, "_filter_log", lambda src, vf, seconds: log(90, 30))
    assert pvid._idet(Path("x.mp4"), 5.0)[:2] == (0.75, "tff")


def test_every_entry_point_uses_one_list_of_media_types():
    from vidimg import mcp_server

    assert auto.VIDEO_EXT is media.VIDEO_EXT is series.VIDEO_EXT is mcp_server.VIDEO_EXT
    assert auto.IMAGE_EXT is media.IMAGE_EXT is mcp_server.IMAGE_EXT
    assert {".vob", ".3gp", ".webm", ".mts"} <= media.VIDEO_EXT and ".heic" not in media.IMAGE_EXT
    assert not media.IMAGE_EXT & media.VIDEO_EXT


def test_review_page_keeps_same_name_photos_apart_and_quotes_links(tmp_path):
    def rec(src):
        return {"source": str(src), "passed": True, "plan": "x2", "attempts": [], "delivered": None, "notes": []}

    a, b = tmp_path / "A" / "IMG #1.jpg", tmp_path / "B" / "IMG #1.jpg"
    sheets = {str(a): tmp_path / "sheets" / "IMG #1-aaaa.jpg", str(b): tmp_path / "sheets" / "IMG #1-bbbb.jpg"}
    page = compare.review_page([rec(a), rec(b)], sheets, tmp_path / "sheets" / "index.html").read_text(encoding="utf-8")
    assert 'src="IMG%20%231-aaaa.jpg"' in page and 'src="IMG%20%231-bbbb.jpg"' in page   # "#" would cut the link
    old_style = compare.review_page([rec(a)], {"IMG #1.jpg": tmp_path / "sheets" / "x.jpg"}, tmp_path / "sheets" / "i2.html")
    assert 'src="x.jpg"' in old_style.read_text(encoding="utf-8")                        # callers that key by name


def test_video_sheets_are_named_after_the_result(tmp_path, monkeypatch):
    from vidimg import pipeline_video as pv

    monkeypatch.setattr(pv.media, "probe", lambda p, exact=False: {"frames": 100})
    monkeypatch.setattr(pv, "_frames_by_index", lambda path, idx, folder, tag, pre_vf=None: {i: Path(f"{tag}{i}") for i in idx})
    made = []
    monkeypatch.setattr(compare, "sheet", lambda s, d, out, title="": made.append(out) or out)
    pv.review_frames(tmp_path / "work" / "ep01" / "prepass.nut", tmp_path / "ep01_restored.mkv", tmp_path / "sheets")
    pv.review_frames(tmp_path / "work" / "ep02" / "prepass.nut", tmp_path / "ep02_restored.mkv", tmp_path / "sheets")
    assert len({p.name for p in made}) == 6 and made[0].name == "ep01_restored_frame1.jpg"
