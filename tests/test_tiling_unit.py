"""PyTorch worker tiling: whole frames when the card has room (no seams), tiles otherwise."""

from vidimg.tiling import OVERLAP, TILE, plan_frames

GB = 2 ** 30


def test_sd_runs_whole_frames_on_common_cards():
    assert plan_frames(720, 576, True, 16 * GB) == (0, 2)          # 3.3 GB per frame, 9.6 GB budget
    assert plan_frames(720, 576, True, 6 * GB) == (0, 1)           # the minimum card still fits one
    assert plan_frames(640, 480, True, 8 * GB)[0] == 0


def test_large_frames_or_small_cards_use_tiles_one_at_a_time():
    assert plan_frames(1920, 1080, True, 16 * GB) == (TILE, 1)
    assert plan_frames(720, 576, True, 4 * GB) == (TILE, 1)
    assert plan_frames(720, 576, True, None) == (TILE, 1)          # no CUDA: tiles on the CPU


def test_light_models_are_never_tiled():
    assert plan_frames(1920, 1080, False, 4 * GB) == (0, 4)


def test_overlap_is_wide_enough():
    assert OVERLAP >= 32 and TILE >= 8 * OVERLAP
