"""How the PyTorch worker splits frames (plain Python: the worker in tools/sr-venv imports it by path,
the tests import it as vidimg.tiling)."""

from __future__ import annotations

# Tiles: the overlap is cropped away, so it must cover the models' reach. 16 px left seams of up to 134/255
# on RealESRNet (measured against the whole frame); 512 px tiles with 32 px kept them small and cost
# about the same time. Whole frames have no seams at all and are as fast on SD.
TILE, OVERLAP = 512, 32
# GPU memory per input pixel for the heavy RRDB models in fp16, whole frame (measured: 640x480 2.4 GB,
# 720x576 3.2 GB, 1280x720 7.1 GB, in GiB -> about 8.3 KB per pixel); budget = 60% of the card
HEAVY_BYTES_PER_PIXEL, MEMORY_BUDGET = 8_500, 0.6


def plan_frames(w: int, h: int, heavy: bool, total_bytes: int | None, max_batch: int = 4) -> tuple[int, int]:
    """(tile, batch) for video: whole frames, batched as far as the memory budget allows, or tiles."""
    if not heavy:
        return 0, max_batch
    if not total_bytes:
        return TILE, 1
    fit = int(total_bytes * MEMORY_BUDGET // (HEAVY_BYTES_PER_PIXEL * w * h))
    return (0, min(max_batch, fit)) if fit >= 1 else (TILE, 1)
