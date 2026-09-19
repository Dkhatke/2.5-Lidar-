"""
The frame's single spatial ordering — computed once, reused by every stage.

THE IDEA
--------
The map hierarchy is powers of two above a 5 cm base.  If the allocation tile
is also a node of that hierarchy — 1.6 m = 0.05 x 2^5, i.e. 32 x 32 base cells
— then the tile id is *literally a prefix of the level-0 Morton code*::

    tile_code  = code0 >> (2 * TILE_LEVEL)
    level_l_code = code0 >> (2 * l)

So a single argsort of the level-0 codes puts the cloud in an order that is
simultaneously grouped by tile AND grouped by cell at every level, because
right-shifting is monotonic: if code0 is non-decreasing then so is any shift
of it.  Every grouping downstream becomes a run-boundary scan over that one
ordering — no further sorting anywhere.

That replaces four separate 128k-element sorts per frame (tile grouping, the
ground height field, the allocation cost table, and the map insert) with one.

It is also the right design rather than merely the fast one: the allocation
unit is now a cell of the same aligned hierarchy as the map, so a tile
boundary can never fall inside a map cell, and the "no alignment error"
guarantee extends from the map to the allocation grid.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from adaptive_lidar.pipeline.types import ResolutionLevel
from adaptive_lidar.utils.grouping import morton_2d, morton_2d_inverse

BASE = ResolutionLevel.BASE          # 0.05 m

#: Tile = 1.6 m = 0.05 * 2^5.  A node of the map's own hierarchy.
TILE_LEVEL = 5
TILE_SIZE = BASE * (2 ** TILE_LEVEL)


@dataclass
class MortonIndex:
    """One frame's cloud, ordered once by level-0 Morton code."""

    code0: np.ndarray        # (N,) int64 — level-0 code, original point order
    order: np.ndarray        # (N,) int64 — argsort of code0
    sorted_code0: np.ndarray # (N,) int64 — code0[order]

    # Tile grouping (a prefix of code0)
    tile_codes: np.ndarray   # (M,) int64 distinct tile codes, ascending
    tile_starts: np.ndarray  # (M+1,) CSR offsets into `order`
    tile_of_point: np.ndarray  # (N,) int32 dense tile index, original order

    def codes_at_level(self, level: int) -> np.ndarray:
        """Sorted level-``level`` codes for every point, in `order`."""
        return self.sorted_code0 >> (2 * level)

    def group_at_level(self, level: int) -> Tuple[np.ndarray, np.ndarray]:
        """(unique_codes, starts) at ``level`` — a run scan, never a sort."""
        c = self.codes_at_level(level)
        return _runs(c)

    def group_subset_at_level(self, mask_sorted: np.ndarray, level: int):
        """Group a subset of the sorted order at ``level``.

        ``mask_sorted`` selects positions in ``order``.  Because the subset
        preserves the ordering, its level-l codes are still non-decreasing, so
        grouping is again a run scan.

        Returns (unique_codes, starts, subset_positions).
        """
        pos = np.flatnonzero(mask_sorted)
        if pos.size == 0:
            return (np.empty(0, np.int64), np.zeros(1, np.int64), pos)
        c = self.sorted_code0[pos] >> (2 * level)
        uniq, starts = _runs(c)
        return uniq, starts, pos

    @property
    def n_tiles(self) -> int:
        return len(self.tile_codes)

    def tile_centres(self) -> Tuple[np.ndarray, np.ndarray]:
        ix, iy = morton_2d_inverse(self.tile_codes, TILE_LEVEL)
        return ((ix + 0.5) * TILE_SIZE).astype(np.float32), \
               ((iy + 0.5) * TILE_SIZE).astype(np.float32)

    def tile_indices(self) -> Tuple[np.ndarray, np.ndarray]:
        return morton_2d_inverse(self.tile_codes, TILE_LEVEL)


def _runs(sorted_codes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Run boundaries of an already-sorted integer array."""
    n = len(sorted_codes)
    if n == 0:
        return np.empty(0, np.int64), np.zeros(1, np.int64)
    first = np.flatnonzero(np.r_[True, sorted_codes[1:] != sorted_codes[:-1]])
    return sorted_codes[first], np.r_[first, n].astype(np.int64)


def build_morton_index(points: np.ndarray) -> MortonIndex:
    """The one sort per frame."""
    n = len(points)
    if n == 0:
        z64 = np.empty(0, np.int64)
        return MortonIndex(z64, z64, z64, z64, np.zeros(1, np.int64),
                           np.empty(0, np.int32))

    inv = 1.0 / BASE
    ix = np.floor(points[:, 0] * inv).astype(np.int64)
    iy = np.floor(points[:, 1] * inv).astype(np.int64)
    code0 = morton_2d(ix, iy)

    order = np.argsort(code0, kind="quicksort")
    sorted_code0 = code0[order]

    tile_codes, tile_starts = _runs(sorted_code0 >> (2 * TILE_LEVEL))

    tile_of_point = np.empty(n, np.int32)
    sizes = np.diff(tile_starts)
    tile_of_point[order] = np.repeat(
        np.arange(len(tile_codes), dtype=np.int32), sizes)

    return MortonIndex(code0, order, sorted_code0,
                       tile_codes, tile_starts, tile_of_point)
