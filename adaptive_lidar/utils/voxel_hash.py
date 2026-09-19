"""
Voxel hash — current-frame spatial query index, CSR-structured.

IMPORTANT ARCHITECTURAL DISTINCTION:
  This is a PER-FRAME QUERY INDEX (built in S1).
  It is NOT the persistent adaptive map (that lives in mapping/adaptive_map.py).
  Do not merge these two concepts.  The query index is thrown away every frame;
  the map accumulates across frames and is what the whole project is about.

STORAGE
-------
Three flat arrays, not a Python dict::

    unique_keys : (M,)   int64, sorted        — one per occupied voxel
    offsets     : (M+1,) int64                — CSR boundaries
    sorted_idx  : (N,)   int64                — point indices, grouped

Lookup is ``np.searchsorted`` on ``unique_keys`` — O(log M) for one query and
fully vectorised for a batch.  The old implementation built a
``dict[(ix,iy,iz)] -> list[int]`` with a Python loop over every point, which
cost ~100 ms at 120k points and allocated ~120k list objects.
"""
from __future__ import annotations

from typing import Iterator, List, Tuple

import numpy as np

from adaptive_lidar.utils.grouping import (
    group_by_key,
    group_sizes,
    pack_keys_3d,
    unpack_keys_3d,
)

VoxelKey = Tuple[int, int, int]


def voxel_key(x: float, y: float, z: float, vsize: float) -> VoxelKey:
    """Integer voxel coordinate of a single point (scalar convenience)."""
    return (int(np.floor(x / vsize)),
            int(np.floor(y / vsize)),
            int(np.floor(z / vsize)))


class VoxelHash:
    """CSR-backed voxel index with a thin mapping-like facade.

    The facade (``__contains__`` / ``__getitem__`` / ``keys`` / ``values`` /
    ``items``) exists only so existing callers that expect a dict keep working.
    Hot paths should use :meth:`lookup_many`, :attr:`counts`, or the raw
    ``unique_keys``/``offsets``/``sorted_idx`` arrays directly.
    """

    __slots__ = ("vsize", "unique_keys", "offsets", "sorted_idx", "n_points")

    def __init__(
        self,
        vsize: float,
        unique_keys: np.ndarray,
        offsets: np.ndarray,
        sorted_idx: np.ndarray,
        n_points: int,
    ):
        self.vsize = float(vsize)
        self.unique_keys = unique_keys
        self.offsets = offsets
        self.sorted_idx = sorted_idx
        self.n_points = int(n_points)

    # ── vectorised API (use this) ─────────────────────────────
    @property
    def n_voxels(self) -> int:
        return int(len(self.unique_keys))

    @property
    def counts(self) -> np.ndarray:
        """(M,) points per occupied voxel."""
        return group_sizes(self.offsets)

    def lookup_many(self, keys: np.ndarray) -> np.ndarray:
        """Vectorised key → voxel slot.  Returns -1 for keys not present."""
        keys = np.asarray(keys, dtype=np.int64)
        if self.n_voxels == 0:
            return np.full(keys.shape, -1, dtype=np.int64)
        pos = np.searchsorted(self.unique_keys, keys)
        pos_c = np.clip(pos, 0, self.n_voxels - 1)
        hit = self.unique_keys[pos_c] == keys
        return np.where(hit, pos_c, -1)

    def count_at_points(self, points: np.ndarray) -> np.ndarray:
        """(N,) neighbour count in each point's own voxel.

        This is the "neighbour count in a 0.5 m voxel" feature of the MLP —
        computed without any neighbour search.
        """
        keys = self.keys_for_points(points)
        slots = self.lookup_many(keys)
        counts = self.counts
        return np.where(slots >= 0, counts[np.clip(slots, 0, None)], 0).astype(np.int32)

    def keys_for_points(self, points: np.ndarray) -> np.ndarray:
        """(N,) packed voxel key for each point."""
        inv = 1.0 / self.vsize
        ix = np.floor(points[:, 0] * inv).astype(np.int32)
        iy = np.floor(points[:, 1] * inv).astype(np.int32)
        iz = np.floor(points[:, 2] * inv).astype(np.int32)
        return pack_keys_3d(ix, iy, iz)

    def indices_of_slot(self, slot: int) -> np.ndarray:
        """Point indices inside voxel slot *slot*."""
        return self.sorted_idx[self.offsets[slot]:self.offsets[slot + 1]]

    def voxel_coords(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(M,) ix, iy, iz of every occupied voxel."""
        return unpack_keys_3d(self.unique_keys)

    def centroids(self, points: np.ndarray) -> np.ndarray:
        """(M, 3) mean xyz of each occupied voxel — one reduceat, no loop."""
        if self.n_voxels == 0:
            return np.zeros((0, 3), dtype=np.float32)
        s = np.add.reduceat(points[self.sorted_idx], self.offsets[:-1], axis=0)
        return (s / self.counts[:, None]).astype(np.float32)

    # ── dict-like facade (compatibility only) ─────────────────
    def __len__(self) -> int:
        return self.n_voxels

    def __contains__(self, key) -> bool:
        return self._slot_of(key) >= 0

    def __getitem__(self, key) -> np.ndarray:
        slot = self._slot_of(key)
        if slot < 0:
            raise KeyError(key)
        return self.indices_of_slot(slot)

    def get(self, key, default=None):
        slot = self._slot_of(key)
        return default if slot < 0 else self.indices_of_slot(slot)

    def _slot_of(self, key) -> int:
        packed = (pack_keys_3d(np.int64(key[0]), np.int64(key[1]), np.int64(key[2]))
                  if isinstance(key, tuple) else np.int64(key))
        return int(self.lookup_many(np.asarray([packed]))[0])

    def keys(self) -> List[VoxelKey]:
        ix, iy, iz = self.voxel_coords()
        return list(zip(ix.tolist(), iy.tolist(), iz.tolist()))

    def values(self) -> Iterator[np.ndarray]:
        for s in range(self.n_voxels):
            yield self.indices_of_slot(s)

    def items(self):
        return zip(self.keys(), self.values())


def build_voxel_hash(points: np.ndarray, vsize: float = 0.25) -> VoxelHash:
    """Build the CSR voxel index for one frame.

    Parameters
    ----------
    points : (N, 3) float32
    vsize  : voxel edge length in metres
    """
    points = np.asarray(points)
    n = len(points)
    if n == 0:
        return VoxelHash(vsize, np.empty(0, np.int64), np.zeros(1, np.int64),
                         np.empty(0, np.int64), 0)

    inv = 1.0 / vsize
    ix = np.floor(points[:, 0] * inv).astype(np.int32)
    iy = np.floor(points[:, 1] * inv).astype(np.int32)
    iz = np.floor(points[:, 2] * inv).astype(np.int32)
    keys = pack_keys_3d(ix, iy, iz)

    unique_keys, offsets, order = group_by_key(keys)
    return VoxelHash(vsize, unique_keys, offsets, order, n)


def voxel_centroids(points: np.ndarray, vhash: VoxelHash) -> np.ndarray:
    """(M, 3) array of per-voxel mean xyz."""
    return vhash.centroids(points)
