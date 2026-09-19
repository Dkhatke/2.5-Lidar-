"""
Voxel hash — current-frame spatial query index.

Maps voxel_key (ix, iy, iz) → list of point indices.

IMPORTANT ARCHITECTURAL DISTINCTION:
  This is a PER-FRAME QUERY INDEX (S1).
  It is NOT the persistent adaptive map (that lives in mapping/adaptive_map.py).
  Do not merge these two concepts.
"""
from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple


VoxelKey = Tuple[int, int, int]


def voxel_key(x: float, y: float, z: float, vsize: float) -> VoxelKey:
    return (int(np.floor(x / vsize)),
            int(np.floor(y / vsize)),
            int(np.floor(z / vsize)))


def build_voxel_hash(
    points: np.ndarray,
    vsize: float = 0.25,
) -> Dict[VoxelKey, List[int]]:
    """
    Build a dict mapping voxel keys to lists of point indices.

    Parameters
    ----------
    points : (N, 3) float32
    vsize  : voxel edge length in metres

    Returns
    -------
    hash : dict { (ix, iy, iz) : [idx, ...] }
    """
    inv = 1.0 / vsize
    keys_x = np.floor(points[:, 0] * inv).astype(np.int32)
    keys_y = np.floor(points[:, 1] * inv).astype(np.int32)
    keys_z = np.floor(points[:, 2] * inv).astype(np.int32)

    vhash: Dict[VoxelKey, List[int]] = {}
    for i in range(len(points)):
        k = (int(keys_x[i]), int(keys_y[i]), int(keys_z[i]))
        if k not in vhash:
            vhash[k] = []
        vhash[k].append(i)

    return vhash


def voxel_centroids(
    points: np.ndarray,
    vhash: Dict[VoxelKey, List[int]],
) -> np.ndarray:
    """Return (M, 3) array of per-voxel mean xyz."""
    centroids = []
    for indices in vhash.values():
        centroids.append(points[indices].mean(axis=0))
    return np.array(centroids, dtype=np.float32) if centroids else np.zeros((0, 3))
