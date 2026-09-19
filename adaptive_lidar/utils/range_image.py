"""
Range image builder.

Projects a point cloud onto a 2D (ring × azimuth) image.
Used for visualisation and angular neighbourhood reasoning.

PROTOTYPE NOTE: ring assignment uses elevation-angle binning
(future: use hardware ring IDs from sensor if available).
"""
from __future__ import annotations
import numpy as np
from typing import Tuple


def build_range_image(
    points: np.ndarray,
    num_rings: int = 64,
    h_res_deg: float = 0.2,
    v_fov_deg: float = 30.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project (N,3) xyz points onto a (H, W) range image.

    Returns
    -------
    ri      : (H, W) float32  — range values (0 = empty)
    ri_xyz  : (H, W, 3) float32 — xyz of closest point per pixel
    """
    H = num_rings
    W = int(360.0 / h_res_deg)

    ri = np.zeros((H, W), dtype=np.float32)
    ri_xyz = np.zeros((H, W, 3), dtype=np.float32)

    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2)
    valid = r > 0

    # Azimuth angle → column index
    azimuth = np.degrees(np.arctan2(y[valid], x[valid]))  # -180…+180
    azimuth = (azimuth + 180.0) % 360.0                   # 0…360
    col = (azimuth / h_res_deg).astype(np.int32)
    col = np.clip(col, 0, W - 1)

    # Elevation angle → row index
    v_half = v_fov_deg / 2.0
    elev = np.degrees(np.arcsin(np.clip(z[valid] / r[valid], -1, 1)))
    row = ((elev + v_half) / v_fov_deg * H).astype(np.int32)
    row = np.clip(row, 0, H - 1)

    r_valid = r[valid]
    pts_valid = points[valid]

    # For each (row, col) keep the closest point
    # Process in reverse-range order so closest wins
    order = np.argsort(r_valid)[::-1]
    ri[row[order], col[order]] = r_valid[order]
    ri_xyz[row[order], col[order]] = pts_valid[order]

    return ri, ri_xyz
