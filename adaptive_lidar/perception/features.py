"""
Per-point feature extraction — the input to every semantic backend.

Every feature comes from the range image or the voxel hash, both already built
in S1.  That is the whole design constraint: no neighbour search, no kd-tree,
no per-point Python.  A 3x3 window on the range image IS a neighbourhood, and
it costs eight array shifts for the entire cloud.

FEATURE_NAMES is the frozen ordering.  The trained checkpoint records it and
:func:`extract_features` is checked against it at load time, so a feature
reordering can never silently corrupt inference.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from adaptive_lidar.utils.range_image import (
    RangeImage,
    local_window_stats,
    vertical_run_lengths,
)

FEATURE_NAMES = (
    "height_above_ground",   # z - smooth ground field
    "z_rel_local_ground",    # z - lowest z in the local range-image window
    "intensity_norm",        # range + incidence corrected
    "local_height_var",      # 3x3 range-image window
    "local_range_grad",      # range-image neighbours
    "vertical_run",          # consecutive rings at this azimuth
    "range_r",               # distance from sensor
    "incidence_cos",         # cos(beam direction, local surface normal)
    "penetration_ratio",     # return_number / return_count
    "voxel_neighbour_count", # occupancy of this point's 0.5 m voxel
    "planarity",             # local covariance, from range-image neighbours
    "verticality",           # 1 - |normal_z|
)
N_FEATURES = len(FEATURE_NAMES)


# ────────────────────────────────────────────────────────────────
# Intensity normalisation (Phase 3.4)
# ────────────────────────────────────────────────────────────────
def normalise_intensity(
    intensity: np.ndarray,
    r: np.ndarray,
    cos_incidence: np.ndarray,
    r_ref: float = 20.0,
) -> np.ndarray:
    """Correct raw intensity for range and incidence angle.

    The LiDAR equation gives returned power proportional to
    ``rho * cos(theta) / r^2``.  Inverting that recovers an estimate of the
    surface reflectance ``rho``, which is a material property and therefore
    comparable between a point at 5 m and a point at 80 m.

    Skipping this step makes intensity pure noise — asphalt at 5 m and
    retroreflective paint at 70 m return the same raw value — so every
    intensity-derived signal (the terrain variance layer, the MLP's material
    cue) depends on it.
    """
    r = np.maximum(np.asarray(r, dtype=np.float32), 1e-3)
    cos_i = np.clip(np.asarray(cos_incidence, dtype=np.float32), 0.15, 1.0)
    rho = np.asarray(intensity, dtype=np.float32) * (r / r_ref) ** 2 / cos_i
    return np.clip(rho, 0.0, 4.0).astype(np.float32)


# ────────────────────────────────────────────────────────────────
# Surface normals from the range image
# ────────────────────────────────────────────────────────────────
def range_image_normals(
    ri: RangeImage,
    points: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """(H,W,3) unit surface normals and (H,W) cos(incidence).

    The normal is the cross product of the two tangent vectors obtained by
    central-differencing the pixel's 3D position along the row and column axes.
    That is the standard organised-point-cloud normal estimate and it is two
    subtractions and one cross product for the whole image.
    """
    H, W = ri.shape
    xyz = np.zeros((H, W, 3), dtype=np.float32)
    xyz[ri.valid] = points[ri.index[ri.valid]]

    # Component arithmetic rather than np.cross / np.linalg.norm: on a
    # (64, 2048, 3) array those allocate several full temporaries each and
    # dominate the feature-extraction cost.
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    dux = np.zeros_like(x); duy = np.zeros_like(x); duz = np.zeros_like(x)
    dux[:, 1:-1] = x[:, 2:] - x[:, :-2]
    duy[:, 1:-1] = y[:, 2:] - y[:, :-2]
    duz[:, 1:-1] = z[:, 2:] - z[:, :-2]
    dvx = np.zeros_like(x); dvy = np.zeros_like(x); dvz = np.zeros_like(x)
    dvx[1:-1, :] = x[2:, :] - x[:-2, :]
    dvy[1:-1, :] = y[2:, :] - y[:-2, :]
    dvz[1:-1, :] = z[2:, :] - z[:-2, :]

    nx = duy * dvz - duz * dvy
    ny = duz * dvx - dux * dvz
    nz = dux * dvy - duy * dvx
    ln = np.sqrt(nx * nx + ny * ny + nz * nz)
    good = ln > 1e-9
    inv = np.where(good, 1.0 / np.maximum(ln, 1e-9), 0.0).astype(np.float32)
    nx *= inv; ny *= inv; nz *= inv

    # Beam direction is the unit vector from sensor to point.
    rn = np.sqrt(x * x + y * y + z * z)
    inv_r = 1.0 / np.maximum(rn, 1e-6)
    cos_i = np.abs(nx * x * inv_r + ny * y * inv_r + nz * z * inv_r)
    # A degenerate normal (empty neighbourhood) must not become cos = 0.
    cos_i = np.where(good & ri.valid, cos_i, 1.0).astype(np.float32)

    nrm = np.stack([nx, ny, nz], axis=2).astype(np.float32)
    return nrm, cos_i


# ────────────────────────────────────────────────────────────────
# Main extractor
# ────────────────────────────────────────────────────────────────
def extract_features(frame) -> np.ndarray:
    """(N, N_FEATURES) float32 feature matrix for every point in the frame.

    Also fills ``frame.intensity_norm`` and caches the vertical-run length per
    point on ``frame._vertical_run`` (the allocation controller's geometric
    safety pin reads it).
    """
    pts = frame.points
    n = len(pts)
    if n == 0:
        return np.zeros((0, N_FEATURES), dtype=np.float32)
    cols = [np.zeros(n, dtype=np.float32) for _ in range(N_FEATURES)]

    ri: Optional[RangeImage] = getattr(frame, "_range_image", None)
    # The range image already holds every point's range; reuse it rather than
    # recomputing a 128k-element norm.
    r = getattr(frame, "_range", None)
    if r is None:
        r = np.sqrt((pts.astype(np.float32) ** 2).sum(axis=1))
        frame._range = r

    # ── range-image derived, scattered back to points ─────────
    if ri is not None:
        nrm, cos_i_img = range_image_normals(ri, pts)
        z_var_img, grad_img, planarity_img = local_window_stats(ri, pts, half=1)
        run_img = vertical_run_lengths(ri)

        # Lowest z inside the local window — a local ground reference that does
        # not depend on the global ground field.
        H, W = ri.shape
        zimg = np.full((H, W), np.inf, dtype=np.float32)
        zimg[ri.valid] = pts[ri.index[ri.valid], 2]
        # Separable 3x3 minimum: two 1-D passes, not nine whole-image copies.
        zmin = np.minimum(np.minimum(zimg, np.roll(zimg, 1, 1)), np.roll(zimg, -1, 1))
        tmp = zmin.copy()
        zmin[1:, :] = np.minimum(zmin[1:, :], tmp[:-1, :])
        zmin[:-1, :] = np.minimum(zmin[:-1, :], tmp[1:, :])
        zmin = np.where(np.isfinite(zmin), zmin, 0.0)

        px = ri.valid
        idx = ri.index[px]
        cos_inc = np.ones(n, dtype=np.float32)
        cos_inc[idx] = cos_i_img[px]
        # Scatter into contiguous 1-D columns; writing into a (N, 12) row-major
        # array one column at a time touches every cache line 12 times.
        cols[3][idx] = z_var_img[px]
        cols[4][idx] = grad_img[px]
        cols[5][idx] = run_img[px]
        cols[10][idx] = planarity_img[px]
        cols[11][idx] = 1.0 - np.abs(nrm[:, :, 2][px])
        cols[1][idx] = pts[idx, 2] - zmin[px]
        vertical_run = np.zeros(n, dtype=np.int16)
        vertical_run[idx] = run_img[px]
    else:
        cos_inc = np.ones(n, dtype=np.float32)
        vertical_run = np.zeros(n, dtype=np.int16)

    frame._vertical_run = vertical_run
    frame._cos_incidence = cos_inc

    # ── intensity normalisation ───────────────────────────────
    inorm = normalise_intensity(frame.intensity, r, cos_inc)
    frame.intensity_norm = inorm

    # ── remaining features ────────────────────────────────────
    hag = frame.height_above_gnd
    cols[0][:] = hag if hag is not None else pts[:, 2]
    cols[2][:] = inorm
    cols[6][:] = r
    # cos rather than the angle: monotonically related, and arccos over 128k
    # points is a transcendental call the network gains nothing from.
    cols[7][:] = cos_inc

    rc = frame.return_count
    rn = frame.return_number
    if rc is not None and rn is not None:
        rcf = np.asarray(rc, np.float32)
        cols[8][:] = np.where(rcf > 1,
                              np.asarray(rn, np.float32) / np.maximum(rcf, 1.0),
                              0.0)

    vh = frame.voxel_hash
    if vh is not None:
        cols[9][:] = vh.count_at_points(pts).astype(np.float32)

    feats = np.stack(cols, axis=1)
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0, copy=False)


#: Per-feature normalisation applied before the MLP.  Fixed constants rather
#: than dataset statistics so that a checkpoint trained on synthetic data
#: behaves identically on a real scan.
FEATURE_SCALE = np.array([
    3.0,     # height_above_ground
    2.0,     # z_rel_local_ground
    1.0,     # intensity_norm
    0.5,     # local_height_var
    2.0,     # local_range_grad
    16.0,    # vertical_run
    50.0,    # range_r
    1.0,     # incidence_cos
    1.0,     # penetration_ratio
    20.0,    # voxel_neighbour_count
    1.0,     # planarity
    1.0,     # verticality
], dtype=np.float32)


def normalise_features(feats: np.ndarray) -> np.ndarray:
    """Scale features into roughly [-2, 2] for the network."""
    return np.clip(feats / FEATURE_SCALE, -4.0, 4.0).astype(np.float32)
