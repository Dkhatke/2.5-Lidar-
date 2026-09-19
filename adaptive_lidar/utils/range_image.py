"""
Range image via scan unfolding.

THE IMAGE HOLDS POINT INDICES, NOT RANGE VALUES.
``range_image[row, col]`` is an index into ``frame.points`` (-1 where empty).
This makes the image an *index* into the cloud rather than a copy of it: every
downstream feature (neighbour heights, range gradients, vertical runs, local
covariance, free-space carving) is then pure array indexing — no kd-tree, no
neighbour search, and no risk of the image and the cloud disagreeing.

Row = ring, column = azimuth bin.

Three ways to get the row, in order of fidelity:

1. **Native** — the sensor (or our synthetic generator) supplies ``ring`` and
   ``azimuth_bin`` per point.  Exact.
2. **Scan unfolding** — SemanticKITTI ships points in firing order: laser by
   laser, sweeping in azimuth.  A wraparound in azimuth marks a new ring, so
   the ring id is the cumulative count of wraparounds.  This is the standard
   reconstruction and is markedly better than (3).
3. **Spherical projection** — compute the elevation angle of each point and bin
   it.  Loses points wherever two beams' elevation ranges overlap, which is
   exactly what the collision counter below measures.  Kept behind a config
   flag so the two can be compared; the valid-pixel rate of (2) vs (3) is a
   reported metric.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class RangeImage:
    index: np.ndarray        # (H, W) int32 — point index, -1 = empty
    valid: np.ndarray        # (H, W) bool
    rng: np.ndarray          # (H, W) float32 — range of the point at that pixel
    elev_angles_deg: np.ndarray   # (H,) beam elevation, ascending with row
    method: str
    collisions: int          # points discarded because their pixel was taken
    valid_rate: float        # fraction of pixels that received a point

    @property
    def shape(self) -> Tuple[int, int]:
        return self.index.shape


def default_elevations(num_rings: int = 64,
                       fov_lo: float = -24.0,
                       fov_hi: float = 2.0) -> np.ndarray:
    """Beam elevation angles for a generic 64-beam spinning LiDAR.

    -24 deg to +2 deg matches the HDL-64E / Ouster OS1-64 envelope and is what
    the synthetic generator uses, so native and reconstructed rings agree.
    """
    return np.linspace(fov_lo, fov_hi, num_rings).astype(np.float32)


# ────────────────────────────────────────────────────────────────
# Row assignment
# ────────────────────────────────────────────────────────────────
def unfold_rings(points: np.ndarray, num_rings: int = 64) -> np.ndarray:
    """Reconstruct ring ids from firing order (scan unfolding).

    Points arrive laser-by-laser sweeping in azimuth.  Each time the azimuth
    wraps past +pi the scan has begun the next ring, so the ring id is the
    running count of wraparounds.  Vectorised: one diff, one cumsum.
    """
    n = len(points)
    if n == 0:
        return np.zeros(0, dtype=np.int16)
    az = np.arctan2(points[:, 1], points[:, 0])
    # A wraparound is a large positive jump from -pi back towards +pi.
    wrapped = np.r_[False, (az[1:] - az[:-1]) > np.pi * 0.5]
    ring = np.cumsum(wrapped)
    return np.clip(ring, 0, num_rings - 1).astype(np.int16)


def spherical_rows(points: np.ndarray, elev_angles_deg: np.ndarray) -> np.ndarray:
    """Row from elevation angle — the lossy fallback.

    Each point is assigned to the nearest beam elevation.  Where two beams'
    footprints overlap in elevation (which they do at range, and on slopes) the
    two points compete for one pixel and one is lost.
    """
    n = len(points)
    if n == 0:
        return np.zeros(0, dtype=np.int16)
    r = np.linalg.norm(points, axis=1)
    elev = np.degrees(np.arcsin(np.clip(points[:, 2] / np.maximum(r, 1e-6), -1, 1)))
    # elev_angles_deg is ascending, so searchsorted gives the bracket directly.
    pos = np.searchsorted(elev_angles_deg, elev)
    lo = np.clip(pos - 1, 0, len(elev_angles_deg) - 1)
    hi = np.clip(pos, 0, len(elev_angles_deg) - 1)
    pick_hi = np.abs(elev_angles_deg[hi] - elev) < np.abs(elev_angles_deg[lo] - elev)
    return np.where(pick_hi, hi, lo).astype(np.int16)


# ────────────────────────────────────────────────────────────────
# Builder
# ────────────────────────────────────────────────────────────────
def build_range_image(
    points: np.ndarray,
    ring: Optional[np.ndarray] = None,
    azimuth_bin: Optional[np.ndarray] = None,
    num_rings: int = 64,
    num_azimuth: int = 2048,
    elev_angles_deg: Optional[np.ndarray] = None,
    method: str = "auto",
) -> RangeImage:
    """Project a cloud into an index-valued range image.

    method : "auto" | "native" | "unfold" | "spherical"
    """
    n = len(points)
    H, W = int(num_rings), int(num_azimuth)
    if elev_angles_deg is None:
        elev_angles_deg = default_elevations(H)

    index = np.full((H, W), -1, dtype=np.int32)
    rng = np.zeros((H, W), dtype=np.float32)
    if n == 0:
        return RangeImage(index, index >= 0, rng, elev_angles_deg, "empty", 0, 0.0)

    r = np.linalg.norm(points, axis=1).astype(np.float32)

    # ── rows ──────────────────────────────────────────────────
    has_native = (ring is not None and azimuth_bin is not None
                  and np.any(np.asarray(ring) >= 0))
    if method == "native" or (method == "auto" and has_native):
        rows = np.clip(np.asarray(ring), 0, H - 1).astype(np.int32)
        cols = np.mod(np.asarray(azimuth_bin), W).astype(np.int32)
        used = "native"
    elif method == "spherical":
        rows = spherical_rows(points, elev_angles_deg).astype(np.int32)
        cols = _azimuth_cols(points, W)
        used = "spherical"
    else:
        rows = unfold_rings(points, H).astype(np.int32)
        cols = _azimuth_cols(points, W)
        used = "unfold"

    # ── scatter, nearest point wins each pixel ────────────────
    # Sort by descending range so the closest point is written last.
    order = np.argsort(-r, kind="stable")
    flat = rows[order].astype(np.int64) * W + cols[order]
    index.flat[flat] = order.astype(np.int32)
    rng.flat[flat] = r[order]

    valid = index >= 0
    n_filled = int(valid.sum())
    collisions = n - n_filled

    return RangeImage(
        index=index,
        valid=valid,
        rng=rng,
        elev_angles_deg=np.asarray(elev_angles_deg, dtype=np.float32),
        method=used,
        collisions=collisions,
        valid_rate=n_filled / float(H * W),
    )


def _azimuth_cols(points: np.ndarray, W: int) -> np.ndarray:
    az = np.arctan2(points[:, 1], points[:, 0])          # -pi..pi
    col = ((az + np.pi) / (2 * np.pi) * W).astype(np.int32)
    return np.clip(col, 0, W - 1)


# ────────────────────────────────────────────────────────────────
# Range-image derived signals (Phase 3.3)
# ────────────────────────────────────────────────────────────────
def vertical_run_lengths(
    ri: RangeImage,
    points: np.ndarray,
    range_tol: float = 0.35,
) -> np.ndarray:
    """(H, W) length of the consecutive-ring VERTICAL run each pixel is in.

    A thin vertical structure — a pole, a sign post, a standing person — is
    seen by several consecutive rings at essentially the same azimuth and
    essentially the same range, AND the successive hits climb in z rather than
    running outward along the ground.

    Both conditions are necessary.  Testing range similarity alone fires on
    near-field ground, where consecutive rings land only centimetres apart in
    range: in a 70 m scan that marked more than half of all points as
    "vertical structure", which makes the pin meaningless.  Requiring the
    step between rings to be steeper than 45 degrees (|dz| > |d_rho|) is what
    makes it a detector of vertical things rather than of nearby things.

    THIS IS THE GEOMETRIC SAFETY PIN.  It fires on "small, isolated,
    vertically-extended cluster above ground" without knowing what the object
    is, so the retention guarantee survives a segmentation failure.

    Pure column-wise scanning over the image: a loop over 64 ROWS, never over
    points.
    """
    H, W = ri.shape
    z = np.zeros((H, W), dtype=np.float32)
    rho = np.zeros((H, W), dtype=np.float32)
    v = ri.valid
    idx = ri.index[v]
    z[v] = points[idx, 2]
    rho[v] = np.hypot(points[idx, 0], points[idx, 1])

    dz = np.abs(z[1:, :] - z[:-1, :])
    drho = np.abs(rho[1:, :] - rho[:-1, :])
    dr = np.abs(ri.rng[1:, :] - ri.rng[:-1, :])

    same = (v[1:, :] & v[:-1, :]
            & (dr < range_tol)          # the two beams hit the same object
            & (dz > drho))              # and the object rises rather than runs

    # Run length via a forward then backward cumulative pass over rows.
    up = np.zeros((H, W), dtype=np.int16)
    for i in range(1, H):               # H = 64 — a loop over ROWS, not points
        up[i] = np.where(same[i - 1], up[i - 1] + 1, 0)
    down = np.zeros((H, W), dtype=np.int16)
    for i in range(H - 2, -1, -1):
        down[i] = np.where(same[i], down[i + 1] + 1, 0)
    return (up + down + 1).astype(np.int16) * v


def _box_sum(a: np.ndarray, half: int) -> np.ndarray:
    """Separable (2*half+1)^2 box sum, wrapping in azimuth, clamping in ring.

    Azimuth wraps because the scan is a full rotation; rings do not, so the
    edge rows simply see a smaller window. Two 1-D passes rather than
    (2*half+1)^2 whole-image shifts.
    """
    out = a.copy()
    for k in range(1, half + 1):                 # azimuth: wrap
        out += np.roll(a, k, axis=1) + np.roll(a, -k, axis=1)
    acc = out.copy()
    for k in range(1, half + 1):                 # rings: clamp
        acc[k:, :] += out[:-k, :]
        acc[:-k, :] += out[k:, :]
    return acc


def local_window_stats(
    ri: RangeImage,
    points: np.ndarray,
    half: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(H,W) local height variance, range gradient and planarity.

    A (2*half+1)^2 window over the range image, accumulated by shifting the
    whole image — never by iterating pixels.  ``half=1`` gives the 3x3 window
    the MLP feature vector expects.
    """
    H, W = ri.shape
    z = np.zeros((H, W), dtype=np.float32)
    z[ri.valid] = points[ri.index[ri.valid], 2]
    r = ri.rng
    v = ri.valid.astype(np.float32)

    zv, rv = z * v, r * v
    # A separable box filter: two 1-D passes instead of (2*half+1)^2 shifts,
    # and cumulative sums instead of np.roll, which copies the whole image.
    def box(a):
        return _box_sum(a, half)

    acc_z = box(zv)
    acc_z2 = box(zv * z)
    acc_r = box(rv)
    acc_r2 = box(rv * r)
    cnt = box(v)

    c = np.maximum(cnt, 1.0)
    z_mean = acc_z / c
    z_var = np.maximum(acc_z2 / c - z_mean * z_mean, 0.0)
    r_mean = acc_r / c
    r_var = np.maximum(acc_r2 / c - r_mean * r_mean, 0.0)

    # Range gradient: central difference along both axes.
    gr = np.zeros_like(r)
    gr[:, 1:-1] = np.abs(r[:, 2:] - r[:, :-2]) * 0.5
    gc = np.zeros_like(r)
    gc[1:-1, :] = np.abs(r[2:, :] - r[:-2, :]) * 0.5
    grad = np.hypot(gr, gc)

    # Planarity proxy: low range variance relative to the local mean range
    # means the window lies on one smooth surface.
    planarity = 1.0 / (1.0 + r_var / np.maximum(r_mean * 0.02, 1e-3))
    return z_var, grad, planarity.astype(np.float32)


def isolated_return_mask(
    ri: RangeImage,
    intensity: np.ndarray,
    intensity_thresh: float = 0.05,
    min_neighbours: int = 2,
    range_tol: float = 0.5,
) -> np.ndarray:
    """(N,) True for likely dust/rain returns.

    Low intensity AND few range-image neighbours at a similar range.  This is
    deliberately NOT a generic statistical outlier filter: such a filter
    deletes poles, which are by construction sparse and isolated.
    """
    H, W = ri.shape
    n = len(intensity)
    r = ri.rng
    v = ri.valid

    nbrs = np.zeros((H, W), dtype=np.int16)
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        shifted_r = np.roll(np.roll(r, dy, 0), dx, 1)
        shifted_v = np.roll(np.roll(v, dy, 0), dx, 1)
        nbrs += (shifted_v & (np.abs(shifted_r - r) < range_tol)).astype(np.int16)

    isolated_px = v & (nbrs < min_neighbours)
    out = np.zeros(n, dtype=bool)
    idx = ri.index[isolated_px]
    out[idx] = intensity[idx] < intensity_thresh
    return out
