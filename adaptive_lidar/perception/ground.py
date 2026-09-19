"""
Geometric ground segmentation and the smooth ground-height field.

Two methods, one interface:

``ring_geometry``   (primary, needs a range image)
    On flat ground, consecutive beams hit at radii predictable from their
    elevation angles and the sensor height.  Comparing the radius measured by
    ring *i* against ring *i+1* at the same azimuth and testing the deviation
    from the predicted flat-ground delta separates ground from structure
    without any plane fitting.  Vectorised over the whole range image.

``height_grid``     (fallback, always available)
    Minimum z per coarse grid cell, then "within h_thresh of the local floor".
    Cheap, robust, slightly over-segments on slopes.

Patchwork++ is tried first if installed (``pip install patchworkpp``); if the
import fails we do not fight it — the methods above are always available.

Both produce the same three outputs:
    ground_mask      (N,) bool
    ground_z         (N,) float32  — smooth ground height *under* each point
    height_above_gnd (N,) float32  — z - ground_z
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from adaptive_lidar.utils.grouping import (
    group_by_key,
    pack_keys_2d,
    scatter_to_points,
    segment_percentile_sorted,
    segment_reduce,
    segment_sort,
)

_PATCHWORK = None
_PATCHWORK_TRIED = False


def _try_patchwork():
    """Import patchworkpp once. Never raises."""
    global _PATCHWORK, _PATCHWORK_TRIED
    if _PATCHWORK_TRIED:
        return _PATCHWORK
    _PATCHWORK_TRIED = True
    try:  # pragma: no cover - depends on the machine
        import pypatchworkpp  # noqa: F401
        _PATCHWORK = pypatchworkpp
    except Exception:
        _PATCHWORK = None
    return _PATCHWORK


# ────────────────────────────────────────────────────────────────
# Smooth ground-height field
# ────────────────────────────────────────────────────────────────
def ground_height_field(
    points: np.ndarray,
    ground_mask: np.ndarray,
    cell: float = None,
    smooth_iters: int = 2,
    index=None,
) -> np.ndarray:
    """(N,) ground height under each point, from a smoothed coarse grid.

    The grid holds the 20th percentile of ground-point z per cell.  Empty cells
    are filled by iterative 4-neighbour dilation so points over a hole (a roof,
    a car bonnet) still get a plausible ground reference.  Purely vectorised.
    """
    from adaptive_lidar.utils.spatial_index import TILE_SIZE
    n = len(points)
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    cell = TILE_SIZE if cell is None else cell

    inv = 1.0 / cell
    ix = np.floor(points[:, 0] * inv).astype(np.int32)
    iy = np.floor(points[:, 1] * inv).astype(np.int32)

    gx, gy = ix[ground_mask], iy[ground_mask]
    gz = points[ground_mask, 2].astype(np.float32)
    if gz.size == 0:
        return np.zeros(n, dtype=np.float32)

    x0, y0 = int(ix.min()), int(iy.min())
    w = int(ix.max()) - x0 + 1
    h = int(iy.max()) - y0 + 1
    # Guard against a pathological extent (shouldn't happen after range filter).
    if w * h > 4_000_000:
        return np.full(n, float(np.percentile(gz, 20)), dtype=np.float32)

    keys = (gy - y0).astype(np.int64) * w + (gx - x0)
    uniq, starts, order = group_by_key(keys)
    p20 = segment_percentile_sorted(segment_sort(gz[order], starts), starts, 20)

    grid = np.full((h, w), np.nan, dtype=np.float32)
    grid.flat[uniq] = p20

    # Fill holes: repeated nan-aware 4-neighbour mean.
    for _ in range(smooth_iters + 3):
        nan = np.isnan(grid)
        if not nan.any():
            break
        filled = np.where(nan, 0.0, grid)
        valid = (~nan).astype(np.float32)
        acc = np.zeros_like(filled)
        cnt = np.zeros_like(valid)
        for arr, out in ((filled, acc), (valid, cnt)):
            out[1:, :] += arr[:-1, :]
            out[:-1, :] += arr[1:, :]
            out[:, 1:] += arr[:, :-1]
            out[:, :-1] += arr[:, 1:]
        with np.errstate(invalid="ignore", divide="ignore"):
            nbr = acc / np.maximum(cnt, 1e-6)
        grid = np.where(nan & (cnt > 0), nbr, grid)
    grid = np.nan_to_num(grid, nan=float(np.percentile(gz, 20)))

    # Light box smoothing so the field has no step artefacts.
    for _ in range(smooth_iters):
        acc = grid.copy()
        acc[1:, :] += grid[:-1, :]
        acc[:-1, :] += grid[1:, :]
        acc[:, 1:] += grid[:, :-1]
        acc[:, :-1] += grid[:, 1:]
        w_cnt = np.full_like(grid, 1.0)
        w_cnt[1:, :] += 1.0
        w_cnt[:-1, :] += 1.0
        w_cnt[:, 1:] += 1.0
        w_cnt[:, :-1] += 1.0
        grid = acc / w_cnt

    return grid[np.clip(iy - y0, 0, h - 1), np.clip(ix - x0, 0, w - 1)].astype(np.float32)


# ────────────────────────────────────────────────────────────────
# Method A — height grid (always available)
# ────────────────────────────────────────────────────────────────
def segment_height_grid(
    points: np.ndarray,
    grid_res: float = 1.0,
    h_thresh: float = 0.3,
    sensor_h: float = 1.8,
    index=None,
) -> np.ndarray:
    """Ground mask from "within h_thresh of the local grid floor"."""
    n = len(points)
    if n == 0:
        return np.zeros(0, dtype=bool)

    z = points[:, 2].astype(np.float32)
    if index is not None:
        # Reuse the frame's existing ordering rather than sorting again.
        starts, order = index.tile_starts, index.order
    else:
        inv = 1.0 / grid_res
        ix = np.floor(points[:, 0] * inv).astype(np.int32)
        iy = np.floor(points[:, 1] * inv).astype(np.int32)
        _, starts, order = group_by_key(pack_keys_2d(ix, iy))
    # 5th percentile rather than min: one noise point must not define the floor.
    floor = segment_percentile_sorted(segment_sort(z[order], starts), starts, 5)
    local_min = scatter_to_points(floor, starts, order)

    return ((z - local_min) < h_thresh) & (z < sensor_h * 0.6)


# ────────────────────────────────────────────────────────────────
# Method B — ring geometry (primary, uses the range image)
# ────────────────────────────────────────────────────────────────
def segment_ring_geometry(
    points: np.ndarray,
    range_image: np.ndarray,          # (H, W) int32 point index, -1 empty
    range_image_valid: np.ndarray,    # (H, W) bool
    elev_angles_deg: np.ndarray,      # (H,) beam elevation, ascending with row
    sensor_h: float = 1.8,
    delta_tol: float = 0.35,
    max_slope_deg: float = 20.0,
) -> np.ndarray:
    """Ground mask from inter-ring radius differences.

    For a flat ground plane at depth ``sensor_h`` below the sensor, ring *i*
    with elevation ``e_i`` (negative, pointing down) lands at horizontal radius
    ``rho_i = sensor_h / tan(-e_i)``.  Two adjacent rings therefore differ by a
    *predictable* ``rho_{i+1} - rho_i`` at every azimuth.  Where the measured
    difference collapses towards zero the two beams hit the same vertical face:
    structure, not ground.  Where it matches the prediction within tolerance,
    the surface is locally flat: ground.

    Fully vectorised over the (H, W) image — no loops, no plane fitting.
    """
    n = len(points)
    ground = np.zeros(n, dtype=bool)
    if n == 0 or range_image is None:
        return ground

    H, W = range_image.shape
    idx = range_image
    valid = range_image_valid

    # Horizontal radius and z of the point at each pixel.
    rho = np.zeros((H, W), dtype=np.float32)
    zz = np.zeros((H, W), dtype=np.float32)
    flat = idx[valid]
    rho[valid] = np.hypot(points[flat, 0], points[flat, 1]).astype(np.float32)
    zz[valid] = points[flat, 2].astype(np.float32)

    # Predicted flat-ground radius per ring.
    e = np.radians(np.asarray(elev_angles_deg, dtype=np.float32))
    down = e < -1e-3
    rho_pred = np.full(H, np.nan, dtype=np.float32)
    rho_pred[down] = sensor_h / np.tan(-e[down])
    with np.errstate(invalid="ignore"):
        pred_delta = np.diff(rho_pred)              # (H-1,) expected rho[i+1]-rho[i]
    finite = np.isfinite(pred_delta)
    pred_delta = np.where(finite, pred_delta, 1.0)

    # Measured inter-ring delta, both directions.
    meas_delta = rho[1:, :] - rho[:-1, :]
    pair_ok = valid[1:, :] & valid[:-1, :]

    # Ratio of measured to predicted; ~1 on flat ground, ~0 on a vertical face.
    pd = np.where(finite, pred_delta, 1.0)[:, None].astype(np.float32)
    ratio = np.where(pair_ok, meas_delta / np.where(np.abs(pd) < 1e-3, 1e-3, pd), 0.0)

    # Local slope implied by the pair: dz / d_rho.
    with np.errstate(invalid="ignore", divide="ignore"):
        slope = np.abs(zz[1:, :] - zz[:-1, :]) / np.maximum(np.abs(meas_delta), 1e-3)
    slope_ok = slope < np.tan(np.radians(max_slope_deg))

    consistent = pair_ok & (np.abs(ratio - 1.0) < delta_tol) & slope_ok & finite[:, None]

    # A pixel is ground if it is consistent with the ring above or below it.
    px_ground = np.zeros((H, W), dtype=bool)
    px_ground[:-1, :] |= consistent
    px_ground[1:, :] |= consistent

    # The lowest few rings that see anything are ground by construction unless
    # the geometry above rejected them; keep the height sanity check.
    px_ground &= valid & (zz < sensor_h * 0.6)

    ground[idx[px_ground]] = True
    return ground


# ────────────────────────────────────────────────────────────────
# Dispatcher
# ────────────────────────────────────────────────────────────────
class GroundSegmenter:
    """Selects a method once and reports which one is active."""

    def __init__(self, config: Dict[str, Any]):
        self.cfg = config
        gcfg = config.get("thresholds", {}).get("ground", {})
        self.h_thresh = gcfg.get("height_threshold", 0.3)
        self.grid_res = gcfg.get("grid_resolution", 1.0)
        self.sensor_h = config.get("sensor", {}).get("height", 1.8)
        self.method = config.get("perception", {}).get("ground_method", "auto")
        self._pw = _try_patchwork() if self.method in ("auto", "patchwork") else None
        self.active = "patchworkpp" if self._pw is not None else None

    def __call__(
        self,
        points: np.ndarray,
        range_image: Optional[np.ndarray] = None,
        range_image_valid: Optional[np.ndarray] = None,
        elev_angles_deg: Optional[np.ndarray] = None,
        index=None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (ground_mask, ground_z, height_above_gnd).

        ``index`` is the frame's shared MortonIndex; when supplied the height
        field is built on the tile grouping it already holds instead of
        re-sorting the cloud into its own grid.
        """
        n = len(points)
        if n == 0:
            z = np.zeros(0, dtype=np.float32)
            return np.zeros(0, dtype=bool), z, z

        mask = None
        if self._pw is not None:  # pragma: no cover - optional dependency
            try:
                self._pw.estimateGround(np.ascontiguousarray(points[:, :3], np.float64))
                gi = self._pw.getGroundIndices()
                mask = np.zeros(n, dtype=bool)
                mask[np.asarray(gi, dtype=np.int64)] = True
                self.active = "patchworkpp"
            except Exception:
                self._pw = None
                mask = None

        if mask is None and self.method in ("auto", "ring_geometry") \
                and range_image is not None and elev_angles_deg is not None:
            mask = segment_ring_geometry(
                points, range_image, range_image_valid, elev_angles_deg,
                sensor_h=self.sensor_h)
            self.active = "ring_geometry"
            # Ring geometry is strict; if it found implausibly little ground the
            # scan probably has no usable ring structure — fall through.
            if mask.mean() < 0.05:
                mask = None
                self.active = None

        if mask is None:
            mask = segment_height_grid(
                points, self.grid_res, self.h_thresh, self.sensor_h,
                index=index)
            self.active = "height_grid"

        gz = ground_height_field(points, mask, index=index)
        hag = (points[:, 2] - gz).astype(np.float32)
        # A point near the ground field is ground even if the primary method
        # was strict; this keeps the field and the mask mutually consistent.
        mask = mask | (np.abs(hag) < 0.08)
        return mask, gz, (points[:, 2] - gz).astype(np.float32)
