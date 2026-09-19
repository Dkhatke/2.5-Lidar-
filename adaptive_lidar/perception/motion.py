"""
Range-image residual moving-object segmentation (MOS).

4DMOS and its relatives are sparse 4D convolution networks and need CUDA to
build.  This is the CPU-feasible member of the same published family: the
*residual image* line of work — LMNet ("Moving Object Segmentation in 3D LiDAR
Data: A Learning-based Approach Exploiting Sequential Data", Chen et al., RA-L
2021) and the residual-image variants that followed it.  Those methods feed a
stack of residual images to a network; we threshold them directly, which is the
same signal without the training.

THE IDEA
--------
Take the previous frames' points, transform them into the *current* sensor
frame using the poses, and re-project them into the range image.  A static
surface projects back onto itself, so the expected and the measured range agree.
A surface that moved leaves two disagreements: the range where it *was* (now
showing the background behind it, so measured > expected) and the range where
it *is* (now closer than the background, so measured < expected).

One frame of disagreement is noise — occlusion boundaries, pose error, a
missing return.  Disagreement that is *consistent across the window* is motion.

The semantic prior gates it: a building that appears to move is pose error.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

#: Only these classes may be reported as moving.
MOVABLE_CLASSES = (3, 4)


def transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply a 4x4 rigid transform to (N,3) points."""
    R = T[:3, :3]
    t = T[:3, 3]
    return (points @ R.T + t).astype(np.float32)


def residual_mos(frame, ctx, window: int = 2, tol: float = 0.45) -> np.ndarray:
    """(N,) moving probability per point.

    ``window`` previous frames are compared; a point needs disagreement in the
    majority of available comparisons to be called moving.
    """
    n = len(frame.points)
    out = np.zeros(n, dtype=np.float32)
    ri = getattr(frame, "_range_image", None)
    if ri is None or n == 0:
        return out

    hist = [f for f in ctx.frame_history[-window:] if f is not frame]
    if not hist:
        return out
    # Comparing against every previous frame costs a full transform and
    # re-projection each time for very little extra evidence, so the window is
    # deliberately short: a moving object disagrees with the frame before it.

    cur_pose = frame.pose if frame.pose is not None else np.eye(4)
    try:
        inv_cur = np.linalg.inv(cur_pose)
    except np.linalg.LinAlgError:
        inv_cur = np.eye(4)

    H, W = ri.shape
    votes = np.zeros((H, W), dtype=np.int16)
    comparisons = 0

    for prev in hist:
        if len(prev.points) == 0:
            continue
        prev_pose = prev.pose if prev.pose is not None else np.eye(4)
        # prev sensor frame -> world -> current sensor frame
        T = inv_cur @ prev_pose
        pp = transform_points(prev.points, T)

        rows, cols, rr = _project(pp, ri, H, W)
        ok = rows >= 0
        if not np.any(ok):
            continue

        # Expected range image from the previous scan: nearest point per pixel.
        expected = np.full((H, W), np.inf, dtype=np.float32)
        order = np.argsort(-rr[ok], kind="stable")
        flat = rows[ok][order].astype(np.int64) * W + cols[ok][order]
        expected.flat[flat] = rr[ok][order]

        both = ri.valid & np.isfinite(expected)
        # Signed residual: the sign distinguishes "something arrived" from
        # "something left", and both are evidence of motion.
        resid = np.zeros((H, W), dtype=np.float32)
        resid[both] = ri.rng[both] - expected[both]
        # Scale the tolerance with range: a 0.45 m residual at 70 m is within
        # beam divergence, at 5 m it is a moving object.
        scaled_tol = tol * np.maximum(1.0, ri.rng / 25.0)
        votes += (both & (np.abs(resid) > scaled_tol)).astype(np.int16)
        comparisons += 1

    if comparisons == 0:
        return out

    px_prob = votes.astype(np.float32) / comparisons
    idx = ri.index[ri.valid]
    out[idx] = px_prob[ri.valid]

    # ── semantic gate ─────────────────────────────────────────
    cls = frame.sem_class
    if cls is not None and len(cls) == n:
        movable = np.isin(cls, MOVABLE_CLASSES)
        # Non-movable classes are damped, not zeroed: a misclassified moving car
        # should still be able to reach the gate on strong residual alone.
        out = np.where(movable, out, out * 0.25)

    # Points near the ground are almost never independently moving, and the
    # ground is where pose error produces the largest residual.
    hag = frame.height_above_gnd
    if hag is not None and len(hag) == n:
        out = np.where(hag < 0.2, out * 0.2, out)

    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _project(points: np.ndarray, ri, H: int, W: int):
    """Project (N,3) into the range image grid, matching how it was built."""
    r = np.linalg.norm(points, axis=1).astype(np.float32)
    good = r > 1e-3
    az = np.arctan2(points[:, 1], points[:, 0])
    cols = np.clip(((az + np.pi) / (2 * np.pi) * W).astype(np.int32), 0, W - 1)

    elev = np.degrees(np.arcsin(np.clip(points[:, 2] / np.maximum(r, 1e-6), -1, 1)))
    ang = ri.elev_angles_deg
    pos = np.searchsorted(ang, elev)
    lo = np.clip(pos - 1, 0, H - 1)
    hi = np.clip(pos, 0, H - 1)
    pick = np.abs(ang[hi] - elev) < np.abs(ang[lo] - elev)
    rows = np.where(pick, hi, lo).astype(np.int32)
    # Reject points outside the sensor's elevation envelope.
    good &= (elev >= ang[0] - 1.0) & (elev <= ang[-1] + 1.0)
    return np.where(good, rows, -1), cols, r
