"""
S0 — LiDAR Ingestion.

Packages whatever the loader produced into a :class:`Frame` with every
point-level contract field allocated at the right length and dtype, so no
later stage has to guess whether a field exists.

Accepts either a bare (N,4) array or a dict from ``data.loader`` carrying
``points``, ``intensity``, ``ring``, ``azimuth_bin``, ``gt_label``, ``pose``,
``return_number`` / ``return_count``.

Filtering (NaN, range window) is applied ONCE here and every optional
per-point array is filtered with the same mask, so index alignment holds by
construction rather than by convention.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import numpy as np

from adaptive_lidar.pipeline.timing import stage_timer
from adaptive_lidar.pipeline.types import NUM_CLASSES, Frame


def load_bin(path: str) -> np.ndarray:
    """KITTI-style .bin → (N, 4) float32."""
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)


def load_npy(path: str) -> np.ndarray:
    pts = np.load(path)
    if pts.ndim != 2 or pts.shape[1] not in (3, 4):
        raise ValueError(f"Expected Nx3 or Nx4 array, got shape {pts.shape}")
    if pts.shape[1] == 3:
        pts = np.column_stack([pts, np.zeros(len(pts), np.float32)])
    return pts.astype(np.float32)


def load_ply(path: str) -> np.ndarray:
    """Minimal PLY reader for x y z. No open3d dependency."""
    with open(path, "rb") as f:
        while True:
            line = f.readline().decode("ascii", errors="replace").strip()
            if line == "end_header" or line == "":
                break
        data = np.frombuffer(f.read(), dtype=np.float32)
    n = data.size // 3
    pts = data[: n * 3].reshape(n, 3)
    return np.column_stack([pts, np.zeros(n, np.float32)]).astype(np.float32)


def _load_path(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".bin":
        return load_bin(path)
    if ext == ".npy":
        return load_npy(path)
    if ext == ".ply":
        return load_ply(path)
    raise ValueError(f"Unsupported file extension: {ext}")


def _take(arr, mask, dtype, fill):
    """Filter an optional per-point array, or synthesise a filled default."""
    n = int(mask.sum())
    if arr is None:
        return np.full(n, fill, dtype=dtype)
    return np.asarray(arr)[mask].astype(dtype)


def ingest(
    path: Optional[str],
    frame_id: int,
    timestamp: float,
    config: Dict[str, Any],
    synthetic_cloud: Optional[Any] = None,
) -> Frame:
    """Load, filter and package raw LiDAR into a Frame.

    ``synthetic_cloud`` may be an (N,4) array or a dict of per-point arrays.
    """
    timing: Dict[str, Any] = {}

    with stage_timer("S0", timing):
        cfg_s = config.get("sensor", {})
        max_range = float(cfg_s.get("max_range", 100.0))
        min_range = float(cfg_s.get("min_range", 0.3))

        scan: Dict[str, Any]
        if isinstance(synthetic_cloud, dict):
            scan = dict(synthetic_cloud)
        elif synthetic_cloud is not None:
            raw = np.asarray(synthetic_cloud)
            scan = {"points": raw[:, :3],
                    "intensity": raw[:, 3] if raw.shape[1] >= 4
                    else np.zeros(len(raw), np.float32)}
        elif path is not None:
            raw = _load_path(path)
            scan = {"points": raw[:, :3], "intensity": raw[:, 3]}
        else:
            raise ValueError("No input path or cloud provided to S0.")

        xyz = np.asarray(scan["points"], dtype=np.float32)[:, :3]
        inten = np.asarray(
            scan.get("intensity", np.zeros(len(xyz))), dtype=np.float32).ravel()

        # ONE mask, applied to every per-point array — alignment by construction.
        r = np.linalg.norm(xyz, axis=1)
        keep = (np.all(np.isfinite(xyz), axis=1)
                & np.isfinite(r) & (r >= min_range) & (r <= max_range))

        pts = xyz[keep]
        n = len(pts)

        frame = Frame(
            frame_id=int(frame_id),
            timestamp=float(timestamp),
            points=pts,
            intensity=inten[keep] if len(inten) == len(xyz)
            else np.zeros(n, np.float32),
            timing=timing,
        )

        # ── Point-level contract fields, all length n ─────────
        frame.ring = _take(scan.get("ring"), keep, np.int16, -1)
        frame.azimuth_bin = _take(scan.get("azimuth_bin"), keep, np.int32, -1)
        frame.return_number = _take(scan.get("return_number"), keep, np.int8, 1)
        frame.return_count = _take(scan.get("return_count"), keep, np.int8, 1)

        # EVALUATION ONLY — see the comment at Frame.gt_label.
        frame.gt_label = _take(scan.get("gt_label"), keep, np.int8, -1)
        frame.gt_instance = _take(scan.get("gt_instance"), keep, np.int32, -1)
        frame.gt_moving = _take(scan.get("gt_moving"), keep, bool, False)

        # Allocated here, filled by later stages. Present from the start so no
        # stage has to test for None.
        frame.intensity_norm = np.zeros(n, np.float32)
        frame.height_above_gnd = np.zeros(n, np.float32)
        frame.sem_evidence = np.zeros((n, NUM_CLASSES), np.float32)
        frame.sem_class = np.full(n, -1, np.int8)
        frame.sem_entropy = np.ones(n, np.float32)
        frame.moving_prob = np.zeros(n, np.float32)
        frame.instance_id = np.full(n, -1, np.int32)

        pose = scan.get("pose")
        frame.pose = np.asarray(pose, dtype=np.float64) if pose is not None else np.eye(4)

    return frame
