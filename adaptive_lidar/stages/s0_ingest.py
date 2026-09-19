"""
S0 — LiDAR Ingestion.

Supports:
  - .bin  (KITTI-style: x y z intensity, float32)
  - .npy  (N×3 or N×4)
  - .ply  (via simple reader)
  - synthetic scene (automatic fallback)

Measures S0 latency.
"""
from __future__ import annotations
import os
import time
import numpy as np
from typing import Optional, Dict, Any

from adaptive_lidar.pipeline.types import Frame
from adaptive_lidar.pipeline.timing import stage_timer


def load_bin(path: str) -> np.ndarray:
    """Load KITTI-style .bin file → (N, 4) float32."""
    pts = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    return pts


def load_npy(path: str) -> np.ndarray:
    """Load .npy file → (N, 4) float32."""
    pts = np.load(path)
    if pts.ndim != 2 or pts.shape[1] not in (3, 4):
        raise ValueError(f"Expected Nx3 or Nx4 array, got shape {pts.shape}")
    if pts.shape[1] == 3:
        intensity = np.zeros((pts.shape[0], 1), dtype=np.float32)
        pts = np.concatenate([pts, intensity], axis=1)
    return pts.astype(np.float32)


def load_ply(path: str) -> np.ndarray:
    """Minimal PLY loader (ASCII/binary) for x y z. No open3d dependency."""
    try:
        # Try numpy-based approach for simple PLY
        with open(path, "rb") as f:
            header = []
            while True:
                line = f.readline().decode("ascii", errors="replace").strip()
                header.append(line)
                if line == "end_header":
                    break
            data = np.frombuffer(f.read(), dtype=np.float32)
        n = data.size // 3
        pts = data[: n * 3].reshape(n, 3)
        intensity = np.zeros((n, 1), dtype=np.float32)
        return np.concatenate([pts, intensity], axis=1)
    except Exception:
        raise ValueError(f"Cannot parse PLY file: {path}")


def ingest(
    path: Optional[str],
    frame_id: int,
    timestamp: float,
    config: Dict[str, Any],
    synthetic_cloud: Optional[np.ndarray] = None,
) -> Frame:
    """
    S0 — Load, filter and package raw LiDAR into a Frame.

    Priority:
      1. Use synthetic_cloud if provided directly.
      2. Load from path if given.
      3. Raise if neither is available.
    """
    timing: Dict[str, float] = {}

    with stage_timer("S0", timing):
        max_range = config.get("sensor", {}).get("max_range", 75.0)
        min_range = config.get("sensor", {}).get("min_range", 0.3)

        if synthetic_cloud is not None:
            raw = synthetic_cloud
        elif path is not None:
            ext = os.path.splitext(path)[1].lower()
            if ext == ".bin":
                raw = load_bin(path)
            elif ext == ".npy":
                raw = load_npy(path)
            elif ext == ".ply":
                raw = load_ply(path)
            else:
                raise ValueError(f"Unsupported file extension: {ext}")
        else:
            raise ValueError("No input path or synthetic cloud provided to S0.")

        # Basic filtering
        xyz = raw[:, :3]
        intensity = raw[:, 3] if raw.shape[1] >= 4 else np.zeros(len(raw))

        # Remove NaN / Inf
        valid = np.all(np.isfinite(xyz), axis=1)
        xyz = xyz[valid]
        intensity = intensity[valid]

        # Range filter
        r = np.linalg.norm(xyz, axis=1)
        in_range = (r >= min_range) & (r <= max_range)
        xyz = xyz[in_range]
        intensity = intensity[in_range]

    frame = Frame(
        frame_id=frame_id,
        timestamp=timestamp,
        points=xyz.astype(np.float32),
        intensity=intensity.astype(np.float32),
        timing=timing,
    )
    return frame
