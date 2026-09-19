"""
Synthetic LiDAR scene generator.

Produces realistic-looking multi-frame LiDAR scans without any external dataset.
The scene contains:
  - ground plane + road + curb + sidewalk
  - 2-3 parked vehicles
  - 1 moving vehicle (moves across frames)
  - 1-2 pedestrians (one stationary, one slowly moving)
  - vegetation clusters
  - a vertical obstacle / wall
  - distant sparse returns

Each frame is returned as (N, 4) numpy array: x y z intensity
"""
from __future__ import annotations
import numpy as np
from typing import List, Tuple


RNG = np.random.default_rng(42)


# ─── helpers ────────────────────────────────────────────────────
def _jitter(arr: np.ndarray, sigma: float = 0.02) -> np.ndarray:
    return arr + RNG.normal(0, sigma, arr.shape)


def _box_points(
    cx: float, cy: float, cz: float,
    lx: float, ly: float, lz: float,
    n: int = 200,
    intensity: float = 0.5,
) -> np.ndarray:
    """Sample points on the surface of a box."""
    pts = []
    faces = [
        # -x +x
        (RNG.uniform(cx - lx/2, cx + lx/2, n//6),
         RNG.uniform(cy - ly/2, cy + ly/2, n//6),
         np.full(n//6, cz - lz/2)),
        (RNG.uniform(cx - lx/2, cx + lx/2, n//6),
         RNG.uniform(cy - ly/2, cy + ly/2, n//6),
         np.full(n//6, cz + lz/2)),
        (np.full(n//6, cx - lx/2),
         RNG.uniform(cy - ly/2, cy + ly/2, n//6),
         RNG.uniform(cz - lz/2, cz + lz/2, n//6)),
        (np.full(n//6, cx + lx/2),
         RNG.uniform(cy - ly/2, cy + ly/2, n//6),
         RNG.uniform(cz - lz/2, cz + lz/2, n//6)),
        (RNG.uniform(cx - lx/2, cx + lx/2, n//6),
         np.full(n//6, cy - ly/2),
         RNG.uniform(cz - lz/2, cz + lz/2, n//6)),
        (RNG.uniform(cx - lx/2, cx + lx/2, n//6),
         np.full(n//6, cy + ly/2),
         RNG.uniform(cz - lz/2, cz + lz/2, n//6)),
    ]
    for x, y, z in faces:
        pts.append(np.stack([x, y, z, np.full(len(x), intensity)], axis=1))
    return np.concatenate(pts, axis=0)


def _ground_plane(x_range, y_range, n: int = 3000) -> np.ndarray:
    x = RNG.uniform(*x_range, n)
    y = RNG.uniform(*y_range, n)
    z = RNG.normal(0.0, 0.02, n)
    intensity = RNG.uniform(0.1, 0.3, n)
    return np.stack([x, y, z, intensity], axis=1)


def _curb(side: float = 3.5, n: int = 400) -> np.ndarray:
    """Simple road curb as raised strip."""
    pts = []
    for sign in [-1, 1]:
        y_c = sign * side
        x = RNG.uniform(-25, 25, n // 2)
        y = RNG.normal(y_c, 0.05, n // 2)
        z = RNG.uniform(0.0, 0.15, n // 2)
        intensity = np.full(n // 2, 0.6)
        pts.append(np.stack([x, y, z, intensity], axis=1))
    return np.concatenate(pts)


def _vegetation_cluster(cx, cy, n: int = 150) -> np.ndarray:
    """Irregular vertical cluster simulating bush/tree."""
    x = RNG.normal(cx, 0.6, n)
    y = RNG.normal(cy, 0.6, n)
    z = RNG.uniform(0.0, RNG.uniform(0.5, 2.5), n)
    # irregular density
    z = z * RNG.uniform(0.3, 1.0, n)
    intensity = RNG.uniform(0.05, 0.25, n)
    return np.stack([x, y, z, intensity], axis=1)


def _pedestrian(cx, cy, frame_offset: float = 0.0) -> np.ndarray:
    """Thin vertical cluster."""
    n = 80
    x = RNG.normal(cx + frame_offset * 0.3, 0.15, n)
    y = RNG.normal(cy, 0.15, n)
    z = RNG.uniform(0.0, 1.75, n)
    intensity = RNG.uniform(0.3, 0.6, n)
    return np.stack([x, y, z, intensity], axis=1)


def _distant_sparse(n: int = 500) -> np.ndarray:
    """Distant sparse returns."""
    angle = RNG.uniform(0, 2 * np.pi, n)
    r = RNG.uniform(45, 75, n)
    x = r * np.cos(angle)
    y = r * np.sin(angle)
    z = RNG.uniform(-0.5, 3.0, n)
    intensity = RNG.uniform(0.01, 0.15, n)
    return np.stack([x, y, z, intensity], axis=1)


# ─── public API ─────────────────────────────────────────────────
def generate_scene(frame_idx: int = 0, num_frames: int = 8) -> np.ndarray:
    """
    Generate one synthetic LiDAR frame as (N, 4) array [x, y, z, intensity].

    frame_idx: 0 … num_frames-1
    Moving vehicle travels along x axis.
    Moving pedestrian drifts slowly.
    """
    t = frame_idx / max(num_frames - 1, 1)  # normalised time [0, 1]

    parts = []

    # 1. Ground plane (road)
    parts.append(_ground_plane((-30, 30), (-3.5, 3.5), n=2500))

    # 2. Sidewalk (slight roughness, higher z)
    sidewalk = _ground_plane((-25, 25), (3.5, 7.0), n=800)
    sidewalk[:, 2] += 0.12
    parts.append(sidewalk)
    sidewalk_l = _ground_plane((-25, 25), (-7.0, -3.5), n=800)
    sidewalk_l[:, 2] += 0.12
    parts.append(sidewalk_l)

    # 3. Curbs
    parts.append(_curb(side=3.5, n=500))

    # 4. Parked vehicles (static)
    parts.append(_box_points(10, 5.5, 0.75, 4.5, 1.8, 1.5, n=350, intensity=0.7))
    parts.append(_box_points(-8, -5.5, 0.75, 4.2, 1.8, 1.45, n=350, intensity=0.65))
    parts.append(_box_points(20, 5.5, 0.75, 4.5, 1.8, 1.5, n=300, intensity=0.72))

    # 5. Moving vehicle — travels from x=-25 to x=+25
    mv_x = -25 + t * 50
    mv_y = -1.5  # inner lane
    parts.append(_box_points(mv_x, mv_y, 0.75, 4.5, 1.8, 1.5, n=400, intensity=0.8))

    # 6. Pedestrians
    parts.append(_pedestrian(5, 4.5))          # stationary
    ped_x = -5 + t * 8                         # slowly walking
    parts.append(_pedestrian(ped_x, -5.0))

    # 7. Vegetation clusters
    for vx, vy in [(-15, 6), (0, 7), (15, 6), (-20, -6), (5, -7)]:
        parts.append(_vegetation_cluster(vx, vy))

    # 8. Vertical obstacle / wall section
    wall = _box_points(-30, 0, 1.0, 0.3, 10.0, 2.0, n=500, intensity=0.4)
    parts.append(wall)

    # 9. Distant sparse returns
    parts.append(_distant_sparse(n=400))

    cloud = np.concatenate(parts, axis=0)

    # Apply jitter to simulate sensor noise
    cloud[:, :3] = _jitter(cloud[:, :3], sigma=0.025)

    # Filter to max range and remove NaN
    r = np.linalg.norm(cloud[:, :3], axis=1)
    valid = np.isfinite(r) & (r > 0.3) & (r < 80.0)
    cloud = cloud[valid]

    return cloud.astype(np.float32)


def generate_multi_frame(num_frames: int = 8) -> List[np.ndarray]:
    """Return a list of num_frames synthetic LiDAR scans."""
    return [generate_scene(i, num_frames) for i in range(num_frames)]
