"""
Synthetic spinning-LiDAR simulator.

This is not a point-cloud sketch; it is a sensor model.  64 rings at fixed
elevation angles, 2048 azimuth steps, and every point is produced by
RAYCASTING a beam into the scene and taking the first hit.

WHY RAYCASTING AND NOT SCATTERED POINTS
---------------------------------------
Almost every claim this project makes depends on properties that only appear
when the sampling geometry is real:

  * ground density falls off as ~1/r^3, which is why a fixed 5 cm grid is
    pointless at 80 m and why the derived resolution law has any content;
  * occlusion is real, so a tile behind a wall has *no* returns rather than
    sparse ones — which is the difference between "empty" and "unobserved"
    that the observability term depends on;
  * sparsity at range is what makes retaining the 70 m pedestrian hard, and
    therefore what makes retaining it worth demonstrating.

Scattering random points reproduces none of these, so a system tuned against
scattered points would be tuned against the wrong problem.

CONTENTS (the mixed_urban scene)
--------------------------------
undulating ground, a road strip, kerbs (12 cm), potholes (15 cm), building
walls, poles, trees whose canopy overhangs the road, parked and moving
vehicles, and pedestrians — including one at 60-80 m, the demo's centrepiece.

Deterministic given a seed: same seed, identical scene, byte for byte.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from adaptive_lidar.data.label_maps import (
    GROUND_DRIVABLE,
    GROUND_ROUGH,
    STATIC_OBSTACLE,
    VEGETATION,
    VEHICLE,
    VRU,
)

# ── Sensor ──────────────────────────────────────────────────────
NUM_RINGS = 64
NUM_AZIMUTH = 2048
ELEV_LO, ELEV_HI = -24.0, 2.0
SENSOR_HEIGHT = 1.8
MAX_RANGE = 100.0
RANGE_NOISE = 0.02          # sigma at 0 m, grows with range

# ── Material base reflectance (NIR, 905 nm) ─────────────────────
REFLECTANCE = {
    "asphalt": 0.15,
    "concrete": 0.35,
    "road_marking": 0.90,    # retroreflective paint
    "grass": 0.45,
    "foliage": 0.50,         # vegetation is bright in NIR
    "mud": 0.20,
    "metal": 0.55,
    "glass": 0.25,
    "cloth": 0.40,           # pedestrians
    "brick": 0.30,
}


# ════════════════════════════════════════════════════════════
# Primitives — each supports a vectorised ray intersection
# ════════════════════════════════════════════════════════════
@dataclass
class Prim:
    kind: str
    cls: int
    material: str
    instance: int = -1
    moving: bool = False
    velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    params: Dict[str, Any] = field(default_factory=dict)
    penetrable: bool = False        # emits a second return


def _box_hit(origin, dirs, lo, hi):
    """Slab method, vectorised over all rays. Returns (t, normal)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / dirs
        t1 = (lo - origin) * inv
        t2 = (hi - origin) * inv
    tmin = np.minimum(t1, t2)
    tmax = np.maximum(t1, t2)
    t_near = np.nanmax(tmin, axis=1)
    t_far = np.nanmin(tmax, axis=1)
    hit = (t_far >= np.maximum(t_near, 0.0)) & (t_near > 0)
    t = np.where(hit, t_near, np.inf)

    # Normal is the axis whose t_near was the maximum.
    axis = np.argmax(np.nan_to_num(tmin, nan=-np.inf), axis=1)
    nrm = np.zeros_like(dirs)
    nrm[np.arange(len(dirs)), axis] = -np.sign(dirs[np.arange(len(dirs)), axis])
    return t, nrm


def _cylinder_hit(origin, dirs, cx, cy, radius, z0, z1):
    """Infinite vertical cylinder clipped to [z0, z1]."""
    ox, oy = origin[:, 0] - cx, origin[:, 1] - cy
    dx, dy = dirs[:, 0], dirs[:, 1]
    a = dx * dx + dy * dy
    b = 2.0 * (ox * dx + oy * dy)
    c = ox * ox + oy * oy - radius * radius
    disc = b * b - 4 * a * c
    ok = (disc > 0) & (a > 1e-12)
    sq = np.sqrt(np.where(ok, disc, 0.0))
    t0 = np.where(ok, (-b - sq) / (2 * a), np.inf)
    t1 = np.where(ok, (-b + sq) / (2 * a), np.inf)
    t = np.where(t0 > 1e-4, t0, t1)
    z = origin[:, 2] + t * dirs[:, 2]
    good = ok & (t > 1e-4) & (z >= z0) & (z <= z1)
    t = np.where(good, t, np.inf)
    t_safe = np.where(good, t, 0.0)          # inf * 0 would be NaN
    px = origin[:, 0] + t_safe * dx - cx
    py = origin[:, 1] + t_safe * dy - cy
    pn = np.hypot(px, py)
    nrm = np.stack([px / np.maximum(pn, 1e-6),
                    py / np.maximum(pn, 1e-6),
                    np.zeros_like(px)], axis=1)
    return t, nrm


def _ellipsoid_hit(origin, dirs, c, r):
    """Axis-aligned ellipsoid — used for tree canopies."""
    o = (origin - c) / r
    d = dirs / r
    a = np.sum(d * d, axis=1)
    b = 2.0 * np.sum(o * d, axis=1)
    cc = np.sum(o * o, axis=1) - 1.0
    disc = b * b - 4 * a * cc
    ok = disc > 0
    sq = np.sqrt(np.where(ok, disc, 0.0))
    t0 = np.where(ok, (-b - sq) / (2 * a), np.inf)
    t1 = np.where(ok, (-b + sq) / (2 * a), np.inf)
    t = np.where(t0 > 1e-4, t0, t1)
    t = np.where(ok & (t > 1e-4), t, np.inf)
    t_safe = np.where(np.isfinite(t), t, 0.0)
    p = origin + t_safe[:, None] * dirs - c
    n = p / (r * r)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    return t, n / np.maximum(ln, 1e-9)


# ════════════════════════════════════════════════════════════
# Ground height field
# ════════════════════════════════════════════════════════════
class GroundField:
    """Single-valued terrain height z = f(x, y).

    Potholes and kerbs are modifiers on this surface, not separate primitives:
    a pothole is a depression *in* the surface being raycast, and making it a
    box would create a second surface a beam could slip between.
    """

    def __init__(self, road_half_width=3.5, kerb_h=0.12, seed=0):
        self.road_half_width = road_half_width
        self.kerb_h = kerb_h
        rng = np.random.default_rng(seed)
        # Two low-frequency sinusoids give gentle, deterministic undulation.
        self.phase = rng.uniform(0, 2 * np.pi, 4)
        self.potholes = [
            (18.0, -1.2, 0.55, 0.15),
            (42.0, 1.4, 0.70, 0.16),
            (-12.0, 0.8, 0.50, 0.13),
        ]

    def height(self, x, y):
        z = (0.06 * np.sin(x / 23.0 + self.phase[0])
             + 0.05 * np.sin(y / 17.0 + self.phase[1])
             + 0.03 * np.sin(x / 7.0 + self.phase[2]))
        # Kerb: a step up at the road edge, then the pavement.
        ay = np.abs(y)
        w = self.road_half_width
        kerb = np.clip((ay - w) / 0.12, 0.0, 1.0) * self.kerb_h
        z = z + kerb
        # Potholes.
        for px, py, pr, pd in self.potholes:
            d = np.hypot(x - px, y - py)
            z = z - pd * np.clip(1.0 - (d / pr) ** 2, 0.0, 1.0)
        return z.astype(np.float32)

    def classify(self, x, y):
        ay = np.abs(y)
        w = self.road_half_width
        on_road = ay <= w
        # Lane markings: centre line and edge lines.
        marking = (np.abs(ay - w + 0.25) < 0.08) | (ay < 0.09)
        return np.where(on_road, GROUND_DRIVABLE, GROUND_ROUGH), (on_road & marking)

    def raycast(self, origin, dirs, t_max):
        """First intersection of each ray with the height field.

        Start from the analytic flat-plane hit at z = 0 and refine with a few
        secant steps on ``g(t) = ray_z(t) - f(ray_xy(t))``.  The field varies by
        at most ~0.3 m, so this converges in a handful of iterations without
        the dense marching a general height field would need.
        """
        dz = dirs[:, 2]
        down = dz < -1e-4
        t = np.where(down, -origin[:, 2] / np.where(down, dz, -1.0), np.inf)
        t = np.where(np.isfinite(t) & (t > 0), t, np.inf)
        live = np.isfinite(t) & (t < t_max)

        for _ in range(6):
            p = origin + np.where(live, t, 0.0)[:, None] * dirs
            g = p[:, 2] - self.height(p[:, 0], p[:, 1])
            # dg/dt ~ dz (the surface gradient is small compared with the beam).
            step = np.where(np.abs(dz) > 1e-4, g / np.where(np.abs(dz) > 1e-4, dz, 1.0), 0.0)
            t = np.where(live, t - step, t)
            t = np.where(t > 0, t, np.inf)
            live &= np.isfinite(t) & (t < t_max)

        t = np.where(live, t, np.inf)
        # Surface normal by finite differences.
        p = origin + np.where(np.isfinite(t), t, 0.0)[:, None] * dirs
        e = 0.25
        hx = (self.height(p[:, 0] + e, p[:, 1]) - self.height(p[:, 0] - e, p[:, 1])) / (2 * e)
        hy = (self.height(p[:, 0], p[:, 1] + e) - self.height(p[:, 0], p[:, 1] - e)) / (2 * e)
        n = np.stack([-hx, -hy, np.ones_like(hx)], axis=1)
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        return t, n


# ════════════════════════════════════════════════════════════
# Scene construction
# ════════════════════════════════════════════════════════════
class Scene:
    def __init__(self, ground: GroundField):
        self.ground = ground
        self.prims: List[Prim] = []
        self._next_inst = 1

    def add(self, p: Prim) -> int:
        if p.instance < 0:
            p.instance = self._next_inst
            self._next_inst += 1
        self.prims.append(p)
        return p.instance

    # ── convenience builders ─────────────────────────────────
    def box(self, cx, cy, cz, lx, ly, lz, cls, material,
            moving=False, velocity=(0, 0, 0)):
        return self.add(Prim("box", cls, material, moving=moving, velocity=velocity,
                             params={"c": np.array([cx, cy, cz], np.float64),
                                     "h": np.array([lx / 2, ly / 2, lz / 2], np.float64)}))

    def pole(self, cx, cy, radius, height, cls=STATIC_OBSTACLE, material="metal"):
        return self.add(Prim("cylinder", cls, material,
                             params={"cx": cx, "cy": cy, "r": radius,
                                     "z0": 0.0, "z1": height}))

    def tree(self, cx, cy, trunk_h=2.6, canopy_r=(3.0, 3.0, 1.9), canopy_z=4.0):
        self.add(Prim("cylinder", STATIC_OBSTACLE, "brick",
                      params={"cx": cx, "cy": cy, "r": 0.18,
                              "z0": 0.0, "z1": trunk_h}))
        # The canopy is penetrable: it emits a first return from the foliage
        # and a last return from whatever is behind it.
        return self.add(Prim("ellipsoid", VEGETATION, "foliage", penetrable=True,
                             params={"c": np.array([cx, cy, canopy_z], np.float64),
                                     "r": np.array(canopy_r, np.float64)}))

    def pedestrian(self, cx, cy, moving=False, velocity=(0, 0, 0)):
        # Torso + head, as two boxes, so the cluster has realistic vertical
        # structure for the vertical-run detector to find.
        i = self.box(cx, cy, 0.95, 0.58, 0.42, 1.36, VRU, "cloth",
                     moving=moving, velocity=velocity)
        self.add(Prim("box", VRU, "cloth", instance=i, moving=moving, velocity=velocity,
                      params={"c": np.array([cx, cy, 1.70], np.float64),
                              "h": np.array([0.13, 0.13, 0.14], np.float64)}))
        return i

    def vehicle(self, cx, cy, length=4.4, width=1.85, moving=False, velocity=(0, 0, 0)):
        i = self.box(cx, cy, 0.70, length, width, 1.10, VEHICLE, "metal",
                     moving=moving, velocity=velocity)
        self.add(Prim("box", VEHICLE, "glass", instance=i, moving=moving,
                      velocity=velocity,
                      params={"c": np.array([cx - 0.15, cy, 1.45], np.float64),
                              "h": np.array([length * 0.22, width * 0.44, 0.28], np.float64)}))
        return i


# ════════════════════════════════════════════════════════════
# The simulator
# ════════════════════════════════════════════════════════════
def _beam_directions(num_rings=NUM_RINGS, num_azimuth=NUM_AZIMUTH):
    """(H*W, 3) unit directions, plus the ring and azimuth index of each."""
    elev = np.radians(np.linspace(ELEV_LO, ELEV_HI, num_rings)).astype(np.float64)
    az = (np.arange(num_azimuth) / num_azimuth) * 2 * np.pi - np.pi
    E, A = np.meshgrid(elev, az, indexing="ij")
    ce = np.cos(E)
    dirs = np.stack([ce * np.cos(A), ce * np.sin(A), np.sin(E)], axis=-1)
    ring = np.repeat(np.arange(num_rings, dtype=np.int16), num_azimuth)
    abin = np.tile(np.arange(num_azimuth, dtype=np.int32), num_rings)
    return dirs.reshape(-1, 3), ring, abin


def _intersect_all(scene: Scene, origin, dirs, t_max, skip_penetrable=False):
    """Nearest hit across all primitives + the ground.

    Returns t, normal, class, instance, material index, and a penetrable flag.
    """
    n = len(dirs)
    best_t = np.full(n, np.inf)
    best_n = np.zeros((n, 3))
    best_c = np.full(n, -1, np.int8)
    best_i = np.full(n, -1, np.int32)
    best_m = np.zeros(n, np.int16)
    best_p = np.zeros(n, bool)
    mats = list(REFLECTANCE)

    def take(t, nrm, cls, inst, mat, pen):
        nonlocal best_t, best_n, best_c, best_i, best_m, best_p
        better = t < best_t
        if not np.any(better):
            return
        best_t = np.where(better, t, best_t)
        best_n = np.where(better[:, None], nrm, best_n)
        best_c = np.where(better, cls, best_c)
        best_i = np.where(better, inst, best_i)
        best_m = np.where(better, mats.index(mat), best_m)
        best_p = np.where(better, pen, best_p)

    for p in scene.prims:
        if skip_penetrable and p.penetrable:
            continue
        if p.kind == "box":
            c, h = p.params["c"], p.params["h"]
            t, nrm = _box_hit(origin, dirs, c - h, c + h)
        elif p.kind == "cylinder":
            t, nrm = _cylinder_hit(origin, dirs, p.params["cx"], p.params["cy"],
                                   p.params["r"], p.params["z0"], p.params["z1"])
        elif p.kind == "ellipsoid":
            t, nrm = _ellipsoid_hit(origin, dirs, p.params["c"], p.params["r"])
        else:
            continue
        t = np.where(t < t_max, t, np.inf)
        take(t, nrm, p.cls, p.instance, p.material, p.penetrable)

    gt, gn = scene.ground.raycast(origin, dirs, t_max)
    gp = origin + np.where(np.isfinite(gt), gt, 0.0)[:, None] * dirs
    gcls, marking = scene.ground.classify(gp[:, 0], gp[:, 1])
    better = gt < best_t
    best_t = np.where(better, gt, best_t)
    best_n = np.where(better[:, None], gn, best_n)
    best_c = np.where(better, gcls, best_c)
    best_i = np.where(better, 0, best_i)
    gmat = np.where(marking, mats.index("road_marking"),
                    np.where(gcls == GROUND_DRIVABLE,
                             mats.index("asphalt"), mats.index("grass")))
    best_m = np.where(better, gmat, best_m)
    best_p = np.where(better, False, best_p)

    return best_t, best_n, best_c, best_i, best_m, best_p


def simulate_scan(
    scene: Scene,
    sensor_xy: Tuple[float, float] = (0.0, 0.0),
    heading: float = 0.0,
    seed: int = 0,
    num_rings: int = NUM_RINGS,
    num_azimuth: int = NUM_AZIMUTH,
    max_range: float = MAX_RANGE,
) -> Dict[str, np.ndarray]:
    """One full rotation. Returns per-point arrays in the SENSOR frame."""
    rng = np.random.default_rng(seed)
    dirs, ring, abin = _beam_directions(num_rings, num_azimuth)

    # Sensor pose. Heading rotates the beams into world orientation; the
    # returned points are rotated back, so they are sensor-frame as a real
    # sensor would deliver them.
    ch, sh = np.cos(heading), np.sin(heading)
    Rz = np.array([[ch, -sh, 0.0], [sh, ch, 0.0], [0.0, 0.0, 1.0]])
    world_dirs = dirs @ Rz.T
    sensor_z = scene.ground.height(np.array([sensor_xy[0]]),
                                   np.array([sensor_xy[1]]))[0] + SENSOR_HEIGHT
    origin = np.tile(np.array([sensor_xy[0], sensor_xy[1], sensor_z]),
                     (len(dirs), 1))

    t, nrm, cls, inst, mat, pen = _intersect_all(scene, origin, world_dirs, max_range)

    hit = np.isfinite(t) & (t > 0.3)
    # Beam dropout — a real sensor loses a small fraction of returns.
    hit &= rng.random(len(t)) > 0.004

    def build(mask, t_vals, nrm_v, cls_v, inst_v, mat_v, ret_no, ret_cnt):
        r = t_vals[mask]
        sigma = RANGE_NOISE * (1.0 + r / 120.0)
        r = r + rng.normal(0.0, 1.0, len(r)) * sigma
        p_world = origin[mask] + r[:, None] * world_dirs[mask]
        # Back into the sensor frame.
        p = (p_world - origin[mask]) @ Rz
        n_w = nrm_v[mask]
        cos_i = np.abs(np.sum(n_w * world_dirs[mask], axis=1))
        cos_i = np.clip(cos_i, 0.05, 1.0)
        base = np.array(list(REFLECTANCE.values()))[mat_v[mask]]
        # LiDAR equation, normalised so a 20 m asphalt return sits near 0.15.
        inten = base * cos_i * (20.0 / np.maximum(r, 0.5)) ** 2
        inten *= (1.0 + rng.normal(0.0, 0.05, len(r)))
        return {
            "points": p.astype(np.float32),
            "intensity": np.clip(inten, 0.0, 1.0).astype(np.float32),
            "ring": ring[mask],
            "azimuth_bin": abin[mask],
            "gt_label": cls_v[mask].astype(np.int8),
            "gt_instance": inst_v[mask].astype(np.int32),
            "return_number": np.full(int(mask.sum()), ret_no, np.int8),
            "return_count": np.full(int(mask.sum()), ret_cnt, np.int8),
            "_incidence": cos_i.astype(np.float32),
        }

    first = build(hit, t, nrm, cls, inst, mat, 1, 1)

    # ── multi-echo: vegetation returns twice ─────────────────
    # First from the foliage, last from whatever is behind it.  Solid surfaces
    # return once. This is what makes the penetration-ratio layer demonstrable
    # without adding noise to a feature whose whole signal is "this is foliage".
    veg = hit & pen
    second = None
    if np.any(veg):
        idx = np.flatnonzero(veg)
        o2 = origin[idx] + (t[idx] + 0.35)[:, None] * world_dirs[idx]
        t2, n2, c2, i2, m2, _ = _intersect_all(
            scene, o2, world_dirs[idx], max_range, skip_penetrable=True)
        ok = np.isfinite(t2)
        if np.any(ok):
            sub = idx[ok]
            r2 = t[sub] + 0.35 + t2[ok]
            sigma = RANGE_NOISE * (1.0 + r2 / 120.0)
            r2 = r2 + rng.normal(0.0, 1.0, len(r2)) * sigma
            pw = origin[sub] + r2[:, None] * world_dirs[sub]
            p2 = (pw - origin[sub]) @ Rz
            cos_i = np.clip(np.abs(np.sum(n2[ok] * world_dirs[sub], axis=1)), 0.05, 1.0)
            base = np.array(list(REFLECTANCE.values()))[m2[ok]]
            # A last return through foliage is attenuated.
            inten = 0.45 * base * cos_i * (20.0 / np.maximum(r2, 0.5)) ** 2
            second = {
                "points": p2.astype(np.float32),
                "intensity": np.clip(inten, 0, 1).astype(np.float32),
                "ring": ring[sub],
                "azimuth_bin": abin[sub],
                "gt_label": c2[ok].astype(np.int8),
                "gt_instance": i2[ok].astype(np.int32),
                "return_number": np.full(len(sub), 2, np.int8),
                "return_count": np.full(len(sub), 2, np.int8),
                "_incidence": cos_i.astype(np.float32),
            }
            # Mark the first returns of those beams as the first of two.
            fmap = np.zeros(len(t), bool)
            fmap[sub] = True
            first["return_count"][fmap[hit]] = 2

    out = first if second is None else {
        k: np.concatenate([first[k], second[k]]) for k in first}

    # Per-point moving flag from the instance's primitive.
    moving_ids = {p.instance for p in scene.prims if p.moving}
    out["gt_moving"] = np.isin(out["gt_instance"],
                               np.fromiter(moving_ids or {-999}, np.int32))
    out["sensor_z"] = sensor_z
    return out


# ════════════════════════════════════════════════════════════
# Scenarios
# ════════════════════════════════════════════════════════════
SCENARIOS = ("empty_road", "pedestrian_far", "canopy_over_road",
             "moving_vehicle", "mixed_urban")

#: The demo's centrepiece — a pedestrian at exactly this range, which is where
#: a uniform coarse grid loses them and the adaptive map must not.
FAR_PEDESTRIAN_X = 70.0
FAR_PEDESTRIAN_Y = 1.6


def build_scene(name: str = "mixed_urban", t: float = 0.0, seed: int = 42) -> Scene:
    """Construct the scene for a scenario at simulated time ``t`` seconds."""
    g = GroundField(seed=seed)
    s = Scene(g)

    if name == "empty_road":
        return s

    if name == "pedestrian_far":
        s.pedestrian(FAR_PEDESTRIAN_X, FAR_PEDESTRIAN_Y)
        return s

    if name == "canopy_over_road":
        # Canopy centred over the road: z_max would read 4 m and mark the road
        # BLOCKED unless overhead_clearance is handled properly.
        s.tree(14.0, 0.0, canopy_r=(4.5, 5.0, 2.0), canopy_z=4.2)
        s.tree(30.0, 0.5, canopy_r=(4.0, 5.5, 1.8), canopy_z=4.0)
        return s

    if name == "moving_vehicle":
        # Crosses the road over ~20 frames at 10 Hz.
        s.vehicle(18.0, -14.0 + 8.0 * t, moving=True, velocity=(0.0, 8.0, 0.0))
        return s

    # ── mixed_urban: everything at once ──────────────────────
    rng = np.random.default_rng(seed)

    # Buildings along both sides.
    for x in range(-20, 100, 24):
        s.box(x + 10.0, 11.5, 4.0, 22.0, 6.0, 8.0, STATIC_OBSTACLE, "brick")
        s.box(x + 10.0, -11.5, 3.6, 22.0, 6.0, 7.2, STATIC_OBSTACLE, "concrete")

    # Poles — thin, 0.1 m radius, 4 m tall.
    for x in (8.0, 22.0, 36.0, 52.0, 66.0, 84.0):
        s.pole(x, 5.2, 0.10, 4.2)
    s.pole(30.0, -5.2, 0.10, 4.0)
    s.pole(58.0, -5.2, 0.10, 4.5)

    # Trees whose canopies overhang the road.
    s.tree(26.0, 6.4, canopy_r=(3.4, 4.2, 1.9), canopy_z=4.1)
    s.tree(48.0, -6.6, canopy_r=(3.2, 4.4, 1.8), canopy_z=4.0)
    s.tree(74.0, 6.8, canopy_r=(3.0, 3.6, 1.7), canopy_z=3.9)

    # Parked vehicles.
    s.vehicle(12.0, 5.0)
    s.vehicle(40.0, -5.1, length=5.2, width=2.0)
    s.vehicle(62.0, 5.1)

    # Moving vehicles.
    s.vehicle(34.0 + 12.0 * t, -1.9, moving=True, velocity=(12.0, 0.0, 0.0))
    s.vehicle(90.0 - 9.0 * t, 1.9, moving=True, velocity=(-9.0, 0.0, 0.0))

    # Pedestrians, including the far one.
    s.pedestrian(9.0, 4.4)
    s.pedestrian(21.0 + 1.3 * t, -4.6, moving=True, velocity=(1.3, 0.0, 0.0))
    s.pedestrian(FAR_PEDESTRIAN_X, FAR_PEDESTRIAN_Y)     # ← the centrepiece

    # A low wall and a few bushes for clutter.
    s.box(45.0, 7.6, 0.55, 14.0, 0.35, 1.10, STATIC_OBSTACLE, "concrete")
    for bx, by in ((17.0, -7.4), (55.0, 7.2), (80.0, -7.0)):
        s.add(Prim("ellipsoid", VEGETATION, "foliage", penetrable=True,
                   params={"c": np.array([bx, by, 0.7]),
                           "r": np.array([1.1, 1.1, 0.7])}))
    return s


def generate_scenario(
    name: str = "mixed_urban",
    frame_idx: int = 0,
    num_frames: int = 20,
    seed: int = 42,
    ego_speed: float = 10.0,
    num_rings: int = NUM_RINGS,
    num_azimuth: int = NUM_AZIMUTH,
    static_ego: bool = False,
) -> Dict[str, Any]:
    """One frame of a scenario, as a dict of per-point arrays + a pose.

    The sensor travels along the road at ``ego_speed`` m/s (10 Hz frames) and
    moving objects move independently with known velocities.
    """
    fps = 10.0
    t = frame_idx / fps
    scene = build_scene(name, t=t, seed=seed)

    ego_x = 0.0 if static_ego else ego_speed * t
    ego_y = 0.0
    scan = simulate_scan(scene, sensor_xy=(ego_x, ego_y), heading=0.0,
                         seed=seed * 1000 + frame_idx,
                         num_rings=num_rings, num_azimuth=num_azimuth)

    pose = np.eye(4)
    pose[0, 3] = ego_x
    pose[1, 3] = ego_y
    pose[2, 3] = scan.pop("sensor_z")

    scan["pose"] = pose
    scan["frame_id"] = frame_idx
    scan["timestamp"] = t
    scan["scenario"] = name
    return scan


def generate_scenario_frames(name: str = "mixed_urban", num_frames: int = 20,
                             **kw) -> List[Dict[str, Any]]:
    return [generate_scenario(name, i, num_frames, **kw) for i in range(num_frames)]


# ════════════════════════════════════════════════════════════
# Backwards-compatible shims
# ════════════════════════════════════════════════════════════
def generate_scene(frame_idx: int = 0, num_frames: int = 8) -> np.ndarray:
    """(N,4) [x y z intensity] — the old interface, kept so nothing breaks."""
    s = generate_scenario("mixed_urban", frame_idx, num_frames)
    return np.column_stack([s["points"], s["intensity"]]).astype(np.float32)


def generate_multi_frame(num_frames: int = 8) -> List[np.ndarray]:
    return [generate_scene(i, num_frames) for i in range(num_frames)]
