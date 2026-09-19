"""
Multi-object tracking — Hungarian association + constant-velocity Kalman.

Produces the OBJECT TABLE: ``id, class, centroid, extent, velocity,
covariance, age, n_obs, state``.

TWO DESIGN POINTS THAT MATTER
-----------------------------
1. **Velocity lives here, never in map cells.**  One car covers ~200 cells;
   storing its velocity 200 times is redundant and goes inconsistent the moment
   the estimate updates.  Cells hold a 2-byte object handle instead.

2. **Three states, not two.**  ``MOVABLE_BUT_STATIONARY`` is a parked car: it
   belongs in the persistent map (it is really there) but it must not be
   treated as permanent structure.  Collapsing it into STATIC gives permanent
   holes in car parks when it drives away; collapsing it into MOVING means
   parked cars never enter the map at all.

Tracking runs in WORLD coordinates so that ego motion is not mistaken for
object motion.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from adaptive_lidar.pipeline.types import Instance, TrackState

try:
    from scipy.optimize import linear_sum_assignment
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False

MOVABLE_CLASSES = (3, 4)


class _Track:
    """Constant-velocity Kalman filter on a 6-state [x y z vx vy vz]."""

    __slots__ = ("id", "x", "P", "cls", "age", "n_obs", "misses",
                 "bbox_min", "bbox_max", "state", "last_t",
                 "origin", "path_len", "_prev_pos")

    def __init__(self, tid: int, inst: Instance, t: float,
                 pos_var: float, vel_var: float,
                 world_pos: np.ndarray = None):
        self.id = tid
        self.x = np.zeros(6, dtype=np.float64)
        # WORLD position, not the sensor-frame centroid the detection carries.
        # Seeding origin from the sensor frame and then overwriting the state
        # with world coordinates made every stationary track accrue the EGO's
        # own displacement, so a parked car read as travelling 12 m.
        p0 = (np.asarray(inst.centroid, dtype=np.float64) if world_pos is None
              else np.asarray(world_pos, dtype=np.float64))
        self.x[:3] = p0
        self.P = np.diag([pos_var, pos_var, pos_var,
                          vel_var, vel_var, vel_var]).astype(np.float64)
        self.cls = inst.semantic_class
        self.age = 1
        self.n_obs = 1
        self.misses = 0
        self.bbox_min = inst.bbox_min.copy()
        self.bbox_max = inst.bbox_max.copy()
        self.state = TrackState.STATIC
        self.last_t = t
        # Where this track was first seen, and how far it has actually
        # travelled since — see _state_of for why both are needed.
        self.origin = p0.copy()
        self.path_len = 0.0
        self._prev_pos = p0.copy()

    def predict(self, dt: float, q_pos: float, q_vel: float):
        F = np.eye(6)
        F[0, 3] = F[1, 4] = F[2, 5] = dt
        self.x = F @ self.x
        Q = np.diag([q_pos, q_pos, q_pos, q_vel, q_vel, q_vel]) * max(dt, 1e-3)
        self.P = F @ self.P @ F.T + Q
        self.age += 1

    def update(self, z: np.ndarray, r_meas: float):
        H = np.zeros((3, 6))
        H[0, 0] = H[1, 1] = H[2, 2] = 1.0
        R = np.eye(3) * r_meas
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ H) @ self.P
        self.n_obs += 1
        self.misses = 0
        self.path_len += float(np.linalg.norm(self.x[:3] - self._prev_pos))
        self._prev_pos = self.x[:3].copy()

    @property
    def pos(self) -> np.ndarray:
        return self.x[:3]

    @property
    def vel(self) -> np.ndarray:
        return self.x[3:]

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.x[3:]))

    @property
    def displacement(self) -> float:
        """How far the track has actually got from where it started."""
        return float(np.linalg.norm(self.x[:3] - self.origin))

    @property
    def diag(self) -> float:
        """The object's own footprint diagonal — the scale its apparent
        centroid can drift over without the object having moved at all."""
        return float(np.linalg.norm((self.bbox_max - self.bbox_min)[:2]))

    @property
    def straightness(self) -> float:
        """Net displacement over path length: 1 = went somewhere, 0 = wandered.

        THE DISCRIMINATOR FOR A PARKED CAR SEEN FROM A MOVING VEHICLE.
        As the ego approaches a stationary object, the set of surfaces the
        beams can reach changes — you see less of its rear and more of its
        side — so the centroid of the observed points drifts by a metre or
        two even though nothing moved. Instantaneous speed cannot tell that
        apart from real motion: in the `convoy` scenario the parked car and
        the car travelling at the ego speed BOTH show ~1 m of centroid
        movement per frame.

        What separates them is coherence. Real motion accumulates in one
        direction, so net displacement tracks path length. Visibility drift
        wanders inside the object's own footprint and cancels: measured on
        `convoy`, the two genuinely moving vehicles score 1.00 and the parked
        one scores 0.10 over 15.6 m of wandered path.
        """
        return self.displacement / max(self.path_len, 1e-6)


class Tracker:
    def __init__(self, config: Dict[str, Any]):
        t = config.get("tracking", {})
        self.gate_m = float(t.get("gate_distance", 3.5))
        self.class_penalty = float(t.get("class_mismatch_penalty", 2.5))
        self.max_misses = int(t.get("max_misses", 3))
        self.min_speed = float(t.get("min_speed", 0.6))       # m/s to call MOVING
        self.confirm_obs = int(t.get("confirm_observations", 2))
        # Net displacement / path length required to call a track MOVING, and
        # the absolute distance it must have covered. Both guard against
        # visibility-induced centroid drift on stationary objects.
        self.min_straightness = float(t.get("min_straightness", 0.5))
        self.min_displacement = float(t.get("min_displacement", 1.5))
        self.extent_factor = float(t.get("displacement_extent_factor", 1.0))
        self.q_pos = float(t.get("process_noise_pos", 0.05))
        self.q_vel = float(t.get("process_noise_vel", 2.0))
        self.r_meas = float(t.get("measurement_noise", 0.15))
        self._tracks: Dict[int, _Track] = {}
        self._next_id = 1
        self._last_t: float | None = None

    # ────────────────────────────────────────────────────────
    def update(self, detections: List[Instance], timestamp: float,
               pose: np.ndarray | None) -> List[Instance]:
        """Associate detections to tracks and return the object table.

        Detections arrive in the SENSOR frame; tracking happens in WORLD.
        """
        T = np.eye(4) if pose is None else np.asarray(pose, dtype=np.float64)
        R, tr = T[:3, :3], T[:3, 3]

        dt = 0.1 if self._last_t is None else max(timestamp - self._last_t, 1e-3)
        self._last_t = timestamp
        for trk in self._tracks.values():
            trk.predict(dt, self.q_pos, self.q_vel)

        world_cent = np.array([d.centroid for d in detections], dtype=np.float64) @ R.T + tr \
            if detections else np.zeros((0, 3))

        assigned = self._associate(detections, world_cent)

        # ── update / spawn ────────────────────────────────────
        seen = set()
        for di, ti in assigned.items():
            trk = self._tracks[ti]
            trk.update(world_cent[di], self.r_meas)
            trk.cls = detections[di].semantic_class
            trk.bbox_min = detections[di].bbox_min
            trk.bbox_max = detections[di].bbox_max
            seen.add(ti)

        for di, det in enumerate(detections):
            if di in assigned:
                continue
            tid = self._next_id
            self._next_id += 1
            self._tracks[tid] = _Track(tid, det, timestamp,
                                       pos_var=0.5, vel_var=4.0,
                                       world_pos=world_cent[di])
            assigned[di] = tid
            seen.add(tid)

        # ── age out ───────────────────────────────────────────
        for tid, trk in list(self._tracks.items()):
            if tid not in seen:
                trk.misses += 1
                if trk.misses > self.max_misses:
                    del self._tracks[tid]

        # ── classify state and emit the object table ──────────
        out: List[Instance] = []
        inv_R = R.T
        for di, det in enumerate(detections):
            tid = assigned[di]
            trk = self._tracks[tid]
            trk.state = self._state_of(trk)
            det.cluster_id = det.cluster_id if det.cluster_id >= 0 else det.instance_id
            det.instance_id = tid
            # Report velocity in the sensor frame for the current frame's use.
            det.velocity = (inv_R @ trk.vel).astype(np.float32)
            det.covariance = trk.P.astype(np.float32)
            det.state = trk.state
            det.is_dynamic = trk.state == TrackState.MOVING
            det.age = trk.age
            det.n_obs = trk.n_obs
            det.frames_seen = trk.n_obs
            if trk.state == TrackState.MOVING:
                det.motion_probability = float(
                    np.clip(0.55 + (trk.speed - self.min_speed) * 0.25, 0.55, 1.0))
            else:
                det.motion_probability = float(
                    np.clip(trk.speed / max(self.min_speed, 1e-3) * 0.35, 0.0, 0.5))
            out.append(det)
        return out

    # ────────────────────────────────────────────────────────
    def _associate(self, detections, world_cent) -> Dict[int, int]:
        """Hungarian assignment on centroid distance + class mismatch."""
        if not detections or not self._tracks:
            return {}
        tids = list(self._tracks)
        tpos = np.array([self._tracks[t].pos for t in tids])
        tcls = np.array([self._tracks[t].cls for t in tids])
        dcls = np.array([d.semantic_class for d in detections])

        cost = np.linalg.norm(world_cent[:, None, :] - tpos[None, :, :], axis=2)
        cost = cost + self.class_penalty * (dcls[:, None] != tcls[None, :])

        big = self.gate_m + self.class_penalty + 1e3
        gated = np.where(cost <= self.gate_m + self.class_penalty, cost, big)

        if _HAVE_SCIPY:
            rows, cols = linear_sum_assignment(gated)
        else:  # pragma: no cover - greedy fallback
            rows, cols = _greedy(gated)

        return {int(r): tids[int(c)] for r, c in zip(rows, cols)
                if gated[r, c] < big}

    def _state_of(self, trk: _Track) -> int:
        movable = trk.cls in MOVABLE_CLASSES
        if trk.n_obs < self.confirm_obs:
            # Not yet confirmed: assume the cautious reading for movable things.
            return TrackState.MOVABLE_BUT_STATIONARY if movable else TrackState.STATIC
        if not movable:
            return TrackState.STATIC

        # MOVING requires speed AND coherence. Speed alone calls a parked car
        # moving, because its apparent centroid drifts as the ego drives past
        # and the visible surfaces change. See _Track.straightness.
        # The gate scales with the object's OWN SIZE. Visibility drift is
        # bounded by the footprint - as the ego drives past a 4.6 m car the
        # observed centroid slides along it and no further - so "has it moved
        # further than its own length?" is the question that separates a
        # parked car from a moving one. Measured on `convoy`: the parked car
        # drifts 4.1 m against a 5.0 m diagonal, the two moving vehicles
        # cover 19-20 m against diagonals of 4.8 and 2.3.
        gate = max(self.min_displacement, self.extent_factor * trk.diag)
        going_somewhere = (trk.straightness >= self.min_straightness
                           and trk.displacement >= gate)
        if trk.speed > self.min_speed and going_somewhere:
            return TrackState.MOVING
        return TrackState.MOVABLE_BUT_STATIONARY


def _greedy(cost: np.ndarray):  # pragma: no cover
    rows, cols = [], []
    c = cost.copy()
    for _ in range(min(c.shape)):
        r, col = np.unravel_index(np.argmin(c), c.shape)
        rows.append(r)
        cols.append(col)
        c[r, :] = np.inf
        c[:, col] = np.inf
    return np.array(rows), np.array(cols)
