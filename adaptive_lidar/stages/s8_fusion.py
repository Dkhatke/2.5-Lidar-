"""
S8 — Temporal fusion and the MOS gate.

This stage used to be ``pass``.  Everything below is what that cost: moving
objects fused straight into the persistent map and left trails.

THE GATE
--------
    incoming points -> MOS gate
                         |- MOVING -> dynamic overlay, cleared every frame
                         `- STATIC -> persistent map, Bayesian fusion

Moving points are NEVER WRITTEN to the persistent map.  Not written and then
decayed out of it — never written.  Decay is a treatment and gating is a cure:
at 10 m/s a one-second decay leaves a ten-metre phantom wall behind every
passing car, which is precisely the trail the Phase 6 test measures.

STAGE ORDER
-----------
The gate necessarily runs BEFORE the map write, so the orchestrator calls
S8 then S7.  The stage decomposition is unchanged — only the dependency is
made explicit, because a gate that runs after the thing it gates is not a gate.

FUSION IS ADDITION
------------------
Every quantity the map accumulates is associative, so temporal fusion needs no
special case:
    occupancy  -> log-odds addition
    semantics  -> evidence addition (the conjugate categorical update)
    elevation  -> Kalman scalar fusion with a range-dependent variance
    dynamics   -> binary Bayes filter
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from adaptive_lidar.pipeline.timing import stage_timer
from adaptive_lidar.pipeline.types import Frame, PipelineContext, TrackState

#: Classes an object can plausibly belong to and still drive away.
MOVABLE_CLASSES = (3, 4)


class DynamicOverlay:
    """Moving objects, rebuilt from scratch every frame.

    Never persisted, so it cannot leave a trail by construction.
    """

    def __init__(self):
        self.points = np.zeros((0, 3), np.float32)
        self.classes = np.zeros(0, np.int8)
        self.instance_ids = np.zeros(0, np.int32)
        self.objects: list = []
        self.frame_id = -1

    def rebuild(self, frame: Frame, mask: np.ndarray):
        pose = frame.pose if frame.pose is not None else np.eye(4)
        pts = frame.points[mask]
        self.points = (pts @ pose[:3, :3].T + pose[:3, 3]).astype(np.float32)
        self.classes = frame.sem_class[mask]
        self.instance_ids = frame.instance_id[mask]
        self.objects = [i for i in (frame.instances or [])
                        if i.state == TrackState.MOVING]
        self.frame_id = frame.frame_id

    def __len__(self):
        return len(self.points)


class S8Fusion:
    def __init__(self, config: Dict[str, Any], ctx: PipelineContext | None = None):
        self.cfg = config
        m = config.get("motion", {})
        self.gate_thresh = float(m.get("gate_threshold", 0.5))
        self.enabled = bool(m.get("gate_enabled", True))
        self.decay_every = int(config.get("map", {}).get("decay_every_frames", 10))
        self.confirm_obs = int(
            config.get("tracking", {}).get("confirm_observations", 2)) + 1
        self.overlay = DynamicOverlay()
        if ctx is not None:
            ctx.dynamic_overlay = self.overlay

    def process(self, frame: Frame, ctx: PipelineContext):
        with stage_timer("S8", frame.timing):
            n = len(frame.points)
            if n == 0:
                frame.static_mask = np.zeros(0, bool)
                return

            if self.enabled:
                moving = self._gate(frame)
            else:
                # Ablation path: everything enters the map. This is the
                # configuration that produces the trail, and reporting its
                # number alongside the gated one is the point.
                moving = np.zeros(n, bool)

            frame.static_mask = ~moving
            frame.moving_mask = moving

            self.overlay.rebuild(frame, moving)
            ctx.dynamic_overlay = self.overlay

            # Class-dependent decay of the persistent map, run periodically
            # rather than every frame because it touches every cell.
            if ctx.amap is not None and self.decay_every > 0 \
                    and frame.frame_id % self.decay_every == 0 and frame.frame_id > 0:
                ctx.amap.decay()

            frame.timing["S8_diag"] = {
                "gate_enabled": self.enabled,
                "n_moving": int(moving.sum()),
                "n_static": int((~moving).sum()),
                "moving_fraction": float(moving.mean()),
                "overlay_objects": len(self.overlay.objects),
            }

    # ────────────────────────────────────────────────────────
    def _gate(self, frame: Frame) -> np.ndarray:
        """(N,) True where the point belongs to something that is moving.

        Two sources, OR-ed:
          - the per-point residual MOS probability, and
          - the track state, which is the more reliable of the two because it
            integrates over frames and covers the whole object rather than the
            pixels whose residual happened to clear the threshold.
        """
        n = len(frame.points)
        moving = frame.moving_prob >= self.gate_thresh

        instances = frame.instances or []
        if instances and frame.instance_id is not None:
            moving_ids = np.array(
                [i.instance_id for i in instances if i.state == TrackState.MOVING],
                dtype=np.int64)
            if moving_ids.size:
                moving |= np.isin(frame.instance_id, moving_ids)

            # Motion cannot be detected without history, so on an object's
            # first one or two frames the residual is silent and a genuinely
            # moving car writes itself into the permanent map before anything
            # can stop it - the first two frames of a 20-frame run accounted
            # for most of the trail that survived the gate.
            #
            # The fix is the third track state doing its job: a movable-class
            # object is held OUT of the persistent map until the tracker has
            # seen it long enough to say it is standing still. A parked car
            # enters the map 0.2 s late, which costs nothing; a moving car
            # never enters it at all.
            unconfirmed = np.array(
                [i.instance_id for i in instances
                 if i.semantic_class in MOVABLE_CLASSES
                 and i.n_obs < self.confirm_obs], dtype=np.int64)
            if unconfirmed.size:
                moving |= np.isin(frame.instance_id, unconfirmed)

        # A point on the ground under a moving car is ground, not car.
        if frame.height_above_gnd is not None:
            moving &= frame.height_above_gnd > 0.15
        return moving
