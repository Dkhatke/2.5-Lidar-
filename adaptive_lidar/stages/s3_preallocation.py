"""
S3 — Pre-Allocation.

A cheap, geometry-only pass of the allocation controller, run BEFORE semantic
perception so that S4 knows roughly where the interesting parts of the scene
are.  S6 re-runs the same controller with the full value function once
semantics and motion are available.

S3 : geometry only  (G term)
S6 : geometry + semantics + uncertainty + motion + safety pins  (closed loop)

The scoring itself lives in ``mapping/allocation.py``.  This stage only wires
the controller to the frame, so the policy string is the single thing that
changes between the ablation's ten configurations.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from adaptive_lidar.mapping.allocation import AllocationController
from adaptive_lidar.pipeline.timing import stage_timer
from adaptive_lidar.pipeline.types import Frame, PipelineContext


class S3Preallocation:
    def __init__(self, config: Dict[str, Any]):
        self.cfg = config
        self.ctrl = AllocationController(config)

    def process(self, frame: Frame, ctx: PipelineContext):
        with stage_timer("S3", frame.timing):
            tiles = frame.tiles
            if not tiles:
                return
            # Geometry-only preview. Semantic and motion terms are still zero
            # at this point, so asking for them would just add noise.
            policy = ctx.allocation_policy
            preview = policy if policy.startswith("uniform_") else "distance_geometry"
            levels = self.ctrl.allocate(tiles, budget=ctx.budget, policy=preview)
            for t, l in zip(tiles, levels):
                t.resolution_level = int(l)
                t.selected = bool(l <= 2)
            frame.timing["S3_diag"] = dict(self.ctrl.last_diag)
