"""
Pipeline orchestrator — runs S0 through S9 in order.
Each stage receives and mutates the Frame + PipelineContext.

STAGE ORDER
-----------
    S0 ingest -> S1 indices -> S2 geometry -> S3 pre-allocation
       -> S4 semantics -> S5 motion+tracking -> S6 final allocation
       -> S8 MOS gate -> S7 map update -> S9 telemetry

S8 runs before S7.  The gate decides which points are allowed into the
persistent map, and a gate that runs after the thing it gates is not a gate —
that ordering is what used to let moving objects leave trails.  The stage
decomposition itself is unchanged.
"""
from __future__ import annotations

import time
from typing import Any, Dict, Optional

from adaptive_lidar.pipeline.timing import stage_timer
from adaptive_lidar.pipeline.types import Frame, PipelineContext


class Pipeline:
    """
    Usage:
        pipe = Pipeline(config)
        pipe.build_stages(backend="auto")
        frame = pipe.run(cloud, frame_id=0)
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.context = PipelineContext(config=config)
        self.context.budget = float(config.get("budget", {}).get("default", 0.8))
        self.context.allocation_policy = config.get(
            "allocation", {}).get("policy", "full")
        self._stages_built = False

    # ────────────────────────────────────────────────────────
    def build_stages(self, backend: str = "auto", policy: Optional[str] = None):
        from adaptive_lidar.stages.s0_ingest import ingest
        from adaptive_lidar.stages.s1_indices import S1Indices
        from adaptive_lidar.stages.s2_geometry import S2Geometry
        from adaptive_lidar.stages.s3_preallocation import S3Preallocation
        from adaptive_lidar.stages.s4_semantics import S4Semantics
        from adaptive_lidar.stages.s5_motion import S5Motion
        from adaptive_lidar.stages.s6_allocation import S6Allocation
        from adaptive_lidar.stages.s7_map import S7Map
        from adaptive_lidar.stages.s8_fusion import S8Fusion
        from adaptive_lidar.stages.s9_output import S9Output

        if policy is not None:
            self.context.allocation_policy = policy

        self._ingest = ingest
        self._s1 = S1Indices(self.config)
        self._s2 = S2Geometry(self.config)
        self._s3 = S3Preallocation(self.config)
        self._s4 = S4Semantics(self.config, backend=backend)
        self._s5 = S5Motion(self.config)
        self._s6 = S6Allocation(self.config)
        self._s7 = S7Map(self.config, self.context)
        self._s8 = S8Fusion(self.config, self.context)
        self._s9 = S9Output(self.config)

        self.context.semantic_backend_name = self._s4.backend_name
        self._stages_built = True

    # ────────────────────────────────────────────────────────
    def run(
        self,
        cloud_np=None,
        path: Optional[str] = None,
        frame_id: Optional[int] = None,
        timestamp: Optional[float] = None,
    ) -> Frame:
        if not self._stages_built:
            self.build_stages()

        if frame_id is None:
            frame_id = self.context.frame_count
        if timestamp is None:
            timestamp = time.time()

        frame = self._ingest(path=path, frame_id=frame_id, timestamp=timestamp,
                             config=self.config, synthetic_cloud=cloud_np)

        self._s1.process(frame, self.context)          # indices
        self._s2.process(frame, self.context)          # ground + tiles
        self._s3.process(frame, self.context)          # geometry pre-allocation
        self._s4.process(frame, self.context)          # per-point semantics
        self._s5.process(frame, self.context)          # motion + tracking
        self._s6.process(frame, self.context)          # final allocation
        self._s8.process(frame, self.context)          # MOS gate  (before S7)
        self._s7.process(frame, self.context)          # map update
        self._s9.process(frame, self.context)          # telemetry

        self.context.add_frame(frame)
        for stage, ms in frame.timing.items():
            if isinstance(ms, float):
                self.context.record_timing(stage, ms)
        return frame

    # ────────────────────────────────────────────────────────
    def set_budget(self, budget: float):
        self.context.budget = max(0.0, min(1.0, budget))

    def set_policy(self, policy: str):
        self.context.allocation_policy = policy

    @property
    def amap(self):
        return self.context.amap

    @property
    def map_cells(self):
        return {} if self.context.amap is None else self.context.amap.cell_counts()

    @property
    def frame_count(self):
        return self.context.frame_count
