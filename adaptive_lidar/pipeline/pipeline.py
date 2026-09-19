"""
Pipeline orchestrator — runs S0 through S9 in order.
Each stage receives and mutates the Frame + PipelineContext.
"""
from __future__ import annotations
import time
from typing import Optional, Dict, Any

from adaptive_lidar.pipeline.types import Frame, PipelineContext
from adaptive_lidar.pipeline.timing import stage_timer


class Pipeline:
    """
    Thin wrapper that calls stages in order and records timing.

    Usage:
        pipe = Pipeline(config)
        pipe.build_stages(backend="auto")
        frame = pipe.run(cloud_np, frame_id=0)
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.context = PipelineContext(config=config)
        self._stages_built = False

    # ────────────────────────────────────────────────────────
    def build_stages(self, backend: str = "auto"):
        """Lazy-import and instantiate all stages."""
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

        # Expose backend name for UI
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
        """
        Run the full pipeline on one point cloud.

        cloud_np : numpy (N,4) array [x,y,z,intensity]
        path     : alternatively a file path (S0 will load it)
        """
        if not self._stages_built:
            self.build_stages()

        if frame_id is None:
            frame_id = self.context.frame_count
        if timestamp is None:
            timestamp = time.time()

        # S0 — ingest
        frame = self._ingest(
            path=path,
            frame_id=frame_id,
            timestamp=timestamp,
            config=self.config,
            synthetic_cloud=cloud_np,
        )

        # S1 — build indices
        self._s1.process(frame, self.context)

        # S2 — cheap geometry + tile features
        self._s2.process(frame, self.context)

        # S3 — pre-allocation / information scoring
        self._s3.process(frame, self.context)

        # S4 — sparse semantic perception (selected tiles only)
        self._s4.process(frame, self.context)

        # S5 — motion + instance extraction
        self._s5.process(frame, self.context)

        # S6 — final allocation (informed by S4 uncertainty + S5 motion)
        self._s6.process(frame, self.context)

        # S7 — update adaptive 2.5D map
        self._s7.process(frame, self.context)

        # S8 — temporal fusion
        self._s8.process(frame, self.context)

        # S9 — collate output telemetry
        self._s9.process(frame, self.context)

        # Record to history
        self.context.add_frame(frame)

        # Accumulate timing stats
        for stage, ms in frame.timing.items():
            self.context.record_timing(stage, ms)

        return frame

    # ────────────────────────────────────────────────────────
    def set_budget(self, budget: float):
        """Update computation budget (0–1) at runtime."""
        self.context.budget = max(0.05, min(1.0, budget))

    @property
    def map_cells(self):
        return self.context.map_cells

    @property
    def frame_count(self):
        return self.context.frame_count
