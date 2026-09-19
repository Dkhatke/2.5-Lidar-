"""
S1 — Build Reusable Spatial Indices.

Creates TWO representations:
  A. Range image  (2-D azimuth × ring projection)
  B. Voxel hash   (voxel_key → list of point indices)

IMPORTANT ARCHITECTURAL NOTE:
  This voxel hash is a CURRENT-FRAME QUERY INDEX.
  It is NOT the persistent map (that lives in S7/mapping/).
  Do not merge these concepts.
"""
from __future__ import annotations
from typing import Dict, Any

from adaptive_lidar.pipeline.types import Frame, PipelineContext
from adaptive_lidar.pipeline.timing import stage_timer
from adaptive_lidar.utils.range_image import build_range_image
from adaptive_lidar.utils.voxel_hash import build_voxel_hash


class S1Indices:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def process(self, frame: Frame, ctx: PipelineContext):
        with stage_timer("S1", frame.timing):
            sensor_cfg = self.config.get("sensor", {})
            voxel_cfg = self.config.get("voxel", {})

            # A — range image
            ri, ri_xyz = build_range_image(
                frame.points,
                num_rings=sensor_cfg.get("num_rings", 64),
                h_res_deg=sensor_cfg.get("horizontal_resolution", 0.2),
                v_fov_deg=sensor_cfg.get("vertical_fov", 30.0),
            )
            frame.range_image = ri
            frame.range_image_xyz = ri_xyz

            # B — voxel hash (query index)
            vsize = voxel_cfg.get("query_size", 0.25)
            frame.voxel_hash = build_voxel_hash(frame.points, vsize)
