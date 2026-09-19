"""
S1 — Build Reusable Spatial Indices.

Creates TWO representations, both reused by every later stage:

  A. Range image  (ring x azimuth), holding POINT INDICES — see
     utils/range_image.py.  Source of ground segmentation, vertical runs,
     local window statistics, incidence angles, free-space carving and the
     motion residual.  Built once, read everywhere.

  B. Voxel hash   (CSR: unique_keys / offsets / sorted_idx) — neighbour counts
     and instance clustering.

IMPORTANT ARCHITECTURAL NOTE:
  This voxel hash is a CURRENT-FRAME QUERY INDEX.
  It is NOT the persistent map (that lives in S7/mapping/).
  Do not merge these concepts.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from adaptive_lidar.pipeline.timing import stage_timer
from adaptive_lidar.pipeline.types import Frame, PipelineContext
from adaptive_lidar.utils.range_image import build_range_image, default_elevations
from adaptive_lidar.utils.spatial_index import build_morton_index
from adaptive_lidar.utils.voxel_hash import build_voxel_hash


class S1Indices:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        s = config.get("sensor", {})
        self.num_rings = int(s.get("num_rings", 64))
        self.num_azimuth = int(s.get("num_azimuth", 2048))
        self.elev = default_elevations(
            self.num_rings,
            float(s.get("elev_min_deg", -24.0)),
            float(s.get("elev_max_deg", 2.0)),
        )
        self.ri_method = config.get("perception", {}).get("range_image_method", "auto")
        self.vsize = float(config.get("voxel", {}).get("query_size", 0.5))

    def process(self, frame: Frame, ctx: PipelineContext):
        with stage_timer("S1", frame.timing):
            ri = build_range_image(
                frame.points,
                ring=frame.ring,
                azimuth_bin=frame.azimuth_bin,
                num_rings=self.num_rings,
                num_azimuth=self.num_azimuth,
                elev_angles_deg=self.elev,
                method=self.ri_method,
            )
            frame.range_image = ri.index
            frame.range_image_valid = ri.valid
            frame.range_image_range = ri.rng
            frame._range_image = ri                   # full object for S2/S3/S4
            frame._elev_angles_deg = ri.elev_angles_deg

            # If the cloud had no native ring/azimuth, adopt what the image
            # reconstructed so the rest of the pipeline sees a filled contract.
            n = len(frame.points)
            if frame.ring is None or not np.any(np.asarray(frame.ring) >= 0):
                rows = np.full(n, -1, dtype=np.int16)
                cols = np.full(n, -1, dtype=np.int32)
                rr, cc = np.nonzero(ri.valid)
                idx = ri.index[rr, cc]
                rows[idx] = rr.astype(np.int16)
                cols[idx] = cc.astype(np.int32)
                frame.ring = rows
                frame.azimuth_bin = cols

            # The frame's single spatial ordering. One sort here replaces the
            # four that tile grouping, the ground height field, the allocation
            # cost table and the map insert used to each do separately.
            frame.morton = build_morton_index(frame.points)
            frame._range = np.sqrt(
                (frame.points.astype(np.float32) ** 2).sum(axis=1))

            frame.voxel_hash = build_voxel_hash(frame.points, self.vsize)

            frame.timing["S1_diag"] = {
                "ri_method": ri.method,
                "ri_shape": list(ri.shape),
                "ri_valid_rate": round(ri.valid_rate, 4),
                "ri_collisions": ri.collisions,
                "ri_collision_rate": round(ri.collisions / max(n, 1), 4),
                "n_voxels": frame.voxel_hash.n_voxels,
                "n_tiles": frame.morton.n_tiles,
            }
