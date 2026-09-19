"""
S5 — Motion + Instance Extraction.

PROTOTYPE NOTE:
  Instance extraction: connected components on non-ground voxels (scipy).
  Motion estimation: centroid displacement / Δt (labelled clearly).
  Future replacement: 4DMOS (sparse 4D temporal convolution).
  Tracking: nearest-centroid association.
  Future replacement: Hungarian assignment + Kalman filter.
"""
from __future__ import annotations
import numpy as np
from typing import Dict, Any, List, Optional

from adaptive_lidar.pipeline.types import Frame, PipelineContext, Instance
from adaptive_lidar.pipeline.timing import stage_timer


class S5Motion:
    def __init__(self, config: Dict[str, Any]):
        self.cfg = config

    def _extract_instances(
        self, points: np.ndarray, non_ground_mask: np.ndarray
    ) -> List[Instance]:
        """
        Connected components on non-ground points.
        PROTOTYPE: uses scipy label on voxelised grid.
        """
        from scipy.ndimage import label as scipy_label

        ng_pts = points[non_ground_mask]
        if len(ng_pts) < 5:
            return []

        # Voxelise non-ground points at 0.4 m
        vsize = 0.4
        inv = 1.0 / vsize
        keys = np.floor(ng_pts * inv).astype(np.int32)
        kmin = keys.min(axis=0)
        keys_off = keys - kmin

        grid_shape = tuple(keys_off.max(axis=0) + 1)
        occupancy = np.zeros(grid_shape, dtype=np.uint8)
        occupancy[keys_off[:, 0], keys_off[:, 1], keys_off[:, 2]] = 1

        labeled, n_labels = scipy_label(occupancy)

        instances = []
        ng_indices = np.where(non_ground_mask)[0]

        for lbl in range(1, n_labels + 1):
            voxel_mask = labeled[keys_off[:, 0], keys_off[:, 1], keys_off[:, 2]] == lbl
            if voxel_mask.sum() < 3:
                continue
            inst_pts = ng_pts[voxel_mask]
            orig_indices = ng_indices[voxel_mask]

            centroid = inst_pts.mean(axis=0)
            bbox_min = inst_pts.min(axis=0)
            bbox_max = inst_pts.max(axis=0)
            height = bbox_max[2] - bbox_min[2]
            footprint = max(bbox_max[0] - bbox_min[0], bbox_max[1] - bbox_min[1])

            # Simple semantic class heuristic
            if height > 0.3 and footprint > 2.0:
                sem_class = 3  # vehicle
            elif 0.5 < height < 2.2 and footprint < 1.5:
                sem_class = 4  # vru
            elif height > 1.5:
                sem_class = 2  # static_obstacle
            else:
                sem_class = 2  # static_obstacle default

            inst = Instance(
                instance_id=lbl,
                centroid=centroid,
                bbox_min=bbox_min,
                bbox_max=bbox_max,
                point_count=int(voxel_mask.sum()),
                semantic_class=sem_class,
            )
            instances.append(inst)

        return instances

    def _match_and_estimate_motion(
        self,
        current: List[Instance],
        previous: List[Instance],
        dt: float,
    ) -> List[Instance]:
        """
        Nearest-centroid association.
        PROTOTYPE NOTE: Future replacement: Hungarian assignment + Kalman filter.
        """
        if not previous or dt <= 0:
            return current

        prev_centroids = np.array([i.centroid for i in previous])
        motion_thresh = self.cfg.get("thresholds", {}).get(
            "motion", {}).get("displacement", 0.3)

        for inst in current:
            dists = np.linalg.norm(prev_centroids - inst.centroid, axis=1)
            nearest_i = int(np.argmin(dists))
            nearest_d = float(dists[nearest_i])

            prev_inst = previous[nearest_i]
            displacement = inst.centroid - prev_inst.centroid
            velocity = displacement / dt
            inst.velocity = velocity
            speed = float(np.linalg.norm(velocity))

            if nearest_d < 5.0:  # within 5 m — plausible match
                inst.frames_seen = prev_inst.frames_seen + 1
                if speed > motion_thresh:
                    inst.motion_probability = min(
                        0.5 + (speed - motion_thresh) * 0.5, 1.0)
                    inst.is_dynamic = True
                else:
                    inst.motion_probability = max(
                        0.0, speed / motion_thresh * 0.4)
                    inst.is_dynamic = False
            else:
                inst.motion_probability = 0.0
                inst.velocity = np.zeros(3)

        return current

    def _update_tile_motion(self, frame: Frame, instances: List[Instance]):
        """Propagate motion probability back to tiles for S6."""
        if not frame.tiles:
            return
        for tile in frame.tiles:
            # Check if any dynamic instance overlaps this tile
            max_motion = 0.0
            tsize = self.cfg.get("tiles", {}).get("size", 2.0)
            tx_lo, tx_hi = tile.cx - tsize/2, tile.cx + tsize/2
            ty_lo, ty_hi = tile.cy - tsize/2, tile.cy + tsize/2
            for inst in instances:
                cx, cy = inst.centroid[0], inst.centroid[1]
                if tx_lo <= cx < tx_hi and ty_lo <= cy < ty_hi:
                    max_motion = max(max_motion, inst.motion_probability)
            tile.motion_probability = max_motion
            tile.score_dynamic = max_motion

    def process(self, frame: Frame, ctx: PipelineContext):
        with stage_timer("S5", frame.timing):
            non_ground = frame.non_ground_mask
            if non_ground is None:
                non_ground = np.ones(len(frame.points), dtype=bool)

            instances = self._extract_instances(frame.points, non_ground)

            # Match against previous frame
            prev_frame = ctx.prev_frame()
            dt = 0.1  # assume ~10 Hz if no pose info
            if prev_frame is not None:
                dt = max(frame.timestamp - prev_frame.timestamp, 0.05)

            prev_instances = []
            if ctx.instance_history:
                prev_instances = list(ctx.instance_history[-1].values())

            instances = self._match_and_estimate_motion(instances, prev_instances, dt)

            # Store in context history
            inst_dict = {inst.instance_id: inst for inst in instances}
            ctx.instance_history.append(inst_dict)
            if len(ctx.instance_history) > ctx.config.get(
                    "pipeline", {}).get("max_frames_history", 5):
                ctx.instance_history.pop(0)

            # Back-propagate motion to tiles
            self._update_tile_motion(frame, instances)
