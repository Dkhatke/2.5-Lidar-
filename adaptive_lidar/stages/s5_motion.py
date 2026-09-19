"""
S5 — Motion segmentation, instance extraction and tracking.

Three parts:

1. **Range-image residual MOS** (``perception/motion.py``).  4DMOS needs CUDA;
   residual-image MOS is the CPU-feasible member of the same published family
   (the LMNet / "Moving Object Segmentation in 3D LiDAR Data" lineage) and it
   reuses the range image S1 already built.

2. **Instance extraction** by 26-connected components on a 0.4 m voxel grid,
   gated on shared dominant semantic class so a pedestrian does not merge into
   the wall behind them.  No DBSCAN: it needs the kd-tree the brief bans from
   the per-frame path.
   The previous implementation ran one full fancy-index over every non-ground
   point *per connected component* — O(L*N).  Here the fancy-index happens ONCE
   to get a per-point label array, then everything is a segment reduction.

3. **Tracking** by Hungarian association + constant-velocity Kalman filter,
   producing the object table.  Velocity lives in that table, never in cells.

Ends by validating the frame data contract.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from adaptive_lidar.perception.motion import residual_mos
from adaptive_lidar.perception.tracking import Tracker
from adaptive_lidar.pipeline.timing import stage_timer
from adaptive_lidar.pipeline.types import (
    CLASS_IMPORTANCE,
    NUM_CLASSES,
    Frame,
    Instance,
    PipelineContext,
    validate_frame_contract,
)
from adaptive_lidar.utils.grouping import (
    group_by_key,
    group_sizes,
    pack_keys_3d,
    scatter_to_points,
    segment_bincount,
    segment_reduce,
)

#: Classes whose instances are allowed to be moving. A building never moves,
#: so a residual there is pose error, not motion.
MOVABLE_CLASSES = (3, 4)


class S5Motion:
    def __init__(self, config: Dict[str, Any]):
        self.cfg = config
        mcfg = config.get("motion", {})
        self.voxel = float(mcfg.get("instance_voxel", 0.4))
        self.min_voxels = int(mcfg.get("min_instance_voxels", 3))
        self.tracker = Tracker(config)

    # ── Instance extraction ───────────────────────────────────
    def _extract_instances(self, frame: Frame) -> np.ndarray:
        """(N,) instance id per point, -1 for ground / unclustered.

        Class-gated 26-connectivity on the occupied-voxel set.
        """
        from scipy.ndimage import label as nd_label

        pts = frame.points
        n = len(pts)
        inst = np.full(n, -1, dtype=np.int32)
        ng = frame.non_ground_mask
        if ng is None or n == 0:
            return inst

        sel = np.flatnonzero(ng)
        if sel.size < 5:
            return inst

        inv = 1.0 / self.voxel
        vi = np.floor(pts[sel] * inv).astype(np.int32)
        vmin = vi.min(axis=0)
        vi -= vmin
        shape = tuple((vi.max(axis=0) + 1).tolist())
        if int(np.prod(shape)) > 40_000_000:      # pathological extent guard
            return inst

        # Per-voxel dominant semantic class, so connectivity can be gated.
        vkeys = pack_keys_3d(vi[:, 0], vi[:, 1], vi[:, 2])
        uniq, starts, order = group_by_key(vkeys)
        cls = np.clip(frame.sem_class[sel], 0, NUM_CLASSES - 1)
        hist = segment_bincount(cls[order], starts, NUM_CLASSES)
        vox_cls = np.argmax(hist, axis=1).astype(np.int8)

        # Two occupancy volumes: "movable" classes and everything else.
        # Labelling them separately is the gate — a VRU voxel and a wall voxel
        # are never 26-adjacent in the same volume, so they cannot merge.
        occ = np.zeros(shape, dtype=np.uint8)
        vux, vuy, vuz = _unpack_shifted(uniq)
        movable = np.isin(vox_cls, MOVABLE_CLASSES)
        occ[vux, vuy, vuz] = np.where(movable, 2, 1).astype(np.uint8)

        structure = np.ones((3, 3, 3), dtype=bool)   # 26-connectivity
        lab_a, n_a = nd_label(occ == 1, structure=structure)
        lab_b, n_b = nd_label(occ == 2, structure=structure)
        # Offset the second set so the two label spaces do not collide.
        lab = np.where(lab_b > 0, lab_b + n_a, lab_a)

        # ONE fancy-index to get the per-voxel label, then scatter to points.
        vox_label = lab[vux, vuy, vuz].astype(np.int32)
        # Drop components smaller than min_voxels.
        counts = np.bincount(vox_label, minlength=n_a + n_b + 1)
        keep = counts >= self.min_voxels
        keep[0] = False
        vox_label = np.where(keep[vox_label], vox_label, 0)

        # Renumber survivors to a dense 0..K-1.
        present = np.flatnonzero(np.bincount(vox_label, minlength=n_a + n_b + 1))
        remap = np.full(n_a + n_b + 1, -1, dtype=np.int32)
        remap[present] = np.arange(len(present), dtype=np.int32)
        remap[0] = -1
        vox_label = remap[vox_label]

        per_point = scatter_to_points(vox_label, starts, order)
        inst[sel] = per_point
        return inst

    def _instances_from_labels(self, frame: Frame, inst: np.ndarray) -> List[Instance]:
        """Build the object table from the per-point instance ids."""
        sel = np.flatnonzero(inst >= 0)
        if sel.size == 0:
            return []
        uniq, starts, order = group_by_key(inst[sel].astype(np.int64))
        pidx = sel[order]
        pts = frame.points[pidx]

        cmin = segment_reduce(pts, starts, "min")
        cmax = segment_reduce(pts, starts, "max")
        cent = segment_reduce(pts, starts, "mean")
        counts = group_sizes(starts)
        cls = np.clip(frame.sem_class[pidx], 0, NUM_CLASSES - 1)
        hist = segment_bincount(cls, starts, NUM_CLASSES)
        # Weight the class vote by safety stake so a mostly-background cluster
        # containing clear VRU evidence is still reported as a VRU.
        dom = np.argmax(hist.astype(np.float32) * CLASS_IMPORTANCE[None, :], axis=1)
        mprob = segment_reduce(frame.moving_prob[pidx], starts, "mean")

        out: List[Instance] = []
        for j in range(len(uniq)):
            out.append(Instance(
                instance_id=int(uniq[j]),
                cluster_id=int(uniq[j]),
                centroid=cent[j].astype(np.float32),
                bbox_min=cmin[j].astype(np.float32),
                bbox_max=cmax[j].astype(np.float32),
                point_count=int(counts[j]),
                semantic_class=int(dom[j]),
                motion_probability=float(mprob[j]),
            ))
        return out

    # ── Main ──────────────────────────────────────────────────
    def process(self, frame: Frame, ctx: PipelineContext):
        with stage_timer("S5", frame.timing):
            n = len(frame.points)
            if n == 0:
                frame.instances = []
                validate_frame_contract(frame)
                return

            # 1. motion segmentation
            frame.moving_prob = residual_mos(frame, ctx)

            # 2. instances
            inst = self._extract_instances(frame)
            frame.instance_id = inst
            frame.instance_ids = inst          # legacy alias
            instances = self._instances_from_labels(frame, inst)

            # 3. tracking — Hungarian + constant-velocity Kalman
            instances = self.tracker.update(instances, frame.timestamp, frame.pose)
            frame.instances = instances

            # Re-key the per-point ids from cluster labels to TRACK ids, so
            # that anything downstream holding a track id (the MOS gate, the
            # dynamic overlay, the cell handle) can look points up directly.
            if instances:
                lut = np.full(int(inst.max()) + 2, -1, np.int32)
                for i in instances:
                    if i.cluster_id >= 0:
                        lut[i.cluster_id] = i.instance_id
                has = inst >= 0
                inst = np.where(has, lut[np.clip(inst, 0, None)], -1)
                frame.instance_id = inst
                frame.instance_ids = inst

            # Push the track-level motion decision back onto the points, so the
            # MOS gate in S8 acts on whole objects rather than on noisy
            # per-point residuals.
            self._apply_track_motion(frame, instances)
            self._rollup_tiles(frame, instances)

            frame.timing["S5_diag"] = {
                "n_instances": len(instances),
                "n_moving": sum(1 for i in instances if i.is_dynamic),
                "mean_moving_prob": float(frame.moving_prob.mean()),
            }

            validate_frame_contract(frame)

    # ────────────────────────────────────────────────────────
    @staticmethod
    def _apply_track_motion(frame: Frame, instances: List[Instance]):
        if not instances:
            return
        ids = np.array([i.instance_id for i in instances], dtype=np.int64)
        prob = np.array([i.motion_probability for i in instances], dtype=np.float32)
        lut = np.zeros(int(ids.max()) + 2, dtype=np.float32)
        lut[ids] = prob
        has = frame.instance_id >= 0
        frame.moving_prob = np.where(
            has, np.maximum(frame.moving_prob, lut[np.clip(frame.instance_id, 0, None)]),
            frame.moving_prob)

    def _rollup_tiles(self, frame: Frame, instances: List[Instance]):
        tiles = frame.tiles
        mi = getattr(frame, "morton", None)
        if not tiles or mi is None or mi.n_tiles == 0:
            return
        mmax = segment_reduce(
            frame.moving_prob[mi.order], mi.tile_starts, "max")
        for j in range(mi.n_tiles):
            t = tiles[j]
            t.max_moving_prob = float(mmax[j])
            t.motion_probability = float(mmax[j])
            t.score_dynamic = float(mmax[j])


def _unpack_shifted(keys: np.ndarray):
    from adaptive_lidar.utils.grouping import unpack_keys_3d
    return unpack_keys_3d(keys)
