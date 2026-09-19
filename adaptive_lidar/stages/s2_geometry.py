"""
S2 — Cheap Signal Pass: ground separation + tile allocation features.

No neural network runs here.  Everything is geometric and fully vectorised.
The original implementation looped over 1600 tiles running a full four-term
boolean mask over all N points each time — O(1600*N), 862 ms at 120k points.

This version does not even sort.  S1 has already ordered the cloud once by
level-0 Morton code (``utils/spatial_index.py``), and because the tile is a
node of the same power-of-two hierarchy, the tile grouping is a run scan over
that existing order.  Every per-tile statistic is then one ``reduceat``.

A tile carries ALLOCATION features only.  It deliberately carries no semantic
content: a pedestrian on a road shares a tile with that road, and one
tile-level class would erase one of them.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from adaptive_lidar.perception.ground import GroundSegmenter
from adaptive_lidar.pipeline.timing import stage_timer
from adaptive_lidar.pipeline.types import Frame, PipelineContext, Tile
from adaptive_lidar.utils.grouping import (
    group_sizes,
    segment_bincount,
    segment_percentile_sorted,
    segment_reduce,
    segment_sort,
)
from adaptive_lidar.utils.range_image import isolated_return_mask
from adaptive_lidar.utils.spatial_index import TILE_SIZE


class S2Geometry:
    def __init__(self, config: Dict[str, Any]):
        self.cfg = config
        self.tsize = TILE_SIZE
        self.ground = GroundSegmenter(config)

    def process(self, frame: Frame, ctx: PipelineContext):
        with stage_timer("S2", frame.timing):
            pts = frame.points
            n = len(pts)

            # -- Ground separation + smooth height field --------
            gmask, gz, hag = self.ground(
                pts,
                range_image=frame.range_image,
                range_image_valid=frame.range_image_valid,
                elev_angles_deg=getattr(frame, "_elev_angles_deg", None),
                index=getattr(frame, "morton", None),
            )
            # Isolated-return filter: low intensity AND few range-image
            # neighbours at a similar range, i.e. dust or rain.  Deliberately
            # NOT a generic statistical outlier filter, which deletes poles -
            # poles are sparse and isolated by construction.
            # The map's z_max is a true maximum (so that coarsening stays
            # exact), which makes removing these points here rather than
            # papering over them with a per-cell percentile the right place.
            ri = getattr(frame, "_range_image", None)
            if ri is not None and n:
                noise = isolated_return_mask(ri, frame.intensity)
            else:
                noise = np.zeros(n, bool)
            frame.noise_mask = noise

            # A dust return is neither ground NOR an object: excluding it from
            # BOTH masks is the point. Clearing it only from the ground mask
            # promoted every speck to non-ground, which inflated the
            # connected-component volume and tripled the cost of instance
            # extraction while inventing instances out of rain.
            frame.ground_mask = gmask & ~noise
            frame.non_ground_mask = ~gmask & ~noise
            frame.ground_z = gz
            frame.height_above_gnd = hag
            frame.timing["S2_ground_method"] = self.ground.active

            if n == 0:
                frame.tiles = []
                frame.tile_of_point = np.zeros(0, dtype=np.int32)
                return

            mi = frame.morton
            order = mi.order
            starts = mi.tile_starts
            m = mi.n_tiles
            frame.tile_of_point = mi.tile_of_point

            # -- Per-tile reductions, all on the existing order -
            z = pts[:, 2].astype(np.float32)
            r = frame._range
            inten = frame.intensity.astype(np.float32)

            zs = z[order]
            counts = group_sizes(starts).astype(np.int32)
            z_mean = segment_reduce(zs, starts, "mean")
            z_var = segment_reduce(zs, starts, "var")
            r_mean = segment_reduce(r[order], starts, "mean")
            i_mean = segment_reduce(inten[order], starts, "mean")

            hs = hag.astype(np.float32)[order]
            # Non-ground vertical spread: ground points sit near zero, so the
            # max height above ground IS the spread that matters.
            z_spread = np.maximum(segment_reduce(hs, starts, "max"), 0.0)

            # Largest empty vertical band in the tile - a bridge, a canopy, or
            # a torso above legs. One sort serves both order statistics.
            hp = segment_percentile_sorted(
                segment_sort(hs, starts), starts, [60, 90])
            hist_gap = np.maximum(hp[:, 1] - hp[:, 0], 0.0)

            # Roughness: spread of the GROUND points only, so a wall face does
            # not read as rough terrain.
            gsel = gmask[order].astype(np.float32)
            gz_only = np.where(gsel > 0, zs, 0.0)
            g_cnt = np.maximum(segment_reduce(gsel, starts, "sum"), 1.0)
            g_mean = segment_reduce(gz_only, starts, "sum") / g_cnt
            g_sq = segment_reduce(gz_only * gz_only, starts, "sum") / g_cnt
            roughness = np.sqrt(np.maximum(g_sq - g_mean * g_mean, 0.0))

            # Boundary score: quadrant occupancy imbalance inside the tile.
            tix, tiy = mi.tile_indices()
            gid = np.repeat(np.arange(m, dtype=np.int64), counts.astype(np.int64))
            local_x = pts[order, 0] - tix[gid].astype(np.float32) * self.tsize
            local_y = pts[order, 1] - tiy[gid].astype(np.float32) * self.tsize
            half = self.tsize / 2.0
            quad = ((local_x > half).astype(np.int64) * 2
                    + (local_y > half).astype(np.int64))
            qhist = segment_bincount(quad, starts, 4).astype(np.float32)
            boundary = np.clip(
                qhist.std(axis=1) / (qhist.mean(axis=1) + 1e-6), 0.0, 1.0)

            verticality = np.clip(z_spread / 3.0, 0.0, 1.0)
            density = counts / (self.tsize * self.tsize)
            cx, cy = mi.tile_centres()

            # -- Materialise Tile objects -----------------------
            # One Python object per OCCUPIED tile (typically 400-3000), never
            # per point. Point indices are a slice of the shared order.
            tiles: List[Tile] = []
            for j in range(m):
                t = Tile(tile_id=j, ix=int(tix[j]), iy=int(tiy[j]),
                         cx=float(cx[j]), cy=float(cy[j]))
                t.point_count = int(counts[j])
                t.density = float(density[j])
                t.height_variance = float(z_var[j])
                t.height_mean = float(z_mean[j])
                t.range_mean = float(r_mean[j])
                t.intensity_mean = float(i_mean[j])
                t.verticality = float(verticality[j])
                t.roughness = float(roughness[j])
                t.boundary_score = float(boundary[j])
                t.z_spread = float(z_spread[j])
                t.histogram_gap = float(hist_gap[j])
                tiles.append(t)

            frame.tiles = tiles
            frame._tile_order = order
            frame._tile_starts = starts

            frame.timing["S2_diag"] = {
                "n_tiles": m,
                "tile_size_m": self.tsize,
                "ground_fraction": float(gmask.mean()),
                "ground_method": self.ground.active,
            }


def tile_point_indices(frame: Frame, tile_id: int) -> np.ndarray:
    """Point indices of one tile - a slice of the shared order, not a copy."""
    s = frame._tile_starts
    return frame._tile_order[s[tile_id]:s[tile_id + 1]]
