"""
S2 — Cheap Signal Pass: Ground Separation + Tile Feature Extraction.

PROTOTYPE NOTE:
  Ground separation uses a fast height-grid method.
  Future replacement: Patchwork++ (labelled clearly in UI).

No neural network runs here. Everything is geometric.
"""
from __future__ import annotations
import numpy as np
from typing import Dict, Any, List

from adaptive_lidar.pipeline.types import Frame, PipelineContext, Tile, ResolutionLevel
from adaptive_lidar.pipeline.timing import stage_timer


class S2Geometry:
    def __init__(self, config: Dict[str, Any]):
        self.cfg = config

    # ── Ground separation ─────────────────────────────────────
    def _ground_separate(self, points: np.ndarray) -> np.ndarray:
        """
        PROTOTYPE: height-grid ground segmentation.
        Future replacement: Patchwork++
        Returns boolean mask (True = ground).
        """
        gt_cfg = self.cfg.get("thresholds", {}).get("ground", {})
        h_thresh = gt_cfg.get("height_threshold", 0.3)
        grid_res = gt_cfg.get("grid_resolution", 1.0)
        sensor_h = self.cfg.get("sensor", {}).get("height", 1.8)

        x, y, z = points[:, 0], points[:, 1], points[:, 2]

        # Build 2D height grid — minimum z per cell
        gx = np.floor(x / grid_res).astype(np.int32)
        gy = np.floor(y / grid_res).astype(np.int32)

        # Offset to make indices non-negative
        gx_off = gx - gx.min()
        gy_off = gy - gy.min()

        gW = gx_off.max() + 1
        gH = gy_off.max() + 1

        min_z_grid = np.full((gH, gW), np.inf, dtype=np.float32)
        np.minimum.at(min_z_grid, (gy_off, gx_off), z)

        # Replace inf with 0
        min_z_grid[min_z_grid == np.inf] = 0.0

        # Per-point: local minimum z from its grid cell
        local_min_z = min_z_grid[gy_off, gx_off]

        # Ground: close to local minimum and below sensor height
        ground_mask = (
            (z - local_min_z < h_thresh) &
            (z < sensor_h * 0.5)
        )
        return ground_mask.astype(bool)

    # ── Tile grid ─────────────────────────────────────────────
    def _build_tiles(self, points: np.ndarray) -> List[Tile]:
        tile_cfg = self.cfg.get("tiles", {})
        tsize = tile_cfg.get("size", 2.0)
        xr = tile_cfg.get("x_range", [-40, 40])
        yr = tile_cfg.get("y_range", [-40, 40])

        xs = np.arange(xr[0], xr[1], tsize)
        ys = np.arange(yr[0], yr[1], tsize)

        tiles: List[Tile] = []
        tid = 0
        for ixi, tx in enumerate(xs):
            for iyi, ty in enumerate(ys):
                cx = tx + tsize / 2
                cy = ty + tsize / 2
                # Points inside this tile
                in_tile = (
                    (points[:, 0] >= tx) & (points[:, 0] < tx + tsize) &
                    (points[:, 1] >= ty) & (points[:, 1] < ty + tsize)
                )
                idx = np.where(in_tile)[0].astype(np.int32)
                t = Tile(tile_id=tid, ix=ixi, iy=iyi, cx=cx, cy=cy,
                         point_indices=idx)
                tiles.append(t)
                tid += 1
        return tiles

    # ── Per-tile geometry features ────────────────────────────
    def _compute_tile_features(
        self,
        tile: Tile,
        points: np.ndarray,
        intensity: np.ndarray,
        ground_mask: np.ndarray,
        tsize: float,
    ):
        idx = tile.point_indices
        tile.point_count = len(idx)

        if len(idx) == 0:
            tile.density = 0.0
            tile.height_variance = 0.0
            tile.height_mean = 0.0
            tile.verticality = 0.0
            tile.range_mean = float(np.sqrt(tile.cx**2 + tile.cy**2))
            tile.roughness = 0.0
            tile.boundary_score = 0.0
            tile.intensity_mean = 0.0
            return

        pts = points[idx]
        z = pts[:, 2]
        r = np.linalg.norm(pts, axis=1)

        tile.density = len(idx) / (tsize * tsize)
        tile.height_variance = float(np.var(z))
        tile.height_mean = float(np.mean(z))
        tile.range_mean = float(np.mean(r))
        tile.intensity_mean = float(np.mean(intensity[idx]))

        # Verticality — ratio of non-ground points with significant height
        ng = ~ground_mask[idx]
        non_ground_pts = pts[ng]
        if len(non_ground_pts) > 3:
            z_ng = non_ground_pts[:, 2]
            z_spread = z_ng.max() - z_ng.min()
            tile.verticality = min(z_spread / 3.0, 1.0)  # normalised to ~3m max
        else:
            tile.verticality = 0.0

        # Roughness — std of residuals from mean z
        if len(idx) > 2:
            tile.roughness = float(np.std(z))
        else:
            tile.roughness = 0.0

        # Boundary score — gradient of point density at edges
        # Simplified: how different is density in adjacent quadrants
        half = tsize / 4.0
        cx, cy = tile.cx, tile.cy
        def quad_count(xsign, ysign):
            mask = (
                ((pts[:, 0] - cx) * xsign >= 0) &
                ((pts[:, 1] - cy) * ysign >= 0)
            )
            return mask.sum()
        counts = [quad_count(1,1), quad_count(-1,1),
                  quad_count(1,-1), quad_count(-1,-1)]
        if max(counts) > 0:
            tile.boundary_score = float(np.std(counts) / (np.mean(counts) + 1e-6))
            tile.boundary_score = min(tile.boundary_score, 1.0)
        else:
            tile.boundary_score = 0.0

    # ── Main process ──────────────────────────────────────────
    def process(self, frame: Frame, ctx: PipelineContext):
        with stage_timer("S2", frame.timing):
            points = frame.points
            intensity = frame.intensity

            # Ground separation
            ground_mask = self._ground_separate(points)
            frame.ground_mask = ground_mask
            frame.non_ground_mask = ~ground_mask

            # Build tile grid
            tsize = self.cfg.get("tiles", {}).get("size", 2.0)
            tiles = self._build_tiles(points)

            # Compute per-tile geometry features
            for tile in tiles:
                self._compute_tile_features(tile, points, intensity, ground_mask, tsize)

            frame.tiles = tiles
