"""
Adaptive 2.5D map — hash-map of MapCell objects.

DESIGN: Only allocates cells that are actually observed.
No dense global matrix — memory-efficient and resolution-adaptive.

Each cell stores:
  - elevation (ground_z, z_max)
  - log-odds occupancy
  - semantic probability vector (6 classes)
  - dynamic probability
  - provenance (observation_count, timestamp, resolution_level)
  - unknown flag (no LiDAR return ≠ free space)
"""
from __future__ import annotations
import numpy as np
from typing import Dict, Any, Tuple, List

from adaptive_lidar.pipeline.types import MapCell, ResolutionLevel

CellKey = Tuple[int, int]  # (ix, iy) at map resolution


class AdaptiveMap:
    """
    Hash-map based adaptive 2.5D map.
    Multiple resolution levels coexist.
    """

    def __init__(self, config: Dict[str, Any]):
        self.cfg = config
        map_cfg = config.get("map", {})
        self.default_res = map_cfg.get("resolution", 0.10)
        self.log_odds_free = map_cfg.get("log_odds_free", -0.3)
        self.log_odds_occ = map_cfg.get("log_odds_occupied", 0.5)
        self.lo_min = map_cfg.get("log_odds_min", -2.0)
        self.lo_max = map_cfg.get("log_odds_max", 2.0)
        self.temporal_decay = map_cfg.get("temporal_decay", 0.85)

        # Main storage: cell_key → MapCell
        self.cells: Dict[Tuple[CellKey, float], MapCell] = {}

    def _make_key(self, x: float, y: float, res: float) -> Tuple[CellKey, float]:
        ix = int(np.floor(x / res))
        iy = int(np.floor(y / res))
        return ((ix, iy), res)

    def _get_or_create(self, x: float, y: float, res: float,
                       timestamp: float) -> MapCell:
        key = self._make_key(x, y, res)
        if key not in self.cells:
            ix, iy = key[0]
            cell = MapCell(
                cx=(ix + 0.5) * res,
                cy=(iy + 0.5) * res,
                resolution=res,
                last_timestamp=timestamp,
                resolution_level=ResolutionLevel.LEVEL_2,
                is_unknown=True,
            )
            self.cells[key] = cell
        return self.cells[key]

    def update_from_points(
        self,
        points: np.ndarray,         # (N, 3)
        ground_mask: np.ndarray,    # (N,) bool
        semantic_labels: np.ndarray,   # (N,) int
        semantic_probs_map: Dict,   # tile_id → (6,) probs or None
        tile_list,                  # List[Tile]
        timestamp: float,
        resolution: float = None,
    ):
        """Update map cells from current frame observation."""
        if resolution is None:
            resolution = self.default_res

        if len(points) == 0:
            return

        # Index points into cells
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        inv = 1.0 / resolution
        ix_arr = np.floor(x * inv).astype(np.int32)
        iy_arr = np.floor(y * inv).astype(np.int32)

        # Collect z values per cell
        from collections import defaultdict
        cell_z: Dict[tuple, list] = defaultdict(list)
        cell_gnd: Dict[tuple, list] = defaultdict(list)
        cell_sem: Dict[tuple, list] = defaultdict(list)

        for i in range(len(points)):
            k = (int(ix_arr[i]), int(iy_arr[i]))
            cell_z[k].append(float(z[i]))
            cell_gnd[k].append(bool(ground_mask[i]))
            cell_sem[k].append(int(semantic_labels[i]) if semantic_labels is not None else 0)

        # Build a quick tile-lookup by grid cell for O(1) tile access
        tile_probs_lookup = {}
        if tile_list:
            tsize = self.cfg.get("tiles", {}).get("size", 2.0)
            obs_h = self.cfg.get("thresholds", {}).get("obstacle", {}).get("height", 0.4)
            for tile in tile_list:
                if tile.semantic_probs is not None:
                    tix = int(np.floor(tile.cx / tsize))
                    tiy = int(np.floor(tile.cy / tsize))
                    tile_probs_lookup[(tix, tiy)] = tile.semantic_probs
        else:
            tsize = 2.0
            obs_h = 0.4

        for k, z_vals in cell_z.items():
            ix, iy = k
            cx = (ix + 0.5) * resolution
            cy = (iy + 0.5) * resolution

            cell = self._get_or_create(cx, cy, resolution, timestamp)

            z_arr = np.array(z_vals, dtype=np.float32)

            # Elevation update (fast percentile)
            cell.ground_z = float(np.percentile(z_arr, 10))
            cell.z_max = float(np.percentile(z_arr, 95))
            cell.height_variance = float(np.var(z_arr))

            # Occupancy log-odds update
            is_obstacle = (cell.z_max - cell.ground_z) > obs_h
            cell.log_odds = float(np.clip(
                cell.log_odds + (self.log_odds_occ if is_obstacle else self.log_odds_free),
                self.lo_min, self.lo_max))
            cell.occupancy_confidence = float(1.0 / (1.0 + np.exp(-cell.log_odds)))

            # Semantic update from tile-level probs (O(1) lookup)
            if tile_probs_lookup:
                tix = int(np.floor(cx / tsize))
                tiy = int(np.floor(cy / tsize))
                tile_probs = tile_probs_lookup.get((tix, tiy))
                if tile_probs is not None:
                    alpha = self.temporal_decay
                    if cell.semantic_probs is None:
                        cell.semantic_probs = tile_probs.copy()
                    else:
                        cell.semantic_probs = (alpha * cell.semantic_probs +
                                               (1 - alpha) * tile_probs)
                        cell.semantic_probs /= cell.semantic_probs.sum()
                    cell.semantic_class = int(np.argmax(cell.semantic_probs))
                    cell.semantic_confidence = float(cell.semantic_probs.max())

            cell.observation_count += 1
            cell.last_timestamp = timestamp
            cell.is_unknown = False

    def all_cells_as_arrays(self):
        """Return (cx, cy, ground_z, z_max, occupancy, sem_class, sem_conf, is_unknown) arrays."""
        if not self.cells:
            return tuple(np.array([]) for _ in range(8))

        cxs, cys, gzs, zmx, occs, scls, scnfs, unks, dyn = [], [], [], [], [], [], [], [], []
        for cell in self.cells.values():
            cxs.append(cell.cx)
            cys.append(cell.cy)
            gzs.append(cell.ground_z)
            zmx.append(cell.z_max)
            occs.append(cell.occupancy_confidence)
            scls.append(cell.semantic_class)
            scnfs.append(cell.semantic_confidence)
            unks.append(float(cell.is_unknown))
            dyn.append(cell.dynamic_probability)

        return (np.array(cxs), np.array(cys), np.array(gzs),
                np.array(zmx), np.array(occs), np.array(scls),
                np.array(scnfs), np.array(unks), np.array(dyn))
