"""
The adaptive variable-resolution 2.5D map.

THIS IS THE PROBLEM STATEMENT.  Cells in this map have genuinely different
physical sizes — 5 cm near the sensor and around safety-critical structure,
80 cm in open ground at range — and the per-tile resolution decided by the
allocation controller is what determines them.

ALIGNED POWER-OF-TWO HIERARCHY
------------------------------
One global origin, fixed at construction, never moved.  Five levels whose cell
sizes are ``0.05 * 2^l`` — 5 / 10 / 20 / 40 / 80 cm.  Each point gets a 2D
Morton code at level 0; the code at level *l* is that code shifted right by
``2*l`` bits.

That single identity is what satisfies the PS's "without causing alignment
errors or data loss during projection from 3D to 2.5D".  A fine cell is
*always* wholly contained in exactly one coarse cell, because containment is
a prefix relation on the bits.  There is no float boundary case, no point that
rounds into two different parents, and coarsening is an exact sum rather than
a resampling.  (A 5 -> 50 cm ratio, the PS's illustrative example, is not a
power of two and has none of these properties — see DECISIONS.md 4.1.)

Z IS CELL CONTENT, NOT PART OF THE ADDRESS
------------------------------------------
The map is 2.5D.  The Morton code interleaves x and y only.  Height lives
inside the cell as a small set of extracted statistics.

STORAGE
-------
One NumPy structured array (``CELL_DTYPE``) per level plus a sorted int64 key
array, never a dict of Python objects.  A ``MapCell`` dataclass instance costs
~350 bytes of interpreter overhead against 27 bytes of payload; the memory
reduction figure is the headline claim of M5 and has to be measurable on the
real storage, not on a proxy.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from adaptive_lidar.pipeline.types import (
    CELL_DTYPE,
    NUM_CLASSES,
    CellFlags,
    MapCell,
    Occupancy,
    ResolutionLevel,
    Traversability,
    VehicleProfile,
)
from adaptive_lidar.utils.grouping import (
    group_by_key,
    merge_sorted_unique,
    group_sizes,
    morton_2d,
    morton_2d_inverse,
    scatter_to_points,
    segment_bincount,
    segment_percentile_sorted,
    segment_reduce,
    segment_sort,
)

N_LEVELS = ResolutionLevel.N_LEVELS
BASE = ResolutionLevel.BASE

#: Vertical histogram resolution used during insertion (transient).
HIST_BINS = 24
HIST_BIN_M = 0.15                      # 24 * 0.15 = 3.6 m of vertical extent
HIST_Z0 = -0.6                         # histogram starts 0.6 m below local ground


def _f16(a):
    return np.asarray(a, dtype=np.float16)


def _u8(a, lo=0.0, hi=1.0):
    """Quantise a float array into a byte over [lo, hi]."""
    return np.clip((np.asarray(a, np.float32) - lo) / max(hi - lo, 1e-9) * 255.0,
                   0, 255).astype(np.uint8)


def _from_u8(a, lo=0.0, hi=1.0):
    return np.asarray(a, np.float32) / 255.0 * (hi - lo) + lo


class LevelStore:
    """Sorted Morton keys + parallel structured cell array for one level."""

    __slots__ = ("level", "size", "keys", "cells", "_centres", "_centres_for")

    def __init__(self, level: int):
        self.level = int(level)
        self.size = BASE * (2 ** int(level))
        self.keys = np.empty(0, dtype=np.int64)
        self.cells = np.empty(0, dtype=CELL_DTYPE)
        self._centres = None
        self._centres_for = -1

    def __len__(self) -> int:
        return len(self.keys)

    def find(self, keys: np.ndarray) -> np.ndarray:
        """Vectorised key -> row index, -1 where absent."""
        keys = np.asarray(keys, dtype=np.int64)
        if len(self.keys) == 0:
            return np.full(keys.shape, -1, dtype=np.int64)
        pos = np.searchsorted(self.keys, keys)
        posc = np.clip(pos, 0, len(self.keys) - 1)
        return np.where(self.keys[posc] == keys, posc, -1)

    @staticmethod
    def _blank(n: int) -> np.ndarray:
        c = np.zeros(n, dtype=CELL_DTYPE)
        c["overhead_clearance"] = np.float16(np.inf)
        c["evidence_top3_cls"] = 255
        return c

    def ensure(self, keys: np.ndarray, sorted_unique: bool = False) -> np.ndarray:
        """Insert any missing keys (keeping order) and return row indices.

        ``sorted_unique=True`` promises the caller already holds sorted distinct
        keys — which it does whenever they came out of ``group_by_key`` — and
        skips a redundant sort of the whole key array.
        """
        keys = np.asarray(keys, dtype=np.int64)
        if not sorted_unique:
            keys = np.unique(keys)

        if len(self.keys) == 0:
            self.keys = keys
            self.cells = self._blank(len(keys))
            self._centres = None
            self._centres_for = -1
            return np.arange(len(keys), dtype=np.int64)

        merged, insert_at, n_new = merge_sorted_unique(self.keys, keys)
        if n_new:
            new_cells = np.empty(len(merged), dtype=CELL_DTYPE)
            # Splice the existing rows around the inserted blanks in one pass.
            keep = np.ones(len(merged), bool)
            slot = insert_at + np.arange(n_new, dtype=np.int64)
            keep[slot] = False
            new_cells[keep] = self.cells
            new_cells[slot] = self._blank(n_new)
            self.keys = merged
            self.cells = new_cells
            self._centres = None
            self._centres_for = -1
        return np.searchsorted(self.keys, keys)

    def evict(self, keep: np.ndarray):
        self.keys = self.keys[keep]
        self.cells = self.cells[keep]
        self._centres = None
        self._centres_for = -1

    def centres(self) -> Tuple[np.ndarray, np.ndarray]:
        ix, iy = morton_2d_inverse(self.keys, self.level)
        return ((ix + 0.5) * self.size).astype(np.float32), \
               ((iy + 0.5) * self.size).astype(np.float32)

    def nbytes(self) -> int:
        return int(self.keys.nbytes + self.cells.nbytes)


class AdaptiveMap:
    """Variable-resolution 2.5D map with per-level storage."""

    def __init__(self, config: Dict[str, Any] | None = None):
        cfg = config or {}
        m = cfg.get("map", {})
        self.cfg = cfg
        self.levels: List[LevelStore] = [LevelStore(l) for l in range(N_LEVELS)]

        self.log_odds_free = float(m.get("log_odds_free", -0.4))
        self.log_odds_occ = float(m.get("log_odds_occupied", 0.85))
        self.lo_min = float(m.get("log_odds_min", -4.0))
        self.lo_max = float(m.get("log_odds_max", 5.0))
        self.occ_thresh = float(m.get("occupied_logodds", 0.5))
        self.free_thresh = float(m.get("free_logodds", -0.5))

        self.obstacle_height = float(
            cfg.get("thresholds", {}).get("obstacle", {}).get("height", 0.30))
        self.clearance_height = float(m.get("vehicle_clearance_height", 2.5))
        self.multi_surface_gap = float(m.get("multi_surface_gap", 0.5))
        self.carve_min_level = int(m.get("carve_min_level", 3))
        # A cell must be confirmed by SEVERAL observations before free-space
        # carving is forbidden from retracting it. Protecting anything already
        # above the occupied threshold made a single obstacle observation
        # permanent, so the cells a passing car wrote became carve-proof the
        # instant it wrote them - which is exactly the phantom the gate exists
        # to prevent. A wall seen from twenty frames clears this bar easily; a
        # car seen from two does not.
        self.carve_protect = float(
            m.get("carve_protect_logodds", 2.0 * self.log_odds_occ))
        self.window_m = float(m.get("window_size", 200.0))

        self.frame_index = 0
        self.ego_xy = np.zeros(2, dtype=np.float64)
        self._last_slide_xy = np.array([1e9, 1e9])
        self.stats: Dict[str, Any] = {}

    # ════════════════════════════════════════════════════════
    # Insertion
    # ════════════════════════════════════════════════════════
    def update_from_points(
        self,
        points: np.ndarray,                  # (N,3) WORLD coordinates
        levels: np.ndarray,                  # (N,)  resolution level per point
        ground_z: np.ndarray,                # (N,)  local ground height
        evidence: np.ndarray,                # (N,6) per-point class evidence
        intensity: Optional[np.ndarray] = None,
        penetration: Optional[np.ndarray] = None,
        entropy: Optional[np.ndarray] = None,
        disagreement: Optional[np.ndarray] = None,
        safety_pinned: Optional[np.ndarray] = None,
        timestamp: float = 0.0,
        frame_index: Optional[int] = None,
        code0: Optional[np.ndarray] = None,
        sorted_order: Optional[np.ndarray] = None,
    ):
        """Insert one frame's STATIC points.

        ``levels`` is the per-point resolution level, which comes from the tile
        the point falls in.  This is the connection that used to be missing:
        the allocation controller's decision reaches the map here and nowhere
        else.  Points assigned different levels land in different level stores
        and therefore become cells of genuinely different physical size.
        """
        n = len(points)
        if n == 0:
            return
        if frame_index is not None:
            self.frame_index = int(frame_index)

        if intensity is None:
            intensity = np.zeros(n, np.float32)
        if penetration is None:
            penetration = np.zeros(n, np.float32)
        if entropy is None:
            entropy = np.ones(n, np.float32)
        if disagreement is None:
            disagreement = np.zeros(n, bool)
        if safety_pinned is None:
            safety_pinned = np.zeros(n, bool)

        # Level-0 grid index. Every coarser level is a right shift of it.
        if code0 is None:
            inv0 = 1.0 / BASE
            ix0 = np.floor(points[:, 0] * inv0).astype(np.int64)
            iy0 = np.floor(points[:, 1] * inv0).astype(np.int64)
            code0 = morton_2d(ix0, iy0)

        lv = np.clip(np.asarray(levels, dtype=np.int64), 0, N_LEVELS - 1)

        # If the caller hands us the frame's existing Morton ordering, every
        # level's grouping is a run scan rather than a sort: right-shifting a
        # non-decreasing sequence leaves it non-decreasing, and selecting a
        # subset preserves that. Five sorts per frame become none.
        touched = {}
        for level in range(N_LEVELS):
            sel = np.flatnonzero(lv[sorted_order] == level) if sorted_order is not None \
                else np.flatnonzero(lv == level)
            if sel.size == 0:
                continue
            idx = sorted_order[sel] if sorted_order is not None else sel
            codes = code0[idx] >> (2 * level)
            rows = self._insert_level(
                level, codes, points[idx], ground_z[idx], evidence[idx],
                intensity[idx], penetration[idx], entropy[idx],
                disagreement[idx], safety_pinned[idx],
                presorted=sorted_order is not None)
            touched[level] = rows

        return touched

    # ────────────────────────────────────────────────────────
    def _insert_level(self, level, codes, pts, gz, ev, inten, pen, ent,
                      disagree, pinned, presorted: bool = False) -> np.ndarray:
        store = self.levels[level]
        if presorted:
            # Codes are already non-decreasing: grouping is one boolean diff.
            n_c = len(codes)
            first = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1]])
            uniq = codes[first]
            starts = np.r_[first, n_c].astype(np.int64)
            order = np.arange(n_c, dtype=np.int64)
        else:
            uniq, starts, order = group_by_key(codes)
        rows = store.ensure(uniq, sorted_unique=True)
        cells = store.cells

        z = pts[order, 2].astype(np.float32)
        g = gz[order].astype(np.float32)
        rel = z - g                          # height above local ground
        counts = group_sizes(starts).astype(np.float32)

        # ── per-cell vertical histogram (transient) ──────────
        # 24 bins x 15 cm relative to local ground.  One bincount builds every
        # cell's histogram at once; it is read for the five statistics below
        # and then discarded — persisting it would cost 67 MB at 700k cells.
        gid = np.repeat(np.arange(len(uniq), dtype=np.int64), counts.astype(np.int64))
        bin_idx = np.clip(((rel - HIST_Z0) / HIST_BIN_M).astype(np.int64),
                          0, HIST_BINS - 1)
        hist = np.bincount(gid * HIST_BINS + bin_idx,
                           minlength=len(uniq) * HIST_BINS
                           ).reshape(len(uniq), HIST_BINS)

        # ── elevation statistics from percentiles, never min/max ──
        # One noise point must not define a cell.
        #
        # ONE within-group sort serves every order statistic below.  Because
        # the sorted values are ascending, "the points inside the clearance
        # band" and "the points within 0.3 m of the ground" are each a PREFIX
        # of the sorted array, so a percentile restricted to either is just a
        # gather at a different rank — no second sort, no second array.
        z_sorted = segment_sort(z, starts)
        in_band = rel <= self.clearance_height
        n_band = segment_reduce(in_band.astype(np.float32), starts, "sum")
        n_low = segment_reduce((rel < 0.3).astype(np.float32), starts, "sum")

        ground_new = segment_percentile_sorted(z_sorted, starts, 10)
        # z_max is the TRUE maximum inside the clearance band, not a per-cell
        # percentile.  Robustness belongs at the point level - the isolated
        # return filter in S2 removes dust and rain before anything reaches
        # here - because a per-cell percentile does not commute with taking
        # the union of cells, and a map whose statistics cannot be aggregated
        # exactly is worse than one that needed a noise filter anyway.
        # max() commutes, so coarsening stays exact.
        # A masked max, not a rank into the sorted array: `ground_z` varies
        # between points inside one cell, so "inside the clearance band" is not
        # a prefix of the sorted heights and a rank would pick the wrong point.
        z_max_new = segment_reduce(
            np.where(in_band, z, -np.inf), starts, "max")

        # A cell whose ONLY returns are above the clearance band - tree canopy
        # over a road, an overpass deck - contains no obstacle in the band, so
        # both its ground and its z_max are the ground surface underneath.
        # Falling back to the 10th percentile of z here instead put the canopy
        # height into z_max, which then survived coarsening and made a cell
        # that natively reads "clear road" read "3 m obstacle" once merged.
        # The canopy belongs in overhead_clearance, and only there.
        # The fallback must itself be exactly aggregable, since coarsening
        # combines z_max with max(): max-of-maxes equals max-of-union, whereas
        # mean-of-means does not, and the mismatch would show up as a
        # millimetre-scale disagreement between a coarsened and a native map.
        gnd_max = segment_reduce(g, starts, "max")
        gnd_mean = segment_reduce(g, starts, "mean")
        ground_new = np.where(n_band > 0, ground_new, gnd_mean)
        z_max_new = np.where(n_band > 0, z_max_new, gnd_max)

        z_low = np.where(rel < 0.3, z, 0.0)
        c_low = np.maximum(n_low, 1.0)
        m_low = segment_reduce(z_low, starts, "sum") / c_low
        s_low = segment_reduce(z_low * z_low, starts, "sum") / c_low
        z_var_new = np.maximum(s_low - m_low * m_low, 0.0).astype(np.float32)

        clearance, multi, n_surf = self._histogram_features(hist)

        # ── existential obstacle rule ────────────────────────
        # If ANY point sits above ground + obstacle_height, the cell carries an
        # obstacle — regardless of point count or of which class won the vote.
        # A 4 m pole returns three points and must never lose a majority vote.
        above = ((rel > self.obstacle_height) & in_band).astype(np.float32)
        has_obstacle = segment_reduce(above, starts, "sum") > 0

        # ── evidence accumulation: ADDITION, never voting ────
        # Summing evidence IS the conjugate Bayesian update for a categorical
        # likelihood, so temporal fusion and inter-level aggregation are the
        # same operation. Priority-weighted voting would corrupt the posterior
        # and destroy the ability to measure semantic accuracy; safety
        # priorities are applied at allocation and query time instead.
        ev_sum = segment_reduce(ev[order], starts, "sum")

        i_mean = segment_reduce(inten[order], starts, "mean")
        i_var = segment_reduce(inten[order], starts, "var")
        p_mean = segment_reduce(pen[order], starts, "mean")
        e_max = segment_reduce(ent[order], starts, "max")
        dis_any = segment_reduce(disagree[order].astype(np.float32), starts, "sum") > 0
        pin_any = segment_reduce(pinned[order].astype(np.float32), starts, "sum") > 0

        # ── fuse into the store ──────────────────────────────
        old_n = cells["n_points"][rows].astype(np.float32)
        new_n = counts
        tot = old_n + new_n
        w_old = old_n / np.maximum(tot, 1.0)
        w_new = new_n / np.maximum(tot, 1.0)
        fresh = old_n == 0

        # Elevation: inverse-variance (Kalman scalar) fusion, weighted by the
        # observation counts that stand in for it.
        cells["ground_z"][rows] = _f16(
            np.where(fresh, ground_new,
                     w_old * cells["ground_z"][rows].astype(np.float32)
                     + w_new * ground_new))
        cells["z_max"][rows] = _f16(
            np.maximum(np.where(fresh, -np.inf,
                                cells["z_max"][rows].astype(np.float32)), z_max_new))
        cells["z_var"][rows] = _f16(
            np.where(fresh, z_var_new,
                     w_old * cells["z_var"][rows].astype(np.float32) + w_new * z_var_new))
        old_clear = cells["overhead_clearance"][rows].astype(np.float32)
        cells["overhead_clearance"][rows] = _f16(
            np.where(fresh, clearance, np.minimum(old_clear, clearance)))
        # Evidence: decode the stored top-3, add, re-encode.
        # The decode scales by n_points, so it MUST happen before n_points is
        # updated - otherwise every fusion silently re-weights the existing
        # evidence by the incoming count.
        prior = self._decode_evidence(cells[rows])
        cells["n_points"][rows] = np.clip(tot, 0, 65535).astype(np.uint16)
        self._encode_evidence(cells, rows, prior + ev_sum)

        cells["entropy"][rows] = _u8(e_max)
        cells["intensity_mean"][rows] = _u8(
            np.where(fresh, i_mean,
                     w_old * _from_u8(cells["intensity_mean"][rows]) + w_new * i_mean))
        cells["intensity_var"][rows] = _u8(i_var, 0.0, 0.5)
        cells["penetration"][rows] = _u8(p_mean)
        cells["observability"][rows] = np.clip(
            np.maximum(cells["observability"][rows].astype(np.float32), counts * 8.0),
            0, 255).astype(np.uint8)

        lo = cells["occupancy_logodds"][rows].astype(np.float32) * 0.1
        lo = np.where(has_obstacle, lo + self.log_odds_occ, lo + 0.15)
        cells["occupancy_logodds"][rows] = np.clip(
            np.round(np.clip(lo, self.lo_min, self.lo_max) * 10.0), -128, 127
        ).astype(np.int8)

        cells["last_seen"][rows] = np.uint16(self.frame_index % 65536)

        flags = cells["flags"][rows]
        flags = np.where(multi, flags | CellFlags.MULTI_SURFACE, flags)
        flags = np.where(dis_any, flags | CellFlags.GEOM_SEM_DISAGREEMENT, flags)
        flags = np.where(pin_any, flags | CellFlags.SAFETY_PINNED, flags)
        flags = np.where(level <= 1, flags | CellFlags.REFINED, flags)
        flags = flags & ~np.uint8(CellFlags.STALE)
        cells["flags"][rows] = flags.astype(np.uint8)

        self.stats.setdefault("multi_surface_cells", 0)
        self.stats["multi_surface_cells"] += int(multi.sum())
        return rows

    # ────────────────────────────────────────────────────────
    def _histogram_features(self, hist: np.ndarray):
        """overhead_clearance, multi-surface flag and cluster count per cell.

        ``overhead_clearance`` is the height of the bottom of the lowest
        populated cluster ABOVE the clearance band.  This is the single field
        that stops tree canopy over a drivable road from marking it BLOCKED,
        and it removes that whole failure class for two bytes.
        """
        m, B = hist.shape
        occ = hist > 0
        z_of = HIST_Z0 + (np.arange(B) + 0.5) * HIST_BIN_M
        # ceil, not floor: the bin containing the clearance height straddles
        # it, so flooring made every cell with a return at exactly 2.5 m report
        # a clearance of 2.475 m and marked clear road BLOCKED.
        band_bin = int(np.ceil((self.clearance_height - HIST_Z0) / HIST_BIN_M))
        band_bin = int(np.clip(band_bin, 1, B - 1))

        # Lowest occupied bin strictly above the clearance band.
        above = occ[:, band_bin:]
        any_above = above.any(axis=1)
        first_above = np.argmax(above, axis=1)
        clearance = np.where(any_above, z_of[band_bin + first_above], np.inf
                             ).astype(np.float32)

        # Multi-surface: an empty vertical gap of >= multi_surface_gap between
        # two populated clusters (a bridge deck, an overpass, a canopy).
        gap_bins = int(np.ceil(self.multi_surface_gap / HIST_BIN_M))
        # Count transitions occupied -> empty -> occupied with a long enough run.
        multi = np.zeros(m, dtype=bool)
        if B > gap_bins + 1:
            # A run of >= gap_bins empty bins with occupancy on both sides.
            empty_run = np.ones((m, B - gap_bins + 1), dtype=bool)
            for k in range(gap_bins):
                empty_run &= ~occ[:, k:B - gap_bins + 1 + k]
            below = np.cumsum(occ, axis=1) > 0
            above_c = (np.cumsum(occ[:, ::-1], axis=1) > 0)[:, ::-1]
            starts_ok = below[:, :B - gap_bins + 1]
            ends_ok = above_c[:, gap_bins - 1:]
            multi = (empty_run & starts_ok & ends_ok).any(axis=1)

        n_clusters = (occ[:, 1:] & ~occ[:, :-1]).sum(axis=1) + occ[:, 0]
        return clearance, multi, n_clusters.astype(np.int16)

    # ────────────────────────────────────────────────────────
    @staticmethod
    def _decode_evidence(cells: np.ndarray) -> np.ndarray:
        """Reconstruct the (M, 6) evidence vector from the stored top-3."""
        m = len(cells)
        out = np.zeros((m, NUM_CLASSES), np.float32)
        if m == 0:
            return out
        # A never-written cell has n_points == 0 and must decode to zero
        # evidence, not to one point's worth.
        scale = cells["n_points"].astype(np.float32)
        cls = cells["evidence_top3_cls"]
        val = cells["evidence_top3_val"].astype(np.float32) / 255.0
        # One bincount over a flattened (row, class) key: np.add.at is an
        # unbuffered scatter and runs at Python-loop speed on large inputs.
        c = cls.reshape(-1).astype(np.int64)
        v = (val * scale[:, None]).reshape(-1)
        rows = np.repeat(np.arange(m, dtype=np.int64), 3)
        ok = c < NUM_CLASSES
        flat = np.bincount(rows[ok] * NUM_CLASSES + c[ok], weights=v[ok],
                           minlength=m * NUM_CLASSES)
        out += flat[:m * NUM_CLASSES].reshape(m, NUM_CLASSES).astype(np.float32)
        resid = cells["evidence_residual"].astype(np.float32) / 255.0 * scale
        # Spread the residual over the classes OUTSIDE the top 3 only - adding
        # it to all six would double-count the three whose mass is already
        # stored explicitly.
        outside = np.ones((m, NUM_CLASSES), np.float32)
        valid_c = np.where(c < NUM_CLASSES, c, 0)
        outside[rows, valid_c] = np.where(ok, 0.0, outside[rows, valid_c])
        n_outside = np.maximum(outside.sum(axis=1), 1.0)
        out += outside * (resid / n_outside)[:, None]
        return out

    @staticmethod
    def _encode_evidence(cells: np.ndarray, rows: np.ndarray, ev: np.ndarray):
        """Store the top-3 classes + residual mass, normalised to a byte each."""
        tot = np.maximum(ev.sum(axis=1, keepdims=True), 1e-9)
        p = ev / tot
        top3 = np.argsort(-p, axis=1)[:, :3]
        taken = np.take_along_axis(p, top3, axis=1)
        cells["evidence_top3_cls"][rows] = top3.astype(np.uint8)
        cells["evidence_top3_val"][rows] = np.clip(taken * 255.0, 0, 255).astype(np.uint8)
        cells["evidence_residual"][rows] = np.clip(
            (1.0 - taken.sum(axis=1)) * 255.0, 0, 255).astype(np.uint8)

    # ════════════════════════════════════════════════════════
    # Free-space carving
    # ════════════════════════════════════════════════════════
    def carve_free_space(self, frame, pose: np.ndarray, level: Optional[int] = None,
                         step: float = 2.0, max_range: float = 50.0,
                         beam_stride: int = 16):
        """Mark cells along each beam, in front of its hit, as FREE.

        Each range-image pixel IS one real beam with a known direction, so
        there is no ray enumeration to do: sampling along the beam's own
        direction between the sensor and the hit is enough.

        Carving happens at level >= 3 (40 cm) only.  Free space carries no
        shape information, so fine carving buys nothing and would cost 64x the
        cells.  Absence of a return is never treated as free — only the space
        a beam actually traversed.
        """
        ri = getattr(frame, "_range_image", None)
        if ri is None:
            return 0
        level = self.carve_min_level if level is None else max(level, self.carve_min_level)
        store = self.levels[level]
        size = store.size

        valid = ri.valid
        idx = ri.index[valid]
        if idx.size == 0:
            return 0
        # Subsample beams: adjacent beams traverse almost the same 40 cm cells,
        # so carving every one is nearly pure duplication.
        idx = idx[::beam_stride]
        hits = frame.points[idx]
        rng = np.linalg.norm(hits, axis=1)
        keep = rng > step
        hits, rng = hits[keep], rng[keep]
        if len(hits) == 0:
            return 0
        dirs = hits / rng[:, None]

        R, t = pose[:3, :3], pose[:3, 3]
        n_steps = int(min(max_range, float(rng.max())) / step)
        # Cap the number of range samples: at 40 cm cells a 2 m step already
        # skips cells, and doubling the sample count doubles the cost for
        # free space that carries no shape information anyway.
        n_steps = min(n_steps, 24)
        keys = []
        for k in range(1, n_steps + 1):
            d = k * step
            m = rng > d + size          # stop short of the hit
            if not np.any(m):
                break
            p = dirs[m] * d
            pw = p @ R.T + t
            ix = np.floor(pw[:, 0] / size).astype(np.int64)
            iy = np.floor(pw[:, 1] / size).astype(np.int64)
            keys.append(morton_2d(ix, iy, level=level))
        if not keys:
            return 0

        codes = np.unique(np.concatenate(keys))
        rows = store.ensure(codes, sorted_unique=True)
        cells = store.cells
        # Only decrement cells that are not already strongly occupied — a beam
        # that grazes a wall must not carve the wall away.
        lo = cells["occupancy_logodds"][rows].astype(np.float32) * 0.1
        protect = lo >= self.carve_protect
        lo = np.where(protect, lo, lo + self.log_odds_free)
        cells["occupancy_logodds"][rows] = np.clip(
            np.round(np.clip(lo, self.lo_min, self.lo_max) * 10.0), -128, 127
        ).astype(np.int8)

        # Free space established at 40 cm invalidates whatever finer cells sit
        # inside it.  Without this, carving can never undo an obstacle written
        # by a moving object: the car's points land in 5 cm cells, the beams
        # that later pass through its empty parking space are carved at 40 cm,
        # and the two never meet — so the phantom survives every observation
        # that disproves it.  The containment is the Morton prefix again, so
        # propagating the verdict downward is one searchsorted per level.
        freed = codes[(lo <= self.free_thresh) & ~protect]
        if len(freed):
            self._invalidate_finer(freed, level)
        return len(codes)

    def _invalidate_finer(self, freed_codes: np.ndarray, level: int):
        """Clear obstacle evidence in cells contained by a freed coarse cell."""
        for fl in range(0, level):
            st = self.levels[fl]
            if len(st) == 0:
                continue
            shift = 2 * (level - fl)
            parents = st.keys >> shift
            hit = np.isin(parents, freed_codes)
            if not hit.any():
                continue
            c = st.cells
            # The surface is gone, not the terrain: ground_z is what we knew
            # about the floor and remains true, while z_max, the obstacle, has
            # been observed through.
            c["z_max"][hit] = c["ground_z"][hit]
            c["n_points"][hit] = 0
            c["occupancy_logodds"][hit] = np.int8(
                round(self.free_thresh * 10.0))
            c["flags"][hit] = (c["flags"][hit] | CellFlags.STALE).astype(np.uint8)

    # ════════════════════════════════════════════════════════
    # Maintenance
    # ════════════════════════════════════════════════════════
    def slide_window(self, ego_xy: np.ndarray, force: bool = False):
        """Evict cells outside a fixed-size window centred on the ego vehicle.

        Bounded accumulation also bounds how far pose drift can smear the map.
        """
        ego_xy = np.asarray(ego_xy, dtype=np.float64)[:2]
        # Evicting is O(cells); doing it every frame when the ego has moved a
        # metre inside a 200 m window is wasted work. Slide once the ego has
        # travelled far enough for anything to have left the window.
        moved = float(np.hypot(*(ego_xy - getattr(self, "_last_slide_xy",
                                                  np.array([1e9, 1e9])))))
        if not force and moved < self.window_m * 0.05:
            self.ego_xy = ego_xy
            return 0
        self._last_slide_xy = ego_xy.copy()
        self.ego_xy = ego_xy
        half = self.window_m / 2.0
        removed = 0
        for store in self.levels:
            if len(store) == 0:
                continue
            cx, cy = store.centres()
            keep = (np.abs(cx - self.ego_xy[0]) <= half) & \
                   (np.abs(cy - self.ego_xy[1]) <= half)
            if not keep.all():
                removed += int((~keep).sum())
                store.evict(keep)
        return removed

    def decay(self, class_half_life_frames: Optional[np.ndarray] = None):
        """Class-dependent evidence decay.

        Buildings and walls essentially never decay, poles and kerbs slowly,
        vegetation and vehicles at a medium rate, low-evidence cells fast.
        Decay is applied to the occupancy log-odds; evidence itself is left
        alone so the semantic posterior stays a sum of real observations.
        """
        if class_half_life_frames is None:
            # index by class: ground, rough, static, vehicle, vru, vegetation
            class_half_life_frames = np.array(
                [400.0, 400.0, 2000.0, 60.0, 25.0, 120.0], np.float32)
        for store in self.levels:
            if len(store) == 0:
                continue
            cells = store.cells
            cls = cells["evidence_top3_cls"][:, 0].astype(np.int64)
            cls = np.where(cls < NUM_CLASSES, cls, 2)
            hl = class_half_life_frames[cls]
            age = (np.uint16(self.frame_index % 65536).astype(np.int32)
                   - cells["last_seen"].astype(np.int32)) % 65536
            f = np.power(0.5, age / np.maximum(hl, 1.0)).astype(np.float32)
            lo = cells["occupancy_logodds"].astype(np.float32) * 0.1 * f
            cells["occupancy_logodds"] = np.clip(
                np.round(lo * 10.0), -128, 127).astype(np.int8)
            stale = age > (hl * 2)
            cells["flags"] = np.where(
                stale, cells["flags"] | CellFlags.STALE, cells["flags"]).astype(np.uint8)

    # ════════════════════════════════════════════════════════
    # Aggregation — the exactness proof
    # ════════════════════════════════════════════════════════
    def coarsen_to(self, target_level: int) -> "AdaptiveMap":
        """Return a new map with every cell aggregated to ``target_level``.

        Because the level-l key is the level-0 key shifted right by 2l bits,
        aggregation is exactly a group-by on the shifted key.  ``n_points`` and
        the evidence sums are therefore EXACT sums, not resampled estimates,
        and ``z_max`` is an exact max.  ``test_map.py`` asserts this against a
        map built natively at the target level.
        """
        out = AdaptiveMap(self.cfg)
        out.frame_index = self.frame_index
        dst = out.levels[target_level]

        all_keys, all_cells = [], []
        for store in self.levels:
            if len(store) == 0:
                continue
            shift = 2 * (target_level - store.level)
            if shift < 0:
                # Already coarser than the target: a coarse cell cannot be
                # split into fine ones without inventing data, so it is
                # promoted whole.
                keys = store.keys << (-shift)
            else:
                keys = store.keys >> shift
            all_keys.append(keys)
            all_cells.append(store.cells)
        if not all_keys:
            return out

        keys = np.concatenate(all_keys)
        cells = np.concatenate(all_cells)

        uniq, starts, order = group_by_key(keys)
        c = cells[order]
        dst.keys = uniq
        dst.cells = np.zeros(len(uniq), dtype=CELL_DTYPE)
        d = dst.cells

        npts = c["n_points"].astype(np.float32)
        tot = segment_reduce(npts, starts, "sum")
        d["n_points"] = np.clip(tot, 0, 65535).astype(np.uint16)

        # Weighted means for the fields that are means; max/min for extrema.
        def wmean(field, lo=None, hi=None):
            v = (c[field].astype(np.float32) if lo is None
                 else _from_u8(c[field], lo, hi))
            s = segment_reduce(v * npts, starts, "sum")
            return s / np.maximum(tot, 1.0)

        d["ground_z"] = _f16(wmean("ground_z"))
        d["z_var"] = _f16(wmean("z_var"))
        d["z_max"] = _f16(segment_reduce(c["z_max"].astype(np.float32), starts, "max"))
        d["overhead_clearance"] = _f16(
            segment_reduce(c["overhead_clearance"].astype(np.float32), starts, "min"))
        d["intensity_mean"] = _u8(wmean("intensity_mean", 0.0, 1.0))
        d["intensity_var"] = _u8(wmean("intensity_var", 0.0, 0.5), 0.0, 0.5)
        d["penetration"] = _u8(wmean("penetration", 0.0, 1.0))
        d["entropy"] = _u8(segment_reduce(
            _from_u8(c["entropy"]), starts, "max"))
        d["observability"] = segment_reduce(
            c["observability"], starts, "max").astype(np.uint8)
        d["occupancy_logodds"] = np.clip(segment_reduce(
            c["occupancy_logodds"].astype(np.float32), starts, "max"),
            -128, 127).astype(np.int8)
        d["dynamic_prob"] = segment_reduce(c["dynamic_prob"], starts, "max").astype(np.uint8)
        d["last_seen"] = segment_reduce(
            c["last_seen"].astype(np.int32), starts, "max").astype(np.uint16)
        # Flags are a bit set: OR them, never sum them.
        d["flags"] = _segment_or(c["flags"], starts)

        ev = self._decode_evidence(c)
        ev_sum = segment_reduce(ev, starts, "sum")
        self._encode_evidence(d, np.arange(len(uniq)), ev_sum)
        return out

    # ════════════════════════════════════════════════════════
    # Queries
    # ════════════════════════════════════════════════════════
    def all_cells_arrays(self) -> Dict[str, np.ndarray]:
        """Flatten every level into parallel arrays for rendering / metrics."""
        cx, cy, size, lvl, rows = [], [], [], [], []
        for store in self.levels:
            if len(store) == 0:
                continue
            a, b = store.centres()
            cx.append(a)
            cy.append(b)
            size.append(np.full(len(store), store.size, np.float32))
            lvl.append(np.full(len(store), store.level, np.int8))
            rows.append(store.cells)
        if not cx:
            empty = np.zeros(0, np.float32)
            return {"cx": empty, "cy": empty, "size": empty,
                    "level": np.zeros(0, np.int8),
                    "cells": np.zeros(0, dtype=CELL_DTYPE)}
        cells = np.concatenate(rows)
        ev = self._decode_evidence(cells)
        p = ev / np.maximum(ev.sum(axis=1, keepdims=True), 1e-9)
        lo = cells["occupancy_logodds"].astype(np.float32) * 0.1
        occ_state = np.where(lo >= self.occ_thresh, Occupancy.OCCUPIED,
                             np.where(lo <= self.free_thresh, Occupancy.FREE,
                                      Occupancy.UNKNOWN)).astype(np.int8)
        return {
            "cx": np.concatenate(cx),
            "cy": np.concatenate(cy),
            "size": np.concatenate(size),
            "level": np.concatenate(lvl),
            "cells": cells,
            "ground_z": cells["ground_z"].astype(np.float32),
            "z_max": cells["z_max"].astype(np.float32),
            "overhead_clearance": cells["overhead_clearance"].astype(np.float32),
            "z_var": cells["z_var"].astype(np.float32),
            "n_points": cells["n_points"].astype(np.int32),
            "sem_class": np.argmax(p, axis=1).astype(np.int8),
            "sem_prob": p,
            "entropy": _from_u8(cells["entropy"]),
            "occupancy_logodds": lo,
            "occupancy_state": occ_state,
            "dynamic_prob": _from_u8(cells["dynamic_prob"]),
            "intensity_mean": _from_u8(cells["intensity_mean"]),
            "intensity_var": _from_u8(cells["intensity_var"], 0.0, 0.5),
            "observability": _from_u8(cells["observability"]),
            "penetration": _from_u8(cells["penetration"]),
            "flags": cells["flags"],
            "last_seen": cells["last_seen"].astype(np.int32),
        }

    def cell_counts(self) -> Dict[int, int]:
        return {l: len(s) for l, s in enumerate(self.levels)}

    @property
    def n_cells(self) -> int:
        return sum(len(s) for s in self.levels)

    def nbytes(self) -> int:
        """Payload bytes actually held by the level stores."""
        return sum(s.nbytes() for s in self.levels)

    def cells_view(self) -> Dict[tuple, MapCell]:
        """Legacy dict-of-MapCell view. Debugging and the cell inspector only —
        materialising this defeats the whole point of the packed storage."""
        out = {}
        a = self.all_cells_arrays()
        for i in range(len(a["cx"])):
            out[(float(a["cx"][i]), float(a["cy"][i]))] = self._make_cell(a, i)
        return out

    def _make_cell(self, a, i) -> MapCell:
        c = a["cells"][i]
        return MapCell(
            cx=float(a["cx"][i]), cy=float(a["cy"][i]),
            resolution=float(a["size"][i]), level=int(a["level"][i]),
            ground_z=float(a["ground_z"][i]), z_max=float(a["z_max"][i]),
            overhead_clearance=float(a["overhead_clearance"][i]),
            z_var=float(a["z_var"][i]), n_points=int(a["n_points"][i]),
            evidence=a["sem_prob"][i], semantic_class=int(a["sem_class"][i]),
            entropy=float(a["entropy"][i]),
            occupancy_logodds=float(a["occupancy_logodds"][i]),
            occupancy_state=int(a["occupancy_state"][i]),
            dynamic_probability=float(a["dynamic_prob"][i]),
            last_seen=int(a["last_seen"][i]),
            intensity_mean=float(a["intensity_mean"][i]),
            intensity_var=float(_from_u8(c["intensity_var"], 0.0, 0.5)),
            penetration=float(a["penetration"][i]),
            observability=float(c["observability"]) / 255.0,
            flags=int(c["flags"]))

    def cell_at(self, x: float, y: float) -> Optional[MapCell]:
        """Finest cell covering (x, y), or None.

        Searched finest-first: if a fine cell exists there it is the more
        informative answer.
        """
        ix0 = int(np.floor(x / BASE))
        iy0 = int(np.floor(y / BASE))
        code0 = int(morton_2d(np.array([ix0]), np.array([iy0]))[0])
        for level in range(N_LEVELS):
            store = self.levels[level]
            if len(store) == 0:
                continue
            row = int(store.find(np.array([code0 >> (2 * level)]))[0])
            if row >= 0:
                a = self._level_arrays(level)
                return self._make_cell(a, row)
        return None

    def _level_arrays(self, level: int) -> Dict[str, np.ndarray]:
        store = self.levels[level]
        cx, cy = store.centres()
        cells = store.cells
        ev = self._decode_evidence(cells)
        p = ev / np.maximum(ev.sum(axis=1, keepdims=True), 1e-9)
        lo = cells["occupancy_logodds"].astype(np.float32) * 0.1
        return {
            "cx": cx, "cy": cy,
            "size": np.full(len(store), store.size, np.float32),
            "level": np.full(len(store), level, np.int8),
            "cells": cells,
            "ground_z": cells["ground_z"].astype(np.float32),
            "z_max": cells["z_max"].astype(np.float32),
            "overhead_clearance": cells["overhead_clearance"].astype(np.float32),
            "z_var": cells["z_var"].astype(np.float32),
            "n_points": cells["n_points"].astype(np.int32),
            "sem_class": np.argmax(p, axis=1).astype(np.int8),
            "sem_prob": p,
            "entropy": _from_u8(cells["entropy"]),
            "occupancy_logodds": lo,
            "occupancy_state": np.where(
                lo >= self.occ_thresh, Occupancy.OCCUPIED,
                np.where(lo <= self.free_thresh, Occupancy.FREE,
                         Occupancy.UNKNOWN)).astype(np.int8),
            "dynamic_prob": _from_u8(cells["dynamic_prob"]),
            "intensity_mean": _from_u8(cells["intensity_mean"]),
            "intensity_var": _from_u8(cells["intensity_var"], 0.0, 0.5),
            "observability": _from_u8(cells["observability"]),
            "penetration": _from_u8(cells["penetration"]),
            "flags": cells["flags"],
            "last_seen": cells["last_seen"].astype(np.int32),
        }

    # ════════════════════════════════════════════════════════
    # M2 — traversability
    # ════════════════════════════════════════════════════════
    def traversability_arrays(
        self,
        profile: Optional[VehicleProfile] = None,
    ) -> Dict[str, np.ndarray]:
        """Vectorised traversability verdict for every cell.

        Nothing here is stored.  Slope, roughness, step height and clearance
        are all derived at query time from the physical terrain properties the
        map does store — which is what makes the same map answer correctly for
        a wheeled and a tracked vehicle.
        """
        profile = profile or VehicleProfile.wheeled()
        a = self.all_cells_arrays()
        n = len(a["cx"])
        if n == 0:
            return {"verdict": np.zeros(0, np.int8), "reason": [],
                    "slope_deg": np.zeros(0, np.float32),
                    "step": np.zeros(0, np.float32), **a}

        slope, step = self._local_gradients()
        rough = np.sqrt(np.maximum(a["z_var"], 0.0))
        clear = a["overhead_clearance"]
        cls = a["sem_class"]
        obst = np.maximum(a["z_max"] - a["ground_z"], 0.0)

        blocked = (
            (obst > profile.max_step_m * 2.0)
            | (slope > profile.max_slope_deg)
            | (step > profile.max_step_m)
            | (clear < profile.min_clearance_m)
            | np.isin(cls, (2, 3, 4))
        )
        caution = (
            (rough > profile.max_roughness)
            | (slope > profile.max_slope_deg * 0.6)
            | (step > profile.max_step_m * 0.6)
            | (cls == 1) | (cls == 5)
            # Sparsely observed, not "occupancy unknown": a cell that holds
            # points has been observed by definition, and the three-state
            # occupancy describes free space, not surface confidence. Using it
            # here marked almost every surface cell CAUTION.
            | (a["n_points"] < 2)
        )
        verdict = np.where(blocked, Traversability.BLOCKED,
                           np.where(caution, Traversability.CAUTION,
                                    Traversability.DRIVABLE)).astype(np.int8)
        return {**a, "verdict": verdict, "slope_deg": slope,
                "step": step, "roughness": rough, "obstacle_height": obst}

    def traversability(self, x: float, y: float,
                       profile: Optional[VehicleProfile] = None
                       ) -> Tuple[int, str]:
        """Verdict plus a human-readable reason for one location."""
        profile = profile or VehicleProfile.wheeled()
        cell = self.cell_at(x, y)
        if cell is None:
            return Traversability.CAUTION, "unobserved (no cell)"

        obst = cell.obstacle_height
        rough = float(np.sqrt(max(cell.z_var, 0.0)))
        slope, step = self._neighbour_slope_step(x, y, cell)

        reasons = []
        if obst > profile.max_step_m * 2.0:
            reasons.append(f"obstacle {obst:.2f} m > {profile.max_step_m * 2:.2f} m")
        if slope > profile.max_slope_deg:
            reasons.append(f"slope {slope:.1f}deg > {profile.max_slope_deg:.0f}deg")
        if step > profile.max_step_m:
            reasons.append(f"step {step:.2f} m > {profile.max_step_m:.2f} m")
        if cell.overhead_clearance < profile.min_clearance_m:
            reasons.append(
                f"overhead clearance {cell.overhead_clearance:.2f} m "
                f"< {profile.min_clearance_m:.2f} m")
        if cell.semantic_class in (2, 3, 4):
            from adaptive_lidar.pipeline.types import CLASS_NAMES
            reasons.append(f"class {CLASS_NAMES[cell.semantic_class]}")
        if reasons:
            return Traversability.BLOCKED, "; ".join(reasons)

        soft = []
        if rough > profile.max_roughness:
            soft.append(f"roughness {rough:.3f} > {profile.max_roughness:.3f}")
        if slope > profile.max_slope_deg * 0.6:
            soft.append(f"slope {slope:.1f}deg approaching limit")
        if cell.n_points < 2:
            soft.append(f"sparsely observed ({cell.n_points} point(s))")
        if cell.semantic_class in (1, 5):
            soft.append("rough / vegetated terrain")
        if soft:
            return Traversability.CAUTION, "; ".join(soft)

        return Traversability.DRIVABLE, (
            f"slope {slope:.1f}deg, step {step:.2f} m, roughness {rough:.3f}, "
            f"clearance {cell.overhead_clearance:.1f} m — all within "
            f"{profile.name} limits")

    # ────────────────────────────────────────────────────────
    def _local_gradients(self) -> Tuple[np.ndarray, np.ndarray]:
        """Slope (deg) and max neighbour step (m) for every cell.

        Computed on a uniform coarse raster of ``ground_z`` so that cells of
        different sizes can be compared to their neighbours at all.
        """
        a = self.all_cells_arrays()
        n = len(a["cx"])
        if n == 0:
            return np.zeros(0, np.float32), np.zeros(0, np.float32)

        res = 0.4
        ix = np.floor(a["cx"] / res).astype(np.int64)
        iy = np.floor(a["cy"] / res).astype(np.int64)
        x0, y0 = ix.min(), iy.min()
        w = int(ix.max() - x0) + 1
        h = int(iy.max() - y0) + 1
        if w * h > 20_000_000:
            return np.zeros(n, np.float32), np.zeros(n, np.float32)

        grid = np.full((h, w), np.nan, np.float32)
        grid[iy - y0, ix - x0] = a["ground_z"]
        filled = np.nan_to_num(grid, nan=0.0)
        known = ~np.isnan(grid)

        def shift(arr, dy, dx):
            return np.roll(np.roll(arr, dy, 0), dx, 1)

        steps = np.zeros_like(filled)
        gx = np.zeros_like(filled)
        gy = np.zeros_like(filled)
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            d = np.abs(shift(filled, dy, dx) - filled)
            d = np.where(shift(known, dy, dx) & known, d, 0.0)
            steps = np.maximum(steps, d)
        gx = np.where(shift(known, 0, 1) & shift(known, 0, -1),
                      (shift(filled, 0, -1) - shift(filled, 0, 1)) / (2 * res), 0.0)
        gy = np.where(shift(known, 1, 0) & shift(known, -1, 0),
                      (shift(filled, -1, 0) - shift(filled, 1, 0)) / (2 * res), 0.0)
        slope_img = np.degrees(np.arctan(np.hypot(gx, gy)))

        return (slope_img[iy - y0, ix - x0].astype(np.float32),
                steps[iy - y0, ix - x0].astype(np.float32))

    def _neighbour_slope_step(self, x, y, cell) -> Tuple[float, float]:
        d = max(cell.resolution, 0.2)
        gz = []
        for dx, dy in ((d, 0), (-d, 0), (0, d), (0, -d)):
            c = self.cell_at(x + dx, y + dy)
            if c is not None:
                gz.append(c.ground_z)
        if not gz:
            return 0.0, 0.0
        gz = np.array(gz, np.float32)
        step = float(np.max(np.abs(gz - cell.ground_z)))
        slope = float(np.degrees(np.arctan(step / d)))
        return slope, step

    # ── legacy alias ──────────────────────────────────────────
    @property
    def cells(self):
        return self.cells_view()


def _segment_or(values: np.ndarray, group_starts: np.ndarray) -> np.ndarray:
    """Bitwise OR within each group."""
    return np.bitwise_or.reduceat(values, group_starts[:-1]).astype(np.uint8)


class UniformReference:
    """A uniform-resolution map over exactly the same observed area.

    Exists so the memory comparison is like-for-like.  The adaptive map
    accumulates across frames and slides a world window; comparing it against a
    single frame's uniform cell count would flatter it enormously, so this
    accumulates and slides identically and stores one int64 key per observed
    cell.  Only cells the sensor actually saw are counted — a dense raster over
    the bounding box would be a rigged comparison, since the adaptive map is
    sparse by construction and would "win" on emptiness alone.
    """

    def __init__(self, cell: float = 0.05, window_m: float = 200.0,
                 bytes_per_cell: Optional[int] = None):
        from adaptive_lidar.pipeline.types import BYTES_PER_CELL
        self.cell = float(cell)
        self.window_m = float(window_m)
        self.bytes_per_cell = BYTES_PER_CELL if bytes_per_cell is None else bytes_per_cell
        self.keys = np.empty(0, dtype=np.int64)

    def update(self, points_xy: np.ndarray):
        if len(points_xy) == 0:
            return
        ix = np.floor(points_xy[:, 0] / self.cell).astype(np.int64)
        iy = np.floor(points_xy[:, 1] / self.cell).astype(np.int64)
        self.update_codes(np.sort(morton_2d(ix, iy)))

    def update_codes(self, sorted_codes: np.ndarray):
        """Accumulate from already-sorted level-0 Morton codes.

        The map insert has just computed and sorted these, so the reference
        costs one run scan instead of its own Morton pass, sort and unique.
        """
        if len(sorted_codes) == 0:
            return
        uniq = sorted_codes[np.r_[True, sorted_codes[1:] != sorted_codes[:-1]]]
        self.keys, _, _ = merge_sorted_unique(self.keys, uniq)

    def slide_window(self, ego_xy: np.ndarray, force: bool = False):
        if len(self.keys) == 0:
            return
        ix, iy = morton_2d_inverse(self.keys, 0)
        cx = (ix + 0.5) * self.cell
        cy = (iy + 0.5) * self.cell
        half = self.window_m / 2.0
        keep = (np.abs(cx - ego_xy[0]) <= half) & (np.abs(cy - ego_xy[1]) <= half)
        self.keys = self.keys[keep]

    @property
    def n_cells(self) -> int:
        return int(len(self.keys))

    def nbytes(self) -> int:
        return self.n_cells * self.bytes_per_cell


def uniform_grid_memory(points_xy: np.ndarray, cell: float = 0.05,
                        bytes_per_cell: Optional[int] = None) -> Tuple[int, int]:
    """(n_cells, bytes) a uniform grid at ``cell`` would need over the same area.

    Counts only cells the sensor actually observed — a dense raster over the
    bounding box would be a rigged comparison, since the adaptive map is sparse
    by construction and would "win" on emptiness alone.
    """
    from adaptive_lidar.pipeline.types import BYTES_PER_CELL
    if bytes_per_cell is None:
        bytes_per_cell = BYTES_PER_CELL
    if len(points_xy) == 0:
        return 0, 0
    ix = np.floor(points_xy[:, 0] / cell).astype(np.int64)
    iy = np.floor(points_xy[:, 1] / cell).astype(np.int64)
    n = len(np.unique(morton_2d(ix, iy)))
    return n, n * bytes_per_cell
