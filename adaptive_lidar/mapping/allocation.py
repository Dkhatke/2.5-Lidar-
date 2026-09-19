"""
The allocation controller — which tile gets which cell size, under a budget.

This is the research contribution.  Everything else in the system exists to
feed it or to measure it.

THE BUDGET IS CELLS, NOT TILES
------------------------------
The previous implementation computed ``n_budget = int(n_total * budget)`` — a
*tile-count* budget, which bounds nothing about memory.  Refining 10% of tiles
by four levels costs more than refining every tile by one.  Here the budget is
a number of cells (trivially convertible to megabytes), which is the quantity
the PS's memory claim is actually about.

VALUE AND COST
--------------
    V(tile) = w_g*G + w_s*S + w_u*U + w_d*D

    G  geometric complexity : z_spread, z_var, has_vertical_run, histogram gap
    S  semantic stake       : max class importance over the tile's points
    U  epistemic uncertainty: max entropy, geom-sem disagreement, low observability
    D  dynamic relevance    : max moving_prob * speed

Refining a tile by one level multiplies its cell count by four, so tiles are
ranked by value per unit cost, ``rho = V / delta_cells``, not by V.

SAFETY IS A CONSTRAINT, NOT A TERM
----------------------------------
A weighted sum makes safety tradeable: with enough competing tiles the 70 m
pedestrian is outbid, and the system degrades exactly where it must not.  Pins
define the feasible set instead — they consume budget FIRST and are never
ranked against it.

The pin that matters is the GEOMETRIC one (``has_vertical_run``).  A semantic
pin is only as good as the classifier; if the network misses a pedestrian, no
policy built on its output protects them.  The geometric pin fires on "small,
isolated, vertically-extended cluster standing above the ground" without
knowing what the thing is, so the retention guarantee survives a segmentation
failure.  Semantics are an enhancement on top of it.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from adaptive_lidar.pipeline.types import ResolutionLevel, Tile

N_LEVELS = ResolutionLevel.N_LEVELS
COARSEST = N_LEVELS - 1


# ════════════════════════════════════════════════════════════
# 5.6 — the resolution schedule, derived rather than hardcoded
# ════════════════════════════════════════════════════════════
def resolution_law(r: np.ndarray, target_points_per_cell: float = 4.0,
                   anchor_r: float = 10.0, anchor_size: float = 0.05,
                   exponent: float = 1.0) -> np.ndarray:
    """Continuous cell size (m) that holds points-per-cell constant with range.

    A spinning LiDAR's beams diverge linearly in azimuth and, striking a ground
    plane at a shallow grazing angle, spread quadratically in the radial
    direction.  The ground area sampled per beam therefore grows roughly as
    r^3, and the number of beams falling on a fixed patch of ground falls off
    as 1/r^3.  Holding the expected point count per cell constant requires the
    cell *area* to grow as r^2, i.e. the cell *size* to grow as r:

        s(r) = anchor_size * (r / anchor_r)^exponent

    Anchoring 5 cm at 10 m with the linear law gives

        s(100 m) = 0.05 * (100/10) = 0.50 m

    which is exactly the "~50 cm far" figure the problem statement offers as an
    example — derived from the sensor geometry rather than assumed.  The
    schedule is then quantised onto the power-of-two level ladder, so the
    coarsest level used in practice is 40 cm at 80 m and 80 cm beyond.
    """
    r = np.maximum(np.asarray(r, dtype=np.float32), 1e-3)
    scale = np.sqrt(max(target_points_per_cell, 1e-3) / 4.0)
    return (anchor_size * (r / anchor_r) ** exponent * scale).astype(np.float32)


def quantise_to_level(size: np.ndarray) -> np.ndarray:
    """Nearest power-of-two level for a continuous cell size."""
    sizes = ResolutionLevel.sizes_array()
    # Nearest in log space: the ladder is geometric, so relative error is the
    # right thing to minimise.
    d = np.abs(np.log(np.asarray(size, np.float32)[:, None] / sizes[None, :]))
    return np.argmin(d, axis=1).astype(np.int8)


def distance_levels(ranges: np.ndarray, **kw) -> np.ndarray:
    """Level per tile from the derived schedule alone."""
    return quantise_to_level(resolution_law(ranges, **kw))


# ════════════════════════════════════════════════════════════
# Feature extraction from tiles
# ════════════════════════════════════════════════════════════
def tile_arrays(tiles: Sequence[Tile]) -> Dict[str, np.ndarray]:
    """Vectorise the tile list once; every policy reads these arrays."""
    n = len(tiles)
    a = {k: np.zeros(n, np.float32) for k in
         ("z_spread", "z_var", "hist_gap", "importance", "entropy", "moving",
          "range", "count", "valid_px", "roughness", "boundary", "density")}
    a["vertical_run"] = np.zeros(n, bool)
    a["disagree"] = np.zeros(n, bool)
    for i, t in enumerate(tiles):
        a["z_spread"][i] = t.z_spread
        a["z_var"][i] = t.height_variance
        a["hist_gap"][i] = t.histogram_gap
        a["importance"][i] = t.max_class_importance
        a["entropy"][i] = t.max_entropy
        a["moving"][i] = t.max_moving_prob
        a["range"][i] = t.range_mean
        a["count"][i] = t.point_count
        a["valid_px"][i] = t.valid_pixel_count
        a["roughness"][i] = t.roughness
        a["boundary"][i] = t.boundary_score
        a["density"][i] = t.density
        a["vertical_run"][i] = t.has_vertical_run
    return a


def tile_cells_at_level(index, n_tiles: int) -> np.ndarray:
    """(n_tiles, 5) EXACT number of occupied cells each tile would have.

    The budget is only a memory budget if the cost model is the real cell
    count.  Estimating it as ``min(n_points, capacity)`` overstates the fine
    levels by two to three times, because LiDAR points lie on surfaces and
    cluster heavily: 400 points in a tile occupy nothing like 400 distinct
    5 cm cells.

    Exact is also free here.  The frame is already ordered by level-0 Morton
    code and the tile is a node of the same hierarchy, so the level-l code is
    a right shift of the sorted code and the occupied-cell count at level l is
    simply the number of runs of that shift within each tile.  Five boolean
    diffs, no sort.
    """
    out = np.zeros((n_tiles, N_LEVELS), dtype=np.int64)
    if index is None or len(index.sorted_code0) == 0:
        return out

    sc = index.sorted_code0
    starts = index.tile_starts
    counts = np.diff(starts)
    tid = np.repeat(np.arange(n_tiles, dtype=np.int64), counts)

    new_tile = np.zeros(len(sc), bool)
    new_tile[starts[:-1]] = True

    for lev in range(N_LEVELS):
        cl = sc >> (2 * lev)
        first = new_tile | np.r_[True, cl[1:] != cl[:-1]]
        out[:, lev] = np.bincount(tid[first], minlength=n_tiles)
    return out


def _norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float32)
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi - lo < 1e-6:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def value_terms(a: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """The four value components, each normalised into [0, 1]."""
    G = (0.35 * _norm(a["z_spread"])
         + 0.25 * _norm(a["z_var"])
         + 0.20 * a["vertical_run"].astype(np.float32)
         + 0.20 * _norm(a["hist_gap"]))
    S = a["importance"]
    # Low observability is uncertainty: few beams reached the tile, so what is
    # there is genuinely unknown rather than known to be empty.
    obs = 1.0 - _norm(np.log1p(a["valid_px"]))
    U = np.clip(0.55 * a["entropy"] + 0.25 * obs
                + 0.20 * a["disagree"].astype(np.float32), 0.0, 1.0)
    D = a["moving"]
    return {"G": G.astype(np.float32), "S": S.astype(np.float32),
            "U": U.astype(np.float32), "D": D.astype(np.float32)}


# ════════════════════════════════════════════════════════════
# The allocator
# ════════════════════════════════════════════════════════════
class AllocationController:
    """One interface, many policies: ``allocate(tiles, budget) -> levels``.

    Policy selection is a single config string.  That is a deliberate design
    decision: the Phase 7 ablation is ~10 policies x 5 budgets, which is an
    afternoon behind one interface and a week of refactoring if policy logic is
    scattered through the map code.
    """

    POLICIES = (
        "uniform_5", "uniform_10", "uniform_20", "uniform_40", "uniform_80",
        "distance_only",
        "distance_geometry",
        "distance_geometry_semantic",
        "distance_geometry_semantic_uncertainty",
        "full",
        "random",
    )

    def __init__(self, config: Dict[str, Any] | None = None):
        cfg = config or {}
        w = cfg.get("weights", {})
        self.wg = float(w.get("geometry", 0.30))
        self.ws = float(w.get("semantic", 0.30))
        self.wu = float(w.get("uncertainty", 0.20))
        self.wd = float(w.get("dynamic", 0.20))

        al = cfg.get("allocation", {})
        self.tile_size = float(cfg.get("tiles", {}).get("size", 2.0))
        self.min_points_per_child = int(al.get("min_points_per_child", 4))
        self.braking_envelope_m = float(al.get("braking_envelope", 30.0))
        self.collision_band = tuple(al.get("collision_band", [0.3, 2.5]))
        self.enforce_continuity = bool(al.get("enforce_continuity", True))
        # Two pin tiers. A pin guarantees "resolved finely enough to see the
        # hazard", and that is not the same resolution for a pedestrian as for
        # a building face: pinning every wall within the braking envelope at
        # 5 cm consumes the whole budget on structure that 20 cm resolves
        # perfectly well, and then there is nothing left to protect the
        # pedestrian with.
        self.pin_level = int(al.get("pin_level", 0))            # small hazards
        self.pin_level_coarse = int(al.get("pin_level_coarse", 2))  # extended structure
        self.max_cells = int(al.get("max_cells", 1_200_000))
        self.seed = int(al.get("random_seed", 7))
        self.last_diag: Dict[str, Any] = {}

    # ────────────────────────────────────────────────────────
    def capacity(self, level) -> np.ndarray:
        """Cells a FULL tile would hold at a level: (tile_size / cell_size)^2."""
        sizes = ResolutionLevel.sizes_array()[np.clip(level, 0, COARSEST)]
        return np.round((self.tile_size / sizes) ** 2).astype(np.int64)

    def set_cost_table(self, table):
        """Install the exact (n_tiles, 5) occupied-cell table for this frame."""
        self._cost_table = table

    def cells_for_level(self, level, count=None) -> np.ndarray:
        """Cells a tile will ACTUALLY occupy at a level.

        The map only allocates cells that contain points, so a tile's cost is
        bounded by its point count, not by the tile's full capacity: 300 points
        cannot occupy more than 300 of a 5 cm tile's 1600 cells.  Costing tiles
        at full capacity was what made the budget non-binding — a pinned tile
        was charged 1600 cells for the 300 it really uses, so the budget was
        exhausted on paper while the map kept growing.
        """
        table = getattr(self, "_cost_table", None)
        lv = np.clip(np.atleast_1d(np.asarray(level, np.int64)), 0, COARSEST)
        if table is not None and len(table) == len(lv):
            return table[np.arange(len(table)), lv]
        cap = self.capacity(level)
        if count is None:
            return cap
        return np.minimum(cap, np.maximum(np.asarray(count, np.int64), 1))

    # ────────────────────────────────────────────────────────
    def allocate(
        self,
        tiles: Sequence[Tile],
        budget: float = 1.0,
        policy: str = "full",
        vehicle_max_step: float = 0.15,
        reference_cells: Optional[int] = None,
        cost_table: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return the resolution level for every tile.

        ``budget`` is a fraction of the UNIFORM 5 cm MAP'S CELL COUNT over the
        same observed area, which ``reference_cells`` supplies.  Budget 0.25
        therefore means, literally, "a quarter of the memory a uniform 5 cm map
        would need" — the unit the PS's memory claim is about, and the unit the
        equal-memory comparison in Phase 7 needs.

        When ``reference_cells`` is not supplied it is estimated from the tile
        point counts (a tile with k points occupies at most min(k, 1600)
        level-0 cells), which is close enough for the interactive slider.
        """
        n = len(tiles)
        if n == 0:
            self.last_diag = {}
            return np.zeros(0, np.int8)

        self.set_cost_table(cost_table)
        a = tile_arrays(tiles)
        v = value_terms(a)

        # ── uniform policies: the honest baselines ───────────
        if policy.startswith("uniform_"):
            lvl = {"uniform_5": 0, "uniform_10": 1, "uniform_20": 2,
                   "uniform_40": 3, "uniform_80": 4}[policy]
            out = np.full(n, lvl, np.int8)
            self._diag(out, a, np.zeros(n, bool), policy)
            return out

        base = distance_levels(np.maximum(a["range"], 1.0))

        if policy == "distance_only":
            out = base.astype(np.int8)
            self._diag(out, a, np.zeros(n, bool), policy)
            return out

        # ── value function, progressively enabled ────────────
        if policy == "distance_geometry":
            V = v["G"]
        elif policy == "distance_geometry_semantic":
            V = (self.wg * v["G"] + self.ws * v["S"]) / max(self.wg + self.ws, 1e-6)
        elif policy == "distance_geometry_semantic_uncertainty":
            w = self.wg + self.ws + self.wu
            V = (self.wg * v["G"] + self.ws * v["S"] + self.wu * v["U"]) / max(w, 1e-6)
        elif policy in ("full", "random"):
            V = (self.wg * v["G"] + self.ws * v["S"]
                 + self.wu * v["U"] + self.wd * v["D"])
        else:
            raise ValueError(f"unknown allocation policy: {policy!r}")
        V = np.clip(V, 0.0, 1.0).astype(np.float32)

        for i, t in enumerate(tiles):
            t.score_geometry = float(v["G"][i])
            t.score_semantic = float(v["S"][i])
            t.score_uncertainty = float(v["U"][i])
            t.score_dynamic = float(v["D"][i])
            t.info_value = float(V[i])

        # ── safety pins: only the `full` policy has them ─────
        if policy == "full":
            pinned, reasons, pin_level = self._safety_pins(tiles, a, vehicle_max_step)
        else:
            pinned = np.zeros(n, bool)
            reasons = [""] * n
            pin_level = np.full(n, COARSEST, np.int8)

        # ── cell budget ──────────────────────────────────────
        total_budget = self._cell_budget(a, budget, reference_cells)

        if policy == "random":
            out = self._random_allocate(base, total_budget, n, a["count"])
            self._diag(out, a, pinned, policy, budget=total_budget)
            return out

        nbr = self._neighbour_index(tiles) if self.enforce_continuity else None
        out = self._greedy_allocate(base, V, a, pinned, total_budget, nbr, pin_level)
        if nbr is not None:
            out = self._continuity_closure(out, nbr)

        for i, t in enumerate(tiles):
            t.resolution_level = int(out[i])
            t.safety_pinned = bool(pinned[i])
            t.safety_reason = reasons[i]
            t.selected = bool(out[i] <= 2 or pinned[i])

        self._diag(out, a, pinned, policy, budget=total_budget)
        return out

    # ────────────────────────────────────────────────────────
    def _cell_budget(self, a, budget: float,
                     reference_cells: Optional[int] = None) -> int:
        """Cells available = budget x the uniform 5 cm map's cell count."""
        if reference_cells is None:
            table = getattr(self, "_cost_table", None)
            if table is not None:
                reference_cells = int(table[:, 0].sum())
            else:
                reference_cells = int(
                    np.minimum(a["count"], (self.tile_size / 0.05) ** 2).sum())
        b = float(max(budget, 0.0))
        # The coarsest map is free: it is the floor below which the budget
        # cannot push, since every observed tile needs at least its 80 cm cells.
        floor_cost = int(self.cells_for_level(
            np.full(len(a["count"]), COARSEST), a["count"]).sum())
        return int(min(max(b * reference_cells, floor_cost), self.max_cells))

    # ────────────────────────────────────────────────────────
    def _safety_pins(self, tiles, a, vehicle_max_step):
        """Tiles that must be fine regardless of budget.

        Pins consume budget FIRST and are never ranked against it.
        """
        n = len(tiles)
        pinned = np.zeros(n, bool)
        reasons: List[List[str]] = [[] for _ in range(n)]

        lo, hi = self.collision_band

        # ── TIER 1: small, isolated hazards -> finest level ──
        # (a) THE GEOMETRIC PIN - the one the retention guarantee is stated
        #     over. Fires on "a few consecutive rings at one azimuth, standing
        #     above the ground, in a sparse neighbourhood" without knowing what
        #     the object is, so it survives a segmentation failure.
        geo = a["vertical_run"]
        for i in np.flatnonzero(geo):
            reasons[i].append("vertical run (geometric)")

        # (b) semantic VRU - an ENHANCEMENT on top of (a), not the guarantee.
        vru = a["importance"] >= 0.99
        for i in np.flatnonzero(vru):
            reasons[i].append("VRU class")

        # (c) confirmed dynamic
        dyn = a["moving"] > 0.55
        for i in np.flatnonzero(dyn):
            reasons[i].append("dynamic object")

        # ── TIER 2: extended structure -> a coarser guaranteed level ──
        # (d) obstacle in the collision band within the braking envelope.
        #     Restricted to compact returns: a 20 m building facade is an
        #     obstacle, but it is one obstacle, and 20 cm resolves it.
        band = ((a["z_spread"] > lo) & (a["z_spread"] < hi * 2.0)
                & (a["range"] < self.braking_envelope_m)
                & (a["count"] > 3) & (a["count"] < 600))
        for i in np.flatnonzero(band):
            reasons[i].append(f"obstacle in braking envelope ({a['range'][i]:.0f} m)")

        # (e) vertical step beyond the vehicle's capability (kerbs, trenches)
        step = ((a["roughness"] > vehicle_max_step) & (a["count"] > 6)
                & (a["range"] < self.braking_envelope_m * 1.5))
        for i in np.flatnonzero(step):
            reasons[i].append(f"step {a['roughness'][i]:.2f} m > {vehicle_max_step:.2f} m")

        # (f) boundary between UNKNOWN and drivable space
        sparse = (a["valid_px"] > 0) & (a["valid_px"] < 6) & (a["range"] < 50)
        for i in np.flatnonzero(sparse):
            reasons[i].append("unknown/drivable boundary")

        fine = geo | vru | dyn
        coarse = (band | step | sparse) & ~fine
        pinned = fine | coarse
        pin_level = np.where(fine, self.pin_level,
                             np.where(coarse, self.pin_level_coarse, COARSEST)
                             ).astype(np.int8)
        return pinned, ["; ".join(r) for r in reasons], pin_level

    # ────────────────────────────────────────────────────────
    @staticmethod
    def _neighbour_index(tiles) -> np.ndarray:
        """(n, 4) index of each tile's 4-neighbours, -1 where absent."""
        from adaptive_lidar.utils.grouping import pack_keys_2d
        ix = np.array([t.ix for t in tiles], np.int64)
        iy = np.array([t.iy for t in tiles], np.int64)
        keys = pack_keys_2d(ix, iy)
        order = np.argsort(keys)
        skeys = keys[order]
        out = np.full((len(tiles), 4), -1, np.int64)
        for k, (dx, dy) in enumerate(((1, 0), (-1, 0), (0, 1), (0, -1))):
            nb = pack_keys_2d(ix + dx, iy + dy)
            pos = np.clip(np.searchsorted(skeys, nb), 0, len(skeys) - 1)
            hit = skeys[pos] == nb
            out[:, k] = np.where(hit, order[pos], -1)
        return out

    def _continuity_closure(self, level: np.ndarray, nbr: np.ndarray) -> np.ndarray:
        """Smallest refinement of ``level`` satisfying 2:1 continuity.

        Continuity means adjacent tiles differ by AT MOST ONE level, i.e.
        ``level[i] <= level[j] + 1`` for every neighbour pair.  The closure only
        ever makes tiles *finer*, and it is what prevents a 5 cm tile abutting
        an 80 cm tile — a seam across which the ground-height gradient is
        meaningless, which is the "alignment error" artefact at map scale.
        """
        lvl = level.astype(np.int16).copy()
        for _ in range(COARSEST + 1):
            cap = np.full(len(lvl), COARSEST, np.int16)
            for k in range(nbr.shape[1]):
                j = nbr[:, k]
                has = j >= 0
                cap = np.where(has, np.minimum(cap, lvl[np.clip(j, 0, None)] + 1), cap)
            newl = np.minimum(lvl, cap)
            if np.array_equal(newl, lvl):
                break
            lvl = newl
        return np.clip(lvl, 0, COARSEST).astype(np.int8)

    def _greedy_allocate(self, base, V, a, pinned, cell_budget, nbr=None,
                         pin_level=None) -> np.ndarray:
        """Pins first (plus their mandatory continuity closure), then greedy
        refinement by value per unit cost."""
        n = len(base)
        counts = np.asarray(a["count"], np.float64).copy()
        cur = np.full(n, COARSEST, np.int8)

        # 1. Pins consume budget FIRST and are never ranked against it.
        if pin_level is None:
            cur[pinned] = self.pin_level
        else:
            cur[pinned] = pin_level[pinned]
        if nbr is not None and self.enforce_continuity:
            # The closure around a pin is mandatory, so it is charged too.
            cur = self._continuity_closure(cur, nbr)
        # Points redistribute as a tile is refined, so the effective count per
        # tile at its current level is what bounds its cost.
        spent = int(self.cells_for_level(cur, np.maximum(counts, 1)).sum())
        self.last_diag["pin_cells"] = spent

        # 2. Unpinned tiles refine from the coarsest level by rho.
        #    The distance schedule caps how fine a tile may go: refining past it
        #    manufactures cells with fewer points than they can support.
        cap = np.minimum(base, COARSEST).astype(np.int8)

        import heapq
        heap = []
        for i in np.flatnonzero(~pinned):
            if cur[i] > cap[i]:
                r = self._rho(V[i], cur[i], counts[i], int(i))
                if r > 0:
                    heapq.heappush(heap, (-r, int(i)))

        while heap and spent < cell_budget:
            _, i = heapq.heappop(heap)
            if cur[i] <= cap[i]:
                continue
            # 5.5 REFINEMENT GATE - never split unless each child can hold
            # enough points. Otherwise "high uncertainty -> refine" at 90 m
            # manufactures empty fine cells, burning budget while *raising*
            # per-cell uncertainty.
            if counts[i] / 4.0 < self.min_points_per_child:
                continue
            # 5.3 CONTINUITY - reject a refinement that would open a >1 level
            # seam, rather than cascading into neighbours we did not choose.
            if nbr is not None and self.enforce_continuity:
                j = nbr[i][nbr[i] >= 0]
                if j.size and np.any(cur[j] > cur[i]):
                    continue
            delta = self._delta_cells_tile(int(i), int(cur[i]), counts[i])
            if spent + delta > cell_budget:
                continue
            cur[i] -= 1
            spent += delta
            counts[i] = counts[i] / 4.0
            if cur[i] > cap[i]:
                r = self._rho(V[i], cur[i], counts[i], int(i))
                if r > 0:
                    heapq.heappush(heap, (-r, i))

        self.last_diag["cells_spent"] = spent
        return cur

    def _rho(self, v: float, level: int, count: float, i: int = -1) -> float:
        if level <= 0:
            return 0.0
        d = (self._delta_cells_tile(i, int(level), count) if i >= 0
             else self._delta_cells(level, count))
        return float(v) / float(d)

    def _delta_cells_tile(self, i: int, level: int, count: float) -> int:
        """Marginal cells from refining tile ``i`` one level — exact when the
        cost table is available."""
        table = getattr(self, "_cost_table", None)
        if table is not None and 0 <= i < len(table) and level >= 1:
            return max(int(table[i, level - 1] - table[i, level]), 1)
        return self._delta_cells(level, count)

    def _delta_cells(self, level: int, count: float = None) -> int:
        """Extra cells from refining a tile one level.

        Capacity quadruples, but the tile can never hold more cells than it has
        points, so the real marginal cost saturates once the tile is already
        finer than its own point density. That saturation is what stops the
        controller from paying four times over for a refinement that splits
        nothing.
        """
        here = int(round((self.tile_size / ResolutionLevel.size(level)) ** 2))
        finer = int(round((self.tile_size / ResolutionLevel.size(level - 1)) ** 2))
        if count is not None:
            c = max(int(count), 1)
            here, finer = min(here, c), min(finer, c)
        return max(finer - here, 1)

    # ────────────────────────────────────────────────────────
    def _random_allocate(self, base, cell_budget, n, counts=None) -> np.ndarray:
        """CONTROL policy: same budget, levels assigned at random.

        If `full` does not beat this at equal budget, the value function is not
        doing anything and the whole contribution is illusory.
        """
        rng = np.random.default_rng(self.seed)
        level = np.full(n, COARSEST, np.int8)
        counts = np.ones(n) if counts is None else np.asarray(counts, float).copy()
        spent = int(self.cells_for_level(level, counts).sum())
        order = rng.permutation(n)
        for i in order:
            while level[i] > 0:
                delta = self._delta_cells(level[i], counts[i])
                if spent + delta > cell_budget:
                    break
                level[i] -= 1
                spent += delta
                counts[i] = counts[i] / 4.0
            if spent >= cell_budget:
                break
        return level

    # ────────────────────────────────────────────────────────
    # ────────────────────────────────────────────────────────
    def _diag(self, level, a, pinned, policy, budget=None):
        cells = self.cells_for_level(level, a["count"])
        self.last_diag = {
            "policy": policy,
            "n_tiles": int(len(level)),
            "cells_total": int(cells.sum()),
            "cell_budget": int(budget) if budget is not None else None,
            "n_pinned": int(pinned.sum()),
            "pin_cells": self.last_diag.get("pin_cells", 0),
            "budget_exceeded_by_pins": bool(
                budget is not None and self.last_diag.get("pin_cells", 0) > budget),
            "level_hist": np.bincount(
                np.clip(level, 0, COARSEST), minlength=N_LEVELS).tolist(),
            "mean_level": float(np.mean(level)),
        }


def levels_to_points(tiles: Sequence[Tile], levels: np.ndarray,
                     tile_of_point: np.ndarray, n_points: int) -> np.ndarray:
    """Broadcast the per-tile level onto every point. One gather, no loop.

    THIS IS THE CONNECTION THAT WAS MISSING (BROKEN 1): the level the
    controller decided now travels with the point into the map.
    """
    out = np.full(n_points, COARSEST, np.int8)
    if len(levels) == 0:
        return out
    ok = tile_of_point >= 0
    out[ok] = levels[tile_of_point[ok]]
    return out
