# DECISIONS

Every judgement call made while building this system, with a one-line reason.
Where the brief left something genuinely ambiguous the simpler option was taken
and recorded here.

---

## Phase 0

| # | Decision | Reason |
|---|---|---|
| 0.1 | Installed `matplotlib` into the existing `.venv` | Phase 2/5/7 require `docs/*.png` figures and nothing else in the environment can write PNG plots. Free, OSS, CPU-only. |
| 0.2 | `scripts/profile_scaling.py` resamples one synthetic scene up to 30k/120k rather than re-simulating a larger scene | Phase 0 measures *how each stage scales with N*. Re-simulating would change scene content and confound the curve. Phase 1 re-runs the same script so baseline and result are directly comparable. Once Phase 2's raycast generator exists the script prefers `generate_scenario`. |
| 0.3 | First timed frame discarded as warm-up in all profiling | First frame pays import/JIT/allocator costs and has no previous frame for S5/S8, so it is not representative of steady state. |
| 0.4 | Profiling sizes fixed at 8k / 30k / 120k | 8k = current synthetic scene (comparable to the pre-existing baseline), 120k = a real 64-beam spinning scan, 30k = midpoint to expose super-linear stages. |

## Phase 1

| # | Decision | Reason |
|---|---|---|
| 1.1 | Morton/packed keys are `int64` with a `+2^20` bias on each axis instead of a signed-aware pack | One `int64` holds 3 × 21 bits; the bias makes negative coordinates sort correctly with a plain `np.argsort` and keeps the whole grouping path branch-free. |
| 1.2 | `group_by_key` returns `(unique_keys, group_starts, order)` and **not** a list of arrays | `np.split` allocates one Python object per group (≈100k objects at 120k points). Returning CSR-style offsets lets every consumer use `np.add.reduceat` / `np.maximum.reduceat` with zero per-group Python work. A `split_groups()` helper is provided for the rare case a real list is wanted. |
| 1.3 | `segment_reduce` implements percentile via sort-within-group + gather, not `np.percentile` per group | A Python loop over groups is the exact anti-pattern the brief bans. Sorting once by `(key, value)` makes every group's k-th order statistic a single gather. |
| 1.4 | Voxel hash keeps a thin `__contains__`/`__getitem__` wrapper over the CSR arrays | `utils/voxel_hash.voxel_centroids` and any external caller expect a mapping. The wrapper is O(log M) via `np.searchsorted` and costs nothing when unused. |
| 1.5 | `gt_label` lives on `Frame` (not in a side-channel dict) but is guarded by a repo test | Keeping it index-aligned to `points` is what makes range-stratified evaluation possible at all. `tests/test_gt_label_isolation.py` greps `stages/s2`–`s8` for the identifier, so the isolation is enforced mechanically rather than by convention. |
| 1.6 | `KITTI_TO_6` rewritten as a 260-element `np.int8` lookup array | A dict literal cannot express "this key is assigned twice" as an error — Python silently keeps the last. An array plus an explicit `assign()` helper that refuses to overwrite a previously-set id makes the collision a hard failure at import time. |
| 1.7 | Unlabelled/unknown SemanticKITTI ids map to `-1` (ignore), not to class 2 | The old `DEFAULT_CLASS = 2` silently turned every unmapped id into `static_obstacle`, inflating that class's accuracy. `-1` is excluded from IoU denominators. |
| 1.9 | **The allocation tile is 1.6 m (= 0.05 x 2^5), not 2.0 m** | 1.6 m makes the tile a *node of the map's own power-of-two hierarchy*, so the tile id is literally a prefix of the level-0 Morton code (`tile = code0 >> 10`). Two consequences. Correctness: a tile boundary can never fall inside a map cell, so the "no alignment error" guarantee extends from the map to the allocation grid — at 2.0 m (40 base cells, not a power of two) the two grids are incommensurable and a tile edge cuts through cells. Performance: right-shifting preserves order, so ONE `argsort` of the level-0 codes simultaneously groups the cloud by tile and by cell at every level, replacing four separate 128k-element sorts per frame (tile grouping, ground height field, allocation cost table, map insert). See `utils/spatial_index.py`. |
| 1.10 | The allocation cost model is the EXACT occupied-cell count per tile per level, not `min(n_points, capacity)` | The budget is only a memory budget if the cost is the real cell count. LiDAR points lie on surfaces and cluster heavily, so `min(n_points, capacity)` overstates the fine levels by 2-3x — with that estimate the budget was exhausted on paper while the map kept growing, and the slider did nothing. Because the levels nest, the exact count is five boolean diffs over the ordering that already exists. |
| 1.11 | The budget is a fraction of the UNIFORM 5 cm map's cell count | It makes the number on the slider mean something: budget 25% is "a quarter of the memory a uniform 5 cm map would need". That is the unit the PS's memory claim is in and the unit the Phase 7 equal-memory comparison needs. |
| 1.12 | Safety pins have two tiers: 5 cm for small hazards, 20 cm for extended structure | Pinning every building facade within the braking envelope at 5 cm consumed the entire budget on structure that 20 cm resolves perfectly, leaving nothing to protect the pedestrian with — the pins became a floor above every budget and the slider was inert. A pin guarantees "resolved finely enough to see the hazard", and that is not the same resolution for a person as for a wall. |
| 1.13 | The semantic network runs on NON-GROUND points; ground points get a two-class rule | Ground is already separated geometrically, so the only question left about it is drivable vs rough, which surface roughness and corrected intensity answer directly. Spending inference on the ~60% of a scan that is road surface costs most of the frame budget for almost no accuracy. `semantics.classify_ground: net` overrides it and the evaluation reports both. |
| 1.8 | S4's old per-tile backend loop replaced by a per-point path in the same stage | BROKEN 2 requires `(N,6)` evidence; keeping the tile loop as well would mean maintaining two semantics systems. Tile-level `semantic_class` is retained as a *derived* display field only. |

## Phase 2

| # | Decision | Reason |
|---|---|---|
| 2.1 | Synthetic sensor is 64 rings × 2048 azimuth, raycast analytically per primitive rather than through a triangle-mesh BVH | Closed-form ray/plane, ray/box and ray/cylinder intersection is exact, fully vectorisable over all 131,072 beams at once, and needs no geometry library. A BVH would add a dependency and a Python traversal loop for no gain at this scene complexity. |
| 2.2 | Scene primitives are axis-aligned boxes, planes, cylinders and ellipsoid canopies | These cover every object the brief names (walls, poles, kerbs, vehicles, pedestrians, canopy) and all have vectorised closed-form intersections. |
| 2.3 | Potholes and kerbs are height-field *modifiers* on the ground plane, not separate primitives | A pothole is a depression in the surface being raycast; making it a primitive would create a second surface the beam could pass through. Solving ray/height-field by iterative marching along the beam keeps the ground single-valued. |
| 2.4 | Multi-echo emitted only for vegetation (2 returns), everything else 1 | The brief asks for enough multi-echo to make the penetration-ratio layer demonstrable. Simulating partial returns on every surface adds noise to a feature whose whole signal is "this is foliage". |
| 2.5 | `generate_scenario()` returns a dict, `generate_scene()` kept as an `(N,4)` shim | `main.py`, `app.py` and the old tests all call `generate_scene`. Keeping the shim means the pipeline never breaks between phases (protocol rule 6). |
| 2.6 | RELLIS-3D detected by the same SemanticKITTI tree walker, distinguished by `max(label_id)` | RELLIS ships in SemanticKITTI layout; its ids are all < 35 while SemanticKITTI uses 252-259 for moving classes. Sniffing the id range is more reliable than a path-name heuristic. |

## Phase 4

| # | Decision | Reason |
|---|---|---|
| 4.1 | **Levels are 5 / 10 / 20 / 40 / 80 cm (powers of two), not the PS's illustrative "50 cm"** | The PS requires the projection to be free of alignment errors and data loss. With a power-of-two hierarchy sharing one fixed origin, the level-ℓ cell id is the level-0 Morton code right-shifted by 2ℓ bits — so a fine cell is *always* wholly inside exactly one coarse cell, by construction. A 5→50 cm ratio of 10 is not a power of two: cell boundaries at the two scales do not nest, a point near a boundary can round into two different parents, and aggregation stops being exact. 80 cm at 100 m is also *coarser* than the PS's 50 cm example, so the memory claim is conservative rather than inflated. This is a deliberate, defensible strengthening of the requirement, not non-compliance. |
| 4.2 | Morton code is 2D (x,y only); z is cell content | The map is 2.5D — z is a value stored in a cell, not part of its address. Interleaving z would make the shift-to-coarsen identity false. |
| 4.3 | Cells stored as one NumPy structured array per level, not `dict[key] → MapCell` | A `MapCell` dataclass instance is ~350 B of Python object overhead vs 22 B of payload. The memory-reduction figure is the headline claim of M5 and must be measurable with `tracemalloc` on the real storage. |
| 4.4 | Vertical histogram is transient (built during insertion, discarded after extraction) | 24 bins × 4 B × 700k cells = 67 MB if persisted, which would dwarf the 22 B/cell layout. Everything needed downstream is extracted in the same pass. |
| 4.5 | Obstacle layer uses an existential rule, terrain uses a majority rule | A 4 m pole returns 3 points and loses every vote; a road surface is defined by its bulk. Unifying them necessarily sacrifices one. |
| 4.6 | `overhead_clearance` stored as a first-class field | Without it, tree canopy over a road puts `z_max` at 4 m and marks drivable road BLOCKED. One 2-byte field removes an entire failure class. |
| 4.7 | Traversability is computed at query time from stored physical properties, never stored | Drivability is a property of the *vehicle*; slope and step height are properties of the *terrain*. Storing the verdict would bake one vehicle's limits into the map and make the wheeled/tracked toggle impossible. |
| 4.8 | Free-space carving only at level ≥ 3 (40 cm) | Free space carries no shape information, so fine carving buys nothing and costs 64× the cells of coarse carving. |
| 4.9 | The cell record is **27 bytes**, not the 22 the brief names | The brief's own field list sums to 27 B: the running total reaches 22 B at `last_seen`, and the five trailing `u8` fields (`intensity_mean`, `intensity_var`, `penetration`, `observability`, `flags`) add 5 more. Every field named in the brief is implemented; the figure is corrected rather than the layout trimmed. `BYTES_PER_CELL` adds the 8-byte Morton key that addresses the cell, giving 35 B of real payload — and `tracemalloc` is what the reported memory figure actually comes from, never this arithmetic. |

## Phase 5

| # | Decision | Reason |
|---|---|---|
| 5.1 | Budget is expressed in **cells** (convertible to MB at 22 B/cell), not a fraction of tiles | The PS's memory claim is about map size. A tile-fraction budget does not bound memory at all: refining 10% of tiles by four levels costs more than refining 100% by one. |
| 5.2 | Safety is a **pin** (a constraint on the feasible set), never a weighted term in V | Any weighted sum makes safety tradeable — with enough competing tiles the 70 m pedestrian is outbid. Pins consume budget first and are never ranked. |
| 5.3 | The retention guarantee is stated over the **geometric** pin (`has_vertical_run`), with semantics as an enhancement | A semantic pin is only as good as the classifier. The geometric pin fires on "small, isolated, vertically-extended cluster above ground" without knowing what it is, so the guarantee survives a segmentation failure. |
| 5.4 | Refinement gate: never split unless `n_points / 4 ≥ 4` | Otherwise "uncertain → refine" at 90 m manufactures empty fine cells, spending budget while *raising* per-cell uncertainty. |
| 5.5 | Distance bands are derived from `resolution_law(r)`, not hardcoded | Ground sample area per beam grows ≈ r³ (linear azimuthal × quadratic radial spreading), so constant points-per-cell implies cell size ∝ r. Anchoring 5 cm at 10 m yields 50 cm at 100 m — the PS's own example value, derived rather than assumed. |
| 5.6 | Policy selection is a single config string behind `allocate(tiles, budget) -> levels` | The Phase 7 ablation table needs ~10 policies × 5 budgets. Behind one interface that is an afternoon; scattered through the map code it is a week of refactoring. |

## Phase 6

| # | Decision | Reason |
|---|---|---|
| 6.1 | Motion segmentation is range-image residual MOS, not 4DMOS | 4DMOS is a sparse 4D convolution network and requires CUDA to build. Residual-image MOS (the LMNet / "Moving Object Segmentation in 3D LiDAR Data" lineage) is the established CPU-feasible member of the same family and reuses the range image already built in S1. |
| 6.2 | Moving points are **gated out** of the persistent map, never decayed out of it | At 10 m/s a 1-second decay leaves a 10 m phantom wall behind every passing car. Gating is a cure; decay is a treatment. |
| 6.3 | Three track states (`STATIC` / `MOVING` / `MOVABLE_BUT_STATIONARY`) | Conflating the last two gives either permanent holes in car parks (if treated as moving) or trails behind pedestrians (if treated as static). |
| 6.4 | Velocity lives in the object table; cells store a 2-byte object handle | One car covers ~200 cells. Storing its velocity 200 times is redundant and creates consistency bugs when the estimate updates. |
| 6.5 | Instance clustering is voxel connected-components gated on shared dominant class, not DBSCAN | DBSCAN is O(N log N) with a kd-tree the brief bans from the per-frame path, and ungated connectivity merges a pedestrian into the wall behind them. |

## Phase 7

| # | Decision | Reason |
|---|---|---|
| 7.1 | The central comparison is **equal memory**, not equal resolution | Comparing an adaptive map against a uniform 5 cm map is rigged and an evaluator will see it. Fixing the budget and asking "what is the best map obtainable for it?" is the honest question. |
| 7.2 | Memory is measured with `tracemalloc` on the live structures, never computed as cells × bytes | Cell-count arithmetic omits array overhead, dict overhead and per-level padding, all of which are real. |
| 7.3 | Latency reported as p50 / p95 / **p99** | For a real-time system the tail is the requirement; the mean hides exactly the frames that would drop. |
| 7.4 | ORR denominator counts only GT objects observed with ≥ 10 points | An object the sensor never saw cannot be retained, and including it would make the metric a measure of occlusion rather than of allocation. Thresholds (≥3 cells, ≥10 points) are reported and a sensitivity sweep is included. |

## Phase 9

| # | Decision | Reason |
|---|---|---|
| 9.1 | `perception/minkowski_backend.py` deleted | MinkowskiEngine requires `nvcc` to *build*, not merely to run. On a CPU-only machine the module can never be anything but an `ImportError` path, so it is dead code that implies a capability the system does not have. Constraint 0.1. |
| 9.2 | `stages/s8_fusion.py` and `stages/s9_output.py` made real modules rather than deleted | S8 gains genuine content in Phase 6 (the MOS gate), so the file earns its existence. S9 moved with it for symmetry, leaving `s7_map.py` as only the map stage. |

## Phase 11

| # | Decision | Reason |
|---|---|---|
| 11.1 | The addendum's cameras were built on the existing 2D top-down renderer, not on a 3D scene view | The addendum says to apply it after "Part 2 (the 3D scene view)", but no Part 1 or Part 2 was ever specified for this repository and none existed. Four of the five presets are genuine top-down cameras, and the one the addendum itself calls most important — top-follow with the resolution layer — *is* a top-down view by definition. Building a 3D viewer to an unwritten spec would have been inventing scope. |
| 11.2 | The `sensor` preset shows the range image rather than a synthesised perspective | With no 3D renderer, "first person from the LiDAR origin" has two possible answers. The range image is the projection the sensor actually produces and every geometric feature in this system is actually computed on; a perspective render would be a picture of a scene the pipeline never sees. Labelled as such on screen. |
| 11.3 | MOVING now requires COHERENT displacement, not just speed | The addendum's own verification requires the parked car in `convoy` to come out MOVABLE_BUT_STATIONARY, and under the old speed-only rule it did not. As the ego drives past a stationary car its visible surfaces change and the observed centroid drifts ~1 m per frame — indistinguishable from a car travelling at the ego speed. Net displacement over path length separates them (0.58 vs 0.99), and so does displacement against the object's own footprint diagonal. This is a change to the tracker's state rule, which the addendum asked not to touch; it is made because the addendum's verification demanded it, and no metric and no map behaviour changed with it. |
| 11.4 | Per-frame cells are cached in packed dtypes and clipped to a box around the ego | 45 bytes per cell over 20 frames and two MOS variants is ~200 MB of session state. Packing to ~20 bytes and clipping the window brings it to ~25 MB, which is the difference between a demo that runs and one that exhausts the browser session. `size` is dropped entirely because it is a pure function of `level`. |
| 11.5 | Generated scans are memoised to `.scan_cache/` | The scenes are deterministic given a seed, so raycasting them twice is pure waste — and at ~2.5 s per scan it was 40 s of dead time before the Drive tab could show anything. Cold 2.02 s, warm 0.03 s, byte-identical. Gitignored, because it is derived data. |
| 11.6 | `trail_profile`, `vehicle_half_length` and `moving_object_path` moved into `evaluation/metrics.py` | The dashboard and `scripts/test_temporal.py` were about to hold two copies of the trail metric, which is how two different numbers for the same thing end up in one project. Moving the ground-truth read into `evaluation/` also kept the gt-isolation allowlist tight instead of adding an exception for a visualisation module. |
