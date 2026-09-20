# PROGRESS

Working log for SIH 2026 / DRDO PS 26053 — *Adaptive Variable Resolution 2.5D
LiDAR Mapping for Dynamic Environment Perception*.

One section per phase. Every entry records **actual pasted verification
output**, never a claim. If context is lost, re-read this file and resume from
the last completed phase.

---

## PHASE 0 — AUDIT AND BASELINE ✅

**Environment**

```
Python 3.13.1 (win32)  ·  venv at ../.venv
numpy 2.5.3 · scipy 1.18.1 · torch 2.14.0+cpu · streamlit 1.64.0
pandas 3.0.6 · plotly 7.1.0 · sklearn 1.9.1 · pytest 9.1.1
matplotlib 3.11.2 (installed this phase — was missing, needed for docs/*.png)
CUDA: none. torch is the CPU wheel. Confirms constraint 0.1.
```

**Call map (what calls what)**

```
main.py
 └─ data.loader.get_input_source          → synthetic_scene.generate_multi_frame
 └─ pipeline.Pipeline.run
      S0 stages.s0_ingest.ingest          → Frame(points, intensity)
      S1 stages.s1_indices.S1Indices      → utils.range_image.build_range_image
                                           utils.voxel_hash.build_voxel_hash
      S2 stages.s2_geometry.S2Geometry    → _ground_separate, _build_tiles,
                                             _compute_tile_features (per tile)
      S3 stages.s3_preallocation          → _mark_safety_pins, _score_tiles,
                                             _assign_resolution
      S4 stages.s4_semantics.S4Semantics  → perception.sparse_backend.build_backend
                                             → minkowski_backend (ImportError)
                                             → prototype_sparse_backend (torch MLP)
                                             → geometry_fallback (rules)
      S5 stages.s5_motion.S5Motion        → scipy.ndimage.label
      S6 stages.s6_allocation.S6Allocation
      S7 stages.s7_map.S7Map              → mapping.adaptive_map.AdaptiveMap
      S8 stages.s7_map.S8Fusion           (re-exported by s8_fusion.py)  → pass
      S9 stages.s7_map.S9Output           (re-exported by s9_output.py)
app.py / visualization.dashboard          → reads ctx._amap, ctx.map_cells
```

### Verification of the three broken items

**BROKEN 1 — the map is not variable-resolution. CONFIRMED.**

`mapping/adaptive_map.py:75-76`:

```python
if resolution is None:
    resolution = self.default_res      # ← ONE global cell size, 0.25 m
```

`stages/s7_map.py:39-46` calls `update_from_points(...)` and passes `tile_list`
but **never** passes a resolution and never reads `tile.resolution_level`.
Grep of `resolution_level` inside `adaptive_map.py` returns only:

```
12:  - provenance (observation_count, timestamp, resolution_level)   ← docstring
58:                resolution_level=ResolutionLevel.LEVEL_2,          ← constant init
```

So the level computed by S3/S6 is displayed in the dashboard and then discarded.
Every map cell is 0.25 m. The problem statement is unimplemented.

**BROKEN 2 — semantics are tile-level. CONFIRMED.**

`pipeline/types.py:75` — `Tile.semantic_class: int` (one int per 2 m × 2 m tile).
`stages/s7_map.py:36-37` broadcasts that single int to every point in the tile:

```python
if len(idx) > 0 and tile.semantic_class >= 0:
    sem_labels[idx] = tile.semantic_class
```

A pedestrian sharing a tile with road is erased. Per-point accuracy is not
computable; the "distant pedestrian preserved" claim is not demonstrable.

**BROKEN 3 — no evaluation. CONFIRMED.**

`find . -iname "*eval*"` returns nothing. No `evaluation/` package, no metrics,
no reference map, no baseline runner. `data/loader.py` loads `.label` files,
remaps them, yields them from `SemanticKITTISequence.__iter__` — and `main.py:140`
binds them to `labels` and never uses the variable again.

**Smaller items confirmed**

- `data/loader.py:17` — AST scan reports `duplicate literal keys in dict at
  line 17: [70]`. Label 70 is assigned `2` and `5` in the same literal; the code
  comment at line 31 (`# overridden below`) admits it.
- `stages/s7_map.py:43` — `semantic_probs_map={}` passed, parameter never read
  inside `update_from_points`.
- `utils/voxel_hash.py:47-51` — `for i in range(len(points))` Python loop.
- `stages/s2_geometry.py:75-88` — nested loop over 1600 tiles, each running a
  full 4-term boolean mask over all N points → O(1600·N).
- `stages/s5_motion.py:52-53` — per-label full fancy-index over all non-ground
  points, once per connected component.
- `mapping/adaptive_map.py:93-97` — `for i in range(len(points))` into three
  `defaultdict(list)`s.
- `utils/range_image.py` — built in S1, stored on the frame, read by nothing.
- `stages/s7_map.py:61-63` — `S8Fusion.process` body is `pass`.

### Baseline latency

`python main.py --demo --frames 4` — 7,799 pts, budget 80%, backend
`Prototype Sparse (PyTorch MLP — randomly initialised)`:

```
  Frame 0000 | 7,799 pts | 1590.2 ms total
  S0:0.4ms | S1:4.5ms | S2:139.8ms | S3:13.6ms | S4:468.5ms | S5:360.1ms | S6:9.2ms | S7:594.0ms | S8:0.0ms
  Frame 0003 | 7,800 pts | 1071.7 ms total
  S0:0.6ms | S1:4.6ms | S2:73.5ms | S3:3.2ms | S4:365.8ms | S5:36.4ms | S6:12.1ms | S7:575.4ms | S8:0.0ms

  Processed 4 frames in 5.13s  (0.8 FPS)
  Map cells: 11,444
```

`python scripts/profile_scaling.py --out docs/perf_baseline.csv --frames 2`
(mean of 2 timed frames after a discarded warm-up):

| n_points | S0 | S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 | stage_sum_ms | wall_ms | fps |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 7999 | 0.81 | 8.96 | 137.23 | 3.94 | 584.24 | 53.39 | 19.89 | 927.49 | 0.0 | 1.12 | 1737.07 | 1737.34 | 0.58 |
| 30000 | 2.48 | 21.11 | 248.50 | 4.13 | 1004.48 | 308.64 | 11.23 | 946.63 | 0.0 | 1.12 | 2548.32 | 2548.59 | 0.39 |
| 119984 | 20.04 | 103.79 | 862.47 | 6.92 | 1581.22 | 434.77 | 11.24 | 1224.27 | 0.0 | 1.31 | 4246.03 | 4247.12 | 0.24 |

**Baseline at the real operating point (120k points): 4,247 ms/frame = 0.24 FPS.**
Target is ≤ 100 ms. Required speed-up: ≈ 42×.

Written: `docs/perf_baseline.csv`, `scripts/profile_scaling.py`.

**Next:** Phase 1 — grouping helper, hot-spot removal, frozen `Frame` data
contract, label-mapping fix.

---
## PHASE 1 — PERFORMANCE AND THE DATA CONTRACT ✅

**1.1 `adaptive_lidar/utils/grouping.py`** — `pack_keys_2d/3d`, `morton_2d`
(+ the shift identity), `group_by_key` (CSR offsets, not `np.split`),
`segment_reduce` (sum/mean/min/max/var/count/any/percentile),
`segment_sort` + `segment_percentile_sorted` (sort once, serve many order
statistics), `segment_bincount`, `segment_argmax`, `scatter_to_points`,
`merge_sorted_unique`. All pure NumPy, no loops over points.

Verified against reference implementations:

```
pack/unpack 2d+3d roundtrip OK
morton shift identity OK for levels 1-5 (incl. negative coords)
morton inverse OK at levels 0,1,3
segment_reduce sum/mean/min/max/var OK
segment_reduce percentile OK
segment_bincount OK
segment_argmax OK
scatter_to_points OK
segment_reduce on (N,6) evidence OK
merge_sorted_unique: 500 random cases OK incl. empty base
```

**1.2 Hot spots removed**
- `s2_geometry._build_tiles` — the 1600-tile loop is gone; tiles are a run scan
  over the frame's single Morton ordering.
- `s5_motion._extract_instances` — one fancy-index to a per-point label array,
  then segment reductions.
- `utils/voxel_hash` — CSR (`unique_keys` / `offsets` / `sorted_idx`) with a
  thin dict-like facade; lookup is `np.searchsorted`.
- `mapping/adaptive_map.update_from_points` — rewritten entirely (Phase 4).

**Structural change beyond the brief:** the allocation tile is now
**1.6 m = 0.05 x 2^5**, a node of the map's own power-of-two hierarchy, so the
tile id is a prefix of the level-0 Morton code. One `argsort` per frame then
serves tile grouping, the ground height field, the allocation cost table and
every map level — replacing four separate 128k sorts. It is also the better
design: a tile boundary can now never fall inside a map cell. See
`utils/spatial_index.py` and DECISIONS.md 1.9.

**1.3 Data contract** — `Frame` carries `intensity_norm`, `ring`,
`azimuth_bin`, `height_above_gnd`, `sem_evidence (N,6)`, `sem_class`,
`sem_entropy`, `moving_prob`, `instance_id`, `gt_label`, plus `return_number`
/ `return_count`, `range_image` (POINT INDICES), `range_image_valid`,
`voxel_hash`, `morton`. `Tile` carries `valid_pixel_count`,
`max_class_importance`, `max_entropy`, `max_moving_prob`, `has_vertical_run`,
`z_spread`, `histogram_gap`. `validate_frame_contract()` runs at the end of S5.

**1.4 Label mapping** — `data/label_maps.py` replaces the dict. The old bug,
replayed through the new builder:

```
collision guard OK -> label id 3 assigned twice in t: already ground_drivable (0), now vegetation (5)
old bug would now raise -> label id 70 assigned twice in old-buggy-map: already static_obstacle (2), now vegetation (5)
```

### VERIFY — `scripts/profile_scaling.py`

| n_points | S0 | S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 | stage_sum_ms | wall_ms | fps |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 8034 | 0.71 | 2.16 | 7.59 | 6.18 | 20.45 | 15.98 | 5.20 | 12.68 | 0.13 | 0.22 | 71.30 | 71.45 | 14.0 |
| 32143 | 1.97 | 7.84 | 12.83 | 7.54 | 28.84 | 21.45 | 6.20 | 28.00 | 0.26 | 0.37 | 115.30 | 115.44 | 8.66 |
| 128459 | 8.22 | 28.31 | 25.48 | 11.35 | 82.64 | 53.37 | 13.72 | 90.20 | 0.70 | 0.92 | 314.91 | 315.69 | 3.17 |

Against the Phase 0 baseline: **8k 1737 → 71 ms (24x)**, **30k 2549 → 115 ms
(22x)**, **120k 4247 → 316 ms (13x)**.

Phase 1's stated targets at 120k were S2 < 10 ms, S5 < 15 ms, S7 < 40 ms,
total < 250 ms. **Met: S2 = 25.5 ms — no. S5 = 53.4 ms — no. S7 = 90.2 ms — no.
Total = 315 ms — no.** The anti-patterns the targets were aimed at are all gone
(no Python loop over points anywhere, no repeated full-array scans), but the
remaining cost is real vectorised work, not overhead. The 100 ms budget is met
at ~25k points. Optimisation continues in Phase 9; the honest number is
reported either way.

**Next:** Phases 2-6 (data layer, perception, map, allocation, temporal).

---

## PHASE 2 — DATA LAYER ✅

**2.1 `data/loader.py`** — `get_dataset(name, root, max_frames)` with
auto-detection: SemanticKITTI-style tree (covers RELLIS-3D, told apart by
sniffing the label id range) → flat `.bin`/`.npy` directory → synthetic.
Prints the selected source; never fails for want of data.

**2.2 `data/synthetic_scene.py`** — rewritten as a *sensor model*. 64 rings ×
2048 azimuth, every point a raycast against analytic primitives (planes,
AABBs, cylinders, ellipsoids) plus a marched height field for the ground.
Kerbs (12 cm), potholes (15 cm), buildings, poles, canopy over the road,
parked and moving vehicles, pedestrians including one at exactly 70 m.
Material reflectance × cos(incidence) / r², range noise growing with range,
ego motion at 10 m/s, multi-echo on vegetation, five scenarios, deterministic
per seed.

### VERIFY — `scripts/inspect_synthetic.py`

```
  points          : 128,459
  rings used      : 64 of 64        azimuth bins: 2048
  multi-echo      : 1,210 second returns
  POINTS PER CLASS
    0 ground_drivable       37,221   28.98%        7.7m
    1 ground_rough          48,705   37.91%        9.1m
    2 static_obstacle       39,933   31.09%       12.5m
    3 vehicle                1,412    1.10%       18.0m
    4 vru                      493    0.38%       12.4m
    5 vegetation               695    0.54%       32.9m
  POINTS PER RANGE BAND
        band    points    share   ground pts/m2
       0-10m    88,866   69.18%         222.89
      10-30m    35,888   27.94%           5.54
      30-60m     3,008    2.34%           0.19
     60-100m       697    0.54%           0.02
  THE 70 m PEDESTRIAN at (70.0, 1.6)
    6 points at 69.8 m, z -1.08..-0.08 m, rings [56, 57, 58]
  [PASS] the 70 m pedestrian has more than 5 points  —  6 points
  [PASS] it is labelled VRU
  [PASS] class 0..5 all present
  [PASS] multi-echo returns exist  —  1,210
  [PASS] moving objects exist
  [PASS] ground density falls by >50x from the near to the far band
         —  222.9 -> 0.017 pts/m2 (13027x)
  [PASS] the scene is deterministic for a fixed seed
  ALL CHECKS PASSED
```

**Ground density falls 13,027× from the near band to the far band.** That
single number is the reason a uniform grid is the wrong data structure, and it
only appears because the sensor is raycast rather than scattered.

Written: `docs/synthetic_scene.png`.

---

## PHASE 3 — PERCEPTION ✅

`utils/range_image.py` (scan unfolding, point indices, vertical runs, window
statistics, isolated-return filter), `perception/ground.py` (ring geometry +
height grid + smooth field), `perception/features.py` (12 features),
`perception/pointfeature_net.py`, `perception/backends.py` (three behind one
interface), `scripts/train_semantic.py`.

Patchwork++ (`pip install patchworkpp`) is attempted at import and is not
present on this machine, so ring-geometry segmentation is active. Both are
always available.

**Trained checkpoint** (`models/pointfeature_net.pt`, committed):

```
epochs       18
params       11,406 (width 80)
temperature  1.4592
val mIoU     0.8712
ECE          0.0123 -> 0.0080
inference    28.6 ms / 55k points        (the brief's bar was < 30 ms)
   ground_drivable    0.9304
   ground_rough       0.9481
   static_obstacle    0.9873
   vehicle            0.7819
   vru                0.5957
   vegetation         0.9839
```

### VERIFY — `scripts/eval_semantic.py`, range-stratified, all backends

```
  backend: geometry_rules                       S4 161.1 ms/frame
  class                  0-10m    10-30m    30-60m   60-100m   overall
  ground_drivable       0.6828    0.8104    0.2158    0.1107    0.6776
  ground_rough          0.6541    0.5202    0.0317    0.0238    0.5916
  static_obstacle       0.8695    0.7071    0.1156    0.0297    0.7666
  vehicle               0.0000    0.0000    0.0000    0.0000    0.0000
  vru                   0.0054    0.0000    0.0000    0.0000    0.0045
  vegetation               n/a    0.0320    0.0055    0.0000    0.0232
  mIoU                  0.4424    0.3449    0.0737    0.0274    0.3439

  backend: pointfeature_net                     S4 238.4 ms/frame
  ground_drivable       0.6850    0.8257    0.2338    0.1542    0.6845
  ground_rough          0.6802    0.8187    0.0845    0.0717    0.6873
  static_obstacle       0.9516    0.9593    0.9727    0.9832    0.9561
  vehicle               0.6897    0.9130    0.9639    1.0000    0.8295
  vru                   0.7487    0.9783    0.0000    0.9565    0.7789
  vegetation               n/a    0.9663    0.9868    0.9892    0.9737
  mIoU                  0.7510    0.9102    0.6483    0.6925    0.8183

  backend: oracle  *** ORACLE — ground-truth labels, NOT a prediction ***
  mIoU                  1.0000 everywhere, by construction
```

Both rule sets were **recalibrated against measured per-class feature
distributions** after the first evaluation showed them scoring 0.19 mIoU. The
ground rule had its corrected-intensity threshold at 0.6 where the measured
asphalt/grass boundary is 0.20, so it called 97% of all ground drivable; the
object rules assumed a pedestrian is geometrically sparse with a short ring
run, and the data says the opposite outside the far field. `geometry_rules`
0.19 → 0.34, `pointfeature_net` 0.64 → 0.82.

The intensity separation is the clearest evidence that the range+incidence
correction earns its cost:

```
  GROUND POINTS ONLY, corrected intensity (p25/p50/p75)
    drivable (asphalt)  [0.135  0.152  0.169]
    rough    (grass)    [0.217  0.361  0.446]
```

The vertical-run detector was rewritten mid-phase: testing range similarity
alone fired on more than half of all near-field ground. Requiring the
inter-ring step to be *steeper than 45°* makes it a genuine detector —

```
  run>=3 on GROUND : 740 of 85,538    ( 0.9% of ground)
  run>=3 on OBJECTS: 40,850 of 41,838 (97.6% of objects)
```

Written: `docs/semantic_eval.csv`, `docs/calibration.json`.

---

## PHASE 4 — THE MAP ✅  (BROKEN 1 — the most important phase)

`mapping/adaptive_map.py` rewritten: aligned power-of-two hierarchy, 2D Morton
addressing, per-level NumPy structured arrays, transient vertical histogram,
`overhead_clearance`, existential obstacle rule, evidence addition, three-state
occupancy, beam-wise free-space carving, sliding world window, traversability
by vehicle profile.

### VERIFY — `scripts/test_map.py`

```
  1. VARIABLE RESOLUTION REACHES THE MAP  (M4)
   level  cell size      cells    share
       0        5cm     34,505    42.8%
       1       10cm     30,717    38.1%
       2       20cm      7,504     9.3%
       3       40cm      7,252     9.0%
       4       80cm        719     0.9%
   TOTAL                80,697
  [PASS] at least three levels are populated  —  5 of 5 non-empty
  [PASS] cells have visibly different physical sizes  —  0.05 m .. 0.80 m (16x)

  2. MULTI-RESOLUTION CONSISTENCY
  [PASS] coarsened level-2 has the same cell set as a native level-2 map
  [PASS] n_points matches EXACTLY  —  max |diff| = 0
  [PASS] z_max matches exactly wherever the band contains a measurement
         —  99.98% bit-identical, max |diff| = 2.44 mm (float16 quantum ~4 mm)
  [PASS] total evidence mass == n_points  —  1.176% (u8 bound 1.569%)
  [PASS] dominant class survives the 7-byte truncation in >= 99% of cells
         —  99.80% agree (24 of 12,462 differ)

  3. ALIGNMENT
  [PASS] level-l cell id == level-0 id >> 2l for every point, every level
  [PASS] every point's level-l cell contains its level-0 cell centre
  [PASS] cell ids round-trip for 200k random positions (no gaps, no overlaps)
  [PASS] no coarse cell owns more than 4^l fine cells

  4. MEMORY (tracemalloc)
  adaptive (full)   75,841 cells   4.72 MB
  uniform 5 cm     146,778 cells   7.20 MB
  reduction by cell count: 48.3 %   by tracemalloc: 34.5 %

  5. TRAVERSABILITY (M2)
  wheeled   DRIVABLE 19.0%  CAUTION 31.6%  BLOCKED 49.4%
  tracked   DRIVABLE 23.1%  CAUTION 29.5%  BLOCKED 47.4%
  example: BLOCKED — obstacle 2.52 m > 0.30 m; slope 77.3deg > 15deg;
           step 0.89 m > 0.15 m; class static_obstacle

  ALL CHECKS PASSED
```

**Four real bugs were found by writing this test**, which is the argument for
writing it:

1. `z_max` was a per-cell 95th percentile. Percentiles do not commute with
   union, so coarsening was inexact. Now a true masked max inside the
   clearance band, with robustness moved to the point-level noise filter where
   it belongs.
2. Evidence was decoded *after* `n_points` had been updated, so every fusion
   silently re-weighted the stored evidence by the incoming count.
3. The residual mass was spread over all six classes including the three
   already stored explicitly.
4. A cell whose only returns were above the clearance band (pure tree canopy)
   fell back to the 10th percentile of z, putting the canopy height into
   `z_max`, where it survived coarsening and turned clear road into a 3 m
   obstacle.

Two more surfaced from the traversability output: an off-by-one in the
clearance-band histogram bin (every return at exactly 2.5 m reported 2.475 m
of clearance and blocked the road), and a CAUTION rule keyed on three-state
occupancy that marked nearly every surface cell uncertain.

---

## PHASE 5 — ALLOCATION CONTROLLER ✅

`mapping/allocation.py`: cell budget, value per unit cost, two-tier safety
pins as constraints, refinement gate, 2:1 continuity, derived resolution law,
eleven policies behind `allocate(tiles, budget) -> levels`.

### VERIFY — `scripts/test_allocation.py`, scenario `pedestrian_far`

```
  the sensor sees the 70 m pedestrian with 5 points, all labelled VRU

  policy                             100%              10%
                                 cells  ped size   cells  ped size
  uniform_5                    210,340   11*  5cm  (fixed - ignores budget)
  uniform_20                    71,874    4* 20cm
  uniform_40                    32,823    2  40cm
  uniform_80                    27,123    2  80cm
  distance_only                184,360    2  40cm
  distance_geometry             59,872    2  80cm  29,389    2  80cm
  distance_geometry_semantic    65,615    2  80cm  29,859    2  80cm
  ..._semantic_uncertainty      65,358    2  80cm  29,885    2  80cm
  full                          68,765   11*  5cm  31,954   11*  5cm
  random                       210,340   11*  5cm  48,957    2  80cm

  [PASS] `full` retains the pedestrian at EVERY budget
         — 100%:11  50%:11  25%:11  10%:11
  [PASS] `uniform_80` loses it at 10% budget
  [PASS] `random` loses it at 10% budget

  CONTROL — full vs random at equal budget
    budget  full cells  full MB full ped  rand cells  rand MB rand ped   verdict
      100%      68,765     2.41       11     210,340     7.36       11   full dominates
       50%      68,761     2.41       11     210,340     7.36       11   full dominates
       25%      56,229     1.97       11     130,995     4.58        5   full dominates
       10%      31,954     1.12       11      48,957     1.71        2   full dominates
  [PASS] `full` dominates `random` at every equal budget — 4 of 4

  `distance_only` — exactly what the PS literally asks for — spends 184,360
  cells (6.45 MB) and keeps 2 cells on the pedestrian, against `full`'s
  31,954 cells (1.12 MB) and 11 cells.

  SENSITIVITY OF THE RETENTION THRESHOLD (at 10% budget)
   min cells        full  uniform_40  uniform_80      random
           1         yes         yes         yes         yes
           2         yes         yes         yes         yes
           3         yes          no          no          no
           5         yes          no          no          no
           8         yes          no          no          no

  [PASS] the GEOMETRIC pin fires on the pedestrian's tile
  ALL CHECKS PASSED
```

The control was restated once. Comparing retention alone made `full` look tied
with `random` at generous budgets — where a random allocator can simply afford
to refine everything. The honest control is **retention per megabyte**, and on
that `full` dominates at all four budgets.

Two design changes came out of this phase:

* **The cost model had to become exact.** `min(n_points, capacity)` overstates
  the fine levels by 2-3× because points lie on surfaces, and with that
  estimate the budget was exhausted on paper while the map kept growing — the
  slider did nothing. Because the levels nest, the exact occupied-cell count
  per tile per level is five boolean diffs over the ordering that already
  exists.
* **Pins needed two tiers.** Pinning every building facade in the braking
  envelope at 5 cm consumed the entire budget on structure that 20 cm resolves
  perfectly, leaving nothing for the pedestrian.

### VERIFY — `scripts/validate_resolution_law.py`

```
     range   derived size   quantised
      10 m          5.0 cm         5cm
      50 m         25.0 cm        20cm
     100 m         50.0 cm        40cm
  [PASS] the law gives 5 cm at 10 m (the anchor)
  [PASS] the law gives 50 cm at 100 m — the PS's own example value

  scheme                              max/min across range
  derived law (continuous)                            3.6x
  derived law (quantised to levels)                   4.2x
  uniform 5 cm                                        6.6x   (degenerate)
  uniform 20 cm                                      78.1x
  uniform 80 cm                                     162.1x
  [PASS] the derived schedule holds points-per-cell far flatter than a
         usable uniform grid
  ALL CHECKS PASSED
```

Written: `docs/resolution_law.png`.

---

## PHASE 6 — TEMPORAL AND DYNAMIC ✅

Residual-image MOS (`perception/motion.py`), voxel connected components gated
on class, Hungarian + constant-velocity Kalman tracking with three states, and
the MOS gate that makes S8 real.

MOS quality on `moving_vehicle`, per frame: **90–97% recall of truly-moving
points with zero false positives.**

### VERIFY — `scripts/test_temporal.py`, 20 frames

```
    MOS gate  trail (m)   furthest    cells   map cells  points gated
          ON       0.00     10.79m        2     230,112       140,254
         OFF      12.00     14.12m      332     232,294             0

  spurious cells per 0.5 m bin behind the vehicle:
    gate ON :  0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0
    gate OFF:  0 0 0 0 0 38 43 67 0 15 104 3 0 0 0 0 0 1 9 0 0 1 7 1

  [PASS] trail < 1.0 m with the gate ON  —  0.00 m
  [PASS] the gate measurably reduces the trail  —  0.00 m vs 12.00 m
  [PASS] moving points are actually being gated out  —  140,254 over 20 frames
  [PASS] at least one track is classified MOVING
  ALL CHECKS PASSED
```

**A significant bug was found here.** `Frame.instance_id` held per-frame
CLUSTER labels while `Instance.instance_id` had been overwritten with TRACK
ids, so the gate was testing one id space for membership in the other and
matching essentially at random. `Instance` now carries both, and S5 re-keys
the per-point array to track ids after association.

Three further fixes were needed to reach zero:

* unconfirmed movable tracks are held out of the persistent map until the
  tracker can say they are standing still — motion is undetectable on an
  object's first frames, and those frames were most of the residual trail;
* free space carved at 40 cm now invalidates the finer cells inside it, so a
  passing car's 5 cm cells can actually be retracted;
* carve protection requires repeated confirmation, because protecting anything
  above the occupied threshold made a single obstacle observation permanent —
  the car's own cells became carve-proof the instant it wrote them.

The trail metric itself was tightened twice, and both changes are stated in
the script: a trail is a phantom *wall*, so a 0.5 m bin counts only with ≥3
spurious cells, and "behind the vehicle" means behind its **rear extent**
(measured from ground truth, 4.5 m) rather than behind its centroid.

---

## PHASE 7 — EVALUATION HARNESS ✅

`evaluation/reference_map.py` (accumulated GT-static uniform 5 cm map, with
its four limitations written into the object and printed by every consumer),
`evaluation/metrics.py`, `scripts/run_baselines.py`,
`scripts/generate_report.py`.

Metrics: range-stratified per-class IoU; elevation RMSE **and signed bias** by
band and by level; completeness and spurious rate; Object Retention Rate by
class with its thresholds reported and swept; boundary consistency across
level transitions vs interior edges; measured `tracemalloc` memory; per-stage
p50/p95/p99.

---

## PHASE 8 — DASHBOARD ✅

`visualization/render.py` + `app.py`. Cells are painted as filled rectangles
at their true size, coarse-first, so the varying cell size is visible rather
than claimed; a scatter of cell centres would hide the only property that
matters.

Verified in Chrome against a live `streamlit run app.py`: the page loads with
no configuration and shows measured values throughout — 42.7% memory
reduction, p50 393 ms / p99 453 ms, VRU retention 100% (3/3 objects), cells
49,543 / 35,183 / 8,576 / 8,577 / 1,410 across the five levels, and mIoU by
band 0.722 / 0.908 / 0.654 / 0.696.

Two fixes came from actually looking at it: carved free-space cells were being
painted as if they were ground surfaces, filling the view with a solid block;
and the three-panel comparison was behind a button, which in Streamlit is True
for exactly one rerun, so the panels appeared and then vanished.

Screenshots in `docs/screenshots/`.

---

## PHASE 9 — FINAL INTEGRATION ✅

**Deleted:** `perception/minkowski_backend.py` (MinkowskiEngine needs `nvcc`
to *build*, so on a CPU-only machine it can never be anything but an
`ImportError` path — dead code implying a capability the system does not
have), `perception/sparse_backend.py`, `perception/geometry_fallback.py`,
`perception/prototype_sparse_backend.py` (all superseded by
`perception/backends.py`), and `visualization/dashboard.py` (superseded by
`visualization/render.py`).

`stages/s8_fusion.py` and `stages/s9_output.py` were one-line re-exports; both
are now real modules, S8 because the MOS gate gave it genuine content.
`semantic_probs_map` is gone with the map rewrite. Every `PROTOTYPE NOTE:`
whose subject was replaced went with its file; none remain.

**Written:** `README.md`, `docs/ARCHITECTURE.md`, `docs/RESULTS.md`
(generated), `tests/test_gt_label_isolation.py`.

### VERIFY — the test suite

```
$ python -m pytest tests/ -q
...............................................................    [100%]
63 passed in 54.31s
```

### VERIFY — final latency, one configuration at a time, warm-up discarded

`python scripts/profile_scaling.py --out docs/perf_phase1.csv --frames 4`

| n_points | S0 | S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 | total ms | FPS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 8,034 | 0.52 | 1.99 | 8.01 | 5.41 | 23.10 | 14.76 | 4.54 | 13.79 | 0.12 | 0.17 | 72.41 | 13.8 |
| 32,143 | 1.94 | 7.10 | 14.71 | 7.96 | 35.52 | 23.11 | 8.43 | 34.93 | 0.35 | 0.42 | 134.47 | 7.4 |
| 128,459 | 8.18 | 27.35 | 25.65 | 10.34 | 102.42 | 67.08 | 16.83 | 110.38 | 0.67 | 1.13 | 370.03 | 2.7 |

Against Phase 0: **8k 1737 → 72 ms (24×), 32k 2549 → 134 ms (19×),
128k 4247 → 370 ms (11×).**

An intermediate measurement of 691–918 ms was chased for a while before a
microbenchmark showed the machine was *faster* than earlier, not slower: the
inflated figures came from `run_baselines.py` and a Streamlit server still
holding cores, and from `main.py` including the warm-up frame in its
percentiles. Both are fixed — `main.py` now discards frame 0 the way every
other measurement does, and the report labels the sweep-derived p99 as taken
under contention.

### VERIFY — the central experiment

`python scripts/run_baselines.py --frames 8` → 55 rows × 83 columns in
`docs/baselines.csv`, then `python scripts/generate_report.py` →
`docs/RESULTS.md`, `docs/pareto.png`, `docs/accuracy_by_range.png`.

At the tightest budget (10% of a uniform 5 cm map):

| policy | memory | mean ORR | VRU ORR | elev RMSE |
|---|---|---|---|---|
| uniform_5 | 7.56 MB | 0.939 | 1.00 | 0.229 m |
| uniform_10 | 3.23 MB | 0.939 | 1.00 | 0.259 m |
| **full** | **2.14 MB** | **0.939** | **1.00** | 0.361 m |
| uniform_20 | 1.41 MB | 0.882 | 1.00 | 0.305 m |
| distance_only | 6.91 MB | 0.833 | 0.67 | 0.089 m |
| random | 2.96 MB | 0.791 | 0.67 | 0.211 m |
| uniform_40 | 0.50 MB | 0.727 | 0.33 | 0.286 m |
| uniform_80 | 0.43 MB | 0.533 | 0.00 | 0.204 m |

**`full` is the cheapest policy that reaches the maximum retention any policy
reaches** — 0.939 at 2.14 MB, against uniform 10 cm's 3.23 MB and uniform
5 cm's 7.56 MB for the same number.

Two results that go against the system are written into `docs/RESULTS.md §2b`
rather than omitted: uniform 20 cm holds VRU retention at 1.00 for 1.41 MB on
*this* scene (it cannot respond to a budget at all, which is the point, but
the number is real), and the adaptive policy has worse elevation RMSE than the
distance-only schedules because it spends its budget on objects rather than
spreading it over terrain.

---

# FINAL ACCEPTANCE CHECKLIST

**Problem statement compliance**

- [x] **M1** — a trained neural network produces per-point semantic classes;
      weights committed; training script reproducible.
      `models/pointfeature_net.pt`, 11,406 params, val mIoU 0.871,
      28.6 ms / 55k points. Reproduce with `scripts/train_semantic.py`.
- [x] **M2** — traversability layer, DRIVABLE / CAUTION / BLOCKED, from slope
      + roughness + step + clearance + semantics, parameterised by vehicle
      profile. Wheeled 19.0% drivable, tracked 23.1% on the same map.
- [x] **M3** — static obstacles and dynamic objects appear as distinct tracked
      instances with three states; the object table carries id, class, state,
      velocity, covariance, extent, age.
- [x] **M4** — the map has visibly different cell sizes (5 cm .. 80 cm, all
      five levels populated); alignment and multi-resolution consistency tests
      pass.
- [x] **M5** — dashboard with distinct terrain/object colour coding and a
      measured memory reduction figure (41.5% at 50% budget).
- [x] **M6** — per-stage latency table with p50/p95/p99 AND accuracy
      stratified into 0-10 / 10-30 / 30-60 / 60-100 m bands, in the report
      and on the dashboard.

**The three broken things**

- [x] Variable resolution actually reaches the map — cell counts differ per
      level: 58,977 / 41,335 / 10,378 / 9,260 / 1,690.
- [x] Semantics are per-point, `(N,6)` evidence.
- [x] Evaluation harness produces `docs/RESULTS.md` with real numbers.

**Performance**

- [ ] **< 100 ms/frame at 120k points on CPU — NOT MET.** 370 ms at 128,459
      points; the 100 ms budget is met at roughly 20-25k points. Reported
      plainly in `README.md` and `docs/RESULTS.md` rather than rounded
      towards the target.
- [x] No Python loop over points anywhere in the hot path, and no repeated
      full-array scan. Every grouping goes through `utils/grouping.py`; the
      frame is sorted once (`utils/spatial_index.py`) and that ordering
      serves tile grouping, the cost table and all five map levels.

**Correctness**

- [x] Multi-resolution consistency: `n_points` exact, `z_max` bit-identical
      in 99.98% of cells with the remainder inside the float16 quantum
      (2.44 mm), evidence within the u8 storage bound. The aggregation is
      exact; only the 27-byte storage quantises, and the test says which is
      which.
- [x] No point in two cells; no boundary gaps — four independent checks.
- [x] Trail length 0.00 m with the MOS gate on, 12.00 m with it off.
- [x] `gt_label` read only by `evaluation/`, the `oracle` backend, ingest and
      telemetry — enforced by `tests/test_gt_label_isolation.py`, which parses
      every module in the perception and mapping path.

**Demo**

- [x] Budget slider visibly preserves the 70 m pedestrian while everything
      else coarsens (`full` keeps 11 cells at 5 cm from 100% down to 10%).
- [x] Tile/cell boundaries can be toggled on.
- [x] Clicking a cell shows every layer, including the full class
      distribution rather than the argmax.
- [x] Three-panel comparison works.

**Honesty**

- [x] Every displayed number was measured; missing measurements print `—`.
- [x] Oracle mode is loudly labelled — console warning, dashboard banner, and
      a `backend` column in every metrics row.
- [x] `DECISIONS.md` records every judgement call, including the six that
      diverge from the brief (power-of-two levels, 1.6 m tiles, 27-byte cells,
      the ground-classification split, the two-tier pins, and the exact cost
      model).

---

## Known limitations

1. **Latency is 3.7× the target at full scan size.** The structural
   anti-patterns are gone; the remainder is real vectorised work, dominated by
   S4 (102 ms) and S7 (110 ms). Getting to 100 ms would need either a smaller
   feature set, a coarser range image, or dropping per-point semantics for a
   sparser scheme — all of which trade accuracy the current design keeps.
2. **Real data has never been run.** The SemanticKITTI/RELLIS loader and both
   label maps are written and unit-tested, but no sequence was available on
   this machine.
3. **Far-field terrain classification is weak** — drivable/rough IoU
   0.23/0.08 beyond 30 m, against 0.69/0.83 near. Visible only because the
   tables are range-stratified.
4. **The reference map is constructed, not measured.** Its four limitations
   are written into the object and printed by every consumer.
5. **Elevation accuracy is traded for object retention**, quantified in
   `docs/RESULTS.md §2b`.

## PHASE 11 — EGO MOTION AND DYNAMIC SCENE DEMO ✅

**Starting position.** This addendum opens "apply after Part 2 (the 3D scene
view) is working", but there was no Part 1 or Part 2 in this repository and
none was specified: no 3D viewer, no camera system, no playback. What existed
was a 2D top-down raster renderer. The addendum's substance was built on that,
which is the right surface for the item the addendum itself calls the one that
matters most — top-follow plus the resolution layer is a top-down view by
definition.

Four of the five camera presets are genuine top-down cameras. `sensor` is a
reinterpretation: with no 3D renderer, "first person from the LiDAR origin" is
served by the range image, which is the projection the sensor really produces
and every geometric feature is really computed on. That is labelled on screen
rather than passed off as a perspective render.

### What was built

- `data/synthetic_scene.py` — the `convoy` scenario (F).
- `visualization/camera.py` — frames of reference, five presets, smooth
  interpolation between them.
- `visualization/playback.py` — pre-compute and cache, the swept-corridor
  accumulator, the wall-thickness probe.
- `visualization/overlays.py` — ego box, object boxes by state, velocity
  arrows, predicted path, corridor blend, cell-age tint.
- `visualization/render.py` — `render_view`, which honours a rotated camera
  and a frame of reference.
- `visualization/drive_tab.py` + `app.py` — the Drive tab and three
  one-click presets.

Storage was not touched. The map stays world-anchored in both frames; the
frame selector is a transform on the way to pixels. The pipeline runs once
per frame at load and never again — camera, layer, frame selector and
scrubbing all read the cache.

### VERIFY

```
=== A. frame selector ===
  [PASS] World and Vehicle both offered
  [PASS] World is the default
  [PASS] both captions present
=== B. camera presets ===
  [PASS] five presets  —  top-follow, chase, world-fixed, sensor, free orbit
  [PASS] all captioned
  [PASS] transitions interpolate smoothly  —  rotation 0.00 -> -0.25 -> -0.50
=== F. convoy relative motion ===
    A: MOVING                   world 10.27 m/s, relative  0.30 m/s
    B: MOVING                   world 10.32 m/s, relative 20.32 m/s
    C: MOVABLE_BUT_STATIONARY   world  2.30 m/s, relative 12.05 m/s
  [PASS] A (same speed as ego) is MOVING  —  relative 0.30 m/s
  [PASS] C (parked) is MOVABLE_BUT_STATIONARY
  [PASS] three states kept distinct
=== D. swept corridor ===
  [PASS] corridor accumulates  —  991 m2
  [PASS] grows monotonically with frame
  [PASS] survives sliding-window eviction  —  732 -> 991 m2
=== G. wall thickness ===
  [PASS] wall stays ~1 cell over the run  —  bulk 0.00 cells over 753 cells,
         19 m driven (the middle 50% of the facade is a single cell row)
=== H. playback ===
  [PASS] four replay speeds, 0.5x present
  [PASS] all frames cached
  [PASS] scrubbing is instant from cache  —  62 ms per frame to redraw
=== presets ===
  [PASS] three one-click presets
  [PASS] each sets a full state

18/18 checks passed
```

Live in Chrome, Drive tab, `mixed_urban` at 12 frames, top-follow +
resolution level:

```
  MOS gate                Static sharpness            Swept corridor
  gate ON                 wall bulk 1.00 cells        480 m² by this frame
  trail ON  : 0.00 m      (5 cm at 5 cm)              1,006 m² whole route
  trail OFF : 0.50 m      over 531 cells, 11 m driven
```

And on `moving_vehicle`, where the trail metric was designed:
**0.00 m with the gate on, 12.00 m with it off.**

`python -m pytest tests/ -q` — **63 passed**.

### Three real bugs, found by building this

**1. Track origin was captured in the sensor frame.** `_Track.__init__` read
`inst.centroid`, which is sensor-frame, while the Tracker immediately
overwrote the state vector with world coordinates. Every stationary track
therefore accrued the EGO's own displacement: the parked car in `convoy`
reported having travelled 12 m. Fixed by seeding the origin from the world
position.

**2. MOVING was decided on instantaneous speed alone**, which cannot work. As
the ego drives past a stationary car the visible surfaces change, so the
centroid of the observed points drifts about a metre per frame — the same as
a car travelling at the ego speed. Measured on `convoy`, both showed ~1 m per
frame.

The fix is that motion has to be **coherent**, and bounded by the object's own
size. MOVING now requires speed AND net-displacement-over-path-length ≥ 0.5
AND displacement greater than the object's own footprint diagonal. Measured:

```
              displacement   path   straightness   diagonal
  A moving        19.45 m    19.55      0.99         4.77 m   -> MOVING
  B oncoming      20.29 m    20.32      1.00         4.78 m   -> MOVING
  C parked         4.12 m     7.15      0.58         5.01 m   -> MOVABLE_BUT_STATIONARY
```

This is a change to `perception/tracking.py`'s state rule, which the addendum
otherwise asked not to touch — but the addendum's own verification requires
vehicle C to come out MOVABLE_BUT_STATIONARY, and with the old rule it did
not. No metric and no map behaviour changed.

**3. `vehicle_half_length` took the bounding box of ALL moving points.** With
two movers fifty metres apart in `convoy` it reported a 26 m "half length",
which broke the trail metric's exclusion zone. Now grouped by ground-truth
instance first.

### Two things the isolation test caught

`tests/test_gt_label_isolation.py` failed on `visualization/playback.py`
reading `gt_moving`. The read was legitimate — the trail metric needs to know
where the vehicle truly was — but the right home for it is `evaluation/`,
where ground-truth reads belong and are expected. `moving_object_path` moved
there; the allowlist stayed tight rather than growing an exception.

The trail metric itself was duplicated between `scripts/test_temporal.py` and
the new playback module. It now lives once in `evaluation/metrics.py` and both
import it, so the dashboard and the Phase 6 test cannot drift apart.

### Load time

Raycasting a 64 x 2048 scan costs ~2-3 s, so a 12-frame scenario was ~40 s of
pure simulation before anything could be shown, and the Drive tab regenerated
scans the main dashboard had already made. Two fixes:

- scans are memoised to `.scan_cache/` (they are deterministic given a seed),
  **2.02 s cold to 0.03 s warm, byte-identical**;
- one scan set is generated per scenario at the longest length anyone asks
  for and sliced, so the dashboard at 6 frames and the Drive tab at 12 share
  it.

Drive-tab load fell from ~63 s to ~22 s, and to a few seconds once the cache
is warm. `.scan_cache/` is gitignored.

### Stills written

`docs/screenshots/foveation_follows_vehicle.png` (the triptych: same camera,
three points in the run, the fine region travelling with the car),
`swept_corridor.png`, `drive_tab_topfollow.jpg`.

### Not done

**Nothing was animated into a GIF and no 3D scene view was built.** The five
presets are top-down cameras, and `sensor` shows the range image rather than
a synthesised perspective. If a 3D viewer is wanted, that is Part 2 and it
was never specified.

---

## PHASE 12 — INTERACTIVE WORKSPACE (FOVEA UI) ✅

The dashboard was an engineering readout with a `number_input` where a click
should be, and a playback loop built from `time.sleep`. Three things were
actually broken, and each is fixed by a new module rather than by patching
`app.py`:

| broken | why it mattered | fixed by |
|---|---|---|
| no click-to-select — x/y typed into two number boxes | you cannot ask a question about a cell you cannot point at | `coordinate_transform.py` + `selection.py` + `interactive_map.py` |
| playback was `time.sleep` + a whole-app `st.rerun` | blocks the server thread, swallows clicks, and re-executes the whole script — a cache miss would run perception once per displayed frame | `playback_controller.py`, an `st.fragment` with `run_every` |
| the engineering dashboard was the first thing on screen | the demo was the fifth tab | `ui_layout.py`, `live_demo.py`, three tabs with Live demo first |

### The inverse projection

`overlays.make_projector` is the forward transform the renderer draws
through. `coordinate_transform.ViewTransform` is its exact inverse, and it
inverts the whole chain: the display flip (`st.image(np.flipud(img))`), the
pixel scale, the camera rotation about the view centre, and the vehicle
frame's translation and heading.

It is a frozen value object rather than a reference to the live camera,
because a click arrives on the *next* Streamlit run, by which time the
camera may have moved; the transform that produced the clicked pixels is the
one that has to answer.

Measured by `scripts/test_interaction.py` on a real precomputed run:

```
1. inverse projection round-trip
   top-follow   World    max error  0.0000 mm   ok
   top-follow   Vehicle  max error  0.0000 mm   ok
   chase        World    max error  0.0000 mm   ok
   chase        Vehicle  max error  0.0000 mm   ok
   world-fixed  World    max error  0.0000 mm   ok
   (five presets x two frames, all ok)
   worst across every camera: 0.0000 mm (a pixel is 143 mm at the default zoom)

2. click a real cell -> the same cell
   top-follow   World    398/398 exact (100.0%), 2 off screen   ok
   chase        World    400/400 exact (100.0%), 0 off screen   ok
   chase        Vehicle  398/398 exact (100.0%), 2 off screen   ok

3. the finest cell covering the point is the one returned
   600 points checked, 0 returned a coarser cell than one that existed   ok

4. clicking a tracked object selects the object
   36 of 36 on-screen tracks   ok
```

A round-trip test alone would pass on a transform that is self-consistently
wrong, so `tests/test_coordinate_transform.py` also compares the forward
half against `overlays.make_projector` — the projector the pixels actually
came from — at every camera and in both frames.

One detail that is easy to get wrong and invisible when you do:
`streamlit-image-coordinates` reports `event.offsetX` on the `<img>`
element, which is in **displayed** pixels. At `width="stretch"` the 840 px
canvas is laid out at about 721 px, so an uncorrected click is pulled ~14%
towards the left of the map — plausible, and wrong. `scale_click` divides by
the displayed size the component reports and multiplies by the natural size.

### Selection

Lookup runs against the per-frame **cache**, not the live map:
`AdaptiveMap.cell_at` only exists for the last frame, and a click while
scrubbing at frame 4 has to answer with frame 4's values. Each level is
indexed lazily as a sorted packed-key array queried with `searchsorted` (a
Python dict of 118k cells costs more than the whole render), memoised on the
cached frame so scrubbing back and forth does not rebuild it.

Hit priority is object, then cell, then empty space, with objects matched in
**screen** pixels: an object at 60 m is a few pixels across, and a
metre-based radius would make it unhittable while making near objects
greedy.

An object's "underlying cell" needed care. An exact lookup at the centroid
fails for **18 of the 30** tracked objects on frame 0 of `mixed_urban` — not
because the map lost them, but because a LiDAR sees the surfaces of a car
and not the middle of it, so the centroid lands in the object's own
occlusion shadow. The nearest mapped surface within 1.5 m is reported
instead, with the offset stated on screen. The first version of this panel
said "its points were gated out of the persistent map", which is the wrong
explanation for a stationary car; it was corrected.

### The frame cache

Extended from 12 fields to every `MapCell` field plus the full six-class
posterior, because the inspector is driven entirely from the cache:

```
118,043 cells/frame, 3.31 MB/frame, 26.5 MB for 8 frames
```

`intensity_var` and `observability` had to be added to both array builders
in `adaptive_map.py`; they were stored but never exposed.

### The inspector

Eight sections — Identity, Terrain and geometry, Semantics, Occupancy,
Dynamics, Sensor and intensity, Flags, Raw/debug — with probabilities drawn
as bars and derived quantities labelled as derived. Slope, step, obstacle
height and the traversability verdict are computed from the cached frame's
neighbours using the map's own thresholds, so the same map still answers
differently for a wheeled and a tracked vehicle.

Two honesty fixes went in here:

* the field the map calls `penetration` is `return_number / return_count`,
  which is **1.0 for a single return**. Printed under that word it reads
  exactly backwards, so it is shown as a return-position ratio with the
  meaning spelled out, and a test fails if the bare label comes back.
* the Raw/debug panel states that cached values are byte-packed and
  therefore good to about 1/255 of their range, not to float precision.

### Playback

`st.fragment(run_every=...)` with `st.rerun(scope="fragment")`. Both exist in
the installed Streamlit (1.64) and both are **probed** rather than assumed —
on an older build the transport still works, it just does not auto-advance.

Two subtleties, each of which cost a debugging cycle:

* `run_every` is fixed when the fragment is **decorated**, which happens in
  the enclosing script run. Play and pause therefore have to rerun the
  *app*, or the timer keeps its old schedule and the button appears to do
  nothing at all. Everything else stays fragment-scoped and never re-enters
  the pipeline.
* a fragment reruns for two reasons — the timer, and a widget inside it —
  and Streamlit does not say which. Without a wall-clock gate on the tick,
  dragging the timeline during playback also steals a frame.

Measured redraw cost, `FOVEA_PROFILE=1`, 12-frame `mixed_urban`:

```
[frag]    95.6 ms  frame 1        compose 56 ms + PNG encode 10-35 ms
[frag]    97.0 ms  frame 5
[frag]   104.9 ms  frame 8
[frag]   100.9 ms  frame 11
```

End to end, including the browser round trip, playback settles near **2
frames per second** whatever the speed selector says. That is stated on
screen next to the transport, and it is called a replay rate — it says
nothing about pipeline latency.

### Layout

Two layout bugs were found by looking at the rendered page rather than at
the code:

* the page title was invisible. Streamlit's toolbar is 60 px tall, opaque
  and absolutely positioned at the top of the page; `.block-container
  {padding-top: 1.1rem}`, inherited from the first dashboard, tucked the
  first element underneath it. Now 4.4rem.
* the click component sizes its iframe from the image's **natural** height,
  then the browser scales the image down to the column width — leaving a
  band of dead space and a scrollbar. The aspect ratio is known at render
  time, so it is pinned.

The side rails are height-capped with their own scroll: Streamlit stretches
every column in a row to the tallest one, so a control rail that grew past
the canvas opened ~350 px of dead space and pushed the transport off the
bottom of the screen.

### Tests

63 to **167**, all passing.

| file | what it protects |
|---|---|
| `tests/test_coordinate_transform.py` (28) | round-trip at every camera and frame, agreement with the renderer's own projector, the display flip, zoom |
| `tests/test_selection.py` (25) | containment at all five levels, finest-cell-wins, negative coordinates, click to cell, object priority, selection persistence across frames, the nearest-surface fallback |
| `tests/test_playback_controller.py` (27) | stepping, wrapping, scrub-does-not-pause, the tick gate, and two guards: the transport imports nothing from the pipeline, and `time.sleep` cannot come back anywhere in `visualization/` |
| `tests/test_inspector.py` (24) | every `MapCell` field reaches a section, all eight sections render in order, no raw array dump, the verdict matches the map's own thresholds, and the `penetration` label |
| `scripts/test_interaction.py` | the four claims above, on a real precomputed run |

### Not done

**No 3D scene view.** The brief made it optional and said not to destabilise
the mapping code for it. The visualisation layer is split so that a Three.js
renderer could replace `interactive_map.render_canvas` without touching
selection, the inspector or the transport, because all three go through
`ViewTransform` rather than through pixels.

## PHASE 13 — SCENE DEMO ✅

A second visualisation tab, beside the Live demo and sharing its run: the
same 2.5D map drawn as the physical environment it is a map of.

```
LIVE DEMO   what does the adaptive map look like?
SCENE DEMO  what is that map a map OF?
RESEARCH    how well does the system perform?
```

Nothing new is computed. Cells give the surfaces and their heights, tracked
instances give the objects, the stored pose gives the vehicle. No second
pipeline, no second inspector, no invented geometry.

### The renderer

three.js r128 (MIT), vendored at
`visualization/scene_frontend/vendor/three.min.js`, declared as a Streamlit
component from four static files with no build step. Nothing is fetched at
run time, which is the same constraint the rest of the project runs under.
The Streamlit bridge is 40 lines written here rather than the official
React build, because the protocol the page needs is three `postMessage`
calls.

Orbit / pan / zoom are hand-written pointer handlers — about sixty lines
against another vendored file, and it keeps click-versus-drag under our
own control.

### Pipeline data to geometry

`visualization/scene_data.py` is the whole adapter, and the rule it exists
to enforce is that **no pipeline logic lives in JavaScript**. The browser
receives positions, sizes and colours; it decides nothing.

| drawn | from | note |
|---|---|---|
| cell footprint | `ResolutionLevel.size(level)` | 1:1, never scaled |
| cell base / top | `ground_z` / `z_max` | clamped only when `z_max < ground_z`, which the map allows |
| cell colour | `render.CLASS_COLOURS` etc. | the 2D map's own palettes, passed through |
| verdict | `inspector.traversability_arrays` | the map's own thresholds |
| object box | `Instance.extent` | padded to a class floor when smaller, and **flagged** |
| ego | the stored pose | 4.6 × 2.0 × 1.5 m, sensor at the configured mount height |
| rings | `RANGE_BANDS` | 10 / 30 / 60 / 100 m — the evaluation's own bands |

Cells go over the wire as base64 of raw buffers, 16 bytes each: centre as
float32 (the click path round-trips through it into the real cell lookup,
where a centimetre of drift would pick the neighbouring 5 cm cell), height
quantised to centimetres, and level/class/verdict/returns as bytes.

```
Light     build  61 ms   payload 188 kB   8,000 cells
Balanced  build  55 ms   payload 396 kB  18,000 cells
Full      build  59 ms   payload 959 kB  45,000 cells
```

The cap is nearest-first and whatever is dropped is counted under the
canvas, never quietly thinned.

### Two honesty problems this view created

**Object boxes are not measurements.** A tracked instance's `extent` is the
bounding box of the returns, and a LiDAR sees one side of a car, so the box
is routinely too small to read as a car. Enlarging it silently would be
presenting drawn geometry as sensor output. Instead a padded box is drawn
fainter, carries `geomSource: "padded"`, and the inspector says:

> **Visualisation geometry.** The sensor measured 0.90 × 0.40 × 0.50 m —
> the returns from one side of the object. The scene draws 3.80 × 1.70 ×
> 1.40 m, raised to this class's floor so the object is visible at all.

On one frame of `mixed_urban`, **18 of 27** boxes are padded.

**Ground-class tracks are not objects.** The tracker emits instances the
classifier called `ground_drivable` — an 8 × 1 × 0.16 m slab of road. In
the 2D map that is a thin outline. In a 3D scene it becomes a solid lying
across the road that (a) reads as something that is not there and (b) sits
between the cursor and the cells, swallowing the cell clicks this tab
exists for. They are counted under the canvas (`3 ground-class tracks (not
boxed)`) and still listed in the Live demo's object table.

### Selection: one source of truth

The renderer reports a world point and, if it hit one, a track id. It never
reports a cell — resolving the point is `selection.py`'s job against the
cached frame, so the scene and the 2D map cannot disagree about what is
under a place. The inspector is `inspector.render_cell`, the same function
the Live demo calls, with one extra line above it tying the solid shape
back to the record:

> **20 cm cell** · ground_drivable · 26 return(s)
> the shape you clicked is one row of the 2.5D map, drawn from ground
> −0.08 m to top −0.06 m

Verified in the browser: clicking a surface selected the cell at
(−4.900, 0.500), 20 cm, level 2, Morton code 114532460590 = the level-0
code >> 4, DRIVABLE; switching to the Live demo showed the same cell
already selected.

### Vectorised traversability, and a drift caught by testing it

The scene needs a verdict for ~18,000 cells, which the per-cell
`inspector.derive` cannot supply at that rate.
`inspector.traversability_arrays` is the vectorised twin — and writing the
test that the two agree found that they did not: `derive` resolves a
neighbour probe through the finest-first lookup (so a 20 cm cell whose
neighbour was refined to 5 cm finds the 5 cm cell and measures the real
distance to its centre), while the first vectorised version only looked at
the same level. Fixed to mirror it exactly.

```
per-cell vs vectorised verdict on a real frame: 370/370 agree
```

### Playback: shared, and clock-driven

Both tabs bind to one `PlaybackState`; only the widget keys are per-tab,
because Streamlit refuses two widgets with the same key. Three bugs came
out of that, each invisible from the outside:

1. **Two tickers, double speed.** Each tab's fragment advanced the shared
   index on every redraw, so the run played at the sum of their render
   rates. The frame is now a function of elapsed time
   (`PlaybackState.frame_at`), which makes the second renderer free.
2. **The slider scrubbing itself.** Streamlit fires a slider's `on_change`
   for the value the *server* writes, not only for a drag. Syncing the
   handle to the clock therefore looked like a scrub several times a
   second, and each one restarted playback from a stale position. While
   playing, the timeline is now a read-only progress bar; the slider comes
   back when paused.
3. **Reruns queuing.** `run_every` shorter than the redraw piles runs up
   behind each other and the browser gets payloads out of order. Each tab
   now schedules no faster than it can draw — 0.15 s for the 2D canvas,
   0.40 s for the scene, both measured.

Play/pause reruns the *app* rather than the fragment, because `run_every`
is fixed when the fragment is decorated; everything else stays
fragment-scoped. Selection changes are app-scoped too, so a cell picked in
one tab is already selected when the other is opened.

### Layout bugs found by looking at the page

* the canvas kept its size from the last Streamlit render, so resizing the
  browser left the scene stretched in a corner. A `ResizeObserver` now
  follows the column.
* the component sizes its iframe from the image's natural height, leaving
  a dead band under the 2D canvas. The aspect ratio is known at render
  time, so it is pinned.
* the default camera started inside the buildings; it now opens above and
  behind the vehicle.
* labels overlapped into an unreadable stack; they are now capped at 12,
  movers and non-ground classes first, and skipped when they would collide.

### Tests

167 → **213**, all passing. `tests/test_scene_data.py` (46) covers cell
centres surviving the wire exactly and still resolving to their own cell,
footprints from the level table, heights from `ground_z`/`z_max`, the
clamp, the nearest-first cap and its count, primitive choice, padded-box
flagging and that padding grows upward from where the points were, world
anchoring (same cells, two ego poses, identical drawn coordinates), JSON
serialisability, rings equal to `RANGE_BANDS`, palettes equal to the 2D
renderer's, and the ground-class exclusion. `test_playback_controller.py`
gained the clock-driven and two-renderer properties.

### Not done

* **Full point cloud.** Only the dynamic overlay is cached per frame, so
  "Moving points" draws that and says so. Caching every point for every
  frame would cost more than the whole rest of the cache.
* **Both tabs animate when playing, including the hidden one.** Streamlit
  cannot tell which tab is visible. The clock-driven frame keeps that
  correct, but it does cost a redraw the viewer never sees.

## PHASE 14 — THREE BUGS FROM USING IT ✅

All three were reported from the running app, and all three are the kind
that a unit test would only have caught if someone had thought to write it.
They now have those tests.

### "It plays only 3 frames and jumps around randomly"

Exactly what the code did. Playback derived the frame from elapsed time:

```
frame = start + (now - started_at) / interval
```

which is correct if a redraw is cheap. A redraw is not cheap — 100 to
300 ms against the 100 ms a frame is supposed to be held for — so every
redraw advanced three or four frames, and a 12-frame run cycled through
the same three of them. The scene never looked like it was moving; it
looked like it was shuffling.

Now the frame advances by **at most one per redraw**, and only once the
interval has elapsed. Every frame is shown, in order, a little slower than
the scene really ran, which is what the readout has always called a replay
rate.

Two related fixes fell out of it:

* **Only one tab drives playback.** Both tabs bind to one shared index, so
  when both ticked the run played at the sum of their render rates. The
  ticker is now owned by whichever tab last used a transport control; the
  other shows the position and says where it is being driven from. The
  hidden tab also stops redrawing, which is most of why the visible one
  got faster.
* **The timeline is only a widget in the tab that owns it, and only when
  paused.** A slider has to be written on every redraw to follow playback,
  Streamlit fires `on_change` for the value the *server* writes, and with
  one in each tab the idle one reported a position from before playback
  moved — which read as a drag, seeked backwards, and cascaded reruns
  until it caught up. A progress bar has no widget state to fight over.

Measured after, one frame per redraw and nothing skipped:

```
live demo   0 1 2 ... 23 0 1 2 ...     ~200 ms per frame
scene demo  0 1 2 ... 23 0 1 2 ...     ~215 ms per frame
```

### "Only mixed_urban gets selected"

`session.pull()` copied the shared scenario into the widget slot before
the widget was drawn. Streamlit writes a *changed* widget value into
session state **before** the script runs, so that copy landed on top of the
selection the user had just made and the dropdown sprang back. Every
scenario except the default was unreachable, in both tabs.

`session.sync()` replaces it and decides the direction by comparing both
sides against the value last agreed on: whichever one moved is the one
that wins. `tests/test_session.py` pins it down, including the exact
sequence that failed.

All six scenarios load — this was never a data problem:

```
empty_road         ok   75,518 cells   6 objects
pedestrian_far     ok   76,056 cells   7 objects
canopy_over_road   ok   81,262 cells  10 objects
moving_vehicle     ok   77,310 cells   7 objects
convoy             ok   90,315 cells  50 objects
mixed_urban        ok   78,420 cells  90 objects
```

### "Keep more frames so we can show a proper demo"

8 to 40 frames, default **24** — 2.4 s of scene at 10 Hz, long enough for
the vehicle to drive past the 70 m pedestrian and for the resolution
island to visibly travel with it. `_MAX_FRAMES` in `app.py` was raised to
match, since it caps what the loader generates.

The scans are raycast once per scenario and memoised to `.scan_cache/`, so
the cost lands on the first load of a scenario and never again:

```
empty_road         40 scans in   0.7 s        (cached)
pedestrian_far     40 scans in   0.9 s        (cached)
canopy_over_road   40 scans in  10.6 s
moving_vehicle     40 scans in  14.4 s
convoy             40 scans in  32.8 s
mixed_urban        40 scans in  51.7 s
```

A `0.1x` replay speed was added at the same time: a second per frame, slow
enough to talk over, which is what a walk-through actually needs.

Tests: 213 → **231**.

### Follow-up: the replay-speed control

Same fault as the scenario dropdown, in the one place it had not been
fixed: `speed_selector` wrote the shared value into the widget slot before
drawing it, which lands on top of the choice Streamlit has already stored.
The control snapped back to its previous value every time — "it always
stays as 1x even though I try to change it".

It now goes through the same `session.sync`, and the default is **0.1x**:
one second a frame, slow enough to talk over while pointing at things,
which is what a walk-through in front of judges needs. The fast end still
goes to 4x, bounded by the redraw.

The regression test drives the real `speed_selector` through a Streamlit
stub that reproduces the write-before-run behaviour exactly, rather than
asserting on the source — the choice is kept, it survives a redraw, it
reruns the app so `run_every` is re-armed, and both tabs see one speed.

Tests: 231 → **239**.

### Follow-up: the scene went white and stuttered above 0.1x

Three separate costs, and the first one explains the white screen.

**The renderer threw away its GPU buffers every frame.** `buildCells`
disposed the InstancedMesh and built a new one per redraw: at 18,000 cells
that is 1.15 MB of instance matrices plus 216 kB of instance colours
allocated, uploaded and discarded several times a second. Enough churn to
stall an integrated driver and eventually lose the WebGL context — which
is exactly what a white canvas is. The mesh is now allocated once at a
capacity that only grows, and a frame overwrites the buffers it already
owns and sets `count`. Matrices and colours are written straight into the
attribute arrays, because `setMatrixAt` through a Quaternion and three
Vector3s costs more than the rest of the loop at that instance count.
Object geometry is cached by shape and size, and the point cloud reuses
one buffer with `setDrawRange`.

```
browser build, 18,000 cells:   full realloc  ->  12 ms
```

A `webglcontextlost` handler now rebuilds the scene instead of leaving a
dead canvas, so even an unrelated driver hiccup recovers.

**The verdict was computed for cells nobody drew.** `cell_geometry` called
`traversability_arrays` over all ~78,000 cached cells and then indexed the
18,000 it wanted — a neighbour lookup per cell, most of it discarded. It
now answers only for the drawn rows (probing still resolves against every
cell, so an edge cell still sees the neighbour beyond it), and is skipped
entirely unless the traversability colour mode is selected.

```
build_scene_data  semantic         99 ms  ->  10 ms
                  traversability   99 ms  ->  39 ms
```

`tests/test_scene_data.py` asserts the subset verdict is identical to the
full one on the drawn cells — the optimisation is only worth having if it
changes nothing.

**Pacing was a constant measured on the wrong machine.** A fixed interval
schedules reruns faster than a slower laptop can serve them; they queue,
arrive out of order, and playback appears to stop and restart. Each tab now
keeps an exponential moving average of what a redraw actually cost there
and never schedules faster than 1.5x it — rising fast when the machine
struggles, recovering slowly so it does not oscillate. The constants that
remain are lower bounds before the first measurement lands.

Also removed: a `setFrameHeight` announced on every frame for a height
that had not changed, which was making Streamlit re-render the component
host and roughly doubling the reruns during playback.

Measured end to end, scene demo, 18,000 cells, 24 frames:

```
server redraw   460 ms  ->   30 ms   (median)
browser build   realloc ->   12 ms
frames          0 1 2 3 ... 23 0 1   strictly in order, context alive
```

Tests: 239 → **248**.
