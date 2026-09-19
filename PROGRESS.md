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
