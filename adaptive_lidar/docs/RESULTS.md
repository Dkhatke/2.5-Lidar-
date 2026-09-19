# RESULTS

Adaptive Variable-Resolution 2.5D LiDAR Mapping — SIH 2026 / DRDO PS 26053.

**Every number below was measured by a script in `scripts/` and read from a file in `docs/`.** Where a measurement is absent the cell says `—` rather than being filled in. Re-generate with:

```
python scripts/run_baselines.py
python scripts/generate_report.py
```

Generated 2026-09-19 15:31:56 · source `synthetic/mixed_urban` · 4 frames · 128,413 points/frame · backend `pointfeature_net`

## 1. Headline

| quantity | value | requirement |
|---|---|---|
| map memory (measured) | 1.50 MB | — |
| uniform 5 cm over the same area | 5.04 MB | — |
| **memory reduction** | **70.3 %** | M5 — 'significant memory reduction' |
| cells at 5 cm / 10 / 20 / 40 / 80 | 17,197 / 12,591 / 4,174 / 7,223 / 1,611 | M4 — variable cell size |
| latency p50 / p95 / p99 | 865.0 / 950.7 / 955.0 ms | M6 — low latency |
| VRU object retention | 1.000 | M3 — dynamic objects preserved |
| semantic mIoU | 0.473 | M1 — segmentation |
| elevation RMSE vs reference | 0.385 m | — |

(`full` policy at 25% budget, where budget means *this fraction of the cell count a uniform 5 cm map of the same observed area would need*.)

## 2. The central experiment — equal memory, not equal resolution

Comparing an adaptive map against a uniform 5 cm map is rigged: of course it is smaller, it was told to be. The honest question fixes the memory and asks what the best map obtainable for it looks like. Uniform spends the budget evenly; this system spends it where the value function says it matters.

**memory (MB) / VRU retention**, by policy and budget:

| policy | 100% | 25% |
|---|---|---|
| full | 3.49 MB / 1.00 | 1.50 MB / 1.00 |
| random | 5.26 MB / 1.00 | 4.23 MB / 1.00 |
| uniform_20 | 1.06 MB / 0.67 | 1.06 MB / 0.67 |

![Retention vs memory](pareto.png)

*`full` should sit above and to the left of everything else: the same retention for less memory, or more retention for the same.*

## 3. Accuracy across varying distances (M6)

The problem statement asks for accuracy *across varying distances*, which is not an aggregate. Both tables are stratified into the four range bands.

### 3.1 Elevation RMSE against the reference map, by range band

| policy | 0-10m | 10-30m | 30-60m | 60-100m | overall | bias (overall) |
|---|---|---|---|---|---|---|

Signed bias is reported alongside RMSE because a planner can absorb variance but not a systematic offset: a consistent 10 cm underestimate of a kerb is what drives a vehicle into it.

### 3.2 Semantic mIoU by range band

| metric | 0-10m | 10-30m | 30-60m | 60-100m | overall |
|---|---|---|---|---|---|
| mIoU | 0.315 | 0.618 | 0.650 | 0.677 | 0.473 |
| accuracy | 0.514 | 0.724 | 0.611 | 0.590 | 0.574 |

Per-point semantics do not depend on the allocation policy — the same classifier sees the same points — so this table is the same for every policy. What the policy changes is how much of that classification survives into the map, which is the retention table above.

## 4. Semantic segmentation (M1)

| class | IoU | ORR | objects observed |
|---|---|---|---|
| ground_drivable | 0.425 | — | 0 |
| ground_rough | 0.008 | — | 0 |
| static_obstacle | 0.854 | 0.952 | 21 |
| vehicle | 0.337 | 0.800 | 5 |
| vru | 0.252 | 1.000 | 3 |
| vegetation | 0.959 | 1.000 | 6 |

### Calibration

| quantity | value |
|---|---|
| temperature | 1.3341 |
| ECE before | 0.0097 |
| ECE after | 0.0064 |
| validation points | 256,658 |
| validation mIoU | 0.843 |

*Raw softmax from a cross-entropy-trained network is systematically overconfident. The allocation controller's uncertainty term and the map's entropy layer both read this distribution, so it is calibrated before either sees it.*

## 5. Latency (M6)

### Per-stage latency vs point count

| points | S0 | S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 | total ms | FPS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 8,034 | 0.7 | 2.2 | 7.6 | 6.2 | 20.4 | 16.0 | 5.2 | 12.7 | 0.1 | 0.2 | 71.3 | 14.0 |
| 32,143 | 2.0 | 7.8 | 12.8 | 7.5 | 28.8 | 21.4 | 6.2 | 28.0 | 0.3 | 0.4 | 115.3 | 8.7 |
| 128,459 | 8.2 | 28.3 | 25.5 | 11.3 | 82.6 | 53.4 | 13.7 | 90.2 | 0.7 | 0.9 | 314.9 | 3.2 |

### Against the pre-existing implementation

| points | before (ms) | after (ms) | speed-up |
|---|---|---|---|
| 8,034 | 1737 | 71 | 24.4x |
| 32,143 | 2548 | 115 | 22.1x |
| 128,459 | 4246 | 315 | 13.5x |

### p50 / p99 per stage at the operating point

| stage | p50 (ms) | p99 (ms) |
|---|---|---|
| S0 | 22.14 | 23.37 |
| S1 | 69.30 | 77.94 |
| S2 | 72.96 | 79.22 |
| S3 | 29.93 | 33.81 |
| S4 | 239.54 | 271.59 |
| S5 | 232.26 | 256.18 |
| S6 | 24.56 | 27.41 |
| S7 | 169.36 | 177.93 |
| S8 | 2.97 | 5.47 |
| S9 | 1.97 | 2.07 |
| **TOTAL** | **865.0** | **955.0** |

p99 is reported because for a real-time system the tail *is* the requirement — the mean hides exactly the frames that would miss their deadline.

## 6. Map correctness (M4)

`scripts/test_map.py` asserts all of the following and prints the measured values; see PROGRESS.md for its full output.

## 8. What the reference map is, and is not

No dataset ships ground-truth 2.5D maps, so the reference is constructed: accumulate frames by pose, drop GT-moving points, build a uniform 5 cm map. Its limitations bound every number scored against it:

- poses are exact on synthetic data, SLAM-derived (and drifting) on SemanticKITTI
- only label-declared moving points are removed; a car parked for the whole window is permanent structure here
- surfaces no beam reached are absent, so completeness is scored only where the reference has a cell
- it is itself a 5 cm map, so a coarse adaptive cell is charged for the quantisation the budget asked for

---

*Generated by `scripts/generate_report.py` at 2026-09-19 15:33:04.*
