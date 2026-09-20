# Adaptive Variable-Resolution 2.5D LiDAR Mapping

**SIH 2026 · DRDO problem statement 26053 — Adaptive Variable Resolution 2.5D
LiDAR Mapping for Dynamic Environment Perception**

A 2.5D map whose cells are genuinely different physical sizes — 5 cm where it
matters, 80 cm where it does not — with the size chosen by a budgeted
controller that treats safety as a constraint rather than a weighted term.
CPU only; no CUDA anywhere.

---

## Run it in three commands

```bash
pip install -r adaptive_lidar/requirements.txt

cd adaptive_lidar
python main.py --demo                 # the pipeline, end to end
streamlit run app.py                  # FOVEA — opens on the Live demo tab
```

No data download, no configuration, no GPU. The synthetic sensor model and the
trained network weights are both in the repository, so a fresh clone runs.

To reproduce every reported number:

```bash
python scripts/run_baselines.py && python scripts/generate_report.py
```

---

## What it does

| | requirement | where |
|---|---|---|
| **M1** | a trained network gives every point a semantic class | `perception/pointfeature_net.py`; weights committed at `models/pointfeature_net.pt` |
| **M2** | terrain analysis: drivable / caution / blocked | `AdaptiveMap.traversability()` — slope, roughness, step, clearance, class, per vehicle profile |
| **M3** | static obstacles and dynamic objects as tracked instances | `perception/tracking.py`, three track states |
| **M4** | cell size grows with distance, **no alignment error or data loss** | `mapping/adaptive_map.py`; proved by `scripts/test_map.py` |
| **M5** | dashboard, distinct terrain/object colours, measured memory reduction | `app.py` + `visualization/live_demo.py` — click any cell for all 21 stored fields |
| **M6** | latency p50/p95/p99 and **accuracy stratified by range** | `evaluation/metrics.py`, every table in the report |

---

## The three ideas

**One aligned power-of-two hierarchy.** Cells are `0.05 × 2^ℓ` above a fixed
global origin, addressed by 2D Morton code, so the level-ℓ cell id is the
level-0 code shifted right by 2ℓ bits. Containment is a prefix relation on
bits: a fine cell is always wholly inside exactly one coarse cell. That is the
whole answer to *"without causing alignment errors or data loss"* — it is
structural, not a tolerance. The allocation tile (1.6 m = 0.05 × 2⁵) is a node
of the same hierarchy, so one sort of the cloud groups it by tile and by cell
at every level at once.

**Allocation is a constrained optimisation.** Tiles are ranked by value per
unit cost, `ρ = V / Δcells`, under a budget expressed as a fraction of a
uniform 5 cm map's memory. Safety-critical tiles are **pinned** — they consume
budget first and are never ranked against it — because a weighted sum makes
safety tradeable and the 70 m pedestrian loses. The pin that carries the
guarantee is geometric, not semantic: it fires on "small, isolated, vertically
extended cluster above the ground" without knowing what the object is, so it
survives a segmentation failure.

**Everything the map accumulates is associative.** Occupancy is log-odds
addition, semantics is evidence addition, elevation is Kalman fusion. Temporal
fusion and inter-level aggregation are therefore the same operation, and
neither needs a special case.

---

## Results

Full tables, every number traced to a CSV, in
[`adaptive_lidar/docs/RESULTS.md`](adaptive_lidar/docs/RESULTS.md). Each
verification script asserts its own claims and prints what it measured:

```bash
python scripts/inspect_synthetic.py        # the sensor model
python scripts/validate_resolution_law.py  # the derived schedule
python scripts/test_map.py                 # M4: alignment and aggregation
python scripts/test_allocation.py          # retention vs budget
python scripts/test_temporal.py            # the MOS gate
python scripts/eval_semantic.py            # M1/M6: per-class IoU by range
python scripts/test_interaction.py         # click -> cell, on a real run
python -m pytest tests/ -q                 # 239 tests
```

Selected measured outcomes:

* **The 70 m pedestrian survives every budget.** `full` keeps it at 11 cells
  of 5 cm down to a 10% budget, using 1.12 MB; `random` at the same budget
  keeps 2 cells at 80 cm using 1.71 MB; `distance_only` — exactly what the
  problem statement literally asks for — spends 6.45 MB and keeps 2 cells.
* **The derived resolution law reproduces the PS's own example.** Cell size
  ∝ r, anchored 5 cm at 10 m, gives 50 cm at 100 m. It holds points-per-cell
  to 4.2× variation across range, where a uniform 20 cm grid varies 78× and a
  uniform 80 cm grid 162×.
* **The MOS gate removes the trail.** 12.00 m of phantom wall with the gate
  off, 0.00 m with it on, on the same 20-frame run.
* **Motion is judged in the world, not relative to the sensor.** In the
  `convoy` scenario a car travelling at exactly the ego speed (0.30 m/s
  relative) is correctly MOVING, and a genuinely parked one is
  MOVABLE_BUT_STATIONARY — the reading a naive frame-difference would get
  backwards in both cases.
* **Driving 19 m does not smear the map.** A building facade accumulated over
  a whole run is still one cell thick.
* **The map aggregates exactly.** Coarsening a level-0 map to level 2 matches
  a natively-built level-2 map: `n_points` bit-identical, `z_max` bit-identical
  in 99.98% of cells and the rest within the float16 quantum (~4 mm).
* **The semantic network.** 11,406 parameters, validation mIoU 0.871, ECE
  0.0123 → 0.0080 after temperature scaling, 28.6 ms for 55k points on CPU.
* **Speed-up over the previous implementation:** 4,247 → ~300 ms per frame at
  120k points (24× at 8k, 22× at 30k, 13× at 120k).

### The dashboard

`streamlit run app.py` opens on **Live demo**: a scene you can drive, with a
compact control rail on the left, a clickable map in the middle and a cell
inspector on the right.

* **Click any cell** and the inspector opens all 21 stored fields in eight
  sections — identity and Morton address, terrain and the traversability
  verdict with its reason, the full six-class posterior rather than the
  argmax, occupancy, dynamics, sensor and intensity, flags, and a named raw
  table. Clicking a tracked object opens the track instead, with a link down
  to the cell beneath it. The hit test resolves to the *finest* cell
  covering the point; `scripts/test_interaction.py` checks 1,196 clicks
  across three cameras and gets the right cell every time.
* **Playback** is an `st.fragment`, not a sleep loop: the controls stay live
  while it runs, and advancing a frame never re-enters the pipeline. The
  rate on screen is labelled a replay rate — it is bounded by drawing the
  canvas, not by perception.
**Scene demo** draws the same run as the physical environment the map is a
map of: cells become surfaces at their real footprint, from `ground_z` to
`z_max`; tracked instances become boxes, poles and figures; the stored pose
becomes the vehicle. Orbit, top, chase and sensor cameras; click a surface
and the *same* inspector opens on the *same* cell. It is a three.js
component built from four static files vendored in the repository — no
build step, no CDN, nothing fetched at run time.

Nothing there is a second perception run, and where drawn geometry is not
measured geometry it says so: a LiDAR sees one side of a car, so a box
enlarged to be visible is shaded fainter and the inspector prints both the
measured and the drawn extent.

* Two more tabs hold the engineering work: **Research & evaluation** (the
  equal-memory comparison, range-stratified accuracy, per-stage latency) and
  **Debug & cell inspector** (query the live map by coordinate).

Everything on screen states what it is: the header carries a *synthetic /
demo data* badge and the actual configured backend, and every metric block
repeats that these are demo-pipeline numbers, not deployment claims.

### Where it falls short

**Latency.** The requirement is ≤ 100 ms per frame at 120,000 points. The
system reaches ~300 ms at 128k points and meets 100 ms at roughly 25k. The
anti-patterns that made the original 4,247 ms are gone — there is no Python
loop over points anywhere in the hot path and no repeated full-array scan —
and what remains is real vectorised work, dominated by per-point semantics and
the map insert. The full per-stage breakdown is in `docs/RESULTS.md`; the
number is reported rather than rounded towards the target.

**Real data is untested.** `get_dataset` reads SemanticKITTI and RELLIS-3D
trees, and the label maps for both are written and unit-tested, but no real
sequence was available on this machine, so that path has never been run
against one. The synthetic sensor model is a genuine raycast simulation — real
occlusion, 1/r³ ground falloff, multi-echo, material reflectance — but it is
not a substitute for that check.

**Far-field terrain classification is weak.** Drivable-vs-rough IoU falls from
0.69/0.83 in the near bands to 0.23/0.08 beyond 30 m, where the beam footprint
mixes road and verge. The range-stratified tables exist precisely so this is
visible rather than averaged away.

---

## Repository layout

[`adaptive_lidar/docs/ARCHITECTURE.md`](adaptive_lidar/docs/ARCHITECTURE.md)
has the stage-by-stage description and the full data contract.

```
adaptive_lidar/
  main.py  app.py  config.yaml
  pipeline/   types (the data contract), orchestration, timing
  stages/     S0..S9
  perception/ features, ground, backends, the MLP, motion, tracking
  mapping/    adaptive_map.py (the map), allocation.py (the controller)
  evaluation/ reference map, metrics
  data/       raycast sensor model, loader, label maps
  utils/      grouping, spatial index, range image, voxel hash
  scripts/    profiling, training, verification, baselines, report
  tests/      239 tests
  models/     the trained weights
  docs/       RESULTS.md, ARCHITECTURE.md, figures, CSVs
PROGRESS.md   the working log, with pasted verification output
DECISIONS.md  every judgement call and its reason
```

---

## Honesty notes

* Every number in the dashboard and in `docs/RESULTS.md` was measured by a
  script in `scripts/` and read from a file in `docs/`. Where a measurement is
  missing, the cell says so rather than being filled in.
* `Frame.gt_label` is ground truth. It is read by `evaluation/` and by the
  `oracle` backend and nowhere else;
  `tests/test_gt_label_isolation.py` parses every perception and mapping
  module and fails if any of them touches it.
* Oracle mode measures the map, never the segmenter. It warns on the console,
  shows a banner in the dashboard, and is recorded in every metrics row.
* Memory is measured with `tracemalloc` on the live structures, never computed
  as cells × bytes.
* The uniform-5 cm baseline is accumulated over the same frames and the same
  sliding window as the adaptive map, so the comparison is like-for-like.
* The central experiment fixes the memory and asks which map still contains
  the pedestrian. Comparing an adaptive map against a uniform 5 cm map on size
  alone is rigged, and is not the comparison reported.

---

## Requirements

Python 3.11+ with `numpy`, `scipy`, `torch` (CPU wheel), `streamlit`, `pandas`,
`matplotlib`, `pyyaml`, `pillow`, `pytest`. All free and open source; nothing
requires a GPU, and nothing requires a network connection at run time.
