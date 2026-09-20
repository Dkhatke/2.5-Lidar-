# ARCHITECTURE

Adaptive Variable-Resolution 2.5D LiDAR Mapping — SIH 2026 / DRDO PS 26053.

---

## The shape of the system

One LiDAR scan enters, one variable-resolution 2.5D map is updated, and a
table of tracked objects comes out. Ten stages, each timed:

```
                 ┌──────────────────────────────────────────────┐
  scan ─────────▶│ S0  ingest      Frame + the data contract      │
                 ├──────────────────────────────────────────────┤
                 │ S1  indices     range image (POINT INDICES),   │
                 │                 voxel hash (CSR), ONE Morton   │
                 │                 sort reused by everything      │
                 ├──────────────────────────────────────────────┤
                 │ S2  geometry    ground segmentation, smooth    │
                 │                 height field, tile features    │
                 ├──────────────────────────────────────────────┤
                 │ S3  pre-alloc   geometry-only preview          │
                 ├──────────────────────────────────────────────┤
                 │ S4  semantics   PER-POINT (N,6) evidence       │
                 ├──────────────────────────────────────────────┤
                 │ S5  motion      residual MOS, instances,       │
                 │                 Hungarian + Kalman tracking    │
                 │                 → validate_frame_contract()    │
                 ├──────────────────────────────────────────────┤
                 │ S6  allocation  THE CONTROLLER — level/tile,   │
                 │                 then level/point               │
                 ├──────────────────────────────────────────────┤
                 │ S8  MOS gate    moving → overlay (cleared),    │
                 │                 static → persistent map        │
                 ├──────────────────────────────────────────────┤
                 │ S7  map update  variable-resolution insert     │
                 ├──────────────────────────────────────────────┤
                 │ S9  telemetry   every number the UI displays   │
                 └──────────────────────────────────────────────┘
                              │                │
                        adaptive map     object table
```

**S8 runs before S7.** The gate decides which points are allowed into the
persistent map, and a gate that runs after the thing it gates is not a gate —
that ordering is what used to let moving objects leave trails. The stage
decomposition is otherwise unchanged.

---

## The three ideas the system is built on

### 1. One aligned power-of-two hierarchy, addressed by Morton code

Cell sizes are `0.05 × 2^ℓ` — 5 / 10 / 20 / 40 / 80 cm — above one global
origin fixed at construction. Each point gets a 2D Morton code at level 0, and

```
level-ℓ cell id  ==  level-0 code >> (2ℓ)
```

That single identity is the whole answer to the problem statement's *"without
causing alignment errors or data loss during projection from 3D to 2.5D"*.
Containment becomes a prefix relation on bits: a fine cell is **always** wholly
inside exactly one coarse cell, there is no float boundary case, no point can
round into two parents, and coarsening is an exact regrouping rather than a
resampling. `scripts/test_map.py` tests all four of those claims.

The **allocation tile is also a node of this hierarchy** — 1.6 m = 0.05 × 2⁵ —
so a tile id is a prefix of the level-0 code too. A tile boundary therefore
can never fall inside a map cell, and one `argsort` of the level-0 codes
simultaneously groups the cloud by tile and by cell at every level. That one
sort replaced four (`utils/spatial_index.py`).

The PS offers "~50 cm far" as an example; 40/80 cm are used instead because
10 is not a power of two and the nesting property above is worth more than
matching an illustrative number. See DECISIONS.md 4.1.

### 2. Allocation is a constrained optimisation, not a threshold

```
V(tile) = w_g·G + w_s·S + w_u·U + w_d·D
ρ       = V / Δcells          ← rank by value per unit cost, not by value
```

Refining a tile one level multiplies its cell count by four, so the budget is
spent in order of ρ. The budget itself is **a fraction of the cell count a
uniform 5 cm map of the same observed area would need**, which makes the
number on the slider mean megabytes.

Safety is a **constraint, not a term**. A weighted sum makes safety tradeable —
with enough competing tiles the 70 m pedestrian is outbid. Pins define the
feasible set: they consume budget first and are never ranked. The pin that
carries the guarantee is **geometric** (`has_vertical_run`), because a semantic
pin is only as good as the classifier.

### 3. Everything the map accumulates is associative

Occupancy is log-odds addition, semantics is evidence addition (the conjugate
categorical update), elevation is Kalman scalar fusion, dynamics is a binary
Bayes filter. Temporal fusion and inter-level aggregation are therefore the
same operation, and neither needs a special case.

**Never** priority-weighted voting: it corrupts the posterior and destroys the
ability to measure semantic accuracy. Safety priorities are applied at the
allocation and query layers instead.

---

## The data contract

`pipeline/types.py`. Frozen once written; `validate_frame_contract()` asserts
it at the end of S5.

| field | shape | dtype | meaning |
|---|---|---|---|
| `points` | (N,3) | f32 | xyz, sensor frame |
| `intensity` | (N,) | f32 | raw |
| `intensity_norm` | (N,) | f32 | range + incidence corrected |
| `ring` | (N,) | i16 | -1 if unavailable |
| `azimuth_bin` | (N,) | i32 | -1 if unavailable |
| `height_above_gnd` | (N,) | f32 | above the smooth ground field |
| `sem_evidence` | (N,6) | f32 | per-point class evidence |
| `sem_class` | (N,) | i8 | argmax |
| `sem_entropy` | (N,) | f32 | normalised, 0..1 |
| `moving_prob` | (N,) | f32 | residual MOS |
| `instance_id` | (N,) | i32 | TRACK id, -1 = none |
| `point_level` | (N,) | i8 | the allocation decision |
| `gt_label` | (N,) | i8 | **EVALUATION ONLY** |
| `range_image` | (H,W) | i32 | **POINT INDEX**, -1 = empty |
| `morton` | — | — | the frame's single spatial ordering |

`gt_label` is read by `evaluation/`, the `oracle` backend and nothing else.
`tests/test_gt_label_isolation.py` parses every module in the perception and
mapping path and fails if any of them touches it.

---

## The cell

27 bytes of record plus an 8-byte Morton key, stored as one NumPy structured
array per level — never a dict of Python objects, which would cost ~350 bytes
of interpreter overhead each and make the memory claim unmeasurable.

```
ground_z f16 | z_max f16 | overhead_clearance f16 | z_var f16 | n_points u16
evidence_top3 7B | entropy u8 | occupancy_logodds i8 | dynamic_prob u8
last_seen u16 | intensity_mean u8 | intensity_var u8 | penetration u8
observability u8 | flags u8
```

**Derived at query time, never stored:** obstacle height, surface normal,
slope, traversability, risk, resolution level.

Three fields deserve their bytes:

* `overhead_clearance` — the bottom of the lowest cluster *above* the vehicle
  clearance band. Without it, tree canopy over a road puts `z_max` at 4 m and
  marks clear road BLOCKED. Two bytes remove a whole failure class.
* `z_max` is the **true maximum inside the clearance band**, not a percentile.
  Max commutes with union, so coarsening stays exact; noise is removed at the
  point level by the isolated-return filter, which is where robustness belongs.
* The **existential obstacle rule**: if *any* point sits above ground + 0.3 m,
  the obstacle layer is set regardless of point count or majority class. A 4 m
  pole returns three points and must never lose a vote. Terrain classification
  stays majoritarian — different rules for different layers, deliberately.

---

## Perception, on a CPU

No CUDA, which rules out every sparse-convolution network. The design puts the
spatial context into the **features** instead, all of them computed from the
range image or the voxel hash with array shifts:

```
height_above_ground · z_rel_local_ground · intensity_norm · local_height_var
local_range_grad · vertical_run · range_r · incidence_cos
penetration_ratio · voxel_neighbour_count · planarity · verticality
```

A 3×80×80×40×6 MLP (11,406 parameters) then learns the decision boundary:
28.6 ms for 55k points, val mIoU 0.871, temperature-scaled so its entropy is
a real uncertainty. Three backends behind one interface —
`geometry_rules` (the honest floor), `pointfeature_net` (default), `oracle`
(evaluation only, and loud about it).

Two signals are worth singling out:

* **Intensity normalisation earns its place.** Inverting the LiDAR equation
  recovers surface reflectance, which separates asphalt (median 0.152) from
  grass (0.361) almost cleanly. Without it the same two surfaces are
  indistinguishable, because a bright surface at 60 m returns less than a dark
  one at 5 m.
* **The vertical-run detector is the safety pin.** Consecutive rings at one
  azimuth, at the same range, *climbing* rather than running outward: 97.6% of
  object points, 0.9% of ground points. It fires without knowing what the
  object is, which is why the retention guarantee survives a segmentation
  failure.

---

## Module map

```
adaptive_lidar/
  main.py                  CLI
  app.py                   Streamlit dashboard
  config.yaml              every constant

  pipeline/
    types.py               the data contract, cell layout, enums
    pipeline.py            S0..S9 orchestration
    timing.py              per-stage timers

  stages/                  s0_ingest .. s9_output

  utils/
    grouping.py            packed keys, Morton, CSR grouping, segment reduce
    spatial_index.py       the frame's ONE sort
    range_image.py         scan unfolding, vertical runs, window stats
    voxel_hash.py          CSR voxel index
    config.py

  perception/
    features.py            12 per-point features
    ground.py              ring-geometry + height-grid segmentation
    backends.py            geometry_rules | pointfeature_net | oracle
    pointfeature_net.py    the MLP and its calibrated inference
    motion.py              range-image residual MOS
    tracking.py            Hungarian + constant-velocity Kalman

  mapping/
    adaptive_map.py        THE MAP
    allocation.py          THE CONTROLLER

  evaluation/
    reference_map.py       the constructed ground truth, with its limits
    metrics.py             every reported number

  data/
    synthetic_scene.py     the raycast sensor model
    loader.py              auto-detecting dataset interface
    label_maps.py          collision-proof label tables

  visualization/
    render.py              map -> image, including a rotated camera
    camera.py              frames of reference and the five presets
    overlays.py            ego, object boxes by state, corridor, age tint,
                           the selection highlight
    playback.py            pre-compute + cache (every MapCell field),
                           swept corridor, wall probe
    coordinate_transform.py  screen <-> world; the EXACT inverse of the
                           renderer's projector
    selection.py           click -> object | cell | empty, finest cell wins
    interactive_map.py     the clickable canvas and its click scaling
    inspector.py           the eight-section cell and object inspectors
    playback_controller.py the transport; no Streamlit in the state machine
    ui_layout.py           header, status badges, CSS, metrics strip
    session.py             the state the two visualisation tabs share
    profile.py             FOVEA_PROFILE=1 diagnostic trace
    live_demo.py           the Live demo tab (one st.fragment)
    scene_data.py          pipeline data -> scene geometry (the adapter)
    scene_component.py     the three.js component, declared from files
    scene_frontend/        index.html, scene.js, vendored three.js
    scene_demo.py          the Scene demo tab
    drive_tab.py           scene presets + the diagnostics panel
  scripts/                 profiling, training, the five verifications,
                           baselines, report generation
  tests/                   248 tests, including the gt-isolation guard
```

---

## The dashboard

Three tabs: **Live demo** (default), **Research & evaluation**, **Debug &
cell inspector**. The demo is first because the engineering readout answers
the second question a reviewer has, not the first.

The Live demo is one `st.fragment`. Everything inside it reads the playback
cache, so advancing a frame, changing the camera or clicking a cell
re-executes only that function — the pipeline lives above it behind
`st.cache_resource` and is never re-entered. With `run_every` set while
playing, the same mechanism animates the scene, which is why there is no
`time.sleep` anywhere in `visualization/` and a test enforces that.

Clicking is real. `coordinate_transform.ViewTransform` is the exact inverse
of `overlays.make_projector` — it undoes the display flip, the pixel scale,
the camera rotation and the vehicle frame — and `selection.py` resolves the
resulting world point to the finest cached cell containing it, with tracked
objects taking priority. `scripts/test_interaction.py` verifies the
round-trip and the click on a real precomputed run.

The replay rate on screen is a replay rate. It is bounded by drawing and
shipping the canvas (~95 ms of server time per displayed frame, ~460 ms end
to end), and it says nothing about pipeline latency, which is reported
separately and stratified by range.

---

## The scene view

The Scene demo draws the same run as the physical environment the map is a
map of. It runs no perception of its own: `scene_data.build_scene_data`
turns one cached frame into geometry — cells become boxes at their real
footprint from `ground_z` to `z_max`, tracked instances become primitives,
the stored pose becomes the vehicle — and three.js puts it on screen.

**No pipeline logic lives in JavaScript.** Positions, sizes, colours and
traversability verdicts are all decided in Python, from the same tables the
2D renderer uses. The browser reports a click back as a world point and an
optional track id; resolving that to a cell is still `selection.py`'s job
against the cached frame, so the two views cannot disagree about what is
under a place.

Where drawn geometry is not measured geometry, it says so. A tracked
object's box is the bounding box of the returns, and a LiDAR sees one side
of a car, so a box smaller than its class floor is enlarged to be visible —
drawn fainter, flagged `padded`, and the inspector prints both the measured
and the drawn extent.

Frame, playback and selection are shared with the Live demo through
`session.py`. The displayed frame is derived from the clock rather than
incremented per redraw, which is what lets two tabs render the same run at
their own rates without playing it twice as fast.

---

## Ego motion and the frame of reference

The map is **world-anchored in both view frames**. The frame selector in the
Drive tab is a transform on the way to pixels and nothing else; there is no
ego-frame storage mode, because storing in the ego frame is precisely what
makes static geometry smear.

World is the default view. In the vehicle frame the entire static scene
slides past, which reads as "everything is moving" — the opposite of the
point — so the vehicle frame is reserved for the moment that illusion is the
subject, which is the `convoy` scenario.

Three things are accumulated outside the map because the map cannot answer
them:

* the **swept corridor**, because the sliding window has already evicted the
  cells behind the vehicle;
* **wall thickness**, a correctness check — a facade accumulated over a whole
  run should still be one cell thick, and would not be if the pose transform
  were wrong;
* **trail length**, which needs ground truth and therefore lives in
  `evaluation/`.

The pipeline runs once per frame when a scenario loads. Camera movement,
layer changes and scrubbing all read the cache, which is why the replay rate
on screen is labelled a replay rate and never as FPS.

---

## What is real and what is a prototype

**Real, trained, measured:** the map and its hierarchy; the allocation
controller and all eleven policies; the per-point semantic network and its
calibration; residual MOS; Hungarian + Kalman tracking; the evaluation harness;
every number in `docs/RESULTS.md`.

**Simulated:** the sensor. The synthetic generator is a genuine raycast model —
64 rings, 2048 azimuth steps, real occlusion, 1/r³ ground falloff, multi-echo,
material reflectance — but it is a model. Real-data support exists
(`get_dataset` reads SemanticKITTI and RELLIS-3D trees) and is untested against
a real sequence, because none was available on this machine. That is stated
rather than implied.

**Not attempted, deliberately:** CUDA, sparse-convolution networks, pretrained
large LiDAR models, a separate 3D detector, SLAM, octrees, camera fusion,
ROS 2, point-cloud compression, DBSCAN, kd-trees. See DECISIONS.md.
