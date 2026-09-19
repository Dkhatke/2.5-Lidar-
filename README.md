# SIH26 — Adaptive Foveated 2.5D LiDAR Perception

> **Smart Horizons Innovation Challenge 2026**  
> Prototype: Intelligent Adaptive 2.5D LiDAR Mapping

---

## What This Demonstrates

Instead of running expensive perception uniformly across the entire LiDAR scene,
this system:

1. Ingests LiDAR data (SemanticKITTI or synthetic)
2. Builds cheap spatial representations (range image + voxel hash)
3. Computes geometric features per tile (density, height variance, roughness…)
4. Scores each tile by **information value** — geometry × uncertainty × motion × semantic relevance
5. Allocates high-resolution sparse processing **only to high-value tiles**
6. Runs sparse semantic perception selectively (MinkowskiEngine → PyTorch fallback → geometry fallback)
7. Estimates motion + instances across frames
8. Updates an **adaptive-resolution 2.5D map** where cells carry full semantic probability vectors
9. Fuses temporally and outputs telemetry

```
ENTIRE LiDAR SCENE
        │
        ▼
  CHEAP GLOBAL ANALYSIS     (S1 + S2)
        │
        ▼
  INFORMATION VALUE SCORE   (S3)
   geometry + uncertainty
   + semantic + motion
        │
        ▼
  ADAPTIVE COMPUTE BUDGET   (S3 / S6)
  COARSE ── MEDIUM ── FINE
        │
        ▼
  SPARSE PERCEPTION         (S4)   ← only selected tiles enter here
        │
   ┌────┴────┐
   ▼         ▼
SEMANTIC   MOTION           (S4 + S5)
   │         │
   └────┬────┘
        ▼
  UNCERTAINTY FEEDBACK      (S6)
        │
        ▼
  ADAPTIVE 2.5D MAP         (S7 + S8)
        │
        ▼
  DASHBOARD / TELEMETRY     (S9)
```

---

## Quick Start

```bash
# 1 — create environment
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2 — install dependencies
pip install -r adaptive_lidar/requirements.txt

# 3 — run dashboard (synthetic demo — no dataset needed)
cd adaptive_lidar
streamlit run app.py

# 4 — CLI demo
python main.py --demo
python main.py --synthetic
python main.py --input dataset/sequences/00 --frames 100
python main.py --budget 50 --backend auto
```

---

## Architecture: S0–S9 Stages

| Stage | Name | Description |
|-------|------|-------------|
| **S0** | Ingest | Load .bin / .npy / .ply / synthetic; filter range + NaN |
| **S1** | Indices | Range image + voxel hash (query index, NOT the map) |
| **S2** | Geometry | Ground separation; per-tile density/roughness/verticality |
| **S3** | Pre-allocation | Information value score; assign resolution levels |
| **S4** | Semantics | Sparse perception on selected tiles only |
| **S5** | Motion | Instance extraction; centroid velocity; dynamic layer |
| **S6** | Allocation | Refine resolution with S4 uncertainty + S5 motion |
| **S7** | Map update | Adaptive 2.5D map; log-odds occupancy; semantic probs |
| **S8** | Temporal fusion | Weighted fusion; temporal decay; pose compensation |
| **S9** | Output | Telemetry collation; async-ready interface |

---

## Dataset

**Primary**: [SemanticKITTI](https://semantic-kitti.org/dataset.html)

```
dataset/sequences/00/velodyne/000000.bin
dataset/sequences/00/labels/000000.label
dataset/sequences/00/poses.txt
```

**Fallback**: built-in synthetic scene (auto-activated when no dataset path given)

---

## Semantic Backend

The system never crashes because of a missing dependency:

```
SparseSemanticBackend
├─ 1. MinkowskiBackend        (real sparse convolution — optional)
├─ 2. PrototypeSparseBackend  (lightweight PyTorch — always available)
└─ 3. GeometryFallbackBackend (pure numpy rules — always available)
```

The active backend is always shown in the dashboard sidebar.

---

## Honest Prototype Labels

| What you see | What it means |
|---|---|
| "Ground separation — prototype height-grid" | Future: Patchwork++ |
| "Semantic backend — Prototype Sparse" | Future: MinkUNet / SPVCNN |
| "Motion — centroid displacement" | Future: 4DMOS |
| "Tracking — nearest centroid" | Future: Hungarian + Kalman |
| "Fusion — weighted average" | Future: probabilistic elevation mapping |

---

## Configuration

All constants live in `config.yaml` — no magic numbers in code.

```yaml
budget:
  default: 0.80           # 80% compute budget by default
weights:
  geometry: 0.35
  semantic: 0.20
  uncertainty: 0.25
  dynamic: 0.20
tiles:
  size: 2.0               # tile edge length (metres)
```

---

## Running Tests

```bash
python -m pytest tests/ -v
```

---

## Future Upgrade Path

| Prototype | Research replacement |
|-----------|---------------------|
| Height-grid ground seg | Patchwork++ |
| PrototypeSparseBackend | MinkUNet / SPVCNN |
| Centroid motion | 4DMOS |
| Python pipeline | C++/CUDA critical path |
| Streamlit | ROS2 + RViz |

Upgrading any module does **not** require changing other stages — interfaces are stable.

---

## Project Structure

```
adaptive_lidar/
├── app.py                  ← Streamlit dashboard
├── main.py                 ← CLI entry point
├── config.yaml             ← All configuration
├── requirements.txt
├── data/
│   ├── loader.py           ← SemanticKITTI loader
│   └── synthetic_scene.py  ← Synthetic fallback
├── pipeline/
│   ├── pipeline.py         ← Orchestrator
│   ├── types.py            ← Frame, Tile, MapCell, Context
│   └── timing.py           ← Stage timers
├── stages/
│   ├── s0_ingest.py … s9_output.py
├── perception/
│   ├── sparse_backend.py   ← Auto-detect + fallback chain
│   ├── minkowski_backend.py
│   ├── prototype_sparse_backend.py
│   └── geometry_fallback.py
├── mapping/
│   ├── adaptive_map.py
│   ├── elevation.py
│   ├── occupancy.py
│   └── semantic_layers.py
├── visualization/
│   ├── dashboard.py
│   ├── pointcloud_view.py
│   ├── range_view.py
│   ├── allocation_view.py
│   └── map_view.py
├── utils/
│   ├── voxel_hash.py
│   ├── range_image.py
│   └── config.py
├── tests/
└── docs/
    ├── DEPENDENCIES.md
    └── INSTALL_MINKOWSKI.md
```
