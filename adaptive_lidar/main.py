"""
main.py - CLI entry point.

  python main.py --demo
  python main.py --scenario pedestrian_far --frames 10
  python main.py --input dataset/sequences/00 --frames 100
  python main.py --budget 25 --policy uniform_40 --backend geometry_rules
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
for p in (_PARENT, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger("main")


def parse_args():
    from adaptive_lidar.data.synthetic_scene import SCENARIOS
    from adaptive_lidar.mapping.allocation import AllocationController
    from adaptive_lidar.perception.backends import BACKEND_NAMES

    p = argparse.ArgumentParser(
        description="Adaptive Variable-Resolution 2.5D LiDAR Mapping (DRDO PS 26053)",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--demo", action="store_true", help="run the default synthetic demo")
    p.add_argument("--synthetic", action="store_true", help="force synthetic data")
    p.add_argument("--scenario", default="mixed_urban", choices=SCENARIOS)
    p.add_argument("--input", default=None,
                   help="SemanticKITTI/RELLIS sequence dir, or a dir of .bin/.npy")
    p.add_argument("--frames", type=int, default=10)
    p.add_argument("--budget", type=float, default=60.0, help="cell budget %%")
    p.add_argument("--policy", default="full", choices=AllocationController.POLICIES)
    p.add_argument("--backend", default="auto",
                   choices=("auto",) + tuple(BACKEND_NAMES))
    p.add_argument("--config", default=None)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def banner():
    print("\n" + "=" * 74)
    print("  ADAPTIVE VARIABLE-RESOLUTION 2.5D LiDAR MAPPING")
    print("  SIH 2026 / DRDO PS 26053 - dynamic environment perception")
    print("=" * 74 + "\n")


def frame_line(frame):
    t = frame.timing.get("telemetry", {})
    names = t.get("level_names", ["5cm", "10cm", "20cm", "40cm", "80cm"])
    cpl = t.get("cells_per_level", {})
    tot = max(sum(cpl.values()), 1)
    dist = " ".join(f"{names[int(k)]}:{v:,}" for k, v in sorted(cpl.items()) if v)
    print(f"  Frame {frame.frame_id:04d} | {t.get('point_count', 0):,} pts | "
          f"{t.get('total_latency_ms', 0):.1f} ms ({t.get('fps', 0):.1f} FPS)")
    print(f"    cells by size       : {dist}")
    print(f"    map cells           : {t.get('map_cells', 0):,} "
          f"({t.get('map_mb', 0):.2f} MB)  vs uniform 5 cm "
          f"{t.get('uniform5_cells', 0):,} ({t.get('uniform5_mb', 0):.2f} MB) "
          f"-> {t.get('memory_reduction_pct', 0):.1f}% smaller")
    print(f"    objects             : {t.get('n_instances', 0)} tracked, "
          f"{t.get('n_moving', 0)} moving, "
          f"{t.get('safety_pinned', 0)} tiles safety-pinned")
    ms = t.get("stage_ms", {})
    print("    " + " ".join(f"{k}:{v:.1f}" for k, v in ms.items()))
    print()


def summary(pipe, wall, n_frames):
    from adaptive_lidar.pipeline.types import ResolutionLevel
    ctx = pipe.context
    amap = ctx.amap
    t = ctx.telemetry

    print("=" * 74)
    print("  SUMMARY")
    print("=" * 74)
    print(f"  Frames            : {n_frames} in {wall:.2f} s "
          f"({n_frames / max(wall, 1e-6):.1f} FPS wall clock)")
    print(f"  Data source       : {ctx.data_source}")
    print(f"  Semantic backend  : {ctx.semantic_backend_name}")
    print(f"  Allocation policy : {ctx.allocation_policy}  @ budget "
          f"{ctx.budget * 100:.0f}%")
    print()
    print("  MAP CELLS BY PHYSICAL SIZE  (M4 - variable resolution)")
    counts = amap.cell_counts() if amap else {}
    total = max(sum(counts.values()), 1)
    for lvl in range(5):
        c = counts.get(lvl, 0)
        bar = "#" * int(40 * c / total)
        print(f"    {ResolutionLevel.name(lvl):>5s}  {c:>9,}  "
              f"{100 * c / total:5.1f}%  {bar}")
    print(f"    TOTAL  {total:>9,}")
    print()
    print("  MEMORY  (M5)")
    print(f"    adaptive map      : {t.get('map_mb', 0):8.2f} MB "
          f"({t.get('map_cells', 0):,} cells)")
    print(f"    uniform 5 cm      : {t.get('uniform5_mb', 0):8.2f} MB "
          f"({t.get('uniform5_cells', 0):,} cells)")
    print(f"    reduction         : {t.get('memory_reduction_pct', 0):8.1f} %")
    print()
    print("  PER-STAGE LATENCY  (M6)")
    print(f"    {'stage':<8}{'p50':>9}{'p95':>9}{'p99':>9}")
    tot50 = 0.0
    for s in ("S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9"):
        if s not in ctx.cumulative_timing:
            continue
        p50, p95, p99 = ctx.p50(s), ctx.p95(s), ctx.p99(s)
        tot50 += p50
        print(f"    {s:<8}{p50:>9.2f}{p95:>9.2f}{p99:>9.2f}")
    print(f"    {'TOTAL':<8}{tot50:>9.2f}")
    print(f"    -> {1000 / max(tot50, 1e-6):.1f} FPS at "
          f"{t.get('point_count', 0):,} points, CPU only")
    print("=" * 74)
    print("\n  Dashboard:  streamlit run app.py")
    print("  Report:     python scripts/run_baselines.py && "
          "python scripts/generate_report.py\n")


def main():
    args = parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    banner()

    from adaptive_lidar.data.loader import get_dataset
    from adaptive_lidar.pipeline.pipeline import Pipeline
    from adaptive_lidar.utils.config import load_config

    config = load_config(args.config)

    use_synth = args.demo or args.synthetic or args.input is None
    ds = get_dataset(
        name="synthetic" if use_synth else "auto",
        root=None if use_synth else args.input,
        max_frames=args.frames,
        scenario=args.scenario,
        seed=config.get("synthetic", {}).get("seed", 42),
    )

    pipe = Pipeline(config)
    pipe.build_stages(backend=args.backend, policy=args.policy)
    pipe.set_budget(args.budget / 100.0)
    pipe.context.data_source = ds.name

    logger.info("Backend: %s | policy: %s | budget: %.0f%%",
                pipe.context.semantic_backend_name, args.policy, args.budget)
    print("\nProcessing frames...\n")

    t0 = time.perf_counter()
    for f in ds:
        frame = pipe.run(cloud_np=f, frame_id=f["frame_id"], timestamp=f["timestamp"])
        frame_line(frame)
    wall = time.perf_counter() - t0

    summary(pipe, wall, pipe.frame_count)


if __name__ == "__main__":
    main()
