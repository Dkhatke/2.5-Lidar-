"""
main.py — CLI entry point for the SIH26 adaptive LiDAR pipeline.

Usage:
  python main.py --demo
  python main.py --synthetic
  python main.py --input dataset/sequences/00 --frames 100
  python main.py --budget 50 --backend auto
  python main.py --input file.bin
"""
from __future__ import annotations
import argparse
import logging
import sys
import os
import time
import numpy as np

# Make sure the parent of adaptive_lidar/ is on the path
_HERE = os.path.dirname(os.path.abspath(__file__))          # …/adaptive_lidar
_PARENT = os.path.dirname(_HERE)                             # …/SIH26 2.5 Lidar
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("main")


def parse_args():
    p = argparse.ArgumentParser(
        description="SIH26 — Adaptive Foveated 2.5D LiDAR Perception",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --demo
  python main.py --synthetic
  python main.py --input dataset/sequences/00 --frames 100
  python main.py --budget 50 --backend prototype
        """,
    )
    p.add_argument("--demo", action="store_true",
                   help="Run synthetic demo (alias for --synthetic)")
    p.add_argument("--synthetic", action="store_true",
                   help="Use synthetic LiDAR scene")
    p.add_argument("--input", type=str, default=None,
                   help="Path to SemanticKITTI sequence dir or single .bin/.npy file")
    p.add_argument("--frames", type=int, default=8,
                   help="Max frames to process (default: 8)")
    p.add_argument("--budget", type=float, default=80.0,
                   help="Computation budget %% (10–100, default: 80)")
    p.add_argument("--backend", type=str, default="auto",
                   choices=["auto", "minkowski", "prototype", "geometry"],
                   help="Semantic backend selection (default: auto)")
    p.add_argument("--config", type=str, default=None,
                   help="Path to config.yaml (default: auto-detect)")
    p.add_argument("--verbose", action="store_true",
                   help="Debug-level logging")
    return p.parse_args()


def print_banner():
    print("\n" + "=" * 62)
    print("  SIH26 — Adaptive Foveated 2.5D LiDAR Perception")
    print("  Prototype: intelligent computation allocation")
    print("=" * 62 + "\n")


def print_frame_summary(frame, ctx):
    t = frame.timing
    telem = t.get("telemetry", {})
    total_ms = telem.get("total_latency_ms", 0.0)
    n_tiles = telem.get("total_tiles", 0)
    n_hi = telem.get("high_res_tiles", 0)
    n_med = telem.get("medium_tiles", 0)
    n_low = telem.get("low_res_tiles", 0)
    n_pts = telem.get("point_count", 0)
    backend = telem.get("backend", "unknown")

    pct_hi = 100 * n_hi / max(n_tiles, 1)
    pct_med = 100 * n_med / max(n_tiles, 1)
    pct_low = 100 * n_low / max(n_tiles, 1)

    print(f"  Frame {frame.frame_id:04d} | {n_pts:,} pts | {total_ms:.1f} ms total")
    print(f"  Backend: {backend}")
    print(f"  Tiles: {n_tiles} total | "
          f"HIGH {n_hi} ({pct_hi:.0f}%) | "
          f"MED {n_med} ({pct_med:.0f}%) | "
          f"LOW {n_low} ({pct_low:.0f}%)")
    # Stage breakdown
    stages = ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"]
    timings = " | ".join(
        f"{s}:{t.get(s, 0.0):.1f}ms" for s in stages if s in t)
    print(f"  {timings}")
    print()


def main():
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    print_banner()

    # Load config
    from adaptive_lidar.utils.config import load_config
    config = load_config(args.config)

    # Build pipeline
    from adaptive_lidar.pipeline.pipeline import Pipeline
    pipe = Pipeline(config)
    pipe.build_stages(backend=args.backend)
    pipe.set_budget(args.budget / 100.0)

    logger.info(f"Backend: {pipe.context.semantic_backend_name}")
    logger.info(f"Budget:  {args.budget:.0f}%")

    # Get data source
    from adaptive_lidar.data.loader import get_input_source

    input_path = args.input
    use_synthetic = args.demo or args.synthetic or (input_path is None)
    if use_synthetic:
        input_path = None

    source = get_input_source(
        input_path=input_path,
        max_frames=args.frames,
        num_synthetic_frames=args.frames,
    )

    print(f"Processing frames...\n")
    wall_start = time.perf_counter()

    for cloud, labels, frame_id, timestamp in source:
        frame = pipe.run(
            cloud_np=cloud,
            frame_id=frame_id,
            timestamp=timestamp,
        )
        print_frame_summary(frame, pipe.context)

    wall_total = time.perf_counter() - wall_start
    n_frames = pipe.frame_count
    fps = n_frames / wall_total if wall_total > 0 else 0

    print("=" * 62)
    print(f"  Processed {n_frames} frames in {wall_total:.2f}s  ({fps:.1f} FPS)")
    print(f"  Map cells: {len(pipe.map_cells):,}")
    print(f"  TARGET: ~40 ms/frame  |  ACTUAL: {1000*wall_total/max(n_frames,1):.1f} ms/frame")
    print("=" * 62)
    print("\n  Run 'streamlit run app.py' for the visual dashboard.\n")


if __name__ == "__main__":
    main()
