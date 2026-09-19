"""
eval_semantic.py — Phase 3 verification.

Per-class IoU and mIoU, STRATIFIED BY RANGE BAND, for all three semantic
backends, plus ECE before and after calibration.  Writes docs/semantic_eval.csv.

Range stratification is the point.  An aggregate mIoU is dominated by the
near field, where 70% of a LiDAR's points are and where nothing is hard; the
number that matters for this project is what happens at 60-100 m, where an
object is six points.

    python scripts/eval_semantic.py
    python scripts/eval_semantic.py --frames 8 --backends geometry_rules,pointfeature_net
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
for p in (os.path.dirname(_PKG), _PKG):
    if p not in sys.path:
        sys.path.insert(0, p)

from adaptive_lidar.data.loader import get_dataset                 # noqa: E402
from adaptive_lidar.evaluation.metrics import (                    # noqa: E402
    BAND_NAMES,
    range_stratified_iou,
)
from adaptive_lidar.perception.backends import BACKEND_NAMES       # noqa: E402
from adaptive_lidar.pipeline.pipeline import Pipeline              # noqa: E402
from adaptive_lidar.pipeline.types import CLASS_NAMES, NUM_CLASSES  # noqa: E402
from adaptive_lidar.utils.config import load_config                # noqa: E402


def ece(probs, true, n_bins=15):
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    ok = (pred == true).astype(np.float64)
    edges = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(ok[m].mean() - conf[m].mean())
    return float(e)


def run(cfg, frames, backend):
    pipe = Pipeline(cfg)
    pipe.build_stages(backend=backend, policy="full")
    pipe.set_budget(0.5)
    P, T, R, EV = [], [], [], []
    t0 = time.perf_counter()
    ms = []
    for f in frames:
        fr = pipe.run(cloud_np=f, frame_id=f["frame_id"], timestamp=f["timestamp"])
        if fr.gt_label is None:
            continue
        P.append(np.asarray(fr.sem_class))
        T.append(np.asarray(fr.gt_label))
        R.append(fr._range)
        EV.append(np.asarray(fr.sem_evidence))
        ms.append(float(fr.timing.get("S4", 0.0)))
    return (pipe.context.semantic_backend_name,
            np.concatenate(P), np.concatenate(T), np.concatenate(R),
            np.concatenate(EV), float(np.median(ms[1:] or ms)),
            time.perf_counter() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--scenario", default="mixed_urban")
    ap.add_argument("--backends", default=",".join(BACKEND_NAMES))
    ap.add_argument("--out", default=os.path.join(_PKG, "docs", "semantic_eval.csv"))
    args = ap.parse_args()

    cfg = load_config()
    frames = list(get_dataset("synthetic", None, args.frames,
                              scenario=args.scenario, quiet=True))
    cal = {}
    calp = os.path.join(_PKG, "docs", "calibration.json")
    if os.path.isfile(calp):
        with open(calp) as fh:
            cal = json.load(fh)

    rows = []
    print("=" * 96)
    print(f"  SEMANTIC EVALUATION — {args.scenario}, {len(frames)} frames, "
          f"{sum(len(f['points']) for f in frames):,} points")
    print("=" * 96)

    for backend in args.backends.split(","):
        backend = backend.strip()
        try:
            name, pred, true, rng, ev, s4_ms, wall = run(cfg, frames, backend)
        except Exception as e:
            print(f"\n  {backend}: unavailable ({type(e).__name__}: {e})")
            continue

        res = range_stratified_iou(pred, true, rng)
        valid = true >= 0
        p = ev[valid] / np.maximum(ev[valid].sum(1, keepdims=True), 1e-9)
        e_cal = ece(p, true[valid].astype(np.int64))

        banner = ("  *** ORACLE — ground-truth labels, NOT a prediction. "
                  "Measures the MAP, never the segmenter. ***"
                  if backend == "oracle" else "")
        print(f"\n  backend: {name}{banner}")
        print(f"  S4 latency (median): {s4_ms:.1f} ms/frame")
        print(f"\n  {'class':<18}" + "".join(f"{b:>10}" for b in BAND_NAMES)
              + f"{'overall':>10}{'support':>12}")
        for c in range(NUM_CLASSES):
            cells = []
            for b in BAND_NAMES:
                bd = res["bands"].get(b)
                v = bd["iou"][c] if bd else float("nan")
                cells.append("       n/a" if not np.isfinite(v) else f"{v:>10.4f}")
            ov = res["overall"]["iou"][c]
            print(f"  {CLASS_NAMES[c]:<18}" + "".join(cells)
                  + ("       n/a" if not np.isfinite(ov) else f"{ov:>10.4f}")
                  + f"{res['overall']['support'][c]:>12,}")
        print(f"  {'mIoU':<18}"
              + "".join(f"{(res['bands'][b]['miou'] if res['bands'].get(b) else float('nan')):>10.4f}"
                        for b in BAND_NAMES)
              + f"{res['overall']['miou']:>10.4f}")
        print(f"  {'accuracy':<18}"
              + "".join(f"{(res['bands'][b]['accuracy'] if res['bands'].get(b) else float('nan')):>10.4f}"
                        for b in BAND_NAMES)
              + f"{res['overall']['accuracy']:>10.4f}")
        print(f"  {'points':<18}"
              + "".join(f"{(res['bands'][b]['n'] if res['bands'].get(b) else 0):>10,}"
                        for b in BAND_NAMES)
              + f"{res['overall']['n']:>10,}")

        row = {"backend": name, "requested": backend,
               "miou": res["overall"]["miou"],
               "accuracy": res["overall"]["accuracy"],
               "ece": e_cal, "s4_ms": s4_ms,
               "n_points": res["overall"]["n"]}
        for b in BAND_NAMES:
            bd = res["bands"].get(b)
            row[f"miou_{b}"] = bd["miou"] if bd else float("nan")
            row[f"acc_{b}"] = bd["accuracy"] if bd else float("nan")
            row[f"n_{b}"] = bd["n"] if bd else 0
        for c in range(NUM_CLASSES):
            row[f"iou_{CLASS_NAMES[c]}"] = res["overall"]["iou"][c]
            for b in BAND_NAMES:
                bd = res["bands"].get(b)
                row[f"iou_{CLASS_NAMES[c]}_{b}"] = bd["iou"][c] if bd else float("nan")
        rows.append(row)

    if cal:
        print(f"\n  CALIBRATION (pointfeature_net, from training)")
        print(f"    temperature : {cal.get('temperature', float('nan')):.4f}")
        print(f"    ECE before  : {cal.get('ece_before', float('nan')):.4f}")
        print(f"    ECE after   : {cal.get('ece_after', float('nan')):.4f}")
        print(f"    Raw softmax from a cross-entropy-trained network is "
              f"overconfident. The\n    allocation controller's uncertainty "
              f"term and the map's entropy layer both\n    read this "
              f"distribution, so it is calibrated before either sees it.")
        for r in rows:
            if r["requested"] == "pointfeature_net":
                r["ece_before_train"] = cal.get("ece_before")
                r["ece_after_train"] = cal.get("ece_after")
                r["temperature"] = cal.get("temperature")

    if rows:
        keys = sorted({k for r in rows for k in r})
        head = ["backend", "requested", "miou"] + \
               [f"miou_{b}" for b in BAND_NAMES] + ["accuracy", "ece", "s4_ms"]
        cols = head + [k for k in keys if k not in head]
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        print(f"\n  Wrote {args.out}")


if __name__ == "__main__":
    main()
