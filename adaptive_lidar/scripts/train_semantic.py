"""
train_semantic.py — train PointFeatureNet (M1).

Trains the per-point semantic MLP on whatever dataset is available; synthetic
always works, so the demo is reproducible on a clone with no data.

    python scripts/train_semantic.py
    python scripts/train_semantic.py --frames 40 --epochs 25 --width 80
    python scripts/train_semantic.py --input dataset/sequences/00

CLASS-BALANCED SAMPLING
-----------------------
VRU and pole points are ~0.3% of a scan.  Under a plain cross-entropy loss the
network reaches 99% accuracy by never predicting them, which is precisely the
failure this project exists to avoid.  Training therefore draws an equal number
of points per class per batch, and the reported metric is per-class IoU rather
than accuracy.

TEMPERATURE SCALING
-------------------
A network trained with cross-entropy is systematically overconfident, so its
raw softmax is not a probability.  Every uncertainty claim downstream (the U
term in the allocation value function, the per-cell entropy layer) depends on
it being one, so a single temperature is fitted on a held-out split by
minimising NLL, and ECE before and after is written to docs/calibration.json.
"""
from __future__ import annotations

import argparse
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

import torch                                                        # noqa: E402
import torch.nn as nn                                               # noqa: E402

from adaptive_lidar.data.loader import get_dataset                  # noqa: E402
from adaptive_lidar.perception.features import (                    # noqa: E402
    FEATURE_NAMES,
    N_FEATURES,
    extract_features,
    normalise_features,
)
from adaptive_lidar.perception.pointfeature_net import (            # noqa: E402
    DEFAULT_CHECKPOINT,
    PointFeatureNet,
)
from adaptive_lidar.pipeline.pipeline import Pipeline               # noqa: E402
from adaptive_lidar.pipeline.types import CLASS_NAMES, NUM_CLASSES  # noqa: E402
from adaptive_lidar.utils.config import load_config                 # noqa: E402


# ════════════════════════════════════════════════════════════
# Dataset assembly
# ════════════════════════════════════════════════════════════
def collect(config, n_frames, root=None, scenarios=("mixed_urban",), seed=42):
    """Run S0-S2 on each frame and harvest (features, gt_label).

    Only the stages the features depend on are run — the allocation, map and
    tracking stages have no bearing on what the classifier sees.
    """
    X, Y, R = [], [], []
    for scen in scenarios:
        ds = get_dataset("synthetic" if root is None else "auto", root,
                         n_frames, scenario=scen, seed=seed, quiet=True)
        pipe = Pipeline(config)
        # geometry_rules avoids needing a checkpoint that does not exist yet.
        pipe.build_stages(backend="geometry_rules")
        for f in ds:
            frame = pipe._ingest(path=None, frame_id=f["frame_id"],
                                 timestamp=f["timestamp"], config=config,
                                 synthetic_cloud=f)
            pipe._s1.process(frame, pipe.context)
            pipe._s2.process(frame, pipe.context)
            feats = extract_features(frame)
            gt = frame.gt_label
            keep = np.asarray(gt) >= 0
            if not keep.any():
                continue
            X.append(feats[keep])
            Y.append(np.asarray(gt)[keep].astype(np.int64))
            R.append(frame._range[keep])
        print(f"  {scen:<18} {sum(len(a) for a in X):>9,} points collected")
    return (np.concatenate(X), np.concatenate(Y), np.concatenate(R))


def balanced_indices(y, n_per_class, rng):
    """Draw n_per_class indices per class, with replacement where rare."""
    out = []
    for c in range(NUM_CLASSES):
        idx = np.flatnonzero(y == c)
        if idx.size == 0:
            continue
        out.append(rng.choice(idx, n_per_class, replace=idx.size < n_per_class))
    return np.concatenate(out) if out else np.empty(0, np.int64)


# ════════════════════════════════════════════════════════════
# Metrics
# ════════════════════════════════════════════════════════════
def per_class_iou(pred, true):
    ious, present = [], []
    for c in range(NUM_CLASSES):
        p, t = pred == c, true == c
        union = (p | t).sum()
        if t.sum() == 0:
            ious.append(float("nan"))
            continue
        ious.append(float((p & t).sum() / max(union, 1)))
        present.append(ious[-1])
    return ious, (float(np.mean(present)) if present else 0.0)


def expected_calibration_error(probs, true, n_bins=15):
    """Standard ECE: |accuracy - confidence| averaged over confidence bins."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == true).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if not m.any():
            continue
        ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def fit_temperature(logits, true, iters=250):
    """One scalar, fitted by minimising NLL on a held-out split."""
    lg = torch.from_numpy(logits.astype(np.float32))
    ty = torch.from_numpy(true.astype(np.int64))
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=iters)
    lossf = nn.CrossEntropyLoss()

    def closure():
        opt.zero_grad()
        loss = lossf(lg / torch.exp(log_t), ty)
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.exp(log_t).item())


# ════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--width", type=int, default=80,
                    help="hidden width; 80 keeps inference under 30 ms "
                         "for 55k points on CPU")
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--input", default=None)
    ap.add_argument("--out", default=DEFAULT_CHECKPOINT)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
    rng = np.random.default_rng(args.seed)
    config = load_config()

    print("=" * 70)
    print("  TRAINING PointFeatureNet  (M1 — per-point semantic segmentation)")
    print("=" * 70)
    print(f"  features ({N_FEATURES}): {', '.join(FEATURE_NAMES)}\n")

    t0 = time.perf_counter()
    print("Collecting training data...")
    scen = ("mixed_urban", "canopy_over_road", "pedestrian_far", "moving_vehicle")
    X, Y, R = collect(config, args.frames, args.input, scen, seed=42)
    # A disjoint scene seed for validation: a held-out split drawn from the
    # same frames would share neighbourhoods with training and flatter both
    # accuracy and calibration.
    Xv, Yv, Rv = collect(config, max(args.frames // 3, 2), args.input,
                         ("mixed_urban",), seed=1234)
    print(f"\n  train {len(X):,} points   validation {len(Xv):,} points")

    counts = np.bincount(Y, minlength=NUM_CLASSES)
    print("\n  class balance (train):")
    for c in range(NUM_CLASSES):
        print(f"    {c} {CLASS_NAMES[c]:<18} {counts[c]:>9,}"
              f"  {100 * counts[c] / len(Y):6.3f}%")

    Xn = normalise_features(X)
    Xvn = normalise_features(Xv)

    model = PointFeatureNet(n_in=N_FEATURES, n_out=NUM_CLASSES, width=args.width)
    print(f"\n  model: 3 hidden layers x {args.width}, "
          f"{model.n_params:,} parameters")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    lossf = nn.CrossEntropyLoss()

    n_per_class = args.batch // NUM_CLASSES
    steps = max(len(X) // args.batch, 40)
    xv_t = torch.from_numpy(Xvn)

    print(f"\n  {'epoch':>6}{'loss':>10}{'val mIoU':>10}{'vru IoU':>10}{'sec':>8}")
    best = -1.0
    best_state = None
    for ep in range(args.epochs):
        model.train()
        te = time.perf_counter()
        tot = 0.0
        for _ in range(steps):
            idx = balanced_indices(Y, n_per_class, rng)
            xb = torch.from_numpy(Xn[idx])
            yb = torch.from_numpy(Y[idx])
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
            tot += float(loss.item())
        sched.step()

        model.eval()
        with torch.no_grad():
            pv = model(xv_t).argmax(1).numpy()
        ious, miou = per_class_iou(pv, Yv)
        vru = ious[4] if not np.isnan(ious[4]) else 0.0
        print(f"  {ep + 1:>6}{tot / steps:>10.4f}{miou:>10.4f}{vru:>10.4f}"
              f"{time.perf_counter() - te:>8.1f}")
        if miou > best:
            best = miou
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    # ── calibration ──────────────────────────────────────────
    print("\n  Fitting temperature on the held-out split...")
    with torch.no_grad():
        logits = model(xv_t).numpy()
    p_raw = torch.softmax(torch.from_numpy(logits), 1).numpy()
    ece_before = expected_calibration_error(p_raw, Yv)
    T = fit_temperature(logits, Yv)
    p_cal = torch.softmax(torch.from_numpy(logits) / T, 1).numpy()
    ece_after = expected_calibration_error(p_cal, Yv)
    print(f"    temperature = {T:.4f}")
    print(f"    ECE before  = {ece_before:.4f}")
    print(f"    ECE after   = {ece_after:.4f}")

    # ── final report ─────────────────────────────────────────
    pred = p_cal.argmax(1)
    ious, miou = per_class_iou(pred, Yv)
    print("\n  per-class IoU (validation, calibrated):")
    for c in range(NUM_CLASSES):
        v = ious[c]
        print(f"    {c} {CLASS_NAMES[c]:<18} "
              + ("   n/a" if np.isnan(v) else f"{v:6.4f}"))
    print(f"    {'mIoU':<22}{miou:6.4f}")

    # inference speed at the operating point
    n_probe = min(55_000, len(Xvn))
    xb = torch.from_numpy(Xvn[:n_probe])
    with torch.no_grad():
        model(xb[:1000])
        t = time.perf_counter()
        model(xb)
        ms = (time.perf_counter() - t) * 1000
    print(f"\n  inference: {ms:.1f} ms for {n_probe:,} points "
          f"({os.cpu_count()} cores, {torch.get_num_threads()} threads)")

    # ── save ─────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "feature_names": list(FEATURE_NAMES),
        "width": args.width,
        "temperature": T,
        "n_params": model.n_params,
        "classes": CLASS_NAMES,
        "trained_on": args.input or f"synthetic {list(scen)}",
        "train_points": int(len(X)),
        "val_points": int(len(Xv)),
        "val_miou": float(miou),
        "val_iou_per_class": [None if np.isnan(v) else float(v) for v in ious],
        "ece_before": ece_before,
        "ece_after": ece_after,
        "inference_ms_55k": float(ms * 55_000 / max(n_probe, 1)),
        "epochs": args.epochs,
        "seed": args.seed,
    }, args.out)
    print(f"\n  saved {args.out}")

    docs = os.path.join(_PKG, "docs")
    os.makedirs(docs, exist_ok=True)
    with open(os.path.join(docs, "calibration.json"), "w") as fh:
        json.dump({
            "temperature": T,
            "ece_before": ece_before,
            "ece_after": ece_after,
            "n_bins": 15,
            "val_points": int(len(Xv)),
            "val_miou": float(miou),
            "val_iou_per_class": {
                CLASS_NAMES[c]: (None if np.isnan(ious[c]) else float(ious[c]))
                for c in range(NUM_CLASSES)},
            "method": "single-parameter temperature scaling, LBFGS on held-out NLL",
            "note": ("Raw softmax from a cross-entropy-trained network is "
                     "systematically overconfident. The allocation "
                     "controller's uncertainty term and the map's entropy "
                     "layer both read this distribution, so it is calibrated "
                     "before either sees it."),
        }, fh, indent=2)
    print(f"  saved {os.path.join(docs, 'calibration.json')}")
    print(f"\n  total {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
