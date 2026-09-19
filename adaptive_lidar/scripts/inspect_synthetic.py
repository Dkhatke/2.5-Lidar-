"""
inspect_synthetic.py — Phase 2 verification.

Confirms that the raycast sensor model produces a scene the rest of the
project can actually be evaluated on: the right classes in the right places,
the 1/r^3 density falloff that motivates the whole resolution law, and the
70 m pedestrian the demo is built around.

Writes docs/synthetic_scene.png.

    python scripts/inspect_synthetic.py
    python scripts/inspect_synthetic.py --scenario canopy_over_road
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
for p in (os.path.dirname(_PKG), _PKG):
    if p not in sys.path:
        sys.path.insert(0, p)

import matplotlib                                                  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib.colors import ListedColormap                       # noqa: E402

from adaptive_lidar.data.synthetic_scene import (                  # noqa: E402
    FAR_PEDESTRIAN_X,
    FAR_PEDESTRIAN_Y,
    SCENARIOS,
    generate_scenario,
)
from adaptive_lidar.evaluation.metrics import BAND_NAMES, RANGE_BANDS  # noqa: E402
from adaptive_lidar.pipeline.types import CLASS_NAMES              # noqa: E402

# Colour-blind-safe, terrain cool / objects warm, VRU unmissable.
CLASS_COLOURS = ["#4C6EA8", "#7FA05A", "#8C8C8C", "#E8A33D", "#D62728", "#2E8B57"]
FAILURES = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  —  {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="mixed_urban", choices=SCENARIOS)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(_PKG, "docs", "synthetic_scene.png"))
    args = ap.parse_args()

    s = generate_scenario(args.scenario, args.frame)
    p = s["points"]
    gt = np.asarray(s["gt_label"])
    r = np.linalg.norm(p, axis=1)
    n = len(p)

    print("=" * 74)
    print(f"  SYNTHETIC SCENE — {args.scenario}, frame {args.frame}")
    print("=" * 74)
    print(f"  points          : {n:,}")
    print(f"  extent          : x [{p[:, 0].min():7.1f}, {p[:, 0].max():7.1f}] "
          f"y [{p[:, 1].min():7.1f}, {p[:, 1].max():7.1f}] "
          f"z [{p[:, 2].min():6.2f}, {p[:, 2].max():6.2f}]")
    print(f"  rings used      : {len(np.unique(s['ring']))} "
          f"of {int(s['ring'].max()) + 1}")
    print(f"  azimuth bins    : {len(np.unique(s['azimuth_bin']))}")
    print(f"  multi-echo      : {(s['return_count'] == 2).sum():,} second returns")
    print(f"  moving points   : {np.asarray(s['gt_moving']).sum():,}")
    print(f"  instances       : {len(np.unique(s['gt_instance']))}")
    print(f"  intensity       : mean {s['intensity'].mean():.3f}  "
          f"p95 {np.percentile(s['intensity'], 95):.3f}")

    print("\n  POINTS PER CLASS")
    print(f"  {'':>3} {'class':<18}{'points':>10}{'share':>9}{'mean range':>12}")
    for c in range(len(CLASS_NAMES)):
        m = gt == c
        print(f"  {c:>3} {CLASS_NAMES[c]:<18}{m.sum():>10,}{100 * m.mean():>8.2f}%"
              f"{(r[m].mean() if m.any() else 0):>11.1f}m")
    print(f"      {'unlabelled':<18}{(gt < 0).sum():>10,}")

    print("\n  POINTS PER RANGE BAND  (and ground density, which is what the")
    print("  resolution law exists to compensate for)")
    print(f"  {'band':>10}{'points':>10}{'share':>9}{'ground pts/m2':>16}")
    for (lo, hi), name in zip(RANGE_BANDS, BAND_NAMES):
        m = (r >= lo) & (r < hi)
        area = np.pi * (hi ** 2 - lo ** 2)
        g = m & np.isin(gt, (0, 1))
        print(f"  {name:>10}{m.sum():>10,}{100 * m.mean():>8.2f}%"
              f"{g.sum() / area:>15.2f}")

    # ── the centrepiece ──────────────────────────────────────
    d = np.hypot(p[:, 0] - FAR_PEDESTRIAN_X, p[:, 1] - FAR_PEDESTRIAN_Y)
    near = d < 1.2
    print(f"\n  THE 70 m PEDESTRIAN at ({FAR_PEDESTRIAN_X}, {FAR_PEDESTRIAN_Y})")
    if near.any():
        print(f"    {near.sum()} points at {r[near].mean():.1f} m, "
              f"z {p[near, 2].min():.2f}..{p[near, 2].max():.2f} m, "
              f"rings {sorted(set(s['ring'][near].tolist()))}")
        print(f"    GT classes: "
              f"{[CLASS_NAMES[c] for c in np.unique(gt[near]) if c >= 0]}")

    # ── assertions ───────────────────────────────────────────
    print()
    if args.scenario in ("mixed_urban", "pedestrian_far"):
        check("the 70 m pedestrian has more than 5 points", near.sum() > 5,
              f"{near.sum()} points")
        check("it is labelled VRU", bool((gt[near] == 4).all()) if near.any() else False)
    if args.scenario == "mixed_urban":
        for c in range(len(CLASS_NAMES)):
            check(f"class {c} ({CLASS_NAMES[c]}) is present", (gt == c).any(),
                  f"{int((gt == c).sum()):,} points")
        check("multi-echo returns exist (penetration layer is demonstrable)",
              (s["return_count"] == 2).any(),
              f"{int((s['return_count'] == 2).sum()):,}")
        check("moving objects exist", np.asarray(s["gt_moving"]).any())
        # Ground density must fall off steeply with range - the whole premise.
        dens = []
        for lo, hi in RANGE_BANDS:
            m = (r >= lo) & (r < hi) & np.isin(gt, (0, 1))
            dens.append(m.sum() / (np.pi * (hi ** 2 - lo ** 2)))
        check("ground density falls by >50x from the near to the far band",
              dens[0] / max(dens[-1], 1e-9) > 50,
              f"{dens[0]:.1f} -> {dens[-1]:.3f} pts/m2 "
              f"({dens[0] / max(dens[-1], 1e-9):.0f}x)")

    check("the scene is deterministic for a fixed seed",
          np.array_equal(generate_scenario(args.scenario, args.frame)["points"], p))

    # ── figure ───────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.6),
                             gridspec_kw={"width_ratios": [1.45, 1]})
    cmap = ListedColormap(CLASS_COLOURS)
    ax = axes[0]
    order = np.argsort(gt)          # draw rare classes last, on top
    sc = ax.scatter(p[order, 0], p[order, 1], c=np.clip(gt[order], 0, 5),
                    cmap=cmap, s=0.35, vmin=-0.5, vmax=5.5, linewidths=0)
    ax.scatter([0], [0], marker="*", s=220, c="k", zorder=6, label="sensor")
    if near.any():
        ax.scatter([FAR_PEDESTRIAN_X], [FAR_PEDESTRIAN_Y], s=420,
                   facecolors="none", edgecolors="#D62728", linewidths=2.2,
                   zorder=7)
        ax.annotate(f"pedestrian at {r[near].mean():.0f} m\n({near.sum()} points)",
                    (FAR_PEDESTRIAN_X, FAR_PEDESTRIAN_Y),
                    textcoords="offset points", xytext=(14, 18), fontsize=9,
                    color="#D62728", weight="bold")
    for rad in (10, 30, 60, 100):
        ax.add_patch(plt.Circle((0, 0), rad, fill=False, ec="0.75",
                                lw=0.7, ls=":"))
        ax.annotate(f"{rad} m", (rad * 0.71, rad * 0.71), fontsize=7,
                    color="0.5")
    ax.set_aspect("equal")
    ax.set_xlim(-30, 105)
    ax.set_ylim(-35, 35)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"{args.scenario} — {n:,} points, coloured by ground-truth class",
                 fontsize=11)
    cb = fig.colorbar(sc, ax=ax, ticks=range(6), fraction=0.03, pad=0.01)
    cb.ax.set_yticklabels(CLASS_NAMES, fontsize=8)

    ax = axes[1]
    edges = np.arange(0, 102, 2.5)
    for c in range(len(CLASS_NAMES)):
        m = gt == c
        if m.sum() < 5:
            continue
        ax.hist(r[m], bins=edges, histtype="step", lw=1.6,
                color=CLASS_COLOURS[c], label=CLASS_NAMES[c])
    ax.set_yscale("log")
    ax.set_xlabel("range (m)")
    ax.set_ylabel("points per 2.5 m shell (log)")
    ax.set_title("Range distribution per class\n"
                 "the far-field sparsity the resolution law answers",
                 fontsize=11)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=135)
    plt.close(fig)
    print(f"\n  Wrote {args.out}")

    print("\n" + "=" * 74)
    if FAILURES:
        print(f"  {len(FAILURES)} CHECK(S) FAILED")
        for f in FAILURES:
            print(f"    - {f}")
        sys.exit(1)
    print("  ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
