"""
generate_report.py — turn the measured CSVs into docs/RESULTS.md.

Every table and every figure here is read from a file some other script
measured.  Nothing is illustrative; if a number is missing, the row says so
rather than being filled in.

    python scripts/run_baselines.py && python scripts/generate_report.py
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

import matplotlib                                                  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402

from adaptive_lidar.evaluation.metrics import BAND_NAMES           # noqa: E402
from adaptive_lidar.pipeline.types import CLASS_NAMES              # noqa: E402

DOCS = os.path.join(_PKG, "docs")
MISSING = "—"


def load_csv(path):
    if not os.path.isfile(path):
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def load_json(path, default=None):
    if not os.path.isfile(path):
        return default
    with open(path) as fh:
        return json.load(fh)


def num(row, key, default=float("nan")):
    v = row.get(key, "")
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def fmt(v, spec=".3f"):
    if v is None:
        return MISSING
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return MISSING if not np.isfinite(f) else format(f, spec)


def table(headers, rows) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


# ════════════════════════════════════════════════════════════
# Figures
# ════════════════════════════════════════════════════════════
def pareto_plot(rows, path):
    """Safety-critical retention vs MEASURED memory. The central figure."""
    if not rows:
        return False
    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    groups = {}
    for r in rows:
        groups.setdefault(r["policy"], []).append(r)

    style = {
        "full": dict(color="#C2410C", marker="o", ms=9, lw=2.4, zorder=5),
        "random": dict(color="#6B7280", marker="x", ms=8, lw=1.6, ls="--"),
        "distance_only": dict(color="#2563EB", marker="s", ms=7, lw=1.8),
    }
    for policy, rs in sorted(groups.items()):
        rs = sorted(rs, key=lambda r: num(r, "map_mb"))
        x = [num(r, "map_mb") for r in rs]
        y = [num(r, "orr_vru") for r in rs]
        if not any(np.isfinite(v) for v in y):
            continue
        st = style.get(policy, dict(alpha=0.55, lw=1.2, marker=".", ms=5))
        ax.plot(x, y, label=policy, **st)

    ax.set_xscale("log")
    ax.set_xlabel("measured map memory (MB, tracemalloc — log scale)")
    ax.set_ylabel("VRU object retention rate")
    ax.set_title("Retention of safety-critical objects vs memory\n"
                 "up and to the left is better", fontsize=11)
    ax.grid(alpha=0.25, which="both")
    ax.set_ylim(-0.04, 1.06)
    ax.legend(fontsize=7.5, ncol=2, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return True


def accuracy_by_range_plot(rows, path):
    if not rows:
        return False
    keep = [r for r in rows
            if r["policy"] in ("full", "uniform_20", "uniform_40", "distance_only")
            and abs(num(r, "budget") - 0.5) < 1e-6]
    if not keep:
        return False
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    xs = np.arange(len(BAND_NAMES))
    w = 0.8 / max(len(keep), 1)
    for i, r in enumerate(keep):
        ys = [num(r, f"elev_rmse_{b}") for b in BAND_NAMES]
        ax.bar(xs + i * w - 0.4 + w / 2, ys, w, label=r["policy"])
    ax.set_xticks(xs)
    ax.set_xticklabels(BAND_NAMES)
    ax.set_ylabel("elevation RMSE vs reference (m)")
    ax.set_xlabel("range band")
    ax.set_title("Accuracy across varying distances (M6), equal 50% budget",
                 fontsize=11)
    ax.grid(alpha=0.25, axis="y")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return True


# ════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(DOCS, "RESULTS.md"))
    args = ap.parse_args()

    rows = load_csv(os.path.join(DOCS, "baselines.csv"))
    meta = load_json(os.path.join(DOCS, "baselines_meta.json"), {})
    cal = load_json(os.path.join(DOCS, "calibration.json"), {})
    perf0 = load_csv(os.path.join(DOCS, "perf_baseline.csv"))
    perf1 = load_csv(os.path.join(DOCS, "perf_phase1.csv"))
    semev = load_csv(os.path.join(DOCS, "semantic_eval.csv"))
    resolution_law = os.path.isfile(os.path.join(DOCS, "resolution_law.png"))

    L = []
    A = L.append

    A("# RESULTS")
    A("")
    A("Adaptive Variable-Resolution 2.5D LiDAR Mapping — SIH 2026 / DRDO "
      "PS 26053.")
    A("")
    A("**Every number below was measured by a script in `scripts/` and read "
      "from a file in `docs/`.** Where a measurement is absent the cell says "
      f"`{MISSING}` rather than being filled in. Re-generate with:")
    A("")
    A("```\npython scripts/run_baselines.py\npython scripts/generate_report.py\n```")
    A("")
    if meta:
        A(f"Generated {meta.get('generated', '?')} · source "
          f"`{meta.get('data_source', '?')}` · {meta.get('n_frames', '?')} frames "
          f"· {meta.get('n_points_per_frame', 0):,} points/frame · backend "
          f"`{meta.get('backend_active', '?')}`")
        A("")

    # ── 1. headline ──────────────────────────────────────────
    A("## 1. Headline")
    A("")
    full = [r for r in rows if r["policy"] == "full"]
    if full:
        best = min(full, key=lambda r: abs(num(r, "budget") - 0.5))
        A(table(
            ["quantity", "value", "requirement"],
            [["map memory (measured)", f"{fmt(num(best, 'map_mb'), '.2f')} MB", "—"],
             ["uniform 5 cm over the same area",
              f"{fmt(num(best, 'uniform5_mb'), '.2f')} MB", "—"],
             ["**memory reduction**",
              f"**{fmt(num(best, 'memory_reduction_pct'), '.1f')} %**",
              "M5 — 'significant memory reduction'"],
             ["cells at 5 cm / 10 / 20 / 40 / 80",
              " / ".join(fmt(num(best, f'cells_L{l}'), ',.0f') for l in range(5)),
              "M4 — variable cell size"],
             ["latency p50 / p95 / p99",
              f"{fmt(num(best, 'latency_p50_ms'), '.1f')} / "
              f"{fmt(num(best, 'latency_p95_ms'), '.1f')} / "
              f"{fmt(num(best, 'latency_p99_ms'), '.1f')} ms",
              "M6 — low latency"],
             ["VRU object retention", fmt(num(best, "orr_vru")),
              "M3 — dynamic objects preserved"],
             ["semantic mIoU", fmt(num(best, "miou")), "M1 — segmentation"],
             ["elevation RMSE vs reference",
              f"{fmt(num(best, 'elev_rmse_m'))} m", "—"]]))
        A("")
        A(f"(`full` policy at {fmt(num(best, 'budget'), '.0%')} budget, where "
          "budget means *this fraction of the cell count a uniform 5 cm map of "
          "the same observed area would need*.)")
        A("")

    # ── 2. the central experiment ────────────────────────────
    A("## 2. The central experiment — equal memory, not equal resolution")
    A("")
    A("Comparing an adaptive map against a uniform 5 cm map is rigged: of "
      "course it is smaller, it was told to be. The honest question fixes the "
      "memory and asks what the best map obtainable for it looks like. "
      "Uniform spends the budget evenly; this system spends it where the "
      "value function says it matters.")
    A("")
    if rows:
        budgets = sorted({num(r, "budget") for r in rows}, reverse=True)
        hdr = ["policy"] + [f"{b:.0%}" for b in budgets]
        body = []
        for policy in sorted({r["policy"] for r in rows}):
            cells = [policy]
            for b in budgets:
                m = [r for r in rows if r["policy"] == policy
                     and abs(num(r, "budget") - b) < 1e-9]
                if not m:
                    cells.append(MISSING)
                    continue
                r = m[0]
                cells.append(f"{fmt(num(r, 'map_mb'), '.2f')} MB / "
                             f"{fmt(num(r, 'orr_vru'), '.2f')}")
            body.append(cells)
        A("**memory (MB) / VRU retention**, by policy and budget:")
        A("")
        A(table(hdr, body))
        A("")

    if pareto_plot(rows, os.path.join(DOCS, "pareto.png")):
        A("![Retention vs memory](pareto.png)")
        A("")
        A("*`full` should sit above and to the left of everything else: the "
          "same retention for less memory, or more retention for the same.*")
        A("")

    # ── 3. accuracy across varying distances (M6) ────────────
    A("## 3. Accuracy across varying distances (M6)")
    A("")
    A("The problem statement asks for accuracy *across varying distances*, "
      "which is not an aggregate. Both tables are stratified into the four "
      "range bands.")
    A("")
    if rows:
        A("### 3.1 Elevation RMSE against the reference map, by range band")
        A("")
        sel = [r for r in rows if abs(num(r, "budget") - 0.5) < 1e-9]
        A(table(["policy"] + list(BAND_NAMES) + ["overall", "bias (overall)"],
                [[r["policy"]]
                 + [fmt(num(r, f"elev_rmse_{b}")) for b in BAND_NAMES]
                 + [fmt(num(r, "elev_rmse_m")), fmt(num(r, "elev_bias_m"), "+.3f")]
                 for r in sorted(sel, key=lambda r: r["policy"])]))
        A("")
        A("Signed bias is reported alongside RMSE because a planner can "
          "absorb variance but not a systematic offset: a consistent 10 cm "
          "underestimate of a kerb is what drives a vehicle into it.")
        A("")

        A("### 3.2 Semantic mIoU by range band")
        A("")
        r = sel[0] if sel else (rows[0] if rows else None)
        if r:
            A(table(["metric"] + list(BAND_NAMES) + ["overall"],
                    [["mIoU"] + [fmt(num(r, f"miou_{b}")) for b in BAND_NAMES]
                     + [fmt(num(r, "miou"))],
                     ["accuracy"] + [fmt(num(r, f"acc_{b}")) for b in BAND_NAMES]
                     + [fmt(num(r, "accuracy"))]]))
            A("")
            A("Per-point semantics do not depend on the allocation policy — "
              "the same classifier sees the same points — so this table is "
              "the same for every policy. What the policy changes is how much "
              "of that classification survives into the map, which is the "
              "retention table above.")
            A("")

    if accuracy_by_range_plot(rows, os.path.join(DOCS, "accuracy_by_range.png")):
        A("![Accuracy by range](accuracy_by_range.png)")
        A("")

    # ── 4. per-class semantics ───────────────────────────────
    A("## 4. Semantic segmentation (M1)")
    A("")
    if full:
        r = min(full, key=lambda x: abs(num(x, "budget") - 0.5))
        A(table(["class", "IoU", "ORR", "objects observed"],
                [[CLASS_NAMES[c], fmt(num(r, f"iou_{CLASS_NAMES[c]}")),
                  fmt(num(r, f"orr_{CLASS_NAMES[c]}")),
                  fmt(num(r, f"orr_n_{CLASS_NAMES[c]}"), ".0f")]
                 for c in range(len(CLASS_NAMES))]))
        A("")
    if cal:
        A("### Calibration")
        A("")
        A(table(["quantity", "value"],
                [["temperature", fmt(cal.get("temperature"), ".4f")],
                 ["ECE before", fmt(cal.get("ece_before"), ".4f")],
                 ["ECE after", fmt(cal.get("ece_after"), ".4f")],
                 ["validation points", f"{cal.get('val_points', 0):,}"],
                 ["validation mIoU", fmt(cal.get("val_miou"))]]))
        A("")
        A(f"*{cal.get('note', '')}*")
        A("")
    if semev:
        A("### Backend comparison (from `scripts/eval_semantic.py`)")
        A("")
        cols = ["backend"] + [f"miou_{b}" for b in BAND_NAMES] + ["miou", "ece"]
        have = [c for c in cols if c in semev[0]]
        A(table(have, [[fmt(r.get(c), ".4f") if c != "backend" else r.get(c)
                        for c in have] for r in semev]))
        A("")

    # ── 5. latency ───────────────────────────────────────────
    A("## 5. Latency (M6)")
    A("")
    if perf1:
        A("### Per-stage latency vs point count")
        A("")
        stages = ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9"]
        A(table(["points"] + stages + ["total ms", "FPS"],
                [[f"{int(float(r['n_points'])):,}"]
                 + [fmt(num(r, s), ".1f") for s in stages]
                 + [fmt(num(r, "stage_sum_ms"), ".1f"), fmt(num(r, "fps"), ".1f")]
                 for r in perf1]))
        A("")
    if perf0 and perf1:
        A("### Against the pre-existing implementation")
        A("")
        body = []
        for a, b in zip(perf0, perf1):
            t0, t1 = num(a, "stage_sum_ms"), num(b, "stage_sum_ms")
            body.append([f"{int(float(b['n_points'])):,}",
                         fmt(t0, ".0f"), fmt(t1, ".0f"),
                         f"{fmt(t0 / t1 if t1 else float('nan'), '.1f')}x"])
        A(table(["points", "before (ms)", "after (ms)", "speed-up"], body))
        A("")
    if full:
        r = min(full, key=lambda x: abs(num(x, "budget") - 0.5))
        st = ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9"]
        A("### p50 / p99 per stage at the operating point")
        A("")
        A(table(["stage", "p50 (ms)", "p99 (ms)"],
                [[s, fmt(num(r, f"{s}_p50_ms"), ".2f"),
                  fmt(num(r, f"{s}_p99_ms"), ".2f")] for s in st]
                + [["**TOTAL**", f"**{fmt(num(r, 'latency_p50_ms'), '.1f')}**",
                    f"**{fmt(num(r, 'latency_p99_ms'), '.1f')}**"]]))
        A("")
        A("p99 is reported because for a real-time system the tail *is* the "
          "requirement — the mean hides exactly the frames that would miss "
          "their deadline.")
        A("")

    # ── 6. map correctness ───────────────────────────────────
    A("## 6. Map correctness (M4)")
    A("")
    A("`scripts/test_map.py` asserts all of the following and prints the "
      "measured values; see PROGRESS.md for its full output.")
    A("")
    if rows:
        sel = [r for r in rows if r["policy"] == "full"
               and abs(num(r, "budget") - 0.5) < 1e-9]
        if sel:
            r = sel[0]
            A(table(["check", "measured"],
                    [["boundary p95 across level transitions",
                      f"{fmt(num(r, 'boundary_p95_transition_m'))} m"],
                     ["boundary p95 across interior edges",
                      f"{fmt(num(r, 'boundary_p95_interior_m'))} m"],
                     ["ratio (1.0 = no seam artefact)",
                      fmt(num(r, "boundary_p95_ratio"), ".2f")],
                     ["map completeness vs reference",
                      fmt(num(r, "completeness"))],
                     ["spurious-cell rate", fmt(num(r, "spurious_rate"))]]))
            A("")
            A("If the elevation step across a level-transition edge looks "
              "like the step across an ordinary interior edge, the hierarchy "
              "is not producing a seam — which is the problem statement's "
              "\"without causing alignment errors\", measured rather than "
              "argued.")
            A("")

    if resolution_law:
        A("## 7. The derived resolution law")
        A("")
        A("![Resolution law](resolution_law.png)")
        A("")
        A("Cell size is not a hardcoded distance table. Ground sample area "
          "per beam grows roughly as r^3, so holding expected points-per-cell "
          "constant requires cell size proportional to r. Anchoring 5 cm at "
          "10 m gives **50 cm at 100 m — the problem statement's own example "
          "value, derived rather than assumed** — which is then quantised "
          "onto the power-of-two ladder.")
        A("")

    # ── reference-map caveats ────────────────────────────────
    A("## 8. What the reference map is, and is not")
    A("")
    A("No dataset ships ground-truth 2.5D maps, so the reference is "
      "constructed: accumulate frames by pose, drop GT-moving points, build "
      "a uniform 5 cm map. Its limitations bound every number scored against "
      "it:")
    A("")
    for lim in (meta.get("reference_limitations") or
                ["(reference metadata not found — run scripts/run_baselines.py)"]):
        A(f"- {lim}")
    A("")

    A("---")
    A("")
    A(f"*Generated by `scripts/generate_report.py` at "
      f"{time.strftime('%Y-%m-%d %H:%M:%S')}.*")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"Wrote {args.out}  ({len(L)} lines)")
    for f in ("pareto.png", "accuracy_by_range.png"):
        p = os.path.join(DOCS, f)
        if os.path.isfile(p):
            print(f"Wrote {p}")


if __name__ == "__main__":
    main()
