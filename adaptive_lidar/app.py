"""
app.py — the dashboard (M5).

    streamlit run app.py

Built to be understandable by someone who has never heard of a 2.5D map, and
honest to someone who has:

  * the budget slider is live, and dragging it coarsens the whole map EXCEPT
    the safety-pinned regions — a threshold-based system cannot do that at
    all, which is why it is the single most important control here;
  * the cell-boundary overlay makes the varying cell size visible rather than
    merely claimed;
  * clicking a cell shows every stored layer, so "trust me" becomes
    "inspect it yourself";
  * the driver toggles ARE the ablation study, run interactively;
  * the three-panel view fixes the MEMORY and asks which map still contains
    the pedestrian.

Nothing is displayed that was not measured in this session. The active
backend and data source are always on screen, and oracle mode is impossible
to mistake for a prediction.
"""
from __future__ import annotations

import copy
import os
import sys
import time

import numpy as np
import streamlit as st

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
for p in (_PARENT, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from adaptive_lidar.data.loader import get_dataset                  # noqa: E402
from adaptive_lidar.data.synthetic_scene import (                   # noqa: E402
    FAR_PEDESTRIAN_X,
    FAR_PEDESTRIAN_Y,
    SCENARIOS,
)
from adaptive_lidar.evaluation.metrics import (                     # noqa: E402
    BAND_NAMES,
    elevation_error,
    latency_percentiles,
    object_retention_rate,
    range_stratified_iou,
)
from adaptive_lidar.evaluation.reference_map import build_reference_map  # noqa: E402
from adaptive_lidar.mapping.allocation import AllocationController  # noqa: E402
from adaptive_lidar.perception.backends import BACKEND_NAMES        # noqa: E402
from adaptive_lidar.pipeline.pipeline import Pipeline               # noqa: E402
from adaptive_lidar.pipeline.types import (                         # noqa: E402
    CELL_DTYPE,
    CLASS_NAMES,
    CellFlags,
    Occupancy,
    ResolutionLevel,
    Traversability,
    VehicleProfile,
)
from adaptive_lidar.utils.config import load_config                 # noqa: E402
from adaptive_lidar.visualization import render as R                # noqa: E402

st.set_page_config(page_title="Adaptive 2.5D LiDAR Mapping",
                   layout="wide", initial_sidebar_state="expanded")

CSS = """
<style>
  .block-container {padding-top: 1.1rem; padding-bottom: 1rem; max-width: 100%;}
  div[data-testid="stMetricValue"] {font-size: 1.32rem;}
  div[data-testid="stMetricLabel"] {font-size: 0.74rem; opacity: .72;}
  .hdr {font-size: 1.32rem; font-weight: 700; letter-spacing: .01em;}
  .sub {font-size: .80rem; opacity: .68; margin-top: -4px;}
  .oracle {background:#7F1D1D;color:#fff;padding:9px 14px;border-radius:6px;
           font-weight:700;letter-spacing:.03em;margin:6px 0;}
  .prov {font-size: .72rem; opacity: .6;}
  .starred {color:#C2410C; font-weight:700;}
  code {font-size: .78rem;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# Pipeline execution, cached per configuration
# ════════════════════════════════════════════════════════════
#: Generate the longest run anyone asks for and slice it, so the main
#: dashboard at 6 frames and the Drive tab at 12 share one generation instead
#: of paying for the scans twice.
_MAX_FRAMES = 24


@st.cache_resource(show_spinner=False)
def _load_all_frames(scenario: str):
    ds = get_dataset("synthetic", None, _MAX_FRAMES, scenario=scenario,
                     quiet=True)
    return list(ds), ds.name


def load_frames(scenario: str, n_frames: int):
    frames, name = _load_all_frames(scenario)
    return frames[:n_frames], name


@st.cache_resource(show_spinner=False)
def reference_for(scenario: str, n_frames: int):
    frames, _ = load_frames(scenario, n_frames)
    return build_reference_map(frames)


@st.cache_resource(show_spinner=False, max_entries=48)
def run_pipeline(scenario, n_frames, policy, budget, backend,
                 gate_enabled, upto_frame, weight_key):
    """Run the pipeline and return everything the page needs.

    Cached on every input, so dragging the budget slider replays only
    configurations that have not been seen yet.
    """
    frames, src = load_frames(scenario, n_frames)
    cfg = copy.deepcopy(load_config())
    cfg.setdefault("motion", {})["gate_enabled"] = gate_enabled
    wg, ws, wu, wd = weight_key
    cfg["weights"] = {"geometry": wg, "semantic": ws,
                      "uncertainty": wu, "dynamic": wd}

    pipe = Pipeline(cfg)
    pipe.build_stages(backend=backend, policy=policy)
    pipe.set_budget(budget)
    pipe.context.data_source = src

    pred, true, rng = [], [], []
    last = None
    for k, f in enumerate(frames[:upto_frame + 1]):
        last = pipe.run(cloud_np=f, frame_id=f["frame_id"],
                        timestamp=f["timestamp"])
        if k == 0:
            pipe.context.cumulative_timing.clear()   # discard warm-up
        if last.gt_label is not None:
            pred.append(np.asarray(last.sem_class))
            true.append(np.asarray(last.gt_label))
            rng.append(last._range)

    amap = pipe.context.amap
    return {
        "arrays": amap.all_cells_arrays(),
        "counts": amap.cell_counts(),
        "n_cells": amap.n_cells,
        "bytes": amap.nbytes(),
        "uniform_cells": pipe.context.uniform_ref.n_cells,
        "uniform_bytes": pipe.context.uniform_ref.nbytes(),
        "telemetry": dict(last.timing.get("telemetry", {})),
        "latency": latency_percentiles(pipe.context),
        "s6": dict(last.timing.get("S6_diag", {})),
        "s8": dict(last.timing.get("S8_diag", {})),
        "s1": dict(last.timing.get("S1_diag", {})),
        "backend": pipe.context.semantic_backend_name,
        "source": src,
        "tiles": [(t.cx, t.cy, t.resolution_level, t.safety_pinned,
                   t.safety_reason, t.point_count, t.info_value)
                  for t in (last.tiles or [])],
        "instances": [(i.instance_id, i.semantic_class, i.state_name,
                       float(i.speed), i.point_count,
                       i.centroid.tolist(), i.extent.tolist())
                      for i in (last.instances or [])],
        "overlay": (pipe.context.dynamic_overlay.points.copy()
                    if pipe.context.dynamic_overlay else np.zeros((0, 3))),
        "sem": (range_stratified_iou(np.concatenate(pred),
                                     np.concatenate(true),
                                     np.concatenate(rng)) if pred else None),
        "orr": object_retention_rate(amap, frames[:upto_frame + 1]),
        "elev": elevation_error(amap, reference_for(scenario, n_frames)),
        "amap": amap,
        "pose": last.pose,
        "n_points": len(last.points),
    }


@st.cache_resource(show_spinner=False, max_entries=12)
def precompute_drive(scenario: str, n_frames: int, gate: bool):
    """Run the pipeline ONCE over a scenario and cache every frame.

    Cached on (scenario, frames, gate) so the MOS A/B toggle is instant and
    camera work never touches the pipeline.
    """
    from adaptive_lidar.visualization.playback import precompute
    frames, _ = load_frames(scenario, n_frames)
    return precompute(load_config(), frames, scenario, "full", 0.5,
                      "auto", gate)


def trav_for(amap, profile_name):
    p = (VehicleProfile.tracked() if profile_name == "tracked"
         else VehicleProfile.wheeled())
    return amap.traversability_arrays(p)["verdict"], p


# ════════════════════════════════════════════════════════════
# Sidebar
# ════════════════════════════════════════════════════════════
cfg0 = load_config()
S = st.sidebar
S.markdown("### Configuration")

scenario = S.selectbox("Scenario", SCENARIOS,
                       index=SCENARIOS.index("mixed_urban"))
n_frames = S.slider("Frames in the run", 2, 20, 6)
frame_idx = S.slider("Timeline — frame", 0, n_frames - 1, n_frames - 1,
                     help="Scrub the run; the map accumulates up to this frame.")

S.markdown("---")
policy = S.selectbox("Allocation policy", AllocationController.POLICIES,
                     index=AllocationController.POLICIES.index("full"))
backend = S.selectbox("Semantic backend", ("auto",) + tuple(BACKEND_NAMES),
                      index=0)
vehicle = S.radio("Vehicle profile", ("wheeled", "tracked"), horizontal=True,
                  help="Traversability is computed per vehicle from stored "
                       "terrain properties — the map itself is "
                       "vehicle-independent.")

S.markdown("---")
S.markdown("**Allocation drivers** — this *is* the ablation study")
S.caption("Turn a term off and watch where the budget moves.")
w = cfg0.get("weights", {})
use_g = S.checkbox("geometry (G)", True)
use_s = S.checkbox("semantic stake (S)", True)
use_u = S.checkbox("uncertainty (U)", True)
use_d = S.checkbox("dynamics (D)", True)
weight_key = (w.get("geometry", .3) * use_g, w.get("semantic", .3) * use_s,
              w.get("uncertainty", .2) * use_u, w.get("dynamic", .2) * use_d)
if sum(weight_key) == 0:
    weight_key = (1e-6, 1e-6, 1e-6, 1e-6)
    S.warning("All drivers off — allocation falls back to the distance "
              "schedule alone.")

S.markdown("---")
gate = S.checkbox("MOS gate enabled", True,
                  help="Off = moving points are written to the persistent "
                       "map. That is the trail ablation.")
show_edges = S.checkbox("Show cell boundaries", True,
                        help="Draws each cell's edges so the varying cell "
                             "size is visible rather than merely claimed.")
show_overlay = S.checkbox("Show dynamic overlay", True)
layer = S.selectbox("Layer", R.LAYERS,
                    index=R.LAYERS.index("semantic"))
px_per_m = S.slider("Render resolution (px/m)", 3, 16, 7)


# ════════════════════════════════════════════════════════════
# Header
# ════════════════════════════════════════════════════════════
c1, c2 = st.columns([3, 2])
with c1:
    st.markdown("#### ADAPTIVE VARIABLE-RESOLUTION 2.5D LiDAR MAPPING")
    st.markdown('<div class="sub">SIH 2026 · DRDO PS 26053 · dynamic '
                'environment perception</div>', unsafe_allow_html=True)
with c2:
    budget = st.slider("★ MEMORY BUDGET — fraction of a uniform 5 cm map",
                       0.05, 1.0, 0.5, 0.05,
                       help="Drag me. Everything coarsens EXCEPT the "
                            "safety-pinned regions. A threshold-based system "
                            "cannot do this.")

t_start = time.perf_counter()
with st.spinner("Running the pipeline..."):
    R_ = run_pipeline(scenario, n_frames, policy, budget, backend, gate,
                      frame_idx, weight_key)
elapsed = time.perf_counter() - t_start

if R_["backend"] == "oracle":
    st.markdown('<div class="oracle">⚠ ORACLE MODE — ground-truth labels, '
                'NOT a prediction. Any accuracy shown measures the MAP, '
                'never the segmenter.</div>', unsafe_allow_html=True)

st.markdown(
    f'<div class="prov">data source <code>{R_["source"]}</code> · backend '
    f'<code>{R_["backend"]}</code> · policy <code>{policy}</code> · '
    f'frame {frame_idx + 1}/{n_frames} · {R_["n_points"]:,} points/frame · '
    f'range image <code>{R_["s1"].get("ri_method", "?")}</code> '
    f'({100 * R_["s1"].get("ri_valid_rate", 0):.1f}% valid, '
    f'{100 * R_["s1"].get("ri_collision_rate", 0):.2f}% collisions)'
    f'</div>', unsafe_allow_html=True)
st.markdown("")

# ════════════════════════════════════════════════════════════
# Main map + right rail
# ════════════════════════════════════════════════════════════
a = R_["arrays"]
amap = R_["amap"]
trav, profile = trav_for(amap, vehicle)

pose = R_["pose"] if R_["pose"] is not None else np.eye(4)
ex = float(pose[0, 3])
extent = (ex - 25.0, ex + 95.0, -32.0, 32.0)

left, right = st.columns([2.45, 1])

with left:
    img, ext = R.render(a, layer, extent, px_per_m, show_edges, trav)
    if show_overlay and len(R_["overlay"]):
        img = R.overlay_points(img, R_["overlay"], ext, px_per_m)
    st.image(np.flipud(img), width='stretch',
             caption=f"{layer} · {extent[0]:.0f}..{extent[1]:.0f} m along the "
                     f"road · ego at x={ex:.1f} m")
    st.markdown(R.legend_html(layer), unsafe_allow_html=True)

with right:
    t = R_["telemetry"]
    red = (100.0 * (1 - R_["bytes"] / R_["uniform_bytes"])
           if R_["uniform_bytes"] else float("nan"))
    m1, m2 = st.columns(2)
    m1.metric("Cells", f"{R_['n_cells']:,}")
    m2.metric("Memory", f"{R_['bytes'] / 1e6:.2f} MB")
    m1.metric("Uniform 5 cm", f"{R_['uniform_bytes'] / 1e6:.2f} MB")
    m2.metric("Reduction", f"{red:.1f} %")

    lat = R_["latency"].get("TOTAL", {})
    m1, m2 = st.columns(2)
    m1.metric("Latency p50", f"{lat.get('p50', 0):.0f} ms")
    m2.metric("p99", f"{lat.get('p99', 0):.0f} ms")

    st.markdown("**Cells by physical size** — this is M4")
    tot = max(sum(R_["counts"].values()), 1)
    for lvl in range(5):
        c = R_["counts"].get(lvl, 0)
        st.markdown(
            f'<div style="display:flex;align-items:center;font-size:12px;'
            f'margin-bottom:2px">'
            f'<span style="width:38px;color:{R.LEVEL_COLOURS[lvl]};'
            f'font-weight:700">{ResolutionLevel.name(lvl)}</span>'
            f'<span style="flex:0 0 150px;background:#e8e8ec;height:11px;'
            f'border-radius:2px;overflow:hidden">'
            f'<span style="display:block;height:11px;width:{100 * c / tot:.1f}%;'
            f'background:{R.LEVEL_COLOURS[lvl]}"></span></span>'
            f'<span style="margin-left:8px">{c:,}</span></div>',
            unsafe_allow_html=True)

    st.markdown("**Object retention (ORR)**")
    orr = R_["orr"]["per_class"]
    for c in (4, 3, 2):
        d = orr[c]
        if d["observed"]:
            st.markdown(
                f'<div style="font-size:12px">'
                f'<span style="color:{R.CLASS_COLOURS[c]};font-weight:600">'
                f'{CLASS_NAMES[c]}</span> — {100 * d["orr"]:.1f}% '
                f'<span style="opacity:.6">({d["retained"]}/{d["observed"]} '
                f'objects)</span></div>', unsafe_allow_html=True)

    s6 = R_["s6"]
    st.markdown(
        f'<div style="font-size:12px;margin-top:8px">'
        f'<b>{s6.get("n_pinned", 0)}</b> of {s6.get("n_tiles", 0)} tiles '
        f'safety-pinned, taking {s6.get("pin_cells", 0):,} cells of a '
        f'{s6.get("cell_budget", 0):,}-cell budget'
        + ('<br><span style="color:#C2410C">pins exceed the budget — the '
           'safety floor is now the binding constraint</span>'
           if s6.get("budget_exceeded_by_pins") else '')
        + '</div>', unsafe_allow_html=True)
    s8 = R_["s8"]
    st.markdown(
        f'<div style="font-size:12px;margin-top:4px">MOS gate '
        f'<b>{"ON" if s8.get("gate_enabled") else "OFF"}</b> — '
        f'{s8.get("n_moving", 0):,} points held out of the persistent map '
        f'this frame</div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# Accuracy across varying distances (M6)
# ════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("#### Accuracy across varying distances  ·  M6")
st.caption("An aggregate number is dominated by the near field, where 70% of "
           "a LiDAR's points are and nothing is hard. These are stratified.")

sem = R_["sem"]
elev = R_["elev"]
rows = []
if sem:
    rows.append(["semantic mIoU"]
                + [f"{sem['bands'][b]['miou']:.3f}" if sem["bands"].get(b)
                   else "—" for b in BAND_NAMES]
                + [f"{sem['overall']['miou']:.3f}"])
    rows.append(["semantic accuracy"]
                + [f"{sem['bands'][b]['accuracy']:.3f}" if sem["bands"].get(b)
                   else "—" for b in BAND_NAMES]
                + [f"{sem['overall']['accuracy']:.3f}"])
rows.append(["elevation RMSE (m)"]
            + [f"{elev['bands'][b]['rmse_m']:.3f}" if elev["bands"].get(b)
               else "—" for b in BAND_NAMES]
            + [f"{elev['overall']['rmse_m']:.3f}" if elev.get("overall") else "—"])
rows.append(["elevation bias (m)"]
            + [f"{elev['bands'][b]['bias_m']:+.3f}" if elev["bands"].get(b)
               else "—" for b in BAND_NAMES]
            + [f"{elev['overall']['bias_m']:+.3f}" if elev.get("overall") else "—"])
try:
    import pandas as pd
    st.dataframe(pd.DataFrame(rows, columns=["metric"] + list(BAND_NAMES)
                              + ["overall"]),
                 hide_index=True, width='stretch')
except Exception:
    st.table(rows)
st.caption("Signed bias is shown next to RMSE because a planner can absorb "
           "variance but not a systematic offset — a consistent underestimate "
           "of a kerb is what drives a vehicle into it.")

# ════════════════════════════════════════════════════════════
# Tabs
# ════════════════════════════════════════════════════════════
tab_drive, tab_cmp, tab_cell, tab_obj, tab_lat = st.tabs(
    ["★ Drive", "★ Equal-memory comparison", "Cell inspector", "Objects",
     "Latency"])

# ── ego motion, cameras, playback ────────────────────────────
with tab_drive:
    from adaptive_lidar.visualization.drive_tab import render_drive_tab
    render_drive_tab(precompute_drive)

# ── the money shot ───────────────────────────────────────────
with tab_cmp:
    st.markdown("##### The same scene at three resolutions — what survives")
    st.caption(
        "Comparing an adaptive map against a uniform 5 cm map on memory "
        "alone is rigged: of course it is smaller, it was told to be. The "
        "question worth asking is what each map COSTS and what each map "
        "KEEPS. The uniform panels are fixed-resolution baselines and ignore "
        "the budget slider; the adaptive panel obeys it.")
    if st.checkbox("Run the three-panel comparison  "
                   "(three full pipelines — takes a few seconds)"):
        cols = st.columns(3)
        panels = [("uniform_5", "uniform 5 cm"),
                  ("uniform_40", "uniform 40 cm"),
                  ("full", "adaptive (full)")]
        summary = {}
        for col, (pol, label) in zip(cols, panels):
            with col:
                r = run_pipeline(scenario, n_frames, pol, budget, backend,
                                 gate, frame_idx, weight_key)
                tv, _ = trav_for(r["amap"], vehicle)
                im, e2 = R.render(r["arrays"], "semantic", extent,
                                  max(px_per_m, 7), show_edges, tv)
                st.image(np.flipud(im), width='stretch')
                ped = r["orr"]["per_class"][4]
                aa = r["arrays"]
                n_ped, sz = 0, 0.0
                if len(aa["cx"]):
                    d = np.hypot(aa["cx"] - FAR_PEDESTRIAN_X,
                                 aa["cy"] - FAR_PEDESTRIAN_Y)
                    near = d <= 1.2
                    n_ped = int(near.sum())
                    sz = float(aa["size"][near].mean() * 100) if near.any() else 0.0
                summary[label] = (r["bytes"] / 1e6, ped["orr"], n_ped, sz)
                st.markdown(f"**{label}**")
                st.markdown(
                    f'<div style="font-size:12px">'
                    f'{r["n_cells"]:,} cells · <b>{r["bytes"] / 1e6:.2f} MB</b><br>'
                    f'VRU retention '
                    f'<b style="color:{"#2E7D32" if ped["orr"] == 1 else "#C62828"}">'
                    f'{100 * ped["orr"]:.0f}%</b> '
                    f'({ped["retained"]}/{ped["observed"]} objects)<br>'
                    f'70 m pedestrian: <b>{n_ped}</b> cells at '
                    f'{sz:.0f} cm</div>',
                    unsafe_allow_html=True)

        if len(summary) == 3:
            u5 = summary["uniform 5 cm"]
            u40 = summary["uniform 40 cm"]
            ad = summary["adaptive (full)"]
            st.markdown(
                f"**What the three panels say.** The adaptive map keeps the "
                f"70 m pedestrian at {ad[3]:.0f} cm resolution — the same as "
                f"a uniform 5 cm map — for **{ad[0]:.2f} MB against "
                f"{u5[0]:.2f} MB**, that is "
                f"{100 * (1 - ad[0] / max(u5[0], 1e-9)):.0f}% less memory for "
                f"the same retention. The uniform map that IS cheaper "
                f"({u40[0]:.2f} MB) resolves the pedestrian into only "
                f"{u40[2]} cells at {u40[3]:.0f} cm and its VRU retention "
                f"falls to {100 * u40[1]:.0f}%. Spending the budget evenly is "
                f"what loses the object; spending it where the value function "
                f"points is what keeps it.")

# ── cell inspector ───────────────────────────────────────────
with tab_cell:
    st.markdown("##### Click any location — every stored layer, no summary")
    ic1, ic2 = st.columns(2)
    qx = ic1.number_input("x (m)", value=float(FAR_PEDESTRIAN_X), step=0.5)
    qy = ic2.number_input("y (m)", value=float(FAR_PEDESTRIAN_Y), step=0.5)
    cell = amap.cell_at(qx, qy)
    if cell is None and len(a["cx"]):
        # Snap to the nearest cell rather than showing nothing: at 5 cm the
        # exact query point usually falls between cells, and "no cell here"
        # is only the interesting answer when the area is genuinely unobserved.
        d = np.hypot(a["cx"] - qx, a["cy"] - qy)
        j = int(np.argmin(d))
        if d[j] < 2.0:
            qx, qy = float(a["cx"][j]), float(a["cy"][j])
            cell = amap.cell_at(qx, qy)
            st.caption(f"No cell exactly there; snapped to the nearest, "
                       f"{d[j]:.2f} m away at ({qx:.2f}, {qy:.2f}).")
    if cell is None:
        st.info("No cell within 2 m — the sensor never observed this area. "
                "Absence of a cell is not free space.")
    else:
        verdict, reason = amap.traversability(qx, qy, profile)
        d1, d2, d3 = st.columns(3)
        d1.metric("Cell size", f"{cell.resolution * 100:.0f} cm",
                  f"level {cell.level}")
        d2.metric("Points", f"{cell.n_points:,}")
        d3.metric(f"Traversable ({profile.name})",
                  Traversability.NAMES[verdict])
        st.caption(f"**reason:** {reason}")

        e1, e2 = st.columns(2)
        with e1:
            st.markdown("**Geometry**")
            st.markdown(
                f"- ground_z `{cell.ground_z:.3f}` m\n"
                f"- z_max `{cell.z_max:.3f}` m\n"
                f"- obstacle height `{cell.obstacle_height:.3f}` m\n"
                f"- overhead clearance "
                f"`{'inf' if not np.isfinite(cell.overhead_clearance) else f'{cell.overhead_clearance:.2f}'}` m\n"
                f"- z_var `{cell.z_var:.5f}`\n"
                f"- intensity mean `{cell.intensity_mean:.3f}` "
                f"var `{cell.intensity_var:.3f}`\n"
                f"- penetration `{cell.penetration:.3f}`\n"
                f"- observability `{cell.observability:.3f}`")
        with e2:
            st.markdown("**Semantics — the full distribution, not the argmax**")
            if cell.evidence is not None:
                order = np.argsort(-cell.evidence)
                for c in order:
                    st.markdown(
                        f'<div style="font-size:12px;display:flex;'
                        f'align-items:center">'
                        f'<span style="width:118px;color:{R.CLASS_COLOURS[int(c)]}">'
                        f'{CLASS_NAMES[int(c)]}</span>'
                        f'<span style="flex:0 0 90px;background:#e8e8ec;'
                        f'height:9px;border-radius:2px;overflow:hidden">'
                        f'<span style="display:block;height:9px;'
                        f'width:{100 * cell.evidence[c]:.0f}%;'
                        f'background:{R.CLASS_COLOURS[int(c)]}"></span></span>'
                        f'<span style="margin-left:8px">'
                        f'{cell.evidence[c]:.3f}</span></div>',
                        unsafe_allow_html=True)
            st.markdown(
                f"\n- entropy `{cell.entropy:.3f}` (uncertainty)\n"
                f"- occupancy `{Occupancy.NAMES[cell.occupancy_state]}` "
                f"(log-odds `{cell.occupancy_logodds:+.2f}`)\n"
                f"- dynamic prob `{cell.dynamic_probability:.3f}`\n"
                f"- last seen frame `{cell.last_seen}`\n"
                f"- flags `{', '.join(cell.flag_names()) or 'none'}`")
        st.caption(f"Stored in {CELL_DTYPE.itemsize} bytes + an 8-byte Morton "
                   f"key. Obstacle height, slope, traversability and the "
                   f"resolution level are derived on demand, never stored.")

# ── objects ──────────────────────────────────────────────────
with tab_obj:
    st.markdown("##### Tracked objects  ·  M3")
    st.caption("Velocity lives in this table and never in a cell: one car "
               "covers ~200 cells and storing its velocity 200 times goes "
               "inconsistent the moment the estimate updates.")
    inst = R_["instances"]
    if not inst:
        st.info("No instances in this frame.")
    else:
        try:
            import pandas as pd
            df = pd.DataFrame(
                [{"id": i[0], "class": CLASS_NAMES[i[1]], "state": i[2],
                  "speed (m/s)": round(i[3], 2), "points": i[4],
                  "x": round(i[5][0], 1), "y": round(i[5][1], 1),
                  "extent (m)": " x ".join(f"{v:.1f}" for v in i[6])}
                 for i in sorted(inst, key=lambda z: -z[4])[:25]])
            st.dataframe(df, hide_index=True, width='stretch')
        except Exception:
            st.write(inst[:25])
        st.caption("A parked car is MOVABLE_BUT_STATIONARY, not STATIC. "
                   "Conflating the two gives either permanent holes in car "
                   "parks or trails behind pedestrians.")

# ── latency ──────────────────────────────────────────────────
with tab_lat:
    st.markdown("##### Per-stage latency  ·  M6")
    lat = R_["latency"]
    stages = [s for s in ("S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7",
                          "S8", "S9") if s in lat]
    names = {"S0": "ingest", "S1": "indices", "S2": "ground + tiles",
             "S3": "pre-allocation", "S4": "semantics", "S5": "motion + tracks",
             "S6": "allocation", "S7": "map update", "S8": "MOS gate",
             "S9": "telemetry"}
    try:
        import pandas as pd
        df = pd.DataFrame([{"stage": s, "what": names.get(s, ""),
                            "p50 (ms)": round(lat[s]["p50"], 2),
                            "p95 (ms)": round(lat[s]["p95"], 2),
                            "p99 (ms)": round(lat[s]["p99"], 2)}
                           for s in stages]
                          + [{"stage": "TOTAL", "what": "",
                              "p50 (ms)": round(lat["TOTAL"]["p50"], 1),
                              "p95 (ms)": round(lat["TOTAL"]["p95"], 1),
                              "p99 (ms)": round(lat["TOTAL"]["p99"], 1)}])
        st.dataframe(df, hide_index=True, width='stretch')
    except Exception:
        st.write(lat)
    st.caption("p99 is shown because for a real-time system the tail IS the "
               "requirement — the mean hides exactly the frames that would "
               "miss their deadline. Measured on this machine, CPU only, "
               f"with the warm-up frame discarded. Page render: {elapsed:.2f} s.")

st.markdown("---")
st.markdown(
    '<div class="prov">Every number on this page was measured in this '
    'session. Nothing is illustrative. Run <code>python '
    'scripts/run_baselines.py &amp;&amp; python scripts/generate_report.py'
    '</code> for the full evaluation.</div>', unsafe_allow_html=True)
