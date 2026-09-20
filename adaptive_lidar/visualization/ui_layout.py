"""
Page chrome: the header, the status badges, the CSS, the metrics strip.

Kept in one module so that the layout can be changed without touching
anything that computes a number, and so that ``app.py`` reads as a sequence
of sections rather than as a wall of inline HTML.

On the badges: they state what is actually configured, read from the running
pipeline — the backend name comes from ``PipelineContext``, the data source
from the loader. Nothing here invents a label. The demo-data badge is not
decoration; a judge should be able to tell at a glance that the sensor is
simulated, without reading a caption.
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

CSS = """
<style>
  /* Streamlit's toolbar is 60px tall, opaque and absolutely positioned at
     the top of the page, so any padding below that tucks the first element
     underneath it. The previous 1.1rem hid the page title entirely. */
  .block-container {padding-top: 4.4rem; padding-bottom: 1rem; max-width: 100%;}
  div[data-testid="stMetricValue"] {font-size: 1.20rem;}
  div[data-testid="stMetricLabel"] {font-size: 0.70rem; opacity: .72;}

  /* header */
  .fv-wrap {display:flex; align-items:baseline; gap:.65rem; flex-wrap:wrap;}
  .fv-mark {font-size:1.62rem; font-weight:800; letter-spacing:.11em;
            line-height:1; color:#C2410C;}
  .fv-title {font-size:1.00rem; font-weight:650; letter-spacing:.005em;}
  .fv-ps {font-size:.76rem; opacity:.62; letter-spacing:.02em;}
  .fv-badges {margin-top:.35rem;}

  /* badges */
  .bdg {display:inline-block; padding:2px 9px; margin:0 5px 3px 0;
        border-radius:10px; font-size:10.5px; font-weight:700;
        letter-spacing:.035em; text-transform:uppercase;}
  .bdg-demo {background:#7F1D1D; color:#fff;}
  .bdg-cpu  {background:#0E7490; color:#fff;}
  .bdg-map  {background:#374151; color:#fff;}
  .bdg-warn {background:#C2410C; color:#fff;}
  .bdg-mute {background:#e7e8ec; color:#3a3d44;}

  /* compact metrics strip */
  .strip {display:flex; flex-wrap:wrap; gap:.15rem 1.35rem; align-items:baseline;
          padding:.42rem .7rem; border:1px solid #e3e4e9; border-radius:7px;
          background:#fbfbfd; margin:.15rem 0 .5rem 0;}
  .strip .sv {font-size:.95rem; font-weight:700; font-variant-numeric:tabular-nums;}
  .strip .sl {font-size:.66rem; opacity:.62; text-transform:uppercase;
              letter-spacing:.04em; margin-left:.3rem;}
  .strip .si {display:flex; align-items:baseline;}

  /* panels */
  .panel-h {font-size:.70rem; font-weight:800; letter-spacing:.10em;
            text-transform:uppercase; opacity:.55; margin:.55rem 0 .18rem 0;}
  .prov {font-size:.71rem; opacity:.6;}
  .oracle {background:#7F1D1D;color:#fff;padding:9px 14px;border-radius:6px;
           font-weight:700;letter-spacing:.03em;margin:6px 0;}
  .disclaim {font-size:.71rem; opacity:.62; border-left:2px solid #d4d5da;
             padding-left:.55rem; margin:.35rem 0;}
  .starred {color:#C2410C; font-weight:700;}
  code {font-size:.78rem;}

  /* the click surface should not look like a decorative picture */
  iframe[title="streamlit_image_coordinates.streamlit_image_coordinates"] {
      border-radius:6px;
  }
  section[data-testid="stSidebar"] {width: 21rem !important;}
</style>
"""

#: Shown wherever a metric could be mistaken for a deployment claim.
METRICS_DISCLAIMER = (
    "Metrics shown here are from the current demo/evaluation pipeline on "
    "synthetic data. They are not claims of real-world deployment "
    "performance, and no real sensor sequence has been run through this "
    "system.")

DEMO_DISCLAIMER = (
    "DEMO DATA · Synthetic scenario — a raycast sensor model (64 rings, "
    "2048 azimuth steps, real occlusion and multi-echo), not a recording.")


def inject_css() -> None:
    import streamlit as st
    st.markdown(CSS, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# Header
# ════════════════════════════════════════════════════════════
def badge(text: str, kind: str = "mute") -> str:
    return f'<span class="bdg bdg-{kind}">{text}</span>'


def status_badges(*, backend: str, data_source: str, oracle: bool = False,
                  extra: Sequence[Tuple[str, str]] = ()) -> str:
    """The row under the title. Every badge states something verifiable."""
    out = [badge("Synthetic / demo data", "demo"),
           badge(f"CPU backend · {backend}", "cpu"),
           badge("World-anchored map", "map")]
    if oracle:
        out.insert(0, badge("Oracle labels — not a prediction", "warn"))
    out.append(badge(f"source: {data_source}", "mute"))
    out.extend(badge(t, k) for t, k in extra)
    return f'<div class="fv-badges">{"".join(out)}</div>'


def header(*, backend: str, data_source: str, oracle: bool = False,
           extra: Sequence[Tuple[str, str]] = ()) -> None:
    """FOVEA — the wordmark, what it is, and what is actually running."""
    import streamlit as st
    st.markdown(
        '<div class="fv-wrap">'
        '<span class="fv-mark">FOVEA</span>'
        '<span class="fv-title">Adaptive 2.5D LiDAR Mapping</span>'
        '<span class="fv-ps">SIH 2026 · DRDO PS 26053</span>'
        '</div>' + status_badges(backend=backend, data_source=data_source,
                                 oracle=oracle, extra=extra),
        unsafe_allow_html=True)


def oracle_banner() -> None:
    import streamlit as st
    st.markdown('<div class="oracle">⚠ ORACLE MODE — ground-truth labels, '
                'NOT a prediction. Any accuracy shown measures the MAP, '
                'never the segmenter.</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# Compact metrics strip
# ════════════════════════════════════════════════════════════
def metrics_strip(items: Iterable[Tuple[str, str]]) -> None:
    """One line of numbers, not a grid of cards.

    The brief's metrics all still exist; they are simply no longer the first
    thing on the page competing with the scene for attention.
    """
    import streamlit as st
    cells: List[str] = [
        f'<span class="si"><span class="sv">{v}</span>'
        f'<span class="sl">{k}</span></span>' for k, v in items]
    st.markdown(f'<div class="strip">{"".join(cells)}</div>',
                unsafe_allow_html=True)


def disclaimer(text: str = METRICS_DISCLAIMER) -> None:
    import streamlit as st
    st.markdown(f'<div class="disclaim">{text}</div>', unsafe_allow_html=True)


def rail_heading(text: str) -> None:
    import streamlit as st
    st.markdown(f'<div class="panel-h">{text}</div>', unsafe_allow_html=True)


def provenance(text: str) -> None:
    import streamlit as st
    st.markdown(f'<div class="prov">{text}</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# Layout helpers
# ════════════════════════════════════════════════════════════
#: LEFT controls · CENTRE scene · RIGHT inspector. The centre is given most
#: of the width because the scene is the thing being demonstrated.
WORKSPACE_RATIO = (1.05, 3.25, 1.5)


def workspace(ratio: Optional[Sequence[float]] = None):
    import streamlit as st
    return st.columns(list(ratio or WORKSPACE_RATIO), gap="medium")
