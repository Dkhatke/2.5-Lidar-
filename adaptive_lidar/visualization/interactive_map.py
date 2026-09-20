"""
The clickable map canvas.

This is the only place that knows the map is displayed as a raster in a
browser. It owns three things that have to agree exactly or selection lands
on the wrong cell:

  1. the image the renderer produced,
  2. the :class:`ViewTransform` describing the camera that produced it,
  3. the scale factor between the image's natural pixels and the pixels the
     browser actually laid it out at.

(3) is the one that is easy to miss. ``streamlit-image-coordinates`` reports
``event.offsetX`` on the ``<img>`` element, which is in DISPLAYED pixels; the
column is narrower than the 840 px canvas, so a click at the right edge comes
back as ~700 rather than ~840. The component also returns the displayed
width and height, which is what makes the correction possible.

The rendering path itself is unchanged — ``render.render_view`` plus the
existing overlays. Nothing about perception or mapping is touched here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from adaptive_lidar.visualization import camera as CAM
from adaptive_lidar.visualization import overlays as OV
from adaptive_lidar.visualization import render as R
from adaptive_lidar.visualization import selection as SEL
from adaptive_lidar.visualization.coordinate_transform import ViewTransform


@dataclass
class CanvasResult:
    """What one draw of the canvas produced."""
    image: np.ndarray                       # as displayed (already flipped)
    transform: ViewTransform
    drawn_objects: List[Dict[str, Any]]
    corridor_area: float
    click: Optional[Tuple[float, float]]    # display px, None if no new click


# ════════════════════════════════════════════════════════════
# Building the view and its transform together
# ════════════════════════════════════════════════════════════
def build_view(preset: str, frame_of_reference: str, ego_xy, heading: float,
               *, zoom: float = 1.0, pan: Tuple[float, float] = (0.0, 0.0),
               half_x: float = 60.0, half_y: float = 32.0,
               px_per_m: float = 7.0):
    """The camera window, in the chosen frame's coordinates."""
    v = CAM.compute_view(preset, frame_of_reference, ego_xy, heading,
                         zoom=zoom, pan=pan, half_x=half_x, half_y=half_y,
                         px_per_m=px_per_m)
    return CAM.frame_view(v, frame_of_reference, ego_xy, heading)


def transform_for(view, frame_of_reference: str, ego_xy, heading: float,
                  image_hw: Tuple[int, int]) -> ViewTransform:
    return ViewTransform.from_view(view, frame_of_reference, ego_xy, heading,
                                   image_hw, displayed_flipped=True)


def image_size_for(view) -> Tuple[int, int]:
    """(H, W) the renderer will produce for this view. Mirrors render_view."""
    x0, x1, y0, y1 = view.extent
    W = max(int((x1 - x0) * view.px_per_m), 16)
    H = max(int((y1 - y0) * view.px_per_m), 16)
    return H, W


# ════════════════════════════════════════════════════════════
# Drawing
# ════════════════════════════════════════════════════════════
def compose(snapshot, view, *,
            layer: str = "resolution level",
            frame_of_reference: str = "World",
            show_edges: bool = False,
            age_tint: bool = False,
            show_objects: bool = True,
            show_overlay: bool = True,
            show_prediction: bool = False,
            corridor=None,
            corridor_upto: Optional[int] = None,
            selected: Optional[SEL.Selection] = None,
            ) -> Tuple[np.ndarray, List[Dict[str, Any]], float]:
    """Render one frame through one camera, with overlays and the selection.

    Returns the image in RENDER orientation (+y up, not yet flipped for
    display), the objects actually drawn, and the corridor area.
    """
    ego, head = snapshot.ego_xy, snapshot.heading
    img = R.render_view(
        snapshot.cells, view, layer,
        frame_of_reference=frame_of_reference, ego_xy=ego, heading=head,
        show_cell_edges=show_edges,
        age_tint_frame=snapshot.frame_id if age_tint else None)

    project = OV.make_projector(view, frame_of_reference, ego, head)

    area = 0.0
    if corridor is not None and corridor_upto is not None:
        img, area = OV.draw_corridor(img, project, corridor, corridor_upto)

    drawn: List[Dict[str, Any]] = []
    if show_objects:
        img, drawn = OV.draw_objects(img, project, snapshot.objects,
                                     show_prediction=show_prediction)
    if show_overlay and len(snapshot.overlay):
        img = OV.draw_overlay_points(img, project, snapshot.overlay)
    img = OV.draw_ego(img, project, ego, head)

    # Selection last, so nothing paints over it.
    if selected is not None and not selected.is_empty:
        if selected.kind == "object" and selected.obj is not None:
            img = OV.draw_object_highlight(img, project, selected.obj)
        if selected.cell is not None:
            img = OV.draw_cell_highlight(
                img, project, selected.cell["cx"], selected.cell["cy"],
                selected.cell["resolution"])
        img = OV.draw_crosshair(img, project, *selected.world_xy)
    return img, drawn, area


# ════════════════════════════════════════════════════════════
# Click capture
# ════════════════════════════════════════════════════════════
def _clicks_available() -> bool:
    try:
        import streamlit_image_coordinates  # noqa: F401
        return True
    except Exception:
        return False


def scale_click(raw: Dict[str, Any], natural_hw: Tuple[int, int]
                ) -> Optional[Tuple[float, float]]:
    """Component pixels -> the image's own pixels.

    The component reports the click against the ``<img>`` element as the
    browser laid it out, which at ``width="stretch"`` is the column width,
    not the 840 px the renderer produced. Ignoring this scales every
    selection towards the left of the map by up to 20%, which looks like a
    plausible-but-wrong answer rather than an obvious bug.
    """
    if not raw:
        return None
    x, y = raw.get("x"), raw.get("y")
    if x is None or y is None:
        return None
    H, W = natural_hw
    dw = float(raw.get("width") or W) or float(W)
    dh = float(raw.get("height") or H) or float(H)
    return float(x) * W / dw, float(y) * H / dh


def show(img_display: np.ndarray, key: str) -> Optional[Dict[str, Any]]:
    """Draw the already-flipped image as a click target.

    Falls back to ``st.image`` when the component is not installed: the map
    still renders and everything except clicking still works, which is the
    right way to lose an optional dependency.
    """
    import streamlit as st
    if not _clicks_available():
        st.image(img_display, width="stretch")
        st.caption("Click-to-inspect needs `streamlit-image-coordinates` "
                   "(in requirements.txt). The map is read-only without it.")
        return None
    from streamlit_image_coordinates import streamlit_image_coordinates
    # The component sizes its iframe from the image's NATURAL height, then
    # the browser scales the image down to the column width — leaving a band
    # of dead space below it. The aspect ratio is known here, so pin it.
    h, w = img_display.shape[:2]
    st.markdown(
        f'<style>iframe[title*="image_coordinates"]'
        f'{{height:auto !important;aspect-ratio:{w}/{h};}}</style>',
        unsafe_allow_html=True)
    return streamlit_image_coordinates(
        img_display, key=key, width="stretch", cursor="crosshair",
        image_format="PNG", png_compression_level=1)


def consume_click(raw: Optional[Dict[str, Any]], natural_hw, state_key: str
                  ) -> Optional[Tuple[float, float]]:
    """A click, but only once.

    The component returns the SAME dict on every later rerun, so without
    de-duplication playback would re-select the last clicked point on every
    frame. ``unix_time`` is the component's own click timestamp, which is
    what makes a repeat distinguishable from a new click at the same pixel.
    """
    import streamlit as st
    if not raw:
        return None
    stamp = raw.get("unix_time")
    if stamp is not None and st.session_state.get(state_key) == stamp:
        return None
    st.session_state[state_key] = stamp
    return scale_click(raw, natural_hw)


# ════════════════════════════════════════════════════════════
# The whole canvas, in one call
# ════════════════════════════════════════════════════════════
def render_canvas(snapshot, *, key: str, layer: str,
                  preset: str, frame_of_reference: str,
                  zoom: float = 1.0, pan: Tuple[float, float] = (0.0, 0.0),
                  show_edges: bool = False, age_tint: bool = False,
                  show_objects: bool = True, show_overlay: bool = True,
                  show_prediction: bool = False,
                  corridor=None, corridor_upto: Optional[int] = None,
                  selected: Optional[SEL.Selection] = None,
                  px_per_m: float = 7.0) -> CanvasResult:
    """Build the view, draw it, show it, and report any new click."""
    ego, head = snapshot.ego_xy, snapshot.heading
    view = build_view(preset, frame_of_reference, ego, head, zoom=zoom,
                      pan=pan, px_per_m=px_per_m)
    hw = image_size_for(view)
    transform = transform_for(view, frame_of_reference, ego, head, hw)

    img, drawn, area = compose(
        snapshot, view, layer=layer,
        frame_of_reference=frame_of_reference, show_edges=show_edges,
        age_tint=age_tint, show_objects=show_objects,
        show_overlay=show_overlay, show_prediction=show_prediction,
        corridor=corridor, corridor_upto=corridor_upto, selected=selected)

    # Contiguous: the component hands the array to PIL, which
    # cannot take the negative stride flipud leaves behind.
    displayed = np.ascontiguousarray(np.flipud(img))
    raw = show(displayed, key=key)
    click = consume_click(raw, hw, f"_{key}_last_click")
    return CanvasResult(displayed, transform, drawn, area, click)
