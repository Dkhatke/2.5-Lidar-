"""
The scene renderer, as a Streamlit component.

Four static files under ``scene_frontend/`` — ``index.html``, ``scene.js``
and a vendored three.js and Streamlit bridge — declared with
``components.declare_component(path=...)``. (The directory is deliberately
not named after this module: a sibling directory sharing the name would be
importable as a namespace package and shadow it.) No build step, no node, no CDN:
a fresh clone runs offline, which is the same constraint the rest of the
project is built under.

The component is bidirectional. Python sends the scene payload built by
``scene_data.build_scene_data``; the browser sends back a click as a world
point plus an optional track id. It never sends back a cell — the
authoritative lookup stays in ``selection.py``, against the real cached
frame, so the 3D view and the 2D map cannot disagree about what was
clicked.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "scene_frontend")

_COMPONENT = None
_UNAVAILABLE: Optional[str] = None


def available() -> bool:
    """Whether the renderer can be declared at all."""
    return _load() is not None


def unavailable_reason() -> str:
    _load()
    return _UNAVAILABLE or ""


def _load():
    """Declare the component once per process.

    Wrapped because a missing vendored file should degrade to a message in
    the tab, not take the whole app down on import.
    """
    global _COMPONENT, _UNAVAILABLE
    if _COMPONENT is not None or _UNAVAILABLE is not None:
        return _COMPONENT
    missing = [f for f in ("index.html", "scene.js",
                           os.path.join("vendor", "three.min.js"),
                           os.path.join("vendor", "streamlit-bridge.js"))
               if not os.path.isfile(os.path.join(_DIR, f))]
    if missing:
        _UNAVAILABLE = f"missing renderer files: {', '.join(missing)}"
        return None
    try:
        import streamlit.components.v1 as components
        _COMPONENT = components.declare_component("fovea_scene", path=_DIR)
    except Exception as exc:                       # pragma: no cover
        _UNAVAILABLE = f"{type(exc).__name__}: {exc}"
        return None
    return _COMPONENT


def render(*, run: Optional[Dict[str, Any]], state: Dict[str, Any],
           key: str = "fovea_scene", height: int = 560,
           show_grid: bool = True, show_objects: bool = True,
           show_labels: bool = True, show_rings: bool = True,
           show_velocity: bool = True) -> Optional[Dict[str, Any]]:
    """Draw the scene and return the browser's last message, or None.

    ``run`` carries every frame's geometry and is sent ONCE — when the
    scenario, frame count, MOS setting or draw budget changes. On every
    other rerun it is ``None`` and the browser draws from the copy it
    already holds. Streamlit re-sends an element's args on every rerun, so
    passing the payload each time is precisely what made playback
    re-serialise the map per displayed frame.

    ``state`` is the small part: which frame to show, what is selected,
    and the display options. A few hundred bytes.
    """
    comp = _load()
    if comp is None:
        return None
    return comp(run=run, state=state, height=int(height),
                showGrid=bool(show_grid), showObjects=bool(show_objects),
                showLabels=bool(show_labels), showRings=bool(show_rings),
                showVelocity=bool(show_velocity), key=key, default=None)


def needs_run(raw: Optional[Dict[str, Any]]) -> bool:
    """Whether the browser is asking for the run payload again.

    It asks when it has none — a page reload drops the cached copy while
    Streamlit's session still believes it was sent. Without this the scene
    would stay empty until something else happened to change the run.
    """
    return bool(raw) and isinstance(raw, dict) and raw.get("kind") == "need_run"


def consume_click(raw: Optional[Dict[str, Any]], state_key: str
                  ) -> Optional[Dict[str, Any]]:
    """A click, but only once.

    The component's value persists across reruns, so without this the scene
    would re-select the same point on every frame of playback. The browser
    stamps each click with ``t``; a repeat carries the same stamp.
    """
    import streamlit as st
    if not raw or not isinstance(raw, dict) or raw.get("kind") == "need_run":
        return None
    stamp = raw.get("t")
    if stamp is not None and st.session_state.get(state_key) == stamp:
        return None
    st.session_state[state_key] = stamp
    return raw


def click_to_world(raw: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    x, y = raw.get("x"), raw.get("y")
    if x is None or y is None:
        return None
    return float(x), float(y)
