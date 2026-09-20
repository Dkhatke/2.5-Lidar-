"""
The state the two visualisation tabs share.

Live demo and Scene demo are two views of one run, so they must agree about
which frame is showing, how fast it is playing and what is selected. That
only works if there is exactly one copy of each — select a cell in the map,
switch tab, and the same cell is still selected in the scene.

The split that makes this work: **playback state is shared, playback
widgets are not.** Both tabs draw a transport, and two Streamlit widgets
cannot share a key, so each tab keys its own buttons while both bind to the
same :class:`PlaybackState` object under :data:`PLAYBACK_STATE_KEY`.
"""
from __future__ import annotations

from typing import Any, Optional

#: The one selected cell or object. Holds a ``selection.Selection``.
SELECTION_KEY = "fovea_selection"

#: The shared :class:`PlaybackState`. Widget keys stay per-tab.
PLAYBACK_STATE_KEY = "_fovea_playback"

#: Scenario, frame count and the MOS ablation: the three inputs that
#: decide WHICH RUN is on screen, so both tabs must agree on them.
#:
#: These are plain session slots, never widget keys. Streamlit refuses two
#: widgets with the same key even in different tabs, so each tab keys its
#: own control and both :func:`pull` from and :func:`push` to these.
SCENARIO_KEY = "fovea_scenario"
N_FRAMES_KEY = "fovea_n_frames"
GATE_KEY = "fovea_gate"


def pull(widget_key: str, shared_key: str, default: Any) -> None:
    """Load the shared value into this tab's widget slot, before drawing.

    This is what makes a scenario chosen in one tab show up in the other:
    Streamlit reads a keyed widget's value out of session state, so writing
    it beforehand is how a widget is told to move.
    """
    import streamlit as st
    st.session_state.setdefault(shared_key, default)
    if st.session_state.get(widget_key) != st.session_state[shared_key]:
        st.session_state[widget_key] = st.session_state[shared_key]


def push(widget_key: str, shared_key: str) -> None:
    """Publish what the user just set, after drawing."""
    import streamlit as st
    if widget_key in st.session_state:
        st.session_state[shared_key] = st.session_state[widget_key]


def get_selection() -> Optional[Any]:
    import streamlit as st
    return st.session_state.get(SELECTION_KEY)


def set_selection(sel: Optional[Any]) -> None:
    import streamlit as st
    st.session_state[SELECTION_KEY] = sel


def refresh(scope: str = "app") -> None:
    """Redraw after a change to shared state.

    APP-scoped by default, and that is the whole point. A fragment-scoped
    rerun redraws only the fragment that asked for it, so a cell selected
    in one tab would leave the other tab showing the previous selection
    until something else happened to re-run it — which is exactly the
    "two views, one selection" promise broken.

    The cost is a re-render, not a re-computation: the pipeline and the
    playback pre-compute both sit above this behind ``st.cache_resource``.
    Timer-driven playback stays fragment-scoped, because that fires several
    times a second and changes nothing another tab needs to hear about.
    """
    import streamlit as st
    from adaptive_lidar.visualization.playback_controller import (
        PlaybackController)
    if PlaybackController.supports_fragment_scope():
        st.rerun(scope=scope)
    else:
        st.rerun()
