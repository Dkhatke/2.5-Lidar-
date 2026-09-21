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
SPEED_KEY = "fovea_speed"
N_FRAMES_KEY = "fovea_n_frames"
GATE_KEY = "fovea_gate"


#: The tab that currently drives playback. Only the owner schedules a
#: timer and only the owner advances the frame — two tickers on one shared
#: index makes the run play at the sum of their rates.
TICKER_KEY = "fovea_ticker"


def sync(widget_key: str, shared_key: str, default: Any) -> None:
    """Two-way bind one tab's widget to a shared session slot.

    Call once, immediately BEFORE creating the widget.

    The naive version of this — "copy shared into the widget, draw, copy
    the widget back" — silently reverts the user. Streamlit writes a
    changed widget value into session state before the script runs, so the
    copy-in overwrites the selection that was just made and the dropdown
    springs back to its previous value. It is exactly why every scenario
    except the default appeared not to work.

    So the direction is decided by comparing both sides against the value
    last agreed on: whichever one moved is the one that wins.
    """
    import streamlit as st
    seen = f"_{widget_key}_seen"
    st.session_state.setdefault(shared_key, default)
    st.session_state.setdefault(widget_key, st.session_state[shared_key])

    widget = st.session_state[widget_key]
    shared = st.session_state[shared_key]
    agreed = st.session_state.get(seen, shared)

    if widget != agreed:            # this tab's control moved
        st.session_state[shared_key] = widget
    elif shared != agreed:          # the other tab moved it
        st.session_state[widget_key] = shared
    st.session_state[seen] = st.session_state[shared_key]


#: Per-tab exponential moving average of how long a redraw actually took,
#: in seconds. Playback is paced from this rather than from a constant,
#: because the constant was measured on one machine and the demo runs on
#: another — on a weaker laptop a fixed interval schedules reruns faster
#: than they can be served, they queue, and playback stutters and appears
#: to restart.
def redraw_cost(tab: str, default: float = 0.25) -> float:
    import streamlit as st
    return float(st.session_state.get(f"_redraw_{tab}", default))


def record_redraw(tab: str, seconds: float) -> None:
    """Fold one measured redraw into the average for ``tab``."""
    import streamlit as st
    key = f"_redraw_{tab}"
    prev = st.session_state.get(key)
    # Rises fast, recovers steadily. The recovery used to be much slower,
    # which trapped the pace: one expensive redraw stretched the interval,
    # a long interval means few samples, and few samples meant it took
    # minutes to come back down.
    if prev is None:
        new = float(seconds)
    elif seconds > prev:
        new = 0.4 * prev + 0.6 * float(seconds)
    else:
        new = 0.6 * prev + 0.4 * float(seconds)
    st.session_state[key] = min(max(new, 0.02), 3.0)


def paced_interval(tab: str, requested: float, *, headroom: float = 1.5,
                   floor: float = 0.1) -> float:
    """The timer to schedule: never faster than this tab can redraw."""
    return max(requested, floor, redraw_cost(tab) * headroom)


def owns_ticker(tab: str, default: str = "live") -> bool:
    """Whether ``tab`` is the one driving playback right now."""
    import streamlit as st
    return st.session_state.get(TICKER_KEY, default) == tab


def claim_ticker(tab: str) -> None:
    """Take over playback. Called when a transport control is used."""
    import streamlit as st
    st.session_state[TICKER_KEY] = tab


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
