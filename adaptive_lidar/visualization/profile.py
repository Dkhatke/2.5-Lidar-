"""
One switch for the UI's diagnostic trace.

    FOVEA_PROFILE=1 streamlit run app.py

Off by default and costing a single dict lookup when off. It exists because
the two hardest bugs in this UI were both invisible from the outside and
obvious from four lines of log:

* ``scrub`` — Streamlit fires a slider's ``on_change`` for the value the
  SERVER writes, not only for a drag. With two tabs syncing one shared
  frame index, every redraw looked like a scrub and restarted playback.
* ``arm`` / ``clock`` — ``st.fragment(run_every=...)`` fixes its interval
  when the fragment is decorated, so whether playback actually animates
  depends on an app-level rerun happening at the right moment.

Channels: ``frag`` (redraw cost), ``arm`` (timer interval chosen),
``clock`` (playback clock restarts), ``toggle``, ``scrub``.
"""
from __future__ import annotations

import os

ENABLED = bool(os.environ.get("FOVEA_PROFILE"))


def log(channel: str, message: str) -> None:
    """Print one trace line to the server console, if tracing is on."""
    if ENABLED:
        print(f"[{channel}] {message}", flush=True)
