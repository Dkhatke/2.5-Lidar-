"""
Playback transport: the frame index, and how it advances.

WHY THIS IS NOT ``time.sleep`` + ``st.rerun``
---------------------------------------------
The obvious way to animate in Streamlit is to sleep at the bottom of the
script and rerun. It works, and it is wrong for this application:

  * ``time.sleep`` blocks the server thread, so every control on the page is
    dead for the duration of each frame;
  * a whole-app ``st.rerun`` re-executes the entire script, which means every
    widget is rebuilt and anything not memoised is recomputed — here that is
    the LiDAR pipeline, and a cache miss (a changed scenario, a cold start)
    would run perception once per displayed frame;
  * a click captured mid-sleep is simply lost.

So the state machine lives here as plain Python, with no Streamlit in it at
all, and the Streamlit binding drives it from inside an ``st.fragment`` with
``run_every``. A fragment rerun re-executes only the fragment, so advancing a
frame redraws the canvas and nothing else. ``st.fragment`` and
``st.rerun(scope=...)`` are both present in the installed Streamlit (1.64);
the binding checks and degrades to manual stepping rather than calling an API
that may not be there.

The pipeline is never invoked from anything in this module. That is the
property ``tests/test_playback_controller.py`` asserts directly, by handing
the controller a callable that raises.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

from adaptive_lidar.visualization import profile as PROFILE

#: Replay speed multipliers offered in the transport.
#: 0.1x is a second per frame: slow enough to talk over, which is what a
#: walk-through actually needs. The fast end is bounded by the redraw, so
#: 2x and 4x mostly mean "as fast as this view can draw".
SPEEDS: Dict[str, float] = {"0.1x": 0.1, "0.25x": 0.25, "0.5x": 0.5,
                            "1x": 1.0, "2x": 2.0, "4x": 4.0}

#: Wall-clock seconds per displayed frame at 1x. The scenes are 10 Hz, so
#: this is real time; it is a REPLAY rate and is labelled as one, because it
#: says nothing about how long the pipeline took to produce a frame.
BASE_FRAME_S = 0.10

#: Streamlit will not schedule a fragment faster than this usefully, and
#: trying only queues reruns faster than they can be served.
MIN_INTERVAL_S = 0.05


# ════════════════════════════════════════════════════════════
# The state machine — no Streamlit, so it is directly testable
# ════════════════════════════════════════════════════════════
@dataclass
class PlaybackState:
    """Where playback is, and where it goes next."""

    n_frames: int
    frame_idx: int = 0
    playing: bool = False
    #: One second a frame. The demo is watched while someone talks over
    #: it, and every frame is shown, so slow is the useful default; the
    #: selector goes up to 4x for anyone who wants to skim.
    speed: str = "0.1x"
    loop: bool = True
    #: When the current playing stretch began, and the frame it began on.
    #: The displayed frame is computed FROM THE CLOCK rather than by
    #: counting redraws — see :meth:`frame_at`. Both live on the state
    #: because the controller is rebuilt on every script run while the
    #: state persists in session state.
    started_at: Optional[float] = None
    start_frame: int = 0

    def __post_init__(self) -> None:
        self.n_frames = max(int(self.n_frames), 0)
        self.frame_idx = self._clamp(self.frame_idx)

    # ── bounds ───────────────────────────────────────────────
    def _clamp(self, i: int) -> int:
        if self.n_frames <= 0:
            return 0
        return int(min(max(i, 0), self.n_frames - 1))

    @property
    def last(self) -> int:
        return max(self.n_frames - 1, 0)

    @property
    def at_end(self) -> bool:
        return self.frame_idx >= self.last

    # ── transitions ──────────────────────────────────────────
    def seek(self, i: int) -> None:
        """Scrub. Scrubbing does NOT pause: the timeline is a live position."""
        self.frame_idx = self._clamp(i)
        self.started_at = None          # the clock restarts from here

    def next(self) -> None:
        """One frame forward. Stepping pauses — it is for pointing at things."""
        self.playing = False
        self.frame_idx = self._wrap(self.frame_idx + 1)
        self.started_at = None

    def previous(self) -> None:
        self.playing = False
        self.frame_idx = self._wrap(self.frame_idx - 1)
        self.started_at = None

    def reset(self) -> None:
        self.playing = False
        self.frame_idx = 0
        self.started_at = None

    def toggle(self) -> None:
        if not self.playing and self.at_end and not self.loop:
            self.frame_idx = 0          # replay rather than sit at the end
        self.playing = not self.playing
        self.started_at = None

    def tick(self) -> None:
        """Advance because the timer fired. A no-op when paused.

        Kept separate from :meth:`next` because the timer must not be able to
        clear ``playing`` — that is what stepping means, not what a tick
        means.
        """
        if not self.playing or self.n_frames == 0:
            return
        if self.at_end and not self.loop:
            self.playing = False
            return
        self.frame_idx = self._wrap(self.frame_idx + 1)

    def start_clock(self, now: float) -> None:
        self.started_at = float(now)
        self.start_frame = self.frame_idx

    def due(self, now: float) -> bool:
        """Whether enough wall clock has passed to show the next frame."""
        if self.started_at is None:
            return False
        return (float(now) - self.started_at) >= self.interval_s()

    def _wrap(self, i: int) -> int:
        if self.n_frames <= 0:
            return 0
        return int(i % self.n_frames) if self.loop else self._clamp(i)

    # ── derived ──────────────────────────────────────────────
    def interval_s(self) -> float:
        """Wall-clock seconds the current frame should be held for."""
        return max(BASE_FRAME_S / SPEEDS.get(self.speed, 1.0), MIN_INTERVAL_S)

    def resize(self, n_frames: int) -> None:
        """A different scenario or frame count arrived."""
        n = max(int(n_frames), 0)
        if n != self.n_frames:
            self.n_frames = n
            self.frame_idx = self._clamp(self.frame_idx)

    def caption(self, scene_time_s: Optional[float] = None) -> str:
        """The one-line readout beside the transport."""
        parts = [f"Frame {self.frame_idx + 1}/{max(self.n_frames, 1)}"]
        if scene_time_s is not None:
            parts.append(f"Scene time {scene_time_s:.2f} s")
        parts.append(f"Speed {self.speed}")
        return "  ·  ".join(parts)


# ════════════════════════════════════════════════════════════
# The Streamlit binding
# ════════════════════════════════════════════════════════════
@dataclass
class PlaybackController:
    """Binds a :class:`PlaybackState` to ``st.session_state`` and widgets."""

    key: str = "pb"
    state: PlaybackState = field(default_factory=lambda: PlaybackState(1))

    # ── session-state persistence ────────────────────────────
    @classmethod
    def bind(cls, n_frames: int, key: str = "pb",
             state_key: Optional[str] = None) -> "PlaybackController":
        """Bind widgets keyed by ``key`` to the state under ``state_key``.

        The two are separate so that two tabs can each draw a transport —
        Streamlit widgets cannot share a key — while both drive ONE
        playback state. Without that split, switching tab would jump to a
        different frame, which is the opposite of sharing.
        """
        import streamlit as st
        st_key = state_key or f"_{key}_state"
        s = st.session_state.get(st_key)
        if not isinstance(s, PlaybackState):
            s = PlaybackState(n_frames)
            st.session_state[st_key] = s
        s.resize(n_frames)
        return cls(key=key, state=s)

    # ── capability probe ─────────────────────────────────────
    @staticmethod
    def supports_fragments() -> bool:
        """Whether the installed Streamlit can auto-advance without a sleep.

        Checked rather than assumed: on an older Streamlit the transport
        still works, it just does not animate on its own, which is a far
        better failure than an AttributeError on load.
        """
        import streamlit as st
        return hasattr(st, "fragment")

    @staticmethod
    def supports_fragment_scope() -> bool:
        import inspect
        import streamlit as st
        try:
            return "scope" in inspect.signature(st.rerun).parameters
        except (TypeError, ValueError):
            return False

    def run_every(self) -> Optional[float]:
        """The ``st.fragment(run_every=...)`` value for the current state."""
        if not self.state.playing or self.state.n_frames <= 1:
            return None
        return self.state.interval_s()

    # ── the transport row ────────────────────────────────────
    def render_transport(self, scene_time_s: Optional[float] = None,
                         *, on_change: Optional[Callable[[], None]] = None,
                         driving: bool = True) -> int:
        """Draw [◀ Previous][▶ Play][⏭ Next][Reset] + timeline + readout.

        Returns the frame index to display. Nothing here loads or computes
        anything; it only moves an integer.
        """
        import streamlit as st

        s = self.state
        scoped = self.supports_fragment_scope()

        def _claim():
            """This tab is now the one driving playback.

            Only one tab may tick: two tabs each advancing the shared
            index would play the run at the sum of their render rates.
            Ownership follows the last transport press, which is the tab
            the user is looking at.
            """
            from adaptive_lidar.visualization import session as _sess
            _sess.claim_ticker(self.key)

        def _rerun(scope: str = "fragment"):
            """Redraw.

            ``scope`` matters here in a way that is easy to miss.
            ``run_every`` is fixed when the fragment is DECORATED, which
            happens in the enclosing script run — so starting or stopping
            playback has to re-run the app, or the timer keeps whatever
            schedule it had and Play does nothing visible. Every other
            control only changes what is drawn, so it stays fragment-scoped
            and never re-enters the pipeline.
            """
            if on_change is not None:
                on_change()
            if scoped:
                st.rerun(scope=scope)
            else:
                st.rerun()

        c_prev, c_play, c_next, c_reset, c_bar, c_read = st.columns(
            [1.0, 1.0, 1.0, 1.0, 6.0, 2.6], vertical_alignment="center")

        with c_prev:
            if st.button("◀ Previous", key=f"{self.key}_prev",
                         width="stretch", disabled=s.n_frames <= 1,
                         help="One frame back. Pauses."):
                _claim()
                s.previous()
                _rerun("app")          # stepping pauses, so the timer stops
        with c_play:
            label = "⏸ Pause" if s.playing else "▶ Play"
            if st.button(label, key=f"{self.key}_play", width="stretch",
                         type="primary" if not s.playing else "secondary",
                         disabled=s.n_frames <= 1):
                _claim()
                s.toggle()
                PROFILE.log("toggle", f"{self.key} playing={s.playing}")
                _rerun("app")          # the timer schedule changes
        with c_next:
            if st.button("⏭ Next", key=f"{self.key}_next", width="stretch",
                         disabled=s.n_frames <= 1,
                         help="One frame forward. Pauses."):
                _claim()
                s.next()
                _rerun("app")          # stepping pauses, so the timer stops
        with c_reset:
            if st.button("↺ Reset", key=f"{self.key}_reset", width="stretch",
                         help="Back to frame 1 and pause."):
                _claim()
                s.reset()
                _rerun("app")          # reset pauses, so the timer stops

        with c_bar:
            if s.n_frames <= 1:
                st.caption("Single frame — nothing to play.")
            elif s.playing or not driving:
                # A READOUT, not a widget, whenever this tab is not the one
                # a scrub should come from.
                #
                # Two things went wrong with a live slider here. Streamlit
                # fires `on_change` for the value the SERVER writes, so
                # syncing the handle to playback looked like a scrub
                # several times a second. And with a slider in each tab
                # bound to one index, the one that had been idle reported
                # a position from before playback moved — which read as a
                # drag, seeked backwards, and cascaded reruns until it
                # caught up. A progress bar has no widget state to fight
                # over, and the position is still visible.
                frac = (s.frame_idx / max(s.last, 1)) * 100.0
                st.markdown(
                    f'<div style="height:6px;border-radius:3px;'
                    f'background:#e6e7ea;overflow:hidden;margin:.55rem 0">'
                    f'<div style="height:100%;width:{frac:.1f}%;'
                    f'background:#C2410C"></div></div>',
                    unsafe_allow_html=True)
            else:
                bar = f"{self.key}_bar"
                echo = f"_{bar}_written"
                # The value we wrote LAST time. If the frontend sends it
                # back after the index has moved on, that is a stale echo
                # rather than a drag, and acting on it walks the playhead
                # backwards.
                stale = st.session_state.get(echo)
                st.session_state[bar] = s.frame_idx
                st.session_state[echo] = s.frame_idx
                i = st.slider("Timeline", 0, s.last, key=bar,
                              label_visibility="collapsed")
                if i != s.frame_idx and i != stale:
                    _claim()
                    s.seek(i)
                    if on_change is not None:
                        on_change()

        with c_read:
            st.markdown(
                f'<div style="font-size:12px;line-height:1.4">'
                f'{s.caption(scene_time_s)}'
                f'<br><span style="opacity:.6">a replay rate, not a latency '
                f'measurement. Measured end to end it settles near 2 frames '
                f'per second whatever this is set to — the limit is drawing '
                f'and shipping the canvas to the browser, not the '
                f'pipeline</span></div>',
                unsafe_allow_html=True)

        if s.playing and not driving:
            st.caption(
                "Playing, but driven from the other tab — press ▶ here to "
                "take it over. Only one view advances the frame; two would "
                "play the run at twice the rate.")
        return s.frame_idx

    def speed_selector(self, label: str = "Replay speed") -> str:
        """The replay-speed control, bound to the shared playback state.

        It goes through the same two-way sync as the scenario selector,
        and for the same reason: writing the shared value into the widget
        slot before drawing it lands on top of the choice the user just
        made — Streamlit has already stored that choice by the time the
        script runs — so the control silently snapped back to its previous
        value every time.
        """
        import streamlit as st
        from adaptive_lidar.visualization import session as _sess

        s = self.state
        before = s.speed
        sk = f"{self.key}_speed"
        _sess.sync(sk, _sess.SPEED_KEY, s.speed)
        chosen = st.selectbox(
            label, list(SPEEDS), key=sk,
            help="How long each frame is held for. It changes nothing the "
                 "pipeline measured. One frame is shown per redraw, so a "
                 "setting faster than the redraw plays at the redraw rate "
                 "rather than skipping frames. Set FOVEA_PROFILE=1 to "
                 "print the real per-redraw cost.")
        s.speed = chosen
        if chosen != before:
            # The interval feeds run_every, which is fixed when the
            # fragment is decorated, so a new speed only takes hold on an
            # app rerun.
            s.started_at = None
            PROFILE.log("speed", f"{self.key} -> {chosen}")
            _sess.refresh("app")
        return chosen

    def tick_if_playing(self, now: Optional[float] = None) -> bool:
        """Called at the END of the fragment body, once it has drawn.

        Advances by AT MOST ONE frame, and only once the interval has
        elapsed. Deriving the index from the clock instead — frame =
        elapsed / interval — looks right on paper and is wrong in
        practice: a redraw costs far more than the 100 ms a frame is meant
        to be shown for, so every redraw jumped three or four frames and
        the run cycled through the same three of twelve. Showing every
        frame in order, a little slower than real time, is what a demo
        needs; the readout calls it a replay rate for exactly this reason.

        Returns True if the frame index moved.
        """
        import time as _time
        s = self.state
        if not s.playing or s.n_frames <= 1:
            s.started_at = None
            return False
        t = _time.monotonic() if now is None else float(now)
        if s.started_at is None:
            PROFILE.log("clock", f"{self.key} start at {s.frame_idx}")
            s.start_clock(t)
            return False
        if not s.due(t):
            return False
        s.start_clock(t)
        s.frame_idx = s._wrap(s.frame_idx + 1)
        if not s.loop and s.frame_idx >= s.last:
            s.playing = False
        return True
