"""
The state the two visualisation tabs share.

Both bugs these tests pin down were reported from the app rather than
caught here, which is the reason the tests exist now:

* every scenario except the default appeared not to work — picking
  ``convoy`` snapped straight back to ``mixed_urban``;
* playback "plays only 3 frames and jumps around randomly".

The first is :func:`session.sync` choosing the wrong direction. The second
is two tabs both advancing one shared index, which ownership prevents.

``session`` imports streamlit lazily inside each function, so a stub in
``sys.modules`` is enough to exercise the real logic without a server.
"""
from __future__ import annotations

import sys
import types

import pytest

from adaptive_lidar.visualization import session as SESSION


@pytest.fixture
def state(monkeypatch):
    """A stand-in ``st.session_state``: a plain dict is the whole API used."""
    store: dict = {}
    stub = types.SimpleNamespace(session_state=store)
    monkeypatch.setitem(sys.modules, "streamlit", stub)
    return store


# ════════════════════════════════════════════════════════════
# Two-way binding
# ════════════════════════════════════════════════════════════
def test_first_use_seeds_both_sides_from_the_default(state):
    SESSION.sync("live_scenario", SESSION.SCENARIO_KEY, "mixed_urban")
    assert state["live_scenario"] == "mixed_urban"
    assert state[SESSION.SCENARIO_KEY] == "mixed_urban"


def test_this_tabs_choice_wins_and_is_published(state):
    """The reported bug: choosing any other scenario snapped back.

    Streamlit writes a changed widget value into session state BEFORE the
    script runs, so a naive "copy shared into widget, draw, copy back"
    overwrites the choice that was just made.
    """
    SESSION.sync("live_scenario", SESSION.SCENARIO_KEY, "mixed_urban")
    state["live_scenario"] = "convoy"          # the user picked convoy
    SESSION.sync("live_scenario", SESSION.SCENARIO_KEY, "mixed_urban")
    assert state["live_scenario"] == "convoy", "the choice was reverted"
    assert state[SESSION.SCENARIO_KEY] == "convoy"


def test_the_other_tabs_choice_arrives(state):
    SESSION.sync("live_scenario", SESSION.SCENARIO_KEY, "mixed_urban")
    SESSION.sync("scene_scenario", SESSION.SCENARIO_KEY, "mixed_urban")
    state["scene_scenario"] = "canopy_over_road"
    SESSION.sync("scene_scenario", SESSION.SCENARIO_KEY, "mixed_urban")
    # ...and the first tab picks it up on its next run
    SESSION.sync("live_scenario", SESSION.SCENARIO_KEY, "mixed_urban")
    assert state["live_scenario"] == "canopy_over_road"


def test_a_choice_survives_repeated_syncs(state):
    """Idempotence. A redraw must not undo a selection."""
    SESSION.sync("live_scenario", SESSION.SCENARIO_KEY, "mixed_urban")
    state["live_scenario"] = "pedestrian_far"
    for _ in range(5):
        SESSION.sync("live_scenario", SESSION.SCENARIO_KEY, "mixed_urban")
    assert state["live_scenario"] == "pedestrian_far"
    assert state[SESSION.SCENARIO_KEY] == "pedestrian_far"


def test_both_tabs_converge_after_either_one_moves(state):
    for tab in ("live_n", "scene_n"):
        SESSION.sync(tab, SESSION.N_FRAMES_KEY, 24)
    state["scene_n"] = 32
    SESSION.sync("scene_n", SESSION.N_FRAMES_KEY, 24)
    SESSION.sync("live_n", SESSION.N_FRAMES_KEY, 24)
    assert state["live_n"] == state["scene_n"] == 32


def test_sync_works_for_a_boolean(state):
    SESSION.sync("live_gate", SESSION.GATE_KEY, True)
    state["live_gate"] = False
    SESSION.sync("live_gate", SESSION.GATE_KEY, True)
    assert state[SESSION.GATE_KEY] is False


# ════════════════════════════════════════════════════════════
# Who drives playback
# ════════════════════════════════════════════════════════════
def test_live_drives_until_something_says_otherwise(state):
    assert SESSION.owns_ticker("live") is True
    assert SESSION.owns_ticker("scene") is False


def test_ownership_transfers_on_a_transport_press(state):
    SESSION.claim_ticker("scene")
    assert SESSION.owns_ticker("scene") is True
    assert SESSION.owns_ticker("live") is False
    SESSION.claim_ticker("live")
    assert SESSION.owns_ticker("live") is True


def test_exactly_one_tab_owns_the_ticker(state):
    """Two tickers on one index play the run at twice the rate."""
    for owner in ("live", "scene"):
        SESSION.claim_ticker(owner)
        owning = [t for t in ("live", "scene") if SESSION.owns_ticker(t)]
        assert owning == [owner]


# ════════════════════════════════════════════════════════════
# Selection
# ════════════════════════════════════════════════════════════
def test_selection_is_one_slot_both_tabs_read(state):
    assert SESSION.get_selection() is None
    SESSION.set_selection("a-selection")
    assert SESSION.get_selection() == "a-selection"
    assert state[SESSION.SELECTION_KEY] == "a-selection"


def test_the_shared_keys_are_not_widget_keys():
    """Streamlit refuses two widgets with one key, so the shared slots
    must not be the keys either tab binds a widget to."""
    for key in (SESSION.SCENARIO_KEY, SESSION.N_FRAMES_KEY,
                SESSION.GATE_KEY, SESSION.TICKER_KEY):
        assert not key.startswith(("drive_", "scene_", "live_"))


# ════════════════════════════════════════════════════════════
# Self-tuning playback pace
# ════════════════════════════════════════════════════════════
def test_pacing_never_schedules_faster_than_a_redraw(state):
    """The reported symptom: on a weaker laptop the scene went white and
    playback stuttered and restarted.

    A fixed interval measured on one machine schedules reruns faster than
    another can serve them; they queue, arrive out of order, and playback
    appears to restart. The interval is derived from what a redraw here
    actually cost.
    """
    for _ in range(6):
        SESSION.record_redraw("scene", 0.62)        # a slow machine
    assert SESSION.redraw_cost("scene") == pytest.approx(0.62, abs=0.05)
    # A request for 10 fps is honoured at the rate the machine can hold.
    assert SESSION.paced_interval("scene", 0.1) >= 0.62


def test_a_fast_machine_is_not_held_back(state):
    for _ in range(10):
        SESSION.record_redraw("live", 0.03)
    # The requested speed wins when the machine can keep up with it.
    assert SESSION.paced_interval("live", 0.5, floor=0.1) ==         pytest.approx(0.5)


def test_the_floor_applies_before_any_measurement(state):
    assert SESSION.paced_interval("scene", 0.01, floor=0.18) >= 0.18


def test_it_backs_off_fast_and_recovers_slowly(state):
    """A machine that just struggled should slow down at once.

    Earning the speed back gradually avoids oscillating between a rate
    that works and one that does not.
    """
    for _ in range(10):
        SESSION.record_redraw("scene", 0.10)
    quick = SESSION.redraw_cost("scene")
    SESSION.record_redraw("scene", 1.00)            # one bad redraw
    assert SESSION.redraw_cost("scene") > quick * 3
    slow = SESSION.redraw_cost("scene")
    SESSION.record_redraw("scene", 0.10)            # one good one
    assert SESSION.redraw_cost("scene") > slow * 0.7


def test_the_pace_is_clamped_to_something_sane(state):
    SESSION.record_redraw("scene", 99.0)
    assert SESSION.redraw_cost("scene") <= 3.0
    SESSION.record_redraw("live", 0.0)
    assert SESSION.redraw_cost("live") >= 0.02


def test_each_tab_is_paced_separately(state):
    """The 2D canvas is cheap and the scene is not; one number for both
    would hold the cheap one back."""
    SESSION.record_redraw("live", 0.05)
    SESSION.record_redraw("scene", 0.60)
    assert SESSION.redraw_cost("live") < SESSION.redraw_cost("scene")
