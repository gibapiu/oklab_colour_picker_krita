import numpy as np

from oklab_colour_picker.domain.colour_state import ColourIntent
from oklab_colour_picker.domain.readout_interaction import (
    EditExit,
    ReadoutAction,
    ReadoutSession,
    ReadoutState,
)
from tests.helpers import presented_colour


def _presented(rgb: tuple[float, float, float]):
    return presented_colour(np.array(rgb, dtype=float))


def test_initial_seed_sets_current_and_revert_baseline():
    session = ReadoutSession()
    colour = _presented((0.5, 0.0, 0.0))

    result = session.seed_initial(colour)

    assert result.action is ReadoutAction.APPLY
    assert result.colour is colour
    assert session.current is colour
    assert session.previous is colour
    assert session.state is ReadoutState.IDLE


def test_committed_broadcast_advances_previous_but_preview_does_not():
    session = ReadoutSession()
    first = _presented((0.5, 0.0, 0.0))
    preview = _presented((0.6, 0.01, 0.0))
    committed = _presented((0.7, 0.02, 0.0))
    session.show_colour(first, committed=True)

    session.show_colour(preview, committed=False, preview=True)

    assert session.current is preview
    assert session.previous is None

    session.show_colour(committed, committed=True)

    assert session.current is committed
    assert session.previous is preview


def test_edit_cancel_applies_latest_latched_colour():
    session = ReadoutSession()
    original = _presented((0.5, 0.0, 0.0))
    external_a = _presented((0.6, 0.01, 0.0))
    external_b = _presented((0.7, 0.02, 0.0))
    session.show_colour(original, committed=True)
    session.begin_edit()

    assert session.show_colour(external_a, committed=True).action is ReadoutAction.NONE
    assert session.show_colour(external_b, committed=False, preview=True).action is ReadoutAction.NONE

    result = session.finish_edit(EditExit.CANCEL)

    assert result.action is ReadoutAction.APPLY
    assert result.colour is external_b
    assert session.current is external_b
    assert session.previous is None


def test_edit_commit_discards_latched_colour():
    session = ReadoutSession()
    original = _presented((0.5, 0.0, 0.0))
    external = _presented((0.7, 0.02, 0.0))
    session.show_colour(original, committed=True)
    session.begin_edit()
    session.show_colour(external, committed=True)

    result = session.finish_edit(EditExit.COMMIT)

    assert result.action is ReadoutAction.NONE
    assert session.current is original
    assert session.state is ReadoutState.IDLE


def test_draft_preview_echo_syncs_presentation_without_latching():
    session = ReadoutSession()
    original = _presented((0.5, 0.0, 0.0))
    draft = ColourIntent.from_lch(0.7, 0.02, 1.0)
    echo = presented_colour(draft)
    session.show_colour(original, committed=True)
    session.begin_edit()
    session.set_draft(draft)

    result = session.show_colour(echo, committed=False, preview=True)

    assert result.action is ReadoutAction.SYNC_DRAFT_PRESENTATION
    assert result.colour is echo
    assert session.current is original
