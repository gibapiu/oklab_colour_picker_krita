"""Explicit invariant + coverage suite for the SelectorWidget state machine.

One named test per state-machine invariant plus state/transition coverage.
Deterministic, offscreen, no sleeps.
"""

import numpy as np
import pytest

pytest.importorskip("pytestqt")
pytest.importorskip("PyQt5")

from PyQt5 import QtCore, QtGui, QtWidgets

from oklab_colour_picker.domain import color_math
from oklab_colour_picker.app.controller import normalize_oklab_for_krita
from oklab_colour_picker.models import LightnessChromaSliceModel, LightnessSliceModel
from oklab_colour_picker.ui.selectors import SelectorWidget
from oklab_colour_picker.domain.colour_state import ColourIntent
from tests.helpers import presented_colour


SIZE = (64, 32)


def _shown(qtbot, model=None, size=SIZE):
    widget = SelectorWidget(model or LightnessChromaSliceModel(hue=0.0))
    widget.resize(*size)
    qtbot.addWidget(widget)
    widget.show()
    return widget


def _present(colour):
    return presented_colour(colour)


def _mouse(widget, kind, point, button, buttons):
    event = QtGui.QMouseEvent(
        kind, QtCore.QPointF(point), button, buttons, QtCore.Qt.NoModifier
    )
    QtWidgets.QApplication.sendEvent(widget, event)


def _press_release(widget, point):
    _mouse(widget, QtCore.QEvent.MouseButtonPress, point, QtCore.Qt.LeftButton, QtCore.Qt.LeftButton)
    _mouse(widget, QtCore.QEvent.MouseButtonRelease, point, QtCore.Qt.LeftButton, QtCore.Qt.NoButton)


def _valid_point(model, size=SIZE):
    width, height = size
    for y in range(height):
        for x in range(width):
            if model.color_at_position((x, y), size) is not None:
                return QtCore.QPoint(x, y)
    raise AssertionError("no valid point")


def _invalid_point(model, size=SIZE):
    width, height = size
    for point in (
        QtCore.QPoint(0, 0),
        QtCore.QPoint(width - 1, 0),
        QtCore.QPoint(width - 1, height - 1),
        QtCore.QPoint(0, height - 1),
    ):
        if model.color_at_position((point.x(), point.y()), size) is None:
            return point
    raise AssertionError("no invalid point")


# -- Anchor lifetime -------------------------------------------------------


def _distinct_valid_colour(model, reference, size=SIZE):
    width, height = size
    for y in range(height):
        for x in range(width):
            colour = model.color_at_position((x, y), size)
            if colour is not None and not np.allclose(colour, reference, atol=1e-3):
                return colour
    raise AssertionError("no second distinct valid colour")


def test_inv1_anchor_exists_only_in_anchored_states(qtbot):
    widget = _shown(qtbot)
    assert widget.state == "IDLE"
    assert widget.anchor is None

    point = _valid_point(widget.model)
    _press_release(widget, point)
    assert widget.state == "PINNED"
    assert widget.anchor == pytest.approx((float(point.x()), float(point.y())))

    # Resize while PINNED drops the absolute-pixel anchor (no stale memory).
    widget.resize(128, 64)
    assert widget.state == "IDLE"
    assert widget.anchor is None


def test_inv1_failed_drag_retains_no_anchor(qtbot):
    widget = _shown(qtbot, LightnessSliceModel(lightness=0.5), (40, 80))
    invalid = _invalid_point(widget.model, (40, 80))

    _mouse(widget, QtCore.QEvent.MouseButtonPress, invalid, QtCore.Qt.LeftButton, QtCore.Qt.LeftButton)
    _mouse(widget, QtCore.QEvent.MouseButtonRelease, invalid, QtCore.Qt.LeftButton, QtCore.Qt.NoButton)

    assert widget.state == "IDLE"
    assert widget.anchor is None


# -- Indicator is a pure function of state ---------------------------------


def test_inv2_idle_indicator_is_model_derived(qtbot):
    widget = _shown(qtbot)
    colour = widget.model.color_at_position((20, 16), SIZE)
    widget.show_colour(_present(colour))

    assert widget.state == "IDLE"
    assert widget.indicator_position() == pytest.approx(
        widget.model.position_for_intent(color_math.oklab_to_oklch(colour), SIZE)
    )


def test_inv2_anchored_indicator_is_the_anchor(qtbot):
    widget = _shown(qtbot)
    point = _valid_point(widget.model)
    _press_release(widget, point)

    assert widget.indicator_position() == pytest.approx(
        (float(point.x()), float(point.y()))
    )


# -- Echo absorption is local ----------------------------------------------


def test_inv3_pinned_swallows_its_own_echo(qtbot):
    widget = _shown(qtbot)
    point = _valid_point(widget.model)
    _press_release(widget, point)
    assert widget.state == "PINNED"

    widget.show_colour(_present(widget.selected_colour))

    assert widget.state == "PINNED"
    log = widget.transition_log
    assert not any(
        log[i : i + 3] == ("PINNED", "IDLE", "PINNED") for i in range(len(log) - 2)
    )


def test_inv3_pinned_yields_to_a_different_colour(qtbot):
    widget = _shown(qtbot)
    _press_release(widget, _valid_point(widget.model))
    pinned = widget.selected_colour
    assert widget.state == "PINNED"

    other = _distinct_valid_colour(widget.model, pinned)
    widget.show_colour(_present(other))

    assert widget.state == "IDLE"
    np.testing.assert_allclose(widget.selected_colour, other)


# -- Quantization parity ---------------------------------------------------


def test_inv4_quantized_equal_echo_is_swallowed(qtbot):
    widget = _shown(qtbot)
    _press_release(widget, _valid_point(widget.model))
    pinned = widget.selected_colour
    assert widget.state == "PINNED"

    # Differs in raw float but collapses under Krita 8-bit normalization.
    near = pinned + np.array([1e-9, -1e-9, 1e-9])
    np.testing.assert_array_equal(
        normalize_oklab_for_krita(near), normalize_oklab_for_krita(pinned)
    )
    transitions_before = len(widget.transition_log)
    widget.show_colour(_present(near))

    assert widget.state == "PINNED"
    assert len(widget.transition_log) == transitions_before


# -- One writer ------------------------------------------------------------


def test_inv5_show_colour_emits_no_intent_signals(qtbot):
    widget = _shown(qtbot)
    previews, commits = [], []
    widget.previewed.connect(lambda c, _ps=previews: _ps.append(_paint_of(c)))
    widget.committed.connect(lambda c, _cs=commits: _cs.append(_paint_of(c)))

    widget.show_colour(_present(widget.model.color_at_position((20, 16), SIZE)))
    _press_release(widget, _valid_point(widget.model))
    previews.clear()
    commits.clear()
    widget.show_colour(_present(widget.model.color_at_position((30, 8), SIZE)))

    assert previews == []
    assert commits == []


# -- Out-of-gamut continuity -----------------------------------------------


def test_inv6_drag_leaving_gamut_stays_continuous_and_commits_last_valid(qtbot):
    widget = _shown(qtbot, LightnessChromaSliceModel(hue=0.0))
    valid = QtCore.QPoint(12, 16)
    far = QtCore.QPoint(63, 16)
    assert widget.model.color_at_position((valid.x(), valid.y()), SIZE) is not None
    assert widget.model.color_at_position((far.x(), far.y()), SIZE) is None
    snapped = widget.model.snapped_color_at_position((far.x(), far.y()), SIZE)
    assert snapped is not None

    previews, commits = [], []
    widget.previewed.connect(lambda c, _ps=previews: _ps.append(_paint_of(c)))
    widget.committed.connect(lambda c, _cs=commits: _cs.append(_paint_of(c)))

    _mouse(widget, QtCore.QEvent.MouseButtonPress, valid, QtCore.Qt.LeftButton, QtCore.Qt.LeftButton)
    _mouse(widget, QtCore.QEvent.MouseMove, far, QtCore.Qt.NoButton, QtCore.Qt.LeftButton)
    _mouse(widget, QtCore.QEvent.MouseButtonRelease, far, QtCore.Qt.LeftButton, QtCore.Qt.NoButton)

    assert not any(preview is None for preview in previews)
    assert len(commits) == 1
    np.testing.assert_allclose(commits[0], snapped)
    np.testing.assert_allclose(widget.selected_colour, snapped)


# -- State + transition coverage -------------------------------------------


def test_state_coverage_every_state_is_entered(qtbot):
    widget = _shown(qtbot)
    seen = set()

    seen.add(widget.state)  # IDLE
    point = _valid_point(widget.model)
    _mouse(widget, QtCore.QEvent.MouseButtonPress, point, QtCore.Qt.LeftButton, QtCore.Qt.LeftButton)
    seen.add(widget.state)  # DRAGGING
    _mouse(widget, QtCore.QEvent.MouseButtonRelease, point, QtCore.Qt.LeftButton, QtCore.Qt.NoButton)
    seen.add(widget.state)  # PINNED

    widget.resize(widget.width() + 1, widget.height() + 1)
    widget.resize(*SIZE)
    assert widget.state == "IDLE"
    widget.show_colour(_present(widget.model.color_at_position((point.x(), point.y()), SIZE)))
    press = QtGui.QKeyEvent(QtCore.QEvent.KeyPress, QtCore.Qt.Key_Right, QtCore.Qt.NoModifier)
    QtWidgets.QApplication.sendEvent(widget, press)
    seen.add(widget.state)  # KEYBOARD

    assert {"IDLE", "DRAGGING", "KEYBOARD", "PINNED"} <= seen


def test_transition_guard_false_branches_do_not_change_state(qtbot):
    widget = _shown(qtbot)

    # show_colour while IN-FLIGHT (DRAGGING) is ignored.
    point = _valid_point(widget.model)
    _mouse(widget, QtCore.QEvent.MouseButtonPress, point, QtCore.Qt.LeftButton, QtCore.Qt.LeftButton)
    assert widget.state == "DRAGGING"
    dragging_colour = widget.selected_colour
    widget.show_colour(_present(dragging_colour))
    assert widget.state == "DRAGGING"
    np.testing.assert_allclose(widget.selected_colour, dragging_colour)
    _mouse(widget, QtCore.QEvent.MouseButtonRelease, point, QtCore.Qt.LeftButton, QtCore.Qt.NoButton)

    # Non-LMB press is ignored (guard-false on the press transition).
    state_before = widget.state
    _mouse(widget, QtCore.QEvent.MouseButtonPress, point, QtCore.Qt.RightButton, QtCore.Qt.RightButton)
    assert widget.state == state_before


def _paint_of(c):
    if c is None:
        return None
    return c.paint_oklab if isinstance(c, ColourIntent) else c
