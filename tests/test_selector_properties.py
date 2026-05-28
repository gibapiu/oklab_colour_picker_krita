import math

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings, strategies as st

from oklab_colour_picker import color_math
from oklab_colour_picker.controller import normalize_oklab_for_krita
from oklab_colour_picker.selector_models import (
    HueLightnessSliceModel,
    LightnessChromaSliceModel,
    LightnessSliceModel,
)
from oklab_colour_picker.selector_interaction import StateKind


MODEL_CASES = (
    ("lightness-slice", LightnessSliceModel(lightness=0.55)),
    ("hue-lightness-slice", HueLightnessSliceModel(chroma=0.03)),
    ("lightness-chroma-slice", LightnessChromaSliceModel(hue=1.0)),
)
STATE_CASES = (
    StateKind.IDLE,
    StateKind.DRAGGING,
    StateKind.KEYBOARD,
    StateKind.PINNED,
)
_QT_APP = None


@settings(
    max_examples=200,
    derandomize=True,
    deadline=None,
    # The lightness-slice case rejects ~88% of random pixel draws as OOG.
    suppress_health_check=[HealthCheck.filter_too_much],
)
@given(
    case=st.sampled_from(MODEL_CASES),
    width=st.integers(min_value=16, max_value=96),
    height=st.integers(min_value=16, max_value=96),
    x_fraction=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    y_fraction=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)
def test_p1_model_position_colour_round_trip_is_stable_within_quantization(case, width, height, x_fraction, y_fraction):
    _name, model = case
    size = (width, height)
    position = (x_fraction * (width - 1.0), y_fraction * (height - 1.0))
    colour = model.color_at_position(position, size)
    assume(colour is not None)

    round_tripped_position = model.position_for_intent(color_math.oklab_to_oklch(colour), size)
    assume(round_tripped_position is not None)
    round_tripped_colour = model.color_at_position(round_tripped_position, size)

    assert round_tripped_colour is not None
    np.testing.assert_allclose(
        normalize_oklab_for_krita(round_tripped_colour),
        normalize_oklab_for_krita(colour),
        atol=1e-12,
    )


@settings(
    max_examples=200,
    derandomize=True,
    deadline=None,
    suppress_health_check=[HealthCheck.filter_too_much],
)
@given(
    case=st.sampled_from(MODEL_CASES),
    width=st.integers(min_value=16, max_value=96),
    height=st.integers(min_value=16, max_value=96),
    x_fraction=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    y_fraction=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)
def test_p2_model_position_lookup_is_idempotent_for_same_colour(case, width, height, x_fraction, y_fraction):
    _name, model = case
    size = (width, height)
    colour = model.color_at_position((x_fraction * (width - 1.0), y_fraction * (height - 1.0)), size)
    assume(colour is not None)

    once = model.position_for_intent(color_math.oklab_to_oklch(colour), size)
    twice = model.position_for_intent(color_math.oklab_to_oklch(np.asarray(colour, dtype=float).copy()), size)

    assert once == pytest.approx(twice)


@settings(max_examples=200, derandomize=True, deadline=None)
@given(
    case=st.sampled_from(MODEL_CASES),
    width=st.integers(min_value=40, max_value=96),
    height=st.integers(min_value=40, max_value=96),
    x_fraction=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    y_fraction=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    points=st.lists(
        st.tuples(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        ),
        min_size=1,
        max_size=5,
    ),
)
def test_p3_idle_indicator_is_independent_of_interaction_history(
    case, width, height, x_fraction, y_fraction, points
):
    _ensure_qapp()
    from oklab_colour_picker.widgets import SelectorWidget

    _name, model = case
    size = (width, height)
    colour = model.color_at_position((x_fraction * (width - 1.0), y_fraction * (height - 1.0)), size)
    assume(colour is not None)

    expected = SelectorWidget(model)
    expected.resize(*size)
    expected.show_colour(colour)
    expected_position = expected.indicator_position()

    history = SelectorWidget(model)
    history.resize(*size)
    _drive_gesture_history(history, points)
    history.resize(width + 1, height + 1)
    _process_qt_events()
    history.resize(*size)
    _process_qt_events()
    history.show_colour(colour)

    assert history.state == "IDLE"
    assert history.indicator_position() == pytest.approx(expected_position)


@settings(max_examples=200, derandomize=True, deadline=None)
@given(
    case=st.sampled_from(MODEL_CASES),
    state_kind=st.sampled_from(STATE_CASES),
    width=st.integers(min_value=40, max_value=96),
    height=st.integers(min_value=40, max_value=96),
    x_fraction=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    y_fraction=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    colour=st.tuples(st.floats(0.0, 1.0), st.floats(-0.1, 0.1), st.floats(-0.1, 0.1)),
)
def test_p2_show_colour_echo_idempotence_from_any_state(
    case, state_kind, width, height, x_fraction, y_fraction, colour
):
    _ensure_qapp()
    from oklab_colour_picker.widgets import SelectorWidget

    _name, model = case
    size = (width, height)
    point = (x_fraction * (width - 1.0), y_fraction * (height - 1.0))
    assume(model.color_at_position(point, size) is not None)
    widget = SelectorWidget(model)
    widget.resize(*size)
    assume(_enter_widget_state(widget, state_kind, point))
    normalized = normalize_oklab_for_krita(np.asarray(colour, dtype=float))

    widget.show_colour(normalized)
    once = _selector_snapshot(widget)
    widget.show_colour(normalized)

    assert _selector_snapshot(widget) == once


@settings(max_examples=200, derandomize=True, deadline=None)
@given(
    case=st.sampled_from(MODEL_CASES),
    width=st.integers(min_value=16, max_value=96),
    height=st.integers(min_value=16, max_value=96),
    points=st.lists(
        st.tuples(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        ),
        min_size=1,
        max_size=5,
    ),
)
def test_p4_no_orphan_anchor_after_random_gesture_sequence(case, width, height, points):
    _ensure_qapp()
    from oklab_colour_picker.widgets import SelectorWidget

    _name, model = case
    widget = SelectorWidget(model)
    widget.resize(width + 40, height + 40)
    _drive_gesture_history(widget, points)
    actual_w, actual_h = widget.width(), widget.height()
    widget.resize(actual_w + 17, actual_h + 13)
    _process_qt_events()

    assert widget.state == "IDLE"
    assert widget.anchor is None
    widget.close()


def test_edge_hue_wrap_position_is_stable_across_zero_tau():
    zero_model = LightnessChromaSliceModel(hue=0.0)
    tau_model = LightnessChromaSliceModel(hue=math.tau - 1e-12)
    colour = color_math.oklch_to_oklab([0.5, 0.02, 0.0])

    zero_position = zero_model.position_for_intent(color_math.oklab_to_oklch(colour), (101, 101))
    tau_position = tau_model.position_for_intent(color_math.oklab_to_oklch(colour), (101, 101))

    assert zero_position is not None
    assert tau_position == pytest.approx(zero_position, abs=1e-6)


@pytest.mark.parametrize(("lightness", "expected_y"), [(0.0, 100.0), (1.0, 0.0)])
def test_edge_lightness_chroma_slice_accepts_achromatic_lightness_boundaries(lightness, expected_y):
    model = LightnessChromaSliceModel(hue=0.0)
    colour = color_math.oklch_to_oklab([lightness, 0.0, 0.0])

    position = model.position_for_intent(color_math.oklab_to_oklch(colour), (101, 101))
    round_tripped = model.color_at_position((0.0, expected_y), (101, 101))

    assert position == pytest.approx((0.0, expected_y), abs=1e-6)
    assert round_tripped is not None
    np.testing.assert_allclose(round_tripped, colour, atol=1e-12)


@pytest.mark.parametrize("lightness", [0.0, 1.0])
def test_edge_hue_lightness_slice_rejects_positive_chroma_at_lightness_boundaries(lightness):
    model = HueLightnessSliceModel(chroma=0.05)
    colour = color_math.oklch_to_oklab([lightness, 0.05, 0.0])

    assert model.position_for_intent(color_math.oklab_to_oklch(colour), (101, 101)) is None


def test_edge_quantization_boundary_colours_compare_equal_after_krita_normalization():
    raw = np.array([0.55, 0.02, -0.03])
    normalized_once = normalize_oklab_for_krita(raw)

    np.testing.assert_allclose(
        normalize_oklab_for_krita(normalized_once),
        normalized_once,
        atol=0.0,
    )


def _ensure_qapp():
    global _QT_APP

    pytest.importorskip("PyQt5")
    from PyQt5 import QtWidgets

    _QT_APP = QtWidgets.QApplication.instance() or _QT_APP or QtWidgets.QApplication([])
    return _QT_APP


def _enter_widget_state(widget, state_kind, point):
    from PyQt5 import QtCore, QtGui, QtWidgets

    widget.show()
    _process_qt_events()
    qpoint = QtCore.QPoint(
        int(np.clip(round(point[0]), 0, widget.width() - 1)),
        int(np.clip(round(point[1]), 0, widget.height() - 1)),
    )
    colour = widget.model.color_at_position(
        (qpoint.x(), qpoint.y()), (widget.width(), widget.height())
    )
    if colour is None:
        return False
    if state_kind is StateKind.IDLE:
        widget.show_colour(colour)
        return widget.state == "IDLE"
    if state_kind is StateKind.DRAGGING:
        _send_mouse(
            widget,
            QtCore.QEvent.MouseButtonPress,
            qpoint,
            QtCore.Qt.LeftButton,
            QtCore.Qt.LeftButton,
        )
        return widget.state == "DRAGGING"
    if state_kind is StateKind.PINNED:
        _send_mouse(
            widget,
            QtCore.QEvent.MouseButtonPress,
            qpoint,
            QtCore.Qt.LeftButton,
            QtCore.Qt.LeftButton,
        )
        _send_mouse(
            widget,
            QtCore.QEvent.MouseButtonRelease,
            qpoint,
            QtCore.Qt.LeftButton,
            QtCore.Qt.NoButton,
        )
        return widget.state == "PINNED"
    if state_kind is StateKind.KEYBOARD:
        widget.show_colour(colour)
        for key in (
            QtCore.Qt.Key_Right,
            QtCore.Qt.Key_Left,
            QtCore.Qt.Key_Down,
            QtCore.Qt.Key_Up,
        ):
            event = QtGui.QKeyEvent(QtCore.QEvent.KeyPress, key, QtCore.Qt.NoModifier)
            QtWidgets.QApplication.sendEvent(widget, event)
            if event.isAccepted() and widget.state == "KEYBOARD":
                return True
        return False
    raise AssertionError(f"unhandled state kind: {state_kind!r}")


def _drive_gesture_history(widget, points):
    from PyQt5 import QtCore

    widget.show()
    _process_qt_events()
    actual_w, actual_h = widget.width(), widget.height()
    qpoints = [
        QtCore.QPoint(int(fx * (actual_w - 1)), int(fy * (actual_h - 1)))
        for fx, fy in points
    ]

    _send_mouse(
        widget,
        QtCore.QEvent.MouseButtonPress,
        qpoints[0],
        QtCore.Qt.LeftButton,
        QtCore.Qt.LeftButton,
    )
    for point in qpoints[1:]:
        _send_mouse(
            widget,
            QtCore.QEvent.MouseMove,
            point,
            QtCore.Qt.NoButton,
            QtCore.Qt.LeftButton,
        )
    _send_mouse(
        widget,
        QtCore.QEvent.MouseButtonRelease,
        qpoints[-1],
        QtCore.Qt.LeftButton,
        QtCore.Qt.NoButton,
    )


def _send_mouse(widget, event_type, point, button, buttons):
    from PyQt5 import QtCore, QtGui, QtWidgets

    QtWidgets.QApplication.sendEvent(
        widget,
        QtGui.QMouseEvent(
            event_type, QtCore.QPointF(point), button, buttons, QtCore.Qt.NoModifier
        ),
    )


def _process_qt_events():
    from PyQt5 import QtWidgets

    QtWidgets.QApplication.processEvents()


def _selector_snapshot(widget):
    return (
        widget.state,
        _rounded_tuple(widget.anchor),
        _rounded_tuple(widget.indicator_position()),
        _rounded_tuple(widget.selected_colour),
    )


def _rounded_tuple(values):
    if values is None:
        return None
    return tuple(np.round(np.asarray(values, dtype=float), 12))
