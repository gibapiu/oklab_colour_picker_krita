import math

import numpy as np
import pytest

pytest.importorskip("pytestqt")
pytest.importorskip("PyQt5")

from PyQt5 import QtCore, QtGui, QtWidgets

from oklab_colour_picker.domain import color_math
from oklab_colour_picker.domain.colour_state import ColourIntent
from oklab_colour_picker.models import (
    HueLightnessSliceModel,
    LightnessChromaSliceModel,
    LightnessSliceModel,
)
from oklab_colour_picker.ui.selectors import SelectorWidget
from tests.helpers import presented_colour


def test_indicator_position_comes_from_model(qtbot):
    widget = SelectorWidget(LightnessChromaSliceModel(hue=math.pi / 3.0))
    widget.resize(100, 50)
    qtbot.addWidget(widget)
    colour = color_math.oklch_to_oklab([0.25, 0.02, math.pi / 3.0])

    widget.set_selected_colour(presented_colour(colour))

    expected = widget.model.position_for_intent(
        color_math.oklab_to_oklch(colour),
        _size(widget),
    )
    assert expected is not None
    assert widget.indicator_position() == pytest.approx(expected)


def test_indicator_position_tracks_widget_resize(qtbot):
    widget = SelectorWidget(LightnessChromaSliceModel(hue=math.pi / 3.0))
    widget.resize(80, 40)
    qtbot.addWidget(widget)
    colour = widget.model.color_at_position((30, 12), _size(widget))
    assert colour is not None
    widget.set_selected_colour(presented_colour(colour))

    widget.resize(160, 90)

    expected = widget.model.position_for_intent(
        color_math.oklab_to_oklch(colour),
        _size(widget),
    )
    assert expected is not None
    assert widget.indicator_position() == pytest.approx(expected, abs=1.0)


def test_widget_keeps_intent_lch_and_paint_colour_together(qtbot):
    hue = math.radians(210.0)
    widget = SelectorWidget(HueLightnessSliceModel(chroma=0.0))
    widget.resize(121, 121)
    qtbot.addWidget(widget)

    widget.set_selected_colour(
        presented_colour(ColourIntent.from_lch(0.5, 0.0, hue))
    )
    selected = widget.selected_colour
    assert selected is not None
    selected[:] = 0.0

    assert widget.colour is not None
    assert widget.colour.selector_lch == pytest.approx((0.5, 0.0, hue))
    np.testing.assert_allclose(widget.selected_colour, [0.5, 0.0, 0.0], atol=1e-12)
    assert widget.indicator_position() == pytest.approx(
        (
            60.0 + 30.0 * math.cos(hue),
            60.0 - 30.0 * math.sin(hue),
        )
    )


def test_indicator_position_stays_strict_for_out_of_gamut_leaf_colour(qtbot):
    widget = SelectorWidget(LightnessSliceModel(lightness=0.5))
    widget.resize(101, 101)
    qtbot.addWidget(widget)
    colour = color_math.oklch_to_oklab(
        [0.5, color_math.SRGB_MAX_CHROMA, 0.0]
    )
    lch = color_math.oklab_to_oklch(colour)
    assert widget.model.position_for_intent(lch, _size(widget)) is None
    assert widget.model.geometric_position_for_intent(lch, _size(widget)) is not None

    widget.set_selected_colour(presented_colour(colour))

    assert widget.indicator_position() is None


def test_out_of_gamut_indicator_uses_presented_fallback_selector_position(qtbot):
    hue = math.radians(110.0)
    lightness = 0.05
    chroma = float(color_math.max_chroma_for_lh(lightness, hue)) * 1.2
    model = LightnessChromaSliceModel(hue=hue)
    widget = SelectorWidget(model)
    widget.resize(101, 101)
    qtbot.addWidget(widget)
    intent = ColourIntent.from_lch(lightness, chroma, hue)
    fallback = ColourIntent.from_lch(
        lightness,
        float(color_math.max_chroma_for_lh(lightness, hue)) * 0.5,
        hue,
    )

    widget.set_selected_colour(
        presented_colour(intent, in_gamut=False, fallback=fallback)
    )
    indicator = widget.model_indicator()

    desired = model.geometric_position_for_intent(intent.selector_lch, _size(widget))
    landed = model.position_for_intent(fallback.selector_lch, _size(widget))
    assert desired is not None
    assert landed is not None
    assert len(indicator.rings) == 2
    assert indicator.rings[0].solid is True
    assert indicator.rings[0].position == pytest.approx(desired)
    assert indicator.rings[1].solid is False
    assert indicator.rings[1].position == pytest.approx(landed)
    assert indicator.rings[1].position != pytest.approx(desired)


def test_out_of_gamut_indicator_omits_fallback_off_selector_slice(qtbot):
    widget = SelectorWidget(LightnessChromaSliceModel(hue=0.0))
    widget.resize(101, 101)
    qtbot.addWidget(widget)
    intent = ColourIntent.from_lch(
        0.6,
        color_math.SRGB_MAX_CHROMA,
        math.pi / 2.0,
    )
    fallback = ColourIntent.from_lch(0.6, 0.05, math.pi / 2.0)

    widget.set_selected_colour(
        presented_colour(intent, in_gamut=False, fallback=fallback)
    )

    assert widget.model_indicator().rings == ()


def test_absorbed_echo_with_model_swap_preserves_intent_lch(qtbot):
    widget = SelectorWidget(HueLightnessSliceModel(chroma=0.05))
    widget.resize(121, 121)
    qtbot.addWidget(widget)
    widget.show()
    click = QtCore.QPoint(60, 30)
    _send_mouse(widget, QtCore.QEvent.MouseButtonPress, click, QtCore.Qt.LeftButton)
    _send_mouse(widget, QtCore.QEvent.MouseButtonRelease, click, QtCore.Qt.NoButton)

    assert widget.colour is not None
    lightness, _chroma, hue = widget.colour.selector_lch
    refreshed_intent = ColourIntent.from_lch(lightness, 0.0501, hue)
    new_model = HueLightnessSliceModel(chroma=refreshed_intent.chroma)
    echo_intent = refreshed_intent.with_krita_paint_oklab(
        refreshed_intent.quantized_paint_oklab
    )

    widget.show_colour(
        presented_colour(echo_intent),
        model_factory=lambda: new_model,
    )

    assert widget.model is new_model
    assert widget.colour is not None
    assert widget.colour.selector_lch == pytest.approx(
        refreshed_intent.selector_lch
    )
    assert widget.indicator_position() is not None


def test_invalid_press_on_warm_idle_is_a_no_op(qtbot):
    widget = SelectorWidget(LightnessSliceModel(lightness=0.5))
    widget.resize(121, 121)
    qtbot.addWidget(widget)
    widget.set_selected_colour(
        presented_colour(ColourIntent.from_lch(0.5, 0.05, 0.0))
    )
    rings_before = widget.model_indicator().rings
    assert rings_before
    widget.set_model(InvalidPickModel(widget.model))

    _send_mouse(
        widget,
        QtCore.QEvent.MouseButtonPress,
        QtCore.QPoint(60, 60),
        QtCore.Qt.LeftButton,
    )

    assert widget.model_indicator().rings == rings_before


@pytest.mark.parametrize(
    "model",
    [
        LightnessSliceModel(lightness=0.5),
        HueLightnessSliceModel(chroma=0.03),
        LightnessChromaSliceModel(hue=0.0),
    ],
)
def test_tiny_widget_size_does_not_raise(model, qtbot):
    widget = SelectorWidget(model)
    widget.setMinimumSize(0, 0)
    widget.resize(1, 1)
    qtbot.addWidget(widget)
    widget.set_selected_colour(presented_colour(np.array([0.5, 0.0, 0.0])))

    assert widget.indicator_position() is None
    image = QtGui.QImage(QtCore.QSize(1, 1), QtGui.QImage.Format_RGBA8888)
    painter = QtGui.QPainter(image)
    widget.render(painter)
    painter.end()


def test_paint_event_renders_selector_image(qtbot):
    widget = SelectorWidget(LightnessChromaSliceModel(hue=0.0))
    widget.resize(32, 24)
    qtbot.addWidget(widget)
    widget.show()

    image = QtGui.QImage(widget.size(), QtGui.QImage.Format_RGBA8888)
    image.fill(QtCore.Qt.transparent)
    painter = QtGui.QPainter(image)
    widget.render(painter)
    painter.end()

    assert any(
        QtGui.QColor(image.pixel(x, y)).alpha() != 0
        for y in range(image.height())
        for x in range(image.width())
    )


def _send_mouse(widget, event_type, position, buttons):
    event = QtGui.QMouseEvent(
        event_type,
        QtCore.QPointF(position),
        QtCore.Qt.LeftButton,
        buttons,
        QtCore.Qt.NoModifier,
    )
    QtWidgets.QApplication.sendEvent(widget, event)
    assert event.isAccepted()


def _size(widget):
    return widget.width(), widget.height()


class InvalidPickModel:
    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def color_at_position(self, *args, **kwargs):
        return None

    def selection_at_position(self, *args, **kwargs):
        return None

    def snapped_selector_selection_at_position(self, *args, **kwargs):
        return None
