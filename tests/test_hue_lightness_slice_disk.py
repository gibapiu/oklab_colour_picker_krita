import numpy as np
import pytest

pytest.importorskip("pytestqt")
pytest.importorskip("PyQt5")

from PyQt5 import QtCore, QtGui

from oklab_colour_picker.models import HueLightnessSliceModel
from oklab_colour_picker.ui.selectors import HueLightnessSliceDiskWidget
from oklab_colour_picker.domain.colour_state import ColourIntent


def _widget(model):
    return HueLightnessSliceDiskWidget(model)


def _selector(model):
    from oklab_colour_picker.ui.selectors.selector import SelectorWidget

    return SelectorWidget(model)


def test_disk_widget_picks_through_lightness_overlay(qtbot):
    widget = _widget(HueLightnessSliceModel(chroma=0.03))
    widget.resize(120, 120)
    qtbot.addWidget(widget)
    widget.show()

    commits = []
    widget.committed.connect(lambda c, _cs=commits: _cs.append(_paint_of(c)))

    pos = QtCore.QPoint(75, 60)
    _send_mouse(widget, QtCore.QEvent.MouseButtonPress, pos, QtCore.Qt.LeftButton, QtCore.Qt.LeftButton)
    _send_mouse(widget, QtCore.QEvent.MouseButtonRelease, pos, QtCore.Qt.LeftButton, QtCore.Qt.NoButton)

    expected = widget.model.color_at_position((pos.x(), pos.y()), (widget.width(), widget.height()))
    assert expected is not None
    assert len(commits) == 1
    np.testing.assert_allclose(commits[0], expected)


def test_disk_widget_renders_lightness_guide_rings_on_top_of_base(qtbot):
    from oklab_colour_picker.ui.selectors.selector import SelectorWidget

    model = HueLightnessSliceModel(chroma=0.03)
    overlay = _widget(model)
    overlay.resize(81, 81)
    qtbot.addWidget(overlay)
    overlay.show()

    bare = _selector(model)
    bare.resize(81, 81)
    qtbot.addWidget(bare)
    bare.show()

    overlay_pixels = _render_to_rgba_array(overlay)
    bare_pixels = _render_to_rgba_array(bare)

    assert overlay_pixels.shape == bare_pixels.shape
    diff = np.any(overlay_pixels != bare_pixels, axis=-1)
    assert int(diff.sum()) > 100

    cx, cy = 40.0, 40.0
    radius_px = 40.0 * 0.50
    cardinals = [
        (int(round(cx + radius_px)), int(cy)),
        (int(round(cx - radius_px)), int(cy)),
        (int(cx), int(round(cy + radius_px))),
        (int(cx), int(round(cy - radius_px))),
    ]
    assert any(diff[y, x] for x, y in cardinals)


def test_disk_widget_indicator_follows_click_on_achromatic_slice(qtbot):
    # At chroma=0 every angle yields the same greyscale OKLab, so the model
    # can't recover hue from the colour. The widget must still place the
    # indicator at the click point instead of snapping to the hue=0 axis.
    widget = _widget(HueLightnessSliceModel(chroma=0.0))
    widget.resize(121, 121)
    qtbot.addWidget(widget)
    widget.show()

    pos = QtCore.QPoint(60, 20)
    _send_mouse(widget, QtCore.QEvent.MouseButtonPress, pos, QtCore.Qt.LeftButton, QtCore.Qt.LeftButton)
    _send_mouse(widget, QtCore.QEvent.MouseButtonRelease, pos, QtCore.Qt.LeftButton, QtCore.Qt.NoButton)

    indicator = widget.indicator_position()
    assert indicator is not None
    assert indicator == pytest.approx((float(pos.x()), float(pos.y())))


def test_disk_widget_drops_achromatic_override_after_resize(qtbot):
    # The recorded click point is absolute widget pixels, so it must not
    # survive a resize: the indicator should fall back to model placement
    # rather than report the stale pre-resize pixel.
    widget = _widget(HueLightnessSliceModel(chroma=0.0))
    widget.resize(121, 121)
    qtbot.addWidget(widget)
    widget.show()

    pos = QtCore.QPoint(60, 20)
    _send_mouse(widget, QtCore.QEvent.MouseButtonPress, pos, QtCore.Qt.LeftButton, QtCore.Qt.LeftButton)
    _send_mouse(widget, QtCore.QEvent.MouseButtonRelease, pos, QtCore.Qt.LeftButton, QtCore.Qt.NoButton)
    assert widget.indicator_position() == pytest.approx((float(pos.x()), float(pos.y())))

    widget.resize(201, 201)
    QtCore.QCoreApplication.processEvents()
    indicator = widget.indicator_position()

    assert indicator is not None
    assert indicator != pytest.approx((float(pos.x()), float(pos.y())))


def _render_to_rgba_array(widget) -> np.ndarray:
    image = QtGui.QImage(widget.size(), QtGui.QImage.Format_ARGB32)
    image.fill(0)
    widget.render(image)
    ptr = image.bits()
    ptr.setsize(image.byteCount())
    raw = np.frombuffer(ptr, dtype=np.uint8).reshape(image.height(), image.width(), 4).copy()
    return np.dstack((raw[..., 2], raw[..., 1], raw[..., 0], raw[..., 3]))


def _send_mouse(widget, event_type, pos, button, buttons):
    event = QtGui.QMouseEvent(event_type, pos, button, buttons, QtCore.Qt.NoModifier)
    QtCore.QCoreApplication.sendEvent(widget, event)
    assert event.isAccepted()


def _paint_of(c):
    if c is None:
        return None
    return c.paint_oklab if isinstance(c, ColourIntent) else c
