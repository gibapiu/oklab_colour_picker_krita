import numpy as np
import pytest

pytest.importorskip("pytestqt")
pytestmark = pytest.mark.qt

from oklab_colour_picker.infrastructure.qt_facade import QtCore, QtGui
from oklab_colour_picker.models import HueLightnessSliceModel, LightnessSliceModel
from oklab_colour_picker.render.renderers import render_rgba
from oklab_colour_picker.ui.selectors import (
    HueLightnessSliceDiskWidget,
    LightnessSliceDiskWidget,
    SelectorWidget,
)
from tests.qt_helpers import render_pixels, send_mouse


class ContourDisk(LightnessSliceDiskWidget):
    def _paint_chroma_rings(self, painter):
        pass


class RectangularSlice(LightnessSliceModel):
    def colors_at_positions(self, x, y, size):
        x, y = np.broadcast_arrays(x, y)
        colours = np.zeros(x.shape + (3,))
        colours[..., 0] = self.lightness
        return colours, (x >= 10) & (x < size[0] - 10) & (y >= 10) & (y < size[1] - 10)


@pytest.mark.parametrize("lightness", [0.25, 0.9])
def test_outline_has_light_and_dark_bands_inside_the_gamut(qtbot, lightness):
    widget = ContourDisk(RectangularSlice(lightness))
    widget.resize(81, 81)
    qtbot.addWidget(widget)
    pixels = render_pixels(widget)
    light_band = pixels[40, 10, :3].mean()
    dark_band = pixels[40, 11, :3].mean()
    interior = pixels[40, 13, :3].mean()
    assert light_band > dark_band + 30
    assert dark_band < interior - 10


@pytest.mark.parametrize("widget_type, model, guide", [
    (LightnessSliceDiskWidget, LightnessSliceModel(0.5), (46, 40)),
    (HueLightnessSliceDiskWidget, HueLightnessSliceModel(0.03), (60, 40)),
])
def test_disk_guides_are_visible_and_picks_reach_the_model(qtbot, widget_type, model, guide):
    widget = widget_type(model)
    bare = SelectorWidget(model)
    for item in (widget, bare):
        item.resize(81, 81)
        qtbot.addWidget(item)
    actual, background = render_pixels(widget), render_pixels(bare)
    x, y = guide
    assert np.any(actual[y, x] != background[y, x])

    commits = []
    widget.committed.connect(commits.append)
    point = QtCore.QPoint(50, 40)
    send_mouse(widget, "press", point)
    send_mouse(widget, "release", point)
    assert len(commits) == 1
    np.testing.assert_allclose(commits[0].paint_oklab, model.color_at_position((50, 40), (81, 81)))


@pytest.mark.parametrize("size, device_ratio", [(501, 1), (1001, 1), (501, 2)])
def test_blue_outline_keeps_transparent_gap_clear(qtbot, size, device_ratio):
    model = LightnessSliceModel(0.426041573241465)
    widget = ContourDisk(model)
    widget.resize(size, size)
    qtbot.addWidget(widget)
    actual = render_pixels(widget, device_ratio)
    base = render_rgba(model, (size, size))
    base = np.repeat(np.repeat(base, device_ratio, axis=0), device_ratio, axis=1)
    mask = base[..., 3] != 0
    np.testing.assert_array_equal(actual[..., 3] != 0, mask)
    assert np.any(actual[mask] != base[mask])


@pytest.mark.parametrize("change", ["model", "size"])
def test_outline_refreshes_after_model_or_size_change(qtbot, change):
    widget = ContourDisk(LightnessSliceModel(0.4))
    widget.resize(81, 81)
    qtbot.addWidget(widget)
    before = render_pixels(widget)
    if change == "model":
        widget.set_model(LightnessSliceModel(0.8))
    else:
        widget.resize(121, 121)
    actual = render_pixels(widget)
    fresh = ContourDisk(widget.model)
    fresh.resize(widget.size())
    qtbot.addWidget(fresh)
    np.testing.assert_array_equal(actual, render_pixels(fresh))
    assert actual.shape != before.shape or np.any(actual != before)


def test_outline_preserves_existing_painter_clip(qtbot):
    widget = ContourDisk(LightnessSliceModel(0.5))
    widget.resize(101, 101)
    qtbot.addWidget(widget)
    image = QtGui.QImage(widget.size(), QtGui.QImage.Format.Format_RGBA8888)
    image.fill(0)
    painter = QtGui.QPainter(image)
    painter.setClipRect(QtCore.QRect(0, 0, 50, 101))
    widget._paint_gamut_contour(painter)
    assert painter.clipBoundingRect() == QtCore.QRectF(0, 0, 50, 101)
    painter.end()
    pixels = image.bits()
    pixels.setsize(image.sizeInBytes())
    alpha = np.frombuffer(pixels, dtype=np.uint8).reshape(101, 101, 4)[..., 3]
    assert np.any(alpha[:, :50])
    assert not np.any(alpha[:, 50:])


def test_achromatic_pick_preserves_hue_when_resized(qtbot):
    widget = HueLightnessSliceDiskWidget(HueLightnessSliceModel(0))
    widget.resize(121, 121)
    qtbot.addWidget(widget)
    widget.show()
    point = QtCore.QPoint(60, 20)
    send_mouse(widget, "press", point)
    send_mouse(widget, "release", point)
    assert widget.indicator_position() == (60, 20)
    widget.resize(201, 201)
    assert widget.indicator_position() == pytest.approx((100, 100 / 3))
