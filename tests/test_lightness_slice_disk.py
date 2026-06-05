import math

import numpy as np
import pytest

pytest.importorskip("pytestqt")
pytest.importorskip("PyQt5")

from PyQt5 import QtCore, QtGui

from oklab_colour_picker import color_math
from oklab_colour_picker.selector_models import (
    LightnessSliceModel,
)
from oklab_colour_picker.widgets import LightnessSliceDiskWidget
from oklab_colour_picker.colour_state import ColourIntent


def _widget(model):
    return LightnessSliceDiskWidget(model)


def _selector(model):
    from oklab_colour_picker.widgets.selector import SelectorWidget

    return SelectorWidget(model)


def test_disk_widget_picks_through_overlay(qtbot):
    widget = _widget(LightnessSliceModel(lightness=0.6))
    widget.resize(120, 120)
    qtbot.addWidget(widget)
    widget.show()

    commits = []
    widget.committed.connect(lambda c, _cs=commits: _cs.append(_paint_of(c)))

    # Stay close to the centre so the position is in-gamut for any hue.
    pos = QtCore.QPoint(65, 55)
    _send_mouse(widget, QtCore.QEvent.MouseButtonPress, pos, QtCore.Qt.LeftButton, QtCore.Qt.LeftButton)
    _send_mouse(widget, QtCore.QEvent.MouseButtonRelease, pos, QtCore.Qt.LeftButton, QtCore.Qt.NoButton)

    expected = widget.model.color_at_position((pos.x(), pos.y()), (widget.width(), widget.height()))
    assert expected is not None
    assert len(commits) == 1
    np.testing.assert_allclose(commits[0], expected)


def test_disk_widget_renders_overlay_pixels_on_top_of_base(qtbot):
    # Render the disk widget twice at the same model and size: once with the
    # overlay subclass, once with the bare SelectorWidget that paints only
    # the disk image. Both share the same disk pixels, so any rendered
    # difference comes from the rings + gamut contour the subclass adds.
    from oklab_colour_picker.widgets.selector import SelectorWidget

    model = LightnessSliceModel(lightness=0.5)
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
    differing_pixels = int(diff.sum())

    # The contour stroke alone covers ~360 sampled hues × ~3 px (halo + stroke)
    # plus five chroma rings, so the difference should be substantial — in
    # the hundreds of pixels — not a stray rounding artifact.
    assert differing_pixels > 200

    # And the overlay must reach across the disk: the C=0.05 ring sits well
    # inside the leaf at every hue, so at least one of the four cardinal
    # samples on that ring must differ from the bare render.
    cx, cy = 40.0, 40.0
    radius_px = 40.0 * (0.05 / color_math.SRGB_MAX_CHROMA)
    cardinals = [
        (int(round(cx + radius_px)), int(cy)),
        (int(round(cx - radius_px)), int(cy)),
        (int(cx), int(round(cy + radius_px))),
        (int(cx), int(round(cy - radius_px))),
    ]
    differs_on_inner_ring = any(
        diff[y, x] for x, y in cardinals
    )
    assert differs_on_inner_ring


def _render_to_rgba_array(widget) -> np.ndarray:
    image = QtGui.QImage(widget.size(), QtGui.QImage.Format_ARGB32)
    image.fill(0)
    widget.render(image)
    # Format_ARGB32 packs 0xAARRGGBB as uint32; on little-endian the bytes
    # land as B, G, R, A. Re-pack as RGBA so channel asserts read naturally.
    ptr = image.bits()
    ptr.setsize(image.byteCount())
    raw = np.frombuffer(ptr, dtype=np.uint8).reshape(image.height(), image.width(), 4).copy()
    return np.dstack((raw[..., 2], raw[..., 1], raw[..., 0], raw[..., 3]))


def test_set_model_invalidates_gamut_path_cache(qtbot):
    widget = _widget(LightnessSliceModel(lightness=0.5))
    widget.resize(80, 80)
    qtbot.addWidget(widget)
    widget.show()

    # Force the cache to populate.
    first = widget._gamut_path(widget.model.lightness)
    assert first is not None
    assert widget._gamut_path_cache_key is not None

    widget.set_model(LightnessSliceModel(lightness=0.8))
    assert widget._gamut_path_cache_key is None

    second = widget._gamut_path(widget.model.lightness)
    assert second is not None
    # Different lightness → different leaf shape, so a different path object.
    assert second is not first


def test_gamut_contour_uses_180_hue_samples(qtbot):
    widget = _widget(LightnessSliceModel(lightness=0.5))
    widget.resize(80, 80)
    qtbot.addWidget(widget)
    widget.show()

    path = widget._gamut_path(widget.model.lightness)

    assert path is not None
    assert path.elementCount() == 181


def test_resize_invalidates_gamut_path_cache(qtbot):
    widget = _widget(LightnessSliceModel(lightness=0.5))
    widget.resize(80, 80)
    qtbot.addWidget(widget)
    widget.show()

    widget._gamut_path(widget.model.lightness)
    assert widget._gamut_path_cache_key == (0.5, 80, 80)
    assert widget._gamut_contour_cache_key == 0.5

    widget.resize(100, 100)
    assert widget._gamut_path_cache_key is None
    assert widget._gamut_contour_cache_key == 0.5


def test_resize_reuses_expensive_gamut_contour_cache(qtbot, monkeypatch):
    widget = _widget(LightnessSliceModel(lightness=0.5))
    widget.resize(80, 80)
    qtbot.addWidget(widget)
    widget.show()

    first = widget._gamut_path(widget.model.lightness)
    assert first is not None

    def fail_on_boundary_solver(*_args, **_kwargs):
        raise AssertionError("resize should reuse normalized gamut contour")

    monkeypatch.setattr(color_math, "max_chroma_for_lh", fail_on_boundary_solver)
    widget.resize(100, 100)
    second = widget._gamut_path(widget.model.lightness)

    assert second is not None
    assert second is not first


def test_gamut_path_caps_at_disk_radius(qtbot):
    # Pick a lightness whose cusp chroma is comfortably below the disk's
    # radial extent, so every sample on the contour should sit strictly
    # inside the rim.
    widget = _widget(LightnessSliceModel(lightness=0.95))
    widget.resize(101, 101)
    qtbot.addWidget(widget)
    widget.show()

    path = widget._gamut_path(0.95)
    assert path is not None

    radius = (min(widget.width(), widget.height()) - 1) / 2.0
    cx = (widget.width() - 1) / 2.0
    cy = (widget.height() - 1) / 2.0
    bounds = path.boundingRect()
    # Every point on the contour is within the disk radius (allow a 1 px
    # tolerance for rounding).
    assert bounds.left() >= cx - radius - 1
    assert bounds.right() <= cx + radius + 1
    assert bounds.top() >= cy - radius - 1
    assert bounds.bottom() <= cy + radius + 1


def test_drag_past_gamut_leaf_snaps_to_cusp_at_cursor_hue(qtbot):
    # At L=0.5 the +x rim of the disk sits past the per-hue gamut leaf. The
    # base SelectorWidget already pins the preview to the last in-gamut
    # colour visited during the drag (rewrite/ux-keep-last-valid-selection),
    # but that pin tracks the colour from a few pixels in, not the cursor's
    # current hue. With snap, the drag commits the in-gamut *cusp* chroma at
    # the cursor's hue — even when the cursor is well past the leaf.
    widget = _widget(LightnessSliceModel(lightness=0.5))
    widget.resize(101, 101)
    qtbot.addWidget(widget)
    widget.show()

    # Start the drag at a position whose hue differs from the release hue, so
    # "last in-gamut colour" and "snap at release-hue cusp" produce different
    # answers and the test actually checks the snap path.
    start = QtCore.QPoint(55, 60)  # hue ≈ atan2(10, 5) — well above +x axis
    rim = QtCore.QPoint(99, 50)    # past the leaf along hue=0 (+x)

    commits = []
    widget.committed.connect(lambda c, _cs=commits: _cs.append(_paint_of(c)))

    _send_mouse(widget, QtCore.QEvent.MouseButtonPress, start, QtCore.Qt.LeftButton, QtCore.Qt.LeftButton)
    _send_mouse(widget, QtCore.QEvent.MouseMove, rim, QtCore.Qt.NoButton, QtCore.Qt.LeftButton)
    _send_mouse(widget, QtCore.QEvent.MouseButtonRelease, rim, QtCore.Qt.LeftButton, QtCore.Qt.NoButton)

    assert len(commits) == 1
    lightness, chroma, hue = (float(v) for v in color_math.oklab_to_oklch(commits[0]))
    np.testing.assert_allclose(lightness, 0.5, atol=1e-9)
    # Snap takes the cursor's hue at release, not the press hue.
    np.testing.assert_allclose(hue % math.tau, 0.0, atol=1e-6)
    # And the chroma equals the per-hue cusp, which is strictly less than
    # color_math.SRGB_MAX_CHROMA (the rim).
    expected_chroma = float(color_math.max_chroma_for_lh(0.5, 0.0))
    np.testing.assert_allclose(chroma, expected_chroma, atol=1e-9)


def test_hover_outside_drag_does_not_snap(qtbot):
    # Snapping is drag-only — strict in-gamut picking (color_at) outside a
    # drag still returns None past the leaf, so keyboard nav and other
    # non-drag callers see strict in-gamut behaviour.
    widget = _widget(LightnessSliceModel(lightness=0.5))
    widget.resize(101, 101)
    qtbot.addWidget(widget)
    widget.show()

    assert widget.intent_at((99, 50)) is None


def test_click_without_drag_on_out_of_gamut_does_not_snap(qtbot):
    # A plain click (press + release with no intervening move) on a
    # transparent pixel inside the disk circle must NOT commit a snapped
    # cusp colour — otherwise tapping anywhere on the disk would land a
    # boundary pick. Snap is reserved for actual drags.
    widget = _widget(LightnessSliceModel(lightness=0.5))
    widget.resize(101, 101)
    qtbot.addWidget(widget)
    widget.show()

    commits = []
    previews = []
    widget.committed.connect(lambda c, _cs=commits: _cs.append(_paint_of(c)))
    widget.previewed.connect(lambda c, _ps=previews: _ps.append(_paint_of(c)))

    out_of_gamut = QtCore.QPoint(99, 50)  # inside the disk, past the leaf
    _send_mouse(widget, QtCore.QEvent.MouseButtonPress, out_of_gamut, QtCore.Qt.LeftButton, QtCore.Qt.LeftButton)
    _send_mouse(widget, QtCore.QEvent.MouseButtonRelease, out_of_gamut, QtCore.Qt.LeftButton, QtCore.Qt.NoButton)

    # No commit happened because there was no valid pick along the (zero
    # length) interaction.
    assert commits == []
    # The base widget signalled cancellation with a previewed(None) on
    # release, mirroring the strict pre-snap behaviour.
    assert previews and previews[-1] is None


def test_drag_resumes_snapping_after_cursor_re_enters(qtbot):
    # Qt keeps delivering mouse events to the pressed widget via the
    # implicit mouse grab even after the cursor leaves. A leave should
    # NOT close the snap window for the rest of the same drag —
    # subsequent moves with the left button still held need to keep
    # snapping at the cursor's hue.
    widget = _widget(LightnessSliceModel(lightness=0.5))
    widget.resize(101, 101)
    qtbot.addWidget(widget)
    widget.show()

    start = QtCore.QPoint(55, 50)
    rim = QtCore.QPoint(99, 50)
    commits = []
    widget.committed.connect(lambda c, _cs=commits: _cs.append(_paint_of(c)))

    _send_mouse(widget, QtCore.QEvent.MouseButtonPress, start, QtCore.Qt.LeftButton, QtCore.Qt.LeftButton)
    _send_mouse(widget, QtCore.QEvent.MouseMove, rim, QtCore.Qt.NoButton, QtCore.Qt.LeftButton)
    # Simulate the cursor leaving the widget mid-drag.
    QtCore.QCoreApplication.sendEvent(widget, QtCore.QEvent(QtCore.QEvent.Leave))
    _send_mouse(widget, QtCore.QEvent.MouseMove, rim, QtCore.Qt.NoButton, QtCore.Qt.LeftButton)
    _send_mouse(widget, QtCore.QEvent.MouseButtonRelease, rim, QtCore.Qt.LeftButton, QtCore.Qt.NoButton)

    assert len(commits) == 1
    lightness, chroma, hue = (float(v) for v in color_math.oklab_to_oklch(commits[0]))
    np.testing.assert_allclose(lightness, 0.5, atol=1e-9)
    np.testing.assert_allclose(hue % math.tau, 0.0, atol=1e-6)
    np.testing.assert_allclose(chroma, color_math.max_chroma_for_lh(0.5, 0.0), atol=1e-9)


def _send_mouse(widget, event_type, pos, button, buttons):
    event = QtGui.QMouseEvent(event_type, pos, button, buttons, QtCore.Qt.NoModifier)
    QtCore.QCoreApplication.sendEvent(widget, event)


def _paint_of(c):
    if c is None:
        return None
    return c.paint_oklab if isinstance(c, ColourIntent) else c
