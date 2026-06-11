import numpy as np
import pytest

pytest.importorskip("pytestqt")

from oklab_colour_picker.infrastructure.qt_facade import QtWidgets

from oklab_colour_picker.render import renderers
from oklab_colour_picker.ui.readout.axis import AxisTrackPresenter, ReadoutAxisRows


def test_axis_track_presenter_caches_by_fixed_coordinates_and_slider(qtbot, monkeypatch):
    parent = QtWidgets.QWidget()
    qtbot.addWidget(parent)
    rows = ReadoutAxisRows.create(parent)
    for row in rows.as_tuple():
        row.slider.resize(120, 24)
    calls = []

    def render(axis, fixed, _max_chroma, size, *, hue_chroma_floor=0.0):
        calls.append((axis, fixed, size, hue_chroma_floor))
        return np.zeros((size[1], size[0], 4), dtype=np.uint8)

    monkeypatch.setattr(renderers, "render_axis_track", render)
    presenter = AxisTrackPresenter()

    presenter.refresh(rows, 0.5, 0.1, 1.0)
    presenter.refresh(rows, 0.5, 0.1, 1.0)
    presenter.refresh(rows, 0.5, 0.2, 1.0)

    assert [call[0] for call in calls] == [
        renderers.AXIS_L,
        renderers.AXIS_C,
        renderers.AXIS_H,
        renderers.AXIS_L,
        renderers.AXIS_H,
    ]

    replacement_rows = ReadoutAxisRows.create(parent)
    for row in replacement_rows.as_tuple():
        row.slider.resize(120, 24)

    presenter.refresh(replacement_rows, 0.5, 0.2, 1.0)

    assert [call[0] for call in calls[-3:]] == [
        renderers.AXIS_L,
        renderers.AXIS_C,
        renderers.AXIS_H,
    ]
