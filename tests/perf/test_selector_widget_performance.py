import os
import statistics
import time

import pytest

pytest.importorskip("pytestqt")
pytestmark = [pytest.mark.qt, pytest.mark.perf]

from oklab_colour_picker.models import HueLightnessSliceModel, LightnessChromaSliceModel, LightnessSliceModel
from oklab_colour_picker.ui.selectors import HueLightnessSliceDiskWidget, LightnessSliceDiskWidget, SelectorWidget
from tests.qt_helpers import render_pixels


@pytest.mark.parametrize("widget_type, model_type, coordinate", [
    (LightnessSliceDiskWidget, LightnessSliceModel, 0.42),
    (HueLightnessSliceDiskWidget, HueLightnessSliceModel, 0.05),
    (SelectorWidget, LightnessChromaSliceModel, 1.25),
])
def test_selector_frame_budgets_include_image_outline_and_clip(qtbot, widget_type, model_type, coordinate):
    widget = widget_type(model_type(coordinate))
    widget.resize(256, 256)
    qtbot.addWidget(widget)
    cold_frames = []
    for sample in range(7):
        widget.set_model(model_type(coordinate + sample * 0.01))
        start = time.perf_counter()
        render_pixels(widget)
        cold_frames.append(time.perf_counter() - start)
    cached_frames = []
    for _ in range(21):
        start = time.perf_counter()
        render_pixels(widget)
        cached_frames.append(time.perf_counter() - start)
    multiplier = 2 if os.environ.get("CI") else 1
    assert statistics.median(cold_frames) <= 0.025 * multiplier
    assert statistics.median(cached_frames) <= 0.005 * multiplier
