import os
import statistics
import time

import pytest

from oklab_colour_picker.render import renderers
from oklab_colour_picker.render.renderers import render_rgba
from oklab_colour_picker.models import (
    LightnessChromaSliceModel,
    HueLightnessSliceModel,
    LightnessSliceModel,
)


PERFORMANCE_BUDGET_SECONDS = 0.005
COLD_RENDER_BUDGET_SECONDS = 0.020
CI_PERFORMANCE_BUDGET_MULTIPLIER = 2.0
SAMPLE_COUNT = 21
COLD_SAMPLE_COUNT = 7


def _budget(base_seconds: float) -> float:
    if os.environ.get("CI"):
        return base_seconds * CI_PERFORMANCE_BUDGET_MULTIPLIER
    return base_seconds


@pytest.mark.perf
def test_256_renderers_meet_median_budget():
    cases = [
        LightnessSliceModel(lightness=0.55),
        HueLightnessSliceModel(chroma=0.05),
        LightnessChromaSliceModel(hue=1.25),
    ]

    for model in cases:
        render_rgba(model, (256, 256))
        timings = []
        for _ in range(SAMPLE_COUNT):
            start = time.perf_counter()
            render_rgba(model, (256, 256))
            timings.append(time.perf_counter() - start)

        budget = _budget(PERFORMANCE_BUDGET_SECONDS)
        median = statistics.median(timings)
        assert median <= budget, (
            f"{type(model).__name__} cached 256px render took {median:.4f}s; "
            f"budget is {budget:.4f}s"
        )


@pytest.mark.perf
def test_256_cold_renderers_meet_startup_budget_without_cache_warmup():
    cases = [
        LightnessSliceModel(lightness=0.55),
        HueLightnessSliceModel(chroma=0.05),
        LightnessChromaSliceModel(hue=1.25),
    ]

    for model in cases:
        timings = []
        for _ in range(COLD_SAMPLE_COUNT):
            renderers._render_rgba_cached.cache_clear()
            renderers._pixel_grid.cache_clear()
            start = time.perf_counter()
            render_rgba(model, (256, 256))
            timings.append(time.perf_counter() - start)

        median = statistics.median(timings)
        budget = _budget(COLD_RENDER_BUDGET_SECONDS)
        assert median <= budget, (
            f"{type(model).__name__} cold 256px render took {median:.4f}s; "
            f"budget is {budget:.4f}s"
        )
