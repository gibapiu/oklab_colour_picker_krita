import math
import os
import statistics
import time

import pytest

from oklab_colour_picker.models import (
    LightnessChromaSliceModel,
    HueLightnessSliceModel,
    LightnessSliceModel,
)


PERFORMANCE_BUDGET_SECONDS = 0.005
CI_PERFORMANCE_BUDGET_MULTIPLIER = 2.0
SAMPLE_COUNT = 21


def _budget(base_seconds: float) -> float:
    if os.environ.get("CI"):
        return base_seconds * CI_PERFORMANCE_BUDGET_MULTIPLIER
    return base_seconds


@pytest.mark.perf
def test_blue_gap_projection_meets_interaction_budget():
    lch = (0.426041573241465, 0.272, math.radians(264.207587271901))
    models = (
        LightnessSliceModel(lch[0]),
        HueLightnessSliceModel(lch[1]),
        LightnessChromaSliceModel(lch[2]),
    )
    for model in models:
        timings = []
        for _ in range(SAMPLE_COUNT):
            start = time.perf_counter()
            resolved = model.project_onto_slice(lch)
            timings.append(time.perf_counter() - start)
            assert resolved is not None
        assert statistics.median(timings) <= _budget(PERFORMANCE_BUDGET_SECONDS)
