import os
import subprocess
import sys

import pytest


COLD_COLOR_MATH_IMPORT_BUDGET_SECONDS = 0.120
CI_PERFORMANCE_BUDGET_MULTIPLIER = 2.0


def _budget(base_seconds: float) -> float:
    if os.environ.get("CI"):
        return base_seconds * CI_PERFORMANCE_BUDGET_MULTIPLIER
    return base_seconds


@pytest.mark.perf
def test_color_math_import_meets_cold_start_budget():
    script = """
import time
start = time.perf_counter()
import oklab_colour_picker.domain.color_math
print(time.perf_counter() - start)
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    elapsed = float(completed.stdout.strip())
    budget = _budget(COLD_COLOR_MATH_IMPORT_BUDGET_SECONDS)

    assert elapsed <= budget, (
        f"cold color_math import took {elapsed:.4f}s; "
        f"budget is {budget:.4f}s"
    )
