import pytest

pytest.importorskip("pytestqt")
pytest.importorskip("PyQt5")

from oklab_colour_picker.infrastructure.krita_adapter import (
    FOREGROUND_POLL_INTERVAL_MS,
    QtForegroundTimer,
)


def test_foreground_timer_runs_at_fixed_interval(qtbot):
    timer = QtForegroundTimer()

    timer.start(lambda: None)
    try:
        assert timer._timer.isActive()
        assert timer._timer.interval() == FOREGROUND_POLL_INTERVAL_MS
    finally:
        timer.stop()

    assert not timer._timer.isActive()
