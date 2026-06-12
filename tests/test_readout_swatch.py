import pytest

pytest.importorskip("pytestqt")
pytestmark = pytest.mark.qt

from oklab_colour_picker.infrastructure.qt_facade import QtGui

from oklab_colour_picker.domain import color_math
from oklab_colour_picker.ui.readout.swatch import UnifiedSwatch, hex_to_oklab


@pytest.mark.parametrize(
    "hex_value",
    ["#000000", "#ffffff", "#4a8fb2", "#7f3322"],
)
def test_hex_round_trip(hex_value):
    oklab = hex_to_oklab(hex_value)

    assert oklab is not None
    assert _oklab_to_hex(oklab) == hex_value


def test_hex_accepts_uppercase_and_missing_hash():
    assert _oklab_to_hex(hex_to_oklab("4A8FB2")) == "#4a8fb2"
    assert _oklab_to_hex(hex_to_oklab("#FF00AA")) == "#ff00aa"


@pytest.mark.parametrize("bad", ["", "not-hex", "#12345", "#1234567", "#zzzzzz"])
def test_hex_rejects_malformed(bad):
    assert hex_to_oklab(bad) is None


def test_hex_editor_emits_changed_value(qtbot):
    swatch = UnifiedSwatch()
    qtbot.addWidget(swatch)
    committed = []
    swatch.hex_committed.connect(committed.append)

    swatch._enter_edit_mode()
    swatch._hex_edit.setText("#4a8fb2")
    swatch._hex_edit.editingFinished.emit()

    assert committed == ["#4a8fb2"]


def _oklab_to_hex(oklab) -> str:
    srgb8 = color_math.quantize_srgb8(color_math.oklab_to_srgb(oklab))
    return QtGui.QColor(*(int(value) for value in srgb8)).name(QtGui.QColor.NameFormat.HexRgb)
