"""Shared readout widget styling helpers."""

from __future__ import annotations

from PyQt5 import QtGui


DARK_INK = QtGui.QColor("#1e1e1e")
LIGHT_INK = QtGui.QColor("#f2f2f2")

HANDLE_WIDTH = 10
HANDLE_BORDER = 2

SWATCH_HEIGHT = 48
CORNER_BUTTON_SIZE = 20


def perceived_luminance(r: int, g: int, b: int) -> float:
    """Simple Rec.709 luma on 0-255 sRGB bytes."""

    return (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0


def ink_for(r: int, g: int, b: int) -> QtGui.QColor:
    return DARK_INK if perceived_luminance(r, g, b) > 0.55 else LIGHT_INK


def qcolor_from_srgb8(srgb8: tuple[int, int, int]) -> QtGui.QColor:
    return QtGui.QColor(int(srgb8[0]), int(srgb8[1]), int(srgb8[2]))
