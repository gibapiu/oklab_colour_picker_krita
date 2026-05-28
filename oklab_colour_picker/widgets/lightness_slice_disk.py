"""Hue/Chroma disk widget with chroma-reference rings and gamut contour.

Subclasses :class:`SelectorWidget` and adds two overlays drawn between the
disk image and the indicator:

- Concentric chroma rings at fixed absolute OKLCh chroma values, plus a small
  centre marker for the C=0 neutral axis. These give the eye a scale across
  L values that's invariant under hue rotation.
- A thin contour stroke along the per-(L, hue) sRGB gamut leaf so the cusp
  stays legible on dark backgrounds and at small dock widths.

The overlays are presentation-only — they don't affect picking. The contour
path is cached per (lightness, width, height) since rebuilding it requires
the Halley-iterated gamut math at each sampled hue.

Drag picks also snap to the gamut leaf or disk rim: when the cursor leaves the
leaf during a drag, ``LightnessSliceModel.snapped_color_at_position`` clamps to
the nearest in-gamut chroma at the cursor's hue so the preview stays
continuous along the boundary.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from PyQt5 import QtCore, QtGui

from oklab_colour_picker import color_math
from oklab_colour_picker.selector_models import (
    LightnessSliceModel,
    disk_geometry,
)
from oklab_colour_picker.widgets.selector import SelectorWidget


class LightnessSliceDiskWidget(SelectorWidget):
    """Hue/Chroma disk that overlays chroma rings and a gamut contour."""

    _CHROMA_RINGS: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.25)
    _GAMUT_HUE_SAMPLES = 180
    _CONTOUR_LIGHTNESS_KEY_PRECISION = 4

    def __init__(self, model: LightnessSliceModel, parent=None) -> None:
        super().__init__(model, parent)
        self._gamut_path_cache_key: tuple[float, int, int] | None = None
        self._gamut_path_cache: QtGui.QPainterPath | None = None
        self._gamut_contour_cache_key: float | None = None
        self._gamut_contour_cache: tuple[np.ndarray, np.ndarray] | None = None

    def set_model(self, model) -> None:  # type: ignore[override]
        super().set_model(model)
        self._invalidate_gamut_path()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._invalidate_gamut_path()

    def _paint_indicator(self, painter: QtGui.QPainter) -> None:
        # Drawing overlays here (instead of overriding paintEvent) keeps the
        # parent's painter lifecycle intact; rings/contour land on top of the
        # disk image and under the selection indicator.
        self._paint_chroma_rings(painter)
        self._paint_gamut_contour(painter)
        super()._paint_indicator(painter)

    def _paint_chroma_rings(self, painter: QtGui.QPainter) -> None:
        geometry = self._disk_geometry()
        if geometry is None:
            return
        cx, cy, radius = geometry

        painter.save()
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        painter.setBrush(QtCore.Qt.NoBrush)
        pen = QtGui.QPen(QtGui.QColor(255, 255, 255, 90), 1.0)
        pen.setCosmetic(True)
        painter.setPen(pen)
        for chroma in self._CHROMA_RINGS:
            ring_radius = radius * (chroma / color_math.SRGB_MAX_CHROMA)
            if ring_radius <= 0.5 or ring_radius > radius:
                continue
            painter.drawEllipse(QtCore.QPointF(cx, cy), ring_radius, ring_radius)

        # Tiny centre dot marks the C=0 neutral axis. Keep it small enough
        # that it doesn't obscure the selection indicator at the centre.
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QColor(255, 255, 255, 160))
        painter.drawEllipse(QtCore.QPointF(cx, cy), 1.5, 1.5)
        painter.restore()

    def _paint_gamut_contour(self, painter: QtGui.QPainter) -> None:
        model = self._model
        if not isinstance(model, LightnessSliceModel):
            return
        path = self._gamut_path(model.lightness)
        if path is None:
            return

        painter.save()
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        painter.setBrush(QtCore.Qt.NoBrush)
        # Dark halo first, light stroke second — keeps the contour readable
        # both on washed-out high-L slices and dark low-L slices.
        halo = QtGui.QPen(QtGui.QColor(0, 0, 0, 180), 2.0)
        halo.setCosmetic(True)
        painter.setPen(halo)
        painter.drawPath(path)
        stroke = QtGui.QPen(QtGui.QColor(255, 255, 255, 220), 1.0)
        stroke.setCosmetic(True)
        painter.setPen(stroke)
        painter.drawPath(path)
        painter.restore()

    def _gamut_path(self, lightness: float) -> QtGui.QPainterPath | None:
        geometry = self._disk_geometry()
        if geometry is None:
            return None
        cx, cy, radius = geometry
        key = (float(lightness), self.width(), self.height())
        if self._gamut_path_cache_key == key and self._gamut_path_cache is not None:
            return self._gamut_path_cache

        hues, normalized_radii = self._gamut_contour(lightness)
        radii = radius * normalized_radii
        xs = cx + radii * np.cos(hues)
        ys = cy - radii * np.sin(hues)

        path = QtGui.QPainterPath()
        path.moveTo(float(xs[0]), float(ys[0]))
        for i in range(1, len(hues)):
            path.lineTo(float(xs[i]), float(ys[i]))
        path.closeSubpath()

        self._gamut_path_cache_key = key
        self._gamut_path_cache = path
        return path

    def _gamut_contour(self, lightness: float) -> tuple[np.ndarray, np.ndarray]:
        key = round(float(lightness), self._CONTOUR_LIGHTNESS_KEY_PRECISION)
        if self._gamut_contour_cache_key == key and self._gamut_contour_cache is not None:
            return self._gamut_contour_cache

        hues = np.linspace(0.0, math.tau, self._GAMUT_HUE_SAMPLES, endpoint=False)
        max_chroma = np.asarray(
            color_math.max_chroma_for_lh(np.full_like(hues, key), hues),
            dtype=float,
        )
        # Cap at the disk's chroma extent so the contour traces the rim
        # rather than running off the widget where the gamut leaf bulges
        # past color_math.SRGB_MAX_CHROMA.
        capped = np.minimum(max_chroma, color_math.SRGB_MAX_CHROMA)
        normalized_radii = capped / color_math.SRGB_MAX_CHROMA
        self._gamut_contour_cache_key = key
        self._gamut_contour_cache = (hues, normalized_radii)
        return self._gamut_contour_cache

    def _disk_geometry(self) -> tuple[float, float, float] | None:
        return disk_geometry((self.width(), self.height()))

    def _invalidate_gamut_path(self) -> None:
        self._gamut_path_cache_key = None
        self._gamut_path_cache = None
