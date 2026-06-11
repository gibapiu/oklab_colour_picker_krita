"""Hue/Lightness disk widget with lightness-reference rings."""

from __future__ import annotations

from oklab_colour_picker.infrastructure.qt_facade import QtCore, QtGui

from oklab_colour_picker.models import HueLightnessSliceModel
from oklab_colour_picker.models.geometry import disk_geometry
from oklab_colour_picker.ui.selectors.selector import SelectorWidget


class HueLightnessSliceDiskWidget(SelectorWidget):
    """Hue/Lightness disk that overlays lightness guide rings."""

    _LIGHTNESS_RINGS: tuple[float, ...] = (0.25, 0.50, 0.75)

    def __init__(
        self,
        model: HueLightnessSliceModel,
        parent=None,
    ) -> None:
        super().__init__(model, parent)

    def _paint_indicator(self, painter: QtGui.QPainter) -> None:
        self._paint_lightness_rings(painter)
        super()._paint_indicator(painter)

    def _paint_lightness_rings(self, painter: QtGui.QPainter) -> None:
        geometry = self._disk_geometry()
        if geometry is None:
            return
        cx, cy, radius = geometry

        painter.save()
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        pen = QtGui.QPen(QtGui.QColor(255, 255, 255, 90), 1.0)
        pen.setCosmetic(True)
        painter.setPen(pen)
        for lightness in self._LIGHTNESS_RINGS:
            ring_radius = radius * (1.0 - lightness)
            if ring_radius <= 0.5 or ring_radius > radius:
                continue
            painter.drawEllipse(QtCore.QPointF(cx, cy), ring_radius, ring_radius)

        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(QtGui.QColor(255, 255, 255, 160))
        painter.drawEllipse(QtCore.QPointF(cx, cy), 1.5, 1.5)
        painter.restore()

    def _disk_geometry(self) -> tuple[float, float, float] | None:
        return disk_geometry((self.width(), self.height()))
