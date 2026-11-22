"""Hue/Chroma disk with chroma rings and a gamut outline."""

from __future__ import annotations

from oklab_colour_picker.infrastructure.qt_facade import QtCore, QtGui

from oklab_colour_picker.domain import color_math
from oklab_colour_picker.models import LightnessSliceModel, SelectorModel
from oklab_colour_picker.models.geometry import disk_geometry
from oklab_colour_picker.render import renderers
from oklab_colour_picker.ui.selectors.selector import SelectorWidget


class LightnessSliceDiskWidget(SelectorWidget):
    """Hue/Chroma disk that overlays chroma rings and a gamut contour."""

    _CHROMA_RINGS: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.25)

    def __init__(
        self,
        model: LightnessSliceModel,
        parent=None,
    ) -> None:
        super().__init__(model, parent)
        self._gamut_path_cache_key: tuple[SelectorModel, int, int] | None = None
        self._gamut_path_cache: QtGui.QPainterPath | None = None

    def _paint_indicator(self, painter: QtGui.QPainter) -> None:
        self._paint_chroma_rings(painter)
        self._paint_gamut_contour(painter)
        super()._paint_indicator(painter)

    def _paint_chroma_rings(self, painter: QtGui.QPainter) -> None:
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
        for chroma in self._CHROMA_RINGS:
            ring_radius = radius * (chroma / color_math.SRGB_MAX_CHROMA)
            if ring_radius <= 0.5 or ring_radius > radius:
                continue
            painter.drawEllipse(QtCore.QPointF(cx, cy), ring_radius, ring_radius)

        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(QtGui.QColor(255, 255, 255, 160))
        painter.drawEllipse(QtCore.QPointF(cx, cy), 1.5, 1.5)
        painter.restore()

    def _paint_gamut_contour(self, painter: QtGui.QPainter) -> None:
        model = self._model
        if not isinstance(model, LightnessSliceModel):
            return
        path = self._gamut_path()
        if path is None:
            return

        painter.save()
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        halo = QtGui.QPen(QtGui.QColor(0, 0, 0, 180), 2.0)
        halo.setCosmetic(True)
        painter.setPen(halo)
        painter.drawPath(path)
        stroke = QtGui.QPen(QtGui.QColor(255, 255, 255, 220), 1.0)
        stroke.setCosmetic(True)
        painter.setPen(stroke)
        painter.drawPath(path)
        painter.restore()

    def _gamut_path(self) -> QtGui.QPainterPath | None:
        if self._disk_geometry() is None:
            return None
        key = (self._model, self.width(), self.height())
        if self._gamut_path_cache_key == key and self._gamut_path_cache is not None:
            return self._gamut_path_cache

        self._selector_image()
        if self._image_cache_buffer is None:
            raise AssertionError("rendered selector must have an RGBA buffer")
        segments = renderers.gamut_contour_segments(self._image_cache_buffer[..., 3] != 0)
        path = QtGui.QPainterPath()
        for start, end in segments:
            path.moveTo(float(start[0]), float(start[1]))
            path.lineTo(float(end[0]), float(end[1]))

        self._gamut_path_cache_key = key
        self._gamut_path_cache = path
        return path

    def _disk_geometry(self) -> tuple[float, float, float] | None:
        return disk_geometry((self.width(), self.height()))
