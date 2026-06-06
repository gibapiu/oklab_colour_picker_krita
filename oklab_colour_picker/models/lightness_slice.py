"""Lightness-fixed hue/chroma selector model."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import numpy.typing as npt

from oklab_colour_picker.domain import color_math
from oklab_colour_picker.models.base import (
    OKLCh,
    Position,
    SelectorModel,
    SelectorSelection,
)
from oklab_colour_picker.models.geometry import (
    circle_geometry,
    circle_geometry_arrays,
    circle_geometry_projected,
    empty_color_grid,
    position_from_circle,
)


CHROMA_EPSILON = 1e-9
LIGHTNESS_EPSILON = 1e-9


@dataclass(frozen=True)
class LightnessSliceModel(SelectorModel):
    """Circular hue/chroma selector at a fixed OKLab lightness.

    Radius maps linearly to absolute OKLCh chroma in
    ``[0, color_math.SRGB_MAX_CHROMA]``, matching the Lightness tab's x-axis
    extent. Pixels whose chroma exceeds the per-hue sRGB cusp at this
    lightness fall outside the gamut leaf and render transparent, so the
    irregular gamut outline is visible directly on the disk.
    """

    lightness: float

    def __post_init__(self) -> None:
        _validate_lightness(self.lightness)

    def color_at_position(self, position: Sequence[float], size: Sequence[float]) -> np.ndarray | None:
        lch = self._lch_at_position(position, size)
        if lch is None:
            return None
        return color_math.oklch_to_oklab(list(lch))

    def selection_at_position(
        self, position: Sequence[float], size: Sequence[float]
    ) -> SelectorSelection | None:
        lch = self._lch_at_position(position, size)
        if lch is None:
            return None
        return SelectorSelection(lch, (float(position[0]), float(position[1])))

    def _lch_at_position(self, position: Sequence[float], size: Sequence[float]) -> OKLCh | None:
        geometry = circle_geometry(position, size)
        if geometry is None:
            return None
        normalized_radius, hue, _, _ = geometry
        chroma = normalized_radius * color_math.SRGB_MAX_CHROMA
        max_chroma = color_math.max_chroma_for_lh(self.lightness, hue)
        if chroma > max_chroma + CHROMA_EPSILON:
            return None
        return (self.lightness, float(chroma), float(hue))

    def colors_at_positions(
        self,
        x: npt.ArrayLike,
        y: npt.ArrayLike,
        size: Sequence[float],
    ) -> tuple[np.ndarray, np.ndarray]:
        geometry = circle_geometry_arrays(x, y, size)
        if geometry is None:
            return empty_color_grid(x), np.zeros_like(np.asarray(x), dtype=bool)

        normalized_radius, hue, circle_valid = geometry
        chroma = normalized_radius * color_math.SRGB_MAX_CHROMA
        oklch = np.stack(
            (
                np.full_like(normalized_radius, self.lightness, dtype=float),
                chroma,
                hue,
            ),
            axis=-1,
        )
        oklab = color_math.oklch_to_oklab(oklch)
        valid = circle_valid & color_math.in_srgb_gamut(
            color_math.oklab_to_srgb(oklab), epsilon=1e-6
        )
        return oklab, valid

    def position_for_intent(self, lch: OKLCh, size: Sequence[float]) -> Position | None:
        geometric = self.geometric_position_for_intent(lch, size)
        if geometric is None:
            return None
        chroma, hue = float(lch[1]), float(lch[2])
        if chroma > color_math.max_chroma_for_lh(self.lightness, hue) + CHROMA_EPSILON:
            return None
        return geometric

    def snapped_selector_selection_at_position(
        self, position: Sequence[float], size: Sequence[float]
    ) -> SelectorSelection | None:
        snapped = self._snapped_lch_at_position(position, size)
        if snapped is None:
            return None
        rendered = self.position_for_intent(snapped, size)
        if rendered is None:
            rendered = (float(position[0]), float(position[1]))
        return SelectorSelection(snapped, rendered)

    def _snapped_lch_at_position(self, position: Sequence[float], size: Sequence[float]) -> OKLCh | None:
        geometry = circle_geometry_projected(position, size)
        if geometry is None:
            return None
        normalized_radius, hue = geometry
        desired_chroma = normalized_radius * color_math.SRGB_MAX_CHROMA
        max_chroma = float(color_math.max_chroma_for_lh(self.lightness, hue))
        chroma = max(0.0, min(desired_chroma, max_chroma))
        return (self.lightness, float(chroma), float(hue))

    def geometric_position_for_intent(self, lch: OKLCh, size: Sequence[float]) -> Position | None:
        lightness, chroma, hue = float(lch[0]), float(lch[1]), float(lch[2])
        if abs(lightness - self.lightness) > LIGHTNESS_EPSILON:
            return None
        if chroma > color_math.SRGB_MAX_CHROMA + CHROMA_EPSILON:
            return None
        normalized_radius = float(np.clip(chroma / color_math.SRGB_MAX_CHROMA, 0.0, 1.0))
        return position_from_circle(normalized_radius, hue, size)


def _validate_lightness(lightness: float) -> None:
    if not math.isfinite(lightness) or not 0.0 <= lightness <= 1.0:
        raise ValueError("lightness must be finite and in [0, 1]")
