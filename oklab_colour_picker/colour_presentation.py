"""Pure selected-colour presentation state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from oklab_colour_picker.colour_state import ColourIntent
from oklab_colour_picker.gamut_fallback import (
    ClippedSrgbFallbackStrategy,
    FallbackResult,
    FallbackStrategy,
)


@dataclass(frozen=True)
class PresentedColour:
    intent: ColourIntent
    fallback: FallbackResult

    @property
    def paint_oklab(self) -> np.ndarray:
        return self.intent.paint_oklab

    @property
    def selector_lch(self) -> tuple[float, float, float]:
        return self.intent.selector_lch

    @property
    def srgb8(self) -> tuple[int, int, int]:
        return self.fallback.srgb8

    @property
    def in_gamut(self) -> bool:
        return self.fallback.in_gamut


@dataclass(frozen=True)
class ColourPresenter:
    fallback_strategy: FallbackStrategy

    def present(
        self,
        colour: ColourIntent | Sequence[float],
        *,
        achromatic_hue: float = 0.0,
    ) -> PresentedColour:
        intent = ColourIntent.from_value(colour, achromatic_hue=achromatic_hue)
        return PresentedColour(intent, self.fallback_strategy.resolve(intent))


def default_colour_presenter() -> ColourPresenter:
    return ColourPresenter(ClippedSrgbFallbackStrategy())
