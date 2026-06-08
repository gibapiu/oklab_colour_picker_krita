"""Pure selected-colour presentation state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from oklab_colour_picker.domain.colour_state import ColourIntent
from oklab_colour_picker.domain.gamut_fallback import (
    ClippedSrgbFallbackStrategy,
    FallbackResult,
    FallbackStrategy,
)


@dataclass(frozen=True)
class PresentedColour:
    """Derived display read model built by ``ColourPresenter`` and read by views.

    Construction is confined to the presenter by an import-discipline test,
    so fallback policy has a single owner; everywhere else receives and reads it.
    """

    intent: ColourIntent
    fallback: FallbackResult

    @property
    def paint_oklab(self) -> np.ndarray:
        return self.intent.paint_oklab

    @property
    def selector_lch(self) -> tuple[float, float, float]:
        return self.intent.selector_lch

    @property
    def resolved_lch(self) -> tuple[float, float, float]:
        return self.fallback.resolved.selector_lch

    @property
    def srgb8(self) -> tuple[int, int, int]:
        return self.fallback.srgb8

    @property
    def in_gamut(self) -> bool:
        return self.fallback.in_gamut


@dataclass(frozen=True)
class ColourPresenter:
    """Project a colour intent into the view model used by Qt widgets."""

    fallback_strategy: FallbackStrategy

    def with_fallback_strategy(self, strategy: FallbackStrategy) -> "ColourPresenter":
        """Return a presenter that resolves fallback through ``strategy``."""

        return ColourPresenter(strategy)

    def present(
        self,
        colour: ColourIntent | Sequence[float],
        *,
        achromatic_hue: float = 0.0,
    ) -> PresentedColour:
        intent = ColourIntent.from_value(colour, achromatic_hue=achromatic_hue)
        return PresentedColour(intent, self.fallback_strategy.resolve(intent))


def require_presented_colour(colour: PresentedColour | None) -> None:
    if colour is not None and not isinstance(colour, PresentedColour):
        raise TypeError("displayed colours must be PresentedColour")


def default_colour_presenter() -> ColourPresenter:
    return ColourPresenter(ClippedSrgbFallbackStrategy())
