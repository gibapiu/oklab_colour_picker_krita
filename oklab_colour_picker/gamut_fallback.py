"""Pure out-of-gamut fallback policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np

from oklab_colour_picker import color_math
from oklab_colour_picker.colour_state import ColourIntent


@dataclass(frozen=True)
class FallbackResult:
    source: ColourIntent
    fallback: ColourIntent
    srgb8: tuple[int, int, int]
    in_gamut: bool


class FallbackStrategy(Protocol):
    def resolve(self, colour: ColourIntent | Sequence[float]) -> FallbackResult:
        ...


@dataclass(frozen=True)
class ClippedSrgbFallbackStrategy:
    gamut_epsilon: float = 1e-9

    def resolve(self, colour: ColourIntent | Sequence[float]) -> FallbackResult:
        source = ColourIntent.from_value(colour)
        srgb = color_math.oklab_to_srgb(source.paint_oklab)
        srgb8_array = color_math.quantize_srgb8(srgb)
        srgb8 = tuple(int(v) for v in srgb8_array)
        fallback_oklab = color_math.srgb_to_oklab(srgb8_array.astype(float) / 255.0)
        return FallbackResult(
            source=source,
            fallback=ColourIntent.from_oklab(fallback_oklab, achromatic_hue=source.hue),
            srgb8=srgb8,
            in_gamut=bool(color_math.in_srgb_gamut(srgb, epsilon=self.gamut_epsilon)),
        )
