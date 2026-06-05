from __future__ import annotations

from collections.abc import Sequence

from oklab_colour_picker.colour_presentation import PresentedColour
from oklab_colour_picker.colour_state import ColourIntent
from oklab_colour_picker.gamut_fallback import FallbackResult


def presented_colour(
    colour: ColourIntent | Sequence[float],
    *,
    srgb8: tuple[int, int, int] = (12, 34, 56),
    in_gamut: bool = True,
    fallback: ColourIntent | Sequence[float] | None = None,
    achromatic_hue: float = 0.0,
) -> PresentedColour:
    intent = ColourIntent.from_value(colour, achromatic_hue=achromatic_hue)
    fallback_intent = (
        intent
        if fallback is None
        else ColourIntent.from_value(fallback, achromatic_hue=achromatic_hue)
    )
    return PresentedColour(
        intent=intent,
        fallback=FallbackResult(
            source=intent,
            resolved=fallback_intent,
            srgb8=srgb8,
            in_gamut=in_gamut,
        ),
    )
