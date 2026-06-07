"""Pure mapping from presented colour state to readout display state."""

from __future__ import annotations

from dataclasses import dataclass

from oklab_colour_picker.domain.colour_presentation import PresentedColour


SelectorLch = tuple[float, float, float]
Srgb8 = tuple[int, int, int]


@dataclass(frozen=True)
class ReadoutDisplay:
    selector_lch: SelectorLch
    srgb8: Srgb8
    out_of_gamut: bool
    revert_hex: str | None


class ReadoutPresenter:
    """Build the complete display state consumed by the readout widgets."""

    def present(
        self,
        colour: PresentedColour,
        *,
        previous: PresentedColour | None,
        selector_lch: SelectorLch | None = None,
    ) -> ReadoutDisplay:
        displayed_lch = colour.selector_lch if selector_lch is None else selector_lch
        lightness, chroma, hue = displayed_lch
        return ReadoutDisplay(
            selector_lch=(float(lightness), float(chroma), float(hue)),
            srgb8=colour.srgb8,
            out_of_gamut=not colour.in_gamut,
            revert_hex=None if previous is None else _hex_from_srgb8(previous.srgb8),
        )


def _hex_from_srgb8(srgb8: Srgb8) -> str:
    return "#{:02x}{:02x}{:02x}".format(*srgb8)
