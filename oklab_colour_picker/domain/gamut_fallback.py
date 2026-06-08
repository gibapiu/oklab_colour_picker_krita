"""Pure out-of-gamut fallback policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from oklab_colour_picker.domain import color_math
from oklab_colour_picker.domain.colour_state import ColourIntent


OKLCh = tuple[float, float, float]


@dataclass(frozen=True)
class FallbackResult:
    source: ColourIntent
    resolved: ColourIntent
    srgb8: tuple[int, int, int]
    in_gamut: bool


class FallbackStrategy(Protocol):
    def resolve(self, colour: ColourIntent | Sequence[float]) -> FallbackResult:
        ...


class SliceProjection(Protocol):
    """Narrow port: project an OKLCh onto a slice's in-gamut leaf, or ``None``."""

    def project_onto_slice(self, lch: OKLCh) -> OKLCh | None:
        ...


@dataclass(frozen=True)
class ClippedSrgbFallbackStrategy:
    gamut_epsilon: float = 1e-9

    def resolve(self, colour: ColourIntent | Sequence[float]) -> FallbackResult:
        source = ColourIntent.from_value(colour)
        srgb = color_math.oklab_to_srgb(source.paint_oklab)
        srgb8_array = color_math.quantize_srgb8(srgb)
        srgb8 = tuple(int(v) for v in srgb8_array)
        resolved_oklab = color_math.srgb_to_oklab(srgb8_array.astype(float) / 255.0)
        return FallbackResult(
            source=source,
            resolved=ColourIntent.from_oklab(resolved_oklab, achromatic_hue=source.hue),
            srgb8=srgb8,
            in_gamut=bool(color_math.in_srgb_gamut(srgb, epsilon=self.gamut_epsilon)),
        )


@dataclass(frozen=True)
class SliceProjectionFallbackStrategy:
    """Resolve fallback by projecting onto the active slice's plane.

    In-gamut colours pass through untouched. Out-of-gamut colours project onto the slice's in-gamut leaf,
    so the swatch, the Krita write, and the dashed ring all land on the same on-plane colour.
    """

    projection: SliceProjection
    gamut_epsilon: float = 1e-9

    def resolve(self, colour: ColourIntent | Sequence[float]) -> FallbackResult:
        source = ColourIntent.from_value(colour)
        srgb = color_math.oklab_to_srgb(source.paint_oklab)
        if bool(color_math.in_srgb_gamut(srgb, epsilon=self.gamut_epsilon)):
            return FallbackResult(
                source=source,
                resolved=source,
                srgb8=_srgb8(source.paint_oklab),
                in_gamut=True,
            )

        resolved_lch = self.projection.project_onto_slice(source.selector_lch)
        if resolved_lch is None:
            return ClippedSrgbFallbackStrategy(self.gamut_epsilon).resolve(source)

        resolved = ColourIntent.from_lch(*resolved_lch)
        return FallbackResult(
            source=source,
            resolved=resolved,
            srgb8=_srgb8(resolved.paint_oklab),
            in_gamut=False,
        )


def _srgb8(oklab: Sequence[float]) -> tuple[int, int, int]:
    return tuple(int(v) for v in color_math.oklab_to_srgb8(oklab))
