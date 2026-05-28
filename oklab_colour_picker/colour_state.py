"""Selected-colour domain state."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property
from typing import Sequence

import numpy as np

from oklab_colour_picker import color_math


@dataclass(frozen=True, init=False)
class ColourIntent:
    _paint_oklab: tuple[float, float, float]
    lightness: float
    chroma: float
    hue: float

    @classmethod
    def from_lch(cls, lightness: float, chroma: float, hue: float) -> "ColourIntent":
        normalized_lightness = _finite_float(lightness, "lightness")
        normalized_chroma = max(0.0, _finite_float(chroma, "chroma"))
        normalized_hue = _normalized_hue(_finite_float(hue, "hue"))
        if color_math.is_achromatic_chroma(normalized_chroma):
            normalized_chroma = 0.0
        paint = color_math.oklch_to_oklab(
            [normalized_lightness, normalized_chroma, normalized_hue]
        )
        return cls._create(
            _paint_tuple(paint),
            normalized_lightness,
            normalized_chroma,
            normalized_hue,
        )

    @classmethod
    def from_oklab(
        cls,
        oklab: Sequence[float],
        *,
        achromatic_hue: float = 0.0,
    ) -> "ColourIntent":
        paint = _as_oklab(oklab)
        lightness, chroma, hue = color_math.oklab_to_oklch(paint)
        normalized_chroma = max(0.0, float(chroma))
        if color_math.is_achromatic_chroma(normalized_chroma):
            normalized_chroma = 0.0
            hue = achromatic_hue
        return cls._create(
            _paint_tuple(paint),
            _finite_float(float(lightness), "lightness"),
            normalized_chroma,
            _normalized_hue(_finite_float(float(hue), "hue")),
        )

    @classmethod
    def from_value(
        cls,
        value: "ColourIntent | Sequence[float]",
        *,
        achromatic_hue: float = 0.0,
    ) -> "ColourIntent":
        """Adopt an existing intent unchanged; use ``achromatic_hue`` only for raw OKLab."""

        if isinstance(value, ColourIntent):
            return value
        return cls.from_oklab(value, achromatic_hue=achromatic_hue)

    @classmethod
    def _create(
        cls,
        paint_oklab: tuple[float, float, float],
        lightness: float,
        chroma: float,
        hue: float,
    ) -> "ColourIntent":
        intent = cls.__new__(cls)
        object.__setattr__(intent, "_paint_oklab", paint_oklab)
        object.__setattr__(intent, "lightness", lightness)
        object.__setattr__(intent, "chroma", chroma)
        object.__setattr__(intent, "hue", hue)
        return intent

    @property
    def paint_oklab(self) -> np.ndarray:
        return np.asarray(self._paint_oklab, dtype=float)

    @property
    def selector_lch(self) -> tuple[float, float, float]:
        return (
            float(np.clip(self.lightness, 0.0, 1.0)),
            self.chroma,
            self.hue,
        )

    @property
    def is_achromatic(self) -> bool:
        return color_math.is_achromatic_chroma(self.chroma)

    def with_lightness(self, lightness: float) -> "ColourIntent":
        return self.from_lch(lightness, self.chroma, self.hue)

    def with_chroma(self, chroma: float) -> "ColourIntent":
        return self.from_lch(self.lightness, chroma, self.hue)

    def with_hue(self, hue: float) -> "ColourIntent":
        return self.from_lch(self.lightness, self.chroma, hue)

    def with_krita_paint_oklab(self, oklab: Sequence[float]) -> "ColourIntent":
        paint = _as_oklab(oklab)
        if not np.array_equal(normalize_oklab_for_krita(paint), self.quantized_paint_oklab):
            raise ValueError("Krita paint readback must quantize-equal the intent paint")
        return ColourIntent._create(
            _paint_tuple(paint),
            self.lightness,
            self.chroma,
            self.hue,
        )

    @cached_property
    def quantized_paint_oklab(self) -> np.ndarray:
        colour = normalize_oklab_for_krita(self.paint_oklab)
        colour.setflags(write=False)
        return colour

    def quantized_paint_equal(self, other: "ColourIntent | Sequence[float]") -> bool:
        compared = ColourIntent.from_value(other, achromatic_hue=self.hue)
        return bool(
            np.array_equal(
                self.quantized_paint_oklab,
                compared.quantized_paint_oklab,
            )
        )


def normalize_oklab_for_krita(oklab: Sequence[float]) -> np.ndarray:
    """Normalize OKLab through Krita's 8-bit sRGB foreground precision."""

    srgb8 = color_math.oklab_to_srgb8(_as_oklab(oklab))
    return color_math.srgb_to_oklab(srgb8.astype(float) / 255.0)


def _as_oklab(oklab: Sequence[float]) -> np.ndarray:
    colour = np.asarray(oklab, dtype=float)
    if colour.shape != (3,):
        raise ValueError("OKLab colour must contain exactly three components")
    if not np.all(np.isfinite(colour)):
        raise ValueError("OKLab colour components must be finite")
    return colour.copy()


def _paint_tuple(oklab: Sequence[float]) -> tuple[float, float, float]:
    colour = _as_oklab(oklab)
    return (float(colour[0]), float(colour[1]), float(colour[2]))


def _finite_float(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _normalized_hue(hue: float) -> float:
    return float(hue % math.tau)
