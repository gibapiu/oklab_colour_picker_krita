"""Selector slice model construction and reuse policy."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Protocol

from oklab_colour_picker.domain.colour_state import ColourIntent
from oklab_colour_picker.models import (
    HueLightnessSliceModel,
    LightnessChromaSliceModel,
    LightnessSliceModel,
    SelectorModel,
)


class SelectorMode(str, Enum):
    LIGHTNESS_SLICE = "lightness_slice"
    HUE_LIGHTNESS_SLICE = "hue_lightness_slice"
    LIGHTNESS_CHROMA_SLICE = "lightness_chroma_slice"


# OKLab -> OKLCh recovery can jitter by a few ulps for fixed hue/chroma slices.
_COORDINATE_ROUNDTRIP_EPSILON = 1.0 / (255.0**3)


class _SliceCoordinate(Protocol):
    def equivalent_to(self, other: object) -> bool:
        ...


@dataclass(frozen=True)
class _LinearCoordinate:
    value: float

    def equivalent_to(self, other: object) -> bool:
        return isinstance(other, _LinearCoordinate) and math.isclose(
            self.value,
            other.value,
            rel_tol=0.0,
            abs_tol=_COORDINATE_ROUNDTRIP_EPSILON,
        )


@dataclass(frozen=True)
class _ChromaCoordinate:
    value: float
    hue_when_achromatic: float

    def equivalent_to(self, other: object) -> bool:
        if not isinstance(other, _ChromaCoordinate):
            return False
        if self.value == other.value == 0.0:
            return (
                _circular_distance(
                    self.hue_when_achromatic,
                    other.hue_when_achromatic,
                )
                <= _COORDINATE_ROUNDTRIP_EPSILON
            )
        return math.isclose(
            self.value,
            other.value,
            rel_tol=0.0,
            abs_tol=_COORDINATE_ROUNDTRIP_EPSILON,
        )


@dataclass(frozen=True)
class _HueCoordinate:
    radians: float

    def equivalent_to(self, other: object) -> bool:
        return (
            isinstance(other, _HueCoordinate)
            and _circular_distance(self.radians, other.radians)
            <= _COORDINATE_ROUNDTRIP_EPSILON
        )


@dataclass(frozen=True)
class _ModePolicy:
    model_factory: Callable[[ColourIntent], SelectorModel]
    coordinate_factory: Callable[[ColourIntent], _SliceCoordinate]


@dataclass(frozen=True)
class _CacheEntry:
    coordinate: _SliceCoordinate
    model: SelectorModel


class SelectorModelCache:
    """Return one model per selector mode until its fixed slice changes."""

    def __init__(self) -> None:
        self._entries: dict[SelectorMode, _CacheEntry] = {}

    def model_for(self, mode: SelectorMode, intent: ColourIntent) -> SelectorModel:
        policy = _MODE_POLICIES[mode]
        coordinate = policy.coordinate_factory(intent)
        cached = self._entries.get(mode)
        if cached is not None and cached.coordinate.equivalent_to(coordinate):
            return cached.model

        model = policy.model_factory(intent)
        self._entries[mode] = _CacheEntry(coordinate, model)
        return model


def _lightness_model(intent: ColourIntent) -> SelectorModel:
    lightness, _chroma, _hue = intent.selector_lch
    return LightnessSliceModel(lightness=lightness)


def _hue_lightness_model(intent: ColourIntent) -> SelectorModel:
    if intent.is_achromatic:
        return HueLightnessSliceModel(
            chroma=intent.chroma,
            achromatic_indicator_hue=intent.hue,
        )
    return HueLightnessSliceModel(chroma=intent.chroma)


def _lightness_chroma_model(intent: ColourIntent) -> SelectorModel:
    return LightnessChromaSliceModel(hue=intent.hue)


def _lightness_coordinate(intent: ColourIntent) -> _SliceCoordinate:
    lightness, _chroma, _hue = intent.selector_lch
    return _LinearCoordinate(lightness)


def _chroma_coordinate(intent: ColourIntent) -> _SliceCoordinate:
    hue = intent.hue if intent.is_achromatic else 0.0
    return _ChromaCoordinate(intent.chroma, hue)


def _hue_coordinate(intent: ColourIntent) -> _SliceCoordinate:
    return _HueCoordinate(intent.hue % math.tau)


_MODE_POLICIES = {
    SelectorMode.LIGHTNESS_SLICE: _ModePolicy(
        _lightness_model,
        _lightness_coordinate,
    ),
    SelectorMode.HUE_LIGHTNESS_SLICE: _ModePolicy(
        _hue_lightness_model,
        _chroma_coordinate,
    ),
    SelectorMode.LIGHTNESS_CHROMA_SLICE: _ModePolicy(
        _lightness_chroma_model,
        _hue_coordinate,
    ),
}


def _circular_distance(left: float, right: float) -> float:
    distance = abs((left - right) % math.tau)
    return min(distance, math.tau - distance)
