import math

import pytest

from oklab_colour_picker.app.selector_model_cache import (
    SelectorMode,
    SelectorModelCache,
)
from oklab_colour_picker.domain.colour_state import ColourIntent
from oklab_colour_picker.models import (
    HueLightnessSliceModel,
    LightnessChromaSliceModel,
    LightnessSliceModel,
)


@pytest.mark.parametrize(
    ("mode", "first_lch", "second_lch", "expected"),
    [
        (
            SelectorMode.LIGHTNESS_SLICE,
            (0.42, 0.03, 0.20),
            (0.42, 0.11, 2.40),
            LightnessSliceModel(lightness=0.42),
        ),
        (
            SelectorMode.HUE_LIGHTNESS_SLICE,
            (0.35, 0.07, 0.20),
            (0.78, 0.07, 2.40),
            HueLightnessSliceModel(chroma=0.07),
        ),
        (
            SelectorMode.LIGHTNESS_CHROMA_SLICE,
            (0.35, 0.07, 1.25),
            (0.78, 0.13, 1.25),
            LightnessChromaSliceModel(hue=1.25),
        ),
    ],
)
def test_reuses_model_when_fixed_coordinate_is_unchanged(
    mode,
    first_lch,
    second_lch,
    expected,
):
    cache = SelectorModelCache()

    first = cache.model_for(mode, ColourIntent.from_lch(*first_lch))
    second = cache.model_for(mode, ColourIntent.from_lch(*second_lch))

    assert first == expected
    assert second is first


@pytest.mark.parametrize(
    ("mode", "first_lch", "second_lch"),
    [
        (
            SelectorMode.LIGHTNESS_SLICE,
            (0.42, 0.07, 1.25),
            (0.58, 0.07, 1.25),
        ),
        (
            SelectorMode.HUE_LIGHTNESS_SLICE,
            (0.42, 0.05, 1.25),
            (0.42, 0.10, 1.25),
        ),
        (
            SelectorMode.LIGHTNESS_CHROMA_SLICE,
            (0.42, 0.07, 0.25),
            (0.42, 0.07, 0.75),
        ),
    ],
)
def test_replaces_model_when_fixed_coordinate_changes(
    mode,
    first_lch,
    second_lch,
):
    cache = SelectorModelCache()

    first = cache.model_for(mode, ColourIntent.from_lch(*first_lch))
    second = cache.model_for(mode, ColourIntent.from_lch(*second_lch))

    assert second is not first


def test_treats_hue_seam_as_same_slice():
    cache = SelectorModelCache()

    first = cache.model_for(
        SelectorMode.LIGHTNESS_CHROMA_SLICE,
        ColourIntent.from_lch(0.50, 0.06, 1e-12),
    )
    second = cache.model_for(
        SelectorMode.LIGHTNESS_CHROMA_SLICE,
        ColourIntent.from_lch(0.50, 0.06, math.tau - 1e-12),
    )

    assert second is first


@pytest.mark.parametrize(
    "mode",
    [
        SelectorMode.HUE_LIGHTNESS_SLICE,
        SelectorMode.LIGHTNESS_CHROMA_SLICE,
    ],
)
def test_achromatic_hue_change_replaces_hue_dependent_model(mode):
    cache = SelectorModelCache()

    first = cache.model_for(mode, ColourIntent.from_lch(0.40, 0.0, 1.25))
    second = cache.model_for(mode, ColourIntent.from_lch(0.70, 0.0, 2.50))

    assert second is not first


def test_reused_achromatic_model_preserves_explicit_hue_intent():
    cache = SelectorModelCache()
    hue = 1.25

    first = cache.model_for(
        SelectorMode.LIGHTNESS_CHROMA_SLICE,
        ColourIntent.from_lch(0.40, 0.0, hue),
    )
    second = cache.model_for(
        SelectorMode.LIGHTNESS_CHROMA_SLICE,
        ColourIntent.from_lch(0.70, 0.0, hue),
    )

    assert second is first
    assert first.hue == pytest.approx(hue)


@pytest.mark.parametrize("mode", list(SelectorMode))
def test_fallback_strategy_projects_onto_the_same_cached_slice(mode):
    cache = SelectorModelCache()
    intent = ColourIntent.from_lch(0.5, 0.08, 1.0)

    strategy = cache.fallback_strategy_for(mode, intent)

    # The strategy's plane is the very model the selector draws on, so the
    # dashed ring and the swatch can never project onto different slices.
    assert strategy.projection is cache.model_for(mode, intent)


def test_fallback_strategy_follows_the_slice_when_its_fixed_coordinate_changes():
    cache = SelectorModelCache()
    mode = SelectorMode.LIGHTNESS_CHROMA_SLICE

    first = cache.fallback_strategy_for(mode, ColourIntent.from_lch(0.5, 0.08, 1.0))
    second = cache.fallback_strategy_for(mode, ColourIntent.from_lch(0.5, 0.08, 2.5))

    assert first.projection is not second.projection
    assert second.projection is cache.model_for(mode, ColourIntent.from_lch(0.5, 0.08, 2.5))
