import math

import numpy as np

from oklab_colour_picker.domain import color_math
from oklab_colour_picker.domain.colour_state import ColourIntent
from oklab_colour_picker.domain.gamut_fallback import (
    ClippedSrgbFallbackStrategy,
    SliceProjectionFallbackStrategy,
)
from oklab_colour_picker.models import LightnessChromaSliceModel


def test_clipped_srgb_fallback_uses_quantized_krita_colour():
    intent = ColourIntent.from_lch(0.6, color_math.SRGB_MAX_CHROMA, 0.0)

    result = ClippedSrgbFallbackStrategy().resolve(intent)

    expected_srgb8 = tuple(int(v) for v in color_math.oklab_to_srgb8(intent.paint_oklab))
    assert result.srgb8 == expected_srgb8
    np.testing.assert_allclose(
        result.resolved.paint_oklab,
        color_math.srgb_to_oklab(np.asarray(expected_srgb8, dtype=float) / 255.0),
        atol=1e-12,
    )
    assert not result.in_gamut


def test_in_gamut_colour_still_has_canonical_srgb8_sample():
    intent = ColourIntent.from_value(color_math.srgb_to_oklab(np.array([0.2, 0.4, 0.6])))

    result = ClippedSrgbFallbackStrategy().resolve(intent)

    assert result.srgb8 == (51, 102, 153)
    assert result.in_gamut


def test_slice_projection_passes_in_gamut_colour_through_untouched():
    hue = math.radians(140.0)
    model = LightnessChromaSliceModel(hue=hue)
    intent = ColourIntent.from_lch(0.5, 0.02, hue)

    result = SliceProjectionFallbackStrategy(model).resolve(intent)

    assert result.in_gamut
    assert result.resolved is intent
    assert result.srgb8 == tuple(int(v) for v in color_math.oklab_to_srgb8(intent.paint_oklab))


def test_slice_projection_lands_out_of_gamut_colour_on_the_slice_leaf():
    hue = math.radians(140.0)
    lightness = 0.5
    model = LightnessChromaSliceModel(hue=hue)
    chroma = float(color_math.max_chroma_for_lh(lightness, hue)) * 1.5
    intent = ColourIntent.from_lch(lightness, chroma, hue)

    result = SliceProjectionFallbackStrategy(model).resolve(intent)

    assert not result.in_gamut
    # Resolved is exactly the model's on-slice projection, in-gamut on this plane.
    assert result.resolved.selector_lch == model.project_onto_slice(intent.selector_lch)
    assert model.position_for_intent(result.resolved.selector_lch, (101.0, 101.0)) is not None
    assert result.srgb8 == tuple(int(v) for v in color_math.oklab_to_srgb8(result.resolved.paint_oklab))


def test_slice_projection_falls_back_to_clip_when_slice_hosts_no_in_gamut_point():
    class _NoProjection:
        def project_onto_slice(self, _lch):
            return None

    intent = ColourIntent.from_lch(0.6, color_math.SRGB_MAX_CHROMA, 0.0)

    result = SliceProjectionFallbackStrategy(_NoProjection()).resolve(intent)
    clipped = ClippedSrgbFallbackStrategy().resolve(intent)

    assert result.srgb8 == clipped.srgb8
    np.testing.assert_allclose(result.resolved.paint_oklab, clipped.resolved.paint_oklab, atol=1e-12)
    assert not result.in_gamut
