import numpy as np

from oklab_colour_picker.domain import color_math
from oklab_colour_picker.domain.colour_state import ColourIntent
from oklab_colour_picker.domain.gamut_fallback import ClippedSrgbFallbackStrategy


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
