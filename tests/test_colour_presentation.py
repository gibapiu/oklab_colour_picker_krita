import numpy as np

from oklab_colour_picker import color_math
from oklab_colour_picker.colour_presentation import ColourPresenter
from oklab_colour_picker.colour_state import ColourIntent
from oklab_colour_picker.gamut_fallback import ClippedSrgbFallbackStrategy


def test_colour_presenter_resolves_intent_and_fallback_once():
    intent = ColourIntent.from_lch(0.6, color_math.SRGB_MAX_CHROMA, 0.0)

    presentation = ColourPresenter(ClippedSrgbFallbackStrategy()).present(intent)

    assert presentation.intent is intent
    assert presentation.fallback.source is intent
    assert presentation.resolved_lch == presentation.fallback.resolved.selector_lch
    assert presentation.fallback.srgb8 == tuple(
        int(v) for v in color_math.oklab_to_srgb8(intent.paint_oklab)
    )


def test_colour_presenter_preserves_achromatic_hue_for_raw_oklab():
    grey = color_math.srgb_to_oklab(np.array([0.5, 0.5, 0.5]))
    hue = np.radians(210.0)

    presentation = ColourPresenter(ClippedSrgbFallbackStrategy()).present(
        grey,
        achromatic_hue=hue,
    )

    assert presentation.intent.hue == hue
