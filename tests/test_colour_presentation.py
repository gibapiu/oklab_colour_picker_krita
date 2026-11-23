import numpy as np

from oklab_colour_picker.domain import color_math
from oklab_colour_picker.domain.colour_presentation import ColourPresenter
from oklab_colour_picker.domain.colour_state import ColourIntent
from oklab_colour_picker.domain.gamut_fallback import ClippedSrgbFallbackStrategy


def test_colour_presenter_resolves_intent_and_fallback_once():
    intent = ColourIntent.from_lch(0.6, color_math.SRGB_MAX_CHROMA, 0.0)
    fallback = ClippedSrgbFallbackStrategy().resolve(intent)
    calls = []

    class Strategy:
        def resolve(self, colour):
            calls.append(colour)
            return fallback

    presentation = ColourPresenter(Strategy()).present(intent)

    assert calls == [intent]
    assert presentation.intent is intent
    assert presentation.fallback is fallback
    assert presentation.resolved_lch == fallback.resolved.selector_lch
    assert presentation.srgb8 == fallback.srgb8


def test_colour_presenter_preserves_achromatic_hue_for_raw_oklab():
    grey = color_math.srgb_to_oklab(np.array([0.5, 0.5, 0.5]))
    hue = np.radians(210.0)

    presentation = ColourPresenter(ClippedSrgbFallbackStrategy()).present(
        grey,
        achromatic_hue=hue,
    )

    assert presentation.intent.hue == hue
