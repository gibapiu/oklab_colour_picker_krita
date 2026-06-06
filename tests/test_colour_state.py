import math

import numpy as np
import pytest

from oklab_colour_picker.domain import color_math
from oklab_colour_picker.domain.colour_state import ColourIntent, normalize_oklab_for_krita


def test_colour_intent_from_lch_derives_paint_oklab():
    intent = ColourIntent.from_lch(0.62, 0.11, math.radians(245.0))

    np.testing.assert_allclose(
        intent.paint_oklab,
        color_math.oklch_to_oklab([0.62, 0.11, math.radians(245.0)]),
    )
    assert intent.selector_lch == pytest.approx(
        (0.62, 0.11, math.radians(245.0))
    )


def test_colour_intent_preserves_achromatic_hue_separately_from_paint():
    intent = ColourIntent.from_lch(0.5, 0.0, math.radians(210.0))

    np.testing.assert_allclose(intent.paint_oklab, [0.5, 0.0, 0.0], atol=1e-12)
    assert intent.hue == pytest.approx(math.radians(210.0))
    assert intent.is_achromatic


def test_colour_intent_from_achromatic_oklab_uses_fallback_hue():
    intent = ColourIntent.from_oklab(
        color_math.oklch_to_oklab([0.5, 0.0, 0.0]),
        achromatic_hue=math.radians(210.0),
    )

    assert intent.selector_lch == pytest.approx((0.5, 0.0, math.radians(210.0)))


def test_colour_intent_adopts_chromatic_hue_from_oklab():
    intent = ColourIntent.from_oklab(
        color_math.oklch_to_oklab([0.5, 0.04, math.radians(120.0)]),
        achromatic_hue=math.radians(210.0),
    )

    assert intent.hue == pytest.approx(math.radians(120.0))


def test_colour_intent_can_preserve_coordinates_across_krita_readback():
    intent = ColourIntent.from_lch(0.5, 0.0, math.radians(210.0))
    readback = intent.quantized_paint_oklab

    normalized = intent.with_krita_paint_oklab(readback)

    np.testing.assert_allclose(normalized.paint_oklab, readback)
    assert normalized.selector_lch == pytest.approx(intent.selector_lch)


def test_colour_intent_rejects_mismatched_krita_readback():
    intent = ColourIntent.from_lch(0.5, 0.0, math.radians(210.0))

    with pytest.raises(ValueError):
        intent.with_krita_paint_oklab(color_math.oklch_to_oklab([0.8, 0.0, 0.0]))


def test_colour_intent_quantized_paint_equality_ignores_achromatic_hue():
    first = ColourIntent.from_lch(0.5, 0.0, 0.0)
    second = ColourIntent.from_lch(0.5, 0.0, math.radians(210.0))

    assert first.quantized_paint_equal(second)


def test_colour_intent_quantized_paint_is_cached_and_immutable():
    intent = ColourIntent.from_lch(0.5, 0.0, 0.0)

    assert intent.quantized_paint_oklab is intent.quantized_paint_oklab
    assert not intent.quantized_paint_oklab.flags.writeable
    np.testing.assert_allclose(
        intent.quantized_paint_oklab,
        normalize_oklab_for_krita(intent.paint_oklab),
    )


def test_colour_intent_exposes_paint_only_by_name():
    intent = ColourIntent.from_lch(0.5, 0.0, 0.0)

    assert not hasattr(intent, "copy")
    with pytest.raises(TypeError):
        np.asarray(intent, dtype=float)


def test_colour_intent_rejects_public_positional_construction():
    with pytest.raises(TypeError):
        ColourIntent((0.5, 0.0, 0.0), 0.5, 0.0, 0.0)


@pytest.mark.parametrize(
    "factory,args",
    [
        (ColourIntent.from_lch, (math.nan, 0.0, 0.0)),
        (ColourIntent.from_lch, (0.5, math.inf, 0.0)),
        (ColourIntent.from_lch, (0.5, 0.0, math.nan)),
        (ColourIntent.from_oklab, ([0.5, 0.0],)),
        (ColourIntent.from_oklab, ([0.5, 0.0, math.nan],)),
    ],
)
def test_colour_intent_rejects_invalid_values(factory, args):
    with pytest.raises(ValueError):
        factory(*args)
