import math

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings, strategies as st

from oklab_colour_picker.domain import color_math
from oklab_colour_picker.app.controller import normalize_oklab_for_krita
from oklab_colour_picker.models import (
    HueLightnessSliceModel,
    LightnessChromaSliceModel,
    LightnessSliceModel,
)


MODEL_CASES = (
    ("lightness-slice", LightnessSliceModel(lightness=0.55)),
    ("hue-lightness-slice", HueLightnessSliceModel(chroma=0.03)),
    ("lightness-chroma-slice", LightnessChromaSliceModel(hue=1.0)),
)
@settings(
    max_examples=200,
    derandomize=True,
    deadline=None,
    # The lightness-slice case rejects ~88% of random pixel draws as OOG.
    suppress_health_check=[HealthCheck.filter_too_much],
)
@given(
    case=st.sampled_from(MODEL_CASES),
    width=st.integers(min_value=16, max_value=96),
    height=st.integers(min_value=16, max_value=96),
    x_fraction=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    y_fraction=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)
def test_p1_model_position_colour_round_trip_is_stable_within_quantization(case, width, height, x_fraction, y_fraction):
    _name, model = case
    size = (width, height)
    position = (x_fraction * (width - 1.0), y_fraction * (height - 1.0))
    colour = model.color_at_position(position, size)
    assume(colour is not None)

    round_tripped_position = model.position_for_intent(color_math.oklab_to_oklch(colour), size)
    assume(round_tripped_position is not None)
    round_tripped_colour = model.color_at_position(round_tripped_position, size)

    assert round_tripped_colour is not None
    np.testing.assert_allclose(
        normalize_oklab_for_krita(round_tripped_colour),
        normalize_oklab_for_krita(colour),
        atol=1e-12,
    )


@settings(
    max_examples=200,
    derandomize=True,
    deadline=None,
    suppress_health_check=[HealthCheck.filter_too_much],
)
@given(
    case=st.sampled_from(MODEL_CASES),
    width=st.integers(min_value=16, max_value=96),
    height=st.integers(min_value=16, max_value=96),
    x_fraction=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    y_fraction=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)
def test_p2_model_position_lookup_is_idempotent_for_same_colour(case, width, height, x_fraction, y_fraction):
    _name, model = case
    size = (width, height)
    colour = model.color_at_position((x_fraction * (width - 1.0), y_fraction * (height - 1.0)), size)
    assume(colour is not None)

    once = model.position_for_intent(color_math.oklab_to_oklch(colour), size)
    twice = model.position_for_intent(color_math.oklab_to_oklch(np.asarray(colour, dtype=float).copy()), size)

    assert once == pytest.approx(twice)


def test_edge_hue_wrap_position_is_stable_across_zero_tau():
    zero_model = LightnessChromaSliceModel(hue=0.0)
    tau_model = LightnessChromaSliceModel(hue=math.tau - 1e-12)
    colour = color_math.oklch_to_oklab([0.5, 0.02, 0.0])

    zero_position = zero_model.position_for_intent(color_math.oklab_to_oklch(colour), (101, 101))
    tau_position = tau_model.position_for_intent(color_math.oklab_to_oklch(colour), (101, 101))

    assert zero_position is not None
    assert tau_position == pytest.approx(zero_position, abs=1e-6)


@pytest.mark.parametrize(("lightness", "expected_y"), [(0.0, 100.0), (1.0, 0.0)])
def test_edge_lightness_chroma_slice_accepts_achromatic_lightness_boundaries(lightness, expected_y):
    model = LightnessChromaSliceModel(hue=0.0)
    colour = color_math.oklch_to_oklab([lightness, 0.0, 0.0])

    position = model.position_for_intent(color_math.oklab_to_oklch(colour), (101, 101))
    round_tripped = model.color_at_position((0.0, expected_y), (101, 101))

    assert position == pytest.approx((0.0, expected_y), abs=1e-6)
    assert round_tripped is not None
    np.testing.assert_allclose(round_tripped, colour, atol=1e-12)


@pytest.mark.parametrize("lightness", [0.0, 1.0])
def test_edge_hue_lightness_slice_rejects_positive_chroma_at_lightness_boundaries(lightness):
    model = HueLightnessSliceModel(chroma=0.05)
    colour = color_math.oklch_to_oklab([lightness, 0.05, 0.0])

    assert model.position_for_intent(color_math.oklab_to_oklch(colour), (101, 101)) is None


def test_edge_quantization_boundary_colours_compare_equal_after_krita_normalization():
    raw = np.array([0.55, 0.02, -0.03])
    normalized_once = normalize_oklab_for_krita(raw)

    np.testing.assert_allclose(
        normalize_oklab_for_krita(normalized_once),
        normalized_once,
        atol=0.0,
    )


def _on_plane_lch(model, first, second):
    """Build an OKLCh that lies on ``model``'s plane (its fixed coordinate)."""

    lightness = 0.02 + 0.96 * first
    if isinstance(model, LightnessSliceModel):
        return (model.lightness, second * color_math.SRGB_MAX_CHROMA, first * math.tau)
    if isinstance(model, HueLightnessSliceModel):
        return (lightness, model.chroma, second * math.tau)
    return (lightness, second * color_math.SRGB_MAX_CHROMA, model.hue)


@settings(max_examples=300, derandomize=True, deadline=None)
@given(
    case=st.sampled_from(MODEL_CASES),
    first=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    second=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)
def test_p_projection_always_lands_on_slice_inside_gamut(case, first, second):
    _name, model = case
    lch = _on_plane_lch(model, first, second)

    resolved = model.project_onto_slice(lch)

    # These concrete slice models can host every generated on-plane colour.
    # position_for_intent enforces that the result stayed on the same plane and
    # landed inside its gamut leaf.
    assert resolved is not None
    assert model.position_for_intent(resolved, (101.0, 101.0)) is not None


@settings(max_examples=400, derandomize=True, deadline=None)
@given(
    chroma=st.floats(min_value=0.0, max_value=0.30, allow_nan=False, allow_infinity=False),
    hue=st.floats(min_value=0.0, max_value=math.tau, allow_nan=False, allow_infinity=False),
    lightness=st.floats(min_value=0.02, max_value=0.98, allow_nan=False, allow_infinity=False),
)
def test_p_hue_lightness_projection_never_blinks_off_the_disk(chroma, hue, lightness):
    # The fixed-chroma disk has hues that cannot host a high chroma. The hue
    # magnet keeps the fallback on the disk for every such colour (within the
    # sRGB chroma extent), so it never falls off-plane to the global clip.
    model = HueLightnessSliceModel(chroma=chroma)

    resolved = model.project_onto_slice((lightness, chroma, hue))

    assert resolved is not None
    assert resolved[1] == pytest.approx(chroma)
    assert model.position_for_intent(resolved, (101.0, 101.0)) is not None
