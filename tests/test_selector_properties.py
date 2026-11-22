import math

import numpy as np
import pytest
from hypothesis import example, given, settings, strategies as st

from oklab_colour_picker.domain import color_math
from oklab_colour_picker.models import (
    HueLightnessSliceModel,
    LightnessChromaSliceModel,
    LightnessSliceModel,
)


MODES = st.sampled_from(("hue_chroma", "hue_lightness", "lightness_chroma"))
UNIT = st.floats(0, 1, allow_nan=False, allow_infinity=False)


def _model(mode, lch):
    lightness, chroma, hue = lch
    return {
        "hue_chroma": lambda: LightnessSliceModel(lightness),
        "hue_lightness": lambda: HueLightnessSliceModel(chroma),
        "lightness_chroma": lambda: LightnessChromaSliceModel(hue),
    }[mode]()


@settings(max_examples=250, derandomize=True, deadline=None)
@given(mode=MODES, rgb=st.tuples(UNIT, UNIT, UNIT), width=st.integers(16, 128), height=st.integers(16, 128))
def test_in_gamut_colour_round_trips_through_scalar_and_vector_coordinates(mode, rgb, width, height):
    lab = color_math.srgb_to_oklab(rgb)
    lch = color_math.oklab_to_oklch(lab)
    model = _model(mode, lch)
    size = (width, height)
    position = model.position_for_intent(lch, size)
    assert position is not None
    colour = model.color_at_position(position, size)
    assert colour is not None
    np.testing.assert_allclose(colour, lab, atol=1e-10, rtol=0)
    vector_colour, selectable = model.colors_at_positions(np.array(position[0]), np.array(position[1]), size)
    assert selectable
    np.testing.assert_allclose(vector_colour, lab, atol=1e-10, rtol=0)


@settings(max_examples=500, derandomize=True, deadline=None)
@example(mode="hue_lightness", lightness=0.5, chroma=0.22, hue=math.radians(213.8))
@example(mode="lightness_chroma", lightness=0.426041573241465, chroma=0.272, hue=math.radians(264.207587271901))
@given(mode=MODES, lightness=UNIT, chroma=st.floats(0, 0.30), hue=st.floats(0, math.tau))
def test_projection_is_valid_on_its_plane_and_preserves_valid_colours(mode, lightness, chroma, hue):
    lch = (lightness, chroma, hue)
    model = _model(mode, lch)
    resolved = model.project_onto_slice(lch)
    assert resolved is not None
    rgb = color_math.oklab_to_srgb(color_math.oklch_to_oklab(resolved))
    assert np.all((rgb >= -1e-6) & (rgb <= 1 + 1e-6))
    assert model.position_for_intent(resolved, (101, 101)) is not None
    if mode == "hue_chroma":
        assert resolved[0] == lightness
    elif mode == "hue_lightness":
        assert resolved[1] == chroma
    else:
        assert abs((resolved[2] - hue + math.pi) % math.tau - math.pi) < 1e-9
    source_rgb = color_math.oklab_to_srgb(color_math.oklch_to_oklab(lch))
    if np.all((source_rgb >= -1e-6) & (source_rgb <= 1 + 1e-6)):
        np.testing.assert_allclose(resolved[:2], lch[:2], atol=1e-12, rtol=0)
        assert abs((resolved[2] - hue + math.pi) % math.tau - math.pi) < 1e-9


def test_hue_wrap_keeps_the_same_slice_position():
    colour = (0.5, 0.02, 0.0)
    position = LightnessChromaSliceModel(0).position_for_intent(colour, (101, 101))
    assert position is not None
    assert LightnessChromaSliceModel(math.tau - 1e-12).position_for_intent(colour, (101, 101)) == pytest.approx(position)


@pytest.mark.parametrize("lightness, expected_y", [(0.0, 100), (1.0, 0)])
def test_lightness_endpoints_keep_neutrals_and_reject_chroma(lightness, expected_y):
    model = LightnessChromaSliceModel(0)
    lch = (lightness, 0.0, 0.0)
    assert model.position_for_intent(lch, (101, 101)) == (0.0, expected_y)
    np.testing.assert_allclose(model.color_at_position((0, expected_y), (101, 101)), (lightness, 0, 0), atol=1e-12)
    assert HueLightnessSliceModel(0.05).position_for_intent((lightness, 0.05, 0), (101, 101)) is None
