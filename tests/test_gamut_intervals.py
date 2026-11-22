"""Regressions for sRGB slices that leave and re-enter the gamut near blue."""

import math

import numpy as np
import pytest

from oklab_colour_picker.domain import color_math
from oklab_colour_picker.domain.colour_state import ColourIntent
from oklab_colour_picker.domain.gamut_fallback import SliceProjectionFallbackStrategy
from oklab_colour_picker.models import (
    HueLightnessSliceModel,
    LightnessChromaSliceModel,
    LightnessSliceModel,
)


BLUE_LCH = color_math.oklab_to_oklch(
    color_math.srgb_to_oklab(np.array([0, 30, 224]) / 255.0)
)
SIZE = (501, 501)


def _models(lch):
    lightness, chroma, hue = lch
    return (
        LightnessSliceModel(lightness),
        HueLightnessSliceModel(chroma),
        LightnessChromaSliceModel(hue),
    )


@pytest.mark.parametrize("chroma", [0.270, 0.272, BLUE_LCH[1], 0.280])
def test_blue_gap_membership_agrees_for_picking_position_and_rendering(chroma):
    lightness, _, hue = BLUE_LCH
    lch = (lightness, chroma, hue)
    srgb = color_math.oklab_to_srgb(color_math.oklch_to_oklab(lch))
    expected = bool(color_math.in_srgb_gamut(srgb, epsilon=1e-6))
    assert expected == (chroma != 0.272)

    for model in _models(lch):
        point = model.geometric_position_for_intent(lch, SIZE)
        assert point is not None
        _, rendered = model.colors_at_positions(np.array(point[0]), np.array(point[1]), SIZE)
        assert bool(rendered) is expected
        assert (model.color_at_position(point, SIZE) is not None) is expected
        assert (model.position_for_intent(lch, SIZE) is not None) is expected


@pytest.mark.parametrize("chroma", [0.2715, 0.2728])
def test_chroma_projection_chooses_nearest_side_of_blue_gap(chroma):
    lightness, _, hue = BLUE_LCH
    expected = _local_boundary((lightness, 0.272, hue), 1, 0.270 if chroma == 0.2715 else 0.280)
    for model in (LightnessSliceModel(lightness), LightnessChromaSliceModel(hue)):
        lch = (lightness, chroma, hue)
        result = model.project_onto_slice(lch)
        assert result is not None
        assert result == pytest.approx((lightness, expected, hue), abs=1e-10)
        _assert_in_gamut(result)
        point = model.geometric_position_for_intent(lch, SIZE)
        snapped = model.snapped_selector_selection_at_position(point, SIZE)
        assert snapped is not None
        assert snapped.lch == pytest.approx(result, abs=1e-10)


def test_lightness_projection_chooses_nearest_side_of_internal_gap():
    _, _, hue = BLUE_LCH
    model = HueLightnessSliceModel(0.272)
    for lightness, valid_lightness in ((0.425, 0.424), (0.4268, 0.428)):
        expected = _local_boundary((lightness, model.chroma, hue), 0, valid_lightness)
        result = model.project_onto_slice((lightness, model.chroma, hue))
        assert result == pytest.approx((expected, model.chroma, hue), abs=1e-10)
        _assert_in_gamut(result)


def test_fallback_resolves_blue_gap_and_preserves_reentered_blue():
    lightness, _, hue = BLUE_LCH
    strategy = SliceProjectionFallbackStrategy(LightnessChromaSliceModel(hue))
    invalid = strategy.resolve(ColourIntent.from_lch(lightness, 0.272, hue))
    assert not invalid.in_gamut
    assert invalid.resolved.chroma != pytest.approx(0.272, abs=1e-6)
    _assert_in_gamut(invalid.resolved.selector_lch)
    valid = ColourIntent.from_lch(*BLUE_LCH)
    assert strategy.resolve(valid).resolved is valid


def test_chroma_intervals_include_both_valid_blue_bands():
    lightness, _, hue = BLUE_LCH
    actual = color_math.chroma_intervals_for_lh(lightness, hue)
    assert len(actual) == 2
    assert actual[0][0] == 0.0
    assert actual[0][1] == pytest.approx(_local_boundary((lightness, 0.272, hue), 1, 0.270), abs=1e-10)
    assert actual[1][0] == pytest.approx(_local_boundary((lightness, 0.272, hue), 1, 0.280), abs=1e-10)
    assert actual[1][1] == pytest.approx(_local_boundary((lightness, 0.31, hue), 1, 0.280), abs=1e-10)


@pytest.mark.parametrize("hue_degrees", [0, 90, 180, 264, 264.052, 264.1, 264.207587, 264.21, 264.4, 300])
@pytest.mark.parametrize("axis", ["chroma", "lightness"])
def test_intervals_agree_with_independent_dense_rgb_samples(hue_degrees, axis):
    hue = math.radians(hue_degrees)
    if axis == "chroma":
        values = np.linspace(0, color_math.SRGB_MAX_CHROMA, 10001)
        intervals = color_math.chroma_intervals_for_lh(BLUE_LCH[0], hue)
        lch = np.stack((np.full_like(values, BLUE_LCH[0]), values, np.full_like(values, hue)), axis=-1)
    else:
        values = np.linspace(0, 1, 10001)
        intervals = color_math.lightness_intervals_for_ch(0.272, hue)
        lch = np.stack((values, np.full_like(values, 0.272), np.full_like(values, hue)), axis=-1)
    actual = np.zeros_like(values, dtype=bool)
    for lower, upper in intervals:
        actual |= (lower <= values) & (values <= upper)
    srgb = color_math.oklab_to_srgb(color_math.oklch_to_oklab(lch))
    epsilon = color_math.SRGB_GAMUT_EPSILON
    away_from_faces = np.all((np.abs(srgb + epsilon) > 1e-10) & (np.abs(srgb - 1 - epsilon) > 1e-10), axis=-1)
    expected = color_math.in_srgb_gamut(srgb)
    np.testing.assert_array_equal(actual[away_from_faces], expected[away_from_faces])


@pytest.mark.parametrize("lightness", [0.0, 1.0])
def test_chroma_projection_handles_neutral_endpoints(lightness):
    model = LightnessChromaSliceModel(BLUE_LCH[2])
    result = model.project_onto_slice((lightness, 0.1, model.hue))
    assert result is not None
    assert result[0] == lightness
    expected = _local_boundary((lightness, 0.1, model.hue), 1, 0)
    assert result[1] == pytest.approx(expected, abs=1e-8)
    _assert_in_gamut(result)


def test_empty_lightness_slice_has_no_intervals():
    assert color_math.lightness_intervals_for_ch(0.5, 0) == ()


@pytest.mark.parametrize("rgb", [(1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 1, 1), (1, 0, 1), (1, 1, 0)])
def test_lightness_intervals_preserve_gamut_vertices_and_nearby_projections(rgb):
    lightness, chroma, hue = color_math.oklab_to_oklch(color_math.srgb_to_oklab(rgb))
    intervals = color_math.lightness_intervals_for_ch(chroma, hue)
    assert intervals
    assert min(abs(np.clip(lightness, lo, hi) - lightness) for lo, hi in intervals) < 1e-10
    for desired in (lightness - 0.01, lightness + 0.01):
        snapped = color_math.snap_lightness(desired, chroma, hue)
        expected = _local_boundary((desired, chroma, hue), 0, lightness)
        assert snapped == pytest.approx(expected, abs=1e-10)
        _assert_in_gamut((snapped, chroma, hue))


@pytest.mark.parametrize("chroma", [BLUE_LCH[1], 0.280])
def test_chroma_projection_preserves_valid_colours_in_second_interval(chroma):
    lightness, _, hue = BLUE_LCH
    for model in (LightnessSliceModel(lightness), LightnessChromaSliceModel(hue)):
        desired = (lightness, chroma, hue)
        assert model.project_onto_slice(desired) == desired


def test_fallback_and_rendering_share_boundary_tolerance():
    lab = color_math.srgb_to_oklab((-0.5e-6, 0.2, 0.8))
    lch = color_math.oklab_to_oklch(lab)
    model = LightnessChromaSliceModel(lch[2])
    point = model.position_for_intent(lch, SIZE)
    assert point is not None
    _, valid = model.colors_at_positions(np.array(point[0]), np.array(point[1]), SIZE)
    assert valid
    assert SliceProjectionFallbackStrategy(model).resolve(lab).in_gamut


def _assert_in_gamut(lch):
    srgb = color_math.oklab_to_srgb(color_math.oklch_to_oklab(lch))
    assert color_math.in_srgb_gamut(srgb, epsilon=1e-6)


def _local_boundary(invalid_lch, axis, valid_coordinate):
    invalid = invalid_lch[axis]
    valid = valid_coordinate
    lch = list(invalid_lch)
    for _ in range(60):
        lch[axis] = (invalid + valid) * 0.5
        rgb = color_math.oklab_to_srgb(color_math.oklch_to_oklab(lch))
        if color_math.in_srgb_gamut(rgb, epsilon=1e-6):
            valid = lch[axis]
        else:
            invalid = lch[axis]
    return valid


def test_near_black_intervals_match_tolerant_edge_without_singleton_artifacts():
    lightness, hue = 0.0002, math.radians(227.56)
    intervals = color_math.chroma_intervals_for_lh(lightness, hue)
    assert len(intervals) == 1
    lower, upper = intervals[0]
    assert lower == 0.0
    expected = _local_boundary((lightness, 0.1, hue), 1, 0)
    assert upper == pytest.approx(expected, abs=1e-8)
    assert color_math.snap_chroma(lightness, 0.1, hue) == pytest.approx(expected, abs=1e-8)


def test_chroma_invariant_is_checked_explicitly(monkeypatch):
    monkeypatch.setattr(color_math, "chroma_intervals_for_lh", lambda *_args: ())
    with pytest.raises(AssertionError, match="host neutral"):
        color_math.snap_chroma(0.5, 0.3, 0)


def test_chroma_projection_keeps_boundary_roundoff_inside_tolerance():
    lightness, hue = 0.6409797422760979, 3.9650397599946285
    chroma = color_math.snap_chroma(lightness, 0.3, hue)
    _assert_in_gamut((lightness, chroma, hue))
