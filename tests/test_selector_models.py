import math

import numpy as np
import pytest

from oklab_colour_picker.domain import color_math
from oklab_colour_picker.models import (
    LightnessChromaSliceModel,
    HueLightnessSliceModel,
    LightnessSliceModel,
)


def test_lightness_slice_maps_center_to_neutral_current_lightness():
    model = LightnessSliceModel(lightness=0.62)

    actual = model.color_at_position((50.0, 50.0), (101.0, 101.0))

    np.testing.assert_allclose(actual, [0.62, 0.0, 0.0], atol=1e-12)


@pytest.mark.parametrize("lightness", [-0.01, 1.01, math.nan])
def test_lightness_slice_validates_lightness(lightness):
    with pytest.raises(ValueError, match="lightness"):
        LightnessSliceModel(lightness=lightness)


def test_lightness_slice_radius_maps_to_absolute_chroma():
    model = LightnessSliceModel(lightness=0.5)
    actual = model.color_at_position((75.0, 50.0), (101.0, 101.0))

    expected_chroma = 0.5 * color_math.SRGB_MAX_CHROMA
    np.testing.assert_allclose(actual, [0.5, expected_chroma, 0.0], atol=1e-12)


def test_lightness_slice_renders_transparent_outside_per_hue_gamut():
    model = LightnessSliceModel(lightness=0.5)
    assert model.color_at_position((100.0, 50.0), (101.0, 101.0)) is None


def test_lightness_slice_keeps_in_gamut_pixel_inside_per_hue_leaf():
    model = LightnessSliceModel(lightness=0.5)
    chroma = 0.20

    fraction = chroma / color_math.SRGB_MAX_CHROMA
    y = 50.0 + fraction * 50.0
    actual = model.color_at_position((50.0, y), (101.0, 101.0))

    np.testing.assert_allclose(
        actual,
        color_math.oklch_to_oklab([0.5, chroma, 3.0 * math.pi / 2.0]),
        atol=1e-12,
    )


@pytest.mark.parametrize("position", [(-1.0, 50.0), (50.0, -1.0), (101.0, 50.0), (50.0, 101.0), (0.0, 0.0)])
def test_lightness_slice_rejects_outside_square_or_circle(position):
    model = LightnessSliceModel(lightness=0.5)

    assert model.color_at_position(position, (101.0, 101.0)) is None


def test_lightness_slice_rejects_degenerate_size():
    model = LightnessSliceModel(lightness=0.5)

    assert model.color_at_position((0.0, 0.0), (1.0, 1.0)) is None
    assert model.position_for_intent((0.5, 0.0, 0.0), (1.0, 1.0)) is None


def test_lightness_slice_rejects_inverse_outside_gamut_boundary():
    model = LightnessSliceModel(lightness=0.5)

    assert model.position_for_intent((0.5, 0.3, 0.0), (101.0, 101.0)) is None


def test_lightness_slice_snap_clamps_to_per_hue_gamut():
    model = LightnessSliceModel(lightness=0.5)
    snapped = model.snapped_color_at_position((100.0, 50.0), (101.0, 101.0))

    assert snapped is not None
    lightness, chroma, hue = color_math.oklab_to_oklch(snapped)
    np.testing.assert_allclose(lightness, 0.5, atol=1e-12)
    np.testing.assert_allclose(hue % math.tau, 0.0, atol=1e-9)
    assert chroma < color_math.SRGB_MAX_CHROMA
    assert np.min(color_math.oklab_to_srgb(snapped)) == pytest.approx(-color_math.SRGB_GAMUT_EPSILON, abs=1e-10)


def test_lightness_slice_snap_returns_in_gamut_position_unchanged():
    model = LightnessSliceModel(lightness=0.55)
    position = (60.0, 45.0)
    expected = model.color_at_position(position, (101.0, 101.0))
    assert expected is not None

    snapped = model.snapped_color_at_position(position, (101.0, 101.0))
    np.testing.assert_allclose(snapped, expected, atol=1e-12)


@pytest.mark.parametrize("position", [(-1.0, 50.0), (50.0, 101.0), (0.0, 0.0)])
def test_lightness_slice_snap_projects_outside_positions_to_disk(position):
    model = LightnessSliceModel(lightness=0.5)
    snapped = model.snapped_color_at_position(position, (101.0, 101.0))

    assert snapped is not None
    assert model.position_for_intent(color_math.oklab_to_oklch(snapped), (101.0, 101.0)) is not None


def test_lightness_chart_chroma_max_envelopes_srgb_cusp():
    hues = np.linspace(0.0, math.tau, 4096, endpoint=False)
    _, chroma_cusp = color_math.find_cusp(np.cos(hues), np.sin(hues))
    assert color_math.SRGB_MAX_CHROMA >= float(np.max(chroma_cusp))


@pytest.mark.parametrize("model, size", [
    (LightnessSliceModel(0.55), (33, 23)),
    (HueLightnessSliceModel(0.05), (48, 32)),
    (LightnessChromaSliceModel(1.25), (34, 25)),
])
def test_vectorized_selector_valid_masks_match_scalar_picker_semantics(model, size):
    y, x = np.indices((size[1], size[0]), dtype=float)

    _, actual = model.colors_at_positions(x, y, size)
    expected = np.zeros((size[1], size[0]), dtype=bool)
    for row in range(size[1]):
        for column in range(size[0]):
            expected[row, column] = model.color_at_position((column, row), size) is not None

    np.testing.assert_array_equal(actual, expected)


def test_lightness_chroma_slice_maps_x_to_absolute_chroma():
    model = LightnessChromaSliceModel(hue=1.25)

    actual = model.color_at_position((25.0, 25.0), (101.0, 101.0))

    lightness = 0.75
    chroma = 0.25 * color_math.SRGB_MAX_CHROMA
    expected = color_math.oklch_to_oklab([lightness, chroma, 1.25])
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_lightness_chroma_slice_rejects_position_outside_per_hue_gamut():
    model = LightnessChromaSliceModel(hue=0.0)
    assert model.color_at_position((100.0, 50.0), (101.0, 101.0)) is None


def test_lightness_chroma_slice_snap_clamps_to_gamut_leaf():
    model = LightnessChromaSliceModel(hue=0.0)
    snapped = model.snapped_color_at_position((100.0, 50.0), (101.0, 101.0))

    assert snapped is not None
    lightness, chroma, hue = color_math.oklab_to_oklch(snapped)
    np.testing.assert_allclose(lightness, 0.5, atol=1e-12)
    np.testing.assert_allclose(hue, 0.0, atol=1e-12)
    assert chroma < color_math.SRGB_MAX_CHROMA
    assert np.min(color_math.oklab_to_srgb(snapped)) == pytest.approx(-color_math.SRGB_GAMUT_EPSILON, abs=1e-10)


def test_lightness_chroma_slice_snap_projects_outside_rect_to_edge():
    model = LightnessChromaSliceModel(hue=1.0)
    snapped = model.snapped_color_at_position((-20.0, 50.0), (101.0, 101.0))

    expected = model.color_at_position((0.0, 50.0), (101.0, 101.0))
    assert expected is not None
    np.testing.assert_allclose(snapped, expected, atol=1e-12)


@pytest.mark.parametrize("hue", [math.nan, math.inf, -math.inf])
def test_lightness_chroma_slice_validates_hue(hue):
    with pytest.raises(ValueError, match="hue"):
        LightnessChromaSliceModel(hue=hue)


def test_lightness_chroma_slice_normalizes_hue():
    model = LightnessChromaSliceModel(hue=math.tau + 0.25)

    assert model.hue == pytest.approx(0.25)


def test_lightness_chroma_slice_achromatic_endpoints_only_at_left_edge():
    model = LightnessChromaSliceModel(hue=0.0)

    np.testing.assert_allclose(model.color_at_position((0.0, 0.0), (101.0, 101.0)), [1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(model.color_at_position((0.0, 100.0), (101.0, 101.0)), [0.0, 0.0, 0.0], atol=1e-12)
    assert model.color_at_position((50.0, 0.0), (101.0, 101.0)) is None
    assert model.color_at_position((50.0, 100.0), (101.0, 101.0)) is None


def test_lightness_chroma_slice_rejects_degenerate_or_out_of_bounds_positions():
    model = LightnessChromaSliceModel(hue=0.0)

    assert model.color_at_position((0.0, 0.0), (1.0, 100.0)) is None
    assert model.color_at_position((101.0, 50.0), (101.0, 101.0)) is None
    assert model.color_at_position((50.0, -1.0), (101.0, 101.0)) is None


def test_lightness_chroma_slice_rejects_color_from_different_hue_plane():
    model = LightnessChromaSliceModel(hue=0.0)
    color = color_math.oklch_to_oklab([0.5, 0.05, math.pi / 2.0])

    assert model.position_for_intent(color_math.oklab_to_oklch(color), (101.0, 101.0)) is None

def test_lightness_chroma_slice_rejects_color_from_antipodal_hue_ray():
    model = LightnessChromaSliceModel(hue=0.0)
    color = color_math.oklch_to_oklab([0.5, 0.05, math.pi])

    assert model.position_for_intent(color_math.oklab_to_oklch(color), (101.0, 101.0)) is None


def test_lightness_chroma_slice_rejects_color_with_lightness_out_of_range():
    model = LightnessChromaSliceModel(hue=0.0)

    assert model.position_for_intent((1.5, 0.0, 0.0), (101.0, 101.0)) is None


def test_lightness_chroma_slice_accepts_near_neutral_round_trip_without_hue_sensitivity():
    model = LightnessChromaSliceModel(hue=1.75)
    color = color_math.oklch_to_oklab([0.5, 1e-12, 0.0])

    np.testing.assert_allclose(model.position_for_intent(color_math.oklab_to_oklch(color), (101.0, 101.0)), (0.0, 50.0), atol=1e-9)


def test_hue_lightness_slice_maps_angle_to_hue_and_radius_to_lightness():
    chroma = 0.05
    model = HueLightnessSliceModel(chroma=chroma)

    actual = model.color_at_position((50.0, 25.0), (101.0, 101.0))

    expected = color_math.oklch_to_oklab([0.5, chroma, math.pi / 2.0])
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_hue_lightness_slice_maps_center_to_white_and_rim_to_black():
    model = HueLightnessSliceModel(chroma=0.0)

    np.testing.assert_allclose(model.color_at_position((50.0, 50.0), (101.0, 101.0)), [1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(model.color_at_position((100.0, 50.0), (101.0, 101.0)), [0.0, 0.0, 0.0], atol=1e-12)


def test_hue_lightness_slice_selection_keeps_achromatic_click_hue():
    model = HueLightnessSliceModel(chroma=0.0)

    selection = model.selection_at_position((50.0, 20.0), (101.0, 101.0))

    assert selection is not None
    assert selection.lch == pytest.approx((0.4, 0.0, math.pi / 2.0))
    np.testing.assert_allclose(selection.paint_oklab, [0.4, 0.0, 0.0], atol=1e-12)


@pytest.mark.parametrize("chroma", [-0.01, math.nan])
def test_hue_lightness_slice_validates_chroma(chroma):
    with pytest.raises(ValueError, match="chroma"):
        HueLightnessSliceModel(chroma=chroma)


def test_hue_lightness_slice_rejects_position_outside_per_lightness_hue_gamut():
    model = HueLightnessSliceModel(chroma=0.2)

    assert model.color_at_position((50.0, 50.0), (101.0, 101.0)) is None


def test_hue_lightness_slice_snap_finds_nearest_valid_lightness_on_spoke():
    model = HueLightnessSliceModel(chroma=0.2)
    snapped = model.snapped_color_at_position((50.0, 50.0), (101.0, 101.0))

    assert snapped is not None
    assert model.position_for_intent(color_math.oklab_to_oklch(snapped), (101.0, 101.0)) is not None


def test_hue_lightness_slice_snap_projects_outside_circle_to_rim_angle():
    model = HueLightnessSliceModel(chroma=0.0)
    snapped = model.snapped_color_at_position((150.0, 50.0), (101.0, 101.0))

    expected = model.color_at_position((100.0, 50.0), (101.0, 101.0))
    assert expected is not None
    np.testing.assert_allclose(snapped, expected, atol=1e-12)


def test_hue_lightness_slice_achromatic_indicator_can_use_explicit_hue():
    hue = math.radians(210.0)
    model = HueLightnessSliceModel(chroma=0.0, achromatic_indicator_hue=hue)
    colour = color_math.oklch_to_oklab([0.5, 0.0, 0.0])

    assert model.position_for_intent(color_math.oklab_to_oklch(colour), (121.0, 121.0)) == pytest.approx(
        (
            60.0 + 30.0 * math.cos(hue),
            60.0 - 30.0 * math.sin(hue),
        )
    )


def test_hue_lightness_slice_normalizes_explicit_achromatic_indicator_hue_at_read_time():
    hue = math.radians(210.0)
    model = HueLightnessSliceModel(chroma=0.0, achromatic_indicator_hue=math.tau + hue)
    colour = color_math.oklch_to_oklab([0.5, 0.0, 0.0])

    assert model.position_for_intent(color_math.oklab_to_oklch(colour), (121.0, 121.0)) == pytest.approx(
        (
            60.0 + 30.0 * math.cos(hue),
            60.0 - 30.0 * math.sin(hue),
        )
    )


def test_hue_lightness_slice_rejects_degenerate_or_out_of_bounds_positions():
    model = HueLightnessSliceModel(chroma=0.05)

    assert model.color_at_position((0.0, 0.0), (1.0, 100.0)) is None
    assert model.color_at_position((101.0, 50.0), (101.0, 101.0)) is None
    assert model.color_at_position((50.0, -1.0), (101.0, 101.0)) is None
    assert model.color_at_position((0.0, 0.0), (101.0, 101.0)) is None


def test_hue_lightness_slice_rejects_color_with_mismatched_chroma():
    model = HueLightnessSliceModel(chroma=0.05)
    color = color_math.oklch_to_oklab([0.5, 0.06, math.pi / 2.0])

    assert model.position_for_intent(color_math.oklab_to_oklch(color), (101.0, 101.0)) is None


def test_lightness_slice_geometric_position_anchors_out_of_gamut_chroma():
    model = LightnessSliceModel(lightness=0.5)
    lch = color_math.oklab_to_oklch(color_math.oklch_to_oklab([0.5, color_math.SRGB_MAX_CHROMA, 0.0]))
    rim_at_hue_zero = (100.0, 50.0)

    assert model.position_for_intent(lch, (101.0, 101.0)) is None
    np.testing.assert_allclose(
        model.geometric_position_for_intent(lch, (101.0, 101.0)), rim_at_hue_zero, atol=1e-9
    )


def test_lightness_chroma_slice_geometric_position_anchors_out_of_gamut_chroma():
    model = LightnessChromaSliceModel(hue=0.0)
    lch = color_math.oklab_to_oklch(color_math.oklch_to_oklab([0.5, color_math.SRGB_MAX_CHROMA, 0.0]))
    full_chroma_mid_lightness = (100.0, 50.0)

    assert model.position_for_intent(lch, (101.0, 101.0)) is None
    np.testing.assert_allclose(
        model.geometric_position_for_intent(lch, (101.0, 101.0)), full_chroma_mid_lightness, atol=1e-9
    )


def test_lightness_slice_geometric_position_vanishes_beyond_srgb_max_chroma():
    model = LightnessSliceModel(lightness=0.5)
    lch = color_math.oklab_to_oklch(color_math.oklch_to_oklab([0.5, color_math.SRGB_MAX_CHROMA * 1.5, 0.0]))
    assert model.geometric_position_for_intent(lch, (101.0, 101.0)) is None


def test_lightness_slice_geometric_position_rejects_mismatched_lightness():
    model = LightnessSliceModel(lightness=0.5)
    off_slice = color_math.oklab_to_oklch(color_math.oklch_to_oklab([0.2, 0.05, 0.0]))
    assert model.geometric_position_for_intent(off_slice, (101.0, 101.0)) is None


def test_hue_lightness_projection_magnets_an_unreachable_hue_onto_the_disk():
    chroma = 0.20
    hue = math.radians(213.8)
    model = HueLightnessSliceModel(chroma=chroma)

    resolved = model.project_onto_slice((0.5, chroma, hue))

    assert resolved is not None
    assert resolved[1] == pytest.approx(chroma)        # stayed on this disk
    assert resolved[2] != pytest.approx(hue)           # hue was magnetted
    assert model.position_for_intent(resolved, (101.0, 101.0)) is not None
