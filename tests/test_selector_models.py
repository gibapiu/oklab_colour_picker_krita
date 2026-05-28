import math

import numpy as np
import pytest

from oklab_colour_picker import color_math
from oklab_colour_picker import models as selector_model_package
from oklab_colour_picker import selector_models as selector_model_facade
from oklab_colour_picker.controller import normalize_oklab_for_krita
from oklab_colour_picker.selector_models import (
    IndicatorSpec,
    LightnessChromaSliceModel,
    HueLightnessSliceModel,
    LightnessSliceModel,
    SelectorModel,
    SelectorSelection,
    disk_geometry,
)


def test_selector_models_facade_exports_model_package_contract():
    assert selector_model_package.IndicatorSpec is IndicatorSpec
    assert selector_model_package.SelectorModel is SelectorModel
    assert selector_model_package.SelectorSelection is SelectorSelection
    assert selector_model_package.LightnessSliceModel is LightnessSliceModel
    assert selector_model_facade.disk_geometry is disk_geometry


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
    # Halfway from centre to the rim along +x (hue=0). Radius is normalised
    # against color_math.SRGB_MAX_CHROMA, not the per-hue gamut max.
    actual = model.color_at_position((75.0, 50.0), (101.0, 101.0))

    expected_chroma = 0.5 * color_math.SRGB_MAX_CHROMA
    if expected_chroma > color_math.max_chroma_for_lh(0.5, 0.0):
        assert actual is None
    else:
        np.testing.assert_allclose(actual, [0.5, expected_chroma, 0.0], atol=1e-12)


def test_lightness_slice_renders_transparent_outside_per_hue_gamut():
    # At hue=0 (red-ish) the L=0.5 cusp chroma sits well below
    # color_math.SRGB_MAX_CHROMA, so the rim along +x is outside gamut.
    model = LightnessSliceModel(lightness=0.5)
    assert model.color_at_position((100.0, 50.0), (101.0, 101.0)) is None


def test_lightness_slice_keeps_in_gamut_pixel_inside_per_hue_leaf():
    model = LightnessSliceModel(lightness=0.5)
    # Blue (hue=3pi/2). Sit inside the gamut leaf at chroma well below cusp.
    chroma = 0.20
    assert chroma < color_math.max_chroma_for_lh(0.5, 3.0 * math.pi / 2.0)

    fraction = chroma / color_math.SRGB_MAX_CHROMA
    # hue=3pi/2 -> atan2(dy,dx) with dy<0, dx=0; dy = center_y - y, so y > center_y.
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
    max_chroma = color_math.max_chroma_for_lh(0.5, 0.0)

    assert model.position_for_intent((0.5, max_chroma * 1.01, 0.0), (101.0, 101.0)) is None


def test_lightness_slice_snap_clamps_to_per_hue_gamut():
    # At hue=0 (red-ish) and L=0.5 the cusp chroma sits well below
    # color_math.SRGB_MAX_CHROMA, so the rim along +x is past the leaf.
    model = LightnessSliceModel(lightness=0.5)
    snapped = model.snapped_color_at_position((100.0, 50.0), (101.0, 101.0))

    assert snapped is not None
    lightness, chroma, hue = color_math.oklab_to_oklch(snapped)
    np.testing.assert_allclose(lightness, 0.5, atol=1e-12)
    np.testing.assert_allclose(hue % math.tau, 0.0, atol=1e-9)
    np.testing.assert_allclose(chroma, color_math.max_chroma_for_lh(0.5, 0.0), atol=1e-9)


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


@pytest.mark.parametrize("position", [(60.0, 45.0), (40.0, 50.0), (50.0, 60.0)])
def test_lightness_slice_round_trips_position_and_color(position):
    # Positions chosen to sit well inside the L=0.55 gamut leaf for any hue,
    # so round-tripping doesn't depend on per-hue cusp variation.
    model = LightnessSliceModel(lightness=0.55)

    color = model.color_at_position(position, (101.0, 101.0))
    assert color is not None
    actual = model.position_for_intent(color_math.oklab_to_oklch(color), (101.0, 101.0))

    np.testing.assert_allclose(actual, position, atol=1e-9)


def test_lightness_chart_chroma_max_envelopes_srgb_cusp():
    # Hardcoded chart extent must stay above the largest sRGB cusp chroma so
    # the entire gamut leaf remains addressable on the picker.
    hues = np.linspace(0.0, math.tau, 4096, endpoint=False)
    _, chroma_cusp = color_math.find_cusp(np.cos(hues), np.sin(hues))
    assert color_math.SRGB_MAX_CHROMA >= float(np.max(chroma_cusp))


@pytest.mark.parametrize(
    "model",
    [
        LightnessSliceModel(lightness=0.55),
        HueLightnessSliceModel(chroma=0.05),
        LightnessChromaSliceModel(hue=1.25),
    ],
)
def test_vectorized_selector_render_paths_do_not_use_halley_boundary_solver(monkeypatch, model):
    def fail_on_boundary_solver(*_args, **_kwargs):
        raise AssertionError("colors_at_positions must not call max_chroma_for_lh")

    monkeypatch.setattr(color_math, "max_chroma_for_lh", fail_on_boundary_solver)
    y, x = np.indices((32, 32), dtype=float)

    oklab, valid = model.colors_at_positions(x, y, (32, 32))

    assert oklab.shape == (32, 32, 3)
    assert valid.shape == (32, 32)


@pytest.mark.parametrize(
    "model",
    [
        LightnessSliceModel(lightness=0.55),
        HueLightnessSliceModel(chroma=0.05),
        LightnessChromaSliceModel(hue=1.25),
    ],
)
@pytest.mark.parametrize(
    "size",
    [
        (33, 33),
        (48, 32),
        # Fixed-seed, non-square sizes harden the strict mask parity check
        # without making collection depend on runtime randomness.
        (58, 71),
        (50, 34),
    ],
)
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
    # Hue 0 (red) cusp chroma is well below color_math.SRGB_MAX_CHROMA, so the right
    # edge at mid-lightness lies outside the achievable per-hue gamut.
    assert model.color_at_position((100.0, 50.0), (101.0, 101.0)) is None


def test_lightness_chroma_slice_snap_clamps_to_gamut_leaf():
    model = LightnessChromaSliceModel(hue=0.0)
    snapped = model.snapped_color_at_position((100.0, 50.0), (101.0, 101.0))

    assert snapped is not None
    lightness, chroma, hue = color_math.oklab_to_oklch(snapped)
    np.testing.assert_allclose(lightness, 0.5, atol=1e-12)
    np.testing.assert_allclose(hue, 0.0, atol=1e-12)
    np.testing.assert_allclose(chroma, color_math.max_chroma_for_lh(0.5, 0.0), atol=1e-9)


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
    # Off the L=0/L=1 axis chroma must vanish, so positive x is out of gamut.
    assert model.color_at_position((50.0, 0.0), (101.0, 101.0)) is None
    assert model.color_at_position((50.0, 100.0), (101.0, 101.0)) is None


@pytest.mark.parametrize(("color", "expected"), [([1.0, 0.0, 0.0], (0.0, 0.0)), ([0.0, 0.0, 0.0], (0.0, 100.0))])
def test_lightness_chroma_slice_inverse_collapses_achromatic_endpoint_rows_to_left_edge(color, expected):
    model = LightnessChromaSliceModel(hue=0.0)

    np.testing.assert_allclose(model.position_for_intent(color_math.oklab_to_oklch(color), (101.0, 101.0)), expected, atol=1e-12)


def test_lightness_chroma_slice_rejects_degenerate_or_out_of_bounds_positions():
    model = LightnessChromaSliceModel(hue=0.0)

    assert model.color_at_position((0.0, 0.0), (1.0, 100.0)) is None
    assert model.color_at_position((101.0, 50.0), (101.0, 101.0)) is None
    assert model.color_at_position((50.0, -1.0), (101.0, 101.0)) is None


@pytest.mark.parametrize("position", [(0.0, 50.0), (15.0, 20.0), (10.0, 75.0)])
def test_lightness_chroma_slice_round_trips_position_and_color(position):
    model = LightnessChromaSliceModel(hue=2.0)

    color = model.color_at_position(position, (101.0, 101.0))
    actual = model.position_for_intent(color_math.oklab_to_oklch(color), (101.0, 101.0))

    np.testing.assert_allclose(actual, position, atol=1e-9)


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
    color = color_math.oklch_to_oklab([0.5, 1e-12, model.hue])

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


def test_hue_lightness_slice_accepts_achromatic_full_lightness_range():
    model = HueLightnessSliceModel(chroma=0.0)

    np.testing.assert_allclose(model.color_at_position((50.0, 50.0), (101.0, 101.0)), [1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(model.color_at_position((50.0, 0.0), (101.0, 101.0)), [0.0, 0.0, 0.0], atol=1e-12)


def test_hue_lightness_slice_accepts_normalized_achromatic_colour():
    model = HueLightnessSliceModel(chroma=0.0)
    colour = normalize_oklab_for_krita(color_math.oklch_to_oklab([0.5, 0.0, 0.0]))

    assert model.indicator_for_intent(color_math.oklab_to_oklch(colour), (101.0, 101.0)) is not None


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


@pytest.mark.parametrize("position", [(65.0, 50.0), (50.0, 25.0), (35.0, 50.0)])
def test_hue_lightness_slice_round_trips_position_and_color(position):
    chroma = 0.03
    model = HueLightnessSliceModel(chroma=chroma)

    color = model.color_at_position(position, (101.0, 101.0))
    assert color is not None
    actual = model.position_for_intent(color_math.oklab_to_oklch(color), (101.0, 101.0))

    np.testing.assert_allclose(actual, position, atol=1e-9)


def test_hue_lightness_slice_rejects_color_with_mismatched_chroma():
    model = HueLightnessSliceModel(chroma=0.05)
    color = color_math.oklch_to_oklab([0.5, 0.06, math.pi / 2.0])

    assert model.position_for_intent(color_math.oklab_to_oklch(color), (101.0, 101.0)) is None


# -- fallback indicator helpers --------------------------------------------


def test_selector_model_default_indicator_uses_position_for_intent():
    class LinearSelectorModel(SelectorModel):
        def color_at_position(self, position, size):
            return np.array([0.5, 0.0, 0.0])

        def selection_at_position(self, position, size):
            return SelectorSelection((0.5, 0.0, 0.0), (float(position[0]), float(position[1])))

        def colors_at_positions(self, x, y, size):
            return np.zeros(np.asarray(x).shape + (3,), dtype=float), np.ones(np.asarray(x).shape, dtype=bool)

        def position_for_intent(self, lch, size):
            return 2.0, 3.0

    indicator = LinearSelectorModel().indicator_for_intent((0.5, 0.0, 0.0), (10.0, 10.0))
    assert indicator == IndicatorSpec(desired=(2.0, 3.0))
    assert indicator.snapped is None
    assert indicator.out_of_gamut is False


def test_selector_model_default_snap_returns_none():
    class LinearSelectorModel(SelectorModel):
        def color_at_position(self, position, size):
            return np.array([0.5, 0.0, 0.0])

        def selection_at_position(self, position, size):
            return SelectorSelection((0.5, 0.0, 0.0), (float(position[0]), float(position[1])))

        def colors_at_positions(self, x, y, size):
            return np.zeros(np.asarray(x).shape + (3,), dtype=float), np.ones(np.asarray(x).shape, dtype=bool)

        def position_for_intent(self, lch, size):
            return 2.0, 3.0

    assert LinearSelectorModel().snapped_color_at_position((1.0, 1.0), (10.0, 10.0)) is None


def test_lightness_slice_indicator_returns_out_of_leaf_location():
    # At L=0.5, hue=0 the cusp chroma is below SRGB_MAX_CHROMA, so a colour at
    # SRGB_MAX_CHROMA is OOG on this slice and position_for_intent returns None.
    model = LightnessSliceModel(lightness=0.5)
    oklab = color_math.oklch_to_oklab([0.5, color_math.SRGB_MAX_CHROMA, 0.0])
    assert model.position_for_intent(color_math.oklab_to_oklch(oklab), (101.0, 101.0)) is None

    indicator = model.indicator_for_intent(color_math.oklab_to_oklch(oklab), (101.0, 101.0))
    assert indicator is not None
    assert indicator.out_of_gamut is True
    # Hue=0 lands on the +x rim of the disk.
    np.testing.assert_allclose(indicator.desired, (100.0, 50.0), atol=1e-9)


def test_lightness_slice_indicator_snapped_position_clamps_chroma_to_leaf():
    model = LightnessSliceModel(lightness=0.5)
    oklab = color_math.oklch_to_oklab([0.5, color_math.SRGB_MAX_CHROMA, 0.0])
    indicator = model.indicator_for_intent(color_math.oklab_to_oklch(oklab), (101.0, 101.0))
    assert indicator is not None
    assert indicator.snapped is not None
    # The snapped position must be strictly inside the rim, since the cusp
    # chroma at L=0.5, hue=0 is well below SRGB_MAX_CHROMA.
    assert 50.0 < indicator.snapped[0] < 100.0
    assert indicator.snapped[1] == pytest.approx(50.0, abs=1e-9)


def test_lightness_chroma_slice_desired_and_snapped_for_oog_chroma():
    model = LightnessChromaSliceModel(hue=0.0)
    oklab = color_math.oklch_to_oklab([0.5, color_math.SRGB_MAX_CHROMA, 0.0])
    assert model.position_for_intent(color_math.oklab_to_oklch(oklab), (101.0, 101.0)) is None

    indicator = model.indicator_for_intent(color_math.oklab_to_oklch(oklab), (101.0, 101.0))
    assert indicator is not None
    assert indicator.out_of_gamut is True
    np.testing.assert_allclose(indicator.desired, (100.0, 50.0), atol=1e-9)

    assert indicator.snapped is not None
    assert 0.0 <= indicator.snapped[0] < 100.0
    assert indicator.snapped[1] == pytest.approx(50.0, abs=1e-9)


def test_hue_lightness_slice_snapped_position_pulls_back_into_gamut():
    # At chroma=0.2, hue=0 the gamut leaf is narrow in L; extreme L values are
    # OOG so position_for_intent returns None and snapped pulls the marker into
    # the valid lightness band.
    chroma = 0.2
    hue = 0.0
    model = HueLightnessSliceModel(chroma=chroma)
    oklab = color_math.oklch_to_oklab([0.05, chroma, hue])
    assert model.position_for_intent(color_math.oklab_to_oklch(oklab), (101.0, 101.0)) is None

    indicator = model.indicator_for_intent(color_math.oklab_to_oklch(oklab), (101.0, 101.0))
    assert indicator is not None
    assert indicator.out_of_gamut is True
    assert indicator.snapped is not None
    # Hue=0 puts the marker along +x from centre. The snapped lightness must be
    # closer to the centre than the requested L=0.05 (which would lie near the
    # rim at the equivalent normalized radius = 0.95).
    cx, cy = 50.0, 50.0
    snapped_radius = math.hypot(indicator.snapped[0] - cx, cy - indicator.snapped[1])
    desired_radius = (1.0 - 0.05) * 50.0
    assert snapped_radius < desired_radius


def test_lightness_slice_helpers_reject_mismatched_lightness():
    model = LightnessSliceModel(lightness=0.5)
    # Colour sits on a different lightness slice; both helpers must reject it
    # so the widget does not paint a stale indicator on the wrong slice.
    oklab = color_math.oklch_to_oklab([0.2, 0.05, 0.0])
    assert model.indicator_for_intent(color_math.oklab_to_oklch(oklab), (101.0, 101.0)) is None


def test_hue_lightness_slice_helpers_reject_mismatched_chroma():
    model = HueLightnessSliceModel(chroma=0.05)
    # Colour at a different chroma than this fixed-chroma slice.
    oklab = color_math.oklch_to_oklab([0.5, 0.10, 0.0])
    assert model.indicator_for_intent(color_math.oklab_to_oklch(oklab), (101.0, 101.0)) is None


def test_lightness_chroma_slice_helpers_reject_mismatched_hue():
    model = LightnessChromaSliceModel(hue=0.0)
    # Colour on a perpendicular hue plane.
    oklab = color_math.oklch_to_oklab([0.5, 0.05, math.pi / 2.0])
    assert model.indicator_for_intent(color_math.oklab_to_oklch(oklab), (101.0, 101.0)) is None


def test_lightness_slice_indicator_contract_combines_desired_and_snapped_positions():
    model = LightnessSliceModel(lightness=0.5)
    oklab = color_math.oklch_to_oklab([0.5, color_math.SRGB_MAX_CHROMA, 0.0])

    indicator = model.indicator_for_intent(color_math.oklab_to_oklch(oklab), (101.0, 101.0))

    assert indicator is not None
    assert indicator.out_of_gamut is True
    np.testing.assert_allclose(indicator.desired, (100.0, 50.0), atol=1e-9)
    assert indicator.snapped is not None
    assert 50.0 < indicator.snapped[0] < 100.0


def test_indicator_contract_preserves_snapped_only_fallback():
    model = LightnessSliceModel(lightness=0.5)
    oklab = color_math.oklch_to_oklab([0.5, color_math.SRGB_MAX_CHROMA * 1.5, 0.0])
    assert model.position_for_intent(color_math.oklab_to_oklch(oklab), (101.0, 101.0)) is None

    indicator = model.indicator_for_intent(color_math.oklab_to_oklch(oklab), (101.0, 101.0))

    assert indicator is not None
    assert indicator.snapped is None
    assert indicator.out_of_gamut is False
    assert 50.0 < indicator.desired[0] < 100.0
    assert indicator.desired[1] == pytest.approx(50.0, abs=1e-9)
