import math

import numpy as np
import pytest

from oklab_colour_picker.domain import color_math
from oklab_colour_picker.render import renderers
from oklab_colour_picker.render.renderers import render_rgba
from oklab_colour_picker.models import (
    LightnessChromaSliceModel,
    HueLightnessSliceModel,
    LightnessSliceModel,
)


@pytest.mark.parametrize(
    "model",
    [
        LightnessSliceModel(lightness=0.55),
        HueLightnessSliceModel(chroma=0.05),
        LightnessChromaSliceModel(hue=1.25),
    ],
)
def test_renderers_return_uint8_rgba_buffers(model):
    actual = render_rgba(model, (23, 17))

    assert actual.shape == (17, 23, 4)
    assert actual.dtype == np.uint8


@pytest.mark.parametrize("size", [(1, 10), (10, 1)])
def test_renderers_reject_degenerate_sizes(size):
    with pytest.raises(ValueError, match="at least 2x2"):
        render_rgba(LightnessSliceModel(lightness=0.55), size)


def test_render_rgba_returns_mutable_copy_without_corrupting_cache():
    model = LightnessSliceModel(lightness=0.55)
    original = render_rgba(model, (17, 17))

    original[:, :, :] = 0
    actual = render_rgba(model, (17, 17))

    assert np.count_nonzero(actual[..., 3]) > 0


@pytest.mark.parametrize(
    ("model", "size", "probes"),
    [
        (
            LightnessSliceModel(lightness=0.55),
            (33, 33),
            [(16, 16), (32, 16), (16, 0), (0, 0)],
        ),
        (
            HueLightnessSliceModel(chroma=0.05),
            (33, 33),
            [(24, 16), (16, 8), (16, 16), (0, 0), (33, 16)],
        ),
        (
            LightnessChromaSliceModel(hue=1.25),
            (33, 21),
            [(0, 0), (16, 10), (32, 20), (33, 10)],
        ),
    ],
)
def test_renderer_pixels_match_model_at_probe_points(model, size, probes):
    rgba = render_rgba(model, size)

    for x, y in probes:
        model_color = model.color_at_position((x, y), size)
        if model_color is None:
            if 0 <= x < size[0] and 0 <= y < size[1]:
                assert rgba[y, x, 3] == 0
            continue

        assert rgba[y, x, 3] == 255
        np.testing.assert_array_equal(rgba[y, x, :3], _quantize8(model_color))


@pytest.mark.parametrize("size", [(64, 64), (200, 120)])
def test_lightness_renderer_preserves_coordinate_semantics_across_sizes(size):
    model = LightnessSliceModel(lightness=0.5)
    rgba = render_rgba(model, size)
    position = model.position_for_intent((0.5, 0.0, 0.0), size)
    x, y = round(position[0]), round(position[1])

    assert rgba[y, x, 3] == 255
    np.testing.assert_array_equal(rgba[y, x, :3], _quantize8(model.color_at_position((x, y), size)))


def test_lightness_chroma_slice_renderer_alpha_marks_per_hue_gamut():
    model = LightnessChromaSliceModel(hue=math.pi / 3.0)
    rgba = render_rgba(model, (101, 101))

    # The left edge (chroma=0) is always in gamut; the right edge sits at the
    # global max chroma which exceeds the per-hue cusp for almost every row.
    assert np.all(rgba[:, 0, 3] == 255)
    assert np.count_nonzero(rgba[..., 3] == 0) > 0
    assert np.count_nonzero(rgba[..., 3] == 255) > 0


def test_hue_lightness_slice_renderer_alpha_marks_fixed_chroma_gamut():
    model = HueLightnessSliceModel(chroma=0.15)
    rgba = render_rgba(model, (101, 101))

    assert np.count_nonzero(rgba[..., 3] == 0) > 0
    assert np.count_nonzero(rgba[..., 3] == 255) > 0


def test_axis_track_hue_marks_out_of_gamut_with_checker():
    rgba = renderers.render_axis_track(
        renderers.AXIS_H,
        (0.95, color_math.SRGB_MAX_CHROMA * 0.9),
        color_math.SRGB_MAX_CHROMA,
        (256, 12),
    )

    assert np.any(np.all(rgba[..., :3] == 120, axis=-1))


def test_axis_track_chroma_starts_in_gamut_at_zero_chroma():
    rgba = renderers.render_axis_track(
        renderers.AXIS_C,
        (0.5, 0.0),
        color_math.SRGB_MAX_CHROMA,
        (256, 12),
    )
    left = rgba[:, 0, :3]

    assert not np.any(np.all(left == 120, axis=-1))
    assert not np.any(np.all(left == 200, axis=-1))


def test_axis_track_lightness_extremes_are_out_of_gamut_for_chroma():
    rgba = renderers.render_axis_track(
        renderers.AXIS_L,
        (0.15, 0.0),
        color_math.SRGB_MAX_CHROMA,
        (256, 12),
    )

    for column in (0, -1):
        assert tuple(rgba[0, column, :3]) in {
            (120, 120, 120),
            (200, 200, 200),
        }


def test_axis_track_rejects_unknown_axis():
    with pytest.raises(ValueError):
        renderers.render_axis_track(
            "Q",
            (0.5, 0.0),
            color_math.SRGB_MAX_CHROMA,
            (32, 10),
        )


def test_axis_track_hue_chroma_floor_lifts_neutral_colours():
    flat = renderers.render_axis_track(
        renderers.AXIS_H,
        (0.5, 0.0),
        color_math.SRGB_MAX_CHROMA,
        (64, 8),
    )
    floored = renderers.render_axis_track(
        renderers.AXIS_H,
        (0.5, 0.0),
        color_math.SRGB_MAX_CHROMA,
        (64, 8),
        hue_chroma_floor=0.08,
    )

    assert np.all(flat[..., :3] == flat[0, 0, :3])
    assert len({tuple(floored[0, x, :3]) for x in range(floored.shape[1])}) > 8


def test_axis_track_hue_chroma_floor_preserves_gamut_classification():
    rgba = renderers.render_axis_track(
        renderers.AXIS_H,
        (0.5, color_math.SRGB_MAX_CHROMA * 0.95),
        color_math.SRGB_MAX_CHROMA,
        (256, 12),
        hue_chroma_floor=0.001,
    )

    assert np.any(np.all(rgba[..., :3] == 120, axis=-1))


def _quantize8(oklab):
    srgb = color_math.clip_srgb(color_math.oklab_to_srgb(oklab))
    return np.rint(srgb * 255.0).astype(np.uint8)
