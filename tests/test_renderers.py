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


def test_gamut_contour_preserves_hole_and_separate_island():
    mask = np.zeros((5, 7), dtype=bool)
    mask[1:4, 1:4] = True
    mask[2, 2] = False
    mask[2, 5] = True
    segments = renderers.gamut_contour_segments(mask)
    edges = {tuple(map(tuple, segment)) for segment in segments}
    assert len(segments) == len(edges) == 20
    assert {
        ((1.5, 1.5), (1.5, 2.5)),
        ((2.5, 1.5), (2.5, 2.5)),
        ((1.5, 1.5), (2.5, 1.5)),
        ((1.5, 2.5), (2.5, 2.5)),
    } <= edges
    assert {
        ((4.5, 1.5), (4.5, 2.5)),
        ((5.5, 1.5), (5.5, 2.5)),
        ((4.5, 1.5), (5.5, 1.5)),
        ((4.5, 2.5), (5.5, 2.5)),
    } <= edges


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
    assert rgba.shape == (size[1], size[0], 4)
    assert rgba.dtype == np.uint8

    for x, y in probes:
        model_color = model.color_at_position((x, y), size)
        if model_color is None:
            if 0 <= x < size[0] and 0 <= y < size[1]:
                assert rgba[y, x, 3] == 0
            continue

        assert rgba[y, x, 3] == 255
        np.testing.assert_array_equal(rgba[y, x, :3], color_math.oklab_to_srgb8(model_color))


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
