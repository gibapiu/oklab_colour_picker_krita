import math

import numpy as np
import pytest

import oklab_colour_picker.domain.color_math as color_math


def test_srgb_oklab_known_reference_values():
    # Reference primary values use Bjorn Ottosson's OKLab matrices:
    # https://bottosson.github.io/posts/oklab/
    samples = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    actual = color_math.srgb_to_oklab(samples)

    np.testing.assert_allclose(actual[0], [0.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(actual[1], [1.0, 0.0, 0.0], atol=1e-7)
    np.testing.assert_allclose(actual[2], [0.62795536, 0.22486306, 0.12584630], atol=1e-7)
    np.testing.assert_allclose(actual[3], [0.86643961, -0.23388757, 0.17949848], atol=1e-7)
    np.testing.assert_allclose(actual[4], [0.45201372, -0.03245698, -0.31152815], atol=1e-7)


def test_srgb_oklab_round_trip_for_scalar_and_vector_inputs():
    samples = np.array(
        [
            [0.12, 0.34, 0.56],
            [0.90, 0.20, 0.10],
            [0.04, 0.95, 0.70],
        ]
    )

    round_tripped = color_math.oklab_to_srgb(color_math.srgb_to_oklab(samples))
    scalar_round_tripped = color_math.oklab_to_srgb(color_math.srgb_to_oklab(samples[0]))

    np.testing.assert_allclose(round_tripped, samples, atol=1e-12)
    np.testing.assert_allclose(scalar_round_tripped, samples[0], atol=1e-12)


def test_neutral_srgb_stays_neutral_in_oklab():
    values = np.linspace(-1.0, 2.0, 17)
    samples = np.stack((values, values, values), axis=-1)

    lab = color_math.srgb_to_oklab(samples)
    chroma = np.hypot(lab[:, 1], lab[:, 2])

    np.testing.assert_allclose(chroma, np.zeros_like(chroma), atol=1.2e-7)


def test_oklab_oklch_round_trip_preserves_hue_and_chroma():
    lab = np.array(
        [
            [0.5, 0.1, 0.0],
            [0.7, -0.05, 0.08],
            [0.3, 0.0, -0.2],
        ]
    )

    lch = color_math.oklab_to_oklch(lab)
    actual = color_math.oklch_to_oklab(lch)

    assert np.all((0.0 <= lch[:, 2]) & (lch[:, 2] < math.tau))
    np.testing.assert_allclose(actual, lab, atol=1e-12)


def test_gamut_predicate_and_clip_work_for_scalar_and_vector_inputs():
    srgb = np.array(
        [
            [0.0, 0.5, 1.0],
            [-0.01, 0.5, 1.0],
            [0.0, 0.5, 1.01],
        ]
    )

    np.testing.assert_array_equal(color_math.in_srgb_gamut(srgb), [True, False, False])
    assert color_math.in_srgb_gamut(np.array([0.0, 0.5, 1.0])) is True
    np.testing.assert_allclose(
        color_math.clip_srgb(srgb),
        [[0.0, 0.5, 1.0], [0.0, 0.5, 1.0], [0.0, 0.5, 1.0]],
    )


def test_oklab_to_srgb_returns_unclipped_out_of_gamut_values():
    srgb = color_math.oklab_to_srgb(np.array([0.5, 1.0, 0.0]))

    assert np.any(srgb < 0.0) or np.any(srgb > 1.0)
    assert color_math.in_srgb_gamut(srgb) is False


@pytest.mark.parametrize("hue", [0.0, 1.0, 2.0, 4.0, 5.5])
def test_find_cusp_lands_on_srgb_gamut_boundary(hue):
    a_ = math.cos(hue)
    b_ = math.sin(hue)
    lightness, chroma = color_math.find_cusp(a_, b_)

    srgb = color_math.oklab_to_srgb(np.array([lightness, chroma * a_, chroma * b_]))
    just_outside = color_math.oklab_to_srgb(np.array([lightness, chroma * 1.001 * a_, chroma * 1.001 * b_]))

    assert color_math.in_srgb_gamut(srgb, epsilon=1e-6) is True
    assert color_math.in_srgb_gamut(just_outside, epsilon=1e-6) is False
