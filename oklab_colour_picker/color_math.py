"""Pure OKLab, OKLCh, sRGB, and gamut-boundary helpers."""

from __future__ import annotations

import math

import numpy as np


SRGB_GAMMA_THRESHOLD = 0.04045
LINEAR_SRGB_GAMMA_THRESHOLD = 0.0031308
ACHROMATIC_CHROMA_EPSILON = 1e-7
MAX_SATURATION_HALLEY_ITERATIONS = 3
GAMUT_INTERSECTION_HALLEY_ITERATIONS = 3

_LINEAR_SRGB_TO_LMS = np.array(
    [
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005],
    ]
)
_LMS_TO_OKLAB = np.array(
    [
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660],
    ]
)
_OKLAB_TO_LMS = np.linalg.inv(_LMS_TO_OKLAB)
_LMS3_TO_LINEAR_SRGB = np.linalg.inv(_LINEAR_SRGB_TO_LMS)

_MAX_SATURATION_POLY_COEFFICIENTS = np.array(
    [
        [
            1.19086277,
            1.76576728,
            0.59662641,
            0.75515197,
            0.56771245,
        ],
        [
            0.73956515,
            -0.45954404,
            0.08285427,
            0.12541070,
            0.14503204,
        ],
        [
            1.35733652,
            -0.00915799,
            -1.15130210,
            -0.50559606,
            0.00692167,
        ],
    ]
)


def srgb_to_linear(srgb):
    """Convert gamma-encoded sRGB component values in [0, 1] to linear sRGB."""

    srgb = np.asarray(srgb, dtype=float)
    with np.errstate(invalid="ignore"):
        return np.where(
            srgb <= SRGB_GAMMA_THRESHOLD,
            srgb / 12.92,
            np.power((srgb + 0.055) / 1.055, 2.4),
        )


def linear_to_srgb(linear):
    """Convert linear sRGB component values to unclipped gamma-encoded sRGB."""

    linear = np.asarray(linear, dtype=float)
    return np.where(
        linear <= LINEAR_SRGB_GAMMA_THRESHOLD,
        12.92 * linear,
        1.055 * np.power(np.maximum(linear, 0.0), 1.0 / 2.4) - 0.055,
    )


def srgb_to_oklab(srgb):
    """Convert gamma-encoded sRGB triples to OKLab triples."""

    linear = srgb_to_linear(srgb)
    return linear_srgb_to_oklab(linear)


def oklab_to_srgb(oklab):
    """Convert OKLab triples to unclipped gamma-encoded sRGB triples."""

    linear = oklab_to_linear_srgb(oklab)
    return linear_to_srgb(linear)


def linear_srgb_to_oklab(linear_srgb):
    """Convert linear sRGB triples to OKLab triples."""

    linear_srgb = np.asarray(linear_srgb, dtype=float)
    lms = linear_srgb @ _LINEAR_SRGB_TO_LMS.T
    return np.cbrt(lms) @ _LMS_TO_OKLAB.T


def oklab_to_linear_srgb(oklab):
    """Convert OKLab triples to linear sRGB triples."""

    oklab = np.asarray(oklab, dtype=float)
    lms = oklab @ _OKLAB_TO_LMS.T
    return (lms * lms * lms) @ _LMS3_TO_LINEAR_SRGB.T


def oklab_to_oklch(oklab):
    """Convert OKLab triples to OKLCh triples with hue in radians [0, tau)."""

    oklab = np.asarray(oklab, dtype=float)
    lightness = oklab[..., 0]
    a = oklab[..., 1]
    b = oklab[..., 2]
    chroma = np.hypot(a, b)
    hue = np.mod(np.arctan2(b, a), math.tau)
    return np.stack((lightness, chroma, hue), axis=-1)


def oklch_to_oklab(oklch):
    """Convert OKLCh triples with hue in radians to OKLab triples."""

    oklch = np.asarray(oklch, dtype=float)
    lightness = oklch[..., 0]
    chroma = oklch[..., 1]
    hue = oklch[..., 2]
    return np.stack((lightness, chroma * np.cos(hue), chroma * np.sin(hue)), axis=-1)


def is_achromatic_chroma(chroma: float) -> bool:
    """Return whether chroma is at or below the 1e-7 achromatic epsilon."""

    return float(chroma) <= ACHROMATIC_CHROMA_EPSILON


def in_srgb_gamut(srgb, *, epsilon=0.0):
    """Return whether gamma-encoded sRGB values are inside the display gamut."""

    srgb = np.asarray(srgb, dtype=float)
    return _bool_if_scalar(np.all((srgb >= -epsilon) & (srgb <= 1.0 + epsilon), axis=-1))


def clip_srgb(srgb):
    """Clip gamma-encoded sRGB values to the display gamut."""

    return np.clip(np.asarray(srgb, dtype=float), 0.0, 1.0)


def quantize_srgb8(srgb):
    """Clip sRGB floats to [0, 1] and round to a uint8 array."""

    return np.rint(clip_srgb(srgb) * 255.0).astype(np.uint8)


def oklab_to_srgb8(oklab):
    """Quantize OKLab to 8-bit clipped sRGB; returns a uint8 array."""

    return quantize_srgb8(oklab_to_srgb(oklab))


def compute_max_saturation(a_, b_):
    """Return maximum OKLab saturation for a normalized hue direction."""

    a_, b_ = np.broadcast_arrays(
        np.asarray(a_, dtype=float),
        np.asarray(b_, dtype=float),
    )
    red = -1.88170328 * a_ - 0.80936493 * b_ > 1.0
    green = (1.81444104 * a_ - 1.19445276 * b_ > 1.0) & ~red
    region = np.where(red, 0, np.where(green, 1, 2))
    poly_coeffs = _MAX_SATURATION_POLY_COEFFICIENTS[region]
    k0, k1, k2, k3, k4 = np.moveaxis(poly_coeffs, -1, 0)
    wl, wm, ws = np.moveaxis(_LMS3_TO_LINEAR_SRGB[region], -1, 0)

    saturation = k0 + k1 * a_ + k2 * b_ + k3 * a_ * a_ + k4 * a_ * b_
    k_l, k_m, k_s = _oklab_hue_to_lms_coefficients(a_, b_)

    for _ in range(MAX_SATURATION_HALLEY_ITERATIONS):
        l_ = _OKLAB_TO_LMS[0, 0] + saturation * k_l
        m_ = _OKLAB_TO_LMS[1, 0] + saturation * k_m
        s_ = _OKLAB_TO_LMS[2, 0] + saturation * k_s
        l = l_ * l_ * l_
        m = m_ * m_ * m_
        s = s_ * s_ * s_
        l_d_s = 3.0 * k_l * l_ * l_
        m_d_s = 3.0 * k_m * m_ * m_
        s_d_s = 3.0 * k_s * s_ * s_
        l_d_s2 = 6.0 * k_l * k_l * l_
        m_d_s2 = 6.0 * k_m * k_m * m_
        s_d_s2 = 6.0 * k_s * k_s * s_

        f = wl * l + wm * m + ws * s
        f1 = wl * l_d_s + wm * m_d_s + ws * s_d_s
        f2 = wl * l_d_s2 + wm * m_d_s2 + ws * s_d_s2
        saturation = saturation - f * f1 / (f1 * f1 - 0.5 * f * f2)
    return _scalar_if_scalar(saturation)


def find_cusp(a_, b_):
    """Return ``(L_cusp, C_cusp)`` for a normalized OKLab hue direction."""

    a_, b_ = np.broadcast_arrays(
        np.asarray(a_, dtype=float),
        np.asarray(b_, dtype=float),
    )
    saturation = compute_max_saturation(a_, b_)
    lab_at_max = np.stack((np.ones_like(a_), saturation * a_, saturation * b_), axis=-1)
    rgb_at_max = oklab_to_linear_srgb(lab_at_max)
    lightness = np.cbrt(1.0 / np.max(rgb_at_max, axis=-1))
    chroma = lightness * saturation
    return _scalar_if_scalar(lightness), _scalar_if_scalar(chroma)


def find_gamut_intersection(a_, b_, l1, c1, l0, cusp=None):
    """Find the first sRGB gamut intersection on an OKLab lightness/chroma ray.

    ``c1`` must be greater than zero. A zero-chroma ray has no chroma boundary
    to intersect.
    """

    a_, b_, l1, c1, l0 = np.broadcast_arrays(
        np.asarray(a_, dtype=float),
        np.asarray(b_, dtype=float),
        np.asarray(l1, dtype=float),
        np.asarray(c1, dtype=float),
        np.asarray(l0, dtype=float),
    )
    if np.any(c1 <= 0.0):
        raise ValueError("c1 must be greater than zero")

    if cusp is None:
        l_cusp, c_cusp = find_cusp(a_, b_)
    else:
        l_cusp, c_cusp = cusp

    l_cusp = np.asarray(l_cusp, dtype=float)
    c_cusp = np.asarray(c_cusp, dtype=float)
    lower_half = (l1 - l0) * c_cusp - (l_cusp - l0) * c1 <= 0.0

    lower_t = c_cusp * l0 / (c1 * l_cusp + c_cusp * (l0 - l1))
    upper_t = c_cusp * (l0 - 1.0) / (c1 * (l_cusp - 1.0) + c_cusp * (l0 - l1))
    t = np.where(lower_half, lower_t, upper_t)

    t = np.where(lower_half, t, _refine_upper_intersection_t(a_, b_, l1, c1, l0, t))
    return _scalar_if_scalar(np.maximum(t, 0.0))


def max_chroma_for_lh(lightness, hue):
    """Return the maximum in-gamut OKLCh chroma for lightness and hue radians."""

    lightness, hue = np.broadcast_arrays(
        np.asarray(lightness, dtype=float),
        np.asarray(hue, dtype=float),
    )
    a_ = np.cos(hue)
    b_ = np.sin(hue)
    cusp = find_cusp(a_, b_)
    chroma = find_gamut_intersection(a_, b_, lightness, np.ones_like(lightness), lightness, cusp=cusp)
    chroma = np.where((lightness <= 0.0) | (lightness >= 1.0), 0.0, chroma)
    return _scalar_if_scalar(chroma)


def _refine_upper_intersection_t(a_, b_, l1, c1, l0, t):
    d_l = l1 - l0
    d_c = c1
    k_l, k_m, k_s = _oklab_hue_to_lms_coefficients(a_, b_)

    l_dt = d_l + d_c * k_l
    m_dt = d_l + d_c * k_m
    s_dt = d_l + d_c * k_s

    for _ in range(GAMUT_INTERSECTION_HALLEY_ITERATIONS):
        l = l0 * (1.0 - t) + t * l1
        c = t * c1
        l_ = _OKLAB_TO_LMS[0, 0] * l + c * k_l
        m_ = _OKLAB_TO_LMS[1, 0] * l + c * k_m
        s_ = _OKLAB_TO_LMS[2, 0] * l + c * k_s

        r, t_r = _halley_channel_t(
            *_LMS3_TO_LINEAR_SRGB[0],
            l_,
            m_,
            s_,
            l_dt,
            m_dt,
            s_dt,
        )
        g, t_g = _halley_channel_t(
            *_LMS3_TO_LINEAR_SRGB[1],
            l_,
            m_,
            s_,
            l_dt,
            m_dt,
            s_dt,
        )
        b, t_b = _halley_channel_t(
            *_LMS3_TO_LINEAR_SRGB[2],
            l_,
            m_,
            s_,
            l_dt,
            m_dt,
            s_dt,
        )
        active_channel = np.argmax(np.stack((r, g, b), axis=0), axis=0)
        corrections = np.stack((t_r, t_g, t_b), axis=0)
        correction = np.take_along_axis(corrections, active_channel[None, ...], axis=0)[0]
        t = t + correction

    return t


def _halley_channel_t(wl, wm, ws, l_, m_, s_, l_dt, m_dt, s_dt):
    channel = wl * l_**3 + wm * m_**3 + ws * s_**3
    channel_dt = (
        3.0 * wl * l_dt * l_**2
        + 3.0 * wm * m_dt * m_**2
        + 3.0 * ws * s_dt * s_**2
    )
    channel_dt2 = (
        6.0 * wl * l_dt * l_dt * l_
        + 6.0 * wm * m_dt * m_dt * m_
        + 6.0 * ws * s_dt * s_dt * s_
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        u_channel = (1.0 - channel) / channel_dt
        correction = u_channel / (1.0 + 0.5 * channel_dt2 * u_channel / channel_dt)
    return channel, correction


def _scalar_if_scalar(value):
    value = np.asarray(value)
    if value.shape == ():
        return float(value)
    return value


def _bool_if_scalar(value):
    value = np.asarray(value)
    if value.shape == ():
        return bool(value)
    return value


def _oklab_hue_to_lms_coefficients(a_, b_):
    return (
        _OKLAB_TO_LMS[0, 1] * a_ + _OKLAB_TO_LMS[0, 2] * b_,
        _OKLAB_TO_LMS[1, 1] * a_ + _OKLAB_TO_LMS[1, 2] * b_,
        _OKLAB_TO_LMS[2, 1] * a_ + _OKLAB_TO_LMS[2, 2] * b_,
    )


# Rounded-up global sRGB OKLCh chroma extent. Keeping this as a literal avoids
# a dense 256x720 gamut-boundary sweep during plugin import while leaving a
# small margin above the sampled maximum (~0.3214).
SRGB_MAX_CHROMA = 0.325
