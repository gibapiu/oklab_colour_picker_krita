"""NumPy-backed RGBA renderers for selector models."""

from __future__ import annotations

from functools import lru_cache
from typing import Protocol, Sequence

import numpy as np
import numpy.typing as npt

from oklab_colour_picker.domain import color_math


_RENDER_CACHE_MAXSIZE = 128


class VectorizedSelectorModel(Protocol):
    def colors_at_positions(
        self,
        x: npt.ArrayLike,
        y: npt.ArrayLike,
        size: Sequence[float],
    ) -> tuple[np.ndarray, np.ndarray]:
        ...


def render_rgba(model: VectorizedSelectorModel, size: Sequence[int]) -> np.ndarray:
    """Render ``model`` to a ``(height, width, 4)`` uint8 RGBA buffer."""

    width, height = _validate_size(size)
    return _render_rgba_cached(model, width, height).copy()


@lru_cache(maxsize=_RENDER_CACHE_MAXSIZE)
def _render_rgba_cached(model: VectorizedSelectorModel, width: int, height: int) -> np.ndarray:
    x, y = _pixel_grid(width, height)
    oklab, selectable = model.colors_at_positions(x, y, (width, height))
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[..., :3] = color_math.oklab_to_srgb8(oklab)
    rgba[..., 3] = np.where(selectable, 255, 0).astype(np.uint8)
    rgba.setflags(write=False)
    return rgba


@lru_cache(maxsize=16)
def _pixel_grid(width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    y, x = np.indices((height, width), dtype=float)
    return x, y


def _validate_size(size: Sequence[int]) -> tuple[int, int]:
    width, height = int(size[0]), int(size[1])
    if width <= 1 or height <= 1:
        raise ValueError("renderer size must be at least 2x2")
    return width, height


# Axis identifiers for OKLCh slider tracks.
AXIS_L = "L"
AXIS_C = "C"
AXIS_H = "H"

# Checkerboard tile size (px) for the out-of-gamut indicator inside slider
# tracks. Small enough to read as a pattern at slider heights of ~14 px.
_GAMUT_GAP_TILE = 4
_GAMUT_GAP_LIGHT = np.array([200, 200, 200, 255], dtype=np.uint8)
_GAMUT_GAP_DARK = np.array([120, 120, 120, 255], dtype=np.uint8)


def render_axis_track(
    axis: str,
    fixed: tuple[float, float],
    chroma_max: float,
    size: Sequence[int],
    *,
    hue_chroma_floor: float = 0.0,
) -> np.ndarray:
    """Render an OKLCh axis gradient bar with out-of-gamut regions hatched.

    ``axis`` is one of ``AXIS_L``, ``AXIS_C``, ``AXIS_H``. ``fixed`` carries
    the other two OKLCh components in OKLCh order (omitting the swept axis):

    - ``AXIS_L``: ``(chroma, hue)``
    - ``AXIS_C``: ``(lightness, hue)``
    - ``AXIS_H``: ``(lightness, chroma)``

    ``chroma_max`` is the C-axis full-scale value (typically
    ``color_math.SRGB_MAX_CHROMA``); ignored for L and H axes.

    ``hue_chroma_floor`` applies only to ``AXIS_H``: when the current chroma is
    below this floor, the rail is *rendered* at the floor so the hue sweep
    stays colourful for near-neutral colours. Gamut classification still uses
    the real fixed chroma, so the out-of-gamut checker reflects the actual
    selected colour, not the rendering floor.

    Returns a ``(height, width, 4)`` uint8 RGBA buffer.
    """

    width, height = _validate_size(size)
    swept = _swept_values(axis, width, chroma_max)
    oklch = _oklch_grid(axis, fixed, swept, height, width)
    if axis == AXIS_H and hue_chroma_floor > float(fixed[1]):
        display_oklch = _oklch_grid(
            axis, (float(fixed[0]), float(hue_chroma_floor)), swept, height, width
        )
        in_gamut, _ = _classify_track_pixels(oklch)
        _, srgb8 = _classify_track_pixels(display_oklch)
    else:
        in_gamut, srgb8 = _classify_track_pixels(oklch)
    return _compose_track_rgba(srgb8, in_gamut, width, height)


def _swept_values(axis: str, width: int, chroma_max: float) -> np.ndarray:
    fraction = np.linspace(0.0, 1.0, width)
    if axis == AXIS_L:
        return fraction
    if axis == AXIS_C:
        return fraction * float(chroma_max)
    if axis == AXIS_H:
        return fraction * (2.0 * np.pi)
    raise ValueError(f"unknown axis: {axis!r}")


def _oklch_grid(
    axis: str,
    fixed: tuple[float, float],
    swept: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    # Build the per-column OKLCh triple, then broadcast to (height, width, 3).
    if axis == AXIS_L:
        chroma, hue = float(fixed[0]), float(fixed[1])
        lightness = swept
        oklch_row = np.stack(
            (lightness, np.full_like(swept, chroma), np.full_like(swept, hue)),
            axis=-1,
        )
    elif axis == AXIS_C:
        lightness, hue = float(fixed[0]), float(fixed[1])
        oklch_row = np.stack(
            (np.full_like(swept, lightness), swept, np.full_like(swept, hue)),
            axis=-1,
        )
    elif axis == AXIS_H:
        lightness, chroma = float(fixed[0]), float(fixed[1])
        oklch_row = np.stack(
            (np.full_like(swept, lightness), np.full_like(swept, chroma), swept),
            axis=-1,
        )
    else:
        raise ValueError(f"unknown axis: {axis!r}")
    return np.broadcast_to(oklch_row[None, :, :], (height, width, 3))


def _classify_track_pixels(oklch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    oklab = color_math.oklch_to_oklab(oklch)
    srgb = color_math.oklab_to_srgb(oklab)
    in_gamut = color_math.in_srgb_gamut(srgb, epsilon=1e-6)
    return in_gamut, color_math.quantize_srgb8(srgb)


def _compose_track_rgba(
    srgb8: np.ndarray, in_gamut: np.ndarray, width: int, height: int
) -> np.ndarray:
    pattern = _checker_tile(width, height)
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[..., :3] = srgb8
    rgba[..., 3] = 255
    out_of_gamut = ~in_gamut
    if np.any(out_of_gamut):
        rgba[out_of_gamut] = np.where(
            pattern[out_of_gamut][..., None],
            _GAMUT_GAP_LIGHT,
            _GAMUT_GAP_DARK,
        )
    return rgba


def _checker_tile(width: int, height: int) -> np.ndarray:
    y, x = np.indices((height, width))
    return ((x // _GAMUT_GAP_TILE) + (y // _GAMUT_GAP_TILE)) % 2 == 0
