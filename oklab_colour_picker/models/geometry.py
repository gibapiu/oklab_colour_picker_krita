"""Shared geometry helpers for pure selector models and disk widgets."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from oklab_colour_picker.models.base import Position


POSITION_EPSILON = 1e-12


def circle_geometry(position: Sequence[float], size: Sequence[float]):
    return _circle_geometry_core(position, size, project=False)


def circle_geometry_projected(position: Sequence[float], size: Sequence[float]):
    geometry = _circle_geometry_core(position, size, project=True)
    if geometry is None:
        return None
    normalized_radius, hue, _, _ = geometry
    return normalized_radius, hue


def circle_geometry_arrays(x, y, size: Sequence[float]):
    bounds = position_in_bounds_arrays(x, y, size)
    if bounds is None:
        return None

    x, y, width, height, bounds_valid = bounds
    geometry = disk_geometry((width, height))
    if geometry is None:
        return None

    center_x, center_y, radius = geometry
    dx = x - center_x
    dy = center_y - y
    distance = np.hypot(dx, dy)
    circle_valid = bounds_valid & (distance <= radius + POSITION_EPSILON)
    normalized_radius = np.minimum(distance / radius, 1.0)
    hue = np.where(distance <= POSITION_EPSILON, 0.0, np.mod(np.arctan2(dy, dx), math.tau))
    return normalized_radius, hue, circle_valid


def rect_geometry_projected(position: Sequence[float], size: Sequence[float]):
    bounds = size_bounds(size)
    if bounds is None:
        return None

    width, height = bounds
    x = float(np.clip(float(position[0]), 0.0, width - 1.0))
    y = float(np.clip(float(position[1]), 0.0, height - 1.0))
    return x, y, width, height


def position_from_circle(
    normalized_radius: float, hue: float, size: Sequence[float]
) -> Position | None:
    geometry = disk_geometry(size)
    if geometry is None:
        return None

    center_x, center_y, radius = geometry
    distance = normalized_radius * radius
    return (
        center_x + distance * math.cos(hue),
        center_y - distance * math.sin(hue),
    )


def disk_geometry(size: Sequence[float]) -> tuple[float, float, float] | None:
    bounds = size_bounds(size)
    if bounds is None:
        return None

    width, height = bounds
    radius = (min(width, height) - 1.0) / 2.0
    if radius <= 0.0:
        return None
    center_x = (width - 1.0) / 2.0
    center_y = (height - 1.0) / 2.0
    return center_x, center_y, radius


def position_in_bounds(position: Sequence[float], size: Sequence[float]):
    bounds = size_bounds(size)
    if bounds is None:
        return None

    width, height = bounds
    x, y = (float(position[0]), float(position[1]))
    if not (0.0 <= x <= width - 1.0 and 0.0 <= y <= height - 1.0):
        return None
    return x, y, width, height


def position_in_bounds_arrays(x, y, size: Sequence[float]):
    bounds = size_bounds(size)
    if bounds is None:
        return None

    width, height = bounds
    x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    valid = (0.0 <= x) & (x <= width - 1.0) & (0.0 <= y) & (y <= height - 1.0)
    return x, y, width, height, valid


def size_bounds(size: Sequence[float]):
    width, height = (float(size[0]), float(size[1]))
    if width <= 1.0 or height <= 1.0:
        return None
    return width, height


def empty_color_grid(x):
    return np.zeros(np.asarray(x).shape + (3,), dtype=float)


def _circle_geometry_core(position: Sequence[float], size: Sequence[float], *, project: bool):
    bounds = size_bounds(size)
    if bounds is None:
        return None

    width, height = bounds
    x, y = (float(position[0]), float(position[1]))
    if not project and not (0.0 <= x <= width - 1.0 and 0.0 <= y <= height - 1.0):
        return None

    geometry = disk_geometry((width, height))
    if geometry is None:
        return None

    center_x, center_y, radius = geometry
    dx = x - center_x
    dy = center_y - y
    distance = math.hypot(dx, dy)
    if not project and distance > radius + POSITION_EPSILON:
        return None

    hue = 0.0 if distance <= POSITION_EPSILON else math.atan2(dy, dx) % math.tau
    return min(distance / radius, 1.0), hue, center_x, radius
