"""Base contracts for pure selector models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import numpy.typing as npt

from oklab_colour_picker.domain import color_math


Position = tuple[float, float]
Size = tuple[float, float]
OKLCh = tuple[float, float, float]


@dataclass(frozen=True)
class SelectorSelection:
    lch: OKLCh
    position: Position

    @property
    def paint_oklab(self) -> np.ndarray:
        return color_math.oklch_to_oklab(self.lch)


class SelectorModel(ABC):
    """Coordinate and gamut contracts for OKLCh intent and OKLab output."""

    @abstractmethod
    def color_at_position(self, position: Sequence[float], size: Sequence[float]) -> np.ndarray | None:
        """Return the selectable OKLab colour at ``position`` or ``None``."""

    @abstractmethod
    def selection_at_position(
        self, position: Sequence[float], size: Sequence[float]
    ) -> SelectorSelection | None:
        """Return the selectable OKLCh selection at ``position`` or ``None``."""

    @abstractmethod
    def colors_at_positions(
        self,
        x: npt.ArrayLike,
        y: npt.ArrayLike,
        size: Sequence[float],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return vectorized OKLab colours and a selectable mask."""

    @abstractmethod
    def position_for_intent(self, lch: OKLCh, size: Sequence[float]) -> Position | None:
        """Return the in-gamut selector position for ``lch`` or ``None``."""

    @abstractmethod
    def geometric_position_for_intent(
        self, lch: OKLCh, size: Sequence[float]
    ) -> Position | None:
        """Return where ``lch`` belongs on this slice, ignoring gamut clamping."""

    @abstractmethod
    def project_onto_slice(self, lch: OKLCh) -> OKLCh | None:
        """Project an out-of-gamut ``lch`` onto this slice's in-gamut leaf."""

    @abstractmethod
    def snapped_selector_selection_at_position(
        self, position: Sequence[float], size: Sequence[float]
    ) -> SelectorSelection | None:
        """Return a snapped selection, or None when no valid snap is available."""


def positions_close(a: Position, b: Position) -> bool:
    return abs(a[0] - b[0]) <= 0.5 and abs(a[1] - b[1]) <= 0.5
