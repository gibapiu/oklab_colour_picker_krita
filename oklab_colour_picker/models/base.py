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
    """Pure coordinate contract shared by selector widgets and renderers.

    Position queries take canonical OKLCh ``(lightness, chroma, hue)`` - the
    UI's source of truth. OKLab is reserved for
    ``color_at_position`` outputs and the renderer pixel grid, where it is
    the natural representation. Querying with OKLCh removes the lossy
    ``oklab_to_oklch`` recovery that broke under Krita 8-bit normalisation.
    """

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

    def geometric_position_for_intent(
        self, lch: OKLCh, size: Sequence[float]
    ) -> Position | None:
        """Return where ``lch`` belongs on this slice, ignoring gamut clamping."""

        return self.position_for_intent(lch, size)

    def project_onto_slice(self, lch: OKLCh) -> OKLCh | None:
        """Project an out-of-gamut ``lch`` onto this slice's in-gamut leaf.

        The projection keeps the slice's fixed coordinate and clamps the free axes to the slice's gamut leaf,
        so the result stays on this plane and inside sRGB.
        """

        return None

    def snapped_color_at_position(
        self, position: Sequence[float], size: Sequence[float]
    ) -> np.ndarray | None:
        """Return a drag-continuity snap colour or ``None`` for strict models."""

        selection = self.snapped_selector_selection_at_position(position, size)
        if selection is None:
            return None
        return selection.paint_oklab

    def snapped_selector_selection_at_position(
        self, position: Sequence[float], size: Sequence[float]
    ) -> SelectorSelection | None:
        """Return a drag-continuity snap selection or ``None`` for strict models."""

        return None


def positions_close(a: Position, b: Position) -> bool:
    return abs(a[0] - b[0]) <= 0.5 and abs(a[1] - b[1]) <= 0.5
