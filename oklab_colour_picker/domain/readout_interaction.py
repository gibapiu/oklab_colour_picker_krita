"""Pure readout panel interaction state."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from oklab_colour_picker.domain.colour_presentation import PresentedColour
from oklab_colour_picker.domain.colour_state import ColourIntent


class ReadoutState(Enum):
    IDLE = auto()
    EDITING = auto()


class EditExit(Enum):
    COMMIT = auto()
    CANCEL = auto()


class ReadoutAction(Enum):
    NONE = auto()
    APPLY = auto()
    SYNC_DRAFT_PRESENTATION = auto()


@dataclass(frozen=True)
class _LatchedColour:
    colour: PresentedColour
    committed: bool


@dataclass(frozen=True)
class ReadoutResult:
    action: ReadoutAction = ReadoutAction.NONE
    colour: PresentedColour | None = None

    @classmethod
    def none(cls) -> "ReadoutResult":
        return cls()

    @classmethod
    def apply(cls, colour: PresentedColour) -> "ReadoutResult":
        return cls(ReadoutAction.APPLY, colour)

    @classmethod
    def sync_draft(cls, colour: PresentedColour) -> "ReadoutResult":
        return cls(ReadoutAction.SYNC_DRAFT_PRESENTATION, colour)


class ReadoutSession:
    """State machine for readout broadcasts and edit latching.

    Owns only view interaction state: current/previous presented colours, the active edit latch,
    and the draft intent used to identify the controller echo of an in-flight preview.
    """

    def __init__(self) -> None:
        self._current: PresentedColour | None = None
        self._previous: PresentedColour | None = None
        self._state = ReadoutState.IDLE
        self._latched_colour: _LatchedColour | None = None
        self._draft_intent: ColourIntent | None = None

    @property
    def current(self) -> PresentedColour | None:
        return self._current

    @property
    def previous(self) -> PresentedColour | None:
        return self._previous

    @property
    def state(self) -> ReadoutState:
        return self._state

    def seed_initial(self, colour: PresentedColour) -> ReadoutResult:
        self._current = colour
        self._previous = colour
        self._state = ReadoutState.IDLE
        self._latched_colour = None
        self._draft_intent = None
        return ReadoutResult.apply(colour)

    def show_colour(
        self,
        colour: PresentedColour,
        *,
        committed: bool,
        preview: bool = False,
    ) -> ReadoutResult:
        if self._state is ReadoutState.EDITING:
            if preview and colour.intent == self._draft_intent:
                return ReadoutResult.sync_draft(colour)
            self._latched_colour = _LatchedColour(
                colour=colour,
                committed=committed,
            )
            return ReadoutResult.none()
        self._apply_colour(colour, committed=committed)
        return ReadoutResult.apply(colour)

    def _apply_colour(self, colour: PresentedColour, *, committed: bool) -> None:
        if (
            committed
            and self._current is not None
            and colour.intent != self._current.intent
        ):
            self._previous = self._current
        self._current = colour

    def begin_edit(self) -> None:
        if self._state is ReadoutState.EDITING:
            return
        self._state = ReadoutState.EDITING
        self._latched_colour = None

    def set_draft(self, intent: ColourIntent) -> None:
        self._draft_intent = intent

    def finish_edit(self, exit_kind: EditExit) -> ReadoutResult:
        if self._state is ReadoutState.IDLE:
            return ReadoutResult.none()
        latched = self._latched_colour
        self._state = ReadoutState.IDLE
        self._latched_colour = None
        self._draft_intent = None
        if exit_kind is not EditExit.CANCEL:
            return ReadoutResult.none()
        if latched is not None:
            self._apply_colour(latched.colour, committed=latched.committed)
            return ReadoutResult.apply(latched.colour)
        if self._current is not None:
            return ReadoutResult.apply(self._current)
        return ReadoutResult.none()
