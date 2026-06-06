"""Pure readout panel interaction state."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from oklab_colour_picker.colour_presentation import PresentedColour
from oklab_colour_picker.colour_state import ColourIntent


class ReadoutState(Enum):
    IDLE = "IDLE"
    EDITING = "EDITING"


class EditExit(Enum):
    COMMIT = "COMMIT"
    CANCEL = "CANCEL"


class ReadoutAction(Enum):
    NONE = "NONE"
    APPLY = "APPLY"
    SYNC_DRAFT_PRESENTATION = "SYNC_DRAFT_PRESENTATION"


@dataclass(frozen=True)
class LatchedColour:
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

    The Qt widget owns the controls. This object owns only view interaction
    state: current/previous presented colours, the active edit latch, and the
    draft intent used to identify the controller echo of an in-flight preview.
    """

    def __init__(self) -> None:
        self.current: PresentedColour | None = None
        self.previous: PresentedColour | None = None
        self.state = ReadoutState.IDLE
        self.latched_colour: LatchedColour | None = None
        self.draft_intent: ColourIntent | None = None

    @property
    def state_name(self) -> str:
        return self.state.value

    def seed_initial(self, colour: PresentedColour) -> ReadoutResult:
        self.current = colour
        self.previous = colour
        self.state = ReadoutState.IDLE
        self.latched_colour = None
        self.draft_intent = None
        return ReadoutResult.apply(colour)

    def show_colour(
        self,
        colour: PresentedColour,
        *,
        committed: bool,
        preview: bool = False,
    ) -> ReadoutResult:
        if self.state is ReadoutState.EDITING:
            if preview and colour.intent == self.draft_intent:
                return ReadoutResult.sync_draft(colour)
            self.latched_colour = LatchedColour(colour=colour, committed=committed)
            return ReadoutResult.none()
        self.apply_colour(colour, committed=committed)
        return ReadoutResult.apply(colour)

    def set_previous_colour(self, colour: PresentedColour | None) -> None:
        self.previous = colour

    def apply_colour(self, colour: PresentedColour, *, committed: bool) -> None:
        if (
            committed
            and self.current is not None
            and colour.intent != self.current.intent
        ):
            self.previous = self.current
        self.current = colour

    def begin_edit(self) -> None:
        if self.state is ReadoutState.EDITING:
            return
        self.state = ReadoutState.EDITING
        self.latched_colour = None

    def set_draft(self, intent: ColourIntent) -> None:
        self.draft_intent = intent

    def finish_edit(self, exit_kind: EditExit) -> ReadoutResult:
        if self.state is ReadoutState.IDLE:
            return ReadoutResult.none()
        latched = self.latched_colour
        self.state = ReadoutState.IDLE
        self.latched_colour = None
        self.draft_intent = None
        if exit_kind is not EditExit.CANCEL:
            return ReadoutResult.none()
        if latched is not None:
            self.apply_colour(latched.colour, committed=latched.committed)
            return ReadoutResult.apply(latched.colour)
        if self.current is not None:
            return ReadoutResult.apply(self.current)
        return ReadoutResult.none()
