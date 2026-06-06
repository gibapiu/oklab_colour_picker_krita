"""Pure, Qt-free interaction state machine for the selector view.

States are *objects* (GoF State pattern): each one owns its data, answers its
own questions (``anchor``, ``broadcast``, ``indicator``) and returns typed
interaction results. There is no conditional dispatch on a state tag and no
ad-hoc "anchored" / "in-flight" state groups — those questions are answered
polymorphically by the state itself.

The module imports nothing from Qt, Krita, or the widget/dock layer, so the
whole machine is unit-testable against a plain fake ``Ctx`` (see
``tests/test_selector_interaction.py``). The Qt ``SelectorWidget`` is a humble
adapter: it translates Qt events into state calls and renders what the state
reports.

``Ctx`` is the narrow port the widget implements. Colours are opaque to this
module; copying/quantization are the widget's responsibility through ``Ctx``.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar, Protocol


Position = tuple[float, float]
Point = tuple[float, float]


class StateKind(str, Enum):
    IDLE = "IDLE"
    DRAGGING = "DRAGGING"
    KEYBOARD = "KEYBOARD"
    PINNED = "PINNED"


@dataclass(frozen=True, slots=True)
class ExactPick:
    colour: object


@dataclass(frozen=True, slots=True)
class SnappedPick:
    colour: object
    position: Position


@dataclass(frozen=True, slots=True)
class InvalidPick:
    pass


Pick = ExactPick | SnappedPick | InvalidPick


class PickResult:
    """Factory facade for illegal-state-free pick results."""

    @staticmethod
    def exact(colour: object) -> Pick:
        return ExactPick(colour)

    @staticmethod
    def snapped(colour: object, position: Position) -> Pick:
        return SnappedPick(colour, position)

    @staticmethod
    def invalid() -> Pick:
        return InvalidPick()


@dataclass(frozen=True, slots=True)
class PointerPress:
    point: Point

    def dispatch(self, state: "State", ctx: "Ctx") -> "InteractionResult":
        return state.press(ctx, self.point)


@dataclass(frozen=True, slots=True)
class PointerMove:
    point: Point

    def dispatch(self, state: "State", ctx: "Ctx") -> "InteractionResult":
        return state.move(ctx, self.point)


@dataclass(frozen=True, slots=True)
class PointerRelease:
    point: Point

    def dispatch(self, state: "State", ctx: "Ctx") -> "InteractionResult":
        return state.release(ctx, self.point)


@dataclass(frozen=True, slots=True)
class Navigation:
    point: Point
    colour: object

    def dispatch(self, state: "State", ctx: "Ctx") -> "InteractionResult":
        return state.nav(ctx, self.point, self.colour)


@dataclass(frozen=True, slots=True)
class KeyRelease:
    def dispatch(self, state: "State", ctx: "Ctx") -> "InteractionResult":
        return state.key_release(ctx)


@dataclass(frozen=True, slots=True)
class FocusOut:
    def dispatch(self, state: "State", ctx: "Ctx") -> "InteractionResult":
        return state.focus_out(ctx)


@dataclass(frozen=True, slots=True)
class Reframe:
    def dispatch(self, state: "State", ctx: "Ctx") -> "InteractionResult":
        return state.reframe(ctx)


@dataclass(frozen=True, slots=True)
class Broadcast:
    colour: object | None

    def dispatch(self, state: "State", ctx: "Ctx") -> "InteractionResult":
        return state.broadcast(ctx, self.colour)


SelectorCommand = (
    PointerPress
    | PointerMove
    | PointerRelease
    | Navigation
    | KeyRelease
    | FocusOut
    | Reframe
    | Broadcast
)


@dataclass(frozen=True, slots=True)
class Ring:
    """One indicator ring to stroke; ``solid`` picks solid vs. dashed."""

    position: Position
    solid: bool


@dataclass(frozen=True, slots=True)
class Indicator:
    """What the view should draw for the current state (empty == nothing)."""

    rings: tuple[Ring, ...] = ()

    @classmethod
    def nothing(cls) -> "Indicator":
        return cls(())

    @classmethod
    def at(cls, position: Position) -> "Indicator":
        return cls((Ring(position, True),))


class Ctx(Protocol):
    """Narrow port the widget implements for the state machine to use."""

    @property
    def colour(self) -> object | None: ...
    def set_colour(self, colour: object | None) -> None: ...
    def preview(self, colour: object | None) -> None: ...
    def commit(self, colour: object) -> None: ...
    def pick(self, point: Point) -> Pick: ...
    def quantized_equal(self, a: object | None, b: object | None) -> bool: ...
    def model_indicator(self) -> Indicator: ...
    def model_position(self) -> Position | None: ...


@dataclass(frozen=True, slots=True)
class InteractionResult:
    state: "State"
    handled: bool = True
    rendered_broadcast: bool = False
    absorbed_echo: bool = False


class State(ABC):
    """Base state. Every event defaults to a no-op self-transition, so each
    concrete state only overrides the events it actually handles."""

    kind: ClassVar[StateKind]

    @property
    def name(self) -> str:
        return self.kind.value

    @property
    def anchor(self) -> Position | None:
        return None

    def indicator(self, ctx: Ctx) -> Indicator:
        return ctx.model_indicator()

    def indicator_position(self, ctx: Ctx) -> Position | None:
        anchor = self.anchor
        return anchor if anchor is not None else ctx.model_position()

    def navigation_origin(self, ctx: Ctx, fallback: Position) -> Position | None:
        return self.indicator_position(ctx) or fallback

    # Events ----------------------------------------------------------
    def broadcast(self, ctx: Ctx, colour: object | None) -> InteractionResult:
        return InteractionResult(self, handled=True, rendered_broadcast=False)

    def press(self, ctx: Ctx, point: Point) -> InteractionResult:
        return _begin_drag(ctx, point)

    def move(self, ctx: Ctx, point: Point) -> InteractionResult:
        return InteractionResult(self, handled=False)

    def release(self, ctx: Ctx, point: Point) -> InteractionResult:
        return InteractionResult(self, handled=False)

    def nav(self, ctx: Ctx, point: Point, colour: object) -> InteractionResult:
        return _begin_keyboard(ctx, point, colour)

    def key_release(self, ctx: Ctx) -> InteractionResult:
        return InteractionResult(self, handled=False)

    def focus_out(self, ctx: Ctx) -> InteractionResult:
        return InteractionResult(self, handled=False)

    def reframe(self, ctx: Ctx) -> InteractionResult:
        """Model or widget-size change."""

        return InteractionResult(self)


class _Anchored(State):
    """States that pin the indicator to a concrete pixel (DRAGGING/KEYBOARD/
    PINNED). The anchor *is* the indicator — no model round trip."""

    _anchor: Position

    @property
    def anchor(self) -> Position | None:
        return self._anchor

    def indicator(self, ctx: Ctx) -> Indicator:
        return Indicator.at(self._anchor) if ctx.colour is not None else Indicator.nothing()


@dataclass(frozen=True, slots=True)
class Idle(State):
    """Rendering an externally pushed colour; no anchor."""

    kind: ClassVar[StateKind] = StateKind.IDLE

    def broadcast(self, ctx: Ctx, colour: object | None) -> InteractionResult:
        # The controller currently broadcasts only concrete colours; accepting
        # None keeps direct/programmatic clears well-defined for the facade.
        ctx.set_colour(colour)
        return InteractionResult(self, rendered_broadcast=True)


@dataclass(frozen=True, slots=True)
class Dragging(_Anchored):
    """Pointer held; emitting previews. Owns the cancel target and the last
    in-gamut colour for the out-of-gamut release fallback."""

    _anchor: Position
    _before: object | None
    _last_valid: object | None
    kind: ClassVar[StateKind] = StateKind.DRAGGING

    def broadcast(self, ctx: Ctx, colour: object | None) -> InteractionResult:
        return InteractionResult(self)

    def navigation_origin(self, ctx: Ctx, fallback: Position) -> Position | None:
        return None

    def move(self, ctx: Ctx, point: Point) -> InteractionResult:
        return self._pick(ctx, point)

    def release(self, ctx: Ctx, point: Point) -> InteractionResult:
        picked = ctx.pick(point)
        selection = _drag_selection(
            picked,
            point,
            has_last_valid=self._last_valid is not None,
        )
        if selection is not None:
            ctx.set_colour(selection.colour)
            ctx.commit(selection.colour)
            return InteractionResult(Pinned(selection.colour, selection.position))
        if self._last_valid is not None:
            ctx.set_colour(self._last_valid)
            ctx.commit(self._last_valid)
            return InteractionResult(Pinned(self._last_valid, self._anchor))
        ctx.set_colour(self._before)
        ctx.preview(self._before)
        return InteractionResult(Idle())

    def _pick(self, ctx: Ctx, point: Point) -> InteractionResult:
        picked = ctx.pick(point)
        selection = _drag_selection(
            picked,
            point,
            has_last_valid=self._last_valid is not None,
        )
        if selection is not None:
            ctx.set_colour(selection.colour)
            ctx.preview(selection.colour)
            return InteractionResult(
                Dragging(selection.position, self._before, selection.colour)
            )
        if self._last_valid is not None:
            return InteractionResult(self)
        ctx.set_colour(None)
        ctx.preview(None)
        return InteractionResult(Dragging(self._anchor, self._before, None))


@dataclass(frozen=True, slots=True)
class Keyboard(_Anchored):
    """Arrow/page navigation in flight; commit pending until key-up/blur."""

    _anchor: Position
    kind: ClassVar[StateKind] = StateKind.KEYBOARD

    def broadcast(self, ctx: Ctx, colour: object | None) -> InteractionResult:
        return InteractionResult(self)

    def nav(self, ctx: Ctx, point: Point, colour: object) -> InteractionResult:
        ctx.set_colour(colour)
        ctx.preview(colour)
        return InteractionResult(Keyboard(_position(point)))

    def key_release(self, ctx: Ctx) -> InteractionResult:
        return self._flush(ctx)

    def focus_out(self, ctx: Ctx) -> InteractionResult:
        return self._flush(ctx)

    def _flush(self, ctx: Ctx) -> InteractionResult:
        colour = ctx.colour
        if colour is None:
            return InteractionResult(Idle())
        ctx.commit(colour)
        return InteractionResult(Pinned(colour, self._anchor))


@dataclass(frozen=True, slots=True)
class Pinned(_Anchored):
    """Post-commit: holds the committed colour at its pixel until something
    external supersedes it. The deliberate UX terminal state."""

    _colour: object | None
    _anchor: Position
    kind: ClassVar[StateKind] = StateKind.PINNED

    def broadcast(self, ctx: Ctx, colour: object | None) -> InteractionResult:
        if colour is not None and ctx.quantized_equal(colour, self._colour):
            return InteractionResult(self, absorbed_echo=True)
        return Idle().broadcast(ctx, colour)

    def reframe(self, ctx: Ctx) -> InteractionResult:
        return InteractionResult(Idle())


def _begin_drag(ctx: Ctx, point: Point) -> InteractionResult:
    before = ctx.colour
    picked = ctx.pick(point)
    selection = _drag_selection(picked, point, has_last_valid=before is not None)
    if selection is not None:
        ctx.set_colour(selection.colour)
        ctx.preview(selection.colour)
        return InteractionResult(Dragging(selection.position, before, selection.colour))
    if before is not None:
        return InteractionResult(Idle(), handled=False)
    ctx.set_colour(None)
    ctx.preview(None)
    return InteractionResult(Dragging(_position(point), before, None))


def _begin_keyboard(ctx: Ctx, point: Point, colour: object) -> InteractionResult:
    ctx.set_colour(colour)
    ctx.preview(colour)
    return InteractionResult(Keyboard(_position(point)))


class SelectorInteraction:
    def __init__(self, initial: State | None = None) -> None:
        self._state = initial or Idle()
        self._transition_log: list[StateKind] = [self._state.kind]

    @property
    def state_kind(self) -> StateKind:
        return self._state.kind

    @property
    def state_name(self) -> str:
        return self._state.name

    @property
    def anchor(self) -> Position | None:
        return self._state.anchor

    @property
    def transition_log(self) -> tuple[str, ...]:
        return tuple(kind.value for kind in self._transition_log)

    def dispatch(self, ctx: Ctx, command: SelectorCommand) -> InteractionResult:
        result = command.dispatch(self._state, ctx)
        self._adopt(result.state)
        return result

    def indicator(self, ctx: Ctx) -> Indicator:
        return self._state.indicator(ctx)

    def indicator_position(self, ctx: Ctx) -> Position | None:
        return self._state.indicator_position(ctx)

    def navigation_origin(self, ctx: Ctx, fallback: Position) -> Position | None:
        return self._state.navigation_origin(ctx, fallback)

    def _adopt(self, state: State) -> None:
        if state.kind is not self._state.kind:
            self._transition_log.append(state.kind)
        self._state = state


@dataclass(frozen=True, slots=True)
class DragSelection:
    colour: object
    position: Position


def _drag_selection(
    picked: Pick,
    point: Point,
    *,
    has_last_valid: bool,
) -> DragSelection | None:
    match picked:
        case ExactPick(colour=colour):
            return DragSelection(colour, _position(point))
        case SnappedPick(colour=colour, position=position) if has_last_valid:
            return DragSelection(colour, position)
        case InvalidPick() | SnappedPick():
            # Snapping starts only after a drag has first owned a valid colour;
            # an off-gamut press stays cancellable and restores the prior state.
            return None

def _position(point: Point) -> Position:
    return float(point[0]), float(point[1])
