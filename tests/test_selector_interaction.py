"""Pure, Qt-free tests for the selector interaction state machine.

These exercise the State pattern directly through a fake ``Ctx`` — no
QApplication, no widgets — which is the whole point of the humble-object
split: the interaction logic is unit-testable in isolation.
"""

from oklab_colour_picker.selector_interaction import (
    Broadcast,
    Dragging,
    FocusOut,
    Idle,
    KeyRelease,
    Keyboard,
    Navigation,
    PickResult,
    PointerMove,
    PointerPress,
    PointerRelease,
    Pinned,
    Reframe,
    SelectorInteraction,
    StateKind,
)


class FakeCtx:
    """A 1-D selector: x in [0, 10) is in gamut; colour == float(x)."""

    WIDTH = 10

    def __init__(self):
        self.colour = None
        self.previews = []
        self.commits = []

    # Ctx port -------------------------------------------------------
    def set_colour(self, colour):
        self.colour = colour

    def preview(self, colour):
        self.previews.append(colour)

    def commit(self, colour):
        self.commits.append(colour)

    def pick(self, point):
        x = point[0]
        if 0 <= x < self.WIDTH:
            return PickResult.exact(float(x))
        snapped_x = float(min(self.WIDTH - 1, max(0, x)))
        return PickResult.snapped(snapped_x, (snapped_x, 0.0))

    def quantized_equal(self, a, b):
        return a is not None and b is not None and round(a) == round(b)

    def model_indicator(self):
        return None

    def model_position(self):
        if self.colour is None or not (0 <= self.colour < self.WIDTH):
            return None
        return (self.colour, 0.0)


def test_idle_absorb_sets_colour_and_stays_idle():
    ctx = FakeCtx()
    result = Idle().broadcast(ctx, 3.0)
    state = result.state
    assert isinstance(state, Idle)
    assert result.rendered_broadcast
    assert ctx.colour == 3.0
    assert ctx.previews == [] and ctx.commits == []


def test_click_drag_release_is_idle_dragging_pinned():
    ctx = FakeCtx()
    state = Idle().press(ctx, (2.0, 0.0)).state
    assert isinstance(state, Dragging)
    assert state.anchor == (2.0, 0.0)
    assert ctx.previews == [2.0]

    state = state.move(ctx, (5.0, 0.0)).state
    assert isinstance(state, Dragging) and state.anchor == (5.0, 0.0)
    assert ctx.previews == [2.0, 5.0]

    state = state.release(ctx, (7.0, 0.0)).state
    assert isinstance(state, Pinned)
    assert state.anchor == (7.0, 0.0)
    assert ctx.commits == [7.0]


def test_dragging_ignores_inbound_colour_in_flight():
    ctx = FakeCtx()
    state = Idle().press(ctx, (2.0, 0.0)).state
    same = state
    result = state.broadcast(ctx, 9.0)
    state = result.state
    assert state is same
    assert not result.rendered_broadcast
    assert ctx.colour == 2.0  # unchanged by the ignored broadcast


def test_cold_start_invalid_press_is_cancellable_and_never_commits():
    ctx = FakeCtx()
    ctx.pick = lambda _point: PickResult.invalid()
    state = Idle().press(ctx, (50.0, 0.0)).state
    assert isinstance(state, Dragging)
    assert ctx.previews == [None]

    state = state.release(ctx, (60.0, 0.0)).state
    assert isinstance(state, Idle)
    assert ctx.colour is None
    assert ctx.previews[-1] is None
    assert ctx.commits == []


def test_warm_invalid_press_is_a_no_op_and_preserves_colour():
    ctx = FakeCtx()
    ctx.set_colour(4.0)
    ctx.pick = lambda _point: PickResult.invalid()
    result = Idle().press(ctx, (50.0, 0.0))
    assert isinstance(result.state, Idle)
    assert result.handled is False
    assert ctx.colour == 4.0
    assert ctx.previews == []
    assert ctx.commits == []


def test_drag_leaving_gamut_snaps_continuously_and_commits_snapped():
    ctx = FakeCtx()
    state = Idle().press(ctx, (3.0, 0.0)).state       # valid: last_valid set
    state = state.move(ctx, (99.0, 0.0)).state        # off-gamut -> snapped
    assert state.anchor == (9.0, 0.0)
    assert None not in ctx.previews
    state = state.release(ctx, (99.0, 0.0)).state
    assert isinstance(state, Pinned)
    assert state.anchor == (9.0, 0.0)
    assert ctx.commits == [9.0]                      # snapped to the rim


def test_keyboard_nav_then_release_commits():
    ctx = FakeCtx()
    state = Idle().nav(ctx, (4.0, 0.0), 4.0).state
    assert isinstance(state, Keyboard)
    assert ctx.previews == [4.0] and ctx.commits == []

    state = state.nav(ctx, (5.0, 0.0), 5.0).state
    assert isinstance(state, Keyboard)
    state = state.key_release(ctx).state
    assert isinstance(state, Pinned)
    assert ctx.commits == [5.0]


def test_keyboard_focus_out_flushes_commit():
    ctx = FakeCtx()
    state = Idle().nav(ctx, (4.0, 0.0), 4.0).state.focus_out(ctx).state
    assert isinstance(state, Pinned)
    assert ctx.commits == [4.0]


def test_mouse_press_during_keyboard_cancels_without_commit():
    ctx = FakeCtx()
    state = Idle().nav(ctx, (4.0, 0.0), 4.0).state
    state = state.press(ctx, (2.0, 0.0)).state
    assert isinstance(state, Dragging)
    assert ctx.commits == []


def test_pinned_swallows_quantized_equal_echo_but_yields_to_difference():
    ctx = FakeCtx()
    pinned = Pinned(5.0, (5.0, 0.0))

    result = pinned.broadcast(ctx, 5.4)  # round-equal -> echo
    same = result.state
    assert same is pinned
    assert not result.rendered_broadcast
    assert ctx.colour is None  # not re-applied

    other = pinned.broadcast(ctx, 8.0).state
    assert isinstance(other, Idle)
    assert ctx.colour == 8.0


def test_pinned_reports_absorbed_echo_for_coordinate_refresh():
    ctx = FakeCtx()
    pinned = Pinned(5.0, (5.0, 0.0))

    result = pinned.broadcast(ctx, 5.4)

    assert result.state is pinned
    assert result.absorbed_echo
    assert not result.rendered_broadcast
    assert ctx.colour is None


def test_reframe_only_resets_pinned():
    ctx = FakeCtx()
    assert isinstance(Pinned(1.0, (1.0, 0.0)).reframe(ctx).state, Idle)
    idle = Idle()
    assert idle.reframe(ctx).state is idle
    drag = Idle().press(ctx, (2.0, 0.0)).state
    assert drag.reframe(ctx).state is drag


def test_interaction_facade_dispatches_commands_and_records_transitions():
    ctx = FakeCtx()
    interaction = SelectorInteraction()

    result = interaction.dispatch(ctx, PointerPress((2.0, 0.0)))
    assert result.handled
    assert interaction.state_kind is StateKind.DRAGGING
    interaction.dispatch(ctx, PointerMove((3.0, 0.0)))
    interaction.dispatch(ctx, PointerRelease((3.0, 0.0)))
    assert interaction.state_kind is StateKind.PINNED

    result = interaction.dispatch(ctx, Broadcast(3.4))
    assert result.handled
    assert not result.rendered_broadcast
    assert interaction.transition_log == ("IDLE", "DRAGGING", "PINNED")

    result = interaction.dispatch(ctx, Reframe())
    assert result.handled
    assert interaction.state_kind is StateKind.IDLE


def test_interaction_facade_handles_keyboard_and_focus_commands():
    ctx = FakeCtx()
    interaction = SelectorInteraction()

    interaction.dispatch(ctx, Navigation((4.0, 0.0), 4.0))
    assert interaction.state_kind is StateKind.KEYBOARD
    interaction.dispatch(ctx, FocusOut())
    assert interaction.state_kind is StateKind.PINNED

    interaction.dispatch(ctx, Reframe())
    interaction.dispatch(ctx, Navigation((5.0, 0.0), 5.0))
    interaction.dispatch(ctx, KeyRelease())
    assert ctx.commits[-1] == 5.0
