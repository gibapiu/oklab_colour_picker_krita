import pytest

from oklab_colour_picker.domain.selector_interaction import (
    Broadcast,
    FocusOut,
    Indicator,
    KeyRelease,
    Navigation,
    PickResult,
    PointerMove,
    PointerPress,
    PointerRelease,
    Reframe,
    SelectorInteraction,
)


class FakeCtx:
    def __init__(self):
        self.colour = None
        self.previews = []
        self.commits = []

    def set_colour(self, colour):
        self.colour = colour

    def preview(self, colour):
        self.previews.append(colour)

    def commit(self, colour):
        self.commits.append(colour)

    def pick(self, point):
        if 0 <= point[0] <= 9:
            return PickResult.exact(float(point[0]))
        return PickResult.snapped(9.0, (9.0, 0.0))

    def quantized_equal(self, a, b):
        return a is not None and b is not None and round(a) == round(b)

    def model_position(self):
        return None if self.colour is None else (self.colour, 0.0)

    def model_indicator(self):
        point = self.model_position()
        return Indicator.nothing() if point is None else Indicator.at(point)


def test_drag_previews_commits_and_absorbs_its_echo():
    ctx = FakeCtx()
    interaction = SelectorInteraction()
    interaction.dispatch(ctx, Broadcast(1.0))
    assert ctx.previews == ctx.commits == []
    interaction.dispatch(ctx, PointerPress((2.0, 3.0)))
    interaction.dispatch(ctx, PointerMove((5.0, 3.0)))
    assert ctx.previews == [2.0, 5.0]
    assert ctx.commits == []
    interaction.dispatch(ctx, PointerRelease((5.0, 3.0)))
    assert ctx.commits == [5.0]
    assert interaction.indicator_position(ctx) == (5.0, 3.0)
    result = interaction.dispatch(ctx, Broadcast(5.4))
    assert result.absorbed_echo
    assert not result.rendered_broadcast
    assert interaction.indicator_position(ctx) == (5.0, 3.0)
    result = interaction.dispatch(ctx, Broadcast(8.0))
    assert result.rendered_broadcast
    assert ctx.colour == 8.0
    assert interaction.indicator_position(ctx) == (8.0, 0.0)


def test_drag_keeps_local_selection_during_broadcast_and_reframe():
    ctx = FakeCtx()
    interaction = SelectorInteraction()
    interaction.dispatch(ctx, PointerPress((2.0, 3.0)))
    result = interaction.dispatch(ctx, Broadcast(8.0))
    interaction.dispatch(ctx, Reframe())
    assert not result.rendered_broadcast
    assert ctx.colour == 2.0
    assert interaction.indicator_position(ctx) == (2.0, 3.0)
    assert interaction.navigation_origin(ctx, (0.0, 0.0)) is None
    interaction.dispatch(ctx, PointerRelease((2.0, 3.0)))
    assert ctx.commits == [2.0]


@pytest.mark.parametrize("previous, previews", [(None, [None, None]), (4.0, [])])
def test_invalid_press_and_release_preserve_previous_colour(previous, previews):
    ctx = FakeCtx()
    ctx.colour = previous
    ctx.pick = lambda _point: PickResult.invalid()
    interaction = SelectorInteraction()
    interaction.dispatch(ctx, PointerPress((50.0, 0.0)))
    interaction.dispatch(ctx, PointerRelease((60.0, 0.0)))
    assert ctx.colour == previous
    assert ctx.previews == previews
    assert ctx.commits == []
    assert interaction.navigation_origin(ctx, (0.0, 0.0)) is not None


def test_drag_outside_gamut_commits_the_snapped_colour():
    ctx = FakeCtx()
    interaction = SelectorInteraction()
    interaction.dispatch(ctx, PointerPress((3.0, 0.0)))
    interaction.dispatch(ctx, PointerMove((99.0, 0.0)))
    assert ctx.previews == [3.0, 9.0]
    assert interaction.indicator_position(ctx) == (9.0, 0.0)
    interaction.dispatch(ctx, PointerRelease((99.0, 0.0)))
    assert ctx.commits == [9.0]


@pytest.mark.parametrize("finish", [KeyRelease(), FocusOut()])
def test_keyboard_selection_commits_on_release_or_focus_loss(finish):
    ctx = FakeCtx()
    interaction = SelectorInteraction()
    interaction.dispatch(ctx, Navigation((4.0, 3.0), 4.0))
    interaction.dispatch(ctx, Navigation((5.0, 3.0), 5.0))
    interaction.dispatch(ctx, Broadcast(8.0))
    assert ctx.previews == [4.0, 5.0]
    assert ctx.commits == []
    interaction.dispatch(ctx, finish)
    assert ctx.commits == [5.0]
    assert interaction.indicator_position(ctx) == (5.0, 3.0)


def test_mouse_selection_replaces_pending_keyboard_commit():
    ctx = FakeCtx()
    interaction = SelectorInteraction()
    interaction.dispatch(ctx, Navigation((4.0, 0.0), 4.0))
    interaction.dispatch(ctx, PointerPress((2.0, 0.0)))
    interaction.dispatch(ctx, PointerRelease((2.0, 0.0)))
    interaction.dispatch(ctx, KeyRelease())
    assert ctx.commits == [2.0]


def test_reframe_releases_committed_indicator_anchor():
    ctx = FakeCtx()
    interaction = SelectorInteraction()
    interaction.dispatch(ctx, PointerPress((2.0, 3.0)))
    interaction.dispatch(ctx, PointerRelease((2.0, 3.0)))
    assert interaction.indicator_position(ctx) == (2.0, 3.0)
    interaction.dispatch(ctx, Reframe())
    assert interaction.indicator_position(ctx) == (2.0, 0.0)
