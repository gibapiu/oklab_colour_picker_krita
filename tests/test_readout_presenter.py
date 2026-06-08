import pytest

from oklab_colour_picker.app.readout_presenter import ReadoutPresenter
from oklab_colour_picker.domain.colour_state import ColourIntent
from tests.helpers import presented_colour


def test_maps_presented_colour_to_complete_readout_display():
    presenter = ReadoutPresenter()
    current = presented_colour(
        ColourIntent.from_lch(0.62, 0.11, 1.25),
        srgb8=(74, 143, 178),
        in_gamut=False,
    )
    previous = presented_colour(
        ColourIntent.from_lch(0.40, 0.05, 2.0),
        srgb8=(17, 34, 51),
    )

    display = presenter.present(current, previous=previous)

    assert display.selector_lch == pytest.approx(current.selector_lch)
    assert display.srgb8 == (74, 143, 178)
    assert display.out_of_gamut
    assert display.revert_hex == "#112233"


def test_current_edit_values_override_colour_axes_without_changing_presentation():
    presenter = ReadoutPresenter()
    colour = presented_colour(
        ColourIntent.from_lch(0.40, 0.05, 0.25),
        srgb8=(10, 20, 30),
        in_gamut=True,
    )

    display = presenter.present(
        colour,
        previous=None,
        selector_lch=(0.75, 0.10, 2.50),
    )

    assert display.selector_lch == pytest.approx((0.75, 0.10, 2.50))
    assert display.srgb8 == (10, 20, 30)
    assert not display.out_of_gamut
    assert display.revert_hex is None
