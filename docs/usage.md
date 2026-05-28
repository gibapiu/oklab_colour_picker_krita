# Usage

Open the docker from **Settings &rarr; Dockers &rarr; OKLab Colour Selector**
and dock it wherever it fits your workspace.

## Selector tabs

Each tab fixes one OKLCh axis and lets you pick the other two on a 2D slice:

- **Hue / Chroma** &mdash; pick hue and saturation at a fixed lightness.
  Good for picking different colors without breaking the light and dark balance.
- **Hue / Lightness** &mdash; pick hue and lightness at a fixed chroma.
  Good for cases where you need a locked chroma, for example keeping the shadow saturation on the level.
- **Lightness / Chroma** &mdash; pick lightness and saturation at a fixed hue.
  Closer to a regular color picker, but allows you to see the limits of each hue.

The axis the tab keeps fixed is the one you control with the matching L, C,
or H slider underneath.

## Controls

- **Drag** inside the selector to preview live, **release** to commit.
- **L, C, H gradient sliders** for direct OKLCH adjustment.
- **Number boxes** for precise values.
- **Hex field** on the swatch — type six digits, with or without `#`.
- **Revert arrow** on the swatch jumps back to the previous colour.

## Out of gamut

Intrinsic differences between colors mean that some colors do no fit into you LCH selection. If your pick sits outside the
visible gamut, the swatch shows a small **⚠** marker and the committed colour
is clipped to the closest in-gamut neighbour.

## Sync

Switching tabs keeps the current colour — you're just looking at it from a
different angle. The docker reads and writes Krita's foreground colour, so it
stays in sync with any other color changes you do.
