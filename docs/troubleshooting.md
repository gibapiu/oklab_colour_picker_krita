# Troubleshooting

## Plugin doesn't show up in the Plugin Manager

The `.desktop` file has to sit **directly** inside `pykrita/`, not nested inside
the package folder:

```text
pykrita/
├── oklab_colour_picker.desktop   ← here
└── oklab_colour_picker/
    └── ...                        ← not here
```

Restart Krita after copying files. Python plugins are only picked up at startup.

## Docker shows a "missing dependency" message

NumPy isn't available to Krita's Python.

On Windows, click **Install NumPy** in the docker. On other platforms, install
it manually — see [install.md](install.md) — then restart Krita.

## Docker doesn't open

Make sure the plugin is enabled:

1. **Settings &rarr; Configure Krita&hellip; &rarr; Python Plugin Manager**.
2. Tick **OKLab Colour Selector**.
3. Restart Krita.
4. **Settings &rarr; Dockers &rarr; OKLab Colour Selector**.

## Startup errors

Open **Tools &rarr; Scripts &rarr; Python Script Editor** to read the
traceback, or launch Krita from a terminal so errors print there.

## Edits don't show up

Krita loads Python plugins at startup. Restart after editing.

For active development, symlink the package and `.desktop` file from `pykrita/`
into your working copy so changes apply on each restart without re-copying.
