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

## Docker says NumPy could not be loaded

Krita found NumPy, but that installation is not working. This is different from
NumPy being missing, which prompts for installation permission.

The plugin keeps private NumPy installations separate for each Python ABI. A
compatible unversioned installation from an older plugin release is reused.

To find the private dependency folder for the current Krita runtime, run this in
Krita's Python Script Editor:

```python
from krita import Krita
from oklab_colour_picker.infrastructure.dependency_paths import vendor_site_packages_path

print(vendor_site_packages_path(Krita.instance().getAppDataLocation()))
```

Close Krita, remove only the printed ABI-specific folder, then restart Krita and
install NumPy again. Do not remove another ABI folder; a parallel Krita version
may still use it.

If the problem continues, [open an issue](https://github.com/gibapiu/oklab_colour_picker_krita/issues/new)
and include the detailed error printed when Krita is launched from a terminal.

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
