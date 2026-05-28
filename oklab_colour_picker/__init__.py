"""OKLab colour picker Krita plugin package."""

from oklab_colour_picker.plugin import register_plugin


# Krita loads the package named by X-KDE-Library, so package import performs
# registration when Krita's Python API is present. Outside Krita this is a
# no-op.
register_plugin()


__all__ = ["register_plugin"]
