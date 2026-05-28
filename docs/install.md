# Install

## Requirements

- Krita 5.2 or newer.
- NumPy available to Krita's Python.

Krita's bundled Python on Windows and macOS doesn't always ship NumPy. If it's
missing, the docker shows a dependency message instead of the selector.

## 1. In Krita go into Tools &rarr Scripts &rarr Import Python Plugin From File

## 2. Open **Settings &rarr; Dockers &rarr; OKLab Colour Selector**.

## 3. Install NumPy

**Linux** — usually picks up your system Python:

```sh
python3 -m pip install --user numpy
```

Or use your distro's package (`sudo apt install python3-numpy`,
`sudo dnf install python3-numpy`, etc).

**Windows** — open the docker. If NumPy is missing, click **Install NumPy**.
The plugin installs it into:

```text
%APPDATA%\krita\oklab_colour_picker\site-packages
```

To do it by hand:

```bat
"C:\Program Files\Krita (x64)\bin\python.exe" -m pip install ^
  --upgrade --only-binary=:all: ^
  --target "%APPDATA%\krita\oklab_colour_picker\site-packages" ^
  "numpy>=1.26,<3"
```

**macOS** — use Krita's bundled Python:

```sh
/Applications/krita.app/Contents/MacOS/krita_python -m pip install numpy
```

If your build doesn't ship `krita_python`, follow Krita's docs for installing
Python packages into its bundled Python.

## 4. Enable the plugin

1. Restart Krita.
2. Open **Settings &rarr; Dockers &rarr; OKLab Colour Selector**.
3. Proceed to install dependencies as needed.
4. Restart Krita again.
