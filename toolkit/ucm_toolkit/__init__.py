"""UCM toolkit command-line package."""

__all__ = ["__version__"]

try:
    from ._version import __version__
except ModuleNotFoundError:
    # Editable installs read the same authority as the native UCM package.
    from pathlib import Path

    __version__ = next(
        line.partition("=")[2]
        for line in (Path(__file__).resolve().parents[2] / "version.ini")
        .read_text()
        .splitlines()
        if line.startswith("UCM_VERSION=")
    )
