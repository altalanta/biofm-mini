"""BioFM package namespace."""

try:
    from ._version import __version__
except ImportError:  # pragma: no cover - package not built with setuptools_scm yet
    __version__ = "0.0.0"

__all__ = ["__version__"]
