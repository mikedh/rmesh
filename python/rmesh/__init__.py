# Re-export native module
from . import _compatibility
from .rmesh import *  # noqa: F403

__doc__ = """rmesh: A fast mesh processing library."""

try:
    from .rmesh import __all__ as _native_all

    __all__ = [*_native_all, "_compatibility"]
except ImportError:
    __all__ = ["_compatibility"]
