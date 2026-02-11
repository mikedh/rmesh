# Re-export native module
from . import compatibility
from .rmesh import *  # noqa: F403

__doc__ = """rmesh: A fast mesh processing library."""

try:
    from .rmesh import __all__ as _native_all

    __all__ = [*_native_all, "compatibility"]
except ImportError:
    __all__ = ["compatibility"]

# Attach .show() methods that delegate to the optional rmesh-viewer package.
# These always exist on the classes; if rmesh-viewer is not installed,
# calling .show() raises a helpful ImportError.
from . import viewer as _viewer

try:
    from .rmesh import Trimesh, Scene, Polygon2D, Path2D

    Trimesh.show = lambda self, **kw: _viewer.show_trimesh(self, **kw)
    Scene.show = lambda self, **kw: _viewer.show_scene(self, **kw)
    Polygon2D.show = lambda self, **kw: _viewer.show_polygon2d(self, **kw)
    Path2D.show = lambda self, **kw: _viewer.show_path2d(self, **kw)
except ImportError:
    pass
