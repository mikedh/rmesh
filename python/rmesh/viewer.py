"""Bridge to optional rmesh-viewer package for interactive display."""


def _viewer():
    """Import the rmesh_viewer native module, raising a helpful error if missing."""
    try:
        import rmesh_viewer

        return rmesh_viewer
    except ImportError:
        raise ImportError(
            "Interactive viewer requires the rmesh-viewer package.\n"
            "Install it with: pip install rmesh-viewer"
        ) from None


def show_trimesh(mesh, *, title="rmesh viewer", width=1280, height=720, background=None):
    """Show a Trimesh in an interactive 3D viewer window."""
    kw = {"title": title, "width": width, "height": height}
    if background is not None:
        kw["background"] = background
    _viewer().show_trimesh(mesh, **kw)


def show_scene(scene, *, title="rmesh viewer", width=1280, height=720, background=None):
    """Show a Scene in an interactive 3D viewer window."""
    kw = {"title": title, "width": width, "height": height}
    if background is not None:
        kw["background"] = background
    _viewer().show_scene(scene, **kw)


def show_polygon2d(poly, *, title="rmesh 2D", width=1280, height=720, background=None):
    """Show a Polygon2D in an interactive 2D viewer window."""
    kw = {"title": title, "width": width, "height": height}
    if background is not None:
        kw["background"] = background
    interiors = getattr(poly, "interiors", None)
    _viewer().show_polygon2d(poly.exterior, interiors=interiors, **kw)


def show_path2d(path, *, title="rmesh 2D", width=1280, height=720, background=None):
    """Show a Path2D in an interactive 2D viewer window."""
    kw = {"title": title, "width": width, "height": height}
    if background is not None:
        kw["background"] = background
    polylines = path.discretize()
    _viewer().show_polylines_2d(polylines, **kw)
