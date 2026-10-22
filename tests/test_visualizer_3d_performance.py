"""Behavioral guards for the fast 3D view and lazy main-window refresh."""

import numpy as np
import pytest

from optics_app.engine.lde import SurfaceStack
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.gui.qt_compat import QT_AVAILABLE, QtWidgets
from optics_app.gui.visualizer_3d import (
    HAS_MPL_QT, Visualizer3DWidget, _surface_sag_values,
)


def _stack():
    stack = SurfaceStack()
    stack.add_surface(Surface(name="Source", surface_type=SurfaceType.OBJECT, thickness=20))
    stack.add_surface(Surface(name="Lens front", radius=50, thickness=5, glass="N-BK7", semi_diameter=10))
    stack.add_surface(Surface(name="Lens rear", radius=-50, thickness=20, glass="AIR", semi_diameter=10))
    stack.add_surface(Surface(name="Sensor", surface_type=SurfaceType.DETECTOR, semi_diameter=10))
    return stack


def test_vectorized_mesh_sag_matches_surface_prescription():
    radii = np.linspace(0, 8, 25).reshape(5, 5)
    for surface in (
        Surface(radius=0),
        Surface(radius=50),
        Surface(radius=-50, conic=-0.7, asphere_coeffs=[1e-6, -1e-9]),
    ):
        expected = np.vectorize(surface.sag)(radii)
        assert np.allclose(_surface_sag_values(surface, radii), expected, rtol=1e-12, atol=1e-12)


@pytest.mark.skipif(not (QT_AVAILABLE and HAS_MPL_QT), reason="Qt Matplotlib unavailable")
def test_fast_detail_setting_reduces_polygons_without_removing_solids():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    widget = Visualizer3DWidget()
    try:
        widget.update_layout(surface_stack=_stack(), rays=[])
        widget.canvas.draw()
        fast_solids = len(widget.ax.collections)
        fast_polygons = sum(len(getattr(c, "_segslices", [])) for c in widget.ax.collections)
        widget.set_mesh_quality("detailed")
        widget.canvas.draw()
        detailed_polygons = sum(len(getattr(c, "_segslices", [])) for c in widget.ax.collections)
        assert widget.mesh_quality == "detailed"
        assert len(widget.ax.collections) == fast_solids
        assert detailed_polygons > fast_polygons
        widget.set_mesh_quality("interactive")
        assert widget.combo_detail.currentIndex() == 0
    finally:
        widget.close()


@pytest.mark.skipif(not (QT_AVAILABLE and HAS_MPL_QT), reason="Qt Matplotlib unavailable")
def test_main_window_defers_3d_rebuild_until_3d_tab_is_active(monkeypatch):
    from optics_app.gui.main_window import MainWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    try:
        window.viewer_tabs.setCurrentWidget(window.dual_view)
        calls = []
        original = window.visualizer_3d.update_layout

        def counted(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        monkeypatch.setattr(window.visualizer_3d, "update_layout", counted)
        window.model.set_surface_stack(_stack())
        window.on_trace_rays()
        assert len(calls) == 0
        assert window.visualizer_3d.surface_stack is not None
        assert window.visualizer_3d.rays

        window.viewer_tabs.setCurrentWidget(window.visualizer_3d)
        assert len(calls) >= 1
        before = len(calls)
        window.on_trace_rays()
        assert len(calls) == before + 1
    finally:
        window.close()
