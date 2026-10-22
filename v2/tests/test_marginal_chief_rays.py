"""
Unit and integration tests for Marginal and Chief ray visualization in 2D Optics.
"""

import os
import pytest
import numpy as np

from optics_app.gui.qt_compat import QT_AVAILABLE, QtWidgets
from optics_app.gui.dual_view import DualViewWidget, HAS_MPL_QT
from optics_app.gui.visualizer_3d import Visualizer3DWidget
from optics_app.gui.main_window import MainWindow
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.system_config import SystemConfig, FieldPoint, FieldType
from optics_app.engine.ray import Ray


if QT_AVAILABLE:
    app = QtWidgets.QApplication.instance()
    if app is None:
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        app = QtWidgets.QApplication([])


def _build_singlet_stack() -> SurfaceStack:
    stack = SurfaceStack()
    stack.surfaces = [
        Surface(name="Object", surface_type=SurfaceType.OBJECT, thickness=50.0, glass="AIR", semi_diameter=10.0),
        Surface(name="Stop", surface_type=SurfaceType.STOP, thickness=10.0, glass="AIR", semi_diameter=8.0),
        Surface(name="Lens_Front", radius=35.0, thickness=5.0, glass="N-BK7", semi_diameter=12.7),
        Surface(name="Lens_Rear", radius=-35.0, thickness=40.0, glass="AIR", semi_diameter=12.7),
        Surface(name="Image", surface_type=SurfaceType.IMAGE, thickness=0.0, glass="AIR", semi_diameter=15.0),
    ]
    stack._update_accumulated_positions()
    return stack


def test_system_config_marginal_and_chief_ray_tagging():
    """Verify generate_input_rays marks marginal and chief rays accurately."""
    stack = _build_singlet_stack()
    cfg = SystemConfig()
    cfg.fields = [FieldPoint(x=0.0, y=0.0), FieldPoint(x=0.0, y=5.0)]
    cfg.field_type = FieldType.OBJECT_HEIGHT_MM
    cfg.num_rays = 7

    rays = cfg.generate_input_rays(surfaces=stack.surfaces, wavelengths=[0.5876], num_rays=7)
    assert len(rays) > 0

    chief_rays = [r for r in rays if getattr(r, "is_chief", False)]
    marginal_rays = [r for r in rays if getattr(r, "is_marginal", False)]

    # Every field has a chief ray (Px=0, Py=0)
    assert len(chief_rays) >= 2
    for cr in chief_rays:
        px, py = getattr(cr, "pupil_coord", (None, None))
        assert abs(px) < 1e-5
        assert abs(py) < 1e-5

    # On-axis field has marginal rays (Px=0, Py=±1.0)
    assert len(marginal_rays) >= 2
    upper = [r for r in marginal_rays if getattr(r, "is_upper_marginal", False)]
    lower = [r for r in marginal_rays if getattr(r, "is_lower_marginal", False)]
    assert len(upper) >= 1
    assert len(lower) >= 1


@pytest.mark.skipif(not (QT_AVAILABLE and HAS_MPL_QT), reason="Qt Matplotlib unavailable")
def test_dual_view_marginal_chief_toggle():
    """Verify DualViewWidget checkbox and setter for marginal & chief rays."""
    stack = _build_singlet_stack()
    cfg = SystemConfig()
    input_rays = cfg.generate_input_rays(surfaces=stack.surfaces, wavelengths=[0.5876], num_rays=7)
    from optics_app.engine.tracer import OpticalSystem
    sys_tracer = OpticalSystem(stack)
    rays = [sys_tracer.trace_ray(r) for r in input_rays]

    widget = DualViewWidget()
    widget.surface_stack = stack
    widget.rays = rays
    widget.rz_view_active = True
    widget.xz_view_active = True
    widget.update_layout(surface_stack=stack, rays=rays)

    # Defaults to False
    assert widget.show_marginal_chief is False
    if widget.cb_marginal_chief:
        assert widget.cb_marginal_chief.isChecked() is False

    signals = []
    widget.marginal_chief_visibility_changed.connect(lambda v: signals.append(v))

    # Enable marginal & chief ray highlighting
    widget.set_show_marginal_chief(True)
    assert widget.show_marginal_chief is True
    assert len(signals) == 1
    assert signals[0] is True
    if widget.cb_marginal_chief:
        assert widget.cb_marginal_chief.isChecked() is True

    # Check that ax_rz rendered LineCollection artists
    assert len(widget._ray_artists_rz) > 0

    # Test marginal_chief_only filtering
    widget.set_marginal_chief_only(True)
    assert widget.marginal_chief_only is True
    assert len(widget._ray_artists_rz) > 0

    widget.set_marginal_chief_only(False)
    widget.set_show_marginal_chief(False)
    assert widget.show_marginal_chief is False
    assert len(signals) == 2
    assert signals[1] is False


@pytest.mark.skipif(not (QT_AVAILABLE and HAS_MPL_QT), reason="Qt Matplotlib unavailable")
def test_visualizer_3d_marginal_chief_support():
    """Verify Visualizer3DWidget supports marginal and chief ray visualization."""
    stack = _build_singlet_stack()
    cfg = SystemConfig()
    rays = cfg.generate_input_rays(surfaces=stack.surfaces, wavelengths=[0.5876], num_rays=7)

    v3d = Visualizer3DWidget()
    v3d.surface_stack = stack
    v3d.rays = rays

    assert v3d.show_marginal_chief is False
    v3d.set_show_marginal_chief(True)
    assert v3d.show_marginal_chief is True

    v3d.set_marginal_chief_only(True)
    assert v3d.marginal_chief_only is True

    v3d.set_marginal_chief_only(False)
    v3d.set_show_marginal_chief(False)
    assert v3d.show_marginal_chief is False


@pytest.mark.skipif(not (QT_AVAILABLE and HAS_MPL_QT), reason="Qt Matplotlib unavailable")
def test_main_window_marginal_chief_integration():
    """Verify MainWindow menu action and synchronization with DualView."""
    window = MainWindow()
    try:
        assert hasattr(window, "act_show_marginal_chief")
        assert hasattr(window, "act_marginal_chief_only")
        assert window.act_show_marginal_chief.shortcut().toString() == "Ctrl+M"

        # Initially unchecked
        assert window.act_show_marginal_chief.isChecked() is False
        assert window.dual_view.show_marginal_chief is False

        # Trigger via action
        window.act_show_marginal_chief.setChecked(True)
        assert window.dual_view.show_marginal_chief is True
        if window.dual_view.cb_marginal_chief:
            assert window.dual_view.cb_marginal_chief.isChecked() is True

        # Trigger via dual_view checkbox
        if window.dual_view.cb_marginal_chief:
            window.dual_view.cb_marginal_chief.setChecked(False)
            assert window.act_show_marginal_chief.isChecked() is False

        # Test trace rays with marginal & chief active
        window.act_show_marginal_chief.setChecked(True)
        window.on_trace_rays()
        assert window._last_traced_rays is not None
        assert len(window._last_traced_rays) > 0

        # Marginal & Chief rays are marked in the traced set
        chiefs = [r for r in window._last_traced_rays if getattr(r, "is_chief", False)]
        marginals = [r for r in window._last_traced_rays if getattr(r, "is_marginal", False)]
        assert len(chiefs) > 0
        assert len(marginals) > 0
    finally:
        window.close()
