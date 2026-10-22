"""
Unit tests for 3D Visualizer No Transparency (Opaque Mode) and Blender-Style View Switching.
Verifies:
1. Opaque mode toggling, signal emission, and solid alpha mesh rendering.
2. Orthographic plane switching: Front (XY), Back (-XY), Top (XZ), Bottom (-XZ), Side (ZY), Opposite Side (-ZY).
3. In-between diagonal angles (45° turntable and elevation angles, 4 isometric corners).
4. Blender numpad navigation hotkeys and ease-out transitions.
5. MainWindow View menu action synchronization.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock

from optics_app.gui.qt_compat import QT_AVAILABLE, Qt, QtWidgets
from optics_app.gui.visualizer_3d import Visualizer3DWidget, HAS_MPL_QT
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.surface import Surface, SurfaceType


def _ensure_qapp():
    if QT_AVAILABLE and hasattr(QtWidgets, "QApplication"):
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])
        return app
    return None


def build_test_stack():
    """Build a multi-element optical system with lenses, stop, and mirrors."""
    stack = SurfaceStack()
    s0 = Surface(name="OBJ", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=20.0, semi_diameter=5.0)
    s1 = Surface(name="L1_Front", surface_type=SurfaceType.STANDARD, radius=50.0, thickness=5.0, glass="N-BK7", semi_diameter=12.0)
    s2 = Surface(name="L1_Rear", surface_type=SurfaceType.STANDARD, radius=-50.0, thickness=15.0, glass="AIR", semi_diameter=12.0)
    s3 = Surface(name="STOP", surface_type=SurfaceType.STOP, radius=0.0, thickness=10.0, semi_diameter=6.0)
    s4 = Surface(name="Fold_Mirror", surface_type=SurfaceType.MIRROR, radius=0.0, thickness=20.0, glass="MIRROR", semi_diameter=10.0)
    s5 = Surface(name="DET", surface_type=SurfaceType.DETECTOR, radius=0.0, thickness=0.0, semi_diameter=8.0)
    for s in [s0, s1, s2, s3, s4, s5]:
        stack.add_surface(s)
    stack._update_accumulated_positions()
    return stack


@pytest.fixture
def vis_widget():
    """Create a test Visualizer3DWidget instance."""
    _ensure_qapp()
    stack = build_test_stack()
    widget = Visualizer3DWidget(auto_refresh=False)
    widget.surface_stack = stack
    if widget.ax is not None:
        widget.update_layout()
    return widget


def test_no_transparency_toggle_and_signal(vis_widget):
    """Test no_transparency state, toggle_no_transparency, and signal emission."""
    assert vis_widget.no_transparency is False

    signals_received = []
    if QT_AVAILABLE and hasattr(vis_widget, "transparency_changed"):
        vis_widget.transparency_changed.connect(lambda val: signals_received.append(val))

    # Enable No Transparency (Opaque Mode)
    vis_widget.set_no_transparency(True)
    assert vis_widget.no_transparency is True
    if hasattr(vis_widget, "cb_no_transparency") and vis_widget.cb_no_transparency:
        assert vis_widget.cb_no_transparency.isChecked() is True
    if signals_received:
        assert signals_received[-1] is True

    # Check status badge contains Opaque
    if vis_widget.lbl_status:
        assert "Opaque" in vis_widget.lbl_status.text()

    # Toggle off via toggle_no_transparency()
    new_state = vis_widget.toggle_no_transparency()
    assert new_state is False
    assert vis_widget.no_transparency is False
    if hasattr(vis_widget, "cb_no_transparency") and vis_widget.cb_no_transparency:
        assert vis_widget.cb_no_transparency.isChecked() is False
    if signals_received:
        assert signals_received[-1] is False


def test_mesh_alphas_in_opaque_and_transparent_modes(vis_widget):
    """Verify that mesh alpha is 1.0 in opaque mode and semi-transparent in transparent mode."""
    if not (QT_AVAILABLE and HAS_MPL_QT) or vis_widget.ax is None:
        pytest.skip("Matplotlib 3D Qt not available")

    # Mode 1: Transparent (default)
    vis_widget.set_no_transparency(False)
    vis_widget.update_layout()

    # Find surface collections or Poly3DCollections in axes
    has_transparent_mesh = False
    for coll in vis_widget.ax.collections:
        facecolors = getattr(coll, '_facecolors2d', getattr(coll, '_facecolor3d', None))
        if facecolors is not None and len(facecolors) > 0:
            alpha = facecolors[0][3]
            if alpha < 0.95:
                has_transparent_mesh = True
                break
    assert has_transparent_mesh is True, "Expected semi-transparent meshes in default transparent mode"

    # Mode 2: Opaque (No transparency)
    vis_widget.set_no_transparency(True)
    vis_widget.update_layout()

    for coll in vis_widget.ax.collections:
        facecolors = getattr(coll, '_facecolors2d', getattr(coll, '_facecolor3d', None))
        if facecolors is not None and len(facecolors) > 0:
            for fc in facecolors:
                # In opaque mode, all optical component mesh alphas must be 1.0 (fully solid)
                assert fc[3] == pytest.approx(1.0, abs=0.01)


def test_blender_orthographic_planes(vis_widget):
    """Test switching directly to all standard orthographic and inverted planes."""
    if vis_widget.ax is None:
        pytest.skip("Matplotlib 3D not available")

    # Front (X-Y aperture plane): elev = 0, azim = 0
    vis_widget.set_view_front(animate=False)
    assert vis_widget.ax.elev == pytest.approx(0.0)
    assert vis_widget.ax.azim == pytest.approx(0.0)

    # Back (-X-Y plane): elev = 0, azim = 180
    vis_widget.set_view_back(animate=False)
    assert vis_widget.ax.elev == pytest.approx(0.0)
    assert abs(vis_widget.ax.azim) == pytest.approx(180.0)

    # Side (Z-Y plane): elev = 0, azim = -90
    vis_widget.set_view_side(animate=False)
    assert vis_widget.ax.elev == pytest.approx(0.0)
    assert vis_widget.ax.azim == pytest.approx(-90.0)

    # Opposite Side (-Z-Y plane): elev = 0, azim = 90
    vis_widget.set_view_side_opposite(animate=False)
    assert vis_widget.ax.elev == pytest.approx(0.0)
    assert vis_widget.ax.azim == pytest.approx(90.0)

    # Top (X-Z plane): elev = 90, azim = -90
    vis_widget.set_view_top(animate=False)
    assert vis_widget.ax.elev == pytest.approx(90.0)
    assert vis_widget.ax.azim == pytest.approx(-90.0)

    # Bottom (-X-Z plane): elev = -90, azim = -90
    vis_widget.set_view_bottom(animate=False)
    assert vis_widget.ax.elev == pytest.approx(-90.0)
    assert vis_widget.ax.azim == pytest.approx(-90.0)

    # Isometric (Default): elev = 20, azim = -60
    vis_widget.set_view_isometric(animate=False)
    assert vis_widget.ax.elev == pytest.approx(20.0)
    assert vis_widget.ax.azim == pytest.approx(-60.0)


def test_in_between_diagonal_views(vis_widget):
    """Test setting in-between diagonal angles."""
    if vis_widget.ax is None:
        pytest.skip("Matplotlib 3D not available")

    vis_widget.set_view_in_between("front_side", animate=False)
    assert vis_widget.ax.elev == pytest.approx(0.0)
    assert vis_widget.ax.azim == pytest.approx(-45.0)

    vis_widget.set_view_in_between("side_back", animate=False)
    assert vis_widget.ax.elev == pytest.approx(0.0)
    assert vis_widget.ax.azim == pytest.approx(-135.0)

    vis_widget.set_view_in_between("back_opposite", animate=False)
    assert vis_widget.ax.elev == pytest.approx(0.0)
    assert vis_widget.ax.azim == pytest.approx(135.0)

    vis_widget.set_view_in_between("opposite_front", animate=False)
    assert vis_widget.ax.elev == pytest.approx(0.0)
    assert vis_widget.ax.azim == pytest.approx(45.0)

    vis_widget.set_view_in_between("top_front", animate=False)
    assert vis_widget.ax.elev == pytest.approx(45.0)
    assert vis_widget.ax.azim == pytest.approx(0.0)

    vis_widget.set_view_in_between("iso_ne", animate=False)
    assert vis_widget.ax.elev == pytest.approx(30.0)
    assert vis_widget.ax.azim == pytest.approx(-45.0)


def test_orbit_and_flip_view(vis_widget):
    """Test incremental turntable orbiting and 180° flip view."""
    if vis_widget.ax is None:
        pytest.skip("Matplotlib 3D not available")

    vis_widget.set_view_front(animate=False)  # elev=0, azim=0
    assert vis_widget.ax.azim == pytest.approx(0.0)

    # Orbit right +15°
    vis_widget.orbit_horizontal(15.0, animate=False)
    assert vis_widget.ax.azim == pytest.approx(15.0)

    # Orbit left -30° -> -15°
    vis_widget.orbit_horizontal(-30.0, animate=False)
    assert vis_widget.ax.azim == pytest.approx(-15.0)

    # Orbit up +15°
    vis_widget.orbit_vertical(15.0, animate=False)
    assert vis_widget.ax.elev == pytest.approx(15.0)

    # Flip 180°
    vis_widget.flip_view(animate=False)
    assert vis_widget.ax.azim == pytest.approx(165.0)


def test_blender_keyboard_navigation(vis_widget):
    """Test Blender keyboard shortcuts dispatch to the correct view methods."""
    if vis_widget.ax is None:
        pytest.skip("Matplotlib 3D not available")

    # Key '1' -> Front
    vis_widget._handle_navigation_key('1', is_ctrl=False)
    assert vis_widget.ax.elev == pytest.approx(0.0)
    assert vis_widget.ax.azim == pytest.approx(0.0)

    # Ctrl+'1' -> Back
    vis_widget._handle_navigation_key('1', is_ctrl=True)
    assert vis_widget.ax.elev == pytest.approx(0.0)
    assert abs(vis_widget.ax.azim) == pytest.approx(180.0)

    # Key '3' -> Side
    vis_widget._handle_navigation_key('3', is_ctrl=False)
    assert vis_widget.ax.elev == pytest.approx(0.0)
    assert vis_widget.ax.azim == pytest.approx(-90.0)

    # Ctrl+'3' -> Opposite Side
    vis_widget._handle_navigation_key('3', is_ctrl=True)
    assert vis_widget.ax.elev == pytest.approx(0.0)
    assert vis_widget.ax.azim == pytest.approx(90.0)

    # Key '7' -> Top
    vis_widget._handle_navigation_key('7', is_ctrl=False)
    assert vis_widget.ax.elev == pytest.approx(90.0)
    assert vis_widget.ax.azim == pytest.approx(-90.0)

    # Ctrl+'7' -> Bottom
    vis_widget._handle_navigation_key('7', is_ctrl=True)
    assert vis_widget.ax.elev == pytest.approx(-90.0)
    assert vis_widget.ax.azim == pytest.approx(-90.0)

    # Key '0' -> Isometric
    vis_widget._handle_navigation_key('0', is_ctrl=False)
    assert vis_widget.ax.elev == pytest.approx(20.0)
    assert vis_widget.ax.azim == pytest.approx(-60.0)

    # Key 't' -> Toggle Opaque Mode
    init_no_transp = vis_widget.no_transparency
    vis_widget._handle_navigation_key('t', is_ctrl=False)
    assert vis_widget.no_transparency != init_no_transp


def test_main_window_no_transparency_action():
    """Test MainWindow View menu action integration with 3D Visualizer."""
    if not QT_AVAILABLE:
        pytest.skip("Qt not available")

    from optics_app.gui.main_window import MainWindow
    window = MainWindow()

    assert hasattr(window, "act_3d_no_transp")
    assert window.act_3d_no_transp is not None
    assert window.act_3d_no_transp.isChecked() is False
    assert window.visualizer_3d.no_transparency is False

    # Trigger action in View menu
    window.act_3d_no_transp.setChecked(True)
    assert window.visualizer_3d.no_transparency is True

    # Toggle inside visualizer widget -> syncs back to menu action
    window.visualizer_3d.set_no_transparency(False)
    assert window.act_3d_no_transp.isChecked() is False
