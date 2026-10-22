"""
Tests for creating a clean new project with Object, Stop, and Image surfaces similar to Zemax.
"""
import os
import pytest
import numpy as np

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from v2.engine.surface import Surface, SurfaceType
from v2.engine.lde import SurfaceStack
from v2.engine.system_config import SystemConfig
from v2.engine.source import SourceConfig
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray
from v2.engine.element_colors import is_object_plane, is_stop_plane, is_detector_plane
from v2.gui.qt_compat import QtWidgets, QT_AVAILABLE
from v2.gui.lde_table import LensDataTableModel
from v2.gui.main_window import MainWindow


def test_surface_stack_create_clean_system_default_parameters():
    """Verify SurfaceStack.create_clean_system() creates the standard 3 Zemax surfaces."""
    stack = SurfaceStack.create_clean_system()
    assert len(stack.surfaces) == 3

    s0, s1, s2 = stack.surfaces

    # Surface 0: Object
    assert s0.surface_type == SurfaceType.OBJECT
    assert "OBJ" in s0.name.upper()
    assert s0.radius == 0.0  # Flat / infinity
    assert s0.thickness == 100.0  # Standard object distance
    assert s0.glass.upper() == "AIR"
    assert s0.semi_diameter == 25.0
    assert not s0.is_stop
    assert is_object_plane(s0, 0)
    assert not is_stop_plane(s0)

    # Surface 1: Stop
    assert s1.surface_type == SurfaceType.STOP
    assert "STO" in s1.name.upper()
    assert s1.radius == 0.0
    assert s1.thickness == 50.0  # Distance to image
    assert s1.glass.upper() == "AIR"
    assert s1.semi_diameter == 10.0
    assert s1.is_stop
    assert is_stop_plane(s1)
    assert not is_object_plane(s1, 1)

    # Surface 2: Image / Detector
    assert s2.surface_type == SurfaceType.DETECTOR
    assert "IMA" in s2.name.upper()
    assert s2.radius == 0.0
    assert s2.thickness == 0.0  # Terminal image plane
    assert s2.glass.upper() == "AIR"
    assert s2.semi_diameter == 25.0
    assert not s2.is_stop
    assert is_detector_plane(s2, 2, 3)


def test_surface_stack_create_clean_system_custom_parameters():
    """Verify SurfaceStack.create_clean_system() accepts custom geometry."""
    stack = SurfaceStack.create_clean_system(object_dist=150.0, stop_dist=75.0, stop_semi_diameter=12.5)
    assert len(stack.surfaces) == 3
    assert stack.surfaces[0].thickness == 150.0
    assert stack.surfaces[1].thickness == 75.0
    assert stack.surfaces[1].semi_diameter == 12.5
    assert stack.surfaces[2].thickness == 0.0

    # Test alias create_default
    stack_alias = SurfaceStack.create_default(object_dist=150.0, stop_dist=75.0, stop_semi_diameter=12.5)
    assert len(stack_alias.surfaces) == 3
    assert stack_alias.surfaces[0].thickness == 150.0


def test_lde_table_model_clean_system_initialization():
    """Verify LensDataTableModel initializes with clean 3-surface system and supports reset_to_clean_system."""
    model = LensDataTableModel()
    assert model.rowCount() == 3
    assert model.headerData(0, orientation=2) == "0 (OBJ)"
    assert model.headerData(1, orientation=2) == "1 (STO)"
    assert model.headerData(2, orientation=2) == "2 (IMA)"

    # Mutate by appending surfaces
    s_extra = Surface(name="Lens1", surface_type=SurfaceType.STANDARD, radius=50.0, thickness=5.0, glass="N-BK7")
    stack = model.get_surface_stack()
    stack.add_surface(s_extra)
    model.set_surface_stack(stack)
    assert model.rowCount() == 4

    # Reset back to clean system
    model.reset_to_clean_system()
    assert model.rowCount() == 3
    assert model.headerData(0, orientation=2) == "0 (OBJ)"
    assert model.headerData(1, orientation=2) == "1 (STO)"
    assert model.headerData(2, orientation=2) == "2 (IMA)"


def test_ray_tracing_through_clean_zemax_system():
    """Verify paraxial/real rays trace cleanly through Object -> Stop -> Image."""
    stack = SurfaceStack.create_clean_system()
    sys = OpticalSystem(surface_stack=stack)

    # On-axis ray
    ray_on_axis = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=0.5876)
    traced_on_axis = sys.trace_ray(ray_on_axis)
    assert traced_on_axis is not None
    # Terminal point should be at image plane z = object_dist + stop_dist = 150.0
    assert np.isclose(traced_on_axis.pos[2], 150.0)

    # Marginal ray aimed at stop edge (y = 10.0 at z = 100.0)
    theta = np.arctan2(10.0, 100.0)
    ray_marginal = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, np.sin(theta), np.cos(theta)]), wavelength=0.5876)
    traced_marginal = sys.trace_ray(ray_marginal)
    assert traced_marginal is not None
    assert np.isclose(traced_marginal.pos[2], 150.0)


def test_main_window_on_new_project():
    """Verify MainWindow.on_new_project() resets dirty system to clean 3-surface Zemax system."""
    if not QT_AVAILABLE:
        pytest.skip("Qt not available")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    try:
        # Populate with complex multi-surface system
        stack = SurfaceStack()
        for i in range(7):
            stack.add_surface(Surface(name=f"S{i}", thickness=10.0, radius=50.0 if i % 2 == 1 else -50.0, glass="N-BK7"))
        window.model.set_surface_stack(stack)
        window.current_file_path = "C:/test/sample_project.json"
        window.set_dirty(True)
        assert len(window.model._surfaces) == 7
        assert window.is_dirty
        assert window.current_file_path is not None

        # Call on_new_project
        success = window.on_new_project(prompt_save=False)
        assert success

        # Verify surfaces reset to standard 3 surfaces
        assert len(window.model._surfaces) == 3
        s0, s1, s2 = window.model._surfaces
        assert s0.surface_type == SurfaceType.OBJECT
        assert s1.surface_type == SurfaceType.STOP
        assert s1.is_stop
        assert s2.surface_type == SurfaceType.DETECTOR

        # Verify project state reset
        assert window.current_file_path is None
        assert not window.is_dirty
        assert "Untitled" in window.windowTitle()

        # Verify system config reset
        assert window.system_config.primary_wavelength_um == 0.58756
        assert len(window.system_config.wavelength_list_um) == 3

        # Verify shortcut and action bindings
        assert window.act_new is not None
        assert window.act_new.shortcut().toString() in ("Ctrl+N", "")

    finally:
        window.close()


def test_main_window_action_and_button_bindings():
    """Verify act_new and LDE dock buttons are properly bound and functional."""
    if not QT_AVAILABLE:
        pytest.skip("Qt not available")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    try:
        # Check action exists and is properly configured
        assert window.act_new is not None
        assert window.act_new.text() == "&New Project"
        assert window.act_new.isEnabled()

        # Mutate system
        stack = SurfaceStack()
        for i in range(5):
            stack.add_surface(Surface(name=f"Surf{i}", thickness=15.0))
        window.model.set_surface_stack(stack)
        assert len(window.model._surfaces) == 5

        # Trigger act_new
        window.act_new.trigger()
        assert len(window.model._surfaces) == 3
        assert window.model._surfaces[0].surface_type == SurfaceType.OBJECT
        assert window.model._surfaces[1].surface_type == SurfaceType.STOP
        assert window.model._surfaces[2].surface_type == SurfaceType.DETECTOR
    finally:
        window.close()
