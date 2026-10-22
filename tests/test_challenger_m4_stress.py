"""
Adversarial Stress Test Suite for Milestone 4 Desktop GUI Package.
Tests LensDataTableModel, MainWindow, DualViewWidget, and AnalysisTabsWidget.
"""

import os
import tempfile
import pytest
import numpy as np

from optics_app.gui.qt_compat import QT_AVAILABLE, QModelIndex, QtWidgets

if QT_AVAILABLE and hasattr(QtWidgets, "QApplication"):
    try:
        _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    except Exception:
        pass

if hasattr(QtWidgets, "QProgressDialog") and not hasattr(QtWidgets.QProgressDialog, "setWindowTitle"):
    QtWidgets.QProgressDialog.setWindowTitle = lambda self, t: None

from optics_app.gui.lde_table import LensDataTableModel
from optics_app.gui.main_window import MainWindow
from optics_app.gui.dual_view import DualViewWidget
from optics_app.gui.analysis_tabs import AnalysisTabsWidget, SpotDiagramTab, RayFanTab, GratingDispersionTab, CCDHeatmapTab
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.ray import Ray
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ccd import CCDSensor
from optics_app.analysis.spot import SpotDiagram, generate_spot_diagram_from_system


# ============================================================================
# Scenario 1: LensDataTableModel Stress Tests
# ============================================================================

def test_scenario1_lde_table_20_surfaces_sync_and_validation():
    """Test adding 20 surfaces, modifying parameters, setting invalid data, and verifying SurfaceStack sync."""
    model = LensDataTableModel()
    initial_count = model.rowCount()
    assert initial_count == 3, f"Default count should be 3, got {initial_count}"

    # 1. Add 20 surfaces sequentially
    for i in range(20):
        res = model.insertRows(position=1 + i, count=1)
        assert res is True, f"Failed to insert surface at row {1+i}"

    assert model.rowCount() == 23, f"Expected 23 rows, got {model.rowCount()}"

    # 2. Modify radii, thicknesses, glass, semi-diameters, conic, decenters, tilts for all 20 added surfaces
    for i in range(1, 21):
        radius = 40.0 + i * 2.5
        thickness = 3.0 + (i % 5)
        glass = "N-BK7" if i % 2 == 0 else "N-SF11"
        semi_diam = 15.0 + i * 0.5
        conic = (i - 10) * 0.1
        dec_x = (i - 10) * 0.2
        dec_y = (i - 10) * -0.1
        tilt_x = (i - 10) * 0.5

        assert model.setData(i, 2, radius) is True
        assert model.setData(i, 3, thickness) is True
        assert model.setData(i, 4, glass) is True
        assert model.setData(i, 5, semi_diam) is True
        assert model.setData(i, 6, conic) is True
        assert model.setData(i, 7, dec_x) is True
        assert model.setData(i, 8, dec_y) is True
        assert model.setData(i, 9, tilt_x) is True

    # 3. Verify SurfaceStack synchronization
    stack = model.get_surface_stack()
    assert len(stack.surfaces) == 23, f"SurfaceStack size mismatch: {len(stack.surfaces)} vs 23"

    for i in range(1, 21):
        surf = stack.surfaces[i]
        expected_radius = 40.0 + i * 2.5
        expected_thickness = 3.0 + (i % 5)
        expected_glass = "N-BK7" if i % 2 == 0 else "N-SF11"
        expected_semi_diam = 15.0 + i * 0.5
        expected_conic = (i - 10) * 0.1
        expected_dec_x = (i - 10) * 0.2

        assert abs(surf.radius - expected_radius) < 1e-5
        assert abs(surf.thickness - expected_thickness) < 1e-5
        assert surf.glass == expected_glass
        assert abs(surf.semi_diameter - expected_semi_diam) < 1e-5
        assert abs(surf.conic - expected_conic) < 1e-5
        assert abs(surf.decenter_x - expected_dec_x) < 1e-5

    # 4. Stress-test invalid data types on numeric columns
    invalid_inputs = ["invalid_str", "abc", None, [1, 2], {"a": 1}]
    for col in [2, 3, 5, 6, 7, 8, 9, 10, 11, 12]:
        for inv in invalid_inputs:
            res = model.setData(1, col, inv)
            assert res is False, f"setData should return False for invalid input '{inv}' in col {col}"

    # 5. String float conversion tests ("100.5", "inf", "infinity")
    assert model.setData(1, 2, "100.5") is True
    assert model.data(1, 2) == 100.5

    assert model.setData(1, 2, "inf") is True
    assert model.data(1, 2) == 0.0

    assert model.setData(1, 2, "infinity") is True
    assert model.data(1, 2) == 0.0

    # 6. Test out-of-bounds row and column access
    assert model.setData(-1, 2, 10.0) is False
    assert model.setData(100, 2, 10.0) is False
    assert model.setData(1, -1, 10.0) is False
    assert model.setData(1, 50, 10.0) is False
    assert model.data(-1, 0) is None
    assert model.data(100, 0) is None

    # 7. Test removing surfaces down to 1
    for _ in range(20):
        assert model.removeRows(1, 1) is True
    assert model.rowCount() == 3

    # Re-verify stack sync after removal
    stack_after = model.get_surface_stack()
    assert len(stack_after.surfaces) == 3


# ============================================================================
# Scenario 2: MainWindow Integration & Action Triggers
# ============================================================================

def test_scenario2_main_window_action_triggers():
    """Test MainWindow file/action handlers: Open Zemax .zmx, Export 3D STEP, Trace Rays, CAD surface loading."""
    window = MainWindow()

    # 1. Test on_trace_rays & on_refresh_all
    assert window.on_trace_rays() is True
    assert window.on_refresh_all() is True

    # 2. Test on_open_zmx with valid temporary .zmx file
    zmx_content = """! Temporary test Zemax file
SURF 0
  TYPE OBJ
  THIC 100.000000
SURF 1
  CURV 0.02
  THIC 10.000000
  GLAS N-BK7 1.5168 64.17
  DIAM 25.000000
SURF 2
  CURV -0.015
  THIC 80.000000
  GLAS AIR
  DIAM 25.000000
SURF 3
  TYPE IMA
  THIC 0.000000
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.zmx', delete=False) as f:
        f.write(zmx_content)
        zmx_path = f.name

    try:
        res_zmx = window.on_open_zmx(zmx_path)
        assert res_zmx is True, "on_open_zmx failed on valid .zmx file"
        assert window.model.rowCount() == 4, f"Expected 4 surfaces from .zmx, got {window.model.rowCount()}"
    finally:
        if os.path.exists(zmx_path):
            os.remove(zmx_path)

    # Test on_open_zmx with non-existent file
    assert window.on_open_zmx("non_existent_file_path_xyz.zmx") is False

    # 3. Test on_export_step with valid temporary output path
    with tempfile.NamedTemporaryFile(suffix='.step', delete=False) as f:
        step_out_path = f.name
        f.close()

    try:
        res_step = window.on_export_step(step_out_path)
        assert res_step is True, "on_export_step failed"
        if hasattr(window, "cad_worker") and window.cad_worker is not None:
            window.cad_worker.wait(60000)
        assert os.path.exists(step_out_path)
        assert os.path.getsize(step_out_path) > 0, "Exported STEP file is empty"
    finally:
        if os.path.exists(step_out_path):
            os.remove(step_out_path)

    # 4. Test on_load_cad_surface with sample CAD file if available
    sample_step = os.path.join(os.path.dirname(__file__), "sample_cad", "spherical_lens.step")
    if os.path.exists(sample_step):
        res_cad = window.on_load_cad_surface(sample_step)
        assert res_cad is True, "on_load_cad_surface failed on valid CAD file"

    # Test on_load_cad_surface with invalid file
    assert window.on_load_cad_surface("invalid_cad_path.step") is False

    # 5. Test surface insert/delete actions via MainWindow
    cur_rows = window.model.rowCount()
    assert window.on_insert_surface() is True
    assert window.model.rowCount() == cur_rows + 1

    assert window.on_delete_surface() is True
    assert window.model.rowCount() == cur_rows


# ============================================================================
# Scenario 3: DualViewWidget Layout & Multi-Wavelength Rendering
# ============================================================================

def test_scenario3_dual_view_complex_stack_multi_wavelength():
    """Test DualViewWidget with complex tilted/decentered surface stack and multi-wavelength rays."""
    dual_view = DualViewWidget()

    # Create a complex tilted/decentered surface stack
    stack = SurfaceStack()

    surf_obj = Surface(name="OBJ", surface_type=SurfaceType.OBJECT, radius=0.0, thickness=50.0, glass="AIR", semi_diameter=20.0)
    stack.add_surface(surf_obj)

    surf1 = Surface(name="L1_Front", surface_type=SurfaceType.ASPHERIC, radius=45.0, thickness=8.0, glass="N-BK7", semi_diameter=25.0, decenter_x=2.0, decenter_y=-1.5, tilt_x=5.0)
    stack.add_surface(surf1)

    surf2 = Surface(name="L1_Back", surface_type=SurfaceType.STANDARD, radius=-60.0, thickness=15.0, glass="AIR", semi_diameter=25.0, decenter_x=2.0, decenter_y=-1.5, tilt_x=5.0)
    stack.add_surface(surf2)

    surf_grat = Surface(name="Grating_Sto", surface_type=SurfaceType.GRATING, radius=0.0, thickness=20.0, glass="AIR", semi_diameter=20.0, decenter_x=0.0, decenter_y=0.0, tilt_x=-10.0)
    setattr(surf_grat, 'grating_lines_per_mm', 600.0)
    stack.add_surface(surf_grat)

    surf_det = Surface(name="IMA", surface_type=SurfaceType.DETECTOR, radius=0.0, thickness=0.0, glass="AIR", semi_diameter=30.0)
    stack.add_surface(surf_det)

    # Generate synthetic multi-wavelength rays with histories
    wavelengths = [0.450, 0.546, 0.650]
    rays = []
    for wl in wavelengths:
        for py in np.linspace(-10.0, 10.0, 5):
            r = Ray(pos=np.array([0.0, py, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl)
            # Add synthetic history steps
            r.history = [
                {"z_global": 0.0, "y_global": py, "x_global": 0.0},
                {"z_global": 50.0, "y_global": py * 0.9 + 1.0, "x_global": 0.5},
                {"z_global": 58.0, "y_global": py * 0.85 + 1.2, "x_global": 0.8},
                {"z_global": 73.0, "y_global": py * 0.6 + (wl - 0.55)*20.0, "x_global": 1.0},
                {"z_global": 93.0, "y_global": py * 0.2 + (wl - 0.55)*40.0, "x_global": 1.5}
            ]
            rays.append(r)

    # Invoke update_layout with complex stack and multi-wavelength rays
    dual_view.update_layout(surface_stack=stack, rays=rays)

    # Test toggling views
    dual_view.set_rz_view_active(True)
    dual_view.set_xy_view_active(False)
    dual_view.refresh_layout()

    dual_view.set_rz_view_active(False)
    dual_view.set_xy_view_active(True)
    dual_view.refresh_layout()

    dual_view.set_rz_view_active(True)
    dual_view.set_xy_view_active(True)
    dual_view.refresh_layout()

    # Test update_system method
    res_sys = dual_view.update_system(stack)
    assert res_sys is True


# ============================================================================
# Scenario 4: AnalysisTabsWidget High-Density Diagnostic Updates
# ============================================================================

def test_scenario4_analysis_tabs_high_density_updates():
    """Test AnalysisTabsWidget updating all 4 tabs with high-density ray trace data."""
    # Build a doublet optical system for realistic analysis trace
    stack = SurfaceStack()
    stack.add_surface(Surface(name="OBJ", surface_type=SurfaceType.OBJECT, radius=0.0, thickness=100.0, glass="AIR", semi_diameter=25.0))
    stack.add_surface(Surface(name="S1", surface_type=SurfaceType.STANDARD, radius=50.0, thickness=8.0, glass="N-BK7", semi_diameter=20.0))
    stack.add_surface(Surface(name="S2", surface_type=SurfaceType.STANDARD, radius=-40.0, thickness=4.0, glass="N-SF11", semi_diameter=20.0))
    stack.add_surface(Surface(name="S3", surface_type=SurfaceType.STANDARD, radius=-120.0, thickness=90.0, glass="AIR", semi_diameter=20.0))
    stack.add_surface(Surface(name="IMA", surface_type=SurfaceType.DETECTOR, radius=0.0, thickness=0.0, glass="AIR", semi_diameter=25.0))

    system = OpticalSystem(stack)
    model = LensDataTableModel(stack)

    analysis_suite = AnalysisTabsWidget(model=model)

    # Test 1: Spot Diagram Tab with SpotDiagram object
    spot = generate_spot_diagram_from_system(system, num_rays=300)
    assert len(spot.x) > 0, "Spot diagram generated 0 ray hits"

    tab_spot = getattr(analysis_suite, "tab_spot", SpotDiagramTab())
    tab_spot.update_plot(system=system, spot_diagram=spot)

    # Test 2: Ray Fan Tab
    tab_fan = getattr(analysis_suite, "tab_rayfan", RayFanTab())
    tab_fan.update_plot(system=system)

    # Test 3: Grating Dispersion Tab
    tab_disp = getattr(analysis_suite, "tab_dispersion", GratingDispersionTab())
    tab_disp.update_plot(system=system)

    # Test 4: CCD Heatmap Tab
    ccd = CCDSensor(width_mm=12.0, height_mm=12.0, pixel_cols=80, pixel_rows=80)
    ccd.deposit_rays(spot.x, spot.y)
    assert ccd.total_rays_deposited > 0, "CCD sensor received 0 rays"

    tab_ccd = getattr(analysis_suite, "tab_ccd", CCDHeatmapTab())
    tab_ccd.update_plot(system=system, ccd_sensor=ccd)

    # Test 5: Switch tabs via set_active_tab and call refresh_analysis
    for tab_name in ["Spot Diagram", "Ray Fan", "Grating Dispersion", "CCD Heatmap"]:
        res = analysis_suite.set_active_tab(tab_name)
        assert res is True, f"Failed to activate tab {tab_name}"

    assert analysis_suite.set_active_tab("NonExistentTab") is False

    # Test 6: Trigger update_analysis with system
    assert analysis_suite.update_analysis(system) is True
