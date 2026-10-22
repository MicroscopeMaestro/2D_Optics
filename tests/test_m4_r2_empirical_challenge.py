"""
Adversarial Stress Test Suite for Milestone 4 Iteration 2 Desktop GUI Package.
Covers LensDataTableModel 14-column editing, conic preservation, row moving, NaN/Inf rejection,
MainWindow getters/action aliases/file actions (.zmx import, .step export),
and DualViewWidget & AnalysisTabsWidget ray trace visualization updates.
"""

import os
import math
import tempfile
import pytest
import numpy as np

from optics_app.gui.lde_table import LensDataTableModel, LensDataEditorWidget
from optics_app.gui.main_window import MainWindow
from optics_app.gui.dual_view import DualViewWidget
from optics_app.gui.analysis_tabs import (
    AnalysisTabsWidget, SpotDiagramTab, RayFanTab, GratingDispersionTab, CCDHeatmapTab
)
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.ray import Ray
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ccd import CCDSensor
from optics_app.analysis.spot import generate_spot_diagram_from_system
from optics_app.gui.qt_compat import QT_AVAILABLE, QtWidgets

if QT_AVAILABLE and hasattr(QtWidgets, "QApplication"):
    try:
        _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    except Exception:
        pass


# ============================================================================
# Scenario 1: LensDataTableModel Stress Tests
# ============================================================================

def test_s1_1_14_column_grid_editing():
    """Verify LensDataTableModel grid has 14+ columns and supports editing columns."""
    model = LensDataTableModel()
    assert model.columnCount() >= 14, f"Expected at least 14 columns, got {model.columnCount()}"

    # Verify column header names
    expected_headers = [
        "Surface", "Type", "Radius (mm)", "Thickness (mm)",
        "Glass", "Semi-Diameter (mm)", "Conic", "Decenter X",
        "Decenter Y", "Tilt X", "Tilt Y", "Tilt Z", "Grating Lines", "CAD File"
    ]
    for c, header in enumerate(expected_headers):
        assert model.headerData(c, None) == header, f"Header col {c} mismatch: {model.headerData(c, None)} vs {header}"

    # Insert a surface at row 1
    model.insertRows(1, 1)
    r = 1

    # Edit all 14 columns
    col_values = [
        (0, "Lens1_Front"),
        (1, "Aspheric"),
        (2, 42.5),
        (3, 7.5),
        (4, "N-BK7"),
        (5, 22.0),
        (6, -0.65),
        (7, 1.2),
        (8, -0.8),
        (9, 2.5),
        (10, -1.0),
        (11, 0.5),
        (12, 300.0),
        (13, "c:/shapes/lens1.step")
    ]

    for col, val in col_values:
        res = model.setData(r, col, val)
        assert res is True, f"Failed to setData on column {col} with value {val}"
        ret_val = model.data(r, col)
        if isinstance(val, float):
            assert abs(ret_val - val) < 1e-5, f"Column {col} stored value {ret_val} mismatch with {val}"
        else:
            assert ret_val == val, f"Column {col} stored value {ret_val} mismatch with {val}"

    # Verify underlying SurfaceStack synchronization
    stack = model.get_surface_stack()
    surf = stack.surfaces[r]
    assert surf.name == "Lens1_Front"
    assert surf.surface_type == SurfaceType.ASPHERIC
    assert abs(surf.radius - 42.5) < 1e-5
    assert abs(surf.thickness - 7.5) < 1e-5
    assert surf.glass == "N-BK7"
    assert abs(surf.semi_diameter - 22.0) < 1e-5
    assert abs(surf.conic - (-0.65)) < 1e-5
    assert abs(surf.decenter_x - 1.2) < 1e-5
    assert abs(surf.decenter_y - (-0.8)) < 1e-5
    assert abs(surf.tilt_x - 2.5) < 1e-5
    assert abs(surf.tilt_y - (-1.0)) < 1e-5
    assert abs(surf.tilt_z - 0.5) < 1e-5
    assert abs(surf.grating_lines_per_mm - 300.0) < 1e-5
    assert getattr(surf, 'cad_file_path', '') == "c:/shapes/lens1.step"


def test_s1_2_conic_parameter_preservation():
    """Verify conic parameter and asphere coefficients are preserved across modifications and stack sync."""
    model = LensDataTableModel()
    model.insertRows(1, 1)

    # Set conic constant
    assert model.setData(1, 6, -1.25) is True
    assert abs(model.data(1, 6) - (-1.25)) < 1e-5

    # Modify other parameters (radius, thickness, glass)
    assert model.setData(1, 2, 85.0) is True
    assert model.setData(1, 3, 12.0) is True
    assert model.setData(1, 4, "N-SF11") is True

    # Verify conic parameter is preserved
    assert abs(model.data(1, 6) - (-1.25)) < 1e-5
    stack = model.get_surface_stack()
    assert abs(stack.surfaces[1].conic - (-1.25)) < 1e-5

    # Test setting a SurfaceStack with custom conic and asphere_coeffs into model
    custom_stack = SurfaceStack()
    s_obj = Surface(name="OBJ", surface_type=SurfaceType.OBJECT, thickness=50.0)
    s_asph = Surface(
        name="AsphericLens",
        surface_type=SurfaceType.ASPHERIC,
        radius=50.0,
        thickness=10.0,
        glass="N-BK7",
        semi_diameter=20.0,
        conic=-2.5,
        asphere_coeffs=[1e-5, -2e-8, 3e-11]
    )
    s_ima = Surface(name="IMA", surface_type=SurfaceType.DETECTOR, thickness=0.0)
    custom_stack.add_surface(s_obj)
    custom_stack.add_surface(s_asph)
    custom_stack.add_surface(s_ima)

    model.set_surface_stack(custom_stack)
    assert abs(model.data(1, 6) - (-2.5)) < 1e-5

    rebuilt_stack = model.get_surface_stack()
    assert abs(rebuilt_stack.surfaces[1].conic - (-2.5)) < 1e-5
    assert rebuilt_stack.surfaces[1].asphere_coeffs == [1e-5, -2e-8, 3e-11]


def test_s1_3_row_moving():
    """Verify row moving methods: moveRow, move_row, move_surface_up, move_surface_down."""
    model = LensDataTableModel()
    # Populate distinct surface names
    model.insertRows(1, 2)
    model.setData(1, 0, "Surf_A")
    model.setData(2, 0, "Surf_B")

    # Current order: OBJ (0), Surf_A (1), Surf_B (2), IMA (3)
    assert model.data(1, 0) == "Surf_A"
    assert model.data(2, 0) == "Surf_B"

    # Test move_surface_down(1): Move Surf_A down to row 2
    res_down = model.move_surface_down(1)
    assert res_down is True
    assert model.data(1, 0) == "Surf_B"
    assert model.data(2, 0) == "Surf_A"

    # Test move_surface_up(2): Move Surf_A back up to row 1
    res_up = model.move_surface_up(2)
    assert res_up is True
    assert model.data(1, 0) == "Surf_A"
    assert model.data(2, 0) == "Surf_B"

    # Test move_row(1, 2): Move row 1 to row 2
    res_mr = model.move_row(1, 2)
    assert res_mr is True
    assert model.data(1, 0) == "Surf_B"
    assert model.data(2, 0) == "Surf_A"

    # Test Qt standard moveRow(None, 2, None, 1): Move row 2 to before row 1
    res_qmr = model.moveRow(None, 2, None, 1)
    assert res_qmr is True
    assert model.data(1, 0) == "Surf_A"
    assert model.data(2, 0) == "Surf_B"

    # Check SurfaceStack reflects final order
    stack = model.get_surface_stack()
    assert stack.surfaces[1].name == "Surf_A"
    assert stack.surfaces[2].name == "Surf_B"


def test_s1_4_nan_inf_validation_rejection():
    """Verify NaN and Inf inputs are rejected for numeric columns and semi-diameter < 0 is rejected."""
    model = LensDataTableModel()
    numeric_cols = [2, 3, 5, 6, 7, 8, 9, 10, 11, 12]

    # Test NaN rejection
    nan_inputs = [float('nan'), np.nan, "nan", "NaN", "  nan  "]
    for col in numeric_cols:
        orig_val = model.data(1, col)
        for nan_val in nan_inputs:
            res = model.setData(1, col, nan_val)
            assert res is False, f"setData accepted NaN input '{nan_val}' on column {col}"
            assert model.data(1, col) == orig_val

    # Test Inf rejection (float inf and string -inf/-infinity)
    inf_inputs = [float('inf'), float('-inf'), np.inf, "-inf", "-infinity", "  -inf  "]
    for col in numeric_cols:
        orig_val = model.data(1, col)
        for inf_val in inf_inputs:
            res = model.setData(1, col, inf_val)
            assert res is False, f"setData accepted Inf input '{inf_val}' on column {col}"
            assert model.data(1, col) == orig_val

    # Test Zemax flat surface infinity alias ("inf", "infinity") on numeric columns
    # Should convert to 0.0 for radius/conic etc.
    assert model.setData(1, 2, "inf") is True
    assert model.data(1, 2) == 0.0

    assert model.setData(1, 2, "infinity") is True
    assert model.data(1, 2) == 0.0

    # Test semi-diameter < 0 rejection (column 5)
    orig_semi = model.data(1, 5)
    assert model.setData(1, 5, -5.0) is False
    assert model.data(1, 5) == orig_semi
    assert model.setData(1, 5, 10.0) is True
    assert model.data(1, 5) == 10.0


# ============================================================================
# Scenario 2: MainWindow Integration & Actions
# ============================================================================

def test_s2_1_main_window_getters():
    """Verify MainWindow getter methods return expected sub-widget instances."""
    win = MainWindow()
    assert win.get_model() is win.model
    assert isinstance(win.get_model(), LensDataTableModel)

    assert win.get_dual_view() is win.dual_view
    assert isinstance(win.get_dual_view(), DualViewWidget)

    assert win.get_analysis_tabs() is win.analysis_tabs
    assert isinstance(win.get_analysis_tabs(), AnalysisTabsWidget)


def test_s2_2_main_window_action_aliases():
    """Verify MainWindow action aliases: on_add_surface, on_remove_surface, on_move_surface_up/down."""
    win = MainWindow()
    initial_rows = win.model.rowCount()

    # Test on_add_surface
    res_add = win.on_add_surface(1)
    assert res_add is True
    assert win.model.rowCount() == initial_rows + 1

    # Test on_remove_surface
    res_rem = win.on_remove_surface(1)
    assert res_rem is True
    assert win.model.rowCount() == initial_rows

    # Test on_move_surface_down & up
    win.on_add_surface(1)
    win.model.setData(1, 0, "Surf_X")
    win.model.setData(2, 0, "Surf_Y")

    res_down = win.on_move_surface_down(1)
    assert res_down is True
    assert win.model.data(2, 0) == "Surf_X"

    res_up = win.on_move_surface_up(2)
    assert res_up is True
    assert win.model.data(1, 0) == "Surf_X"


def test_s2_3_main_window_file_actions():
    """Verify MainWindow .zmx import, .step export, and CAD surface loading."""
    win = MainWindow()

    # 1. Test .zmx import
    zmx_data = """! Sample Zemax file
SURF 0
  TYPE OBJ
  THIC 50.000000
SURF 1
  CURV 0.025
  THIC 6.000000
  GLAS N-BK7 1.5168 64.17
  DIAM 20.000000
  CONI -0.5
SURF 2
  CURV -0.02
  THIC 40.000000
  GLAS AIR
  DIAM 20.000000
SURF 3
  TYPE IMA
  THIC 0.000000
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.zmx', delete=False) as f:
        f.write(zmx_data)
        zmx_path = f.name

    try:
        res_zmx = win.on_open_zmx(zmx_path)
        assert res_zmx is True
        assert win.model.rowCount() == 4
        assert abs(win.model.data(1, 2) - 40.0) < 1e-4  # Radius = 1/0.025 = 40.0
        assert abs(win.model.data(1, 6) - (-0.5)) < 1e-4
    finally:
        if os.path.exists(zmx_path):
            os.remove(zmx_path)

    # 2. Test .step export
    with tempfile.NamedTemporaryFile(suffix='.step', delete=False) as f:
        step_out_path = f.name

    try:
        res_step = win.on_export_step(step_out_path)
        assert res_step is True
        assert os.path.exists(step_out_path)
        assert os.path.getsize(step_out_path) > 0
    finally:
        if os.path.exists(step_out_path):
            os.remove(step_out_path)

    # 3. Test non-existent file handling
    assert win.on_open_zmx("non_existent_path.zmx") is False
    assert win.on_load_cad_surface("non_existent_cad.step") is False


# ============================================================================
# Scenario 3: DualViewWidget & AnalysisTabsWidget Visualization Updates
# ============================================================================

def test_s3_1_dual_view_ray_trace_updates():
    """Verify DualViewWidget layout updates with complex optical system and traced rays."""
    dual_view = DualViewWidget()
    stack = SurfaceStack()

    stack.add_surface(Surface(name="OBJ", surface_type=SurfaceType.OBJECT, thickness=50.0, glass="AIR", semi_diameter=25.0))
    stack.add_surface(Surface(name="L1_Front", surface_type=SurfaceType.ASPHERIC, radius=40.0, thickness=7.0, glass="N-BK7", semi_diameter=20.0, conic=-0.5))
    stack.add_surface(Surface(name="L1_Back", surface_type=SurfaceType.STANDARD, radius=-60.0, thickness=10.0, glass="AIR", semi_diameter=20.0))
    stack.add_surface(Surface(name="Grating", surface_type=SurfaceType.GRATING, radius=0.0, thickness=30.0, glass="AIR", semi_diameter=15.0, grating_lines_per_mm=600.0))
    stack.add_surface(Surface(name="IMA", surface_type=SurfaceType.DETECTOR, thickness=0.0, glass="AIR", semi_diameter=25.0))

    system = OpticalSystem(stack)

    # Generate rays
    rays = []
    for py in np.linspace(-10.0, 10.0, 5):
        for wl in [0.45, 0.546, 0.65]:
            r = Ray(pos=np.array([0.0, py, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl)
            traced = system.trace_ray(r)
            rays.append(traced)

    # Test update_layout
    dual_view.update_layout(surface_stack=stack, rays=rays)
    assert dual_view.refresh_layout() is True

    # Test view toggling
    dual_view.set_rz_view_active(True)
    dual_view.set_xy_view_active(False)
    assert dual_view.refresh_layout() is True

    dual_view.set_rz_view_active(False)
    dual_view.set_xy_view_active(True)
    assert dual_view.refresh_layout() is True

    dual_view.set_rz_view_active(True)
    dual_view.set_xy_view_active(True)
    assert dual_view.refresh_layout() is True

    # Test update_system
    assert dual_view.update_system(stack) is True


def test_s3_2_analysis_tabs_visualization_updates():
    """Verify AnalysisTabsWidget updates all 4 diagnostic tabs without errors."""
    stack = SurfaceStack()
    stack.add_surface(Surface(name="OBJ", surface_type=SurfaceType.OBJECT, thickness=100.0, glass="AIR", semi_diameter=25.0))
    stack.add_surface(Surface(name="L1", surface_type=SurfaceType.STANDARD, radius=50.0, thickness=8.0, glass="N-BK7", semi_diameter=20.0))
    stack.add_surface(Surface(name="L2", surface_type=SurfaceType.STANDARD, radius=-50.0, thickness=80.0, glass="AIR", semi_diameter=20.0))
    stack.add_surface(Surface(name="IMA", surface_type=SurfaceType.DETECTOR, thickness=0.0, glass="AIR", semi_diameter=25.0))

    system = OpticalSystem(stack)
    model = LensDataTableModel(stack)

    analysis_widget = AnalysisTabsWidget(model=model)

    # Test switching tabs and updating each tab
    for tab in ["Spot Diagram", "Ray Fan", "Grating Dispersion", "CCD Heatmap"]:
        res = analysis_widget.set_active_tab(tab)
        assert res is True, f"Failed to set active tab to {tab}"
        assert analysis_widget.active_tab == tab
        assert analysis_widget.refresh_analysis() is True

    # Test individual tab update_plot directly
    spot = generate_spot_diagram_from_system(system, num_rays=100)
    tab_spot = getattr(analysis_widget, "tab_spot", SpotDiagramTab())
    tab_spot.update_plot(system=system, spot_diagram=spot)

    tab_rayfan = getattr(analysis_widget, "tab_rayfan", RayFanTab())
    tab_rayfan.update_plot(system=system)

    tab_dispersion = getattr(analysis_widget, "tab_dispersion", GratingDispersionTab())
    tab_dispersion.update_plot(system=system)

    ccd = CCDSensor(width_mm=10.0, height_mm=10.0, pixel_cols=50, pixel_rows=50)
    if len(spot.x) > 0:
        ccd.deposit_rays(spot.x, spot.y)
    tab_ccd = getattr(analysis_widget, "tab_ccd", CCDHeatmapTab())
    tab_ccd.update_plot(system=system, ccd_sensor=ccd)

    # Test update_analysis alias
    assert analysis_widget.update_analysis(system) is True


if __name__ == "__main__":
    pytest.main([__file__])
