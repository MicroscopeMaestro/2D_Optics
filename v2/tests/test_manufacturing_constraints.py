"""
Unit Tests for Optical Manufacturing Constraints (MNCG, MXCG, MNEG, MNCA, MNEA):
- Verification that lenses do not balloon into heavy glass cylinders.
- Verification that edge thickness does not fall below minimum manufacturable threshold.
- Verification of air gap and back focal length clearance.
- Verification of GUI controls in OptimizerDockWidget.
"""

import pytest
import numpy as np
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.optimizer import OpticalOptimizer


def test_manufacturing_constraints_center_and_edge_thickness():
    """Verify that optimizer respects MNCG, MXCG, and MNEG constraints."""
    stack = SurfaceStack()
    stack.add_surface(Surface(name="OBJ", thickness=100.0))
    # Starting with a thick lens
    stack.add_surface(Surface(name="Front", radius=35.0, thickness=12.0, glass="N-BK7", semi_diameter=12.5))
    stack.add_surface(Surface(name="Back", radius=-35.0, thickness=40.0, glass="AIR", semi_diameter=12.5))
    stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR))

    sys = OpticalSystem(stack)
    optimizer = OpticalOptimizer(
        sys,
        method="Nelder-Mead",
        max_iter=35,
        num_rays=15,
        enforce_manufacturing_constraints=True,
        min_glass_thickness=2.0,
        max_glass_thickness=7.0,
        min_edge_thickness=1.5,
        min_air_gap=1.0,
        min_bfl=15.0
    )

    res = optimizer.optimize(variable_indices={"thickness": [1, 2], "radius": [1, 2]})
    assert res.success
    opt_stack = res.optimized_stack
    t_glass = opt_stack.surfaces[1].thickness
    t_bfl = opt_stack.surfaces[2].thickness

    # Glass thickness must stay within [2.0, 7.0] mm
    assert 1.8 <= t_glass <= 7.2, f"Glass thickness {t_glass} mm outside allowed manufacturing bounds [2.0, 7.0]"
    # Back focal length must stay >= 15.0 mm
    assert t_bfl >= 14.5, f"BFL {t_bfl} mm smaller than minimum clearance 15.0 mm"

    # Edge thickness check
    s1 = opt_stack.surfaces[1]
    s2 = opt_stack.surfaces[2]
    r_edge = min(s1.semi_diameter, s2.semi_diameter)
    edge_t = t_glass + s2.sag(r_edge) - s1.sag(r_edge)
    assert edge_t >= 1.0, f"Edge thickness {edge_t} mm is knife-edged (< 1.0 mm)"


def test_optimizer_dock_manufacturing_constraints_ui():
    """Verify OptimizerDockWidget exposes and reads manufacturing constraints."""
    from optics_app.gui.qt_compat import QT_AVAILABLE, QtWidgets
    if not QT_AVAILABLE or not hasattr(QtWidgets, "QApplication"):
        pytest.skip("Qt not available in environment")
    _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(["-platform", "offscreen"])

    from optics_app.gui.optimizer_dock import OptimizerDockWidget
    from optics_app.gui.lde_table import LensDataTableModel

    model = LensDataTableModel()
    dock = OptimizerDockWidget(model)

    assert hasattr(dock, "grp_mfg")
    assert hasattr(dock, "combo_mfg_preset")
    assert hasattr(dock, "spin_min_glass")
    assert hasattr(dock, "spin_max_glass")
    assert hasattr(dock, "spin_min_edge")
    assert hasattr(dock, "spin_min_air")
    assert hasattr(dock, "spin_min_bfl")
    assert hasattr(dock, "spin_max_aspect")
    assert hasattr(dock, "chk_desensitize")
    assert hasattr(dock, "btn_fit_test_plates")

    mfg = dock.get_manufacturing_constraints()
    assert mfg["enforce"] is True
    assert mfg["min_glass_thickness"] == 1.5
    assert mfg["max_glass_thickness"] == 18.0
    assert mfg["min_edge_thickness"] == 1.0
    assert mfg["min_air_gap"] == 0.5
    assert mfg["min_bfl"] == 10.0
    assert mfg["max_aspect_ratio"] == 10.0
    assert mfg["desensitize_tolerances"] is False

    # Test preset switching
    dock.combo_mfg_preset.setCurrentText("High-Precision (Laser / Metrology)")
    mfg_hp = dock.get_manufacturing_constraints()
    assert mfg_hp["min_glass_thickness"] == 2.5
    assert mfg_hp["min_edge_thickness"] == 2.0
    assert mfg_hp["max_aspect_ratio"] == 6.0
    assert mfg_hp["desensitize_tolerances"] is True


def test_test_plate_catalog_fitting():
    """Verify standard test plate catalog binary search and optimizer test plate snapping."""
    from optics_app.engine.manufacturing_standards import TestPlateCatalog

    # Exact standard radius
    r_exact, delta, fringes = TestPlateCatalog.find_nearest_test_plate(50.0, semi_diameter_mm=12.5)
    assert abs(r_exact - 50.0) < 1e-4
    assert abs(fringes) < 1e-4

    # Non-standard radius (e.g. 52.3 mm) should snap to 50.0 mm
    r_snap, delta, fringes = TestPlateCatalog.find_nearest_test_plate(52.3, semi_diameter_mm=12.5)
    assert r_snap in TestPlateCatalog.STANDARD_RADII_MM
    assert fringes > 0.0

    # Negative radius
    r_neg, delta, fringes = TestPlateCatalog.find_nearest_test_plate(-78.4, semi_diameter_mm=12.5)
    assert r_neg < 0
    assert abs(r_neg) in TestPlateCatalog.STANDARD_RADII_MM

    # Flat / Plano
    r_plano, delta, fringes = TestPlateCatalog.find_nearest_test_plate(1e6, semi_diameter_mm=12.5)
    assert r_plano == 0.0
    assert fringes == 0.0

    # System-level fit_to_test_plates
    stack = SurfaceStack()
    stack.add_surface(Surface(name="OBJ", thickness=100.0))
    stack.add_surface(Surface(name="L1F", radius=52.3, thickness=5.0, glass="N-BK7", semi_diameter=12.5))
    stack.add_surface(Surface(name="L1B", radius=-47.1, thickness=45.0, glass="AIR", semi_diameter=12.5))
    stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR))

    sys = OpticalSystem(stack)
    opt = OpticalOptimizer(sys, method="Nelder-Mead", num_rays=15)
    fit_res = opt.fit_to_test_plates(re_optimize_bfl=True)
    assert fit_res.optimized_stack is not None
    s_front = fit_res.optimized_stack.surfaces[1]
    s_rear = fit_res.optimized_stack.surfaces[2]
    assert abs(s_front.radius) in TestPlateCatalog.STANDARD_RADII_MM
    assert abs(s_rear.radius) in TestPlateCatalog.STANDARD_RADII_MM


def test_manufacturing_audit_tab_and_report():
    """Verify ISO 10110 audit report generation and ManufacturingAuditTab GUI component."""
    from optics_app.engine.manufacturing_standards import (
        QualityPreset,
        ManufacturingConstraintsConfig,
        audit_optical_system_manufacturing
    )

    stack = SurfaceStack()
    stack.add_surface(Surface(name="OBJ", thickness=100.0))
    # Compliant element
    stack.add_surface(Surface(name="S1", radius=50.0, thickness=5.0, glass="N-BK7", semi_diameter=12.5))
    stack.add_surface(Surface(name="S2", radius=-50.0, thickness=45.0, glass="AIR", semi_diameter=12.5))
    stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR))

    report = audit_optical_system_manufacturing(stack.surfaces)
    assert report.is_compliant is True
    assert report.compliance_score == 100.0
    assert len(report.elements) == 1
    assert report.elements[0].glass == "N-BK7"
    assert report.elements[0].is_compliant is True

    md = report.format_markdown()
    assert "Optical Manufacturing & Mounting Audit" in md
    assert "PASS" in md

    # Now verify with an unphysical thin lens violating MNCG and D/t
    stack_bad = SurfaceStack()
    stack_bad.add_surface(Surface(name="OBJ", thickness=100.0))
    stack_bad.add_surface(Surface(name="S1", radius=50.0, thickness=0.4, glass="N-BK7", semi_diameter=15.0))
    stack_bad.add_surface(Surface(name="S2", radius=-50.0, thickness=45.0, glass="AIR", semi_diameter=15.0))
    stack_bad.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR))

    report_bad = audit_optical_system_manufacturing(stack_bad.surfaces)
    assert report_bad.is_compliant is False
    assert len(report_bad.overall_errors) > 0
    assert any("MNCG" in err or "Center thickness" in err for err in report_bad.overall_errors)

    # Test ManufacturingAuditTab GUI
    from optics_app.gui.qt_compat import QT_AVAILABLE, QtWidgets
    if QT_AVAILABLE and hasattr(QtWidgets, "QApplication"):
        _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(["-platform", "offscreen"])
        from optics_app.gui.analysis_tabs import ManufacturingAuditTab, AnalysisTabsWidget
        from optics_app.gui.lde_table import LensDataTableModel

        model = LensDataTableModel()
        model.set_surface_stack(stack)

        tab = ManufacturingAuditTab(model=model)
        tab.update_plot(system=OpticalSystem(stack))
        assert "PASS" in tab.cached_report

        suite = AnalysisTabsWidget(model=model)
        assert "Manufacturing & Mounting Audit" in suite.tabs
        assert hasattr(suite, "tab_mfg")

