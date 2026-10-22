"""
Unit Tests for Optical Analysis Diagnostics & First-Order Paraxial Engine:
- EFL, BFL, FFL, Power (diopters)
- Front & Rear Principal Planes (P1, P2, Delta P) in local and Global Z coordinates
- Entrance Pupil (EnP) and Exit Pupil / Effective Back Pupil (ExP)
- Pupil Magnification and ExP to Image Distance
- FirstOrderDiagnosticsTab Matplotlib Rendering
- Paraxial Sanity Validation
"""

import pytest
import math
import numpy as np

from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.first_order import calculate_first_order_data, format_zemax_first_order_report
from optics_app.engine.diagnostics import ModelSanityValidator, DiagnosticSeverity


def test_first_order_cardinal_points_and_pupils_singlet():
    """Verify first-order cardinal points and pupils on an N-BK7 singlet."""
    stack = SurfaceStack()
    stack.add_surface(Surface(name="OBJ", surface_type=SurfaceType.OBJECT, thickness=100.0, glass="AIR"))
    stack.add_surface(Surface(name="Front", radius=50.0, thickness=5.0, glass="N-BK7", semi_diameter=12.5))
    stack.add_surface(Surface(name="Rear", radius=-50.0, thickness=45.0, glass="AIR", semi_diameter=12.5))
    stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR))

    data = calculate_first_order_data(stack.surfaces, wavelength_um=0.5876)

    # 1. Focal lengths and power
    assert not data["is_afocal"]
    assert 48.0 < data["effl"] < 52.0
    assert 45.0 < data["bfl"] < 49.0
    assert 45.0 < data["ffl"] < 49.0
    assert 18.0 < data["optical_power_diopters"] < 21.0

    # 2. Principal planes
    assert 1.0 < data["p1"] < 2.5
    assert -2.5 < data["p2"] < -1.0
    assert data["inter_p"] > 0.0

    # 3. Global coordinates
    gp = data["global_positions"]
    assert gp["v1_z"] == 100.0
    assert gp["v2_z"] == 105.0
    assert gp["image_z"] == 150.0
    assert abs(gp["p1_z"] - (100.0 + data["p1"])) < 1e-6
    assert abs(gp["p2_z"] - (105.0 + data["p2"])) < 1e-6
    assert abs(gp["f2_z"] - (105.0 + data["bfl"])) < 1e-6

    # 4. Pupils (no explicit stop -> entrance at front, exit pupil through singlet)
    assert data["epd"] == 25.0
    assert data["enp_pos"] == 0.0
    assert 24.0 < data["expd"] < 26.5
    assert data["exp_to_image"] > 0.0

    # 5. Report formatting
    report = format_zemax_first_order_report(data, title="N-BK7 Singlet")
    assert "Effective Focal Length (EFFL)" in report
    assert "Effective Back Pupil" in report
    assert "Front Principal Plane (P1)" in report
    assert "Rear Principal Plane (P2)" in report


def test_first_order_internal_aperture_stop_pupils():
    """Verify entrance and exit pupil calculations when an internal aperture stop is present."""
    stack = SurfaceStack()
    stack.add_surface(Surface(name="OBJ", surface_type=SurfaceType.OBJECT, thickness=100.0, glass="AIR"))
    stack.add_surface(Surface(name="L1_F", radius=40.0, thickness=4.0, glass="N-BK7", semi_diameter=12.0))
    stack.add_surface(Surface(name="L1_B", radius=-60.0, thickness=8.0, glass="AIR", semi_diameter=12.0))
    stack.add_surface(Surface(name="STOP", surface_type=SurfaceType.STOP, semi_diameter=5.0, thickness=6.0, glass="AIR"))
    stack.add_surface(Surface(name="L2_F", radius=50.0, thickness=3.5, glass="SF11", semi_diameter=10.0))
    stack.add_surface(Surface(name="L2_B", radius=-35.0, thickness=30.0, glass="AIR", semi_diameter=10.0))
    stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR))

    data = calculate_first_order_data(stack.surfaces, wavelength_um=0.5876)

    # Stop has diameter 10.0 mm
    # Entrance pupil is image of stop through L1 -> should be slightly larger and shifted
    assert data["epd"] > 0.0
    assert data["expd"] > 0.0
    # Both entrance and exit pupil positions should be computed from their respective vertices
    assert abs(data["enp_pos"]) > 0.0 or data["epd"] != 10.0
    assert abs(data["exp_pos"]) > 0.0 or data["expd"] != 10.0


def test_first_order_afocal_telescope():
    """Verify afocal system handling (infinite EFL)."""
    stack = SurfaceStack()
    stack.add_surface(Surface(name="OBJ", surface_type=SurfaceType.OBJECT, thickness=100.0, glass="AIR"))
    # Flat plate
    stack.add_surface(Surface(name="P1", radius=0.0, thickness=10.0, glass="N-BK7", semi_diameter=12.5))
    stack.add_surface(Surface(name="P2", radius=0.0, thickness=30.0, glass="AIR", semi_diameter=12.5))
    stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR))

    data = calculate_first_order_data(stack.surfaces, wavelength_um=0.5876)
    assert data["is_afocal"]
    assert math.isinf(data["effl"])
    assert data["optical_power_diopters"] == 0.0


def test_model_sanity_paraxial_first_order_diagnostics():
    """Verify ModelSanityValidator paraxial inspection checks."""
    stack = SurfaceStack()
    stack.add_surface(Surface(name="OBJ", surface_type=SurfaceType.OBJECT, thickness=100.0, glass="AIR"))
    # Highly diverging lens -> negative BFL (virtual image)
    stack.add_surface(Surface(name="L1_F", radius=-30.0, thickness=3.0, glass="N-BK7", semi_diameter=10.0))
    stack.add_surface(Surface(name="L1_B", radius=30.0, thickness=20.0, glass="AIR", semi_diameter=10.0))
    stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR))

    v_res, data = ModelSanityValidator.inspect_paraxial_first_order(stack, wavelength_um=0.5876)
    assert not data["is_afocal"]
    assert data["effl"] < 0.0  # Diverging lens
    assert any(item.code == "PARAXIAL_VIRTUAL_IMAGE" for item in v_res.items)


def test_first_order_diagnostics_tab_headless():
    """Verify FirstOrderDiagnosticsTab executes and updates without headless exceptions."""
    from optics_app.gui.qt_compat import QT_AVAILABLE, QtWidgets
    if not QT_AVAILABLE or not hasattr(QtWidgets, "QApplication"):
        pytest.skip("Qt not available in environment")
    _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(["-platform", "offscreen"])

    from optics_app.gui.analysis_tabs import FirstOrderDiagnosticsTab

    stack = SurfaceStack()
    stack.add_surface(Surface(name="OBJ", surface_type=SurfaceType.OBJECT, thickness=100.0, glass="AIR"))
    stack.add_surface(Surface(name="Front", radius=36.52, thickness=6.0, glass="N-BAF10", semi_diameter=12.7))
    stack.add_surface(Surface(name="Cemented", radius=-23.59, thickness=2.5, glass="SF10", semi_diameter=12.7))
    stack.add_surface(Surface(name="Rear", radius=-169.63, thickness=45.29, glass="AIR", semi_diameter=12.7))
    stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR))

    sys = OpticalSystem(stack)
    tab = FirstOrderDiagnosticsTab()
    tab.update_plot(system=sys, wavelengths=[0.5876])
    assert tab.cached_report != ""
    assert "Effective Focal Length (EFFL)" in tab.cached_report
    assert "Effective Back Pupil" in tab.cached_report


def test_compute_paraxial_marginal_and_principal_rays_singlet():
    """Verify paraxial marginal and chief ray tracing through standard biconvex singlet."""
    from optics_app.engine.first_order import compute_paraxial_marginal_and_principal_rays
    stack = SurfaceStack()
    stack.add_surface(Surface(name="OBJ", surface_type=SurfaceType.OBJECT, thickness=0.0, glass="AIR"))
    stack.add_surface(Surface(name="S1", radius=50.0, thickness=5.0, glass="N-BK7", semi_diameter=12.5))
    stack.add_surface(Surface(name="S2", radius=-50.0, thickness=48.0, glass="AIR", semi_diameter=12.5))
    stack.add_surface(Surface(name="IMA", surface_type=SurfaceType.DETECTOR))

    res = compute_paraxial_marginal_and_principal_rays(stack.surfaces, wavelength_um=0.5876, epd=10.0, field_angle_deg=5.0)
    assert res["is_vignetted"] is False
    assert res["system_vignetting_factor"] == 0.0
    assert len(res["marginal_trajectory"]["z"]) == 3
    assert len(res["chief_trajectory"]["y"]) == 3
    # Marginal ray at S1 starts at y = epd/2 = 5.0
    assert abs(res["marginal_trajectory"]["y"][0] - 5.0) < 1e-4
    # Chief ray passes through pupil/stop (S1) at y = 0
    assert abs(res["chief_trajectory"]["y"][0]) < 1e-4


def test_paraxial_aperture_clipping_and_vignetting_factor():
    """Verify paraxial vignetting detection when aperture is smaller than beam envelope."""
    from optics_app.engine.first_order import compute_paraxial_marginal_and_principal_rays
    stack = SurfaceStack()
    stack.add_surface(Surface(name="OBJ", surface_type=SurfaceType.OBJECT, thickness=0.0, glass="AIR"))
    # Severely clipped semi-diameter on S2
    stack.add_surface(Surface(name="S1", radius=100.0, thickness=10.0, glass="N-BK7", semi_diameter=20.0))
    stack.add_surface(Surface(name="S2", radius=-100.0, thickness=50.0, glass="AIR", semi_diameter=2.0))
    stack.add_surface(Surface(name="IMA", surface_type=SurfaceType.DETECTOR))

    res = compute_paraxial_marginal_and_principal_rays(stack.surfaces, wavelength_um=0.5876, epd=20.0)
    assert res["is_vignetted"] is True
    assert res["system_vignetting_factor"] > 0.5
    assert res["limiting_surface_index"] == 2
    assert res["limiting_surface_name"] == "S2"


def test_paraxial_afocal_flat_plate_and_window():
    """Verify paraxial ray tracing through afocal plano window."""
    from optics_app.engine.first_order import compute_paraxial_marginal_and_principal_rays
    stack = SurfaceStack()
    stack.add_surface(Surface(name="OBJ", surface_type=SurfaceType.OBJECT, thickness=0.0, glass="AIR"))
    stack.add_surface(Surface(name="W1", radius=0.0, thickness=10.0, glass="N-BK7", semi_diameter=15.0))
    stack.add_surface(Surface(name="W2", radius=0.0, thickness=20.0, glass="AIR", semi_diameter=15.0))
    stack.add_surface(Surface(name="IMA", surface_type=SurfaceType.DETECTOR))

    res = compute_paraxial_marginal_and_principal_rays(stack.surfaces, wavelength_um=0.5876, epd=10.0)
    assert res["is_vignetted"] is False
    # Marginal ray stays parallel through flat plate
    assert abs(res["marginal_trajectory"]["y"][0] - 5.0) < 1e-4
    assert abs(res["marginal_trajectory"]["y"][1] - 5.0) < 1e-4


def test_paraxial_finite_conjugate_object_height():
    """Verify finite conjugate object distance and height propagation."""
    from optics_app.engine.first_order import compute_paraxial_marginal_and_principal_rays
    stack = SurfaceStack()
    stack.add_surface(Surface(name="OBJ", surface_type=SurfaceType.OBJECT, thickness=100.0, glass="AIR"))
    stack.add_surface(Surface(name="L1", radius=40.0, thickness=5.0, glass="N-BK7", semi_diameter=15.0))
    stack.add_surface(Surface(name="L2", radius=-40.0, thickness=100.0, glass="AIR", semi_diameter=15.0))
    stack.add_surface(Surface(name="IMA", surface_type=SurfaceType.DETECTOR))

    res = compute_paraxial_marginal_and_principal_rays(stack.surfaces, wavelength_um=0.5876, epd=10.0, object_height=10.0)
    assert len(res["surfaces_data"]) == 3
    # Chief ray slope is non-zero from object height to pupil
    assert abs(res["surfaces_data"][0]["u_chief"]) > 1e-4
    # Chief ray arrives at rear surface with non-zero height
    assert abs(res["surfaces_data"][1]["y_chief"]) > 1e-4


def test_first_order_diagnostics_tab_with_vignetting_and_field():
    """Verify FirstOrderDiagnosticsTab handles vignetting badges and multi-field envelope."""
    from optics_app.gui.qt_compat import QT_AVAILABLE, QtWidgets
    if not QT_AVAILABLE or not hasattr(QtWidgets, "QApplication"):
        pytest.skip("Qt not available in environment")
    _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(["-platform", "offscreen"])

    from optics_app.gui.analysis_tabs import FirstOrderDiagnosticsTab
    from optics_app.engine.system_config import SystemConfig, FieldPoint

    stack = SurfaceStack()
    stack.add_surface(Surface(name="OBJ", surface_type=SurfaceType.OBJECT, thickness=0.0, glass="AIR"))
    stack.add_surface(Surface(name="S1", radius=50.0, thickness=5.0, glass="N-BK7", semi_diameter=12.5))
    stack.add_surface(Surface(name="S2", radius=-50.0, thickness=48.0, glass="AIR", semi_diameter=3.0)) # Clipped
    stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR))

    cfg = SystemConfig()
    cfg.fields = [FieldPoint(0.0, 0.0, 1.0), FieldPoint(0.0, 10.0, 1.0)]

    sys = OpticalSystem(stack)
    tab = FirstOrderDiagnosticsTab()
    tab.update_plot(system=sys, wavelengths=[0.5876], system_config=cfg)
    assert "Paraxial Beam Vignetting" in tab.cached_report
    assert "beam clipping" in tab.cached_report

