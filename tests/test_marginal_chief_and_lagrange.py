import pytest
import math
import numpy as np

from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.first_order import (
    calculate_first_order_data,
    compute_stops_windows_and_lagrange,
    compute_paraxial_marginal_and_principal_rays,
    format_zemax_first_order_report,
)


def create_classical_doublet_system():
    """
    Standard achromatic doublet with stop between elements and an image field stop.
    """
    surfaces = [
        Surface(name="Object", surface_type=SurfaceType.OBJECT, thickness=float("inf")),
        Surface(name="Crown_Front", radius=50.0, thickness=6.0, glass="N-BK7", semi_diameter=15.0),
        Surface(name="Crown_Rear", radius=-40.0, thickness=2.5, glass="N-SF11", semi_diameter=15.0),
        Surface(name="Flint_Rear", radius=-120.0, thickness=5.0, glass="AIR", semi_diameter=15.0),
        Surface(name="ApertureStop", surface_type=SurfaceType.STOP, thickness=35.0, glass="AIR", semi_diameter=9.0),
        Surface(name="Image", surface_type=SurfaceType.IMAGE, thickness=0.0, glass="AIR", semi_diameter=12.0),
    ]
    return surfaces


def test_classical_marginal_and_chief_ray_definitions():
    """
    Verify Marginal and Chief rays follow rigorous classical optical design definitions.
    - Marginal ray: Starts on-axis (y=0), touches Stop edge (y_stop = r_stop), crosses axis at image (y_img = 0).
    - Chief ray: Starts off-axis, passes through Stop center (y_stop = 0), reaches chief image height at image.
    """
    surfaces = create_classical_doublet_system()
    res = compute_paraxial_marginal_and_principal_rays(surfaces, wavelength_um=0.5876, field_angle_deg=1.5)

    ym = res["marginal_trajectory"]["y"]
    yc = res["chief_trajectory"]["y"]
    stop_idx = res["stop_index"]

    # In optical surfaces (start_idx = 1):
    # Stop surface index in opt_surfaces is stop_idx - 1
    opt_stop_idx = stop_idx - 1

    # 1. Marginal ray touches Aperture Stop rim (within paraxial scaling)
    r_stop = surfaces[stop_idx].semi_diameter
    assert abs(abs(ym[opt_stop_idx]) - r_stop) < 1e-6, f"Marginal ray height {ym[opt_stop_idx]} must equal stop radius {r_stop}"

    # 2. Chief ray passes through center of Aperture Stop (y_c = 0)
    assert abs(yc[opt_stop_idx]) < 1e-12, f"Chief ray height {yc[opt_stop_idx]} at stop must be zero"

    # 3. Marginal ray converges toward optical axis near focal plane
    assert abs(ym[-1]) < abs(ym[0]), "Marginal ray should converge toward optical axis through converging doublet"

    # 4. Chief ray at image plane is non-zero (field height)
    assert abs(yc[-1]) > 0.1, "Chief ray at image plane must be non-zero off-axis height"


def test_stops_and_windows_identification():
    """
    Verify Aperture Stop, Field Stop, Entrance Pupil, Exit Pupil, Entrance Window, and Exit Window.
    """
    surfaces = create_classical_doublet_system()
    fod = calculate_first_order_data(surfaces, wavelength_um=0.5876)
    sw = fod.get("stops_and_windows", {})

    assert sw, "stops_and_windows must not be empty"

    # Aperture Stop
    as_info = sw["aperture_stop"]
    assert as_info["surface_index"] == 4, f"Aperture Stop should be S4, got {as_info['surface_index']}"
    assert as_info["semi_diameter"] == 9.0
    assert as_info["diameter"] == 18.0
    assert not math.isinf(as_info["z_global"])

    # Field Stop
    fs_info = sw["field_stop"]
    assert fs_info["surface_index"] == 5, f"Field Stop should be Image S5, got {fs_info['surface_index']}"
    assert fs_info["semi_diameter"] == 12.0
    assert fs_info["max_unvignetted_field_deg"] > 0.0

    # Entrance Pupil
    enp_info = sw["entrance_pupil"]
    assert enp_info["diameter"] > 0.0
    assert not math.isinf(enp_info["z_global"])

    # Exit Pupil
    exp_info = sw["exit_pupil"]
    assert exp_info["diameter"] > 0.0
    assert not math.isinf(exp_info["z_global"])

    # Entrance Window
    enw_info = sw["entrance_window"]
    assert enw_info["diameter"] > 0.0
    assert enw_info["angular_fov_deg"] > 0.0

    # Exit Window
    exw_info = sw["exit_window"]
    assert exw_info["diameter"] > 0.0


def test_helmholtz_lagrange_conservation():
    """
    Verify exact surface-by-surface conservation of Helmholtz-Lagrange Invariant:
    H = n(yc * um - ym * uc) = const.
    """
    surfaces = create_classical_doublet_system()
    fod = calculate_first_order_data(surfaces, wavelength_um=0.5876)
    hl = fod.get("helmholtz_lagrange", {})

    assert hl, "helmholtz_lagrange must not be empty"

    H_nom = hl["lagrange_invariant_H"]
    assert abs(H_nom) > 1e-6, "Lagrange invariant must be non-zero for non-zero pupil and field"

    # Verify paraxial conservation across all surfaces
    max_err = hl["max_conservation_error"]
    assert max_err < 1e-12, f"Lagrange conservation error {max_err} exceeds 1e-12"
    assert hl["is_conserved"] is True

    # Surface steps check
    steps = hl["surface_steps"]
    assert len(steps) >= 4
    for s in steps:
        assert abs(s["delta_H"]) < 1e-12, f"Surface {s['name']} delta H {s['delta_H']} exceeds 1e-12"

    # Etendue check: G = pi^2 * H^2
    etendue_G = hl["etendue_G"]
    expected_G = (math.pi**2) * (H_nom**2)
    assert abs(etendue_G - expected_G) < 1e-12
    assert etendue_G > 0.0

    # Delano Diagram check
    delano = hl["delano_diagram"]
    assert len(delano["ym"]) == len(steps)
    assert len(delano["yc"]) == len(steps)
    assert delano["polygon_area"] > 0.0

    # Smith-Helmholtz Product check
    sm_target = hl["smith_helmholtz_target"]
    assert sm_target == 1.0  # Object and image both in air (n=1)

    # Cardinal Invariants check
    cardinal = hl["cardinal_invariants"]
    assert "aperture_stop" in cardinal
    assert "exit_pupil" in cardinal
    assert "field_stop" in cardinal
    assert "image_plane" in cardinal

    # Abbe Sine Condition check
    abbe_comp = hl["non_paraxial_abbe_comparison"]
    assert len(abbe_comp) == 4
    for zone_data in abbe_comp:
        assert zone_data["pupil_zone"] in (0.25, 0.50, 0.707, 1.00)
        assert zone_data["discrepancy_pct"] >= 0.0


def test_zemax_report_formatting():
    """
    Verify format_zemax_first_order_report includes Stops & Windows and Lagrange sections.
    """
    surfaces = create_classical_doublet_system()
    fod = calculate_first_order_data(surfaces, wavelength_um=0.5876)
    report = format_zemax_first_order_report(fod, title="Test Doublet")

    assert "STOPS AND WINDOWS (PUPILS & DELANO CONJUGATES)" in report
    assert "Aperture Stop (AS)" in report
    assert "Field Stop (FS)" in report
    assert "Entrance Pupil" in report
    assert "Exit Pupil" in report
    assert "Entrance Window" in report
    assert "Exit Window" in report
    assert "HELMHOLTZ-LAGRANGE OPTICAL INVARIANT CONSERVATION" in report
    assert "Optical Invariant (H)" in report
    assert "Optical Etendue" in report
    assert "Delano Polygon Area" in report
    assert "Smith-Helmholtz" in report


def test_first_order_diagnostics_tab_headless():
    """
    Test FirstOrderDiagnosticsTab GUI rendering for Mode 0, Mode 1, Light and Dark themes.
    """
    import os
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])

    from optics_app.gui.analysis_tabs import FirstOrderDiagnosticsTab

    surfaces = create_classical_doublet_system()
    stack = SurfaceStack()
    stack.surfaces = surfaces
    sys_obj = OpticalSystem(stack)

    tab = FirstOrderDiagnosticsTab()

    # Test multi-wavelength detection & switching
    multi_wls = [1.087, 1.225, 1.350]
    tab.update_plot(system=sys_obj, wavelengths=multi_wls)
    assert hasattr(tab, "combo_wl") and tab.combo_wl.count() == 3, f"Expected 3 wavelengths in dropdown, got {tab.combo_wl.count()}"

    # Switch wavelength via dropdown
    tab.combo_wl.setCurrentIndex(1)
    tab.update_plot(system=sys_obj, wavelengths=multi_wls)
    assert abs(tab.combo_wl.currentData() - 1.225) < 1e-4
    assert "1.225" in tab.lbl_badge.text()

    tab.combo_wl.setCurrentIndex(0)
    tab.update_plot(system=sys_obj, wavelengths=multi_wls)
    assert abs(tab.combo_wl.currentData() - 1.087) < 1e-4
    assert "1.087" in tab.lbl_badge.text()

    # Test Mode 0 (Stops & Windows)
    tab.combo_mode.setCurrentIndex(0)
    tab.set_dark_mode(False)
    tab.update_plot(system=sys_obj, wavelengths=multi_wls)
    tab.set_dark_mode(True)
    tab.update_plot(system=sys_obj, wavelengths=multi_wls)

    # Test Mode 1 (Helmholtz-Lagrange & Delano with multi-wavelength overlay)
    tab.combo_mode.setCurrentIndex(1)
    tab.set_dark_mode(False)
    tab.update_plot(system=sys_obj, wavelengths=multi_wls)
    tab.set_dark_mode(True)
    tab.update_plot(system=sys_obj, wavelengths=multi_wls)


def test_dual_view_widget_pupils_windows_headless():
    """
    Test DualViewWidget with Marg/Chief and Pupils/Windows overlays in Light and Dark themes.
    """
    import os
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])

    from optics_app.gui.dual_view import DualViewWidget

    surfaces = create_classical_doublet_system()
    stack = SurfaceStack()
    stack.surfaces = surfaces

    view = DualViewWidget()
    view.surface_stack = stack

    # Enable Marginal/Chief and Pupils/Windows
    view.set_show_marginal_chief(True)
    view.set_show_pupils_windows(True)

    # Test light and dark refresh
    view.set_dark_mode(False)
    view.refresh_layout()

    view.set_dark_mode(True)
    view.refresh_layout()


def test_spectrometer_default_marginal_rays_reach_detector():
    """
    Verify that in default.json:
    1. S4 Aperture Stop clear aperture radius is respected by entrance pupil calculations (~7.11 mm, not 22.58 mm).
    2. Zero marginal rays are vignetted/blocked at S4.
    3. 100% of marginal rays at the primary design wavelength (1.225 um) successfully reach the detector (S27).
    """
    import os
    from optics_app.engine.models import SpectrometerSystem
    from optics_app.engine.tracer import OpticalSystem, Ray
    from optics_app.engine.system_config import SystemConfig
    from optics_app.engine.pupil_solve import calculate_entrance_pupil

    assert os.path.exists("default.json"), "default.json must exist in root"
    model = SpectrometerSystem.load_from_json("default.json")
    surfs = model.surfaces

    # 1. Check stop radius & Entrance Pupil
    s4 = surfs[4]
    assert s4.is_stop or "STOP" in s4.name.upper()
    z_ep, r_ep, stop_idx = calculate_entrance_pupil(surfs, wavelength_um=1.225)
    assert stop_idx == 4
    assert 6.0 < r_ep < 8.0, f"Entrance pupil radius should be ~7.11 mm, got {r_ep:.2f} mm"

    # 2. Trace Marginal Rays
    cfg = SystemConfig()
    cfg.from_dict(model.system_config)
    rays = cfg.generate_input_rays(surfs)
    marginal_rays = [r for r in rays if getattr(r, "is_marginal", False)]
    assert len(marginal_rays) > 0

    opt_sys = OpticalSystem(model.stack)

    hits_detector = 0
    primary_hits = 0
    primary_total = 0

    for r in marginal_rays:
        traced = opt_sys.trace_ray(Ray(pos=r.pos.copy(), dir=r.dir.copy(), wavelength=r.wavelength))
        assert len(traced.history) > 5, "Marginal ray must not be halted at S4 (Aperture Stop)"
        last_hit = traced.history[-1]
        last_s = last_hit["surface_index"]
        if last_s == 27:
            hits_detector += 1

        if abs(r.wavelength - 1.225) < 1e-4:
            primary_total += 1
            if last_s == 27 and not traced.vignetted:
                primary_hits += 1

    # At primary wavelength, 100% of marginal rays must reach detector unvignetted
    assert primary_total > 0
    assert primary_hits == primary_total, f"Expected all {primary_total} primary marginal rays to reach detector unvignetted, got {primary_hits}"

    # Overall across all wavelengths, over 80% reach detector
    ratio = hits_detector / len(marginal_rays)
    assert ratio >= 0.80, f"Expected >= 80% marginal rays reaching detector, got {hits_detector}/{len(marginal_rays)} ({ratio*100:.1f}%)"

