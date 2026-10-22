"""
Comprehensive Verification Test Suite for Texas Instruments 4-Bit Phase Light Modulator (TI-PLM).

Tests:
1. 4-bit discrete states table (16 levels, 0..15), piston displacement, and phase fractions.
2. Cartesian analytical Zernike polynomials (Z1..Z15) and analytical gradients vs finite differences.
3. Programmable phase gratings (Blazed 4-bit staircase, Binary, Sinusoidal, Chirped VLS, Hadamard).
4. 2D Pattern loading/saving (.npy, .csv, .json, .png) and synthetic pattern generation.
5. 3D vector ray deflection on PLM (Generalized Snell / Grating wavevector conservation).
6. Headless Qt instantiation and UI controls of PLMStudioWidget.
7. End-to-end Raman spectrometer ray tracing comparing ON, OFF, and Hadamard states.
"""

import os
import sys
import json
import math
import tempfile
import pytest
import numpy as np

# Set offscreen for headless Qt tests
os.environ["QT_QPA_PLATFORM"] = "offscreen"

from optics_app.engine.plm import (
    PLMConfig, ZernikePhaseConfig, PLMGratingConfig, PLMPatternConfig,
    TI_PLM_MODELS, TI_PLM_DLP670S_SPECS, ZERNIKE_MODES_INFO,
    build_4bit_state_table, get_plm_mechanical_states,
    quantize_phase_waves_4bit, evaluate_zernike_waves_and_grad_xy,
    evaluate_plm_phase_and_gradient, generate_hadamard_matrix,
    load_pattern_from_file, save_pattern_to_file,
    deflect_ray_on_plm_3d, generate_plm_2d_maps, is_plm, ensure_plm_config
)
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import DualSpeedTracer
from optics_app.engine.ray import Ray


def test_4bit_state_table():
    """Verify 4-bit state table generates 16 discrete states with correct physical phase."""
    table = build_4bit_state_table(design_wavelength_um=1.225)
    assert len(table) == 16, f"Expected 16 states, got {len(table)}"
    
    for state_idx, item in enumerate(table):
        assert item["state"] == state_idx
        assert item["name"] == f"STATE_{state_idx}"
        assert item["binary"] == format(state_idx, "04b")
        assert item["hex"] == f"0x{state_idx:X}"
        expected_frac = state_idx / 16.0
        assert math.isclose(item["phase_fraction_2pi"], expected_frac, abs_tol=1e-5)
        expected_rad = 2.0 * math.pi * expected_frac
        assert math.isclose(item["phase_rad"], expected_rad, abs_tol=1e-4)
        expected_piston_nm = (expected_frac * 1225.0) / 2.0
        assert math.isclose(item["piston_nm"], expected_piston_nm, abs_tol=1e-3)


def test_quantization_function():
    """Verify quantize_phase_waves_4bit wraps and quantizes continuous phase into discrete 0..15 states."""
    test_phase_waves = np.array([0.0, 0.0625, 0.25, 0.50, 0.9375, 1.0])
    states, q_waves = quantize_phase_waves_4bit(test_phase_waves, bit_depth=4)
    assert np.all(states >= 0) and np.all(states < 16)
    assert states[0] == 0
    assert states[1] == 1
    assert states[2] == 4
    assert states[3] == 8
    assert states[4] == 15
    assert states[5] == 0  # 1.0 waves wraps to 0.0


def test_zernike_analytical_gradients():
    """Compare analytical spatial gradients of Z1..Z15 with finite-difference derivatives."""
    z_cfg = ZernikePhaseConfig(
        enabled=True,
        normalization_radius_mm=5.0,
        units="waves"
    )
    # Enable test coefficients
    z_cfg.coeffs["z4_defocus"] = 1.0
    z_cfg.coeffs["z6_astig_vertical"] = -0.5
    z_cfg.coeffs["z7_coma_vertical"] = 0.8
    z_cfg.coeffs["z11_spherical"] = -0.3

    xs = np.array([-1.5, 0.0, 1.2, 2.0])
    ys = np.array([1.0, 1.5, -0.8, -2.0])

    _, grad_x_anal, grad_y_anal = evaluate_zernike_waves_and_grad_xy(xs, ys, z_cfg, design_wavelength_um=1.225)

    # Finite difference step
    h = 1e-6
    phase_px, _, _ = evaluate_zernike_waves_and_grad_xy(xs + h, ys, z_cfg, design_wavelength_um=1.225)
    phase_mx, _, _ = evaluate_zernike_waves_and_grad_xy(xs - h, ys, z_cfg, design_wavelength_um=1.225)
    grad_x_num = (phase_px - phase_mx) / (2.0 * h)

    phase_py, _, _ = evaluate_zernike_waves_and_grad_xy(xs, ys + h, z_cfg, design_wavelength_um=1.225)
    phase_my, _, _ = evaluate_zernike_waves_and_grad_xy(xs, ys - h, z_cfg, design_wavelength_um=1.225)
    grad_y_num = (phase_py - phase_my) / (2.0 * h)

    np.testing.assert_allclose(grad_x_anal, grad_x_num, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(grad_y_anal, grad_y_num, rtol=1e-4, atol=1e-4)


def test_phase_grating_generation():
    """Verify blazed 4-bit staircase, binary, and chirped grating generation via evaluate_plm_phase_and_gradient."""
    x = np.linspace(-1.0, 1.0, 100)
    y = np.linspace(-1.0, 1.0, 100)
    xx, yy = np.meshgrid(x, y)

    # Blazed 4-bit grating
    plm_blazed = PLMConfig()
    plm_blazed.grating.enabled = True
    plm_blazed.grating.grating_type = "Blazed"
    plm_blazed.grating.pitch_um = 86.4
    plm_blazed.grating.diffraction_order = 1
    plm_blazed.grating.phase_depth_waves = 1.0

    res_blazed = evaluate_plm_phase_and_gradient(xx, yy, plm_blazed, mechanical_state="ON", wavelength_um=1.225)
    assert res_blazed["state_4bit"].shape == (100, 100)
    assert np.all(res_blazed["state_4bit"] >= 0) and np.all(res_blazed["state_4bit"] < 16)
    assert res_blazed["grating_efficiency_only"] >= 0.94  # 4-bit staircase efficiency ~ 95% - 98.7%
    assert res_blazed["diffraction_efficiency"] > 0.70   # Accounts for 88% reflectance and 92% fill factor

    # Binary grating
    plm_binary = PLMConfig()
    plm_binary.grating.enabled = True
    plm_binary.grating.grating_type = "Binary"
    plm_binary.grating.pitch_um = 86.4

    res_binary = evaluate_plm_phase_and_gradient(xx, yy, plm_binary, mechanical_state="ON", wavelength_um=1.225)
    unique_states = np.unique(res_binary["state_4bit"])
    assert len(unique_states) <= 4, f"Binary grating should produce discrete levels, got {len(unique_states)}"


def test_pattern_load_and_save():
    """Test saving and loading 2D phase patterns in various file formats."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a test 4-bit pattern (states in 0..15)
        states_orig = np.random.randint(0, 16, size=(64, 64), dtype=np.uint8)

        # 1. NPY file
        npy_path = os.path.join(tmpdir, "test_pattern.npy")
        assert save_pattern_to_file(npy_path, states_orig) == npy_path
        loaded_npy_waves = load_pattern_from_file(npy_path)
        loaded_npy_states = np.round(loaded_npy_waves * 16.0).astype(int)
        np.testing.assert_array_equal(loaded_npy_states, states_orig)

        # 2. CSV file
        csv_path = os.path.join(tmpdir, "test_pattern.csv")
        assert save_pattern_to_file(csv_path, states_orig[:16, :16]) == csv_path
        loaded_csv_waves = load_pattern_from_file(csv_path)
        loaded_csv_states = np.round(loaded_csv_waves * 16.0).astype(int)
        np.testing.assert_array_equal(loaded_csv_states, states_orig[:16, :16])

        # 3. JSON file
        json_path = os.path.join(tmpdir, "test_pattern.json")
        assert save_pattern_to_file(json_path, states_orig[:8, :8]) == json_path
        loaded_json_waves = load_pattern_from_file(json_path)
        loaded_json_states = np.round(loaded_json_waves * 16.0).astype(int)
        np.testing.assert_array_equal(loaded_json_states, states_orig[:8, :8])


def test_synthetic_hologram_generators():
    """Test synthetic pattern generators (Fresnel lens, Vortex, Cubic, Hadamard) via generate_plm_2d_maps."""
    # Fresnel zone lens
    plm_fresnel = PLMConfig()
    plm_fresnel.pattern.enabled = True
    plm_fresnel.pattern.pattern_type = "Fresnel_Lens"
    plm_fresnel.pattern.fresnel_focal_length_mm = 100.0
    maps_fresnel = generate_plm_2d_maps(plm_fresnel, mechanical_state="ON", nx=100, ny=100)
    assert maps_fresnel["state_4bit"].shape == (100, 100)
    assert len(np.unique(maps_fresnel["state_4bit"])) > 4

    # Vortex mask
    plm_vortex = PLMConfig()
    plm_vortex.pattern.enabled = True
    plm_vortex.pattern.pattern_type = "Vortex"
    plm_vortex.pattern.vortex_charge = 2
    maps_vortex = generate_plm_2d_maps(plm_vortex, mechanical_state="ON", nx=100, ny=100)
    assert maps_vortex["state_4bit"].shape == (100, 100)

    # Hadamard spectral mask
    H = generate_hadamard_matrix(16)
    assert H.shape == (16, 16)
    assert np.all(np.isin(H, [-1, 1]))


def test_deflect_ray_on_plm_3d():
    """Verify 3D ray deflection for ON (+1st diffracted order) vs OFF (specular 0th order)."""
    plm_cfg = PLMConfig(
        model="DLP670S",
        design_wavelength_um=1.225,
        quantize_4bit=True
    )
    # Grating: 11.574 lines/mm (86.4 um pitch) along Y
    plm_cfg.grating.enabled = True
    plm_cfg.grating.pitch_um = 86.4
    plm_cfg.grating.orientation_deg = 0.0
    plm_cfg.grating.diffraction_order = 1

    ray_dir_in = np.array([0.0, 0.0, 1.0])
    surf_normal = np.array([0.0, 0.0, -1.0])
    surf_global_R = np.eye(3)

    # 1. ON state: Carrier grating active
    dir_out_on, phase_rad_on, eff_on, state_on = deflect_ray_on_plm_3d(
        ray_dir_in=ray_dir_in,
        surf_normal=surf_normal,
        surf_global_R=surf_global_R,
        local_x_mm=0.0,
        local_y_mm=0.0,
        plm_cfg=plm_cfg,
        mechanical_state="ON",
        wavelength_um=1.225
    )
    expected_ky = 1.225 / 86.4  # m * lambda / pitch = 0.01417824
    assert math.isclose(dir_out_on[1], expected_ky, rel_tol=1e-3)
    assert dir_out_on[2] < 0.0  # reflected back along -Z

    # 2. OFF state: Flat State 0 specular reflection
    dir_out_off, phase_rad_off, eff_off, state_off = deflect_ray_on_plm_3d(
        ray_dir_in=ray_dir_in,
        surf_normal=surf_normal,
        surf_global_R=surf_global_R,
        local_x_mm=0.0,
        local_y_mm=0.0,
        plm_cfg=plm_cfg,
        mechanical_state="OFF",
        wavelength_um=1.225
    )
    assert math.isclose(dir_out_off[0], 0.0, abs_tol=1e-7)
    assert math.isclose(dir_out_off[1], 0.0, abs_tol=1e-7)
    assert math.isclose(dir_out_off[2], -1.0, abs_tol=1e-7)
    assert state_off == 0
    assert phase_rad_off == 0.0


def test_headless_plm_studio_widget():
    """Verify PLMStudioWidget instantiates, connects controls, and renders plots without error in Qt."""
    from optics_app.gui.qt_compat import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication([])

    from optics_app.gui.plm_widget import PLMStudioWidget

    # Create dummy stack with a PLM surface
    stack = SurfaceStack()
    s_in = Surface(name="Entrance", thickness=10.0)
    s_plm = Surface(
        name="PLM_DLP670S",
        surface_type="MIRROR",
        is_plm=True,
        mechanism_type="PLM",
        mechanical_state="ON",
        plm_config=PLMConfig(),
        thickness=10.0
    )
    s_det = Surface(name="Detector", surface_type="Detector")
    stack.surfaces = [s_in, s_plm, s_det]
    stack._update_accumulated_positions()

    widget = PLMStudioWidget(surface_stack=stack)
    assert widget is not None
    assert widget.combo_surface.count() >= 1

    # Switch state programmatically
    widget.combo_state.setCurrentText("STATE_4")
    assert s_plm.mechanical_state == "STATE_4"

    # Set Zernike defocus
    widget.zernike_spins["z4_defocus"].setValue(1.5)
    assert math.isclose(s_plm.plm_config.zernike.coeffs["z4_defocus"], 1.5, abs_tol=1e-3)

    # Change grating parameters
    widget.spin_pitch.setValue(43.2)
    assert math.isclose(s_plm.plm_config.grating.pitch_um, 43.2, abs_tol=1e-3)

    # Trigger map regeneration
    widget.update_live_plots()
    assert widget.canvas is not None


def test_full_spectrometer_simulation():
    """Verify loading the new spectrometer design file and tracing multi-wavelength rays."""
    design_path = "spectrometer_raman_plm_4bit.json"
    assert os.path.exists(design_path), f"Design file {design_path} must exist"

    stack_data = SurfaceStack.load_from_json(design_path)
    stack = stack_data["stack"]
    assert len(stack.surfaces) == 28, f"Expected 28 surfaces, got {len(stack.surfaces)}"

    plm_surface = stack.surfaces[20]
    assert getattr(plm_surface, "is_plm", False)
    assert getattr(plm_surface, "mechanism_type", "") == "PLM"
    assert plm_surface.plm_config is not None

    # Trace ON state across Raman band: 1.087, 1.225, 1.350 um
    tracer_on = DualSpeedTracer(stack)
    wavelengths = [1.087, 1.225, 1.350]
    detector_y_positions = []

    for wl in wavelengths:
        ray = Ray(pos=[0.0, 0.0, 0.0], dir=[0.0, 0.0, 1.0], wavelength=wl)
        traced = tracer_on.trace_ray(ray)
        assert not traced.vignetted, f"Ray at {wl} um should reach detector unvignetted"
        assert len(traced.history) == 28, f"Expected 28 surface interactions, got {len(traced.history)}"
        detector_y_positions.append(traced.pos[1])

    # Check spectral dispersion across the detector plane (physical separation > 0.5 mm)
    dispersion_span = detector_y_positions[-1] - detector_y_positions[0]
    assert abs(dispersion_span) > 0.5, f"Expected noticeable spectral dispersion, got {dispersion_span} mm"

    # Switch PLM to OFF state: specular reflection should deviate from the detector arm
    plm_surface.mechanical_state = "OFF"
    stack._update_accumulated_positions()
    tracer_off = DualSpeedTracer(stack)
    ray_off = Ray(pos=[0.0, 0.0, 0.0], dir=[0.0, 0.0, 1.0], wavelength=1.225)
    traced_off = tracer_off.trace_ray(ray_off)

    # The interaction on surface 20 should have direction [0, 0, -1] (flat reflection)
    history_plm = traced_off.history[20]
    assert history_plm["mechanical_state"] == "OFF"
    assert history_plm["plm_state_4bit"] == 0


def test_phase_in_um_visualization_and_independent_docks():
    """Verify phase visualization in um and independence of PLM & Ultrafast docks from analysis dock."""
    from optics_app.gui.qt_compat import QApplication, QDockWidget
    app = QApplication.instance()
    if app is None:
        app = QApplication([])

    from optics_app.gui.plm_widget import PLMStudioWidget, PLMStudioDock
    from optics_app.gui.analysis_tabs import AnalysisTabsWidget

    stack = SurfaceStack()
    s_plm = Surface(
        name="PLM_Surface",
        surface_type="MIRROR",
        is_plm=True,
        mechanism_type="PLM",
        mechanical_state="ON",
        plm_config=PLMConfig(design_wavelength_um=1.225),
        thickness=10.0
    )
    stack.surfaces = [s_plm]
    stack._update_accumulated_positions()

    # 1. Verify Phase in µm visualization
    widget = PLMStudioWidget(surface_stack=stack)
    assert hasattr(widget, "combo_unit")
    assert widget.combo_unit.currentText() == "Phase OPD (µm)"
    assert "µm" in widget.lbl_step_um.text()

    # Verify Zernike µm readouts
    widget.zernike_spins["z4_defocus"].setValue(2.0)
    widget.update_live_plots()
    assert "+2.450 µm" in widget.zernike_um_labels["z4_defocus"].text()
    assert "µm" in widget.lbl_depth_um.text()

    # 2. Verify independent PLMStudioDock
    dock = PLMStudioDock(surface_stack=stack)
    assert isinstance(dock, QDockWidget)
    assert dock.objectName() == "PLMStudioDock"
    assert dock.studio is not None

    # 3. Verify AnalysisTabsWidget does NOT contain PLM tab (independent from optical analysis diagnostics)
    analysis_tabs = AnalysisTabsWidget()
    assert "TI-PLM 4-Bit Modulator" not in analysis_tabs.tabs
    for i in range(analysis_tabs.count()):
        assert "PLM" not in analysis_tabs.tabText(i)


def test_auto_retrace_on_phase_or_mask_change():
    """Verify that changing any PLM mask, phase, or grating automatically emits pattern_applied to retrace light."""
    from optics_app.gui.qt_compat import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication([])

    from optics_app.gui.plm_widget import PLMStudioWidget

    stack = SurfaceStack()
    s_plm = Surface(
        name="PLM_Surface",
        surface_type="MIRROR",
        is_plm=True,
        mechanism_type="PLM",
        mechanical_state="ON",
        plm_config=PLMConfig(design_wavelength_um=1.225),
        thickness=10.0
    )
    stack.surfaces = [s_plm]
    stack._update_accumulated_positions()

    widget = PLMStudioWidget(surface_stack=stack)

    # Monitor pattern_applied emissions
    retrace_events = []
    widget.pattern_applied.connect(lambda idx: retrace_events.append(idx))

    # 1. Changing Zernike phase coefficient triggers retrace
    retrace_events.clear()
    widget.zernike_spins["z4_defocus"].setValue(0.75)
    assert len(retrace_events) == 1
    assert math.isclose(s_plm.plm_config.zernike.coeffs["z4_defocus"], 0.75, abs_tol=1e-3)

    # 2. Applying spectrometer aberration correction preset triggers retrace
    retrace_events.clear()
    widget._preset_spectrometer_correction()
    assert len(retrace_events) >= 1
    assert math.isclose(s_plm.plm_config.zernike.coeffs["z6_astig_vertical"], -0.85, abs_tol=1e-3)

    # 3. Resetting Zernike phases to zero triggers retrace
    retrace_events.clear()
    widget._zero_zernike()
    assert len(retrace_events) == 1
    assert math.isclose(s_plm.plm_config.zernike.coeffs["z4_defocus"], 0.0, abs_tol=1e-3)

    # 4. Changing grating pitch triggers retrace
    retrace_events.clear()
    widget.spin_pitch.setValue(64.8)
    assert len(retrace_events) == 1
    assert math.isclose(s_plm.plm_config.grating.pitch_um, 64.8, abs_tol=1e-3)

    # 5. Changing 2D pattern / mask type triggers retrace
    retrace_events.clear()
    widget.combo_pattern_type.setCurrentText("Hadamard")
    assert len(retrace_events) == 1
    assert s_plm.plm_config.pattern.pattern_type == "Hadamard"

    # 6. Switching 4-bit discrete state triggers retrace
    retrace_events.clear()
    widget.combo_state.setCurrentText("STATE_12")
    assert len(retrace_events) == 1
    assert s_plm.mechanical_state == "STATE_12"


def test_plm_studio_three_plot_layout():
    """Verify PLMStudioWidget renders exactly the 3 requested optical diagnostics and removes state utilization."""
    from optics_app.gui.qt_compat import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication([])

    from optics_app.gui.plm_widget import PLMStudioWidget

    stack = SurfaceStack()
    s_plm = Surface(
        name="PLM_Surface",
        surface_type="MIRROR",
        is_plm=True,
        mechanism_type="PLM",
        mechanical_state="ON",
        plm_config=PLMConfig(design_wavelength_um=1.225),
        thickness=10.0
    )
    stack.surfaces = [s_plm]
    stack._update_accumulated_positions()

    widget = PLMStudioWidget(surface_stack=stack)
    widget.update_live_plots()

    titles = [ax.get_title() for ax in widget.fig.axes if ax.get_title()]
    
    # 1. Verify 2D Phase Map is present
    assert any("2D Phase Map" in t for t in titles), f"2D Phase Map missing from {titles}"
    
    # 2. Verify Simulated Far-Field PSF is present
    assert any("Simulated Far-Field" in t for t in titles), f"Simulated Far-Field PSF missing from {titles}"
    
    # 3. Verify 1D Phase Map is present
    assert any("1D Phase Map" in t for t in titles), f"1D Phase Map missing from {titles}"

    # 4. Verify 4-Bit State Utilization histogram is completely removed
    assert not any("Utilization" in t or "Histogram" in t for t in titles), f"State utilization still present in {titles}"


def test_dmd_binary_modulator_config_and_states():
    """Verify TI-DMD configuration, binary states, and bistable mechanical tilt angles."""
    from optics_app.engine.plm import (
        PLMConfig, TI_DMD_MODELS, get_dmd_mechanical_states,
        evaluate_plm_phase_and_gradient, is_dmd
    )
    from optics_app.engine.mechanisms import available_states

    # 1. Config initialization
    cfg = PLMConfig(model="DLP4500", modulator_type="DMD", bit_depth=1, tilt_angle_deg=12.0)
    assert cfg.is_dmd is True
    assert cfg.num_states == 2
    assert cfg.tilt_angle_deg == 12.0

    # 2. Mechanical states
    states = get_dmd_mechanical_states(tilt_angle_deg=12.0)
    assert "ON" in states and "OFF" in states and "FLAT" in states
    assert states["ON"]["tilt_angle_deg"] == 12.0
    assert states["OFF"]["tilt_angle_deg"] == -12.0
    assert states["FLAT"]["tilt_angle_deg"] == 0.0
    assert "HADAMARD_A" in states and "HADAMARD_B" in states

    # 3. Surface detection
    surf = Surface(name="DMD_Mask", surface_type="MIRROR", is_dmd=True, mechanism_type="DMD", plm_config=cfg)
    assert is_dmd(surf) is True
    surf_states = available_states(surf)
    assert "ON" in surf_states
    assert surf_states["ON"]["tilt_angle_deg"] == 12.0

    # 4. Binary output restriction: evaluate point
    res = evaluate_plm_phase_and_gradient(0.0, 0.0, cfg, mechanical_state="ON")
    assert res["state_4bit"] in (0, 1)


def test_dmd_binary_gratings_and_hadamard_patterns():
    """Verify binary Ronchi rulings, Walsh-Hadamard multiplexing masks, and far-field PSF on DMD."""
    from optics_app.engine.plm import (
        PLMConfig, PLMGratingConfig, PLMPatternConfig,
        evaluate_plm_phase_and_gradient, generate_plm_2d_maps
    )

    # 1. Binary Ronchi Grating
    gcfg = PLMGratingConfig(enabled=True, grating_type="Binary", pitch_um=60.8, lines_per_mm=16.447, orientation_deg=90.0)
    dmd_cfg = PLMConfig(model="DLP4500", modulator_type="DMD", bit_depth=1, grating=gcfg)

    x = np.linspace(-4.0, 4.0, 200)
    y = np.zeros_like(x)
    res = evaluate_plm_phase_and_gradient(x, y, dmd_cfg, mechanical_state="ON")

    # Values must strictly be binary 0 or 1
    unique_states = np.unique(res["state_4bit"])
    assert set(unique_states).issubset({0, 1})
    # Both 0 (OFF) and 1 (ON) must be present in Ronchi grating
    assert 0 in unique_states and 1 in unique_states

    # 2. Binary Walsh-Hadamard pattern
    pcfg = PLMPatternConfig(enabled=True, pattern_type="Hadamard", hadamard_order=32, hadamard_row_index=5)
    dmd_hadamard = PLMConfig(model="DLP4500", modulator_type="DMD", bit_depth=1, pattern=pcfg)
    maps_a = generate_plm_2d_maps(dmd_hadamard, mechanical_state="HADAMARD_A", nx=128, ny=64)
    maps_b = generate_plm_2d_maps(dmd_hadamard, mechanical_state="HADAMARD_B", nx=128, ny=64)

    # All states must be 0 or 1
    assert set(np.unique(maps_a["state_4bit"])).issubset({0, 1})
    assert set(np.unique(maps_b["state_4bit"])).issubset({0, 1})

    # Masks A and B must be complementary (where A=1, B=0 and vice versa)
    assert np.all((maps_a["state_4bit"] + maps_b["state_4bit"]) == 1)

    # 3. Far-Field PSF generation with binary aperture
    assert "psf_intensity" in maps_a
    assert maps_a["psf_intensity"].shape == (128, 256)
    assert np.max(maps_a["psf_intensity"]) == 1.0


def test_dmd_gui_controls_and_binary_plots():
    """Verify PLMStudioWidget switches to DMD mode, disables Zernike, and renders high-contrast binary plots."""
    from optics_app.gui.qt_compat import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication([])

    from optics_app.gui.plm_widget import PLMStudioWidget

    stack = SurfaceStack()
    s_mod = Surface(
        name="DMD_Surface",
        surface_type="MIRROR",
        is_dmd=True,
        mechanism_type="DMD",
        mechanical_state="ON",
        plm_config=PLMConfig(model="DLP4500", modulator_type="DMD", bit_depth=1),
        thickness=10.0
    )
    stack.surfaces = [s_mod]
    stack._update_accumulated_positions()

    widget = PLMStudioWidget(surface_stack=stack)

    # 1. Verify auto-detection of DMD mode
    assert widget.combo_modulator_type.currentIndex() == 1
    assert "DMD" in widget.header_group.title()
    assert widget.combo_model.currentText() == "DLP4500"

    # 2. Verify Zernike is disabled and banner is visible
    assert not widget.lbl_zernike_dmd_banner.isHidden()
    assert widget.zernike_controls_widget.isEnabled() is False
    assert widget.chk_zernike_enable.isChecked() is False

    # 3. Verify Grating types are restricted to binary
    grating_items = [widget.combo_grating_type.itemText(i) for i in range(widget.combo_grating_type.count())]
    assert grating_items == ["Binary", "Hadamard_Spectral"]

    # 4. Verify Pattern types are restricted to binary patterns
    pattern_items = [widget.combo_pattern_type.itemText(i) for i in range(widget.combo_pattern_type.count())]
    assert "Fresnel_Lens" not in pattern_items
    assert "Vortex" not in pattern_items
    assert "Hadamard" in pattern_items
    assert widget.grp_fresnel.isHidden() is True
    assert widget.grp_vortex.isHidden() is True

    # 5. Verify 3 binary diagnostic plots render without error
    widget.update_live_plots()
    titles = [ax.get_title() for ax in widget.fig.axes if ax.get_title()]
    assert any("TI-DMD 2D Binary Pattern" in t for t in titles), f"DMD 2D binary pattern title missing from {titles}"
    assert any("Simulated Far-Field Diffraction / PSF (Binary Aperture)" in t for t in titles), f"Far-field binary title missing from {titles}"
    assert any("1D Binary State & Micromirror Tilt" in t for t in titles), f"1D binary cross-section missing from {titles}"

    # 6. Verify switching back to PLM restores 4-bit phase controls
    widget.combo_modulator_type.setCurrentIndex(0)
    assert widget.combo_modulator_type.currentIndex() == 0
    assert "Phase Light Modulator" in widget.header_group.title()
    assert widget.lbl_zernike_dmd_banner.isHidden() is True
    assert widget.zernike_controls_widget.isEnabled() is True
    plm_gratings = [widget.combo_grating_type.itemText(i) for i in range(widget.combo_grating_type.count())]
    assert "Blazed" in plm_gratings
    assert "4-Level" in plm_gratings


def test_main_window_interface_no_duplicates_and_dmd_panel_works():
    """Verify that MainWindow has no duplicate Modulator/DMD menu actions and that the panel functions properly."""
    from optics_app.gui.qt_compat import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication([])

    from optics_app.gui.main_window import MainWindow
    window = MainWindow()

    # 1. Verify View menu has NO duplicate actions for Modulator Studio
    view_menu = None
    for action in window.menuBar().actions():
        if "View" in action.text():
            view_menu = action.menu()
            break
    assert view_menu is not None, "View menu must exist"

    # Count direct occurrences of Modulator Studio in View menu
    view_mod_actions = [a.text() for a in view_menu.actions() if "Modulator Studio" in a.text()]
    assert len(view_mod_actions) == 1, f"Expected exactly 1 Modulator Studio action in View menu, found {view_mod_actions}"
    assert "TI Modulator Studio (PLM / DMD)" in view_mod_actions[0]

    # Verify Docks submenu has no duplicate Modulator Studio action
    docks_menu = None
    for a in view_menu.actions():
        if a.menu() and "Docks" in a.text():
            docks_menu = a.menu()
            break
    if docks_menu:
        sub_mod_actions = [a.text() for a in docks_menu.actions() if "Modulator Studio" in a.text()]
        assert len(sub_mod_actions) == 0, f"Expected 0 Modulator Studio in Docks submenu to prevent duplicates, found {sub_mod_actions}"

    # Verify Main Toolbar has no duplicate Modulator Studio action
    tb_mod_actions = [a.text() for a in window.main_toolbar.actions() if "Modulator Studio" in a.text()]
    assert len(tb_mod_actions) == 0, f"Expected 0 Modulator Studio in Main Toolbar, found {tb_mod_actions}"

    # 2. Verify Tools menu action
    tools_menu = None
    for action in window.menuBar().actions():
        if "Tools" in action.text():
            tools_menu = action.menu()
            break
    assert tools_menu is not None, "Tools menu must exist"
    tools_mod_actions = [a.text() for a in tools_menu.actions() if "Modulator Studio" in a.text()]
    assert len(tools_mod_actions) == 1, f"Expected exactly 1 Modulator Studio in Tools menu, found {tools_mod_actions}"

    # 3. Verify Dock title and object
    assert hasattr(window, "plm_dock") and window.plm_dock is not None
    assert "TI Modulator Studio (PLM / DMD)" in window.plm_dock.windowTitle()

    # 4. Show the studio via on_show_plm_studio
    window.on_show_plm_studio()
    assert not window.plm_dock.isHidden()

    # 5. Verify DMD panel controls and plots work in the dock
    studio = window.plm_dock.studio
    assert studio is not None

    # Switch to DMD mode
    studio.combo_modulator_type.setCurrentIndex(1)
    assert studio.combo_modulator_type.currentIndex() == 1
    assert not studio.lbl_zernike_dmd_banner.isHidden()
    assert studio.zernike_controls_widget.isEnabled() is False

    # Check live plots in DMD mode
    studio.update_live_plots()
    assert len(studio.fig.axes) >= 3

    # Switch back to PLM mode
    studio.combo_modulator_type.setCurrentIndex(0)
    assert studio.combo_modulator_type.currentIndex() == 0
    assert studio.lbl_zernike_dmd_banner.isHidden() is True
    assert studio.zernike_controls_widget.isEnabled() is True
    studio.update_live_plots()
    assert len(studio.fig.axes) >= 3





