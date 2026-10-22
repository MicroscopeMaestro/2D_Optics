"""Comprehensive unit test suite for Milestones 2-4:
- High-NA Olympus Objective Prescriptions (20x 1.0 NA W & 25x 1.05 NA W)
- Dynamic Objective Selection in Folded Real-World SRS Template
- Dedicated Suite of 5 Interactive Tools (Setup Builder, Galvo Diagnostic, Spectral Focusing, Lock-In, Bench Alignment)
- SRSSimulationTools Facade
- Dual-Tree Code Parity between optics_app and v2
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from optics_app.analysis.ultrafast_nonlinear import (
    OBJECTIVE_CATALOG,
    SCAN_LENS_CATALOG,
    TUBE_LENS_CATALOG,
    ThorlabsSRSMicroscopeConfig,
    ThorlabsSRSMicroscopeSimulator,
    OilSampleType,
    OilRamanStandard,
    SRSPhotodiodeSignal,
    SRSScanFieldResult,
    PostSampleSRSProbeResult,
    build_folded_real_world_srs_template,
    build_srs_microscope_system,
    SRSSetupBuilderTool,
    SRSGalvoFieldDiagnosticTool,
    SRSSpectralFocusingTool,
    SRSSignalLockInTool,
    SRSBenchAlignmentTool,
    SRSSimulationTools,
)

import v2.analysis.ultrafast_nonlinear as v2_srs


# ==============================================================================
# 1. HIGH-NA OLYMPUS OBJECTIVE PRESCRIPTIONS & DYNAMIC TEMPLATE SELECTION (M2)
# ==============================================================================

def test_olympus_20x_1na_catalog_and_selection():
    """Verify Olympus 20x 1.0 NA W objective prescriptions in catalog and template."""
    assert "Olympus XLUMPLFLN20XW (20x NA 1.00 W)" in OBJECTIVE_CATALOG
    assert "Olympus XLUMPLFLN20XW (20x NA 1.0 W Multiphoton)" in OBJECTIVE_CATALOG
    assert "Olympus UAPON20XW340 (20x NA 1.00 W)" in OBJECTIVE_CATALOG

    info = OBJECTIVE_CATALOG["Olympus XLUMPLFLN20XW (20x NA 1.00 W)"]
    assert info["numerical_aperture"] == 1.00
    assert info["focal_length_mm"] == 9.0
    assert info["pupil_diameter_mm"] == 18.0
    assert info["working_distance_mm"] == 2.0
    assert "Water" in info["immersion"]

    cfg = ThorlabsSRSMicroscopeConfig(objective_model="Olympus XLUMPLFLN20XW (20x NA 1.00 W)")
    stack = build_folded_real_world_srs_template(cfg)
    assert len(stack.surfaces) == 37

    # Find objective stop and objective surface
    stop_surf = next(s for s in stack.surfaces if str(getattr(s, "surface_type", "")).upper() == "STOP" and "Objective" in s.name)
    assert math.isclose(stop_surf.semi_diameter, 9.0, abs_tol=1e-5)  # EPD = 18.0 -> semi-diam = 9.0

    obj_surf = next(s for s in stack.surfaces if "Objective (" in s.name)
    assert "XLUMPLFLN20XW" in obj_surf.name


def test_olympus_25x_105na_catalog_and_selection():
    """Verify Olympus 25x 1.05 NA W multiphoton objective prescription."""
    info = OBJECTIVE_CATALOG["Olympus XLPLN25XWMP2 (25x NA 1.05 W Multiphoton)"]
    assert info["numerical_aperture"] == 1.05
    assert info["focal_length_mm"] == 8.0
    assert info["pupil_diameter_mm"] == 16.8
    assert info["working_distance_mm"] == 2.0

    cfg = ThorlabsSRSMicroscopeConfig(objective_model="Olympus XLPLN25XWMP2 (25x NA 1.05 W Multiphoton)")
    stack = build_folded_real_world_srs_template(cfg)
    stop_surf = next(s for s in stack.surfaces if str(getattr(s, "surface_type", "")).upper() == "STOP" and "Objective" in s.name)
    assert math.isclose(stop_surf.semi_diameter, 8.4, abs_tol=1e-5)  # EPD = 16.8 -> semi-diam = 8.4

    obj_surf = next(s for s in stack.surfaces if "Objective (" in s.name)
    assert "XLPLN25XWMP2" in obj_surf.name


# ==============================================================================
# 2. DEDICATED SUITE OF INTERACTIVE SRS TOOLS (M4 / R8)
# ==============================================================================

def test_tool1_srs_setup_builder():
    """Verify SRSSetupBuilderTool (Tool 1) system generation and validation."""
    builder = SRSSetupBuilderTool()
    assert len(builder.get_available_objectives()) >= 5
    assert len(builder.get_available_scan_lenses()) >= 2
    assert len(builder.get_available_tube_lenses()) >= 2

    sys = builder.build_system(objective_model="Olympus XLUMPLFLN20XW (20x NA 1.00 W)")
    assert sys is not None
    assert len(sys.surface_stack.surfaces) == 37

    stack = builder.build_surface_stack()
    validation = builder.validate_prescription(stack)
    assert validation["is_valid"] is True
    assert validation["surface_count"] == 37
    assert validation["has_objective"] is True
    assert validation["has_aperture_stop"] is True
    assert validation["has_detector"] is True
    assert validation["has_galvos"] is True
    assert validation["has_dichroic_combiner"] is True
    assert validation["dump_count"] >= 3


def test_tool2_srs_galvo_field_diagnostic():
    """Verify SRSGalvoFieldDiagnosticTool (Tool 2) scanning diagnostics."""
    tool = SRSGalvoFieldDiagnosticTool()
    cfg = ThorlabsSRSMicroscopeConfig()

    field_map = tool.simulate_field_map(cfg, grid_points=5)
    assert field_map["scan_efficiency_map"].shape == (5, 5)
    assert field_map["srs_signal_map_v"].shape == (5, 5)
    assert field_map["chromatic_walk_off_map_um"].shape == (5, 5)

    pt = tool.evaluate_scan_point(cfg, 1.5, 2.0)
    assert isinstance(pt, SRSScanFieldResult)
    assert pt.sample_x_um != 0.0
    assert pt.sample_y_um != 0.0

    fov = tool.compute_fov_dimensions(cfg)
    assert fov["fov_x_um"] > 100.0
    assert fov["fov_y_um"] > 100.0
    assert fov["fov_diagonal_um"] > fov["fov_x_um"]

    aberr = tool.compute_aberration_profile(cfg, max_angle_deg=4.0, steps=9)
    assert len(aberr["angles_deg"]) == 9
    assert aberr["defocus_um"][-1] > aberr["defocus_um"][0]
    assert aberr["scan_efficiency"][-1] < aberr["scan_efficiency"][0]


def test_tool3_srs_spectral_focusing():
    """Verify SRSSpectralFocusingTool (Tool 3) dispersion and delay sweep."""
    tool = SRSSpectralFocusingTool()
    cfg = ThorlabsSRSMicroscopeConfig()

    disp = tool.compute_dispersion_parameters(cfg)
    assert disp["chirped_pump_fwhm_ps"] > 0.5
    assert disp["chirped_stokes_fwhm_ps"] > 0.5
    assert 0.8 <= disp["chirp_matching_ratio"] <= 1.2
    assert disp["spectral_resolution_cm1"] > 0.0

    probe = tool.probe_at_delay(cfg, delay_ps=0.0)
    assert isinstance(probe, PostSampleSRSProbeResult)
    assert probe.fractional_srl > 0.0
    assert probe.transmitted_pump_power_mw > 0.0

    scan_res = tool.simulate_delay_scan(cfg, 0.0, 0.0)
    assert len(scan_res["wavenumbers_cm1"]) == cfg.delay_samples
    assert len(scan_res["on_axis_lockin_voltage_v"]) == cfg.delay_samples

    peaks = tool.find_resonant_delays(cfg, OilSampleType.MINERAL_OIL)
    assert len(peaks) >= 3
    for p in peaks:
        assert "wavenumber_cm1" in p
        assert "required_delay_ps" in p
        assert "stage_position_mm" in p


def test_tool4_srs_signal_lockin():
    """Verify SRSSignalLockInTool (Tool 4) RF lock-in demodulation & SNR."""
    tool = SRSSignalLockInTool()
    cfg = ThorlabsSRSMicroscopeConfig()

    sig = tool.process_signal(cfg, fractional_srl=5e-5)
    assert isinstance(sig, SRSPhotodiodeSignal)
    assert sig.lockin_magnitude_r_v > 0.0
    assert sig.snr_db > 10.0

    ptrain = tool.demodulate_pulse_train(rep_rate_mhz=80.0, mod_freq_mhz=10.0, duration_ns=100.0)
    assert len(ptrain["time_ns"]) > 0
    assert len(ptrain["modulated_signal"]) > 0

    p_curve = tool.snr_vs_laser_power_curve(cfg, pump_powers_mw=np.array([10.0, 20.0, 40.0]))
    assert len(p_curve["snr_db"]) == 3
    assert p_curve["snr_db"][-1] > p_curve["snr_db"][0]

    limit = tool.evaluate_detection_limit(cfg, target_snr_db=3.0)
    assert limit["min_detectable_voltage_v"] > 0.0
    assert limit["min_detectable_fractional_srl"] > 0.0


def test_tool5_srs_bench_alignment():
    """Verify SRSBenchAlignmentTool (Tool 5) alignment and optical density checks."""
    tool = SRSBenchAlignmentTool()
    cfg_rel = ThorlabsSRSMicroscopeConfig(scanning_architecture="4f-Relayed (Zero Walk-Off)")

    walk = tool.check_pupil_walk_off(cfg_rel, angle_deg=3.0)
    assert walk["is_zero_walk_off"] is True
    assert walk["is_strictly_conjugated"] is True
    assert walk["pupil_fill_factor"] > 0.85

    od = tool.evaluate_optical_density_stack(cfg_rel)
    assert od["total_stokes_rejection_od"] >= 12.0
    assert od["stokes_optical_leakage_fraction"] <= 1e-12
    assert od["safe_for_lockin"] is True

    sys = SRSSetupBuilderTool.build_system(cfg_rel, folded=True)
    clearances = tool.verify_37_surface_clearances(sys.surface_stack)
    assert clearances["is_37_surface_train"] is True
    assert clearances["collision_free_clearance"] is True

    diag = tool.run_full_bench_diagnostic(cfg_rel)
    assert diag["overall_status"] == "ALIGNED_AND_VERIFIED"
    assert "laser_delivery" in diag


def test_srs_simulation_tools_facade():
    """Verify SRSSimulationTools facade links to all 5 dedicated tools."""
    assert SRSSimulationTools.SetupBuilder is SRSSetupBuilderTool
    assert SRSSimulationTools.GalvoDiagnostic is SRSGalvoFieldDiagnosticTool
    assert SRSSimulationTools.SpectralFocusing is SRSSpectralFocusingTool
    assert SRSSimulationTools.SignalLockIn is SRSSignalLockInTool
    assert SRSSimulationTools.BenchAlignment is SRSBenchAlignmentTool


# ==============================================================================
# 3. DUAL-TREE PARITY TESTS FOR TOOLS (M0 / F18)
# ==============================================================================

def test_v2_dual_tree_parity_for_tools():
    """Verify identical behavior between optics_app and v2 tools."""
    assert hasattr(v2_srs, "SRSSetupBuilderTool")
    assert hasattr(v2_srs, "SRSGalvoFieldDiagnosticTool")
    assert hasattr(v2_srs, "SRSSpectralFocusingTool")
    assert hasattr(v2_srs, "SRSSignalLockInTool")
    assert hasattr(v2_srs, "SRSBenchAlignmentTool")
    assert hasattr(v2_srs, "SRSSimulationTools")

    cfg1 = ThorlabsSRSMicroscopeConfig()
    cfg2 = v2_srs.ThorlabsSRSMicroscopeConfig()

    d1 = SRSGalvoFieldDiagnosticTool.compute_fov_dimensions(cfg1)
    d2 = v2_srs.SRSGalvoFieldDiagnosticTool.compute_fov_dimensions(cfg2)
    assert math.isclose(d1["fov_x_um"], d2["fov_x_um"], rel_tol=1e-5)

    disp1 = SRSSpectralFocusingTool.compute_dispersion_parameters(cfg1)
    disp2 = v2_srs.SRSSpectralFocusingTool.compute_dispersion_parameters(cfg2)
    assert math.isclose(disp1["chirp_matching_ratio"], disp2["chirp_matching_ratio"], rel_tol=1e-5)


def test_ultrafast_dock_exposes_srs_tools_in_both_trees():
    """Verify UltrafastAnalysisDock exposes the 5 SRS tools in both optics_app and v2."""
    from optics_app.gui.qt_compat import QApplication
    _app = QApplication.instance() or QApplication([])

    from optics_app.gui.ultrafast_dock import UltrafastAnalysisDock as Dock1
    from v2.gui.ultrafast_dock import UltrafastAnalysisDock as Dock2

    d1 = Dock1()
    d2 = Dock2()

    for d in (d1, d2):
        assert hasattr(d, "srs_tools")
        assert hasattr(d, "setup_builder_tool")
        assert hasattr(d, "galvo_diagnostic_tool")
        assert hasattr(d, "spectral_focusing_tool")
        assert hasattr(d, "signal_lockin_tool")
        assert hasattr(d, "bench_alignment_tool")
        assert d.srs_tools.SetupBuilder is d.setup_builder_tool
        assert d.srs_tools.GalvoDiagnostic is d.galvo_diagnostic_tool
        assert d.srs_tools.SpectralFocusing is d.spectral_focusing_tool
        assert d.srs_tools.SignalLockIn is d.signal_lockin_tool
        assert d.srs_tools.BenchAlignment is d.bench_alignment_tool
