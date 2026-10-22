"""Tests for Thorlabs Laser Scanning Microscope (LSM Core) & SRS Microscope.

Validates:
1. Core 4f Telecentric LSM Layout (17 Surfaces):
   - Galvo Scanner (GVS002 X & Y, close-coupled 10 mm separation at front focal plane)
   - Scan Lens (SL50-2P2 Front & Rear, f = 50 mm)
   - Intermediate Image Plane (4f intermediate focus)
   - Tube Lens (TTL200MP Front & Rear, f = 200 mm, 4x pupil magnification)
   - Objective BFP (Pupil conjugate plane)
   - Objective (TL15X-2P, NA 0.70, 15x)
   - Oil Sample (Interaction Plane, transparent transmission)
   - Condenser (ACL25416U-B Aspheric, NA 0.79)
   - Stokes Dump Filter (FESH0850, OD > 6)
   - Collector Lens (AC254-050-B Achromat, pupil descan onto detector)
   - Photodiode Detector (DET10N2)
2. Ray tracing through all 17 surfaces without vignetting.
3. Full SRS Layout with dual lasers (Pump 797 nm, Stokes 1040 nm) and AOM/EOM active RF modulation.
4. Oil Raman Standards (Mineral Oil, Olive Oil, Immersion Oil) and Spectral Focusing reconstruction.
5. Scanning field aberrations (chromatic walk-off, Petzval field curvature defocus, pupil vignetting, FOV mapping).
6. Photodiode detection & Lock-In demodulation (TIA gain, shot noise, thermal noise, laser RIN, SNR).
7. Lossless JSON export and import round-trip for both LSM Core and Full SRS presets.
8. GUI UltrafastAnalysisDock Tab 6 integration (LSM template and SRS template loading).
9. Dual-tree parity between optics_app and v2.
"""

import math
import os
import tempfile
import numpy as np
import pytest

from optics_app.analysis.ultrafast_nonlinear import (
    OilSampleType,
    OilRamanStandard,
    SRSPhotodiodeModel,
    SRSPhotodiodeSignal,
    SRSScanFieldResult,
    PostSampleSRSProbeResult,
    ThorlabsSRSMicroscopeConfig,
    ThorlabsSRSMicroscopeSimulator,
    build_thorlabs_lsm_template,
    build_thorlabs_lsm_system,
    save_thorlabs_lsm_json,
    load_thorlabs_lsm_json,
    build_srs_microscope_template,
    build_srs_microscope_relayed_template,
    build_edmund_optics_srs_template,
    build_olympus_high_na_srs_template,
    build_srs_microscope_system,
    save_srs_microscope_json,
    load_srs_microscope_json,
    OBJECTIVE_CATALOG,
    AOMMaterial,
    AcoustoOpticModulator,
    ElectroOpticModulator,
    EOMMaterial,
    DispersiveGlassMaterial,
    delay_from_stage_displacement_mm,
    stage_displacement_from_delay_ps,
)
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from optics_app.gui.qt_compat import QT_AVAILABLE, QtWidgets


if QT_AVAILABLE and hasattr(QtWidgets, "QApplication"):
    _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


# ==============================================================================
# 1. CORE LASER SCANNING MICROSCOPE (LSM) PRESCRIPTION TESTS
# ==============================================================================

def test_thorlabs_lsm_core_prescription():
    """Verify the pristine 17-surface 4f telecentric Laser Scanning Microscope layout."""
    cfg = ThorlabsSRSMicroscopeConfig()
    stack = build_thorlabs_lsm_template(cfg)

    # Must contain exactly 17 optical surfaces
    assert len(stack.surfaces) == 17

    names = [s.name for s in stack.surfaces]
    assert names[0] == "Laser Entrance Pupil"
    assert names[1] == "Galvo Mirror X (GVS002)"
    assert names[2] == "Galvo Mirror Y (GVS002)"
    assert names[3] == "Scan Lens Front (SL50-2P2)"
    assert names[4] == "Scan Lens Rear (SL50-2P2)"
    assert names[5] == "Intermediate Image Plane"
    assert names[6] == "Tube Lens Front (TTL200MP)"
    assert names[7] == "Tube Lens Rear (TTL200MP)"
    assert names[8] == "Objective BFP (Pupil Conjugate)"
    assert names[9] == "Objective (TL15X-2P NA 0.70)"
    assert names[10] == "Oil Sample (Interaction Plane)"
    assert names[11] == "Condenser Front (ACL25416U-B)"
    assert names[12] == "Condenser Rear (ACL25416U-B)"
    assert names[13] == "Stokes Dump Filter (FESH0850)"
    assert names[14] == "Collector Front (AC254-050-B)"
    assert names[15] == "Collector Rear (AC254-050-B)"
    assert names[16] == "Photodiode (DET10N2)"

    # Verify routes
    routes = stack.optical_routes
    assert routes is not None
    assert routes.excitation is not None
    assert "Scan Beam" in routes.excitation.name


def test_thorlabs_lsm_core_ray_tracing_complete_transmission():
    """Verify that ray bundle traces through all 17 surfaces to the photodiode without vignetting."""
    cfg = ThorlabsSRSMicroscopeConfig()
    stack = build_thorlabs_lsm_template(cfg)
    system = OpticalSystem(stack)

    # Launch collimated ray bundle (on-axis and off-axis across 4 mm pupil)
    rays = [
        Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=0.797),
        Ray(pos=np.array([1.5, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=0.797),
        Ray(pos=np.array([-1.5, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=0.797),
        Ray(pos=np.array([0.0, 1.5, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=0.797),
        Ray(pos=np.array([0.0, -1.5, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=0.797),
    ]

    for ray in rays:
        traced = system.trace_ray(ray)
        assert not traced.vignetted
        assert len(traced.history) == 17
        # Final hit must be on the photodiode (Surface 16)
        assert traced.history[-1]["surface_index"] == 16


# ==============================================================================
# 2. FULL SRS MICROSCOPE OPTICAL TRAIN (DUAL LASERS & MODULATORS)
# ==============================================================================

def test_srs_microscope_full_template_prescriptions():
    """Verify full SRS Microscope layout with dual lasers, modulators, dumps, and 4f LSM column."""
    # EOM mode: 18 LSM column surfaces + Stokes Laser + EOM + PBS + Modulator Dump + Combiner Dump = 23 surfaces
    cfg_eom = ThorlabsSRSMicroscopeConfig(modulator_type="EOM")
    stack_eom = build_srs_microscope_template(cfg_eom)
    assert len(stack_eom.surfaces) == 23

    surface_names = [s.name for s in stack_eom.surfaces]
    assert any("Pump Laser" in n for n in surface_names)
    assert any("Dichroic Combiner" in n for n in surface_names)
    assert any("Galvo Mirror X" in n for n in surface_names)
    assert any("Galvo Mirror Y" in n for n in surface_names)
    assert any("Scan Lens Front" in n for n in surface_names)
    assert any("Intermediate Image" in n for n in surface_names)
    assert any("Tube Lens Front" in n for n in surface_names)
    assert any("Objective BFP" in n for n in surface_names)
    assert any("Objective" in n for n in surface_names)
    assert any("Oil Sample" in n for n in surface_names)
    assert any("Condenser Front" in n for n in surface_names)
    assert any("Stokes Dump Filter" in n for n in surface_names)
    assert any("Collector Front" in n for n in surface_names)
    assert any("Photodiode" in n for n in surface_names)
    assert any("Stokes Laser" in n for n in surface_names)
    assert any("Stokes EOM" in n for n in surface_names)
    assert any("Stokes PBS" in n for n in surface_names)
    assert any("Polarization Dump" in n for n in surface_names)
    assert any("Combiner Leak Dump" in n for n in surface_names)

    # Routes check
    routes = stack_eom.optical_routes
    assert routes is not None
    assert routes.excitation is not None
    assert "Pump Beam" in routes.excitation.name
    assert routes.dump is not None
    assert "Modulator Dump" in routes.dump.name
    assert "stokes" in routes.extra_routes
    assert "srs" in routes.extra_routes
    assert "combiner_dump" in routes.extra_routes


def test_srs_microscope_aom_vs_eom_template():
    """Verify surface count differences between AOM (22 surfaces) and EOM (23 surfaces)."""
    cfg_eom = ThorlabsSRSMicroscopeConfig(modulator_type="EOM")
    stack_eom = build_srs_microscope_template(cfg_eom)
    assert len(stack_eom.surfaces) == 23

    cfg_aom = ThorlabsSRSMicroscopeConfig(modulator_type="AOM")
    stack_aom = build_srs_microscope_template(cfg_aom)
    # AOM deflects directly without separate PBS, yielding 22 surfaces
    assert len(stack_aom.surfaces) == 22
    names_aom = [s.name for s in stack_aom.surfaces]
    assert any("AOM" in n for n in names_aom)
    assert any("AOM 0th-Order Dump" in n for n in names_aom)
    assert not any("Stokes PBS" in n for n in names_aom)


# ==============================================================================
# 3. ACTIVE RF MODULATION TESTS (AOM & EOM)
# ==============================================================================

def test_eom_modulation_response():
    """Verify EOM Pockels cell phase retardation and lock-in modulation depth."""
    cfg = ThorlabsSRSMicroscopeConfig(
        modulator_type="EOM",
        modulator_v_pi_v=180.0,
        modulator_applied_v=180.0,
    )
    scan = ThorlabsSRSMicroscopeSimulator.simulate_scanning_point(cfg, 0.0, 0.0)
    sig = ThorlabsSRSMicroscopeSimulator.process_photodiode_signal(cfg, srl_fraction=1e-4, scan_result=scan)

    assert math.isclose(sig.details["modulation_factor"], 1.0, rel_tol=1e-4)
    assert sig.srl_modulation_depth > 0.0

    cfg_half = ThorlabsSRSMicroscopeConfig(
        modulator_type="EOM",
        modulator_v_pi_v=180.0,
        modulator_applied_v=90.0,
    )
    sig_half = ThorlabsSRSMicroscopeSimulator.process_photodiode_signal(cfg_half, srl_fraction=1e-4, scan_result=scan)
    assert math.isclose(sig_half.details["modulation_factor"], 0.5, rel_tol=1e-4)


def test_aom_modulation_response():
    """Verify TeO2 AOM Bragg diffraction efficiency and beam transmission."""
    cfg = ThorlabsSRSMicroscopeConfig(
        modulator_type="AOM",
        aom_rf_power_w=1.0,
        aom_saturation_power_w=1.0,
    )
    scan = ThorlabsSRSMicroscopeSimulator.simulate_scanning_point(cfg, 0.0, 0.0)
    sig = ThorlabsSRSMicroscopeSimulator.process_photodiode_signal(cfg, srl_fraction=1e-4, scan_result=scan)

    assert 0.70 <= sig.details["modulation_factor"] <= 0.85
    assert math.isclose(sig.details["modulation_factor"], 0.7715, rel_tol=1e-2)
    assert sig.srl_modulation_depth > 0.0


# ==============================================================================
# 4. OIL RAMAN STANDARDS & SPECTRAL RECONSTRUCTION TESTS
# ==============================================================================

def test_oil_raman_standards_cross_sections():
    """Verify analytical Lorentzian vibrational profiles for Mineral Oil, Olive Oil, and Immersion Oil."""
    mineral = OilRamanStandard.create(OilSampleType.MINERAL_OIL)
    assert mineral.refractive_index == 1.470
    peaks_mineral = [p[0] for p in mineral.vibrational_peaks]
    assert 2850.0 in peaks_mineral
    assert 2935.0 in peaks_mineral
    cs_2850 = mineral.evaluate_raman_cross_section(2850.0)
    cs_off = mineral.evaluate_raman_cross_section(2700.0)
    assert cs_2850 > 1.0 > cs_off

    olive = OilRamanStandard.create(OilSampleType.OLIVE_OIL)
    peaks_olive = [p[0] for p in olive.vibrational_peaks]
    assert 3005.0 in peaks_olive
    cs_3005 = olive.evaluate_raman_cross_section(3005.0)
    cs_mineral_3005 = mineral.evaluate_raman_cross_section(3005.0)
    assert cs_3005 > cs_mineral_3005 * 3.0

    imm = OilRamanStandard.create(OilSampleType.IMMERSION_OIL)
    peaks_imm = [p[0] for p in imm.vibrational_peaks]
    assert 3050.0 in peaks_imm
    assert imm.evaluate_raman_cross_section(3050.0) > 0.3


def test_delay_stage_displacement_conversions():
    """Verify double-pass delay stage displacement and optical delay math."""
    delay_ps = delay_from_stage_displacement_mm(1.0, passes=2)
    assert math.isclose(delay_ps, 6.67128, rel_tol=1e-3)

    disp_mm = stage_displacement_from_delay_ps(delay_ps, passes=2)
    assert math.isclose(disp_mm, 1.0, rel_tol=1e-6)


def test_spectral_focusing_oil_spectrum_simulation():
    """Verify Spectral Focusing SRS spectrum simulation and chirp matching."""
    cfg = ThorlabsSRSMicroscopeConfig(
        sample_type=OilSampleType.MINERAL_OIL,
        optical_delay_scan_ps=3.0,
        delay_samples=61,
    )
    res = ThorlabsSRSMicroscopeSimulator.simulate_oil_spectrum(cfg, angle_x_deg=0.0, angle_y_deg=0.0)

    assert res["sample_name"] == "Mineral Oil (Paraffin)"
    assert len(res["wavenumbers_cm1"]) == 61
    assert len(res["on_axis_lockin_voltage_v"]) == 61
    assert res["chirp_matching_ratio"] > 0.0

    peak_idx = int(np.argmax(res["on_axis_lockin_voltage_v"]))
    peak_wavenumber = res["wavenumbers_cm1"][peak_idx]
    assert 2800.0 <= peak_wavenumber <= 3000.0


# ==============================================================================
# 5. SCANNING ABERRATIONS & FIELD EFFECTS TESTS
# ==============================================================================

def test_scanning_aberrations_on_axis_vs_off_axis():
    """Verify on-axis perfection vs off-axis chromatic walk-off, Petzval defocus and vignetting."""
    cfg = ThorlabsSRSMicroscopeConfig(
        galvo_max_mechanical_angle_deg=5.0,
        lateral_chromatic_coeff=0.005,
        petzval_radius_mm=25.0,
        pupil_vignetting_coeff=0.30,
    )

    on_axis = ThorlabsSRSMicroscopeSimulator.simulate_scanning_point(cfg, 0.0, 0.0)
    assert on_axis.sample_x_um == 0.0
    assert on_axis.sample_y_um == 0.0
    assert on_axis.field_radius_um == 0.0
    assert on_axis.chromatic_walk_off_um == 0.0
    assert on_axis.field_curvature_defocus_um == 0.0
    assert on_axis.lateral_overlap_factor == 1.0
    assert on_axis.pupil_vignetting_factor == 1.0
    assert on_axis.total_scan_efficiency > 0.85

    off_axis = ThorlabsSRSMicroscopeSimulator.simulate_scanning_point(cfg, 4.0, 3.0)
    assert off_axis.field_radius_um > 500.0
    assert off_axis.chromatic_walk_off_um > 0.0
    assert off_axis.field_curvature_defocus_um > 0.0
    assert off_axis.defocused_pump_waist_um > off_axis.pump_waist_um
    assert off_axis.lateral_overlap_factor < 1.0
    assert off_axis.pupil_vignetting_factor < 1.0
    assert off_axis.total_scan_efficiency < on_axis.total_scan_efficiency


def test_2d_scan_field_simulation_patterns():
    """Verify 2D raster scan field simulation across patterns (Uniform, Droplets, Grid)."""
    cfg = ThorlabsSRSMicroscopeConfig(
        sample_pattern="Oil Droplets",
        galvo_max_mechanical_angle_deg=3.0,
    )
    field_res = ThorlabsSRSMicroscopeSimulator.simulate_2d_scan_field(cfg, grid_points=11)

    assert field_res["scan_efficiency_map"].shape == (11, 11)
    assert field_res["sample_concentration_map"].shape == (11, 11)
    assert field_res["srs_signal_map_v"].shape == (11, 11)
    assert field_res["snr_map_db"].shape == (11, 11)

    center_conc = field_res["sample_concentration_map"][5, 5]
    corner_conc = field_res["sample_concentration_map"][0, 0]
    assert center_conc > corner_conc


# ==============================================================================
# 6. PHOTODIODE DETECTOR & LOCK-IN DEMODULATION TESTS
# ==============================================================================

def test_photodiode_responsivity_tia_gain_and_noise():
    """Verify Thorlabs DET10N2 photocurrent, TIA voltage, shot noise, thermal noise and RIN."""
    cfg = ThorlabsSRSMicroscopeConfig(
        pump_laser_power_mw=25.0,
        objective_transmission=0.90,
        condenser_transmission=0.90,
        stokes_dump_transmission_at_797=0.95,
    )
    scan = ThorlabsSRSMicroscopeSimulator.simulate_scanning_point(cfg, 0.0, 0.0)
    srl_val = 2.0e-5
    sig = ThorlabsSRSMicroscopeSimulator.process_photodiode_signal(cfg, srl_val, scan)

    expected_p_opt = 0.025 * (0.90 * 0.90 * 0.95)
    assert math.isclose(sig.dc_optical_power_w, expected_p_opt, rel_tol=1e-4)

    expected_i_dc = cfg.photodiode.responsivity_a_per_w * expected_p_opt + cfg.photodiode.dark_current_a
    assert math.isclose(sig.dc_photocurrent_a, expected_i_dc, rel_tol=1e-4)
    expected_v_dc = expected_i_dc * cfg.photodiode.transimpedance_gain_v_per_a
    assert math.isclose(sig.dc_output_voltage_v, expected_v_dc, rel_tol=1e-4)

    assert sig.shot_noise_current_a > 0.0
    assert sig.thermal_noise_current_a > 0.0
    assert sig.rin_noise_current_a > 0.0
    assert sig.total_noise_current_a > sig.shot_noise_current_a
    assert sig.lockin_magnitude_r_v > 0.0
    assert sig.snr_db > 0.0
    assert not sig.is_saturated


def test_photodiode_saturation_detection():
    """Verify photodiode saturation flag when optical power exceeds threshold."""
    cfg = ThorlabsSRSMicroscopeConfig(
        pump_laser_power_mw=100.0,
        photodiode=SRSPhotodiodeModel(saturation_power_w=0.010),
    )
    scan = ThorlabsSRSMicroscopeSimulator.simulate_scanning_point(cfg, 0.0, 0.0)
    sig = ThorlabsSRSMicroscopeSimulator.process_photodiode_signal(cfg, 1e-4, scan)
    assert sig.is_saturated


# ==============================================================================
# 7. JSON SERIALIZATION & ROUND-TRIP PRESET TESTS
# ==============================================================================

def test_thorlabs_lsm_and_srs_json_roundtrip():
    """Verify lossless export and import of both LSM Core (17 surfaces) and Full SRS (23 surfaces)."""
    # 1. Test LSM Core (17 surfaces)
    cfg_lsm = ThorlabsSRSMicroscopeConfig()
    sys_lsm = build_thorlabs_lsm_system(cfg_lsm)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        tmp_lsm_path = tf.name

    try:
        save_thorlabs_lsm_json(tmp_lsm_path, sys_lsm)
        assert os.path.exists(tmp_lsm_path)
        sys_lsm_loaded = load_thorlabs_lsm_json(tmp_lsm_path)
        assert sys_lsm_loaded is not None
        assert len(sys_lsm_loaded.surface_stack.surfaces) == 17
    finally:
        if os.path.exists(tmp_lsm_path):
            os.remove(tmp_lsm_path)

    # 2. Test Full SRS Layout (23 surfaces EOM, 22 surfaces AOM)
    cfg_srs_eom = ThorlabsSRSMicroscopeConfig(modulator_type="EOM")
    sys_srs_eom = build_srs_microscope_system(cfg_srs_eom)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        tmp_srs_path = tf.name

    try:
        save_srs_microscope_json(tmp_srs_path, sys_srs_eom)
        assert os.path.exists(tmp_srs_path)
        sys_srs_loaded = load_srs_microscope_json(tmp_srs_path)
        assert sys_srs_loaded is not None
        assert len(sys_srs_loaded.surface_stack.surfaces) == 23
    finally:
        if os.path.exists(tmp_srs_path):
            os.remove(tmp_srs_path)


def test_shipped_preset_files_lsm_and_srs():
    """Verify shipped thorlabs_lsm_microscope.json and thorlabs_srs_microscope.json files."""
    # Check LSM preset
    sys_lsm = load_thorlabs_lsm_json("thorlabs_lsm_microscope.json")
    assert len(sys_lsm.surface_stack.surfaces) == 17
    assert sys_lsm.surface_stack.surfaces[3].name == "Scan Lens Front (SL50-2P2)"
    assert sys_lsm.surface_stack.surfaces[6].name == "Tube Lens Front (TTL200MP)"
    assert sys_lsm.surface_stack.surfaces[9].name == "Objective (TL15X-2P NA 0.70)"

    # Check SRS preset
    sys_srs = load_srs_microscope_json("thorlabs_srs_microscope.json")
    assert len(sys_srs.surface_stack.surfaces) == 23


# ==============================================================================
# 8. GUI DOCK TAB 6 INTEGRATION TESTS
# ==============================================================================

@pytest.mark.skipif(not QT_AVAILABLE, reason="Qt bindings not available")
def test_ultrafast_dock_srs_microscope_tab():
    """Verify UltrafastAnalysisDock Tab 6 initialization, template loading & simulation."""
    from optics_app.gui.ultrafast_dock import UltrafastAnalysisDock

    class MockModel:
        def __init__(self):
            self.surface_stack = None

    class MockMainWindow:
        def __init__(self):
            self.model = MockModel()
            self.undo_manager = None
            self.lde_table = None
            self.visualizer_3d = None
        def statusBar(self):
            return None

    mock_mw = MockMainWindow()
    dock = UltrafastAnalysisDock(main_window=mock_mw)

    assert dock.tabs.count() >= 6
    assert "Thorlabs SRS Microscope" in dock.tabs.tabText(5)

    # Test loading Core LSM template (17 surfaces)
    dock.load_thorlabs_lsm_template_into_project()
    assert mock_mw.model.surface_stack is not None
    assert len(mock_mw.model.surface_stack.surfaces) == 17

    # Test loading Full SRS template (23 surfaces)
    dock.load_srs_microscope_template_into_project()
    assert len(mock_mw.model.surface_stack.surfaces) == 23

    # Test running simulation
    dock.run_srs_microscope_simulation()
    assert "On-Axis" in dock.lbl_srs_summary.text()
    assert dock.tbl_srs_spectrum.rowCount() > 0


# ==============================================================================
# 9. DUAL-TREE PARITY TESTS (optics_app vs v2)
# ==============================================================================

def test_dual_tree_parity_v2_import():
    """Verify identical behavior between optics_app and v2 modules."""
    import v2.analysis.ultrafast_nonlinear as v2_srs

    cfg1 = ThorlabsSRSMicroscopeConfig(sample_type=OilSampleType.OLIVE_OIL)
    cfg2 = v2_srs.ThorlabsSRSMicroscopeConfig(sample_type=v2_srs.OilSampleType.OLIVE_OIL)

    stack1 = build_thorlabs_lsm_template(cfg1)
    stack2 = v2_srs.build_thorlabs_lsm_template(cfg2)
    assert len(stack1.surfaces) == len(stack2.surfaces) == 17

    res1 = ThorlabsSRSMicroscopeSimulator.simulate_oil_spectrum(cfg1, 2.0, 1.0)
    res2 = v2_srs.ThorlabsSRSMicroscopeSimulator.simulate_oil_spectrum(cfg2, 2.0, 1.0)

    assert np.allclose(res1["wavenumbers_cm1"], res2["wavenumbers_cm1"])
    assert np.allclose(res1["on_axis_lockin_voltage_v"], res2["on_axis_lockin_voltage_v"])
    assert np.isclose(res1["peak_sig_off_axis"].lockin_magnitude_r_v, res2["peak_sig_off_axis"].lockin_magnitude_r_v)


# ==============================================================================
# 10. POST-SAMPLE SRS PROBE & 2D RAY PROPAGATION TESTS
# ==============================================================================

def test_post_sample_srs_probe_spectral_focusing():
    """Verify virtual probe immediately after sample plane observes generated intensity at specific delay."""
    cfg = ThorlabsSRSMicroscopeConfig(sample_type=OilSampleType.MINERAL_OIL)

    # 1. Probe at zero delay
    probe0 = ThorlabsSRSMicroscopeSimulator.probe_post_sample_intensity(cfg, delay_ps=0.0)
    assert isinstance(probe0, PostSampleSRSProbeResult)
    assert np.isclose(probe0.time_delay_ps, 0.0)
    assert np.isclose(probe0.stage_displacement_mm, 0.0)
    assert probe0.probed_wavenumber_cm1 > 2800.0
    assert probe0.fractional_srl > 0.0
    assert probe0.transmitted_pump_power_mw > 0.0
    assert probe0.generated_signal_power_uw > 0.0
    assert probe0.photodiode_signal.lockin_magnitude_r_v > 0.0
    assert len(probe0.subpicosecond_time_fs) == 401
    assert len(probe0.pulse_train_time_ns) == 2001


def test_post_sample_probe_on_vs_off_resonance():
    """Verify probe distinguishes on-resonance Raman peaks from off-resonance baseline."""
    cfg = ThorlabsSRSMicroscopeConfig(sample_type=OilSampleType.MINERAL_OIL)

    # Base wavenumber is ~2928.3 cm^-1 (near CH3 peak 2935 cm^-1)
    probe_on = ThorlabsSRSMicroscopeSimulator.probe_post_sample_intensity(cfg, delay_ps=0.05)
    # Off-resonance by tuning delay to shift wavenumber away from C-H region
    probe_off = ThorlabsSRSMicroscopeSimulator.probe_post_sample_intensity(cfg, delay_ps=-2.5)

    assert probe_on.is_on_resonance
    assert probe_on.fractional_srl > probe_off.fractional_srl * 2.0
    assert probe_on.photodiode_signal.lockin_magnitude_r_v > probe_off.photodiode_signal.lockin_magnitude_r_v * 2.0


def test_srs_microscope_sequential_ray_tracing_unvignetted():
    """Verify 2D and 3D ray tracing traverses all sequential surfaces in the SRS system unvignetted."""
    cfg = ThorlabsSRSMicroscopeConfig()
    stack = build_srs_microscope_template(cfg)
    system = OpticalSystem(stack)

    # Trace on-axis pump ray (797 nm)
    ray_on_axis = Ray(pos=[0.0, 0.0, 0.0], dir=[0.0, 0.0, 1.0], wavelength=0.797)
    traced_on = system.trace_ray(ray_on_axis)

    assert not traced_on.vignetted
    assert len(traced_on.history) == 18
    assert traced_on.history[-1]["surface_index"] == 17
    assert stack.surfaces[17].name == "Photodiode (DET10N2)"

    # Trace off-axis marginal ray
    ray_marginal = Ray(pos=[0.0, 1.0, 0.0], dir=[0.0, 0.0, 1.0], wavelength=0.797)
    traced_mar = system.trace_ray(ray_marginal)
    assert not traced_mar.vignetted
    assert len(traced_mar.history) == 18


@pytest.mark.skipif(not QT_AVAILABLE, reason="Qt bindings not available")
def test_ultrafast_dock_post_sample_probe_gui():
    """Verify Post-Sample Probe controls and 2D ray tracing invocation in Ultrafast Dock."""
    from optics_app.gui.ultrafast_dock import UltrafastAnalysisDock

    class MockModel:
        def __init__(self):
            self.surface_stack = None

    class MockDualView:
        def __init__(self):
            self.layout_updated = False
        def update_layout(self, surface_stack=None, rays=None):
            self.layout_updated = True

    class MockViewerTabs:
        def __init__(self):
            self.current_widget = None
        def setCurrentWidget(self, widget):
            self.current_widget = widget

    class MockMainWindow:
        def __init__(self):
            self.model = MockModel()
            self.undo_manager = None
            self.lde_table = None
            self.visualizer_3d = None
            self.dual_view = MockDualView()
            self.viewer_tabs = MockViewerTabs()
            self.trace_called = False
        def on_trace_rays(self):
            self.trace_called = True
        def statusBar(self):
            return None

    mock_mw = MockMainWindow()
    dock = UltrafastAnalysisDock(main_window=mock_mw)

    # Check probe controls exist
    assert hasattr(dock, "spin_srs_probe_delay_ps")
    assert hasattr(dock, "spin_srs_probe_stage_mm")
    assert hasattr(dock, "lbl_srs_probe_wavenumber")
    assert hasattr(dock, "lbl_srs_probe_intensity_res")

    # Set delay and run probe
    dock.spin_srs_probe_delay_ps.setValue(0.5)
    dock.run_post_sample_probe()
    assert "Post-Sample SRL" in dock.lbl_srs_probe_intensity_res.text()
    assert "Lock-In" in dock.lbl_srs_probe_intensity_res.text()

    # Test quick-jump button
    dock._set_probe_target_cm1(2850.0)
    lbl_txt = dock.lbl_srs_probe_wavenumber.text()
    assert "2849" in lbl_txt or "2850" in lbl_txt or "ON-Resonance" in lbl_txt

    # Loading templates must call on_trace_rays and switch to dual_view
    mock_mw.trace_called = False
    dock.load_thorlabs_lsm_template_into_project()
    assert mock_mw.trace_called
    assert mock_mw.viewer_tabs.current_widget is mock_mw.dual_view

    mock_mw.trace_called = False
    dock.load_srs_microscope_template_into_project()
    assert mock_mw.trace_called
    assert mock_mw.viewer_tabs.current_widget is mock_mw.dual_view


def test_dual_tree_parity_post_sample_probe():
    """Verify identical Post-Sample Probe calculations between optics_app and v2."""
    import v2.analysis.ultrafast_nonlinear as v2_srs

    cfg1 = ThorlabsSRSMicroscopeConfig(sample_type=OilSampleType.MINERAL_OIL)
    cfg2 = v2_srs.ThorlabsSRSMicroscopeConfig(sample_type=v2_srs.OilSampleType.MINERAL_OIL)

    probe1 = ThorlabsSRSMicroscopeSimulator.probe_post_sample_intensity(cfg1, delay_ps=0.3, angle_x_deg=1.0)
    probe2 = v2_srs.ThorlabsSRSMicroscopeSimulator.probe_post_sample_intensity(cfg2, delay_ps=0.3, angle_x_deg=1.0)

    assert np.isclose(probe1.probed_wavenumber_cm1, probe2.probed_wavenumber_cm1)
    assert np.isclose(probe1.fractional_srl, probe2.fractional_srl)
    assert np.isclose(probe1.generated_signal_power_uw, probe2.generated_signal_power_uw)
    assert np.isclose(probe1.photodiode_signal.lockin_magnitude_r_v, probe2.photodiode_signal.lockin_magnitude_r_v)
    assert np.isclose(probe1.photodiode_signal.snr_db, probe2.photodiode_signal.snr_db)


def test_thorlabs_lsm_and_srs_json_presets_structure_and_components():
    """Verify JSON preset files exist in all catalogs and contain complete Thorlabs component metadata."""
    import json
    import os

    locations = [
        ("thorlabs_lsm_microscope.json", "thorlabs_srs_microscope.json"),
        ("optics_app/catalogs/presets/thorlabs_lsm_microscope.json", "optics_app/catalogs/presets/thorlabs_srs_microscope.json"),
        ("v2/catalogs/presets/thorlabs_lsm_microscope.json", "v2/catalogs/presets/thorlabs_srs_microscope.json"),
    ]

    expected_lsm_parts = {"GVS002", "SL50-2P2", "TTL200MP", "TL15X-2P", "ACL25416U-B", "FESH0850", "AC254-050-B", "DET10N2"}
    expected_srs_parts = expected_lsm_parts | {"DMLP900", "BT610", "EO-AM-NR-C1", "PBS252"}

    for lsm_path, srs_path in locations:
        assert os.path.exists(lsm_path), f"Missing {lsm_path}"
        assert os.path.exists(srs_path), f"Missing {srs_path}"

        with open(lsm_path, "r", encoding="utf-8") as f:
            d_lsm = json.load(f)
        assert len(d_lsm["surfaces"]) == 17
        assert len(d_lsm["components"]) == 12
        lsm_part_numbers = {c.get("metadata", {}).get("part_number") for c in d_lsm["components"]}
        for exp in expected_lsm_parts:
            assert any(exp in str(pn) for pn in lsm_part_numbers), f"LSM missing {exp}"

        with open(srs_path, "r", encoding="utf-8") as f:
            d_srs = json.load(f)
        assert len(d_srs["surfaces"]) >= 21
        assert len(d_srs["components"]) >= 16
        srs_part_numbers = {c.get("metadata", {}).get("part_number") for c in d_srs["components"]}
        for exp in expected_srs_parts:
            assert any(exp in str(pn) for pn in srs_part_numbers), f"SRS missing {exp}"


def test_thorlabs_lsm_and_srs_2d_ray_tracing():
    """Verify that 2D sequential rays propagate through all Thorlabs elements without vignetting."""
    from optics_app.engine.lde import SurfaceStack
    from optics_app.engine.tracer import OpticalSystem
    from optics_app.engine.ray import Ray

    # 1. LSM layout
    res_lsm = SurfaceStack.load_from_json("thorlabs_lsm_microscope.json")
    stack_lsm = res_lsm["stack"]
    sys_lsm = OpticalSystem(stack_lsm)
    ray_lsm = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=0.797)
    tr_lsm = sys_lsm.trace_ray(ray_lsm)
    assert len(tr_lsm.history) == 17
    assert not tr_lsm.vignetted

    # 2. SRS layout
    res_srs = SurfaceStack.load_from_json("thorlabs_srs_microscope.json")
    stack_srs = res_srs["stack"]
    sys_srs = OpticalSystem(stack_srs)
    ray_srs = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=0.797)
    tr_srs = sys_srs.trace_ray(ray_srs)
    assert len(tr_srs.history) >= 17
    assert not tr_srs.vignetted


@pytest.mark.skipif(not QT_AVAILABLE, reason="Qt bindings not available")
def test_main_window_loads_preset_and_updates_components_and_ray_tracing():
    """Verify MainWindow._load_project loads Thorlabs SRS preset, populates components, and traces rays."""
    from optics_app.gui.main_window import MainWindow

    win = MainWindow()
    loaded = win.on_load_project("thorlabs_srs_microscope.json")
    assert loaded

    stack = win.model.get_surface_stack()
    assert len(stack.surfaces) >= 21
    assert len(stack.components) >= 16

    # Verify ray tracing produced rays
    assert len(win._last_traced_rays) > 0

    # Verify component editor updated with all surfaces
    assert win.component_editor_dock.surface_combo.count() == len(stack.surfaces)


@pytest.mark.skipif(not QT_AVAILABLE, reason="Qt bindings not available")
def test_main_window_ultrafast_dock_visibility_and_actions():
    """Verify MainWindow provides View->Docks, View direct, Window, and Toolbar access for Ultrafast Dock."""
    from optics_app.gui.main_window import MainWindow

    win = MainWindow()
    assert hasattr(win, 'ultrafast_dock')
    assert hasattr(win, 'act_open_ultrafast')

    # 1. Check Toolbar button
    assert win.act_open_ultrafast.text() == "🔬 Ultrafast & SRS"

    # 2. Check _show_ultrafast_dock
    win._show_ultrafast_dock()
    assert not win.ultrafast_dock.isHidden()
    assert win.ultrafast_dock.tabs.currentIndex() == 5  # SRS tab

    # 3. Check menus exist and contain Ultrafast actions
    menubar = win.menuBar()
    actions = menubar.actions()
    action_texts = [a.text() for a in actions]
    assert "&View" in action_texts
    assert "&Window" in action_texts

    # Find View menu
    view_action = next(a for a in actions if a.text() == "&View")
    view_menu = view_action.menu()
    view_texts = [a.text() for a in view_menu.actions()]
    assert any("Ultrafast" in t for t in view_texts)
    assert any("Docks" in t for t in view_texts)

    # Find Window menu
    win_action = next(a for a in actions if a.text() == "&Window")
    window_menu = win_action.menu()
    win_texts = [a.text() for a in window_menu.actions()]
    assert any("Ultrafast" in t for t in win_texts)


# ==============================================================================
# 10. BACK PUPIL CONJUGATION & HIGH-NA OPTICS TESTS
# ==============================================================================

def test_back_pupil_conjugation_and_walk_off_physics():
    """Verify BFP pupil conjugation, magnification, fill factor, and walk-off difference between Close-Coupled and 4f-Relayed."""
    # 1. Close-Coupled Galvo (Thorlabs GVS002)
    cfg_close = ThorlabsSRSMicroscopeConfig(
        scanning_architecture="Close-Coupled (Thorlabs GVS002)",
        scan_lens_efl_mm=50.0,
        tube_lens_efl_mm=200.0,
        input_beam_diameter_mm=4.5,
        inter_galvo_separation_mm=9.5,
    )
    # On-axis
    res_close_0 = ThorlabsSRSMicroscopeSimulator.simulate_scanning_point(cfg_close, 0.0, 0.0)
    assert res_close_0.pupil_magnification == 4.0
    assert res_close_0.back_pupil_beam_diameter_mm == 18.0
    assert abs(res_close_0.objective_pupil_diameter_mm - 18.667) < 0.01
    assert abs(res_close_0.pupil_fill_factor - (18.0 / 18.667)) < 0.01
    assert res_close_0.back_pupil_walk_off_total_um == 0.0
    assert not res_close_0.is_strictly_conjugated

    # Off-axis at theta_x = 2.5 deg (mechanical)
    res_close_off = ThorlabsSRSMicroscopeSimulator.simulate_scanning_point(cfg_close, 2.5, 0.0)
    # Walk-off = M_pupil * dz * tan(2 * theta_x) = 4.0 * 9.5 mm * tan(5 deg) = 38 * 0.08749 = 3.32 mm = 3324 um
    assert res_close_off.back_pupil_walk_off_total_um > 2000.0
    assert res_close_off.telecentricity_error_mrad > 100.0
    assert res_close_off.pupil_vignetting_factor < res_close_0.pupil_vignetting_factor

    # 2. 4f-Relayed Galvo (Zero Walk-Off)
    cfg_relayed = ThorlabsSRSMicroscopeConfig(
        scanning_architecture="4f-Relayed (Zero Walk-Off)",
        scan_lens_efl_mm=50.0,
        tube_lens_efl_mm=200.0,
        input_beam_diameter_mm=4.5,
    )
    res_relayed_off = ThorlabsSRSMicroscopeSimulator.simulate_scanning_point(cfg_relayed, 2.5, 0.0)
    assert res_relayed_off.pupil_magnification == 4.0
    assert res_relayed_off.back_pupil_walk_off_x_um == 0.0
    assert res_relayed_off.back_pupil_walk_off_y_um == 0.0
    assert res_relayed_off.back_pupil_walk_off_total_um == 0.0
    assert res_relayed_off.telecentricity_error_mrad == 0.0
    assert res_relayed_off.is_strictly_conjugated
    # Relayed scanning must have higher scan efficiency than close-coupled off-axis
    assert res_relayed_off.total_scan_efficiency > res_close_off.total_scan_efficiency


def test_4f_relayed_template_prescription_and_ray_tracing():
    """Verify 4f-Relayed SRS template with 1:1 inter-galvo relay lenses and complete ray tracing."""
    cfg = ThorlabsSRSMicroscopeConfig(scanning_architecture="4f-Relayed (Zero Walk-Off)")
    stack = build_srs_microscope_relayed_template(cfg)

    names = [s.name for s in stack.surfaces]
    assert "Galvo Relay Lens 1 Front" in names
    assert "Relay Pupil Conjugate" in names
    assert "Galvo Relay Lens 2 Front" in names
    assert "Scan Lens Front (SL50-2P2)" in names
    assert "Tube Lens Front (TTL200MP)" in names
    assert "Photodiode (DET10N2)" in names

    system = OpticalSystem(stack)
    ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=0.797)
    traced = system.trace_ray(ray)
    assert not traced.vignetted
    assert len(traced.history) >= 18


def test_olympus_high_numerical_aperture_srs_template():
    """Verify Olympus high numerical aperture objectives (XLPLN25XWMP2 NA 1.05 W, XLUMPLFLN20XW NA 1.00 W, UPLSAPO60XW NA 1.20 W)."""
    # 1. Check catalog entries
    assert "Olympus XLPLN25XWMP2 (25x NA 1.05 W Multiphoton)" in OBJECTIVE_CATALOG
    assert "Olympus XLUMPLFLN20XW (20x NA 1.00 W)" in OBJECTIVE_CATALOG
    assert "Olympus UPLSAPO60XW (60x NA 1.20 W High-NA)" in OBJECTIVE_CATALOG

    info_25x = OBJECTIVE_CATALOG["Olympus XLPLN25XWMP2 (25x NA 1.05 W Multiphoton)"]
    assert info_25x["numerical_aperture"] == 1.05
    assert info_25x["working_distance_mm"] == 2.0
    assert "Water" in info_25x["immersion"]

    # 2. Build Olympus High-NA SRS template
    cfg = ThorlabsSRSMicroscopeConfig(
        objective_model="Olympus XLPLN25XWMP2 (25x NA 1.05 W Multiphoton)",
    )
    stack = build_olympus_high_na_srs_template(cfg)
    assert len(stack.surfaces) == 23

    names = [s.name for s in stack.surfaces]
    assert any("XLPLN25XWMP2" in n for n in names)
    assert "Scan Lens Front (SL50-2P2)" in names
    assert "Tube Lens Front (TTL200MP)" in names
    assert "Photodiode (DET10N2)" in names

    # 3. Ray tracing
    system = OpticalSystem(stack)
    ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=0.797)
    traced = system.trace_ray(ray)
    assert not traced.vignetted
    assert len(traced.history) == 18

    # 4. Focal waist with Olympus NA 1.05 vs NA 0.70
    res_olymp = ThorlabsSRSMicroscopeSimulator.simulate_scanning_point(cfg, 0.0, 0.0)
    cfg_standard = ThorlabsSRSMicroscopeConfig()
    res_standard = ThorlabsSRSMicroscopeSimulator.simulate_scanning_point(cfg_standard, 0.0, 0.0)

    # w0_pump = 0.61 * 0.797 / NA -> Olympus waist must be substantially smaller
    assert res_olymp.pump_waist_um < res_standard.pump_waist_um
    assert abs(res_olymp.pump_waist_um - (0.61 * 0.797 / 1.05)) < 0.01


def test_edmund_optics_telecentric_scan_system_template():
    """Verify Edmund Optics #64-422 telecentric scan lens and #58-520 tube lens template and ray tracing."""
    cfg = ThorlabsSRSMicroscopeConfig(scanning_architecture="Edmund Optics Telecentric")
    stack = build_edmund_optics_srs_template(cfg)
    assert len(stack.surfaces) == 23

    names = [s.name for s in stack.surfaces]
    assert "Scan Lens Front (EO #64-422)" in names
    assert "Scan Lens Rear (EO #64-422)" in names
    assert "Tube Lens Front (EO #58-520)" in names
    assert "Tube Lens Rear (EO #58-520)" in names
    assert any("Nikon" in n for n in names)

    system = OpticalSystem(stack)
    ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=0.797)
    traced = system.trace_ray(ray)
    assert not traced.vignetted
    assert len(traced.history) == 18


@pytest.mark.skipif(not QT_AVAILABLE, reason="Qt bindings not available")
def test_ultrafast_dock_tab6_architecture_and_objective_controls():
    """Verify UltrafastAnalysisDock Tab 6 architecture selection, Olympus loader, and BFP conjugation readout."""
    from optics_app.gui.main_window import MainWindow

    win = MainWindow()
    dock = win.ultrafast_dock
    assert hasattr(dock, "combo_srs_scan_arch")
    assert hasattr(dock, "combo_srs_objective")
    assert hasattr(dock, "lbl_srs_pupil_metrics")
    assert hasattr(dock, "btn_load_relayed_srs")
    assert hasattr(dock, "btn_load_olympus_srs")
    assert hasattr(dock, "btn_load_edmund_srs")

    # Select Olympus High-NA objective
    dock.combo_srs_objective.setCurrentIndex(1)  # Olympus XLPLN25XWMP2 NA 1.05 W
    assert "1.05" in dock.lbl_srs_pupil_metrics.text() or "Olympus" in dock.combo_srs_objective.currentText()

    # Load Olympus template into MainWindow
    dock.load_olympus_high_na_template_into_project()
    stack = win.model.get_surface_stack()
    names = [s.name for s in stack.surfaces]
    assert any("XLPLN25XWMP2" in n for n in names)
    assert len(win._last_traced_rays) > 0

    # Load 4f-Relayed template into MainWindow
    dock.load_relayed_srs_template_into_project()
    stack_rel = win.model.get_surface_stack()
    names_rel = [s.name for s in stack_rel.surfaces]
    assert "Galvo Relay Lens 1 Front" in names_rel
    assert len(win._last_traced_rays) > 0

    # Load Folded Real-World SRS template into MainWindow
    assert hasattr(dock, "btn_load_folded_srs")
    assert hasattr(dock, "btn_load_real_srs")
    assert hasattr(dock, "chk_srs_galvo_fold")
    dock.load_folded_real_world_srs_template_into_project()
    stack_fold = win.model.get_surface_stack()
    names_fold = [s.name for s in stack_fold.surfaces]
    assert "Pump Half-Wave Plate (WPH10M-780)" in names_fold
    assert "Pump Steering Mirror (PF10-03-P01)" in names_fold
    assert "Dichroic Combiner (DMLP900)" in names_fold
    assert "Galvo Mirror X (GVS002)" in names_fold
    assert "Galvo Relay Lens 1 Front (AC254-050-B)" in names_fold
    assert "Galvo Mirror Y (GVS002)" in names_fold
    assert "Scan Lens Front (SL50-2P2)" in names_fold
    assert any("XLPLN25XWMP2" in n for n in names_fold)
    assert "Raman Sample (chi(3) Material)" in names_fold
    assert "Stokes EOM (EO-AM-NR-C1)" in names_fold
    assert "Pump Clean-Up Filter (FBH790-10)" in names_fold
    assert "Stokes Clean-Up Filter (FELH1000)" in names_fold
    assert "Stokes Dump Filter 1 (FESH0850)" in names_fold
    assert "Stokes Dump Filter 2 (FESH0800)" in names_fold
    assert len(win._last_traced_rays) > 0


def test_folded_real_world_srs_template_broken_optical_axis():
    """Verify that galvo mirrors genuinely break/reflect the optical axis, 4f relay conjugates galvos, dichroic joins fs pulses, and EOM is active."""
    from optics_app.analysis.ultrafast_nonlinear import (
        build_folded_real_world_srs_template,
        save_folded_srs_microscope_json,
        load_folded_srs_microscope_json,
    )

    cfg = ThorlabsSRSMicroscopeConfig(
        scanning_architecture="Folded Real-World SRS (Dichroic + Reflected Galvos + Steering Mirrors)",
        objective_model="Olympus XLPLN25XWMP2 (25x NA 1.05 W Multiphoton)",
    )
    stack = build_folded_real_world_srs_template(cfg)
    assert len(stack.surfaces) == 37

    names = [s.name for s in stack.surfaces]
    assert "Pump Laser (797 nm fs)" in names
    assert "Pump Half-Wave Plate (WPH10M-780)" in names
    assert "Pump PBS (PBS252)" in names
    assert "Pump Clean-Up Filter (FBH790-10)" in names
    assert "Pump Steering Mirror (PF10-03-P01)" in names
    assert "Dichroic Combiner (DMLP900)" in names
    assert "Galvo Mirror X (GVS002)" in names
    assert "Galvo Relay Lens 1 Front (AC254-050-B)" in names
    assert "Galvo Relay Lens 1 Rear (AC254-050-B)" in names
    assert "Inter-Galvo Relay Pupil Conjugate" in names
    assert "Galvo Relay Lens 2 Front (AC254-050-B)" in names
    assert "Galvo Relay Lens 2 Rear (AC254-050-B)" in names
    assert "Galvo Mirror Y (GVS002)" in names
    assert "Scan Lens Front (SL50-2P2)" in names
    assert "Tube Lens Front (TTL200MP)" in names
    assert "Objective (XLPLN25XWMP2)" in names
    assert "Raman Sample (chi(3) Material)" in names
    assert "Stokes Dump Filter 1 (FESH0850)" in names
    assert "Stokes Dump Filter 2 (FESH0800)" in names
    assert "Photodiode (DET10N2)" in names
    assert "Stokes Laser (1040 nm fs)" in names
    assert "Stokes Half-Wave Plate (WPH10M-1064)" in names
    assert "Stokes Clean-Up Filter (FELH1000)" in names
    assert "Stokes Steering Mirror (PF10-03-P01)" in names
    assert "Stokes EOM (EO-AM-NR-C1)" in names
    assert "Stokes PBS (PBS252)" in names
    assert "Pump Power Dump (BT610)" in names
    assert "Polarization Dump (BT610)" in names
    assert "Combiner Leak Dump (BT610)" in names

    # 1. Verify mirror tilts that break the optical axis
    s_pump_steer = stack.surfaces[4]
    s_dichroic = stack.surfaces[5]
    s_gx = stack.surfaces[6]
    s_gy = stack.surfaces[12]

    assert s_pump_steer.is_reflective()
    assert s_pump_steer.tilt_x == 45.0
    assert s_dichroic.is_reflective()
    assert s_dichroic.tilt_x == -45.0
    assert s_gx.is_reflective()
    assert s_gx.tilt_y == 45.0
    assert s_gy.is_reflective()
    assert s_gy.tilt_y == -45.0

    # 2. Vector ray tracing: verify beam direction changes at folds
    system = OpticalSystem(stack)
    ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=0.797)
    traced = system.trace_ray(ray)
    assert not traced.vignetted
    assert len(traced.history) == 28
    assert traced.history[-1]["surface_index"] == 27
    assert stack.surfaces[27].name == "Photodiode (DET10N2)"

    # Ray before pump mirror: +Z direction
    h0 = traced.history[0]
    assert h0["dir_z"] > 0.99

    # Ray after pump mirror: folded into +Y direction
    h4 = traced.history[4]
    assert h4["dir_y"] > 0.99

    # Ray after dichroic combiner: folded back into +Z direction
    h5 = traced.history[5]
    assert h5["dir_z"] > 0.99

    # Ray after Galvo X: folded into -X direction
    h6 = traced.history[6]
    assert h6["dir_x"] < -0.99

    # Rays through inter-galvo 4f relay: propagate along -X
    h7 = traced.history[7]
    assert h7["dir_x"] < -0.99
    h11 = traced.history[11]
    assert h11["dir_x"] < -0.99

    # Ray after Galvo Y: folded back into +Z direction toward Scan Lens
    h12 = traced.history[12]
    assert h12["dir_z"] > 0.99

    # Ray through objective, chi(3) Raman sample, and photodiode
    assert traced.history[19]["dir_z"] > 0.99
    assert traced.history[20]["dir_z"] > 0.99
    assert traced.history[27]["dir_z"] > 0.99

    # 3. JSON round-trip
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        tmp_path = tf.name
    try:
        save_folded_srs_microscope_json(tmp_path, cfg)
        assert os.path.exists(tmp_path)
        loaded = load_folded_srs_microscope_json(tmp_path)
        assert loaded is not None
        assert len(loaded.surface_stack.surfaces) == 37
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_galvo_reflection_fold_on_standard_templates():
    """Verify that enable_galvo_reflection_fold breaks optical axis on both LSM core and full SRS."""
    # LSM Core
    cfg_lsm = ThorlabsSRSMicroscopeConfig(enable_galvo_reflection_fold=True)
    stack_lsm = build_thorlabs_lsm_template(cfg_lsm)
    assert stack_lsm.surfaces[1].tilt_y == 45.0
    assert stack_lsm.surfaces[2].tilt_y == -45.0

    sys_lsm = OpticalSystem(stack_lsm)
    ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=0.797)
    tr_lsm = sys_lsm.trace_ray(ray)
    assert not tr_lsm.vignetted
    assert len(tr_lsm.history) == 17

    # Full SRS
    cfg_srs = ThorlabsSRSMicroscopeConfig(enable_galvo_reflection_fold=True)
    stack_srs = build_srs_microscope_template(cfg_srs)
    assert stack_srs.surfaces[2].tilt_y == 45.0
    assert stack_srs.surfaces[3].tilt_y == -45.0

    sys_srs = OpticalSystem(stack_srs)
    tr_srs = sys_srs.trace_ray(ray)
    assert not tr_srs.vignetted
    assert len(tr_srs.history) == 18




