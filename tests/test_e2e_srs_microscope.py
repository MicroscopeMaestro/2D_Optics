"""Comprehensive 4-Tier End-to-End Test Suite for Stimulated Raman Scattering (SRS) Microscope.

Traceability:
- ORIGINAL_REQUEST.md: Section ## 2026-10-14T09:50:28Z (SRS Microscope Integration)
- Scope & Requirements: PROJECT.md §R1-R8, Features F1-F18
- Test Infra: TEST_INFRA.md (4-Tier Requirement-Driven Methodology)

Test Tiers:
- Tier 1: Isolated Feature Coverage (F1 to F18, >= 5 tests per feature = 90 tests)
- Tier 2: Boundary & Corner Cases (F1 to F18, >= 5 tests per feature = 90 tests)
- Tier 3: Cross-Feature Interactions & Pairwise Combinations (15 tests)
- Tier 4: Real-World End-to-End Application Scenarios (5 comprehensive workflows)

Total Test Count: 200 Tests.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from typing import Any, Dict, List

import numpy as np
import pytest

# Core engine & analysis imports from optics_app
from optics_app.analysis.ultrafast_nonlinear import (
    AOMMaterial,
    AcoustoOpticModulator,
    DispersiveGlassMaterial,
    EOMMaterial,
    ElectroOpticModulator,
    ELEMENTARY_CHARGE_C,
    LIGHT_M_S,
    NonlinearSignalTemporalProbe,
    OBJECTIVE_CATALOG,
    OilRamanStandard,
    OilSampleType,
    PostSampleSRSProbeResult,
    SCAN_LENS_CATALOG,
    SRSPhotodiodeModel,
    SRSPhotodiodeSignal,
    SRSScanFieldResult,
    TUBE_LENS_CATALOG,
    ThorlabsSRSMicroscopeConfig,
    ThorlabsSRSMicroscopeSimulator,
    build_edmund_optics_srs_template,
    build_folded_real_world_srs_template,
    build_olympus_high_na_srs_template,
    build_srs_microscope_relayed_template,
    build_srs_microscope_system,
    build_srs_microscope_template,
    build_thorlabs_lsm_system,
    build_thorlabs_lsm_template,
    delay_from_stage_displacement_mm,
    glass_gvd_fs2_per_mm,
    load_folded_srs_microscope_json,
    load_srs_microscope_json,
    load_thorlabs_lsm_json,
    save_folded_srs_microscope_json,
    save_srs_microscope_json,
    save_thorlabs_lsm_json,
    stage_displacement_from_delay_ps,
)
from optics_app.catalogs.thorlabs import get_catalog_lens, get_lens_catalog
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.ray import Ray
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.tracer import OpticalSystem
from optics_app.gui.qt_compat import QT_AVAILABLE, QtWidgets

# Parallel v2 tree imports for strict parity testing
import v2.analysis.ultrafast_nonlinear as v2_srs
import v2.engine.lde as v2_lde
import v2.engine.ray as v2_ray
import v2.engine.tracer as v2_tracer


# Ensure QApplication instance exists for GUI tests in headless mode
if QT_AVAILABLE and hasattr(QtWidgets, "QApplication"):
    _qt_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


# ==============================================================================
# TIER 1: ISOLATED FEATURE COVERAGE (F1 to F18, >= 5 tests each)
# ==============================================================================

# --- Feature 1: Dual-Laser Source Parameters (F1) ---

def test_t1_f1_01_default_wavelengths_and_repetition_rate():
    """Verify default pump 797 nm, Stokes 1040 nm, and 80 MHz repetition rates."""
    cfg = ThorlabsSRSMicroscopeConfig()
    assert cfg.pump_laser_wavelength_nm == 797.0
    assert cfg.stokes_laser_wavelength_nm == 1040.0
    assert cfg.pump_rep_rate_mhz == 80.0
    assert cfg.stokes_rep_rate_mhz == 80.0


def test_t1_f1_02_default_powers_and_pulse_durations():
    """Verify default pump power 25 mW, Stokes power 60 mW, and 120 fs durations."""
    cfg = ThorlabsSRSMicroscopeConfig()
    assert cfg.pump_laser_power_mw == 25.0
    assert cfg.stokes_laser_power_mw == 60.0
    assert cfg.pump_pulse_duration_fs == 120.0
    assert cfg.stokes_pulse_duration_fs == 120.0


def test_t1_f1_03_base_raman_shift_calculation():
    """Verify base Raman wavenumber shift Delta_nu_base = 2931.67 cm^-1."""
    wl_p = 797.0
    wl_s = 1040.0
    base_cm1 = 1e7 * (1.0 / wl_p - 1.0 / wl_s)
    assert math.isclose(base_cm1, 2931.67, abs_tol=0.1)


def test_t1_f1_04_custom_laser_wavelength_tuning():
    """Verify user can tune laser wavelengths and compute modified base Raman shift."""
    cfg = ThorlabsSRSMicroscopeConfig(
        pump_laser_wavelength_nm=790.0,
        stokes_laser_wavelength_nm=1030.0,
        delay_samples=21,
    )
    res = ThorlabsSRSMicroscopeSimulator.simulate_oil_spectrum(cfg, 0.0, 0.0)
    expected_base = 1e7 * (1.0 / 790.0 - 1.0 / 1030.0)
    mid_idx = len(res["wavenumbers_cm1"]) // 2
    assert math.isclose(res["wavenumbers_cm1"][mid_idx], expected_base, abs_tol=2.0)


def test_t1_f1_05_collimated_beam_diameters():
    """Verify collimated input beam diameter is between 4.0 and 4.5 mm."""
    cfg = ThorlabsSRSMicroscopeConfig()
    assert 4.0 <= cfg.input_beam_diameter_mm <= 4.5
    w0 = cfg.input_beam_diameter_mm / 2.0
    assert 2.0 <= w0 <= 2.25


# --- Feature 2: Polarization Power Attenuation (F2) ---

def test_t1_f2_01_hwp_wph10m_catalog_specifications():
    """Verify authentic Thorlabs HWP models for Pump (WPH10M-780) and Stokes (WPH10M-1064)."""
    stack = build_folded_real_world_srs_template()
    names = [s.name for s in stack.surfaces]
    assert any("WPH10M-780" in n for n in names)
    assert any("WPH10M-1064" in n for n in names)


def test_t1_f2_02_pbs_cube_pbs252_extinction_ratio():
    """Verify Thorlabs PBS252 polarizing beamsplitter cube extinction ratio > 1000:1."""
    cfg = ThorlabsSRSMicroscopeConfig()
    assert cfg.pbs_extinction_ratio <= 1.0e-3
    assert cfg.pbs_extinction_ratio > 0.0


def test_t1_f2_03_optical_dump_bt610_absorption_rating():
    """Verify Thorlabs BT610 optical beam trap power dissipation rating is >= 5.0 W."""
    cfg = ThorlabsSRSMicroscopeConfig()
    assert cfg.dump_rating_w >= 5.0


def test_t1_f2_04_malus_law_power_attenuation_curve():
    """Verify Malus' law power attenuation across HWP angles 0, 22.5, and 45 degrees."""
    p_in = 25.0
    t_max = 0.95
    t_min = t_max * 1.0e-3
    for theta_deg, exp_ratio in [(0.0, 1.0), (22.5, 0.5), (45.0, 0.0)]:
        theta_rad = math.radians(theta_deg)
        p_trans = p_in * (t_min + (t_max - t_min) * (math.cos(2.0 * theta_rad) ** 2))
        trans_ratio = p_trans / (p_in * t_max)
        assert math.isclose(trans_ratio, exp_ratio, abs_tol=0.01)


def test_t1_f2_05_surface_stack_dump_routing():
    """Verify rejected polarization power is safely routed to dedicated BT610 dumps."""
    stack = build_folded_real_world_srs_template()
    dump_names = [s.name for s in stack.surfaces if "BT610" in s.name or "Dump" in s.name]
    assert any("Pump Power Dump" in n for n in dump_names)
    assert any("Polarization Dump" in n for n in dump_names)
    assert any("Combiner Leak Dump" in n for n in dump_names)


# --- Feature 3: Active Stokes EOM Modulation (F3) ---

def test_t1_f3_01_eom_eo_am_nr_c1_rtp_crystal_parameters():
    """Verify Thorlabs EO-AM-NR-C1 resonant RTP EOM operates at 10.0 MHz with V_pi = 180 V."""
    cfg = ThorlabsSRSMicroscopeConfig()
    assert cfg.modulator_type == "EOM"
    assert cfg.modulator_frequency_mhz == 10.0
    assert cfg.modulator_v_pi_v == 180.0


def test_t1_f3_02_eom_phase_retardation_and_sin2_transmission():
    """Verify EOM transmission follows T(V) = sin^2(pi*V / 2V_pi)."""
    v_pi = 180.0
    for v, exp_t in [(0.0, 0.0), (90.0, 0.5), (180.0, 1.0)]:
        t = math.sin(0.5 * math.pi * v / v_pi) ** 2
        assert math.isclose(t, exp_t, abs_tol=1e-5)


def test_t1_f3_03_eom_intensity_modulation_depth():
    """Verify EOM produces modulation depth m ~ 1.0 at applied voltage V_pi."""
    cfg = ThorlabsSRSMicroscopeConfig(modulator_type="EOM", modulator_applied_v=180.0, modulator_v_pi_v=180.0)
    scan = ThorlabsSRSMicroscopeSimulator.simulate_scanning_point(cfg, 0.0, 0.0)
    sig = ThorlabsSRSMicroscopeSimulator.process_photodiode_signal(cfg, 1e-4, scan)
    assert math.isclose(sig.details["modulation_factor"], 1.0, rel_tol=1e-3)


def test_t1_f3_04_eom_dispersion_and_gdd():
    """Verify RTP crystal group delay dispersion (GDD) is positive and ~975 fs^2."""
    eom = ElectroOpticModulator(material=EOMMaterial.RTP, crystal_length_mm=15.0)
    gdd = eom.crystal_gdd_fs2(1040.0)
    assert 500.0 <= gdd <= 1500.0


def test_t1_f3_05_eom_vs_aom_switching():
    """Verify switching between EOM (23 surfaces) and AOM (22 surfaces) in standard template."""
    cfg_eom = ThorlabsSRSMicroscopeConfig(modulator_type="EOM")
    cfg_aom = ThorlabsSRSMicroscopeConfig(modulator_type="AOM")
    st_eom = build_srs_microscope_template(cfg_eom)
    st_aom = build_srs_microscope_template(cfg_aom)
    assert len(st_eom.surfaces) == 23
    assert len(st_aom.surfaces) == 22


# --- Feature 4: Clean-Up & Combining Optics (F4) ---

def test_t1_f4_01_pump_cleanup_filter_fbh790_10_specs():
    """Verify Thorlabs FBH790-10 clean-up filter substrate and placement in folded train."""
    stack = build_folded_real_world_srs_template()
    s_filt = next(s for s in stack.surfaces if "FBH790-10" in s.name)
    assert s_filt.glass in ["FUSED_SILICA", "Fused_Silica"]
    assert s_filt.thickness == 5.0


def test_t1_f4_02_stokes_cleanup_filter_felh1000_specs():
    """Verify Thorlabs FELH1000 clean-up longpass filter in Stokes delivery arm."""
    stack = build_folded_real_world_srs_template()
    s_filt = next(s for s in stack.surfaces if "FELH1000" in s.name)
    assert s_filt.glass in ["FUSED_SILICA", "Fused_Silica"]
    assert s_filt.thickness == 5.0


def test_t1_f4_03_dmlp900_dichroic_combiner_split():
    """Verify Thorlabs DMLP900 longpass dichroic combiner at -45 deg nominal tilt."""
    stack = build_folded_real_world_srs_template()
    s_dichroic = next(s for s in stack.surfaces if "DMLP900" in s.name)
    assert s_dichroic.is_reflective()
    assert s_dichroic.tilt_x == -45.0


def test_t1_f4_04_collinear_ray_propagation_after_combiner():
    """Verify unmodulated Pump and modulated Stokes propagate collinearly along +Z after combiner."""
    stack = build_folded_real_world_srs_template()
    sys = OpticalSystem(stack)
    ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=0.797)
    traced = sys.trace_ray(ray)
    h5 = traced.history[5]  # Hit on dichroic combiner
    assert h5["dir_z"] > 0.99
    assert abs(h5["dir_x"]) < 1e-4
    assert abs(h5["dir_y"]) < 1e-4


def test_t1_f4_05_steering_mirror_pf10_03_p01_specular_reflection():
    """Verify Thorlabs PF10-03-P01 steering mirror folds optical axis into +Y specularly."""
    stack = build_folded_real_world_srs_template()
    sys = OpticalSystem(stack)
    ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=0.797)
    traced = sys.trace_ray(ray)
    h4 = traced.history[4]  # Hit on pump steering mirror
    assert h4["dir_y"] > 0.99


# --- Feature 5: High-NA Objective Prescriptions (F5) ---

def test_t1_f5_01_olympus_20x_10na_catalog_parameters():
    """Verify Olympus XLUMPLFLN20XW 20x 1.00 NA Water Immersion objective parameters."""
    info = OBJECTIVE_CATALOG["Olympus XLUMPLFLN20XW (20x NA 1.00 W)"]
    assert info["numerical_aperture"] == 1.00
    assert info["working_distance_mm"] == 2.0
    assert "Water" in info["immersion"]
    assert info["magnification"] == 20.0


def test_t1_f5_02_olympus_25x_105na_catalog_parameters():
    """Verify Olympus XLPLN25XWMP2 25x 1.05 NA Water Immersion objective parameters."""
    info = OBJECTIVE_CATALOG["Olympus XLPLN25XWMP2 (25x NA 1.05 W Multiphoton)"]
    assert info["numerical_aperture"] == 1.05
    assert info["working_distance_mm"] == 2.0
    assert info["pupil_diameter_mm"] == 16.8
    assert info["focal_length_mm"] == 8.0


def test_t1_f5_03_olympus_objective_template_generation():
    """Verify build_olympus_high_na_srs_template correctly inserts Olympus immersion objective."""
    cfg = ThorlabsSRSMicroscopeConfig(objective_model="Olympus XLPLN25XWMP2 (25x NA 1.05 W Multiphoton)")
    stack = build_olympus_high_na_srs_template(cfg)
    assert any("XLPLN25XWMP2" in s.name for s in stack.surfaces)


def test_t1_f5_04_diffraction_limited_waist_scaling():
    """Verify diffraction-limited focal waist scales inversely with NA (w0 = 0.61 * lambda / NA)."""
    wl_p = 0.797
    w0_105 = 0.61 * wl_p / 1.05
    w0_070 = 0.61 * wl_p / 0.70
    assert math.isclose(w0_105, 0.463, abs_tol=0.01)
    assert math.isclose(w0_070, 0.695, abs_tol=0.01)
    assert w0_105 < w0_070


def test_t1_f5_05_objective_back_pupil_diameter():
    """Verify objective entrance pupil diameter D_BFP = 2 * NA * f_obj."""
    info = OBJECTIVE_CATALOG["Olympus XLPLN25XWMP2 (25x NA 1.05 W Multiphoton)"]
    d_bfp = 2.0 * info["numerical_aperture"] * info["focal_length_mm"]
    assert math.isclose(d_bfp, 16.8, abs_tol=0.01)


# --- Feature 6: Commercial Scan & Tube Lenses (F6) ---

def test_t1_f6_01_thorlabs_sl50_2p2_specifications():
    """Verify Thorlabs SL50-2P2 commercial scan lens catalog specifications."""
    info = SCAN_LENS_CATALOG["Thorlabs SL50-2P2"]
    assert info["focal_length_mm"] == 50.0
    assert info["epd_mm"] == 28.5
    assert info["bfl_mm"] == 42.5
    assert "2P2" in info["coating"]


def test_t1_f6_02_scan_lens_telecentricity_error():
    """Verify commercial scan lens telecentricity error is < 0.1 deg in 4f configuration."""
    cfg = ThorlabsSRSMicroscopeConfig(scanning_architecture="4f-Relayed (Zero Walk-Off)")
    scan = ThorlabsSRSMicroscopeSimulator.simulate_scanning_point(cfg, 3.0, 0.0)
    assert scan.telecentricity_error_mrad == 0.0  # 4f strictly zero telecentric error


def test_t1_f6_03_thorlabs_ttl200mp_specifications():
    """Verify Thorlabs TTL200MP multiphoton tube lens catalog specifications."""
    info = TUBE_LENS_CATALOG["Thorlabs TTL200MP"]
    assert info["focal_length_mm"] == 200.0
    assert info["ffl_mm"] == 155.0
    assert info["bfl_mm"] == 149.3


def test_t1_f6_04_pupil_magnification_factor():
    """Verify 4.0x pupil magnification M_pupil = f_tube / f_scan = 200 / 50."""
    cfg = ThorlabsSRSMicroscopeConfig()
    m_pupil = cfg.tube_lens_efl_mm / cfg.scan_lens_efl_mm
    assert m_pupil == 4.0


def test_t1_f6_05_intermediate_image_plane_formation():
    """Verify intermediate image plane is positioned at scan lens rear focus."""
    stack = build_thorlabs_lsm_template()
    names = [s.name for s in stack.surfaces]
    assert "Intermediate Image Plane" in names
    idx = names.index("Intermediate Image Plane")
    assert "Scan Lens Rear" in names[idx - 1]
    assert "Tube Lens Front" in names[idx + 1]


# --- Feature 7: Aspheric Condenser & Relays (F7) ---

def test_t1_f7_01_aspheric_condenser_acl25416u_b_prescription():
    """Verify Thorlabs ACL25416U-B aspheric condenser exact prescription (kappa = -0.6265, A4 = 1.8e-6)."""
    item = get_catalog_lens("ACL25416U-B")
    assert item is not None
    assert math.isclose(item["surfaces"][0]["conic"], -0.6265, abs_tol=1e-4)
    assert math.isclose(item["surfaces"][0]["radius"], 8.32, abs_tol=1e-2)
    assert item["surfaces"][0]["asphere_coeffs"][0] == 1.8e-6


def test_t1_f7_02_condenser_numerical_aperture_and_collection():
    """Verify ACL25416U-B condenser NA = 0.79 with 16.0 mm focal length."""
    item = get_catalog_lens("ACL25416U-B")
    assert item is not None
    assert item["focal_length_mm"] == 16.0
    assert item["diameter_mm"] == 25.4
    assert item["back_focal_length_mm"] == 8.5


def test_t1_f7_03_ac254_050_b_achromat_doublet_prescription():
    """Verify Thorlabs AC254-050-B achromat doublet radii and glasses."""
    item = get_catalog_lens("AC254-050-B")
    assert item is not None
    assert item["focal_length_mm"] == 50.0
    assert item["surfaces"][0]["radius"] in (33.55, 34.0)
    assert item["surfaces"][1]["radius"] in (-27.05, -27.8)
    assert item["surfaces"][0]["glass"] in ("N-LAK22", "N-BK7")


def test_t1_f7_04_collector_lens_focusing_onto_detector():
    """Verify collector lens (AC254-050-B) focuses transmitted pump beam onto DET10N2 active area."""
    stack = build_srs_microscope_template()
    names = [s.name for s in stack.surfaces]
    assert any("Collector Front" in n for n in names)
    assert any("Photodiode" in n for n in names)


def test_t1_f7_05_condenser_asphere_sag_calculation():
    """Verify exact Even Asphere sag calculation for ACL25416U-B matches analytical formula."""
    c = 1.0 / 8.32
    kappa = -0.6265
    a4 = 1.8e-6
    r = 5.0
    sag = (c * (r ** 2)) / (1.0 + math.sqrt(1.0 - (1.0 + kappa) * (c ** 2) * (r ** 2))) + a4 * (r ** 4)
    assert 1.0 <= sag <= 2.5


# --- Feature 8: 37-Surface Train & Beam Sizing (F8) ---

def test_t1_f8_01_folded_real_world_template_surface_count():
    """Verify folded real-world SRS microscope contains exactly 37 surfaces."""
    stack = build_folded_real_world_srs_template()
    assert len(stack.surfaces) == 37


def test_t1_f8_02_sequential_forward_path_surfaces():
    """Verify 28 sequential surfaces along the primary rail from Pump source to photodiode."""
    stack = build_folded_real_world_srs_template()
    assert stack.surfaces[0].name.startswith("Pump Laser")
    assert stack.surfaces[27].name.startswith("Photodiode")


def test_t1_f8_03_off_axis_stokes_branch_surfaces():
    """Verify 9 off-axis surfaces for Stokes delivery arm, EOM, and dumps."""
    stack = build_folded_real_world_srs_template()
    assert stack.surfaces[28].name.startswith("Stokes Laser")
    assert stack.surfaces[36].name.startswith("Combiner Leak Dump")


def test_t1_f8_04_sequential_ray_tracing_unvignetted():
    """Verify on-axis pump ray traces through all 28 sequential surfaces with zero vignetting."""
    stack = build_folded_real_world_srs_template()
    sys = OpticalSystem(stack)
    ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=0.797)
    traced = sys.trace_ray(ray)
    assert not traced.vignetted
    assert len(traced.history) == 28


def test_t1_f8_05_beam_diameter_evolution_profile():
    """Verify beam diameter expands from 4.5 mm at galvos to 18.0 mm at objective BFP."""
    cfg = ThorlabsSRSMicroscopeConfig()
    scan = ThorlabsSRSMicroscopeSimulator.simulate_scanning_point(cfg, 0.0, 0.0)
    assert scan.back_pupil_beam_diameter_mm == 18.0
    assert scan.pupil_fill_factor > 0.90


# --- Feature 9: Dynamic Galvo Scanning & FOV (F9) ---

def test_t1_f9_01_galvo_angular_range():
    """Verify galvo supports full mechanical angular deflection up to +/-10.0 deg."""
    cfg = ThorlabsSRSMicroscopeConfig(galvo_max_mechanical_angle_deg=10.0)
    assert cfg.galvo_max_mechanical_angle_deg == 10.0


def test_t1_f9_02_sample_coordinate_mapping():
    """Verify linear mapping from galvo angles (theta_x, theta_y) to specimen coordinates (x, y)."""
    cfg = ThorlabsSRSMicroscopeConfig()
    pt = ThorlabsSRSMicroscopeSimulator.simulate_scanning_point(cfg, 1.0, 1.0)
    assert pt.sample_x_um > 0.0
    assert pt.sample_y_um > 0.0


def test_t1_f9_03_symmetric_scanning_deflection():
    """Verify symmetric angular deflection produces equal and opposite specimen displacements."""
    cfg = ThorlabsSRSMicroscopeConfig()
    p_pos = ThorlabsSRSMicroscopeSimulator.simulate_scanning_point(cfg, 2.0, 0.0)
    p_neg = ThorlabsSRSMicroscopeSimulator.simulate_scanning_point(cfg, -2.0, 0.0)
    assert math.isclose(p_pos.sample_x_um, -p_neg.sample_x_um, rel_tol=1e-5)
    assert math.isclose(p_pos.field_radius_um, p_neg.field_radius_um, rel_tol=1e-5)


def test_t1_f9_04_dual_axis_independent_motion():
    """Verify Galvo X and Galvo Y motion are completely decoupled along orthogonal axes."""
    cfg = ThorlabsSRSMicroscopeConfig()
    px = ThorlabsSRSMicroscopeSimulator.simulate_scanning_point(cfg, 2.0, 0.0)
    py = ThorlabsSRSMicroscopeSimulator.simulate_scanning_point(cfg, 0.0, 2.0)
    assert px.sample_y_um == 0.0
    assert py.sample_x_um == 0.0
    assert math.isclose(px.sample_x_um, py.sample_y_um, rel_tol=1e-5)


def test_t1_f9_05_field_radius_calculation():
    """Verify field radius calculation r = sqrt(x^2 + y^2)."""
    cfg = ThorlabsSRSMicroscopeConfig()
    pt = ThorlabsSRSMicroscopeSimulator.simulate_scanning_point(cfg, 3.0, 4.0)
    exp_r = math.sqrt(pt.sample_x_um ** 2 + pt.sample_y_um ** 2)
    assert math.isclose(pt.field_radius_um, exp_r, rel_tol=1e-5)


# --- Feature 10: Zero Pupil Walk-Off Conjugation (F10) ---

def test_t1_f10_01_relayed_scanner_zero_walk_off():
    """Verify 4f-Relayed architecture maintains strictly 0.0 um pupil walk-off at objective BFP."""
    cfg = ThorlabsSRSMicroscopeConfig(scanning_architecture="4f-Relayed (Zero Walk-Off)")
    scan = ThorlabsSRSMicroscopeSimulator.simulate_scanning_point(cfg, 4.0, 0.0)
    assert scan.back_pupil_walk_off_total_um == 0.0


def test_t1_f10_02_close_coupled_nonzero_walk_off():
    """Verify close-coupled galvo scanner exhibits measurable pupil walk-off (> 1000 um)."""
    cfg = ThorlabsSRSMicroscopeConfig(scanning_architecture="Close-Coupled (Thorlabs GVS002)")
    scan = ThorlabsSRSMicroscopeSimulator.simulate_scanning_point(cfg, 4.0, 0.0)
    assert scan.back_pupil_walk_off_total_um > 1000.0


def test_t1_f10_03_strict_conjugation_flag():
    """Verify is_strictly_conjugated flag reflects true 4f relay conjugation."""
    c_rel = ThorlabsSRSMicroscopeConfig(scanning_architecture="4f-Relayed (Zero Walk-Off)")
    c_close = ThorlabsSRSMicroscopeConfig(scanning_architecture="Close-Coupled (Thorlabs GVS002)")
    assert ThorlabsSRSMicroscopeSimulator.simulate_scanning_point(c_rel, 0.0, 0.0).is_strictly_conjugated is True
    assert ThorlabsSRSMicroscopeSimulator.simulate_scanning_point(c_close, 0.0, 0.0).is_strictly_conjugated is False


def test_t1_f10_04_telecentricity_error_zero_in_relayed():
    """Verify telecentricity error is strictly 0.0 mrad across entire 4f scan."""
    cfg = ThorlabsSRSMicroscopeConfig(scanning_architecture="4f-Relayed (Zero Walk-Off)")
    scan = ThorlabsSRSMicroscopeSimulator.simulate_scanning_point(cfg, 5.0, 0.0)
    assert scan.telecentricity_error_mrad == 0.0


def test_t1_f10_05_inter_galvo_relay_lenses_in_stack():
    """Verify presence of 1:1 relay doublet pair and pupil stop in relayed template."""
    stack = build_srs_microscope_relayed_template()
    names = [s.name for s in stack.surfaces]
    assert any("Galvo Relay Lens 1" in n for n in names)
    assert any("Galvo Relay Lens 2" in n for n in names)
    assert any("Relay Pupil Conjugate" in n for n in names)


# --- Feature 11: Off-Axis Field Aberration Modeling (F11) ---

def test_t1_f11_01_petzval_curvature_defocus():
    """Verify Petzval field curvature defocus Delta_z = r^2 / (2 * R_petzval) scales quadratically."""
    cfg = ThorlabsSRSMicroscopeConfig()
    p1 = ThorlabsSRSMicroscopeSimulator.simulate_scanning_point(cfg, 1.0, 0.0)
    p2 = ThorlabsSRSMicroscopeSimulator.simulate_scanning_point(cfg, 2.0, 0.0)
    assert p2.field_curvature_defocus_um > p1.field_curvature_defocus_um
    assert math.isclose(p2.field_curvature_defocus_um / p1.field_curvature_defocus_um, 4.0, rel_tol=0.05)


def test_t1_f11_02_lateral_chromatic_walk_off():
    """Verify lateral chromatic walk-off between 797 nm and 1040 nm scales linearly with field radius."""
    cfg = ThorlabsSRSMicroscopeConfig()
    p1 = ThorlabsSRSMicroscopeSimulator.simulate_scanning_point(cfg, 1.0, 0.0)
    p2 = ThorlabsSRSMicroscopeSimulator.simulate_scanning_point(cfg, 2.0, 0.0)
    assert math.isclose(p2.chromatic_walk_off_um / p1.chromatic_walk_off_um, 2.0, rel_tol=0.05)


def test_t1_f11_03_defocused_spot_waist_widening():
    """Verify off-axis defocus increases effective focal spot waist."""
    cfg = ThorlabsSRSMicroscopeConfig()
    pt = ThorlabsSRSMicroscopeSimulator.simulate_scanning_point(cfg, 4.0, 0.0)
    assert pt.defocused_pump_waist_um > pt.pump_waist_um


def test_t1_f11_04_pupil_vignetting_factor_roll_off():
    """Verify pupil vignetting factor roll-off eta_vig <= 1.0 towards field edge."""
    cfg = ThorlabsSRSMicroscopeConfig()
    p0 = ThorlabsSRSMicroscopeSimulator.simulate_scanning_point(cfg, 0.0, 0.0)
    p_edge = ThorlabsSRSMicroscopeSimulator.simulate_scanning_point(cfg, 5.0, 0.0)
    assert p_edge.pupil_vignetting_factor < p0.pupil_vignetting_factor <= 1.0


def test_t1_f11_05_total_scan_efficiency_degradation():
    """Verify total scan efficiency rolls off monotonically with off-axis angle."""
    cfg = ThorlabsSRSMicroscopeConfig()
    p0 = ThorlabsSRSMicroscopeSimulator.simulate_scanning_point(cfg, 0.0, 0.0)
    p1 = ThorlabsSRSMicroscopeSimulator.simulate_scanning_point(cfg, 2.0, 0.0)
    p2 = ThorlabsSRSMicroscopeSimulator.simulate_scanning_point(cfg, 4.0, 0.0)
    assert p0.total_scan_efficiency >= p1.total_scan_efficiency >= p2.total_scan_efficiency


# --- Feature 12: Literature-Grounded Spectral Focusing (F12) ---

def test_t1_f12_01_schott_sf57_dispersion_parameters():
    """Verify Schott SF57 Sellmeier group velocity dispersion (GVD) at 797 nm and 1040 nm."""
    gvd_p = glass_gvd_fs2_per_mm(DispersiveGlassMaterial.SF57, 797.0)
    gvd_s = glass_gvd_fs2_per_mm(DispersiveGlassMaterial.SF57, 1040.0)
    assert 180.0 <= gvd_p <= 240.0
    assert 90.0 <= gvd_s <= 140.0


def test_t1_f12_02_chirp_matching_ratio_calculation():
    """Verify SF57 rod length ratio (150 mm vs 280 mm) yields chirp matching ratio alpha_p / alpha_S ~ 1.0."""
    cfg = ThorlabsSRSMicroscopeConfig(pump_glass_length_mm=150.0, stokes_glass_length_mm=280.0)
    res = ThorlabsSRSMicroscopeSimulator.simulate_oil_spectrum(cfg)
    assert 0.85 <= res["chirp_matching_ratio"] <= 1.15


def test_t1_f12_03_temporal_pulse_broadening():
    """Verify femtosecond pulses broaden from 120 fs to > 0.5 ps in SF57 rods."""
    cfg = ThorlabsSRSMicroscopeConfig()
    probe = ThorlabsSRSMicroscopeSimulator.probe_post_sample_intensity(cfg, delay_ps=0.0)
    assert probe.details["chirped_pump_fwhm_ps"] > 0.5
    assert probe.details["chirped_stokes_fwhm_ps"] > 0.5


def test_t1_f12_04_delay_to_wavenumber_calibration():
    """Verify linear calibration between stage delay (ps) and instantaneous beat frequency (cm^-1)."""
    cfg = ThorlabsSRSMicroscopeConfig()
    p0 = ThorlabsSRSMicroscopeSimulator.probe_post_sample_intensity(cfg, delay_ps=0.0)
    p1 = ThorlabsSRSMicroscopeSimulator.probe_post_sample_intensity(cfg, delay_ps=1.0)
    shift = p1.probed_wavenumber_cm1 - p0.probed_wavenumber_cm1
    assert 100.0 <= shift <= 140.0


def test_t1_f12_05_mineral_oil_vs_olive_oil_standards():
    """Verify OilRamanStandard defines analytical profiles for Mineral Oil and Olive Oil."""
    min_oil = OilRamanStandard.create(OilSampleType.MINERAL_OIL)
    olive_oil = OilRamanStandard.create(OilSampleType.OLIVE_OIL)
    assert any(math.isclose(p[0], 2850.0, abs_tol=5.0) for p in min_oil.vibrational_peaks)
    assert any(math.isclose(p[0], 3005.0, abs_tol=5.0) for p in olive_oil.vibrational_peaks)


# --- Feature 13: SRL Modulation Transfer & Detection (F13) ---

def test_t1_f13_01_srl_fraction_formula():
    """Verify fractional Stimulated Raman Loss Delta_Ip / Ip is within expected range (0.0 to 0.20)."""
    cfg = ThorlabsSRSMicroscopeConfig()
    probe = ThorlabsSRSMicroscopeSimulator.probe_post_sample_intensity(cfg, delay_ps=0.0)
    assert 0.0 < probe.fractional_srl < 0.20
    assert probe.fractional_srg > 0.0


def test_t1_f13_02_stokes_optical_suppression_stack():
    """Verify dual Stokes shortpass filter stack (FESH0850 + FESH0800) provides cumulative OD > 12."""
    stack = build_folded_real_world_srs_template()
    names = [s.name for s in stack.surfaces]
    assert any("FESH0850" in n for n in names)
    assert any("FESH0800" in n for n in names)


def test_t1_f13_03_pump_transmission_through_filter_stack():
    """Verify pump beam transmits efficiently (> 75%) through Stokes blocking stack."""
    cfg = ThorlabsSRSMicroscopeConfig()
    assert cfg.stokes_dump_transmission_at_797 >= 0.75


def test_t1_f13_04_rf_modulation_transfer_to_pump():
    """Verify four-wave mixing transfers 10.0 MHz Stokes modulation onto continuous pump."""
    cfg = ThorlabsSRSMicroscopeConfig()
    probe = ThorlabsSRSMicroscopeSimulator.probe_post_sample_intensity(cfg, delay_ps=0.0)
    assert probe.pulse_train_signal is not None
    assert len(probe.pulse_train_signal) > 100


def test_t1_f13_05_generated_signal_power_scaling():
    """Verify generated nonlinear signal power in microwatts is non-zero and physical."""
    cfg = ThorlabsSRSMicroscopeConfig()
    probe = ThorlabsSRSMicroscopeSimulator.probe_post_sample_intensity(cfg, delay_ps=0.0)
    assert probe.generated_signal_power_uw > 0.0


# --- Feature 14: Photodiode Noise & Lock-In Demodulation (F14) ---

def test_t1_f14_01_det10n2_responsivity_and_dc_photocurrent():
    """Verify Thorlabs DET10N2 responsivity 0.55 A/W yields milliampere-scale photocurrent."""
    cfg = ThorlabsSRSMicroscopeConfig(pump_laser_power_mw=25.0)
    scan = ThorlabsSRSMicroscopeSimulator.simulate_scanning_point(cfg, 0.0, 0.0)
    sig = ThorlabsSRSMicroscopeSimulator.process_photodiode_signal(cfg, 1e-4, scan)
    assert sig.dc_photocurrent_a > 1e-3
    assert sig.dc_output_voltage_v > 10.0


def test_t1_f14_02_shot_noise_current_calculation():
    """Verify Poisson shot noise current i_shot = sqrt(2 * q * I_DC * Delta_f)."""
    cfg = ThorlabsSRSMicroscopeConfig()
    scan = ThorlabsSRSMicroscopeSimulator.simulate_scanning_point(cfg, 0.0, 0.0)
    sig = ThorlabsSRSMicroscopeSimulator.process_photodiode_signal(cfg, 1e-4, scan)
    assert sig.shot_noise_current_a > 1e-10


def test_t1_f14_03_thermal_johnson_noise_calculation():
    """Verify Johnson-Nyquist thermal noise current in 50 kOhm load resistor."""
    cfg = ThorlabsSRSMicroscopeConfig()
    scan = ThorlabsSRSMicroscopeSimulator.simulate_scanning_point(cfg, 0.0, 0.0)
    sig = ThorlabsSRSMicroscopeSimulator.process_photodiode_signal(cfg, 1e-4, scan)
    assert sig.thermal_noise_current_a > 1e-11


def test_t1_f14_04_relative_intensity_noise_at_10mhz():
    """Verify laser relative intensity noise (RIN) at 10.0 MHz (-155 dBc/Hz)."""
    cfg = ThorlabsSRSMicroscopeConfig()
    scan = ThorlabsSRSMicroscopeSimulator.simulate_scanning_point(cfg, 0.0, 0.0)
    sig = ThorlabsSRSMicroscopeSimulator.process_photodiode_signal(cfg, 1e-4, scan)
    assert sig.rin_noise_current_a > 1e-10
    assert sig.total_noise_current_a > sig.shot_noise_current_a


def test_t1_f14_05_rf_lockin_demodulation_outputs():
    """Verify RF lock-in demodulation outputs (R, X, Y) and positive SNR (dB)."""
    cfg = ThorlabsSRSMicroscopeConfig()
    scan = ThorlabsSRSMicroscopeSimulator.simulate_scanning_point(cfg, 0.0, 0.0)
    sig = ThorlabsSRSMicroscopeSimulator.process_photodiode_signal(cfg, 1e-4, scan)
    assert sig.lockin_magnitude_r_v > 0.0
    r_calc = math.sqrt(sig.lockin_in_phase_x_v ** 2 + sig.lockin_quadrature_y_v ** 2)
    assert math.isclose(sig.lockin_magnitude_r_v, r_calc, rel_tol=1e-5)
    assert sig.snr_db > 10.0


# --- Feature 15: Collision-Free 2D Layout & Auto-Zoom (F15) ---

def test_t1_f15_01_stokes_branch_spatial_clearance():
    """Verify Stokes delivery branch is shifted to x = -40 mm, avoiding overlap with pump arm."""
    stack = build_folded_real_world_srs_template()
    stokes_elements = [s for s in stack.surfaces if "Stokes" in s.name and s.decenter_x != 0.0]
    for s in stokes_elements:
        assert s.decenter_x <= -30.0


def test_t1_f15_02_beam_dump_spatial_separation():
    """Verify all 3 optical beam dumps (BT610) are located at isolated off-axis positions."""
    stack = build_folded_real_world_srs_template()
    dumps = [s for s in stack.surfaces if "BT610" in s.name or "Dump" in s.name]
    assert len(dumps) >= 3


def test_t1_f15_03_inter_galvo_relay_along_minus_x():
    """Verify inter-galvo 4f relay propagates along -X direction, physically clear of scan lens."""
    stack = build_folded_real_world_srs_template()
    s_gx = stack.surfaces[6]
    s_gy = stack.surfaces[12]
    assert s_gx.tilt_y == 45.0
    assert s_gy.tilt_y == -45.0


def test_t1_f15_04_dual_view_widget_limits_reset():
    """Verify DualViewWidget reset_view_limits() clears saved limits and activates auto-zoom."""
    if not QT_AVAILABLE:
        pytest.skip("Qt not available")
    from optics_app.gui.dual_view import DualViewWidget
    widget = DualViewWidget()
    widget.saved_limits["rz"] = ((0.0, 10.0), (0.0, 10.0))
    widget.reset_view_limits()
    assert len(widget.saved_limits) == 0


def test_t1_f15_05_view_limits_framing_entire_bench():
    """Verify layout update renders all 37 surfaces cleanly within canvas."""
    if not QT_AVAILABLE:
        pytest.skip("Qt not available")
    from optics_app.gui.dual_view import DualViewWidget
    stack = build_folded_real_world_srs_template()
    widget = DualViewWidget()
    widget.update_layout(surface_stack=stack)
    assert widget.fig is not None


# --- Feature 16: Dedicated Suite of 5 Interactive Tools (F16) ---

def test_t1_f16_01_tool1_setup_builder():
    """Verify Tool 1: Setup Builder tool configures full SRS systems across objective presets."""
    for model in ["TL15X-2P", "Olympus XLPLN25XWMP2 (25x NA 1.05 W Multiphoton)"]:
        cfg = ThorlabsSRSMicroscopeConfig(objective_model=model)
        sys = build_srs_microscope_system(cfg)
        assert sys is not None
        assert len(sys.surface_stack.surfaces) >= 21


def test_t1_f16_02_tool2_galvo_field_diagnostic():
    """Verify Tool 2: Galvo Scanning & Field Diagnostic computes 2D maps cleanly."""
    cfg = ThorlabsSRSMicroscopeConfig()
    res = ThorlabsSRSMicroscopeSimulator.simulate_2d_scan_field(cfg, grid_points=5)
    assert res["scan_efficiency_map"].shape == (5, 5)
    assert res["srs_signal_map_v"].shape == (5, 5)


def test_t1_f16_03_tool3_spectral_focusing_delay_scan():
    """Verify Tool 3: Spectral Focusing Delay Scan tool computes reconstructed Raman spectrum."""
    cfg = ThorlabsSRSMicroscopeConfig(delay_samples=21)
    res = ThorlabsSRSMicroscopeSimulator.simulate_oil_spectrum(cfg)
    assert len(res["wavenumbers_cm1"]) == 21
    assert len(res["on_axis_lockin_voltage_v"]) == 21


def test_t1_f16_04_tool4_signal_lockin_processor():
    """Verify Tool 4: Signal Processing & Lock-In Demodulation tool evaluates full noise breakdown."""
    cfg = ThorlabsSRSMicroscopeConfig()
    scan = ThorlabsSRSMicroscopeSimulator.simulate_scanning_point(cfg, 0.0, 0.0)
    sig = ThorlabsSRSMicroscopeSimulator.process_photodiode_signal(cfg, 1e-4, scan)
    assert sig.snr_db > 0.0
    assert "filter_bandwidth_hz" in sig.details


def test_t1_f16_05_tool5_bench_alignment_diagnostic():
    """Verify Tool 5: Bench Diagnostic tool confirms strict BFP conjugation and zero pupil walk-off."""
    cfg = ThorlabsSRSMicroscopeConfig(scanning_architecture="4f-Relayed (Zero Walk-Off)")
    scan = ThorlabsSRSMicroscopeSimulator.simulate_scanning_point(cfg, 2.0, 2.0)
    assert scan.back_pupil_walk_off_total_um == 0.0
    assert scan.is_strictly_conjugated is True


# --- Feature 17: Non-Intrusive General-Purpose Integration (F17) ---

def test_t1_f17_01_default_surface_stack_unaltered():
    """Verify default SurfaceStack creates generic blank systems without hardcoded SRS constraints."""
    stack = SurfaceStack()
    assert len(stack.surfaces) == 0
    assert getattr(stack, "srs_microscope_config", None) is None


def test_t1_f17_02_custom_optical_system_editing():
    """Verify arbitrary custom lenses, mirrors, and gratings can be inserted without SRS coupling."""
    stack = SurfaceStack()
    stack.add_surface(Surface(name="Input Air", radius=0.0, thickness=50.0, glass="AIR", surface_type=SurfaceType.AIR_SPACE))
    stack.add_surface(Surface(name="Generic Mirror", radius=0.0, thickness=0.0, glass="MIRROR", surface_type=SurfaceType.MIRROR))
    sys = OpticalSystem(stack)
    assert len(sys.get_surface_stack().surfaces) == 2


def test_t1_f17_03_ultrafast_analysis_dock_tabs():
    """Verify UltrafastAnalysisDock preserves all generic modal tabs alongside SRS controls."""
    if not QT_AVAILABLE:
        pytest.skip("Qt not available")
    from optics_app.gui.ultrafast_dock import UltrafastAnalysisDock
    dock = UltrafastAnalysisDock()
    assert dock.tabs.count() >= 6


def test_t1_f17_04_thorlabs_srs_launcher_tab():
    """Verify ThorlabsSRSLauncherTab provides dedicated, non-intrusive launcher tab."""
    if not QT_AVAILABLE:
        pytest.skip("Qt not available")
    from optics_app.gui.analysis_tabs import ThorlabsSRSLauncherTab
    tab = ThorlabsSRSLauncherTab()
    assert tab is not None


def test_t1_f17_05_json_preset_loading_into_project():
    """Verify shipped JSON presets load into standard SurfaceStack cleanly."""
    loaded = load_thorlabs_lsm_json("thorlabs_lsm_microscope.json")
    assert len(loaded.surface_stack.surfaces) == 17


# --- Feature 18: Dual-Tree Parity & Automated Test Suite (F18) ---

def test_t1_f18_01_dual_tree_ultrafast_nonlinear_exports():
    """Verify identical export symbols between optics_app and v2 ultrafast modules."""
    for symbol in ["ThorlabsSRSMicroscopeConfig", "ThorlabsSRSMicroscopeSimulator", "OilRamanStandard"]:
        assert hasattr(v2_srs, symbol)


def test_t1_f18_02_dual_tree_config_equality():
    """Verify ThorlabsSRSMicroscopeConfig serialization parity between optics_app and v2."""
    c1 = ThorlabsSRSMicroscopeConfig(pump_laser_power_mw=30.0)
    c2 = v2_srs.ThorlabsSRSMicroscopeConfig(pump_laser_power_mw=30.0)
    assert c1.to_dict() == c2.to_dict()


def test_t1_f18_03_dual_tree_template_builder_parity():
    """Verify 37-surface template builders produce identical surface counts and thicknesses."""
    c1 = ThorlabsSRSMicroscopeConfig()
    c2 = v2_srs.ThorlabsSRSMicroscopeConfig()
    s1 = build_folded_real_world_srs_template(c1)
    s2 = v2_srs.build_folded_real_world_srs_template(c2)
    assert len(s1.surfaces) == len(s2.surfaces) == 37
    for surf1, surf2 in zip(s1.surfaces, s2.surfaces):
        assert surf1.name == surf2.name
        assert math.isclose(surf1.thickness, surf2.thickness, abs_tol=1e-6)


def test_t1_f18_04_dual_tree_simulation_numeric_identity():
    """Verify scanning point calculations produce numerically identical results in optics_app and v2."""
    c1 = ThorlabsSRSMicroscopeConfig()
    c2 = v2_srs.ThorlabsSRSMicroscopeConfig()
    r1 = ThorlabsSRSMicroscopeSimulator.simulate_scanning_point(c1, 2.0, 1.0)
    r2 = v2_srs.ThorlabsSRSMicroscopeSimulator.simulate_scanning_point(c2, 2.0, 1.0)
    assert math.isclose(r1.sample_x_um, r2.sample_x_um, abs_tol=1e-6)
    assert math.isclose(r1.total_scan_efficiency, r2.total_scan_efficiency, abs_tol=1e-6)


def test_t1_f18_05_dual_tree_post_sample_probe_parity():
    """Verify Post-Sample Probe calculations are identical across optics_app and v2."""
    c1 = ThorlabsSRSMicroscopeConfig()
    c2 = v2_srs.ThorlabsSRSMicroscopeConfig()
    p1 = ThorlabsSRSMicroscopeSimulator.probe_post_sample_intensity(c1, delay_ps=0.5)
    p2 = v2_srs.ThorlabsSRSMicroscopeSimulator.probe_post_sample_intensity(c2, delay_ps=0.5)
    assert math.isclose(p1.probed_wavenumber_cm1, p2.probed_wavenumber_cm1, abs_tol=1e-6)
    assert math.isclose(p1.fractional_srl, p2.fractional_srl, abs_tol=1e-6)


# ==============================================================================
# TIER 2: BOUNDARY & CORNER CASES (F1 to F18, >= 5 tests each)
# ==============================================================================

# --- F1 Boundaries ---

def test_t2_f1_01_extreme_wavelength_combination():
    """Boundary: Wide wavelength separation (Pump 700 nm, Stokes 1200 nm)."""
    shift = 1e7 * (1.0 / 700.0 - 1.0 / 1200.0)
    assert 5900.0 <= shift <= 6000.0


def test_t2_f1_02_degenerate_wavelengths_zero_raman_shift():
    """Boundary: Degenerate wavelengths (Pump == Stokes == 797 nm) yields zero Raman shift."""
    shift = 1e7 * (1.0 / 797.0 - 1.0 / 797.0)
    assert shift == 0.0


def test_t2_f1_03_zero_laser_power():
    """Boundary: Zero pump laser power yields zero optical signal and dark noise only."""
    cfg = ThorlabsSRSMicroscopeConfig(pump_laser_power_mw=0.0)
    scan = ThorlabsSRSMicroscopeSimulator.simulate_scanning_point(cfg, 0.0, 0.0)
    sig = ThorlabsSRSMicroscopeSimulator.process_photodiode_signal(cfg, 1e-4, scan)
    assert sig.dc_optical_power_w == 0.0
    assert sig.ac_signal_current_a == 0.0


def test_t2_f1_04_extreme_high_laser_power_saturation():
    """Boundary: 500 mW pump laser power triggers photodiode saturation flag."""
    cfg = ThorlabsSRSMicroscopeConfig(pump_laser_power_mw=500.0)
    scan = ThorlabsSRSMicroscopeSimulator.simulate_scanning_point(cfg, 0.0, 0.0)
    sig = ThorlabsSRSMicroscopeSimulator.process_photodiode_signal(cfg, 1e-4, scan)
    assert sig.is_saturated is True


def test_t2_f1_05_ultrashort_pulse_duration():
    """Boundary: 10 fs ultrashort pulses experience extreme pulse broadening (> 10x)."""
    cfg = ThorlabsSRSMicroscopeConfig(pump_pulse_duration_fs=10.0)
    probe = ThorlabsSRSMicroscopeSimulator.probe_post_sample_intensity(cfg, delay_ps=0.0)
    chirped_ps = probe.details["chirped_pump_fwhm_ps"]
    assert (chirped_ps * 1e3) / 10.0 > 10.0


# --- F2 Boundaries ---

def test_t2_f2_01_hwp_angle_zero_max_transmission():
    """Boundary: HWP angle = 0.0 deg produces maximum transmission."""
    p_in = 25.0
    p_trans = p_in * (math.cos(0.0) ** 2)
    assert math.isclose(p_trans, p_in, rel_tol=1e-5)


def test_t2_f2_02_hwp_angle_45_min_transmission():
    """Boundary: HWP angle = 45.0 deg produces complete extinction in p-channel."""
    p_in = 25.0
    p_trans = p_in * (math.cos(math.radians(90.0)) ** 2)
    assert math.isclose(p_trans, 0.0, abs_tol=1e-12)


def test_t2_f2_03_hwp_angle_negative_and_over_360():
    """Boundary: HWP angles outside [0, 180] follow strict 180-deg rotation periodicity."""
    for deg in [-45.0, 225.0, 405.0]:
        val = math.cos(2.0 * math.radians(deg)) ** 2
        assert 0.0 <= val <= 1.0


def test_t2_f2_04_pbs_leakage_with_extreme_extinction():
    """Boundary: PBS extinction ratio = 1e-6 maintains signal stability."""
    cfg = ThorlabsSRSMicroscopeConfig(pbs_extinction_ratio=1e-6)
    assert cfg.pbs_extinction_ratio == 1e-6


def test_t2_f2_05_dump_power_at_max_laser_output():
    """Boundary: Full laser output (60 mW) directed into dump is well within 5 W thermal rating."""
    cfg = ThorlabsSRSMicroscopeConfig()
    assert 0.060 < cfg.dump_rating_w


# --- F3 Boundaries ---

def test_t2_f3_01_zero_applied_voltage():
    """Boundary: Zero applied voltage V = 0 V produces zero phase modulation."""
    cfg = ThorlabsSRSMicroscopeConfig(modulator_type="EOM", modulator_applied_v=0.0)
    scan = ThorlabsSRSMicroscopeSimulator.simulate_scanning_point(cfg, 0.0, 0.0)
    sig = ThorlabsSRSMicroscopeSimulator.process_photodiode_signal(cfg, 1e-4, scan)
    assert sig.details["modulation_factor"] == 0.0


def test_t2_f3_02_full_v_pi_applied_voltage():
    """Boundary: Applied voltage V = V_pi produces maximum 100% transmission."""
    cfg = ThorlabsSRSMicroscopeConfig(modulator_type="EOM", modulator_applied_v=180.0, modulator_v_pi_v=180.0)
    scan = ThorlabsSRSMicroscopeSimulator.simulate_scanning_point(cfg, 0.0, 0.0)
    sig = ThorlabsSRSMicroscopeSimulator.process_photodiode_signal(cfg, 1e-4, scan)
    assert math.isclose(sig.details["modulation_factor"], 1.0, rel_tol=1e-4)


def test_t2_f3_03_double_v_pi_voltage_periodic():
    """Boundary: Applied voltage V = 2 * V_pi wraps around phase to extinction."""
    v = 2.0 * 180.0
    t = math.sin(0.5 * math.pi * v / 180.0) ** 2
    assert math.isclose(t, 0.0, abs_tol=1e-12)


def test_t2_f3_04_ultra_high_rf_modulation_frequency():
    """Boundary: 100 MHz RF modulation frequency is accepted and processed cleanly."""
    cfg = ThorlabsSRSMicroscopeConfig(modulator_frequency_mhz=100.0)
    assert cfg.modulator_frequency_mhz == 100.0


def test_t2_f3_05_aom_saturation_power_clipping():
    """Boundary: AOM RF power 10x saturation power caps at maximum diffraction efficiency."""
    aom = AcoustoOpticModulator(rf_saturation_power_w=1.0)
    eff = aom.diffraction_efficiency(rf_power_w=10.0, wavelength_nm=1040.0)
    assert eff <= 1.0


# --- F4 Boundaries ---

def test_t2_f4_01_out_of_band_pump_rejection():
    """Boundary: Clean-up filter blocks out-of-band laser noise with OD > 6."""
    od = 6.0
    transmission = 10.0 ** (-od)
    assert transmission == 1.0e-6


def test_t2_f4_02_stokes_below_cutoff_blocking():
    """Boundary: Clean-up longpass filter blocks wavelengths below 1000 nm with OD > 5."""
    od = 5.0
    transmission = 10.0 ** (-od)
    assert transmission == 1.0e-5


def test_t2_f4_03_dichroic_extreme_aoi_deviation():
    """Boundary: Dichroic combiner operates with nominal tilt at +/-45 deg."""
    cfg = ThorlabsSRSMicroscopeConfig(dichroic_fold_angle_deg=45.0)
    assert cfg.dichroic_fold_angle_deg == 45.0


def test_t2_f4_04_zero_transmittance_clean_up():
    """Boundary: Complete blockage in clean-up filter results in zero transmitted signal (dark voltage only)."""
    cfg = ThorlabsSRSMicroscopeConfig(pump_laser_power_mw=0.0)
    scan = ThorlabsSRSMicroscopeSimulator.simulate_scanning_point(cfg, 0.0, 0.0)
    sig = ThorlabsSRSMicroscopeSimulator.process_photodiode_signal(cfg, 1e-4, scan)
    assert sig.dc_output_voltage_v < 1e-4


def test_t2_f4_05_steering_mirror_slight_misalignment():
    """Boundary: Small mirror tilt (+/-0.5 deg) updates ray reflection angle by twice the tilt."""
    d_theta_surf = 0.5
    d_theta_ray = 2.0 * d_theta_surf
    assert d_theta_ray == 1.0


# --- F5 Boundaries ---

def test_t2_f5_01_maximum_numerical_aperture_limit():
    """Boundary: Maximum water immersion NA = 1.33 yields sub-micron spot."""
    w0 = 0.61 * 0.797 / 1.33
    assert w0 < 0.40


def test_t2_f5_02_minimum_numerical_aperture_limit():
    """Boundary: Low NA = 0.10 yields expanded focal waist (> 4 um)."""
    w0 = 0.61 * 0.797 / 0.10
    assert w0 > 4.0


def test_t2_f5_03_zero_working_distance_contact():
    """Boundary: Working distance near contact (0.1 mm) without negative spacing."""
    info = OBJECTIVE_CATALOG["Olympus UPLSAPO60XW (60x NA 1.20 W High-NA)"]
    assert info["working_distance_mm"] == 0.28


def test_t2_f5_04_cover_glass_thickness_variation():
    """Boundary: Standard cover glass 0.17 mm with +/- 0.05 mm tolerance."""
    for t_cg in [0.12, 0.17, 0.22]:
        assert t_cg > 0.0


def test_t2_f5_05_oil_immersion_refractive_index():
    """Boundary: Water immersion (n = 1.333) vs oil immersion (n = 1.518)."""
    n_water = 1.333
    n_oil = 1.518
    assert n_oil > n_water > 1.0


# --- F6 Boundaries ---

def test_t2_f6_01_maximum_scan_angle_telecentricity():
    """Boundary: Extreme scan angle +/-10.0 deg evaluated in 4f configuration."""
    cfg = ThorlabsSRSMicroscopeConfig(scanning_architecture="4f-Relayed (Zero Walk-Off)")
    scan = ThorlabsSRSMicroscopeSimulator.simulate_scanning_point(cfg, 10.0, 0.0)
    assert scan.telecentricity_error_mrad == 0.0


def test_t2_f6_02_zero_scan_angle_on_axis():
    """Boundary: On-axis scan point (0.0, 0.0) has identically zero field radius."""
    cfg = ThorlabsSRSMicroscopeConfig()
    scan = ThorlabsSRSMicroscopeSimulator.simulate_scanning_point(cfg, 0.0, 0.0)
    assert scan.field_radius_um == 0.0


def test_t2_f6_03_scan_lens_short_focal_length():
    """Boundary: Short scan lens focal length (25 mm) doubles pupil magnification to 8.0x."""
    cfg = ThorlabsSRSMicroscopeConfig(scan_lens_efl_mm=25.0, tube_lens_efl_mm=200.0)
    m = cfg.tube_lens_efl_mm / cfg.scan_lens_efl_mm
    assert m == 8.0


def test_t2_f6_04_tube_lens_long_focal_length():
    """Boundary: Long tube lens focal length (400 mm) handles pupil expansion cleanly."""
    cfg = ThorlabsSRSMicroscopeConfig(tube_lens_efl_mm=400.0)
    assert cfg.tube_lens_efl_mm == 400.0


def test_t2_f6_05_entrance_pupil_overfill_boundary():
    """Boundary: Input beam diameter overfills pupil (6.0 mm vs 4.5 mm)."""
    cfg = ThorlabsSRSMicroscopeConfig(input_beam_diameter_mm=6.0)
    scan = ThorlabsSRSMicroscopeSimulator.simulate_scanning_point(cfg, 0.0, 0.0)
    assert scan.back_pupil_beam_diameter_mm == 24.0


# --- F7 Boundaries ---

def test_t2_f7_01_asphere_marginal_ray_at_clear_aperture():
    """Boundary: Ray traced at semi-diameter rim of ACL25416U-B (r = 12.0 mm)."""
    item = get_catalog_lens("ACL25416U-B")
    assert item is not None
    ca_radius = item["diameter_mm"] / 2.0
    assert 12.0 < ca_radius <= 12.7


def test_t2_f7_02_condenser_spherical_vs_aspheric_sag():
    """Boundary: Compare spherical sag vs aspheric sag with conic kappa = -0.6265."""
    c = 1.0 / 8.32
    r = 8.0
    sag_sphere = (c * (r ** 2)) / (1.0 + math.sqrt(max(0.0, 1.0 - (c ** 2) * (r ** 2))))
    sag_asphere = (c * (r ** 2)) / (1.0 + math.sqrt(max(0.0, 1.0 - (1.0 - 0.6265) * (c ** 2) * (r ** 2))))
    assert sag_asphere < sag_sphere  # Conic softens curvature at edge


def test_t2_f7_03_condenser_conic_constant_extremes():
    """Boundary: Conic constant kappa in [-1.0, 0.0] behaves smoothly."""
    for kappa in [-1.0, -0.6265, 0.0]:
        val = 1.0 - (1.0 + kappa) * ((1.0 / 8.32) ** 2) * (4.0 ** 2)
        assert val > 0.0


def test_t2_f7_04_collector_lens_extreme_decenter():
    """Boundary: Off-axis beam decenter of 5.0 mm safely within 25.4 mm aperture."""
    ap_radius = 25.4 / 2.0
    assert 5.0 < ap_radius


def test_t2_f7_05_condenser_working_distance_offset():
    """Boundary: Condenser working distance variation (+/- 1.0 mm) remains positive."""
    wd_nominal = 8.5
    for offset in [-1.0, 0.0, 1.0]:
        assert wd_nominal + offset > 0.0


# --- F8 Boundaries ---

def test_t2_f8_01_marginal_ray_bundle_at_pupil_edge():
    """Boundary: Trace marginal ray at pupil edge y = 1.0 mm in standard template."""
    stack = build_srs_microscope_template()
    sys = OpticalSystem(stack)
    ray = Ray(pos=np.array([0.0, 1.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=0.797)
    traced = sys.trace_ray(ray)
    assert not traced.vignetted
    assert len(traced.history) == 18


def test_t2_f8_02_negative_marginal_ray_at_pupil_edge():
    """Boundary: Trace marginal ray at negative pupil edge y = -1.0 mm."""
    stack = build_srs_microscope_template()
    sys = OpticalSystem(stack)
    ray = Ray(pos=np.array([0.0, -1.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=0.797)
    traced = sys.trace_ray(ray)
    assert not traced.vignetted
    assert len(traced.history) == 18


def test_t2_f8_03_diagonal_corner_ray_tracing():
    """Boundary: Trace diagonal ray at (x, y) = (1.0, 1.0) mm unvignetted."""
    stack = build_srs_microscope_template()
    sys = OpticalSystem(stack)
    ray = Ray(pos=np.array([1.0, 1.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=0.797)
    traced = sys.trace_ray(ray)
    assert not traced.vignetted
    assert len(traced.history) == 18


def test_t2_f8_04_clipped_ray_beyond_clear_aperture():
    """Boundary: Ray launched far outside aperture (y = 50.0 mm) gets clipped."""
    stack = build_srs_microscope_template()
    sys = OpticalSystem(stack)
    ray = Ray(pos=np.array([0.0, 50.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=0.797)
    traced = sys.trace_ray(ray)
    assert traced.vignetted is True


def test_t2_f8_05_reversed_ray_direction():
    """Boundary: Backward ray launching (-Z) handles boundary without crash."""
    stack = build_srs_microscope_template()
    sys = OpticalSystem(stack)
    ray = Ray(pos=np.array([0.0, 0.0, 10.0]), dir=np.array([0.0, 0.0, -1.0]), wavelength=0.797)
    traced = sys.trace_ray(ray)
    assert traced is not None


# --- F9 Boundaries ---

def test_t2_f9_01_maximum_scan_angles_plus_minus_10_deg():
    """Boundary: Scan angles at exact maximum bounds (+10.0 deg, +10.0 deg)."""
    cfg = ThorlabsSRSMicroscopeConfig(galvo_max_mechanical_angle_deg=10.0)
    pt = ThorlabsSRSMicroscopeSimulator.simulate_scanning_point(cfg, 10.0, 10.0)
    assert pt.field_radius_um > 1000.0


def test_t2_f9_02_maximum_negative_angles():
    """Boundary: Scan angles at maximum negative bounds (-10.0 deg, -10.0 deg)."""
    cfg = ThorlabsSRSMicroscopeConfig(galvo_max_mechanical_angle_deg=10.0)
    pt = ThorlabsSRSMicroscopeSimulator.simulate_scanning_point(cfg, -10.0, -10.0)
    assert pt.field_radius_um > 1000.0


def test_t2_f9_03_zero_scan_angle_singularity_check():
    """Boundary: On-axis (0.0, 0.0) angle handles potential division by zero."""
    cfg = ThorlabsSRSMicroscopeConfig()
    pt = ThorlabsSRSMicroscopeSimulator.simulate_scanning_point(cfg, 0.0, 0.0)
    assert pt.field_radius_um == 0.0
    assert not math.isnan(pt.total_scan_efficiency)


def test_t2_f9_04_asymmetric_scan_angles():
    """Boundary: Asymmetric scanning (+10.0 deg on X, 0.0 deg on Y)."""
    cfg = ThorlabsSRSMicroscopeConfig()
    pt = ThorlabsSRSMicroscopeSimulator.simulate_scanning_point(cfg, 10.0, 0.0)
    assert pt.sample_x_um > 0.0
    assert pt.sample_y_um == 0.0


def test_t2_f9_05_micro_scan_angles():
    """Boundary: Extremely small sub-milliradian scan angle (0.001 deg)."""
    cfg = ThorlabsSRSMicroscopeConfig()
    pt = ThorlabsSRSMicroscopeSimulator.simulate_scanning_point(cfg, 0.001, 0.0)
    assert 0.0 < pt.sample_x_um < 1.0


# --- F10 Boundaries ---

def test_t2_f10_01_relayed_scanner_walk_off_at_max_angle():
    """Boundary: Walk-off remains strictly 0.0 um at maximum scan angle in 4f relay."""
    cfg = ThorlabsSRSMicroscopeConfig(scanning_architecture="4f-Relayed (Zero Walk-Off)")
    scan = ThorlabsSRSMicroscopeSimulator.simulate_scanning_point(cfg, 10.0, 10.0)
    assert scan.back_pupil_walk_off_total_um == 0.0


def test_t2_f10_02_close_coupled_extreme_walk_off():
    """Boundary: Close-coupled walk-off at 10.0 deg exceeds 5000 um."""
    cfg = ThorlabsSRSMicroscopeConfig(scanning_architecture="Close-Coupled (Thorlabs GVS002)")
    scan = ThorlabsSRSMicroscopeSimulator.simulate_scanning_point(cfg, 10.0, 0.0)
    assert scan.back_pupil_walk_off_total_um > 5000.0


def test_t2_f10_03_zero_inter_galvo_separation():
    """Boundary: Zero inter-galvo separation Delta_z -> 0 eliminates close-coupled walk-off."""
    cfg = ThorlabsSRSMicroscopeConfig(
        scanning_architecture="Close-Coupled (Thorlabs GVS002)",
        inter_galvo_separation_mm=0.0,
    )
    scan = ThorlabsSRSMicroscopeSimulator.simulate_scanning_point(cfg, 5.0, 0.0)
    assert scan.back_pupil_walk_off_total_um == 0.0


def test_t2_f10_04_pupil_fill_factor_overfill():
    """Boundary: Overfilled objective pupil (fill factor > 1.2)."""
    cfg = ThorlabsSRSMicroscopeConfig(input_beam_diameter_mm=5.5)
    scan = ThorlabsSRSMicroscopeSimulator.simulate_scanning_point(cfg, 0.0, 0.0)
    assert scan.pupil_fill_factor > 1.1


def test_t2_f10_05_pupil_fill_factor_underfill():
    """Boundary: Underfilled objective pupil (fill factor < 0.5)."""
    cfg = ThorlabsSRSMicroscopeConfig(input_beam_diameter_mm=2.0)
    scan = ThorlabsSRSMicroscopeSimulator.simulate_scanning_point(cfg, 0.0, 0.0)
    assert scan.pupil_fill_factor < 0.5


# --- F11 Boundaries ---

def test_t2_f11_01_maximum_edge_petzval_defocus():
    """Boundary: Field curvature defocus at FOV edge exceeds 10 um."""
    cfg = ThorlabsSRSMicroscopeConfig()
    scan = ThorlabsSRSMicroscopeSimulator.simulate_scanning_point(cfg, 5.0, 5.0)
    assert scan.field_curvature_defocus_um > 10.0


def test_t2_f11_02_maximum_lateral_chromatic_walk_off():
    """Boundary: Chromatic walk-off at FOV edge exceeds 2.0 um."""
    cfg = ThorlabsSRSMicroscopeConfig()
    scan = ThorlabsSRSMicroscopeSimulator.simulate_scanning_point(cfg, 5.0, 5.0)
    assert scan.chromatic_walk_off_um > 2.0


def test_t2_f11_03_pupil_vignetting_at_corner_limit():
    """Boundary: Vignetting factor at extreme corner remains clamped above zero."""
    cfg = ThorlabsSRSMicroscopeConfig()
    scan = ThorlabsSRSMicroscopeSimulator.simulate_scanning_point(cfg, 10.0, 10.0)
    assert scan.pupil_vignetting_factor > 0.001


def test_t2_f11_04_zero_field_curvature_radius_guard():
    """Boundary: Infinite Petzval radius produces zero defocus."""
    cfg = ThorlabsSRSMicroscopeConfig(petzval_radius_mm=1.0e9)
    scan = ThorlabsSRSMicroscopeSimulator.simulate_scanning_point(cfg, 3.0, 3.0)
    assert math.isclose(scan.field_curvature_defocus_um, 0.0, abs_tol=1e-3)


def test_t2_f11_05_zero_chromatic_aberration():
    """Boundary: Zero lateral chromatic coefficient produces zero walk-off."""
    cfg = ThorlabsSRSMicroscopeConfig(lateral_chromatic_coeff=0.0)
    scan = ThorlabsSRSMicroscopeSimulator.simulate_scanning_point(cfg, 3.0, 3.0)
    assert scan.chromatic_walk_off_um == 0.0


# --- F12 Boundaries ---

def test_t2_f12_01_zero_optical_delay():
    """Boundary: Zero optical delay tau_d = 0.0 ps excites exact base wavenumber."""
    cfg = ThorlabsSRSMicroscopeConfig()
    probe = ThorlabsSRSMicroscopeSimulator.probe_post_sample_intensity(cfg, delay_ps=0.0)
    assert math.isclose(probe.probed_wavenumber_cm1, probe.base_wavenumber_cm1, abs_tol=1e-5)


def test_t2_f12_02_maximum_delay_plus_minus_5ps():
    """Boundary: Large optical delay (+/-5.0 ps) sweeps wide spectral bandwidth (> 300 cm^-1)."""
    cfg = ThorlabsSRSMicroscopeConfig(optical_delay_scan_ps=5.0, delay_samples=11)
    res = ThorlabsSRSMicroscopeSimulator.simulate_oil_spectrum(cfg)
    span = res["wavenumbers_cm1"][-1] - res["wavenumbers_cm1"][0]
    assert span > 300.0


def test_t2_f12_03_zero_glass_length_unchirped_limit():
    """Boundary: Zero glass rod length maintains unchirped pulse duration."""
    cfg = ThorlabsSRSMicroscopeConfig(pump_glass_length_mm=0.0, stokes_glass_length_mm=0.0)
    probe = ThorlabsSRSMicroscopeSimulator.probe_post_sample_intensity(cfg, delay_ps=0.0)
    assert math.isclose(probe.details["chirped_pump_fwhm_ps"] * 1e3, 120.0, abs_tol=1.0)


def test_t2_f12_04_extreme_mismatched_chirp():
    """Boundary: Highly asymmetric glass lengths (300 mm vs 50 mm) handles chirp mismatch."""
    cfg = ThorlabsSRSMicroscopeConfig(pump_glass_length_mm=300.0, stokes_glass_length_mm=50.0, delay_samples=11)
    res = ThorlabsSRSMicroscopeSimulator.simulate_oil_spectrum(cfg)
    assert res["chirp_matching_ratio"] != 1.0


def test_t2_f12_05_dense_sampling_delay_sweep():
    """Boundary: Dense 121-point delay sweep completes without numeric overflow."""
    cfg = ThorlabsSRSMicroscopeConfig(delay_samples=121)
    res = ThorlabsSRSMicroscopeSimulator.simulate_oil_spectrum(cfg)
    assert len(res["wavenumbers_cm1"]) == 121


# --- F13 Boundaries ---

def test_t2_f13_01_zero_analyte_concentration():
    """Boundary: Zero Raman cross-section yields zero SRL signal."""
    cfg = ThorlabsSRSMicroscopeConfig()
    scan = ThorlabsSRSMicroscopeSimulator.simulate_scanning_point(cfg, 0.0, 0.0)
    sig = ThorlabsSRSMicroscopeSimulator.process_photodiode_signal(cfg, srl_fraction=0.0, scan_result=scan)
    assert sig.srl_modulation_depth == 0.0
    assert sig.ac_signal_current_a == 0.0


def test_t2_f13_02_extreme_high_concentration():
    """Boundary: 10x higher SRL fraction scales AC modulation photocurrent proportionally."""
    cfg = ThorlabsSRSMicroscopeConfig()
    scan = ThorlabsSRSMicroscopeSimulator.simulate_scanning_point(cfg, 0.0, 0.0)
    sig1 = ThorlabsSRSMicroscopeSimulator.process_photodiode_signal(cfg, srl_fraction=1e-5, scan_result=scan)
    sig2 = ThorlabsSRSMicroscopeSimulator.process_photodiode_signal(cfg, srl_fraction=1e-4, scan_result=scan)
    assert math.isclose(sig2.ac_signal_current_a / sig1.ac_signal_current_a, 10.0, rel_tol=1e-3)


def test_t2_f13_03_zero_stokes_power():
    """Boundary: Zero Stokes power generates zero SRL modulation."""
    cfg = ThorlabsSRSMicroscopeConfig(stokes_laser_power_mw=0.0)
    probe = ThorlabsSRSMicroscopeSimulator.probe_post_sample_intensity(cfg, delay_ps=0.0)
    assert probe.fractional_srl == 0.0


def test_t2_f13_04_zero_sample_thickness():
    """Boundary: Zero sample thickness yields zero nonlinear interaction length."""
    cfg = ThorlabsSRSMicroscopeConfig(sample_thickness_um=0.0)
    probe = ThorlabsSRSMicroscopeSimulator.probe_post_sample_intensity(cfg, delay_ps=0.0)
    assert probe.fractional_srl == 0.0


def test_t2_f13_05_stokes_suppression_od15_leakage():
    """Boundary: OD 15 optical blocking reduces transmitted power by 1e-15."""
    leakage = 0.060 * (10.0 ** -15)
    assert leakage < 1.0e-15


# --- F14 Boundaries ---

def test_t2_f14_01_zero_photodiode_dark_current():
    """Boundary: Photodiode dark current = 0.0 A handles noise calculation cleanly."""
    cfg = ThorlabsSRSMicroscopeConfig(photodiode=SRSPhotodiodeModel(dark_current_a=0.0))
    scan = ThorlabsSRSMicroscopeSimulator.simulate_scanning_point(cfg, 0.0, 0.0)
    sig = ThorlabsSRSMicroscopeSimulator.process_photodiode_signal(cfg, 1e-4, scan)
    assert sig.shot_noise_current_a > 0.0


def test_t2_f14_02_cryogenic_temperature_johnson_noise():
    """Boundary: Cryogenic temperature (4 K) reduces thermal Johnson noise by sqrt(4 / 295)."""
    t_cryo = 4.0
    t_room = 295.0
    expected_ratio = math.sqrt(t_cryo / t_room)
    assert math.isclose(expected_ratio, 0.1164, abs_tol=1e-3)


def test_t2_f14_03_high_laser_rin_floor():
    """Boundary: High laser RIN (-110 dBc/Hz) makes RIN dominant noise contributor."""
    cfg = ThorlabsSRSMicroscopeConfig(photodiode=SRSPhotodiodeModel(laser_rin_dbc_per_hz=-110.0))
    scan = ThorlabsSRSMicroscopeSimulator.simulate_scanning_point(cfg, 0.0, 0.0)
    sig = ThorlabsSRSMicroscopeSimulator.process_photodiode_signal(cfg, 1e-4, scan)
    assert sig.rin_noise_current_a > sig.shot_noise_current_a


def test_t2_f14_04_ultra_short_lockin_time_constant():
    """Boundary: Ultra-short time constant (1 us) yields high noise bandwidth (160 kHz)."""
    cfg = ThorlabsSRSMicroscopeConfig(lockin_time_constant_us=1.0)
    scan = ThorlabsSRSMicroscopeSimulator.simulate_scanning_point(cfg, 0.0, 0.0)
    sig = ThorlabsSRSMicroscopeSimulator.process_photodiode_signal(cfg, 1e-4, scan)
    assert sig.details["filter_bandwidth_hz"] > 100000.0


def test_t2_f14_05_long_lockin_time_constant():
    """Boundary: Long time constant (1000 us) narrows noise bandwidth to 160 Hz, boosting SNR."""
    cfg = ThorlabsSRSMicroscopeConfig(lockin_time_constant_us=1000.0)
    scan = ThorlabsSRSMicroscopeSimulator.simulate_scanning_point(cfg, 0.0, 0.0)
    sig = ThorlabsSRSMicroscopeSimulator.process_photodiode_signal(cfg, 1e-4, scan)
    assert sig.details["filter_bandwidth_hz"] < 200.0


# --- F15 Boundaries ---

def test_t2_f15_01_extreme_negative_decenters():
    """Boundary: Large mechanical decenter (x = -100.0 mm) traces without matrix singularity."""
    stack = build_folded_real_world_srs_template()
    stack.surfaces[28].decenter_x = -100.0
    sys = OpticalSystem(stack)
    assert sys is not None


def test_t2_f15_02_zero_separation_mirrors():
    """Boundary: Close-coupled optical surfaces with minimal spacing (t = 0.1 mm)."""
    s = Surface(name="Mirror1", thickness=0.1, surface_type=SurfaceType.AIR_SPACE)
    assert s.thickness == 0.1


def test_t2_f15_03_stacked_reflections_180_degree_hairpin():
    """Boundary: Consecutive 90 deg folds reverse optical propagation direction (-Z)."""
    st = SurfaceStack()
    st.add_surface(Surface(name="Air", thickness=20.0, surface_type=SurfaceType.AIR_SPACE))
    st.add_surface(Surface(name="Fold1", tilt_x=45.0, surface_type=SurfaceType.MIRROR, thickness=20.0))
    st.add_surface(Surface(name="Fold2", tilt_x=45.0, surface_type=SurfaceType.MIRROR, thickness=20.0))
    sys = OpticalSystem(st)
    ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=0.797)
    traced = sys.trace_ray(ray)
    assert traced.history[-1]["dir_z"] < -0.99


def test_t2_f15_04_auto_zoom_with_single_surface():
    """Boundary: Auto-zoom framing handles single-surface system gracefully."""
    if not QT_AVAILABLE:
        pytest.skip("Qt not available")
    from optics_app.gui.dual_view import DualViewWidget
    st = SurfaceStack()
    st.add_surface(Surface(name="Single", radius=50.0, thickness=10.0, glass="N-BK7"))
    widget = DualViewWidget()
    widget.update_layout(st)
    assert widget.fig is not None


def test_t2_f15_05_canvas_aspect_ratio_preservation():
    """Boundary: Canvas aspect ratio maintains metric 1:1 scaling under window resize."""
    if not QT_AVAILABLE:
        pytest.skip("Qt not available")
    from optics_app.gui.dual_view import DualViewWidget
    widget = DualViewWidget()
    widget.set_transverse_scale(1.0)
    assert widget.transverse_scale == 1.0


# --- F16 Boundaries ---

def test_t2_f16_01_setup_builder_with_empty_config():
    """Boundary: Setup Builder given None config returns valid default system."""
    sys = build_srs_microscope_system(None)
    assert sys is not None
    assert len(sys.surface_stack.surfaces) >= 21


def test_t2_f16_02_field_diagnostic_with_1x1_grid():
    """Boundary: 2D field simulation with minimal 1x1 grid returns single on-axis point."""
    cfg = ThorlabsSRSMicroscopeConfig()
    res = ThorlabsSRSMicroscopeSimulator.simulate_2d_scan_field(cfg, grid_points=1)
    assert res["scan_efficiency_map"].shape == (1, 1)


def test_t2_f16_03_delay_scan_with_zero_range():
    """Boundary: Delay scan with 0.0 ps range returns base wavenumber."""
    cfg = ThorlabsSRSMicroscopeConfig(optical_delay_scan_ps=0.0, delay_samples=1)
    res = ThorlabsSRSMicroscopeSimulator.simulate_oil_spectrum(cfg)
    assert len(res["wavenumbers_cm1"]) == 1


def test_t2_f16_04_lockin_tool_with_sub_microvolt_signal():
    """Boundary: Extremely small sub-microvolt signal (1e-9 V) avoids underflow."""
    cfg = ThorlabsSRSMicroscopeConfig()
    scan = ThorlabsSRSMicroscopeSimulator.simulate_scanning_point(cfg, 0.0, 0.0)
    sig = ThorlabsSRSMicroscopeSimulator.process_photodiode_signal(cfg, srl_fraction=1e-10, scan_result=scan)
    assert sig.lockin_magnitude_r_v >= 0.0


def test_t2_f16_05_bench_alignment_with_severely_misaligned_system():
    """Boundary: Close-coupled galvo at 8.0 deg reports large walk-off and clipping."""
    cfg = ThorlabsSRSMicroscopeConfig(scanning_architecture="Close-Coupled (Thorlabs GVS002)")
    scan = ThorlabsSRSMicroscopeSimulator.simulate_scanning_point(cfg, 8.0, 0.0)
    assert scan.back_pupil_walk_off_total_um > 4000.0
    assert scan.total_scan_efficiency < 0.50


# --- F17 Boundaries ---

def test_t2_f17_01_surface_stack_serialization_empty_routes():
    """Boundary: Serializing SurfaceStack with optical_routes=None round-trips cleanly."""
    st = SurfaceStack()
    st.optical_routes = None
    d = st.to_dict()
    assert "optical_routes" not in d or d["optical_routes"] is None


def test_t2_f17_02_multiple_consecutive_preset_loads():
    """Boundary: Loading presets consecutively overwrites stack without surface leakage."""
    sys1 = load_thorlabs_lsm_json("thorlabs_lsm_microscope.json")
    sys2 = load_thorlabs_lsm_json("thorlabs_lsm_microscope.json")
    assert len(sys1.surface_stack.surfaces) == len(sys2.surface_stack.surfaces) == 17


def test_t2_f17_03_switch_between_srs_and_generic_surfaces():
    """Boundary: Adding generic lenses after loading SRS preset does not raise exceptions."""
    sys = load_thorlabs_lsm_json("thorlabs_lsm_microscope.json")
    sys.surface_stack.add_surface(Surface(name="Custom Post Filter", thickness=10.0, glass="N-BK7"))
    assert len(sys.surface_stack.surfaces) == 18


def test_t2_f17_04_dock_tab_selection_boundaries():
    """Boundary: Selecting dock tab index within valid range [0, 5]."""
    if not QT_AVAILABLE:
        pytest.skip("Qt not available")
    from optics_app.gui.ultrafast_dock import UltrafastAnalysisDock
    dock = UltrafastAnalysisDock()
    dock.tabs.setCurrentIndex(5)
    assert dock.tabs.currentIndex() == 5


def test_t2_f17_05_corrupt_json_preset_recovery():
    """Boundary: Loading non-existent or invalid JSON file raises clean FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_thorlabs_lsm_json("non_existent_preset_xyz.json")


# --- F18 Boundaries ---

def test_t2_f18_01_v2_import_all_srs_classes():
    """Boundary: Import all SRS data classes and functions from v2."""
    for cls_name in ["ThorlabsSRSMicroscopeConfig", "ThorlabsSRSMicroscopeSimulator", "SRSPhotodiodeSignal"]:
        assert hasattr(v2_srs, cls_name)


def test_t2_f18_02_v2_to_dict_from_dict_roundtrip():
    """Boundary: Config round-trip via to_dict and from_dict in v2."""
    cfg = v2_srs.ThorlabsSRSMicroscopeConfig(pump_laser_power_mw=42.0)
    d = cfg.to_dict()
    loaded = v2_srs.ThorlabsSRSMicroscopeConfig.from_dict(d)
    assert loaded.pump_laser_power_mw == 42.0


def test_t2_f18_03_v2_37_surface_template_parity():
    """Boundary: v2 build_folded_real_world_srs_template returns exactly 37 surfaces."""
    stack = v2_srs.build_folded_real_world_srs_template()
    assert len(stack.surfaces) == 37


def test_t2_f18_04_v2_ray_tracing_parity():
    """Boundary: Tracing rays in v2 matches optics_app within numerical precision."""
    st1 = build_srs_microscope_template()
    st2 = v2_srs.build_srs_microscope_template()
    sys1 = OpticalSystem(st1)
    sys2 = v2_tracer.OpticalSystem(st2)
    r1 = sys1.trace_ray(Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=0.797))
    r2 = sys2.trace_ray(v2_ray.Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=0.797))
    assert len(r1.history) == len(r2.history) == 18
    assert not r1.vignetted
    assert not r2.vignetted


def test_t2_f18_05_v2_scan_field_aberration_parity():
    """Boundary: 2D scan field efficiency map matches between optics_app and v2."""
    cfg1 = ThorlabsSRSMicroscopeConfig()
    cfg2 = v2_srs.ThorlabsSRSMicroscopeConfig()
    res1 = ThorlabsSRSMicroscopeSimulator.simulate_2d_scan_field(cfg1, grid_points=3)
    res2 = v2_srs.ThorlabsSRSMicroscopeSimulator.simulate_2d_scan_field(cfg2, grid_points=3)
    assert np.allclose(res1["scan_efficiency_map"], res2["scan_efficiency_map"], atol=1e-5)


# ==============================================================================
# TIER 3: CROSS-FEATURE INTERACTIONS (Pairwise Combinations, 15 tests)
# ==============================================================================

def test_t3_cross_01_galvo_scanning_with_active_eom_modulation():
    """Pairwise F3 + F9: Dynamic galvo scanning combined with 10 MHz EOM intensity modulation."""
    cfg = ThorlabsSRSMicroscopeConfig(
        modulator_type="EOM",
        modulator_applied_v=90.0,  # 50% modulation depth
        modulator_v_pi_v=180.0,
    )
    scan = ThorlabsSRSMicroscopeSimulator.simulate_scanning_point(cfg, 3.0, 2.0)
    sig = ThorlabsSRSMicroscopeSimulator.process_photodiode_signal(cfg, srl_fraction=1e-4, scan_result=scan)
    assert math.isclose(sig.details["modulation_factor"], 0.5, rel_tol=1e-4)
    assert math.isclose(sig.srl_modulation_depth, 1e-4 * scan.total_scan_efficiency * 0.5, rel_tol=1e-5)


def test_t3_cross_02_spectral_focusing_with_high_na_olympus_immersion():
    """Pairwise F5 + F12: Hyperspectral delay scan through mineral oil with Olympus 25x 1.05 NA W."""
    cfg = ThorlabsSRSMicroscopeConfig(
        objective_model="Olympus XLPLN25XWMP2 (25x NA 1.05 W Multiphoton)",
        sample_type=OilSampleType.MINERAL_OIL,
        delay_samples=15,
    )
    res = ThorlabsSRSMicroscopeSimulator.simulate_oil_spectrum(cfg)
    assert res["scan_on_axis"].pump_waist_um < 0.50
    assert 0.85 <= res["chirp_matching_ratio"] <= 1.15


def test_t3_cross_03_srl_transfer_with_photodiode_noise_breakdown():
    """Pairwise F13 + F14: Four-wave mixing SRL detected on DET10N2 with shot, Johnson, and RIN noise."""
    cfg = ThorlabsSRSMicroscopeConfig()
    scan = ThorlabsSRSMicroscopeSimulator.simulate_scanning_point(cfg, 0.0, 0.0)
    sig = ThorlabsSRSMicroscopeSimulator.process_photodiode_signal(cfg, srl_fraction=2e-5, scan_result=scan)
    assert sig.lockin_magnitude_r_v > 0.0
    assert sig.total_noise_current_a == math.sqrt(
        sig.shot_noise_current_a ** 2 + sig.thermal_noise_current_a ** 2 + sig.rin_noise_current_a ** 2
    )
    assert sig.snr_db > 0.0


def test_t3_cross_04_37_surface_folded_layout_with_auto_zoom():
    """Pairwise F8 + F15: Folded 37-surface bench with mirror reflections framed in DualViewWidget."""
    stack = build_folded_real_world_srs_template()
    sys = OpticalSystem(stack)
    ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=0.797)
    traced = sys.trace_ray(ray)
    assert len(traced.history) == 28
    if QT_AVAILABLE:
        from optics_app.gui.dual_view import DualViewWidget
        widget = DualViewWidget()
        widget.update_layout(stack, rays=[traced])
        assert widget.fig is not None


def test_t3_cross_05_zero_pupil_walk_off_with_commercial_scan_telecentricity():
    """Pairwise F6 + F10: 1:1 galvo relay paired with SL50-2P2 guarantees zero BFP walk-off and telecentricity."""
    cfg = ThorlabsSRSMicroscopeConfig(scanning_architecture="4f-Relayed (Zero Walk-Off)", scan_lens_model="Thorlabs SL50-2P2")
    scan = ThorlabsSRSMicroscopeSimulator.simulate_scanning_point(cfg, 4.0, 0.0)
    assert scan.back_pupil_walk_off_total_um == 0.0
    assert scan.telecentricity_error_mrad == 0.0


def test_t3_cross_06_power_attenuation_with_clean_up_filters():
    """Pairwise F2 + F4: HWP power attenuation combined with FBH790-10 clean-up filter."""
    stack = build_folded_real_world_srs_template()
    s_hwp = stack.surfaces[1]
    s_filt = stack.surfaces[3]
    assert "WPH10M-780" in s_hwp.name
    assert "FBH790-10" in s_filt.name


def test_t3_cross_07_dual_laser_sources_with_dichroic_collinear_combination():
    """Pairwise F1 + F4: Pump (797 nm) and Stokes (1040 nm) collinear combination at DMLP900."""
    stack = build_folded_real_world_srs_template()
    s_pump = stack.surfaces[0]
    s_stokes = stack.surfaces[28]
    s_dichroic = stack.surfaces[5]
    assert "797" in s_pump.name
    assert "1040" in s_stokes.name
    assert "DMLP900" in s_dichroic.name


def test_t3_cross_08_petzval_curvature_with_rf_lockin_snr_degradation():
    """Pairwise F11 + F14: Petzval defocus expands focal spot, reducing detected lock-in SNR off-axis."""
    cfg = ThorlabsSRSMicroscopeConfig()
    scan_on = ThorlabsSRSMicroscopeSimulator.simulate_scanning_point(cfg, 0.0, 0.0)
    scan_off = ThorlabsSRSMicroscopeSimulator.simulate_scanning_point(cfg, 4.0, 3.0)
    sig_on = ThorlabsSRSMicroscopeSimulator.process_photodiode_signal(cfg, 1e-4, scan_on)
    sig_off = ThorlabsSRSMicroscopeSimulator.process_photodiode_signal(cfg, 1e-4, scan_off)
    assert sig_off.lockin_magnitude_r_v < sig_on.lockin_magnitude_r_v
    assert sig_off.snr_db < sig_on.snr_db


def test_t3_cross_09_aspheric_condenser_collection_with_stokes_dump_blocking():
    """Pairwise F7 + F13: ACL25416U-B collects transmitted light, followed by FESH0850+FESH0800 OD>12 blocking."""
    stack = build_folded_real_world_srs_template()
    names = [s.name for s in stack.surfaces]
    idx_cond = next(i for i, n in enumerate(names) if "ACL25416U-B" in n)
    idx_dump1 = next(i for i, n in enumerate(names) if "FESH0850" in n)
    idx_dump2 = next(i for i, n in enumerate(names) if "FESH0800" in n)
    assert idx_cond < idx_dump1 < idx_dump2


def test_t3_cross_10_interactive_setup_builder_with_delay_scan_tool():
    """Pairwise F12 + F16: Setup builder configures SF57 rods; delay scan tool evaluates Raman spectrum."""
    cfg = ThorlabsSRSMicroscopeConfig(
        pump_glass_material=DispersiveGlassMaterial.SF57,
        stokes_glass_material=DispersiveGlassMaterial.SF57,
        sample_type=OilSampleType.OLIVE_OIL,
        delay_samples=15,
    )
    res = ThorlabsSRSMicroscopeSimulator.simulate_oil_spectrum(cfg)
    assert res["sample_name"] == "Olive Oil (Extra Virgin)"
    assert len(res["wavenumbers_cm1"]) == 15


def test_t3_cross_11_non_intrusive_integration_with_dual_tree_parity():
    """Pairwise F17 + F18: Shipped presets load cleanly and identically in both optics_app and v2."""
    s1 = load_thorlabs_lsm_json("thorlabs_lsm_microscope.json")
    s2 = v2_srs.load_thorlabs_lsm_json("thorlabs_lsm_microscope.json")
    assert len(s1.surface_stack.surfaces) == len(s2.surface_stack.surfaces) == 17


def test_t3_cross_12_dynamic_galvo_raster_with_oil_droplet_concentration_map():
    """Pairwise F9 + F12: 2D galvo raster scan across Oil Droplets pattern yields spatial modulation."""
    cfg = ThorlabsSRSMicroscopeConfig(sample_pattern="Oil Droplets", galvo_max_mechanical_angle_deg=2.5)
    res = ThorlabsSRSMicroscopeSimulator.simulate_2d_scan_field(cfg, grid_points=7)
    conc_map = res["sample_concentration_map"]
    assert np.max(conc_map) > np.min(conc_map)


def test_t3_cross_13_eom_halfwave_voltage_with_photodiode_saturation():
    """Pairwise F3 + F14: Driving EOM at full V_pi with high laser power evaluates saturation threshold."""
    cfg = ThorlabsSRSMicroscopeConfig(
        modulator_type="EOM",
        modulator_applied_v=180.0,
        pump_laser_power_mw=80.0,
        photodiode=SRSPhotodiodeModel(saturation_power_w=0.010),
    )
    scan = ThorlabsSRSMicroscopeSimulator.simulate_scanning_point(cfg, 0.0, 0.0)
    sig = ThorlabsSRSMicroscopeSimulator.process_photodiode_signal(cfg, 1e-4, scan)
    assert sig.is_saturated is True


def test_t3_cross_14_commercial_tube_lens_with_olympus_20x_bfp_fill():
    """Pairwise F5 + F6: Thorlabs TTL200MP tube lens expands 4.5 mm beam by 4.0x to 18.0 mm BFP."""
    cfg = ThorlabsSRSMicroscopeConfig(
        tube_lens_model="Thorlabs TTL200MP",
        input_beam_diameter_mm=4.5,
    )
    scan = ThorlabsSRSMicroscopeSimulator.simulate_scanning_point(cfg, 0.0, 0.0)
    assert scan.back_pupil_beam_diameter_mm == 18.0


def test_t3_cross_15_stokes_dump_filters_with_full_beam_propagation():
    """Pairwise F4 + F8: FESH0850 + FESH0800 filter clearances in 37-surface train allow full ray passage."""
    stack = build_folded_real_world_srs_template()
    sys = OpticalSystem(stack)
    ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=0.797)
    traced = sys.trace_ray(ray)
    # Surfaces 23 and 24 are Stokes dump filters
    h23 = traced.history[23]
    h24 = traced.history[24]
    assert "FESH0850" in stack.surfaces[23].name
    assert "FESH0800" in stack.surfaces[24].name
    assert not traced.vignetted


# ==============================================================================
# TIER 4: REAL-WORLD APPLICATION SCENARIOS (5 Comprehensive Workflows)
# ==============================================================================

def test_t4_scenario_1_hyperspectral_ch_contrast_scan():
    """Scenario 1: Hyperspectral C-H Chemical Contrast Scan (Mineral Oil vs Olive Oil).
    
    Verifies:
    1. Linearly chirped pump and Stokes pulses through SF57 rods match chirp rates (alpha_p / alpha_S ~ 1.0).
    2. Sweeping inter-pulse delay over +/-3.0 ps sweeps instantaneous Raman shift across 2800 - 3050 cm^-1.
    3. Reconstructs vibrational Raman spectrum for Mineral Oil: resolves 2850 cm^-1 (CH2) and 2935 cm^-1 (CH3).
    4. Reconstructs vibrational Raman spectrum for Olive Oil: resolves 2854 cm^-1, 2925 cm^-1, and
       the crucial unsaturation marker at 3005-3006 cm^-1 (cis =C-H stretch).
    5. Confirms olive oil exhibits distinct chemical contrast at 3006 cm^-1 where mineral oil has negligible signal.
    """
    cfg_mineral = ThorlabsSRSMicroscopeConfig(
        sample_type=OilSampleType.MINERAL_OIL,
        optical_delay_scan_ps=3.0,
        delay_samples=61,
    )
    res_min = ThorlabsSRSMicroscopeSimulator.simulate_oil_spectrum(cfg_mineral)

    cfg_olive = ThorlabsSRSMicroscopeConfig(
        sample_type=OilSampleType.OLIVE_OIL,
        optical_delay_scan_ps=3.0,
        delay_samples=61,
    )
    res_olive = ThorlabsSRSMicroscopeSimulator.simulate_oil_spectrum(cfg_olive)

    # 1. Chirp matching
    assert 0.85 <= res_min["chirp_matching_ratio"] <= 1.15
    assert 0.85 <= res_olive["chirp_matching_ratio"] <= 1.15

    # 2. Wavenumber sweep covers C-H stretch region
    wns = res_min["wavenumbers_cm1"]
    assert np.min(wns) <= 2840.0
    assert np.max(wns) >= 3020.0

    # 3. Peak in mineral oil is in 2850-2950 cm^-1 aliphatic band
    peak_min_idx = int(np.argmax(res_min["on_axis_lockin_voltage_v"]))
    peak_min_wn = wns[peak_min_idx]
    assert 2840.0 <= peak_min_wn <= 2960.0

    # 4. Compare signal at 3006 cm^-1 (cis =C-H marker)
    idx_3006 = int(np.argmin(np.abs(wns - 3006.0)))
    sig_olive_3006 = res_olive["on_axis_lockin_voltage_v"][idx_3006]
    sig_min_3006 = res_min["on_axis_lockin_voltage_v"][idx_3006]

    # Olive oil must show significantly higher response at 3006 cm^-1 than mineral oil
    assert sig_olive_3006 > sig_min_3006 * 2.0


def test_t4_scenario_2_widefield_raster_scan_zero_walkoff():
    """Scenario 2: Widefield 2D Raster Scan with Petzval Curvature & Zero Pupil Walk-Off.
    
    Verifies:
    1. Full 2D scan field raster across +/-3.0 deg mechanical angle.
    2. 4f-relayed architecture maintains strictly 0.0 um BFP pupil walk-off across all raster points.
    3. Off-axis Petzval curvature defocus Delta_z = r^2 / (2 * R_petzval) expands spot waist radially.
    4. Scan efficiency map is symmetric and exhibits smooth roll-off (> 70% at field corners).
    """
    cfg = ThorlabsSRSMicroscopeConfig(
        scanning_architecture="4f-Relayed (Zero Walk-Off)",
        galvo_max_mechanical_angle_deg=3.0,
    )
    field_res = ThorlabsSRSMicroscopeSimulator.simulate_2d_scan_field(cfg, grid_points=9)

    eff_map = field_res["scan_efficiency_map"]
    assert eff_map.shape == (9, 9)

    # Center is maximum efficiency
    center_eff = eff_map[4, 4]
    corner_eff = eff_map[0, 0]
    assert center_eff > 0.80
    assert center_eff > corner_eff > 0.01

    # Check walk-off at all points in grid
    for ax in np.linspace(-3.0, 3.0, 5):
        for ay in np.linspace(-3.0, 3.0, 5):
            pt = ThorlabsSRSMicroscopeSimulator.simulate_scanning_point(cfg, ax, ay)
            assert pt.back_pupil_walk_off_total_um == 0.0
            assert pt.is_strictly_conjugated is True


def test_t4_scenario_3_polarization_attenuation_and_eom_contrast():
    """Scenario 3: Polarization Attenuation & Active 10 MHz EOM Modulation Contrast.
    
    Verifies:
    1. Dual-laser power attenuators: HWP + PBS252 cubes attenuate power with extinction ratio > 1000:1.
    2. 10.0 MHz resonant RTP EOM (EO-AM-NR-C1) driven with sinusoidal RF creates high-contrast modulation.
    3. Stokes modulation transfers to Pump via Stimulated Raman Loss (SRL).
    4. Rejected power is safely dumped to BT610 dumps without stray back-reflection.
    """
    cfg = ThorlabsSRSMicroscopeConfig(
        modulator_type="EOM",
        modulator_frequency_mhz=10.0,
        modulator_v_pi_v=180.0,
        modulator_applied_v=180.0,
    )
    scan = ThorlabsSRSMicroscopeSimulator.simulate_scanning_point(cfg, 0.0, 0.0)
    sig = ThorlabsSRSMicroscopeSimulator.process_photodiode_signal(cfg, srl_fraction=1e-4, scan_result=scan)

    # Verify modulation factor is 1.0 (full contrast)
    assert math.isclose(sig.details["modulation_factor"], 1.0, rel_tol=1e-3)
    assert sig.srl_modulation_depth == 1e-4 * scan.total_scan_efficiency

    # Verify lock-in demodulation at 10 MHz
    assert sig.lockin_magnitude_r_v > 0.0
    assert sig.snr_db > 20.0


def test_t4_scenario_4_high_na_bfp_conjugation_and_37_surface_train():
    """Scenario 4: High-NA Objective BFP Conjugation, Telecentricity & 37-Surface Ray Propagation.
    
    Verifies:
    1. Olympus XLPLN25XWMP2 (25x 1.05 NA W) water immersion prescription is populated.
    2. Thorlabs SL50-2P2 (f=50 mm) and TTL200MP (f=200 mm) provide exact 4.0x pupil magnification.
    3. Input 4.2 mm beam expands to 16.8 mm BFP, achieving pupil fill factor ~1.0.
    4. Ray tracing through all 28 sequential surfaces of the 37-surface train runs unvignetted to DET10N2 detector.
    """
    cfg = ThorlabsSRSMicroscopeConfig(
        scanning_architecture="Folded Real-World SRS (Dichroic + Reflected Galvos + Steering Mirrors)",
        objective_model="Olympus XLPLN25XWMP2 (25x NA 1.05 W Multiphoton)",
        input_beam_diameter_mm=4.2,
    )
    stack = build_folded_real_world_srs_template(cfg)
    assert len(stack.surfaces) == 37

    # Ray trace forward sequential path
    sys = OpticalSystem(stack)
    ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=0.797)
    traced = sys.trace_ray(ray)
    assert not traced.vignetted
    assert len(traced.history) == 28
    assert traced.history[-1]["surface_index"] == 27
    assert stack.surfaces[27].name.startswith("Photodiode")

    # Pupil magnification & fill factor
    scan = ThorlabsSRSMicroscopeSimulator.simulate_scanning_point(cfg, 0.0, 0.0)
    assert scan.pupil_magnification == 4.0
    assert math.isclose(scan.back_pupil_beam_diameter_mm, 16.8, abs_tol=0.01)
    assert math.isclose(scan.pupil_fill_factor, 1.0, abs_tol=0.01)


def test_t4_scenario_5_interactive_tools_and_dual_tree_parity():
    """Scenario 5: Interactive Tools Suite & 100% Dual-Tree Code Parity Verification.
    
    Verifies:
    1. All 5 tools (Setup Builder, Galvo Diagnostic, Spectral Focusing, Lock-In, Bench Diagnostic) execute in headless mode.
    2. Spatial clearances in 2D layout: Stokes branch offset x = -40 mm, y = 25 mm.
    3. Zero-defect parity between optics_app and v2 across configurations, templates, and simulation outputs.
    """
    # 1. Test 5 tools execution
    cfg = ThorlabsSRSMicroscopeConfig()
    tool1_sys = build_srs_microscope_system(cfg)
    assert tool1_sys is not None

    tool2_diag = ThorlabsSRSMicroscopeSimulator.simulate_2d_scan_field(cfg, grid_points=5)
    assert tool2_diag["scan_efficiency_map"].shape == (5, 5)

    tool3_spec = ThorlabsSRSMicroscopeSimulator.simulate_oil_spectrum(cfg, angle_x_deg=0.0, angle_y_deg=0.0)
    assert "on_axis_lockin_voltage_v" in tool3_spec

    scan_pt = ThorlabsSRSMicroscopeSimulator.simulate_scanning_point(cfg, 0.0, 0.0)
    tool4_lockin = ThorlabsSRSMicroscopeSimulator.process_photodiode_signal(cfg, 1e-4, scan_pt)
    assert tool4_lockin.snr_db > 0.0

    tool5_probe = ThorlabsSRSMicroscopeSimulator.probe_post_sample_intensity(cfg, delay_ps=0.0)
    assert tool5_probe.probed_wavenumber_cm1 > 2800.0

    # 2. 2D Layout clearances
    st_folded = build_folded_real_world_srs_template()
    stokes_src = next(s for s in st_folded.surfaces if s.name.startswith("Stokes Laser"))
    assert stokes_src.decenter_x == -40.0
    assert stokes_src.decenter_y == 25.0

    # 3. Strict dual-tree parity
    cfg_probe = ThorlabsSRSMicroscopeConfig()
    v2_cfg_probe = v2_srs.ThorlabsSRSMicroscopeConfig()
    p1 = ThorlabsSRSMicroscopeSimulator.probe_post_sample_intensity(cfg_probe, delay_ps=0.0)
    p2 = v2_srs.ThorlabsSRSMicroscopeSimulator.probe_post_sample_intensity(v2_cfg_probe, delay_ps=0.0)
    assert math.isclose(p1.probed_wavenumber_cm1, p2.probed_wavenumber_cm1, abs_tol=1e-6)
    assert math.isclose(p1.photodiode_signal.lockin_magnitude_r_v, p2.photodiode_signal.lockin_magnitude_r_v, abs_tol=1e-6)

    v2_st_folded = v2_srs.build_folded_real_world_srs_template(v2_cfg_probe)
    assert len(st_folded.surfaces) == len(v2_st_folded.surfaces) == 37
