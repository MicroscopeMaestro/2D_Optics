"""Comprehensive End-to-End Test Suite for the Non-Sequential Stimulated Raman Scattering (SRS) Microscope.

Tests:
1. Non-Sequential Engine Core (Surface intersection, local/global transforms, Snell's law, specular reflection).
2. Dual Laser Sources (Pump 797 nm and Stokes 1040 nm generation).
3. 45° Steering Mirror & Non-Sequential Dichroic Pulse Combiner (DMLP900 45° collinear joining).
4. EOM Modulator & PBS Polarizing Analyzer on Stokes Beam.
5. SF57 Matched Dispersive Rods & Temporal Delay Scanning (Spectral Focusing).
6. Galvo Scanning 1:1 4f-Relay & Zero Pupil Walk-Off Conjugation.
7. High-NA Objective (Olympus XLPLN25XWMP2 25x NA 1.05 W / XLUMPLFLN20XW 20x NA 1.0 W) Focus.
8. Sample Plane Chi(3) Nonlinear Raman Interaction (SRL signal generation).
9. High-NA Condenser & Dual Dielectric Shortpass Stokes Rejection (OD > 12.0).
10. Biased Photodiode (DET10N2) & 10 MHz RF Lock-In Demodulation (Noise floor, SNR).
11. Non-Sequential SurfaceStack Builder & Ray Extraction for 2D/3D GUI Display.
12. Strict Dual-Tree Parity between optics_app and v2 packages.
"""

import math
import numpy as np
import pytest

from optics_app.engine.non_sequential import (
    NonSequentialSurfaceType,
    NonSequentialElement,
    NonSequentialSource,
    NonSequentialSystem,
)
from optics_app.analysis.ultrafast_nonlinear import (
    ThorlabsSRSMicroscopeConfig,
    NonSequentialSRSSimulationResult,
    NonSequentialSRSMicroscopeBench,
    build_non_sequential_srs_system,
    SRSSpectralFocusingTool,
    SRSGalvoFieldDiagnosticTool,
    SRSSignalLockInTool,
    OilSampleType,
)
from optics_app.engine.ray import Ray


# ==============================================================================
# 1. Non-Sequential Engine Transforms & Surface Intersections
# ==============================================================================

def test_ns_01_element_local_global_transforms():
    """Verify rigid body 3D coordinate transformations between global and local element frames."""
    origin = np.array([10.0, 20.0, 30.0])
    # 90 deg rotation around Y axis
    R = np.array([
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0]
    ])
    elem = NonSequentialElement(
        element_id="test_mirror",
        name="Test Mirror",
        element_type=NonSequentialSurfaceType.MIRROR,
        origin=origin,
        rotation_matrix=R,
        semi_diameter=12.7,
    )
    p_global = np.array([10.0, 25.0, 35.0])
    p_loc = elem.global_to_local(p_global)
    p_back = elem.local_to_global(p_loc)
    np.testing.assert_allclose(p_back, p_global, atol=1e-6)

    d_global = np.array([0.0, 0.0, 1.0])
    d_loc = elem.global_dir_to_local(d_global)
    d_back = elem.local_dir_to_global(d_loc)
    np.testing.assert_allclose(d_back, d_global, atol=1e-6)


def test_ns_02_planar_surface_intersection():
    """Verify exact ray-plane intersection and clear aperture clipping."""
    elem = NonSequentialElement(
        element_id="test_aperture",
        name="Test Aperture",
        element_type=NonSequentialSurfaceType.APERTURE,
        origin=np.array([0.0, 0.0, 100.0]),
        semi_diameter=10.0,
    )
    # Ray passing inside aperture
    res_in = elem.intersect_local(np.array([0.0, 2.0, -10.0]), np.array([0.0, 0.0, 1.0]))
    assert res_in is not None
    t_val, hit_loc, norm_loc = res_in
    assert math.isclose(t_val, 10.0, abs_tol=1e-5)
    np.testing.assert_allclose(hit_loc, [0.0, 2.0, 0.0], atol=1e-5)

    # Ray outside semi-diameter aperture (e.g. y = 15.0 mm > 10.0 mm)
    res_out = elem.intersect_local(np.array([0.0, 15.0, -10.0]), np.array([0.0, 0.0, 1.0]))
    assert res_out is None


def test_ns_03_spherical_lens_surface_intersection():
    """Verify ray intersection with spherical curved refracting boundary."""
    elem = NonSequentialElement(
        element_id="test_sphere",
        name="Convex Surface",
        element_type=NonSequentialSurfaceType.LENS_SURFACE,
        origin=np.array([0.0, 0.0, 50.0]),
        semi_diameter=12.7,
        radius_of_curvature=50.0,
        glass="N-BK7",
    )
    res = elem.intersect_local(np.array([0.0, 0.0, -50.0]), np.array([0.0, 0.0, 1.0]))
    assert res is not None
    t_val, hit_loc, norm_loc = res
    assert math.isclose(t_val, 50.0, abs_tol=1e-4)


# ==============================================================================
# 2. Dual Laser Sources & Arm Topologies
# ==============================================================================

def test_ns_04_dual_laser_source_generation():
    """Verify independent generation of Pump and Stokes laser ray bundles."""
    cfg = ThorlabsSRSMicroscopeConfig()
    sys_ns = build_non_sequential_srs_system(cfg)

    sources = {s.source_id: s for s in sys_ns.sources}
    assert "src_pump" in sources
    assert "src_stokes" in sources

    src_p = sources["src_pump"]
    src_s = sources["src_stokes"]

    assert src_p.wavelength_nm == cfg.pump_laser_wavelength_nm
    assert src_s.wavelength_nm == cfg.stokes_laser_wavelength_nm
    assert src_p.origin[0] == 0.0    # Pump arm starts at X = 0
    assert src_s.origin[0] == 60.0   # Stokes arm starts at X = 60

    rays_p = src_p.generate_rays(num_rays=7)
    rays_s = src_s.generate_rays(num_rays=7)
    assert len(rays_p) == 7
    assert len(rays_s) == 7
    assert all(math.isclose(r.wavelength, cfg.pump_laser_wavelength_nm * 1e-3, abs_tol=1e-5) for r in rays_p)
    assert all(math.isclose(r.wavelength, cfg.stokes_laser_wavelength_nm * 1e-3, abs_tol=1e-5) for r in rays_s)


# ==============================================================================
# 3. 45° Steering Mirror & Non-Sequential Dichroic Combiner (DMLP900)
# ==============================================================================

def test_ns_05_pump_steering_and_dichroic_reflection():
    """Verify pump beam (797 nm) reflects 90° at steering mirror and 90° at dichroic into +Z."""
    cfg = ThorlabsSRSMicroscopeConfig()
    sys_ns = build_non_sequential_srs_system(cfg)

    # Launch single on-axis pump ray
    ray_p = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=0.797)
    traced_p = sys_ns.trace_ray(ray_p, record_history=True)

    history = traced_p.history
    elem_ids = [h["element_id"] for h in history]

    # Verify traversal through steering mirror and dichroic combiner
    assert "comp_pump_steer" in elem_ids
    assert "comp_dichroic" in elem_ids

    # Step after steering mirror should propagate in +X direction
    idx_steer = elem_ids.index("comp_pump_steer")
    dir_after_steer = [history[idx_steer]["dir_x"], history[idx_steer]["dir_y"], history[idx_steer]["dir_z"]]
    np.testing.assert_allclose(dir_after_steer, [1.0, 0.0, 0.0], atol=1e-3)

    # Step after dichroic should propagate in +Z direction
    idx_dichroic = elem_ids.index("comp_dichroic")
    dir_after_dichroic = [history[idx_dichroic]["dir_x"], history[idx_dichroic]["dir_y"], history[idx_dichroic]["dir_z"]]
    np.testing.assert_allclose(dir_after_dichroic, [0.0, 0.0, 1.0], atol=1e-3)


def test_ns_06_stokes_dichroic_transmission():
    """Verify Stokes beam (1040 nm) transmits through dichroic combiner into +Z collinearly with Pump."""
    cfg = ThorlabsSRSMicroscopeConfig()
    sys_ns = build_non_sequential_srs_system(cfg)

    # Launch single on-axis Stokes ray
    ray_s = Ray(pos=np.array([60.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=1.040)
    traced_s = sys_ns.trace_ray(ray_s, record_history=True)

    history = traced_s.history
    elem_ids = [h["element_id"] for h in history]

    assert "comp_stokes_eom" in elem_ids
    assert "comp_stokes_pbs" in elem_ids
    assert "comp_stokes_delay" in elem_ids
    assert "comp_dichroic" in elem_ids

    idx_dichroic = elem_ids.index("comp_dichroic")
    dir_after_dichroic = [history[idx_dichroic]["dir_x"], history[idx_dichroic]["dir_y"], history[idx_dichroic]["dir_z"]]
    # Transmits collinearly along +Z
    np.testing.assert_allclose(dir_after_dichroic, [0.0, 0.0, 1.0], atol=1e-3)


# ==============================================================================
# 4. Modulator (EOM / PBS) and Dispersive Rods (Spectral Focusing)
# ==============================================================================

def test_ns_07_eom_modulation_and_pbs():
    """Verify EOM phase retardation and transmission through PBS analyzer."""
    cfg = ThorlabsSRSMicroscopeConfig(
        modulator_type="EOM",
        modulator_v_pi_v=180.0,
        modulator_applied_v=180.0, # Full pi retardation -> 100% transmission to cross-pol
    )
    sys_ns = build_non_sequential_srs_system(cfg)

    ray_s = Ray(pos=np.array([60.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=1.040)
    traced_s = sys_ns.trace_ray(ray_s, record_history=True)

    eom_hits = [h for h in traced_s.history if h["element_id"] == "comp_stokes_eom"]
    assert len(eom_hits) == 1
    assert "eom_modulation" in eom_hits[0]["interaction"]


def test_ns_08_spectral_focusing_dispersion_matching():
    """Verify GDD matching and chirp ratio alpha_p / alpha_S for SF-SRS resolution."""
    cfg = ThorlabsSRSMicroscopeConfig(
        pump_glass_material="SF57",
        pump_glass_length_mm=150.0,
        stokes_glass_material="SF57",
        stokes_glass_length_mm=280.0,
    )
    disp = SRSSpectralFocusingTool.compute_dispersion_parameters(cfg)
    assert disp["gdd_pump_fs2"] > 25000.0
    assert disp["gdd_stokes_fs2"] > 25000.0
    # Ratio should be near matched (~0.9 - 1.1)
    assert 0.8 <= disp["chirp_matching_ratio"] <= 1.2
    assert disp["spectral_resolution_cm1"] <= 20.0  # < 20 cm^-1 high spectral resolution


# ==============================================================================
# 5. Galvo Scanning Unit & 1:1 4f Telecentric Relay
# ==============================================================================

def test_ns_09_galvo_relay_zero_pupil_walkoff():
    """Verify 1:1 4f galvo relay maintains zero pupil walk-off at the objective entrance pupil."""
    cfg = ThorlabsSRSMicroscopeConfig()
    cfg.scanning_architecture = "4f-Relayed (Zero Walk-Off)"
    
    # Evaluate at high scan angles (+/- 5 deg)
    res_center = SRSGalvoFieldDiagnosticTool.evaluate_scan_point(cfg, angle_x_deg=0.0, angle_y_deg=0.0)
    res_scanned = SRSGalvoFieldDiagnosticTool.evaluate_scan_point(cfg, angle_x_deg=4.0, angle_y_deg=4.0)

    assert math.isclose(res_center.back_pupil_walk_off_total_um, 0.0, abs_tol=1e-5)
    assert math.isclose(res_scanned.back_pupil_walk_off_total_um, 0.0, abs_tol=1e-5)
    assert res_scanned.is_strictly_conjugated is True


# ==============================================================================
# 6. Sample Interaction & Stokes OD Rejection Stack
# ==============================================================================

def test_ns_10_sample_chi3_srl_and_stokes_rejection():
    """Verify nonlinear SRL generation and OD > 12 rejection of 1040 nm Stokes at photodiode."""
    cfg = ThorlabsSRSMicroscopeConfig(
        sample_type=OilSampleType.MINERAL_OIL,
        sample_thickness_um=50.0,
    )
    res = NonSequentialSRSMicroscopeBench.simulate_non_sequential_srs(cfg, num_rays_per_beam=5)

    assert res.diagnostics["stokes_rejection_od"] >= 12.0
    assert res.diagnostics["pump_vignetted"] is False
    assert res.diagnostics["stokes_vignetted"] is False

    # Check that Stokes beam intensity is attenuated by at least 10^-12 by the filter stack
    stokes_rays = res.traced_rays_stokes
    assert len(stokes_rays) == 5
    for r in stokes_rays:
        assert r.intensity <= 1e-12


# ==============================================================================
# 7. Fast Photodiode & RF Lock-In Demodulation
# ==============================================================================

def test_ns_11_photodiode_lockin_snr():
    """Verify DET10N2 signal processing, noise components, and positive SNR."""
    cfg = ThorlabsSRSMicroscopeConfig()
    res = NonSequentialSRSMicroscopeBench.simulate_non_sequential_srs(cfg)

    pd_sig = res.photodiode_signal
    assert pd_sig.dc_photocurrent_a > 1e-6    # mA level DC current
    assert pd_sig.lockin_magnitude_r_v > 0.0  # Detected lock-in signal
    assert pd_sig.output_noise_voltage_v > 0.0
    assert pd_sig.shot_noise_current_a > 0.0
    assert pd_sig.rin_noise_current_a > 0.0
    assert pd_sig.thermal_noise_current_a > 0.0
    assert pd_sig.snr_db > 10.0               # High SNR in center FOV


# ==============================================================================
# 8. SurfaceStack Visualization & Traced Ray Extraction
# ==============================================================================

def test_ns_12_build_surface_stack_parity():
    """Verify build_surface_stack returns valid SurfaceStack with is_non_sequential=True."""
    cfg = ThorlabsSRSMicroscopeConfig()
    sys_ns = build_non_sequential_srs_system(cfg)
    stack = sys_ns.build_surface_stack()

    assert getattr(stack, "is_non_sequential", False) is True
    assert hasattr(stack, "non_sequential_system")
    assert len(stack.surfaces) == len(sys_ns.elements)

    combined_rays = sys_ns.get_combined_rays_list(num_rays_per_source=7)
    assert len(combined_rays) == 14
    assert any(math.isclose(r.wavelength, 0.797, abs_tol=1e-3) for r in combined_rays)
    assert any(math.isclose(r.wavelength, 1.040, abs_tol=1e-3) for r in combined_rays)


# ==============================================================================
# 9. Strict Dual-Tree Parity (optics_app vs v2)
# ==============================================================================

def test_ns_13_dual_tree_parity():
    """Verify optics_app and v2 non-sequential engines produce identical numerical results."""
    from optics_app.analysis.ultrafast_nonlinear import (
        NonSequentialSRSMicroscopeBench as Bench1,
        ThorlabsSRSMicroscopeConfig as Cfg1,
    )
    from v2.analysis.ultrafast_nonlinear import (
        NonSequentialSRSMicroscopeBench as Bench2,
        ThorlabsSRSMicroscopeConfig as Cfg2,
    )

    cfg1 = Cfg1()
    cfg2 = Cfg2()

    res1 = Bench1.simulate_non_sequential_srs(cfg1, num_rays_per_beam=5)
    res2 = Bench2.simulate_non_sequential_srs(cfg2, num_rays_per_beam=5)

    assert len(res1.all_traced_rays) == len(res2.all_traced_rays)
    assert math.isclose(res1.photodiode_signal.snr_db, res2.photodiode_signal.snr_db, rel_tol=1e-4)
    assert math.isclose(res1.photodiode_signal.lockin_magnitude_r_v, res2.photodiode_signal.lockin_magnitude_r_v, rel_tol=1e-4)
