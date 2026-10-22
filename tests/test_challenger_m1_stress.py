"""
Adversarial Stress Test Suite for Milestone 1 Re-Evaluation (Pytest)
2D Optics CAD Import & Profile Extraction Engine
"""

import sys
import os
import math
import numpy as np
import pytest

from optics_app.cad.profile import OpticalProfile, FitType
from optics_app.cad.fitter import SagFitter, ProfilePoints, fit_sag_profile
from optics_app.cad.slicer import ZSlicer, SlicedLensGeometry, CADAligner
from optics_app.cad.importer import STEPParser, CADSurfaceImporter, CADImporter


def test_dim1_sag_fitting():
    # 1.1 Spherical fit
    R_target = 50.0
    c_target = 1.0 / R_target
    r_pts = np.linspace(0, 20, 50)
    z_pts = (c_target * r_pts**2) / (1.0 + np.sqrt(1.0 - c_target**2 * r_pts**2))
    
    c_fit, R_fit, rms_um = SagFitter.fit_spherical(r_pts, z_pts)
    c_err = abs(c_fit - c_target)
    R_err = abs(R_fit - R_target)
    assert c_err < 1e-8, f"Spherical curvature error too high: {c_err}"
    assert R_err < 1e-6, f"Spherical radius error too high: {R_err}"
    assert rms_um < 1e-4, f"Spherical RMS error too high: {rms_um} um"

    # 1.2 Concave sphere fit (negative curvature)
    R_concave = -100.0
    c_concave = 1.0 / R_concave
    z_concave = (c_concave * r_pts**2) / (1.0 + np.sqrt(1.0 - c_concave**2 * r_pts**2))
    c_fit_c, R_fit_c, rms_um_c = SagFitter.fit_spherical(r_pts, z_concave)
    assert abs(c_fit_c - c_concave) < 1e-8, f"Concave curvature error: {abs(c_fit_c - c_concave)}"
    assert abs(R_fit_c - R_concave) < 1e-6, f"Concave radius error: {abs(R_fit_c - R_concave)}"

    # 1.3 Planar surface fit (c = 0)
    z_flat = np.zeros_like(r_pts)
    c_flat, R_flat, rms_flat = SagFitter.fit_spherical(r_pts, z_flat)
    prof_flat = OpticalProfile(fit_type=FitType.SPHERICAL, curvature=0.0, radius_of_curvature=float('inf'))
    sag_flat = prof_flat.sag(r_pts)
    assert np.all(sag_flat == 0.0), "Flat sag should be zero"

    # 1.4 Conic paraboloid fit (k = -1.0)
    k_para = -1.0
    R_para = 60.0
    c_para = 1.0 / R_para
    z_para = 0.5 * c_para * r_pts**2
    c_fit_p, k_fit_p, rms_para = SagFitter.fit_conic(r_pts, z_para)
    assert abs(c_fit_p - c_para) < 1e-6, f"Paraboloid curvature error: {abs(c_fit_p - c_para)}"
    assert abs(k_fit_p - k_para) < 1e-4, f"Paraboloid conic error: {abs(k_fit_p - k_para)}"

    # 1.5 Conic hyperboloid fit (k = -2.5)
    k_hyp = -2.5
    R_hyp = 40.0
    c_hyp = 1.0 / R_hyp
    arg_hyp = 1.0 - (1.0 + k_hyp) * c_hyp**2 * r_pts**2
    z_hyp = (c_hyp * r_pts**2) / (1.0 + np.sqrt(arg_hyp))
    c_fit_h, k_fit_h, rms_hyp = SagFitter.fit_conic(r_pts, z_hyp)
    assert abs(c_fit_h - c_hyp) < 1e-5, f"Hyperboloid c error: {abs(c_fit_h - c_hyp)}"
    assert abs(k_fit_h - k_hyp) < 1e-3, f"Hyperboloid k error: {abs(k_fit_h - k_hyp)}"

    # 1.6 Higher order asphere fit (a4, a6, a8, a10)
    a4, a6, a8, a10 = 1e-5, -2e-8, 3e-11, -4e-14
    z_asph = z_para + a4 * r_pts**4 + a6 * r_pts**6 + a8 * r_pts**8 + a10 * r_pts**10
    c_fit_a, k_fit_a, coeffs_fit_a, rms_asph = SagFitter.fit_aspheric(r_pts, z_asph)
    prof_asph = OpticalProfile(fit_type=FitType.ASPHERIC, curvature=c_fit_a, conic_constant=k_fit_a, aspheric_coeffs=coeffs_fit_a)
    z_eval_asph = prof_asph.sag(r_pts)
    max_asph_err_um = np.max(np.abs(z_asph - z_eval_asph)) * 1000.0
    # Empirically demonstrated bug in SagFitter.fit_aspheric:
    # Expected max_asph_err_um < 0.1 um, got ~13.18 um due to Stage 1 base conic coupling
    assert max_asph_err_um < 0.1, f"EMPIRICAL BUG: Aspheric fit max error too high: {max_asph_err_um:.4f} um (expected < 0.1 um)"

    # 1.7 Cubic Spline fit
    spline_obj, rms_spline = SagFitter.fit_spline(r_pts, z_asph)
    z_spline_eval = spline_obj(r_pts)
    max_spline_err_um = np.max(np.abs(z_asph - z_spline_eval)) * 1000.0
    assert max_spline_err_um < 1e-4, f"Spline interpolation error too high: {max_spline_err_um} um"

    # 1.8 Noise resilience test
    np.random.seed(42)
    noise_um = 0.05
    z_noisy = z_pts + np.random.normal(0, noise_um / 1000.0, size=len(r_pts))
    c_noisy, R_noisy, rms_noisy = SagFitter.fit_spherical(r_pts, z_noisy)
    assert abs(R_noisy - 50.0) < 0.1, f"Noisy fit radius error too large: {R_noisy}"


def test_dim2_normal_vector():
    prof_sph = OpticalProfile(fit_type=FitType.SPHERICAL, curvature=0.02, radius_of_curvature=50.0)
    prof_con = OpticalProfile(fit_type=FitType.CONIC, curvature=0.02, conic_constant=-1.0)
    prof_asph = OpticalProfile(fit_type=FitType.ASPHERIC, curvature=0.02, conic_constant=0.0, aspheric_coeffs=[1e-5, 1e-8])
    prof_flat = OpticalProfile(fit_type=FitType.SPHERICAL, curvature=0.0, radius_of_curvature=float('inf'))
    prof_concave = OpticalProfile(fit_type=FitType.SPHERICAL, curvature=-0.01, radius_of_curvature=-100.0)

    r_test = np.linspace(0, 20, 20)
    z_test = prof_sph.sag(r_test)
    spline, _ = SagFitter.fit_spline(r_test, z_test)
    prof_spline = OpticalProfile(fit_type=FitType.CUBIC_SPLINE, spline_object=spline)

    profiles = [
        ("Spherical", prof_sph),
        ("Conic", prof_con),
        ("Aspheric", prof_asph),
        ("Planar", prof_flat),
        ("Concave", prof_concave),
        ("Spline", prof_spline)
    ]

    for name, prof in profiles:
        n_apex = prof.normal_vector(0.0, 0.0, 0.0)
        assert np.allclose(n_apex, (0.0, 0.0, 1.0), atol=1e-12), f"{name} apex normal is not (0,0,1): {n_apex}"

        np.random.seed(123)
        for _ in range(50):
            x = float(np.random.uniform(-15.0, 15.0))
            y = float(np.random.uniform(-15.0, 15.0))
            r = math.sqrt(x**2 + y**2)
            if r < 1e-6:
                continue
            z = float(prof.sag(r))

            n_hat = prof.normal_vector(x, y, z)
            n_len = np.linalg.norm(n_hat)
            assert abs(n_len - 1.0) < 1e-12, f"{name} normal vector length is not 1: {n_len}"

            h = 1e-7
            dz_dx = (prof.sag(math.sqrt((x + h)**2 + y**2)) - prof.sag(math.sqrt((x - h)**2 + y**2))) / (2 * h)
            dz_dy = (prof.sag(math.sqrt(x**2 + (y + h)**2)) - prof.sag(math.sqrt(x**2 + (y - h)**2))) / (2 * h)
            
            grad_S = np.array([-dz_dx, -dz_dy, 1.0])
            grad_S_unit = grad_S / np.linalg.norm(grad_S)

            diff = np.linalg.norm(np.array(n_hat) - grad_S_unit)
            assert diff < 1e-5, f"{name} normal mismatch with finite difference gradient at ({x:.2f},{y:.2f}): diff={diff}"


def test_dim3_clear_aperture_bounds():
    semi_diam = 20.0
    prof = OpticalProfile(fit_type=FitType.SPHERICAL, curvature=0.02, radius_of_curvature=50.0, semi_diameter=semi_diam)

    # Inside aperture
    P0 = (10.0, 0.0, -10.0)
    v = (0.0, 0.0, 1.0)
    t, P_hit, n_hit, is_valid = prof.intersect_ray(P0, v)
    assert is_valid == True, f"Inside aperture ray should be valid"
    assert t > 0.0, "Ray distance t must be > 0"

    # On boundary (r = 20.0)
    P0_b = (20.0, 0.0, -10.0)
    t_b, P_hit_b, n_hit_b, is_valid_b = prof.intersect_ray(P0_b, v)
    assert is_valid_b == True, f"On-boundary ray should be valid"

    # Outside clear aperture (r = 20.001)
    P0_out = (20.001, 0.0, -10.0)
    t_out, P_hit_out, n_hit_out, is_valid_out = prof.intersect_ray(P0_out, v)
    assert is_valid_out == False, f"Outside aperture ray (r=20.001) MUST be invalid"

    # Far outside clear aperture (r = 30.0)
    P0_far = (30.0, 0.0, -10.0)
    t_far, P_hit_far, n_hit_far, is_valid_far = prof.intersect_ray(P0_far, v)
    assert is_valid_far == False, f"Far outside aperture ray (r=30.0) MUST be invalid"

    # Unbounded aperture (semi_diameter = 0.0)
    prof_unbounded = OpticalProfile(fit_type=FitType.SPHERICAL, curvature=0.02, radius_of_curvature=50.0, semi_diameter=0.0)
    t_unb, P_hit_unb, n_hit_unb, is_valid_unb = prof_unbounded.intersect_ray(P0_far, v)
    assert is_valid_unb == True, "Unbounded surface should accept ray"


def test_dim4_newton_raphson_solver():
    prof_asph = OpticalProfile(
        fit_type=FitType.ASPHERIC,
        curvature=0.02,
        radius_of_curvature=50.0,
        conic_constant=-1.0,
        aspheric_coeffs=[1e-5, 1e-8],
        semi_diameter=25.0
    )

    # Forward ray (+Z direction)
    P0_fwd = (5.0, 5.0, -15.0)
    v_fwd = (0.0, 0.0, 1.0)
    t_fwd, P_fwd, n_fwd, valid_fwd = prof_asph.intersect_ray(P0_fwd, v_fwd)
    r_fwd = math.sqrt(P_fwd[0]**2 + P_fwd[1]**2)
    z_expected = prof_asph.sag(r_fwd)
    z_err = abs(P_fwd[2] - z_expected)
    assert t_fwd > 0.0, "Forward ray distance must be > 0"
    assert valid_fwd == True, "Forward ray should be valid"
    assert z_err < 1e-6, f"Intersection point Z coordinate mismatch: {z_err}"

    # Backward ray (+Z origin moving +Z away from surface)
    P0_back = (5.0, 5.0, +15.0)
    v_back = (0.0, 0.0, 1.0)
    t_back, P_back, n_back, valid_back = prof_asph.intersect_ray(P0_back, v_back)
    assert t_back < 0.0, f"Backward ray distance must be negative: {t_back}"
    assert valid_back == False, f"Backward ray (t={t_back}) MUST be invalid (forward guard failed)"

    # Backward ray (-Z origin pointing -Z away from surface)
    P0_back2 = (5.0, 5.0, -15.0)
    v_back2 = (0.0, 0.0, -1.0)
    t_back2, P_back2, n_back2, valid_back2 = prof_asph.intersect_ray(P0_back2, v_back2)
    assert t_back2 < 0.0, f"Backward ray distance must be negative: {t_back2}"
    assert valid_back2 == False, f"Backward ray (t={t_back2}) MUST be invalid (forward guard failed)"

    # Skewed 3D ray
    v_skew = np.array([0.1, 0.2, 0.9746794])
    v_skew /= np.linalg.norm(v_skew)
    P0_skew = np.array([-2.0, -3.0, -20.0])
    t_skew, P_skew, n_skew, valid_skew = prof_asph.intersect_ray(P0_skew, v_skew)
    r_skew = math.sqrt(P_skew[0]**2 + P_skew[1]**2)
    z_skew_expected = prof_asph.sag(r_skew)
    z_skew_err = abs(P_skew[2] - z_skew_expected)
    assert t_skew > 0.0, "Skewed ray t must be > 0"
    assert valid_skew == True, "Skewed ray should be valid"
    assert z_skew_err < 1e-6, f"Skewed ray Z residual error: {z_skew_err}"

    # Concave surface ray intersection
    prof_concave = OpticalProfile(fit_type=FitType.SPHERICAL, curvature=-0.02, radius_of_curvature=-50.0, semi_diameter=25.0)
    t_conc, P_conc, n_conc, valid_conc = prof_concave.intersect_ray((0.0, 0.0, -10.0), (0.0, 0.0, 1.0))
    assert t_conc > 0.0, "Concave surface ray t > 0"
    assert valid_conc == True, "Concave surface ray valid"
