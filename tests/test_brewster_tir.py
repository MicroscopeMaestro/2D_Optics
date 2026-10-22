"""
Unit Tests for Brewster Angle & Total Internal Reflection (TIR) Phase Shift (Requirement R5).
"""
import pytest
import numpy as np
from optics_app.engine.polarization import compute_fresnel_coefficients


def test_brewster_angle_dielectric():
    """Test Brewster angle zero-reflection condition for p-polarization."""
    n1 = 1.0
    n2 = 1.5
    theta_B = np.arctan(n2 / n1)  # ~56.30993 deg
    cos_theta_B = np.cos(theta_B)

    res = compute_fresnel_coefficients(n1, n2, cos_theta_B)

    assert abs(res.r_p) < 1e-6, f"Expected r_p ~ 0 at Brewster angle, got {res.r_p}"
    assert abs(res.r_s) > 0.1, "Expected r_s > 0 at Brewster angle"

    R_p = abs(res.r_p)**2
    assert R_p < 1e-12


def test_tir_critical_angle_and_phase_shift():
    """Test Total Internal Reflection (TIR) critical angle and phase retardation."""
    n1 = 1.5
    n2 = 1.0
    theta_c = np.arcsin(n2 / n1)  # ~41.8103 deg

    # Below critical angle (30 deg): partial reflection
    res_sub = compute_fresnel_coefficients(n1, n2, np.cos(np.radians(30)))
    assert not res_sub.tir
    assert abs(res_sub.r_s)**2 < 1.0

    # Above critical angle (50 deg): 100% reflection (TIR)
    res_tir = compute_fresnel_coefficients(n1, n2, np.cos(np.radians(50)))
    assert res_tir.tir
    assert abs(abs(res_tir.r_s) - 1.0) < 1e-6
    assert abs(abs(res_tir.r_p) - 1.0) < 1e-6

    # Test Fresnel rhomb phase retardation: at theta_1 ~ 53.3 deg
    theta_rhomb = np.radians(53.3)
    res_rhomb = compute_fresnel_coefficients(n1, n2, np.cos(theta_rhomb))
    delta_s = np.angle(res_rhomb.r_s)
    delta_p = np.angle(res_rhomb.r_p)

    # Relative phase shift per reflection ~ 45 degrees (pi/4)
    raw_diff_deg = np.degrees(delta_p - delta_s)
    delta_deg = (raw_diff_deg + 180.0) % 360.0 - 180.0
    retardance = abs(delta_deg)
    # Two TIR reflections give 90 deg retardance (QWP)
    assert abs(retardance - 45.0) < 1.0
