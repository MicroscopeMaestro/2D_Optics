"""
Unit Tests for Fresnel Reflection/Transmission Equations (Requirement R5).
"""
import pytest
import numpy as np
from optics_app.engine.polarization import compute_fresnel_coefficients


def test_fresnel_dielectric_normal_incidence():
    """Test Air-to-Glass (n1=1.0 -> n2=1.5) normal incidence Fresnel coefficients."""
    n1 = 1.0
    n2 = 1.5
    cos_theta1 = 1.0  # 0 degrees

    res = compute_fresnel_coefficients(n1, n2, cos_theta1)

    assert abs(res.r_s - (-0.2)) < 1e-6
    assert abs(res.r_p - 0.2) < 1e-6 or abs(res.r_p - (-0.2)) < 1e-6  # Magnitude 0.2

    R_s = abs(res.r_s)**2
    R_p = abs(res.r_p)**2
    assert abs(R_s - 0.04) < 1e-6
    assert abs(R_p - 0.04) < 1e-6

    # Intensity Transmittance T = (n2 * cos_theta2) / (n1 * cos_theta1) * |t|^2
    T_s = (n2 / n1) * abs(res.t_s)**2
    T_p = (n2 / n1) * abs(res.t_p)**2

    assert abs(T_s - 0.96) < 1e-6
    assert abs(T_p - 0.96) < 1e-6
    assert abs((R_s + T_s) - 1.0) < 1e-6
    assert abs((R_p + T_p) - 1.0) < 1e-6


def test_fresnel_dielectric_oblique_energy_conservation():
    """Test energy conservation R + T = 1 at oblique angles (15 to 75 deg)."""
    n1 = 1.0
    n2 = 1.5

    for angle_deg in [15, 30, 45, 60, 75]:
        theta1 = np.radians(angle_deg)
        cos_theta1 = np.cos(theta1)
        res = compute_fresnel_coefficients(n1, n2, cos_theta1)

        sin_theta2 = (n1 / n2) * np.sin(theta1)
        cos_theta2 = np.sqrt(1.0 - sin_theta2**2)

        R_s = abs(res.r_s)**2
        R_p = abs(res.r_p)**2

        T_s = (n2 * cos_theta2) / (n1 * cos_theta1) * abs(res.t_s)**2
        T_p = (n2 * cos_theta2) / (n1 * cos_theta1) * abs(res.t_p)**2

        assert abs((R_s + T_s) - 1.0) < 1e-5, f"Failed s-conservation at {angle_deg} deg"
        assert abs((R_p + T_p) - 1.0) < 1e-5, f"Failed p-conservation at {angle_deg} deg"


def test_fresnel_metal_reflectance():
    """Test complex index metallic reflection (Au, Al, Ag, Cu)."""
    # Gold (Au): n = 0.18 + 3.0i
    res_au = compute_fresnel_coefficients(1.0, complex(0.18, 3.0), 1.0)
    R_au = abs(res_au.r_s)**2
    expected_au = ((1 - 0.18)**2 + 3.0**2) / ((1 + 0.18)**2 + 3.0**2)
    assert abs(R_au - expected_au) < 1e-5
    assert abs(R_au - 0.930718) < 1e-3

    # Aluminum (Al): n = 1.097 + 6.79i
    res_al = compute_fresnel_coefficients(1.0, complex(1.097, 6.79), 1.0)
    R_al = abs(res_al.r_s)**2
    expected_al = ((1 - 1.097)**2 + 6.79**2) / ((1 + 1.097)**2 + 6.79**2)
    assert abs(R_al - expected_al) < 1e-5

    # Silver (Ag): n = 0.05 + 4.27i
    res_ag = compute_fresnel_coefficients(1.0, complex(0.05, 4.27), 1.0)
    R_ag = abs(res_ag.r_s)**2
    assert abs(R_ag - 0.989656) < 1e-3
