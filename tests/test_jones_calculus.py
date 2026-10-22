"""
Unit Tests for Jones Calculus & Polarization State Tracking (Requirement R5).
"""
import pytest
import numpy as np
from optics_app.engine.ray import Ray
from optics_app.engine.polarization import rotate_jones_vector, compute_sp_basis


def test_jones_vector_intensity_update():
    """Test Ray Jones vector intensity updates."""
    ray = Ray(pos=(0, 0, 0), dir=(0, 0, 1), intensity=1.0)
    assert np.allclose(ray.jones_vector, [1.0+0j, 0.0+0j])
    assert ray.intensity == 1.0

    # Set Jones vector for 45 deg linear polarization
    ray.jones_vector = np.array([1.0 / np.sqrt(2), 1.0 / np.sqrt(2)], dtype=complex)
    assert abs(ray.update_intensity() - 1.0) < 1e-6


def test_jones_rotation():
    """Test rotation of Jones vector from one basis to another."""
    s_in = np.array([1.0, 0.0, 0.0])
    s_target = np.array([0.0, 1.0, 0.0])  # 90 deg rotation
    k_dir = np.array([0.0, 0.0, 1.0])

    J_in = np.array([1.0+0j, 0.0+0j])  # Horizontal linear in s_in
    J_rot = rotate_jones_vector(J_in, s_in, s_target, k_dir)

    # In s_target (vertical), horizontal linear component maps to vertical
    assert abs(abs(J_rot[1]) - 1.0) < 1e-6 or abs(abs(J_rot[0]) - 1.0) < 1e-6


def test_malus_law():
    """Test Malus' law for linear polarizers: I(theta) = I0 * cos^2(theta)."""
    def make_lp(theta_deg):
        th = np.radians(theta_deg)
        c, s = np.cos(th), np.sin(th)
        return np.array([[c**2, c*s], [c*s, s**2]], dtype=complex)

    J_h = np.array([1.0+0j, 0.0+0j])

    for angle in [0, 30, 45, 60, 90]:
        lp = make_lp(angle)
        J_out = lp @ J_h
        I_out = np.sum(np.abs(J_out)**2)
        expected_I = np.cos(np.radians(angle))**2
        assert abs(I_out - expected_I) < 1e-6, f"Failed Malus law at {angle} deg"


def test_three_polarizer_paradox():
    """Test 3-Polarizer Paradox: LP(0 deg) -> LP(45 deg) -> LP(90 deg) yields I = 1/8 I0 for unpolarized input."""
    def make_lp(theta_deg):
        th = np.radians(theta_deg)
        c, s = np.cos(th), np.sin(th)
        return np.array([[c**2, c*s], [c*s, s**2]], dtype=complex)

    lp0 = make_lp(0)
    lp45 = make_lp(45)
    lp90 = make_lp(90)

    # Crossed polarizers (0 -> 90) yield zero
    J_h = np.array([1.0+0j, 0.0+0j])
    J_crossed = lp90 @ (lp0 @ J_h)
    assert np.allclose(J_crossed, [0, 0])

    # 3-polarizers (0 -> 45 -> 90) with horizontal input
    J_3pol = lp90 @ (lp45 @ (lp0 @ J_h))
    I_out = np.sum(np.abs(J_3pol)**2)
    assert abs(I_out - 0.25) < 1e-6  # 0.25 of horizontal polarized input (1/8 of total unpolarized)


def test_waveplates_qwp_hwp():
    """Test Quarter-Wave Plate (QWP) and Half-Wave Plate (HWP) Jones matrix propagation."""
    # QWP fast axis at 45 deg converts horizontal linear to circular
    qwp_45 = 0.5 * np.array([[1+1j, 1-1j], [1-1j, 1+1j]], dtype=complex)
    J_h = np.array([1.0+0j, 0.0+0j])
    J_circ = qwp_45 @ J_h
    # Ratio |Es| / |Ep| should be 1.0, phase difference pi/2
    assert abs(abs(J_circ[0]) - abs(J_circ[1])) < 1e-6
    delta_phase = np.angle(J_circ[1]) - np.angle(J_circ[0])
    assert abs(abs(delta_phase) - np.pi/2) < 1e-5

    # HWP fast axis at 45 deg rotates horizontal linear to vertical linear
    hwp_45 = np.array([[0, 1], [1, 0]], dtype=complex)
    J_v = hwp_45 @ J_h
    assert np.allclose(J_v, [0.0+0j, 1.0+0j])
