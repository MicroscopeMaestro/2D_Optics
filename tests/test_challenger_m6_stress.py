"""
Adversarial Physics & Polarization Engine Stress Tests (Requirement R5 & Milestone 6).
"""
import pytest
import numpy as np

from optics_app.engine.polarization import (
    compute_fresnel_coefficients, compute_sp_basis, rotate_jones_vector
)
from optics_app.engine.ray import Ray
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.utils.materials import get_complex_refractive_index
from optics_app.engine.gaussian import GaussianBeam, trace_system_gaussian_beam


def test_normal_and_grazing_incidence():
    # 1. Normal Incidence
    f_norm = compute_fresnel_coefficients(1.0, 1.5, 1.0)
    assert abs(f_norm.r_s - (-0.2)) < 1e-12
    assert abs(f_norm.r_p - 0.2) < 1e-12
    assert abs(f_norm.t_s - 0.8) < 1e-12
    assert abs(f_norm.t_p - 0.8) < 1e-12

    R_norm = abs(f_norm.r_s)**2
    T_norm = (1.5 / 1.0) * abs(f_norm.t_s)**2
    assert abs(R_norm + T_norm - 1.0) < 1e-12

    # 2. Grazing Incidence (up to 89.999 deg)
    for theta_deg in [89.0, 89.9, 89.99, 89.999]:
        cos_inc = np.cos(np.radians(theta_deg))
        f_graz = compute_fresnel_coefficients(1.0, 1.5, cos_inc)

        assert not np.isnan(f_graz.r_s) and not np.isinf(f_graz.r_s)
        assert not np.isnan(f_graz.r_p) and not np.isinf(f_graz.r_p)

        R_s = abs(f_graz.r_s)**2
        R_p = abs(f_graz.r_p)**2

        sin_t2 = (1.0 / 1.5) * np.sin(np.radians(theta_deg))
        cos_t2 = np.sqrt(1.0 - sin_t2**2)
        T_s = (1.5 * cos_t2) / (1.0 * cos_inc) * abs(f_graz.t_s)**2
        T_p = (1.5 * cos_t2) / (1.0 * cos_inc) * abs(f_graz.t_p)**2

        assert abs(R_s + T_s - 1.0) < 1e-5
        assert abs(R_p + T_p - 1.0) < 1e-5


def test_complex_metal_refractive_indices():
    metals = {
        "GOLD": 0.930718,
        "SILVER": 0.989656,
        "ALUMINUM": 0.90,  # approximate >0.85
        "COPPER": 0.60     # approximate >0.50
    }

    for key, min_r in metals.items():
        n_comp = get_complex_refractive_index(key)
        assert isinstance(n_comp, complex)
        assert n_comp.imag > 0

        n_val = n_comp.real
        k_val = n_comp.imag

        R_analytical = ((n_val - 1.0)**2 + k_val**2) / ((n_val + 1.0)**2 + k_val**2)
        f_metal = compute_fresnel_coefficients(1.0, n_comp, 1.0)
        R_fresnel_s = abs(f_metal.r_s)**2

        assert abs(R_fresnel_s - R_analytical) < 1e-6
        assert R_analytical > min_r


def test_tir_and_fresnel_rhomb():
    n1, n2 = 1.5, 1.0
    f_sub = compute_fresnel_coefficients(n1, n2, np.cos(np.radians(40)))
    assert not f_sub.tir
    assert abs(f_sub.r_s) < 1.0

    f_sup = compute_fresnel_coefficients(n1, n2, np.cos(np.radians(50)))
    assert f_sup.tir
    assert abs(abs(f_sup.r_s) - 1.0) < 1e-12
    assert abs(abs(f_sup.r_p) - 1.0) < 1e-12

    # Fresnel Rhomb (53.3 deg)
    f_rhomb = compute_fresnel_coefficients(n1, n2, np.cos(np.radians(53.3)))
    delta_single = np.degrees((np.angle(f_rhomb.r_p) - np.angle(f_rhomb.r_s)) % (2 * np.pi))
    delta_double = (2 * delta_single) % 360.0

    phase_mag_single = min(delta_single % 180, 180 - (delta_single % 180))
    phase_mag_double = min(delta_double % 180, 180 - (delta_double % 180))

    assert abs(phase_mag_single - 45.0) < 1.5
    assert abs(phase_mag_double - 90.0) < 3.0


def test_jones_calculus_polarizers_and_waveplates():
    def make_linear_polarizer(angle_deg):
        th = np.radians(angle_deg)
        c, s = np.cos(th), np.sin(th)
        return np.array([[c**2, c*s], [c*s, s**2]], dtype=complex)

    def make_qwp(fast_axis_deg):
        th = np.radians(fast_axis_deg)
        c, s = np.cos(th), np.sin(th)
        J_0 = np.array([[1.0, 0.0], [0.0, 1j]], dtype=complex)
        R = np.array([[c, -s], [s, c]], dtype=complex)
        return R @ J_0 @ R.T

    def make_hwp(fast_axis_deg):
        th = np.radians(fast_axis_deg)
        c, s = np.cos(th), np.sin(th)
        J_0 = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
        R = np.array([[c, -s], [s, c]], dtype=complex)
        return R @ J_0 @ R.T

    # Malus Law
    J_h = np.array([1.0+0j, 0.0+0j])
    for angle_deg in [0, 30, 45, 60, 90]:
        pol = make_linear_polarizer(angle_deg)
        J_out = pol @ J_h
        I_out = float(np.sum(np.abs(J_out)**2))
        I_expected = np.cos(np.radians(angle_deg))**2
        assert abs(I_out - I_expected) < 1e-12

    # 3-Polarizer Paradox
    p0 = make_linear_polarizer(0)
    p45 = make_linear_polarizer(45)
    p90 = make_linear_polarizer(90)

    J_3pol = p90 @ (p45 @ (p0 @ J_h))
    I_3pol = float(np.sum(np.abs(J_3pol)**2))
    assert abs(I_3pol - 0.25) < 1e-12

    # QWP 45 deg
    qwp45 = make_qwp(45)
    J_circ = qwp45 @ J_h
    assert abs(abs(J_circ[0]) - abs(J_circ[1])) < 1e-12

    # HWP 45 deg (rotates horizontal to vertical)
    hwp45 = make_hwp(45)
    J_vert = hwp45 @ J_h
    assert abs(J_vert[0]) < 1e-12
    assert abs(abs(J_vert[1]) - 1.0) < 1e-12


def test_gaussian_beam_q_propagation():
    w0 = 0.05
    wavelength = 0.6328
    n = 1.0

    beam = GaussianBeam(w0=w0, wavelength=wavelength, n=n, z=0.0)
    zR = beam.z_R

    beam.propagate_distance(zR)
    assert abs(beam.spot_radius - w0 * np.sqrt(2.0)) < 1e-6
    assert abs(beam.radius_of_curvature - 2.0 * zR) < 1e-4

    beam.pass_thin_lens(f=100.0)
    assert abs(beam.spot_radius - w0 * np.sqrt(2.0)) < 1e-6

    beam.reflect_mirror(radius=-200.0)
    assert beam.spot_radius > 0

    stack = SurfaceStack()
    stack.add_surface(Surface(name="Object", thickness=50.0))
    stack.add_surface(Surface(name="Lens1_Front", radius=50.0, thickness=5.0, glass="N-BK7"))
    stack.add_surface(Surface(name="Lens1_Back", radius=-50.0, thickness=100.0, glass="AIR"))
    stack.add_surface(Surface(name="Detector", surface_type=SurfaceType.DETECTOR, thickness=0.0))

    system = OpticalSystem(stack)
    trace_res = system.trace_gaussian_beam(w0=0.1, wavelength=0.5876, source_z=0.0)
    assert len(trace_res.nodes) == 4
    assert trace_res.final_w > 0
