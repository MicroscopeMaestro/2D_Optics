"""
Empirical Challenge Tests for Milestone 2 Gate Evaluation.
"""
import math
import numpy as np
import pytest

from optics_app.engine.surface import Surface, SurfaceType, SphericalSurface, AsphericSurface
from optics_app.engine.lde import SurfaceStack, LensDataEditor
from optics_app.engine.ray import Ray
from optics_app.engine.tracer import snell_refract_3d, ray_surface_intersect, OpticalSystem
from optics_app.engine.grating import grating_diffract_3d, DiffractionGrating
from optics_app.analysis.grating_dispersion import GratingDispersionAnalysis
from optics_app.engine.dispersion import SellmeierDispersion, CauchyDispersion, get_refractive_index
from optics_app.utils.materials import GlassCatalog
from optics_app.engine.ccd import CCDSensor, CCDMatrix
from optics_app.io.zmx_parser import parse_zmx_string, parse_zmx_file


def test_aspheric_surface_normal_correctness():
    """Empirical Test: Verify if AsphericSurface computes true surface normal instead of flat plane normal [0, 0, 1]."""
    radius = 50.0
    # Parabola k = -1.0 -> z(r) = 0.5 * c * r^2 = 0.5 * (1/50) * r^2
    asph_surf = AsphericSurface(radius=radius, conic=-1.0)
    ray = Ray(pos=np.array([10.0, 0.0, -10.0]), dir=np.array([0.0, 0.0, 1.0]))
    hit_pos, normal = asph_surf.intersect(ray)

    # For a parabolic surface at r=10, z(r) = 1.0.
    # dz/dx = x / R = 10 / 50 = 0.2
    # The true outward unit normal N = (-dz/dx, -dz/dy, 1) / sqrt(1 + (dz/dx)^2 + (dz/dy)^2)
    # N_x = -0.2 / sqrt(1 + 0.04) = -0.196116
    # N_z = 1.0 / sqrt(1 + 0.04) = 0.980581
    expected_normal_x = -0.2 / np.sqrt(1.04)
    expected_normal_z = 1.0 / np.sqrt(1.04)

    print(f"\nCalculated Normal: {normal}")
    print(f"Expected Normal: [{expected_normal_x:.6f}, 0.0, {expected_normal_z:.6f}]")

    # Assert true normal calculation (this will fail if hardcoded to [0, 0, 1])
    assert not np.allclose(normal, [0.0, 0.0, 1.0]), "AsphericSurface returned flat plane normal [0, 0, 1] instead of true surface normal!"
    assert np.isclose(normal[0], expected_normal_x, atol=1e-3), f"Normal X mismatch: got {normal[0]}, expected {expected_normal_x}"


def test_grating_3d_vector_dispersion_equation():
    """Empirical Test: Verify 3D off-axis ray vector grating diffraction equation."""
    line_density = 600.0  # lines/mm
    order = 1
    wl = 0.5  # um
    d_um = 1000.0 / line_density  # 1.666667 um

    # Incident ray with off-axis Y component: dir = [0.0, 0.6, 0.8]
    v_in = np.array([0.0, 0.6, 0.8])
    v_out, evan = grating_diffract_3d(v_in, [0, 0, 1], [1, 0, 0], line_density, order, wl)

    assert not evan
    assert v_out is not None

    # In vector grating diffraction with grating vector along X:
    # v_out_x must strictly equal v_in_x + (order * wl / d_um) = 0.0 + (1 * 0.5 / 1.666667) = 0.3
    # v_out_y must equal v_in_y = 0.6
    # v_out_z must equal sqrt(1 - v_out_x^2 - v_out_y^2) = sqrt(1 - 0.09 - 0.36) = sqrt(0.55) = 0.7416198
    expected_x = 0.3
    expected_y = 0.6
    expected_z = np.sqrt(1.0 - expected_x**2 - expected_y**2)

    print(f"\n3D Grating v_out: {v_out}")
    print(f"Expected v_out: [{expected_x}, {expected_y}, {expected_z:.6f}]")

    assert np.isclose(v_out[0], expected_x, atol=1e-5), f"Grating X dispersion violated: got {v_out[0]}, expected {expected_x}"
    assert np.isclose(v_out[1], expected_y, atol=1e-5), f"Grating Y component modified: got {v_out[1]}, expected {expected_y}"
    assert np.isclose(v_out[2], expected_z, atol=1e-5), f"Grating Z component incorrect: got {v_out[2]}, expected {expected_z}"


def test_grating_sign_convention_consistency():
    """Empirical Test: Verify consistency between grating_diffract_3d and DiffractionGrating."""
    g = DiffractionGrating(lines_per_mm=600.0, order=1)
    wl = 0.5
    inc_angle_deg = 30.0
    inc_angle_rad = np.radians(inc_angle_deg)

    # DiffractionGrating method
    theta_m_calc = g.calculate_diffraction_angle(inc_angle_rad, wl)
    sin_m_calc = np.sin(theta_m_calc)

    # grating_diffract_3d method
    v_in = np.array([np.sin(inc_angle_rad), 0.0, np.cos(inc_angle_rad)])
    v_out, evan = grating_diffract_3d(v_in, [0, 0, 1], [1, 0, 0], 600.0, 1, wl)
    sin_m_3d = v_out[0]

    print(f"\nDiffractionGrating sin_m: {sin_m_calc}")
    print(f"grating_diffract_3d sin_m: {sin_m_3d}")

    # Standard grating equation: sin(theta_m) = sin(theta_i) + m * lambda / d
    # Or sin(theta_m) = m * lambda / d - sin(theta_i) depending on sign convention of incident angle.
    # But BOTH methods in the codebase MUST be consistent!
    assert np.isclose(sin_m_calc, sin_m_3d, atol=1e-5), f"Sign convention mismatch: DiffractionGrating got {sin_m_calc}, grating_diffract_3d got {sin_m_3d}"


def test_surface_tilts_in_ray_tracing():
    """Empirical Test: Verify if surface tilt angles tilt_x, tilt_y, tilt_z are applied in OpticalSystem.trace_ray."""
    surf = Surface(name="Tilted Mirror", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=10.0, tilt_x=45.0)
    stack = SurfaceStack()
    stack.add_surface(surf)
    sys = OpticalSystem(stack)

    ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]))
    traced = sys.trace_ray(ray)

    # If surface is tilted by 45 degrees in X, a ray incident along Z should be reflected or refracted at an angle!
    # If tilts are ignored, ray direction will remain [0, 0, 1].
    print(f"\nTraced ray direction after tilted surface: {traced.dir}")
    assert not np.allclose(traced.dir, [0.0, 0.0, 1.0]), "Surface tilt_x was ignored during ray tracing!"
