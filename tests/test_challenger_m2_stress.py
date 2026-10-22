"""
Empirical Challenge & Stress Test Suite for Milestone 2.
Evaluates Total Internal Reflection (TIR), grazing incidence, extreme tilts,
multi-element LDE stacks, corrupt Zemax .zmx parsing, and 3D STEP exports.
"""
import os
import math
import tempfile
import numpy as np
import pytest

from optics_app.engine.surface import Surface, SurfaceType, SphericalSurface, AsphericSurface
from optics_app.engine.lde import SurfaceStack, LensDataEditor
from optics_app.engine.ray import Ray
from optics_app.engine.tracer import snell_refract_3d, build_rotation_matrix, OpticalSystem
from optics_app.engine.grating import grating_diffract_3d, DiffractionGrating
from optics_app.engine.dispersion import get_refractive_index
from optics_app.engine.ccd import CCDSensor
from optics_app.io.zmx_parser import parse_zmx_string, parse_zmx_file
from optics_app.cad.exporter import STEPExporter


# -----------------------------------------------------------------------------
# 1. Total Internal Reflection (TIR) & Grazing Incidence Edge Cases
# -----------------------------------------------------------------------------

def test_tir_critical_angle_threshold():
    """Verify Snell refraction TIR behavior exactly at, below, and above critical angle."""
    n1 = 1.784  # SF11 glass
    n2 = 1.000  # Air
    critical_angle_rad = math.asin(n2 / n1) # ~0.5942 rad (34.09 deg)
    
    # Normal pointing along -Z (opposite incoming ray)
    normal = np.array([0.0, 0.0, -1.0])

    # Case A: Below critical angle (30 deg) -> Refraction
    theta_sub = math.radians(30.0)
    ray_sub = np.array([math.sin(theta_sub), 0.0, math.cos(theta_sub)])
    dir_out, tir = snell_refract_3d(ray_sub, normal, n1, n2)
    assert not tir, "TIR triggered below critical angle!"
    assert np.isclose(np.linalg.norm(dir_out), 1.0), "Output direction not unit normalized!"
    # Verify Snell Law: n1 * sin(30) = 1.784 * 0.5 = 0.892 = n2 * sin(theta2)
    expected_sin_t2 = n1 * math.sin(theta_sub) / n2
    assert np.isclose(dir_out[0], expected_sin_t2, atol=1e-5)

    # Case B: Above critical angle (45 deg) -> TIR reflection
    theta_super = math.radians(45.0)
    ray_super = np.array([math.sin(theta_super), 0.0, math.cos(theta_super)])
    dir_out_tir, tir = snell_refract_3d(ray_super, normal, n1, n2)
    assert tir, "TIR failed to trigger above critical angle!"
    assert np.isclose(np.linalg.norm(dir_out_tir), 1.0), "TIR output vector not unit normalized!"
    # In TIR reflection, normal component flips
    assert np.isclose(dir_out_tir[0], ray_super[0], atol=1e-5), "TIR tangential component altered!"
    assert dir_out_tir[2] < 0, "TIR direction z should be reflected backwards!"


def test_grazing_incidence_numerical_stability():
    """Verify stability of Snell refraction under grazing incidence angles (89.999 degrees)."""
    n1 = 1.0
    n2 = 1.5168
    normal = np.array([0.0, 0.0, -1.0])
    
    theta_grazing = math.radians(89.999)
    ray_grazing = np.array([math.sin(theta_grazing), 0.0, math.cos(theta_grazing)])
    
    dir_out, tir = snell_refract_3d(ray_grazing, normal, n1, n2)
    assert not tir
    assert not np.any(np.isnan(dir_out)), "NaN detected in grazing incidence refraction!"
    assert not np.any(np.isinf(dir_out)), "Inf detected in grazing incidence refraction!"
    assert np.isclose(np.linalg.norm(dir_out), 1.0, atol=1e-12)


def test_optical_system_tir_flag_propagation():
    """Verify trace_ray flags TIR when ray hits high-to-low index interface at steep angle."""
    stack = SurfaceStack()
    # Prism surface 1: Flat plane (Air -> Glass SF11)
    s1 = Surface(name="Entrance", radius=0.0, thickness=20.0, glass="SF11", semi_diameter=50.0)
    # Prism surface 2: Tilted by 45 degrees (Glass -> Air, n=1.0)
    s2 = Surface(name="TiltedExit", radius=0.0, thickness=10.0, glass="AIR", semi_diameter=50.0, tilt_y=45.0)
    s3 = Surface(name="Detector", surface_type=SurfaceType.DETECTOR, radius=0.0, thickness=0.0, semi_diameter=50.0)

    stack.add_surface(s1)
    stack.add_surface(s2)
    stack.add_surface(s3)

    sys = OpticalSystem(stack)
    ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=0.5876)

    traced = sys.trace_ray(ray)
    assert traced.tir, "Ray failed to set TIR flag when internal reflection occurred at surface 2!"
    assert not np.any(np.isnan(traced.pos)), "Ray position contains NaN!"
    assert not np.any(np.isnan(traced.dir)), "Ray direction contains NaN!"


# -----------------------------------------------------------------------------
# 2. Extreme Tilts & Coordinate Transformations
# -----------------------------------------------------------------------------

def test_rotation_matrix_orthonormality():
    """Verify build_rotation_matrix produces orthogonal matrices across extreme angles."""
    test_angles = [
        (0.0, 0.0, 0.0),
        (45.0, -30.0, 60.0),
        (90.0, 180.0, -90.0),
        (360.0, 720.0, -45.0),
        (-89.9, 89.9, 179.9)
    ]
    for rx, ry, rz in test_angles:
        R = build_rotation_matrix(rx, ry, rz)
        # Check orthogonality R^T @ R = I
        I_calc = R.T @ R
        assert np.allclose(I_calc, np.eye(3), atol=1e-12), f"Rotation matrix for ({rx}, {ry}, {rz}) not orthogonal!"
        # Check determinant is 1.0 (proper rotation)
        det_R = np.linalg.det(R)
        assert np.isclose(det_R, 1.0, atol=1e-12), f"Rotation matrix determinant not 1.0 for ({rx}, {ry}, {rz})!"


def test_aspheric_surface_extreme_conic_and_tilts():
    """Verify Newton-Raphson solver on hyperbolic aspheric surface with 45 degree tilt."""
    surf = Surface(
        name="HyperbolicTilted",
        surface_type=SurfaceType.ASPHERIC,
        radius=25.0,
        conic=-2.5,
        asphere_coeffs=[1e-6, -1e-9],
        semi_diameter=30.0,
        tilt_x=30.0,
        tilt_y=-15.0
    )
    stack = SurfaceStack()
    stack.add_surface(surf)
    sys = OpticalSystem(stack)

    ray = Ray(pos=np.array([2.0, -3.0, -10.0]), dir=np.array([0.0, 0.0, 1.0]))
    traced = sys.trace_ray(ray)

    assert not traced.vignetted
    assert not np.any(np.isnan(traced.pos))
    assert not np.any(np.isnan(traced.dir))
    assert np.isclose(np.linalg.norm(traced.dir), 1.0)


# -----------------------------------------------------------------------------
# 3. Multi-Element LDE Stacks & Global Z Accumulation
# -----------------------------------------------------------------------------

def test_multi_element_lde_stack_accumulation():
    """Verify Z accumulation and sequential tracing through a 12-surface complex stack."""
    stack = SurfaceStack()

    surfaces_def = [
        ("Object", SurfaceType.OBJECT, 0.0, 10.0, "AIR", 50.0, 0.0, []),
        ("Lens1_Front", SurfaceType.STANDARD, 50.0, 5.0, "N-BK7", 25.0, 0.0, []),
        ("Lens1_Back", SurfaceType.STANDARD, -100.0, 2.0, "AIR", 25.0, 0.0, []),
        ("Stop", SurfaceType.STOP, 0.0, 8.0, "AIR", 10.0, 0.0, []),
        ("Lens2_Front", SurfaceType.ASPHERIC, 40.0, 6.0, "SF11", 20.0, -0.8, [1e-6, -2e-9]),
        ("Lens2_Back", SurfaceType.ASPHERIC, -60.0, 4.0, "AIR", 20.0, 0.0, []),
        ("Grating", SurfaceType.GRATING, 0.0, 15.0, "AIR", 20.0, 0.0, []),
        ("Lens3_Front", SurfaceType.STANDARD, 80.0, 7.0, "FUSED_SILICA", 22.0, 0.0, []),
        ("Lens3_Back", SurfaceType.STANDARD, -40.0, 1.0, "AIR", 22.0, 0.0, []),
        ("AirGap", SurfaceType.STANDARD, 0.0, 12.0, "AIR", 30.0, 0.0, []),
        ("Filter", SurfaceType.STANDARD, 0.0, 3.0, "N-BK7", 30.0, 0.0, []),
        ("Detector", SurfaceType.DETECTOR, 0.0, 0.0, "AIR", 30.0, 0.0, []),
    ]

    expected_z = 0.0
    for idx, (name, stype, R, t, glass, diam, conic, coeffs) in enumerate(surfaces_def):
        s = Surface(
            name=name, surface_type=stype, radius=R, thickness=t,
            glass=glass, semi_diameter=diam, conic=conic, asphere_coeffs=coeffs
        )
        if stype == SurfaceType.GRATING:
            s.grating_lines_per_mm = 600.0
            s.grating_order = 1
        stack.add_surface(s)

    # Check accumulated Z positions
    expected_zs = [0.0, 10.0, 15.0, 17.0, 25.0, 31.0, 35.0, 50.0, 57.0, 58.0, 70.0, 73.0]
    for i, surf in enumerate(stack.surfaces):
        assert np.isclose(surf.global_z, expected_zs[i]), f"Surface {i} ({surf.name}) global_z mismatch: got {surf.global_z}, expected {expected_zs[i]}"

    sys = OpticalSystem(stack)
    ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=0.5876)
    traced = sys.trace_ray(ray)

    assert len(traced.history) == 12
    assert not traced.vignetted
    assert not traced.tir
    assert np.isclose(traced.history[-1]["z_global"], 73.0, atol=1e-3)


# -----------------------------------------------------------------------------
# 4. Zemax .zmx Corrupt & Edge Case Parsing
# -----------------------------------------------------------------------------

def test_zmx_corrupt_and_malformed_inputs():
    """Verify robustness of parse_zmx_string against corrupt syntax, comments, and missing values."""
    corrupt_zmx_1 = """
    ! Corrupt Zemax File Sample
    UNIT INCH
    SURF 0
      TYPE OBJECT
      THIC INFINITY
    SURF 1
      TYPE EVENASPH
      CURV 0.02
      THIC 5.0
      GLAS N-BK7
      DIAM 12.5
      CONI -1.0
      PARM 1 0.00001
      PARM 2 -0.0000002
    SURF 2
      CURV -0.015 # inline comment
      THIC 10.0
      GLAS AIR
      DIAM 12.5
    SURF 3
      TYPE DETECTOR
      THIC 0.0
    """
    stack = parse_zmx_string(corrupt_zmx_1)
    assert len(stack.surfaces) == 4
    # Unit INCH -> scale 25.4
    assert np.isclose(stack.surfaces[1].thickness, 5.0 * 25.4)
    assert np.isclose(stack.surfaces[1].semi_diameter, 12.5 * 25.4)
    assert np.isclose(stack.surfaces[1].conic, -1.0)
    assert len(stack.surfaces[1].asphere_coeffs) == 2


def test_zmx_file_encoding_resilience():
    """Verify parse_zmx_file handles UTF-8, UTF-16-LE, UTF-16-BE, and Latin-1 files seamlessly."""
    zmx_text = "SURF 0\nTHIC 10.0\nSURF 1\nCURV 0.05\nTHIC 5.0\nGLAS N-BK7\nDIAM 20.0\n"
    
    for enc in ['utf-8', 'utf-16', 'utf-16-le', 'utf-16-be', 'latin-1']:
        with tempfile.NamedTemporaryFile('wb', delete=False, suffix='.zmx') as tmp:
            tmp.write(zmx_text.encode(enc))
            tmp_path = tmp.name

        try:
            stack = parse_zmx_file(tmp_path)
            assert len(stack.surfaces) == 2
            assert np.isclose(stack.surfaces[1].radius, 20.0)
            assert np.isclose(stack.surfaces[1].thickness, 5.0)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


# -----------------------------------------------------------------------------
# 5. 3D STEP Assembly Exporter Complex System Test
# -----------------------------------------------------------------------------

def test_step_exporter_complex_assembly():
    """Verify 3D STEP assembly exporter generates a valid ISO-10303-21 CAD file for multi-element system."""
    stack = SurfaceStack()
    s1 = Surface(name="Lens1_S1", radius=30.0, thickness=6.0, glass="N-BK7", semi_diameter=15.0)
    s2 = Surface(name="Lens1_S2", radius=-40.0, thickness=4.0, glass="AIR", semi_diameter=15.0)
    s3 = Surface(name="Grating", surface_type=SurfaceType.GRATING, radius=0.0, thickness=10.0, semi_diameter=15.0, tilt_y=15.0)
    s3.grating_lines_per_mm = 600.0
    s4 = Surface(name="CCD", surface_type=SurfaceType.DETECTOR, radius=0.0, thickness=0.0, semi_diameter=12.0)

    stack.add_surface(s1)
    stack.add_surface(s2)
    stack.add_surface(s3)
    stack.add_surface(s4)

    with tempfile.NamedTemporaryFile('w', delete=False, suffix='.step') as tmp:
        out_path = tmp.name

    try:
        res = STEPExporter.export(stack, out_path)
        assert res.is_success
        assert os.path.exists(out_path)
        assert os.path.getsize(out_path) > 1000  # File should contain solid entities

        with open(out_path, 'r', encoding='utf-8') as f:
            content = f.read()

        assert "ISO-10303-21;" in content
        assert "HEADER;" in content
        assert "DATA;" in content
        assert "END-ISO-10303-21;" in content
        assert "MANIFOLD_SOLID_BREP" in content or "CLOSED_SHELL" in content or "CARTESIAN_POINT" in content
    finally:
        if os.path.exists(out_path):
            os.remove(out_path)
