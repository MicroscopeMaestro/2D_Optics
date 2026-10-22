"""
Tier 5 White-Box Adversarial Hardening Test Suite.
Covers multi-surface Total Internal Reflection (TIR) ray termination, extreme Sellmeier wavelength limits,
zero-radius mirrors, complex CAD B-spline fitting residuals, 3D Euler tilts/decenters, grating evanescent limits,
Zemax .zmx parser edge cases, and 3D STEP assembly export.
"""

import os
import tempfile
import math
import numpy as np
import pytest

from optics_app.engine.surface import Surface, SurfaceType, SphericalSurface, AsphericSurface
from optics_app.engine.ray import Ray
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem, snell_refract_3d, build_rotation_matrix
from optics_app.engine.dispersion import SellmeierDispersion, CauchyDispersion, get_refractive_index
from optics_app.engine.grating import grating_diffract_3d, DiffractionGrating
from optics_app.utils.materials import GlassCatalog
from optics_app.cad.fitter import SagFitter, fit_sag_profile, ProfilePoints
from optics_app.cad.profile import OpticalProfile, FitType
from optics_app.cad.exporter import STEPExporter, STEPExportOptions, STEPExportEngine, export_to_step
from optics_app.io.zmx_parser import parse_zmx_string, parse_zmx_file
from optics_app.analysis.spot import SpotDiagram, calculate_spot_metrics, generate_spot_diagram_from_system
from optics_app.analysis.rayfan import RayFanAnalysis, calculate_ray_fan
from optics_app.analysis.aberration import calculate_seidel_coefficients, SeidelAberrations


def test_tier5_multi_surface_tir_termination():
    """
    Adversarial Test 1: Multi-surface Total Internal Reflection (TIR) ray propagation and termination.
    Rays entering glass (n=1.5) at steep angles relative to internal interfaces encounter TIR.
    """
    stack = SurfaceStack()
    stack.add_surface(Surface(name="Object", surface_type=SurfaceType.OBJECT, thickness=10.0))
    # Front face of glass block
    stack.add_surface(Surface(name="S1_Front", radius=0.0, thickness=20.0, glass="BK7", semi_diameter=50.0))
    # Straight normal glass-air surface (tilt 0 deg) -> No TIR
    stack.add_surface(Surface(name="S2_Straight", radius=0.0, thickness=20.0, glass="AIR", tilt_x=0.0, semi_diameter=50.0))
    stack.add_surface(Surface(name="Detector", surface_type=SurfaceType.DETECTOR, semi_diameter=100.0))

    system = OpticalSystem(stack)

    # Ray 1: On-axis normal ray (no TIR at 0 deg interface)
    r1 = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=0.5876)
    tr1 = system.trace_ray(r1)
    assert not tr1.tir, "On-axis ray at normal interface should not experience TIR"

    # Ray 2: Glass-air interface tilted 45 deg causing TIR
    tir_stack = SurfaceStack()
    tir_stack.add_surface(Surface(name="Object", surface_type=SurfaceType.OBJECT, thickness=10.0))
    tir_stack.add_surface(Surface(name="S1_Front", radius=0.0, thickness=20.0, glass="BK7", semi_diameter=50.0))
    tir_stack.add_surface(Surface(name="S2_AngledTIR", radius=0.0, thickness=20.0, glass="AIR", tilt_x=45.0, semi_diameter=50.0))
    tir_stack.add_surface(Surface(name="Detector", surface_type=SurfaceType.DETECTOR, semi_diameter=100.0))

    sys_tir = OpticalSystem(tir_stack)
    r2 = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=0.5876)
    tr2 = sys_tir.trace_ray(r2)

    assert tr2.tir, "Ray at 45-deg glass-to-air interface must trigger TIR flag"
    assert len(tr2.history) > 0, "Ray history should record surface interactions"
    s2_hist = next((h for h in tr2.history if h["surface_index"] == 2), None)
    assert s2_hist is not None, "Surface index 2 should be in history"
    assert s2_hist["tir"] is True, "Surface index 2 history entry should record tir=True"

    # Vector test for snell_refract_3d TIR return
    # Glass (n=1.5168) to Air (n=1.0) with normal (0,0,1) and ray dir (0, sin(50deg), cos(50deg))
    v_in = np.array([0.0, math.sin(math.radians(50.0)), math.cos(math.radians(50.0))])
    normal = np.array([0.0, 0.0, 1.0])
    v_ref, is_tir = snell_refract_3d(v_in, normal, n1=1.5168, n2=1.0)
    assert is_tir is True, "50-deg incidence glass-to-air must be TIR"
    # Specular reflection direction: v_y remains positive, v_z reverses to negative
    assert math.isclose(np.linalg.norm(v_ref), 1.0, abs_tol=1e-9), "Reflected ray must be unit vector"
    assert v_ref[2] < 0, "Z-component of reflected ray should reverse direction"


def test_tier5_extreme_sellmeier_wavelength_limits():
    """
    Adversarial Test 2: Extreme Sellmeier wavelength limits, resonance poles, and catalog aliases.
    """
    # 1. Standard BK7 Sellmeier calculation
    bk7 = GlassCatalog.get_glass("BK7")
    n_d = bk7.get_index(0.58756)
    assert 1.51 < n_d < 1.53, f"BK7 index at d-line expected ~1.5168, got {n_d}"

    # 2. Resonance pole rejection
    # BK7 has poles C1=0.00600069867 (wl ~ 0.0774 um), C2=0.0200179144 (wl ~ 0.14148 um), C3=103.560653 (wl ~ 10.176 um)
    pole_wl = math.sqrt(0.00600069867)
    with pytest.raises(ValueError, match="resonance pole"):
        bk7.get_index(pole_wl)

    # 3. Cauchy zero wavelength rejection
    cauchy = CauchyDispersion(A=1.5, B=0.005)
    with pytest.raises(ValueError, match="Wavelength cannot be zero"):
        cauchy.get_index(0.0)

    # 4. Glass catalog lookup case-insensitivity and whitespace stripping
    glass_aliases = ["bk7", " N-BK7 ", "fused_silica", " F_SILICA ", "MIRROR", "AIR", "VACUUM"]
    for alias in glass_aliases:
        g = GlassCatalog.get_glass(alias)
        assert g is not None, f"Failed to resolve alias '{alias}'"

    # 5. Unknown glass material raises ValueError
    with pytest.raises(ValueError, match="Unknown glass material"):
        GlassCatalog.get_glass("UNOBTAINIUM_99")

    # 6. Abbe number computation
    v_d = GlassCatalog.calculate_abbe_number("BK7")
    assert 63.0 < v_d < 65.0, f"BK7 Abbe number expected ~64.17, got {v_d}"


def test_tier5_zero_radius_and_curved_mirrors():
    """
    Adversarial Test 3: Zero-radius planar mirrors, concave mirrors, convex mirrors, and 3D tilts.
    """
    # 1. Flat planar mirror (radius = 0.0)
    flat_mirror_stack = SurfaceStack()
    flat_mirror_stack.add_surface(Surface(name="Object", thickness=10.0))
    flat_mirror_stack.add_surface(Surface(name="FlatMirror", radius=0.0, glass="MIRROR", thickness=10.0, semi_diameter=50.0))
    flat_mirror_stack.add_surface(Surface(name="Detector", surface_type=SurfaceType.DETECTOR, semi_diameter=50.0))

    sys_flat = OpticalSystem(flat_mirror_stack)
    ray_inc = Ray(pos=np.array([0.0, 10.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=0.5876)
    tr_flat = sys_flat.trace_ray(ray_inc)
    assert not tr_flat.vignetted, "Flat mirror ray should trace successfully"
    # Reflected direction at flat mirror should be -Z
    last_hit = tr_flat.history[1]  # FlatMirror hit
    assert math.isclose(last_hit["dir_z"], -1.0, abs_tol=1e-6), f"Flat mirror reflection Z dir should be -1.0, got {last_hit['dir_z']}"

    # 2. Concave spherical mirror (radius = 100.0 mm centered at z = +100)
    concave_stack = SurfaceStack()
    concave_stack.add_surface(Surface(name="Object", thickness=0.0))
    concave_stack.add_surface(Surface(name="ConcaveMirror", radius=100.0, glass="MIRROR", thickness=-50.0, semi_diameter=50.0))

    sys_concave = OpticalSystem(concave_stack)
    # Parallel marginal ray at height y = 10 mm
    ray_para = Ray(pos=np.array([0.0, 10.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=0.5876)
    tr_conc = sys_concave.trace_ray(ray_para)
    assert not tr_conc.vignetted, "Concave mirror ray trace should succeed"
    # Verify ray reflects backwards towards negative Z
    dir_reflect = tr_conc.dir
    assert dir_reflect[2] < 0.0, "Concave mirror reflection Z direction must be negative (backwards)"

    # 3. Tilted planar mirror (30 degree tilt around X axis)
    tilted_stack = SurfaceStack()
    tilted_stack.add_surface(Surface(name="Object", thickness=10.0))
    tilted_stack.add_surface(Surface(name="TiltedMirror", radius=0.0, glass="MIRROR", tilt_x=30.0, thickness=10.0, semi_diameter=50.0))

    sys_tilted = OpticalSystem(tilted_stack)
    ray_tilt = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=0.5876)
    tr_tilted = sys_tilted.trace_ray(ray_tilt)
    # Reflection angle at 30 deg tilted mirror: angle of incidence = 30 deg, deviation = 60 deg
    dir_out = tr_tilted.dir
    expected_y_dir = math.sin(math.radians(60.0))
    expected_z_dir = -math.cos(math.radians(60.0))
    assert math.isclose(dir_out[1], expected_y_dir, abs_tol=1e-5), f"Expected Y dir {expected_y_dir}, got {dir_out[1]}"
    assert math.isclose(dir_out[2], expected_z_dir, abs_tol=1e-5), f"Expected Z dir {expected_z_dir}, got {dir_out[2]}"


def test_tier5_complex_cad_bspline_fitting_residuals():
    """
    Adversarial Test 4: Complex CAD B-spline fitting, non-monotonic points, noise, and conic fitting.
    """
    # 1. Noisy, non-monotonic radial data points
    r_raw = np.array([0.0, 5.0, 2.0, 8.0, 8.0, 12.0, 15.0, 20.0])
    z_raw = r_raw**2 / 100.0 + np.array([0.0, 0.01, -0.005, 0.02, 0.02, -0.01, 0.015, 0.0])

    spline_obj, rms_um = SagFitter.fit_spline(r_raw, z_raw)
    assert spline_obj is not None, "Spline fitting must return valid UnivariateSpline"
    assert rms_um >= 0.0, "RMS residual must be non-negative"

    # 2. Perfect paraboloid sag z = r^2 / (2 * 100.0) -> radius R = 100.0, conic k = -1.0
    r_parab = np.linspace(0.0, 25.0, 51)
    z_parab = r_parab**2 / 200.0

    fit_res = fit_sag_profile(np.column_stack([r_parab, z_parab]), fit_type='conic')
    assert math.isclose(fit_res["radius"], 100.0, rel_tol=1e-3), f"Expected radius 100.0, got {fit_res['radius']}"
    assert math.isclose(fit_res["conic"], -1.0, abs_tol=1e-3), f"Expected conic constant -1.0, got {fit_res['conic']}"

    # 3. High-order polynomial asphere profile fit
    r_asph = np.linspace(0.0, 20.0, 41)
    z_asph = r_asph**2 / 200.0 + 1e-6 * r_asph**4 - 1e-9 * r_asph**6
    asph_res = fit_sag_profile(np.column_stack([r_asph, z_asph]), fit_type='aspheric')
    assert asph_res["rms_residual_um"] < 1.0, f"Aspheric fit RMS residual should be < 1.0 um, got {asph_res['rms_residual_um']}"

    # 4. SagFitter fallback on single / minimal points
    r_tiny = np.array([0.0, 5.0])
    z_tiny = np.array([0.0, 0.125])
    c_tiny, R_tiny = SagFitter.fit_algebraic_sphere(r_tiny, z_tiny)
    assert R_tiny > 0, "Algebraic sphere fit should return positive default radius for minimal points"


def test_tier5_extreme_decenter_tilt_and_asphere():
    """
    Adversarial Test 5: Extreme decenters, 180-degree 3D Euler flips, and aspheric surfaces.
    """
    # 1. 3D Euler Rotation matrix orthogonality check
    R_mat = build_rotation_matrix(45.0, -30.0, 60.0)
    det = np.linalg.det(R_mat)
    assert math.isclose(det, 1.0, abs_tol=1e-7), f"Rotation matrix determinant must be 1.0, got {det}"
    assert np.allclose(R_mat.T @ R_mat, np.eye(3), atol=1e-7), "Rotation matrix must be orthogonal"

    # 2. Surface with extreme decenters and 180-deg tilt flip
    surf_flip = Surface(
        name="FlippedSurf",
        radius=100.0,
        thickness=10.0,
        glass="BK7",
        decenter_x=50.0,
        decenter_y=-25.0,
        tilt_x=180.0,
        semi_diameter=50.0
    )
    ray = Ray(pos=np.array([50.0, -25.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=0.5876)
    stack = SurfaceStack()
    stack.add_surface(Surface(name="Object", thickness=10.0))
    stack.add_surface(surf_flip)
    stack.add_surface(Surface(name="Detector", surface_type=SurfaceType.DETECTOR))

    system = OpticalSystem(stack)
    tr_flip = system.trace_ray(ray)
    assert len(tr_flip.history) > 0, "Flipped surface should interact with ray"

    # 3. Newton-Raphson aspheric surface intersection solver
    asph_surf = AsphericSurface(radius=50.0, conic=-1.0, asphere_coeffs=[1e-5, -1e-8])
    sag_val = asph_surf.sag(10.0)
    assert sag_val > 0, "Sag at r=10mm for convex parabola must be > 0"
    normal_vec = asph_surf.normal(np.array([10.0, 0.0, sag_val]))
    assert math.isclose(np.linalg.norm(normal_vec), 1.0, abs_tol=1e-7), "Normal vector must be normalized"


def test_tier5_grating_evanescent_wave_boundaries():
    """
    Adversarial Test 6: Diffraction grating 3D equations, orders m=0,1,-1, and evanescent waves.
    """
    # 1. Standard 600 lines/mm transmission grating at 0.6328 um
    grating = DiffractionGrating(lines_per_mm=600.0, order=1)
    angle_diff = grating.calculate_diffraction_angle(incident_angle_rad=0.0, wavelength_um=0.6328)
    # sin(theta_m) = 1 * 0.6328 / (1000 / 600) = 0.37968 -> theta_m ~ 22.31 deg
    expected_deg = math.degrees(math.asin(0.37968))
    assert math.isclose(math.degrees(angle_diff), expected_deg, abs_tol=1e-3)

    # 2. Evanescent wave detection: 3600 lines/mm at 1.0 um -> d = 0.2777 um
    # sin(theta_m) = 1 * 1.0 / 0.2777 = 3.6 > 1.0 -> evanescent wave
    grating_high = DiffractionGrating(lines_per_mm=3600.0, order=1)
    with pytest.raises(ValueError, match="Evanescent order"):
        grating_high.calculate_diffraction_angle(incident_angle_rad=0.0, wavelength_um=1.0)

    # 3. Vector 3D grating solver evanescent flag
    v_out, is_evan = grating_diffract_3d(
        ray_dir=[0.0, 0.0, 1.0],
        grating_normal=[0.0, 0.0, 1.0],
        line_vector=[1.0, 0.0, 0.0],
        line_density_lines_per_mm=3600.0,
        order=1,
        wavelength_um=1.0
    )
    assert is_evan is True, "Vector solver must set evanescent flag to True for non-propagating order"
    assert v_out is None, "Evanescent wave output vector should be None"

    # 4. Zero order (m=0) specular transmission
    v_out_0, is_evan_0 = grating_diffract_3d(
        ray_dir=[0.0, 0.5, 0.8660254],
        grating_normal=[0.0, 0.0, 1.0],
        line_vector=[1.0, 0.0, 0.0],
        line_density_lines_per_mm=600.0,
        order=0,
        wavelength_um=0.5876
    )
    assert not is_evan_0, "Order m=0 must propagate"
    assert np.allclose(v_out_0, [0.0, 0.5, 0.8660254], atol=1e-5), "Order m=0 direction must equal incident ray direction"


def test_tier5_zmx_parser_token_edge_cases():
    """
    Adversarial Test 7: Zemax .zmx prescription parser with unit conversions, comments, and aspheric fields.
    """
    zmx_content = """
    ! Zemax Prescription Test File
    # Comment line
    UNIT INCH
    SURF 0
      TYPE OBJECT
      THIC INFINITY
    SURF 1
      TYPE EVENASPH
      CURV 0.02
      THIC 0.5
      GLAS BK7
      DIAM 1.0
      CONI -0.5
      PARM 1 1e-4
      PARM 2 -5e-7
      DISP 0.1 0.2 5.0 10.0 0.0
    SURF 2
      TYPE DETECTOR
      THIC 0.0
      DIAM 2.0
    """

    stack = parse_zmx_string(zmx_content)
    assert len(stack.surfaces) == 3, f"Expected 3 surfaces, got {len(stack.surfaces)}"

    # Check unit conversion INCH -> MM (scale factor 25.4)
    s1 = stack.surfaces[1]
    assert s1.surface_type == SurfaceType.ASPHERIC
    # CURV 0.02 in 1/INCH -> curv_scaled = 0.02 / 25.4 1/mm -> radius = 25.4 / 0.02 = 1270.0 mm
    assert math.isclose(s1.radius, 1270.0, rel_tol=1e-3), f"Expected radius 1270.0 mm, got {s1.radius}"
    # THIC 0.5 inch -> 12.7 mm
    assert math.isclose(s1.thickness, 12.7, rel_tol=1e-3), f"Expected thickness 12.7 mm, got {s1.thickness}"
    # DIAM 1.0 inch -> semi-diameter 25.4 mm
    assert math.isclose(s1.semi_diameter, 25.4, rel_tol=1e-3), f"Expected semi-diameter 25.4 mm, got {s1.semi_diameter}"
    assert math.isclose(s1.conic, -0.5, abs_tol=1e-5)
    assert len(s1.asphere_coeffs) >= 2
    assert s1.asphere_coeffs[0] == 1e-4
    assert s1.asphere_coeffs[1] == -5e-7

    # Check DISP decenters (scaled by 25.4) and tilts (in degrees)
    assert math.isclose(s1.decenter_x, 2.54, rel_tol=1e-3)
    assert math.isclose(s1.decenter_y, 5.08, rel_tol=1e-3)
    assert math.isclose(s1.tilt_x, 5.0, abs_tol=1e-5)
    assert math.isclose(s1.tilt_y, 10.0, abs_tol=1e-5)


def test_tier5_analysis_diagnostics_edge_cases():
    """
    Adversarial Test 8: Zemax-style diagnostic analysis edge cases (SpotDiagram, RayFan, Seidel Aberrations).
    """
    # 1. SpotDiagram empty arrays
    spot_empty = SpotDiagram(x_coords=[], y_coords=[])
    cx, cy = spot_empty.calculate_centroid()
    assert cx == 0.0 and cy == 0.0, "Empty spot diagram centroid must be (0,0)"
    assert spot_empty.calculate_rms_radius() == 0.0, "Empty spot diagram RMS radius must be 0"
    assert spot_empty.calculate_geo_radius() == 0.0, "Empty spot diagram GEO radius must be 0"

    # 2. SpotDiagram negative parameter error checks
    spot_err = SpotDiagram(x_coords=[1.0], y_coords=[1.0], wavelength_um=-0.5, f_number=4.0)
    with pytest.raises(ValueError, match="Wavelength and F/# must be positive"):
        spot_err.calculate_airy_disk_radius()

    # 3. RayFanAnalysis synthetic wavefront evaluation
    analyzer = RayFanAnalysis(py_sampling=21)
    py, ex, ey = analyzer.generate_tangential_fan(W040=0.001, W131=0.0, W220=0.0)
    assert len(py) == 21
    # Cubic spherical aberration profile EY(Py) = 4 * W040 * Py^3
    assert math.isclose(ey[-1], 0.004, abs_tol=1e-6), f"Expected EY(1.0)=0.004, got {ey[-1]}"

    # 4. SeidelAberrations calculation for a simple singlet lens
    singlet = SurfaceStack()
    singlet.add_surface(Surface(name="Object", thickness=100.0))
    singlet.add_surface(Surface(name="S1", radius=50.0, thickness=5.0, glass="BK7", semi_diameter=12.5))
    singlet.add_surface(Surface(name="S2", radius=-50.0, thickness=95.0, glass="AIR", semi_diameter=12.5))
    singlet.add_surface(Surface(name="Detector", surface_type=SurfaceType.DETECTOR))

    seidel = calculate_seidel_coefficients(singlet)
    assert isinstance(seidel, SeidelAberrations)
    assert len(seidel.surface_details) > 0, "Surface details must be populated"
    assert not math.isnan(seidel.S_I), "Spherical aberration S_I must not be NaN"


def test_tier5_cad_step_exporter_assembly():
    """
    Adversarial Test 9: 3D STEP CAD assembly exporter validation and file generation.
    """
    stack = SurfaceStack()
    stack.add_surface(Surface(name="Object", thickness=10.0))
    stack.add_surface(Surface(name="Lens1_Front", radius=50.0, thickness=5.0, glass="BK7", semi_diameter=15.0))
    stack.add_surface(Surface(name="Lens1_Back", radius=-50.0, thickness=10.0, glass="AIR", semi_diameter=15.0))
    stack.add_surface(Surface(name="Grating1", surface_type=SurfaceType.GRATING, grating_lines_per_mm=600.0, thickness=15.0, semi_diameter=12.5))
    stack.add_surface(Surface(name="Stop1", surface_type=SurfaceType.STOP, thickness=10.0, semi_diameter=10.0))
    stack.add_surface(Surface(name="CCD_Sensor", surface_type=SurfaceType.DETECTOR, semi_diameter=15.0))

    with tempfile.TemporaryDirectory() as tmp_dir:
        export_file = os.path.join(tmp_dir, "nested_dir", "test_optical_system.step")
        
        result = STEPExporter.export(stack, export_file)
        
        assert result.is_success, f"STEP export failed: {result.error_message}"
        assert os.path.exists(export_file), "Exported STEP file must exist on disk"
        assert result.file_size_bytes > 0, "Exported STEP file size must be > 0 bytes"
        assert result.num_components_exported > 0, "Should export solid CAD components"

        # Verify ISO-10303-21 STEP File Structure
        with open(export_file, 'r', encoding='utf-8') as f:
            content = f.read()

        assert "ISO-10303-21;" in content, "STEP file must contain ISO-10303-21 header"
        assert "HEADER;" in content and "ENDSEC;" in content, "STEP file must contain valid HEADER section"
        assert "FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));" in content, "STEP file must declare AUTOMOTIVE_DESIGN schema"
        assert "CLOSED_SHELL" in content, "STEP file must contain CLOSED_SHELL solid geometry entities"
        assert "MANIFOLD_SOLID_BREP" in content, "STEP file must contain MANIFOLD_SOLID_BREP entities"
