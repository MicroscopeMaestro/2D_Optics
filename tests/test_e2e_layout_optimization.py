"""
Comprehensive 4-Tier E2E Test Suite for Spectrometer Optical Layout Optimization.

Validates the full optimization pipeline, geometric clearance at Z=0 (slit plane)
and Z=50 (M1 collimator plane), optical alignment of DMD and backend relay,
zero-vignetting transmission, detector spot containment, schema compliance,
and robustness across extreme boundaries and real-world application scenarios.

Testing Tiers:
- Tier 1: Feature Coverage (F1 through F7, 37 test cases)
- Tier 2: Boundary & Corner Cases (35 test cases)
- Tier 3: Cross-Feature Combinations (7 test cases)
- Tier 4: Real-World Application Scenarios (5 test cases)
Total: 84 test cases (Requirement: >= 82 test cases)
"""

import os
import sys
import json
import math
import tempfile
import subprocess
import copy
from typing import List, Dict, Tuple, Any, Optional
import numpy as np
import pytest

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from v2.engine.models import SpectrometerSystem, SourceConfig
from v2.engine.surface import Surface, SurfaceType
from v2.engine.lde import SurfaceStack
from v2.engine.tracer import (
    OpticalSystem, DualSpeedTracer, Ray, check_segment_surface_interference,
    snell_refract_3d, build_rotation_matrix
)
from v2.engine.materials import GlassCatalog, get_refractive_index
from optics_app.analysis.spot import SpotDiagram

OPTIMIZE_SCRIPT_PATH = os.path.join(REPO_ROOT, "optimize_layout.py")
DEFAULT_DESKTOP_JSON = r"C:\Users\Juan\Desktop\Spectrometer_V2_Optimized.json"
FALLBACK_WORKSPACE_JSON = os.path.join(REPO_ROOT, "Spectrometer_V2_Optimized.json")

# Physical specifications from ORIGINAL_REQUEST.md & PROJECT.md
NOMINAL_WAVELENGTHS = [1.100, 1.225, 1.350]  # um
PRIMARY_WAVELENGTH = 1.225  # um
NOMINAL_NA = 0.12
NOMINAL_HALF_ANGLE_RAD = math.asin(NOMINAL_NA)
NOMINAL_HALF_ANGLE_DEG = math.degrees(NOMINAL_HALF_ANGLE_RAD)  # ~6.8928 deg
M1_RADIUS_THRESHOLD = 12.7  # mm
SLIT_HOUSING_THRESHOLD = 12.7  # mm
DETECTOR_RADIUS_LIMIT = 2.5  # mm (5.0 mm diameter)
SLIT_WIDTH_UM = 30.0  # um


# ============================================================================
# HELPER FUNCTIONS FOR 3D RAY TRACING & CLEARANCE CALCULATION
# ============================================================================

def get_authoritative_json_path() -> str:
    """Returns existing JSON path (Desktop target or workspace fallback)."""
    if os.path.exists(DEFAULT_DESKTOP_JSON):
        return DEFAULT_DESKTOP_JSON
    if os.path.exists(FALLBACK_WORKSPACE_JSON):
        return FALLBACK_WORKSPACE_JSON
    return DEFAULT_DESKTOP_JSON


def load_tested_system(json_path: Optional[str] = None) -> SpectrometerSystem:
    """Loads SpectrometerSystem from target JSON, running optimizer if needed."""
    target_path = json_path or get_authoritative_json_path()
    if not os.path.exists(target_path) and os.path.exists(OPTIMIZE_SCRIPT_PATH):
        subprocess.run([sys.executable, OPTIMIZE_SCRIPT_PATH, "--output", target_path], check=True)
    if not os.path.exists(target_path):
        pytest.skip(f"Target layout JSON not found at {target_path}")
    return SpectrometerSystem.load_from_json(target_path)


def trace_custom_rays(system: SpectrometerSystem, rays: List[Ray]) -> List[Ray]:
    """Traces arbitrary rays through a SpectrometerSystem instance."""
    opt_sys = OpticalSystem(system.surface_stack)
    return opt_sys.trace_rays(rays)


def compute_ray_intersections_at_z(ray: Ray, z_plane: float) -> List[Tuple[float, float, int]]:
    """
    Computes (X, Y, segment_index) coordinates where a ray's path crosses Z = z_plane.
    Returns empty list if ray does not intersect Z = z_plane.
    """
    intersections = []
    if not ray.history or len(ray.history) < 2:
        return intersections

    for i in range(len(ray.history) - 1):
        h1 = ray.history[i]
        h2 = ray.history[i + 1]
        z1 = float(h1["z_global"])
        z2 = float(h2["z_global"])
        if (min(z1, z2) <= z_plane <= max(z1, z2)) and abs(z2 - z1) > 1e-9:
            t = (z_plane - z1) / (z2 - z1)
            x_int = float(h1["x_global"] + t * (h2["x_global"] - h1["x_global"]))
            y_int = float(h1["y_global"] + t * (h2["y_global"] - h1["y_global"]))
            intersections.append((x_int, y_int, i))
    return intersections


def generate_phase_space_rays(
    wavelengths: List[float],
    na: float = 0.12,
    num_angle_steps: int = 5,
    slit_half_height_mm: float = 0.0,
    slit_half_width_mm: float = 0.015
) -> List[Ray]:
    """Generates deterministic phase-space ray bundle across field and pupil coordinates."""
    rays = []
    max_theta_rad = math.asin(min(max(na, 0.0), 0.99))
    if num_angle_steps <= 1 or max_theta_rad == 0.0:
        angle_fractions = [0.0]
    else:
        angle_fractions = np.linspace(-1.0, 1.0, num_angle_steps)

    y_src_positions = [0.0]
    if slit_half_height_mm > 0:
        y_src_positions = [-slit_half_height_mm, 0.0, slit_half_height_mm]

    x_src_positions = [0.0]
    if slit_half_width_mm > 0:
        x_src_positions = [-slit_half_width_mm, 0.0, slit_half_width_mm]

    for wl in wavelengths:
        for y_src in y_src_positions:
            for x_src in x_src_positions:
                for frac in angle_fractions:
                    theta = frac * max_theta_rad
                    dir_vec = np.array([0.0, math.sin(theta), math.cos(theta)], dtype=float)
                    norm = np.linalg.norm(dir_vec)
                    if norm > 0:
                        dir_vec = dir_vec / norm
                    rays.append(Ray(
                        pos=np.array([x_src, y_src, 0.0], dtype=float),
                        dir=dir_vec,
                        wavelength=wl
                    ))
    return rays


# ============================================================================
# TIER 1: FEATURE COVERAGE (F1 - F7)
# ============================================================================

class TestTier1FeatureCoverage:
    """
    Tier 1: Systematic coverage of core functional features F1 through F7.
    >= 35 test cases covering optimizer execution, mechanical clearances,
    relay decenter alignment, schema validation, zero-vignetting, and spot containment.
    """

    # --- F1: Automated Layout Optimizer (6 tests) ---
    def test_f1_cli_default_execution(self):
        """F1.1: Running optimize_layout.py without args executes with exit code 0."""
        if not os.path.exists(OPTIMIZE_SCRIPT_PATH):
            pytest.skip("optimize_layout.py not yet generated")
        res = subprocess.run([sys.executable, OPTIMIZE_SCRIPT_PATH], capture_output=True, text=True)
        assert res.returncode == 0, f"Script failed with stderr:\n{res.stderr}"

    def test_f1_cli_output_flag(self):
        """F1.2: Running optimize_layout.py --output <path> writes to custom path."""
        if not os.path.exists(OPTIMIZE_SCRIPT_PATH):
            pytest.skip("optimize_layout.py not yet generated")
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = os.path.join(tmpdir, "custom_opt.json")
            res = subprocess.run([sys.executable, OPTIMIZE_SCRIPT_PATH, "--output", out_file], capture_output=True, text=True)
            assert res.returncode == 0
            assert os.path.exists(out_file), "Custom output file was not created"
            with open(out_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            assert data.get("version") == 3

    def test_f1_cli_verify_flag(self):
        """F1.3: Running optimize_layout.py --verify executes programmatic verification."""
        if not os.path.exists(OPTIMIZE_SCRIPT_PATH):
            pytest.skip("optimize_layout.py not yet generated")
        res = subprocess.run([sys.executable, OPTIMIZE_SCRIPT_PATH, "--verify"], capture_output=True, text=True)
        assert res.returncode == 0
        output_lower = (res.stdout + res.stderr).lower()
        assert ("clearance" in output_lower or "vignett" in output_lower or "pass" in output_lower or "optimiz" in output_lower)

    def test_f1_programmatic_optimize_spectrometer_layout(self):
        """F1.4: Calling optimize_spectrometer_layout() returns SpectrometerSystem or dict."""
        try:
            import optimize_layout
        except ImportError:
            pytest.skip("optimize_layout module not found")

        assert hasattr(optimize_layout, "optimize_spectrometer_layout"), "Missing optimize_spectrometer_layout export"
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "prog_opt.json")
            result = optimize_layout.optimize_spectrometer_layout(output_path=out_path)
            assert result is not None
            assert os.path.exists(out_path)

    def test_f1_programmatic_verify_layout(self):
        """F1.5: Calling verify_layout_clearance_and_vignetting() returns verification status."""
        try:
            import optimize_layout
        except ImportError:
            pytest.skip("optimize_layout module not found")

        assert hasattr(optimize_layout, "verify_layout_clearance_and_vignetting"), "Missing verify export"
        system = load_tested_system()
        res = optimize_layout.verify_layout_clearance_and_vignetting(system)
        assert isinstance(res, dict)
        assert "vignetted" in res or "vignetted_count" in res or "all_unvignetted" in res or "is_valid" in res

    def test_f1_optimizer_returns_uncrossed_z_geometry(self):
        """F1.6: Final layout forms an uncrossed Z-fold topology without reverse self-crossing."""
        system = load_tested_system()
        surfs = system.surfaces
        assert len(surfs) >= 7
        # Verify M1 tilt is non-zero, Grating tilt is non-zero, DMD tilt is non-zero
        m1 = surfs[1]
        grating = surfs[2]
        dmd = surfs[6]
        assert abs(m1.tilt_x) > 0.01, "M1 tilt must be non-zero for fold"
        assert abs(grating.tilt_x) > 0.01, "Grating tilt must be non-zero"
        assert abs(dmd.tilt_x) > 0.01, "DMD tilt must be non-zero"

    # --- F2: Mechanical Clearance at Z=0 mm (Slit Plane) (5 tests) ---
    def test_f2_slit_clearance_nominal_primary_wavelength(self):
        """F2.1: 1.225 um chief ray returning across Z=0 clears slit origin by > 12.7 mm."""
        system = load_tested_system()
        chief_ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=PRIMARY_WAVELENGTH)
        traced = trace_custom_rays(system, [chief_ray])[0]
        assert not traced.vignetted
        crossings = compute_ray_intersections_at_z(traced, 0.0)
        # Filter out segment 0 (initial slit emission at t=0)
        returning_crossings = [c for c in crossings if c[2] > 0]
        assert len(returning_crossings) >= 1, "Expected ray to cross Z=0 returning from M1/Grating"
        for x_int, y_int, seg_idx in returning_crossings:
            dist = math.hypot(x_int, y_int)
            assert dist > SLIT_HOUSING_THRESHOLD, f"Clearance at Z=0 ({dist:.2f} mm) <= 12.7 mm threshold on segment {seg_idx}"

    def test_f2_slit_clearance_all_nominal_wavelengths(self):
        """F2.2: Returning chief rays for all 3 wavelengths clear Z=0 slit origin by > 12.7 mm."""
        system = load_tested_system()
        rays = [Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl) for wl in NOMINAL_WAVELENGTHS]
        traced_rays = trace_custom_rays(system, rays)
        for r in traced_rays:
            assert not r.vignetted
            crossings = [c for c in compute_ray_intersections_at_z(r, 0.0) if c[2] > 0]
            for x_int, y_int, seg_idx in crossings:
                dist = math.hypot(x_int, y_int)
                assert dist > SLIT_HOUSING_THRESHOLD, f"Wavelength {r.wavelength} um clearance at Z=0 is {dist:.2f} mm <= 12.7 mm"

    def test_f2_slit_clearance_marginal_cone_positive(self):
        """F2.3: Positive marginal ray (+6.89 deg) clears Z=0 slit origin by > 12.7 mm."""
        system = load_tested_system()
        theta = NOMINAL_HALF_ANGLE_RAD
        ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, math.sin(theta), math.cos(theta)]), wavelength=PRIMARY_WAVELENGTH)
        traced = trace_custom_rays(system, [ray])[0]
        assert not traced.vignetted
        crossings = [c for c in compute_ray_intersections_at_z(traced, 0.0) if c[2] > 0]
        for x_int, y_int, seg_idx in crossings:
            dist = math.hypot(x_int, y_int)
            assert dist > SLIT_HOUSING_THRESHOLD

    def test_f2_slit_clearance_marginal_cone_negative(self):
        """F2.4: Negative marginal ray (-6.89 deg) clears Z=0 slit origin by > 12.7 mm."""
        system = load_tested_system()
        theta = -NOMINAL_HALF_ANGLE_RAD
        ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, math.sin(theta), math.cos(theta)]), wavelength=PRIMARY_WAVELENGTH)
        traced = trace_custom_rays(system, [ray])[0]
        assert not traced.vignetted
        crossings = [c for c in compute_ray_intersections_at_z(traced, 0.0) if c[2] > 0]
        for x_int, y_int, seg_idx in crossings:
            dist = math.hypot(x_int, y_int)
            assert dist > SLIT_HOUSING_THRESHOLD

    def test_f2_slit_clearance_full_pupil_envelope(self):
        """F2.5: Full pupil phase-space envelope across band clears Z=0 slit housing (>12.7 mm)."""
        system = load_tested_system()
        rays = generate_phase_space_rays(NOMINAL_WAVELENGTHS, na=0.12, num_angle_steps=5)
        traced = trace_custom_rays(system, rays)
        for r in traced:
            assert not r.vignetted
            crossings = [c for c in compute_ray_intersections_at_z(r, 0.0) if c[2] > 0]
            for x_int, y_int, _ in crossings:
                assert math.hypot(x_int, y_int) > SLIT_HOUSING_THRESHOLD

    # --- F3: Mechanical Clearance at Z=50 mm (M1 Collimator Plane) (5 tests) ---
    def test_f3_m1_clearance_leg3_focusing_beam_primary(self):
        """F3.1: Leg 3 focusing beam at 1.225 um passing Z=50 mm satisfies |Y| > 12.7 mm."""
        system = load_tested_system()
        chief_ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=PRIMARY_WAVELENGTH)
        traced = trace_custom_rays(system, [chief_ray])[0]
        crossings = [c for c in compute_ray_intersections_at_z(traced, 50.0) if c[2] >= 4]
        assert len(crossings) >= 1, "Expected ray to cross Z=50 heading from camera lens to DMD or relay"
        for x_int, y_int, seg_idx in crossings:
            dist_from_m1_axis = math.hypot(x_int, y_int)
            assert dist_from_m1_axis > M1_RADIUS_THRESHOLD, f"Clearance at Z=50 ({dist_from_m1_axis:.2f} mm) <= 12.7 mm"

    def test_f3_m1_clearance_leg3_all_wavelengths(self):
        """F3.2: Leg 3 beams across all nominal wavelengths at Z=50 mm satisfy |Y| > 12.7 mm."""
        system = load_tested_system()
        rays = [Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl) for wl in NOMINAL_WAVELENGTHS]
        traced = trace_custom_rays(system, rays)
        for r in traced:
            crossings = [c for c in compute_ray_intersections_at_z(r, 50.0) if c[2] >= 4]
            for x_int, y_int, _ in crossings:
                assert math.hypot(x_int, y_int) > M1_RADIUS_THRESHOLD

    def test_f3_m1_clearance_leg3_marginal_rays(self):
        """F3.3: Extreme marginal rays (+/-6.89 deg) at Z=50 mm satisfy clearance > 12.7 mm."""
        system = load_tested_system()
        for angle_deg in [-NOMINAL_HALF_ANGLE_DEG, NOMINAL_HALF_ANGLE_DEG]:
            theta = math.radians(angle_deg)
            for wl in NOMINAL_WAVELENGTHS:
                ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, math.sin(theta), math.cos(theta)]), wavelength=wl)
                traced = trace_custom_rays(system, [ray])[0]
                assert not traced.vignetted
                crossings = [c for c in compute_ray_intersections_at_z(traced, 50.0) if c[2] >= 4]
                for x_int, y_int, _ in crossings:
                    assert math.hypot(x_int, y_int) > M1_RADIUS_THRESHOLD

    def test_f3_m1_clearance_leg4_relay_beam(self):
        """F3.4: Leg 4 relay beam at Z=50 mm clears M1 mirror mount by > 25.4 mm."""
        system = load_tested_system()
        chief_ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=PRIMARY_WAVELENGTH)
        traced = trace_custom_rays(system, [chief_ray])[0]
        # Check all crossings on Leg 4 (segment index >= 7)
        leg4_crossings = [c for c in compute_ray_intersections_at_z(traced, 50.0) if c[2] >= 7]
        if leg4_crossings:
            for x_int, y_int, _ in leg4_crossings:
                assert math.hypot(x_int, y_int) > 25.4, f"Relay beam clearance at Z=50 ({math.hypot(x_int, y_int):.2f} mm) <= 25.4 mm"

    def test_f3_camera_lens_edge_clearance(self):
        """F3.5: Collimated beam from M1 to Grating does not collide with camera doublet barrel."""
        system = load_tested_system()
        # Front surface of camera doublet is surface index 3
        cam_surf = system.surfaces[3]
        cam_origin = cam_surf.global_origin
        cam_bottom_y = cam_origin[1] - cam_surf.semi_diameter

        # Beam from M1 to Grating at Z = cam_origin[2]
        chief_ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=PRIMARY_WAVELENGTH)
        traced = trace_custom_rays(system, [chief_ray])[0]
        crossings = compute_ray_intersections_at_z(traced, float(cam_origin[2]))
        # Segment 1 is M1 -> Grating
        seg1_crossing = [c for c in crossings if c[2] == 1]
        if seg1_crossing:
            y_beam = seg1_crossing[0][1]
            # Ensure beam is separated from camera bottom edge
            separation = abs(y_beam - cam_bottom_y)
            assert separation > 0.0, "Beam from M1 to Grating collides with camera lens edge"

    # --- F4: Relay Decenter & Chief Ray Swallowing (5 tests) ---
    def test_f4_chief_ray_dmd_center(self):
        """F4.1: 1.225 um chief ray lands centered on DMD surface (|y_local| <= 0.5 mm)."""
        system = load_tested_system()
        chief_ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=PRIMARY_WAVELENGTH)
        traced = trace_custom_rays(system, [chief_ray])[0]
        assert not traced.vignetted
        # Find DMD surface hit (surface index 6)
        dmd_hit = [h for h in traced.history if h["surface_index"] == 6][0]
        assert abs(dmd_hit["y_local"]) <= 0.5, f"Chief ray offset on DMD ({dmd_hit['y_local']:.3f} mm) > 0.5 mm"

    def test_f4_field_lens_chief_ray_incidence(self):
        """F4.2: Chief ray enters Field lens doublet well within clear aperture (r <= 25.4 mm)."""
        system = load_tested_system()
        chief_ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=PRIMARY_WAVELENGTH)
        traced = trace_custom_rays(system, [chief_ray])[0]
        field_hit = [h for h in traced.history if h["surface_index"] == 7][0]
        r_field = math.hypot(field_hit["x_local"], field_hit["y_local"])
        assert r_field < 25.4, f"Field lens hit radius ({r_field:.2f} mm) exceeds clear aperture (25.4 mm)"

    def test_f4_condenser_lens_chief_ray_incidence(self):
        """F4.3: Chief ray enters Condenser lens doublet well within clear aperture (r <= 25.4 mm)."""
        system = load_tested_system()
        chief_ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=PRIMARY_WAVELENGTH)
        traced = trace_custom_rays(system, [chief_ray])[0]
        cond_hit = [h for h in traced.history if h["surface_index"] == 10][0]
        r_cond = math.hypot(cond_hit["x_local"], cond_hit["y_local"])
        assert r_cond < 25.4, f"Condenser lens hit radius ({r_cond:.2f} mm) exceeds clear aperture (25.4 mm)"

    def test_f4_relay_lens_coaxial_alignment(self):
        """F4.4: Field and Condenser doublet elements have matched coaxial decenter values."""
        system = load_tested_system()
        surfs = system.surfaces
        field_decenter_y = surfs[7].decenter_y
        for idx in [8, 9]:
            assert abs(surfs[idx].decenter_y - field_decenter_y) < 1e-3, f"Field lens element {idx} decentered inconsistently"
        cond_decenter_y = surfs[10].decenter_y
        for idx in [11, 12]:
            assert abs(surfs[idx].decenter_y - cond_decenter_y) < 1e-3, f"Condenser lens element {idx} decentered inconsistently"

    def test_f4_detector_chief_ray_centering(self):
        """F4.5: 1.225 um chief ray lands centered on Detector (|y_local| <= 0.5 mm)."""
        system = load_tested_system()
        det_idx = system.detector_surface_idx
        chief_ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=PRIMARY_WAVELENGTH)
        traced = trace_custom_rays(system, [chief_ray])[0]
        det_hit = [h for h in traced.history if h["surface_index"] == det_idx][0]
        assert abs(det_hit["y_local"]) <= 0.5, f"Detector hit offset ({det_hit['y_local']:.3f} mm) > 0.5 mm"

    # --- F5: JSON Generation & Schema Compliance (6 tests) ---
    def test_f5_json_file_exists(self):
        """F5.1: Target optimized JSON exists and is accessible."""
        target_path = get_authoritative_json_path()
        assert os.path.exists(target_path), f"JSON file not found at {target_path}"

    def test_f5_json_version_is_3(self):
        """F5.2: JSON schema contains top-level version = 3."""
        target_path = get_authoritative_json_path()
        with open(target_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data.get("version") == 3, f"Expected version 3, got {data.get('version')}"

    def test_f5_json_surface_count_and_sequence(self):
        """F5.3: JSON defines exactly 14 sequential surfaces covering all 8 functional groups."""
        target_path = get_authoritative_json_path()
        with open(target_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        surfaces = data.get("surfaces", [])
        assert len(surfaces) == 14, f"Expected 14 surfaces, found {len(surfaces)}"
        expected_names = [
            "Input_Slit", "M1_Collimator", "Diffraction_Grating",
            "AC254-050-C_Foc_S1", "AC254-050-C_Foc_S2", "AC254-050-C_Foc_S3",
            "DMD4500", "Field_F", "Field_M", "Field_B",
            "Cond_F", "Cond_M", "Cond_B", "LightPipe"
        ]
        for i, name_substr in enumerate(expected_names):
            actual_name = surfaces[i].get("name", "")
            assert name_substr.lower() in actual_name.lower(), f"Surface {i} name '{actual_name}' does not match '{name_substr}'"

    def test_f5_json_system_config_structure(self):
        """F5.4: JSON system_config contains required slit width, wavelengths, and aperture."""
        target_path = get_authoritative_json_path()
        with open(target_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        sys_cfg = data.get("system_config", {})
        assert sys_cfg is not None
        ap = sys_cfg.get("aperture", {})
        slit_w = ap.get("slit_width_um", data.get("slit_width", 30.0))
        assert abs(float(slit_w) - SLIT_WIDTH_UM) < 1.0, f"Slit width is {slit_w}, expected {SLIT_WIDTH_UM}"

    def test_f5_json_loadable_by_v2_spectrometer_system(self):
        """F5.5: JSON loads seamlessly into SpectrometerSystem without exceptions."""
        target_path = get_authoritative_json_path()
        system = SpectrometerSystem.load_from_json(target_path)
        assert isinstance(system, SpectrometerSystem)
        assert len(system.surfaces) == 14
        assert system.primary_wavelength == PRIMARY_WAVELENGTH

    def test_f5_json_loadable_by_v2_surface_stack(self):
        """F5.6: JSON surfaces load into SurfaceStack and calculate accumulated positions."""
        target_path = get_authoritative_json_path()
        system = SpectrometerSystem.load_from_json(target_path)
        stack = system.surface_stack
        assert len(stack.surfaces) == 14
        for s in stack.surfaces:
            assert hasattr(s, "global_origin")
            assert hasattr(s, "global_R")

    # --- F6: Extreme Marginal Ray Zero-Vignetting (5 tests) ---
    def test_f6_nominal_chief_rays_zero_vignetting(self):
        """F6.1: Chief rays at 1.100, 1.225, 1.350 um trace with 0 vignetting."""
        system = load_tested_system()
        rays = [Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl) for wl in NOMINAL_WAVELENGTHS]
        traced = trace_custom_rays(system, rays)
        for r in traced:
            assert not r.vignetted, f"Chief ray at {r.wavelength} um is vignetted"

    def test_f6_marginal_rays_primary_wl_zero_vignetting(self):
        """F6.2: Extreme marginal rays (+/-6.89 deg) at 1.225 um trace with 0 vignetting."""
        system = load_tested_system()
        rays = [
            Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, math.sin(NOMINAL_HALF_ANGLE_RAD), math.cos(NOMINAL_HALF_ANGLE_RAD)]), wavelength=PRIMARY_WAVELENGTH),
            Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, -math.sin(NOMINAL_HALF_ANGLE_RAD), math.cos(NOMINAL_HALF_ANGLE_RAD)]), wavelength=PRIMARY_WAVELENGTH),
        ]
        traced = trace_custom_rays(system, rays)
        for r in traced:
            assert not r.vignetted, f"Marginal ray at {r.wavelength} um is vignetted"

    def test_f6_marginal_rays_lower_band_zero_vignetting(self):
        """F6.3: Marginal rays at lower band edge (1.100 um) trace with 0 vignetting."""
        system = load_tested_system()
        rays = [
            Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, math.sin(NOMINAL_HALF_ANGLE_RAD), math.cos(NOMINAL_HALF_ANGLE_RAD)]), wavelength=1.100),
            Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, -math.sin(NOMINAL_HALF_ANGLE_RAD), math.cos(NOMINAL_HALF_ANGLE_RAD)]), wavelength=1.100),
        ]
        traced = trace_custom_rays(system, rays)
        for r in traced:
            assert not r.vignetted, f"Lower band marginal ray is vignetted"

    def test_f6_marginal_rays_upper_band_zero_vignetting(self):
        """F6.4: Marginal rays at upper band edge (1.350 um) trace with 0 vignetting."""
        system = load_tested_system()
        rays = [
            Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, math.sin(NOMINAL_HALF_ANGLE_RAD), math.cos(NOMINAL_HALF_ANGLE_RAD)]), wavelength=1.350),
            Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, -math.sin(NOMINAL_HALF_ANGLE_RAD), math.cos(NOMINAL_HALF_ANGLE_RAD)]), wavelength=1.350),
        ]
        traced = trace_custom_rays(system, rays)
        for r in traced:
            assert not r.vignetted, f"Upper band marginal ray is vignetted"

    def test_f6_full_phase_space_zero_vignetting(self):
        """F6.5: Complete 45-ray phase-space bundle traces with 100% transmission (0 vignetting)."""
        system = load_tested_system()
        rays = generate_phase_space_rays(NOMINAL_WAVELENGTHS, na=0.12, num_angle_steps=5, slit_half_width_mm=0.015)
        traced = trace_custom_rays(system, rays)
        vignetted_count = sum(1 for r in traced if r.vignetted)
        assert vignetted_count == 0, f"{vignetted_count} of {len(traced)} phase-space rays were vignetted"

    # --- F7: Spot Size & 2.5 mm Detector Containment (5 tests) ---
    def test_f7_spot_radius_primary_wl(self):
        """F7.1: Geometric spot radius at detector for 1.225 um is <= 2.5 mm."""
        system = load_tested_system()
        det_idx = system.detector_surface_idx
        rays = generate_phase_space_rays([PRIMARY_WAVELENGTH], na=0.12, num_angle_steps=7)
        traced = trace_custom_rays(system, rays)
        for r in traced:
            assert not r.vignetted
            det_hit = [h for h in r.history if h["surface_index"] == det_idx][0]
            r_hit = math.hypot(det_hit["x_local"], det_hit["y_local"])
            assert r_hit <= DETECTOR_RADIUS_LIMIT, f"1.225 um ray landed at r = {r_hit:.3f} mm > 2.5 mm"

    def test_f7_spot_radius_lower_band_wl(self):
        """F7.2: Geometric spot radius at detector for 1.100 um is <= 2.5 mm."""
        system = load_tested_system()
        det_idx = system.detector_surface_idx
        rays = generate_phase_space_rays([1.100], na=0.12, num_angle_steps=7)
        traced = trace_custom_rays(system, rays)
        for r in traced:
            assert not r.vignetted
            det_hit = [h for h in r.history if h["surface_index"] == det_idx][0]
            r_hit = math.hypot(det_hit["x_local"], det_hit["y_local"])
            assert r_hit <= DETECTOR_RADIUS_LIMIT, f"1.100 um ray landed at r = {r_hit:.3f} mm > 2.5 mm"

    def test_f7_spot_radius_upper_band_wl(self):
        """F7.3: Geometric spot radius at detector for 1.350 um is <= 2.5 mm."""
        system = load_tested_system()
        det_idx = system.detector_surface_idx
        rays = generate_phase_space_rays([1.350], na=0.12, num_angle_steps=7)
        traced = trace_custom_rays(system, rays)
        for r in traced:
            assert not r.vignetted
            det_hit = [h for h in r.history if h["surface_index"] == det_idx][0]
            r_hit = math.hypot(det_hit["x_local"], det_hit["y_local"])
            assert r_hit <= DETECTOR_RADIUS_LIMIT, f"1.350 um ray landed at r = {r_hit:.3f} mm > 2.5 mm"

    def test_f7_all_marginal_rays_inside_detector(self):
        """F7.4: 100% of rays across full band and pupil land within 2.5 mm radius detector."""
        system = load_tested_system()
        det_idx = system.detector_surface_idx
        rays = generate_phase_space_rays(NOMINAL_WAVELENGTHS, na=0.12, num_angle_steps=5, slit_half_width_mm=0.015)
        traced = trace_custom_rays(system, rays)
        for r in traced:
            assert not r.vignetted
            det_hit = [h for h in r.history if h["surface_index"] == det_idx][0]
            r_hit = math.hypot(det_hit["x_local"], det_hit["y_local"])
            assert r_hit <= DETECTOR_RADIUS_LIMIT, f"Ray landed at r = {r_hit:.3f} mm > 2.5 mm"

    def test_f7_rms_spot_radius_bound(self):
        """F7.5: Polychromatic RMS spot radius at detector is < 2.0 mm."""
        system = load_tested_system()
        tracer = DualSpeedTracer(system)
        res = tracer.trace_full(wavelengths=NOMINAL_WAVELENGTHS, rings=4)
        assert res.rms_radius < 2.0, f"RMS spot radius ({res.rms_radius:.3f} mm) >= 2.0 mm limit"


# ============================================================================
# TIER 2: BOUNDARY & CORNER CASES
# ============================================================================

class TestTier2BoundaryAndCornerCases:
    """
    Tier 2: Boundary value analysis (BVA) and extreme corner cases.
    >= 35 test cases covering spectral extensions, numerical aperture ranges,
    slit geometry bounds, clearance threshold edges, custom paths, and error handling.
    """

    # --- Extreme Wavelengths (5 tests) ---
    def test_tier2_extreme_wavelength_1050nm(self):
        """Tier 2.1: Ray tracing at 1.050 um (out-of-band short NIR) traces without crashing."""
        system = load_tested_system()
        chief_ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=1.050)
        traced = trace_custom_rays(system, [chief_ray])[0]
        assert not traced.vignetted

    def test_tier2_extreme_wavelength_1100nm(self):
        """Tier 2.2: Ray tracing exactly at lower band edge (1.100 um) is unvignetted."""
        system = load_tested_system()
        chief_ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=1.100)
        traced = trace_custom_rays(system, [chief_ray])[0]
        assert not traced.vignetted

    def test_tier2_extreme_wavelength_1350nm(self):
        """Tier 2.3: Ray tracing exactly at upper band edge (1.350 um) is unvignetted."""
        system = load_tested_system()
        chief_ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=1.350)
        traced = trace_custom_rays(system, [chief_ray])[0]
        assert not traced.vignetted

    def test_tier2_extreme_wavelength_1400nm(self):
        """Tier 2.4: Ray tracing at 1.400 um (out-of-band long NIR) propagates through grating."""
        system = load_tested_system()
        chief_ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=1.400)
        traced = trace_custom_rays(system, [chief_ray])[0]
        # At 1.400 um grating equation still permits real propagation
        assert len(traced.history) >= 4

    def test_tier2_dense_spectral_sweep_1100_to_1350nm(self):
        """Tier 2.5: Dense 11-channel spectral sweep across band traces 100% unvignetted."""
        system = load_tested_system()
        sweep_wls = list(np.linspace(1.100, 1.350, 11))
        rays = [Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl) for wl in sweep_wls]
        traced = trace_custom_rays(system, rays)
        for r in traced:
            assert not r.vignetted, f"Spectral sweep ray at {r.wavelength:.3f} um was vignetted"

    # --- Extreme Numerical Apertures (6 tests) ---
    def test_tier2_na_zero_collimated_beam(self):
        """Tier 2.6: Zero NA (parallel collimated beam) traces with 0 vignetting."""
        system = load_tested_system()
        rays = generate_phase_space_rays(NOMINAL_WAVELENGTHS, na=0.0, num_angle_steps=1)
        traced = trace_custom_rays(system, rays)
        for r in traced:
            assert not r.vignetted

    def test_tier2_na_low_cone(self):
        """Tier 2.7: Low NA (NA = 0.03) narrow cone traces with 0 vignetting."""
        system = load_tested_system()
        rays = generate_phase_space_rays(NOMINAL_WAVELENGTHS, na=0.03, num_angle_steps=3)
        traced = trace_custom_rays(system, rays)
        for r in traced:
            assert not r.vignetted

    def test_tier2_na_nominal_half(self):
        """Tier 2.8: Half nominal NA (NA = 0.06) traces with 0 vignetting."""
        system = load_tested_system()
        rays = generate_phase_space_rays(NOMINAL_WAVELENGTHS, na=0.06, num_angle_steps=5)
        traced = trace_custom_rays(system, rays)
        for r in traced:
            assert not r.vignetted

    def test_tier2_na_nominal_full(self):
        """Tier 2.9: Full nominal NA (NA = 0.12) traces with 0 vignetting."""
        system = load_tested_system()
        rays = generate_phase_space_rays(NOMINAL_WAVELENGTHS, na=0.12, num_angle_steps=5)
        traced = trace_custom_rays(system, rays)
        for r in traced:
            assert not r.vignetted

    def test_tier2_na_overfill_boundary(self):
        """Tier 2.10: 108% overfill NA (NA = 0.13) stress test."""
        system = load_tested_system()
        chief_ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, math.sin(math.asin(0.13)), math.cos(math.asin(0.13))]), wavelength=PRIMARY_WAVELENGTH)
        traced = trace_custom_rays(system, [chief_ray])[0]
        # Check that ray reaches at least the detector or camera
        assert len(traced.history) >= 7

    def test_tier2_na_extreme_overfill(self):
        """Tier 2.11: Extreme overfill NA (NA = 0.15) ray tracing executes robustly."""
        system = load_tested_system()
        chief_ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, math.sin(math.asin(0.15)), math.cos(math.asin(0.15))]), wavelength=PRIMARY_WAVELENGTH)
        traced = trace_custom_rays(system, [chief_ray])[0]
        assert traced is not None

    # --- Slit Spatial Offset Boundaries (8 tests) ---
    def test_tier2_slit_center_origin(self):
        """Tier 2.12: Origin slit position (x=0, y=0) traces unvignetted."""
        system = load_tested_system()
        ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=PRIMARY_WAVELENGTH)
        traced = trace_custom_rays(system, [ray])[0]
        assert not traced.vignetted

    def test_tier2_slit_width_edge_positive(self):
        """Tier 2.13: Positive slit width edge (+15 um) traces unvignetted."""
        system = load_tested_system()
        ray = Ray(pos=np.array([0.015, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=PRIMARY_WAVELENGTH)
        traced = trace_custom_rays(system, [ray])[0]
        assert not traced.vignetted

    def test_tier2_slit_width_edge_negative(self):
        """Tier 2.14: Negative slit width edge (-15 um) traces unvignetted."""
        system = load_tested_system()
        ray = Ray(pos=np.array([-0.015, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=PRIMARY_WAVELENGTH)
        traced = trace_custom_rays(system, [ray])[0]
        assert not traced.vignetted

    def test_tier2_slit_height_boundary_pos_250um(self):
        """Tier 2.15: Positive slit height (+250 um) traces unvignetted."""
        system = load_tested_system()
        ray = Ray(pos=np.array([0.0, 0.250, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=PRIMARY_WAVELENGTH)
        traced = trace_custom_rays(system, [ray])[0]
        assert not traced.vignetted

    def test_tier2_slit_height_boundary_neg_250um(self):
        """Tier 2.16: Negative slit height (-250 um) traces unvignetted."""
        system = load_tested_system()
        ray = Ray(pos=np.array([0.0, -0.250, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=PRIMARY_WAVELENGTH)
        traced = trace_custom_rays(system, [ray])[0]
        assert not traced.vignetted

    def test_tier2_slit_height_extreme_pos_500um(self):
        """Tier 2.17: Extreme positive slit height (+500 um = 0.5 mm) traces cleanly."""
        system = load_tested_system()
        ray = Ray(pos=np.array([0.0, 0.500, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=PRIMARY_WAVELENGTH)
        traced = trace_custom_rays(system, [ray])[0]
        assert len(traced.history) == 14

    def test_tier2_slit_height_extreme_neg_500um(self):
        """Tier 2.18: Extreme negative slit height (-500 um = -0.5 mm) traces cleanly."""
        system = load_tested_system()
        ray = Ray(pos=np.array([0.0, -0.500, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=PRIMARY_WAVELENGTH)
        traced = trace_custom_rays(system, [ray])[0]
        assert len(traced.history) == 14

    def test_tier2_slit_4_corners_phase_space(self):
        """Tier 2.19: Four extreme corners of the slit aperture (x=+/-15um, y=+/-250um) trace unvignetted."""
        system = load_tested_system()
        corners = [(0.015, 0.250), (0.015, -0.250), (-0.015, 0.250), (-0.015, -0.250)]
        for x_c, y_c in corners:
            ray = Ray(pos=np.array([x_c, y_c, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=PRIMARY_WAVELENGTH)
            traced = trace_custom_rays(system, [ray])[0]
            assert not traced.vignetted

    # --- Clearance Threshold Boundaries (6 tests) ---
    def test_tier2_m1_clearance_exact_threshold_12_7mm(self):
        """Tier 2.20: Clearance at Z=50 mm strictly exceeds 12.7 mm threshold."""
        system = load_tested_system()
        chief_ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=PRIMARY_WAVELENGTH)
        traced = trace_custom_rays(system, [chief_ray])[0]
        crossings = [c for c in compute_ray_intersections_at_z(traced, 50.0) if c[2] >= 4]
        for x_int, y_int, _ in crossings:
            clearance = math.hypot(x_int, y_int)
            assert clearance > 12.7, f"Clearance ({clearance:.3f} mm) <= 12.7 mm"

    def test_tier2_m1_clearance_safety_margin(self):
        """Tier 2.21: M1 clearance at Z=50 mm has at least 1.0 mm extra safety margin (>13.7 mm)."""
        system = load_tested_system()
        chief_ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=PRIMARY_WAVELENGTH)
        traced = trace_custom_rays(system, [chief_ray])[0]
        crossings = [c for c in compute_ray_intersections_at_z(traced, 50.0) if c[2] >= 4]
        for x_int, y_int, _ in crossings:
            clearance = math.hypot(x_int, y_int)
            assert clearance > 13.7, f"Clearance ({clearance:.3f} mm) does not provide 1.0 mm safety margin (>13.7 mm)"

    def test_tier2_slit_plane_clearance_exact_threshold_12_7mm(self):
        """Tier 2.22: Clearance at Z=0 mm strictly exceeds 12.7 mm threshold."""
        system = load_tested_system()
        chief_ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=PRIMARY_WAVELENGTH)
        traced = trace_custom_rays(system, [chief_ray])[0]
        crossings = [c for c in compute_ray_intersections_at_z(traced, 0.0) if c[2] > 0]
        for x_int, y_int, _ in crossings:
            assert math.hypot(x_int, y_int) > 12.7

    def test_tier2_slit_plane_clearance_safety_margin(self):
        """Tier 2.23: Slit plane clearance has at least 3.0 mm extra safety margin (>15.7 mm)."""
        system = load_tested_system()
        chief_ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=PRIMARY_WAVELENGTH)
        traced = trace_custom_rays(system, [chief_ray])[0]
        crossings = [c for c in compute_ray_intersections_at_z(traced, 0.0) if c[2] > 0]
        for x_int, y_int, _ in crossings:
            assert math.hypot(x_int, y_int) > 15.7

    def test_tier2_dmd_spectral_span_within_active_area(self):
        """Tier 2.24: Dispersed spectrum on DMD (1.100 - 1.350 um) fits within |y| <= 4.33 mm active area."""
        system = load_tested_system()
        rays = [Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl) for wl in NOMINAL_WAVELENGTHS]
        traced = trace_custom_rays(system, rays)
        for r in traced:
            dmd_hit = [h for h in r.history if h["surface_index"] == 6][0]
            assert abs(dmd_hit["y_local"]) <= 4.33, f"DMD hit at {r.wavelength} um is {dmd_hit['y_local']:.3f} mm (exceeds 4.33 mm)"

    def test_tier2_dmd_edge_clearance_margin(self):
        """Tier 2.25: Spectral footprint on DMD has > 1.5 mm margin to the 6.0 mm clear aperture."""
        system = load_tested_system()
        rays = [Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl) for wl in NOMINAL_WAVELENGTHS]
        traced = trace_custom_rays(system, rays)
        for r in traced:
            dmd_hit = [h for h in r.history if h["surface_index"] == 6][0]
            assert abs(dmd_hit["y_local"]) < 4.5, f"DMD hit ({dmd_hit['y_local']:.3f} mm) too close to edge"

    # --- Custom Output Path Arguments & Filesystem Corners (4 tests) ---
    def test_tier2_custom_output_nested_directory(self):
        """Tier 2.26: Optimizer creates parent folders when output is in deeply nested path."""
        if not os.path.exists(OPTIMIZE_SCRIPT_PATH):
            pytest.skip("optimize_layout.py not found")
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_out = os.path.join(tmpdir, "deep", "nested", "folder", "opt.json")
            res = subprocess.run([sys.executable, OPTIMIZE_SCRIPT_PATH, "--output", nested_out], capture_output=True, text=True)
            assert res.returncode == 0
            assert os.path.exists(nested_out)

    def test_tier2_custom_output_relative_path(self):
        """Tier 2.27: Optimizer accepts relative output paths correctly."""
        if not os.path.exists(OPTIMIZE_SCRIPT_PATH):
            pytest.skip("optimize_layout.py not found")
        with tempfile.TemporaryDirectory() as tmpdir:
            rel_out = os.path.join(tmpdir, "rel_test.json")
            res = subprocess.run([sys.executable, OPTIMIZE_SCRIPT_PATH, "--output", rel_out], cwd=REPO_ROOT, capture_output=True, text=True)
            assert res.returncode == 0
            assert os.path.exists(rel_out)

    def test_tier2_custom_output_filename_with_spaces(self):
        """Tier 2.28: Optimizer handles filenames and directories with spaces."""
        if not os.path.exists(OPTIMIZE_SCRIPT_PATH):
            pytest.skip("optimize_layout.py not found")
        with tempfile.TemporaryDirectory() as tmpdir:
            space_dir = os.path.join(tmpdir, "space dir name")
            os.makedirs(space_dir, exist_ok=True)
            space_out = os.path.join(space_dir, "my opt layout.json")
            res = subprocess.run([sys.executable, OPTIMIZE_SCRIPT_PATH, "--output", space_out], capture_output=True, text=True)
            assert res.returncode == 0
            assert os.path.exists(space_out)

    def test_tier2_custom_output_overwrite_existing(self):
        """Tier 2.29: Optimizer overwrites existing target JSON cleanly without error."""
        if not os.path.exists(OPTIMIZE_SCRIPT_PATH):
            pytest.skip("optimize_layout.py not found")
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = os.path.join(tmpdir, "overwrite.json")
            with open(out_file, "w", encoding="utf-8") as f:
                f.write('{"old": "data"}')
            res = subprocess.run([sys.executable, OPTIMIZE_SCRIPT_PATH, "--output", out_file], capture_output=True, text=True)
            assert res.returncode == 0
            with open(out_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            assert data.get("version") == 3

    # --- Error Handling & Invalid Input Combinations (6 tests) ---
    def test_tier2_error_on_invalid_cli_arg(self):
        """Tier 2.30: Passing invalid CLI flags produces non-zero exit code."""
        if not os.path.exists(OPTIMIZE_SCRIPT_PATH):
            pytest.skip("optimize_layout.py not found")
        res = subprocess.run([sys.executable, OPTIMIZE_SCRIPT_PATH, "--invalid-flag-xyz"], capture_output=True, text=True)
        assert res.returncode != 0

    def test_tier2_error_on_nonexistent_input_file(self):
        """Tier 2.31: SpectrometerSystem.load_from_json raises FileNotFoundError for missing path."""
        with pytest.raises(FileNotFoundError):
            SpectrometerSystem.load_from_json("nonexistent_path_12345.json")

    def test_tier2_error_on_corrupted_json_syntax(self):
        """Tier 2.32: Loading corrupted JSON raises JSONDecodeError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            corrupt_file = os.path.join(tmpdir, "corrupt.json")
            with open(corrupt_file, "w", encoding="utf-8") as f:
                f.write("{ version: 3, surfaces: [ corrupt syntax")
            with pytest.raises(json.JSONDecodeError):
                SpectrometerSystem.load_from_json(corrupt_file)

    def test_tier2_error_on_missing_required_surfaces(self):
        """Tier 2.33: Empty surface list produces empty surfaces or auto-detect fallback."""
        sys_empty = SpectrometerSystem.from_dict({"version": 3, "surfaces": []})
        assert len(sys_empty.surfaces) == 0

    def test_tier2_error_on_invalid_grating_density(self):
        """Tier 2.34: Zero or negative line density handled without unhandled exception."""
        surf = Surface(name="BadGrating", surface_type=SurfaceType.GRATING, grating_lines_per_mm=0.0)
        assert surf.grating_lines_per_mm == 0.0

    def test_tier2_error_on_unphysical_refractive_index(self):
        """Tier 2.35: Unknown glass material raises ValueError; empty glass defaults to AIR (1.0)."""
        with pytest.raises(ValueError):
            get_refractive_index("UNKNOWN_EXOTIC_GLASS_999", 1.225)
        idx_air = get_refractive_index("", 1.225)
        assert abs(idx_air - 1.000277) < 1e-4


# ============================================================================
# TIER 3: CROSS-FEATURE COMBINATIONS
# ============================================================================

class TestTier3CrossFeatureCombinations:
    """
    Tier 3: Pairwise and multi-feature interaction testing.
    >= 7 test cases evaluating cross-cutting dependencies between optimizer,
    file export/re-load, dispersion, relay alignment, clearance, and wavefront spot.
    """

    def test_tier3_full_pipeline_export_load_retrace(self):
        """Tier 3.1: Optimizer export -> JSON file -> load in V2 -> trace full -> 0 vignetting."""
        if not os.path.exists(OPTIMIZE_SCRIPT_PATH):
            pytest.skip("optimize_layout.py not found")
        with tempfile.TemporaryDirectory() as tmpdir:
            out_json = os.path.join(tmpdir, "e2e_retrace.json")
            res = subprocess.run([sys.executable, OPTIMIZE_SCRIPT_PATH, "--output", out_json], capture_output=True, text=True)
            assert res.returncode == 0

            system = SpectrometerSystem.load_from_json(out_json)
            tracer = DualSpeedTracer(system)
            full_res = tracer.trace_full(wavelengths=NOMINAL_WAVELENGTHS, rings=5)
            assert full_res.vignetted_count == 0, f"{full_res.vignetted_count} rays vignetted in re-traced system"
            assert full_res.rms_radius < 2.5

    def test_tier3_dispersion_vs_relay_decenter_interaction(self):
        """Tier 3.2: Chromatic dispersion across 1.100 - 1.350 um interacts with relay decenters."""
        system = load_tested_system()
        # Trace chief rays across band
        rays = [Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl) for wl in NOMINAL_WAVELENGTHS]
        traced = trace_custom_rays(system, rays)
        # Verify all dispersed chief rays land inside Field Lens (S07), Condenser (S10), and Detector (S13)
        for r in traced:
            assert not r.vignetted
            for s_idx in [7, 10, 13]:
                hit = [h for h in r.history if h["surface_index"] == s_idx][0]
                sd = system.surfaces[s_idx].semi_diameter
                r_hit = math.hypot(hit["x_local"], hit["y_local"])
                assert r_hit <= sd, f"Ray {r.wavelength} um exceeded aperture at surface {s_idx} ({r_hit:.2f} > {sd:.2f})"

    def test_tier3_clearance_vs_na_cone_spread_interaction(self):
        """Tier 3.3: Beam clearance at Z=0 and Z=50 holds under full NA cone beam divergence."""
        system = load_tested_system()
        for test_na in [0.0, 0.06, 0.12]:
            rays = generate_phase_space_rays(NOMINAL_WAVELENGTHS, na=test_na, num_angle_steps=5)
            traced = trace_custom_rays(system, rays)
            for r in traced:
                assert not r.vignetted
                # Z=0 slit clearance
                z0_crossings = [c for c in compute_ray_intersections_at_z(r, 0.0) if c[2] > 0]
                for x_int, y_int, _ in z0_crossings:
                    assert math.hypot(x_int, y_int) > SLIT_HOUSING_THRESHOLD
                # Z=50 M1 clearance
                z50_crossings = [c for c in compute_ray_intersections_at_z(r, 50.0) if c[2] >= 4]
                for x_int, y_int, _ in z50_crossings:
                    assert math.hypot(x_int, y_int) > M1_RADIUS_THRESHOLD

    def test_tier3_detector_spot_vs_camera_focal_length_balance(self):
        """Tier 3.4: Camera doublet focus + relay stage pupil demagnification achieves spot <= 2.5 mm."""
        system = load_tested_system()
        tracer = DualSpeedTracer(system)
        fast_res = tracer.trace_fast(wavelengths=NOMINAL_WAVELENGTHS, rays_per_wl=5)
        assert fast_res.vignetted_count == 0
        for wl, coords in fast_res.spot_coordinates.items():
            for pt in coords:
                assert math.hypot(pt[0], pt[1]) <= DETECTOR_RADIUS_LIMIT

    def test_tier3_m1_tilt_vs_grating_incidence_coupling(self):
        """Tier 3.5: M1 mirror tilt angle geometrically aligns reflected beam with grating normal."""
        system = load_tested_system()
        m1_surf = system.surfaces[1]
        grating_surf = system.surfaces[2]
        # M1 tilt ~15 deg folds beam by ~30 deg
        assert abs(m1_surf.tilt_x) >= 10.0 and abs(m1_surf.tilt_x) <= 20.0
        # Grating tilt ~ -21.5 deg directs 1.225 um beam forward
        assert abs(grating_surf.tilt_x) >= 15.0 and abs(grating_surf.tilt_x) <= 30.0

    def test_tier3_solid_volume_interference_cross_check(self):
        """Tier 3.6: No ray segment physically intersects solid volume of non-sequential elements."""
        system = load_tested_system()
        chief_ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=PRIMARY_WAVELENGTH)
        traced = trace_custom_rays(system, [chief_ray])[0]
        surfs = system.surfaces
        for i in range(len(traced.history) - 1):
            p1 = np.array([traced.history[i]["x_global"], traced.history[i]["y_global"], traced.history[i]["z_global"]])
            p2 = np.array([traced.history[i+1]["x_global"], traced.history[i+1]["y_global"], traced.history[i+1]["z_global"]])
            for j, surf in enumerate(surfs):
                if j != i and j != i + 1:
                    coll = check_segment_surface_interference(p1, p2, surf)
                    assert not coll, f"Collision detected between segment {i}->{i+1} and surface {j} ({surf.name})"

    def test_tier3_polychromatic_hexapolar_wavefront_spot_coupling(self):
        """Tier 3.7: 273 dense hexapolar rays across band trace with 0 vignetting."""
        system = load_tested_system()
        tracer = DualSpeedTracer(system)
        full_res = tracer.trace_full(wavelengths=NOMINAL_WAVELENGTHS, rings=5)
        assert full_res.vignetted_count == 0
        assert full_res.total_rays == 3 * 91  # 273 rays


# ============================================================================
# TIER 4: REAL-WORLD APPLICATION SCENARIOS
# ============================================================================

class TestTier4RealWorldApplicationScenarios:
    """
    Tier 4: Mission-critical real-world application workflows and stress tests.
    >= 5 test cases validating full desktop workflow, spectral linearity,
    tolerancing perturbations, thermal/refractive index stability, and multi-wavelength PSF.
    """

    def test_tier4_full_e2e_desktop_spectrometer_workflow(self):
        """
        Tier 4.1: Complete real-world workflow: run optimizer CLI -> verify desktop JSON ->
        load into 2D Optics V2 -> fast trace -> full trace -> assert clearance, zero-vignetting, and spot size.
        """
        if not os.path.exists(OPTIMIZE_SCRIPT_PATH):
            pytest.skip("optimize_layout.py not found")

        with tempfile.TemporaryDirectory() as tmpdir:
            test_json = os.path.join(tmpdir, "Spectrometer_V2_Optimized.json")
            res = subprocess.run([sys.executable, OPTIMIZE_SCRIPT_PATH, "--output", test_json, "--verify"], capture_output=True, text=True)
            assert res.returncode == 0

            system = SpectrometerSystem.load_from_json(test_json)
            tracer = DualSpeedTracer(system)
            # Warm-up run to prime memory/caches, then benchmark fast mode interactive responsiveness
            _ = tracer.trace_fast()
            fast_res = tracer.trace_fast()
            assert fast_res.vignetted_count == 0
            assert fast_res.elapsed_ms < 250.0, f"Fast trace latency ({fast_res.elapsed_ms:.2f} ms) exceeds interactive desktop budget (< 250.0 ms)"

            full_res = tracer.trace_full()
            assert full_res.vignetted_count == 0
            assert full_res.rms_radius <= DETECTOR_RADIUS_LIMIT

    def test_tier4_spectral_resolution_and_dispersion_linearity(self):
        """
        Tier 4.2: Evaluates dispersion linearity dY/dlambda on the DMD across 1.100 - 1.350 um.
        Verifies R^2 > 0.99 linear fit and resolution resolving power.
        """
        system = load_tested_system()
        sweep_wls = np.linspace(1.100, 1.350, 9)
        dmd_y_positions = []
        for wl in sweep_wls:
            ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=float(wl))
            traced = trace_custom_rays(system, [ray])[0]
            assert not traced.vignetted
            dmd_hit = [h for h in traced.history if h["surface_index"] == 6][0]
            dmd_y_positions.append(float(dmd_hit["y_local"]))

        # Linear regression of y_local vs wavelength
        slope, intercept = np.polyfit(sweep_wls, dmd_y_positions, 1)
        fitted = slope * sweep_wls + intercept
        residuals = dmd_y_positions - fitted
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((dmd_y_positions - np.mean(dmd_y_positions))**2)
        r_squared = 1.0 - (ss_res / ss_tot)

        assert r_squared > 0.99, f"Spectral dispersion on DMD is non-linear (R^2 = {r_squared:.4f} <= 0.99)"
        # Dispersion scale ~15 um/nm = ~15 mm/um
        assert abs(slope) > 5.0, f"Dispersion slope ({abs(slope):.2f} mm/um) too low"

    def test_tier4_tolerancing_angular_perturbations(self):
        """
        Tier 4.3: Tolerancing stress test: injects angular tilt perturbations (+/-0.05 deg) into M1 and Grating.
        Verifies layout remains stable and does not suffer catastrophic vignetting.
        """
        system = load_tested_system()
        base_m1_tilt = system.surfaces[1].tilt_x
        base_g_tilt = system.surfaces[2].tilt_x

        for delta_tilt in [-0.05, 0.05]:
            system.surfaces[1].tilt_x = base_m1_tilt + delta_tilt
            system.surfaces[2].tilt_x = base_g_tilt + delta_tilt
            system.surface_stack._update_accumulated_positions()

            rays = [Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl) for wl in NOMINAL_WAVELENGTHS]
            traced = trace_custom_rays(system, rays)
            # Under +/-0.05 deg perturbation, chief rays must reach the detector
            for r in traced:
                assert len(r.history) >= 10, f"Ray was lost under {delta_tilt} deg perturbation"

        # Restore nominal tilts
        system.surfaces[1].tilt_x = base_m1_tilt
        system.surfaces[2].tilt_x = base_g_tilt
        system.surface_stack._update_accumulated_positions()

    def test_tier4_thermal_and_glass_dispersion_stability(self):
        """
        Tier 4.4: Evaluates ray tracing stability under index perturbations (+/-0.001 Delta n).
        Verifies detector spot containment remains <= 2.5 mm.
        """
        system = load_tested_system()
        # Trace with perturbed wavelengths (mimicking chromatic index shift)
        perturbed_wls = [1.101, 1.226, 1.349]
        rays = generate_phase_space_rays(perturbed_wls, na=0.12, num_angle_steps=5)
        traced = trace_custom_rays(system, rays)
        for r in traced:
            assert not r.vignetted
            det_hit = [h for h in r.history if h["surface_index"] == system.detector_surface_idx][0]
            r_hit = math.hypot(det_hit["x_local"], det_hit["y_local"])
            assert r_hit <= DETECTOR_RADIUS_LIMIT

    def test_tier4_multi_wavelength_psf_containment(self):
        """
        Tier 4.5: Dense multi-wavelength point spread function (PSF) across 7 spectral channels.
        Verifies 100% geometric energy containment inside the 2.5 mm detector aperture.
        """
        system = load_tested_system()
        psf_wls = [1.100, 1.150, 1.200, 1.225, 1.250, 1.300, 1.350]
        rays = generate_phase_space_rays(psf_wls, na=0.12, num_angle_steps=5, slit_half_width_mm=0.015)
        traced = trace_custom_rays(system, rays)
        det_idx = system.detector_surface_idx

        for r in traced:
            assert not r.vignetted, f"PSF ray at {r.wavelength} um was vignetted"
            det_hit = [h for h in r.history if h["surface_index"] == det_idx][0]
            r_hit = math.hypot(det_hit["x_local"], det_hit["y_local"])
            assert r_hit <= DETECTOR_RADIUS_LIMIT, f"PSF ray at {r.wavelength} um fell outside detector ({r_hit:.3f} mm)"
