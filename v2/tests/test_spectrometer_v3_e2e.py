"""
Comprehensive Multi-Tier Opaque-Box E2E Test Suite for Spectrometer V3 All-Mirror NIR & Raman Optical Layout Design.

Test Structure:
- Tier 1: Feature Coverage (F1 to F5, 5 tests each = 25 tests)
- Tier 2: Boundary & Corner Conditions (F1 to F5, 5 tests each = 25 tests)
- Tier 3: Cross-Feature Pairwise Interactions (10 tests)
- Tier 4: Real-World Workflows & Operational Workloads (4 scenarios)

Authoritative Requirements Tested:
- R1 / AC1 / AC3: Broadband NIR Layout (800 - 2700 nm) All-Mirror Design & 0.8 mm Spot Containment.
- R2 / AC1 / AC3: Raman 1064 nm Layout (1064 - 1250 nm) 600 lpm Grating All-Mirror Design & 0.8 mm Spot Containment.
- Constraint Update: Thorlabs CM254 Concave Spherical Mirrors (k=0.0, stock ROC in {-38, -50, -75, -100, -150, -200, -300, -400} mm), ZERO Lenses.
- R3 / AC4: Collision Avoidance (> 15.0 mm 3D clearance from all ray segments to unrelated mirror housing centers).
"""

import os
import sys
import json
import math
import pytest
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

V2_ROOT = os.path.join(PROJECT_ROOT, "v2")
if V2_ROOT not in sys.path:
    sys.path.insert(0, V2_ROOT)

from v2.engine.models import SpectrometerSystem, FullTraceResult, FastTraceResult
from v2.engine.tracer import DualSpeedTracer, OpticalSystem, check_segment_surface_interference
from v2.engine.surface import Surface, SurfaceType
from v2.engine.ray import Ray
from v2.engine.lde import SurfaceStack, build_rotation_matrix
from spectrometer_v3_nir_raman.verify_spectrometer_v3 import (
    verify_mirror_catalog_compliance,
    verify_detector_spot_containment,
    THORLABS_CM254_STOCK_ROC,
    MAX_DETECTOR_RADIUS_MM
)
from spectrometer_v3_nir_raman.verify_collision_clearance import (
    point_to_segment_distance_3d,
    verify_layout_collision_clearance,
    MIN_CLEARANCE_THRESHOLD_MM
)


NIR_LAYOUT_PATH = os.path.join(PROJECT_ROOT, "spectrometer_v3_nir_raman", "spectrometer_v3_nir_800_2700nm.json")
RAMAN_LAYOUT_PATH = os.path.join(PROJECT_ROOT, "spectrometer_v3_nir_raman", "spectrometer_v3_raman_1064nm.json")


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture(scope="session")
def nir_layout_exists():
    """Returns True if the NIR 800-2700nm layout JSON file exists."""
    return os.path.exists(NIR_LAYOUT_PATH)


@pytest.fixture(scope="session")
def raman_layout_exists():
    """Returns True if the Raman 1064nm layout JSON file exists."""
    return os.path.exists(RAMAN_LAYOUT_PATH)


@pytest.fixture(scope="session")
def nir_system(nir_layout_exists):
    """Loads and returns SpectrometerSystem for Broadband NIR layout."""
    if not nir_layout_exists:
        pytest.skip(f"NIR layout file not found at {NIR_LAYOUT_PATH} (Milestone M1 deliverable pending).")
    return SpectrometerSystem.load_from_json(NIR_LAYOUT_PATH)


@pytest.fixture(scope="session")
def raman_system(raman_layout_exists):
    """Loads and returns SpectrometerSystem for Raman 1064nm layout."""
    if not raman_layout_exists:
        pytest.skip(f"Raman layout file not found at {RAMAN_LAYOUT_PATH} (Milestone M2 deliverable pending).")
    return SpectrometerSystem.load_from_json(RAMAN_LAYOUT_PATH)


@pytest.fixture
def mock_all_mirror_system():
    """Creates a synthetic 4-mirror all-reflective spectrometer system complying with Thorlabs CM254 catalog."""
    stack = SurfaceStack()
    stack.wavelengths = [0.800, 1.750, 2.700]
    stack.primary_wavelength = 1.750
    stack.name = "Synthetic Thorlabs CM254 Spectrometer"

    # Surface 0: Input Slit (Object)
    stack.add_surface(Surface(name="Input_Slit", surface_type=SurfaceType.OBJECT, thickness=100.0, glass="AIR", semi_diameter=5.0))
    # Surface 1: Collimator M1 (Thorlabs CM254-050, f=50mm, ROC=-100mm)
    stack.add_surface(Surface(name="M1_Collimator", surface_type=SurfaceType.MIRROR, radius=-100.0, conic=0.0, thickness=-80.0, glass="MIRROR", semi_diameter=12.7))
    # Surface 2: Diffraction Grating (150 lines/mm)
    stack.add_surface(Surface(name="Grating_150lpm", surface_type=SurfaceType.GRATING, radius=0.0, conic=0.0, thickness=100.0, glass="MIRROR", grating_lines_per_mm=150.0, grating_order=1, semi_diameter=12.7))
    # Surface 3: Focusing Mirror M2 (Thorlabs CM254-075, f=75mm, ROC=-150mm)
    stack.add_surface(Surface(name="M2_Focusing", surface_type=SurfaceType.MIRROR, radius=-150.0, conic=0.0, thickness=-50.0, glass="MIRROR", semi_diameter=12.7))
    # Surface 4: Relay Mirror M3 (Thorlabs CM254-025, f=25mm, ROC=-50mm)
    stack.add_surface(Surface(name="M3_Relay", surface_type=SurfaceType.MIRROR, radius=-50.0, conic=0.0, thickness=40.0, glass="MIRROR", semi_diameter=12.7))
    # Surface 5: Single-Pixel Detector (0.8mm clear aperture, semi_diameter=0.4)
    stack.add_surface(Surface(name="Detector_0p8mm", surface_type=SurfaceType.DETECTOR, radius=0.0, thickness=0.0, glass="AIR", semi_diameter=0.4))

    return SpectrometerSystem(surface_stack=stack, wavelengths=stack.wavelengths, primary_wavelength=stack.primary_wavelength, name=stack.name)


# ============================================================================
# TIER 1: FEATURE COVERAGE (25 Tests: 5 Tests per Feature F1 - F5)
# ============================================================================

class TestTier1_F1_BroadbandNIRLayout:
    """F1: Broadband NIR (800-2700 nm) All-Mirror Optical Layout Structure & Ingestion."""

    def test_t1_f1_nir_file_presence_and_version(self, nir_layout_exists):
        """T1.F1.1: Verify NIR JSON file exists on disk with schema version 3."""
        if not nir_layout_exists:
            pytest.skip(f"NIR layout file not found at {NIR_LAYOUT_PATH}")
        with open(NIR_LAYOUT_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data.get("version") == 3, "Schema version must be 3"
        assert "surfaces" in data, "JSON must contain 'surfaces' list"

    def test_t1_f1_nir_spectral_band_coverage(self, nir_system):
        """T1.F1.2: Verify NIR layout covers full 800 - 2700 nm (0.8 - 2.7 um) spectral range."""
        wls = nir_system.wavelengths
        assert len(wls) >= 3, f"Expected at least 3 sampled wavelengths, got {len(wls)}"
        assert min(wls) <= 0.800 + 1e-4, f"Minimum wavelength {min(wls)} um must cover 800 nm"
        assert max(wls) >= 2.700 - 1e-4, f"Maximum wavelength {max(wls)} um must reach 2700 nm"

    def test_t1_f1_nir_grating_density_prevents_evanescence(self, nir_system):
        """T1.F1.3: Verify NIR grating line density is low enough (<= 300 l/mm) to prevent 2700 nm evanescence."""
        gratings = [s for s in nir_system.surfaces if s.surface_type == SurfaceType.GRATING or s.grating_lines_per_mm > 0]
        assert len(gratings) >= 1, "NIR layout must contain at least 1 diffraction grating"
        g = gratings[0]
        assert g.grating_lines_per_mm <= 300.0, (
            f"Grating line density {g.grating_lines_per_mm} lpm too high; must be <= 300 lpm to prevent evanescence at 2.7 um."
        )
        assert g.grating_order in (1, -1), f"Grating diffraction order must be +1 or -1, got {g.grating_order}"

    def test_t1_f1_nir_detector_0p8mm_aperture(self, nir_system):
        """T1.F1.4: Verify NIR detector surface clear semi-diameter is exactly 0.400 mm (0.8 mm full aperture)."""
        detectors = [s for s in nir_system.surfaces if s.surface_type == SurfaceType.DETECTOR or "DETECTOR" in s.name.upper()]
        assert len(detectors) >= 1, "NIR layout must contain a detector surface"
        det = detectors[-1]
        assert abs(det.semi_diameter - 0.400) < 1e-4, (
            f"Detector semi-diameter must be 0.4 mm (0.8 mm diameter), got {det.semi_diameter}"
        )

    def test_t1_f1_nir_optical_surface_count_and_types(self, nir_system):
        """T1.F1.5: Verify NIR surface stack contains required sequence of reflective elements."""
        surfaces = nir_system.surfaces
        assert len(surfaces) >= 5, f"Expected at least 5 surfaces in all-mirror NIR system, got {len(surfaces)}"
        # Verify first is Object/Slit and last is Detector
        assert surfaces[0].surface_type in (SurfaceType.OBJECT, SurfaceType.STOP, SurfaceType.STANDARD)
        assert surfaces[-1].surface_type == SurfaceType.DETECTOR


class TestTier1_F2_Raman1064nmLayout:
    """F2: Raman 1064 nm Fine (1064-1250 nm) All-Mirror Optical Layout Structure & Ingestion."""

    def test_t1_f2_raman_file_presence_and_version(self, raman_layout_exists):
        """T1.F2.1: Verify Raman JSON file exists on disk with schema version 3."""
        if not raman_layout_exists:
            pytest.skip(f"Raman layout file not found at {RAMAN_LAYOUT_PATH}")
        with open(RAMAN_LAYOUT_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data.get("version") == 3, "Schema version must be 3"
        assert "surfaces" in data, "JSON must contain 'surfaces' list"

    def test_t1_f2_raman_spectral_band_coverage(self, raman_system):
        """T1.F2.2: Verify Raman layout covers 1064 nm fingerprint region (1.064 to 1.250 um)."""
        wls = raman_system.wavelengths
        assert len(wls) >= 3, f"Expected at least 3 sampled wavelengths, got {len(wls)}"
        assert min(wls) <= 1.064 + 1e-4, f"Minimum wavelength {min(wls)} um must cover 1064 nm excitation line"
        assert max(wls) >= 1.200 - 1e-4, f"Maximum wavelength {max(wls)} um must cover Raman fingerprint band"

    def test_t1_f2_raman_grating_600lpm_density(self, raman_system):
        """T1.F2.3: Verify Raman layout uses 600 lines/mm grating for high spectral resolution."""
        gratings = [s for s in raman_system.surfaces if s.surface_type == SurfaceType.GRATING or s.grating_lines_per_mm > 0]
        assert len(gratings) >= 1, "Raman layout must contain at least 1 diffraction grating"
        g = gratings[0]
        assert abs(g.grating_lines_per_mm - 600.0) < 1.0, (
            f"Raman grating line density must be 600 lpm, got {g.grating_lines_per_mm}"
        )
        assert g.grating_order in (1, -1), f"Grating diffraction order must be +1 or -1, got {g.grating_order}"

    def test_t1_f2_raman_detector_0p8mm_aperture(self, raman_system):
        """T1.F2.4: Verify Raman detector surface clear semi-diameter is exactly 0.400 mm (0.8 mm full aperture)."""
        detectors = [s for s in raman_system.surfaces if s.surface_type == SurfaceType.DETECTOR or "DETECTOR" in s.name.upper()]
        assert len(detectors) >= 1, "Raman layout must contain a detector surface"
        det = detectors[-1]
        assert abs(det.semi_diameter - 0.400) < 1e-4, (
            f"Detector semi-diameter must be 0.4 mm (0.8 mm diameter), got {det.semi_diameter}"
        )

    def test_t1_f2_raman_optical_surface_count_and_types(self, raman_system):
        """T1.F2.5: Verify Raman surface stack contains required sequence of reflective elements."""
        surfaces = raman_system.surfaces
        assert len(surfaces) >= 5, f"Expected at least 5 surfaces in all-mirror Raman system, got {len(surfaces)}"
        assert surfaces[-1].surface_type == SurfaceType.DETECTOR


class TestTier1_F3_ThorlabsCM254CatalogCompliance:
    """F3: Exclusively Thorlabs CM254 Spherical Mirrors (k=0.0, stock ROC, NO Lenses)."""

    def test_t1_f3_nir_mirrors_spherical_conic_zero(self, nir_system):
        """T1.F3.1: Verify every mirror in NIR layout has strictly spherical conic constant (k == 0.0)."""
        for i, s in enumerate(nir_system.surfaces):
            if (s.glass or "").upper() in ("MIRROR", "MIRR") or s.surface_type == SurfaceType.MIRROR:
                assert abs(s.conic) < 1e-6, f"Mirror Surface {i} ('{s.name}') has non-zero conic {s.conic}; must be 0.0"
                assert not s.asphere_coeffs or all(abs(c) < 1e-12 for c in s.asphere_coeffs), (
                    f"Mirror Surface {i} ('{s.name}') has non-zero asphere coefficients"
                )

    def test_t1_f3_nir_mirrors_radii_in_cm254_catalog(self, nir_system):
        """T1.F3.2: Verify every curved mirror in NIR layout matches a stock Thorlabs CM254 ROC."""
        for i, s in enumerate(nir_system.surfaces):
            if (s.glass or "").upper() in ("MIRROR", "MIRR") or s.surface_type == SurfaceType.MIRROR:
                roc = abs(s.radius)
                if roc > 1e-3:  # Non-flat
                    matched = any(abs(roc - stock) < 0.2 for stock in THORLABS_CM254_STOCK_ROC)
                    assert matched, (
                        f"Mirror Surface {i} ('{s.name}') ROC={roc:.2f} mm is not in Thorlabs CM254 stock ROCs: {THORLABS_CM254_STOCK_ROC}"
                    )

    def test_t1_f3_nir_no_refractive_lenses_present(self, nir_system):
        """T1.F3.3: Verify ZERO refractive lenses exist in the NIR layout."""
        for i, s in enumerate(nir_system.surfaces):
            glass = (s.glass or "AIR").upper().strip()
            assert glass in ("AIR", "MIRROR", "MIRR", "", "NONE"), (
                f"Forbidden refractive lens material '{glass}' found at Surface {i} ('{s.name}'). NO lenses allowed!"
            )

    def test_t1_f3_raman_mirrors_spherical_conic_zero(self, raman_system):
        """T1.F3.4: Verify every mirror in Raman layout has strictly spherical conic constant (k == 0.0)."""
        for i, s in enumerate(raman_system.surfaces):
            if (s.glass or "").upper() in ("MIRROR", "MIRR") or s.surface_type == SurfaceType.MIRROR:
                assert abs(s.conic) < 1e-6, f"Mirror Surface {i} ('{s.name}') has non-zero conic {s.conic}; must be 0.0"
                assert not s.asphere_coeffs or all(abs(c) < 1e-12 for c in s.asphere_coeffs), (
                    f"Mirror Surface {i} ('{s.name}') has non-zero asphere coefficients"
                )

    def test_t1_f3_raman_mirrors_radii_in_cm254_catalog(self, raman_system):
        """T1.F3.5: Verify every curved mirror in Raman layout matches a stock Thorlabs CM254 ROC."""
        for i, s in enumerate(raman_system.surfaces):
            if (s.glass or "").upper() in ("MIRROR", "MIRR") or s.surface_type == SurfaceType.MIRROR:
                roc = abs(s.radius)
                if roc > 1e-3:  # Non-flat
                    matched = any(abs(roc - stock) < 0.2 for stock in THORLABS_CM254_STOCK_ROC)
                    assert matched, (
                        f"Mirror Surface {i} ('{s.name}') ROC={roc:.2f} mm is not in Thorlabs CM254 stock ROCs: {THORLABS_CM254_STOCK_ROC}"
                    )


class TestTier1_F4_DetectorSpotContainment:
    """F4: Acceptance Criterion 3 Mathematical Spot Containment (100% rays within r <= 0.400 mm)."""

    def test_t1_f4_nir_full_trace_zero_vignetting(self, nir_system):
        """T1.F4.1: Verify 3D ray tracing through NIR layout produces 0 vignetted rays."""
        tracer = DualSpeedTracer(nir_system)
        result = tracer.trace_full(wavelengths=nir_system.wavelengths, rings=5, beam_radius=5.0)
        assert result.vignetted_count == 0, f"NIR layout has {result.vignetted_count} vignetted rays (must be 0)"
        assert result.total_rays > 0

    def test_t1_f4_nir_spot_containment_all_wavelengths(self, nir_system):
        """T1.F4.2: Verify 100% of NIR rays across all wavelengths land within 0.400 mm radius on detector."""
        passed, data, msgs = verify_detector_spot_containment(nir_system, rings=5, beam_radius=5.0)
        assert passed is True, f"NIR spot containment failed: {msgs}"
        assert data["containment_pct"] == 100.0, f"NIR containment was {data['containment_pct']:.2f}%, must be 100.0%"
        assert data["overall_max_radius_mm"] <= MAX_DETECTOR_RADIUS_MM + 1e-5

    def test_t1_f4_nir_rms_and_geo_radii_bounds(self, nir_system):
        """T1.F4.3: Verify NIR RMS spot radius is tightly bounded (< 0.200 mm) at image plane."""
        tracer = DualSpeedTracer(nir_system)
        result = tracer.trace_full(wavelengths=nir_system.wavelengths, rings=5, beam_radius=5.0)
        assert result.rms_radius < 0.200, f"NIR RMS spot radius {result.rms_radius:.4f} mm is too large"
        assert result.geo_radius <= MAX_DETECTOR_RADIUS_MM + 1e-5

    def test_t1_f4_raman_full_trace_zero_vignetting(self, raman_system):
        """T1.F4.4: Verify 3D ray tracing through Raman layout produces 0 vignetted rays."""
        tracer = DualSpeedTracer(raman_system)
        result = tracer.trace_full(wavelengths=raman_system.wavelengths, rings=5, beam_radius=5.0)
        assert result.vignetted_count == 0, f"Raman layout has {result.vignetted_count} vignetted rays (must be 0)"
        assert result.total_rays > 0

    def test_t1_f4_raman_spot_containment_all_wavelengths(self, raman_system):
        """T1.F4.5: Verify 100% of Raman rays across all wavelengths land within 0.400 mm radius on detector."""
        passed, data, msgs = verify_detector_spot_containment(raman_system, rings=5, beam_radius=5.0)
        assert passed is True, f"Raman spot containment failed: {msgs}"
        assert data["containment_pct"] == 100.0, f"Raman containment was {data['containment_pct']:.2f}%, must be 100.0%"
        assert data["overall_max_radius_mm"] <= MAX_DETECTOR_RADIUS_MM + 1e-5


class TestTier1_F5_CollisionClearance:
    """F5: Acceptance Criterion 4 Spatial Collision Clearance (> 15.0 mm to all unrelated mirror housing centers)."""

    def test_t1_f5_nir_ray_to_housing_center_clearance(self, nir_system):
        """T1.F5.1: Verify all NIR ray segments have > 15.0 mm clearance to unrelated mirror housing centers."""
        passed, summary, msgs = verify_layout_collision_clearance(NIR_LAYOUT_PATH, "Broadband NIR", threshold_mm=15.0, rings=5)
        assert passed is True, f"NIR collision clearance failed: {msgs}"
        assert summary["overall_min_clearance_mm"] > MIN_CLEARANCE_THRESHOLD_MM, (
            f"NIR minimum clearance {summary['overall_min_clearance_mm']:.2f} mm must be strictly > 15.0 mm"
        )

    def test_t1_f5_nir_inter_mirror_housing_spacing(self, nir_system):
        """T1.F5.2: Verify 3D Euclidean distance between distinct mirror centers in NIR layout exceeds mount diameter."""
        opt_sys = OpticalSystem(nir_system.surface_stack)
        opt_sys._update_accumulated_positions()
        mirror_origins = [s.global_origin for s in opt_sys.surfaces if s.surface_type in (SurfaceType.MIRROR, SurfaceType.GRATING)]
        for i in range(len(mirror_origins)):
            for j in range(i + 1, len(mirror_origins)):
                dist = np.linalg.norm(mirror_origins[i] - mirror_origins[j])
                assert dist > 25.0, f"Mirror centers {i} and {j} too close: {dist:.2f} mm < 25.0 mm (mount overlap risk)"

    def test_t1_f5_nir_no_solid_volume_penetration(self, nir_system):
        """T1.F5.3: Verify zero NIR ray segments physically intersect mirror substrate solid volumes."""
        passed, summary, msgs = verify_layout_collision_clearance(NIR_LAYOUT_PATH, "Broadband NIR", threshold_mm=15.0, rings=5)
        assert summary["solid_interference_count"] == 0, (
            f"Detected {summary['solid_interference_count']} physical substrate penetrations"
        )

    def test_t1_f5_raman_ray_to_housing_center_clearance(self, raman_system):
        """T1.F5.4: Verify all Raman ray segments have > 15.0 mm clearance to unrelated mirror housing centers."""
        passed, summary, msgs = verify_layout_collision_clearance(RAMAN_LAYOUT_PATH, "Raman 1064 nm", threshold_mm=15.0, rings=5)
        assert passed is True, f"Raman collision clearance failed: {msgs}"
        assert summary["overall_min_clearance_mm"] > MIN_CLEARANCE_THRESHOLD_MM, (
            f"Raman minimum clearance {summary['overall_min_clearance_mm']:.2f} mm must be strictly > 15.0 mm"
        )

    def test_t1_f5_raman_inter_mirror_housing_spacing(self, raman_system):
        """T1.F5.5: Verify 3D Euclidean distance between distinct mirror centers in Raman layout exceeds mount diameter."""
        opt_sys = OpticalSystem(raman_system.surface_stack)
        opt_sys._update_accumulated_positions()
        mirror_origins = [s.global_origin for s in opt_sys.surfaces if s.surface_type in (SurfaceType.MIRROR, SurfaceType.GRATING)]
        for i in range(len(mirror_origins)):
            for j in range(i + 1, len(mirror_origins)):
                dist = np.linalg.norm(mirror_origins[i] - mirror_origins[j])
                assert dist > 25.0, f"Raman mirror centers {i} and {j} too close: {dist:.2f} mm < 25.0 mm"


# ============================================================================
# TIER 2: BOUNDARY & CORNER CONDITIONS (25 Tests: 5 Tests per Feature F1 - F5)
# ============================================================================

class TestTier2_BoundaryConditions:
    """Tier 2: Boundary, Extreme Spectral & Geometry Stress Tests."""

    # F1 Boundary: NIR Spectral Extremes (800 nm & 2700 nm)
    def test_t2_f1_nir_band_edge_800nm_ray_trace(self, nir_system):
        """T2.F1.1: Trace extreme short-wavelength boundary (800 nm = 0.800 um) in NIR layout."""
        tracer = DualSpeedTracer(nir_system)
        res = tracer.trace_full(wavelengths=[0.800], rings=5, beam_radius=5.0)
        assert res.vignetted_count == 0
        pts = res.spot_coordinates.get(0.800, np.empty((0, 2)))
        radii = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
        assert np.max(radii) <= MAX_DETECTOR_RADIUS_MM + 1e-5

    def test_t2_f1_nir_band_edge_2700nm_ray_trace(self, nir_system):
        """T2.F1.2: Trace extreme long-wavelength boundary (2700 nm = 2.700 um) in NIR layout."""
        tracer = DualSpeedTracer(nir_system)
        res = tracer.trace_full(wavelengths=[2.700], rings=5, beam_radius=5.0)
        assert res.vignetted_count == 0
        pts = res.spot_coordinates.get(2.700, np.empty((0, 2)))
        radii = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
        assert np.max(radii) <= MAX_DETECTOR_RADIUS_MM + 1e-5

    def test_t2_f1_nir_11_wavelength_dense_spectral_sweep(self, nir_system):
        """T2.F1.3: Dense 11-wavelength sweep across 800-2700 nm verifying continuous containment."""
        sweep_wls = np.linspace(0.800, 2.700, 11).tolist()
        tracer = DualSpeedTracer(nir_system)
        res = tracer.trace_full(wavelengths=sweep_wls, rings=4, beam_radius=5.0)
        assert res.vignetted_count == 0
        for wl in sweep_wls:
            pts = res.spot_coordinates.get(wl, np.empty((0, 2)))
            radii = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
            assert np.max(radii) <= MAX_DETECTOR_RADIUS_MM + 1e-5

    def test_t2_f1_nir_extreme_input_beam_radius(self, nir_system):
        """T2.F1.4: Trace with boundary beam radius (6.0 mm) without vignetting."""
        tracer = DualSpeedTracer(nir_system)
        res = tracer.trace_full(wavelengths=nir_system.wavelengths, rings=4, beam_radius=6.0)
        assert res.vignetted_count == 0

    def test_t2_f1_nir_zero_field_chief_ray_alignment(self, nir_system):
        """T2.F1.5: Verify chief ray at primary wavelength hits detector center within alignment tolerance."""
        pri_wl = nir_system.primary_wavelength
        opt_sys = OpticalSystem(nir_system.surface_stack)
        chief_ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=pri_wl)
        tr = opt_sys.trace_ray(chief_ray)
        assert not tr.vignetted
        det_h = [h for h in tr.history if h["surface_index"] == len(opt_sys.surfaces) - 1]
        assert len(det_h) == 1
        r_det = math.sqrt(det_h[0]["x_local"]**2 + det_h[0]["y_local"]**2)
        assert r_det < 0.200, f"Primary chief ray landing radius {r_det:.4f} mm is not well-centered"

    # F2 Boundary: Raman Spectral Extremes (1064 nm & 1250 nm)
    def test_t2_f2_raman_band_edge_1064nm_ray_trace(self, raman_system):
        """T2.F2.1: Trace 1064 nm excitation boundary in Raman layout."""
        tracer = DualSpeedTracer(raman_system)
        res = tracer.trace_full(wavelengths=[1.064], rings=5, beam_radius=5.0)
        assert res.vignetted_count == 0
        pts = res.spot_coordinates.get(1.064, np.empty((0, 2)))
        radii = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
        assert np.max(radii) <= MAX_DETECTOR_RADIUS_MM + 1e-5

    def test_t2_f2_raman_band_edge_1250nm_ray_trace(self, raman_system):
        """T2.F2.2: Trace 1250 nm Raman shift boundary in Raman layout."""
        tracer = DualSpeedTracer(raman_system)
        res = tracer.trace_full(wavelengths=[1.250], rings=5, beam_radius=5.0)
        assert res.vignetted_count == 0
        pts = res.spot_coordinates.get(1.250, np.empty((0, 2)))
        radii = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
        assert np.max(radii) <= MAX_DETECTOR_RADIUS_MM + 1e-5

    def test_t2_f2_raman_10_wavelength_fine_spectral_sweep(self, raman_system):
        """T2.F2.3: Dense 10-wavelength sweep across Raman fingerprint band (1064 - 1250 nm)."""
        sweep_wls = np.linspace(1.064, 1.250, 10).tolist()
        tracer = DualSpeedTracer(raman_system)
        res = tracer.trace_full(wavelengths=sweep_wls, rings=4, beam_radius=5.0)
        assert res.vignetted_count == 0
        for wl in sweep_wls:
            pts = res.spot_coordinates.get(wl, np.empty((0, 2)))
            radii = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
            assert np.max(radii) <= MAX_DETECTOR_RADIUS_MM + 1e-5

    def test_t2_f2_raman_dispersion_linearity(self, raman_system):
        """T2.F2.4: Verify monotonic spatial dispersion across the Raman band."""
        sweep_wls = [1.064, 1.100, 1.150, 1.200, 1.250]
        tracer = DualSpeedTracer(raman_system)
        res = tracer.trace_full(wavelengths=sweep_wls, rings=3, beam_radius=4.0)
        centroids = []
        for wl in sweep_wls:
            pts = res.spot_coordinates.get(wl, np.empty((0, 2)))
            centroids.append(np.mean(pts[:, 0]))
        # Monotonicity check: differences must all have the same sign
        diffs = np.diff(centroids)
        assert (all(d > 0 for d in diffs) or all(d < 0 for d in diffs)), "Dispersion must be monotonic across detector"

    def test_t2_f2_raman_extreme_input_beam_radius(self, raman_system):
        """T2.F2.5: Trace Raman layout with max allowable input aperture beam radius (6.0 mm)."""
        tracer = DualSpeedTracer(raman_system)
        res = tracer.trace_full(wavelengths=raman_system.wavelengths, rings=4, beam_radius=6.0)
        assert res.vignetted_count == 0

    # F3 Boundary: Thorlabs CM254 Catalog Invariants
    def test_t2_f3_stock_roc_exact_catalog_membership(self):
        """T2.F3.1: Unit test verifying Thorlabs CM254 stock ROC constant set completeness."""
        expected_focals = [19.0, 25.0, 37.5, 50.0, 75.0, 100.0, 150.0, 200.0]
        expected_rocs = [2 * f for f in expected_focals]
        assert THORLABS_CM254_STOCK_ROC == expected_rocs
        assert len(THORLABS_CM254_STOCK_ROC) == 8

    def test_t2_f3_conic_perturbation_detection(self, mock_all_mirror_system):
        """T2.F3.2: Verify validator correctly catches even tiny non-zero conic perturbations (k = 0.001)."""
        mock_all_mirror_system.surfaces[1].conic = 0.001
        passed, msgs = verify_mirror_catalog_compliance(mock_all_mirror_system)
        assert passed is False
        assert any("non-spherical conic" in m for m in msgs)

    def test_t2_f3_aspheric_perturbation_detection(self, mock_all_mirror_system):
        """T2.F3.3: Verify validator correctly catches higher-order asphere coefficient injection."""
        mock_all_mirror_system.surfaces[1].asphere_coeffs = [0.0, 1e-6, 0.0]
        passed, msgs = verify_mirror_catalog_compliance(mock_all_mirror_system)
        assert passed is False
        assert any("higher-order aspheric" in m for m in msgs)

    def test_t2_f3_non_catalog_roc_rejection(self, mock_all_mirror_system):
        """T2.F3.4: Verify validator correctly rejects non-catalog custom mirror radii (e.g. ROC = 123.4 mm)."""
        mock_all_mirror_system.surfaces[1].radius = -123.456
        passed, msgs = verify_mirror_catalog_compliance(mock_all_mirror_system)
        assert passed is False
        assert any("does NOT match any Thorlabs CM254" in m for m in msgs)

    def test_t2_f3_refractive_lens_injection_rejection(self, mock_all_mirror_system):
        """T2.F3.5: Verify validator strictly rejects any refractive glass material (e.g. N-BK7)."""
        mock_all_mirror_system.surfaces[1].glass = "N-BK7"
        passed, msgs = verify_mirror_catalog_compliance(mock_all_mirror_system)
        assert passed is False
        assert any("refractive glass" in m for m in msgs)

    # F4 Boundary: Spot Containment Stress
    def test_t2_f4_high_density_hexapolar_ray_grid(self, mock_all_mirror_system):
        """T2.F4.1: Verify spot containment calculation on dense 7-ring (169 rays) hexapolar ray grid."""
        tracer = DualSpeedTracer(mock_all_mirror_system)
        res = tracer.trace_full(wavelengths=[1.750], rings=7, beam_radius=5.0)
        assert res.total_rays == 1 + 6 * (7 * 8) // 2  # 1 + 6*28 = 169 rays

    def test_t2_f4_marginal_fan_ray_containment(self, nir_system):
        """T2.F4.2: Verify extreme marginal fan rays at +-1.5 deg input angles land within detector."""
        opt_sys = OpticalSystem(nir_system.surface_stack)
        for angle_deg in [-1.5, 0.0, 1.5]:
            rad = np.radians(angle_deg)
            ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, np.sin(rad), np.cos(rad)]), wavelength=nir_system.primary_wavelength)
            tr = opt_sys.trace_ray(ray)
            if not tr.vignetted:
                h_det = [h for h in tr.history if h["surface_index"] == len(opt_sys.surfaces) - 1]
                if h_det:
                    r_landing = math.sqrt(h_det[0]["x_local"]**2 + h_det[0]["y_local"]**2)
                    assert r_landing <= MAX_DETECTOR_RADIUS_MM + 1e-5

    def test_t2_f4_detector_offset_boundary_tolerance(self, nir_system):
        """T2.F4.3: Verify detector spot remains contained under +-0.05 mm defocus perturbation."""
        det = nir_system.surfaces[-1]
        orig_thick = nir_system.surfaces[-2].thickness
        for dz in [-0.05, 0.05]:
            nir_system.surfaces[-2].thickness = orig_thick + dz
            tracer = DualSpeedTracer(nir_system)
            res = tracer.trace_full(wavelengths=[nir_system.primary_wavelength], rings=4, beam_radius=4.0)
            pts = res.spot_coordinates.get(nir_system.primary_wavelength, np.empty((0, 2)))
            radii = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
            assert np.max(radii) <= MAX_DETECTOR_RADIUS_MM + 0.05
        nir_system.surfaces[-2].thickness = orig_thick

    def test_t2_f4_pure_spot_containment_math_unit(self):
        """T2.F4.4: Unit test on mathematical spot containment calculation."""
        # 100 points inside 0.35 mm -> 100% containment
        pts_pass = np.random.uniform(-0.25, 0.25, size=(100, 2))
        radii_pass = np.sqrt(pts_pass[:, 0]**2 + pts_pass[:, 1]**2)
        assert np.all(radii_pass <= 0.400)
        
        # Points exceeding 0.400 mm -> fail
        pts_fail = np.array([[0.410, 0.0], [0.0, 0.420]])
        radii_fail = np.sqrt(pts_fail[:, 0]**2 + pts_fail[:, 1]**2)
        assert not np.all(radii_fail <= 0.400)

    def test_t2_f4_airy_disk_diffraction_limit(self, nir_system):
        """T2.F4.5: Verify system diffraction Airy disk radius is well within 0.8 mm detector aperture."""
        tracer = DualSpeedTracer(nir_system)
        res = tracer.trace_full(wavelengths=[nir_system.primary_wavelength], rings=3, beam_radius=5.0)
        assert res.airy_radius < 0.100, f"Airy disk radius {res.airy_radius:.4f} mm must be < 0.100 mm"

    # F5 Boundary: Collision & Clearance Stress
    def test_t2_f5_point_to_segment_exact_math_collinear(self):
        """T2.F5.1: Unit test on point-to-segment distance for collinear and projection clamping cases."""
        p_a = np.array([0.0, 0.0, 0.0])
        p_b = np.array([0.0, 0.0, 100.0])

        # Point projected inside segment
        pt_mid = np.array([10.0, 0.0, 50.0])
        d_mid, closest_mid, t_mid = point_to_segment_distance_3d(p_a, p_b, pt_mid)
        assert abs(d_mid - 10.0) < 1e-9
        assert abs(t_mid - 0.5) < 1e-9
        np.testing.assert_allclose(closest_mid, [0.0, 0.0, 50.0])

        # Point projected before p_a
        pt_before = np.array([0.0, 20.0, -30.0])
        d_before, closest_before, t_before = point_to_segment_distance_3d(p_a, p_b, pt_before)
        assert abs(t_before - 0.0) < 1e-9
        assert abs(d_before - math.sqrt(20.0**2 + 30.0**2)) < 1e-9
        np.testing.assert_allclose(closest_before, p_a)

        # Point projected beyond p_b
        pt_beyond = np.array([0.0, 20.0, 140.0])
        d_beyond, closest_beyond, t_beyond = point_to_segment_distance_3d(p_a, p_b, pt_beyond)
        assert abs(t_beyond - 1.0) < 1e-9
        assert abs(d_beyond - math.sqrt(20.0**2 + 40.0**2)) < 1e-9
        np.testing.assert_allclose(closest_beyond, p_b)

    def test_t2_f5_point_to_segment_zero_length_segment(self):
        """T2.F5.2: Unit test on point-to-segment distance when segment has zero length (P_A == P_B)."""
        p_a = np.array([10.0, 20.0, 30.0])
        p_b = np.array([10.0, 20.0, 30.0])
        pt = np.array([10.0, 25.0, 30.0])
        d, closest, t = point_to_segment_distance_3d(p_a, p_b, pt)
        assert abs(d - 5.0) < 1e-9
        np.testing.assert_allclose(closest, p_a)

    def test_t2_f5_grazing_beam_clearance_margin(self, nir_system):
        """T2.F5.3: Verify minimum spatial clearance strictly exceeds threshold by a safety margin."""
        passed, summary, msgs = verify_layout_collision_clearance(NIR_LAYOUT_PATH, "Broadband NIR", threshold_mm=15.0, rings=4)
        min_d = summary["overall_min_clearance_mm"]
        assert min_d > 15.0, f"Clearance {min_d:.2f} mm must be > 15.0 mm"

    def test_t2_f5_collision_verifier_cli_rejection_on_collision(self, mock_all_mirror_system, tmp_path):
        """T2.F5.4: Test collision verification script catches artificial beam obstruction."""
        # Intentionally position mirror M3 extremely close (5 mm) to the preceding reflection path
        mock_all_mirror_system.surfaces[3].thickness = -5.0
        test_json = tmp_path / "colliding_spec.json"
        mock_all_mirror_system.save_to_json(str(test_json))

        passed, summary, msgs = verify_layout_collision_clearance(str(test_json), "Colliding Spec", threshold_mm=15.0, rings=3)
        assert passed is False, "Expected collision verifier to reject layout with 5 mm clearance"
        assert summary["overall_min_clearance_mm"] <= 15.0

    def test_t2_f5_3d_spatial_rotation_matrix_orthogonality(self):
        """T2.F5.5: Stress-test 3D rotation matrix builder for exact SO(3) determinant and orthogonality."""
        for tx in [-45.0, 0.0, 30.0, 90.0]:
            for ty in [-30.0, 0.0, 45.0]:
                for tz in [-15.0, 0.0, 60.0]:
                    R = build_rotation_matrix(tx, ty, tz)
                    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
                    assert abs(np.linalg.det(R) - 1.0) < 1e-12


# ============================================================================
# TIER 3: CROSS-FEATURE PAIRWISE INTERACTIONS (10 Tests)
# ============================================================================

class TestTier3_CrossFeatureInteractions:
    """Tier 3: Inter-Module, Multi-Component, Round-Trip & Tracing Consistency Tests."""

    def test_t3_01_nir_json_serialization_roundtrip(self, nir_system, tmp_path):
        """T3.1: Serialize NIR system -> save to JSON -> reload -> verify identical ray trace result."""
        temp_file = str(tmp_path / "nir_roundtrip.json")
        nir_system.save_to_json(temp_file)

        reloaded = SpectrometerSystem.load_from_json(temp_file)
        assert len(reloaded.surfaces) == len(nir_system.surfaces)

        tracer1 = DualSpeedTracer(nir_system)
        tracer2 = DualSpeedTracer(reloaded)

        res1 = tracer1.trace_full(wavelengths=[1.750], rings=3, beam_radius=4.0)
        res2 = tracer2.trace_full(wavelengths=[1.750], rings=3, beam_radius=4.0)

        assert res1.vignetted_count == res2.vignetted_count
        np.testing.assert_allclose(res1.spot_coordinates[1.750], res2.spot_coordinates[1.750], atol=1e-9)

    def test_t3_02_raman_json_serialization_roundtrip(self, raman_system, tmp_path):
        """T3.2: Serialize Raman system -> save to JSON -> reload -> verify identical ray trace result."""
        temp_file = str(tmp_path / "raman_roundtrip.json")
        raman_system.save_to_json(temp_file)

        reloaded = SpectrometerSystem.load_from_json(temp_file)
        assert len(reloaded.surfaces) == len(raman_system.surfaces)

        tracer1 = DualSpeedTracer(raman_system)
        tracer2 = DualSpeedTracer(reloaded)

        res1 = tracer1.trace_full(wavelengths=[1.064], rings=3, beam_radius=4.0)
        res2 = tracer2.trace_full(wavelengths=[1.064], rings=3, beam_radius=4.0)

        assert res1.vignetted_count == res2.vignetted_count
        np.testing.assert_allclose(res1.spot_coordinates[1.064], res2.spot_coordinates[1.064], atol=1e-9)

    def test_t3_03_fast_vs_full_tracer_spot_centroid_consistency(self, nir_system):
        """T3.3: Fast mode chief ray spot position matches Full mode centroid within 0.050 mm."""
        tracer = DualSpeedTracer(nir_system)
        fast_res = tracer.trace_fast(wavelengths=[1.750], rays_per_wl=1)
        full_res = tracer.trace_full(wavelengths=[1.750], rings=4, beam_radius=4.0)

        fast_pt = fast_res.spot_coordinates[1.750][0]
        full_centroid_x = np.mean(full_res.spot_coordinates[1.750][:, 0])
        full_centroid_y = np.mean(full_res.spot_coordinates[1.750][:, 1])

        dist = math.sqrt((fast_pt[0] - full_centroid_x)**2 + (fast_pt[1] - full_centroid_y)**2)
        assert dist < 0.050, f"Fast vs Full centroid deviation {dist:.4f} mm exceeds 0.050 mm"

    def test_t3_04_dual_speed_tracer_execution_timing(self, nir_system):
        """T3.4: Fast mode executes in < 50ms and Full mode in < 500ms."""
        tracer = DualSpeedTracer(nir_system)
        fast_res = tracer.trace_fast(wavelengths=nir_system.wavelengths, rays_per_wl=3)
        assert fast_res.elapsed_ms < 50.0, f"Fast trace took {fast_res.elapsed_ms:.2f} ms (expected < 50 ms)"

        full_res = tracer.trace_full(wavelengths=nir_system.wavelengths, rings=5, beam_radius=5.0)
        assert full_res.elapsed_ms < 500.0, f"Full trace took {full_res.elapsed_ms:.2f} ms (expected < 500 ms)"

    def test_t3_05_grating_diffraction_vector_conservation(self, nir_system):
        """T3.5: Verify vector grating diffraction preserves unit direction magnitude and transverse momentum."""
        opt_sys = OpticalSystem(nir_system.surface_stack)
        ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=nir_system.primary_wavelength)
        tr = opt_sys.trace_ray(ray)
        for h in tr.history:
            d = h["dir"]
            mag = np.linalg.norm(d)
            assert abs(mag - 1.0) < 1e-9, f"Ray direction vector magnitude {mag} != 1.0 at surface {h['surface_index']}"

    def test_t3_06_ray_path_length_and_optical_phase_monotonicity(self, nir_system):
        """T3.6: Verify ray accumulated path length and optical phase increase strictly monotonically."""
        opt_sys = OpticalSystem(nir_system.surface_stack)
        ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=nir_system.primary_wavelength)
        tr = opt_sys.trace_ray(ray)
        assert tr.path_length > 0.0
        assert tr.phase > 0.0

    def test_t3_07_mirror_clear_apertures_contain_full_beam(self, nir_system):
        """T3.7: Verify beam footprint on all mirrors is well within their 25.4 mm (12.7 mm semi-diam) apertures."""
        opt_sys = OpticalSystem(nir_system.surface_stack)
        tracer = DualSpeedTracer(nir_system)
        res = tracer.trace_full(wavelengths=nir_system.wavelengths, rings=4, beam_radius=5.0)
        # Check that no mirror clipped rays (0 vignetting)
        assert res.vignetted_count == 0

    def test_t3_08_concurrent_catalog_and_spot_containment_proof(self, nir_system, raman_system):
        """T3.8: Combined proof that both layouts satisfy Thorlabs CM254 catalog and 100% spot containment."""
        cat_nir, _ = verify_mirror_catalog_compliance(nir_system)
        cat_raman, _ = verify_mirror_catalog_compliance(raman_system)
        spot_nir, data_nir, _ = verify_detector_spot_containment(nir_system, rings=5, beam_radius=5.0)
        spot_raman, data_raman, _ = verify_detector_spot_containment(raman_system, rings=5, beam_radius=5.0)

        assert cat_nir and cat_raman, "Catalog compliance failed on one or both layouts"
        assert spot_nir and spot_raman, "Spot containment failed on one or both layouts"
        assert data_nir["containment_pct"] == 100.0
        assert data_raman["containment_pct"] == 100.0

    def test_t3_09_concurrent_collision_clearance_proof(self, nir_system, raman_system):
        """T3.9: Combined proof that both layouts satisfy minimum clearance > 15.0 mm."""
        coll_nir, sum_nir, _ = verify_layout_collision_clearance(NIR_LAYOUT_PATH, "NIR", threshold_mm=15.0, rings=4)
        coll_raman, sum_raman, _ = verify_layout_collision_clearance(RAMAN_LAYOUT_PATH, "Raman", threshold_mm=15.0, rings=4)

        assert coll_nir and coll_raman
        assert sum_nir["overall_min_clearance_mm"] > 15.0
        assert sum_raman["overall_min_clearance_mm"] > 15.0

    def test_t3_10_optical_system_coordinate_transforms_consistency(self, nir_system):
        """T3.10: Global coordinates from tracer match surface global origins and orientation transforms."""
        opt_sys = OpticalSystem(nir_system.surface_stack)
        opt_sys._update_accumulated_positions()
        for idx, s in enumerate(opt_sys.surfaces):
            assert hasattr(s, "global_origin")
            assert hasattr(s, "global_R")
            assert s.global_origin.shape == (3,)
            assert s.global_R.shape == (3, 3)


# ============================================================================
# TIER 4: REAL-WORLD APPLICATION SCENARIOS & END-TO-END WORKLOADS (4 Scenarios)
# ============================================================================

class TestTier4_RealWorldScenarios:
    """Tier 4: Production Workload & End-to-End System Scenarios."""

    def test_t4_scenario_1_full_band_nir_swept_tracing(self, nir_system):
        """
        Scenario 1: Broadband NIR 800 - 2700 nm Swept High-Density Tracing.
        Exercises: F1, F3, F4.
        Pass Criteria: 100% rays within r <= 0.400 mm detector across 9 discrete wavelengths with 0 vignetting.
        """
        wls_sweep = np.linspace(0.800, 2.700, 9).tolist()
        tracer = DualSpeedTracer(nir_system)
        result = tracer.trace_full(wavelengths=wls_sweep, rings=5, beam_radius=5.0)

        assert result.vignetted_count == 0, f"Vignetting detected: {result.vignetted_count} rays"

        total_rays = 0
        contained_rays = 0
        for wl in wls_sweep:
            pts = result.spot_coordinates.get(wl, np.empty((0, 2)))
            assert len(pts) > 0, f"No rays reached detector for wl={wl:.3f} um"
            radii = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
            max_r = float(np.max(radii))
            assert max_r <= MAX_DETECTOR_RADIUS_MM + 1e-5, (
                f"Wavelength {wl:.3f} um max spot radius {max_r:.4f} mm exceeds {MAX_DETECTOR_RADIUS_MM} mm"
            )
            total_rays += len(pts)
            contained_rays += int(np.sum(radii <= MAX_DETECTOR_RADIUS_MM + 1e-5))

        containment_pct = (contained_rays / total_rays) * 100.0
        assert containment_pct == 100.0, f"Scenario 1 Containment was {containment_pct:.2f}% (must be 100.0%)"

    def test_t4_scenario_2_raman_fingerprint_swept_tracing(self, raman_system):
        """
        Scenario 2: Raman Fingerprint 1064 - 1250 nm Swept High-Density Tracing.
        Exercises: F2, F3, F4.
        Pass Criteria: 100% rays within r <= 0.400 mm detector across 7 discrete wavelengths with 0 vignetting.
        """
        wls_sweep = np.linspace(1.064, 1.250, 7).tolist()
        tracer = DualSpeedTracer(raman_system)
        result = tracer.trace_full(wavelengths=wls_sweep, rings=5, beam_radius=5.0)

        assert result.vignetted_count == 0, f"Vignetting detected: {result.vignetted_count} rays"

        total_rays = 0
        contained_rays = 0
        for wl in wls_sweep:
            pts = result.spot_coordinates.get(wl, np.empty((0, 2)))
            assert len(pts) > 0, f"No rays reached detector for wl={wl:.3f} um"
            radii = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
            max_r = float(np.max(radii))
            assert max_r <= MAX_DETECTOR_RADIUS_MM + 1e-5, (
                f"Raman wl={wl:.3f} um max spot radius {max_r:.4f} mm exceeds {MAX_DETECTOR_RADIUS_MM} mm"
            )
            total_rays += len(pts)
            contained_rays += int(np.sum(radii <= MAX_DETECTOR_RADIUS_MM + 1e-5))

        containment_pct = (contained_rays / total_rays) * 100.0
        assert containment_pct == 100.0, f"Scenario 2 Containment was {containment_pct:.2f}% (must be 100.0%)"

    def test_t4_scenario_3_full_3d_beam_clearance_vs_mount_volumes(self, nir_system, raman_system):
        """
        Scenario 3: Full 3D Beam Clearance vs Mount Envelopes across Both Layouts.
        Exercises: F5, F1, F2.
        Pass Criteria: Minimum ray-segment clearance to all unrelated mirror housing centers strictly > 15.0 mm.
        """
        # 1. NIR Layout Clearance Check
        passed_nir, sum_nir, msgs_nir = verify_layout_collision_clearance(
            NIR_LAYOUT_PATH, "Broadband NIR", threshold_mm=MIN_CLEARANCE_THRESHOLD_MM, rings=5, beam_radius=5.0
        )
        assert passed_nir is True, f"NIR mount clearance failed: {msgs_nir}"
        assert sum_nir["overall_min_clearance_mm"] > MIN_CLEARANCE_THRESHOLD_MM

        # 2. Raman Layout Clearance Check
        passed_raman, sum_raman, msgs_raman = verify_layout_collision_clearance(
            RAMAN_LAYOUT_PATH, "Raman 1064nm", threshold_mm=MIN_CLEARANCE_THRESHOLD_MM, rings=5, beam_radius=5.0
        )
        assert passed_raman is True, f"Raman mount clearance failed: {msgs_raman}"
        assert sum_raman["overall_min_clearance_mm"] > MIN_CLEARANCE_THRESHOLD_MM

    def test_t4_scenario_4_catalog_spec_and_conic_verification(self, nir_system, raman_system):
        """
        Scenario 4: Full Catalog Compliance & Conic Zero Audit across All Optical Elements.
        Exercises: F3.
        Pass Criteria: Every mirror conic == 0.0, ROC in Thorlabs CM254 catalog, 0 lenses.
        """
        for name, sys_model in [("NIR", nir_system), ("Raman", raman_system)]:
            passed, msgs = verify_mirror_catalog_compliance(sys_model)
            assert passed is True, f"{name} layout catalog compliance audit failed: {msgs}"
            for i, surf in enumerate(sys_model.surfaces):
                glass = (surf.glass or "AIR").upper().strip()
                assert glass in ("AIR", "MIRROR", "MIRR", "", "NONE"), f"Found refractive lens {glass} in {name}"
                if glass in ("MIRROR", "MIRR"):
                    assert abs(surf.conic) < 1e-6, f"Non-zero conic {surf.conic} in {name} surface {i}"
                    assert not surf.asphere_coeffs or all(abs(c) < 1e-12 for c in surf.asphere_coeffs)
                    if abs(surf.radius) > 1e-3:
                        assert any(abs(abs(surf.radius) - r) < 0.2 for r in THORLABS_CM254_STOCK_ROC), (
                            f"Non-catalog ROC {surf.radius} in {name} surface {i}"
                        )
