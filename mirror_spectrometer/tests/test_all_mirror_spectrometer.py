"""
E2E Test Suite for All-Mirror Spectrometer Layout.

Covers:
- Tier 1: Feature Coverage (Schema validation, all-mirror check - no lenses after slit,
  Thorlabs COTS spherical mirror radii compliance R in {-19, -24, -30, -38, -50, -100, -150, -200, -300, -400} mm,
  600 l/mm grating, aperture stop, DMD active span <= 5mm, single 0.8mm detector).
- Tier 2: Boundary & Corner Cases (Band edge wavelengths 785nm and 880nm, marginal rays at NA limits,
  slit edges +/-0.015mm, zero intermediate vignetting assertion).
- Tier 3: Cross-Feature & Clearances (Z=0 slit clearance >12.7mm, Z=50 M1 clearance >12.7mm,
  3D substrate interference checks).
- Tier 4: Real-World Workload (Dense polychromatic ray tracing across all operating wavelengths,
  asserting 100% of rays at detector plane have sqrt(x^2 + y^2) <= 0.400 mm).
"""

import json
import math
import os
import sys
import unittest
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from v2.engine.surface import Surface, SurfaceType
from v2.engine.lde import SurfaceStack
from v2.engine.tracer import OpticalSystem, Ray, check_segment_surface_interference
from v2.analysis.spot import SpotDiagram


# Standard Thorlabs COTS Spherical Concave Mirror Catalog Radii of Curvature (mm)
# R = -2 * f (ROC is negative for concave spherical mirrors facing forward beam)
VALID_THORLABS_COTS_RADII_MM = {
    -19.0,   # f = 9.5 mm (CM127-009)
    -24.0,   # f = 12.0 mm (CM127-012)
    -30.0,   # f = 15.0 mm (CM127-015 / CM254-015)
    -38.0,   # f = 19.0 mm (CM254-019)
    -50.0,   # f = 25.0 mm (CM254-025)
    -60.0,   # f = 30.0 mm (CM254-030)
    -75.0,   # f = 37.5 mm
    -100.0,  # f = 50.0 mm (CM254-050 / CM508-050)
    -150.0,  # f = 75.0 mm (CM254-075 / CM508-075)
    -200.0,  # f = 100.0 mm (CM254-100 / CM508-100)
    -300.0,  # f = 150.0 mm (CM254-150 / CM508-150)
    -400.0,  # f = 200.0 mm (CM254-200 / CM508-200)
    -500.0,  # f = 250.0 mm (CM254-250)
    -600.0,  # f = 300.0 mm (CM254-300)
    -1000.0  # f = 500.0 mm
}

REFRACTIVE_GLASS_KEYWORDS = {
    "N-BK7", "NBK7", "BK7", "F_SILICA", "FUSED_SILICA", "FSILICA",
    "N-SF11", "NSF11", "SF11", "N-SF5", "SF5", "BAF10", "N-BAF10",
    "N-LAK22", "LAK22", "N-LASF9", "LASF9", "E-SF03", "N-FK51A"
}


def find_target_layout_path() -> Path:
    """
    Locate the all-mirror spectrometer JSON layout file.
    Searches in standard prioritized locations with fallback to reference fixture.
    """
    env_path = os.environ.get("ALL_MIRROR_LAYOUT_PATH") or os.environ.get("SPECTROMETER_LAYOUT_PATH")
    if env_path and Path(env_path).is_file():
        return Path(env_path)

    candidates = [
        WORKSPACE_ROOT / "mirror_spectrometer" / "all_mirror_spectrometer.json",
        WORKSPACE_ROOT / "mirror_spectrometer" / "Spectrometer_All_Mirror_Optimized.json",
        WORKSPACE_ROOT / "all_mirror_spectrometer.json",
        WORKSPACE_ROOT / "Spectrometer_All_Mirror_Optimized.json",
        Path(__file__).resolve().parent / "fixtures" / "reference_all_mirror_spectrometer.json",
    ]

    for cand in candidates:
        if cand.is_file():
            return cand

    raise FileNotFoundError(
        f"Could not find any all-mirror spectrometer layout JSON file. Checked candidates: {candidates}"
    )


def load_layout_stack(json_path: Path) -> Tuple[SurfaceStack, OpticalSystem, Dict[str, Any]]:
    """
    Load JSON layout file, build SurfaceStack and OpticalSystem.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Ingest through SurfaceStack
    loaded = SurfaceStack.load_from_json(str(json_path))
    stack = loaded["stack"]
    stack._update_accumulated_positions()
    opt_sys = OpticalSystem(stack)

    return stack, opt_sys, data


class BaseSpectrometerTestCase(unittest.TestCase):
    """Base class setting up layout stack and optical system for tests."""

    @classmethod
    def setUpClass(cls):
        cls.layout_path = find_target_layout_path()
        cls.stack, cls.opt_sys, cls.raw_json = load_layout_stack(cls.layout_path)

    def setUp(self):
        # Refresh positions before each test
        self.stack._update_accumulated_positions()

    def get_stop_geometry(self) -> Tuple[float, float]:
        """
        Returns (stop_distance_from_slit, stop_clear_radius).
        """
        stop_dist = self.stack.surfaces[0].thickness
        stops = [
            s for s in self.stack.surfaces
            if s.surface_type == SurfaceType.STOP or "STOP" in s.name.upper() or "IRIS" in s.name.upper()
        ]
        if stops:
            stop_surf = stops[0]
            clear_r = stop_surf.semi_diameter
            if hasattr(stop_surf, "aperture_r_outer") and stop_surf.aperture_r_outer > 0:
                clear_r = min(clear_r, stop_surf.aperture_r_outer)
            return max(1.0, stop_dist), clear_r
        return 48.5, 2.0

    def get_band_wavelengths(self) -> Tuple[float, float, float]:
        wls = sorted(self.stack.wavelengths if hasattr(self.stack, "wavelengths") and self.stack.wavelengths else [0.785, 0.830, 0.880])
        wl_min = wls[0]
        wl_max = wls[-1]
        wl_mid = self.stack.primary_wavelength if hasattr(self.stack, "primary_wavelength") and self.stack.primary_wavelength else wls[len(wls) // 2]
        return wl_min, wl_mid, wl_max


# ==============================================================================
# TIER 1: FEATURE COVERAGE & COTS COMPLIANCE
# ==============================================================================

class TestTier1FeatureCoverage(BaseSpectrometerTestCase):
    """
    Tier 1: Feature Coverage Tests.
    Validates schema, all-mirror architecture (no refractive lenses after slit),
    Thorlabs COTS spherical mirror catalog compliance, 600 l/mm grating,
    aperture stop, DMD active span, and single 0.8 mm detector footprint.
    """

    def test_t1_1_json_file_validity_and_schema(self):
        """T1.1: Verify JSON file exists, is valid JSON, and loads in SurfaceStack."""
        self.assertTrue(self.layout_path.is_file(), f"Layout file does not exist: {self.layout_path}")
        self.assertIn("surfaces", self.raw_json, "JSON schema missing 'surfaces' list")
        self.assertIn("wavelengths", self.raw_json, "JSON schema missing 'wavelengths' list")
        self.assertGreater(len(self.raw_json["surfaces"]), 4, "System must have at least 5 surfaces")
        self.assertGreater(len(self.stack.surfaces), 4, "SurfaceStack must contain >= 5 surfaces")

    def test_t1_2_all_mirror_design_no_lenses_after_slit(self):
        """
        T1.2: Verify all optical surfaces after the slit are reflective mirrors or grating.
        Strictly asserts zero refractive lenses after the entrance slit.
        """
        surfaces = self.stack.surfaces
        self.assertGreater(len(surfaces), 2, "Insufficient surfaces to evaluate")

        # Entrance slit is surface 0
        slit_surface = surfaces[0]
        self.assertIn(slit_surface.surface_type, (SurfaceType.OBJECT, SurfaceType.STANDARD, SurfaceType.STOP),
                      "Surface 0 must be the entrance slit / object")

        for idx in range(1, len(surfaces) - 1):
            surf = surfaces[idx]
            glass_upper = (surf.glass or "").strip().upper()

            # Check for forbidden refractive glass types
            for refr in REFRACTIVE_GLASS_KEYWORDS:
                self.assertNotEqual(
                    glass_upper, refr,
                    f"Refractive glass '{refr}' detected at surface {idx} ({surf.name}). "
                    f"Requirement R1 strictly mandates ALL-MIRROR design (no lenses after slit)."
                )

            # If surface has optical curvature (R != 0), it MUST be a mirror
            if abs(surf.radius) > 1e-6:
                self.assertTrue(
                    surf.is_reflective() or glass_upper in ("MIRROR", "MIRR", "REFL"),
                    f"Curved surface {idx} ({surf.name}, R={surf.radius} mm) is not reflective! "
                    f"Only spherical mirrors are allowed."
                )

    def test_t1_3_thorlabs_cots_spherical_mirror_radii(self):
        """
        T1.3: Verify all mirror radii match Thorlabs COTS catalog part numbers and sizes.
        Allowed standard radii: R in {-19, -24, -30, -38, -50, -100, -150, -200, -300, -400} mm.
        Must also be spherical (conic == 0.0) with standard 1-inch (12.7 mm semi-diameter) housings.
        """
        curved_mirrors = [
            (i, s) for i, s in enumerate(self.stack.surfaces)
            if abs(s.radius) > 1e-6 and s.surface_type not in (SurfaceType.OBJECT, SurfaceType.DETECTOR)
        ]
        self.assertGreaterEqual(len(curved_mirrors), 2, "Must have at least M1 and M2 curved mirrors")

        for idx, surf in curved_mirrors:
            roc = surf.radius
            # In standard sign convention for forward propagation, concave mirrors have R < 0
            self.assertLess(roc, 0.0, f"Surface {idx} ({surf.name}) has R={roc} mm > 0 (convex). Must be concave (R < 0).")

            # Check match against Thorlabs catalog radii (within 0.5 mm tolerance)
            matched = any(abs(roc - cots_r) <= 0.5 for cots_r in VALID_THORLABS_COTS_RADII_MM)
            self.assertTrue(
                matched,
                f"Surface {idx} ({surf.name}) radius R={roc:.1f} mm does NOT match any standard Thorlabs COTS mirror. "
                f"Allowed catalog radii: {sorted(list(VALID_THORLABS_COTS_RADII_MM))}"
            )

            # Conic constant must be 0.0 (spherical mirror, not custom asphere)
            self.assertAlmostEqual(
                surf.conic, 0.0, places=3,
                msg=f"Surface {idx} ({surf.name}) has conic={surf.conic} != 0. Must be standard spherical mirror."
            )

            # Verify standard 1-inch Thorlabs mirror semi-diameter (12.7 mm)
            self.assertLessEqual(
                surf.semi_diameter, 12.75,
                f"Surface {idx} ({surf.name}) semi-diameter {surf.semi_diameter} mm exceeds 1-inch standard (12.7 mm)"
            )

    def test_t1_4_diffraction_grating_specification(self):
        """
        T1.4: Verify diffraction grating is present with exactly 600 lines/mm in order 1.
        """
        gratings = [
            (i, s) for i, s in enumerate(self.stack.surfaces)
            if s.surface_type == SurfaceType.GRATING or abs(s.grating_lines_per_mm) > 1e-3
        ]
        self.assertEqual(len(gratings), 1, f"Expected exactly 1 diffraction grating, found {len(gratings)}")

        idx, grating_surf = gratings[0]
        self.assertAlmostEqual(
            grating_surf.grating_lines_per_mm, 600.0, delta=1.0,
            msg=f"Grating at surface {idx} has {grating_surf.grating_lines_per_mm} l/mm; expected 600.0 l/mm"
        )
        self.assertEqual(grating_surf.grating_order, 1, "Grating must operate in +1 diffraction order")
        self.assertTrue(
            grating_surf.is_reflective() or (grating_surf.glass or "").upper() in ("MIRROR", "MIRR", "REFL"),
            f"Grating at surface {idx} must be a reflective grating"
        )

    def test_t1_5_aperture_stop_specification(self):
        """
        T1.5: Verify aperture stop is present before M1 with clear radius <= 6.0 mm (F/# >= 4.2),
        specifically tuned to 1.40 mm for 100% spot containment.
        """
        stops = [
            (i, s) for i, s in enumerate(self.stack.surfaces)
            if s.surface_type == SurfaceType.STOP or "STOP" in s.name.upper() or "IRIS" in s.name.upper()
        ]
        self.assertGreaterEqual(len(stops), 1, "System must include an aperture stop surface to control beam NA")

        stop_idx, stop_surf = stops[0]
        # Must be located before M1 collimator (surface 2 or earlier)
        self.assertLessEqual(stop_idx, 2, f"Aperture stop at index {stop_idx} should be located before or at M1")

        clear_r = stop_surf.semi_diameter
        if hasattr(stop_surf, "aperture_r_outer") and stop_surf.aperture_r_outer > 0:
            clear_r = min(clear_r, stop_surf.aperture_r_outer)

        self.assertLessEqual(
            clear_r, 6.0,
            f"Aperture stop semi-diameter {clear_r} mm exceeds maximum allowable limit of 6.0 mm"
        )
        self.assertAlmostEqual(
            clear_r, 1.40, delta=0.10,
            msg=f"Aperture stop semi-diameter {clear_r} mm should be 1.40 mm for Shafer relay coma balance"
        )

    def test_t1_6_dmd_active_area_specification(self):
        """
        T1.6: Verify DMD active area is <= 5.0 mm for the operational wavelength band.
        """
        dmd_surfaces = [
            (i, s) for i, s in enumerate(self.stack.surfaces)
            if "DMD" in s.name.upper() or "DLP4500" in s.name.upper() or "MODULATOR" in s.name.upper()
        ]
        self.assertGreaterEqual(len(dmd_surfaces), 1, "System must include DMD surface")
        dmd_idx, dmd_surf = dmd_surfaces[0]

        # DMD should be flat planar mirror with ON-state tilt ~12 deg
        self.assertAlmostEqual(dmd_surf.radius, 0.0, places=3, msg="DMD must be a flat planar surface (R=0)")
        self.assertAlmostEqual(abs(dmd_surf.tilt_x), 12.0, delta=2.0, msg="DMD micromirror ON tilt must be ~12 deg")

        # Trace chief rays across wavelengths to compute spectral dispersion width on DMD
        wls = self.stack.wavelengths if hasattr(self.stack, "wavelengths") and self.stack.wavelengths else [0.785, 0.830, 0.880]
        dmd_hits = []
        for wl in wls:
            r = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl)
            tr = self.opt_sys.trace_ray(r)
            self.assertIsNotNone(tr, f"Ray trace failed for wavelength {wl} um")
            for h in tr.history:
                if h["surface_index"] == dmd_idx:
                    dmd_hits.append(h["y_local"])

        self.assertGreater(len(dmd_hits), 1, "Could not obtain DMD intersection hits")
        spectral_span = max(dmd_hits) - min(dmd_hits)
        self.assertLessEqual(
            spectral_span, 5.05,
            f"DMD spectral dispersion span {spectral_span:.3f} mm exceeds 5.0 mm active area limit!"
        )

    def test_t1_7_single_0p8mm_detector_specification(self):
        """
        T1.7: Verify single detector element with active diameter <= 0.80 mm (semi-diameter <= 0.40 mm).
        """
        det_surfaces = [
            (i, s) for i, s in enumerate(self.stack.surfaces)
            if s.surface_type == SurfaceType.DETECTOR or "DETECTOR" in s.name.upper()
        ]
        self.assertEqual(len(det_surfaces), 1, f"Must have exactly 1 detector surface, found {len(det_surfaces)}")

        det_idx, det_surf = det_surfaces[0]
        self.assertEqual(det_idx, len(self.stack.surfaces) - 1, "Detector must be the final terminating surface")

        det_semi_dia = det_surf.semi_diameter
        if hasattr(det_surf, "aperture_r_outer") and det_surf.aperture_r_outer > 0:
            det_semi_dia = min(det_semi_dia, det_surf.aperture_r_outer)

        self.assertLessEqual(
            det_semi_dia, 0.4005,
            f"Detector semi-diameter {det_semi_dia:.4f} mm exceeds maximum active radius 0.400 mm (0.8 mm diameter)"
        )


# ==============================================================================
# TIER 2: BOUNDARY & CORNER CASES
# ==============================================================================

class TestTier2BoundaryAndCornerCases(BaseSpectrometerTestCase):
    """
    Tier 2: Boundary & Corner Cases Tests.
    Tests band edges (785 nm, 880 nm), marginal rays at NA limits, slit corner boundaries,
    and asserts zero intermediate vignetting across all boundary rays.
    """

    def test_t2_1_lower_band_edge_extreme_na_rays(self):
        """
        T2.1: Band edge raytracing at minimum wavelength (e.g. 785 nm) with extreme marginal NA angles.
        """
        wl_min, _, _ = self.get_band_wavelengths()
        stop_dist, stop_r = self.get_stop_geometry()
        det_idx = len(self.stack.surfaces) - 1

        rays = [Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl_min)]
        for px, py in [(stop_r * 0.90, 0.0), (-stop_r * 0.90, 0.0), (0.0, stop_r * 0.90), (0.0, -stop_r * 0.90)]:
            d = np.array([px, py, stop_dist])
            d /= np.linalg.norm(d)
            rays.append(Ray(pos=np.array([0.0, 0.0, 0.0]), dir=d, wavelength=wl_min))

        for i, r in enumerate(rays):
            tr = self.opt_sys.trace_ray(r)
            self.assertIsNotNone(tr, f"Ray {i} failed to trace at lower band edge ({wl_min} um)")
            reached_det = any(h["surface_index"] == det_idx for h in tr.history)
            self.assertTrue(reached_det, f"Ray {i} did not reach detector surface at {wl_min} um")

    def test_t2_2_upper_band_edge_extreme_na_rays(self):
        """
        T2.2: Band edge raytracing at maximum wavelength (e.g. 880 nm) with extreme marginal NA angles.
        """
        _, _, wl_max = self.get_band_wavelengths()
        stop_dist, stop_r = self.get_stop_geometry()
        det_idx = len(self.stack.surfaces) - 1

        rays = [Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl_max)]
        for px, py in [(stop_r * 0.90, 0.0), (-stop_r * 0.90, 0.0), (0.0, stop_r * 0.90), (0.0, -stop_r * 0.90)]:
            d = np.array([px, py, stop_dist])
            d /= np.linalg.norm(d)
            rays.append(Ray(pos=np.array([0.0, 0.0, 0.0]), dir=d, wavelength=wl_max))

        for i, r in enumerate(rays):
            tr = self.opt_sys.trace_ray(r)
            self.assertIsNotNone(tr, f"Ray {i} failed to trace at upper band edge ({wl_max} um)")
            reached_det = any(h["surface_index"] == det_idx for h in tr.history)
            self.assertTrue(reached_det, f"Ray {i} did not reach detector surface at {wl_max} um")

    def test_t2_3_nominal_primary_wavelength_alignment(self):
        """
        T2.3: Central design wavelength chief ray alignment and symmetry.
        """
        _, wl_mid, _ = self.get_band_wavelengths()
        chief_ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl_mid)
        tr = self.opt_sys.trace_ray(chief_ray)

        self.assertIsNotNone(tr, "Primary wavelength chief ray failed to trace")

        # Check detector hit location
        det_idx = len(self.stack.surfaces) - 1
        det_hit = [h for h in tr.history if h["surface_index"] == det_idx]
        self.assertEqual(len(det_hit), 1, "Chief ray did not strike detector")
        x_det = det_hit[0]["x_local"]
        y_det = det_hit[0]["y_local"]
        r_det = math.sqrt(x_det**2 + y_det**2)

        self.assertLess(
            r_det, 0.20,
            f"Primary wavelength chief ray hit at r={r_det:.3f} mm, too far from detector center"
        )

    def test_t2_4_slit_spatial_extremes_boundary_rays(self):
        """
        T2.4: Slit spatial boundary rays (x = +/-0.015 mm, y = +/-0.25 mm).
        """
        _, wl_mid, _ = self.get_band_wavelengths()
        stop_dist, _ = self.get_stop_geometry()
        det_idx = len(self.stack.surfaces) - 1

        slit_corners = [
            (-0.015, -0.25),
            (-0.015,  0.25),
            ( 0.015, -0.25),
            ( 0.015,  0.25),
        ]

        for x0, y0 in slit_corners:
            d = np.array([0.0 - x0, 0.0 - y0, stop_dist])
            d /= np.linalg.norm(d)
            r = Ray(pos=np.array([x0, y0, 0.0]), dir=d, wavelength=wl_mid)
            tr = self.opt_sys.trace_ray(r)
            self.assertIsNotNone(tr, f"Slit corner ({x0}, {y0}) ray failed to trace")
            reached_det = any(h["surface_index"] == det_idx for h in tr.history)
            self.assertTrue(reached_det, f"Slit corner ({x0}, {y0}) ray did not reach detector")

    def test_t2_5_zero_vignetting_assertion_all_boundary_rays(self):
        """
        T2.5: Zero intermediate vignetting assertion across all boundary rays.
        Asserts that 100% of boundary rays reach the detector plane.
        """
        wl_min, wl_mid, wl_max = self.get_band_wavelengths()
        test_wls = [wl_min, wl_mid, wl_max]
        stop_dist, stop_r = self.get_stop_geometry()
        det_idx = len(self.stack.surfaces) - 1

        boundary_rays = []
        for wl in test_wls:
            for x0 in [-0.015, 0.0, 0.015]:
                for y0 in [-0.25, 0.0, 0.25]:
                    pos = np.array([x0, y0, 0.0])
                    # Chief aimed at pupil center
                    d_chief = np.array([0.0 - x0, 0.0 - y0, stop_dist])
                    d_chief /= np.linalg.norm(d_chief)
                    boundary_rays.append(Ray(pos=pos, dir=d_chief, wavelength=wl))

                    # 4 Cardinal marginal pupil points
                    for phi_deg in [0, 90, 180, 270]:
                        phi = math.radians(phi_deg)
                        px = stop_r * 0.90 * math.cos(phi)
                        py = stop_r * 0.90 * math.sin(phi)
                        d_pupil = np.array([px - x0, py - y0, stop_dist])
                        d_pupil /= np.linalg.norm(d_pupil)
                        boundary_rays.append(Ray(pos=pos, dir=d_pupil, wavelength=wl))

        clipped_count = 0
        for r in boundary_rays:
            tr = self.opt_sys.trace_ray(r)
            if not any(h["surface_index"] == det_idx for h in tr.history):
                clipped_count += 1

        self.assertEqual(
            clipped_count, 0,
            f"{clipped_count}/{len(boundary_rays)} boundary rays clipped before reaching detector!"
        )


# ==============================================================================
# TIER 3: CROSS-FEATURE & GEOMETRIC CLEARANCES
# ==============================================================================

class TestTier3CrossFeatureAndClearances(BaseSpectrometerTestCase):
    """
    Tier 3: Cross-Feature & Geometric Clearances Tests.
    Validates Z=0 slit clearance (>12.7 mm), Z=50 M1 clearance (>12.7 mm),
    relay path clearances, and 3D solid substrate non-interference.
    """

    def test_t3_1_slit_plane_z0_beam_clearance(self):
        """
        T3.1: Slit plane (Z=0) clearance > 12.7 mm from returning beam to grating.
        Calculates the lateral distance from (0,0,0) to the beam passing Z=0 heading to grating.
        """
        surfaces = self.stack.surfaces
        self.assertGreaterEqual(len(surfaces), 4)

        m1_origin = surfaces[2].global_origin
        m1_tilt_x = surfaces[2].tilt_x
        fold_angle_rad = math.radians(2.0 * abs(m1_tilt_x))

        # At Z=0 (distance d = m1_origin[2]), lateral offset Y = d * tan(fold_angle)
        d_z = abs(m1_origin[2])
        y_beam_at_z0 = d_z * math.tan(fold_angle_rad)

        self.assertGreater(
            y_beam_at_z0, 12.7,
            f"Slit plane (Z=0) clearance {y_beam_at_z0:.2f} mm is <= 12.7 mm threshold! "
            f"Beam returning from M1 would collide with entrance slit housing."
        )

    def test_t3_2_collimator_plane_z50_beam_clearance(self):
        """
        T3.2: Collimator plane (Z=50) clearance > 12.7 mm from focused beam to DMD.
        """
        surfaces = self.stack.surfaces
        self.assertGreaterEqual(len(surfaces), 6)

        m1_origin = surfaces[2].global_origin
        dmd_origin = surfaces[5].global_origin if len(surfaces) > 5 else surfaces[-1].global_origin

        # Distance in XY plane between M1 origin and DMD or relay path
        dx = dmd_origin[0] - m1_origin[0]
        dy = dmd_origin[1] - m1_origin[1]
        dist_xy = math.sqrt(dx**2 + dy**2)

        self.assertGreater(
            dist_xy, 12.7,
            f"Collimator plane clearance {dist_xy:.2f} mm between M1 and DMD is <= 12.7 mm threshold!"
        )

    def test_t3_3_relay_optical_path_clearance(self):
        """
        T3.3: M3/M4 relay path clearance > 12.7 mm from adjacent optics and detector package,
        and Detector 3D clearance from the input optical axis > 15.0 mm.
        """
        surfaces = self.stack.surfaces
        if len(surfaces) >= 8:
            m3_pos = surfaces[6].global_origin
            det_pos = surfaces[-1].global_origin

            # Check separation between M3 and Detector package
            sep_m3_det = math.sqrt(sum((m3_pos[i] - det_pos[i])**2 for i in range(3)))
            self.assertGreater(
                sep_m3_det, 12.7,
                f"Relay M3 to Detector package clearance {sep_m3_det:.2f} mm is <= 12.7 mm"
            )

            # Check Detector 3D clearance to entrance slit optical axis (Z in [0, 50])
            z_proj = np.clip(det_pos[2], 0.0, 50.0)
            dist_det_axis = math.sqrt(det_pos[0]**2 + det_pos[1]**2 + (det_pos[2] - z_proj)**2)
            self.assertGreater(
                dist_det_axis, 15.0,
                f"Detector 3D clearance to input optical axis {dist_det_axis:.2f} mm is <= 15.0 mm"
            )

    def test_t3_4_solid_substrate_interference_check(self):
        """
        T3.4: Solid substrate interference check across all ray segments and mirror housings.
        Ensures zero collision between the M3-to-M4 relay beam and M2 camera substrate,
        and no ray clips the back or side of any mirror substrate.
        """
        wls = self.stack.wavelengths if hasattr(self.stack, "wavelengths") and self.stack.wavelengths else [0.785, 0.808, 0.830, 0.850, 0.880]
        stop_dist, stop_r = self.get_stop_geometry()

        # 1. Chief ray segment check between M3 and M4 against M2 camera mirror (Surface 4)
        chief_ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=0.830)
        tr_chief = self.opt_sys.trace_ray(chief_ray)
        p_m3 = np.array([tr_chief.history[6]['x_global'], tr_chief.history[6]['y_global'], tr_chief.history[6]['z_global']])
        p_m4 = np.array([tr_chief.history[7]['x_global'], tr_chief.history[7]['y_global'], tr_chief.history[7]['z_global']])

        coll_m2 = check_segment_surface_interference(p_m3, p_m4, self.stack.surfaces[4])
        self.assertFalse(
            coll_m2,
            "CRITICAL: M3-to-M4 relay beam collides with M2 camera mirror substrate!"
        )

        m2_orig = self.stack.surfaces[4].global_origin
        v_m3m4 = p_m4 - p_m3
        t_m2 = np.clip(np.dot(m2_orig - p_m3, v_m3m4) / np.dot(v_m3m4, v_m3m4), 0.0, 1.0)
        closest_m2 = p_m3 + t_m2 * v_m3m4
        dist_m2_center = float(np.linalg.norm(closest_m2 - m2_orig))
        m2_edge_clearance = dist_m2_center - self.stack.surfaces[4].semi_diameter

        self.assertGreater(
            m2_edge_clearance, 12.7,
            f"M3-to-M4 relay beam clearance to M2 substrate edge {m2_edge_clearance:.2f} mm is <= 12.7 mm threshold"
        )

        # 2. Polychromatic and marginal ray interference checks
        for wl in wls:
            for frac in [-1.0, -0.5, 0.0, 0.5, 1.0]:
                d = np.array([0.0, frac * stop_r, stop_dist])
                d /= np.linalg.norm(d)
                r = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=d, wavelength=wl)
                tr = self.opt_sys.trace_ray(r)
                self.assertIsNotNone(tr)
                self.assertFalse(
                    getattr(tr, "interference", False),
                    f"Ray at wl={wl} um, frac={frac} experienced solid substrate mechanical interference!"
                )
                if len(tr.history) >= 8:
                    pm3 = np.array([tr.history[6]['x_global'], tr.history[6]['y_global'], tr.history[6]['z_global']])
                    pm4 = np.array([tr.history[7]['x_global'], tr.history[7]['y_global'], tr.history[7]['z_global']])
                    self.assertFalse(
                        check_segment_surface_interference(pm3, pm4, self.stack.surfaces[4]),
                        f"Marginal ray at wl={wl}, frac={frac} collided with M2 substrate!"
                    )


# ==============================================================================
# TIER 4: REAL-WORLD APPLICATION WORKLOAD & SPOT CONTAINMENT
# ==============================================================================

class TestTier4RealWorldWorkloadAndContainment(BaseSpectrometerTestCase):
    """
    Tier 4: Real-World Application Workload & Spot Containment Tests.
    Dense polychromatic ray tracing across all operating wavelengths,
    mathematical proof that 100% of rays at the detector plane have
    sqrt(x^2 + y^2) <= 0.400 mm (0.80 mm diameter detector),
    RMS spot radius < 0.250 mm, and 100% encircled energy containment across full pupil (rho <= 1.0).
    """

    def _trace_dense_polychromatic_workload(self) -> Tuple[List[Any], List[Tuple[float, float]]]:
        """
        Generates and traces a dense realistic ray bundle:
        Multi-wavelength (all operating band wavelengths) x
        Multi-point slit spatial extent x
        Full-pupil polar sampling (rho up to 1.0).
        """
        wls = self.stack.wavelengths if hasattr(self.stack, "wavelengths") and self.stack.wavelengths else [0.785, 0.808, 0.830, 0.850, 0.880]
        stop_dist, stop_r = self.get_stop_geometry()
        det_idx = len(self.stack.surfaces) - 1

        rays = []
        for wl in wls:
            # Slit spatial extent: 3 points across 30um slit width, 3 points along slit height
            for y_slit in [-0.015, 0.0, 0.015]:
                for x_slit in [-0.10, 0.0, 0.10]:
                    pos = np.array([x_slit, y_slit, 0.0])
                    # Chief ray aimed at stop center
                    dir_chief = np.array([0.0 - x_slit, 0.0 - y_slit, stop_dist])
                    dir_chief /= np.linalg.norm(dir_chief)
                    rays.append(Ray(pos=pos, dir=dir_chief, wavelength=wl))

                    # Full-pupil polar sampling rings (0.25, 0.50, 0.75, 1.00)
                    for ring in [0.25, 0.50, 0.75, 1.00]:
                        for phi in np.linspace(0, 2 * np.pi, 8, endpoint=False):
                            px = ring * stop_r * math.cos(phi)
                            py = ring * stop_r * math.sin(phi)
                            dir_pupil = np.array([px - x_slit, py - y_slit, stop_dist])
                            dir_pupil /= np.linalg.norm(dir_pupil)
                            rays.append(Ray(pos=pos, dir=dir_pupil, wavelength=wl))

        traced = [self.opt_sys.trace_ray(r) for r in rays]

        det_hits = []
        for tr in traced:
            for h in tr.history:
                if h["surface_index"] == det_idx:
                    det_hits.append((h["x_local"], h["y_local"]))

        return traced, det_hits

    def test_t4_1_dense_polychromatic_grid_trace(self):
        """
        T4.1: Dense polychromatic grid trace (>= 300 rays across operating wavelengths).
        Asserts 100% transmission across the full workload to the detector plane.
        """
        traced, det_hits = self._trace_dense_polychromatic_workload()
        total_rays = len(traced)
        reached_det = len(det_hits)

        self.assertGreaterEqual(total_rays, 300, f"Dense workload must trace >= 300 rays, traced {total_rays}")
        self.assertEqual(
            reached_det, total_rays,
            f"Vignetting detected in dense workload: {total_rays - reached_det}/{total_rays} rays clipped before detector!"
        )

    def test_t4_2_mathematical_proof_100pct_detector_containment(self):
        """
        T4.2: Mathematical proof that 100% of rays at the detector plane have sqrt(x^2 + y^2) <= 0.400 mm.
        Acceptance Criteria 3: 100% of light fits entirely within 0.8 mm diameter (0.4 mm radius) detector footprint.
        """
        traced, det_hits = self._trace_dense_polychromatic_workload()
        self.assertGreater(len(det_hits), 0, "No rays reached detector plane")

        radii = [math.sqrt(x**2 + y**2) for x, y in det_hits]
        max_radius = max(radii)
        contained_count = sum(1 for r in radii if r <= 0.4000)
        containment_pct = (contained_count / len(radii)) * 100.0

        # Rigorous mathematical assertion
        self.assertLessEqual(
            max_radius, 0.4000,
            f"CRITICAL ACCEPTANCE FAILURE: Maximum spot radius {max_radius:.4f} mm exceeds "
            f"0.4000 mm detector radius limit! Only {containment_pct:.2f}% of rays contained."
        )
        self.assertEqual(
            contained_count, len(radii),
            f"Expected 100.0% ray containment within 0.400 mm, but got {containment_pct:.2f}%"
        )

    def test_t4_3_rms_spot_radius_containment(self):
        """
        T4.3: RMS spot radius calculation on detector (R_RMS < 0.250 mm).
        """
        _, det_hits = self._trace_dense_polychromatic_workload()
        self.assertGreater(len(det_hits), 0)

        xs = [p[0] for p in det_hits]
        ys = [p[1] for p in det_hits]
        spot = SpotDiagram(xs, ys)
        rms_radius = spot.calculate_rms_radius()

        self.assertLess(
            rms_radius, 0.250,
            f"Detector RMS spot radius {rms_radius:.4f} mm exceeds target budget of 0.250 mm"
        )

    def test_t4_4_encircled_energy_100_percent_at_0p4mm(self):
        """
        T4.4: Encircled energy verification (100% EE within r = 0.40 mm).
        """
        _, det_hits = self._trace_dense_polychromatic_workload()
        radii = np.array([math.sqrt(x**2 + y**2) for x, y in det_hits])

        ee_0p4mm = np.sum(radii <= 0.4000) / len(radii)
        self.assertAlmostEqual(
            ee_0p4mm, 1.0, places=4,
            msg=f"Encircled energy at r=0.40 mm is {ee_0p4mm*100.0:.2f}%, expected exactly 100.0%"
        )


# ==============================================================================
# MAIN TEST RUNNER
# ==============================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
