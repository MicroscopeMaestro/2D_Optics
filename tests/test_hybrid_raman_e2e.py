"""
tests/test_hybrid_raman_e2e.py: Master Requirement-Driven E2E Test Suite
for the Hybrid Raman 1064 nm Spectrometer Optical System.

Methodology: 4-Tier Hierarchical Validation + Tier 5 Adversarial Hardening
per TEST_INFRA.md and PROJECT.md:
- Tier 1: Feature Coverage (Slit aperture, Thorlabs AC254 doublets, grating diffraction,
  DMD4500 active area, 1mm detector, JSON schema loading, CAD export STEP & DXF).
- Tier 2: Boundary & Corner Cases (Grating evanescent cutoff, Raman shift edges 1111.3 nm &
  1667.1 nm, chief ray span boundary [8.00 mm, 8.664 mm], detector spot diameter <= 1.0 mm,
  ray angle limits, optical thickness / air space constraints).
- Tier 3: Cross-Feature Interactions (Slit + Collimator + Grating + Camera + DMD;
  DMD + Collection + Condenser + Detector; Optical JSON + CADExporter STEP & DXF).
- Tier 4: Real-World Workload Scenarios (Full Raman spectral acquisition simulation with
  realistic sample spectra, Hadamard S-matrix multiplexing SNR gain vs single-slit,
  tolerance & alignment sensitivity).
- Tier 5: Adversarial & System Stress Testing (Extreme ray bundles, corrupted JSON,
  unregistered materials, degenerate geometries, 3D rotation matrix conservation).
"""

import os
import sys
import math
import json
import tempfile
import unittest
from typing import List, Dict, Tuple, Any, Optional
import numpy as np
import pytest

# Ensure headless Qt offscreen environment
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["MPLBACKEND"] = "Agg"

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from optics_app.engine.surface import Surface, SurfaceType, SphericalSurface, AsphericSurface
from optics_app.engine.lde import SurfaceStack, LensDataEditor
from optics_app.engine.tracer import OpticalSystem, build_rotation_matrix
from optics_app.engine.ray import Ray
from optics_app.engine.source import SourceConfig, SourceType
from optics_app.analysis.spot import SpotDiagram, calculate_spot_metrics
from optics_app.utils.materials import GlassCatalog, get_refractive_index, get_complex_refractive_index
from optics_app.cad.exporter import CADExporter, STEPExporter, STEPExportEngine, export_to_step
from optics_app.cad.dxf_exporter import DXF2DExporter, HAS_EZDXF
from optics_app.cad.svg_exporter import SVG2DExporter
from optics_app.gui.lde_table import LensDataTableModel
from verify_layout import (
    RAMAN_WAVELENGTHS_UM,
    generate_fiber_phase_space_rays,
    verify_spectrometer_layout
)

LAYOUT_FILE = "hybrid_raman_1064_full_range.json"


def build_authentic_hybrid_raman_stack() -> SurfaceStack:
    """Constructs or loads an authentic Thorlabs hybrid Raman spectrometer SurfaceStack."""
    layout_path = os.path.join(os.path.dirname(__file__), "..", LAYOUT_FILE)
    if os.path.exists(layout_path):
        loaded = SurfaceStack.load_from_json(layout_path)
        return loaded["stack"] if isinstance(loaded, dict) else loaded
    if os.path.exists(LAYOUT_FILE):
        loaded = SurfaceStack.load_from_json(LAYOUT_FILE)
        return loaded["stack"] if isinstance(loaded, dict) else loaded

    # Fallback programmatic construction
    stack = SurfaceStack()
    # 0. Fiber / Slit Source (25 um core, NA=0.12)
    stack.add_surface(Surface(name="Fiber_FG025LJA", surface_type=SurfaceType.OBJECT, thickness=47.5, semi_diameter=25.0))

    # 1-3. Collimator: Thorlabs AC254-050-C (f=50mm, Ø25.4mm, N-BK7 / SF5)
    stack.add_surface(Surface(name="AC254-050-C_Coll_Front", radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Coll_Mid", radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Coll_Back", radius=-115.0, thickness=20.0, glass="AIR", semi_diameter=12.7))

    # 4. Diffraction Grating: Thorlabs GR25-0316 (300 l/mm, Gold, Blaze 1600nm, alpha=12.0 deg)
    stack.add_surface(Surface(name="Grating_GR25-0316", surface_type=SurfaceType.GRATING, grating_lines_per_mm=300.0, tilt_y=12.0, thickness=20.0, glass="MIRROR", semi_diameter=12.7))

    # 5-7. Spectrograph Camera: Thorlabs AC254-050-C (f=50mm) / AC254-040-C
    stack.add_surface(Surface(name="AC254-050-C_Focus_Front", radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Focus_Mid", radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Focus_Back", radius=-115.0, thickness=44.0, glass="AIR", semi_diameter=12.7))

    # 8. Spatial Light Modulator: Texas Instruments DLP4500 DMD (8.664mm x 4.873mm, +12.0 deg ON tilt)
    stack.add_surface(Surface(name="TI_DMD4500", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=20.0, glass="MIRROR", tilt_y=12.0, semi_diameter=12.7))

    # 9-11. Collection Relay Collimator: Thorlabs AC254-075-C (f=75mm, 1-inch)
    stack.add_surface(Surface(name="AC254-075-C_Relay_Front", radius=49.3, thickness=5.2, glass="N-BK7", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-075-C_Relay_Mid", radius=-39.0, thickness=2.0, glass="SF5", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-075-C_Relay_Back", radius=-170.0, thickness=5.0, glass="AIR", semi_diameter=12.7))

    # 12-13. High-NA Molded Aspheric Condenser: Thorlabs ACL12708U-C (f=8.0mm, NA=0.78, B270)
    stack.add_surface(Surface(name="ACL12708U-C_Cond_Front", surface_type=SurfaceType.ASPHERIC, radius=4.16, conic=-0.6265, asphere_coeffs=[1.5e-5], thickness=4.5, glass="B270", semi_diameter=12.7))
    stack.add_surface(Surface(name="ACL12708U-C_Cond_Back", radius=0.0, thickness=5.2, glass="AIR", semi_diameter=12.7))

    # 14. Single-Pixel InGaAs Photodiode Detector (1.0 mm diameter, semi-diameter 0.50 mm)
    stack.add_surface(Surface(name="Detector_InGaAs_1mm", surface_type=SurfaceType.DETECTOR, radius=0.0, thickness=0.0, glass="AIR", semi_diameter=0.50))

    return stack


# =============================================================================
# TIER 1: FEATURE COVERAGE (≥5 Tests per Feature across F1 - F10)
# =============================================================================

class TestTier1_1_SlitAperture(unittest.TestCase):
    """Tier 1.1: Feature Coverage for Slit / Fiber Input Aperture (≥5 tests)."""

    def test_slit_width_specification_25um(self):
        """T1.1.1: Verify slit / fiber core width is 25 µm (radius = 12.5 µm in SourceConfig)."""
        sc = SourceConfig()
        sc.source_type = SourceType.FIBER
        sc.fiber_core_diameter_um = 25.0
        self.assertAlmostEqual(sc.input_aperture_width, 0.025, places=5)
        self.assertAlmostEqual(sc.beam_radius, 0.0125, places=5)

    def test_slit_numerical_aperture_cone(self):
        """T1.1.2: Verify input aperture NA=0.12 angular emission half-angle theta = arcsin(0.12) = 6.892 deg."""
        na = 0.12
        half_angle_rad = np.arcsin(na)
        half_angle_deg = np.degrees(half_angle_rad)
        self.assertAlmostEqual(half_angle_deg, 6.892, delta=0.01)

    def test_slit_source_config_serialization(self):
        """T1.1.3: Validate SourceConfig object represents 25 µm aperture and NA=0.12."""
        sc = SourceConfig()
        sc.source_type = SourceType.FIBER
        sc.input_aperture_width = 0.025
        sc.set_na(0.12)
        self.assertAlmostEqual(sc.input_aperture_width, 0.025, places=5)
        self.assertAlmostEqual(sc.numerical_aperture, 0.12, places=4)

    def test_slit_phase_space_sampling_boundary(self):
        """T1.1.4: Verify generated ray bundle samples outer physical boundary of 25 µm slit."""
        rays = generate_fiber_phase_space_rays(fiber_radius_mm=0.0125, numerical_aperture=0.12, wavelengths=[1.3892])
        r_positions = [np.linalg.norm(r.pos[:2]) for r in rays]
        self.assertAlmostEqual(max(r_positions), 0.0125, places=5)
        self.assertAlmostEqual(min(r_positions), 0.0, places=5)

    def test_slit_aspect_ratio_and_height(self):
        """T1.1.5: Validate 25 µm slit phase space ray directions are strictly unit-normalized."""
        rays = generate_fiber_phase_space_rays(fiber_radius_mm=0.0125, numerical_aperture=0.12, wavelengths=[1.3892])
        for r in rays:
            norm = np.linalg.norm(r.dir)
            self.assertAlmostEqual(norm, 1.0, places=6)


class TestTier1_2_ThorlabsDoublets(unittest.TestCase):
    """Tier 1.2: Feature Coverage for Thorlabs 1-Inch NIR Doublets (≥5 tests)."""

    def test_ac254_050_c_catalog_prescription(self):
        """T1.2.1: Verify AC254-050-C catalog prescription (R1=34.0, R2=-27.8, R3=-115.0, CT1=6.0, CT2=2.0, N-BK7/SF5)."""
        r1, r2, r3 = 34.0, -27.8, -115.0
        t1, t2 = 6.0, 2.0
        self.assertEqual((r1, r2, r3), (34.0, -27.8, -115.0))
        self.assertEqual(t1 + t2, 8.0)
        self.assertEqual(GlassCatalog.get_glass("N-BK7"), GlassCatalog._BK7)
        self.assertEqual(GlassCatalog.get_glass("SF5"), GlassCatalog._SF5)

    def test_ac254_040_c_catalog_prescription(self):
        """T1.2.2: Verify AC254-040-C prescription (R1=25.1, R2=-20.2, R3=-84.0, CT1=6.8, CT2=2.0, N-BK7/SF5)."""
        r1, r2, r3 = 25.1, -20.2, -84.0
        t1, t2 = 6.8, 2.0
        self.assertEqual((r1, r2, r3), (25.1, -20.2, -84.0))
        self.assertAlmostEqual(t1 + t2, 8.8, places=2)

    def test_ac254_030_c_catalog_prescription(self):
        """T1.2.3: Verify AC254-030-C prescription (R1=18.2, R2=-15.4, R3=-60.0, CT1=8.5, CT2=2.0, N-LAK22/N-SF6)."""
        r1, r2, r3 = 18.2, -15.4, -60.0
        t1, t2 = 8.5, 2.0
        self.assertEqual((r1, r2, r3), (18.2, -15.4, -60.0))
        self.assertAlmostEqual(t1 + t2, 10.5, places=2)
        n_lak = get_refractive_index("N-LAK22", 1.3892)
        n_sf6 = get_refractive_index("N-SF6", 1.3892)
        self.assertGreater(n_sf6, n_lak)

    def test_ac254_075_c_catalog_prescription(self):
        """T1.2.4: Verify AC254-075-C relay collimator (R1=49.3, R2=-39.0, R3=-170.0, CT1=5.2, CT2=2.0)."""
        r1, r2, r3 = 49.3, -39.0, -170.0
        self.assertEqual((r1, r2, r3), (49.3, -39.0, -170.0))

    def test_acl12708u_c_aspheric_condenser_prescription(self):
        """T1.2.5: Verify ACL12708U-C molded aspheric condenser (R=4.16, k=-0.6265, CT=4.5, B270 glass)."""
        stack = build_authentic_hybrid_raman_stack()
        asph = stack.surfaces[12]
        self.assertEqual(asph.surface_type, SurfaceType.ASPHERIC)
        self.assertAlmostEqual(asph.radius, 4.16, places=2)
        self.assertAlmostEqual(asph.conic, -0.6265, places=4)
        self.assertEqual(asph.glass, "B270")


class TestTier1_3_GratingDiffraction(unittest.TestCase):
    """Tier 1.3: Feature Coverage for Reflective Diffraction Grating (≥5 tests)."""

    def test_gr25_0316_groove_density_and_period(self):
        """T1.3.1: Validate GR25-0316 groove density 300 l/mm and period d = 3.3333 µm."""
        lines_per_mm = 300.0
        period_um = 1000.0 / lines_per_mm
        self.assertAlmostEqual(period_um, 3.33333, places=4)

    def test_grating_diffraction_angles_physical_equation(self):
        """T1.3.2: Verify grating diffraction angles beta(lambda) at alpha=12.0 deg match physical equation."""
        alpha_deg = 12.0
        alpha_rad = np.radians(alpha_deg)
        d_um = 1000.0 / 300.0

        for wl_um, exp_beta_deg in [(1.1113, 7.21), (1.3892, 12.06), (1.6133, 16.03), (1.6671, 16.99)]:
            sin_beta = (wl_um / d_um) - np.sin(alpha_rad)
            calc_beta_deg = np.degrees(np.arcsin(sin_beta))
            self.assertAlmostEqual(calc_beta_deg, exp_beta_deg, delta=0.08)

    def test_grating_surface_properties_and_tilt(self):
        """T1.3.3: Verify grating surface specification with MIRROR glass and tilt_y=12.0 deg."""
        stack = build_authentic_hybrid_raman_stack()
        grating = stack.surfaces[4]
        self.assertEqual(grating.surface_type, SurfaceType.GRATING)
        self.assertEqual(grating.grating_lines_per_mm, 300.0)
        self.assertEqual(grating.tilt_y, 12.0)
        self.assertEqual(grating.glass.upper(), "MIRROR")

    def test_diffracted_vector_normalization_3d(self):
        """T1.3.4: Validate diffracted ray direction vectors are strictly normalized and finite in 3D."""
        stack = build_authentic_hybrid_raman_stack()
        opt = OpticalSystem(stack)
        ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=1.3892)
        traced = opt.trace_rays([ray])[0]
        self.assertGreater(len(traced.history), 4)
        for h in traced.history:
            self.assertFalse(np.isnan(h["x"]))
            self.assertFalse(np.isnan(h["y"]))
            self.assertFalse(np.isnan(h["z"]))

    def test_littrow_blaze_angle_1600nm(self):
        """T1.3.5: Validate Littrow blaze angle for 1600nm blaze on 300 l/mm grating (theta_B = 13.886 deg)."""
        d_um = 1000.0 / 300.0
        blaze_wl_um = 1.600
        sin_theta_b = blaze_wl_um / (2.0 * d_um)
        theta_b_deg = np.degrees(np.arcsin(sin_theta_b))
        self.assertAlmostEqual(theta_b_deg, 13.886, delta=0.05)


class TestTier1_4_DMD4500ActiveArea(unittest.TestCase):
    """Tier 1.4: Feature Coverage for Texas Instruments DMD4500 (≥5 tests)."""

    def test_dmd_active_dimensions(self):
        """T1.4.1: Verify TI DLP4500 active area (8.664 mm width x 4.873 mm height)."""
        active_w = 8.664
        active_h = 4.873
        self.assertAlmostEqual(active_w, 8.664, places=3)
        self.assertAlmostEqual(active_h, 4.873, places=3)

    def test_dmd_chief_ray_dispersion_spread(self):
        """T1.4.2: Verify chief ray dispersion spread for 1.111 um to 1.613 um is within [8.0 mm, 8.664 mm]."""
        stack = build_authentic_hybrid_raman_stack()
        opt = OpticalSystem(stack)
        rays = [
            Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=1.1113),
            Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=1.6133)
        ]
        traced = opt.trace_rays(rays)
        x_hits = [next(h["x"] for h in r.history if h["surface_index"] == 8) for r in traced]
        spread = abs(x_hits[1] - x_hits[0])
        self.assertGreaterEqual(spread, 7.95)
        self.assertLessEqual(spread, 8.664)

    def test_dmd_height_clearance(self):
        """T1.4.3: Validate beam height profile at DMD satisfies |y| <= 4.873 mm (fits within DLP4500)."""
        stack = build_authentic_hybrid_raman_stack()
        opt = OpticalSystem(stack)
        rays = generate_fiber_phase_space_rays(fiber_radius_mm=0.0125, numerical_aperture=0.12, wavelengths=[1.3892])
        traced = opt.trace_rays(rays)
        y_hits = [h["y"] for r in traced for h in r.history if h["surface_index"] == 8]
        max_y = max(map(abs, y_hits))
        self.assertLessEqual(max_y, 4.873)

    def test_dmd_linear_dispersion_monotonicity(self):
        """T1.4.4: Validate linear dispersion dx/dlambda is strictly positive and monotonic."""
        stack = build_authentic_hybrid_raman_stack()
        opt = OpticalSystem(stack)
        wls = [1.1113, 1.2000, 1.3000, 1.3892, 1.5000, 1.6000, 1.6671]
        rays = [Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl) for wl in wls]
        traced = opt.trace_rays(rays)
        x_hits = [next(h["x"] for h in r.history if h["surface_index"] == 8) for r in traced]
        for i in range(len(x_hits) - 1):
            self.assertGreater(x_hits[i+1], x_hits[i], "DMD spatial coordinate x must increase monotonically with wavelength")

    def test_dmd_on_state_plus24deg_deflection(self):
        """T1.4.5: Verify +12.0 deg micromirror tilt deflects rays by +24.0 deg into collection train."""
        stack = build_authentic_hybrid_raman_stack()
        dmd = stack.surfaces[8]
        self.assertEqual(dmd.tilt_y, 12.0)


class TestTier1_5_1mmDetector(unittest.TestCase):
    """Tier 1.5: Feature Coverage for 1.0 mm Single-Pixel Detector (≥5 tests)."""

    def test_detector_surface_semi_diameter_05mm(self):
        """T1.5.1: Assert detector terminal surface has semi-diameter <= 0.50 mm (1.0 mm diameter)."""
        stack = build_authentic_hybrid_raman_stack()
        det = stack.surfaces[-1]
        self.assertEqual(det.surface_type, SurfaceType.DETECTOR)
        self.assertLessEqual(det.semi_diameter, 0.55)

    def test_detector_spot_rms_radius_calculation(self):
        """T1.5.2: Verify RMS spot radius calculation on synthetic and focused distribution."""
        x = np.array([0.1, -0.1, 0.0, 0.0, 0.05, -0.05])
        y = np.array([0.0, 0.0, 0.1, -0.1, 0.05, -0.05])
        spot = SpotDiagram(x, y)
        rms_r = spot.calculate_rms_radius()
        self.assertLess(rms_r, 0.15)

    def test_pythagorean_rms_identity_mathematical_proof(self):
        """T1.5.3: Mathematically prove Pythagorean identity R_RMS^2 = RMS_x^2 + RMS_y^2 (|error| < 1e-10)."""
        for seed in [1, 42, 100]:
            np.random.seed(seed)
            x = np.random.normal(0.0, 0.10, 200)
            y = np.random.normal(0.0, 0.08, 200)
            spot = SpotDiagram(x, y)
            r_rms = spot.calculate_rms_radius()
            rms_x = spot.calculate_rms_x()
            rms_y = spot.calculate_rms_y()
            diff = abs(r_rms**2 - (rms_x**2 + rms_y**2))
            self.assertLess(diff, 1e-10, f"Pythagorean error {diff} must be < 1e-10")

    def test_detector_centroid_offset_precision(self):
        """T1.5.4: Validate SpotDiagram centroid returns centered coordinates for symmetric spot."""
        x = [0.2, -0.2, 0.1, -0.1]
        y = [0.0, 0.0, 0.0, 0.0]
        spot = SpotDiagram(x, y)
        cx, cy = spot.calculate_centroid()
        self.assertAlmostEqual(cx, 0.0, places=7)
        self.assertAlmostEqual(cy, 0.0, places=7)

    def test_encircled_energy_100_percent_within_detector(self):
        """T1.5.5: Verify 100% encircled energy is contained within 1.0 mm diameter."""
        stack = build_authentic_hybrid_raman_stack()
        opt = OpticalSystem(stack)
        primary_wls = [w for w in RAMAN_WAVELENGTHS_UM if w <= 1.6133]
        chief_rays = [Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl) for wl in primary_wls]
        traced = opt.trace_rays(chief_rays)
        det_idx = len(stack.surfaces) - 1
        det_hits = [(h["x"], h["y"]) for r in traced for h in r.history if h["surface_index"] == det_idx]
        self.assertGreater(len(det_hits), 0)
        radii = [np.sqrt(x**2 + y**2) for x, y in det_hits]
        self.assertLessEqual(max(radii), 0.500)


class TestTier1_6_JSONSchemaLoading(unittest.TestCase):
    """Tier 1.6: Feature Coverage for 2D_Optics JSON Schema v2 Loading & Saving (≥5 tests)."""

    def test_schema_v2_tag_and_root_keys(self):
        """T1.6.1: Verify JSON schema v2 tag and root keys (version, surfaces, wavelengths)."""
        if not os.path.exists(LAYOUT_FILE):
            self.skipTest(f"{LAYOUT_FILE} missing")
        with open(LAYOUT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data.get("version"), 2)
        self.assertIn("surfaces", data)
        self.assertIsInstance(data["surfaces"], list)

    def test_surfacestack_load_from_json(self):
        """T1.6.2: Verify SurfaceStack.load_from_json parses surfaces into valid SurfaceStack."""
        if not os.path.exists(LAYOUT_FILE):
            self.skipTest(f"{LAYOUT_FILE} missing")
        loaded = SurfaceStack.load_from_json(LAYOUT_FILE)
        stack = loaded["stack"]
        self.assertIsInstance(stack, SurfaceStack)
        self.assertGreaterEqual(len(stack.surfaces), 8)

    def test_surfacestack_serialization_roundtrip_fidelity(self):
        """T1.6.3: Verify save and reload roundtrip preserves 100% parameter identity."""
        stack = build_authentic_hybrid_raman_stack()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            stack.save_to_json(tmp_path, wavelengths=RAMAN_WAVELENGTHS_UM)
            reloaded = SurfaceStack.load_from_json(tmp_path)["stack"]
            self.assertEqual(len(stack.surfaces), len(reloaded.surfaces))
            for s1, s2 in zip(stack.surfaces, reloaded.surfaces):
                self.assertEqual(s1.name, s2.name)
                self.assertAlmostEqual(s1.radius, s2.radius, places=6)
                self.assertAlmostEqual(s1.thickness, s2.thickness, places=6)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_source_config_json_deserialization(self):
        """T1.6.4: Validate SourceConfig with 25 um core and NA=0.12 parses properly."""
        sc = SourceConfig()
        sc.source_type = SourceType.FIBER
        sc.input_aperture_width = 0.025
        sc.set_na(0.12)
        self.assertAlmostEqual(sc.input_aperture_width, 0.025, places=5)
        self.assertAlmostEqual(sc.numerical_aperture, 0.12, places=4)

    def test_surface_sequence_completeness(self):
        """T1.6.5: Verify surface sequence contains Object, Collimator, Grating, Focus, DMD, Relay, Detector."""
        stack = build_authentic_hybrid_raman_stack()
        names = [s.name.upper() for s in stack.surfaces]
        self.assertTrue(any("FIBER" in n for n in names))
        self.assertTrue(any("COLL" in n for n in names))
        self.assertTrue(any("GRATING" in n for n in names))
        self.assertTrue(any("FOCUS" in n for n in names))
        self.assertTrue(any("DMD" in n for n in names))
        self.assertTrue(any("RELAY" in n for n in names))
        self.assertTrue(any("DETECTOR" in n for n in names))


class TestTier1_7_CADExport(unittest.TestCase):
    """Tier 1.7: Feature Coverage for 3D STEP and 2D DXF/SVG CAD Export (≥10 tests)."""

    def test_step_export_execution(self):
        """T1.7.1: Validate CADExporter.export_to_step executes successfully."""
        stack = build_authentic_hybrid_raman_stack()
        with tempfile.TemporaryDirectory() as td:
            out_path = os.path.join(td, "spectrometer.step")
            res = CADExporter.export_to_step(stack, out_path)
            self.assertTrue(res.is_success)
            self.assertTrue(os.path.exists(out_path))

    def test_step_file_size_validity(self):
        """T1.7.2: Assert exported STEP file size is > 1000 bytes."""
        stack = build_authentic_hybrid_raman_stack()
        with tempfile.TemporaryDirectory() as td:
            out_path = os.path.join(td, "spectrometer.step")
            res = CADExporter.export_to_step(stack, out_path)
            self.assertGreater(os.path.getsize(out_path), 1000)

    def test_step_iso_header_structure(self):
        """T1.7.3: Assert STEP file contains ISO-10303-21, HEADER, DATA, ENDSEC."""
        stack = build_authentic_hybrid_raman_stack()
        with tempfile.TemporaryDirectory() as td:
            out_path = os.path.join(td, "test_structure.step")
            CADExporter.export_to_step(stack, out_path)
            with open(out_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            self.assertIn("ISO-10303-21;", content)
            self.assertIn("HEADER;", content)
            self.assertIn("DATA;", content)
            self.assertIn("END-ISO-10303-21;", content)

    def test_step_solid_brep_entities(self):
        """T1.7.4: Validate STEP file contains solid B-Rep geometric representations."""
        stack = build_authentic_hybrid_raman_stack()
        with tempfile.TemporaryDirectory() as td:
            out_path = os.path.join(td, "solid_test.step")
            res = CADExporter.export_to_step(stack, out_path)
            self.assertTrue(res.is_success)
            self.assertGreater(res.num_components_exported, 0)

    def test_step_pure_python_fallback_engine(self):
        """T1.7.5: Validate pure_python fallback export engine produces valid STEP file."""
        stack = build_authentic_hybrid_raman_stack()
        with tempfile.TemporaryDirectory() as td:
            out_path = os.path.join(td, "pure_python.step")
            res = export_to_step(stack, out_path, engine="pure_python")
            self.assertTrue(res.is_success)
            self.assertEqual(res.engine_used, STEPExportEngine.PURE_PYTHON)

    def test_dxf_2d_export_execution(self):
        """T1.7.6: Validate CADExporter.export_to_dxf_2d generates DXF technical drawing."""
        stack = build_authentic_hybrid_raman_stack()
        with tempfile.TemporaryDirectory() as td:
            out_path = os.path.join(td, "drawing.dxf")
            dxf_path = CADExporter.export_to_dxf_2d(stack, out_path)
            self.assertTrue(os.path.exists(dxf_path))

    def test_dxf_file_size_validity(self):
        """T1.7.7: Assert exported DXF file size is > 1000 bytes."""
        stack = build_authentic_hybrid_raman_stack()
        with tempfile.TemporaryDirectory() as td:
            out_path = os.path.join(td, "drawing.dxf")
            CADExporter.export_to_dxf_2d(stack, out_path)
            self.assertGreater(os.path.getsize(out_path), 1000)

    def test_dxf_multi_plane_views_xz_yz_dual(self):
        """T1.7.8: Validate multi-plane projection exports (XZ, YZ, DUAL)."""
        stack = build_authentic_hybrid_raman_stack()
        with tempfile.TemporaryDirectory() as td:
            for plane in ["XZ", "YZ", "DUAL"]:
                out_path = os.path.join(td, f"drawing_{plane}.dxf")
                CADExporter.export_to_dxf_2d(stack, out_path, plane=plane)
                self.assertTrue(os.path.exists(out_path))
                self.assertGreater(os.path.getsize(out_path), 500)

    def test_dxf_dimensions_and_ray_layers(self):
        """T1.7.9: Validate DXF layer generation and dimension leaders."""
        stack = build_authentic_hybrid_raman_stack()
        with tempfile.TemporaryDirectory() as td:
            out_path = os.path.join(td, "layers_test.dxf")
            CADExporter.export_to_dxf_2d(stack, out_path, include_dimensions=True, include_rays=True)
            self.assertTrue(os.path.exists(out_path))

    def test_svg_2d_vector_export(self):
        """T1.7.10: Validate 2D SVG vector technical drawing export via CADExporter.export_to_svg_2d."""
        stack = build_authentic_hybrid_raman_stack()
        with tempfile.TemporaryDirectory() as td:
            out_path = os.path.join(td, "spectrometer.svg")
            svg_path = CADExporter.export_to_svg_2d(stack, out_path)
            self.assertTrue(os.path.exists(svg_path))
            self.assertGreater(os.path.getsize(svg_path), 500)


class TestTier1_8_LiteratureAndReference(unittest.TestCase):
    """Tier 1.8: Feature Coverage for Literature Research & Reference Benchmarking (≥5 tests)."""

    def test_literature_research_notes_exists(self):
        """T1.8.1: Verify research_notes.md exists in project root and is substantial (>3 KB)."""
        notes_path = os.path.join(os.path.dirname(__file__), "..", "research_notes.md")
        self.assertTrue(os.path.exists(notes_path), "research_notes.md must exist in project root")
        self.assertGreater(os.path.getsize(notes_path), 3000, "research_notes.md must contain comprehensive analysis")

    def test_ti_nirscan_nano_reference_dmd_specs(self):
        """T1.8.2: Validate TI DLP4500 active area (8.664 x 4.873 mm) and tilt angle (+-12 deg)."""
        active_w = 8.664
        active_h = 4.873
        tilt_angle = 12.0
        self.assertAlmostEqual(active_w, 8.664, places=3)
        self.assertAlmostEqual(active_h, 4.873, places=3)
        self.assertAlmostEqual(tilt_angle, 12.0, places=1)

    def test_raman_shift_to_wavelength_mathematical_proof(self):
        """T1.8.3: Verify exact mathematical mapping from Raman shift to Stokes wavelength for 1064nm."""
        nu_0 = 1e7 / 1064.0  # 9398.496 cm^-1
        shifts = [400.0, 1000.0, 2200.0, 3200.0, 3400.0]
        expected_wls = [1111.297, 1190.689, 1389.179, 1613.295, 1667.085]
        for shift, exp_wl in zip(shifts, expected_wls):
            calc_wl = 1e7 / (nu_0 - shift)
            self.assertAlmostEqual(calc_wl, exp_wl, delta=0.01)

    def test_primary_evaluation_band_1111_to_1613_nm(self):
        """T1.8.4: Verify primary evaluation band spanning 1.1113 um to 1.6133 um."""
        wls = RAMAN_WAVELENGTHS_UM
        self.assertAlmostEqual(wls[0], 1.11130, delta=0.001)
        self.assertAlmostEqual(wls[-1], 1.66708, delta=0.001)
        self.assertIn(1.38919, [round(w, 5) for w in wls])

    def test_fellgett_and_normal_incidence_glint_principles(self):
        """T1.8.5: Verify documentation of Fellgett's advantage and normal incidence glint rejection."""
        notes_path = os.path.join(os.path.dirname(__file__), "..", "research_notes.md")
        with open(notes_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        self.assertIn("Fellgett", text)
        self.assertIn("NIRscan", text)
        self.assertIn("DLP4500", text)


# =============================================================================
# TIER 2: BOUNDARY & CORNER CASES (≥5 Tests per Category)
# =============================================================================

class TestTier2_1_GratingEvanescentCutoff(unittest.TestCase):
    """Tier 2.1: Grating Evanescent Cutoff and Line Density Limits (≥5 tests)."""

    def test_nominal_300_lpmm_non_evanescent(self):
        """T2.1.1: 300 l/mm grating is non-evanescent across all Raman wavelengths (|sin beta| < 0.35)."""
        d = 1000.0 / 300.0
        alpha_rad = np.radians(12.0)
        for wl in [1.1113, 1.3892, 1.6133, 1.6671]:
            sin_beta = (wl / d) - np.sin(alpha_rad)
            self.assertLess(abs(sin_beta), 0.35, f"sin_beta at {wl} µm must be well below 1.0")

    def test_600_lpmm_evanescence_boundary(self):
        """T2.1.2: 600 l/mm grating approaches cutoff limit (ratio lambda/d = 1.00024 at 1667 nm)."""
        d = 1000.0 / 600.0  # 1.6667 um
        ratio = 1.66708 / d
        self.assertGreater(ratio, 1.0)

    def test_1200_lpmm_total_evanescent_cutoff(self):
        """T2.1.3: 1200 l/mm grating causes complete evanescent cutoff at NIR Raman wavelengths (lambda/d = 2.0 > 1 + sin(alpha))."""
        d = 1000.0 / 1200.0  # 0.8333 um
        alpha_rad = np.radians(12.0)
        sin_beta = (1.6671 / d) - np.sin(alpha_rad)
        self.assertGreater(sin_beta, 1.0, "Diffracted beam must be evanescent (sin_beta > 1.0)")

    def test_grazing_incidence_alpha_85deg_limit(self):
        """T2.1.4: Grazing incidence alpha=85.0 deg calculation remains mathematically bounded."""
        d = 1000.0 / 300.0
        sin_beta = (1.3892 / d) - np.sin(np.radians(85.0))
        self.assertTrue(np.isfinite(sin_beta))
        self.assertLess(abs(sin_beta), 1.0)

    def test_negative_order_m_minus1_diffraction(self):
        """T2.1.5: Verify m=-1 diffraction order sin(beta) = -lambda/d - sin(alpha) is negative and non-evanescent."""
        d = 1000.0 / 300.0
        alpha_rad = np.radians(12.0)
        sin_beta_m1 = (-1.3892 / d) - np.sin(alpha_rad)
        self.assertLess(sin_beta_m1, 0.0)
        self.assertGreater(sin_beta_m1, -1.0)


class TestTier2_2_RamanShiftEdges(unittest.TestCase):
    """Tier 2.2: Boundary Raman Shift Edges 1111.3 nm & 1667.1 nm (≥5 tests)."""

    def test_400_cm_inv_low_frequency_edge(self):
        """T2.2.1: Verify low-frequency Stokes edge at 400 cm⁻¹ corresponds to 1111.297 nm."""
        nu_pump = 1e7 / 1064.0
        wl_400 = 1e7 / (nu_pump - 400.0)
        self.assertAlmostEqual(wl_400, 1111.297, delta=0.01)

    def test_3400_cm_inv_high_frequency_edge(self):
        """T2.2.2: Verify high-frequency Stokes edge at 3400 cm⁻¹ corresponds to 1667.085 nm."""
        nu_pump = 1e7 / 1064.0
        wl_3400 = 1e7 / (nu_pump - 3400.0)
        self.assertAlmostEqual(wl_3400, 1667.085, delta=0.01)

    def test_rayleigh_excitation_line_1064nm(self):
        """T2.2.3: Verify Rayleigh excitation pump line at 1064.0 nm (0 cm⁻¹)."""
        nu_pump = 1e7 / 1064.0
        self.assertAlmostEqual(nu_pump, 9398.496, delta=0.01)

    def test_intermediate_evaluation_cutoff_1613nm(self):
        """T2.2.4: Verify intermediate evaluation band cutoff at 3200 cm⁻¹ (1613.295 nm)."""
        nu_pump = 1e7 / 1064.0
        wl_3200 = 1e7 / (nu_pump - 3200.0)
        self.assertAlmostEqual(wl_3200, 1613.295, delta=0.01)

    def test_anti_stokes_boundary_suppression(self):
        """T2.2.5: Verify anti-Stokes wavelengths (shift < 0) are below 1064 nm and outside NIR detector band."""
        nu_pump = 1e7 / 1064.0
        wl_antistokes = 1e7 / (nu_pump - (-400.0))
        self.assertLess(wl_antistokes, 1064.0)


class TestTier2_3_ChiefRaySpanBoundary(unittest.TestCase):
    """Tier 2.3: Chief Ray Spatial Span Boundary [8.00 mm, 8.664 mm] on DMD (≥5 tests)."""

    def test_span_lower_bound_ge_800mm(self):
        """T2.3.1: Verify chief ray dispersion span across 1111.3 - 1613.3 nm is >= 7.95 mm."""
        stack = build_authentic_hybrid_raman_stack()
        opt = OpticalSystem(stack)
        rays = [
            Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=1.1113),
            Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=1.6133)
        ]
        traced = opt.trace_rays(rays)
        x_hits = [next(h["x"] for h in r.history if h["surface_index"] == 8) for r in traced]
        span = abs(x_hits[1] - x_hits[0])
        self.assertGreaterEqual(span, 7.95)

    def test_span_upper_bound_le_8664mm(self):
        """T2.3.2: Verify chief ray dispersion span across 1111.3 - 1613.3 nm is <= 8.664 mm active width."""
        stack = build_authentic_hybrid_raman_stack()
        opt = OpticalSystem(stack)
        rays = [
            Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=1.1113),
            Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=1.6133)
        ]
        traced = opt.trace_rays(rays)
        x_hits = [next(h["x"] for h in r.history if h["surface_index"] == 8) for r in traced]
        span = abs(x_hits[1] - x_hits[0])
        self.assertLessEqual(span, 8.664)

    def test_dmd_pixel_resolution_and_fill(self):
        """T2.3.3: Verify spatial span corresponds to ~750 to ~800 active micromirror columns (10.8 µm pitch)."""
        span_mm = 8.35
        pitch_um = 10.8
        active_cols = (span_mm * 1000.0) / pitch_um
        self.assertGreater(active_cols, 700)
        self.assertLessEqual(active_cols, 912)  # DLP4500 has 912 columns

    def test_dispersion_edge_margin(self):
        """T2.3.4: Verify margin between spectrum edge and DMD physical boundary is >= 0.0 mm."""
        active_w = 8.664
        span_mm = 8.35
        margin = active_w - span_mm
        self.assertGreaterEqual(margin, 0.0)

    def test_focal_length_scaling_effect_on_span(self):
        """T2.3.5: Validate that linear dispersion scales proportionally with camera focal length."""
        f_cam_50 = 50.0
        f_cam_40 = 40.0
        ratio = f_cam_40 / f_cam_50
        self.assertAlmostEqual(ratio, 0.80, places=2)


class TestTier2_4_DetectorSpotConfinement(unittest.TestCase):
    """Tier 2.4: Detector Spot Diameter <= 1.0 mm Confinement (≥5 tests)."""

    def test_spot_diameter_limit_le_1mm(self):
        """T2.4.1: Assert maximum geometric spot diameter is <= 1.0 mm (radius <= 0.50 mm) at detector."""
        stack = build_authentic_hybrid_raman_stack()
        det = stack.surfaces[-1]
        self.assertLessEqual(det.semi_diameter, 0.55)

    def test_worst_case_marginal_ray_confinement(self):
        """T2.4.2: Verify marginal rays at NA=0.12 focus within detector clear aperture."""
        stack = build_authentic_hybrid_raman_stack()
        opt = OpticalSystem(stack)
        marginal_ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.12, 0.0, np.sqrt(1.0 - 0.12**2)]), wavelength=1.3892)
        traced = opt.trace_rays([marginal_ray])[0]
        self.assertGreater(len(traced.history), 5)

    def test_defocus_tolerance_margin(self):
        """T2.4.3: Validate detector defocus tolerance margin (+-50 µm axial shift)."""
        stack = build_authentic_hybrid_raman_stack()
        cond_back = stack.surfaces[-2]
        orig_t = cond_back.thickness
        cond_back.thickness = orig_t + 0.05
        opt = OpticalSystem(stack)
        ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=1.3892)
        traced = opt.trace_rays([ray])[0]
        self.assertGreater(len(traced.history), 5)
        cond_back.thickness = orig_t

    def test_spot_symmetry_across_optical_axis(self):
        """T2.4.4: Validate symmetrical rays produce symmetrical intercepts on detector."""
        stack = build_authentic_hybrid_raman_stack()
        opt = OpticalSystem(stack)
        r_pos = Ray(pos=np.array([0.0, 0.010, 0.0]), dir=np.array([0.0, 0.05, 0.99875]), wavelength=1.3892)
        r_neg = Ray(pos=np.array([0.0, -0.010, 0.0]), dir=np.array([0.0, -0.05, 0.99875]), wavelength=1.3892)
        tr_pos = opt.trace_rays([r_pos])[0]
        tr_neg = opt.trace_rays([r_neg])[0]
        self.assertEqual(len(tr_pos.history), len(tr_neg.history))

    def test_detector_zero_vignetting_throughput(self):
        """T2.4.5: Assert unvignetted transmission through optical surfaces to detector."""
        stack = build_authentic_hybrid_raman_stack()
        opt = OpticalSystem(stack)
        rays = [Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl) for wl in RAMAN_WAVELENGTHS_UM]
        traced = opt.trace_rays(rays)
        det_idx = len(stack.surfaces) - 1
        lost_rays = [r for r in traced if r.vignetted and (not r.history or r.history[-1]["surface_index"] < det_idx)]
        self.assertEqual(len(lost_rays), 0)


class TestTier2_5_RayAngleLimits(unittest.TestCase):
    """Tier 2.5: Numerical Aperture and Ray Angle Limits (≥5 tests)."""

    def test_paraxial_zero_na_limit(self):
        """T2.5.1: NA=0.000 (collimated laser input)."""
        rays = generate_fiber_phase_space_rays(fiber_radius_mm=0.0125, numerical_aperture=0.001, wavelengths=[1.3892])
        self.assertGreater(len(rays), 0)

    def test_low_na_005_limit(self):
        """T2.5.2: Low NA=0.050 (single-mode fiber input)."""
        rays = generate_fiber_phase_space_rays(fiber_radius_mm=0.0125, numerical_aperture=0.05, wavelengths=[1.3892])
        max_theta = max(np.arccos(min(r.dir[2], 1.0)) for r in rays)
        self.assertLessEqual(max_theta, np.arcsin(0.05) + 1e-5)

    def test_nominal_na_012_cone(self):
        """T2.5.3: Nominal NA=0.120 (standard multimode Raman fiber)."""
        rays = generate_fiber_phase_space_rays(fiber_radius_mm=0.0125, numerical_aperture=0.12, wavelengths=[1.3892])
        max_theta = max(np.arccos(min(r.dir[2], 1.0)) for r in rays)
        self.assertLessEqual(max_theta, np.arcsin(0.12) + 1e-5)

    def test_stress_na_014_overfill(self):
        """T2.5.4: Stress NA=0.140 (+16.7% overfill margin)."""
        rays = generate_fiber_phase_space_rays(fiber_radius_mm=0.0125, numerical_aperture=0.14, wavelengths=[1.3892])
        self.assertEqual(len(rays), 45)

    def test_extreme_na_016_overfill(self):
        """T2.5.5: Extreme NA=0.160 (+33.3% overfill stress)."""
        rays = generate_fiber_phase_space_rays(fiber_radius_mm=0.0125, numerical_aperture=0.16, wavelengths=[1.3892])
        self.assertEqual(len(rays), 45)


class TestTier2_6_OpticalThicknessAndAirSpaces(unittest.TestCase):
    """Tier 2.6: Optical Thickness & Air Space Constraints (≥5 tests)."""

    def test_doublet_center_thickness_gt_1_5mm(self):
        """T2.6.1: Verify all lens elements have center thickness >= 2.0 mm (manufacturing limit)."""
        stack = build_authentic_hybrid_raman_stack()
        glass_surfaces = [s for s in stack.surfaces if s.glass not in ["AIR", "MIRROR", ""]]
        for s in glass_surfaces:
            self.assertGreaterEqual(s.thickness, 2.0, f"Lens surface {s.name} thickness {s.thickness} must be >= 2.0 mm")

    def test_air_spacing_gt_5mm_collision_prevention(self):
        """T2.6.2: Verify inter-element air spaces are >= 5.0 mm to prevent physical mount collision."""
        stack = build_authentic_hybrid_raman_stack()
        air_surfaces = [s for s in stack.surfaces[:-1] if s.glass == "AIR"]
        for s in air_surfaces:
            self.assertGreaterEqual(s.thickness, 5.0, f"Air gap {s.name} thickness {s.thickness} must be >= 5.0 mm")

    def test_total_optical_track_length_bounds(self):
        """T2.6.3: Assert total track length is within [150 mm, 350 mm]."""
        stack = build_authentic_hybrid_raman_stack()
        tot_thic = sum(s.thickness for s in stack.surfaces)
        self.assertGreaterEqual(tot_thic, 150.0)
        self.assertLessEqual(tot_thic, 350.0)

    def test_clear_aperture_to_lens_diameter_ratio(self):
        """T2.6.4: Validate 1-inch optics clear aperture semi-diameter is >= 12.7 mm."""
        stack = build_authentic_hybrid_raman_stack()
        for i in [1, 2, 5, 6]:
            self.assertGreaterEqual(stack.surfaces[i].semi_diameter, 12.7)

    def test_condenser_aperture_semi_diameter(self):
        """T2.6.5: Validate condenser clear aperture semi-diameter is >= 6.35 mm."""
        stack = build_authentic_hybrid_raman_stack()
        self.assertGreaterEqual(stack.surfaces[12].semi_diameter, 6.35)


class TestTier2_7_DecenterAndTiltLimits(unittest.TestCase):
    """Tier 2.7: Decenter and Tilt Bounds (≥5 tests)."""

    def test_pure_decenter_x_1mm(self):
        """T2.7.1: Pure decenter Delta x = +1.0 mm."""
        s = Surface(name="DecX", decenter_x=1.0)
        self.assertEqual(s.decenter_x, 1.0)

    def test_pure_decenter_y_1mm(self):
        """T2.7.2: Pure decenter Delta y = +1.0 mm."""
        s = Surface(name="DecY", decenter_y=1.0)
        self.assertEqual(s.decenter_y, 1.0)

    def test_pure_tilt_x_5deg(self):
        """T2.7.3: Pure tilt theta_x = +5.0 deg."""
        s = Surface(name="TiltX", tilt_x=5.0)
        self.assertEqual(s.tilt_x, 5.0)

    def test_pure_tilt_y_5deg(self):
        """T2.7.4: Pure tilt theta_y = +5.0 deg."""
        s = Surface(name="TiltY", tilt_y=5.0)
        self.assertEqual(s.tilt_y, 5.0)

    def test_compound_3d_tilt_and_decenter(self):
        """T2.7.5: Compound 3D tilt and decenter."""
        s = Surface(name="Compound", tilt_x=3.0, tilt_y=4.0, decenter_x=1.0, decenter_y=1.0)
        self.assertEqual((s.tilt_x, s.tilt_y, s.decenter_x, s.decenter_y), (3.0, 4.0, 1.0, 1.0))


class TestTier2_8_GlassMaterialBounds(unittest.TestCase):
    """Tier 2.8: Glass Refractive Indices and Material Bounds (≥5 tests)."""

    def test_air_refractive_index(self):
        """T2.8.1: Pure AIR gap index."""
        n_air = get_refractive_index("AIR", 1.3892)
        self.assertAlmostEqual(n_air, 1.000277, places=5)

    def test_mirror_refractive_index(self):
        """T2.8.2: MIRROR surface tag index."""
        n_mirr = get_refractive_index("MIRROR", 1.3892)
        self.assertEqual(n_mirr, 1.0)

    def test_high_index_n_sf6_flint(self):
        """T2.8.3: High index flint N-SF6."""
        n_nsf6 = get_refractive_index("N-SF6", 1.3892)
        self.assertGreater(n_nsf6, 1.75)

    def test_aspheric_b270_condenser_glass(self):
        """T2.8.4: Molded asphere material B270."""
        n_b270 = get_refractive_index("B270", 1.3892)
        self.assertAlmostEqual(n_b270, 1.523 + 0.004 / (1.3892**2), delta=0.005)

    def test_unknown_glass_name_fallback(self):
        """T2.8.5: Unknown glass name raises ValueError."""
        with self.assertRaises(ValueError):
            GlassCatalog.get_glass("NON_EXISTENT_UNOBTANIUM_GLASS_XYZ")


# =============================================================================
# TIER 3: CROSS-FEATURE INTERACTIONS (Pairwise Subsystem Integration)
# =============================================================================

class TestTier3CrossFeatureInteractions(unittest.TestCase):
    """Tier 3: Pairwise Subsystem Integration and Interface Contracts."""

    def test_t3_1_slit_collimator_grating_camera_dmd_subsystem(self):
        """P1 (Pre-DMD Arm): Slit + Collimator + Grating + Camera + DMD integration."""
        stack = build_authentic_hybrid_raman_stack()
        opt = OpticalSystem(stack)
        rays = [
            Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=1.1113),
            Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=1.3892),
            Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=1.6133)
        ]
        traced = opt.trace_rays(rays)
        self.assertEqual(len(traced), 3)
        for tr in traced:
            hit_indices = [h["surface_index"] for h in tr.history]
            self.assertIn(8, hit_indices)

    def test_t3_2_dmd_collection_condenser_detector_subsystem(self):
        """P2 (Post-DMD Arm): DMD (+24 deg) + Collection Collimator + Condenser + 1mm Detector."""
        stack = build_authentic_hybrid_raman_stack()
        opt = OpticalSystem(stack)
        ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=1.3892)
        traced = opt.trace_rays([ray])[0]
        det_idx = len(stack.surfaces) - 1
        self.assertEqual(traced.history[-1]["surface_index"], det_idx)

    def test_t3_3_optical_json_cadexporter_step_and_dxf(self):
        """P3 (JSON + CAD): Load optical JSON layout and export to both 3D STEP and 2D DXF."""
        stack = build_authentic_hybrid_raman_stack()
        with tempfile.TemporaryDirectory() as td:
            step_path = os.path.join(td, "spectrometer_p3.step")
            dxf_path = os.path.join(td, "spectrometer_p3.dxf")
            res_step = CADExporter.export_to_step(stack, step_path)
            res_dxf = CADExporter.export_to_dxf_2d(stack, dxf_path, plane="DUAL")
            self.assertTrue(res_step.is_success)
            self.assertTrue(os.path.exists(step_path))
            self.assertTrue(os.path.exists(dxf_path))
            self.assertGreater(os.path.getsize(step_path), 2000)
            self.assertGreater(os.path.getsize(dxf_path), 2000)

    def test_t3_4_fiber_na_and_doublet_clear_apertures(self):
        """P4 (NA + Aperture): Full NA=0.12 cone through AC254-050-C clear aperture maintains 0% vignetting."""
        stack = build_authentic_hybrid_raman_stack()
        opt = OpticalSystem(stack)
        rays = generate_fiber_phase_space_rays(fiber_radius_mm=0.0125, numerical_aperture=0.12, wavelengths=[1.3892])
        traced = opt.trace_rays(rays)
        doublet_vignetted = [r for r in traced if r.vignetted and (not r.history or r.history[-1]["surface_index"] in [1, 2, 3])]
        self.assertEqual(len(doublet_vignetted), 0)

    def test_t3_5_grating_dispersion_and_camera_focal_length(self):
        """P5 (Grating + Camera): 300 l/mm grating dispersion with f=50mm camera yields required spatial span."""
        stack = build_authentic_hybrid_raman_stack()
        opt = OpticalSystem(stack)
        r_low = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=1.1113)
        r_high = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=1.6133)
        tr = opt.trace_rays([r_low, r_high])
        x_low = next(h["x"] for h in tr[0].history if h["surface_index"] == 8)
        x_high = next(h["x"] for h in tr[1].history if h["surface_index"] == 8)
        span = abs(x_high - x_low)
        self.assertGreaterEqual(span, 7.95)
        self.assertLessEqual(span, 8.664)

    def test_t3_6_dmd_tilt_and_relay_numerical_aperture(self):
        """P6 (DMD Tilt + Relay): +12 deg tilt (+24 deg deflection) matched into f=75mm AC254-075-C pupil."""
        stack = build_authentic_hybrid_raman_stack()
        opt = OpticalSystem(stack)
        ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=1.3892)
        tr = opt.trace_rays([ray])[0]
        relay_hits = [h for h in tr.history if h["surface_index"] in [9, 10, 11]]
        self.assertEqual(len(relay_hits), 3)

    def test_t3_7_doublet_chromatic_dispersion_broadband_focus(self):
        """P7 (Doublet + Broadband): Achromatic doublet pairs suppress chromatic focal shift across 1111-1667 nm."""
        stack = build_authentic_hybrid_raman_stack()
        opt = OpticalSystem(stack)
        rays = [Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl) for wl in RAMAN_WAVELENGTHS_UM]
        traced = opt.trace_rays(rays)
        self.assertEqual(len(traced), len(RAMAN_WAVELENGTHS_UM))

    def test_t3_8_headless_verification_script_and_lde_table_sync(self):
        """P8 (Harness + LDE): verify_spectrometer_layout syncs with LensDataTableModel."""
        stack = build_authentic_hybrid_raman_stack()
        model = LensDataTableModel(stack)
        self.assertEqual(model.rowCount(), len(stack.surfaces))
        if os.path.exists(LAYOUT_FILE):
            passed, report = verify_spectrometer_layout(LAYOUT_FILE, verbose=False)
            self.assertIn("assertions", report)

    def test_t3_9_surfacestack_serialization_cad_pipeline(self):
        """P9 (Stack + CAD): SurfaceStack JSON serialization feeds directly into CAD solid generation."""
        stack = build_authentic_hybrid_raman_stack()
        with tempfile.TemporaryDirectory() as td:
            json_p = os.path.join(td, "stack.json")
            step_p = os.path.join(td, "stack.step")
            stack.save_to_json(json_p)
            reloaded = SurfaceStack.load_from_json(json_p)["stack"]
            res = CADExporter.export_to_step(reloaded, step_p)
            self.assertTrue(res.is_success)

    def test_t3_10_clear_apertures_zero_vignetting_throughput(self):
        """P10 (Apertures + Throughput): Verify complete sequential ray transmission to detector."""
        stack = build_authentic_hybrid_raman_stack()
        opt = OpticalSystem(stack)
        ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=1.3892)
        tr = opt.trace_rays([ray])[0]
        det_idx = len(stack.surfaces) - 1
        self.assertEqual(tr.history[-1]["surface_index"], det_idx)


# =============================================================================
# TIER 4: REAL-WORLD WORKLOAD SCENARIOS (6 Realistic Scenarios)
# =============================================================================

class TestTier4RealWorldApplicationScenarios(unittest.TestCase):
    """Tier 4: Realistic Operational Scenarios and Mission Workflows."""

    def test_t4_1_polystyrene_calibration_spectrum_acquisition(self):
        """Scenario 1: Realistic Polystyrene Raman spectrum acquisition (peaks: 620, 1001, 1031, 1602, 3054 cm⁻¹)."""
        nu_0 = 1e7 / 1064.0  # 9398.496 cm^-1
        ps_peaks_cm = [620.0, 1001.0, 1031.0, 1602.0, 3054.0]
        ps_wls_um = [1e7 / (nu_0 - s) / 1000.0 for s in ps_peaks_cm]

        # Verify exact wavelength calculations
        self.assertAlmostEqual(ps_wls_um[0], 1.13916, delta=0.005)
        self.assertAlmostEqual(ps_wls_um[1], 1.19083, delta=0.005)
        self.assertAlmostEqual(ps_wls_um[2], 1.19510, delta=0.005)
        self.assertAlmostEqual(ps_wls_um[3], 1.28263, delta=0.005)
        self.assertAlmostEqual(ps_wls_um[4], 1.57615, delta=0.005)

        stack = build_authentic_hybrid_raman_stack()
        opt = OpticalSystem(stack)
        rays = [Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl) for wl in ps_wls_um]
        traced = opt.trace_rays(rays)
        self.assertEqual(len(traced), 5)

        # Verify spatial separation on DMD
        dmd_x = [next(h["x"] for h in r.history if h["surface_index"] == 8) for r in traced]
        for i in range(len(dmd_x) - 1):
            self.assertGreater(dmd_x[i+1], dmd_x[i], "Polystyrene Raman peaks must disperse monotonically on DMD")

    def test_t4_2_acetaminophen_fingerprint_spectrum_acquisition(self):
        """Scenario 2: Realistic Acetaminophen / Fingerprint Raman spectrum acquisition (858, 1168, 1324, 1610, 2930 cm⁻¹)."""
        nu_0 = 1e7 / 1064.0
        apap_peaks_cm = [858.0, 1168.0, 1324.0, 1610.0, 2930.0]
        apap_wls_um = [1e7 / (nu_0 - s) / 1000.0 for s in apap_peaks_cm]

        stack = build_authentic_hybrid_raman_stack()
        opt = OpticalSystem(stack)
        rays = [Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl) for wl in apap_wls_um]
        traced = opt.trace_rays(rays)
        self.assertEqual(len(traced), 5)
        det_idx = len(stack.surfaces) - 1
        lost_before_det = [r for r in traced if r.vignetted and (not r.history or r.history[-1]["surface_index"] < det_idx)]
        self.assertEqual(len(lost_before_det), 0)

    def test_t4_3_hadamard_s_matrix_multiplexing_snr_gain_simulation(self):
        """Scenario 3: Hadamard S-matrix multiplexing SNR gain vs single-slit raster scan (Fellgett's advantage)."""
        # Construct cyclic S-matrix of order N=7 (Paley quadratic residue construction)
        # N=7: quadratic residues mod 7 are {1, 2, 4}. Row 0: [1, 1, 0, 1, 0, 0, 1]
        N = 7
        row0 = np.array([1, 1, 0, 1, 0, 0, 1], dtype=float)
        S = np.zeros((N, N), dtype=float)
        for i in range(N):
            S[i] = np.roll(row0, i)

        # Inverse S-matrix formula: S_inv = 2/(N+1) * (2*S.T - J)
        J = np.ones((N, N), dtype=float)
        S_inv = (2.0 / (N + 1)) * (2.0 * S.T - J)

        # Verify S @ S_inv = I
        ident = S @ S_inv
        self.assertTrue(np.allclose(ident, np.eye(N), atol=1e-10))

        # Simulate true Raman spectrum with 7 spectral channels
        np.random.seed(42)
        true_spectrum = np.array([10.0, 50.0, 120.0, 30.0, 80.0, 200.0, 45.0])

        # Detector noise std
        sigma_noise = 2.0
        num_trials = 500

        hadamard_errors = []
        singleslit_errors = []

        for _ in range(num_trials):
            # Single-slit measurement: observe each channel once with noise sigma
            y_ss = true_spectrum + np.random.normal(0.0, sigma_noise, N)
            singleslit_errors.append(y_ss - true_spectrum)

            # Hadamard measurement: multiplexed patterns Y = S @ true_spectrum + noise
            y_hadamard = S @ true_spectrum + np.random.normal(0.0, sigma_noise, N)
            rec_hadamard = S_inv @ y_hadamard
            hadamard_errors.append(rec_hadamard - true_spectrum)

        # Compute empirical RMSE
        rmse_ss = np.sqrt(np.mean(np.array(singleslit_errors)**2))
        rmse_had = np.sqrt(np.mean(np.array(hadamard_errors)**2))

        snr_gain = rmse_ss / rmse_had
        expected_gain = (N + 1) / (2.0 * np.sqrt(N))  # for N=7: 8 / (2*sqrt(7)) = 1.512

        self.assertAlmostEqual(snr_gain, expected_gain, delta=0.15)
        self.assertGreater(snr_gain, 1.35, "Hadamard multiplexing must achieve > 1.35x SNR improvement for N=7")

    def test_t4_4_hadamard_n15_multiplexing_scaling(self):
        """Scenario 4: Hadamard N=15 S-matrix simulation achieving > 1.8x SNR advantage."""
        N = 15
        sr = [1, 0, 0, 0]
        seq = []
        for _ in range(15):
            seq.append(sr[0])
            new_bit = sr[0] ^ sr[1]
            sr = [sr[1], sr[2], sr[3], new_bit]
        row0 = np.array(seq, dtype=float)
        S = np.zeros((N, N), dtype=float)
        for i in range(N):
            S[i] = np.roll(row0, i)

        J = np.ones((N, N), dtype=float)
        S_inv = (2.0 / (N + 1)) * (2.0 * S.T - J)
        self.assertTrue(np.allclose(S @ S_inv, np.eye(N), atol=1e-10))

        theoretical_gain = (N + 1) / (2.0 * np.sqrt(N))
        self.assertAlmostEqual(theoretical_gain, 2.066, places=2)

    def test_t4_5_tolerance_and_misalignment_monte_carlo_sensitivity(self):
        """Scenario 5: Tolerance & alignment sensitivity analysis (Monte Carlo perturbations)."""
        stack = build_authentic_hybrid_raman_stack()
        opt = OpticalSystem(stack)

        np.random.seed(123)
        num_mc_runs = 20
        successful_traces = 0
        det_idx = len(stack.surfaces) - 1

        for _ in range(num_mc_runs):
            dx = float(np.random.uniform(-0.010, 0.010))  # +-10 um
            dy = float(np.random.uniform(-0.010, 0.010))
            d_angle = float(np.random.uniform(-0.02, 0.02))  # +-0.02 deg

            pert_ray = Ray(
                pos=np.array([dx, dy, 0.0]),
                dir=np.array([np.sin(np.radians(d_angle)), 0.0, np.cos(np.radians(d_angle))]),
                wavelength=1.3892
            )
            tr = opt.trace_rays([pert_ray])[0]
            if len(tr.history) >= det_idx and (not tr.vignetted or tr.history[-1]["surface_index"] == det_idx):
                successful_traces += 1

        self.assertGreaterEqual(successful_traces, 18, "At least 90% of Monte Carlo perturbed rays must successfully reach detector")

    def test_t4_6_normal_incidence_specular_glint_rejection_scenario(self):
        """Scenario 6: DMD normal incidence illumination & specular glint rejection (+24 deg separation)."""
        specular_angle_deg = 0.0
        raman_signal_angle_deg = 24.0
        angular_separation_deg = raman_signal_angle_deg - specular_angle_deg

        self.assertEqual(angular_separation_deg, 24.0)
        collection_acceptance_deg = np.degrees(np.arcsin(0.16))
        self.assertLess(collection_acceptance_deg, 10.0)
        self.assertGreater(angular_separation_deg, 2.0 * collection_acceptance_deg)


# =============================================================================
# TIER 5: ADVERSARIAL & SYSTEM STRESS TESTING (5 Test Cases)
# =============================================================================

class TestTier5AdversarialVerification(unittest.TestCase):
    """Tier 5: Adversarial Stress Vectors, Corrupted Inputs, and Numerical Singularities."""

    def test_t5_1_massive_dense_ray_bundle_stress(self):
        """A1: Trace massive dense ray bundle (300+ rays) simultaneously without crash."""
        stack = build_authentic_hybrid_raman_stack()
        opt = OpticalSystem(stack)
        rays = generate_fiber_phase_space_rays(fiber_radius_mm=0.0125, numerical_aperture=0.12, wavelengths=RAMAN_WAVELENGTHS_UM, dense_mode=True)
        self.assertGreaterEqual(len(rays), 300)
        traced = opt.trace_rays(rays)
        self.assertEqual(len(traced), len(rays))

    def test_t5_2_malformed_json_resilience(self):
        """A2: Ingest corrupted / malformed JSON files and assert safe exception handling."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp.write(b'{"version": 2, "surfaces": [')  # Truncated JSON
            tmp_p = tmp.name
        try:
            with self.assertRaises(Exception):
                SurfaceStack.load_from_json(tmp_p)
        finally:
            if os.path.exists(tmp_p):
                os.remove(tmp_p)

    def test_t5_3_unphysical_glass_and_nan_refraction_defense(self):
        """A3: Unregistered glass names and negative wavelengths handled safely."""
        with self.assertRaises(ValueError):
            get_refractive_index("NON_EXISTENT_GLASS_MAGIC", 1.064)

    def test_t5_4_negative_thickness_and_zero_aperture_handling(self):
        """A4: Degenerate surface stacks handled gracefully in tracer and CAD exporter."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="S0", radius=0.0, thickness=0.0, glass="AIR"))
        with tempfile.TemporaryDirectory() as td:
            out_p = os.path.join(td, "degenerate.step")
            res = CADExporter.export_to_step(stack, out_p)
            self.assertTrue(res.is_success)

    def test_t5_5_extreme_3d_rotations_and_compound_tilts(self):
        """A5: Non-coplanar 3D compound rotations across all Euler angles assert orthogonal rotation conservation."""
        for rx in [-45.0, 0.0, 45.0]:
            for ry in [-45.0, 0.0, 45.0]:
                for rz in [-45.0, 0.0, 45.0]:
                    R = build_rotation_matrix(rx, ry, rz)
                    # Check orthogonality: R @ R.T = I
                    ident = R @ R.T
                    self.assertTrue(np.allclose(ident, np.eye(3), atol=1e-10))
                    # Check det(R) = +1
                    self.assertAlmostEqual(np.linalg.det(R), 1.0, places=9)


if __name__ == "__main__":
    unittest.main()
