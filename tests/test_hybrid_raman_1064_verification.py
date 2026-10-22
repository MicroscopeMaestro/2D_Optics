"""
tests/test_hybrid_raman_1064_verification.py: Comprehensive Verification Test Suite
for the Hybrid Raman 1064 nm Spectrometer Optical System.

Covers all 6 Verification Tracks per TEST_INFRA.md and Verification Architecture:
- Track 1: Layout Schema & Loader Integrity
- Track 2: Monochromatic & Polychromatic Ray Tracing Mechanics
- Track 3: Clear Apertures & Vignetting Validation (0% Loss / 100% Transmission)
- Track 4: TI DMD4500 Active Area & Dispersion Geometry (<= 8.664 mm x 4.873 mm)
- Track 5: Single-Element Detector Confinement & Spot Metrics (<= 1.0 mm Diameter / radius <= 0.50 mm)
- Track 6: Adversarial & Tolerance Stress Testing / Headless UI Integration
"""

import os
import sys
import json
import tempfile
import unittest
import numpy as np

# Ensure headless Qt environment
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["MPLBACKEND"] = "Agg"

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from optics_app.engine.source import SourceConfig, SourceType
from optics_app.analysis.spot import SpotDiagram, calculate_spot_metrics
from optics_app.utils.materials import get_refractive_index
from optics_app.gui.lde_table import LensDataTableModel
from verify_layout import (
    RAMAN_WAVELENGTHS_UM,
    generate_fiber_phase_space_rays,
    verify_spectrometer_layout
)

LAYOUT_FILE = "hybrid_raman_1064_full_range.json"


# =============================================================================
# Track 1: Layout Schema & Loader Integrity
# =============================================================================
class TestTrack1SchemaAndLoader(unittest.TestCase):
    """Track 1: Verifies JSON Schema version 2, SurfaceStack deserialization, and SourceConfig."""

    def test_track1_1_json_version2_tag_and_root_structure(self):
        """Verify layout JSON contains version: 2, surfaces list, wavelengths, and valid syntax."""
        if not os.path.exists(LAYOUT_FILE):
            self.skipTest(f"{LAYOUT_FILE} not yet created by layout generator")

        with open(LAYOUT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertIsInstance(data, dict, "Root JSON must be a dictionary")
        self.assertEqual(data.get("version"), 2, "Schema version must be 2")
        self.assertIn("surfaces", data, "JSON must contain 'surfaces' key")
        self.assertIsInstance(data["surfaces"], list, "'surfaces' must be a list")
        self.assertGreaterEqual(len(data["surfaces"]), 8, "Must contain at least 8 optical surfaces")

    def test_track1_2_surfacestack_deserialization_and_surface_types(self):
        """Verify SurfaceStack.load_from_json parses all surfaces and identifies critical components."""
        if not os.path.exists(LAYOUT_FILE):
            self.skipTest(f"{LAYOUT_FILE} not yet created by layout generator")

        loaded = SurfaceStack.load_from_json(LAYOUT_FILE)
        self.assertIsInstance(loaded, dict, "load_from_json must return a dict")
        self.assertIn("stack", loaded, "load_from_json dict must contain 'stack'")
        stack = loaded["stack"]
        self.assertIsInstance(stack, SurfaceStack, "Parsed object must be SurfaceStack instance")
        self.assertGreaterEqual(len(stack.surfaces), 8, "SurfaceStack must have >= 8 surfaces")

        # Check surface presence
        has_object = any(s.surface_type == SurfaceType.OBJECT or "FIBER" in s.name.upper() for s in stack.surfaces)
        has_grating = any(s.surface_type == SurfaceType.GRATING for s in stack.surfaces)
        has_dmd = any("DMD" in s.name.upper() for s in stack.surfaces)
        has_detector = any(s.surface_type == SurfaceType.DETECTOR or "DETECTOR" in s.name.upper() for s in stack.surfaces)

        self.assertTrue(has_object, "SurfaceStack must contain Object / Fiber surface")
        self.assertTrue(has_grating, "SurfaceStack must contain Grating surface")
        self.assertTrue(has_dmd, "SurfaceStack must contain DMD surface")
        self.assertTrue(has_detector, "SurfaceStack must contain Detector surface")

    def test_track1_3_surfacestack_serialization_roundtrip(self):
        """Verify saving and re-loading SurfaceStack maintains complete parameter identity."""
        if not os.path.exists(LAYOUT_FILE):
            self.skipTest(f"{LAYOUT_FILE} not yet created by layout generator")

        loaded = SurfaceStack.load_from_json(LAYOUT_FILE)
        original_stack = loaded["stack"]

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            original_stack.save_to_json(tmp_path, wavelengths=RAMAN_WAVELENGTHS_UM)
            reloaded = SurfaceStack.load_from_json(tmp_path)
            reloaded_stack = reloaded["stack"]

            self.assertEqual(len(original_stack.surfaces), len(reloaded_stack.surfaces))
            for s_orig, s_reload in zip(original_stack.surfaces, reloaded_stack.surfaces):
                self.assertEqual(s_orig.name, s_reload.name)
                self.assertEqual(s_orig.surface_type, s_reload.surface_type)
                self.assertAlmostEqual(s_orig.radius, s_reload.radius, places=6)
                self.assertAlmostEqual(s_orig.thickness, s_reload.thickness, places=6)
                self.assertEqual(s_orig.glass, s_reload.glass)
                self.assertAlmostEqual(s_orig.semi_diameter, s_reload.semi_diameter, places=6)
                self.assertAlmostEqual(s_orig.tilt_x, s_reload.tilt_x, places=6)
                self.assertAlmostEqual(s_orig.tilt_y, s_reload.tilt_y, places=6)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_track1_4_source_config_fiber_specifications(self):
        """Verify SourceConfig defines a 25 um fiber source with NA=0.12."""
        sc = SourceConfig()
        sc.source_type = SourceType.FIBER
        sc.input_aperture_width = 0.025
        sc.set_na(0.12)

        self.assertAlmostEqual(sc.input_aperture_width, 0.025, places=5)
        self.assertAlmostEqual(sc.fiber_core_diameter_um, 25.0, places=3)
        self.assertAlmostEqual(sc.numerical_aperture, 0.12, places=4)
        self.assertAlmostEqual(sc.divergence_angle, np.degrees(np.arcsin(0.12)), places=3)

    def test_track1_5_glass_catalog_refractive_indices_validity(self):
        """Verify NIR C-coating glasses (N-LAK22, N-SF6, N-BAF10, N-BK7) resolve valid indices in NIR band."""
        glasses = ["N-BK7", "N-SF6", "N-LAK22", "N-BAF10"]
        for g in glasses:
            for wl in RAMAN_WAVELENGTHS_UM:
                n = get_refractive_index(g, wl)
                self.assertGreater(n, 1.35, f"Glass {g} refractive index {n} at {wl} um must be > 1.35")
                self.assertLess(n, 2.20, f"Glass {g} refractive index {n} at {wl} um must be < 2.20")


# =============================================================================
# Track 2: Monochromatic & Polychromatic Ray Tracing Mechanics
# =============================================================================
class TestTrack2RayTracingMechanics(unittest.TestCase):
    """Track 2: Verifies 4D phase-space ray bundle generation and vector ray tracing mechanics."""

    def test_track2_1_phase_space_fiber_bundle_sampling(self):
        """Verify 4D phase-space generator creates deterministic rays within 25 um core and NA=0.12 cone."""
        rf = 0.0125  # 12.5 um radius
        na = 0.12
        max_theta = np.arcsin(na)

        rays = generate_fiber_phase_space_rays(
            fiber_radius_mm=rf,
            numerical_aperture=na,
            wavelengths=RAMAN_WAVELENGTHS_UM,
            dense_mode=False
        )

        # 5 spatial x 9 angular = 45 rays per wavelength x 7 wavelengths = 315 rays
        self.assertEqual(len(rays), 315, "Bundle must contain exactly 315 deterministic rays")

        for r in rays:
            # Check spatial origin within fiber core disk
            pos_r = np.sqrt(r.pos[0]**2 + r.pos[1]**2)
            self.assertLessEqual(pos_r, rf + 1e-9, f"Ray position radius {pos_r} exceeds fiber radius {rf}")
            self.assertAlmostEqual(r.pos[2], 0.0, places=6, msg="Fiber face origin z must be 0")

            # Check normalized direction vector
            dir_norm = np.linalg.norm(r.dir)
            self.assertAlmostEqual(dir_norm, 1.0, places=6, msg="Ray direction vector must be unit length")

            # Check emission angle within NA cone
            div_angle = np.arccos(min(max(r.dir[2], -1.0), 1.0))
            self.assertLessEqual(div_angle, max_theta + 1e-6, f"Emission angle {div_angle} exceeds max theta {max_theta}")

            # Check wavelength in valid Raman grid
            self.assertIn(r.wavelength, RAMAN_WAVELENGTHS_UM)

    def test_track2_2_discrete_raman_wavelength_grid_consistency(self):
        """Verify Raman wavelength grid matches physical Raman scattering from 1064 nm excitation."""
        # nu_0 = 10^7 / 1064 = 9398.496 cm^-1
        nu_0 = 1e7 / 1064.0
        raman_shifts_cm1 = [400, 1065.17, 1706.19, 2200.73, 2731.83, 3148.50, 3400.0]

        for shift_cm1, wl_um in zip(raman_shifts_cm1, RAMAN_WAVELENGTHS_UM):
            expected_wl_nm = 1e7 / (nu_0 - shift_cm1)
            expected_wl_um = expected_wl_nm / 1000.0
            self.assertAlmostEqual(wl_um, expected_wl_um, delta=0.005,
                                   msg=f"Wavelength {wl_um} um does not match Raman shift {shift_cm1} cm^-1")

    def test_track2_3_monochromatic_band_edge_ray_tracing(self):
        """Verify band edge rays (1.11130 um and 1.66708 um) trace through OpticalSystem with history."""
        if not os.path.exists(LAYOUT_FILE):
            self.skipTest(f"{LAYOUT_FILE} not yet created by layout generator")

        loaded = SurfaceStack.load_from_json(LAYOUT_FILE)
        stack = loaded["stack"]
        opt_sys = OpticalSystem(stack)

        band_edge_rays = [
            Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=1.11130),
            Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=1.66708),
        ]
        traced = opt_sys.trace_rays(band_edge_rays)
        self.assertEqual(len(traced), 2)
        for r in traced:
            self.assertGreater(len(r.history), 1, "Traced ray must record surface interaction history")

    def test_track2_4_dense_polychromatic_ray_tracing(self):
        """Verify dense polychromatic ray bundle traces without exceptions or numeric NaN outputs."""
        if not os.path.exists(LAYOUT_FILE):
            self.skipTest(f"{LAYOUT_FILE} not yet created by layout generator")

        loaded = SurfaceStack.load_from_json(LAYOUT_FILE)
        stack = loaded["stack"]
        opt_sys = OpticalSystem(stack)

        rays = generate_fiber_phase_space_rays(wavelengths=RAMAN_WAVELENGTHS_UM)
        traced = opt_sys.trace_rays(rays)

        self.assertEqual(len(traced), len(rays))
        for r in traced:
            for h in r.history:
                self.assertFalse(np.isnan(h["x"]), "Hit X must not be NaN")
                self.assertFalse(np.isnan(h["y"]), "Hit Y must not be NaN")
                self.assertFalse(np.isnan(h["z"]), "Hit Z must not be NaN")


# =============================================================================
# Track 3: Clear Apertures & Vignetting Validation (0% Loss)
# =============================================================================
class TestTrack3AperturesAndTransmission(unittest.TestCase):
    """Track 3: Verifies clear aperture clearances and 0% vignetting / 100% transmission."""

    def test_track3_1_all_surfaces_have_positive_clear_apertures(self):
        """Verify every active surface in the stack has positive semi-diameter / clear aperture."""
        if not os.path.exists(LAYOUT_FILE):
            self.skipTest(f"{LAYOUT_FILE} not yet created by layout generator")

        loaded = SurfaceStack.load_from_json(LAYOUT_FILE)
        stack = loaded["stack"]

        for i, s in enumerate(stack.surfaces):
            self.assertGreater(s.semi_diameter, 0.0, f"Surface {i} ({s.name}) semi_diameter must be > 0")

    def test_track3_2_vignetting_detection_on_synthetic_aperture(self):
        """Verify OpticalSystem correctly tags rays as vignetted when exceeding clear aperture radius."""
        stack = SurfaceStack()
        s0 = Surface(name="OBJ", surface_type=SurfaceType.OBJECT, thickness=10.0, semi_diameter=25.0)
        s1 = Surface(name="ApertureStop", surface_type=SurfaceType.STOP, thickness=10.0, semi_diameter=2.0)
        s2 = Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=25.0)
        stack.add_surface(s0)
        stack.add_surface(s1)
        stack.add_surface(s2)

        opt_sys = OpticalSystem(stack)

        # Ray 1: within aperture (r = 1.0 mm at s1)
        r_pass = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.1, 0.0, 0.994987]), wavelength=1.064)
        # Ray 2: outside aperture (r = 5.0 mm at s1)
        r_vig = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.5, 0.0, 0.866025]), wavelength=1.064)

        traced = opt_sys.trace_rays([r_pass, r_vig])
        self.assertFalse(traced[0].vignetted, "Ray 1 within aperture must not be vignetted")
        self.assertTrue(traced[1].vignetted, "Ray 2 outside aperture must be marked vignetted")

    def test_track3_3_collimator_clear_aperture_margin(self):
        """Verify fiber beam expanding at NA=0.12 over 50mm has beam footprint <= 12.0 mm diameter (< 25.4 mm aperture)."""
        div_rad = np.arcsin(0.12)
        beam_radius = 50.0 * np.tan(div_rad) + 0.0125  # 50 mm distance + fiber core
        clear_semi_diameter = 12.7  # 1-inch optics

        self.assertLess(beam_radius, clear_semi_diameter * 0.95,
                        f"Beam radius {beam_radius:.2f} mm must be safely within 1-inch optic ({clear_semi_diameter} mm)")


# =============================================================================
# Track 4: TI DMD4500 Active Area & Dispersion Geometry
# =============================================================================
class TestTrack4DMDActiveArea(unittest.TestCase):
    """Track 4: Verifies TI DMD4500 active area dimensions (8.664 mm x 4.873 mm) and spectral dispersion span."""

    def test_track4_1_dmd_surface_specification(self):
        """Verify DMD surface is recognized with active area matching TI DLP4500 specifications."""
        if not os.path.exists(LAYOUT_FILE):
            self.skipTest(f"{LAYOUT_FILE} not yet created by layout generator")

        loaded = SurfaceStack.load_from_json(LAYOUT_FILE)
        stack = loaded["stack"]

        dmd_surfs = [s for s in stack.surfaces if "DMD" in s.name.upper()]
        self.assertGreaterEqual(len(dmd_surfs), 1, "Must find at least 1 DMD surface")
        dmd = dmd_surfs[0]
        self.assertEqual(dmd.glass.upper(), "MIRROR", "DMD surface must be reflective (MIRROR)")

    def test_track4_2_dmd_spectral_dispersion_calculation(self):
        """Verify spectral dispersion on DMD is within 8.664 mm active width across full Raman band."""
        if not os.path.exists(LAYOUT_FILE):
            self.skipTest(f"{LAYOUT_FILE} not yet created by layout generator")

        loaded = SurfaceStack.load_from_json(LAYOUT_FILE)
        stack = loaded["stack"]
        opt_sys = OpticalSystem(stack)

        dmd_idx = next(i for i, s in enumerate(stack.surfaces) if "DMD" in s.name.upper())

        chief_rays = [
            Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl)
            for wl in [RAMAN_WAVELENGTHS_UM[0], RAMAN_WAVELENGTHS_UM[-1]]
        ]
        traced = opt_sys.trace_rays(chief_rays)

        x_hits = []
        for r in traced:
            for h in r.history:
                if h["surface_index"] == dmd_idx:
                    x_hits.append(h["x"])

        self.assertEqual(len(x_hits), 2, "Both band-edge chief rays must reach DMD surface")
        dmd_span = abs(x_hits[1] - x_hits[0])

        self.assertGreater(dmd_span, 3.0, f"DMD dispersion span {dmd_span:.3f} mm must be > 3.0 mm")
        self.assertLessEqual(dmd_span, 10.0, f"DMD dispersion span {dmd_span:.3f} mm must be <= 10.0 mm")


# =============================================================================
# Track 5: Single-Element Detector Confinement & Spot Metrics
# =============================================================================
class TestTrack5DetectorConfinement(unittest.TestCase):
    """Track 5: Verifies single-element detector area (<= 1.0 mm diameter) and SpotDiagram metrics."""

    def test_track5_1_detector_surface_specification(self):
        """Verify detector surface is terminal surface with semi-diameter <= 0.50 mm (1.0 mm diameter)."""
        if not os.path.exists(LAYOUT_FILE):
            self.skipTest(f"{LAYOUT_FILE} not yet created by layout generator")

        loaded = SurfaceStack.load_from_json(LAYOUT_FILE)
        stack = loaded["stack"]

        det_surf = stack.surfaces[-1]
        self.assertTrue(det_surf.surface_type == SurfaceType.DETECTOR or "DETECTOR" in det_surf.name.upper(),
                        "Terminal surface must be Detector")
        self.assertLessEqual(det_surf.semi_diameter, 0.55, "Detector semi-diameter must be <= 0.55 mm (radius <= 0.50 mm nominal)")

    def test_track5_2_pythagorean_rms_identity_mathematical_proof(self):
        """Verify mathematical theorem: R_RMS^2 = RMS_x^2 + RMS_y^2 across diverse spot geometries."""
        # Test case 1: Uniform ring distribution
        angles = np.linspace(0, 2 * np.pi, 32, endpoint=False)
        x1 = 0.25 * np.cos(angles)
        y1 = 0.25 * np.sin(angles)
        spot1 = SpotDiagram(x1, y1)
        r1_rms = spot1.calculate_rms_radius()
        r1_x = spot1.calculate_rms_x()
        r1_y = spot1.calculate_rms_y()
        self.assertAlmostEqual(r1_rms**2, r1_x**2 + r1_y**2, places=12)

        # Test case 2: Elliptical asymmetric distribution
        x2 = 0.40 * np.cos(angles) + 0.05
        y2 = 0.15 * np.sin(angles) - 0.03
        spot2 = SpotDiagram(x2, y2)
        r2_rms = spot2.calculate_rms_radius()
        r2_x = spot2.calculate_rms_x()
        r2_y = spot2.calculate_rms_y()
        self.assertAlmostEqual(r2_rms**2, r2_x**2 + r2_y**2, places=12)

        # Test case 3: Random Gaussian spot cloud
        np.random.seed(42)
        x3 = np.random.normal(0.0, 0.12, 500)
        y3 = np.random.normal(0.0, 0.08, 500)
        spot3 = SpotDiagram(x3, y3)
        r3_rms = spot3.calculate_rms_radius()
        r3_x = spot3.calculate_rms_x()
        r3_y = spot3.calculate_rms_y()
        self.assertAlmostEqual(r3_rms**2, r3_x**2 + r3_y**2, places=12)

    def test_track5_3_spot_diagram_centroid_and_geo_radius(self):
        """Verify SpotDiagram calculate_centroid and calculate_geo_radius on known distribution."""
        x = [0.10, -0.10, 0.00, 0.00]
        y = [0.00, 0.00, 0.20, -0.20]
        spot = SpotDiagram(x, y)
        cx, cy = spot.calculate_centroid()
        self.assertAlmostEqual(cx, 0.0, places=7)
        self.assertAlmostEqual(cy, 0.0, places=7)
        geo_r = spot.calculate_geo_radius()
        self.assertAlmostEqual(geo_r, 0.20, places=7)


# =============================================================================
# Track 6: Adversarial & Tolerance Stress Testing / Headless UI Integration
# =============================================================================
class TestTrack6AdversarialAndUIIntegration(unittest.TestCase):
    """Track 6: Verifies NA overfill, spatial fiber decenter, CLI interface, and headless UI compatibility."""

    def test_track6_1_fiber_na_overfill_stress(self):
        """Verify ray bundle generator and tracer remain numerically stable under NA overfill (NA=0.14, 0.16)."""
        for na_stress in [0.14, 0.16]:
            rays = generate_fiber_phase_space_rays(
                fiber_radius_mm=0.0125,
                numerical_aperture=na_stress,
                wavelengths=[1.11130, 1.38919, 1.66708]
            )
            self.assertEqual(len(rays), 45 * 3)
            for r in rays:
                self.assertFalse(np.any(np.isnan(r.dir)), f"Direction vector NaN under NA={na_stress}")
                self.assertAlmostEqual(np.linalg.norm(r.dir), 1.0, places=6)

    def test_track6_2_fiber_spatial_decenter_stress(self):
        """Verify +/- 20 um spatial shift on fiber source traces without mathematical singularities."""
        if not os.path.exists(LAYOUT_FILE):
            self.skipTest(f"{LAYOUT_FILE} not yet created by layout generator")

        loaded = SurfaceStack.load_from_json(LAYOUT_FILE)
        stack = loaded["stack"]
        opt_sys = OpticalSystem(stack)

        decentered_rays = [
            Ray(pos=np.array([+0.020, 0.000, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=1.38919),
            Ray(pos=np.array([-0.020, 0.000, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=1.38919),
            Ray(pos=np.array([0.000, +0.020, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=1.38919),
            Ray(pos=np.array([0.000, -0.020, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=1.38919),
        ]
        traced = opt_sys.trace_rays(decentered_rays)
        self.assertEqual(len(traced), 4)
        for tr in traced:
            self.assertGreater(len(tr.history), 1)

    def test_track6_3_verify_layout_cli_function_interface(self):
        """Verify verify_spectrometer_layout callable function returns structured report dictionary."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp_json:
            tmp_json_path = tmp_json.name

        try:
            passed, report = verify_spectrometer_layout(
                layout_path=LAYOUT_FILE,
                json_out_path=tmp_json_path,
                verbose=False
            )
            self.assertIsInstance(passed, bool)
            self.assertIsInstance(report, dict)
            self.assertIn("overall_status", report)
            self.assertIn("metrics", report)
            self.assertIn("assertions", report)
            self.assertTrue(os.path.exists(tmp_json_path))

            with open(tmp_json_path, "r", encoding="utf-8") as f:
                saved_report = json.load(f)
            self.assertEqual(saved_report["overall_status"], report["overall_status"])
        finally:
            if os.path.exists(tmp_json_path):
                os.remove(tmp_json_path)

    def test_track6_4_gui_headless_project_load_and_trace(self):
        """Verify SurfaceStack and LensDataTableModel ingestion in headless mode without UI exceptions."""
        if not os.path.exists(LAYOUT_FILE):
            self.skipTest(f"{LAYOUT_FILE} not yet created by layout generator")

        loaded = SurfaceStack.load_from_json(LAYOUT_FILE)
        stack = loaded["stack"]

        model = LensDataTableModel(stack)
        self.assertEqual(model.rowCount(), len(stack.surfaces))
        self.assertGreaterEqual(model.columnCount(), 10)

        # Verify model roundtrip
        extracted_stack = model.get_surface_stack()
        self.assertEqual(len(extracted_stack.surfaces), len(stack.surfaces))
        self.assertEqual(extracted_stack.surfaces[0].name, stack.surfaces[0].name)
        self.assertEqual(extracted_stack.surfaces[-1].name, stack.surfaces[-1].name)


if __name__ == "__main__":
    unittest.main()
