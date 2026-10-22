"""
Unit tests for Zemax-style Spot Diagram Diagnostics (Centroid, RMS Radius, GEO Radius, Airy Disk Circle).
Covers Tiers 1-4 per TEST_INFRA.md (Feature 10).
"""

import unittest
import numpy as np
from optics_app.analysis.spot import SpotDiagram, calculate_spot_metrics, generate_spot_diagram_from_system
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem


class TestSpotDiagramTier1(unittest.TestCase):
    """Tier 1: Basic Centroid, RMS Spot Radius, GEO Radius, Airy Disk Calculations."""

    def test_t1_1_centroid_calculation_centered_pattern(self):
        x = [-10.0, 10.0, 0.0, 0.0, 0.0]
        y = [0.0, 0.0, -10.0, 10.0, 0.0]
        spot = SpotDiagram(x, y)
        cx, cy = spot.calculate_centroid()
        self.assertAlmostEqual(cx, 0.0, places=5)
        self.assertAlmostEqual(cy, 0.0, places=5)

    def test_t1_2_rms_spot_radius(self):
        # 4 points at r=10mm from origin
        x = [10.0, -10.0, 0.0, 0.0]
        y = [0.0, 0.0, 10.0, -10.0]
        spot = SpotDiagram(x, y)
        rms = spot.calculate_rms_radius()
        self.assertAlmostEqual(rms, 10.0, places=5)

    def test_t1_3_geo_spot_radius(self):
        x = [1.0, 2.0, 15.0, -3.0]
        y = [0.0, 0.0, 0.0, 0.0]
        spot = SpotDiagram(x, y)
        cx, _ = spot.calculate_centroid()  # cx = 3.75
        # Max dist from 3.75 is 15.0 - 3.75 = 11.25
        geo = spot.calculate_geo_radius()
        self.assertAlmostEqual(geo, 11.25, places=5)

    def test_t1_4_airy_disk_radius_formula(self):
        # lambda = 0.5 um, F/# = 4.0 -> r_airy = 1.22 * 0.5 * 4 = 2.44 um
        spot = SpotDiagram([0], [0], wavelength_um=0.5, f_number=4.0)
        airy_r = spot.calculate_airy_disk_radius()
        self.assertAlmostEqual(airy_r, 2.44, places=5)

    def test_t1_5_diffraction_limited_check(self):
        # RMS = 1.0 um, Airy = 2.44 um -> True
        spot = SpotDiagram([0.0, 1.0], [0.0, 0.0], wavelength_um=0.5, f_number=4.0)
        self.assertTrue(spot.is_diffraction_limited())


class TestSpotDiagramTier2(unittest.TestCase):
    """Tier 2: Boundary Value Analysis & Edge Cases."""

    def test_t2_1_single_ray_spot(self):
        spot = SpotDiagram([5.0], [10.0])
        cx, cy = spot.calculate_centroid()
        self.assertAlmostEqual(cx, 5.0, places=5)
        self.assertAlmostEqual(cy, 10.0, places=5)
        self.assertAlmostEqual(spot.calculate_rms_radius(), 0.0, places=5)
        self.assertAlmostEqual(spot.calculate_geo_radius(), 0.0, places=5)

    def test_t2_2_uniform_circular_spot_array(self):
        angles = np.linspace(0, 2*np.pi, 100, endpoint=False)
        r = 50.0
        x = r * np.cos(angles)
        y = r * np.sin(angles)
        spot = SpotDiagram(x, y)
        self.assertAlmostEqual(spot.calculate_rms_radius(), 50.0, places=4)
        self.assertAlmostEqual(spot.calculate_geo_radius(), 50.0, places=4)

    def test_t2_3_negative_wavelength_raises_error(self):
        spot = SpotDiagram([0], [0], wavelength_um=-0.5, f_number=4.0)
        with self.assertRaises(ValueError):
            spot.calculate_airy_disk_radius()

    def test_t2_4_zero_f_number_raises_error(self):
        spot = SpotDiagram([0], [0], wavelength_um=0.5, f_number=0.0)
        with self.assertRaises(ValueError):
            spot.calculate_airy_disk_radius()

    def test_t2_5_empty_spot_diagram_metrics(self):
        spot = SpotDiagram([], [])
        cx, cy = spot.calculate_centroid()
        self.assertEqual(cx, 0.0)
        self.assertEqual(cy, 0.0)
        self.assertEqual(spot.calculate_rms_radius(), 0.0)
        self.assertEqual(spot.calculate_geo_radius(), 0.0)


class TestSpotDiagramTier3(unittest.TestCase):
    """Tier 3: Pairwise Off-Axis Centroid Shift + Spot Size Expansion."""

    def test_t3_1_off_axis_field_spot_shift(self):
        x = np.array([100.0, 102.0, 98.0, 100.0])
        y = np.array([50.0, 50.0, 52.0, 48.0])
        metrics = calculate_spot_metrics(x, y)
        self.assertAlmostEqual(metrics["centroid_x"], 100.0, places=4)
        self.assertAlmostEqual(metrics["centroid_y"], 50.0, places=4)
        self.assertGreater(metrics["rms_radius"], 0.0)

    def test_t3_2_generate_spot_diagram_from_system(self):
        stack = SurfaceStack()
        stack.add_surface(Surface(name="Obj", thickness=50.0))
        stack.add_surface(Surface(name="Lens1", radius=50.0, thickness=10.0, glass="BK7", semi_diameter=25.0))
        stack.add_surface(Surface(name="Det", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=25.0))

        sys = OpticalSystem(stack)
        spot_diag = generate_spot_diagram_from_system(sys, num_rays=50, wavelength_um=0.5876, entrance_pupil_radius=10.0)
        self.assertIsInstance(spot_diag, SpotDiagram)
        self.assertGreater(len(spot_diag.x), 0)


class TestSpotDiagramTier4(unittest.TestCase):
    """Tier 4: Aspheric Singlet Spot Diagram Workload Evaluation."""

    def test_t4_1_aspheric_spot_diagram_metrics(self):
        # 1000 random rays forming aberration spot
        np.random.seed(42)
        x = np.random.normal(0.0, 12.5, 1000)
        y = np.random.normal(0.0, 12.5, 1000)
        spot = SpotDiagram(x, y, wavelength_um=0.5876, f_number=2.8)

        self.assertAlmostEqual(spot.calculate_centroid()[0], 0.0, delta=2.0)
        self.assertGreater(spot.calculate_geo_radius(), spot.calculate_rms_radius())


if __name__ == "__main__":
    unittest.main()
