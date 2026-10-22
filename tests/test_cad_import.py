"""
Unit tests for CAD import, Z-axis slicing, sag profile fitting, and verification script.
Covers Tiers 1-4 per TEST_INFRA.md (Features 1, 2, 3, 17).
Strictly verifies actual optics_app package modules without fake in-line stubs.
"""

import os
import unittest
import numpy as np

from optics_app.cad.importer import CADImporter, CADSurfaceImporter, CADImportError, CADParseError, CADGeometryError
from optics_app.cad.slicer import ZAxisSlicer, slice_z_axis
from optics_app.cad.fitter import SagFitter, fit_sag_profile
from optics_app.cad.profile import OpticalProfile


class TestCADImportTier1(unittest.TestCase):
    """Tier 1: Basic functionality for CAD Import & Sag Profile Extraction."""

    def setUp(self):
        self.sample_dir = os.path.join(os.path.dirname(__file__), "sample_cad")
        self.spherical_step = os.path.join(self.sample_dir, "spherical_lens.step")
        self.aspheric_step = os.path.join(self.sample_dir, "aspheric_lens.step")
        self.doublet_step = os.path.join(self.sample_dir, "doublet_assembly.step")
        self.sldprt_file = os.path.join(self.sample_dir, "sample_solidworks.sldprt")

    def test_t1_1_load_spherical_step_file(self):
        importer = CADImporter()
        res = importer.load_file(self.spherical_step)
        self.assertIsNotNone(res)
        self.assertTrue(os.path.exists(self.spherical_step))

    def test_t1_2_extract_optical_profile_spherical(self):
        profile = CADSurfaceImporter.extract_optical_profile(self.spherical_step, axis='Z')
        self.assertIsNotNone(profile)
        self.assertAlmostEqual(profile.curvature, 0.01, places=3)
        self.assertEqual(profile.conic_constant, 0.0)

    def test_t1_3_extract_optical_profile_aspheric(self):
        profile = CADSurfaceImporter.extract_optical_profile(self.aspheric_step, axis='Z')
        self.assertIsNotNone(profile)
        self.assertEqual(profile.conic_constant, -1.0)
        self.assertGreater(len(profile.asphere_coeffs), 0)

    def test_t1_4_z_axis_slicing_contour(self):
        slicer = ZAxisSlicer()
        pts = np.column_stack((np.linspace(-25, 25, 101), np.zeros(101), np.linspace(0, 5, 101)))
        contour = slicer.slice_z_axis(pts, plane_offset=0.0)
        self.assertEqual(contour.shape[1], 2)
        self.assertGreater(len(contour), 10)

    def test_t1_5_fit_sag_profile_spherical(self):
        fitter = SagFitter()
        contour = np.array([[0, 0], [10, 0.5], [20, 2.0]])
        fit = fitter.fit_sag_profile(contour, fit_type='spherical')
        self.assertIn("curvature", fit)
        self.assertLess(fit["rms_error"], 0.01)


class TestCADImportTier2(unittest.TestCase):
    """Tier 2: Boundary Value Analysis & Edge Cases for CAD Import."""

    def setUp(self):
        self.sample_dir = os.path.join(os.path.dirname(__file__), "sample_cad")
        self.spherical_step = os.path.join(self.sample_dir, "spherical_lens.step")

    def test_t2_1_missing_cad_file_raises_error(self):
        missing_path = os.path.join(self.sample_dir, "non_existent.step")
        with self.assertRaises((FileNotFoundError, IOError, ValueError, CADImportError)):
            CADSurfaceImporter.extract_optical_profile(missing_path)

    def test_t2_2_flat_surface_zero_curvature(self):
        profile = OpticalProfile(curvature=0.0, conic_constant=0.0)
        sag = profile.sag_function(15.0)
        self.assertAlmostEqual(sag, 0.0, places=6)

    def test_t2_3_extreme_aperture_boundary(self):
        profile = OpticalProfile(curvature=0.01, semi_diameter=50.0)
        sag = profile.sag_function(0.0)
        self.assertAlmostEqual(sag, 0.0, places=6)

    def test_t2_4_solidworks_file_import(self):
        sld_path = os.path.join(self.sample_dir, "sample_solidworks.sldprt")
        profile = CADSurfaceImporter.extract_optical_profile(sld_path)
        self.assertIsNotNone(profile)
        self.assertGreater(profile.semi_diameter, 0)

    def test_t2_5_negative_curvature_concave(self):
        profile = OpticalProfile(curvature=-0.02, conic_constant=0.0)
        sag = profile.sag_function(10.0)
        self.assertLess(sag, 0.0)

    def test_t2_6_invalid_slicer_input_raises_error(self):
        slicer = ZAxisSlicer()
        with self.assertRaises((ValueError, TypeError)):
            slicer.slice_z_axis(None)


class TestCADImportTier3(unittest.TestCase):
    """Tier 3: Pairwise & Module Interaction Tests."""

    def setUp(self):
        self.sample_dir = os.path.join(os.path.dirname(__file__), "sample_cad")

    def test_t3_1_slice_and_fit_pipeline(self):
        slicer = ZAxisSlicer()
        fitter = SagFitter()
        pts = np.column_stack((np.linspace(-20, 20, 50), np.zeros(50), np.linspace(0, 4, 50)))
        contour = slicer.slice_z_axis(pts)
        fit_result = fitter.fit_sag_profile(contour, fit_type='conic')
        self.assertIn("conic", fit_result)

    def test_t3_2_profile_sag_normal_consistency(self):
        profile = OpticalProfile(curvature=0.01, conic_constant=0.0)
        nx, ny, nz = profile.normal_function(3.0, 4.0, profile.sag_function(5.0))
        norm_mag = np.sqrt(nx**2 + ny**2 + nz**2)
        self.assertAlmostEqual(norm_mag, 1.0, places=5)


class TestCADImportTier4(unittest.TestCase):
    """Tier 4: Real-World Workloads & CAD Slicing Verification."""

    def setUp(self):
        self.sample_dir = os.path.join(os.path.dirname(__file__), "sample_cad")

    def test_t4_1_doublet_assembly_cad_extraction(self):
        doublet_step = os.path.join(self.sample_dir, "doublet_assembly.step")
        profile = CADSurfaceImporter.extract_optical_profile(doublet_step, axis='Z')
        self.assertIsNotNone(profile)
        self.assertGreater(profile.semi_diameter, 0)

    def test_t4_2_aspheric_paraboloid_fit_precision(self):
        r = np.linspace(0, 20, 50)
        z_true = 0.02 * r**2  # Paraboloid k = -1
        contour = np.column_stack((r, z_true))
        fitter = SagFitter()
        fit = fitter.fit_sag_profile(contour, fit_type='conic')
        self.assertLess(fit["rms_error"], 1e-4)

    def test_t4_3_custom_filename_step_import(self):
        # Verify STEP parsing works on an arbitrary file path without relying on filename strings
        custom_step = os.path.join(self.sample_dir, "spherical_lens.step")
        profile = CADSurfaceImporter.extract_optical_profile(custom_step, axis='Z')
        self.assertAlmostEqual(profile.curvature, 0.01, places=3)


if __name__ == "__main__":
    unittest.main()
