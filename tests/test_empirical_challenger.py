"""
Adversarial empirical challenge suite for Milestone 1 CAD Import & Profile Extraction Engine.
"""

import os
import sys
import shutil
import tempfile
import unittest
import numpy as np

from optics_app.cad.importer import (
    CADImporter,
    CADSurfaceImporter,
    STEPParser,
    CADParseError,
    CADGeometryError,
    CADImportError,
    CADFileFormat
)
from optics_app.cad.slicer import ZSlicer, SlicedLensGeometry
from optics_app.cad.fitter import SagFitter, ProfilePoints, fit_sag_profile
from optics_app.cad.profile import OpticalProfile, FitType


class TestEmpiricalCADChallenge(unittest.TestCase):

    def setUp(self):
        self.sample_dir = os.path.join(os.path.dirname(__file__), "sample_cad")
        self.spherical_src = os.path.join(self.sample_dir, "spherical_lens.step")
        self.aspheric_src = os.path.join(self.sample_dir, "aspheric_lens.step")
        self.sldprt_src = os.path.join(self.sample_dir, "sample_solidworks.sldprt")

    def test_arbitrary_step_filename_1(self):
        """Test loading STEP file with arbitrary non-standard name: test_element_xyz.step"""
        with tempfile.TemporaryDirectory() as tmpdir:
            custom_step = os.path.join(tmpdir, "test_element_xyz.step")
            shutil.copy(self.spherical_src, custom_step)
            prof = CADSurfaceImporter.extract_optical_profile(custom_step)
            self.assertIsNotNone(prof)
            self.assertAlmostEqual(prof.curvature, 0.01, places=4)

    def test_arbitrary_step_filename_2_spaces(self):
        """Test loading STEP file with spaces and numbers: my_custom optics 123 99.stp"""
        with tempfile.TemporaryDirectory() as tmpdir:
            custom_step = os.path.join(tmpdir, "my_custom optics 123 99.stp")
            shutil.copy(self.aspheric_src, custom_step)
            prof = CADSurfaceImporter.extract_optical_profile(custom_step)
            self.assertIsNotNone(prof)
            self.assertEqual(prof.conic_constant, -1.0)

    def test_uppercase_step_extension(self):
        """Test loading STEP file with uppercase extension: UPPERCASE_LENS.STEP"""
        with tempfile.TemporaryDirectory() as tmpdir:
            custom_step = os.path.join(tmpdir, "UPPERCASE_LENS.STEP")
            shutil.copy(self.spherical_src, custom_step)
            prof = CADSurfaceImporter.extract_optical_profile(custom_step)
            self.assertIsNotNone(prof)
            self.assertAlmostEqual(prof.curvature, 0.01, places=4)

    def test_arbitrary_solidworks_filename_and_params(self):
        """Test SolidWorks profile extraction with custom filename and parameter string"""
        with tempfile.TemporaryDirectory() as tmpdir:
            custom_sw = os.path.join(tmpdir, "weird_optics_component.sldprt")
            with open(custom_sw, 'w', encoding='utf-8') as f:
                f.write("SolidWorks Part Header\nPARAM R1=120.0 R2=-120.0 Th=12.5\nEND")
            front_sw, back_sw, th_sw = STEPParser.extract_solidworks_profiles(custom_sw)
            self.assertAlmostEqual(front_sw.radius_of_curvature, 120.0)
            self.assertAlmostEqual(th_sw, 12.5)

    def test_non_existent_file_raises_error(self):
        """Test that non-existent path raises FileNotFoundError"""
        with self.assertRaises(FileNotFoundError):
            CADSurfaceImporter.extract_optical_profile("non_existent_file_path_12345.step")

    def test_zero_byte_step_file_raises_parse_error(self):
        """Test that zero-byte STEP file raises CADParseError"""
        with tempfile.TemporaryDirectory() as tmpdir:
            empty_file = os.path.join(tmpdir, "empty.step")
            open(empty_file, 'w').close()
            with self.assertRaises(CADParseError):
                CADSurfaceImporter.extract_optical_profile(empty_file)

    def test_corrupt_step_file_raises_error(self):
        """Test that corrupt STEP file with no optical entities raises CADGeometryError or CADParseError"""
        with tempfile.TemporaryDirectory() as tmpdir:
            corrupt_file = os.path.join(tmpdir, "corrupt.step")
            with open(corrupt_file, 'w') as f:
                f.write("ISO-10303-21;\nHEADER;\nENDSEC;\nDATA;\nNOT_AN_ENTITY;\nENDSEC;\nEND-ISO-10303-21;\n")
            with self.assertRaises((CADParseError, CADGeometryError)):
                CADSurfaceImporter.extract_optical_profile(corrupt_file)

    def test_zslicer_invalid_inputs(self):
        """Test ZSlicer strict error handling for None and invalid types"""
        slicer = ZSlicer()
        with self.assertRaises(ValueError):
            slicer.slice_cad_mesh(None)
        with self.assertRaises(TypeError):
            slicer.slice_cad_mesh("invalid_string_input")

    def test_edge_case_flat_surface(self):
        """Test OpticalProfile with zero curvature (flat surface)"""
        flat_prof = OpticalProfile(surface_name="Flat Surface", curvature=0.0, conic_constant=0.0, semi_diameter=25.0)
        self.assertEqual(flat_prof.sag(15.0), 0.0)
        self.assertEqual(flat_prof.normal_vector(10.0, 10.0, 0.0), (0.0, 0.0, 1.0))
        t, P, n, is_valid = flat_prof.intersect_ray((5.0, 5.0, -10.0), (0.0, 0.0, 1.0))
        self.assertTrue(is_valid)
        self.assertAlmostEqual(P[2], 0.0)

    def test_edge_case_concave_surface(self):
        """Test OpticalProfile with negative curvature (concave surface)"""
        concave_prof = OpticalProfile(surface_name="Concave Surface", curvature=-0.02, conic_constant=0.0, semi_diameter=20.0)
        sag = concave_prof.sag(10.0)
        self.assertLess(sag, 0.0)
        t, P, n, is_valid = concave_prof.intersect_ray((10.0, 0.0, -10.0), (0.0, 0.0, 1.0))
        self.assertTrue(is_valid)
        self.assertAlmostEqual(P[2], sag, places=4)

    def test_edge_case_hyperboloid(self):
        """Test OpticalProfile with conic k = -2.0 (hyperboloid)"""
        hyp_prof = OpticalProfile(surface_name="Hyperboloid", curvature=0.02, conic_constant=-2.0, semi_diameter=20.0)
        sag = hyp_prof.sag(10.0)
        self.assertFalse(np.isnan(sag))
        t, P, n, is_valid = hyp_prof.intersect_ray((10.0, 0.0, -10.0), (0.0, 0.0, 1.0))
        self.assertTrue(is_valid)

    def test_edge_case_oblate_spheroid(self):
        """Test OpticalProfile with conic k = +2.0 (oblate spheroid)"""
        oblate_prof = OpticalProfile(surface_name="Oblate Spheroid", curvature=0.02, conic_constant=2.0, semi_diameter=20.0)
        sag = oblate_prof.sag(15.0)
        self.assertFalse(np.isnan(sag))

    def test_ray_intersection_forward_guard(self):
        """Test that ray going backwards (t <= 0) returns is_valid=False"""
        flat_prof = OpticalProfile(surface_name="Flat Surface", curvature=0.0, conic_constant=0.0, semi_diameter=25.0)
        t, P, n, is_valid = flat_prof.intersect_ray((0.0, 0.0, 10.0), (0.0, 0.0, 1.0))
        self.assertFalse(is_valid)

    def test_ray_intersection_aperture_guard(self):
        """Test that ray hitting outside clear aperture returns is_valid=False"""
        bounded_prof = OpticalProfile(curvature=0.01, semi_diameter=20.0)
        t, P, n, is_valid = bounded_prof.intersect_ray((25.0, 0.0, -10.0), (0.0, 0.0, 1.0))
        self.assertFalse(is_valid)

    def test_high_order_aspheric_profile(self):
        """Test high-order aspheric terms [a4, a6, a8] sag and ray intersection"""
        asph_prof = OpticalProfile(
            curvature=0.01,
            conic_constant=-1.0,
            aspheric_coeffs=[1e-5, -1e-8, 1e-11],
            semi_diameter=20.0
        )
        sag = asph_prof.sag(10.0)
        t, P, n, is_valid = asph_prof.intersect_ray((5.0, 5.0, -10.0), (0.0, 0.0, 1.0))
        self.assertTrue(is_valid)


if __name__ == "__main__":
    unittest.main()
