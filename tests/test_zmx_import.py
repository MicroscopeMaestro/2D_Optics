"""
Unit tests for Zemax .zmx Prescription File Importer (Feature 19).
Covers Tiers 1-4 per TEST_INFRA.md.
"""
import unittest
import os
import tempfile
from optics_app.io.zmx_parser import parse_zmx_string, parse_zmx_file
from optics_app.engine.surface import SurfaceType
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray

class TestZmxImportTier1(unittest.TestCase):
    """Tier 1: Basic .zmx Prescription Import & Surface Stack Population."""

    def test_t1_1_parse_singlet_zmx_string(self):
        zmx_data = """
        ! Sample Zemax Prescription
        SURF 0
          TYPE STANDARD
          THIC 100.0
          GLAS AIR
        SURF 1
          TYPE STANDARD
          CURV 0.02
          THIC 5.0
          GLAS N-BK7
          DIAM 25.0
        SURF 2
          TYPE STANDARD
          CURV -0.02
          THIC 45.0
          GLAS AIR
          DIAM 25.0
        SURF 3
          TYPE DETECTOR
          THIC 0.0
        """
        stack = parse_zmx_string(zmx_data)
        self.assertEqual(len(stack.surfaces), 4)
        self.assertAlmostEqual(stack.surfaces[1].radius, 50.0)
        self.assertEqual(stack.surfaces[1].glass, "N-BK7")
        self.assertAlmostEqual(stack.get_total_system_length(), 150.0)

    def test_t1_2_parse_conic_and_aspheric_params(self):
        zmx_data = """
        SURF 1
          TYPE EVENASPH
          CURV 0.02
          CONI -1.0
          PARM 1 1.5e-6
          PARM 2 -3.2e-9
        """
        stack = parse_zmx_string(zmx_data)
        surf = stack.surfaces[0]
        self.assertEqual(surf.surface_type, SurfaceType.ASPHERIC)
        self.assertEqual(surf.conic, -1.0)
        self.assertEqual(len(surf.asphere_coeffs), 2)
        self.assertAlmostEqual(surf.asphere_coeffs[0], 1.5e-6)
        self.assertAlmostEqual(surf.asphere_coeffs[1], -3.2e-9)

    def test_t1_3_parse_stop_surface_identification(self):
        zmx_data = """
        SURF 1
          STOP
          DIAM 10.0
        """
        stack = parse_zmx_string(zmx_data)
        self.assertEqual(stack.surfaces[0].surface_type, SurfaceType.STOP)
        self.assertEqual(stack.surfaces[0].semi_diameter, 10.0)


class TestZmxImportTier2(unittest.TestCase):
    """Tier 2: Boundary Value Analysis, File Handling & Encoding Support."""

    def test_t2_1_empty_zmx_string(self):
        stack = parse_zmx_string("")
        self.assertEqual(len(stack.surfaces), 0)

    def test_t2_2_zmx_file_reading_tempfile(self):
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.zmx', encoding='utf-8') as tmp:
            tmp.write("SURF 1\nCURV 0.01\nTHIC 10.0\nGLAS N-BK7\n")
            tmp_path = tmp.name
        try:
            stack = parse_zmx_file(tmp_path)
            self.assertEqual(len(stack.surfaces), 1)
            self.assertAlmostEqual(stack.surfaces[0].radius, 100.0)
            self.assertEqual(stack.surfaces[0].glass, "N-BK7")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_t2_3_utf16_encoded_zmx_file(self):
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.zmx', encoding='utf-16') as tmp:
            tmp.write("SURF 0\nTHIC 100.0\nSURF 1\nCURV 0.05\nTHIC 5.0\n")
            tmp_path = tmp.name
        try:
            stack = parse_zmx_file(tmp_path)
            self.assertEqual(len(stack.surfaces), 2)
            self.assertAlmostEqual(stack.surfaces[1].radius, 20.0)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_t2_4_infinite_thickness_parsing(self):
        zmx_data = """
        SURF 0
          THIC INFINITY
        """
        stack = parse_zmx_string(zmx_data)
        self.assertEqual(stack.surfaces[0].thickness, 1e10)


class TestZmxImportTier3(unittest.TestCase):
    """Tier 3: Multi-Element Prescription & Grating Surface Parsing."""

    def test_t3_1_grating_surface_parsing(self):
        zmx_data = """
        SURF 1
          TYPE GRATING
          THIC 10.0
          PARM 1 600.0
        """
        stack = parse_zmx_string(zmx_data)
        surf = stack.surfaces[0]
        self.assertEqual(surf.surface_type, SurfaceType.GRATING)
        self.assertEqual(surf.grating_lines_per_mm, 600.0)


class TestZmxImportTier4(unittest.TestCase):
    """Tier 4: End-to-End System Integration with Ray Tracer."""

    def test_t4_1_imported_prescription_ray_trace(self):
        zmx_data = """
        SURF 0
          TYPE STANDARD
          THIC 100.0
          GLAS AIR
        SURF 1
          TYPE STANDARD
          CURV 0.02
          THIC 5.0
          GLAS N-BK7
          DIAM 25.0
        SURF 2
          TYPE STANDARD
          CURV -0.02
          THIC 45.0
          GLAS AIR
          DIAM 25.0
        SURF 3
          TYPE DETECTOR
          THIC 0.0
        """
        stack = parse_zmx_string(zmx_data)
        opt_sys = OpticalSystem(stack)
        ray = Ray(pos=(0.0, 0.0, 0.0), dir=(0.0, 0.0, 1.0), wavelength=0.5876)
        traced = opt_sys.trace_ray(ray)
        self.assertFalse(traced.vignetted)
        self.assertEqual(len(traced.history), 4)
        self.assertAlmostEqual(traced.history[-1]["z"], 150.0)


if __name__ == "__main__":
    unittest.main()
