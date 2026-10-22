"""
Unit tests for 3D STEP CAD Assembly Exporter (Feature 20).
Covers Tiers 1-4 per TEST_INFRA.md.
"""
import unittest
import os
import sys
import tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack, LensDataEditor
from optics_app.io.zmx_parser import parse_zmx_string
from optics_app.cad.exporter import STEPExporter, STEPExportOptions, STEPExportEngine, export_to_step

class TestStepExportTier1(unittest.TestCase):
    """Tier 1: Basic 3D STEP Assembly Export Functionality."""

    def test_t1_1_export_single_spherical_lens(self):
        stack = SurfaceStack()
        stack.add_surface(Surface(name="S1", radius=50.0, thickness=5.0, glass="N-BK7", semi_diameter=25.0))
        stack.add_surface(Surface(name="S2", radius=-50.0, thickness=45.0, glass="AIR", semi_diameter=25.0))

        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = os.path.join(tmpdir, "singlet.step")
            res = STEPExporter.export(stack, out_file)
            self.assertTrue(res.is_success)
            self.assertTrue(os.path.exists(out_file))
            self.assertGreater(res.file_size_bytes, 0)
            self.assertGreaterEqual(res.num_components_exported, 1)

    def test_t1_2_export_aspheric_lens(self):
        stack = SurfaceStack()
        stack.add_surface(Surface(name="Asph1", surface_type=SurfaceType.ASPHERIC, radius=50.0, conic=-1.0, asphere_coeffs=[1e-6, -2e-9], thickness=6.0, glass="N-BK7"))
        stack.add_surface(Surface(name="S2", radius=-100.0, thickness=50.0, glass="AIR"))

        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = os.path.join(tmpdir, "asphere.step")
            res = export_to_step(stack, out_file)
            self.assertTrue(res.is_success)
            with open(out_file, 'r', encoding='utf-8') as f:
                content = f.read()
            self.assertIn("ISO-10303-21;", content)
            self.assertIn("HEADER;", content)
            self.assertIn("DATA;", content)

    def test_t1_3_export_grating_and_ccd(self):
        stack = SurfaceStack()
        stack.add_surface(Surface(name="Grating", surface_type=SurfaceType.GRATING, grating_lines_per_mm=600.0, thickness=20.0, semi_diameter=15.0))
        stack.add_surface(Surface(name="CCD", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=10.0))

        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = os.path.join(tmpdir, "grating_ccd.step")
            res = export_to_step(stack, out_file)
            self.assertTrue(res.is_success)
            self.assertEqual(res.num_components_exported, 2)


class TestStepExportTier2(unittest.TestCase):
    """Tier 2: Boundary Value Analysis & Edge Cases for STEP Exporter."""

    def test_t2_1_export_flat_window_element(self):
        stack = SurfaceStack()
        stack.add_surface(Surface(name="FlatIn", radius=0.0, thickness=3.0, glass="FUSED_SILICA"))
        stack.add_surface(Surface(name="FlatOut", radius=0.0, thickness=10.0, glass="AIR"))

        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = os.path.join(tmpdir, "window.step")
            res = export_to_step(stack, out_file)
            self.assertTrue(res.is_success)

    def test_t2_2_export_extreme_decenter_and_tilt(self):
        stack = SurfaceStack()
        stack.add_surface(Surface(name="TiltedLens", radius=40.0, thickness=5.0, glass="N-BK7", decenter_x=5.0, decenter_y=-2.0, tilt_x=15.0, tilt_y=5.0))
        stack.add_surface(Surface(name="Back", radius=-40.0, thickness=30.0, glass="AIR"))

        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = os.path.join(tmpdir, "tilted.step")
            res = export_to_step(stack, out_file)
            self.assertTrue(res.is_success)

    def test_t2_3_export_invalid_path_raises_error(self):
        stack = SurfaceStack()
        stack.add_surface(Surface(name="S1", radius=50.0, thickness=5.0, glass="N-BK7"))
        stack.add_surface(Surface(name="S2", radius=-50.0, thickness=10.0))

        # Attempt to export to invalid non-existent path on Windows
        invalid_path = "Z:\\NonExistentFolder\\TargetDir\\test.step"
        res = export_to_step(stack, invalid_path)
        self.assertFalse(res.is_success)

    def test_t2_4_export_empty_surface_stack(self):
        stack = SurfaceStack()
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = os.path.join(tmpdir, "empty.step")
            res = export_to_step(stack, out_file)
            self.assertTrue(res.is_success)
            self.assertEqual(res.num_components_exported, 0)


class TestStepExportTier3(unittest.TestCase):
    """Tier 3: Pairwise & System Integration Tests."""

    def test_t3_1_export_helper_function(self):
        lde = LensDataEditor()
        lde.add_surface(Surface(name="Front", radius=60.0, thickness=8.0, glass="N-BK7"))
        lde.add_surface(Surface(name="Back", radius=-60.0, thickness=50.0, glass="AIR"))
        lde.add_surface(Surface(name="Detector", surface_type=SurfaceType.DETECTOR))

        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = os.path.join(tmpdir, "lde_export.step")
            res = export_to_step(lde, out_file, engine="pure_python")
            self.assertTrue(res.is_success)
            self.assertEqual(res.engine_used, STEPExportEngine.PURE_PYTHON)

    def test_t3_2_export_assembly_step_alias(self):
        from optics_app.cad.exporter import CADExporter
        stack = SurfaceStack()
        stack.add_surface(Surface(name="S1", radius=50.0, thickness=5.0, glass="N-BK7", semi_diameter=25.0))
        stack.add_surface(Surface(name="S2", radius=-50.0, thickness=45.0, glass="AIR", semi_diameter=25.0))

        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = os.path.join(tmpdir, "alias_test.step")
            result_path = CADExporter.export_assembly_step(stack, out_file)
            self.assertEqual(result_path, out_file)
            self.assertTrue(os.path.exists(out_file))

            out_file_step = os.path.join(tmpdir, "alias_step_test.step")
            res_step_path = STEPExporter.export_assembly_step(stack, out_file_step)
            self.assertEqual(res_step_path, out_file_step)
            self.assertTrue(os.path.exists(out_file_step))


class TestStepExportTier4(unittest.TestCase):
    """Tier 4: End-to-End ZMX Import to STEP Export Workflows."""

    def test_t4_1_export_cooke_triplet_assembly(self):
        # 3 Lenses + Stop + Detector = 5 components
        stack = SurfaceStack()
        # Lens 1
        stack.add_surface(Surface(radius=40.0, thickness=6.0, glass="N-SK16"))
        stack.add_surface(Surface(radius=-140.0, thickness=10.0, glass="AIR"))
        # Stop
        stack.add_surface(Surface(surface_type=SurfaceType.STOP, thickness=5.0))
        # Lens 2
        stack.add_surface(Surface(radius=-35.0, thickness=3.0, glass="F2"))
        stack.add_surface(Surface(radius=35.0, thickness=10.0, glass="AIR"))
        # Lens 3
        stack.add_surface(Surface(radius=100.0, thickness=5.0, glass="N-SK16"))
        stack.add_surface(Surface(radius=-40.0, thickness=40.0, glass="AIR"))
        # Detector
        stack.add_surface(Surface(surface_type=SurfaceType.DETECTOR))

        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = os.path.join(tmpdir, "cooke_triplet.step")
            res = export_to_step(stack, out_file)
            self.assertTrue(res.is_success)
            self.assertGreaterEqual(res.num_components_exported, 4)

    def test_t4_2_zmx_import_to_step_export_pipeline(self):
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
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = os.path.join(tmpdir, "pipeline_zmx.step")
            res = export_to_step(stack, out_file)
            self.assertTrue(res.is_success)
            self.assertTrue(os.path.exists(out_file))


if __name__ == "__main__":
    unittest.main()
