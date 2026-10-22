"""
Unit tests for Zemax OpticStudio .zmx File Import/Export and Lens Insertion/Replacement Tools.
"""

import unittest
import numpy as np
import math
import tempfile
import os

from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from optics_app.io.zmx_parser import parse_zmx_string, parse_zmx_file, export_to_zmx_string, export_to_zmx_file
from optics_app.gui.lde_table import LensDataTableModel


class TestZemaxImportExportAndLensTools(unittest.TestCase):
    """Test Zemax .zmx import/export and lens addition/replacement capabilities."""

    def test_zemax_zmx_import_realistic(self):
        """Verify parsing of a realistic Zemax .zmx prescription."""
        zmx_text = """
! Zemax OpticStudio Prescription
VERS 140410 1 0
NAME "Thorlabs Doublet Collimator"
UNIT MM
WAVL 0.785000 1.0000
WAVL 0.830000 0.5000
WAVL 0.880000 0.5000
SURF 0
  COMM "Fiber Source"
  TYPE STANDARD
  CURV 0.0000000000
  THIC 37.800000
  GLAS AIR
  DIAM 12.700000
SURF 1
  COMM "AC254-040-B Front"
  TYPE STANDARD
  CURV 0.0398406375
  THIC 6.800000
  GLAS N-BK7
  DIAM 12.700000
SURF 2
  COMM "AC254-040-B Cemented"
  TYPE STANDARD
  CURV -0.0495049505
  THIC 2.000000
  GLAS SF5
  DIAM 12.700000
SURF 3
  COMM "AC254-040-B Back"
  TYPE STANDARD
  CURV -0.0119047619
  THIC 50.000000
  GLAS AIR
  DIAM 12.700000
SURF 4
  COMM "Detector"
  TYPE DETECTOR
  CURV 0.0000000000
  THIC 0.000000
  GLAS AIR
  DIAM 12.700000
"""
        stack = parse_zmx_string(zmx_text)
        self.assertEqual(len(stack.surfaces), 5)
        self.assertEqual(stack.surfaces[0].name, "Fiber Source")
        self.assertAlmostEqual(stack.surfaces[0].thickness, 37.8, places=3)
        self.assertEqual(stack.surfaces[1].glass, "N-BK7")
        self.assertAlmostEqual(stack.surfaces[1].radius, 25.1, places=1)
        self.assertAlmostEqual(stack.surfaces[1].thickness, 6.8, places=3)
        self.assertEqual(stack.surfaces[2].glass, "SF5")
        self.assertAlmostEqual(stack.surfaces[2].radius, -20.2, places=1)
        self.assertEqual(stack.surfaces[4].surface_type, SurfaceType.DETECTOR)

        # Check ray tracing through imported system
        opt_sys = OpticalSystem(stack)
        r = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=0.785)
        tr = opt_sys.trace_ray(r)
        self.assertTrue(tr is not None)
        self.assertEqual(len(tr.history), 5)

    def test_zemax_roundtrip_export_import(self):
        """Verify that export_to_zmx_string and parse_zmx_string preserve prescription exactly."""
        orig_stack = SurfaceStack()
        orig_stack.add_surface(Surface(name="OBJ", thickness=40.0, glass="AIR"))
        orig_stack.add_surface(Surface(name="L1_Front", radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
        orig_stack.add_surface(Surface(name="L1_Back", radius=-115.0, thickness=45.0, glass="AIR", semi_diameter=12.7))
        orig_stack.add_surface(Surface(name="IMA", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=10.0))

        zmx_exported = export_to_zmx_string(orig_stack, title="Roundtrip Test", wavelengths=[0.5876, 0.785])
        self.assertIn("SURF 0", zmx_exported)
        self.assertIn("L1_Front", zmx_exported)
        self.assertIn("N-BK7", zmx_exported)

        # Re-import
        reimported_stack = parse_zmx_string(zmx_exported)
        self.assertEqual(len(reimported_stack.surfaces), 4)
        self.assertAlmostEqual(reimported_stack.surfaces[1].radius, 34.0, places=3)
        self.assertAlmostEqual(reimported_stack.surfaces[1].thickness, 6.0, places=3)
        self.assertEqual(reimported_stack.surfaces[1].glass, "N-BK7")

    def test_insert_surface_after(self):
        """Verify inserting a new surface after a specific surface index."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="Source", thickness=50.0))
        stack.add_surface(Surface(name="Detector", surface_type=SurfaceType.DETECTOR, thickness=0.0))

        # Insert aperture stop between source and detector (after surface 0)
        new_stop = Surface(name="Aperture_Stop", surface_type=SurfaceType.STOP, thickness=25.0, semi_diameter=8.0)
        stack.insert_surface(1, new_stop)

        self.assertEqual(len(stack.surfaces), 3)
        self.assertEqual(stack.surfaces[1].name, "Aperture_Stop")
        self.assertEqual(stack.surfaces[2].name, "Detector")

    def test_insert_singlet_and_doublet_after(self):
        """Verify inserting single and doublet lenses after a specific surface."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="Source", thickness=50.0))
        stack.add_surface(Surface(name="Detector", surface_type=SurfaceType.DETECTOR, thickness=0.0))

        # Insert singlet after surface 0
        indices = stack.insert_singlet(after_surface_index=0, name="Focus_Lens", radius_front=30.0, thickness=4.0, radius_back=-40.0)
        self.assertEqual(indices, [1, 2])
        self.assertEqual(len(stack.surfaces), 4)
        self.assertEqual(stack.surfaces[1].name, "Focus_Lens_Front")
        self.assertEqual(stack.surfaces[2].name, "Focus_Lens_Back")

        # Insert doublet after surface 2
        d_indices = stack.insert_doublet(after_surface_index=2, name="Achromat", r1=34.0, t1=6.0, glass1="N-BK7", r2=-27.8, t2=2.0, glass2="SF5", r3=-115.0)
        self.assertEqual(d_indices, [3, 4, 5])
        self.assertEqual(len(stack.surfaces), 7)
        self.assertEqual(stack.surfaces[3].name, "Achromat_Front")
        self.assertEqual(stack.surfaces[4].name, "Achromat_Mid")
        self.assertEqual(stack.surfaces[5].name, "Achromat_Back")

    def test_insert_and_replace_catalog_lens(self):
        """Verify inserting and replacing commercial Thorlabs catalog lenses."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="Source", thickness=30.0))
        stack.add_surface(Surface(name="Detector", surface_type=SurfaceType.DETECTOR, thickness=0.0))

        # Insert AC254-050-B after surface 0
        ins_indices = stack.insert_catalog_lens(after_surface_index=0, part_number="AC254-050-B", air_gap_after=45.0)
        self.assertEqual(len(stack.surfaces), 5)
        self.assertEqual(stack.surfaces[1].glass, "N-LAK22")
        self.assertAlmostEqual(stack.surfaces[1].radius, 33.55, places=2)
        self.assertEqual(stack.surfaces[2].glass, "N-SF6HT")
        self.assertAlmostEqual(stack.surfaces[2].radius, -27.05, places=2)
        self.assertAlmostEqual(stack.surfaces[3].thickness, 45.0, places=2)

        # Replace AC254-050-B (at surface 1) with AC254-040-B
        rep_indices = stack.replace_with_catalog_lens(surface_index=1, part_number="AC254-040-B")
        self.assertEqual(len(stack.surfaces), 5)
        self.assertEqual(stack.surfaces[1].glass, "N-BAF10")
        self.assertAlmostEqual(stack.surfaces[1].radius, 26.12, places=2)
        self.assertEqual(stack.surfaces[2].glass, "N-SF6HT")
        self.assertAlmostEqual(stack.surfaces[2].radius, -21.28, places=2)
        # Preserves downstream air gap
        self.assertAlmostEqual(stack.surfaces[3].thickness, 45.0, places=2)

    def test_table_model_lens_insertion_and_replacement(self):
        """Verify that LensDataTableModel table methods insert and replace lenses seamlessly."""
        model = LensDataTableModel()
        init_rows = model.rowCount()
        
        # Insert catalog lens after row 1
        model.insert_catalog_lens_after(row=1, part_number="AC254-075-B")
        self.assertEqual(model.rowCount(), init_rows + 3)

        # Replace with AC254-100-B
        model.replace_with_catalog_lens(row=2, part_number="AC254-100-B")
        self.assertEqual(model.rowCount(), init_rows + 3)
        self.assertAlmostEqual(model.data(2, 2), 66.68, places=2)

    def test_zemax_mirror_and_coordinate_break_roundtrip(self):
        """Verify Zemax standard export and import for coordinate breaks, flat mirrors, and aspheric mirrors."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=50.0))
        # Coordinate Break
        stack.add_surface(Surface(name="FoldCB", surface_type=SurfaceType.COORDINATE_BREAK, decenter_x=3.5, tilt_y=45.0, thickness=0.0))
        # Flat mirror
        stack.add_surface(Surface(name="FlatMirror", surface_type=SurfaceType.MIRROR, thickness=-40.0))
        # Aspheric mirror
        stack.add_surface(Surface(name="ParabolicPrimary", surface_type=SurfaceType.ASPHERIC, radius=-200.0, conic=-1.0, glass="MIRROR", thickness=80.0))
        # Stop
        stack.add_surface(Surface(name="StopSurface", is_stop=True, radius=0.0, thickness=20.0, glass="AIR"))
        # Image
        stack.add_surface(Surface(name="Detector", surface_type=SurfaceType.DETECTOR, thickness=0.0))

        zmx_str = export_to_zmx_string(stack, title="Zemax Mirror & CB Test")

        # Verify export tokens follow standard Zemax OpticStudio definitions
        self.assertIn("TYPE COORDBRK", zmx_str)
        self.assertIn("PARM 1 3.500000", zmx_str)
        self.assertIn("PARM 4 45.000000", zmx_str)
        self.assertIn("TYPE STANDARD", zmx_str)
        self.assertIn("GLAS MIRROR", zmx_str)
        self.assertIn("TYPE EVENASPH", zmx_str)
        self.assertIn("CONI -1.000000", zmx_str)
        self.assertIn("STOP", zmx_str)

        # Re-import and check preservation
        reimported = parse_zmx_string(zmx_str)
        self.assertEqual(len(reimported.surfaces), 6)

        cb_surf = reimported.surfaces[1]
        self.assertTrue(cb_surf.is_coord_break)
        self.assertAlmostEqual(cb_surf.decenter_x, 3.5, places=3)
        self.assertAlmostEqual(cb_surf.tilt_y, 45.0, places=3)

        flat_mirror = reimported.surfaces[2]
        self.assertEqual(flat_mirror.glass, "MIRROR")
        self.assertTrue(flat_mirror.is_reflective())

        asph_mirror = reimported.surfaces[3]
        self.assertEqual(asph_mirror.surface_type, SurfaceType.ASPHERIC)
        self.assertEqual(asph_mirror.glass, "MIRROR")
        self.assertTrue(asph_mirror.is_reflective())
        self.assertAlmostEqual(asph_mirror.radius, -200.0, places=3)
        self.assertAlmostEqual(asph_mirror.conic, -1.0, places=3)

        stop_surf = reimported.surfaces[4]
        self.assertTrue(stop_surf.is_stop)

    def test_table_model_mirror_and_glass_editing(self):
        """Verify that LDE table model synchronizes Mirror type, MIRROR glass, and keeps Glass column editable."""
        model = LensDataTableModel()
        
        # Row 1 is a Standard surface by default with glass "N-BK7" or similar
        # Set Surface Type (Col 1) to "Mirror"
        model.setData(1, 1, "Mirror")
        self.assertEqual(model.data(1, 1), "Mirror")
        self.assertEqual(model.data(1, 4), "MIRROR")
        self.assertTrue(model._surfaces[1].is_reflective())

        # Test that Glass cell (Col 4) is not locked and can be edited back to refractive glass
        model.setData(1, 4, "N-SF11")
        self.assertEqual(model.data(1, 4), "N-SF11")
        self.assertEqual(model.data(1, 1), "Standard")
        self.assertFalse(model._surfaces[1].is_reflective())

        # Set Glass cell to "MIRROR" directly
        model.setData(1, 4, "MIRROR")
        self.assertEqual(model.data(1, 4), "MIRROR")
        self.assertEqual(model.data(1, 1), "Mirror")
        self.assertTrue(model._surfaces[1].is_reflective())

    def test_surface_material_formatting_standard_definitions(self):
        """Verify format_surface_material_display adheres strictly to Zemax OpticStudio naming."""
        from optics_app.engine.element_colors import format_surface_material_display, get_surface_role_description

        s_mirror = Surface(name="Fold Mirror", surface_type=SurfaceType.MIRROR)
        self.assertEqual(format_surface_material_display(s_mirror), "MIRROR")
        self.assertEqual(get_surface_role_description(s_mirror), "Mirror Substrate (Reflector)")

        s_glass = Surface(name="Lens 1 Front", glass="N-BK7")
        self.assertEqual(format_surface_material_display(s_glass), "N-BK7")

        s_air = Surface(name="Air Gap", glass="AIR")
        self.assertEqual(format_surface_material_display(s_air), "AIR")


if __name__ == "__main__":
    unittest.main()
