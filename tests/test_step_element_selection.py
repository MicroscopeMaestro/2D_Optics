"""
Unit tests for Selective 3D STEP Assembly Export and Element Selection Dialog.
Verifies get_exportable_elements, STEPExportDialog, selective STEP export filtering,
and 3D visualizer export integration.
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.cad.exporter import (
    get_exportable_elements,
    STEPExporter,
    CADExporter,
    STEPExportEngine,
    STEPExportOptions,
    export_to_step,
)
from optics_app.gui.step_export_dialog import STEPExportDialog
from optics_app.gui.visualizer_3d import Visualizer3DWidget


def create_sample_spectrometer_stack() -> SurfaceStack:
    """Build a representative multi-component optical system."""
    stack = SurfaceStack()
    # 0: Object / slit
    stack.add_surface(Surface(name="Input_Slit", surface_type=SurfaceType.OBJECT, thickness=50.0, semi_diameter=5.0))
    # 1-2: Collimator Singlet (N-BK7)
    stack.add_surface(Surface(name="Coll_Front", radius=60.0, thickness=6.0, glass="N-BK7", semi_diameter=25.0))
    stack.add_surface(Surface(name="Coll_Back", radius=-60.0, thickness=30.0, glass="AIR", semi_diameter=25.0))
    # 3: Aperture Stop
    stack.add_surface(Surface(name="Aperture_Stop", surface_type=SurfaceType.STOP, thickness=20.0, semi_diameter=12.5))
    # 4: Fold Mirror
    stack.add_surface(Surface(name="Fold_Mirror", surface_type=SurfaceType.MIRROR, thickness=-25.0, semi_diameter=20.0))
    # 5: Diffraction Grating
    stack.add_surface(Surface(name="Grating_600lp", surface_type=SurfaceType.GRATING, grating_lines_per_mm=600.0, thickness=30.0, semi_diameter=20.0))
    # 6-8: Achromatic Doublet (N-LAK22 / SF6)
    stack.add_surface(Surface(name="Dub_Crown", radius=50.0, thickness=7.0, glass="N-LAK22", semi_diameter=22.0))
    stack.add_surface(Surface(name="Dub_Interface", radius=-40.0, thickness=4.0, glass="SF6", semi_diameter=22.0))
    stack.add_surface(Surface(name="Dub_Flint", radius=-120.0, thickness=45.0, glass="AIR", semi_diameter=22.0))
    # 9: CCD Detector
    stack.add_surface(Surface(name="CCD_Sensor", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=16.0))
    return stack


class TestStepElementSelection(unittest.TestCase):
    """Test suite for element discovery, selective CAD assembly export, and selection dialog."""

    def test_get_exportable_elements_system_breakdown(self):
        """Verify get_exportable_elements correctly partitions surfaces into physical optics components."""
        stack = create_sample_spectrometer_stack()
        elements = get_exportable_elements(stack)

        self.assertGreater(len(elements), 0)

        types_found = [el["type"] for el in elements]
        self.assertIn("Sample Stage", types_found)
        self.assertIn("Singlet", types_found)
        self.assertIn("Aperture Stop", types_found)
        self.assertIn("Mirror", types_found)
        self.assertIn("Grating", types_found)
        self.assertIn("Doublet", types_found)
        self.assertIn("Detector", types_found)

        # Check Doublet surface indices cover Crown, Interface, and Rear
        doublet = next(el for el in elements if el["type"] == "Doublet")
        self.assertEqual(len(doublet["surface_indices"]), 3)
        self.assertEqual(doublet["material"], "N-LAK22 / SF6")
        self.assertAlmostEqual(doublet["semi_diameter"], 22.0)
        self.assertAlmostEqual(doublet["clear_aperture_diam"], 44.0)

        # Check Singlet covers Front and Back
        singlet = next(el for el in elements if el["type"] == "Singlet")
        self.assertEqual(len(singlet["surface_indices"]), 2)
        self.assertEqual(singlet["material"], "N-BK7")
        self.assertAlmostEqual(singlet["semi_diameter"], 25.0)

    def test_selective_step_export_pure_python(self):
        """Verify selective export in Pure Python mode outputs only chosen components."""
        stack = create_sample_spectrometer_stack()
        elements = get_exportable_elements(stack)

        # Identify indices for Singlet and Doublet only
        lens_indices = [el["index"] for el in elements if el["type"] in ["Singlet", "Doublet"]]
        self.assertGreaterEqual(len(lens_indices), 2)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = os.path.join(tmpdir, "lenses_only_pure.step")
            res = STEPExporter.export(
                stack,
                out_file,
                engine=STEPExportEngine.PURE_PYTHON,
                selected_element_indices=lens_indices
            )

            self.assertTrue(res.is_success)
            self.assertTrue(os.path.exists(out_file))
            self.assertEqual(res.num_components_exported, len(lens_indices))

            with open(out_file, "r", encoding="utf-8") as f:
                content = f.read()

            self.assertIn("ISO-10303-21;", content)
            self.assertIn("END-ISO-10303-21;", content)
            self.assertIn("Coll_Front", content)
            self.assertIn("Dub_Crown", content)
            # Detector and Mirror were not selected
            self.assertNotIn("CCD_Sensor", content)
            self.assertNotIn("Fold_Mirror", content)

    def test_selective_step_export_build123d(self):
        """Verify selective export with build123d engine outputs exact component count."""
        stack = create_sample_spectrometer_stack()
        elements = get_exportable_elements(stack)

        # Pick only Mirror and Aperture Stop
        chosen = [el["index"] for el in elements if el["type"] in ["Mirror", "Aperture Stop"]]
        self.assertEqual(len(chosen), 2)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = os.path.join(tmpdir, "mirror_and_stop_b3d.step")
            res = STEPExporter.export(
                stack,
                out_file,
                engine=STEPExportEngine.BUILD123D,
                selected_element_indices=chosen
            )

            self.assertTrue(res.is_success)
            self.assertTrue(os.path.exists(out_file))
            self.assertEqual(res.num_components_exported, 2)
            self.assertGreater(os.path.getsize(out_file), 100)

    def test_selective_step_export_empty_selection(self):
        """Verify passing an empty selection list generates a valid empty/header STEP file without crashing."""
        stack = create_sample_spectrometer_stack()

        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = os.path.join(tmpdir, "empty_sel.step")
            res = export_to_step(
                stack,
                out_file,
                engine="pure_python",
                selected_element_indices=[]
            )

            self.assertTrue(res.is_success)
            self.assertEqual(res.num_components_exported, 0)
            self.assertTrue(os.path.exists(out_file))

    def test_step_export_dialog_batch_and_filtering(self):
        """Verify STEPExportDialog supports selection toggles, batch filtering, and accessors."""
        stack = create_sample_spectrometer_stack()
        dialog = STEPExportDialog(stack, default_file_path="custom_optical_assembly.step")

        # Initial default: physical components selected, air volumes excluded
        initial_sel = dialog.get_selected_element_indices()
        self.assertGreater(len(initial_sel), 0)

        # 1. Deselect All
        dialog.deselect_all()
        self.assertEqual(len(dialog.get_selected_element_indices()), 0)
        self.assertEqual(len(dialog.get_selected_elements()), 0)
        self.assertEqual(len(dialog.get_selected_surface_indices()), 0)

        # 2. Select All
        dialog.select_all()
        self.assertEqual(len(dialog.get_selected_element_indices()), len(dialog.elements))

        # 3. Lenses Only
        dialog.select_by_type(["Singlet", "Doublet", "Lens"])
        sel_lenses = dialog.get_selected_elements()
        for el in sel_lenses:
            self.assertIn(el["type"], ["Singlet", "Doublet", "Lens"])

        # 4. Invert Selection
        lens_count = len(dialog.get_selected_element_indices())
        dialog.invert_selection()
        inverted_count = len(dialog.get_selected_element_indices())
        self.assertEqual(lens_count + inverted_count, len(dialog.elements))

        # 5. File path and engine getters
        self.assertEqual(dialog.get_file_path(), "custom_optical_assembly.step")
        self.assertIn(dialog.get_engine(), ["auto", "build123d", "pure_python"])

    def test_visualizer_3d_integration(self):
        """Verify 3D Visualizer contains btn_export_step button and export_step_dialog handler."""
        stack = create_sample_spectrometer_stack()
        vis = Visualizer3DWidget()
        vis.surface_stack = stack

        self.assertTrue(hasattr(vis, "btn_export_step"))
        self.assertTrue(hasattr(vis, "export_step_dialog"))
        self.assertTrue(callable(vis.export_step_dialog))


if __name__ == "__main__":
    unittest.main()
