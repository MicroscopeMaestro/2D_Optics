"""Mechanical optical routing tests for DMD and future stateful components."""

import os
import tempfile
import unittest

import numpy as np

from optics_app.ai.nemotron_client import parse_optical_command_heuristics
from optics_app.ai.tools import set_mechanical_state
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.diagnostics import ModelSanityValidator
from optics_app.engine.mechanisms import (
    MechanicalStateController,
    active_state_name,
    interaction_rotation,
    simulate_state_throughput,
    state_transmission,
)
from optics_app.engine.ray import Ray
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.undo_manager import UndoManager
from optics_app.gui.component_editor_dock import ComponentEditorDockWidget
from optics_app.gui.lde_table import LensDataTableModel
from optics_app.gui.qt_compat import QT_AVAILABLE, QtWidgets


if QT_AVAILABLE and hasattr(QtWidgets, "QApplication"):
    _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


ROOT = os.path.dirname(os.path.dirname(__file__))
SPECTROMETER = os.path.join(ROOT, "spectrometer_raman_50mm_100mm.json")


class TestDMDMechanicalRouting(unittest.TestCase):
    def test_dmd_on_reaches_detector_and_off_misses_detector(self):
        loaded = SurfaceStack.load_from_json(SPECTROMETER)
        stack = loaded["stack"]
        rays = [
            Ray(pos=np.zeros(3), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl)
            for wl in loaded["wavelengths"]
        ]

        results = simulate_state_throughput(stack, "DMD4500", rays)

        self.assertGreater(results["ON"]["detector_hits"], 0)
        self.assertEqual(results["OFF"]["detector_hits"], 0)
        self.assertEqual(results["ON"]["route"], "detector")
        self.assertEqual(results["OFF"]["route"], "beam dump")

    def test_dmd_state_rotates_micromirror_not_housing_or_downstream_frame(self):
        stack = SurfaceStack.load_from_json(SPECTROMETER)["stack"]
        index = next(i for i, s in enumerate(stack.surfaces) if "DMD" in s.name.upper())
        dmd = stack.surfaces[index]
        detector_origin = stack.surfaces[-1].global_origin.copy()
        housing_rotation = dmd.global_R.copy()

        MechanicalStateController(stack).set_state(index, "OFF")
        stack._update_accumulated_positions()

        self.assertTrue(np.allclose(dmd.global_R, housing_rotation))
        self.assertTrue(np.allclose(stack.surfaces[-1].global_origin, detector_origin))
        self.assertFalse(np.allclose(interaction_rotation(dmd), housing_rotation))

    def test_state_persists_in_project_json(self):
        stack = SurfaceStack.load_from_json(SPECTROMETER)["stack"]
        MechanicalStateController(stack).set_state("DMD", "OFF")
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "stateful.json")
            stack.save_to_json(path)
            restored = SurfaceStack.load_from_json(path)["stack"]
        dmd = next(s for s in restored.surfaces if "DMD" in s.name.upper())
        self.assertEqual(active_state_name(dmd), "OFF")
        self.assertIn("ON", dmd.mechanical_states)
        self.assertIn("OFF", dmd.mechanical_states)

    def test_generic_state_schema_supports_future_filters(self):
        filter_surface = Surface(
            name="Color_Filter_Wheel",
            mechanical_state="RED",
            mechanical_states={
                "RED": {"wavelength_range_um": [0.62, 0.75], "transmission": 0.9},
                "BLUE": {"wavelength_range_um": [0.42, 0.50], "transmission": 0.8},
            },
        )
        self.assertAlmostEqual(state_transmission(filter_surface, 0.65), 0.9)
        self.assertEqual(state_transmission(filter_surface, 0.46), 0.0)
        filter_surface.mechanical_state = "BLUE"
        self.assertAlmostEqual(state_transmission(filter_surface, 0.46), 0.8)

    def test_diagnostics_warn_when_dmd_is_off(self):
        stack = SurfaceStack.load_from_json(SPECTROMETER)["stack"]
        MechanicalStateController(stack).set_state("DMD", "OFF")
        report = ModelSanityValidator.inspect_geometry(stack)
        self.assertIn("DMD_OFF_DETECTOR_DARK", [item.code for item in report.items])


class TestMechanicalStateUIAndAI(unittest.TestCase):
    def setUp(self):
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", surface_type=SurfaceType.OBJECT, thickness=10.0))
        stack.add_surface(Surface(name="DMD4500", glass="MIRROR", tilt_x=12.0, thickness=10.0))
        stack.add_surface(Surface(name="Detector", surface_type=SurfaceType.DETECTOR))
        self.model = LensDataTableModel(stack=stack)
        self.model.undo_manager = UndoManager()
        self.dock = ComponentEditorDockWidget(self.model)
        self.dock.set_selected_surface(1)

    def test_editor_switch_is_visible_and_undoable(self):
        self.assertTrue(self.dock.mechanism_group.isEnabled())
        self.assertEqual(self.dock.mechanical_state_combo.count(), 3)
        # TI devices are controlled exclusively from TI Modulator Studio:
        # duplicate quick buttons are hidden and the Studio shortcut is shown.
        self.assertTrue(self.dock.btn_dmd_off.isHidden())
        self.assertFalse(self.dock.btn_open_ti_studio.isHidden())
        self.assertTrue(self.dock._apply_mechanical_state("OFF"))
        self.assertEqual(self.model._surfaces[1].mechanical_state, "OFF")
        self.assertTrue(self.model.undo_manager.can_undo())

        current = self.model.get_surface_stack()
        restored = self.model.undo_manager.undo(current)
        self.assertIsNotNone(restored)
        self.assertEqual(restored.stack_surfaces[1].mechanical_state, "")

        restored_stack = SurfaceStack()
        for surface in restored.stack_surfaces:
            restored_stack.add_surface(surface)
        redone = self.model.undo_manager.redo(restored_stack)
        self.assertIsNotNone(redone)
        self.assertEqual(redone.stack_surfaces[1].mechanical_state, "OFF")

    def test_ai_understands_dmd_on_off_and_tool_applies_it(self):
        parsed = parse_optical_command_heuristics("turn the DMD off")
        self.assertEqual(parsed["name"], "set_mechanical_state")
        self.assertEqual(parsed["arguments"]["state"], "OFF")
        message = set_mechanical_state("DMD", "OFF", system=OpticalSystem(self.model.get_surface_stack()))
        self.assertIn("beam dump", message)
        self.assertEqual(self.model._surfaces[1].mechanical_state, "OFF")


if __name__ == "__main__":
    unittest.main()
