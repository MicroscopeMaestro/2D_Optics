"""
Unit Tests for DMD Simulation in 2D Optics Application.

Validates:
1. Single reflective surface DMD model (no redundant CB surfaces).
2. Beam deflection into dump space when DMD is OFF (48 deg deflected path with terminal segment).
3. Rays reaching detector when DMD is ON, and 0 rays reaching detector when DMD is OFF.
4. Optical elements remain strictly motionless in 3D global coordinate space when DMD is toggled.
5. Interactive UI toggling in DualViewWidget, Visualizer3DWidget, and MainWindow.
6. Component editor auto-targeting of DMD.
7. Diagnostic reporting and analysis tabs handling when DMD is OFF.
"""

import os
import unittest
import numpy as np

from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from optics_app.engine.mechanisms import (
    MechanicalStateController,
    active_state_name,
    is_dmd,
    simulate_state_throughput,
)
from optics_app.gui.lde_table import LensDataTableModel
from optics_app.gui.dual_view import DualViewWidget
from optics_app.gui.visualizer_3d import Visualizer3DWidget
from optics_app.gui.main_window import MainWindow
from optics_app.gui.component_editor_dock import ComponentEditorDockWidget
from optics_app.gui.analysis_tabs import SpotDiagramTab, CCDHeatmapTab
from optics_app.gui.qt_compat import QT_AVAILABLE, QtWidgets


if QT_AVAILABLE and hasattr(QtWidgets, "QApplication"):
    _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


ROOT = os.path.dirname(os.path.dirname(__file__))
SPECTROMETER = os.path.join(ROOT, "spectrometer_raman_50mm_100mm.json")
MANUAL_SPECTROMETER_DESKTOP = os.path.join(
    os.environ.get("USERPROFILE", "C:/Users/Juan"),
    "Desktop",
    "spectrometer_raman_50mm_100mm_manual.json",
)


class TestDMDSimulation(unittest.TestCase):
    def setUp(self):
        # Load standard model if available, else build minimal DMD model
        if os.path.exists(SPECTROMETER):
            loaded = SurfaceStack.load_from_json(SPECTROMETER)
            self.stack = loaded["stack"]
            self.wavelengths = loaded.get("wavelengths", [0.450, 0.546, 0.650])
        else:
            self.stack = SurfaceStack()
            self.stack.add_surface(Surface(name="OBJ", surface_type=SurfaceType.OBJECT, thickness=20.0))
            self.stack.add_surface(Surface(name="DMD4500", glass="MIRROR", tilt_x=12.0, thickness=50.0))
            self.stack.add_surface(Surface(name="Condenser", radius=25.0, glass="N-BK7", thickness=5.0, semi_diameter=12.5))
            self.stack.add_surface(Surface(name="Condenser_R2", radius=-50.0, glass="AIR", thickness=30.0, semi_diameter=12.5))
            self.stack.add_surface(Surface(name="Detector", surface_type=SurfaceType.DETECTOR, semi_diameter=5.0))
            self.wavelengths = [0.532]

    def test_single_surface_dmd_no_redundant_cb(self):
        """Verify model uses single DMD surface without redundant CB surfaces."""
        dmd_surfs = [s for s in self.stack.surfaces if is_dmd(s)]
        self.assertEqual(len(dmd_surfs), 1, "Expected exactly 1 DMD reflective surface")
        dmd = dmd_surfs[0]
        self.assertEqual(dmd.glass, "MIRROR", "DMD must be a reflective mirror surface")
        self.assertGreater(dmd.thickness, 0.0, "DMD must carry propagation thickness to next optic")

        # Check there is no CB_After_DMD
        cb_after = [s for s in self.stack.surfaces if "CB_After_DMD" in s.name or "CB_Post_DMD" in s.name]
        self.assertEqual(len(cb_after), 0, "Redundant post-DMD coordinate break surface should not exist")

    def test_dmd_on_reaches_detector_and_off_deflects_48_deg(self):
        """Verify rays reach detector in ON state and deflect 48 deg into beam dump when OFF."""
        ctrl = MechanicalStateController(self.stack)
        dmd_idx = next(i for i, s in enumerate(self.stack.surfaces) if is_dmd(s))

        # 1. Test ON state
        ctrl.set_state(dmd_idx, "ON")
        self.stack._update_accumulated_positions()
        sys_on = OpticalSystem(self.stack)
        rays_on = [
            Ray(pos=np.zeros(3), dir=np.array([0.0, 0.0, 1.0]), wavelength=self.wavelengths[0])
        ]
        traced_on = sys_on.trace_ray(rays_on[0])
        det_idx = len(self.stack.surfaces) - 1

        hits_on = [h for h in traced_on.history if h.get("surface_index") == det_idx and not h.get("vignetted", False)]
        self.assertGreater(len(hits_on), 0, "Rays must reach detector when DMD is ON")

        # 2. Test OFF state
        ctrl.set_state(dmd_idx, "OFF")
        self.stack._update_accumulated_positions()
        sys_off = OpticalSystem(self.stack)
        rays_off = [
            Ray(pos=np.zeros(3), dir=np.array([0.0, 0.0, 1.0]), wavelength=self.wavelengths[0])
        ]
        traced_off = sys_off.trace_ray(rays_off[0])

        hits_off = [h for h in traced_off.history if h.get("surface_index") == det_idx and not h.get("vignetted", False)]
        self.assertEqual(len(hits_off), 0, "0 rays must reach detector when DMD is OFF")

        # 3. Verify terminal dump segment exists in ray history
        self.assertGreaterEqual(len(traced_off.history), 2, "Ray history should contain reflection and dump segment")
        last_step = traced_off.history[-1]
        self.assertTrue(last_step.get("vignetted", False), "Terminal step in dump must be marked vignetted")
        self.assertEqual(last_step.get("intensity", 1.0), 0.0, "Dump intensity should be 0.0")

        # Verify dump segment position is projected outward
        dmd_step = next(h for h in traced_off.history if h.get("surface_index") == dmd_idx)
        dmd_pt = np.array([dmd_step["x"], dmd_step["y"], dmd_step["z"]])
        dump_pt = np.array([last_step["x"], last_step["y"], last_step["z"]])
        dist = np.linalg.norm(dump_pt - dmd_pt)
        self.assertGreaterEqual(dist, 50.0, "Dump segment should extend at least 50mm outward along deflected path")

        # Verify 48 deg deflection between ON and OFF reflected ray directions
        dmd_step_on = next(h for h in traced_on.history if h.get("surface_index") == dmd_idx)
        next_step_on = traced_on.history[traced_on.history.index(dmd_step_on) + 1]
        dir_on = np.array([next_step_on["x"] - dmd_step_on["x"], next_step_on["y"] - dmd_step_on["y"], next_step_on["z"] - dmd_step_on["z"]])
        dir_on = dir_on / np.linalg.norm(dir_on)

        dir_off = np.array([dump_pt[0] - dmd_pt[0], dump_pt[1] - dmd_pt[1], dump_pt[2] - dmd_pt[2]])
        dir_off = dir_off / np.linalg.norm(dir_off)

        cos_angle = np.clip(np.dot(dir_on, dir_off), -1.0, 1.0)
        deflection_deg = float(np.degrees(np.arccos(cos_angle)))
        self.assertGreater(deflection_deg, 45.0, "Reflected beam must divert by >45 deg away from detector path")
        self.assertLess(deflection_deg, 60.0, "Reflected beam diversion must be within expected ~48-56 deg range")

    def test_optical_elements_remain_motionless(self):
        """Verify all downstream element positions and rotations remain completely fixed between ON and OFF."""
        ctrl = MechanicalStateController(self.stack)
        dmd_idx = next(i for i, s in enumerate(self.stack.surfaces) if is_dmd(s))

        ctrl.set_state(dmd_idx, "ON")
        self.stack._update_accumulated_positions()
        origins_on = [s.global_origin.copy() for s in self.stack.surfaces]
        rotations_on = [s.global_R.copy() for s in self.stack.surfaces]

        ctrl.set_state(dmd_idx, "OFF")
        self.stack._update_accumulated_positions()
        origins_off = [s.global_origin.copy() for s in self.stack.surfaces]
        rotations_off = [s.global_R.copy() for s in self.stack.surfaces]

        for idx, (p_on, p_off) in enumerate(zip(origins_on, origins_off)):
            diff = np.linalg.norm(p_on - p_off)
            self.assertAlmostEqual(diff, 0.0, delta=1e-9, msg=f"Surface {idx} moved between ON and OFF states!")

        for idx, (r_on, r_off) in enumerate(zip(rotations_on, rotations_off)):
            diff = np.max(np.abs(r_on - r_off))
            self.assertAlmostEqual(diff, 0.0, delta=1e-9, msg=f"Surface {idx} rotated between ON and OFF states!")

    def test_dual_view_widget_dmd_toggle(self):
        """Test DMD toggling via DualViewWidget interface."""
        model = LensDataTableModel(stack=self.stack)
        dual_view = DualViewWidget(model=model)
        dual_view.surface_stack = self.stack

        info = dual_view.get_dmd_surface_info()
        self.assertIsNotNone(info)
        dmd_idx, surf, cur_state = info
        expected_toggle_1 = "OFF" if cur_state == "ON" else "ON"
        expected_toggle_2 = cur_state

        # Toggle once
        new_state = dual_view.toggle_dmd_state()
        self.assertEqual(new_state, expected_toggle_1)
        self.assertEqual(active_state_name(self.stack.surfaces[dmd_idx]), expected_toggle_1)
        if hasattr(dual_view, "btn_dmd_toggle") and dual_view.btn_dmd_toggle:
            self.assertIn(expected_toggle_1, dual_view.btn_dmd_toggle.text())

        # Toggle back
        new_state_2 = dual_view.toggle_dmd_state()
        self.assertEqual(new_state_2, expected_toggle_2)
        self.assertEqual(active_state_name(self.stack.surfaces[dmd_idx]), expected_toggle_2)
        if hasattr(dual_view, "btn_dmd_toggle") and dual_view.btn_dmd_toggle:
            self.assertIn(expected_toggle_2, dual_view.btn_dmd_toggle.text())

    def test_visualizer_3d_widget_dmd_toggle(self):
        """Test DMD toggling via Visualizer3DWidget interface."""
        model = LensDataTableModel(stack=self.stack)
        vis3d = Visualizer3DWidget(model=model)
        vis3d.surface_stack = self.stack

        info = vis3d.get_dmd_surface_info()
        self.assertIsNotNone(info)
        cur_state = info[2]
        expected_state = "OFF" if cur_state == "ON" else "ON"

        new_state = vis3d.toggle_dmd_state()
        self.assertEqual(new_state, expected_state)
        if hasattr(vis3d, "btn_dmd_state") and vis3d.btn_dmd_state:
            self.assertIn(expected_state, vis3d.btn_dmd_state.text())

    def test_component_editor_auto_targets_dmd(self):
        """Test that clicking DMD ON/OFF in ComponentEditor applies to DMD even if non-DMD row is selected."""
        model = LensDataTableModel(stack=self.stack)
        editor = ComponentEditorDockWidget(model=model)

        # Select surface 0 (OBJECT, definitely not DMD)
        editor.set_selected_surface(0)
        self.assertEqual(editor._selected_index, 0)

        # Click DMD ON
        res = editor._apply_mechanical_state("ON")
        self.assertTrue(res)

        dmd_idx = next(i for i, s in enumerate(self.stack.surfaces) if is_dmd(s))
        self.assertEqual(active_state_name(self.stack.surfaces[dmd_idx]), "ON")

        # Click DMD OFF
        res = editor._apply_mechanical_state("OFF")
        self.assertTrue(res)
        self.assertEqual(active_state_name(self.stack.surfaces[dmd_idx]), "OFF")

    def test_main_window_dmd_action(self):
        """Test MainWindow DMD action integration."""
        main_win = MainWindow()
        main_win.model.set_surface_stack(self.stack)

        # Verify action exists
        self.assertIsNotNone(main_win.act_dmd_toggle)
        self.assertTrue(main_win.act_dmd_toggle.isVisible())

        # Toggle via MainWindow
        new_state = main_win._toggle_dmd_state()
        self.assertIn(new_state, ["ON", "OFF"])
        self.assertIn(new_state, main_win.act_dmd_toggle.text())

    def test_analysis_tabs_dmd_off_handling(self):
        """Test SpotDiagramTab and CCDHeatmapTab handle DMD OFF state properly."""
        ctrl = MechanicalStateController(self.stack)
        dmd_idx = next(i for i, s in enumerate(self.stack.surfaces) if is_dmd(s))
        ctrl.set_state(dmd_idx, "OFF")
        self.stack._update_accumulated_positions()
        opt_sys = OpticalSystem(self.stack)

        # Spot Diagram Tab
        spot_tab = SpotDiagramTab()
        spot_tab.update_plot(system=opt_sys, wavelengths=self.wavelengths)

        # CCD Heatmap Tab
        ccd_tab = CCDHeatmapTab()
        ccd_tab.update_plot(system=opt_sys)


if __name__ == "__main__":
    unittest.main()
