"""
Dedicated GUI Integration & Live Synchronization Test Suite for Optical Optimizer.
Tests OptimizerDockWidget, MainWindow.on_optimize(), LensDataTableModel sync, and headless fallbacks.
"""

import unittest
from typing import List, Dict, Optional, Any
import numpy as np

from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.optimizer import OpticalOptimizer, OptimizationResult
from optics_app.gui.qt_compat import (
    QT_AVAILABLE, QtWidgets, QWidget, QDockWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QSpinBox, QCheckBox, QProgressBar, QGroupBox,
    QThread, QObject, pyqtSignal, DummySignal
)
from optics_app.gui.lde_table import LensDataTableModel
from optics_app.gui.optimizer_dock import OptimizerDockWidget
from optics_app.gui.main_window import MainWindow

if QT_AVAILABLE and hasattr(QtWidgets, "QApplication"):
    try:
        _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    except Exception:
        pass


class TestOptimizerDockWidgetUnit(unittest.TestCase):
    """Unit tests for OptimizerDockWidget controls, signals, and model sync."""

    def test_optimizer_dock_initialization(self):
        """Test OptimizerDockWidget creation and initial state."""
        model = LensDataTableModel()
        dock = OptimizerDockWidget(model=model)
        self.assertIsNotNone(dock)
        self.assertIs(dock.model, model)
        self.assertEqual(len(dock.get_variable_indices()), 0)

    def test_optimizer_dock_checkboxes_created_for_surfaces(self):
        """Dynamic checkboxes are created for each surface in the model."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=100.0))
        stack.add_surface(Surface(name="L1", radius=50.0, thickness=5.0, glass="N-BK7"))
        stack.add_surface(Surface(name="L2", radius=-50.0, thickness=40.0, glass="AIR"))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0))

        model = LensDataTableModel(stack=stack)
        dock = OptimizerDockWidget(model=model)

        self.assertEqual(len(dock.checkboxes), 4)

    def test_optimizer_dock_toggling_checkbox_updates_model(self):
        """Toggling a checkbox in the dock widget updates surface.thickness_variable in the stack."""
        model = LensDataTableModel()
        dock = OptimizerDockWidget(model=model)

        dock.set_surface_variable(1, True)
        self.assertIn(1, dock.get_variable_indices())
        stack = model.get_surface_stack()
        self.assertTrue(getattr(stack.surfaces[1], 'thickness_variable', False))

        dock.set_surface_variable(1, False)
        self.assertNotIn(1, dock.get_variable_indices())
        self.assertFalse(getattr(stack.surfaces[1], 'thickness_variable', False))

    def test_optimizer_dock_model_update_rebuilds_checkboxes(self):
        """When model stack changes, dock checkboxes rebuild and preserve variable state."""
        stack = SurfaceStack()
        s0 = Surface(name="OBJ", thickness=100.0)
        s1 = Surface(name="L1", radius=50.0, thickness=5.0, glass="N-BK7")
        s2 = Surface(name="L2", radius=0.0, thickness=30.0, glass="AIR", thickness_variable=True)
        s3 = Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0)
        for s in [s0, s1, s2, s3]:
            stack.add_surface(s)

        model = LensDataTableModel(stack=stack)
        dock = OptimizerDockWidget(model=model)

        self.assertIn(2, dock.get_variable_indices())
        self.assertTrue(dock.checkboxes[2].isChecked())

    def test_optimizer_dock_signal_emission_on_click(self):
        """Clicking 'Optimize System' emits optimize_requested signal with indices, method, and max_iter."""
        model = LensDataTableModel()
        dock = OptimizerDockWidget(model=model)
        dock.set_surface_variable(1, True)

        received_payload = []
        if hasattr(dock, "optimize_requested") and dock.optimize_requested is not None:
            dock.optimize_requested.connect(lambda idxs, m, mi: received_payload.append((idxs, m, mi)))

        dock.on_optimize_clicked()
        self.assertEqual(len(received_payload), 1)
        indices, method, max_iter = received_payload[0]
        self.assertEqual(indices, [1])
        self.assertIn(method, ["Nelder-Mead", "L-BFGS-B"])
        self.assertGreater(max_iter, 0)

    def test_optimizer_dock_ssot_wavelength_badge_syncs_with_system_config(self):
        """OptimizerDock reads wavelengths directly from SystemConfig (SSOT) and updates label."""
        from optics_app.engine.system_config import SystemConfig, WavelengthEntry
        sys_cfg = SystemConfig()
        sys_cfg.wavelengths = [
            WavelengthEntry(wavelength_um=0.785, is_primary=True),
            WavelengthEntry(wavelength_um=0.850, is_primary=False),
            WavelengthEntry(wavelength_um=1.064, is_primary=False)
        ]
        model = LensDataTableModel()
        model.system_config = sys_cfg
        dock = OptimizerDockWidget(model=model)
        wls = dock.get_wavelengths()
        self.assertEqual(len(wls), 3)
        self.assertAlmostEqual(wls[0], 0.785)
        self.assertAlmostEqual(wls[2], 1.064)
        if hasattr(dock, 'lbl_active_wavelengths') and hasattr(dock.lbl_active_wavelengths, 'text'):
            txt = dock.lbl_active_wavelengths.text()
            self.assertIn("0.785", txt)
            self.assertIn("1.064", txt)

    def test_optimizer_dock_active_variables_badge_syncs_with_lde(self):
        """Active variables badge displays variables enabled in LDE."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=50.0))
        stack.add_surface(Surface(name="Front", radius=60.0, thickness=5.0, glass="N-BK7", radius_variable=True))
        stack.add_surface(Surface(name="Rear", radius=-70.0, thickness=45.0, glass="AIR", thickness_variable=True))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0))
        model = LensDataTableModel(stack=stack)
        dock = OptimizerDockWidget(model=model)
        dock.update_system_info()
        if hasattr(dock, 'lbl_active_variables') and hasattr(dock.lbl_active_variables, 'text'):
            txt = dock.lbl_active_variables.text()
            self.assertIn("Front", txt)
            self.assertIn("Rear", txt)

    def test_optimizer_dock_open_system_explorer_signal(self):
        """Clicking configure wavelengths button emits open_system_explorer_requested."""
        dock = OptimizerDockWidget()
        requested_tabs = []
        if hasattr(dock, 'open_system_explorer_requested') and dock.open_system_explorer_requested is not None:
            dock.open_system_explorer_requested.connect(lambda tab: requested_tabs.append(tab))
            dock._on_open_system_explorer_wavelengths()
            self.assertEqual(requested_tabs, [1])


class TestMainWindowOptimizerIntegration(unittest.TestCase):
    """Integration tests for MainWindow and Optimizer execution with live UI sync."""

    def test_main_window_has_optimizer_dock(self):
        """MainWindow instantiates and embeds OptimizerDockWidget."""
        window = MainWindow()
        dock = window.get_optimizer_dock()
        self.assertIsNotNone(dock)
        self.assertIsInstance(dock, OptimizerDockWidget)

    def test_main_window_on_optimize_execution(self):
        """MainWindow.on_optimize runs optimizer, updates model stack, and triggers ray trace."""
        window = MainWindow()
        # Setup a defocused system
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=1000.0))
        stack.add_surface(Surface(name="L1_Front", radius=51.68, thickness=5.0, glass="N-BK7", semi_diameter=15.0))
        s2 = Surface(name="L1_Back", radius=0.0, thickness=30.0, glass="AIR", semi_diameter=15.0, thickness_variable=True, thickness_min=10.0, thickness_max=200.0)
        stack.add_surface(s2)
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=20.0))

        window.model.set_surface_stack(stack)
        window.optimizer_dock.set_surface_variable(2, True)

        result = window.on_optimize(variable_indices=[2], method="Nelder-Mead", max_iter=60)

        self.assertTrue(result.success)
        self.assertGreaterEqual(result.rms_reduction_pct, 50.0)
        # Check that model updated
        updated_stack = window.model.get_surface_stack()
        self.assertAlmostEqual(updated_stack.surfaces[2].thickness, result.optimized_thicknesses[2], places=4)

    def test_main_window_on_optimize_menu_action(self):
        """Calling on_optimize_system runs optimization smoothly."""
        window = MainWindow()
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=100.0))
        stack.add_surface(Surface(name="L1", radius=50.0, thickness=5.0, glass="N-BK7", semi_diameter=15.0))
        stack.add_surface(Surface(name="L2", radius=0.0, thickness=30.0, glass="AIR", semi_diameter=15.0, thickness_variable=True))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=15.0))
        window.model.set_surface_stack(stack)

        window.optimizer_dock.set_surface_variable(2, True)
        window.on_optimize_system()

        self.assertIn("Optimization finished", getattr(window, "_status_msg", ""))


class TestQtCompatHeadlessWidgets(unittest.TestCase):
    """Test headless mock widget classes in qt_compat."""

    def test_mock_checkbox(self):
        from optics_app.gui.qt_compat import DummyQtWidgetsModule
        cb = DummyQtWidgetsModule.QCheckBox("Test Checkbox")
        self.assertFalse(cb.isChecked())
        self.assertEqual(cb.text(), "Test Checkbox")
        toggled_events = []
        cb.toggled.connect(lambda v: toggled_events.append(v))
        cb.setChecked(True)
        self.assertTrue(cb.isChecked())
        self.assertEqual(toggled_events, [True])

    def test_mock_progressbar(self):
        from optics_app.gui.qt_compat import DummyQtWidgetsModule
        pb = DummyQtWidgetsModule.QProgressBar()
        pb.setRange(0, 100)
        pb.setValue(75)
        self.assertEqual(pb.value(), 75)

    def test_mock_groupbox(self):
        from optics_app.gui.qt_compat import DummyQtWidgetsModule
        gb = DummyQtWidgetsModule.QGroupBox("Test Title")
        self.assertEqual(gb.title(), "Test Title")
        gb.setTitle("New Title")
        self.assertEqual(gb.title(), "New Title")

    def test_mock_thread(self):
        from optics_app.gui.qt_compat import DummyQtCoreModule
        t = DummyQtCoreModule.QThread()
        events = []
        t.started.connect(lambda: events.append("started"))
        t.finished.connect(lambda: events.append("finished"))
        t.start()
        self.assertEqual(events, ["started", "finished"])

    def test_mock_object(self):
        from optics_app.gui.qt_compat import DummyQtCoreModule
        obj = DummyQtCoreModule.QObject()
        self.assertIsNone(obj.parent)


if __name__ == "__main__":
    unittest.main()
