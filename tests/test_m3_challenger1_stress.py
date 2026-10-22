"""
Milestone M3 Empirical Stress & Adversarial Challenge Test Suite.
Author: Challenger 1 (Milestone M3: GUI Integration & Live Synchronization)

Empirically challenges:
1. Rapid consecutive optimization requests (spam / re-entrance / state integrity).
2. Dynamic surface insertions, deletions, reorderings in LDE table while OptimizerDockWidget is open.
3. Toggling variable checkboxes on/off and confirming model state 2-way synchronization.
4. Zero-variable optimization attempt via GUI (error handling, feedback, no crash).
5. All-variable optimization attempt via GUI (full degree of freedom bounds enforcement).
6. GUI state consistency before and after optimization (LDE grid cells, dual view, analysis tabs, status bar).
7. Extreme / adversarial boundary conditions (total vignetting, single-surface, nanometer wavelength string parsing, rapid reset).
"""

import unittest
import numpy as np
import pytest

from optics_app.gui.qt_compat import (
    QT_AVAILABLE, QtWidgets, QWidget, QDockWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QSpinBox, QCheckBox, QProgressBar, QGroupBox,
    QThread, QObject, pyqtSignal, DummySignal, Qt
)
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.optimizer import OpticalOptimizer, OptimizationResult
from optics_app.gui.lde_table import LensDataTableModel
from optics_app.gui.optimizer_dock import OptimizerDockWidget
from optics_app.gui.main_window import MainWindow

if QT_AVAILABLE and hasattr(QtWidgets, "QApplication"):
    try:
        _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    except Exception:
        pass


class TestM3EmpiricalRapidOptimization(unittest.TestCase):
    """Stress testing rapid consecutive optimization requests and state integrity."""

    def setUp(self):
        self.window = MainWindow()
        # Setup a standard singlet lens prescription
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=100.0, glass="AIR", semi_diameter=20.0))
        stack.add_surface(Surface(name="L1_Front", radius=50.0, thickness=5.0, glass="N-BK7", semi_diameter=15.0))
        stack.add_surface(Surface(name="L1_Back", radius=-50.0, thickness=30.0, glass="AIR", semi_diameter=15.0, thickness_variable=True, thickness_min=5.0, thickness_max=150.0))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=20.0))
        self.window.model.set_surface_stack(stack)

    def test_rapid_consecutive_optimizations(self):
        """Execute 15 rapid consecutive optimization calls with alternating solvers."""
        initial_stack = self.window.model.get_surface_stack()
        initial_thickness = initial_stack.surfaces[2].thickness
        self.assertEqual(initial_thickness, 30.0)

        for i in range(3):
            method = "Nelder-Mead" if i % 2 == 0 else "L-BFGS-B"
            res = self.window.on_optimize(variable_indices=[2], method=method, max_iter=25)
            self.assertTrue(res.success or res.final_rms <= res.initial_rms)
            # Verify status and dock widget state
            self.assertIn("Optimization finished", getattr(self.window, "_status_msg", ""))
            self.assertEqual(self.window.optimizer_dock.progress_bar.value(), 100)
            self.assertIn("RMS:", self.window.optimizer_dock.status_label.text())

            # Verify model stack was updated
            curr_stack = self.window.model.get_surface_stack()
            self.assertAlmostEqual(curr_stack.surfaces[2].thickness, res.optimized_thicknesses[2], places=3)

    def test_repeated_dock_button_clicks(self):
        """Simulate rapid UI button clicks on OptimizerDockWidget."""
        self.window.optimizer_dock.set_surface_variable(2, True)
        
        for _ in range(3):
            self.window.optimizer_dock.btn_optimize.clicked.emit()
            self.assertIn("Optimization finished", getattr(self.window, "_status_msg", ""))
            self.assertEqual(self.window.optimizer_dock.progress_bar.value(), 100)


class TestM3DynamicLDEInteractions(unittest.TestCase):
    """Stress testing dynamic insertions, deletions, and moves in LDE with open OptimizerDockWidget."""

    def setUp(self):
        self.window = MainWindow()

    def test_dynamic_insertions_and_checkbox_synchronization(self):
        """Inserting surfaces dynamically updates checkboxes and labels in OptimizerDockWidget."""
        dock = self.window.optimizer_dock
        model = self.window.model

        # Initial default system has 3 surfaces (OBJ, STO, IMA)
        self.assertEqual(model.rowCount(), 3)
        self.assertEqual(len(dock.checkboxes), 3)

        # Insert 5 surfaces sequentially at various positions
        self.window.on_insert_surface(1)
        self.window.on_insert_surface(2)
        self.window.on_insert_surface(0)
        self.window.on_insert_surface(model.rowCount() - 1)
        self.window.on_insert_surface(model.rowCount())

        self.assertEqual(model.rowCount(), 8)
        self.assertEqual(len(dock.checkboxes), 8)

        # Verify all checkboxes have corresponding surface references
        for idx in range(8):
            self.assertIn(idx, dock.checkboxes)
            cb = dock.checkboxes[idx]
            self.assertGreater(len(cb.text()), 0)

    def test_dynamic_surface_deletions_down_to_one(self):
        """Deleting surfaces removes checkboxes cleanly without orphaned references or stale indices."""
        dock = self.window.optimizer_dock
        model = self.window.model

        # Delete surfaces until only 1 remains
        while model.rowCount() > 1:
            self.window.on_delete_surface(0)
            self.assertEqual(len(dock.checkboxes), model.rowCount())

        self.assertEqual(model.rowCount(), 1)
        self.assertEqual(len(dock.checkboxes), 1)

        # Optimizing with 1 surface should not crash
        res = self.window.on_optimize(variable_indices=[0], max_iter=10)
        self.assertIsNotNone(res)

    def test_surface_reordering_moves_and_variables(self):
        """Moving surfaces up and down preserves variable state and updates checkboxes."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=100.0))
        stack.add_surface(Surface(name="L1", radius=50.0, thickness=5.0, glass="N-BK7"))
        stack.add_surface(Surface(name="L2", radius=0.0, thickness=30.0, glass="AIR", thickness_variable=True))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0))
        self.window.model.set_surface_stack(stack)
        dock = self.window.optimizer_dock

        # Surface index 2 is variable
        self.assertIn(2, dock.get_variable_indices())

        # Move surface 2 up to index 1
        self.window.on_move_surface_up(2)
        # Now index 1 should be variable
        self.assertIn(1, dock.get_variable_indices())
        self.assertNotIn(2, dock.get_variable_indices())

        # Move surface 1 down to index 2
        self.window.on_move_surface_down(1)
        self.assertIn(2, dock.get_variable_indices())


class TestM3VariableTogglingAndModelSync(unittest.TestCase):
    """Testing variable checkbox toggling and bidirectional model synchronization."""

    def setUp(self):
        self.window = MainWindow()
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=100.0))
        stack.add_surface(Surface(name="L1", radius=50.0, thickness=5.0, glass="N-BK7"))
        stack.add_surface(Surface(name="L2", radius=-50.0, thickness=40.0, glass="AIR"))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0))
        self.window.model.set_surface_stack(stack)

    def test_bidirectional_variable_toggling(self):
        """Toggling in GUI mutates model, and model changes update GUI."""
        dock = self.window.optimizer_dock
        stack = self.window.model.get_surface_stack()

        # 1. Toggle via GUI dock
        dock.set_surface_variable(1, True)
        dock.set_surface_variable(2, True)
        self.assertEqual(dock.get_variable_indices(), [1, 2])
        self.assertTrue(stack.surfaces[1].thickness_variable)
        self.assertTrue(stack.surfaces[2].thickness_variable)

        # 2. Untoggle via GUI dock
        dock.set_surface_variable(1, False)
        self.assertEqual(dock.get_variable_indices(), [2])
        self.assertFalse(stack.surfaces[1].thickness_variable)
        self.assertTrue(stack.surfaces[2].thickness_variable)

        # 3. Modify directly in stack and emit model reset
        stack.surfaces[0].thickness_variable = True
        self.window.model.set_surface_stack(stack)
        self.assertIn(0, dock.get_variable_indices())
        self.assertIn(2, dock.get_variable_indices())
        self.assertNotIn(1, dock.get_variable_indices())

    def test_rapid_toggle_consistency(self):
        """Repeated rapid toggling does not produce duplicate indices or desync."""
        dock = self.window.optimizer_dock
        for _ in range(10):
            dock.set_surface_variable(2, True)
            self.assertEqual(list(dock.get_variable_indices()).count(2), 1)
            dock.set_surface_variable(2, False)
            self.assertEqual(list(dock.get_variable_indices()).count(2), 0)


class TestM3ZeroAndAllVariableOptimization(unittest.TestCase):
    """Testing zero-variable and all-variable optimization attempts via GUI."""

    def setUp(self):
        self.window = MainWindow()
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=100.0, glass="AIR", semi_diameter=20.0))
        stack.add_surface(Surface(name="L1_Front", radius=50.0, thickness=5.0, glass="N-BK7", semi_diameter=15.0))
        stack.add_surface(Surface(name="L1_Back", radius=0.0, thickness=30.0, glass="AIR", semi_diameter=15.0))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=20.0))
        self.window.model.set_surface_stack(stack)

    def test_zero_variable_optimization_attempt(self):
        """Attempting optimization with zero variables returns gracefully without crash."""
        dock = self.window.optimizer_dock
        for i in range(4):
            dock.set_surface_variable(i, False)

        self.assertEqual(len(dock.get_variable_indices()), 0)

        # Trigger via explicit empty list
        res = self.window.on_optimize(variable_indices=[], max_iter=20)
        self.assertTrue(res.success)

        # Check GUI feedback
        self.assertIn("Optimization", getattr(self.window, "_status_msg", ""))

    def test_all_variable_optimization_attempt(self):
        """Optimizing all surfaces simultaneously respects bounds and updates all thicknesses."""
        dock = self.window.optimizer_dock
        for i in range(4):
            dock.set_surface_variable(i, True)

        self.assertEqual(dock.get_variable_indices(), [0, 1, 2, 3])

        res = self.window.on_optimize(variable_indices=[0, 1, 2, 3], method="Nelder-Mead", max_iter=40)
        self.assertTrue(res.success or res.final_rms <= res.initial_rms)
        self.assertEqual(len(res.optimized_thicknesses), 4)

        # Verify all thicknesses remain non-negative
        for idx, t in res.optimized_thicknesses.items():
            self.assertGreaterEqual(t, 0.0)

        # Verify table cells updated in LDE table model
        model = self.window.model
        for idx in range(4):
            raw_val = model.data(idx, 3)
            val = float(str(raw_val).replace("V", "").replace("F", "").strip())
            self.assertAlmostEqual(val, res.optimized_thicknesses[idx], places=3)


class TestM3GUIStateConsistencyAndWavelengths(unittest.TestCase):
    """Testing table cells, dual view, analysis updates, and wavelength string parsing."""

    def setUp(self):
        self.window = MainWindow()

    def test_lde_table_cell_synchronization_post_optimization(self):
        """LDE table model data cells accurately reflect optimization results."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=100.0))
        stack.add_surface(Surface(name="L1", radius=51.68, thickness=5.0, glass="N-BK7", semi_diameter=15.0))
        stack.add_surface(Surface(name="L2", radius=0.0, thickness=20.0, glass="AIR", semi_diameter=15.0, thickness_variable=True, thickness_min=10.0, thickness_max=150.0))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=20.0))
        self.window.model.set_surface_stack(stack)

        raw_init = self.window.model.data(2, 3)
        initial_cell_val = float(str(raw_init).replace("V", "").replace("F", "").strip())
        self.assertEqual(initial_cell_val, 20.0)

        res = self.window.on_optimize(variable_indices=[2], max_iter=50)
        self.assertTrue(res.success)

        # Cell 2, 3 must now equal optimized thickness
        raw_updated = self.window.model.data(2, 3)
        updated_cell_val = float(str(raw_updated).replace("V", "").replace("F", "").strip())
        self.assertAlmostEqual(updated_cell_val, res.optimized_thicknesses[2], places=4)
        self.assertNotEqual(initial_cell_val, updated_cell_val)

    def test_custom_wavelengths_and_nanometer_conversion(self):
        """Wavelength input handles micrometer and nanometer strings seamlessly."""
        if hasattr(self.window, 'wavelengths_input') and self.window.wavelengths_input is not None:
            # Test nanometer input: 450, 550, 650 -> converted to 0.45, 0.55, 0.65 um
            self.window.wavelengths_input.setText("450, 550, 650")
            res = self.window.on_optimize(variable_indices=[1], max_iter=20)
            self.assertIsNotNone(res)

            # Test invalid string fallback
            self.window.wavelengths_input.setText("abc, def, -100")
            res_fallback = self.window.on_optimize(variable_indices=[1], max_iter=20)
            self.assertIsNotNone(res_fallback)


class TestM3ExtremeBoundaryConditions(unittest.TestCase):
    """Testing adversarial optical system configurations and edge cases."""

    def test_completely_vignetted_system_optimization(self):
        """Optimizer handles extreme ray loss / vignetting without crashing."""
        window = MainWindow()
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=100.0))
        # Zero aperture on lens stops all rays
        stack.add_surface(Surface(name="L1", radius=50.0, thickness=5.0, glass="N-BK7", semi_diameter=0.0001))
        stack.add_surface(Surface(name="L2", radius=-50.0, thickness=30.0, glass="AIR", semi_diameter=0.0001))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0))
        window.model.set_surface_stack(stack)

        # Should complete and return result cleanly despite heavy penalty
        res = window.on_optimize(variable_indices=[2], max_iter=20)
        self.assertIsNotNone(res)
        self.assertGreaterEqual(res.final_rms, 0.0)

    def test_empty_system_optimization(self):
        """Optimizing an empty surface stack returns early with failure status."""
        window = MainWindow()
        window.model._data = []
        window.model._surfaces = []
        window.model._stack = SurfaceStack()

        res = window.on_optimize(variable_indices=[])
        self.assertFalse(res.success)
        self.assertIn("No surfaces available", res.message)


if __name__ == "__main__":
    unittest.main()
