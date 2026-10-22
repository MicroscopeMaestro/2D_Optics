"""
Adversarial Stress Test Suite: Milestone 3 (UI Delivery, Diagnostics & Tree Synchronization)
Author: Challenger 1 (Milestone 3)

Empirically tests and stress-tests:
1. Rapid-fire re-optimization (10+ consecutive cycles) on MainWindow and LensDataTableModel
   in headless mode, checking for race conditions, stack corruption, and valid optimizer_dock.last_result.
2. Optical edge cases: single lenses, flat windows, systems with stop at surface 1, and afocal systems.
3. Atomic stack updates: ensuring aborted or cancelled optimizations do not corrupt SurfaceStack.
4. UI trigger & dock wiring: verifying on_optimize_clicked and optimizer_dock execution.
5. Byte-for-byte SHA-256 tree parity between optics_app/ and v2/.
"""

import os
import sys
import math
import filecmp
import hashlib
import unittest
import pytest

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from optics_app.gui.qt_compat import QT_AVAILABLE, QtWidgets
if QT_AVAILABLE:
    _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.optimizer import OpticalOptimizer, OptimizationResult, TerminationStatus
from optics_app.gui.lde_table import LensDataTableModel
from optics_app.gui.optimizer_dock import OptimizerDockWidget
from optics_app.gui.main_window import MainWindow


class TestChallenger1M3RapidReoptimization(unittest.TestCase):
    """Stress tests rapid-fire consecutive optimizations on MainWindow and LensDataTableModel."""

    def setUp(self):
        self.window = MainWindow()
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=100.0, semi_diameter=20.0))
        stack.add_surface(Surface(name="S1", radius=50.0, thickness=5.0, glass="N-BK7", semi_diameter=15.0))
        stack.add_surface(Surface(name="S2", radius=-40.0, thickness=3.0, glass="N-SF2", semi_diameter=15.0))
        stack.add_surface(Surface(name="S3", radius=-120.0, thickness=45.0, glass="AIR", semi_diameter=15.0))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=20.0))
        self.window.model.set_surface_stack(stack)

    def test_ten_consecutive_reoptimizations(self):
        """Execute 10 consecutive optimization runs alternating solvers in headless mode."""
        solvers = ["DLS", "Nelder-Mead"]
        for cycle in range(10):
            method = solvers[cycle % 2]
            var_dict = {"thickness": [3], "radius": [1, 2]}
            res = self.window.on_optimize(
                variable_indices=var_dict,
                method=method,
                max_iter=10,
                num_rays=11
            )
            # 1. Verify result is valid
            self.assertIsNotNone(res)
            self.assertTrue(res.success or res.final_rms <= res.initial_rms)

            # 2. Verify optimizer_dock.last_result is updated accurately after every run
            dock_res = self.window.optimizer_dock.last_result
            self.assertIsNotNone(dock_res, f"Cycle {cycle}: optimizer_dock.last_result is None")
            self.assertTrue(hasattr(dock_res, "final_merit"), f"Cycle {cycle}: missing final_merit")
            self.assertIsInstance(dock_res.final_merit, float, f"Cycle {cycle}: final_merit not float")
            self.assertGreaterEqual(dock_res.final_merit, 0.0, f"Cycle {cycle}: negative final_merit")

            self.assertTrue(hasattr(dock_res, "iterations"), f"Cycle {cycle}: missing iterations")
            self.assertIsInstance(dock_res.iterations, int, f"Cycle {cycle}: iterations not int")
            self.assertGreaterEqual(dock_res.iterations, 0, f"Cycle {cycle}: negative iterations")

            self.assertTrue(hasattr(dock_res, "initial_merit"), f"Cycle {cycle}: missing initial_merit")
            self.assertIsInstance(dock_res.converged, bool, f"Cycle {cycle}: converged not bool")

            # 3. Verify atomic model updates and table synchronization
            m_stack = self.window.model.get_surface_stack()
            self.assertEqual(len(m_stack.surfaces), 5, f"Cycle {cycle}: surface count altered")
            self.assertEqual(self.window.model.rowCount(), 5, f"Cycle {cycle}: table row count altered")

            # Check that model table cell matches stack thickness
            opt_t3 = m_stack.surfaces[3].thickness
            table_t3 = float(self.window.model._data[3][3])
            self.assertAlmostEqual(opt_t3, table_t3, places=4, msg=f"Cycle {cycle}: table cell mismatch")

            # Check dock progress bar reset to 100 on finish
            self.assertEqual(self.window.optimizer_dock.progress_bar.value(), 100)


class TestChallenger1M3EdgeCaseSystems(unittest.TestCase):
    """Stress tests optical edge case configurations in headless UI."""

    def setUp(self):
        self.window = MainWindow()

    def test_single_lens_system(self):
        """Singlet lens (4 surfaces: OBJ, S1, S2, DET) optimization and UI delivery."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=100.0, semi_diameter=20.0))
        stack.add_surface(Surface(name="S1", radius=50.0, thickness=5.0, glass="N-BK7", semi_diameter=15.0))
        stack.add_surface(Surface(name="S2", radius=-50.0, thickness=45.0, glass="AIR", semi_diameter=15.0))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=20.0))
        self.window.model.set_surface_stack(stack)

        res = self.window.on_optimize(
            variable_indices={"thickness": [2], "radius": [1, 2]},
            method="DLS",
            max_iter=15,
            num_rays=11
        )
        self.assertIsNotNone(res)
        dock_res = self.window.optimizer_dock.last_result
        self.assertIsNotNone(dock_res)
        self.assertIsInstance(dock_res.final_merit, float)
        self.assertGreaterEqual(dock_res.final_merit, 0.0)
        self.assertIsInstance(dock_res.iterations, int)
        self.assertEqual(self.window.model.rowCount(), 4)

    def test_flat_window_system(self):
        """Plane-parallel plate / flat window system (R1=0, R2=0) with curvature clamping."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=50.0, semi_diameter=10.0))
        stack.add_surface(Surface(name="S1", radius=0.0, thickness=5.0, glass="N-BK7", semi_diameter=10.0))
        stack.add_surface(Surface(name="S2", radius=0.0, thickness=50.0, glass="AIR", semi_diameter=10.0))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=10.0))
        self.window.model.set_surface_stack(stack)

        res = self.window.on_optimize(
            variable_indices={"thickness": [2], "radius": [1, 2]},
            method="DLS",
            max_iter=10,
            num_rays=11
        )
        self.assertIsNotNone(res)
        dock_res = self.window.optimizer_dock.last_result
        self.assertIsNotNone(dock_res)
        self.assertIsInstance(dock_res.final_merit, float)
        self.assertFalse(math.isnan(dock_res.final_merit))
        self.assertFalse(math.isinf(dock_res.final_merit))

    def test_stop_at_surface_one_system(self):
        """System with aperture stop positioned directly at Surface 1 (OBJ, STOP, S2, S3, DET)."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=100.0, semi_diameter=20.0))
        stack.add_surface(Surface(name="STOP", surface_type=SurfaceType.STOP, thickness=5.0, glass="AIR", semi_diameter=10.0))
        stack.add_surface(Surface(name="S2", radius=40.0, thickness=6.0, glass="N-BK7", semi_diameter=15.0))
        stack.add_surface(Surface(name="S3", radius=-60.0, thickness=40.0, glass="AIR", semi_diameter=15.0))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=20.0))
        self.window.model.set_surface_stack(stack)

        res = self.window.on_optimize(
            variable_indices={"thickness": [3], "radius": [2, 3]},
            method="DLS",
            max_iter=15,
            num_rays=11
        )
        self.assertIsNotNone(res)
        dock_res = self.window.optimizer_dock.last_result
        self.assertIsNotNone(dock_res)
        self.assertIsInstance(dock_res.final_merit, float)
        self.assertGreaterEqual(dock_res.final_merit, 0.0)

    def test_afocal_optical_system(self):
        """Afocal system with infinite conjugate and separated elements."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=100.0, semi_diameter=10.0))
        stack.add_surface(Surface(name="L1_front", radius=50.0, thickness=5.0, glass="N-BK7", semi_diameter=10.0))
        stack.add_surface(Surface(name="L1_back", radius=0.0, thickness=80.0, glass="AIR", semi_diameter=10.0))
        stack.add_surface(Surface(name="L2_front", radius=0.0, thickness=3.0, glass="N-SF2", semi_diameter=8.0))
        stack.add_surface(Surface(name="L2_back", radius=-30.0, thickness=50.0, glass="AIR", semi_diameter=8.0))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=10.0))
        self.window.model.set_surface_stack(stack)

        res = self.window.on_optimize(
            variable_indices={"thickness": [2, 4], "radius": [1, 4]},
            method="DLS",
            max_iter=10,
            num_rays=11
        )
        self.assertIsNotNone(res)
        dock_res = self.window.optimizer_dock.last_result
        self.assertIsNotNone(dock_res)
        self.assertIsInstance(dock_res.final_merit, float)

    def test_degenerate_zero_variable_system(self):
        """System optimization with empty variables returns gracefully without modifying stack."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=100.0, semi_diameter=10.0))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=10.0))
        self.window.model.set_surface_stack(stack)

        res = self.window.on_optimize(variable_indices={}, method="DLS", max_iter=10)
        self.assertIsNotNone(res)
        self.assertEqual(res.status, TerminationStatus.NO_VARIABLES)
        self.assertEqual(self.window.model.rowCount(), 2)


class TestChallenger1M3AtomicStackUpdates(unittest.TestCase):
    """Stress tests stack atomicity under abort and cancellation."""

    def setUp(self):
        self.window = MainWindow()
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=100.0, semi_diameter=20.0))
        stack.add_surface(Surface(name="S1", radius=50.0, thickness=5.0, glass="N-BK7", semi_diameter=15.0))
        stack.add_surface(Surface(name="S2", radius=-40.0, thickness=3.0, glass="N-SF2", semi_diameter=15.0))
        stack.add_surface(Surface(name="S3", radius=-120.0, thickness=45.0, glass="AIR", semi_diameter=15.0))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=20.0))
        self.window.model.set_surface_stack(stack)

    def test_abort_maintains_atomic_stack_integrity(self):
        """Aborting optimization mid-run preserves valid stack without NaN or corrupt state."""
        aborted = []
        def on_dock_progress(val):
            if val >= 20 and not aborted:
                aborted.append(True)
                self.window._abort_optimization()

        self.window.optimizer_dock.set_progress = on_dock_progress
        res = self.window.on_optimize(
            variable_indices={"thickness": [3], "radius": [1, 2]},
            method="Nelder-Mead",
            max_iter=100
        )
        self.assertIsNotNone(res)
        self.assertEqual(res.status, TerminationStatus.USER_ABORTED)
        self.assertIn("aborted", res.message.lower())

        # Verify stack is intact
        m_stack = self.window.model.get_surface_stack()
        self.assertEqual(len(m_stack.surfaces), 5)
        for s in m_stack.surfaces:
            self.assertFalse(math.isnan(s.thickness))
            self.assertFalse(math.isnan(s.radius))
            self.assertFalse(math.isinf(s.thickness))


class TestChallenger1M3BugsAndDiscrepancies(unittest.TestCase):
    """
    Adversarial tests specifically catching uncovered defects in Milestone 3:
    1. Missing 'os' and 'sys' imports in optimizer_dock.py breaking on_optimize_clicked.
    2. File divergence in visualizer_3d.py violating 100% tree parity between optics_app/ and v2/.
    """

    def test_optimizer_dock_on_optimize_clicked_nameerror(self):
        """
        Verify if on_optimize_clicked() can execute without NameError: name 'os' is not defined.
        If optimizer_dock.py is missing 'import os, sys', clicking 'Optimize System' button
        fails and sets status_label to an error message instead of triggering optimization.
        """
        model = LensDataTableModel()
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=100.0, semi_diameter=20.0))
        stack.add_surface(Surface(name="S1", radius=50.0, thickness=5.0, glass="N-BK7", semi_diameter=15.0))
        stack.add_surface(Surface(name="S2", radius=-50.0, thickness=45.0, glass="AIR", semi_diameter=15.0))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=20.0))
        model.set_surface_stack(stack)

        dock = OptimizerDockWidget(model=model)
        dock.set_surface_variable(2, True)

        # Trigger on_optimize_clicked
        dock.on_optimize_clicked()
        status_text = dock.status_label.text()

        # Adversarially assert that status_text does NOT contain "name 'os' is not defined"
        self.assertNotIn("name 'os' is not defined", status_text,
                         "BUG DETECTED: optimizer_dock.py line 707 references 'os' and 'sys' without importing them!")

    def test_tree_synchronization_shared_files_parity(self):
        """
        Empirically verify 100% byte-for-byte SHA-256 equivalence across all shared files
        between optics_app/ and v2/.
        Fails if any shared file has divergent contents (e.g. visualizer_3d.py).
        """
        mismatched_files = []
        for root, dirs, files in os.walk("optics_app"):
            if "__pycache__" in root:
                continue
            for f in files:
                if f.endswith(".pyc"):
                    continue
                p1 = os.path.join(root, f)
                rel = os.path.relpath(p1, "optics_app")
                p2 = os.path.join("v2", rel)
                if os.path.exists(p2):
                    if not filecmp.cmp(p1, p2, shallow=False):
                        mismatched_files.append(rel)

        self.assertEqual(
            mismatched_files, [],
            f"BUG DETECTED: Shared files differ between optics_app/ and v2/: {mismatched_files}"
        )


if __name__ == "__main__":
    import math
    unittest.main()
