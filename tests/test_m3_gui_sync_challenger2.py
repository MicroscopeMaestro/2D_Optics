"""
Empirical Challenger 2 Verification Harness for Milestone M3:
GUI Integration & Live Synchronization

Tests:
1. Live Visualization Synchronization:
   - MainWindow.on_optimize() triggers ray tracing and updates dual_view CAD layout and analysis_tabs.tab_spot.
   - Spot diagram and dual view reflect new geometry and RMS spot radius.
2. Multi-Wavelength Ray Tracing:
   - Verification via GUI source editor and toolbar wavelengths input.
   - Tracing multi-wavelength rays before and after optimization.
3. Optimizer Dock Algorithm Invocations:
   - Nelder-Mead and L-BFGS-B algorithm selections and execution via dock signal/button.
   - Parameter bounds compliance and iteration control.
4. Performance and Memory Stress:
   - 50 consecutive optimization cycles on MainWindow.
   - Leak detection via tracemalloc and execution timing profile.
5. Circular Import & Headless Qt Robustness.
"""

import sys
import os
import time
import tracemalloc
import unittest
import numpy as np

# Ensure offscreen Qt
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["MPLBACKEND"] = "Agg"

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.source import SourceConfig, SourceType
from optics_app.gui.qt_compat import QT_AVAILABLE, QtWidgets
from optics_app.gui.lde_table import LensDataTableModel
from optics_app.gui.optimizer_dock import OptimizerDockWidget
from optics_app.gui.main_window import MainWindow
from optics_app.gui.dual_view import DualViewWidget
from optics_app.gui.analysis_tabs import AnalysisTabsWidget, SpotDiagramTab

if QT_AVAILABLE and hasattr(QtWidgets, "QApplication"):
    _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(["test_harness"])


def create_test_singlet_defocused() -> SurfaceStack:
    """Creates a singlet lens system with 30mm defocus."""
    stack = SurfaceStack()
    stack.add_surface(Surface(name="OBJ", surface_type=SurfaceType.OBJECT, thickness=1000.0))
    stack.add_surface(Surface(name="L1_Front", radius=51.68, thickness=5.0, glass="N-BK7", semi_diameter=15.0))
    # Defocused back surface to detector distance: 30mm (ideal is ~96.5mm)
    s2 = Surface(
        name="L1_Back",
        radius=0.0,
        thickness=30.0,
        glass="AIR",
        semi_diameter=15.0,
        thickness_variable=True,
        thickness_min=10.0,
        thickness_max=200.0
    )
    stack.add_surface(s2)
    stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=25.0))
    return stack


class TestM3LiveVisualizationSync(unittest.TestCase):
    """Empirical challenges for live CAD layout and spot diagram synchronization."""

    def setUp(self):
        self.window = MainWindow()
        self.stack = create_test_singlet_defocused()
        self.window.model.set_surface_stack(self.stack)

    def test_on_optimize_updates_model_dual_view_and_spot_tab(self):
        """
        Verify that MainWindow.on_optimize():
        1. Reduces RMS spot radius by >= 50%
        2. Updates model stack thickness
        3. Updates dual_view surface_stack and rays
        4. Updates analysis_tabs.tab_spot with new rays
        """
        initial_thickness = self.window.model.get_surface_stack().surfaces[2].thickness
        self.assertEqual(initial_thickness, 30.0)

        # Initial ray trace
        self.window.on_trace_rays()
        init_rays = self.window.dual_view.rays
        self.assertIsNotNone(init_rays)
        self.assertGreater(len(init_rays), 0)

        # Run optimization
        result = self.window.on_optimize(variable_indices=[2], method="Nelder-Mead", max_iter=80)

        self.assertTrue(result.success)
        self.assertGreater(result.rms_reduction_pct, 50.0)
        self.assertLess(result.final_rms, result.initial_rms * 0.5)

        # Verify model stack was updated
        updated_stack = self.window.model.get_surface_stack()
        new_thickness = updated_stack.surfaces[2].thickness
        self.assertAlmostEqual(new_thickness, result.optimized_thicknesses[2], places=3)
        self.assertNotEqual(new_thickness, 30.0)

        # Verify dual_view layout was updated with new geometry and rays
        self.assertIsNotNone(self.window.dual_view.surface_stack)
        self.assertAlmostEqual(self.window.dual_view.surface_stack.surfaces[2].thickness, new_thickness, places=3)
        self.assertIsNotNone(self.window.dual_view.rays)
        self.assertGreater(len(self.window.dual_view.rays), 0)

        # Verify spot tab is accessible and refreshed
        self.assertIsNotNone(self.window.analysis_tabs.tab_spot)
        self.assertEqual(self.window.analysis_tabs.active_tab, "Spot Diagram")

    def test_optimizer_dock_signal_triggering_live_sync(self):
        """Verify that triggering optimization via OptimizerDock signal synchronizes the whole GUI."""
        dock = self.window.optimizer_dock
        dock.set_surface_variable(2, True)
        self.assertEqual(dock.get_variable_indices(), [2])

        # Click dock optimize button
        dock.on_optimize_clicked()

        # Check that dock status and progress updated
        status_text = dock.status_label.text() if hasattr(dock, 'status_label') else ""
        self.assertIn("RMS:", status_text)
        if hasattr(dock, 'progress_bar') and hasattr(dock.progress_bar, 'value'):
            self.assertEqual(dock.progress_bar.value(), 100)


class TestM3MultiWavelengthRayTracing(unittest.TestCase):
    """Empirical challenges for multi-wavelength ray tracing through GUI source editor and toolbar."""

    def setUp(self):
        self.window = MainWindow()
        self.stack = create_test_singlet_defocused()
        self.window.model.set_surface_stack(self.stack)

    def test_toolbar_wavelength_parsing_and_tracing(self):
        """Verify multi-wavelength input in toolbar parses and traces rays for all wavelengths."""
        if hasattr(self.window, 'wavelengths_input') and self.window.wavelengths_input is not None:
            # Set 3 distinct wavelengths in micrometers
            self.window.wavelengths_input.setText("0.4861, 0.5876, 0.6563")
            self.window.on_trace_rays()

            rays = self.window.dual_view.rays
            self.assertIsNotNone(rays)
            wls_in_rays = set(round(r.wavelength, 4) for r in rays)
            self.assertIn(0.4861, wls_in_rays)
            self.assertIn(0.5876, wls_in_rays)
            self.assertIn(0.6563, wls_in_rays)

    def test_toolbar_wavelength_nanometer_conversion(self):
        """Verify that user inputs in nanometers (e.g. 486, 587, 656) are auto-converted to micrometers."""
        if hasattr(self.window, 'wavelengths_input') and self.window.wavelengths_input is not None:
            self.window.wavelengths_input.setText("486.1, 587.6, 656.3")
            self.window.on_trace_rays()

            rays = self.window.dual_view.rays
            self.assertIsNotNone(rays)
            wls_in_rays = set(round(r.wavelength, 4) for r in rays)
            self.assertIn(0.4861, wls_in_rays)
            self.assertIn(0.5876, wls_in_rays)
            self.assertIn(0.6563, wls_in_rays)

    def test_source_editor_point_source_and_divergence(self):
        """Verify SourceEditorWidget point source mode generates diverging rays across wavelengths."""
        self.window.source_config.source_type = SourceType.POINT
        self.window.source_config.divergence_angle = 15.0
        self.window.source_config.num_rays = 9
        self.window.on_trace_rays()

        rays = self.window.dual_view.rays
        self.assertIsNotNone(rays)
        self.assertGreater(len(rays), 0)
        # Check that starting positions and directions in history show diverging fan
        dir_y_vals = []
        for r in rays:
            self.assertGreater(len(r.history), 0)
            h0 = r.history[0]
            self.assertAlmostEqual(h0["x_global"], 0.0)
            self.assertAlmostEqual(h0["y_global"], 0.0)
            dir_y_vals.append(h0["dir_y"])

        # Verify angles cover positive and negative range up to sin(15 deg) ~ 0.2588
        self.assertLess(min(dir_y_vals), -0.20)
        self.assertGreater(max(dir_y_vals), 0.20)


    def test_optimization_with_multi_wavelength_source(self):
        """Verify that on_optimize uses multi-wavelength dispersion to minimize chromatic/geometric blur."""
        if hasattr(self.window, 'wavelengths_input') and self.window.wavelengths_input is not None:
            self.window.wavelengths_input.setText("0.450, 0.550, 0.650")

        result = self.window.on_optimize(variable_indices=[2], method="Nelder-Mead", max_iter=80)
        self.assertTrue(result.success)
        self.assertGreater(result.rms_reduction_pct, 40.0)


class TestM3OptimizerDockAlgorithms(unittest.TestCase):
    """Empirical challenges for Nelder-Mead and L-BFGS-B invocations from OptimizerDock."""

    def setUp(self):
        self.window = MainWindow()
        self.stack = create_test_singlet_defocused()
        self.window.model.set_surface_stack(self.stack)

    def test_nelder_mead_invocation(self):
        """Test Nelder-Mead invocation from dock."""
        dock = self.window.optimizer_dock
        if hasattr(dock, 'method_combo') and dock.method_combo is not None:
            dock.method_combo.setCurrentText("Nelder-Mead")

        dock.set_surface_variable(2, True)
        result = self.window.on_optimize(variable_indices=[2], method="Nelder-Mead", max_iter=80)

        self.assertTrue(result.success)
        self.assertGreaterEqual(result.rms_reduction_pct, 50.0)
        self.assertGreater(result.optimized_thicknesses[2], 0.0)

    def test_l_bfgs_b_invocation(self):
        """Test L-BFGS-B invocation from dock with explicit bounds."""
        dock = self.window.optimizer_dock
        if hasattr(dock, 'method_combo') and dock.method_combo is not None:
            dock.method_combo.setCurrentText("L-BFGS-B")

        dock.set_surface_variable(2, True)
        result = self.window.on_optimize(variable_indices=[2], method="L-BFGS-B", max_iter=50)

        self.assertTrue(result.success)
        self.assertGreaterEqual(result.rms_reduction_pct, 50.0)
        # Check bounds: 10.0 <= t <= 200.0
        t_opt = result.optimized_thicknesses[2]
        self.assertGreaterEqual(t_opt, 10.0)
        self.assertLessEqual(t_opt, 200.0)

    def test_bounds_enforcement_under_extreme_initial_condition(self):
        """Test that optimizer does not violate thickness_min or thickness_max."""
        # Restrict thickness_max to 50.0 (where optimal focus is at ~96.5mm)
        self.stack.surfaces[2].thickness_max = 50.0
        self.window.model.set_surface_stack(self.stack)

        result = self.window.on_optimize(variable_indices=[2], method="L-BFGS-B", max_iter=50)
        self.assertLessEqual(result.optimized_thicknesses[2], 50.0001)


class TestM3PerformanceAndMemoryStress(unittest.TestCase):
    """Stress testing performance and memory footprint over repeated GUI optimization cycles."""

    def test_50_repeated_optimization_cycles(self):
        """
        Run 50 consecutive on_optimize() cycles on MainWindow.
        Verify:
        1. No unhandled exceptions or crashes.
        2. Execution time per cycle remains bounded.
        3. Memory growth (leakage) is strictly bounded (< 25MB total growth).
        """
        window = MainWindow()
        stack = create_test_singlet_defocused()
        window.model.set_surface_stack(stack)
        if hasattr(window, 'wavelengths_input') and window.wavelengths_input is not None:
            window.wavelengths_input.setText("0.5876")

        tracemalloc.start()
        snap_start = tracemalloc.take_snapshot()

        times = []
        n_cycles = 50

        for cycle in range(n_cycles):
            # Reset defocus
            stack.surfaces[2].thickness = 30.0 + (cycle % 5) * 2.0
            window.model.set_surface_stack(stack)

            t0 = time.perf_counter()
            res = window.on_optimize(variable_indices=[2], method="Nelder-Mead", max_iter=30)
            t_elapsed = time.perf_counter() - t0
            times.append(t_elapsed)

            self.assertTrue(res.success or res.final_rms < res.initial_rms)

        snap_end = tracemalloc.take_snapshot()
        stats = snap_end.compare_to(snap_start, 'lineno')
        total_growth_bytes = sum(s.size_diff for s in stats if s.size_diff > 0)
        total_growth_mb = total_growth_bytes / (1024 * 1024)

        tracemalloc.stop()

        avg_time = np.mean(times)
        max_time = np.max(times)

        print(f"\n[PERFORMANCE] 50 Cycles: Avg time = {avg_time*1000:.1f} ms, Max time = {max_time*1000:.1f} ms")
        print(f"[MEMORY] Memory growth over 50 cycles: {total_growth_mb:.2f} MB")

        # Bounds: Avg time per cycle < 500ms, memory growth < 25 MB
        self.assertLess(avg_time, 0.50, f"Average cycle time {avg_time:.3f}s exceeds 500ms limit")
        self.assertLess(total_growth_mb, 25.0, f"Memory growth {total_growth_mb:.2f}MB exceeds 25MB limit")


if __name__ == "__main__":
    unittest.main(verbosity=2)
