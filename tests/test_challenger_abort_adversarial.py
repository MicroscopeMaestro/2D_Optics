import unittest
import numpy as np
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.optimizer import OpticalOptimizer, OptimizationResult
from optics_app.gui.qt_compat import QT_AVAILABLE, QtWidgets
from optics_app.gui.main_window import MainWindow

if QT_AVAILABLE and hasattr(QtWidgets, "QApplication"):
    try:
        _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    except Exception:
        pass

class TestAdversarialAbortAndCancellation(unittest.TestCase):
    def setUp(self):
        # Build standard singlet system
        self.stack = SurfaceStack()
        self.stack.add_surface(Surface(name="OBJ", thickness=100.0))
        self.stack.add_surface(Surface(name="L1", radius=50.0, thickness=5.0, glass="N-BK7", semi_diameter=15.0))
        s_var = Surface(name="L2", radius=0.0, thickness=30.0, glass="AIR", semi_diameter=15.0)
        setattr(s_var, "thickness_variable", True)
        setattr(s_var, "thickness_min", 5.0)
        setattr(s_var, "thickness_max", 150.0)
        self.stack.add_surface(s_var)
        self.stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=20.0))
        self.sys = OpticalSystem(self.stack)

    def test_pre_abort_resets_and_succeeds(self):
        """Test calling opt.stop() before optimize() returns USER_ABORTED and subsequent call succeeds."""
        opt = OpticalOptimizer(self.sys, max_iter=30)
        opt.stop()
        self.assertTrue(opt.abort)
        res = opt.optimize([2])
        self.assertFalse(res.success)
        self.assertFalse(opt.abort)
        self.assertIn("aborted", res.message.lower())
        # Subsequent normal optimization run succeeds
        res2 = opt.optimize([2])
        self.assertTrue(res2.success)
        self.assertIn("successfully", res2.message.lower())

    def test_abort_on_first_progress_callback(self):
        """Test stopping immediately on first progress callback invocation."""
        opt = OpticalOptimizer(self.sys, max_iter=50)
        calls = []
        def on_prog(*args):
            calls.append(args)
            opt.stop()
        opt.progress_callback = on_prog
        res = opt.optimize([2])
        self.assertFalse(res.success)
        self.assertIn("aborted", res.message.lower())
        self.assertGreaterEqual(len(calls), 1)

    def test_abort_on_iteration_threshold(self):
        """Test stopping after reaching specific iteration threshold."""
        opt = OpticalOptimizer(self.sys, max_iter=50)
        iter_count = [0]
        def on_prog(cur=None, tot=None):
            iter_count[0] += 1
            if iter_count[0] >= 3:
                opt.stop()
        opt.progress_callback = on_prog
        res = opt.optimize([2])
        self.assertFalse(res.success)
        self.assertIn("aborted", res.message.lower())
        self.assertGreaterEqual(iter_count[0], 3)
        self.assertIsNotNone(res.final_rms)
        self.assertGreater(res.final_rms, 0.0)

    def test_abort_across_all_solvers(self):
        """Test in-flight cancellation across Nelder-Mead, L-BFGS-B, Powell, SLSQP."""
        solvers = ["Nelder-Mead", "L-BFGS-B", "Powell", "SLSQP"]
        for method in solvers:
            opt = OpticalOptimizer(self.sys, method=method, max_iter=50)
            count = [0]
            def on_prog(*args):
                count[0] += 1
                if count[0] >= 2:
                    opt.stop()
            opt.progress_callback = on_prog
            res = opt.optimize([2], bounds=[(5.0, 150.0)])
            self.assertFalse(res.success, f"Solver {method} should return success=False on abort")
            self.assertIn("aborted", res.message.lower(), f"Solver {method} message should contain 'aborted'")

    def test_repeated_abort_and_restart_cycles(self):
        """Test reusing the same OpticalOptimizer across abort -> run -> abort -> run."""
        opt = OpticalOptimizer(self.sys, max_iter=30)
        
        # Cycle 1: Aborted
        count = [0]
        opt.progress_callback = lambda *a: (count.__setitem__(0, count[0]+1), opt.stop() if count[0]>=2 else None)
        res1 = opt.optimize([2])
        self.assertFalse(res1.success)

        # Cycle 2: Full run (no abort)
        opt.progress_callback = None
        res2 = opt.optimize([2])
        self.assertTrue(res2.success)
        self.assertGreater(res2.rms_reduction_pct, 40.0)

        # Cycle 3: Aborted again
        count = [0]
        opt.progress_callback = lambda *a: (count.__setitem__(0, count[0]+1), opt.stop() if count[0]>=1 else None)
        res3 = opt.optimize([2])
        self.assertFalse(res3.success)

        # Cycle 4: Full run again
        opt.progress_callback = None
        res4 = opt.optimize([2])
        self.assertTrue(res4.success)

    def test_gui_cancel_button_cleanup(self):
        """Test GUI on_optimize flow when cancelled via progress_dialog / active_optimizer.stop()."""
        win = MainWindow()
        win.model.set_surface_stack(self.stack)
        win.optimizer_dock.set_surface_variable(2, True)
        
        # Mock active_optimizer stopping during progress
        count = [0]
        def hooking_progress(cur=None, tot=None):
            count[0] += 1
            if win.active_optimizer is not None and count[0] >= 2:
                win.active_optimizer.stop()
        
        orig_optimize = OpticalOptimizer.optimize
        def wrapped_optimize(self_opt, *args, **kwargs):
            prev_cb = self_opt.progress_callback
            def chained_cb(*cb_args):
                hooking_progress()
                if prev_cb:
                    prev_cb(*cb_args)
            self_opt.progress_callback = chained_cb
            return orig_optimize(self_opt, *args, **kwargs)
        
        OpticalOptimizer.optimize = wrapped_optimize
        try:
            res = win.on_optimize(variable_indices=[2], max_iter=50)
            self.assertFalse(res.success)
            self.assertIn("aborted", res.message.lower())
            # Ensure cleanup occurred in finally block
            self.assertIsNone(win.active_optimizer, "active_optimizer must be reset to None")
            self.assertIn("Optimization failed", getattr(win, "_status_msg", ""))
        finally:
            OpticalOptimizer.optimize = orig_optimize

if __name__ == "__main__":
    unittest.main()
