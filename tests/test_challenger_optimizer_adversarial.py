"""
Challenger R2-2 Adversarial Stress Test Suite for Optimizer Reliability & Physical Boundaries.

Empirically tests optimizer convergence and boundary enforcement under hostile conditions:
1. Zero-variable configurations (empty lists, empty dicts, unconfigured systems)
2. Inverted bounds (b_min >= b_max, degenerate zero-width bounds)
3. Unphysical starting geometries (T_air < 0, T_c < 0, extreme negative values)
4. Self-intersecting lens caps (T_edge <= 0, steep biconvex/meniscus edge clashes)
5. Steep prisms inducing total internal reflection (TIR)
6. Total ray vignetting (pinhole apertures, extreme mirror misalignments, complete ray loss)
7. Nelder-Mead and L-BFGS-B solver safety (no ZeroDivisionError, no crashes, valid TerminationStatus)
8. OptimizationResult pure optical RMS decoupling from numerical barrier penalties
"""

import unittest
import math
import numpy as np

from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from optics_app.engine.optimizer import (
    OpticalOptimizer, OptimizationResult, TerminationStatus, OptimizationAbortedException
)
from optics_app.engine.diagnostics import OptimizationValidator, DiagnosticSeverity, ValidationResult


class TestZeroVariableAdversarial(unittest.TestCase):
    """Hostile Condition 1: Zero-variable configurations."""

    def test_zero_variables_empty_list_nelder_mead(self):
        """Optimizer with empty list variable_indices=[] returns immediately without crash."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=50.0))
        stack.add_surface(Surface(name="L1", radius=40.0, thickness=5.0, glass="N-BK7", semi_diameter=12.0))
        stack.add_surface(Surface(name="L2", radius=-40.0, thickness=30.0, glass="AIR", semi_diameter=12.0))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0))

        sys = OpticalSystem(stack)
        opt = OpticalOptimizer(sys, method="Nelder-Mead")
        res = opt.optimize(variable_indices=[])

        self.assertIsInstance(res, OptimizationResult)
        self.assertTrue(res.success)
        self.assertEqual(res.status, TerminationStatus.NO_VARIABLES)
        self.assertEqual(res.iterations, 0)
        self.assertEqual(res.initial_rms, res.final_rms)
        self.assertEqual(res.rms_reduction_pct, 0.0)
        self.assertEqual(res.optimized_thicknesses, {})
        self.assertEqual(res.optimized_radii, {})

    def test_zero_variables_empty_dict_lbfgsb(self):
        """Optimizer with empty dict variable_indices={} under L-BFGS-B halts cleanly."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=50.0))
        stack.add_surface(Surface(name="L1", radius=40.0, thickness=5.0, glass="N-BK7", semi_diameter=12.0))
        stack.add_surface(Surface(name="L2", radius=-40.0, thickness=30.0, glass="AIR", semi_diameter=12.0))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0))

        sys = OpticalSystem(stack)
        opt = OpticalOptimizer(sys, method="L-BFGS-B")
        res = opt.optimize(variable_indices={"thickness": [], "radius": []})

        self.assertIsInstance(res, OptimizationResult)
        self.assertTrue(res.success)
        self.assertEqual(res.status, TerminationStatus.NO_VARIABLES)
        self.assertEqual(res.iterations, 0)
        self.assertEqual(res.initial_rms, res.final_rms)

    def test_zero_variables_preflight_validation_and_auto_repair(self):
        """Pre-flight check detects zero variables and auto-repair restores air gap variables."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=50.0))
        stack.add_surface(Surface(name="L1", radius=40.0, thickness=5.0, glass="N-BK7", semi_diameter=12.0))
        stack.add_surface(Surface(name="L2", radius=-40.0, thickness=30.0, glass="AIR", semi_diameter=12.0))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0))

        val = OptimizationValidator.validate_system(stack, variable_indices={"thickness": []})
        self.assertFalse(val.is_valid)
        self.assertTrue(val.has_errors)
        self.assertTrue(any(item.code == "NO_VARIABLES_SELECTED" for item in val.items))

        repaired_stack, var_dict, msgs = OptimizationValidator.auto_repair(stack, variable_indices={"thickness": []})
        self.assertGreater(len(var_dict["thickness"]), 0)
        self.assertGreater(len(msgs), 0)


class TestInvertedBoundsAdversarial(unittest.TestCase):
    """Hostile Condition 2: Inverted bounds (b_min >= b_max) & degenerate bounds."""

    def test_inverted_bounds_validation(self):
        """Pre-flight validator identifies inverted bounds (b_min >= b_max) as critical error."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=50.0))
        stack.add_surface(Surface(name="L1", radius=40.0, thickness=5.0, glass="N-BK7", semi_diameter=12.0))
        stack.add_surface(Surface(name="L2", radius=-40.0, thickness=30.0, glass="AIR", semi_diameter=12.0))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0))

        val = OptimizationValidator.validate_system(stack, variable_indices=[2], bounds=[(80.0, 10.0)])
        self.assertFalse(val.is_valid)
        self.assertTrue(any(item.code == "INVERTED_BOUNDS" for item in val.items))

    def test_inverted_bounds_nelder_mead_does_not_crash(self):
        """Nelder-Mead given inverted bounds directly runs safely without raising unhandled exception."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=50.0))
        stack.add_surface(Surface(name="L1", radius=40.0, thickness=5.0, glass="N-BK7", semi_diameter=12.0))
        stack.add_surface(Surface(name="L2", radius=-40.0, thickness=30.0, glass="AIR", semi_diameter=12.0))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0))

        sys = OpticalSystem(stack)
        opt = OpticalOptimizer(sys, method="Nelder-Mead", max_iter=20)
        # Pass inverted bounds
        res = opt.optimize(variable_indices=[2], bounds=[(80.0, 10.0)])
        self.assertIsInstance(res, OptimizationResult)
        self.assertFalse(math.isnan(res.final_rms))
        self.assertFalse(math.isinf(res.final_rms))

    def test_inverted_bounds_lbfgsb_does_not_crash(self):
        """L-BFGS-B given inverted bounds directly runs safely without raising unhandled exception."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=50.0))
        stack.add_surface(Surface(name="L1", radius=40.0, thickness=5.0, glass="N-BK7", semi_diameter=12.0))
        stack.add_surface(Surface(name="L2", radius=-40.0, thickness=30.0, glass="AIR", semi_diameter=12.0))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0))

        sys = OpticalSystem(stack)
        opt = OpticalOptimizer(sys, method="L-BFGS-B", max_iter=20)
        res = opt.optimize(variable_indices=[2], bounds=[(80.0, 10.0)])
        self.assertIsInstance(res, OptimizationResult)
        self.assertFalse(math.isnan(res.final_rms))
        self.assertFalse(math.isinf(res.final_rms))

    def test_degenerate_equal_bounds_both_solvers(self):
        """Degenerate zero-width bounds (b_min == b_max) converge exactly to the fixed bound."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=50.0))
        stack.add_surface(Surface(name="L1", radius=40.0, thickness=5.0, glass="N-BK7", semi_diameter=12.0))
        stack.add_surface(Surface(name="L2", radius=-40.0, thickness=10.0, glass="AIR", semi_diameter=12.0))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0))

        sys = OpticalSystem(stack)
        for method in ["Nelder-Mead", "L-BFGS-B"]:
            opt = OpticalOptimizer(sys, method=method, max_iter=20)
            res = opt.optimize(variable_indices=[2], bounds=[(25.0, 25.0)])
            self.assertIsInstance(res, OptimizationResult)
            self.assertAlmostEqual(res.optimized_thicknesses[2], 25.0, delta=0.5)


class TestUnphysicalStartingGeometriesAdversarial(unittest.TestCase):
    """Hostile Condition 3: Unphysical starting geometries (T_air < 0, T_c < 0)."""

    def test_negative_air_gap_starting_nelder_mead(self):
        """Nelder-Mead recovers from negative air gap T_air = -40.0 mm to physically valid positive gap."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=50.0))
        stack.add_surface(Surface(name="L1", radius=40.0, thickness=5.0, glass="N-BK7", semi_diameter=12.0))
        stack.add_surface(Surface(name="L2", radius=-40.0, thickness=-40.0, glass="AIR", semi_diameter=12.0))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0))

        sys = OpticalSystem(stack)
        opt = OpticalOptimizer(sys, method="Nelder-Mead", num_rays=21, max_iter=40)
        res = opt.optimize(variable_indices=[2], bounds=[(1.0, 80.0)])

        self.assertTrue(res.success)
        self.assertGreaterEqual(res.optimized_thicknesses[2], 1.0)
        self.assertGreaterEqual(res.optimized_stack.surfaces[2].thickness, 1.0)
        self.assertLess(res.final_rms, 10.0)

    def test_negative_air_gap_starting_lbfgsb(self):
        """L-BFGS-B recovers from negative air gap T_air = -30.0 mm to physically valid positive gap."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=50.0))
        stack.add_surface(Surface(name="L1", radius=40.0, thickness=5.0, glass="N-BK7", semi_diameter=12.0))
        stack.add_surface(Surface(name="L2", radius=-40.0, thickness=-30.0, glass="AIR", semi_diameter=12.0))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0))

        sys = OpticalSystem(stack)
        opt = OpticalOptimizer(sys, method="L-BFGS-B", num_rays=21, max_iter=40)
        res = opt.optimize(variable_indices=[2], bounds=[(1.0, 80.0)])

        self.assertTrue(res.success)
        self.assertGreaterEqual(res.optimized_thicknesses[2], 1.0)
        self.assertLess(res.final_rms, 10.0)

    def test_negative_glass_center_thickness_starting(self):
        """Optimizer recovers from negative glass center thickness T_c = -5.0 mm to T_c >= 2.0 mm."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=50.0))
        stack.add_surface(Surface(name="L1", radius=35.0, thickness=-5.0, glass="N-BK7", semi_diameter=12.0))
        stack.add_surface(Surface(name="L2", radius=-35.0, thickness=25.0, glass="AIR", semi_diameter=12.0))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0))

        sys = OpticalSystem(stack)
        opt = OpticalOptimizer(sys, method="Nelder-Mead", num_rays=21, max_iter=30)
        res = opt.optimize(variable_indices={"thickness": [1, 2]}, bounds=[(2.0, 10.0), (5.0, 60.0)])

        self.assertTrue(res.success)
        self.assertGreaterEqual(res.optimized_thicknesses[1], 2.0)
        self.assertGreaterEqual(res.optimized_thicknesses[2], 5.0)

    def test_barrier_penalty_magnitude_on_negative_thickness(self):
        """Merit function evaluation imposes steep quadratic penalty on negative thicknesses."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=50.0))
        stack.add_surface(Surface(name="L1", radius=40.0, thickness=5.0, glass="N-BK7", semi_diameter=12.0))
        stack.add_surface(Surface(name="L2", radius=-40.0, thickness=20.0, glass="AIR", semi_diameter=12.0))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0))

        sys = OpticalSystem(stack)
        opt = OpticalOptimizer(sys)

        pure_rms, penalty = opt._evaluate_merit_and_penalty(np.array([-10.0]), {"thickness": [2]})
        self.assertGreater(penalty, 50000.0)


class TestSelfIntersectingLensCapsAdversarial(unittest.TestCase):
    """Hostile Condition 4: Self-intersecting lens caps (T_edge <= 0)."""

    def test_self_intersecting_lens_detection_and_penalty(self):
        """Merit evaluation flags self-intersecting lens (T_edge <= 0) with heavy barrier penalty."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=60.0))
        stack.add_surface(Surface(name="Front", radius=10.0, thickness=0.5, glass="N-BK7", semi_diameter=8.0))
        stack.add_surface(Surface(name="Back", radius=-10.0, thickness=25.0, glass="AIR", semi_diameter=8.0))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0))

        sys = OpticalSystem(stack)
        opt = OpticalOptimizer(sys)

        pure_rms, penalty = opt._evaluate_merit_and_penalty(np.array([0.5]), {"thickness": [1]})
        self.assertGreater(penalty, 1000.0, "Self-intersecting lens must incur severe penalty")

    def test_self_intersecting_lens_nelder_mead_escape(self):
        """Nelder-Mead escapes self-intersecting starting configuration and yields T_edge > 0."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=80.0))
        stack.add_surface(Surface(name="L1", radius=15.0, thickness=1.0, glass="N-BK7", semi_diameter=12.0))
        stack.add_surface(Surface(name="L2", radius=-15.0, thickness=30.0, glass="AIR", semi_diameter=12.0))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0))

        sys = OpticalSystem(stack)
        opt = OpticalOptimizer(sys, method="Nelder-Mead", num_rays=21, max_iter=40)

        var_dict = {"radius": [1, 2], "thickness": [1, 2]}
        bounds = [(20.0, 100.0), (-100.0, -20.0), (3.0, 10.0), (10.0, 70.0)]
        res = opt.optimize(variable_indices=var_dict, bounds=bounds)

        self.assertTrue(res.success)
        opt_s1 = res.optimized_stack.surfaces[1]
        opt_s2 = res.optimized_stack.surfaces[2]

        r_edge = min(opt_s1.semi_diameter, opt_s2.semi_diameter)
        t_edge = opt_s1.thickness + opt_s2.sag(r_edge) - opt_s1.sag(r_edge)
        self.assertGreater(t_edge, 0.0, "Optimized lens edge thickness must be strictly positive")

    def test_self_intersecting_lens_lbfgsb_escape(self):
        """L-BFGS-B escapes self-intersecting starting configuration and yields T_edge > 0."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=80.0))
        stack.add_surface(Surface(name="L1", radius=15.0, thickness=1.0, glass="N-BK7", semi_diameter=12.0))
        stack.add_surface(Surface(name="L2", radius=-15.0, thickness=30.0, glass="AIR", semi_diameter=12.0))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0))

        sys = OpticalSystem(stack)
        opt = OpticalOptimizer(sys, method="L-BFGS-B", num_rays=21, max_iter=40)

        var_dict = {"radius": [1, 2], "thickness": [1, 2]}
        bounds = [(20.0, 100.0), (-100.0, -20.0), (3.0, 10.0), (10.0, 70.0)]
        res = opt.optimize(variable_indices=var_dict, bounds=bounds)

        self.assertTrue(res.success)
        opt_s1 = res.optimized_stack.surfaces[1]
        opt_s2 = res.optimized_stack.surfaces[2]

        r_edge = min(opt_s1.semi_diameter, opt_s2.semi_diameter)
        t_edge = opt_s1.thickness + opt_s2.sag(r_edge) - opt_s1.sag(r_edge)
        self.assertGreater(t_edge, 0.0, "Optimized lens edge thickness must be strictly positive")


class TestTotalInternalReflectionAdversarial(unittest.TestCase):
    """Hostile Condition 5: Steep prisms inducing total internal reflection (TIR)."""

    def test_steep_prism_tir_nelder_mead_no_crash(self):
        """Nelder-Mead with steep high-index prism (tilt=70 deg) inducing TIR evaluates safely without crash."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=30.0))
        stack.add_surface(Surface(name="PrismF", radius=0.0, thickness=15.0, glass="N-SF11", tilt_x=70.0, semi_diameter=15.0))
        stack.add_surface(Surface(name="PrismB", radius=0.0, thickness=30.0, glass="AIR", tilt_x=-70.0, semi_diameter=15.0))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0))

        sys = OpticalSystem(stack)
        opt = OpticalOptimizer(sys, method="Nelder-Mead", num_rays=21, max_iter=25)
        res = opt.optimize(variable_indices=[2], bounds=[(5.0, 50.0)])

        self.assertIsInstance(res, OptimizationResult)
        self.assertFalse(math.isnan(res.final_rms))
        self.assertFalse(math.isinf(res.final_rms))
        self.assertIn(res.status, [TerminationStatus.CONVERGED, TerminationStatus.RAY_TRACE_FAILURE, TerminationStatus.MAX_ITERATIONS_REACHED])

    def test_steep_prism_tir_lbfgsb_no_crash(self):
        """L-BFGS-B with steep high-index prism (tilt=70 deg) inducing TIR evaluates safely without crash."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=30.0))
        stack.add_surface(Surface(name="PrismF", radius=0.0, thickness=15.0, glass="N-SF11", tilt_x=70.0, semi_diameter=15.0))
        stack.add_surface(Surface(name="PrismB", radius=0.0, thickness=30.0, glass="AIR", tilt_x=-70.0, semi_diameter=15.0))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0))

        sys = OpticalSystem(stack)
        opt = OpticalOptimizer(sys, method="L-BFGS-B", num_rays=21, max_iter=25)
        res = opt.optimize(variable_indices=[2], bounds=[(5.0, 50.0)])

        self.assertIsInstance(res, OptimizationResult)
        self.assertFalse(math.isnan(res.final_rms))
        self.assertFalse(math.isinf(res.final_rms))


class TestTotalRayVignettingAdversarial(unittest.TestCase):
    """Hostile Condition 6: Total ray vignetting (complete ray loss)."""

    def test_extreme_pinhole_vignetting_nelder_mead(self):
        """Pinhole blocking 100% of rays terminates cleanly under Nelder-Mead without division-by-zero."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=50.0))
        stack.add_surface(Surface(name="Pinhole", radius=0.0, thickness=10.0, glass="AIR", semi_diameter=1e-6))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0))

        sys = OpticalSystem(stack)
        opt = OpticalOptimizer(sys, method="Nelder-Mead", num_rays=15, max_iter=15)
        res = opt.optimize(variable_indices=[1], bounds=[(1.0, 50.0)])

        self.assertIsInstance(res, OptimizationResult)
        self.assertFalse(math.isnan(res.final_rms))
        self.assertFalse(math.isinf(res.final_rms))
        self.assertIn(res.status, [TerminationStatus.RAY_TRACE_FAILURE, TerminationStatus.CONVERGED, TerminationStatus.MAX_ITERATIONS_REACHED])

    def test_extreme_pinhole_vignetting_lbfgsb(self):
        """Pinhole blocking 100% of rays terminates cleanly under L-BFGS-B without division-by-zero."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=50.0))
        stack.add_surface(Surface(name="Pinhole", radius=0.0, thickness=10.0, glass="AIR", semi_diameter=1e-6))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0))

        sys = OpticalSystem(stack)
        opt = OpticalOptimizer(sys, method="L-BFGS-B", num_rays=15, max_iter=15)
        res = opt.optimize(variable_indices=[1], bounds=[(1.0, 50.0)])

        self.assertIsInstance(res, OptimizationResult)
        self.assertFalse(math.isnan(res.final_rms))
        self.assertFalse(math.isinf(res.final_rms))

    def test_extreme_misaligned_mirror_vignetting(self):
        """Mirror tilted at 89.5 deg throwing beam away from detector evaluates safely."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=50.0))
        stack.add_surface(Surface(name="Mirror", glass="MIRROR", thickness=30.0, tilt_x=89.5, semi_diameter=15.0))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=10.0))

        sys = OpticalSystem(stack)
        opt = OpticalOptimizer(sys, method="Nelder-Mead", num_rays=15, max_iter=15)
        res = opt.optimize(variable_indices=[1], bounds=[(5.0, 60.0)])

        self.assertIsInstance(res, OptimizationResult)
        self.assertFalse(math.isnan(res.final_rms))


class TestSolverConvergenceAndDecoupledRMS(unittest.TestCase):
    """Hostile Conditions 7 & 8: Solver convergence stability & Pure optical RMS decoupling."""

    def test_pure_optical_rms_decoupling_with_fixed_catalog_lens(self):
        """Verify OptimizationResult outputs pure optical RMS without numerical barrier penalties."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=100.0))
        cat_indices = stack.insert_catalog_lens(0, "AC254-075-B", air_gap_after=60.0)
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=15.0))
        stack.set_all_lenses_fixed()

        sys = OpticalSystem(stack)
        opt = OpticalOptimizer(sys, num_rays=25, max_iter=30)
        res = opt.optimize(variable_indices=[cat_indices[-1]], bounds=[(20.0, 100.0)])

        self.assertTrue(res.success)
        self.assertLess(res.initial_rms, 5.0)
        self.assertLess(res.final_rms, 5.0)
        self.assertGreater(res.rms_reduction_pct, 0.0)
        self.assertEqual(res.status, TerminationStatus.CONVERGED)

    def test_per_wavelength_metrics_accuracy(self):
        """Verify per-wavelength metrics are accurately populated and finite."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=80.0))
        idx = stack.insert_singlet(0, "Achromat", radius_front=40.0, thickness=5.0, glass="N-BK7", radius_back=-40.0, air_gap_after=30.0)
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0))

        wls = [0.486, 0.5876, 0.656]
        sys = OpticalSystem(stack)
        opt = OpticalOptimizer(sys, wavelengths=wls, num_rays=21, max_iter=25)
        res = opt.optimize(variable_indices=[idx[1]], bounds=[(10.0, 60.0)])

        self.assertTrue(res.success)
        self.assertEqual(len(res.per_wavelength_metrics), 3)
        for wl in wls:
            metric = res.per_wavelength_metrics[wl]
            self.assertGreater(metric.valid_rays, 0)
            self.assertLess(metric.final_rms, 10.0)
            self.assertGreater(metric.airy_radius, 0.0)

    def test_joint_multivariable_optimization_both_solvers(self):
        """Joint multi-variable optimization (radii, thicknesses, mirror tilt) succeeds on both solvers."""
        for method in ["Nelder-Mead", "L-BFGS-B"]:
            stack = SurfaceStack()
            stack.add_surface(Surface(name="OBJ", thickness=80.0))
            stack.add_surface(Surface(name="LensF", radius=35.0, thickness=5.0, glass="N-BK7", semi_diameter=12.0))
            stack.add_surface(Surface(name="LensB", radius=-35.0, thickness=20.0, glass="AIR", semi_diameter=12.0))
            stack.add_surface(Surface(name="Mirror", glass="MIRROR", thickness=30.0, tilt_x=43.0, semi_diameter=15.0))
            stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=20.0))

            sys = OpticalSystem(stack)
            opt = OpticalOptimizer(sys, method=method, num_rays=21, max_iter=30)
            var_dict = {"radius": [1], "thickness": [2], "tilt_x": [3]}
            bounds = [(20.0, 70.0), (10.0, 40.0), (38.0, 52.0)]

            res = opt.optimize(variable_indices=var_dict, bounds=bounds)
            self.assertTrue(res.success, f"Method {method} failed to converge")
            self.assertLess(res.final_rms, res.initial_rms)


if __name__ == "__main__":
    unittest.main()
