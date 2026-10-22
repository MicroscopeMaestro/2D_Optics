"""
Comprehensive 4-Tier Test Suite for Optical Optimizer Boundary Conditions & Reliability.

Tiers Covered:
- Tier 1: Non-negative air gaps, center thickness minimum (Tc >= T_min,glass > 0).
- Tier 2: Analytical edge thickness (T_edge = Tc + z2(r) - z1(r) >= T_min,edge > 0), lens self-intersection detection.
- Tier 3: Ray trace crash prevention (TIR, total vignetting, unphysical starting geometries) converging or halting gracefully with clean termination status.
- Tier 4: Decoupled pure optical RMS reporting and joint optimization of radii, air gaps, and mirror tilts.
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
from optics_app.engine.diagnostics import OptimizationValidator, DiagnosticSeverity


class TestTier1ThicknessAndAirGapBoundaries(unittest.TestCase):
    """
    Tier 1: Non-negative air gaps and center thickness minimums (Tc >= T_min,glass > 0).
    """

    def test_non_negative_air_gap_penalty_enforcement(self):
        """Tier 1: Optimizer severely penalizes negative air gaps, keeping air gap >= 0."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=50.0))
        stack.add_surface(Surface(name="L1", radius=40.0, thickness=5.0, glass="N-BK7", semi_diameter=12.0))
        stack.add_surface(Surface(name="L2", radius=-40.0, thickness=20.0, glass="AIR", semi_diameter=12.0))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0))

        sys = OpticalSystem(stack)
        opt = OpticalOptimizer(sys, num_rays=21, max_iter=30)

        # Test merit function evaluation directly with negative thickness
        pure_rms, penalty = opt._evaluate_merit_and_penalty(np.array([-5.0]), {"thickness": [2]})
        self.assertGreater(penalty, 1000.0, "Negative air gap must incur significant barrier penalty")

        # Run optimizer with lower bound at 0.5 mm
        res = opt.optimize(variable_indices=[2], bounds=[(0.5, 100.0)])
        self.assertTrue(res.success)
        self.assertGreaterEqual(res.optimized_thicknesses[2], 0.5)

    def test_glass_center_thickness_minimum(self):
        """Tier 1: Glass center thickness minimum Tc >= T_min,glass > 0 enforced."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=60.0))
        # Variable singlet glass thickness
        stack.add_surface(Surface(name="L1", radius=35.0, thickness=4.0, glass="N-BK7", semi_diameter=12.0, thickness_variable=True))
        stack.add_surface(Surface(name="L2", radius=-35.0, thickness=25.0, glass="AIR", semi_diameter=12.0, thickness_variable=True))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0))

        sys = OpticalSystem(stack)
        opt = OpticalOptimizer(sys, num_rays=21, max_iter=30)

        # Test evaluating zero or near-zero glass thickness incurs heavy penalty
        pure_rms, pen_zero = opt._evaluate_merit_and_penalty(np.array([0.001, 25.0]), {"thickness": [1, 2]})
        self.assertGreater(pen_zero, 100.0, "Near-zero glass thickness must incur penalty")

        # Optimize with glass thickness bound [1.0, 15.0]
        res = opt.optimize(variable_indices={"thickness": [1, 2]}, bounds=[(1.0, 15.0), (5.0, 80.0)])
        self.assertTrue(res.success)
        self.assertGreaterEqual(res.optimized_thicknesses[1], 1.0)
        self.assertGreaterEqual(res.optimized_thicknesses[2], 5.0)

    def test_multiple_air_gap_boundaries(self):
        """Tier 1: Multi-element system keeps all intermediate air gaps non-negative."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=40.0))
        # Singlet 1
        stack.add_surface(Surface(name="L1_F", radius=50.0, thickness=5.0, glass="N-BK7", semi_diameter=12.0))
        stack.add_surface(Surface(name="L1_B", radius=-50.0, thickness=15.0, glass="AIR", semi_diameter=12.0))
        # Singlet 2
        stack.add_surface(Surface(name="L2_F", radius=30.0, thickness=4.0, glass="N-BK7", semi_diameter=12.0))
        stack.add_surface(Surface(name="L2_B", radius=-30.0, thickness=20.0, glass="AIR", semi_diameter=12.0))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0))

        sys = OpticalSystem(stack)
        opt = OpticalOptimizer(sys, num_rays=21, max_iter=25)

        # Optimize both air gaps: surface 2 and surface 4
        res = opt.optimize(variable_indices=[2, 4], bounds=[(1.0, 50.0), (1.0, 50.0)])
        self.assertTrue(res.success)
        self.assertGreaterEqual(res.optimized_thicknesses[2], 1.0)
        self.assertGreaterEqual(res.optimized_thicknesses[4], 1.0)


class TestTier2EdgeThicknessAndSelfIntersection(unittest.TestCase):
    """
    Tier 2: Analytical edge thickness T_edge = Tc + z2(r) - z1(r) >= T_min,edge > 0
    and lens self-intersection prevention.
    """

    def test_analytical_edge_thickness_formula_biconvex(self):
        """Tier 2: Analytical edge thickness formula calculates exact physical clearance on biconvex lens."""
        r = 10.0
        r1 = 25.0
        r2 = -25.0
        tc = 5.0

        # Sag formulas: z = (1/R * r^2) / (1 + sqrt(1 - (1/R)^2 * r^2))
        s1 = Surface(radius=r1, thickness=tc, glass="N-BK7", semi_diameter=r)
        s2 = Surface(radius=r2, thickness=20.0, glass="AIR", semi_diameter=r)

        sag1 = s1.sag(r)
        sag2 = s2.sag(r)

        # For biconvex lens (r1 > 0, r2 < 0):
        # Front surface bulges forward (+z), back surface bulges backward (-z).
        # T_edge = Tc + sag2 - sag1
        t_edge = tc + sag2 - sag1
        self.assertLess(t_edge, tc) # Biconvex edge is thinner than center
        self.assertGreater(t_edge, 0.0)

    def test_biconvex_self_intersection_penalty(self):
        """Tier 2: Steep biconvex lens where edges self-intersect (T_edge <= 0) incurs barrier penalty."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=50.0))
        # Very thin steep biconvex lens: R1=12, R2=-12, semi_diameter=10, Tc=1.0 mm -> T_edge < 0!
        stack.add_surface(Surface(name="SteepFront", radius=12.0, thickness=1.0, glass="N-BK7", semi_diameter=10.0))
        stack.add_surface(Surface(name="SteepBack", radius=-12.0, thickness=30.0, glass="AIR", semi_diameter=10.0))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0))

        s1 = stack.surfaces[1]
        s2 = stack.surfaces[2]
        sag1 = s1.sag(10.0)
        sag2 = s2.sag(10.0)
        edge_thick = 1.0 + sag2 - sag1
        self.assertLess(edge_thick, 0.0, "This configuration must produce a self-intersecting lens")

        sys = OpticalSystem(stack)
        opt = OpticalOptimizer(sys, num_rays=21)

        # When surfaces are variable, penalty must be triggered
        pure_rms, penalty = opt._evaluate_merit_and_penalty(np.array([1.0]), {"thickness": [1]})
        self.assertGreater(penalty, 1000.0, "Self-intersecting lens edge must incur severe barrier penalty")

    def test_joint_curvature_and_thickness_prevents_self_intersection(self):
        """Tier 2: Joint optimization of radii and thicknesses converges to physical lens with T_edge > 0."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=80.0))
        stack.add_surface(Surface(name="L1", radius=30.0, thickness=4.0, glass="N-BK7", semi_diameter=10.0))
        stack.add_surface(Surface(name="L2", radius=-30.0, thickness=35.0, glass="AIR", semi_diameter=10.0))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0))

        sys = OpticalSystem(stack)
        opt = OpticalOptimizer(sys, num_rays=21, max_iter=30)

        # Optimize radii and glass thickness
        var_dict = {"radius": [1, 2], "thickness": [1, 2]}
        bounds = [(15.0, 100.0), (-100.0, -15.0), (2.0, 10.0), (10.0, 60.0)]
        res = opt.optimize(variable_indices=var_dict, bounds=bounds)

        self.assertTrue(res.success)
        opt_s1 = res.optimized_stack.surfaces[1]
        opt_s2 = res.optimized_stack.surfaces[2]

        # Verify center thickness
        self.assertGreaterEqual(opt_s1.thickness, 2.0)

        # Verify edge thickness
        sag1 = opt_s1.sag(10.0)
        sag2 = opt_s2.sag(10.0)
        t_edge = opt_s1.thickness + sag2 - sag1
        self.assertGreater(t_edge, 0.0, "Final optimized lens must have physically valid positive edge thickness")


class TestTier3RayTraceCrashPreventionAndTermination(unittest.TestCase):
    """
    Tier 3: Ray trace crash prevention (TIR, total vignetting, unphysical starting geometries)
    converging or halting gracefully with clean termination status.
    """

    def test_total_internal_reflection_does_not_crash(self):
        """Tier 3: Steep prism or high incident angle triggering TIR does not crash optimizer."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=30.0))
        # Prism tilted at 60 deg causing extreme internal incident angles
        stack.add_surface(Surface(name="PrismF", radius=0.0, thickness=15.0, glass="N-SF11", tilt_x=60.0, semi_diameter=15.0))
        stack.add_surface(Surface(name="PrismB", radius=0.0, thickness=30.0, glass="AIR", tilt_x=-60.0, semi_diameter=15.0))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0))

        sys = OpticalSystem(stack)
        opt = OpticalOptimizer(sys, num_rays=21, max_iter=20)

        # Optimize without throwing exception
        res = opt.optimize(variable_indices=[2], bounds=[(5.0, 50.0)])
        self.assertIsNotNone(res)
        self.assertFalse(math.isnan(res.final_rms))
        self.assertFalse(math.isinf(res.final_rms))

    def test_total_vignetting_starts_and_halts_safely(self):
        """Tier 3: Extreme starting geometry with 100% ray vignetting returns clean status without exception."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=50.0))
        # Tiny pinhole semi_diameter = 0.001 mm blocking 100% of rays
        stack.add_surface(Surface(name="Blocker", radius=0.0, thickness=10.0, glass="AIR", semi_diameter=0.001))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0))

        sys = OpticalSystem(stack)
        opt = OpticalOptimizer(sys, num_rays=15, max_iter=15)

        res = opt.optimize(variable_indices=[1], bounds=[(1.0, 50.0)])
        self.assertIsNotNone(res)
        # Should report clean failure status or handle gracefully
        self.assertIn(res.status, [TerminationStatus.RAY_TRACE_FAILURE, TerminationStatus.CONVERGED, TerminationStatus.MAX_ITERATIONS_REACHED])

    def test_unphysical_inverted_bounds_validation_and_auto_repair(self):
        """Tier 3: Inverted bounds (b_min >= b_max) detected by pre-flight check and auto-repaired."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=40.0))
        s = Surface(name="L1", radius=50.0, thickness=5.0, glass="N-BK7")
        stack.add_surface(s)
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0))

        sys = OpticalSystem(stack)
        opt = OpticalOptimizer(sys)

        # Pre-flight check detects inverted bounds
        val = opt.validate(variable_indices=[1], bounds=[(50.0, 10.0)])
        self.assertFalse(val.is_valid)
        self.assertTrue(any(item.code == "INVERTED_BOUNDS" for item in val.items))

        # Auto-repair repairs zero variable state and recovers valid stack
        repaired_stack, rep_vars, msgs = opt.auto_repair(variable_indices={"thickness": [], "radius": []})
        self.assertIsNotNone(repaired_stack)
        self.assertGreater(len(msgs), 0)
        self.assertGreater(len(rep_vars.get("thickness", [])), 0)

    def test_abort_signal_and_state_reset(self):
        """Tier 3: User abort cleanly halts optimization; next run resets abort flag automatically."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=60.0))
        stack.add_surface(Surface(name="L1", radius=45.0, thickness=5.0, glass="N-BK7", semi_diameter=15.0))
        stack.add_surface(Surface(name="L2", radius=-45.0, thickness=30.0, glass="AIR", semi_diameter=15.0))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0))

        sys = OpticalSystem(stack)
        opt = OpticalOptimizer(sys, max_iter=50)

        # Abort during run via callback
        def abort_cb(current=0, total=50):
            if current >= 2:
                opt.stop()

        opt.progress_callback = abort_cb
        res_abort = opt.optimize(variable_indices=[2], bounds=[(5.0, 100.0)])

        self.assertEqual(res_abort.status, TerminationStatus.USER_ABORTED)
        self.assertFalse(res_abort.success)

        # Second run without abort callback resets abort flag and succeeds
        opt.progress_callback = None
        res_success = opt.optimize(variable_indices=[2], bounds=[(5.0, 100.0)])
        self.assertTrue(res_success.success)
        self.assertEqual(res_success.status, TerminationStatus.CONVERGED)
        self.assertFalse(opt.abort)


class TestTier4DecoupledRMSAndJointOptimization(unittest.TestCase):
    """
    Tier 4: Decoupled pure optical RMS reporting and joint optimization of radii, air gaps, and mirror tilts.
    """

    def test_decoupled_pure_optical_rms_reporting(self):
        """Tier 4: OptimizationResult reports pure optical RMS (< 10 mm) decoupled from numerical penalties."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=100.0))
        # Insert fixed catalog doublet AC254-050-B
        cat_indices = stack.insert_catalog_lens(0, "AC254-050-B", air_gap_after=40.0)
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=15.0))
        stack.set_all_lenses_fixed()

        sys = OpticalSystem(stack)
        opt = OpticalOptimizer(sys, num_rays=25, max_iter=30)

        # Optimize only air gap after doublet
        res = opt.optimize(variable_indices=[cat_indices[-1]], bounds=[(10.0, 80.0)])

        self.assertTrue(res.success)
        # Verify pure optical RMS values
        self.assertLess(res.initial_rms, 10.0, "Initial RMS must be pure optical spot radius (mm), not penalty")
        self.assertLess(res.final_rms, 10.0, "Final RMS must be pure optical spot radius (mm)")
        self.assertGreater(res.rms_reduction_pct, 0.0)
        # Solver penalty stored separately
        self.assertIsInstance(res.penalty_value, float)

    def test_joint_radii_airgap_mirror_tilt_optimization(self):
        """Tier 4: Joint optimization of lens radii, propagation air gap, and fold mirror tilt."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=80.0))

        # Singlet lens with variable front radius
        stack.add_surface(Surface(name="LensFront", radius=40.0, thickness=6.0, glass="N-BK7", semi_diameter=15.0))
        stack.add_surface(Surface(name="LensBack", radius=-40.0, thickness=25.0, glass="AIR", semi_diameter=15.0))

        # Folding Mirror with initial slight misalignment
        mirror = Surface(name="FoldMirror", glass="MIRROR", thickness=35.0, tilt_x=42.0, semi_diameter=15.0)
        stack.add_surface(mirror)

        # Detector
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=25.0))

        sys = OpticalSystem(stack)
        opt = OpticalOptimizer(sys, num_rays=25, max_iter=40)

        # Joint variables: radius of S1, thickness of S2 (air gap), tilt_x of S3 (mirror)
        var_dict = {
            "radius": [1],
            "thickness": [2],
            "tilt_x": [3]
        }
        bounds = [(20.0, 80.0), (10.0, 50.0), (35.0, 55.0)]

        res = opt.optimize(variable_indices=var_dict, bounds=bounds)

        self.assertTrue(res.success)
        self.assertIn(1, res.optimized_radii)
        self.assertIn(2, res.optimized_thicknesses)
        self.assertIn(3, res.optimized_tilts)
        self.assertLess(res.final_rms, res.initial_rms)

    def test_polychromatic_multi_wavelength_optimization(self):
        """Tier 4: Polychromatic optimization simultaneously minimizes spot radius across 3 wavelengths."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=100.0))
        indices = stack.insert_singlet(0, "AchromatSinglet", radius_front=45.0, thickness=6.0, glass="N-BK7", radius_back=-45.0, air_gap_after=35.0)
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0))

        wls = [0.486, 0.5876, 0.656] # F, d, C lines
        sys = OpticalSystem(stack)
        opt = OpticalOptimizer(sys, wavelengths=wls, num_rays=21, max_iter=30)

        res = opt.optimize(variable_indices=[indices[1]], bounds=[(10.0, 70.0)])
        self.assertTrue(res.success)
        self.assertEqual(len(res.per_wavelength_metrics), 3)
        for wl in wls:
            self.assertIn(wl, res.per_wavelength_metrics)
            metric = res.per_wavelength_metrics[wl]
            self.assertGreater(metric.valid_rays, 0)
            self.assertLess(metric.final_rms, 10.0)


if __name__ == "__main__":
    unittest.main()
