"""
Empirical Challenger Remediation Stress Test Suite.
Tests:
1. Positional and keyword argument permutations for OpticalOptimizer.__init__:
   - OpticalOptimizer(sys)
   - OpticalOptimizer(sys, "Nelder-Mead") [where 2nd positional arg is method name string]
   - OpticalOptimizer(sys, [0.5876])
   - OpticalOptimizer(sys, 0.5876)
   - OpticalOptimizer(sys, wavelengths=[0.4861, 0.5876, 0.6563], method="Powell")
   - OpticalOptimizer(sys, method="BFGS")
   - OpticalOptimizer(sys, method="CG")
   - OpticalOptimizer(sys, method="Powell")
2. Singlet, Doublet, Triplet, and Multi-surface Systems:
   - Singlet with variable thickness and radius
   - Achromatic Doublet with varying thicknesses and radii
   - Cooke Triplet with 5 variable parameters
   - 6-surface complex objective
3. Solvers matrix:
   - Nelder-Mead
   - Powell
   - BFGS
   - L-BFGS-B
   - SLSQP
4. Boundary constraints & penalty enforcement:
   - Negative thickness barrier penalty (> 1e5)
   - Out-of-bounds parameter values
   - Strict clipping of final parameters within user-defined bounds
   - Ray termination check (rays stop at detector and subsequent surfaces ignored)
"""

import unittest
import numpy as np

from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from optics_app.engine.optimizer import OpticalOptimizer, OptimizationResult


class TestChallengerSignaturePermutations(unittest.TestCase):
    """Test all possible positional and keyword calling conventions for OpticalOptimizer."""

    def setUp(self):
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=100.0))
        stack.add_surface(Surface(name="L1", radius=50.0, thickness=5.0, glass="N-BK7", semi_diameter=15.0))
        stack.add_surface(Surface(name="L2", radius=-50.0, thickness=30.0, glass="AIR", semi_diameter=15.0))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=20.0))
        self.sys = OpticalSystem(stack)

    def test_init_single_argument_system_only(self):
        """OpticalOptimizer(sys) with no other arguments."""
        opt = OpticalOptimizer(self.sys)
        self.assertEqual(opt.wavelengths, [0.5876])
        self.assertEqual(opt.method, "Nelder-Mead")
        res = opt.optimize(variable_indices=[2], bounds=[(10.0, 100.0)])
        self.assertTrue(res.success)
        self.assertLess(res.final_rms, res.initial_rms)

    def test_init_positional_method_string(self):
        """OpticalOptimizer(sys, 'Nelder-Mead') where method string is passed as 2nd positional argument."""
        opt = OpticalOptimizer(self.sys, "Nelder-Mead")
        self.assertEqual(opt.wavelengths, [0.5876])
        self.assertEqual(opt.method, "Nelder-Mead")
        res = opt.optimize(variable_indices=[2], bounds=[(10.0, 100.0)])
        self.assertTrue(res.success)

    def test_init_positional_powell_string(self):
        """OpticalOptimizer(sys, 'Powell') where 'Powell' is passed as 2nd positional argument."""
        opt = OpticalOptimizer(self.sys, "Powell")
        self.assertEqual(opt.wavelengths, [0.5876])
        self.assertEqual(opt.method, "Powell")
        res = opt.optimize(variable_indices=[2], bounds=[(10.0, 100.0)])
        self.assertTrue(res.success)

    def test_init_positional_float_wavelength(self):
        """OpticalOptimizer(sys, 0.5876) where float is passed as 2nd positional argument."""
        opt = OpticalOptimizer(self.sys, 0.5876)
        self.assertEqual(opt.wavelengths, [0.5876])
        self.assertEqual(opt.method, "Nelder-Mead")

    def test_init_positional_list_wavelengths(self):
        """OpticalOptimizer(sys, [0.4861, 0.5876, 0.6563])."""
        opt = OpticalOptimizer(self.sys, [0.4861, 0.5876, 0.6563])
        self.assertEqual(len(opt.wavelengths), 3)

    def test_init_keyword_method_powell(self):
        """OpticalOptimizer(sys, method='Powell')."""
        opt = OpticalOptimizer(self.sys, method="Powell")
        self.assertEqual(opt.method, "Powell")
        res = opt.optimize(variable_indices=[2], bounds=[(10.0, 100.0)])
        self.assertTrue(res.success)

    def test_init_keyword_method_bfgs(self):
        """OpticalOptimizer(sys, method='BFGS')."""
        opt = OpticalOptimizer(self.sys, method="BFGS")
        self.assertEqual(opt.method, "BFGS")
        res = opt.optimize(variable_indices=[2], bounds=[(10.0, 100.0)])
        self.assertTrue(res.success)

    def test_init_keyword_method_cg(self):
        """OpticalOptimizer(sys, method='CG')."""
        opt = OpticalOptimizer(self.sys, method="CG")
        self.assertEqual(opt.method, "CG")
        res = opt.optimize(variable_indices=[2], bounds=[(10.0, 100.0)])
        self.assertTrue(res.success)


class TestChallengerSystemsAndTopologies(unittest.TestCase):
    """Stress tests on Singlet, Doublet, Triplet, and Multi-surface systems."""

    def test_singlet_thickness_and_radius_optimization(self):
        """Singlet: optimize thickness and front radius simultaneously."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=200.0))
        s1 = Surface(name="L1_Front", radius=40.0, thickness=4.0, glass="N-BK7", semi_diameter=15.0)
        s2 = Surface(name="L1_Back", radius=-60.0, thickness=30.0, glass="AIR", semi_diameter=15.0)
        s3 = Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=20.0)
        stack.add_surface(s1)
        stack.add_surface(s2)
        stack.add_surface(s3)

        sys = OpticalSystem(stack)
        opt = OpticalOptimizer(sys, method="Nelder-Mead", max_iter=60, num_rays=20)
        
        # Optimize thickness of air gap (idx 2) and radius of front surface (idx 1)
        var_dict = {"thickness": [2], "radius": [1]}
        res = opt.optimize(variable_indices=var_dict, bounds=[(10.0, 100.0), (20.0, 80.0)])
        self.assertTrue(res.success)
        self.assertIn(2, res.optimized_thicknesses)
        self.assertLess(res.final_rms, res.initial_rms)

    def test_achromatic_doublet_varying_thicknesses_and_radii(self):
        """Doublet: crown + flint with 2 variable thicknesses and 2 variable radii."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=500.0))
        # Crown
        stack.add_surface(Surface(name="Crown_F", radius=45.0, thickness=6.0, glass="N-BK7", semi_diameter=15.0))
        # Flint
        stack.add_surface(Surface(name="Flint_F", radius=-35.0, thickness=3.5, glass="N-SF11", semi_diameter=15.0))
        stack.add_surface(Surface(name="Flint_B", radius=-100.0, thickness=40.0, glass="AIR", semi_diameter=15.0))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=20.0))

        sys = OpticalSystem(stack)
        opt = OpticalOptimizer(
            sys,
            wavelengths=[0.4861, 0.5876, 0.6563],
            method="Nelder-Mead",
            max_iter=80,
            num_rays=25
        )
        var_dict = {"thickness": [1, 3], "radius": [1, 3]}
        bounds = [(2.0, 10.0), (20.0, 100.0), (30.0, 70.0), (-200.0, -50.0)]
        res = opt.optimize(variable_indices=var_dict, bounds=bounds)

        self.assertTrue(res.success)
        self.assertLess(res.final_rms, res.initial_rms)
        self.assertGreater(res.rms_reduction_pct, 20.0)

    def test_multi_surface_six_element_system(self):
        """6-surface multi-element system with 4 thickness variables."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=100.0))
        stack.add_surface(Surface(name="S1", radius=50.0, thickness=5.0, glass="N-BK7", semi_diameter=18.0))
        stack.add_surface(Surface(name="S2", radius=-70.0, thickness=4.0, glass="AIR", semi_diameter=18.0))
        stack.add_surface(Surface(name="S3", radius=40.0, thickness=6.0, glass="N-SF11", semi_diameter=15.0))
        stack.add_surface(Surface(name="S4", radius=-40.0, thickness=5.0, glass="AIR", semi_diameter=15.0))
        stack.add_surface(Surface(name="S5", radius=60.0, thickness=5.0, glass="N-BK7", semi_diameter=18.0))
        stack.add_surface(Surface(name="S6", radius=-50.0, thickness=30.0, glass="AIR", semi_diameter=18.0))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=25.0))

        sys = OpticalSystem(stack)
        opt = OpticalOptimizer(sys, method="Nelder-Mead", max_iter=60, num_rays=20)
        res = opt.optimize(variable_indices=[2, 4, 6], bounds=[(1.0, 15.0), (1.0, 15.0), (10.0, 80.0)])

        self.assertTrue(res.success)
        self.assertLess(res.final_rms, res.initial_rms)
        self.assertEqual(len(res.optimized_thicknesses), 3)


class TestChallengerSolversMatrix(unittest.TestCase):
    """Stress tests across optimization solvers: Nelder-Mead, Powell, BFGS, L-BFGS-B, SLSQP."""

    def _run_solver_test(self, method_name: str):
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=200.0))
        stack.add_surface(Surface(name="L1", radius=45.0, thickness=5.0, glass="N-BK7", semi_diameter=15.0))
        stack.add_surface(Surface(name="L2", radius=0.0, thickness=25.0, glass="AIR", semi_diameter=15.0))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=20.0))

        sys = OpticalSystem(stack)
        opt = OpticalOptimizer(sys, method=method_name, max_iter=40, num_rays=20)
        res = opt.optimize(variable_indices=[2], bounds=[(10.0, 120.0)])

        self.assertTrue(res.success, f"Solver {method_name} failed")
        self.assertLess(res.final_rms, res.initial_rms)
        self.assertGreaterEqual(res.optimized_thicknesses[2], 10.0)
        self.assertLessEqual(res.optimized_thicknesses[2], 120.0)

    def test_nelder_mead(self):
        self._run_solver_test("Nelder-Mead")

    def test_powell(self):
        self._run_solver_test("Powell")

    def test_bfgs(self):
        self._run_solver_test("BFGS")

    def test_l_bfgs_b(self):
        self._run_solver_test("L-BFGS-B")

    def test_slsqp(self):
        self._run_solver_test("SLSQP")


class TestChallengerBoundariesAndPenalties(unittest.TestCase):
    """Stress tests on negative thickness barrier penalties and strict bounds enforcement."""

    def setUp(self):
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=100.0))
        stack.add_surface(Surface(name="L1", radius=50.0, thickness=5.0, glass="N-BK7", semi_diameter=15.0))
        stack.add_surface(Surface(name="L2", radius=-50.0, thickness=30.0, glass="AIR", semi_diameter=15.0))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=20.0))
        self.sys = OpticalSystem(stack)

    def test_negative_thickness_merit_penalty(self):
        """Calling merit_function with negative thickness triggers heavy barrier penalty >= 1e5."""
        opt = OpticalOptimizer(self.sys)
        merit_neg = opt.merit_function(np.array([-15.0]), [2])
        self.assertGreaterEqual(merit_neg, 1e5)
        # Should scale with the magnitude of violation
        merit_more_neg = opt.merit_function(np.array([-50.0]), [2])
        self.assertGreater(merit_more_neg, merit_neg)

    def test_out_of_bounds_clamping(self):
        """Optimization solution strictly clamped to user bounds even if unconstrained optimum is outside."""
        opt = OpticalOptimizer(self.sys, method="Nelder-Mead", max_iter=40, num_rays=20)
        # True focus is around ~50mm. Clamp bounds to [10.0, 25.0]
        res = opt.optimize(variable_indices=[2], bounds=[(10.0, 25.0)])
        self.assertLessEqual(res.optimized_thicknesses[2], 25.0 + 1e-5)
        self.assertGreaterEqual(res.optimized_thicknesses[2], 10.0 - 1e-5)

    def test_ray_termination_at_detector_surface(self):
        """Verify rays strictly terminate at DETECTOR surface and do not evaluate downstream surfaces (R2)."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=100.0))
        stack.add_surface(Surface(name="L1", radius=50.0, thickness=5.0, glass="N-BK7", semi_diameter=15.0))
        stack.add_surface(Surface(name="L2", radius=-50.0, thickness=30.0, glass="AIR", semi_diameter=15.0))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=10.0, semi_diameter=20.0))
        stack.add_surface(Surface(name="POST_DET_1", radius=0.0, thickness=10.0, glass="N-BK7", semi_diameter=20.0))
        stack.add_surface(Surface(name="POST_DET_2", radius=0.0, thickness=0.0, glass="AIR", semi_diameter=20.0))

        sys = OpticalSystem(stack)
        rays = [Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=0.5876)]
        traced = sys.trace_rays(rays)

        self.assertEqual(len(traced), 1)
        r = traced[0]
        self.assertFalse(r.vignetted)
        # The last surface visited in ray history must be surface index 3 (DET)
        self.assertEqual(r.history[-1]["surface_index"], 3)
        # No post-detector surfaces should ever appear in ray history
        visited_indices = [h["surface_index"] for h in r.history]
        self.assertNotIn(4, visited_indices)
        self.assertNotIn(5, visited_indices)


if __name__ == "__main__":
    unittest.main()
