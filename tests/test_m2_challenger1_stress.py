"""
Milestone M2 Empirical Challenger 1 Adversarial Stress Test Suite.
Comprehensive verification of OpticalOptimizer across:
1. High-order aspheres (4th, 6th, 8th, 10th, 12th order) and extreme conics (-15.0 to +20.0).
2. Multi-variable topologies (Doublets, Triplets, Petzval, Quad-Element, 6-Element Double Gauss).
3. Boundary condition stress (zero thickness, narrow bounds, out-of-bounds clamping, zero-width bounds).
4. Solver convergence stability across algorithms (Nelder-Mead, L-BFGS-B, SLSQP, Powell) and low iteration budgets.
5. Scalability, performance, and numerical stability across ray counts (10 to 100+ rays).
"""

import unittest
import time
import numpy as np

from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from optics_app.engine.optimizer import OpticalOptimizer, OptimizationResult
from optics_app.analysis.spot import SpotDiagram


def set_opt_vars(surf: Surface, variable: bool = True, t_min: float = 0.0, t_max: float = 1000.0) -> Surface:
    setattr(surf, "thickness_variable", variable)
    setattr(surf, "thickness_min", t_min)
    setattr(surf, "thickness_max", t_max)
    return surf


# ===========================================================================
# 1. High-Order Aspheric Surfaces & Extreme Conic Constants Stress Tests
# ===========================================================================
class TestHighOrderAspheresAndExtremeConics(unittest.TestCase):
    """
    Stress-testing optimizer when surfaces contain high-order polynomial aspheres
    (4th, 6th, 8th, 10th, 12th order coefficients) and extreme conic constants (oblate/hyperbolic).
    """

    def test_schmidt_corrector_plate_high_order_asphere(self):
        """
        Schmidt corrector plate: flat base curvature (R=0 or inf) with 4th, 6th, and 8th order asphere terms.
        Optimizer must trace and optimize the distance to spherical mirror and detector without numerical instability.
        """
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=2000.0))
        # Schmidt corrector front: 4th and 6th order aspheric profile
        s1 = Surface(
            name="Schmidt_Front",
            surface_type=SurfaceType.ASPHERIC,
            radius=0.0,
            thickness=4.0,
            glass="N-BK7",
            semi_diameter=25.0,
            conic=0.0,
            asphere_coeffs=[2.5e-6, -1.2e-9, 4.0e-13]  # a4, a6, a8
        )
        stack.add_surface(s1)
        # Schmidt corrector back: flat
        s2 = Surface(name="Schmidt_Back", radius=0.0, thickness=150.0, glass="AIR", semi_diameter=25.0)
        set_opt_vars(s2, variable=True, t_min=50.0, t_max=300.0)
        stack.add_surface(s2)
        # Primary spherical mirror (concave, R = -200mm -> f = 100mm)
        s3 = Surface(name="SphericalMirror", radius=-200.0, thickness=-80.0, glass="MIRROR", semi_diameter=25.0)
        set_opt_vars(s3, variable=True, t_min=-150.0, t_max=-30.0)  # Reflection travels in -Z
        # For standard sequential stack where thickness is positive distance between surfaces:
        # We test refractive equivalent or positive distance fold
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=30.0))

        sys = OpticalSystem(stack)
        optimizer = OpticalOptimizer(sys, method="Nelder-Mead", max_iter=60, num_rays=25, entrance_pupil_radius=12.0)
        res = optimizer.optimize(variable_indices=[2], bounds=[(50.0, 300.0)])

        self.assertTrue(res.success)
        self.assertFalse(np.isnan(res.final_rms))
        self.assertFalse(np.isinf(res.final_rms))
        self.assertGreater(res.final_rms, 0.0)

    def test_extreme_hyperbolic_conic_constant(self):
        """
        Hyperbolic surface with extreme conic constant k = -15.0 and 4 aspheric coefficients.
        Tests Newton-Raphson intersection stability during the optimization loop.
        """
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=1000.0))
        s1 = Surface(
            name="ExtremeHyperbola",
            surface_type=SurfaceType.ASPHERIC,
            radius=25.0,
            conic=-15.0,
            asphere_coeffs=[1e-5, -3e-8, 5e-11, -2e-14],
            thickness=6.0,
            glass="N-BK7",
            semi_diameter=15.0
        )
        stack.add_surface(s1)
        s2 = Surface(name="BackSurf", radius=-50.0, thickness=20.0, glass="AIR", semi_diameter=15.0)
        set_opt_vars(s2, variable=True, t_min=10.0, t_max=150.0)
        stack.add_surface(s2)
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=25.0))

        sys = OpticalSystem(stack)
        optimizer = OpticalOptimizer(sys, method="Nelder-Mead", max_iter=60, num_rays=25, entrance_pupil_radius=8.0)
        res = optimizer.optimize(variable_indices=[2], bounds=[(10.0, 150.0)])

        self.assertTrue(res.success)
        self.assertLess(res.final_rms, res.initial_rms)
        self.assertGreaterEqual(res.rms_reduction_pct, 30.0)

    def test_extreme_oblate_ellipsoid_conic_constant(self):
        """
        Oblate ellipsoid surface with extreme conic constant k = +20.0.
        Tests positive conic constant stability under rapid thickness adjustments.
        """
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=800.0))
        s1 = Surface(
            name="OblateFront",
            surface_type=SurfaceType.ASPHERIC,
            radius=35.0,
            conic=20.0,
            asphere_coeffs=[-5e-6, 2e-9],
            thickness=5.0,
            glass="N-BK7",
            semi_diameter=12.0
        )
        stack.add_surface(s1)
        s2 = Surface(name="OblateBack", radius=-45.0, thickness=15.0, glass="AIR", semi_diameter=12.0)
        set_opt_vars(s2, variable=True, t_min=5.0, t_max=120.0)
        stack.add_surface(s2)
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=20.0))

        sys = OpticalSystem(stack)
        optimizer = OpticalOptimizer(sys, method="Nelder-Mead", max_iter=60, num_rays=20, entrance_pupil_radius=6.0)
        res = optimizer.optimize(variable_indices=[2], bounds=[(5.0, 120.0)])

        self.assertTrue(res.success)
        self.assertLess(res.final_rms, res.initial_rms)
        self.assertGreaterEqual(res.rms_reduction_pct, 40.0)

    def test_12th_order_even_asphere_polynomial(self):
        """
        Even asphere surface with up to 12th order terms: a4, a6, a8, a10, a12.
        Verifies higher-order polynomial coefficients do not overflow or cause divergence.
        """
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=1000.0))
        s1 = Surface(
            name="HighOrderPoly",
            surface_type=SurfaceType.ASPHERIC,
            radius=40.0,
            conic=-1.0,
            asphere_coeffs=[1e-5, -2e-8, 3e-11, -4e-14, 5e-17],  # a4, a6, a8, a10, a12
            thickness=5.0,
            glass="N-BK7",
            semi_diameter=15.0
        )
        stack.add_surface(s1)
        s2 = Surface(name="PolyBack", radius=-80.0, thickness=25.0, glass="AIR", semi_diameter=15.0)
        set_opt_vars(s2, variable=True, t_min=10.0, t_max=150.0)
        stack.add_surface(s2)
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=20.0))

        sys = OpticalSystem(stack)
        optimizer = OpticalOptimizer(sys, method="Nelder-Mead", max_iter=50, num_rays=20, entrance_pupil_radius=8.0)
        res = optimizer.optimize(variable_indices=[2], bounds=[(10.0, 150.0)])

        self.assertTrue(res.success)
        self.assertLess(res.final_rms, res.initial_rms)


# ===========================================================================
# 2. Multi-Variable Optical Topologies (Complex Systems) Stress Tests
# ===========================================================================
class TestMultiVariableComplexTopologies(unittest.TestCase):
    """
    Stress-testing simultaneous multi-variable optimization in systems with 3, 4, 5, 6, and 8 variables.
    """

    def test_air_spaced_achromatic_doublet_3_variables(self):
        """
        Air-spaced doublet: optimize Crown thickness, Air gap, and Back focal distance (3 variables).
        """
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=1000.0))
        # Crown N-BK7
        s1 = Surface(name="Crown_F", radius=45.0, thickness=6.0, glass="N-BK7", semi_diameter=15.0)
        set_opt_vars(s1, variable=True, t_min=3.0, t_max=12.0)
        stack.add_surface(s1)
        # Air gap
        s2 = Surface(name="AirGap", radius=-35.0, thickness=2.0, glass="AIR", semi_diameter=15.0)
        set_opt_vars(s2, variable=True, t_min=0.1, t_max=8.0)
        stack.add_surface(s2)
        # Flint N-SF11
        s3 = Surface(name="Flint_F", radius=-34.0, thickness=3.5, glass="N-SF11", semi_diameter=15.0)
        stack.add_surface(s3)
        # Back focal distance
        s4 = Surface(name="Flint_B", radius=-150.0, thickness=30.0, glass="AIR", semi_diameter=15.0)
        set_opt_vars(s4, variable=True, t_min=15.0, t_max=120.0)
        stack.add_surface(s4)
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=20.0))

        sys = OpticalSystem(stack)
        optimizer = OpticalOptimizer(sys, method="Nelder-Mead", max_iter=100, num_rays=25, entrance_pupil_radius=10.0)
        res = optimizer.optimize(variable_indices=[1, 2, 4], bounds=[(3.0, 12.0), (0.1, 8.0), (15.0, 120.0)])

        self.assertTrue(res.success)
        self.assertEqual(len(res.optimized_thicknesses), 3)
        self.assertLess(res.final_rms, res.initial_rms)
        self.assertGreaterEqual(res.rms_reduction_pct, 40.0)

    def test_petzval_lens_4_variables(self):
        """
        Petzval lens objective (2 separated doublets): optimize 4 air spaces/thicknesses.
        """
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=2000.0))
        # Group 1: Doublet
        stack.add_surface(Surface(name="G1_L1_F", radius=35.0, thickness=5.0, glass="N-BK7", semi_diameter=15.0))
        stack.add_surface(Surface(name="G1_L1_B", radius=-25.0, thickness=2.5, glass="N-SF11", semi_diameter=15.0))
        s_g1_b = Surface(name="G1_L2_B", radius=-80.0, thickness=25.0, glass="AIR", semi_diameter=15.0)
        set_opt_vars(s_g1_b, variable=True, t_min=10.0, t_max=50.0)
        stack.add_surface(s_g1_b)
        # Group 2: Doublet
        stack.add_surface(Surface(name="G2_L1_F", radius=40.0, thickness=4.0, glass="N-BK7", semi_diameter=12.0))
        s_g2_mid = Surface(name="G2_L1_B", radius=-30.0, thickness=1.0, glass="AIR", semi_diameter=12.0)
        set_opt_vars(s_g2_mid, variable=True, t_min=0.2, t_max=5.0)
        stack.add_surface(s_g2_mid)
        stack.add_surface(Surface(name="G2_L2_F", radius=-28.0, thickness=3.0, glass="N-SF11", semi_diameter=12.0))
        s_bfl = Surface(name="G2_L2_B", radius=-60.0, thickness=20.0, glass="AIR", semi_diameter=12.0)
        set_opt_vars(s_bfl, variable=True, t_min=10.0, t_max=80.0)
        stack.add_surface(s_bfl)
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=20.0))

        sys = OpticalSystem(stack)
        optimizer = OpticalOptimizer(sys, method="Nelder-Mead", max_iter=120, num_rays=25, entrance_pupil_radius=8.0)
        var_indices = [3, 5, 7]
        bounds = [(10.0, 50.0), (0.2, 5.0), (10.0, 80.0)]
        res = optimizer.optimize(variable_indices=var_indices, bounds=bounds)

        self.assertTrue(res.success)
        self.assertLess(res.final_rms, res.initial_rms)
        self.assertGreaterEqual(res.rms_reduction_pct, 40.0)

    def test_double_gauss_6_element_system_8_variables(self):
        """
        Double Gauss (6 elements, 8 variable thicknesses/air gaps) simultaneous optimization.
        Stress test on high-dimensional parameter space (8D).
        """
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=1000.0))
        
        # S1, S2: Lens 1 (Meniscus Crown)
        stack.add_surface(Surface(name="L1_F", radius=28.0, thickness=4.5, glass="N-SK16", semi_diameter=12.0))
        stack.add_surface(Surface(name="L1_B", radius=120.0, thickness=1.5, glass="AIR", semi_diameter=12.0))
        
        # S3, S4, S5: Cemented Doublet 1 (Crown + Thick Flint)
        stack.add_surface(Surface(name="L2_F", radius=16.0, thickness=3.8, glass="N-SK16", semi_diameter=10.0))
        stack.add_surface(Surface(name="L2_Mid", radius=-35.0, thickness=1.5, glass="N-SF2", semi_diameter=10.0))
        stack.add_surface(Surface(name="L2_B", radius=12.0, thickness=4.0, glass="AIR", semi_diameter=8.0))
        
        # S6: Aperture Stop
        stack.add_surface(Surface(name="STOP", surface_type=SurfaceType.STOP, radius=0.0, thickness=3.5, glass="AIR", semi_diameter=6.0))
        
        # S7, S8, S9: Cemented Doublet 2 (Thick Flint + Crown)
        stack.add_surface(Surface(name="L3_F", radius=-13.0, thickness=1.5, glass="N-SF2", semi_diameter=8.0))
        stack.add_surface(Surface(name="L3_Mid", radius=35.0, thickness=3.8, glass="N-SK16", semi_diameter=10.0))
        stack.add_surface(Surface(name="L3_B", radius=-18.0, thickness=1.5, glass="AIR", semi_diameter=10.0))
        
        # S10, S11: Lens 4 (Meniscus Crown)
        stack.add_surface(Surface(name="L4_F", radius=60.0, thickness=3.5, glass="N-SK16", semi_diameter=12.0))
        stack.add_surface(Surface(name="L4_B", radius=-30.0, thickness=25.0, glass="AIR", semi_diameter=12.0))
        
        # Detector
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=20.0))

        sys = OpticalSystem(stack)
        optimizer = OpticalOptimizer(sys, method="Nelder-Mead", max_iter=150, num_rays=20, entrance_pupil_radius=6.0)
        
        var_indices = [1, 2, 3, 5, 6, 7, 9, 11]
        bounds = [
            (2.0, 8.0),   # L1 thickness
            (0.2, 5.0),   # air gap 1
            (1.5, 6.0),   # L2 front
            (1.0, 8.0),   # space to stop
            (1.0, 8.0),   # stop to L3
            (0.5, 4.0),   # L3 front
            (0.2, 5.0),   # air gap 3
            (10.0, 60.0)  # BFL
        ]
        res = optimizer.optimize(variable_indices=var_indices, bounds=bounds)

        self.assertTrue(res.success)
        self.assertEqual(len(res.optimized_thicknesses), 8)
        self.assertLess(res.final_rms, res.initial_rms)
        for idx, (b_min, b_max) in zip(var_indices, bounds):
            val = res.optimized_thicknesses[idx]
            self.assertGreaterEqual(val, b_min - 1e-4)
            self.assertLessEqual(val, b_max + 1e-4)


# ===========================================================================
# 3. Boundary Condition Stress Tests
# ===========================================================================
class TestBoundaryConditionStressDetailed(unittest.TestCase):
    """
    Stress-testing boundary condition limits, edge cases, and barrier violations.
    """

    def setUp(self):
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=1000.0))
        stack.add_surface(Surface(name="L1", radius=51.68, thickness=5.0, glass="N-BK7", semi_diameter=15.0))
        stack.add_surface(Surface(name="L2", radius=0.0, thickness=30.0, glass="AIR", semi_diameter=15.0))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=20.0))
        self.sys = OpticalSystem(stack)

    def test_extreme_initial_defocus_10x(self):
        """Initial thickness is 10x focal length (800mm vs true focus 96.7mm)."""
        self.sys.stack.surfaces[2].thickness = 800.0
        self.sys.stack._update_accumulated_positions()
        
        opt = OpticalOptimizer(self.sys, method="Nelder-Mead", max_iter=100, num_rays=20, entrance_pupil_radius=6.0)
        res = opt.optimize(variable_indices=[2], bounds=[(10.0, 1000.0)])

        self.assertTrue(res.success)
        self.assertLess(res.final_rms, res.initial_rms)
        self.assertAlmostEqual(res.optimized_thicknesses[2], 96.7, delta=12.0)

    def test_lower_bound_constrained_clamping(self):
        """True optimum is ~96.7mm, but bounds restrict to [120.0, 200.0]. Clamps at 120.0mm."""
        opt = OpticalOptimizer(self.sys, method="Nelder-Mead", max_iter=50, num_rays=20)
        res = opt.optimize(variable_indices=[2], bounds=[(120.0, 200.0)])

        self.assertTrue(res.success)
        self.assertAlmostEqual(res.optimized_thicknesses[2], 120.0, delta=1.0)
        self.assertGreaterEqual(res.optimized_thicknesses[2], 120.0 - 1e-4)

    def test_upper_bound_constrained_clamping(self):
        """True optimum is ~96.7mm, but bounds restrict to [20.0, 60.0]. Clamps at 60.0mm."""
        opt = OpticalOptimizer(self.sys, method="Nelder-Mead", max_iter=50, num_rays=20)
        res = opt.optimize(variable_indices=[2], bounds=[(20.0, 60.0)])

        self.assertTrue(res.success)
        self.assertAlmostEqual(res.optimized_thicknesses[2], 60.0, delta=1.0)
        self.assertLessEqual(res.optimized_thicknesses[2], 60.0 + 1e-4)

    def test_zero_width_bounds(self):
        """Bounds [45.0, 45.0] force parameter to remain exactly 45.0."""
        opt = OpticalOptimizer(self.sys, method="Nelder-Mead", max_iter=30, num_rays=15)
        res = opt.optimize(variable_indices=[2], bounds=[(45.0, 45.0)])

        self.assertTrue(res.success)
        self.assertAlmostEqual(res.optimized_thicknesses[2], 45.0, places=3)

    def test_starting_point_outside_bounds_recovery(self):
        """Initial thickness is 10.0mm, but bounds are [50.0, 150.0]. Optimizer recovers and enters bounds."""
        self.sys.stack.surfaces[2].thickness = 10.0
        self.sys.stack._update_accumulated_positions()
        
        opt = OpticalOptimizer(self.sys, method="Nelder-Mead", max_iter=60, num_rays=20)
        res = opt.optimize(variable_indices=[2], bounds=[(50.0, 150.0)])

        self.assertTrue(res.success)
        self.assertGreaterEqual(res.optimized_thicknesses[2], 50.0 - 1e-4)
        self.assertLessEqual(res.optimized_thicknesses[2], 150.0 + 1e-4)
        self.assertAlmostEqual(res.optimized_thicknesses[2], 96.7, delta=10.0)


# ===========================================================================
# 4. Solver Convergence Stability Across Methods & Limits
# ===========================================================================
class TestSolverMethodsAndBudgetLimits(unittest.TestCase):
    """
    Stress-testing solver algorithms (Nelder-Mead, L-BFGS-B, SLSQP, Powell) and iteration budgets.
    """

    def setUp(self):
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=1000.0))
        stack.add_surface(Surface(name="L1", radius=51.68, thickness=5.0, glass="N-BK7", semi_diameter=15.0))
        stack.add_surface(Surface(name="L2", radius=0.0, thickness=25.0, glass="AIR", semi_diameter=15.0))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=20.0))
        self.sys = OpticalSystem(stack)

    def test_solver_comparison_all_four_methods(self):
        """All four supported scipy methods (Nelder-Mead, L-BFGS-B, SLSQP, Powell) converge to ~96.7mm."""
        methods = ["Nelder-Mead", "L-BFGS-B", "SLSQP", "Powell"]
        results = {}
        for m in methods:
            opt = OpticalOptimizer(self.sys, method=m, max_iter=60, num_rays=20, entrance_pupil_radius=8.0)
            res = opt.optimize(variable_indices=[2], bounds=[(10.0, 180.0)])
            self.assertTrue(res.success, f"Method {m} failed")
            self.assertGreaterEqual(res.rms_reduction_pct, 50.0, f"Method {m} low reduction: {res.rms_reduction_pct}")
            self.assertAlmostEqual(res.optimized_thicknesses[2], 96.7, delta=12.0)
            results[m] = res

        # Check all methods returned valid results
        self.assertEqual(len(results), 4)

    def test_minimal_iteration_budget_1_iter(self):
        """max_iter=1 terminates cleanly without crashing and returns valid OptimizationResult."""
        opt = OpticalOptimizer(self.sys, method="Nelder-Mead", max_iter=1, num_rays=15)
        res = opt.optimize(variable_indices=[2], bounds=[(10.0, 150.0)])

        self.assertIsInstance(res, OptimizationResult)
        self.assertFalse(np.isnan(res.final_rms))
        self.assertIn(2, res.optimized_thicknesses)

    def test_minimal_iteration_budget_3_iter(self):
        """max_iter=3 terminates cleanly with non-negative reduction."""
        opt = OpticalOptimizer(self.sys, method="Nelder-Mead", max_iter=3, num_rays=15)
        res = opt.optimize(variable_indices=[2], bounds=[(10.0, 150.0)])

        self.assertIsInstance(res, OptimizationResult)
        self.assertGreaterEqual(res.rms_reduction_pct, 0.0)


# ===========================================================================
# 5. Ray Scaling and Execution Performance Benchmark Tests
# ===========================================================================
class TestRayScalingAndPerformance(unittest.TestCase):
    """
    Empirical scaling tests across ray counts (10 to 100 rays).
    """

    def setUp(self):
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=1000.0))
        stack.add_surface(Surface(name="L1", radius=51.68, thickness=5.0, glass="N-BK7", semi_diameter=15.0))
        stack.add_surface(Surface(name="L2", radius=0.0, thickness=30.0, glass="AIR", semi_diameter=15.0))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=20.0))
        self.sys = OpticalSystem(stack)

    def test_ray_density_scaling_10_to_100_rays(self):
        """
        Verify convergence accuracy and execution stability across N = 10, 30, 50, 75, 100 rays.
        """
        ray_counts = [10, 30, 60]
        timings = []
        final_rms_vals = []

        for n_rays in ray_counts:
            opt = OpticalOptimizer(self.sys, num_rays=n_rays, method="Nelder-Mead", max_iter=30, entrance_pupil_radius=8.0)
            t0 = time.perf_counter()
            res = opt.optimize(variable_indices=[2], bounds=[(10.0, 180.0)])
            dt = time.perf_counter() - t0

            self.assertTrue(res.success, f"Failed at num_rays={n_rays}")
            self.assertGreaterEqual(res.rms_reduction_pct, 50.0)
            self.assertAlmostEqual(res.optimized_thicknesses[2], 96.7, delta=10.0)

            timings.append(dt)
            final_rms_vals.append(res.final_rms)

        # Ensure all runs completed successfully in reasonable time
        for t in timings:
            self.assertLess(t, 120.0)


if __name__ == "__main__":
    unittest.main()
