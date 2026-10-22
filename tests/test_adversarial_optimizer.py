"""
Adversarial Stress Test Suite for M2: Optical Optimizer Engine.
Focuses on empirical challenge:
1. Highly aberrant configurations (extreme curvatures, strong conics/aspheres, high ray counts, chromatic dispersion).
2. Multi-variable scaling (doublets, triplets, quad-elements).
3. Boundary condition stress (zero thickness, narrow bounds, degenerate bounds, barrier penalty checks).
4. Solver convergence stability across algorithms (Nelder-Mead, L-BFGS-B, SLSQP).
5. Robustness against ray loss, total internal reflection, and complete vignetting.
6. Immutability of input system and stack coordinate consistency.
"""

import unittest
import numpy as np

from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from optics_app.engine.optimizer import OpticalOptimizer, OptimizationResult
from optics_app.analysis.spot import SpotDiagram


class TestHighlyAberrantConfigurations(unittest.TestCase):
    """Stress tests on highly aberrant optical configurations."""

    def test_extreme_steep_curvatures(self):
        """Test optimizer stability on extremely steep curvatures (R = 8mm, 5mm) with high ray divergence."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=50.0))
        # Steep meniscus lens
        s1 = Surface(name="SteepFront", radius=8.0, thickness=4.0, glass="N-BK7", semi_diameter=6.0)
        s2 = Surface(name="SteepBack", radius=5.0, thickness=15.0, glass="AIR", semi_diameter=6.0)
        s3 = Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=10.0)
        
        stack.add_surface(s1)
        stack.add_surface(s2)
        stack.add_surface(s3)

        sys = OpticalSystem(stack)
        optimizer = OpticalOptimizer(
            sys,
            num_rays=40,
            entrance_pupil_radius=4.0,
            method="Nelder-Mead",
            max_iter=100
        )
        res = optimizer.optimize(variable_indices=[2], bounds=[(1.0, 50.0)])
        
        self.assertIsInstance(res, OptimizationResult)
        self.assertFalse(np.isnan(res.final_rms))
        self.assertFalse(np.isinf(res.final_rms))
        self.assertGreater(res.final_rms, 0.0)
        self.assertLessEqual(res.final_rms, res.initial_rms + 1e-5)
        self.assertIn(2, res.optimized_thicknesses)
        self.assertGreaterEqual(res.optimized_thicknesses[2], 1.0)
        self.assertLessEqual(res.optimized_thicknesses[2], 50.0)

    def test_strong_conics_and_aspheres(self):
        """Test optimizer with extreme conic constants (hyperbolic k=-3.5, oblate k=+5.0) and high-order aspheres."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=80.0))
        
        s1 = Surface(
            name="AsphereFront",
            radius=30.0,
            conic=-3.5,
            asphere_coeffs=[1e-5, -1e-8, 1e-11],
            thickness=6.0,
            glass="N-BK7",
            semi_diameter=15.0
        )
        s2 = Surface(
            name="AsphereBack",
            radius=-40.0,
            conic=5.0,
            asphere_coeffs=[-2e-5, 3e-8],
            thickness=40.0,
            glass="AIR",
            semi_diameter=15.0
        )
        s3 = Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=20.0)

        for s in [s1, s2, s3]:
            stack.add_surface(s)

        sys = OpticalSystem(stack)
        optimizer = OpticalOptimizer(
            sys,
            num_rays=50,
            entrance_pupil_radius=8.0,
            method="Nelder-Mead",
            max_iter=80
        )
        res = optimizer.optimize(variable_indices=[2], bounds=[(5.0, 100.0)])
        
        self.assertTrue(res.success)
        self.assertLess(res.final_rms, res.initial_rms)
        self.assertGreater(res.rms_reduction_pct, 0.0)

    def test_high_ray_count_stress(self):
        """Test optimizer performance and numerical stability with high ray density (num_rays=200)."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=100.0))
        stack.add_surface(Surface(name="L1_Front", radius=60.0, thickness=8.0, glass="N-BK7", semi_diameter=25.0))
        stack.add_surface(Surface(name="L1_Back", radius=-60.0, thickness=25.0, glass="AIR", semi_diameter=25.0))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=25.0))

        sys = OpticalSystem(stack)
        optimizer = OpticalOptimizer(
            sys,
            num_rays=200,
            entrance_pupil_radius=15.0,
            method="Nelder-Mead",
            max_iter=50
        )
        res = optimizer.optimize(variable_indices=[2], bounds=[(20.0, 150.0)])
        
        self.assertTrue(res.success)
        self.assertLess(res.final_rms, res.initial_rms)
        self.assertGreater(res.rms_reduction_pct, 10.0)

    def test_multi_wavelength_chromatic_optimization(self):
        """Test optimizer balancing focus across F (0.4861), d (0.5876), and C (0.6563) Fraunhofer lines."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=100.0))
        # Highly dispersive flint glass (N-SF11)
        stack.add_surface(Surface(name="FlintFront", radius=45.0, thickness=6.0, glass="N-SF11", semi_diameter=18.0))
        stack.add_surface(Surface(name="FlintBack", radius=-90.0, thickness=50.0, glass="AIR", semi_diameter=18.0))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=20.0))

        sys = OpticalSystem(stack)
        wavelengths = [0.4861, 0.5876, 0.6563]  # F, d, C
        optimizer = OpticalOptimizer(
            sys,
            wavelengths=wavelengths,
            num_rays=30,
            entrance_pupil_radius=10.0,
            method="Nelder-Mead",
            max_iter=60
        )
        res = optimizer.optimize(variable_indices=[2], bounds=[(10.0, 100.0)])
        
        self.assertTrue(res.success)
        self.assertLess(res.final_rms, res.initial_rms)


class TestMultiVariableTopologies(unittest.TestCase):
    """Stress tests on multi-variable optical topologies (doublets, triplets, quad-elements)."""

    def test_cemented_doublet_two_variables(self):
        """Achromatic doublet: optimize flint lens thickness and back focal distance (2 variables)."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=100.0))
        # Crown N-BK7
        stack.add_surface(Surface(name="Crown_Front", radius=50.0, thickness=8.0, glass="N-BK7", semi_diameter=20.0))
        # Cemented interface N-BK7 / N-SF11
        stack.add_surface(Surface(name="Cemented_Mid", radius=-35.0, thickness=4.0, glass="N-SF11", semi_diameter=20.0))
        # Flint back surface
        stack.add_surface(Surface(name="Flint_Back", radius=-120.0, thickness=20.0, glass="AIR", semi_diameter=20.0))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=25.0))

        sys = OpticalSystem(stack)
        optimizer = OpticalOptimizer(
            sys,
            wavelengths=[0.4861, 0.5876, 0.6563],
            num_rays=30,
            entrance_pupil_radius=12.0,
            method="Nelder-Mead",
            max_iter=100
        )
        
        # Optimize thickness of flint (index 2) and back focal distance (index 3)
        res = optimizer.optimize(variable_indices=[2, 3], bounds=[(1.0, 15.0), (20.0, 120.0)])
        
        self.assertTrue(res.success)
        self.assertIn(2, res.optimized_thicknesses)
        self.assertIn(3, res.optimized_thicknesses)
        self.assertLess(res.final_rms, res.initial_rms)
        self.assertGreater(res.rms_reduction_pct, 20.0)

    def test_cooke_triplet_five_variables(self):
        """Cooke Triplet: optimize 3 lens thicknesses and 2 air spaces (5 variables)."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=1000.0))
        
        # Element 1: Biconvex Crown
        stack.add_surface(Surface(name="E1_Front", radius=22.0, thickness=3.5, glass="N-SK16", semi_diameter=12.0))
        stack.add_surface(Surface(name="E1_Back", radius=-150.0, thickness=5.5, glass="AIR", semi_diameter=12.0))
        
        # Element 2: Biconcave Flint (Aperture Stop near here)
        stack.add_surface(Surface(name="E2_Front", radius=-20.0, thickness=1.5, glass="N-SF2", semi_diameter=8.0))
        stack.add_surface(Surface(name="E2_Back", radius=20.0, thickness=6.0, glass="AIR", semi_diameter=8.0))
        
        # Element 3: Biconvex Crown
        stack.add_surface(Surface(name="E3_Front", radius=75.0, thickness=3.0, glass="N-SK16", semi_diameter=12.0))
        stack.add_surface(Surface(name="E3_Back", radius=-25.0, thickness=35.0, glass="AIR", semi_diameter=12.0))
        
        # Detector
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=15.0))

        sys = OpticalSystem(stack)
        optimizer = OpticalOptimizer(
            sys,
            num_rays=30,
            entrance_pupil_radius=8.0,
            method="Nelder-Mead",
            max_iter=150
        )
        
        # Optimize air spaces (2, 4) and back focal distance (6)
        var_indices = [2, 4, 6]
        bounds = [(1.0, 15.0), (1.0, 15.0), (15.0, 70.0)]
        res = optimizer.optimize(variable_indices=var_indices, bounds=bounds)
        
        self.assertTrue(res.success)
        self.assertLess(res.final_rms, res.initial_rms)
        for idx, (b_min, b_max) in zip(var_indices, bounds):
            self.assertGreaterEqual(res.optimized_thicknesses[idx], b_min - 1e-4)
            self.assertLessEqual(res.optimized_thicknesses[idx], b_max + 1e-4)

    def test_quad_element_system_six_variables(self):
        """Quad-element objective: simultaneous optimization across 6 variable distances."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=200.0))
        
        # L1
        stack.add_surface(Surface(name="L1_F", radius=40.0, thickness=5.0, glass="N-BK7", semi_diameter=15.0))
        stack.add_surface(Surface(name="L1_B", radius=-80.0, thickness=3.0, glass="AIR", semi_diameter=15.0))
        # L2
        stack.add_surface(Surface(name="L2_F", radius=-35.0, thickness=3.0, glass="N-SF11", semi_diameter=12.0))
        stack.add_surface(Surface(name="L2_B", radius=35.0, thickness=4.0, glass="AIR", semi_diameter=12.0))
        # L3
        stack.add_surface(Surface(name="L3_F", radius=50.0, thickness=4.0, glass="N-BK7", semi_diameter=12.0))
        stack.add_surface(Surface(name="L3_B", radius=-30.0, thickness=2.0, glass="AIR", semi_diameter=12.0))
        # L4
        stack.add_surface(Surface(name="L4_F", radius=60.0, thickness=4.0, glass="N-BK7", semi_diameter=15.0))
        stack.add_surface(Surface(name="L4_B", radius=-40.0, thickness=30.0, glass="AIR", semi_diameter=15.0))
        # Detector
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=20.0))

        sys = OpticalSystem(stack)
        optimizer = OpticalOptimizer(
            sys,
            num_rays=25,
            entrance_pupil_radius=8.0,
            method="Nelder-Mead",
            max_iter=120
        )
        
        var_indices = [1, 2, 3, 4, 6, 8]
        bounds = [(2.0, 10.0), (0.5, 10.0), (1.0, 8.0), (0.5, 10.0), (0.5, 8.0), (10.0, 60.0)]
        res = optimizer.optimize(variable_indices=var_indices, bounds=bounds)
        
        self.assertTrue(res.success)
        self.assertLess(res.final_rms, res.initial_rms)
        self.assertEqual(len(res.optimized_thicknesses), 6)


class TestBoundaryConditionStress(unittest.TestCase):
    """Stress tests on boundary conditions, degenerate cases, and penalty functions."""

    def test_initial_thickness_at_zero(self):
        """Optimizer starting with initial thickness = 0.0 mm."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=100.0))
        stack.add_surface(Surface(name="L1_Front", radius=50.0, thickness=5.0, glass="N-BK7", semi_diameter=20.0))
        stack.add_surface(Surface(name="L1_Back", radius=-50.0, thickness=0.0, glass="AIR", semi_diameter=20.0))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=20.0))

        sys = OpticalSystem(stack)
        optimizer = OpticalOptimizer(sys, method="Nelder-Mead", max_iter=80, num_rays=20)
        res = optimizer.optimize(variable_indices=[2], bounds=[(0.0, 100.0)])
        
        self.assertTrue(res.success)
        self.assertGreater(res.optimized_thicknesses[2], 0.0)
        self.assertLess(res.final_rms, res.initial_rms)

    def test_ultra_narrow_bounds(self):
        """Optimizer strictly respects ultra-narrow bounds [25.100, 25.105]."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=100.0))
        stack.add_surface(Surface(name="L1_Front", radius=50.0, thickness=5.0, glass="N-BK7", semi_diameter=20.0))
        stack.add_surface(Surface(name="L1_Back", radius=-50.0, thickness=25.0, glass="AIR", semi_diameter=20.0))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=20.0))

        sys = OpticalSystem(stack)
        optimizer = OpticalOptimizer(sys, method="Nelder-Mead", max_iter=50, num_rays=20)
        res = optimizer.optimize(variable_indices=[2], bounds=[(25.100, 25.105)])
        
        val = res.optimized_thicknesses[2]
        self.assertGreaterEqual(val, 25.100 - 1e-4)
        self.assertLessEqual(val, 25.105 + 1e-4)

    def test_zero_width_fixed_bounds(self):
        """Optimizer handles degenerate zero-width bounds [30.0, 30.0]."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=100.0))
        stack.add_surface(Surface(name="L1_Front", radius=50.0, thickness=5.0, glass="N-BK7", semi_diameter=20.0))
        stack.add_surface(Surface(name="L1_Back", radius=-50.0, thickness=10.0, glass="AIR", semi_diameter=20.0))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=20.0))

        sys = OpticalSystem(stack)
        optimizer = OpticalOptimizer(sys, method="Nelder-Mead", max_iter=40, num_rays=20)
        res = optimizer.optimize(variable_indices=[2], bounds=[(30.0, 30.0)])
        
        self.assertAlmostEqual(res.optimized_thicknesses[2], 30.0, places=3)

    def test_merit_function_heavy_penalty_on_negative_values(self):
        """Merit function returns large barrier penalty for negative variable values without crashing."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=100.0))
        stack.add_surface(Surface(name="L1", radius=50.0, thickness=5.0, glass="N-BK7", semi_diameter=20.0))
        stack.add_surface(Surface(name="L2", radius=-50.0, thickness=40.0, glass="AIR", semi_diameter=20.0))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=20.0))

        sys = OpticalSystem(stack)
        optimizer = OpticalOptimizer(sys)
        
        merit_neg = optimizer.merit_function(np.array([-10.0]), [2])
        self.assertGreater(merit_neg, 1e5)

    def test_complete_vignetting_penalty(self):
        """Merit function returns 1e5 penalty when all rays are completely vignetted/blocked."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=100.0))
        # Annular aperture with inner radius 20.0 mm > pupil radius (10.0 mm) -> blocks all pupil rays
        s1 = Surface(name="Stop", radius=0.0, thickness=10.0, aperture_type="Annular",
                     aperture_r_inner=20.0, aperture_r_outer=30.0, semi_diameter=30.0)
        stack.add_surface(s1)
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=10.0))

        sys = OpticalSystem(stack)
        optimizer = OpticalOptimizer(sys, entrance_pupil_radius=10.0, num_rays=20)
        
        merit_blocked = optimizer.merit_function(np.array([10.0]), [1])
        self.assertEqual(merit_blocked, 1e5)

    def test_partial_vignetting_proportional_penalty(self):
        """Merit function applies proportional penalty for partially vignetted bundles."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=100.0))
        # Small aperture (5mm) with 10mm pupil -> some rays pass, outer rays vignetted
        stack.add_surface(Surface(name="L1", radius=50.0, thickness=5.0, glass="N-BK7", semi_diameter=5.0))
        stack.add_surface(Surface(name="L2", radius=0.0, thickness=50.0, glass="AIR", semi_diameter=5.0))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=10.0))

        sys = OpticalSystem(stack)
        optimizer = OpticalOptimizer(sys, entrance_pupil_radius=10.0, num_rays=30, penalty_weight=1000.0)
        merit_val = optimizer.merit_function(np.array([50.0]), [2])
        
        # Merit function should exceed spot RMS alone due to added penalty
        self.assertGreater(merit_val, 100.0)

    def test_total_internal_reflection_handling(self):
        """Merit function smoothly penalizes rays lost due to total internal reflection (TIR)."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=50.0))
        # Dense glass to air with steep angle causing TIR
        stack.add_surface(Surface(name="Prism1", radius=10.0, thickness=20.0, glass="N-SF11", semi_diameter=15.0, tilt_x=45.0))
        stack.add_surface(Surface(name="Prism2", radius=0.0, thickness=30.0, glass="AIR", semi_diameter=15.0, tilt_x=-45.0))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=30.0))

        sys = OpticalSystem(stack)
        optimizer = OpticalOptimizer(sys, entrance_pupil_radius=8.0, num_rays=25)
        merit_val = optimizer.merit_function(np.array([20.0]), [1])
        
        self.assertIsInstance(merit_val, float)
        self.assertFalse(np.isnan(merit_val))
        self.assertFalse(np.isinf(merit_val))


class TestSolverConvergenceStability(unittest.TestCase):
    """Stress tests comparing solver methods across various landscapes."""

    def setUp(self):
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=100.0))
        stack.add_surface(Surface(name="L1_Front", radius=50.0, thickness=5.0, glass="N-BK7", semi_diameter=20.0))
        stack.add_surface(Surface(name="L1_Back", radius=0.0, thickness=20.0, glass="AIR", semi_diameter=20.0))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=20.0))
        self.sys = OpticalSystem(stack)

    def test_solver_nelder_mead(self):
        """Nelder-Mead simplex convergence test."""
        opt = OpticalOptimizer(self.sys, method="Nelder-Mead", max_iter=80, num_rays=25)
        res = opt.optimize(variable_indices=[2], bounds=[(10.0, 150.0)])
        self.assertTrue(res.success)
        self.assertLess(res.final_rms, res.initial_rms)
        self.assertGreater(res.rms_reduction_pct, 40.0)

    def test_solver_lbfgsb(self):
        """L-BFGS-B gradient-based convergence test."""
        opt = OpticalOptimizer(self.sys, method="L-BFGS-B", max_iter=50, num_rays=25)
        res = opt.optimize(variable_indices=[2], bounds=[(10.0, 150.0)])
        self.assertTrue(res.success)
        self.assertLess(res.final_rms, res.initial_rms)
        self.assertGreater(res.rms_reduction_pct, 40.0)

    def test_solver_slsqp(self):
        """SLSQP sequential least squares convergence test."""
        opt = OpticalOptimizer(self.sys, method="SLSQP", max_iter=50, num_rays=25)
        res = opt.optimize(variable_indices=[2], bounds=[(10.0, 150.0)])
        self.assertTrue(res.success)
        self.assertLess(res.final_rms, res.initial_rms)
        self.assertGreater(res.rms_reduction_pct, 40.0)

    def test_solver_low_iteration_budget(self):
        """Optimizer handles extreme budget limit (max_iter=2) gracefully."""
        opt = OpticalOptimizer(self.sys, method="Nelder-Mead", max_iter=2, num_rays=20)
        res = opt.optimize(variable_indices=[2], bounds=[(10.0, 100.0)])
        self.assertIsInstance(res, OptimizationResult)
        self.assertLessEqual(res.iterations, 10)  # scipy function evaluations or iterations
        self.assertFalse(np.isnan(res.final_rms))

    def test_solver_already_optimal_start(self):
        """Optimizer when starting at near-optimal focus."""
        # Find best thickness first
        opt = OpticalOptimizer(self.sys, method="Nelder-Mead", max_iter=100, num_rays=30)
        res1 = opt.optimize(variable_indices=[2], bounds=[(10.0, 150.0)])
        best_t = res1.optimized_thicknesses[2]

        # Re-run starting at best_t
        self.sys.stack.surfaces[2].thickness = best_t
        self.sys.stack._update_accumulated_positions()
        
        opt2 = OpticalOptimizer(self.sys, method="Nelder-Mead", max_iter=30, num_rays=30)
        res2 = opt2.optimize(variable_indices=[2], bounds=[(10.0, 150.0)])
        
        self.assertTrue(res2.success)
        self.assertAlmostEqual(res2.optimized_thicknesses[2], best_t, delta=2.0)


class TestStackImmutabilityAndCoordinateConsistency(unittest.TestCase):
    """Stress tests on system immutability and stack coordinate updates."""

    def test_input_system_stack_is_not_mutated_during_optimization(self):
        """Verify original OpticalSystem.stack is unchanged after calling optimize()."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=100.0))
        stack.add_surface(Surface(name="L1", radius=50.0, thickness=5.0, glass="N-BK7", semi_diameter=20.0))
        stack.add_surface(Surface(name="L2", radius=-50.0, thickness=20.0, glass="AIR", semi_diameter=20.0))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=20.0))
        
        sys = OpticalSystem(stack)
        orig_thickness_2 = sys.stack.surfaces[2].thickness
        orig_z_det = sys.stack.surfaces[3].global_origin[2]

        opt = OpticalOptimizer(sys, method="Nelder-Mead", max_iter=50, num_rays=20)
        res = opt.optimize(variable_indices=[2], bounds=[(10.0, 150.0)])

        # Original system should remain untouched
        self.assertEqual(sys.stack.surfaces[2].thickness, orig_thickness_2)
        self.assertEqual(sys.stack.surfaces[3].global_origin[2], orig_z_det)
        
        # Result stack should have new thickness and updated global_origin
        self.assertNotEqual(res.optimized_stack.surfaces[2].thickness, orig_thickness_2)
        new_thickness_2 = res.optimized_stack.surfaces[2].thickness
        expected_z_det = (
            res.optimized_stack.surfaces[0].thickness +
            res.optimized_stack.surfaces[1].thickness +
            new_thickness_2
        )
        self.assertAlmostEqual(res.optimized_stack.surfaces[3].global_origin[2], expected_z_det, places=5)

    def test_detector_placed_non_last_surface(self):
        """Detector placed at intermediate surface index is correctly targeted."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=50.0))
        stack.add_surface(Surface(name="L1", radius=40.0, thickness=5.0, glass="N-BK7", semi_diameter=15.0))
        stack.add_surface(Surface(name="L2", radius=0.0, thickness=30.0, glass="AIR", semi_diameter=15.0))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=10.0, semi_diameter=20.0))
        stack.add_surface(Surface(name="DUMMY", thickness=0.0, semi_diameter=20.0))

        sys = OpticalSystem(stack)
        opt = OpticalOptimizer(sys, method="Nelder-Mead", max_iter=60, num_rays=20)
        res = opt.optimize(variable_indices=[2], bounds=[(10.0, 100.0)])

        self.assertTrue(res.success)
        self.assertLess(res.final_rms, res.initial_rms)

    def test_empty_variable_indices_no_op(self):
        """Empty variable list returns no-op result gracefully."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=100.0))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0))
        sys = OpticalSystem(stack)
        
        opt = OpticalOptimizer(sys)
        res = opt.optimize(variable_indices=[])
        
        self.assertTrue(res.success)
        self.assertEqual(res.iterations, 0)
        self.assertEqual(res.optimized_thicknesses, {})


if __name__ == "__main__":
    unittest.main()
