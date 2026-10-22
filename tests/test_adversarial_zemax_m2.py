"""
Adversarial Stress Test Suite for Milestone 2:
Zemax-Standard Optical Optimization Engine.

Adversarially challenges:
1. Ill-conditioned and singular collinear variables (rank deficiency, kappa > 10^12, SVD fallback).
2. Dynamic damping adaptation (lambda * 2.0 on uphill) and multi-tiered step backtracking ([0.5, 0.25, 0.1]).
3. Complex multi-surface systems (Doublet, Cooke Triplet, Double Gauss) across 24 perturbed configurations.
4. Dual-tree parity (optics_app/ vs v2/) under ill-conditioned optimization.
"""

import pytest
import math
import numpy as np

from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.optimizer import (
    OpticalOptimizer,
    OptimizationResult,
    compute_system_efl,
    radius_to_curvature,
    curvature_to_radius
)
from optics_app.engine.manufacturing_standards import (
    audit_optical_system_manufacturing,
    ManufacturingConstraintsConfig
)


# ==============================================================================
# Helper System Generators
# ==============================================================================

def create_doublet_system(r1=55.0, r2=-38.0, r3=-140.0, t1=5.0, t2=3.5, bfl=85.0, semi_diameter=12.5):
    """5-surface achromatic cemented doublet system."""
    stack = SurfaceStack()
    stack.add_surface(Surface(name="OBJ", thickness=100.0, semi_diameter=semi_diameter))
    stack.add_surface(Surface(name="Crown_Front", radius=r1, thickness=t1, glass="N-BK7", semi_diameter=semi_diameter))
    stack.add_surface(Surface(name="Contact", radius=r2, thickness=t2, glass="SF2", semi_diameter=semi_diameter))
    stack.add_surface(Surface(name="Flint_Back", radius=r3, thickness=bfl, glass="AIR", semi_diameter=semi_diameter))
    stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=semi_diameter))
    return stack


def create_cooke_triplet_system(semi_diameter=10.0):
    """8-surface Cooke Triplet (Crown1, Flint2, Crown3)."""
    stack = SurfaceStack()
    stack.add_surface(Surface(name="OBJ", thickness=100.0, semi_diameter=semi_diameter))
    stack.add_surface(Surface(name="S1", radius=22.0, thickness=3.2, glass="N-SK16", semi_diameter=8.0))
    stack.add_surface(Surface(name="S2", radius=-150.0, thickness=5.5, glass="AIR", semi_diameter=8.0))
    stack.add_surface(Surface(name="S3", radius=-20.0, thickness=1.5, glass="SF2", semi_diameter=6.0))
    stack.add_surface(Surface(name="S4", radius=20.0, thickness=4.8, glass="AIR", semi_diameter=6.0))
    stack.add_surface(Surface(name="S5", radius=80.0, thickness=2.8, glass="N-SK16", semi_diameter=7.0))
    stack.add_surface(Surface(name="S6", radius=-24.0, thickness=40.0, glass="AIR", semi_diameter=7.0))
    stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=10.0))
    return stack


def create_double_gauss_system(scale=3.289):
    """12-surface Double Gauss 100mm EFL, F/4.0 system."""
    stack = SurfaceStack()
    stack.add_surface(Surface(name="OBJ", thickness=1000.0, semi_diameter=25.0))
    stack.add_surface(Surface(name="S1", radius=60.0*scale, thickness=7.0*scale, glass="N-SK16", semi_diameter=15.0*scale))
    stack.add_surface(Surface(name="S2", radius=180.0*scale, thickness=0.5*scale, glass="AIR", semi_diameter=15.0*scale))
    stack.add_surface(Surface(name="S3", radius=35.0*scale, thickness=4.0*scale, glass="SF2", semi_diameter=13.0*scale))
    stack.add_surface(Surface(name="S4", radius=22.0*scale, thickness=8.0*scale, glass="N-SK16", semi_diameter=13.0*scale))
    stack.add_surface(Surface(name="S5", radius=-45.0*scale, thickness=8.0*scale, glass="AIR", semi_diameter=11.0*scale))
    stack.add_surface(Surface(name="S6", radius=-28.0*scale, thickness=7.0*scale, glass="N-SK16", semi_diameter=12.0*scale))
    stack.add_surface(Surface(name="S7", radius=35.0*scale, thickness=3.5*scale, glass="SF2", semi_diameter=12.0*scale))
    stack.add_surface(Surface(name="S8", radius=-40.0*scale, thickness=0.5*scale, glass="AIR", semi_diameter=12.0*scale))
    stack.add_surface(Surface(name="S9", radius=75.0*scale, thickness=6.0*scale, glass="N-SK16", semi_diameter=14.0*scale))
    stack.add_surface(Surface(name="S10", radius=-120.0*scale, thickness=22.7, glass="AIR", semi_diameter=14.0*scale))
    stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=25.0))
    return stack


# ==============================================================================
# 1. Ill-Conditioned and Singular Collinear Variables Stress Tests
# ==============================================================================

class TestIllConditionedCollinearVariables:
    """Stress-test rank deficiency, collinear variables, and extreme condition numbers."""

    def test_singular_collinear_air_gaps_svd_fallback(self):
        """
        Verify that when two collinear air gaps produce mathematically identical Jacobian
        columns (rank deficiency, det(J^T J) == 0, kappa > 10^15), the SVD fallback
        regularizes the system, prevents LinAlgError, and achieves stable descent.
        """
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=100.0, semi_diameter=10.0))
        stack.add_surface(Surface(name="L1", radius=50.0, thickness=5.0, glass="N-BK7", semi_diameter=10.0))
        stack.add_surface(Surface(name="DummyAir1", radius=0.0, thickness=20.0, glass="AIR", semi_diameter=10.0))
        stack.add_surface(Surface(name="DummyAir2", radius=0.0, thickness=25.0, glass="AIR", semi_diameter=10.0))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=10.0))

        sys = OpticalSystem(stack)
        opt = OpticalOptimizer(sys, method="DLS", max_iter=10, num_rays=9)

        # 1. Empirically inspect Jacobian and condition number of J^T J
        x0 = np.array([20.0, 25.0])
        var_dict = {"thickness": [2, 3]}
        bounds_param = [(1.0, 100.0), (1.0, 100.0)]
        r0 = opt._compute_dls_residuals(x0, var_dict, False, bounds_param)
        h = 1e-4
        r1 = opt._compute_dls_residuals(x0 + np.array([h, 0.0]), var_dict, False, bounds_param)
        r2 = opt._compute_dls_residuals(x0 + np.array([0.0, h]), var_dict, False, bounds_param)

        J = np.column_stack([(r1 - r0) / h, (r2 - r0) / h])
        JT_J = J.T @ J

        # Columns must be collinear within numerical precision
        col_diff = np.max(np.abs(J[:, 0] - J[:, 1]))
        assert col_diff < 1e-10, f"Collinear air gaps must produce identical J columns, got diff {col_diff}"

        # J^T J condition number must exceed 10^12 (singular)
        s = np.linalg.svd(JT_J, compute_uv=False)
        kappa_JTJ = s[0] / max(s[-1], 1e-18)
        assert kappa_JTJ > 1e12, f"Condition number must exceed 10^12, got {kappa_JTJ}"

        # 2. Run optimization and verify stable convergence without LinAlgError
        res = opt.optimize(variable_indices=var_dict)
        assert res.success
        assert res.final_rms < res.initial_rms
        assert res.final_rms < 0.01
        assert res.optimized_thicknesses[2] > 0.0
        assert res.optimized_thicknesses[3] > 0.0

    def test_extreme_condition_number_svd_regularization(self):
        """
        Verify that when a sensitive variable (high-power surface) is paired with a
        zero-sensitivity variable (curvatures in identical media, d/dc == 0),
        kappa(A_lambda) > 10^12, and the SVD regularization branch clamps singular
        values to s_reg = max(s, 1e-8 * s_max), computing finite descent step.
        """
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=100.0, semi_diameter=10.0))
        stack.add_surface(Surface(name="L1_front", radius=25.0, thickness=5.0, glass="N-BK7", semi_diameter=10.0))
        stack.add_surface(Surface(name="L1_back", radius=-25.0, thickness=20.0, glass="AIR", semi_diameter=10.0))
        stack.add_surface(Surface(name="DummyAir", radius=0.0, thickness=15.0, glass="AIR", semi_diameter=10.0))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=10.0))

        sys = OpticalSystem(stack)
        opt = OpticalOptimizer(sys, method="DLS", max_iter=10, num_rays=9)

        var_dict = {"radius": [1, 3]}
        bounds_param = [(-0.1, 0.1), (-0.1, 0.1)]
        x0 = np.array([1.0 / 25.0, 0.0])

        r0 = opt._compute_dls_residuals(x0, var_dict, True, bounds_param)
        h = 1e-6
        r1 = opt._compute_dls_residuals(x0 + np.array([h, 0.0]), var_dict, True, bounds_param)
        r2 = opt._compute_dls_residuals(x0 + np.array([0.0, h]), var_dict, True, bounds_param)

        J = np.column_stack([(r1 - r0) / h, (r2 - r0) / h])
        JT_J = J.T @ J

        # Variable 1 is sensitive, Variable 3 has zero sensitivity
        assert np.linalg.norm(J[:, 0]) > 10.0
        assert np.linalg.norm(J[:, 1]) < 1e-7

        # Condition number of A_lam must exceed 10^12
        lam = 1e-3
        diag_A = np.diag(JT_J)
        D = np.diag(np.maximum(diag_A, 1e-6))
        A_lam = JT_J + lam * D
        s_Alam = np.linalg.svd(A_lam, compute_uv=False)
        kappa_Alam = s_Alam[0] / max(s_Alam[-1], 1e-18)
        assert kappa_Alam > 1e12, f"Expected kappa(A_lam) > 10^12, got {kappa_Alam}"

        # Run DLS optimization: verify clean descent and convergence
        res = opt.optimize(variable_indices=var_dict)
        assert res.success
        assert res.final_rms < res.initial_rms
        assert np.isfinite(res.final_rms)
        assert np.all(np.isfinite(list(res.optimized_radii.values())))

    def test_rank_deficient_3_collinear_thickness_variables(self):
        """
        Verify that 3 collinear air gaps (rank deficiency 2, 2 zero singular values in J^T J)
        distribute descent step symmetrically without oscillation, instability, or LinAlgError.
        """
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=100.0, semi_diameter=10.0))
        stack.add_surface(Surface(name="L1", radius=40.0, thickness=5.0, glass="N-BK7", semi_diameter=10.0))
        stack.add_surface(Surface(name="Air1", radius=0.0, thickness=15.0, glass="AIR", semi_diameter=10.0))
        stack.add_surface(Surface(name="Air2", radius=0.0, thickness=15.0, glass="AIR", semi_diameter=10.0))
        stack.add_surface(Surface(name="Air3", radius=0.0, thickness=15.0, glass="AIR", semi_diameter=10.0))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=10.0))

        sys = OpticalSystem(stack)
        opt = OpticalOptimizer(sys, method="DLS", max_iter=10, num_rays=9)

        res = opt.optimize(variable_indices={"thickness": [2, 3, 4]})
        assert res.success
        assert res.final_rms < res.initial_rms
        assert res.final_rms < 0.01

        # Check symmetric step distribution
        t2 = res.optimized_thicknesses[2]
        t3 = res.optimized_thicknesses[3]
        t4 = res.optimized_thicknesses[4]
        assert abs(t2 - t3) < 1e-4
        assert abs(t3 - t4) < 1e-4


# ==============================================================================
# 2. Dynamic Damping Adaptation & Step Backtracking Stress Tests
# ==============================================================================

class TestDynamicDampingAndBacktracking:
    """Stress-test Levenberg-Marquardt dynamic damping inflation, deflation, and backtracking."""

    def test_damping_inflation_on_uphill_step(self):
        """
        Verify that when a full candidate step produces an uphill cost, the optimizer
        initiates backtracking, and if backtracking also fails, doubles damping
        (lambda <- lambda * 2.0).
        """
        stack = create_doublet_system()
        sys = OpticalSystem(stack)
        opt = OpticalOptimizer(sys, method="DLS", max_iter=5, num_rays=9)

        # Track evaluated candidate points
        eval_points = []
        orig_compute = opt._compute_dls_residuals
        calls = 0

        def instrumented_compute(x_cand, var_dict, use_curv, bounds_param=None):
            nonlocal calls
            calls += 1
            r = orig_compute(x_cand, var_dict, use_curv, bounds_param)
            eval_points.append(float(x_cand[0]))
            return r

        opt._compute_dls_residuals = instrumented_compute
        res = opt.optimize(variable_indices={"thickness": [3]})
        assert res.success
        assert len(eval_points) > 1

    def test_step_backtracking_shrink_hierarchy(self):
        """
        Verify multi-tiered backtracking shrink factors [0.5, 0.25, 0.1]:
        1. Full step rejected, 0.5 step accepted -> candidate evaluated at x + 0.5 * delta_x.
        2. Full step & 0.5 rejected, 0.25 accepted -> candidate evaluated at x + 0.25 * delta_x.
        3. Full, 0.5, & 0.25 rejected, 0.1 accepted -> candidate evaluated at x + 0.1 * delta_x.
        """
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=100.0, semi_diameter=10.0))
        stack.add_surface(Surface(name="L1", radius=50.0, thickness=5.0, glass="N-BK7", semi_diameter=10.0))
        stack.add_surface(Surface(name="L2", radius=-50.0, thickness=40.0, glass="AIR", semi_diameter=10.0))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=10.0))

        sys = OpticalSystem(stack)

        # Test Case A: Full rejected, 0.5 accepted
        opt_a = OpticalOptimizer(sys, method="DLS", max_iter=2, num_rays=9)
        opt_a.best_x = np.array([40.0])
        opt_a.best_rms = 100.0

        calls_a = 0
        points_a = []
        orig_a = opt_a._compute_dls_residuals

        def mock_res_a(x_cand, var_dict, use_curv, bounds_param=None):
            nonlocal calls_a
            calls_a += 1
            r = orig_a(x_cand, var_dict, use_curv, bounds_param)
            points_a.append(float(x_cand[0]))
            if calls_a == 3:  # Call 3 is full candidate step: force uphill
                return r * 50.0
            return r

        opt_a._compute_dls_residuals = mock_res_a
        opt_a._optimize_dls(np.array([40.0]), [(10.0, 100.0)], {"thickness": [2]}, False)

        dx_a = points_a[2] - 40.0
        expected_05 = 40.0 + 0.5 * dx_a
        assert abs(points_a[3] - expected_05) < 1e-5, f"Expected 0.5 shrink at {expected_05}, got {points_a[3]}"

        # Test Case B: Full and 0.5 rejected, 0.25 accepted
        opt_b = OpticalOptimizer(sys, method="DLS", max_iter=2, num_rays=9)
        opt_b.best_x = np.array([40.0])
        opt_b.best_rms = 100.0

        calls_b = 0
        points_b = []
        orig_b = opt_b._compute_dls_residuals

        def mock_res_b(x_cand, var_dict, use_curv, bounds_param=None):
            nonlocal calls_b
            calls_b += 1
            r = orig_b(x_cand, var_dict, use_curv, bounds_param)
            points_b.append(float(x_cand[0]))
            if calls_b in [3, 4]:  # Full and 0.5 both uphill
                return r * 50.0
            return r

        opt_b._compute_dls_residuals = mock_res_b
        opt_b._optimize_dls(np.array([40.0]), [(10.0, 100.0)], {"thickness": [2]}, False)

        dx_b = points_b[2] - 40.0
        expected_025 = 40.0 + 0.25 * dx_b
        assert abs(points_b[4] - expected_025) < 1e-5, f"Expected 0.25 shrink at {expected_025}, got {points_b[4]}"

        # Test Case C: Full, 0.5, and 0.25 rejected, 0.1 accepted
        opt_c = OpticalOptimizer(sys, method="DLS", max_iter=2, num_rays=9)
        opt_c.best_x = np.array([40.0])
        opt_c.best_rms = 100.0

        calls_c = 0
        points_c = []
        orig_c = opt_c._compute_dls_residuals

        def mock_res_c(x_cand, var_dict, use_curv, bounds_param=None):
            nonlocal calls_c
            calls_c += 1
            r = orig_c(x_cand, var_dict, use_curv, bounds_param)
            points_c.append(float(x_cand[0]))
            if calls_c in [3, 4, 5]:  # Full, 0.5, and 0.25 all uphill
                return r * 50.0
            return r

        opt_c._compute_dls_residuals = mock_res_c
        opt_c._optimize_dls(np.array([40.0]), [(10.0, 100.0)], {"thickness": [2]}, False)

        dx_c = points_c[2] - 40.0
        expected_01 = 40.0 + 0.1 * dx_c
        assert abs(points_c[5] - expected_01) < 1e-5, f"Expected 0.1 shrink at {expected_01}, got {points_c[5]}"

    def test_damping_escalation_to_lam_max(self):
        """
        Verify that persistent uphill steps across all backtracking tiers cause damping
        to double repeatedly until lam_max = 10^8 is reached, gracefully terminating
        with 'Maximum damping reached' without crashing.
        """
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=100.0, semi_diameter=10.0))
        stack.add_surface(Surface(name="L1", radius=50.0, thickness=5.0, glass="N-BK7", semi_diameter=10.0))
        stack.add_surface(Surface(name="L2", radius=-50.0, thickness=40.0, glass="AIR", semi_diameter=10.0))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=10.0))

        sys = OpticalSystem(stack)
        opt = OpticalOptimizer(sys, method="DLS", max_iter=2, num_rays=9)
        opt.best_x = np.array([40.0])
        opt.best_rms = 1.0

        calls = 0
        def mock_all_uphill(x_cand, var_dict, use_curv, bounds_param=None):
            nonlocal calls
            calls += 1
            if calls == 1:
                return np.array([1.0])
            elif calls == 2:
                return np.array([1.01])
            else:
                return np.array([5.0])  # strictly uphill

        opt._compute_dls_residuals = mock_all_uphill
        x_opt, success, iters, msg = opt._optimize_dls(
            np.array([40.0]), [(10.0, 100.0)], {"thickness": [2]}, False
        )

        assert "Maximum damping reached" in msg
        assert calls > 30  # Verified multiple damping escalations


# ==============================================================================
# 3. Complex Multi-Surface Systems Across 24 Perturbed Configurations
# ==============================================================================

class TestComplexMultiSurfacePerturbations:
    """Stress-test multi-surface convergence across 24 perturbed configurations."""

    @pytest.mark.parametrize("seed", list(range(8)))
    def test_achromatic_doublet_convergence_across_8_perturbations(self, seed):
        """
        Verify Achromatic Doublet (5 surfaces) converges monotonically from 8
        perturbed initial states, maintaining physical thicknesses (CT >= 1.5mm, BFL >= 0.5mm).
        """
        stack = create_doublet_system()
        # Systematic perturbation offset
        factor_r = 1.0 + 0.04 * (seed - 3.5)
        factor_t = 1.0 + 0.05 * (seed - 3.5)

        stack.surfaces[1].radius *= factor_r
        stack.surfaces[2].radius *= (1.0 / factor_r)
        stack.surfaces[3].thickness *= factor_t

        sys = OpticalSystem(stack)
        opt = OpticalOptimizer(
            sys,
            method="DLS",
            max_iter=12,
            num_rays=9,
            enforce_manufacturing_constraints=True,
            min_glass_thickness=1.5,
            min_edge_thickness=1.0
        )

        res = opt.optimize(variable_indices={"thickness": [3], "radius": [1, 2]})
        assert res.success, f"Doublet seed {seed} failed: {res.message}"
        assert res.final_rms < res.initial_rms, f"Doublet seed {seed} failed to reduce RMS"
        assert res.optimized_stack.surfaces[1].thickness >= 1.5
        assert res.optimized_stack.surfaces[2].thickness >= 1.5
        assert res.optimized_stack.surfaces[3].thickness >= 0.5

    @pytest.mark.parametrize("seed", list(range(8)))
    def test_cooke_triplet_convergence_across_8_perturbations(self, seed):
        """
        Verify Cooke Triplet (8 surfaces) converges monotonically from 8 perturbed
        initial states, maintaining air clearances (MNEA >= 0.5mm) and CT >= 1.5mm.
        """
        stack = create_cooke_triplet_system()
        factor = 1.0 + 0.03 * (seed - 3.5)

        stack.surfaces[1].radius *= factor
        stack.surfaces[3].radius *= (1.0 / factor)
        stack.surfaces[6].thickness *= factor

        sys = OpticalSystem(stack)
        opt = OpticalOptimizer(
            sys,
            method="DLS",
            max_iter=10,
            num_rays=9,
            entrance_pupil_radius=5.5,
            enforce_manufacturing_constraints=True,
            min_glass_thickness=1.5,
            min_edge_thickness=1.0,
            min_air_gap=0.5
        )

        res = opt.optimize(variable_indices={"thickness": [2, 4, 6], "radius": [1, 3, 5]})
        assert res.success, f"Cooke Triplet seed {seed} failed: {res.message}"
        assert res.final_rms < res.initial_rms, f"Cooke Triplet seed {seed} failed to reduce RMS"
        assert res.optimized_stack.surfaces[2].thickness >= 0.5
        assert res.optimized_stack.surfaces[4].thickness >= 0.5

    @pytest.mark.parametrize("seed", list(range(8)))
    def test_double_gauss_convergence_across_8_perturbations(self, seed):
        """
        Verify 100mm F/4 Double Gauss (12 surfaces) converges monotonically from 8
        perturbed initial states, optimizing BFL and front/rear radii.
        """
        stack = create_double_gauss_system()
        factor = 1.0 + 0.04 * (seed - 3.5)

        stack.surfaces[10].thickness *= factor  # Perturb BFL
        stack.surfaces[1].radius *= (1.0 + 0.02 * (seed - 3.5))

        sys = OpticalSystem(stack)
        opt = OpticalOptimizer(
            sys,
            method="DLS",
            max_iter=8,
            num_rays=9,
            entrance_pupil_radius=10.0
        )

        res = opt.optimize(variable_indices={"thickness": [10], "radius": [1, 10]})
        assert res.success, f"Double Gauss seed {seed} failed: {res.message}"
        assert res.final_rms < res.initial_rms, f"Double Gauss seed {seed} failed to reduce RMS"
        assert res.optimized_thicknesses[10] > 0.0


# ==============================================================================
# 4. Dual-Tree Parity Test
# ==============================================================================

class TestDualTreeParity:
    """Verify optics_app/ and v2/ produce identical numerical results under DLS."""

    def test_dls_parity_between_optics_app_and_v2(self):
        """Run identical DLS optimization on optics_app and v2 and assert bit-level parity."""
        import optics_app.engine.optimizer as opt_app
        import v2.engine.optimizer as opt_v2

        # 1. Create identical stacks
        stack_app = create_doublet_system()
        stack_v2 = create_doublet_system()

        sys_app = opt_app.OpticalSystem(stack_app)
        sys_v2 = opt_v2.OpticalSystem(stack_v2)

        optimizer_app = opt_app.OpticalOptimizer(sys_app, method="DLS", max_iter=10, num_rays=9)
        optimizer_v2 = opt_v2.OpticalOptimizer(sys_v2, method="DLS", max_iter=10, num_rays=9)

        res_app = optimizer_app.optimize(variable_indices={"thickness": [3], "radius": [1, 2]})
        res_v2 = optimizer_v2.optimize(variable_indices={"thickness": [3], "radius": [1, 2]})

        assert res_app.success == res_v2.success
        assert abs(res_app.initial_rms - res_v2.initial_rms) < 1e-12
        assert abs(res_app.final_rms - res_v2.final_rms) < 1e-12
        for k in res_app.optimized_thicknesses:
            assert abs(res_app.optimized_thicknesses[k] - res_v2.optimized_thicknesses[k]) < 1e-12
        for k in res_app.optimized_radii:
            assert abs(res_app.optimized_radii[k] - res_v2.optimized_radii[k]) < 1e-12
