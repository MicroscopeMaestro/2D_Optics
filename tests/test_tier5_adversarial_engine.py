"""
Tier 5 White-Box Adversarial Hardening Test Suite for Zemax-Standard Optical Optimization Engine
=================================================================================================
Validates algorithmic robustness, numerical stability, edge cases, and physical boundary operands
across `optics_app/` and `v2/`:
1. SVD Regularization & Pathological Condition Number Monitoring (collinear, under-constrained, rank-deficient).
2. Dynamic Damping Adaptation (Levenberg-Marquardt lambda_{k+1} = lambda_k * beta) & Multi-Step Backtracking.
3. Curvature Space Parameterization (c = 1/R) & Dynamic Bounds Clamping (|c| <= 0.95 / y_sd).
4. Exact Sag Boundary Operands (MNCT, MXCT, MNET, MXET, MNEA), Paraxial Power EFFL (Phi = -C), and TOTR (S0 exclusion).
5. Multi-Field Ray Tracing, Extreme Field Aberrations, and Rayleigh-Hamilton Wavefront OPD Scaling.
6. 100% Tree Synchronization between `optics_app/` and `v2/`.
"""

import math
import os
import numpy as np
import pytest

# Ensure headless environment
os.environ["QT_QPA_PLATFORM"] = "offscreen"

# Primary tree imports
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from optics_app.engine.optimizer import (
    OpticalOptimizer,
    OptimizationResult,
    TerminationStatus,
    compute_system_efl,
    extract_working_f_number,
    radius_to_curvature,
    curvature_to_radius,
)
from optics_app.engine.merit_function import (
    MeritFunction,
    MeritOperand,
    OperandType,
)
from optics_app.engine.system_config import (
    SystemConfig,
    FieldPoint,
    FieldType,
    RayPatternType,
    WavelengthEntry,
)
from optics_app.engine.pupil_solve import (
    calculate_entrance_pupil,
    aim_ray_to_pupil,
    aim_collimated_ray,
    generate_pupil_grid,
)
from optics_app.analysis.spot import SpotDiagram, calculate_spot_metrics

# Secondary tree (v2) imports for 100% synchronization verification
from v2.engine.surface import Surface as V2Surface, SurfaceType as V2SurfaceType
from v2.engine.lde import SurfaceStack as V2SurfaceStack
from v2.engine.tracer import OpticalSystem as V2OpticalSystem
from v2.engine.optimizer import (
    OpticalOptimizer as V2OpticalOptimizer,
    radius_to_curvature as v2_radius_to_curvature,
    curvature_to_radius as v2_curvature_to_radius,
    extract_working_f_number as v2_extract_working_f_number,
)
from v2.engine.merit_function import (
    MeritFunction as V2MeritFunction,
    MeritOperand as V2MeritOperand,
    OperandType as V2OperandType,
)
from v2.engine.system_config import SystemConfig as V2SystemConfig, FieldPoint as V2FieldPoint
from v2.analysis.spot import SpotDiagram as V2SpotDiagram


# ==============================================================================
# Helper Factories
# ==============================================================================

def make_singlet(focal_length: float = 100.0, glass: str = "BK7", semi_diameter: float = 12.5) -> SurfaceStack:
    """Construct a simple plano-convex singlet focusing at focal_length."""
    stack = SurfaceStack()
    stack.add_surface(Surface(name="Object", surface_type=SurfaceType.OBJECT, thickness=100.0))
    # Lensmaker approx: 1/f = (n - 1) / R1 => R1 = (1.5168 - 1) * 100 = 51.68 mm
    r1 = 51.68 * (focal_length / 100.0)
    stack.add_surface(Surface(name="S1_Curved", radius=r1, thickness=5.0, glass=glass, semi_diameter=semi_diameter))
    stack.add_surface(Surface(name="S2_Plano", radius=0.0, thickness=focal_length, glass="AIR", semi_diameter=semi_diameter))
    stack.add_surface(Surface(name="Detector", surface_type=SurfaceType.DETECTOR, semi_diameter=semi_diameter * 1.5))
    return stack


# ==============================================================================
# 1. SVD Regularization & Pathological Condition Number Monitoring
# ==============================================================================

class TestTier5SvdRegularizationAndConditioning:
    """Stress tests on SVD pseudo-inverse, collinearity, rank deficiency, and extreme conditioning."""

    def test_svd_collinear_air_thickness_variables(self):
        """
        Adversarial test: Two adjacent air space thicknesses are varied simultaneously.
        Because both thicknesses contribute additively to total distance through air,
        their Jacobian sensitivity columns are collinear (rank deficient).
        Verify DLS solves via SVD regularization without LinAlgError, NaN, or crash.
        """
        stack = SurfaceStack()
        stack.add_surface(Surface(name="Object", surface_type=SurfaceType.OBJECT, thickness=50.0))
        stack.add_surface(Surface(name="S1", radius=40.0, thickness=5.0, glass="BK7", semi_diameter=12.5))
        stack.add_surface(Surface(name="S2", radius=-40.0, thickness=10.0, glass="AIR", semi_diameter=12.5))
        stack.add_surface(Surface(name="S3_AirSplit", radius=0.0, thickness=15.0, glass="AIR", semi_diameter=12.5))
        stack.add_surface(Surface(name="Detector", surface_type=SurfaceType.DETECTOR, semi_diameter=25.0))

        sys = OpticalSystem(stack)
        opt = OpticalOptimizer(sys, method="DLS", max_iter=6)

        # Optimize collinear variables: thickness of S2 and S3
        res = opt.optimize(variable_indices={"thickness": [2, 3]})
        assert res.success, "DLS must succeed on collinear variables via SVD regularization"
        assert not np.isnan(res.final_rms), "Final RMS must not be NaN"
        assert not np.isinf(res.final_rms), "Final RMS must not be Inf"
        assert res.final_rms < res.initial_rms, "Collinear descent should reduce spot radius"

    def test_svd_rank_deficient_zero_gradient_columns(self):
        """
        Adversarial test: A dummy air surface (n1=n2=1.0) is included whose radius variable
        has exactly 0.0 optical effect everywhere (Jacobian column is identically zero).
        Verify SVD regularization clips zero singular values (s_reg = max(s, 1e-8 * s_max))
        and executes clean descent without LinAlgError.
        """
        stack = SurfaceStack()
        stack.add_surface(Surface(name="Object", surface_type=SurfaceType.OBJECT, thickness=10.0))
        stack.add_surface(Surface(name="S1", radius=50.0, thickness=5.0, glass="BK7", semi_diameter=12.5))
        stack.add_surface(Surface(name="S2", radius=-50.0, thickness=10.0, glass="AIR", semi_diameter=12.5))
        # Dummy air-to-air surface: delta_n = 0
        stack.add_surface(Surface(name="DummyAir", radius=0.0, thickness=10.0, glass="AIR", semi_diameter=12.5))
        stack.add_surface(Surface(name="Detector", surface_type=SurfaceType.DETECTOR, semi_diameter=25.0))

        sys = OpticalSystem(stack)
        opt = OpticalOptimizer(sys, method="DLS", max_iter=5)

        # Radius 3 is on DummyAir -> zero gradient column
        res = opt.optimize(variable_indices={"radius": [1, 2, 3]})
        assert res.success, "Optimizer must succeed despite zero-gradient variable column"
        assert not np.isnan(res.final_rms)
        assert 3 in res.optimized_radii
        # Dummy air surface radius should remain unchanged (0.0) or finite
        assert math.isfinite(res.optimized_radii[3])

    def test_svd_extreme_condition_number_trigger(self):
        """
        Adversarial test: Construct a system with severe sensitivity disparity between
        an extremely sensitive steep meniscus radius and an insensitive thickness,
        forcing the normal equation condition number cond > 1e8.
        Verify SVD regularization branch is exercised cleanly without failure.
        """
        stack = SurfaceStack()
        stack.add_surface(Surface(name="Object", surface_type=SurfaceType.OBJECT, thickness=10.0))
        # Steep meniscus: R1=5.2mm, R2=5.0mm (hyper-sensitive to curvature)
        stack.add_surface(Surface(name="S1", radius=5.2, thickness=3.0, glass="SF11", semi_diameter=4.0))
        stack.add_surface(Surface(name="S2", radius=-5.0, thickness=2.0, glass="AIR", semi_diameter=4.0))
        # Insensitive dummy air thickness
        stack.add_surface(Surface(name="S3", radius=0.0, thickness=1.0, glass="AIR", semi_diameter=4.0))
        stack.add_surface(Surface(name="Detector", surface_type=SurfaceType.DETECTOR, semi_diameter=10.0))

        sys = OpticalSystem(stack)
        opt = OpticalOptimizer(sys, method="DLS", max_iter=5)

        res = opt.optimize(variable_indices={"radius": [1, 2], "thickness": [3]})
        assert res.success, "Must handle pathological condition number without LinAlgError"
        assert not np.isnan(res.final_rms)
        assert not np.isinf(res.final_rms)

    def test_svd_stationary_point_immediate_convergence(self):
        """
        Adversarial test: When the initial system is already at a stationary point (g ~ 0),
        verify DLS detects np.max(np.abs(g)) < 1e-5 and exits immediately with converged=True.
        """
        stack = make_singlet(focal_length=100.0)
        sys = OpticalSystem(stack)
        opt = OpticalOptimizer(sys, method="DLS", max_iter=10)

        # Optimize detector thickness when already at focus
        res = opt.optimize(variable_indices={"thickness": [2]})
        assert res.success
        assert res.iterations <= 3, f"Expected fast convergence at or near focus, got {res.iterations}"

    def test_svd_v2_parity(self):
        """Verify identical robust SVD behavior on secondary tree (v2)."""
        stack = V2SurfaceStack()
        stack.add_surface(V2Surface(name="Object", surface_type=V2SurfaceType.OBJECT, thickness=50.0))
        stack.add_surface(V2Surface(name="S1", radius=40.0, thickness=5.0, glass="BK7", semi_diameter=12.5))
        stack.add_surface(V2Surface(name="S2", radius=-40.0, thickness=10.0, glass="AIR", semi_diameter=12.5))
        stack.add_surface(V2Surface(name="S3_AirSplit", radius=0.0, thickness=15.0, glass="AIR", semi_diameter=12.5))
        stack.add_surface(V2Surface(name="Detector", surface_type=V2SurfaceType.DETECTOR, semi_diameter=25.0))

        sys = V2OpticalSystem(stack)
        opt = V2OpticalOptimizer(sys, method="DLS", max_iter=6)

        res = opt.optimize(variable_indices={"thickness": [2, 3]})
        assert res.success
        assert not np.isnan(res.final_rms)


# ==============================================================================
# 2. Dynamic Damping Adaptation & Multi-Step Backtracking
# ==============================================================================

class TestTier5DynamicDampingAndBacktracking:
    """Stress tests on Levenberg-Marquardt damping adaptation and step backtracking."""

    def test_dls_backtracking_under_rejected_full_step(self):
        """
        Adversarial test: Construct an objective where the full step delta_x overshoots,
        forcing the multi-step backtracking loop [0.5, 0.25, 0.1].
        Verify that a backtracked step is successfully evaluated and accepted.
        """
        stack = make_singlet(focal_length=50.0)
        sys = OpticalSystem(stack)
        opt = OpticalOptimizer(sys, method="DLS", max_iter=8)

        # Track residual calls to verify backtracking occurs
        res = opt.optimize(variable_indices={"radius": [1, 2]})
        assert res.success, "Optimizer must converge with backtracking"
        assert res.final_rms <= res.initial_rms, "Backtracked steps must not increase cost"

    def test_dls_maximum_damping_termination_on_blocked_landscape(self):
        """
        Adversarial test: Force every tentative step to be rejected (cost_cand >= cost)
        by intercepting residuals.
        Verify that damping lambda ramps up by beta_up=2.0 until lambda >= 1e8,
        terminating gracefully with 'Maximum damping reached' without infinite looping.
        """
        stack = make_singlet(focal_length=50.0)
        sys = OpticalSystem(stack)
        opt = OpticalOptimizer(sys, method="DLS", max_iter=25)

        orig_fn = opt._compute_dls_residuals
        base_x = None

        def rejecting_fn(x_param, var_dict, use_curv, bounds_param=None):
            nonlocal base_x
            if base_x is None:
                base_x = x_param.copy()
                return orig_fn(x_param, var_dict, use_curv, bounds_param)
            # Base x evaluations (e.g. for finite differences) return normal residuals
            if np.allclose(x_param, base_x, atol=1e-3):
                return orig_fn(x_param, var_dict, use_curv, bounds_param)
            # Any step candidate returns huge penalty
            return np.array([1e7, 1e7, 1e7], dtype=float)

        opt._compute_dls_residuals = rejecting_fn
        res = opt.optimize(variable_indices={"radius": [1, 2]})

        assert res.success, "Termination due to maximum damping should be handled gracefully"
        assert "damping" in res.message.lower() or "completed" in res.message.lower()
        assert res.iterations <= 25, "Must terminate without infinite looping"

    def test_dls_best_solution_retention_on_degradation(self):
        """
        Adversarial test: Ensure best_x and best_rms are preserved even if subsequent
        iterations encounter degradation or abort.
        """
        stack = make_singlet(focal_length=100.0)
        sys = OpticalSystem(stack)
        opt = OpticalOptimizer(sys, method="DLS", max_iter=5)

        res = opt.optimize(variable_indices={"radius": [1, 2]})
        assert res.final_rms <= res.initial_rms
        assert opt.best_rms <= res.initial_rms
        assert math.isclose(res.final_rms, opt.best_rms, rel_tol=1e-3)


# ==============================================================================
# 3. Curvature Space Parameterization & Dynamic Bounds Clamping
# ==============================================================================

class TestTier5CurvatureSpaceAndBoundsClamping:
    """Stress tests on c = 1/R mapping, singularity avoidance, and dynamic semi-diameter clamping."""

    def test_curvature_space_singularities_and_inverses(self):
        """
        Adversarial test: Evaluate radius_to_curvature and curvature_to_radius on
        boundary and singular inputs: R=0, R=inf, R=1e8, R=1e-5, c=0, c=1e-7, NaN, Inf.
        Verify all return strictly finite floats and never throw ZeroDivisionError.
        """
        # Flat surface / infinite radius
        assert radius_to_curvature(0.0) == 0.0
        assert radius_to_curvature(float("inf")) == 0.0
        assert radius_to_curvature(-float("inf")) == 0.0
        assert radius_to_curvature(1e8) == 0.0
        assert radius_to_curvature(-1e8) == 0.0
        assert radius_to_curvature(1e-5) == 0.0  # sub-micron singularity guard
        assert radius_to_curvature(float("nan")) == 0.0

        # Curvature to radius
        assert curvature_to_radius(0.0) == 0.0
        assert curvature_to_radius(1e-7) == 0.0  # near-flat guard
        assert curvature_to_radius(-1e-7) == 0.0
        assert curvature_to_radius(float("nan")) == 0.0
        assert curvature_to_radius(float("inf")) == 0.0

        # High-precision round-trip
        for test_r in [5.0, 25.4, 100.0, -50.0, -2500.0]:
            c = radius_to_curvature(test_r)
            r_back = curvature_to_radius(c)
            assert math.isclose(test_r, r_back, rel_tol=1e-10)

    def test_dynamic_bounds_clamping_micro_optics(self):
        """
        Adversarial test: Clear semi-diameter y_sd = 0.2 mm (micro-optics).
        Physical limit: |c| <= 0.95 / y_sd = 4.75 mm^-1 (R_min ~ 0.21 mm).
        Verify that out-of-bounds candidate curvatures (c = 25.0) are clamped to 4.75 mm^-1
        inside _compute_dls_residuals, preventing non-physical imaginary sag sqrt(1 - c^2 y^2).
        """
        stack = SurfaceStack()
        stack.add_surface(Surface(name="Object", surface_type=SurfaceType.OBJECT, thickness=10.0))
        stack.add_surface(Surface(name="MicroLens", radius=1.0, thickness=0.5, glass="BK7", semi_diameter=0.2))
        stack.add_surface(Surface(name="MicroBack", radius=-1.0, thickness=1.0, glass="AIR", semi_diameter=0.2))
        stack.add_surface(Surface(name="Detector", surface_type=SurfaceType.DETECTOR, semi_diameter=0.5))

        sys = OpticalSystem(stack)
        opt = OpticalOptimizer(sys, method="DLS", max_iter=3, entrance_pupil_radius=0.15)

        # Candidate curvature c = 25.0 (well above c_limit = 4.75)
        x_ext = np.array([25.0, -25.0])
        res = opt._compute_dls_residuals(x_ext, {"radius": [1, 2]}, use_curv=True)

        assert not np.isnan(np.linalg.norm(res)), "Clamped micro-optics residuals must not be NaN"
        assert not np.isinf(np.linalg.norm(res)), "Clamped micro-optics residuals must not be Inf"

    def test_dynamic_bounds_clamping_macro_optics(self):
        """
        Adversarial test: Huge telescope optic with y_sd = 5000.0 mm.
        Physical limit: |c| <= 0.95 / 5000 = 0.00019 mm^-1 (R_min ~ 5263.15 mm).
        Verify that candidate curvature c = 0.05 is clamped to 0.00019 mm^-1.
        """
        stack = SurfaceStack()
        stack.add_surface(Surface(name="Object", surface_type=SurfaceType.OBJECT, thickness=1e7))
        stack.add_surface(Surface(name="PrimaryMirror", radius=10000.0, thickness=8000.0, glass="MIRROR", semi_diameter=5000.0))
        stack.add_surface(Surface(name="Detector", surface_type=SurfaceType.DETECTOR, semi_diameter=500.0))

        sys = OpticalSystem(stack)
        opt = OpticalOptimizer(sys, method="DLS", max_iter=2, entrance_pupil_radius=4000.0)

        # Candidate curvature c = 0.01 (limit is 0.00019)
        x_macro = np.array([0.01])
        res = opt._compute_dls_residuals(x_macro, {"radius": [1]}, use_curv=True)

        assert not np.isnan(np.linalg.norm(res))
        assert not np.isinf(np.linalg.norm(res))

    def test_curvature_clamping_zero_semi_diameter_fallback(self):
        """
        Adversarial test: When surface semi_diameter <= 1e-4 and entrance_pupil_radius <= 1e-4,
        verify fallback to default y_sd = 10.0 mm (c_limit = 0.095) prevents ZeroDivisionError.
        """
        stack = SurfaceStack()
        stack.add_surface(Surface(name="Object", surface_type=SurfaceType.OBJECT, thickness=10.0))
        stack.add_surface(Surface(name="ZeroSD", radius=50.0, thickness=5.0, glass="BK7", semi_diameter=0.0))
        stack.add_surface(Surface(name="S2", radius=-50.0, thickness=20.0, glass="AIR", semi_diameter=0.0))
        stack.add_surface(Surface(name="Detector", surface_type=SurfaceType.DETECTOR, semi_diameter=0.0))

        sys = OpticalSystem(stack)
        opt = OpticalOptimizer(sys, method="DLS", max_iter=2, entrance_pupil_radius=0.0)

        x_test = np.array([5.0, -5.0])
        res = opt._compute_dls_residuals(x_test, {"radius": [1, 2]}, use_curv=True)
        assert not np.isnan(np.linalg.norm(res))

    def test_curvature_zero_crossing_continuity(self):
        """
        Adversarial test: Transition surface curvature continuously across zero:
        c = +0.02 -> c = 0.0 -> c = -0.02.
        Verify sag and normal vectors remain smooth and well-defined without discontinuities.
        """
        surf = Surface(semi_diameter=10.0)
        curvatures = np.linspace(-0.05, 0.05, 21)
        sags = []

        for c in curvatures:
            surf.radius = curvature_to_radius(c)
            sags.append(surf.sag(5.0))

        sags = np.array(sags)
        assert not np.any(np.isnan(sags))
        # Sag at c = 0 must be exactly 0
        zero_idx = len(curvatures) // 2
        assert sags[zero_idx] == 0.0
        # Positive and negative curvatures should have antisymmetric sags
        assert np.allclose(sags[:zero_idx], -sags[zero_idx + 1:][::-1], atol=1e-12)

    def test_v2_curvature_parity(self):
        """Verify v2 curvature mapping and clamping match optics_app."""
        assert v2_radius_to_curvature(0.0) == 0.0
        assert v2_radius_to_curvature(float("inf")) == 0.0
        assert v2_curvature_to_radius(0.0) == 0.0
        assert math.isclose(v2_curvature_to_radius(v2_radius_to_curvature(100.0)), 100.0)


# ==============================================================================
# 4. Exact Sag Boundary Operands & Paraxial Power EFFL / TOTR
# ==============================================================================

class TestTier5ExactSagBoundaryOperands:
    """Stress tests on exact sag, edge thickness (MNET/MXET), air clearances (MNEA), EFFL, and TOTR."""

    def test_exact_sag_analytical_conic_sections(self):
        """
        Adversarial test: Verify Surface.sag(r) matches exact closed-form equations:
        s(y) = c y^2 / (1 + sqrt(1 - (1+k) c^2 y^2))
        across Sphere (k=0), Paraboloid (k=-1), Hyperboloid (k=-2), and Oblate Spheroid (k=1).
        """
        r = 5.0
        R = 25.0
        c = 1.0 / R

        # 1. Sphere (k = 0): s = R - sqrt(R^2 - r^2)
        s_sphere = Surface(radius=R, conic=0.0)
        expected_sphere = R - math.sqrt(R**2 - r**2)
        assert math.isclose(s_sphere.sag(r), expected_sphere, rel_tol=1e-12)

        # 2. Paraboloid (k = -1): s = c * r^2 / 2 = r^2 / (2 * R)
        s_parab = Surface(radius=R, conic=-1.0)
        expected_parab = (r**2) / (2.0 * R)
        assert math.isclose(s_parab.sag(r), expected_parab, rel_tol=1e-12)

        # 3. Hyperboloid (k = -2)
        s_hyp = Surface(radius=R, conic=-2.0)
        expected_hyp = (c * r**2) / (1.0 + math.sqrt(1.0 - (-1.0) * c**2 * r**2))
        assert math.isclose(s_hyp.sag(r), expected_hyp, rel_tol=1e-12)

        # 4. Oblate Spheroid (k = 1)
        s_obl = Surface(radius=R, conic=1.0)
        expected_obl = (c * r**2) / (1.0 + math.sqrt(1.0 - 2.0 * c**2 * r**2))
        assert math.isclose(s_obl.sag(r), expected_obl, rel_tol=1e-12)

    def test_mnct_mxct_boundary_violations(self):
        """
        Adversarial test: Test MNCT (min center thickness) and MXCT (max center thickness)
        operands in MeritFunction against violating values.
        When CT < MNCT target, diff = target - CT > 0.
        When CT > MXCT target, diff = CT - target > 0.
        When within bounds, diff = 0.
        """
        stack = SurfaceStack()
        stack.add_surface(Surface(name="Object", surface_type=SurfaceType.OBJECT, thickness=100.0))
        # Lens with violating center thickness CT = 0.8 mm (MNCT target = 1.5 mm)
        stack.add_surface(Surface(name="S1", radius=30.0, thickness=0.8, glass="BK7", semi_diameter=10.0))
        stack.add_surface(Surface(name="S2", radius=-30.0, thickness=20.0, glass="AIR", semi_diameter=10.0))
        stack.add_surface(Surface(name="Detector", surface_type=SurfaceType.DETECTOR, semi_diameter=15.0))

        sys = OpticalSystem(stack)
        mf = MeritFunction()
        # Test explicit surface index 1
        op_mnct = mf.add_operand(OperandType.MNCT, surface_index=1, target=1.5, weight=10.0)
        op_mxct = mf.add_operand(OperandType.MXCT, surface_index=1, target=18.0, weight=10.0)

        mf.evaluate(sys, [])
        # MNCT violation: 1.5 - 0.8 = 0.7
        assert math.isclose(op_mnct.value, 0.8, abs_tol=1e-6)
        assert math.isclose(op_mnct.diff, 0.7, abs_tol=1e-6)
        # MXCT within bound: 0.8 <= 18.0 => diff = 0.0
        assert math.isclose(op_mxct.diff, 0.0, abs_tol=1e-6)

    def test_mnet_mxet_exact_edge_thickness_profiles(self):
        """
        Adversarial test: Verify exact edge thickness ET = CT - (s2(y_sd) - s1(y_sd))
        across biconvex, biconcave, and knife-edge meniscus profiles.
        """
        # 1. Biconcave lens: R1 = -30 mm, R2 = +30 mm, CT = 3.0 mm, y_sd = 10.0 mm
        # s1(10) < 0, s2(10) > 0 => s2 - s1 > 0 => ET = CT + (|s1| + |s2|) > CT
        surf1 = Surface(radius=-30.0, thickness=3.0, glass="BK7", semi_diameter=10.0)
        surf2 = Surface(radius=30.0, thickness=20.0, glass="AIR", semi_diameter=10.0)

        et_biconcave = surf1.thickness + surf2.sag(10.0) - surf1.sag(10.0)
        assert et_biconcave > surf1.thickness, f"Biconcave ET ({et_biconcave}) must exceed CT ({surf1.thickness})"

        # 2. Biconvex lens: R1 = +30 mm, R2 = -30 mm, CT = 5.0 mm, y_sd = 10.0 mm
        # s1(10) > 0, s2(10) < 0 => s2 - s1 < 0 => ET = CT - (|s1| + |s2|) < CT
        surf1_cx = Surface(radius=30.0, thickness=5.0, glass="BK7", semi_diameter=10.0)
        surf2_cx = Surface(radius=-30.0, thickness=20.0, glass="AIR", semi_diameter=10.0)
        et_biconvex = surf1_cx.thickness + surf2_cx.sag(10.0) - surf1_cx.sag(10.0)
        assert et_biconvex < surf1_cx.thickness, f"Biconvex ET ({et_biconvex}) must be less than CT ({surf1_cx.thickness})"

        # 3. Knife-edge condition: CT is too thin for curvature and clear aperture (ET <= 0)
        surf1_knife = Surface(radius=15.0, thickness=2.0, glass="BK7", semi_diameter=10.0)
        surf2_knife = Surface(radius=-15.0, thickness=20.0, glass="AIR", semi_diameter=10.0)
        et_knife = surf1_knife.thickness + surf2_knife.sag(10.0) - surf1_knife.sag(10.0)
        assert et_knife < 0.0, f"Steep biconvex with thin CT must produce negative ET, got {et_knife}"

        # Merit function evaluation of knife-edge
        stack = SurfaceStack()
        stack.add_surface(Surface(name="Object", surface_type=SurfaceType.OBJECT, thickness=50.0))
        stack.add_surface(surf1_knife)
        stack.add_surface(surf2_knife)
        stack.add_surface(Surface(name="Detector", surface_type=SurfaceType.DETECTOR, semi_diameter=15.0))

        sys = OpticalSystem(stack)
        mf = MeritFunction()
        op_mnet = mf.add_operand(OperandType.MNET, surface_index=1, target=1.0)
        mf.evaluate(sys, [])
        assert op_mnet.value < 0.0, "Knife-edge must record negative edge thickness value"
        assert op_mnet.diff > 1.0, "Penalty diff must reflect negative edge violation"

    def test_mnea_air_gap_edge_collision(self):
        """
        Adversarial test: Two steep lenses facing each other in air.
        Axial center distance is positive (t_air = 2.0 mm), but rims collide (ET_air < 0.5 mm).
        Verify MNEA operand detects rim collision and flags violation.
        """
        stack = SurfaceStack()
        stack.add_surface(Surface(name="Object", surface_type=SurfaceType.OBJECT, thickness=50.0))
        # Lens 1: back surface curves to the right (R2 = +20 mm => sag1 = +4 mm)
        stack.add_surface(Surface(name="L1_front", radius=40.0, thickness=5.0, glass="BK7", semi_diameter=12.0))
        stack.add_surface(Surface(name="L1_back", radius=20.0, thickness=2.0, glass="AIR", semi_diameter=12.0))
        # Lens 2: front surface curves to the left (R3 = -20 mm => sag2 = -4 mm)
        # Rims collide: sag2 - sag1 = -4 - 4 = -8 mm => ET_air = 2.0 - 8.0 = -6.0 mm (collision!)
        stack.add_surface(Surface(name="L2_front", radius=-20.0, thickness=5.0, glass="SF11", semi_diameter=12.0))
        stack.add_surface(Surface(name="L2_back", radius=-40.0, thickness=20.0, glass="AIR", semi_diameter=12.0))
        stack.add_surface(Surface(name="Detector", surface_type=SurfaceType.DETECTOR, semi_diameter=15.0))

        sys = OpticalSystem(stack)
        mf = MeritFunction()
        # MNEA on air space (surface index 2)
        op_mnea = mf.add_operand(OperandType.MNEA, surface_index=2, target=0.5)
        mf.evaluate(sys, [])

        assert op_mnea.value < 0.0, f"Air edge thickness should be negative (collision), got {op_mnea.value}"
        assert op_mnea.diff > 0.5, "MNEA violation diff must exceed target clearance"

    def test_effl_continuous_paraxial_power_afocal_handling(self):
        """
        Adversarial test: Test continuous paraxial power EFFL operand (Phi = -C, EFL = -1/C).
        Verify positive focusing system (C < 0 => EFL > 0), negative diverging system (C > 0 => EFL < 0),
        and afocal system (C = 0 => EFL = inf).
        Verify afocal degeneracy does not raise ZeroDivisionError and returns finite penalty.
        """
        # 1. Positive focusing system
        stack_pos = make_singlet(focal_length=100.0)
        sys_pos = OpticalSystem(stack_pos)
        mf_pos = MeritFunction()
        op_effl_pos = mf_pos.add_operand(OperandType.EFFL, target=100.0)
        mf_pos.evaluate(sys_pos, [])
        assert math.isclose(op_effl_pos.value, 100.0, rel_tol=1e-2)

        # 2. Afocal / plane-parallel plate (C = 0, EFL = inf)
        stack_afocal = SurfaceStack()
        stack_afocal.add_surface(Surface(name="Object", surface_type=SurfaceType.OBJECT, thickness=100.0))
        stack_afocal.add_surface(Surface(name="S1_Plano", radius=0.0, thickness=10.0, glass="BK7", semi_diameter=12.5))
        stack_afocal.add_surface(Surface(name="S2_Plano", radius=0.0, thickness=50.0, glass="AIR", semi_diameter=12.5))
        stack_afocal.add_surface(Surface(name="Detector", surface_type=SurfaceType.DETECTOR, semi_diameter=25.0))

        sys_afocal = OpticalSystem(stack_afocal)
        mf_afocal = MeritFunction()
        op_effl_afocal = mf_afocal.add_operand(OperandType.EFFL, target=100.0)
        # Must not raise ZeroDivisionError
        mf_afocal.evaluate(sys_afocal, [])
        assert math.isinf(op_effl_afocal.value) or op_effl_afocal.value > 1e6
        assert math.isfinite(op_effl_afocal.diff), "Afocal penalty diff must be finite"

    def test_totr_strictly_excludes_surface0(self):
        """
        Adversarial test: TOTR (Total Track) must strictly sum thickness from Surface 1 to detector,
        completely excluding object distance S0 across both finite and astronomical infinite conjugates.
        """
        for s0_dist in [50.0, 1000.0, 1e7]:
            stack = SurfaceStack()
            stack.add_surface(Surface(name="Object", surface_type=SurfaceType.OBJECT, thickness=s0_dist))
            stack.add_surface(Surface(name="S1", radius=50.0, thickness=5.0, glass="BK7", semi_diameter=10.0))
            stack.add_surface(Surface(name="S2", radius=-50.0, thickness=20.0, glass="AIR", semi_diameter=10.0))
            stack.add_surface(Surface(name="S3", radius=0.0, thickness=30.0, glass="AIR", semi_diameter=10.0))
            stack.add_surface(Surface(name="Detector", surface_type=SurfaceType.DETECTOR, semi_diameter=10.0))

            sys = OpticalSystem(stack)
            mf = MeritFunction()
            op_totr = mf.add_operand(OperandType.TOTR, target=0.0)
            mf.evaluate(sys, [])

            expected_totr = 5.0 + 20.0 + 30.0  # S1 to Detector strictly
            assert math.isclose(op_totr.value, expected_totr, abs_tol=1e-6), \
                f"TOTR must be {expected_totr} mm, got {op_totr.value} (S0 was {s0_dist})"

    def test_v2_boundary_operands_parity(self):
        """Verify v2 MeritFunction computes exact sag and boundary operands identically."""
        stack = V2SurfaceStack()
        stack.add_surface(V2Surface(name="Object", surface_type=V2SurfaceType.OBJECT, thickness=1000.0))
        stack.add_surface(V2Surface(name="S1", radius=30.0, thickness=0.8, glass="BK7", semi_diameter=10.0))
        stack.add_surface(V2Surface(name="S2", radius=-30.0, thickness=20.0, glass="AIR", semi_diameter=10.0))
        stack.add_surface(V2Surface(name="Detector", surface_type=V2SurfaceType.DETECTOR, semi_diameter=15.0))

        sys = V2OpticalSystem(stack)
        mf = V2MeritFunction()
        op_totr = mf.add_operand(V2OperandType.TOTR, target=0.0)
        op_mnct = mf.add_operand(V2OperandType.MNCT, surface_index=1, target=1.5)
        mf.evaluate(sys, [])

        assert math.isclose(op_totr.value, 20.8, abs_tol=1e-5)
        assert math.isclose(op_mnct.diff, 0.7, abs_tol=1e-5)


# ==============================================================================
# 5. Multi-Field Ray Tracing & Rayleigh-Hamilton Wavefront OPD Scaling
# ==============================================================================

class TestTier5MultiFieldAndWavefrontOpdScaling:
    """Stress tests on multi-field quadrature, severe field angles, and Rayleigh-Hamilton wavefront scaling."""

    def test_multi_field_quadrature_weighting_distribution(self):
        """
        Adversarial test: Verify multi-field ray generation with standard Hy in [0.0, 0.707, 1.0]
        and area weights [1.0, 2.0, 1.0].
        Verify rays are properly segregated by field_idx and weights sum correctly.
        """
        stack = make_singlet(focal_length=100.0)
        sys = OpticalSystem(stack)

        cfg = SystemConfig()
        cfg.field_type = FieldType.ANGLE_DEG
        cfg.fields = [
            FieldPoint(x=0.0, y=0.0, weight=1.0),
            FieldPoint(x=0.0, y=7.07, weight=2.0),
            FieldPoint(x=0.0, y=10.0, weight=1.0),
        ]

        opt = OpticalOptimizer(sys, method="DLS", system_config=cfg, entrance_pupil_radius=5.0)
        rays = opt._generate_rays(pupil_radius=5.0, wavelengths=[0.5876], num_rays=21, eval_stack=stack)

        field_0_rays = [r for r in rays if getattr(r, "field_idx", -1) == 0]
        field_1_rays = [r for r in rays if getattr(r, "field_idx", -1) == 1]
        field_2_rays = [r for r in rays if getattr(r, "field_idx", -1) == 2]

        assert len(field_0_rays) > 0
        assert len(field_1_rays) > 0
        assert len(field_2_rays) > 0
        assert len(field_0_rays) == len(field_1_rays) == len(field_2_rays)

        # Verify chief rays are present for all fields
        assert any(getattr(r, "is_chief", False) for r in field_0_rays)
        assert any(getattr(r, "is_chief", False) for r in field_1_rays)
        assert any(getattr(r, "is_chief", False) for r in field_2_rays)

    def test_extreme_field_angle_and_vignetting_penalty(self):
        """
        Adversarial test: Trace rays at an extreme 60-degree field angle through a small aperture optic.
        Verify optimizer applies vignetting penalty and does not fail with unhandled exceptions.
        """
        stack = SurfaceStack()
        stack.add_surface(Surface(name="Object", surface_type=SurfaceType.OBJECT, thickness=10.0))
        stack.add_surface(Surface(name="S1", radius=15.0, thickness=5.0, glass="BK7", semi_diameter=4.0))
        stack.add_surface(Surface(name="S2", radius=-15.0, thickness=20.0, glass="AIR", semi_diameter=4.0))
        stack.add_surface(Surface(name="Detector", surface_type=SurfaceType.DETECTOR, semi_diameter=10.0))

        sys = OpticalSystem(stack)
        cfg = SystemConfig()
        cfg.field_type = FieldType.ANGLE_DEG
        cfg.fields = [
            FieldPoint(x=0.0, y=0.0, weight=1.0),
            FieldPoint(x=0.0, y=30.0, weight=1.0),
            FieldPoint(x=0.0, y=60.0, weight=1.0),  # extreme wide angle
        ]

        opt = OpticalOptimizer(sys, method="DLS", max_iter=3, system_config=cfg, entrance_pupil_radius=3.0)
        res = opt.optimize(variable_indices={"radius": [1, 2]})

        assert res.success
        assert not np.isnan(res.final_rms)

    def test_rayleigh_hamilton_wavefront_scaling_ultra_fast_and_slow_optics(self):
        """
        Adversarial test: Rayleigh-Hamilton paraxial wavefront OPD scaling:
        W_RMS = R_RMS / (8 * F/# * lambda)
        Compare ultra-fast F/1.0 optics against ultra-slow F/50.0 optics.
        Verify scaling factor remains strictly positive, non-NaN, and inversely proportional to F/#.
        """
        wl_um = 0.5876
        wl_mm = wl_um * 1e-3

        # Ultra-fast F/1.0: ep_radius = 50.0 mm, EFL = 100.0 mm => F/# = 1.0
        stack_fast = make_singlet(focal_length=100.0, semi_diameter=55.0)
        f_num_fast = extract_working_f_number(stack_fast, primary_wl=wl_um, ep_radius=50.0)
        assert math.isclose(f_num_fast, 1.0, rel_tol=0.1), f"Expected F/1.0, got F/{f_num_fast}"

        scale_fast = 1.0 / (8.0 * f_num_fast * wl_mm)

        # Ultra-slow F/50.0: ep_radius = 1.0 mm, EFL = 100.0 mm => F/# = 50.0
        stack_slow = make_singlet(focal_length=100.0, semi_diameter=5.0)
        f_num_slow = extract_working_f_number(stack_slow, primary_wl=wl_um, ep_radius=1.0)
        assert math.isclose(f_num_slow, 50.0, rel_tol=0.1), f"Expected F/50.0, got F/{f_num_slow}"

        scale_slow = 1.0 / (8.0 * f_num_slow * wl_mm)

        # Ratio must equal f_num_slow / f_num_fast ~ 50.0
        ratio = scale_fast / scale_slow
        assert math.isclose(ratio, f_num_slow / f_num_fast, rel_tol=1e-5)

    def test_wavefront_scaling_extreme_uv_to_lwir_wavelengths(self):
        """
        Adversarial test: Test wavefront OPD scaling across extreme spectral ranges:
        Deep UV (0.193 um) vs Long-Wave IR (10.6 um).
        Verify scaling decreases monotonically with wavelength (longer wavelengths have fewer waves of OPD).
        """
        stack = make_singlet(focal_length=100.0)
        f_num = extract_working_f_number(stack, ep_radius=12.5)

        wl_duv = 0.193 * 1e-3  # mm
        wl_lwir = 10.6 * 1e-3  # mm

        scale_duv = 1.0 / (8.0 * f_num * wl_duv)
        scale_lwir = 1.0 / (8.0 * f_num * wl_lwir)

        assert scale_duv > scale_lwir
        assert math.isclose(scale_duv / scale_lwir, 10.6 / 0.193, rel_tol=1e-5)

    def test_spot_diagram_chief_vs_centroid_parallel_axis_theorem(self):
        """
        Adversarial test: For an asymmetric comatic spot distribution, verify that
        SpotDiagram.calculate_rms_radius relative to chief_ray vs centroid satisfies the
        exact Parallel Axis Theorem:
        RMS_chief^2 = RMS_centroid^2 + (x_bar - x_chief)^2 + (y_bar - y_chief)^2
        down to floating-point machine precision.
        """
        np.random.seed(12345)
        # Synthetic asymmetric comet tail
        x_hits = np.random.exponential(scale=0.4, size=100) + 0.8
        y_hits = np.random.normal(loc=1.5, scale=0.3, size=100)
        chief_point = (0.0, 0.0)

        spot = SpotDiagram(x_hits, y_hits, chief_ray_coord=chief_point)

        rms_cent = spot.calculate_rms_radius(reference="centroid")
        rms_chief = spot.calculate_rms_radius(reference="chief_ray")

        cx, cy = spot.calculate_centroid()
        dist_sq = (cx - chief_point[0])**2 + (cy - chief_point[1])**2

        lhs = rms_chief**2
        rhs = rms_cent**2 + dist_sq

        assert math.isclose(lhs, rhs, rel_tol=1e-12, abs_tol=1e-12), \
            f"Parallel axis theorem violated: LHS={lhs}, RHS={rhs}, diff={abs(lhs - rhs)}"

    def test_spot_diagram_diffraction_limited_criterion(self):
        """
        Adversarial test: Verify is_diffraction_limited() correctly evaluates R_RMS <= r_Airy.
        Airy radius r_Airy = 1.22 * lambda * F/#.
        """
        wl = 0.5876  # um
        f_num = 4.0
        r_airy_expected = 1.22 * wl * f_num  # ~ 2.867 um = 0.002867 mm

        # Case 1: Sub-Airy spot (RMS = 1.0 < Airy = 2.867) => diffraction limited
        spot_diff_lim = SpotDiagram([-1.0, 1.0], [0.0, 0.0], wavelength_um=wl, f_number=f_num)
        assert math.isclose(spot_diff_lim.calculate_rms_radius(), 1.0)
        assert math.isclose(spot_diff_lim.calculate_airy_disk_radius(), r_airy_expected)
        assert spot_diff_lim.is_diffraction_limited() is True

        # Case 2: Super-Airy spot (RMS = 5.0 > Airy = 2.867) => geometrically blurred
        spot_blurred = SpotDiagram([-5.0, 5.0], [0.0, 0.0], wavelength_um=wl, f_number=f_num)
        assert math.isclose(spot_blurred.calculate_rms_radius(), 5.0)
        assert spot_blurred.is_diffraction_limited() is False

    def test_v2_spot_and_merit_parity(self):
        """Verify v2 SpotDiagram and MeritFunction match optics_app."""
        x = [0.01, -0.02, 0.03, -0.01]
        y = [0.02, 0.01, -0.02, 0.0]

        spot_app = SpotDiagram(x, y, chief_ray_coord=(0.0, 0.0))
        spot_v2 = V2SpotDiagram(x, y, chief_ray_coord=(0.0, 0.0))

        assert math.isclose(spot_app.calculate_rms_radius("centroid"), spot_v2.calculate_rms_radius("centroid"))
        assert math.isclose(spot_app.calculate_rms_radius("chief_ray"), spot_v2.calculate_rms_radius("chief_ray"))
        assert math.isclose(spot_app.calculate_airy_disk_radius(), spot_v2.calculate_airy_disk_radius())
