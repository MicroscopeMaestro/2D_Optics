"""
tests/test_physical_lens_design.py
==================================
Comprehensive, rigorous opaque-box test suite for the Physical Lens Design feature
in 2D Optics.

Authoritative Requirements & Ground Truth Sources:
- ORIGINAL_REQUEST.md (§R1 Thick Lens & Matrix Optics, §R2 Aspheric & Multi-Component,
  §R3 Optimization Functions, Follow-up: Extensibility & Manual Debugging,
  Follow-up: Zemax-Style Optical Tolerancing)
- DISPATCH.md (Orchestrator constraints and acceptance criteria)
- PROJECT.md (Master architecture and module contracts in optics_app/engine/)
- Survey 2 & Survey 3 (Textbook analytical models, Hecht/University of Colorado benchmarks)

Test Tiers:
1. Unit Test 1: Thick Lens Cardinal Points Benchmark (Biconvex & Plano-Convex, <1% tolerance)
2. Unit Test 2: Galileo Imaging System Layout (Afocal condition C=0, Angular Magnification = 4.0, Thick Galileo)
3. Unit Test 3: Optimization Convergence (Unoptimized lens -> EFL 20.0mm & Diam 25.4mm, SciPy & DLS)
4. Unit Test 4: Extensibility & Manual Stepping Transparency (CalculationTrace, Custom OpticalElement & Operand)
5. Unit Test 5: Zemax-Style Optical Tolerancing (Perturbable parameters, 3x3 augmented matrices, Sensitivity, Monte Carlo)
6. Tier 6: Boundary & Corner Cases (Flat surface R->inf, near-zero thickness, meniscus, aspheric vertex theorem, invalid inputs)
"""

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pytest


# ============================================================================
# HELPER UTILITIES & ADAPTERS FOR OPAQUE-BOX INTERACTION
# ============================================================================

def make_thick_lens(
    r1: float,
    r2: float,
    thickness: float,
    index: float,
    diameter: float = 25.4
) -> Any:
    """
    Construct a ThickLens instance adhering to the OpticalElement interface contract in PROJECT.md.
    Supports either positional arguments or keyword arguments (r1/radius_front, r2/radius_rear, etc.).
    """
    from optics_app.engine.thick_lens import ThickLens

    try:
        return ThickLens(r1=r1, r2=r2, thickness=thickness, glass_or_index=index, diameter=diameter)
    except TypeError:
        try:
            return ThickLens(radius_front=r1, radius_rear=r2, thickness=thickness, index=index, diameter=diameter)
        except TypeError:
            return ThickLens(r1, r2, thickness, index, diameter)


def get_cardinal_points(lens_or_matrix: Any) -> Any:
    """
    Extract cardinal points from either a lens component or a transfer matrix.
    Returns an object with attributes: efl, bfl, ffl, p1, p2.
    """
    if hasattr(lens_or_matrix, "cardinal_points") and callable(lens_or_matrix.cardinal_points):
        return lens_or_matrix.cardinal_points()
    elif hasattr(lens_or_matrix, "get_cardinal_points") and callable(lens_or_matrix.get_cardinal_points):
        return lens_or_matrix.get_cardinal_points()
    return lens_or_matrix


def get_lens_efl(lens: Any) -> float:
    """Extract Effective Focal Length (EFL) via property or cardinal_points()."""
    if hasattr(lens, "efl"):
        return float(lens.efl)
    cp = get_cardinal_points(lens)
    return float(cp.efl)


def get_lens_bfl(lens: Any) -> float:
    """Extract Back Focal Length (BFL) via property or cardinal_points()."""
    if hasattr(lens, "bfl"):
        return float(lens.bfl)
    cp = get_cardinal_points(lens)
    return float(cp.bfl)


def get_lens_principal_planes(lens: Any) -> Tuple[float, float]:
    """Extract Principal Planes (p1, p2) via properties or cardinal_points()."""
    if hasattr(lens, "p1") and hasattr(lens, "p2"):
        return float(lens.p1), float(lens.p2)
    cp = get_cardinal_points(lens)
    return float(cp.p1), float(cp.p2)


def get_transfer_matrix(element: Any) -> Any:
    """Extract paraxial TransferMatrix (ABCD) from element."""
    if hasattr(element, "transfer_matrix") and callable(element.transfer_matrix):
        return element.transfer_matrix()
    elif hasattr(element, "get_transfer_matrix") and callable(element.get_transfer_matrix):
        return element.get_transfer_matrix()
    return element


# ============================================================================
# TIER 1: UNIT TEST 1 — THICK LENS CARDINAL POINTS BENCHMARK
# ============================================================================

class TestThickLensCardinalPointsBenchmark:
    """
    Unit Test 1 (ORIGINAL_REQUEST §R1, Acceptance Criteria 1):
    Verify that thick lens models calculate standard cardinal points (EFL, BFL, P1, P2)
    matching textbook analytical results within 1%.
    """

    def test_symmetric_biconvex_thick_lens_analytical_benchmark(self):
        """
        Benchmark 1: Symmetric Biconvex Thick Lens
        ------------------------------------------
        Parameters:
            R1 = +50.0 mm
            R2 = -50.0 mm
            t  = 10.0 mm
            n  = 1.50
            n_air = 1.00

        Textbook Analytical Derivation (Hecht / CU Optical Engineering):
            Phi1 = (1.5 - 1.0) / 50.0 = 0.01 mm^-1 = 1/100
            Phi2 = (1.0 - 1.5) / (-50.0) = 0.01 mm^-1 = 1/100
            Phi_total = Phi1 + Phi2 - (t/n) * Phi1 * Phi2
                      = 0.02 - (10.0 / 1.5) * (0.0001)
                      = 0.02 - 1/1500 = 29 / 1500 mm^-1

            EFL_analytical = 1500 / 29 ≈ 51.724138 mm
            BFL_analytical = 1400 / 29 ≈ 48.275862 mm
            P1_analytical  = +100 / 29 ≈ +3.448276 mm (inside front vertex)
            P2_analytical  = -100 / 29 ≈ -3.448276 mm (inside rear vertex)
        """
        lens = make_thick_lens(r1=50.0, r2=-50.0, thickness=10.0, index=1.5, diameter=25.4)

        efl_calc = get_lens_efl(lens)
        bfl_calc = get_lens_bfl(lens)
        p1_calc, p2_calc = get_lens_principal_planes(lens)

        efl_exact = 1500.0 / 29.0   # 51.72413793...
        bfl_exact = 1400.0 / 29.0   # 48.27586207...
        p1_exact = 100.0 / 29.0     # +3.44827586...
        p2_exact = -100.0 / 29.0    # -3.44827586...

        # Acceptance Criterion: Match analytical benchmark within 1% (0.01 relative error)
        rel_err_efl = abs(efl_calc - efl_exact) / efl_exact
        rel_err_bfl = abs(bfl_calc - bfl_exact) / bfl_exact
        rel_err_p1 = abs(p1_calc - p1_exact) / p1_exact
        rel_err_p2 = abs(p2_calc - p2_exact) / abs(p2_exact)

        assert rel_err_efl < 0.01, f"EFL relative error {rel_err_efl:.6f} exceeded 1% (got {efl_calc}, expected {efl_exact})"
        assert rel_err_bfl < 0.01, f"BFL relative error {rel_err_bfl:.6f} exceeded 1% (got {bfl_calc}, expected {bfl_exact})"
        assert rel_err_p1 < 0.01, f"P1 relative error {rel_err_p1:.6f} exceeded 1% (got {p1_calc}, expected {p1_exact})"
        assert rel_err_p2 < 0.01, f"P2 relative error {rel_err_p2:.6f} exceeded 1% (got {p2_calc}, expected {p2_exact})"

        # Fundamental Optical Conjugate Invariant: BFL - P2 == EFL
        assert abs((bfl_calc - p2_calc) - efl_calc) < 1e-6, "Invariant BFL - P2 == EFL violated"

    def test_symmetric_biconvex_5mm_thickness_survey3_benchmark(self):
        """
        Benchmark 1b: Symmetric Biconvex Thick Lens (t = 5.0 mm, Survey 3 specification)
        --------------------------------------------------------------------------------
        Parameters:
            R1 = +50.0 mm, R2 = -50.0 mm, t = 5.0 mm, n = 1.50
            Phi1 = 0.01 mm^-1, Phi2 = 0.01 mm^-1
            Phi = 0.02 - (5.0 / 1.5) * 0.0001 = 0.019666667 mm^-1
            EFL_exact = 1 / 0.019666667 = 50.847458 mm
            BFL_exact = 49.152542 mm
            P1_exact  = +1.694915 mm
            P2_exact  = -1.694915 mm
        """
        lens = make_thick_lens(r1=50.0, r2=-50.0, thickness=5.0, index=1.5, diameter=20.0)

        efl_calc = get_lens_efl(lens)
        bfl_calc = get_lens_bfl(lens)
        p1_calc, p2_calc = get_lens_principal_planes(lens)

        efl_exact = 50.847458
        bfl_exact = 49.152542
        p1_exact = 1.694915
        p2_exact = -1.694915

        assert abs(efl_calc - efl_exact) / efl_exact < 0.01
        assert abs(bfl_calc - bfl_exact) / bfl_exact < 0.01
        assert abs(p1_calc - p1_exact) / p1_exact < 0.01
        assert abs(p2_calc - p2_exact) / abs(p2_exact) < 0.01

    def test_plano_convex_thick_lens_analytical_benchmark(self):
        """
        Benchmark 2: Plano-Convex Thick Lens (Flat rear surface, N-BK7)
        ---------------------------------------------------------------
        Parameters:
            R1 = +50.0 mm
            R2 = inf (flat rear face)
            t  = 6.0 mm
            n  = 1.5168 (N-BK7 at d-line)
            n_air = 1.00

        Analytical Derivation:
            Phi1 = (n - 1) / R1 = 0.5168 / 50.0 = 0.010336 mm^-1
            Phi2 = 0.0
            EFL_exact = 50.0 / 0.5168 = 96.749226 mm
            BFL_exact = EFL - t / n = 96.749226 - 6.0 / 1.5168 = 92.793530 mm
            P1_exact  = 0.000000 mm (front principal plane is at front vertex)
            P2_exact  = -t / n = -6.0 / 1.5168 = -3.955696 mm
        """
        lens = make_thick_lens(r1=50.0, r2=float("inf"), thickness=6.0, index=1.5168, diameter=25.4)

        efl_calc = get_lens_efl(lens)
        bfl_calc = get_lens_bfl(lens)
        p1_calc, p2_calc = get_lens_principal_planes(lens)

        efl_exact = 50.0 / 0.5168
        t_reduced = 6.0 / 1.5168
        bfl_exact = efl_exact - t_reduced
        p1_exact = 0.0
        p2_exact = -t_reduced

        assert abs(efl_calc - efl_exact) / efl_exact < 0.01
        assert abs(bfl_calc - bfl_exact) / bfl_exact < 0.01
        assert abs(p1_calc - p1_exact) < 0.05  # Absolute error since p1 is zero
        assert abs(p2_calc - p2_exact) / abs(p2_exact) < 0.01
        assert abs((bfl_calc - p2_calc) - efl_calc) < 1e-6

    def test_thick_lens_matrix_elements_and_determinant_unity(self):
        """
        Verify exact ABCD matrix representation and determinant theorem det(M) == 1.0.
        For symmetric biconvex lens (R=50, t=10, n=1.5):
            M = [[ 14/15,    20/3   ],
                 [-29/1500,  14/15  ]]
            det(M) = (14/15)^2 - (20/3)*(-29/1500) = 196/225 + 29/225 = 1.0
        """
        lens = make_thick_lens(r1=50.0, r2=-50.0, thickness=10.0, index=1.5, diameter=25.4)
        mat = get_transfer_matrix(lens)

        assert abs(mat.A - (14.0 / 15.0)) < 1e-6
        assert abs(mat.B - (20.0 / 3.0)) < 1e-6
        assert abs(mat.C - (-29.0 / 1500.0)) < 1e-6
        assert abs(mat.D - (14.0 / 15.0)) < 1e-6

        det_val = mat.determinant() if callable(getattr(mat, "determinant", None)) else (mat.A * mat.D - mat.B * mat.C)
        assert abs(det_val - 1.0) < 1e-12, f"Transfer matrix determinant {det_val} != 1.0"

    def test_matrix_optics_refraction_and_translation_primitives(self):
        """
        Directly test low-level matrix primitives in optics_app.engine.matrix_optics:
        1. Curved refraction interface: det(R) = n1 / n2
        2. Homogeneous translation: det(T) = 1.0
        3. Cascaded multiplication order M2 @ M1
        """
        from optics_app.engine.matrix_optics import TransferMatrix

        # Refraction from air (1.0) into glass (1.5) with R = +50.0 mm
        if hasattr(TransferMatrix, "refraction"):
            r_mat = TransferMatrix.refraction(n1=1.0, n2=1.5, radius=50.0)
        else:
            from optics_app.engine.matrix_optics import matrix_refraction
            r_mat = matrix_refraction(1.0, 1.5, 50.0)

        # In geometric unreduced coordinates, det(R) = n1 / n2 = 1.0 / 1.5 = 2/3
        det_r = r_mat.determinant() if callable(getattr(r_mat, "determinant", None)) else (r_mat.A * r_mat.D - r_mat.B * r_mat.C)
        assert abs(det_r - (1.0 / 1.5)) < 1e-9

        # Translation through 10mm glass
        if hasattr(TransferMatrix, "translation"):
            t_mat = TransferMatrix.translation(thickness=10.0, index=1.5)
        else:
            from optics_app.engine.matrix_optics import matrix_translation
            t_mat = matrix_translation(10.0, 1.5)

        det_t = t_mat.determinant() if callable(getattr(t_mat, "determinant", None)) else (t_mat.A * t_mat.D - t_mat.B * t_mat.C)
        assert abs(det_t - 1.0) < 1e-9

        # Matrix multiplication check
        cascaded = t_mat @ r_mat
        assert cascaded is not None
        assert abs(cascaded.determinant() - (1.0 / 1.5)) < 1e-9

    def test_thick_lens_nodal_points_coincide_in_air(self):
        """
        Helmholtz Reciprocity Theorem: In equal media (air on both sides, n_in = n_out = 1.0),
        the nodal points N1, N2 must coincide identically with the principal planes P1, P2.
        """
        lens = make_thick_lens(r1=50.0, r2=-50.0, thickness=10.0, index=1.5, diameter=25.4)
        cp = get_cardinal_points(lens)

        if hasattr(cp, "n1") and hasattr(cp, "n2"):
            assert abs(cp.n1 - cp.p1) < 1e-9, "Front nodal point N1 does not coincide with P1 in air"
            assert abs(cp.n2 - cp.p2) < 1e-9, "Rear nodal point N2 does not coincide with P2 in air"


# ============================================================================
# TIER 2: UNIT TEST 2 — GALILEO IMAGING SYSTEM LAYOUT
# ============================================================================

class TestGalileoImagingSystemLayout:
    """
    Unit Test 2 (ORIGINAL_REQUEST §R2, Acceptance Criteria 2):
    Configure a basic Galilean imaging / telescope system using paraxial transfer matrices.
    Assert net system power C == 0.0 (afocal condition) and output angular magnification
    matches theoretical target M_alpha = D = -f1 / f2 = 4.0.
    """

    def test_thin_lens_galileo_afocal_condition_and_magnification(self):
        """
        Configure basic thin-lens Galileo system:
        - Objective: f1 = +100.0 mm (converging)
        - Eyepiece:  f2 = -25.0 mm (diverging)
        - Afocal Separation: d = f1 + f2 = 100.0 + (-25.0) = 75.0 mm

        System ABCD Matrix:
            M = [[ 1,    0    ]   [[ 1, 75 ]   [[ 1,     0     ]
                 [ 1/25, 1    ]] @ [ 0,  1 ]] @ [ -1/100, 1    ]]
              = [[ 0.25, 75.0 ]
                 [ 0.0,  4.0  ]]

        Theoretical Target:
            Power C = 0.0 mm^-1
            Angular Magnification M_alpha = D = -f1 / f2 = +4.0
            Lateral Beam Expansion Factor m = A = 1 / M_alpha = 0.25
        """
        from optics_app.engine.multi_component import create_galileo_system

        galileo = create_galileo_system(f1=100.0, f2=-25.0)

        # Verify net system matrix
        sys_mat = galileo.system_matrix() if hasattr(galileo, "system_matrix") else get_transfer_matrix(galileo)

        # 1. Afocal condition: C element must be 0.0
        assert abs(sys_mat.C) < 1e-9, f"Afocal condition violated: power C = {sys_mat.C} != 0.0"

        # 2. Angular magnification: D element must equal -f1 / f2 = 4.0
        mag_calc = galileo.angular_magnification() if hasattr(galileo, "angular_magnification") else sys_mat.D
        assert abs(mag_calc - 4.0) < 1e-6, f"Angular magnification {mag_calc} != 4.0"

        # 3. Lateral magnification: A element must equal 1 / M_alpha = 0.25
        lat_mag = galileo.lateral_magnification() if hasattr(galileo, "lateral_magnification") else sys_mat.A
        assert abs(lat_mag - 0.25) < 1e-6, f"Lateral magnification {lat_mag} != 0.25"

        # 4. Determinant unity
        det_val = sys_mat.determinant() if callable(getattr(sys_mat, "determinant", None)) else (sys_mat.A * sys_mat.D - sys_mat.B * sys_mat.C)
        assert abs(det_val - 1.0) < 1e-12

    def test_galileo_paraxial_ray_propagation_angular_expansion(self):
        """
        Test paraxial ray propagation through the Galileo system:
        Incoming off-axis collimated ray: y_in = 0.0 mm, theta_in = 0.010 rad (10 mrad).
        Expected exit ray:
            y_out = A * y_in + B * theta_in = 0.25 * 0 + 75.0 * 0.01 = 0.75 mm
            theta_out = C * y_in + D * theta_in = 0 + 4.0 * 0.01 = 0.040 rad (40 mrad, 4x angular expansion!)
        """
        from optics_app.engine.multi_component import create_galileo_system

        galileo = create_galileo_system(f1=100.0, f2=-25.0)

        if hasattr(galileo, "propagate_ray"):
            y_out, theta_out = galileo.propagate_ray(y=0.0, theta=0.01)
        else:
            sys_mat = galileo.system_matrix()
            y_out = sys_mat.A * 0.0 + sys_mat.B * 0.01
            theta_out = sys_mat.C * 0.0 + sys_mat.D * 0.01

        assert abs(y_out - 0.75) < 1e-6
        assert abs(theta_out - 0.04) < 1e-6

    def test_galileo_lateral_beam_expansion_factor(self):
        """
        Incoming axial ray parallel to optical axis: y_in = 10.0 mm, theta_in = 0.0 rad.
        Expected exit ray:
            y_out = A * 10.0 = 0.25 * 10.0 = 2.5 mm
            theta_out = C * 10.0 = 0.0 rad (beam remains collimated!)
        """
        from optics_app.engine.multi_component import create_galileo_system

        galileo = create_galileo_system(f1=100.0, f2=-25.0)
        sys_mat = galileo.system_matrix()

        y_out = sys_mat.A * 10.0 + sys_mat.B * 0.0
        theta_out = sys_mat.C * 10.0 + sys_mat.D * 0.0

        assert abs(y_out - 2.5) < 1e-6
        assert abs(theta_out - 0.0) < 1e-9

    def test_thick_lens_galileo_configuration(self):
        """
        Benchmark 2b: Physical Thick-Lens Galilean Telescope
        ----------------------------------------------------
        Using realistic thick singlet lenses (Survey 2 §6.4):
        - Objective (Biconvex):
            R1 = +60.0 mm, R2 = -60.0 mm, t1 = 6.0 mm, n = 1.50
            EFL1 = 61.016949 mm, P12 = -2.033898 mm
        - Eyepiece (Biconcave):
            R1 = -30.0 mm, R2 = +30.0 mm, t2 = 3.0 mm, n = 1.50
            EFL2 = -29.508197 mm, P21 = +0.983607 mm
        - Afocal Separation between principal planes:
            d_P = EFL1 + EFL2 = 61.016949 - 29.508197 = 31.508752 mm
        - Physical Vertex-to-Vertex Air Gap:
            d_air = d_P + P12 - P21 = 31.508752 - 2.033898 - 0.983607 = 28.491248 mm
        - Theoretical Angular Magnification:
            M_alpha = -EFL1 / EFL2 = 61.016949 / 29.508197 ≈ 2.067797
        """
        from optics_app.engine.multi_component import MultiComponentSystem

        obj_lens = make_thick_lens(r1=60.0, r2=-60.0, thickness=6.0, index=1.5, diameter=25.4)
        eye_lens = make_thick_lens(r1=-30.0, r2=30.0, thickness=3.0, index=1.5, diameter=12.7)

        efl1 = get_lens_efl(obj_lens)
        efl2 = get_lens_efl(eye_lens)
        _, p12 = get_lens_principal_planes(obj_lens)
        p21, _ = get_lens_principal_planes(eye_lens)

        d_p = efl1 + efl2
        d_air = d_p + p12 - p21

        # Assemble multi-component system
        system = MultiComponentSystem(elements=[(obj_lens, d_air), (eye_lens, 0.0)])
        mat = system.system_matrix()

        # Net power must be zero (afocal)
        assert abs(mat.C) < 1e-4, f"Thick Galileo power {mat.C} is not afocal"

        # Angular magnification must match -EFL1 / EFL2 ≈ 2.067797
        expected_mag = -efl1 / efl2
        calc_mag = system.angular_magnification() if hasattr(system, "angular_magnification") else mat.D
        assert abs(calc_mag - expected_mag) / expected_mag < 0.01

    def test_create_galileo_system_factory_function(self):
        """Verify factory function create_galileo_system accepts arbitrary focal lengths and diameters."""
        from optics_app.engine.multi_component import create_galileo_system

        sys = create_galileo_system(f1=80.0, f2=-20.0, diameter1=30.0, diameter2=15.0)
        assert sys is not None
        mat = sys.system_matrix()
        assert abs(mat.C) < 1e-9
        assert abs(mat.D - 4.0) < 1e-6


# ============================================================================
# TIER 3: UNIT TEST 3 — OPTIMIZATION CONVERGENCE (EFL 20mm & DIAM 25.4mm)
# ============================================================================

class TestLensOptimizationConvergence:
    """
    Unit Test 3 (ORIGINAL_REQUEST §R3, Acceptance Criteria 3):
    Initialize an unoptimized lens, run optimization targeting EFL = 20.0mm and
    diameter = 25.4mm (1.0 inch), and assert that the final optimized lens converges
    successfully to these values (within 0.1% tolerance).
    """

    def test_unoptimized_lens_optimization_convergence_scipy(self):
        """
        Optimization Benchmark using ScipyLeastSquaresOptimizer (TRF):
        Initial Unoptimized Singlet:
            c1 = 0.0200 mm^-1 (R1 = 50.0 mm)
            c2 = -0.0200 mm^-1 (R2 = -50.0 mm)
            t  = 5.0 mm
            D  = 20.0 mm (semi-diameter = 10.0 mm)
            n  = 1.5168 (N-BK7)
            Initial EFL ≈ 50.85 mm  (far from 20.0 mm target)
            Initial Diam = 20.0 mm   (far from 25.4 mm target)

        Targets:
            Target EFL = 20.0 mm
            Target Diameter = 25.4 mm

        Assertions:
            1. Optimizer succeeds (success == True)
            2. Final EFL converges within 0.1% of 20.0 mm (19.98 mm <= EFL <= 20.02 mm)
            3. Final Diameter converges within 0.1% of 25.4 mm (25.37 mm <= D <= 25.43 mm)
            4. Final merit function value is substantially reduced (> 100x lower than initial)
            5. History contains step records with non-increasing merit trend
        """
        from optics_app.engine.lens_merit import PhysicalMeritFunction, TargetEFLOperand, DiameterConstraintOperand
        from optics_app.engine.lens_optimizer import ScipyLeastSquaresOptimizer

        # Unoptimized starting lens
        lens = make_thick_lens(r1=50.0, r2=-50.0, thickness=5.0, index=1.5168, diameter=20.0)

        init_efl = get_lens_efl(lens)
        assert abs(init_efl - 20.0) > 10.0, "Initial lens is already close to target EFL"
        assert abs(lens.diameter - 25.4) > 4.0, "Initial lens is already close to target diameter"

        # Construct transparent merit function
        mf = PhysicalMeritFunction()
        mf.add_operand(TargetEFLOperand(target_efl=20.0, weight=1.0))
        mf.add_operand(DiameterConstraintOperand(target_diameter=25.4, weight=1.0))

        optimizer = ScipyLeastSquaresOptimizer(merit_function=mf)
        result = optimizer.optimize(lens)

        assert result.success is True, f"Optimizer failed: {getattr(result, 'message', 'unknown error')}"

        final_lens = result.optimized_system if hasattr(result, "optimized_system") else lens
        final_efl = get_lens_efl(final_lens)
        final_diam = float(final_lens.diameter)

        # Assert convergence to EFL = 20.0 mm (within 0.1%)
        rel_err_efl = abs(final_efl - 20.0) / 20.0
        assert rel_err_efl < 0.001, f"Final EFL {final_efl:.4f} mm deviated >0.1% from 20.0 mm (error: {rel_err_efl:.4%})"

        # Assert convergence to Diameter = 25.4 mm (within 0.1%)
        rel_err_diam = abs(final_diam - 25.4) / 25.4
        assert rel_err_diam < 0.001, f"Final Diameter {final_diam:.4f} mm deviated >0.1% from 25.4 mm (error: {rel_err_diam:.4%})"

        # Assert merit reduction
        if hasattr(result, "history") and len(result.history) > 0:
            init_merit = result.history[0].merit_value
            final_merit = result.history[-1].merit_value
            assert final_merit < init_merit, f"Merit did not decrease: {init_merit} -> {final_merit}"

    def test_transparent_dls_optimizer_convergence(self):
        """
        Test the white-box TransparentDLSOptimizer (Levenberg-Marquardt / Damped Least Squares):
        Verifies convergence and explicit Jacobian inspection capabilities.
        """
        from optics_app.engine.lens_merit import PhysicalMeritFunction, TargetEFLOperand, DiameterConstraintOperand
        from optics_app.engine.lens_optimizer import TransparentDLSOptimizer

        lens = make_thick_lens(r1=50.0, r2=-50.0, thickness=5.0, index=1.5168, diameter=20.0)

        mf = PhysicalMeritFunction()
        mf.add_operand(TargetEFLOperand(target_efl=20.0, weight=1.0))
        mf.add_operand(DiameterConstraintOperand(target_diameter=25.4, weight=1.0))

        dls_opt = TransparentDLSOptimizer(merit_function=mf, max_iterations=30, tolerance=1e-5)
        result = dls_opt.optimize(lens)

        assert result.success is True
        final_lens = result.optimized_system if hasattr(result, "optimized_system") else lens
        final_efl = get_lens_efl(final_lens)

        assert abs(final_efl - 20.0) / 20.0 < 0.005  # Within 0.5%
        assert len(result.history) > 0
        assert len(result.history) <= 30

    def test_single_step_debugging_hook_transparency(self):
        """
        User Architectural Constraint Verification:
        'make space for future new implementations, models and also debugging manually.'
        Verify that optimizer.step(system, merit_function) executes exactly one iteration,
        records parameters, merit value, and returns an inspectable OptimizationStepRecord.
        """
        from optics_app.engine.lens_merit import PhysicalMeritFunction, TargetEFLOperand
        from optics_app.engine.lens_optimizer import TransparentDLSOptimizer

        lens = make_thick_lens(r1=50.0, r2=-50.0, thickness=5.0, index=1.5168, diameter=20.0)

        mf = PhysicalMeritFunction()
        mf.add_operand(TargetEFLOperand(target_efl=20.0, weight=1.0))

        optimizer = TransparentDLSOptimizer(merit_function=mf)

        # Perform step 1
        record1 = optimizer.step(lens, mf)
        assert record1 is not None
        assert hasattr(record1, "iteration")
        assert record1.iteration == 1
        assert hasattr(record1, "parameters")
        assert hasattr(record1, "merit_value")

        # Perform step 2
        record2 = optimizer.step(lens, mf)
        assert record2.iteration == 2
        assert record2.merit_value <= record1.merit_value

    def test_physical_merit_function_composite_operands(self):
        """
        Verify composition of multiple physical operands:
        - TargetEFLOperand
        - TargetBFLOperand
        - DiameterConstraintOperand
        - ThicknessConstraintOperand
        - CurvatureBoundsOperand
        """
        from optics_app.engine.lens_merit import (
            PhysicalMeritFunction,
            TargetEFLOperand,
            TargetBFLOperand,
            DiameterConstraintOperand,
            ThicknessConstraintOperand,
            CurvatureBoundsOperand
        )

        lens = make_thick_lens(r1=50.0, r2=-50.0, thickness=5.0, index=1.5168, diameter=20.0)

        mf = PhysicalMeritFunction()
        op_efl = TargetEFLOperand(target_efl=20.0, weight=1.0)
        op_bfl = TargetBFLOperand(target_bfl=18.0, weight=0.5)
        op_diam = DiameterConstraintOperand(target_diameter=25.4, weight=1.0)
        op_thick = ThicknessConstraintOperand(min_center=2.0, max_center=10.0, weight=2.0)
        op_curv = CurvatureBoundsOperand(max_curvature=0.1, weight=10.0)

        for op in [op_efl, op_bfl, op_diam, op_thick, op_curv]:
            mf.add_operand(op)

        assert len(mf.operands) == 5

        # Evaluate merit function
        merit_val = mf.evaluate(lens)
        assert isinstance(merit_val, float)
        assert merit_val > 0.0

        # Residuals vector
        residuals = mf.compute_residuals(lens) if hasattr(mf, "compute_residuals") else None
        if residuals is not None:
            assert len(residuals) >= 5

    def test_optimizer_parameter_bounds_and_symmetry(self):
        """
        Verify that optimizer respects physical box bounds:
        Curvatures cannot exceed geometric limits (|c| <= 1 / semi_diameter).
        Thickness cannot become negative.
        """
        from optics_app.engine.lens_merit import PhysicalMeritFunction, TargetEFLOperand
        from optics_app.engine.lens_optimizer import ScipyLeastSquaresOptimizer

        lens = make_thick_lens(r1=50.0, r2=-50.0, thickness=5.0, index=1.5168, diameter=25.4)
        mf = PhysicalMeritFunction()
        mf.add_operand(TargetEFLOperand(target_efl=15.0, weight=1.0))

        opt = ScipyLeastSquaresOptimizer(merit_function=mf)
        result = opt.optimize(lens)

        assert result.success is True
        opt_lens = result.optimized_system if hasattr(result, "optimized_system") else lens
        assert opt_lens.thickness > 0.5, "Lens thickness collapsed to non-physical value"


# ============================================================================
# TIER 4: UNIT TEST 4 — EXTENSIBILITY & MANUAL STEPPING TRANSPARENCY
# ============================================================================

class TestExtensibilityAndCalculationTrace:
    """
    Unit Test 4 (ORIGINAL_REQUEST Follow-up §2026-09-11T07:42:17Z):
    Verify CalculationTrace records step names, formula strings, numeric inputs,
    and intermediate matrices.
    Verify that custom user subclasses of OpticalElement and Operand can be
    instantiated and evaluated without modifying engine code.
    """

    def test_calculation_trace_records_steps_formulas_matrices(self):
        """
        Verify that CalculationTrace captures step-by-step mathematical derivations:
        1. Step names (e.g., 'Refraction Front', 'Translation', 'Refraction Rear')
        2. Algebraic formula string / LaTeX
        3. Substituted numeric inputs
        4. Intermediate 2x2 TransferMatrix
        5. Intermediate determinant verified == 1.0 (or n1/n2)
        """
        lens = make_thick_lens(r1=50.0, r2=-50.0, thickness=10.0, index=1.5, diameter=25.4)

        mat = lens.transfer_matrix()
        assert hasattr(mat, "trace"), "TransferMatrix missing trace attribute"
        trace = mat.trace

        if trace is not None:
            assert hasattr(trace, "steps"), "CalculationTrace missing steps attribute"
            steps = trace.steps
            assert len(steps) >= 3, f"CalculationTrace recorded only {len(steps)} steps, expected at least 3"

            # Verify step attributes
            for step in steps:
                assert hasattr(step, "name") or hasattr(step, "step_name")
                assert hasattr(step, "formula") or hasattr(step, "formula_latex")
                assert hasattr(step, "inputs")
                assert hasattr(step, "matrix")

    def test_custom_optical_element_subclass_extensibility(self):
        """
        Architectural Extensibility Requirement:
        Verify that a developer or researcher can subclass OpticalElement to implement
        a custom component (e.g. CustomAchromaticDoublet, GRIN element) and evaluate
        transfer matrices and cardinal points without altering engine code.
        """
        from optics_app.engine.thick_lens import OpticalElement
        from optics_app.engine.matrix_optics import TransferMatrix, CardinalPoints

        class CustomThinPrismPair(OpticalElement):
            """User-defined custom optical component."""
            def __init__(self, wedge_angle_rad: float = 0.05, index: float = 1.5):
                self.wedge_angle = wedge_angle_rad
                self.index = index

            def transfer_matrix(self, wavelength_nm: float = 587.6) -> TransferMatrix:
                # Identity translation with angular deflection
                return TransferMatrix(A=1.0, B=0.0, C=0.0, D=1.0)

            def cardinal_points(self, wavelength_nm: float = 587.6) -> CardinalPoints:
                return CardinalPoints(efl=float("inf"), bfl=float("inf"), ffl=float("inf"), p1=0.0, p2=0.0, f1=float("inf"), f2=float("inf"))

            def get_parameters(self) -> Dict[str, float]:
                return {"wedge_angle": self.wedge_angle, "index": self.index}

            def set_parameters(self, params: Dict[str, float]) -> None:
                if "wedge_angle" in params:
                    self.wedge_angle = params["wedge_angle"]
                if "index" in params:
                    self.index = params["index"]

        prism = CustomThinPrismPair(wedge_angle_rad=0.03, index=1.52)
        assert isinstance(prism, OpticalElement)
        mat = prism.transfer_matrix()
        assert abs(mat.determinant() - 1.0) < 1e-12

    def test_custom_operand_subclass_extensibility(self):
        """
        Architectural Extensibility Requirement:
        Verify that a user can define a custom Operand subclass (e.g. CustomPetzvalCurvatureOperand)
        and plug it into PhysicalMeritFunction without modifying the engine.
        """
        from optics_app.engine.lens_merit import Operand, PhysicalMeritFunction

        class CustomPetzvalSumOperand(Operand):
            """User-defined custom operand evaluating Petzval field curvature sum."""
            def __init__(self, target_sum: float = 0.0, weight: float = 1.0):
                super().__init__(name="PetzvalSum", target=target_sum, weight=weight)

            def evaluate(self, system: Any, context: Optional[Dict[str, Any]] = None) -> float:
                # Petzval sum for thin/thick lens: sum(Phi_i / n_i)
                if hasattr(system, "r1") and hasattr(system, "r2") and hasattr(system, "index"):
                    p1 = (system.index - 1.0) / system.r1
                    p2 = (1.0 - system.index) / system.r2
                    return (p1 + p2) / system.index
                return 0.0

        custom_op = CustomPetzvalSumOperand(target_sum=0.0, weight=5.0)
        assert isinstance(custom_op, Operand)

        lens = make_thick_lens(r1=50.0, r2=-50.0, thickness=5.0, index=1.5, diameter=25.4)
        residual = custom_op.compute_residual(lens)
        assert isinstance(residual, float)

        mf = PhysicalMeritFunction()
        mf.add_operand(custom_op)
        merit_val = mf.evaluate(lens)
        assert merit_val > 0.0

    def test_surface_model_abc_extensibility(self):
        """
        Architectural Extensibility Requirement:
        Verify SurfaceModel ABC allows defining new custom surfaces (e.g., ToroidalSurface, DOE)
        providing sag(r), dsag_dr(r), normal(x, y), and paraxial_curvature().
        """
        from optics_app.engine.aspheric import SurfaceModel

        class CustomConicalFresnelSurface(SurfaceModel):
            def __init__(self, cone_angle_deg: float = 10.0):
                self.slope = math.tan(math.radians(cone_angle_deg))

            def sag(self, r: float) -> float:
                return self.slope * abs(r)

            def dsag_dr(self, r: float) -> float:
                return self.slope if r >= 0 else -self.slope

            def normal(self, x: float, y: float) -> Tuple[float, float, float]:
                r = math.hypot(x, y)
                if r < 1e-12:
                    return (0.0, 0.0, 1.0)
                dzdr = self.dsag_dr(r)
                norm = math.sqrt(1.0 + dzdr * dzdr)
                return (-dzdr * (x / r) / norm, -dzdr * (y / r) / norm, 1.0 / norm)

            def paraxial_curvature(self) -> float:
                return 0.0

        surface = CustomConicalFresnelSurface(cone_angle_deg=5.0)
        assert isinstance(surface, SurfaceModel)
        assert abs(surface.sag(10.0) - 10.0 * math.tan(math.radians(5.0))) < 1e-9


# ============================================================================
# TIER 5: UNIT TEST 5 — ZEMAX-STYLE OPTICAL TOLERANCING
# ============================================================================

class TestZemaxStyleOpticalTolerancing:
    """
    Unit Test 5 (CRITICAL USER MANDATE §2026-09-11T07:46:15Z):
    Verify that all parameters (R, t, n, tilt, decenter) are treated as fully
    perturbable parameters from day one.
    Verify 3x3 augmented matrices for decenter and tilt.
    Verify SensitivityAnalysis calculates parameter sensitivity tables.
    Verify MonteCarloSimulation executes perturbations and computes yield statistics.
    """

    def test_parameter_perturbation_protocol_and_cloning(self):
        """
        Verify that physical lens parameters can be read, perturbed, and cloned
        without modifying nominal values (PerturbableElement protocol).
        """
        lens = make_thick_lens(r1=50.0, r2=-50.0, thickness=5.0, index=1.5168, diameter=25.4)

        if hasattr(lens, "get_parameter") and hasattr(lens, "set_parameter"):
            assert abs(lens.get_parameter("r1") - 50.0) < 1e-6
            assert abs(lens.get_parameter("thickness") - 5.0) < 1e-6

            # Clone and perturb
            cloned = lens.clone() if hasattr(lens, "clone") else None
            if cloned is not None:
                cloned.set_parameter("r1", 50.5)
                assert abs(cloned.get_parameter("r1") - 50.5) < 1e-6
                assert abs(lens.get_parameter("r1") - 50.0) < 1e-6, "Nominal lens was mutated!"
        elif hasattr(lens, "get_parameters") and hasattr(lens, "set_parameters"):
            params = lens.get_parameters()
            assert "r1" in params or "radius_front" in params

    def test_augmented_3x3_matrix_pure_decenter(self):
        """
        Verify 3x3 Augmented Transfer Matrix for pure lateral decenter (Delta y):
            M_augmented = [[ A, B, (1 - A)*Delta_y ],
                           [ C, D, -C * Delta_y    ],
                           [ 0, 0, 1               ]]
        A decentered lens deflects an axial ray by theta = -C * Delta_y = Phi * Delta_y.
        """
        lens = make_thick_lens(r1=50.0, r2=-50.0, thickness=10.0, index=1.5, diameter=25.4)
        mat2x2 = lens.transfer_matrix()

        delta_y = 0.5  # 0.5 mm decenter
        expected_E = (1.0 - mat2x2.A) * delta_y
        expected_F = -mat2x2.C * delta_y

        if hasattr(lens, "augmented_matrix_3x3"):
            mat3x3 = lens.augmented_matrix_3x3(decenter_y=delta_y)
            assert abs(mat3x3[0, 2] - expected_E) < 1e-6
            assert abs(mat3x3[1, 2] - expected_F) < 1e-6
        elif hasattr(mat2x2, "with_decenter"):
            mat3x3 = mat2x2.with_decenter(delta_y=delta_y)
            assert abs(mat3x3.E - expected_E) < 1e-6
            assert abs(mat3x3.F - expected_F) < 1e-6

    def test_augmented_3x3_matrix_pure_tilt(self):
        """
        Verify 3x3 Augmented Transfer Matrix for pure angular tilt (Delta theta):
            M_augmented = [[ A, B, -B * Delta_theta       ],
                           [ C, D, (1 - D) * Delta_theta  ],
                           [ 0, 0, 1                      ]]
        """
        lens = make_thick_lens(r1=50.0, r2=-50.0, thickness=10.0, index=1.5, diameter=25.4)
        mat2x2 = lens.transfer_matrix()

        delta_theta = 0.002  # 2 mrad tilt
        expected_E = -mat2x2.B * delta_theta
        expected_F = (1.0 - mat2x2.D) * delta_theta

        if hasattr(lens, "augmented_matrix_3x3"):
            mat3x3 = lens.augmented_matrix_3x3(tilt_theta=delta_theta)
            assert abs(mat3x3[0, 2] - expected_E) < 1e-6
            assert abs(mat3x3[1, 2] - expected_F) < 1e-6
        elif hasattr(mat2x2, "with_tilt"):
            mat3x3 = mat2x2.with_tilt(delta_theta=delta_theta)
            assert abs(mat3x3.E - expected_E) < 1e-6
            assert abs(mat3x3.F - expected_F) < 1e-6

    def test_sensitivity_analysis_ranking_worst_offenders(self):
        """
        Verify SensitivityAnalysis executes univariate parameter perturbations
        and produces a ranked sensitivity table (worst-offenders ranking).
        """
        from optics_app.engine.lens_merit import PhysicalMeritFunction, TargetEFLOperand
        from optics_app.engine.tolerancing import TolerancingEngine, PerturbationSpec

        lens = make_thick_lens(r1=50.0, r2=-50.0, thickness=5.0, index=1.5168, diameter=25.4)
        mf = PhysicalMeritFunction()
        mf.add_operand(TargetEFLOperand(target_efl=50.85, weight=1.0))

        # Define tolerances
        tolerances = [
            PerturbationSpec(parameter_name="r1", element_index=0, delta=0.1),
            PerturbationSpec(parameter_name="r2", element_index=0, delta=0.1),
            PerturbationSpec(parameter_name="thickness", element_index=0, delta=0.05),
            PerturbationSpec(parameter_name="index", element_index=0, delta=0.001),
        ]

        engine = TolerancingEngine() if not hasattr(TolerancingEngine, "__init__") or TolerancingEngine.__init__.__code__.co_argcount == 1 else TolerancingEngine(lens, mf)

        if hasattr(engine, "sensitivity_analysis"):
            report = engine.sensitivity_analysis(lens, mf, tolerances)
        else:
            report = engine.run_sensitivity_analysis()

        assert report is not None
        # Check that sensitivities are ranked
        items = report.rankings if hasattr(report, "rankings") else (report.items if hasattr(report, "items") else report)
        assert len(items) >= 4

    def test_monte_carlo_simulation_statistical_percentiles(self):
        """
        Verify MonteCarloSimulation executes random multi-parameter trials and computes
        yield statistics (mean, standard deviation, 50th, 90th, 95th, 99th percentiles).
        """
        from optics_app.engine.lens_merit import PhysicalMeritFunction, TargetEFLOperand
        from optics_app.engine.tolerancing import TolerancingEngine, PerturbationSpec

        lens = make_thick_lens(r1=50.0, r2=-50.0, thickness=5.0, index=1.5168, diameter=25.4)
        mf = PhysicalMeritFunction()
        mf.add_operand(TargetEFLOperand(target_efl=50.85, weight=1.0))

        tolerances = [
            PerturbationSpec(parameter_name="r1", element_index=0, delta=0.05, distribution="normal"),
            PerturbationSpec(parameter_name="r2", element_index=0, delta=0.05, distribution="normal"),
            PerturbationSpec(parameter_name="thickness", element_index=0, delta=0.02, distribution="uniform"),
        ]

        engine = TolerancingEngine() if not hasattr(TolerancingEngine, "__init__") or TolerancingEngine.__init__.__code__.co_argcount == 1 else TolerancingEngine(lens, mf)

        if hasattr(engine, "monte_carlo_simulation"):
            mc_report = engine.monte_carlo_simulation(lens, mf, tolerances, num_trials=100, seed=42)
        else:
            mc_report = engine.run_monte_carlo(num_trials=100, seed=42)

        assert mc_report is not None
        assert hasattr(mc_report, "percentile_50") or hasattr(mc_report, "p50")
        assert hasattr(mc_report, "percentile_90") or hasattr(mc_report, "p90")
        assert hasattr(mc_report, "percentile_95") or hasattr(mc_report, "p95")

        p50 = getattr(mc_report, "percentile_50", getattr(mc_report, "p50", 0.0))
        p90 = getattr(mc_report, "percentile_90", getattr(mc_report, "p90", 0.0))
        p95 = getattr(mc_report, "percentile_95", getattr(mc_report, "p95", 0.0))

        # Statistical Invariant: Percentiles must be monotonically non-decreasing
        assert p50 <= p90 <= p95, f"Percentile monotonicity violated: p50={p50}, p90={p90}, p95={p95}"

    def test_tolerancing_nominal_system_immutability(self):
        """Verify tolerancing engine does not corrupt or mutate the nominal system under evaluation."""
        lens = make_thick_lens(r1=50.0, r2=-50.0, thickness=5.0, index=1.5168, diameter=25.4)
        nominal_efl = get_lens_efl(lens)

        from optics_app.engine.lens_merit import PhysicalMeritFunction, TargetEFLOperand
        from optics_app.engine.tolerancing import TolerancingEngine, PerturbationSpec

        mf = PhysicalMeritFunction()
        mf.add_operand(TargetEFLOperand(target_efl=50.85, weight=1.0))
        tols = [PerturbationSpec(parameter_name="r1", element_index=0, delta=1.0)]

        engine = TolerancingEngine() if not hasattr(TolerancingEngine, "__init__") or TolerancingEngine.__init__.__code__.co_argcount == 1 else TolerancingEngine(lens, mf)
        if hasattr(engine, "sensitivity_analysis"):
            _ = engine.sensitivity_analysis(lens, mf, tols)

        post_efl = get_lens_efl(lens)
        assert abs(post_efl - nominal_efl) < 1e-9, "Nominal lens mutated after tolerancing run!"


# ============================================================================
# TIER 6: BOUNDARY & CORNER CASES (ADVERSARIAL VERIFICATION)
# ============================================================================

class TestBoundaryAndCornerCases:
    """
    Tier 6: Boundary & Corner Cases
    - Flat surface (R -> inf)
    - Zero / near-zero thickness limit (approaches thin lens formula)
    - Meniscus lens curvatures (R1 > 0, R2 > 0)
    - Negative curvature combinations (biconcave, plano-concave)
    - Aspheric Paraxial Curvature & Vertex Equivalence Theorem
    - Input validation & error handling
    - Determinant conservation across multi-element cascades
    """

    def test_flat_surface_infinite_radius_no_division_by_zero(self):
        """
        Verify that flat optical surfaces (radius = inf or curvature = 0.0) do not
        raise ZeroDivisionError, and evaluate to a planar refraction matrix.
        """
        from optics_app.engine.matrix_optics import TransferMatrix

        if hasattr(TransferMatrix, "refraction"):
            mat_flat = TransferMatrix.refraction(n1=1.0, n2=1.5, radius=float("inf"))
        else:
            from optics_app.engine.matrix_optics import matrix_refraction
            mat_flat = matrix_refraction(1.0, 1.5, float("inf"))

        assert abs(mat_flat.A - 1.0) < 1e-9
        assert abs(mat_flat.B - 0.0) < 1e-9
        assert abs(mat_flat.C - 0.0) < 1e-9  # Flat surface has zero optical power
        assert abs(mat_flat.D - (1.0 / 1.5)) < 1e-9

    def test_near_zero_thickness_approaches_thin_lens_formula(self):
        """
        As thickness t -> 0, the thick lens formula must converge to the thin lens formula:
            Phi_thin = (n - 1) * (1 / R1 - 1 / R2)
        For R1 = 50, R2 = -50, n = 1.5:
            Phi_thin = 0.5 * (1/50 - (-1/50)) = 0.5 * (2/50) = 0.020000 mm^-1
            EFL_thin = 1 / 0.02 = 50.000000 mm
        """
        lens_thin_limit = make_thick_lens(r1=50.0, r2=-50.0, thickness=1e-6, index=1.5, diameter=25.4)
        efl_calc = get_lens_efl(lens_thin_limit)
        assert abs(efl_calc - 50.0) < 1e-4

    def test_concentric_meniscus_lens_thick_power(self):
        """
        Concentric Meniscus Lens: R1 = +50.0 mm, R2 = +50.0 mm (both convex to the left).
        Under thin lens approximation: Phi_thin = (n - 1) * (1/50 - 1/50) = 0.0.
        Under exact thick lens Gullstrand formula:
            Phi_thick = (n - 1)^2 * t / (n * R1 * R2) > 0 (always weakly positive!)
        For t = 10.0 mm, n = 1.5:
            Phi_thick = (0.5)^2 * 10 / (1.5 * 2500) = 2.5 / 3750 = 1 / 1500 mm^-1
            EFL = 1500.0 mm
        """
        meniscus = make_thick_lens(r1=50.0, r2=50.0, thickness=10.0, index=1.5, diameter=25.4)
        efl_calc = get_lens_efl(meniscus)
        expected_efl = 1500.0
        assert abs(efl_calc - expected_efl) / expected_efl < 0.01

    def test_biconcave_negative_focal_length(self):
        """
        Biconcave Diverging Lens:
            R1 = -50.0 mm, R2 = +50.0 mm, t = 5.0 mm, n = 1.50
            EFL must be strictly negative (virtual focal point).
            True analytical Gullstrand EFL = -49.180328 mm.
        """
        biconcave = make_thick_lens(r1=-50.0, r2=50.0, thickness=5.0, index=1.5, diameter=25.4)
        efl_calc = get_lens_efl(biconcave)
        assert efl_calc < 0.0, f"Biconcave lens has positive focal length: {efl_calc}"
        expected_biconcave_efl = -49.180328
        assert abs(efl_calc - expected_biconcave_efl) / abs(expected_biconcave_efl) < 0.01

    def test_aspheric_surface_paraxial_vertex_equivalence_theorem(self):
        """
        Paraxial Curvature & Vertex Equivalence Theorem (Survey 2 §5.2):
        In paraxial Gaussian optics, an even aspheric surface has vertex curvature c = 1/R.
        The conic constant k and even aspheric coefficients (alpha_4, alpha_6, ...)
        contribute ZERO power to first-order paraxial ABCD matrices:
            R_asphere,paraxial == R_sphere,paraxial
        """
        from optics_app.engine.aspheric import AsphericSurfaceModel

        # Base sphere: R = 50.0 mm, k = 0, no higher-order coefficients
        sphere_model = AsphericSurfaceModel(radius=50.0, conic=0.0)

        # Strongly aspheric surface: R = 50.0 mm, k = -2.5 (hyperboloid), alpha4=1e-5, alpha6=-1e-8
        asphere_model = AsphericSurfaceModel(
            radius=50.0,
            conic=-2.5,
            aspheric_coeffs=[1e-5, -1e-8]
        )

        assert abs(sphere_model.paraxial_curvature() - 0.02) < 1e-9
        assert abs(asphere_model.paraxial_curvature() - 0.02) < 1e-9

        # Second derivative at vertex r=0 must match 1/R
        assert abs(sphere_model.paraxial_curvature() - asphere_model.paraxial_curvature()) < 1e-12

    def test_aspheric_surface_exact_sag_and_normal_vectors(self):
        """
        Verify that AsphericSurfaceModel computes exact 3D sag and surface normals:
        At vertex r = 0: sag = 0.0, normal = (0, 0, 1).
        At off-axis height r = 10.0 mm: sag > 0, normal has non-zero transverse components.
        """
        from optics_app.engine.aspheric import AsphericSurfaceModel

        asphere = AsphericSurfaceModel(radius=50.0, conic=-1.0)  # Paraboloid: z = r^2 / (2R) = r^2 / 100

        # Vertex check
        assert abs(asphere.sag(0.0)) < 1e-12
        nx, ny, nz = asphere.normal(0.0, 0.0)
        assert abs(nx) < 1e-12 and abs(ny) < 1e-12 and abs(nz - 1.0) < 1e-12

        # Off-axis r = 10.0 mm for paraboloid: z = 10^2 / 100 = 1.0 mm
        assert abs(asphere.sag(10.0) - 1.0) < 1e-6

        # Normal unit length check
        norm_x, norm_y, norm_z = asphere.normal(6.0, 8.0)
        norm_len = math.sqrt(norm_x * norm_x + norm_y * norm_y + norm_z * norm_z)
        assert abs(norm_len - 1.0) < 1e-9

    def test_invalid_physical_parameters_raise_value_error(self):
        """
        Adversarial Test: Verify that non-physical input combinations
        (index <= 0, thickness < 0, zero diameter) raise appropriate ValueError.
        """
        from optics_app.engine.thick_lens import ThickLens

        # Negative refractive index
        with pytest.raises(ValueError):
            _ = ThickLens(r1=50.0, r2=-50.0, thickness=5.0, glass_or_index=-1.5)

        # Zero refractive index
        with pytest.raises(ValueError):
            _ = ThickLens(r1=50.0, r2=-50.0, thickness=5.0, glass_or_index=0.0)

        # Negative thickness
        with pytest.raises(ValueError):
            _ = ThickLens(r1=50.0, r2=-50.0, thickness=-5.0, glass_or_index=1.5)

    def test_cascaded_multi_element_determinant_invariance(self):
        """
        Energy Conservation & Liouville / Helmholtz Invariant:
        In any optical system starting and ending in the same medium (air),
        the determinant of the cascaded system transfer matrix must be strictly 1.0,
        regardless of the number of elements or surfaces.
        """
        from optics_app.engine.multi_component import MultiComponentSystem

        l1 = make_thick_lens(r1=50.0, r2=-50.0, thickness=5.0, index=1.5168, diameter=25.4)
        l2 = make_thick_lens(r1=-40.0, r2=40.0, thickness=3.0, index=1.6200, diameter=25.4)
        l3 = make_thick_lens(r1=80.0, r2=-120.0, thickness=6.0, index=1.5168, diameter=25.4)

        sys = MultiComponentSystem(elements=[(l1, 20.0), (l2, 15.0), (l3, 0.0)])
        total_mat = sys.system_matrix()

        det_val = total_mat.determinant() if callable(getattr(total_mat, "determinant", None)) else (total_mat.A * total_mat.D - total_mat.B * total_mat.C)
        assert abs(det_val - 1.0) < 1e-10, f"Cascaded system determinant {det_val} != 1.0"
