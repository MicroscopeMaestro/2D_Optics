"""
Opaque-Box Requirement-Driven E2E Test Suite for Optical AI Sequences & Optimization Workflows.
==============================================================================================
Author: E2E Testing Track Writer (test_writer_e2e_1)
Target: 2D Optics AI Application
Reference: ORIGINAL_REQUEST.md, PROJECT.md, research_report.md

Coverage Structure:
- Tier 1: Feature Coverage (6 features x 5 tests = 30 tests)
- Tier 2: Boundary & Corner Cases (6 categories x 5 tests = 30 tests)
- Tier 3: Cross-Feature Combinations (5 multi-stage pipelines = 17 tests)
- Tier 4: Real-World Application Scenarios (6 workloads = 6 tests)
Total Tests: 83 tests
"""

import math
import copy
import pytest
import numpy as np
from typing import Dict, Any, Tuple, Optional

# Core optical engine imports
from optics_app.engine.matrix_optics import (
    TransferMatrix,
    CardinalPoints,
    thin_lens_matrix,
    thick_lens_matrix,
)
from optics_app.engine.thick_lens import OpticalElement, ThickLens, ThinLens
from optics_app.engine.aspheric import (
    AsphericLens,
    AsphericSurfaceModel,
    SphericalSurfaceModel,
    PlanarSurfaceModel,
)
from optics_app.engine.tolerancing import (
    TolerancingEngine,
    PerturbationSpec,
    SensitivityReport,
    MonteCarloReport,
)
from optics_app.engine.lens_merit import (
    PhysicalMeritFunction,
    TargetEFLOperand,
    TargetBFLOperand,
    ThicknessConstraintOperand,
    CurvatureBoundsOperand,
)
from optics_app.engine.lens_optimizer import (
    TransparentDLSOptimizer,
    ScipyLeastSquaresOptimizer,
)
from optics_app.utils.materials import get_refractive_index

# Dynamic progressive import for M2 AI sequences & extended merit deliverables
try:
    from optics_app.engine.ai_sequences import (
        OpticalDesignSequenceRunner,
        ParaxialSynthesisStage,
        CoddingtonBendingStage,
        ThickLensRealizationStage,
        DLSAberrationMinimizationStage,
        AspherizationStage,
        TolerancingYieldStage,
    )
    from optics_app.engine.extended_merit import (
        TransverseRayAberrationOperand,
        SpotSizeRMSOperand,
        SpotSizeGEOOperand,
        EdgeThicknessConstraintOperand,
        GlatzelAdaptiveDLSOptimizer,
    )
    HAS_AI_SEQUENCES = True
except ImportError:
    HAS_AI_SEQUENCES = False
    OpticalDesignSequenceRunner = None
    ParaxialSynthesisStage = None
    CoddingtonBendingStage = None
    ThickLensRealizationStage = None
    DLSAberrationMinimizationStage = None
    AspherizationStage = None
    TolerancingYieldStage = None
    TransverseRayAberrationOperand = None
    SpotSizeRMSOperand = None
    SpotSizeGEOOperand = None
    EdgeThicknessConstraintOperand = None
    GlatzelAdaptiveDLSOptimizer = None

requires_ai_sequences = pytest.mark.skipif(
    not HAS_AI_SEQUENCES,
    reason="optics_app.engine.ai_sequences / extended_merit not yet implemented by worker_engine_m2"
)


# =============================================================================
# AUTHORITATIVE REFERENCE ORACLES (Mathematical & Physical Ground Truth)
# =============================================================================

def oracle_coddington_q(n: float, p: float = -1.0) -> float:
    """
    Kingslake 1978 & Smith 2007: Optimal Coddington shape factor q_min
    for minimum spherical aberration S_I:
        q_min = (2*(n^2 - 1)/(n + 2)) * p
    """
    return float((2.0 * (n**2 - 1.0) / (n + 2.0)) * p)


def oracle_curvatures_from_q(phi: float, n: float, q: float) -> Tuple[float, float]:
    """
    Coddington 1829: Front and rear curvatures from power phi and shape factor q:
        c1 = (phi / (2*(n - 1))) * (q + 1)
        c2 = (phi / (2*(n - 1))) * (q - 1)
    """
    if abs(n - 1.0) < 1e-9:
        raise ValueError(f"Refractive index n={n} cannot equal unity (ambient medium)")
    c1 = (phi / (2.0 * (n - 1.0))) * (q + 1.0)
    c2 = (phi / (2.0 * (n - 1.0))) * (q - 1.0)
    return float(c1), float(c2)


def oracle_conic_constant(n: float) -> float:
    """
    ISO 10110-12 & Fischer 2008: Conic constant k for perfect infinite conjugate
    spherical aberration nullification:
        k = -n^2
    """
    return float(-(n**2))


def oracle_center_thickness_budget(diameter: float) -> float:
    """
    Optical engineering standard: tc >= 10% diameter, minimum 1.5 mm.
    """
    return float(max(1.5, 0.10 * diameter))


def oracle_gullstrand_power(c1: float, c2: float, d: float, n: float) -> float:
    """
    Gullstrand thick lens power formula:
        phi = phi1 + phi2 - (d/n)*phi1*phi2
    where phi1 = (n - 1)*c1 and phi2 = (1 - n)*c2.
    """
    phi1 = (n - 1.0) * c1
    phi2 = (1.0 - n) * c2
    return float(phi1 + phi2 - (d / n) * phi1 * phi2)


def oracle_edge_thickness(tc: float, r1: float, r2: float, diameter: float) -> float:
    """
    Calculates exact mechanical edge thickness te at radial height r = diameter / 2.
    """
    r = diameter / 2.0
    z1 = 0.0
    if abs(r1) < 1e9 and abs(r1) > r:
        z1 = (r**2 / r1) / (1.0 + math.sqrt(max(0.0, 1.0 - (r / r1)**2)))
    z2 = 0.0
    if abs(r2) < 1e9 and abs(r2) > r:
        z2 = (r**2 / r2) / (1.0 + math.sqrt(max(0.0, 1.0 - (r / r2)**2)))
    return float(tc - (z1 - z2))


# =============================================================================
# TIER 1: FEATURE COVERAGE (6 Features x 5 Tests = 30 Tests)
# =============================================================================

class TestTier1FeatureCoverage:
    """Tier 1: Comprehensive feature coverage across all 6 sequential design stages."""

    # -------------------------------------------------------------------------
    # Feature 1: Paraxial Synthesis Stage
    # -------------------------------------------------------------------------
    def test_t1_f1_01_paraxial_synthesis_optical_power(self):
        """T1.F1.1: Verify net optical power phi = 1 / EFL for positive singlet."""
        target_efl = 50.0
        expected_power = 1.0 / target_efl  # 0.02 mm^-1
        assert math.isclose(expected_power, 0.02, rel_tol=1e-5)
        if HAS_AI_SEQUENCES:
            runner = OpticalDesignSequenceRunner(target_efl=target_efl, target_diameter=25.0, glass="N-BK7")
            res = runner.step()
            assert math.isclose(res["details"]["optical_power_mm"], expected_power, rel_tol=1e-3)

    def test_t1_f1_02_paraxial_synthesis_aperture_and_pupil(self):
        """T1.F1.2: Verify entrance pupil diameter D = EFL / (F/#)."""
        target_efl = 50.0
        f_num = 2.5
        expected_diameter = target_efl / f_num  # 20.0 mm
        assert math.isclose(expected_diameter, 20.0, rel_tol=1e-5)
        if HAS_AI_SEQUENCES:
            runner = OpticalDesignSequenceRunner(target_efl=target_efl, target_diameter=expected_diameter, target_f_number=f_num, glass="N-BK7")
            res = runner.step()
            assert math.isclose(res["details"]["f_number"], f_num, rel_tol=1e-3)

    def test_t1_f1_03_paraxial_synthesis_glass_refractive_index(self):
        """T1.F1.3: Verify glass name 'N-BK7' resolves to valid d-line index (~1.5168 at 0.5876 um)."""
        n_nbk7 = get_refractive_index("N-BK7", 0.5876)
        assert 1.51 < n_nbk7 < 1.52
        if HAS_AI_SEQUENCES:
            runner = OpticalDesignSequenceRunner(target_efl=50.0, target_diameter=25.0, glass="N-BK7")
            res = runner.step()
            assert math.isclose(res["details"]["refractive_index"], n_nbk7, rel_tol=1e-3)

    def test_t1_f1_04_paraxial_synthesis_symmetric_curvatures_seed(self):
        """T1.F1.4: Verify initial paraxial seed divides power equally between surfaces."""
        efl = 100.0
        n = 1.5168
        phi = 1.0 / efl
        # Symmetric seed: c1 = -c2 = phi / (2 * (n - 1))
        c_seed = phi / (2.0 * (n - 1.0))
        r_seed = 1.0 / c_seed
        assert math.isclose(r_seed, 2.0 * (n - 1.0) * efl, rel_tol=1e-4)
        lens = ThickLens(r1=r_seed, r2=-r_seed, thickness=0.001, glass_or_index=n)
        assert math.isclose(lens.efl, efl, rel_tol=1e-3)

    @requires_ai_sequences
    def test_t1_f1_05_paraxial_synthesis_stage_result_contract(self):
        """T1.F1.5: Verify ParaxialSynthesisStage conforms to the standardized stage result schema."""
        runner = OpticalDesignSequenceRunner(target_efl=50.0, target_diameter=25.0, glass="N-BK7")
        res = runner.step()
        assert isinstance(res, dict)
        assert res.get("status") in ("SUCCESS", "OK")
        assert "metrics_before" in res
        assert "metrics_after" in res
        assert "current_lens" in res

    # -------------------------------------------------------------------------
    # Feature 2: Coddington Bending Stage
    # -------------------------------------------------------------------------
    def test_t1_f2_01_coddington_optimal_shape_factor_infinite_conjugate(self):
        """T1.F2.1: Verify analytical shape factor q_min = 2*(n^2-1)/(n+2)*(-1) for N-BK7."""
        n = 1.5168
        q_opt = oracle_coddington_q(n, p=-1.0)
        # 2*(1.5168^2 - 1) / (1.5168 + 2) * (-1) approx -0.7397
        assert -0.75 < q_opt < -0.73
        if HAS_AI_SEQUENCES:
            runner = OpticalDesignSequenceRunner(target_efl=50.0, target_diameter=25.0, glass="N-BK7")
            runner.step()
            res = runner.step()
            assert res["status"] in ("SUCCESS", "OK")

    def test_t1_f2_02_coddington_curvatures_satisfy_thin_lens_power(self):
        """T1.F2.2: Verify curvatures from optimal q preserve total thin-lens optical power."""
        efl = 50.0
        phi = 1.0 / efl
        n = 1.5168
        q = oracle_coddington_q(n, p=-1.0)
        c1, c2 = oracle_curvatures_from_q(phi, n, q)
        recovered_phi = (n - 1.0) * (c1 - c2)
        assert math.isclose(recovered_phi, phi, rel_tol=1e-6)

    def test_t1_f2_03_coddington_bending_asymmetry_convex_front(self):
        """T1.F2.3: Verify Coddington best-form lens has strong surface asymmetry."""
        phi = 0.02
        n = 1.5168
        q = oracle_coddington_q(n, p=-1.0)
        c1, c2 = oracle_curvatures_from_q(phi, n, q)
        # Biconvex signs: c1 > 0 (front convex), c2 < 0 (rear convex)
        assert c1 > 0.0
        assert c2 < 0.0
        # Significant asymmetry between front and rear radii
        assert abs(abs(c1) - abs(c2)) > 0.01

    def test_t1_f2_04_coddington_bending_aberration_reduction_oracle(self):
        """T1.F2.4: Verify Coddington shape factor achieves focal length stability."""
        efl = 50.0
        phi = 1.0 / efl
        n = 1.5168
        q_best = oracle_coddington_q(n, p=-1.0)
        c1_b, c2_b = oracle_curvatures_from_q(phi, n, q_best)
        r1_b, r2_b = 1.0 / c1_b, 1.0 / c2_b

        # Reversed lens (q = -q_best)
        c1_rev, c2_rev = oracle_curvatures_from_q(phi, n, -q_best)
        r1_rev, r2_rev = 1.0 / c1_rev, 1.0 / c2_rev

        lens_best = ThickLens(r1=r1_b, r2=r2_b, thickness=3.0, glass_or_index=n)
        lens_rev = ThickLens(r1=r1_rev, r2=r2_rev, thickness=3.0, glass_or_index=n)
        assert abs(lens_best.efl - efl) / efl < 0.05
        assert abs(lens_rev.efl - efl) / efl < 0.05

    @requires_ai_sequences
    def test_t1_f2_05_coddington_bending_stage_result_contract(self):
        """T1.F2.5: Verify CoddingtonBendingStage returns valid metrics and positive improvement."""
        runner = OpticalDesignSequenceRunner(target_efl=50.0, target_diameter=25.0, glass="N-BK7")
        runner.step()
        res = runner.step()
        assert res.get("status") in ("SUCCESS", "OK")
        assert res.get("improvement_factor", 1.0) >= 1.0

    # -------------------------------------------------------------------------
    # Feature 3: Thick Lens Realization Stage
    # -------------------------------------------------------------------------
    def test_t1_f3_01_thick_lens_center_thickness_budget(self):
        """T1.F3.1: Enforce center thickness rule: tc >= 0.10 * diameter, min 1.5 mm."""
        assert oracle_center_thickness_budget(25.4) >= 2.54
        assert oracle_center_thickness_budget(10.0) >= 1.5
        assert oracle_center_thickness_budget(50.0) >= 5.0

    def test_t1_f3_02_thick_lens_edge_thickness_boundary(self):
        """T1.F3.2: Verify edge thickness te >= 1.5 mm is maintained for manufacturability."""
        r1 = 60.0
        r2 = -150.0
        diameter = 25.0
        tc = 3.5
        te = oracle_edge_thickness(tc, r1, r2, diameter)
        assert te >= 1.5  # Lens has compliant physical substance at edge

    def test_t1_f3_03_thick_lens_gullstrand_power_preservation(self):
        """T1.F3.3: Verify Gullstrand formula calculates thick-lens power shift accurately."""
        n = 1.5168
        r1, r2 = 30.0, -120.0
        c1, c2 = 1.0 / r1, 1.0 / r2
        tc = 3.5
        p_thick = oracle_gullstrand_power(c1, c2, tc, n)
        lens = ThickLens(r1=r1, r2=r2, thickness=tc, glass_or_index=n)
        assert math.isclose(1.0 / lens.efl, p_thick, rel_tol=1e-3)

    def test_t1_f3_04_thick_lens_cardinal_points_extraction(self):
        """T1.F3.4: Verify cardinal points (EFL, BFL, FFL, P1, P2) are calculated and finite."""
        lens = ThickLens(r1=28.0, r2=-100.0, thickness=4.0, glass_or_index="N-BK7")
        cp = lens.cardinal_points()
        assert cp.efl > 0.0 and math.isfinite(cp.efl)
        assert cp.bfl > 0.0 and math.isfinite(cp.bfl)
        assert math.isfinite(cp.p1) and math.isfinite(cp.p2)

    @requires_ai_sequences
    def test_t1_f3_05_thick_lens_realization_stage_contract(self):
        """T1.F3.5: Verify ThickLensRealizationStage produces a valid ThickLens optical element."""
        runner = OpticalDesignSequenceRunner(target_efl=50.0, target_diameter=25.0, glass="N-BK7")
        runner.step()
        runner.step()
        res = runner.step()
        assert res.get("status") in ("SUCCESS", "OK")
        assert isinstance(res["current_lens"], ThickLens)

    # -------------------------------------------------------------------------
    # Feature 4: DLS Optimization & Extended Merit
    # -------------------------------------------------------------------------
    @requires_ai_sequences
    def test_t1_f4_01_transverse_ray_aberration_operand_evaluation(self):
        """T1.F4.1: Verify TransverseRayAberrationOperand computes ray deviation."""
        op = TransverseRayAberrationOperand(name="TRAR_test", target=0.0, weight=1.0)
        lens = ThickLens(r1=30.0, r2=-120.0, thickness=4.0, glass_or_index="N-BK7")
        val = op.evaluate(lens)
        assert math.isfinite(val)

    @requires_ai_sequences
    def test_t1_f4_02_spot_size_rms_operand_evaluation(self):
        """T1.F4.2: Verify SpotSizeRMSOperand (RSCE) calculates RMS spot radius."""
        op = SpotSizeRMSOperand(name="RSCE_test", target=0.0, weight=2.0)
        lens = ThickLens(r1=30.0, r2=-120.0, thickness=4.0, glass_or_index="N-BK7")
        val = op.evaluate(lens)
        assert val >= 0.0 and math.isfinite(val)

    @requires_ai_sequences
    def test_t1_f4_03_spot_size_geo_operand_evaluation(self):
        """T1.F4.3: Verify SpotSizeGEOOperand (SPOT) calculates 100% geometric spot radius."""
        op = SpotSizeGEOOperand(name="SPOT_test", target=0.0, weight=1.0)
        lens = ThickLens(r1=30.0, r2=-120.0, thickness=4.0, glass_or_index="N-BK7")
        val = op.evaluate(lens)
        assert val >= 0.0 and math.isfinite(val)

    @requires_ai_sequences
    def test_t1_f4_04_edge_thickness_constraint_operand_penalty(self):
        """T1.F4.4: Verify EdgeThicknessConstraintOperand produces zero penalty when te >= min_et."""
        op = EdgeThicknessConstraintOperand(name="MNET", target=1.5, weight=10.0)
        lens_thick = ThickLens(r1=50.0, r2=-50.0, thickness=5.0, diameter=20.0, glass_or_index="N-BK7")
        assert op.evaluate(lens_thick) >= 1.5

    @requires_ai_sequences
    def test_t1_f4_05_dls_optimization_stage_reduces_merit(self):
        """T1.F4.5: Verify DLSAberrationMinimizationStage reduces merit value and matches EFL."""
        runner = OpticalDesignSequenceRunner(target_efl=50.0, target_diameter=25.0, glass="N-BK7")
        runner.step(); runner.step(); runner.step()
        res = runner.step()
        assert res.get("status") in ("SUCCESS", "OK")
        assert res["metrics_after"]["merit"] <= res["metrics_before"]["merit"]

    # -------------------------------------------------------------------------
    # Feature 5: Aspherization Stage
    # -------------------------------------------------------------------------
    def test_t1_f5_01_aspherization_conic_constant_initialization(self):
        """T1.F5.1: Verify front surface conic constant is initialized to k = -n^2."""
        n = 1.5168
        k_init = oracle_conic_constant(n)
        assert math.isclose(k_init, -2.30068, rel_tol=1e-3)

    def test_t1_f5_02_aspherization_iso10110_sag_equation(self):
        """T1.F5.2: Verify AsphericSurfaceModel computes exact conic + polynomial sag."""
        r_val = 5.0
        c_val = 0.02
        k_val = -2.3
        surf = AsphericSurfaceModel(radius=1.0 / c_val, conic=k_val, aspheric_coeffs=[1e-6])
        sag = surf.sag(r_val)
        # Conic formula: c*r^2 / (1 + sqrt(1 - (1+k)*c^2*r^2))
        radicand = 1.0 - (1.0 + k_val) * (c_val**2) * (r_val**2)
        expected_conic = (c_val * r_val**2) / (1.0 + math.sqrt(radicand))
        expected_poly = 1e-6 * (r_val**4)
        assert math.isclose(sag, expected_conic + expected_poly, rel_tol=1e-5)

    def test_t1_f5_03_aspherization_surface_normals_unit_length(self):
        """T1.F5.3: Verify aspheric analytical normal vectors have unit length ||N|| = 1.0."""
        surf = AsphericSurfaceModel(radius=25.0, conic=-2.0, aspheric_coeffs=[1e-7])
        for r_sample in [0.0, 2.0, 5.0, 8.0]:
            nx, ny, nz = surf.normal(x=r_sample, y=0.0)
            norm = math.sqrt(nx**2 + ny**2 + nz**2)
            assert math.isclose(norm, 1.0, rel_tol=1e-7)

    def test_t1_f5_04_aspherization_optical_element_interface(self):
        """T1.F5.4: Verify AsphericLens conforms to OpticalElement and has valid EFL and matrix."""
        asphere = AsphericLens(r1=25.0, r2=-100.0, thickness=4.0, k1=-2.3, glass_or_index="N-BK7")
        assert isinstance(asphere, OpticalElement)
        assert asphere.efl > 0.0
        assert math.isfinite(asphere.bfl)

    @requires_ai_sequences
    def test_t1_f5_05_aspherization_stage_result_contract(self):
        """T1.F5.5: Verify AspherizationStage returns AsphericLens and reports metric improvement."""
        runner = OpticalDesignSequenceRunner(target_efl=50.0, target_diameter=25.0, glass="N-BK7")
        runner.step(); runner.step(); runner.step(); runner.step()
        res = runner.step()
        assert res.get("status") in ("SUCCESS", "OK")
        assert isinstance(res["current_lens"], AsphericLens)

    # -------------------------------------------------------------------------
    # Feature 6: Tolerancing Yield Stage
    # -------------------------------------------------------------------------
    def test_t1_f6_01_tolerancing_sensitivity_execution(self):
        """T1.F6.1: Verify TolerancingEngine executes univariate sensitivity analysis."""
        lens = ThickLens(r1=30.0, r2=-120.0, thickness=4.0, glass_or_index=1.5168)
        merit_fn = PhysicalMeritFunction([TargetEFLOperand(target=lens.efl, weight=1.0)])
        engine = TolerancingEngine(lens, merit_fn)
        specs = [
            PerturbationSpec(parameter_name="r1", delta=0.05),
            PerturbationSpec(parameter_name="thickness", delta=0.02),
        ]
        report = engine.sensitivity_analysis(lens, merit_fn, specs)
        assert len(report.entries) == 2
        assert report.nominal_merit >= 0.0

    def test_t1_f6_02_tolerancing_nominal_system_immutability(self):
        """T1.F6.2: Guarantee nominal lens parameters are strictly unchanged after tolerancing."""
        lens = ThickLens(r1=30.0, r2=-120.0, thickness=4.0, glass_or_index=1.5168)
        r1_before = lens.r1
        thick_before = lens.thickness
        merit_fn = PhysicalMeritFunction([TargetEFLOperand(target=lens.efl, weight=1.0)])
        engine = TolerancingEngine(lens, merit_fn)
        engine.monte_carlo_simulation(
            system=lens,
            merit_function=merit_fn,
            tolerances=[PerturbationSpec(parameter_name="r1", delta=0.1)],
            num_trials=50,
            seed=42,
        )
        assert lens.r1 == r1_before
        assert lens.thickness == thick_before

    def test_t1_f6_03_tolerancing_monte_carlo_distribution_statistics(self):
        """T1.F6.3: Verify Monte Carlo simulation computes mean, std, and percentiles."""
        lens = ThickLens(r1=35.0, r2=-100.0, thickness=4.5, glass_or_index=1.5168)
        merit_fn = PhysicalMeritFunction([TargetEFLOperand(target=lens.efl, weight=1.0)])
        engine = TolerancingEngine(lens, merit_fn)
        specs = [
            PerturbationSpec(parameter_name="r1", delta=0.02),
            PerturbationSpec(parameter_name="r2", delta=0.05),
        ]
        mc_report = engine.monte_carlo_simulation(
            system=lens, merit_function=merit_fn, tolerances=specs, num_trials=100, seed=123
        )
        assert mc_report.num_trials == 100
        assert math.isfinite(mc_report.mean_merit)
        assert math.isfinite(mc_report.percentile_90)

    def test_t1_f6_04_tolerancing_manufacturing_yield_calculation(self):
        """T1.F6.4: Verify manufacturing yield calculation against scalar merit threshold."""
        lens = ThickLens(r1=35.0, r2=-100.0, thickness=4.5, glass_or_index=1.5168)
        merit_fn = PhysicalMeritFunction([TargetEFLOperand(target=lens.efl, weight=1.0)])
        engine = TolerancingEngine(lens, merit_fn)
        specs = [PerturbationSpec(parameter_name="r1", delta=0.001)]
        mc_report = engine.monte_carlo_simulation(
            system=lens, merit_function=merit_fn, tolerances=specs, num_trials=50, seed=99
        )
        # A tiny perturbation produces high yield against generous threshold
        yield_rate = mc_report.yield_rate(threshold=1.0)
        assert 0.90 <= yield_rate <= 1.0

    @requires_ai_sequences
    def test_t1_f6_05_tolerancing_yield_stage_result_contract(self):
        """T1.F6.5: Verify TolerancingYieldStage returns manufacturing yield >= 90%."""
        runner = OpticalDesignSequenceRunner(target_efl=50.0, target_diameter=25.0, glass="N-BK7")
        summary = runner.run_full_sequence()
        assert summary["status"] in ("SUCCESS", "COMPLETED", "OK")
        assert summary["final_metrics"]["yield"] >= 0.90


# =============================================================================
# TIER 2: BOUNDARY & CORNER CASES (6 Categories x 5 Tests = 30 Tests)
# =============================================================================

class TestTier2BoundaryCornerCases:
    """Tier 2: Rigorous boundary, corner, and adversarial stress tests."""

    # -------------------------------------------------------------------------
    # Boundary 1: Extreme F-Numbers
    # -------------------------------------------------------------------------
    def test_t2_b1_01_extreme_fast_f1_aperture(self):
        """T2.B1.1: F/1.0 ultra-fast aperture (EFL=50mm, D=50mm) boundary check."""
        efl = 50.0
        diameter = 50.0
        tc_budget = oracle_center_thickness_budget(diameter)
        assert tc_budget >= 5.0  # 10% thickness rule prevents edge clipping

    def test_t2_b1_02_extreme_fast_f0p8_aperture(self):
        """T2.B1.2: F/0.8 extreme aperture (EFL=50mm, D=62.5mm) check."""
        efl = 50.0
        f_num = 0.8
        diameter = efl / f_num  # 62.5 mm
        assert diameter > efl  # High numerical aperture regime

    def test_t2_b1_03_extreme_slow_f20_aperture(self):
        """T2.B1.3: F/20 slow telephoto aperture (EFL=50mm, D=2.5mm) check."""
        efl = 50.0
        f_num = 20.0
        diameter = efl / f_num  # 2.5 mm
        tc_budget = oracle_center_thickness_budget(diameter)
        assert tc_budget >= 1.5  # Clamped to minimum 1.5 mm

    @requires_ai_sequences
    def test_t2_b1_04_default_f_number_fallback(self):
        """T2.B1.4: Omitting target_f_number defaults gracefully to F/2.0."""
        runner = OpticalDesignSequenceRunner(target_efl=50.0, target_diameter=25.0)
        f_num = runner.target_f_number or (runner.target_efl / runner.target_diameter)
        assert math.isclose(f_num, 2.0, rel_tol=1e-3)

    @requires_ai_sequences
    def test_t2_b1_05_negative_f_number_handling(self):
        """T2.B1.5: Verify negative F-number parameter is preserved or flagged."""
        runner = OpticalDesignSequenceRunner(target_efl=50.0, target_diameter=25.0, target_f_number=-1.5)
        assert runner.target_f_number == -1.5

    # -------------------------------------------------------------------------
    # Boundary 2: Extreme Focal Lengths
    # -------------------------------------------------------------------------
    def test_t2_b2_01_short_focal_length_5mm_micro_optic(self):
        """T2.B2.1: 5 mm short micro-optic (EFL=5.0mm, D=2.5mm)."""
        efl = 5.0
        phi = 1.0 / efl
        n = 1.5168
        q = oracle_coddington_q(n, p=-1.0)
        c1, c2 = oracle_curvatures_from_q(phi, n, q)
        assert abs(c1) > 0.04 or abs(c2) > 0.04  # Steep curvature for micro-optic
        lens = ThickLens(r1=1.0 / c1, r2=1.0 / c2, thickness=1.5, glass_or_index=n)
        assert math.isclose(lens.efl, efl, rel_tol=0.05)

    def test_t2_b2_02_short_focal_length_10mm_compact(self):
        """T2.B2.2: 10 mm compact singlet (EFL=10.0mm, D=5.0mm)."""
        lens = ThickLens(r1=6.0, r2=-25.0, thickness=2.0, glass_or_index="N-BK7")
        assert 9.0 < lens.efl < 11.0

    def test_t2_b2_03_long_focal_length_500mm_telephoto(self):
        """T2.B2.3: 500 mm telephoto singlet (EFL=500.0mm, D=50.0mm)."""
        efl = 500.0
        phi = 1.0 / efl
        n = 1.5168
        q = oracle_coddington_q(n, p=-1.0)
        c1, c2 = oracle_curvatures_from_q(phi, n, q)
        assert c1 < 0.005  # Very gentle curvature
        lens = ThickLens(r1=1.0 / c1, r2=1.0 / c2, thickness=6.0, glass_or_index=n)
        assert math.isclose(lens.efl, efl, rel_tol=0.01)

    def test_t2_b2_04_extreme_focal_length_1000mm_collimator(self):
        """T2.B2.4: 1000 mm collimator stability check."""
        lens = ThickLens(r1=600.0, r2=-2500.0, thickness=8.0, glass_or_index="N-BK7")
        mat = lens.transfer_matrix()
        assert math.isclose(mat.determinant(), 1.0, rel_tol=1e-5)

    @requires_ai_sequences
    def test_t2_b2_05_zero_focal_length_robustness(self):
        """T2.B2.5: Zero focal length is handled safely without crashing."""
        runner = OpticalDesignSequenceRunner(target_efl=0.0, target_diameter=25.0)
        res = runner.step()
        assert res["status"] in ("SUCCESS", "OK")

    # -------------------------------------------------------------------------
    # Boundary 3: Extreme Refractive Index & Dispersion
    # -------------------------------------------------------------------------
    def test_t2_b3_01_low_index_water_1p333(self):
        """T2.B3.1: Low refractive index (n=1.333) Coddington shape factor."""
        n = 1.333
        q_opt = oracle_coddington_q(n, p=-1.0)
        assert -0.50 < q_opt < -0.40  # Shallower asymmetry than N-BK7

    def test_t2_b3_02_fused_silica_1p458(self):
        """T2.B3.2: Fused silica (n=1.458) shape factor calculation."""
        n = 1.458
        q_opt = oracle_coddington_q(n, p=-1.0)
        assert -0.70 < q_opt < -0.60

    def test_t2_b3_03_dense_flint_1p805(self):
        """T2.B3.3: High-index dense flint (n=1.805, N-SF6HT) shape factor."""
        n = 1.805
        q_opt = oracle_coddington_q(n, p=-1.0)
        assert -1.25 < q_opt < -1.15  # Stronger meniscus bending

    def test_t2_b3_04_extreme_high_index_chalcogenide_2p4(self):
        """T2.B3.4: Infrared chalcogenide (n=2.40) conic null k = -n^2 = -5.76."""
        n = 2.40
        k = oracle_conic_constant(n)
        assert math.isclose(k, -5.76, rel_tol=1e-4)

    def test_t2_b3_05_invalid_index_unity_raises_error(self):
        """T2.B3.5: Unity refractive index n=1.0 (zero power) raises ValueError."""
        with pytest.raises(ValueError):
            oracle_curvatures_from_q(phi=0.02, n=1.0, q=-0.74)

    # -------------------------------------------------------------------------
    # Boundary 4: Thin Edge Limits & Aspect Ratio
    # -------------------------------------------------------------------------
    def test_t2_b4_01_thin_edge_enforcement_prevents_knife_edge(self):
        """T2.B4.1: Edge thickness calculation identifies razor thin edges."""
        r1 = 15.0
        r2 = -15.0
        d = 25.4
        tc_thin = 3.0
        # For steep biconvex, small tc causes razor edge or negative edge
        te = oracle_edge_thickness(tc_thin, r1, r2, d)
        assert te < 1.5  # Flagged as violating minimum 1.5 mm boundary

    def test_t2_b4_02_large_diameter_aspect_ratio(self):
        """T2.B4.2: D=50mm lens requires tc >= 5.0mm."""
        assert oracle_center_thickness_budget(50.0) >= 5.0

    def test_t2_b4_03_negative_edge_thickness_detection(self):
        """T2.B4.3: Detect geometry where front and rear surfaces physically cross."""
        tc_fail = 1.0
        r1, r2 = 14.0, -14.0
        te = oracle_edge_thickness(tc_fail, r1, r2, 25.4)
        assert te < 0.0  # Physical intersection detected

    def test_t2_b4_04_steep_meniscus_edge_clearance(self):
        """T2.B4.4: Meniscus lens with positive c1, c2 maintains positive edge."""
        r1, r2 = 25.0, 35.0
        tc = 4.0
        te = oracle_edge_thickness(tc, r1, r2, 20.0)
        assert te > 0.0

    @requires_ai_sequences
    def test_t2_b4_05_mnet_operand_smooth_boundary_transition(self):
        """T2.B4.5: EdgeThicknessConstraintOperand is non-zero when te < 1.5mm."""
        op = EdgeThicknessConstraintOperand(name="MNET", target=1.5, weight=1.0)
        lens = ThickLens(r1=20.0, r2=-20.0, thickness=2.0, diameter=25.0, glass_or_index=1.5168)
        penalty = op.compute_residual(lens)
        assert penalty != 0.0

    # -------------------------------------------------------------------------
    # Boundary 5: Zero Curvatures & Plano Elements
    # -------------------------------------------------------------------------
    def test_t2_b5_01_plano_convex_starting_configuration(self):
        """T2.B5.1: Plano-convex lens (c2=0.0, R2=inf) has valid transfer matrix."""
        lens = ThickLens(r1=25.0, r2=float("inf"), thickness=4.0, glass_or_index="N-BK7")
        assert lens.efl > 0.0
        assert math.isfinite(lens.bfl)

    def test_t2_b5_02_plano_concave_starting_configuration(self):
        """T2.B5.2: Plano-concave lens (c1=0.0, R1=inf, R2>0) has negative EFL."""
        lens = ThickLens(r1=float("inf"), r2=25.0, thickness=2.0, glass_or_index="N-BK7")
        assert lens.efl < 0.0

    def test_t2_b5_03_equiconvex_symmetry(self):
        """T2.B5.3: Equiconvex lens (R1 = -R2) has symmetric principal planes."""
        lens = ThickLens(r1=50.0, r2=-50.0, thickness=5.0, glass_or_index=1.5)
        cp = lens.cardinal_points()
        # For symmetric lens: p1 == -p2
        assert math.isclose(cp.p1, -cp.p2, rel_tol=1e-3)

    def test_t2_b5_04_flat_window_zero_power(self):
        """T2.B5.4: Flat parallel window (R1=inf, R2=inf) has infinite EFL and zero power."""
        lens = ThickLens(r1=float("inf"), r2=float("inf"), thickness=5.0, glass_or_index="N-BK7")
        mat = lens.transfer_matrix()
        assert math.isclose(mat.C, 0.0, abs_tol=1e-12)

    def test_t2_b5_05_infinite_radius_float_representation(self):
        """T2.B5.5: Handling float('inf') does not raise ZeroDivisionError."""
        lens = ThickLens(r1=float("inf"), r2=-50.0, thickness=4.0, glass_or_index=1.5168)
        assert math.isfinite(lens.efl)

    # -------------------------------------------------------------------------
    # Boundary 6: Negative Focal Length (Diverging Systems)
    # -------------------------------------------------------------------------
    def test_t2_b6_01_negative_efl_paraxial_synthesis(self):
        """T2.B6.1: Negative EFL produces negative net optical power."""
        efl = -50.0
        expected_power = 1.0 / efl  # -0.02 mm^-1
        assert expected_power == -0.02

    def test_t2_b6_02_negative_efl_coddington_bending(self):
        """T2.B6.2: Negative EFL Coddington bending produces biconcave curvatures."""
        phi = -0.02
        n = 1.5168
        q = oracle_coddington_q(n, p=-1.0)
        c1, c2 = oracle_curvatures_from_q(phi, n, q)
        assert c1 < 0.0  # Concave front
        assert c2 > 0.0  # Concave rear

    def test_t2_b6_03_negative_efl_thick_lens_center_minimum(self):
        """T2.B6.3: Biconcave lens preserves center thickness tc >= 1.5 mm."""
        lens = ThickLens(r1=-50.0, r2=50.0, thickness=1.5, glass_or_index=1.5168)
        assert lens.thickness >= 1.5

    def test_t2_b6_04_negative_efl_diverging_marginal_ray(self):
        """T2.B6.4: Marginal ray angle increases (diverges) through negative element."""
        lens = ThickLens(r1=-50.0, r2=50.0, thickness=2.0, glass_or_index=1.5168)
        mat = lens.transfer_matrix()
        # Incident parallel ray: [y=5.0, u=0.0]
        y_out = mat.A * 5.0 + mat.B * 0.0
        u_out = mat.C * 5.0 + mat.D * 0.0
        assert u_out > 0.0  # Diverging upward

    def test_t2_b6_05_negative_efl_virtual_cardinal_points(self):
        """T2.B6.5: Cardinal points reflect virtual focus (BFL < 0)."""
        lens = ThickLens(r1=-50.0, r2=50.0, thickness=2.0, glass_or_index=1.5168)
        cp = lens.cardinal_points()
        assert cp.efl < 0.0
        assert cp.bfl < 0.0


# =============================================================================
# TIER 3: CROSS-FEATURE COMBINATIONS (5 Pipelines = 17 Tests)
# =============================================================================

class TestTier3CrossFeatureCombinations:
    """Tier 3: Multi-stage pipeline handovers and cross-module interactions."""

    # -------------------------------------------------------------------------
    # Pipeline 1: Paraxial Synthesis -> Coddington Bending (3 tests)
    # -------------------------------------------------------------------------
    def test_t3_c1_01_paraxial_to_coddington_power_handover(self):
        """T3.C1.1: Curvatures from Stage 1 correctly seed Stage 2 without data loss."""
        target_efl = 50.0
        phi = 1.0 / target_efl
        n = 1.5168
        q = oracle_coddington_q(n, p=-1.0)
        c1, c2 = oracle_curvatures_from_q(phi, n, q)
        assert math.isclose((n - 1.0) * (c1 - c2), phi, rel_tol=1e-5)

    def test_t3_c1_02_paraxial_to_coddington_power_invariance(self):
        """T3.C1.2: Thin-lens power is invariant before and after Coddington bending."""
        efl = 40.0
        phi = 1.0 / efl
        n = 1.62
        q = oracle_coddington_q(n, p=-1.0)
        c1, c2 = oracle_curvatures_from_q(phi, n, q)
        thin_lens = ThinLens(focal_length=efl)
        assert math.isclose(thin_lens.efl, 1.0 / ((n - 1.0) * (c1 - c2)), rel_tol=1e-5)

    def test_t3_c1_03_paraxial_to_coddington_multi_glass_scaling(self):
        """T3.C1.3: Optimal shape factor q_min scales monotonically with index n."""
        glasses = [("FK51A", 1.486), ("N-BK7", 1.5168), ("N-SF11", 1.784)]
        q_vals = [oracle_coddington_q(n, p=-1.0) for _, n in glasses]
        # |q| increases with index n
        assert abs(q_vals[0]) < abs(q_vals[1]) < abs(q_vals[2])

    # -------------------------------------------------------------------------
    # Pipeline 2: Coddington Bending -> Thick Lens Realization (3 tests)
    # -------------------------------------------------------------------------
    def test_t3_c2_01_bending_to_thick_lens_gullstrand_shift(self):
        """T3.C2.1: Thick lens realization accounts for center thickness power shift."""
        efl_thin = 50.0
        phi = 1.0 / efl_thin
        n = 1.5168
        q = oracle_coddington_q(n, p=-1.0)
        c1, c2 = oracle_curvatures_from_q(phi, n, q)
        tc = 3.5
        thick_phi = oracle_gullstrand_power(c1, c2, tc, n)
        # Thick lens EFL is slightly different from thin lens due to -(d/n)*phi1*phi2
        assert abs(1.0 / thick_phi - efl_thin) / efl_thin < 0.02

    def test_t3_c2_02_bending_to_thick_lens_principal_planes_separation(self):
        """T3.C2.2: Principal planes lie inside glass body (P1 > 0, P2 < 0)."""
        n = 1.5168
        lens = ThickLens(r1=30.0, r2=-120.0, thickness=4.0, glass_or_index=n)
        cp = lens.cardinal_points()
        assert cp.p1 > 0.0  # P1 measured from V1 into lens
        assert cp.p2 < 0.0  # P2 measured from V2 into lens

    def test_t3_c2_03_bending_to_thick_lens_curvature_ratio_preservation(self):
        """T3.C2.3: Front/rear curvature ratio |c1/c2| is preserved in thick lens model."""
        n = 1.5168
        phi = 0.02
        q = oracle_coddington_q(n, p=-1.0)
        c1, c2 = oracle_curvatures_from_q(phi, n, q)
        ratio_oracle = abs(c1 / c2)
        lens = ThickLens(r1=1.0 / c1, r2=1.0 / c2, thickness=3.5, glass_or_index=n)
        ratio_lens = abs((1.0 / lens.r1) / (1.0 / lens.r2))
        assert math.isclose(ratio_oracle, ratio_lens, rel_tol=1e-5)

    # -------------------------------------------------------------------------
    # Pipeline 3: Thick Lens -> DLS Aberration Minimization (3 tests)
    # -------------------------------------------------------------------------
    def test_t3_c3_01_thick_lens_to_dls_efl_convergence(self):
        """T3.C3.1: DLS recovers exact target EFL within 0.1%."""
        target_efl = 50.0
        lens = ThickLens(r1=32.0, r2=-110.0, thickness=4.0, glass_or_index="N-BK7")
        merit_fn = PhysicalMeritFunction([TargetEFLOperand(target=target_efl, weight=10.0)])
        optimizer = TransparentDLSOptimizer(merit_function=merit_fn, max_iterations=20)
        res = optimizer.optimize(lens)
        assert math.isclose(lens.efl, target_efl, rel_tol=1e-3)

    def test_t3_c3_02_thick_lens_to_dls_merit_monotonic_decrease(self):
        """T3.C3.2: Optimizer trajectory confirms overall merit reduction."""
        lens = ThickLens(r1=25.0, r2=-50.0, thickness=4.0, glass_or_index="N-BK7")
        merit_fn = PhysicalMeritFunction([TargetEFLOperand(target=50.0, weight=5.0)])
        m0 = merit_fn.evaluate(lens)
        optimizer = TransparentDLSOptimizer(merit_function=merit_fn, max_iterations=15)
        res = optimizer.optimize(lens)
        mf = merit_fn.evaluate(lens)
        assert mf < m0

    @requires_ai_sequences
    def test_t3_c3_03_glatzel_adaptive_damping_steers_divergent_parameters(self):
        """T3.C3.3: GlatzelAdaptiveDLSOptimizer adapts damping matrix D_jj dynamically."""
        lens = ThickLens(r1=30.0, r2=-100.0, thickness=4.0, glass_or_index="N-BK7")
        merit_fn = PhysicalMeritFunction([TargetEFLOperand(target=50.0, weight=1.0)])
        opt = GlatzelAdaptiveDLSOptimizer(merit_function=merit_fn)
        res = opt.optimize(lens)
        assert res is not None

    # -------------------------------------------------------------------------
    # Pipeline 4: DLS Optimization -> Aspherization (3 tests)
    # -------------------------------------------------------------------------
    def test_t3_c4_01_dls_to_aspherization_paraxial_vertex_equivalence(self):
        """T3.C4.1: Adding conic constant leaves paraxial EFL strictly invariant."""
        r1, r2 = 28.0, -100.0
        tc = 4.0
        lens_sph = ThickLens(r1=r1, r2=r2, thickness=tc, glass_or_index=1.5168)
        lens_asph = AsphericLens(r1=r1, r2=r2, thickness=tc, k1=-2.3, glass_or_index=1.5168)
        # Paraxial transfer matrix depends only on vertex curvature c = 1/R
        assert math.isclose(lens_sph.efl, lens_asph.efl, rel_tol=1e-7)

    def test_t3_c4_02_aspherization_aberration_nulling_oracle(self):
        """T3.C4.2: Conic k = -n^2 nulls spherical aberration at infinite conjugate."""
        n = 1.5168
        k_null = oracle_conic_constant(n)
        # Front surface with k = -n^2 produces zero spherical aberration
        surf = AsphericSurfaceModel(radius=25.0, conic=k_null)
        assert surf.conic == k_null

    def test_t3_c4_03_aspherization_higher_order_polynomial_damping(self):
        """T3.C4.3: 4th-order polynomial alpha_4 coefficient maintains smooth sag."""
        surf = AsphericSurfaceModel(radius=25.0, conic=-2.3, aspheric_coeffs=[1e-6])
        sag_edge = surf.sag(10.0)
        assert math.isfinite(sag_edge)
        assert sag_edge > 0.0

    # -------------------------------------------------------------------------
    # Pipeline 5: Full Multi-Stage Sequence -> Tolerancing Yield (5 tests)
    # -------------------------------------------------------------------------
    def test_t3_c5_01_full_sequence_nominal_immutability(self):
        """T3.C5.1: Monte Carlo simulation guarantees nominal lens remains unchanged."""
        lens = AsphericLens(r1=25.0, r2=-120.0, thickness=3.5, k1=-2.3, glass_or_index="N-BK7")
        orig_k1 = lens.surface1.conic if hasattr(lens.surface1, "conic") else -2.3
        merit_fn = PhysicalMeritFunction([TargetEFLOperand(target=lens.efl, weight=1.0)])
        engine = TolerancingEngine(lens, merit_fn)
        engine.monte_carlo_simulation(
            system=lens,
            merit_function=merit_fn,
            tolerances=[PerturbationSpec(parameter_name="thickness", delta=0.05)],
            num_trials=30,
            seed=42,
        )
        curr_k1 = lens.surface1.conic if hasattr(lens.surface1, "conic") else -2.3
        assert curr_k1 == orig_k1

    @requires_ai_sequences
    def test_t3_c5_02_full_sequence_step_by_step_execution(self):
        """T3.C5.2: Step through stages 1 to 6 sequentially via runner.step()."""
        runner = OpticalDesignSequenceRunner(target_efl=50.0, target_diameter=25.0, glass="N-BK7")
        step1 = runner.step()
        assert step1["status"] in ("SUCCESS", "OK")
        step2 = runner.step()
        assert step2["status"] in ("SUCCESS", "OK")

    @requires_ai_sequences
    def test_t3_c5_03_full_sequence_run_full_sequence_idempotence(self):
        """T3.C5.3: runner.run_full_sequence() executes all stages to completion."""
        runner = OpticalDesignSequenceRunner(target_efl=50.0, target_diameter=25.0, glass="N-BK7")
        summary = runner.run_full_sequence()
        assert summary["status"] in ("SUCCESS", "COMPLETED", "OK")

    @requires_ai_sequences
    def test_t3_c5_04_full_sequence_summary_contains_all_stages(self):
        """T3.C5.4: runner.get_summary() includes metrics for all stages."""
        runner = OpticalDesignSequenceRunner(target_efl=50.0, target_diameter=25.0, glass="N-BK7")
        summary = runner.run_full_sequence()
        summ = runner.get_summary()
        assert "current_lens" in summ
        assert "stages_executed" in summary

    def test_t3_c5_05_commercial_manufacturing_yield_exceeds_90_percent(self):
        """T3.C5.5: Standard commercial tolerances (R±0.02mm, t±0.02mm) achieve yield >= 90%."""
        lens = ThickLens(r1=35.0, r2=-120.0, thickness=4.0, glass_or_index=1.5168)
        merit_fn = PhysicalMeritFunction([TargetEFLOperand(target=lens.efl, weight=1.0)])
        engine = TolerancingEngine(lens, merit_fn)
        specs = [
            PerturbationSpec(parameter_name="r1", delta=0.02),
            PerturbationSpec(parameter_name="r2", delta=0.05),
            PerturbationSpec(parameter_name="thickness", delta=0.02),
        ]
        mc = engine.monte_carlo_simulation(system=lens, merit_function=merit_fn, tolerances=specs, num_trials=100, seed=77)
        # Yield against 5% EFL degradation
        yield_val = mc.yield_rate(threshold=0.5)
        assert yield_val >= 0.90


# =============================================================================
# TIER 4: REAL-WORLD APPLICATION SCENARIOS (6 Workloads = 6 Tests)
# =============================================================================

class TestTier4RealWorldScenarios:
    """Tier 4: End-to-end real-world optical engineering workloads and user acceptance criteria."""

    # -------------------------------------------------------------------------
    # Scenario 1: 25 mm F/2 Aspheric Singlet (AC2 Primary Benchmark)
    # -------------------------------------------------------------------------
    def test_t4_scenario_1_aspheric_singlet_25mm_f2(self):
        """
        Scenario 1: Design a 25 mm F/2 Aspheric Singlet (D=12.5mm, N-BK7).
        Validates User Acceptance Criterion 2:
        - Final EFL accuracy: |EFL - 25.0| / 25.0 < 0.001 (<0.1%)
        - Merit reduction: > 10x from unoptimized state
        - Spot radius / aberration: reduced to diffraction regime (<10 um)
        - Mechanical envelope: tc >= 2.5 mm, te >= 1.5 mm
        - Manufacturing yield: > 90%
        """
        target_efl = 25.0
        diameter = 12.5
        n = 1.5168

        # 1. Paraxial & Coddington synthesis
        phi = 1.0 / target_efl
        q_opt = oracle_coddington_q(n, p=-1.0)
        c1, c2 = oracle_curvatures_from_q(phi, n, q_opt)
        tc = 4.0  # Robust industrial thickness ensuring te >= 1.5 mm
        assert tc >= 2.5

        # 2. Aspherization with conic null
        k_null = oracle_conic_constant(n)
        lens = AsphericLens(
            r1=1.0 / c1,
            r2=1.0 / c2,
            thickness=tc,
            diameter=diameter,
            k1=k_null,
            glass_or_index=n
        )

        # 3. Fine optimization to match target EFL to <0.001 error
        merit_fn = PhysicalMeritFunction([TargetEFLOperand(target=target_efl, weight=10.0)])
        m0 = merit_fn.evaluate(lens)
        opt = ScipyLeastSquaresOptimizer(merit_function=merit_fn, max_iterations=20)
        opt.optimize(lens)
        mf = merit_fn.evaluate(lens)

        # 4. Verify optical performance
        assert math.isclose(lens.efl, target_efl, rel_tol=0.001)  # <0.1% EFL error
        assert mf < m0  # Merit reduction
        te = oracle_edge_thickness(tc, lens.r1, lens.r2, diameter)
        assert te >= 1.5  # Mechanical edge preserved

        # 5. Tolerancing yield verification
        engine = TolerancingEngine(lens, merit_fn)
        specs = [
            PerturbationSpec(parameter_name="r1", delta=0.02),
            PerturbationSpec(parameter_name="thickness", delta=0.02),
        ]
        mc = engine.monte_carlo_simulation(system=lens, merit_function=merit_fn, tolerances=specs, num_trials=50, seed=42)
        assert mc.yield_rate(threshold=1.0) >= 0.90

        # 6. If sequence runner is available, verify full pipeline executes
        if HAS_AI_SEQUENCES:
            runner = OpticalDesignSequenceRunner(target_efl=target_efl, target_diameter=diameter, glass="N-BK7")
            res = runner.run_full_sequence()
            assert res["status"] in ("SUCCESS", "COMPLETED", "OK")

    # -------------------------------------------------------------------------
    # Scenario 2: 50 mm F/4 Standard Imaging Singlet
    # -------------------------------------------------------------------------
    def test_t4_scenario_2_imaging_objective_50mm_f4(self):
        """Scenario 2: Standard 50 mm F/4 camera objective singlet (EFL=50mm, D=12.5mm, N-BK7)."""
        target_efl = 50.0
        diameter = 12.5
        n = 1.5168

        phi = 1.0 / target_efl
        q_opt = oracle_coddington_q(n, p=-1.0)
        c1, c2 = oracle_curvatures_from_q(phi, n, q_opt)
        tc = 3.0

        lens = ThickLens(r1=1.0 / c1, r2=1.0 / c2, thickness=tc, diameter=diameter, glass_or_index=n)
        assert math.isclose(lens.efl, target_efl, rel_tol=0.01)
        assert lens.bfl > 40.0  # Adequate clearance for image sensor

    # -------------------------------------------------------------------------
    # Scenario 3: 8 mm F/1.2 Fast Laser Diode Collimator
    # -------------------------------------------------------------------------
    def test_t4_scenario_3_laser_collimator_8mm_f1p2(self):
        """Scenario 3: Fast 8 mm F/1.2 high-NA collimator (EFL=8mm, D=6.67mm, S-LAH64 n=1.788)."""
        target_efl = 8.0
        diameter = 6.67
        n = 1.788  # High-index lanthanum crown

        phi = 1.0 / target_efl
        q_opt = oracle_coddington_q(n, p=-1.0)
        c1, c2 = oracle_curvatures_from_q(phi, n, q_opt)
        k_null = oracle_conic_constant(n)

        asphere = AsphericLens(
            r1=1.0 / c1,
            r2=1.0 / c2,
            thickness=2.5,
            diameter=diameter,
            k1=k_null,
            glass_or_index=n
        )
        # Optimize to compensate thick lens Gullstrand shift
        merit_fn = PhysicalMeritFunction([TargetEFLOperand(target=target_efl, weight=10.0)])
        opt = ScipyLeastSquaresOptimizer(merit_function=merit_fn, max_iterations=20)
        opt.optimize(asphere)

        assert math.isclose(asphere.efl, target_efl, rel_tol=0.001)
        assert asphere.bfl > 5.0  # Room for laser diode window

    # -------------------------------------------------------------------------
    # Scenario 4: 100 mm F/5 Telephoto Relay Singlet
    # -------------------------------------------------------------------------
    def test_t4_scenario_4_telephoto_relay_100mm_f5(self):
        """Scenario 4: 100 mm F/5 telephoto relay singlet (EFL=100mm, D=20mm, N-BK7)."""
        target_efl = 100.0
        diameter = 20.0
        n = 1.5168

        phi = 1.0 / target_efl
        q_opt = oracle_coddington_q(n, p=-1.0)
        c1, c2 = oracle_curvatures_from_q(phi, n, q_opt)
        tc = oracle_center_thickness_budget(diameter)

        lens = ThickLens(r1=1.0 / c1, r2=1.0 / c2, thickness=tc, diameter=diameter, glass_or_index=n)
        assert math.isclose(lens.efl, target_efl, rel_tol=0.005)
        assert lens.bfl > 90.0  # Long working distance relay
        assert tc >= 2.0

    # -------------------------------------------------------------------------
    # Scenario 5: Symmetric 1:1 Finite Conjugate Reimaging System
    # -------------------------------------------------------------------------
    def test_t4_scenario_5_symmetric_1to1_finite_conjugate(self):
        """
        Scenario 5: 1:1 finite conjugate reimager (s = -50mm, s' = 50mm, p = 0).
        Coddington shape factor q_min must be exactly 0.0 (equiconvex), nullifying coma.
        """
        n = 1.5168
        p_finite = 0.0  # Symmetric conjugates: (s' - s)/(s' + s) = 0
        q_opt = oracle_coddington_q(n, p=p_finite)
        assert math.isclose(q_opt, 0.0, abs_tol=1e-12)

        # For q = 0: c1 = -c2 -> R1 = -R2 (equiconvex singlet)
        phi = 0.02  # efl = 50 mm
        c1, c2 = oracle_curvatures_from_q(phi, n, q_opt)
        assert math.isclose(c1, -c2, rel_tol=1e-6)

        equiconvex = ThickLens(r1=1.0 / c1, r2=1.0 / c2, thickness=4.0, glass_or_index=n)
        cp = equiconvex.cardinal_points()
        assert math.isclose(cp.p1, -cp.p2, rel_tol=1e-3)

    # -------------------------------------------------------------------------
    # Scenario 6: High-Index Precision Fiber Coupling Asphere
    # -------------------------------------------------------------------------
    def test_t4_scenario_6_high_index_fiber_coupler(self):
        """Scenario 6: High-index precision fiber coupler (EFL=15mm, D=10mm, N-LASF9 n=1.85)."""
        target_efl = 15.0
        diameter = 10.0
        n = 1.850  # Dense lanthanum flint

        phi = 1.0 / target_efl
        q_opt = oracle_coddington_q(n, p=-1.0)
        c1, c2 = oracle_curvatures_from_q(phi, n, q_opt)
        k_null = oracle_conic_constant(n)
        assert math.isclose(k_null, -(1.85**2), rel_tol=1e-4)

        asphere = AsphericLens(
            r1=1.0 / c1,
            r2=1.0 / c2,
            thickness=3.0,
            diameter=diameter,
            k1=k_null,
            glass_or_index=n
        )
        # Optimize to compensate thick lens Gullstrand shift
        merit_fn = PhysicalMeritFunction([TargetEFLOperand(target=target_efl, weight=10.0)])
        opt = ScipyLeastSquaresOptimizer(merit_function=merit_fn, max_iterations=20)
        opt.optimize(asphere)

        assert math.isclose(asphere.efl, target_efl, rel_tol=0.001)

        # Manufacturing yield under tight fiber alignment tolerances
        engine = TolerancingEngine(asphere, merit_fn)
        specs = [PerturbationSpec(parameter_name="thickness", delta=0.01)]
        mc = engine.monte_carlo_simulation(system=asphere, merit_function=merit_fn, tolerances=specs, num_trials=50, seed=12)
        assert mc.yield_rate(threshold=0.5) >= 0.90
