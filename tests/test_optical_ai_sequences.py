"""
Unit and Integration Tests for Optical AI Sequences & Extended Merit Operands.
=============================================================================
Tests:
- Extended Merit Operands:
  - TransverseRayAberrationOperand (TRAR)
  - SpotSizeRMSOperand (RSCE)
  - SpotSizeGEOOperand (SPOT)
  - EdgeThicknessConstraintOperand (MNET / ETMN)
  - ExtendedMeritFunction
- GlatzelAdaptiveDLSOptimizer:
  - Parameter-specific damping matrix D_jj = p^2 * (J^T W^2 J)_jj * gamma_j
  - Dynamic steering factor adaptation (gamma_j update based on prediction discrepancy)
  - Single-step debugging (.step()) and full convergence (.optimize())
- Modular Design Stages:
  - ParaxialSynthesisStage
  - CoddingtonBendingStage
  - ThickLensRealizationStage
  - DLSAberrationMinimizationStage
  - AspherizationStage
  - TolerancingYieldStage
- OpticalDesignSequenceRunner:
  - Manual step-by-step debugging workflow (.step())
  - Full automated sequence execution (.run_full_sequence())
  - Focal length accuracy (< 0.1% error)
  - Merit function reduction (> 10x improvement)
  - Near-diffraction-limited spot size (< 10 um)
  - Monte Carlo manufacturing yield (>= 90%)
  - Reset and custom stage extensibility
"""

import math
from typing import Any, Dict, List, Optional
import numpy as np
import pytest

from optics_app.engine.thick_lens import ThickLens
from optics_app.engine.aspheric import AsphericLens
from optics_app.engine.extended_merit import (
    TransverseRayAberrationOperand,
    SpotSizeRMSOperand,
    SpotSizeGEOOperand,
    EdgeThicknessConstraintOperand,
    ExtendedMeritFunction,
    GlatzelAdaptiveDLSOptimizer,
    build_evaluation_optical_system,
    trace_pupil_rays,
)
from optics_app.engine.ai_sequences import (
    OpticalDesignSequenceRunner,
    DesignStage,
    ParaxialSynthesisStage,
    CoddingtonBendingStage,
    ThickLensRealizationStage,
    DLSAberrationMinimizationStage,
    AspherizationStage,
    TolerancingYieldStage,
    evaluate_lens_metrics,
)
from optics_app.engine.lens_merit import TargetEFLOperand, ThicknessConstraintOperand


# =============================================================================
# 1. Extended Merit Operands Unit Tests
# =============================================================================

class TestExtendedMeritOperands:
    """Test suite for TRAR, RSCE, SPOT, MNET operands and ExtendedMeritFunction."""

    @pytest.fixture
    def sample_thick_lens(self):
        """Standard biconvex N-BK7 lens: R1=50mm, R2=-50mm, t=5mm, diam=25.4mm."""
        return ThickLens(r1=50.0, r2=-50.0, thickness=5.0, glass_or_index="N-BK7", diameter=25.4)

    def test_transverse_ray_aberration_operand_evaluation(self, sample_thick_lens):
        """TRAR evaluates ray intercept error (y - y_chief) at focal plane."""
        trar_marginal = TransverseRayAberrationOperand(ray_height_fraction=1.0, weight=1.0)
        trar_zonal = TransverseRayAberrationOperand(ray_height_fraction=0.707, weight=1.0)

        err_marginal = trar_marginal.evaluate(sample_thick_lens)
        err_zonal = trar_zonal.evaluate(sample_thick_lens)

        # Uncorrected spherical singlet exhibits undercorrected spherical aberration (negative intercept error)
        assert isinstance(err_marginal, float)
        assert abs(err_marginal) > 0.1  # Significant aberration on 50mm uncorrected lens
        assert abs(err_zonal) < abs(err_marginal)  # Zonal ray has less aberration than marginal

        # Test residual computation
        residual = trar_marginal.compute_residual(sample_thick_lens)
        assert np.isclose(residual, err_marginal - trar_marginal.target)

    def test_spot_size_rms_operand_evaluation(self, sample_thick_lens):
        """RSCE evaluates RMS spot radius sqrt((1/N) * sum((y_i - y_bar)^2))."""
        rsce_op = SpotSizeRMSOperand(num_rays=21, target=0.0, weight=1.0)
        rms_val = rsce_op.evaluate(sample_thick_lens)

        assert isinstance(rms_val, float)
        assert rms_val > 0.05  # Standard biconvex lens has RMS blur radius > 50 um
        assert rms_val < 5.0  # Physically bounded

        # Residual matches actual when target is 0.0
        assert np.isclose(rsce_op.compute_residual(sample_thick_lens), rms_val)
        assert np.isclose(rsce_op.compute_weighted_residual(sample_thick_lens), rsce_op.weight * rms_val)

    def test_spot_size_geo_operand_and_inequality(self, sample_thick_lens):
        """SPOT evaluates 100% geometric spot radius, strictly satisfying SPOT >= RSCE."""
        spot_op = SpotSizeGEOOperand(num_rays=21, target=0.0, weight=1.0)
        rsce_op = SpotSizeRMSOperand(num_rays=21, target=0.0, weight=1.0)

        geo_val = spot_op.evaluate(sample_thick_lens)
        rms_val = rsce_op.evaluate(sample_thick_lens)

        assert isinstance(geo_val, float)
        assert geo_val > 0.0
        # Fundamental statistical theorem: maximum deviation >= RMS deviation
        assert geo_val >= rms_val

    def test_edge_thickness_constraint_operand_penalty_barrier(self):
        """MNET enforces te >= 1.5 mm with smooth one-sided quadratic penalty."""
        mnet_op = EdgeThicknessConstraintOperand(min_edge_thickness=1.5, weight=10.0)

        # 1. Lens with safe edge thickness: t=5.0mm, gentle curves R1=100, R2=-100 -> te ~ 4.2mm > 1.5mm
        safe_lens = ThickLens(r1=100.0, r2=-100.0, thickness=5.0, glass_or_index="N-BK7", diameter=25.4)
        te_safe = mnet_op.evaluate(safe_lens)
        assert te_safe > 1.5
        assert mnet_op.compute_residual(safe_lens) == 0.0  # Zero penalty when satisfied

        # 2. Lens violating edge thickness: very thin t=1.0mm, steep curves R1=30, R2=-30
        thin_lens = ThickLens(r1=30.0, r2=-30.0, thickness=1.0, glass_or_index="N-BK7", diameter=25.4)
        te_thin = mnet_op.evaluate(thin_lens)
        assert te_thin < 1.5
        residual_thin = mnet_op.compute_residual(thin_lens)
        assert residual_thin > 0.0
        assert np.isclose(residual_thin, 1.5 - te_thin)
        assert np.isclose(mnet_op.compute_weighted_residual(thin_lens), 10.0 * (1.5 - te_thin))

    def test_edge_thickness_evaluation_on_aspheric_lens(self):
        """MNET evaluates exact edge thickness using aspheric surface sag models."""
        asph = AsphericLens(
            r1=40.0,
            r2=-40.0,
            thickness=6.0,
            glass_or_index="N-BK7",
            k1=-2.0,
            diameter=25.4,
        )
        mnet = EdgeThicknessConstraintOperand(min_edge_thickness=1.5)
        te = mnet.evaluate(asph)
        assert isinstance(te, float)
        assert te > 0.0
        assert te < asph.thickness  # Edge thickness is thinner than center thickness for biconvex asphere

    def test_extended_merit_function_composite_evaluation(self, sample_thick_lens):
        """ExtendedMeritFunction aggregates EFL, spot, and edge thickness operands."""
        mf = ExtendedMeritFunction(target_efl=50.0, min_edge_thickness=1.5, num_rays=15)
        assert len(mf.operands) == 3

        merit_val = mf.evaluate(sample_thick_lens)
        assert isinstance(merit_val, float)
        assert merit_val > 0.0

        residuals = mf.residuals(sample_thick_lens)
        assert len(residuals) == 3
        weighted_res = mf.weighted_residuals(sample_thick_lens)
        assert len(weighted_res) == 3

        report = mf.operand_report(sample_thick_lens)
        assert len(report) == 3
        assert any(entry["name"] == "TargetEFL" for entry in report)
        assert any(entry["name"] == "RSCE" for entry in report)
        assert any(entry["name"] == "MNET" for entry in report)


# =============================================================================
# 2. Glatzel Adaptive DLS Optimizer Unit Tests
# =============================================================================

class TestGlatzelAdaptiveDLSOptimizer:
    """Test suite for GlatzelAdaptiveDLSOptimizer parameter-specific damping steering."""

    def test_glatzel_parameter_specific_damping_steering(self):
        """Verify Glatzel optimizer dynamically adjusts gamma steering factors."""
        lens = ThickLens(r1=50.0, r2=-50.0, thickness=5.0, glass_or_index="N-BK7", diameter=25.4)
        mf = ExtendedMeritFunction(target_efl=50.0, min_edge_thickness=1.5)

        opt = GlatzelAdaptiveDLSOptimizer(
            merit_function=mf,
            max_iterations=10,
            initial_damping=0.01,
            variable_names=["c1", "c2", "thickness"],
        )

        # Initial state before stepping
        assert opt.gamma is None

        # Execute single step
        step_rec = opt.step(lens)
        assert opt.gamma is not None
        assert len(opt.gamma) == 3
        # Damping factors should adaptively steer
        assert all(g > 0.0 for g in opt.gamma)
        assert step_rec.iteration == 1
        assert "gamma_mean" in step_rec.notes

    def test_glatzel_single_step_debugging_record_transparency(self):
        """Verify .step() returns comprehensive OptimizationStepRecord for manual debugging."""
        lens = ThickLens(r1=40.0, r2=-60.0, thickness=5.0, glass_or_index="N-BK7", diameter=25.4)
        mf = ExtendedMeritFunction(target_efl=40.0, min_edge_thickness=1.5)

        opt = GlatzelAdaptiveDLSOptimizer(merit_function=mf, max_iterations=5)
        rec = opt.step(lens)

        assert isinstance(rec.merit_value, float)
        assert isinstance(rec.step_norm, float)
        assert rec.jacobian is not None
        assert rec.jacobian.shape == (len(mf.operands), 3)
        assert "c1" in rec.parameters
        assert "c2" in rec.parameters
        assert "thickness" in rec.parameters

    def test_glatzel_convergence_on_thick_lens(self):
        """Verify full Glatzel DLS optimization converges and reduces merit significantly."""
        lens = ThickLens(r1=35.0, r2=-35.0, thickness=4.0, glass_or_index="N-BK7", diameter=25.4)
        target_efl = 50.0

        mf = ExtendedMeritFunction(target_efl=target_efl, min_edge_thickness=1.5)
        initial_merit = mf.evaluate(lens)

        opt = GlatzelAdaptiveDLSOptimizer(merit_function=mf, max_iterations=25, tolerance=1e-5)
        result = opt.optimize(lens)

        assert result.success
        assert result.final_merit < initial_merit
        # Paraxial EFL matches target within 1%
        assert np.isclose(lens.analytical_efl(), target_efl, rtol=0.01)
        assert len(result.history) > 1


# =============================================================================
# 3. Modular Design Stages Unit Tests
# =============================================================================

class TestModularDesignStages:
    """Test suite verifying each of the 6 design stages independently."""

    def test_paraxial_synthesis_stage_execution(self):
        """ParaxialSynthesisStage solves optical power and initializes symmetrical singlet."""
        runner = OpticalDesignSequenceRunner(target_efl=50.0, target_diameter=25.4, glass="N-BK7")
        stage = ParaxialSynthesisStage()

        result = stage.execute(runner)
        assert result["stage_name"] == "ParaxialSynthesisStage"
        assert result["status"] == "SUCCESS"
        assert runner.current_lens is not None

        lens = runner.current_lens
        assert isinstance(lens, ThickLens)
        assert np.isclose(lens.r1, -lens.r2)  # Symmetrical biconvex starting form
        assert np.isclose(lens.analytical_efl(), 50.0, rtol=0.05)
        assert result["details"]["optical_power_mm"] == pytest.approx(1.0 / 50.0)

    def test_coddington_bending_stage_minimizes_spherical_aberration(self):
        """CoddingtonBendingStage derives best-form shape factor and reduces spot size."""
        runner = OpticalDesignSequenceRunner(target_efl=50.0, target_diameter=25.4, glass="N-BK7")
        ParaxialSynthesisStage().execute(runner)

        sym_spot = evaluate_lens_metrics(runner.current_lens, 50.0)["spot_radius"]

        stage = CoddingtonBendingStage()
        result = stage.execute(runner)

        assert result["status"] == "SUCCESS"
        bent_lens = runner.current_lens

        # Best-form shape factor for N-BK7 facing infinity: q_min ~ 0.7397 (c1 > |c2|)
        assert abs(bent_lens.r1) < abs(bent_lens.r2)
        assert result["details"]["optimal_shape_factor_q"] == pytest.approx(0.7397, abs=0.01)

        bent_spot = result["metrics_after"]["spot_radius"]
        # Coddington bent lens has strictly less spherical aberration than symmetrical lens
        assert bent_spot < sym_spot
        assert result["improvement_factor"] > 1.0

    def test_thick_lens_realization_stage_boundary_enforcement(self):
        """ThickLensRealizationStage enforces tc >= 0.10*D and te >= 1.5 mm while preserving EFL."""
        runner = OpticalDesignSequenceRunner(target_efl=50.0, target_diameter=25.4, glass="N-BK7")
        ParaxialSynthesisStage().execute(runner)
        CoddingtonBendingStage().execute(runner)

        stage = ThickLensRealizationStage()
        result = stage.execute(runner)

        assert result["status"] == "SUCCESS"
        lens = runner.current_lens

        # Check center thickness constraint tc >= 0.10 * diameter
        assert lens.thickness >= 0.10 * runner.target_diameter
        assert result["details"]["center_thickness"] >= 2.54

        # Check edge thickness constraint te >= 1.5 mm
        assert result["details"]["edge_thickness"] >= 1.5

        # EFL preserved via Gullstrand power compensation
        assert np.isclose(lens.analytical_efl(), 50.0, rtol=0.01)

    def test_dls_aberration_minimization_stage_execution(self):
        """DLSAberrationMinimizationStage minimizes spot size and merit function."""
        runner = OpticalDesignSequenceRunner(target_efl=50.0, target_diameter=25.4, glass="N-BK7")
        ParaxialSynthesisStage().execute(runner)
        CoddingtonBendingStage().execute(runner)
        ThickLensRealizationStage().execute(runner)

        stage = DLSAberrationMinimizationStage()
        result = stage.execute(runner)

        assert result["status"] == "SUCCESS"
        assert result["metrics_after"]["merit"] <= result["metrics_before"]["merit"]
        assert len(result["details"]["gamma_steering_factors"]) == 3

    def test_aspherization_stage_achieves_diffraction_limited_spot(self):
        """AspherizationStage converts to AsphericLens, applies k=-n^2, and nulls residual spherical aberration."""
        runner = OpticalDesignSequenceRunner(target_efl=50.0, target_diameter=25.4, target_f_number=2.0, glass="N-BK7")
        ParaxialSynthesisStage().execute(runner)
        CoddingtonBendingStage().execute(runner)
        ThickLensRealizationStage().execute(runner)
        DLSAberrationMinimizationStage().execute(runner)

        stage = AspherizationStage()
        result = stage.execute(runner)

        assert result["status"] == "SUCCESS"
        asph_lens = runner.current_lens
        assert isinstance(asph_lens, AsphericLens)

        # Theoretical initial conic constant k = -n^2 ~ -2.30
        assert np.isclose(result["details"]["conic_theoretical"], -1.5168 ** 2, rtol=0.01)

        # Final spot size is reduced to under 10 um (< 0.010 mm)
        spot_um = result["details"]["final_spot_radius_um"]
        assert spot_um < 10.0
        assert np.isclose(asph_lens.cardinal_points().efl, 50.0, rtol=0.001)

    def test_tolerancing_yield_stage_verifies_high_manufacturing_yield(self):
        """TolerancingYieldStage verifies Monte Carlo yield >= 90% under commercial tolerances."""
        runner = OpticalDesignSequenceRunner(target_efl=50.0, target_diameter=25.4, target_f_number=2.0, glass="N-BK7")
        runner.run_full_sequence()  # Runs through aspherization first

        # Inspect yield stage result
        yield_result = runner.history[-1]
        assert yield_result["stage_name"] == "TolerancingYieldStage"
        assert yield_result["status"] == "SUCCESS"
        yield_rate = yield_result["details"]["yield_rate"]
        assert yield_rate >= 0.90  # Exceeds 90% manufacturing yield requirement


# =============================================================================
# 4. OpticalDesignSequenceRunner & Manual Debugging Tests
# =============================================================================

class TestOpticalDesignSequenceRunner:
    """Test suite for full runner execution, manual .step() debugging, and extensibility."""

    def test_step_by_step_manual_debugging_hook(self):
        """Verify .step() executes stages sequentially, maintaining full state transparency."""
        runner = OpticalDesignSequenceRunner(target_efl=25.0, target_diameter=12.5, target_f_number=2.0)
        assert runner.current_stage_index == 0
        assert not runner.is_finished

        stage_names = [
            "ParaxialSynthesisStage",
            "CoddingtonBendingStage",
            "ThickLensRealizationStage",
            "DLSAberrationMinimizationStage",
            "AspherizationStage",
            "TolerancingYieldStage",
        ]

        for expected_name in stage_names:
            assert runner.current_stage_name == expected_name
            res = runner.step()
            assert res["stage_name"] == expected_name
            assert res["status"] == "SUCCESS"
            assert "metrics_before" in res
            assert "metrics_after" in res
            assert "improvement_factor" in res
            assert runner.current_lens is not None

        assert runner.is_finished
        assert len(runner.history) == 6

        # Stepping after completion returns completed status safely
        post_step = runner.step()
        assert post_step["stage_name"] == "COMPLETED"

    def test_run_full_sequence_metrics_and_improvements(self):
        """Run full sequence and assert focal accuracy, merit reduction, and spot size improvements."""
        runner = OpticalDesignSequenceRunner(target_efl=25.0, target_diameter=12.5, target_f_number=2.0, glass="N-BK7")
        summary = runner.run_full_sequence()

        assert summary["status"] == "SUCCESS"
        assert summary["stages_executed"] == 6

        # 1. Focal Length Accuracy: < 0.1% error
        final_efl = summary["final_metrics"]["efl"]
        efl_error_rel = abs(final_efl - 25.0) / 25.0
        assert efl_error_rel < 0.001

        # 2. Total Improvement Factor: > 10x
        assert summary["total_improvement_factor"] > 10.0

        # 3. Final RMS Spot Size: < 10 um
        final_spot_um = summary["final_metrics"]["spot_radius"] * 1000.0
        assert final_spot_um < 10.0

        # 4. Final Manufacturing Yield: >= 90%
        assert summary["final_metrics"]["yield"] >= 0.90

        # 5. Final Lens Type: AsphericLens
        assert isinstance(summary["final_lens"], AsphericLens)

    def test_runner_reset_and_reproducibility(self):
        """Verify runner can be reset and produces identical results upon re-execution."""
        runner = OpticalDesignSequenceRunner(target_efl=30.0, target_diameter=15.0)
        res1 = runner.run_full_sequence()

        runner.reset()
        assert runner.current_stage_index == 0
        assert runner.current_lens is None
        assert len(runner.history) == 0

        res2 = runner.run_full_sequence()
        assert np.isclose(res1["final_metrics"]["efl"], res2["final_metrics"]["efl"])
        assert np.isclose(res1["final_metrics"]["spot_radius"], res2["final_metrics"]["spot_radius"])

    def test_custom_stage_pipeline_extensibility(self):
        """Verify custom stage sequences can be injected to support future models and extensions."""
        class CustomScaleStage(DesignStage):
            @property
            def stage_name(self) -> str:
                return "CustomScaleStage"

            def execute(self, runner: "OpticalDesignSequenceRunner") -> Dict[str, Any]:
                runner.current_lens.diameter = 30.0
                return {
                    "stage_name": self.stage_name,
                    "status": "SUCCESS",
                    "metrics_before": {},
                    "metrics_after": {},
                    "improvement_factor": 1.0,
                    "current_lens": runner.current_lens,
                    "details": {"new_diameter": 30.0},
                }

        custom_stages = [ParaxialSynthesisStage(), CustomScaleStage()]
        runner = OpticalDesignSequenceRunner(target_efl=40.0, stages=custom_stages)

        runner.step()  # ParaxialSynthesisStage
        assert runner.current_lens.diameter == 25.4

        runner.step()  # CustomScaleStage
        assert runner.current_lens.diameter == 30.0
        assert runner.is_finished
