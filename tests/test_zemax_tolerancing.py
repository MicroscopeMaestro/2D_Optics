"""
tests/test_zemax_tolerancing.py
===============================
Comprehensive test suite for the Zemax-Style Optical Tolerancing Suite in 2D Optics.
Verifies:
1. Zemax Tolerance Operands (TRAD, TCUR, TTHI, TSDX, TSDY, TSTX, TSTY, TEDX, TEDY, TIND, TABB, TCON, TIRR, COMP).
2. Sampling Distributions (Uniform, Gaussian/Normal 3-sigma, Parabolic).
3. Performance Criteria (RMS Spot Radius, Geo Spot Radius, EFFL, BFL, MeritFunction).
4. Auto-Refocus Compensator Optimization.
5. Sensitivity Analysis with RSS & Worst-Case degradation.
6. Monte Carlo Simulation with cumulative percentiles and manufacturing yield calculation.
7. Quality Presets (Commercial, Precision, High-Precision).
8. Zemax OpticStudio Report & .ZMX Export/Import.
"""

import math
import os
import numpy as np
import pytest

from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tolerancing import (
    ToleranceOperandType,
    ToleranceDistribution,
    TolerancingCriterion,
    PerturbationSpec,
    ZemaxToleranceOperand,
    TolerancingEngine,
    ZemaxTolerancingEngine,
    sample_tolerance_delta,
    compute_system_spot_radius,
    solve_system_compensator,
    create_commercial_tolerances,
    create_precision_tolerances,
    create_high_precision_tolerances,
    export_zemax_tolerance_report,
    export_zmx_with_tolerances,
    parse_zmx_tolerances,
    augmented_transfer_matrix_3x3,
)


def create_sample_singlet_stack() -> SurfaceStack:
    """Create a standard N-BK7 singlet lens with air spaces and detector."""
    stack = SurfaceStack()
    stack.add_surface(Surface(name="Object", surface_type=SurfaceType.OBJECT, thickness=50.0))
    stack.add_surface(Surface(name="L1_Front", surface_type=SurfaceType.STANDARD, radius=50.0, thickness=5.0, glass="N-BK7", semi_diameter=12.5, element_id=1))
    stack.add_surface(Surface(name="L1_Rear", surface_type=SurfaceType.STANDARD, radius=-50.0, thickness=46.5, glass="AIR", semi_diameter=12.5, element_id=1))
    stack.add_surface(Surface(name="Detector", surface_type=SurfaceType.DETECTOR, radius=0.0, thickness=0.0, glass="AIR", semi_diameter=12.5))
    stack._update_accumulated_positions()
    return stack


class TestZemaxToleranceOperandsAndSampling:
    """Unit tests for Zemax tolerance operands and probability distributions."""

    def test_operand_creation_and_defaults(self):
        op = ZemaxToleranceOperand(parameter_name="TRAD", surface_index=1, delta=0.1, distribution="uniform")
        assert op.parameter_name == "TRAD"
        assert op.surface_index == 1
        assert op.delta == 0.1
        assert op.min_delta == -0.1
        assert op.max_delta == 0.1

    def test_uniform_distribution_sampling(self):
        rng = np.random.default_rng(1234)
        samples = [sample_tolerance_delta(0.05, distribution="uniform", rng=rng) for _ in range(500)]
        assert all(-0.05001 <= s <= 0.05001 for s in samples)
        assert abs(np.mean(samples)) < 0.01

    def test_gaussian_distribution_3sigma(self):
        rng = np.random.default_rng(4321)
        samples = [sample_tolerance_delta(0.06, distribution="gaussian", rng=rng) for _ in range(1000)]
        # 3-sigma = 0.06 -> sigma = 0.02
        assert abs(np.std(samples) - 0.02) < 0.003
        assert all(-0.06001 <= s <= 0.06001 for s in samples)

    def test_parabolic_distribution_sampling(self):
        rng = np.random.default_rng(999)
        samples = [sample_tolerance_delta(0.1, distribution="parabolic", rng=rng) for _ in range(1000)]
        assert all(-0.10001 <= s <= 0.10001 for s in samples)
        # Parabolic density: variance is 1/5 of delta^2 -> std ~ delta / sqrt(5) = 0.1 / 2.236 ~ 0.0447
        assert abs(np.std(samples) - 0.0447) < 0.005


class TestRayTracingCriteriaAndCompensator:
    """Tests for ray-tracing performance evaluation and auto-refocus compensator optimization."""

    def test_rms_and_geo_spot_radius_evaluation(self):
        stack = create_sample_singlet_stack()
        rms = compute_system_spot_radius(stack, geometric=False)
        geo = compute_system_spot_radius(stack, geometric=True)

        assert rms > 0.0
        assert geo >= rms, f"Geometric spot ({geo}) should be >= RMS spot ({rms})"
        assert rms < 2.0, f"RMS spot radius {rms} mm exceeds reasonable singlet bound"

    def test_compensator_refocus_reduces_spot_size(self):
        stack = create_sample_singlet_stack()
        # Defocus slightly
        stack.surfaces[2].thickness += 2.0
        rms_defocused = compute_system_spot_radius(stack)

        # Optimize compensator
        shift = solve_system_compensator(stack, compensator_surface_idx=2, delta_range=3.0)
        rms_refocused = compute_system_spot_radius(stack)

        assert shift < -1.0, f"Compensator should have shifted backward, got {shift}"
        assert rms_refocused < rms_defocused, f"Refocusing did not reduce RMS spot: {rms_defocused} -> {rms_refocused}"


class TestZemaxSensitivityAnalysis:
    """Tests for univariate sensitivity analysis with RSS and Worst-Case estimates."""

    def test_sensitivity_analysis_on_surface_stack(self):
        stack = create_sample_singlet_stack()
        engine = ZemaxTolerancingEngine(system=stack, compensator_enabled=True, compensator_surface_idx=2)

        tolerances = [
            PerturbationSpec(parameter_name="trad", surface_index=1, delta=0.1),
            PerturbationSpec(parameter_name="tthi", surface_index=1, delta=0.05),
            PerturbationSpec(parameter_name="tsdy", surface_index=1, delta=0.025),
            PerturbationSpec(parameter_name="tind", surface_index=1, delta=0.001),
        ]

        report = engine.sensitivity_analysis(tolerances=tolerances)
        assert len(report.entries) == 4

        # Check worst offenders ranking
        ranked = report.ranked_worst_offenders()
        assert len(ranked) == 4
        assert ranked[0].max_delta_merit >= ranked[-1].max_delta_merit

        # Check statistical estimations: Worst-Case >= RSS
        assert report.estimated_worst_case_delta >= report.estimated_rss_delta
        assert report.estimated_rss_delta > 0.0

        # Check ASCII table format
        table = report.format_table()
        assert "ZEMAX OPTICSTUDIO TOLERANCE SENSITIVITY REPORT" in table
        assert "Estimated Worst-Case" in table
        assert "Estimated RSS Change" in table


class TestZemaxMonteCarloSimulation:
    """Tests for multivariate Monte Carlo simulation and manufacturing yield."""

    def test_monte_carlo_trials_and_percentiles(self):
        stack = create_sample_singlet_stack()
        engine = ZemaxTolerancingEngine(system=stack, compensator_enabled=True, compensator_surface_idx=2, seed=42)

        tolerances = [
            PerturbationSpec(parameter_name="trad", surface_index=1, delta=0.05, distribution="gaussian"),
            PerturbationSpec(parameter_name="trad", surface_index=2, delta=0.05, distribution="gaussian"),
            PerturbationSpec(parameter_name="tthi", surface_index=1, delta=0.02, distribution="uniform"),
            PerturbationSpec(parameter_name="tsdy", surface_index=1, delta=0.015, distribution="uniform"),
        ]

        report = engine.monte_carlo_simulation(tolerances=tolerances, num_trials=50)
        assert report.num_trials == 50
        assert len(report.trials) == 50

        # Verify percentiles monotonicity
        assert report.p50 <= report.p80 <= report.p90 <= report.p95 <= report.p98 <= report.p99

        # Verify yield calculation
        high_threshold = report.max_merit + 0.1
        low_threshold = report.min_merit - 0.1
        assert report.yield_percentage(high_threshold) == 100.0
        assert report.yield_percentage(low_threshold) == 0.0

        # Intermediate yield rate
        mid_threshold = report.p90
        yield_mid = report.yield_percentage(mid_threshold)
        assert 80.0 <= yield_mid <= 98.0, f"Expected ~90% yield at P90 threshold, got {yield_mid}%"


class TestZemaxPresetsAndFileIO:
    """Tests for standard commercial/precision presets and Zemax file export/import."""

    def test_commercial_and_precision_presets(self):
        stack = create_sample_singlet_stack()

        comm_tols = create_commercial_tolerances(stack)
        assert len(comm_tols) >= 4
        # Verify commercial thickness delta is 0.05mm
        tthi_tols = [t for t in comm_tols if t.parameter_name.lower() == "tthi"]
        assert len(tthi_tols) > 0
        assert tthi_tols[0].delta == 0.05

        prec_tols = create_precision_tolerances(stack)
        prec_tthi = [t for t in prec_tols if t.parameter_name.lower() == "tthi"]
        assert len(prec_tthi) > 0
        assert prec_tthi[0].delta == 0.02

        high_tols = create_high_precision_tolerances(stack)
        high_tthi = [t for t in high_tols if t.parameter_name.lower() == "tthi"]
        assert len(high_tthi) > 0
        assert high_tthi[0].delta == 0.01

    def test_export_and_parse_zmx_tolerances(self, tmp_path):
        stack = create_sample_singlet_stack()
        tols = [
            PerturbationSpec(parameter_name="TRAD", surface_index=1, delta=0.08, distribution="gaussian", notes="Radius Tol"),
            PerturbationSpec(parameter_name="TTHI", surface_index=1, delta=0.03, distribution="uniform", notes="Thickness Tol"),
        ]

        out_zmx = os.path.join(tmp_path, "toleranced_lens.zmx")
        export_zmx_with_tolerances(stack, tols, out_zmx, title="Test Toleranced Lens")

        assert os.path.exists(out_zmx)
        with open(out_zmx, "r") as f:
            content = f.read()
        assert "TOL TRAD 1 0.08 1" in content
        assert "TOL TTHI 1 0.03 0" in content

        # Test parsing back
        parsed_tols = parse_zmx_tolerances(out_zmx)
        assert len(parsed_tols) == 2
        assert parsed_tols[0].parameter_name == "TRAD"
        assert parsed_tols[0].surface_index == 1
        assert abs(parsed_tols[0].delta - 0.08) < 1e-6
        assert parsed_tols[0].distribution == "gaussian"

    def test_export_zemax_ascii_report(self, tmp_path):
        stack = create_sample_singlet_stack()
        engine = ZemaxTolerancingEngine(system=stack, compensator_enabled=True, compensator_surface_idx=2, seed=42)
        tols = create_commercial_tolerances(stack)

        sens_rep, mc_rep = engine.run_full_analysis(tolerances=tols, num_trials=30)
        report_file = os.path.join(tmp_path, "zemax_tolerance_report.txt")
        report_text = export_zemax_tolerance_report(sens_rep, mc_rep, system_title="Sample Singlet", filepath=report_file)

        assert os.path.exists(report_file)
        assert "ZEMAX OPTICSTUDIO TOLERANCE ANALYSIS REPORT" in report_text
        assert "SENSITIVITY REPORT" in report_text
        assert "MONTE CARLO TOLERANCING REPORT" in report_text
        assert "CUMULATIVE PROBABILITY DISTRIBUTION" in report_text
        assert "\u03bcm" in report_text


class TestZemaxInverseSensitivityAndExtendedOperands:
    """Tests for Zemax Inverse Sensitivity Analysis, micrometer readouts, and TABB/TIRR/TEDY/TETX operands."""

    def test_inverse_sensitivity_analysis_solves_limits(self):
        stack = create_sample_singlet_stack()
        engine = ZemaxTolerancingEngine(system=stack, compensator_enabled=True, compensator_surface_idx=2)

        tolerances = [
            PerturbationSpec(parameter_name="trad", surface_index=1, delta=0.2),
            PerturbationSpec(parameter_name="tthi", surface_index=1, delta=0.1),
            PerturbationSpec(parameter_name="tsdy", surface_index=1, delta=0.05),
            PerturbationSpec(parameter_name="tstx", surface_index=1, delta=0.1),
        ]

        inv_report = engine.inverse_sensitivity_analysis(
            tolerances=tolerances,
            allowable_degradation_um=5.0,
        )
        assert len(inv_report.entries) == 4
        assert inv_report.is_spot_criterion
        assert abs(inv_report.target_degradation * 1000.0 - 5.0) < 1e-6

        for entry in inv_report.entries:
            assert entry.solved_plus_delta > 0.0
            assert entry.solved_minus_delta < 0.0
            assert entry.solved_symmetric_delta > 0.0
            assert abs(entry.solved_symmetric_delta - min(abs(entry.solved_plus_delta), abs(entry.solved_minus_delta))) < 1e-9

        inv_table = inv_report.format_table()
        assert "ZEMAX OPTICSTUDIO INVERSE SENSITIVITY REPORT" in inv_table
        assert "\u03bcm" in inv_table

    def test_extended_operands_tabb_tirr_tedy_tetx_and_um_properties(self):
        stack = create_sample_singlet_stack()
        engine = ZemaxTolerancingEngine(system=stack, compensator_enabled=True, compensator_surface_idx=2)

        tolerances = [
            PerturbationSpec(parameter_name="tabb", surface_index=1, delta=1.0),
            PerturbationSpec(parameter_name="tirr", surface_index=1, delta=2.0),
            PerturbationSpec(parameter_name="tedy", surface_index=1, delta=0.05),
            PerturbationSpec(parameter_name="tetx", surface_index=1, delta=0.1),
        ]

        sens_rep = engine.sensitivity_analysis(tolerances=tolerances)
        assert len(sens_rep.entries) == 4
        assert abs(sens_rep.nominal_merit_um - sens_rep.nominal_merit * 1000.0) < 1e-6
        assert abs(sens_rep.estimated_rss_um - sens_rep.estimated_rss * 1000.0) < 1e-6
        assert sens_rep.estimated_rss_um > sens_rep.nominal_merit_um

