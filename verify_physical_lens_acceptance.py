"""
Standalone Verification Script for Physical Lens Design Acceptance Criteria.
=============================================================================
Verifies:
- AC1: Thick Lens Cardinal Points Benchmark (<1% error against analytical benchmark).
- AC2: Galileo Imaging System Layout (Afocal condition C=0.0, Angular Magnification = 4.0).
- AC3: Unoptimized Lens Optimization Convergence (EFL=20.0mm, Diam=25.4mm within <= 10 iterations).
- AC4: Clean Isolation into dedicated engine modules (no UI imports, modular structure).
- Tolerancing: Sensitivity Analysis and Monte Carlo Yield Simulation (Zemax-style).
"""

import sys
import os
import math
import numpy as np

# Ensure project root in sys.path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def verify_ac1_thick_lens_benchmark():
    print("\n" + "=" * 70)
    print("VERIFYING AC1: Thick Lens Analytical Benchmark")
    print("=" * 70)
    from optics_app.engine.thick_lens import ThickLens

    # Analytical benchmark parameters:
    # R1 = +50.0 mm, R2 = -50.0 mm, t = 10.0 mm, n = 1.50
    lens = ThickLens(r1=50.0, r2=-50.0, thickness=10.0, glass_or_index=1.5, diameter=25.4)
    cp = lens.cardinal_points()

    efl_exact = 1500.0 / 29.0   # 51.72413793...
    bfl_exact = 1400.0 / 29.0   # 48.27586207...
    p1_exact = 100.0 / 29.0     # +3.44827586...
    p2_exact = -100.0 / 29.0    # -3.44827586...

    rel_err_efl = abs(cp.efl - efl_exact) / efl_exact
    rel_err_bfl = abs(cp.bfl - bfl_exact) / bfl_exact
    rel_err_p1 = abs(cp.p1 - p1_exact) / p1_exact
    rel_err_p2 = abs(cp.p2 - p2_exact) / abs(p2_exact)

    print(f"  EFL:  Calculated = {cp.efl:12.6f} mm | Analytical = {efl_exact:12.6f} mm | Rel Error = {rel_err_efl:.6%}")
    print(f"  BFL:  Calculated = {cp.bfl:12.6f} mm | Analytical = {bfl_exact:12.6f} mm | Rel Error = {rel_err_bfl:.6%}")
    print(f"  P1:   Calculated = {cp.p1:12.6f} mm | Analytical = {p1_exact:12.6f} mm | Rel Error = {rel_err_p1:.6%}")
    print(f"  P2:   Calculated = {cp.p2:12.6f} mm | Analytical = {p2_exact:12.6f} mm | Rel Error = {rel_err_p2:.6%}")

    assert rel_err_efl < 0.01, f"AC1 Failed: EFL error {rel_err_efl:.4%} >= 1%"
    assert rel_err_bfl < 0.01, f"AC1 Failed: BFL error {rel_err_bfl:.4%} >= 1%"
    assert rel_err_p1 < 0.01, f"AC1 Failed: P1 error {rel_err_p1:.4%} >= 1%"
    assert rel_err_p2 < 0.01, f"AC1 Failed: P2 error {rel_err_p2:.4%} >= 1%"

    # Optical invariant check: BFL - P2 == EFL
    conjugate_err = abs((cp.bfl - cp.p2) - cp.efl)
    print(f"  Conjugate Invariant (BFL - P2 == EFL): Discrepancy = {conjugate_err:.2e} mm")
    assert conjugate_err < 1e-6

    # Transfer matrix determinant check
    mat = lens.transfer_matrix()
    det_val = mat.determinant()
    print(f"  Transfer Matrix Determinant det(M) = {det_val:.12f} (Error from 1.0: {abs(det_val - 1.0):.2e})")
    assert abs(det_val - 1.0) < 1e-12

    print("[PASS] AC1 Thick Lens Analytical Benchmark Passed within <1% (<1e-6 actual discrepancy)!")


def verify_ac2_galileo_system():
    print("\n" + "=" * 70)
    print("VERIFYING AC2: Galileo Imaging System Layout")
    print("=" * 70)
    from optics_app.engine.multi_component import create_galileo_system

    # Basic Galilean layout: f1 = +100 mm, f2 = -25 mm
    galileo = create_galileo_system(f1=100.0, f2=-25.0)
    sys_mat = galileo.system_matrix()

    print(f"  Galileo System Matrix ABCD:")
    print(f"    [ {sys_mat.A:10.6f}  {sys_mat.B:10.6f} ]")
    print(f"    [ {sys_mat.C:10.6e}  {sys_mat.D:10.6f} ]")

    # Afocal condition: Power C == 0.0
    print(f"  Afocal condition power C = {sys_mat.C:.6e} mm^-1")
    assert abs(sys_mat.C) < 1e-9, f"AC2 Failed: Afocal condition violated (C={sys_mat.C})"

    # Angular magnification: D == 4.0
    ang_mag = galileo.angular_magnification()
    print(f"  Angular Magnification M_alpha = {ang_mag:.6f}x (Target = 4.000000x)")
    assert abs(ang_mag - 4.0) < 1e-6, f"AC2 Failed: Angular magnification {ang_mag} != 4.0"

    # Lateral magnification: A == 0.25
    lat_mag = galileo.lateral_magnification()
    print(f"  Lateral Magnification m = {lat_mag:.6f} (Target = 0.250000)")
    assert abs(lat_mag - 0.25) < 1e-6, f"AC2 Failed: Lateral magnification {lat_mag} != 0.25"

    # Paraxial ray tracing check
    y_out, u_out = galileo.propagate_ray(y=0.0, theta=0.01)
    print(f"  Off-axis ray tracing (y_in=0 mm, theta_in=10 mrad):")
    print(f"    y_out = {y_out:.6f} mm (Expected = 0.75 mm)")
    print(f"    theta_out = {u_out:.6f} rad (Expected = 0.04 rad, 4x angular expansion)")
    assert abs(y_out - 0.75) < 1e-6
    assert abs(u_out - 0.04) < 1e-6

    print("[PASS] AC2 Galileo System Layout Passed (C=0.0 afocal, M_alpha=4.0)!")


def verify_ac3_optimization_convergence():
    print("\n" + "=" * 70)
    print("VERIFYING AC3: Lens Optimization Convergence")
    print("=" * 70)
    from optics_app.engine.thick_lens import ThickLens
    from optics_app.engine.lens_merit import PhysicalMeritFunction, TargetEFLOperand, DiameterConstraintOperand
    from optics_app.engine.lens_optimizer import ScipyLeastSquaresOptimizer

    # Starting unoptimized lens:
    # R1 = 50 mm, R2 = -50 mm, t = 5 mm, D = 20 mm, n = 1.5168
    # Initial EFL ~ 50.85 mm, initial diameter = 20.0 mm
    lens = ThickLens(r1=50.0, r2=-50.0, thickness=5.0, glass_or_index=1.5168, diameter=20.0)
    initial_efl = lens.cardinal_points().efl
    initial_diam = lens.diameter

    print(f"  Initial State: EFL = {initial_efl:.4f} mm, Diameter = {initial_diam:.4f} mm")

    mf = PhysicalMeritFunction()
    mf.add_operand(TargetEFLOperand(target_efl=20.0, weight=1.0))
    mf.add_operand(DiameterConstraintOperand(target_diameter=25.4, weight=1.0))

    initial_merit = mf.evaluate(lens)
    print(f"  Initial Merit Value: {initial_merit:.6f}")

    optimizer = ScipyLeastSquaresOptimizer(merit_function=mf, max_iterations=20)
    result = optimizer.optimize(lens)

    final_efl = lens.cardinal_points().efl
    final_diam = lens.diameter
    final_merit = result.final_merit
    iterations = result.iterations

    print(f"  Optimizer Converged: {result.success}")
    print(f"  Iterations: {iterations} (Required: <= 10 iterations)")
    print(f"  Final State: EFL = {final_efl:.6f} mm (Target: 20.0 mm, Error: {abs(final_efl - 20.0)/20.0:.4%})")
    print(f"  Final State: Diameter = {final_diam:.6f} mm (Target: 25.4 mm, Error: {abs(final_diam - 25.4)/25.4:.4%})")
    print(f"  Final Merit Value: {final_merit:.6e} (Reduction: {initial_merit / max(final_merit, 1e-12):.1f}x)")

    assert result.success is True, "AC3 Failed: Optimizer did not converge"
    assert iterations <= 10, f"AC3 Failed: Convergence took {iterations} iterations (>10)"
    assert abs(final_efl - 20.0) / 20.0 < 0.001, f"AC3 Failed: EFL {final_efl} deviated >0.1% from 20.0 mm"
    assert abs(final_diam - 25.4) / 25.4 < 0.001, f"AC3 Failed: Diameter {final_diam} deviated >0.1% from 25.4 mm"

    print("[PASS] AC3 Optimization Convergence Passed (EFL=20.0mm, Diam=25.4mm in <= 10 iterations)!")


def verify_ac4_code_isolation():
    print("\n" + "=" * 70)
    print("VERIFYING AC4: Code Architecture & Module Isolation")
    print("=" * 70)

    engine_dir = os.path.join(PROJECT_ROOT, "optics_app", "engine")
    required_modules = [
        "matrix_optics.py",
        "thick_lens.py",
        "aspheric.py",
        "multi_component.py",
        "lens_merit.py",
        "lens_optimizer.py",
        "tolerancing.py",
    ]

    for mod in required_modules:
        path = os.path.join(engine_dir, mod)
        assert os.path.isfile(path), f"AC4 Failed: Required module {mod} missing in {engine_dir}"
        size = os.path.getsize(path)
        assert size > 500, f"AC4 Failed: Module {mod} is suspiciously empty ({size} bytes)"

        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        # Verify zero UI dependencies
        for forbidden in ["PyQt5", "PyQt6", "PySide2", "PySide6", "tkinter", "wx"]:
            assert f"import {forbidden}" not in content, f"AC4 Violation: {mod} imports UI module {forbidden}"
            assert f"from {forbidden}" not in content, f"AC4 Violation: {mod} imports from UI module {forbidden}"

        print(f"  [OK] optics_app/engine/{mod:<22} ({size:>6} bytes, cleanly isolated, zero UI imports)")

    print("[PASS] AC4 Clean Module Isolation Verified!")


def verify_tolerancing_and_monte_carlo():
    print("\n" + "=" * 70)
    print("VERIFYING TOLERANCING: Sensitivity Analysis & Monte Carlo Yield")
    print("=" * 70)
    from optics_app.engine.thick_lens import ThickLens
    from optics_app.engine.lens_merit import PhysicalMeritFunction, TargetEFLOperand
    from optics_app.engine.tolerancing import TolerancingEngine, PerturbationSpec

    lens = ThickLens(r1=50.0, r2=-50.0, thickness=5.0, glass_or_index=1.5168, diameter=25.4)
    nom_efl = lens.cardinal_points().efl

    mf = PhysicalMeritFunction()
    mf.add_operand(TargetEFLOperand(target_efl=nom_efl, weight=1.0))

    tols = [
        PerturbationSpec(parameter_name="r1", element_index=0, delta=0.1, distribution="normal"),
        PerturbationSpec(parameter_name="r2", element_index=0, delta=0.1, distribution="normal"),
        PerturbationSpec(parameter_name="thickness", element_index=0, delta=0.05, distribution="uniform"),
        PerturbationSpec(parameter_name="index", element_index=0, delta=0.001, distribution="uniform"),
    ]

    engine = TolerancingEngine()

    # 1. Sensitivity Analysis
    sens_report = engine.sensitivity_analysis(lens, mf, tols)
    ranked = sens_report.ranked_worst_offenders()
    print("  Sensitivity Analysis Ranked Worst Offenders:")
    for idx, item in enumerate(ranked, 1):
        print(f"    Rank {idx}: Parameter={item.parameter_name:<10} | Delta={item.delta:+.4f} | Delta Merit={item.delta_merit:+.4e} | Sensitivity={item.sensitivity:+.4e}")

    assert len(ranked) == 4
    # Verify nominal immutability
    post_efl = lens.cardinal_points().efl
    assert abs(post_efl - nom_efl) < 1e-12, "Nominal lens mutated during sensitivity analysis!"

    # 2. Monte Carlo Simulation
    mc_report = engine.monte_carlo_simulation(lens, mf, tols, num_trials=500, seed=42)
    print(f"\n  Monte Carlo Simulation ({mc_report.num_trials} trials):")
    print(f"    Mean Merit = {mc_report.mean_merit:.6f} +/- {mc_report.std_merit:.6f}")
    print(f"    P50 (Median) = {mc_report.p50:.6f}")
    print(f"    P90 = {mc_report.p90:.6f}")
    print(f"    P95 = {mc_report.p95:.6f}")
    print(f"    P99 = {mc_report.p99:.6f}")
    print(f"    Paraxial EFL Distribution = {mc_report.mean_efl:.4f} +/- {mc_report.std_efl:.4f} mm")

    # Statistical invariant: Monotonic percentiles
    assert mc_report.p50 <= mc_report.p90 <= mc_report.p95 <= mc_report.p99, "Percentile monotonicity violated!"

    # Manufacturing yield rate check for threshold
    yield_val = mc_report.yield_rate(threshold=mc_report.p95)
    print(f"    Yield Rate at P95 threshold: {yield_val:.1%} (Expected ~ 95%)")
    assert 0.90 <= yield_val <= 0.98

    print("[PASS] Optical Tolerancing & Monte Carlo Simulation Verified!")


def main():
    print("\n" + "#" * 70)
    print("# PHYSICAL LENS DESIGN FEATURE: ACCEPTANCE CRITERIA VERIFICATION")
    print("#" * 70)

    try:
        verify_ac1_thick_lens_benchmark()
        verify_ac2_galileo_system()
        verify_ac3_optimization_convergence()
        verify_ac4_code_isolation()
        verify_tolerancing_and_monte_carlo()

        print("\n" + "#" * 70)
        print("# ALL 4 ACCEPTANCE CRITERIA & TOLERANCING FULLY VERIFIED (100% PASS)")
        print("#" * 70 + "\n")
        return 0
    except Exception as e:
        print(f"\n[FAIL] Verification encountered an error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
