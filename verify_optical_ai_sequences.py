"""
Standalone Verification & Demonstration Script for Optical AI Sequences (Milestones 3 & 4).
============================================================================================
Demonstrates and validates:
1. Multi-stage automated optical design sequence on a 25 mm aspheric singlet (F/2, N-BK7).
2. Step-by-step manual execution and inspection via OpticalDesignSequenceRunner.step().
3. AI Tool Registry integration via registry.execute("run_optical_design_sequence", ...).
4. Granular AI sequence tools (bending, aspherization, tolerancing).
5. Natural language command execution via interpret_and_execute_command.
6. Programmatic validation of user acceptance criteria:
   - Focal length accuracy: |EFL - 25.0| / 25.0 < 0.001 (<0.1% error).
   - Merit function reduction: > 10x improvement from initial unoptimized state.
   - Spot size reduction: < 10 um RMS spot radius (diffraction-limited regime).
   - Manufacturing yield: >= 90% under standard commercial fabrication tolerances.
"""

import sys
import os
import math
from typing import Dict, Any, List

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from optics_app.engine.ai_sequences import OpticalDesignSequenceRunner
from optics_app.ai.tools import registry, interpret_and_execute_command
from optics_app.engine.aspheric import AsphericLens


def print_banner(text: str, char: str = "=") -> None:
    line = char * 80
    print(f"\n{line}\n  {text}\n{line}")


def verify_step_by_step_sequence() -> Dict[str, Any]:
    """Execute sequence step-by-step to demonstrate manual debugging transparency."""
    print_banner("PART 1: STEP-BY-STEP MANUAL DEBUGGING EXECUTION (.step())")
    print("Initializing OpticalDesignSequenceRunner(target_efl=25.0, target_diameter=12.5, target_f_number=2.0, glass='N-BK7')...")
    
    runner = OpticalDesignSequenceRunner(
        target_efl=25.0,
        target_diameter=12.5,
        target_f_number=2.0,
        glass="N-BK7"
    )

    stage_records: List[Dict[str, Any]] = []

    stage_idx = 1
    while not runner.is_finished:
        st_name = runner.current_stage_name
        print(f"\n>>> Executing Stage {stage_idx}: {st_name}...")
        res = runner.step()
        stage_records.append(res)

        m_before = res.get("metrics_before", {})
        m_after = res.get("metrics_after", {})
        impr = res.get("improvement_factor", 1.0)
        
        print(f"    Status:              {res.get('status')}")
        print(f"    EFL:                 {m_after.get('efl', 0.0):.4f} mm (Target: 25.0000 mm)")
        print(f"    Merit:               {m_before.get('merit', 0.0):.4f} -> {m_after.get('merit', 0.0):.6f}")
        print(f"    Spot Radius:         {m_before.get('spot_radius', 0.0)*1000.0:8.2f} um -> {m_after.get('spot_radius', 0.0)*1000.0:8.2f} um")
        print(f"    Improvement Factor:  {impr:.2f}x")
        if "yield" in m_after and m_after["yield"] > 0:
            print(f"    Manufacturing Yield: {m_after['yield']:.1%}")
        stage_idx += 1

    summary = runner.get_summary()
    return {
        "runner": runner,
        "records": stage_records,
        "summary": summary
    }


def verify_tool_registry_execution() -> Dict[str, Any]:
    """Verify tool execution through the central ToolRegistry."""
    print_banner("PART 2: AI TOOL REGISTRY EXECUTION (registry.execute)")
    
    # 1. Full sequence tool
    print("Calling registry.execute('run_optical_design_sequence', ...)...")
    res_full = registry.execute("run_optical_design_sequence", {
        "target_efl": 25.0,
        "target_diameter": 25.4,
        "target_f_number": 2.0,
        "glass": "N-BK7"
    })
    assert isinstance(res_full, dict), f"Expected dict result, got: {type(res_full)}"
    assert res_full.get("status") == "SUCCESS", f"Sequence tool failed: {res_full}"
    print("  [PASS] run_optical_design_sequence executed successfully!")

    # 2. Lens bending tool
    print("Calling registry.execute('run_lens_bending_optimization', ...)...")
    res_bend = registry.execute("run_lens_bending_optimization", {
        "efl": 25.0,
        "diameter": 12.5,
        "glass": "N-BK7"
    })
    assert res_bend.get("status") == "SUCCESS", f"Bending tool failed: {res_bend}"
    print(f"  [PASS] run_lens_bending_optimization SUCCESS (Shape factor bent to R1={res_bend['current_lens'].r1:.2f}mm, R2={res_bend['current_lens'].r2:.2f}mm)")

    # 3. Aspheric optimization tool
    print("Calling registry.execute('run_aspheric_optimization', ...)...")
    res_asph = registry.execute("run_aspheric_optimization", {
        "efl": 25.0,
        "diameter": 12.5,
        "glass": "N-BK7"
    })
    assert res_asph.get("status") == "SUCCESS", f"Aspheric tool failed: {res_asph}"
    print(f"  [PASS] run_aspheric_optimization SUCCESS (Conic k1={res_asph['current_lens'].k1:.4f}, Spot={res_asph['metrics_after']['spot_radius']*1000.0:.2f} um)")

    # 4. Tolerancing sequence tool
    print("Calling registry.execute('run_tolerancing_sequence', ...)...")
    res_tol = registry.execute("run_tolerancing_sequence", {
        "efl": 25.0,
        "diameter": 12.5,
        "trials": 100
    })
    assert res_tol.get("status") == "SUCCESS", f"Tolerancing tool failed: {res_tol}"
    print(f"  [PASS] run_tolerancing_sequence SUCCESS (Yield rate: {res_tol['yield_rate']:.1%}, Trials: {res_tol['trials']})")

    return res_full


def verify_natural_language_nlp_integration() -> None:
    """Verify natural language heuristic interpretation into tool execution."""
    print_banner("PART 3: NATURAL LANGUAGE NLP COMMAND INTEGRATION")
    
    test_queries = [
        "run optical design sequence for 25mm lens",
        "optimize 25mm lens with ai sequence",
        "design optical element efl 25 diameter 25.4"
    ]

    for q in test_queries:
        print(f"Query: \"{q}\"")
        exec_res = interpret_and_execute_command(q)
        assert exec_res.get("success"), f"Command failed to execute: {q}"
        assert len(exec_res.get("tool_calls", [])) > 0, f"No tool calls generated for: {q}"
        tc_name = exec_res["tool_calls"][0]["function"]["name"]
        print(f"  -> Dispatched Tool: {tc_name} (Success: {exec_res['success']})")
        assert tc_name == "run_optical_design_sequence", f"Unexpected tool name: {tc_name}"

    print("  [PASS] All natural language command queries successfully routed and executed!")


def assert_and_print_summary(step_data: Dict[str, Any], registry_res: Dict[str, Any]) -> None:
    """Assert all quantitative acceptance criteria and display summary table."""
    print_banner("PART 4: ACCEPTANCE CRITERIA ASSERTIONS & SUMMARY TABLE")

    records = step_data["records"]
    init_m = records[0]["metrics_after"]
    final_m = records[-1]["metrics_after"]

    # 1. Focal length accuracy: < 0.1% relative error
    target_efl = 25.0
    actual_efl = final_m["efl"]
    efl_error_rel = abs(actual_efl - target_efl) / target_efl
    print(f"\n1. Focal Length Accuracy: Target = {target_efl:.4f} mm | Actual = {actual_efl:.6f} mm")
    print(f"   Relative Discrepancy = {efl_error_rel:.6%} (Threshold: < 0.100%)")
    assert efl_error_rel < 0.001, f"FAILED: EFL error {efl_error_rel:.4%} exceeds 0.1%"
    print("   [CRITERION 1 PASSED]: EFL error is well below 0.1%.")

    # 2. Merit function reduction: > 10x
    init_merit = init_m["merit"]
    final_merit = final_m["merit"]
    merit_reduction = init_merit / max(final_merit, 1e-12)
    print(f"\n2. Merit Function Reduction: Initial = {init_merit:.4f} | Final = {final_merit:.6f}")
    print(f"   Reduction Factor = {merit_reduction:.1f}x (Threshold: > 10.0x)")
    assert merit_reduction > 10.0, f"FAILED: Merit reduction factor {merit_reduction:.1f}x is <= 10x"
    print("   [CRITERION 2 PASSED]: Merit function reduced by > 10x.")

    # 3. Spot size reduction: < 10 um
    final_spot_um = final_m["spot_radius"] * 1000.0
    init_spot_um = init_m["spot_radius"] * 1000.0
    spot_reduction = init_spot_um / max(final_spot_um, 1e-6)
    print(f"\n3. Spot Size Reduction: Initial = {init_spot_um:.2f} um | Final = {final_spot_um:.3f} um")
    print(f"   Spot Size Reduction Factor = {spot_reduction:.1f}x (Threshold: Final < 10.0 um)")
    assert final_spot_um < 10.0, f"FAILED: Spot radius {final_spot_um:.2f} um is >= 10.0 um"
    print("   [CRITERION 3 PASSED]: Diffraction-limited spot radius < 10 um achieved.")

    # 4. Monte Carlo manufacturing yield: >= 90%
    yield_val = final_m.get("yield", 0.0)
    print(f"\n4. Monte Carlo Manufacturing Yield: Actual = {yield_val:.1%} (Threshold: >= 90.0%)")
    assert yield_val >= 0.90, f"FAILED: Manufacturing yield {yield_val:.1%} is < 90%"
    print("   [CRITERION 4 PASSED]: Manufacturing yield exceeds 90%.")

    # 5. Tool registry execution validation
    print("\n5. Central Tool Registry Direct Invocation:")
    reg_efl = registry_res["final_metrics"]["efl"]
    reg_spot_um = registry_res["final_metrics"]["spot_radius"] * 1000.0
    print(f"   Tool Return Status: {registry_res['status']}")
    print(f"   Tool Result EFL:    {reg_efl:.4f} mm")
    print(f"   Tool Result Spot:   {reg_spot_um:.3f} um")
    assert registry_res["status"] == "SUCCESS"
    assert abs(reg_efl - 25.0) / 25.0 < 0.001
    assert reg_spot_um < 10.0
    print("   [CRITERION 5 PASSED]: Tool registry execution confirmed.")

    # Formatted progression table
    print_banner("OPTICAL AI DESIGN SEQUENCE PROGRESSION TABLE", char="-")
    header = f"| {'#':<2} | {'Stage Name':<32} | {'Status':<7} | {'EFL (mm)':<10} | {'Merit':<12} | {'Spot (um)':<10} | {'Yield':<6} |"
    sep = "+" + "-" * 4 + "+" + "-" * 34 + "+" + "-" * 9 + "+" + "-" * 12 + "+" + "-" * 14 + "+" + "-" * 12 + "+" + "-" * 8 + "+"
    print(sep)
    print(header)
    print(sep)

    for idx, r in enumerate(records, 1):
        name = r["stage_name"]
        status = r["status"]
        m = r["metrics_after"]
        efl_str = f"{m.get('efl', 0.0):.4f}"
        merit_str = f"{m.get('merit', 0.0):.4e}" if m.get('merit', 0.0) < 0.01 else f"{m.get('merit', 0.0):.4f}"
        spot_str = f"{m.get('spot_radius', 0.0)*1000.0:.2f}"
        yield_str = f"{m.get('yield', 0.0):.0%}" if m.get("yield", 0.0) > 0 else "N/A"
        print(f"| {idx:<2} | {name:<32} | {status:<7} | {efl_str:<10} | {merit_str:<12} | {spot_str:<10} | {yield_str:<6} |")

    print(sep)
    print(f"\nFinal Lens Component: {type(step_data['runner'].current_lens).__name__}")
    lens = step_data['runner'].current_lens
    if isinstance(lens, AsphericLens):
        print(f"  Front Radius (R1):    {lens.r1:.4f} mm")
        print(f"  Rear Radius (R2):     {lens.r2:.4f} mm")
        print(f"  Center Thickness (tc):{lens.thickness:.4f} mm")
        print(f"  Conic Constant (k1):  {lens.k1:.6f}")
        alpha4 = lens.surface1.aspheric_coeffs[0] if len(lens.surface1.aspheric_coeffs) > 0 else 0.0
        print(f"  4th-Order Coeff (a4): {alpha4:.6e}")
        print(f"  Glass / Index:        {lens.glass_or_index}")

    print_banner("ALL 5 ACCEPTANCE CRITERIA & AI WORKFLOWS FULLY VERIFIED (100% PASS)")


def main():
    print_banner("2D OPTICS AI — OPTICAL SEQUENCES VERIFICATION & DEMONSTRATION")
    try:
        step_data = verify_step_by_step_sequence()
        registry_res = verify_tool_registry_execution()
        verify_natural_language_nlp_integration()
        assert_and_print_summary(step_data, registry_res)
        sys.exit(0)
    except Exception as e:
        print(f"\n[FATAL ERROR] Verification failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
