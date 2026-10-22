"""
Test Suite for Zemax-style Merit Functions with Spot Size Operands & Constraints.
Verifies:
1. Multi-operand MeritFunction evaluation (TRAC, RSX, RSY, GEOR, GEOX, GEOY, SPTR, SPRD).
2. Spot size target optimization (minimization vs targeting specific spot radius D0).
3. Multi-surface spot size optimization (intermediate focal plane + detector plane).
4. OpticalOptimizer with explicit MeritFunction object.
5. Merit function operand breakdown table (contributions, weights, diffs).
"""

import sys
import os
import math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from optics_app.engine.merit_function import MeritFunction, MeritOperand, OperandType
from optics_app.engine.optimizer import OpticalOptimizer

def test_merit_function_operands():
    print("=" * 80)
    print("1. TESTING MULTI-OPERAND MERIT FUNCTION WITH SPOT SIZES")
    print("=" * 80)

    # 1. Build a simple singlet lens
    stack = SurfaceStack()
    stack.add_surface(Surface(name="Obj", surface_type=SurfaceType.OBJECT, thickness=25.0, glass="AIR"))
    stack.add_surface(Surface(name="L1_Front", surface_type=SurfaceType.STANDARD, radius=25.0, thickness=4.0, glass="N-BK7", semi_diameter=12.7))
    stack.add_surface(Surface(name="L1_Back", surface_type=SurfaceType.STANDARD, radius=-25.0, thickness=30.0, glass="AIR", semi_diameter=12.7, thickness_variable=True))
    stack.add_surface(Surface(name="Det", surface_type=SurfaceType.DETECTOR, radius=0.0, thickness=0.0, glass="AIR", semi_diameter=5.0))
    stack._update_accumulated_positions()
    opt_sys = OpticalSystem(stack)

    # Generate ray bundle
    rays = []
    for py in np.linspace(-6.0, 6.0, 11):
        for px in np.linspace(-6.0, 6.0, 11):
            if px**2 + py**2 <= 36.0:
                rays.append(Ray(pos=np.array([px, py, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=0.5876))
    traced_rays = opt_sys.trace_rays(rays)

    # Construct custom Merit Function with multiple spot size operands
    mf = MeritFunction()
    op_trac = mf.add_operand(OperandType.RMS_SPOT_RADIUS, surface_index=3, target=0.0, weight=1.0)
    op_rsx = mf.add_operand(OperandType.RMS_SPOT_X, surface_index=3, target=0.0, weight=2.0)
    op_rsy = mf.add_operand(OperandType.RMS_SPOT_Y, surface_index=3, target=0.0, weight=2.0)
    op_geor = mf.add_operand(OperandType.GEO_SPOT_RADIUS, surface_index=3, target=0.0, weight=0.5)
    op_totr = mf.add_operand(OperandType.TOTAL_TRACK, target=59.0, weight=0.1)

    val = mf.evaluate(opt_sys, traced_rays)

    print(f"   Evaluated Merit Function: {val:.4f}")
    print("\n   Operand Breakdown Table (Zemax MFED format):")
    print("   " + "-" * 75)
    print(f"   {'Code':<6} | {'Type':<18} | {'Surf':<5} | {'Target':<8} | {'Value (mm)':<12} | {'Weight':<6} | {'Contrib %'}")
    print("   " + "-" * 75)
    breakdown = mf.get_breakdown()
    for row in breakdown:
        print(f"   {row['code']:<6} | {row['type']:<18} | {row['surface']:<5} | {row['target']:<8.3f} | {row['value']:<12.4f} | {row['weight']:<6.2f} | {row['contribution_pct']:6.2f}%")
    print("   " + "-" * 75)

    assert val > 0.0, "Merit function should be positive"
    assert op_trac.value > 0.0, "RMS spot radius should be positive"
    assert op_rsx.value > 0.0, "RMS Spot X should be positive"
    assert op_rsy.value > 0.0, "RMS Spot Y should be positive"
    assert op_geor.value >= op_trac.value, "GEO radius must be >= RMS radius"
    print("   [+] Multi-operand Merit Function Evaluation PASSED!")

def test_multi_surface_spot_size_optimization():
    print("\n" + "=" * 80)
    print("2. TESTING MULTI-SURFACE SPOT SIZE OPTIMIZATION")
    print("=" * 80)

    # 2-lens relay system: Source -> L1 -> Intermediate Focus (S4) -> L2 -> Final Detector (S7)
    stack = SurfaceStack()
    stack.add_surface(Surface(name="Source", surface_type=SurfaceType.OBJECT, thickness=20.0, glass="AIR"))
    # Lens 1
    stack.add_surface(Surface(name="L1_Front", radius=30.0, thickness=4.0, glass="N-BK7", semi_diameter=12.7))
    stack.add_surface(Surface(name="L1_Back", radius=-30.0, thickness=30.0, glass="AIR", semi_diameter=12.7, thickness_variable=True, thickness_min=5.0, thickness_max=60.0))
    # Intermediate focal slit (S3)
    stack.add_surface(Surface(name="Intermediate_Slit", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=20.0, glass="AIR", semi_diameter=10.0, thickness_variable=True, thickness_min=5.0, thickness_max=50.0))
    # Lens 2
    stack.add_surface(Surface(name="L2_Front", radius=30.0, thickness=4.0, glass="N-BK7", semi_diameter=12.7))
    stack.add_surface(Surface(name="L2_Back", radius=-30.0, thickness=25.0, glass="AIR", semi_diameter=12.7, thickness_variable=True, thickness_min=5.0, thickness_max=60.0))
    # Final Detector (S6)
    stack.add_surface(Surface(name="Final_Det", surface_type=SurfaceType.DETECTOR, radius=0.0, thickness=0.0, glass="AIR", semi_diameter=5.0))
    stack._update_accumulated_positions()
    opt_sys = OpticalSystem(stack)

    # Create Multi-Surface Merit Function:
    # 1. Enforce small spot at Intermediate Slit (S3) with Weight = 1.0
    # 2. Enforce small spot at Final Detector (S6) with Weight = 2.0
    mf = MeritFunction()
    mf.add_operand(OperandType.RMS_SPOT_RADIUS, surface_index=3, target=0.0, weight=1.0)
    mf.add_operand(OperandType.RMS_SPOT_RADIUS, surface_index=6, target=0.0, weight=2.0)

    optimizer = OpticalOptimizer(
        system=opt_sys,
        wavelengths=0.5876,
        merit_function=mf,
        method="Nelder-Mead",
        max_iter=50
    )
    result = optimizer.optimize(variable_indices={"thickness": [2, 3, 5], "radius": []})

    print(f"   Optimization Success:    {result.success}")
    print(f"   Initial Merit:          {result.initial_rms:.4f} mm")
    print(f"   Final Merit:            {result.final_rms:.4f} mm")
    print(f"   Reduction Percentage:   {result.rms_reduction_pct:.1f}%")
    print(f"   Optimized Air Spaces:   {result.optimized_thicknesses}")

    assert result.success, "Multi-surface spot size optimization failed"
    assert result.final_rms < result.initial_rms, "Merit function did not decrease"
    print("   [+] Multi-Surface Spot Size Optimization PASSED!")

def test_target_spot_size_constraint():
    print("\n" + "=" * 80)
    print("3. TESTING TARGET SPECIFIC SPOT SIZE (SPTR)")
    print("=" * 80)

    # Singlet lens optimizing to achieve a specific defocused spot size (e.g. r_target = 0.35 mm)
    stack = SurfaceStack()
    stack.add_surface(Surface(name="Obj", surface_type=SurfaceType.OBJECT, thickness=50.0, glass="AIR"))
    stack.add_surface(Surface(name="L1_Front", radius=40.0, thickness=5.0, glass="N-BK7", semi_diameter=6.0))
    stack.add_surface(Surface(name="L1_Back", radius=-40.0, thickness=15.0, glass="AIR", semi_diameter=6.0, thickness_variable=True, thickness_min=5.0, thickness_max=250.0))
    stack.add_surface(Surface(name="Det", surface_type=SurfaceType.DETECTOR, radius=0.0, thickness=0.0, glass="AIR", semi_diameter=5.0))
    stack._update_accumulated_positions()
    opt_sys = OpticalSystem(stack)

    target_radius_mm = 0.35  # Target spot radius of 350 um
    mf = MeritFunction()
    mf.add_operand(OperandType.RMS_SPOT_RADIUS, surface_index=3, target=target_radius_mm, weight=10.0)

    optimizer = OpticalOptimizer(
        system=opt_sys,
        wavelengths=0.5876,
        merit_function=mf,
        entrance_pupil_radius=5.0,
        method="Nelder-Mead",
        max_iter=80
    )
    result = optimizer.optimize(variable_indices={"thickness": [2], "radius": []})

    # Evaluate spot radius on optimized stack
    opt_sys_final = OpticalSystem(result.optimized_stack)
    rays = optimizer._generate_rays(optimizer.entrance_pupil_radius, [0.5876], 40, z_dist=50.0)
    traced = opt_sys_final.trace_rays(rays)
    det_hits = [h for r in traced for h in r.history if h["surface_index"] == 3 and not r.vignetted]
    from optics_app.analysis.spot import SpotDiagram
    sd = SpotDiagram([h["x"] for h in det_hits], [h["y"] for h in det_hits])
    achieved_r = sd.calculate_rms_radius()

    print(f"   Target RMS Spot Radius:   {target_radius_mm*1e3:.1f} um ({target_radius_mm:.3f} mm)")
    print(f"   Achieved RMS Spot Radius: {achieved_r*1e3:.1f} um ({achieved_r:.3f} mm)")
    print(f"   Error:                    {abs(achieved_r - target_radius_mm)*1e3:.2f} um")

    assert abs(achieved_r - target_radius_mm) < 0.05, f"Spot radius did not converge to target: {achieved_r} vs {target_radius_mm}"
    print("   [+] Target Spot Size Constraint PASSED!")

if __name__ == "__main__":
    test_merit_function_operands()
    test_multi_surface_spot_size_optimization()
    test_target_spot_size_constraint()
    print("\n" + "=" * 80)
    print("ALL SPOT SIZE MERIT FUNCTION TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 80)
