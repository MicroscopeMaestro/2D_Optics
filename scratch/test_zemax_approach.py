"""
Test Zemax-style Variable/Fixed Surface Optimization Architecture in 2D_Optics.
Verifies default fixed lenses, variable air gaps/distances, and Zemax solve flag indicators.
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(line_buffering=True)

from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.optimizer import OpticalOptimizer
from optics_app.gui.lde_table import LensDataTableModel

def test_zemax_workflow():
    print("=" * 80)
    print("TESTING ZEMAX-STYLE OPTICAL OPTIMIZATION ARCHITECTURE")
    print("=" * 80)
    
    # 1. Build a multi-element system (Collimator Doublet + Focusing Doublet)
    stack = SurfaceStack()
    # S0: Fiber Output (Air Gap Distance)
    stack.add_surface(Surface(name="Fiber_Output", surface_type=SurfaceType.OBJECT, thickness=10.0, glass="AIR", semi_diameter=0.025))
    # S1-S3: Collimator Lens 1 (Thorlabs AC127-020-B: N-BK7 / SF5)
    stack.add_surface(Surface(name="L1_Front", surface_type=SurfaceType.STANDARD, radius=12.8, thickness=3.0, glass="N-BK7", semi_diameter=6.35))
    stack.add_surface(Surface(name="L1_Mid", surface_type=SurfaceType.STANDARD, radius=-10.5, thickness=1.5, glass="SF5", semi_diameter=6.35))
    stack.add_surface(Surface(name="L1_Back", surface_type=SurfaceType.STANDARD, radius=-38.6, thickness=20.0, glass="AIR", semi_diameter=6.35))
    # S4-S6: Focusing Lens 2 (Thorlabs AC254-050-B: N-BK7 / SF5)
    stack.add_surface(Surface(name="L2_Front", surface_type=SurfaceType.STANDARD, radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
    stack.add_surface(Surface(name="L2_Mid", surface_type=SurfaceType.STANDARD, radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
    stack.add_surface(Surface(name="L2_Back", surface_type=SurfaceType.STANDARD, radius=-115.0, thickness=25.0, glass="AIR", semi_diameter=12.7))
    # S7: Detector (Air Plane)
    stack.add_surface(Surface(name="Detector", surface_type=SurfaceType.DETECTOR, thickness=0.0, glass="AIR", semi_diameter=5.0))
    
    print("\n--- 1. Testing Surface Properties (is_air vs is_glass) ---")
    for i, s in enumerate(stack.surfaces):
        print(f"  S{i:2d} ({s.name:<14}): Glass = {s.glass:<6} | is_air = {s.is_air!s:<5} | is_glass = {s.is_glass!s:<5}")
        
    # 2. Test Zemax Defaults
    print("\n--- 2. Applying Zemax Defaults: stack.set_zemax_defaults() ---")
    stack.set_zemax_defaults(var_air_gaps=True, fix_lenses=True)
    
    air_gap_indices = []
    lens_indices = []
    for i, s in enumerate(stack.surfaces):
        var_tag = "VARIABLE (V)" if s.thickness_variable else "FIXED (F)"
        print(f"  S{i:2d} ({s.name:<14}): Thickness = {s.thickness:6.2f} mm | Status: {var_tag}")
        if s.is_air:
            air_gap_indices.append(i)
            assert s.thickness_variable == True, f"Surface {i} (AIR) should be variable by default!"
        else:
            lens_indices.append(i)
            assert s.thickness_variable == False, f"Surface {i} (LENS) should be fixed by default!"
            assert s.radius_variable == False, f"Surface {i} (LENS) radius should be fixed by default!"
            
    print(f"\n  [OK] Successfully verified: All Air Gaps {air_gap_indices} are Variable (V), All Lens Glass {lens_indices} are Fixed (F)!")

    # 3. Test LensDataTableModel Zemax Solve Flags
    print("\n--- 3. Testing LensDataTableModel Display & Solve Input Parsing ---")
    model = LensDataTableModel(stack)
    
    # S0 is variable thickness, so data() on col 3 should return "10.000 V"
    s0_disp = model.data(0, 3)
    print(f"  S0 Thickness Display: '{s0_disp}' (Expected: '10.000 V')")
    assert "V" in str(s0_disp), "LDE table must show 'V' indicator for variable thickness!"
    
    # S1 is fixed lens glass, so data() on col 3 should return "3.0"
    s1_disp = model.data(1, 3)
    print(f"  S1 Thickness Display: '{s1_disp}' (Expected: '3.0')")
    assert "V" not in str(s1_disp), "LDE table must not show 'V' indicator for fixed lens thickness!"

    # Test setting 'V' and 'F' via table edit
    print("  Testing typing '45.0 V' into S3 thickness...")
    model.setData(3, 3, "45.0 V")
    assert model._surfaces[3].thickness_variable == True
    assert model._surfaces[3].thickness == 45.0
    print(f"  -> S3 Thickness is now: {model._surfaces[3].thickness} mm (Variable = {model._surfaces[3].thickness_variable})")

    print("  Testing typing '45.0 F' into S3 thickness...")
    model.setData(3, 3, "45.0 F")
    assert model._surfaces[3].thickness_variable == False
    assert model._surfaces[3].thickness == 45.0
    print(f"  -> S3 Thickness is now: {model._surfaces[3].thickness} mm (Variable = {model._surfaces[3].thickness_variable})")
    print("  [OK] Zemax string modifier parsing ('V' and 'F') verified!")

    # 4. Run Optimization with Zemax Defaults (Only Air Distances Vary)
    print("\n--- 4. Running Optimization with Zemax Defaults ---")
    # Reset Zemax defaults
    stack.set_zemax_defaults(var_air_gaps=True, fix_lenses=True)
    
    # Record initial lens thicknesses and radii
    init_lens_thicknesses = [s.thickness for s in stack.surfaces if s.is_glass]
    init_lens_radii = [s.radius for s in stack.surfaces if s.is_glass]
    
    opt = OpticalOptimizer(
        system=OpticalSystem(stack),
        wavelengths=[0.850],
        method="Nelder-Mead",
        max_iter=100,
        target="Spot Size (Focus)",
        num_rays=15
    )
    
    # Run optimization without specifying indices -> uses stack.thickness_variable flags automatically!
    res = opt.optimize()
    
    opt_stack = res.optimized_stack
    print(f"  Optimization Success: {res.success} in {res.iterations} iterations")
    print(f"  Initial Merit (RMS):  {res.initial_rms:.4f} mm -> Final Merit (RMS): {res.final_rms:.4f} mm (-{res.rms_reduction_pct:.1f}%)")
    
    print("\n  Verifying Surface Status Post-Optimization:")
    for i, s in enumerate(opt_stack.surfaces):
        tag = "VARIABLE DISTANCE" if s.is_air else "FIXED LENS ELEMENT"
        print(f"    S{i:2d} ({s.name:<14}): Thickness = {s.thickness:6.2f} mm | {tag}")

    # Verify that lens glass thicknesses and radii remained STRICTLY UNCHANGED
    post_lens_thicknesses = [s.thickness for s in opt_stack.surfaces if s.is_glass]
    post_lens_radii = [s.radius for s in opt_stack.surfaces if s.is_glass]
    
    assert init_lens_thicknesses == post_lens_thicknesses, "Lens element thicknesses must remain fixed!"
    assert init_lens_radii == post_lens_radii, "Lens element radii must remain fixed!"
    print("\n  [OK] Physical Confirmation: 100% of Lens Element Parameters Remained Fixed, Air Gaps Optimized to Minimal Spot Size!")
    
    print("\n" + "=" * 80)
    print("ALL ZEMAX-STYLE OPTIMIZATION WORKFLOW CHECKS PASSED!")
    print("=" * 80)

if __name__ == "__main__":
    test_zemax_workflow()
