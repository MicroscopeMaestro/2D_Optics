"""
Comprehensive Verification & Test Suite for Optical Optimizer Tools in 2D_Optics.
Tests all targets, algorithms, variable types, constraints, and multi-wavelength scenarios.
"""
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(line_buffering=True)

from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from optics_app.engine.optimizer import OpticalOptimizer, OptimizationResult

def test_1_spot_size_focus_thickness():
    print("\n" + "=" * 80)
    print("TEST 1: Spot Size (Focus) Optimization - Thickness Tuning (Single Lens)")
    print("=" * 80)
    
    # Plano-convex lens f ~ 50 mm (R1=25.84, R2=inf, BK7, t=5.0 mm)
    # Object at s = -100 mm. Thin lens image distance s' = 100 mm. Thick lens BFL ~ 82 mm.
    # Start detector intentionally out of focus at thickness = 20.0 mm
    stack = SurfaceStack()
    stack.add_surface(Surface(name="Object", surface_type=SurfaceType.OBJECT, thickness=100.0, semi_diameter=5.0))
    stack.add_surface(Surface(name="Lens_Front", surface_type=SurfaceType.STANDARD, radius=25.84, thickness=5.0, glass="N-BK7", semi_diameter=12.7))
    stack.add_surface(Surface(name="Lens_Back", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=20.0, glass="AIR", semi_diameter=12.7, thickness_variable=True, thickness_min=5.0, thickness_max=120.0))
    stack.add_surface(Surface(name="Detector", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=10.0))
    
    opt = OpticalOptimizer(
        system=OpticalSystem(stack),
        wavelengths=[0.5876],
        method="Nelder-Mead",
        max_iter=100,
        target="Spot Size (Focus)",
        num_rays=21
    )
    
    res = opt.optimize(variable_indices=[2])
    
    bfl_opt = res.optimized_stack.surfaces[2].thickness
    print(f"  Initial Thickness: 20.000 mm | Initial Merit (RMS): {res.initial_rms:.4f} mm")
    print(f"  Optimized BFL:     {bfl_opt:.4f} mm | Final Merit (RMS):   {res.final_rms:.4f} mm")
    print(f"  RMS Reduction:     {res.rms_reduction_pct:.2f}% | Iterations: {res.iterations}")
    
    passed = 75.0 <= bfl_opt <= 90.0 and res.rms_reduction_pct > 90.0 and res.success
    print(f"  -> TEST 1 STATUS: {'[PASS - VERIFIED]' if passed else '[FAIL]'}")
    return passed

def test_2_collimation_angular_spread():
    print("\n" + "=" * 80)
    print("TEST 2: Angular Spread (Collimation) Optimization from Divergent Point Source")
    print("=" * 80)
    
    # Point source (fiber) -> Collimating Achromat (f=30 mm, Thorlabs AC127-030-B)
    # Start fiber-lens distance intentionally de-collimated at t = 10.0 mm
    stack = SurfaceStack()
    stack.add_surface(Surface(name="Point_Source", surface_type=SurfaceType.OBJECT, thickness=10.0, semi_diameter=0.025, thickness_variable=True, thickness_min=5.0, thickness_max=50.0))
    stack.add_surface(Surface(name="L1_Front", surface_type=SurfaceType.STANDARD, radius=19.1, thickness=2.5, glass="SF5", semi_diameter=6.35))
    stack.add_surface(Surface(name="L1_Mid", surface_type=SurfaceType.STANDARD, radius=-17.8, thickness=1.0, glass="N-BK7", semi_diameter=6.35))
    stack.add_surface(Surface(name="L1_Back", surface_type=SurfaceType.STANDARD, radius=-82.3, thickness=30.0, glass="AIR", semi_diameter=6.35))
    stack.add_surface(Surface(name="Detector", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=10.0))
    
    opt = OpticalOptimizer(
        system=OpticalSystem(stack),
        wavelengths=[0.850],
        method="Nelder-Mead",
        max_iter=100,
        target="Angular Spread (Collimation)",
        num_rays=15,
        target_surface_index=3 # Evaluate collimation right after L1_Back
    )
    
    res = opt.optimize(variable_indices=[0])
    opt_dist = res.optimized_stack.surfaces[0].thickness
    
    print(f"  Initial Source Distance: 10.000 mm | Initial Angular Variance Merit: {res.initial_rms:.4f}")
    print(f"  Optimized Source Dist:   {opt_dist:.4f} mm | Final Angular Variance Merit:   {res.final_rms:.4f}")
    print(f"  Merit Reduction:         {res.rms_reduction_pct:.2f}% | Iterations: {res.iterations}")
    
    passed = 15.0 <= opt_dist <= 20.0 and res.rms_reduction_pct > 90.0 and res.success
    print(f"  -> TEST 2 STATUS: {'[PASS - VERIFIED]' if passed else '[FAIL]'}")
    return passed

def test_3_joint_thickness_and_radius():
    print("\n" + "=" * 80)
    print("TEST 3: Joint Optimization of Thicknesses AND Radii (Shape Bending for Min Spherical Aberration)")
    print("=" * 80)
    
    # Optimize a single lens: bend front radius and adjust back focal length simultaneously
    stack = SurfaceStack()
    stack.add_surface(Surface(name="Object", surface_type=SurfaceType.OBJECT, thickness=100.0, semi_diameter=5.0))
    stack.add_surface(Surface(name="Lens_Front", surface_type=SurfaceType.STANDARD, radius=35.0, thickness=5.0, glass="N-BK7", semi_diameter=12.7, radius_variable=True, radius_min=15.0, radius_max=80.0))
    stack.add_surface(Surface(name="Lens_Back", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=30.0, glass="AIR", semi_diameter=12.7, thickness_variable=True, thickness_min=10.0, thickness_max=120.0))
    stack.add_surface(Surface(name="Detector", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=10.0))
    
    opt = OpticalOptimizer(
        system=OpticalSystem(stack),
        wavelengths=[0.5876],
        method="Nelder-Mead",
        max_iter=150,
        target="Spot Size (Focus)",
        num_rays=25
    )
    
    # Optimize radius of surface 1 and thickness of surface 2
    res = opt.optimize(variable_indices={"thickness": [2], "radius": [1]})
    
    opt_r1 = res.optimized_stack.surfaces[1].radius
    opt_t2 = res.optimized_stack.surfaces[2].thickness
    
    print(f"  Initial R1: 35.000 mm, T2: 30.000 mm | Initial RMS: {res.initial_rms:.4f} mm")
    print(f"  Optimized R1: {opt_r1:.4f} mm, T2: {opt_t2:.4f} mm | Final RMS: {res.final_rms:.4f} mm")
    print(f"  RMS Reduction: {res.rms_reduction_pct:.2f}% | Iterations: {res.iterations}")
    
    passed = res.rms_reduction_pct > 90.0 and res.final_rms < 0.50 and res.success
    print(f"  -> TEST 3 STATUS: {'[PASS - VERIFIED]' if passed else '[FAIL]'}")
    return passed

def test_4_multi_wavelength_achromatic_focus():
    print("\n" + "=" * 80)
    print("TEST 4: Multi-Wavelength Achromatic Optimization (Broadband 486 - 656 nm)")
    print("=" * 80)
    
    # Doublet: F-d-C lines (0.4861, 0.5876, 0.6563 um)
    stack = SurfaceStack()
    stack.add_surface(Surface(name="Object", surface_type=SurfaceType.OBJECT, thickness=100.0, semi_diameter=5.0))
    stack.add_surface(Surface(name="L1_Front", surface_type=SurfaceType.STANDARD, radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
    stack.add_surface(Surface(name="L1_Mid", surface_type=SurfaceType.STANDARD, radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
    stack.add_surface(Surface(name="L1_Back", surface_type=SurfaceType.STANDARD, radius=-115.0, thickness=25.0, glass="AIR", semi_diameter=12.7, thickness_variable=True, thickness_min=10.0, thickness_max=200.0))
    stack.add_surface(Surface(name="Detector", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=10.0))
    
    wls = [0.4861, 0.5876, 0.6563]
    opt = OpticalOptimizer(
        system=OpticalSystem(stack),
        wavelengths=wls,
        method="Nelder-Mead",
        max_iter=100,
        target="Spot Size (Focus)",
        num_rays=15
    )
    
    res = opt.optimize(variable_indices=[3])
    opt_bfl = res.optimized_stack.surfaces[3].thickness
    
    print(f"  Wavelengths: {wls} um")
    print(f"  Initial BFL: 25.000 mm | Initial Multi-WL RMS: {res.initial_rms:.4f} mm")
    print(f"  Optimized BFL: {opt_bfl:.4f} mm | Final Multi-WL RMS:   {res.final_rms:.4f} mm")
    print(f"  RMS Reduction: {res.rms_reduction_pct:.2f}% | Iterations: {res.iterations}")
    
    passed = 170.0 <= opt_bfl <= 200.0 and res.rms_reduction_pct > 90.0 and res.success
    print(f"  -> TEST 4 STATUS: {'[PASS - VERIFIED]' if passed else '[FAIL]'}")
    return passed

def test_5_boundary_constraint_enforcement():
    print("\n" + "=" * 80)
    print("TEST 5: Boundary Constraints (thickness_min / thickness_max) Enforcement")
    print("=" * 80)
    
    # Test strict constraint limits: force optimizer with a narrow artificial bound
    stack = SurfaceStack()
    stack.add_surface(Surface(name="Object", surface_type=SurfaceType.OBJECT, thickness=100.0, semi_diameter=5.0))
    stack.add_surface(Surface(name="Lens", surface_type=SurfaceType.STANDARD, radius=25.84, thickness=5.0, glass="N-BK7", semi_diameter=12.7))
    # True unconstrained optimum is ~82 mm, but we restrict thickness to [10.0, 35.0] mm
    stack.add_surface(Surface(name="Back", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=20.0, glass="AIR", semi_diameter=12.7, thickness_variable=True, thickness_min=10.0, thickness_max=35.0))
    stack.add_surface(Surface(name="Detector", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=10.0))
    
    opt = OpticalOptimizer(
        system=OpticalSystem(stack),
        wavelengths=[0.5876],
        method="Nelder-Mead",
        max_iter=100,
        target="Spot Size (Focus)",
        num_rays=15
    )
    
    res = opt.optimize(variable_indices=[2])
    opt_t = res.optimized_stack.surfaces[2].thickness
    
    print(f"  Bounds: [10.0, 35.0] mm (True unconstrained optimum would be ~82.0 mm)")
    print(f"  Result Thickness: {opt_t:.4f} mm")
    
    # Must be pinned at upper boundary 35.0 mm without exceeding it
    passed = 34.5 <= opt_t <= 35.001 and res.success
    print(f"  -> TEST 5 STATUS: {'[PASS - VERIFIED]' if passed else '[FAIL]'}")
    return passed

def test_6_telecentricity():
    print("\n" + "=" * 80)
    print("TEST 6: Telecentricity Optimization (Exit Rays Parallel to Optical Axis)")
    print("=" * 80)
    
    # Point source at focal plane -> Collimated Telecentric Exit
    stack = SurfaceStack()
    stack.add_surface(Surface(name="Source", surface_type=SurfaceType.OBJECT, thickness=15.0, semi_diameter=0.025, thickness_variable=True, thickness_min=5.0, thickness_max=60.0))
    stack.add_surface(Surface(name="Lens_Front", surface_type=SurfaceType.STANDARD, radius=25.84, thickness=5.0, glass="N-BK7", semi_diameter=12.7))
    stack.add_surface(Surface(name="Lens_Back", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=20.0, glass="AIR", semi_diameter=12.7))
    stack.add_surface(Surface(name="Detector", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=10.0))
    
    opt = OpticalOptimizer(
        system=OpticalSystem(stack),
        wavelengths=[0.5876],
        method="Nelder-Mead",
        max_iter=80,
        target="Telecentricity",
        num_rays=15,
        target_surface_index=2
    )
    
    res = opt.optimize(variable_indices=[0])
    opt_d = res.optimized_stack.surfaces[0].thickness
    
    print(f"  Initial Source Distance: 15.000 mm | Initial Telecentricity Deviation: {res.initial_rms:.4f}")
    print(f"  Optimized Source Dist:   {opt_d:.4f} mm | Final Telecentricity Deviation:   {res.final_rms:.4f}")
    print(f"  Merit Reduction:         {res.rms_reduction_pct:.2f}% | Iterations: {res.iterations}")
    
    passed = 38.0 <= opt_d <= 45.0 and res.rms_reduction_pct > 90.0 and res.success
    print(f"  -> TEST 6 STATUS: {'[PASS - VERIFIED]' if passed else '[FAIL]'}")
    return passed

def run_all_tests():
    t_start = time.time()
    print("=" * 80)
    print("STARTING 2D_OPTICS OPTIMIZER ENGINE VERIFICATION SUITE")
    print("=" * 80)
    
    results = [
        ("Test 1: Spot Size (Focus) Thickness", test_1_spot_size_focus_thickness()),
        ("Test 2: Angular Spread (Collimation)", test_2_collimation_angular_spread()),
        ("Test 3: Joint Thickness & Radius Optimization", test_3_joint_thickness_and_radius()),
        ("Test 4: Multi-Wavelength Achromatization", test_4_multi_wavelength_achromatic_focus()),
        ("Test 5: Boundary Constraints Enforcement", test_5_boundary_constraint_enforcement()),
        ("Test 6: Telecentricity Optimization", test_6_telecentricity())
    ]
    
    print("\n" + "=" * 80)
    print("SUMMARY OF OPTIMIZATION VERIFICATION RESULTS:")
    print("=" * 80)
    all_passed = True
    for name, ok in results:
        print(f"  {name:<50}: {'PASS [OK]' if ok else 'FAIL [X]'}")
        if not ok: all_passed = False
        
    print(f"\n  Total Execution Time: {time.time() - t_start:.2f} seconds")
    print(f"  Overall Suite Status: {'ALL 6 OPTIMIZER TESTS PASSED PERFECTLY!' if all_passed else 'SOME TESTS FAILED'}")
    print("=" * 80)
    return all_passed

if __name__ == "__main__":
    success = run_all_tests()
    if not success:
        sys.exit(1)
