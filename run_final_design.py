import os
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.optimizer import OpticalOptimizer

def main():
    json_path = "fiber_spectrometer_setup.json"
    print(f"Loading {json_path}...")
    res = SurfaceStack.load_from_json(json_path)
    stack = res['stack']
    
    # 785nm Raman wavelengths (cm-1: 400, 1900, 3400)
    wls = [0.8104, 0.9406, 1.0708]

    # Stage 1: Collimation (Front-end: Fiber to L1)
    # The variable index for Fiber->L1 air gap is 0.
    print("\n--- STAGE 1: Collimation ---")
    opt1 = OpticalOptimizer(
        system=OpticalSystem(stack),
        wavelengths=[0.9406], # Optimize middle wavelength for collimation
        method="Nelder-Mead",
        max_iter=300,
        target="Angular Spread (Collimation)",
        num_rays=9
    )
    res1 = opt1.optimize(variable_indices=[0])
    stack = res1.optimized_stack
    print(f"Stage 1 Collimation success: {res1.success}")
    print(f"New Fiber-L1 distance: {stack.surfaces[0].thickness:.4f}")

    # Stage 2: Focus (Back-end: Condenser to Detector)
    # The variable index for Condenser->Detector air gap is 13.
    print("\n--- STAGE 2: Focus ---")
    opt2 = OpticalOptimizer(
        system=OpticalSystem(stack),
        wavelengths=[0.9406], # Middle wl focus
        method="Nelder-Mead",
        max_iter=300,
        target="Spot Size (Focus)",
        num_rays=9
    )
    res2 = opt2.optimize(variable_indices=[13])
    stack = res2.optimized_stack
    print(f"Stage 2 Focus success: {res2.success}")
    print(f"New Condenser-Detector distance: {stack.surfaces[13].thickness:.4f}")

    # Stage 3: Whole System Balancing
    print("\n--- STAGE 3: System Optimization ---")
    # All variable air gaps:
    var_indices = [i for i, s in enumerate(stack.surfaces) if getattr(s, 'thickness_variable', False)]
    opt3 = OpticalOptimizer(
        system=OpticalSystem(stack),
        wavelengths=wls,
        method="Nelder-Mead",
        max_iter=500,
        target="Spot Size (Focus)",
        num_rays=21
    )
    res3 = opt3.optimize(variable_indices=var_indices)
    stack = res3.optimized_stack
    print(f"Stage 3 Optimization success: {res3.success}")
    for idx in var_indices:
        print(f"  Surface {idx} thickness: {stack.surfaces[idx].thickness:.4f}")

    stack.save_to_json(json_path, wls)
    print(f"\nSaved fully optimized optical design to {json_path}")

if __name__ == "__main__":
    main()
