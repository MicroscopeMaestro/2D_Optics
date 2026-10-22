import sys
import os
import math
import numpy as np

# Ensure the v2 package is in the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from v2.engine.surface import Surface, SurfaceType
from v2.engine.lde import SurfaceStack
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray
from v2.engine.optimizer import OpticalOptimizer

def create_pupil_relay():
    stack = SurfaceStack()
    
    # 0. Source representing the focused beam from M2 on the DMD plane
    # The beam has NA ~ 0.12 (from M2)
    stack.add_surface(Surface(
        name="DMD_Surface", surface_type=SurfaceType.OBJECT,
        radius=0.0, thickness=15.0, glass="AIR", semi_diameter=4.92,
        thickness_variable=True, thickness_min=5.0, thickness_max=40.0
    ))
    
    # 1. Field Lens: AC254-030-C (f=30mm)
    stack.add_surface(Surface(
        name="Field_AC254-030-C_Front", surface_type=SurfaceType.STANDARD,
        radius=19.9, thickness=5.0, glass="N-BAF10", semi_diameter=12.7
    ))
    stack.add_surface(Surface(
        name="Field_AC254-030-C_Mid", surface_type=SurfaceType.STANDARD,
        radius=-15.3, thickness=2.0, glass="N-SF6", semi_diameter=12.7
    ))
    stack.add_surface(Surface(
        name="Field_AC254-030-C_Back", surface_type=SurfaceType.STANDARD,
        radius=-60.0, thickness=20.0, glass="AIR", semi_diameter=12.7,
        thickness_variable=True, thickness_min=5.0, thickness_max=80.0
    ))
    
    # 2. Condenser 1: AC127-019-C (f=19mm)
    stack.add_surface(Surface(
        name="Cond1_AC127-019-C_Front", surface_type=SurfaceType.STANDARD,
        radius=11.6, thickness=4.0, glass="N-BAF10", semi_diameter=6.35
    ))
    stack.add_surface(Surface(
        name="Cond1_AC127-019-C_Mid", surface_type=SurfaceType.STANDARD,
        radius=-10.0, thickness=1.5, glass="N-SF6", semi_diameter=6.35
    ))
    stack.add_surface(Surface(
        name="Cond1_AC127-019-C_Back", surface_type=SurfaceType.STANDARD,
        radius=-40.0, thickness=1.0, glass="AIR", semi_diameter=6.35,
        thickness_variable=True, thickness_min=0.5, thickness_max=15.0
    ))
    
    # 3. Condenser 2: AC127-019-C (Reversed)
    stack.add_surface(Surface(
        name="Cond2_AC127-019-C_Front", surface_type=SurfaceType.STANDARD,
        radius=40.0, thickness=1.5, glass="N-SF6", semi_diameter=6.35
    ))
    stack.add_surface(Surface(
        name="Cond2_AC127-019-C_Mid", surface_type=SurfaceType.STANDARD,
        radius=10.0, thickness=4.0, glass="N-BAF10", semi_diameter=6.35
    ))
    stack.add_surface(Surface(
        name="Cond2_AC127-019-C_Back", surface_type=SurfaceType.STANDARD,
        radius=-11.6, thickness=10.0, glass="AIR", semi_diameter=6.35,
        thickness_variable=True, thickness_min=2.0, thickness_max=30.0
    ))
    
    # 4. Detector
    stack.add_surface(Surface(
        name="Detector_0.8mm", surface_type=SurfaceType.DETECTOR,
        radius=0.0, thickness=0.0, glass="AIR", semi_diameter=0.4
    ))
    
    return OpticalSystem(stack)

if __name__ == "__main__":
    print("Initializing V2 Optical Optimizer...")
    system = create_pupil_relay()
    
    # We trace 3 wavelengths representing the fingerprint region edges and center
    wavelengths = [1.0, 1.064, 1.1] 
    
    optimizer = OpticalOptimizer(system, wavelengths=wavelengths)
    optimizer.method = 'Nelder-Mead'
    optimizer.max_iter = 500
    
    print("Running optimization (enforcing DMD bounds & distances)...")
    result = optimizer.optimize()
    
    if result.success or result.rms_reduction_pct > 0:
        print("\n--- OPTIMIZATION SUCCESS ---")
        print(f"Final Spot RMS: {result.final_rms:.4f} mm")
        print(f"Throughput Constraint / Penalty: {result.penalty_value:.1f}")
        print("\n--- OPTIMAL LENS SPACINGS ---")
        
        stack = result.optimized_stack
        print(f"DMD to Field Lens:      {stack.surfaces[0].thickness:.2f} mm")
        print(f"Field Lens to Cond 1:   {stack.surfaces[3].thickness:.2f} mm")
        print(f"Cond 1 to Cond 2:       {stack.surfaces[6].thickness:.2f} mm")
        print(f"Cond 2 to Detector:     {stack.surfaces[9].thickness:.2f} mm")
    else:
        print("\nOptimization struggled to converge entirely.")
        print(result.message)
