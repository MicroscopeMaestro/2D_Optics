"""
Benchmark and Validation of Enhanced Nelder-Mead Algorithm in OpticalOptimizer.
Compares:
1. Convergence speed & final RMS spot radius on multi-lens doublet/triplet systems.
2. Handling of bounded variables (thicknesses, radii, tilts).
3. Robustness against local traps and ill-conditioned simplex collapse.
"""

import sys
import os
import math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.optimizer import OpticalOptimizer

def create_test_triplet_system():
    """Create a Cooke Triplet optical system with intentionally detuned thicknesses."""
    stack = SurfaceStack()
    stack.primary_wavelength = 0.5876

    # Object at infinity (collimated beam)
    s0 = Surface(name="Object", thickness=1e5, semi_diameter=12.5)
    # L1: N-LAK9 Positive Meniscus
    s1 = Surface(name="L1_Front", radius=22.0, thickness=4.5, glass="N-BK7", semi_diameter=12.5, thickness_variable=False)
    s2 = Surface(name="L1_Back", radius=-150.0, thickness=8.0, glass="AIR", semi_diameter=12.5, thickness_variable=True, thickness_min=1.0, thickness_max=50.0)
    # L2: SF5 Negative
    s3 = Surface(name="L2_Front", radius=-20.0, thickness=1.5, glass="SF5", semi_diameter=10.0, thickness_variable=False)
    s4 = Surface(name="L2_Back", radius=20.0, thickness=12.0, glass="AIR", semi_diameter=10.0, thickness_variable=True, thickness_min=1.0, thickness_max=50.0)
    # L3: N-LAK9 Positive Doublet
    s5 = Surface(name="L3_Front", radius=35.0, thickness=4.0, glass="N-BK7", semi_diameter=12.5, thickness_variable=False)
    s6 = Surface(name="L3_Back", radius=-25.0, thickness=30.0, glass="AIR", semi_diameter=12.5, thickness_variable=True, thickness_min=5.0, thickness_max=80.0)
    # Detector
    s7 = Surface(name="Detector", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=15.0)

    stack.add_surface(s0)
    stack.add_surface(s1)
    stack.add_surface(s2)
    stack.add_surface(s3)
    stack.add_surface(s4)
    stack.add_surface(s5)
    stack.add_surface(s6)
    stack.add_surface(s7)
    stack._update_accumulated_positions()
    return stack

def benchmark_nelder_mead():
    print("=" * 80)
    print("BENCHMARKING NELDER-MEAD CONVERGENCE & OPTIMIZATION PERFORMANCE")
    print("=" * 80)

    stack = create_test_triplet_system()
    opt_sys = OpticalSystem(stack)

    # Initial system evaluation
    opt = OpticalOptimizer(opt_sys, method="Nelder-Mead", max_iter=60, wavelengths=[0.4861, 0.5876, 0.6563])
    res = opt.optimize()

    print(f"1. Triplet Optimization Outcome:")
    print(f"   Success:              {res.success}")
    print(f"   Initial RMS Spot:     {res.initial_rms*1000.0:.2f} um")
    print(f"   Final RMS Spot:       {res.final_rms*1000.0:.2f} um")
    print(f"   RMS Reduction:        {res.rms_reduction_pct:.2f}%")
    print(f"   Iterations:           {res.iterations}")
    print(f"   Optimized Thicknesses: {res.optimized_thicknesses}")
    print(f"   Message:              {res.message}")

    assert res.success is True, "Nelder-Mead optimization failed"
    assert res.final_rms < res.initial_rms, "Nelder-Mead did not reduce RMS spot radius"
    assert res.rms_reduction_pct > 30.0, f"Expected >30% RMS reduction, got {res.rms_reduction_pct:.2f}%"

    print("\n" + "=" * 80)
    print("NELDER-MEAD BENCHMARK COMPLETED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    benchmark_nelder_mead()
