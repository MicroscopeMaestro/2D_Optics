"""
Comprehensive Unit Tests for Zemax-Style Optimization Algorithms:
1. Curvature-Space Parameterization (c = 1/R).
2. Damped Least Squares (DLS / Levenberg-Marquardt / TRF).
3. Zemax Seidel Aberration Reduction (S_I .. S_V, C_L, C_T).
4. Wavefront Error Optimization (RMS Optical Path Difference).
5. Zemax Hammer Optimizer (Global Stochastic Perturbation + DLS Polish).
6. Zemax Global Search (Multi-Start with Curvature Permutations).
7. Monte Carlo Optimization (Stochastic Sampling + DLS Refinement).
"""

import pytest
import math
import numpy as np

from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.optimizer import (
    OpticalOptimizer,
    OptimizationResult,
    compute_system_efl,
    radius_to_curvature,
    curvature_to_radius
)
from optics_app.engine.merit_function import MeritFunction, OperandType
from optics_app.analysis.aberration import calculate_seidel_coefficients


def test_curvature_space_mapping():
    """Verify smooth mapping between radius of curvature and curvature space."""
    # Flat surface
    assert radius_to_curvature(0.0) == 0.0
    assert radius_to_curvature(float('inf')) == 0.0
    assert curvature_to_radius(0.0) == 0.0

    # Convex surface
    assert abs(radius_to_curvature(50.0) - 0.02) < 1e-9
    assert abs(curvature_to_radius(0.02) - 50.0) < 1e-9

    # Concave surface
    assert abs(radius_to_curvature(-25.0) - (-0.04)) < 1e-9
    assert abs(curvature_to_radius(-0.04) - (-25.0)) < 1e-9

    # Smooth transition near zero
    assert curvature_to_radius(1e-7) == 0.0
    assert radius_to_curvature(1e8) == 0.0


def test_dls_converges_on_singlet():
    """Verify that Damped Least Squares (DLS) converges on a singlet lens."""
    stack = SurfaceStack()
    stack.add_surface(Surface(name="OBJ", thickness=100.0))
    stack.add_surface(Surface(name="L1_Front", radius=50.0, thickness=5.0, glass="N-BK7", semi_diameter=12.5))
    stack.add_surface(Surface(name="L1_Back", radius=0.0, thickness=30.0, glass="AIR", semi_diameter=12.5))
    stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=12.5))

    sys = OpticalSystem(stack)
    optimizer = OpticalOptimizer(
        sys,
        method="Damped Least Squares (DLS)",
        max_iter=30,
        num_rays=15,
        entrance_pupil_radius=5.0
    )
    result = optimizer.optimize(variable_indices={"thickness": [2]})
    assert result.success
    assert result.iterations > 0
    assert result.final_rms < result.initial_rms
    assert 2 in result.optimized_thicknesses
    # Back focal distance for R1=50, flat rear, N-BK7 is ~93.4 mm
    assert 85.0 < result.optimized_thicknesses[2] < 105.0


def test_dls_multi_surface_cooke_triplet():
    """Verify DLS convergence on 6-surface Cooke Triplet."""
    stack = SurfaceStack()
    stack.add_surface(Surface(name="OBJ", thickness=100.0))
    # Crown 1
    stack.add_surface(Surface(name="S1", radius=22.0, thickness=3.2, glass="N-SK16", semi_diameter=8.0))
    stack.add_surface(Surface(name="S2", radius=-150.0, thickness=5.5, glass="AIR", semi_diameter=8.0))
    # Flint 2
    stack.add_surface(Surface(name="S3", radius=-20.0, thickness=1.5, glass="SF2", semi_diameter=6.0))
    stack.add_surface(Surface(name="S4", radius=20.0, thickness=4.8, glass="AIR", semi_diameter=6.0))
    # Crown 3
    stack.add_surface(Surface(name="S5", radius=80.0, thickness=2.8, glass="N-SK16", semi_diameter=7.0))
    stack.add_surface(Surface(name="S6", radius=-24.0, thickness=40.0, glass="AIR", semi_diameter=7.0))
    stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=10.0))

    sys = OpticalSystem(stack)
    optimizer = OpticalOptimizer(
        sys,
        method="DLS",
        max_iter=30,
        num_rays=15,
        entrance_pupil_radius=5.0
    )
    # Optimize thicknesses and back focal distance
    result = optimizer.optimize(variable_indices={"thickness": [2, 4, 6], "radius": [1, 3, 5]})
    assert result.success
    assert result.final_rms < result.initial_rms
    assert result.rms_reduction_pct > 90.0, f"DLS must achieve massive RMS reduction, got {result.rms_reduction_pct}%"


def test_seidel_aberration_reduction_target():
    """Verify optimization with target='Aberration Reduction (Seidel S1-S5 + Color)'."""
    stack = SurfaceStack()
    stack.add_surface(Surface(name="OBJ", thickness=100.0))
    stack.add_surface(Surface(name="L1_Front", radius=40.0, thickness=6.0, glass="N-BK7", semi_diameter=10.0))
    stack.add_surface(Surface(name="L1_Back", radius=-40.0, thickness=45.0, glass="AIR", semi_diameter=10.0))
    stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=10.0))

    sys = OpticalSystem(stack)
    seidel_init = calculate_seidel_coefficients(sys)
    init_s1 = abs(seidel_init.S_I)

    optimizer = OpticalOptimizer(
        sys,
        method="DLS",
        target="Aberration Reduction (Seidel S1-S5 + Color)",
        max_iter=40,
        num_rays=15,
        entrance_pupil_radius=5.0
    )
    result = optimizer.optimize(variable_indices={"radius": [1, 2], "thickness": [2]})
    assert result.success

    seidel_final = calculate_seidel_coefficients(OpticalSystem(result.optimized_stack))
    final_s1 = abs(seidel_final.S_I)
    assert final_s1 < init_s1, f"Spherical aberration must be reduced: init={init_s1}, final={final_s1}"


def test_wavefront_error_rms_opd_target():
    """Verify optimization with target='Wavefront Error (RMS OPD)'."""
    stack = SurfaceStack()
    stack.add_surface(Surface(name="OBJ", thickness=100.0))
    stack.add_surface(Surface(name="L1_Front", radius=45.0, thickness=5.0, glass="N-BK7", semi_diameter=10.0))
    stack.add_surface(Surface(name="L1_Back", radius=-100.0, thickness=40.0, glass="AIR", semi_diameter=10.0))
    stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=10.0))

    sys = OpticalSystem(stack)
    optimizer = OpticalOptimizer(
        sys,
        method="DLS",
        target="Wavefront Error (RMS OPD)",
        max_iter=30,
        num_rays=15,
        entrance_pupil_radius=5.0
    )
    result = optimizer.optimize(variable_indices={"radius": [1]})
    assert result.success
    assert result.final_rms < result.initial_rms
    # RMS OPD should be tiny (in waves)
    assert result.final_rms < 1.0, f"RMS Wavefront Error expected < 1 wave, got {result.final_rms}"
    assert 1 in result.optimized_radii


def test_zemax_hammer_optimizer():
    """Verify Zemax Hammer Optimizer escapes local minima via stochastic perturbations."""
    stack = SurfaceStack()
    stack.add_surface(Surface(name="OBJ", thickness=100.0))
    # Suboptimal starting point
    stack.add_surface(Surface(name="S1", radius=30.0, thickness=5.0, glass="N-BK7", semi_diameter=10.0))
    stack.add_surface(Surface(name="S2", radius=-30.0, thickness=20.0, glass="AIR", semi_diameter=10.0))
    stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=10.0))

    sys = OpticalSystem(stack)
    optimizer = OpticalOptimizer(
        sys,
        method="Hammer Optimizer (Global)",
        max_iter=40,
        num_rays=15,
        entrance_pupil_radius=5.0
    )
    result = optimizer.optimize(variable_indices={"thickness": [2], "radius": [1, 2]})
    assert result.success
    assert result.final_rms < result.initial_rms
    assert result.iterations > 0


def test_zemax_global_search_multi_start():
    """Verify Global Search tests curvature sign permutations and returns optimal candidate."""
    stack = SurfaceStack()
    stack.add_surface(Surface(name="OBJ", thickness=100.0))
    # Plano-convex inverted or standard
    stack.add_surface(Surface(name="S1", radius=-40.0, thickness=5.0, glass="N-BK7", semi_diameter=10.0))
    stack.add_surface(Surface(name="S2", radius=0.0, thickness=30.0, glass="AIR", semi_diameter=10.0))
    stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=10.0))

    sys = OpticalSystem(stack)
    optimizer = OpticalOptimizer(
        sys,
        method="Global Search (Multi-Start)",
        max_iter=30,
        num_rays=15,
        entrance_pupil_radius=5.0
    )
    result = optimizer.optimize(variable_indices={"radius": [1], "thickness": [2]})
    assert result.success
    # Global search should flip curvature to positive R1 > 0 for standard collimated focusing
    assert result.optimized_radii[1] > 0, f"Expected positive R1, got {result.optimized_radii[1]}"


def test_monte_carlo_optimization():
    """Verify Monte Carlo Optimization explores design space and refines with DLS."""
    stack = SurfaceStack()
    stack.add_surface(Surface(name="OBJ", thickness=100.0))
    stack.add_surface(Surface(name="S1", radius=50.0, thickness=5.0, glass="N-BK7", semi_diameter=10.0))
    stack.add_surface(Surface(name="S2", radius=-50.0, thickness=30.0, glass="AIR", semi_diameter=10.0))
    stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=10.0))

    sys = OpticalSystem(stack)
    optimizer = OpticalOptimizer(
        sys,
        method="Monte Carlo Optimization",
        max_iter=30,
        num_rays=15,
        entrance_pupil_radius=5.0
    )
    result = optimizer.optimize(variable_indices={"thickness": [2]})
    assert result.success
    assert result.final_rms < result.initial_rms
    assert 2 in result.optimized_thicknesses
    # Optimal focal distance should be ~45-55 mm for biconvex R=50 mm
    assert 35.0 < result.optimized_thicknesses[2] < 65.0


def test_merit_function_with_seidel_and_wavefront_operands():
    """Verify MeritFunction evaluation with SPHA, COMA, and OPDX operands."""
    stack = SurfaceStack()
    stack.add_surface(Surface(name="OBJ", thickness=100.0))
    stack.add_surface(Surface(name="S1", radius=50.0, thickness=5.0, glass="N-BK7", semi_diameter=10.0))
    stack.add_surface(Surface(name="S2", radius=-50.0, thickness=45.0, glass="AIR", semi_diameter=10.0))
    stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=10.0))
    sys = OpticalSystem(stack)

    # 1. Seidel Merit Function
    mf_seidel = MeritFunction.create_seidel_default(weight_s1=2.0, weight_color=1.0)
    from optics_app.engine.ray import Ray
    r_bundle = [Ray(pos=np.array([0.0, y, 0.0]), dir=np.array([0.0, 0.0, 1.0])) for y in [-3.0, 0.0, 3.0]]
    traced = sys.trace_rays(r_bundle)
    val_seidel = mf_seidel.evaluate(sys, traced, wavelengths=[0.5876])
    assert val_seidel > 0.0
    assert any(op.operand_type == OperandType.SEIDEL_S1 for op in mf_seidel.operands)

    # 2. Wavefront Merit Function
    mf_wave = MeritFunction.create_wavefront_default()
    val_wave = mf_wave.evaluate(sys, traced, wavelengths=[0.5876])
    assert val_wave >= 0.0
    assert any(op.operand_type == OperandType.WAVEFRONT_OPD for op in mf_wave.operands)
