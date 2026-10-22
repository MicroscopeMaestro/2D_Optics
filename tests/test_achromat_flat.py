"""
Unit tests for flat-interface achromatic doublet synthesis and optimization.
Verifies analytical thin-lens and thick-lens solutions with N-BAF10 and SF10.
"""
import pytest
import numpy as np

from optics_app.engine.achromat_designer import (
    AchromatDesignSpec,
    optimize_achromatic_doublet,
    solve_thin_achromat,
)
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.optimizer import OpticalOptimizer, compute_system_efl


def test_solve_thin_achromat_flat_interface():
    """Verify analytical thin achromat solution for N-BAF10 + SF10 with flat interface."""
    r1, r2, r3, phi1, phi2 = solve_thin_achromat(
        target_efl=50.0,
        crown_glass="N-BAF10",
        flint_glass="SF10",
        flat_interface=True
    )
    # Check radii
    assert 12.0 < r1 < 14.5, f"Expected R1 ~ 13.2 mm, got {r1}"
    assert r2 >= 1e7, f"Expected flat interface (r2 >= 1e7), got {r2}"
    assert 20.0 < r3 < 26.0, f"Expected R3 ~ 23.7 mm, got {r3}"
    assert phi1 > 0, "Crown power must be positive"
    assert phi2 < 0, "Flint power must be negative"


def test_optimize_achromatic_doublet_flat_interface():
    """Verify thick doublet optimization with flat cemented interface constraint."""
    spec = AchromatDesignSpec(
        target_efl=50.0,
        diameter=10.0,
        crown_glass="N-BAF10",
        flint_glass="SF10",
        doublet_type="cemented",
        flat_interface=True,
        t1=7.695,
        t2=6.110
    )
    res = optimize_achromatic_doublet(spec)
    assert abs(res.efl - 50.0) < 1.0, f"Target EFL 50 mm not met: {res.efl}"
    surfs = res.to_surfaces()
    assert len(surfs) == 3
    # S1 Crown front
    assert 12.0 < surfs[0].radius < 17.0
    # S2 Cemented flat contact
    assert surfs[1].radius == 0.0, f"Interface must be 0.0 (flat plane), got {surfs[1].radius}"
    # S3 Flint rear
    assert 13.0 < surfs[2].radius < 25.0
    assert surfs[2].radius > 0, "Flint rear must be positive for diverging meniscus element"


def test_optimizer_achromatic_focus_target():
    """Verify OpticalOptimizer with target='Achromatic Focus' converges to true achromat."""
    stack = SurfaceStack()
    stack.add_surface(Surface(surface_type=SurfaceType.OBJECT, thickness=100.0, semi_diameter=5.0))
    stack.add_surface(Surface(surface_type=SurfaceType.STOP, thickness=5.0, semi_diameter=5.0))
    # Start with split singlet (R1 > 0, R3 < 0)
    stack.add_surface(Surface(radius=62.683, thickness=7.695, glass="N_BAF10", semi_diameter=5.0))
    stack.add_surface(Surface(radius=0.0, thickness=6.110, glass="SF10", semi_diameter=5.0))
    stack.add_surface(Surface(radius=-74.414, thickness=29.870, glass="AIR", semi_diameter=5.0))
    stack.add_surface(Surface(surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=5.0))

    opt = OpticalOptimizer(
        OpticalSystem(stack),
        wavelengths=[0.48613, 0.58756, 0.65627],
        target="Achromatic Focus (Min Spot + Min Color)",
        target_efl=50.0,
        num_rays=21,
        entrance_pupil_radius=5.0,
        max_iter=100
    )
    res = opt.optimize(variable_indices={"radius": [2, 4], "thickness": [4]})
    assert res.success
    r1_opt = res.optimized_radii[2]
    r3_opt = res.optimized_radii[4]
    # R1 must be strong positive (~13-17 mm) and R3 must be positive (~14-25 mm)
    assert 12.0 < r1_opt < 20.0, f"Expected R1 in [12, 20], got {r1_opt}"
    assert 12.0 < r3_opt < 28.0, f"Expected R3 in [12, 28], got {r3_opt}"
