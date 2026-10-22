"""
Unit tests for Optical System Effective Focal Length (EFL) constraint,
anti-vignetting merit function, and spot size optimization on detector.
"""
import pytest
import math
import numpy as np

from optics_app.engine.surface import Surface
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.optimizer import OpticalOptimizer, compute_system_efl, resolve_solves_and_pickups
from optics_app.gui.optimizer_dock import OptimizerDockWidget
from optics_app.gui.qt_compat import QT_AVAILABLE


def test_compute_system_efl():
    """Verify paraxial ABCD calculation of system EFL and BFL."""
    # N-BK7 Plano-Convex Lens: R1 = 50.0 mm, R2 = flat, thickness = 5.0 mm, n = 1.5168 @ 0.5876 um
    # f = R / (n - 1) = 50.0 / 0.5168 = 96.75 mm
    s0 = Surface(name="OBJ", radius=0.0, thickness=0.0, glass="AIR")
    s1 = Surface(name="S1", radius=50.0, thickness=5.0, glass="N-BK7")
    s2 = Surface(name="S2", radius=0.0, thickness=93.45, glass="AIR")
    s3 = Surface(name="DET", radius=0.0, thickness=0.0, glass="AIR")

    stack = SurfaceStack()
    stack.surfaces = [s0, s1, s2, s3]

    efl, bfl = compute_system_efl(stack.surfaces, wavelength_um=0.5876)
    assert abs(efl - 96.75) < 0.5
    assert abs(bfl - 93.45) < 0.5


def test_resolve_solves_and_pickups():
    """Verify dynamic solve resolution during optimization: pickups and marginal ray focus."""
    s1 = Surface(name="S1", radius=30.0, thickness=4.0, glass="N-BK7")
    s2 = Surface(name="S2", radius=0.0, thickness=20.0, glass="AIR")
    s2.radius_solve_type = "Pickup"
    s2.radius_pickup_surface = 0
    s2.radius_pickup_scale = -1.0 # Equi-convex

    s2.thickness_solve_type = "MarginalRayFocus"

    stack = SurfaceStack()
    stack.surfaces = [s1, s2]

    resolve_solves_and_pickups(stack, primary_wl=0.5876)
    assert stack.surfaces[1].radius == -30.0
    # Thickness should be solved to positive back focal distance
    assert stack.surfaces[1].thickness > 10.0


def test_anti_vignetting_penalty():
    """Verify that lost or clipped rays incur severe loss penalties."""
    s1 = Surface(name="S1", radius=25.0, thickness=4.0, glass="N-BK7", semi_diameter=12.5)
    s2 = Surface(name="S2", radius=-25.0, thickness=45.0, glass="AIR", semi_diameter=12.5)
    s3 = Surface(name="DET", radius=0.0, thickness=0.0, glass="AIR", semi_diameter=10.0)

    stack = SurfaceStack()
    stack.surfaces = [s1, s2, s3]
    system = OpticalSystem(stack)

    opt = OpticalOptimizer(system=system, wavelengths=[0.5876], num_rays=21, entrance_pupil_radius=5.0)
    
    # Normal case: all rays transmit
    pure1, pen1 = opt._evaluate_merit_and_penalty(np.array([]), {})
    assert pen1 < 100.0

    # Degenerate case with extreme radius clipping rays
    opt_bad = OpticalOptimizer(system=system, wavelengths=[0.5876], num_rays=21, entrance_pupil_radius=5.0)
    # Set extreme thickness causing extreme aberration / clipping
    s1.semi_diameter = 0.5 # Tiny aperture clipping most rays
    pure2, pen2 = opt_bad._evaluate_merit_and_penalty(np.array([]), {})
    # Should incur significant loss penalty
    assert pen2 > 500.0


def test_optical_optimizer_with_target_efl():
    """Verify optimizing an optical doublet to a specified EFL target (50.0 mm)."""
    # Achromatic doublet with initial non-optimized parameters
    s0 = Surface(name="OBJ", radius=0.0, thickness=0.0, glass="AIR")
    s1 = Surface(name="S1", radius=22.0, thickness=4.0, glass="N-BAF10")
    s2 = Surface(name="S2", radius=0.0, thickness=2.0, glass="SF10")
    s3 = Surface(name="S3", radius=-22.0, thickness=45.0, glass="AIR")
    s3.thickness_solve_type = "MarginalRayFocus"
    s4 = Surface(name="DET", radius=0.0, thickness=0.0, glass="AIR")

    stack = SurfaceStack()
    stack.surfaces = [s0, s1, s2, s3, s4]
    system = OpticalSystem(stack)

    target_efl = 50.0
    optimizer = OpticalOptimizer(
        system=system,
        wavelengths=[0.5876],
        method="Nelder-Mead",
        max_iter=60,
        target="Spot Size (Focus)",
        num_rays=15,
        entrance_pupil_radius=5.0,
        target_efl=target_efl,
        efl_weight=100.0
    )

    # Variables: R1 (surface 1), R3 (surface 3)
    var_indices = {
        "radius": [1, 3]
    }

    res = optimizer.optimize(variable_indices=var_indices)
    assert res is not None

    # Calculate final achieved EFL
    final_efl, final_bfl = compute_system_efl(res.optimized_stack.surfaces, wavelength_um=0.5876)
    print(f"Target EFL: {target_efl:.2f} mm | Achieved EFL: {final_efl:.2f} mm")
    
    # Must be within 5% of target EFL
    assert abs(final_efl - target_efl) / target_efl < 0.05
    # Final spot should be small and tightly focused
    assert res.final_rms < 0.25 # < 250 um


def test_optimizer_dock_efl_controls(qapp):
    """Verify OptimizerDockWidget EFL constraint checkboxes, spinbox, and config serialization."""
    dock = OptimizerDockWidget()
    try:
        assert hasattr(dock, 'chk_constrain_efl')
        assert hasattr(dock, 'target_efl_spin')

        # Initially unchecked
        assert dock.chk_constrain_efl.isChecked() is False
        assert dock.target_efl_spin.isEnabled() is False

        # Toggle constraint
        dock.chk_constrain_efl.setChecked(True)
        assert dock.target_efl_spin.isEnabled() is True
        dock.target_efl_spin.setValue(75.5)

        # Config serialization
        cfg = dock.get_config()
        assert cfg["constrain_efl"] is True
        assert cfg["target_efl"] == 75.5

        # Config restoration
        new_cfg = {"constrain_efl": False, "target_efl": 100.0}
        dock.set_config(new_cfg)
        assert dock.chk_constrain_efl.isChecked() is False
        assert dock.target_efl_spin.value() == 100.0
    finally:
        dock.close()


def test_all_flat_achromatic_doublet_bootstrapping():
    """Verify that optimizer automatically escapes completely all-flat plates (R1=0, R2=0, R3=0) to EFL=50mm."""
    s0 = Surface(name="OBJ", radius=0.0, thickness=0.0, glass="AIR")
    s1 = Surface(name="S1", radius=0.0, thickness=4.0, glass="N-BAF10")
    s2 = Surface(name="S2", radius=0.0, thickness=2.0, glass="SF10")  # Flat interface constraint
    s3 = Surface(name="S3", radius=0.0, thickness=45.0, glass="AIR")
    s3.thickness_solve_type = "MarginalRayFocus"
    s4 = Surface(name="DET", radius=0.0, thickness=0.0, glass="AIR")

    stack = SurfaceStack()
    stack.surfaces = [s0, s1, s2, s3, s4]
    sys = OpticalSystem(stack)

    opt = OpticalOptimizer(
        system=sys,
        wavelengths=[0.4861, 0.5876, 0.6563],
        method="Nelder-Mead",
        max_iter=90,
        target="Spot Size (Focus)",
        num_rays=15,
        entrance_pupil_radius=5.0,
        target_efl=50.0,
        efl_weight=100.0
    )

    res = opt.optimize(variable_indices={"radius": [1, 3]})
    assert res.success is True
    final_efl, final_bfl = compute_system_efl(res.optimized_stack.surfaces)
    assert abs(final_efl - 50.0) / 50.0 < 0.05
    assert res.final_rms < 0.10

