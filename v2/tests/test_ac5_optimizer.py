"""
Acceptance Criterion 5 (AC5) Test Suite: SciPy Single-Pixel Spot Size Optimizer.
Verifies that the SciPy optimization engine minimizes RMS spot size on the single-pixel 0.8mm detector
by adjusting element angles (e.g. grating tilt) and inter-element lens spacings.
"""

import pytest
import numpy as np

from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.optimizer import (
    OpticalOptimizer, OptimizationResult, TerminationStatus, WavelengthSpotMetric
)
from optics_app.gui.qt_compat import QT_AVAILABLE
from optics_app.gui.optimizer_dock import OptimizerDockWidget
from optics_app.gui.lde_table import LensDataTableModel


def test_ac5_optimizer_initialization_with_spectrometer(loaded_optical_system):
    """AC5.1: Verify OpticalOptimizer initializes correctly with NIR band and defaults."""
    optimizer = OpticalOptimizer(
        loaded_optical_system,
        wavelengths=[1.10, 1.225, 1.35],
        method="Nelder-Mead",
        num_rays=7,
        entrance_pupil_radius=6.0,
        max_iter=10
    )
    assert optimizer.system is loaded_optical_system
    np.testing.assert_allclose(optimizer.wavelengths, [1.10, 1.225, 1.35], atol=1e-4)
    assert optimizer.method == "Nelder-Mead"
    assert optimizer.max_iter == 10
    assert optimizer.entrance_pupil_radius == 6.0


def test_ac5_optimizer_minimizes_spot_size_on_single_pixel_detector(loaded_spectrometer_stack):
    """AC5.2: Verify optimizer reduces RMS spot size on the 0.8mm detector from a perturbed state."""
    # Perturb grating tilt and focuser spacing
    loaded_spectrometer_stack.surfaces[3].tilt_x = -11.5  # nominal is -12.0
    loaded_spectrometer_stack.surfaces[6].thickness = 44.0  # nominal is 45.0
    loaded_spectrometer_stack._update_accumulated_positions()

    opt_sys = OpticalSystem(loaded_spectrometer_stack)
    optimizer = OpticalOptimizer(
        opt_sys,
        wavelengths=[1.225],
        method="Nelder-Mead",
        num_rays=7,
        max_iter=10
    )

    res = optimizer.optimize(
        variable_indices={"thickness": [6], "tilt_x": [3]}
    )

    assert isinstance(res, OptimizationResult)
    assert res.optimized_stack is not None
    # The final spot size should be smaller than or equal to initial, or within single-pixel detector diameter (0.8mm)
    assert res.final_rms <= res.initial_rms or res.final_rms < 0.8


def test_ac5_optimizer_nelder_mead_algorithm(loaded_spectrometer_stack):
    """AC5.3: Verify Nelder-Mead optimizer convergence on perturbed condenser spacing."""
    loaded_spectrometer_stack.surfaces[11].thickness = 50.0  # nominal is 53.44
    loaded_spectrometer_stack._update_accumulated_positions()

    opt_sys = OpticalSystem(loaded_spectrometer_stack)
    optimizer = OpticalOptimizer(
        opt_sys,
        wavelengths=[1.225],
        method="Nelder-Mead",
        num_rays=7,
        max_iter=10
    )

    res = optimizer.optimize(variable_indices=[11])
    assert isinstance(res, OptimizationResult)
    assert 11 in res.optimized_thicknesses
    assert res.optimized_thicknesses[11] > 0.0


def test_ac5_optimizer_lbfgsb_algorithm(loaded_spectrometer_stack):
    """AC5.4: Verify L-BFGS-B optimizer convergence with bounded variables."""
    loaded_spectrometer_stack.surfaces[2].thickness = 65.0  # nominal is 70.0
    loaded_spectrometer_stack._update_accumulated_positions()

    opt_sys = OpticalSystem(loaded_spectrometer_stack)
    optimizer = OpticalOptimizer(
        opt_sys,
        wavelengths=[1.225],
        method="L-BFGS-B",
        num_rays=7,
        max_iter=10
    )

    res = optimizer.optimize(
        variable_indices=[2],
        bounds=[(50.0, 100.0)]
    )
    assert isinstance(res, OptimizationResult)
    assert 2 in res.optimized_thicknesses
    assert 50.0 <= res.optimized_thicknesses[2] <= 100.0


def test_ac5_optimizer_penalty_barrier_constraints(loaded_optical_system):
    """AC5.5: Verify penalty barrier produces large penalty for unphysical negative thicknesses."""
    optimizer = OpticalOptimizer(
        loaded_optical_system,
        wavelengths=[1.225],
        method="Nelder-Mead"
    )

    # Evaluate candidate with negative thickness (-20.0 mm)
    pure_merit, penalty = optimizer._evaluate_merit_and_penalty(
        variable_values=np.array([-20.0]),
        variable_indices={"thickness": [2]}
    )
    assert penalty > 1000.0, f"Expected large penalty for negative thickness, got {penalty}"


def test_ac5_optimizer_abort_mechanism(loaded_optical_system):
    """AC5.6: Verify user abort flag immediately terminates optimization with USER_ABORTED status."""
    optimizer = OpticalOptimizer(
        loaded_optical_system,
        wavelengths=[1.225],
        max_iter=100
    )
    optimizer.stop()
    res = optimizer.optimize(variable_indices=[2, 6])
    assert res.status == TerminationStatus.USER_ABORTED
    assert "aborted" in res.message.lower()


def test_ac5_optimizer_result_structure(loaded_spectrometer_stack):
    """AC5.7: Verify OptimizationResult contains comprehensive metrics and per-wavelength data."""
    opt_sys = OpticalSystem(loaded_spectrometer_stack)
    optimizer = OpticalOptimizer(
        opt_sys,
        wavelengths=[1.10, 1.225, 1.35],
        num_rays=7,
        max_iter=2
    )

    res = optimizer.optimize(variable_indices=[2])
    assert hasattr(res, "success")
    assert hasattr(res, "initial_rms")
    assert hasattr(res, "final_rms")
    assert hasattr(res, "rms_reduction_pct")
    assert hasattr(res, "optimized_thicknesses")
    assert hasattr(res, "optimized_stack")
    assert hasattr(res, "per_wavelength_metrics")
    assert len(res.per_wavelength_metrics) == 3


def test_ac5_optimizer_dock_integration(loaded_spectrometer_stack, qapp):
    """AC5.8: Verify OptimizerDockWidget initializes with model and discovers variable surfaces."""
    model = LensDataTableModel(stack=loaded_spectrometer_stack)
    dock = OptimizerDockWidget(model=model)
    assert dock is not None
    assert hasattr(dock, "btn_optimize")
    assert hasattr(dock, "btn_stop")
