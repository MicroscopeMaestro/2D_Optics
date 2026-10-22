"""
Unit Tests for Optical Optimizer Pre-Flight Diagnostics, Parameter Validation, and Auto-Fix.
Verifies that invalid or conflicting user parameters produce clear, structured warnings and errors,
and that automated repair recovers systems to valid, optimizable states.
"""

import pytest
import numpy as np

from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.optimizer import (
    OpticalOptimizer, OptimizationResult, TerminationStatus,
    OptimizationValidator, ValidationResult, DiagnosticSeverity
)
from optics_app.gui.qt_compat import QT_AVAILABLE, QtWidgets
from optics_app.gui.lde_table import LensDataTableModel
from optics_app.gui.optimizer_dock import OptimizerDockWidget

if QT_AVAILABLE:
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])


def create_standard_doublet_stack() -> SurfaceStack:
    """Helper to construct standard doublet test stack."""
    stack = SurfaceStack()
    # Surf 0: Object
    stack.add_surface(Surface(name="Object", surface_type=SurfaceType.OBJECT, thickness=20.0, glass="AIR"))
    # Surf 1: Lens Front (BK7)
    stack.add_surface(Surface(name="Lens_Front", surface_type=SurfaceType.STANDARD, radius=45.0, thickness=5.0, glass="N-BK7", semi_diameter=12.5))
    # Surf 2: Lens Back
    stack.add_surface(Surface(name="Lens_Back", surface_type=SurfaceType.STANDARD, radius=-45.0, thickness=40.0, glass="AIR", semi_diameter=12.5))
    # Surf 3: Detector
    stack.add_surface(Surface(name="Detector", surface_type=SurfaceType.DETECTOR, thickness=0.0, glass="AIR", semi_diameter=15.0))
    stack._update_accumulated_positions()
    return stack


def test_preflight_no_variables_detection():
    """Verify pre-flight validator flags zero variable parameters and auto-repairs air gaps."""
    stack = create_standard_doublet_stack()
    
    # Pre-flight check with no variables selected
    res = OptimizationValidator.validate_system(
        stack=stack,
        variable_indices={"thickness": [], "radius": []}
    )
    
    assert res.has_errors is True
    assert any(item.code == "NO_VARIABLES_SELECTED" for item in res.items)
    assert res.auto_fix_available is True
    
    # Run auto-repair
    repaired_stack, var_dict, fixes = OptimizationValidator.auto_repair(
        stack=stack,
        variable_indices={"thickness": [], "radius": []}
    )
    assert len(var_dict["thickness"]) > 0
    assert 0 in var_dict["thickness"] or 2 in var_dict["thickness"]
    assert len(fixes) > 0

    # Post-repair validation should pass zero variables check
    res_after = OptimizationValidator.validate_system(stack=repaired_stack, variable_indices=var_dict)
    assert not any(item.code == "NO_VARIABLES_SELECTED" for item in res_after.items)


def test_preflight_inverted_bounds_detection():
    """Verify inverted bounds (min >= max) are detected as errors."""
    stack = create_standard_doublet_stack()
    
    # Inverted bounds on variable 0: (50.0, 10.0)
    bounds = [(50.0, 10.0), (1.0, 30.0)]
    res = OptimizationValidator.validate_system(
        stack=stack,
        variable_indices={"thickness": [0, 2]},
        bounds=bounds
    )
    
    assert res.has_errors is True
    assert any(item.code == "INVERTED_BOUNDS" for item in res.items)


def test_preflight_negative_thickness_detection_and_repair():
    """Verify negative thicknesses trigger warnings/errors and auto-repair resets them."""
    stack = create_standard_doublet_stack()
    stack.surfaces[1].thickness = -5.0  # Negative lens thickness!
    stack.surfaces[2].thickness = -10.0 # Negative air space!
    
    res = OptimizationValidator.validate_system(
        stack=stack,
        variable_indices={"thickness": [2]}
    )
    
    assert res.has_errors is True or res.has_warnings is True
    assert any(item.code == "NEGATIVE_THICKNESS" for item in res.items)
    
    # Run auto-repair
    repaired_stack, var_dict, fixes = OptimizationValidator.auto_repair(stack=stack, variable_indices={"thickness": [2]})
    assert repaired_stack.surfaces[1].thickness > 0
    assert repaired_stack.surfaces[2].thickness >= 0


def test_preflight_total_vignetting_detection():
    """Verify pre-flight check identifies blocking surface when rays cannot pass."""
    stack = create_standard_doublet_stack()
    # Decenter surface 1 far away so all rays are blocked
    stack.surfaces[1].semi_diameter = 1.0
    stack.surfaces[1].decenter_x = 50.0 # Shifted out of beam path
    stack._update_accumulated_positions()
    
    res = OptimizationValidator.validate_system(
        stack=stack,
        variable_indices={"thickness": [2]},
        entrance_pupil_radius=5.0
    )
    
    assert res.has_errors is True
    vignetting_item = next((item for item in res.items if item.code == "TOTAL_RAY_VIGNETTING"), None)
    assert vignetting_item is not None
    assert vignetting_item.surface_index == 1


def test_preflight_spectrometer_single_wavelength_error():
    """Verify Spectrometer Spread target requires multi-wavelength input."""
    stack = create_standard_doublet_stack()
    
    res = OptimizationValidator.validate_system(
        stack=stack,
        variable_indices={"thickness": [2]},
        wavelengths=0.785, # Single wavelength
        target="Spectrometer Spread"
    )
    
    assert res.has_errors is True
    assert any(item.code == "SPECTROMETER_NEEDS_MULTI_WAVELENGTH" for item in res.items)


def test_preflight_invalid_eval_surface():
    """Verify invalid or out-of-bounds evaluation surface indices are caught."""
    stack = create_standard_doublet_stack()
    
    res = OptimizationValidator.validate_system(
        stack=stack,
        variable_indices={"thickness": [2]},
        target_surface_index=99 # Out of bounds
    )
    
    assert res.has_errors is True
    assert any(item.code == "INVALID_EVALUATION_SURFACE" for item in res.items)


def test_optical_optimizer_validate_and_result_diagnostics():
    """Verify OpticalOptimizer.validate() returns ValidationResult and attaches to OptimizationResult."""
    stack = create_standard_doublet_stack()
    sys = OpticalSystem(stack)
    optimizer = OpticalOptimizer(sys, wavelengths=[0.5876], max_iter=10)
    
    # Pre-flight check via optimizer
    val_res = optimizer.validate(variable_indices={"thickness": [2]})
    assert isinstance(val_res, ValidationResult)
    
    # Run optimization
    opt_res = optimizer.optimize(variable_indices={"thickness": [2]})
    assert isinstance(opt_res, OptimizationResult)
    assert opt_res.validation_result is not None
    assert isinstance(opt_res.validation_result, ValidationResult)


def test_optimizer_dock_validate_and_auto_repair():
    """Verify OptimizerDockWidget validation and auto-repair methods work in headless/GUI mode."""
    stack = create_standard_doublet_stack()
    model = LensDataTableModel()
    model.set_surface_stack(stack)
    
    dock = OptimizerDockWidget(model=model)
    # Initially no checkboxes selected
    dock.selected_indices = []
    dock.selected_rad_indices = []
    
    val = dock.validate_current_configuration()
    assert isinstance(val, ValidationResult)
    assert val.has_errors is True
    
    # Auto-repair via dock
    fixes = dock.auto_repair_configuration()
    assert len(fixes) > 0
    
    # After auto-repair, at least one thickness checkbox is checked
    vars_now = dock.get_variable_indices()
    assert len(vars_now.get("thickness", [])) > 0
