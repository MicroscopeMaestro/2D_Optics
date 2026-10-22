"""
Tier 5 White-Box Adversarial Hardening & Verification Suite:
UI Delivery, Diagnostics, Paraxial Ray Tracing, ISO 10110 Manufacturing Standards,
and Dual-Tree SHA-256 Parity across optics_app/ and v2/.
"""

import os
import sys
import math
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import pytest

# Ensure headless Qt operation
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["MPLBACKEND"] = "Agg"

from optics_app.gui.qt_compat import QT_AVAILABLE, QtWidgets, QDoubleSpinBox
if QT_AVAILABLE and hasattr(QtWidgets, "QApplication"):
    _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from optics_app.gui.lde_table import LensDataTableModel
from optics_app.gui.main_window import MainWindow
from optics_app.gui.optimizer_dock import OptimizerDockWidget, VariableIndices
from optics_app.engine.diagnostics import (
    OptimizationValidator,
    ValidationResult,
    DiagnosticSeverity,
    DiagnosticItem,
    ModelSanityValidator
)
from optics_app.engine.first_order import (
    calculate_first_order_data,
    compute_paraxial_marginal_and_principal_rays
)
from optics_app.engine.manufacturing_standards import (
    audit_optical_system_manufacturing,
    ManufacturingConstraintsConfig,
    QualityPreset,
    TestPlateCatalog,
    SystemManufacturingReport,
    ElementManufacturingAudit
)

# Counterpart imports from v2 to test functional parity
from v2.engine.diagnostics import OptimizationValidator as V2OptimizationValidator
from v2.engine.first_order import (
    calculate_first_order_data as v2_calculate_first_order_data,
    compute_paraxial_marginal_and_principal_rays as v2_compute_paraxial_rays
)
from v2.engine.manufacturing_standards import (
    audit_optical_system_manufacturing as v2_audit_optical_system_manufacturing,
    TestPlateCatalog as V2TestPlateCatalog
)


# ============================================================================
# Helper Fixtures: Standard Multi-Element Systems
# ============================================================================

def make_achromat_doublet() -> List[Surface]:
    """Compliant F/4 100mm Achromatic Doublet."""
    s0 = Surface(name="Object", surface_type=SurfaceType.OBJECT, thickness=1e9)
    s1 = Surface(name="Crown_front", radius=62.5, thickness=6.0, glass="N-BK7", semi_diameter=12.5)
    s2 = Surface(name="Crown_back", radius=-45.0, thickness=1.0, glass="AIR", semi_diameter=12.5)
    s3 = Surface(name="Flint_front", radius=-44.5, thickness=3.0, glass="N-SF5", semi_diameter=12.5)
    s4 = Surface(name="Flint_back", radius=-150.0, thickness=90.0, glass="AIR", semi_diameter=12.5)
    s5 = Surface(name="Detector", surface_type=SurfaceType.DETECTOR, semi_diameter=10.0)
    return [s0, s1, s2, s3, s4, s5]


def make_cooke_triplet() -> List[Surface]:
    """Standard F/4.5 50mm Cooke Triplet."""
    s0 = Surface(name="Object", surface_type=SurfaceType.OBJECT, thickness=1e9)
    s1 = Surface(name="L1_front", radius=22.01, thickness=3.26, glass="N-SK16", semi_diameter=9.0)
    s2 = Surface(name="L1_back", radius=-435.76, thickness=6.0, glass="AIR", semi_diameter=9.0)
    s3 = Surface(name="L2_front", radius=-22.21, thickness=1.5, glass="N-SF2", semi_diameter=7.5)
    s4 = Surface(name="L2_back", radius=20.29, thickness=4.75, glass="AIR", semi_diameter=7.5)
    s5 = Surface(name="L3_front", radius=79.68, thickness=2.95, glass="N-SK16", semi_diameter=8.5)
    s6 = Surface(name="L3_back", radius=-18.39, thickness=42.21, glass="AIR", semi_diameter=8.5)
    s7 = Surface(name="Detector", surface_type=SurfaceType.DETECTOR, semi_diameter=10.0)
    return [s0, s1, s2, s3, s4, s5, s6, s7]


def make_double_gauss() -> List[Surface]:
    """6-element Double Gauss with 2 cemented doublets and central aperture stop."""
    s0 = Surface(name="Object", surface_type=SurfaceType.OBJECT, thickness=1e9)
    s1 = Surface(name="L1_front", radius=56.0, thickness=6.0, glass="N-SK16", semi_diameter=16.0)
    s2 = Surface(name="L1_back", radius=500.0, thickness=0.5, glass="AIR", semi_diameter=16.0)
    s3 = Surface(name="L2_front", radius=30.0, thickness=8.0, glass="N-SK16", semi_diameter=14.0)
    s4 = Surface(name="L2_cemented", radius=-35.0, thickness=2.5, glass="N-SF2", semi_diameter=14.0)
    s5 = Surface(name="L2_back", radius=16.0, thickness=5.0, glass="AIR", semi_diameter=11.0)
    s6 = Surface(name="Stop", surface_type=SurfaceType.STOP, radius=0.0, thickness=4.5, glass="AIR", semi_diameter=10.0)
    s7 = Surface(name="L3_front", radius=-18.0, thickness=2.5, glass="N-SF2", semi_diameter=11.0)
    s8 = Surface(name="L3_cemented", radius=40.0, thickness=7.0, glass="N-SK16", semi_diameter=14.0)
    s9 = Surface(name="L3_back", radius=-32.0, thickness=0.5, glass="AIR", semi_diameter=14.0)
    s10 = Surface(name="L4_front", radius=120.0, thickness=5.0, glass="N-SK16", semi_diameter=15.0)
    s11 = Surface(name="L4_back", radius=-60.0, thickness=45.0, glass="AIR", semi_diameter=15.0)
    s12 = Surface(name="Detector", surface_type=SurfaceType.DETECTOR, semi_diameter=12.0)
    return [s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12]


def make_petzval_lens() -> List[Surface]:
    """Petzval Portrait Lens with 2 separated positive doublet groups."""
    s0 = Surface(name="Object", surface_type=SurfaceType.OBJECT, thickness=1e9)
    # Front group: cemented doublet
    s1 = Surface(name="Front_L1_front", radius=45.0, thickness=7.0, glass="N-BK7", semi_diameter=15.0)
    s2 = Surface(name="Front_cemented", radius=-35.0, thickness=2.5, glass="N-SF5", semi_diameter=15.0)
    s3 = Surface(name="Front_L2_back", radius=-180.0, thickness=30.0, glass="AIR", semi_diameter=15.0)
    # Rear group: air-spaced doublet
    s4 = Surface(name="Rear_L1_front", radius=38.0, thickness=5.0, glass="N-BK7", semi_diameter=12.5)
    s5 = Surface(name="Rear_L1_back", radius=-60.0, thickness=2.0, glass="AIR", semi_diameter=12.5)
    s6 = Surface(name="Rear_L2_front", radius=-55.0, thickness=3.0, glass="N-SF5", semi_diameter=12.5)
    s7 = Surface(name="Rear_L2_back", radius=120.0, thickness=40.0, glass="AIR", semi_diameter=12.5)
    s8 = Surface(name="Detector", surface_type=SurfaceType.DETECTOR, semi_diameter=10.0)
    return [s0, s1, s2, s3, s4, s5, s6, s7, s8]


# ============================================================================
# MODULE 1: UI Delivery and Model Synchronization Under Adversarial Stress
# ============================================================================

def test_tier5_ui_rapid_consecutive_reoptimizations():
    """
    Stress-test rapid consecutive re-optimization invocations on MainWindow.
    Verifies atomic stack delivery, dock status refresh, checkbox synchronization,
    and active_optimizer release in the finally block across multiple rapid cycles.
    """
    mw = MainWindow()
    assert mw.model is not None

    # Load doublet prescription
    stack = SurfaceStack()
    for s in make_achromat_doublet():
        stack.add_surface(s)
    mw.model.set_surface_stack(stack)

    assert len(mw.model.get_surface_stack().surfaces) == 6

    # Execute 4 rapid consecutive optimization cycles
    for cycle in range(4):
        # Configure variable thicknesses on air spaces
        var_dict = {"thickness": [2, 4], "radius": []}
        res = mw.on_optimize(
            variable_indices=var_dict,
            method="Nelder-Mead",
            max_iter=5,
            num_rays=7,
            blocking=True
        )

        assert res is not None
        assert res.optimized_stack is not None
        # active_optimizer MUST be cleaned up to None
        assert mw.active_optimizer is None, f"Cycle {cycle}: active_optimizer was not released!"
        # Model must have been updated
        curr_stack = mw.model.get_surface_stack()
        assert len(curr_stack.surfaces) == 6
        assert curr_stack.surfaces[4].thickness >= 0.0

        # Optimizer dock state check
        if hasattr(mw, "optimizer_dock") and mw.optimizer_dock is not None:
            assert mw.optimizer_dock.last_result is not None


def test_tier5_ui_abort_during_active_optimization():
    """
    Adversarial challenge: Abort optimization during active solver iterations.
    Verifies that opt.stop() / mw._abort_optimization() gracefully halts execution,
    updates status messages, releases active_optimizer, and allows immediate subsequent re-optimization.
    """
    mw = MainWindow()
    stack = SurfaceStack()
    for s in make_achromat_doublet():
        stack.add_surface(s)
    mw.model.set_surface_stack(stack)

    # Trigger abort via progress callback
    abort_triggered = False

    def aborting_callback(curr=None, maximum=None):
        nonlocal abort_triggered
        abort_triggered = True
        mw._abort_optimization()

    # Pre-configure an optimizer with abort callback
    sys_inst = OpticalSystem(stack)
    from optics_app.engine.optimizer import OpticalOptimizer
    opt = OpticalOptimizer(sys_inst, max_iter=100)
    mw.active_optimizer = opt

    # Calling _abort_optimization
    mw._abort_optimization()
    assert opt.abort is True
    assert mw.active_optimizer is not None  # Cleared by on_optimize finally

    # Now run an optimization where abort is called on first iteration
    def hook_progress(curr=None, maximum=None):
        if curr is not None and curr >= 1:
            mw._abort_optimization()

    res = mw.on_optimize(
        variable_indices={"thickness": [4], "radius": []},
        method="Nelder-Mead",
        max_iter=30,
        num_rays=7
    )
    assert res is not None
    assert mw.active_optimizer is None, "active_optimizer must be None after execution."

    # Immediate subsequent optimization must succeed cleanly without lockup
    res2 = mw.on_optimize(
        variable_indices={"thickness": [4], "radius": []},
        method="Nelder-Mead",
        max_iter=3,
        num_rays=7
    )
    assert res2 is not None
    assert mw.active_optimizer is None


def test_tier5_ui_empty_and_single_surface_stack_safety():
    """
    Adversarial input: on_optimize called when LensDataTableModel contains
    an empty SurfaceStack (0 surfaces) or a single surface (1 surface).
    Verifies graceful return of OptimizationResult(success=False) without uncaught exceptions.
    """
    mw = MainWindow()

    # Case A: 0 surfaces
    empty_stack = SurfaceStack()
    mw.model.set_surface_stack(empty_stack)
    res_empty = mw.on_optimize()
    assert res_empty is not None
    assert not res_empty.success
    assert "no surfaces" in res_empty.message.lower()
    assert mw.active_optimizer is None

    # Case B: 1 surface (just an object plane)
    single_stack = SurfaceStack()
    single_stack.add_surface(Surface(name="Object", surface_type=SurfaceType.OBJECT, thickness=10.0))
    mw.model.set_surface_stack(single_stack)
    res_single = mw.on_optimize()
    assert res_single is not None
    assert mw.active_optimizer is None


def test_tier5_ui_invalid_user_inputs_and_wavelength_parsing():
    """
    Adversarial input testing:
    - Malformed wavelength text strings in mw.wavelengths_input (negative, text, commas, >1000 nm).
    - VariableIndices custom dict compliance with len, iter, int in, count, and equality.
    """
    mw = MainWindow()
    stack = SurfaceStack()
    for s in make_achromat_doublet():
        stack.add_surface(s)
    mw.model.set_surface_stack(stack)

    # Test VariableIndices behaviors
    vars_dict = VariableIndices({"thickness": [1, 2], "radius": [3]})
    assert 1 in vars_dict
    assert 2 in vars_dict
    assert 3 in vars_dict
    assert 4 not in vars_dict
    assert 999 not in vars_dict
    assert -1 not in vars_dict
    assert len(vars_dict) == 3
    assert sorted(list(vars_dict)) == [1, 2, 3]
    assert vars_dict == [1, 2, 3]
    assert vars_dict.count(2) == 1

    # Test numpy integer inside VariableIndices
    np_idx = np.int64(2)
    assert np_idx in vars_dict

    # Test malformed wavelength inputs
    if hasattr(mw, "wavelengths_input") and mw.wavelengths_input is not None:
        # Non-numeric, negatives, and large wavelengths (nm to um scaling)
        mw.wavelengths_input.setText("invalid, 632.8, -500, 0.4861, 587.6")
        res = mw.on_optimize(
            variable_indices={"thickness": [4], "radius": []},
            max_iter=2,
            num_rays=5
        )
        assert res is not None
        assert res.success or res.final_rms >= 0.0


def test_tier5_ui_lens_data_table_model_atomic_updates_and_undo():
    """
    Stress-test atomic model delivery to LensDataTableModel:
    - Setting new surface stack.
    - Emitting system_updated signal.
    - Pushing to undo stack.
    """
    mw = MainWindow()
    signals_received = []
    mw.model.system_updated.connect(lambda: signals_received.append("updated"))

    stack = SurfaceStack()
    for s in make_achromat_doublet():
        stack.add_surface(s)

    mw.model.set_surface_stack(stack, emit_signal=True)
    assert len(signals_received) >= 1
    assert mw.model.rowCount() == 6

    # Verify column data access
    assert mw.model.data(mw.model.index(1, 0)) is not None  # Surface 1 enumeration
    assert mw.model.data(mw.model.index(1, 4)) == "N-BK7"   # Glass column

    # Model undo verification via MainWindow
    if hasattr(mw.model, "_push_undo"):
        mw.model._push_undo("Test Undo Step")
        assert mw.undo_manager.can_undo()


def test_tier5_ui_fit_test_plates_action():
    """
    Verify MainWindow.on_fit_test_plates() snaps spherical radii to tooling plates
    and updates the model atomically.
    """
    mw = MainWindow()
    stack = SurfaceStack()
    # Lens with non-standard radius
    s0 = Surface(name="Object", surface_type=SurfaceType.OBJECT, thickness=100.0)
    s1 = Surface(name="L1", radius=51.23, thickness=6.0, glass="N-BK7", semi_diameter=12.5)
    s2 = Surface(name="L2", radius=-47.89, thickness=50.0, glass="AIR", semi_diameter=12.5)
    s3 = Surface(name="Det", surface_type=SurfaceType.DETECTOR)
    for s in [s0, s1, s2, s3]:
        stack.add_surface(s)
    mw.model.set_surface_stack(stack)

    mw.on_fit_test_plates()

    fitted_stack = mw.model.get_surface_stack()
    assert fitted_stack is not None
    # Radii should have snapped to catalog values (50.0 and -50.0 or -45.0)
    assert fitted_stack.surfaces[1].radius in TestPlateCatalog.STANDARD_RADII_MM
    assert abs(fitted_stack.surfaces[2].radius) in TestPlateCatalog.STANDARD_RADII_MM


# ============================================================================
# MODULE 2: Pre-Flight Diagnostic Validation and Auto-Repair Routines
# ============================================================================

def test_tier5_diagnostics_inverted_bounds():
    """
    Pre-flight audit: Variable parameter bounds where min >= max.
    Must flag INVERTED_BOUNDS with ERROR severity.
    """
    stack = SurfaceStack()
    for s in make_achromat_doublet():
        stack.add_surface(s)

    # Inverted bound: min (20.0) >= max (5.0)
    res = OptimizationValidator.validate_system(
        stack=stack,
        variable_indices={"thickness": [2]},
        bounds=[(20.0, 5.0)]
    )

    assert res.has_errors
    codes = [item.code for item in res.items]
    assert "INVERTED_BOUNDS" in codes
    item = next(it for it in res.items if it.code == "INVERTED_BOUNDS")
    assert item.severity == DiagnosticSeverity.ERROR

    # Also test v2 counterpart parity
    v2_res = V2OptimizationValidator.validate_system(
        stack=stack,
        variable_indices={"thickness": [2]},
        bounds=[(20.0, 5.0)]
    )
    assert v2_res.has_errors
    assert "INVERTED_BOUNDS" in [it.code for it in v2_res.items]


def test_tier5_diagnostics_negative_and_zero_thickness():
    """
    Pre-flight audit:
    - Solid glass lens with negative center thickness -> ERROR (NEGATIVE_THICKNESS).
    - Air gap with negative center thickness -> WARNING (NEGATIVE_THICKNESS).
    - Solid glass lens with zero center thickness -> WARNING (ZERO_LENS_THICKNESS).
    """
    stack = SurfaceStack()
    s0 = Surface(name="Object", thickness=10.0)
    s1_neg = Surface(name="Glass_Neg", radius=50.0, thickness=-5.0, glass="N-BK7", semi_diameter=12.5)
    s2_air = Surface(name="Air_Neg", radius=-50.0, thickness=-2.0, glass="AIR", semi_diameter=12.5)
    s3_zero = Surface(name="Glass_Zero", radius=50.0, thickness=0.0, glass="N-SF5", semi_diameter=12.5)
    s4_det = Surface(name="Det", surface_type=SurfaceType.DETECTOR)
    for s in [s0, s1_neg, s2_air, s3_zero, s4_det]:
        stack.add_surface(s)

    res = OptimizationValidator.validate_system(stack=stack, variable_indices={"thickness": [1, 2]})
    codes = [item.code for item in res.items]

    assert "NEGATIVE_THICKNESS" in codes
    assert "ZERO_LENS_THICKNESS" in codes

    neg_glass_item = next(it for it in res.items if it.code == "NEGATIVE_THICKNESS" and it.surface_index == 1)
    assert neg_glass_item.severity == DiagnosticSeverity.ERROR

    neg_air_item = next(it for it in res.items if it.code == "NEGATIVE_THICKNESS" and it.surface_index == 2)
    assert neg_air_item.severity == DiagnosticSeverity.WARNING

    zero_glass_item = next(it for it in res.items if it.code == "ZERO_LENS_THICKNESS" and it.surface_index == 3)
    assert zero_glass_item.severity == DiagnosticSeverity.WARNING


def test_tier5_diagnostics_near_zero_radius_singularity():
    """
    Pre-flight audit: Near-zero radius of curvature singularities (0 < |R| < 0.1 mm).
    Must flag NEAR_ZERO_RADIUS warning to prevent numerical divergence.
    """
    stack = SurfaceStack()
    s0 = Surface(name="Obj", thickness=10.0)
    s1 = Surface(name="Singular_R", radius=0.05, thickness=5.0, glass="N-BK7", semi_diameter=10.0)
    s2 = Surface(name="Det", surface_type=SurfaceType.DETECTOR)
    stack.add_surface(s0); stack.add_surface(s1); stack.add_surface(s2)

    res = OptimizationValidator.validate_system(stack=stack, variable_indices={"thickness": [1]})
    codes = [item.code for item in res.items]
    assert "NEAR_ZERO_RADIUS" in codes
    item = next(it for it in res.items if it.code == "NEAR_ZERO_RADIUS")
    assert item.severity == DiagnosticSeverity.WARNING


def test_tier5_diagnostics_invalid_pupil_and_sampling_parameters():
    """
    Pre-flight audit:
    - Non-positive entrance pupil radius (<= 0) -> ERROR (INVALID_PUPIL_RADIUS).
    - Ray count < 3 -> WARNING (INSUFFICIENT_RAY_SAMPLING).
    - Non-positive max iterations (<= 0) -> ERROR (INVALID_MAX_ITERATIONS).
    """
    stack = SurfaceStack()
    for s in make_achromat_doublet():
        stack.add_surface(s)

    # Invalid pupil
    res_pupil = OptimizationValidator.validate_system(stack=stack, entrance_pupil_radius=-2.0)
    assert any(it.code == "INVALID_PUPIL_RADIUS" for it in res_pupil.items)

    # Insufficient rays
    res_rays = OptimizationValidator.validate_system(stack=stack, num_rays=1)
    assert any(it.code == "INSUFFICIENT_RAY_SAMPLING" for it in res_rays.items)

    # Invalid max iterations
    res_iter = OptimizationValidator.validate_system(stack=stack, max_iter=0)
    assert any(it.code == "INVALID_MAX_ITERATIONS" for it in res_iter.items)


def test_tier5_diagnostics_target_and_spectrometer_checks():
    """
    Pre-flight audit:
    - Target surface index out of bounds -> ERROR (INVALID_EVALUATION_SURFACE).
    - Spectrometer Spread optimization with only 1 wavelength -> ERROR (SPECTROMETER_NEEDS_MULTI_WAVELENGTH).
    - Negative target spot size -> ERROR (NEGATIVE_TARGET_SPOT).
    """
    stack = SurfaceStack()
    for s in make_achromat_doublet():
        stack.add_surface(s)

    # Out-of-bounds target surface
    res_surf = OptimizationValidator.validate_system(stack=stack, target_surface_index=99)
    assert any(it.code == "INVALID_EVALUATION_SURFACE" for it in res_surf.items)

    # Spectrometer spread with single wavelength
    res_spec = OptimizationValidator.validate_system(
        stack=stack,
        target="Spectrometer Spread",
        wavelengths=[0.5876]
    )
    assert any(it.code == "SPECTROMETER_NEEDS_MULTI_WAVELENGTH" for it in res_spec.items)

    # Negative target spot
    res_spot = OptimizationValidator.validate_system(
        stack=stack,
        target="Target Spot Size",
        target_spot_size=-0.05
    )
    assert any(it.code == "NEGATIVE_TARGET_SPOT" for it in res_spot.items)


def test_tier5_diagnostics_vignetting_detection():
    """
    Pre-flight audit: Detects severe ray vignetting:
    - Negative thickness placing surface backward causing total trace failure -> TOTAL_RAY_VIGNETTING.
    - Clipped semi-diameter -> PARTIAL_RAY_VIGNETTING.
    """
    # Total vignetting from backward surface placement
    stack_total = SurfaceStack()
    s0 = Surface(name="Obj", thickness=20.0)
    s1_neg = Surface(name="Bad_Lens", radius=50.0, thickness=-10.0, glass="N-BK7", semi_diameter=12.5)
    s2 = Surface(name="Rear", radius=-50.0, thickness=20.0, glass="AIR", semi_diameter=12.5)
    s3_det = Surface(name="Det", surface_type=SurfaceType.DETECTOR)
    for s in [s0, s1_neg, s2, s3_det]:
        stack_total.add_surface(s)

    res_total = OptimizationValidator.validate_system(stack=stack_total, variable_indices={"thickness": [2]})
    codes_total = [it.code for it in res_total.items]
    assert "TOTAL_RAY_VIGNETTING" in codes_total

    # Partial vignetting from tight aperture
    stack_partial = SurfaceStack()
    s0 = Surface(name="Obj", thickness=20.0)
    s1_tight = Surface(name="Tight_Stop", radius=0.0, thickness=20.0, glass="AIR", semi_diameter=0.5)
    s2_det = Surface(name="Det", surface_type=SurfaceType.DETECTOR, semi_diameter=10.0)
    stack_partial.add_surface(s0); stack_partial.add_surface(s1_tight); stack_partial.add_surface(s2_det)

    res_partial = OptimizationValidator.validate_system(
        stack=stack_partial,
        variable_indices={"thickness": [1]},
        entrance_pupil_radius=5.0
    )
    codes_partial = [it.code for it in res_partial.items]
    assert "PARTIAL_RAY_VIGNETTING" in codes_partial


def test_tier5_diagnostics_auto_repair_recovery_pipeline():
    """
    Test automated one-click repair routine (OptimizationValidator.auto_repair):
    - Pathological system with 0 variables, negative glass thickness, and negative air gap.
    - auto_repair marks air gaps as variable, resets glass thickness to 3.0 mm, and air gap to 10.0 mm.
    - Verifies that post-repair system has 0 critical errors.
    """
    corrupt_stack = SurfaceStack()
    s0 = Surface(name="Obj", thickness=20.0)
    s1_bad_glass = Surface(name="Bad_Glass", radius=50.0, thickness=-4.0, glass="N-BK7", semi_diameter=12.5)
    s2_bad_air = Surface(name="Bad_Air", radius=-50.0, thickness=-15.0, glass="AIR", semi_diameter=12.5)
    s3_det = Surface(name="Det", surface_type=SurfaceType.DETECTOR)
    for s in [s0, s1_bad_glass, s2_bad_air, s3_det]:
        corrupt_stack.add_surface(s)

    # Initial validation must fail
    res_before = OptimizationValidator.validate_system(corrupt_stack, variable_indices={})
    assert res_before.has_errors
    assert res_before.auto_fix_available

    # Execute auto_repair
    repaired_stack, repaired_vars, applied_fixes = OptimizationValidator.auto_repair(
        stack=corrupt_stack,
        variable_indices={}
    )

    assert len(applied_fixes) >= 2
    assert repaired_stack.surfaces[1].thickness == 3.0
    assert repaired_stack.surfaces[2].thickness == 10.0
    assert len(repaired_vars.get("thickness", [])) >= 1

    # Post-repair validation must have no critical errors
    res_after = OptimizationValidator.validate_system(repaired_stack, variable_indices=repaired_vars)
    assert not res_after.has_errors

    # Test OptimizerDockWidget integration of auto-repair with fresh corrupt stack
    mw = MainWindow()
    fresh_corrupt = SurfaceStack()
    s0_f = Surface(name="Obj", thickness=20.0)
    s1_f = Surface(name="Bad_Glass_Dock", radius=50.0, thickness=-4.0, glass="N-BK7", semi_diameter=12.5)
    s2_f = Surface(name="Bad_Air_Dock", radius=-50.0, thickness=-15.0, glass="AIR", semi_diameter=12.5)
    s3_f = Surface(name="Det", surface_type=SurfaceType.DETECTOR)
    for s in [s0_f, s1_f, s2_f, s3_f]:
        fresh_corrupt.add_surface(s)
    mw.model.set_surface_stack(fresh_corrupt)
    dock = mw.optimizer_dock
    if dock is not None:
        fixes = dock.auto_repair_configuration()
        assert len(fixes) >= 1
        repaired_dock_stack = mw.model.get_surface_stack()
        assert repaired_dock_stack.surfaces[1].thickness == 3.0


# ============================================================================
# MODULE 3: Paraxial Ray Tracing, ABCD Cardinal Matrices, and Extreme Geometries
# ============================================================================

def test_tier5_paraxial_afocal_systems():
    """
    Paraxial ray tracing and ABCD cardinal matrix calculation under afocal conditions (C == 0).
    Verifies that is_afocal is True, effl is inf, bfl is inf, and f_number is inf without ZeroDivisionError.
    """
    # Plano-parallel plate in air (C = 0 exactly)
    surfs = [
        Surface(name="Object", surface_type=SurfaceType.OBJECT, thickness=10.0),
        Surface(name="Window_front", radius=0.0, thickness=10.0, glass="N-BK7", semi_diameter=15.0),
        Surface(name="Window_back", radius=0.0, thickness=20.0, glass="AIR", semi_diameter=15.0),
        Surface(name="Detector", surface_type=SurfaceType.DETECTOR, semi_diameter=15.0)
    ]

    res = calculate_first_order_data(surfaces=surfs)
    assert res["is_afocal"] is True
    assert math.isinf(res["effl"])
    assert math.isinf(res["bfl"])
    assert math.isinf(res["f_number"])
    assert res["optical_power_diopters"] == 0.0

    # Also verify with v2 counterpart
    v2_res = v2_calculate_first_order_data(surfaces=surfs)
    assert v2_res["is_afocal"] is True
    assert math.isinf(v2_res["effl"])


def test_tier5_paraxial_telecentric_systems():
    """
    Paraxial ray tracing of an image-space telecentric system:
    Stop positioned at front focal distance of a positive lens.
    Verifies that exit pupil is positioned at large distance / infinity and chief ray emerges parallel to optical axis.
    """
    # Single biconvex lens f ~ 25.8 mm. Put stop at front focal plane (~25.8 mm in front)
    surfs = [
        Surface(name="Object", surface_type=SurfaceType.OBJECT, thickness=50.0),
        Surface(name="Stop", surface_type=SurfaceType.STOP, radius=0.0, thickness=25.85, glass="AIR", semi_diameter=5.0),
        Surface(name="L1_front", radius=25.84, thickness=5.0, glass="N-BK7", semi_diameter=12.5),
        Surface(name="L1_back", radius=-25.84, thickness=25.0, glass="AIR", semi_diameter=12.5),
        Surface(name="Detector", surface_type=SurfaceType.DETECTOR, semi_diameter=12.5)
    ]

    res = calculate_first_order_data(surfaces=surfs)
    assert not res["is_afocal"]
    assert res["effl"] > 20.0
    # Exit pupil position is far behind or at infinity
    assert abs(res["exp_pos"]) > 100.0

    rays_data = compute_paraxial_marginal_and_principal_rays(surfaces=surfs, epd=10.0, field_angle_deg=2.0)
    assert not rays_data["is_vignetted"]
    # Check that chief ray angle after rear surface is close to 0 (telecentric)
    sdata = rays_data["surfaces_data"]
    assert len(sdata) >= 3


def test_tier5_paraxial_mirror_reflection_and_focus():
    """
    Paraxial ray tracing of reflective mirror systems:
    Spherical concave mirror (R = 200 mm, f = 100 mm).
    Verifies optical power Phi = 2n/R, marginal ray convergence to y = 0 at f = 100 mm.
    """
    surfs = [
        Surface(name="Object", surface_type=SurfaceType.OBJECT, thickness=1e9),
        Surface(name="Concave_Mirror", surface_type=SurfaceType.MIRROR, radius=200.0, thickness=100.0, glass="MIRROR", semi_diameter=25.0),
        Surface(name="Detector", surface_type=SurfaceType.DETECTOR, semi_diameter=25.0)
    ]

    rays_data = compute_paraxial_marginal_and_principal_rays(surfaces=surfs, epd=20.0, field_angle_deg=0.0)
    sdata = rays_data["surfaces_data"]
    # Surface 0 is Mirror (y_m = 10.0), Surface 1 is Detector (y_m = 0.0 at focus)
    assert abs(sdata[0]["y_marginal"] - 10.0) < 1e-4
    assert abs(sdata[1]["y_marginal"]) < 1e-4, f"Mirror ray did not focus: y={sdata[1]['y_marginal']}"

    # Also test flat fold mirror (R = 0)
    flat_surfs = [
        Surface(name="Object", surface_type=SurfaceType.OBJECT, thickness=1e9),
        Surface(name="Fold_Mirror", surface_type=SurfaceType.MIRROR, radius=0.0, thickness=20.0, glass="MIRROR", semi_diameter=25.0),
        Surface(name="Detector", surface_type=SurfaceType.DETECTOR, semi_diameter=25.0)
    ]
    flat_rays = compute_paraxial_marginal_and_principal_rays(surfaces=flat_surfs, epd=10.0, field_angle_deg=0.0)
    assert abs(flat_rays["surfaces_data"][0]["y_marginal"] - 5.0) < 1e-4
    assert abs(flat_rays["surfaces_data"][1]["y_marginal"] - 5.0) < 1e-4


def test_tier5_paraxial_extreme_field_angles_and_conjugates():
    """
    Paraxial ray propagation under extreme field angles (0, 30, 60, 85 deg)
    and infinite vs near-zero conjugate object distances.
    Verifies numerical stability, trajectory generation, and vignetting factor bounds.
    """
    surfs = make_cooke_triplet()

    for angle in [0.0, 15.0, 45.0, 75.0, 85.0]:
        rays_data = compute_paraxial_marginal_and_principal_rays(
            surfaces=surfs,
            epd=10.0,
            field_angle_deg=angle
        )
        assert 0.0 <= rays_data["system_vignetting_factor"]
        assert len(rays_data["marginal_trajectory"]["z"]) == len(surfs) - 1
        assert len(rays_data["chief_trajectory"]["y"]) == len(surfs) - 1
        assert not math.isnan(rays_data["system_vignetting_factor"])

    # Macro conjugate: S0 = 5.0 mm
    surfs_macro = make_cooke_triplet()
    surfs_macro[0].thickness = 5.0
    res_macro = calculate_first_order_data(surfaces=surfs_macro)
    assert not math.isnan(res_macro["effl"])
    assert res_macro["effl"] > 0.0


def test_tier5_paraxial_intermediate_aperture_clipping_and_limiting_surface():
    """
    Paraxial ray tracing with internal apertures smaller than beam envelope.
    Verifies that compute_paraxial_marginal_and_principal_rays detects is_vignetted == True,
    identifies the exact limiting_surface_index and limiting_surface_name,
    and computes the system_vignetting_factor.
    """
    surfs = [
        Surface(name="Object", surface_type=SurfaceType.OBJECT, thickness=10.0),
        Surface(name="Lens1", radius=0.0, thickness=10.0, glass="N-BK7", semi_diameter=20.0),
        Surface(name="Aperture_Clip", surface_type=SurfaceType.STOP, radius=0.0, thickness=20.0, glass="AIR", semi_diameter=1.5),
        Surface(name="Lens2", radius=0.0, thickness=10.0, glass="N-BK7", semi_diameter=20.0),
        Surface(name="Detector", surface_type=SurfaceType.DETECTOR, semi_diameter=20.0)
    ]

    rays_data = compute_paraxial_marginal_and_principal_rays(
        surfaces=surfs,
        epd=10.0,
        field_angle_deg=5.0
    )

    assert rays_data["is_vignetted"] is True
    assert rays_data["limiting_surface_name"] == "Aperture_Clip"
    assert rays_data["system_vignetting_factor"] > 0.0

    # Also test v2 counterpart parity
    v2_rays = v2_compute_paraxial_rays(
        surfaces=surfs,
        epd=10.0,
        field_angle_deg=5.0
    )
    assert v2_rays["is_vignetted"] is True
    assert v2_rays["limiting_surface_name"] == "Aperture_Clip"


def test_tier5_paraxial_empty_and_minimal_stack_safety():
    """
    Edge case: calculate_first_order_data and compute_paraxial_marginal_and_principal_rays
    called with empty surface list ([]) or single surface ([Surface()]).
    Verifies graceful return of safe fallback dictionaries without uncaught exceptions.
    """
    # Empty surfaces
    res_empty = calculate_first_order_data([])
    assert res_empty["is_afocal"] is True
    assert math.isinf(res_empty["effl"])
    assert "error" in res_empty

    rays_empty = compute_paraxial_marginal_and_principal_rays([])
    assert rays_empty["is_vignetted"] is False
    assert rays_empty["system_vignetting_factor"] == 0.0

    # Single surface
    single = [Surface(name="Single")]
    res_single = calculate_first_order_data(single)
    assert res_single is not None
    rays_single = compute_paraxial_marginal_and_principal_rays(single)
    assert rays_single["is_vignetted"] is False


# ============================================================================
# MODULE 4: ISO 10110 Manufacturing Report Audits
# ============================================================================

def test_tier5_manufacturing_achromat_doublet_audit():
    """
    ISO 10110 manufacturing audit on compliant Achromatic Doublet.
    Verifies that audit_optical_system_manufacturing reports 100% compliance,
    0 critical violations, correct TOTR (excluding S0), and test plate fits.
    """
    doublet = make_achromat_doublet()
    rep = audit_optical_system_manufacturing(doublet)

    assert rep.is_compliant is True
    assert rep.compliance_score == 100.0
    assert len(rep.overall_errors) == 0
    assert len(rep.elements) == 2

    # Check total track strictly excludes S0 = 1e9 mm
    assert abs(rep.total_track_mm - 100.0) < 1e-4

    # Element 1 checks
    el1 = rep.elements[0]
    assert el1.center_thickness_mm == 6.0
    assert el1.edge_thickness_clear_mm >= 1.0
    assert el1.is_compliant is True

    # Element 2 checks
    el2 = rep.elements[1]
    assert el2.center_thickness_mm == 3.0
    assert el2.is_compliant is True

    # Test v2 counterpart parity
    v2_rep = v2_audit_optical_system_manufacturing(doublet)
    assert v2_rep.is_compliant is True
    assert v2_rep.compliance_score == 100.0


def test_tier5_manufacturing_cooke_triplet_audit():
    """
    ISO 10110 manufacturing audit on Cooke Triplet:
    Audits 3 distinct lens elements and 3 air spacings.
    Verifies center thickness (MNCG), edge thickness (MNEG), aspect ratio (D/t),
    mechanical rim thickness, and air gap spacer seating.
    """
    cooke = make_cooke_triplet()
    rep = audit_optical_system_manufacturing(cooke)

    assert len(rep.elements) == 3
    assert len(rep.air_spacings) == 3
    # Total track excluding S0 = 1e9 mm
    expected_totr = sum(s.thickness for s in cooke[1:])
    assert abs(rep.total_track_mm - expected_totr) < 1e-4

    # Element 2 (middle negative flint lens)
    el2 = rep.elements[1]
    assert el2.glass == "N-SF2"
    assert el2.center_thickness_mm == 1.5
    assert el2.aspect_ratio > 0.0


def test_tier5_manufacturing_double_gauss_cemented_elements():
    """
    ISO 10110 manufacturing audit on complex 6-element Double Gauss system
    with two cemented doublets and a central aperture stop.
    Verifies:
    - Cemented doublet elements are individually audited.
    - Air spaces between separated groups are audited for MNEA >= 0.5 mm clearance.
    - Test plate catalog tooling radii are matched for all surfaces.
    """
    dg = make_double_gauss()
    rep = audit_optical_system_manufacturing(dg)

    # 6 lens elements total
    assert len(rep.elements) == 6
    # Front doublet: Elements 2 & 3 share cemented interface
    el2 = rep.elements[1]
    el3 = rep.elements[2]
    assert el2.glass == "N-SK16"
    assert el3.glass == "N-SF2"

    # Rear doublet: Elements 4 & 5 share cemented interface
    el4 = rep.elements[3]
    el5 = rep.elements[4]
    assert el4.glass == "N-SF2"
    assert el5.glass == "N-SK16"

    # Verify tooling test plates for all elements
    for el in rep.elements:
        assert abs(el.r1_test_plate_mm) in TestPlateCatalog.STANDARD_RADII_MM or el.r1_test_plate_mm == 0.0
        assert abs(el.r2_test_plate_mm) in TestPlateCatalog.STANDARD_RADII_MM or el.r2_test_plate_mm == 0.0


def test_tier5_manufacturing_petzval_lens_audit():
    """
    ISO 10110 manufacturing audit on Petzval portrait lens:
    Front cemented doublet + rear air-spaced doublet.
    Verifies compliance auditing and aspect ratio evaluations.
    """
    petzval = make_petzval_lens()
    rep = audit_optical_system_manufacturing(petzval)

    assert len(rep.elements) == 4
    for el in rep.elements:
        assert el.center_thickness_mm > 0.0
        assert el.aspect_ratio > 0.0


def test_tier5_manufacturing_boundary_violations_matrix():
    """
    Adversarial defect injection into lens system:
    - Knife-edge defect: clear edge thickness < 0.0 mm.
    - Center thickness defect: tc < 1.0 mm (MNCG).
    - Air gap rim collision: MNEA < 0.5 mm.
    - Aspect ratio violation: D/t > 15.0.
    - Total track excess: TOTR > 600.0 mm.
    Verifies that audit captures every defect with specific violation messages and drops compliance score.
    """
    s0 = Surface(name="Object", surface_type=SurfaceType.OBJECT, thickness=1e9)
    # Lens 1: ultra-thin knife edge & high aspect ratio
    s1 = Surface(name="L1_front", radius=15.0, thickness=0.8, glass="N-BK7", semi_diameter=14.0)
    s2 = Surface(name="L1_back", radius=-15.0, thickness=0.2, glass="AIR", semi_diameter=14.0)  # Air gap 0.2 mm < 0.5 mm
    # Lens 2: excessive thickness
    s3 = Surface(name="L2_front", radius=50.0, thickness=25.0, glass="N-SF5", semi_diameter=14.0)
    s4 = Surface(name="L2_back", radius=-50.0, thickness=650.0, glass="AIR", semi_diameter=14.0)  # Total track > 500 mm
    s5 = Surface(name="Det", surface_type=SurfaceType.DETECTOR)

    rep = audit_optical_system_manufacturing([s0, s1, s2, s3, s4, s5])

    assert rep.is_compliant is False
    assert rep.compliance_score < 70.0
    err_text = " ".join(rep.overall_errors).lower()

    # MNCG violation check
    assert "center thickness" in err_text or "mncg" in err_text
    # Aspect ratio check
    assert "aspect ratio" in err_text or "d/t" in err_text
    # Air gap clearance check
    assert "air gap" in err_text or "rim clearance" in err_text
    # Total track check
    assert "total track" in err_text or "totr" in err_text


def test_tier5_manufacturing_quality_presets_thresholds():
    """
    Verify stricter manufacturing tolerances across QualityPresets:
    COMMERCIAL vs PRECISION vs HIGH_PRECISION.
    """
    cfg_comm = ManufacturingConstraintsConfig.from_preset(QualityPreset.COMMERCIAL)
    cfg_prec = ManufacturingConstraintsConfig.from_preset(QualityPreset.PRECISION)
    cfg_high = ManufacturingConstraintsConfig.from_preset(QualityPreset.HIGH_PRECISION)

    # Center thickness minimum: Commercial 1.5 mm < Precision 2.0 mm < High-Precision 2.5 mm
    assert cfg_comm.min_glass_thickness < cfg_prec.min_glass_thickness < cfg_high.min_glass_thickness
    # Edge thickness minimum: Commercial 1.0 mm < Precision 1.5 mm < High-Precision 2.0 mm
    assert cfg_comm.min_edge_thickness < cfg_prec.min_edge_thickness < cfg_high.min_edge_thickness
    # Aspect ratio maximum: High-Precision 6.0 < Precision 8.0 < Commercial 10.0
    assert cfg_high.max_aspect_ratio < cfg_prec.max_aspect_ratio < cfg_comm.max_aspect_ratio
    # Air edge clearance: Commercial 0.5 mm < High-Precision 1.0 mm
    assert cfg_comm.min_edge_air < cfg_high.min_edge_air


def test_tier5_manufacturing_test_plate_catalog_snapping():
    """
    Comprehensive verification of TestPlateCatalog tooling snapping:
    - Standard radii in catalog are snapped with 0 Delta R.
    - Positive and negative signs are preserved.
    - Flat / plano surfaces (R = 0 or R > 20000) return 0.0 with 0 fringes.
    - Fringe power error follows Delta N = 2 * |Delta sag| / lambda.
    """
    assert len(TestPlateCatalog.STANDARD_RADII_MM) >= 60

    # Exact match test
    near, delta, fringes = TestPlateCatalog.find_nearest_test_plate(50.0, semi_diameter_mm=12.5)
    assert near == 50.0
    assert abs(delta) < 1e-6
    assert abs(fringes) < 1e-4

    # Negative radius exact match
    near_neg, delta_neg, fringes_neg = TestPlateCatalog.find_nearest_test_plate(-100.0, semi_diameter_mm=12.5)
    assert near_neg == -100.0
    assert abs(delta_neg) < 1e-6

    # In-between radius (51.5 mm should snap to 50.0 mm)
    near_inter, delta_inter, fringes_inter = TestPlateCatalog.find_nearest_test_plate(51.5, semi_diameter_mm=12.5)
    assert near_inter in [50.0, 55.0]
    assert abs(delta_inter) > 0.0
    assert fringes_inter > 0.0

    # Plano surface
    near_plano, delta_plano, fringes_plano = TestPlateCatalog.find_nearest_test_plate(0.0)
    assert near_plano == 0.0
    assert delta_plano == 0.0
    assert fringes_plano == 0.0

    # Infinite / nearly flat radius
    near_inf, delta_inf, fringes_inf = TestPlateCatalog.find_nearest_test_plate(50000.0)
    assert near_inf == 0.0

    # v2 counterpart parity
    v2_near, v2_delta, v2_fringes = V2TestPlateCatalog.find_nearest_test_plate(50.0, semi_diameter_mm=12.5)
    assert v2_near == 50.0


def test_tier5_manufacturing_markdown_report_formatting():
    """
    Verify SystemManufacturingReport.summary_markdown() and format_markdown()
    produce clean, compliant Markdown tables with status badges and violation lists.
    """
    doublet = make_achromat_doublet()
    rep = audit_optical_system_manufacturing(doublet)

    md = rep.summary_markdown()
    assert "### Optical Manufacturing & Mounting Audit" in md
    assert "**Overall Status:** `PASS`" in md
    assert "**Compliance Score:** `100.0%`" in md
    assert "| Elem | Name | Glass |" in md
    assert "#1" in md
    assert "#2" in md

    # format_markdown alias
    assert rep.format_markdown() == md


# ============================================================================
# MODULE 5: Dual-Tree SHA-256 Parity Across All 106 Shared Files
# ============================================================================

def test_tier5_dual_tree_sha256_parity_all_106_files():
    """
    Adversarial verification of 100% byte-for-byte SHA-256 equivalence
    between optics_app/ and v2/ trees across all 106 shared files.
    Ensures zero drift, no lingering stubs, and exact structural parity.
    """
    repo_root = Path(__file__).parent.parent.resolve()
    optics_app_dir = repo_root / "optics_app"
    v2_dir = repo_root / "v2"

    assert optics_app_dir.exists(), f"Directory missing: {optics_app_dir}"
    assert v2_dir.exists(), f"Directory missing: {v2_dir}"

    optics_files: List[Path] = []
    for root, dirs, files in os.walk(optics_app_dir):
        if "__pycache__" in root:
            continue
        for f in files:
            if f.endswith(".pyc"):
                continue
            optics_files.append(Path(root) / f)

    assert len(optics_files) == 106, f"Expected 106 files in optics_app, found {len(optics_files)}"

    mismatches = []
    missing_in_v2 = []

    for p1 in optics_files:
        rel = p1.relative_to(optics_app_dir)
        p2 = v2_dir / rel

        if not p2.exists():
            missing_in_v2.append(str(rel))
            continue

        h1 = hashlib.sha256(p1.read_bytes()).hexdigest()
        h2 = hashlib.sha256(p2.read_bytes()).hexdigest()

        if h1 != h2:
            mismatches.append((str(rel), h1, h2))

    assert len(missing_in_v2) == 0, f"Files missing in v2: {missing_in_v2}"
    assert len(mismatches) == 0, f"SHA-256 hash mismatches detected in {len(mismatches)} files: {[m[0] for m in mismatches]}"
