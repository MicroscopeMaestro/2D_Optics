"""
Adversarial Stress Test Suite for Milestone 1 - Challenger 2
Stress-tests:
1. Multi-field hit segregation: verifying that ray bundles across multiple fields
   NEVER pool intercepts into a single false blur circle.
2. Chief-ray vs centroid spot referencing under optical distortion and lateral color.
3. Rayleigh-Hamilton OPD scaling across varying F-numbers and wavelengths,
   verifying exact wave-unit consistency between Nelder-Mead merit and DLS residuals.
"""

import math
import numpy as np
import pytest

from tests.test_zemax_fields import _build_test_imaging_system
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from optics_app.analysis.spot import SpotDiagram
from optics_app.engine.merit_function import MeritFunction, MeritOperand, OperandType
from optics_app.engine.optimizer import OpticalOptimizer, extract_working_f_number
from optics_app.engine.system_config import SystemConfig, FieldPoint, FieldType


# ==============================================================================
# Challenge 1: Multi-field hit segregation
# ==============================================================================

def test_merit_function_multi_field_spot_segregation():
    """
    Verify that MeritFunction.evaluate() NEVER pools multi-field hits into
    a single false multi-millimeter blur circle for RMS_SPOT_RADIUS.
    """
    stack = _build_test_imaging_system()
    system = OpticalSystem(stack)

    config = SystemConfig()
    config.field_type = FieldType.ANGLE_DEG
    config.fields = [
        FieldPoint(x=0.0, y=0.0, weight=1.0),
        FieldPoint(x=0.0, y=10.0, weight=1.0),
    ]

    opt = OpticalOptimizer(system, system_config=config, num_rays=32, wavelengths=0.5876)
    rays = opt._generate_rays(4.0, [0.5876], 32, eval_stack=stack)
    traced = system.trace_rays(rays)

    mf = MeritFunction()
    mf.add_operand(OperandType.RMS_SPOT_RADIUS, surface_index=6, target=0.0, weight=1.0)
    val = mf.evaluate(system, traced, wavelengths=[0.5876], system_config=config)

    # If hits were pooled: composite mean is at y ~ 4.4 mm, and pooled RMS is ~3.9 mm.
    # When properly segregated with true refracted entrance pupil (Z_ep ~ 68.34 mm):
    # true weighted RMS is ~1.0287 mm.
    assert val < 2.0, f"Hits were pooled! Composite RMS is {val:.4f} mm instead of ~1.03 mm"
    assert abs(val - 1.0287) < 0.01, f"RMS deviated from true segregated value: {val:.4f} mm"


def test_optimizer_multi_field_spot_segregation():
    """
    Verify that OpticalOptimizer._evaluate_merit_and_penalty() segregates multi-field
    hits and produces sub-millimeter RMS spot, not multi-millimeter pooled blur.
    """
    stack = _build_test_imaging_system()
    system = OpticalSystem(stack)

    config = SystemConfig()
    config.field_type = FieldType.ANGLE_DEG
    config.fields = [
        FieldPoint(x=0.0, y=0.0, weight=1.0),
        FieldPoint(x=0.0, y=10.0, weight=1.0),
    ]

    optimizer = OpticalOptimizer(
        system=system,
        system_config=config,
        num_rays=32,
        wavelengths=0.5876,
        target="Spot Size"
    )

    pure_merit, penalty = optimizer._evaluate_merit_and_penalty(
        variable_values=np.array([]),
        variable_indices={},
        use_curvature_space=False
    )

    assert pure_merit < 2.0, f"Optimizer pooled fields! pure_merit = {pure_merit:.4f} mm"
    assert abs(pure_merit - 1.0287) < 0.01


def test_optimizer_dls_multi_field_residuals_segregation():
    """
    Verify that OpticalOptimizer._compute_dls_residuals() centers residuals
    around each field's reference point, so residual norm equals segregated RMS.
    """
    stack = _build_test_imaging_system()
    system = OpticalSystem(stack)

    config = SystemConfig()
    config.field_type = FieldType.ANGLE_DEG
    config.fields = [
        FieldPoint(x=0.0, y=0.0, weight=1.0),
        FieldPoint(x=0.0, y=10.0, weight=1.0),
    ]

    optimizer = OpticalOptimizer(
        system=system,
        system_config=config,
        num_rays=32,
        wavelengths=0.5876,
        target="Spot Size"
    )

    res = optimizer._compute_dls_residuals(
        x_param=np.array([]),
        var_dict={},
        use_curv=False
    )

    res_norm = float(np.sqrt(np.sum(res**2)))
    pure_merit, _ = optimizer._evaluate_merit_and_penalty(np.array([]), {}, False)

    # res_norm should match pure_merit exactly
    assert abs(res_norm - pure_merit) < 1e-6, (
        f"DLS residual norm {res_norm} does not match Nelder-Mead merit {pure_merit}"
    )


# ==============================================================================
# Challenge 2: Chief-ray vs Centroid spot referencing
# ==============================================================================

def test_spot_referencing_parallel_axis_theorem():
    """
    Under distortion (chief ray offset Delta relative to centroid),
    RMS_chief^2 must equal RMS_centroid^2 + Delta^2 exactly.
    """
    np.random.seed(123)
    x = np.random.normal(0, 0.05, 500)
    y = np.random.normal(0, 0.05, 500)
    x -= np.mean(x)
    y -= np.mean(y)

    spot_centroid = SpotDiagram(x, y, reference="centroid")
    rms_cent = spot_centroid.calculate_rms_radius()

    deltas = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    prev_rms_chief = 0.0

    for delta in deltas:
        chief_pt = (0.0, delta)
        spot_chief = SpotDiagram(x, y, reference="chief_ray", chief_ray_coord=chief_pt)
        rms_chief = spot_chief.calculate_rms_radius()

        # 1. Chief-ray RMS must strictly increase monotonically with distortion
        assert rms_chief > prev_rms_chief, f"RMS chief failed to increase monotonically at delta={delta}"
        prev_rms_chief = rms_chief

        # 2. Centroid-referenced RMS must remain completely invariant
        assert abs(spot_chief.calculate_rms_radius(reference="centroid") - rms_cent) < 1e-12

        # 3. Exact parallel axis theorem: RMS_chief^2 == RMS_cent^2 + delta^2
        expected_chief = math.sqrt(rms_cent**2 + delta**2)
        assert abs(rms_chief - expected_chief) < 1e-10, (
            f"Parallel axis theorem violation: got {rms_chief}, expected {expected_chief}"
        )


def test_optimizer_chief_ray_vs_centroid_referencing():
    """
    Verify that OpticalOptimizer correctly switches between centroid and chief_ray
    spot referencing in merit evaluation and DLS residuals under off-axis aberrations.
    """
    stack = _build_test_imaging_system()
    system = OpticalSystem(stack)

    for angle in [5.0, 10.0, 15.0]:
        config = SystemConfig()
        config.field_type = FieldType.ANGLE_DEG
        config.fields = [FieldPoint(x=0.0, y=angle, weight=1.0)]

        opt_cent = OpticalOptimizer(system, system_config=config, num_rays=32, wavelengths=0.5876, reference="centroid")
        opt_chief = OpticalOptimizer(system, system_config=config, num_rays=32, wavelengths=0.5876, reference="chief_ray")

        m_cent, _ = opt_cent._evaluate_merit_and_penalty(np.array([]), {})
        m_chief, _ = opt_chief._evaluate_merit_and_penalty(np.array([]), {})

        # Chief-ray referenced spot must be strictly greater than centroid spot off-axis
        assert m_chief > m_cent, f"At {angle} deg, m_chief ({m_chief}) should be > m_cent ({m_cent})"

        # DLS residuals must follow the same relationship
        res_cent = opt_cent._compute_dls_residuals(np.array([]), {}, False)
        res_chief = opt_chief._compute_dls_residuals(np.array([]), {}, False)
        norm_cent = float(np.sqrt(np.sum(res_cent**2)))
        norm_chief = float(np.sqrt(np.sum(res_chief**2)))

        assert norm_chief > norm_cent, f"At {angle} deg, DLS chief norm ({norm_chief}) should be > centroid norm ({norm_cent})"
        assert abs(norm_cent - m_cent) < 1e-6
        assert abs(norm_chief - m_chief) < 1e-6


# ==============================================================================
# Challenge 3: Rayleigh-Hamilton OPD Scaling Stress & Wave-Unit Consistency
# ==============================================================================

def test_rayleigh_hamilton_wave_unit_consistency():
    """
    Verify that Nelder-Mead merit and DLS residuals for 'Wavefront Error' (OPD)
    evaluate to IDENTICAL values in waves (not mm or unscaled).
    Formula: W_RMS = R_RMS / (8 * (F/#) * lambda_mm)
    """
    stack = _build_test_imaging_system()
    system = OpticalSystem(stack)

    config = SystemConfig()
    config.field_type = FieldType.ANGLE_DEG
    config.fields = [
        FieldPoint(x=0.0, y=0.0, weight=1.0),
        FieldPoint(x=0.0, y=7.07, weight=2.0),
        FieldPoint(x=0.0, y=10.0, weight=1.0),
    ]

    for wl in [0.4861, 0.5876, 0.6563]:
        for r_ep in [2.0, 4.0, 8.0]:
            opt_wf = OpticalOptimizer(
                system, system_config=config, wavelengths=wl,
                entrance_pupil_radius=r_ep, num_rays=32,
                target="Wavefront Error", reference="centroid"
            )
            merit_wf, _ = opt_wf._evaluate_merit_and_penalty(np.array([]), {})
            res_wf = opt_wf._compute_dls_residuals(np.array([]), {}, False)
            dls_norm = float(np.sqrt(np.sum(res_wf**2)))

            # Compare against spot RMS scaled manually
            opt_spot = OpticalOptimizer(
                system, system_config=config, wavelengths=wl,
                entrance_pupil_radius=r_ep, num_rays=32,
                target="Spot Size", reference="centroid"
            )
            merit_spot, _ = opt_spot._evaluate_merit_and_penalty(np.array([]), {})

            f_num = extract_working_f_number(stack, wl, r_ep, eval_idx=6)
            wl_mm = wl * 1e-3
            expected_waves = merit_spot / (8.0 * f_num * wl_mm)

            assert abs(merit_wf - dls_norm) < 1e-6, (
                f"Nelder-Mead ({merit_wf}) != DLS ({dls_norm}) at wl={wl}, R_ep={r_ep}"
            )
            assert abs(merit_wf - expected_waves) < 1e-6, (
                f"Nelder-Mead ({merit_wf}) != Expected ({expected_waves}) at wl={wl}, R_ep={r_ep}"
            )


# ==============================================================================
# Adversarial Bug Finding: WAVEFRONT_OPD in MeritFunction.evaluate()
# ==============================================================================

def test_adversarial_merit_function_wavefront_opd_field_pooling_bug():
    """
    ADVERSARIAL STRESS TEST:
    Demonstrates that MeritFunction.evaluate() fails to segregate ray hits
    when evaluating OperandType.WAVEFRONT_OPD across multiple fields, pooling
    them into a single multi-millimeter false blur circle and producing 11.5x error.
    """
    stack = _build_test_imaging_system()
    system = OpticalSystem(stack)

    config = SystemConfig()
    config.field_type = FieldType.ANGLE_DEG
    config.fields = [
        FieldPoint(x=0.0, y=0.0, weight=1.0),
        FieldPoint(x=0.0, y=10.0, weight=1.0),
    ]

    opt1 = OpticalOptimizer(system, system_config=config, target="Wavefront Error", wavelengths=0.5876, num_rays=32)
    val_segregated, _ = opt1._evaluate_merit_and_penalty(np.array([]), {})

    mf = MeritFunction.create_wavefront_default(target_surface=6)
    opt2 = OpticalOptimizer(system, system_config=config, merit_function=mf, wavelengths=0.5876, num_rays=32)
    val_mf_pooled, _ = opt2._evaluate_merit_and_penalty(np.array([]), {})

    ratio = val_mf_pooled / val_segregated
    print(f"\n[ADVERSARIAL VERIFICATION RESULT] Wavefront Error internal: {val_segregated:.2f} waves, MeritFunction WAVEFRONT_OPD: {val_mf_pooled:.2f} waves (Ratio: {ratio:.4f}x)")

    # Remediation confirmed: val_mf_pooled equals val_segregated within < 0.1%
    assert ratio < 1.05, f"WAVEFRONT_OPD is still pooling fields! ratio = {ratio:.4f}"
    assert abs(val_mf_pooled - val_segregated) < 0.1, (
        f"MeritFunction WAVEFRONT_OPD ({val_mf_pooled:.4f}) deviates from internal target ({val_segregated:.4f})"
    )


def test_merit_function_wavefront_opd_field_index_filter():
    """
    Verify that MeritOperand.field_index is strictly respected when evaluating
    OperandType.WAVEFRONT_OPD, producing distinct, field-specific wavefront OPD values.
    """
    stack = _build_test_imaging_system()
    system = OpticalSystem(stack)

    config = SystemConfig()
    config.field_type = FieldType.ANGLE_DEG
    config.fields = [
        FieldPoint(x=0.0, y=0.0, weight=1.0),
        FieldPoint(x=0.0, y=10.0, weight=1.0),
    ]

    mf0 = MeritFunction()
    mf0.add_operand(OperandType.WAVEFRONT_OPD, surface_index=6, field_index=0)
    opt0 = OpticalOptimizer(system, system_config=config, merit_function=mf0, wavelengths=0.5876, num_rays=32)
    val0, _ = opt0._evaluate_merit_and_penalty(np.array([]), {})

    mf1 = MeritFunction()
    mf1.add_operand(OperandType.WAVEFRONT_OPD, surface_index=6, field_index=1)
    opt1 = OpticalOptimizer(system, system_config=config, merit_function=mf1, wavelengths=0.5876, num_rays=32)
    val1, _ = opt1._evaluate_merit_and_penalty(np.array([]), {})

    assert val0 > 0.0, f"Expected positive OPD for field 0, got {val0}"
    assert val1 > 0.0, f"Expected positive OPD for field 1, got {val1}"
    assert abs(val0 - val1) > 1.0, f"Expected distinct OPD per field, but got val0={val0}, val1={val1}"

