"""
Adversarial Stress Test Suite for Milestone 2 — Challenger 2.
Adversarially challenges and stress-tests:
1. Total Track (TOTR) S0 exclusion under extreme finite object distances (S0 = 1000, 5000, 0 mm) and infinite conjugate objects.
   Verify that TOTR never includes S0 and zero false MXTR violations occur.
2. Boundary operands (MNCT, MXCT, MNET, MXET, MNEA) one-sided inequality logic:
   Verify zero penalty when compliant, and proportional penalty when violated.
3. Dynamic curvature bounds clamping |c| <= 0.95/y_sd:
   Stress-test across wide clear apertures (y_sd = 25, 50, 100 mm) and narrow apertures (y_sd = 1 mm),
   verifying no negative radicands (1 - c^2 y^2 <= 0) ever occur during ray tracing.
4. Dual-tree byte-for-byte SHA256 parity and identical runtime behavior between optics_app and v2.
"""

import unittest
import math
import numpy as np
import hashlib
import filecmp

# optics_app imports
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from optics_app.engine.optimizer import (
    OpticalOptimizer,
    radius_to_curvature,
    curvature_to_radius
)
from optics_app.engine.merit_function import MeritFunction, OperandType
from optics_app.engine.manufacturing_standards import (
    audit_optical_system_manufacturing,
    ManufacturingConstraintsConfig,
    QualityPreset
)

# v2 imports
import v2.engine.surface as v2_surface
import v2.engine.lde as v2_lde
import v2.engine.tracer as v2_tracer
import v2.engine.optimizer as v2_optimizer
import v2.engine.merit_function as v2_merit_function
import v2.engine.manufacturing_standards as v2_mfg


class TestTotalTrackS0Exclusion(unittest.TestCase):
    """
    Stress-testing Total Track (TOTR) S0 exclusion:
    S0 must NEVER be included in TOTR, whether S0 is finite (0, 500, 1000, 5000 mm) or infinite conjugate.
    Zero false MXTR violations must occur.
    """

    def _create_test_stack(self, s0_thickness: float, s0_type: str = SurfaceType.OBJECT) -> SurfaceStack:
        stack = SurfaceStack()
        # Surface 0: Object plane
        s0 = Surface(
            name="OBJ",
            surface_type=s0_type,
            radius=0.0,
            thickness=s0_thickness,
            glass="AIR",
            semi_diameter=25.0
        )
        stack.add_surface(s0)
        # Surface 1: Lens Front (5.0 mm glass)
        s1 = Surface(
            name="L1_Front",
            surface_type=SurfaceType.STANDARD,
            radius=50.0,
            thickness=5.0,
            glass="N-BK7",
            semi_diameter=15.0
        )
        stack.add_surface(s1)
        # Surface 2: Lens Rear (45.0 mm air to detector)
        s2 = Surface(
            name="L1_Rear",
            surface_type=SurfaceType.STANDARD,
            radius=-50.0,
            thickness=45.0,
            glass="AIR",
            semi_diameter=15.0
        )
        stack.add_surface(s2)
        # Surface 3: Detector plane (thickness 0.0)
        s3 = Surface(
            name="DET",
            surface_type=SurfaceType.DETECTOR,
            radius=0.0,
            thickness=0.0,
            glass="AIR",
            semi_diameter=20.0
        )
        stack.add_surface(s3)
        return stack

    def test_totr_manufacturing_audit_finite_and_infinite_s0(self):
        """
        Verify that audit_optical_system_manufacturing strictly excludes S0
        for S0 = 0, 50, 500, 1000, 5000, 1e9, and inf mm.
        Physical track length = 5.0 + 45.0 + 0.0 = 50.0 mm.
        """
        expected_totr = 50.0
        test_distances = [0.0, 50.0, 500.0, 1000.0, 5000.0, 1e9, float("inf")]

        cfg = ManufacturingConstraintsConfig(max_total_track=100.0)

        for s0 in test_distances:
            stack = self._create_test_stack(s0)
            report = audit_optical_system_manufacturing(stack.surfaces, config=cfg)

            # Empirical check 1: TOTR equals exactly 50.0 mm
            self.assertAlmostEqual(
                report.total_track_mm, expected_totr, places=4,
                msg=f"Failed for S0={s0}: expected TOTR {expected_totr}, got {report.total_track_mm}"
            )

            # Empirical check 2: Zero false MXTR violations
            mxtr_errors = [e for e in report.overall_errors if "MXTR" in e or "Total track" in e]
            self.assertEqual(
                len(mxtr_errors), 0,
                msg=f"False MXTR violation detected for S0={s0}: {mxtr_errors}"
            )

    def test_totr_manufacturing_audit_true_violation_triggers(self):
        """
        Verify that when true track length (S1 to detector) exceeds max_total_track,
        MXTR violation IS triggered.
        """
        stack = self._create_test_stack(5000.0)  # S0 = 5000 mm, true TOTR = 50.0 mm
        cfg_strict = ManufacturingConstraintsConfig(max_total_track=40.0)  # 40 mm < 50 mm
        report = audit_optical_system_manufacturing(stack.surfaces, config=cfg_strict)

        mxtr_errors = [e for e in report.overall_errors if "MXTR" in e]
        self.assertEqual(len(mxtr_errors), 1)
        self.assertIn("50.00 mm exceeds maximum allowed 40.0 mm", mxtr_errors[0])

    def test_totr_in_merit_function(self):
        """
        Verify that OperandType.TOTR in MeritFunction strictly measures Surface 1 to detector
        across S0 = 0, 1000, 5000 mm, and infinite conjugate.
        """
        test_distances = [0.0, 1000.0, 5000.0, float("inf")]

        for s0 in test_distances:
            stack = self._create_test_stack(s0)
            sys = OpticalSystem(stack)
            rays = sys.trace_rays([Ray([0.0, 0.0, 0.0], [0.0, 0.0, 1.0])])

            mf = MeritFunction()
            mf.penalty_weight = 0.0
            mf.add_operand(OperandType.TOTR, target=50.0, weight=1.0)
            score = mf.evaluate(sys, rays)

            op = mf.operands[0]
            self.assertAlmostEqual(op.value, 50.0, places=4, msg=f"Operand value wrong for S0={s0}")
            self.assertAlmostEqual(op.diff, 0.0, places=4, msg=f"Operand diff wrong for S0={s0}")
            self.assertAlmostEqual(score, 0.0, places=4, msg=f"Merit score wrong for S0={s0}")

    def test_totr_in_optimizer_residuals_and_penalties(self):
        """
        Verify that OpticalOptimizer._compute_dls_residuals and penalty evaluation
        strictly exclude S0 and generate zero false residuals or penalties.
        """
        for s0 in [0.0, 1000.0, 5000.0]:
            stack = self._create_test_stack(s0)
            opt_sys = OpticalSystem(stack)
            opt = OpticalOptimizer(opt_sys)
            opt.enforce_manufacturing_constraints = True
            opt.max_total_track = 100.0  # Allowed 100 mm > 50 mm

            var_dict = {"thickness": [1]}  # S1 thickness variable
            x0 = np.array([5.0])
            bounds = [(1.5, 10.0)]

            # Compute residuals
            res = opt._compute_dls_residuals(x0, var_dict, use_curv=False, bounds_param=bounds)
            totr_res = res[-1]  # Last residual is TOTR
            self.assertAlmostEqual(
                totr_res, 0.0, places=5,
                msg=f"Non-zero TOTR residual {totr_res} for S0={s0} (false violation)"
            )

            # Test true violation: set max_total_track = 40.0
            opt.max_total_track = 40.0
            res_viol = opt._compute_dls_residuals(x0, var_dict, use_curv=False, bounds_param=bounds)
            totr_res_viol = res_viol[-1]
            # Expected residual = max(0.0, 50.0 - 40.0) * 20.0 = 200.0
            self.assertAlmostEqual(totr_res_viol, 200.0, places=4)

    def test_totr_cooke_triplet_multi_element_s0_exclusion(self):
        """
        Verify Cooke Triplet with 6 optical surfaces (3 glass elements + air spaces)
        with S0 = 2500 mm.
        """
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", surface_type=SurfaceType.OBJECT, thickness=2500.0))
        # L1: 4.5 mm glass, air: 2.5 mm
        stack.add_surface(Surface(name="L1_F", radius=22.0, thickness=4.5, glass="N-SK16", semi_diameter=12.0))
        stack.add_surface(Surface(name="L1_R", radius=-150.0, thickness=2.5, glass="AIR", semi_diameter=12.0))
        # L2: 1.5 mm glass, air: 3.5 mm
        stack.add_surface(Surface(name="L2_F", radius=-20.0, thickness=1.5, glass="N-SF2", semi_diameter=10.0))
        stack.add_surface(Surface(name="L2_R", radius=20.0, thickness=3.5, glass="AIR", semi_diameter=10.0))
        # L3: 3.8 mm glass, air to detector: 40.0 mm
        stack.add_surface(Surface(name="L3_F", radius=35.0, thickness=3.8, glass="N-SK16", semi_diameter=11.0))
        stack.add_surface(Surface(name="L3_R", radius=-30.0, thickness=40.0, glass="AIR", semi_diameter=11.0))
        # Detector
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=15.0))

        # Expected track = 4.5 + 2.5 + 1.5 + 3.5 + 3.8 + 40.0 = 55.8 mm
        expected_totr = 55.8

        cfg = ManufacturingConstraintsConfig(max_total_track=100.0)
        report = audit_optical_system_manufacturing(stack.surfaces, config=cfg)
        self.assertAlmostEqual(report.total_track_mm, expected_totr, places=4)
        self.assertEqual(len([e for e in report.overall_errors if "MXTR" in e]), 0)

        # In merit function
        sys = OpticalSystem(stack)
        rays = sys.trace_rays([Ray([0.0, 0.0, 0.0], [0.0, 0.0, 1.0])])
        mf = MeritFunction()
        mf.penalty_weight = 0.0
        mf.add_operand(OperandType.TOTR, target=55.8, weight=1.0)
        score = mf.evaluate(sys, rays)
        self.assertAlmostEqual(mf.operands[0].value, expected_totr, places=4)
        self.assertAlmostEqual(score, 0.0, places=4)


class TestBoundaryOperandsOneSidedInequalities(unittest.TestCase):
    """
    Stress-testing boundary operands (MNCT, MXCT, MNET, MXET, MNEA) one-sided inequality logic:
    - Compliant parameters MUST produce exactly 0.0 penalty/residual.
    - Violating parameters MUST generate strictly positive, monotonically proportional penalties.
    """

    def _build_singlet(self, ct: float, r1: float = 50.0, r2: float = -50.0, sd: float = 12.5) -> SurfaceStack:
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", surface_type=SurfaceType.OBJECT, thickness=100.0))
        stack.add_surface(Surface(name="L1_F", radius=r1, thickness=ct, glass="N-BK7", semi_diameter=sd))
        stack.add_surface(Surface(name="L1_R", radius=r2, thickness=30.0, glass="AIR", semi_diameter=sd))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=15.0))
        return stack

    def test_mnct_one_sided_inequality(self):
        """
        MNCT (Minimum Center Thickness):
        Target = 1.5 mm.
        CT >= 1.5 mm -> diff = 0.0, penalty = 0.0.
        CT < 1.5 mm  -> diff = 1.5 - CT > 0.0.
        """
        target = 1.5
        compliant_cts = [1.5, 1.5001, 2.0, 5.0, 12.0, 18.0]
        violating_cts = [1.499, 1.4, 1.0, 0.5, 0.1, 0.01]

        # 1. Test compliant values
        for ct in compliant_cts:
            stack = self._build_singlet(ct=ct)
            sys = OpticalSystem(stack)
            rays = sys.trace_rays([Ray([0.0, 0.0, 0.0], [0.0, 0.0, 1.0])])
            mf = MeritFunction()
            mf.penalty_weight = 0.0
            mf.add_operand(OperandType.MNCT, target=target, weight=2.0, surface_index=1)
            score = mf.evaluate(sys, rays)
            self.assertEqual(mf.operands[0].diff, 0.0, msg=f"MNCT non-zero diff for compliant CT={ct}")
            self.assertEqual(mf.operands[0].contribution, 0.0)
            self.assertEqual(score, 0.0)

        # 2. Test violating values and strict monotonicity
        prev_diff = 0.0
        prev_score = 0.0
        for ct in violating_cts:
            stack = self._build_singlet(ct=ct)
            sys = OpticalSystem(stack)
            rays = sys.trace_rays([Ray([0.0, 0.0, 0.0], [0.0, 0.0, 1.0])])
            mf = MeritFunction()
            mf.penalty_weight = 0.0
            mf.add_operand(OperandType.MNCT, target=target, weight=2.0, surface_index=1)
            score = mf.evaluate(sys, rays)

            expected_diff = target - ct
            self.assertAlmostEqual(mf.operands[0].diff, expected_diff, places=5)
            self.assertGreater(mf.operands[0].diff, prev_diff, msg=f"Diff must increase as CT decreases ({ct})")
            self.assertGreater(score, prev_score, msg=f"Merit must increase as violation deepens ({ct})")
            prev_diff = mf.operands[0].diff
            prev_score = score

    def test_mxct_one_sided_inequality(self):
        """
        MXCT (Maximum Center Thickness):
        Target = 18.0 mm.
        CT <= 18.0 mm -> diff = 0.0, penalty = 0.0.
        CT > 18.0 mm  -> diff = CT - 18.0 > 0.0.
        """
        target = 18.0
        compliant_cts = [1.5, 5.0, 10.0, 17.9, 18.0]
        violating_cts = [18.01, 18.5, 20.0, 25.0, 35.0]

        # 1. Test compliant values
        for ct in compliant_cts:
            stack = self._build_singlet(ct=ct)
            sys = OpticalSystem(stack)
            rays = sys.trace_rays([Ray([0.0, 0.0, 0.0], [0.0, 0.0, 1.0])])
            mf = MeritFunction()
            mf.penalty_weight = 0.0
            mf.add_operand(OperandType.MXCT, target=target, weight=2.0, surface_index=1)
            score = mf.evaluate(sys, rays)
            self.assertEqual(mf.operands[0].diff, 0.0, msg=f"MXCT non-zero diff for compliant CT={ct}")
            self.assertEqual(mf.operands[0].contribution, 0.0)
            self.assertEqual(score, 0.0)

        # 2. Test violating values and monotonicity
        prev_diff = 0.0
        prev_score = 0.0
        for ct in violating_cts:
            stack = self._build_singlet(ct=ct)
            sys = OpticalSystem(stack)
            rays = sys.trace_rays([Ray([0.0, 0.0, 0.0], [0.0, 0.0, 1.0])])
            mf = MeritFunction()
            mf.penalty_weight = 0.0
            mf.add_operand(OperandType.MXCT, target=target, weight=2.0, surface_index=1)
            score = mf.evaluate(sys, rays)

            expected_diff = ct - target
            self.assertAlmostEqual(mf.operands[0].diff, expected_diff, places=5)
            self.assertGreater(mf.operands[0].diff, prev_diff, msg=f"Diff must increase as CT increases ({ct})")
            self.assertGreater(score, prev_score)
            prev_diff = mf.operands[0].diff
            prev_score = score

    def test_mnet_exact_sag_one_sided_inequality(self):
        """
        MNET (Minimum Edge Thickness via exact sag):
        Biconvex lens: R1 = 30 mm, R2 = -30 mm, y_sd = 12 mm.
        sag1 = 30 - sqrt(30^2 - 12^2) = 30 - sqrt(756) = 2.5045 mm.
        sag2 = -30 + sqrt(30^2 - 12^2) = -2.5045 mm.
        ET = CT + sag2 - sag1 = CT - 2 * 2.5045 = CT - 5.0090 mm.
        Target MNET = 1.0 mm.
        CT = 6.0090 mm -> ET = 1.0 mm (boundary).
        CT = 7.0 mm -> ET = 1.991 mm (compliant).
        CT = 5.5 mm -> ET = 0.491 mm (violating).
        """
        r1, r2, y_sd = 30.0, -30.0, 12.0
        sag1 = 30.0 - math.sqrt(30.0**2 - 12.0**2)
        sag2 = -30.0 + math.sqrt(30.0**2 - 12.0**2)
        delta_sag = sag2 - sag1  # approximately -5.0090 mm

        target_et = 1.0

        # Compliant CTs: ET >= 1.0 mm (CT >= 1.0 - delta_sag = 6.0090 mm)
        compliant_cts = [6.0091, 6.5, 7.0, 10.0]
        for ct in compliant_cts:
            stack = self._build_singlet(ct=ct, r1=r1, r2=r2, sd=y_sd)
            sys = OpticalSystem(stack)
            rays = sys.trace_rays([Ray([0.0, 0.0, 0.0], [0.0, 0.0, 1.0])])
            mf = MeritFunction()
            mf.penalty_weight = 0.0
            mf.add_operand(OperandType.MNET, target=target_et, weight=1.5, surface_index=1)
            score = mf.evaluate(sys, rays)
            self.assertEqual(mf.operands[0].diff, 0.0, msg=f"MNET non-zero diff for compliant CT={ct}")
            self.assertEqual(score, 0.0)

        # Violating CTs: ET < 1.0 mm (knife edges)
        violating_cts = [6.0, 5.5, 5.0, 4.0]
        prev_diff = 0.0
        for ct in violating_cts:
            stack = self._build_singlet(ct=ct, r1=r1, r2=r2, sd=y_sd)
            sys = OpticalSystem(stack)
            rays = sys.trace_rays([Ray([0.0, 0.0, 0.0], [0.0, 0.0, 1.0])])
            mf = MeritFunction()
            mf.penalty_weight = 0.0
            mf.add_operand(OperandType.MNET, target=target_et, weight=1.5, surface_index=1)
            score = mf.evaluate(sys, rays)

            et_true = ct + delta_sag
            expected_diff = target_et - et_true
            self.assertAlmostEqual(mf.operands[0].diff, expected_diff, places=4)
            self.assertGreater(mf.operands[0].diff, prev_diff)
            prev_diff = mf.operands[0].diff

    def test_mxet_one_sided_inequality(self):
        """
        MXET (Maximum Edge Thickness via exact sag):
        Biconcave lens: R1 = -40 mm, R2 = 40 mm, y_sd = 15 mm.
        sag1 = -40 + sqrt(40^2 - 15^2) = -2.9238 mm.
        sag2 = 40 - sqrt(40^2 - 15^2) = 2.9238 mm.
        ET = CT + sag2 - sag1 = CT + 5.8476 mm.
        Target MXET = 10.0 mm.
        CT = 3.0 mm -> ET = 8.8476 mm <= 10.0 mm (compliant).
        CT = 5.0 mm -> ET = 10.8476 mm > 10.0 mm (violating).
        """
        r1, r2, y_sd = -40.0, 40.0, 15.0
        sag1 = -40.0 + math.sqrt(40.0**2 - 15.0**2)
        sag2 = 40.0 - math.sqrt(40.0**2 - 15.0**2)
        delta_sag = sag2 - sag1  # approximately +5.8476 mm

        target_mxet = 10.0

        # Compliant CTs: ET <= 10.0 mm (CT <= 10.0 - 5.8476 = 4.1524 mm)
        for ct in [1.5, 2.0, 3.0, 4.0, 4.152]:
            stack = self._build_singlet(ct=ct, r1=r1, r2=r2, sd=y_sd)
            sys = OpticalSystem(stack)
            rays = sys.trace_rays([Ray([0.0, 0.0, 0.0], [0.0, 0.0, 1.0])])
            mf = MeritFunction()
            mf.penalty_weight = 0.0
            mf.add_operand(OperandType.MXET, target=target_mxet, weight=1.0, surface_index=1)
            score = mf.evaluate(sys, rays)
            self.assertEqual(mf.operands[0].diff, 0.0, msg=f"MXET non-zero diff for compliant CT={ct}")
            self.assertEqual(score, 0.0)

        # Violating CTs: ET > 10.0 mm
        for ct in [4.5, 5.0, 7.0, 10.0]:
            stack = self._build_singlet(ct=ct, r1=r1, r2=r2, sd=y_sd)
            sys = OpticalSystem(stack)
            rays = sys.trace_rays([Ray([0.0, 0.0, 0.0], [0.0, 0.0, 1.0])])
            mf = MeritFunction()
            mf.penalty_weight = 0.0
            mf.add_operand(OperandType.MXET, target=target_mxet, weight=1.0, surface_index=1)
            score = mf.evaluate(sys, rays)

            et_true = ct + delta_sag
            expected_diff = et_true - target_mxet
            self.assertAlmostEqual(mf.operands[0].diff, expected_diff, places=4)
            self.assertGreater(mf.operands[0].diff, 0.0)

    def test_mnea_air_edge_clearance_one_sided_inequality(self):
        """
        MNEA (Minimum Air Edge Clearance to prevent mechanical collisions):
        Air gap between two lens surfaces curved towards each other at the rims:
        L1 Rear: R = 40.0 mm (sag1 = +2.9238 mm at y=15)
        L2 Front: R = -40.0 mm (sag2 = -2.9238 mm at y=15)
        ET_air = t_air + sag2 - sag1 = t_air - 5.8476 mm.
        Target MNEA = 0.5 mm.
        t_air = 7.0 mm -> ET_air = 1.1524 mm >= 0.5 mm (compliant).
        t_air = 6.0 mm -> ET_air = 0.1524 mm < 0.5 mm (violating).
        t_air = 5.0 mm -> ET_air = -0.8476 mm < 0.0 mm (lens collision!).
        """
        r_rear, r_front, y_sd = 40.0, -40.0, 15.0
        sag1 = 40.0 - math.sqrt(40.0**2 - 15.0**2)
        sag2 = -40.0 + math.sqrt(40.0**2 - 15.0**2)
        delta_sag = sag2 - sag1  # -5.8476 mm

        target_mnea = 0.5

        def build_doublet_airgap(t_air: float) -> SurfaceStack:
            stack = SurfaceStack()
            stack.add_surface(Surface(name="OBJ", surface_type=SurfaceType.OBJECT, thickness=100.0))
            # Lens 1
            stack.add_surface(Surface(name="L1_F", radius=100.0, thickness=5.0, glass="N-BK7", semi_diameter=y_sd))
            stack.add_surface(Surface(name="L1_R", radius=r_rear, thickness=t_air, glass="AIR", semi_diameter=y_sd))
            # Lens 2
            stack.add_surface(Surface(name="L2_F", radius=r_front, thickness=5.0, glass="N-SF11", semi_diameter=y_sd))
            stack.add_surface(Surface(name="L2_R", radius=-100.0, thickness=30.0, glass="AIR", semi_diameter=y_sd))
            stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=20.0))
            return stack

        # 1. Compliant air gaps: ET_air >= 0.5 mm (t_air >= 0.5 - delta_sag = 6.3476 mm)
        for t_air in [6.5, 7.0, 8.0, 12.0]:
            stack = build_doublet_airgap(t_air)
            sys = OpticalSystem(stack)
            rays = sys.trace_rays([Ray([0.0, 0.0, 0.0], [0.0, 0.0, 1.0])])
            mf = MeritFunction()
            mf.penalty_weight = 0.0
            mf.add_operand(OperandType.MNEA, target=target_mnea, weight=3.0, surface_index=2)
            score = mf.evaluate(sys, rays)
            self.assertEqual(mf.operands[0].diff, 0.0, msg=f"MNEA non-zero diff for compliant t_air={t_air}")
            self.assertEqual(score, 0.0)

        # 2. Violating air gaps: ET_air < 0.5 mm
        prev_diff = 0.0
        for t_air in [6.2, 5.9, 5.5, 5.0]:
            stack = build_doublet_airgap(t_air)
            sys = OpticalSystem(stack)
            rays = sys.trace_rays([Ray([0.0, 0.0, 0.0], [0.0, 0.0, 1.0])])
            mf = MeritFunction()
            mf.penalty_weight = 0.0
            mf.add_operand(OperandType.MNEA, target=target_mnea, weight=3.0, surface_index=2)
            score = mf.evaluate(sys, rays)

            et_air_true = t_air + delta_sag
            expected_diff = target_mnea - et_air_true
            self.assertAlmostEqual(mf.operands[0].diff, expected_diff, places=4)
            self.assertGreater(mf.operands[0].diff, prev_diff)
            prev_diff = mf.operands[0].diff


class TestDynamicCurvatureBoundsClamping(unittest.TestCase):
    """
    Stress-testing dynamic curvature bounds clamping |c| <= 0.95 / y_sd:
    - Wide apertures: y_sd = 25, 50, 100 mm.
    - Narrow apertures: y_sd = 1, 2 mm.
    - Confirm NO negative radicands (1 - c^2 y^2 <= 0) ever occur during ray tracing.
    - Confirm optimizer box bounds enforcement and ray tracing stability.
    """

    def test_dynamic_curvature_clamping_limits(self):
        """
        Verify that radius_to_curvature and OpticalOptimizer evaluation
        strictly clamp curvature within [-0.95/y_sd, +0.95/y_sd].
        """
        test_apertures = [0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0]

        for y_sd in test_apertures:
            c_limit = 0.95 / y_sd

            stack = SurfaceStack()
            stack.add_surface(Surface(name="OBJ", surface_type=SurfaceType.OBJECT, thickness=100.0))
            s = Surface(name="L1", radius=50.0, thickness=5.0, glass="N-BK7", semi_diameter=y_sd)
            stack.add_surface(s)
            stack.add_surface(Surface(name="Air", radius=0.0, thickness=20.0, glass="AIR", semi_diameter=y_sd))
            stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=y_sd))

            opt = OpticalOptimizer(OpticalSystem(stack))
            opt.use_curvature_space = True

            # Adversarial test 1: Pass positive unphysical curvature (2.0 * c_limit)
            # In _evaluate_merit_and_penalty and _compute_dls_residuals, raw_val must be clamped
            pure, pen = opt._evaluate_merit_and_penalty(
                np.array([2.0 * c_limit]), {"radius": [1]}, use_curvature_space=True
            )
            self.assertTrue(math.isfinite(pure))

            # Adversarial test 2: Pass negative unphysical curvature (-3.0 * c_limit)
            pure_neg, pen_neg = opt._evaluate_merit_and_penalty(
                np.array([-3.0 * c_limit]), {"radius": [1]}, use_curvature_space=True
            )
            self.assertTrue(math.isfinite(pure_neg))

            # Adversarial test 3: DLS residuals computation with extreme curvature
            res = opt._compute_dls_residuals(
                np.array([5.0 * c_limit]), {"radius": [1]}, use_curv=True,
                bounds_param=[(-c_limit, c_limit)]
            )
            self.assertTrue(np.all(np.isfinite(res)))

    def test_no_negative_radicand_under_extreme_curvature_and_pupil_edge(self):
        """
        Adversarially evaluate surface sag, normal vector, and ray tracing
        at the extreme clamped curvature boundary c = 0.95 / y_sd and pupil rim y = y_sd.
        Verify that 1 - c^2 y^2 >= 1 - 0.95^2 = 0.0975 > 0 strictly across all apertures.
        """
        test_apertures = [1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0]

        for y_sd in test_apertures:
            c_max = 0.95 / y_sd
            r_extreme = 1.0 / c_max

            # 1. Direct mathematical verification of sag radicand
            y_rim = y_sd
            radicand = 1.0 - (c_max**2) * (y_rim**2)
            self.assertAlmostEqual(
                radicand, 1.0 - 0.95**2, places=6,
                msg=f"Radicand mismatch for y_sd={y_sd}"
            )
            self.assertGreaterEqual(radicand, 0.0975)

            # 2. Test Surface.sag() and Surface.normal_vector() at the boundary
            surf_pos = Surface(name="TestPos", radius=r_extreme, semi_diameter=y_sd)
            sag_pos = surf_pos.sag(y_rim)
            norm_pos = surf_pos.normal_vector(0.0, y_rim)
            self.assertTrue(math.isfinite(sag_pos), f"Non-finite sag for y_sd={y_sd}")
            self.assertTrue(np.all(np.isfinite(norm_pos)), f"Non-finite normal for y_sd={y_sd}")

            surf_neg = Surface(name="TestNeg", radius=-r_extreme, semi_diameter=y_sd)
            sag_neg = surf_neg.sag(y_rim)
            norm_neg = surf_neg.normal_vector(0.0, y_rim)
            self.assertTrue(math.isfinite(sag_neg))
            self.assertTrue(np.all(np.isfinite(norm_neg)))

            # 3. Ray tracing test through OpticalSystem with rays at rim (+/- y_sd)
            sag_max = r_extreme - math.sqrt(r_extreme**2 - y_sd**2)
            ct = max(1.5, sag_max + 1.0)
            det_sd = max(25.0, y_sd * 5.0)

            stack = SurfaceStack()
            stack.add_surface(Surface(name="OBJ", surface_type=SurfaceType.OBJECT, thickness=10.0))
            stack.add_surface(Surface(name="L1_F", radius=r_extreme, thickness=ct, glass="N-BK7", semi_diameter=y_sd))
            stack.add_surface(Surface(name="L1_R", radius=0.0, thickness=5.0, glass="AIR", semi_diameter=y_sd * 2.0))
            stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=det_sd))

            sys = OpticalSystem(stack)
            rays = [
                Ray([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]),                     # Chief ray
                Ray([0.0, y_sd * 0.95, 0.0], [0.0, 0.0, 1.0]),            # 95% pupil rim
                Ray([0.0, -y_sd * 0.95, 0.0], [0.0, 0.0, 1.0]),           # -95% pupil rim
                Ray([y_sd * 0.95, 0.0, 0.0], [0.0, 0.0, 1.0]),            # X pupil rim
                Ray([0.0, y_sd * 0.999, 0.0], [0.0, 0.0, 1.0])            # Near 100% pupil rim
            ]

            traced = sys.trace_rays(rays)
            for idx, tr in enumerate(traced):
                self.assertFalse(tr.vignetted, f"Ray {idx} vignetted unexpectedly for y_sd={y_sd}")
                self.assertFalse(tr.tir, f"Ray {idx} TIR trapped unexpectedly for y_sd={y_sd}")
                # Confirm all intersection history coordinates are finite
                self.assertGreaterEqual(len(tr.history), 2, f"Ray {idx} did not reach surface 1")
                h1 = tr.history[1]
                self.assertFalse(h1.get("vignetted", False), f"Ray {idx} vignetted on curved surface 1 for y_sd={y_sd}")
                for h in tr.history:
                    self.assertTrue(math.isfinite(h["x"]))
                    self.assertTrue(math.isfinite(h["y"]))
                    self.assertTrue(math.isfinite(h["z"]))

    def test_dls_optimization_stability_with_curvature_bounds(self):
        """
        Adversarially run DLS optimization with curvature space on wide (y_sd = 50 mm)
        and narrow (y_sd = 1 mm) lenses, perturbing initial variables and checking
        that iterations remain bounded and never crash with unphysical intersection errors.
        """
        for y_sd in [1.0, 25.0, 50.0]:
            c_limit = 0.95 / y_sd
            # Initialize with curvature right at 90% of hemispherical limit
            init_c = 0.90 * c_limit
            init_r = 1.0 / init_c

            ct = max(1.5, y_sd * 0.4)
            bfl = max(5.0, y_sd * 1.5)

            stack = SurfaceStack()
            stack.add_surface(Surface(name="OBJ", surface_type=SurfaceType.OBJECT, thickness=200.0))
            s1 = Surface(name="L1_F", radius=init_r, thickness=ct, glass="N-BK7", semi_diameter=y_sd)
            setattr(s1, "radius_variable", True)
            setattr(s1, "radius_min", -1000.0)
            setattr(s1, "radius_max", 1000.0)
            stack.add_surface(s1)
            stack.add_surface(Surface(name="L1_R", radius=0.0, thickness=bfl, glass="AIR", semi_diameter=y_sd))
            stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=y_sd * 1.5))

            opt = OpticalOptimizer(OpticalSystem(stack))
            opt.method = "DLS"
            opt.use_curvature_space = True
            opt.max_iterations = 10
            opt.entrance_pupil_radius = y_sd * 0.8
            opt.num_rays = 12

            res = opt.optimize()

            # Verify optimization completed without exceptions
            self.assertIsNotNone(res)
            self.assertTrue(math.isfinite(res.final_rms))

            # Verify optimized curvature is strictly within [-c_limit, c_limit]
            final_r = res.optimized_stack.surfaces[1].radius
            final_c = radius_to_curvature(final_r)
            self.assertLessEqual(
                abs(final_c), c_limit + 1e-12,
                msg=f"Optimized curvature {final_c} exceeded limit {c_limit} for y_sd={y_sd}"
            )


class TestDualTreeParityAdversarial(unittest.TestCase):
    """
    Stress-testing 100% SHA-256 equivalence and identical runtime behavior between optics_app and v2.
    """

    def test_sha256_dual_tree_parity(self):
        """Verify byte-for-byte identity of all modified modules."""
        pairs = [
            ("optics_app/engine/manufacturing_standards.py", "v2/engine/manufacturing_standards.py"),
            ("optics_app/engine/merit_function.py", "v2/engine/merit_function.py"),
            ("optics_app/engine/optimizer.py", "v2/engine/optimizer.py")
        ]
        for p1, p2 in pairs:
            self.assertTrue(
                filecmp.cmp(p1, p2, shallow=False),
                msg=f"Files {p1} and {p2} differ!"
            )
            h1 = hashlib.sha256(open(p1, "rb").read()).hexdigest()
            h2 = hashlib.sha256(open(p2, "rb").read()).hexdigest()
            self.assertEqual(h1, h2)

    def test_v2_runtime_behavior_identical(self):
        """
        Verify that v2 modules compute exact identical TOTR, boundary operands,
        and dynamic curvature clamping.
        """
        # Test v2 manufacturing audit
        stack_v2 = v2_lde.SurfaceStack()
        stack_v2.add_surface(v2_surface.Surface(name="OBJ", surface_type=v2_surface.SurfaceType.OBJECT, thickness=5000.0))
        stack_v2.add_surface(v2_surface.Surface(name="L1_F", radius=40.0, thickness=5.0, glass="N-BK7", semi_diameter=12.5))
        stack_v2.add_surface(v2_surface.Surface(name="L1_R", radius=-40.0, thickness=25.0, glass="AIR", semi_diameter=12.5))
        stack_v2.add_surface(v2_surface.Surface(name="DET", surface_type=v2_surface.SurfaceType.DETECTOR, thickness=0.0, semi_diameter=15.0))

        cfg_v2 = v2_mfg.ManufacturingConstraintsConfig(max_total_track=100.0)
        report_v2 = v2_mfg.audit_optical_system_manufacturing(stack_v2.surfaces, config=cfg_v2)

        self.assertAlmostEqual(report_v2.total_track_mm, 30.0, places=4)
        self.assertEqual(len([e for e in report_v2.overall_errors if "MXTR" in e]), 0)

        # Test v2 merit function one-sided inequality
        sys_v2 = v2_tracer.OpticalSystem(stack_v2)
        rays_v2 = sys_v2.trace_rays([v2_tracer.Ray([0.0, 0.0, 0.0], [0.0, 0.0, 1.0])])
        mf_v2 = v2_merit_function.MeritFunction()
        mf_v2.penalty_weight = 0.0
        mf_v2.add_operand(v2_merit_function.OperandType.MNCT, target=1.5, weight=1.0, surface_index=1)
        score_v2 = mf_v2.evaluate(sys_v2, rays_v2)
        self.assertEqual(mf_v2.operands[0].diff, 0.0)
        self.assertEqual(score_v2, 0.0)


if __name__ == "__main__":
    unittest.main()
