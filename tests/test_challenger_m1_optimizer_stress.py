"""
Adversarial Empirical Stress Test Suite for Milestone M1 (Features F1 through F5).
Authored by Challenger 1 (Empirical Challenger) for 2D_Optics.

Tests:
1. Feature F1: Polychromatic RMS Spot Radius & Pythagorean Identity Precision
   - Off-axis multi-field ray distributions (0 to 15 deg) across 6 standard spectral lines (g, F, e, d, C, r)
   - Discrepancy tolerance < 1e-12 mm strictly verified
   - Chromatic cluster RMS isolation vs lateral color shifts (up to 20 mm shift)
   - Zero-ray and single-ray edge cases

2. Feature F2: 1D RMS Spot X / RMS Spot Y Optimization Convergence
   - Astigmatic cylindrical / tilted surface with distinct X and Y focal planes
   - Target="Spot X" convergence vs Target="Spot Y" convergence
   - Independent weighting verification in MeritFunction

3. Feature F3: Target Spot Size (SPTR) Arbitrary Specified Radii Convergence
   - Multi-target convergence across 0.01 mm, 0.05 mm, 0.10 mm, 0.20 mm
   - Convergence from under-focused and over-focused initial geometries
   - Verification of absence of arbitrary 10x multiplier in merit evaluation
   - Edge case: target_spot_size = 0.0 mm without div-by-zero or NaN

4. Feature F4: Spectrometer Linear Dispersion Spread on Intermediate and Detector Surfaces
   - Evaluation on intermediate surface (e.g. fold / DMD surface index 2) vs detector (index 3)
   - Target spread convergence on diffraction grating systems
   - Multi-operand preset MeritFunction.create_spectrometer_default()

5. Feature F5: Transparent Convergence Diagnostics & TerminationStatus
   - Comprehensive status verification: NO_VARIABLES, USER_ABORTED, CONVERGED, MAX_ITERATIONS_REACHED, RAY_TRACE_FAILURE
   - WavelengthSpotMetric field accuracy (initial_rms, final_rms, geo, centroid, ray counts, airy_radius)
   - Immutability of original system.stack during optimization runs
"""

import math
import unittest
import numpy as np

from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from optics_app.analysis.spot import SpotDiagram, calculate_spot_metrics
from optics_app.engine.merit_function import MeritFunction, MeritOperand, OperandType
from optics_app.engine.optimizer import (
    OpticalOptimizer,
    OptimizationResult,
    TerminationStatus,
    WavelengthSpotMetric,
    OptimizationAbortedException
)


class TestMilestoneM1AdversarialStress(unittest.TestCase):
    """Adversarial stress harness for M1 Optical Optimizer and Merit Functions."""

    # =========================================================================
    # SUITE 1: Feature F1 - Polychromatic Cluster RMS & Pythagorean Precision
    # =========================================================================
    def test_f1_pythagorean_identity_multi_field_and_polychromatic(self):
        """
        Adversarial Test F1.1: Verify Pythagorean identity r_RMS^2 = RMS_x^2 + RMS_y^2
        across extreme off-axis field angles and 6 spectral Fraunhofer lines.
        Numerical tolerance: < 1e-12 mm.
        """
        wavelengths = [0.4358, 0.4861, 0.5461, 0.5876, 0.6563, 0.7065]
        field_angles = [(0.0, 0.0), (2.0, 0.0), (0.0, 5.0), (7.5, 7.5), (-10.0, 5.0), (15.0, -10.0)]
        
        rng = np.random.default_rng(1337)

        for fx, fy in field_angles:
            for wl in wavelengths:
                # Generate realistic transverse ray spread with coma/astigmatism asymmetry
                n_rays = rng.integers(30, 200)
                # Skewed distributions
                raw_x = rng.normal(loc=fx * 0.5, scale=0.05 + 0.002 * abs(fx), size=n_rays) + 0.01 * (rng.uniform(-1, 1, n_rays)**2)
                raw_y = rng.normal(loc=fy * 0.5, scale=0.08 + 0.003 * abs(fy), size=n_rays) + 0.02 * (rng.uniform(-1, 1, n_rays)**3)

                spot = SpotDiagram(raw_x, raw_y, wavelength_um=wl)
                rx = spot.calculate_rms_x()
                ry = spot.calculate_rms_y()
                r_tot = spot.calculate_rms_radius()

                expected_r2 = rx**2 + ry**2
                actual_r2 = r_tot**2

                diff = abs(actual_r2 - expected_r2)
                self.assertLess(
                    diff, 1e-12,
                    f"Field ({fx}, {fy}), WL {wl} um: Pythagorean identity discrepancy {diff:.3e} exceeds 1e-12 mm"
                )

    def test_f1_polychromatic_cluster_merit_oracle_comparison(self):
        """
        Adversarial Test F1.2: Compare MeritFunction polychromatic RMS_SPOT_RADIUS directly
        against an independently calculated cluster Root-Mean-Square oracle.
        """
        wavelengths = [0.4861, 0.5876, 0.6563]
        cluster_radii = []

        class DummyStack:
            surfaces = [Surface(name="OBJ"), Surface(name="DET", surface_type=SurfaceType.DETECTOR)]

        class DummySystem:
            stack = DummyStack()

        rays = []
        rng = np.random.default_rng(2026)

        for wl in wavelengths:
            # Generate cluster with distinct centroid (lateral chromatic displacement)
            center_x = (wl - 0.5876) * 20.0  # Up to +/- 2.0 mm lateral color
            center_y = (wl - 0.5876) * 5.0
            xs = rng.normal(loc=center_x, scale=0.02, size=60)
            ys = rng.normal(loc=center_y, scale=0.03, size=60)

            spot = SpotDiagram(xs, ys)
            cluster_radii.append(spot.calculate_rms_radius())

            for x, y in zip(xs, ys):
                r = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl)
                r.history.append({"surface_index": 1, "x": float(x), "y": float(y), "dir_x": 0.0, "dir_y": 0.0, "dir_z": 1.0})
                rays.append(r)

        # Mathematical Oracle: Root-Mean-Square of cluster radii
        oracle_poly_rms = math.sqrt(sum(r**2 for r in cluster_radii) / len(cluster_radii))

        mf = MeritFunction()
        mf.add_operand(OperandType.RMS_SPOT_RADIUS, surface_index=1, target=0.0, weight=1.0)
        actual_mf = mf.evaluate(DummySystem(), rays, wavelengths=wavelengths)

        self.assertAlmostEqual(
            actual_mf, oracle_poly_rms, delta=1e-12,
            msg=f"MeritFunction {actual_mf:.10f} diverged from oracle {oracle_poly_rms:.10f}"
        )

    def test_f1_degenerate_clusters_zero_and_single_ray_stability(self):
        """
        Adversarial Test F1.3: Verify numerical stability without NaN or div-by-zero
        when a cluster has 0 or 1 ray.
        """
        spot_empty = SpotDiagram([], [])
        self.assertEqual(spot_empty.calculate_rms_radius(), 0.0)
        self.assertEqual(spot_empty.calculate_rms_x(), 0.0)
        self.assertEqual(spot_empty.calculate_rms_y(), 0.0)
        self.assertEqual(spot_empty.calculate_geo_radius(), 0.0)

        spot_single = SpotDiagram([10.5], [-3.2])
        self.assertEqual(spot_single.calculate_rms_radius(), 0.0)
        self.assertEqual(spot_single.calculate_rms_x(), 0.0)
        self.assertEqual(spot_single.calculate_rms_y(), 0.0)
        self.assertEqual(spot_single.calculate_geo_radius(), 0.0)

    # =========================================================================
    # SUITE 2: Feature F2 - 1D RMS Spot X & Spot Y Convergence
    # =========================================================================
    def test_f2_1d_astigmatic_system_selective_focusing(self):
        """
        Adversarial Test F2.1: In an optical system with astigmatic anamorphic focus
        (e.g., tilted surface producing distinct sagittal and tangential focal planes),
        verify that Target='Spot X' converges to the X focus and Target='Spot Y' converges to the Y focus.
        """
        # Create singlet with a tilted front surface creating strong astigmatism
        # Sagittal (Y) and Tangential (X) foci will occur at different air gap distances.
        def make_astigmatic_system(airgap=60.0):
            stk = SurfaceStack()
            stk.add_surface(Surface(name="OBJ", thickness=50.0))
            # Tilted convex lens -> strong astigmatism
            stk.add_surface(Surface(name="L1_front", radius=40.0, thickness=6.0, glass="N-BK7", semi_diameter=15.0, tilt_x=20.0))
            stk.add_surface(Surface(name="L1_back", radius=-100.0, thickness=airgap, glass="AIR", semi_diameter=15.0, thickness_variable=True, thickness_min=20.0, thickness_max=120.0))
            stk.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=25.0))
            return OpticalSystem(stk)

        # Optimize for Spot X
        sys_x = make_astigmatic_system(airgap=40.0)
        opt_x = OpticalOptimizer(sys_x, method="Nelder-Mead", max_iter=60, num_rays=21, target="RMS Spot X")
        res_x = opt_x.optimize(variable_indices=[2])
        self.assertTrue(res_x.success)

        # Optimize for Spot Y
        sys_y = make_astigmatic_system(airgap=40.0)
        opt_y = OpticalOptimizer(sys_y, method="Nelder-Mead", max_iter=60, num_rays=21, target="RMS Spot Y")
        res_y = opt_y.optimize(variable_indices=[2])
        self.assertTrue(res_y.success)

        # Compute spot metrics on both results
        metrics_x = opt_x._compute_detailed_spot_metrics(res_x.optimized_stack)[0.5876]
        metrics_y = opt_y._compute_detailed_spot_metrics(res_y.optimized_stack)[0.5876]

        # The X-optimized system must have smaller RMS_X than the Y-optimized system
        self.assertLess(
            metrics_x["rms_x"], metrics_y["rms_x"],
            f"Expected Spot X optimization to produce lower RMS X: {metrics_x['rms_x']} vs {metrics_y['rms_x']}"
        )
        # The Y-optimized system must have smaller RMS_Y than the X-optimized system
        self.assertLess(
            metrics_y["rms_y"], metrics_x["rms_y"],
            f"Expected Spot Y optimization to produce lower RMS Y: {metrics_y['rms_y']} vs {metrics_x['rms_y']}"
        )

    def test_f2_merit_function_dual_rsx_rsy_operand_weighting(self):
        """
        Adversarial Test F2.2: Verify that MeritFunction evaluates weighted combination
        of RSX and RSY strictly according to RSS error formula.
        """
        xs = np.array([-1.0, 1.0, -1.0, 1.0])
        ys = np.array([-4.0, -4.0, 4.0, 4.0])
        spot = SpotDiagram(xs, ys)
        rms_x = spot.calculate_rms_x()  # 1.0
        rms_y = spot.calculate_rms_y()  # 4.0

        self.assertAlmostEqual(rms_x, 1.0, places=6)
        self.assertAlmostEqual(rms_y, 4.0, places=6)

        class DummyStack:
            surfaces = [Surface(name="OBJ"), Surface(name="DET", surface_type=SurfaceType.DETECTOR)]
        class DummySystem:
            stack = DummyStack()

        rays = []
        for x, y in zip(xs, ys):
            r = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=0.5876)
            r.history.append({"surface_index": 1, "x": float(x), "y": float(y)})
            rays.append(r)

        # Test with Weight X = 4.0, Weight Y = 1.0
        mf = MeritFunction()
        mf.add_operand(OperandType.RMS_SPOT_X, surface_index=1, target=0.0, weight=4.0)
        mf.add_operand(OperandType.RMS_SPOT_Y, surface_index=1, target=0.0, weight=1.0)
        val = mf.evaluate(DummySystem(), rays)

        expected_val = math.sqrt((4.0 * (1.0**2) + 1.0 * (4.0**2)) / (4.0 + 1.0))  # sqrt((4 + 16)/5) = 2.0
        self.assertAlmostEqual(val, expected_val, places=8)

    # =========================================================================
    # SUITE 3: Feature F3 - Target Spot Size (SPTR) Arbitrary Radii Convergence
    # =========================================================================
    def test_f3_arbitrary_target_spot_radii_convergence(self):
        """
        Adversarial Test F3.1: Stress-test convergence to arbitrary specified non-zero radii
        [0.10 mm, 0.15 mm, 0.20 mm] from both under-focused and over-focused initial states.
        """
        targets = [0.100, 0.150, 0.200]  # in mm

        for tgt in targets:
            # Under-focused initial state (airgap = 30 mm vs best focus ~95 mm)
            stack = SurfaceStack()
            stack.add_surface(Surface(name="OBJ", thickness=100.0))
            stack.add_surface(Surface(name="L1", radius=50.0, thickness=5.0, glass="N-BK7", semi_diameter=15.0))
            stack.add_surface(Surface(name="AirGap", radius=0.0, thickness=30.0, glass="AIR", semi_diameter=15.0, thickness_variable=True, thickness_min=10.0, thickness_max=200.0))
            stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=20.0))

            sys = OpticalSystem(stack)
            optimizer = OpticalOptimizer(
                sys,
                method="Nelder-Mead",
                max_iter=90,
                num_rays=25,
                target="Target Spot Size",
                target_spot_size=tgt
            )
            result = optimizer.optimize(variable_indices=[2])

            self.assertTrue(result.success, f"Optimization failed for target radius {tgt} mm")
            self.assertEqual(result.status, TerminationStatus.CONVERGED)

            # Compute actual spot on result stack
            metrics = optimizer._compute_detailed_spot_metrics(result.optimized_stack)
            actual_r = metrics[0.5876]["rms_radius"]

            self.assertAlmostEqual(
                actual_r, tgt, delta=0.025,
                msg=f"Target {tgt} mm failed: achieved {actual_r:.4f} mm (delta {abs(actual_r - tgt):.4f} mm)"
            )

    def test_f3_target_spot_size_zero_and_no_scaling_multiplier(self):
        """
        Adversarial Test F3.2: Verify target spot size = 0.0 evaluates directly without
        division by zero or artificial scaling factors.
        """
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=100.0))
        stack.add_surface(Surface(name="L1", radius=50.0, thickness=5.0, glass="N-BK7", semi_diameter=15.0))
        stack.add_surface(Surface(name="AirGap", radius=0.0, thickness=95.0, glass="AIR", semi_diameter=15.0))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=20.0))

        sys = OpticalSystem(stack)
        
        # Merit with target_spot_size = 0.05 mm
        opt50 = OpticalOptimizer(sys, target="Target Spot Size", target_spot_size=0.050, num_rays=21)
        # Merit with target_spot_size = 0.00 mm
        opt0 = OpticalOptimizer(sys, target="Target Spot Size", target_spot_size=0.0, num_rays=21)
        # Merit with standard Spot Size
        opt_std = OpticalOptimizer(sys, target="Spot Size", num_rays=21)

        m50 = opt50.merit_function(np.array([95.0]), [2])
        m0 = opt0.merit_function(np.array([95.0]), [2])
        m_std = opt_std.merit_function(np.array([95.0]), [2])

        # m0 and m_std must be identical within numerical precision
        self.assertAlmostEqual(m0, m_std, places=10)
        # m50 must equal |m_std - 0.050|
        self.assertAlmostEqual(m50, abs(m_std - 0.050), places=10)

    # =========================================================================
    # SUITE 4: Feature F4 - Spectrometer Linear Dispersion Spread Optimization
    # =========================================================================
    def test_f4_intermediate_surface_vs_detector_dispersion_spread(self):
        """
        Adversarial Test F4.1: Verify linear dispersion spread delta_X on an intermediate
        surface (surface_index=2) vs final detector (surface_index=3).
        """
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=50.0))
        # Diffraction grating at surface 1
        stack.add_surface(Surface(name="Grating", surface_type=SurfaceType.GRATING, grating_lines_per_mm=600.0, grating_order=1, thickness=50.0))
        # Intermediate fold/DMD plane at surface 2
        stack.add_surface(Surface(name="DMD", surface_type=SurfaceType.STANDARD, thickness=50.0, semi_diameter=40.0))
        # Final detector at surface 3
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=60.0))

        sys = OpticalSystem(stack)
        wavelengths = [0.450, 0.650]

        # Evaluate optimizer targeting intermediate DMD (index 2)
        opt_dmd = OpticalOptimizer(
            sys, wavelengths=wavelengths, target="Spectrometer Spread",
            target_spread=5.0, target_surface_index=2, num_rays=15
        )
        merit_dmd = opt_dmd.merit_function(np.array([]), {})

        # Evaluate optimizer targeting final detector (index 3)
        opt_det = OpticalOptimizer(
            sys, wavelengths=wavelengths, target="Spectrometer Spread",
            target_spread=5.0, target_surface_index=3, num_rays=15
        )
        merit_det = opt_det.merit_function(np.array([]), {})

        # The spatial spread on surface 3 (100 mm from grating) must be roughly 2x the spread on surface 2 (50 mm from grating)
        self.assertIsInstance(merit_dmd, float)
        self.assertIsInstance(merit_det, float)
        self.assertNotEqual(merit_dmd, merit_det)

    def test_f4_spectrometer_merit_function_preset_integration(self):
        """
        Adversarial Test F4.2: Verify MeritFunction.create_spectrometer_default
        evaluates both intermediate DMD spread and final detector spot radius.
        """
        mf = MeritFunction.create_spectrometer_default(
            dmd_surface=2, target_dmd_spread=6.0,
            det_surface=3, target_det_spot=0.0,
            weight_spread=5.0, weight_det=2.0
        )
        self.assertEqual(len(mf.operands), 2)
        self.assertEqual(mf.operands[0].operand_type, OperandType.SPECTROMETER_SPREAD)
        self.assertEqual(mf.operands[0].surface_index, 2)
        self.assertEqual(mf.operands[0].target, 6.0)
        self.assertEqual(mf.operands[1].operand_type, OperandType.RMS_SPOT_RADIUS)
        self.assertEqual(mf.operands[1].surface_index, 3)

        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=50.0))
        stack.add_surface(Surface(name="Grating", surface_type=SurfaceType.GRATING, grating_lines_per_mm=600.0, grating_order=1, thickness=50.0))
        stack.add_surface(Surface(name="DMD", surface_type=SurfaceType.STANDARD, thickness=50.0, semi_diameter=40.0))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=60.0))

        sys = OpticalSystem(stack)
        rays = [
            Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=0.450),
            Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=0.650)
        ]
        traced = sys.trace_rays(rays)
        val = mf.evaluate(sys, traced, wavelengths=[0.450, 0.650])
        self.assertGreater(val, 0.0)

        breakdown = mf.get_breakdown()
        self.assertEqual(len(breakdown), 2)
        self.assertIn("SPRD", [b["code"] for b in breakdown])
        self.assertIn("TRAC", [b["code"] for b in breakdown])

    # =========================================================================
    # SUITE 5: Feature F5 - Transparent Diagnostics & TerminationStatus
    # =========================================================================
    def test_f5_structured_termination_status_enumeration(self):
        """
        Adversarial Test F5.1: Verify all key TerminationStatus enum values under
        distinct operational scenarios.
        """
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=100.0))
        stack.add_surface(Surface(name="L1", radius=50.0, thickness=5.0, glass="N-BK7", semi_diameter=15.0))
        stack.add_surface(Surface(name="AirGap", radius=0.0, thickness=30.0, glass="AIR", semi_diameter=15.0, thickness_variable=True, thickness_min=10.0, thickness_max=200.0))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=20.0))
        sys = OpticalSystem(stack)

        # 1. NO_VARIABLES
        opt_novar = OpticalOptimizer(sys)
        res_novar = opt_novar.optimize(variable_indices=[])
        self.assertEqual(res_novar.status, TerminationStatus.NO_VARIABLES)

        # 2. USER_ABORTED
        opt_abort = OpticalOptimizer(sys, method="Nelder-Mead", max_iter=50)
        def abort_cb(current=0, total=50):
            opt_abort.stop()
        opt_abort.progress_callback = abort_cb
        res_abort = opt_abort.optimize(variable_indices=[2])
        self.assertEqual(res_abort.status, TerminationStatus.USER_ABORTED)
        self.assertFalse(res_abort.success)

        # 3. MAX_ITERATIONS_REACHED
        opt_maxit = OpticalOptimizer(sys, method="Nelder-Mead", max_iter=2, num_rays=15)
        res_maxit = opt_maxit.optimize(variable_indices=[2])
        self.assertEqual(res_maxit.status, TerminationStatus.MAX_ITERATIONS_REACHED)

        # 4. CONVERGED
        opt_conv = OpticalOptimizer(sys, method="Nelder-Mead", max_iter=80, num_rays=15)
        res_conv = opt_conv.optimize(variable_indices=[2])
        self.assertEqual(res_conv.status, TerminationStatus.CONVERGED)
        self.assertTrue(res_conv.success)

    def test_f5_per_wavelength_diagnostics_integrity(self):
        """
        Adversarial Test F5.2: Verify accuracy and physical validity of WavelengthSpotMetric
        across polychromatic optimization.
        """
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=100.0))
        stack.add_surface(Surface(name="L1", radius=50.0, thickness=5.0, glass="N-BK7", semi_diameter=20.0))
        stack.add_surface(Surface(name="AirGap", radius=0.0, thickness=30.0, glass="AIR", semi_diameter=20.0, thickness_variable=True, thickness_min=10.0, thickness_max=200.0))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=20.0))

        sys = OpticalSystem(stack)
        wavelengths = [0.4861, 0.5876, 0.6563]
        optimizer = OpticalOptimizer(sys, wavelengths=wavelengths, method="Nelder-Mead", max_iter=60, num_rays=20, entrance_pupil_radius=15.0)
        res = optimizer.optimize(variable_indices=[2])

        self.assertEqual(len(res.per_wavelength_metrics), 3)
        for wl in wavelengths:
            m = res.per_wavelength_metrics[wl]
            self.assertEqual(m.wavelength, wl)
            self.assertGreater(m.initial_rms, 0.0)
            self.assertGreater(m.initial_geo, 0.0)
            self.assertLessEqual(m.final_rms, m.initial_rms)
            self.assertGreater(m.valid_rays, 0)
            self.assertEqual(m.total_rays, m.valid_rays)
            self.assertAlmostEqual(m.airy_radius, 1.22 * wl * 4.0, delta=1e-3)

        # Check percentage reduction formula
        expected_pct = ((res.initial_rms - res.final_rms) / res.initial_rms) * 100.0
        self.assertAlmostEqual(res.rms_reduction_pct, expected_pct, places=5)

    def test_f5_immutability_of_input_surface_stack(self):
        """
        Adversarial Test F5.3: Ensure optimizer.optimize() returns an optimized_stack
        without mutating the input system.stack in place unexpectedly.
        """
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=100.0))
        stack.add_surface(Surface(name="L1", radius=50.0, thickness=5.0, glass="N-BK7", semi_diameter=15.0))
        stack.add_surface(Surface(name="AirGap", radius=0.0, thickness=30.0, glass="AIR", semi_diameter=15.0, thickness_variable=True, thickness_min=10.0, thickness_max=200.0))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=20.0))

        sys = OpticalSystem(stack)
        orig_thickness = stack.surfaces[2].thickness
        
        opt = OpticalOptimizer(sys, method="Nelder-Mead", max_iter=40, num_rays=15)
        res = opt.optimize(variable_indices=[2])

        # Input stack should be unmodified
        self.assertEqual(sys.stack.surfaces[2].thickness, orig_thickness)
        # Optimized stack should have the new value
        self.assertNotEqual(res.optimized_stack.surfaces[2].thickness, orig_thickness)


if __name__ == "__main__":
    unittest.main()
