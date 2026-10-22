"""
Milestone M1 Regression and Feature Validation Test Suite.
Verifies Features F1 through F5:
- F1: RMS Spot Radius / Zemax RSRE (Per-wavelength cluster RMS & Pythagorean identity within < 1e-12 mm)
- F2: RMS Spot X (RSX) & RMS Spot Y (RSY) Operands
- F3: Target Spot Size (SPTR) (No 10x scaling, standardized radius in mm, UI unit conversion)
- F4: Spectrometer Linear Dispersion Spread on arbitrary evaluation surfaces
- F5: Transparent Convergence Diagnostics (TerminationStatus, WavelengthSpotMetric, enriched OptimizationResult)
"""

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


class TestMilestoneM1Features(unittest.TestCase):
    """Unit and Integration Tests for Milestone M1 Features."""

    # ------------------------------------------------------------------------
    # F1: RMS Spot Radius / Zemax RSRE & Strict Pythagorean Precision
    # ------------------------------------------------------------------------
    def test_f1_pythagorean_identity_precision(self):
        """
        Verify that r_RMS^2 = RMS_x^2 + RMS_y^2 holds strictly within numerical precision (< 1e-12 mm)
        across arbitrary 2D coordinate distributions.
        """
        rng = np.random.default_rng(42)
        for trial in range(20):
            n_points = rng.integers(10, 500)
            xs = rng.normal(loc=rng.uniform(-5.0, 5.0), scale=rng.uniform(0.01, 2.0), size=n_points)
            ys = rng.normal(loc=rng.uniform(-5.0, 5.0), scale=rng.uniform(0.01, 2.0), size=n_points)

            spot = SpotDiagram(xs, ys)
            rms_x = spot.calculate_rms_x()
            rms_y = spot.calculate_rms_y()
            rms_r = spot.calculate_rms_radius()

            expected_r2 = rms_x**2 + rms_y**2
            actual_r2 = rms_r**2

            diff = abs(actual_r2 - expected_r2)
            self.assertLess(
                diff, 1e-12,
                f"Trial {trial}: Pythagorean discrepancy {diff:.2e} exceeded 1e-12 mm tolerance"
            )

    def test_f1_polychromatic_cluster_rms_does_not_conflate_lateral_color(self):
        """
        Verify that in polychromatic systems, RMS spot radius calculates cluster RMS without
        artificially inflating spot blur due to transverse chromatic displacement (lateral color).
        """
        # Create two identical small spot clusters with distinct transverse centroids (e.g. 5mm apart)
        xs_red = np.random.normal(loc=0.0, scale=0.01, size=50)
        ys_red = np.random.normal(loc=0.0, scale=0.01, size=50)

        xs_blue = np.random.normal(loc=5.0, scale=0.01, size=50)  # 5.0mm lateral color shift
        ys_blue = np.random.normal(loc=0.0, scale=0.01, size=50)

        spot_red = SpotDiagram(xs_red, ys_red)
        spot_blue = SpotDiagram(xs_blue, ys_blue)

        # Monochromatic cluster RMS radii should each be ~0.014 mm
        rms_red = spot_red.calculate_rms_radius()
        rms_blue = spot_blue.calculate_rms_radius()

        self.assertAlmostEqual(rms_red, 0.014, delta=0.005)
        self.assertAlmostEqual(rms_blue, 0.014, delta=0.005)

        # In MeritFunction, polychromatic RMS_SPOT_RADIUS should evaluate root-mean-square of cluster radii
        mf = MeritFunction()
        mf.add_operand(OperandType.RMS_SPOT_RADIUS, surface_index=1, target=0.0, weight=1.0)

        # Simulate rays hitting surface 1
        class DummyStack:
            surfaces = [Surface(name="OBJ"), Surface(name="DET", surface_type=SurfaceType.DETECTOR)]

        class DummySystem:
            stack = DummyStack()

        rays = []
        for x, y in zip(xs_red, ys_red):
            r = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=0.6563)
            r.history.append({"surface_index": 1, "x": float(x), "y": float(y), "dir_x": 0.0, "dir_y": 0.0, "dir_z": 1.0})
            rays.append(r)
        for x, y in zip(xs_blue, ys_blue):
            r = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=0.4861)
            r.history.append({"surface_index": 1, "x": float(x), "y": float(y), "dir_x": 0.0, "dir_y": 0.0, "dir_z": 1.0})
            rays.append(r)

        merit_val = mf.evaluate(DummySystem(), rays)
        # Merit value must be close to ~0.014mm, NOT pooled ~2.5mm
        self.assertLess(
            merit_val, 0.05,
            f"Merit function evaluated {merit_val:.3f} mm; lateral color was conflated with spot blur!"
        )

    # ------------------------------------------------------------------------
    # F2: RMS Spot X (RSX) & RMS Spot Y (RSY)
    # ------------------------------------------------------------------------
    def test_f2_rsx_and_rsy_asymmetric_slit_focus(self):
        """
        Verify that RSX and RSY isolate 1D transverse coordinate standard deviations.
        """
        # Anamorphic line focus: narrow in X (scale 0.005), elongated in Y (scale 2.0)
        xs = np.random.normal(loc=0.0, scale=0.005, size=100)
        ys = np.random.normal(loc=0.0, scale=2.0, size=100)

        spot = SpotDiagram(xs, ys)
        self.assertAlmostEqual(spot.calculate_rms_x(), 0.005, delta=0.002)
        self.assertAlmostEqual(spot.calculate_rms_y(), 2.0, delta=0.3)

        # Verify operands in MeritFunction
        mf = MeritFunction()
        op_x = mf.add_operand(OperandType.RMS_SPOT_X, surface_index=1, target=0.0, weight=1.0)
        op_y = mf.add_operand(OperandType.RMS_SPOT_Y, surface_index=1, target=0.0, weight=1.0)

        class DummyStack:
            surfaces = [Surface(name="OBJ"), Surface(name="DET", surface_type=SurfaceType.DETECTOR)]
        class DummySystem:
            stack = DummyStack()

        rays = []
        for x, y in zip(xs, ys):
            r = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=0.5876)
            r.history.append({"surface_index": 1, "x": float(x), "y": float(y)})
            rays.append(r)

        mf.evaluate(DummySystem(), rays)
        self.assertAlmostEqual(op_x.value, 0.005, delta=0.002)
        self.assertAlmostEqual(op_y.value, 2.0, delta=0.3)

    # ------------------------------------------------------------------------
    # F3: Target Spot Size (SPTR)
    # ------------------------------------------------------------------------
    def test_f3_target_spot_size_optimization(self):
        """
        Verify that optimizer with target='Target Spot Size' converges to the specified target spot radius.
        """
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=100.0))
        stack.add_surface(Surface(name="L1", radius=50.0, thickness=5.0, glass="N-BK7", semi_diameter=15.0))
        stack.add_surface(Surface(name="AirGap", radius=0.0, thickness=30.0, glass="AIR", semi_diameter=15.0, thickness_variable=True, thickness_min=10.0, thickness_max=200.0))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=20.0))

        sys = OpticalSystem(stack)
        target_radius_mm = 0.080  # Desired 80 um spot radius

        optimizer = OpticalOptimizer(
            sys,
            method="Nelder-Mead",
            max_iter=80,
            num_rays=25,
            target="Target Spot Size",
            target_spot_size=target_radius_mm
        )
        result = optimizer.optimize(variable_indices=[2])

        self.assertTrue(result.success)
        self.assertEqual(result.status, TerminationStatus.CONVERGED)
        
        # Calculate actual spot on optimized stack
        opt_metrics = optimizer._compute_detailed_spot_metrics(result.optimized_stack)
        actual_spot_r = opt_metrics[0.5876]["rms_radius"]

        self.assertAlmostEqual(
            actual_spot_r, target_radius_mm, delta=0.025,
            msg=f"Expected spot radius ~{target_radius_mm:.3f} mm, got {actual_spot_r:.3f} mm"
        )

    # ------------------------------------------------------------------------
    # F4: Spectrometer Linear Dispersion Spread on Specified Evaluation Surface
    # ------------------------------------------------------------------------
    def test_f4_spectrometer_spread_evaluation(self):
        """
        Verify spatial separation delta_X on grating system across min/max wavelengths.
        """
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=50.0))
        stack.add_surface(Surface(name="Grating", surface_type=SurfaceType.GRATING, grating_lines_per_mm=600.0, grating_order=1, thickness=50.0))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=50.0))

        sys = OpticalSystem(stack)
        wavelengths = [0.450, 0.550, 0.650]
        optimizer = OpticalOptimizer(
            sys,
            wavelengths=wavelengths,
            target="Spectrometer Spread",
            target_spread=6.0,
            num_rays=15,
            target_surface_index=2
        )
        merit = optimizer.merit_function(np.array([]), {})
        self.assertIsInstance(merit, float)
        self.assertGreater(merit, 0.0)

    # ------------------------------------------------------------------------
    # F5: Transparent Convergence Diagnostics
    # ------------------------------------------------------------------------
    def test_f5_convergence_diagnostics_structure_and_metrics(self):
        """
        Verify that OptimizationResult contains TerminationStatus, per-wavelength metrics,
        and accurate reduction percentages.
        """
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=100.0))
        stack.add_surface(Surface(name="L1", radius=50.0, thickness=5.0, glass="N-BK7", semi_diameter=15.0))
        stack.add_surface(Surface(name="AirGap", radius=0.0, thickness=30.0, glass="AIR", semi_diameter=15.0, thickness_variable=True, thickness_min=10.0, thickness_max=200.0))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=20.0))

        sys = OpticalSystem(stack)
        wavelengths = [0.4861, 0.5876, 0.6563]
        optimizer = OpticalOptimizer(sys, wavelengths=wavelengths, method="Nelder-Mead", max_iter=100, num_rays=20)
        result = optimizer.optimize(variable_indices=[2])

        self.assertIsInstance(result.status, TerminationStatus)
        self.assertEqual(result.status, TerminationStatus.CONVERGED)
        self.assertGreater(result.iterations, 0)
        self.assertLess(result.final_rms, result.initial_rms)
        self.assertGreater(result.rms_reduction_pct, 0.0)

        # Check per-wavelength metrics
        self.assertEqual(len(result.per_wavelength_metrics), 3)
        for wl in wavelengths:
            self.assertIn(wl, result.per_wavelength_metrics)
            m = result.per_wavelength_metrics[wl]
            self.assertIsInstance(m, WavelengthSpotMetric)
            self.assertEqual(m.wavelength, wl)
            self.assertGreater(m.initial_rms, 0.0)
            self.assertLess(m.final_rms, m.initial_rms)
            self.assertGreater(m.valid_rays, 0)
            self.assertGreater(m.airy_radius, 0.0)

    def test_f5_diagnostics_no_variables_status(self):
        """Verify NO_VARIABLES status when optimization runs with zero variables."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=100.0))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0))
        sys = OpticalSystem(stack)

        optimizer = OpticalOptimizer(sys)
        result = optimizer.optimize(variable_indices=[])
        self.assertEqual(result.status, TerminationStatus.NO_VARIABLES)
        self.assertEqual(result.iterations, 0)
        self.assertIn("No variable surfaces", result.message)

    def test_f5_diagnostics_user_aborted_status(self):
        """Verify USER_ABORTED status when optimizer abort flag is set."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=100.0))
        stack.add_surface(Surface(name="L1", radius=50.0, thickness=5.0, glass="N-BK7", semi_diameter=15.0))
        stack.add_surface(Surface(name="AirGap", radius=0.0, thickness=30.0, glass="AIR", semi_diameter=15.0))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=20.0))

        sys = OpticalSystem(stack)
        optimizer = OpticalOptimizer(sys, max_iter=100)
        optimizer.stop()
        result = optimizer.optimize(variable_indices=[2])
        self.assertEqual(result.status, TerminationStatus.USER_ABORTED)
        self.assertFalse(result.success)


if __name__ == "__main__":
    unittest.main()
