"""
Unit tests for optics_app/analysis/ package:
Grating Dispersion Analysis, Multi-Wavelength Spatial Separation,
Angular & Linear Dispersion, Spectral Resolution Metrics, and CCD Sensor Integration.
Covers Tiers 1-4 per TEST_INFRA.md (Feature 13).
"""

import unittest
import numpy as np

from optics_app.engine.grating import DiffractionGrating
from optics_app.engine.ccd import CCDSensor
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.analysis.grating_dispersion import (
    GratingDispersionAnalysis,
    WavelengthSpotResult,
    DispersionMetrics,
    GratingDispersionResult
)


class TestAnalysisTier1(unittest.TestCase):
    """Tier 1: Analytical Grating Dispersion & Metrics Calculations."""

    def test_t1_1_analytical_spatial_dispersion_monochromatic(self):
        """Single wavelength should yield a single valid diffraction angle and position."""
        grating = DiffractionGrating(lines_per_mm=600.0, order=1)
        spots = GratingDispersionAnalysis.calculate_spatial_dispersion(
            grating, wavelengths_um=[0.5876], focal_length_mm=100.0
        )
        self.assertEqual(len(spots), 1)
        wl, theta, x = spots[0]
        self.assertAlmostEqual(wl, 0.5876, places=4)
        self.assertIsNotNone(theta)
        self.assertIsNotNone(x)
        self.assertGreater(x, 0.0)

    def test_t1_2_analytical_spatial_dispersion_rgb_separation(self):
        """RGB wavelengths (0.650, 0.532, 0.450 um) should be spatially separated in order x_red > x_green > x_blue."""
        grating = DiffractionGrating(lines_per_mm=600.0, order=1)
        spots = GratingDispersionAnalysis.calculate_spatial_dispersion(
            grating, wavelengths_um=[0.650, 0.532, 0.450], focal_length_mm=100.0
        )
        self.assertEqual(len(spots), 3)
        x_red = spots[0][2]
        x_green = spots[1][2]
        x_blue = spots[2][2]
        self.assertGreater(x_red, x_green)
        self.assertGreater(x_green, x_blue)

    def test_t1_3_angular_dispersion_formula_verification(self):
        """Verify dTheta/dLambda = m / (d * cos(theta_m))."""
        grating = DiffractionGrating(lines_per_mm=600.0, order=1)
        theta_m = grating.calculate_diffraction_angle(0.0, 0.5876)
        metrics = GratingDispersionAnalysis.compute_dispersion_metrics(
            grating_lines_per_mm=600.0,
            order=1,
            diffraction_angle_rad=theta_m,
            focal_length_mm=100.0,
            wavelength_um=0.5876
        )
        # pitch d = 1000/600 = 1.6667 um
        expected_ang_disp = 1.0 / ((1000.0 / 600.0) * np.cos(theta_m))
        self.assertAlmostEqual(metrics.angular_dispersion_rad_per_um, expected_ang_disp, places=4)

    def test_t1_4_linear_dispersion_formula_verification(self):
        """Verify dx/dLambda = f * dTheta/dLambda for focal length f."""
        grating = DiffractionGrating(lines_per_mm=600.0, order=1)
        theta_m = grating.calculate_diffraction_angle(0.0, 0.5876)
        focal_length = 200.0
        metrics = GratingDispersionAnalysis.compute_dispersion_metrics(
            grating_lines_per_mm=600.0,
            order=1,
            diffraction_angle_rad=theta_m,
            focal_length_mm=focal_length,
            wavelength_um=0.5876
        )
        expected_lin_disp = focal_length * metrics.angular_dispersion_rad_per_um
        self.assertAlmostEqual(metrics.linear_dispersion_mm_per_um, expected_lin_disp, places=4)

    def test_t1_5_resolving_power_and_spectral_resolution(self):
        """Verify R = m * N_grooves and delta_lambda_diff = lambda / R."""
        metrics = GratingDispersionAnalysis.compute_dispersion_metrics(
            grating_lines_per_mm=600.0,
            order=1,
            diffraction_angle_rad=0.3,
            focal_length_mm=100.0,
            wavelength_um=0.5,
            beam_diameter_mm=20.0
        )
        expected_R = 1 * 600.0 * 20.0  # 12,000
        self.assertAlmostEqual(metrics.theoretical_resolving_power, expected_R, places=2)
        expected_res_nm = (0.5 * 1000.0) / expected_R  # 500 / 12000 = 0.04167 nm
        self.assertAlmostEqual(metrics.diffraction_resolution_nm, expected_res_nm, places=4)


class TestAnalysisTier2(unittest.TestCase):
    """Tier 2: Edge Cases, Boundary Values & Evanescent Order Safeguards."""

    def test_t2_1_evanescent_order_handling_in_dispersion(self):
        """High wavelength or high line density causing evanescent order returns None/is_evanescent=True without crashing."""
        grating = DiffractionGrating(lines_per_mm=3000.0, order=1)  # pitch = 0.333 um
        spots = GratingDispersionAnalysis.calculate_spatial_dispersion(
            grating, wavelengths_um=[0.800], focal_length_mm=100.0
        )
        self.assertEqual(len(spots), 1)
        self.assertIsNone(spots[0][1])
        self.assertIsNone(spots[0][2])

    def test_t2_2_zero_line_density_grating(self):
        """Zero line density produces zero angular dispersion."""
        metrics = GratingDispersionAnalysis.compute_dispersion_metrics(
            grating_lines_per_mm=0.0,
            order=1,
            diffraction_angle_rad=0.0,
            focal_length_mm=100.0,
            wavelength_um=0.5876
        )
        self.assertEqual(metrics.angular_dispersion_rad_per_um, 0.0)
        self.assertEqual(metrics.linear_dispersion_mm_per_um, 0.0)

    def test_t2_3_reciprocal_linear_dispersion(self):
        """Verify dlambda/dx = 1 / (dx/dlambda) in nm/mm."""
        metrics = GratingDispersionAnalysis.compute_dispersion_metrics(
            grating_lines_per_mm=600.0,
            order=1,
            diffraction_angle_rad=0.2,
            focal_length_mm=100.0,
            wavelength_um=0.5876
        )
        expected_recip = 1.0 / metrics.linear_dispersion_mm_per_nm
        self.assertAlmostEqual(metrics.reciprocal_linear_dispersion_nm_per_mm, expected_recip, places=4)

    def test_t2_4_pixel_limited_resolution_scaling(self):
        """Verify CCD pixel pitch impact on effective spectral resolution delta_lambda_pixel."""
        metrics = GratingDispersionAnalysis.compute_dispersion_metrics(
            grating_lines_per_mm=600.0,
            order=1,
            diffraction_angle_rad=0.2,
            focal_length_mm=100.0,
            wavelength_um=0.5876,
            pixel_pitch_mm=0.01  # 10 um pixel pitch
        )
        self.assertIsNotNone(metrics.pixel_limited_resolution_nm)
        expected_pix_res = 2.0 * 0.01 / metrics.linear_dispersion_mm_per_nm
        self.assertAlmostEqual(metrics.pixel_limited_resolution_nm, expected_pix_res, places=4)

    def test_t2_5_ccd_matrix_out_of_bounds_deposition(self):
        """Rays hitting outside CCD boundary increment out_of_bounds_rays correctly."""
        ccd = CCDSensor(width_mm=10.0, height_mm=10.0, pixel_cols=10, pixel_rows=10)
        ccd.deposit_ray(100.0, 100.0, intensity=1.0)
        self.assertEqual(ccd.out_of_bounds_rays, 1)
        self.assertEqual(ccd.total_rays_deposited, 0)


class TestAnalysisTier3(unittest.TestCase):
    """Tier 3: Multi-Wavelength Ray Tracing System Integration & CCD Heatmap."""

    def test_t3_1_full_system_dispersion_analysis(self):
        """Trace multi-wavelength rays through OpticalSystem with grating and measure spot centroids on detector."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="Obj", thickness=50.0))
        stack.add_surface(Surface(name="Grating", surface_type=SurfaceType.GRATING, grating_lines_per_mm=600.0, grating_order=1, thickness=50.0))
        stack.add_surface(Surface(name="Det", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=50.0))

        sys = OpticalSystem(stack)
        result = GratingDispersionAnalysis.analyze_system_dispersion(
            sys, wavelengths_um=[0.450, 0.532, 0.650], num_rays_per_wavelength=25, pupil_radius_mm=5.0
        )
        self.assertIsInstance(result, GratingDispersionResult)
        self.assertEqual(len(result.spots), 3)
        self.assertGreater(result.metrics.total_spectral_span_mm, 0.0)

    def test_t3_2_ccd_sensor_multi_wavelength_heatmap_deposit(self):
        """Deposit multi-wavelength ray trace results into CCDSensor and verify intensity accumulation."""
        ccd = CCDSensor(width_mm=50.0, height_mm=50.0, pixel_cols=50, pixel_rows=50)
        stack = SurfaceStack()
        stack.add_surface(Surface(name="Obj", thickness=50.0))
        stack.add_surface(Surface(name="Grating", surface_type=SurfaceType.GRATING, grating_lines_per_mm=600.0, grating_order=1, thickness=50.0))
        stack.add_surface(Surface(name="Det", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=50.0))

        sys = OpticalSystem(stack)
        result = GratingDispersionAnalysis.analyze_system_dispersion(
            sys, wavelengths_um=[0.450, 0.532, 0.650], num_rays_per_wavelength=25, pupil_radius_mm=5.0, ccd_sensor=ccd
        )
        self.assertGreater(ccd.get_total_intensity(), 0.0)

    def test_t3_3_aberration_limited_resolution_degradation(self):
        """Verify that large spot RMS radius degrades instrumental resolution delta_lambda_inst = 2*R_RMS / (dx/dlambda)."""
        metrics_small_spot = GratingDispersionAnalysis.compute_dispersion_metrics(
            grating_lines_per_mm=600.0, order=1, diffraction_angle_rad=0.2, focal_length_mm=100.0, wavelength_um=0.5876, rms_spot_radius_mm=0.001
        )
        metrics_large_spot = GratingDispersionAnalysis.compute_dispersion_metrics(
            grating_lines_per_mm=600.0, order=1, diffraction_angle_rad=0.2, focal_length_mm=100.0, wavelength_um=0.5876, rms_spot_radius_mm=0.1
        )
        self.assertGreater(metrics_large_spot.instrumental_resolution_nm, metrics_small_spot.instrumental_resolution_nm)


class TestAnalysisTier4(unittest.TestCase):
    """Tier 4: Spectrograph Workload Evaluation & High Groove Density Echelle Benchmark."""

    def test_t4_1_high_density_monochromator_dispersion_curve(self):
        """Evaluate 1200 lines/mm spectrograph dispersion curve across visible spectrum (0.400-0.700 um)."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="Obj", thickness=100.0))
        stack.add_surface(Surface(name="Grating", surface_type=SurfaceType.GRATING, grating_lines_per_mm=1200.0, grating_order=1, thickness=100.0))
        stack.add_surface(Surface(name="Det", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=100.0))

        sys = OpticalSystem(stack)
        wls = np.linspace(0.400, 0.700, 7).tolist()
        result = GratingDispersionAnalysis.analyze_system_dispersion(
            sys, wavelengths_um=wls, num_rays_per_wavelength=15
        )
        self.assertEqual(len(result.spots), 7)
        self.assertGreater(result.metrics.linear_dispersion_mm_per_um, 0.0)

    def test_t4_2_echelle_grating_resolving_power_benchmark(self):
        """Verify resolving power R > 50,000 for high-order echelle spectrograph setup."""
        metrics = GratingDispersionAnalysis.compute_dispersion_metrics(
            grating_lines_per_mm=300.0,
            order=10,
            diffraction_angle_rad=1.0,
            focal_length_mm=500.0,
            wavelength_um=0.5,
            beam_diameter_mm=50.0
        )
        # R = 10 * 300 * 50 = 150,000
        self.assertGreater(metrics.theoretical_resolving_power, 50000.0)


if __name__ == "__main__":
    unittest.main()
