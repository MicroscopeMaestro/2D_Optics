"""
Empirical Stress & Challenge Test Suite for Milestone 3 (Detector & Zemax-Style Analysis Engine).
Created by Challenger 2.
"""
import pytest
import numpy as np
from optics_app.analysis.spot import SpotDiagram, calculate_spot_metrics
from optics_app.analysis.rayfan import RayFanAnalysis, RayFanData, calculate_ray_fan
from optics_app.analysis.aberration import (
    SeidelAberrations,
    trace_paraxial_marginal_ray,
    trace_paraxial_chief_ray,
    calculate_seidel_coefficients,
    _extract_surfaces
)
from optics_app.analysis.grating_dispersion import (
    GratingDispersionAnalysis,
    WavelengthSpotResult,
    DispersionMetrics,
    GratingDispersionResult
)
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.grating import DiffractionGrating
from optics_app.engine.ccd import CCDSensor


class TestSpotDiagramEmpirical:
    """Empirical challenges for SpotDiagram mathematical formulas and edge cases."""

    def test_centroid_empty_and_single(self):
        """Test centroid behavior on empty and single point inputs."""
        spot_empty = SpotDiagram([], [])
        assert spot_empty.calculate_centroid() == (0.0, 0.0)

        spot_single = SpotDiagram([3.5], [-2.1])
        cx, cy = spot_single.calculate_centroid()
        assert np.isclose(cx, 3.5)
        assert np.isclose(cy, -2.1)

    def test_centroid_and_rms_circle_and_disk(self):
        """Empirically test RMS radius against exact analytical circular geometry."""
        # 1. Circle of radius R
        R = 5.0
        n_pts = 1000
        angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
        x_ring = R * np.cos(angles)
        y_ring = R * np.sin(angles)

        spot_ring = SpotDiagram(x_ring, y_ring)
        cx, cy = spot_ring.calculate_centroid()
        assert np.isclose(cx, 0.0, atol=1e-12)
        assert np.isclose(cy, 0.0, atol=1e-12)
        assert np.isclose(spot_ring.calculate_rms_radius(), R, rtol=1e-5)
        assert np.isclose(spot_ring.calculate_geo_radius(), R, rtol=1e-5)

        # 2. Disk of radius R: theoretical RMS radius is R / sqrt(2) ~ 0.70710678 * R
        r_coords = np.sqrt(np.random.uniform(0, R**2, 50000))
        theta_coords = np.random.uniform(0, 2 * np.pi, 50000)
        x_disk = r_coords * np.cos(theta_coords)
        y_disk = r_coords * np.sin(theta_coords)

        spot_disk = SpotDiagram(x_disk, y_disk)
        expected_rms = R / np.sqrt(2.0)
        assert np.isclose(spot_disk.calculate_rms_radius(), expected_rms, rtol=0.02)

    def test_airy_disk_formula_and_validation(self):
        """Empirically verify Airy disk formula r_airy = 1.22 * lambda * F/#."""
        # Standard d-line (0.5876 um) at F/4
        spot = SpotDiagram([0.0], [0.0], wavelength_um=0.5876, f_number=4.0)
        expected_airy = 1.22 * 0.5876 * 4.0  # 2.867488 um
        assert np.isclose(spot.calculate_airy_disk_radius(), expected_airy, rtol=1e-6)

        # Negative wavelength or F/# should raise ValueError
        with pytest.raises(ValueError):
            SpotDiagram([0.0], [0.0], wavelength_um=-0.5, f_number=4.0).calculate_airy_disk_radius()

        with pytest.raises(ValueError):
            SpotDiagram([0.0], [0.0], wavelength_um=0.5, f_number=0.0).calculate_airy_disk_radius()

    def test_diffraction_limited_flag(self):
        """Verify diffraction-limited boundary condition logic."""
        # Spot RMS radius 1.0 um < Airy radius 2.867 um -> True
        spot_small = SpotDiagram([-1.0, 1.0], [0.0, 0.0], wavelength_um=0.5876, f_number=4.0)
        assert spot_small.calculate_rms_radius() == 1.0
        assert spot_small.is_diffraction_limited() is True

        # Spot RMS radius 5.0 um > Airy radius 2.867 um -> False
        spot_large = SpotDiagram([-5.0, 5.0], [0.0, 0.0], wavelength_um=0.5876, f_number=4.0)
        assert spot_large.calculate_rms_radius() == 5.0
        assert spot_large.is_diffraction_limited() is False


class TestSeidelAberrationsEmpirical:
    """Empirical challenges for 3rd-order Seidel aberration equations and Lagrange invariant."""

    def test_lagrange_invariant_calculation(self):
        """Verify Lagrange Invariant H = n(y u_bar - y_bar u) evaluation in calculate_seidel_coefficients."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", surface_type=SurfaceType.OBJECT, thickness=0.0))
        stack.add_surface(Surface(name="Surf1", radius=50.0, thickness=5.0, glass="N-BK7", semi_diameter=12.5))
        stack.add_surface(Surface(name="Stop", surface_type=SurfaceType.STOP, radius=-50.0, thickness=40.0, glass="AIR", semi_diameter=10.0))
        stack.add_surface(Surface(name="CCD", surface_type=SurfaceType.DETECTOR))

        system = OpticalSystem(stack)
        seidel = calculate_seidel_coefficients(system, wavelength_um=0.5876, field_angle_deg=2.0)

        marginal = trace_paraxial_marginal_ray(system)
        chief = trace_paraxial_chief_ray(system, field_angle_deg=2.0)

        y0, u0, _, _ = marginal[0]
        yb0, ub0, _, _ = chief[0]
        expected_H = float(1.0 * (y0 * ub0 - yb0 * u0))

        assert np.isclose(seidel.lagrange_invariant_H, expected_H, rtol=1e-5)

    def test_mirror_glass_resolution_bug(self):
        """Verify that glass='MIRROR' and 'REFLECTOR' are resolved without throwing ValueError in calculate_seidel_coefficients."""
        for glass_type in ["MIRROR", "REFLECTOR"]:
            stack = SurfaceStack()
            stack.add_surface(Surface(name="OBJ", surface_type=SurfaceType.OBJECT, thickness=0.0))
            stack.add_surface(Surface(name="Mirror", radius=-200.0, thickness=-100.0, glass=glass_type, conic=-1.0, semi_diameter=25.0))
            stack.add_surface(Surface(name="CCD", surface_type=SurfaceType.DETECTOR))

            system = OpticalSystem(stack)
            seidel = calculate_seidel_coefficients(system, wavelength_um=0.5876)
            assert isinstance(seidel, SeidelAberrations)
            assert np.isclose(seidel.surface_details[1].n, -1.0, atol=1e-3)


class TestGratingDispersionEmpirical:
    """Empirical challenges for spectral resolution limits and dispersion metrics."""

    def test_spectral_resolution_limit_formulas(self):
        """Verify diffraction, instrumental, pixel, and effective resolution limit calculations."""
        lines_per_mm = 600.0
        order = 1
        focal_length = 100.0
        wl_um = 0.500  # 500 nm
        beam_dia = 20.0  # 20 mm pupil
        rms_spot = 0.005  # 5 um RMS radius
        pixel_pitch = 0.010  # 10 um pixel

        metrics = GratingDispersionAnalysis.compute_dispersion_metrics(
            grating_lines_per_mm=lines_per_mm,
            order=order,
            diffraction_angle_rad=0.0,
            focal_length_mm=focal_length,
            wavelength_um=wl_um,
            beam_diameter_mm=beam_dia,
            rms_spot_radius_mm=rms_spot,
            pixel_pitch_mm=pixel_pitch
        )

        # 1. Theoretical resolving power R_theo = order * G * W = 1 * 600 * 20 = 12,000
        assert np.isclose(metrics.theoretical_resolving_power, 12000.0)

        # 2. Diffraction resolution limit = lambda (nm) / R = 500 / 12000 = 0.041667 nm
        assert np.isclose(metrics.diffraction_resolution_nm, 500.0 / 12000.0, rtol=1e-5)

        # 3. Angular dispersion d_theta / d_lambda = order / pitch = 1 / 1.66667 um = 0.6 rad/um = 0.0006 rad/nm
        expected_d_theta_d_lambda_nm = 1.0 / (1000.0 / lines_per_mm) / 1000.0
        assert np.isclose(metrics.angular_dispersion_rad_per_nm, expected_d_theta_d_lambda_nm, rtol=1e-5)

        # 4. Linear dispersion dx / d_lambda = f * d_theta / d_lambda = 100 * 0.0006 = 0.06 mm/nm
        expected_dx_d_lambda_nm = focal_length * expected_d_theta_d_lambda_nm
        assert np.isclose(metrics.linear_dispersion_mm_per_nm, expected_dx_d_lambda_nm, rtol=1e-5)

        # 5. Instrumental limit = 2 * RMS / (dx / d_lambda) = 2 * 0.005 mm / 0.06 mm/nm = 0.16667 nm
        expected_inst_res = (2.0 * rms_spot) / expected_dx_d_lambda_nm
        assert np.isclose(metrics.instrumental_resolution_nm, expected_inst_res, rtol=1e-5)

        # 6. Pixel limit = 2 * pixel_pitch / (dx / d_lambda) = 2 * 0.010 mm / 0.06 mm/nm = 0.33333 nm
        expected_pix_res = (2.0 * pixel_pitch) / expected_dx_d_lambda_nm
        assert np.isclose(metrics.pixel_limited_resolution_nm, expected_pix_res, rtol=1e-5)

        # 7. Effective resolution limit = max(diffraction, instrumental, pixel)
        expected_eff_res = max(metrics.diffraction_resolution_nm, expected_inst_res, expected_pix_res)
        assert np.isclose(metrics.effective_resolution_nm, expected_eff_res, rtol=1e-5)
        assert metrics.effective_resolution_nm == expected_pix_res

        # 8. Effective resolving power R_eff = lambda / delta_lambda_eff = 500 / 0.33333 = 1500
        assert np.isclose(metrics.effective_resolving_power, 500.0 / expected_eff_res, rtol=1e-5)


class TestCCDSensorEmpirical:
    """Empirical challenges for CCD sensor accumulation, boundary checks, and intensity matrix."""

    def test_ccd_accumulation_and_out_of_bounds(self):
        """Verify CCD deposit, bounds checking, and matrix total sum."""
        ccd = CCDSensor(width_mm=10.0, height_mm=10.0, pixel_cols=100, pixel_rows=100)

        # Pitch is 0.1 mm x 0.1 mm
        assert np.isclose(ccd.pixel_pitch_x, 0.1)
        assert np.isclose(ccd.pixel_pitch_y, 0.1)

        # Deposit ray at origin (0, 0) -> maps to row 50, col 50
        ccd.deposit_ray(0.0, 0.0, intensity=5.0)

        # Deposit ray out of bounds at (10.0, 0.0) -> col 150 (out of bounds)
        ccd.deposit_ray(10.0, 0.0, intensity=1.0)

        assert ccd.total_rays_deposited == 1
        assert ccd.out_of_bounds_rays == 1
        assert ccd.get_total_intensity() == 5.0

        # Matrix element at (50, 50)
        assert ccd.matrix[50, 50] == 5.0
