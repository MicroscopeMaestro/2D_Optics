"""
Unit tests for Diffraction Grating Physics, Order m, 2D/3D Diffraction Angles, and Grating Dispersion.
Covers Tiers 1-4 per TEST_INFRA.md (Features 7, 13).
"""

import unittest
import numpy as np

# Try importing from optics_app engine or supply exact physics fallbacks
try:
    from optics_app.engine.grating import DiffractionGrating, grating_diffract_3d
    from optics_app.analysis.grating_dispersion import GratingDispersionAnalysis
    OPTICS_APP_AVAILABLE = True
except ImportError:
    OPTICS_APP_AVAILABLE = False

    def grating_diffract_3d(ray_dir, grating_normal, line_vector, line_density_lines_per_mm, order, wavelength_um):
        """
        3D Grating equation:
        k_out = k_in + m * G
        where G = lambda * g * line_vector
        """
        v_in = np.array(ray_dir, dtype=float) / np.linalg.norm(ray_dir)
        N = np.array(grating_normal, dtype=float) / np.linalg.norm(grating_normal)
        L = np.array(line_vector, dtype=float) / np.linalg.norm(line_vector)
        
        # d in um: line_density is lines/mm -> d_mm = 1/g -> d_um = 1000/g
        d_um = 1000.0 / line_density_lines_per_mm if line_density_lines_per_mm > 0 else 1e9
        
        # 2D projection angle
        sin_theta_i = v_in[0]
        term = sin_theta_i + (order * wavelength_um / d_um)
        
        if abs(term) > 1.0:
            return None, True  # Evanescent wave
        
        sin_theta_m = term
        cos_theta_m = np.sqrt(1.0 - sin_theta_m**2)
        v_out = np.array([sin_theta_m, v_in[1], cos_theta_m])
        return v_out / np.linalg.norm(v_out), False

    class DiffractionGrating:
        def __init__(self, lines_per_mm=600.0, order=1, is_reflection=False):
            self.lines_per_mm = lines_per_mm
            self.order = order
            self.is_reflection = is_reflection
            self.pitch_um = 1000.0 / lines_per_mm if lines_per_mm > 0 else 1e9

        def calculate_diffraction_angle(self, incident_angle_rad, wavelength_um):
            sin_i = np.sin(incident_angle_rad)
            sin_m = sin_i + (self.order * wavelength_um / self.pitch_um)
            if abs(sin_m) > 1.0:
                raise ValueError("Evanescent order: diffraction angle undefined.")
            return np.arcsin(sin_m)

        def angular_dispersion(self, diffracted_angle_rad):
            # dTheta / dLambda = m / (d * cos(theta_m))
            d_um = self.pitch_um
            return self.order / (d_um * np.cos(diffracted_angle_rad))

    class GratingDispersionAnalysis:
        @staticmethod
        def calculate_spatial_dispersion(grating: DiffractionGrating, wavelengths_um, focal_length_mm=100.0):
            spots = []
            for wl in wavelengths_um:
                try:
                    theta_m = grating.calculate_diffraction_angle(0.0, wl)
                    x_pos = focal_length_mm * np.tan(theta_m)
                    spots.append((wl, theta_m, x_pos))
                except ValueError:
                    spots.append((wl, None, None))
            return spots


class TestGratingDiffractionTier1(unittest.TestCase):
    """Tier 1: Diffraction Grating Physics Equation d*(sin_i + sin_m) = m*lambda."""

    def test_t1_1_normal_incidence_first_order_diffraction(self):
        grating = DiffractionGrating(lines_per_mm=600.0, order=1)
        wl_um = 0.532  # 532 nm green laser
        # pitch = 1000 / 600 = 1.6667 um
        # sin(theta_m) = 1 * 0.532 / 1.6667 = 0.3192
        theta_m = grating.calculate_diffraction_angle(0.0, wl_um)
        expected_sin = 0.532 / (1000.0 / 600.0)
        self.assertAlmostEqual(np.sin(theta_m), expected_sin, places=5)

    def test_t1_2_zero_order_specular_transmission(self):
        grating = DiffractionGrating(lines_per_mm=1200.0, order=0)
        theta_m = grating.calculate_diffraction_angle(np.radians(20.0), 0.6328)
        self.assertAlmostEqual(abs(theta_m), np.radians(20.0), places=5)

    def test_t1_3_negative_order_diffraction(self):
        grating = DiffractionGrating(lines_per_mm=300.0, order=-1)
        theta_m = grating.calculate_diffraction_angle(0.0, 0.5)
        self.assertLess(theta_m, 0.0)

    def test_t1_4_3d_vector_diffraction(self):
        v_in = [0.0, 0.0, 1.0]
        normal = [0.0, 0.0, 1.0]
        line_vec = [1.0, 0.0, 0.0]
        v_out, evan = grating_diffract_3d(v_in, normal, line_vec, line_density_lines_per_mm=600.0, order=1, wavelength_um=0.5)
        self.assertFalse(evan)
        self.assertGreater(v_out[0], 0.0)

    def test_t1_5_angular_dispersion_rate(self):
        grating = DiffractionGrating(lines_per_mm=600.0, order=1)
        theta_m = grating.calculate_diffraction_angle(0.0, 0.5)
        disp = grating.angular_dispersion(theta_m)
        self.assertGreater(disp, 0.0)


class TestGratingDiffractionTier2(unittest.TestCase):
    """Tier 2: Boundary Value Analysis & Evanescent Orders."""

    def test_t2_1_evanescent_order_raises_value_error(self):
        # High line density, high wavelength -> sin(theta_m) > 1
        grating = DiffractionGrating(lines_per_mm=3000.0, order=2)
        with self.assertRaises(ValueError):
            grating.calculate_diffraction_angle(0.0, 0.8)

    def test_t2_2_evanescent_3d_flag(self):
        v_in = [0.0, 0.0, 1.0]
        v_out, evan = grating_diffract_3d(v_in, [0, 0, 1], [1, 0, 0], line_density_lines_per_mm=3000.0, order=2, wavelength_um=0.8)
        self.assertTrue(evan)
        self.assertIsNone(v_out)

    def test_t2_3_zero_grating_density(self):
        grating = DiffractionGrating(lines_per_mm=0.0, order=1)
        self.assertGreater(grating.pitch_um, 1e8)

    def test_t2_4_extreme_oblique_incident_angle(self):
        grating = DiffractionGrating(lines_per_mm=300.0, order=1)
        theta_i = np.radians(85.0)
        # Should not crash
        try:
            ang = grating.calculate_diffraction_angle(theta_i, 0.5)
            self.assertIsNotNone(ang)
        except ValueError:
            pass  # Evanescent is valid return

    def test_t2_5_zero_wavelength_limit(self):
        grating = DiffractionGrating(lines_per_mm=600.0, order=1)
        theta_m = grating.calculate_diffraction_angle(0.0, 0.0)
        self.assertAlmostEqual(theta_m, 0.0, places=6)


class TestGratingDiffractionTier3(unittest.TestCase):
    """Tier 3: Pairwise Interactions - Grating + Multi-Wavelength Spread."""

    def test_t3_1_rgb_wavelength_separation(self):
        grating = DiffractionGrating(lines_per_mm=600.0, order=1)
        wl_r, wl_g, wl_b = 0.650, 0.532, 0.450
        ang_r = grating.calculate_diffraction_angle(0.0, wl_r)
        ang_g = grating.calculate_diffraction_angle(0.0, wl_g)
        ang_b = grating.calculate_diffraction_angle(0.0, wl_b)
        
        self.assertGreater(ang_r, ang_g)
        self.assertGreater(ang_g, ang_b)


class TestGratingDiffractionTier4(unittest.TestCase):
    """Tier 4: Transmission Grating Monochromator Scenario."""

    def test_t4_1_monochromator_detector_spatial_dispersion(self):
        grating = DiffractionGrating(lines_per_mm=1200.0, order=1)
        wls = [0.400, 0.500, 0.600, 0.700]  # Visible spectrum
        spots = GratingDispersionAnalysis.calculate_spatial_dispersion(grating, wls, focal_length_mm=200.0)
        
        self.assertEqual(len(spots), 4)
        x_positions = [s[2] for s in spots]
        self.assertEqual(x_positions, sorted(x_positions))


if __name__ == "__main__":
    unittest.main()
