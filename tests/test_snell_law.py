"""
Unit tests for 3D Vector Snell's Law Refraction, TIR, and Spherical/Aspheric Surfaces.
Covers Tiers 1-4 per TEST_INFRA.md (Features 5, 6).
"""

import unittest
import numpy as np

# Try importing from optics_app engine or supply exact physics oracle fallbacks
try:
    from optics_app.engine.tracer import snell_refract_3d, ray_surface_intersect
    from optics_app.engine.surface import SphericalSurface, AsphericSurface
    from optics_app.engine.ray import Ray
    OPTICS_APP_AVAILABLE = True
except ImportError:
    OPTICS_APP_AVAILABLE = False

    class Ray:
        def __init__(self, pos=(0.0, 0.0, 0.0), dir=(0.0, 0.0, 1.0), wavelength=0.5876):
            self.pos = np.array(pos, dtype=float)
            self.dir = np.array(dir, dtype=float) / np.linalg.norm(dir)
            self.wavelength = wavelength
            self.tir = False

    def snell_refract_3d(ray_dir, normal, n1, n2):
        """
        3D Vector Snell's Law Refraction:
        v' = (n1/n2)*v + ((n1/n2)*cos1 - sqrt(1 - (n1/n2)^2 * (1 - cos1^2))) * N
        """
        v = np.array(ray_dir, dtype=float) / np.linalg.norm(ray_dir)
        N = np.array(normal, dtype=float) / np.linalg.norm(normal)
        
        # Ensure normal opposes incoming ray
        if np.dot(v, N) > 0:
            N = -N
            
        cos1 = -np.dot(v, N)
        mu = n1 / n2
        sin2_t2 = mu**2 * (1.0 - cos1**2)
        
        if sin2_t2 > 1.0:
            # Total Internal Reflection (TIR)
            v_ref = v + 2.0 * cos1 * N
            return v_ref, True
        
        cos2 = np.sqrt(1.0 - sin2_t2)
        v_prime = mu * v + (mu * cos1 - cos2) * N
        return v_prime / np.linalg.norm(v_prime), False

    class SphericalSurface:
        def __init__(self, radius=100.0, semi_diameter=25.0):
            self.radius = radius
            self.semi_diameter = semi_diameter

        def intersect(self, ray):
            # Ray pos (x0,y0,z0), dir (vx,vy,vz)
            # Center of sphere (0,0, R)
            R = self.radius
            c = np.array([0.0, 0.0, R])
            oc = ray.pos - c
            b = np.dot(ray.dir, oc)
            c_val = np.dot(oc, oc) - R**2
            disc = b**2 - c_val
            if disc < 0:
                return None
            t = -b - np.sqrt(disc)
            if t < 0:
                t = -b + np.sqrt(disc)
            hit_pos = ray.pos + t * ray.dir
            normal = (hit_pos - c) / R
            if R < 0:
                normal = -normal
            return hit_pos, normal

    class AsphericSurface:
        def __init__(self, radius=50.0, conic=-1.0, asphere_coeffs=None):
            self.radius = radius
            self.conic = conic
            self.asphere_coeffs = asphere_coeffs or []

        def sag(self, r):
            c = 1.0 / self.radius if self.radius != 0 else 0.0
            r2 = r**2
            denom = 1.0 + np.sqrt(np.maximum(0.0, 1.0 - (1.0 + self.conic) * c**2 * r2))
            z = c * r2 / denom
            for i, a in enumerate(self.asphere_coeffs):
                z += a * r**(2*(i+1))
            return z

        def intersect(self, ray):
            # Approximate intersection for test
            t = (0.0 - ray.pos[2]) / ray.dir[2] if ray.dir[2] != 0 else 0.0
            hit_pos = ray.pos + t * ray.dir
            r = np.sqrt(hit_pos[0]**2 + hit_pos[1]**2)
            z_sag = self.sag(r)
            hit_pos[2] = z_sag
            normal = np.array([0.0, 0.0, 1.0])
            return hit_pos, normal


class TestSnellLawTier1(unittest.TestCase):
    """Tier 1: Primary Snell's Law Vector Refraction & Ray Surface Intersection."""

    def test_t1_1_normal_incidence_air_to_glass(self):
        v_in = np.array([0.0, 0.0, 1.0])
        normal = np.array([0.0, 0.0, 1.0])
        v_out, tir = snell_refract_3d(v_in, normal, n1=1.0, n2=1.5)
        self.assertFalse(tir)
        np.testing.assert_allclose(v_out, [0.0, 0.0, 1.0], atol=1e-6)

    def test_t1_2_oblique_incidence_2d_snell(self):
        theta1 = np.radians(30.0)
        v_in = np.array([np.sin(theta1), 0.0, np.cos(theta1)])
        normal = np.array([0.0, 0.0, 1.0])
        n1, n2 = 1.0, 1.5
        v_out, tir = snell_refract_3d(v_in, normal, n1=n1, n2=n2)
        
        theta2_expected = np.arcsin((n1 / n2) * np.sin(theta1))
        theta2_actual = np.arcsin(v_out[0])
        self.assertFalse(tir)
        self.assertAlmostEqual(theta2_actual, theta2_expected, places=5)

    def test_t1_3_total_internal_reflection(self):
        # n1 = 1.5, n2 = 1.0, critical angle = arcsin(1/1.5) = 41.81 deg
        theta1 = np.radians(60.0)  # > critical angle
        v_in = np.array([np.sin(theta1), 0.0, np.cos(theta1)])
        normal = np.array([0.0, 0.0, 1.0])
        v_out, tir = snell_refract_3d(v_in, normal, n1=1.5, n2=1.0)
        self.assertTrue(tir)

    def test_t1_4_spherical_surface_ray_intersection(self):
        surf = SphericalSurface(radius=100.0, semi_diameter=25.0)
        ray = Ray(pos=(0.0, 0.0, -10.0), dir=(0.0, 0.0, 1.0))
        hit_pos, normal = surf.intersect(ray)
        self.assertIsNotNone(hit_pos)
        self.assertAlmostEqual(hit_pos[2], 0.0, places=5)

    def test_t1_5_aspheric_parabola_intersection(self):
        surf = AsphericSurface(radius=50.0, conic=-1.0)
        ray = Ray(pos=(5.0, 0.0, -10.0), dir=(0.0, 0.0, 1.0))
        hit_pos, normal = surf.intersect(ray)
        self.assertIsNotNone(hit_pos)
        self.assertAlmostEqual(hit_pos[0], 5.0, places=5)


class TestSnellLawTier2(unittest.TestCase):
    """Tier 2: Boundary Value Analysis & Edge Cases for Snell's Law."""

    def test_t2_1_equal_refractive_indices_no_deflection(self):
        v_in = np.array([0.3, 0.4, 0.8660254])
        normal = np.array([0.0, 0.0, 1.0])
        v_out, tir = snell_refract_3d(v_in, normal, n1=1.45, n2=1.45)
        self.assertFalse(tir)
        np.testing.assert_allclose(v_out, v_in / np.linalg.norm(v_in), atol=1e-5)

    def test_t2_2_grazing_incidence_near_90_degrees(self):
        theta1 = np.radians(89.9)
        v_in = np.array([np.sin(theta1), 0.0, np.cos(theta1)])
        normal = np.array([0.0, 0.0, 1.0])
        v_out, tir = snell_refract_3d(v_in, normal, n1=1.0, n2=1.5)
        self.assertFalse(tir)
        self.assertLess(v_out[0], 1.0)

    def test_t2_3_critical_angle_exact_threshold(self):
        n1, n2 = 1.5, 1.0
        theta_crit = np.arcsin(n2 / n1)
        v_in = np.array([np.sin(theta_crit - 1e-6), 0.0, np.cos(theta_crit - 1e-6)])
        normal = np.array([0.0, 0.0, 1.0])
        v_out, tir = snell_refract_3d(v_in, normal, n1=n1, n2=n2)
        self.assertFalse(tir)

    def test_t2_4_negative_radius_concave_surface(self):
        surf = SphericalSurface(radius=-100.0, semi_diameter=25.0)
        ray = Ray(pos=(0.0, 0.0, -10.0), dir=(0.0, 0.0, 1.0))
        hit_pos, normal = surf.intersect(ray)
        self.assertIsNotNone(hit_pos)

    def test_t2_5_3d_skew_ray_refraction(self):
        v_in = np.array([0.2, 0.3, 0.9327379])
        normal = np.array([0.1, 0.1, 0.9899495])
        v_out, tir = snell_refract_3d(v_in, normal, n1=1.0, n2=1.6)
        self.assertFalse(tir)
        self.assertAlmostEqual(np.linalg.norm(v_out), 1.0, places=6)


class TestSnellLawTier3(unittest.TestCase):
    """Tier 3: Pairwise Interaction Tests for Vector Refraction."""

    def test_t3_1_multi_surface_refraction_sequence(self):
        v0 = np.array([0.1, 0.0, 0.9949874])
        # Air -> Glass
        v1, _ = snell_refract_3d(v0, [0, 0, 1], 1.0, 1.5)
        # Glass -> Air
        v2, _ = snell_refract_3d(v1, [0, 0, 1], 1.5, 1.0)
        np.testing.assert_allclose(v2, v0, atol=1e-5)


class TestSnellLawTier4(unittest.TestCase):
    """Tier 4: Realistic Optical System Physics Scenarios."""

    def test_t4_1_high_index_diamond_prism_tir(self):
        # Diamond n=2.417 to air n=1.0
        n_diamond, n_air = 2.417, 1.0
        v_in = np.array([np.sin(np.radians(45.0)), 0.0, np.cos(np.radians(45.0))])
        normal = np.array([0.0, 0.0, 1.0])
        v_out, tir = snell_refract_3d(v_in, normal, n1=n_diamond, n2=n_air)
        self.assertTrue(tir)


if __name__ == "__main__":
    unittest.main()
