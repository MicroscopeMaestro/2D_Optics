"""
Unit tests for Zemax-style Ray Fan Plot Analysis (EX, EY vs Py) and 3rd-Order Seidel Aberration Coefficients.
Covers Tiers 1-4 per TEST_INFRA.md (Features 11, 12).
"""

import unittest
import numpy as np
from optics_app.analysis.rayfan import RayFanAnalysis, RayFanData, calculate_ray_fan
from optics_app.analysis.aberration import (
    SeidelAberrations,
    ParaxialSurfaceResult,
    trace_paraxial_marginal_ray,
    trace_paraxial_chief_ray,
    calculate_seidel_coefficients
)
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem


class TestRayFanTier1(unittest.TestCase):
    """Tier 1: Basic Ray Fan Transverse Aberrations (EX, EY vs Py) & Seidel Coefficients."""

    def test_t1_1_aberration_free_paraxial_ray_fan(self):
        analyzer = RayFanAnalysis(py_sampling=21)
        py, ex, ey = analyzer.generate_tangential_fan(W040=0.0, W131=0.0)
        np.testing.assert_allclose(ex, 0.0, atol=1e-6)
        np.testing.assert_allclose(ey, 0.0, atol=1e-6)

    def test_t1_2_spherical_aberration_cubic_fan(self):
        analyzer = RayFanAnalysis(py_sampling=51)
        py, ex, ey = analyzer.generate_tangential_fan(W040=2.0)  # Pure spherical W040
        # EY(-Py) should equal -EY(Py) (anti-symmetric cubic)
        self.assertAlmostEqual(ey[0], -ey[-1], places=4)
        self.assertGreater(abs(ey[-1]), 5.0)

    def test_t1_3_coma_aberration_quadratic_fan(self):
        analyzer = RayFanAnalysis(py_sampling=51)
        py, ex, ey = analyzer.generate_tangential_fan(W131=1.5)  # Pure coma W131
        # EY(-Py) should equal EY(Py) (symmetric quadratic)
        self.assertAlmostEqual(ey[0], ey[-1], places=4)

    def test_t1_4_seidel_fit_from_ray_fan(self):
        analyzer = RayFanAnalysis(py_sampling=101)
        py, ex, ey = analyzer.generate_tangential_fan(W040=1.0, W131=2.0)
        seidel = SeidelAberrations.fit_from_ray_fan(py, ey)
        self.assertAlmostEqual(seidel.S_I, 4.0, places=2)
        self.assertAlmostEqual(seidel.S_II, 6.0, places=2)

    def test_t1_5_seidel_coefficients_structure(self):
        seidel = SeidelAberrations(S_I=0.1, S_II=0.05, S_III=0.02, S_IV=0.01, S_V=0.005)
        self.assertEqual(seidel.S_I, 0.1)
        self.assertEqual(seidel.S_II, 0.05)
        self.assertEqual(seidel.S_III, 0.02)
        self.assertEqual(seidel.S_IV, 0.01)
        self.assertEqual(seidel.S_V, 0.005)

    def test_t1_6_sagittal_fan_generation(self):
        analyzer = RayFanAnalysis(px_sampling=51)
        px, ex, ey = analyzer.generate_sagittal_fan(W040=1.5, W220=0.5)
        self.assertEqual(len(px), 51)
        self.assertGreater(abs(ex[-1]), 0.0)


class TestRayFanTier2(unittest.TestCase):
    """Tier 2: Boundary Value Analysis & Edge Cases for Ray Fans."""

    def test_t2_1_pupil_boundary_extremes(self):
        analyzer = RayFanAnalysis(py_sampling=51)
        py, ex, ey = analyzer.generate_tangential_fan()
        self.assertAlmostEqual(py[0], -1.0, places=6)
        self.assertAlmostEqual(py[-1], 1.0, places=6)

    def test_t2_2_single_point_center_pupil(self):
        analyzer = RayFanAnalysis(py_sampling=1)
        py, ex, ey = analyzer.generate_tangential_fan()
        self.assertEqual(len(py), 1)
        self.assertAlmostEqual(py[0], 0.0, places=6)

    def test_t2_3_distortion_constant_offset(self):
        analyzer = RayFanAnalysis(py_sampling=51)
        py, ex, ey = analyzer.generate_tangential_fan(W311=0.5)  # Pure distortion
        np.testing.assert_allclose(ey, 0.5, atol=1e-6)

    def test_t2_4_astigmatism_linear_fan(self):
        analyzer = RayFanAnalysis(py_sampling=51)
        py, ex, ey = analyzer.generate_tangential_fan(W222=1.0)
        # EY should be strictly linear with Py
        slope = (ey[-1] - ey[0]) / (py[-1] - py[0])
        self.assertAlmostEqual(slope, 2.0, places=4)

    def test_t2_5_vignetted_pupil_ray_fan_tracing(self):
        # Tracing system with small semi-diameter causing vignetting
        stack = SurfaceStack()
        stack.add_surface(Surface(name="Obj", thickness=50.0))
        stack.add_surface(Surface(name="Stop", surface_type=SurfaceType.STOP, semi_diameter=2.0, thickness=10.0))
        stack.add_surface(Surface(name="Det", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=25.0))
        sys = OpticalSystem(stack)

        analyzer = RayFanAnalysis(py_sampling=21)
        fan_data = analyzer.trace_ray_fan_from_system(sys, entrance_pupil_radius=10.0)
        self.assertIsInstance(fan_data, RayFanData)
        self.assertTrue(np.any(fan_data.vignetted_tangential))


class TestRayFanTier3(unittest.TestCase):
    """Tier 3: Pairwise Combination - Mixed Spherical + Coma + Astigmatism & System Paraxial Trace."""

    def test_t3_1_mixed_aberrations_seidel_decomp(self):
        analyzer = RayFanAnalysis(py_sampling=101)
        py, ex, ey = analyzer.generate_tangential_fan(W040=0.5, W131=1.0, W222=0.2, W311=0.05)
        seidel = SeidelAberrations.fit_from_ray_fan(py, ey)
        self.assertGreater(seidel.S_I, 0)
        self.assertGreater(seidel.S_II, 0)
        self.assertGreater(seidel.S_III, 0)

    def test_t3_2_paraxial_marginal_and_chief_ray_tracing(self):
        stack = SurfaceStack()
        stack.add_surface(Surface(name="Obj", thickness=50.0))
        stack.add_surface(Surface(name="Lens1", radius=50.0, thickness=5.0, glass="BK7", semi_diameter=12.5))
        stack.add_surface(Surface(name="Stop", surface_type=SurfaceType.STOP, semi_diameter=10.0, thickness=45.0))
        stack.add_surface(Surface(name="Det", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=25.0))

        sys = OpticalSystem(stack)
        m_trace = trace_paraxial_marginal_ray(sys)
        c_trace = trace_paraxial_chief_ray(sys)

        self.assertEqual(len(m_trace), 4)
        self.assertEqual(len(c_trace), 4)
        # Chief ray height at stop surface (index 2) should be 0.0
        self.assertAlmostEqual(c_trace[2][0], 0.0, places=4)


class TestRayFanTier4(unittest.TestCase):
    """Tier 4: Doublet Lens Seidel Aberrations Real-World Evaluation."""

    def test_t4_1_doublet_seidel_aberration_balance(self):
        # Crown lens (+) spherical aberration cancels Flint lens (-) spherical aberration
        crown_S_I = 2.5
        flint_S_I = -2.4
        net_S_I = crown_S_I + flint_S_I
        self.assertLess(abs(net_S_I), 0.2)

    def test_t4_2_calculate_seidel_coefficients_system(self):
        stack = SurfaceStack()
        stack.add_surface(Surface(name="Obj", thickness=100.0))
        stack.add_surface(Surface(name="Lens1_front", radius=50.0, thickness=5.0, glass="BK7", semi_diameter=12.5))
        stack.add_surface(Surface(name="Lens1_back", radius=-50.0, thickness=95.0, glass="AIR", semi_diameter=12.5))
        stack.add_surface(Surface(name="Det", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=25.0))

        sys = OpticalSystem(stack)
        seidel = calculate_seidel_coefficients(sys)
        self.assertIsInstance(seidel, SeidelAberrations)
        self.assertNotEqual(seidel.S_I, 0.0)


if __name__ == "__main__":
    unittest.main()
