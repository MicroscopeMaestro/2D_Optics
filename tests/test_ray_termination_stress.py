"""
Adversarial Stress Test Suite for Ray Termination (Milestone M1)
Author: Challenger 1 (Empirical Challenger)
Coverage:
1. System with 5+ dummy/active surfaces (lenses, mirrors, gratings, aspheres) placed after SurfaceType.DETECTOR.
2. Extreme angle rays hitting 3D tilted/decentered detector.
3. Rays missing detector aperture or vignetted before detector.
4. Systems with multiple sequential detectors.
5. Spot diagram and RMS radius invariance with post-detector surfaces.
6. Ray fan transverse aberration invariance with post-detector surfaces.
7. Aspheric detector surface profile with termination.
8. Minimal systems (Object + Detector only, or standalone Detector).
9. Polarization, Jones vectors, and phase accumulation integrity at termination boundary.
10. Large-scale Monte Carlo stress test (1,000 rays) verifying history invariants.
"""

import math
import numpy as np
import pytest

from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.ray import Ray
from optics_app.engine.tracer import OpticalSystem
from optics_app.analysis.spot import SpotDiagram, generate_spot_diagram_from_system, calculate_spot_metrics
from optics_app.analysis.rayfan import RayFanAnalysis


def create_system_with_post_detector_surfaces():
    """Create an optical system with 5 surfaces placed after DETECTOR."""
    stack = SurfaceStack()
    # 0: Object
    stack.add_surface(Surface(name="OBJ", surface_type=SurfaceType.OBJECT, thickness=20.0))
    # 1: Lens Front (Convex)
    stack.add_surface(Surface(name="L1_Front", surface_type=SurfaceType.STANDARD, radius=40.0, thickness=6.0, glass="N-BK7", semi_diameter=20.0))
    # 2: Lens Back (Convex)
    stack.add_surface(Surface(name="L1_Back", surface_type=SurfaceType.STANDARD, radius=-60.0, thickness=35.0, glass="AIR", semi_diameter=20.0))
    # 3: DETECTOR (Surface index 3)
    stack.add_surface(Surface(name="DETECTOR", surface_type=SurfaceType.DETECTOR, radius=0.0, thickness=15.0, semi_diameter=30.0))
    # 4: Post-detector Dummy Lens Front
    stack.add_surface(Surface(name="POST_L2_Front", surface_type=SurfaceType.STANDARD, radius=25.0, thickness=5.0, glass="SF11", semi_diameter=15.0))
    # 5: Post-detector Dummy Lens Back
    stack.add_surface(Surface(name="POST_L2_Back", surface_type=SurfaceType.STANDARD, radius=-25.0, thickness=10.0, glass="AIR", semi_diameter=15.0))
    # 6: Post-detector Dummy Mirror
    stack.add_surface(Surface(name="POST_Mirror", surface_type=SurfaceType.STANDARD, radius=50.0, thickness=20.0, glass="MIRROR", semi_diameter=20.0))
    # 7: Post-detector Dummy Grating
    stack.add_surface(Surface(name="POST_Grating", surface_type=SurfaceType.GRATING, radius=0.0, thickness=10.0, grating_lines_per_mm=600.0, grating_order=1, semi_diameter=15.0))
    # 8: Post-detector Final Dummy Flat
    stack.add_surface(Surface(name="POST_Final", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=0.0, glass="AIR", semi_diameter=20.0))
    return stack


def test_adversarial_post_detector_surfaces_never_evaluated():
    """Test that rays never propagate to surfaces placed after DETECTOR."""
    stack = create_system_with_post_detector_surfaces()
    assert len(stack.surfaces) == 9
    sys = OpticalSystem(stack)

    # 1. Axial ray
    r_axial = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=0.5876)
    traced_axial = sys.trace_ray(r_axial)

    assert not traced_axial.vignetted
    # History must have exactly 4 entries: surfaces 0, 1, 2, 3
    assert len(traced_axial.history) == 4, f"Expected 4 history entries, got {len(traced_axial.history)}"
    
    # Assert each history step matches expected surface indices 0, 1, 2, 3
    recorded_indices = [h["surface_index"] for h in traced_axial.history]
    assert recorded_indices == [0, 1, 2, 3], f"Indices mismatch: {recorded_indices}"
    
    # history[-1] must be strictly the detector
    last_hit = traced_axial.history[-1]
    assert last_hit["surface_index"] == 3
    assert abs(last_hit["z_global"] - (20.0 + 6.0 + 35.0)) < 1e-6
    assert np.allclose(traced_axial.pos, np.array([last_hit["x_global"], last_hit["y_global"], last_hit["z_global"]]))

    # Ensure none of the post-detector indices (4, 5, 6, 7, 8) appear anywhere
    for post_idx in [4, 5, 6, 7, 8]:
        assert post_idx not in recorded_indices, f"Post-detector surface {post_idx} was touched!"


def test_adversarial_grid_and_marginal_rays_termination():
    """Test a 2D grid of rays covering the pupil with post-detector surfaces."""
    stack = create_system_with_post_detector_surfaces()
    sys = OpticalSystem(stack)

    x_vals = np.linspace(-10.0, 10.0, 11)
    y_vals = np.linspace(-10.0, 10.0, 11)

    rays = []
    for x in x_vals:
        for y in y_vals:
            rays.append(Ray(pos=np.array([x, y, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=0.5876))

    traced_rays = sys.trace_rays(rays)
    assert len(traced_rays) == 121

    for idx, r in enumerate(traced_rays):
        if not r.vignetted:
            assert len(r.history) == 4, f"Ray {idx} has {len(r.history)} history entries"
            assert r.history[-1]["surface_index"] == 3, f"Ray {idx} last surface index is {r.history[-1]['surface_index']}"
            for h in r.history:
                assert h["surface_index"] <= 3, f"Ray {idx} touched post-detector surface {h['surface_index']}"
        else:
            # If vignetted, history must not contain any surface > 3
            for h in r.history:
                assert h["surface_index"] <= 3


def test_adversarial_tilted_and_decentered_detector_extreme_angles():
    """Test extreme angle rays hitting tilted and decentered detector."""
    stack = SurfaceStack()
    stack.add_surface(Surface(name="OBJ", surface_type=SurfaceType.OBJECT, thickness=10.0))
    stack.add_surface(Surface(name="L1", surface_type=SurfaceType.STANDARD, radius=30.0, thickness=5.0, glass="N-BK7", semi_diameter=25.0))
    stack.add_surface(Surface(name="AIR", surface_type=SurfaceType.STANDARD, radius=-30.0, thickness=25.0, glass="AIR", semi_diameter=25.0))
    
    # Detector tilted by 25 deg around X, -15 deg around Y, and decentered by (3.0, -4.0)
    det = Surface(
        name="TILTED_DETECTOR",
        surface_type=SurfaceType.DETECTOR,
        radius=0.0,
        thickness=20.0,
        tilt_x=25.0,
        tilt_y=-15.0,
        decenter_x=3.0,
        decenter_y=-4.0,
        semi_diameter=50.0
    )
    stack.add_surface(det)
    
    # Add 3 dummy surfaces after tilted detector
    stack.add_surface(Surface(name="POST_1", surface_type=SurfaceType.STANDARD, radius=20.0, thickness=5.0, glass="SF11"))
    stack.add_surface(Surface(name="POST_2", surface_type=SurfaceType.STANDARD, radius=40.0, thickness=10.0, glass="MIRROR"))
    stack.add_surface(Surface(name="POST_3", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=0.0))

    sys = OpticalSystem(stack)

    # Test various extreme field angles (up to 45 degrees)
    angles_deg = [-45.0, -30.0, -15.0, 0.0, 15.0, 30.0, 45.0]
    for angle_x in angles_deg:
        for angle_y in angles_deg:
            theta_x = np.radians(angle_x)
            theta_y = np.radians(angle_y)
            dir_vec = np.array([np.sin(theta_x), np.sin(theta_y), np.cos(theta_x) * np.cos(theta_y)])
            dir_vec /= np.linalg.norm(dir_vec)

            r = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=dir_vec, wavelength=0.5876)
            traced = sys.trace_ray(r)

            if not traced.vignetted:
                assert traced.history[-1]["surface_index"] == 3
                assert len(traced.history) == 4
                for h in traced.history:
                    assert h["surface_index"] <= 3
                
                # Check that local Z on flat detector is close to 0
                last_hit = traced.history[-1]
                assert abs(last_hit["z_local"]) < 1e-6
                
                # Check coordinate consistency
                R_mat = det.global_R
                orig = det.global_origin
                hit_local = np.array([last_hit["x_local"], last_hit["y_local"], last_hit["z_local"]])
                expected_global = orig + R_mat @ hit_local
                actual_global = np.array([last_hit["x_global"], last_hit["y_global"], last_hit["z_global"]])
                assert np.allclose(actual_global, expected_global, atol=1e-6)


def test_adversarial_vignetted_and_missing_rays():
    """Test rays missing apertures or vignetted before/at detector with post-detector surfaces."""
    stack = create_system_with_post_detector_surfaces()
    sys = OpticalSystem(stack)

    # 1. Ray starting far outside lens semi-diameter (r = 50 mm, lens semi_diam = 20 mm)
    r_wide = Ray(pos=np.array([50.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]))
    traced_wide = sys.trace_ray(r_wide)
    assert traced_wide.vignetted is True
    # Should be vignetted at lens surface (index 1), never reaching detector or post-detector surfaces
    assert len(traced_wide.history) <= 2
    for h in traced_wide.history:
        assert h["surface_index"] < 3

    # 2. Ray passing through lens but aimed so it misses detector aperture (semi_diameter=30.0)
    # Give it an extreme angle from the object edge
    dir_diverge = np.array([0.8, 0.0, 0.6])
    dir_diverge /= np.linalg.norm(dir_diverge)
    r_diverge = Ray(pos=np.array([15.0, 0.0, 0.0]), dir=dir_diverge)
    traced_diverge = sys.trace_ray(r_diverge)
    assert traced_diverge.vignetted is True
    # No post-detector surface should EVER be logged
    for h in traced_diverge.history:
        assert h["surface_index"] <= 3

    # 3. Ray pointing backwards (-Z)
    r_back = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, -1.0]))
    traced_back = sys.trace_ray(r_back)
    assert traced_back.vignetted is True
    assert len(traced_back.history) == 0 or traced_back.history[0]["surface_index"] == 0
    for h in traced_back.history:
        assert h["surface_index"] < 3


def test_adversarial_multiple_detector_surfaces():
    """Test optical system containing multiple sequential DETECTOR surfaces."""
    stack = SurfaceStack()
    stack.add_surface(Surface(name="OBJ", surface_type=SurfaceType.OBJECT, thickness=20.0))
    stack.add_surface(Surface(name="L1", surface_type=SurfaceType.STANDARD, radius=50.0, thickness=5.0, glass="N-BK7"))
    stack.add_surface(Surface(name="AIR1", surface_type=SurfaceType.STANDARD, radius=-50.0, thickness=30.0, glass="AIR"))
    
    # First Detector at index 3
    stack.add_surface(Surface(name="DETECTOR_1", surface_type=SurfaceType.DETECTOR, radius=0.0, thickness=15.0))
    
    # Intermediate dummy lens
    stack.add_surface(Surface(name="L2", surface_type=SurfaceType.STANDARD, radius=40.0, thickness=5.0, glass="N-BK7"))
    stack.add_surface(Surface(name="AIR2", surface_type=SurfaceType.STANDARD, radius=-40.0, thickness=25.0, glass="AIR"))
    
    # Second Detector at index 6
    stack.add_surface(Surface(name="DETECTOR_2", surface_type=SurfaceType.DETECTOR, radius=0.0, thickness=10.0))
    
    # Post-detector surface at index 7
    stack.add_surface(Surface(name="POST_SURF", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=0.0))

    sys = OpticalSystem(stack)
    r = Ray(pos=np.array([0.0, 1.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]))
    traced = sys.trace_ray(r)

    assert not traced.vignetted
    # Must strictly terminate at DETECTOR_1 (index 3)
    assert len(traced.history) == 4
    assert traced.history[-1]["surface_index"] == 3
    assert traced.history[-1]["surface_index"] != 6
    for h in traced.history:
        assert h["surface_index"] <= 3


def test_adversarial_spot_diagram_metrics_with_post_detector_stack():
    """Test that SpotDiagram and RMS radius calculation work identically with or without post-detector surfaces."""
    # Clean system without post-detector surfaces
    stack_clean = SurfaceStack()
    stack_clean.add_surface(Surface(name="OBJ", surface_type=SurfaceType.OBJECT, thickness=20.0))
    stack_clean.add_surface(Surface(name="L1_F", surface_type=SurfaceType.STANDARD, radius=40.0, thickness=6.0, glass="N-BK7", semi_diameter=20.0))
    stack_clean.add_surface(Surface(name="L1_B", surface_type=SurfaceType.STANDARD, radius=-60.0, thickness=35.0, glass="AIR", semi_diameter=20.0))
    stack_clean.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, radius=0.0, thickness=0.0, semi_diameter=30.0))
    
    # Polluted system with 5 post-detector surfaces
    stack_polluted = create_system_with_post_detector_surfaces()

    sys_clean = OpticalSystem(stack_clean)
    sys_polluted = OpticalSystem(stack_polluted)

    spot_clean = generate_spot_diagram_from_system(sys_clean, num_rays=80, entrance_pupil_radius=8.0)
    spot_polluted = generate_spot_diagram_from_system(sys_polluted, num_rays=80, entrance_pupil_radius=8.0)

    assert len(spot_clean.x) > 0
    assert len(spot_clean.x) == len(spot_polluted.x)
    assert np.allclose(spot_clean.x, spot_polluted.x, atol=1e-12)
    assert np.allclose(spot_clean.y, spot_polluted.y, atol=1e-12)

    rms_clean = spot_clean.calculate_rms_radius()
    rms_polluted = spot_polluted.calculate_rms_radius()
    assert abs(rms_clean - rms_polluted) < 1e-12, f"RMS radius mismatch: clean={rms_clean}, polluted={rms_polluted}"

    geo_clean = spot_clean.calculate_geo_radius()
    geo_polluted = spot_polluted.calculate_geo_radius()
    assert abs(geo_clean - geo_polluted) < 1e-12

    c_clean = spot_clean.calculate_centroid()
    c_polluted = spot_polluted.calculate_centroid()
    assert abs(c_clean[0] - c_polluted[0]) < 1e-12
    assert abs(c_clean[1] - c_polluted[1]) < 1e-12


def test_adversarial_ray_fan_invariance_with_post_detector_surfaces():
    """Test that RayFan analysis gives identical results whether or not dummy surfaces trail the detector."""
    stack_clean = SurfaceStack()
    stack_clean.add_surface(Surface(name="OBJ", surface_type=SurfaceType.OBJECT, thickness=20.0))
    stack_clean.add_surface(Surface(name="L1_F", surface_type=SurfaceType.STANDARD, radius=40.0, thickness=6.0, glass="N-BK7", semi_diameter=20.0))
    stack_clean.add_surface(Surface(name="L1_B", surface_type=SurfaceType.STANDARD, radius=-60.0, thickness=35.0, glass="AIR", semi_diameter=20.0))
    stack_clean.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, radius=0.0, thickness=0.0, semi_diameter=30.0))

    stack_polluted = create_system_with_post_detector_surfaces()

    sys_clean = OpticalSystem(stack_clean)
    sys_polluted = OpticalSystem(stack_polluted)

    analyzer = RayFanAnalysis(py_sampling=31, px_sampling=31)
    fan_clean = analyzer.trace_ray_fan_from_system(sys_clean, entrance_pupil_radius=10.0)
    fan_polluted = analyzer.trace_ray_fan_from_system(sys_polluted, entrance_pupil_radius=10.0)

    # Tangential aberrations
    np.testing.assert_allclose(fan_clean.tangential_ex, fan_polluted.tangential_ex, atol=1e-12)
    np.testing.assert_allclose(fan_clean.tangential_ey, fan_polluted.tangential_ey, atol=1e-12)
    np.testing.assert_array_equal(fan_clean.vignetted_tangential, fan_polluted.vignetted_tangential)

    # Sagittal aberrations
    np.testing.assert_allclose(fan_clean.sagittal_ex, fan_polluted.sagittal_ex, atol=1e-12)
    np.testing.assert_allclose(fan_clean.sagittal_ey, fan_polluted.sagittal_ey, atol=1e-12)
    np.testing.assert_array_equal(fan_clean.vignetted_sagittal, fan_polluted.vignetted_sagittal)


def test_adversarial_curved_and_aspheric_detector_termination():
    """Test curved and aspheric detector surface geometry terminates propagation."""
    stack = SurfaceStack()
    stack.add_surface(Surface(name="OBJ", surface_type=SurfaceType.OBJECT, thickness=15.0))
    stack.add_surface(Surface(name="L1", surface_type=SurfaceType.STANDARD, radius=30.0, thickness=5.0, glass="N-BK7", semi_diameter=20.0))
    stack.add_surface(Surface(name="AIR", surface_type=SurfaceType.STANDARD, radius=-30.0, thickness=20.0, glass="AIR", semi_diameter=20.0))
    
    # Curved / aspheric detector
    det_asph = Surface(
        name="CURVED_DETECTOR",
        surface_type=SurfaceType.DETECTOR,
        radius=50.0,
        conic=-1.0,
        asphere_coeffs=[1e-6, -2e-9],
        thickness=10.0,
        semi_diameter=25.0
    )
    stack.add_surface(det_asph)
    # Trailing dummy surfaces
    stack.add_surface(Surface(name="POST_1", surface_type=SurfaceType.STANDARD, radius=20.0, thickness=10.0))
    stack.add_surface(Surface(name="POST_2", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=0.0))

    sys = OpticalSystem(stack)
    for y_offset in [0.0, 2.0, -5.0, 8.0]:
        r = Ray(pos=np.array([0.0, y_offset, 0.0]), dir=np.array([0.0, 0.0, 1.0]))
        traced = sys.trace_ray(r)
        if not traced.vignetted:
            assert len(traced.history) == 4
            assert traced.history[-1]["surface_index"] == 3
            # Ensure post surfaces 4 and 5 were never reached
            for h in traced.history:
                assert h["surface_index"] <= 3


def test_adversarial_minimal_systems_termination():
    """Test minimal optical systems (Object + Detector, or Detector alone)."""
    # System A: Object + Detector + 2 Post surfaces
    stack_a = SurfaceStack()
    stack_a.add_surface(Surface(name="OBJ", surface_type=SurfaceType.OBJECT, thickness=50.0))
    stack_a.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, radius=0.0, thickness=20.0))
    stack_a.add_surface(Surface(name="POST1", surface_type=SurfaceType.STANDARD, radius=30.0, thickness=10.0))
    stack_a.add_surface(Surface(name="POST2", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=0.0))

    sys_a = OpticalSystem(stack_a)
    r_a = Ray(pos=np.array([2.0, 3.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]))
    traced_a = sys_a.trace_ray(r_a)

    assert not traced_a.vignetted
    assert len(traced_a.history) == 2
    assert [h["surface_index"] for h in traced_a.history] == [0, 1]
    assert traced_a.history[-1]["surface_index"] == 1

    # System B: Detector at Index 0 + Post surfaces
    stack_b = SurfaceStack()
    stack_b.add_surface(Surface(name="DET_FIRST", surface_type=SurfaceType.DETECTOR, radius=0.0, thickness=10.0))
    stack_b.add_surface(Surface(name="POST1", surface_type=SurfaceType.STANDARD, radius=20.0, thickness=5.0))
    stack_b.add_surface(Surface(name="POST2", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=0.0))

    sys_b = OpticalSystem(stack_b)
    r_b = Ray(pos=np.array([1.0, -1.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]))
    traced_b = sys_b.trace_ray(r_b)

    assert not traced_b.vignetted
    assert len(traced_b.history) == 1
    assert traced_b.history[0]["surface_index"] == 0


def test_adversarial_polarization_and_phase_integrity_at_termination():
    """Test that polarization Jones vector, intensity, and phase are properly finalized at detector."""
    stack = create_system_with_post_detector_surfaces()
    sys = OpticalSystem(stack)

    # Diagonal polarized ray
    j_in = np.array([1.0 / np.sqrt(2), 1.0 / np.sqrt(2)], dtype=complex)
    r = Ray(pos=np.array([0.0, 2.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=0.6328, jones_vector=j_in, intensity=1.0)

    traced = sys.trace_ray(r)
    assert not traced.vignetted
    assert len(traced.history) == 4
    det_history = traced.history[-1]

    assert det_history["surface_index"] == 3
    assert det_history["intensity"] > 0.0
    assert det_history["phase"] > 0.0
    assert np.allclose(traced.jones_vector, det_history["jones_vector"])
    assert abs(traced.intensity - det_history["intensity"]) < 1e-12
    assert abs(traced.phase - det_history["phase"]) < 1e-12


def test_adversarial_monte_carlo_random_rays():
    """Stress test with 1,000 random rays through optical system with post-detector surfaces."""
    np.random.seed(999)
    stack = create_system_with_post_detector_surfaces()
    sys = OpticalSystem(stack)

    num_rays = 1000
    rays = []
    for _ in range(num_rays):
        # Random entrance pupil position
        r_pupil = np.random.uniform(0.0, 18.0)
        phi_pupil = np.random.uniform(0.0, 2.0 * math.pi)
        px = r_pupil * math.cos(phi_pupil)
        py = r_pupil * math.sin(phi_pupil)

        # Random small field angle (-5 deg to +5 deg)
        ang_x = np.random.uniform(-5.0, 5.0)
        ang_y = np.random.uniform(-5.0, 5.0)
        tx = math.radians(ang_x)
        ty = math.radians(ang_y)
        dz = math.sqrt(max(0.0, 1.0 - math.sin(tx)**2 - math.sin(ty)**2))
        d_vec = np.array([math.sin(tx), math.sin(ty), dz])
        d_vec /= np.linalg.norm(d_vec)

        # Random wavelength (visible spectrum: 0.4 to 0.7 um)
        wl = np.random.uniform(0.400, 0.700)

        rays.append(Ray(pos=np.array([px, py, 0.0]), dir=d_vec, wavelength=wl))

    traced_list = sys.trace_rays(rays)
    assert len(traced_list) == num_rays

    valid_count = 0
    vignetted_count = 0

    for i, r in enumerate(traced_list):
        if not r.vignetted:
            valid_count += 1
            assert len(r.history) == 4, f"Ray {i} history length is {len(r.history)}, expected 4"
            assert r.history[-1]["surface_index"] == 3, f"Ray {i} last surface index is {r.history[-1]['surface_index']}"
            # Check all history surface indices are in {0, 1, 2, 3}
            for step in r.history:
                assert step["surface_index"] in (0, 1, 2, 3)
                assert step["surface_index"] < 4
        else:
            vignetted_count += 1
            for step in r.history:
                assert step["surface_index"] < 4

    assert valid_count > 0, "Expected at least some valid rays in Monte Carlo test"
    print(f"Monte Carlo Test: {valid_count} valid rays, {vignetted_count} vignetted rays out of {num_rays}. All termination invariants verified!")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
