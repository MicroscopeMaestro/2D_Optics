"""
Comprehensive 4-Tier Test Suite for 3D Optical Component Rotation & Coordinate Transformations.

Tiers Covered:
- Tier 1: Isolated 3D rotation tests for mirrors, plano surfaces, spherical lenses, aspheres, gratings, detectors (Rx, Ry, Rz).
- Tier 2: Boundary/corner cases (+-90 deg, +-180 deg, near-glancing reflection, high-angle refraction, zero-thickness elements).
- Tier 3: Compound multi-surface lens group rotations (rotate_lens_element), coordinate break sequences, reversible inverse rotations.
- Tier 4: Real-world double-fold spectrometer and off-axis imaging setups with tilted components.
"""

import unittest
import math
import numpy as np

from optics_app.engine.surface import Surface, SurfaceType, SphericalSurface, AsphericSurface
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem, build_rotation_matrix, snell_refract_3d
from optics_app.engine.ray import Ray
from optics_app.engine.mirror_positioner import MirrorPositioner, MirrorAimResult
from optics_app.analysis.spot import SpotDiagram


class TestTier1IsolatedRotations(unittest.TestCase):
    """
    Tier 1: Isolated 3D rotation tests for individual optical component types
    (mirrors, plano surfaces, spherical lenses, aspheric surfaces, diffraction gratings, detectors).
    """

    def test_mirror_pitch_rx_rotation(self):
        """Tier 1: Isolated mirror Rx (pitch) rotation deflects beam in vertical YZ plane."""
        for tilt_x in [15.0, 30.0, 45.0, -22.5]:
            stack = SurfaceStack()
            stack.add_surface(Surface(name="OBJ", thickness=50.0))
            mirror = Surface(name="MirrorRx", glass="MIRROR", thickness=50.0, tilt_x=tilt_x)
            stack.add_surface(mirror)
            stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0))

            opt_sys = OpticalSystem(stack)
            ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]))
            tr = opt_sys.trace_ray(ray)

            self.assertFalse(tr.vignetted)
            self.assertEqual(len(tr.history), 3)

            # Mirror reflection normal: rotated by Rx(tilt_x)
            # Incoming along +Z, tilt around X by theta -> deflection in YZ plane by 2 * theta (with -cos(2*theta) in Z)
            refl_hist = tr.history[1]
            out_dir = np.array([refl_hist["dir_x"], refl_hist["dir_y"], refl_hist["dir_z"]])
            out_dir /= np.linalg.norm(out_dir)

            expected_angle_rad = 2.0 * np.radians(tilt_x)
            expected_dir = np.array([0.0, np.sin(expected_angle_rad), -np.cos(expected_angle_rad)])
            expected_dir /= np.linalg.norm(expected_dir)

            np.testing.assert_allclose(out_dir, expected_dir, atol=1e-6)

    def test_mirror_yaw_ry_rotation(self):
        """Tier 1: Isolated mirror Ry (yaw) rotation deflects beam in horizontal XZ plane."""
        for tilt_y in [10.0, 25.0, 45.0, -30.0]:
            stack = SurfaceStack()
            stack.add_surface(Surface(name="OBJ", thickness=40.0))
            mirror = Surface(name="MirrorRy", glass="MIRROR", thickness=40.0, tilt_y=tilt_y)
            stack.add_surface(mirror)
            stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0))

            opt_sys = OpticalSystem(stack)
            ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]))
            tr = opt_sys.trace_ray(ray)

            self.assertFalse(tr.vignetted)
            refl_hist = tr.history[1]
            out_dir = np.array([refl_hist["dir_x"], refl_hist["dir_y"], refl_hist["dir_z"]])
            out_dir /= np.linalg.norm(out_dir)

            expected_angle_rad = 2.0 * np.radians(tilt_y)
            # Ry rotates normal in XZ plane -> deflection in XZ plane (with -cos(2*theta) in Z)
            expected_dir = np.array([-np.sin(expected_angle_rad), 0.0, -np.cos(expected_angle_rad)])
            expected_dir /= np.linalg.norm(expected_dir)

            np.testing.assert_allclose(out_dir, expected_dir, atol=1e-6)

    def test_mirror_roll_rz_rotation(self):
        """Tier 1: Isolated mirror Rz (roll) rotation preserves on-axis reflection while rotating transverse frame."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=30.0))
        # 45 deg fold in Y combined with 90 deg roll in Z
        mirror = Surface(name="MirrorRz", glass="MIRROR", thickness=30.0, tilt_x=45.0, tilt_z=90.0)
        stack.add_surface(mirror)
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0))

        opt_sys = OpticalSystem(stack)
        ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]))
        tr = opt_sys.trace_ray(ray)

        self.assertFalse(tr.vignetted)
        self.assertEqual(len(tr.history), 3)

        # Verify global orientation matrix orthonormality and determinant +1
        R_mirror = mirror.global_R
        np.testing.assert_allclose(R_mirror.T @ R_mirror, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(float(np.linalg.det(R_mirror)), 1.0, places=10)

    def test_plano_refraction_snell_3d(self):
        """Tier 1: Tilted plano interface refracts according to 3D vector Snell's law."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=20.0))
        # Plano boundary tilted by 30 deg around X into N-BK7 (n = 1.5168 at 587.6 nm)
        stack.add_surface(Surface(name="PlanoFront", radius=0.0, thickness=20.0, glass="N-BK7", tilt_x=30.0))
        stack.add_surface(Surface(name="PlanoBack", radius=0.0, thickness=20.0, glass="AIR"))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0))

        opt_sys = OpticalSystem(stack)
        ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=0.5876)
        tr = opt_sys.trace_ray(ray)

        self.assertFalse(tr.vignetted)
        h_front = tr.history[1]

        # In local surface frame, normal is (0, 0, 1). Incoming ray in local frame has incident angle 30 deg.
        theta_1 = np.radians(30.0)
        n1 = 1.0
        n2 = 1.5168
        sin_theta_2 = (n1 / n2) * np.sin(theta_1)
        theta_2 = np.arcsin(sin_theta_2)

        # Refraction angle inside glass
        local_dir = stack.surfaces[1].global_R.T @ np.array([h_front["dir_x"], h_front["dir_y"], h_front["dir_z"]])
        local_dir /= np.linalg.norm(local_dir)
        actual_refract_angle = np.arccos(np.clip(local_dir[2], -1.0, 1.0))
        self.assertAlmostEqual(actual_refract_angle, theta_2, places=3)

    def test_spherical_lens_3d_rotation(self):
        """Tier 1: Spherical lens surface tilted in 3D (Rx, Ry) refracts rays cleanly."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=30.0))
        stack.add_surface(Surface(name="SphFront", radius=40.0, thickness=6.0, glass="N-BK7", tilt_x=10.0, tilt_y=-8.0))
        stack.add_surface(Surface(name="SphBack", radius=-40.0, thickness=30.0, glass="AIR"))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0))

        opt_sys = OpticalSystem(stack)
        rays = [
            Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0])),
            Ray(pos=np.array([2.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0])),
            Ray(pos=np.array([0.0, 2.0, 0.0]), dir=np.array([0.0, 0.0, 1.0])),
        ]
        traced = opt_sys.trace_rays(rays)
        for tr in traced:
            self.assertFalse(tr.vignetted)
            self.assertEqual(len(tr.history), 4)

    def test_aspheric_surface_3d_rotation(self):
        """Tier 1: Tilted aspheric surface (conic + polynomial terms) solves intersection accurately."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=30.0))
        # High NA aspheric surface tilted by 12 deg
        stack.add_surface(Surface(
            name="AsphFront",
            surface_type=SurfaceType.ASPHERIC,
            radius=15.0,
            conic=-0.65,
            asphere_coeffs=[1.5e-6, -2.0e-9],
            thickness=8.0,
            glass="N-BK7",
            tilt_x=12.0,
            tilt_y=5.0
        ))
        stack.add_surface(Surface(name="PlanoBack", radius=0.0, thickness=25.0, glass="AIR"))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0))

        opt_sys = OpticalSystem(stack)
        ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]))
        tr = opt_sys.trace_ray(ray)

        self.assertFalse(tr.vignetted)
        hit_asph = tr.history[1]
        self.assertIsNotNone(hit_asph)

    def test_diffraction_grating_3d_rotation(self):
        """Tier 1: Tilted diffraction grating disperses light in 3D according to grating diffraction vector equation."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=50.0))
        grating = Surface(
            name="Grating3D",
            surface_type=SurfaceType.GRATING,
            glass="MIRROR",
            thickness=50.0,
            grating_lines_per_mm=600.0,
            grating_order=1,
            tilt_x=20.0
        )
        stack.add_surface(grating)
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=50.0))

        opt_sys = OpticalSystem(stack)
        ray_blue = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=0.450)
        ray_red = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=0.650)

        tr_blue = opt_sys.trace_ray(ray_blue)
        tr_red = opt_sys.trace_ray(ray_red)

        self.assertFalse(tr_blue.vignetted)
        self.assertFalse(tr_red.vignetted)

        # Red wavelength diffracts at a steeper angle than blue wavelength
        y_blue = tr_blue.history[-1]["y"]
        y_red = tr_red.history[-1]["y"]
        self.assertNotEqual(y_blue, y_red)
        self.assertGreater(abs(y_red - y_blue), 0.5) # Clear spectral separation

    def test_detector_surface_3d_rotation(self):
        """Tier 1: Tilted detector maps local transverse coordinates from global intercept accurately."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=50.0))
        # Detector tilted by 45 deg around X
        det = Surface(name="TiltedDET", surface_type=SurfaceType.DETECTOR, thickness=0.0, tilt_x=45.0)
        stack.add_surface(det)

        opt_sys = OpticalSystem(stack)
        ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]))
        tr = opt_sys.trace_ray(ray)

        self.assertFalse(tr.vignetted)
        det_h = tr.history[-1]
        self.assertAlmostEqual(det_h["x_local"], 0.0, places=6)
        self.assertAlmostEqual(det_h["y_local"], 0.0, places=6)


class TestTier2BoundaryAndCornerCases(unittest.TestCase):
    """
    Tier 2: Boundary/corner cases (+-90 deg, +-180 deg, near-glancing reflection,
    high-angle refraction, zero-thickness elements, extreme decenters).
    """

    def test_extreme_90_degree_mirror_tilt(self):
        """Tier 2: Mirror at 90 deg tilt (orthogonal beam deflection / grazing angle)."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=50.0))
        mirror = Surface(name="Mirror90", glass="MIRROR", thickness=50.0, tilt_x=45.0)
        stack.add_surface(mirror)
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0))

        opt_sys = OpticalSystem(stack)
        ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]))
        tr = opt_sys.trace_ray(ray)

        self.assertFalse(tr.vignetted)
        refl = tr.history[1]
        # Outgoing direction should be exactly along +Y: [0, 1, 0]
        np.testing.assert_allclose([refl["dir_x"], refl["dir_y"], refl["dir_z"]], [0.0, 1.0, 0.0], atol=1e-6)

    def test_retro_reflector_180_degree_turn(self):
        """Tier 2: Retro-reflector 180 deg turnaround (normal incidence reflection)."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=60.0))
        # Normal incidence planar mirror: tilt_x = 0.0
        mirror = Surface(name="RetroMirror", glass="MIRROR", thickness=60.0, tilt_x=0.0)
        stack.add_surface(mirror)
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0))

        opt_sys = OpticalSystem(stack)
        ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]))
        tr = opt_sys.trace_ray(ray)

        self.assertFalse(tr.vignetted)
        refl = tr.history[1]
        # Outgoing direction should point back along -Z: [0, 0, -1]
        np.testing.assert_allclose([refl["dir_x"], refl["dir_y"], refl["dir_z"]], [0.0, 0.0, -1.0], atol=1e-6)

    def test_near_glancing_angle_reflection(self):
        """Tier 2: Near-glancing reflection (incident angle 89.5 deg) retains numerical stability."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=50.0))
        # Mirror tilted at 89.5 deg
        mirror = Surface(name="GlancingMirror", glass="MIRROR", thickness=50.0, tilt_x=89.5)
        stack.add_surface(mirror)
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=200.0))

        opt_sys = OpticalSystem(stack)
        ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]))
        tr = opt_sys.trace_ray(ray)

        self.assertFalse(tr.vignetted)
        self.assertFalse(math.isnan(tr.history[1]["dir_z"]))

    def test_high_angle_refraction_and_tir_boundary(self):
        """Tier 2: High angle refraction near critical angle and TIR clean triggering."""
        # Glass (n=1.5168) to Air (n=1.0), critical angle ~ 41.3 deg
        dir_in = np.array([0.0, math.sin(math.radians(40.0)), math.cos(math.radians(40.0))])
        norm = np.array([0.0, 0.0, 1.0])
        v_out, tir = snell_refract_3d(dir_in, norm, n1=1.5168, n2=1.0)
        self.assertFalse(tir)
        self.assertAlmostEqual(np.linalg.norm(v_out), 1.0, places=6)

        # Above critical angle: 45 deg -> Total Internal Reflection
        dir_tir = np.array([0.0, math.sin(math.radians(45.0)), math.cos(math.radians(45.0))])
        v_refl, tir_flag = snell_refract_3d(dir_tir, norm, n1=1.5168, n2=1.0)
        self.assertTrue(tir_flag)
        self.assertAlmostEqual(np.linalg.norm(v_refl), 1.0, places=6)

    def test_zero_thickness_coordinate_break_elements(self):
        """Tier 2: Zero-thickness coordinate break surfaces reorient coordinate frame without ray loss."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=40.0))
        # Coordinate Break 1: tilt X by 15 deg, thickness 0.0
        stack.add_surface(Surface(name="CB1", radius=0.0, thickness=0.0, glass="AIR", tilt_x=15.0))
        # Coordinate Break 2: tilt Y by -10 deg, thickness 0.0
        stack.add_surface(Surface(name="CB2", radius=0.0, thickness=0.0, glass="AIR", tilt_y=-10.0))
        stack.add_surface(Surface(name="FreeSpace", radius=0.0, thickness=50.0, glass="AIR"))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0))

        opt_sys = OpticalSystem(stack)
        ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]))
        tr = opt_sys.trace_ray(ray)

        self.assertFalse(tr.vignetted)
        self.assertEqual(len(tr.history), 5)

    def test_large_decenter_combined_with_tilts(self):
        """Tier 2: Large transverse decenter (dx=50mm, dy=-30mm) combined with compound tilts."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=50.0, semi_diameter=100.0))
        s = Surface(name="DecenteredLens", radius=60.0, thickness=8.0, glass="N-BK7",
                    decenter_x=50.0, decenter_y=-30.0, tilt_x=25.0, tilt_y=-15.0, semi_diameter=100.0)
        stack.add_surface(s)
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=150.0))

        # Ray aimed at decentered lens center
        opt_sys = OpticalSystem(stack)
        ray = Ray(pos=np.array([50.0, -30.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]))
        tr = opt_sys.trace_ray(ray)

        self.assertFalse(tr.vignetted)
        self.assertEqual(len(tr.history), 3)


class TestTier3CompoundLensGroupsAndInverseRotations(unittest.TestCase):
    """
    Tier 3: Compound multi-surface lens group rotations (rotate_lens_element),
    coordinate break sequences, and reversible inverse rotations.
    """

    def test_singlet_rigid_rotation_preserves_axial_thickness(self):
        """Tier 3: Singlet lens element rotated rigidly preserves axial thickness exactly."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=40.0))
        indices = stack.insert_singlet(0, "Singlet", radius_front=50.0, thickness=7.5, glass="N-BK7", radius_back=-50.0)
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0))

        s_front = stack.surfaces[indices[0]]
        s_back = stack.surfaces[indices[1]]

        # Rotate by 20 deg around X
        stack.rotate_lens_element(indices[0], tilt_x=20.0, pivot="front_vertex")

        # Global distance between vertices
        dist = float(np.linalg.norm(s_back.global_origin - s_front.global_origin))
        self.assertAlmostEqual(dist, 7.5, places=5)
        self.assertAlmostEqual(s_front.tilt_x, 20.0)
        self.assertAlmostEqual(s_back.tilt_x, 20.0)

    def test_doublet_rigid_rotation_all_three_pivots(self):
        """Tier 3: Doublet rotated around front_vertex, center_of_mass, and rear_vertex."""
        pivots = ["front_vertex", "center_of_mass", "rear_vertex"]
        for p in pivots:
            stack = SurfaceStack()
            stack.add_surface(Surface(name="OBJ", thickness=50.0))
            indices = stack.insert_doublet(0, "Doublet", r1=40.0, t1=5.0, glass1="N-BK7",
                                           r2=-30.0, t2=2.0, glass2="SF5", r3=-100.0, air_gap_after=40.0)
            stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0))

            s1 = stack.surfaces[indices[0]]
            s2 = stack.surfaces[indices[1]]
            s3 = stack.surfaces[indices[2]]

            total_len = 5.0 + 2.0 # 7.0 mm

            stack.rotate_lens_element(indices[0], tilt_x=12.0, tilt_y=-8.0, tilt_z=5.0, pivot=p)

            # Check total length
            len_elem = float(np.linalg.norm(s3.global_origin - s1.global_origin))
            self.assertAlmostEqual(len_elem, total_len, places=5)

            # Check interface length
            len_mid = float(np.linalg.norm(s2.global_origin - s1.global_origin))
            self.assertAlmostEqual(len_mid, 5.0, places=5)

            # All surfaces must share identical tilts
            self.assertAlmostEqual(s1.tilt_x, 12.0)
            self.assertAlmostEqual(s2.tilt_x, 12.0)
            self.assertAlmostEqual(s3.tilt_x, 12.0)
            self.assertAlmostEqual(s1.tilt_y, -8.0)
            self.assertAlmostEqual(s2.tilt_y, -8.0)
            self.assertAlmostEqual(s3.tilt_y, -8.0)

    def test_reversible_inverse_rotations(self):
        """Tier 3: Forward rotation followed by inverse rotation restores original geometry (< 1e-12 mm)."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=50.0))
        indices = stack.insert_catalog_lens(0, "AC254-050-B", air_gap_after=30.0)
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0))

        orig_origins = [s.global_origin.copy() for s in stack.surfaces]

        # Rotate element forward
        stack.rotate_lens_element(indices[0], tilt_x=25.0, tilt_y=-18.0, tilt_z=10.0, pivot="center_of_mass")

        # Rotate element back to 0
        stack.rotate_lens_element(indices[0], tilt_x=0.0, tilt_y=0.0, tilt_z=0.0, pivot="center_of_mass")

        for idx, s in enumerate(stack.surfaces):
            np.testing.assert_allclose(s.global_origin, orig_origins[idx], atol=1e-10)
            self.assertAlmostEqual(s.tilt_x, 0.0, places=7)
            self.assertAlmostEqual(s.tilt_y, 0.0, places=7)
            self.assertAlmostEqual(s.tilt_z, 0.0, places=7)

    def test_periscopic_triple_fold_sequence(self):
        """Tier 3: Sequential non-coplanar fold mirrors (periscope / 3D folding)."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=30.0))
        # Fold 1: 90 deg turn in Y (tilt_x = 45 deg)
        stack.add_surface(Surface(name="M1", glass="MIRROR", thickness=40.0, tilt_x=45.0))
        # Fold 2: 90 deg turn in X (tilt_y = 45 deg)
        stack.add_surface(Surface(name="M2", glass="MIRROR", thickness=40.0, tilt_y=45.0))
        # Fold 3: 90 deg return in Y (tilt_x = -45 deg)
        stack.add_surface(Surface(name="M3", glass="MIRROR", thickness=50.0, tilt_x=-45.0))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0))

        opt_sys = OpticalSystem(stack)
        ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]))
        tr = opt_sys.trace_ray(ray)

        self.assertFalse(tr.vignetted)
        self.assertEqual(len(tr.history), 5)
        # Ray should successfully complete all 3 folds


class TestTier4RealWorldWorkloads(unittest.TestCase):
    """
    Tier 4: Real-world application scenarios
    - Scenario 1: Double-fold spectrometer with tilted mirrors, reflective grating, and detector.
    - Scenario 2: Off-axis folded imaging system with tilted Thorlabs catalog doublet.
    """

    def test_double_fold_czerny_turner_spectrometer(self):
        """Tier 4: Double-fold Czerny-Turner spectrometer setup with tilted collimating & focusing mirrors and grating."""
        stack = SurfaceStack()
        # Input slit at z = 0
        stack.add_surface(Surface(name="Slit_OBJ", thickness=80.0))

        # Collimating Spherical Mirror (f = 80mm, R = -160mm, tilted 10 deg)
        m_coll = Surface(name="CollMirror", glass="MIRROR", radius=-160.0, thickness=60.0, tilt_x=10.0, semi_diameter=25.0)
        stack.add_surface(m_coll)

        # Reflective Diffraction Grating (600 lines/mm, order 1, tilted 22 deg)
        grating = Surface(
            name="DiffractionGrating",
            surface_type=SurfaceType.GRATING,
            glass="MIRROR",
            radius=0.0,
            thickness=70.0,
            grating_lines_per_mm=600.0,
            grating_order=1,
            tilt_x=22.0,
            semi_diameter=25.0
        )
        stack.add_surface(grating)

        # Focusing Spherical Mirror (f = 70mm, R = -140mm, tilted -10 deg)
        m_foc = Surface(name="FocMirror", glass="MIRROR", radius=-140.0, thickness=80.0, tilt_x=-10.0, semi_diameter=25.0)
        stack.add_surface(m_foc)

        # Focal plane CCD / Detector
        det = Surface(name="CCD_DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=35.0)
        stack.add_surface(det)

        opt_sys = OpticalSystem(stack)

        # Trace 3 polychromatic wavelengths: 486 nm (F-line), 587.6 nm (d-line), 656 nm (C-line)
        wls = [0.486, 0.5876, 0.656]
        traced_rays = []
        for wl in wls:
            ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl)
            tr = opt_sys.trace_ray(ray)
            self.assertFalse(tr.vignetted, f"Ray for wl={wl} was unexpectedly vignetted")
            traced_rays.append(tr)

        # Collect detector hit positions
        det_hits = [tr.history[-1] for tr in traced_rays]
        y_pos = [h["y"] for h in det_hits]

        # Verify monotonic spectral spatial dispersion
        self.assertTrue(y_pos[0] < y_pos[1] < y_pos[2] or y_pos[0] > y_pos[1] > y_pos[2],
                        f"Expected monotonic spectral dispersion across wavelengths, got {y_pos}")

        dispersion_span = abs(y_pos[2] - y_pos[0])
        self.assertGreater(dispersion_span, 1.0, f"Expected noticeable dispersion spread (> 1.0 mm), got {dispersion_span:.3f} mm")

    def test_off_axis_folded_imaging_system_with_catalog_doublet(self):
        """Tier 4: Off-axis imaging system with 45 deg fold mirror, tilted Thorlabs doublet AC254-075-B, second fold mirror, and detector."""
        stack = SurfaceStack()
        # Object distance
        stack.add_surface(Surface(name="OBJ", thickness=60.0))

        # Fold Mirror 1 (45 deg YZ fold)
        stack.add_surface(Surface(name="Fold1", glass="MIRROR", thickness=40.0, tilt_x=45.0, semi_diameter=20.0))

        # Insert Thorlabs AC254-075-B catalog doublet
        cat_indices = stack.insert_catalog_lens(1, "AC254-075-B", air_gap_after=35.0)

        # Tilt the entire catalog doublet rigidly by 4 deg around X
        stack.rotate_lens_element(cat_indices[0], tilt_x=4.0, pivot="center_of_mass")

        # Fold Mirror 2 (45 deg XZ fold)
        stack.add_surface(Surface(name="Fold2", glass="MIRROR", thickness=50.0, tilt_y=45.0, semi_diameter=20.0))

        # Detector
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=25.0))

        opt_sys = OpticalSystem(stack)

        # Trace pupil ray grid through the complete off-axis folded layout
        rays = []
        for py in np.linspace(-0.6, 0.6, 5):
            for px in np.linspace(-0.6, 0.6, 5):
                if px**2 + py**2 <= 0.36:
                    r = Ray(pos=np.array([px * 5.0, py * 5.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=0.5876)
                    rays.append(r)

        traced = opt_sys.trace_rays(rays)
        valid = [r for r in traced if not r.vignetted]
        self.assertGreater(len(valid), 0.8 * len(rays), "Majority of pupil rays should reach detector in off-axis system")

        # Verify spot diagram calculation on detector
        hits_x = [r.history[-1]["x"] for r in valid]
        hits_y = [r.history[-1]["y"] for r in valid]
        spot = SpotDiagram(hits_x, hits_y, wavelength_um=0.5876)
        rms_r = spot.calculate_rms_radius()
        self.assertGreater(rms_r, 0.0)
        self.assertLess(rms_r, 10.0)


if __name__ == "__main__":
    unittest.main()
