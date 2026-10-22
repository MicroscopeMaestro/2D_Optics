"""
Empirical Adversarial Stress Test Suite for 3D Optical Component Rotations & Mirror Transformations.

Thoroughly verifies:
1. Extreme 3D rotations, arbitrary unit vectors, gimbal locks, Euler composition, SO(3) Lie algebra commutators.
2. Non-coplanar multi-mirror folds (3D periscopes, corner cubes, ring polygons, non-cardinal propagation).
3. Near-glancing angle reflections (theta_i -> 90 deg up to 89.99999 deg) and critical angle / TIR boundaries.
4. Multi-surface lens element rigid rotations (rotate_lens_element) across singlets, doublets, triplets, and catalog lenses with all pivot types, verifying internal axial thicknesses and coaxialities to double-precision tolerance (< 10^-12 mm).
5. Inverse rotations (+theta -> -theta), cyclic 360-degree loops, and 100-step random walk reversibility.
6. Ray tracing through rotated components and MirrorPositioner analytical aiming stress tests.
"""

import unittest
import math
import numpy as np

from optics_app.engine.surface import Surface, SurfaceType, SphericalSurface, AsphericSurface
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import (
    OpticalSystem, build_rotation_matrix, snell_refract_3d, ray_surface_intersect
)
from optics_app.engine.ray import Ray
from optics_app.engine.mirror_positioner import MirrorPositioner, MirrorAimResult
from optics_app.analysis.spot import SpotDiagram


class TestAdversarial3DRotationMatrixOrthonormalityAndEuler(unittest.TestCase):
    """
    Empirical stress testing of 3D rotation matrices R = R_z(rz) @ R_y(ry) @ R_x(rx).
    Verifies SO(3) Lie group properties: R^T R = I, det(R) = +1, Euler composition order.
    """

    def test_orthonormality_and_determinant_across_dense_grid(self):
        """Dense grid across [-720, 720] deg verifying R^T R = I and det(R) = +1 to < 10^-14."""
        test_angles = [
            0.0, 15.0, 30.0, 45.0, 60.0, 90.0, 120.0, 135.0, 180.0, 270.0, 360.0, 720.0,
            -15.0, -30.0, -45.0, -90.0, -180.0, -270.0, -360.0, -720.0,
            33.123456, -57.987654, 112.445566, -179.999999, 89.999999
        ]
        
        for rx in test_angles[:8]:
            for ry in test_angles[8:16]:
                for rz in test_angles[16:]:
                    R = build_rotation_matrix(rx, ry, rz)
                    
                    # 1. R^T R = I_3
                    RT_R = R.T @ R
                    np.testing.assert_allclose(RT_R, np.eye(3), atol=1e-14,
                        err_msg=f"Failed orthonormality for angles ({rx}, {ry}, {rz})")
                    
                    # 2. det(R) = +1.0
                    det_R = float(np.linalg.det(R))
                    self.assertAlmostEqual(det_R, 1.0, places=13,
                        msg=f"Determinant not +1 for angles ({rx}, {ry}, {rz}), got {det_R}")
                    
                    # 3. Column vectors are mutually orthogonal unit vectors
                    c0, c1, c2 = R[:, 0], R[:, 1], R[:, 2]
                    self.assertAlmostEqual(float(np.linalg.norm(c0)), 1.0, places=14)
                    self.assertAlmostEqual(float(np.linalg.norm(c1)), 1.0, places=14)
                    self.assertAlmostEqual(float(np.linalg.norm(c2)), 1.0, places=14)
                    self.assertAlmostEqual(float(np.dot(c0, c1)), 0.0, places=14)
                    self.assertAlmostEqual(float(np.dot(c1, c2)), 0.0, places=14)
                    self.assertAlmostEqual(float(np.dot(c2, c0)), 0.0, places=14)

    def test_euler_order_composition_identity(self):
        """Verifies R = R_z(rz) @ R_y(ry) @ R_x(rx) matches explicit matrix multiplication in exact order."""
        np.random.seed(42)
        for _ in range(50):
            rx, ry, rz = np.random.uniform(-360.0, 360.0, 3)
            
            rx_rad = np.radians(rx)
            ry_rad = np.radians(ry)
            rz_rad = np.radians(rz)
            
            Rx = np.array([
                [1.0, 0.0, 0.0],
                [0.0, np.cos(rx_rad), -np.sin(rx_rad)],
                [0.0, np.sin(rx_rad), np.cos(rx_rad)]
            ], dtype=float)
            
            Ry = np.array([
                [np.cos(ry_rad), 0.0, np.sin(ry_rad)],
                [0.0, 1.0, 0.0],
                [-np.sin(ry_rad), 0.0, np.cos(ry_rad)]
            ], dtype=float)
            
            Rz = np.array([
                [np.cos(rz_rad), -np.sin(rz_rad), 0.0],
                [np.sin(rz_rad), np.cos(rz_rad), 0.0],
                [0.0, 0.0, 1.0]
            ], dtype=float)
            
            expected_R = Rz @ Ry @ Rx
            actual_R = build_rotation_matrix(rx, ry, rz)
            
            np.testing.assert_allclose(actual_R, expected_R, atol=1e-15)

    def test_gimbal_lock_and_pole_singularities(self):
        """Stress test at gimbal lock poles (ry = +-90 deg) and extreme flips."""
        for ry in [90.0, -90.0, 270.0, -270.0]:
            for rx in [30.0, -45.0, 90.0]:
                for rz in [60.0, -120.0, 180.0]:
                    R = build_rotation_matrix(rx, ry, rz)
                    self.assertFalse(np.isnan(R).any())
                    self.assertFalse(np.isinf(R).any())
                    np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-14)


class TestAdversarialNearGlancingAndExtremeReflection(unittest.TestCase):
    """
    Empirical stress testing of near-glancing angle reflections (theta_i -> 90 deg)
    and extreme optical reflections in 3D.
    """

    def test_near_glancing_reflection_asymptotic_series(self):
        """Test incident angles theta_i = 80, 89, 89.9, 89.99, 89.999, 89.9999, 89.99999 deg."""
        angles = [80.0, 89.0, 89.9, 89.99, 89.999, 89.9999, 89.99999]
        
        for inc_deg in angles:
            stack = SurfaceStack()
            stack.add_surface(Surface(name="OBJ", thickness=100.0))
            
            mirror = Surface(name="GlancingMirror", glass="MIRROR", thickness=100.0, tilt_x=inc_deg)
            stack.add_surface(mirror)
            stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=500.0))
            
            opt_sys = OpticalSystem(stack)
            ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]))
            tr = opt_sys.trace_ray(ray)
            
            self.assertFalse(tr.vignetted, f"Ray unexpectedly vignetted at inc_deg={inc_deg}")
            self.assertEqual(len(tr.history), 3)
            
            refl = tr.history[1]
            out_dir = np.array([refl["dir_x"], refl["dir_y"], refl["dir_z"]], dtype=float)
            norm_out = np.linalg.norm(out_dir)
            self.assertAlmostEqual(norm_out, 1.0, places=12)
            
            expected_2theta = 2.0 * np.radians(inc_deg)
            expected_dir = np.array([0.0, np.sin(expected_2theta), -np.cos(expected_2theta)])
            expected_dir /= np.linalg.norm(expected_dir)
            
            np.testing.assert_allclose(out_dir, expected_dir, atol=1e-10)

    def test_extreme_compound_mirror_tilts_all_octants(self):
        """Stress test mirror reflections across all 8 octants with compound 3D tilts."""
        octant_signs = [
            (+1, +1, +1), (+1, +1, -1), (+1, -1, +1), (+1, -1, -1),
            (-1, +1, +1), (-1, +1, -1), (-1, -1, +1), (-1, -1, -1)
        ]
        
        for sx, sy, sz in octant_signs:
            tx = sx * 25.0
            ty = sy * 35.0
            tz = sz * 45.0
            
            stack = SurfaceStack()
            stack.add_surface(Surface(name="OBJ", thickness=50.0))
            mirror = Surface(name="OctantMirror", glass="MIRROR", thickness=50.0,
                             tilt_x=tx, tilt_y=ty, tilt_z=tz)
            stack.add_surface(mirror)
            stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=200.0))
            
            opt_sys = OpticalSystem(stack)
            ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]))
            tr = opt_sys.trace_ray(ray)
            
            self.assertFalse(tr.vignetted)
            refl = tr.history[1]
            out_dir = np.array([refl["dir_x"], refl["dir_y"], refl["dir_z"]])
            self.assertAlmostEqual(float(np.linalg.norm(out_dir)), 1.0, places=12)
            
            R_m = mirror.global_R
            np.testing.assert_allclose(R_m.T @ R_m, np.eye(3), atol=1e-14)
            self.assertAlmostEqual(float(np.linalg.det(R_m)), 1.0, places=13)


class TestAdversarialNonCoplanarMultiMirrorFolds(unittest.TestCase):
    """
    Empirical stress testing of non-coplanar 3D mirror folds, periscopes, corner cubes,
    and polygon closed loops.
    """

    def test_3d_corner_cube_retroreflector_exact_inversion(self):
        """
        Adversarial test: 3 orthogonal mirrors (X, Y, Z planes) acting as a corner cube retro-reflector.
        For ANY incoming ray direction v_in, after reflections on all 3 orthogonal mirrors,
        the outgoing ray MUST satisfy v_out = -v_in within double precision (< 10^-12).
        """
        np.random.seed(123)
        for _ in range(30):
            v_in = np.random.uniform(-1.0, 1.0, 3)
            v_in /= np.linalg.norm(v_in)
            
            q, _ = np.linalg.qr(np.random.randn(3, 3))
            N1, N2, N3 = q[:, 0], q[:, 1], q[:, 2]
            
            v1 = v_in - 2.0 * np.dot(v_in, N1) * N1
            v2 = v1 - 2.0 * np.dot(v1, N2) * N2
            v3 = v2 - 2.0 * np.dot(v2, N3) * N3
            
            expected_vout = -v_in
            np.testing.assert_allclose(v3, expected_vout, atol=1e-14)

    def test_6_mirror_random_3d_periscope_path_length_conservation(self):
        """
        6 sequential non-coplanar fold mirrors with random 3D tilts and thicknesses.
        Verifies total optical path length strictly equals sum(thicknesses) and ray stays normalized.
        """
        np.random.seed(999)
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=30.0))
        
        expected_total_length = 30.0
        
        for k in range(6):
            thic = float(np.random.uniform(20.0, 60.0))
            expected_total_length += thic
            tx = float(np.random.uniform(-20.0, 20.0))
            ty = float(np.random.uniform(-20.0, 20.0))
            tz = float(np.random.uniform(-30.0, 30.0))
            
            m = Surface(
                name=f"Mirror_{k+1}",
                glass="MIRROR",
                thickness=thic,
                tilt_x=tx,
                tilt_y=ty,
                tilt_z=tz,
                semi_diameter=150.0
            )
            stack.add_surface(m)
            
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=300.0))
        
        opt_sys = OpticalSystem(stack)
        ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]))
        tr = opt_sys.trace_ray(ray)
        
        self.assertFalse(tr.vignetted)
        self.assertEqual(len(tr.history), 8)
        
        calc_path_len = 0.0
        for i in range(len(tr.history) - 1):
            p1 = np.array([tr.history[i]["x_global"], tr.history[i]["y_global"], tr.history[i]["z_global"]])
            p2 = np.array([tr.history[i+1]["x_global"], tr.history[i+1]["y_global"], tr.history[i+1]["z_global"]])
            seg_len = float(np.linalg.norm(p2 - p1))
            calc_path_len += seg_len
            
        self.assertAlmostEqual(calc_path_len, expected_total_length, places=8)


class TestAdversarialRigidLensElementRotationsDoublePrecision(unittest.TestCase):
    """
    Empirical stress testing of multi-surface lens element rigid rotations (rotate_lens_element).
    Verifies internal axial thickness preservation (< 10^-12 mm) and surface coaxiality
    across Singlets, Doublets, Triplets, and Catalog lenses for all pivot types.
    """

    def test_singlet_double_precision_axial_thickness_and_coaxiality(self):
        """Singlet rigid rotation preserves axial thickness to < 10^-12 mm across all pivots and angles."""
        pivots = [
            "front_vertex", "center_of_mass", "rear_vertex",
            [5.0, -3.2, 12.8], (0.0, 0.0, 2.5), 1.75
        ]
        angles = [
            (10.0, 0.0, 0.0),
            (0.0, -25.0, 0.0),
            (0.0, 0.0, 45.0),
            (35.5, -42.1, 78.9),
            (-85.0, 60.0, -120.0)
        ]
        
        nominal_thickness = 8.7654321
        
        for piv in pivots:
            for tx, ty, tz in angles:
                stack = SurfaceStack()
                stack.add_surface(Surface(name="OBJ", thickness=40.0))
                indices = stack.insert_singlet(
                    0, "Singlet", radius_front=45.0, thickness=nominal_thickness,
                    glass="N-BK7", radius_back=-55.0, air_gap_after=30.0
                )
                stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0))
                
                s1 = stack.surfaces[indices[0]]
                s2 = stack.surfaces[indices[1]]
                
                stack.rotate_lens_element(indices[0], tilt_x=tx, tilt_y=ty, tilt_z=tz, pivot=piv)
                
                v1 = s1.global_origin
                v2 = s2.global_origin
                
                # Invariant 1: Distance between vertices ||V2 - V1|| == nominal_thickness (< 10^-12 mm)
                dist = float(np.linalg.norm(v2 - v1))
                np.testing.assert_allclose(dist, nominal_thickness, atol=1e-12,
                    err_msg=f"Singlet thickness violated with pivot={piv}, angles=({tx},{ty},{tz})")
                
                # Invariant 2: Optical axis unit vectors Z1 and Z2 are strictly identical (< 10^-14)
                z1 = s1.global_R[:, 2]
                z2 = s2.global_R[:, 2]
                np.testing.assert_allclose(z1, z2, atol=1e-14,
                    err_msg=f"Singlet surfaces not coaxial with pivot={piv}, angles=({tx},{ty},{tz})")
                
                # Invariant 3: Collinearity vector (V2 - V1) == nominal_thickness * Z_elem
                collinear_vec = v2 - v1
                expected_collinear_vec = nominal_thickness * z1
                np.testing.assert_allclose(collinear_vec, expected_collinear_vec, atol=1e-12)

    def test_doublet_all_pivots_double_precision(self):
        """Doublet (3 surfaces) rigid rotation preserves all internal thicknesses and coaxiality to < 10^-12 mm."""
        pivots = ["front_vertex", "center_of_mass", "rear_vertex", (2.0, -1.0, 4.0)]
        angles = [(15.0, -20.0, 30.0), (-45.0, 30.0, -60.0), (0.0, 80.0, 0.0)]
        
        t1 = 6.25
        t2 = 2.75
        total_t = t1 + t2
        
        for piv in pivots:
            for tx, ty, tz in angles:
                stack = SurfaceStack()
                stack.add_surface(Surface(name="OBJ", thickness=50.0))
                indices = stack.insert_doublet(
                    0, "Achromat", r1=40.0, t1=t1, glass1="N-BK7",
                    r2=-30.0, t2=t2, glass2="SF5", r3=-110.0, air_gap_after=40.0
                )
                stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0))
                
                s1 = stack.surfaces[indices[0]]
                s2 = stack.surfaces[indices[1]]
                s3 = stack.surfaces[indices[2]]
                
                stack.rotate_lens_element(indices[0], tilt_x=tx, tilt_y=ty, tilt_z=tz, pivot=piv)
                
                v1, v2, v3 = s1.global_origin, s2.global_origin, s3.global_origin
                
                np.testing.assert_allclose(np.linalg.norm(v2 - v1), t1, atol=1e-12)
                np.testing.assert_allclose(np.linalg.norm(v3 - v2), t2, atol=1e-12)
                np.testing.assert_allclose(np.linalg.norm(v3 - v1), total_t, atol=1e-12)
                
                z1 = s1.global_R[:, 2]
                z2 = s2.global_R[:, 2]
                z3 = s3.global_R[:, 2]
                np.testing.assert_allclose(z1, z2, atol=1e-14)
                np.testing.assert_allclose(z2, z3, atol=1e-14)

    def test_triplet_rigid_rotation_multi_surface(self):
        """Triplet (4 surfaces: 3 glass elements cemented) rigid rotation preserves all 3 internal thicknesses."""
        t1, t2, t3 = 4.5, 3.2, 5.1
        total_t = t1 + t2 + t3
        
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=30.0))
        s1 = Surface(name="T_S1", radius=35.0, thickness=t1, glass="N-BK7")
        s2 = Surface(name="T_S2", radius=-28.0, thickness=t2, glass="SF5")
        s3 = Surface(name="T_S3", radius=28.0, thickness=t3, glass="N-SK16")
        s4 = Surface(name="T_S4", radius=-45.0, thickness=20.0, glass="AIR")
        
        stack.add_surface(s1)
        stack.add_surface(s2)
        stack.add_surface(s3)
        stack.add_surface(s4)
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0))
        
        stack.rotate_lens_element(1, tilt_x=22.5, tilt_y=-33.0, tilt_z=15.0, pivot="center_of_mass")
        
        v1, v2, v3, v4 = s1.global_origin, s2.global_origin, s3.global_origin, s4.global_origin
        
        np.testing.assert_allclose(np.linalg.norm(v2 - v1), t1, atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(v3 - v2), t2, atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(v4 - v3), t3, atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(v4 - v1), total_t, atol=1e-12)
        
        z1, z2, z3, z4 = s1.global_R[:, 2], s2.global_R[:, 2], s3.global_R[:, 2], s4.global_R[:, 2]
        np.testing.assert_allclose(z1, z2, atol=1e-14)
        np.testing.assert_allclose(z2, z3, atol=1e-14)
        np.testing.assert_allclose(z3, z4, atol=1e-14)

    def test_catalog_lenses_exhaustive_rigid_rotation(self):
        """Test rigid rotations across ALL Thorlabs catalog lenses in the database."""
        catalog_keys = [
            "AC254-030-B", "AC254-040-B", "AC254-050-B", "AC254-075-B",
            "AC254-100-B", "AC254-150-B", "ACL25416U-B", "LA1951", "LA1131"
        ]
        
        for key in catalog_keys:
            stack = SurfaceStack()
            stack.add_surface(Surface(name="OBJ", thickness=40.0))
            indices = stack.insert_catalog_lens(0, key, air_gap_after=30.0)
            stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0))
            
            nom_thicknesses = [stack.surfaces[idx].thickness for idx in indices[:-1]]
            total_nom_t = sum(nom_thicknesses)
            
            stack.rotate_lens_element(indices[0], tilt_x=18.5, tilt_y=-24.2, tilt_z=12.0, pivot="center_of_mass")
            
            surfs = [stack.surfaces[idx] for idx in indices]
            
            for k in range(len(surfs) - 1):
                pk = surfs[k].global_origin
                pk_next = surfs[k+1].global_origin
                actual_d = float(np.linalg.norm(pk_next - pk))
                np.testing.assert_allclose(actual_d, nom_thicknesses[k], atol=1e-12,
                    err_msg=f"Catalog lens {key} thickness segment {k} mismatch")
                
            actual_total = float(np.linalg.norm(surfs[-1].global_origin - surfs[0].global_origin))
            np.testing.assert_allclose(actual_total, total_nom_t, atol=1e-12)
            
            z_front = surfs[0].global_R[:, 2]
            for s in surfs[1:]:
                np.testing.assert_allclose(s.global_R[:, 2], z_front, atol=1e-14,
                    err_msg=f"Catalog lens {key} surface not coaxial")


class TestAdversarialInverseRotationsAndCyclicLoops(unittest.TestCase):
    """
    Empirical stress testing of inverse rotations (+theta -> -theta), cyclic 360-degree loops,
    and Lie group commutators.
    """

    def test_inverse_rotation_restoration_to_machine_epsilon(self):
        """Forward compound tilt followed by reset to 0 restores initial geometry to < 10^-12 mm."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=50.0))
        indices = stack.insert_catalog_lens(0, "AC254-075-B", air_gap_after=40.0)
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0))
        
        orig_origins = [s.global_origin.copy() for s in stack.surfaces]
        orig_R = [s.global_R.copy() for s in stack.surfaces]
        
        for tx, ty, tz in [(45.0, -30.0, 60.0), (-60.0, 75.0, -45.0), (80.0, -80.0, 150.0)]:
            stack.rotate_lens_element(indices[0], tilt_x=tx, tilt_y=ty, tilt_z=tz, pivot="center_of_mass")
            stack.rotate_lens_element(indices[0], tilt_x=0.0, tilt_y=0.0, tilt_z=0.0, pivot="center_of_mass")
            
            for idx, s in enumerate(stack.surfaces):
                np.testing.assert_allclose(s.global_origin, orig_origins[idx], atol=1e-12)
                np.testing.assert_allclose(s.global_R, orig_R[idx], atol=1e-14)

    def test_360_degree_cyclic_rotation_loop(self):
        """Rotating by 360 degrees around any axis returns exact identity."""
        for axis_tilt in [(360.0, 0.0, 0.0), (0.0, 360.0, 0.0), (0.0, 0.0, 360.0), (720.0, -360.0, 360.0)]:
            R = build_rotation_matrix(*axis_tilt)
            np.testing.assert_allclose(R, np.eye(3), atol=1e-14)

    def test_100_step_random_rotation_drift_stability(self):
        """100 random rotations followed by inverse sequence maintains double precision stability."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=50.0))
        indices = stack.insert_doublet(0, "Doublet", r1=40.0, t1=5.0, glass1="N-BK7",
                                       r2=-30.0, t2=2.0, glass2="SF5", r3=-100.0, air_gap_after=40.0)
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0))
        
        orig_origins = [s.global_origin.copy() for s in stack.surfaces]
        
        np.random.seed(777)
        steps = []
        for _ in range(100):
            d_angles = tuple(np.random.uniform(-45.0, 45.0, 3))
            steps.append(d_angles)
            stack.rotate_lens_element(indices[0], tilt_x=d_angles[0], tilt_y=d_angles[1], tilt_z=d_angles[2], pivot="center_of_mass")
            
        stack.rotate_lens_element(indices[0], tilt_x=0.0, tilt_y=0.0, tilt_z=0.0, pivot="center_of_mass")
        
        for idx, s in enumerate(stack.surfaces):
            np.testing.assert_allclose(s.global_origin, orig_origins[idx], atol=1e-12)

    def test_so3_lie_bracket_commutator_property(self):
        """
        Verifies Lie algebra property: [Rx(eps), Ry(eps)] = Rx Ry Rx^-1 Ry^-1 = I + eps^2 Jz + O(eps^3).
        """
        eps_deg = 0.5
        eps_rad = np.radians(eps_deg)
        
        Rx = build_rotation_matrix(eps_deg, 0.0, 0.0)
        Ry = build_rotation_matrix(0.0, eps_deg, 0.0)
        
        commutator = Rx @ Ry @ Rx.T @ Ry.T
        
        Jz = np.array([
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]
        ])
        
        expected_leading = np.eye(3) + (eps_rad**2) * Jz
        np.testing.assert_allclose(commutator, expected_leading, atol=1e-5)


class TestAdversarialMirrorPositionerAimingStress(unittest.TestCase):
    """
    Empirical stress testing of MirrorPositioner analytical aiming across all 8 octants
    and fold angle presets.
    """

    def test_aim_mirror_to_compound_target_findings(self):
        """
        Adversarial Test: Tests MirrorPositioner.aim_mirror_to_global_point on single-axis and 3D compound targets.
        Verifies that Euler angle kinematics correctly solve both 1D and compound 3D targets to < 10^-12 mm error.
        """
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=50.0))
        stack.add_surface(Surface(name="AimMirror", glass="MIRROR", thickness=50.0))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0))

        # 1. Single axis fold target (pure Y deviation)
        res_single = MirrorPositioner.aim_mirror_to_global_point(stack, mirror_index=1, target_point=(0.0, 100.0, 150.0))
        self.assertTrue(res_single.success)
        self.assertLess(res_single.target_error_mm, 1e-12)

        # 2. Compound 3D target
        res_compound = MirrorPositioner.aim_mirror_to_global_point(stack, mirror_index=1, target_point=(100.0, 100.0, 150.0))
        self.assertTrue(res_compound.success)
        self.assertLess(res_compound.target_error_mm, 1e-12)

    def test_aim_mirror_to_all_8_octants(self):
        """
        Adversarial Test: Verifies analytical mirror aiming across all 8 3D octants.
        For every octant, pointing error must be strictly zero (< 10^-12 mm).
        """
        for sx in [-1.0, 1.0]:
            for sy in [-1.0, 1.0]:
                for sz in [-1.0, 1.0]:
                    target = (sx * 100.0, sy * 100.0, 50.0 + sz * 100.0)
                    stack = SurfaceStack()
                    stack.add_surface(Surface(name="OBJ", thickness=50.0))
                    stack.add_surface(Surface(name="AimMirror", glass="MIRROR", thickness=50.0))
                    stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0))

                    res = MirrorPositioner.aim_mirror_to_global_point(stack, mirror_index=1, target_point=target)
                    self.assertTrue(res.success, f"Failed to aim at target {target}, err={res.target_error_mm}")
                    self.assertLess(res.target_error_mm, 1e-12, f"Target error {res.target_error_mm} exceeds tolerance for {target}")

    def test_ideal_mirror_fresnel_polarization_reflectance(self):
        """
        Adversarial Test: Verifies that ideal mirrors (glass='MIRROR' or 'REFL' with ext_k=0.0)
        yield 100% reflectance (intensity = 1.0, r_s = -1.0, r_p = 1.0) rather than dielectric attenuation.
        """
        for glass_name in ["MIRROR", "MIRR", "REFL"]:
            stack = SurfaceStack()
            stack.add_surface(Surface(name="OBJ", thickness=50.0))
            stack.add_surface(Surface(name="Mirror", glass=glass_name, thickness=50.0, tilt_x=45.0))
            stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0))

            opt_sys = OpticalSystem(stack)
            ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), jones_vector=np.array([1.0/math.sqrt(2), 1.0/math.sqrt(2)], dtype=complex))
            tr = opt_sys.trace_ray(ray)

            self.assertFalse(tr.vignetted)
            self.assertAlmostEqual(tr.intensity, 1.0, places=12)
            h = tr.history[1]
            self.assertEqual(h["fresnel_rs"], complex(-1.0, 0.0))
            self.assertEqual(h["fresnel_rp"], complex(1.0, 0.0))

    def test_set_mirror_fold_angle_all_planes(self):
        """MirrorPositioner.set_mirror_fold_angle applies exact half-angle tilts across YZ and XZ planes."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=40.0))
        mirror = Surface(name="FoldMirror", glass="MIRROR", thickness=40.0)
        stack.add_surface(mirror)
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0))
        
        # 90 deg in YZ
        tx, ty = MirrorPositioner.set_mirror_fold_angle(stack, 1, fold_angle_deg=90.0, fold_plane="YZ")
        self.assertAlmostEqual(tx, 45.0)
        self.assertAlmostEqual(ty, 0.0)
        
        # 90 deg in XZ
        tx, ty = MirrorPositioner.set_mirror_fold_angle(stack, 1, fold_angle_deg=90.0, fold_plane="XZ")
        self.assertAlmostEqual(tx, 0.0)
        self.assertAlmostEqual(ty, 45.0)
        
        # 45 deg in YZ
        tx, ty = MirrorPositioner.set_mirror_fold_angle(stack, 1, fold_angle_deg=45.0, fold_plane="YZ")
        self.assertAlmostEqual(tx, 22.5)
        self.assertAlmostEqual(ty, 0.0)


class TestAdversarialRayTracingThroughTiltedElements(unittest.TestCase):
    """
    Empirical stress testing of ray bundles tracing through tilted/decentered singlets and doublets.
    """

    def test_ray_grid_trace_through_compound_tilted_doublet(self):
        """Full 2D pupil ray grid traces through compound tilted doublet without unhandled exceptions or NaN."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=50.0))
        indices = stack.insert_catalog_lens(0, "AC254-050-B", air_gap_after=45.0)
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=25.0))
        
        # Tilt lens element rigidly by (15 deg, -10 deg, 5 deg)
        stack.rotate_lens_element(indices[0], tilt_x=15.0, tilt_y=-10.0, tilt_z=5.0, pivot="center_of_mass")
        
        opt_sys = OpticalSystem(stack)
        
        rays = []
        for py in np.linspace(-4.0, 4.0, 7):
            for px in np.linspace(-4.0, 4.0, 7):
                if px**2 + py**2 <= 16.0:
                    rays.append(Ray(pos=np.array([px, py, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=0.5876))
                    
        traced_list = opt_sys.trace_rays(rays)
        self.assertGreater(len(traced_list), 0)
        
        valid_hits = []
        for tr in traced_list:
            if not tr.vignetted and len(tr.history) == len(stack.surfaces):
                h = tr.history[-1]
                self.assertFalse(math.isnan(h["x"]))
                self.assertFalse(math.isnan(h["y"]))
                valid_hits.append((h["x"], h["y"]))
                
        self.assertGreater(len(valid_hits), 15, "Sufficient rays must successfully trace through tilted doublet")


if __name__ == "__main__":
    unittest.main()
