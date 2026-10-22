"""
Unit tests for Milestone M1 & M2:
1. Rigid body lens element rotation (SurfaceStack.rotate_lens_element) with various pivots and multi-surface groups.
2. Standard 3D Euler rotation matrix composition and coordinate accumulation.
3. Joint optimization variables (tilt_x, tilt_y, decenter_x, decenter_y, thickness, radius).
4. Decoupled optical RMS reporting vs barrier penalties in OptimizationResult.
5. Fixed catalog lens penalty artifact elimination.
6. Robustness against zero rays, unphysical geometries, and abort reset.
"""

import unittest
import numpy as np
import math

from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem, build_rotation_matrix
from optics_app.engine.ray import Ray
from optics_app.engine.optimizer import OpticalOptimizer, OptimizationResult, TerminationStatus


class TestLensElementRigidRotation(unittest.TestCase):
    """Test rigid body rotation of multi-surface lens elements without wedge distortion."""

    def setUp(self):
        self.stack = SurfaceStack()
        self.stack.add_surface(Surface(name="OBJ", thickness=50.0))
        # Insert a doublet lens (3 surfaces)
        self.doublet_indices = self.stack.insert_doublet(
            after_surface_index=0,
            name="Acro",
            r1=50.0,
            t1=6.0,
            glass1="N-BK7",
            r2=-40.0,
            t2=2.5,
            glass2="SF5",
            r3=-120.0,
            semi_diameter=15.0,
            air_gap_after=40.0
        )
        self.stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0))

    def test_singlet_rigid_rotation_front_vertex_pivot(self):
        """Singlet rotated around front vertex preserves element thickness and orientation."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=50.0))
        indices = stack.insert_singlet(
            after_surface_index=0,
            name="Singlet",
            radius_front=50.0,
            thickness=8.0,
            glass="N-BK7",
            radius_back=-50.0,
            air_gap_after=30.0
        )
        s_front = stack.surfaces[indices[0]]
        s_back = stack.surfaces[indices[1]]

        # Rotate by 15 deg around local X
        stack.rotate_lens_element(indices[0], tilt_x=15.0, tilt_y=0.0, tilt_z=0.0, pivot="front_vertex")

        self.assertAlmostEqual(s_front.tilt_x, 15.0)
        self.assertAlmostEqual(s_back.tilt_x, 15.0)
        self.assertAlmostEqual(s_front.decenter_x, 0.0)
        self.assertAlmostEqual(s_front.decenter_y, 0.0)

        # Vector between front and rear global vertices
        v_elem = s_back.global_origin - s_front.global_origin
        elem_len = np.linalg.norm(v_elem)
        self.assertAlmostEqual(elem_len, 8.0, places=5)

        # Orientation vector of element optical axis
        unit_elem = v_elem / elem_len
        expected_dir = np.array([0.0, -np.sin(np.radians(15.0)), np.cos(np.radians(15.0))])
        np.testing.assert_allclose(unit_elem, expected_dir, atol=1e-5)

    def test_doublet_rigid_rotation_center_of_mass_pivot(self):
        """Doublet rotated around center of mass maintains relative interface geometry."""
        s1 = self.stack.surfaces[self.doublet_indices[0]]
        s2 = self.stack.surfaces[self.doublet_indices[1]]
        s3 = self.stack.surfaces[self.doublet_indices[2]]

        total_glass = 6.0 + 2.5 # 8.5 mm

        # Rotate by 10 deg around Y
        self.stack.rotate_lens_element(self.doublet_indices[0], tilt_x=0.0, tilt_y=10.0, tilt_z=0.0, pivot="center_of_mass")

        self.assertAlmostEqual(s1.tilt_y, 10.0)
        self.assertAlmostEqual(s2.tilt_y, 10.0)
        self.assertAlmostEqual(s3.tilt_y, 10.0)

        # Check total element length
        v_tot = s3.global_origin - s1.global_origin
        self.assertAlmostEqual(np.linalg.norm(v_tot), total_glass, places=5)

        # Check cemented interface position along the tilted element axis
        v_mid = s2.global_origin - s1.global_origin
        self.assertAlmostEqual(np.linalg.norm(v_mid), 6.0, places=5)

    def test_doublet_rigid_rotation_rear_vertex_pivot(self):
        """Doublet rotated around rear vertex keeps rear vertex stationary."""
        orig_s3_origin = self.stack.surfaces[self.doublet_indices[2]].global_origin.copy()

        self.stack.rotate_lens_element(self.doublet_indices[0], tilt_x=12.0, tilt_y=-8.0, tilt_z=0.0, pivot="rear_vertex")

        s1 = self.stack.surfaces[self.doublet_indices[0]]
        s3 = self.stack.surfaces[self.doublet_indices[2]]

        # Rear vertex should remain stationary in reference frame
        self.assertAlmostEqual(s3.decenter_x, 0.0, places=5)
        self.assertAlmostEqual(s3.decenter_y, 0.0, places=5)

        # Distance between front and rear vertex is strictly preserved
        v_tot = s3.global_origin - s1.global_origin
        self.assertAlmostEqual(np.linalg.norm(v_tot), 8.5, places=5)

    def test_catalog_lens_rotation(self):
        """Thorlabs catalog doublet (AC254-050-B) rotates rigidly without distortion."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=100.0))
        cat_indices = stack.insert_catalog_lens(0, "AC254-050-B", air_gap_after=50.0)

        stack.rotate_lens_element(cat_indices[0], tilt_x=5.0, tilt_y=5.0, tilt_z=0.0, pivot="front_vertex")

        s_front = stack.surfaces[cat_indices[0]]
        s_back = stack.surfaces[cat_indices[-1]]
        dist = np.linalg.norm(s_back.global_origin - s_front.global_origin)
        # Expected catalog total glass thickness: 6.0 + 2.0 = 8.0 mm
        self.assertAlmostEqual(dist, 8.0, places=5)


class TestEulerRotationMath(unittest.TestCase):
    """Test standard 3D Euler rotation matrix composition R = Rz(theta_z) Ry(theta_y) Rx(theta_x)."""

    def test_rotation_matrix_orthonormality(self):
        """Verify R is strictly orthonormal: R^T @ R = I and det(R) = +1."""
        for tx in [-30.0, 0.0, 45.0]:
            for ty in [-45.0, 0.0, 30.0]:
                for tz in [-90.0, 0.0, 60.0]:
                    R = build_rotation_matrix(tx, ty, tz)
                    np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-12)
                    np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-12)

    def test_euler_composition_order(self):
        """Verify R = Rz @ Ry @ Rx order explicitly."""
        tx, ty, tz = 20.0, 30.0, 40.0
        rx, ry, rz = np.radians(tx), np.radians(ty), np.radians(tz)
        Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
        Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
        Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])

        R_expected = Rz @ Ry @ Rx
        R_actual = build_rotation_matrix(tx, ty, tz)
        np.testing.assert_allclose(R_actual, R_expected, atol=1e-12)


class TestJointOptimizerVariables(unittest.TestCase):
    """Test optimizer support for tilt_x, tilt_y, decenter_x, decenter_y alongside thickness and curvature."""

    def test_joint_tilt_and_thickness_optimization(self):
        """Optimize air gap and folding mirror tilt jointly to minimize spot size."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=100.0))
        stack.add_surface(Surface(name="L1", radius=50.0, thickness=5.0, glass="N-BK7", semi_diameter=15.0))
        stack.add_surface(Surface(name="L2", radius=-50.0, thickness=20.0, glass="AIR", semi_diameter=15.0))
        # Fold mirror
        mirror = Surface(name="FoldMirror", glass="MIRROR", thickness=30.0, semi_diameter=15.0)
        mirror.tilt_x = 40.0 # Slightly misaligned from 45 deg
        stack.add_surface(mirror)
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=25.0))

        sys = OpticalSystem(stack)
        optimizer = OpticalOptimizer(sys, num_rays=25, max_iter=40)

        # Optimize thickness of surface 2 and tilt_x of mirror (surface 3)
        var_dict = {"thickness": [2], "tilt_x": [3]}
        res = optimizer.optimize(variable_indices=var_dict, bounds=[(10.0, 50.0), (35.0, 55.0)])

        self.assertTrue(res.success)
        self.assertIn(2, res.optimized_thicknesses)
        self.assertIn(3, res.optimized_tilts)
        self.assertLess(res.final_rms, res.initial_rms)

    def test_joint_decenter_and_radius_optimization(self):
        """Optimize decenter and radius jointly."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=80.0))
        s1 = Surface(name="LensFront", radius=45.0, thickness=6.0, glass="N-BK7", semi_diameter=15.0, decenter_x=2.0)
        s2 = Surface(name="LensBack", radius=-60.0, thickness=40.0, glass="AIR", semi_diameter=15.0)
        det = Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=20.0)
        stack.add_surface(s1)
        stack.add_surface(s2)
        stack.add_surface(det)

        sys = OpticalSystem(stack)
        optimizer = OpticalOptimizer(sys, num_rays=21, max_iter=30)
        var_dict = {"radius": [1], "decenter_x": [1], "thickness": [2]}
        res = optimizer.optimize(variable_indices=var_dict, bounds=[(20.0, 100.0), (-5.0, 5.0), (10.0, 80.0)])

        self.assertTrue(res.success)
        self.assertIn(1, res.optimized_radii)
        self.assertIn(1, res.optimized_decenters)


class TestOptimizerDecouplingAndFixedLensArtifact(unittest.TestCase):
    """Test pure optical RMS decoupling from barrier penalties and fixed lens artifact elimination."""

    def test_fixed_lens_penalty_artifact_eliminated(self):
        """Fixed steep catalog doublet does NOT inject +57,370 penalty onto air gap optimization."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=100.0))
        # Insert fixed steep catalog doublet
        cat_indices = stack.insert_catalog_lens(0, "AC254-030-B", air_gap_after=25.0)
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=15.0))

        # Ensure all lens surfaces are marked Fixed
        stack.set_all_lenses_fixed()

        sys = OpticalSystem(stack)
        optimizer = OpticalOptimizer(sys, num_rays=30, entrance_pupil_radius=5.0, max_iter=40)

        # Optimize ONLY the air gap after the doublet (cat_indices[-1])
        res = optimizer.optimize(variable_indices=[cat_indices[-1]], bounds=[(5.0, 60.0)])

        self.assertTrue(res.success)
        # Initial and final RMS must be pure optical RMS (not ~57,370 mm!)
        self.assertLess(res.initial_rms, 100.0)
        self.assertLess(res.final_rms, 100.0)
        self.assertGreater(res.rms_reduction_pct, 0.0)
        self.assertNotEqual(res.status, TerminationStatus.RAY_TRACE_FAILURE)

    def test_optimizer_abort_reset(self):
        """Calling opt.stop() before optimize() returns USER_ABORTED and subsequent run converges."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=100.0))
        stack.add_surface(Surface(name="L1", radius=50.0, thickness=5.0, glass="N-BK7", semi_diameter=15.0))
        stack.add_surface(Surface(name="L2", radius=0.0, thickness=30.0, glass="AIR", semi_diameter=15.0))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=20.0))

        sys = OpticalSystem(stack)
        opt = OpticalOptimizer(sys, max_iter=30)
        opt.stop()
        self.assertTrue(opt.abort)

        # First optimize call returns USER_ABORTED and clears abort flag
        res1 = opt.optimize([2], bounds=[(5.0, 100.0)])
        self.assertFalse(res1.success)
        self.assertEqual(res1.status, TerminationStatus.USER_ABORTED)
        self.assertFalse(opt.abort)

        # Subsequent optimize call converges normally
        res2 = opt.optimize([2], bounds=[(5.0, 100.0)])
        self.assertTrue(res2.success)
        self.assertEqual(res2.status, TerminationStatus.CONVERGED)


if __name__ == "__main__":
    unittest.main()
