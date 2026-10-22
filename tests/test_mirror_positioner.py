"""
Unit Tests for Automated Mirror Spatial Positioning, Ray Aiming, and Multi-DOF Optimization.
"""

import unittest
import numpy as np
import math

from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from optics_app.engine.mirror_positioner import MirrorPositioner, MirrorAimResult, MirrorPositionResult
from optics_app.engine.optimizer import OpticalOptimizer


class TestMirrorPositioner(unittest.TestCase):
    """Test automatic tools for mirror spatial positioning and ray aiming."""

    def test_set_mirror_fold_angle_90deg_yz(self):
        """Verify set_mirror_fold_angle sets 45-degree tilt for 90-degree turn in YZ."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="Source", thickness=50.0))
        stack.add_surface(Surface(name="Fold_Mirror", glass="MIRROR", thickness=50.0, semi_diameter=12.7))
        stack.add_surface(Surface(name="Detector", surface_type=SurfaceType.DETECTOR, thickness=0.0))

        tx, ty = MirrorPositioner.set_mirror_fold_angle(stack, mirror_index=1, fold_angle_deg=90.0, fold_plane="YZ")
        self.assertAlmostEqual(tx, 45.0, places=3)
        self.assertAlmostEqual(ty, 0.0, places=3)

        # Confirm ray travels in +Y direction after mirror
        opt_sys = OpticalSystem(stack)
        r = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=0.5876)
        tr = opt_sys.trace_ray(r)
        self.assertTrue(tr is not None)
        h_det = tr.history[2]
        # Detector is at global (0, 50, 50)
        self.assertAlmostEqual(h_det["x_global"], 0.0, places=2)
        self.assertAlmostEqual(h_det["y_global"], 50.0, places=2)
        self.assertAlmostEqual(h_det["z_global"], 50.0, places=2)

    def test_set_mirror_fold_angle_90deg_xz(self):
        """Verify set_mirror_fold_angle sets 45-degree tilt for 90-degree turn in XZ."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="Source", thickness=50.0))
        stack.add_surface(Surface(name="Fold_Mirror", glass="MIRROR", thickness=50.0, semi_diameter=12.7))
        stack.add_surface(Surface(name="Detector", surface_type=SurfaceType.DETECTOR, thickness=0.0))

        tx, ty = MirrorPositioner.set_mirror_fold_angle(stack, mirror_index=1, fold_angle_deg=90.0, fold_plane="XZ")
        self.assertAlmostEqual(tx, 0.0, places=3)
        self.assertAlmostEqual(ty, 45.0, places=3)

    def test_aim_mirror_to_global_point(self):
        """
        Verify analytical ray aiming to an arbitrary 3D target point in space.
        """
        stack = SurfaceStack()
        # Mirror is located at global (0, 0, 50)
        stack.add_surface(Surface(name="Source", thickness=50.0))
        stack.add_surface(Surface(name="Steering_Mirror", glass="MIRROR", thickness=50.0, semi_diameter=25.0))
        stack.add_surface(Surface(name="Target_Plane", surface_type=SurfaceType.DETECTOR, thickness=0.0))
        stack._update_accumulated_positions()

        # Target point is at (0, 50, 50) [requires exactly 90-deg turn in +Y, tilt_x = 45.0]
        target_point = (0.0, 50.0, 50.0)
        res = MirrorPositioner.aim_mirror_to_global_point(
            stack=stack,
            mirror_index=1,
            target_point=target_point
        )

        self.assertTrue(res.success)
        self.assertLess(res.target_error_mm, 0.01)
        self.assertAlmostEqual(res.solved_tilt_x, 45.0, places=2)
        self.assertAlmostEqual(res.solved_tilt_y, 0.0, places=2)

    def test_auto_position_mirror_distance(self):
        """
        Verify automatic axial distance solve to position mirror for a specific beam footprint.
        """
        stack = SurfaceStack()
        stack.add_surface(Surface(name="Source", thickness=20.0))
        stack.add_surface(Surface(name="Mirror", glass="MIRROR", thickness=50.0, semi_diameter=12.7))
        stack.add_surface(Surface(name="Detector", surface_type=SurfaceType.DETECTOR, thickness=0.0))
        stack._update_accumulated_positions()

        opt_dist = MirrorPositioner.auto_position_mirror_distance(
            stack, mirror_index=1, target_beam_diameter_mm=10.0
        )
        self.assertGreater(opt_dist, 0.0)
        self.assertAlmostEqual(stack.surfaces[0].thickness, opt_dist, places=3)

    def test_multi_dof_spatial_optimization(self):
        """
        Verify multi-variable spatial positioning optimization of a fold mirror.
        """
        stack = SurfaceStack()
        stack.add_surface(Surface(name="Source", thickness=45.0))
        stack.add_surface(Surface(
            name="Fold_Mirror",
            glass="MIRROR",
            thickness=55.0,
            tilt_x=45.0,
            semi_diameter=12.7
        ))
        stack.add_surface(Surface(name="Detector", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=10.0))
        stack._update_accumulated_positions()

        res = MirrorPositioner.optimize_mirror_spatial_position(
            stack=stack,
            mirror_index=1,
            target_surface_index=2,
            target_xy=(0.0, 0.0),
            optimize_tilt=True,
            optimize_distance=True,
            optimize_decenter=False,
            max_iter=30
        )

        self.assertTrue(res.success)
        self.assertIn("tilt_x", res.optimized_params)
        self.assertIn("pre_thickness", res.optimized_params)

    def test_optical_optimizer_mirror_methods(self):
        """
        Verify that OpticalOptimizer exposes auto_aim_mirror and optimize_mirror_position.
        """
        stack = SurfaceStack()
        stack.add_surface(Surface(name="Source", thickness=50.0))
        stack.add_surface(Surface(name="Mirror", glass="MIRROR", thickness=50.0, semi_diameter=12.7))
        stack.add_surface(Surface(name="Detector", surface_type=SurfaceType.DETECTOR, thickness=0.0))
        stack._update_accumulated_positions()

        opt_sys = OpticalSystem(stack)
        optimizer = OpticalOptimizer(opt_sys)

        aim_res = optimizer.auto_aim_mirror(mirror_index=1, target_surface_index=2, target_xy=(0.0, 0.0))
        self.assertTrue(aim_res.success)


if __name__ == "__main__":
    unittest.main()
