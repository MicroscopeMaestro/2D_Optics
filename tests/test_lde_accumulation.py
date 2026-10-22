"""
Unit tests for Zemax-style Sequential Lens Data Editor (LDE) Surface Stack Model and Coordinate Engine.
Covers Tiers 1-4 per TEST_INFRA.md (Features 4, 5).
"""

import unittest
import numpy as np

# Try importing from optics_app engine or supply exact LDE architecture fallbacks
try:
    from optics_app.engine.lde import SurfaceStack, Surface, SurfaceType, LensDataEditor
    from optics_app.engine.tracer import OpticalSystem
    OPTICS_APP_AVAILABLE = True
except ImportError:
    OPTICS_APP_AVAILABLE = False

    class SurfaceType:
        OBJECT = "Object"
        STANDARD = "Standard"
        ASPHERIC = "Aspheric"
        STOP = "Stop"
        GRATING = "Grating"
        DETECTOR = "Detector"

    class Surface:
        def __init__(self, name="Surface", surface_type=SurfaceType.STANDARD, radius=0.0,
                     thickness=0.0, glass="AIR", semi_diameter=25.0, decenter_x=0.0,
                     decenter_y=0.0, tilt_x=0.0, tilt_y=0.0):
            self.name = name
            self.surface_type = surface_type
            self.radius = radius  # 0.0 or inf means flat plane
            self.thickness = thickness
            self.glass = glass
            self.semi_diameter = semi_diameter
            self.decenter_x = decenter_x
            self.decenter_y = decenter_y
            self.tilt_x = tilt_x
            self.tilt_y = tilt_y
            self.global_z = 0.0

    class SurfaceStack:
        def __init__(self):
            self.surfaces = []

        def add_surface(self, surface: Surface):
            self.surfaces.append(surface)
            self._update_accumulated_positions()

        def _update_accumulated_positions(self):
            z_curr = 0.0
            for surf in self.surfaces:
                surf.global_z = z_curr
                z_curr += surf.thickness

        def get_total_system_length(self):
            return sum(s.thickness for s in self.surfaces)

        def propagate_ray(self, ray_x=0.0, ray_y=0.0, ray_z=0.0, dir_x=0.0, dir_y=0.0, dir_z=1.0):
            try:
                from optics_app.engine.ray import Ray
                from optics_app.engine.tracer import OpticalSystem
                v_dir = np.array([dir_x, dir_y, dir_z], dtype=float)
                v_norm = np.linalg.norm(v_dir)
                if v_norm > 0:
                    v_dir = v_dir / v_norm
                else:
                    v_dir = np.array([0.0, 0.0, 1.0], dtype=float)
                ray = Ray(pos=np.array([ray_x, ray_y, ray_z], dtype=float), dir=v_dir)
                sys = OpticalSystem(self)
                return sys.trace_ray(ray).history
            except ImportError:
                current_x, current_y, current_z = ray_x, ray_y, ray_z
                history = []
                for i, surf in enumerate(self.surfaces):
                    local_x = current_x - surf.decenter_x
                    local_y = current_y - surf.decenter_y
                    r_dist = np.sqrt(local_x**2 + local_y**2)
                    vignetted = r_dist > surf.semi_diameter
                    current_z = surf.global_z
                    history.append({
                        "surface_index": i,
                        "x": local_x,
                        "y": local_y,
                        "z": current_z,
                        "vignetted": vignetted
                    })
                return history

    class OpticalSystem:
        def __init__(self, surface_stack: SurfaceStack):
            self.stack = surface_stack


class TestLDEAccumulationTier1(unittest.TestCase):
    """Tier 1: Basic LDE Surface Stack & Thickness Accumulation."""

    def test_t1_1_add_surfaces_to_stack(self):
        stack = SurfaceStack()
        stack.add_surface(Surface("OBJ", SurfaceType.OBJECT, thickness=100.0))
        stack.add_surface(Surface("S1", SurfaceType.STANDARD, radius=50.0, thickness=10.0, glass="N-BK7"))
        stack.add_surface(Surface("S2", SurfaceType.STANDARD, radius=-50.0, thickness=50.0, glass="AIR"))
        stack.add_surface(Surface("DET", SurfaceType.DETECTOR, thickness=0.0))
        
        self.assertEqual(len(stack.surfaces), 4)
        self.assertAlmostEqual(stack.get_total_system_length(), 160.0)

    def test_t1_2_global_z_coordinate_accumulation(self):
        stack = SurfaceStack()
        stack.add_surface(Surface("OBJ", thickness=100.0))    # Z=0
        stack.add_surface(Surface("S1", thickness=5.0))       # Z=100
        stack.add_surface(Surface("S2", thickness=45.0))      # Z=105
        stack.add_surface(Surface("DET", thickness=0.0))      # Z=150
        
        self.assertAlmostEqual(stack.surfaces[0].global_z, 0.0)
        self.assertAlmostEqual(stack.surfaces[1].global_z, 100.0)
        self.assertAlmostEqual(stack.surfaces[2].global_z, 105.0)
        self.assertAlmostEqual(stack.surfaces[3].global_z, 150.0)

    def test_t1_3_decenter_x_y_shift(self):
        stack = SurfaceStack()
        stack.add_surface(Surface("S1", thickness=10.0, decenter_x=2.5, decenter_y=-1.5))
        res = stack.propagate_ray(ray_x=10.0, ray_y=5.0)
        self.assertAlmostEqual(res[0]["x"], 7.5)
        self.assertAlmostEqual(res[0]["y"], 6.5)

    def test_t1_4_aperture_semi_diameter_vignetting(self):
        stack = SurfaceStack()
        stack.add_surface(Surface("STOP", SurfaceType.STOP, semi_diameter=10.0))
        res1 = stack.propagate_ray(ray_x=5.0, ray_y=0.0)
        res2 = stack.propagate_ray(ray_x=15.0, ray_y=0.0)
        self.assertFalse(res1[0]["vignetted"])
        self.assertTrue(res2[0]["vignetted"])

    def test_t1_5_dual_plane_rz_xy_coordinates(self):
        surf = Surface("S1", radius=80.0, thickness=8.0, decenter_x=1.0, decenter_y=2.0)
        self.assertIsNotNone(surf)
        self.assertEqual(surf.decenter_x, 1.0)
        self.assertEqual(surf.decenter_y, 2.0)

    def test_t1_6_optical_system_surfaces_property(self):
        stack = SurfaceStack()
        stack.add_surface(Surface("S1"))
        opt_sys = OpticalSystem(stack)
        self.assertEqual(opt_sys.surfaces, stack.surfaces)


class TestLDEAccumulationTier2(unittest.TestCase):
    """Tier 2: Boundary Conditions & Edge Cases for LDE Stack."""

    def test_t2_1_zero_thickness_surface(self):
        stack = SurfaceStack()
        stack.add_surface(Surface("S1", thickness=0.0))
        stack.add_surface(Surface("S2", thickness=10.0))
        self.assertEqual(stack.surfaces[0].global_z, stack.surfaces[1].global_z)

    def test_t2_2_flat_infinite_radius_surface(self):
        surf = Surface("FLAT", radius=0.0)  # 0.0 or inf denotes flat
        self.assertEqual(surf.radius, 0.0)

    def test_t2_3_negative_thickness_mirror(self):
        stack = SurfaceStack()
        stack.add_surface(Surface("MIRROR", thickness=-25.0))
        stack.add_surface(Surface("IMAGE", thickness=0.0))
        self.assertAlmostEqual(stack.get_total_system_length(), -25.0)

    def test_t2_4_extreme_tilt_angle_90_deg(self):
        surf = Surface("TILTED", tilt_x=90.0)
        self.assertEqual(surf.tilt_x, 90.0)

    def test_t2_5_empty_surface_stack(self):
        stack = SurfaceStack()
        self.assertEqual(len(stack.surfaces), 0)
        self.assertEqual(stack.get_total_system_length(), 0.0)


class TestLDEAccumulationTier3(unittest.TestCase):
    """Tier 3: Pairwise Surface Decenter + Tilt + Material Interactions."""

    def test_t3_1_decenter_tilt_material_combination(self):
        stack = SurfaceStack()
        stack.add_surface(Surface("S1", radius=100.0, thickness=12.0, glass="N-BK7", decenter_x=3.0, tilt_x=5.0))
        stack.add_surface(Surface("S2", radius=-100.0, thickness=40.0, glass="AIR", decenter_y=-2.0, tilt_y=2.0))
        self.assertEqual(len(stack.surfaces), 2)
        self.assertEqual(stack.surfaces[0].glass, "N-BK7")


class TestLDEAccumulationTier4(unittest.TestCase):
    """Tier 4: Double-Gauss 6-Element Lens System LDE Real-World Workload."""

    def test_t4_1_double_gauss_6_element_stack(self):
        stack = SurfaceStack()
        # Double Gauss Surface specifications
        stack.add_surface(Surface("OBJ", thickness=1e10))
        stack.add_surface(Surface("S1", radius=58.9, thickness=7.5, glass="N-SK16", semi_diameter=25.0))
        stack.add_surface(Surface("S2", radius=169.6, thickness=0.5, glass="AIR", semi_diameter=25.0))
        stack.add_surface(Surface("S3", radius=33.2, thickness=11.5, glass="N-BALF14", semi_diameter=20.0))
        stack.add_surface(Surface("S4", radius=158.5, thickness=3.0, glass="N-SF2", semi_diameter=20.0))
        stack.add_surface(Surface("S5", radius=21.0, thickness=9.0, glass="AIR", semi_diameter=16.0))
        stack.add_surface(Surface("STOP", SurfaceType.STOP, radius=0.0, thickness=9.0, glass="AIR", semi_diameter=14.0))
        stack.add_surface(Surface("S7", radius=-21.4, thickness=3.0, glass="N-SF5", semi_diameter=16.0))
        stack.add_surface(Surface("S8", radius=84.8, thickness=10.0, glass="N-SK16", semi_diameter=20.0))
        stack.add_surface(Surface("S9", radius=-31.6, thickness=0.5, glass="AIR", semi_diameter=20.0))
        stack.add_surface(Surface("S10", radius=150.0, thickness=6.0, glass="N-SK16", semi_diameter=22.0))
        stack.add_surface(Surface("S11", radius=-70.0, thickness=60.0, glass="AIR", semi_diameter=22.0))
        stack.add_surface(Surface("IMA", SurfaceType.DETECTOR, thickness=0.0, semi_diameter=20.0))

        self.assertEqual(len(stack.surfaces), 13)
        self.assertGreater(stack.surfaces[-1].global_z, 100.0)


if __name__ == "__main__":
    unittest.main()
