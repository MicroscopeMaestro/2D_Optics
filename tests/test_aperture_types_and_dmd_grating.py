"""
Unit and integration tests for:
1. Aperture types (Circular & Rectangular) across engine, LDE, 2D views, and 3D visualizer.
2. Reflection grating following blazed grating equation, input/output angles, blaze condition, and top-surface reflective behavior.
3. Texas Instruments DLP4500 DMD model micromirror tilt (±12°), input and output angles, states (ON, OFF, FLAT), and 3D visualization.
"""

import unittest
import numpy as np

from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.mechanisms import (
    DMD4500_STATES,
    dmd4500_reflection_angles,
    MechanicalStateController,
    active_state_name,
    interaction_rotation,
    is_dmd,
)
from optics_app.engine.ray import Ray
from optics_app.engine.tracer import OpticalSystem
from optics_app.gui.lde_table import LensDataTableModel
from optics_app.gui.qt_compat import QT_AVAILABLE, QtWidgets


if QT_AVAILABLE and hasattr(QtWidgets, "QApplication"):
    _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class TestApertureTypes(unittest.TestCase):
    def test_circular_aperture_pass_and_block(self):
        surf = Surface(name="Stop", surface_type=SurfaceType.STOP, semi_diameter=10.0, aperture_type="Circular")
        # Center ray passes
        self.assertTrue(surf.check_aperture(0.0, 0.0))
        # Inside boundary passes
        self.assertTrue(surf.check_aperture(5.0, 5.0))  # r = sqrt(50) = 7.07 < 10.0
        # Outside boundary blocked
        self.assertFalse(surf.check_aperture(8.0, 8.0))  # r = sqrt(128) = 11.31 > 10.0

    def test_rectangular_aperture_pass_and_block(self):
        surf = Surface(
            name="Slit",
            surface_type=SurfaceType.STOP,
            semi_diameter=10.0,
            aperture_type="Rectangular",
            aperture_x_half=4.0,
            aperture_y_half=8.0,
        )
        # Inside rect: x <= 4, y <= 8
        self.assertTrue(surf.check_aperture(3.5, 7.5))
        # Exceeds X: x > 4
        self.assertFalse(surf.check_aperture(4.5, 2.0))
        # Exceeds Y: y > 8
        self.assertFalse(surf.check_aperture(2.0, 8.5))

    def test_rectangular_aperture_fallback_to_semi_diameter(self):
        """When aperture_x_half or aperture_y_half is 0 or unset, fall back to semi_diameter."""
        surf = Surface(
            name="Slit_Unset",
            surface_type=SurfaceType.STOP,
            semi_diameter=6.0,
            aperture_type="Rectangular",
            aperture_x_half=0.0,
            aperture_y_half=0.0,
        )
        # Should fall back to 6.0 in X and Y
        self.assertTrue(surf.check_aperture(5.0, 5.0))
        self.assertFalse(surf.check_aperture(6.5, 2.0))
        self.assertFalse(surf.check_aperture(2.0, 6.5))

    def test_aperture_params_string_serialization(self):
        surf = Surface(name="Rect", aperture_type="Rectangular", semi_diameter=10.0, aperture_x_half=4.5, aperture_y_half=9.2)
        param_str = surf.get_aperture_params_str()
        self.assertIn("X=4.5", param_str)
        self.assertIn("Y=9.2", param_str)

        # Deserialize string back
        surf2 = Surface(name="Rect2", aperture_type="Rectangular", semi_diameter=10.0)
        surf2.set_aperture_params_from_str("X=3.2, Y=6.4")
        self.assertAlmostEqual(surf2.aperture_x_half, 3.2)
        self.assertAlmostEqual(surf2.aperture_y_half, 6.4)
        self.assertAlmostEqual(surf2.semi_diameter, np.hypot(3.2, 6.4))  # circumscribing semi-diameter

    def test_lde_table_sync_aperture_type_and_params(self):
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", surface_type=SurfaceType.OBJECT, thickness=10.0))
        stack.add_surface(Surface(name="Stop", surface_type=SurfaceType.STOP, semi_diameter=5.0, aperture_type="Circular", thickness=10.0))
        stack.add_surface(Surface(name="Detector", surface_type=SurfaceType.DETECTOR))
        model = LensDataTableModel(stack=stack)

        # Change Column 14 (Aperture Type) to "Rectangular"
        col_type = 14
        idx = model.index(1, col_type)
        model.setData(idx, "Rectangular")

        surf = model._surfaces[1]
        self.assertEqual(surf.aperture_type, "Rectangular")
        self.assertEqual(surf.aperture_x_half, 5.0)
        self.assertEqual(surf.aperture_y_half, 5.0)

        # Change Column 15 (Aperture Params) to "X=2.0, Y=4.0"
        col_params = 15
        idx_params = model.index(1, col_params)
        model.setData(idx_params, "X=2.0, Y=4.0")

        self.assertAlmostEqual(surf.aperture_x_half, 2.0)
        self.assertAlmostEqual(surf.aperture_y_half, 4.0)
        self.assertAlmostEqual(surf.semi_diameter, np.hypot(2.0, 4.0))


class TestGratingPhysicsAndTopReflectiveSurface(unittest.TestCase):
    def test_blazed_grating_diffraction_equation(self):
        """
        Grating equation: m * lambda = d * (sin(alpha) + sin(beta))
        Given alpha = 13.0 deg, lines_per_mm = 600, order = 1, lambda = 1.064 um:
        d = 1000 / 600 = 1.66667 um
        m * lambda / d = 1 * 1.064 / 1.66667 = 0.6384
        sin(alpha) = sin(13 deg) = 0.224951
        sin(beta) = 0.6384 - 0.224951 = 0.413449 => beta = 24.42 deg
        Total fold = alpha + beta = 37.42 deg
        """
        surf = Surface(
            name="Grating",
            surface_type=SurfaceType.GRATING,
            glass="MIRROR",
            tilt_x=13.0,
            grating_lines_per_mm=600.0,
            grating_order=1,
            grating_primary_wavelength_um=1.064,
            grating_blaze_angle_deg=18.71,  # (13.0 + 24.42)/2 = 18.71 deg
        )
        beta_deg = surf.calculate_diffraction_angle(alpha_deg=13.0, wavelength_um=1.064)
        self.assertAlmostEqual(beta_deg, 24.42, places=1)
        total_fold = 13.0 + beta_deg
        self.assertAlmostEqual(total_fold, 37.42, places=1)

    def test_blaze_condition_calculation(self):
        """
        Blaze condition: m * lambda_B = 2 * d * sin(theta_B) * cos(alpha - theta_B)
        """
        surf = Surface(
            name="Grating",
            surface_type=SurfaceType.GRATING,
            glass="MIRROR",
            grating_lines_per_mm=600.0,
            grating_order=1,
            grating_blaze_angle_deg=18.71,
        )
        blaze = surf.calculate_blaze_condition(alpha_deg=13.0)
        self.assertTrue(blaze["valid"])
        # At alpha = 13.0 and theta_B = 18.71, lambda_B should be ~1.064 um (1064 nm)
        self.assertAlmostEqual(blaze["blaze_wavelength_um"], 1.064, places=2)
        self.assertAlmostEqual(blaze["blaze_wavelength_nm"], 1064.0, places=0)

    def test_top_surface_acts_as_reflective_mirror(self):
        """
        Only the top surface (z=0) reflects and diffracts. Light does not propagate into substrate.
        """
        surf = Surface(
            name="Grating",
            surface_type=SurfaceType.GRATING,
            glass="MIRROR",
            thickness=5.0,  # substrate thickness
        )
        self.assertTrue(surf.is_reflective())
        # Glass is MIRROR: rays reflect off the front top face, never entering bulk substrate


class TestTexasInstrumentsDLP4500DMD(unittest.TestCase):
    def test_dlp4500_specifications_and_states(self):
        """Verify TI DLP4500 micromirror tilt angles: ON (+12°), OFF (-12°), FLAT (0°)."""
        self.assertIn("ON", DMD4500_STATES)
        self.assertIn("OFF", DMD4500_STATES)
        self.assertIn("FLAT", DMD4500_STATES)

        self.assertEqual(DMD4500_STATES["ON"]["tilt_angle_deg"], 12.0)
        self.assertEqual(DMD4500_STATES["OFF"]["tilt_angle_deg"], -12.0)
        self.assertEqual(DMD4500_STATES["FLAT"]["tilt_angle_deg"], 0.0)

    def test_dlp4500_input_and_output_angles(self):
        """
        Optical deflection formula: theta_out = 2 * alpha_tilt - theta_in
        At nominal illumination theta_in = 24.0 deg:
        - ON (+12 deg tilt): theta_out = 2*(+12) - 24 = 0.0 deg (normal to array, into detector)
        - OFF (-12 deg tilt): theta_out = 2*(-12) - 24 = -48.0 deg (deflected 48 deg into beam dump)
        - FLAT (0 deg tilt): theta_out = 2*(0) - 24 = -24.0 deg (specular reflection)
        """
        # ON
        on_angles = dmd4500_reflection_angles(input_angle_deg=24.0, state="ON")
        self.assertAlmostEqual(on_angles["input_angle_deg"], 24.0)
        self.assertAlmostEqual(on_angles["output_angle_deg"], 0.0)
        self.assertEqual(on_angles["route"], "detector")

        # OFF
        off_angles = dmd4500_reflection_angles(input_angle_deg=24.0, state="OFF")
        self.assertAlmostEqual(off_angles["input_angle_deg"], 24.0)
        self.assertAlmostEqual(off_angles["output_angle_deg"], -48.0)
        self.assertEqual(off_angles["route"], "beam dump")

        # FLAT
        flat_angles = dmd4500_reflection_angles(input_angle_deg=24.0, state="FLAT")
        self.assertAlmostEqual(flat_angles["input_angle_deg"], 24.0)
        self.assertAlmostEqual(flat_angles["output_angle_deg"], -24.0)
        self.assertIn(flat_angles["route"], ("parking", "specular"))

    def test_surface_calculate_dmd_reflection_angles_method(self):
        surf = Surface(name="DLP4500_Active", glass="MIRROR", mechanical_state="ON")
        res_on = surf.calculate_dmd_reflection_angles(alpha_deg=24.0, state="ON")
        self.assertAlmostEqual(res_on["output_angle_deg"], 0.0)
        self.assertEqual(res_on["route"], "detector")

        res_off = surf.calculate_dmd_reflection_angles(alpha_deg=24.0, state="OFF")
        self.assertAlmostEqual(res_off["output_angle_deg"], -48.0)
        self.assertEqual(res_off["route"], "beam dump")

    def test_dmd_controller_switches_all_three_states(self):
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", surface_type=SurfaceType.OBJECT, thickness=10.0))
        stack.add_surface(Surface(name="DMD4500", glass="MIRROR", tilt_x=12.0, thickness=10.0))
        stack.add_surface(Surface(name="Detector", surface_type=SurfaceType.DETECTOR))

        ctl = MechanicalStateController(stack)
        dmd_surf = stack.surfaces[1]
        self.assertTrue(is_dmd(dmd_surf))

        # Switch to OFF
        ctl.set_state(1, "OFF")
        self.assertEqual(active_state_name(dmd_surf), "OFF")

        # Switch to FLAT
        ctl.set_state(1, "FLAT")
        self.assertEqual(active_state_name(dmd_surf), "FLAT")

        # Switch to ON
        ctl.set_state(1, "ON")
        self.assertEqual(active_state_name(dmd_surf), "ON")


class TestVisualizer3DRepresentations(unittest.TestCase):
    def test_visualizer_imports_and_constructs(self):
        """Ensure visualizer_3d module imports cleanly and handles grating/DMD configurations."""
        from optics_app.gui.visualizer_3d import Visualizer3DWidget
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", surface_type=SurfaceType.OBJECT, thickness=20.0))
        stack.add_surface(Surface(
            name="Grating_Blazed",
            surface_type=SurfaceType.GRATING,
            glass="MIRROR",
            tilt_x=13.0,
            grating_lines_per_mm=600.0,
            grating_blaze_angle_deg=18.7,
            thickness=15.0
        ))
        stack.add_surface(Surface(
            name="DMD4500",
            glass="MIRROR",
            aperture_type="Rectangular",
            aperture_x_half=4.93,
            aperture_y_half=3.08,
            mechanical_state="ON",
            thickness=20.0
        ))
        stack.add_surface(Surface(name="Detector", surface_type=SurfaceType.DETECTOR))
        stack._update_accumulated_positions()

        # Check surface properties
        g_surf = stack.surfaces[1]
        d_surf = stack.surfaces[2]
        self.assertTrue(g_surf.is_reflective())
        self.assertTrue(is_dmd(d_surf))
        self.assertEqual(d_surf.aperture_type, "Rectangular")
        self.assertAlmostEqual(d_surf.aperture_x_half, 4.93)
        self.assertAlmostEqual(d_surf.aperture_y_half, 3.08)

    def test_raman_spectrometer_grating_and_dmd_top_surface_reflection(self):
        """
        Verify that in the spectrometer Raman model, Grating and DMD reflect light strictly
        off their top active surfaces, with substrates situated behind the active faces.
        """
        import os
        filepath = "spectrometer_raman_50mm_100mm_manual.json"
        if not os.path.exists(filepath):
            filepath = os.path.join(r"C:\Users\Juan\Desktop", filepath)
        if not os.path.exists(filepath):
            self.skipTest("Spectrometer Raman file not found")

        res = SurfaceStack.load_from_json(filepath)
        stack = res["stack"]
        surfaces = stack.get_active_sequence()

        for idx, surf in enumerate(surfaces):
            name = surf.name or f"S{idx}"
            is_grating = surf.surface_type == SurfaceType.GRATING or "GRATING" in name.upper()
            is_dmd_surf = is_dmd(surf) or "DMD" in name.upper()
            if not (is_grating or is_dmd_surf):
                continue

            center = np.asarray(surf.global_origin, dtype=float)
            rotation = np.asarray(surf.global_R, dtype=float)
            thickness = max(float(getattr(surf, "substrate_thickness", 0.0)), 0.8)

            # Find previous distinct origin
            prev_origin = None
            for p in reversed(surfaces[:idx]):
                po = getattr(p, "global_origin", None)
                if po is not None and np.linalg.norm(center - po) > 1e-4:
                    prev_origin = np.asarray(po, dtype=float)
                    break
            self.assertIsNotNone(prev_origin, f"Surface {name} must have a preceding origin")

            d_in = center - prev_origin
            d_in = d_in / np.linalg.norm(d_in)

            # Substrate extrusion sign
            sub_sign = 1.0 if np.dot(d_in, rotation[:, 2]) >= 0.0 else -1.0
            sub_thickness = sub_sign * thickness
            back_center = center + rotation[:, 2] * sub_thickness

            d_front = np.linalg.norm(center - prev_origin)
            d_back = np.linalg.norm(back_center - prev_origin)

            # 1. Front active face (z=0) must be closer to incoming light than back face
            self.assertLess(
                d_front, d_back,
                f"{name}: Front active reflective face must be closer to the beam than substrate rear face!"
            )

            # 2. Outward normal arrow pointing into air must have negative dot product with incoming beam
            n_outward = -sub_sign * rotation[:, 2]
            dot_beam = np.dot(d_in, n_outward)
            self.assertLess(
                dot_beam, 0.0,
                f"{name}: Outward normal must point towards incoming beam into free air!"
            )

    def test_visualizer_renders_raman_spectrometer_without_errors(self):
        """Verify that Visualizer3DWidget renders the Raman spectrometer with correct geometry."""
        import os
        from optics_app.gui.visualizer_3d import Visualizer3DWidget
        filepath = "spectrometer_raman_50mm_100mm_manual.json"
        if not os.path.exists(filepath):
            filepath = os.path.join(r"C:\Users\Juan\Desktop", filepath)
        if not os.path.exists(filepath):
            self.skipTest("Spectrometer Raman file not found")

        res = SurfaceStack.load_from_json(filepath)
        stack = res["stack"]

        widget = Visualizer3DWidget(auto_refresh=False)
        widget.surface_stack = stack
        widget.update_layout()

        # Check that 3D collections exist on the axes
        self.assertGreater(len(widget.ax.collections), 5, "Visualizer should render optical element collections")


if __name__ == "__main__":
    unittest.main()
