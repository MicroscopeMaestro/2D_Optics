"""
Unit Tests for Dynamic DMD Configuration and 2D Viewer Ray Updates.

Validates:
1. Dynamic ray tracing deflection when changing DMD state (ON, OFF, FLAT).
2. Dynamic ray tracing deflection when enabling and changing DMD binary grating parameters
   (pitch, lines/mm, diffraction order, orientation).
3. Signal flow from TI Modulator Studio (`pattern_applied`) triggering automatic ray re-tracing
   and 2D viewer layout updates (`DualViewWidget.update_layout`).
4. DualViewWidget geometry fingerprint and ray artist update when DMD configuration changes.
"""

import unittest
import numpy as np

from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from optics_app.engine.plm import ensure_plm_config, is_dmd
from optics_app.engine.mechanisms import MechanicalStateController, get_effective_pose
from optics_app.gui.plm_widget import PLMStudioWidget
from optics_app.gui.dual_view import DualViewWidget
from optics_app.gui.qt_compat import QT_AVAILABLE, QtWidgets


if QT_AVAILABLE and hasattr(QtWidgets, "QApplication"):
    _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class TestDMDDynamicRays(unittest.TestCase):
    def setUp(self):
        self.stack = SurfaceStack()
        self.s_obj = Surface(name="Object", surface_type=SurfaceType.OBJECT, thickness=50.0)
        self.s_dmd = Surface(name="TI_DMD_DLP4500", surface_type=SurfaceType.MIRROR, thickness=50.0)
        self.s_dmd.is_dmd = True
        self.s_dmd.tilt_x = 12.0
        self.s_dmd.semi_diameter = 10.0
        self.s_det = Surface(name="Detector", surface_type=SurfaceType.DETECTOR, thickness=0.0)
        self.s_det.semi_diameter = 20.0

        self.stack.add_surface(self.s_obj)
        self.stack.add_surface(self.s_dmd)
        self.stack.add_surface(self.s_det)
        self.stack._update_accumulated_positions()

        self.cfg = ensure_plm_config(self.s_dmd)
        self.cfg.modulator_type = "DMD"

    def test_dmd_states_dynamically_change_ray_trajectories(self):
        """Verify that ON, OFF, and FLAT states produce distinctly deflected rays in 3D & 2D."""
        sys = OpticalSystem(self.stack)
        ray_in = Ray(pos=[0.0, 0.0, 0.0], dir=[0.0, 0.0, 1.0], wavelength=1.225)

        # 1. State ON
        self.s_dmd.mechanical_state = "ON"
        res_on = sys.trace_ray(ray_in.copy())
        dir_on = res_on.history[1]["dir"]

        # 2. State OFF
        self.s_dmd.mechanical_state = "OFF"
        res_off = sys.trace_ray(ray_in.copy())
        dir_off = res_off.history[1]["dir"]

        # 3. State FLAT
        self.s_dmd.mechanical_state = "FLAT"
        res_flat = sys.trace_ray(ray_in.copy())
        dir_flat = res_flat.history[1]["dir"]

        # Assert all 3 directions are non-identical and physically distinct
        self.assertFalse(np.allclose(dir_on, dir_off, atol=1e-3))
        self.assertFalse(np.allclose(dir_on, dir_flat, atol=1e-3))
        self.assertFalse(np.allclose(dir_off, dir_flat, atol=1e-3))

        # Check angular deflection between ON and OFF is ~48 degrees
        cos_angle = np.clip(np.dot(dir_on, dir_off), -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(cos_angle))
        self.assertAlmostEqual(angle_deg, 48.0, delta=2.0)

    def test_dmd_binary_grating_dynamically_alters_deflection(self):
        """Verify that enabling/modifying binary grating parameters on DMD deflects rays."""
        sys = OpticalSystem(self.stack)
        ray_in = Ray(pos=[0.0, 0.0, 0.0], dir=[0.0, 0.0, 1.0], wavelength=1.225)
        self.s_dmd.mechanical_state = "ON"

        # Baseline: No grating
        self.cfg.grating.enabled = False
        res_nograt = sys.trace_ray(ray_in.copy())
        dir_nograt = res_nograt.history[1]["dir"]

        # Grating with 50 lines/mm, order +1
        self.cfg.grating.enabled = True
        self.cfg.grating.lines_per_mm = 50.0
        self.cfg.grating.diffraction_order = 1
        self.cfg.grating.orientation_deg = 0.0
        res_g50 = sys.trace_ray(ray_in.copy())
        dir_g50 = res_g50.history[1]["dir"]

        # Ray must deflect differently with 50 l/mm grating
        self.assertFalse(np.allclose(dir_nograt, dir_g50, atol=1e-3))

        # Increase line density to 100 lines/mm
        self.cfg.grating.lines_per_mm = 100.0
        res_g100 = sys.trace_ray(ray_in.copy())
        dir_g100 = res_g100.history[1]["dir"]

        # Grating 100 l/mm must deflect further than 50 l/mm
        deflection_50 = np.linalg.norm(dir_g50 - dir_nograt)
        deflection_100 = np.linalg.norm(dir_g100 - dir_nograt)
        self.assertGreater(deflection_100, deflection_50)

        # Order -1 must deflect in opposing direction relative to order +1
        self.cfg.grating.diffraction_order = -1
        res_m1 = sys.trace_ray(ray_in.copy())
        dir_m1 = res_m1.history[1]["dir"]

        self.assertFalse(np.allclose(dir_g100, dir_m1, atol=1e-3))
        # Y direction component of -1 order must be less than specular / +1 order
        self.assertLess(dir_m1[1], dir_nograt[1])
        self.assertGreater(dir_g100[1], dir_nograt[1])

    def test_ti_modulator_studio_dmd_sync_and_retrace_signals(self):
        """Verify that UI controls in PLMStudioWidget sync to DMD surface and emit pattern_applied."""
        widget = PLMStudioWidget(surface_stack=self.stack)
        if widget.combo_surface.count() > 1:
            widget.combo_surface.setCurrentIndex(1)

        # Track signals
        emitted_indices = []
        widget.pattern_applied.connect(lambda idx: emitted_indices.append(idx))

        # Switch to DMD mode
        widget.combo_modulator_type.setCurrentIndex(1)
        self.assertTrue(widget.combo_modulator_type.currentIndex() == 1)

        # Change state to OFF
        widget.combo_state.setCurrentText("OFF")
        self.assertEqual(self.s_dmd.mechanical_state, "OFF")
        self.assertIn(1, emitted_indices)

        # Change grating lines/mm
        emitted_indices.clear()
        widget.chk_grating_enable.setChecked(True)
        widget.spin_lpmm.setValue(75.0)
        self.assertTrue(self.s_dmd.plm_config.grating.enabled)
        self.assertAlmostEqual(self.s_dmd.plm_config.grating.lines_per_mm, 75.0, places=1)
        self.assertAlmostEqual(self.s_dmd.grating_lines_per_mm, 75.0, places=1)
        self.assertIn(1, emitted_indices)

    def test_dual_view_widget_updates_on_dmd_parameter_changes(self):
        """Verify that DualViewWidget updates its 2D canvas and ray artists when DMD changes."""
        dv = DualViewWidget()

        sys = OpticalSystem(self.stack)
        ray_in = Ray(pos=[0.0, 0.0, 0.0], dir=[0.0, 0.0, 1.0], wavelength=1.225)
        rays_on = [sys.trace_ray(ray_in.copy())]

        dv.update_layout(surface_stack=self.stack, rays=rays_on)
        fp_on = dv._last_geom_fingerprint

        # Toggle DMD state to OFF
        self.s_dmd.mechanical_state = "OFF"
        rays_off = [sys.trace_ray(ray_in.copy())]

        dv.update_layout(surface_stack=self.stack, rays=rays_off)
        fp_off = dv._last_geom_fingerprint

        # Fingerprint must change because mechanical state / effective pose changed
        self.assertNotEqual(fp_on, fp_off)

        # Grating change also changes fingerprint
        self.cfg.grating.enabled = True
        self.cfg.grating.lines_per_mm = 60.0
        rays_grat = [sys.trace_ray(ray_in.copy())]
        dv.update_layout(surface_stack=self.stack, rays=rays_grat)
        fp_grat = dv._last_geom_fingerprint
        self.assertNotEqual(fp_off, fp_grat)

    def test_2d_pattern_mask_takes_effect_on_rays_and_vignetting(self):
        """Verify that 2D pattern masks (Hadamard, slits, checkerboard) deflect & vignette rays on DMD."""
        sys = OpticalSystem(self.stack)
        self.s_dmd.mechanical_state = "ON"
        self.cfg.pattern.enabled = True
        self.cfg.pattern.pattern_type = "hadamard"
        self.cfg.pattern.hadamard_order = 16
        self.cfg.pattern.hadamard_row_index = 1
        self.cfg.pattern.hadamard_axis = "Y"

        # Trace a fan of rays across Y aperture
        y_vals = np.linspace(-3.5, 3.5, 9)
        traced_rays = []
        for y in y_vals:
            r = Ray(pos=[0.0, float(y), 0.0], dir=[0.0, 0.0, 1.0], wavelength=1.225)
            traced = sys.trace_ray(r)
            traced_rays.append(traced)

        # There must be both ON (unvignetted) and OFF (vignetted dump) rays
        vignetted_rays = [r for r in traced_rays if r.vignetted]
        aligned_rays = [r for r in traced_rays if not r.vignetted]

        self.assertGreater(len(vignetted_rays), 0, "Pattern mask must deflect OFF rays into the dump")
        self.assertGreater(len(aligned_rays), 0, "Pattern mask must preserve ON rays along the channel")

        # Vignetted rays have zero intensity
        for r in vignetted_rays:
            self.assertEqual(r.intensity, 0.0)

        # Deflected OFF rays have ~ -48 deg deflection compared to aligned ON rays
        d_on = aligned_rays[0].history[1]["dir"]
        d_off = vignetted_rays[0].history[1]["dir"]
        dot_prod = np.dot(d_on, d_off)
        angle_deg = np.degrees(np.arccos(np.clip(dot_prod, -1.0, 1.0)))
        self.assertAlmostEqual(angle_deg, 48.0, delta=2.0)

    def test_2d_pattern_mask_rendered_in_dual_view_2d_projections(self):
        """Verify that 2D pattern mask is rendered in Front View (X-Y) and annotated in 2D projections."""
        dv = DualViewWidget()
        dv.set_xy_front_view_active(True)
        self.assertTrue(dv.xy_front_view_active)

        self.s_dmd.mechanical_state = "ON"
        self.cfg.pattern.enabled = True
        self.cfg.pattern.pattern_type = "hadamard"
        self.cfg.pattern.hadamard_order = 16
        self.cfg.pattern.hadamard_row_index = 1
        self.cfg.pattern.hadamard_axis = "Y"

        sys = OpticalSystem(self.stack)
        rays = [sys.trace_ray(Ray(pos=[0.0, y, 0.0], dir=[0.0, 0.0, 1.0], wavelength=1.225)) for y in [-2.0, 2.0]]
        dv.update_layout(surface_stack=self.stack, rays=rays)

        # Front View (X-Y) should have collections (pcolormesh and ray lines) and texts/badges
        self.assertIsNotNone(dv.ax_xy_front)
        self.assertGreater(len(dv.ax_xy_front.collections), 0)
        front_texts = [t.get_text() for t in dv.ax_xy_front.texts]
        self.assertTrue(any("TI-DMD Mask:" in t for t in front_texts), f"Front view missing badge: {front_texts}")

        # Longitudinal View (R-Z / Y-Z) should show the mask badge
        self.assertIsNotNone(dv.ax_rz)
        rz_texts = [t.get_text() for t in dv.ax_rz.texts]
        self.assertTrue(any("TI-DMD Mask:" in t for t in rz_texts), f"R-Z view missing badge: {rz_texts}")

    def test_far_field_psf_zoom_out_and_hdr_log_scale(self):
        """Verify that Far-Field PSF can be zoomed out to full field and displays in HDR log scale."""
        widget = PLMStudioWidget(surface_stack=self.stack)
        self.assertTrue(hasattr(widget, "btn_psf_zoom_out"))
        self.assertTrue(hasattr(widget, "combo_psf_zoom"))
        self.assertTrue(hasattr(widget, "chk_psf_log"))

        # Default zoom level is Full Field (Zoom Out 1x)
        self.assertEqual(widget.combo_psf_zoom.currentIndex(), 0)
        self.assertTrue(widget.chk_psf_log.isChecked())

        # Change zoom to Medium (4x)
        widget.combo_psf_zoom.setCurrentIndex(2)
        widget.update_live_plots()
        titles = [ax.get_title() for ax in widget.fig.axes if ax.get_title()]
        self.assertTrue(any("4×" in t for t in titles))

        # Click Zoom Out button -> returns to Full Field (1x)
        widget.btn_psf_zoom_out.click()
        self.assertEqual(widget.combo_psf_zoom.currentIndex(), 0)
        widget.update_live_plots()
        titles = [ax.get_title() for ax in widget.fig.axes if ax.get_title()]
        self.assertTrue(any("Zoom Out 1×" in t for t in titles))

        # Toggle Log Scale off and on
        widget.chk_psf_log.setChecked(False)
        widget.update_live_plots()
        widget.chk_psf_log.setChecked(True)
        widget.update_live_plots()

    def test_dmd_column_range_control_and_ray_vignetting(self):
        """Verify programmable column range (slit/bandpass): selected columns ON, rest OFF & dumped."""
        sys = OpticalSystem(self.stack)
        self.s_dmd.mechanical_state = "ON"
        self.cfg.pattern.enabled = True
        self.cfg.pattern.pattern_type = "Column_Range"
        # DLP4500 has 1140 columns across 8.664 mm width (-4.332 mm to +4.332 mm)
        # Column 570 is at x = 0.0 mm
        self.cfg.pattern.column_start = 550
        self.cfg.pattern.column_end = 650
        self.cfg.pattern.column_invert = False

        # Ray 1: Passes right through the center (x = 0.0 mm -> column 570, ON)
        ray_center = sys.trace_ray(Ray(pos=[0.0, 0.0, 0.0], dir=[0.0, 0.0, 1.0], wavelength=1.225))
        self.assertFalse(ray_center.vignetted, "Center ray in [550, 650] column range must reflect ON")
        self.assertGreater(ray_center.intensity, 0.0)

        # Ray 2: Hits at x = -2.5 mm -> column ~241 (OFF)
        ray_outside = sys.trace_ray(Ray(pos=[-2.5, 0.0, 0.0], dir=[0.0, 0.0, 1.0], wavelength=1.225))
        self.assertTrue(ray_outside.vignetted, "Outside ray (col 241) must be tilted OFF into beam dump")
        self.assertEqual(ray_outside.intensity, 0.0)

        # Deflection angle check: OFF ray should be deflected by ~48 deg into beam dump
        d_on = ray_center.history[1]["dir"]
        d_off = ray_outside.history[1]["dir"]
        dot_prod = np.clip(np.dot(d_on, d_off), -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(dot_prod))
        self.assertAlmostEqual(angle_deg, 48.0, delta=2.0)

        # Invert mode: columns in [550, 650] are now OFF (notch), while outside columns are ON
        self.cfg.pattern.column_invert = True
        ray_center_inv = sys.trace_ray(Ray(pos=[0.0, 0.0, 0.0], dir=[0.0, 0.0, 1.0], wavelength=1.225))
        ray_outside_inv = sys.trace_ray(Ray(pos=[-2.5, 0.0, 0.0], dir=[0.0, 0.0, 1.0], wavelength=1.225))

        self.assertTrue(ray_center_inv.vignetted, "Invert mode: center ray must now be deflected to dump")
        self.assertEqual(ray_center_inv.intensity, 0.0)
        self.assertFalse(ray_outside_inv.vignetted, "Invert mode: outside ray must now reflect ON")
        self.assertGreater(ray_outside_inv.intensity, 0.0)

    def test_dmd_column_range_gui_sync_and_presets(self):
        """Verify PLMStudioWidget column range controls, presets, and retrace triggering."""
        widget = PLMStudioWidget(surface_stack=self.stack)
        if widget.combo_surface.count() > 1:
            widget.combo_surface.setCurrentIndex(1)

        # Switch to DMD mode
        widget.combo_modulator_type.setCurrentIndex(1)
        self.assertTrue(widget.combo_modulator_type.currentIndex() == 1)

        # Ensure Column Range controls exist
        self.assertTrue(hasattr(widget, "grp_column_range"))
        self.assertTrue(hasattr(widget, "spin_col_start"))
        self.assertTrue(hasattr(widget, "spin_col_end"))
        self.assertTrue(hasattr(widget, "chk_col_invert"))
        self.assertTrue(hasattr(widget, "lbl_col_width_info"))

        # Select Column_Range in pattern combo
        widget.combo_pattern_type.setCurrentText("Column_Range")
        self.assertFalse(widget.grp_column_range.isHidden())

        # Test preset: Center 50 Cols
        widget._set_column_preset(50)
        c_start = widget.spin_col_start.value()
        c_end = widget.spin_col_end.value()
        num_cols = c_end - c_start + 1
        self.assertEqual(num_cols, 50, "Preset Center 50 must configure exactly 50 columns")
        self.assertEqual(self.s_dmd.plm_config.pattern.pattern_type, "Column_Range")
        self.assertEqual(self.s_dmd.plm_config.pattern.column_start, c_start)
        self.assertEqual(self.s_dmd.plm_config.pattern.column_end, c_end)

        # Verify active slit width info readout
        info_text = widget.lbl_col_width_info.text()
        self.assertIn("50 cols", info_text)
        self.assertIn("0.380 mm", info_text)

        # Test preset: Full array (All ON)
        widget._set_column_preset(-1)
        self.assertEqual(widget.spin_col_start.value(), 0)
        self.assertEqual(widget.spin_col_end.value(), self.s_dmd.plm_config.array_cols - 1)

    def test_dmd_column_range_dual_view_rendering(self):
        """Verify DualViewWidget front and longitudinal views update with Column Range badge."""
        dv = DualViewWidget()
        dv.set_xy_front_view_active(True)
        dv.set_xz_view_active(True)

        self.s_dmd.mechanical_state = "ON"
        self.cfg.pattern.enabled = True
        self.cfg.pattern.pattern_type = "Column_Range"
        self.cfg.pattern.column_start = 500
        self.cfg.pattern.column_end = 600
        self.cfg.pattern.column_invert = False

        sys = OpticalSystem(self.stack)
        rays = [sys.trace_ray(Ray(pos=[0.0, 0.0, 0.0], dir=[0.0, 0.0, 1.0], wavelength=1.225))]
        dv.update_layout(surface_stack=self.stack, rays=rays)

        # Front View (X-Y) should show Column Range badge
        front_texts = [t.get_text() for t in dv.ax_xy_front.texts]
        self.assertTrue(any("Cols 500..600" in t for t in front_texts), f"Front view badge missing column range: {front_texts}")

        # Longitudinal View (R-Z / X-Z) should also display the column range
        xz_texts = [t.get_text() for t in dv.ax_xz.texts]
        self.assertTrue(any("Cols 500..600" in t for t in xz_texts), f"X-Z view badge missing column range: {xz_texts}")

    def test_panel_rotation_clocking_deflection_minus_45_and_90(self):
        """Verify panel clocking (tilt_z): -45° deflects diagonally along X=Y; 90° rotates deflection to X axis."""
        sys = OpticalSystem(self.stack)
        ray_in = Ray(pos=[0.0, 0.0, 0.0], dir=[0.0, 0.0, 1.0], wavelength=1.225)
        self.s_dmd.mechanical_state = "ON"
        self.cfg.grating.enabled = False

        # 1. Baseline: 0° tilt_z (pure vertical deflection in Y)
        self.s_dmd.tilt_z = 0.0
        self.stack._update_accumulated_positions()
        res_0 = sys.trace_ray(ray_in.copy())
        dir_0 = res_0.history[1]["dir"]
        self.assertAlmostEqual(dir_0[0], 0.0, delta=1e-3)
        self.assertGreater(dir_0[1], 0.40)

        # 2. -45° panel rotation (diagonal deflection: dir_x == dir_y > 0)
        self.s_dmd.tilt_z = -45.0
        self.stack._update_accumulated_positions()
        res_m45 = sys.trace_ray(ray_in.copy())
        dir_m45 = res_m45.history[1]["dir"]
        self.assertGreater(dir_m45[0], 0.25)
        self.assertGreater(dir_m45[1], 0.25)
        self.assertAlmostEqual(dir_m45[0], dir_m45[1], delta=1e-3, msg="At -45° clocking, X and Y deflections must match symmetrically")

        # 3. 90° panel rotation (deflection rotated from Y axis to horizontal X axis)
        self.s_dmd.tilt_z = 90.0
        self.stack._update_accumulated_positions()
        res_90 = sys.trace_ray(ray_in.copy())
        dir_90 = res_90.history[1]["dir"]
        self.assertAlmostEqual(dir_90[1], 0.0, delta=1e-3, msg="At 90° clocking, Y deflection must be zero")
        self.assertLess(dir_90[0], -0.40, msg="At 90° clocking, ray must deflect along horizontal X axis")

        # 4. Verify DualViewWidget geometry fingerprint updates when tilt_z changes
        dv = DualViewWidget()
        self.s_dmd.tilt_z = 0.0
        self.stack._update_accumulated_positions()
        res_0 = sys.trace_ray(ray_in.copy())
        dv.update_layout(surface_stack=self.stack, rays=[res_0])
        fp_0 = dv._last_geom_fingerprint

        self.s_dmd.tilt_z = -45.0
        self.stack._update_accumulated_positions()
        res_m45 = sys.trace_ray(ray_in.copy())
        dv.update_layout(surface_stack=self.stack, rays=[res_m45])
        fp_m45 = dv._last_geom_fingerprint
        self.assertNotEqual(fp_0, fp_m45, "DualView fingerprint must update when panel is clocked to -45°")

    def test_dmd_slit_pattern_rotation_minus_45_and_90(self):
        """Verify rotating pattern slit angle on DMD array (-45° diagonal slit, 90° horizontal rows)."""
        sys = OpticalSystem(self.stack)
        self.s_dmd.mechanical_state = "ON"
        self.s_dmd.tilt_z = 0.0
        self.stack._update_accumulated_positions()
        self.cfg.grating.enabled = False
        self.cfg.pattern.enabled = True
        self.cfg.pattern.pattern_type = "Column_Range"
        self.cfg.pattern.column_start = 550
        self.cfg.pattern.column_end = 650
        self.cfg.pattern.column_invert = False

        # 1. Angle = 0° (standard vertical column slit):
        # Ray along Y center is unvignetted, ray offset in X outside [550, 650] is vignetted
        self.cfg.pattern.pattern_angle_deg = 0.0
        r_vert_on = sys.trace_ray(Ray(pos=[0.0, 1.0, 0.0], dir=[0, 0, 1], wavelength=1.225))
        r_horiz_off = sys.trace_ray(Ray(pos=[2.5, 0.0, 0.0], dir=[0, 0, 1], wavelength=1.225))
        self.assertFalse(r_vert_on.vignetted, "0° slit: center ray must reflect ON")
        self.assertTrue(r_horiz_off.vignetted, "0° slit: horizontal offset ray must be vignetted into dump")

        # 2. Angle = 90° (horizontal row slit):
        # Ray along X center is unvignetted, ray offset in Y is vignetted
        self.cfg.pattern.pattern_angle_deg = 90.0
        r_horiz_on = sys.trace_ray(Ray(pos=[2.0, 0.0, 0.0], dir=[0, 0, 1], wavelength=1.225))
        r_vert_off = sys.trace_ray(Ray(pos=[0.0, 2.5, 0.0], dir=[0, 0, 1], wavelength=1.225))
        self.assertFalse(r_horiz_on.vignetted, "90° slit: ray along horizontal row must reflect ON")
        self.assertTrue(r_vert_off.vignetted, "90° slit: ray offset vertically in Y must be vignetted into dump")

        # 3. Angle = -45° (diagonal slit):
        # Ray along diagonal (x=1.0, y=1.0) is unvignetted, ray orthogonal to diagonal (x=2.0, y=-2.0) is vignetted
        self.cfg.pattern.pattern_angle_deg = -45.0
        r_diag_on = sys.trace_ray(Ray(pos=[1.0, 1.0, 0.0], dir=[0, 0, 1], wavelength=1.225))
        r_diag_off = sys.trace_ray(Ray(pos=[2.0, -2.0, 0.0], dir=[0, 0, 1], wavelength=1.225))
        self.assertFalse(r_diag_on.vignetted, "-45° diagonal slit: ray along diagonal must reflect ON")
        self.assertTrue(r_diag_off.vignetted, "-45° diagonal slit: ray orthogonal to diagonal must be vignetted into dump")

    def test_plm_studio_widget_rotation_presets_and_signals(self):
        """Verify PLMStudioWidget rotation presets (-45°, 90°, 0°, +45°) for both panel clocking and pattern slit."""
        widget = PLMStudioWidget(surface_stack=self.stack)
        if widget.combo_surface.count() > 1:
            widget.combo_surface.setCurrentIndex(1)

        # Track signals
        emitted_indices = []
        widget.pattern_applied.connect(lambda idx: emitted_indices.append(idx))

        # 1. Panel rotation presets
        widget.btn_panel_rot_m45.click()
        self.assertEqual(widget.spin_panel_rotation.value(), -45.0)
        self.assertEqual(self.s_dmd.tilt_z, -45.0)
        self.assertEqual(self.s_dmd.plm_config.panel_rotation_deg, -45.0)
        self.assertIn(1, emitted_indices)

        emitted_indices.clear()
        widget.btn_panel_rot_90.click()
        self.assertEqual(widget.spin_panel_rotation.value(), 90.0)
        self.assertEqual(self.s_dmd.tilt_z, 90.0)
        self.assertEqual(self.s_dmd.plm_config.panel_rotation_deg, 90.0)
        self.assertIn(1, emitted_indices)

        widget.btn_panel_rot_0.click()
        self.assertEqual(widget.spin_panel_rotation.value(), 0.0)
        self.assertEqual(self.s_dmd.tilt_z, 0.0)

        widget.btn_panel_rot_p45.click()
        self.assertEqual(widget.spin_panel_rotation.value(), 45.0)
        self.assertEqual(self.s_dmd.tilt_z, 45.0)

        # 2. Slit / Pattern angle presets
        widget.combo_pattern_type.setCurrentText("Column_Range")
        self.assertTrue(self.s_dmd.plm_config.pattern.enabled)
        emitted_indices.clear()
        widget._set_pattern_angle(-45.0)
        self.assertEqual(widget.spin_pattern_angle.value(), -45.0)
        self.assertEqual(self.s_dmd.plm_config.pattern.pattern_angle_deg, -45.0)
        self.assertIn(1, emitted_indices)
        self.assertIn("@ -45.0°", widget.lbl_col_width_info.text())

        emitted_indices.clear()
        widget._set_pattern_angle(90.0)
        self.assertEqual(widget.spin_pattern_angle.value(), 90.0)
        self.assertEqual(self.s_dmd.plm_config.pattern.pattern_angle_deg, 90.0)
        self.assertIn(1, emitted_indices)
        self.assertIn("@ +90.0°", widget.lbl_col_width_info.text())

        # 3. DualViewWidget annotation displays rotated slit angle
        dv = DualViewWidget()
        dv.set_xy_front_view_active(True)
        sys = OpticalSystem(self.stack)
        rays = [sys.trace_ray(Ray(pos=[0.0, 0.0, 0.0], dir=[0.0, 0.0, 1.0], wavelength=1.225))]
        dv.update_layout(surface_stack=self.stack, rays=rays)
        front_texts = [t.get_text() for t in dv.ax_xy_front.texts]
        self.assertTrue(any("@ +90°" in t for t in front_texts), f"Front view missing angle annotation: {front_texts}")

    def test_dmd_multi_range_specific_states_and_ray_routing(self):
        """Verify setting discrete states (0=OFF, 1=ON) across different column ranges on DMD."""
        sys = OpticalSystem(self.stack)
        self.s_dmd.mechanical_state = "ON"
        self.s_dmd.tilt_z = 0.0
        self.stack._update_accumulated_positions()
        self.cfg.grating.enabled = False
        self.cfg.pattern.enabled = True
        self.cfg.pattern.pattern_type = "Column_Range"
        self.cfg.pattern.column_background_state = 0

        # Configure 4 distinct column ranges:
        # Range 1: 0..200 (state 0 -> OFF / dump)
        # Range 2: 201..500 (state 1 -> ON / reflect)
        # Range 3: 501..800 (state 0 -> OFF / dump)
        # Range 4: 801..1139 (state 1 -> ON / reflect)
        self.cfg.pattern.column_ranges = [
            {"start": 0, "end": 200, "state": 0},
            {"start": 201, "end": 500, "state": 1},
            {"start": 501, "end": 800, "state": 0},
            {"start": 801, "end": 1139, "state": 1},
        ]

        # DLP4500: 1140 cols across 8.664 mm -> width span is [-4.332, +4.332] mm
        # Ray 1 hitting Range 1 (col ~100, x ~ -3.57 mm): must be vignetted into dump
        r1 = sys.trace_ray(Ray(pos=[-3.57, 0.0, 0.0], dir=[0, 0, 1], wavelength=1.225))
        self.assertTrue(r1.vignetted, "Ray in Range 1 (State 0 / OFF) must be dumped")
        self.assertEqual(r1.intensity, 0.0)

        # Ray 2 hitting Range 2 (col ~350, x ~ -1.67 mm): must reflect ON
        r2 = sys.trace_ray(Ray(pos=[-1.67, 0.0, 0.0], dir=[0, 0, 1], wavelength=1.225))
        self.assertFalse(r2.vignetted, "Ray in Range 2 (State 1 / ON) must reflect along detector channel")
        self.assertGreater(r2.intensity, 0.0)

        # Ray 3 hitting Range 3 (col ~650, x ~ +0.61 mm): must be vignetted into dump
        r3 = sys.trace_ray(Ray(pos=[0.61, 0.0, 0.0], dir=[0, 0, 1], wavelength=1.225))
        self.assertTrue(r3.vignetted, "Ray in Range 3 (State 0 / OFF) must be dumped")
        self.assertEqual(r3.intensity, 0.0)

        # Ray 4 hitting Range 4 (col ~950, x ~ +2.89 mm): must reflect ON
        r4 = sys.trace_ray(Ray(pos=[2.89, 0.0, 0.0], dir=[0, 0, 1], wavelength=1.225))
        self.assertFalse(r4.vignetted, "Ray in Range 4 (State 1 / ON) must reflect along detector channel")
        self.assertGreater(r4.intensity, 0.0)

    def test_plm_multi_range_4bit_discrete_states_and_phase_opd(self):
        """Verify placing specific 4-bit discrete states (0, 1, 2, ..., 15) on column ranges on PLM."""
        from optics_app.engine.plm import evaluate_plm_phase_and_gradient
        s_plm = Surface(name="TI_PLM", surface_type=SurfaceType.MIRROR, thickness=50.0)
        s_plm.is_plm = True
        s_plm.semi_diameter = 10.0

        p_cfg = ensure_plm_config(s_plm)
        p_cfg.modulator_type = "PLM"
        p_cfg.bit_depth = 4
        p_cfg.grating.enabled = False
        p_cfg.zernike.enabled = False
        p_cfg.pattern.enabled = True
        p_cfg.pattern.pattern_type = "Column_Range"
        p_cfg.pattern.column_background_state = 0

        # Configure 5 custom ranges with states 0, 2, 5, 10, 15:
        p_cfg.pattern.column_ranges = [
            {"start": 0, "end": 200, "state": 0},
            {"start": 201, "end": 400, "state": 2},
            {"start": 401, "end": 600, "state": 5},
            {"start": 601, "end": 900, "state": 10},
            {"start": 901, "end": 1357, "state": 15},
        ]

        # Evaluate at coordinates corresponding to each range:
        # Range 1 (col 100): State 0 -> 0.0 waves OPD
        x1 = (100.5 / 1358.0 * 2.0 - 1.0) * 7.3332
        ev1 = evaluate_plm_phase_and_gradient(x1, 0.0, p_cfg, "ON")
        self.assertEqual(ev1["state_4bit"], 0)
        self.assertAlmostEqual(ev1["quantized_waves"], 0.0, places=4)

        # Range 2 (col 300): State 2 -> 2/16 waves OPD (0.125 waves)
        x2 = (300.5 / 1358.0 * 2.0 - 1.0) * 7.3332
        ev2 = evaluate_plm_phase_and_gradient(x2, 0.0, p_cfg, "ON")
        self.assertEqual(ev2["state_4bit"], 2)
        self.assertAlmostEqual(ev2["quantized_waves"], 2.0 / 16.0, places=4)

        # Range 3 (col 500): State 5 -> 5/16 waves OPD (0.3125 waves)
        x3 = (500.5 / 1358.0 * 2.0 - 1.0) * 7.3332
        ev3 = evaluate_plm_phase_and_gradient(x3, 0.0, p_cfg, "ON")
        self.assertEqual(ev3["state_4bit"], 5)
        self.assertAlmostEqual(ev3["quantized_waves"], 5.0 / 16.0, places=4)

        # Range 4 (col 750): State 10 -> 10/16 waves OPD (0.625 waves)
        x4 = (750.5 / 1358.0 * 2.0 - 1.0) * 7.3332
        ev4 = evaluate_plm_phase_and_gradient(x4, 0.0, p_cfg, "ON")
        self.assertEqual(ev4["state_4bit"], 10)
        self.assertAlmostEqual(ev4["quantized_waves"], 10.0 / 16.0, places=4)

        # Range 5 (col 1100): State 15 -> 15/16 waves OPD (0.9375 waves)
        x5 = (1100.5 / 1358.0 * 2.0 - 1.0) * 7.3332
        ev5 = evaluate_plm_phase_and_gradient(x5, 0.0, p_cfg, "ON")
        self.assertEqual(ev5["state_4bit"], 15)
        self.assertAlmostEqual(ev5["quantized_waves"], 15.0 / 16.0, places=4)

    def test_plm_studio_widget_multi_range_table_and_presets(self):
        """Verify PLMStudioWidget multi-range controls: adding, editing, presets, and clearing ranges."""
        widget = PLMStudioWidget(surface_stack=self.stack)
        if widget.combo_surface.count() > 1:
            widget.combo_surface.setCurrentIndex(1)
        widget.combo_modulator_type.setCurrentIndex(1)
        widget.combo_pattern_type.setCurrentText("Column_Range")

        # 1. Add range 1: cols 0..200, state 0
        widget.spin_col_start.setValue(0)
        widget.spin_col_end.setValue(200)
        widget.spin_col_state.setValue(0)
        widget.btn_add_range.click()
        self.assertEqual(len(widget._active_column_ranges), 1)
        self.assertEqual(widget.table_col_ranges.rowCount(), 1)

        # 2. Add range 2: cols 201..500, state 1
        widget.spin_col_start.setValue(201)
        widget.spin_col_end.setValue(500)
        widget.spin_col_state.setValue(1)
        widget.btn_add_range.click()
        self.assertEqual(len(widget._active_column_ranges), 2)
        self.assertEqual(widget.table_col_ranges.rowCount(), 2)

        # 3. Add range 3: cols 501..800, state 0
        widget.spin_col_start.setValue(501)
        widget.spin_col_end.setValue(800)
        widget.spin_col_state.setValue(0)
        widget.btn_add_range.click()
        self.assertEqual(len(widget._active_column_ranges), 3)

        # Verify surface config synchronized
        self.assertEqual(len(self.s_dmd.plm_config.pattern.column_ranges), 3)
        self.assertEqual(self.s_dmd.plm_config.pattern.column_ranges[1]["state"], 1)

        # 4. Click row in table to load values into spinboxes
        widget._on_table_range_clicked(1, 0)
        self.assertEqual(widget.spin_col_start.value(), 201)
        self.assertEqual(widget.spin_col_end.value(), 500)
        self.assertEqual(widget.spin_col_state.value(), 1)

        # 5. Staircase preset in PLM mode
        widget.combo_modulator_type.setCurrentIndex(0)  # PLM mode (4-bit)
        widget._set_staircase_preset()
        self.assertEqual(len(widget._active_column_ranges), 16)
        self.assertEqual(widget.table_col_ranges.rowCount(), 16)
        self.assertEqual(widget._active_column_ranges[0]["state"], 0)
        self.assertEqual(widget._active_column_ranges[15]["state"], 15)

        # 6. Clear Ranges button
        widget.btn_clear_ranges.click()
        self.assertEqual(len(widget._active_column_ranges), 0)
        self.assertEqual(widget.table_col_ranges.rowCount(), 0)

    def test_column_range_has_no_grating_stripes_by_default(self):
        """Verify that column range patterns do not have high-frequency grating stripes superimposed."""
        from optics_app.engine.plm import generate_plm_2d_maps

        widget = PLMStudioWidget(surface_stack=self.stack)
        if widget.combo_surface.count() > 1:
            widget.combo_surface.setCurrentIndex(1)
        widget.combo_modulator_type.setCurrentIndex(1)  # DMD mode

        # Activate Column Range preset (50 columns)
        widget._set_column_preset(50)
        # Carrier grating must be False by default
        self.assertFalse(widget.chk_grating_enable.isChecked())
        self.assertFalse(widget.chk_col_carrier_grating.isChecked())
        self.assertFalse(self.s_dmd.plm_config.grating.enabled)

        # Generate 2D map: verify that inside the active columns, all rows are uniformly ON (no zebra stripes)
        maps = generate_plm_2d_maps(self.s_dmd.plm_config, mechanical_state="ON", nx=100, ny=50)
        state_map = maps["state_4bit"]  # shape (50, 100)
        # Find any column that is ON (at center)
        mid_col = state_map.shape[1] // 2
        col_slice = state_map[:, mid_col]
        # In a pure column slit, the entire vertical column must be ON (1) for every row!
        self.assertTrue(np.all(col_slice == 1), "Column slit must be completely solid ON without horizontal grating stripes!")

        # Also test with grating explicitly enabled: binary mask still remains solid column
        self.s_dmd.plm_config.grating.enabled = True
        self.s_dmd.plm_config.grating.lines_per_mm = 50.0
        maps_grat = generate_plm_2d_maps(self.s_dmd.plm_config, mechanical_state="ON", nx=100, ny=50)
        col_slice_grat = maps_grat["state_4bit"][:, mid_col]
        self.assertTrue(np.all(col_slice_grat == 1), "Even with grating enabled, column mask must not be gated into Ronchi stripes!")


if __name__ == "__main__":
    unittest.main()
