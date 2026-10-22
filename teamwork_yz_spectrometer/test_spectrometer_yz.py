"""
Comprehensive Test Suite for 2D YZ-Plane Refractive 1064 nm Spectrometer.
Validates:
1. Strict 2D YZ-Plane geometry (all x_global = 0, tilt_y = 0, tilt_z = 0, decenter_x = 0).
2. Exact optical sequence: Collection Optics -> Slit -> Collimator -> Grating (600 l/mm) ->
   Focusing Lens -> DMD (12 deg tilt) -> Relay Doublet -> Condenser -> 0.8 mm Detector.
3. DMD dispersion span across active area.
4. Programmatic spot radius verification (<= 0.400 mm radius, 100% throughput).
5. Multi-field 30 µm slit confinement.
6. 2D YZ collision avoidance and mechanical clearance (+5 mm margin).
"""
import os
import sys
import json
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v2.engine.tracer import OpticalSystem
from v2.engine.surface import SurfaceType
from v2.engine.ray import Ray
from teamwork_yz_spectrometer.check_spot import verify_spot_radius
from teamwork_yz_spectrometer.check_collisions import verify_collisions

LAYOUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'spectrometer_yz_1064.json')


class TestYZSpectrometer(unittest.TestCase):
    def setUp(self):
        self.assertTrue(os.path.exists(LAYOUT_FILE), f"Layout file {LAYOUT_FILE} must exist")
        with open(LAYOUT_FILE, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.sys_opt = OpticalSystem.from_json(LAYOUT_FILE)
        self.sys_opt._update_accumulated_positions()

    def test_r1_strict_yz_plane_geometry(self):
        """Verify the entire optical layout is flat and drawn strictly in the YZ plane (no out-of-plane X folds)."""
        self.assertEqual(self.data.get('version'), 3)
        self.assertIn('surfaces', self.data)

        # 1. Check surface definition parameters: tilt_y=0, tilt_z=0, decenter_x=0
        for s in self.sys_opt.surfaces:
            self.assertAlmostEqual(
                getattr(s, 'tilt_y', 0.0), 0.0, places=5,
                msg=f"Surface {s.name} has non-zero tilt_y: {s.tilt_y}"
            )
            self.assertAlmostEqual(
                getattr(s, 'tilt_z', 0.0), 0.0, places=5,
                msg=f"Surface {s.name} has non-zero tilt_z: {s.tilt_z}"
            )
            self.assertAlmostEqual(
                getattr(s, 'decenter_x', 0.0), 0.0, places=5,
                msg=f"Surface {s.name} has non-zero decenter_x: {s.decenter_x}"
            )
            self.assertAlmostEqual(
                s.global_origin[0], 0.0, places=5,
                msg=f"Surface {s.name} has non-zero global X coordinate: {s.global_origin[0]}"
            )

        # 2. Ray trace in YZ plane: all global X coordinates must be strictly 0.0
        wls = self.data.get('wavelengths', [1.087, 1.225, 1.350])
        for wl in wls:
            r = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl)
            tr = self.sys_opt.trace_ray(r)
            for h in tr.history:
                self.assertAlmostEqual(
                    h['x_global'], 0.0, places=5,
                    msg=f"Ray at surface index {h['surface_index']} has non-zero x_global: {h['x_global']}"
                )
                self.assertAlmostEqual(
                    h['dir_x'], 0.0, places=5,
                    msg=f"Ray direction at surface index {h['surface_index']} has non-zero dir_x: {h['dir_x']}"
                )

    def test_r2_optical_components_and_sequence(self):
        """Verify complete optical component prescription and sequence."""
        surfs = self.sys_opt.surfaces

        # 1. Collection Optics before slit
        collect_surfs = [s for s in surfs if 'COLLECT' in s.name.upper()]
        self.assertGreaterEqual(len(collect_surfs), 2, "Must contain collection optics before slit")

        # 2. Slit (30 µm)
        slit_surfs = [s for s in surfs if 'SLIT' in s.name.upper()]
        self.assertEqual(len(slit_surfs), 1, "Must contain slit surface")

        # 3. Collimator
        col_surfs = [s for s in surfs if 'COLLIMATOR' in s.name.upper()]
        self.assertGreaterEqual(len(col_surfs), 2, "Must contain Collimator lens surfaces")

        # 4. Aperture Stop
        stop_surfs = [s for s in surfs if s.surface_type == SurfaceType.STOP or 'STOP' in s.name.upper()]
        self.assertGreaterEqual(len(stop_surfs), 1, "Must contain Aperture Stop")

        # 5. Grating: 600 l/mm reflective grating
        grating_surfs = [s for s in surfs if s.surface_type == SurfaceType.GRATING or getattr(s, 'grating_lines_per_mm', 0) > 0]
        self.assertEqual(len(grating_surfs), 1, "Must contain exactly one grating surface")
        grating = grating_surfs[0]
        self.assertEqual(grating.grating_lines_per_mm, 600.0)
        self.assertEqual(grating.glass.upper(), "MIRROR")
        self.assertNotEqual(grating.tilt_x, 0.0, "Grating must have non-zero fold tilt_x in YZ plane")

        # 6. Focusing Lens
        foc_surfs = [s for s in surfs if 'FOCUSING' in s.name.upper()]
        self.assertGreaterEqual(len(foc_surfs), 2, "Must contain Focusing lens surfaces")

        # 7. DMD4500: micromirrors tilted at +/- 12 degrees
        dmd_surfs = [s for s in surfs if 'DMD' in s.name.upper() and not s.is_coord_break]
        self.assertEqual(len(dmd_surfs), 1, "Must contain exactly one DMD mirror surface")
        dmd = dmd_surfs[0]
        self.assertAlmostEqual(abs(dmd.tilt_x), 12.0, places=1, msg="DMD must have 12 deg mechanical tilt")
        self.assertEqual(dmd.glass.upper(), "MIRROR")
        self.assertLessEqual(dmd.semi_diameter, 5.0)

        # 8. Relay System: Relay doublet + Condenser
        relay_surfs = [s for s in surfs if 'RELAY' in s.name.upper()]
        self.assertGreaterEqual(len(relay_surfs), 2, "Must contain Relay lens surfaces")

        cond_surfs = [s for s in surfs if 'COND' in s.name.upper()]
        self.assertGreaterEqual(len(cond_surfs), 2, "Must contain Condenser lens surfaces")

        # 9. Detector: 0.8 mm diameter (semi-diameter = 0.40 mm)
        det_surf = surfs[-1]
        self.assertTrue(det_surf.surface_type == SurfaceType.DETECTOR or 'DETECTOR' in det_surf.name.upper())
        self.assertAlmostEqual(det_surf.semi_diameter, 0.40, places=3)

    def test_r2_dmd_dispersion_span(self):
        """Verify dispersed Raman spectrum maps across active area of DMD."""
        dmd_idx = next(i for i, s in enumerate(self.sys_opt.surfaces) if 'DMD' in s.name.upper() and not s.is_coord_break)
        wls = self.data.get('wavelengths', [1.087, 1.225, 1.350])

        dmd_coords = []
        for wl in wls:
            r = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl)
            tr = self.sys_opt.trace_ray(r)
            for h in tr.history:
                if h['surface_index'] == dmd_idx:
                    dmd_coords.append((h['x_local'], h['y_local']))
                    break

        self.assertEqual(len(dmd_coords), len(wls))
        span_y = max(pt[1] for pt in dmd_coords) - min(pt[1] for pt in dmd_coords)
        span_x = max(pt[0] for pt in dmd_coords) - min(pt[0] for pt in dmd_coords)
        total_span = np.sqrt(span_x ** 2 + span_y ** 2)

        self.assertGreater(total_span, 3.0, "Dispersion span must be >= 3.0 mm across DMD active area")
        self.assertLessEqual(total_span, 10.0, "Dispersion span must fit on physical DMD array")

    def test_r3_spot_radius_and_throughput(self):
        """Verify maximum spot radius <= 0.400 mm and 100% throughput across Raman band."""
        passed, max_r, n_hits, total = verify_spot_radius(LAYOUT_FILE, verbose=False)
        self.assertTrue(passed, f"Max spot radius {max_r:.4f} mm exceeds 0.400 mm limit or has vignetting")
        self.assertEqual(n_hits, total, "Must achieve 100% ray throughput (zero vignetting) reaching detector")
        self.assertLessEqual(max_r, 0.400, "Spot radius must be <= 0.400 mm")

    def test_r3_multifield_slit_containment(self):
        """Verify multi-field ray containment across 30 µm slit span (y in [-15 µm, +15 µm])."""
        wls = self.data.get('wavelengths', [1.087, 1.225, 1.350])
        slit_y_positions = [-0.015, -0.0075, 0.0, 0.0075, 0.015]

        hits = []
        total_rays = 0
        vignettes = 0
        for y_slit in slit_y_positions:
            for wl in wls:
                for theta in np.linspace(0.001, np.arcsin(0.04), 4):
                    for phi in np.linspace(0, 2 * np.pi, 6, endpoint=False):
                        dx = np.sin(theta) * np.cos(phi)
                        dy = np.sin(theta) * np.sin(phi)
                        dz = np.cos(theta)
                        r = Ray(pos=np.array([0.0, y_slit, 0.0]), dir=np.array([dx, dy, dz]), wavelength=wl)
                        tr = self.sys_opt.trace_ray(r)
                        total_rays += 1
                        if tr.vignetted:
                            vignettes += 1
                        else:
                            h = tr.history[-1]
                            hits.append(np.sqrt(h['x_local'] ** 2 + h['y_local'] ** 2))

        self.assertEqual(vignettes, 0, "No multi-field rays should be vignetted")
        self.assertEqual(len(hits), total_rays, "All multi-field rays must reach detector")
        self.assertLessEqual(max(hits), 0.400, f"Max spot radius {max(hits):.4f} mm must be <= 0.400 mm across all fields")

    def test_r4_2d_yz_collision_clearance(self):
        """Programmatically verify no ray segment intersects or violates the physical radius (+5 mm margin) of unrelated housings."""
        passed, min_dist_to_center, min_margin, n_violations = verify_collisions(LAYOUT_FILE, verbose=False)
        self.assertTrue(passed, f"Collision check failed with {n_violations} violations (min margin: {min_margin:.2f} mm)")
        self.assertEqual(n_violations, 0, "Violations count must be strictly 0")
        self.assertGreater(min_dist_to_center, 0.0, "Distance to housing center must be > 0")
        self.assertGreater(min_margin, 0.0, "Margin clearance must be > 0 mm beyond physical radius + 5mm")


if __name__ == '__main__':
    unittest.main()
