import os
import sys
import json
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v2.engine.tracer import OpticalSystem
from v2.engine.surface import SurfaceType
from v2.engine.ray import Ray
from teamwork_refractive_1064.check_spot import verify_spot_radius
from teamwork_refractive_1064.check_collisions import verify_collisions

LAYOUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'spectrometer_refractive_1064.json')

class TestRefractive1064Spectrometer(unittest.TestCase):
    """
    Test suite for 3D-Folded Refractive 1064 nm Spectrometer.
    Validates optical schema, design constraints, spot radius (<= 0.4 mm),
    100% spectral throughput across all design wavelengths, and 3D collision clearance.
    """

    def setUp(self):
        self.assertTrue(os.path.exists(LAYOUT_FILE), f"Layout file {LAYOUT_FILE} must exist")
        with open(LAYOUT_FILE, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.sys_opt = OpticalSystem.from_json(LAYOUT_FILE)
        self.sys_opt._update_accumulated_positions()

    def test_r1_optical_schema_and_thorlabs_components(self):
        """Verify layout schema version 3, standard Thorlabs refractive lenses, grating, and DMD specifications."""
        self.assertEqual(self.data.get('version'), 3)
        self.assertIn('surfaces', self.data)
        self.assertGreaterEqual(len(self.data['surfaces']), 12)
        
        # Check grating specifications: 600 l/mm, -45 deg tilt, reflective
        grating_surfs = [s for s in self.sys_opt.surfaces if s.surface_type == SurfaceType.GRATING or getattr(s, 'grating_lines_per_mm', 0) > 0]
        self.assertEqual(len(grating_surfs), 1, "Must contain exactly one grating surface")
        grating = grating_surfs[0]
        self.assertEqual(grating.grating_lines_per_mm, 600.0)
        self.assertEqual(grating.tilt_x, -45.0)
        self.assertEqual(grating.glass.upper(), "MIRROR")

        # Check DMD specification: +/- 12 deg tilt, active area <= 5.0 mm semi-diameter
        dmd_surfs = [s for s in self.sys_opt.surfaces if 'DMD' in s.name.upper() and not s.is_coord_break]
        self.assertEqual(len(dmd_surfs), 1, "Must contain exactly one DMD mirror surface")
        dmd = dmd_surfs[0]
        self.assertEqual(abs(dmd.tilt_y), 12.0)
        self.assertEqual(dmd.glass.upper(), "MIRROR")
        self.assertLessEqual(dmd.semi_diameter, 5.0)

        # Check Out-of-Plane fold coordinate break
        cb_before_grat = next((s for s in self.sys_opt.surfaces if 'CB_Before_Grating' in s.name), None)
        self.assertIsNotNone(cb_before_grat, "Must contain out-of-plane coordinate break before grating")
        self.assertEqual(cb_before_grat.tilt_z, 90.0)

        # Check Detector: 0.8 mm diameter -> semi-diameter 0.40 mm
        det_surf = self.sys_opt.surfaces[-1]
        self.assertTrue(det_surf.surface_type == SurfaceType.DETECTOR or 'DETECTOR' in det_surf.name.upper())
        self.assertAlmostEqual(det_surf.semi_diameter, 0.40, places=3)

    def test_r2_dmd_dispersion_span(self):
        """Verify dispersed light maps across the active area of the DMD."""
        dmd_idx = next(i for i, s in enumerate(self.sys_opt.surfaces) if 'DMD' in s.name.upper() and not s.is_coord_break)
        wls = self.data.get('wavelengths', [1.087, 1.225, 1.350])
        
        dmd_coords = []
        for wl in wls:
            r = Ray(pos=np.array([0., 0., 0.]), dir=np.array([0., 0., 1.]), wavelength=wl)
            tr = self.sys_opt.trace_ray(r)
            for h in tr.history:
                if h['surface_index'] == dmd_idx:
                    dmd_coords.append((h['x_local'], h['y_local']))
                    break

        self.assertEqual(len(dmd_coords), len(wls))
        span_y = max(pt[1] for pt in dmd_coords) - min(pt[1] for pt in dmd_coords)
        span_x = max(pt[0] for pt in dmd_coords) - min(pt[0] for pt in dmd_coords)
        total_span = np.sqrt(span_x**2 + span_y**2)
        
        self.assertGreater(total_span, 3.0, "Dispersion span must be >= 3.0 mm")
        self.assertLessEqual(total_span, 10.0, "Dispersion span must be within physical DMD dimensions")

    def test_r3_spot_radius_and_spectral_confinement(self):
        """Programmatically verify maximum spot radius for non-vignetted rays is <= 0.40 mm across all design wavelengths."""
        passed, max_r, n_hits, total = verify_spot_radius(LAYOUT_FILE, verbose=False)
        self.assertTrue(passed, f"Max spot radius {max_r:.4f} mm must be <= 0.400 mm and cover all wavelengths")
        self.assertEqual(n_hits, total, "Must have 100% ray throughput (zero vignetting) reaching detector")
        self.assertLessEqual(max_r, 0.400, "Spot radius must be <= 0.400 mm")

    def test_r3_multifield_slit_containment(self):
        """Verify multi-field ray containment across 30 µm slit span (y in [-15 µm, +15 µm])."""
        wls = self.data.get('wavelengths', [1.087, 1.225, 1.350])
        slit_y_positions = [-0.015, -0.0075, 0.0, 0.0075, 0.015]
        
        hits = []
        total_rays = 0
        for y_slit in slit_y_positions:
            for wl in wls:
                for theta in np.linspace(0.001, np.arcsin(0.04), 4):
                    for phi in np.linspace(0, 2*np.pi, 6, endpoint=False):
                        dx = np.sin(theta)*np.cos(phi)
                        dy = np.sin(theta)*np.sin(phi)
                        dz = np.cos(theta)
                        r = Ray(pos=np.array([0., y_slit, 0.]), dir=np.array([dx, dy, dz]), wavelength=wl)
                        tr = self.sys_opt.trace_ray(r)
                        total_rays += 1
                        if not tr.vignetted:
                            h = tr.history[-1]
                            hits.append(np.sqrt(h['x_local']**2 + h['y_local']**2))
                            
        self.assertEqual(len(hits), total_rays, "All multi-field rays must reach detector without vignetting")
        self.assertLessEqual(max(hits), 0.400, "Maximum spot radius across all field points must be <= 0.400 mm")

    def test_r4_3d_collision_clearance(self):
        """Programmatically verify no ray segment intersects or violates the physical radius (+5 mm margin) of unrelated housings."""
        passed, min_dist_to_center, min_margin, n_violations = verify_collisions(LAYOUT_FILE, verbose=False)
        self.assertTrue(passed, f"Collision check failed with {n_violations} violations")
        self.assertEqual(n_violations, 0, "Violations count must be strictly 0")
        self.assertGreater(min_dist_to_center, 0.0, "Distance to housing center must be > 0")
        self.assertGreater(min_margin, 0.0, "Margin clearance must be > 0 mm beyond physical radius + 5mm")

if __name__ == '__main__':
    unittest.main()
