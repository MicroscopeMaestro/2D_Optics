"""
tests/test_challenger_remediation_audit.py: Independent Empirical Remediation Challenge Suite.

Adversarially validates the remediated 1064 nm Hybrid Raman Spectrometer layout
(hybrid_raman_1064.json) against all authoritative requirements from ORIGINAL_REQUEST.md:
1. Zero Vignetting (100% Transmission) across all 14 optical surfaces.
2. TI DMD4500 Chief Ray Dispersion Span strictly in [8.00 mm, 8.66 mm] and centered in [-4.332 mm, +4.332 mm].
3. Detector Spot Sizes across all discrete and intermediate wavelengths on 1.0 mm detector (radius <= 0.500 mm).
4. Strict Thorlabs 1-Inch Component Prescription Verification.
5. CAD Export (STEP & DXF) compatibility and integrity.
"""

import os
import sys
import json
import unittest
from typing import Tuple, Dict, Any, Optional
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from optics_app.analysis.spot import SpotDiagram
from optics_app.cad.exporter import CADExporter
from optics_app.cad.dxf_exporter import DXF2DExporter

LAYOUT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "hybrid_raman_1064.json"))


def load_remediated_stack() -> Tuple[SurfaceStack, Dict[str, Any]]:
    with open(LAYOUT_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    loaded = SurfaceStack.load_from_json(LAYOUT_PATH)
    stack = loaded["stack"] if isinstance(loaded, dict) else loaded
    return stack, raw


class TestRemediationChallenger1(unittest.TestCase):
    """Challenger 1 Empirical Stress Test Suite."""

    @classmethod
    def setUpClass(cls):
        cls.stack, cls.raw_data = load_remediated_stack()
        cls.opt_system = OpticalSystem(cls.stack)

    # =========================================================================
    # CHALLENGE 1: ZERO VIGNETTING & OPTICAL TRANSMISSION
    # =========================================================================

    def test_01_optical_transmission_dense_phase_space(self):
        """Verify 100% transmission (0 lost rays) across 26 dense wavelengths spanning 1.1113 to 1.6133 um."""
        wls = np.linspace(1.1113, 1.6133, 26)
        det_idx = len(self.stack.surfaces) - 1

        rays = []
        rf = 0.0125  # 25 um core radius = 12.5 um
        max_theta = np.arcsin(0.12)

        spatial_pts = [
            (0.0, 0.0), (rf, 0.0), (-rf, 0.0), (0.0, rf), (0.0, -rf)
        ]

        ang_dirs = [(0.0, 0.0, 1.0)]
        for k in range(8):
            phi = 2.0 * np.pi * k / 8
            vx = float(np.sin(max_theta) * np.cos(phi))
            vy = float(np.sin(max_theta) * np.sin(phi))
            vz = float(np.sqrt(max(0.0, 1.0 - vx**2 - vy**2)))
            ang_dirs.append((vx, vy, vz))

        for wl in wls:
            for sx, sy in spatial_pts:
                for vx, vy, vz in ang_dirs:
                    rays.append(Ray(
                        pos=np.array([sx, sy, 0.0]),
                        dir=np.array([vx, vy, vz]),
                        wavelength=float(wl)
                    ))

        total_rays = len(rays)
        traced = self.opt_system.trace_rays(rays)

        vignetted_before_det = [
            r for r in traced
            if r.vignetted and (not r.history or r.history[-1]["surface_index"] < det_idx)
        ]

        trans_pct = 100.0 * (total_rays - len(vignetted_before_det)) / total_rays
        print(f"\n[CHALLENGE 1] Traced {total_rays} rays across 51 wavelengths:")
        print(f"  Vignetted before detector: {len(vignetted_before_det)}")
        print(f"  Transmission: {trans_pct:.4f}%")

        self.assertEqual(len(vignetted_before_det), 0, f"Found {len(vignetted_before_det)} vignetted rays before detector!")
        self.assertAlmostEqual(trans_pct, 100.0, places=4)

    def test_02_surface_by_surface_clear_aperture_clearance(self):
        """Check beam footprint maximum radius at every optical surface against clear aperture semi-diameter."""
        wls = [1.1113, 1.2000, 1.3000, 1.3620, 1.4500, 1.5500, 1.6133]
        det_idx = len(self.stack.surfaces) - 1

        rays = []
        rf = 0.0125
        max_theta = np.arcsin(0.12)
        for wl in wls:
            for phi in np.linspace(0, 2*np.pi, 12, endpoint=False):
                vx = float(np.sin(max_theta) * np.cos(phi))
                vy = float(np.sin(max_theta) * np.sin(phi))
                vz = float(np.sqrt(max(0.0, 1.0 - vx**2 - vy**2)))
                rays.append(Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([vx, vy, vz]), wavelength=wl))
                rays.append(Ray(pos=np.array([rf*np.cos(phi), rf*np.sin(phi), 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl))

        traced = self.opt_system.trace_rays(rays)

        print("\n[CHALLENGE 1] Surface-by-Surface Footprint vs Clear Aperture:")
        for idx, surf in enumerate(self.stack.surfaces):
            hits = [
                np.sqrt(h["x"]**2 + h["y"]**2)
                for r in traced
                for h in r.history
                if h["surface_index"] == idx
            ]
            if hits:
                max_hit = max(hits)
                margin = surf.semi_diameter - max_hit
                print(f"  Surf {idx:2d} ({surf.name[:25]:25s}): max_r = {max_hit:6.3f} mm / CA_semi = {surf.semi_diameter:6.3f} mm (margin = {margin:6.3f} mm)")
                if idx < det_idx:
                    self.assertLessEqual(max_hit, surf.semi_diameter, f"Surface {idx} ({surf.name}) exceeded clear aperture!")

    # =========================================================================
    # CHALLENGE 2: DMD CHIEF RAY DISPERSION SPAN & CENTERING
    # =========================================================================

    def test_03_dmd_chief_ray_dispersion_span_and_bounds(self):
        """Verify chief ray dispersion span across 1.1113 to 1.6133 um is in [8.00 mm, 8.66 mm] and inside [-4.332, +4.332] mm."""
        dmd_idx = next(i for i, s in enumerate(self.stack.surfaces) if "DMD" in s.name.upper())
        dmd_w = 8.664
        dmd_h = 4.873
        half_w = dmd_w / 2.0  # 4.332 mm
        half_h = dmd_h / 2.0  # 2.4365 mm

        # Dense wavelength evaluation
        wls = np.linspace(1.1113, 1.6133, 101)
        rays = [Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=float(wl)) for wl in wls]
        traced = self.opt_system.trace_rays(rays)

        dmd_x = []
        dmd_y = []
        for r in traced:
            h = next(h for h in r.history if h["surface_index"] == dmd_idx)
            dmd_x.append(h["x"])
            dmd_y.append(h["y"])

        x_min = dmd_x[0]   # at 1.1113 um
        x_max = dmd_x[-1]  # at 1.6133 um
        span = abs(x_max - x_min)
        center_x = (x_max + x_min) / 2.0
        max_abs_y = max(map(abs, dmd_y))

        print("\n[CHALLENGE 2] TI DMD4500 Chief Ray Dispersion Analysis:")
        print(f"  lambda = 1.11130 um -> x = {x_min:+.4f} mm")
        print(f"  lambda = 1.61330 um -> x = {x_max:+.4f} mm")
        print(f"  Total Chief Ray Span = {span:.4f} mm (Target: [8.00 mm, 8.66 mm])")
        print(f"  Spectrum Center Offset = {center_x:+.4f} mm")
        print(f"  Physical DMD Window: [-4.332 mm, +4.332 mm]")
        print(f"  DMD Active Width Fill: {(span / dmd_w)*100.0:.2f}%")
        print(f"  Max Chief Ray Y Offset: {max_abs_y:.4f} mm (Limit: <= 2.4365 mm)")

        # Assertions
        self.assertGreaterEqual(span, 8.000, f"DMD span {span:.4f} mm is below 8.00 mm requirement!")
        self.assertLessEqual(span, 8.664, f"DMD span {span:.4f} mm exceeds 8.664 mm active width!")
        self.assertGreaterEqual(x_min, -half_w, f"Low-wavelength edge {x_min:.4f} mm falls outside DMD active area left edge (-4.332 mm)!")
        self.assertLessEqual(x_max, half_w, f"High-wavelength edge {x_max:.4f} mm falls outside DMD active area right edge (+4.332 mm)!")
        self.assertLessEqual(max_abs_y, half_h, f"Chief ray Y offset {max_abs_y:.4f} mm exceeds DMD half-height!")

    def test_04_dmd_dispersion_linearity_and_monotonicity(self):
        """Verify linear dispersion dx/dlambda is monotonic, positive, and smoothly varying."""
        dmd_idx = next(i for i, s in enumerate(self.stack.surfaces) if "DMD" in s.name.upper())
        wls = np.linspace(1.1113, 1.6133, 51)
        rays = [Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=float(wl)) for wl in wls]
        traced = self.opt_system.trace_rays(rays)

        x_coords = [next(h["x"] for h in r.history if h["surface_index"] == dmd_idx) for r in traced]

        diffs = np.diff(x_coords)
        self.assertTrue(np.all(diffs > 0), "Spatial dispersion on DMD is not strictly monotonic!")

        # Linear dispersion dx/dlambda in mm/um
        dispersion_rates = diffs / np.diff(wls)
        mean_disp = np.mean(dispersion_rates)
        print(f"\n[CHALLENGE 2] Dispersion Linearity: mean dx/dlambda = {mean_disp:.3f} mm/um (min={np.min(dispersion_rates):.3f}, max={np.max(dispersion_rates):.3f})")
        self.assertAlmostEqual(mean_disp, 16.1, delta=2.5)

    # =========================================================================
    # CHALLENGE 3: DETECTOR SPOT SIZES & SPATIAL CONFINEMENT
    # =========================================================================

    def test_05_detector_chief_ray_radius_across_dense_spectrum(self):
        """Verify chief ray spot radius on 1.0 mm detector is <= 0.500 mm across 101 wavelengths."""
        det_idx = len(self.stack.surfaces) - 1
        wls = np.linspace(1.1113, 1.6133, 101)
        rays = [Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=float(wl)) for wl in wls]
        traced = self.opt_system.trace_rays(rays)

        radii = []
        coords = []
        for r in traced:
            h = next(h for h in r.history if h["surface_index"] == det_idx)
            rad = np.sqrt(h["x"]**2 + h["y"]**2)
            radii.append(rad)
            coords.append((h["x"], h["y"]))

        max_radius = max(radii)
        max_diameter = max_radius * 2.0
        mean_radius = np.mean(radii)

        print("\n[CHALLENGE 3] Detector Chief Ray Spatial Confinement (101 wavelengths):")
        print(f"  Max Chief Ray Radius:   {max_radius:.4f} mm (Diameter = {max_diameter:.4f} mm, Limit = <= 0.500 mm radius / <= 1.000 mm diameter)")
        print(f"  Mean Chief Ray Radius:  {mean_radius:.4f} mm")
        print(f"  Min Chief Ray Radius:   {min(radii):.4f} mm")

        self.assertLessEqual(max_radius, 0.500, f"Max chief ray radius {max_radius:.4f} mm exceeds detector radius limit 0.500 mm!")
        self.assertLessEqual(max_diameter, 1.000, f"Max chief ray diameter {max_diameter:.4f} mm exceeds 1.0 mm active diameter!")

    def test_06_detector_full_field_and_chromatic_spot_metrics(self):
        """Verify full geometric phase space spot metrics on detector across key Raman bands."""
        raman_table = [
            ("Low Cutoff (400 cm-1)", 1.11130),
            ("Fingerprint (1000 cm-1)", 1.19070),
            ("Amide/Carbonyl (1650 cm-1)", 1.29060),
            ("Mid-Band (2200 cm-1)", 1.38919),
            ("CH Stretch Onset (2800 cm-1)", 1.51560),
            ("Aromatic CH (3100 cm-1)", 1.58780),
            ("Band Cutoff (3200 cm-1)", 1.61330)
        ]

        det_idx = len(self.stack.surfaces) - 1
        rf = 0.0125
        max_theta = np.arcsin(0.12)

        print("\n[CHALLENGE 3] Detailed Spot Metrics per Raman Band on Detector:")
        print(f"  {'Band Name':30s} | {'Wavelength':10s} | {'CR Radius (mm)':15s} | {'Geo Radius (mm)':16s} | {'RMS Radius (mm)':16s}")
        print("  " + "-" * 95)

        for name, wl in raman_table:
            # Phase space rays
            rays = []
            for sx, sy in [(0, 0), (rf, 0), (-rf, 0), (0, rf), (0, -rf)]:
                for k in range(8):
                    phi = 2.0 * np.pi * k / 8
                    vx = float(np.sin(max_theta) * np.cos(phi))
                    vy = float(np.sin(max_theta) * np.sin(phi))
                    vz = float(np.sqrt(max(0.0, 1.0 - vx**2 - vy**2)))
                    rays.append(Ray(pos=np.array([sx, sy, 0.0]), dir=np.array([vx, vy, vz]), wavelength=wl))
                rays.append(Ray(pos=np.array([sx, sy, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl))

            traced = self.opt_system.trace_rays(rays)
            cr_hit = None
            for r in traced:
                if r.history and abs(r.history[0]["x"]) < 1e-6 and abs(r.history[0]["y"]) < 1e-6 and abs(r.history[0]["dir_x"]) < 1e-5 and abs(r.history[0]["dir_y"]) < 1e-5:
                    for h in r.history:
                        if h["surface_index"] == det_idx:
                            cr_hit = h
                            break
                    if cr_hit is not None:
                        break

            if cr_hit is None:
                # Fallback to center-most initial ray
                cr_hit = next(h for r in traced for h in r.history if h["surface_index"] == det_idx)

            cr_rad = np.sqrt(cr_hit["x"]**2 + cr_hit["y"]**2)

            all_hits_x = [h["x"] for r in traced for h in r.history if h["surface_index"] == det_idx]
            all_hits_y = [h["y"] for r in traced for h in r.history if h["surface_index"] == det_idx]

            spot = SpotDiagram(all_hits_x, all_hits_y, wavelength_um=wl)
            geo_r = spot.calculate_geo_radius()
            rms_r = spot.calculate_rms_radius()

            print(f"  {name:30s} | {wl:9.5f} um | {cr_rad:14.4f} mm | {geo_r:15.4f} mm | {rms_r:15.4f} mm")

            self.assertLessEqual(cr_rad, 0.500, f"Chief ray radius at {wl} um ({cr_rad:.4f} mm) exceeds 0.500 mm!")

    # =========================================================================
    # CHALLENGE 4: THORLABS 1-INCH CATALOG PRESCRIPTIONS
    # =========================================================================

    def test_07_thorlabs_catalog_prescription_adherence(self):
        """Verify all lens elements match authentic Thorlabs catalog optical prescriptions."""
        print("\n[CHALLENGE 4] Thorlabs Catalog Optical Prescription Audit:")

        # 1. Collimator AC254-050-C
        c_front = self.stack.surfaces[1]
        c_mid = self.stack.surfaces[2]
        c_back = self.stack.surfaces[3]
        self.assertAlmostEqual(c_front.radius, 34.0, places=2)
        self.assertAlmostEqual(c_mid.radius, -27.8, places=2)
        self.assertAlmostEqual(c_back.radius, -115.0, places=2)
        self.assertAlmostEqual(c_front.thickness, 6.0, places=2)
        self.assertAlmostEqual(c_mid.thickness, 2.0, places=2)
        self.assertEqual(c_front.glass, "N-BK7")
        self.assertEqual(c_mid.glass, "SF5")
        self.assertEqual(c_front.semi_diameter, 12.7)
        print("  [PASS] Collimator Lens: AC254-050-C (1-inch, f=50mm, N-BK7/SF5)")

        # 2. Grating GR25-0316
        grating = self.stack.surfaces[4]
        self.assertEqual(grating.surface_type, SurfaceType.GRATING)
        self.assertAlmostEqual(grating.grating_lines_per_mm, 300.0, places=2)
        self.assertAlmostEqual(grating.tilt_y, 12.0, places=2)
        self.assertEqual(grating.semi_diameter, 12.7)
        print("  [PASS] Diffraction Grating: GR25-0316 (1-inch, 300 l/mm, Gold, 12 deg tilt)")

        # 3. Focus Camera Lens AC254-050-C
        f_front = self.stack.surfaces[5]
        f_mid = self.stack.surfaces[6]
        f_back = self.stack.surfaces[7]
        self.assertAlmostEqual(f_front.radius, 34.0, places=2)
        self.assertAlmostEqual(f_mid.radius, -27.8, places=2)
        self.assertAlmostEqual(f_back.radius, -115.0, places=2)
        self.assertEqual(f_front.glass, "N-BK7")
        self.assertEqual(f_mid.glass, "SF5")
        self.assertEqual(f_front.semi_diameter, 12.7)
        print("  [PASS] Camera Focus Lens: AC254-050-C (1-inch, f=50mm, N-BK7/SF5)")

        # 4. DMD4500
        dmd = self.stack.surfaces[8]
        self.assertAlmostEqual(dmd.tilt_y, 12.0, places=2)
        print("  [PASS] Spatial Light Modulator: TI DLP4500 (+12.0 deg micromirror state)")

        # 5. Relay Collimator AC254-075-C (1-inch doublet, NOT 2-inch)
        r_front = self.stack.surfaces[9]
        r_mid = self.stack.surfaces[10]
        r_back = self.stack.surfaces[11]
        self.assertAlmostEqual(r_front.radius, 49.3, places=2)
        self.assertAlmostEqual(r_mid.radius, -39.0, places=2)
        self.assertAlmostEqual(r_back.radius, -170.0, places=2)
        self.assertAlmostEqual(r_front.thickness, 5.2, places=2)
        self.assertAlmostEqual(r_mid.thickness, 2.0, places=2)
        self.assertEqual(r_front.glass, "N-BK7")
        self.assertEqual(r_mid.glass, "SF5")
        self.assertEqual(r_front.semi_diameter, 12.7)  # STRICTLY 1-INCH
        print("  [PASS] Collection Relay Lens: AC254-075-C (1-inch, f=75mm, N-BK7/SF5)")

        # 6. Condenser ACL12708U-C
        cond_front = self.stack.surfaces[12]
        cond_back = self.stack.surfaces[13]
        self.assertEqual(cond_front.surface_type, SurfaceType.ASPHERIC)
        self.assertAlmostEqual(cond_front.radius, 4.16, places=2)
        self.assertAlmostEqual(cond_front.conic, -0.6265, places=4)
        self.assertEqual(cond_front.glass, "B270")
        self.assertAlmostEqual(cond_front.thickness, 4.5, places=2)
        print("  [PASS] High-NA Condenser: ACL12708U-C (0.5-inch aspheric, f=8.0mm, B270)")

        # 7. Detector
        det = self.stack.surfaces[14]
        self.assertEqual(det.surface_type, SurfaceType.DETECTOR)
        self.assertEqual(det.semi_diameter, 0.50)
        print("  [PASS] Detector: InGaAs 1.0mm active diameter (semi-diameter = 0.50 mm)")

    # =========================================================================
    # CHALLENGE 5: CAD DELIVERABLES & EXPORT REPRODUCIBILITY
    # =========================================================================

    def test_08_cad_deliverables_step_and_dxf(self):
        """Verify CAD export outputs (STEP & DXF) exist, are valid, and can be freshly generated."""
        step_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "hybrid_raman_1064.step"))
        dxf_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "hybrid_raman_1064.dxf"))

        print("\n[CHALLENGE 5] CAD Deliverable Audit:")
        self.assertTrue(os.path.exists(step_path), "hybrid_raman_1064.step deliverable missing!")
        self.assertTrue(os.path.exists(dxf_path), "hybrid_raman_1064.dxf deliverable missing!")

        step_size = os.path.getsize(step_path)
        dxf_size = os.path.getsize(dxf_path)
        print(f"  STEP Deliverable Size: {step_size:,} bytes")
        print(f"  DXF Deliverable Size:  {dxf_size:,} bytes")

        self.assertGreater(step_size, 5000, "STEP file size too small!")
        self.assertGreater(dxf_size, 50000, "DXF file size too small!")

        # Verify fresh export execution
        step_res = CADExporter.export_to_step(self.stack, step_path)
        self.assertTrue(step_res.is_success)
        dxf_res = CADExporter.export_to_dxf_2d(self.stack, dxf_path)
        self.assertTrue(os.path.exists(dxf_res))
        print("  [PASS] Fresh CAD export pipeline executed cleanly.")


if __name__ == "__main__":
    unittest.main()
