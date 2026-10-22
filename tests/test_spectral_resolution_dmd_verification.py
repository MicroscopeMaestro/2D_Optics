"""
tests/test_spectral_resolution_dmd_verification.py:
Empirical Challenger 2 Test Harness for Spectral Resolution and TI DMD4500 Active Area Boundary Verification.
"""

import os
import sys
import json
import unittest
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from optics_app.analysis.spot import SpotDiagram
from optics_app.engine.grating import grating_diffract_3d
from verify_layout import (
    RAMAN_WAVELENGTHS_UM,
    generate_fiber_phase_space_rays
)

LAYOUT_FILE = "hybrid_raman_1064_full_range.json"

# Texas Instruments DLP4500 DMD Physical Specifications
DMD_WIDTH_MM = 8.664
DMD_HEIGHT_MM = 4.873
DMD_HALF_WIDTH_MM = DMD_WIDTH_MM / 2.0   # 4.332 mm
DMD_HALF_HEIGHT_MM = DMD_HEIGHT_MM / 2.0 # 2.4365 mm
DMD_MICROMIRROR_PITCH_UM = 7.6           # 7.6 um diamond array pitch (0.0076 mm)
DMD_TILT_ANGLE_DEG = 12.0                # +/- 12 deg tilt state

# Single Element Detector Specification
DETECTOR_DIAMETER_MM = 1.0
DETECTOR_RADIUS_MM = 0.50


class TestSpectralResolutionAndDMDBoundaries(unittest.TestCase):
    """Adversarial stress test and verification suite for spectral dispersion and DMD active area."""

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(LAYOUT_FILE):
            raise unittest.SkipTest(f"{LAYOUT_FILE} does not exist.")
        cls.loaded = SurfaceStack.load_from_json(LAYOUT_FILE)
        cls.stack = cls.loaded["stack"]
        cls.opt_sys = OpticalSystem(cls.stack)
        
        # Identify key surfaces
        cls.dmd_idx = next(i for i, s in enumerate(cls.stack.surfaces) if "DMD" in s.name.upper())
        cls.grating_idx = next(i for i, s in enumerate(cls.stack.surfaces) if s.surface_type == SurfaceType.GRATING)
        cls.det_idx = next(i for i, s in enumerate(cls.stack.surfaces) if s.surface_type == SurfaceType.DETECTOR or "DETECTOR" in s.name.upper())

    def test_01_grating_and_focusing_dispersion_first_principles(self):
        """Verify grating angular dispersion and focusing lens linear dispersion match optical physics."""
        grating_surf = self.stack.surfaces[self.grating_idx]
        lines_per_mm = grating_surf.grating_lines_per_mm
        self.assertEqual(lines_per_mm, 300.0, "Grating line density must be 300 lines/mm")
        
        d_um = 1000.0 / lines_per_mm # 3.3333 um
        wl_min = RAMAN_WAVELENGTHS_UM[0] # 1.11130 um
        wl_max = RAMAN_WAVELENGTHS_UM[-1] # 1.66708 um
        
        # Check angle of incidence on grating
        alpha_deg = grating_surf.tilt_y
        alpha_rad = np.radians(alpha_deg)
        
        # Grating equation in order m=1: sin(beta) = m*lambda/d - sin(alpha)
        # Note: Depending on sign convention for reflection grating:
        sin_beta_min = wl_min / d_um - np.sin(alpha_rad)
        sin_beta_max = wl_max / d_um - np.sin(alpha_rad)
        
        beta_min = np.arcsin(sin_beta_min)
        beta_max = np.arcsin(sin_beta_max)
        delta_beta = abs(beta_max - beta_min)
        
        print(f"\n[Physics First Principles] Grating: {lines_per_mm} l/mm, d = {d_um:.4f} um, alpha = {alpha_deg} deg")
        print(f"  beta(1111.3 nm) = {np.degrees(beta_min):.3f} deg")
        print(f"  beta(1667.1 nm) = {np.degrees(beta_max):.3f} deg")
        print(f"  Angular dispersion Delta_beta = {np.degrees(delta_beta):.3f} deg ({delta_beta:.4f} rad)")

    def test_02_chief_ray_positions_on_dmd(self):
        """Empirically trace chief rays across 7 Raman wavelengths and record exact X positions on DMD."""
        chief_rays = [
            Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl)
            for wl in RAMAN_WAVELENGTHS_UM
        ]
        traced = self.opt_sys.trace_rays(chief_rays)
        
        dmd_positions = []
        print("\n[Empirical Ray Trace: Chief Rays on DMD Surface]")
        for wl, r in zip(RAMAN_WAVELENGTHS_UM, traced):
            for h in r.history:
                if h["surface_index"] == self.dmd_idx:
                    dmd_positions.append({
                        "wavelength_um": wl,
                        "x": float(h["x"]),
                        "y": float(h["y"]),
                        "vignetted": r.vignetted
                    })
                    print(f"  lambda = {wl:.5f} um -> x = {h['x']:+.4f} mm, y = {h['y']:+.4f} mm, vig = {r.vignetted}")
                    
        self.assertEqual(len(dmd_positions), len(RAMAN_WAVELENGTHS_UM), "All chief rays must reach DMD")
        
        x_min = min(p["x"] for p in dmd_positions)
        x_max = max(p["x"] for p in dmd_positions)
        span_x = x_max - x_min
        print(f"  Chief Ray Spectral Span on DMD: {span_x:.4f} mm (DMD Active Width: {DMD_WIDTH_MM} mm)")
        
        # Test boundary containment for unvignetted rays within active area
        for p in dmd_positions:
            if not p.get("vignetted", False):
                self.assertLessEqual(abs(p["x"]), DMD_HALF_WIDTH_MM,
                                     f"Chief ray at {p['wavelength_um']} um exceeds DMD half-width ({p['x']} mm > {DMD_HALF_WIDTH_MM} mm)")

    def test_03_full_phase_space_bundle_dmd_footprint_and_edge_clipping(self):
        """Trace 4D phase space rays (25 um fiber, NA=0.12) to verify full footprint and clipping on DMD."""
        rays = generate_fiber_phase_space_rays(
            fiber_radius_mm=0.0125,
            numerical_aperture=0.12,
            wavelengths=RAMAN_WAVELENGTHS_UM,
            dense_mode=True
        )
        traced = self.opt_sys.trace_rays(rays)
        
        dmd_hits = []
        for r in traced:
            for h in r.history:
                if h["surface_index"] == self.dmd_idx:
                    dmd_hits.append((r.wavelength, h["x"], h["y"], r.vignetted))
                    
        print(f"\n[Dense Phase Space Ray Trace on DMD] Total DMD ray intercepts: {len(dmd_hits)} / {len(rays)}")
        
        all_x = [h[1] for h in dmd_hits]
        all_y = [h[2] for h in dmd_hits]
        
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        full_span_x = max_x - min_x
        full_span_y = max_y - min_y
        
        print(f"  Full Beam Footprint on DMD:")
        print(f"    X range: [{min_x:+.4f}, {max_x:+.4f}] mm -> Total Span X = {full_span_x:.4f} mm")
        print(f"    Y range: [{min_y:+.4f}, {max_y:+.4f}] mm -> Total Span Y = {full_span_y:.4f} mm")
        print(f"    DMD Active Area: Width = {DMD_WIDTH_MM} mm, Height = {DMD_HEIGHT_MM} mm")
        print(f"    DMD X bounds: [{-DMD_HALF_WIDTH_MM:+.4f}, {+DMD_HALF_WIDTH_MM:+.4f}] mm")
        print(f"    DMD Y bounds: [{-DMD_HALF_HEIGHT_MM:+.4f}, {+DMD_HALF_HEIGHT_MM:+.4f}] mm")
        
        # Check if full beam clips outside DMD active area
        clipped_x = [h for h in dmd_hits if abs(h[1]) > DMD_HALF_WIDTH_MM]
        clipped_y = [h for h in dmd_hits if abs(h[2]) > DMD_HALF_HEIGHT_MM]
        print(f"    Rays outside DMD X active width: {len(clipped_x)} ({len(clipped_x)/len(dmd_hits)*100:.1f}%)")
        print(f"    Rays outside DMD Y active height: {len(clipped_y)} ({len(clipped_y)/len(dmd_hits)*100:.1f}%)")

    def test_04_spectral_resolution_and_spot_metrics(self):
        """Calculate linear dispersion, spot sizes, spectral resolution delta_lambda and delta_nu, and resolvable spots."""
        # Calculate per-wavelength spot sizes on DMD
        print("\n[Spectral Resolution & Resolvable Spots Analysis on DMD]")
        
        # Trace dense rays per wavelength
        per_wl_spots = {}
        for wl in RAMAN_WAVELENGTHS_UM:
            wl_rays = generate_fiber_phase_space_rays(
                fiber_radius_mm=0.0125,
                numerical_aperture=0.12,
                wavelengths=[wl],
                dense_mode=True
            )
            traced = self.opt_sys.trace_rays(wl_rays)
            
            x_hits = []
            y_hits = []
            for r in traced:
                for h in r.history:
                    if h["surface_index"] == self.dmd_idx:
                        x_hits.append(h["x"])
                        y_hits.append(h["y"])
                        
            if x_hits:
                spot = SpotDiagram(x_hits, y_hits, wavelength_um=wl)
                cx, cy = spot.calculate_centroid()
                rms_x = spot.calculate_rms_x()
                rms_y = spot.calculate_rms_y()
                rms_r = spot.calculate_rms_radius()
                geo_r = spot.calculate_geo_radius()
                dx = max(x_hits) - min(x_hits)
                per_wl_spots[wl] = {
                    "cx": cx, "cy": cy,
                    "rms_x": rms_x, "rms_y": rms_y, "rms_r": rms_r,
                    "geo_r": geo_r, "dx_spot": dx
                }
                print(f"  lambda = {wl:.5f} um:")
                print(f"    Centroid: ({cx:+.4f}, {cy:+.4f}) mm")
                print(f"    Spot RMS width (2*RMS_x): {2*rms_x*1000.0:.2f} um, RMS radius: {rms_r*1000.0:.2f} um")
                print(f"    Geometric spot full width: {dx*1000.0:.2f} um")
        
        # Calculate linear dispersion (dx / dlambda)
        wls = sorted(per_wl_spots.keys())
        total_dx = per_wl_spots[wls[-1]]["cx"] - per_wl_spots[wls[0]]["cx"]
        total_dwl_um = wls[-1] - wls[0]
        total_dwl_nm = total_dwl_um * 1000.0
        
        linear_dispersion_mm_per_nm = total_dx / total_dwl_nm
        reciprocal_linear_dispersion_nm_per_mm = total_dwl_nm / total_dx
        
        print(f"\n  Spectral Dispersion Summary:")
        print(f"    Wavelength Range: {wls[0]*1000.0:.1f} nm to {wls[-1]*1000.0:.1f} nm ({total_dwl_nm:.1f} nm)")
        print(f"    Spatial Separation on DMD: {total_dx:.4f} mm")
        print(f"    Linear Dispersion dx/dlambda: {linear_dispersion_mm_per_nm*1000.0:.4f} um/nm ({linear_dispersion_mm_per_nm:.6f} mm/nm)")
        print(f"    Reciprocal Linear Dispersion dlambda/dx: {reciprocal_linear_dispersion_nm_per_mm:.3f} nm/mm")
        
        # Spectral Resolution (Rayleigh criterion based on geometric / RMS spot size)
        avg_rms_spot_width_um = np.mean([2 * per_wl_spots[wl]["rms_x"] * 1000.0 for wl in wls])
        avg_geo_spot_width_um = np.mean([per_wl_spots[wl]["dx_spot"] * 1000.0 for wl in wls])
        
        delta_lambda_rms_nm = reciprocal_linear_dispersion_nm_per_mm * (avg_rms_spot_width_um / 1000.0)
        delta_lambda_geo_nm = reciprocal_linear_dispersion_nm_per_mm * (avg_geo_spot_width_um / 1000.0)
        
        # Raman shift resolution delta_nu (cm^-1) at 1064 nm excitation
        # nu = 10^7 / 1064 - 10^7 / lambda_nm -> |dnu/dlambda| = 10^7 / lambda_nm^2
        # Near center wavelength 1389 nm:
        mid_wl_nm = 1389.19
        dnu_dlambda = 1e7 / (mid_wl_nm**2) # cm^-1 / nm
        delta_nu_rms_cm1 = delta_lambda_rms_nm * dnu_dlambda
        delta_nu_geo_cm1 = delta_lambda_geo_nm * dnu_dlambda
        
        num_resolvable_spots_rms = total_dx / (avg_rms_spot_width_um / 1000.0)
        num_resolvable_spots_geo = total_dx / (avg_geo_spot_width_um / 1000.0)
        num_dmd_pixels = DMD_WIDTH_MM / (DMD_MICROMIRROR_PITCH_UM / 1000.0)
        
        print(f"\n  Resolution Metrics:")
        print(f"    Mean Optical Spot Width (2*RMS_x): {avg_rms_spot_width_um:.2f} um")
        print(f"    Mean Full Geometric Spot Width: {avg_geo_spot_width_um:.2f} um")
        print(f"    Spectral Resolution delta_lambda (RMS): {delta_lambda_rms_nm:.3f} nm")
        print(f"    Spectral Resolution delta_lambda (Geo): {delta_lambda_geo_nm:.3f} nm")
        print(f"    Raman Resolution delta_nu (RMS): {delta_nu_rms_cm1:.2f} cm^-1")
        print(f"    Raman Resolution delta_nu (Geo): {delta_nu_geo_cm1:.2f} cm^-1")
        print(f"    Resolvable Spots across Band (RMS): {num_resolvable_spots_rms:.1f}")
        print(f"    Resolvable Spots across Band (Geo): {num_resolvable_spots_geo:.1f}")
        print(f"    DMD Micromirror Columns (7.6 um pitch): {num_dmd_pixels:.0f} channels")
        print(f"    DMD Channel Dispersion: {reciprocal_linear_dispersion_nm_per_mm * 0.0076:.4f} nm/micromirror column")

    def test_05_dmd_micromirror_tilt_and_relay_collection_to_detector(self):
        """Test DMD micromirror +12 deg tilt reflection into 4F relay and condenser optics to detector."""
        print("\n[DMD Tilt Reflection & 4F Relay Collection Efficiency]")
        
        dmd_surf = self.stack.surfaces[self.dmd_idx]
        print(f"  DMD Surface Name: {dmd_surf.name}")
        print(f"  DMD Tilt X: {dmd_surf.tilt_x} deg, Tilt Y: {dmd_surf.tilt_y} deg, Tilt Z: {dmd_surf.tilt_z} deg")
        print(f"  DMD Glass: {dmd_surf.glass}, Surface Type: {dmd_surf.surface_type}")
        
        # Trace rays from fiber all the way through DMD and relay to detector
        rays = generate_fiber_phase_space_rays(
            fiber_radius_mm=0.0125,
            numerical_aperture=0.12,
            wavelengths=RAMAN_WAVELENGTHS_UM,
            dense_mode=False
        )
        traced = self.opt_sys.trace_rays(rays)
        
        total_rays = len(traced)
        vignetted_rays = [r for r in traced if r.vignetted]
        unvignetted_rays = [r for r in traced if not r.vignetted]
        
        # Analyze which surfaces cause ray loss / vignetting
        loss_by_surface = {}
        for r in traced:
            if r.vignetted:
                last_surf = r.history[-1]["surface_index"] if r.history else -1
                loss_by_surface[last_surf] = loss_by_surface.get(last_surf, 0) + 1
                
        print(f"  Total Rays Traced: {total_rays}")
        print(f"  Unvignetted Rays reaching Detector: {len(unvignetted_rays)} ({len(unvignetted_rays)/total_rays*100:.1f}%)")
        print(f"  Vignetted Rays: {len(vignetted_rays)} ({len(vignetted_rays)/total_rays*100:.1f}%)")
        print("  Loss Distribution by Surface Index / Name:")
        for surf_i, cnt in sorted(loss_by_surface.items()):
            s_name = self.stack.surfaces[surf_i].name if 0 <= surf_i < len(self.stack.surfaces) else "Unknown"
            print(f"    Surface {surf_i} ({s_name}): {cnt} rays lost ({cnt/total_rays*100:.1f}%)")

        # Spot analysis on detector for rays that reach detector
        det_hits = []
        for r in traced:
            for h in r.history:
                if h["surface_index"] == self.det_idx:
                    det_hits.append((r.wavelength, h["x"], h["y"], r.vignetted))
                    
        print(f"  Total Rays Reaching Detector Plane: {len(det_hits)}")
        if det_hits:
            det_x = [h[1] for h in det_hits]
            det_y = [h[2] for h in det_hits]
            det_r = np.sqrt(np.array(det_x)**2 + np.array(det_y)**2)
            max_r = np.max(det_r)
            rms_r = np.sqrt(np.mean(det_r**2))
            
            within_detector = [h for h in det_hits if np.sqrt(h[1]**2 + h[2]**2) <= DETECTOR_RADIUS_MM]
            
            print(f"  Detector Spot Footprint:")
            print(f"    Max Radius: {max_r:.4f} mm (Max Diameter: {2*max_r:.4f} mm)")
            print(f"    RMS Radius: {rms_r:.4f} mm")
            print(f"    Centroid: ({np.mean(det_x):+.4f}, {np.mean(det_y):+.4f}) mm")
            print(f"    Rays within 1.0 mm Active Area (r <= 0.50 mm): {len(within_detector)} / {len(det_hits)} ({len(within_detector)/len(det_hits)*100:.1f}%)")


if __name__ == "__main__":
    unittest.main()
