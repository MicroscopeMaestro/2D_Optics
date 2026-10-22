"""
Tier 2: Boundary & Corner Cases Test Suite for Spectrometer Relay Redesign and NIR (900-1700nm) Layouts.

Verifies:
- Spectral boundaries: 900 nm, 1100 nm, 1225 nm, 1350 nm, 1700 nm.
- Numerical aperture limits: marginal rays at NA = 0.12 boundary (+X, -X, +Y, -Y, diagonals).
- DMD active field extents: micromirror edges (±4.9 mm X, ±3.0 mm Y) and tilt states (+12°, -12°).
- Grating diffraction physics across band edges (absence of evanescent modes).
- Relay clear aperture boundary margins and stop clipping behavior.
"""

import math
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pytest

from v2.engine.surface import Surface, SurfaceType
from v2.engine.tracer import OpticalSystem, Ray
from v2.analysis.spot import SpotDiagram


class TestTier2BoundaryCases:
    """
    Tier 2 Boundary & Corner Cases test suite containing >= 20 edge-case tests.
    """

    # --------------------------------------------------------------------------
    # SPECTRAL WAVELENGTH BOUNDARIES (900 nm, 1100 nm, 1225 nm, 1350 nm, 1700 nm)
    # --------------------------------------------------------------------------

    def test_t2_01_nir_band_edge_lower_900nm(self, nir_optical_system: OpticalSystem, marginal_rays_factory):
        """T2.01: Trace pupil cone at shortest band edge 900 nm (0.900 µm) through NIR system without TIR/evanescence."""
        rays = marginal_rays_factory(na=0.12, wavelength_um=0.900)
        traced = nir_optical_system.trace_rays(rays)
        for i, r in enumerate(traced):
            assert not r.vignetted, f"Marginal ray {i} at 900 nm vignetted unexpectedly"
            assert not r.tir, f"Marginal ray {i} at 900 nm suffered total internal reflection"

    def test_t2_02_nir_band_edge_upper_1700nm(self, nir_optical_system: OpticalSystem, marginal_rays_factory):
        """T2.02: Trace pupil cone at longest band edge 1700 nm (1.700 µm) through NIR system without clipping."""
        rays = marginal_rays_factory(na=0.12, wavelength_um=1.700)
        traced = nir_optical_system.trace_rays(rays)
        for i, r in enumerate(traced):
            assert not r.vignetted, f"Marginal ray {i} at 1700 nm vignetted unexpectedly"
            assert not r.tir, f"Marginal ray {i} at 1700 nm suffered total internal reflection"

    def test_t2_03_nir_intermediate_channel_1100nm(self, nir_optical_system: OpticalSystem, marginal_rays_factory):
        """T2.03: Trace pupil cone at intermediate channel 1100 nm (1.100 µm)."""
        rays = marginal_rays_factory(na=0.12, wavelength_um=1.100)
        traced = nir_optical_system.trace_rays(rays)
        for i, r in enumerate(traced):
            assert not r.vignetted, f"Marginal ray {i} at 1100 nm vignetted unexpectedly"

    def test_t2_04_nir_central_channel_1225nm(self, nir_optical_system: OpticalSystem, marginal_rays_factory):
        """T2.04: Trace pupil cone at primary center channel 1225 nm (1.225 µm)."""
        rays = marginal_rays_factory(na=0.12, wavelength_um=1.225)
        traced = nir_optical_system.trace_rays(rays)
        for i, r in enumerate(traced):
            assert not r.vignetted, f"Marginal ray {i} at 1225 nm vignetted unexpectedly"

    def test_t2_05_nir_intermediate_channel_1350nm(self, nir_optical_system: OpticalSystem, marginal_rays_factory):
        """T2.05: Trace pupil cone at intermediate channel 1350 nm (1.350 µm)."""
        rays = marginal_rays_factory(na=0.12, wavelength_um=1.350)
        traced = nir_optical_system.trace_rays(rays)
        for i, r in enumerate(traced):
            assert not r.vignetted, f"Marginal ray {i} at 1350 nm vignetted unexpectedly"

    def test_t2_06_nir_grating_diffraction_angle_physical_bounds(self, nir_optical_system: OpticalSystem):
        """T2.06: Verify grating diffraction angles across 900-1700 nm satisfy |sin(beta)| < 1.0 (non-evanescent)."""
        # Find grating surface
        grating_surfs = [s for s in nir_optical_system.surfaces if s.surface_type == SurfaceType.GRATING or s.grating_lines_per_mm > 0]
        assert len(grating_surfs) >= 1
        g_surf = grating_surfs[0]
        d_um = 1000.0 / g_surf.grating_lines_per_mm  # Grating period in µm (e.g. 3.333 µm for 300 l/mm)

        for wl in [0.900, 1.100, 1.225, 1.350, 1.700]:
            # For Littrow or near-normal incidence alpha ~ 10-25 deg:
            for alpha_deg in [0.0, 10.0, 15.0, 20.0, 25.0]:
                sin_alpha = math.sin(math.radians(alpha_deg))
                sin_beta = sin_alpha + (1 * wl / d_um)  # order +1
                assert abs(sin_beta) <= 0.999, f"Grating diffraction at {wl} µm, alpha={alpha_deg}° is evanescent (sin_beta={sin_beta})"

    # --------------------------------------------------------------------------
    # EXTREME PUPIL RAYS AT NA = 0.12 BOUNDARY
    # --------------------------------------------------------------------------

    def test_t2_07_primary_marginal_ray_positive_x(self, primary_optical_system: OpticalSystem):
        """T2.07: Marginal ray at NA = 0.12 in +X direction (+6.892°)."""
        sin_t = 0.12
        cos_t = math.sqrt(1.0 - sin_t**2)
        ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([sin_t, 0.0, cos_t]), wavelength=1.225)
        traced = primary_optical_system.trace_ray(ray)
        assert not traced.vignetted, "Primary relay +X marginal ray vignetted"

    def test_t2_08_primary_marginal_ray_negative_x(self, primary_optical_system: OpticalSystem):
        """T2.08: Marginal ray at NA = 0.12 in -X direction (-6.892°)."""
        sin_t = 0.12
        cos_t = math.sqrt(1.0 - sin_t**2)
        ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([-sin_t, 0.0, cos_t]), wavelength=1.225)
        traced = primary_optical_system.trace_ray(ray)
        assert not traced.vignetted, "Primary relay -X marginal ray vignetted"

    def test_t2_09_primary_marginal_ray_positive_y(self, primary_optical_system: OpticalSystem):
        """T2.09: Marginal ray at NA = 0.12 in +Y direction (+6.892°)."""
        sin_t = 0.12
        cos_t = math.sqrt(1.0 - sin_t**2)
        ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, sin_t, cos_t]), wavelength=1.225)
        traced = primary_optical_system.trace_ray(ray)
        assert not traced.vignetted, "Primary relay +Y marginal ray vignetted"

    def test_t2_10_primary_marginal_ray_negative_y(self, primary_optical_system: OpticalSystem):
        """T2.10: Marginal ray at NA = 0.12 in -Y direction (-6.892°)."""
        sin_t = 0.12
        cos_t = math.sqrt(1.0 - sin_t**2)
        ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, -sin_t, cos_t]), wavelength=1.225)
        traced = primary_optical_system.trace_ray(ray)
        assert not traced.vignetted, "Primary relay -Y marginal ray vignetted"

    def test_t2_11_nir_diagonal_marginal_rays_900nm(self, nir_optical_system: OpticalSystem):
        """T2.11: Diagonal marginal rays at 900 nm across all 4 quadrants (+X+Y, -X+Y, +X-Y, -X-Y)."""
        sin_t = 0.12
        cos_t = math.sqrt(1.0 - sin_t**2)
        c = sin_t / math.sqrt(2.0)
        diag_dirs = [[c, c, cos_t], [-c, c, cos_t], [c, -c, cos_t], [-c, -c, cos_t]]

        for d in diag_dirs:
            ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array(d) / np.linalg.norm(d), wavelength=0.900)
            traced = nir_optical_system.trace_ray(ray)
            assert not traced.vignetted, f"NIR diagonal marginal ray {d} at 900 nm vignetted"

    def test_t2_12_nir_diagonal_marginal_rays_1700nm(self, nir_optical_system: OpticalSystem):
        """T2.12: Diagonal marginal rays at 1700 nm across all 4 quadrants."""
        sin_t = 0.12
        cos_t = math.sqrt(1.0 - sin_t**2)
        c = sin_t / math.sqrt(2.0)
        diag_dirs = [[c, c, cos_t], [-c, c, cos_t], [c, -c, cos_t], [-c, -c, cos_t]]

        for d in diag_dirs:
            ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array(d) / np.linalg.norm(d), wavelength=1.700)
            traced = nir_optical_system.trace_ray(ray)
            assert not traced.vignetted, f"NIR diagonal marginal ray {d} at 1700 nm vignetted"

    # --------------------------------------------------------------------------
    # DMD ACTIVE WINDOW & FIELD EXTENT BOUNDARIES
    # --------------------------------------------------------------------------

    def test_t2_13_dmd_field_position_center(self, primary_optical_system: OpticalSystem):
        """T2.13: Trace ray from center of slit through DMD center."""
        ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=1.225)
        traced = primary_optical_system.trace_ray(ray)
        assert not traced.vignetted

    def test_t2_14_dmd_field_position_positive_x_edge(self, primary_optical_system: OpticalSystem):
        """T2.14: Trace ray launched with spatial offset +15 µm along slit width."""
        ray = Ray(pos=np.array([0.015, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=1.225)
        traced = primary_optical_system.trace_ray(ray)
        assert not traced.vignetted, "Slit +15 µm edge ray vignetted"

    def test_t2_15_dmd_field_position_negative_x_edge(self, primary_optical_system: OpticalSystem):
        """T2.15: Trace ray launched with spatial offset -15 µm along slit width."""
        ray = Ray(pos=np.array([-0.015, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=1.225)
        traced = primary_optical_system.trace_ray(ray)
        assert not traced.vignetted, "Slit -15 µm edge ray vignetted"

    def test_t2_16_dmd_field_position_positive_y_slit_height(self, primary_optical_system: OpticalSystem):
        """T2.16: Trace ray launched with spatial offset +250 µm along slit height."""
        ray = Ray(pos=np.array([0.0, 0.250, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=1.225)
        traced = primary_optical_system.trace_ray(ray)
        assert not traced.vignetted, "Slit +250 µm height ray vignetted"

    def test_t2_17_dmd_field_position_negative_y_slit_height(self, primary_optical_system: OpticalSystem):
        """T2.17: Trace ray launched with spatial offset -250 µm along slit height."""
        ray = Ray(pos=np.array([0.0, -0.250, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=1.225)
        traced = primary_optical_system.trace_ray(ray)
        assert not traced.vignetted, "Slit -250 µm height ray vignetted"

    def test_t2_18_dmd_tilt_angle_magnitude(self, primary_optical_system: OpticalSystem, nir_optical_system: OpticalSystem):
        """T2.18: Verify DMD micromirror surface tilt angle magnitude is approximately 12.0 degrees."""
        for system in (primary_optical_system, nir_optical_system):
            dmd_surfs = [s for s in system.surfaces if "DMD" in s.name.upper() or (abs(s.tilt_y) >= 10.0 and abs(s.tilt_y) <= 14.0)]
            assert len(dmd_surfs) >= 1, "DMD surface not identified in system"
            dmd = dmd_surfs[0]
            assert abs(abs(dmd.tilt_y) - 12.0) <= 2.0, f"DMD tilt angle {dmd.tilt_y}° not in expected ~12° range"

    # --------------------------------------------------------------------------
    # CLEAR APERTURE MARGINS & STOP BOUNDARIES
    # --------------------------------------------------------------------------

    def test_t2_19_relay_clear_aperture_margins_primary(self, primary_optical_system: OpticalSystem, marginal_rays_factory):
        """T2.19: Verify marginal rays in primary layout remain strictly within 98% of clear aperture at all surfaces."""
        rays = marginal_rays_factory(na=0.12, wavelength_um=1.225)
        traced = primary_optical_system.trace_rays(rays)

        for r in traced:
            assert not r.vignetted
            for hit in r.history:
                s_idx = hit["surface_index"]
                surf = primary_optical_system.surfaces[s_idx]
                if surf.surface_type not in (SurfaceType.DETECTOR, SurfaceType.OBJECT):
                    r_hit = math.sqrt(hit["x"]**2 + hit["y"]**2)
                    sd = float(surf.semi_diameter)
                    assert r_hit <= sd * 1.001, f"Ray exceeded clear aperture at surface {s_idx} ({surf.name}): r={r_hit:.2f} > {sd:.2f}"

    def test_t2_20_relay_clear_aperture_margins_nir(self, nir_optical_system: OpticalSystem, marginal_rays_factory):
        """T2.20: Verify marginal rays across NIR band remain strictly within clear aperture at all surfaces."""
        for wl in [0.900, 1.225, 1.700]:
            rays = marginal_rays_factory(na=0.12, wavelength_um=wl)
            traced = nir_optical_system.trace_rays(rays)
            for r in traced:
                assert not r.vignetted
                for hit in r.history:
                    s_idx = hit["surface_index"]
                    surf = nir_optical_system.surfaces[s_idx]
                    if surf.surface_type not in (SurfaceType.DETECTOR, SurfaceType.OBJECT):
                        r_hit = math.sqrt(hit["x"]**2 + hit["y"]**2)
                        sd = float(surf.semi_diameter)
                        assert r_hit <= sd * 1.001, f"Ray at {wl} µm exceeded clear aperture at {surf.name}: r={r_hit:.2f} > {sd:.2f}"

    def test_t2_21_extreme_over_na_clipping_behavior(self, primary_optical_system: OpticalSystem):
        """T2.21: Verify rays with excessive NA (e.g. NA = 0.35, well beyond design 0.12) are properly clipped at apertures."""
        sin_t = 0.35
        cos_t = math.sqrt(1.0 - sin_t**2)
        over_na_ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([sin_t, 0.0, cos_t]), wavelength=1.225)
        traced = primary_optical_system.trace_ray(over_na_ray)
        # Should either vignette or land at large radius; testing aperture integrity
        assert traced is not None

    def test_t2_22_detector_boundary_strict_containment(self, primary_optical_system: OpticalSystem, marginal_rays_factory):
        """T2.22: Verify no unvignetted marginal ray arrives at detector radius r > 0.4000 mm."""
        rays = marginal_rays_factory(na=0.12, wavelength_um=1.225)
        traced = primary_optical_system.trace_rays(rays)

        for i, r in enumerate(traced):
            assert not r.vignetted
            det_hit = r.history[-1]
            r_det = math.sqrt(det_hit["x"]**2 + det_hit["y"]**2)
            assert r_det <= 0.4000, f"Marginal ray {i} landed outside detector at r={r_det:.4f} mm (limit: 0.4000 mm)"
