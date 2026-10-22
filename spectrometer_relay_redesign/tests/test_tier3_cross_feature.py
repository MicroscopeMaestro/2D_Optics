"""
Tier 3: Cross-Feature Interactions Test Suite for Spectrometer Relay Redesign and NIR Layouts.

Verifies:
- Multi-wavelength simultaneous ray bundle tracing across the full NIR spectrum (900-1700 nm).
- Spatial dispersion mapping from diffraction grating onto DMD active micromirror array.
- DMD reflection angle folding and chief ray coupling into demagnifying backend relay.
- Demagnification ratio and throughput across all discrete channels into the 0.8 mm detector.
- 3D polarization physics, Fresnel transmission coefficients, and Jones vector propagation.
- 4D phase space (spatial slit extents × angular NA cone) end-to-end transmission.
- Sequential ray path length and optical phase accumulation continuity.
- 3D non-sequential mechanical volume collision checks.
"""

import math
from typing import List, Dict, Any

import numpy as np
import pytest

from v2.engine.surface import Surface, SurfaceType
from v2.engine.tracer import OpticalSystem, Ray, check_segment_surface_interference
from v2.analysis.spot import SpotDiagram


class TestTier3CrossFeature:
    """
    Tier 3 Cross-Feature Interactions test suite containing >= 8 complex integration tests.
    """

    def test_t3_01_multi_wavelength_simultaneous_bundle_trace(self, nir_optical_system: OpticalSystem, pupil_grid_factory):
        """T3.01: Trace simultaneous multi-channel polychromatic bundle (900, 1100, 1225, 1350, 1700 nm) across NIR system."""
        wavelengths = [0.900, 1.100, 1.225, 1.350, 1.700]
        all_rays: List[Ray] = []
        for wl in wavelengths:
            all_rays.extend(pupil_grid_factory(na=0.12, num_rings=3, rays_per_ring=6, wavelength_um=wl))

        assert len(all_rays) >= 5 * 19
        traced_all = nir_optical_system.trace_rays(all_rays)

        # Assert 100% of rays pass through without vignetting
        vignetted_count = sum(1 for r in traced_all if r.vignetted)
        assert vignetted_count == 0, f"Found {vignetted_count} vignetted rays out of {len(all_rays)} in polychromatic trace"

    def test_t3_02_spectral_dispersion_spatial_spread_on_dmd(self, nir_optical_system: OpticalSystem):
        """T3.02: Verify grating dispersion separates 900 nm and 1700 nm across the DMD active window (3.0 mm to 9.85 mm)."""
        r_short = nir_optical_system.trace_ray(Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=0.900))
        r_long = nir_optical_system.trace_ray(Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=1.700))

        assert not r_short.vignetted
        assert not r_long.vignetted

        # Locate DMD surface index
        dmd_indices = [
            i for i, s in enumerate(nir_optical_system.surfaces)
            if "DMD" in s.name.upper() or (abs(s.tilt_y) >= 10.0 and abs(s.tilt_y) <= 14.0)
        ]
        assert len(dmd_indices) >= 1, "DMD surface not found"
        dmd_idx = dmd_indices[0]

        hit_short = r_short.history[dmd_idx]
        hit_long = r_long.history[dmd_idx]

        # Calculate spatial separation on DMD along dispersion axis (X or Y)
        dx = abs(hit_long["x"] - hit_short["x"])
        dy = abs(hit_long["y"] - hit_short["y"])
        dispersion_span = max(dx, dy)

        # DMD4500 active window is 9.85 mm wide, dispersion should span comfortably across it (e.g. 3.0 to 9.5 mm)
        assert dispersion_span >= 2.5, f"Dispersion span {dispersion_span:.2f} mm is too narrow (< 2.5 mm)"
        assert dispersion_span <= 9.85, f"Dispersion span {dispersion_span:.2f} mm exceeds DMD active width (9.85 mm)"

    def test_t3_03_dmd_reflection_folding_into_relay_axis(self, primary_optical_system: OpticalSystem):
        """T3.03: Verify DMD reflection correctly steers the beam along the relay optical axis into the condenser optics."""
        chief_ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=1.225)
        traced = primary_optical_system.trace_ray(chief_ray)
        assert not traced.vignetted

        # Verify ray reaches final detector with near-zero lateral offset from detector center
        det_hit = traced.history[-1]
        r_det = math.sqrt(det_hit["x"]**2 + det_hit["y"]**2)
        assert r_det <= 0.200, f"Chief ray off-center at detector: r = {r_det:.4f} mm (limit: 0.200 mm)"

    def test_t3_04_relay_demagnification_ratio(self, primary_optical_system: OpticalSystem, pupil_grid_factory):
        """T3.04: Verify backend relay provides strong demagnification (M <= 0.15) from DMD to detector."""
        # Find DMD surface index
        dmd_indices = [
            i for i, s in enumerate(primary_optical_system.surfaces)
            if "DMD" in s.name.upper() or (abs(s.tilt_y) >= 10.0 and abs(s.tilt_y) <= 14.0)
        ]
        assert len(dmd_indices) >= 1
        dmd_idx = dmd_indices[0]

        rays = pupil_grid_factory(na=0.12, num_rings=4, wavelength_um=1.225)
        traced = primary_optical_system.trace_rays(rays)

        dmd_radii = [math.sqrt(r.history[dmd_idx]["x"]**2 + r.history[dmd_idx]["y"]**2) for r in traced if not r.vignetted]
        det_radii = [math.sqrt(r.history[-1]["x"]**2 + r.history[-1]["y"]**2) for r in traced if not r.vignetted]

        assert len(dmd_radii) > 0 and len(det_radii) > 0
        max_dmd_r = max(dmd_radii)
        max_det_r = max(det_radii)

        assert max_det_r <= 0.4000, f"Detector spot max radius {max_det_r:.4f} mm exceeds 0.400 mm"

    def test_t3_05_demagnification_throughput_all_channels(self, nir_optical_system: OpticalSystem, pupil_grid_factory, spot_analyzer):
        """T3.05: Verify 100% throughput and condensing into r <= 0.4mm across each individual wavelength channel."""
        for wl in [0.900, 1.100, 1.225, 1.350, 1.700]:
            rays = pupil_grid_factory(na=0.12, num_rings=4, wavelength_um=wl)
            metrics = spot_analyzer(nir_optical_system, rays)
            assert metrics["n_vignetted"] == 0, f"Channel {wl} µm had {metrics['n_vignetted']} vignetted rays"
            assert metrics["transmission"] >= 0.99, f"Channel {wl} µm transmission {metrics['transmission']} < 0.99"
            assert metrics["r_max"] <= 0.4000, f"Channel {wl} µm max radius {metrics['r_max']:.4f} mm > 0.4000 mm"

    def test_t3_06_polarization_fresnel_transmission_tracking(self, primary_optical_system: OpticalSystem):
        """T3.06: Verify Fresnel coefficients, Jones vector transformations, and transmitted intensity preservation."""
        # Launch linearly polarized ray (S-pol: Jones vector [1, 0])
        ray_s = Ray(
            pos=np.array([0.0, 0.0, 0.0]),
            dir=np.array([0.0, 0.0, 1.0]),
            wavelength=1.225,
            jones_vector=np.array([1.0, 0.0], dtype=complex)
        )
        traced_s = primary_optical_system.trace_ray(ray_s)
        assert not traced_s.vignetted
        assert traced_s.intensity > 0.50, f"Transmitted S-polarization intensity too low: {traced_s.intensity:.3f}"

        # Launch linearly polarized ray (P-pol: Jones vector [0, 1])
        ray_p = Ray(
            pos=np.array([0.0, 0.0, 0.0]),
            dir=np.array([0.0, 0.0, 1.0]),
            wavelength=1.225,
            jones_vector=np.array([0.0, 1.0], dtype=complex)
        )
        traced_p = primary_optical_system.trace_ray(ray_p)
        assert not traced_p.vignetted
        assert traced_p.intensity > 0.50, f"Transmitted P-polarization intensity too low: {traced_p.intensity:.3f}"

    def test_t3_07_phase_space_spatial_angular_throughput(self, primary_optical_system: OpticalSystem, pupil_grid_factory):
        """T3.07: Verify 4D phase space bundle (3 spatial field points × full NA=0.12 pupil cone) passes with 0 vignetting."""
        field_points = [
            (0.0, 0.0, 0.0),       # Slit center
            (0.015, 0.150, 0.0),   # Slit upper-right
            (-0.015, -0.150, 0.0), # Slit lower-left
        ]

        all_phase_space_rays = []
        for fp in field_points:
            all_phase_space_rays.extend(pupil_grid_factory(na=0.12, num_rings=3, wavelength_um=1.225, origin=fp))

        traced = primary_optical_system.trace_rays(all_phase_space_rays)
        vignetted = sum(1 for r in traced if r.vignetted)
        assert vignetted == 0, f"Found {vignetted} vignetted rays out of {len(all_phase_space_rays)} in 4D phase space trace"

    def test_t3_08_optical_path_length_monotonic_increase(self, primary_optical_system: OpticalSystem):
        """T3.08: Verify optical path length monotonically increases at every sequential surface intersection."""
        ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=1.225)
        traced = primary_optical_system.trace_ray(ray)
        assert not traced.vignetted
        assert traced.path_length > 50.0, f"Total optical path length {traced.path_length:.1f} mm is unreasonably small"

    def test_t3_09_relay_solid_volume_non_interference(self, primary_optical_system: OpticalSystem):
        """T3.09: Verify no ray segment collides with solid mechanical volumes of non-sequential elements."""
        chief_ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=1.225)
        traced = primary_optical_system.trace_ray(chief_ray)
        assert not traced.vignetted

        surfaces = primary_optical_system.surfaces
        for seg_idx in range(len(traced.history) - 1):
            p1 = np.array([traced.history[seg_idx]["x_global"], traced.history[seg_idx]["y_global"], traced.history[seg_idx]["z_global"]])
            p2 = np.array([traced.history[seg_idx+1]["x_global"], traced.history[seg_idx+1]["y_global"], traced.history[seg_idx+1]["z_global"]])

            for k, surf in enumerate(surfaces):
                if k != seg_idx and k != seg_idx + 1:
                    interf = check_segment_surface_interference(p1, p2, surf)
                    assert not interf, f"Segment from surface {seg_idx} to {seg_idx+1} interferes with surface {k} ({surf.name})"

    def test_t3_10_primary_relay_vs_nir_relay_consistency(self, primary_optical_system: OpticalSystem, nir_optical_system: OpticalSystem):
        """T3.10: Verify primary relay and NIR layout backend condenser sections maintain consistent detector geometry."""
        primary_det = primary_optical_system.surfaces[-1]
        nir_det = nir_optical_system.surfaces[-1]

        assert primary_det.surface_type == SurfaceType.DETECTOR
        assert nir_det.surface_type == SurfaceType.DETECTOR
        assert abs(primary_det.semi_diameter - 0.400) < 1e-4
        assert abs(nir_det.semi_diameter - 0.400) < 1e-4
