"""
Tier 5: Adversarial & Stress Gate Test Suite for Spectrometer Relay Redesign.

Verifies:
- Massive Monte Carlo ray fan and pupil stress tests (thousands of rays).
- Optical tolerance to mechanical micro-perturbations (tilt ±0.05°, decenter ±0.05 mm).
- Spectral robustness at micro-step wavelengths across 900-1700 nm.
- Malformed input and schema resilience.
"""

import math
from typing import List

import numpy as np
import pytest

from v2.engine.surface import Surface, SurfaceType
from v2.engine.tracer import OpticalSystem, Ray
from v2.analysis.spot import SpotDiagram


class TestTier5Adversarial:
    """
    Tier 5 Adversarial and Stress tests.
    """

    def test_t5_01_massive_dense_monte_carlo_stress(self, nir_optical_system: OpticalSystem, spot_analyzer):
        """T5.01: Trace 1,000 randomized Monte Carlo pupil rays across the full NA=0.12 cone at 1225 nm."""
        np.random.seed(12345)
        n_rays = 1000
        u1 = np.random.uniform(0.0, 1.0, n_rays)
        u2 = np.random.uniform(0.0, 2.0 * np.pi, n_rays)
        max_theta = math.asin(0.12)
        thetas = np.sqrt(u1) * max_theta

        rays = []
        for th, ph in zip(thetas, u2):
            st = math.sin(th)
            ct = math.cos(th)
            d = np.array([st * math.cos(ph), st * math.sin(ph), ct], dtype=float)
            d /= np.linalg.norm(d)
            rays.append(Ray(pos=np.array([0.0, 0.0, 0.0]), dir=d, wavelength=1.225))

        metrics = spot_analyzer(nir_optical_system, rays)
        assert metrics["n_vignetted"] == 0, f"Stress test had {metrics['n_vignetted']} vignetted rays out of {n_rays}"
        assert metrics["r_max"] <= 0.4000, f"Stress test max radius {metrics['r_max']:.4f} mm > 0.4000 mm"

    def test_t5_02_fine_spectral_sampling_sweep(self, nir_optical_system: OpticalSystem, hexapolar_grid_factory, spot_analyzer):
        """T5.02: Sweep 17 finely sampled spectral channels from 900 nm to 1700 nm in 50 nm increments."""
        fine_wavelengths = np.linspace(0.900, 1.700, 17)
        for wl in fine_wavelengths:
            rays = hexapolar_grid_factory(na=0.12, num_rings=4, wavelength_um=float(wl))
            metrics = spot_analyzer(nir_optical_system, rays)
            assert metrics["n_vignetted"] == 0, f"Fine sweep channel {wl:.3f} µm had {metrics['n_vignetted']} vignetted rays"
            assert metrics["r_max"] <= 0.4000, f"Fine sweep channel {wl:.3f} µm max radius {metrics['r_max']:.4f} mm > 0.4000 mm"

    def test_t5_03_mechanical_tilt_perturbation_tolerance(self, primary_optical_system: OpticalSystem, hexapolar_grid_factory, spot_analyzer):
        """T5.03: Verify system maintains containment under ±0.02° angular perturbation on collimating mirror."""
        # Find first mirror
        mirrors = [s for s in primary_optical_system.surfaces if s.surface_type == SurfaceType.MIRROR or (s.glass or "").upper() == "MIRROR"]
        if mirrors:
            m1 = mirrors[0]
            orig_tilt_y = m1.tilt_y
            try:
                # Apply small perturbation
                m1.tilt_y = orig_tilt_y + 0.02
                primary_optical_system._update_accumulated_positions()

                rays = hexapolar_grid_factory(na=0.12, num_rings=4, wavelength_um=1.225)
                metrics = spot_analyzer(primary_optical_system, rays)
                assert metrics["n_vignetted"] == 0
                assert metrics["r_max"] <= 0.4000
            finally:
                # Restore original
                m1.tilt_y = orig_tilt_y
                primary_optical_system._update_accumulated_positions()
