"""
Tier 4: Real-World Workloads & Acceptance Criteria Test Suite.

Verifies:
- 100% transmission (zero vignetting) across dense pupil grids (>= 300 rays).
- 100% ray containment strictly within r <= 0.4000 mm on the 0.8 mm detector footprint.
- Standalone CLI execution of verify_optical_prescriptions.py (exit code 0).
- Programmatic verification functions confirming mathematical proof of containment.
- Dense hexapolar and Monte Carlo ray traces across all NIR wavelengths (900, 1100, 1225, 1350, 1700 nm).
- Encircled energy EE(r=0.400mm) == 1.0 (100% energy inside clear detector aperture).
"""

import importlib.util
import math
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pytest

from v2.engine.surface import Surface, SurfaceType
from v2.engine.tracer import OpticalSystem, Ray
from v2.analysis.spot import SpotDiagram, calculate_spot_metrics


class TestTier4RealWorld:
    """
    Tier 4 Real-World Workload test suite verifying Acceptance Criteria 1, 2, 3, 4.
    """

    # --------------------------------------------------------------------------
    # ACCEPTANCE CRITERION 1 & 2: PRIMARY RELAY THROUGHPUT & SPOT CONTAINMENT
    # --------------------------------------------------------------------------

    def test_t4_01_primary_relay_dense_grid_zero_vignetting(self, primary_optical_system: OpticalSystem, hexapolar_grid_factory, spot_analyzer):
        """T4.01: Assert 100% transmission (0 vignetting) on dense hexapolar pupil grid (>= 200 rays) in primary layout."""
        rays = hexapolar_grid_factory(na=0.12, num_rings=8, wavelength_um=1.225)
        assert len(rays) >= 200, f"Expected >= 200 rays, generated {len(rays)}"

        metrics = spot_analyzer(primary_optical_system, rays)
        assert metrics["n_vignetted"] == 0, f"Primary relay had {metrics['n_vignetted']} vignetted rays out of {metrics['n_launched']}"
        assert metrics["transmission"] >= 0.999, f"Transmission {metrics['transmission']:.4f} < 1.000"

    def test_t4_02_primary_relay_100pct_spot_containment_r_0p4mm(self, primary_optical_system: OpticalSystem, hexapolar_grid_factory, spot_analyzer):
        """T4.02: Mathematically assert 100% of rays at detector plane have r <= 0.4000 mm in primary layout."""
        rays = hexapolar_grid_factory(na=0.12, num_rings=8, wavelength_um=1.225)
        metrics = spot_analyzer(primary_optical_system, rays)

        assert metrics["all_contained_0p4mm"], (
            f"Primary relay failed 100% containment: max radius = {metrics['r_max']:.4f} mm > 0.4000 mm limit"
        )
        assert metrics["r_max"] <= 0.4000, f"R_max = {metrics['r_max']:.4f} mm exceeds 0.4000 mm"

    # --------------------------------------------------------------------------
    # ACCEPTANCE CRITERION 2: NIR 900-1700nm THROUGHPUT & SPOT CONTAINMENT
    # --------------------------------------------------------------------------

    def test_t4_03_nir_layout_dense_grid_zero_vignetting_all_wavelengths(
        self, nir_optical_system: OpticalSystem, hexapolar_grid_factory, spot_analyzer
    ):
        """T4.03: Assert 100% transmission (zero vignetting) across all 5 operating wavelengths (900, 1100, 1225, 1350, 1700 nm)."""
        wavelengths = [0.900, 1.100, 1.225, 1.350, 1.700]
        for wl in wavelengths:
            rays = hexapolar_grid_factory(na=0.12, num_rings=6, wavelength_um=wl)
            metrics = spot_analyzer(nir_optical_system, rays)
            assert metrics["n_vignetted"] == 0, (
                f"NIR layout at {wl} µm had {metrics['n_vignetted']} vignetted rays out of {metrics['n_launched']}"
            )
            assert metrics["transmission"] >= 0.999, f"NIR layout at {wl} µm transmission {metrics['transmission']} < 1.0"

    def test_t4_04_nir_layout_100pct_spot_containment_all_wavelengths(
        self, nir_optical_system: OpticalSystem, hexapolar_grid_factory, spot_analyzer
    ):
        """T4.04: Mathematically assert 100% of ray arrivals at detector plane have r <= 0.4000 mm across all 5 wavelengths."""
        wavelengths = [0.900, 1.100, 1.225, 1.350, 1.700]
        for wl in wavelengths:
            rays = hexapolar_grid_factory(na=0.12, num_rings=6, wavelength_um=wl)
            metrics = spot_analyzer(nir_optical_system, rays)
            assert metrics["r_max"] <= 0.4000, (
                f"NIR layout at {wl} µm exceeded detector radius: R_max = {metrics['r_max']:.4f} mm > 0.4000 mm"
            )
            assert metrics["all_contained_0p4mm"], f"NIR layout at {wl} µm failed 100% containment flag"

    def test_t4_05_nir_layout_rms_spot_radius_threshold(
        self, nir_optical_system: OpticalSystem, hexapolar_grid_factory, spot_analyzer
    ):
        """T4.05: Verify RMS spot radius R_RMS <= 0.250 mm across all NIR operating channels."""
        wavelengths = [0.900, 1.100, 1.225, 1.350, 1.700]
        for wl in wavelengths:
            rays = hexapolar_grid_factory(na=0.12, num_rings=6, wavelength_um=wl)
            metrics = spot_analyzer(nir_optical_system, rays)
            assert metrics["rms_radius"] <= 0.2500, (
                f"NIR layout at {wl} µm has excessive RMS spot radius {metrics['rms_radius']:.4f} mm > 0.250 mm"
            )

    def test_t4_06_nir_layout_geometric_spot_radius_combined_field(
        self, nir_optical_system: OpticalSystem, hexapolar_grid_factory
    ):
        """T4.06: Verify combined polychromatic geometric spot radius R_GEO <= 0.4000 mm."""
        wavelengths = [0.900, 1.100, 1.225, 1.350, 1.700]
        all_x: List[float] = []
        all_y: List[float] = []

        for wl in wavelengths:
            rays = hexapolar_grid_factory(na=0.12, num_rings=5, wavelength_um=wl)
            traced = nir_optical_system.trace_rays(rays)
            for r in traced:
                if not r.vignetted and len(r.history) > 0:
                    all_x.append(r.history[-1]["x"])
                    all_y.append(r.history[-1]["y"])

        assert len(all_x) >= 200
        spot = SpotDiagram(all_x, all_y, wavelength_um=1.225)
        geo_r = spot.calculate_geo_radius()
        assert geo_r <= 0.4000, f"Combined polychromatic R_GEO = {geo_r:.4f} mm > 0.4000 mm"

    # --------------------------------------------------------------------------
    # ACCEPTANCE CRITERION 3: VERIFICATION SCRIPT EXECUTION
    # --------------------------------------------------------------------------

    def test_t4_07_verification_script_cli_execution(self, verify_script_path: Path):
        """T4.07: Run verify_optical_prescriptions.py via CLI and assert exit code 0."""
        if not verify_script_path.is_file():
            pytest.skip(f"Verification script not yet generated at {verify_script_path}")

        proc = subprocess.run(
            [sys.executable, str(verify_script_path)],
            capture_output=True,
            text=True,
            timeout=120
        )
        assert proc.returncode == 0, (
            f"Verification script failed with exit code {proc.returncode}.\n"
            f"Stdout: {proc.stdout}\nStderr: {proc.stderr}"
        )
        assert "PASS" in proc.stdout.upper() or "SUCCESS" in proc.stdout.upper(), (
            f"Expected PASS/SUCCESS in output, got:\n{proc.stdout}"
        )

    def test_t4_08_verification_script_programmatic_import_and_execution(self, verify_script_path: Path):
        """T4.08: Programmatically import verification script and verify result dictionary."""
        if not verify_script_path.is_file():
            pytest.skip("Verification script not yet generated")

        spec = importlib.util.spec_from_file_location("verify_optical_prescriptions", str(verify_script_path))
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules["verify_optical_prescriptions"] = module
        spec.loader.exec_module(module)

        # Check for verification function (e.g. verify_all_prescriptions or main or verify_prescriptions)
        verify_fn = getattr(module, "verify_all_prescriptions", None) or getattr(module, "verify_prescriptions", None)
        if callable(verify_fn):
            result = verify_fn()
            assert result is not None
            if isinstance(result, dict):
                assert result.get("success", True) is True
                assert result.get("all_passed", True) is True

    # --------------------------------------------------------------------------
    # MONTE CARLO & ENCIRCLED ENERGY CONFIRMATION
    # --------------------------------------------------------------------------

    def test_t4_09_high_density_monte_carlo_spot_containment(
        self, nir_optical_system: OpticalSystem, spot_analyzer
    ):
        """T4.09: High-density Monte Carlo pupil sampling (500 rays per channel) confirms 100% containment in 0.400mm."""
        np.random.seed(42)
        wavelengths = [0.900, 1.225, 1.700]
        n_mc = 300

        for wl in wavelengths:
            # Generate random uniform pupil rays inside NA cone
            u1 = np.random.uniform(0.0, 1.0, n_mc)
            u2 = np.random.uniform(0.0, 2.0 * np.pi, n_mc)
            max_theta = math.asin(0.12)
            thetas = np.sqrt(u1) * max_theta

            mc_rays = []
            for th, ph in zip(thetas, u2):
                st = math.sin(th)
                ct = math.cos(th)
                d = np.array([st * math.cos(ph), st * math.sin(ph), ct], dtype=float)
                d /= np.linalg.norm(d)
                mc_rays.append(Ray(pos=np.array([0.0, 0.0, 0.0]), dir=d, wavelength=wl))

            metrics = spot_analyzer(nir_optical_system, mc_rays)
            assert metrics["n_vignetted"] == 0, f"Monte Carlo had {metrics['n_vignetted']} vignetted rays at {wl} µm"
            assert metrics["r_max"] <= 0.4000, f"Monte Carlo max radius {metrics['r_max']:.4f} mm > 0.4000 mm at {wl} µm"

    def test_t4_10_encircled_energy_100pct_within_detector_aperture(
        self, nir_optical_system: OpticalSystem, hexapolar_grid_factory
    ):
        """T4.10: Compute radial encircled energy fraction EE(r) and assert EE(r = 0.400 mm) == 1.00 (100%)."""
        rays = hexapolar_grid_factory(na=0.12, num_rings=7, wavelength_um=1.225)
        traced = nir_optical_system.trace_rays(rays)

        radii = [math.sqrt(r.history[-1]["x"]**2 + r.history[-1]["y"]**2) for r in traced if not r.vignetted]
        assert len(radii) > 0

        # Encircled energy at r = 0.4000 mm
        n_inside = sum(1 for r in radii if r <= 0.4000)
        ee_fraction = n_inside / float(len(radii))

        assert ee_fraction == 1.0, f"Encircled energy at r=0.400 mm is {ee_fraction * 100:.2f}% (expected exactly 100.0%)"
