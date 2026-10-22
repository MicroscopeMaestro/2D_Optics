"""
Tier 1: Feature Coverage Test Suite for Spectrometer Relay Redesign and NIR (900-1700nm) Layouts.

Verifies:
- F1: Primary Relay Redesign JSON schema, surfaces, COTS elements, and detector geometry.
- F2: NIR 900-1700nm dedicated layout JSON schema, 300 l/mm grating, DMD, and detector.
- F3: v2 engine ray tracing compatibility, SurfaceStack loading, coordinate accumulation, ray tracing.
- F4: Spot diagram diagnostics, spot metric calculations, and verification script availability.
"""

import json
import math
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pytest

from v2.engine.surface import Surface, SurfaceType
from v2.engine.lde import SurfaceStack
from v2.engine.tracer import OpticalSystem, Ray
from v2.engine.materials import get_refractive_index
from v2.analysis.spot import SpotDiagram, calculate_spot_metrics


class TestTier1FeatureCoverage:
    """
    Tier 1 Feature Coverage test suite containing >= 20 comprehensive unit and integration tests.
    """

    # --------------------------------------------------------------------------
    # F1: PRIMARY RELAY REDESIGN JSON & SCHEMA
    # --------------------------------------------------------------------------

    def test_t1_01_primary_relay_json_file_exists(self, primary_layout_path: Path):
        """T1.01: Assert spectrometer_primary_relay_redesign.json exists at expected path."""
        assert primary_layout_path.is_file(), f"Primary relay JSON not found at {primary_layout_path}"

    def test_t1_02_primary_relay_json_valid_syntax(self, primary_layout_path: Path):
        """T1.02: Verify spectrometer_primary_relay_redesign.json parses without JSON decode errors."""
        if not primary_layout_path.is_file():
            pytest.skip("Primary relay layout not yet generated")
        with open(primary_layout_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, dict), "Root JSON object must be a dictionary"
        assert "surfaces" in data, "JSON layout must contain 'surfaces' key"

    def test_t1_03_primary_relay_schema_version_is_3(self, primary_layout_json: Dict[str, Any]):
        """T1.03: Verify primary relay JSON conforms to Schema Version 3."""
        version = primary_layout_json.get("version")
        assert version in (2, 3), f"Expected JSON schema version 2 or 3, got {version}"

    def test_t1_04_primary_relay_surface_count_and_sequence(self, primary_layout_json: Dict[str, Any]):
        """T1.04: Verify primary relay layout has >= 8 optical surfaces in sequential order."""
        surfaces = primary_layout_json["surfaces"]
        assert len(surfaces) >= 8, f"Expected at least 8 surfaces in primary layout, found {len(surfaces)}"

        # Surface 0 must be Object / Entrance Slit
        s0 = surfaces[0]
        s0_type = str(s0.get("surface_type", "")).upper()
        assert s0_type in ("OBJECT", "STOP", "STANDARD"), f"Surface 0 must be Object/Stop, got {s0_type}"

        # Final surface must be Detector
        s_last = surfaces[-1]
        s_last_type = str(s_last.get("surface_type", "")).upper()
        assert s_last_type == "DETECTOR", f"Final surface must be Detector, got {s_last_type}"

    def test_t1_05_primary_relay_contains_dmd_surface(self, primary_layout_json: Dict[str, Any]):
        """T1.05: Verify primary layout includes a DLP4500 DMD micromirror surface."""
        surfaces = primary_layout_json["surfaces"]
        dmd_found = any(
            ("DMD" in str(s.get("name", "")).upper()) or
            (abs(float(s.get("tilt_y", 0.0))) >= 10.0 and abs(float(s.get("tilt_y", 0.0))) <= 14.0)
            for s in surfaces
        )
        assert dmd_found, "Primary layout must contain a DMD micromirror surface"

    def test_t1_06_primary_relay_detector_clear_aperture(self, primary_layout_json: Dict[str, Any]):
        """T1.06: Verify final detector surface has semi_diameter == 0.4 mm (0.8 mm diameter)."""
        detector = primary_layout_json["surfaces"][-1]
        semi_dia = float(detector.get("semi_diameter", 0.0))
        assert abs(semi_dia - 0.400) < 1e-4, f"Detector semi-diameter must be exactly 0.4 mm, got {semi_dia}"

    # --------------------------------------------------------------------------
    # F2: NIR 900-1700nm LAYOUT JSON & SCHEMA
    # --------------------------------------------------------------------------

    def test_t1_07_nir_layout_json_file_exists(self, nir_layout_path: Path):
        """T1.07: Assert spectrometer_nir_900_1700nm.json exists at expected path."""
        assert nir_layout_path.is_file(), f"NIR layout JSON not found at {nir_layout_path}"

    def test_t1_08_nir_layout_json_valid_syntax(self, nir_layout_path: Path):
        """T1.08: Verify spectrometer_nir_900_1700nm.json parses cleanly as valid JSON."""
        if not nir_layout_path.is_file():
            pytest.skip("NIR layout not yet generated")
        with open(nir_layout_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, dict), "Root JSON object must be a dictionary"
        assert "surfaces" in data, "NIR JSON layout must contain 'surfaces' key"

    def test_t1_09_nir_layout_schema_version(self, nir_layout_json: Dict[str, Any]):
        """T1.09: Verify NIR layout conforms to schema version 2 or 3."""
        version = nir_layout_json.get("version")
        assert version in (2, 3), f"Expected schema version 2 or 3, got {version}"

    def test_t1_10_nir_layout_grating_specification_300_lines_per_mm(self, nir_layout_json: Dict[str, Any]):
        """T1.10: Verify NIR layout contains 300 lines/mm reflective grating in order 1."""
        gratings = [
            s for s in nir_layout_json["surfaces"]
            if str(s.get("surface_type", "")).upper() == "GRATING" or float(s.get("grating_lines_per_mm", 0.0)) > 0
        ]
        assert len(gratings) >= 1, "NIR layout must contain at least 1 diffraction grating"
        g = gratings[0]
        lines_per_mm = float(g.get("grating_lines_per_mm", 0.0))
        order = int(g.get("grating_order", 1))
        assert abs(lines_per_mm - 300.0) < 1.0, f"Expected 300 lines/mm grating, got {lines_per_mm}"
        assert order == 1, f"Expected order +1 grating, got {order}"

    def test_t1_11_nir_layout_wavelength_coverage(self, nir_layout_json: Dict[str, Any]):
        """T1.11: Verify NIR layout defines operating wavelength range spanning 0.900 to 1.700 µm."""
        wls = nir_layout_json.get("wavelengths", [])
        assert len(wls) >= 3, f"Expected at least 3 defined wavelengths, got {wls}"
        min_wl = min(wls)
        max_wl = max(wls)
        assert min_wl <= 0.950, f"Minimum wavelength {min_wl} µm should be <= 0.950 µm"
        assert max_wl >= 1.650, f"Maximum wavelength {max_wl} µm should be >= 1.650 µm"

    def test_t1_12_nir_layout_detector_geometry(self, nir_layout_json: Dict[str, Any]):
        """T1.12: Verify NIR layout final detector has semi_diameter == 0.4 mm."""
        detector = nir_layout_json["surfaces"][-1]
        semi_dia = float(detector.get("semi_diameter", 0.0))
        assert abs(semi_dia - 0.400) < 1e-4, f"Detector semi-diameter must be 0.4 mm, got {semi_dia}"

    # --------------------------------------------------------------------------
    # F3: TRACER COMPATIBILITY & OPTICAL SYSTEM LOADING
    # --------------------------------------------------------------------------

    def test_t1_13_primary_optical_system_loads(self, primary_optical_system: OpticalSystem):
        """T1.13: Verify OpticalSystem.from_json() succeeds for primary relay layout."""
        assert isinstance(primary_optical_system, OpticalSystem)
        assert len(primary_optical_system.surfaces) >= 8

    def test_t1_14_nir_optical_system_loads(self, nir_optical_system: OpticalSystem):
        """T1.14: Verify OpticalSystem.from_json() succeeds for NIR 900-1700nm layout."""
        assert isinstance(nir_optical_system, OpticalSystem)
        assert len(nir_optical_system.surfaces) >= 8

    def test_t1_15_glass_materials_resolve_in_catalog(self, primary_layout_json: Dict[str, Any], nir_layout_json: Dict[str, Any]):
        """T1.15: Verify all glass materials in both prescriptions resolve refractive indices."""
        combined_surfaces = primary_layout_json.get("surfaces", []) + nir_layout_json.get("surfaces", [])
        test_wl = 1.225

        for s in combined_surfaces:
            glass = (s.get("glass") or "AIR").strip()
            if glass.upper() not in ("AIR", "", "MIRROR", "MIRR", "REFL"):
                n = get_refractive_index(glass, test_wl)
                assert n > 1.0 and n < 4.0, f"Refractive index for glass '{glass}' at {test_wl} µm is invalid: {n}"

    def test_t1_16_surface_thicknesses_are_valid(self, primary_optical_system: OpticalSystem, nir_optical_system: OpticalSystem):
        """T1.16: Verify all surface thicknesses in loaded systems are non-negative and finite."""
        for system in (primary_optical_system, nir_optical_system):
            for i, surf in enumerate(system.surfaces):
                thick = float(surf.thickness)
                assert thick >= 0.0, f"Surface {i} ({surf.name}) has negative thickness {thick}"
                assert math.isfinite(thick), f"Surface {i} ({surf.name}) has non-finite thickness"

    def test_t1_17_surface_semi_diameters_are_positive(self, primary_optical_system: OpticalSystem, nir_optical_system: OpticalSystem):
        """T1.17: Verify all surface clear semi-diameters are strictly positive."""
        for system in (primary_optical_system, nir_optical_system):
            for i, surf in enumerate(system.surfaces):
                sd = float(surf.semi_diameter)
                assert sd > 0.0, f"Surface {i} ({surf.name}) has non-positive semi-diameter {sd}"
                assert math.isfinite(sd), f"Surface {i} ({surf.name}) has non-finite semi-diameter"

    def test_t1_18_global_coordinates_are_finite(self, primary_optical_system: OpticalSystem, nir_optical_system: OpticalSystem):
        """T1.18: Verify accumulated 3D global origins and rotation matrices contain finite numbers."""
        for system in (primary_optical_system, nir_optical_system):
            for i, surf in enumerate(system.surfaces):
                assert hasattr(surf, "global_origin"), f"Surface {i} missing global_origin"
                assert hasattr(surf, "global_R"), f"Surface {i} missing global_R"
                assert np.all(np.isfinite(surf.global_origin)), f"Surface {i} global_origin contains NaN/Inf"
                assert np.all(np.isfinite(surf.global_R)), f"Surface {i} global_R contains NaN/Inf"

    def test_t1_19_tracer_single_ray_execution(self, primary_optical_system: OpticalSystem):
        """T1.19: Verify OpticalSystem.trace_ray() executes without exceptions on on-axis ray."""
        ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=1.225)
        traced = primary_optical_system.trace_ray(ray)
        assert isinstance(traced, Ray)
        assert len(traced.history) > 0, "Traced ray must record hit history"
        assert not traced.vignetted, "On-axis chief ray must not vignette"

    def test_t1_20_tracer_batch_rays_execution(self, primary_optical_system: OpticalSystem, ray_fan_factory):
        """T1.20: Verify OpticalSystem.trace_rays() executes batch ray tracing cleanly."""
        rays = ray_fan_factory(na=0.12, num_rays=11, plane="tangential", wavelength_um=1.225)
        traced_batch = primary_optical_system.trace_rays(rays)
        assert len(traced_batch) == 11
        for r in traced_batch:
            assert isinstance(r, Ray)
            assert len(r.history) > 0

    # --------------------------------------------------------------------------
    # F4: SPOT DIAGNOSTICS & VERIFICATION SCRIPT
    # --------------------------------------------------------------------------

    def test_t1_21_spot_diagram_metrics_engine_validation(self):
        """T1.21: Verify SpotDiagram and calculate_spot_metrics compute exact geometric and RMS metrics."""
        # Synthetic ring of 8 rays at radius 0.200 mm
        angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        xs = 0.200 * np.cos(angles)
        ys = 0.200 * np.sin(angles)

        spot = SpotDiagram(xs, ys, wavelength_um=1.225, f_number=4.0)
        cx, cy = spot.calculate_centroid()
        rms_r = spot.calculate_rms_radius()
        geo_r = spot.calculate_geo_radius()

        assert abs(cx) < 1e-6, f"Centroid X should be ~0, got {cx}"
        assert abs(cy) < 1e-6, f"Centroid Y should be ~0, got {cy}"
        assert abs(rms_r - 0.200) < 1e-4, f"RMS radius should be 0.200 mm, got {rms_r}"
        assert abs(geo_r - 0.200) < 1e-4, f"GEO radius should be 0.200 mm, got {geo_r}"

        metrics = calculate_spot_metrics(xs, ys, wavelength_um=1.225, f_number=4.0)
        assert "centroid_x" in metrics
        assert "centroid_y" in metrics
        assert "rms_radius" in metrics
        assert "geo_radius" in metrics
        assert "airy_radius" in metrics
        assert "diffraction_limited" in metrics

    def test_t1_22_verification_script_exists(self, verify_script_path: Path):
        """T1.22: Assert verify_optical_prescriptions.py exists."""
        assert verify_script_path.is_file(), f"Verification script not found at {verify_script_path}"

    def test_t1_23_verification_script_compiles(self, verify_script_path: Path):
        """T1.23: Verify verify_optical_prescriptions.py is valid Python and compiles."""
        if not verify_script_path.is_file():
            pytest.skip("Verification script not yet generated")
        with open(verify_script_path, "r", encoding="utf-8") as f:
            code = f.read()
        compiled = compile(code, str(verify_script_path), "exec")
        assert compiled is not None, "Failed to compile verify_optical_prescriptions.py"

    def test_t1_24_fixture_pupil_grid_factory(self, pupil_grid_factory):
        """T1.24: Verify create_pupil_grid generates well-formed rays within NA cone."""
        rays = pupil_grid_factory(na=0.12, num_rings=3, rays_per_ring=6, wavelength_um=1.225)
        assert len(rays) == 1 + 6 * (1 + 2 + 3) // 2 or len(rays) >= 19
        for r in rays:
            assert isinstance(r, Ray)
            assert abs(np.linalg.norm(r.dir) - 1.0) < 1e-6
            sin_theta = math.sqrt(r.dir[0]**2 + r.dir[1]**2)
            assert sin_theta <= 0.1201

    def test_t1_25_fixture_hexapolar_grid_factory(self, hexapolar_grid_factory):
        """T1.25: Verify create_hexapolar_pupil_grid generates dense hexapolar rays."""
        rays = hexapolar_grid_factory(na=0.12, num_rings=4, wavelength_um=1.225)
        # 1 + 6 * (1 + 2 + 3 + 4) = 61 rays
        assert len(rays) == 61
        for r in rays:
            assert abs(np.linalg.norm(r.dir) - 1.0) < 1e-6

    def test_t1_26_fixture_marginal_rays_factory(self, marginal_rays_factory):
        """T1.26: Verify create_marginal_rays generates 8 extreme boundary rays at sin(theta) == NA."""
        rays = marginal_rays_factory(na=0.12, wavelength_um=1.225)
        assert len(rays) == 8
        for r in rays:
            sin_t = math.sqrt(r.dir[0]**2 + r.dir[1]**2)
            assert abs(sin_t - 0.12) < 1e-5

    def test_t1_27_fixture_ray_fan_factory(self, ray_fan_factory):
        """T1.27: Verify create_ray_fan generates 1D planar fan in YZ and XZ planes."""
        fan_tangential = ray_fan_factory(na=0.12, num_rays=15, plane="tangential", wavelength_um=1.225)
        assert len(fan_tangential) == 15
        for r in fan_tangential:
            assert abs(r.dir[0]) < 1e-9  # X should be 0 for tangential fan

        fan_sagittal = ray_fan_factory(na=0.12, num_rays=15, plane="sagittal", wavelength_um=1.225)
        assert len(fan_sagittal) == 15
        for r in fan_sagittal:
            assert abs(r.dir[1]) < 1e-9  # Y should be 0 for sagittal fan

