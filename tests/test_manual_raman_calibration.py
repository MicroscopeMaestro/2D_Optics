"""Regression checks for the corrected 1064-nm Raman spectrometer model."""

import json
from pathlib import Path

import numpy as np
import pytest

from optics_app.engine.lde import SurfaceStack
from optics_app.engine.diagnostics import ModelSanityValidator
from optics_app.engine.surface import SurfaceType
from optics_app.engine.mechanisms import _rotation_matrix_xyz, interaction_rotation, is_dmd
from optics_app.engine.ray import Ray
from optics_app.engine.system_config import SystemConfig
from optics_app.engine.tracer import OpticalSystem


MODEL = Path(__file__).resolve().parents[1] / "spectrometer_raman_50mm_100mm_manual.json"
WAVELENGTHS = (1.087, 1.225, 1.350)


def _load():
    raw = json.loads(MODEL.read_text(encoding="utf-8"))
    system = OpticalSystem.from_json(str(MODEL))
    return raw, system


def _find_indices(system):
    grating_idx = next(i for i, s in enumerate(system.surfaces) if s.surface_type == SurfaceType.GRATING or "GRATING" in getattr(s, "name", "").upper())
    dmd_idx = next(i for i, s in enumerate(system.surfaces) if is_dmd(s) or "DMD" in getattr(s, "name", "").upper())
    return grating_idx, dmd_idx


def test_catalog_identity_and_separate_laser_blaze_layout_wavelengths():
    raw, system = _load()
    grating_idx, dmd_idx = _find_indices(system)
    grating = system.surfaces[grating_idx]
    dmd = system.surfaces[dmd_idx]

    assert grating.catalog_metadata["part_number"] == "GR25-0610"
    assert grating.grating_lines_per_mm == 600
    assert grating.grating_blaze_wavelength_um == 1.0
    assert grating.grating_blaze_angle_deg == pytest.approx(17.45)
    assert grating.grating_primary_wavelength_um == 1.225
    assert (grating.aperture_x_half, grating.aperture_y_half) == (12.5, 12.5)

    assert raw["spectrometer_calibration"]["excitation_wavelength_um"] == 1.064
    assert raw["primary_wavelength"] == 1.225
    assert dmd.catalog_metadata["part_number"] == "DLP4500NIR"
    assert (dmd.aperture_x_half, dmd.aperture_y_half) == (3.0807, 4.9275)
    package_rotation = dmd.global_R @ _rotation_matrix_xyz(
        dmd.catalog_metadata["package_plane_offset_from_on_deg"]
    )
    angle = np.degrees(np.arccos(np.clip(np.dot(
        package_rotation[:, 2], interaction_rotation(dmd)[:, 2]
    ), -1.0, 1.0)))
    assert angle == pytest.approx(12.0, abs=1e-6)
    assert not raw["spectrometer_calibration"]["quantitative_throughput_calibrated"]
    assert not raw["spectrometer_calibration"]["physical_beam_dump_modeled"]


def test_full_configured_ray_fan_reaches_detector_only_with_dmd_on():
    raw, system = _load()
    grating_idx, dmd_idx = _find_indices(system)
    dmd = system.surfaces[dmd_idx]
    det_idx = len(system.surfaces) - 1

    config = SystemConfig()
    config.from_dict(raw["system_config"])
    rays = config.generate_input_rays(
        system.surfaces, wavelengths=list(WAVELENGTHS), num_rays=5, system=system
    )
    assert len(rays) >= 50

    dmd.mechanical_state = "ON"
    on_results = [system.trace_ray(ray) for ray in rays]
    for wavelength in WAVELENGTHS:
        band = [ray for ray in on_results if ray.wavelength == wavelength]
        assert band
        slit_passed = [ray for ray in band if any(hit["surface_index"] == dmd_idx for hit in ray.history)]
        assert len(slit_passed) >= 15
        assert all(not ray.vignetted and ray.history[-1]["surface_index"] == det_idx for ray in slit_passed)
        dmd_hits = [
            hit for ray in slit_passed for hit in ray.history if hit["surface_index"] == dmd_idx
        ]
        assert all(abs(hit["x_local"]) <= dmd.aperture_x_half for hit in dmd_hits)
        assert all(abs(hit["y_local"]) <= dmd.aperture_y_half for hit in dmd_hits)

    dmd.mechanical_state = "OFF"
    off_results = [system.trace_ray(ray) for ray in rays]
    assert all(ray.vignetted and ray.history[-1]["surface_index"] < det_idx for ray in off_results)

    dmd.mechanical_state = "ON"
    on_chief = system.trace_ray(Ray(pos=(0, 0, -1), dir=(0, 0, 1), wavelength=1.225))
    dmd.mechanical_state = "OFF"
    off_chief = system.trace_ray(Ray(pos=(0, 0, -1), dir=(0, 0, 1), wavelength=1.225))
    on_dir = next(hit["dir"] for hit in on_chief.history if hit["surface_index"] == dmd_idx)
    off_dir = next(hit["dir"] for hit in off_chief.history if hit["surface_index"] == dmd_idx)
    angle = np.degrees(np.arccos(np.clip(np.dot(on_dir, off_dir), -1.0, 1.0)))
    assert angle == pytest.approx(48.0, abs=1e-6)


def test_component_provenance_and_calibration_survive_project_save(tmp_path):
    raw, system = _load()
    grating_idx, dmd_idx = _find_indices(system)
    saved = tmp_path / "roundtrip.json"
    system.stack.save_to_json(str(saved), wavelengths=raw["wavelengths"])
    reopened = SurfaceStack.load_from_json(str(saved))["stack"]
    assert reopened.spectrometer_calibration == raw["spectrometer_calibration"]
    assert reopened.surfaces[grating_idx].catalog_metadata == system.surfaces[grating_idx].catalog_metadata
    assert reopened.surfaces[dmd_idx].catalog_metadata == system.surfaces[dmd_idx].catalog_metadata


def test_model_sanity_warns_on_uncalibrated_power_and_invalid_geometry():
    _, system = _load()
    grating_idx, dmd_idx = _find_indices(system)
    report = ModelSanityValidator.inspect_geometry(system.stack)
    codes = {item.code for item in report.items}
    assert "SPECTROMETER_THROUGHPUT_UNCALIBRATED" in codes
    assert "DMD_PHYSICAL_DUMP_UNMODELED" in codes
    assert "DMD_ACTIVE_APERTURE_OVERSIZED" not in codes
    assert "GRATING_LAYOUT_REFERENCE_MISMATCH" not in codes

    system.surfaces[grating_idx].grating_primary_wavelength_um = 1.064
    system.surfaces[dmd_idx].aperture_y_half = 15.0
    codes = {item.code for item in ModelSanityValidator.inspect_geometry(system.stack).items}
    assert "GRATING_LAYOUT_REFERENCE_MISMATCH" in codes
    assert "DMD_ACTIVE_APERTURE_OVERSIZED" in codes
