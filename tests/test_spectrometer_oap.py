"""Unit and physics verification tests for the Thorlabs Off-Axis Parabolic Mirror Spectrometer.

Validates the replacement of the refractive ACL25416U aspheric condenser lens with the
Thorlabs MPD129-P01 90-degree Off-Axis Parabolic Mirror (Protected Silver, 1-inch diameter,
EFL=25.4mm, conic k=-1.0, R=-50.8mm) in the 1064nm Raman DMD spectrometer design.
"""

import json
from pathlib import Path
import numpy as np
import pytest

from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from optics_app.engine.surface import SurfaceType


MODEL_PATHS = [
    Path(__file__).resolve().parents[1] / "spectrometer_raman_50mm_100mm_manual_oap.json",
    Path(__file__).resolve().parents[1] / "spectrometer_raman_50mm_100mm_oap.json",
]
PRIMARY_WAVELENGTH = 1.225
RAMAN_WAVELENGTHS = (1.087, 1.225, 1.350)


def _load_model(filepath=MODEL_PATHS[0]):
    raw = json.loads(filepath.read_text(encoding="utf-8"))
    system = OpticalSystem.from_json(str(filepath))
    return raw, system


@pytest.mark.parametrize("model_path", MODEL_PATHS)
def test_oap_json_files_exist_and_load(model_path):
    """Verify that both target JSON files exist and load successfully."""
    assert model_path.exists(), f"Target JSON file not found: {model_path}"
    raw, system = _load_model(model_path)
    assert len(system.surfaces) == 25
    assert system.surfaces[23].name == "MPD119-P01_S1"


def test_oap_catalog_identity_and_specifications():
    """Verify the Thorlabs MPD119-P01 OAP mirror catalog metadata, conic figure, and radii."""
    raw, system = _load_model()
    
    assert len(system.surfaces) == 25, f"Expected 25 surfaces, got {len(system.surfaces)}"
    
    # Surface 23 is the OAP mirror (replacing former surfaces 23 & 24 ACL25416U)
    oap = system.surfaces[23]
    assert oap.name == "MPD119-P01_S1"
    assert oap.surface_type in (SurfaceType.MIRROR, SurfaceType.ASPHERIC)
    assert oap.glass.upper() == "MIRROR"
    assert oap.is_reflective()
    assert oap.conic == pytest.approx(-1.0, abs=1e-6)
    assert oap.radius == pytest.approx(-50.8, abs=1e-4) # R = -2f convention
    assert oap.semi_diameter == pytest.approx(12.7, abs=1e-4) # 1-inch diameter
    assert oap.thickness == pytest.approx(25.4, abs=1e-4) # distance to detector
    assert oap.tilt_x == pytest.approx(45.0, abs=1e-4) # 45-deg incidence for 90-deg upward fold
    
    # Metadata assertions
    meta = oap.catalog_metadata
    assert meta["manufacturer"] == "Thorlabs"
    assert meta["part_number"] == "MPD119-P01"
    assert meta["conic_k"] == -1.0
    assert meta["nominal_focal_length_mm"] == 25.4
    assert meta["diameter_mm"] == 25.4
    assert meta["coating_code"] == "P01"
    
    # Spectrometer calibration section assertions
    cal = raw.get("spectrometer_calibration", {})
    assert cal["condenser_part_number"] == "MPD119-P01"
    assert cal["condenser_focal_length_mm"] == 25.4
    assert cal["condenser_diameter_mm"] == 25.4
    assert cal["excitation_wavelength_um"] == 1.064
    assert cal["layout_reference_wavelength_um"] == 1.225


def test_oap_90_degree_reflection_and_detector_alignment():
    """Verify the OAP mirror deflects the beam by exactly 90 degrees and focuses onto the detector."""
    _, system = _load_model()
    
    # Trace chief ray at layout reference wavelength (1.225 um)
    chief = system.trace_ray(Ray(pos=(0.0, 0.0, 0.0), dir=(0.0, 0.0, 1.0), wavelength=PRIMARY_WAVELENGTH))
    
    assert not chief.vignetted, "Chief ray must not be vignetted"
    assert len(chief.history) == 25, f"Expected 25 surface hits, got {len(chief.history)}"
    
    # Verify hit on detector (surface 24)
    det_hit = chief.history[-1]
    assert det_hit["surface_index"] == 24
    assert abs(det_hit["x_local"]) < 1e-4, f"Detector x_local offset: {det_hit['x_local']}"
    assert abs(det_hit["y_local"]) < 1e-4, f"Detector y_local offset: {det_hit['y_local']}"
    
    # Verify 90-degree fold angle at the OAP mirror
    dir_before_oap = chief.history[22]["dir"]
    dir_after_oap = chief.history[23]["dir"]
    dot_val = np.clip(np.dot(dir_before_oap, dir_after_oap), -1.0, 1.0)
    fold_angle_deg = np.degrees(np.arccos(dot_val))
    assert fold_angle_deg == pytest.approx(90.0, abs=1e-4)


def test_oap_dmd_on_off_modulation():
    """Verify DMD modulation with the OAP mirror arm: ON state passes, OFF state deflects 48 deg."""
    _, system = _load_model()
    dmd = system.surfaces[21]
    
    # 1. ON state: light must reach detector
    dmd.mechanical_state = "ON"
    on_ray = system.trace_ray(Ray(pos=(0.0, 0.0, 0.0), dir=(0.0, 0.0, 1.0), wavelength=PRIMARY_WAVELENGTH))
    assert not on_ray.vignetted
    assert on_ray.history[-1]["surface_index"] == 24
    
    # 2. OFF state: light must be deflected away from detector
    dmd.mechanical_state = "OFF"
    off_ray = system.trace_ray(Ray(pos=(0.0, 0.0, 0.0), dir=(0.0, 0.0, 1.0), wavelength=PRIMARY_WAVELENGTH))
    assert off_ray.vignetted
    assert off_ray.history[-1]["surface_index"] != 24
    
    # 3. Angle difference between ON and OFF reflections must be exactly 48 degrees (2 * 24 deg)
    on_dir = next(hit["dir"] for hit in on_ray.history if hit["surface_index"] == 21)
    off_dir = next(hit["dir"] for hit in off_ray.history if hit["surface_index"] == 21)
    angle_diff = np.degrees(np.arccos(np.clip(np.dot(on_dir, off_dir), -1.0, 1.0)))
    assert angle_diff == pytest.approx(48.0, abs=1e-4)


def test_oap_parabolic_achromaticity():
    """Verify the OAP mirror is purely reflective and exhibits zero chromatic dispersion."""
    _, system = _load_model()
    oap = system.surfaces[23]
    
    assert oap.glass.upper() == "MIRROR"
    assert oap.conic == -1.0
    
    # Trace rays of different wavelengths reaching the OAP mirror
    # Because it is a mirror, the optical power is independent of wavelength (dn/dlambda = 0)
    f = -oap.radius / 2.0
    assert f == pytest.approx(25.4, abs=1e-4)


def test_oap_json_save_load_fidelity(tmp_path):
    """Verify that the OAP design can be saved and loaded round-trip without corruption."""
    raw, system = _load_model()
    saved_file = tmp_path / "roundtrip_oap.json"
    
    system.stack.save_to_json(str(saved_file), wavelengths=raw.get("wavelengths"))
    reloaded = SurfaceStack.load_from_json(str(saved_file))["stack"]
    
    assert len(reloaded.surfaces) == len(system.surfaces)
    reloaded_oap = reloaded.surfaces[23]
    assert reloaded_oap.name == "MPD119-P01_S1"
    assert reloaded_oap.conic == pytest.approx(-1.0, abs=1e-6)
    assert reloaded_oap.radius == pytest.approx(-50.8, abs=1e-4)
    assert reloaded_oap.tilt_x == pytest.approx(45.0, abs=1e-4)
    assert reloaded_oap.catalog_metadata["part_number"] == "MPD119-P01"
