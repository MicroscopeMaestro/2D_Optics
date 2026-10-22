"""Unit and optical verification tests for the 40mm Achromat + Aspheric Lens Spectrometer.

Validates the spectrometer design featuring a Thorlabs AC254-040-C achromatic doublet
collimating relay followed by a Thorlabs ACL25416U aspheric condenser lens in the detection arm.
"""

import json
from pathlib import Path
import numpy as np
import pytest

from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from optics_app.engine.surface import SurfaceType
from optics_app.gui.visualizer_3d import generate_3d_rays


MODEL_PATHS = [
    Path(__file__).resolve().parents[1] / "spectrometer_raman_50mm_100mm_manual_40mm_aspheric.json",
    Path(__file__).resolve().parents[1] / "spectrometer_raman_40mm_relay_aspheric.json",
]
PRIMARY_WAVELENGTH = 1.225
RAMAN_WAVELENGTHS = (1.087, 1.225, 1.350)


def _load_model(filepath=MODEL_PATHS[0]):
    raw = json.loads(filepath.read_text(encoding="utf-8"))
    system = OpticalSystem.from_json(str(filepath))
    return raw, system


@pytest.mark.parametrize("model_path", MODEL_PATHS)
def test_40mm_aspheric_json_files_exist_and_load(model_path):
    """Verify that both target JSON files exist, load successfully, and contain 29 surfaces."""
    assert model_path.exists(), f"Target JSON file not found: {model_path}"
    raw, system = _load_model(model_path)
    assert len(system.surfaces) == 29
    assert system.surfaces[21].name == "DMD4500_Micromirrors"
    assert system.surfaces[23].name == "Relay_AC254-040-C_S1"
    assert system.surfaces[26].name == "Condenser_ACL25416U_S1"
    assert system.surfaces[28].name == "S25 (IMA)"


def test_achromat_and_aspheric_catalog_specifications():
    """Verify the Thorlabs AC254-040-C and ACL25416U prescriptions and catalog metadata."""
    raw, system = _load_model()
    
    # AC254-040-C relay
    ac_s1 = system.surfaces[23]
    ac_s2 = system.surfaces[24]
    ac_s3 = system.surfaces[25]
    
    assert ac_s1.name == "Relay_AC254-040-C_S1"
    assert ac_s1.radius == pytest.approx(143.9, abs=1e-4)
    assert ac_s1.thickness == pytest.approx(1.8, abs=1e-4)
    assert ac_s1.glass.upper() == "N-SF6HT"
    
    assert ac_s2.name == "Relay_AC254-040-C_S2"
    assert ac_s2.radius == pytest.approx(21.1, abs=1e-4)
    assert ac_s2.thickness == pytest.approx(10.0, abs=1e-4)
    assert ac_s2.glass.upper() == "N-LAK22"
    
    assert ac_s3.name == "Relay_AC254-040-C_S3"
    assert ac_s3.radius == pytest.approx(-24.4, abs=1e-4)
    assert ac_s3.thickness == pytest.approx(2.0, abs=1e-4) # relay air gap to condenser
    
    meta_ac = ac_s1.catalog_metadata
    assert meta_ac["manufacturer"] == "Thorlabs"
    assert meta_ac["part_number"] == "AC254-040-C"
    assert meta_ac["focal_length_mm"] == 40.0
    
    # ACL25416U aspheric condenser
    acl_s1 = system.surfaces[26]
    acl_s2 = system.surfaces[27]
    
    assert acl_s1.name == "Condenser_ACL25416U_S1"
    assert acl_s1.radius == pytest.approx(8.3, abs=1e-4)
    assert acl_s1.conic == pytest.approx(-0.6265, abs=1e-4)
    assert acl_s1.thickness == pytest.approx(14.0, abs=1e-4)
    assert acl_s1.glass.upper() == "B270"
    
    assert acl_s2.name == "Condenser_ACL25416U_S2"
    assert acl_s2.radius == pytest.approx(0.0, abs=1e-4)
    assert acl_s2.thickness == pytest.approx(4.5, abs=1e-4) # distance to detector
    
    # Calibration metadata
    cal = raw.get("spectrometer_calibration", {})
    assert cal["relay_part_number"] == "AC254-040-C"
    assert cal["condenser_part_number"] == "ACL25416U"


def test_dmd_reflection_and_detector_focal_spot():
    """Verify rays reflect off the DMD and focus tightly onto the detector."""
    _, system = _load_model()
    
    # Trace chief ray
    chief = system.trace_ray(Ray(pos=(0.0, 0.0, 0.0), dir=(0.0, 0.0, 1.0), wavelength=PRIMARY_WAVELENGTH))
    
    assert not chief.vignetted, "Chief ray must reach detector unvignetted"
    assert len(chief.history) == 29, f"Expected 29 surface hits, got {len(chief.history)}"
    
    # Verify hit on detector
    det_hit = chief.history[-1]
    assert det_hit["surface_index"] == 28
    assert abs(det_hit["x_local"]) < 0.1, f"Detector x_local: {det_hit['x_local']}"
    assert abs(det_hit["y_local"]) < 0.1, f"Detector y_local: {det_hit['y_local']}"


def test_dmd_on_off_modulation():
    """Verify DMD modulation with the 40mm achromat + aspheric arm."""
    _, system = _load_model()
    dmd = system.surfaces[21]
    
    # ON state: reaches detector
    dmd.mechanical_state = "ON"
    on_ray = system.trace_ray(Ray(pos=(0.0, 0.0, 0.0), dir=(0.0, 0.0, 1.0), wavelength=PRIMARY_WAVELENGTH))
    assert not on_ray.vignetted
    assert on_ray.history[-1]["surface_index"] == 28
    
    # OFF state: deflected away
    dmd.mechanical_state = "OFF"
    off_ray = system.trace_ray(Ray(pos=(0.0, 0.0, 0.0), dir=(0.0, 0.0, 1.0), wavelength=PRIMARY_WAVELENGTH))
    assert off_ray.vignetted
    assert off_ray.history[-1]["surface_index"] != 28


def test_visualizer_3d_ray_tracing_and_dmd_reflection():
    """Verify that generate_3d_rays successfully traces through the entire system and reflects from DMD."""
    raw, system = _load_model()
    rays = generate_3d_rays(system.stack)
    
    assert len(rays) > 0, "Visualizer must generate 3D ray bundle"
    
    # Verify all rays hit and reflect from DMD
    dmd_hits = sum(1 for r in rays if any("DMD" in system.stack.surfaces[h["surface_index"]].name for h in r.history))
    assert dmd_hits == len(rays), f"Expected all {len(rays)} rays to reflect from DMD, got {dmd_hits}"
    
    # Verify rays reach detector plane
    det_hits = sum(1 for r in rays if r.history[-1]["surface_index"] == 28)
    assert det_hits == len(rays), f"Expected all {len(rays)} rays to reach detector plane, got {det_hits}"


def test_json_save_load_fidelity(tmp_path):
    """Verify round-trip save and reload fidelity."""
    raw, system = _load_model()
    saved_file = tmp_path / "roundtrip_40mm.json"
    
    system.stack.save_to_json(str(saved_file), wavelengths=raw.get("wavelengths"))
    reloaded = SurfaceStack.load_from_json(str(saved_file))["stack"]
    
    assert len(reloaded.surfaces) == len(system.surfaces)
    assert reloaded.surfaces[23].name == "Relay_AC254-040-C_S1"
    assert reloaded.surfaces[26].name == "Condenser_ACL25416U_S1"
