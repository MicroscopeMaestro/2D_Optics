import json
import math

import numpy as np

from optics_app.analysis.two_photon import (
    PockelsPBSConfig,
    SpectralResponse,
    TwoPhotonMicroscopeConfig,
    TwoPhotonMicroscopeSimulator,
    build_calibrated_template,
    build_two_photon_surface_template,
)
from optics_app.catalogs.thorlabs import get_catalog_sections, get_catalog_lens
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.surface import Surface
from optics_app.engine.undo_manager import UndoManager


def _signal(result, config):
    return result.detected_counts_per_second - config.pmt.dark_count_rate_hz


def test_pockels_pbs_energy_conservation_and_dump_rating():
    config = PockelsPBSConfig(voltage_v=50.0, half_wave_voltage_v=200.0, insertion_loss_fraction=0.03)
    microscope, dump, loss = config.split(0.8)
    assert microscope >= 0 and dump >= 0 and loss >= 0
    assert math.isclose(microscope + dump + loss, 0.8, rel_tol=1e-12)


def test_calibrated_inputs_block_missing_spectral_data():
    config = build_calibrated_template()
    config.pmt.quantum_efficiency = None
    result = TwoPhotonMicroscopeSimulator.run(config)
    assert not result.valid
    assert any(issue.code == "pmt_qe" and issue.level == "error" for issue in result.issues)


def test_two_photon_power_and_pulse_duration_scaling():
    low = build_calibrated_template()
    low.laser.average_power_w = 1e-5
    low.pmt.dark_count_rate_hz = 0.0
    r1 = TwoPhotonMicroscopeSimulator.run(low)

    high = TwoPhotonMicroscopeConfig.from_dict(low.to_dict())
    high.laser.average_power_w *= 2.0
    r2 = TwoPhotonMicroscopeSimulator.run(high)
    assert r1.valid and r2.valid
    assert np.isclose(_signal(r2, high) / _signal(r1, low), 4.0, rtol=0.01)

    broad = TwoPhotonMicroscopeConfig.from_dict(low.to_dict())
    broad.laser.transform_limit_fwhm_fs *= 2.0
    r3 = TwoPhotonMicroscopeSimulator.run(broad)
    assert np.isclose(_signal(r3, broad) / _signal(r1, low), 0.5, rtol=0.01)


def test_saturation_is_bounded_and_warned():
    config = build_calibrated_template()
    config.laser.average_power_w = 5.0
    result = TwoPhotonMicroscopeSimulator.run(config)
    assert 0.0 <= result.excitation_probability_per_molecule_per_pulse <= 1.0
    assert any(issue.code == "saturation" for issue in result.issues)


def test_blocking_filter_leaves_only_dark_counts():
    config = build_calibrated_template()
    config.emission_filter_transmission = SpectralResponse(
        wavelength_nm=[450.0, 750.0], values=[0.0, 0.0], units="fraction", source_url="synthetic-test", retrieved="2026-10-03"
    )
    result = TwoPhotonMicroscopeSimulator.run(config)
    assert result.valid
    assert result.spectral_detection_efficiency == 0.0
    assert result.detected_counts_per_second == config.pmt.dark_count_rate_hz


def test_scan_profile_is_centered_on_point_emitter():
    config = build_calibrated_template()
    config.emitter.x_um = 0.8
    result = TwoPhotonMicroscopeSimulator.run(config)
    peak_position = result.scan_positions_um[int(np.argmax(result.scan_counts_per_second))]
    spacing = result.scan_positions_um[1] - result.scan_positions_um[0]
    assert abs(peak_position - 0.8) <= spacing


def test_vectorial_solver_is_finite_and_normalized():
    config = build_calibrated_template()
    config.objective.numerical_aperture = 0.15
    config.objective.entrance_pupil_diameter_mm = 20.0
    config.laser.beam_diameter_mm = 20.0
    result = TwoPhotonMicroscopeSimulator.run(config, mode="vectorial")
    assert result.valid
    assert result.focus is not None
    assert np.isfinite(result.focus.xy_intensity).all()
    assert np.isclose(np.max(result.focus.xy_intensity), 1.0, atol=1e-9)
    assert result.focus.peak_intensity_w_m2 > 0.0


def test_template_routes_and_component_uids_are_valid():
    stack = build_two_photon_surface_template()
    uids = [surface.component_uid for surface in stack.surfaces]
    assert all(uids)
    assert not stack.optical_routes.validate(uids)
    assert len(list(stack.optical_routes.all_routes())) == 3
    pockels_uids = {s.component_uid for s in stack.surfaces if s.group_name == "Pockels Cell"}
    assert len(pockels_uids) == 1


def test_schema_v4_roundtrip_and_legacy_v3_load(tmp_path):
    stack = build_two_photon_surface_template()
    path = tmp_path / "two_photon.json"
    stack.save_to_json(str(path))
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["version"] == 4
    assert "optical_routes" in raw and "two_photon_config" in raw
    loaded = SurfaceStack.load_from_json(str(path))["stack"]
    assert loaded.two_photon_config.objective.part_number == "TL15X-2P"
    assert not loaded.optical_routes.validate(s.component_uid for s in loaded.surfaces)

    legacy_path = tmp_path / "legacy.json"
    legacy_path.write_text(json.dumps({"version": 3, "surfaces": [{"name": "S0"}], "wavelengths": [0.8]}), encoding="utf-8")
    legacy = SurfaceStack.load_from_json(str(legacy_path))["stack"]
    assert len(legacy.surfaces) == 1
    assert legacy.surfaces[0].component_uid


def test_uid_survives_replacement_and_undo_redo_snapshots():
    stack = SurfaceStack()
    stack.insert_singlet(-1, name="Target", air_gap_after=20.0)
    old_uid = stack.surfaces[0].component_uid
    stack.two_photon_config = build_calibrated_template()
    stack.optical_routes = stack.two_photon_config.routes
    history = UndoManager()
    history.push_state(stack, description="before")
    stack.replace_element_with_catalog(0, "AC254-050-C")
    assert {s.component_uid for s in stack.surfaces[:3]} == {old_uid}
    history.push_state(stack, description="after")
    restored = history.undo(stack)
    assert restored is not None
    assert restored.stack_auxiliary_dict["two_photon_config"]["objective"]["part_number"] == "TL15X-2P"


def test_multiphoton_catalog_sections_are_visible_and_replaceable():
    sections = get_catalog_sections()
    assert "microscope_objectives" in sections
    assert "multiphoton_scan_tube_lenses" in sections
    objective = get_catalog_lens("TL15X-2P")
    assert objective is not None
    assert objective["numerical_aperture"] == 0.70
    stack = SurfaceStack()
    stack.add_surface(Surface(name="Anchor", thickness=10.0))
    indices = stack.insert_catalog_lens(0, "TL15X-2P")
    assert indices
    assert stack.surfaces[indices[0]].group_name == "TL15X-2P"

