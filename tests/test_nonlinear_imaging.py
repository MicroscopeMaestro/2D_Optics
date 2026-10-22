import math

import numpy as np
import pytest

from optics_app.analysis.nonlinear_imaging import (
    NonlinearImagingConfig, NonlinearImagingSimulator,
    build_nonlinear_surface_template,
)
from optics_app.analysis.two_photon import SpectralResponse
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.models import SpectrometerSystem
from optics_app.engine.ray import Ray
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.undo_manager import UndoManager
from optics_app.gui.qt_compat import QT_AVAILABLE, QtWidgets
from optics_app.gui.visualizer_3d import HAS_MPL_QT, Visualizer3DWidget
from optics_app.ai.tools import simulate_nonlinear_imaging


def _curve(start, stop, value=1.0):
    return SpectralResponse([start, stop], [value, value], source_url="test://measured", retrieved="2026-10-03")


def _calibrated():
    cfg = build_nonlinear_surface_template().nonlinear_imaging_config
    cfg.pump.average_power_w = 1e-4
    cfg.stokes.average_power_w = 1e-4
    cfg.combiner.pump_to_sample_fraction = 0.9
    cfg.combiner.stokes_to_sample_fraction = 0.9
    cfg.combiner.calibration_source_url = "test://measured-combiner"
    cfg.combiner.calibration_date = "2026-10-03"
    cfg.sample.refractive_index = cfg.objective.immersion_refractive_index
    cfg.pump.beam_diameter_mm = cfg.objective.entrance_pupil_diameter_mm
    cfg.stokes.beam_diameter_mm = cfg.objective.entrance_pupil_diameter_mm
    cfg.pump_path_transmission = _curve(790, 810)
    cfg.stokes_path_transmission = _curve(1030, 1050)
    cfg.objective.transmission = _curve(700, 1100)
    cfg.detection_transmission = _curve(400, 1100)
    cfg.pmt.quantum_efficiency = _curve(400, 1100, 0.2)
    cfg.pmt.part_number = "Test photon-counter"
    cfg.pmt.max_count_rate_hz = 1e15
    cfg.collection_fraction = 0.1
    cfg.sample.d_eff_pm_per_v = 1e-5
    cfg.sample.chi3_nonres_m2_per_v2 = 1e-28
    cfg.sample.chi3_resonant_m2_per_v2 = 1e-28
    cfg.sample.raman_shift_cm1 = 1e7 * (1 / 800 - 1 / 1040)
    cfg.sample.raman_linewidth_cm1 = 100.0
    cfg.sample.raman_gain_m_per_w = 1e-16
    cfg.sample.phase_mismatch_sfg_m1 = 0
    cfg.sample.phase_mismatch_dfg_m1 = 0
    cfg.sample.phase_mismatch_cars_m1 = 0
    cfg.sample.source_url = "test://measured-sample"
    cfg.sample.measurement_conditions = "test medium, room temperature"
    cfg.sample.relative_uncertainty = 0.1
    return cfg


def test_generated_wavelengths_and_srs_energy_balance():
    cfg = _calibrated()
    result = NonlinearImagingSimulator.run(cfg)
    assert result.valid
    assert math.isclose(result.channels["SFG"].wavelength_nm, 452.173913, rel_tol=1e-6)
    assert math.isclose(result.channels["DFG"].wavelength_nm, 3466.666667, rel_tol=1e-6)
    assert math.isclose(result.channels["CARS"].wavelength_nm, 650.0, rel_tol=1e-6)
    srs = result.channels["SRS"]
    assert srs.generated_power_w is None
    assert srs.stokes_gain_w > 0
    assert math.isclose(srs.pump_loss_w / srs.stokes_gain_w, 1040 / 800, rel_tol=1e-10)
    assert math.isclose(result.rejected_power_w +
                        cfg.pump.average_power_w * cfg.combiner.pump_to_sample_fraction +
                        cfg.stokes.average_power_w * cfg.combiner.stokes_to_sample_fraction,
                        cfg.pump.average_power_w + cfg.stokes.average_power_w, rel_tol=1e-12)


def test_power_scaling_and_delay_suppression():
    cfg = _calibrated()
    baseline = NonlinearImagingSimulator.run(cfg)
    high = NonlinearImagingConfig.from_dict(cfg.to_dict())
    high.pump.average_power_w *= 2
    doubled = NonlinearImagingSimulator.run(high)
    assert math.isclose(doubled.channels["SFG"].generated_power_w /
                        baseline.channels["SFG"].generated_power_w, 2, rel_tol=0.02)
    assert math.isclose(doubled.channels["CARS"].generated_power_w /
                        baseline.channels["CARS"].generated_power_w, 4, rel_tol=0.03)
    late = NonlinearImagingConfig.from_dict(cfg.to_dict())
    late.combiner.delay_fs = 1000
    delayed = NonlinearImagingSimulator.run(late)
    assert delayed.channels["SFG"].generated_power_w < baseline.channels["SFG"].generated_power_w * 1e-4
    assert any(issue.code == "temporal_overlap" for issue in delayed.issues)


def test_missing_calibration_and_pmt_out_of_band_are_explicit():
    cfg = _calibrated()
    cfg.sample.d_eff_pm_per_v = None
    cfg.pmt.quantum_efficiency = _curve(400, 700, 0.2)
    result = NonlinearImagingSimulator.run(cfg)
    assert not result.channels["SFG"].valid
    assert any(issue.code == "missing_d_eff" for issue in result.channels["SFG"].issues)
    assert result.channels["CARS"].valid
    assert result.channels["CARS"].pmt_counts_per_second is not None
    assert result.channels["DFG"].pmt_counts_per_second is None
    assert result.channels["SRS"].pmt_counts_per_second is None
    assert any(issue.code == "pmt_qe_range" for issue in result.channels["SRS"].issues)


def test_raman_resonance_sample_scan_and_voltage_output_pmt():
    cfg = _calibrated()
    on = NonlinearImagingSimulator.run(cfg)
    off = NonlinearImagingConfig.from_dict(cfg.to_dict())
    off.sample.raman_shift_cm1 += 3000
    detuned = NonlinearImagingSimulator.run(off)
    assert on.channels["SRS"].stokes_gain_w > detuned.channels["SRS"].stokes_gain_w
    shifted = NonlinearImagingConfig.from_dict(cfg.to_dict())
    shifted.sample_x_um = 1.0
    scan = NonlinearImagingSimulator.run(shifted)
    assert abs(scan.scan_positions_um[np.argmax(scan.scan_profiles["CARS"])] - 1.0) < 0.1
    assert scan.channels["CARS"].generated_power_w < on.channels["CARS"].generated_power_w
    voltage = NonlinearImagingConfig.from_dict(cfg.to_dict())
    voltage.pmt.part_number = "H15460-40"
    voltage_result = NonlinearImagingSimulator.run(voltage).channels["CARS"]
    assert voltage_result.pmt_photoelectron_rate_hz is not None
    assert voltage_result.pmt_counts_per_second is None
    assert any(issue.code == "voltage_output_pmt" for issue in voltage_result.issues)


def test_routes_project_roundtrip_and_undo(tmp_path):
    stack = build_nonlinear_surface_template()
    trace = OpticalSystem(stack).trace_ray(Ray(pos=(0, 0, -1), dir=(0, 0, 1), wavelength=0.8))
    assert trace.history[-1]["surface_index"] == next(i for i, surface in enumerate(stack.surfaces) if surface.group_name == "Sample")
    assert not trace.vignetted
    known = [surface.component_uid for surface in stack.surfaces]
    assert not stack.optical_routes.validate(known)
    path = tmp_path / "nonlinear.json"
    stack.save_to_json(str(path))
    loaded = SurfaceStack.load_from_json(str(path))["stack"]
    assert loaded.nonlinear_imaging_config.to_dict() == stack.nonlinear_imaging_config.to_dict()
    assert loaded.optical_routes.to_dict() == stack.optical_routes.to_dict()
    assert [surface.component_uid for surface in loaded.surfaces] == known
    system = SpectrometerSystem(surface_stack=loaded)
    system_copy = SpectrometerSystem.from_dict(system.to_dict())
    assert system_copy.surface_stack.nonlinear_imaging_config.to_dict() == loaded.nonlinear_imaging_config.to_dict()
    undo = UndoManager()
    undo.push_state(loaded, description="before")
    loaded.nonlinear_imaging_config.combiner.delay_fs = 125
    prior = undo.undo(loaded)
    assert prior.stack_auxiliary_dict["nonlinear_imaging_config"]["combiner"]["delay_fs"] == 0
    restored = undo.redo(loaded)
    assert restored.stack_auxiliary_dict["nonlinear_imaging_config"]["combiner"]["delay_fs"] == 125


def test_ai_analysis_returns_channels_and_blocking_inputs():
    payload = simulate_nonlinear_imaging(system=build_nonlinear_surface_template())
    assert payload["status"] == "BLOCKED"
    assert math.isclose(payload["channels"]["CARS"]["wavelength_nm"], 650.0)
    assert any(issue["code"] == "combiner_calibration" for issue in payload["issues"])


@pytest.mark.skipif(not (QT_AVAILABLE and HAS_MPL_QT), reason="Qt Matplotlib unavailable")
def test_3d_viewer_shows_both_inputs_rejection_and_generated_routes():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    widget = Visualizer3DWidget()
    try:
        widget.update_layout(surface_stack=build_nonlinear_surface_template(), rays=[])
        names = {line.get_label() for line in widget.ax.lines}
        assert "Route: pump" in names
        assert "Route: stokes" in names
        assert "Route: pump rejection" in names
        assert "Route: SFG" in names
        assert "Route: DFG (IR)" in names
        assert "Route: CARS to PMT" in names
        assert "Route: SRS modulation (Stokes)" in names
    finally:
        widget.close()
