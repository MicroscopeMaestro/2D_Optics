"""Physics, GUI, probe, and persistence coverage for femtosecond pulses."""

import math
import os
import tempfile

import numpy as np
import pytest

from optics_app.analysis.pulse_width import (
    PulseAnalysisConfig,
    PulseWidthAnalysis,
    gaussian_pulse_fwhm_fs,
)
from optics_app.ai.nemotron_client import parse_optical_command_heuristics
from optics_app.ai.tools import analyze_femtosecond_pulse
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.gui.analysis_tabs import AnalysisTabsWidget
from optics_app.gui.lde_table import LensDataTableModel
from optics_app.gui.qt_compat import QT_AVAILABLE, QtWidgets


if QT_AVAILABLE and hasattr(QtWidgets, "QApplication"):
    _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def fused_silica_stack(thickness_mm: float = 10.0) -> SurfaceStack:
    stack = SurfaceStack()
    stack.add_surface(Surface(name="Source", surface_type=SurfaceType.OBJECT, thickness=10.0, glass="AIR"))
    stack.add_surface(Surface(name="Fused Silica Front", thickness=thickness_mm, glass="F_SILICA", semi_diameter=20.0))
    stack.add_surface(Surface(name="Fused Silica Back", thickness=10.0, glass="AIR", semi_diameter=20.0))
    stack.add_surface(Surface(name="Pulse Detector", surface_type=SurfaceType.DETECTOR, semi_diameter=20.0))
    return stack


def test_gaussian_gdd_broadening_matches_closed_form():
    tau_fs = 30.0
    gdd_fs2 = 1000.0
    expected = tau_fs * math.sqrt(1.0 + (4.0 * math.log(2.0) * gdd_fs2 / tau_fs**2) ** 2)
    calculated = gaussian_pulse_fwhm_fs(tau_fs, gdd_fs2, include_tod=False)
    assert calculated == pytest.approx(expected, rel=3e-4)


def test_fused_silica_gdd_at_800_nm_is_physical():
    result = PulseWidthAnalysis.analyze(
        fused_silica_stack(1.0),
        PulseAnalysisConfig(center_wavelength_nm=800.0, transform_limit_fwhm_fs=100.0, include_tod=False),
    )
    exit_probe = result.probe(2)
    assert exit_probe is not None and exit_probe.valid
    # Published fused-silica values near 800 nm are about +36 fs²/mm.
    assert 34.0 <= exit_probe.material_gdd_fs2 <= 38.0
    assert abs(exit_probe.geometric_gdd_fs2) < 0.1


def test_negative_prechirp_compresses_at_a_downstream_probe():
    unchirped = PulseWidthAnalysis.analyze(
        fused_silica_stack(10.0),
        PulseAnalysisConfig(center_wavelength_nm=800.0, transform_limit_fwhm_fs=30.0, include_tod=False),
    )
    glass_gdd = unchirped.probe(2).system_gdd_fs2
    compressed = PulseWidthAnalysis.analyze(
        fused_silica_stack(10.0),
        PulseAnalysisConfig(
            center_wavelength_nm=800.0,
            transform_limit_fwhm_fs=30.0,
            initial_gdd_fs2=-glass_gdd,
            include_tod=False,
            probe_indices=[0, 2],
        ),
    )
    source = compressed.probe(0)
    exit_probe = compressed.probe(2)
    assert source.pulse_fwhm_fs > 40.0
    assert exit_probe.pulse_fwhm_fs < source.pulse_fwhm_fs
    assert exit_probe.pulse_fwhm_fs == pytest.approx(30.0, abs=0.1)


def test_analysis_configuration_and_probe_indices_round_trip_in_project():
    stack = fused_silica_stack()
    config = PulseAnalysisConfig(center_wavelength_nm=1030.0, transform_limit_fwhm_fs=45.0, probe_indices=[1, 3])
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "pulse_project.json")
        stack.save_to_json(path, analysis_config={"pulse_width_analysis": config.to_dict()})
        loaded = SurfaceStack.load_from_json(path)
    restored = PulseAnalysisConfig.from_dict(loaded["analysis_config"]["pulse_width_analysis"])
    assert restored.center_wavelength_nm == 1030.0
    assert restored.transform_limit_fwhm_fs == 45.0
    assert restored.probe_indices == [1, 3]


def test_analysis_tabs_expose_femtosecond_probe_configuration():
    tabs = AnalysisTabsWidget(model=LensDataTableModel(stack=fused_silica_stack()))
    assert "Femtosecond Pulse" in tabs.tabs
    tabs.set_analysis_config({
        "pulse_width_analysis": {
            "center_wavelength_nm": 800.0,
            "transform_limit_fwhm_fs": 25.0,
            "initial_gdd_fs2": -250.0,
            "probe_indices": [2],
        }
    })
    config = tabs.get_analysis_config()["pulse_width_analysis"]
    assert config["transform_limit_fwhm_fs"] == 25.0
    assert config["initial_gdd_fs2"] == -250.0
    assert config["probe_indices"] == [2]


def test_ai_command_and_tool_report_probe_dispersion():
    parsed = parse_optical_command_heuristics("analyze pulse width for an 800 nm 30 fs pulse with GDD -250 fs2")
    assert parsed["name"] == "analyze_femtosecond_pulse"
    assert parsed["arguments"]["center_wavelength_nm"] == 800.0
    assert parsed["arguments"]["transform_limit_fwhm_fs"] == 30.0
    assert parsed["arguments"]["initial_gdd_fs2"] == -250.0

    report = analyze_femtosecond_pulse(
        800.0,
        30.0,
        initial_gdd_fs2=-250.0,
        probe_surfaces=["Fused Silica Back"],
        system=fused_silica_stack(10.0),
    )
    assert "Fused Silica Back" in report
    assert "Total GDD" in report
    assert "Spatial chirp" in report


def test_main_window_save_load_restores_pulse_probes(tmp_path, monkeypatch):
    from optics_app.gui.main_window import MainWindow

    window = MainWindow()
    path = tmp_path / "pulse-probes.json"
    try:
        window.model.set_surface_stack(fused_silica_stack())
        window.analysis_tabs.set_analysis_config({
            "pulse_width_analysis": {
                "center_wavelength_nm": 920.0,
                "transform_limit_fwhm_fs": 55.0,
                "initial_gdd_fs2": -800.0,
                "probe_indices": [1, 2],
            }
        })
        assert window.on_save_project(str(path))

        window.analysis_tabs.set_analysis_config({"pulse_width_analysis": {}})
        monkeypatch.setattr(window, "on_trace_rays", lambda: True)
        assert window.on_load_project(str(path))
        restored = window.analysis_tabs.get_analysis_config()["pulse_width_analysis"]
        assert restored["center_wavelength_nm"] == 920.0
        assert restored["transform_limit_fwhm_fs"] == 55.0
        assert restored["initial_gdd_fs2"] == -800.0
        assert restored["probe_indices"] == [1, 2]
    finally:
        window.close()
