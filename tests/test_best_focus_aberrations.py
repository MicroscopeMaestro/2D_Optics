"""Analytic and instrument checks for through-focus aberration analysis."""

from pathlib import Path

import numpy as np
import pytest

from optics_app.analysis.best_focus import FocusRay, analyze_focus_rays, analyze_system_best_focus
from optics_app.engine.tracer import OpticalSystem


MODEL = Path(__file__).resolve().parents[1] / "spectrometer_raman_50mm_100mm_manual.json"


def test_perfect_cone_finds_exact_least_confusion_plane():
    rays = [
        FocusRay(1.0, x, y, -x / 2.0, -y / 2.0, chief=(x == 0 and y == 0))
        for x in (-1.0, 0.0, 1.0) for y in (-1.0, 0.0, 1.0)
    ]
    result = analyze_focus_rays(rays, half_range_mm=3.0)
    assert result.valid
    assert result.best_shift_mm == pytest.approx(2.0)
    assert result.x_focus_shift_mm == pytest.approx(2.0)
    assert result.y_focus_shift_mm == pytest.approx(2.0)
    assert result.best.rms_radius_um == pytest.approx(0.0, abs=1e-5)
    assert result.nominal.rms_radius_um > 1000.0


def test_astigmatic_line_foci_and_circle_plane_are_distinct():
    rays = [
        FocusRay(1.225, x, y, -x / 2.0, -y / 4.0, chief=(x == 0 and y == 0))
        for x in (-1.0, 0.0, 1.0) for y in (-1.0, 0.0, 1.0)
    ]
    result = analyze_focus_rays(rays, primary_wavelength_um=1.225, half_range_mm=5.0)
    assert result.valid
    assert result.x_focus_shift_mm == pytest.approx(2.0)
    assert result.y_focus_shift_mm == pytest.approx(4.0)
    assert result.best_shift_mm == pytest.approx(2.4)
    assert result.circle_shift_mm == pytest.approx(8.0 / 3.0)
    assert result.best.rms_radius_um < result.nominal.rms_radius_um


def test_afocal_bundle_does_not_report_false_finite_focus():
    rays = [FocusRay(1.0, x, 0.0, 0.0, 0.0) for x in (-1.0, 0.0, 1.0)]
    result = analyze_focus_rays(rays)
    assert not result.valid
    assert result.best_shift_mm is None
    assert "afocal" in " ".join(result.warnings)


def test_manual_raman_model_focus_and_off_state():
    system = OpticalSystem.from_json(str(MODEL))
    result = analyze_system_best_focus(
        system, wavelengths_um=[1.087, 1.225, 1.350], pupil_samples=11,
        half_range_mm=10.0,
    )
    assert result.valid
    assert result.nominal.rms_radius_um >= result.best.rms_radius_um
    assert set(result.per_wavelength_best_mm) == {1.087, 1.225, 1.350}
    assert all(count >= 11 for count in result.traced_by_wavelength.values())
    assert result.reference_wavelength_um == 1.225
    assert result.longitudinal_color_mm is not None
    assert np.all(np.isfinite(result.rms_radial_um))

    system.surfaces[20].mechanical_state = "OFF"
    off = analyze_system_best_focus(system, wavelengths_um=[1.225], pupil_samples=11)
    assert not off.valid
    assert off.traced_by_wavelength[1.225] == 0


def test_manual_raman_focus_is_stable_with_denser_pupil_sampling():
    system = OpticalSystem.from_json(str(MODEL))
    original_detector_origin = system.surfaces[-1].global_origin.copy()
    coarse = analyze_system_best_focus(system, pupil_samples=11)
    dense = analyze_system_best_focus(system, pupil_samples=21)
    assert coarse.valid and dense.valid
    assert dense.best_shift_mm == pytest.approx(coarse.best_shift_mm, abs=0.01)
    assert dense.best.rms_radius_um == pytest.approx(coarse.best.rms_radius_um, abs=5.0)
    np.testing.assert_array_equal(system.surfaces[-1].global_origin, original_detector_origin)
    assert system.surfaces[-1].semi_diameter == pytest.approx(0.4)


def test_gui_exposes_best_focus_analysis_tab():
    from optics_app.gui.analysis_tabs import AnalysisTabsWidget, HAS_MPL_QT
    from optics_app.gui.qt_compat import QT_AVAILABLE, QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([]) if QT_AVAILABLE else None
    tabs = AnalysisTabsWidget()
    assert "Best Focus & Aberrations" in tabs.tabs
    if QT_AVAILABLE:
        assert tabs.tab_best_focus is not None
        if HAS_MPL_QT:
            system = OpticalSystem.from_json(str(MODEL))
            tabs.tab_best_focus.update_plot(system, [1.087, 1.225, 1.350], system.stack.system_config)
            assert tabs.tab_best_focus.last_result is not None
            assert tabs.tab_best_focus.last_result.valid
        tabs.close()


def test_ai_tool_reports_same_best_focus_as_engine():
    from optics_app.ai.tools import analyze_best_focus_aberrations

    system = OpticalSystem.from_json(str(MODEL))
    direct = analyze_system_best_focus(system, pupil_samples=11)
    tool_result = analyze_best_focus_aberrations(system=system, pupil_samples=11)
    assert tool_result["valid"]
    assert tool_result["best_focus_shift_mm"] == pytest.approx(direct.best_shift_mm)
    assert tool_result["best_rms_radius_um"] == pytest.approx(direct.best.rms_radius_um)
    assert tool_result["longitudinal_color_mm"] == pytest.approx(direct.longitudinal_color_mm)
