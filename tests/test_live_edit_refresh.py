"""Regression checks for fast, coherent parameter-edit refreshes."""

from pathlib import Path

import numpy as np
import pytest

from optics_app.engine.lde import SurfaceStack
from optics_app.engine.ray import Ray
from optics_app.engine.tracer import OpticalSystem
from optics_app.gui.lde_table import LensDataTableModel
from optics_app.gui.main_window import MainWindow
from optics_app.gui.qt_compat import QT_AVAILABLE, QtWidgets


MODEL = Path(__file__).resolve().parents[1] / "spectrometer_raman_50mm_100mm_manual.json"


def test_batched_grating_traces_match_individual_traces():
    system = OpticalSystem.from_json(str(MODEL))
    wavelengths = (1.087, 1.225, 1.350)
    batch = system.trace_rays([Ray(pos=(0, 0, -1), dir=(0, 0, 1), wavelength=wl)
                               for wl in wavelengths])
    singles = [system.trace_ray(Ray(pos=(0, 0, -1), dir=(0, 0, 1), wavelength=wl))
               for wl in wavelengths]
    for batched, single in zip(batch, singles):
        assert len(batched.history) == len(single.history)
        assert batched.vignetted == single.vignetted
        for batch_hit, single_hit in zip(batched.history, single.history):
            assert batch_hit["surface_index"] == single_hit["surface_index"]
            assert np.allclose(batch_hit["dir"], single_hit["dir"], atol=1e-12)
            assert np.allclose(
                [batch_hit[key] for key in ("x_global", "y_global", "z_global", "intensity")],
                [single_hit[key] for key in ("x_global", "y_global", "z_global", "intensity")],
                atol=1e-12,
            )


def test_model_refresh_keeps_loaded_project_metadata():
    stack = SurfaceStack.load_from_json(str(MODEL))["stack"]
    model = LensDataTableModel()
    model.set_surface_stack(stack)
    model._sync_stack_from_data()
    refreshed = model.get_surface_stack()
    assert refreshed is stack
    assert refreshed.optical_routes is stack.optical_routes
    assert refreshed.system_config is stack.system_config
    assert refreshed.spectrometer_calibration is stack.spectrometer_calibration


@pytest.mark.skipif(not QT_AVAILABLE, reason="Qt unavailable")
def test_component_edit_signals_schedule_one_trace(monkeypatch):
    from PySide6.QtTest import QTest

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    calls = []
    monkeypatch.setattr(MainWindow, "on_trace_rays", lambda self, *args, **kwargs: calls.append(1))
    window = MainWindow()
    try:
        calls.clear()  # Ignore initial construction trace.
        window.component_editor_dock._emit_change()
        QTest.qWait(80)
        assert len(calls) == 1
    finally:
        window.close()
