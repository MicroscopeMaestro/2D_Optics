"""
Unit tests for Zemax OpticStudio First Order Optical Data calculation and reporting.
"""

import pytest
import math
import numpy as np

from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.first_order import calculate_first_order_data, format_zemax_first_order_report
from optics_app.gui.qt_compat import QT_AVAILABLE, QtWidgets


def test_first_order_data_singlet():
    """Verify first-order optical data for a biconvex N-BK7 singlet."""
    s1 = Surface(name="Front", radius=51.68, thickness=6.0, glass="N-BK7", semi_diameter=12.5)
    s2 = Surface(name="Rear", radius=-51.68, thickness=48.0, glass="AIR", semi_diameter=12.5)
    s_det = Surface(name="Detector", surface_type=SurfaceType.DETECTOR)

    surfaces = [s1, s2, s_det]
    data = calculate_first_order_data(surfaces, wavelength_um=0.5876, epd=25.0)

    # Paraxial EFL for R1=51.68, R2=-51.68, d=6, n=1.5168 is ~51.01 mm
    assert not data["is_afocal"]
    assert 50.5 < data["effl"] < 51.5
    assert 48.0 < data["bfl"] < 50.0
    assert 48.0 < data["ffl"] < 50.0
    assert data["epd"] == 25.0
    assert 1.9 < data["f_number"] < 2.1
    assert data["na"] > 0.2
    assert data["totr"] == 54.0

    # Principal planes
    assert abs(data["p1"]) < 5.0
    assert abs(data["p2"]) < 5.0

    # ABCD Matrix determinant should be 1.0 (unimodular in air)
    det = data["matrix"].determinant()
    assert abs(det - 1.0) < 1e-4


def test_first_order_data_afocal_flat_plate():
    """Verify flat window produces afocal / infinite focal length without crashing."""
    s1 = Surface(name="Window_Front", radius=0.0, thickness=5.0, glass="N-BK7")
    s2 = Surface(name="Window_Rear", radius=0.0, thickness=20.0, glass="AIR")
    s_det = Surface(name="Detector", surface_type=SurfaceType.DETECTOR)

    data = calculate_first_order_data([s1, s2, s_det], wavelength_um=0.5876)
    assert data["is_afocal"]
    assert math.isinf(data["effl"])
    assert math.isinf(data["f_number"])
    assert data["na"] == 0.0


def test_format_zemax_report():
    """Verify formatted report contains standard Zemax keywords."""
    s1 = Surface(name="Front", radius=51.68, thickness=6.0, glass="N-BK7")
    s2 = Surface(name="Rear", radius=-51.68, thickness=48.0, glass="AIR")
    s_det = Surface(name="Detector", surface_type=SurfaceType.DETECTOR)

    data = calculate_first_order_data([s1, s2, s_det], wavelength_um=0.5876, epd=20.0)
    report = format_zemax_first_order_report(data, "Test System")

    assert "ZEMAX OPTICSTUDIO FIRST ORDER DATA" in report
    assert "Effective Focal Length (EFFL)" in report
    assert "Back Focal Length (BFL)" in report
    assert "Total Track Length (TOTR)" in report
    assert "Entrance Pupil Diameter (EPD)" in report
    assert "Image Space F-Number (F/#)" in report
    assert "Paraxial ABCD Ray Transfer Matrix" in report


def test_first_order_dialog_initialization():
    """Verify ZemaxFirstOrderDataDialog instantiates and populates correctly."""
    if not QT_AVAILABLE:
        pytest.skip("Qt not available in environment")

    from optics_app.gui.first_order_dialog import ZemaxFirstOrderDataDialog

    app = QtWidgets.QApplication.instance()
    if not app:
        app = QtWidgets.QApplication([])

    surfaces = [
        Surface(name="S1", radius=50.0, thickness=5.0, glass="N-BK7"),
        Surface(name="S2", radius=-50.0, thickness=45.0, glass="AIR"),
        Surface(name="Det", surface_type=SurfaceType.DETECTOR),
    ]

    dlg = ZemaxFirstOrderDataDialog(surfaces)
    assert "Zemax" in dlg.windowTitle()
    assert len(dlg.cached_report_text) > 50
    assert "EFFL" in dlg.lbl_effl.text()
    assert "Afocal" not in dlg.lbl_effl.text()


def test_main_window_live_first_order_status():
    """Main window exposes the First Order action and live EFFL status label."""
    if not QT_AVAILABLE:
        pytest.skip("Qt not available in environment")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from v2.gui.main_window import MainWindow

    win = MainWindow()
    assert hasattr(win, "on_show_first_order_data")
    assert hasattr(win, "lbl_status_first_order")
    win.on_trace_rays()
    text = win.lbl_status_first_order.text()
    assert text.startswith("EFFL:")
    assert "--" not in text
    win.close()
