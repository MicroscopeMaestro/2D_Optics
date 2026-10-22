"""
Empirical Stress Test Suite for Milestone 4 GUI (Iteration 2)
Verifies MainWindow getter methods, surface slot aliases, and LensDataTableModel numeric validation.
"""

import math
import numpy as np
import pytest
from optics_app.gui.main_window import MainWindow
from optics_app.gui.lde_table import LensDataTableModel
from optics_app.gui.dual_view import DualViewWidget
from optics_app.gui.analysis_tabs import AnalysisTabsWidget
from optics_app.gui.qt_compat import QT_AVAILABLE, QtWidgets

if QT_AVAILABLE and hasattr(QtWidgets, "QApplication"):
    try:
        _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    except Exception:
        pass




def test_main_window_getters():
    """Verify MainWindow getter methods return expected components."""
    win = MainWindow()
    
    # 1. get_model()
    model = win.get_model()
    assert model is not None, "get_model() returned None"
    assert isinstance(model, LensDataTableModel), f"get_model() returned {type(model)}"
    assert model is win.model, "get_model() does not reference win.model"

    # 2. get_dual_view()
    dual_view = win.get_dual_view()
    assert dual_view is not None, "get_dual_view() returned None"
    assert isinstance(dual_view, DualViewWidget), f"get_dual_view() returned {type(dual_view)}"
    assert dual_view is win.dual_view, "get_dual_view() does not reference win.dual_view"

    # 3. get_analysis_tabs()
    analysis_tabs = win.get_analysis_tabs()
    assert analysis_tabs is not None, "get_analysis_tabs() returned None"
    assert isinstance(analysis_tabs, AnalysisTabsWidget), f"get_analysis_tabs() returned {type(analysis_tabs)}"
    assert analysis_tabs is win.analysis_tabs, "get_analysis_tabs() does not reference win.analysis_tabs"


def test_surface_slot_aliases():
    """Verify MainWindow slot aliases on_add_surface and on_remove_surface."""
    win = MainWindow()
    model = win.get_model()
    initial_rows = model.rowCount()
    
    # 1. Test on_add_surface without position
    res_add1 = win.on_add_surface()
    assert res_add1 is True, "on_add_surface() failed"
    assert model.rowCount() == initial_rows + 1, f"Expected {initial_rows + 1} rows, got {model.rowCount()}"

    # 2. Test on_add_surface with explicit position
    res_add2 = win.on_add_surface(position=1)
    assert res_add2 is True, "on_add_surface(position=1) failed"
    assert model.rowCount() == initial_rows + 2, f"Expected {initial_rows + 2} rows, got {model.rowCount()}"

    # 3. Test on_remove_surface with explicit position
    res_rem1 = win.on_remove_surface(position=1)
    assert res_rem1 is True, "on_remove_surface(position=1) failed"
    assert model.rowCount() == initial_rows + 1, f"Expected {initial_rows + 1} rows, got {model.rowCount()}"

    # 4. Test on_remove_surface without position
    res_rem2 = win.on_remove_surface()
    assert res_rem2 is True, "on_remove_surface() failed"
    assert model.rowCount() == initial_rows, f"Expected {initial_rows} rows, got {model.rowCount()}"


def test_set_data_numeric_rejection():
    """Verify LensDataTableModel.setData numeric rejection for NaN, Inf, and negative semi-diameter."""
    model = LensDataTableModel()
    
    # Target row 1 (Stop surface)
    r = 1
    original_radius = model.data(r, 2)
    original_sd = model.data(r, 5)

    # 1. NaN Rejection (float nan, np nan, string "nan")
    for nan_val in [float('nan'), np.nan, "nan", "NaN", "NAN"]:
        ok = model.setData(r, 2, nan_val)
        assert ok is False, f"setData failed to reject NaN value: {nan_val}"
        assert model.data(r, 2) == original_radius, f"Model data corrupted by NaN: {nan_val}"

    # 2. Inf Rejection (float inf, -inf, np inf, np -inf)
    for inf_val in [float('inf'), float('-inf'), np.inf, -np.inf]:
        ok = model.setData(r, 2, inf_val)
        assert ok is False, f"setData failed to reject Inf value: {inf_val}"
        assert model.data(r, 2) == original_radius, f"Model data corrupted by Inf: {inf_val}"

    # Note: string "inf" and "infinity" represent Zemax planar surfaces and convert to 0.0
    ok_zemax_inf = model.setData(r, 2, "inf")
    assert ok_zemax_inf is True, "setData rejected Zemax 'inf' string"
    assert model.data(r, 2) == 0.0, f"Zemax 'inf' should convert to 0.0, got {model.data(r, 2)}"
    
    # Reset radius back
    model.setData(r, 2, original_radius)

    # 3. Semi-Diameter (column 5) Negative Rejection
    invalid_sds = [-5.0, -0.001, -100.0, "-5.0", "-0.1"]
    for bad_sd in invalid_sds:
        ok = model.setData(r, 5, bad_sd)
        assert ok is False, f"setData failed to reject negative semi-diameter: {bad_sd}"
        assert model.data(r, 5) == original_sd, f"Model data corrupted by bad semi-diameter: {bad_sd}"

    # 4. Semi-Diameter Valid Positive Input
    valid_sds = [5.0, 0.0, 15.5, "12.0"]
    for val_sd in valid_sds:
        ok = model.setData(r, 5, val_sd)
        assert ok is True, f"setData rejected valid semi-diameter: {val_sd}"
        expected = float(val_sd)
        assert model.data(r, 5) == expected, f"Expected semi-diameter {expected}, got {model.data(r, 5)}"

    # 5. Radius (column 2) Negative Concave Radius is Allowed
    ok_neg_rad = model.setData(r, 2, -50.0)
    assert ok_neg_rad is True, "setData rejected negative radius (valid concave radius)"
    assert model.data(r, 2) == -50.0, f"Expected radius -50.0, got {model.data(r, 2)}"

    # 6. Non-numeric string rejection on numeric columns
    for bad_str in ["abc", "invalid", "", "   "]:
        ok = model.setData(r, 2, bad_str)
        assert ok is False, f"setData failed to reject non-numeric string: {bad_str}"
