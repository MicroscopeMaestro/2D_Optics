"""
Unit tests for toggling 2D projection labels in DualViewWidget and MainWindow.
"""

import os
import pytest

from optics_app.gui.qt_compat import QT_AVAILABLE, QtWidgets
from optics_app.gui.dual_view import DualViewWidget, HAS_MPL_QT
from optics_app.gui.main_window import MainWindow
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.ray import Ray


if QT_AVAILABLE:
    app = QtWidgets.QApplication.instance()
    if app is None:
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        app = QtWidgets.QApplication([])


def _build_test_stack() -> SurfaceStack:
    stack = SurfaceStack()
    stack.surfaces = [
        Surface(name="Source", surface_type=SurfaceType.OBJECT, thickness=25.0, glass="AIR", semi_diameter=10.0),
        Surface(name="Lens1_Front", radius=45.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7),
        Surface(name="Lens1_Rear", radius=-45.0, thickness=20.0, glass="AIR", semi_diameter=12.7),
        Surface(name="ApertureStop", surface_type=SurfaceType.STOP, thickness=15.0, glass="AIR", semi_diameter=8.0),
        Surface(name="Sensor", surface_type=SurfaceType.DETECTOR, thickness=0.0, glass="AIR", semi_diameter=15.0),
    ]
    stack._update_accumulated_positions()
    return stack


@pytest.mark.skipif(not (QT_AVAILABLE and HAS_MPL_QT), reason="Qt Matplotlib unavailable")
def test_dual_view_labels_toggle():
    stack = _build_test_stack()
    widget = DualViewWidget()
    widget.surface_stack = stack
    widget.selected_surface_index = 0
    widget.rz_view_active = True
    widget.xz_view_active = True
    widget.xy_front_view_active = True
    widget.update_layout(surface_stack=stack)

    # By default, labels are enabled and checkbox is checked
    assert widget.show_labels is True
    if widget.cb_show_labels:
        assert widget.cb_show_labels.isChecked() is True

    # Check that labels/callouts are rendered
    assert len(widget.ax_rz.texts) > 0
    assert len(widget.ax_xz.texts) > 0
    assert len(widget.ax_xy_front.texts) > 0

    # Turn off labels via set_show_labels(False)
    signals = []
    widget.labels_visibility_changed.connect(lambda v: signals.append(v))
    widget.set_show_labels(False)

    assert widget.show_labels is False
    if widget.cb_show_labels:
        assert widget.cb_show_labels.isChecked() is False
    assert signals == [False]

    # Verify no labels are rendered in any 2D projection
    assert len(widget.ax_rz.texts) == 0
    assert len(widget.ax_xz.texts) == 0
    assert len(widget.ax_xy_front.texts) == 0

    # Turn labels back on via checkbox UI
    if widget.cb_show_labels:
        widget.cb_show_labels.setChecked(True)
        assert widget.show_labels is True
        assert len(widget.ax_rz.texts) > 0
        assert len(widget.ax_xz.texts) > 0
        assert len(widget.ax_xy_front.texts) > 0
        assert signals == [False, True]


@pytest.mark.skipif(not (QT_AVAILABLE and HAS_MPL_QT), reason="Qt Matplotlib unavailable")
def test_main_window_2d_labels_action_sync():
    win = MainWindow()
    assert hasattr(win, "act_show_2d_labels")
    assert win.act_show_2d_labels.isChecked() is True
    assert win.dual_view.show_labels is True

    # Toggle action off
    win.act_show_2d_labels.setChecked(False)
    assert win.dual_view.show_labels is False
    if win.dual_view.cb_show_labels:
        assert win.dual_view.cb_show_labels.isChecked() is False

    # Toggle checkbox on
    if win.dual_view.cb_show_labels:
        win.dual_view.cb_show_labels.setChecked(True)
        assert win.dual_view.show_labels is True
        assert win.act_show_2d_labels.isChecked() is True

    # Method set_show_2d_labels
    win.set_show_2d_labels(False)
    assert win.dual_view.show_labels is False
    assert win.act_show_2d_labels.isChecked() is False
