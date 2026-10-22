"""
Unit tests for toggling dark background design in DualViewWidget, Visualizer3DWidget, and MainWindow.
"""

import os
import pytest
import matplotlib.colors as mcolors

from optics_app.gui.qt_compat import QT_AVAILABLE, QtWidgets
from optics_app.gui.dual_view import DualViewWidget, HAS_MPL_QT
from optics_app.gui.visualizer_3d import Visualizer3DWidget
from optics_app.gui.main_window import MainWindow
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.surface import Surface, SurfaceType


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
def test_dual_view_dark_mode_toggle():
    stack = _build_test_stack()
    widget = DualViewWidget()
    widget.surface_stack = stack
    widget.rz_view_active = True
    widget.xz_view_active = True
    widget.xy_front_view_active = True
    widget.update_layout(surface_stack=stack)

    # 1. By default, dark mode is disabled (light mode)
    assert widget.dark_mode is False
    if widget.cb_dark_mode:
        assert widget.cb_dark_mode.isChecked() is False

    # 2. Enable dark mode
    signals = []
    widget.dark_mode_changed.connect(lambda v: signals.append(v))
    widget.set_dark_mode(True)

    assert widget.dark_mode is True
    if widget.cb_dark_mode:
        assert widget.cb_dark_mode.isChecked() is True
    assert signals == [True]

    # Check figure and axis facecolors
    fig_color = mcolors.to_hex(widget.fig.patch.get_facecolor())
    assert fig_color.lower() == '#0f172a'
    ax_color = mcolors.to_hex(widget.ax_rz.get_facecolor())
    assert ax_color.lower() == '#0b0f19'

    # 3. Disable dark mode
    widget.set_dark_mode(False)
    assert widget.dark_mode is False
    if widget.cb_dark_mode:
        assert widget.cb_dark_mode.isChecked() is False
    assert signals == [True, False]

    fig_color_light = mcolors.to_hex(widget.fig.patch.get_facecolor())
    assert fig_color_light.lower() == '#ffffff'


@pytest.mark.skipif(not (QT_AVAILABLE and HAS_MPL_QT), reason="Qt Matplotlib unavailable")
def test_visualizer_3d_dark_mode_toggle():
    stack = _build_test_stack()
    widget = Visualizer3DWidget()
    widget.surface_stack = stack
    widget.update_layout(surface_stack=stack)

    # Default is False
    assert widget.dark_mode is False
    if widget.cb_dark_mode:
        assert widget.cb_dark_mode.isChecked() is False

    signals = []
    widget.dark_mode_changed.connect(lambda v: signals.append(v))
    widget.set_dark_mode(True)

    assert widget.dark_mode is True
    if widget.cb_dark_mode:
        assert widget.cb_dark_mode.isChecked() is True
    assert signals == [True]

    fig_color = mcolors.to_hex(widget.fig.patch.get_facecolor())
    assert fig_color.lower() == '#0f172a'

    # Check serialization in view settings
    settings = widget.get_view_settings()
    assert settings.get("dark_mode") is True

    widget.set_dark_mode(False)
    assert widget.dark_mode is False
    assert signals == [True, False]


@pytest.mark.skipif(not (QT_AVAILABLE and HAS_MPL_QT), reason="Qt Matplotlib unavailable")
def test_main_window_dark_mode_sync():
    win = MainWindow()
    assert hasattr(win, "act_dark_mode")
    assert win.act_dark_mode.isChecked() is False
    assert win.dual_view.dark_mode is False
    assert win.visualizer_3d.dark_mode is False

    # Toggle from MainWindow action
    win.act_dark_mode.setChecked(True)
    assert win.dark_mode is True
    assert win.dual_view.dark_mode is True
    assert win.visualizer_3d.dark_mode is True
    assert win.analysis_tabs.dark_mode is True
    assert win.table_view.dark_mode is True
    assert win.model.dark_mode is True
    assert "#0f172a" in win.styleSheet()

    # Toggle from DualViewWidget
    win.dual_view.set_dark_mode(False)
    assert win.dark_mode is False
    assert win.act_dark_mode.isChecked() is False
    assert win.visualizer_3d.dark_mode is False
    assert win.analysis_tabs.dark_mode is False
    assert win.table_view.dark_mode is False
    assert win.model.dark_mode is False
    assert win.styleSheet() == ""

    # Toggle via set_dark_mode method
    win.set_dark_mode(True)
    assert win.dark_mode is True
    assert win.act_dark_mode.isChecked() is True
    assert win.dual_view.dark_mode is True
    assert win.visualizer_3d.dark_mode is True
    assert win.analysis_tabs.dark_mode is True
    assert win.table_view.dark_mode is True
    assert win.model.dark_mode is True
    assert "#0f172a" in win.styleSheet()

    # Toggle off via set_dark_mode method
    win.set_dark_mode(False)
    assert win.dark_mode is False
    assert win.act_dark_mode.isChecked() is False
    assert win.dual_view.dark_mode is False
    assert win.visualizer_3d.dark_mode is False
    assert win.analysis_tabs.dark_mode is False
    assert win.table_view.dark_mode is False
    assert win.model.dark_mode is False
    assert win.styleSheet() == ""
