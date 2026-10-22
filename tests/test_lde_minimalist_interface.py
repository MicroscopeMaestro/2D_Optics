"""
Tests for Minimalist and Functional Lens Data Editor Interface.
Verifies that redundant toolbars (such as cluster, uncluster, component view, variable, fixed,
solve, and first order buttons) are removed from the direct LDE layout, while preserving
100% of the functional capabilities via standard menus, context menus, and keyboard shortcuts.
"""

import pytest
from optics_app.gui.qt_compat import QT_AVAILABLE, QtWidgets, Qt, QtGui, QtCore
from optics_app.gui.main_window import MainWindow


if QT_AVAILABLE:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.mark.skipif(not QT_AVAILABLE, reason="Qt bindings not available")
def test_lde_dock_has_no_cluttered_button_bar():
    """Verify that the LDE dock layout contains no redundant buttons and only hosts the table view."""
    window = MainWindow()
    try:
        lde_dock = window.lde_dock
        assert lde_dock is not None

        # Confirm that QPushButton instances for cluster, uncluster, var, etc. are NOT in the LDE container
        dock_widget = lde_dock.widget()
        buttons = dock_widget.findChildren(QtWidgets.QPushButton)
        assert len(buttons) == 0, f"Expected 0 direct tool buttons in LDE, found {[b.text() for b in buttons]}"

        # Confirm that the table view is directly in the LDE container
        table_views = dock_widget.findChildren(QtWidgets.QTableView)
        assert len(table_views) >= 1
        assert window.table_view in table_views
    finally:
        window.close()


@pytest.mark.skipif(not QT_AVAILABLE, reason="Qt bindings not available")
def test_component_view_action_in_view_menu():
    """Verify that Component View toggle is cleanly placed in the View menu and functional."""
    window = MainWindow()
    try:
        assert hasattr(window, "act_comp_view")
        assert window.act_comp_view.text() == "🔲 &Component View (Grouped Elements)"
        assert window.act_comp_view.isCheckable()
        assert not window.model.component_name_mode()

        # Toggle it ON
        window.act_comp_view.setChecked(True)
        assert window.model.component_name_mode() is True

        # Toggle it OFF
        window.act_comp_view.setChecked(False)
        assert window.model.component_name_mode() is False
    finally:
        window.close()


@pytest.mark.skipif(not QT_AVAILABLE, reason="Qt bindings not available")
def test_variable_and_fixed_solve_actions():
    """Verify that setting cells to Variable and Fixed works programmatically and via selection."""
    window = MainWindow()
    try:
        table = window.table_view
        assert len(window.model._surfaces) > 2

        # Select surface 1, radius (row 1, col 2)
        idx_radius = window.model.index(1, 2)
        table.setCurrentIndex(idx_radius)

        # Mark Variable
        window._set_selected_cells_solve(is_variable=True)
        surf1 = window.model._surfaces[1]
        assert surf1.radius_variable is True

        # Mark Fixed
        window._set_selected_cells_solve(is_variable=False)
        assert surf1.radius_variable is False

        # Select surface 1, thickness (row 1, col 3)
        idx_thick = window.model.index(1, 3)
        table.setCurrentIndex(idx_thick)
        window._set_selected_cells_solve(is_variable=True)
        assert surf1.thickness_variable is True
        window._set_selected_cells_solve(is_variable=False)
        assert surf1.thickness_variable is False
    finally:
        window.close()


@pytest.mark.skipif(not QT_AVAILABLE, reason="Qt bindings not available")
def test_keyboard_shortcuts_v_and_f():
    """Verify that pressing 'V' and 'F' on the LDE table sets Variable / Fixed."""
    window = MainWindow()
    try:
        table = window.table_view
        assert len(window.model._surfaces) > 2

        # Select surface 1, radius (row 1, col 2)
        idx = window.model.index(1, 2)
        table.setCurrentIndex(idx)

        # Send Key_V event
        event_v = QtGui.QKeyEvent(QtCore.QEvent.Type.KeyPress, Qt.Key.Key_V, Qt.KeyboardModifier.NoModifier, "v")
        table.keyPressEvent(event_v)
        assert window.model._surfaces[1].radius_variable is True

        # Send Key_F event
        event_f = QtGui.QKeyEvent(QtCore.QEvent.Type.KeyPress, Qt.Key.Key_F, Qt.KeyboardModifier.NoModifier, "f")
        table.keyPressEvent(event_f)
        assert window.model._surfaces[1].radius_variable is False
    finally:
        window.close()


@pytest.mark.skipif(not QT_AVAILABLE, reason="Qt bindings not available")
def test_edit_menu_solves_and_clustering():
    """Verify that Edit menu exposes clustering, unclustering, and Zemax solves submenu."""
    window = MainWindow()
    try:
        menubar = window.menuBar()
        actions = menubar.actions()
        edit_action = next(a for a in actions if a.text() == "&Edit")
        edit_menu = edit_action.menu()
        edit_action_texts = [a.text() for a in edit_menu.actions()]

        assert any("Cluster Selected Surfaces" in t for t in edit_action_texts)
        assert any("Uncluster" in t for t in edit_action_texts)
        assert any("Zemax Solves & Variables" in t for t in edit_action_texts)

        # Also verify Tools menu has the Zemax Solve dialog
        tools_action = next(a for a in actions if a.text() == "&Tools")
        tools_menu = tools_action.menu()
        tools_action_texts = [a.text() for a in tools_menu.actions()]
        assert any("Zemax Solve & Pickup Dialog" in t for t in tools_action_texts)
    finally:
        window.close()


@pytest.mark.skipif(not QT_AVAILABLE, reason="Qt bindings not available")
def test_optical_view_rescale_and_zoom_menu_actions():
    """Verify that Optical System Rescaling actions are available in View menu."""
    window = MainWindow()
    try:
        assert hasattr(window, "act_fit_sys")
        assert hasattr(window, "act_rescale_dlg")
        assert hasattr(window, "act_reset_scale")

        menubar = window.menuBar()
        actions = menubar.actions()
        view_action = next(a for a in actions if a.text() == "&View")
        view_menu = view_action.menu()
        view_action_texts = [a.text() for a in view_menu.actions()]

        assert any("Fit View to System" in t for t in view_action_texts)
        assert any("Rescale View Limits" in t for t in view_action_texts)
        assert any("Optical System Rescaling" in t for t in view_action_texts)
    finally:
        window.close()


@pytest.mark.skipif(not QT_AVAILABLE, reason="Qt bindings not available")
def test_dual_view_minimalist_rescaling_and_responsive_resize():
    """Verify that DualViewWidget removes +/- buttons for minimalism and rescales on resize."""
    window = MainWindow()
    try:
        dv = window.dual_view
        assert hasattr(dv, "btn_fit_system")
        assert hasattr(dv, "spin_transverse_scale")

        # Verify that +/- buttons are removed from the viewer top bar to maintain minimalist UI
        assert getattr(dv, "btn_zoom_in", None) is None
        assert getattr(dv, "btn_zoom_out", None) is None

        # Confirm no QPushButton on dual_view top bar has '+' or '-' text
        top_buttons = [b.text() for b in dv.findChildren(QtWidgets.QPushButton)]
        assert not any("➕" in t or "➖" in t for t in top_buttons)

        # Test responsive resize of viewer and main window
        dv._on_widget_resized()
        resize_ev = QtGui.QResizeEvent(QtCore.QSize(1280, 800), QtCore.QSize(1024, 768))
        dv.resizeEvent(resize_ev)
        window.resizeEvent(resize_ev)

        # Test setting view limits explicitly
        dv.set_view_limits(-15.0, 120.0, -35.0, 35.0)
        assert "rz" in dv.saved_limits
        xlim, ylim = dv.saved_limits["rz"]
        assert pytest.approx(xlim[0]) == -15.0
        assert pytest.approx(xlim[1]) == 120.0
        assert pytest.approx(ylim[0]) == -35.0
        assert pytest.approx(ylim[1]) == 35.0

        # Test reset view limits
        dv.reset_view_limits()
        assert len(dv.saved_limits) == 0

        # Test transverse scale
        dv.set_transverse_scale(3.0)
        assert dv.transverse_scale == 3.0
    finally:
        window.close()


@pytest.mark.skipif(not QT_AVAILABLE, reason="Qt bindings not available")
def test_rescale_view_dialog():
    """Verify that RescaleViewDialog correctly initializes and applies limits and scale."""
    from optics_app.gui.dual_view import RescaleViewDialog
    window = MainWindow()
    try:
        dv = window.dual_view
        dlg = RescaleViewDialog(dv)
        assert dlg.spin_scale is not None
        assert dlg.spin_z_min is not None
        assert dlg.spin_z_max is not None

        # Change values and apply
        dlg.spin_scale.setValue(4.0)
        dlg.spin_z_min.setValue(-25.0)
        dlg.spin_z_max.setValue(250.0)
        dlg.spin_y_min.setValue(-50.0)
        dlg.spin_y_max.setValue(50.0)
        dlg.apply_changes()

        assert dv.transverse_scale == 4.0
        assert "rz" in dv.saved_limits
        xlim, ylim = dv.saved_limits["rz"]
        assert pytest.approx(xlim[0]) == -25.0
        assert pytest.approx(xlim[1]) == 250.0
    finally:
        window.close()


@pytest.mark.skipif(not QT_AVAILABLE, reason="Qt bindings not available")
def test_3d_visualizer_navigation_keys():
    """Verify that 3D visualizer handles 'f', 'home', '+', and '-' navigation keys."""
    window = MainWindow()
    try:
        vis = window.visualizer_3d
        assert vis._handle_navigation_key("f") is True
        assert vis._handle_navigation_key("home") is True
        assert vis._handle_navigation_key("+") is True
        assert vis._handle_navigation_key("-") is True
    finally:
        window.close()

