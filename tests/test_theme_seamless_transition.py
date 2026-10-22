"""
End-to-end verification of seamless transition between white and dark background
across all edition tools, docks, and diagnostic analyses.
"""

import pytest
from optics_app.gui.qt_compat import QT_AVAILABLE, QApplication
from optics_app.gui.main_window import MainWindow


@pytest.fixture(scope="module")
def qapp():
    if not QT_AVAILABLE:
        pytest.skip("PyQt is not available")
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_theme_seamless_transition_all_tools_and_analyses(qapp):
    win = MainWindow()
    try:
        # 1. Verify Initial Light Theme
        assert win.dark_mode is False
        assert win.styleSheet() == ""
        assert win.dual_view.dark_mode is False
        assert win.visualizer_3d.dark_mode is False
        assert win.table_view.dark_mode is False
        assert win.model.dark_mode is False
        assert win.analysis_tabs.dark_mode is False

        # Verify initial state of all 9 dock edition tools
        assert getattr(win.component_editor_dock, "dark_mode", None) is False
        assert getattr(win.optimizer_dock, "dark_mode", None) is False
        assert getattr(win.tolerancing_dock, "dark_mode", None) is False
        assert getattr(win.ultrafast_dock, "dark_mode", None) is False
        assert getattr(win.plm_dock.studio, "dark_mode", None) is False
        assert getattr(win.measurement_dock, "dark_mode", None) is False
        assert getattr(win.system_config_dock, "dark_mode", None) is False
        assert getattr(win.history_dock, "dark_mode", None) is False
        assert getattr(win.ai_dock, "dark_mode", None) is False

        # Verify initial state of all 14 analysis tabs
        num_tabs = win.analysis_tabs.count()
        assert num_tabs == 14, f"Expected 14 analysis tabs, got {num_tabs}"
        tabs = [win.analysis_tabs.widget(i) for i in range(num_tabs)]
        for i, t in enumerate(tabs):
            assert getattr(t, "dark_mode", None) is False, f"Tab {i} ({t}) should have dark_mode False"

        # ----------------------------------------------------
        # 2. Transition to Dark Background Theme
        # ----------------------------------------------------
        win.set_dark_mode(True)

        assert win.dark_mode is True
        assert win.styleSheet() != ""
        assert win.dual_view.dark_mode is True
        assert win.visualizer_3d.dark_mode is True
        assert win.table_view.dark_mode is True
        assert win.model.dark_mode is True
        assert win.analysis_tabs.dark_mode is True

        # All 9 Docks transitioned to Dark Mode
        assert win.component_editor_dock.dark_mode is True
        assert win.optimizer_dock.dark_mode is True
        assert win.tolerancing_dock.dark_mode is True
        assert win.ultrafast_dock.dark_mode is True
        assert win.plm_dock.studio.dark_mode is True
        assert win.measurement_dock.dark_mode is True
        assert win.system_config_dock.dark_mode is True
        assert win.history_dock.dark_mode is True
        assert win.ai_dock.dark_mode is True

        # All 14 Analysis Tabs transitioned to Dark Mode
        for i, t in enumerate(tabs):
            assert getattr(t, "dark_mode", None) is True, f"Tab {i} ({t}) should have dark_mode True"

        # Check that axes replotting works cleanly in dark mode
        for t in tabs:
            if hasattr(t, "_replot"):
                t._replot()
        win.plm_dock.studio.update_live_plots()

        # ----------------------------------------------------
        # 3. Transition back to White / Light Background Theme
        # ----------------------------------------------------
        win.set_dark_mode(False)

        assert win.dark_mode is False
        assert win.styleSheet() == ""
        assert win.dual_view.dark_mode is False
        assert win.visualizer_3d.dark_mode is False
        assert win.table_view.dark_mode is False
        assert win.model.dark_mode is False
        assert win.analysis_tabs.dark_mode is False

        # All 9 Docks transitioned back to Light Mode
        assert win.component_editor_dock.dark_mode is False
        assert win.optimizer_dock.dark_mode is False
        assert win.tolerancing_dock.dark_mode is False
        assert win.ultrafast_dock.dark_mode is False
        assert win.plm_dock.studio.dark_mode is False
        assert win.measurement_dock.dark_mode is False
        assert win.system_config_dock.dark_mode is False
        assert win.history_dock.dark_mode is False
        assert win.ai_dock.dark_mode is False

        # All 14 Analysis Tabs transitioned back to Light Mode
        for i, t in enumerate(tabs):
            assert getattr(t, "dark_mode", None) is False, f"Tab {i} ({t}) should have dark_mode False"

        # Check that axes replotting works cleanly in light mode
        for t in tabs:
            if hasattr(t, "_replot"):
                t._replot()
        win.plm_dock.studio.update_live_plots()

    finally:
        win.close()


def test_aperture_dialog_theme_transition(qapp):
    win = MainWindow()
    try:
        from optics_app.engine.surface import Surface, SurfaceType
        from optics_app.gui.aperture_dialog import ApertureStopDialog

        surf = Surface(name="Lens 1", surface_type=SurfaceType.STANDARD, radius=50.0, thickness=5.0)

        # In light mode
        dlg_light = ApertureStopDialog(surface=surf, parent=win)
        assert dlg_light.dark_mode is False

        # Switch to dark mode
        dlg_light.set_dark_mode(True)
        assert dlg_light.dark_mode is True

        # Switch back to light mode
        dlg_light.set_dark_mode(False)
        assert dlg_light.dark_mode is False

        # If created when win is dark
        win.set_dark_mode(True)
        dlg_dark = ApertureStopDialog(surface=surf, parent=win)
        assert dlg_dark.dark_mode is True
    finally:
        win.close()


def test_achromat_dialog_theme_transition(qapp):
    win = MainWindow()
    try:
        from optics_app.gui.achromat_dialog import AchromatDesignerDialog

        # In light mode
        dlg = AchromatDesignerDialog(model=win.model, parent=win)
        assert dlg.dark_mode is False

        # Transition to dark
        dlg.set_dark_mode(True)
        assert dlg.dark_mode is True
        assert "#1e293b" in dlg.lbl_prescription.styleSheet()
        assert "#064e3b" in dlg.lbl_aberrations.styleSheet()

        # Transition back to light
        dlg.set_dark_mode(False)
        assert dlg.dark_mode is False
        assert "#f8fafc" in dlg.lbl_prescription.styleSheet()
        assert "#f0fdf4" in dlg.lbl_aberrations.styleSheet()
    finally:
        win.close()

