from optics_app.gui.main_window import MainWindow
from optics_app.gui.qt_compat import QT_AVAILABLE, QtWidgets


if QT_AVAILABLE:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _action_texts(toolbar):
    return [action.text() for action in toolbar.actions() if not action.isSeparator()]


def test_main_toolbar_contains_only_primary_workflow_actions():
    window = MainWindow()
    try:
        assert _action_texts(window.main_toolbar) == [
            "&New Project",
            "&Load Project (.json)...",
            "💾 &Save Project",
            "&Undo",
            "&Redo",
            "&Trace Rays",
            "🤖 AI Agent",
        ]
    finally:
        window.close()


def test_default_workspace_keeps_ai_agent_visible_with_core_tools():
    window = MainWindow()
    try:
        assert not window.lde_dock.isHidden()
        assert not window.dual_view_dock.isHidden()
        assert not window.component_editor_dock.isHidden()
        assert not window.ai_dock.isHidden()
        for dock in (
            window.system_config_dock,
            window.optimizer_dock,
            window.tolerancing_dock,
            window.history_dock,
            window.analysis_dock,
            window.measurement_dock,
        ):
            assert dock.isHidden()
            assert dock.toggleViewAction() is not None

        window.analysis_dock.show()
        window.ai_dock.show()
        assert not window.analysis_dock.isHidden()
        window._apply_minimal_workspace()
        assert window.analysis_dock.isHidden()
        assert not window.ai_dock.isHidden()
        assert not window.dual_view_dock.isHidden()
    finally:
        window.close()


def test_duplicate_always_visible_controls_are_removed():
    window = MainWindow()
    try:
        assert not hasattr(window.component_editor_dock, "btn_catalog_lens")
        assert not hasattr(window.system_config_dock, "btn_save_cfg")
        assert not hasattr(window.system_config_dock, "btn_load_cfg")
        assert not hasattr(window.system_config_dock, "btn_apply")

        quick_action_groups = [
            child for child in window.ai_dock.findChildren(QtWidgets.QGroupBox)
            if child.title() == "Quick Actions"
        ]
        assert quick_action_groups == []
        assert window.ai_dock.mode_combo.currentData() == "agentic"
        assert window.ai_dock.input_box.isEnabled()
    finally:
        window.close()


def test_ai_agent_can_always_be_reopened_from_toolbar_and_window_menu():
    window = MainWindow()
    try:
        window.ai_dock.hide()
        assert window.ai_dock.isHidden()
        window.act_open_ai_agent.trigger()
        assert not window.ai_dock.isHidden()
        assert window.ai_dock.tabs.currentIndex() == 0

        assert window.ai_dock.toggleViewAction().text() == "🤖 AI Assistant & Agentic Mode"
    finally:
        window.close()
