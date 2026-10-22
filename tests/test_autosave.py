"""
Unit tests for 5-minute Auto-Save, background recovery snapshots, and dirty tracking.
"""
import os
import shutil
import tempfile
import pytest
import json

from optics_app.gui.main_window import MainWindow
from optics_app.engine.surface import Surface
from optics_app.engine.lde import SurfaceStack


@pytest.fixture
def temp_workspace():
    """Create a temporary directory workspace for testing autosave files."""
    old_cwd = os.getcwd()
    tmp_dir = tempfile.mkdtemp(prefix="optics_autosave_test_")
    os.chdir(tmp_dir)
    yield tmp_dir
    os.chdir(old_cwd)
    try:
        shutil.rmtree(tmp_dir)
    except Exception:
        pass


def test_autosave_timer_initialization(qapp, temp_workspace):
    """Verify that MainWindow initializes with 5-minute autosave enabled."""
    win = MainWindow()
    try:
        assert win.autosave_enabled is True
        assert win.autosave_interval_ms == 5 * 60 * 1000  # 300,000 ms (5 minutes)
        assert win._autosave_timer is not None
        assert win._autosave_timer.interval() == 300000
        assert win._autosave_timer.isActive() is True
    finally:
        win.close()


def test_autosave_untitled_project(qapp, temp_workspace):
    """Verify that auto-saving an untitled design silently creates a recovery snapshot without modal popups."""
    win = MainWindow()
    try:
        # Configure surfaces
        s1 = Surface(name="S1", radius=50.0, thickness=5.0, glass="N-BK7")
        s2 = Surface(name="S2", radius=-50.0, thickness=45.0, glass="AIR")
        stack = SurfaceStack()
        stack.surfaces = [s1, s2]
        win.model.set_surface_stack(stack)
        win.set_dirty(True)
        assert win.is_dirty is True
        assert win.current_file_path is None

        # Trigger autosave
        saved_path = win.on_autosave()
        assert saved_path is not None
        assert os.path.exists(saved_path)
        assert "autosave_untitled.json" in saved_path

        # Verify JSON content is valid
        with open(saved_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "surfaces" in data
        assert len(data["surfaces"]) == 2

        # is_dirty is cleared after autosave
        assert win.is_dirty is False
        # current_file_path remains None so user can still do Save As with chosen name
        assert win.current_file_path is None
    finally:
        win.close()


def test_autosave_named_project(qapp, temp_workspace):
    """Verify that auto-saving an active project updates the file, backups to .autosave, and archives to .history."""
    win = MainWindow()
    try:
        proj_path = os.path.join(temp_workspace, "my_achromat.json")

        s1 = Surface(name="S1", radius=13.212, thickness=4.0, glass="N-BAF10")
        s2 = Surface(name="S2", radius=0.0, thickness=2.0, glass="SF10")
        stack = SurfaceStack()
        stack.surfaces = [s1, s2]
        win.model.set_surface_stack(stack)
        
        # Explicitly save once
        win.on_save_project(proj_path)
        assert win.current_file_path == proj_path
        assert win.is_dirty is False

        # Simulate modification
        win.model.setData(win.model.index(0, 2), 15.0)
        win.set_dirty(True)
        assert win.is_dirty is True

        # Trigger autosave
        res = win.on_autosave()
        assert res == proj_path
        assert win.is_dirty is False

        # Check that .autosave copy was created
        backup_file = os.path.join(temp_workspace, ".autosave", "my_achromat_autosave.json")
        assert os.path.exists(backup_file)

        # Check that .history index recorded the snapshot
        history_index = os.path.join(temp_workspace, ".history", "my_achromat_history_index.json")
        assert os.path.exists(history_index)
        with open(history_index, "r", encoding="utf-8") as f:
            records = json.load(f)
        assert len(records) >= 1
        assert any("Auto-saved" in r["description"] for r in records)
    finally:
        win.close()


def test_autosave_interval_presets_and_toggle(qapp, temp_workspace):
    """Verify changing autosave interval and enabling/disabling."""
    win = MainWindow()
    try:
        # Change to 2 minutes
        win.set_autosave_interval(2.0)
        assert win.autosave_interval_ms == 2 * 60 * 1000
        assert win._autosave_timer.interval() == 120000
        assert win.autosave_enabled is True

        # Disable autosave
        win._on_toggle_autosave(False)
        assert win.autosave_enabled is False
        assert win._autosave_timer.isActive() is False

        # Re-enable
        win._on_toggle_autosave(True)
        assert win.autosave_enabled is True
        assert win._autosave_timer.isActive() is True
    finally:
        win.close()


def test_dirty_state_tracking(qapp, temp_workspace):
    """Verify is_dirty flag updates window title and dirty status."""
    win = MainWindow()
    try:
        assert win.is_dirty is False

        # Mark dirty
        win.set_dirty(True)
        assert win.is_dirty is True

        # Save clears dirty
        test_path = os.path.join(temp_workspace, "dirty_test.json")
        win.on_save_project(test_path)
        assert win.is_dirty is False
    finally:
        win.close()
