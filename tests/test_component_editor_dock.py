"""
Unit & Integration Tests for ComponentEditorDockWidget, Real-Time Sync, and Mirror Presets.
Covers Milestone M3 requirements and headless CI verification.
"""

import unittest
import numpy as np

from optics_app.gui.qt_compat import (
    QT_AVAILABLE, QtWidgets, DummyQtWidgetsModule, DummyQtCoreModule,
    QSlider, QDial, QTimer, QDialog, QGridLayout, QFrame, QScrollArea, QComboBox, QCheckBox
)
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.gui.lde_table import LensDataTableModel, LensDataEditorWidget
from optics_app.gui.component_editor_dock import ComponentEditorDockWidget
from optics_app.gui.main_window import MainWindow

if QT_AVAILABLE and hasattr(QtWidgets, "QApplication"):
    try:
        _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    except Exception:
        pass


class TestComponentEditorDockUnit(unittest.TestCase):
    """Unit tests for ComponentEditorDockWidget sliders, spinboxes, and presets."""

    def setUp(self):
        self.stack = SurfaceStack()
        self.stack.add_surface(Surface(name="OBJ", surface_type=SurfaceType.OBJECT, thickness=50.0))
        self.stack.add_surface(Surface(name="FoldMirror", glass="MIRROR", radius=0.0, thickness=40.0))
        self.stack.add_surface(Surface(name="LensFront", surface_type=SurfaceType.STANDARD, radius=35.0, thickness=6.0, glass="N-BK7"))
        self.stack.add_surface(Surface(name="LensBack", surface_type=SurfaceType.STANDARD, radius=-35.0, thickness=30.0, glass="AIR"))
        self.stack.add_surface(Surface(name="IMA", surface_type=SurfaceType.DETECTOR, thickness=0.0))

        self.model = LensDataTableModel(stack=self.stack)
        self.dock = ComponentEditorDockWidget(model=self.model)

    def test_dock_initialization(self):
        """Test dock widget creation and surface combo population."""
        self.assertIsNotNone(self.dock)
        self.assertEqual(self.dock.surface_combo.count(), 5)
        self.assertEqual(self.dock.get_selected_surface_index(), 0)

    def test_surface_selection_updates_controls(self):
        """Selecting a different surface updates the slider and spinbox values."""
        self.stack.surfaces[1].tilt_x = 45.0
        self.stack.surfaces[1].decenter_x = 5.0
        self.model.set_surface_stack(self.stack)
        self.dock.set_selected_surface(1)

        self.assertEqual(self.dock.get_selected_surface_index(), 1)
        self.assertAlmostEqual(self.dock.tilt_x_spin.value(), 45.0, places=2)
        self.assertAlmostEqual(self.dock.dec_x_spin.value(), 5.0, places=2)

    def test_tilt_slider_and_spinbox_sync(self):
        """Slider movements update spinboxes and mutate surface properties."""
        self.dock.set_selected_surface(1)
        
        # Test Tilt X
        self.dock.tilt_x_spin.setValue(30.0)
        self.assertAlmostEqual(self.stack.surfaces[1].tilt_x, 30.0, places=2)
        self.assertEqual(self.dock.tilt_x_slider.value(), 300) # 30 * 10 scale

        # Test Tilt Y
        self.dock.tilt_y_slider.setValue(150) # 15 deg
        self.assertAlmostEqual(self.dock.tilt_y_spin.value(), 15.0, places=2)
        self.assertAlmostEqual(self.stack.surfaces[1].tilt_y, 15.0, places=2)

        # Test Tilt Z
        self.dock.tilt_z_spin.setValue(-45.0)
        self.assertAlmostEqual(self.stack.surfaces[1].tilt_z, -45.0, places=2)
        self.assertEqual(self.dock.tilt_z_slider.value(), -450)

    def test_decenter_slider_and_spinbox_sync(self):
        """Decenter sliders and spinboxes synchronize and update surface translation."""
        self.dock.set_selected_surface(2)

        self.dock.dec_x_spin.setValue(3.5)
        self.assertAlmostEqual(self.stack.surfaces[2].decenter_x, 3.5, places=2)
        self.assertEqual(self.dock.dec_x_slider.value(), 35)

        self.dock.dec_y_slider.setValue(-20) # -2.0 mm
        self.assertAlmostEqual(self.dock.dec_y_spin.value(), -2.0, places=2)
        self.assertAlmostEqual(self.stack.surfaces[2].decenter_y, -2.0, places=2)

    def test_thickness_slider_and_spinbox_sync(self):
        """Thickness controls update surface axial spacing."""
        self.dock.set_selected_surface(2)
        self.dock.thick_spin.setValue(8.5)
        self.assertAlmostEqual(self.stack.surfaces[2].thickness, 8.5, places=2)
        self.assertEqual(self.dock.thick_slider.value(), 85)

    def test_reset_buttons(self):
        """Reset buttons restore tilts and decenters to zero."""
        self.dock.set_selected_surface(1)
        self.dock.tilt_x_spin.setValue(25.0)
        self.dock.dec_x_spin.setValue(4.0)

        self.dock.reset_tilts()
        self.assertAlmostEqual(self.stack.surfaces[1].tilt_x, 0.0, places=2)
        self.assertAlmostEqual(self.dock.tilt_x_spin.value(), 0.0, places=2)

        self.dock.reset_decenters()
        self.assertAlmostEqual(self.stack.surfaces[1].decenter_x, 0.0, places=2)
        self.assertAlmostEqual(self.dock.dec_x_spin.value(), 0.0, places=2)

    def test_quick_fold_presets(self):
        """Quick fold presets calculate and apply exact mirror tilts."""
        self.dock.set_selected_surface(1) # Mirror

        # 90 deg YZ fold (tilt_x = 45 deg)
        self.dock.apply_quick_fold(90.0, "YZ")
        self.assertAlmostEqual(self.stack.surfaces[1].tilt_x, 45.0, places=2)
        self.assertAlmostEqual(self.stack.surfaces[1].tilt_y, 0.0, places=2)

        # 90 deg XZ fold (tilt_y = 45 deg)
        self.dock.apply_quick_fold(90.0, "XZ")
        self.assertAlmostEqual(self.stack.surfaces[1].tilt_x, 0.0, places=2)
        self.assertAlmostEqual(self.stack.surfaces[1].tilt_y, 45.0, places=2)

        # 45 deg YZ fold (tilt_x = 22.5 deg)
        self.dock.apply_quick_fold(45.0, "YZ")
        self.assertAlmostEqual(self.stack.surfaces[1].tilt_x, 22.5, places=2)

        # Retro-reflector 180 deg (tilt_x = 90 deg)
        self.dock.apply_quick_fold(180.0, "YZ")
        self.assertAlmostEqual(self.stack.surfaces[1].tilt_x, 90.0, places=2)

    def test_aim_mirror_at_target(self):
        """Analytical mirror aiming calculates tilts to center chief ray on target detector."""
        self.dock.set_selected_surface(1)
        res = self.dock.aim_selected_mirror_to_target(target_idx=4)
        self.assertTrue(res.success or res.target_error_mm < 0.1)
        self.assertAlmostEqual(self.stack.surfaces[1].tilt_x, res.solved_tilt_x, places=2)

    def test_coupled_lens_element_group_rotation(self):
        """Rotating a coupled multi-surface lens element tilts front and back surfaces together."""
        self.dock.set_selected_surface(2) # LensFront
        self.dock.group_rotation_check.setChecked(True)

        self.dock.rotate_selected_element_group(tilt_x=12.0, tilt_y=0.0, pivot="front_vertex")
        self.assertAlmostEqual(self.stack.surfaces[2].tilt_x, 12.0, places=2)
        self.assertAlmostEqual(self.stack.surfaces[3].tilt_x, 12.0, places=2)

    def test_flip_whole_selected_lens_button(self):
        """The visible flip control reverses the whole lens, never one surface."""
        self.stack.surfaces[2].radius = 35.0
        self.stack.surfaces[3].radius = -70.0
        mirror_before = (
            self.stack.surfaces[1].radius,
            self.stack.surfaces[1].thickness,
            self.stack.surfaces[1].glass,
        )
        self.model.set_surface_stack(self.stack)
        self.dock.update_ui()
        self.dock.set_selected_surface(3)  # Selecting the rear still targets the whole lens.

        self.dock.btn_flip_lens.click()

        flipped = self.model._surfaces
        self.assertEqual(
            [(s.radius, s.thickness, s.glass) for s in flipped[2:4]],
            [(70.0, 6.0, "N-BK7"), (-35.0, 30.0, "AIR")],
        )
        self.assertEqual(
            (flipped[1].radius, flipped[1].thickness, flipped[1].glass),
            mirror_before,
        )

    def test_single_lens_replacement_control_is_accessible(self):
        """Catalog and file replacement share one uncluttered entry point."""
        self.assertFalse(hasattr(self.dock, "btn_catalog_lens"))
        self.assertTrue(hasattr(self.dock, "btn_replace_lens"))
        self.assertIn("Replace Selected Lens", self.dock.btn_replace_lens.text())

    def test_clicked_surface_updates_replace_target_with_two_identical_lenses(self):
        """The Replace button follows the clicked surface, not stale combo text."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", surface_type=SurfaceType.OBJECT, thickness=20.0))
        stack.insert_catalog_lens(0, "AC254-050-C", air_gap_after=25.0)
        stack.insert_catalog_lens(3, "AC254-050-C", air_gap_after=45.0)
        stack.add_surface(Surface(name="Detector", surface_type=SurfaceType.DETECTOR))
        model = LensDataTableModel(stack=stack)
        dock = ComponentEditorDockWidget(model=model)

        dock.set_selected_surface(5)  # Cemented surface of the second copy.

        combo_data = dock.lens_element_combo.currentData()
        self.assertEqual(combo_data["surface_indices"], [4, 5, 6])
        target_index, _ = dock._current_lens_target()
        self.assertEqual(target_index, 5)

        from optics_app.gui.catalog_lens_dialog import CatalogLensDialog
        dialog = CatalogLensDialog(model=model, target=5, mode="replace")
        self.assertEqual(dialog.target_combo.count(), 2)
        self.assertEqual(dialog.target_combo.currentData()["surface_indices"], [4, 5, 6])

        dock._replace_selected_lens_catalog(target_index, "AC254-030-C")
        updated = model._surfaces
        self.assertEqual([s.group_name for s in updated[1:4]], ["AC254-050-C"] * 3)
        self.assertEqual([s.group_name for s in updated[4:7]], ["AC254-030-C"] * 3)
        self.assertAlmostEqual(updated[6].thickness, 45.0)

    def test_thorlabs_catalog_accessible_when_no_lens_elements(self):
        """When system has no solid lens assemblies, Thorlabs catalog buttons and dialog remain accessible."""
        empty_stack = SurfaceStack()
        empty_stack.add_surface(Surface(name="OBJ", thickness=20.0))
        empty_stack.add_surface(Surface(name="Stop", thickness=30.0))
        empty_stack.add_surface(Surface(name="IMA", surface_type=SurfaceType.DETECTOR, thickness=0.0))
        model = LensDataTableModel(stack=empty_stack)
        dock = ComponentEditorDockWidget(model=model)

        self.assertFalse(hasattr(dock, "btn_catalog_lens"))
        self.assertTrue(hasattr(dock, "btn_replace_lens"))
        self.assertEqual(dock.lens_element_combo.count(), 1)
        self.assertIn("No multi-surface lens assembly", dock.lens_element_combo.itemText(0))

        # Test CatalogLensDialog fallback target population
        from optics_app.gui.catalog_lens_dialog import CatalogLensDialog
        dlg = CatalogLensDialog(model=model, target=1, mode="replace")
        self.assertGreaterEqual(dlg.target_combo.count(), 2)
        # Should include Surface 1 as a fallback target
        combo_texts = [dlg.target_combo.itemText(i) for i in range(dlg.target_combo.count())]
        self.assertTrue(any("Surface 1" in t for t in combo_texts))
        # Catalog table should contain Thorlabs lenses including aspherics and achromats
        self.assertGreaterEqual(dlg.lens_table.rowCount(), 20)


class TestMainWindowComponentEditorIntegration(unittest.TestCase):
    """Integration tests for MainWindow, ComponentEditorDockWidget, and real-time debouncing."""

    def test_main_window_has_component_editor_dock(self):
        """MainWindow embeds ComponentEditorDockWidget and provides accessor."""
        window = MainWindow()
        dock = window.get_component_editor_dock()
        self.assertIsNotNone(dock)
        self.assertIsInstance(dock, ComponentEditorDockWidget)

    def test_debounced_trace_trigger(self):
        """Changing component slider triggers debounced ray trace without crashing."""
        window = MainWindow()
        dock = window.get_component_editor_dock()
        dock.set_selected_surface(1)

        # Trigger debounced trace
        window.request_debounced_trace()
        window._on_debounced_trace_timeout()
        self.assertIn("Ray trace complete", getattr(window, "_status_msg", ""))

    def test_lde_table_mirror_actions(self):
        """LDE table model methods for mirror fold presets and aiming execute correctly."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=50.0))
        stack.add_surface(Surface(name="Mirror1", glass="MIRROR", thickness=50.0))
        stack.add_surface(Surface(name="IMA", surface_type=SurfaceType.DETECTOR, thickness=0.0))

        model = LensDataTableModel(stack=stack)
        tx, ty = model.set_mirror_fold_angle(1, fold_angle_deg=90.0, fold_plane="YZ")
        self.assertAlmostEqual(tx, 45.0, places=2)

        success = model.aim_mirror_to_target(1, target_idx=2)
        self.assertTrue(success)


class TestHeadlessQtMocks(unittest.TestCase):
    """Test headless mock widget classes in qt_compat."""

    def test_mock_slider(self):
        from optics_app.gui.qt_compat import DummyQtWidgetsModule
        slider = DummyQtWidgetsModule.QSlider()
        slider.setRange(-100, 100)
        slider.setValue(42)
        self.assertEqual(slider.value(), 42)

    def test_mock_dial(self):
        from optics_app.gui.qt_compat import DummyQtWidgetsModule
        dial = DummyQtWidgetsModule.QDial()
        dial.setRange(0, 360)
        dial.setValue(90)
        self.assertEqual(dial.value(), 90)

    def test_mock_timer(self):
        from optics_app.gui.qt_compat import DummyQtCoreModule
        timer = DummyQtCoreModule.QTimer()
        timer.setInterval(25)
        self.assertEqual(timer.interval(), 25)
        timer.setSingleShot(True)
        self.assertTrue(timer.isSingleShot())
        timer.start()
        self.assertTrue(timer.isActive())
        timer.stop()
        self.assertFalse(timer.isActive())

    def test_mock_dialog(self):
        from optics_app.gui.qt_compat import DummyQtWidgetsModule
        dialog = DummyQtWidgetsModule.QDialog()
        res = dialog.exec()
        self.assertEqual(res, 1)

    def test_mock_grid_layout(self):
        from optics_app.gui.qt_compat import DummyQtWidgetsModule
        grid = DummyQtWidgetsModule.QGridLayout()
        w = DummyQtWidgetsModule.QWidget()
        grid.addWidget(w, 0, 0)
        self.assertEqual(len(grid._items), 1)


if __name__ == "__main__":
    unittest.main()
