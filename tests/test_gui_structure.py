"""
Unit tests for PyQt/PySide Desktop GUI (Lens Data Editor Table Model, Dual-View Layout, Analysis Tabs, and MainWindow).
Covers Tiers 1-4 per TEST_INFRA.md (Features 14, 15, 16).
"""

import unittest
import numpy as np

try:
    from optics_app.gui.lde_table import LensDataTableModel
    from optics_app.gui.dual_view import DualViewWidget
    from optics_app.gui.analysis_tabs import AnalysisTabsWidget
    from optics_app.gui.main_window import MainWindow
    from optics_app.engine.lde import SurfaceStack
    from optics_app.engine.surface import Surface, SurfaceType
    OPTICS_APP_GUI_AVAILABLE = True
except ImportError:
    OPTICS_APP_GUI_AVAILABLE = False

from optics_app.gui.qt_compat import QT_AVAILABLE, QtWidgets

if QT_AVAILABLE and hasattr(QtWidgets, "QApplication"):
    try:
        _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    except Exception:
        pass


class TestGUIStructureTier1(unittest.TestCase):
    """Tier 1: Basic PyQt/PySide Lens Data Editor Model & Widget Interface Architecture."""

    def test_t1_1_table_model_initialization(self):
        model = LensDataTableModel()
        self.assertGreaterEqual(model.rowCount(), 3)
        self.assertGreaterEqual(model.columnCount(), 14)

    def test_t1_2_table_header_labels(self):
        model = LensDataTableModel()
        self.assertEqual(model.headerData(0, None), "Surface")
        self.assertEqual(model.headerData(2, None), "Radius (mm)")
        self.assertEqual(model.headerData(4, None), "Glass")

    def test_t1_3_insert_surface_row(self):
        model = LensDataTableModel()
        initial_rows = model.rowCount()
        success = model.insertRows(1, 1)
        self.assertTrue(success)
        self.assertEqual(model.rowCount(), initial_rows + 1)

    def test_t1_4_delete_surface_row(self):
        model = LensDataTableModel()
        initial_rows = model.rowCount()
        success = model.removeRows(1, 1)
        self.assertTrue(success)
        self.assertEqual(model.rowCount(), initial_rows - 1)

    def test_t1_5_set_and_get_cell_data(self):
        model = LensDataTableModel()
        # Set radius on surface 1 (row 1, col 2)
        success = model.setData(1, 2, 75.5)
        self.assertTrue(success)
        self.assertEqual(model.data(1, 2), 75.5)


class TestGUIStructureTier2(unittest.TestCase):
    """Tier 2: Boundary Conditions & Invalid Input Handling in GUI Model."""

    def test_t2_1_invalid_index_access_returns_none(self):
        model = LensDataTableModel()
        self.assertIsNone(model.data(-1, 0))
        self.assertIsNone(model.data(100, 0))
        self.assertIsNone(model.data(0, 100))

    def test_t2_2_invalid_numeric_data_input(self):
        model = LensDataTableModel()
        # Enter string 'abc' into radius column (col 2)
        success = model.setData(1, 2, "abc_invalid_number")
        self.assertFalse(success)

    def test_t2_3_delete_out_of_bounds_row(self):
        model = LensDataTableModel()
        success = model.removeRows(99, 1)
        self.assertFalse(success)

    def test_t2_4_analysis_tab_switch_invalid_name(self):
        model = LensDataTableModel()
        tabs = AnalysisTabsWidget(model)
        success = tabs.set_active_tab("INVALID_TAB_XYZ")
        self.assertFalse(success)
        self.assertEqual(tabs.active_tab, "Spot Diagram")


class TestGUIStructureTier3(unittest.TestCase):
    """Tier 3: LDE Table Model Modification Signals Dual-View & Analysis Tabs."""

    def test_t3_1_model_update_signal_emission(self):
        model = LensDataTableModel()
        dual_view = DualViewWidget(model)
        analysis_tabs = AnalysisTabsWidget(model)

        initial_signals = model.update_signals_emitted
        model.setData(1, 3, 12.0)  # Change thickness
        self.assertGreater(model.update_signals_emitted, initial_signals)
        self.assertTrue(dual_view.refresh_layout())


class TestGUIStructureTier4(unittest.TestCase):
    """Tier 4: Complete Desktop GUI LDE Workflow Execution."""

    def test_t4_1_full_gui_lde_session_simulation(self):
        model = LensDataTableModel()
        dual_view = DualViewWidget(model)
        analysis_tabs = AnalysisTabsWidget(model)

        # 1. Add 2 lens surfaces
        model.insertRows(1, 2)
        # 2. Modify lens 1: R=50.0, Th=8.0, Glass=N-BK7
        model.setData(1, 1, "Standard")
        model.setData(1, 2, 50.0)
        model.setData(1, 3, 8.0)
        model.setData(1, 4, "N-BK7")

        # 3. Switch between analysis tabs
        for tab in ["Spot Diagram", "Ray Fan", "Grating Dispersion", "CCD Heatmap"]:
            self.assertTrue(analysis_tabs.set_active_tab(tab))
            self.assertEqual(analysis_tabs.active_tab, tab)

        self.assertGreaterEqual(model.rowCount(), 5)


class TestGUIStructureMainWindow(unittest.TestCase):
    """Tier 5: MainWindow Integration and File I/O Operations."""

    def test_t5_1_main_window_instantiation(self):
        window = MainWindow()
        self.assertIsNotNone(window.get_model())
        self.assertIsNotNone(window.get_dual_view())
        self.assertIsNotNone(window.get_analysis_tabs())

    def test_t5_2_main_window_surface_modification_slots(self):
        window = MainWindow()
        model = window.get_model()
        initial_rows = model.rowCount()

        window.on_add_surface()
        self.assertEqual(model.rowCount(), initial_rows + 1)

        window.on_remove_surface()
        self.assertEqual(model.rowCount(), initial_rows)

        # Trigger ray tracing manually
        window.on_trace_rays()


class TestGUIStructureEngineBinding(unittest.TestCase):
    """Tier 6: 2-way Binding with SurfaceStack Engine."""

    def test_t6_1_stack_to_model_sync(self):
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", surface_type=SurfaceType.OBJECT, thickness=100.0))
        stack.add_surface(Surface(name="Lens1", surface_type=SurfaceType.STANDARD, radius=60.0, thickness=10.0, glass="N-BK7", semi_diameter=20.0))
        stack.add_surface(Surface(name="IMA", surface_type=SurfaceType.DETECTOR, thickness=0.0))

        model = LensDataTableModel(stack=stack)
        self.assertEqual(model.rowCount(), 3)
        self.assertEqual(model.data(1, 2), 60.0)
        self.assertEqual(model.data(1, 4), "N-BK7")

        rebuilt_stack = model.get_surface_stack()
        self.assertEqual(len(rebuilt_stack.surfaces), 3)
        self.assertEqual(rebuilt_stack.surfaces[1].radius, 60.0)
        self.assertEqual(rebuilt_stack.surfaces[1].glass, "N-BK7")

    def test_t6_2_row_move_metadata_preservation(self):
        stack = SurfaceStack()
        s0 = Surface(name="OBJ", surface_type=SurfaceType.OBJECT, thickness=100.0)
        s1 = Surface(name="L1_Front", surface_type=SurfaceType.ASPHERIC, radius=50.0, thickness=10.0, glass="N-BK7", conic=-1.0, asphere_coeffs=[1e-5, 2e-7])
        s2 = Surface(name="Stop_Surf", surface_type=SurfaceType.STOP, radius=0.0, thickness=20.0, aperture_type="Spider", aperture_center_radius=3.0, aperture_spider_count=4)
        s3 = Surface(name="IMA", surface_type=SurfaceType.DETECTOR, radius=0.0, thickness=0.0)

        s1.cad_profile = "DUMMY_CAD_PROFILE_S1"
        s1.is_active = True
        s1.grating_order = 2
        s1.jones_matrix = np.array([[1, 0], [0, 0]], dtype=complex)

        for s in [s0, s1, s2, s3]:
            stack.add_surface(s)

        model = LensDataTableModel(stack)
        self.assertTrue(model.move_surface_down(1))

        new_stack = model.get_surface_stack()
        surfs = new_stack.surfaces

        # s1 (L1_Front) moved from index 1 to index 2
        self.assertIs(surfs[2], s1)
        self.assertEqual(surfs[2].asphere_coeffs, [1e-5, 2e-7])
        self.assertEqual(surfs[2].cad_profile, "DUMMY_CAD_PROFILE_S1")
        self.assertTrue(surfs[2].is_active)
        self.assertEqual(surfs[2].grating_order, 2)
        np.testing.assert_array_equal(surfs[2].jones_matrix, np.array([[1, 0], [0, 0]], dtype=complex))

        # s2 (Stop_Surf) is now at index 1
        self.assertIs(surfs[1], s2)
        self.assertEqual(surfs[1].asphere_coeffs, [])
        self.assertIsNone(surfs[1].cad_profile)


if __name__ == "__main__":
    unittest.main()

