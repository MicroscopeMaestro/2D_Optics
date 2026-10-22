"""
Unit tests verifying that optimized achromat prescriptions and dimensions
are fully and visibly delivered into the active application model, LDE table,
visualizer, and optimizer dock.
"""
import os, sys
os.environ["QT_QPA_PLATFORM"] = "offscreen"
import pytest
from unittest.mock import MagicMock

from optics_app.gui.qt_compat import QT_AVAILABLE, QtWidgets
if QT_AVAILABLE:
    _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.gui.lde_table import LensDataTableModel
from optics_app.gui.achromat_dialog import AchromatDesignerDialog
from optics_app.gui.analysis_tabs import AchromatAnalysisTab
from optics_app.gui.optimizer_dock import OptimizerDockWidget
from optics_app.engine.achromat_designer import AchromatDesignSpec, optimize_achromatic_doublet
from optics_app.engine.optimizer import compute_system_efl


def test_achromat_dialog_delivery_to_model():
    """Verify AchromatDesignerDialog inserts doublet into LensDataTableModel."""
    model = LensDataTableModel()
    assert model.rowCount() == 3 # Default: S0, S1, S2

    dialog = AchromatDesignerDialog(model=model)
    # Ensure current_result is computed
    assert dialog.current_result is not None
    dialog.accept = MagicMock()

    # Trigger insertion
    dialog._insert_into_project()

    # The model must now contain the complete 5-surface doublet system:
    # S0 (OBJ), S1 (STO/Crown), S2 (Contact/Flint), S3 (Rear/Air), S4 (IMA)
    assert model.rowCount() == 5
    stack = model.get_surface_stack()
    assert len(stack.surfaces) == 5

    s1 = stack.surfaces[1]
    s2 = stack.surfaces[2]
    s3 = stack.surfaces[3]

    assert s1.glass == dialog.current_result.crown.name
    assert s2.glass == dialog.current_result.flint.name
    assert s3.glass == "AIR"
    assert s1.radius != 0.0
    assert s3.thickness > 0.0

    # LDE table internal data must be fully populated
    assert model._data[1][4] == dialog.current_result.crown.name
    assert model._data[2][4] == dialog.current_result.flint.name
    assert model._data[3][3] == float(s3.thickness)


def test_achromat_analysis_tab_insert_to_model():
    """Verify AchromatAnalysisTab 'Insert Doublet into System' updates model."""
    model = LensDataTableModel()
    tab = AchromatAnalysisTab(model=model)
    tab.spin_efl.setValue(50.0)
    tab._on_optimize_clicked()

    assert tab.last_result is not None
    assert abs(tab.last_result.efl - 50.0) < 0.5

    # Insert into active system
    tab._on_insert_into_system()

    assert model.rowCount() == 5
    stack = model.get_surface_stack()
    assert stack.surfaces[1].glass == "N-BK7"
    assert stack.surfaces[2].glass == "SF10"
    efl, bfl = compute_system_efl(stack.surfaces, 0.5876)
    assert abs(efl - 50.0) < 1.0
    assert abs(bfl - stack.surfaces[3].thickness) < 0.1


def test_optimizer_dock_seed_achromat_from_scratch():
    """Verify OptimizerDock 'Seed Paraxial Achromat' works even on empty/default system."""
    model = LensDataTableModel()
    dock = OptimizerDockWidget(model=model)
    dock.target_efl_spin.setValue(50.0)

    # Click seed on default 3-surface system (no initial glasses)
    dock._on_seed_achromat_clicked()

    # Should have auto-generated the 5-surface doublet system
    assert model.rowCount() == 5
    stack = model.get_surface_stack()
    assert stack.surfaces[1].glass == "N-BK7"
    assert stack.surfaces[2].glass == "N-SF2"
    efl, _ = compute_system_efl(stack.surfaces, 0.5876)
    assert abs(efl - 50.0) < 1.0


def test_optimizer_dock_seed_existing_glasses_updates_data():
    """Verify OptimizerDock 'Seed Paraxial Achromat' on existing doublet updates LDE _data."""
    stack = SurfaceStack()
    stack.add_surface(Surface(name="OBJ", thickness=100.0))
    stack.add_surface(Surface(name="S1", radius=100.0, thickness=5.0, glass="N-BK7"))
    stack.add_surface(Surface(name="S2", radius=0.0, thickness=3.0, glass="N-SF2"))
    stack.add_surface(Surface(name="S3", radius=-100.0, thickness=30.0, glass="AIR"))
    stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0))

    model = LensDataTableModel(stack=stack)
    dock = OptimizerDockWidget(model=model)
    dock.target_efl_spin.setValue(50.0)

    dock._on_seed_achromat_clicked()

    # Radii in model._data must have been updated
    assert model._data[1][2] != 100.0 # R1 changed
    assert model._data[3][3] != 30.0  # BFL thickness changed
