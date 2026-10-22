"""
optics_app.gui Package Exports.
"""
from optics_app.gui import qt_compat
from optics_app.gui.lde_table import LensDataTableModel
from optics_app.gui.delegates import (
    SurfaceTypeDelegate, GlassSelectionDelegate, NumericValueDelegate, CADFilePathDelegate
)
from optics_app.gui.dual_view import DualViewWidget
from optics_app.gui.analysis_tabs import AnalysisTabsWidget
from optics_app.gui.main_window import MainWindow
from optics_app.gui.step_export_dialog import STEPExportDialog
try:
    from optics_app.gui.illumination_dock import IlluminationDockWidget
except ImportError:
    IlluminationDockWidget = None

OPTICS_APP_GUI_AVAILABLE = True

__all__ = [
    "qt_compat",
    "LensDataTableModel",
    "SurfaceTypeDelegate",
    "GlassSelectionDelegate",
    "NumericValueDelegate",
    "CADFilePathDelegate",
    "DualViewWidget",
    "AnalysisTabsWidget",
    "MainWindow",
    "STEPExportDialog",
    "IlluminationDockWidget",
    "OPTICS_APP_GUI_AVAILABLE",
]
