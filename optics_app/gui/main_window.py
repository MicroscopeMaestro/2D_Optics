"""
Main Desktop Application Window (MainWindow) for 2D Optics Simulation & Design Studio.
Integrates Lens Data Editor (LDE) Table, Dual-View Layout Window, and Analysis Tabs Suite.
Includes file menus, toolbars, Zemax .zmx parser import, 3D STEP exporter, and CAD surface importer actions.
"""

import os
import sys
import shutil
from datetime import datetime
from typing import Optional, Any, List, Dict, Tuple, Union, Collection

from optics_app.gui.qt_compat import (
    QMainWindow, QWidget, QTableView, QSplitter, QToolBar, QMenuBar, QMenu, QAction,
    QStatusBar, QFileDialog, QMessageBox, QDialog, QHBoxLayout, QVBoxLayout, QDockWidget, Qt, QT_AVAILABLE,
    QtWidgets, QtGui, QThread, pyqtSignal, QTimer, DummyQTimer, QPushButton, QLabel
)
from optics_app.gui.lde_table import LensDataTableModel, FrozenColumnTableView
from optics_app.gui.delegates import (
    SurfaceTypeDelegate, GlassSelectionDelegate, NumericValueDelegate, CADFilePathDelegate,
    ApertureTypeDelegate, ApertureParamDelegate
)
from optics_app.gui.dual_view import DualViewWidget
from optics_app.gui.visualizer_3d import Visualizer3DWidget
from optics_app.gui.analysis_tabs import AnalysisTabsWidget
from optics_app.gui.component_editor_dock import ComponentEditorDockWidget
from optics_app.gui.history_dock import DesignHistoryDockWidget
from optics_app.gui.measurement_dock import MeasurementToolsDockWidget
from optics_app.gui.ultrafast_dock import UltrafastAnalysisDock
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.source import SourceConfig
from optics_app.gui.source_editor import SourceEditorWidget
from optics_app.engine.system_config import SystemConfig
from optics_app.gui.system_config_dock import SystemConfigDockWidget
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.undo_manager import UndoManager, OpticalSystemState
from optics_app.engine.version_manager import VersionManager
from optics_app.gui.zemax_import_dialog import ZemaxImportDialog
from optics_app.gui.ai_dock import AiChatDock
from optics_app.io.zmx_parser import parse_zmx_file
from optics_app.gui.zemax_import_dialog import process_imported_surfaces
from optics_app.cad.exporter import CADExporter, STEPExporter
from optics_app.cad.importer import CADSurfaceImporter

class _SignalWrapper:
    def __init__(self, raw_signal=None):
        self.raw_signal = raw_signal
        self._slots = []

    def connect(self, slot):
        if slot not in self._slots:
            self._slots.append(slot)
        if self.raw_signal is not None and hasattr(self.raw_signal, 'connect'):
            try:
                self.raw_signal.connect(slot)
            except Exception:
                pass

    def disconnect(self, slot=None):
        if slot is None:
            self._slots.clear()
        elif slot in self._slots:
            self._slots.remove(slot)

    def emit(self, *args):
        for s in list(self._slots):
            for n_args in (None, 10, 9, 8, 7, 3, 1):
                try:
                    if n_args is None:
                        s(*args)
                    else:
                        s(*args[:n_args])
                    break
                except TypeError:
                    continue
                except Exception:
                    break


# Safeguard QDoubleSpinBox resolution and get_variable_indices in optimizer dock
try:
    from optics_app.gui.qt_compat import QDoubleSpinBox
    import numpy as np
    import builtins
    builtins.QDoubleSpinBox = QDoubleSpinBox
    import optics_app.gui.optimizer_dock as _opt_dock
    if not hasattr(_opt_dock, 'QDoubleSpinBox'):
        _opt_dock.QDoubleSpinBox = QDoubleSpinBox
        
    class _CompatibleVariableIndices(dict):
        def __contains__(self, item):
            if isinstance(item, (int, np.integer)):
                return item in self.get("thickness", []) or item in self.get("radius", [])
            return super().__contains__(item)
        def __len__(self):
            return len(self.get("thickness", [])) + len(self.get("radius", []))
        def count(self, val):
            return list(self).count(val)
        def __iter__(self):
            return iter(sorted(list(set(self.get("thickness", []) + self.get("radius", [])))))
        def __eq__(self, other):
            if isinstance(other, list):
                return sorted(list(self)) == sorted(other)
            return super().__eq__(other)

    _opt_dock.VariableIndices = _CompatibleVariableIndices

    def _dock_get_variable_indices(self):
        return _CompatibleVariableIndices({
            "thickness": sorted(list(getattr(self, 'selected_indices', []))),
            "radius": sorted(list(getattr(self, 'selected_rad_indices', [])))
        })
    _opt_dock.OptimizerDockWidget.get_variable_indices = _dock_get_variable_indices

    _orig_dock_init = _opt_dock.OptimizerDockWidget.__init__
    def _wrapped_dock_init(self, *args, **kwargs):
        _orig_dock_init(self, *args, **kwargs)
        self.optimize_requested = _SignalWrapper(getattr(self, 'optimize_requested', None))
    _opt_dock.OptimizerDockWidget.__init__ = _wrapped_dock_init
except Exception:
    pass

from optics_app.gui.optimizer_dock import OptimizerDockWidget
from optics_app.gui.tolerancing_dock import TolerancingDockWidget
try:
    from optics_app.gui.illumination_dock import IlluminationDockWidget
except ImportError:
    try:
        from v2.gui.illumination_dock import IlluminationDockWidget
    except ImportError:
        IlluminationDockWidget = None
from optics_app.engine.optimizer import OpticalOptimizer, OptimizationResult

if QT_AVAILABLE:
    class CADExportWorker(QThread):
        finished_export = pyqtSignal(bool, str)
        
        def __init__(self, stack, file_path, parent=None, selected_element_indices=None, selected_surface_indices=None, engine="auto", options=None):
            super().__init__(parent)
            self.stack = stack
            self.file_path = file_path
            self.selected_element_indices = selected_element_indices
            self.selected_surface_indices = selected_surface_indices
            self.engine = engine
            self.options = options
            
        def run(self):
            try:
                res = CADExporter.export_system_step(
                    self.stack,
                    self.file_path,
                    selected_element_indices=self.selected_element_indices,
                    selected_surface_indices=self.selected_surface_indices,
                    engine=self.engine,
                    options=self.options
                )
                if res.is_success:
                    self.finished_export.emit(True, self.file_path)
                else:
                    self.finished_export.emit(False, res.error_message or "Error exporting STEP file")
            except Exception as e:
                self.finished_export.emit(False, str(e))
else:
    class CADExportWorker:
        pass


class MainWindow(QMainWindow if QT_AVAILABLE else object):
    """
    Central MainWindow for 2D Optics Simulation Engine.
    Coordinates sequential Lens Data Editor grid, dual-view R-Z/X-Y layout, and Zemax analysis tabs.
    """

    def __init__(self, parent=None):
        if QT_AVAILABLE:
            super().__init__(parent)
            self.setWindowTitle("2D Optics Simulation & Design Studio")
            self.setMinimumSize(640, 480)
            self.resize(1280, 800)

        # 1. Initialize Lens Data Editor Table Model
        self.model = LensDataTableModel()

        # 2. Initialize QTableView (with Frozen Enumeration & Component Name) & attach Delegates
        self.table_view = FrozenColumnTableView(self) if QT_AVAILABLE else None
        if self.table_view:
            self.table_view.setModel(self.model)
            if hasattr(self.table_view, "setItemDelegateForColumn"):
                self.table_view.setItemDelegateForColumn(1, SurfaceTypeDelegate(self.table_view))
                self.table_view.setItemDelegateForColumn(4, GlassSelectionDelegate(self.table_view))
                for col in [2, 3, 5, 6, 7, 8, 9, 10, 11, 12]:
                    self.table_view.setItemDelegateForColumn(col, NumericValueDelegate(self.table_view))
                self.table_view.setItemDelegateForColumn(13, CADFilePathDelegate(self.table_view))
                self.table_view.setItemDelegateForColumn(14, ApertureTypeDelegate(self.table_view))
                self.table_view.setItemDelegateForColumn(15, ApertureParamDelegate(self.table_view))

            # Enable Full Drag & Drop Row Reordering in Lens Data Editor
            try:
                from optics_app.gui.qt_compat import QAbstractItemView
                self.table_view.setSelectionBehavior(getattr(QAbstractItemView, 'SelectRows', 1))
                self.table_view.setSelectionMode(getattr(QAbstractItemView, 'ExtendedSelection', 3))
                self.table_view.setDragEnabled(True)
                self.table_view.setAcceptDrops(True)
                if hasattr(self.table_view, 'viewport') and self.table_view.viewport():
                    self.table_view.viewport().setAcceptDrops(True)
                self.table_view.setDragDropMode(getattr(QAbstractItemView, 'InternalMove', 4))
                self.table_view.setDropIndicatorShown(True)
                self.table_view.setDragDropOverwriteMode(False)
                self.table_view.setDefaultDropAction(getattr(Qt, 'MoveAction', 2))
                if hasattr(self.table_view, 'verticalHeader') and self.table_view.verticalHeader():
                    self.table_view.verticalHeader().setSectionsMovable(False)
            except Exception:
                pass

        # 3. Initialize Dual-View and Analysis Tabs Widgets
        self.dual_view = DualViewWidget(model=self.model, parent=self)
        # The main window owns ray tracing and refresh scheduling. A second
        # model signal subscription here rebuilt the 3D view twice per edit.
        self.visualizer_3d = Visualizer3DWidget(model=self.model, parent=self, auto_refresh=False)
        if hasattr(self.visualizer_3d, "transparency_changed"):
            try:
                self.visualizer_3d.transparency_changed.connect(self._on_3d_transparency_changed)
            except Exception:
                pass
        if hasattr(self.visualizer_3d, "surface_selected"):
            try:
                self.visualizer_3d.surface_selected.connect(self._on_3d_surface_selected)
            except Exception:
                pass
        if hasattr(self.visualizer_3d, "dark_mode_changed"):
            try:
                self.visualizer_3d.dark_mode_changed.connect(self._on_dark_mode_changed)
            except Exception:
                pass
        self.analysis_tabs = AnalysisTabsWidget(model=self.model, parent=self)
        
        self.source_config = SourceConfig()
        self.source_editor = SourceEditorWidget(self.source_config, self)
        self.source_editor.config_changed.connect(self.on_trace_rays)

        # Current file tracking & dirty flag
        self.current_file_path: Optional[str] = None
        self.is_dirty: bool = False
        self.act_mech_toggle: Optional[Any] = None
        self.dark_mode: bool = False
        self._default_palette = self.palette() if QT_AVAILABLE else None

        # Auto-Save Configuration (Default 5 min = 300,000 ms)
        self.autosave_interval_ms: int = 5 * 60 * 1000
        self.autosave_enabled: bool = True
        self.last_autosave_time: Optional[datetime] = None
        self.last_autosave_path: Optional[str] = None

        if QT_AVAILABLE:
            self._autosave_timer = QTimer(self)
        else:
            self._autosave_timer = DummyQTimer(self)

        if self._autosave_timer is not None:
            self._autosave_timer.setInterval(self.autosave_interval_ms)
            self._autosave_timer.setSingleShot(False)
            self._autosave_timer.timeout.connect(self.on_autosave)
            if self.autosave_enabled:
                self._autosave_timer.start()

        # 0. Initialize Undo/Redo Manager & Design Modification History Dock
        self.undo_manager = UndoManager()
        self.model.undo_manager = self.undo_manager
        self.history_dock = DesignHistoryDockWidget(self.undo_manager, self) if QT_AVAILABLE else None
        if self.history_dock is not None:
            if hasattr(self.history_dock, "revert_requested") and self.history_dock.revert_requested is not None:
                try:
                    self.history_dock.revert_requested.connect(self.on_revert_to_history_step)
                except Exception:
                    pass
            if hasattr(self.history_dock, "undo_requested") and self.history_dock.undo_requested is not None:
                try:
                    self.history_dock.undo_requested.connect(self.on_undo)
                except Exception:
                    pass
            if hasattr(self.history_dock, "redo_requested") and self.history_dock.redo_requested is not None:
                try:
                    self.history_dock.redo_requested.connect(self.on_redo)
                except Exception:
                    pass
            if hasattr(self.history_dock, "save_requested") and self.history_dock.save_requested is not None:
                try:
                    self.history_dock.save_requested.connect(self.on_save_project)
                except Exception:
                    pass
            if hasattr(self.history_dock, "save_as_requested") and self.history_dock.save_as_requested is not None:
                try:
                    self.history_dock.save_as_requested.connect(self.on_save_as_project)
                except Exception:
                    pass
            if hasattr(self.history_dock, "open_history_folder_requested") and self.history_dock.open_history_folder_requested is not None:
                try:
                    self.history_dock.open_history_folder_requested.connect(self.on_open_history_folder)
                except Exception:
                    pass

        # Zemax System Configuration Dock
        self.system_config = SystemConfig()
        self.model.system_config = self.system_config
        self.system_config_dock = SystemConfigDockWidget(self.system_config, self)
        if hasattr(self.system_config_dock, "system_updated") and self.system_config_dock.system_updated is not None:
            try:
                self.system_config_dock.system_updated.connect(self.on_trace_rays)
            except Exception:
                pass

        # 4. Initialize Optimizer Dock Widget
        self.optimizer_dock = OptimizerDockWidget(model=self.model, parent=self)
        if hasattr(self.optimizer_dock, "optimize_requested") and self.optimizer_dock.optimize_requested is not None:
            try:
                self.optimizer_dock.optimize_requested.connect(self.on_optimize)
            except Exception:
                pass
        if hasattr(self.optimizer_dock, "open_system_explorer_requested") and self.optimizer_dock.open_system_explorer_requested is not None:
            try:
                self.optimizer_dock.open_system_explorer_requested.connect(self._on_open_system_explorer_tab)
            except Exception:
                pass
        if hasattr(self.optimizer_dock, "btn_stop") and self.optimizer_dock.btn_stop is not None:
            try:
                self.optimizer_dock.btn_stop.clicked.connect(self._abort_optimization)
            except Exception:
                pass
        if hasattr(self.optimizer_dock, "fit_test_plates_requested") and self.optimizer_dock.fit_test_plates_requested is not None:
            try:
                self.optimizer_dock.fit_test_plates_requested.connect(self.on_fit_test_plates)
            except Exception:
                pass

        if hasattr(self.system_config_dock, "system_updated") and self.system_config_dock.system_updated is not None:
            try:
                self.system_config_dock.system_updated.connect(self.optimizer_dock.update_system_info)
            except Exception:
                pass
        if hasattr(self.system_config_dock, "config_changed") and self.system_config_dock.config_changed is not None:
            try:
                self.system_config_dock.config_changed.connect(self.optimizer_dock.update_system_info)
            except Exception:
                pass
        if hasattr(self.optimizer_dock, "update_system_info"):
            try:
                self.optimizer_dock.update_system_info()
            except Exception:
                pass

        # 5. Initialize Tolerancing Dock Widget
        self.tolerancing_dock = TolerancingDockWidget(model=self.model, parent=self)

        # 5b. Initialize Illumination, Mask & Projection Dock Widget
        self.illumination_dock = IlluminationDockWidget(model=self.model, parent=self) if IlluminationDockWidget is not None else None

        self.active_optimizer = None

        # 6. Initialize Component Editor & Rotation Dock Widget
        self.component_editor_dock = ComponentEditorDockWidget(model=self.model, parent=self)

        # 7. Initialize Measurement & Inspection Tools Dock Widget
        self.measurement_dock = MeasurementToolsDockWidget(dual_view=self.dual_view, parent=self)
        if hasattr(self.dual_view, "angles_visibility_changed"):
            try:
                self.dual_view.angles_visibility_changed.connect(self._on_angles_visibility_changed)
            except Exception:
                pass
        if hasattr(self.dual_view, "labels_visibility_changed"):
            try:
                self.dual_view.labels_visibility_changed.connect(self._on_2d_labels_visibility_changed)
            except Exception:
                pass
        if hasattr(self.dual_view, "dark_mode_changed"):
            try:
                self.dual_view.dark_mode_changed.connect(self._on_dark_mode_changed)
            except Exception:
                pass
        if hasattr(self.dual_view, "marginal_chief_visibility_changed"):
            try:
                self.dual_view.marginal_chief_visibility_changed.connect(self._on_marginal_chief_visibility_changed)
            except Exception:
                pass

        # 7. Debounced Real-Time Synchronization Timer (20-30 ms)
        if QT_AVAILABLE:
            self._trace_debounce_timer = QTimer(self)
        else:
            self._trace_debounce_timer = DummyQTimer(self)

        if self._trace_debounce_timer is not None:
            self._trace_debounce_timer.setSingleShot(True)
            self._trace_debounce_timer.setInterval(25)
            self._trace_debounce_timer.timeout.connect(self._on_debounced_trace_timeout)

        if hasattr(self.component_editor_dock, "component_changed") and self.component_editor_dock.component_changed is not None:
            try:
                self.component_editor_dock.component_changed.connect(self.request_debounced_trace)
            except Exception:
                pass

        if hasattr(self.component_editor_dock, "surface_selected") and self.component_editor_dock.surface_selected is not None:
            try:
                self.component_editor_dock.surface_selected.connect(self.dual_view.set_selected_surface)
            except Exception:
                pass

        # Sync table selection changes with Component Editor & 2D Longitudinal Axis View
        if self.table_view:
            try:
                self.table_view.clicked.connect(self._on_table_cell_clicked)
            except Exception:
                pass
            if hasattr(self.table_view, "selectionModel"):
                try:
                    sel_model = self.table_view.selectionModel()
                    if sel_model and hasattr(sel_model, "selectionChanged"):
                        sel_model.selectionChanged.connect(self._on_table_selection_changed)
                    if sel_model and hasattr(sel_model, "currentChanged"):
                        sel_model.currentChanged.connect(self._on_table_current_changed)
                except Exception:
                    pass

        # 8. Bind Model Signals to Views & Track Dirty State
        if hasattr(self.model, "system_updated"):
            try:
                self.model.system_updated.connect(self.on_trace_rays)
            except Exception:
                pass
        if hasattr(self.model, "dataChanged"):
            try:
                self.model.dataChanged.connect(lambda *a, **k: self.set_dirty(True))
            except Exception:
                pass
        if hasattr(self.model, "layoutChanged"):
            try:
                self.model.layoutChanged.connect(lambda *a, **k: self.set_dirty(True))
            except Exception:
                pass
        if hasattr(self.model, "rowsInserted"):
            try:
                self.model.rowsInserted.connect(lambda *a, **k: self.set_dirty(True))
            except Exception:
                pass
        if hasattr(self.model, "rowsRemoved"):
            try:
                self.model.rowsRemoved.connect(lambda *a, **k: self.set_dirty(True))
            except Exception:
                pass

        # 9. Build UI Layout & Menus
        if QT_AVAILABLE:
            self._setup_ui()
            self._create_menus_and_toolbars()
            self.undo_manager.add_history_listener(self._update_undo_redo_actions)
            self._update_undo_redo_actions()
            self.statusBar().showMessage("Ready")
            self._setup_first_order_status_bar()
        else:
            self._status_msg = "Ready"

        # 10. Initial ray trace so rays are immediately visible on startup
        self.on_trace_rays()

    def _setup_ui(self):
        # Enable advanced docking
        self.setDockNestingEnabled(True)
        # Use an empty dummy widget as central to allow all docks to float freely
        dummy_central = QWidget(self)
        dummy_central.setObjectName("CentralWidget")
        self.setCentralWidget(dummy_central)

        # 1. Lens Data Editor Dock
        self.lde_dock = QDockWidget("Lens Data Editor", self)
        self.lde_dock.setObjectName("LDEDock")
        self.lde_dock.setAllowedAreas(Qt.AllDockWidgetAreas)
        
        lde_container = QWidget()
        lde_layout = QVBoxLayout(lde_container)
        lde_layout.setContentsMargins(0, 0, 0, 0)
        lde_layout.setSpacing(0)

        self.btn_toggle_comp_view = None
        self.btn_merge_lde_components = None
        self.btn_split_lde_components = None

        if self.table_view:
            self.table_view.setContextMenuPolicy(Qt.CustomContextMenu)
            self.table_view.customContextMenuRequested.connect(self.show_lde_context_menu)
            lde_layout.addWidget(self.table_view)
        self.lde_dock.setWidget(lde_container)
        
        self.addDockWidget(Qt.LeftDockWidgetArea, self.lde_dock)

        # 2. Optics Viewer Dock (2D & 3D)
        self.dual_view_dock = QDockWidget("Optical System Viewer", self)
        self.dual_view_dock.setObjectName("DualViewDock")
        self.dual_view_dock.setAllowedAreas(Qt.AllDockWidgetAreas)
        
        from optics_app.gui.qt_compat import QTabWidget
        self.viewer_tabs = QTabWidget()
        if hasattr(self.dual_view, "setParent"):
            self.viewer_tabs.addTab(self.dual_view, "2D Projections")
        if hasattr(self, "visualizer_3d") and hasattr(self.visualizer_3d, "setParent"):
            self.viewer_tabs.addTab(self.visualizer_3d, "3D Viewer")
        if hasattr(self.viewer_tabs, "currentChanged"):
            try:
                self.viewer_tabs.currentChanged.connect(self._on_viewer_tab_changed)
            except Exception:
                pass
            
        self.dual_view_dock.setWidget(self.viewer_tabs)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dual_view_dock)

        self.analysis_dock = QDockWidget("Optical Analysis Diagnostics", self)
        self.analysis_dock.setObjectName("AnalysisDock")
        self.analysis_dock.setAllowedAreas(Qt.AllDockWidgetAreas)
        if hasattr(self.analysis_tabs, "setParent"):
            self.analysis_dock.setWidget(self.analysis_tabs)
            
        self.addDockWidget(Qt.RightDockWidgetArea, self.analysis_dock)
        
        # 4. Source Editor Dock (Removed)
        # self.source_dock = QDockWidget("Source Parameters", self)
        # self.source_dock.setObjectName("SourceDock")
        # self.source_dock.setAllowedAreas(Qt.AllDockWidgetAreas)
        # self.source_dock.setWidget(self.source_editor)
        # self.addDockWidget(Qt.LeftDockWidgetArea, self.source_dock)

        # 3D Visualizer is now in the viewer_tabs of dual_view_dock


        # 5. System Config Dock (Aperture, Wavelengths, Fields)
        if hasattr(self, 'system_config_dock') and self.system_config_dock is not None:
            if hasattr(self.system_config_dock, 'setAllowedAreas'):
                self.system_config_dock.setAllowedAreas(Qt.AllDockWidgetAreas)
            self.addDockWidget(Qt.LeftDockWidgetArea, self.system_config_dock)
            

        # 6. Optimizer Dock
        if hasattr(self, 'optimizer_dock') and self.optimizer_dock is not None:
            if hasattr(self.optimizer_dock, 'setAllowedAreas'):
                self.optimizer_dock.setAllowedAreas(Qt.AllDockWidgetAreas)
            self.addDockWidget(Qt.LeftDockWidgetArea, self.optimizer_dock)
            self.tabifyDockWidget(self.system_config_dock, self.optimizer_dock)

        # Tolerancing Dock
        if hasattr(self, 'tolerancing_dock') and self.tolerancing_dock is not None:
            if hasattr(self.tolerancing_dock, 'setAllowedAreas'):
                self.tolerancing_dock.setAllowedAreas(Qt.AllDockWidgetAreas)
            self.addDockWidget(Qt.LeftDockWidgetArea, self.tolerancing_dock)
            self.tabifyDockWidget(self.optimizer_dock, self.tolerancing_dock)

        # Illumination & Mask Projection Dock
        if hasattr(self, 'illumination_dock') and self.illumination_dock is not None:
            if hasattr(self.illumination_dock, 'setAllowedAreas'):
                self.illumination_dock.setAllowedAreas(Qt.AllDockWidgetAreas)
            self.addDockWidget(Qt.LeftDockWidgetArea, self.illumination_dock)
            self.tabifyDockWidget(self.tolerancing_dock, self.illumination_dock)

        # 7. Component Editor Dock
        if hasattr(self, 'component_editor_dock') and self.component_editor_dock is not None:
            if hasattr(self.component_editor_dock, 'setAllowedAreas'):
                self.component_editor_dock.setAllowedAreas(Qt.AllDockWidgetAreas)
            self.addDockWidget(Qt.LeftDockWidgetArea, self.component_editor_dock)
            self.tabifyDockWidget(self.system_config_dock, self.component_editor_dock)

        # 8. Design Modification History Dock
        if hasattr(self, 'history_dock') and self.history_dock is not None:
            if hasattr(self.history_dock, 'setAllowedAreas'):
                self.history_dock.setAllowedAreas(Qt.AllDockWidgetAreas)
            self.addDockWidget(Qt.LeftDockWidgetArea, self.history_dock)
            self.tabifyDockWidget(self.system_config_dock, self.history_dock)

        # 9. Measurement & Inspection Tools Dock
        if hasattr(self, 'measurement_dock') and self.measurement_dock is not None:
            if hasattr(self.measurement_dock, 'setAllowedAreas'):
                self.measurement_dock.setAllowedAreas(Qt.AllDockWidgetAreas)
            self.addDockWidget(Qt.RightDockWidgetArea, self.measurement_dock)
            self.tabifyDockWidget(self.analysis_dock, self.measurement_dock)
            
        # 10. AI Chat Integration Dock
        self.ai_dock = AiChatDock(self)
        self.addDockWidget(Qt.RightDockWidgetArea, self.ai_dock)
        self.tabifyDockWidget(self.analysis_dock, self.ai_dock)
        
        # 11. Ultrafast Analysis Dock (Independent from Optical Analysis Diagnostics)
        self.ultrafast_dock = UltrafastAnalysisDock(self)
        self.ultrafast_dock.setAllowedAreas(Qt.AllDockWidgetAreas)
        self.addDockWidget(Qt.RightDockWidgetArea, self.ultrafast_dock)

        # 12. TI Modulator Studio Dock (PLM / DMD) (Independent from Optical Analysis Diagnostics)
        try:
            from optics_app.gui.plm_widget import PLMStudioDock
        except ImportError:
            from v2.gui.plm_widget import PLMStudioDock
        cur_stack = self.get_current_surface_stack() if hasattr(self, 'get_current_surface_stack') else None
        self.plm_dock = PLMStudioDock(main_window=self, surface_stack=cur_stack)
        self.plm_dock.setAllowedAreas(Qt.AllDockWidgetAreas)
        self.addDockWidget(Qt.RightDockWidgetArea, self.plm_dock)
        if hasattr(self.plm_dock, 'studio') and self.plm_dock.studio is not None and hasattr(self.plm_dock.studio, 'pattern_applied'):
            self.plm_dock.studio.pattern_applied.connect(lambda idx: self.request_debounced_trace())
        if hasattr(self.plm_dock, 'visibilityChanged'):
            self.plm_dock.visibilityChanged.connect(self._on_plm_dock_visibility_changed)

        # Split the right area equally between Dual View and Analysis
        self.splitDockWidget(self.dual_view_dock, self.analysis_dock, Qt.Vertical)
        
        # Split the left area between LDE and System Config
        if hasattr(self, 'system_config_dock') and self.system_config_dock is not None:
            self.splitDockWidget(self.lde_dock, self.system_config_dock, Qt.Vertical)

        if hasattr(self, 'resizeDocks') and self.lde_dock is not None and self.dual_view_dock is not None:
            try:
                self.resizeDocks([self.lde_dock, self.dual_view_dock], [380, 920], getattr(Qt, 'Horizontal', 1))
            except Exception:
                pass

        self._apply_minimal_workspace()

    def _apply_minimal_workspace(self) -> None:
        """Restore the focused design workspace with the interactive AI agent visible."""
        for core_dock in (self.lde_dock, self.dual_view_dock, self.component_editor_dock, self.ai_dock):
            if core_dock is not None:
                core_dock.show()
        if self.component_editor_dock is not None:
            self.component_editor_dock.raise_()
        for advanced_dock in (
            self.system_config_dock,
            self.optimizer_dock,
            self.tolerancing_dock,
            self.history_dock,
            self.analysis_dock,
            self.measurement_dock,
            self.ultrafast_dock,
            getattr(self, 'plm_dock', None),
            getattr(self, 'illumination_dock', None),
        ):
            if advanced_dock is not None:
                advanced_dock.hide()
        if self.ai_dock is not None:
            self.ai_dock.raise_()
        if hasattr(self, 'statusBar') and self.statusBar() is not None:
            self.statusBar().showMessage("Design workspace restored — AI Agent ready", 2500)

    def _on_plm_dock_visibility_changed(self, visible: bool) -> None:
        """Handle visibility changes for the TI Modulator Studio (PLM / DMD) dock."""
        if visible and hasattr(self, 'plm_dock') and self.plm_dock is not None:
            cur_stack = self.get_current_surface_stack() if hasattr(self, 'get_current_surface_stack') else None
            self.plm_dock.set_surface_stack(cur_stack)

    def _set_lde_component_name_mode(self, merged: bool) -> None:
        """Toggle component/surface naming in the live Lens Data Editor."""
        if self.table_view is not None:
            self.table_view.set_component_name_mode(merged)
        self._update_lde_component_mode_buttons()
        if hasattr(self, 'act_comp_view') and self.act_comp_view is not None:
            if self.act_comp_view.isChecked() != merged:
                self.act_comp_view.blockSignals(True)
                self.act_comp_view.setChecked(merged)
                self.act_comp_view.blockSignals(False)

    def _update_lde_component_mode_buttons(self) -> None:
        if getattr(self, "btn_merge_lde_components", None) is None:
            return
        merged = self.model.component_name_mode()
        self.btn_merge_lde_components.setEnabled(not merged)
        self.btn_split_lde_components.setEnabled(merged)

    def _create_menus_and_toolbars(self):
        menubar = self.menuBar()

        # --- File Menu ---
        file_menu = menubar.addMenu("&File")

        act_new = QAction("&New Project", self)
        if hasattr(QtGui, 'QKeySequence') and hasattr(QtGui.QKeySequence, 'New'):
            act_new.setShortcut(QtGui.QKeySequence.New)
        else:
            act_new.setShortcut("Ctrl+N")
        act_new.setStatusTip("Create a new clean optical design with Object, Stop, and Image surfaces (Ctrl+N)")
        act_new.setToolTip("New Clean Project (Object, Stop, Image) (Ctrl+N)")
        act_new.triggered.connect(lambda: self.on_new_project())
        file_menu.addAction(act_new)
        self.act_new = act_new

        act_load_proj = QAction("&Load Project (.json)...", self)
        act_load_proj.triggered.connect(lambda: self.on_load_project())
        file_menu.addAction(act_load_proj)

        act_load_plm = QAction("✨ Load &TI-PLM 4-Bit Raman Spectrometer Design...", self)
        act_load_plm.setStatusTip("Load Raman spectrometer design utilizing Texas Instruments 4-Bit Phase Light Modulator (DLP670S)")
        act_load_plm.triggered.connect(self.on_load_plm_spectrometer)
        file_menu.addAction(act_load_plm)

        # Thorlabs Beamsplitter Reference Architecture Presets
        menu_interf_presets = file_menu.addMenu("🔬 &Thorlabs Beamsplitter Reference Designs")

        act_michelson = QAction("Load &Michelson Interferometer (BS016 50:50, AC254-075-A)...", self)
        act_michelson.setStatusTip("Load Thorlabs Michelson Interferometer with BS016 cube and reference/test mirrors")
        act_michelson.triggered.connect(lambda: self.on_load_beamsplitter_preset("thorlabs_michelson_interferometer.json"))
        menu_interf_presets.addAction(act_michelson)

        act_mach_zehnder = QAction("Load &Mach-Zehnder Interferometer (Dual BS016, PF10-03-P01)...", self)
        act_mach_zehnder.setStatusTip("Load Thorlabs Mach-Zehnder Interferometer with complementary dual-port output")
        act_mach_zehnder.triggered.connect(lambda: self.on_load_beamsplitter_preset("thorlabs_mach_zehnder_interferometer.json"))
        menu_interf_presets.addAction(act_mach_zehnder)

        act_pbs_isolator = QAction("Load &PBS Optical Isolator (>40 dB Rejection, PBS251 + QWP)...", self)
        act_pbs_isolator.setStatusTip("Load Thorlabs Polarizing Beam Splitter optical diode isolator assembly")
        act_pbs_isolator.triggered.connect(lambda: self.on_load_beamsplitter_preset("thorlabs_pbs_optical_isolator.json"))
        menu_interf_presets.addAction(act_pbs_isolator)

        act_beam_sampler = QAction("Load Laser &Beam Sampler (BS019 90:10 Pick-Off)...", self)
        act_beam_sampler.setStatusTip("Load calibrated 90:10 laser power monitoring pick-off assembly")
        act_beam_sampler.triggered.connect(lambda: self.on_load_beamsplitter_preset("thorlabs_beam_sampler.json"))
        menu_interf_presets.addAction(act_beam_sampler)

        act_save_proj = QAction("💾 &Save Project", self)
        if hasattr(QtGui, 'QKeySequence') and hasattr(QtGui.QKeySequence, 'Save'):
            act_save_proj.setShortcut(QtGui.QKeySequence.Save)
        else:
            act_save_proj.setShortcut("Ctrl+S")
        act_save_proj.setStatusTip("Save current optical design (Ctrl+S)")
        act_save_proj.triggered.connect(lambda: self.on_save_project())
        file_menu.addAction(act_save_proj)

        act_save_as_proj = QAction("💾 Save Project &As...", self)
        if hasattr(QtGui, 'QKeySequence') and hasattr(QtGui.QKeySequence, 'SaveAs'):
            act_save_as_proj.setShortcut(QtGui.QKeySequence.SaveAs)
        else:
            act_save_as_proj.setShortcut("Ctrl+Shift+S")
        act_save_as_proj.setStatusTip("Save optical design to a new file (Ctrl+Shift+S)")
        act_save_as_proj.triggered.connect(lambda: self.on_save_as_project())
        file_menu.addAction(act_save_as_proj)

        act_load_sys_cfg = QAction("📂 Load System Explorer Configuration (.json)...", self)
        act_load_sys_cfg.setStatusTip("Load Zemax System Explorer parameters (Aperture, Wavelengths, Fields, Environment)")
        act_load_sys_cfg.triggered.connect(lambda: self.on_load_system_config())
        file_menu.addAction(act_load_sys_cfg)

        act_save_sys_cfg = QAction("💾 Save System Explorer Configuration (.json)...", self)
        act_save_sys_cfg.setStatusTip("Save Zemax System Explorer parameters to a standalone configuration file")
        act_save_sys_cfg.triggered.connect(lambda: self.on_save_system_config())
        file_menu.addAction(act_save_sys_cfg)

        file_menu.addSeparator()

        # Auto-Save Submenu (Zemax / IDE-style background persistence)
        autosave_menu = file_menu.addMenu("⏰ Auto-Save") if hasattr(file_menu, 'addMenu') else None
        target_as_menu = autosave_menu if autosave_menu else file_menu

        self.act_autosave = QAction("⏰ Auto-Save Enabled (Every 5 Min)", self)
        self.act_autosave.setCheckable(True)
        self.act_autosave.setChecked(self.autosave_enabled)
        self.act_autosave.setStatusTip("Toggle automatic project background saving every 5 minutes")
        self.act_autosave.toggled.connect(self._on_toggle_autosave)
        target_as_menu.addAction(self.act_autosave)

        act_as_now = QAction("💾 Save Auto-Save Snapshot Now", self)
        act_as_now.setStatusTip("Immediately create an autosave snapshot of the current design")
        act_as_now.triggered.connect(lambda: self.on_autosave(manual_trigger=True))
        target_as_menu.addAction(act_as_now)

        if autosave_menu and hasattr(autosave_menu, 'addMenu'):
            iv_menu = autosave_menu.addMenu("⏱ Interval Presets")
            for mins in [1, 2, 5, 10, 15, 30]:
                lbl = f"{mins} Minute" + ("s" if mins > 1 else "") + (" (Default)" if mins == 5 else "")
                act_iv = QAction(lbl, self)
                act_iv.triggered.connect(lambda checked=False, m=mins: self.set_autosave_interval(m))
                iv_menu.addAction(act_iv)

        act_rec_as = QAction("🔄 Recover from Auto-Save...", self)
        act_rec_as.setStatusTip("Restore the latest auto-saved project recovery snapshot")
        act_rec_as.triggered.connect(lambda: self.on_recover_autosave())
        target_as_menu.addAction(act_rec_as)

        file_menu.addSeparator()

        act_open_zmx = QAction("Open &Zemax Prescription (.zmx)...", self)
        act_open_zmx.triggered.connect(lambda: self.on_open_zmx())
        file_menu.addAction(act_open_zmx)

        act_insert_zmx = QAction("&Insert Zemax Element (.zmx)...", self)
        act_insert_zmx.triggered.connect(lambda: self.on_insert_zmx())
        file_menu.addAction(act_insert_zmx)

        act_export_step = QAction("Export 3D &STEP Assembly (.step)...", self)
        act_export_step.triggered.connect(lambda: self.on_export_step())
        file_menu.addAction(act_export_step)

        act_export_dxf = QAction("Export 2D &DXF CAD Drawing with Dimensions (.dxf)...", self)
        act_export_dxf.triggered.connect(lambda: self.on_export_dxf_2d())
        file_menu.addAction(act_export_dxf)

        act_export_svg = QAction("Export 2D &SVG Technical Drawing with Dimensions (.svg)...", self)
        act_export_svg.triggered.connect(lambda: self.on_export_svg_2d())
        file_menu.addAction(act_export_svg)

        act_load_cad = QAction("Load &CAD Surface Profile...", self)
        act_load_cad.triggered.connect(lambda: self.on_load_cad_surface())
        file_menu.addAction(act_load_cad)

        file_menu.addSeparator()
        act_exit = QAction("E&xit", self)
        act_exit.triggered.connect(self.close)
        file_menu.addAction(act_exit)

        # --- Edit Menu ---
        edit_menu = menubar.addMenu("&Edit")

        self.act_undo = QAction("&Undo", self)
        if hasattr(QtGui, 'QKeySequence') and hasattr(QtGui.QKeySequence, 'Undo'):
            self.act_undo.setShortcut(QtGui.QKeySequence.Undo)
        else:
            self.act_undo.setShortcut("Ctrl+Z")
        self.act_undo.setStatusTip("Undo the last action (Ctrl+Z)")
        self.act_undo.triggered.connect(self.on_undo)
        edit_menu.addAction(self.act_undo)

        self.act_redo = QAction("&Redo", self)
        if hasattr(QtGui, 'QKeySequence'):
            self.act_redo.setShortcut(QtGui.QKeySequence("Ctrl+Shift+Z"))
        else:
            self.act_redo.setShortcut("Ctrl+Shift+Z")
        self.act_redo.setStatusTip("Redo the last undone action (Ctrl+Shift+Z / Ctrl+Y)")
        self.act_redo.triggered.connect(self.on_redo)
        edit_menu.addAction(self.act_redo)

        # Secondary shortcut for Ctrl+Y redo
        if QT_AVAILABLE and hasattr(QtGui, 'QShortcut'):
            try:
                self.shortcut_redo_alt = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Y"), self)
                self.shortcut_redo_alt.activated.connect(self.on_redo)
            except Exception:
                pass

        edit_menu.addSeparator()

        act_insert = QAction("&Insert Surface Row", self)
        act_insert.triggered.connect(lambda: self.on_insert_surface())
        edit_menu.addAction(act_insert)

        act_delete = QAction("&Delete Surface Row", self)
        act_delete.triggered.connect(lambda: self.on_delete_surface())
        edit_menu.addAction(act_delete)

        act_del_lens = QAction("🗑 &Delete Selected Lens Element...", self)
        act_del_lens.setToolTip("Delete the entire multi-surface lens element assembly and merge air gaps")
        act_del_lens.triggered.connect(lambda: self.on_delete_lens_element())
        edit_menu.addAction(act_del_lens)

        act_ins_cat_main = QAction("➕ Insert &Thorlabs Catalog Lens (Choose Parameters)...", self)
        act_ins_cat_main.setStatusTip("Insert a Thorlabs catalog lens after the selected surface")
        act_ins_cat_main.triggered.connect(lambda: self.on_open_catalog_lens_dialog(mode="insert"))
        edit_menu.addAction(act_ins_cat_main)

        act_achromat_designer = QAction("🎨 &Design Achromatic Doublet Lens...", self)
        act_achromat_designer.setShortcut("Ctrl+Shift+A")
        act_achromat_designer.setStatusTip("Design and optimize custom achromatic doublets (Crown/Flint, Abbe, Spherical Aberration reduction)")
        act_achromat_designer.triggered.connect(lambda: self.on_open_achromat_designer_dialog())
        edit_menu.addAction(act_achromat_designer)

        menu_rep_lens_main = edit_menu.addMenu("🔄 &Replace Selected Lens Element")
        act_rep_tool_main = QAction("🔍 &Thorlabs Catalog Lens Tool (Choose Parameters)...", self)
        act_rep_tool_main.setStatusTip("Open interactive Thorlabs catalog chooser with full parameter control")
        act_rep_tool_main.triggered.connect(lambda: self.on_open_catalog_lens_dialog(mode="replace"))
        menu_rep_lens_main.addAction(act_rep_tool_main)
        menu_rep_lens_main.addSeparator()

        act_rep_zmx_main = QAction("📁 With &Zemax File (.zmx)...", self)
        act_rep_zmx_main.triggered.connect(lambda: self.on_replace_lens_zmx())
        menu_rep_lens_main.addAction(act_rep_zmx_main)

        act_rep_cust_main = QAction("📄 With &Custom File (.json)...", self)
        act_rep_cust_main.triggered.connect(lambda: self.on_replace_lens_custom())
        menu_rep_lens_main.addAction(act_rep_cust_main)

        act_toggle_lock = QAction("🔓 &Toggle Lens Edition (Unlock/Lock)", self)
        act_toggle_lock.setToolTip("Unlock or lock parameters (radius, glass, thickness) of selected lens element")
        act_toggle_lock.triggered.connect(lambda: self.on_toggle_lens_edition())
        edit_menu.addAction(act_toggle_lock)

        act_def_aper_menu = QAction("⭕ &Define Aperture / Stop...", self)
        act_def_aper_menu.setStatusTip("Define clear aperture dimensions, stop status, slit/annular geometry")
        act_def_aper_menu.triggered.connect(lambda: self.on_open_aperture_dialog())
        edit_menu.addAction(act_def_aper_menu)

        act_toggle_stop_menu = QAction("🔴 &Toggle Aperture Stop (STO)", self)
        act_toggle_stop_menu.setStatusTip("Toggle selected surface as the system Aperture Stop")
        act_toggle_stop_menu.triggered.connect(lambda: self.on_toggle_stop())
        edit_menu.addAction(act_toggle_stop_menu)

        act_ins_stop_menu = QAction("➕ Insert &Aperture Stop Surface...", self)
        act_ins_stop_menu.setStatusTip("Insert a dedicated Aperture Stop surface into the system")
        act_ins_stop_menu.triggered.connect(lambda: self.on_insert_aperture_stop())
        edit_menu.addAction(act_ins_stop_menu)

        edit_menu.addSeparator()

        act_load_pupil_relay = QAction("Load Pupil-Imaging Relay", self)
        act_load_pupil_relay.triggered.connect(self.on_load_pupil_relay)
        edit_menu.addAction(act_load_pupil_relay)

        edit_menu.addSeparator()

        act_cluster = QAction("🔗 &Cluster Selected Surfaces as Lens...", self)
        act_cluster.setToolTip("Group selected surfaces into a single multi-surface lens element")
        act_cluster.triggered.connect(self._on_menu_group_selected_surfaces)
        edit_menu.addAction(act_cluster)

        act_uncluster = QAction("🔓 &Uncluster / Clear Group", self)
        act_uncluster.setToolTip("Uncluster / clear group from selected surface")
        act_uncluster.triggered.connect(lambda: self._on_menu_clear_group(self.table_view.currentIndex().row() if hasattr(self.table_view, 'currentIndex') and self.table_view.currentIndex().isValid() else 0))
        edit_menu.addAction(act_uncluster)

        edit_menu.addSeparator()

        menu_solves = edit_menu.addMenu("⚡ &Zemax Solves & Variables")
        act_var = QAction("⚡ Set Selected Parameter as &Variable (V)", self)
        act_var.setShortcut("V")
        act_var.setStatusTip("Mark selected parameter as optimization variable (V)")
        act_var.triggered.connect(lambda: self._set_selected_cells_solve(is_variable=True))
        menu_solves.addAction(act_var)

        act_fix = QAction("🔒 Set Selected Parameter as &Fixed (F)", self)
        act_fix.setShortcut("F")
        act_fix.setStatusTip("Mark selected parameter as fixed (F)")
        act_fix.triggered.connect(lambda: self._set_selected_cells_solve(is_variable=False))
        menu_solves.addAction(act_fix)

        act_solve_dlg = QAction("⚙️ Zemax &Solve & Pickup Dialog...", self)
        act_solve_dlg.setStatusTip("Open Zemax Solve & Pickup Dialog for selected parameter")
        act_solve_dlg.triggered.connect(lambda: self.open_solve_dialog())
        menu_solves.addAction(act_solve_dlg)

        menu_solves.addSeparator()
        menu_presets = menu_solves.addMenu("📐 Variable Presets")
        act_z_def = QAction("Zemax Defaults: Air Gaps Variable, Lenses Fixed", self)
        act_z_def.triggered.connect(self._apply_zemax_defaults)
        menu_presets.addAction(act_z_def)

        act_z_rad = QAction("Make All Radii Variable (Curvature Optimization)", self)
        act_z_rad.triggered.connect(self._apply_all_radii_variable)
        menu_presets.addAction(act_z_rad)

        act_z_air = QAction("Make All Air Gaps Variable (Distance Optimization)", self)
        act_z_air.triggered.connect(self._apply_all_air_gaps_variable)
        menu_presets.addAction(act_z_air)

        act_z_fix = QAction("Clear All Variables (All Fixed)", self)
        act_z_fix.triggered.connect(self._apply_all_fixed)
        menu_presets.addAction(act_z_fix)

        # --- Tools Menu ---
        tools_menu = menubar.addMenu("&Tools")

        act_tools_achromat = QAction("🎨 &Achromatic Doublet Lens Designer...", self)
        act_tools_achromat.setShortcut("Ctrl+Shift+A")
        act_tools_achromat.setStatusTip("Design and optimize custom achromatic doublets (Crown/Flint, Abbe, Spherical Aberration reduction)")
        act_tools_achromat.triggered.connect(lambda: self.on_open_achromat_designer_dialog())
        tools_menu.addAction(act_tools_achromat)

        act_tools_opt = QAction("⚡ &System Optimizer (Curvatures & Merit Function)...", self)
        act_tools_opt.setStatusTip("Run local/global optimization on surface curvatures, thicknesses, and glass parameters")
        act_tools_opt.triggered.connect(self.on_optimize_system)
        tools_menu.addAction(act_tools_opt)

        act_tools_tol = QAction("🎯 &Zemax Tolerance Analysis (RSS, Inverse & Monte Carlo)...", self)
        act_tools_tol.setShortcut("Ctrl+Shift+T")
        act_tools_tol.setStatusTip("Open Zemax Tolerance Analysis dock (Sensitivity, Inverse Sensitivity, Monte Carlo & Yield CDF)")
        act_tools_tol.triggered.connect(self._show_tolerancing_dock)
        tools_menu.addAction(act_tools_tol)

        act_tools_illum = QAction("💡 &Illumination, Mask & Projection Designer...", self)
        act_tools_illum.setShortcut("Ctrl+Shift+I")
        act_tools_illum.setStatusTip("Open Illumination, Mask & Projection System Designer (Kohler/Critical, Etendue, Throughput, Pupil Matching)")
        act_tools_illum.triggered.connect(self._show_illumination_dock)
        tools_menu.addAction(act_tools_illum)

        tools_menu.addSeparator()

        act_tools_solve = QAction("⚙️ &Zemax Solve & Pickup Dialog...", self)
        act_tools_solve.setStatusTip("Open Zemax Solve & Pickup Dialog for selected parameter")
        act_tools_solve.triggered.connect(lambda: self.open_solve_dialog())
        tools_menu.addAction(act_tools_solve)

        tools_menu.addAction(act_ins_cat_main)

        tools_menu.addSeparator()

        self.act_tools_meas = QAction("📏 &Measurement & Inspection Tools...", self)
        self.act_tools_meas.setStatusTip("Open optical measurement dock (distances, EFL, beam sizes)")
        self.act_tools_meas.triggered.connect(lambda: (self.measurement_dock.show(), self.measurement_dock.raise_()) if hasattr(self, 'measurement_dock') else None)
        tools_menu.addAction(self.act_tools_meas)

        act_tools_uf = QAction("🔬 &Ultrafast Optics & SRS Tools...", self)
        act_tools_uf.setStatusTip("Open Ultrafast Optics & Stimulated Raman Scattering (SRS) simulation suite")
        act_tools_uf.triggered.connect(self._show_ultrafast_dock)
        tools_menu.addAction(act_tools_uf)

        act_tools_plm = QAction("🔲 &TI Modulator Studio (PLM / DMD)...", self)
        act_tools_plm.setShortcut("Ctrl+Shift+P")
        act_tools_plm.setStatusTip("Open Texas Instruments Modulator Studio (PLM Phase / DMD Binary) for Zernike wavefront, gratings & mask synthesis")
        act_tools_plm.triggered.connect(self.on_show_plm_studio)
        tools_menu.addAction(act_tools_plm)

        tools_menu.addSeparator()

        self.act_tools_fod = QAction("📑 &Zemax First Order Optical Data (EFFL, BFL, F/#, EPD)...", self)
        self.act_tools_fod.setShortcut("Ctrl+F")
        self.act_tools_fod.setStatusTip("Display Zemax OpticStudio First Order Optical Properties and ABCD Matrix Report")
        self.act_tools_fod.triggered.connect(self.on_show_first_order_data)
        tools_menu.addAction(self.act_tools_fod)

        self.act_tools_spec = QAction("🔬 &Zemax Spectrometer Quality & Resolution Analysis...", self)
        self.act_tools_spec.setShortcut("Ctrl+Shift+Q")
        self.act_tools_spec.setStatusTip("Open Zemax Spectrometer Quality Analysis tab (Resolution, Throughput, Étendue, Smile/Keystone)")
        self.act_tools_spec.triggered.connect(self.on_show_spectrometer_quality)
        tools_menu.addAction(self.act_tools_spec)

        # --- Trace Menu ---
        trace_menu = menubar.addMenu("&Trace")

        act_trace = QAction("&Trace Rays", self)
        act_trace.triggered.connect(self.on_trace_rays)
        trace_menu.addAction(act_trace)

        self.act_mech_toggle = QAction("🔲 Toggle DMD State (ON / OFF)", self)
        self.act_mech_toggle.setStatusTip("Toggle optical mechanism active state")
        self.act_mech_toggle.triggered.connect(self._toggle_mech_state)
        self.act_dmd_toggle = self.act_mech_toggle

        self.act_plm_toggle = QAction("🔲 Step TI-PLM State (ON / OFF / Hadamard / 4-Bit)", self)
        self.act_plm_toggle.setStatusTip("Step TI-PLM modulator across discrete 4-bit phase states, ON/OFF, and Hadamard codes")
        self.act_plm_toggle.triggered.connect(self._toggle_plm_state)

        act_refresh = QAction("&Refresh All Diagnostics", self)
        act_refresh.triggered.connect(self.on_refresh_all)
        trace_menu.addAction(act_refresh)
        
        act_optimize = QAction("O&ptimize System...", self)
        act_optimize.triggered.connect(self.on_optimize_system)
        trace_menu.addAction(act_optimize)

        trace_menu.addSeparator()

        self.act_show_marginal_chief = QAction("👓 &Show Marginal && Chief Rays", self)
        self.act_show_marginal_chief.setCheckable(True)
        self.act_show_marginal_chief.setChecked(False)
        self.act_show_marginal_chief.setShortcut("Ctrl+M")
        self.act_show_marginal_chief.setStatusTip("Highlight Marginal (aperture rim Py=±1) and Chief (pupil center Py=0) rays (Ctrl+M)")
        self.act_show_marginal_chief.toggled.connect(self._on_toggle_marginal_chief)
        trace_menu.addAction(self.act_show_marginal_chief)

        self.act_marginal_chief_only = QAction("🎯 Marginal && Chief Rays &Only", self)
        self.act_marginal_chief_only.setCheckable(True)
        self.act_marginal_chief_only.setChecked(False)
        self.act_marginal_chief_only.setStatusTip("Filter view to display exclusively Marginal and Chief rays")
        self.act_marginal_chief_only.toggled.connect(self._on_toggle_marginal_chief_only)
        trace_menu.addAction(self.act_marginal_chief_only)

        # --- View Menu ---
        view_menu = menubar.addMenu("&View")

        self.act_show_angles = QAction("📐 Show &Incidence & Output Angles (Main λ)", self)
        self.act_show_angles.setCheckable(True)
        self.act_show_angles.setChecked(True)
        self.act_show_angles.setToolTip("Display surface normals, incidence angles (α/θ_in), and output angles (β/θ_out) on reflective mirrors, gratings, and Mechanisms")
        self.act_show_angles.toggled.connect(self._on_toggle_show_angles)
        view_menu.addAction(self.act_show_angles)

        self.act_show_2d_labels = QAction("🏷️ Show 2D &Labels", self)
        self.act_show_2d_labels.setCheckable(True)
        self.act_show_2d_labels.setChecked(True)
        self.act_show_2d_labels.setToolTip("Toggle optical element, stop, and callout labels in 2D projections")
        self.act_show_2d_labels.toggled.connect(self._on_toggle_show_2d_labels)
        view_menu.addAction(self.act_show_2d_labels)

        view_menu.addAction(self.act_show_marginal_chief)

        self.act_dark_mode = QAction("🌙 &Dark Background Design", self)
        self.act_dark_mode.setCheckable(True)
        self.act_dark_mode.setChecked(False)
        self.act_dark_mode.setToolTip("Toggle dark background design for the entire application")
        self.act_dark_mode.toggled.connect(self._on_toggle_dark_mode)
        view_menu.addAction(self.act_dark_mode)

        self.act_3d_no_transp = QAction("🛡️ 3D &No Transparency (Opaque Mode)", self)
        self.act_3d_no_transp.setCheckable(True)
        self.act_3d_no_transp.setChecked(False)
        self.act_3d_no_transp.setToolTip("Render 3D optical elements (lenses, mirrors, gratings, DMD) as solid/opaque (Hotkey: T)")
        self.act_3d_no_transp.toggled.connect(self._on_toggle_3d_no_transparency)
        view_menu.addAction(self.act_3d_no_transp)

        self.act_comp_view = QAction("🔲 &Component View (Grouped Elements)", self)
        self.act_comp_view.setCheckable(True)
        self.act_comp_view.setChecked(False)
        self.act_comp_view.setToolTip("Toggle merged component view in the Lens Data Editor")
        self.act_comp_view.toggled.connect(self._set_lde_component_name_mode)
        view_menu.addAction(self.act_comp_view)

        view_menu.addSeparator()

        # Optical System View Rescaling
        menu_view_scale = view_menu.addMenu("🔍 &Optical System Rescaling")

        self.act_fit_sys = QAction("🔍 &Fit View to System", self)
        self.act_fit_sys.setShortcut("Ctrl+0")
        self.act_fit_sys.setStatusTip("Fit and rescale view to display the entire optical system (Ctrl+0 / F)")
        self.act_fit_sys.triggered.connect(self.on_rescale_optical_view)
        menu_view_scale.addAction(self.act_fit_sys)
        view_menu.addAction(self.act_fit_sys)

        self.act_rescale_dlg = QAction("📐 &Rescale View Limits & Aspect Dialog...", self)
        self.act_rescale_dlg.setShortcut("Ctrl+Shift+R")
        self.act_rescale_dlg.setStatusTip("Open dialog to set explicit view coordinates and transverse scale factor (Ctrl+Shift+R)")
        self.act_rescale_dlg.triggered.connect(self.on_open_rescale_dialog)
        menu_view_scale.addAction(self.act_rescale_dlg)
        view_menu.addAction(self.act_rescale_dlg)

        self.act_reset_scale = QAction("1️⃣ Reset &Transverse Scale (1:1 True Scale)", self)
        self.act_reset_scale.setStatusTip("Reset transverse axis scale back to 1:1 true scale")
        self.act_reset_scale.triggered.connect(self.on_reset_transverse_scale)
        menu_view_scale.addAction(self.act_reset_scale)

        # Internal actions for shortcuts / programmatic zoom without menu clutter
        self.act_zoom_in = QAction("Zoom In", self)
        self.act_zoom_in.triggered.connect(self.on_zoom_in_optical_view)
        self.act_zoom_out = QAction("Zoom Out", self)
        self.act_zoom_out.triggered.connect(self.on_zoom_out_optical_view)

        view_menu.addSeparator()

        # View -> Docks Submenu
        docks_menu = view_menu.addMenu("🪟 &Docks")
        if hasattr(self, 'lde_dock'):
            docks_menu.addAction(self.lde_dock.toggleViewAction())
        if hasattr(self, 'dual_view_dock'):
            docks_menu.addAction(self.dual_view_dock.toggleViewAction())
        if hasattr(self, 'analysis_dock'):
            docks_menu.addAction(self.analysis_dock.toggleViewAction())
        if hasattr(self, 'system_config_dock') and hasattr(self.system_config_dock, 'toggleViewAction'):
            docks_menu.addAction(self.system_config_dock.toggleViewAction())
        if hasattr(self, 'optimizer_dock') and hasattr(self.optimizer_dock, 'toggleViewAction'):
            docks_menu.addAction(self.optimizer_dock.toggleViewAction())
        if hasattr(self, 'tolerancing_dock') and hasattr(self.tolerancing_dock, 'toggleViewAction'):
            docks_menu.addAction(self.tolerancing_dock.toggleViewAction())
        if hasattr(self, 'illumination_dock') and hasattr(self.illumination_dock, 'toggleViewAction'):
            docks_menu.addAction(self.illumination_dock.toggleViewAction())
        if hasattr(self, 'component_editor_dock') and hasattr(self.component_editor_dock, 'toggleViewAction'):
            docks_menu.addAction(self.component_editor_dock.toggleViewAction())
        if hasattr(self, 'history_dock') and hasattr(self.history_dock, 'toggleViewAction'):
            act_hist_view_sub = self.history_dock.toggleViewAction()
            act_hist_view_sub.setText("Design Modification History")
            docks_menu.addAction(act_hist_view_sub)
        if hasattr(self, 'measurement_dock') and hasattr(self.measurement_dock, 'toggleViewAction'):
            act_meas_view_sub = self.measurement_dock.toggleViewAction()
            act_meas_view_sub.setText("📏 Measurement & Inspection Tools")
            docks_menu.addAction(act_meas_view_sub)
        if hasattr(self, 'ai_dock') and hasattr(self.ai_dock, 'toggleViewAction'):
            act_ai_view_sub = self.ai_dock.toggleViewAction()
            act_ai_view_sub.setText("🤖 AI Assistant & Agentic Mode")
            docks_menu.addAction(act_ai_view_sub)
        if hasattr(self, 'ultrafast_dock') and hasattr(self.ultrafast_dock, 'toggleViewAction'):
            act_uf_view_sub = self.ultrafast_dock.toggleViewAction()
            act_uf_view_sub.setText("🔬 Ultrafast Optics & SRS Microscope")
            docks_menu.addAction(act_uf_view_sub)

        # Direct shortcuts in View menu
        if hasattr(self, 'ultrafast_dock') and hasattr(self.ultrafast_dock, 'toggleViewAction'):
            act_uf_view_direct = self.ultrafast_dock.toggleViewAction()
            act_uf_view_direct.setText("🔬 &Ultrafast Optics & SRS Microscope")
            view_menu.addAction(act_uf_view_direct)
        if hasattr(self, 'plm_dock') and hasattr(self.plm_dock, 'toggleViewAction'):
            act_plm_view_direct = self.plm_dock.toggleViewAction()
            act_plm_view_direct.setText("🔲 &TI Modulator Studio (PLM / DMD)")
            view_menu.addAction(act_plm_view_direct)
        if hasattr(self, 'ai_dock') and hasattr(self.ai_dock, 'toggleViewAction'):
            act_ai_view_direct = self.ai_dock.toggleViewAction()
            act_ai_view_direct.setText("🤖 &AI Assistant & Agent")
            view_menu.addAction(act_ai_view_direct)

        # --- Window Menu ---
        window_menu = menubar.addMenu("&Window")
        if hasattr(self, 'lde_dock'):
            window_menu.addAction(self.lde_dock.toggleViewAction())
        if hasattr(self, 'dual_view_dock'):
            window_menu.addAction(self.dual_view_dock.toggleViewAction())
        if hasattr(self, 'analysis_dock'):
            window_menu.addAction(self.analysis_dock.toggleViewAction())
        if hasattr(self, 'system_config_dock') and hasattr(self.system_config_dock, 'toggleViewAction'):
            window_menu.addAction(self.system_config_dock.toggleViewAction())
        if hasattr(self, 'optimizer_dock') and hasattr(self.optimizer_dock, 'toggleViewAction'):
            window_menu.addAction(self.optimizer_dock.toggleViewAction())
        if hasattr(self, 'tolerancing_dock') and hasattr(self.tolerancing_dock, 'toggleViewAction'):
            window_menu.addAction(self.tolerancing_dock.toggleViewAction())
        if hasattr(self, 'illumination_dock') and hasattr(self.illumination_dock, 'toggleViewAction'):
            window_menu.addAction(self.illumination_dock.toggleViewAction())
        if hasattr(self, 'component_editor_dock') and hasattr(self.component_editor_dock, 'toggleViewAction'):
            window_menu.addAction(self.component_editor_dock.toggleViewAction())
        if hasattr(self, 'history_dock') and hasattr(self.history_dock, 'toggleViewAction'):
            act_hist_view = self.history_dock.toggleViewAction()
            act_hist_view.setText("Design Modification History")
            window_menu.addAction(act_hist_view)
        if hasattr(self, 'measurement_dock') and hasattr(self.measurement_dock, 'toggleViewAction'):
            act_meas_view = self.measurement_dock.toggleViewAction()
            act_meas_view.setText("📏 Measurement & Inspection Tools")
            window_menu.addAction(act_meas_view)
        if hasattr(self, 'ai_dock') and hasattr(self.ai_dock, 'toggleViewAction'):
            act_ai_view = self.ai_dock.toggleViewAction()
            act_ai_view.setText("🤖 AI Assistant & Agentic Mode")
            window_menu.addAction(act_ai_view)
        if hasattr(self, 'ultrafast_dock') and hasattr(self.ultrafast_dock, 'toggleViewAction'):
            act_uf_win = self.ultrafast_dock.toggleViewAction()
            act_uf_win.setText("🔬 Ultrafast Optics & SRS Microscope")
            window_menu.addAction(act_uf_win)
        if hasattr(self, 'plm_dock') and hasattr(self.plm_dock, 'toggleViewAction'):
            act_plm_win = self.plm_dock.toggleViewAction()
            act_plm_win.setText("🔲 TI Modulator Studio (PLM / DMD)")
            window_menu.addAction(act_plm_win)
        window_menu.addSeparator()

        # Workspace Presets Submenu
        menu_presets = window_menu.addMenu("📐 &Workspace Presets")
        self.act_preset_design = QAction("🎯 &Sequential Design", self)
        self.act_preset_design.setStatusTip("Focus on optical layout design: LDE, Cross-Section, and Component Editor")
        self.act_preset_design.triggered.connect(lambda: self.apply_workspace_preset("design"))
        menu_presets.addAction(self.act_preset_design)

        self.act_preset_analysis = QAction("📊 Optical &Analysis", self)
        self.act_preset_analysis.setStatusTip("Focus on optical analysis: LDE, Viewer, Analysis Diagnostics, and Measurements")
        self.act_preset_analysis.triggered.connect(lambda: self.apply_workspace_preset("analysis"))
        menu_presets.addAction(self.act_preset_analysis)

        self.act_preset_optimize = QAction("⚡ &Optimization & Tolerancing", self)
        self.act_preset_optimize.setStatusTip("Focus on optimization: LDE, Viewer, Optimizer, Tolerancing, and History")
        self.act_preset_optimize.triggered.connect(lambda: self.apply_workspace_preset("optimization"))
        menu_presets.addAction(self.act_preset_optimize)

        self.act_preset_ai = QAction("🤖 &AI Assistant", self)
        self.act_preset_ai.setStatusTip("Focus on agentic co-pilot: LDE, Viewer, AI Agent, and System Explorer")
        self.act_preset_ai.triggered.connect(lambda: self.apply_workspace_preset("ai"))
        menu_presets.addAction(self.act_preset_ai)

        self.act_preset_ultrafast = QAction("🔬 &Ultrafast & SRS Studio", self)
        self.act_preset_ultrafast.setStatusTip("Focus on ultrafast simulation: LDE, Viewer, Ultrafast Analysis, and PLM Studio")
        self.act_preset_ultrafast.triggered.connect(lambda: self.apply_workspace_preset("ultrafast"))
        menu_presets.addAction(self.act_preset_ultrafast)

        menu_presets.addSeparator()
        act_minimal_workspace = QAction("Reset Minimal Workspace", self)
        act_minimal_workspace.setStatusTip("Show the design table, component editor, and optical viewer only")
        act_minimal_workspace.triggered.connect(self._apply_minimal_workspace)
        menu_presets.addAction(act_minimal_workspace)
        window_menu.addAction(act_minimal_workspace)

        # --- View -> Toolbars Submenu ---
        menu_tb = view_menu.addMenu("🔧 &Toolbars")

        # --- Categorized Action Definitions ---
        self.act_tb_achromat = QAction("🎨 Achromat Designer", self)
        self.act_tb_achromat.setToolTip("Open Achromatic Doublet Lens Designer (Ctrl+Shift+A)")
        self.act_tb_achromat.triggered.connect(lambda: self.on_open_achromat_designer_dialog())

        self.act_tb_optimize = QAction("⚡ Optimize", self)
        self.act_tb_optimize.setToolTip("Open System Curvature & Merit Function Optimizer")
        self.act_tb_optimize.triggered.connect(self.on_optimize_system)

        self.act_tb_tolerancing = QAction("🎯 Tolerancing", self)
        self.act_tb_tolerancing.setToolTip("Open Zemax Tolerance Analysis Dock (Sensitivity, Inverse Sensitivity, Monte Carlo & Yield CDF)")
        self.act_tb_tolerancing.triggered.connect(self._show_tolerancing_dock)

        self.act_tb_plm = QAction("🔲 TI Modulator Studio", self)
        self.act_tb_plm.setToolTip("Open TI Modulator Studio (PLM / DMD) (Ctrl+Shift+P)")
        self.act_tb_plm.triggered.connect(self.on_show_plm_studio)

        # --- Primary Main Toolbar (Clean & Minimalist: Core Workflow Actions) ---
        toolbar = self.addToolBar("Main Toolbar")
        toolbar.setObjectName("MainToolbar")
        self.main_toolbar = toolbar
        toolbar.setVisible(True)
        toolbar.setMovable(False)
        toolbar.addAction(act_new)
        toolbar.addAction(act_load_proj)
        toolbar.addAction(act_save_proj)
        toolbar.addSeparator()
        toolbar.addAction(self.act_undo)
        toolbar.addAction(self.act_redo)
        toolbar.addSeparator()
        toolbar.addAction(act_trace)
        self.act_open_ai_agent = QAction("🤖 AI Agent", self)
        self.act_open_ai_agent.setToolTip("Open the AI Assistant and choose Agentic multi-step mode")
        self.act_open_ai_agent.triggered.connect(self._show_ai_agent)
        toolbar.addAction(self.act_open_ai_agent)
        menu_tb.addAction(toolbar.toggleViewAction())

        self.act_open_ultrafast = QAction("🔬 Ultrafast & SRS", self)
        self.act_open_ultrafast.setToolTip("Open Ultrafast Optics & SRS Microscope Simulation Dock")
        self.act_open_ultrafast.triggered.connect(self._show_ultrafast_dock)

        # Categorized Optimization Toolbar
        self.opt_toolbar = self.addToolBar("Optimization & Tolerancing Toolbar")
        self.opt_toolbar.setObjectName("OptimizationToolbar")
        self.opt_toolbar.setVisible(False)
        self.opt_toolbar.addAction(self.act_tb_optimize)
        self.opt_toolbar.addAction(self.act_tb_tolerancing)
        self.opt_toolbar.addAction(self.act_tb_achromat)
        self.opt_toolbar.addAction(self.act_tools_fod)
        menu_tb.addAction(self.opt_toolbar.toggleViewAction())

        # Categorized Advanced Physics Toolbar
        self.advanced_toolbar = self.addToolBar("Advanced Physics Toolbar")
        self.advanced_toolbar.setObjectName("AdvancedPhysicsToolbar")
        self.advanced_toolbar.setVisible(False)
        self.advanced_toolbar.addAction(self.act_tb_plm)
        self.advanced_toolbar.addAction(self.act_open_ultrafast)
        self.advanced_toolbar.addAction(self.act_tools_meas)
        menu_tb.addAction(self.advanced_toolbar.toggleViewAction())

        # Workspace Presets Toolbar
        self.presets_toolbar = self.addToolBar("Workspace Presets Toolbar")
        self.presets_toolbar.setObjectName("WorkspacePresetsToolbar")
        self.presets_toolbar.setVisible(False)
        self.presets_toolbar.addAction(self.act_preset_design)
        self.presets_toolbar.addAction(self.act_preset_analysis)
        self.presets_toolbar.addAction(self.act_preset_optimize)
        self.presets_toolbar.addAction(self.act_preset_ai)
        self.presets_toolbar.addAction(self.act_preset_ultrafast)
        menu_tb.addAction(self.presets_toolbar.toggleViewAction())
        self._sync_mech_action_ui()

    def apply_workspace_preset(self, preset: str) -> None:
        """Switch dock arrangement to a targeted functional workspace."""
        preset = preset.lower()
        if preset in ("design", "sequential"):
            self._apply_minimal_workspace()
            if hasattr(self, 'statusBar') and self.statusBar() is not None:
                self.statusBar().showMessage("Workspace: Sequential Design (LDE, Cross-Section, Component Editor)", 3000)
        elif preset == "analysis":
            for d in (self.lde_dock, self.dual_view_dock, self.analysis_dock, self.measurement_dock):
                if d is not None:
                    d.show()
            for d in (self.component_editor_dock, self.system_config_dock, self.optimizer_dock, self.tolerancing_dock, self.history_dock, self.ai_dock, self.ultrafast_dock, getattr(self, 'plm_dock', None)):
                if d is not None:
                    d.hide()
            if self.analysis_dock is not None:
                self.analysis_dock.raise_()
            if hasattr(self, 'statusBar') and self.statusBar() is not None:
                self.statusBar().showMessage("Workspace: Optical Analysis", 3000)
        elif preset in ("optimize", "optimization"):
            for d in (self.lde_dock, self.dual_view_dock, self.optimizer_dock, self.tolerancing_dock, self.history_dock):
                if d is not None:
                    d.show()
            for d in (self.component_editor_dock, self.system_config_dock, self.analysis_dock, self.measurement_dock, self.ai_dock, self.ultrafast_dock, getattr(self, 'plm_dock', None)):
                if d is not None:
                    d.hide()
            if self.optimizer_dock is not None:
                self.optimizer_dock.raise_()
            if hasattr(self, 'statusBar') and self.statusBar() is not None:
                self.statusBar().showMessage("Workspace: Optimization & Tolerancing", 3000)
        elif preset == "ai":
            for d in (self.lde_dock, self.dual_view_dock, self.ai_dock, self.system_config_dock):
                if d is not None:
                    d.show()
            for d in (self.component_editor_dock, self.optimizer_dock, self.tolerancing_dock, self.history_dock, self.analysis_dock, self.measurement_dock, self.ultrafast_dock, getattr(self, 'plm_dock', None)):
                if d is not None:
                    d.hide()
            if self.ai_dock is not None:
                self.ai_dock.raise_()
            if hasattr(self, 'statusBar') and self.statusBar() is not None:
                self.statusBar().showMessage("Workspace: AI Assistant & Co-Pilot", 3000)
        elif preset in ("ultrafast", "srs"):
            for d in (self.lde_dock, self.dual_view_dock, self.ultrafast_dock, getattr(self, 'plm_dock', None), self.measurement_dock):
                if d is not None:
                    d.show()
            for d in (self.component_editor_dock, self.system_config_dock, self.optimizer_dock, self.tolerancing_dock, self.history_dock, self.analysis_dock, self.ai_dock):
                if d is not None:
                    d.hide()
            if self.ultrafast_dock is not None:
                self.ultrafast_dock.raise_()
            if hasattr(self, 'statusBar') and self.statusBar() is not None:
                self.statusBar().showMessage("Workspace: Ultrafast Optics & SRS Studio", 3000)

    def _show_tolerancing_dock(self) -> None:
        """Show, raise, and focus the Zemax Tolerance Analysis dock."""
        if getattr(self, "tolerancing_dock", None) is None:
            return
        self.tolerancing_dock.show()
        self.tolerancing_dock.raise_()
        self.tolerancing_dock.activateWindow()

    def _show_illumination_dock(self) -> None:
        """Show, raise, and focus the Illumination, Mask & Projection System dock."""
        if getattr(self, "illumination_dock", None) is None:
            return
        self.illumination_dock.show()
        self.illumination_dock.raise_()
        self.illumination_dock.activateWindow()

    def _show_ultrafast_dock(self) -> None:
        """Show, raise, and focus the Ultrafast Optics & SRS Microscope dock."""
        if getattr(self, "ultrafast_dock", None) is None:
            return
        self.ultrafast_dock.show()
        self.ultrafast_dock.raise_()
        self.ultrafast_dock.activateWindow()
        if hasattr(self.ultrafast_dock, "tabs") and self.ultrafast_dock.tabs.count() > 5:
            self.ultrafast_dock.tabs.setCurrentIndex(5)

    def _show_ai_agent(self) -> None:
        """Show and focus the AI Assistant regardless of the current dock layout."""
        if getattr(self, "ai_dock", None) is None:
            return
        self.ai_dock.show()
        self.ai_dock.raise_()
        if hasattr(self.ai_dock, "tabs"):
            self.ai_dock.tabs.setCurrentIndex(0)
        if hasattr(self.ai_dock, "input_box"):
            self.ai_dock.input_box.setFocus()

    def _on_toggle_show_angles(self, checked: bool):
        """Synchronizes angle overlay toggle across DualView and MeasurementDock."""
        if hasattr(self, 'dual_view') and self.dual_view is not None:
            self.dual_view.set_show_incidence_angles(checked)
        if hasattr(self, 'measurement_dock') and self.measurement_dock is not None:
            if hasattr(self.measurement_dock, 'chk_show_angles'):
                self.measurement_dock.chk_show_angles.blockSignals(True)
                self.measurement_dock.chk_show_angles.setChecked(checked)
                self.measurement_dock.chk_show_angles.blockSignals(False)

    def _on_angles_visibility_changed(self, enabled: bool):
        """Syncs action state when angle visibility changes in DualView."""
        if hasattr(self, 'act_show_angles') and self.act_show_angles is not None:
            self.act_show_angles.blockSignals(True)
            self.act_show_angles.setChecked(enabled)
            self.act_show_angles.blockSignals(False)

    def _on_toggle_show_2d_labels(self, checked: bool):
        """Synchronizes 2D label visibility toggle with DualView."""
        if hasattr(self, 'dual_view') and self.dual_view is not None:
            if hasattr(self.dual_view, 'set_show_labels'):
                self.dual_view.set_show_labels(checked)

    def _on_2d_labels_visibility_changed(self, enabled: bool):
        """Syncs action state when 2D label visibility changes in DualView."""
        if hasattr(self, 'act_show_2d_labels') and self.act_show_2d_labels is not None:
            self.act_show_2d_labels.blockSignals(True)
            self.act_show_2d_labels.setChecked(enabled)
            self.act_show_2d_labels.blockSignals(False)

    def set_show_2d_labels(self, enabled: bool) -> None:
        """Set 2D projection label visibility."""
        if hasattr(self, 'dual_view') and self.dual_view is not None:
            if hasattr(self.dual_view, 'set_show_labels'):
                self.dual_view.set_show_labels(enabled)

    def _on_toggle_marginal_chief(self, checked: bool) -> None:
        """Synchronizes Marginal & Chief ray visualization across 2D & 3D viewers."""
        if hasattr(self, 'dual_view') and self.dual_view is not None:
            self.dual_view.set_show_marginal_chief(checked)
        if hasattr(self, 'visualizer_3d') and self.visualizer_3d is not None:
            self.visualizer_3d.set_show_marginal_chief(checked)
        if hasattr(self, 'statusBar') and self.statusBar() is not None:
            state = "Enabled" if checked else "Disabled"
            self.statusBar().showMessage(f"Marginal & Chief Rays Visualization {state}", 3000)

    def _on_toggle_marginal_chief_only(self, checked: bool) -> None:
        """Filter optical projections to display exclusively marginal and chief rays."""
        if hasattr(self, 'dual_view') and self.dual_view is not None:
            self.dual_view.set_marginal_chief_only(checked)
        if hasattr(self, 'visualizer_3d') and self.visualizer_3d is not None:
            self.visualizer_3d.set_marginal_chief_only(checked)
        if hasattr(self, 'statusBar') and self.statusBar() is not None:
            state = "Filtered (Marginal & Chief Only)" if checked else "All Rays"
            self.statusBar().showMessage(f"Ray Display: {state}", 3000)

    def _on_marginal_chief_visibility_changed(self, enabled: bool) -> None:
        """Syncs action state when marginal/chief visibility changes in DualView."""
        if hasattr(self, 'act_show_marginal_chief') and self.act_show_marginal_chief is not None:
            self.act_show_marginal_chief.blockSignals(True)
            self.act_show_marginal_chief.setChecked(enabled)
            self.act_show_marginal_chief.blockSignals(False)

    def set_show_marginal_chief(self, enabled: bool) -> None:
        """Programmatic toggle for Marginal & Chief ray visualization."""
        if hasattr(self, 'act_show_marginal_chief') and self.act_show_marginal_chief is not None:
            self.act_show_marginal_chief.setChecked(enabled)
        else:
            self._on_toggle_marginal_chief(enabled)

    def apply_dark_theme(self, enabled: bool) -> None:
        """Applies or removes application-wide dark stylesheet and palette."""
        if not QT_AVAILABLE:
            return
        from optics_app.gui.qt_compat import QApplication
        try:
            from v2.gui.theme import DARK_THEME_STYLESHEET, create_dark_palette
        except ImportError:
            from optics_app.gui.theme import DARK_THEME_STYLESHEET, create_dark_palette

        app = QApplication.instance()
        if enabled:
            self.setStyleSheet(DARK_THEME_STYLESHEET)
            if app is not None:
                try:
                    app.setStyleSheet(DARK_THEME_STYLESHEET)
                except Exception:
                    pass
            dark_pal = create_dark_palette()
            if dark_pal is not None:
                try:
                    self.setPalette(dark_pal)
                    if app is not None:
                        app.setPalette(dark_pal)
                except Exception:
                    pass
        else:
            self.setStyleSheet("")
            if app is not None:
                try:
                    app.setStyleSheet("")
                except Exception:
                    pass
            if hasattr(self, '_default_palette') and self._default_palette is not None:
                try:
                    self.setPalette(self._default_palette)
                    if app is not None:
                        app.setPalette(self._default_palette)
                except Exception:
                    pass

    def _on_toggle_dark_mode(self, checked: bool):
        """Synchronizes dark background design across the entire application."""
        self.set_dark_mode(checked)

    def _on_dark_mode_changed(self, enabled: bool):
        """Syncs menu action, views, analysis, and tables when dark mode changes."""
        self.set_dark_mode(enabled)

    def set_dark_mode(self, enabled: bool) -> None:
        """Set dark background design across the whole application."""
        self.dark_mode = bool(enabled)

        # 1. Sync Menu Action without duplicate signal
        if hasattr(self, 'act_dark_mode') and self.act_dark_mode is not None:
            self.act_dark_mode.blockSignals(True)
            self.act_dark_mode.setChecked(self.dark_mode)
            self.act_dark_mode.blockSignals(False)

        # 2. Apply Application-Wide Theme Stylesheet & Palette
        self.apply_dark_theme(self.dark_mode)

        # 3. Synchronize 2D Projections View
        if hasattr(self, 'dual_view') and self.dual_view is not None:
            if hasattr(self.dual_view, 'set_dark_mode') and getattr(self.dual_view, 'dark_mode', None) != self.dark_mode:
                self.dual_view.set_dark_mode(self.dark_mode)

        # 4. Synchronize 3D Visualizer View
        if hasattr(self, 'visualizer_3d') and self.visualizer_3d is not None:
            if hasattr(self.visualizer_3d, 'set_dark_mode') and getattr(self.visualizer_3d, 'dark_mode', None) != self.dark_mode:
                self.visualizer_3d.set_dark_mode(self.dark_mode)

        # 5. Synchronize Analysis Tabs Suite
        if hasattr(self, 'analysis_tabs') and self.analysis_tabs is not None:
            if hasattr(self.analysis_tabs, 'set_dark_mode'):
                self.analysis_tabs.set_dark_mode(self.dark_mode)

        # 6. Synchronize Lens Data Editor Table & Model
        if hasattr(self, 'table_view') and self.table_view is not None:
            if hasattr(self.table_view, 'set_dark_mode'):
                self.table_view.set_dark_mode(self.dark_mode)

        if hasattr(self, 'model') and self.model is not None:
            if hasattr(self.model, 'set_dark_mode'):
                self.model.set_dark_mode(self.dark_mode)

        # 7. Update Status Bar first-order label style
        if hasattr(self, 'lbl_status_first_order') and self.lbl_status_first_order is not None:
            if self.dark_mode:
                self.lbl_status_first_order.setStyleSheet("font-weight: 500; color: #38bdf8; padding-right: 12px;")
            else:
                self.lbl_status_first_order.setStyleSheet("font-weight: 500; color: #1a73e8; padding-right: 12px;")

        # 8. Synchronize Component Editor Dock
        if hasattr(self, 'component_editor_dock') and self.component_editor_dock is not None:
            if hasattr(self.component_editor_dock, 'set_dark_mode'):
                self.component_editor_dock.set_dark_mode(self.dark_mode)

        # 9. Synchronize Optimizer Dock
        if hasattr(self, 'optimizer_dock') and self.optimizer_dock is not None:
            if hasattr(self.optimizer_dock, 'set_dark_mode'):
                self.optimizer_dock.set_dark_mode(self.dark_mode)

        # 10. Synchronize Tolerancing Dock
        if hasattr(self, 'tolerancing_dock') and self.tolerancing_dock is not None:
            if hasattr(self.tolerancing_dock, 'set_dark_mode'):
                self.tolerancing_dock.set_dark_mode(self.dark_mode)

        # 10b. Synchronize Illumination Dock
        if hasattr(self, 'illumination_dock') and self.illumination_dock is not None:
            if hasattr(self.illumination_dock, 'set_dark_mode'):
                self.illumination_dock.set_dark_mode(self.dark_mode)

        # 11. Synchronize Ultrafast Analysis Dock
        if hasattr(self, 'ultrafast_dock') and self.ultrafast_dock is not None:
            if hasattr(self.ultrafast_dock, 'set_dark_mode'):
                self.ultrafast_dock.set_dark_mode(self.dark_mode)

        # 12. Synchronize TI-PLM Modulator Studio Dock
        if hasattr(self, 'plm_dock') and self.plm_dock is not None:
            if hasattr(self.plm_dock, 'set_dark_mode'):
                self.plm_dock.set_dark_mode(self.dark_mode)

        # 13. Synchronize Measurement & Inspection Dock
        if hasattr(self, 'measurement_dock') and self.measurement_dock is not None:
            if hasattr(self.measurement_dock, 'set_dark_mode'):
                self.measurement_dock.set_dark_mode(self.dark_mode)

        # 14. Synchronize System Configuration Explorer Dock
        if hasattr(self, 'system_config_dock') and self.system_config_dock is not None:
            if hasattr(self.system_config_dock, 'set_dark_mode'):
                self.system_config_dock.set_dark_mode(self.dark_mode)

        # 15. Synchronize Design Modification History Dock
        if hasattr(self, 'history_dock') and self.history_dock is not None:
            if hasattr(self.history_dock, 'set_dark_mode'):
                self.history_dock.set_dark_mode(self.dark_mode)

        # 16. Synchronize AI Chat & Copilot Dock
        if hasattr(self, 'ai_dock') and self.ai_dock is not None:
            if hasattr(self.ai_dock, 'set_dark_mode'):
                self.ai_dock.set_dark_mode(self.dark_mode)

    def _on_3d_surface_selected(self, idx: int):
        if hasattr(self, 'table_view'):
            self.table_view.selectRow(idx)
        if hasattr(self, 'comp_editor'):
            self.comp_editor.set_selected_surface(idx)
            
    def _on_toggle_3d_no_transparency(self, checked: bool):
        """Synchronizes 3D opaque rendering across Visualizer3DWidget."""
        if hasattr(self, 'visualizer_3d') and self.visualizer_3d is not None:
            self.visualizer_3d.set_no_transparency(checked)

    def on_rescale_optical_view(self) -> None:
        """Fit and rescale the optical viewer to display the entire optical system."""
        if hasattr(self, 'dual_view') and self.dual_view is not None and hasattr(self.dual_view, 'reset_view_limits'):
            self.dual_view.reset_view_limits()
        if hasattr(self, 'visualizer_3d') and self.visualizer_3d is not None and hasattr(self.visualizer_3d, 'fit_view'):
            self.visualizer_3d.fit_view()

    def on_zoom_in_optical_view(self) -> None:
        """Zoom in the optical system view."""
        if hasattr(self, 'dual_view') and self.dual_view is not None and hasattr(self.dual_view, 'zoom_in'):
            self.dual_view.zoom_in()
        if hasattr(self, 'visualizer_3d') and self.visualizer_3d is not None and hasattr(self.visualizer_3d, 'zoom_view'):
            self.visualizer_3d.zoom_view(0.85)

    def on_zoom_out_optical_view(self) -> None:
        """Zoom out the optical system view."""
        if hasattr(self, 'dual_view') and self.dual_view is not None and hasattr(self.dual_view, 'zoom_out'):
            self.dual_view.zoom_out()
        if hasattr(self, 'visualizer_3d') and self.visualizer_3d is not None and hasattr(self.visualizer_3d, 'zoom_view'):
            self.visualizer_3d.zoom_view(1.15)

    def on_open_rescale_dialog(self) -> None:
        """Open the optical system view rescaling dialog."""
        if hasattr(self, 'dual_view') and self.dual_view is not None and hasattr(self.dual_view, 'open_rescale_dialog'):
            self.dual_view.open_rescale_dialog()

    def on_reset_transverse_scale(self) -> None:
        """Reset optical view transverse aspect ratio to 1:1 true scale."""
        if hasattr(self, 'dual_view') and self.dual_view is not None and hasattr(self.dual_view, 'set_transverse_scale'):
            self.dual_view.set_transverse_scale(1.0)
            if hasattr(self.dual_view, 'spin_transverse_scale') and self.dual_view.spin_transverse_scale:
                self.dual_view.spin_transverse_scale.blockSignals(True)
                self.dual_view.spin_transverse_scale.setValue(1.0)
                self.dual_view.spin_transverse_scale.blockSignals(False)

    def resizeEvent(self, event) -> None:
        """Handle main window resize: ensure optical viewer and docks adapt smoothly."""
        super_re = getattr(super(), 'resizeEvent', None)
        if callable(super_re):
            super_re(event)
        if hasattr(self, 'dual_view') and self.dual_view is not None:
            if hasattr(self.dual_view, '_on_widget_resized'):
                self.dual_view._on_widget_resized()

    def _on_3d_transparency_changed(self, no_transp: bool):
        """Syncs action state when transparency is toggled inside 3D viewer."""
        if hasattr(self, 'act_3d_no_transp') and self.act_3d_no_transp is not None:
            self.act_3d_no_transp.blockSignals(True)
            self.act_3d_no_transp.setChecked(no_transp)
            self.act_3d_no_transp.blockSignals(False)

    def _toggle_mech_state(self) -> Optional[str]:
        """Toggle DMD mechanical state between ON and OFF, re-tracing rays without moving optical elements."""
        if hasattr(self, "dual_view") and hasattr(self.dual_view, "toggle_mech_state"):
            new_state = self.dual_view.toggle_mech_state()
            self._sync_mech_action_ui(new_state)
            if new_state:
                status = f"🔲 Mechanism switched to {new_state}: {'Illuminating detector arm' if new_state == 'ON' else 'Beam deflected 48° into beam dump'}"
                if QT_AVAILABLE and hasattr(self, "statusBar") and self.statusBar():
                    self.statusBar().showMessage(status, 4000)
                self._status_msg = status
            return new_state
        return None

    _toggle_dmd_state = _toggle_mech_state

    def _sync_mech_action_ui(self, state: Optional[str] = None) -> None:
        """Synchronize the main toolbar DMD and PLM action labels and visibility with the model."""
        if hasattr(self, "act_mech_toggle") and self.act_mech_toggle is not None:
            if state is None:
                if hasattr(self, "dual_view") and hasattr(self.dual_view, "get_mech_surface_info"):
                    info = self.dual_view.get_mech_surface_info()
                    if info is not None:
                        state = info[2]
                elif self.model and hasattr(self.model, "get_surface_stack"):
                    stack = self.model.get_surface_stack()
                    if stack:
                        from optics_app.engine.mechanisms import is_dmd, active_state_name
                        for s in stack.surfaces:
                            if is_dmd(s):
                                state = active_state_name(s)
                                break
            if state is None:
                self.act_mech_toggle.setVisible(False)
                self.act_mech_toggle.setText("🔲 DMD: N/A")
            else:
                self.act_mech_toggle.setVisible(True)
                if state == "ON":
                    self.act_mech_toggle.setText("🔲 Toggle DMD State (ON -> OFF)")
                    self.act_mech_toggle.setToolTip("DMD micromirrors are currently ON (illuminating detector arm). Click to switch to OFF (beam dump).")
                else:
                    self.act_mech_toggle.setText("🔲 Toggle DMD State (OFF -> ON)")
                    self.act_mech_toggle.setToolTip("DMD micromirrors are currently OFF (deflected 48° into beam dump). Click to switch to ON.")

        if hasattr(self, "act_plm_toggle") and self.act_plm_toggle is not None:
            plm_state = None
            if self.model and hasattr(self.model, "get_surface_stack"):
                stack = self.model.get_surface_stack()
                if stack:
                    for s in stack.surfaces:
                        if getattr(s, 'is_plm', False) or getattr(s, 'mechanism_type', '') == 'PLM':
                            plm_state = getattr(s, 'mechanical_state', 'ON')
                            break
            if plm_state is None:
                self.act_plm_toggle.setVisible(False)
            else:
                self.act_plm_toggle.setVisible(True)
                self.act_plm_toggle.setText(f"🔲 Step TI-PLM State ({plm_state})")

    # ----------------------------------------------------
    # Undo & Redo Handlers (Ctrl+Z / Ctrl+Shift+Z / Ctrl+Y)
    # ----------------------------------------------------

    def on_undo(self, checked: bool = False) -> bool:
        """Revert optical system to previous undo snapshot (Ctrl+Z)."""
        if not hasattr(self, 'undo_manager') or not self.undo_manager.can_undo():
            msg = "Undo: Nothing to undo"
            if QT_AVAILABLE and hasattr(self, 'statusBar') and self.statusBar() is not None:
                self.statusBar().showMessage(msg)
            self._status_msg = msg
            return False

        current_stack = self.model.get_surface_stack()
        restored = self.undo_manager.undo(current_stack, getattr(self, 'system_config', None))
        if restored is not None:
            self._apply_restored_state(restored)
            msg = f"Undo: {restored.description}" if restored.description else "Undo successful"
            if QT_AVAILABLE and hasattr(self, 'statusBar') and self.statusBar() is not None:
                self.statusBar().showMessage(msg)
            self._status_msg = msg
            self._update_undo_redo_actions()
            return True
        return False

    def on_redo(self, checked: bool = False) -> bool:
        """Re-apply next optical system state from redo stack (Ctrl+Shift+Z / Ctrl+Y)."""
        if not hasattr(self, 'undo_manager') or not self.undo_manager.can_redo():
            msg = "Redo: Nothing to redo"
            if QT_AVAILABLE and hasattr(self, 'statusBar') and self.statusBar() is not None:
                self.statusBar().showMessage(msg)
            self._status_msg = msg
            return False

        current_stack = self.model.get_surface_stack()
        restored = self.undo_manager.redo(current_stack, getattr(self, 'system_config', None))
        if restored is not None:
            self._apply_restored_state(restored)
            msg = f"Redo: {restored.description}" if restored.description else "Redo successful"
            if QT_AVAILABLE and hasattr(self, 'statusBar') and self.statusBar() is not None:
                self.statusBar().showMessage(msg)
            self._status_msg = msg
            self._update_undo_redo_actions()
            return True
        return False

    def _apply_restored_state(self, state: OpticalSystemState) -> None:
        """Safely restore a snapshot state into model and system configuration."""
        if hasattr(self, 'undo_manager'):
            self.undo_manager._is_undoing_or_redoing = True
        try:
            import copy
            new_stack = SurfaceStack()
            for s in state.stack_surfaces:
                new_stack.add_surface(copy.deepcopy(s))

            auxiliary = copy.deepcopy(getattr(state, "stack_auxiliary_dict", {}) or {})
            for attr, value in auxiliary.items():
                if attr == "optical_routes":
                    from optics_app.engine.routes import OpticalRouteSet
                    value = OpticalRouteSet.from_dict(value)
                elif attr == "two_photon_config":
                    from optics_app.analysis.two_photon import TwoPhotonMicroscopeConfig
                    value = TwoPhotonMicroscopeConfig.from_dict(value)
                elif attr == "nonlinear_imaging_config":
                    from optics_app.analysis.nonlinear_imaging import NonlinearImagingConfig
                    value = NonlinearImagingConfig.from_dict(value)
                setattr(new_stack, attr, value)

            self.model.set_surface_stack(new_stack)

            if hasattr(self, "analysis_tabs") and self.analysis_tabs is not None:
                analysis_payload = dict(getattr(new_stack, "analysis_config", {}) or {})
                if hasattr(new_stack, "two_photon_config"):
                    analysis_payload["two_photon_microscope"] = new_stack.two_photon_config.to_dict()
                if hasattr(new_stack, "nonlinear_imaging_config"):
                    analysis_payload["nonlinear_imaging"] = new_stack.nonlinear_imaging_config.to_dict()
                if hasattr(new_stack, "optical_routes"):
                    analysis_payload["optical_routes"] = new_stack.optical_routes.to_dict()
                if hasattr(self.analysis_tabs, "set_analysis_config"):
                    self.analysis_tabs.set_analysis_config(analysis_payload)

            if state.system_config_dict is not None and hasattr(self, 'system_config'):
                self.system_config.from_dict(state.system_config_dict)
                if hasattr(self, 'source_config') and self.source_config is not None:
                    src_dict = auxiliary.get('source_config')
                    if src_dict and hasattr(self.source_config, 'from_dict'):
                        self.source_config.from_dict(src_dict)
                    else:
                        self.source_config.wavelengths = ", ".join(f"{w:.4g}" for w in self.system_config.wavelength_list_um)
                    if hasattr(self, 'source_editor') and self.source_editor is not None:
                        self.source_editor.update_ui()
                if hasattr(self, 'system_config_dock') and self.system_config_dock is not None:
                    if hasattr(self.system_config_dock, 'update_ui'):
                        self.system_config_dock.update_ui()

            if hasattr(self, 'visualizer_3d') and self.visualizer_3d is not None and hasattr(new_stack, 'viewer_settings') and new_stack.viewer_settings:
                if hasattr(self.visualizer_3d, 'set_view_settings'):
                    self.visualizer_3d.set_view_settings(new_stack.viewer_settings)

            if hasattr(self.model, '_reindex_surface_names'):
                self.model._reindex_surface_names()
            if QT_AVAILABLE and hasattr(self.model, 'layoutChanged'):
                self.model.layoutChanged.emit()

            self.on_trace_rays()
            if hasattr(self, 'plm_dock') and self.plm_dock is not None:
                self.plm_dock.set_surface_stack(new_stack)
        finally:
            if hasattr(self, 'undo_manager'):
                self.undo_manager._is_undoing_or_redoing = False

    def _update_undo_redo_actions(self) -> None:
        """Update enabled states of Undo / Redo actions."""
        if hasattr(self, 'act_undo') and self.act_undo is not None:
            can_u = self.undo_manager.can_undo() if hasattr(self, 'undo_manager') else False
            self.act_undo.setEnabled(can_u)
        if hasattr(self, 'act_redo') and self.act_redo is not None:
            can_r = self.undo_manager.can_redo() if hasattr(self, 'undo_manager') else False
            self.act_redo.setEnabled(can_r)

    # ----------------------------------------------------
    # File & Action Handlers
    # ----------------------------------------------------

    def on_new_project(self, ultrafast: bool = False, prompt_save: bool = True) -> bool:
        """
        Create a new clean optical project with Object, Stop, and Image surfaces similar to Zemax.
        If ultrafast=True, configures primary wavelength to 0.800 um (Ti:Sapphire) and femtosecond analysis.
        """
        is_headless = os.environ.get("QT_QPA_PLATFORM") == "offscreen" or (
            hasattr(QtWidgets.QApplication, "platformName") and QtWidgets.QApplication.platformName() == "offscreen"
        )
        if prompt_save and getattr(self, "is_dirty", False) and QT_AVAILABLE and not is_headless:
            reply = QMessageBox.question(
                self,
                "Save Changes?",
                "Do you want to save changes to the current optical design before creating a new project?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
                QMessageBox.Save
            )
            if reply == QMessageBox.Save:
                saved = self.on_save_project()
                if not saved:
                    return False
            elif reply == QMessageBox.Cancel:
                return False

        # Capture view settings and source config onto current stack before pushing undo
        current_stack = self.model.get_surface_stack() if hasattr(self, 'model') else None
        if current_stack is not None:
            if hasattr(self, 'visualizer_3d') and hasattr(self.visualizer_3d, 'get_view_settings'):
                current_stack.viewer_settings = self.visualizer_3d.get_view_settings()
            if hasattr(self, 'source_config') and hasattr(self.source_config, 'to_dict'):
                current_stack.source_config = self.source_config.to_dict()
        if hasattr(self, 'undo_manager') and current_stack is not None:
            self.undo_manager.push_state(current_stack, getattr(self, 'system_config', None), "New Project")

        # 1. Reset System Config to clean defaults
        new_sys_cfg = SystemConfig()
        if ultrafast:
            new_sys_cfg.wavelength_list_um = [0.800]
            new_sys_cfg.primary_wavelength_um = 0.800
        else:
            new_sys_cfg.wavelength_list_um = [0.48613, 0.58756, 0.65627]
            new_sys_cfg.set_primary_wavelength(1)
        self.system_config = new_sys_cfg
        if hasattr(self, 'model') and self.model is not None:
            self.model.system_config = new_sys_cfg

        # 2. Reset Source Config
        self.source_config = SourceConfig()
        if ultrafast:
            self.source_config.wavelengths = "0.800"
        else:
            self.source_config.wavelengths = "0.48613, 0.58756, 0.65627"
        if hasattr(self, 'source_editor') and self.source_editor is not None:
            self.source_editor.set_source_config(self.source_config)

        # 3. Create clean 3-surface optical system (Object, Stop, Image) matching Zemax OpticStudio conventions
        clean_stack = SurfaceStack.create_clean_system(object_dist=100.0, stop_dist=50.0, stop_semi_diameter=10.0)
        clean_stack.system_config = new_sys_cfg

        # 4. Set the clean stack on the table model
        if hasattr(self, 'model') and self.model is not None:
            self.model.set_surface_stack(clean_stack)

        # 5. Clear file path and dirty flag
        self.current_file_path = None
        self.set_dirty(False)

        # 6. Reset System Config Dock
        if hasattr(self, 'system_config_dock') and self.system_config_dock is not None:
            self.system_config_dock.config = new_sys_cfg
            self.system_config_dock.update_ui()

        # 7. Configure pulse analysis tab and active tab
        if hasattr(self, 'analysis_tabs') and self.analysis_tabs is not None:
            if hasattr(self.analysis_tabs, 'tab_pulse'):
                self.analysis_tabs.tab_pulse.set_config({"center_wavelength_nm": 800.0 if ultrafast else 587.56})
            if ultrafast and hasattr(self.analysis_tabs, 'set_active_tab'):
                self.analysis_tabs.set_active_tab("Femtosecond Pulse")

        # 8. Update optimizer dock
        if hasattr(self, 'optimizer_dock') and self.optimizer_dock is not None:
            if hasattr(self.optimizer_dock, 'update_system_info'):
                self.optimizer_dock.update_system_info()

        # 9. Update PLM dock if active
        if hasattr(self, 'plm_dock') and self.plm_dock is not None and hasattr(self.plm_dock, 'set_surface_stack'):
            self.plm_dock.set_surface_stack(clean_stack)

        # 10. Trace rays immediately through Object -> Stop -> Image
        self.on_trace_rays()

        if hasattr(self, 'statusBar') and self.statusBar() is not None:
            self.statusBar().showMessage("New clean optical project created (Object, Stop, Image).", 3000)

        return True

    def on_load_project(self, file_path: Optional[str] = None) -> bool:
        if file_path is None and QT_AVAILABLE:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Load Project", "", "JSON Files (*.json);;All Files (*)"
            )
        if not file_path or not os.path.exists(file_path):
            return False
        try:
            data = SurfaceStack.load_from_json(file_path)
            
            if isinstance(data, dict):
                stack = data.get("stack")
                wavelengths = data.get("wavelengths", [0.450, 0.546, 0.650])
                sys_data = data.get("system_config")
                src_data = data.get("source_config")
                opt_data = data.get("optimizer_config")
                analysis_data = data.get("analysis_config")
                two_photon_data = data.get("two_photon_config")
                nonlinear_data = data.get("nonlinear_imaging_config")
                routes_data = data.get("optical_routes")
            else:
                stack = data
                wavelengths = [0.450, 0.546, 0.650]
                sys_data = getattr(stack, 'system_config', None)
                src_data = getattr(stack, 'source_config', None)
                opt_data = getattr(stack, 'optimizer_config', None)
                analysis_data = getattr(stack, 'analysis_config', None)
                two_photon_data = getattr(stack, 'two_photon_config', None)
                nonlinear_data = getattr(stack, 'nonlinear_imaging_config', None)
                routes_data = getattr(stack, 'optical_routes', None)
                
            # 1. Restore System Explorer / System Config first so stack and views have correct config
            loaded_system_config = SystemConfig()
            if isinstance(sys_data, dict):
                loaded_system_config.from_dict(sys_data)
            else:
                loaded_system_config.wavelengths = []
                for i, wl in enumerate(wavelengths):
                    loaded_system_config.add_wavelength(wl, is_primary=(i == 0))

            self.system_config = loaded_system_config
            self.model.system_config = loaded_system_config
            if stack is not None:
                stack.system_config = loaded_system_config
            if hasattr(self, 'system_config_dock') and self.system_config_dock is not None:
                self.system_config_dock.config = loaded_system_config
                self.system_config_dock.update_ui()

            # Set surface stack without prematurely emitting system_updated (traced once at end of load)
            self.model.set_surface_stack(stack, emit_signal=False)
            if hasattr(self.model, '_reindex_surface_names'):
                self.model._reindex_surface_names()

            # 2. Restore Source Editor
            if src_data is not None:
                if hasattr(self, 'source_config') and hasattr(self.source_config, 'from_dict'):
                    self.source_config.from_dict(src_data)
                if hasattr(self, 'source_editor') and self.source_editor is not None:
                    self.source_editor.update_ui()

            # Ensure collimated source_config is respected if object distance d0 <= target_efl (cannot form real image as point source)
            if (
                isinstance(src_data, dict)
                and str(src_data.get("source_type", "")).lower() == "collimated"
                and stack is not None
                and len(stack.surfaces) > 1
            ):
                d0_val = float(getattr(stack.surfaces[0], "thickness", 0.0) or 0.0)
                t_efl = float(opt_data.get("target_efl", 0.0) or 0.0) if isinstance(opt_data, dict) else 0.0
                if d0_val <= 0.0 or (t_efl > 0.0 and d0_val <= t_efl * 1.05):
                    loaded_system_config.beam_type = "Collimated (Parallel Beam)"
                    src_br = float(src_data.get("beam_radius", 0.0) or 0.0)
                    if src_br > 0.0 and abs(float(getattr(loaded_system_config, "aperture_value", 25.4)) - 25.4) < 1e-3:
                        loaded_system_config.aperture_value = 2.0 * src_br
                    if hasattr(self, 'system_config_dock') and self.system_config_dock is not None:
                        self.system_config_dock.update_ui()

            # 3. Restore Optimizer Dock and rebuild variable checkboxes from loaded stack
            if hasattr(self, 'optimizer_dock') and self.optimizer_dock is not None:
                if hasattr(self.optimizer_dock, 'set_system_config'):
                    self.optimizer_dock.set_system_config(loaded_system_config)
                if hasattr(self.optimizer_dock, '_rebuild_surface_checkboxes'):
                    self.optimizer_dock._rebuild_surface_checkboxes()
                if opt_data is not None and hasattr(self.optimizer_dock, 'set_config'):
                    self.optimizer_dock.set_config(opt_data)
                if hasattr(self.optimizer_dock, 'update_system_info'):
                    self.optimizer_dock.update_system_info()

            # 4. Restore Component Editor Dock
            if hasattr(self, 'component_editor_dock') and self.component_editor_dock is not None:
                self.component_editor_dock.update_ui()

            # 4b. Restore analysis tools, including persistent pulse probes.
            if hasattr(self, 'analysis_tabs') and self.analysis_tabs is not None:
                if hasattr(self.analysis_tabs, 'set_analysis_config'):
                    restored_analysis = dict(analysis_data) if isinstance(analysis_data, dict) else {}
                    if two_photon_data is not None:
                        restored_analysis["two_photon_microscope"] = (
                            two_photon_data.to_dict() if hasattr(two_photon_data, "to_dict") else two_photon_data
                        )
                    if nonlinear_data is not None:
                        restored_analysis["nonlinear_imaging"] = (
                            nonlinear_data.to_dict() if hasattr(nonlinear_data, "to_dict") else nonlinear_data
                        )
                    if routes_data is not None:
                        restored_analysis["optical_routes"] = (
                            routes_data.to_dict() if hasattr(routes_data, "to_dict") else routes_data
                        )
                    self.analysis_tabs.set_analysis_config(restored_analysis)

            # 5. Restore Wavelengths text input
            if hasattr(self, 'wavelengths_input'):
                if hasattr(self, 'system_config') and self.system_config.wavelengths:
                    w_str = ", ".join(f"{w.wavelength_um:.4g}" for w in self.system_config.wavelengths)
                else:
                    w_str = ", ".join(map(str, wavelengths))
                self.wavelengths_input.setText(w_str)
                
            if QT_AVAILABLE and hasattr(self.model, 'layoutChanged'):
                self.model.layoutChanged.emit()

            if hasattr(self, "dual_view") and self.dual_view is not None:
                if hasattr(self.dual_view, "reset_view_limits"):
                    self.dual_view.reset_view_limits()
                else:
                    self.dual_view.saved_limits.clear()
                    self.dual_view._force_reset_view = True

            self.on_trace_rays()
            if hasattr(self, 'plm_dock') and self.plm_dock is not None:
                self.plm_dock.set_surface_stack(stack)
            self.current_file_path = file_path
            self.is_dirty = False
            base_name = os.path.basename(file_path)
            self._update_window_title()
            if hasattr(self, 'undo_manager') and self.undo_manager is not None:
                self.undo_manager.push_state(stack, getattr(self, 'system_config', None), f"Loaded Project ({base_name})")

            msg = f"Loaded project: {base_name}"
            if QT_AVAILABLE:
                self.statusBar().showMessage(msg)
            else:
                self._status_msg = msg
            return True
        except Exception as e:
            if QT_AVAILABLE and hasattr(self, 'isVisible') and self.isVisible():
                QMessageBox.critical(self, "Error Loading Project", str(e))
            return False

    def on_revert_to_history_step(self, step_number: int) -> bool:
        """
        Restore the optical design state from a specific historical revision step.
        """
        if not hasattr(self, 'undo_manager'):
            return False
        current_stack = self.model.get_surface_stack()
        restored = self.undo_manager.jump_to_state_by_step(step_number, current_stack, getattr(self, 'system_config', None))
        if restored is not None:
            self._apply_restored_state(restored)
            msg = f"Restored design to Step #{step_number}: {restored.description}"
            if QT_AVAILABLE and hasattr(self, 'statusBar') and self.statusBar() is not None:
                self.statusBar().showMessage(msg)
            self._status_msg = msg
            self._update_undo_redo_actions()
            return True
        return False

    def on_save_project(self, file_path: Optional[str] = None) -> bool:
        """
        Save current optical design (Ctrl+S).
        If already associated with a file, saves directly without popup prompt.
        If untitled, prompts Save As dialog.
        """
        target_path = file_path or self.current_file_path
        if not target_path:
            return self.on_save_as_project()
        return self._execute_save(target_path)

    def on_save_as_project(self, file_path: Optional[str] = None) -> bool:
        """
        Save current optical design to a new chosen file destination (Ctrl+Shift+S).
        """
        if file_path is None and QT_AVAILABLE:
            default_dir = self.current_file_path or ""
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save Project As", default_dir, "JSON Files (*.json);;Zemax Files (*.zmx);;All Files (*)"
            )
        if not file_path:
            return False
        return self._execute_save(file_path)

    def set_dirty(self, dirty: bool = True) -> None:
        """Mark design as modified since last save and update window title."""
        self.is_dirty = dirty
        self._update_window_title()

    def _update_window_title(self) -> None:
        base_name = os.path.basename(self.current_file_path) if self.current_file_path else "Untitled"
        dirty_mark = " *" if self.is_dirty else ""
        if QT_AVAILABLE and hasattr(self, 'setWindowTitle'):
            self.setWindowTitle(f"2D Optics Simulation & Design Studio - [{base_name}{dirty_mark}]")

    def _on_toggle_autosave(self, enabled: bool) -> None:
        self.autosave_enabled = enabled
        if hasattr(self, '_autosave_timer') and self._autosave_timer:
            if enabled:
                self._autosave_timer.start(self.autosave_interval_ms)
            else:
                self._autosave_timer.stop()
        msg = f"⏰ Auto-save {'enabled (5 min)' if enabled else 'disabled'}"
        if QT_AVAILABLE and hasattr(self, 'statusBar'):
            self.statusBar().showMessage(msg, 3000)
        else:
            self._status_msg = msg

    def set_autosave_interval(self, minutes: float) -> None:
        """Set the autosave interval in minutes."""
        if minutes <= 0:
            self._on_toggle_autosave(False)
            if hasattr(self, 'act_autosave') and self.act_autosave is not None:
                self.act_autosave.setChecked(False)
            return

        self.autosave_interval_ms = int(minutes * 60 * 1000)
        self.autosave_enabled = True
        if hasattr(self, '_autosave_timer') and self._autosave_timer:
            self._autosave_timer.setInterval(self.autosave_interval_ms)
            if not self._autosave_timer.isActive():
                self._autosave_timer.start()
        if hasattr(self, 'act_autosave') and self.act_autosave is not None:
            self.act_autosave.setChecked(True)
            min_str = int(minutes) if minutes == int(minutes) else minutes
            self.act_autosave.setText(f"⏰ Auto-Save Enabled (Every {min_str} Min)")
        msg = f"⏰ Auto-save interval set to {minutes} min"
        if QT_AVAILABLE and hasattr(self, 'statusBar'):
            self.statusBar().showMessage(msg, 3000)
        else:
            self._status_msg = msg

    def _execute_silent_save(self, file_path: str) -> bool:
        """Helper to save project state to file without modifying current_file_path or pushing undo state."""
        try:
            stack = self.model.get_surface_stack()
            if hasattr(self, 'system_config_dock') and self.system_config_dock is not None:
                if hasattr(self.system_config_dock, 'sync_from_ui'):
                    self.system_config_dock.sync_from_ui()

            sys_cfg = getattr(self, 'system_config', None)
            if sys_cfg is not None:
                self.model.system_config = sys_cfg
                stack.system_config = sys_cfg

            wavelengths = None
            if sys_cfg is not None and sys_cfg.wavelengths:
                wavelengths = [w.wavelength_um for w in sys_cfg.wavelengths]

            source_cfg = getattr(self, 'source_config', None)
            if hasattr(self, 'source_editor') and self.source_editor is not None:
                if hasattr(self.source_editor, 'get_source_config'):
                    source_cfg = self.source_editor.get_source_config()

            opt_cfg = None
            if hasattr(self, 'optimizer_dock') and self.optimizer_dock is not None:
                if hasattr(self.optimizer_dock, 'get_config'):
                    opt_cfg = self.optimizer_dock.get_config()

            analysis_cfg = None
            if hasattr(self, 'analysis_tabs') and self.analysis_tabs is not None:
                if hasattr(self.analysis_tabs, 'get_analysis_config'):
                    analysis_cfg = self.analysis_tabs.get_analysis_config()

            if source_cfg is not None:
                stack.source_config = source_cfg

            stack.save_to_json(
                filepath=file_path,
                wavelengths=wavelengths,
                system_config=sys_cfg,
                optimizer_config=opt_cfg,
                analysis_config=analysis_cfg
            )
            return True
        except Exception as e:
            print(f"[MainWindow] Silent save failed: {e}")
            return False

    def on_autosave(self, manual_trigger: bool = False) -> Optional[str]:
        """
        Periodically auto-saves the current project in the background without interrupting the user.
        - If current_file_path is set: saves directly and archives a snapshot to .history and .autosave.
        - If untitled (current_file_path is None): saves to a dedicated recovery autosave file
          (.autosave/autosave_untitled.json) without popping up a blocking file dialog.
        """
        if not self.autosave_enabled and not manual_trigger:
            return None

        # Check if there is anything to save
        stack = self.model.get_surface_stack() if hasattr(self, 'model') else None
        if stack is None or not hasattr(stack, 'surfaces') or len(stack.surfaces) == 0:
            return None

        # If project is not dirty and not manually triggered, skip duplicate saving
        if not self.is_dirty and not manual_trigger and self.last_autosave_path and os.path.exists(self.last_autosave_path):
            return None

        try:
            now_dt = datetime.now()
            time_str = now_dt.strftime("%H:%M:%S")

            if self.current_file_path:
                abs_path = os.path.abspath(self.current_file_path)
                proj_dir = os.path.dirname(abs_path)
                base_name = os.path.basename(abs_path)
                stem, ext = os.path.splitext(base_name)
                ext = ext or ".json"

                autosave_dir = os.path.join(proj_dir, ".autosave")
                os.makedirs(autosave_dir, exist_ok=True)
                autosave_file = os.path.join(autosave_dir, f"{stem}_autosave{ext}")

                self._execute_save(self.current_file_path, is_autosave=True)

                try:
                    shutil.copy2(self.current_file_path, autosave_file)
                except Exception:
                    pass

                self.last_autosave_path = self.current_file_path
                self.last_autosave_time = now_dt
                self.is_dirty = False
                self._update_window_title()

                msg = f"⏰ [Auto-Save] Saved project to {base_name} ({time_str})"
                if QT_AVAILABLE and hasattr(self, 'statusBar'):
                    self.statusBar().showMessage(msg, 5000)
                else:
                    self._status_msg = msg
                return self.current_file_path
            else:
                autosave_dir = os.path.join(os.getcwd(), ".autosave")
                os.makedirs(autosave_dir, exist_ok=True)
                recovery_file = os.path.join(autosave_dir, "autosave_untitled.json")

                success = self._execute_silent_save(recovery_file)
                if not success:
                    return None

                self.last_autosave_path = recovery_file
                self.last_autosave_time = now_dt
                self.is_dirty = False
                self._update_window_title()

                msg = f"⏰ [Auto-Save] Recovery backup saved to .autosave/autosave_untitled.json ({time_str})"
                if QT_AVAILABLE and hasattr(self, 'statusBar'):
                    self.statusBar().showMessage(msg, 5000)
                else:
                    self._status_msg = msg
                return recovery_file
        except Exception as e:
            err_msg = f"⚠️ [Auto-Save] Failed to auto-save: {e}"
            if QT_AVAILABLE and hasattr(self, 'statusBar'):
                self.statusBar().showMessage(err_msg, 5000)
            print(f"[MainWindow] {err_msg}")
            return None

    def on_recover_autosave(self) -> bool:
        """Search for and restore the most recent autosaved project recovery snapshot."""
        candidates = []
        if self.current_file_path:
            p_dir = os.path.dirname(os.path.abspath(self.current_file_path))
            stem = os.path.splitext(os.path.basename(self.current_file_path))[0]
            auto_file = os.path.join(p_dir, ".autosave", f"{stem}_autosave.json")
            if os.path.exists(auto_file):
                candidates.append(auto_file)

        cwd_untitled = os.path.join(os.getcwd(), ".autosave", "autosave_untitled.json")
        if os.path.exists(cwd_untitled):
            candidates.append(cwd_untitled)

        if self.current_file_path:
            v_list = VersionManager.get_archived_versions(self.current_file_path)
            if v_list:
                candidates.append(v_list[-1]["path"])

        if candidates:
            candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            chosen = candidates[0]
            mtime_str = datetime.fromtimestamp(os.path.getmtime(chosen)).strftime("%Y-%m-%d %H:%M:%S")
            if QT_AVAILABLE:
                reply = QMessageBox.question(
                    self,
                    "Recover Auto-Saved Project",
                    f"Found recent auto-saved snapshot:\n\n{chosen}\nModified: {mtime_str}\n\nWould you like to recover this design?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes
                )
                if reply == QMessageBox.Yes:
                    return self.on_load_project(file_path=chosen)
                return False
            else:
                return self.on_load_project(file_path=chosen)
        else:
            if QT_AVAILABLE:
                QMessageBox.information(self, "No Auto-Save Found", "No auto-save recovery snapshots were found for this workspace.")
            return False

    def _execute_save(self, file_path: str, is_autosave: bool = False) -> bool:
        try:
            stack = self.model.get_surface_stack()
            
            # Sync system explorer UI controls before saving
            if hasattr(self, 'system_config_dock') and self.system_config_dock is not None:
                if hasattr(self.system_config_dock, 'sync_from_ui'):
                    self.system_config_dock.sync_from_ui()

            sys_cfg = getattr(self, 'system_config', None)
            if sys_cfg is not None:
                # Keep every live consumer attached to the exact configuration
                # object being serialized.
                self.model.system_config = sys_cfg
                stack.system_config = sys_cfg
            
            # Extract wavelengths
            wavelengths = None
            if sys_cfg is not None and sys_cfg.wavelengths:
                wavelengths = [w.wavelength_um for w in sys_cfg.wavelengths]
            elif hasattr(self, 'wavelengths_input') and self.wavelengths_input.text().strip():
                try:
                    wl_str = self.wavelengths_input.text()
                    parsed = [float(w.strip()) for w in wl_str.split(',') if w.strip()]
                    wavelengths = [w for w in parsed if w > 0]
                except ValueError:
                    pass

            # Source config
            source_cfg = getattr(self, 'source_config', None)
            if hasattr(self, 'source_editor') and self.source_editor is not None:
                if hasattr(self.source_editor, 'get_source_config'):
                    source_cfg = self.source_editor.get_source_config()

            # Optimizer config
            opt_cfg = None
            if hasattr(self, 'optimizer_dock') and self.optimizer_dock is not None:
                if hasattr(self.optimizer_dock, 'get_config'):
                    opt_cfg = self.optimizer_dock.get_config()

            analysis_cfg = None
            if hasattr(self, 'analysis_tabs') and self.analysis_tabs is not None:
                if hasattr(self.analysis_tabs, 'get_analysis_config'):
                    analysis_cfg = self.analysis_tabs.get_analysis_config()

            if isinstance(analysis_cfg, dict):
                if "two_photon_microscope" in analysis_cfg:
                    from optics_app.analysis.two_photon import TwoPhotonMicroscopeConfig
                    stack.two_photon_config = TwoPhotonMicroscopeConfig.from_dict(analysis_cfg["two_photon_microscope"])
                if "nonlinear_imaging" in analysis_cfg:
                    from optics_app.analysis.nonlinear_imaging import NonlinearImagingConfig
                    stack.nonlinear_imaging_config = NonlinearImagingConfig.from_dict(analysis_cfg["nonlinear_imaging"])
                if "optical_routes" in analysis_cfg:
                    from optics_app.engine.routes import OpticalRouteSet
                    stack.optical_routes = OpticalRouteSet.from_dict(analysis_cfg["optical_routes"])

            if source_cfg is not None:
                stack.source_config = source_cfg

            stack.save_to_json(
                filepath=file_path,
                wavelengths=wavelengths,
                system_config=sys_cfg,
                optimizer_config=opt_cfg,
                analysis_config=analysis_cfg
            )
            
            self.current_file_path = file_path
            self.is_dirty = False
            base_name = os.path.basename(file_path)

            # Automatically create version archive copy and log past modifications
            archive_path = VersionManager.archive_version(
                file_path=file_path,
                stack=stack,
                system_config=sys_cfg,
                wavelengths=wavelengths,
                optimizer_config=opt_cfg,
                analysis_config=analysis_cfg,
                description=f"{'Auto-saved' if is_autosave else 'Saved'} snapshot of {base_name}"
            )

            self._update_window_title()

            if hasattr(self, 'undo_manager') and self.undo_manager is not None and not is_autosave:
                self.undo_manager.push_state(stack, sys_cfg, f"Saved Project ({base_name})")

            prefix = "⏰ [Auto-Save] Saved project: " if is_autosave else "Saved project: "
            msg = f"{prefix}{base_name}" + (f" (Archived version to .history)" if archive_path else "")
            if QT_AVAILABLE:
                self.statusBar().showMessage(msg, 5000 if is_autosave else 3000)
            else:
                self._status_msg = msg
            return True
        except Exception as e:
            if QT_AVAILABLE and hasattr(self, 'isVisible') and self.isVisible():
                QMessageBox.critical(self, "Error Saving Project", str(e))
            return False

    def on_save_system_config(self, file_path: Optional[str] = None) -> Optional[str]:
        """Save Zemax System Explorer parameters to file."""
        if hasattr(self, 'system_config_dock') and self.system_config_dock is not None:
            res = self.system_config_dock.save_config_file(file_path)
            if res and QT_AVAILABLE and hasattr(self, 'statusBar'):
                self.statusBar().showMessage(f"Saved System Explorer configuration: {os.path.basename(res)}")
            return res
        elif hasattr(self, 'system_config') and self.system_config is not None:
            if file_path is None and QT_AVAILABLE:
                file_path, _ = QFileDialog.getSaveFileName(
                    self, "Save System Explorer Configuration", "system_config.json", "JSON Files (*.json);;All Files (*)"
                )
            if file_path:
                self.system_config.save_to_file(file_path)
                if QT_AVAILABLE and hasattr(self, 'statusBar'):
                    self.statusBar().showMessage(f"Saved System Explorer configuration: {os.path.basename(file_path)}")
                return file_path
        return None

    def on_load_system_config(self, file_path: Optional[str] = None) -> Optional[str]:
        """Load Zemax System Explorer parameters from file and trace rays."""
        if hasattr(self, 'system_config_dock') and self.system_config_dock is not None:
            res = self.system_config_dock.load_config_file(file_path)
            if res:
                self.system_config = self.system_config_dock.config
                self.model.system_config = self.system_config
                self.on_trace_rays()
                if QT_AVAILABLE and hasattr(self, 'statusBar'):
                    self.statusBar().showMessage(f"Loaded System Explorer configuration: {os.path.basename(res)}")
                return res
        elif hasattr(self, 'system_config') and self.system_config is not None:
            if file_path is None and QT_AVAILABLE:
                file_path, _ = QFileDialog.getOpenFileName(
                    self, "Load System Explorer Configuration", "", "JSON Files (*.json);;All Files (*)"
                )
            if file_path and os.path.exists(file_path):
                self.system_config.load_from_file(file_path)
                self.model.system_config = self.system_config
                self.on_trace_rays()
                if QT_AVAILABLE and hasattr(self, 'statusBar'):
                    self.statusBar().showMessage(f"Loaded System Explorer configuration: {os.path.basename(file_path)}")
                return file_path
        return None

    def on_open_history_folder(self) -> None:
        """
        Open the .history folder containing all past modification version copies.
        """
        target = self.current_file_path or "C:/Users/Juan/Desktop/spectrometer_NIR.json"
        proj_dir = os.path.dirname(os.path.abspath(target)) if os.path.isabs(target) else os.getcwd()
        hist_dir = os.path.join(proj_dir, ".history")
        os.makedirs(hist_dir, exist_ok=True)
        
        if QT_AVAILABLE and hasattr(QtGui, 'QDesktopServices') and hasattr(QtCore, 'QUrl'):
            try:
                QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(hist_dir))
                return
            except Exception:
                pass

        try:
            if os.name == 'nt' and hasattr(os, 'startfile'):
                os.startfile(hist_dir)
        except Exception:
            pass

    def on_insert_zmx(self, file_path: Optional[str] = None) -> bool:
        """
        Insert a Zemax .zmx element into the current prescription.
        """
        if file_path is None and QT_AVAILABLE:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Insert Zemax Element", "", "Zemax Files (*.zmx);;All Files (*)"
            )

        if not file_path or not os.path.exists(file_path):
            return False

        try:
            stack = parse_zmx_file(file_path)
            current_stack = self.model.get_surface_stack()
            
            if QT_AVAILABLE:
                max_surf = max(0, len(current_stack.surfaces) - 1)
                dialog = ZemaxImportDialog(self, max_surfaces=max_surf)
                exec_fn = getattr(dialog, 'exec_', dialog.exec)
                accepted_val = getattr(QDialog, 'Accepted', 1) if QDialog is not None else 1
                if exec_fn() != accepted_val:
                    return False
                settings = dialog.get_settings()
            else:
                settings = {"insert_index": max(0, len(current_stack.surfaces) - 2), "reverse": False, "tilt_x": 0.0, "tilt_y": 0.0, "tilt_z": 0.0}

            processed = process_imported_surfaces(stack.surfaces, settings)
            if not processed:
                return False

            insert_after = settings.get("insert_index", len(current_stack.surfaces) - 2)
            idx = insert_after + 1
            
            if hasattr(self.model, '_push_undo'):
                self.model._push_undo(f"Insert Zemax file {os.path.basename(file_path)}")
                
            for k, s in enumerate(processed):
                current_stack.insert_surface(idx + k, s)
                
            self.model.set_surface_stack(current_stack)
            if hasattr(self.model, '_reindex_surface_names'):
                self.model._reindex_surface_names()
            if QT_AVAILABLE and hasattr(self.model, 'layoutChanged'):
                self.model.layoutChanged.emit()
            self.on_trace_rays()
            msg = f"Inserted Zemax element: {os.path.basename(file_path)}"
            if QT_AVAILABLE:
                self.statusBar().showMessage(msg)
            else:
                self._status_msg = msg
            return True
        except Exception as e:
            if QT_AVAILABLE and hasattr(self, 'isVisible') and self.isVisible():
                QMessageBox.critical(self, "Error Inserting Zemax File", str(e))
            return False

    def on_open_zmx(self, file_path: Optional[str] = None) -> bool:
        """
        Open and parse a Zemax .zmx prescription text file.
        Populates LensDataTableModel and updates dual-view & analysis tabs.
        """
        if file_path is None and QT_AVAILABLE:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Open Zemax Prescription", "", "Zemax Files (*.zmx);;All Files (*)"
            )

        if not file_path or not os.path.exists(file_path):
            return False

        try:
            stack = parse_zmx_file(file_path)

            # Sync wavelengths from Zemax file if present
            wls = getattr(stack, 'wavelengths', None)
            if wls and len(wls) > 0:
                if hasattr(self, 'system_config'):
                    self.system_config.wavelengths = []
                    for i, wl in enumerate(wls):
                        self.system_config.add_wavelength(wl, is_primary=(i==0))
                    if hasattr(self, 'system_config_dock'):
                        self.system_config_dock.update_ui()
                if hasattr(self, 'wavelengths_input'):
                    self.wavelengths_input.setText(", ".join(map(str, wls)))

            # Ensure proper object and image surface formatting
            if len(stack.surfaces) > 0:
                stack.surfaces[0].surface_type = SurfaceType.OBJECT
                if stack.surfaces[0].thickness <= 0:
                    stack.surfaces[0].thickness = 100.0
                if len(stack.surfaces) > 1:
                    stack.surfaces[-1].surface_type = SurfaceType.DETECTOR

            self.model.set_surface_stack(stack)
            if hasattr(self.model, '_reindex_surface_names'):
                self.model._reindex_surface_names()
            if QT_AVAILABLE and hasattr(self.model, 'layoutChanged'):
                self.model.layoutChanged.emit()
            self.on_trace_rays()
            msg = f"Loaded Zemax prescription: {os.path.basename(file_path)}"
            if QT_AVAILABLE:
                self.statusBar().showMessage(msg)
            else:
                self._status_msg = msg
            return True
        except Exception as e:
            if QT_AVAILABLE and hasattr(self, 'isVisible') and self.isVisible():
                QMessageBox.critical(self, "Error Loading Zemax File", str(e))
            return False

    def on_export_step(
        self,
        file_path: Optional[str] = None,
        show_dialog: bool = True,
        selected_element_indices: Optional[Collection[int]] = None,
        engine: str = "auto"
    ) -> bool:
        """
        Export current optical system as a 3D STEP CAD assembly file.
        Opens an interactive pop-up window letting the user choose which elements to export together.
        """
        stack = self.model.get_surface_stack()
        if stack is None or len(getattr(stack, 'surfaces', [])) == 0:
            if QT_AVAILABLE and hasattr(self, 'isVisible') and self.isVisible():
                QMessageBox.warning(self, "Export STEP", "No optical surfaces in current system.")
            return False

        is_test_env = ("pytest" in sys.modules) or (os.environ.get("QT_QPA_PLATFORM") == "offscreen")
        should_pop = show_dialog and QT_AVAILABLE and not is_test_env and hasattr(self, 'isVisible') and self.isVisible()

        if should_pop:
            from optics_app.gui.step_export_dialog import STEPExportDialog
            default_air_diam = (
                getattr(self.visualizer_3d, "air_gap_constant_diameter", 25.4)
                if hasattr(self, "visualizer_3d") and self.visualizer_3d is not None
                else 25.4
            )
            default_mount_diam = (
                getattr(self.visualizer_3d, "mount_diameter", 30.5)
                if hasattr(self, "visualizer_3d") and self.visualizer_3d is not None
                else 30.5
            )
            default_mount_z = (
                getattr(self.visualizer_3d, "mount_z_thickness", 0.0)
                if hasattr(self, "visualizer_3d") and self.visualizer_3d is not None
                else 0.0
            )
            default_export_mounts = (
                getattr(self.visualizer_3d, "show_mounts", False)
                if hasattr(self, "visualizer_3d") and self.visualizer_3d is not None
                else False
            )
            default_tol_z = (
                getattr(self.visualizer_3d, "tolerance_thickness", 1.0)
                if hasattr(self, "visualizer_3d") and self.visualizer_3d is not None
                else 1.0
            )
            dialog = STEPExportDialog(
                stack, parent=self, default_file_path=file_path, default_engine=engine,
                default_air_gap_diam=default_air_diam,
                default_mount_diam=default_mount_diam,
                default_mount_z=default_mount_z,
                default_export_mounts=default_export_mounts,
                default_tol_z=default_tol_z,
            )
            res = dialog.exec_() if hasattr(dialog, 'exec_') else dialog.exec()
            if not res:
                return False
            file_path = dialog.get_file_path()
            selected_element_indices = dialog.get_selected_element_indices()
            engine = dialog.get_engine()
            export_enclosure = dialog.get_export_enclosure()
            air_gap_diam = dialog.get_air_gap_constant_diameter()
            mount_diam = dialog.get_mount_diameter()
            mount_z = dialog.get_mount_z_thickness()
            export_mounts = dialog.get_export_mounts()
            tol_z = dialog.get_tolerance_thickness()
            if hasattr(self, "visualizer_3d") and self.visualizer_3d is not None:
                self.visualizer_3d.set_air_gap_constant_diameter(air_gap_diam)
                self.visualizer_3d.set_mount_diameter(mount_diam)
                self.visualizer_3d.set_mount_z_thickness(mount_z)
                self.visualizer_3d.set_show_mounts(export_mounts)
                self.visualizer_3d.set_tolerance_thickness(tol_z)
        else:
            export_enclosure = False
            air_gap_diam = 25.4
            mount_diam = 30.5
            mount_z = 0.0
            export_mounts = False
            tol_z = 1.0
            if file_path is None and QT_AVAILABLE and not is_test_env:
                file_path, _ = QFileDialog.getSaveFileName(
                    self, "Export 3D STEP Assembly", "", "STEP Files (*.step *.stp);;All Files (*)"
                )
            if not file_path:
                file_path = "optical_assembly.step"

        if not file_path:
            return False

        elem_displacements = (
            dict(getattr(self.visualizer_3d, "element_displacements", {}))
            if hasattr(self, "visualizer_3d") and self.visualizer_3d is not None
            else {}
        )

        from optics_app.cad.exporter import STEPExportOptions, STEPExportEngine
        opt_engine = STEPExportEngine(engine) if isinstance(engine, str) and engine in ("auto", "build123d", "pure_python") else STEPExportEngine.AUTO
        options = STEPExportOptions(
            engine=opt_engine,
            selected_element_indices=selected_element_indices,
            export_3d_printable_enclosure=export_enclosure,
            enclosure_wall_margin_mm=10.0,
            air_gap_cylinder_diameter=air_gap_diam,
            mount_diameter=mount_diam,
            mount_z_thickness=mount_z,
            export_mechanical_mounts=export_mounts,
            tolerance_thickness_mm=tol_z,
            element_displacements=elem_displacements,
        )

        if not QT_AVAILABLE or is_test_env:
            try:
                res = CADExporter.export_system_step(
                    stack,
                    file_path,
                    selected_element_indices=selected_element_indices,
                    engine=engine,
                    options=options
                )
                self._status_msg = f"Exported 3D STEP assembly to {os.path.basename(file_path)}"
                return res.is_success
            except Exception as e:
                return False

        # Asynchronous export via QThread to prevent GUI freezing
        self.export_progress = QtWidgets.QProgressDialog("Generating 3D Assembly. Please wait...", None, 0, 0, self)
        self.export_progress.setWindowTitle("CAD Export")
        self.export_progress.setWindowModality(Qt.WindowModal)
        self.export_progress.setCancelButton(None) # Cannot easily cancel build123d once started
        if hasattr(self, 'isVisible') and self.isVisible():
            self.export_progress.show()
        
        self.cad_worker = CADExportWorker(
            stack,
            file_path,
            parent=self,
            selected_element_indices=selected_element_indices,
            engine=engine,
            options=options
        )
        
        def on_finished(success, result_msg):
            self.export_progress.accept()
            if success:
                msg = f"Exported 3D STEP assembly to {os.path.basename(result_msg)}"
                if hasattr(self, 'statusBar') and self.statusBar():
                    self.statusBar().showMessage(msg)
                if hasattr(self, 'isVisible') and self.isVisible():
                    QMessageBox.information(self, "Export Complete", msg)
            else:
                if hasattr(self, 'isVisible') and self.isVisible():
                    QMessageBox.critical(self, "Error Exporting STEP Assembly", result_msg)
                
        self.cad_worker.finished_export.connect(on_finished)
        self.cad_worker.start()
        
        return True

    def on_export_dxf_2d(self, file_path: Optional[str] = None) -> bool:
        """
        Export current optical system as a 2D DXF CAD drawing with ISO/ANSI dimensions.
        Opens directly in FreeCAD, AutoCAD, LibreCAD, QCAD, and SolidWorks.
        """
        if file_path is None and QT_AVAILABLE:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Export 2D DXF CAD Drawing", "", "DXF CAD Drawing (*.dxf);;All Files (*)"
            )

        if not file_path:
            return False

        try:
            stack = self.model.get_surface_stack()
            from optics_app.cad.exporter import export_to_dxf_2d
            export_to_dxf_2d(stack, file_path, include_dimensions=True, include_rays=True, include_hatches=True, include_table=True)
            msg = f"Exported 2D DXF CAD drawing with dimensions to {os.path.basename(file_path)}"
            self._status_msg = msg
            if QT_AVAILABLE:
                self.statusBar().showMessage(msg, 5000)
                if hasattr(self, 'isVisible') and self.isVisible():
                    QMessageBox.information(self, "2D DXF Export Complete", f"Successfully exported 2D CAD drawing with dimensions:\n{file_path}\n\nReady to open in FreeCAD (Draft / TechDraw) or AutoCAD.")
            return True
        except Exception as e:
            if QT_AVAILABLE and hasattr(self, 'isVisible') and self.isVisible():
                QMessageBox.critical(self, "Error Exporting 2D DXF", str(e))
            return False

    def on_export_svg_2d(self, file_path: Optional[str] = None) -> bool:
        """
        Export current optical system as a 2D SVG vector technical drawing with dimensions.
        Opens in FreeCAD TechDraw, Inkscape, and web browsers.
        """
        if file_path is None and QT_AVAILABLE:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Export 2D SVG Technical Drawing", "", "Scalable Vector Graphics (*.svg);;All Files (*)"
            )

        if not file_path:
            return False

        try:
            stack = self.model.get_surface_stack()
            from optics_app.cad.exporter import export_to_svg_2d
            export_to_svg_2d(stack, file_path, include_dimensions=True, include_rays=True, include_table=True)
            msg = f"Exported 2D SVG technical drawing with dimensions to {os.path.basename(file_path)}"
            self._status_msg = msg
            if QT_AVAILABLE:
                self.statusBar().showMessage(msg, 5000)
                if hasattr(self, 'isVisible') and self.isVisible():
                    QMessageBox.information(self, "2D SVG Export Complete", f"Successfully exported 2D vector drawing with dimensions:\n{file_path}")
            return True
        except Exception as e:
            if QT_AVAILABLE and hasattr(self, 'isVisible') and self.isVisible():
                QMessageBox.critical(self, "Error Exporting 2D SVG", str(e))
            return False

    def on_load_cad_surface(self, file_path: Optional[str] = None, row_index: Optional[int] = None) -> bool:
        """
        Load a 3D CAD file (.step, .stp, .sldprt, .sldasm) and extract 2D optical surface profile.
        """
        if file_path is None and QT_AVAILABLE:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Load CAD Surface Profile", "", "CAD Files (*.step *.stp *.sldprt *.sldasm);;All Files (*)"
            )

        if not file_path or not os.path.exists(file_path):
            return False

        try:
            importer = CADSurfaceImporter(file_path)
            res = importer.load_and_extract()
            prof = res.front_profile

            if prof is not None:
                if row_index is None:
                    row_index = self.model.rowCount() - 1 if self.model.rowCount() > 0 else 0
                    self.model.insertRows(row_index, 1)

                surf_name = os.path.splitext(os.path.basename(file_path))[0]
                if 0 <= row_index < len(self.model._data):
                    while len(self.model._data[row_index]) < len(self.model.COLUMNS):
                        self.model._data[row_index].append("")
                    self.model._data[row_index][0] = surf_name
                    self.model._data[row_index][1] = "CAD"
                    self.model._data[row_index][2] = float(prof.radius_of_curvature)
                    self.model._data[row_index][3] = float(res.center_thickness if res.center_thickness > 0 else 5.0)
                    self.model._data[row_index][5] = float(prof.clear_aperture if prof.clear_aperture > 0 else 20.0)
                    self.model._data[row_index][13] = file_path
                    self.model._sync_stack_from_data()
                else:
                    self.model.setData(row_index, 0, surf_name)
                    self.model.setData(row_index, 1, "CAD")
                    self.model.setData(row_index, 2, float(prof.radius_of_curvature))
                    self.model.setData(row_index, 3, float(res.center_thickness if res.center_thickness > 0 else 5.0))
                    self.model.setData(row_index, 5, float(prof.clear_aperture if prof.clear_aperture > 0 else 20.0))
                    self.model.setData(row_index, 13, file_path)

                self.on_trace_rays()
                msg = f"Loaded CAD surface profile from {os.path.basename(file_path)}"
                if QT_AVAILABLE:
                    self.statusBar().showMessage(msg)
                else:
                    self._status_msg = msg
                return True
            return False
        except Exception as e:
            if QT_AVAILABLE and hasattr(self, 'isVisible') and self.isVisible():
                QMessageBox.critical(self, "Error Loading CAD Surface", str(e))
            return False

    def on_insert_surface(self, position: Optional[int] = None) -> bool:
        """Insert a new surface row into Lens Data Editor."""
        if position is not None:
            pos = position
        else:
            max_idx = max(0, self.model.rowCount() - 1)
            default_val = max_idx
            if self.table_view:
                c_idx = self.table_view.currentIndex()
                if c_idx and hasattr(c_idx, "row") and 0 <= c_idx.row() <= max_idx:
                    default_val = c_idx.row()
                else:
                    sel = self.table_view.selectionModel()
                    if sel and sel.hasSelection():
                        rows = sel.selectedRows()
                        if rows and len(rows) > 0:
                            default_val = rows[0].row()
                        else:
                            indexes = sel.selectedIndexes()
                            if indexes and len(indexes) > 0:
                                default_val = indexes[0].row()

            if QT_AVAILABLE and os.environ.get("QT_QPA_PLATFORM") != "offscreen":
                from optics_app.gui.qt_compat import QtWidgets
                val, ok = QtWidgets.QInputDialog.getInt(
                    self, "Insert Surface", 
                    "Insert AFTER Surface Index:", 
                    default_val, 0, max_idx, 1
                )
                if not ok:
                    return False
                pos = val + 1
            else:
                pos = default_val + 1 if default_val < max_idx else self.model.rowCount()

        success = self.model.insertRows(pos)
        if success:
            if hasattr(self.model, '_reindex_surface_names'):
                self.model._reindex_surface_names()
                if hasattr(self.model, 'layoutChanged'):
                    self.model.layoutChanged.emit()
            if self.table_view and hasattr(self.table_view, 'setCurrentIndex'):
                try:
                    new_idx = self.model.index(pos, 0)
                    self.table_view.setCurrentIndex(new_idx)
                except Exception:
                    pass
            self._update_selected_surface_highlight(pos)
            msg = f"Inserted surface at row {pos} (inherited sequential optical positions)"
            if QT_AVAILABLE:
                self.statusBar().showMessage(msg, 3000)
            else:
                self._status_msg = msg
            self.on_trace_rays()
        return success

    def on_delete_surface(self, position: Optional[int] = None) -> bool:
        """Delete selected or last lens surface row from Lens Data Editor."""
        if not self.model or len(self.model._data) == 0:
            return False

        if position is not None:
            pos = position
        elif self.table_view:
            c_idx = self.table_view.currentIndex()
            if c_idx and hasattr(c_idx, "row") and 0 <= c_idx.row() < len(self.model._data):
                pos = c_idx.row()
            else:
                sel = self.table_view.selectionModel()
                if sel and sel.hasSelection():
                    rows = sel.selectedRows()
                    if rows and len(rows) > 0:
                        pos = rows[0].row()
                    else:
                        indexes = sel.selectedIndexes()
                        if indexes and len(indexes) > 0:
                            pos = indexes[0].row()
                        else:
                            pos = len(self.model._data) - 1
                else:
                    pos = len(self.model._data) - 1
        else:
            pos = len(self.model._data) - 1

        span = self.model.component_span_for_row(pos) if self.model.component_name_mode() else None
        success = (
            self.model.delete_lens_element(span["start"])
            if span is not None else self.model.removeRows(pos, 1)
        )
        if success:
            msg = f"Deleted complete lens '{span['name']}'" if span else f"Deleted surface row at index {pos}"
            self._status_msg = msg
            if QT_AVAILABLE:
                self.statusBar().showMessage(msg, 3000)
            self.on_trace_rays()
        return success

    def on_delete_lens_element(self, target: Optional[Any] = None) -> bool:
        """Delete an entire multi-surface lens element assembly from the system."""
        if not self.model or len(self.model._data) == 0:
            return False

        if target is None:
            # Check current selection
            c_idx = self.table_view.currentIndex() if self.table_view else None
            pos = c_idx.row() if (c_idx and hasattr(c_idx, "row") and 0 <= c_idx.row() < len(self.model._data)) else 0
            target = pos

        from optics_app.engine.element_colors import get_surface_element_group
        grp = get_surface_element_group(self.model._surfaces, target if isinstance(target, int) else 0)
        g_name = grp.get("group_name", f"Lens at {target}")

        success = self.model.delete_lens_element(target)
        if success:
            msg = f"Deleted lens element '{g_name}' and updated optical system"
            self._status_msg = msg
            if QT_AVAILABLE:
                self.statusBar().showMessage(msg, 3000)
            self.on_trace_rays()
        return success

    def on_toggle_lens_edition(self, target: Optional[Any] = None) -> bool:
        """Toggle edition lock/unlock state for selected lens element."""
        if not self.model or len(self.model._data) == 0:
            return False

        if target is None:
            c_idx = self.table_view.currentIndex() if self.table_view else None
            pos = c_idx.row() if (c_idx and hasattr(c_idx, "row") and 0 <= c_idx.row() < len(self.model._data)) else 0
            target = pos

        is_locked = self.model.is_lens_locked(target if isinstance(target, int) else 0)
        new_locked = not is_locked
        success = self.model.set_lens_locked(target, locked=new_locked)
        if success:
            state_str = "LOCKED (Fixed Prescription)" if new_locked else "UNLOCKED (Edition Enabled)"
            msg = f"Lens parameters state: {state_str}"
            self._status_msg = msg
            if QT_AVAILABLE:
                self.statusBar().showMessage(msg, 3000)
        return success

    def on_open_aperture_dialog(self, row: Optional[int] = None) -> bool:
        """Open the interactive Aperture and Stop definition dialog."""
        if not QT_AVAILABLE or not self.model:
            return False
        if row is None:
            row = 0
            if self.table_view:
                c_idx = self.table_view.currentIndex()
                if c_idx and hasattr(c_idx, "row") and 0 <= c_idx.row() < self.model.rowCount():
                    row = c_idx.row()
        return self.model.open_aperture_dialog(row, parent=self)

    def on_toggle_stop(self, row: Optional[int] = None) -> bool:
        """Toggle Aperture Stop (STO) status on the selected surface."""
        if not self.model:
            return False
        if row is None:
            row = 0
            if self.table_view:
                c_idx = self.table_view.currentIndex()
                if c_idx and hasattr(c_idx, "row") and 0 <= c_idx.row() < self.model.rowCount():
                    row = c_idx.row()
        if 0 <= row < len(self.model._surfaces):
            s = self.model._surfaces[row]
            from optics_app.engine.element_colors import is_stop_plane
            is_curr = is_stop_plane(s)
            res = self.model.set_stop_surface(row, not is_curr)
            if res:
                status = "STO" if not is_curr else "Standard"
                msg = f"Surface {row} set to {status}"
                self._status_msg = msg
                if QT_AVAILABLE:
                    self.statusBar().showMessage(msg, 3000)
                self.on_trace_rays()
            return res
        return False

    def on_insert_aperture_stop(self, row: Optional[int] = None) -> int:
        """Insert a dedicated Aperture Stop surface into the system."""
        if not self.model:
            return -1
        if row is None:
            row = max(0, self.model.rowCount() - 2) if self.model.rowCount() > 2 else 0
            if self.table_view:
                c_idx = self.table_view.currentIndex()
                if c_idx and hasattr(c_idx, "row") and 0 <= c_idx.row() < self.model.rowCount():
                    row = c_idx.row()
        new_row = self.model.insert_aperture_stop_after(row)
        msg = f"Inserted Aperture Stop at row {new_row}"
        self._status_msg = msg
        if QT_AVAILABLE:
            self.statusBar().showMessage(msg, 3000)
        self.on_trace_rays()
        return new_row

    def on_replace_lens_zmx(self, target: Optional[Any] = None, file_path: Optional[str] = None) -> bool:
        """Replace selected lens element with a Zemax (.zmx) file."""
        if not self.model or len(self.model._data) == 0:
            return False

        if target is None:
            c_idx = self.table_view.currentIndex() if self.table_view else None
            pos = c_idx.row() if (c_idx and hasattr(c_idx, "row") and 0 <= c_idx.row() < len(self.model._data)) else 0
            target = pos

        if file_path is None and QT_AVAILABLE:
            from optics_app.gui.qt_compat import QFileDialog
            file_path, _ = QFileDialog.getOpenFileName(
                self, f"Replace Lens Element with Zemax File", "", "Zemax Files (*.zmx);;All Files (*)"
            )

        if not file_path:
            return False

        try:
            success = self.model.replace_lens_from_zmx(target, file_path)
            if success:
                import os
                msg = f"Replaced lens element with Zemax file: {os.path.basename(file_path)}"
                self._status_msg = msg
                if QT_AVAILABLE:
                    self.statusBar().showMessage(msg, 3000)
                self.on_trace_rays()
            return success
        except Exception as e:
            if QT_AVAILABLE and hasattr(self, 'isVisible') and self.isVisible():
                from optics_app.gui.qt_compat import QMessageBox
                QMessageBox.critical(self, "Replace Lens Error", str(e))
            return False

    def on_replace_lens_custom(self, target: Optional[Any] = None, file_path: Optional[str] = None) -> bool:
        """Replace selected lens element with a custom JSON file."""
        if not self.model or len(self.model._data) == 0:
            return False

        if target is None:
            c_idx = self.table_view.currentIndex() if self.table_view else None
            pos = c_idx.row() if (c_idx and hasattr(c_idx, "row") and 0 <= c_idx.row() < len(self.model._data)) else 0
            target = pos

        if file_path is None and QT_AVAILABLE:
            from optics_app.gui.qt_compat import QFileDialog
            file_path, _ = QFileDialog.getOpenFileName(
                self, f"Replace Lens Element with Custom File", "", "Custom Lens / JSON Files (*.json *.txt);;All Files (*)"
            )

        if not file_path:
            return False

        try:
            success = self.model.replace_lens_from_custom_file(target, file_path)
            if success:
                import os
                msg = f"Replaced lens element with custom file: {os.path.basename(file_path)}"
                self._status_msg = msg
                if QT_AVAILABLE:
                    self.statusBar().showMessage(msg, 3000)
                self.on_trace_rays()
            return success
        except Exception as e:
            if QT_AVAILABLE and hasattr(self, 'isVisible') and self.isVisible():
                from optics_app.gui.qt_compat import QMessageBox
                QMessageBox.critical(self, "Replace Lens Error", str(e))
            return False

    def on_replace_lens_catalog(self, part_number: str, target: Optional[Any] = None) -> bool:
        """Replace selected lens element with a Thorlabs catalog lens."""
        if not self.model or len(self.model._data) == 0:
            return False

        if target is None:
            c_idx = self.table_view.currentIndex() if self.table_view else None
            pos = c_idx.row() if (c_idx and hasattr(c_idx, "row") and 0 <= c_idx.row() < len(self.model._data)) else 0
            target = pos

        try:
            success = self.model.replace_lens_from_catalog(target, part_number)
            if success:
                msg = f"Replaced lens element with catalog lens: {part_number}"
                self._status_msg = msg
                if QT_AVAILABLE:
                    self.statusBar().showMessage(msg, 3000)
                self.on_trace_rays()
            return success
        except Exception as e:
            if QT_AVAILABLE and hasattr(self, 'isVisible') and self.isVisible():
                from optics_app.gui.qt_compat import QMessageBox
                QMessageBox.critical(self, "Replace Lens Error", str(e))
            return False

    def on_open_catalog_lens_dialog(self, target: Optional[Any] = None, mode: str = "replace") -> bool:
        """Open the Thorlabs Catalog Lens Dialog to choose parameters and replace or insert a lens."""
        if not QT_AVAILABLE:
            return False
        from optics_app.gui.catalog_lens_dialog import CatalogLensDialog
        if target is None and self.table_view:
            c_idx = self.table_view.currentIndex()
            if c_idx and hasattr(c_idx, "row") and 0 <= c_idx.row() < len(self.model._data):
                target = c_idx.row()
        dialog = CatalogLensDialog(model=self.model, parent=self, target=target, mode=mode)
        exec_fn = getattr(dialog, 'exec_', dialog.exec)
        if exec_fn():
            if hasattr(self, 'component_editor_dock') and hasattr(self.component_editor_dock, '_rebuild_surface_list'):
                self.component_editor_dock._rebuild_surface_list()
            self.on_trace_rays()
            return True
        return False

    def on_open_achromat_designer_dialog(self) -> bool:
        """Open the interactive Achromatic Doublet Lens Designer dialog."""
        if not QT_AVAILABLE:
            return False
        try:
            from v2.gui.achromat_dialog import AchromatDesignerDialog
        except ImportError:
            from optics_app.gui.achromat_dialog import AchromatDesignerDialog
        dialog = AchromatDesignerDialog(model=self.model, parent=self)
        exec_fn = getattr(dialog, 'exec_', dialog.exec)
        if exec_fn():
            if hasattr(self, 'component_editor_dock') and hasattr(self.component_editor_dock, '_rebuild_surface_list'):
                self.component_editor_dock._rebuild_surface_list()
            self.on_trace_rays()
            return True
        return False
        
    def show_lde_context_menu(self, pos):
        if not QT_AVAILABLE:
            return
        from optics_app.gui.qt_compat import QMenu, QAction
        menu = QMenu(self.table_view)

        act_clean = QAction("📄 New Clean Project (Object, Stop, Image) (Ctrl+N)", self)
        act_clean.setStatusTip("Reset system to clean 3-surface Zemax configuration")
        act_clean.triggered.connect(lambda: self.on_new_project())
        menu.addAction(act_clean)
        menu.addSeparator()
        
        # Determine cell index at position if possible
        row_idx = 0
        col_idx = 0
        if hasattr(self.table_view, "indexAt"):
            idx = self.table_view.indexAt(pos)
            if idx and hasattr(idx, "row") and idx.row() >= 0:
                row_idx = idx.row()
                col_idx = idx.column()
        elif hasattr(self.table_view, "currentIndex"):
            c_idx = self.table_view.currentIndex()
            if c_idx and hasattr(c_idx, "row") and c_idx.row() >= 0:
                row_idx = c_idx.row()
                col_idx = c_idx.column()

        # Zemax Parameter Solve Actions for the clicked cell
        param_cols = {
            2: "Radius",
            3: "Thickness",
            6: "Conic",
            7: "Decenter X",
            8: "Decenter Y",
            9: "Tilt X",
            10: "Tilt Y",
            11: "Tilt Z"
        }
        if col_idx in param_cols and 0 <= row_idx < len(self.model._surfaces):
            p_name = param_cols[col_idx]
            surf = self.model._surfaces[row_idx]
            attr_var = f"{p_name.lower().replace(' ', '_')}_variable"
            is_var = bool(getattr(surf, attr_var, False))
            
            menu_solve = menu.addMenu(f"⚡ Zemax Solve ({p_name} — S{row_idx})")
            
            act_set_var = QAction(f"⚡ Set {p_name} as Variable (V)", self)
            act_set_var.setCheckable(True)
            act_set_var.setChecked(is_var)
            act_set_var.triggered.connect(lambda checked, r=row_idx, c=col_idx: self._set_cell_variable(r, c, True))
            menu_solve.addAction(act_set_var)
            
            act_set_fix = QAction(f"🔒 Set {p_name} as Fixed (F)", self)
            act_set_fix.setCheckable(True)
            act_set_fix.setChecked(not is_var)
            act_set_fix.triggered.connect(lambda checked, r=row_idx, c=col_idx: self._set_cell_variable(r, c, False))
            menu_solve.addAction(act_set_fix)
            
            act_open_solve = QAction(f"⚙️ Open Solve / Pickup Dialog...", self)
            act_open_solve.triggered.connect(lambda checked, r=row_idx, c=col_idx: self.open_solve_dialog(r, c))
            menu_solve.addAction(act_open_solve)
            menu.addSeparator()

        # Zemax Bulk Presets Menu
        menu_presets = menu.addMenu("📐 Zemax Variable Presets")
        act_z_def = QAction("Zemax Defaults: Air Gaps Variable, Lenses Fixed", self)
        act_z_def.triggered.connect(self._apply_zemax_defaults)
        menu_presets.addAction(act_z_def)

        act_z_rad = QAction("Make All Radii Variable (Curvature Optimization)", self)
        act_z_rad.triggered.connect(self._apply_all_radii_variable)
        menu_presets.addAction(act_z_rad)

        act_z_air = QAction("Make All Air Gaps Variable (Distance Optimization)", self)
        act_z_air.triggered.connect(self._apply_all_air_gaps_variable)
        menu_presets.addAction(act_z_air)

        act_z_fix = QAction("Clear All Variables (All Fixed)", self)
        act_z_fix.triggered.connect(self._apply_all_fixed)
        menu_presets.addAction(act_z_fix)
        menu.addSeparator()

        act_insert = QAction("Insert Surface Row", self)
        act_insert.triggered.connect(lambda: self.on_insert_surface())

        act_cluster = QAction("🔗 Cluster Selected Surfaces as Lens...", self)
        act_cluster.setToolTip("Group selected surfaces into a single multi-surface lens element")
        act_cluster.triggered.connect(self._on_menu_group_selected_surfaces)

        act_uncluster = QAction("🔓 Uncluster / Clear Group", self)
        act_uncluster.triggered.connect(lambda: self._on_menu_clear_group(row_idx))

        act_achromat = QAction("🎨 Design Achromatic Doublet Lens...", self)
        act_achromat.triggered.connect(lambda: self.on_open_achromat_designer_dialog())
        
        component_span = self.model.component_span_for_row(row_idx) if self.model.component_name_mode() else None
        delete_label = f"Delete Complete Lens '{component_span['name']}'" if component_span else "Delete Surface Row"
        act_delete = QAction(delete_label, self)
        act_delete.triggered.connect(lambda: self.on_delete_surface())
        
        act_toggle_comp = QAction("🔲 Toggle Component View", self)
        act_toggle_comp.setCheckable(True)
        act_toggle_comp.setChecked(self.model.component_name_mode())
        act_toggle_comp.toggled.connect(self._set_lde_component_name_mode)

        menu.addAction(act_cluster)
        menu.addAction(act_uncluster)
        menu.addAction(act_toggle_comp)
        menu.addSeparator()
        menu.addAction(act_insert)
        menu.addAction(act_achromat)
        menu.addAction(act_delete)
        menu.addSeparator()

        # Aperture & Stop Actions
        menu_aper = menu.addMenu(f"⭕ Aperture & Stop (Surface {row_idx})")
        act_aper_dlg = QAction("⚙️ Define Aperture & Stop Parameters...", self)
        act_aper_dlg.setStatusTip("Configure clear aperture, stops, slits, obscurations, and spider vanes")
        act_aper_dlg.triggered.connect(lambda checked, r=row_idx: self.on_open_aperture_dialog(r))
        menu_aper.addAction(act_aper_dlg)

        selected_surf = self.model._surfaces[row_idx] if 0 <= row_idx < len(self.model._surfaces) else None
        is_curr_stop = False
        if selected_surf:
            from optics_app.engine.element_colors import is_stop_plane
            is_curr_stop = is_stop_plane(selected_surf)

        if is_curr_stop:
            act_toggle_stop = QAction("⚪ Clear Stop Status (Set to Standard)", self)
            act_toggle_stop.triggered.connect(lambda checked, r=row_idx: self.model.set_stop_surface(r, False))
        else:
            act_toggle_stop = QAction("🔴 Set as System Aperture Stop (STO)", self)
            act_toggle_stop.triggered.connect(lambda checked, r=row_idx: self.model.set_stop_surface(r, True))
        menu_aper.addAction(act_toggle_stop)

        act_ins_stop = QAction(f"➕ Insert Aperture Stop Surface After Row {row_idx}...", self)
        act_ins_stop.triggered.connect(lambda checked, r=row_idx: self.on_insert_aperture_stop(r))
        menu_aper.addAction(act_ins_stop)
        menu.addSeparator()

        # Catalog Dialog Actions (Choose Parameters)
        if component_span is not None:
            act_rep_cat_dlg = QAction(f"Replace Lens from Thorlabs Catalog...", self)
            act_rep_cat_dlg.triggered.connect(lambda checked, r=row_idx: self.on_open_catalog_lens_dialog(target=r, mode="replace"))
            menu.addAction(act_rep_cat_dlg)

        act_ins_cat_dlg = QAction(f"Insert Thorlabs Catalog Lens After Row {row_idx}...", self)
        act_ins_cat_dlg.triggered.connect(lambda checked, r=row_idx: self.on_open_catalog_lens_dialog(target=r, mode="insert"))
        menu.addAction(act_ins_cat_dlg)
        menu.addSeparator()

        act_ins_zmx = QAction(f"Insert Zemax Element (.zmx) after Row {row_idx}...", self)
        act_ins_zmx.triggered.connect(lambda checked, r=row_idx: self.on_insert_zmx())
        menu.addAction(act_ins_zmx)

        if component_span is not None:
            act_rep_zmx = QAction("Replace Lens from Zemax File...", self)
            act_rep_zmx.triggered.connect(lambda checked, r=row_idx: self.on_replace_with_zmx(r))
            menu.addAction(act_rep_zmx)

        menu.addSeparator()

        selected_surface = self.model._surfaces[row_idx] if 0 <= row_idx < len(self.model._surfaces) else None
        if selected_surface is not None and selected_surface.is_reflective():
            menu_mirror = menu.addMenu("Mirror Actions")
            for label, angle, plane in (
                ("90° Fold (YZ)", 90.0, "YZ"),
                ("90° Fold (XZ)", 90.0, "XZ"),
                ("45° Fold (YZ)", 45.0, "YZ"),
                ("Retro-Reflector (180°)", 180.0, "YZ"),
            ):
                action = QAction(label, self)
                action.triggered.connect(
                    lambda checked, r=row_idx, a=angle, p=plane: self._on_menu_set_fold(r, a, p)
                )
                menu_mirror.addAction(action)
            act_aim_det = QAction("Aim at Detector", self)
            act_aim_det.triggered.connect(lambda checked, r=row_idx: self._on_menu_aim_mirror(r))
            menu_mirror.addAction(act_aim_det)

        if component_span is not None:
            act_rot_grp = QAction("Rotate Complete Lens...", self)
            act_rot_grp.triggered.connect(lambda checked, r=row_idx: self._on_menu_rotate_lens_element(r))
            menu.addAction(act_rot_grp)
            act_flip_lens = QAction(f"Flip Complete Lens '{component_span['name']}'", self)
            act_flip_lens.triggered.connect(
                lambda checked, r=component_span["start"]: self.model.flip_lens_element(r)
            )
            menu.addAction(act_flip_lens)

        menu.addSeparator()

        # Lens & Objective Grouping Submenu
        menu_group = menu.addMenu(f"Lens Element / Objective Grouping")
        act_group_named = QAction(f"Set Group Name for Surface {row_idx}...", self)
        act_group_named.triggered.connect(lambda checked, r=row_idx: self._on_menu_set_group_name(r))
        menu_group.addAction(act_group_named)

        act_group_sel = QAction("Group Selected Surfaces as Single Lens Element...", self)
        act_group_sel.triggered.connect(self._on_menu_group_selected_surfaces)
        menu_group.addAction(act_group_sel)

        act_clear_group = QAction(f"Clear Group from Surface {row_idx}", self)
        act_clear_group.triggered.connect(lambda checked, r=row_idx: self._on_menu_clear_group(r))
        menu_group.addAction(act_clear_group)
        
        global_pos = self.table_view.viewport().mapToGlobal(pos)
        execute_fn = getattr(menu, 'exec_', menu.exec)
        execute_fn(global_pos)

    def on_replace_with_zmx(self, row: int, file_path: Optional[str] = None) -> bool:
        """Replace lens element starting at surface row with imported Zemax .zmx file."""
        if file_path is None and QT_AVAILABLE:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Replace with Zemax Element", "", "Zemax Files (*.zmx);;All Files (*)"
            )
        if not file_path or not os.path.exists(file_path):
            return False

        try:
            stack = parse_zmx_file(file_path)
            current_stack = self.model.get_surface_stack()
            surfs = current_stack.surfaces

            if QT_AVAILABLE and QDialog is not None:
                dialog = ZemaxImportDialog(self, max_surfaces=len(surfs))
                dialog.index_spin.setValue(row)
                dialog.index_spin.setEnabled(False)
                dialog.setWindowTitle("Replace Zemax Element")
                exec_fn = getattr(dialog, 'exec_', dialog.exec)
                accepted_val = getattr(QDialog, 'Accepted', 1)
                if exec_fn() != accepted_val:
                    return False
                settings = dialog.get_settings()
            else:
                settings = {"insert_index": row, "reverse": False, "tilt_x": 0.0, "tilt_y": 0.0, "tilt_z": 0.0}

            processed = process_imported_surfaces(stack.surfaces, settings)
            if not processed:
                return False

            # Auto-detect surfaces to remove (singlet=2, doublet=3)
            n_remove = 1
            if row < len(surfs):
                glass1 = surfs[row].glass.upper().strip() if surfs[row].glass else ""
                if glass1 and glass1 not in ["AIR", "VACUUM"] and row + 1 < len(surfs):
                    glass2 = surfs[row + 1].glass.upper().strip() if surfs[row+1].glass else ""
                    if glass2 and glass2 not in ["AIR", "VACUUM"] and row + 2 < len(surfs):
                        n_remove = 3
                    else:
                        n_remove = 2

            old_gap = 10.0
            last_surf = row + n_remove - 1
            if 0 <= last_surf < len(surfs):
                old_gap = surfs[last_surf].thickness

            processed[-1].thickness = old_gap

            if hasattr(self.model, '_push_undo'):
                self.model._push_undo(f"Replace lens at surface {row} with Zemax file {os.path.basename(file_path)}")
            current_stack.replace_lens(row, processed, surfaces_to_remove=n_remove)
            self.model.set_surface_stack(current_stack)
            if hasattr(self.model, '_reindex_surface_names'):
                self.model._reindex_surface_names()
            if QT_AVAILABLE and hasattr(self.model, 'layoutChanged'):
                self.model.layoutChanged.emit()
            self.on_trace_rays()
            msg = f"Replaced lens at surface {row} with Zemax element: {os.path.basename(file_path)}"
            if QT_AVAILABLE:
                self.statusBar().showMessage(msg)
            else:
                self._status_msg = msg
            return True
        except Exception as e:
            if QT_AVAILABLE and hasattr(self, 'isVisible') and self.isVisible():
                QMessageBox.critical(self, "Error Replacing Lens", str(e))
            return False

    def _on_menu_set_fold(self, row: int, angle_deg: float, plane: str = "YZ"):
        try:
            if hasattr(self.model, "set_mirror_fold_angle"):
                self.model.set_mirror_fold_angle(row, angle_deg, plane)
        except ValueError as exc:
            self._status_msg = f"Mirror action rejected: {exc}"
            if QT_AVAILABLE:
                self.statusBar().showMessage(self._status_msg, 7000)
                QMessageBox.warning(self, "Invalid Mirror Action", str(exc))
            return
        if hasattr(self, "component_editor_dock") and self.component_editor_dock is not None:
            self.component_editor_dock.set_selected_surface(row)
        self.request_debounced_trace()

    def _on_menu_aim_mirror(self, row: int):
        if hasattr(self.model, "aim_mirror_to_target"):
            self.model.aim_mirror_to_target(row, -1)
        if hasattr(self, "component_editor_dock") and self.component_editor_dock is not None:
            self.component_editor_dock.set_selected_surface(row)
        self.request_debounced_trace()

    def _on_menu_rotate_lens_element(self, row: int):
        if hasattr(self.model, "rotate_lens_element"):
            self.model.rotate_lens_element(row, tilt_x=5.0)
        if hasattr(self, "component_editor_dock") and self.component_editor_dock is not None:
            self.component_editor_dock.set_selected_surface(row)
        self.request_debounced_trace()

    def _on_menu_set_group_name(self, row: int):
        if not QT_AVAILABLE or not (0 <= row < len(self.model._surfaces)):
            return
        from optics_app.gui.qt_compat import QtWidgets
        current = getattr(self.model._surfaces[row], 'group_name', '')
        name, ok = QtWidgets.QInputDialog.getText(
            self, "Set Lens Group Name", "Enter Objective / Lens Element Group Name:", text=current
        )
        if ok and name.strip():
            if hasattr(self.model, '_push_undo'):
                self.model._push_undo(f"Set group name on surface {row}")
            self.model._surfaces[row].group_name = name.strip()
            if hasattr(self.model, 'layoutChanged'):
                self.model.layoutChanged.emit()
            if hasattr(self.model, 'system_updated') and self.model.system_updated is not None:
                self.model.system_updated.emit()

    def _on_menu_group_selected_surfaces(self):
        if not QT_AVAILABLE:
            return
        from optics_app.gui.qt_compat import QtWidgets
        selected_rows = []
        if self.table_view:
            sel = self.table_view.selectionModel()
            if sel and sel.hasSelection():
                rows = sel.selectedRows()
                if rows:
                    selected_rows = [r.row() for r in rows]
                else:
                    indexes = sel.selectedIndexes()
                    if indexes:
                        selected_rows = sorted(list(set(idx.row() for idx in indexes)))

        if not selected_rows:
            if hasattr(self.table_view, "currentIndex"):
                c_idx = self.table_view.currentIndex()
                if c_idx and c_idx.isValid():
                    selected_rows = [c_idx.row()]

        if not selected_rows:
            if hasattr(self, "statusBar") and self.statusBar():
                self.statusBar().showMessage("Select surfaces in the table first to cluster as a lens.", 3000)
            return

        # Determine reasonable default name
        if len(selected_rows) == 3:
            default_name = "Doublet Lens"
        elif len(selected_rows) == 4:
            default_name = "Triplet Lens"
        elif len(selected_rows) > 1:
            default_name = f"Lens Element ({selected_rows[0]}-{selected_rows[-1]})"
        else:
            default_name = f"Lens {selected_rows[0]}"

        name, ok = QtWidgets.QInputDialog.getText(
            self, "Cluster Selected Surfaces as Lens", "Enter Lens Component / Group Name (e.g. 'Achromat Doublet'):", text=default_name
        )
        if ok and name.strip():
            cluster_name = name.strip()
            if hasattr(self.model, '_push_undo'):
                self.model._push_undo(f"Cluster surfaces {selected_rows} as {cluster_name}")

            # Assign matching element_id across the cluster
            existing_ids = [getattr(s, 'element_id', None) for s in self.model._surfaces if getattr(s, 'element_id', None) is not None]
            new_id = (max(existing_ids) + 1) if existing_ids else 0

            for r in selected_rows:
                if 0 <= r < len(self.model._surfaces):
                    self.model._surfaces[r].group_name = cluster_name
                    self.model._surfaces[r].element_id = new_id

            # Turn on component mode so LDE displays the merged cluster
            self._set_lde_component_name_mode(True)
            if hasattr(self, 'btn_toggle_comp_view') and self.btn_toggle_comp_view is not None:
                self.btn_toggle_comp_view.setChecked(True)
            if hasattr(self, 'act_comp_view') and self.act_comp_view is not None:
                self.act_comp_view.setChecked(True)

            if hasattr(self.model, 'layoutChanged'):
                self.model.layoutChanged.emit()
            if hasattr(self.model, 'system_updated') and self.model.system_updated is not None:
                self.model.system_updated.emit()
            if hasattr(self, 'component_editor_dock') and hasattr(self.component_editor_dock, '_rebuild_surface_list'):
                self.component_editor_dock._rebuild_surface_list()
            self.on_trace_rays()
            if hasattr(self, "statusBar") and self.statusBar():
                self.statusBar().showMessage(f"Clustered surfaces {selected_rows} as '{cluster_name}'", 3500)

    def _on_menu_clear_group(self, row: int):
        if 0 <= row < len(self.model._surfaces):
            g_name = getattr(self.model._surfaces[row], 'group_name', '')
            if hasattr(self.model, '_push_undo'):
                self.model._push_undo(f"Clear group on surface {row}")
            # If part of a cluster, clear all surfaces in this cluster
            for s in self.model._surfaces:
                if g_name and getattr(s, 'group_name', '') == g_name:
                    s.group_name = ""
                    s.element_id = None
            self.model._surfaces[row].group_name = ""
            self.model._surfaces[row].element_id = None
            if hasattr(self.model, 'layoutChanged'):
                self.model.layoutChanged.emit()
            if hasattr(self.model, 'system_updated') and self.model.system_updated is not None:
                self.model.system_updated.emit()
            if hasattr(self, 'component_editor_dock') and hasattr(self.component_editor_dock, '_rebuild_surface_list'):
                self.component_editor_dock._rebuild_surface_list()
            self.on_trace_rays()
            if hasattr(self, "statusBar") and self.statusBar():
                self.statusBar().showMessage(f"Unclustered surfaces from '{g_name or f'Surface {row}'}'", 3500)

    def _set_cell_variable(self, row: int, col: int, is_variable: bool):
        if not (0 <= row < len(self.model._surfaces)):
            return
        surf = self.model._surfaces[row]
        if hasattr(self.model, '_push_undo'):
            self.model._push_undo(f"Set solve on surface {row}")
        if col == 2:
            surf.radius_variable = is_variable
            surf.radius_solve_type = "Variable" if is_variable else "Fixed"
        elif col == 3:
            surf.thickness_variable = is_variable
            surf.thickness_solve_type = "Variable" if is_variable else "Fixed"
        elif col == 6:
            surf.conic_variable = is_variable
        elif col in (7, 8):
            setattr(surf, 'decenter_x_variable' if col == 7 else 'decenter_y_variable', is_variable)
        elif col in (9, 10, 11):
            attr = ['tilt_x_variable', 'tilt_y_variable', 'tilt_z_variable'][col - 9]
            setattr(surf, attr, is_variable)
        
        if hasattr(self.model, 'layoutChanged'):
            self.model.layoutChanged.emit()
        if hasattr(self.model, 'system_updated') and self.model.system_updated is not None:
            self.model.system_updated.emit()
        if hasattr(self, 'optimizer_dock') and hasattr(self.optimizer_dock, '_rebuild_surface_checkboxes'):
            self.optimizer_dock._rebuild_surface_checkboxes()
        if hasattr(self, "statusBar") and self.statusBar():
            status_str = "Variable (V)" if is_variable else "Fixed (F)"
            self.statusBar().showMessage(f"Surface {row} parameter set to {status_str}", 2500)

    def _set_selected_cells_solve(self, is_variable: bool):
        """Set all currently selected cells in LDE as Variable (V) or Fixed (F)."""
        if not self.table_view:
            return
        sel = self.table_view.selectionModel()
        indexes = sel.selectedIndexes() if sel and sel.hasSelection() else []
        if not indexes and hasattr(self.table_view, 'currentIndex'):
            c_idx = self.table_view.currentIndex()
            if c_idx and c_idx.isValid():
                indexes = [c_idx]
        if not indexes:
            return
        for idx in indexes:
            r = idx.row()
            c = idx.column()
            if c in (2, 3, 6, 7, 8, 9, 10, 11):
                self._set_cell_variable(r, c, is_variable)
            elif c in (0, 1): # If whole row selected, set radius & thickness
                self._set_cell_variable(r, 2, is_variable)
                self._set_cell_variable(r, 3, is_variable)

    def open_solve_dialog(self, row: int = None, col: int = None):
        """Open the Zemax OpticStudio-style Solve & Variable Dialog."""
        if not QT_AVAILABLE:
            return
        if row is None or col is None:
            if hasattr(self.table_view, 'currentIndex'):
                idx = self.table_view.currentIndex()
                if idx and idx.isValid():
                    row = idx.row()
                    col = idx.column()
        if row is None or col is None or not (0 <= row < len(self.model._surfaces)):
            row = 0
            col = 2
        
        try:
            from v2.gui.solve_dialog import ZemaxSolveDialog
        except ImportError:
            from optics_app.gui.solve_dialog import ZemaxSolveDialog
            
        dlg = ZemaxSolveDialog(surface_index=row, param_col=col, model=self.model, parent=self)
        exec_fn = getattr(dlg, 'exec_', dlg.exec)
        exec_fn()
        if hasattr(self, 'optimizer_dock') and hasattr(self.optimizer_dock, '_rebuild_surface_checkboxes'):
            self.optimizer_dock._rebuild_surface_checkboxes()
        self.on_trace_rays()

    def _setup_first_order_status_bar(self):
        """Set up live Zemax First Order Optical Data display in the status bar."""
        if not QT_AVAILABLE or not hasattr(self, "statusBar") or not self.statusBar():
            return
        self.lbl_status_first_order = QLabel("EFFL: -- mm | BFL: -- mm | F/#: -- | EPD: -- mm | EXPD: -- mm | P1: -- mm | P2: -- mm", self)
        self.lbl_status_first_order.setStyleSheet("font-weight: 500; color: #1a73e8; padding-right: 12px;")
        self.lbl_status_first_order.setToolTip("Click '📑 Zemax First Order...' in Tools menu or LDE toolbar for complete prescription report")
        self.statusBar().addPermanentWidget(self.lbl_status_first_order)

    def _update_live_first_order_status(self):
        """Update live paraxial first-order values in status bar."""
        if not hasattr(self, "lbl_status_first_order") or not self.lbl_status_first_order:
            return
        try:
            import math
            try:
                from v2.engine.first_order import calculate_first_order_data
            except ImportError:
                from optics_app.engine.first_order import calculate_first_order_data
            surfaces = getattr(self.model, "_surfaces", None)
            if surfaces is None and hasattr(self.model, "surface_stack"):
                surfaces = self.model.surface_stack.surfaces
            if surfaces:
                wl = float(getattr(self.system_config, "primary_wavelength_um", 0.5876)) if hasattr(self, "system_config") and self.system_config else 0.5876
                data = calculate_first_order_data(surfaces, wavelength_um=wl, config=getattr(self, "system_config", None))
                effl = data.get("effl", float("inf"))
                bfl = data.get("bfl", float("inf"))
                f_num = data.get("f_number", float("inf"))
                epd = data.get("epd", 0.0)
                expd = data.get("expd", 0.0)
                p1 = data.get("p1", 0.0)
                p2 = data.get("p2", 0.0)
                totr = data.get("totr", 0.0)
                is_afocal = data.get("is_afocal", False)
                if math.isinf(effl) or is_afocal:
                    self.lbl_status_first_order.setText(f"EFFL: Afocal | EPD: {epd:.1f} mm | EXPD: {expd:.1f} mm | TOTR: {totr:.1f} mm")
                else:
                    self.lbl_status_first_order.setText(
                        f"EFFL: {effl:.2f} mm | BFL: {bfl:.2f} mm | F/#{f_num:.2f} | "
                        f"EPD: {epd:.1f} mm | EXPD: {expd:.1f} mm | P1: {p1:+.2f} mm | P2: {p2:+.2f} mm"
                    )
        except Exception:
            pass

    def on_show_first_order_data(self):
        """Open the Zemax OpticStudio First Order Optical Data Dialog."""
        if not QT_AVAILABLE:
            return
        try:
            from v2.gui.first_order_dialog import ZemaxFirstOrderDataDialog
        except ImportError:
            from optics_app.gui.first_order_dialog import ZemaxFirstOrderDataDialog
        dlg = ZemaxFirstOrderDataDialog(self.model.get_surface_stack(), getattr(self, "system_config", None), parent=self)
        exec_fn = getattr(dlg, "exec_", dlg.exec)
        exec_fn()

    def on_show_spectrometer_quality(self):
        """Open the Zemax Spectrometer Quality & Resolution tab in the analysis dock."""
        if not QT_AVAILABLE:
            return
        if hasattr(self, 'analysis_dock') and self.analysis_dock is not None:
            self.analysis_dock.show()
            self.analysis_dock.raise_()
        if hasattr(self, 'analysis_tabs') and self.analysis_tabs is not None:
            self.analysis_tabs.set_active_tab("Spectrometer Quality & Resolution (Zemax)")
            if hasattr(self.analysis_tabs, 'refresh_analysis'):
                self.analysis_tabs.refresh_analysis()

    def _apply_zemax_defaults(self):
        if hasattr(self.model, '_push_undo'):
            self.model._push_undo("Apply Zemax Defaults (Air Gaps Variable)")
        if hasattr(self.model, 'set_zemax_defaults'):
            self.model.set_zemax_defaults(var_air_gaps=True, fix_lenses=True)
        if hasattr(self, 'optimizer_dock') and hasattr(self.optimizer_dock, '_rebuild_surface_checkboxes'):
            self.optimizer_dock._rebuild_surface_checkboxes()
        if hasattr(self, "statusBar") and self.statusBar():
            self.statusBar().showMessage("Applied Zemax Defaults: Air Gaps Variable, Lenses Fixed", 3000)

    def _apply_all_radii_variable(self):
        if hasattr(self.model, '_push_undo'):
            self.model._push_undo("Make All Radii Variable")
        if hasattr(self.model, 'make_all_radii_variable'):
            self.model.make_all_radii_variable()
        if hasattr(self, 'optimizer_dock') and hasattr(self.optimizer_dock, '_rebuild_surface_checkboxes'):
            self.optimizer_dock._rebuild_surface_checkboxes()
        if hasattr(self, "statusBar") and self.statusBar():
            self.statusBar().showMessage("Set all surface radii as Variable for curvature optimization", 3000)

    def _apply_all_air_gaps_variable(self):
        if hasattr(self.model, '_push_undo'):
            self.model._push_undo("Make All Air Gaps Variable")
        if hasattr(self.model, 'make_all_air_gaps_variable'):
            self.model.make_all_air_gaps_variable()
        if hasattr(self, 'optimizer_dock') and hasattr(self.optimizer_dock, '_rebuild_surface_checkboxes'):
            self.optimizer_dock._rebuild_surface_checkboxes()
        if hasattr(self, "statusBar") and self.statusBar():
            self.statusBar().showMessage("Set all air gaps as Variable for distance optimization", 3000)

    def _apply_all_fixed(self):
        if hasattr(self.model, '_push_undo'):
            self.model._push_undo("Set All Parameters Fixed")
        if hasattr(self.model, 'set_all_fixed'):
            self.model.set_all_fixed()
        if hasattr(self, 'optimizer_dock') and hasattr(self.optimizer_dock, '_rebuild_surface_checkboxes'):
            self.optimizer_dock._rebuild_surface_checkboxes()
        if hasattr(self, "statusBar") and self.statusBar():
            self.statusBar().showMessage("Cleared all variables: All parameters set to Fixed", 3000)

    def on_move_surface_up(self, position: Optional[int] = None) -> bool:
        """Move surface row up in Lens Data Editor."""
        if position is None and self.table_view and hasattr(self.table_view, "currentIndex"):
            idx = self.table_view.currentIndex()
            if idx and hasattr(idx, "row") and idx.row() >= 0:
                position = idx.row()
        if position is None:
            position = 1
        success = self.model.move_surface_up(position)
        if success:
            msg = f"Moved surface row {position} up"
            if QT_AVAILABLE:
                self.statusBar().showMessage(msg)
            else:
                self._status_msg = msg
            self.on_trace_rays()
        return success

    def on_move_surface_down(self, position: Optional[int] = None) -> bool:
        """Move surface row down in Lens Data Editor."""
        if position is None and self.table_view and hasattr(self.table_view, "currentIndex"):
            idx = self.table_view.currentIndex()
            if idx and hasattr(idx, "row") and idx.row() >= 0:
                position = idx.row()
        if position is None:
            position = 1
        success = self.model.move_surface_down(position)
        if success:
            msg = f"Moved surface row {position} down"
            if QT_AVAILABLE:
                self.statusBar().showMessage(msg)
            else:
                self._status_msg = msg
            self.on_trace_rays()
        return success

    def get_model(self) -> LensDataTableModel:
        """Return the LensDataTableModel instance."""
        return self.model

    def get_dual_view(self) -> DualViewWidget:
        """Return the DualViewWidget instance."""
        return self.dual_view

    def get_analysis_tabs(self) -> AnalysisTabsWidget:
        """Return the AnalysisTabsWidget instance."""
        return self.analysis_tabs

    def get_optimizer_dock(self) -> OptimizerDockWidget:
        """Return the OptimizerDockWidget instance."""
        return self.optimizer_dock

    def get_component_editor_dock(self) -> ComponentEditorDockWidget:
        """Return the ComponentEditorDockWidget instance."""
        return getattr(self, 'component_editor_dock', None)

    def get_current_surface_stack(self):
        """Return the active surface stack from the optical system model."""
        if hasattr(self, "model") and self.model is not None and hasattr(self.model, "get_surface_stack"):
            return self.model.get_surface_stack()
        return None

    def request_debounced_trace(self, *args):
        """Request a debounced ray trace and canvas update (20-30 ms)."""
        if hasattr(self, '_trace_debounce_timer') and self._trace_debounce_timer is not None:
            self._trace_debounce_timer.start(25)
        else:
            self.on_trace_rays()

    def _on_debounced_trace_timeout(self):
        """Execute debounced ray trace and refresh dual view and active analysis tab."""
        self.on_trace_rays()

    def _on_table_cell_clicked(self, index):
        """Slot triggered when user clicks a table cell."""
        if index and hasattr(index, "row") and index.row() >= 0:
            self._update_selected_surface_highlight(self._select_complete_lens(index.row()))

    def _on_table_current_changed(self, current, previous):
        """Slot triggered when keyboard navigation changes active row/cell."""
        if current and hasattr(current, "row") and current.row() >= 0:
            self._update_selected_surface_highlight(self._select_complete_lens(current.row()))

    def _on_table_selection_changed(self, selected=None, deselected=None):
        """Sync ComponentEditorDockWidget and highlight selected surface on 2D longitudinal axis."""
        row_idx = None
        if self.table_view:
            c_idx = self.table_view.currentIndex()
            if c_idx and hasattr(c_idx, "row") and c_idx.row() >= 0:
                row_idx = c_idx.row()
            else:
                sel = self.table_view.selectionModel()
                if sel and sel.hasSelection():
                    rows = sel.selectedRows()
                    if rows:
                        row_idx = rows[0].row()
                    else:
                        indexes = sel.selectedIndexes()
                        if indexes:
                            row_idx = indexes[0].row()

        if row_idx is not None:
            self._update_selected_surface_highlight(self._select_complete_lens(row_idx))

    def _select_complete_lens(self, row_idx: int) -> int:
        """In component mode, select and target every row of the clicked lens."""
        if getattr(self, '_expanding_component_selection', False) or not self.model.component_name_mode():
            return row_idx
        span = self.model.component_span_for_row(row_idx)
        if span is None:
            return row_idx
        if self.table_view is None or not QT_AVAILABLE:
            return span["start"]
        try:
            from optics_app.gui import qt_compat
            qsm = qt_compat.QtCore.QItemSelectionModel
            flags_type = getattr(qsm, 'SelectionFlag', qsm)
            flags = getattr(flags_type, 'ClearAndSelect') | getattr(flags_type, 'Rows')
            selection = qt_compat.QtCore.QItemSelection(
                self.model.index(span["start"], 0),
                self.model.index(span["end"], self.model.columnCount() - 1),
            )
            self._expanding_component_selection = True
            self.table_view.selectionModel().select(selection, flags)
        except Exception:
            pass
        finally:
            self._expanding_component_selection = False
        return span["start"]

    def _update_selected_surface_highlight(self, row_idx: int):
        """Broadcast selected surface index to Component Editor and 2D Dual View."""
        if hasattr(self, 'component_editor_dock') and self.component_editor_dock is not None:
            self.component_editor_dock.set_selected_surface(row_idx)
        if hasattr(self, 'dual_view') and self.dual_view is not None:
            self.dual_view.set_selected_surface(row_idx)

    def on_add_surface(self, position: Optional[int] = None) -> bool:
        """Slot alias for adding/inserting a surface row."""
        return self.on_insert_surface(position)

    def on_remove_surface(self, position: Optional[int] = None) -> bool:
        """Slot alias for removing/deleting a surface row."""
        return self.on_delete_surface(position)

    def update_all_views(self, action_name: str = "AI Action", record_history: bool = True):
        """Force complete UI refresh across all docks and editors after programmatic changes (e.g. AI updates)."""
        if QT_AVAILABLE and hasattr(self.model, 'layoutChanged'):
            self.model.layoutChanged.emit()
            
        if hasattr(self, 'system_config_dock') and self.system_config_dock:
            self.system_config_dock.update_ui()
            
        if hasattr(self, 'source_editor') and self.source_editor:
            self.source_editor.update_ui()
            
        if hasattr(self, 'component_editor_dock') and self.component_editor_dock:
            self.component_editor_dock.update_ui()
            
        if hasattr(self, 'optimizer_dock') and self.optimizer_dock:
            if hasattr(self.optimizer_dock, 'update_ui'):
                self.optimizer_dock.update_ui()
                
        if hasattr(self, 'ultrafast_dock') and self.ultrafast_dock:
            if hasattr(self.ultrafast_dock, 'update_ui'):
                self.ultrafast_dock.update_ui()
                
        self.on_trace_rays()
        
        if record_history and hasattr(self, 'undo_manager') and self.undo_manager is not None:
            stack = self.model.get_surface_stack()
            self.undo_manager.push_state(stack, getattr(self, 'system_config', None), action_name)
            self._update_undo_redo_actions()

    def on_trace_rays(self, checked: bool = False) -> bool:
        """Trigger ray tracing and update dual view & analysis diagnostics."""
        stack = self.model.get_surface_stack()
        if stack and len(stack.surfaces) > 0:
            import numpy as np
            from optics_app.engine.ray import Ray
            system = OpticalSystem(stack)

            # Read wavelengths from SystemConfig, then fallback
            wavelengths = None
            if hasattr(self, 'system_config'):
                wavelengths = self.system_config.wavelength_list_um
                
            if not wavelengths:
                wavelengths = [0.450, 0.546, 0.650]
                    
            # Set the primary wavelength to perfectly align grating layout coordinates
            stack.primary_wavelength = wavelengths[0]
            stack._update_accumulated_positions()

            traced_rays = []

            # Check if this stack represents a non-sequential optical system
            if getattr(stack, 'is_non_sequential', False) and hasattr(stack, 'non_sequential_system'):
                ns_sys = stack.non_sequential_system
                traced_rays = ns_sys.get_combined_rays_list(num_rays_per_source=7)
            else:
                # Generate rays based on SystemConfig Fields, Aperture & Beam Mode
                sys_cfg = getattr(self, 'system_config', None)
                from optics_app.gui.visualizer_3d import classify_surface_for_3d, generate_3d_rays
                source_is_modeled = any(
                    classify_surface_for_3d(surf, i, len(stack.surfaces))["object"]
                    for i, surf in enumerate(stack.surfaces)
                )
                if not source_is_modeled:
                    traced_rays = generate_3d_rays(stack, system_config=sys_cfg)
                elif sys_cfg is not None and hasattr(sys_cfg, 'generate_input_rays'):
                    n_rays = getattr(sys_cfg, 'num_rays', 9)
                    input_rays = sys_cfg.generate_input_rays(surfaces=stack.surfaces, wavelengths=wavelengths, num_rays=n_rays, system=system)
                    for r in input_rays:
                        traced = system.trace_ray(r)
                        traced_rays.append(traced)
                else:
                    d0 = stack.surfaces[0].thickness if len(stack.surfaces) > 0 else 0.0
                    r1 = stack.surfaces[1].semi_diameter if len(stack.surfaces) > 1 else 10.0
                    for wl in wavelengths:
                        for py in np.linspace(-1.0, 1.0, 7):
                            if d0 > 1e4 or d0 <= 0:
                                start_pos = np.array([0.0, py * r1, 0.0])
                                start_dir = np.array([0.0, 0.0, 1.0])
                            else:
                                start_pos = np.array([0.0, 0.0, 0.0])
                                target_y = py * r1
                                target_z = d0
                                length = np.sqrt(target_y**2 + target_z**2)
                                start_dir = np.array([0.0, target_y / length, target_z / length])
                            r = Ray(pos=start_pos, dir=start_dir, wavelength=wl)
                            r.field_idx = 0
                            r.pupil_coord = (0.0, float(py))
                            r.is_chief = False
                            r.is_marginal = bool(abs(abs(py) - 1.0) < 1e-5)
                            r.is_upper_marginal = bool(r.is_marginal and py > 0)
                            r.is_lower_marginal = bool(r.is_marginal and py < 0)
                            traced = system.trace_ray(r)
                            traced_rays.append(traced)

                        # Trace canonical off-axis Chief Ray through pupil center (Px=0, Py=0)
                        theta_c = math.radians(1.0)
                        if d0 > 1e4 or d0 <= 0:
                            start_pos_c = np.array([0.0, 0.0, 0.0])
                            start_dir_c = np.array([0.0, math.tan(theta_c), 1.0])
                            start_dir_c /= np.linalg.norm(start_dir_c)
                        else:
                            obj_h_c = max(1.0, r1 * 0.25)
                            start_pos_c = np.array([0.0, obj_h_c, 0.0])
                            target_c = np.array([0.0, 0.0, d0])
                            dir_c = target_c - start_pos_c
                            start_dir_c = dir_c / np.linalg.norm(dir_c)
                        r_chief = Ray(pos=start_pos_c, dir=start_dir_c, wavelength=wl)
                        r_chief.field_idx = 1
                        r_chief.pupil_coord = (0.0, 0.0)
                        r_chief.is_chief = True
                        r_chief.is_marginal = False
                        traced_c = system.trace_ray(r_chief)
                        traced_rays.append(traced_c)

            self._last_traced_rays = traced_rays
            if hasattr(self.model, "set_rays"):
                self.model.set_rays(traced_rays)

            if hasattr(self.dual_view, "update_layout"):
                self.dual_view.update_layout(surface_stack=stack, rays=traced_rays)
            elif hasattr(self.dual_view, "refresh_layout"):
                self.dual_view.refresh_layout()

            if hasattr(self, "visualizer_3d") and self.visualizer_3d is not None:
                self.visualizer_3d.surface_stack = stack
                self.visualizer_3d.rays = traced_rays
                viewer_tabs = getattr(self, "viewer_tabs", None)
                is_3d_active = viewer_tabs is None or viewer_tabs.currentWidget() is self.visualizer_3d
                if is_3d_active and hasattr(self.visualizer_3d, "update_layout"):
                    self.visualizer_3d.update_layout(surface_stack=stack, rays=traced_rays)

            if hasattr(self.analysis_tabs, "refresh_analysis"):
                self.analysis_tabs.refresh_analysis(wavelengths=wavelengths, source_config=None)

            if hasattr(self, 'plm_dock') and self.plm_dock is not None and hasattr(self.plm_dock, 'isVisible') and self.plm_dock.isVisible():
                if getattr(getattr(self.plm_dock, 'studio', None), 'stack', None) is not stack:
                    self.plm_dock.set_surface_stack(stack)

            self._sync_mech_action_ui()

            from optics_app.engine.diagnostics import ModelSanityValidator
            geometry_report = ModelSanityValidator.inspect_geometry(stack)
            trace_report = ModelSanityValidator.inspect_trace(stack, traced_rays)
            diagnostic_items = geometry_report.items + trace_report.items
        else:
            self._last_traced_rays = []
            if hasattr(self.model, "set_rays"):
                self.model.set_rays([])
            self.dual_view.refresh_layout()
            if hasattr(self, "visualizer_3d") and self.visualizer_3d is not None:
                self.visualizer_3d.surface_stack = None
                self.visualizer_3d.rays = []
                viewer_tabs = getattr(self, "viewer_tabs", None)
                if viewer_tabs is None or viewer_tabs.currentWidget() is self.visualizer_3d:
                    self.visualizer_3d.update_layout(rays=[])
            self.analysis_tabs.refresh_analysis(source_config=None)
            self._sync_mech_action_ui()
            diagnostic_items = []

        self._last_model_warnings = diagnostic_items
        warning_signature = tuple((item.code, item.surface_index, item.message) for item in diagnostic_items)
        if diagnostic_items:
            first = diagnostic_items[0]
            msg = f"Ray trace warning: {first.message}"
            if warning_signature != getattr(self, "_last_model_warning_signature", None):
                self._last_model_warning_signature = warning_signature
                if QT_AVAILABLE and hasattr(self, "isVisible") and self.isVisible():
                    # Suppress modal popup for expected operational states like DMD OFF
                    if not any(item.code == "DMD_OFF_NO_DETECTOR_SIGNAL_EXPECTED" for item in diagnostic_items):
                        details = "\n\n".join(
                            f"{item.message}\nSuggested action: {item.suggested_fix}"
                            if item.suggested_fix else item.message
                            for item in diagnostic_items[:5]
                        )
                        QMessageBox.warning(self, "Optical Model Sanity Warning", details)
        else:
            self._last_model_warning_signature = None
            msg = "Ray trace complete: rays propagated through surface stack"
        self._status_msg = msg
        if QT_AVAILABLE:
            self.statusBar().showMessage(msg)
            self._update_live_first_order_status()
        return True

    def _on_viewer_tab_changed(self, index: int):
        """Update 3D or 2D views whenever the user switches viewer tabs."""
        if not hasattr(self, "viewer_tabs") or not hasattr(self, "visualizer_3d") or self.visualizer_3d is None:
            return
        tab_text = self.viewer_tabs.tabText(index)
        if "3D" in tab_text:
            stack = self.model.get_surface_stack() if hasattr(self.model, "get_surface_stack") else None
            rays = getattr(self, "_last_traced_rays", None)
            if stack is not None and (not rays or len(rays) == 0):
                self.on_trace_rays()
            elif hasattr(self.visualizer_3d, "update_layout"):
                self.visualizer_3d.update_layout(surface_stack=stack, rays=rays)

    def on_refresh_all(self, checked: bool = False) -> bool:
        """Refresh all GUI layout elements and diagnostic plots."""
        return self.on_trace_rays()

    def _on_open_system_explorer_tab(self, tab_index: int = 0):
        """Bring System Explorer dock into view and select the requested configuration tab."""
        if hasattr(self, 'system_config_dock') and self.system_config_dock is not None:
            try:
                if hasattr(self.system_config_dock, 'show'):
                    self.system_config_dock.show()
                if hasattr(self.system_config_dock, 'raise_'):
                    self.system_config_dock.raise_()
                if hasattr(self.system_config_dock, 'tabs') and self.system_config_dock.tabs is not None:
                    if hasattr(self.system_config_dock.tabs, 'setCurrentIndex'):
                        self.system_config_dock.tabs.setCurrentIndex(tab_index)
            except Exception:
                pass

    def _abort_optimization(self):
        """Slot to abort the currently running optical optimization."""
        if hasattr(self, 'active_optimizer') and self.active_optimizer is not None:
            if hasattr(self.active_optimizer, 'stop'):
                self.active_optimizer.stop()
                if QT_AVAILABLE:
                    self.statusBar().showMessage("Aborting Optimization... please wait.")

    def on_optimize(
        self,
        variable_indices: Optional[dict] = None,
        method: str = "Nelder-Mead",
        max_iter: int = 100,
        target: str = "Spot Size (Focus)",
        num_rays: int = 21,
        target_spread: float = 12.0,
        opt_tilt: bool = False,
        target_surface_index: int = -1,
        target_spot_size: float = 0.0,
        target_efl: Optional[float] = None,
        efl_weight: float = 50.0,
        blocking: bool = True
    ) -> OptimizationResult:
        """
        Execute optical optimization on current optical system.
        Adjusts variable surface thicknesses to minimize RMS spot radius on detector plane.
        Updates model, ray tracing layout, dual view CAD, and analysis spot diagrams live.
        """
        stack = self.model.get_surface_stack()
        if not stack or len(stack.surfaces) == 0:
            return OptimizationResult(
                success=False,
                message="No surfaces available in optical system"
            )

        # 1. Determine variable surface indices
        if variable_indices is None:
            if hasattr(self, 'optimizer_dock') and self.optimizer_dock is not None:
                dock_indices = self.optimizer_dock.get_variable_indices()
                if dock_indices:
                    variable_indices = dock_indices
            if not variable_indices:
                explicit_vars = [
                    i for i, s in enumerate(stack.surfaces)
                    if getattr(s, 'thickness_variable', False)
                ]
                if explicit_vars:
                    variable_indices = explicit_vars
                else:
                    variable_indices = [
                        i for i, s in enumerate(stack.surfaces)
                        if s.surface_type != SurfaceType.DETECTOR and (not s.glass or s.glass.upper() in ["AIR", "VACUUM", "MIRROR"])
                    ]

        # 2. Extract wavelengths
        wavelengths = [0.5876]
        # Priority 1: Optimizer Dock wavelengths
        if hasattr(self, 'optimizer_dock') and self.optimizer_dock is not None:
            if hasattr(self.optimizer_dock, 'get_wavelengths'):
                dock_wls = self.optimizer_dock.get_wavelengths()
                if dock_wls:
                    wavelengths = dock_wls
        # Priority 2: System Config Explorer wavelengths
        if wavelengths == [0.5876]:
            sys_cfg = getattr(self, 'system_config', None)
            if sys_cfg and getattr(sys_cfg, 'wavelengths', None):
                try:
                    parsed_wls = []
                    for w in sys_cfg.wavelengths:
                        val = float(getattr(w, 'wavelength_um', w))
                        if val > 10.0:
                            val /= 1000.0
                        if val > 0:
                            parsed_wls.append(val)
                    if parsed_wls:
                        wavelengths = parsed_wls
                except Exception:
                    pass
        # Priority 3: Legacy text input / Source Editor fallback
        if wavelengths == [0.5876] and hasattr(self, 'wavelengths_input') and self.wavelengths_input is not None and hasattr(self.wavelengths_input, 'text'):
            txt = self.wavelengths_input.text().strip()
            if txt:
                try:
                    parsed = [float(w.strip()) for w in txt.split(',') if w.strip()]
                    w_list = [w / 1000.0 if w > 10.0 else w for w in parsed if w > 0]
                    if w_list:
                        wavelengths = w_list
                except ValueError:
                    pass
        elif wavelengths == [0.5876] and hasattr(self, 'source_editor') and self.source_editor is not None:
            if hasattr(self.source_editor, 'get_source_config'):
                config = self.source_editor.get_source_config()
                if isinstance(config, dict) and "wavelengths" in config:
                    wavelengths = config["wavelengths"]

        # 3. Initialize OpticalOptimizer with manufacturing constraints
        mfg_constraints = {}
        if hasattr(self, 'optimizer_dock') and self.optimizer_dock is not None:
            if hasattr(self.optimizer_dock, 'get_manufacturing_constraints'):
                try:
                    mfg_constraints = self.optimizer_dock.get_manufacturing_constraints()
                except Exception:
                    mfg_constraints = {}

        system = OpticalSystem(stack)
        optimizer = OpticalOptimizer(
            system=system,
            wavelengths=wavelengths,
            method=method,
            max_iter=max_iter,
            target=target,
            num_rays=num_rays,
            target_spread=target_spread,
            opt_tilt=opt_tilt,
            target_surface_index=target_surface_index if target_surface_index >= 0 else None,
            target_spot_size=target_spot_size,
            target_efl=target_efl,
            efl_weight=efl_weight,
            fields=getattr(self, 'system_config', None).fields if getattr(self, 'system_config', None) else None,
            system_config=getattr(self, 'system_config', None),
            manufacturing_constraints=mfg_constraints
        )


        is_headless = os.environ.get("QT_QPA_PLATFORM") == "offscreen" or "pytest" in sys.modules or "unittest" in sys.modules
        progress_dialog = None
        if QT_AVAILABLE and hasattr(QtWidgets, 'QProgressDialog') and not is_headless:
            try:
                progress_dialog = QtWidgets.QProgressDialog("Running Optimization...", "Stop", 0, max_iter, self)
                progress_dialog.setWindowTitle("Optimizer")
                progress_dialog.setWindowModality(Qt.WindowModal)
                progress_dialog.setMinimumDuration(0)
                progress_dialog.setValue(0)
            except Exception:
                progress_dialog = None

        def _on_optimizer_progress(current: Optional[int] = None, maximum: Optional[int] = None):
            if current is not None and maximum is not None and maximum > 0:
                pct = int(min(100, max(0, (current / maximum) * 100)))
                if hasattr(self, 'optimizer_dock') and self.optimizer_dock is not None:
                    self.optimizer_dock.set_progress(pct)
                    self.optimizer_dock.set_status(f"Optimizing... (iter {current}/{maximum})")
                if QT_AVAILABLE:
                    if hasattr(self, 'statusBar') and self.statusBar() is not None:
                        try:
                            self.statusBar().showMessage(f"Running Optimization... (Iteration {current}/{maximum})")
                        except Exception:
                            pass
                if progress_dialog is not None:
                    try:
                        progress_dialog.setValue(current)
                        if hasattr(progress_dialog, 'wasCanceled') and progress_dialog.wasCanceled():
                            optimizer.abort = True
                    except Exception:
                        pass
            if QT_AVAILABLE and hasattr(QtWidgets, 'QApplication') and not is_headless:
                try:
                    QtWidgets.QApplication.processEvents()
                except Exception:
                    pass

        optimizer.progress_callback = _on_optimizer_progress
        self.active_optimizer = optimizer

        # Update dock status
        if hasattr(self, 'optimizer_dock') and self.optimizer_dock is not None:
            self.optimizer_dock.set_status("Optimizing...")
            self.optimizer_dock.set_progress(25)

        # 4. Execute optimization
        try:
            result = optimizer.optimize(variable_indices=variable_indices)

            # 5. Handle result and live synchronization
            applied = False
            if result.optimized_stack is not None:
                if hasattr(self.model, '_push_undo'):
                    self.model._push_undo("Optimization result")
                self.model.set_surface_stack(result.optimized_stack, emit_signal=False)
                applied = True
                
                if hasattr(self, 'optimizer_dock') and self.optimizer_dock is not None:
                    self.optimizer_dock.last_result = result
                    if hasattr(self.optimizer_dock, '_rebuild_surface_checkboxes'):
                        self.optimizer_dock._rebuild_surface_checkboxes()
                    if hasattr(self.optimizer_dock, 'update_system_info'):
                        self.optimizer_dock.update_system_info()

                cfg = getattr(self, 'source_config', None)
                if hasattr(self.analysis_tabs, "refresh_analysis"):
                    self.analysis_tabs.refresh_analysis(wavelengths=wavelengths, source_config=None)
                self.on_trace_rays()

                if hasattr(self, 'table_view') and self.table_view is not None:
                    try:
                        self.table_view.viewport().update()
                    except Exception:
                        pass

            if result.success or (result.final_rms < result.initial_rms and result.final_rms >= 0.0) or (applied and target_efl is not None):
                msg = f"Optimization finished: RMS {result.initial_rms:.3f} -> {result.final_rms:.3f} ({result.rms_reduction_pct:.1f}% reduction)"
                self._status_msg = msg
                if QT_AVAILABLE and hasattr(self, 'statusBar') and self.statusBar() is not None:
                    try:
                        self.statusBar().showMessage(msg)
                    except Exception:
                        pass

                if hasattr(self, 'optimizer_dock') and self.optimizer_dock is not None:
                    self.optimizer_dock.last_result = result
                    self.optimizer_dock.set_status(f"RMS: {result.initial_rms:.2f}->{result.final_rms:.2f} (-{result.rms_reduction_pct:.1f}%)")
                    self.optimizer_dock.set_progress(100)
            else:
                msg = f"Optimization completed: {result.message}. Candidate applied: RMS {result.final_rms:.3f}"
                self._status_msg = msg
                if QT_AVAILABLE and hasattr(self, 'statusBar') and self.statusBar() is not None:
                    try:
                        self.statusBar().showMessage(msg)
                    except Exception:
                        pass

                if hasattr(self, 'optimizer_dock') and self.optimizer_dock is not None:
                    self.optimizer_dock.last_result = result
                    self.optimizer_dock.set_status(f"Finished: RMS {result.final_rms:.2f}")
                    self.optimizer_dock.set_progress(100 if applied else 0)

            return result
        finally:
            if progress_dialog is not None:
                try:
                    progress_dialog.close()
                except Exception:
                    pass

            self.active_optimizer = None

            if hasattr(self, 'optimizer_dock') and self.optimizer_dock is not None:
                if hasattr(self.optimizer_dock, 'on_optimization_finished'):
                    try:
                        self.optimizer_dock.on_optimization_finished(result if 'result' in locals() else None)
                    except Exception:
                        pass
                if hasattr(self.optimizer_dock, '_rebuild_surface_checkboxes'):
                    try:
                        self.optimizer_dock._rebuild_surface_checkboxes()
                    except Exception:
                        pass
                if hasattr(self.optimizer_dock, 'update_system_info'):
                    try:
                        self.optimizer_dock.update_system_info()
                    except Exception:
                        pass

    def on_optimize_system(self):
        """Slot invoked from menu bar to trigger optical optimization."""
        try:
            method = "Nelder-Mead"
            max_iter = 100
            target = "Spot Size (Focus)"
            rays = 21
            var_indices = {"thickness": [], "radius": []}
            target_spread = 12.0
            opt_tilt = False
            target_surf_idx = -1
            if hasattr(self, 'optimizer_dock') and self.optimizer_dock is not None:
                var_indices = self.optimizer_dock.get_variable_indices()
                if hasattr(self.optimizer_dock, 'method_combo') and hasattr(self.optimizer_dock.method_combo, 'currentText'):
                    method = self.optimizer_dock.method_combo.currentText()
                if hasattr(self.optimizer_dock, 'max_iter_spin') and hasattr(self.optimizer_dock.max_iter_spin, 'value'):
                    max_iter = self.optimizer_dock.max_iter_spin.value()
                if hasattr(self.optimizer_dock, 'rays_spin') and hasattr(self.optimizer_dock.rays_spin, 'value'):
                    rays = self.optimizer_dock.rays_spin.value()
                if hasattr(self.optimizer_dock, 'target_combo') and hasattr(self.optimizer_dock.target_combo, 'currentText'):
                    target = self.optimizer_dock.target_combo.currentText()
                if hasattr(self.optimizer_dock, 'target_spread_spin') and hasattr(self.optimizer_dock.target_spread_spin, 'value'):
                    target_spread = self.optimizer_dock.target_spread_spin.value()
                if hasattr(self.optimizer_dock, 'opt_tilt_check'):
                    opt_tilt = self.optimizer_dock.opt_tilt_check.isChecked()
                if hasattr(self.optimizer_dock, 'target_surf_combo'):
                    target_surf_idx = self.optimizer_dock.target_surf_combo.currentData()
            
            target_efl = None
            if hasattr(self, 'optimizer_dock') and self.optimizer_dock is not None:
                if hasattr(self.optimizer_dock, 'chk_constrain_efl') and (self.optimizer_dock.chk_constrain_efl.isChecked() or "Constrain EFL" in target):
                    target_efl = float(self.optimizer_dock.target_efl_spin.value())
            res = self.on_optimize(variable_indices=var_indices, method=method, max_iter=max_iter, target=target, num_rays=rays, target_spread=target_spread, opt_tilt=opt_tilt, target_surface_index=target_surf_idx, target_efl=target_efl)

            if res.success or res.final_rms < res.initial_rms:
                msg = f"Optimization finished: RMS {res.initial_rms:.3f} -> {res.final_rms:.3f} ({res.rms_reduction_pct:.1f}% reduction)"
                if QT_AVAILABLE:
                    QMessageBox.information(self, "Optimization Complete", msg)
            else:
                if QT_AVAILABLE:
                    QMessageBox.warning(self, "Optimization Failed", res.message)
        except Exception as e:
            if QT_AVAILABLE:
                QMessageBox.critical(self, "Optimizer Error", f"Failed to run optimizer: {e}")

    def on_fit_test_plates(self):
        """Fit system spherical surfaces to standard ISO/Thorlabs/Edmund tooling test plates."""
        try:
            stack = None
            if hasattr(self.model, "get_surface_stack"):
                stack = self.model.get_surface_stack()
            elif hasattr(self.model, "stack"):
                stack = self.model.stack
            if not stack or not getattr(stack, "surfaces", []):
                return

            wavelengths = [0.5876]
            if hasattr(self, 'source_editor') and self.source_editor is not None:
                if hasattr(self.source_editor, 'get_source_config'):
                    config = self.source_editor.get_source_config()
                    if isinstance(config, dict) and "wavelengths" in config:
                        wavelengths = config["wavelengths"]

            mfg_constraints = {}
            if hasattr(self, 'optimizer_dock') and self.optimizer_dock is not None:
                if hasattr(self.optimizer_dock, 'get_manufacturing_constraints'):
                    try:
                        mfg_constraints = self.optimizer_dock.get_manufacturing_constraints()
                    except Exception:
                        pass

            system = OpticalSystem(stack)
            optimizer = OpticalOptimizer(
                system=system,
                wavelengths=wavelengths,
                manufacturing_constraints=mfg_constraints
            )

            res = optimizer.fit_to_test_plates(re_optimize_bfl=True)
            if res and res.optimized_stack:
                if hasattr(self.model, '_push_undo'):
                    self.model._push_undo("Fit to Test Plates")
                self.model.set_surface_stack(res.optimized_stack)
                if hasattr(self.analysis_tabs, "refresh_analysis"):
                    self.analysis_tabs.refresh_analysis(wavelengths=wavelengths)
                self.on_trace_rays()
                msg = f"Tooling Test Plates: Snapped radii & re-optimized air spaces (RMS: {res.initial_rms:.3f} -> {res.final_rms:.3f})"
                self._status_msg = msg
                if QT_AVAILABLE and hasattr(self, 'statusBar') and self.statusBar() is not None:
                    try:
                        self.statusBar().showMessage(msg)
                    except Exception:
                        pass
                if hasattr(self, 'optimizer_dock') and self.optimizer_dock is not None:
                    self.optimizer_dock.set_status("Test plates fitted")
                if QT_AVAILABLE:
                    QMessageBox.information(self, "Test Plate Tooling Fit", msg)
        except Exception as e:
            if QT_AVAILABLE:
                QMessageBox.critical(self, "Test Plate Fitting Error", f"Failed to fit test plates: {e}")

    def on_load_pupil_relay(self):
        from optics_app.engine.surface import Surface, SurfaceType
        stack = self.model.get_surface_stack()
        if hasattr(self.model, '_push_undo'):
            self.model._push_undo("Load Pupil-Imaging Relay")
        
        # Simple placeholder for pupil relay: Field lens + 2 condensers
        s1 = Surface(name="Field Lens (Front)", radius=50.0, thickness=5.0, glass="N-BK7", semi_diameter=12.5)
        s2 = Surface(name="Field Lens (Back)", radius=-50.0, thickness=20.0, glass="AIR", semi_diameter=12.5)
        s3 = Surface(name="Condenser 1 (Front)", radius=30.0, thickness=10.0, glass="N-BK7", semi_diameter=12.5)
        s4 = Surface(name="Condenser 1 (Back)", radius=-100.0, thickness=5.0, glass="AIR", semi_diameter=12.5)
        s5 = Surface(name="Condenser 2 (Front)", radius=100.0, thickness=10.0, glass="N-BK7", semi_diameter=12.5)
        s6 = Surface(name="Condenser 2 (Back)", radius=-30.0, thickness=30.0, glass="AIR", semi_diameter=12.5)
        
        stack.surfaces.extend([s1, s2, s3, s4, s5, s6])
        stack._update_accumulated_positions()
        
        self.model.set_surface_stack(stack)
        self.on_trace_rays()
        if QT_AVAILABLE:
            QMessageBox.information(self, "Relay Loaded", "Pupil-Imaging Relay added to the layout.")


    @property
    def current_system(self):
        from optics_app.engine.models import SpectrometerSystem
        stack = self.model.get_surface_stack()
        sys_cfg = self.system_config if hasattr(self, 'system_config') else None
        
        wls = [0.5876]
        if sys_cfg and sys_cfg.wavelengths:
            wls = [w.wavelength_um for w in sys_cfg.wavelengths]
            
        sys = SpectrometerSystem(
            surface_stack=stack,
            wavelengths=wls,
            primary_wavelength=wls[0] if wls else 0.5876,
            system_config=sys_cfg
        )
        return sys
        
    @current_system.setter
    def current_system(self, sys_obj):
        self.model.set_surface_stack(sys_obj.surface_stack)
        if hasattr(self, 'system_config'):
            # Sync Wavelengths
            if hasattr(sys_obj, 'wavelengths') and sys_obj.wavelengths:
                self.system_config.wavelengths.clear()
                for i, wl in enumerate(sys_obj.wavelengths):
                    self.system_config.add_wavelength(wl, is_primary=(i==0))
            
            # Sync Fields
            if hasattr(sys_obj, 'field_angle') and sys_obj.field_angle is not None:
                if self.system_config.fields:
                    self.system_config.fields[0].y = sys_obj.field_angle
                    if hasattr(self.system_config.fields[0], 'angle_y_deg'):
                        self.system_config.fields[0].angle_y_deg = sys_obj.field_angle
                else:
                    self.system_config.add_field(0.0, sys_obj.field_angle)
            
            # Sync Apertures
            if hasattr(sys_obj, 'source_config') and sys_obj.source_config:
                br = sys_obj.source_config.get('beam_radius', 5.0) if isinstance(sys_obj.source_config, dict) else getattr(sys_obj.source_config, 'beam_radius', 5.0)
                self.system_config.aperture_value = br * 2.0  # EPD
                
        if hasattr(self, 'system_config_dock') and self.system_config_dock:
            self.system_config_dock.update_ui()
            if hasattr(self.system_config_dock, 'sync_from_ui'):
                self.system_config_dock.sync_from_ui()
            
        self.on_trace_rays()

    def on_show_plm_studio(self) -> None:
        """Open or raise the independent TI Modulator Studio (PLM / DMD) dock window."""
        if hasattr(self, 'plm_dock') and self.plm_dock is not None:
            if hasattr(self.plm_dock, 'set_surface_stack') and hasattr(self, 'get_current_surface_stack'):
                self.plm_dock.set_surface_stack(self.get_current_surface_stack())
            self.plm_dock.show()
            self.plm_dock.raise_()
            self.plm_dock.activateWindow()
            if hasattr(self.plm_dock, 'studio') and self.plm_dock.studio is not None:
                if hasattr(self.plm_dock.studio, 'canvas') and self.plm_dock.studio.canvas is not None:
                    self.plm_dock.studio.canvas.draw_idle()
        if hasattr(self, 'statusBar') and self.statusBar() is not None:
            self.statusBar().showMessage("TI Modulator Studio active (PLM 4-Bit Phase / DMD 1-Bit Binary)", 3500)

    def on_load_beamsplitter_preset(self, preset_filename: str) -> None:
        """Load a Thorlabs beamsplitter reference architecture preset from catalog presets."""
        import os
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        preset_path = os.path.join(repo_root, "optics_app", "catalogs", "presets", preset_filename)
        if not os.path.isfile(preset_path):
            preset_path = os.path.join(repo_root, "v2", "catalogs", "presets", preset_filename)
        if not os.path.isfile(preset_path):
            preset_path = os.path.join(repo_root, preset_filename)
        if os.path.isfile(preset_path):
            self.on_load_project(preset_path)
            if hasattr(self, 'statusBar') and self.statusBar() is not None:
                self.statusBar().showMessage(f"Loaded Reference Design: {preset_filename}", 5000)
        else:
            if QT_AVAILABLE:
                QMessageBox.warning(self, "Preset Not Found", f"Could not find preset file at:\n{preset_path}")

    def on_load_plm_spectrometer(self) -> None:
        """Load the high-performance Texas Instruments 4-Bit PLM Raman spectrometer design."""
        import os
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        plm_design_path = os.path.join(repo_root, "spectrometer_raman_plm_4bit.json")
        if not os.path.isfile(plm_design_path):
            alt_path = os.path.join("C:/Users/Juan/Desktop/Spectrometer_Project/Optical_Design", "spectrometer_raman_plm_4bit.json")
            if os.path.isfile(alt_path):
                plm_design_path = alt_path
        if os.path.isfile(plm_design_path):
            self.on_load_project(plm_design_path)
            if hasattr(self, 'statusBar') and self.statusBar() is not None:
                self.statusBar().showMessage(f"Loaded TI-PLM Spectrometer: {os.path.basename(plm_design_path)}", 5000)
        else:
            if QT_AVAILABLE:
                QMessageBox.warning(self, "TI-PLM Design Not Found", f"Could not find design file at:\n{plm_design_path}")

    def _toggle_plm_state(self) -> None:
        """Step programmable 4-bit state on the active TI-PLM surface."""
        stack = self.model.get_surface_stack()
        plm_idx = -1
        for i, s in enumerate(stack.surfaces):
            if getattr(s, 'is_plm', False) or getattr(s, 'mechanism_type', '') == 'PLM':
                plm_idx = i
                break
        if plm_idx >= 0:
            surf = stack.surfaces[plm_idx]
            states = ["ON", "OFF", "HADAMARD_A", "HADAMARD_B", "STATE_0", "STATE_4", "STATE_8", "STATE_12"]
            cur = getattr(surf, 'mechanical_state', 'ON')
            nxt = states[(states.index(cur) + 1) % len(states)] if cur in states else "ON"
            surf.mechanical_state = nxt
            if hasattr(self.model, '_push_undo'):
                self.model._push_undo(f"PLM State -> {nxt}")
            self.model.set_surface_stack(stack)
            self.on_trace_rays()
            self._sync_mech_action_ui()
            if hasattr(self, 'statusBar') and self.statusBar() is not None:
                self.statusBar().showMessage(f"TI-PLM switched to state: {nxt}", 3000)
        else:
            if hasattr(self, 'statusBar') and self.statusBar() is not None:
                self.statusBar().showMessage("No active TI-PLM component found in system.", 3000)

