"""
Design Modification History Dock Widget for 2D Optics Simulation & Studio.
Displays chronological timeline of all optical design modifications with step numbers,
timestamps, action descriptions, and surface counts.
Supports 1-click time travel / rollback to any past design version, plus Save and Save As triggers.
"""
import os
from typing import Optional, Any, Dict, List

from optics_app.gui.qt_compat import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QTableWidget,
    QTableWidgetItem, QHeaderView, QDockWidget, Qt, QT_AVAILABLE, QAbstractItemView,
    QtGui, pyqtSignal
)
from optics_app.engine.undo_manager import UndoManager, OpticalSystemState


class DesignHistoryDockWidget(QDockWidget if QT_AVAILABLE else object):
    """
    Dockable widget displaying real-time chronological history of optical design modifications.
    Allows jumping back to any past version, undoing/redoing, and fast saving.
    """
    revert_requested = pyqtSignal(int) if QT_AVAILABLE else None
    undo_requested = pyqtSignal() if QT_AVAILABLE else None
    redo_requested = pyqtSignal() if QT_AVAILABLE else None
    save_requested = pyqtSignal() if QT_AVAILABLE else None
    save_as_requested = pyqtSignal() if QT_AVAILABLE else None
    open_history_folder_requested = pyqtSignal() if QT_AVAILABLE else None

    def __init__(self, undo_manager: Optional[UndoManager] = None, parent=None):
        if QT_AVAILABLE:
            super().__init__("Design Modification History", parent)
            self.setObjectName("DesignHistoryDock")
            self.setAllowedAreas(Qt.AllDockWidgetAreas)
        
        self.undo_manager = undo_manager
        if self.undo_manager is not None:
            self.undo_manager.add_history_listener(self.refresh_history_table)

        self.dark_mode: bool = False
        self._init_ui()
        self._update_theme_styles()

    def set_dark_mode(self, enabled: bool) -> None:
        """Switch between dark and light background themes."""
        self.dark_mode = bool(enabled)
        self._update_theme_styles()
        self.refresh_history_table()

    def _update_theme_styles(self) -> None:
        """Update styled elements to match the current theme."""
        if not QT_AVAILABLE:
            return
        is_dark = getattr(self, 'dark_mode', False)
        if hasattr(self, 'lbl_status') and self.lbl_status:
            col = "#94a3b8" if is_dark else "#555555"
            self.lbl_status.setStyleSheet(f"font-size: 11px; color: {col}; padding: 2px;")

    def _init_ui(self):
        if not QT_AVAILABLE:
            return

        container = QWidget(self)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # 1. Top Quick Action Bar (Save, Save As, Undo, Redo)
        bar_layout = QHBoxLayout()
        bar_layout.setContentsMargins(0, 0, 0, 0)
        bar_layout.setSpacing(4)

        self.btn_save = QPushButton("💾 Save", container)
        self.btn_save.setToolTip("Save current optical design (Ctrl+S)")
        self.btn_save.setStyleSheet("font-weight: bold; background-color: #2E7D32; color: white; padding: 4px 8px; border-radius: 3px;")
        self.btn_save.clicked.connect(self._on_save_clicked)
        bar_layout.addWidget(self.btn_save)

        self.btn_save_as = QPushButton("💾 Save As...", container)
        self.btn_save_as.setToolTip("Save current optical design to a new file (Ctrl+Shift+S)")
        self.btn_save_as.setStyleSheet("font-weight: bold; background-color: #0277BD; color: white; padding: 4px 8px; border-radius: 3px;")
        self.btn_save_as.clicked.connect(self._on_save_as_clicked)
        bar_layout.addWidget(self.btn_save_as)

        self.btn_undo = QPushButton("↩ Undo", container)
        self.btn_undo.setToolTip("Undo last modification (Ctrl+Z)")
        self.btn_undo.clicked.connect(self._on_undo_clicked)
        bar_layout.addWidget(self.btn_undo)

        self.btn_redo = QPushButton("↪ Redo", container)
        self.btn_redo.setToolTip("Redo last undone modification (Ctrl+Shift+Z / Ctrl+Y)")
        self.btn_redo.clicked.connect(self._on_redo_clicked)
        bar_layout.addWidget(self.btn_redo)

        layout.addLayout(bar_layout)

        # 2. Status / Summary Label
        self.lbl_status = QLabel("History: 0 modifications recorded | Ready", container)
        self.lbl_status.setStyleSheet("font-size: 11px; color: #555; padding: 2px;")
        layout.addWidget(self.lbl_status)

        # 3. History Timeline Table
        self.table = QTableWidget(container)
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["Step", "Time", "Action / Modification", "Surfaces", "Status"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)

        header = self.table.horizontalHeader()
        if hasattr(header, 'setSectionResizeMode'):
            header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
            header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
            header.setSectionResizeMode(2, QHeaderView.Stretch)
            header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
            header.setSectionResizeMode(4, QHeaderView.ResizeToContents)

        self.table.doubleClicked.connect(self._on_row_double_clicked)
        layout.addWidget(self.table)

        # 4. Bottom Control Bar (Revert to Selected, Version Backups, Clear)
        bot_layout = QHBoxLayout()
        self.btn_revert = QPushButton("⏮ Revert to Selected Version", container)
        self.btn_revert.setToolTip("Restore the optical design to the state selected in the history list above")
        self.btn_revert.setStyleSheet("font-weight: bold; padding: 4px;")
        self.btn_revert.clicked.connect(self._on_revert_clicked)
        bot_layout.addWidget(self.btn_revert)

        self.btn_history_folder = QPushButton("📂 Version Backups", container)
        self.btn_history_folder.setToolTip("Open the .history folder where version backup copies are stored")
        self.btn_history_folder.clicked.connect(self._on_open_history_folder_clicked)
        bot_layout.addWidget(self.btn_history_folder)

        self.btn_clear = QPushButton("🧹 Clear", container)
        self.btn_clear.setToolTip("Clear history timeline")
        self.btn_clear.clicked.connect(self._on_clear_clicked)
        bot_layout.addWidget(self.btn_clear)

        layout.addLayout(bot_layout)
        self.setWidget(container)

        self.refresh_history_table()

    def _on_open_history_folder_clicked(self):
        if self.open_history_folder_requested:
            self.open_history_folder_requested.emit()

    def set_undo_manager(self, undo_manager: UndoManager) -> None:
        self.undo_manager = undo_manager
        if self.undo_manager is not None:
            self.undo_manager.add_history_listener(self.refresh_history_table)
        self.refresh_history_table()

    def refresh_history_table(self) -> None:
        """Populate history table with chronological design modification states."""
        if not QT_AVAILABLE or not hasattr(self, 'table') or self.table is None:
            return

        if self.undo_manager is None:
            self.table.setRowCount(0)
            if hasattr(self, 'lbl_status'):
                self.lbl_status.setText("History: No undo manager attached")
            return

        timeline = self.undo_manager.get_history_timeline()
        self.table.setRowCount(len(timeline))

        curr_row = -1
        for row_idx, item in enumerate(timeline):
            step_str = f"#{item['step']}"
            time_str = item['timestamp']
            desc_str = item['description']
            surf_str = f"{item['num_surfaces']} surfs"
            
            if item['is_current']:
                status_str = "▶ Current"
                curr_row = row_idx
            elif item['is_undo']:
                status_str = "Past"
            else:
                status_str = "Future (Redo)"

            i_step = QTableWidgetItem(step_str)
            i_time = QTableWidgetItem(time_str)
            i_desc = QTableWidgetItem(desc_str)
            i_surf = QTableWidgetItem(surf_str)
            i_stat = QTableWidgetItem(status_str)

            i_step.setTextAlignment(Qt.AlignCenter)
            i_time.setTextAlignment(Qt.AlignCenter)
            i_surf.setTextAlignment(Qt.AlignCenter)
            i_stat.setTextAlignment(Qt.AlignCenter)

            is_dark = getattr(self, 'dark_mode', False)
            if item['is_current']:
                font = i_desc.font()
                font.setBold(True)
                cur_bg = "#064e3b" if is_dark else "#E8F5E9"
                cur_fg = "#6ee7b7" if is_dark else "#1B5E20"
                for it in [i_step, i_time, i_desc, i_surf, i_stat]:
                    it.setFont(font)
                    if QtGui is not None and hasattr(QtGui, 'QBrush') and hasattr(QtGui, 'QColor'):
                        it.setBackground(QtGui.QBrush(QtGui.QColor(cur_bg)))
                        it.setForeground(QtGui.QBrush(QtGui.QColor(cur_fg)))
            elif not item['is_undo']:
                # Redo item (greyed out)
                if QtGui is not None and hasattr(QtGui, 'QBrush') and hasattr(QtGui, 'QColor'):
                    redo_fg = "#64748b" if is_dark else "#888888"
                    for it in [i_step, i_time, i_desc, i_surf, i_stat]:
                        it.setForeground(QtGui.QBrush(QtGui.QColor(redo_fg)))

            self.table.setItem(row_idx, 0, i_step)
            self.table.setItem(row_idx, 1, i_time)
            self.table.setItem(row_idx, 2, i_desc)
            self.table.setItem(row_idx, 3, i_surf)
            self.table.setItem(row_idx, 4, i_stat)

        if curr_row >= 0:
            self.table.selectRow(curr_row)

        if hasattr(self, 'lbl_status'):
            undo_cnt = self.undo_manager.undo_count
            redo_cnt = self.undo_manager.redo_count
            self.lbl_status.setText(f"History: {len(timeline)} actions logged | {undo_cnt} undoable, {redo_cnt} redoable")

        # Update button enable states
        if hasattr(self, 'btn_undo'):
            self.btn_undo.setEnabled(self.undo_manager.can_undo())
        if hasattr(self, 'btn_redo'):
            self.btn_redo.setEnabled(self.undo_manager.can_redo())

    def _on_row_double_clicked(self, index):
        self._on_revert_clicked()

    def _on_revert_clicked(self):
        if not QT_AVAILABLE or self.table is None:
            return
        selected_rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        if not selected_rows:
            return
        row = selected_rows[0].row()
        timeline = self.undo_manager.get_history_timeline() if self.undo_manager else []
        if 0 <= row < len(timeline):
            target_step = timeline[row]['step']
            if self.revert_requested:
                self.revert_requested.emit(target_step)

    def _on_undo_clicked(self):
        if self.undo_requested:
            self.undo_requested.emit()

    def _on_redo_clicked(self):
        if self.redo_requested:
            self.redo_requested.emit()

    def _on_save_clicked(self):
        if self.save_requested:
            self.save_requested.emit()

    def _on_save_as_clicked(self):
        if self.save_as_requested:
            self.save_as_requested.emit()

    def _on_clear_clicked(self):
        if self.undo_manager:
            self.undo_manager.clear()
