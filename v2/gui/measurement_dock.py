"""
Dedicated Measurement & Inspection Tools Panel for 2D Optics Simulation Engine.
Provides interactive 2-point distance dimensioning, 3-point angle inspection,
live metric readouts, real-time measurements history table, surface snapping controls,
and CSV export.
"""

import os
import sys
import numpy as np
from typing import Optional, List, Dict, Any

from optics_app.gui.qt_compat import (
    QWidget, QVBoxLayout, QHBoxLayout, QDockWidget, QTableWidget, QTableWidgetItem,
    QPushButton, QLabel, QCheckBox, QSpinBox, QDoubleSpinBox, QHeaderView,
    QAbstractItemView, QFileDialog, QMessageBox, pyqtSignal, DummySignal, Qt, QT_AVAILABLE
)

class MeasurementToolsDockWidget(QDockWidget):
    """
    Dedicated Dockable Panel for Optical Measurement and Geometric Inspection Tools.
    """
    mode_changed = pyqtSignal(str) if QT_AVAILABLE else DummySignal()
    measurement_deleted = pyqtSignal(int) if QT_AVAILABLE else DummySignal()
    measurements_cleared = pyqtSignal() if QT_AVAILABLE else DummySignal()

    def __init__(self, dual_view=None, parent=None):
        super().__init__("📏 Measurement & Inspection Tools", parent)
        self.setObjectName("MeasurementToolsDock")
        self.dual_view = dual_view
        self._block_updates = False
        self.dark_mode: bool = False

        self._init_ui()
        self._update_theme_styles()

        if self.dual_view is not None:
            self.bind_dual_view(self.dual_view)

    def bind_dual_view(self, dual_view):
        """Binds this dock widget to an active DualViewWidget instance."""
        self.dual_view = dual_view
        if hasattr(self.dual_view, "measurements_changed"):
            try:
                self.dual_view.measurements_changed.connect(self.on_measurements_updated)
            except Exception:
                pass
        if hasattr(self.dual_view, "angles_visibility_changed"):
            try:
                self.dual_view.angles_visibility_changed.connect(self._on_angles_visibility_changed)
            except Exception:
                pass
        self.sync_from_dual_view()

    def set_dark_mode(self, enabled: bool) -> None:
        """Switch between dark and light background themes."""
        self.dark_mode = bool(enabled)
        self._update_theme_styles()

    def _update_theme_styles(self) -> None:
        """Update styled elements to match the current theme."""
        if not QT_AVAILABLE:
            return
        is_dark = getattr(self, 'dark_mode', False)
        if hasattr(self, 'lbl_readout') and self.lbl_readout:
            if is_dark:
                self.lbl_readout.setStyleSheet(
                    "background-color: #1e3a8a; color: #bfdbfe; font-weight: bold; "
                    "padding: 6px 8px; border-radius: 4px; border: 1px solid #3b82f6;"
                )
            else:
                self.lbl_readout.setStyleSheet(
                    "background-color: #E3F2FD; color: #0D47A1; font-weight: bold; "
                    "padding: 6px 8px; border-radius: 4px; border: 1px solid #90CAF9;"
                )

    def _init_ui(self):
        container = QWidget(self)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # -------------------------------------------------------------
        # 1. Measurement Modes Toolbar
        # -------------------------------------------------------------
        lbl_mode = QLabel("<b>Measurement Mode:</b>", self)
        layout.addWidget(lbl_mode)

        mode_layout = QHBoxLayout()
        mode_layout.setSpacing(4)

        self.btn_mode_navigate = QPushButton("🖱 Pan / Zoom", self)
        self.btn_mode_navigate.setCheckable(True)
        self.btn_mode_navigate.setChecked(True)
        self.btn_mode_navigate.setToolTip("Pan and zoom navigation mode")
        self.btn_mode_navigate.clicked.connect(lambda: self.set_mode(None))
        mode_layout.addWidget(self.btn_mode_navigate)

        self.btn_mode_distance = QPushButton("📏 Distance (2 Pts)", self)
        self.btn_mode_distance.setCheckable(True)
        self.btn_mode_distance.setToolTip("Click 2 points or snap to surface to measure distance (ΔZ, ΔY, θ)")
        self.btn_mode_distance.clicked.connect(lambda: self.set_mode("distance"))
        mode_layout.addWidget(self.btn_mode_distance)

        self.btn_mode_angle = QPushButton("📐 Angle (3 Pts)", self)
        self.btn_mode_angle.setCheckable(True)
        self.btn_mode_angle.setToolTip("Click 3 points (P1 -> Apex -> P2) to measure angle in degrees")
        self.btn_mode_angle.clicked.connect(lambda: self.set_mode("angle"))
        mode_layout.addWidget(self.btn_mode_angle)

        layout.addLayout(mode_layout)

        # Action Buttons (Fit, Clear, Delete)
        action_layout = QHBoxLayout()
        action_layout.setSpacing(4)

        self.btn_fit_view = QPushButton("🔍 Fit View", self)
        self.btn_fit_view.setToolTip("Fit all optics to screen (resets zoom/pan)")
        self.btn_fit_view.clicked.connect(self._on_fit_view)
        action_layout.addWidget(self.btn_fit_view)

        self.btn_delete_selected = QPushButton("🗑 Delete Item", self)
        self.btn_delete_selected.setToolTip("Delete selected measurement from list and plot")
        self.btn_delete_selected.clicked.connect(self._on_delete_selected)
        action_layout.addWidget(self.btn_delete_selected)

        self.btn_clear_all = QPushButton("🧹 Clear All", self)
        self.btn_clear_all.setToolTip("Clear all measurements")
        self.btn_clear_all.clicked.connect(self._on_clear_all)
        action_layout.addWidget(self.btn_clear_all)

        layout.addLayout(action_layout)

        # -------------------------------------------------------------
        # 2. Live Readout Banner
        # -------------------------------------------------------------
        self.lbl_readout = QLabel("Ready: Select a tool above to measure distances or angles on the 2D layout.", self)
        if QT_AVAILABLE:
            self.lbl_readout.setStyleSheet(
                "background-color: #E3F2FD; color: #0D47A1; font-weight: bold; "
                "padding: 6px 8px; border-radius: 4px; border: 1px solid #90CAF9;"
            )
        self.lbl_readout.setWordWrap(True)
        layout.addWidget(self.lbl_readout)

        # -------------------------------------------------------------
        # 3. Active Measurements History Table
        # -------------------------------------------------------------
        lbl_tbl = QLabel("<b>Active Measurements List:</b>", self)
        layout.addWidget(lbl_tbl)

        self.table_measurements = QTableWidget(self)
        self.table_measurements.setColumnCount(7)
        self.table_measurements.setHorizontalHeaderLabels([
            "#", "Type", "Value", "ΔZ (mm)", "ΔY (mm)", "Angle (°)", "Points / Apex"
        ])
        if QT_AVAILABLE:
            self.table_measurements.setSelectionBehavior(QAbstractItemView.SelectRows)
            self.table_measurements.setSelectionMode(QAbstractItemView.SingleSelection)
            self.table_measurements.horizontalHeader().setStretchLastSection(True)
            self.table_measurements.verticalHeader().setVisible(False)
            self.table_measurements.itemSelectionChanged.connect(self._on_table_selection_changed)

        layout.addWidget(self.table_measurements)

        # -------------------------------------------------------------
        # 4. Snapping, Angle Overlays & Export Controls
        # -------------------------------------------------------------
        overlay_layout = QVBoxLayout()
        overlay_layout.setSpacing(3)

        self.chk_show_angles = QCheckBox("📐 Show Incidence & Output Angles (Main λ)", self)
        self.chk_show_angles.setChecked(True)
        self.chk_show_angles.setToolTip("Toggle automatic visualization of surface normals, incidence angles (α/θ_in), and output angles (β/θ_out) on reflective optics, gratings, and DMDs for the primary wavelength.")
        self.chk_show_angles.toggled.connect(self._on_show_angles_toggled)
        overlay_layout.addWidget(self.chk_show_angles)

        snap_layout = QHBoxLayout()
        self.chk_snap = QCheckBox("🧲 Snap to Vertices / Rims", self)
        self.chk_snap.setChecked(True)
        self.chk_snap.setToolTip("Automatically snap measurement points to surface vertices and aperture rims")
        self.chk_snap.toggled.connect(self._on_snap_toggled)
        snap_layout.addWidget(self.chk_snap)

        self.btn_export_csv = QPushButton("💾 Export CSV...", self)
        self.btn_export_csv.setToolTip("Export measurements table to a CSV spreadsheet file")
        self.btn_export_csv.clicked.connect(self._on_export_csv)
        snap_layout.addWidget(self.btn_export_csv)

        overlay_layout.addLayout(snap_layout)
        layout.addLayout(overlay_layout)

        self.setWidget(container)

    # -------------------------------------------------------------------------
    # Mode Control & Synchronization
    # -------------------------------------------------------------------------
    def set_mode(self, mode: Optional[str]):
        """Activates a measurement mode and synchronizes with DualViewWidget."""
        self.btn_mode_navigate.setChecked(mode is None)
        self.btn_mode_distance.setChecked(mode == "distance")
        self.btn_mode_angle.setChecked(mode == "angle")

        if self.dual_view is not None and hasattr(self.dual_view, "set_measure_mode"):
            self.dual_view.set_measure_mode(mode)

        if mode == "distance":
            self.lbl_readout.setText("📏 Distance Tool: Click Point 1 (or snap to surface), then click Point 2.")
        elif mode == "angle":
            self.lbl_readout.setText("📐 Angle Tool: Click Point 1 -> Click Point 2 (Apex) -> Click Point 3.")
        else:
            self.lbl_readout.setText("🖱 Pan / Zoom: Left-drag to pan, scroll wheel to zoom. Click handles to edit.")

    def _on_fit_view(self):
        if self.dual_view is not None and hasattr(self.dual_view, "reset_view_limits"):
            self.dual_view.reset_view_limits()

    def _on_clear_all(self):
        if self.dual_view is not None and hasattr(self.dual_view, "clear_measurements"):
            self.dual_view.clear_measurements()
        self.table_measurements.setRowCount(0)
        self.lbl_readout.setText("All measurements cleared.")

    def _on_delete_selected(self):
        if not QT_AVAILABLE:
            return
        row = self.table_measurements.currentRow()
        if row >= 0 and self.dual_view is not None:
            if hasattr(self.dual_view, "delete_measurement"):
                self.dual_view.delete_measurement(row)
            elif hasattr(self.dual_view, "measurement_history") and 0 <= row < len(self.dual_view.measurement_history):
                self.dual_view.measurement_history.pop(row)
                self.dual_view.refresh_layout()
            self.sync_from_dual_view()

    def _on_show_angles_toggled(self, checked: bool):
        if self.dual_view is not None and hasattr(self.dual_view, "set_show_incidence_angles"):
            self.dual_view.set_show_incidence_angles(checked)
        parent_win = self.parent()
        if parent_win is not None and hasattr(parent_win, "act_show_angles") and hasattr(parent_win.act_show_angles, "setChecked"):
            parent_win.act_show_angles.blockSignals(True)
            parent_win.act_show_angles.setChecked(checked)
            parent_win.act_show_angles.blockSignals(False)

    def _on_angles_visibility_changed(self, enabled: bool):
        if hasattr(self, "chk_show_angles"):
            self.chk_show_angles.blockSignals(True)
            self.chk_show_angles.setChecked(enabled)
            self.chk_show_angles.blockSignals(False)

    def _on_snap_toggled(self, checked: bool):
        if self.dual_view is not None:
            setattr(self.dual_view, "snap_enabled", checked)

    def _on_table_selection_changed(self):
        if self._block_updates or not QT_AVAILABLE:
            return
        row = self.table_measurements.currentRow()
        if row >= 0 and self.dual_view is not None:
            if hasattr(self.dual_view, "select_measurement"):
                self.dual_view.select_measurement(row)
            else:
                self.dual_view.selected_measurement_idx = row
                self.dual_view.refresh_layout()

    # -------------------------------------------------------------------------
    # Updates from DualViewWidget
    # -------------------------------------------------------------------------
    def on_measurements_updated(self, measurements: List[Dict[str, Any]], msg: str = ""):
        """Slot called whenever measurements are added, dragged, or modified in 2D view."""
        if msg:
            self.lbl_readout.setText(msg)
        self.sync_from_dual_view()

    def sync_from_dual_view(self):
        """Refreshes the measurement table with data from the 2D layout view."""
        if not QT_AVAILABLE or self.dual_view is None:
            return
        
        if hasattr(self.dual_view, "show_incidence_angles") and hasattr(self, "chk_show_angles"):
            self.chk_show_angles.blockSignals(True)
            self.chk_show_angles.setChecked(self.dual_view.show_incidence_angles)
            self.chk_show_angles.blockSignals(False)

        self._block_updates = True
        try:
            history = getattr(self.dual_view, "measurement_history", [])
            self.table_measurements.setRowCount(len(history))

            for i, m in enumerate(history):
                m_type = m.get("type", "distance")
                
                # Column 0: Index
                self.table_measurements.setItem(i, 0, QTableWidgetItem(f"#{i+1}"))
                
                # Column 1: Type
                t_str = "📏 Distance" if m_type == "distance" else "📐 Angle"
                self.table_measurements.setItem(i, 1, QTableWidgetItem(t_str))

                if m_type == "distance":
                    dist = m.get("distance", 0.0)
                    dz = m.get("dz", 0.0)
                    dy = m.get("dy", 0.0)
                    ang = m.get("angle", 0.0)
                    p1 = m.get("p1", [0, 0])
                    p2 = m.get("p2", [0, 0])

                    self.table_measurements.setItem(i, 2, QTableWidgetItem(f"{dist:.3f} mm"))
                    self.table_measurements.setItem(i, 3, QTableWidgetItem(f"{dz:.3f}"))
                    self.table_measurements.setItem(i, 4, QTableWidgetItem(f"{dy:.3f}"))
                    self.table_measurements.setItem(i, 5, QTableWidgetItem(f"{ang:.1f}°"))
                    self.table_measurements.setItem(i, 6, QTableWidgetItem(f"({p1[0]:.2f}, {p1[1]:.2f}) -> ({p2[0]:.2f}, {p2[1]:.2f})"))

                elif m_type in ("angle", "3point_angle", "surface_angle"):
                    ang = m.get("angle", 0.0)
                    apex = m.get("apex", [0, 0])
                    p1 = m.get("p1", [0, 0])
                    p2 = m.get("p2", [0, 0])

                    self.table_measurements.setItem(i, 2, QTableWidgetItem(f"{ang:.2f}°"))
                    self.table_measurements.setItem(i, 3, QTableWidgetItem("—"))
                    self.table_measurements.setItem(i, 4, QTableWidgetItem("—"))
                    self.table_measurements.setItem(i, 5, QTableWidgetItem(f"{ang:.2f}°"))
                    self.table_measurements.setItem(i, 6, QTableWidgetItem(f"Apex: ({apex[0]:.2f}, {apex[1]:.2f})"))

            # Highlight selected row
            sel_idx = getattr(self.dual_view, "selected_measurement_idx", None)
            if sel_idx is not None and 0 <= sel_idx < len(history):
                self.table_measurements.selectRow(sel_idx)

        finally:
            self._block_updates = False

    # -------------------------------------------------------------------------
    # CSV Export
    # -------------------------------------------------------------------------
    def _on_export_csv(self):
        if not QT_AVAILABLE or self.dual_view is None:
            return
        history = getattr(self.dual_view, "measurement_history", [])
        if not history:
            QMessageBox.information(self, "Export Measurements", "No active measurements to export.")
            return

        filepath, _ = QFileDialog.getSaveFileName(self, "Export Measurements CSV", "measurements_report.csv", "CSV Files (*.csv)")
        if not filepath:
            return

        try:
            import csv
            with open(filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["#", "Type", "Value", "Delta_Z_mm", "Delta_Y_mm", "Angle_deg", "Point_1", "Apex_Point_2"])
                for i, m in enumerate(history):
                    m_type = m.get("type", "distance")
                    if m_type == "distance":
                        writer.writerow([
                            i+1, "Distance", f"{m.get('distance', 0.0):.4f} mm",
                            f"{m.get('dz', 0.0):.4f}", f"{m.get('dy', 0.0):.4f}",
                            f"{m.get('angle', 0.0):.2f}",
                            f"{m.get('p1', [])}", f"{m.get('p2', [])}"
                        ])
                    else:
                        writer.writerow([
                            i+1, "Angle", f"{m.get('angle', 0.0):.2f} deg",
                            "—", "—", f"{m.get('angle', 0.0):.2f}",
                            f"{m.get('p1', [])}", f"Apex: {m.get('apex', [])}, P2: {m.get('p2', [])}"
                        ])
            QMessageBox.information(self, "Export Success", f"Successfully exported {len(history)} measurements to:\n{filepath}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export CSV: {e}")
