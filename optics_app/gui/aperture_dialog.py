"""
Aperture and Stop Configuration Dialog for Optical Surfaces.
Allows interactive definition of aperture stops, clear apertures, slits, annular obscurations,
and spider vanes with real-time parameter validation and LDE synchronization.
"""

from typing import Optional, Dict, Any
import numpy as np

from optics_app.gui.qt_compat import (
    QT_AVAILABLE, QtWidgets, QDialog, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QComboBox, QDoubleSpinBox,
    QSpinBox, QCheckBox, QGroupBox, Qt
)
from optics_app.engine.surface import Surface, SurfaceType


class ApertureStopDialog(QDialog if QT_AVAILABLE else object):
    """
    Modal or modeless dialog to define the Aperture Stop status and
    aperture geometry (Circular, Rectangular slit, Annular, Elliptical, Spider).
    """
    def __init__(self, surface: Surface, surface_idx: int = 0, model: Optional[Any] = None, parent=None):
        if QT_AVAILABLE:
            super().__init__(parent)
            self.setWindowTitle(f"Define Aperture & Stop — Surface {surface_idx}: {surface.name}")
            self.resize(460, 480)
        self.surface = surface
        self.surface_idx = surface_idx
        self.model = model
        self._syncing = False
        dm = getattr(parent, "dark_mode", None)
        if dm is None and parent is not None and hasattr(parent, "window"):
            w = parent.window()
            if w is not None:
                dm = getattr(w, "dark_mode", None)
        self.dark_mode = bool(dm) if dm is not None else False

        if QT_AVAILABLE:
            self._setup_ui()
            self._update_theme_styles()
            self._load_from_surface()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # 1. Stop Designation Group
        stop_grp = QGroupBox("Aperture Stop Designation", self)
        stop_layout = QVBoxLayout(stop_grp)

        self.chk_is_stop = QCheckBox("🔴 Mark this surface as the System Aperture Stop (STO)", stop_grp)
        self.chk_is_stop.setToolTip(
            "The Aperture Stop physically limits the bundle of light passing through the system\n"
            "and determines the system entrance and exit pupils."
        )
        self.chk_is_stop.toggled.connect(self._on_stop_toggled)
        stop_layout.addWidget(self.chk_is_stop)

        self.lbl_stop_desc = QLabel(
            "When set as Stop, ray tracing uses this element's clear aperture to define the marginal ray cone.",
            stop_grp
        )
        self.lbl_stop_desc.setWordWrap(True)
        self.lbl_stop_desc.setStyleSheet("color: #555; font-size: 11px;")
        stop_layout.addWidget(self.lbl_stop_desc)
        layout.addWidget(stop_grp)

        # 2. Aperture Geometry & Dimensions Group
        geom_grp = QGroupBox("Aperture Geometry & Clear Aperture", self)
        geom_layout = QGridLayout(geom_grp)
        geom_layout.setSpacing(8)

        geom_layout.addWidget(QLabel("Aperture Shape:"), 0, 0)
        self.combo_type = QComboBox(geom_grp)
        self.combo_type.addItems([
            "Circular (Clear Aperture)",
            "Rectangular (Slit / Sensor Mask)",
            "Annular (Central Obscuration)",
            "Elliptical Aperture",
            "Spider (Vanes & Central Hub)",
            "Point (Zero Extent / Point Object/Image)",
        ])
        self.combo_type.currentIndexChanged.connect(self._on_type_changed)
        geom_layout.addWidget(self.combo_type, 0, 1)

        # Point info readout
        self.lbl_point_info = QLabel("Point Object / Image: Ø = 0.000 mm (Rendered as point marker in 3D viewer)")
        self.lbl_point_info.setStyleSheet("color: #0284c7; font-weight: bold;")
        self.lbl_point_info.setVisible(False)
        geom_layout.addWidget(self.lbl_point_info, 10, 0, 1, 2)

        # Circular params
        self.lbl_radius = QLabel("Semi-Diameter R (mm):")
        self.spin_radius = QDoubleSpinBox(geom_grp)
        self.spin_radius.setRange(0.0, 5000.0)
        self.spin_radius.setDecimals(3)
        self.spin_radius.setSingleStep(0.5)
        self.spin_radius.valueChanged.connect(self._on_values_changed)
        geom_layout.addWidget(self.lbl_radius, 1, 0)
        geom_layout.addWidget(self.spin_radius, 1, 1)

        self.lbl_diameter_info = QLabel("Full Diameter Ø = 20.000 mm")
        self.lbl_diameter_info.setStyleSheet("color: #1565c0; font-weight: bold;")
        geom_layout.addWidget(self.lbl_diameter_info, 2, 1)

        # Rectangular params (X-half, Y-half)
        self.lbl_x_half = QLabel("Half-Width X (mm):")
        self.spin_x_half = QDoubleSpinBox(geom_grp)
        self.spin_x_half.setRange(0.001, 5000.0)
        self.spin_x_half.setDecimals(3)
        self.spin_x_half.valueChanged.connect(self._on_values_changed)
        geom_layout.addWidget(self.lbl_x_half, 3, 0)
        geom_layout.addWidget(self.spin_x_half, 3, 1)

        self.lbl_y_half = QLabel("Half-Height Y (mm):")
        self.spin_y_half = QDoubleSpinBox(geom_grp)
        self.spin_y_half.setRange(0.001, 5000.0)
        self.spin_y_half.setDecimals(3)
        self.spin_y_half.valueChanged.connect(self._on_values_changed)
        geom_layout.addWidget(self.lbl_y_half, 4, 0)
        geom_layout.addWidget(self.spin_y_half, 4, 1)

        self.lbl_rect_info = QLabel("Full Slit: 20.000 × 15.000 mm")
        self.lbl_rect_info.setStyleSheet("color: #1565c0; font-weight: bold;")
        geom_layout.addWidget(self.lbl_rect_info, 5, 1)

        # Annular params (Rin, Rout)
        self.lbl_r_inner = QLabel("Inner Radius Rin (mm):")
        self.spin_r_inner = QDoubleSpinBox(geom_grp)
        self.spin_r_inner.setRange(0.0, 5000.0)
        self.spin_r_inner.setDecimals(3)
        self.spin_r_inner.valueChanged.connect(self._on_values_changed)
        geom_layout.addWidget(self.lbl_r_inner, 6, 0)
        geom_layout.addWidget(self.spin_r_inner, 6, 1)

        self.lbl_r_outer = QLabel("Outer Radius Rout (mm):")
        self.spin_r_outer = QDoubleSpinBox(geom_grp)
        self.spin_r_outer.setRange(0.01, 5000.0)
        self.spin_r_outer.setDecimals(3)
        self.spin_r_outer.valueChanged.connect(self._on_values_changed)
        geom_layout.addWidget(self.lbl_r_outer, 7, 0)
        geom_layout.addWidget(self.spin_r_outer, 7, 1)

        # Spider params
        self.lbl_spider_hub = QLabel("Central Hub Radius (mm):")
        self.spin_spider_hub = QDoubleSpinBox(geom_grp)
        self.spin_spider_hub.setRange(0.0, 500.0)
        self.spin_spider_hub.setDecimals(3)
        self.spin_spider_hub.valueChanged.connect(self._on_values_changed)
        geom_layout.addWidget(self.lbl_spider_hub, 8, 0)
        geom_layout.addWidget(self.spin_spider_hub, 8, 1)

        self.lbl_spider_legs = QLabel("Number of Struts / Legs:")
        self.spin_spider_legs = QSpinBox(geom_grp)
        self.spin_spider_legs.setRange(2, 8)
        self.spin_spider_legs.setValue(4)
        self.spin_spider_legs.valueChanged.connect(self._on_values_changed)
        geom_layout.addWidget(self.lbl_spider_legs, 9, 0)
        geom_layout.addWidget(self.spin_spider_legs, 9, 1)

        layout.addWidget(geom_grp)

        # 3. Action Buttons
        btn_layout = QHBoxLayout()
        self.btn_apply = QPushButton("Apply to Surface", self)
        self.btn_apply.clicked.connect(self.apply_changes)
        btn_layout.addWidget(self.btn_apply)

        self.btn_ok = QPushButton("OK", self)
        self.btn_ok.setDefault(True)
        self.btn_ok.clicked.connect(self._on_ok)
        btn_layout.addWidget(self.btn_ok)

        self.btn_cancel = QPushButton("Cancel", self)
        self.btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(self.btn_cancel)

        layout.addLayout(btn_layout)

    def _load_from_surface(self):
        """Populate controls from current surface attributes."""
        self._syncing = True
        try:
            surf = self.surface
            stype = str(getattr(surf, "surface_type", "")).upper()
            is_stop = (
                "STOP" in stype
                or getattr(surf, "is_stop", False)
                or getattr(surf, "is_aperture_stop", False)
                or "STOP" in (surf.name or "").upper()
                or "(STO)" in (surf.name or "").upper()
            )
            self.chk_is_stop.setChecked(is_stop)

            sd = float(getattr(surf, "semi_diameter", 10.0))
            self.spin_radius.setValue(sd)

            ap_type = str(getattr(surf, "aperture_type", "Circular")).capitalize()
            if ap_type == "Point" or (sd <= 0.0 and (self.surface_idx == 0 or "OBJECT" in stype or "DETECTOR" in stype)):
                self.combo_type.setCurrentIndex(5)
            elif ap_type == "Rectangular":
                self.combo_type.setCurrentIndex(1)
            elif ap_type == "Annular":
                self.combo_type.setCurrentIndex(2)
            elif ap_type == "Elliptical":
                self.combo_type.setCurrentIndex(3)
            elif ap_type == "Spider":
                self.combo_type.setCurrentIndex(4)
            else:
                self.combo_type.setCurrentIndex(0)

            self.spin_x_half.setValue(float(getattr(surf, "aperture_x_half", sd)))
            self.spin_y_half.setValue(float(getattr(surf, "aperture_y_half", sd * 0.75)))
            self.spin_r_inner.setValue(float(getattr(surf, "aperture_r_inner", 0.0)))
            self.spin_r_outer.setValue(float(getattr(surf, "aperture_r_outer", sd)))
            self.spin_spider_hub.setValue(float(getattr(surf, "aperture_center_radius", 3.0)))
            self.spin_spider_legs.setValue(int(getattr(surf, "aperture_spider_count", 4)))

            self._update_visibility()
        finally:
            self._syncing = False

    def _update_visibility(self):
        """Show/hide parameter widgets according to selected aperture shape."""
        idx = self.combo_type.currentIndex()

        # Point (idx == 5)
        is_point = (idx == 5)
        if hasattr(self, 'lbl_point_info'):
            self.lbl_point_info.setVisible(is_point)

        # Circular
        show_circ = (idx in (0, 3))
        self.lbl_radius.setVisible(show_circ)
        self.spin_radius.setVisible(show_circ)
        self.lbl_diameter_info.setVisible(show_circ)
        if idx == 3:
            self.lbl_radius.setText("Semi-Major Axis A (mm):")
        else:
            self.lbl_radius.setText("Semi-Diameter R (mm):")

        # Rectangular / Elliptical
        show_rect = (idx in (1, 3))
        self.lbl_x_half.setVisible(show_rect)
        self.spin_x_half.setVisible(show_rect)
        self.lbl_y_half.setVisible(show_rect)
        self.spin_y_half.setVisible(show_rect)
        self.lbl_rect_info.setVisible(show_rect)
        if idx == 3:
            self.lbl_x_half.setText("Semi-Major A (mm):")
            self.lbl_y_half.setText("Semi-Minor B (mm):")
        else:
            self.lbl_x_half.setText("Half-Width X (mm):")
            self.lbl_y_half.setText("Half-Height Y (mm):")

        # Annular
        show_annular = (idx == 2)
        self.lbl_r_inner.setVisible(show_annular)
        self.spin_r_inner.setVisible(show_annular)
        self.lbl_r_outer.setVisible(show_annular)
        self.spin_r_outer.setVisible(show_annular)

        # Spider
        show_spider = (idx == 4)
        self.lbl_spider_hub.setVisible(show_spider)
        self.spin_spider_hub.setVisible(show_spider)
        self.lbl_spider_legs.setVisible(show_spider)
        self.spin_spider_legs.setVisible(show_spider)

    def _on_type_changed(self):
        self._update_visibility()
        self._on_values_changed()

    def set_dark_mode(self, enabled: bool) -> None:
        """Switch between dark and light background themes."""
        self.dark_mode = bool(enabled)
        self._update_theme_styles()

    def _update_theme_styles(self) -> None:
        """Update styled elements to match the current theme."""
        if not QT_AVAILABLE:
            return
        is_dark = getattr(self, 'dark_mode', False)
        primary_col = "#38bdf8" if is_dark else "#0284c7"
        sec_col = "#60a5fa" if is_dark else "#1565c0"
        dim_col = "#94a3b8" if is_dark else "#555555"
        stop_col = "#f87171" if is_dark else "#b71c1c"

        if hasattr(self, 'lbl_point_info') and self.lbl_point_info:
            self.lbl_point_info.setStyleSheet(f"color: {primary_col}; font-weight: bold;")
        if hasattr(self, 'lbl_diameter_info') and self.lbl_diameter_info:
            self.lbl_diameter_info.setStyleSheet(f"color: {sec_col}; font-weight: bold;")
        if hasattr(self, 'lbl_rect_info') and self.lbl_rect_info:
            self.lbl_rect_info.setStyleSheet(f"color: {sec_col}; font-weight: bold;")
        if hasattr(self, 'lbl_stop_desc') and self.lbl_stop_desc:
            if hasattr(self, 'chk_is_stop') and self.chk_is_stop.isChecked():
                self.lbl_stop_desc.setStyleSheet(f"color: {stop_col}; font-weight: bold;")
            else:
                self.lbl_stop_desc.setStyleSheet(f"color: {dim_col}; font-size: 11px;")

    def _on_stop_toggled(self, checked: bool):
        if checked:
            self.lbl_stop_desc.setText(
                "🔴 Surface acts as the Aperture Stop (STO): directly limits beam marginal ray angles."
            )
        else:
            self.lbl_stop_desc.setText(
                "Standard optical surface: aperture acts as clear boundary or mechanical baffle."
            )
        self._update_theme_styles()

    def _on_values_changed(self):
        if self._syncing:
            return
        r = self.spin_radius.value()
        self.lbl_diameter_info.setText(f"Semi-Diameter R = {r:.3f} mm  |  Full Diameter Ø = {2.0 * r:.3f} mm")

        hx = self.spin_x_half.value()
        hy = self.spin_y_half.value()
        self.lbl_rect_info.setText(f"Semi-D (Half-Span): X = {hx:.3f}, Y = {hy:.3f} mm  |  Full Slit: {2.0 * hx:.3f} × {2.0 * hy:.3f} mm")

    def apply_changes(self) -> Dict[str, Any]:
        """Commit dialog values directly to the surface and sync LDE model."""
        surf = self.surface
        is_stop = self.chk_is_stop.isChecked()
        idx = self.combo_type.currentIndex()

        if is_stop:
            surf.surface_type = SurfaceType.STOP
            surf.is_stop = True
            surf.is_aperture_stop = True
            if "(STO)" not in surf.name:
                base = surf.name.replace(" (STO)", "").replace("(STO)", "").strip()
                if base.startswith("S") and len(base) <= 3:
                    surf.name = f"S{self.surface_idx} (STO)"
                else:
                    surf.name = f"{base} (STO)"
        else:
            if surf.surface_type == SurfaceType.STOP:
                surf.surface_type = SurfaceType.STANDARD
            surf.is_stop = False
            surf.is_aperture_stop = False
            if "(STO)" in surf.name:
                surf.name = surf.name.replace(" (STO)", "").replace("(STO)", "").strip()

        type_names = ["Circular", "Rectangular", "Annular", "Elliptical", "Spider", "Point"]
        ap_type = type_names[idx]
        surf.aperture_type = ap_type

        if ap_type == "Point":
            surf.semi_diameter = 0.0
            surf.aperture_x_half = 0.0
            surf.aperture_y_half = 0.0
            surf.aperture_r_inner = 0.0
            surf.aperture_r_outer = 0.0
        elif ap_type == "Circular":
            surf.semi_diameter = float(self.spin_radius.value())
            surf.aperture_r_inner = 0.0
            surf.aperture_r_outer = surf.semi_diameter
        elif ap_type == "Rectangular":
            surf.aperture_x_half = float(self.spin_x_half.value())
            surf.aperture_y_half = float(self.spin_y_half.value())
            surf.semi_diameter = float(np.sqrt(surf.aperture_x_half**2 + surf.aperture_y_half**2))
        elif ap_type == "Annular":
            surf.aperture_r_inner = float(self.spin_r_inner.value())
            surf.aperture_r_outer = float(self.spin_r_outer.value())
            surf.semi_diameter = surf.aperture_r_outer
        elif ap_type == "Elliptical":
            surf.aperture_a_semi = float(self.spin_x_half.value())
            surf.aperture_b_semi = float(self.spin_y_half.value())
            surf.semi_diameter = max(surf.aperture_a_semi, surf.aperture_b_semi)
        elif ap_type == "Spider":
            surf.aperture_center_radius = float(self.spin_spider_hub.value())
            surf.aperture_spider_count = int(self.spin_spider_legs.value())
            surf.semi_diameter = float(self.spin_radius.value())

        # Sync LDE table model if provided
        if self.model is not None:
            if hasattr(self.model, "_data") and 0 <= self.surface_idx < len(self.model._data):
                stype_name = surf.surface_type.value if hasattr(surf.surface_type, "value") else str(surf.surface_type)
                self.model._data[self.surface_idx][0] = surf.name
                self.model._data[self.surface_idx][1] = stype_name
                self.model._data[self.surface_idx][5] = float(surf.semi_diameter)
                if len(self.model._data[self.surface_idx]) > 14:
                    self.model._data[self.surface_idx][14] = str(surf.aperture_type)
                if len(self.model._data[self.surface_idx]) > 15:
                    self.model._data[self.surface_idx][15] = (
                        surf.get_aperture_params_str() if hasattr(surf, "get_aperture_params_str") else f"R={surf.semi_diameter}"
                    )
            if hasattr(self.model, "_sync_stack_from_data"):
                try:
                    self.model._sync_stack_from_data()
                except Exception:
                    pass
            if hasattr(self.model, "dataChanged"):
                try:
                    idx0 = self.model.index(self.surface_idx, 0)
                    idx_end = self.model.index(self.surface_idx, len(self.model.COLUMNS) - 1)
                    self.model.dataChanged.emit(idx0, idx_end)
                except Exception:
                    pass
            if hasattr(self.model, "system_updated") and self.model.system_updated is not None:
                try:
                    self.model.system_updated.emit()
                except Exception:
                    pass

        return {
            "is_stop": is_stop,
            "aperture_type": ap_type,
            "semi_diameter": surf.semi_diameter,
            "name": surf.name,
        }

    def _on_ok(self):
        self.apply_changes()
        self.accept()
