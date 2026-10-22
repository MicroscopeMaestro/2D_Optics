"""
Zemax System Configuration & Explorer Dock Widget.
Provides interactive panels for:
1. Aperture (EPD, F/#, Object Space NA, Apodization)
2. Wavelength Data Editor (Multi-wavelength table, weights, primary selector, optical presets)
3. Field Data Editor (Angle/Height fields, multi-field grid presets)
4. Environment & Units (Lens units, temperature, pressure)
5. Ray Aiming & Pupil Sampling
"""

from typing import Optional, Dict, Any, List
import math
import numpy as np

import os
from optics_app.gui.qt_compat import (
    QWidget, QDockWidget, QTabWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QComboBox, QDoubleSpinBox, QSpinBox, QCheckBox, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QRadioButton, QButtonGroup,
    Signal, pyqtSignal, QT_AVAILABLE, DummySignal, Qt, QScrollArea,
    QFileDialog, QMessageBox
)

from optics_app.engine.system_config import (
    SystemConfig, ApertureType, ApodizationType, FieldType,
    RayPatternType, WavelengthEntry, FieldPoint
)


class SystemConfigDockWidget(QDockWidget if QT_AVAILABLE else object):
    """
    Zemax System Explorer Dockable Panel.
    """
    if QT_AVAILABLE:
        config_changed = pyqtSignal()
        system_updated = pyqtSignal()
    else:
        config_changed = DummySignal()
        system_updated = DummySignal()

    def __init__(self, config: Optional[SystemConfig] = None, parent=None):
        if QT_AVAILABLE:
            super().__init__("System Explorer / Configuration", parent)
            self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        
        self.config = config if config is not None else SystemConfig()
        self._block_signals = False
        self.dark_mode: bool = False

        self._init_ui()
        self._update_theme_styles()

    def set_dark_mode(self, enabled: bool) -> None:
        """Switch between dark and light background themes."""
        self.dark_mode = bool(enabled)
        self._update_theme_styles()

    def _update_theme_styles(self) -> None:
        """Update styled elements to match the current theme."""
        if not QT_AVAILABLE:
            return
        is_dark = getattr(self, 'dark_mode', False)
        if hasattr(self, 'lbl_ap_info') and self.lbl_ap_info:
            if is_dark:
                self.lbl_ap_info.setStyleSheet(
                    "color: #cbd5e1; font-size: 11px; font-style: italic; padding: 4px; "
                    "background-color: #1e293b; border-radius: 3px; border: 1px solid #334155;"
                )
            else:
                self.lbl_ap_info.setStyleSheet(
                    "color: #455A64; font-size: 11px; font-style: italic; padding: 4px; "
                    "background-color: #ECEFF1; border-radius: 3px;"
                )

    def _init_ui(self):
        if not QT_AVAILABLE:
            return

        container = QWidget()
        main_layout = QVBoxLayout(container)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(6)

        # Tab Widget for System Explorer Categories
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # 1. Aperture Tab
        self.tab_aperture = QWidget()
        self._setup_aperture_tab()
        self.tabs.addTab(self.tab_aperture, "Aperture")

        # 2. Wavelengths Tab (Wavelength Data Editor)
        self.tab_wavelengths = QWidget()
        self._setup_wavelengths_tab()
        self.tabs.addTab(self.tab_wavelengths, "Wavelengths")

        # 3. Fields Tab (Field Data Editor)
        self.tab_fields = QWidget()
        self._setup_fields_tab()
        self.tabs.addTab(self.tab_fields, "Fields")

        # 4. Environment & Sampling Tab
        self.tab_env = QWidget()
        self._setup_env_tab()
        self.tabs.addTab(self.tab_env, "Environment")


        if QT_AVAILABLE:
            scroll_area = QScrollArea(self)
            if hasattr(scroll_area, "setWidgetResizable"):
                scroll_area.setWidgetResizable(True)
            scroll_area.setWidget(container)
            if hasattr(scroll_area, "setFrameShape"):
                scroll_area.setFrameShape(getattr(QScrollArea, "NoFrame", 0))

            self.setWidget(scroll_area)
            self.setMinimumSize(250, 150)

    # -------------------------------------------------------------------------
    # 1. APERTURE TAB
    # -------------------------------------------------------------------------
    def _setup_aperture_tab(self):
        layout = QVBoxLayout(self.tab_aperture)
        form = QFormLayout()

        # Beam Mode (Collimated vs Diverging Cone)
        self.combo_beam_type = QComboBox()
        self.combo_beam_type.addItem("Diverging (Point / Slit / Raman Cone)", "Diverging (Point/Slit Source)")
        self.combo_beam_type.addItem("Collimated (Parallel Laser Beam)", "Collimated (Parallel Beam)")
        curr_beam = getattr(self.config, 'beam_type', 'Diverging (Point/Slit Source)')
        idx = self.combo_beam_type.findData(curr_beam)
        if idx >= 0:
            self.combo_beam_type.setCurrentIndex(idx)
        self.combo_beam_type.currentIndexChanged.connect(self._on_aperture_changed)
        form.addRow("Beam Mode:", self.combo_beam_type)

        # Aperture Type
        self.combo_ap_type = QComboBox()
        for ap in ApertureType:
            self.combo_ap_type.addItem(ap.value, ap)
        self.combo_ap_type.setCurrentText(self.config.aperture_type.value)
        self.combo_ap_type.currentIndexChanged.connect(self._on_aperture_changed)
        form.addRow("Aperture Type:", self.combo_ap_type)

        # Aperture Value
        self.spin_ap_val = QDoubleSpinBox()
        self.spin_ap_val.setRange(0.001, 10000.0)
        self.spin_ap_val.setDecimals(3)
        self.spin_ap_val.setValue(self.config.aperture_value)
        self.spin_ap_val.setSuffix(" mm")
        self.spin_ap_val.valueChanged.connect(self._on_aperture_changed)
        form.addRow("Aperture Value:", self.spin_ap_val)

        # Slit Width (µm) - For physical slit or laser spot extent
        self.spin_slit_width = QDoubleSpinBox()
        self.spin_slit_width.setRange(0.0, 5000.0)
        self.spin_slit_width.setDecimals(1)
        self.spin_slit_width.setValue(getattr(self.config, 'slit_width_um', 0.0))
        self.spin_slit_width.setSuffix(" µm")
        self.spin_slit_width.setToolTip("Physical width of entrance slit or focused laser spot (0.0 = ideal point source)")
        self.spin_slit_width.valueChanged.connect(self._on_aperture_changed)
        form.addRow("Slit / Spot Width:", self.spin_slit_width)

        # Number of Rays (Ray Density per Field/Wavelength)
        self.spin_num_rays = QSpinBox()
        self.spin_num_rays.setRange(3, 201)
        self.spin_num_rays.setSingleStep(2)
        self.spin_num_rays.setValue(getattr(self.config, 'num_rays', 9))
        self.spin_num_rays.setSuffix(" rays")
        self.spin_num_rays.setToolTip("Number of rays traced per field and wavelength across 2D/3D views (e.g. 5, 9, 15, 31, 51)")
        self.spin_num_rays.valueChanged.connect(self._on_aperture_changed)
        form.addRow("Number of Rays:", self.spin_num_rays)

        # Apodization Type
        self.combo_apod_type = QComboBox()
        for apod in ApodizationType:
            self.combo_apod_type.addItem(apod.value, apod)
        self.combo_apod_type.setCurrentText(self.config.apodization_type.value)
        self.combo_apod_type.currentIndexChanged.connect(self._on_aperture_changed)
        form.addRow("Apodization Type:", self.combo_apod_type)

        # Apodization Factor
        self.spin_apod_factor = QDoubleSpinBox()
        self.spin_apod_factor.setRange(0.0, 10.0)
        self.spin_apod_factor.setDecimals(2)
        self.spin_apod_factor.setValue(self.config.apodization_factor)
        self.spin_apod_factor.valueChanged.connect(self._on_aperture_changed)
        form.addRow("Apodization Factor:", self.spin_apod_factor)

        # Telecentric in Object Space
        self.chk_telecentric = QCheckBox("Telecentric in Object Space")
        self.chk_telecentric.setChecked(self.config.telecentric_object_space)
        self.chk_telecentric.toggled.connect(self._on_aperture_changed)
        form.addRow("", self.chk_telecentric)

        layout.addLayout(form)

        # Info guidance label
        self.lbl_ap_info = QLabel("💡 Tip: Select 'Diverging' for Raman point/slit emission (NA=0.22). Select 'Collimated' for parallel laser beams.")
        self.lbl_ap_info.setStyleSheet("color: #455A64; font-size: 11px; font-style: italic; padding: 4px; background-color: #ECEFF1; border-radius: 3px;")
        self.lbl_ap_info.setWordWrap(True)
        layout.addWidget(self.lbl_ap_info)

        layout.addStretch()

    def _on_aperture_changed(self):
        if self._block_signals:
            return
        if hasattr(self, 'combo_beam_type') and self.combo_beam_type.currentData():
            self.config.beam_type = self.combo_beam_type.currentData()
        self.config.aperture_type = self.combo_ap_type.currentData()
        self.config.aperture_value = float(self.spin_ap_val.value())
        if hasattr(self, 'spin_slit_width'):
            self.config.slit_width_um = float(self.spin_slit_width.value())
        if hasattr(self, 'spin_num_rays'):
            self.config.num_rays = int(self.spin_num_rays.value())
            if hasattr(self, 'spin_env_num_rays'):
                self.spin_env_num_rays.blockSignals(True)
                self.spin_env_num_rays.setValue(self.config.num_rays)
                self.spin_env_num_rays.blockSignals(False)
        self.config.apodization_type = self.combo_apod_type.currentData()
        self.config.apodization_factor = float(self.spin_apod_factor.value())
        self.config.telecentric_object_space = self.chk_telecentric.isChecked()

        is_collimated = (getattr(self.config, 'beam_type', '') == "Collimated (Parallel Beam)") or self.config.telecentric_object_space

        if is_collimated:
            self.spin_ap_val.setSuffix(" mm (Beam Ø)")
            if hasattr(self, 'lbl_ap_info'):
                self.lbl_ap_info.setText("📏 Collimated Mode: Launches parallel laser rays across the specified beam diameter.")
        elif self.config.aperture_type == ApertureType.OBJECT_SPACE_NA:
            self.spin_ap_val.setSuffix(" NA")
            if hasattr(self, 'lbl_ap_info'):
                self.lbl_ap_info.setText(f"🔆 Diverging Cone: Defined by Numerical Aperture (NA={self.config.aperture_value:.3f}, θ={math.degrees(math.asin(min(0.99, max(0.01, self.config.aperture_value)))):.1f}°).")
        elif self.config.aperture_type == ApertureType.OBJECT_CONE_ANGLE:
            self.spin_ap_val.setSuffix(" deg")
            if hasattr(self, 'lbl_ap_info'):
                self.lbl_ap_info.setText(f"🔆 Diverging Cone: Defined by half-angle (θ={self.config.aperture_value:.1f}°).")
        elif self.config.aperture_type in (ApertureType.IMAGE_SPACE_F_NUM, ApertureType.PARAXIAL_WORKING_F_NUM):
            self.spin_ap_val.setSuffix(" F/#")
        else:
            self.spin_ap_val.setSuffix(" mm (EPD)")
            if hasattr(self, 'lbl_ap_info'):
                self.lbl_ap_info.setText("🔆 Diverging Cone: Fills entrance pupil diameter at pupil plane.")

        self._emit_update()

    # -------------------------------------------------------------------------
    # 2. WAVELENGTHS TAB (Wavelength Data Editor)
    # -------------------------------------------------------------------------
    def _setup_wavelengths_tab(self):
        layout = QVBoxLayout(self.tab_wavelengths)

        # Preset Dropdown
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("Preset:"))
        self.combo_wl_preset = QComboBox()
        self.combo_wl_preset.addItems([
            "Select Preset...",
            "Visible (F, d, C)",
            "Ultrafast 800 nm (editable)",
            "NIR 785 nm Raman",
            "NIR 1064 nm Laser",
            "Telecom C-Band",
            "HeNe Laser (632.8 nm)",
            "UV (i, h, g)"
        ])
        self.combo_wl_preset.currentIndexChanged.connect(self._on_wl_preset_selected)
        preset_layout.addWidget(self.combo_wl_preset)
        layout.addLayout(preset_layout)

        # Table of Wavelengths
        self.table_wl = QTableWidget(0, 4)
        self.table_wl.setHorizontalHeaderLabels(["Wavelength (μm)", "Wavelength (nm)", "Weight", "Primary"])
        if hasattr(self.table_wl, 'horizontalHeader'):
            self.table_wl.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.table_wl)

        # Action Buttons
        btn_layout = QHBoxLayout()
        self.btn_add_wl = QPushButton("+ Add Wavelength")
        self.btn_add_wl.clicked.connect(self._on_add_wavelength)
        self.btn_remove_wl = QPushButton("- Remove Selected")
        self.btn_remove_wl.clicked.connect(self._on_remove_wavelength)
        btn_layout.addWidget(self.btn_add_wl)
        btn_layout.addWidget(self.btn_remove_wl)
        layout.addLayout(btn_layout)

        if hasattr(self.table_wl, 'cellChanged'):
            try:
                self.table_wl.cellChanged.connect(self._on_wl_cell_changed)
            except Exception:
                pass

        self._populate_wavelength_table()

    def _populate_wavelength_table(self):
        was_blocked = self._block_signals
        self._block_signals = True
        if hasattr(self.table_wl, 'blockSignals'):
            try:
                self.table_wl.blockSignals(True)
            except Exception:
                pass
        self.table_wl.setRowCount(len(self.config.wavelengths))
        self.wl_radio_group = QButtonGroup(self)

        for i, w in enumerate(self.config.wavelengths):
            # 1. Wavelength (um)
            item_um = QTableWidgetItem(f"{w.wavelength_um:.5f}")
            self.table_wl.setItem(i, 0, item_um)

            # 2. Wavelength (nm)
            item_nm = QTableWidgetItem(f"{w.wavelength_um * 1000.0:.2f}")
            self.table_wl.setItem(i, 1, item_nm)

            # 3. Weight
            item_wt = QTableWidgetItem(f"{w.weight:.2f}")
            self.table_wl.setItem(i, 2, item_wt)

            # 4. Primary Radio Button
            rad = QRadioButton()
            rad.setChecked(w.is_primary)
            rad.toggled.connect(lambda checked, idx=i: self._on_primary_wl_toggled(checked, idx))
            self.wl_radio_group.addButton(rad)
            cell_widget = QWidget()
            cell_layout = QHBoxLayout(cell_widget)
            cell_layout.addWidget(rad)
            cell_layout.setAlignment(Qt.AlignCenter)
            cell_layout.setContentsMargins(0, 0, 0, 0)
            self.table_wl.setCellWidget(i, 3, cell_widget)

        if hasattr(self.table_wl, 'blockSignals'):
            try:
                self.table_wl.blockSignals(False)
            except Exception:
                pass
        self._block_signals = was_blocked

    def _on_primary_wl_toggled(self, checked, idx):
        if checked and not self._block_signals:
            self.config.set_primary_wavelength(idx)
            self._emit_update()

    def _on_wl_cell_changed(self, row, col):
        if self._block_signals or row >= len(self.config.wavelengths):
            return
        try:
            item = self.table_wl.item(row, col)
            val = float(item.text())
            if col == 0:
                self.config.wavelengths[row].wavelength_um = val
                if hasattr(self.table_wl, 'blockSignals'):
                    self.table_wl.blockSignals(True)
                self.table_wl.item(row, 1).setText(f"{val*1000.0:.2f}")
                if hasattr(self.table_wl, 'blockSignals'):
                    self.table_wl.blockSignals(False)
            elif col == 1:
                um = val / 1000.0
                self.config.wavelengths[row].wavelength_um = um
                if hasattr(self.table_wl, 'blockSignals'):
                    self.table_wl.blockSignals(True)
                self.table_wl.item(row, 0).setText(f"{um:.5f}")
                if hasattr(self.table_wl, 'blockSignals'):
                    self.table_wl.blockSignals(False)
            elif col == 2:
                self.config.wavelengths[row].weight = val
            self._emit_update()
        except ValueError:
            pass

    def _on_add_wavelength(self):
        self.config.add_wavelength(0.5500, weight=1.0)
        self._populate_wavelength_table()
        self._emit_update()

    def _on_remove_wavelength(self):
        row = self.table_wl.currentRow()
        if row >= 0 and len(self.config.wavelengths) > 1:
            self.config.remove_wavelength(row)
            self._populate_wavelength_table()
            self._emit_update()

    def _on_wl_preset_selected(self, index):
        preset_name = self.combo_wl_preset.currentText()
        if preset_name != "Select Preset...":
            self.config.apply_wavelength_preset(preset_name)
            self._populate_wavelength_table()
            self._emit_update()

    # -------------------------------------------------------------------------
    # 3. FIELDS TAB (Field Data Editor)
    # -------------------------------------------------------------------------
    def _setup_fields_tab(self):
        layout = QVBoxLayout(self.tab_fields)

        # Field Type
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Field Type:"))
        self.combo_field_type = QComboBox()
        for ft in FieldType:
            self.combo_field_type.addItem(ft.value, ft)
        self.combo_field_type.setCurrentText(self.config.field_type.value)
        self.combo_field_type.currentIndexChanged.connect(self._on_field_type_changed)
        type_layout.addWidget(self.combo_field_type)
        layout.addLayout(type_layout)

        # Presets
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("Field Preset:"))
        self.combo_field_preset = QComboBox()
        self.combo_field_preset.addItems([
            "Select Preset...",
            "On-Axis Single Point",
            "3-Field Linear",
            "5-Field Cross",
            "9-Field Grid"
        ])
        self.combo_field_preset.currentIndexChanged.connect(self._on_field_preset_selected)
        preset_layout.addWidget(self.combo_field_preset)
        layout.addLayout(preset_layout)

        # Field Table
        self.table_fields = QTableWidget(0, 3)
        self.table_fields.setHorizontalHeaderLabels(["X Angle/Height", "Y Angle/Height", "Weight"])
        if hasattr(self.table_fields, 'horizontalHeader'):
            self.table_fields.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.table_fields)

        # Action Buttons
        btn_layout = QHBoxLayout()
        self.btn_add_fld = QPushButton("+ Add Field")
        self.btn_add_fld.clicked.connect(self._on_add_field)
        self.btn_remove_fld = QPushButton("- Remove Selected")
        self.btn_remove_fld.clicked.connect(self._on_remove_field)
        btn_layout.addWidget(self.btn_add_fld)
        btn_layout.addWidget(self.btn_remove_fld)
        layout.addLayout(btn_layout)

        if hasattr(self.table_fields, 'cellChanged'):
            try:
                self.table_fields.cellChanged.connect(self._on_field_cell_changed)
            except Exception:
                pass

        self._populate_fields_table()

    def _populate_fields_table(self):
        was_blocked = self._block_signals
        self._block_signals = True
        if hasattr(self.table_fields, 'blockSignals'):
            try:
                self.table_fields.blockSignals(True)
            except Exception:
                pass
        self.table_fields.setRowCount(len(self.config.fields))
        for i, f in enumerate(self.config.fields):
            self.table_fields.setItem(i, 0, QTableWidgetItem(f"{f.x:.3f}"))
            self.table_fields.setItem(i, 1, QTableWidgetItem(f"{f.y:.3f}"))
            self.table_fields.setItem(i, 2, QTableWidgetItem(f"{f.weight:.2f}"))
        if hasattr(self.table_fields, 'blockSignals'):
            try:
                self.table_fields.blockSignals(False)
            except Exception:
                pass
        self._block_signals = was_blocked

    def _on_field_type_changed(self):
        if self._block_signals:
            return
        self.config.field_type = self.combo_field_type.currentData()
        self._emit_update()

    def _on_field_cell_changed(self, row, col):
        if self._block_signals or row >= len(self.config.fields):
            return
        try:
            val = float(self.table_fields.item(row, col).text())
            if col == 0:
                self.config.fields[row].x = val
            elif col == 1:
                self.config.fields[row].y = val
            elif col == 2:
                self.config.fields[row].weight = val
            self._emit_update()
        except ValueError:
            pass

    def _on_add_field(self):
        self.config.add_field(0.0, 5.0, weight=1.0)
        self._populate_fields_table()
        self._emit_update()

    def _on_remove_field(self):
        row = self.table_fields.currentRow()
        if row >= 0 and len(self.config.fields) > 1:
            self.config.remove_field(row)
            self._populate_fields_table()
            self._emit_update()

    def _on_field_preset_selected(self, index):
        preset = self.combo_field_preset.currentText()
        if preset != "Select Preset...":
            self.config.apply_field_preset(preset)
            self._populate_fields_table()
            self._emit_update()

    # -------------------------------------------------------------------------
    # 4. ENVIRONMENT & SAMPLING TAB
    # -------------------------------------------------------------------------
    def _setup_env_tab(self):
        layout = QVBoxLayout(self.tab_env)
        form = QFormLayout()

        # Lens Units
        self.combo_units = QComboBox()
        self.combo_units.addItems(["Millimeters", "Centimeters", "Inches", "Meters"])
        self.combo_units.setCurrentText(self.config.lens_units)
        self.combo_units.currentIndexChanged.connect(self._on_env_changed)
        form.addRow("Lens Units:", self.combo_units)

        # Temperature
        self.spin_temp = QDoubleSpinBox()
        self.spin_temp.setRange(-100.0, 300.0)
        self.spin_temp.setValue(self.config.temperature_c)
        self.spin_temp.setSuffix(" °C")
        self.spin_temp.valueChanged.connect(self._on_env_changed)
        form.addRow("Temperature:", self.spin_temp)

        # Pressure
        self.spin_pressure = QDoubleSpinBox()
        self.spin_pressure.setRange(0.0, 100.0)
        self.spin_pressure.setValue(self.config.pressure_atm)
        self.spin_pressure.setSuffix(" ATM")
        self.spin_pressure.valueChanged.connect(self._on_env_changed)
        form.addRow("Pressure:", self.spin_pressure)

        # Ray Pattern
        self.combo_pattern = QComboBox()
        for p in RayPatternType:
            self.combo_pattern.addItem(p.value, p)
        self.combo_pattern.setCurrentText(self.config.ray_pattern.value)
        self.combo_pattern.currentIndexChanged.connect(self._on_env_changed)
        form.addRow("Pupil Sampling Pattern:", self.combo_pattern)

        # Pupil Rings
        self.spin_rings = QSpinBox()
        self.spin_rings.setRange(1, 30)
        self.spin_rings.setValue(self.config.pupil_rings)
        self.spin_rings.valueChanged.connect(self._on_env_changed)
        form.addRow("Pupil Rings Count:", self.spin_rings)

        # Number of 2D/3D Rays (Ray Density)
        self.spin_env_num_rays = QSpinBox()
        self.spin_env_num_rays.setRange(3, 201)
        self.spin_env_num_rays.setSingleStep(2)
        self.spin_env_num_rays.setValue(getattr(self.config, 'num_rays', 9))
        self.spin_env_num_rays.setSuffix(" rays")
        self.spin_env_num_rays.setToolTip("Number of rays traced per field and wavelength across 2D/3D views")
        self.spin_env_num_rays.valueChanged.connect(self._on_env_changed)
        form.addRow("Number of Rays (2D/3D):", self.spin_env_num_rays)

        layout.addLayout(form)
        layout.addStretch()

    def _on_env_changed(self):
        if self._block_signals:
            return
        self.config.lens_units = self.combo_units.currentText()
        self.config.temperature_c = float(self.spin_temp.value())
        self.config.pressure_atm = float(self.spin_pressure.value())
        self.config.ray_pattern = self.combo_pattern.currentData()
        self.config.pupil_rings = int(self.spin_rings.value())
        if hasattr(self, 'spin_env_num_rays'):
            self.config.num_rays = int(self.spin_env_num_rays.value())
            if hasattr(self, 'spin_num_rays'):
                self.spin_num_rays.blockSignals(True)
                self.spin_num_rays.setValue(self.config.num_rays)
                self.spin_num_rays.blockSignals(False)
        self._emit_update()


    def update_ui(self):
        self._block_signals = True
        try:
            # 1. Aperture & Beam Mode
            if hasattr(self, 'combo_beam_type'):
                curr_beam = getattr(self.config, 'beam_type', 'Diverging (Point/Slit Source)')
                idx = self.combo_beam_type.findData(curr_beam)
                if idx >= 0:
                    self.combo_beam_type.setCurrentIndex(idx)
            if hasattr(self, 'combo_ap_type'):
                self.combo_ap_type.setCurrentText(self.config.aperture_type.value)
            if hasattr(self, 'spin_ap_val'):
                self.spin_ap_val.setValue(self.config.aperture_value)
            if hasattr(self, 'spin_slit_width'):
                self.spin_slit_width.setValue(getattr(self.config, 'slit_width_um', 0.0))
            if hasattr(self, 'spin_num_rays'):
                self.spin_num_rays.setValue(getattr(self.config, 'num_rays', 9))
            if hasattr(self, 'combo_apod_type'):
                self.combo_apod_type.setCurrentText(self.config.apodization_type.value)
            if hasattr(self, 'spin_apod_factor'):
                self.spin_apod_factor.setValue(self.config.apodization_factor)
            if hasattr(self, 'chk_telecentric'):
                self.chk_telecentric.setChecked(self.config.telecentric_object_space)

            # 2. Wavelengths
            self._populate_wavelength_table()

            # 3. Fields
            if hasattr(self, 'combo_field_type'):
                self.combo_field_type.setCurrentText(self.config.field_type.value)
            self._populate_fields_table()

            # 4. Environment & Sampling
            if hasattr(self, 'combo_units'):
                self.combo_units.setCurrentText(self.config.lens_units)
            if hasattr(self, 'spin_temp'):
                self.spin_temp.setValue(self.config.temperature_c)
            if hasattr(self, 'spin_pressure'):
                self.spin_pressure.setValue(self.config.pressure_atm)
            if hasattr(self, 'combo_pattern'):
                self.combo_pattern.setCurrentText(self.config.ray_pattern.value)
            if hasattr(self, 'spin_rings'):
                self.spin_rings.setValue(self.config.pupil_rings)
            if hasattr(self, 'spin_env_num_rays'):
                self.spin_env_num_rays.setValue(getattr(self.config, 'num_rays', 9))
        finally:
            self._block_signals = False

    def sync_from_ui(self):
        if not QT_AVAILABLE:
            return
        if hasattr(self, 'combo_beam_type') and self.combo_beam_type.currentData():
            self.config.beam_type = self.combo_beam_type.currentData()
        if hasattr(self, 'combo_ap_type') and self.combo_ap_type.currentData():
            self.config.aperture_type = self.combo_ap_type.currentData()
        if hasattr(self, 'spin_ap_val'):
            self.config.aperture_value = float(self.spin_ap_val.value())
        if hasattr(self, 'spin_slit_width'):
            self.config.slit_width_um = float(self.spin_slit_width.value())
        if hasattr(self, 'spin_num_rays'):
            self.config.num_rays = int(self.spin_num_rays.value())
        if hasattr(self, 'combo_apod_type') and self.combo_apod_type.currentData():
            self.config.apodization_type = self.combo_apod_type.currentData()
        if hasattr(self, 'spin_apod_factor'):
            self.config.apodization_factor = float(self.spin_apod_factor.value())
        if hasattr(self, 'chk_telecentric'):
            self.config.telecentric_object_space = self.chk_telecentric.isChecked()
        if hasattr(self, 'combo_units'):
            self.config.lens_units = self.combo_units.currentText()
        if hasattr(self, 'spin_temp'):
            self.config.temperature_c = float(self.spin_temp.value())
        if hasattr(self, 'spin_pressure'):
            self.config.pressure_atm = float(self.spin_pressure.value())
        if hasattr(self, 'combo_pattern') and self.combo_pattern.currentData():
            self.config.ray_pattern = self.combo_pattern.currentData()
        if hasattr(self, 'spin_rings'):
            self.config.pupil_rings = int(self.spin_rings.value())
        if hasattr(self, 'spin_env_num_rays'):
            self.config.num_rays = int(self.spin_env_num_rays.value())
        
    def save_config_file(self, filepath: Optional[str] = None) -> Optional[str]:
        """Save current system explorer parameters to a JSON file."""
        self.sync_from_ui()
        if not filepath:
            if not QT_AVAILABLE:
                return None
            filepath, _ = QFileDialog.getSaveFileName(
                self,
                "Save System Explorer Configuration",
                "system_config.json",
                "JSON Files (*.json);;All Files (*)"
            )
        if not filepath:
            return None
        self.config.save_to_file(filepath)
        return filepath

    def load_config_file(self, filepath: Optional[str] = None) -> Optional[str]:
        """Load system explorer parameters from a JSON file and update UI & system."""
        if not filepath:
            if not QT_AVAILABLE:
                return None
            filepath, _ = QFileDialog.getOpenFileName(
                self,
                "Load System Explorer Configuration",
                "",
                "JSON Files (*.json);;All Files (*)"
            )
        if not filepath or not os.path.exists(filepath):
            return None
        self.config.load_from_file(filepath)
        self.update_ui()
        self._emit_update()
        return filepath

    def _emit_update(self):
        if hasattr(self, 'config_changed') and self.config_changed is not None:
            try:
                self.config_changed.emit()
            except Exception:
                pass
        if hasattr(self, 'system_updated') and self.system_updated is not None:
            try:
                self.system_updated.emit()
            except Exception:
                pass
