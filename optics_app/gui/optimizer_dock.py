"""
Optimizer Dock Widget for 2D Optics Simulation & Design Studio.
Provides UI controls for designating variable surface thicknesses, selecting optimization algorithms,
configuring iteration limits, triggering optimization, and viewing real-time status and convergence progress.
"""

from typing import List, Dict, Optional, Any
import os
import sys
import math
import numpy as np
from optics_app.gui.qt_compat import (
    QT_AVAILABLE, QtWidgets, QWidget, QDockWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QSpinBox, QCheckBox, QProgressBar,
    QDoubleSpinBox, QMessageBox, pyqtSignal, DummySignal, Qt, QScrollArea, QLineEdit,
    QGroupBox, QGridLayout
)

from optics_app.gui.lde_table import LensDataTableModel
from optics_app.engine.diagnostics import OptimizationValidator, ValidationResult, DiagnosticSeverity


class VariableIndices(dict):
    """Dict containing 'thickness' and 'radius' variable lists, supporting integer 'in' checks, len, and equality."""
    def __contains__(self, item):
        if isinstance(item, (int, np.integer)):
            return item in self.get("thickness", []) or item in self.get("radius", [])
        return super().__contains__(item)

    def __len__(self):
        return len(self.get("thickness", [])) + len(self.get("radius", []))

    def __iter__(self):
        return iter(sorted(list(set(self.get("thickness", []) + self.get("radius", [])))))

    def __eq__(self, other):
        if isinstance(other, list):
            return sorted(list(self)) == sorted(other)
        return super().__eq__(other)

    def count(self, item):
        return list(self).count(item)


_CompatibleVariableIndices = VariableIndices


class _SignalAdapter:
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

    def emit(self, *args, **kwargs):
        for s in list(self._slots):
            for n_args in (None, 10, 9, 8, 7, 3, 1):
                try:
                    if n_args is None:
                        s(*args, **kwargs)
                    else:
                        s(*args[:n_args])
                    break
                except TypeError:
                    continue
                except Exception:
                    break


class OptimizerDockWidget(QDockWidget if QT_AVAILABLE else object):
    """
    Interactive Dock Widget for Optical Parameter Optimization.
    Allows user to select which surface thicknesses are variable, choose solver algorithm,
    set iteration limits, and trigger automated optimization with live GUI synchronization.
    """
    if QT_AVAILABLE:
        optimize_requested = pyqtSignal(object, str, int, str, int, float, bool, int, float, object)
        open_system_explorer_requested = pyqtSignal(int)
        fit_test_plates_requested = pyqtSignal()
    else:
        optimize_requested = None
        open_system_explorer_requested = None
        fit_test_plates_requested = None

    def __init__(self, model: Optional[LensDataTableModel] = None, parent=None):
        if QT_AVAILABLE:
            super().__init__("Optical Optimizer", parent)
            self.setObjectName("OptimizerDockWidget")
            self.setAllowedAreas(Qt.AllDockWidgetAreas)
            self.optimize_requested = _SignalAdapter(getattr(self, 'optimize_requested', None))
            self.open_system_explorer_requested = _SignalAdapter(getattr(self, 'open_system_explorer_requested', None))
            self.fit_test_plates_requested = _SignalAdapter(getattr(self, 'fit_test_plates_requested', None))
        else:
            self.optimize_requested = _SignalAdapter(DummySignal())
            self.open_system_explorer_requested = _SignalAdapter(DummySignal())
            self.fit_test_plates_requested = _SignalAdapter(DummySignal())

        self.model: LensDataTableModel = model or LensDataTableModel()
        self._system_config: Optional[Any] = None
        self.checkboxes = {}        # {index: thick_cb}
        self.rad_checkboxes = {}    # {index: rad_cb}
        self.selected_indices = []  # thickness indices
        self.selected_rad_indices = [] # radius indices
        self.last_result: Optional[Any] = None
        self.dark_mode: bool = False

        self._setup_ui()
        self.update_system_info()

    def _setup_ui(self):
        """Construct dock widget UI components and layout."""
        if QT_AVAILABLE:
            container = QWidget(self)
            layout = QVBoxLayout(container)
            layout.addWidget(QLabel("<b>Optimization Variables:</b> (Lens Data Editor)"))
            
            # Active Variables Summary Badge (SSOT from LDE)
            self.lbl_active_variables = QLabel("Checking variables...", container)
            self.lbl_active_variables.setWordWrap(True)
            layout.addWidget(self.lbl_active_variables)

            # Zemax Preset Action Buttons
            btn_preset_layout = QHBoxLayout()
            self.btn_zemax_def = QPushButton("Zemax Defaults", container)
            self.btn_zemax_def.setToolTip("Set all air gap distances Variable (V) and all solid lenses Fixed (F)")
            self.btn_zemax_def.clicked.connect(self.set_zemax_defaults)
            btn_preset_layout.addWidget(self.btn_zemax_def)

            self.btn_clear_vars = QPushButton("Clear All (Fixed)", container)
            self.btn_clear_vars.setToolTip("Set all surface parameters to Fixed (F)")
            self.btn_clear_vars.clicked.connect(self.clear_all_variables)
            btn_preset_layout.addWidget(self.btn_clear_vars)
            layout.addLayout(btn_preset_layout)

            self.surfaces_container = QWidget(container)
            self.surfaces_layout = QVBoxLayout(self.surfaces_container)
            self.surfaces_layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(self.surfaces_container)

            self._rebuild_surface_checkboxes()

            layout.addWidget(QLabel("<b>Algorithm:</b>"))
            self.method_combo = QComboBox(container)
            self.method_combo.addItems([
                "Nelder-Mead",
                "Damped Least Squares (DLS)",
                "Hammer Optimizer (Global)",
                "Global Search (Multi-Start)",
                "Monte Carlo Optimization",
                "L-BFGS-B"
            ])
            layout.addWidget(self.method_combo)

            layout.addWidget(QLabel("<b>Target Surface:</b>"))
            self.target_surf_combo = QComboBox(container)
            self.target_surf_combo.addItem("Detector (Last Surface)", -1)
            layout.addWidget(self.target_surf_combo)

            layout.addWidget(QLabel("<b>Target Metric:</b>"))
            self.target_combo = QComboBox(container)
            self.target_combo.addItems([
                "Spot Size (Focus)",
                "Spot Size + Constrain EFL (Focus @ Target EFL)",
                "Aberration Reduction (Seidel S1-S5 + Color)",
                "Wavefront Error (RMS OPD)",
                "Achromatic Focus (Min Spot + Min Color)",
                "RMS Spot X (Spectral Focus)",
                "RMS Spot Y (Spatial Focus)",
                "Composite RMS Spot X + Y",
                "Target Spot Size",
                "Angular Spread (Collimation)",
                "Telecentricity",
                "Spectrometer Spread"
            ])
            self.target_combo.currentTextChanged.connect(self._on_target_changed)
            layout.addWidget(self.target_combo)

            self.target_spot_label = QLabel("<b>Target Spot Radius (µm):</b>")
            layout.addWidget(self.target_spot_label)
            self.target_spot_spin = QDoubleSpinBox(container)
            self.target_spot_spin.setRange(0.0, 10000.0)
            self.target_spot_spin.setDecimals(1)
            self.target_spot_spin.setValue(0.0)
            layout.addWidget(self.target_spot_spin)
            self.target_spot_label.hide()
            self.target_spot_spin.hide()

            # EFL Constraint Controls
            efl_box = QHBoxLayout()
            self.chk_constrain_efl = QCheckBox("Constrain EFL", container)
            self.chk_constrain_efl.setChecked(False)
            self.chk_constrain_efl.setToolTip("Enforce an exact target Effective Focal Length during optimization")
            self.chk_constrain_efl.toggled.connect(self._on_constrain_efl_toggled)
            efl_box.addWidget(self.chk_constrain_efl)

            self.target_efl_spin = QDoubleSpinBox(container)
            self.target_efl_spin.setRange(0.1, 10000.0)
            self.target_efl_spin.setValue(50.0)
            self.target_efl_spin.setDecimals(2)
            self.target_efl_spin.setSuffix(" mm")
            self.target_efl_spin.setToolTip("Target system Effective Focal Length (EFL)")
            self.target_efl_spin.setEnabled(False)
            efl_box.addWidget(self.target_efl_spin)
            layout.addLayout(efl_box)

            # Realistic Manufacturing Constraints Group (Zemax MNCG / MNEG / MNCA / MNEA / TFLT)
            self.grp_mfg = QGroupBox("Manufacturing Constraints (ISO 10110 / Zemax)", container)
            self.grp_mfg.setCheckable(True)
            self.grp_mfg.setChecked(True)
            self.grp_mfg.setToolTip("Enforces realistic glass center thickness (MNCG/MXCG), minimum edge thickness (MNEG), air edge clearance (MNEA), aspect ratio limits, and working distance.")
            mfg_layout = QGridLayout(self.grp_mfg)
            mfg_layout.setContentsMargins(6, 6, 6, 6)
            mfg_layout.setSpacing(4)

            mfg_layout.addWidget(QLabel("Preset:"), 0, 0)
            self.combo_mfg_preset = QComboBox(self.grp_mfg)
            self.combo_mfg_preset.addItems([
                "Commercial (ISO 10110 Standard)",
                "Precision (Research / Imaging)",
                "High-Precision (Laser / Metrology)",
                "Custom"
            ])
            self.combo_mfg_preset.currentTextChanged.connect(self._on_preset_changed)
            mfg_layout.addWidget(self.combo_mfg_preset, 0, 1, 1, 3)

            mfg_layout.addWidget(QLabel("Min Glass:"), 1, 0)
            self.spin_min_glass = QDoubleSpinBox(self.grp_mfg)
            self.spin_min_glass.setRange(0.2, 50.0)
            self.spin_min_glass.setValue(1.5)
            self.spin_min_glass.setDecimals(1)
            self.spin_min_glass.setSuffix(" mm")
            self.spin_min_glass.setToolTip("Minimum glass center thickness (MNCG). Prevents lenses from becoming razor-thin.")
            self.spin_min_glass.valueChanged.connect(self._on_mfg_spin_changed)
            mfg_layout.addWidget(self.spin_min_glass, 1, 1)

            mfg_layout.addWidget(QLabel("Max Glass:"), 1, 2)
            self.spin_max_glass = QDoubleSpinBox(self.grp_mfg)
            self.spin_max_glass.setRange(2.0, 150.0)
            self.spin_max_glass.setValue(18.0)
            self.spin_max_glass.setDecimals(1)
            self.spin_max_glass.setSuffix(" mm")
            self.spin_max_glass.setToolTip("Maximum glass center thickness (MXCG). Prevents lenses from ballooning into heavy cylinders.")
            self.spin_max_glass.valueChanged.connect(self._on_mfg_spin_changed)
            mfg_layout.addWidget(self.spin_max_glass, 1, 3)

            mfg_layout.addWidget(QLabel("Min Edge:"), 2, 0)
            self.spin_min_edge = QDoubleSpinBox(self.grp_mfg)
            self.spin_min_edge.setRange(0.1, 20.0)
            self.spin_min_edge.setValue(1.0)
            self.spin_min_edge.setDecimals(1)
            self.spin_min_edge.setSuffix(" mm")
            self.spin_min_edge.setToolTip("Minimum glass edge thickness (MNEG). Prevents knife-edge chipping and self-intersecting surfaces.")
            self.spin_min_edge.valueChanged.connect(self._on_mfg_spin_changed)
            mfg_layout.addWidget(self.spin_min_edge, 2, 1)

            mfg_layout.addWidget(QLabel("Min Air:"), 2, 2)
            self.spin_min_air = QDoubleSpinBox(self.grp_mfg)
            self.spin_min_air.setRange(0.1, 50.0)
            self.spin_min_air.setValue(0.5)
            self.spin_min_air.setDecimals(1)
            self.spin_min_air.setSuffix(" mm")
            self.spin_min_air.setToolTip("Minimum axial air gap (MNCA) & rim clearance (MNEA). Prevents elements from colliding.")
            self.spin_min_air.valueChanged.connect(self._on_mfg_spin_changed)
            mfg_layout.addWidget(self.spin_min_air, 2, 3)

            mfg_layout.addWidget(QLabel("Min BFL:"), 3, 0)
            self.spin_min_bfl = QDoubleSpinBox(self.grp_mfg)
            self.spin_min_bfl.setRange(0.5, 200.0)
            self.spin_min_bfl.setValue(10.0)
            self.spin_min_bfl.setDecimals(1)
            self.spin_min_bfl.setSuffix(" mm")
            self.spin_min_bfl.setToolTip("Minimum working distance / back focal length to detector.")
            self.spin_min_bfl.valueChanged.connect(self._on_mfg_spin_changed)
            mfg_layout.addWidget(self.spin_min_bfl, 3, 1)

            mfg_layout.addWidget(QLabel("Max Aspect:"), 3, 2)
            self.spin_max_aspect = QDoubleSpinBox(self.grp_mfg)
            self.spin_max_aspect.setRange(2.0, 30.0)
            self.spin_max_aspect.setValue(10.0)
            self.spin_max_aspect.setDecimals(1)
            self.spin_max_aspect.setSuffix(" D/t")
            self.spin_max_aspect.setToolTip("Maximum aspect ratio (Diameter / Center Thickness). ISO 10110 recommends D/t <= 10 to avoid polishing flexure.")
            self.spin_max_aspect.valueChanged.connect(self._on_mfg_spin_changed)
            mfg_layout.addWidget(self.spin_max_aspect, 3, 3)

            self.chk_desensitize = QCheckBox("Desensitize for Tolerancing (As-Built)", self.grp_mfg)
            self.chk_desensitize.setChecked(False)
            self.chk_desensitize.setToolTip("Adds penalty against extreme surface curvatures to reduce sensitivity to manufacturing and alignment tolerances.")
            self.chk_desensitize.toggled.connect(self._on_mfg_spin_changed)
            mfg_layout.addWidget(self.chk_desensitize, 4, 0, 1, 4)

            self.btn_fit_test_plates = QPushButton("Fit to Tooling Test Plates (TFLT)", self.grp_mfg)
            self.btn_fit_test_plates.setToolTip("Snaps all lens radii to standard catalog test plate tooling (ISO/Thorlabs/Edmund) and re-optimizes air spaces.")
            self.btn_fit_test_plates.clicked.connect(self._on_fit_test_plates_clicked)
            mfg_layout.addWidget(self.btn_fit_test_plates, 5, 0, 1, 4)

            layout.addWidget(self.grp_mfg)


            self.target_spread_label = QLabel("<b>Target Spread (mm):</b>")
            layout.addWidget(self.target_spread_label)
            self.target_spread_spin = QDoubleSpinBox(container)
            self.target_spread_spin.setRange(0.1, 1000.0)
            self.target_spread_spin.setDecimals(2)
            self.target_spread_spin.setValue(12.0)
            layout.addWidget(self.target_spread_spin)
            
            # Hide by default
            self.target_spread_label.hide()
            self.target_spread_spin.hide()

            self.opt_tilt_check = QCheckBox("Optimize Detector Tilt", container)
            self.opt_tilt_check.setChecked(False)
            layout.addWidget(self.opt_tilt_check)

            # Wavelength Configuration (SSOT: System Explorer)
            layout.addWidget(QLabel("<b>Active Wavelengths:</b> (System Explorer)"))
            self.lbl_active_wavelengths = QLabel("Loading wavelengths...", container)
            self.lbl_active_wavelengths.setWordWrap(True)
            layout.addWidget(self.lbl_active_wavelengths)

            self.btn_open_wavelengths = QPushButton("Configure in System Explorer...", container)
            self.btn_open_wavelengths.setToolTip("Open System Explorer -> Wavelengths tab to configure optical bands, weights, and primary wavelength")
            self.btn_open_wavelengths.clicked.connect(self._on_open_system_explorer_wavelengths)
            layout.addWidget(self.btn_open_wavelengths)

            # Real Imaging Field Configuration (SSOT: System Explorer)
            layout.addWidget(QLabel("<b>Active Fields & FOV:</b> (System Explorer)"))
            self.lbl_active_fields = QLabel("Loading fields...", container)
            self.lbl_active_fields.setWordWrap(True)
            layout.addWidget(self.lbl_active_fields)

            fields_btn_layout = QHBoxLayout()
            self.combo_field_quick_preset = QComboBox(container)
            self.combo_field_quick_preset.addItems([
                "On-Axis (0°)",
                "3-Field Linear (0°, 0.7F, 1.0F)",
                "5-Field Cross (0°, ±X, ±Y)"
            ])
            self.combo_field_quick_preset.setToolTip("Quickly select Zemax multi-field real imaging configuration")
            self.combo_field_quick_preset.currentIndexChanged.connect(self._on_quick_field_preset_selected)
            fields_btn_layout.addWidget(self.combo_field_quick_preset)

            self.btn_open_fields = QPushButton("Configure...", container)
            self.btn_open_fields.setToolTip("Open System Explorer -> Fields tab to configure custom field points, angles, or object heights")
            self.btn_open_fields.clicked.connect(self._on_open_system_explorer_fields)
            fields_btn_layout.addWidget(self.btn_open_fields)
            layout.addLayout(fields_btn_layout)

            # Compatibility controls (hidden from UI, preserved for external tests/scripts)
            self.wl_preset_combo = QComboBox(container)
            self.wl_preset_combo.addItems([
                "Visible Achromat (F, d, C: 486, 588, 656 nm)",
                "Monochromatic (d-line: 587.6 nm)",
                "Broadband Visible (450 - 650 nm, 5 wls)",
                "NIR (785, 850, 1064 nm)",
                "Telecom C-Band (1530, 1550, 1565 nm)",
                "Custom..."
            ])
            self.wl_preset_combo.currentTextChanged.connect(self._on_wl_preset_changed)
            self.wl_preset_combo.hide()

            self.wl_edit = QLineEdit(container)
            self.wl_edit.setText("0.4861, 0.5876, 0.6563")
            self.wl_edit.textChanged.connect(lambda: self.update_system_info())
            self.wl_edit.hide()

            layout.addWidget(QLabel("<b>Rays per Wavelength:</b>"))
            self.rays_spin = QSpinBox(container)
            self.rays_spin.setRange(3, 101)
            self.rays_spin.setValue(21)
            layout.addWidget(self.rays_spin)

            layout.addWidget(QLabel("<b>Max Iterations:</b>"))
            self.max_iter_spin = QSpinBox(container)
            self.max_iter_spin.setRange(10, 2000)
            self.max_iter_spin.setValue(100)
            layout.addWidget(self.max_iter_spin)

            self.btn_optimize = QPushButton("Optimize System", container)
            self.btn_optimize.clicked.connect(self.on_optimize_clicked)
            layout.addWidget(self.btn_optimize)

            self.btn_stop = QPushButton("Stop Optimization", container)
            self.btn_stop.setEnabled(False)
            layout.addWidget(self.btn_stop)

            self.btn_seed_achromat = QPushButton("⚡ Seed Paraxial Achromat", container)
            self.btn_seed_achromat.setToolTip("Analytically compute and set thin-lens achromatic curvatures and focus from glass Abbe numbers and target EFL.")
            self.btn_seed_achromat.clicked.connect(self._on_seed_achromat_clicked)
            layout.addWidget(self.btn_seed_achromat)

            self.progress_bar = QProgressBar(container)
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)
            layout.addWidget(self.progress_bar)

            self.status_label = QLabel("Ready", container)
            layout.addWidget(self.status_label)

            scroll_area = QScrollArea(self)
            if hasattr(scroll_area, "setWidgetResizable"):
                scroll_area.setWidgetResizable(True)
            scroll_area.setWidget(container)
            if hasattr(scroll_area, "setFrameShape"):
                scroll_area.setFrameShape(getattr(QScrollArea, "NoFrame", 0))

            self.setWidget(scroll_area)
            self.setMinimumSize(250, 150)
            self._update_badges_style()
        else:
            self._rebuild_surface_checkboxes()

        if hasattr(self.model, "system_updated"):
            try:
                self.model.system_updated.connect(self._rebuild_surface_checkboxes)
            except Exception:
                pass

    def set_dark_mode(self, enabled: bool) -> None:
        """Update optimizer dock styling when theme switches between light and dark."""
        self.dark_mode = bool(enabled)
        self._update_badges_style()

    def _update_badges_style(self) -> None:
        if not QT_AVAILABLE:
            return
        is_dark = getattr(self, "dark_mode", False)
        if hasattr(self, "lbl_active_variables") and self.lbl_active_variables:
            if is_dark:
                self.lbl_active_variables.setStyleSheet(
                    "QLabel { background-color: #064e3b; color: #4ade80; border: 1px solid #047857; padding: 4px; border-radius: 3px; font-size: 11px; }"
                )
            else:
                self.lbl_active_variables.setStyleSheet(
                    "QLabel { background-color: #f0fdf4; color: #166534; border: 1px solid #bbf7d0; padding: 4px; border-radius: 3px; font-size: 11px; }"
                )
        if hasattr(self, "lbl_active_wavelengths") and self.lbl_active_wavelengths:
            if is_dark:
                self.lbl_active_wavelengths.setStyleSheet(
                    "QLabel { background-color: #0c4a6e; color: #38bdf8; border: 1px solid #0369a1; padding: 4px; border-radius: 3px; font-size: 11px; }"
                )
            else:
                self.lbl_active_wavelengths.setStyleSheet(
                    "QLabel { background-color: #f0f9ff; color: #0369a1; border: 1px solid #bae6fd; padding: 4px; border-radius: 3px; font-size: 11px; }"
                )
        if hasattr(self, "lbl_active_fields") and self.lbl_active_fields:
            if is_dark:
                self.lbl_active_fields.setStyleSheet(
                    "QLabel { background-color: #3b0764; color: #c084fc; border: 1px solid #581c87; padding: 4px; border-radius: 3px; font-size: 11px; }"
                )
            else:
                self.lbl_active_fields.setStyleSheet(
                    "QLabel { background-color: #faf5ff; color: #6b21a8; border: 1px solid #e9d5ff; padding: 4px; border-radius: 3px; font-size: 11px; }"
                )

    def _rebuild_surface_checkboxes(self):
        """
        Dynamically rebuilds checkbox widgets when the underlying optical model updates.
        Synchronizes variable checkboxes with Surface.thickness_variable flags.
        """
        # Clear existing widgets from layout if in Qt mode
        if hasattr(self, 'surfaces_layout') and self.surfaces_layout is not None:
            for cb in self.checkboxes.values():
                self.surfaces_layout.removeWidget(cb)
                if hasattr(cb, 'deleteLater'):
                    cb.deleteLater()
            for cb in self.rad_checkboxes.values():
                self.surfaces_layout.removeWidget(cb)
                if hasattr(cb, 'deleteLater'):
                    cb.deleteLater()
                    
            # Also clear layout items that might be HBoxLayouts containing the checkboxes
            while self.surfaces_layout.count():
                item = self.surfaces_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
                elif item.layout():
                    while item.layout().count():
                        subitem = item.layout().takeAt(0)
                        if subitem.widget():
                            subitem.widget().deleteLater()
                    item.layout().deleteLater()
                    
        self.checkboxes.clear()
        self.rad_checkboxes.clear()

        # Update target surface combo box
        if hasattr(self, 'target_surf_combo'):
            current_target = self.target_surf_combo.currentData()
            self.target_surf_combo.blockSignals(True)
            self.target_surf_combo.clear()
            self.target_surf_combo.addItem("Detector (Last Surface)", -1)
            target_idx_to_set = -1

        stack = self.model.get_surface_stack()
        if not stack:
            if hasattr(self, 'target_surf_combo'):
                self.target_surf_combo.blockSignals(False)
            self.update_system_info()
            return

        for i, surf in enumerate(stack.surfaces):
            name = surf.name or f"Surface {i}"
            is_air = getattr(surf, 'is_air', True)
            medium_tag = "AIR" if is_air else str(surf.glass)
            
            if hasattr(self, 'target_surf_combo'):
                self.target_surf_combo.addItem(f"{i}: {name} ({medium_tag})", i)
                if current_target == i:
                    target_idx_to_set = i
            
            row_widget = QWidget(self)
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            
            label_text = f"S{i} ({name[:12]})"
            label = QLabel(label_text)
            label.setFixedWidth(105)
            row_layout.addWidget(label)
            
            # Thickness / Distance Checkbox (Air vs Lens Glass)
            thick_label = f"S{i}: Dist ({surf.thickness:.1f})" if is_air else f"S{i}: Glass ({surf.thickness:.1f})"
            cb_thic = QCheckBox(thick_label)
            is_var_thic = bool(getattr(surf, 'thickness_variable', False))
            cb_thic.toggled.connect(lambda checked, idx=i: self._on_checkbox_toggled(idx, checked, is_radius=False))
            
            if is_var_thic:
                if hasattr(cb_thic, 'blockSignals'):
                    cb_thic.blockSignals(True)
                cb_thic.setChecked(True)
                if hasattr(cb_thic, 'blockSignals'):
                    cb_thic.blockSignals(False)
                if i not in self.selected_indices:
                    self.selected_indices.append(i)
            elif i in self.selected_indices:
                self.selected_indices.remove(i)
                
            self.checkboxes[i] = cb_thic
            row_layout.addWidget(cb_thic)
            
            # Radius Checkbox
            rad_label = f"S{i}: R={surf.radius:.1f}" if abs(surf.radius) > 1e-4 else f"S{i}: Flat"
            cb_rad = QCheckBox(rad_label)
            is_var_rad = bool(getattr(surf, 'radius_variable', False))
            cb_rad.toggled.connect(lambda checked, idx=i: self._on_checkbox_toggled(idx, checked, is_radius=True))
            
            if is_var_rad:
                if hasattr(cb_rad, 'blockSignals'):
                    cb_rad.blockSignals(True)
                cb_rad.setChecked(True)
                if hasattr(cb_rad, 'blockSignals'):
                    cb_rad.blockSignals(False)
                if i not in self.selected_rad_indices:
                    self.selected_rad_indices.append(i)
            elif i in self.selected_rad_indices:
                self.selected_rad_indices.remove(i)
                
            self.rad_checkboxes[i] = cb_rad
            row_layout.addWidget(cb_rad)
            
            if hasattr(self, 'surfaces_layout') and self.surfaces_layout is not None:
                self.surfaces_layout.addWidget(row_widget)
        
        if hasattr(self, 'target_surf_combo'):
            idx = self.target_surf_combo.findData(target_idx_to_set)
            if idx >= 0:
                self.target_surf_combo.setCurrentIndex(idx)
            self.target_surf_combo.blockSignals(False)
        self.update_system_info()

    def set_zemax_defaults(self):
        """Set all air gaps variable and all solid lenses fixed, like Zemax."""
        if hasattr(self.model, 'set_zemax_defaults'):
            self.model.set_zemax_defaults(var_air_gaps=True, fix_lenses=True)
        self._rebuild_surface_checkboxes()

    def clear_all_variables(self):
        """Set all parameters fixed."""
        if hasattr(self.model, 'set_all_fixed'):
            self.model.set_all_fixed()
        self._rebuild_surface_checkboxes()

    def _on_checkbox_toggled(self, row: int, checked: bool, is_radius: bool = False):
        """
        Handle checkbox toggle event for surface at given row index.
        Updates selected_indices and mutates surface flags in the model stack.
        """
        if is_radius:
            target_list = self.selected_rad_indices
            attr_name = "radius_variable"
        else:
            target_list = self.selected_indices
            attr_name = "thickness_variable"
            
        if checked:
            if row not in target_list:
                target_list.append(row)
        else:
            if row in target_list:
                target_list.remove(row)

        if hasattr(self.model, "_surfaces") and 0 <= row < len(self.model._surfaces):
            setattr(self.model._surfaces[row], attr_name, checked)

        stack = self.model.get_surface_stack()
        if stack and 0 <= row < len(stack.surfaces):
            setattr(stack.surfaces[row], attr_name, checked)
            if hasattr(self.model, '_stack') and self.model._stack is not None and row < len(self.model._stack.surfaces):
                setattr(self.model._stack.surfaces[row], attr_name, checked)
            if hasattr(self.model, 'system_updated') and self.model.system_updated is not None:
                try:
                    self.model.system_updated.emit()
                except Exception:
                    pass
        self.update_system_info()

    def get_variable_indices(self) -> VariableIndices:
        """Return dict of zero-based surface indices marked as variable thicknesses or radii."""
        if not self.selected_indices and not self.selected_rad_indices and self.model is not None:
            stack = self.model.get_surface_stack()
            if stack and stack.surfaces:
                has_flags = any(
                    getattr(s, "thickness_variable", False) or getattr(s, "radius_variable", False)
                    for s in stack.surfaces[:-1]
                )
                if has_flags:
                    self._rebuild_surface_checkboxes()
        return VariableIndices({
            "thickness": sorted(list(self.selected_indices)),
            "radius": sorted(list(self.selected_rad_indices))
        })

    def _on_preset_changed(self, text: str):
        self._updating_preset = True
        try:
            from optics_app.engine.manufacturing_standards import QualityPreset, ManufacturingConstraintsConfig
            if "Commercial" in text:
                cfg = ManufacturingConstraintsConfig.from_preset(QualityPreset.COMMERCIAL)
            elif "High-Precision" in text:
                cfg = ManufacturingConstraintsConfig.from_preset(QualityPreset.HIGH_PRECISION)
            elif "Precision" in text:
                cfg = ManufacturingConstraintsConfig.from_preset(QualityPreset.PRECISION)
            else:
                return
            if hasattr(self, 'spin_min_glass'):
                self.spin_min_glass.setValue(cfg.min_glass_thickness)
            if hasattr(self, 'spin_max_glass'):
                self.spin_max_glass.setValue(cfg.max_glass_thickness)
            if hasattr(self, 'spin_min_edge'):
                self.spin_min_edge.setValue(cfg.min_edge_thickness)
            if hasattr(self, 'spin_min_air'):
                self.spin_min_air.setValue(cfg.min_air_gap)
            if hasattr(self, 'spin_min_bfl'):
                self.spin_min_bfl.setValue(cfg.min_bfl)
            if hasattr(self, 'spin_max_aspect'):
                self.spin_max_aspect.setValue(cfg.max_aspect_ratio)
            if hasattr(self, 'chk_desensitize'):
                self.chk_desensitize.setChecked(cfg.desensitize_tolerances)
        finally:
            self._updating_preset = False

    def _on_mfg_spin_changed(self, *args):
        if getattr(self, '_updating_preset', False):
            return
        if hasattr(self, 'combo_mfg_preset'):
            self.combo_mfg_preset.blockSignals(True)
            for i in range(self.combo_mfg_preset.count()):
                if "Custom" in self.combo_mfg_preset.itemText(i):
                    self.combo_mfg_preset.setCurrentIndex(i)
                    break
            self.combo_mfg_preset.blockSignals(False)

    def _on_fit_test_plates_clicked(self):
        if hasattr(self, 'fit_test_plates_requested') and self.fit_test_plates_requested:
            self.fit_test_plates_requested.emit()

    def get_manufacturing_constraints(self) -> Dict[str, Any]:
        """Return user-configured physical manufacturing constraint parameters."""
        enforce = self.grp_mfg.isChecked() if hasattr(self, 'grp_mfg') else False
        return {
            "enforce": enforce,
            "preset": self.combo_mfg_preset.currentText() if hasattr(self, 'combo_mfg_preset') else "Commercial",
            "min_glass_thickness": float(self.spin_min_glass.value()) if hasattr(self, 'spin_min_glass') else 1.5,
            "max_glass_thickness": float(self.spin_max_glass.value()) if hasattr(self, 'spin_max_glass') else 18.0,
            "min_edge_thickness": float(self.spin_min_edge.value()) if hasattr(self, 'spin_min_edge') else 1.0,
            "min_air_gap": float(self.spin_min_air.value()) if hasattr(self, 'spin_min_air') else 0.5,
            "min_bfl": float(self.spin_min_bfl.value()) if hasattr(self, 'spin_min_bfl') else 10.0,
            "max_aspect_ratio": float(self.spin_max_aspect.value()) if hasattr(self, 'spin_max_aspect') else 10.0,
            "desensitize_tolerances": bool(self.chk_desensitize.isChecked()) if hasattr(self, 'chk_desensitize') else False,
        }


    def set_surface_variable(self, row: int, is_variable: bool = True):
        """Programmatically set whether a surface thickness is variable."""
        if row in self.checkboxes:
            self.checkboxes[row].setChecked(is_variable)
        else:
            self._on_checkbox_toggled(row, is_variable)

    def validate_current_configuration(self) -> ValidationResult:
        """Audit current system setup, parameters, and variable assignments."""
        stack = self.model.get_surface_stack() if hasattr(self.model, 'get_surface_stack') else None
        if not stack:
            return ValidationResult(is_valid=True, items=[])
        
        target = self.target_combo.currentText() if hasattr(self, 'target_combo') else "Spot Size (Focus)"
        target_spread = self.target_spread_spin.value() if hasattr(self, 'target_spread_spin') else 12.0
        target_surf = self.target_surf_combo.currentData() if hasattr(self, 'target_surf_combo') else -1
        target_spot_um = self.target_spot_spin.value() if hasattr(self, 'target_spot_spin') else 0.0
        target_spot_mm = float(target_spot_um) / 1000.0
        max_iter = self.max_iter_spin.value() if hasattr(self, 'max_iter_spin') else 100
        rays = self.rays_spin.value() if hasattr(self, 'rays_spin') else 21
        opt_tilt = self.opt_tilt_check.isChecked() if hasattr(self, 'opt_tilt_check') else False

        return OptimizationValidator.validate_system(
            stack=stack,
            variable_indices=self.get_variable_indices(),
            target=target,
            target_spread=target_spread,
            target_surface_index=target_surf if target_surf is not None and target_surf >= 0 else None,
            target_spot_size=target_spot_mm,
            max_iter=max_iter,
            num_rays=rays,
            opt_tilt=opt_tilt
        )

    def auto_repair_configuration(self) -> List[str]:
        """Apply automated one-click repair for common configuration errors."""
        stack = self.model.get_surface_stack() if hasattr(self.model, 'get_surface_stack') else None
        if not stack:
            return []
        
        target = self.target_combo.currentText() if hasattr(self, 'target_combo') else "Spot Size (Focus)"
        repaired_stack, var_dict, fixes = OptimizationValidator.auto_repair(
            stack=stack,
            variable_indices=self.get_variable_indices(),
            target=target
        )
        
        # Update variable checkboxes in UI
        for s_idx in var_dict.get("thickness", []):
            if s_idx in self.checkboxes:
                self.checkboxes[s_idx].setChecked(True)
        for s_idx in var_dict.get("radius", []):
            if s_idx in self.rad_checkboxes:
                self.rad_checkboxes[s_idx].setChecked(True)

        if hasattr(self.model, 'system_updated') and self.model.system_updated is not None:
            try:
                self.model.system_updated.emit()
            except Exception:
                pass

        return fixes

    def on_optimize_clicked(self):
        """Slot invoked when 'Optimize System' button is pressed with pre-flight check."""
        method = "Nelder-Mead"
        if hasattr(self, 'optimize_requested') and self.optimize_requested is not None:
            try:
                # Pre-flight diagnostic validation
                validation = self.validate_current_configuration()
                is_headless = os.environ.get("QT_QPA_PLATFORM") == "offscreen" or "pytest" in sys.modules or "unittest" in sys.modules
                if validation.has_errors:
                    if QT_AVAILABLE and hasattr(QMessageBox, 'warning') and not is_headless:
                        msg = "Pre-flight check detected configuration issues:\n\n" + validation.summary()
                        if validation.auto_fix_available:
                            msg += "\n\nWould you like to automatically apply recommended fixes?"
                            reply = QMessageBox.question(
                                self,
                                "Optimization Pre-Flight Warning",
                                msg,
                                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                                QMessageBox.Yes
                            )
                            if reply == QMessageBox.Yes:
                                self.auto_repair_configuration()
                            elif reply == QMessageBox.Cancel:
                                self.status_label.setText("Optimization cancelled by user.")
                                return
                        else:
                            QMessageBox.warning(self, "Optimization Pre-Flight Error", msg)
                            self.status_label.setText("Please fix configuration errors before optimizing.")
                            return
                    else:
                        # Headless / mock environment: auto-repair if needed
                        if validation.auto_fix_available:
                            self.auto_repair_configuration()

                # Extract selected method, max iterations, and target
                method = self.method_combo.currentText()
                max_iter = self.max_iter_spin.value()
                rays = self.rays_spin.value() if hasattr(self, 'rays_spin') else 21
                target = self.target_combo.currentText() if hasattr(self, 'target_combo') else "Spot Size (Focus)"
                spread = self.target_spread_spin.value() if hasattr(self, 'target_spread_spin') else 12.0
                opt_tilt = self.opt_tilt_check.isChecked() if hasattr(self, 'opt_tilt_check') else False
                target_surf = self.target_surf_combo.currentData() if hasattr(self, 'target_surf_combo') else -1
                if target_surf is None:
                    target_surf = -1
                target_spot_um = self.target_spot_spin.value() if hasattr(self, 'target_spot_spin') else 0.0
                target_spot_mm = float(target_spot_um) / 1000.0

                target_efl = None
                if (hasattr(self, 'chk_constrain_efl') and self.chk_constrain_efl.isChecked()) or ("Constrain EFL" in str(target)):
                    if hasattr(self, 'target_efl_spin'):
                        target_efl = float(self.target_efl_spin.value())

                # Set status to running
                self.status_label.setText("Running optimization...")
                self.progress_bar.setRange(0, 0)  # Indeterminate progress
                self.btn_optimize.setEnabled(False)
                if hasattr(self, 'btn_stop'):
                    self.btn_stop.setEnabled(True)

                # Emit the signal so the main window can handle the background threading
                self.optimize_requested.emit(self.get_variable_indices(), method, max_iter, target, rays, spread, opt_tilt, target_surf, target_spot_mm, target_efl)
            except Exception as ex:
                self.status_label.setText(f"Error starting optimization: {str(ex)}")

    def on_optimization_finished(self, result: Optional[Any] = None):
        """Restore UI state after optimization completes or aborts."""
        if result is not None:
            self.last_result = result
        if hasattr(self, 'btn_optimize'):
            self.btn_optimize.setEnabled(True)
        if hasattr(self, 'btn_stop'):
            self.btn_stop.setEnabled(False)
        if hasattr(self, 'progress_bar'):
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(100)

    def set_status(self, text: str):
        """Update status label text."""
        if hasattr(self, 'status_label') and self.status_label is not None:
            if hasattr(self.status_label, 'setText'):
                self.status_label.setText(str(text))

    def _on_constrain_efl_toggled(self, checked: bool):
        if hasattr(self, 'target_efl_spin'):
            self.target_efl_spin.setEnabled(checked)

    def _on_target_changed(self, text: str):
        if ("Constrain EFL" in text or "Achromatic Focus" in text) and hasattr(self, 'chk_constrain_efl'):
            self.chk_constrain_efl.setChecked(True)
        if hasattr(self, 'target_spread_label'):
            if text == "Spectrometer Spread":
                self.target_spread_label.show()
                self.target_spread_spin.show()
            else:
                self.target_spread_label.hide()
                self.target_spread_spin.hide()
        if hasattr(self, 'target_spot_label'):
            if text == "Target Spot Size":
                self.target_spot_label.show()
                self.target_spot_spin.show()
            else:
                self.target_spot_label.hide()
                self.target_spot_spin.hide()

    def _on_seed_achromat_clicked(self):
        """Analytically compute and set thin-lens achromatic curvatures and focus into active surfaces."""
        if not self.model:
            return
        stack = getattr(self.model, 'stack', None)
        if stack is None and hasattr(self.model, 'get_surface_stack'):
            stack = self.model.get_surface_stack()
        if stack is None and hasattr(self.model, 'surfaces'):
            surfaces = self.model.surfaces
        elif stack is not None:
            surfaces = stack.surfaces
        else:
            return

        target_efl = float(self.target_efl_spin.value()) if hasattr(self, 'target_efl_spin') else 50.0
        if target_efl <= 0:
            target_efl = 50.0

        glasses = []
        for idx, s in enumerate(surfaces):
            gname = getattr(s, 'glass', 'AIR') or 'AIR'
            if getattr(s, 'is_glass', False) and gname.upper() not in ('AIR', ''):
                try:
                    from optics_app.utils.materials import get_refractive_index
                    nd = get_refractive_index(gname, 0.58756)
                    nF = get_refractive_index(gname, 0.48613)
                    nC = get_refractive_index(gname, 0.65627)
                    vd = (nd - 1.0) / max(1e-6, abs(nF - nC))
                except Exception:
                    nd, vd = 1.5168, 64.17
                glasses.append((idx, gname, nd, vd))

        if len(glasses) < 2:
            try:
                try:
                    from v2.engine.achromat_designer import AchromatDesignSpec, optimize_achromatic_doublet
                    from v2.engine.lde import SurfaceStack
                    from v2.engine.surface import Surface, SurfaceType
                except ImportError:
                    from optics_app.engine.achromat_designer import AchromatDesignSpec, optimize_achromatic_doublet
                    from optics_app.engine.lde import SurfaceStack
                    from optics_app.engine.surface import Surface, SurfaceType

                spec = AchromatDesignSpec(target_efl=target_efl, diameter=25.4, crown_glass="N-BK7", flint_glass="N-SF2")
                res = optimize_achromatic_doublet(spec)
                semi_diam = 12.7
                new_stack = SurfaceStack()
                s0 = Surface(name="S0 (OBJ)", surface_type=SurfaceType.OBJECT, radius=0.0, thickness=1000.0, glass="AIR", semi_diameter=semi_diam)
                s1 = Surface(name="S1 (STO)", surface_type=SurfaceType.STOP, radius=float(res.radii[0]), thickness=float(res.thicknesses[0]), glass="N-BK7", semi_diameter=semi_diam, radius_variable=True, thickness_variable=True)
                s2 = Surface(name="S2 (Contact)", surface_type=SurfaceType.STANDARD, radius=float(res.radii[1]), thickness=float(res.thicknesses[1]), glass="N-SF2", semi_diameter=semi_diam, radius_variable=True, thickness_variable=True)
                s3 = Surface(name="S3 (Rear)", surface_type=SurfaceType.STANDARD, radius=float(res.radii[2]), thickness=max(1.0, float(res.bfl)), glass="AIR", semi_diameter=semi_diam, radius_variable=True, thickness_variable=True)
                s4 = Surface(name="S4 (IMA)", surface_type=SurfaceType.DETECTOR, radius=0.0, thickness=0.0, glass="AIR", semi_diameter=10.0)
                new_stack.add_surface(s0)
                new_stack.add_surface(s1)
                new_stack.add_surface(s2)
                new_stack.add_surface(s3)
                new_stack.add_surface(s4)
                if hasattr(self.model, 'set_surface_stack'):
                    self.model.set_surface_stack(new_stack)
                self._rebuild_surface_checkboxes()
                self.update_system_info()
                if hasattr(self.model, 'system_updated'):
                    self.model.system_updated.emit()
                if hasattr(self, 'status_label'):
                    self.status_label.setText(f"Seeded Achromatic Doublet (EFL={res.efl:.1f} mm, BFL={res.bfl:.1f} mm)")
                return
            except Exception:
                pass

        if len(glasses) >= 2:
            idx1, g1, n1, V1 = glasses[0]
            idx2, g2, n2, V2 = glasses[1]
            if abs(V1 - V2) > 3.0:
                phi_tot = 1.0 / target_efl
                phi1 = phi_tot * (V1 / (V1 - V2))
                phi2 = -phi_tot * (V2 / (V1 - V2))

                inter_idx = idx1 + 1
                back_idx = idx2 + 1
                if inter_idx < len(surfaces) and (abs(surfaces[inter_idx].radius) < 1e-4 or math.isinf(surfaces[inter_idx].radius)):
                    # Flat interface: R1 = (n1-1)/phi1, R2 = 0, R3 = (1-n2)/phi2
                    surfaces[idx1].radius = round(float((n1 - 1.0) / phi1), 3)
                    surfaces[inter_idx].radius = 0.0
                    if back_idx < len(surfaces):
                        surfaces[back_idx].radius = round(float((1.0 - n2) / phi2), 3)
                else:
                    # Curved interface
                    surfaces[idx1].radius = round(float(0.8 * target_efl), 3)
                    surfaces[inter_idx].radius = round(float(-0.35 * target_efl), 3)
                    if back_idx < len(surfaces):
                        surfaces[back_idx].radius = round(float(-1.5 * target_efl), 3)

                try:
                    from optics_app.engine.optimizer import compute_system_efl
                    _, bfl = compute_system_efl(surfaces, 0.58756)
                    if back_idx < len(surfaces) and bfl > 1.0 and not math.isinf(bfl):
                        surfaces[back_idx].thickness = round(float(bfl), 3)
                except Exception:
                    pass

                if hasattr(self.model, 'set_surface_stack') and stack is not None:
                    self.model.set_surface_stack(stack)
                elif hasattr(self.model, 'layoutChanged'):
                    self.model.layoutChanged.emit()
                if hasattr(self.model, 'system_updated'):
                    self.model.system_updated.emit()
                self._rebuild_surface_checkboxes()
                self.update_system_info()
                if hasattr(self, 'status_label'):
                    self.status_label.setText(f"Seeded paraxial achromat (R1={surfaces[idx1].radius}, R3={surfaces[back_idx].radius})")

    def _on_wl_preset_changed(self, text: str):
        if "Visible Achromat" in text:
            if hasattr(self, 'wl_edit'):
                self.wl_edit.setText("0.4861, 0.5876, 0.6563")
        elif "Monochromatic" in text:
            if hasattr(self, 'wl_edit'):
                self.wl_edit.setText("0.5876")
        elif "Broadband Visible" in text:
            if hasattr(self, 'wl_edit'):
                self.wl_edit.setText("0.450, 0.500, 0.550, 0.600, 0.650")
        elif "NIR" in text:
            if hasattr(self, 'wl_edit'):
                self.wl_edit.setText("0.785, 0.850, 1.064")
        elif "Telecom" in text:
            if hasattr(self, 'wl_edit'):
                self.wl_edit.setText("1.530, 1.550, 1.565")
        self.update_system_info()

    def _on_open_system_explorer_wavelengths(self):
        """Request opening the Wavelengths tab (index 1) in System Explorer."""
        if hasattr(self, 'open_system_explorer_requested') and self.open_system_explorer_requested is not None:
            self.open_system_explorer_requested.emit(1)

    def _on_open_system_explorer_fields(self):
        """Request opening the Fields tab (index 2) in System Explorer."""
        if hasattr(self, 'open_system_explorer_requested') and self.open_system_explorer_requested is not None:
            self.open_system_explorer_requested.emit(2)

    def _on_quick_field_preset_selected(self, index: int):
        """Quickly apply standard Zemax multi-field configurations directly from the dock."""
        sys_cfg = self.get_system_config()
        if sys_cfg is None:
            return
        if hasattr(self, 'combo_field_quick_preset'):
            text = self.combo_field_quick_preset.currentText()
            if "On-Axis" in text:
                sys_cfg.apply_field_preset("On-Axis Single Point")
            elif "3-Field" in text:
                sys_cfg.apply_field_preset("3-Field Linear", max_field=10.0)
            elif "5-Field" in text:
                sys_cfg.apply_field_preset("5-Field Cross", max_field=10.0)
            self.update_system_info()
            if hasattr(self.model, 'system_updated') and self.model.system_updated is not None:
                try:
                    self.model.system_updated.emit()
                except Exception:
                    pass

    def get_system_config(self) -> Optional[Any]:
        """Access the system config model instance."""
        if hasattr(self, '_system_config') and self._system_config is not None:
            return self._system_config
        if hasattr(self, 'model') and self.model is not None:
            return getattr(self.model, 'system_config', None)
        return None

    def set_system_config(self, cfg: Any):
        """Set system config reference and update info badges."""
        self._system_config = cfg
        self.update_system_info()

    def update_system_info(self):
        """Update live status badges for wavelengths and active variables."""
        # 1. Update Wavelengths Badge
        wls = self.get_wavelengths()
        if wls:
            formatted_wls = []
            for w in wls:
                nm = round(w * 1000.0)
                formatted_wls.append(f"{w:.4g} µm ({nm} nm)")
            w_str = " • ".join(formatted_wls)
            if hasattr(self, 'lbl_active_wavelengths') and hasattr(self.lbl_active_wavelengths, 'setText'):
                self.lbl_active_wavelengths.setText(w_str)
                if hasattr(self.lbl_active_wavelengths, 'setToolTip'):
                    self.lbl_active_wavelengths.setToolTip(f"Optimizing simultaneously across {len(wls)} active wavelengths defined in System Explorer")
        else:
            if hasattr(self, 'lbl_active_wavelengths') and hasattr(self.lbl_active_wavelengths, 'setText'):
                self.lbl_active_wavelengths.setText("0.5876 µm (Default d-line)")

        # 2. Update Variables Badge
        if hasattr(self, 'lbl_active_variables') and hasattr(self.lbl_active_variables, 'setText'):
            var_names = []
            stack = getattr(self.model, 'get_surface_stack', lambda: None)()
            if stack and hasattr(stack, 'surfaces'):
                for i, surf in enumerate(stack.surfaces):
                    s_name = getattr(surf, 'name', '') or f"S{i}"
                    if getattr(surf, 'thickness_variable', False):
                        kind = "Dist" if getattr(surf, 'is_air', True) else "Glass"
                        var_names.append(f"{s_name} [{kind} T]")
                    if getattr(surf, 'radius_variable', False):
                        var_names.append(f"{s_name} [R]")
            if var_names:
                summary = f"{len(var_names)} active: " + ", ".join(var_names[:4])
                if len(var_names) > 4:
                    summary += f" (+{len(var_names)-4} more)"
                self.lbl_active_variables.setText(summary)
                if hasattr(self.lbl_active_variables, 'setStyleSheet'):
                    self.lbl_active_variables.setStyleSheet("QLabel { background-color: #263326; color: #a8ffa8; padding: 4px; border-radius: 3px; font-size: 11px; }")
            else:
                self.lbl_active_variables.setText("No variables active (All parameters Fixed)")
                if hasattr(self.lbl_active_variables, 'setStyleSheet'):
                    self.lbl_active_variables.setStyleSheet("QLabel { background-color: #332626; color: #ffa8a8; padding: 4px; border-radius: 3px; font-size: 11px; }")

        # 3. Update Fields Badge
        sys_cfg = self.get_system_config()
        if sys_cfg is not None and getattr(sys_cfg, 'fields', None):
            flds = sys_cfg.fields
            ftype = getattr(sys_cfg, 'field_type', None)
            unit_str = "°" if "angle" in str(getattr(ftype, 'value', ftype)).lower() else " mm"
            if len(flds) == 1 and flds[0].x == 0.0 and flds[0].y == 0.0:
                f_str = f"1 Field: On-Axis (0{unit_str})"
            else:
                f_desc = [f"({f.x:g}{unit_str}, {f.y:g}{unit_str})" for f in flds[:3]]
                f_str = f"{len(flds)} Fields: " + ", ".join(f_desc)
                if len(flds) > 3:
                    f_str += f" (+{len(flds)-3} more)"
            if hasattr(self, 'lbl_active_fields') and hasattr(self.lbl_active_fields, 'setText'):
                self.lbl_active_fields.setText(f_str)
                if hasattr(self.lbl_active_fields, 'setToolTip'):
                    self.lbl_active_fields.setToolTip(f"Optimizing across {len(flds)} field points to balance off-axis aberrations")
        else:
            if hasattr(self, 'lbl_active_fields') and hasattr(self.lbl_active_fields, 'setText'):
                self.lbl_active_fields.setText("1 Field: On-Axis (0° Default)")

    def get_wavelengths(self) -> List[float]:
        """
        Extract active optimization wavelengths in micrometers.
        Single Source of Truth: Reads from SystemConfig.wavelengths.
        """
        sys_cfg = self.get_system_config()
        if sys_cfg is not None and getattr(sys_cfg, 'wavelengths', None):
            parsed = []
            for w in sys_cfg.wavelengths:
                val = float(getattr(w, 'wavelength_um', w))
                if val > 10.0:
                    val /= 1000.0
                if val > 0:
                    parsed.append(val)
            if parsed:
                return parsed

        # Backward compatibility with test overrides on wl_edit
        if hasattr(self, 'wl_edit') and self.wl_edit is not None:
            txt = self.wl_edit.text().strip()
            if txt:
                try:
                    vals = [float(w.strip()) for w in txt.split(',') if w.strip()]
                    w_list = [w / 1000.0 if w > 10.0 else w for w in vals if w > 0]
                    if w_list:
                        return w_list
                except Exception:
                    pass

        return [0.4861, 0.5876, 0.6563]

    def set_progress(self, val: int):
        """Update progress bar percentage."""
        if hasattr(self, 'progress_bar') and self.progress_bar is not None:
            if hasattr(self.progress_bar, 'setValue'):
                self.progress_bar.setValue(int(val))

    def get_config(self) -> Dict[str, Any]:
        return {
            "method": self.method_combo.currentText() if hasattr(self, 'method_combo') else "Nelder-Mead",
            "max_iter": self.max_iter_spin.value() if hasattr(self, 'max_iter_spin') else 100,
            "target": self.target_combo.currentText() if hasattr(self, 'target_combo') else "Spot Size (Focus)",
            "rays": self.rays_spin.value() if hasattr(self, 'rays_spin') else 21,
            "target_spread": self.target_spread_spin.value() if hasattr(self, 'target_spread_spin') else 12.0,
            "opt_tilt": self.opt_tilt_check.isChecked() if hasattr(self, 'opt_tilt_check') else False,
            "target_spot_um": self.target_spot_spin.value() if hasattr(self, 'target_spot_spin') else 0.0,
            "constrain_efl": self.chk_constrain_efl.isChecked() if hasattr(self, 'chk_constrain_efl') else False,
            "target_efl": self.target_efl_spin.value() if hasattr(self, 'target_efl_spin') else 50.0,
            "wavelengths": self.wl_edit.text() if hasattr(self, 'wl_edit') else "0.4861, 0.5876, 0.6563"
        }

    def set_config(self, cfg: Dict[str, Any]):
        if not isinstance(cfg, dict):
            return
        if hasattr(self, 'method_combo') and "method" in cfg:
            self.method_combo.setCurrentText(str(cfg["method"]))
        if hasattr(self, 'max_iter_spin') and "max_iter" in cfg:
            self.max_iter_spin.setValue(int(cfg["max_iter"]))
        if hasattr(self, 'target_combo') and "target" in cfg:
            self.target_combo.setCurrentText(str(cfg["target"]))
        if hasattr(self, 'rays_spin') and "rays" in cfg:
            self.rays_spin.setValue(int(cfg["rays"]))
        if hasattr(self, 'target_spread_spin') and "target_spread" in cfg:
            self.target_spread_spin.setValue(float(cfg["target_spread"]))
        if hasattr(self, 'opt_tilt_check') and "opt_tilt" in cfg:
            self.opt_tilt_check.setChecked(bool(cfg["opt_tilt"]))
        if hasattr(self, 'target_spot_spin') and "target_spot_um" in cfg:
            self.target_spot_spin.setValue(float(cfg["target_spot_um"]))
        if hasattr(self, 'chk_constrain_efl') and "constrain_efl" in cfg:
            self.chk_constrain_efl.setChecked(bool(cfg["constrain_efl"]))
        if hasattr(self, 'target_efl_spin') and "target_efl" in cfg:
            self.target_efl_spin.setValue(float(cfg["target_efl"]))
        if hasattr(self, 'wl_edit') and "wavelengths" in cfg:
            self.wl_edit.setText(str(cfg["wavelengths"]))


__all__ = ["OptimizerDockWidget"]
