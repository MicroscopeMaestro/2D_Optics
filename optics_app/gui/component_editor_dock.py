"""
Component Editor Dock Widget for 2D Optics Simulation & Design Studio.
Provides interactive visual sliders, dials, fine-precision spinboxes, quick-fold presets,
one-click mirror ray-aiming, and coupled multi-surface lens element group rotations.
"""

from typing import Optional, List, Tuple, Dict, Any
import math
import numpy as np

from optics_app.gui.qt_compat import (
    QT_AVAILABLE, QtWidgets, QWidget, QDockWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QComboBox, QDoubleSpinBox, QSlider,
    QCheckBox, QGroupBox, QScrollArea, QFrame, pyqtSignal, DummySignal, Qt,
    QTimer
)
from optics_app.gui.lde_table import LensDataTableModel
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.mirror_positioner import MirrorPositioner, MirrorAimResult
from optics_app.engine.laser_scanner import is_galvo, configure_galvo_scanner, GalvoScannerConfig


class ComponentEditorDockWidget(QDockWidget if QT_AVAILABLE else object):
    """
    Interactive Dock Widget for Optical Component Manipulation.
    Allows real-time rotation (tilt_x, tilt_y, tilt_z), translation (decenter_x, decenter_y, thickness),
    preset beam folding (90° YZ/XZ, 45°, retro-reflector), one-click target aiming,
    and rigid multi-surface lens group transformations.
    """
    if QT_AVAILABLE:
        component_changed = pyqtSignal(int)
        surface_selected = pyqtSignal(int)
    else:
        component_changed = None
        surface_selected = None

    def __init__(self, model: Optional[LensDataTableModel] = None, parent=None):
        if QT_AVAILABLE:
            super().__init__("Component Editor & Rotation", parent)
            self.setObjectName("ComponentEditorDockWidget")
            self.setAllowedAreas(Qt.AllDockWidgetAreas)
        else:
            self.component_changed = DummySignal()
            self.surface_selected = DummySignal()

        self.model: Optional[LensDataTableModel] = model
        self._selected_index: int = 0
        self._syncing_ui: bool = False

        # Throttling & Debouncing timers for high-frequency slider scrub events
        self._slider_change_pending = False
        self._throttle_timer = QTimer(self if QT_AVAILABLE else None)
        self._throttle_timer.setInterval(33)  # 30 Hz (33 ms)
        self._throttle_timer.setSingleShot(True)
        if hasattr(self._throttle_timer, "timeout") and hasattr(self._throttle_timer.timeout, "connect"):
            self._throttle_timer.timeout.connect(self._on_throttle_timeout)

        self._debounce_timer = QTimer(self if QT_AVAILABLE else None)
        self._debounce_timer.setInterval(150)  # 150 ms debounce
        self._debounce_timer.setSingleShot(True)
        if hasattr(self._debounce_timer, "timeout") and hasattr(self._debounce_timer.timeout, "connect"):
            self._debounce_timer.timeout.connect(self._on_debounce_timeout)

        self.dark_mode: bool = False
        self._setup_ui()
        self._update_theme_styles()

        if self.model is not None and hasattr(self.model, "system_updated"):
            try:
                self.model.system_updated.connect(self._on_model_updated)
            except Exception:
                pass

        self._rebuild_surface_list()

    def _setup_ui(self):
        """Construct the dock widget interface."""
        scroll_area = QScrollArea(self if QT_AVAILABLE else None)
        if hasattr(scroll_area, "setWidgetResizable"):
            scroll_area.setWidgetResizable(True)

        container = QWidget(scroll_area)
        main_layout = QVBoxLayout(container)
        main_layout.setContentsMargins(6, 6, 6, 6)

        # ----------------------------------------------------
        # ----------------------------------------------------
        # 1. Surface / Component Selector
        # ----------------------------------------------------
        selector_group = QGroupBox("Selected Optical Component", container)
        sel_layout = QVBoxLayout(selector_group)

        self.surface_combo = QComboBox(selector_group)
        self.surface_combo.currentIndexChanged.connect(self._on_surface_combo_changed)
        sel_layout.addWidget(self.surface_combo)

        self.surface_info_label = QLabel("Type: Standard | Medium: AIR", selector_group)
        sel_layout.addWidget(self.surface_info_label)

        main_layout.addWidget(selector_group)

        # ----------------------------------------------------
        # 1a. Mechanical State / Optical Routing
        # ----------------------------------------------------
        self.mechanism_group = QGroupBox("Mechanical State / Optical Routing", container)
        mechanism_layout = QVBoxLayout(self.mechanism_group)

        state_row = QHBoxLayout()
        state_row.addWidget(QLabel("State:"))
        self.mechanical_state_combo = QComboBox(self.mechanism_group)
        self.mechanical_state_combo.setToolTip(
            "Switch the active optical member without moving its housing or downstream hardware."
        )
        state_row.addWidget(self.mechanical_state_combo)
        self.btn_apply_mechanical_state = QPushButton("Apply State", self.mechanism_group)
        self.btn_apply_mechanical_state.clicked.connect(self._on_apply_mechanical_state)
        state_row.addWidget(self.btn_apply_mechanical_state)
        mechanism_layout.addLayout(state_row)

        quick_row = QHBoxLayout()
        self.btn_mech_1 = QPushButton("MECH ON → Detector", self.mechanism_group)
        self.btn_mech_1.clicked.connect(lambda: self._apply_mechanical_state("ON"))
        quick_row.addWidget(self.btn_mech_1)
        self.btn_mech_2 = QPushButton("MECH OFF → Dump", self.mechanism_group)
        self.btn_mech_2.clicked.connect(lambda: self._apply_mechanical_state("OFF"))
        quick_row.addWidget(self.btn_mech_2)
        self.btn_mech_3 = QPushButton("MECH FLAT (0°)", self.mechanism_group)
        self.btn_mech_3.clicked.connect(lambda: self._apply_mechanical_state("FLAT"))
        quick_row.addWidget(self.btn_mech_3)
        self.btn_dmd_on = self.btn_mech_1
        self.btn_dmd_off = self.btn_mech_2
        mechanism_layout.addLayout(quick_row)

        self.btn_open_ti_studio = QPushButton("🔲 Open TI Modulator Studio (PLM / DMD)...", self.mechanism_group)
        self.btn_open_ti_studio.setStyleSheet("background-color: #0288D1; color: white; font-weight: bold; padding: 5px 12px; margin-top: 4px;")
        self.btn_open_ti_studio.setToolTip("Open Texas Instruments Modulator Studio to configure states, 4-bit phase, and binary patterns")
        self.btn_open_ti_studio.clicked.connect(self._open_ti_studio)
        self.btn_open_ti_studio.setVisible(False)
        mechanism_layout.addWidget(self.btn_open_ti_studio)

        self.mechanical_state_status = QLabel("Select an actuated component.", self.mechanism_group)
        self.mechanical_state_status.setWordWrap(True)
        mechanism_layout.addWidget(self.mechanical_state_status)
        main_layout.addWidget(self.mechanism_group)

        # ----------------------------------------------------
        # 1c. Galvanometer Scanner Actuator & Beam Steering
        # ----------------------------------------------------
        self.galvo_group = QGroupBox("Galvanometer Scanner Actuator & Beam Steering", container)
        galvo_layout = QVBoxLayout(self.galvo_group)

        galvo_hdr_layout = QHBoxLayout()
        self.btn_designate_galvo = QPushButton("🎯 Designate as Galvo Scanner", self.galvo_group)
        self.btn_designate_galvo.setToolTip("Configure the selected surface as a motorized galvanometer scanning mirror.")
        self.btn_designate_galvo.clicked.connect(self._on_designate_galvo_clicked)
        galvo_hdr_layout.addWidget(self.btn_designate_galvo)

        self.lbl_galvo_mode = QLabel("Mirror / Passive", self.galvo_group)
        if QT_AVAILABLE:
            self.lbl_galvo_mode.setStyleSheet("font-weight: bold; color: #00796B;")
        galvo_hdr_layout.addWidget(self.lbl_galvo_mode)
        galvo_layout.addLayout(galvo_hdr_layout)

        # Axis, Min, Max, Step
        galvo_cfg_layout = QGridLayout()
        galvo_cfg_layout.addWidget(QLabel("Scan Axis:"), 0, 0)
        self.combo_galvo_axis = QComboBox(self.galvo_group)
        self.combo_galvo_axis.addItems(["X (Tilt Y)", "Y (Tilt X)"])
        self.combo_galvo_axis.currentIndexChanged.connect(self._on_galvo_config_changed)
        galvo_cfg_layout.addWidget(self.combo_galvo_axis, 0, 1)

        galvo_cfg_layout.addWidget(QLabel("Step (°):"), 0, 2)
        self.spin_galvo_step = QDoubleSpinBox(self.galvo_group)
        self.spin_galvo_step.setRange(0.01, 10.0)
        self.spin_galvo_step.setValue(1.0)
        self.spin_galvo_step.setSingleStep(0.5)
        self.spin_galvo_step.setSuffix("°")
        self.spin_galvo_step.valueChanged.connect(self._on_galvo_config_changed)
        galvo_cfg_layout.addWidget(self.spin_galvo_step, 0, 3)

        galvo_cfg_layout.addWidget(QLabel("Min θ:"), 1, 0)
        self.spin_galvo_min = QDoubleSpinBox(self.galvo_group)
        self.spin_galvo_min.setRange(-45.0, 0.0)
        self.spin_galvo_min.setValue(-10.0)
        self.spin_galvo_min.setSingleStep(1.0)
        self.spin_galvo_min.setSuffix("°")
        self.spin_galvo_min.valueChanged.connect(self._on_galvo_config_changed)
        galvo_cfg_layout.addWidget(self.spin_galvo_min, 1, 1)

        galvo_cfg_layout.addWidget(QLabel("Max θ:"), 1, 2)
        self.spin_galvo_max = QDoubleSpinBox(self.galvo_group)
        self.spin_galvo_max.setRange(0.0, 45.0)
        self.spin_galvo_max.setValue(10.0)
        self.spin_galvo_max.setSingleStep(1.0)
        self.spin_galvo_max.setSuffix("°")
        self.spin_galvo_max.valueChanged.connect(self._on_galvo_config_changed)
        galvo_cfg_layout.addWidget(self.spin_galvo_max, 1, 3)
        galvo_layout.addLayout(galvo_cfg_layout)

        # Live Steering Slider & SpinBox
        live_layout = QHBoxLayout()
        live_layout.addWidget(QLabel("Live Angle:"))
        self.slider_galvo_angle = QSlider(Qt.Horizontal, self.galvo_group)
        self.slider_galvo_angle.setRange(-1000, 1000)
        self.slider_galvo_angle.setValue(0)
        self.slider_galvo_angle.valueChanged.connect(self._on_galvo_slider_changed)
        live_layout.addWidget(self.slider_galvo_angle)

        self.spin_galvo_angle = QDoubleSpinBox(self.galvo_group)
        self.spin_galvo_angle.setRange(-45.0, 45.0)
        self.spin_galvo_angle.setValue(0.0)
        self.spin_galvo_angle.setSingleStep(0.5)
        self.spin_galvo_angle.setSuffix("°")
        self.spin_galvo_angle.valueChanged.connect(self._on_galvo_spin_changed)
        live_layout.addWidget(self.spin_galvo_angle)
        galvo_layout.addLayout(live_layout)

        # Live Readout
        self.lbl_galvo_readout = QLabel("θ_mech: 0.00° | θ_opt: 0.00°", self.galvo_group)
        if QT_AVAILABLE:
            self.lbl_galvo_readout.setStyleSheet("font-weight: bold; color: #1976D2; padding: 2px;")
        galvo_layout.addWidget(self.lbl_galvo_readout)

        # Quick Actuation Buttons
        act_btn_layout = QHBoxLayout()
        self.btn_galvo_min = QPushButton("⟪ Min", self.galvo_group)
        self.btn_galvo_min.clicked.connect(lambda: self._set_galvo_live_angle(self.spin_galvo_min.value()))
        act_btn_layout.addWidget(self.btn_galvo_min)

        self.btn_galvo_prev = QPushButton("⟨ -Step", self.galvo_group)
        self.btn_galvo_prev.clicked.connect(lambda: self._step_galvo_live_angle(-self.spin_galvo_step.value()))
        act_btn_layout.addWidget(self.btn_galvo_prev)

        self.btn_galvo_center = QPushButton("Center 0°", self.galvo_group)
        self.btn_galvo_center.clicked.connect(lambda: self._set_galvo_live_angle(0.0))
        act_btn_layout.addWidget(self.btn_galvo_center)

        self.btn_galvo_next = QPushButton("+Step ⟩", self.galvo_group)
        self.btn_galvo_next.clicked.connect(lambda: self._step_galvo_live_angle(+self.spin_galvo_step.value()))
        act_btn_layout.addWidget(self.btn_galvo_next)

        self.btn_galvo_max = QPushButton("Max ⟫", self.galvo_group)
        self.btn_galvo_max.clicked.connect(lambda: self._set_galvo_live_angle(self.spin_galvo_max.value()))
        act_btn_layout.addWidget(self.btn_galvo_max)
        galvo_layout.addLayout(act_btn_layout)

        # Aberration Analysis Link
        self.btn_galvo_analyze = QPushButton("📊 Simulate Field Aberrations →", self.galvo_group)
        self.btn_galvo_analyze.setToolTip("Switch to the Laser Scanning & Field Aberrations tab to compute spot size and distortion across the scan field.")
        self.btn_galvo_analyze.clicked.connect(self._on_galvo_analyze_clicked)
        galvo_layout.addWidget(self.btn_galvo_analyze)

        main_layout.addWidget(self.galvo_group)

        # ----------------------------------------------------
        # 1b. Lens Element Assembly Management & Parameter Protection
        # ----------------------------------------------------
        lens_mgmt_group = QGroupBox("Lens Element Assembly Management", container)
        lens_mgmt_layout = QVBoxLayout(lens_mgmt_group)

        lens_sel_layout = QHBoxLayout()
        lens_sel_layout.addWidget(QLabel("Lens:"))
        self.lens_element_combo = QComboBox(lens_mgmt_group)
        self.lens_element_combo.setToolTip("Select a specific lens element to lock, unlock for editing, or delete.")
        self.lens_element_combo.currentIndexChanged.connect(self._on_lens_element_combo_changed)
        lens_sel_layout.addWidget(self.lens_element_combo)
        lens_mgmt_layout.addLayout(lens_sel_layout)

        self.enable_lens_edition_check = QCheckBox("🔓 Enable Lens Edition (Unlock Parameters)", lens_mgmt_group)
        self.enable_lens_edition_check.setChecked(False)
        self.enable_lens_edition_check.setToolTip("When unchecked, lens prescription (radii, glass, thickness) is locked to prevent accidental changes. Check to edit.")
        self.enable_lens_edition_check.toggled.connect(self._on_enable_lens_edition_toggled)
        lens_mgmt_layout.addWidget(self.enable_lens_edition_check)

        lens_btn_layout = QHBoxLayout()
        self.btn_replace_lens = QPushButton("🔄 Replace Selected Lens ▾", lens_mgmt_group)
        self.btn_replace_lens.setToolTip("Replace the selected lens element with another lens from Thorlabs catalog, Zemax file, or Custom JSON file.")
        self.btn_replace_lens.clicked.connect(self._show_replace_lens_menu)
        lens_btn_layout.addWidget(self.btn_replace_lens)

        self.btn_flip_lens = QPushButton("⇄ Flip Whole Lens", lens_mgmt_group)
        self.btn_flip_lens.setToolTip(
            "Reverse the complete selected singlet, doublet, or multi-surface lens front-to-back. "
            "All curvatures, glass regions, and internal thicknesses are flipped together."
        )
        self.btn_flip_lens.clicked.connect(self._on_flip_lens_clicked)
        lens_btn_layout.addWidget(self.btn_flip_lens)

        self.btn_delete_lens = QPushButton("🗑 Delete Lens", lens_mgmt_group)
        self.btn_delete_lens.setToolTip("Removes all constituent surfaces of the selected lens element from the system.")
        if QT_AVAILABLE:
            self.btn_delete_lens.setStyleSheet("QPushButton { color: #b71c1c; font-weight: bold; padding: 4px; } QPushButton:hover { background-color: #ffebee; }")
        self.btn_delete_lens.clicked.connect(self._on_delete_lens_clicked)
        lens_btn_layout.addWidget(self.btn_delete_lens)

        lens_mgmt_layout.addLayout(lens_btn_layout)

        main_layout.addWidget(lens_mgmt_group)

        # ----------------------------------------------------
        # 2. Multi-Surface Lens Element Group Rotation & Decenter
        # ----------------------------------------------------
        group_rot_box = QGroupBox("Lens Element Group Coupling", container)
        grp_layout = QVBoxLayout(group_rot_box)

        self.group_rotation_check = QCheckBox("Transform (Rotate & Decenter) Element as Rigid Group", group_rot_box)
        self.group_rotation_check.setChecked(True)
        self.group_rotation_check.setToolTip("Couples multi-surface lens elements (singlets/doublets/triplets) so all constituent surfaces rotate and translate together as a rigid object without separating.")
        self.group_transform_check = self.group_rotation_check
        grp_layout.addWidget(self.group_rotation_check)

        pivot_layout = QHBoxLayout()
        pivot_layout.addWidget(QLabel("Pivot Point:"))
        self.pivot_combo = QComboBox(group_rot_box)
        self.pivot_combo.addItems(["Front Vertex", "Center of Mass", "Rear Vertex"])
        pivot_layout.addWidget(self.pivot_combo)
        grp_layout.addLayout(pivot_layout)

        main_layout.addWidget(group_rot_box)

        # ----------------------------------------------------
        # 3. 3D Rotation / Tilt Controls (Tilt X, Tilt Y, Tilt Z)
        # ----------------------------------------------------
        rot_group = QGroupBox("3D Rotation & Tilts", container)
        rot_layout = QGridLayout(rot_group)

        # Scale factor for sliders: 10.0 -> 0.1 deg resolution
        self._TILT_SCALE = 10.0

        # Tilt X (Elevation / Pitch)
        rot_layout.addWidget(QLabel("Tilt X (Pitch):"), 0, 0)
        self.tilt_x_slider = QSlider(Qt.Horizontal if QT_AVAILABLE else 1, rot_group)
        self.tilt_x_slider.setRange(-1800, 1800)
        self.tilt_x_slider.setValue(0)
        self.tilt_x_slider.valueChanged.connect(self._on_tilt_x_slider_changed)
        rot_layout.addWidget(self.tilt_x_slider, 0, 1)

        self.tilt_x_spin = QDoubleSpinBox(rot_group)
        self.tilt_x_spin.setRange(-180.0, 180.0)
        self.tilt_x_spin.setSingleStep(0.5)
        self.tilt_x_spin.setDecimals(2)
        self.tilt_x_spin.setSuffix("°")
        self.tilt_x_spin.valueChanged.connect(self._on_tilt_x_spin_changed)
        rot_layout.addWidget(self.tilt_x_spin, 0, 2)

        # Tilt Y (Azimuth / Yaw)
        rot_layout.addWidget(QLabel("Tilt Y (Yaw):"), 1, 0)
        self.tilt_y_slider = QSlider(Qt.Horizontal if QT_AVAILABLE else 1, rot_group)
        self.tilt_y_slider.setRange(-1800, 1800)
        self.tilt_y_slider.setValue(0)
        self.tilt_y_slider.valueChanged.connect(self._on_tilt_y_slider_changed)
        rot_layout.addWidget(self.tilt_y_slider, 1, 1)

        self.tilt_y_spin = QDoubleSpinBox(rot_group)
        self.tilt_y_spin.setRange(-180.0, 180.0)
        self.tilt_y_spin.setSingleStep(0.5)
        self.tilt_y_spin.setDecimals(2)
        self.tilt_y_spin.setSuffix("°")
        self.tilt_y_spin.valueChanged.connect(self._on_tilt_y_spin_changed)
        rot_layout.addWidget(self.tilt_y_spin, 1, 2)

        # Tilt Z (Roll)
        rot_layout.addWidget(QLabel("Tilt Z (Roll):"), 2, 0)
        self.tilt_z_slider = QSlider(Qt.Horizontal if QT_AVAILABLE else 1, rot_group)
        self.tilt_z_slider.setRange(-1800, 1800)
        self.tilt_z_slider.setValue(0)
        self.tilt_z_slider.valueChanged.connect(self._on_tilt_z_slider_changed)
        rot_layout.addWidget(self.tilt_z_slider, 2, 1)

        self.tilt_z_spin = QDoubleSpinBox(rot_group)
        self.tilt_z_spin.setRange(-180.0, 180.0)
        self.tilt_z_spin.setSingleStep(1.0)
        self.tilt_z_spin.setDecimals(2)
        self.tilt_z_spin.setSuffix("°")
        self.tilt_z_spin.valueChanged.connect(self._on_tilt_z_spin_changed)
        rot_layout.addWidget(self.tilt_z_spin, 2, 2)

        # Reset Tilts Button
        self.btn_reset_tilts = QPushButton("Reset Tilts (0°)", rot_group)
        self.btn_reset_tilts.clicked.connect(self.reset_tilts)
        rot_layout.addWidget(self.btn_reset_tilts, 3, 0, 1, 3)

        main_layout.addWidget(rot_group)

        # ----------------------------------------------------
        # 4. Decenter & Position (Decenter X, Decenter Y, Decenter Z, Thickness)
        # ----------------------------------------------------
        pos_group = QGroupBox("Spatial Decenter & Thickness", container)
        pos_layout = QGridLayout(pos_group)

        self._DEC_SCALE = 10.0

        # Decenter X
        pos_layout.addWidget(QLabel("Decenter X:"), 0, 0)
        self.dec_x_slider = QSlider(Qt.Horizontal if QT_AVAILABLE else 1, pos_group)
        self.dec_x_slider.setRange(-500, 500)
        self.dec_x_slider.setValue(0)
        self.dec_x_slider.valueChanged.connect(self._on_dec_x_slider_changed)
        pos_layout.addWidget(self.dec_x_slider, 0, 1)

        self.dec_x_spin = QDoubleSpinBox(pos_group)
        self.dec_x_spin.setRange(-500.0, 500.0)
        self.dec_x_spin.setSingleStep(0.1)
        self.dec_x_spin.setDecimals(3)
        self.dec_x_spin.setSuffix(" mm")
        self.dec_x_spin.valueChanged.connect(self._on_dec_x_spin_changed)
        pos_layout.addWidget(self.dec_x_spin, 0, 2)

        # Decenter Y
        pos_layout.addWidget(QLabel("Decenter Y:"), 1, 0)
        self.dec_y_slider = QSlider(Qt.Horizontal if QT_AVAILABLE else 1, pos_group)
        self.dec_y_slider.setRange(-500, 500)
        self.dec_y_slider.setValue(0)
        self.dec_y_slider.valueChanged.connect(self._on_dec_y_slider_changed)
        pos_layout.addWidget(self.dec_y_slider, 1, 1)

        self.dec_y_spin = QDoubleSpinBox(pos_group)
        self.dec_y_spin.setRange(-500.0, 500.0)
        self.dec_y_spin.setSingleStep(0.1)
        self.dec_y_spin.setDecimals(3)
        self.dec_y_spin.setSuffix(" mm")
        self.dec_y_spin.valueChanged.connect(self._on_dec_y_spin_changed)
        pos_layout.addWidget(self.dec_y_spin, 1, 2)

        # Decenter Z
        pos_layout.addWidget(QLabel("Decenter Z:"), 2, 0)
        self.dec_z_slider = QSlider(Qt.Horizontal if QT_AVAILABLE else 1, pos_group)
        self.dec_z_slider.setRange(-500, 500)
        self.dec_z_slider.setValue(0)
        self.dec_z_slider.valueChanged.connect(self._on_dec_z_slider_changed)
        pos_layout.addWidget(self.dec_z_slider, 2, 1)

        self.dec_z_spin = QDoubleSpinBox(pos_group)
        self.dec_z_spin.setRange(-500.0, 500.0)
        self.dec_z_spin.setSingleStep(0.1)
        self.dec_z_spin.setDecimals(3)
        self.dec_z_spin.setSuffix(" mm")
        self.dec_z_spin.valueChanged.connect(self._on_dec_z_spin_changed)
        pos_layout.addWidget(self.dec_z_spin, 2, 2)

        # Thickness / Propagation Distance
        self.lbl_thick_name = QLabel("Thickness to Next Surface:", pos_group)
        self.lbl_thick_name.setToolTip("Distance along optical axis from this surface vertex to the subsequent surface vertex.")
        pos_layout.addWidget(self.lbl_thick_name, 3, 0)
        self.thick_slider = QSlider(Qt.Horizontal if QT_AVAILABLE else 1, pos_group)
        self.thick_slider.setRange(0, 2000)
        self.thick_slider.setValue(100)
        self.thick_slider.valueChanged.connect(self._on_thick_slider_changed)
        pos_layout.addWidget(self.thick_slider, 3, 1)

        self.thick_spin = QDoubleSpinBox(pos_group)
        self.thick_spin.setRange(0.0, 2000.0)
        self.thick_spin.setSingleStep(0.5)
        self.thick_spin.setDecimals(3)
        self.thick_spin.setSuffix(" mm")
        self.thick_spin.valueChanged.connect(self._on_thick_spin_changed)
        pos_layout.addWidget(self.thick_spin, 3, 2)

        # Reset Decenters Button
        self.btn_reset_decenters = QPushButton("Reset Decenters (0 mm)", pos_group)
        self.btn_reset_decenters.clicked.connect(self.reset_decenters)
        pos_layout.addWidget(self.btn_reset_decenters, 4, 0, 1, 3)

        main_layout.addWidget(pos_group)

        # ----------------------------------------------------
        # 4b. Aperture & Stop Controls
        # ----------------------------------------------------
        aperture_group = QGroupBox("Aperture & Stop Controls", container)
        aperture_layout = QVBoxLayout(aperture_group)

        self.chk_is_stop = QCheckBox("🔴 System Aperture Stop (STO)", aperture_group)
        self.chk_is_stop.setToolTip("Mark this surface as the Aperture Stop that limits marginal and chief ray bundles.")
        self.chk_is_stop.toggled.connect(self._on_is_stop_toggled)
        aperture_layout.addWidget(self.chk_is_stop)

        aper_sd_layout = QHBoxLayout()
        aper_sd_layout.addWidget(QLabel("Clear Semi-Diameter:"))
        self.spin_semi_diam = QDoubleSpinBox(aperture_group)
        self.spin_semi_diam.setRange(0.0, 5000.0)
        self.spin_semi_diam.setSingleStep(0.5)
        self.spin_semi_diam.setDecimals(3)
        self.spin_semi_diam.setSuffix(" mm")
        self.spin_semi_diam.setToolTip("Clear aperture semi-diameter (radius) in mm (0.0 for point object/image).")
        self.spin_semi_diam.valueChanged.connect(self._on_semi_diam_changed)
        aper_sd_layout.addWidget(self.spin_semi_diam)
        aperture_layout.addLayout(aper_sd_layout)

        # Rectangular Substrate Shape Half-Span Controls (X and Y)
        self.widget_rect_substrate = QWidget(aperture_group)
        rect_sub_layout = QHBoxLayout(self.widget_rect_substrate)
        rect_sub_layout.setContentsMargins(0, 2, 0, 2)
        rect_sub_layout.addWidget(QLabel("Substrate Half-X:"))
        self.spin_sub_x = QDoubleSpinBox(self.widget_rect_substrate)
        self.spin_sub_x.setRange(0.0, 5000.0)
        self.spin_sub_x.setSingleStep(0.5)
        self.spin_sub_x.setDecimals(3)
        self.spin_sub_x.valueChanged.connect(self._on_sub_xy_changed)
        rect_sub_layout.addWidget(self.spin_sub_x)
        
        rect_sub_layout.addWidget(QLabel("Substrate Half-Y:"))
        self.spin_sub_y = QDoubleSpinBox(self.widget_rect_substrate)
        self.spin_sub_y.setRange(0.0, 5000.0)
        self.spin_sub_y.setSingleStep(0.5)
        self.spin_sub_y.setDecimals(3)
        self.spin_sub_y.valueChanged.connect(self._on_sub_xy_changed)
        rect_sub_layout.addWidget(self.spin_sub_y)
        aperture_layout.addWidget(self.widget_rect_substrate)

        aper_type_layout = QHBoxLayout()
        aper_type_layout.addWidget(QLabel("Aperture Shape:"))
        self.combo_aperture_type = QComboBox(aperture_group)
        self.combo_aperture_type.addItems(["Circular", "Rectangular", "Annular", "Elliptical", "Spider", "Point"])
        self.combo_aperture_type.setToolTip("Optical aperture geometry: Circular, Rectangular slit, Annular obscuration, Elliptical, Spider, or Point (zero extent).")
        self.combo_aperture_type.currentIndexChanged.connect(self._on_aperture_type_combo_changed)
        aper_type_layout.addWidget(self.combo_aperture_type)
        aperture_layout.addLayout(aper_type_layout)

        sub_shape_layout = QHBoxLayout()
        sub_shape_layout.addWidget(QLabel("Element/Substrate Shape:"))
        self.combo_substrate_shape = QComboBox(aperture_group)
        self.combo_substrate_shape.addItems(["Auto", "Circular", "Rectangular"])
        self.combo_substrate_shape.setToolTip("Physical shape of the glass element or mirror substrate bounding the semi-diameter.")
        self.combo_substrate_shape.currentIndexChanged.connect(self._on_substrate_shape_combo_changed)
        sub_shape_layout.addWidget(self.combo_substrate_shape)
        aperture_layout.addLayout(sub_shape_layout)

        # Rectangular Aperture Half-Span Controls (X and Y)
        self.widget_rect_aperture = QWidget(aperture_group)
        rect_layout = QHBoxLayout(self.widget_rect_aperture)
        rect_layout.setContentsMargins(0, 2, 0, 2)
        rect_layout.addWidget(QLabel("Half-X (mm):"))
        self.spin_aperture_x = QDoubleSpinBox(self.widget_rect_aperture)
        self.spin_aperture_x.setRange(0.01, 5000.0)
        self.spin_aperture_x.setSingleStep(0.5)
        self.spin_aperture_x.setDecimals(3)
        self.spin_aperture_x.setSuffix(" mm")
        self.spin_aperture_x.setToolTip("Rectangular aperture half-width along X (Full Width = 2 × X)")
        self.spin_aperture_x.valueChanged.connect(self._on_aperture_x_changed)
        rect_layout.addWidget(self.spin_aperture_x)

        rect_layout.addWidget(QLabel("Half-Y (mm):"))
        self.spin_aperture_y = QDoubleSpinBox(self.widget_rect_aperture)
        self.spin_aperture_y.setRange(0.01, 5000.0)
        self.spin_aperture_y.setSingleStep(0.5)
        self.spin_aperture_y.setDecimals(3)
        self.spin_aperture_y.setSuffix(" mm")
        self.spin_aperture_y.setToolTip("Rectangular aperture half-height along Y (Full Height = 2 × Y)")
        self.spin_aperture_y.valueChanged.connect(self._on_aperture_y_changed)
        rect_layout.addWidget(self.spin_aperture_y)

        self.widget_rect_aperture.setVisible(False)
        aperture_layout.addWidget(self.widget_rect_aperture)

        self.lbl_aperture_readout = QLabel("Clear Aperture: Ø = 20.00 mm", aperture_group)
        if QT_AVAILABLE:
            self.lbl_aperture_readout.setStyleSheet("color: #1565c0; font-size: 11px;")
        aperture_layout.addWidget(self.lbl_aperture_readout)

        aper_btns = QHBoxLayout()
        self.btn_configure_aperture = QPushButton("⚙️ Define Aperture / Stop...", aperture_group)
        self.btn_configure_aperture.setToolTip("Open full Aperture & Stop definition dialog (slits, central obscuration, vanes).")
        self.btn_configure_aperture.clicked.connect(self._open_aperture_dialog)
        aper_btns.addWidget(self.btn_configure_aperture)

        self.btn_toggle_stop = QPushButton("⚡ Toggle Stop", aperture_group)
        self.btn_toggle_stop.setToolTip("Toggle this surface as the System Aperture Stop.")
        self.btn_toggle_stop.clicked.connect(self._on_toggle_stop_clicked)
        aper_btns.addWidget(self.btn_toggle_stop)
        aperture_layout.addLayout(aper_btns)

        main_layout.addWidget(aperture_group)

        # ----------------------------------------------------
        # 5. Diffraction Grating & Diffracted Optical Axis Wavelength
        # ----------------------------------------------------
        self.grating_group = QGroupBox("Diffraction Grating & Optical Axis Wavelength", container)
        grating_layout = QVBoxLayout(self.grating_group)

        grating_wl_layout = QHBoxLayout()
        grating_wl_layout.addWidget(QLabel("🎯 Primary Diffracted λ:"))
        self.grating_wl_spin = QDoubleSpinBox(self.grating_group)
        self.grating_wl_spin.setRange(100.0, 25000.0)
        self.grating_wl_spin.setValue(1064.0)
        self.grating_wl_spin.setSingleStep(10.0)
        self.grating_wl_spin.setDecimals(1)
        self.grating_wl_spin.setSuffix(" nm")
        self.grating_wl_spin.setToolTip("Select the primary wavelength diffracted along the downstream optical axis.")
        self.grating_wl_spin.valueChanged.connect(self._on_grating_wl_spin_changed)
        grating_wl_layout.addWidget(self.grating_wl_spin)

        self.btn_grating_auto_wl = QPushButton("Auto (System λ)", self.grating_group)
        self.btn_grating_auto_wl.setToolTip("Follow the system's global primary wavelength automatically.")
        self.btn_grating_auto_wl.clicked.connect(lambda: self._set_grating_wavelength_preset(None))
        grating_wl_layout.addWidget(self.btn_grating_auto_wl)
        grating_layout.addLayout(grating_wl_layout)

        # Quick preset buttons for laser / NIR Raman lines
        presets_layout = QHBoxLayout()
        presets_layout.addWidget(QLabel("Wavelength Presets:"))
        for name_p, wl_p in [("532 nm", 532.0), ("633 nm", 632.8), ("785 nm", 785.0), ("1064 nm", 1064.0), ("1310 nm", 1310.0), ("1550 nm", 1550.0)]:
            btn_p = QPushButton(name_p, self.grating_group)
            btn_p.clicked.connect(lambda checked, val=wl_p: self._set_grating_wavelength_preset(val))
            presets_layout.addWidget(btn_p)
        grating_layout.addLayout(presets_layout)

        # Grating Tilt Angle & Fold Presets (Littrow, 30°, 45°)
        angles_layout = QHBoxLayout()
        self.btn_grating_littrow = QPushButton("📐 Auto Littrow Angle", self.grating_group)
        self.btn_grating_littrow.setToolTip("Calculate and set the exact retro-reflecting Littrow angle: sin(θ) = m*λ / (2*d)")
        self.btn_grating_littrow.clicked.connect(self._apply_grating_littrow_angle)
        angles_layout.addWidget(self.btn_grating_littrow)

        self.btn_grating_fold_30 = QPushButton("Set 30° Fold", self.grating_group)
        self.btn_grating_fold_30.setToolTip("Sets grating tilt to achieve an exact 30° total beam fold angle.")
        self.btn_grating_fold_30.clicked.connect(lambda: self._apply_grating_target_fold_angle(30.0))
        angles_layout.addWidget(self.btn_grating_fold_30)

        self.btn_grating_fold_37 = QPushButton("Set 37.4° Fold (NIR)", self.grating_group)
        self.btn_grating_fold_37.setToolTip("Sets grating tilt to achieve 37.42° fold (13.0° incidence at 1064nm).")
        self.btn_grating_fold_37.clicked.connect(lambda: self._apply_grating_target_fold_angle(37.42))
        angles_layout.addWidget(self.btn_grating_fold_37)

        self.btn_grating_fold_45 = QPushButton("Set 45° Fold", self.grating_group)
        self.btn_grating_fold_45.setToolTip("Sets grating tilt to achieve an exact 45° total beam fold angle.")
        self.btn_grating_fold_45.clicked.connect(lambda: self._apply_grating_target_fold_angle(45.0))
        angles_layout.addWidget(self.btn_grating_fold_45)
        grating_layout.addLayout(angles_layout)

        # Live Angle Readout
        self.lbl_grating_angles = QLabel("Incident: 13.00° | Diffracted: 24.42° | Total Fold: 37.42°", self.grating_group)
        if QT_AVAILABLE:
            self.lbl_grating_angles.setStyleSheet("font-weight: bold; color: #00796B; padding-top: 4px;")
        grating_layout.addWidget(self.lbl_grating_angles)

        main_layout.addWidget(self.grating_group)

        # ----------------------------------------------------
        # 5. Quick-Fold Beam Presets (Mirrors & Fold Elements)
        # ----------------------------------------------------
        fold_group = QGroupBox("Mirror Quick-Fold Presets", container)
        fold_layout = QGridLayout(fold_group)

        self.btn_fold_90_yz = QPushButton("90° Fold (YZ)", fold_group)
        self.btn_fold_90_yz.setToolTip("Sets mirror tilt_x = 45° to deflect beam 90° in the YZ vertical plane.")
        self.btn_fold_90_yz.clicked.connect(lambda: self.apply_quick_fold(90.0, "YZ"))
        fold_layout.addWidget(self.btn_fold_90_yz, 0, 0)

        self.btn_fold_90_xz = QPushButton("90° Fold (XZ)", fold_group)
        self.btn_fold_90_xz.setToolTip("Sets mirror tilt_y = 45° to deflect beam 90° in the XZ horizontal plane.")
        self.btn_fold_90_xz.clicked.connect(lambda: self.apply_quick_fold(90.0, "XZ"))
        fold_layout.addWidget(self.btn_fold_90_xz, 0, 1)

        self.btn_fold_45 = QPushButton("45° Fold (YZ)", fold_group)
        self.btn_fold_45.setToolTip("Sets mirror tilt_x = 22.5° for a 45° deflection.")
        self.btn_fold_45.clicked.connect(lambda: self.apply_quick_fold(45.0, "YZ"))
        fold_layout.addWidget(self.btn_fold_45, 1, 0)

        self.btn_retro = QPushButton("Retro-Reflector (180°)", fold_group)
        self.btn_retro.setToolTip("Sets normal incidence for direct beam return.")
        self.btn_retro.clicked.connect(lambda: self.apply_quick_fold(180.0, "YZ"))
        fold_layout.addWidget(self.btn_retro, 1, 1)

        main_layout.addWidget(fold_group)

        # ----------------------------------------------------
        # 6. One-Click Mirror Aiming
        # ----------------------------------------------------
        aim_group = QGroupBox("Analytical Mirror Aiming", container)
        aim_layout = QVBoxLayout(aim_group)

        aim_target_layout = QHBoxLayout()
        aim_target_layout.addWidget(QLabel("Target Surface:"))
        self.aim_target_combo = QComboBox(aim_group)
        aim_target_layout.addWidget(self.aim_target_combo)
        aim_layout.addLayout(aim_target_layout)

        self.btn_aim_mirror = QPushButton("Aim Selected Mirror at Target", aim_group)
        self.btn_aim_mirror.setToolTip("Analytically calculates exact mirror tilts so the reflected optical axis hits the target surface center.")
        self.btn_aim_mirror.clicked.connect(self.on_aim_mirror_clicked)
        aim_layout.addWidget(self.btn_aim_mirror)

        self.aim_status_label = QLabel("Aim Status: Ready", aim_group)
        aim_layout.addWidget(self.aim_status_label)

        main_layout.addWidget(aim_group)

        main_layout.addStretch()

        # Connect sliderReleased on all interactive sliders
        for _s in [
            getattr(self, "slider_galvo_angle", None),
            getattr(self, "tilt_x_slider", None),
            getattr(self, "tilt_y_slider", None),
            getattr(self, "tilt_z_slider", None),
            getattr(self, "dec_x_slider", None),
            getattr(self, "dec_y_slider", None),
            getattr(self, "dec_z_slider", None),
            getattr(self, "thick_slider", None),
        ]:
            if _s is not None and hasattr(_s, "sliderReleased") and hasattr(_s.sliderReleased, "connect"):
                try:
                    _s.sliderReleased.connect(self._on_slider_released)
                except Exception:
                    pass

        if hasattr(scroll_area, "setWidget"):
            scroll_area.setWidget(container)
            self.setWidget(scroll_area)
        else:
            self.setWidget(container)

    def set_dark_mode(self, enabled: bool) -> None:
        """Switch between dark and light background themes."""
        self.dark_mode = bool(enabled)
        self._update_theme_styles()

    def _update_theme_styles(self) -> None:
        """Update styled elements to match the current theme."""
        if not QT_AVAILABLE:
            return
        if getattr(self, 'dark_mode', False):
            if hasattr(self, 'lbl_galvo_mode') and self.lbl_galvo_mode:
                self.lbl_galvo_mode.setStyleSheet("font-weight: bold; color: #2dd4bf;")
            if hasattr(self, 'lbl_galvo_readout') and self.lbl_galvo_readout:
                self.lbl_galvo_readout.setStyleSheet("font-weight: bold; color: #38bdf8; padding: 2px;")
            if hasattr(self, 'btn_delete_lens') and self.btn_delete_lens:
                self.btn_delete_lens.setStyleSheet(
                    "QPushButton { color: #f87171; font-weight: bold; padding: 4px; border: 1px solid #7f1d1d; border-radius: 3px; } "
                    "QPushButton:hover { background-color: #450a0a; }"
                )
            if hasattr(self, 'lbl_aperture_readout') and self.lbl_aperture_readout:
                self.lbl_aperture_readout.setStyleSheet("color: #60a5fa; font-size: 11px;")
            if hasattr(self, 'lbl_grating_angles') and self.lbl_grating_angles:
                self.lbl_grating_angles.setStyleSheet("font-weight: bold; color: #2dd4bf; padding-top: 4px;")
        else:
            if hasattr(self, 'lbl_galvo_mode') and self.lbl_galvo_mode:
                self.lbl_galvo_mode.setStyleSheet("font-weight: bold; color: #00796B;")
            if hasattr(self, 'lbl_galvo_readout') and self.lbl_galvo_readout:
                self.lbl_galvo_readout.setStyleSheet("font-weight: bold; color: #1976D2; padding: 2px;")
            if hasattr(self, 'btn_delete_lens') and self.btn_delete_lens:
                self.btn_delete_lens.setStyleSheet(
                    "QPushButton { color: #b71c1c; font-weight: bold; padding: 4px; } "
                    "QPushButton:hover { background-color: #ffebee; }"
                )
            if hasattr(self, 'lbl_aperture_readout') and self.lbl_aperture_readout:
                self.lbl_aperture_readout.setStyleSheet("color: #1565c0; font-size: 11px;")
            if hasattr(self, 'lbl_grating_angles') and self.lbl_grating_angles:
                self.lbl_grating_angles.setStyleSheet("font-weight: bold; color: #00796B; padding-top: 4px;")

    # -----------------------------------------------------------------------
    # Slider Throttling & Debouncing Slots
    # -----------------------------------------------------------------------

    def _schedule_slider_change(self):
        """Throttle (33 ms / 30 Hz) and debounce (150 ms) high-frequency slider scrub events."""
        self._slider_change_pending = True

        # Restart debounce timer (fires 150 ms after continuous scrubbing ceases)
        if hasattr(self, "_debounce_timer") and self._debounce_timer is not None:
            self._debounce_timer.start(150)

        # Start throttle timer if not already running (fires at ~30 Hz / 33 ms intervals)
        if hasattr(self, "_throttle_timer") and self._throttle_timer is not None:
            if not self._throttle_timer.isActive():
                self._throttle_timer.start(33)

    def _on_throttle_timeout(self):
        """Fires at ~30 Hz (33 ms) during active slider scrubbing."""
        if self._slider_change_pending:
            self._slider_change_pending = False
            self._emit_change()

    def _on_debounce_timeout(self):
        """Fires 150 ms after slider interaction ceases."""
        if hasattr(self, "_throttle_timer") and self._throttle_timer is not None:
            self._throttle_timer.stop()
        if self._slider_change_pending:
            self._slider_change_pending = False
            self._emit_change()

    def _on_slider_released(self):
        """Fires immediately when the user releases the mouse button on any slider."""
        if hasattr(self, "_throttle_timer") and self._throttle_timer is not None:
            self._throttle_timer.stop()
        if hasattr(self, "_debounce_timer") and self._debounce_timer is not None:
            self._debounce_timer.stop()
        self._slider_change_pending = False
        self._emit_change()

    # -----------------------------------------------------------------------
    # Surface & Model Binding
    # -----------------------------------------------------------------------

    def set_model(self, model: LensDataTableModel):
        """Bind the editor to a LensDataTableModel."""
        if self.model is not None and hasattr(self.model, "system_updated"):
            try:
                self.model.system_updated.disconnect(self._on_model_updated)
            except Exception:
                pass
        self.model = model
        if self.model is not None and hasattr(self.model, "system_updated"):
            try:
                self.model.system_updated.connect(self._on_model_updated)
            except Exception:
                pass
        self._rebuild_surface_list()

    def set_surface_stack(self, stack: SurfaceStack):
        """Update the underlying model with a new SurfaceStack."""
        if self.model is not None:
            self.model.set_surface_stack(stack)
        self._rebuild_surface_list()

    def set_system(self, system: Any):
        """Set system (accepts OpticalSystem or SurfaceStack)."""
        if hasattr(system, "surface_stack"):
            self.set_surface_stack(system.surface_stack)
        elif isinstance(system, SurfaceStack):
            self.set_surface_stack(system)

    def _on_model_updated(self):
        """Slot triggered when the LDE model updates."""
        if self._syncing_ui:
            return
        self._rebuild_surface_list()

    def update_ui(self):
        """Public method to rebuild surface dropdown and sync controls."""
        self._rebuild_surface_list()

    def _rebuild_surface_list(self):
        """Refresh the surface dropdown list and sync current surface values."""
        if self._syncing_ui:
            return

        self._syncing_ui = True
        try:
            current_idx = self._selected_index
            self.surface_combo.clear()
            self.aim_target_combo.clear()
            if hasattr(self, 'lens_element_combo'):
                self.lens_element_combo.clear()

            stack = self._get_stack()
            if not stack or len(stack.surfaces) == 0:
                self.surface_info_label.setText("No surfaces available")
                if hasattr(self, 'enable_lens_edition_check'):
                    self.enable_lens_edition_check.setEnabled(False)
                if hasattr(self, 'btn_delete_lens'):
                    self.btn_delete_lens.setEnabled(False)
                if hasattr(self, 'btn_flip_lens'):
                    self.btn_flip_lens.setEnabled(False)
                return

            from optics_app.engine.element_colors import get_surface_element_group, get_lens_elements, format_surface_material_display
            for i, surf in enumerate(stack.surfaces):
                name = surf.name or f"Surface {i}"
                stype_name = surf.surface_type.name if hasattr(surf.surface_type, 'name') else str(surf.surface_type)
                group_info = get_surface_element_group(stack.surfaces, i)
                g_type = group_info.get("group_type", "")
                e_id = group_info.get("element_id", 0)
                
                g_name = group_info.get("group_name", "") or f"Lens {e_id + 1}"
                if g_type == "LENS":
                    prefix = f"● [{g_name}] "
                    mat_desc = str(surf.glass or "AIR")
                    tag = f"{prefix}S{i}: {name} ({stype_name}, {mat_desc})"
                elif g_type in ["STOP", "DETECTOR", "MIRROR", "GRATING", "OBJECT"]:
                    prefix = f"● [{g_name.capitalize() if g_name else g_type.capitalize()}] "
                    mat_desc = format_surface_material_display(surf, i, len(stack.surfaces))
                    tag = f"{prefix}S{i}: {name} ({stype_name} · {mat_desc})"
                else:
                    prefix = ""
                    mat_desc = format_surface_material_display(surf, i, len(stack.surfaces))
                    tag = f"{prefix}S{i}: {name} ({stype_name}, {mat_desc})"

                self.surface_combo.addItem(tag, i)
                self.aim_target_combo.addItem(f"{prefix}S{i}: {name}", i)

            # Rebuild lens element dropdown
            if hasattr(self, 'lens_element_combo'):
                unlocked = getattr(self.model, '_unlocked_lens_groups', set()) if self.model else set()
                lens_elems = get_lens_elements(stack.surfaces, unlocked_groups=unlocked)
                for elem in lens_elems:
                    self.lens_element_combo.addItem(elem["label"], elem)
                
                has_lenses = len(lens_elems) > 0
                if not has_lenses:
                    self.lens_element_combo.addItem("(No multi-surface lens assembly)", None)
                if hasattr(self, 'enable_lens_edition_check'):
                    self.enable_lens_edition_check.setEnabled(has_lenses)
                if hasattr(self, 'btn_delete_lens'):
                    self.btn_delete_lens.setEnabled(has_lenses)
                if hasattr(self, 'btn_flip_lens'):
                    self.btn_flip_lens.setEnabled(has_lenses)

            # Default aim target to detector (last surface)
            if self.aim_target_combo.count() > 0:
                self.aim_target_combo.setCurrentIndex(self.aim_target_combo.count() - 1)

            if 0 <= current_idx < len(stack.surfaces):
                self.surface_combo.setCurrentIndex(current_idx)
            else:
                self._selected_index = 0
                self.surface_combo.setCurrentIndex(0)

            self._sync_lens_combo_to_surface(self._selected_index)
            self._sync_controls_from_surface(self._selected_index)
        finally:
            self._syncing_ui = False

    def _get_stack(self) -> Optional[SurfaceStack]:
        """Retrieve current SurfaceStack from model."""
        if self.model is not None and hasattr(self.model, "get_surface_stack"):
            return self.model.get_surface_stack()
        return None

    def _get_surface(self, index: int) -> Optional[Surface]:
        """Retrieve surface object at index."""
        stack = self._get_stack()
        if stack and 0 <= index < len(stack.surfaces):
            return stack.surfaces[index]
        return None

    def get_selected_surface_index(self) -> int:
        """Return currently selected surface index."""
        return self._selected_index

    def _sync_lens_combo_to_surface(self, surface_index: int) -> bool:
        """Select the assembly containing ``surface_index`` without changing that surface selection."""
        combo = getattr(self, "lens_element_combo", None)
        if combo is None:
            return False
        for combo_index in range(combo.count()):
            data = combo.itemData(combo_index)
            if isinstance(data, dict) and surface_index in data.get("surface_indices", []):
                previous_sync = self._syncing_ui
                self._syncing_ui = True
                try:
                    combo.setCurrentIndex(combo_index)
                finally:
                    self._syncing_ui = previous_sync
                return True
        return False

    def _current_lens_target(self) -> tuple:
        """Return the clicked lens surface and label; never trust a stale assembly dropdown."""
        selected = int(getattr(self, "_selected_index", 0))
        stack = self._get_stack()
        if stack is not None:
            span = stack.find_lens_element_span(selected)
            if span:
                from optics_app.engine.element_colors import get_surface_element_group
                group = get_surface_element_group(stack.surfaces, selected)
                self._sync_lens_combo_to_surface(selected)
                return selected, str(group.get("group_name", "") or f"Lens at Surface {selected}")

        combo = getattr(self, "lens_element_combo", None)
        data = combo.currentData() if combo is not None and combo.count() > 0 else None
        if isinstance(data, dict) and data.get("surface_indices"):
            return int(data["surface_indices"][0]), str(data.get("group_name", "") or "Selected Lens")
        return selected, f"Surface {selected}"

    def set_selected_surface(self, index: int):
        """Select a surface programmatically."""
        stack = self._get_stack()
        if not stack:
            return
        if 0 <= index < len(stack.surfaces):
            self._selected_index = index
            if hasattr(self.surface_combo, "setCurrentIndex"):
                self.surface_combo.setCurrentIndex(index)
            self._sync_lens_combo_to_surface(index)
            self._sync_controls_from_surface(index)
            if hasattr(self, "surface_selected") and self.surface_selected is not None:
                self.surface_selected.emit(index)

    def _on_surface_combo_changed(self, combo_index: int):
        """Slot triggered when user chooses a surface from the dropdown."""
        if self._syncing_ui:
            return
        surf_idx = self.surface_combo.currentData()
        if surf_idx is None:
            surf_idx = combo_index
        if surf_idx is not None and surf_idx >= 0:
            self._selected_index = surf_idx
            self._sync_lens_combo_to_surface(surf_idx)
            self._sync_controls_from_surface(surf_idx)
            if hasattr(self, "surface_selected") and self.surface_selected is not None:
                self.surface_selected.emit(surf_idx)

    def _sync_mechanical_state_controls(self, surface: Surface) -> None:
        """Populate generic mechanism states for the selected component."""
        from optics_app.engine.mechanisms import available_states, active_state_name, is_dmd

        states = available_states(surface)
        has_states = bool(states)
        self.mechanism_group.setEnabled(has_states)
        self.mechanical_state_combo.clear()
        if not has_states:
            self.mechanical_state_status.setText("This component has no configured mechanical states.")
            self.btn_mech_1.setEnabled(False)
            self.btn_mech_2.setEnabled(False)
            if hasattr(self, 'btn_mech_3'):
                self.btn_mech_3.setEnabled(False)
            return

        current = active_state_name(surface)
        selected_index = 0
        for index, (name, definition) in enumerate(states.items()):
            self.mechanical_state_combo.addItem(str(definition.get("label", name)), name)
            if name == current:
                selected_index = index
        self.mechanical_state_combo.setCurrentIndex(selected_index)
        try:
            from optics_app.engine.mechanisms import is_dmd
            from optics_app.engine.plm import is_plm
        except ImportError:
            from v2.engine.mechanisms import is_dmd
            from v2.engine.plm import is_plm

        if is_dmd(surface) or is_plm(surface) or getattr(surface, 'is_plm', False):
            # TI devices (PLM / DMD) are controlled exclusively from the TI Modulator Studio
            self.btn_mech_1.setVisible(False)
            self.btn_mech_2.setVisible(False)
            if hasattr(self, 'btn_mech_3'):
                self.btn_mech_3.setVisible(False)
            self.mechanical_state_combo.setVisible(False)
            self.btn_apply_mechanical_state.setVisible(False)
            if hasattr(self, 'btn_open_ti_studio'):
                self.btn_open_ti_studio.setVisible(True)
            dname = "TI DLP4500 DMD (1-Bit Binary)" if is_dmd(surface) else "TI DLP670S PLM (4-Bit Phase)"
            self.mechanical_state_status.setText(f"🔲 {dname} — Active State: {current}\nControlled exclusively from the TI Modulator Studio.")
            return

        self.btn_mech_1.setVisible(True)
        self.btn_mech_2.setVisible(True)
        if hasattr(self, 'btn_mech_3'):
            self.btn_mech_3.setVisible(True)
        self.mechanical_state_combo.setVisible(True)
        self.btn_apply_mechanical_state.setVisible(True)
        if hasattr(self, 'btn_open_ti_studio'):
            self.btn_open_ti_studio.setVisible(False)

        self.btn_mech_1.setText("MECH ON → Detector")
        self.btn_mech_2.setText("MECH OFF → Dump")
        if hasattr(self, 'btn_mech_3'):
            self.btn_mech_3.setText("MECH FLAT (0°)")

        mech = bool(available_states(surface))
        if mech:
            self.btn_mech_1.setEnabled("ON" in states)
            self.btn_mech_2.setEnabled("OFF" in states)
            if hasattr(self, 'btn_mech_3'):
                self.btn_mech_3.setEnabled("FLAT" in states)
        else:
            has_system_mech = False
            stack = self.model.get_surface_stack() if hasattr(self.model, "get_surface_stack") else None
            if stack and stack.surfaces:
                for s in stack.surfaces:
                    if available_states(s):
                        has_system_mech = True
                        break
            self.btn_mech_1.setEnabled(has_system_mech)
            self.btn_mech_2.setEnabled(has_system_mech)
            if hasattr(self, 'btn_mech_3'):
                self.btn_mech_3.setEnabled(has_system_mech)
        route = states.get(current, {}).get("route", "unspecified")
        try:
            from optics_app.engine.plm import is_plm
        except ImportError:
            from v2.engine.plm import is_plm
        if is_plm(surface):
            piston_val = states.get(current, {}).get("piston_nm", 0.0)
            phase_val = states.get(current, {}).get("phase_deg", 0.0)
            desc = states.get(current, {}).get("description", "")
            self.mechanical_state_status.setText(
                f"TI DLP670S 4-Bit PLM: State={current} ({desc}) | Piston={piston_val:.1f} nm ({phase_val:.1f}°) | Route: {route}"
            )
        elif is_dmd(surface):
            try:
                from optics_app.engine.mechanisms import dmd4500_reflection_angles
            except ImportError:
                from v2.engine.mechanisms import dmd4500_reflection_angles
            alpha_tilt = 12.0 if current == "ON" else (-12.0 if current == "OFF" else 0.0)
            angles = dmd4500_reflection_angles(24.0, current)
            self.mechanical_state_status.setText(
                f"TI DLP4500 DMD: State={current} (Tilt: {alpha_tilt:+.1f}°) | "
                f"θ_in={angles['input_angle_deg']:.1f}° → θ_out={angles['output_angle_deg']:.1f}° ({route})"
            )
        else:
            self.mechanical_state_status.setText(f"Current: {current} | Optical route: {route}")

    def _open_ti_studio(self) -> None:
        win = self.window()
        if hasattr(win, "on_show_plm_studio"):
            win.on_show_plm_studio()
        elif hasattr(self, "main_window") and hasattr(self.main_window, "on_show_plm_studio"):
            self.main_window.on_show_plm_studio()

    def _on_apply_mechanical_state(self) -> bool:
        state = self.mechanical_state_combo.currentData()
        return self._apply_mechanical_state(state)

    def _apply_mechanical_state(self, state: Any) -> bool:
        """Apply a state through the table model so undo/redo remains atomic."""
        if self._syncing_ui or not self.model or not state:
            return False
        try:
            target_index = self._selected_index
            if str(state) in ("ON", "OFF"):
                from optics_app.engine.mechanisms import is_dmd, available_states
                curr_surf = self._get_surface(target_index)
                if curr_surf is None or not available_states(curr_surf):
                    stack = self.model.get_surface_stack() if hasattr(self.model, "get_surface_stack") else None
                    if stack and stack.surfaces:
                        for s_idx, s in enumerate(stack.surfaces):
                            if available_states(s):
                                target_index = s_idx
                                self.set_selected_surface(s_idx)
                                break
            if not self.model.set_mechanical_state(target_index, str(state)):
                return False
        except (KeyError, ValueError, IndexError) as exc:
            self.mechanical_state_status.setText(f"State change failed: {exc}")
            return False
        self._sync_controls_from_surface(self._selected_index)
        self.component_changed.emit(self._selected_index)
        return True

    def _on_designate_galvo_clicked(self):
        """Designate the selected surface as a galvanometer scanning mirror."""
        surf = self._get_surface(self._selected_index)
        if surf is None:
            return
        axis_txt = self.combo_galvo_axis.currentText() if hasattr(self, "combo_galvo_axis") else "X"
        axis = "Y" if axis_txt.startswith("Y") else "X"
        min_ang = float(self.spin_galvo_min.value()) if hasattr(self, "spin_galvo_min") else -10.0
        max_ang = float(self.spin_galvo_max.value()) if hasattr(self, "spin_galvo_max") else 10.0
        step = float(self.spin_galvo_step.value()) if hasattr(self, "spin_galvo_step") else 1.0
        configure_galvo_scanner(
            surf,
            scan_axis=axis,
            min_angle_deg=min_ang,
            max_angle_deg=max_ang,
            step_size_deg=step,
        )
        self._sync_controls_from_surface(self._selected_index)
        self.component_changed.emit(self._selected_index)

    def _on_galvo_config_changed(self):
        """Update galvo configuration on the active surface."""
        if self._syncing_ui:
            return
        surf = self._get_surface(self._selected_index)
        if surf is None or not (is_galvo(surf) or hasattr(surf, "galvo_config")):
            return
        axis_txt = self.combo_galvo_axis.currentText() if hasattr(self, "combo_galvo_axis") else "X"
        axis = "Y" if axis_txt.startswith("Y") else "X"
        min_ang = float(self.spin_galvo_min.value()) if hasattr(self, "spin_galvo_min") else -10.0
        max_ang = float(self.spin_galvo_max.value()) if hasattr(self, "spin_galvo_max") else 10.0
        step = float(self.spin_galvo_step.value()) if hasattr(self, "spin_galvo_step") else 1.0
        configure_galvo_scanner(
            surf,
            scan_axis=axis,
            min_angle_deg=min_ang,
            max_angle_deg=max_ang,
            step_size_deg=step,
        )

    def _set_galvo_live_angle(self, ang_deg: float):
        """Directly set live mechanical scan angle."""
        if hasattr(self, "spin_galvo_angle"):
            self.spin_galvo_angle.setValue(ang_deg)

    def _step_galvo_live_angle(self, delta_deg: float):
        """Increment/decrement live mechanical scan angle by a step."""
        if hasattr(self, "spin_galvo_angle"):
            curr = self.spin_galvo_angle.value()
            self.spin_galvo_angle.setValue(curr + delta_deg)

    def _on_galvo_slider_changed(self, val: int):
        """Handle live slider movement."""
        if self._syncing_ui:
            return
        ang = float(val) / 100.0
        self._syncing_ui = True
        try:
            if hasattr(self, "spin_galvo_angle"):
                self.spin_galvo_angle.setValue(ang)
        finally:
            self._syncing_ui = False
        self._apply_galvo_angle(ang, is_slider=True)

    def _on_galvo_spin_changed(self, deg: float):
        """Handle spinbox angle changes."""
        if self._syncing_ui:
            return
        self._syncing_ui = True
        try:
            if hasattr(self, "slider_galvo_angle"):
                self.slider_galvo_angle.setValue(int(round(deg * 100.0)))
        finally:
            self._syncing_ui = False
        self._apply_galvo_angle(deg, is_slider=False)

    def _apply_galvo_angle(self, deg: float, is_slider: bool = False):
        """Apply live mechanical scan angle to surface tilts and notify visualizers."""
        surf = self._get_surface(self._selected_index)
        if surf is None:
            return
        axis_txt = self.combo_galvo_axis.currentText() if hasattr(self, "combo_galvo_axis") else "X"
        axis = "Y" if axis_txt.startswith("Y") else "X"

        if axis == "Y":
            surf.tilt_x = deg
        else:
            surf.tilt_y = deg

        if hasattr(surf, "galvo_config") and surf.galvo_config is not None:
            surf.galvo_config.current_angle_deg = deg

        if hasattr(self, "lbl_galvo_readout"):
            self.lbl_galvo_readout.setText(f"θ_mech: {deg:+.2f}° | θ_opt: {deg*2.0:+.2f}°")

        self._syncing_ui = True
        try:
            if axis == "Y" and hasattr(self, "tilt_x_spin"):
                self.tilt_x_spin.setValue(deg)
                if hasattr(self, "tilt_x_slider"):
                    self.tilt_x_slider.setValue(int(round(deg * self._TILT_SCALE)))
            elif axis == "X" and hasattr(self, "tilt_y_spin"):
                self.tilt_y_spin.setValue(deg)
                if hasattr(self, "tilt_y_slider"):
                    self.tilt_y_slider.setValue(int(round(deg * self._TILT_SCALE)))
        finally:
            self._syncing_ui = False

        if is_slider:
            self._schedule_slider_change()
        else:
            self._emit_change()

    def _on_galvo_analyze_clicked(self):
        """Switch to the Laser Scanning & Field Aberrations tab in the main window."""
        win = self.window()
        if hasattr(win, "analysis_tabs") and win.analysis_tabs is not None:
            win.analysis_tabs.set_active_tab("Laser Scanning & Field Aberrations")

    def _on_lens_element_combo_changed(self, combo_index: int):
        """Slot triggered when user chooses a lens element from the assembly dropdown."""
        if self._syncing_ui or not hasattr(self, 'lens_element_combo'):
            return
        data = self.lens_element_combo.currentData()
        if not data:
            return
        surfs = data.get("surface_indices", [])
        if surfs:
            first_surf = surfs[0]
            self.set_selected_surface(first_surf)
            is_locked = data.get("is_locked", True)
            if hasattr(self, 'enable_lens_edition_check'):
                self._syncing_ui = True
                try:
                    self.enable_lens_edition_check.setChecked(not is_locked)
                finally:
                    self._syncing_ui = False

    def _on_enable_lens_edition_toggled(self, checked: bool):
        """Slot triggered when toggling the 'Enable Lens Edition' parameter."""
        if self._syncing_ui or not self.model or not hasattr(self, 'lens_element_combo'):
            return
        data = self.lens_element_combo.currentData()
        if not data:
            return
        g_name = data.get("group_name", "")
        surface_indices = data.get("surface_indices", [])
        if not surface_indices:
            return
        # Use the selected element's concrete surface index, not its display name.
        # Multiple identical catalog lenses can legitimately share a group name.
        target_index = int(surface_indices[0])
        # checked = True means enable edition -> locked = False
        self.model.set_lens_locked(target_index, locked=(not checked))
        self._rebuild_surface_list()

    def _on_delete_lens_clicked(self):
        """Slot triggered when clicking 'Delete Lens Element' button."""
        if not self.model or not hasattr(self, 'lens_element_combo'):
            return
        data = self.lens_element_combo.currentData()
        if not data:
            return
        surface_indices = data.get("surface_indices", [])
        if not surface_indices:
            return
        self.model.delete_lens_element(int(surface_indices[0]))
        self._rebuild_surface_list()

    def _on_flip_lens_clicked(self) -> bool:
        """Flip the selected physical lens as one atomic multi-surface group."""
        if not self.model or not hasattr(self, 'lens_element_combo'):
            return False
        data = self.lens_element_combo.currentData()
        if not isinstance(data, dict):
            return False
        surface_indices = data.get("surface_indices", [])
        if not surface_indices:
            return False
        target_index = int(surface_indices[0])
        if not self.model.flip_lens_element(target_index):
            return False
        self._selected_index = target_index
        self._rebuild_surface_list()
        self.set_selected_surface(target_index)
        return True

    def _show_replace_lens_menu(self):
        """Show context/popup menu for replacing the selected lens element."""
        if not QT_AVAILABLE or not self.model:
            return

        # The exact surface clicked in the LDE/Component Editor is the source
        # of truth. The engine resolves it to the complete physical lens span.
        target_index, g_name = self._current_lens_target()

        # If g_name is just generic, try to get name from the actual surface
        surf = self._get_surface(target_index)
        surf_display_name = (getattr(surf, 'name', '') if surf else '') or g_name

        from optics_app.gui.qt_compat import QMenu, QAction
        menu = QMenu(self)

        act_tool = QAction(f"🔍 Open Thorlabs Catalog Tool (Choose Parameters & Preview)...", self)
        act_tool.triggered.connect(lambda: self.open_catalog_lens_dialog(target=target_index, mode="replace"))
        menu.addAction(act_tool)
        menu.addSeparator()

        act_zmx = QAction(f"📁 Replace '{surf_display_name}' with Zemax File (.zmx)...", self)
        act_zmx.triggered.connect(lambda: self._replace_selected_lens_zmx(target_index, surf_display_name))
        menu.addAction(act_zmx)

        act_custom = QAction(f"📄 Replace '{surf_display_name}' with Custom File (.json)...", self)
        act_custom.triggered.connect(lambda: self._replace_selected_lens_custom(target_index, surf_display_name))
        menu.addAction(act_custom)

        exec_fn = getattr(menu, 'exec_', menu.exec)
        button_pos = None
        if hasattr(self, 'btn_replace_lens') and hasattr(self.btn_replace_lens, 'mapToGlobal'):
            try:
                button_pos = self.btn_replace_lens.mapToGlobal(self.btn_replace_lens.rect().bottomLeft())
            except Exception:
                pass
        if button_pos:
            exec_fn(button_pos)
        else:
            exec_fn()

    def _replace_selected_lens_zmx(self, target_index: int, group_name: str):
        if not QT_AVAILABLE:
            return
        from optics_app.gui.qt_compat import QFileDialog, QMessageBox
        file_path, _ = QFileDialog.getOpenFileName(
            self, f"Replace Lens '{group_name}' with Zemax File", "", "Zemax Files (*.zmx);;All Files (*)"
        )
        if not file_path:
            return
        try:
            self.model.replace_lens_from_zmx(target_index, file_path)
            self._rebuild_surface_list()
        except Exception as e:
            QMessageBox.critical(self, "Replace Lens Error", str(e))

    def _replace_selected_lens_custom(self, target_index: int, group_name: str):
        if not QT_AVAILABLE:
            return
        from optics_app.gui.qt_compat import QFileDialog, QMessageBox
        file_path, _ = QFileDialog.getOpenFileName(
            self, f"Replace Lens '{group_name}' with Custom File", "", "Custom Lens / JSON Files (*.json *.txt);;All Files (*)"
        )
        if not file_path:
            return
        try:
            self.model.replace_lens_from_custom_file(target_index, file_path)
            self._rebuild_surface_list()
        except Exception as e:
            QMessageBox.critical(self, "Replace Lens Error", str(e))

    def _replace_selected_lens_catalog(self, target_index: int, part_number: str):
        if not self.model:
            return
        try:
            self.model.replace_lens_from_catalog(target_index, part_number)
            self._rebuild_surface_list()
        except Exception as e:
            if QT_AVAILABLE:
                from optics_app.gui.qt_compat import QMessageBox
                QMessageBox.critical(self, "Replace Lens Error", str(e))

    def open_catalog_lens_dialog(self, target: Any = None, mode: str = "replace"):
        """Open the Thorlabs Catalog Lens Dialog allowing parameter selection."""
        if not QT_AVAILABLE:
            return
        from optics_app.gui.catalog_lens_dialog import CatalogLensDialog
        if target is None:
            if hasattr(self, 'lens_element_combo') and self.lens_element_combo.count() > 0:
                data = self.lens_element_combo.currentData()
                if isinstance(data, dict):
                    target = data.get("group_name") or (data.get("surface_indices", [None])[0])
            if target is None and hasattr(self, '_selected_index'):
                target = self._selected_index
        dialog = CatalogLensDialog(model=self.model, parent=self, target=target, mode=mode)
        exec_fn = getattr(dialog, 'exec_', dialog.exec)
        if exec_fn():
            self._rebuild_surface_list()

    def _sync_controls_from_surface(self, index: int):
        """Populate sliders and spinboxes from surface properties."""
        surf = self._get_surface(index)
        if surf is None:
            return

        self._syncing_ui = True
        try:
            stype = surf.surface_type.name if hasattr(surf.surface_type, 'name') else str(surf.surface_type)
            glass = surf.glass or "AIR"
            
            from optics_app.engine.element_colors import get_surface_element_group, format_surface_material_display
            stack = self._get_stack()
            grp = get_surface_element_group(stack.surfaces, index) if stack else {}
            g_type = grp.get("group_type", "")
            g_name = grp.get("group_name", "")

            # Sync lens element assembly controls
            if hasattr(self, 'lens_element_combo'):
                for i in range(self.lens_element_combo.count()):
                    cdata = self.lens_element_combo.itemData(i)
                    if cdata and index in cdata.get("surface_indices", []):
                        self.lens_element_combo.setCurrentIndex(i)
                        is_locked = cdata.get("is_locked", True)
                        if hasattr(self, 'enable_lens_edition_check'):
                            self.enable_lens_edition_check.setChecked(not is_locked)
                        break

            lock_badge = " [🔒 LOCKED]" if (self.model and self.model.is_lens_locked(index)) else ""
            if g_type == "LENS" or getattr(surf, 'is_glass', False):
                mat_field = f"Glass: {glass}"
            else:
                total_s = len(stack.surfaces) if stack else -1
                mat_desc = format_surface_material_display(surf, index, total_s)
                mat_field = f"Material: {mat_desc}"

            self.surface_info_label.setText(
                f"Type: {stype} | {mat_field} | R={surf.radius:.2f} mm | Semi-D={surf.semi_diameter:.2f} mm{lock_badge}"
            )
            self._sync_mechanical_state_controls(surf)

            # Tilts
            tx = float(getattr(surf, 'tilt_x', 0.0))
            ty = float(getattr(surf, 'tilt_y', 0.0))
            tz = float(getattr(surf, 'tilt_z', 0.0))

            self.tilt_x_spin.setValue(tx)
            self.tilt_x_slider.setValue(int(round(tx * self._TILT_SCALE)))

            self.tilt_y_spin.setValue(ty)
            self.tilt_y_slider.setValue(int(round(ty * self._TILT_SCALE)))

            self.tilt_z_spin.setValue(tz)
            self.tilt_z_slider.setValue(int(round(tz * self._TILT_SCALE)))

            # Decenters
            dx = float(getattr(surf, 'decenter_x', 0.0))
            dy = float(getattr(surf, 'decenter_y', 0.0))
            dz = float(getattr(surf, 'decenter_z', 0.0))
            th = float(getattr(surf, 'thickness', 0.0))

            self.dec_x_spin.setValue(dx)
            self.dec_x_slider.setValue(int(round(dx * self._DEC_SCALE)))

            self.dec_y_spin.setValue(dy)
            self.dec_y_slider.setValue(int(round(dy * self._DEC_SCALE)))

            self.dec_z_spin.setValue(dz)
            self.dec_z_slider.setValue(int(round(dz * self._DEC_SCALE)))

            self.thick_spin.setValue(th)
            self.thick_slider.setValue(int(round(th * self._DEC_SCALE)))

            if hasattr(self, 'lbl_thick_name'):
                if stack and 0 <= index < len(stack.surfaces) - 1:
                    next_s = stack.surfaces[index + 1]
                    self.lbl_thick_name.setText(f"Thickness to S{index+1} ({next_s.name}):")
                    self.lbl_thick_name.setToolTip(f"Distance along optical axis from S{index} ({surf.name}) to S{index+1} ({next_s.name}).")
                else:
                    self.lbl_thick_name.setText("Thickness:")
                    self.lbl_thick_name.setToolTip("Distance along optical axis from this surface vertex to the subsequent surface vertex.")

            # Grating diffracted optical axis wavelength sync
            is_grating = (getattr(surf, 'surface_type', None) == SurfaceType.GRATING) or ("GRATING" in str(getattr(surf, 'surface_type', '')).upper())
            if hasattr(self, 'grating_group'):
                self.grating_group.setVisible(is_grating)
                if is_grating:
                    gwl = getattr(surf, 'grating_primary_wavelength_um', None)
                    if gwl is not None and gwl > 0:
                        self.grating_wl_spin.setValue(gwl * 1000.0)
                    else:
                        sys_wl = getattr(stack, 'primary_wavelength', 0.5876) * 1000.0 if stack else 1064.0
                    self._update_grating_angle_readout()

            # Aperture & Stop Controls sync
            if hasattr(self, 'chk_is_stop'):
                is_stop = (
                    (getattr(surf, 'surface_type', None) == SurfaceType.STOP)
                    or getattr(surf, 'is_stop', False)
                    or getattr(surf, 'is_aperture_stop', False)
                    or "STOP" in str(getattr(surf, 'surface_type', '')).upper()
                )
                self.chk_is_stop.setChecked(is_stop)
            if hasattr(self, 'spin_semi_diam'):
                self.spin_semi_diam.setValue(float(getattr(surf, 'semi_diameter', 10.0)))
            if hasattr(self, 'combo_aperture_type'):
                ap_type = str(getattr(surf, 'aperture_type', 'Circular')).capitalize()
                for i in range(self.combo_aperture_type.count()):
                    if self.combo_aperture_type.itemText(i).capitalize() == ap_type:
                        self.combo_aperture_type.setCurrentIndex(i)
                        break
            if hasattr(self, 'combo_substrate_shape'):
                sub_shape = str(getattr(surf, 'substrate_shape', 'Auto')).capitalize()
                for i in range(self.combo_substrate_shape.count()):
                    if self.combo_substrate_shape.itemText(i).capitalize() == sub_shape:
                        self.combo_substrate_shape.setCurrentIndex(i)
                        break
            
            sub_is_rect = (sub_shape == 'Rectangular')
            if hasattr(self, 'widget_rect_substrate'):
                self.widget_rect_substrate.setVisible(sub_is_rect)
                if sub_is_rect:
                    sd_val = float(getattr(surf, 'semi_diameter', 10.0))
                    hx = getattr(surf, 'semi_diameter_x', 0.0)
                    hy = getattr(surf, 'semi_diameter_y', 0.0)
                    if hx <= 0: hx = sd_val
                    if hy <= 0: hy = sd_val
                    if hasattr(self, 'spin_sub_x'):
                        self.spin_sub_x.blockSignals(True)
                        self.spin_sub_x.setValue(hx)
                        self.spin_sub_x.blockSignals(False)
                    if hasattr(self, 'spin_sub_y'):
                        self.spin_sub_y.blockSignals(True)
                        self.spin_sub_y.setValue(hy)
                        self.spin_sub_y.blockSignals(False)
                        
            is_rect = (ap_type == 'Rectangular')
            if hasattr(self, 'widget_rect_aperture'):
                self.widget_rect_aperture.setVisible(is_rect)
                if is_rect:
                    sd_val = float(getattr(surf, 'semi_diameter', 10.0))
                    hx = getattr(surf, 'aperture_x_half', sd_val)
                    hy = getattr(surf, 'aperture_y_half', sd_val)
                    if hasattr(self, 'spin_aperture_x'):
                        self.spin_aperture_x.blockSignals(True)
                        self.spin_aperture_x.setValue(hx)
                        self.spin_aperture_x.blockSignals(False)
                    if hasattr(self, 'spin_aperture_y'):
                        self.spin_aperture_y.blockSignals(True)
                        self.spin_aperture_y.setValue(hy)
                        self.spin_aperture_y.blockSignals(False)

            if hasattr(self, 'lbl_aperture_readout'):
                sd = float(getattr(surf, 'semi_diameter', 10.0))
                ap_type = str(getattr(surf, 'aperture_type', 'Circular')).capitalize()
                is_stop = (
                    (getattr(surf, 'surface_type', None) == SurfaceType.STOP)
                    or getattr(surf, 'is_stop', False)
                    or getattr(surf, 'is_aperture_stop', False)
                    or "STOP" in str(getattr(surf, 'surface_type', '')).upper()
                    or "(STO)" in (surf.name or "").upper()
                )
                pfx = "🔴 Aperture Stop: " if is_stop else "Clear Aperture: "
                if ap_type == 'Rectangular':
                    hx = getattr(surf, 'aperture_x_half', sd)
                    hy = getattr(surf, 'aperture_y_half', sd)
                    self.lbl_aperture_readout.setText(f"{pfx}Semi-D: X={hx:.3f}, Y={hy:.3f} mm | Full Slit: {2*hx:.3f} × {2*hy:.3f} mm")
                elif ap_type == 'Annular':
                    rin = getattr(surf, 'aperture_r_inner', 0.0)
                    rout = getattr(surf, 'aperture_r_outer', sd)
                    self.lbl_aperture_readout.setText(f"{pfx}Annular: Rin={rin:.3f}, Rout (Semi-D)={rout:.3f} mm")
                elif ap_type == 'Spider':
                    legs = getattr(surf, 'aperture_spider_count', 4)
                    hub = getattr(surf, 'aperture_center_radius', 3.0)
                    self.lbl_aperture_readout.setText(f"{pfx}Spider: Hub={hub:.3f} mm, {legs} legs | Semi-D={sd:.3f} mm")
                elif ap_type == 'Point' or sd <= 0.0:
                    self.lbl_aperture_readout.setText(f"{pfx}Point Object / Image: Ø = 0.00 mm (Zero Extent)")
                else:
                    self.lbl_aperture_readout.setText(f"{pfx}Semi-Diameter R = {sd:.3f} mm (Full Ø = {2*sd:.3f} mm)")

            # Sync Galvo scanner actuator controls
            if hasattr(self, 'galvo_group'):
                is_g = is_galvo(surf)
                has_cfg = hasattr(surf, 'galvo_config') and surf.galvo_config is not None
                is_mirror = (getattr(surf, 'surface_type', None) == SurfaceType.MIRROR or "MIRROR" in str(getattr(surf, 'glass', '')).upper())
                self.galvo_group.setEnabled(is_g or is_mirror or has_cfg)
                if is_g or has_cfg:
                    cfg = getattr(surf, 'galvo_config', None)
                    axis = cfg.scan_axis if cfg else ("Y" if "Y" in getattr(surf, 'name', '') else "X")
                    min_ang = cfg.min_angle_deg if cfg else -10.0
                    max_ang = cfg.max_angle_deg if cfg else 10.0
                    step = cfg.step_size_deg if cfg else 1.0
                    self.lbl_galvo_mode.setText(f"Active Galvo ({axis})")
                    self.combo_galvo_axis.blockSignals(True)
                    self.combo_galvo_axis.setCurrentIndex(1 if axis == "Y" else 0)
                    self.combo_galvo_axis.blockSignals(False)

                    self.spin_galvo_min.blockSignals(True)
                    self.spin_galvo_min.setValue(min_ang)
                    self.spin_galvo_min.blockSignals(False)

                    self.spin_galvo_max.blockSignals(True)
                    self.spin_galvo_max.setValue(max_ang)
                    self.spin_galvo_max.blockSignals(False)

                    self.spin_galvo_step.blockSignals(True)
                    self.spin_galvo_step.setValue(step)
                    self.spin_galvo_step.blockSignals(False)

                    curr_ang = float(getattr(surf, 'tilt_x', 0.0) if axis == "Y" else getattr(surf, 'tilt_y', 0.0))
                    self.spin_galvo_angle.blockSignals(True)
                    self.spin_galvo_angle.setValue(curr_ang)
                    self.spin_galvo_angle.blockSignals(False)

                    self.slider_galvo_angle.blockSignals(True)
                    self.slider_galvo_angle.setValue(int(round(curr_ang * 100.0)))
                    self.slider_galvo_angle.blockSignals(False)

                    self.lbl_galvo_readout.setText(f"θ_mech: {curr_ang:+.2f}° | θ_opt: {curr_ang*2.0:+.2f}°")
                else:
                    self.lbl_galvo_mode.setText("Mirror / Passive (Click to Designate)" if is_mirror else "Not a scanning mirror")
        finally:
            self._syncing_ui = False

    # -----------------------------------------------------------------------
    # Slider & SpinBox Event Handlers
    # -----------------------------------------------------------------------

    def _on_tilt_x_slider_changed(self, val: int):
        if self._syncing_ui:
            return
        deg = float(val) / self._TILT_SCALE
        self._syncing_ui = True
        try:
            self.tilt_x_spin.setValue(deg)
        finally:
            self._syncing_ui = False
        self._apply_tilt_change(tilt_x=deg, is_slider=True)

    def _on_tilt_x_spin_changed(self, deg: float):
        if self._syncing_ui:
            return
        self._syncing_ui = True
        try:
            self.tilt_x_slider.setValue(int(round(deg * self._TILT_SCALE)))
        finally:
            self._syncing_ui = False
        self._apply_tilt_change(tilt_x=deg, is_slider=False)

    def _on_tilt_y_slider_changed(self, val: int):
        if self._syncing_ui:
            return
        deg = float(val) / self._TILT_SCALE
        self._syncing_ui = True
        try:
            self.tilt_y_spin.setValue(deg)
        finally:
            self._syncing_ui = False
        self._apply_tilt_change(tilt_y=deg, is_slider=True)

    def _on_tilt_y_spin_changed(self, deg: float):
        if self._syncing_ui:
            return
        self._syncing_ui = True
        try:
            self.tilt_y_slider.setValue(int(round(deg * self._TILT_SCALE)))
        finally:
            self._syncing_ui = False
        self._apply_tilt_change(tilt_y=deg, is_slider=False)

    def _on_tilt_z_slider_changed(self, val: int):
        if self._syncing_ui:
            return
        deg = float(val) / self._TILT_SCALE
        self._syncing_ui = True
        try:
            self.tilt_z_spin.setValue(deg)
        finally:
            self._syncing_ui = False
        self._apply_tilt_change(tilt_z=deg, is_slider=True)

    def _on_tilt_z_spin_changed(self, deg: float):
        if self._syncing_ui:
            return
        self._syncing_ui = True
        try:
            self.tilt_z_slider.setValue(int(round(deg * self._TILT_SCALE)))
        finally:
            self._syncing_ui = False
        self._apply_tilt_change(tilt_z=deg, is_slider=False)

    def _on_dec_x_slider_changed(self, val: int):
        if self._syncing_ui:
            return
        mm = float(val) / self._DEC_SCALE
        self._syncing_ui = True
        try:
            self.dec_x_spin.setValue(mm)
        finally:
            self._syncing_ui = False
        self._apply_decenter_change(decenter_x=mm, is_slider=True)

    def _on_dec_x_spin_changed(self, mm: float):
        if self._syncing_ui:
            return
        self._syncing_ui = True
        try:
            self.dec_x_slider.setValue(int(round(mm * self._DEC_SCALE)))
        finally:
            self._syncing_ui = False
        self._apply_decenter_change(decenter_x=mm, is_slider=False)

    def _on_dec_y_slider_changed(self, val: int):
        if self._syncing_ui:
            return
        mm = float(val) / self._DEC_SCALE
        self._syncing_ui = True
        try:
            self.dec_y_spin.setValue(mm)
        finally:
            self._syncing_ui = False
        self._apply_decenter_change(decenter_y=mm, is_slider=True)

    def _on_dec_y_spin_changed(self, mm: float):
        if self._syncing_ui:
            return
        self._syncing_ui = True
        try:
            self.dec_y_slider.setValue(int(round(mm * self._DEC_SCALE)))
        finally:
            self._syncing_ui = False
        self._apply_decenter_change(decenter_y=mm, is_slider=False)

    def _on_dec_z_slider_changed(self, val: int):
        if self._syncing_ui:
            return
        mm = float(val) / self._DEC_SCALE
        self._syncing_ui = True
        try:
            self.dec_z_spin.setValue(mm)
        finally:
            self._syncing_ui = False
        self._apply_decenter_change(decenter_z=mm, is_slider=True)

    def _on_dec_z_spin_changed(self, mm: float):
        if self._syncing_ui:
            return
        self._syncing_ui = True
        try:
            self.dec_z_slider.setValue(int(round(mm * self._DEC_SCALE)))
        finally:
            self._syncing_ui = False
        self._apply_decenter_change(decenter_z=mm, is_slider=False)

    def _on_grating_wl_spin_changed(self, val_nm: float):
        if self._syncing_ui:
            return
        surf = self._get_surface(self._selected_index)
        if surf:
            surf.grating_primary_wavelength_um = float(val_nm) / 1000.0
            stack = self._get_stack()
            if stack:
                stack._update_accumulated_positions()
            self._emit_change()

    def _set_grating_wavelength_preset(self, wl_nm: Optional[float]):
        surf = self._get_surface(self._selected_index)
        if not surf:
            return
        if wl_nm is None:
            surf.grating_primary_wavelength_um = None
            stack = self._get_stack()
            sys_wl = getattr(stack, 'primary_wavelength', 0.5876) * 1000.0 if stack else 1064.0
            self._syncing_ui = True
            try:
                self.grating_wl_spin.setValue(sys_wl)
            finally:
                self._syncing_ui = False
        else:
            surf.grating_primary_wavelength_um = float(wl_nm) / 1000.0
            self._syncing_ui = True
            try:
                self.grating_wl_spin.setValue(float(wl_nm))
            finally:
                self._syncing_ui = False
        
        stack = self._get_stack()
        if stack:
            stack._update_accumulated_positions()
        self._update_grating_angle_readout()
        self._emit_change()

    def _update_grating_angle_readout(self):
        """Update live incident, diffracted, and total fold angles display."""
        if not hasattr(self, 'lbl_grating_angles') or not self.lbl_grating_angles:
            return
        surf = self._get_surface(self._selected_index)
        if not surf:
            return
        stack = self._get_stack()
        alpha_deg = float(getattr(surf, 'tilt_x', 0.0))
        gwl_um = getattr(surf, 'grating_primary_wavelength_um', None)
        if gwl_um is None or gwl_um <= 0:
            gwl_um = getattr(stack, 'primary_wavelength', 0.5876) if stack else 1.064
        lines_mm = float(getattr(surf, 'grating_lines_per_mm', 600.0))
        order = int(getattr(surf, 'grating_order', 1))

        # sin(beta) = m * lambda * N - sin(alpha)
        sin_alpha = np.sin(np.radians(alpha_deg))
        sin_beta = (order * gwl_um * (lines_mm / 1000.0)) - sin_alpha

        if abs(sin_beta) <= 1.0:
            beta_deg = float(np.degrees(np.arcsin(sin_beta)))
            total_fold = float(alpha_deg + beta_deg)
            self.lbl_grating_angles.setText(
                f"Incident: {alpha_deg:.2f}° | Diffracted: {beta_deg:.2f}° | Total Fold: {total_fold:.2f}°"
            )
        else:
            self.lbl_grating_angles.setText(
                f"Incident: {alpha_deg:.2f}° | (Diffraction evanescent: sin(beta) = {sin_beta:.3f})"
            )

    def _apply_grating_littrow_angle(self):
        """Set grating tilt_x to exact Littrow angle: sin(theta_L) = m*lambda / (2*d)."""
        surf = self._get_surface(self._selected_index)
        if not surf:
            return
        stack = self._get_stack()
        gwl_um = getattr(surf, 'grating_primary_wavelength_um', None)
        if gwl_um is None or gwl_um <= 0:
            gwl_um = getattr(stack, 'primary_wavelength', 0.5876) if stack else 1.064
        lines_mm = float(getattr(surf, 'grating_lines_per_mm', 600.0))
        order = int(getattr(surf, 'grating_order', 1))

        sin_littrow = (order * gwl_um * (lines_mm / 1000.0)) / 2.0
        if abs(sin_littrow) <= 1.0:
            littrow_deg = float(np.degrees(np.arcsin(sin_littrow)))
            self._apply_tilt_change(tilt_x=littrow_deg)
            self._sync_controls_from_surface(self._selected_index)

    def _apply_grating_target_fold_angle(self, target_fold_deg: float):
        """
        Calculates and applies the exact grating tilt_x needed to achieve a specified total beam fold angle (alpha + beta).
        Formula: cos(alpha - fold/2) = (m * lambda / d) / (2 * sin(fold/2))
        """
        surf = self._get_surface(self._selected_index)
        if not surf:
            return
        stack = self._get_stack()
        gwl_um = getattr(surf, 'grating_primary_wavelength_um', None)
        if gwl_um is None or gwl_um <= 0:
            gwl_um = getattr(stack, 'primary_wavelength', 0.5876) if stack else 1.064
        lines_mm = float(getattr(surf, 'grating_lines_per_mm', 600.0))
        order = int(getattr(surf, 'grating_order', 1))

        m_lambda_d = order * gwl_um * (lines_mm / 1000.0)
        half_fold_rad = np.radians(target_fold_deg / 2.0)
        denom = 2.0 * np.sin(half_fold_rad)

        if denom > 1e-9:
            cos_val = m_lambda_d / denom
            if abs(cos_val) <= 1.0:
                delta_rad = np.arccos(cos_val)
                alpha_rad = half_fold_rad - delta_rad
                alpha_deg = float(np.degrees(alpha_rad))
                self._apply_tilt_change(tilt_x=alpha_deg)
                self._sync_controls_from_surface(self._selected_index)

    def _on_thick_slider_changed(self, val: int):
        if self._syncing_ui:
            return
        mm = float(val) / self._DEC_SCALE
        self._syncing_ui = True
        try:
            self.thick_spin.setValue(mm)
        finally:
            self._syncing_ui = False
        self._apply_property_change("thickness", mm, col=3, is_slider=True)

    def _on_thick_spin_changed(self, mm: float):
        if self._syncing_ui:
            return
        self._syncing_ui = True
        try:
            self.thick_slider.setValue(int(round(mm * self._DEC_SCALE)))
        finally:
            self._syncing_ui = False
        self._apply_property_change("thickness", mm, col=3, is_slider=False)

    # -----------------------------------------------------------------------
    # Parameter Application & Group Rotation/Decenter Logic
    # -----------------------------------------------------------------------

    def _apply_decenter_change(
        self,
        decenter_x: Optional[float] = None,
        decenter_y: Optional[float] = None,
        decenter_z: Optional[float] = None,
        is_slider: bool = False
    ):
        """Apply spatial decenter adjustments, honoring 'Transform Element as Rigid Group' if checked."""
        idx = self._selected_index
        stack = self._get_stack()
        if not stack or not (0 <= idx < len(stack.surfaces)):
            return

        surf = stack.surfaces[idx]
        dx = float(decenter_x if decenter_x is not None else getattr(surf, 'decenter_x', 0.0))
        dy = float(decenter_y if decenter_y is not None else getattr(surf, 'decenter_y', 0.0))
        dz = float(decenter_z if decenter_z is not None else getattr(surf, 'decenter_z', 0.0))

        is_group = self.group_rotation_check.isChecked()
        is_glass_element = getattr(surf, 'is_glass', False) or (idx > 0 and getattr(stack.surfaces[idx - 1], 'is_glass', False))

        if is_group and is_glass_element:
            pivot = "front_vertex"
            if hasattr(self.pivot_combo, "currentText"):
                p_text = self.pivot_combo.currentText().lower()
                if "center" in p_text:
                    pivot = "center_of_mass"
                elif "rear" in p_text:
                    pivot = "rear_vertex"

            self.decenter_selected_element_group(decenter_x=dx, decenter_y=dy, decenter_z=dz, pivot=pivot, is_slider=is_slider)
        else:
            surf.decenter_x = dx
            surf.decenter_y = dy
            surf.decenter_z = dz
            stack._update_accumulated_positions()

            if self.model is not None and hasattr(self.model, "_data") and idx < len(self.model._data):
                if len(self.model._data[idx]) > 7:
                    self.model._data[idx][7] = dx
                if len(self.model._data[idx]) > 8:
                    self.model._data[idx][8] = dy

            if is_slider:
                self._schedule_slider_change()
            else:
                self._emit_change()

    def decenter_selected_element_group(
        self,
        decenter_x: float = 0.0,
        decenter_y: float = 0.0,
        decenter_z: float = 0.0,
        pivot: str = "front_vertex",
        is_slider: bool = False
    ):
        """
        Translate/decenter all surfaces belonging to the selected multi-surface lens element as a rigid body.
        """
        stack = self._get_stack()
        idx = self._selected_index
        if not stack or not (0 <= idx < len(stack.surfaces)):
            return

        front_idx = idx
        while front_idx > 0 and getattr(stack.surfaces[front_idx - 1], 'is_glass', False):
            front_idx -= 1

        if hasattr(self.model, "decenter_lens_element") and not is_slider:
            self.model.decenter_lens_element(
                row=front_idx,
                decenter_x=decenter_x,
                decenter_y=decenter_y,
                decenter_z=decenter_z,
                pivot=pivot
            )
        elif hasattr(stack, "decenter_lens_element"):
            stack.decenter_lens_element(
                front_surface_index=front_idx,
                decenter_x=decenter_x,
                decenter_y=decenter_y,
                decenter_z=decenter_z,
                pivot=pivot
            )
            if self.model is not None and hasattr(self.model, "_data"):
                for s_i, s in enumerate(stack.surfaces):
                    if s_i < len(self.model._data):
                        if len(self.model._data[s_i]) > 7:
                            self.model._data[s_i][7] = s.decenter_x
                        if len(self.model._data[s_i]) > 8:
                            self.model._data[s_i][8] = s.decenter_y
            if is_slider:
                self._schedule_slider_change()
            else:
                self._emit_change()
        else:
            if is_slider:
                self._schedule_slider_change()
            else:
                self._emit_change()

    # -----------------------------------------------------------------------
    # Parameter Application & Group Rotation Logic
    # -----------------------------------------------------------------------

    def _apply_property_change(self, prop_name: str, value: float, col: int, is_slider: bool = False):
        """Directly update a property on the selected surface and sync with model."""
        idx = self._selected_index
        stack = self._get_stack()
        if not stack or not (0 <= idx < len(stack.surfaces)):
            return

        surf = stack.surfaces[idx]
        setattr(surf, prop_name, float(value))
        stack._update_accumulated_positions()

        # Update model data cell if available
        if self.model is not None:
            if hasattr(self.model, "_data") and idx < len(self.model._data) and col < len(self.model._data[idx]):
                self.model._data[idx][col] = float(value)

        if is_slider:
            self._schedule_slider_change()
        else:
            self._emit_change()

    def _apply_tilt_change(
        self,
        tilt_x: Optional[float] = None,
        tilt_y: Optional[float] = None,
        tilt_z: Optional[float] = None,
        is_slider: bool = False
    ):
        """Apply tilt adjustments, honoring 'Rotate Element as Group' if checked."""
        idx = self._selected_index
        stack = self._get_stack()
        if not stack or not (0 <= idx < len(stack.surfaces)):
            return

        surf = stack.surfaces[idx]
        tx = float(tilt_x if tilt_x is not None else getattr(surf, 'tilt_x', 0.0))
        ty = float(tilt_y if tilt_y is not None else getattr(surf, 'tilt_y', 0.0))
        tz = float(tilt_z if tilt_z is not None else getattr(surf, 'tilt_z', 0.0))

        is_group = self.group_rotation_check.isChecked()
        is_glass_element = getattr(surf, 'is_glass', False) or (idx > 0 and getattr(stack.surfaces[idx - 1], 'is_glass', False))

        if is_group and is_glass_element:
            pivot = "front_vertex"
            if hasattr(self.pivot_combo, "currentText"):
                p_text = self.pivot_combo.currentText().lower()
                if "center" in p_text:
                    pivot = "center_of_mass"
                elif "rear" in p_text:
                    pivot = "rear_vertex"

            self.rotate_selected_element_group(tilt_x=tx, tilt_y=ty, tilt_z=tz, pivot=pivot, is_slider=is_slider)
        else:
            surf.tilt_x = tx
            surf.tilt_y = ty
            surf.tilt_z = tz
            stack._update_accumulated_positions()

            if self.model is not None and hasattr(self.model, "_data") and idx < len(self.model._data):
                self.model._data[idx][9] = tx
                self.model._data[idx][10] = ty
                self.model._data[idx][11] = tz

            self._update_grating_angle_readout()
            if is_slider:
                self._schedule_slider_change()
            else:
                self._emit_change()

    def rotate_selected_element_group(
        self,
        tilt_x: float = 0.0,
        tilt_y: float = 0.0,
        tilt_z: float = 0.0,
        pivot: str = "front_vertex",
        is_slider: bool = False
    ):
        """
        Rotate all surfaces belonging to the selected multi-surface lens element as a rigid body.
        """
        stack = self._get_stack()
        idx = self._selected_index
        if not stack or not (0 <= idx < len(stack.surfaces)):
            return

        # 1. Identify front surface of the lens element
        front_idx = idx
        while front_idx > 0 and getattr(stack.surfaces[front_idx - 1], 'is_glass', False):
            front_idx -= 1

        # 2. Identify rear surface of the lens element
        rear_idx = front_idx
        while rear_idx < len(stack.surfaces) - 1 and getattr(stack.surfaces[rear_idx], 'is_glass', False):
            rear_idx += 1

        group_indices = list(range(front_idx, rear_idx + 1))

        # 3. Check if SurfaceStack has rotate_lens_element method
        if hasattr(stack, "rotate_lens_element"):
            try:
                stack.rotate_lens_element(front_surface_index=front_idx, tilt_x=tilt_x, tilt_y=tilt_y, tilt_z=tilt_z, pivot=pivot)
            except Exception:
                self._fallback_element_rotation(stack, group_indices, tilt_x, tilt_y, tilt_z, pivot)
        else:
            self._fallback_element_rotation(stack, group_indices, tilt_x, tilt_y, tilt_z, pivot)

        # Update model data cells
        if self.model is not None and hasattr(self.model, "_data"):
            for s_i in group_indices:
                if s_i < len(self.model._data) and s_i < len(stack.surfaces):
                    s = stack.surfaces[s_i]
                    self.model._data[s_i][7] = float(s.decenter_x)
                    self.model._data[s_i][8] = float(s.decenter_y)
                    self.model._data[s_i][9] = float(s.tilt_x)
                    self.model._data[s_i][10] = float(s.tilt_y)
                    self.model._data[s_i][11] = float(s.tilt_z)

        if is_slider:
            self._schedule_slider_change()
        else:
            self._emit_change()

    def _fallback_element_rotation(
        self,
        stack: SurfaceStack,
        group_indices: List[int],
        tilt_x: float,
        tilt_y: float,
        tilt_z: float,
        pivot: str
    ):
        """Fallback rigid body element rotation around vertex or center of mass."""
        from optics_app.engine.tracer import build_rotation_matrix

        # Calculate element center of mass / thickness along axis
        total_thickness = sum(stack.surfaces[k].thickness for k in group_indices[:-1])
        if pivot == "center_of_mass":
            pivot_z = total_thickness / 2.0
        elif pivot == "rear_vertex":
            pivot_z = total_thickness
        else:
            pivot_z = 0.0

        p_pivot_local = np.array([0.0, 0.0, pivot_z], dtype=float)
        R_rot = build_rotation_matrix(tilt_x, tilt_y, tilt_z)

        accum_z = 0.0
        for k in group_indices:
            s = stack.surfaces[k]
            s.tilt_x = float(tilt_x)
            s.tilt_y = float(tilt_y)
            s.tilt_z = float(tilt_z)

            # Vertex relative to front vertex
            p_vertex_local = np.array([0.0, 0.0, accum_z], dtype=float)
            p_rel_pivot = p_vertex_local - p_pivot_local
            p_rot = p_pivot_local + R_rot @ p_rel_pivot

            s.decenter_x = float(p_rot[0])
            s.decenter_y = float(p_rot[1])
            accum_z += s.thickness

    # -----------------------------------------------------------------------
    # Aperture & Stop Event Handlers
    # -----------------------------------------------------------------------

    def _on_is_stop_toggled(self, checked: bool):
        if self._syncing_ui or not self.model:
            return
        if hasattr(self.model, "set_stop_surface"):
            self.model.set_stop_surface(self._selected_index, checked)
            self._rebuild_surface_list()
            self._emit_change()

    def _on_semi_diam_changed(self, val: float):
        if self._syncing_ui or not self.model:
            return
        surf = self._get_surface(self._selected_index)
        if surf:
            surf.semi_diameter = float(val)
        self.model.setData(self._selected_index, 5, float(val))
        self._emit_change()

    def _on_sub_xy_changed(self, val: float):
        if self._syncing_ui or not self.model:
            return
        surf = self._get_surface(self._selected_index)
        if surf:
            surf.semi_diameter_x = self.spin_sub_x.value()
            surf.semi_diameter_y = self.spin_sub_y.value()
        self._emit_change()

    def _on_aperture_x_changed(self, val: float):
        if self._syncing_ui or not self.model:
            return
        surf = self._get_surface(self._selected_index)
        if surf:
            surf.aperture_x_half = float(val)
            hx = surf.aperture_x_half
            hy = surf.aperture_y_half
            self.model.setData(self._selected_index, 15, f"X={hx}, Y={hy}")
            if hasattr(self, 'lbl_aperture_readout'):
                self.lbl_aperture_readout.setText(f"Rectangular: {2*hx:.2f} × {2*hy:.2f} mm | Semi-D: {surf.semi_diameter:.2f} mm")
            self._emit_change()

    def _on_aperture_y_changed(self, val: float):
        if self._syncing_ui or not self.model:
            return
        surf = self._get_surface(self._selected_index)
        if surf:
            surf.aperture_y_half = float(val)
            hx = surf.aperture_x_half
            hy = surf.aperture_y_half
            self.model.setData(self._selected_index, 15, f"X={hx}, Y={hy}")
            if hasattr(self, 'lbl_aperture_readout'):
                self.lbl_aperture_readout.setText(f"Rectangular: {2*hx:.2f} × {2*hy:.2f} mm | Semi-D: {surf.semi_diameter:.2f} mm")
            self._emit_change()

    def _on_substrate_shape_combo_changed(self, idx: int):
        if self._syncing_ui or not self.model:
            return
        shape_name = self.combo_substrate_shape.currentText()
        surf = self._get_surface(self._selected_index)
        if surf:
            surf.substrate_shape = shape_name
            if shape_name == "Rectangular":
                if surf.semi_diameter_x <= 0: surf.semi_diameter_x = surf.semi_diameter
                if surf.semi_diameter_y <= 0: surf.semi_diameter_y = surf.semi_diameter
                if hasattr(self, 'widget_rect_substrate'):
                    self.widget_rect_substrate.setVisible(True)
                if hasattr(self, 'spin_sub_x'):
                    self.spin_sub_x.blockSignals(True)
                    self.spin_sub_x.setValue(surf.semi_diameter_x)
                    self.spin_sub_x.blockSignals(False)
                if hasattr(self, 'spin_sub_y'):
                    self.spin_sub_y.blockSignals(True)
                    self.spin_sub_y.setValue(surf.semi_diameter_y)
                    self.spin_sub_y.blockSignals(False)
            else:
                if hasattr(self, 'widget_rect_substrate'):
                    self.widget_rect_substrate.setVisible(False)
            self._emit_change()

    def _on_aperture_type_combo_changed(self, idx: int):
        if self._syncing_ui or not self.model:
            return
        type_name = self.combo_aperture_type.currentText()
        surf = self._get_surface(self._selected_index)
        if surf:
            surf.aperture_type = type_name
            if type_name == "Point":
                surf.semi_diameter = 0.0
                surf.aperture_x_half = 0.0
                surf.aperture_y_half = 0.0
                if hasattr(self, 'widget_rect_aperture'):
                    self.widget_rect_aperture.setVisible(False)
                if hasattr(self, 'spin_semi_diam'):
                    self.spin_semi_diam.blockSignals(True)
                    self.spin_semi_diam.setValue(0.0)
                    self.spin_semi_diam.blockSignals(False)
                if hasattr(self, 'lbl_aperture_readout'):
                    self.lbl_aperture_readout.setText("Point Object / Image: Ø = 0.00 mm (Zero Extent)")
            elif type_name == "Rectangular":
                if surf.aperture_x_half <= 0 or surf.aperture_y_half <= 0:
                    sd_val = float(surf.semi_diameter) if surf.semi_diameter > 0 else 10.0
                    if surf.aperture_x_half <= 0: surf.aperture_x_half = sd_val
                    if surf.aperture_y_half <= 0: surf.aperture_y_half = sd_val
                if hasattr(self, 'widget_rect_aperture'):
                    self.widget_rect_aperture.setVisible(True)
                if hasattr(self, 'spin_aperture_x'):
                    self.spin_aperture_x.blockSignals(True)
                    self.spin_aperture_x.setValue(surf.aperture_x_half)
                    self.spin_aperture_x.blockSignals(False)
                if hasattr(self, 'spin_aperture_y'):
                    self.spin_aperture_y.blockSignals(True)
                    self.spin_aperture_y.setValue(surf.aperture_y_half)
                    self.spin_aperture_y.blockSignals(False)
                if hasattr(self, 'lbl_aperture_readout'):
                    hx, hy = surf.aperture_x_half, surf.aperture_y_half
                    self.lbl_aperture_readout.setText(f"Rectangular: {2*hx:.2f} × {2*hy:.2f} mm | Semi-D: {surf.semi_diameter:.2f} mm")
            else:
                if hasattr(self, 'widget_rect_aperture'):
                    self.widget_rect_aperture.setVisible(False)
                if hasattr(self, 'lbl_aperture_readout'):
                    sd = float(getattr(surf, 'semi_diameter', 10.0))
                    self.lbl_aperture_readout.setText(f"Clear Aperture: Ø = {2*sd:.2f} mm (R = {sd:.2f} mm)")

            self.model.setData(self._selected_index, 14, type_name)
            self._emit_change()

    def _open_aperture_dialog(self):
        if not self.model:
            return
        surf = self._get_surface(self._selected_index)
        if surf is None:
            return
        from optics_app.gui.aperture_dialog import ApertureStopDialog
        dlg = ApertureStopDialog(surface=surf, surface_idx=self._selected_index, model=self.model, parent=self)
        exec_fn = getattr(dlg, 'exec_', dlg.exec)
        if exec_fn():
            self._rebuild_surface_list()
            self._sync_controls_from_surface(self._selected_index)
            self._emit_change()

    def _on_toggle_stop_clicked(self):
        if not self.model:
            return
        surf = self._get_surface(self._selected_index)
        if surf:
            is_stop = (
                (getattr(surf, 'surface_type', None) == SurfaceType.STOP)
                or getattr(surf, 'is_stop', False)
                or getattr(surf, 'is_aperture_stop', False)
                or "STOP" in str(getattr(surf, 'surface_type', '')).upper()
            )
            if hasattr(self.model, "set_stop_surface"):
                self.model.set_stop_surface(self._selected_index, not is_stop)
                self._rebuild_surface_list()
                self._sync_controls_from_surface(self._selected_index)
                self._emit_change()

    def reset_tilts(self):
        """Reset tilt_x, tilt_y, tilt_z to 0.0 on selected component."""
        self.tilt_x_spin.setValue(0.0)
        self.tilt_y_spin.setValue(0.0)
        self.tilt_z_spin.setValue(0.0)

    def reset_decenters(self):
        """Reset decenter_x, decenter_y to 0.0 on selected component."""
        self.dec_x_spin.setValue(0.0)
        self.dec_y_spin.setValue(0.0)

    def apply_quick_fold(self, angle_deg: float, plane: str = "YZ"):
        """
        Quick-fold action for mirrors: sets tilt_x / tilt_y to achieve exact fold angle.
        """
        stack = self._get_stack()
        idx = self._selected_index
        if not stack or not (0 <= idx < len(stack.surfaces)):
            return

        surf = stack.surfaces[idx]
        is_reflective = (
            getattr(surf, "surface_type", None) == SurfaceType.GRATING
            or (hasattr(surf, "is_mirror") and surf.is_mirror())
            or (hasattr(surf, "is_reflective") and surf.is_reflective())
        )
        if not is_reflective:
            message = (
                f"Mirror fold presets cannot be applied to Surface #{idx} ('{surf.name}') "
                "because it is not reflective."
            )
            self.aim_status_label.setText(message)
            from optics_app.gui.qt_compat import QMessageBox
            QMessageBox.warning(self, "Invalid Mirror Action", message)
            return

        tx, ty = MirrorPositioner.set_mirror_fold_angle(
            stack=stack,
            mirror_index=idx,
            fold_angle_deg=angle_deg,
            fold_plane=plane
        )

        self._syncing_ui = True
        try:
            self.tilt_x_spin.setValue(tx)
            self.tilt_x_slider.setValue(int(round(tx * self._TILT_SCALE)))
            self.tilt_y_spin.setValue(ty)
            self.tilt_y_slider.setValue(int(round(ty * self._TILT_SCALE)))
        finally:
            self._syncing_ui = False

        if self.model is not None and hasattr(self.model, "_data") and idx < len(self.model._data):
            self.model._data[idx][9] = tx
            self.model._data[idx][10] = ty

        self.aim_status_label.setText(f"Applied {angle_deg:.1f}° fold ({plane}) on S{idx}")
        self._emit_change()

    def aim_selected_mirror_to_target(self, target_idx: int = -1) -> MirrorAimResult:
        """
        Analytically computes mirror tilt angles to aim at the specified target surface.
        """
        stack = self._get_stack()
        idx = self._selected_index
        if not stack or not (0 <= idx < len(stack.surfaces)):
            return MirrorAimResult(
                success=False, surface_index=idx, target_global_point=(0.0, 0.0, 0.0),
                solved_tilt_x=0.0, solved_tilt_y=0.0, solved_tilt_z=0.0,
                target_error_mm=float('inf'), message="Invalid surface stack"
            )

        if target_idx == -1 or target_idx >= len(stack.surfaces):
            target_idx = len(stack.surfaces) - 1

        result = MirrorPositioner.aim_mirror_to_target(
            stack=stack,
            mirror_index=idx,
            target_surface_index=target_idx
        )

        if result.success or result.target_error_mm < 0.1:
            self._syncing_ui = True
            try:
                self.tilt_x_spin.setValue(result.solved_tilt_x)
                self.tilt_x_slider.setValue(int(round(result.solved_tilt_x * self._TILT_SCALE)))
                self.tilt_y_spin.setValue(result.solved_tilt_y)
                self.tilt_y_slider.setValue(int(round(result.solved_tilt_y * self._TILT_SCALE)))
            finally:
                self._syncing_ui = False

            if self.model is not None and hasattr(self.model, "_data") and idx < len(self.model._data):
                self.model._data[idx][9] = result.solved_tilt_x
                self.model._data[idx][10] = result.solved_tilt_y

            err_um = result.target_error_mm * 1000.0
            self.aim_status_label.setText(f"Aimed S{idx} -> S{target_idx} (Error: {err_um:.2f} µm)")
            self._emit_change()
        else:
            self.aim_status_label.setText(f"Aim failed: {result.message}")

        return result

    def on_aim_mirror_clicked(self):
        """Slot invoked when 'Aim Selected Mirror at Target' button is clicked."""
        target_idx = self.aim_target_combo.currentData()
        if target_idx is None:
            target_idx = -1
        self.aim_selected_mirror_to_target(target_idx)

    def _emit_change(self):
        """Notify listeners of modification to trigger real-time ray trace & canvas refresh."""
        if hasattr(self, "component_changed") and self.component_changed is not None:
            self.component_changed.emit(self._selected_index)

        if self.model is not None and hasattr(self.model, "system_updated") and self.model.system_updated is not None:
            try:
                self.model.system_updated.emit()
            except Exception:
                pass


__all__ = ["ComponentEditorDockWidget"]
