"""
TI Modulator Studio Widget (PLM 4-Bit Phase & DMD 1-Bit Binary Studio).

Provides a comprehensive, interactive GUI panel to:
1. Select and inspect Texas Instruments Phase Light Modulators (PLM, e.g. DLP670S)
   and Digital Micromirror Devices (DMD, e.g. DLP4500, DLP6500, DLP9000, DLP2010).
2. Control 4-bit discrete phase states (16 levels: 0..15) or 1-bit binary states (0/1, ON, OFF, FLAT, HADAMARD_A, HADAMARD_B).
3. Synthesize and fine-tune Zernike wavefront phases (Z1..Z15) on PLM, or safely lock with an explanatory banner on DMD.
4. Program diffractive gratings (Blazed, Binary, 4-Level, Sinusoidal, Chirped VLS for PLM; Binary Ronchi & Hadamard for DMD).
5. Load 2D patterns from disk (.png, .bmp, .npy, .csv, .json) and generate synthetic binary or phase masks.
6. Display 3 core optical diagnostic plots:
   - 2D Phase / Binary Map (Viridis 16-level or high-contrast binary #151922/#FFB300)
   - Simulated Far-Field Diffraction PSF (2D FFT Fraunhofer pattern)
   - 1D Cross-Section (Staircase OPD / Binary Step pulses with micromirror tilt twin axis)
"""

from __future__ import annotations

import os
import math
from typing import Any, Dict, List, Optional
import numpy as np

try:
    from optics_app.gui.qt_compat import (
        QT_AVAILABLE, QtWidgets, QWidget, QDockWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton,
        QTabWidget, QGroupBox, QDoubleSpinBox, QSpinBox, QSlider, QCheckBox,
        QFileDialog, QMessageBox, QScrollArea, QSplitter, QFrame, Qt, pyqtSignal, QApplication,
        QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView
    )
except ImportError:
    from v2.gui.qt_compat import (
        QT_AVAILABLE, QtWidgets, QWidget, QDockWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton,
        QTabWidget, QGroupBox, QDoubleSpinBox, QSpinBox, QSlider, QCheckBox,
        QFileDialog, QMessageBox, QScrollArea, QSplitter, QFrame, Qt, pyqtSignal, QApplication,
        QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView
    )

import matplotlib
try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
except Exception:
    try:
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    except Exception:
        FigureCanvas = None
from matplotlib.figure import Figure
import matplotlib.colors as mcolors

try:
    from optics_app.engine.plm import (
        PLMConfig, ZernikePhaseConfig, PLMGratingConfig, PLMPatternConfig,
        TI_PLM_MODELS, TI_DMD_MODELS, TI_ALL_MODELS, TI_PLM_DLP670S_SPECS, TI_DMD_DLP4500_SPECS,
        ZERNIKE_MODES_INFO, is_plm, is_dmd, ensure_plm_config, generate_plm_2d_maps,
        load_pattern_from_file, save_pattern_to_file, build_4bit_state_table
    )
    from optics_app.engine.mechanisms import available_states, active_state_name, MechanicalStateController
    from optics_app.engine.surface import SurfaceType
except ImportError:
    from v2.engine.plm import (
        PLMConfig, ZernikePhaseConfig, PLMGratingConfig, PLMPatternConfig,
        TI_PLM_MODELS, TI_DMD_MODELS, TI_ALL_MODELS, TI_PLM_DLP670S_SPECS, TI_DMD_DLP4500_SPECS,
        ZERNIKE_MODES_INFO, is_plm, is_dmd, ensure_plm_config, generate_plm_2d_maps,
        load_pattern_from_file, save_pattern_to_file, build_4bit_state_table
    )
    from v2.engine.mechanisms import available_states, active_state_name, MechanicalStateController
    from v2.engine.surface import SurfaceType


class PLMStudioWidget(QWidget):
    """Interactive GUI Studio for TI Phase Light Modulator (4-Bit) and Digital Micromirror Device (1-Bit)."""
    pattern_applied = pyqtSignal(int)  # emits surface_index when pattern is modified

    def __init__(self, surface_stack=None, parent=None):
        super().__init__(parent)
        self.stack = surface_stack
        self.current_surface_idx = -1
        self._block_signals = False
        self.dark_mode: bool = False
        self._active_column_ranges: List[Dict[str, int]] = []
        self._last_surface_count = len(surface_stack.surfaces) if (surface_stack and hasattr(surface_stack, 'surfaces') and surface_stack.surfaces is not None) else 0
        if QT_AVAILABLE:
            try:
                from optics_app.gui.qt_compat import QTimer
            except ImportError:
                from v2.gui.qt_compat import QTimer
            self._plot_debounce_timer = QTimer(self)
            self._plot_debounce_timer.setSingleShot(True)
            self._plot_debounce_timer.setInterval(30)
            self._plot_debounce_timer.timeout.connect(self.update_live_plots)
        else:
            self._plot_debounce_timer = None
        self._init_ui()
        self._update_theme_styles()
        self.refresh_surfaces()

    def set_surface_stack(self, stack, force: bool = False) -> None:
        if not force and self.stack is stack and stack is not None:
            curr_count = len(stack.surfaces) if hasattr(stack, 'surfaces') and stack.surfaces is not None else 0
            if getattr(self, '_last_surface_count', None) == curr_count:
                return
        self.stack = stack
        self._last_surface_count = len(stack.surfaces) if (stack and hasattr(stack, 'surfaces') and stack.surfaces is not None) else 0
        self.refresh_surfaces()

    def showEvent(self, event):
        super().showEvent(event)
        if hasattr(self, 'canvas') and self.canvas is not None:
            self.canvas.draw_idle()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'canvas') and self.canvas is not None:
            self.canvas.draw_idle()

    def _request_live_plot_update(self) -> None:
        if getattr(self, '_plot_debounce_timer', None) is not None:
            self._plot_debounce_timer.start(30)
        else:
            self.update_live_plots()

    def _get_pixel_pitch_um(self) -> float:
        if self.stack and 0 <= self.current_surface_idx < len(self.stack.surfaces):
            surf = self.stack.surfaces[self.current_surface_idx]
            cfg = getattr(surf, "plm_config", None)
            if cfg is not None:
                return max(float(cfg.pixel_pitch_um), 1.0)
        return 10.8

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(6)

        # 1. Top Header: Device & State Selector (Clean 2-Row Architecture)
        self.header_group = QGroupBox("Texas Instruments Modulator Studio (PLM 4-Bit Phase / DMD 1-Bit Binary)")
        header_vbox = QVBoxLayout(self.header_group)
        header_vbox.setContentsMargins(8, 8, 8, 8)
        header_vbox.setSpacing(6)

        # Row 1: Hardware Specifications & Modulator Target
        row1 = QHBoxLayout()
        row1.setContentsMargins(0, 0, 0, 0)
        row1.setSpacing(6)

        row1.addWidget(QLabel("Target Surface:"))
        self.combo_surface = QComboBox()
        self.combo_surface.setMinimumWidth(180)
        self.combo_surface.currentIndexChanged.connect(self._on_surface_selection_changed)
        row1.addWidget(self.combo_surface)

        row1.addWidget(QLabel("Type:"))
        self.combo_modulator_type = QComboBox()
        self.combo_modulator_type.addItems(["TI-PLM (4-Bit Phase)", "TI-DMD (1-Bit Binary)"])
        self.combo_modulator_type.currentIndexChanged.connect(self._on_modulator_type_changed)
        row1.addWidget(self.combo_modulator_type)

        self.lbl_model = QLabel("Model:")
        row1.addWidget(self.lbl_model)
        self.combo_model = QComboBox()
        for k in TI_PLM_MODELS.keys():
            self.combo_model.addItem(k)
        self.combo_model.currentTextChanged.connect(self._on_model_changed)
        row1.addWidget(self.combo_model)

        self.lbl_design_wl = QLabel("λ₀ (µm):")
        row1.addWidget(self.lbl_design_wl)
        self.spin_design_wl = QDoubleSpinBox()
        self.spin_design_wl.setRange(0.2, 5.0)
        self.spin_design_wl.setDecimals(4)
        self.spin_design_wl.setSingleStep(0.01)
        self.spin_design_wl.setValue(1.225)
        self.spin_design_wl.valueChanged.connect(self._on_design_wl_changed)
        row1.addWidget(self.spin_design_wl)
        row1.addStretch()
        header_vbox.addLayout(row1)

        # Row 2: Operational Modulator State, Stepping & Wavefront Units
        row2 = QHBoxLayout()
        row2.setContentsMargins(0, 0, 0, 0)
        row2.setSpacing(6)

        self.lbl_state = QLabel("4-Bit State:")
        row2.addWidget(self.lbl_state)
        self.combo_state = QComboBox()
        self.combo_state.setMinimumWidth(150)
        self.combo_state.currentIndexChanged.connect(self._on_state_changed)
        row2.addWidget(self.combo_state)

        self.btn_prev_state = QPushButton("◀")
        self.btn_prev_state.setMaximumWidth(28)
        self.btn_prev_state.setToolTip("Step to previous modulator state")
        self.btn_prev_state.clicked.connect(self._prev_state)
        row2.addWidget(self.btn_prev_state)

        self.btn_next_state = QPushButton("▶")
        self.btn_next_state.setMaximumWidth(28)
        self.btn_next_state.setToolTip("Step to next modulator state")
        self.btn_next_state.clicked.connect(self._next_state)
        row2.addWidget(self.btn_next_state)

        self.lbl_step_um = QLabel("1-LSB: 0.0766 µm")
        self.lbl_step_um.setStyleSheet("color: #81C784; font-weight: bold; font-size: 11px;")
        row2.addWidget(self.lbl_step_um)

        self.lbl_unit = QLabel("Unit:")
        row2.addWidget(self.lbl_unit)
        self.combo_unit = QComboBox()
        self.combo_unit.addItems(["Phase OPD (µm)", "Physical Piston (µm)", "Phase Angle (°)", "Waves (λ₀)"])
        self.combo_unit.currentTextChanged.connect(self._on_unit_changed)
        row2.addWidget(self.combo_unit)

        self.chk_quantize = QCheckBox("4-Bit Quantize (16 Levels)")
        self.chk_quantize.setChecked(True)
        self.chk_quantize.toggled.connect(self._on_quantize_toggled)
        row2.addWidget(self.chk_quantize)
        row2.addStretch()
        header_vbox.addLayout(row2)

        # Row 3: Physical Panel Mounting & Clocking Rotation (Around Z / Optical Axis)
        row3 = QHBoxLayout()
        row3.setContentsMargins(0, 0, 0, 0)
        row3.setSpacing(6)

        row3.addWidget(QLabel("Panel Clocking (Tilt Z):"))
        self.spin_panel_rotation = QDoubleSpinBox()
        self.spin_panel_rotation.setRange(-180.0, 180.0)
        self.spin_panel_rotation.setSingleStep(5.0)
        self.spin_panel_rotation.setValue(0.0)
        self.spin_panel_rotation.setSuffix("°")
        self.spin_panel_rotation.setToolTip("Rotate the physical DMD / PLM panel around its surface normal / optical axis (tilt_z)")
        self.spin_panel_rotation.valueChanged.connect(self._on_panel_rotation_changed)
        row3.addWidget(self.spin_panel_rotation)

        self.btn_panel_rot_m45 = QPushButton("-45°")
        self.btn_panel_rot_m45.setToolTip("Rotate panel by -45° (common TI DLP diagonal hinge orientation)")
        self.btn_panel_rot_m45.clicked.connect(lambda: self._set_panel_rotation(-45.0))
        row3.addWidget(self.btn_panel_rot_m45)

        self.btn_panel_rot_0 = QPushButton("0°")
        self.btn_panel_rot_0.setToolTip("Reset panel rotation to 0° (standard unclocked orientation)")
        self.btn_panel_rot_0.clicked.connect(lambda: self._set_panel_rotation(0.0))
        row3.addWidget(self.btn_panel_rot_0)

        self.btn_panel_rot_p45 = QPushButton("+45°")
        self.btn_panel_rot_p45.setToolTip("Rotate panel by +45°")
        self.btn_panel_rot_p45.clicked.connect(lambda: self._set_panel_rotation(45.0))
        row3.addWidget(self.btn_panel_rot_p45)

        self.btn_panel_rot_90 = QPushButton("90°")
        self.btn_panel_rot_90.setToolTip("Rotate panel by 90° (horizontal dispersion / vertical hinge)")
        self.btn_panel_rot_90.clicked.connect(lambda: self._set_panel_rotation(90.0))
        row3.addWidget(self.btn_panel_rot_90)

        row3.addStretch()
        header_vbox.addLayout(row3)

        main_layout.addWidget(self.header_group)

        # 2. Main Splitter: Parameter Controls (Left) and Live Visualizations (Right)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left Controls Tabs
        self.tabs = QTabWidget()
        self.tab_zernike = self._build_zernike_tab()
        self.tab_grating = self._build_grating_tab()
        self.tab_pattern = self._build_pattern_tab()
        self.tabs.addTab(self.tab_zernike, "🌊 Zernike Phases")
        self.tabs.addTab(self.tab_grating, "⚡ Phase Gratings")
        self.tabs.addTab(self.tab_pattern, "🖼️ 2D Patterns & Masks")
        splitter.addWidget(self.tabs)

        # Right Plots Canvas
        plot_container = QWidget()
        plot_layout = QVBoxLayout(plot_container)
        plot_layout.setContentsMargins(0, 0, 0, 0)

        # Far-Field PSF Zoom & Dynamic Range Controls
        psf_ctrl_layout = QHBoxLayout()
        psf_ctrl_layout.setContentsMargins(4, 2, 4, 2)
        psf_ctrl_layout.setSpacing(8)

        lbl_zoom = QLabel("Far-Field PSF:")
        lbl_zoom.setStyleSheet("font-size: 11px; font-weight: bold; color: #94a3b8;")
        psf_ctrl_layout.addWidget(lbl_zoom)

        self.btn_psf_zoom_out = QPushButton("🔍 Zoom Out (Full Field)")
        self.btn_psf_zoom_out.setToolTip("Zoom out to view the entire angular Fraunhofer diffraction field")
        self.btn_psf_zoom_out.clicked.connect(self._zoom_out_psf)
        psf_ctrl_layout.addWidget(self.btn_psf_zoom_out)

        self.combo_psf_zoom = QComboBox()
        self.combo_psf_zoom.addItems([
            "Full Field (Zoom Out 1×)",
            "Wide (Zoom 2×)",
            "Medium (Zoom 4×)",
            "Close-up (Zoom 8×)",
        ])
        self.combo_psf_zoom.setCurrentIndex(0)
        self.combo_psf_zoom.setToolTip("Select Zoom Level for Simulated Far-Field PSF. 'Full Field (Zoom Out 1×)' shows all diffraction orders without cropping.")
        self.combo_psf_zoom.currentIndexChanged.connect(lambda idx: self._request_live_plot_update())
        psf_ctrl_layout.addWidget(self.combo_psf_zoom)

        self.chk_psf_log = QCheckBox("Log Scale / HDR (dB)")
        self.chk_psf_log.setChecked(True)
        self.chk_psf_log.setToolTip("Display far-field diffraction intensity on logarithmic / dB scale so diffraction orders and side lobes are clearly visible.")
        self.chk_psf_log.toggled.connect(lambda checked: self._request_live_plot_update())
        psf_ctrl_layout.addWidget(self.chk_psf_log)

        psf_ctrl_layout.addStretch()
        plot_layout.addLayout(psf_ctrl_layout)

        self.fig = Figure(figsize=(7.5, 6.0), dpi=100)
        self.fig.patch.set_facecolor('#1e1e1e')
        self.canvas = FigureCanvas(self.fig)
        self.canvas.mpl_connect('scroll_event', self._on_canvas_scroll)
        plot_layout.addWidget(self.canvas)

        # Plot Subtitle Badge Bar
        badge_layout = QHBoxLayout()
        self.lbl_status = QLabel("Modulator Array: 1358 × 800 (10.8 µm pitch) | 4-Bit Discrete Piston (16 Levels)")
        self.lbl_status.setStyleSheet("font-weight: bold; color: #4FC3F7;")
        badge_layout.addWidget(self.lbl_status)
        badge_layout.addStretch()

        self.btn_apply = QPushButton("🚀 Retrace Optical System")
        self.btn_apply.setStyleSheet("background-color: #0288D1; color: white; font-weight: bold; padding: 4px 12px;")
        self.btn_apply.clicked.connect(self._apply_and_retrace)
        badge_layout.addWidget(self.btn_apply)

        plot_layout.addLayout(badge_layout)
        splitter.addWidget(plot_container)

        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 6)
        main_layout.addWidget(splitter, 1)

    # ------------------------------------------------------------------------
    # Tab 1: Zernike Phase Synthesis & Aberration Correction
    # ------------------------------------------------------------------------
    def _build_zernike_tab(self) -> QWidget:
        container = QWidget()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setSpacing(6)

        # DMD Banner (visible when in DMD mode)
        self.lbl_zernike_dmd_banner = QLabel(
            "⚠️ <b>Zernike Wavefront Phase Synthesis is disabled in DMD mode.</b><br>"
            "Digital Micromirror Devices operate strictly in bistable mechanical tilt (±12°), "
            "not continuous optical piston phase modulation.<br>"
            "Use the <b>Phase Gratings</b> tab (Binary Ronchi / Hadamard) or <b>2D Patterns & Masks</b> tab "
            "for binary pattern control."
        )
        self.lbl_zernike_dmd_banner.setWordWrap(True)
        self.lbl_zernike_dmd_banner.setStyleSheet(
            "background-color: #3b2800; color: #fde047; border: 1px solid #ca8a04; border-radius: 4px; padding: 10px; font-size: 11px;"
        )
        self.lbl_zernike_dmd_banner.setVisible(False)
        layout.addWidget(self.lbl_zernike_dmd_banner)

        # Interactive controls wrapper
        self.zernike_controls_widget = QWidget()
        z_ctrl_layout = QVBoxLayout(self.zernike_controls_widget)
        z_ctrl_layout.setContentsMargins(0, 0, 0, 0)
        z_ctrl_layout.setSpacing(6)

        # Header controls
        top_box = QHBoxLayout()
        self.chk_zernike_enable = QCheckBox("Enable Zernike Wavefront Phase")
        self.chk_zernike_enable.setChecked(True)
        self.chk_zernike_enable.toggled.connect(self._on_zernike_param_changed)
        top_box.addWidget(self.chk_zernike_enable)

        top_box.addWidget(QLabel("Pupil Radius R_norm (mm):"))
        self.spin_zernike_rnorm = QDoubleSpinBox()
        self.spin_zernike_rnorm.setRange(0.1, 50.0)
        self.spin_zernike_rnorm.setValue(4.32)
        self.spin_zernike_rnorm.valueChanged.connect(self._on_zernike_param_changed)
        top_box.addWidget(self.spin_zernike_rnorm)

        btn_autofit = QPushButton("Auto-Fit Aperture")
        btn_autofit.clicked.connect(self._autofit_zernike_radius)
        top_box.addWidget(btn_autofit)

        btn_zero_all = QPushButton("Reset to 0")
        btn_zero_all.clicked.connect(self._zero_zernike)
        top_box.addWidget(btn_zero_all)

        z_ctrl_layout.addLayout(top_box)

        # Presets Box
        preset_box = QHBoxLayout()
        preset_box.addWidget(QLabel("Presets:"))
        btn_pre_defocus = QPushButton("Defocus +1λ")
        btn_pre_defocus.clicked.connect(lambda: self._set_zernike_preset({"z4_defocus": 1.0}))
        preset_box.addWidget(btn_pre_defocus)

        btn_pre_astig = QPushButton("Astigmatism +0.5λ")
        btn_pre_astig.clicked.connect(lambda: self._set_zernike_preset({"z6_astig_vertical": 0.5}))
        preset_box.addWidget(btn_pre_astig)

        btn_pre_coma = QPushButton("Coma +0.3λ")
        btn_pre_coma.clicked.connect(lambda: self._set_zernike_preset({"z7_coma_vertical": 0.3}))
        preset_box.addWidget(btn_pre_coma)

        btn_pre_spectrometer = QPushButton("Correct Spectrometer Aberration")
        btn_pre_spectrometer.setStyleSheet("font-weight: bold; color: #81C784;")
        btn_pre_spectrometer.clicked.connect(self._preset_spectrometer_correction)
        preset_box.addWidget(btn_pre_spectrometer)
        preset_box.addStretch()
        z_ctrl_layout.addLayout(preset_box)

        # Zernike Modes Grid (Z1..Z15)
        self.zernike_spins: Dict[str, QDoubleSpinBox] = {}
        self.zernike_um_labels: Dict[str, QLabel] = {}
        self.modes_group = QGroupBox("Zernike Modes (Noll 1..15, Amplitude in Waves & µm at λ₀)")
        modes_layout = QVBoxLayout(self.modes_group)

        for info in ZERNIKE_MODES_INFO:
            key = info["key"]
            row = QHBoxLayout()
            lbl = QLabel(f"<b>{info['name']}</b> ({info['formula']}):")
            lbl.setMinimumWidth(240)
            row.addWidget(lbl)

            spin = QDoubleSpinBox()
            spin.setRange(-20.0, 20.0)
            spin.setDecimals(3)
            spin.setSingleStep(0.05)
            spin.setValue(0.0)
            spin.valueChanged.connect(self._on_zernike_param_changed)
            self.zernike_spins[key] = spin
            row.addWidget(spin)

            lbl_um = QLabel("(+0.000 µm)")
            lbl_um.setStyleSheet("color: #4FC3F7; font-weight: bold; font-size: 11px; min-width: 80px;")
            self.zernike_um_labels[key] = lbl_um
            row.addWidget(lbl_um)

            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(-500, 500)
            slider.setValue(0)
            slider.valueChanged.connect(lambda val, s=spin: s.setValue(val / 100.0) if abs(s.value() - val / 100.0) > 0.005 else None)
            spin.valueChanged.connect(lambda val, sl=slider: sl.setValue(int(round(val * 100.0))))
            row.addWidget(slider)

            modes_layout.addLayout(row)

        z_ctrl_layout.addWidget(self.modes_group)
        layout.addWidget(self.zernike_controls_widget)
        layout.addStretch()
        scroll.setWidget(content)

        c_layout = QVBoxLayout(container)
        c_layout.setContentsMargins(0, 0, 0, 0)
        c_layout.addWidget(scroll)
        return container

    # ------------------------------------------------------------------------
    # Tab 2: Programmable Phase Gratings & Binary Ronchi Rulings
    # ------------------------------------------------------------------------
    def _build_grating_tab(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(8)

        self.chk_grating_enable = QCheckBox("Enable Programmable Phase Grating on PLM")
        self.chk_grating_enable.setChecked(True)
        self.chk_grating_enable.toggled.connect(self._on_grating_param_changed)
        layout.addWidget(self.chk_grating_enable)

        form_group = QGroupBox("Grating Profile & Periodicity")
        form_layout = QVBoxLayout(form_group)

        # Profile Type
        row_type = QHBoxLayout()
        row_type.addWidget(QLabel("Grating Profile Type:"))
        self.combo_grating_type = QComboBox()
        self.combo_grating_type.addItems([
            "Blazed", "Binary", "4-Level", "Sinusoidal", "Chirped_VLS", "Hadamard_Spectral"
        ])
        self.combo_grating_type.currentTextChanged.connect(self._on_grating_param_changed)
        row_type.addWidget(self.combo_grating_type)
        form_layout.addLayout(row_type)

        # Grating Pitch, Line Density & Pixels per period (all synced!)
        row_pitch = QHBoxLayout()
        row_pitch.addWidget(QLabel("Pitch Λ (µm):"))
        self.spin_pitch = QDoubleSpinBox()
        self.spin_pitch.setRange(5.0, 1000.0)
        self.spin_pitch.setDecimals(2)
        self.spin_pitch.setValue(86.4)
        self.spin_pitch.valueChanged.connect(self._on_pitch_changed)
        row_pitch.addWidget(self.spin_pitch)

        row_pitch.addWidget(QLabel("Lines/mm:"))
        self.spin_lpmm = QDoubleSpinBox()
        self.spin_lpmm.setRange(1.0, 200.0)
        self.spin_lpmm.setDecimals(4)
        self.spin_lpmm.setValue(11.5741)
        self.spin_lpmm.valueChanged.connect(self._on_lpmm_changed)
        row_pitch.addWidget(self.spin_lpmm)

        row_pitch.addWidget(QLabel("Pixels/Period:"))
        self.spin_ppp = QDoubleSpinBox()
        self.spin_ppp.setRange(2.0, 200.0)
        self.spin_ppp.setDecimals(1)
        self.spin_ppp.setValue(8.0)
        self.spin_ppp.valueChanged.connect(self._on_ppp_changed)
        row_pitch.addWidget(self.spin_ppp)
        form_layout.addLayout(row_pitch)

        # Orientation & Diffraction Order
        row_geom = QHBoxLayout()
        row_geom.addWidget(QLabel("Orientation Angle ψ (°):"))
        self.spin_orientation = QDoubleSpinBox()
        self.spin_orientation.setRange(-180.0, 180.0)
        self.spin_orientation.setValue(0.0)
        self.spin_orientation.valueChanged.connect(self._on_grating_param_changed)
        row_geom.addWidget(self.spin_orientation)

        btn_grat_m45 = QPushButton("-45°")
        btn_grat_m45.setToolTip("Orient grating at -45°")
        btn_grat_m45.clicked.connect(lambda: self.spin_orientation.setValue(-45.0))
        row_geom.addWidget(btn_grat_m45)

        btn_grat_0 = QPushButton("0°")
        btn_grat_0.setToolTip("Orient grating at 0°")
        btn_grat_0.clicked.connect(lambda: self.spin_orientation.setValue(0.0))
        row_geom.addWidget(btn_grat_0)

        btn_grat_p45 = QPushButton("+45°")
        btn_grat_p45.setToolTip("Orient grating at +45°")
        btn_grat_p45.clicked.connect(lambda: self.spin_orientation.setValue(45.0))
        row_geom.addWidget(btn_grat_p45)

        btn_grat_90 = QPushButton("90°")
        btn_grat_90.setToolTip("Orient grating at 90°")
        btn_grat_90.clicked.connect(lambda: self.spin_orientation.setValue(90.0))
        row_geom.addWidget(btn_grat_90)

        row_geom.addWidget(QLabel("Tracked Order m:"))
        self.combo_order = QComboBox()
        for o in [-2, -1, 0, 1, 2]:
            self.combo_order.addItem(str(o), o)
        self.combo_order.setCurrentIndex(3)  # +1
        self.combo_order.currentIndexChanged.connect(self._on_grating_param_changed)
        row_geom.addWidget(self.combo_order)
        form_layout.addLayout(row_geom)

        # Phase Depth & Chirp
        self.row_depth_widget = QWidget()
        row_depth = QHBoxLayout(self.row_depth_widget)
        row_depth.setContentsMargins(0, 0, 0, 0)
        self.lbl_phase_depth = QLabel("Phase Modulation Depth (×2π):")
        row_depth.addWidget(self.lbl_phase_depth)
        self.spin_depth = QDoubleSpinBox()
        self.spin_depth.setRange(0.0, 2.0)
        self.spin_depth.setDecimals(3)
        self.spin_depth.setValue(1.0)
        self.spin_depth.valueChanged.connect(self._on_grating_param_changed)
        row_depth.addWidget(self.spin_depth)

        self.lbl_depth_um = QLabel("(1.225 µm)")
        self.lbl_depth_um.setStyleSheet("color: #4FC3F7; font-weight: bold; font-size: 11px;")
        row_depth.addWidget(self.lbl_depth_um)

        self.lbl_chirp = QLabel("VLS Chirp (l/mm²):")
        row_depth.addWidget(self.lbl_chirp)
        self.spin_chirp = QDoubleSpinBox()
        self.spin_chirp.setRange(-1.0, 1.0)
        self.spin_chirp.setDecimals(5)
        self.spin_chirp.setValue(0.0)
        self.spin_chirp.valueChanged.connect(self._on_grating_param_changed)
        row_depth.addWidget(self.spin_chirp)
        form_layout.addWidget(self.row_depth_widget)

        # Fold optical axis
        self.chk_fold_axis = QCheckBox("Fold Sequential Optical Axis with Grating (+1st Order Arm)")
        self.chk_fold_axis.setChecked(True)
        self.chk_fold_axis.toggled.connect(self._on_grating_param_changed)
        form_layout.addWidget(self.chk_fold_axis)

        layout.addWidget(form_group)

        # Grating Presets Group
        self.presets_group = QGroupBox("Quick Grating Presets")
        self.presets_layout = QHBoxLayout(self.presets_group)

        self.btn_preset_1 = QPushButton("8-Pixel Blazed (86.4 µm, 11.57 l/mm)")
        self.btn_preset_1.clicked.connect(self._on_grating_preset_1_clicked)
        self.presets_layout.addWidget(self.btn_preset_1)

        self.btn_preset_2 = QPushButton("4-Pixel Blazed (43.2 µm, 23.15 l/mm)")
        self.btn_preset_2.clicked.connect(self._on_grating_preset_2_clicked)
        self.presets_layout.addWidget(self.btn_preset_2)

        self.btn_preset_3 = QPushButton("Binary 0/π (86.4 µm)")
        self.btn_preset_3.clicked.connect(self._on_grating_preset_3_clicked)
        self.presets_layout.addWidget(self.btn_preset_3)

        layout.addWidget(self.presets_group)
        layout.addStretch()
        return container

    def _on_grating_preset_1_clicked(self) -> None:
        is_dmd_mode = self.combo_modulator_type.currentIndex() == 1
        pitch = self._get_pixel_pitch_um() * 8.0
        self._set_grating_preset(pitch, 0.0, "Binary" if is_dmd_mode else "Blazed")

    def _on_grating_preset_2_clicked(self) -> None:
        is_dmd_mode = self.combo_modulator_type.currentIndex() == 1
        pitch = self._get_pixel_pitch_um() * 4.0
        self._set_grating_preset(pitch, 0.0, "Binary" if is_dmd_mode else "Blazed")

    def _on_grating_preset_3_clicked(self) -> None:
        is_dmd_mode = self.combo_modulator_type.currentIndex() == 1
        pitch = self._get_pixel_pitch_um() * 8.0
        if is_dmd_mode:
            self._set_grating_preset(pitch, 0.0, "Hadamard_Spectral")
        else:
            self._set_grating_preset(pitch, 0.0, "Binary")

    # ------------------------------------------------------------------------
    # Tab 3: 2D Patterns, File Loader & Synthetic Holograms
    # ------------------------------------------------------------------------
    def _build_pattern_tab(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(8)

        self.chk_pattern_enable = QCheckBox("Enable 2D Custom / Synthetic Pattern")
        self.chk_pattern_enable.setChecked(False)
        self.chk_pattern_enable.toggled.connect(self._on_pattern_param_changed)
        layout.addWidget(self.chk_pattern_enable)

        box_source = QGroupBox("Pattern Generation & File Loading")
        source_layout = QVBoxLayout(box_source)

        row_ptype = QHBoxLayout()
        row_ptype.addWidget(QLabel("Pattern Source:"))
        self.combo_pattern_type = QComboBox()
        self.combo_pattern_type.addItems([
            "None", "File", "Hadamard", "Column_Range", "Fresnel_Lens", "Vortex", "Cubic_Airy", "Binary_Checkerboard", "Binary_Slits"
        ])
        self.combo_pattern_type.currentTextChanged.connect(self._on_pattern_type_changed)
        row_ptype.addWidget(self.combo_pattern_type)
        source_layout.addLayout(row_ptype)

        # File load controls
        row_file = QHBoxLayout()
        self.btn_load_file = QPushButton("📂 Load Pattern File (.png, .bmp, .npy, .csv, .json)...")
        self.btn_load_file.clicked.connect(self._load_file_dialog)
        row_file.addWidget(self.btn_load_file)

        self.lbl_file_path = QLabel("No file loaded")
        self.lbl_file_path.setStyleSheet("color: #9e9e9e;")
        row_file.addWidget(self.lbl_file_path, 1)
        source_layout.addLayout(row_file)

        # Specific Synthetic Parameters
        # A. Hadamard
        self.grp_hadamard = QGroupBox("Walsh-Hadamard Spectral Multiplex Parameters")
        h_layout = QHBoxLayout(self.grp_hadamard)
        h_layout.addWidget(QLabel("Bands (Order):"))
        self.combo_hadamard_order = QComboBox()
        for ord_val in [16, 32, 64, 128]:
            self.combo_hadamard_order.addItem(str(ord_val), ord_val)
        self.combo_hadamard_order.setCurrentIndex(1)  # 32
        self.combo_hadamard_order.currentIndexChanged.connect(self._on_pattern_param_changed)
        h_layout.addWidget(self.combo_hadamard_order)

        h_layout.addWidget(QLabel("Row Index:"))
        self.spin_hadamard_row = QSpinBox()
        self.spin_hadamard_row.setRange(0, 127)
        self.spin_hadamard_row.setValue(5)
        self.spin_hadamard_row.valueChanged.connect(self._on_pattern_param_changed)
        h_layout.addWidget(self.spin_hadamard_row)

        h_layout.addWidget(QLabel("Axis:"))
        self.combo_hadamard_axis = QComboBox()
        self.combo_hadamard_axis.addItems(["X (Spectral Columns)", "Y"])
        self.combo_hadamard_axis.currentTextChanged.connect(self._on_pattern_param_changed)
        h_layout.addWidget(self.combo_hadamard_axis)
        source_layout.addWidget(self.grp_hadamard)

        # B. Column Range & Multi-Range Discrete State Mapping
        self.grp_column_range = QGroupBox("Column Ranges & Discrete State Mapping (Active Programmable Slits / Zones)")
        col_layout = QVBoxLayout(self.grp_column_range)
        col_layout.setSpacing(6)

        # Row 1: Start, End, State, State Description, Background State
        col_row1 = QHBoxLayout()
        col_row1.addWidget(QLabel("Start Col:"))
        self.spin_col_start = QSpinBox()
        self.spin_col_start.setRange(0, 4096)
        self.spin_col_start.setValue(0)
        self.spin_col_start.setToolTip("Starting column index (0-based) for active column range")
        self.spin_col_start.valueChanged.connect(self._on_column_range_changed)
        col_row1.addWidget(self.spin_col_start)

        col_row1.addWidget(QLabel("End Col:"))
        self.spin_col_end = QSpinBox()
        self.spin_col_end.setRange(0, 4096)
        self.spin_col_end.setValue(50)
        self.spin_col_end.setToolTip("Ending column index (0-based, inclusive) for active column range")
        self.spin_col_end.valueChanged.connect(self._on_column_range_changed)
        col_row1.addWidget(self.spin_col_end)

        col_row1.addWidget(QLabel("Applied State:"))
        self.spin_col_state = QSpinBox()
        self.spin_col_state.setRange(0, 15)
        self.spin_col_state.setValue(1)
        self.spin_col_state.setToolTip("Discrete state applied to this column range (0..1 for DMD, 0..15 for PLM)")
        self.spin_col_state.valueChanged.connect(self._on_column_range_changed)
        col_row1.addWidget(self.spin_col_state)

        self.lbl_col_state_desc = QLabel("(State 1 / 16)")
        self.lbl_col_state_desc.setStyleSheet("color: #38bdf8; font-weight: bold; font-size: 11px;")
        col_row1.addWidget(self.lbl_col_state_desc)

        col_row1.addWidget(QLabel("Background State:"))
        self.spin_col_bg_state = QSpinBox()
        self.spin_col_bg_state.setRange(0, 15)
        self.spin_col_bg_state.setValue(0)
        self.spin_col_bg_state.setToolTip("Discrete state applied to unselected background columns (default 0)")
        self.spin_col_bg_state.valueChanged.connect(self._on_column_range_changed)
        col_row1.addWidget(self.spin_col_bg_state)
        col_layout.addLayout(col_row1)

        # Row 2: Multi-Range Action Buttons
        range_btn_row = QHBoxLayout()
        self.btn_add_range = QPushButton("➕ Add / Update Range")
        self.btn_add_range.setToolTip("Add current (Start, End, State) to configured ranges list or update selected range")
        self.btn_add_range.clicked.connect(self._on_add_range_clicked)
        range_btn_row.addWidget(self.btn_add_range)

        self.btn_remove_range = QPushButton("❌ Remove Selected")
        self.btn_remove_range.setToolTip("Remove selected range from multi-range list")
        self.btn_remove_range.clicked.connect(self._on_remove_range_clicked)
        range_btn_row.addWidget(self.btn_remove_range)

        self.btn_clear_ranges = QPushButton("🗑️ Clear Ranges")
        self.btn_clear_ranges.setToolTip("Clear all ranges and revert to default single range")
        self.btn_clear_ranges.clicked.connect(self._on_clear_ranges_clicked)
        range_btn_row.addWidget(self.btn_clear_ranges)

        self.chk_col_invert = QCheckBox("Invert Range (Selected OFF, rest ON)")
        self.chk_col_invert.setChecked(False)
        self.chk_col_invert.setToolTip("If checked, selected column range is set to background state while other columns reflect active state")
        self.chk_col_invert.toggled.connect(self._on_column_range_changed)
        range_btn_row.addWidget(self.chk_col_invert)

        self.chk_col_carrier_grating = QCheckBox("Superimpose Carrier Grating")
        self.chk_col_carrier_grating.setChecked(False)
        self.chk_col_carrier_grating.setToolTip("If checked, superimposes carrier diffraction grating on active columns for diffractive steering. Leave unchecked for pure slit / column state control.")
        self.chk_col_carrier_grating.toggled.connect(self._on_col_carrier_grating_toggled)
        range_btn_row.addWidget(self.chk_col_carrier_grating)

        range_btn_row.addStretch()
        col_layout.addLayout(range_btn_row)

        # Row 3: Configured Ranges Table (Compact, scrollable)
        self.table_col_ranges = QTableWidget(0, 5)
        self.table_col_ranges.setHorizontalHeaderLabels(["#", "Columns", "Width (mm)", "State", "Description"])
        if hasattr(self.table_col_ranges, "horizontalHeader") and hasattr(self.table_col_ranges.horizontalHeader(), "setStretchLastSection"):
            self.table_col_ranges.horizontalHeader().setStretchLastSection(True)
        if hasattr(self.table_col_ranges, "setSelectionBehavior"):
            sel_beh = getattr(QAbstractItemView.SelectionBehavior, "SelectRows", 1) if hasattr(QAbstractItemView, "SelectionBehavior") else 1
            self.table_col_ranges.setSelectionBehavior(sel_beh)
        if hasattr(self.table_col_ranges, "setSelectionMode"):
            sel_mod = getattr(QAbstractItemView.SelectionMode, "SingleSelection", 1) if hasattr(QAbstractItemView, "SelectionMode") else 1
            self.table_col_ranges.setSelectionMode(sel_mod)
        self.table_col_ranges.setMaximumHeight(95)
        self.table_col_ranges.cellClicked.connect(self._on_table_range_clicked)
        col_layout.addWidget(self.table_col_ranges)

        # Row 4: Presets
        col_row4 = QHBoxLayout()
        btn_center10 = QPushButton("Center 10 Cols")
        btn_center10.setToolTip("Turn ON 10 columns centered on the array (rest OFF)")
        btn_center10.clicked.connect(lambda: self._set_column_preset(10))
        btn_center50 = QPushButton("Center 50 Cols")
        btn_center50.setToolTip("Turn ON 50 columns centered on the array (rest OFF)")
        btn_center50.clicked.connect(lambda: self._set_column_preset(50))
        btn_all_on = QPushButton("All ON (Full)")
        btn_all_on.setToolTip("Turn all array columns ON (full aperture)")
        btn_all_on.clicked.connect(lambda: self._set_column_preset(-1))
        self.btn_staircase = QPushButton("Staircase (0..15)")
        self.btn_staircase.setToolTip("Divide array columns into equal zones with ascending discrete states 0, 1, 2... up to max bit depth")
        self.btn_staircase.clicked.connect(self._set_staircase_preset)
        self.btn_ronchi = QPushButton("Ronchi (0/1)")
        self.btn_ronchi.setToolTip("Divide array into alternating state 0 and 1 bands")
        self.btn_ronchi.clicked.connect(self._set_ronchi_preset)

        col_row4.addWidget(btn_center10)
        col_row4.addWidget(btn_center50)
        col_row4.addWidget(btn_all_on)
        col_row4.addWidget(self.btn_staircase)
        col_row4.addWidget(self.btn_ronchi)
        col_layout.addLayout(col_row4)

        # Row 5: Slit / Pattern Angle
        col_row5 = QHBoxLayout()
        col_row5.addWidget(QLabel("Slit / Pattern Angle:"))
        self.spin_pattern_angle = QDoubleSpinBox()
        self.spin_pattern_angle.setRange(-180.0, 180.0)
        self.spin_pattern_angle.setSingleStep(5.0)
        self.spin_pattern_angle.setValue(0.0)
        self.spin_pattern_angle.setSuffix("°")
        self.spin_pattern_angle.setToolTip("Rotate the active slit / column pattern on the array (0°=vertical columns, 90°=horizontal rows, ±45°=diagonal slits)")
        self.spin_pattern_angle.valueChanged.connect(self._on_pattern_param_changed)
        col_row5.addWidget(self.spin_pattern_angle)

        btn_pat_m45 = QPushButton("-45°")
        btn_pat_m45.setToolTip("Orient slits at -45° diagonally across array")
        btn_pat_m45.clicked.connect(lambda: self._set_pattern_angle(-45.0))
        col_row5.addWidget(btn_pat_m45)

        btn_pat_0 = QPushButton("0° (Cols)")
        btn_pat_0.setToolTip("Orient slits as vertical columns (0°)")
        btn_pat_0.clicked.connect(lambda: self._set_pattern_angle(0.0))
        col_row5.addWidget(btn_pat_0)

        btn_pat_p45 = QPushButton("+45°")
        btn_pat_p45.setToolTip("Orient slits at +45° diagonally across array")
        btn_pat_p45.clicked.connect(lambda: self._set_pattern_angle(45.0))
        col_row5.addWidget(btn_pat_p45)

        btn_pat_90 = QPushButton("90° (Rows)")
        btn_pat_90.setToolTip("Orient slits as horizontal rows (90°)")
        btn_pat_90.clicked.connect(lambda: self._set_pattern_angle(90.0))
        col_row5.addWidget(btn_pat_90)

        col_row5.addStretch()
        col_layout.addLayout(col_row5)

        # Row 6: Active Slit Info Readout
        col_row6 = QHBoxLayout()
        self.lbl_col_width_info = QLabel("Active Slit: 51 cols (0.388 mm)")
        self.lbl_col_width_info.setStyleSheet("color: #38bdf8; font-weight: bold; font-size: 11px;")
        col_row6.addWidget(self.lbl_col_width_info)
        col_row6.addStretch()
        col_layout.addLayout(col_row6)

        source_layout.addWidget(self.grp_column_range)

        # B. Fresnel Lens
        self.grp_fresnel = QGroupBox("Synthetic Fresnel Zone Lens")
        f_layout = QHBoxLayout(self.grp_fresnel)
        f_layout.addWidget(QLabel("Focal Length f (mm):"))
        self.spin_fresnel_f = QDoubleSpinBox()
        self.spin_fresnel_f.setRange(-2000.0, 2000.0)
        self.spin_fresnel_f.setValue(100.0)
        self.spin_fresnel_f.valueChanged.connect(self._on_pattern_param_changed)
        f_layout.addWidget(self.spin_fresnel_f)
        source_layout.addWidget(self.grp_fresnel)

        # C. Vortex
        self.grp_vortex = QGroupBox("Optical Vortex Phase Plate")
        v_layout = QHBoxLayout(self.grp_vortex)
        v_layout.addWidget(QLabel("Topological Charge ℓ:"))
        self.spin_vortex = QSpinBox()
        self.spin_vortex.setRange(-10, 10)
        self.spin_vortex.setValue(1)
        self.spin_vortex.valueChanged.connect(self._on_pattern_param_changed)
        v_layout.addWidget(self.spin_vortex)
        source_layout.addWidget(self.grp_vortex)

        # Export Button
        row_exp = QHBoxLayout()
        self.btn_export_pattern = QPushButton("💾 Export Pattern (.npy, .csv, .png, .json)...")
        self.btn_export_pattern.clicked.connect(self._export_pattern_dialog)
        row_exp.addWidget(self.btn_export_pattern)
        row_exp.addStretch()
        source_layout.addLayout(row_exp)

        layout.addWidget(box_source)
        layout.addStretch()
        return container

    # ------------------------------------------------------------------------
    # Mode Synchronization (PLM vs DMD)
    # ------------------------------------------------------------------------
    def _update_ui_for_mode(self, is_dmd_mode: bool) -> None:
        """Dynamically update labels, available models, tabs, and restrictions for PLM vs DMD."""
        self._block_signals = True

        if is_dmd_mode:
            self.header_group.setTitle("Texas Instruments Digital Micromirror Device (TI-DMD 1-Bit Binary Tilt Array)")
            self.lbl_state.setText("DMD State:")
            self.lbl_unit.setVisible(False)
            self.combo_unit.setVisible(False)
            self.chk_quantize.setVisible(False)
            self.lbl_step_um.setStyleSheet("color: #FFB74D; font-weight: bold; font-size: 11px;")

            # Populate DMD models
            cur_model = self.combo_model.currentText()
            self.combo_model.clear()
            for k in TI_DMD_MODELS.keys():
                self.combo_model.addItem(k)
            if cur_model in TI_DMD_MODELS:
                self.combo_model.setCurrentText(cur_model)
            else:
                self.combo_model.setCurrentText("DLP4500")

            # Update unit combo
            self.combo_unit.clear()
            self.combo_unit.addItems(["Binary State (0 / 1)", "Micromirror Tilt (±12°)"])

            # Grating tab
            self.chk_grating_enable.setText("Enable Programmable Binary Grating on DMD")
            cur_gt = self.combo_grating_type.currentText()
            self.combo_grating_type.clear()
            self.combo_grating_type.addItems(["Binary", "Hadamard_Spectral"])
            if cur_gt in ("Binary", "Hadamard_Spectral"):
                self.combo_grating_type.setCurrentText(cur_gt)
            else:
                self.combo_grating_type.setCurrentText("Binary")

            pitch = self._get_pixel_pitch_um()
            self.btn_preset_1.setText(f"Binary Ronchi ({pitch*8:.1f} µm, 50% Duty)")
            self.btn_preset_2.setText(f"Binary Slits ({pitch*4:.1f} µm, 50% Duty)")
            self.btn_preset_3.setText("Hadamard Spectral (32 Bands)")
            self.row_depth_widget.setVisible(False)

            # Patterns tab
            self.chk_pattern_enable.setText("Enable 2D Binary Pattern / Mask on DMD")
            cur_pt = self.combo_pattern_type.currentText()
            self.combo_pattern_type.clear()
            self.combo_pattern_type.addItems(["None", "File", "Hadamard", "Column_Range", "Binary_Checkerboard", "Binary_Slits"])
            if cur_pt in ("None", "File", "Hadamard", "Column_Range", "Binary_Checkerboard", "Binary_Slits"):
                self.combo_pattern_type.setCurrentText(cur_pt)
            else:
                self.combo_pattern_type.setCurrentText("Hadamard")

            self.grp_fresnel.setVisible(False)
            self.grp_vortex.setVisible(False)
            self.btn_export_pattern.setText("💾 Export 1-Bit Binary DMD Pattern (.npy, .csv, .png, .json)...")

            if hasattr(self, 'spin_col_state'):
                self.spin_col_state.setMaximum(1)
            if hasattr(self, 'spin_col_bg_state'):
                self.spin_col_bg_state.setMaximum(1)
            if hasattr(self, 'btn_staircase'):
                self.btn_staircase.setText("Binary Split (0/1)")

            # Zernike tab: lock and show banner
            self.lbl_zernike_dmd_banner.setVisible(True)
            self.zernike_controls_widget.setEnabled(False)
            self.chk_zernike_enable.setChecked(False)

        else:
            self.header_group.setTitle("Texas Instruments Phase Light Modulator (4-Bit MEMS Piston Array)")
            self.lbl_state.setText("4-Bit State:")
            self.lbl_unit.setVisible(True)
            self.combo_unit.setVisible(True)
            self.chk_quantize.setVisible(True)
            self.lbl_unit.setText("Unit:")
            self.chk_quantize.setText("4-Bit Quantize (16 Levels)")
            self.lbl_step_um.setStyleSheet("color: #81C784; font-weight: bold; font-size: 11px;")

            # Populate PLM models
            cur_model = self.combo_model.currentText()
            self.combo_model.clear()
            for k in TI_PLM_MODELS.keys():
                self.combo_model.addItem(k)
            if cur_model in TI_PLM_MODELS:
                self.combo_model.setCurrentText(cur_model)
            else:
                self.combo_model.setCurrentText("DLP670S")

            # Update unit combo
            self.combo_unit.clear()
            self.combo_unit.addItems(["Phase OPD (µm)", "Physical Piston (µm)", "Phase Angle (°)", "Waves (λ₀)"])

            # Grating tab
            self.chk_grating_enable.setText("Enable Programmable Phase Grating on PLM")
            cur_gt = self.combo_grating_type.currentText()
            self.combo_grating_type.clear()
            self.combo_grating_type.addItems([
                "Blazed", "Binary", "4-Level", "Sinusoidal", "Chirped_VLS", "Hadamard_Spectral"
            ])
            if cur_gt:
                self.combo_grating_type.setCurrentText(cur_gt)
            else:
                self.combo_grating_type.setCurrentText("Blazed")

            self.btn_preset_1.setText("8-Pixel Blazed (86.4 µm, 11.57 l/mm)")
            self.btn_preset_2.setText("4-Pixel Blazed (43.2 µm, 23.15 l/mm)")
            self.btn_preset_3.setText("Binary 0/π (86.4 µm)")
            self.row_depth_widget.setVisible(True)

            # Patterns tab
            self.chk_pattern_enable.setText("Enable 2D Custom / Synthetic Phase Pattern")
            cur_pt = self.combo_pattern_type.currentText()
            self.combo_pattern_type.clear()
            self.combo_pattern_type.addItems([
                "None", "File", "Hadamard", "Column_Range", "Fresnel_Lens", "Vortex", "Cubic_Airy", "Binary_Checkerboard", "Binary_Slits"
            ])
            if cur_pt:
                self.combo_pattern_type.setCurrentText(cur_pt)
            else:
                self.combo_pattern_type.setCurrentText("None")

            self.grp_fresnel.setVisible(True)
            self.grp_vortex.setVisible(True)
            self.btn_export_pattern.setText("💾 Export 4-Bit Pattern (.npy, .csv, .png, .json)...")

            if hasattr(self, 'spin_col_state'):
                self.spin_col_state.setMaximum(15)
            if hasattr(self, 'spin_col_bg_state'):
                self.spin_col_bg_state.setMaximum(15)
            if hasattr(self, 'btn_staircase'):
                self.btn_staircase.setText("Staircase (0..15)")

            # Zernike tab: restore
            self.lbl_zernike_dmd_banner.setVisible(False)
            self.zernike_controls_widget.setEnabled(True)

        self._update_pattern_subgroups_visibility()
        self._refresh_ranges_table()
        self._update_column_range_readout()
        self._block_signals = False

    def _on_modulator_type_changed(self, index: int) -> None:
        if self._block_signals:
            return
        is_dmd_mode = (index == 1)
        self._update_ui_for_mode(is_dmd_mode)

        if not self.stack or not (0 <= self.current_surface_idx < len(self.stack.surfaces)):
            self.update_live_plots()
            return

        surf = self.stack.surfaces[self.current_surface_idx]
        plm_cfg = ensure_plm_config(surf)

        plm_cfg.modulator_type = "DMD" if is_dmd_mode else "PLM"
        plm_cfg.bit_depth = 1 if is_dmd_mode else 4
        surf.mechanism_type = "DMD" if is_dmd_mode else "PLM"
        surf.is_dmd = is_dmd_mode
        surf.is_plm = not is_dmd_mode

        # Reload available states into combo_state
        self._block_signals = True
        states = available_states(surf)
        cur_st = active_state_name(surf)
        self.combo_state.clear()
        for k in states.keys():
            self.combo_state.addItem(k)
        if cur_st in states:
            self.combo_state.setCurrentText(cur_st)
        elif "ON" in states:
            self.combo_state.setCurrentText("ON")
        self._block_signals = False

        self._sync_to_surface()
        self.update_live_plots()
        self._trigger_retrace()

    # ------------------------------------------------------------------------
    # Surface Discovery & Synchronization
    # ------------------------------------------------------------------------
    def _draw_no_surface_placeholder(self) -> None:
        """Draw an informative status view when no PLM/DMD surface is selected."""
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        is_dark = getattr(self, 'dark_mode', False)
        bg_col = '#0b0f19' if is_dark else '#ffffff'
        text_col = '#38bdf8' if is_dark else '#0284c7'
        self.fig.patch.set_facecolor('#0f172a' if is_dark else '#ffffff')
        ax.set_facecolor(bg_col)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(
            0.5, 0.55,
            "[TI Modulator Studio] Texas Instruments PLM / DMD\n\n"
            "No active modulator surface detected in the optical system.\n"
            "Load a spectrometer design or designate any reflective surface above.",
            ha='center', va='center', color=text_col, fontsize=11, fontweight='bold',
            linespacing=1.6
        )
        self.lbl_status.setText("TI Modulator Studio: No active modulator surface selected.")
        self.fig.tight_layout()
        if hasattr(self, 'canvas') and self.canvas is not None:
            self.canvas.draw_idle()

    def refresh_surfaces(self) -> None:
        self._block_signals = True
        self.combo_surface.clear()
        if not self.stack or not self.stack.surfaces:
            self._block_signals = False
            self.current_surface_idx = -1
            self._draw_no_surface_placeholder()
            return

        mod_indices = []
        fallback_indices = []
        for idx, s in enumerate(self.stack.surfaces):
            name_up = str(getattr(s, "name", "") or "").upper()
            st = str(getattr(s, "surface_type", "") or "")
            glass_up = str(getattr(s, "glass", "") or "").upper()
            if st in (SurfaceType.AIR_SPACE, "Air Space", SurfaceType.OBJECT, SurfaceType.DETECTOR) or name_up.startswith("AIR_AFTER"):
                continue
            if is_dmd(s):
                mod_indices.append((idx, f"S{idx}: {s.name or 'TI DMD'} [TI-DMD 1-Bit Binary]"))
            elif is_plm(s):
                mod_indices.append((idx, f"S{idx}: {s.name or 'TI PLM'} [TI-PLM 4-Bit Phase]"))
            elif "DMD" in name_up:
                mod_indices.append((idx, f"S{idx}: {s.name} [TI-DMD Compatible]"))
            elif "PLM" in name_up:
                mod_indices.append((idx, f"S{idx}: {s.name} [TI-PLM Compatible]"))
            elif s.is_reflective() or "GRATING" in name_up or glass_up in ("MIRROR", "REFLECTOR", "MIRR"):
                fallback_indices.append((idx, f"S{idx}: {s.name} [Designate as Modulator]"))

        all_candidates = mod_indices if mod_indices else fallback_indices
        if not all_candidates:
            for idx, s in enumerate(self.stack.surfaces):
                st = str(getattr(s, "surface_type", "") or "")
                if st not in (SurfaceType.AIR_SPACE, "Air Space", SurfaceType.OBJECT, SurfaceType.DETECTOR) and not str(s.name).startswith("Air_after"):
                    all_candidates.append((idx, f"S{idx}: {s.name}"))

        for idx, label in all_candidates:
            self.combo_surface.addItem(label, idx)

        if all_candidates:
            target_pos = 0
            if 0 <= self.current_surface_idx < len(self.stack.surfaces):
                for c_i, (cand_idx, _) in enumerate(all_candidates):
                    if cand_idx == self.current_surface_idx:
                        target_pos = c_i
                        break
            self.combo_surface.setCurrentIndex(target_pos)
            self.current_surface_idx = all_candidates[target_pos][0]
            self._load_from_surface(self.current_surface_idx)
        else:
            self.current_surface_idx = -1

        self._block_signals = False
        if self.current_surface_idx >= 0:
            self.update_live_plots()
        else:
            self._draw_no_surface_placeholder()

    def _on_surface_selection_changed(self, index: int) -> None:
        if self._block_signals or index < 0:
            return
        surf_idx = self.combo_surface.itemData(index)
        if surf_idx is not None and 0 <= surf_idx < len(self.stack.surfaces):
            self.current_surface_idx = surf_idx
            self._load_from_surface(surf_idx)
            self.update_live_plots()

    def _load_from_surface(self, surf_idx: int) -> None:
        if not self.stack or not (0 <= surf_idx < len(self.stack.surfaces)):
            return
        surf = self.stack.surfaces[surf_idx]
        plm_cfg = ensure_plm_config(surf)
        is_dmd_mode = plm_cfg.is_dmd

        self._block_signals = True
        self.combo_modulator_type.setCurrentIndex(1 if is_dmd_mode else 0)
        self._update_ui_for_mode(is_dmd_mode)

        self.combo_model.setCurrentText(plm_cfg.model)
        self.spin_design_wl.setValue(plm_cfg.design_wavelength_um)
        self.chk_quantize.setChecked(plm_cfg.quantize_4bit)

        # Populate available states in state combo
        states = available_states(surf)
        cur_st = active_state_name(surf)
        self.combo_state.clear()
        for k in states.keys():
            self.combo_state.addItem(k)
        if cur_st in states:
            self.combo_state.setCurrentText(cur_st)
        elif "ON" in states:
            self.combo_state.setCurrentText("ON")

        # Load Zernike params
        zcfg = plm_cfg.zernike
        self.chk_zernike_enable.setChecked(zcfg.enabled and not is_dmd_mode)
        self.spin_zernike_rnorm.setValue(zcfg.normalization_radius_mm)
        for k, spin in self.zernike_spins.items():
            spin.setValue(zcfg.coeffs.get(k, 0.0))

        # Load Grating params
        gcfg = plm_cfg.grating
        self.chk_grating_enable.setChecked(gcfg.enabled)
        if hasattr(self, 'chk_col_carrier_grating'):
            self.chk_col_carrier_grating.setChecked(bool(gcfg.enabled))
        self.combo_grating_type.setCurrentText(gcfg.grating_type)
        self.spin_pitch.setValue(gcfg.pitch_um)
        self.spin_lpmm.setValue(gcfg.lines_per_mm)
        self.spin_ppp.setValue(gcfg.pixels_per_period)
        self.spin_orientation.setValue(gcfg.orientation_deg)
        self.combo_order.setCurrentText(str(gcfg.diffraction_order))
        self.spin_depth.setValue(gcfg.phase_depth_waves)
        self.spin_chirp.setValue(gcfg.chirp_lines_per_mm2)
        self.chk_fold_axis.setChecked(gcfg.fold_optical_axis)

        # Load Pattern params
        pcfg = plm_cfg.pattern
        self.chk_pattern_enable.setChecked(pcfg.enabled)
        self.combo_pattern_type.setCurrentText(pcfg.pattern_type)
        self.lbl_file_path.setText(os.path.basename(pcfg.file_path) if pcfg.file_path else "No file loaded")
        self.combo_hadamard_order.setCurrentText(str(pcfg.hadamard_order))
        self.spin_hadamard_row.setValue(pcfg.hadamard_row_index)
        if hasattr(self, 'combo_hadamard_axis'):
            cur_ax = getattr(pcfg, 'hadamard_axis', 'X')
            self.combo_hadamard_axis.setCurrentIndex(0 if cur_ax == 'X' else 1)
        if hasattr(self, 'spin_col_start') and hasattr(self, 'spin_col_end'):
            cols = max(1, int(plm_cfg.array_cols))
            self.spin_col_start.setMaximum(cols - 1)
            self.spin_col_end.setMaximum(cols - 1)
            self.spin_col_start.setValue(int(getattr(pcfg, "column_start", 0)))
            self.spin_col_end.setValue(int(getattr(pcfg, "column_end", min(50, cols - 1))))
        if hasattr(self, 'chk_col_invert'):
            self.chk_col_invert.setChecked(bool(getattr(pcfg, "column_invert", False)))
        if hasattr(self, 'spin_col_state'):
            self.spin_col_state.setValue(int(getattr(pcfg, "column_state", 1)))
        if hasattr(self, 'spin_col_bg_state'):
            self.spin_col_bg_state.setValue(int(getattr(pcfg, "column_background_state", 0)))
        raw_r = getattr(pcfg, "column_ranges", [])
        self._active_column_ranges = [dict(r) for r in raw_r] if raw_r else []
        self._refresh_ranges_table()
        if hasattr(self, 'spin_pattern_angle'):
            self.spin_pattern_angle.setValue(float(getattr(pcfg, "pattern_angle_deg", 0.0)))
        if hasattr(self, 'spin_panel_rotation'):
            self.spin_panel_rotation.setValue(float(getattr(surf, "tilt_z", 0.0)))
        self.spin_fresnel_f.setValue(pcfg.fresnel_focal_length_mm)
        self.spin_vortex.setValue(pcfg.vortex_charge)
        self._update_pattern_subgroups_visibility(pcfg.pattern_type)
        self._update_column_range_readout()

        self._block_signals = False

    def _sync_to_surface(self) -> Optional[PLMConfig]:
        if not self.stack or not (0 <= self.current_surface_idx < len(self.stack.surfaces)):
            return None
        surf = self.stack.surfaces[self.current_surface_idx]
        plm_cfg = ensure_plm_config(surf)

        is_dmd_mode = (self.combo_modulator_type.currentIndex() == 1)
        plm_cfg.modulator_type = "DMD" if is_dmd_mode else "PLM"
        plm_cfg.bit_depth = 1 if is_dmd_mode else 4
        model_name = self.combo_model.currentText()
        plm_cfg.model = model_name

        if is_dmd_mode and model_name in TI_DMD_MODELS:
            specs = TI_DMD_MODELS[model_name]
            plm_cfg.array_cols = specs["array_cols"]
            plm_cfg.array_rows = specs["array_rows"]
            plm_cfg.pixel_pitch_um = specs["pixel_pitch_um"]
            plm_cfg.active_width_mm = specs["active_width_mm"]
            plm_cfg.active_height_mm = specs["active_height_mm"]
            plm_cfg.tilt_angle_deg = specs.get("tilt_angle_deg", 12.0)
            surf.is_dmd = True
            surf.is_plm = False
            surf.mechanism_type = "DMD"
        elif (not is_dmd_mode) and model_name in TI_PLM_MODELS:
            specs = TI_PLM_MODELS[model_name]
            plm_cfg.array_cols = specs["array_cols"]
            plm_cfg.array_rows = specs["array_rows"]
            plm_cfg.pixel_pitch_um = specs["pixel_pitch_um"]
            plm_cfg.active_width_mm = specs["active_width_mm"]
            plm_cfg.active_height_mm = specs["active_height_mm"]
            surf.is_plm = True
            surf.is_dmd = False
            surf.mechanism_type = "PLM"

        plm_cfg.design_wavelength_um = float(self.spin_design_wl.value())
        plm_cfg.quantize_4bit = bool(self.chk_quantize.isChecked())

        # Zernike
        plm_cfg.zernike.enabled = bool(self.chk_zernike_enable.isChecked()) and not is_dmd_mode
        plm_cfg.zernike.normalization_radius_mm = float(self.spin_zernike_rnorm.value())
        for k, spin in self.zernike_spins.items():
            plm_cfg.zernike.coeffs[k] = float(spin.value())

        # Grating
        if bool(self.chk_pattern_enable.isChecked()) and ("COLUMN" in str(self.combo_pattern_type.currentText()).upper()):
            grating_on = bool(self.chk_col_carrier_grating.isChecked()) if hasattr(self, 'chk_col_carrier_grating') else False
            self.chk_grating_enable.setChecked(grating_on)
            plm_cfg.grating.enabled = grating_on
        else:
            plm_cfg.grating.enabled = bool(self.chk_grating_enable.isChecked())
        plm_cfg.grating.grating_type = self.combo_grating_type.currentText()
        plm_cfg.grating.pitch_um = float(self.spin_pitch.value())
        plm_cfg.grating.lines_per_mm = float(self.spin_lpmm.value())
        plm_cfg.grating.pixels_per_period = float(self.spin_ppp.value())
        plm_cfg.grating.orientation_deg = float(self.spin_orientation.value())
        try:
            plm_cfg.grating.diffraction_order = int(self.combo_order.currentText())
        except Exception:
            plm_cfg.grating.diffraction_order = 1
        plm_cfg.grating.phase_depth_waves = float(self.spin_depth.value())
        plm_cfg.grating.chirp_lines_per_mm2 = float(self.spin_chirp.value())
        plm_cfg.grating.fold_optical_axis = bool(self.chk_fold_axis.isChecked())

        if plm_cfg.grating.enabled:
            surf.grating_lines_per_mm = plm_cfg.grating.lines_per_mm
            surf.grating_order = plm_cfg.grating.diffraction_order
        else:
            surf.grating_lines_per_mm = 0.0
            surf.grating_order = 0

        # Sync active mechanical state
        st_text = self.combo_state.currentText()
        if st_text:
            surf.mechanical_state = st_text

        # Pattern
        plm_cfg.pattern.enabled = bool(self.chk_pattern_enable.isChecked())
        plm_cfg.pattern.pattern_type = self.combo_pattern_type.currentText()
        try:
            plm_cfg.pattern.hadamard_order = int(self.combo_hadamard_order.currentText())
        except Exception:
            plm_cfg.pattern.hadamard_order = 32
        plm_cfg.pattern.hadamard_row_index = int(self.spin_hadamard_row.value())
        if hasattr(self, 'combo_hadamard_axis'):
            plm_cfg.pattern.hadamard_axis = "X" if "X" in self.combo_hadamard_axis.currentText() else "Y"
        if hasattr(self, 'spin_col_start'):
            plm_cfg.pattern.column_start = int(self.spin_col_start.value())
        if hasattr(self, 'spin_col_end'):
            plm_cfg.pattern.column_end = int(self.spin_col_end.value())
        if hasattr(self, 'chk_col_invert'):
            plm_cfg.pattern.column_invert = bool(self.chk_col_invert.isChecked())
        if hasattr(self, 'spin_col_state'):
            plm_cfg.pattern.column_state = int(self.spin_col_state.value())
        if hasattr(self, 'spin_col_bg_state'):
            plm_cfg.pattern.column_background_state = int(self.spin_col_bg_state.value())
        plm_cfg.pattern.column_ranges = [dict(r) for r in getattr(self, '_active_column_ranges', [])]
        if hasattr(self, 'spin_pattern_angle'):
            plm_cfg.pattern.pattern_angle_deg = float(self.spin_pattern_angle.value())
        if hasattr(self, 'spin_panel_rotation'):
            cur_rot = float(self.spin_panel_rotation.value())
            if abs(float(getattr(surf, 'tilt_z', 0.0)) - cur_rot) > 1e-5:
                surf.tilt_z = cur_rot
                self.stack._update_accumulated_positions()
            plm_cfg.panel_rotation_deg = cur_rot
        plm_cfg.pattern.fresnel_focal_length_mm = float(self.spin_fresnel_f.value())
        plm_cfg.pattern.vortex_charge = int(self.spin_vortex.value())

        surf.plm_config = plm_cfg
        return plm_cfg

    # ------------------------------------------------------------------------
    # Parameter Event Handlers & Unit Readouts
    # ------------------------------------------------------------------------
    def _on_unit_changed(self, text: str) -> None:
        self.update_live_plots()

    def _update_um_readouts(self) -> None:
        """Update dynamic readouts across all control tabs."""
        is_dmd_mode = self.combo_modulator_type.currentIndex() == 1
        wl_um = max(float(self.spin_design_wl.value()), 1e-4)

        if hasattr(self, 'lbl_step_um'):
            if is_dmd_mode:
                pitch = self._get_pixel_pitch_um()
                tilt = 12.0
                if self.stack and 0 <= self.current_surface_idx < len(self.stack.surfaces):
                    surf = self.stack.surfaces[self.current_surface_idx]
                    cfg = getattr(surf, "plm_config", None)
                    if cfg is not None:
                        tilt = getattr(cfg, "tilt_angle_deg", 12.0)
                self.lbl_step_um.setText(f"Tilt: ±{tilt:.1f}° ({pitch:.2f} µm pitch)")
            else:
                self.lbl_step_um.setText(f"1-LSB: {wl_um/16.0:.4f} µm ({wl_um/32.0:.4f} µm stroke)")

        if hasattr(self, 'lbl_depth_um') and hasattr(self, 'spin_depth'):
            self.lbl_depth_um.setText(f"({self.spin_depth.value() * wl_um:.3f} µm)")

        if hasattr(self, 'zernike_spins') and hasattr(self, 'zernike_um_labels'):
            for key, spin in self.zernike_spins.items():
                lbl = self.zernike_um_labels.get(key)
                if lbl is not None:
                    lbl.setText(f"({spin.value() * wl_um:+.3f} µm)")

        self._update_column_range_readout()

    def _trigger_retrace(self) -> None:
        """Trigger optical ray tracing whenever modulator mask or phase parameters change."""
        if self._block_signals:
            return
        if self.stack:
            try:
                self.stack._update_accumulated_positions()
            except Exception:
                pass
        self.pattern_applied.emit(self.current_surface_idx)

    def _on_model_changed(self, text: str) -> None:
        if self._block_signals:
            return
        self._sync_to_surface()
        if hasattr(self, 'spin_col_start') and hasattr(self, 'spin_col_end'):
            cols = 1140
            if self.stack and 0 <= self.current_surface_idx < len(self.stack.surfaces):
                cfg = getattr(self.stack.surfaces[self.current_surface_idx], 'plm_config', None)
                if cfg is not None:
                    cols = max(1, int(cfg.array_cols))
            self.spin_col_start.setMaximum(cols - 1)
            self.spin_col_end.setMaximum(cols - 1)
        self._update_um_readouts()
        self.update_live_plots()
        self._trigger_retrace()

    def _on_design_wl_changed(self, val: float) -> None:
        if self._block_signals:
            return
        self._update_um_readouts()
        self._sync_to_surface()
        self.update_live_plots()
        self._trigger_retrace()

    def _on_state_changed(self, index: int) -> None:
        if self._block_signals or index < 0 or not self.stack:
            return
        st = self.combo_state.currentText()
        ctrl = MechanicalStateController(self.stack)
        try:
            ctrl.set_state(self.current_surface_idx, st)
            self.stack._update_accumulated_positions()
        except Exception:
            pass
        self._sync_to_surface()
        self.update_live_plots()
        self._trigger_retrace()

    def _prev_state(self) -> None:
        idx = self.combo_state.currentIndex()
        if idx > 0:
            self.combo_state.setCurrentIndex(idx - 1)
        else:
            self.combo_state.setCurrentIndex(self.combo_state.count() - 1)

    def _next_state(self) -> None:
        idx = self.combo_state.currentIndex()
        if idx < self.combo_state.count() - 1:
            self.combo_state.setCurrentIndex(idx + 1)
        else:
            self.combo_state.setCurrentIndex(0)

    def _on_quantize_toggled(self, checked: bool) -> None:
        if self._block_signals:
            return
        self._sync_to_surface()
        self.update_live_plots()
        self._trigger_retrace()

    def _on_zernike_param_changed(self) -> None:
        if self._block_signals:
            return
        self._update_um_readouts()
        self._sync_to_surface()
        self._request_live_plot_update()
        self._trigger_retrace()

    def _autofit_zernike_radius(self) -> None:
        if not self.stack or not (0 <= self.current_surface_idx < len(self.stack.surfaces)):
            return
        surf = self.stack.surfaces[self.current_surface_idx]
        sd = float(getattr(surf, 'semi_diameter', 4.32))
        self.spin_zernike_rnorm.setValue(sd if sd > 0 else 4.32)

    def _zero_zernike(self) -> None:
        self._block_signals = True
        for spin in self.zernike_spins.values():
            spin.setValue(0.0)
        self._block_signals = False
        self._update_um_readouts()
        self._sync_to_surface()
        self.update_live_plots()
        self._trigger_retrace()

    def _set_zernike_preset(self, values: Dict[str, float]) -> None:
        self._block_signals = True
        for k, spin in self.zernike_spins.items():
            spin.setValue(values.get(k, 0.0))
        self._block_signals = False
        self._update_um_readouts()
        self._sync_to_surface()
        self.update_live_plots()
        self._trigger_retrace()

    def _preset_spectrometer_correction(self) -> None:
        """Apply typical Czerny-Turner / Littrow spectrometer aberration correction on the PLM."""
        self._set_zernike_preset({
            "z4_defocus": -0.45,           # Chromatic defocus correction
            "z6_astig_vertical": -0.85,    # Off-axis grating astigmatism compensation
            "z7_coma_vertical": 0.25,      # Asymmetric tilted doublet coma cancellation
            "z11_spherical": -0.15,        # Spherical aberration minimization
        })

    def _on_grating_param_changed(self) -> None:
        if self._block_signals:
            return
        self._update_um_readouts()
        self._sync_to_surface()
        self._request_live_plot_update()
        self._trigger_retrace()

    def _on_pitch_changed(self, val: float) -> None:
        if self._block_signals:
            return
        self._block_signals = True
        pitch = max(val, 1e-4)
        lpmm = 1000.0 / pitch
        pitch_pix = self._get_pixel_pitch_um()
        ppp = pitch / pitch_pix
        self.spin_lpmm.setValue(lpmm)
        self.spin_ppp.setValue(ppp)
        self._block_signals = False
        self._sync_to_surface()
        self._request_live_plot_update()
        self._trigger_retrace()

    def _on_lpmm_changed(self, val: float) -> None:
        if self._block_signals:
            return
        self._block_signals = True
        pitch = 1000.0 / max(val, 1e-4)
        pitch_pix = self._get_pixel_pitch_um()
        ppp = pitch / pitch_pix
        self.spin_pitch.setValue(pitch)
        self.spin_ppp.setValue(ppp)
        self._block_signals = False
        self._sync_to_surface()
        self._request_live_plot_update()
        self._trigger_retrace()

    def _on_ppp_changed(self, val: float) -> None:
        if self._block_signals:
            return
        self._block_signals = True
        pitch_pix = self._get_pixel_pitch_um()
        pitch = val * pitch_pix
        lpmm = 1000.0 / max(pitch, 1e-4)
        self.spin_pitch.setValue(pitch)
        self.spin_lpmm.setValue(lpmm)
        self._block_signals = False
        self._sync_to_surface()
        self._request_live_plot_update()
        self._trigger_retrace()

    def _set_grating_preset(self, pitch_um: float, orient_deg: float, gtype: str) -> None:
        self._block_signals = True
        self.combo_grating_type.setCurrentText(gtype)
        self.spin_pitch.setValue(pitch_um)
        self.spin_lpmm.setValue(1000.0 / pitch_um)
        pitch_pix = self._get_pixel_pitch_um()
        self.spin_ppp.setValue(pitch_um / pitch_pix)
        self.spin_orientation.setValue(orient_deg)
        self._block_signals = False
        self._sync_to_surface()
        self.update_live_plots()
        self._trigger_retrace()

    def _on_column_range_changed(self) -> None:
        if self._block_signals:
            return
        self._update_column_range_readout()
        self._on_pattern_param_changed()

    def _on_col_carrier_grating_toggled(self, checked: bool) -> None:
        if self._block_signals:
            return
        self.chk_grating_enable.setChecked(checked)
        self._sync_to_surface()
        self.update_live_plots()
        self._trigger_retrace()

    def _update_column_range_readout(self) -> None:
        if not hasattr(self, 'lbl_col_width_info') or not hasattr(self, 'spin_col_start') or not hasattr(self, 'spin_col_end'):
            return
        is_dmd = self.combo_modulator_type.currentIndex() == 1
        pitch_um = self._get_pixel_pitch_um()
        wl_um = max(float(self.spin_design_wl.value()) if hasattr(self, 'spin_design_wl') else 1.225, 1e-4)
        c_start = int(self.spin_col_start.value())
        c_end = int(self.spin_col_end.value())
        if c_start > c_end:
            c_start, c_end = c_end, c_start
        ncols = max(1, c_end - c_start + 1)
        w_mm = ncols * (pitch_um * 1e-3)
        tot_cols = 1140
        if self.stack and 0 <= self.current_surface_idx < len(self.stack.surfaces):
            cfg = getattr(self.stack.surfaces[self.current_surface_idx], 'plm_config', None)
            if cfg is not None:
                tot_cols = max(1, int(cfg.array_cols))
        center_x = ((c_start + c_end) / 2.0 - (tot_cols - 1) / 2.0) * (pitch_um * 1e-3)
        ang = float(self.spin_pattern_angle.value()) if hasattr(self, 'spin_pattern_angle') else 0.0
        ang_str = f" @ {ang:+.1f}°" if abs(ang) > 1e-3 else ""

        # Update state description label
        st_val = int(self.spin_col_state.value()) if hasattr(self, 'spin_col_state') else 1
        bg_st_val = int(self.spin_col_bg_state.value()) if hasattr(self, 'spin_col_bg_state') else 0
        if hasattr(self, 'lbl_col_state_desc'):
            if is_dmd:
                self.lbl_col_state_desc.setText(f"({st_val}: ON / Reflected)" if st_val == 1 else f"({st_val}: OFF / Dump)")
            else:
                opd_um = (st_val / 16.0) * wl_um
                rad_val = (st_val / 16.0) * 2.0 * math.pi
                self.lbl_col_state_desc.setText(f"(State {st_val}/16: {opd_um:.3f} µm, {rad_val:.2f} rad)")

        inv = hasattr(self, 'chk_col_invert') and self.chk_col_invert.isChecked()
        nranges = len(getattr(self, '_active_column_ranges', []))
        if nranges > 1:
            tot_active = sum(abs(int(r.get('end', 0)) - int(r.get('start', 0))) + 1 for r in self._active_column_ranges)
            self.lbl_col_width_info.setText(
                f"{nranges} Custom Ranges Configured ({tot_active} cols / {tot_active * pitch_um * 1e-3:.3f} mm total, Bg State: {bg_st_val}){ang_str}"
            )
        elif inv:
            self.lbl_col_width_info.setText(
                f"Inverted Notch: Cols {c_start}..{c_end} OFF ({ncols} cols / {w_mm:.3f} mm), {tot_cols - ncols} cols ON{ang_str}"
            )
        else:
            st_text = f" -> State {st_val}" if not is_dmd or st_val != 1 else " ON"
            self.lbl_col_width_info.setText(
                f"Active Slit: Cols {c_start}..{c_end}{st_text} ({ncols} cols / {w_mm:.3f} mm wide, center X: {center_x:+.3f} mm){ang_str}"
            )

    def _refresh_ranges_table(self) -> None:
        if not hasattr(self, 'table_col_ranges'):
            return
        is_dmd = self.combo_modulator_type.currentIndex() == 1
        pitch_um = self._get_pixel_pitch_um()
        wl_um = max(float(self.spin_design_wl.value()) if hasattr(self, 'spin_design_wl') else 1.225, 1e-4)

        ranges = getattr(self, '_active_column_ranges', [])
        self.table_col_ranges.setRowCount(len(ranges))
        for row_idx, r in enumerate(ranges):
            r_s = int(r.get('start', 0))
            r_e = int(r.get('end', 0))
            st = int(r.get('state', 1))
            ncols = max(1, abs(r_e - r_s) + 1)
            w_mm = ncols * (pitch_um * 1e-3)

            item_idx = QTableWidgetItem(f"R{row_idx + 1}")
            item_cols = QTableWidgetItem(f"{r_s} .. {r_e} ({ncols} cols)")
            item_w = QTableWidgetItem(f"{w_mm:.3f} mm")
            item_st = QTableWidgetItem(str(st))
            if is_dmd:
                desc = "ON (Detector)" if st == 1 else "OFF (Dump)"
            else:
                opd = (st / 16.0) * wl_um
                desc = f"Phase {opd:.3f} µm (S{st}/16)"
            item_desc = QTableWidgetItem(desc)

            for col_i, item in enumerate([item_idx, item_cols, item_w, item_st, item_desc]):
                if hasattr(item, 'setTextAlignment'):
                    item.setTextAlignment(getattr(Qt.AlignmentFlag, 'AlignCenter', 4) if hasattr(Qt, 'AlignmentFlag') else 4)
                self.table_col_ranges.setItem(row_idx, col_i, item)

    def _on_table_range_clicked(self, row: int, col: int) -> None:
        ranges = getattr(self, '_active_column_ranges', [])
        if 0 <= row < len(ranges):
            r = ranges[row]
            self._block_signals = True
            self.spin_col_start.setValue(int(r.get('start', 0)))
            self.spin_col_end.setValue(int(r.get('end', 0)))
            self.spin_col_state.setValue(int(r.get('state', 1)))
            self._block_signals = False
            self._update_column_range_readout()

    def _on_add_range_clicked(self) -> None:
        c_s = int(self.spin_col_start.value())
        c_e = int(self.spin_col_end.value())
        if c_s > c_e:
            c_s, c_e = c_e, c_s
        st = int(self.spin_col_state.value())
        if not hasattr(self, '_active_column_ranges'):
            self._active_column_ranges = []

        matched = False
        sel_row = self.table_col_ranges.currentRow() if hasattr(self, 'table_col_ranges') else -1
        if 0 <= sel_row < len(self._active_column_ranges):
            self._active_column_ranges[sel_row] = {"start": c_s, "end": c_e, "state": st}
            matched = True
        else:
            for idx, r in enumerate(self._active_column_ranges):
                if r.get("start") == c_s and r.get("end") == c_e:
                    self._active_column_ranges[idx] = {"start": c_s, "end": c_e, "state": st}
                    matched = True
                    break
        if not matched:
            self._active_column_ranges.append({"start": c_s, "end": c_e, "state": st})

        if hasattr(self, 'chk_col_carrier_grating') and not self.chk_col_carrier_grating.isChecked():
            self.chk_grating_enable.setChecked(False)
        self.chk_pattern_enable.setChecked(True)
        self.combo_pattern_type.setCurrentText("Column_Range")
        self._refresh_ranges_table()
        self._update_column_range_readout()
        self._sync_to_surface()
        self.update_live_plots()
        self._trigger_retrace()

    def _on_remove_range_clicked(self) -> None:
        if not hasattr(self, '_active_column_ranges') or not self._active_column_ranges:
            return
        sel_row = self.table_col_ranges.currentRow() if hasattr(self, 'table_col_ranges') else -1
        if 0 <= sel_row < len(self._active_column_ranges):
            self._active_column_ranges.pop(sel_row)
        elif self._active_column_ranges:
            self._active_column_ranges.pop()
        self._refresh_ranges_table()
        self._update_column_range_readout()
        self._sync_to_surface()
        self.update_live_plots()
        self._trigger_retrace()

    def _on_clear_ranges_clicked(self) -> None:
        self._active_column_ranges = []
        self._refresh_ranges_table()
        self._update_column_range_readout()
        self._sync_to_surface()
        self.update_live_plots()
        self._trigger_retrace()

    def _set_staircase_preset(self) -> None:
        if not self.stack or not (0 <= self.current_surface_idx < len(self.stack.surfaces)):
            return
        surf = self.stack.surfaces[self.current_surface_idx]
        plm_cfg = ensure_plm_config(surf)
        tot_cols = max(1, int(plm_cfg.array_cols))
        is_dmd = self.combo_modulator_type.currentIndex() == 1
        num_steps = 2 if is_dmd else 16
        step_cols = max(1, tot_cols // num_steps)

        new_ranges = []
        for s in range(num_steps):
            c_s = s * step_cols
            c_e = min(tot_cols - 1, (s + 1) * step_cols - 1) if s < num_steps - 1 else tot_cols - 1
            new_ranges.append({"start": c_s, "end": c_e, "state": s})

        self._active_column_ranges = new_ranges
        self._block_signals = True
        self.combo_pattern_type.setCurrentText("Column_Range")
        self.chk_pattern_enable.setChecked(True)
        self.chk_grating_enable.setChecked(False)
        if hasattr(self, 'chk_col_carrier_grating'):
            self.chk_col_carrier_grating.setChecked(False)
        self._block_signals = False
        self._refresh_ranges_table()
        self._update_column_range_readout()
        self._sync_to_surface()
        self.update_live_plots()
        self._trigger_retrace()

    def _set_ronchi_preset(self) -> None:
        if not self.stack or not (0 <= self.current_surface_idx < len(self.stack.surfaces)):
            return
        surf = self.stack.surfaces[self.current_surface_idx]
        plm_cfg = ensure_plm_config(surf)
        tot_cols = max(1, int(plm_cfg.array_cols))
        is_dmd = self.combo_modulator_type.currentIndex() == 1
        n_bands = 8
        band_w = max(1, tot_cols // n_bands)
        new_ranges = []
        for b in range(n_bands):
            c_s = b * band_w
            c_e = min(tot_cols - 1, (b + 1) * band_w - 1) if b < n_bands - 1 else tot_cols - 1
            st = (b % 2) if is_dmd else ((b % 2) * 8)
            new_ranges.append({"start": c_s, "end": c_e, "state": st})
        self._active_column_ranges = new_ranges
        self._block_signals = True
        self.combo_pattern_type.setCurrentText("Column_Range")
        self.chk_pattern_enable.setChecked(True)
        self.chk_grating_enable.setChecked(False)
        if hasattr(self, 'chk_col_carrier_grating'):
            self.chk_col_carrier_grating.setChecked(False)
        self._block_signals = False
        self._refresh_ranges_table()
        self._update_column_range_readout()
        self._sync_to_surface()
        self.update_live_plots()
        self._trigger_retrace()

    def _set_column_preset(self, count: int) -> None:
        if not self.stack or not (0 <= self.current_surface_idx < len(self.stack.surfaces)):
            return
        surf = self.stack.surfaces[self.current_surface_idx]
        plm_cfg = ensure_plm_config(surf)
        tot_cols = max(1, int(plm_cfg.array_cols))
        if count <= 0:  # Full array
            c_start = 0
            c_end = tot_cols - 1
        else:
            mid = tot_cols // 2
            half = count // 2
            c_start = max(0, mid - half)
            c_end = min(tot_cols - 1, c_start + count - 1)
        self._block_signals = True
        self.spin_col_start.setMaximum(tot_cols - 1)
        self.spin_col_end.setMaximum(tot_cols - 1)
        self.spin_col_start.setValue(c_start)
        self.spin_col_end.setValue(c_end)
        self.spin_col_state.setValue(1)
        self._active_column_ranges = []  # Single active range
        self.combo_pattern_type.setCurrentText("Column_Range")
        self.chk_pattern_enable.setChecked(True)
        self.chk_grating_enable.setChecked(False)
        if hasattr(self, 'chk_col_carrier_grating'):
            self.chk_col_carrier_grating.setChecked(False)
        self._update_pattern_subgroups_visibility("Column_Range")
        self._refresh_ranges_table()
        self._update_column_range_readout()
        self._block_signals = False
        self._sync_to_surface()
        self.update_live_plots()
        self._trigger_retrace()

    def _set_pattern_angle(self, deg: float) -> None:
        if hasattr(self, 'spin_pattern_angle'):
            self.spin_pattern_angle.setValue(deg)
            self._update_column_range_readout()

    def _on_panel_rotation_changed(self, val: float) -> None:
        if self._block_signals:
            return
        if self.stack and 0 <= self.current_surface_idx < len(self.stack.surfaces):
            surf = self.stack.surfaces[self.current_surface_idx]
            surf.tilt_z = float(val)
            self.stack._update_accumulated_positions()
        self._sync_to_surface()
        self.update_live_plots()
        self._trigger_retrace()

    def _set_panel_rotation(self, deg: float) -> None:
        if hasattr(self, 'spin_panel_rotation'):
            self.spin_panel_rotation.setValue(deg)

    def _update_pattern_subgroups_visibility(self, ptype: Optional[str] = None) -> None:
        if ptype is None:
            ptype = self.combo_pattern_type.currentText()
        ptype_up = str(ptype).strip().upper()
        is_dmd = self.combo_modulator_type.currentIndex() == 1

        if hasattr(self, 'grp_hadamard'):
            self.grp_hadamard.setVisible("HADAMARD" in ptype_up)
        if hasattr(self, 'grp_column_range'):
            self.grp_column_range.setVisible("COLUMN" in ptype_up or ptype_up in ("COLUMN_RANGE", "SLIT"))
        if hasattr(self, 'grp_fresnel'):
            self.grp_fresnel.setVisible(not is_dmd and "FRESNEL" in ptype_up)
        if hasattr(self, 'grp_vortex'):
            self.grp_vortex.setVisible(not is_dmd and "VORTEX" in ptype_up)

    def _on_pattern_param_changed(self) -> None:
        if self._block_signals:
            return
        self._update_column_range_readout()
        self._sync_to_surface()
        self._request_live_plot_update()
        self._trigger_retrace()

    def _on_pattern_type_changed(self, ptype: str) -> None:
        if self._block_signals:
            return
        self._block_signals = True
        self.chk_pattern_enable.setChecked(ptype != "None")
        if "COLUMN" in str(ptype).upper():
            if hasattr(self, 'chk_col_carrier_grating') and not self.chk_col_carrier_grating.isChecked():
                self.chk_grating_enable.setChecked(False)
        self._update_pattern_subgroups_visibility(ptype)
        self._block_signals = False
        self._sync_to_surface()
        self.update_live_plots()
        self._trigger_retrace()

    def _load_file_dialog(self) -> None:
        is_dmd_mode = self.combo_modulator_type.currentIndex() == 1
        prompt = "Load 2D Binary Pattern for TI-DMD" if is_dmd_mode else "Load 2D Phase Pattern for TI-PLM"
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            prompt,
            "",
            "Pattern Files (*.png *.bmp *.jpg *.tif *.npy *.csv *.json);;All Files (*)"
        )
        if not file_path:
            return
        try:
            arr = load_pattern_from_file(file_path)
            surf = self.stack.surfaces[self.current_surface_idx]
            plm_cfg = ensure_plm_config(surf)

            if is_dmd_mode:
                # DMD is strictly binary: auto-binarize/threshold at 0.5
                arr = (arr >= 0.5).astype(float)
                bin_note = "\n(Automatically binarized to 1-bit {0, 1} states for TI-DMD micromirrors)"
            else:
                bin_note = ""

            plm_cfg._custom_pattern_cache = arr
            plm_cfg.pattern.file_path = file_path
            plm_cfg.pattern.pattern_type = "File"
            plm_cfg.pattern.enabled = True

            self._block_signals = True
            self.combo_pattern_type.setCurrentText("File")
            self.chk_pattern_enable.setChecked(True)
            self.lbl_file_path.setText(os.path.basename(file_path))
            self._block_signals = False

            self.update_live_plots()
            self._trigger_retrace()
            QMessageBox.information(
                self, "Pattern Loaded",
                f"Successfully loaded pattern from:\n{file_path}\n"
                f"Shape: {arr.shape} | Range: [{np.nanmin(arr):.3f}, {np.nanmax(arr):.3f}]{bin_note}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load pattern file:\n{str(e)}")

    def _export_pattern_dialog(self) -> None:
        surf = self.stack.surfaces[self.current_surface_idx]
        plm_cfg = ensure_plm_config(surf)
        maps = generate_plm_2d_maps(plm_cfg, mechanical_state=self.combo_state.currentText(), nx=plm_cfg.array_cols, ny=plm_cfg.array_rows)

        is_dmd_mode = plm_cfg.is_dmd
        default_name = "dmd_1bit_pattern.npy" if is_dmd_mode else "plm_4bit_pattern.npy"
        dialog_title = "Export TI-DMD 1-Bit Binary Pattern" if is_dmd_mode else "Export TI-PLM 4-Bit Quantized Pattern"

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            dialog_title,
            default_name,
            "NumPy Array (*.npy);;CSV Text (*.csv);;TI Pattern JSON (*.json);;PNG Image (*.png);;All Files (*)"
        )
        if not file_path:
            return
        try:
            saved = save_pattern_to_file(file_path, states_4bit=maps["state_4bit"], phase_rad=maps["phase_rad"], plm_config=plm_cfg)
            QMessageBox.information(self, "Export Successful", f"Saved modulator pattern to:\n{saved}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to save pattern:\n{str(e)}")

    def _apply_and_retrace(self) -> None:
        self._sync_to_surface()
        if self.stack:
            self.stack._update_accumulated_positions()
        self.pattern_applied.emit(self.current_surface_idx)

    def _zoom_out_psf(self) -> None:
        """Zoom out Far-Field PSF to full field."""
        if hasattr(self, 'combo_psf_zoom') and self.combo_psf_zoom:
            self.combo_psf_zoom.setCurrentIndex(0)
        self._request_live_plot_update()

    def _on_canvas_scroll(self, event) -> None:
        """Handle mouse wheel scrolling on plots (zoom in/out on PSF)."""
        if event.inaxes is None or not hasattr(self, 'combo_psf_zoom') or not self.combo_psf_zoom:
            return
        if hasattr(self, '_last_ax_psf') and event.inaxes is self._last_ax_psf:
            cur = self.combo_psf_zoom.currentIndex()
            if event.button == 'down':
                # Wheel down -> zoom out (towards index 0)
                if cur > 0:
                    self.combo_psf_zoom.setCurrentIndex(cur - 1)
            elif event.button == 'up':
                # Wheel up -> zoom in (towards index 3)
                if cur < self.combo_psf_zoom.count() - 1:
                    self.combo_psf_zoom.setCurrentIndex(cur + 1)

    # ------------------------------------------------------------------------
    # Live Visualizations (2D Phase/Binary Map, Simulated Far-Field PSF, 1D Cross-Section)
    # ------------------------------------------------------------------------
    def update_live_plots(self) -> None:
        if getattr(self, '_plot_debounce_timer', None) is not None and self._plot_debounce_timer.isActive():
            self._plot_debounce_timer.stop()

        if not self.stack or not (0 <= self.current_surface_idx < len(self.stack.surfaces)):
            self._draw_no_surface_placeholder()
            return

        surf = self.stack.surfaces[self.current_surface_idx]
        plm_cfg = ensure_plm_config(surf)
        cur_st = self.combo_state.currentText() or active_state_name(surf) or "ON"
        wl_um = max(float(plm_cfg.design_wavelength_um), 1e-4)

        # Update dynamic readouts
        self._update_um_readouts()

        # Generate fast 2D preview
        maps = generate_plm_2d_maps(plm_cfg, mechanical_state=cur_st, nx=340, ny=200)

        is_dark = getattr(self, 'dark_mode', False)
        bg_fig = '#0f172a' if is_dark else '#ffffff'
        bg_ax = '#0b0f19' if is_dark else '#ffffff'
        fg_text = '#f1f5f9' if is_dark else '#0f172a'
        fg_subtext = '#94a3b8' if is_dark else '#475569'
        spine_col = '#334155' if is_dark else '#cbd5e1'

        self.fig.patch.set_facecolor(bg_fig)
        self.fig.clear()

        # Exactly 3 optical diagnostics: 2D Map, Simulated Far-Field PSF, 1D Cross-Section
        gs = self.fig.add_gridspec(2, 2, height_ratios=[1.1, 1.0])
        ax_map = self.fig.add_subplot(gs[0, 0])
        ax_psf = self.fig.add_subplot(gs[0, 1])
        ax_prof = self.fig.add_subplot(gs[1, :])

        for ax in (ax_map, ax_psf, ax_prof):
            ax.set_facecolor(bg_ax)
            ax.tick_params(colors=fg_subtext, labelsize=8)
            for spine in ax.spines.values():
                spine.set_color(spine_col)

        # --------------------------------------------------------------------
        # BRANCH A: TI-DMD (1-Bit Binary Modulator)
        # --------------------------------------------------------------------
        if plm_cfg.is_dmd:
            tilt_deg = float(getattr(plm_cfg, "tilt_angle_deg", 12.0))
            data_binary = (maps["state_4bit"] > 0).astype(int)

            # High-contrast binary colormap (#151922 OFF, #FFB300 ON)
            dmd_cmap = mcolors.ListedColormap(['#151922', '#ffb300'])
            im_map = ax_map.imshow(
                data_binary,
                cmap=dmd_cmap,
                vmin=0,
                vmax=1,
                extent=[-0.5 * plm_cfg.active_width_mm, 0.5 * plm_cfg.active_width_mm,
                        -0.5 * plm_cfg.active_height_mm, 0.5 * plm_cfg.active_height_mm],
                origin='lower',
                aspect='auto'
            )
            ax_map.set_title(f"TI-DMD 2D Binary Pattern [{cur_st}] (Tilt: ±{tilt_deg:.1f}°)", fontsize=9, color=fg_text, fontweight='bold')
            ax_map.set_xlabel("X (mm)", fontsize=8, color=fg_subtext)
            ax_map.set_ylabel("Y (mm)", fontsize=8, color=fg_subtext)
            cbar_map = self.fig.colorbar(im_map, ax=ax_map, fraction=0.046, pad=0.04, ticks=[0.25, 0.75])
            cbar_map.ax.set_yticklabels([f"0 (OFF / -{tilt_deg:.0f}°)", f"1 (ON / +{tilt_deg:.0f}°)"])
            cbar_map.ax.tick_params(colors=fg_subtext, labelsize=7)
            if hasattr(cbar_map, 'outline') and cbar_map.outline is not None:
                cbar_map.outline.set_edgecolor(spine_col)

            if cur_st.upper() in ("OFF", "FLAT") or np.all(data_binary == 0):
                banner_fc = "#1e293b" if is_dark else "#fffbeb"
                banner_ec = "#f59e0b" if is_dark else "#b45309"
                banner_txt = "#fbbf24" if is_dark else "#92400e"
                ax_map.text(
                    0.5, 0.5,
                    f"[{cur_st}] UNPOWERED / SPECULAR FLAT STATE\n(Micromirrors at rest · -{tilt_deg:.0f}° / 0° Specular Dump)\nSelect 'ON' or choose a binary pattern to tilt micromirrors",
                    ha='center', va='center', transform=ax_map.transAxes,
                    color=banner_txt, fontsize=8.0, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.5", fc=banner_fc, ec=banner_ec, lw=1.2, alpha=0.92)
                )

            # 2. Simulated Far-Field PSF / Diffraction Pattern (2D FFT of binary aperture)
            psf = maps["psf_intensity"]
            ny_p, nx_p = psf.shape
            cy, cx = ny_p // 2, nx_p // 2

            zoom_idx = self.combo_psf_zoom.currentIndex() if hasattr(self, 'combo_psf_zoom') and self.combo_psf_zoom else 0
            ext_full = maps.get("psf_extent_mrad", None)
            if zoom_idx == 1:
                # Wide (Zoom 2×)
                ry = max(10, ny_p // 4)
                rx = max(10, nx_p // 4)
                psf_crop = psf[cy - ry: cy + ry, cx - rx: cx + rx]
                ext = [ext_full[0]/2.0, ext_full[1]/2.0, ext_full[2]/2.0, ext_full[3]/2.0] if ext_full else None
                zoom_tag = " [Wide 2×]"
            elif zoom_idx == 2:
                # Medium (Zoom 4×)
                ry = max(10, ny_p // 8)
                rx = max(10, nx_p // 8)
                psf_crop = psf[cy - ry: cy + ry, cx - rx: cx + rx]
                ext = [ext_full[0]/4.0, ext_full[1]/4.0, ext_full[2]/4.0, ext_full[3]/4.0] if ext_full else None
                zoom_tag = " [Medium 4×]"
            elif zoom_idx == 3:
                # Close-up (Zoom 8×)
                ry = max(10, ny_p // 16)
                rx = max(10, nx_p // 16)
                psf_crop = psf[cy - ry: cy + ry, cx - rx: cx + rx]
                ext = [ext_full[0]/8.0, ext_full[1]/8.0, ext_full[2]/8.0, ext_full[3]/8.0] if ext_full else None
                zoom_tag = " [Close-up 8×]"
            else:
                # Full Field (Zoom Out 1×) - completely uncropped!
                psf_crop = psf
                ext = ext_full
                zoom_tag = " [Zoom Out 1×]"

            use_log = self.chk_psf_log.isChecked() if hasattr(self, 'chk_psf_log') and self.chk_psf_log else True
            if use_log:
                psf_disp = 10.0 * np.log10(np.maximum(psf_crop, 3e-4))
                vmin_psf, vmax_psf = -35.0, 0.0
                cbar_title = "Intensity [dB]"
            else:
                psf_disp = psf_crop
                vmin_psf, vmax_psf = 0.0, 1.0
                cbar_title = "Linear [0..1]"

            if ext is not None:
                im_psf = ax_psf.imshow(psf_disp, cmap="inferno", origin='lower', aspect='auto', extent=ext, vmin=vmin_psf, vmax=vmax_psf)
                ax_psf.set_xlabel("θx (mrad)", fontsize=8, color=fg_subtext)
                ax_psf.set_ylabel("θy (mrad)", fontsize=8, color=fg_subtext)
                ax_psf.tick_params(colors=fg_subtext, labelsize=7)
                for sp in ax_psf.spines.values():
                    sp.set_color(spine_col)
            else:
                im_psf = ax_psf.imshow(psf_disp, cmap="inferno", origin='lower', aspect='auto', vmin=vmin_psf, vmax=vmax_psf)
                ax_psf.set_xticks([])
                ax_psf.set_yticks([])

            ax_psf.set_title(f"Simulated Far-Field Diffraction / PSF (Binary Aperture){zoom_tag}", fontsize=8.5, color=fg_text, fontweight='bold')
            cbar_psf = self.fig.colorbar(im_psf, ax=ax_psf, fraction=0.046, pad=0.04)
            cbar_psf.set_label(cbar_title, color=fg_subtext, fontsize=7.5)
            cbar_psf.ax.tick_params(colors=fg_subtext, labelsize=7)
            if hasattr(cbar_psf, 'outline') and cbar_psf.outline is not None:
                cbar_psf.outline.set_edgecolor(spine_col)
            self._last_ax_psf = ax_psf

            # 3. 1D Cross-Section (Binary Step Pulses across X)
            mid_row = data_binary.shape[0] // 2
            x_mm = maps["x_mm"]
            bin_1d = data_binary[mid_row, :]
            line1_col = '#fbbf24' if is_dark else '#d97706'
            ax_prof.step(x_mm, bin_1d, where='mid', label="Binary State (0/1)", color=line1_col, linewidth=2.0)
            ax_prof.set_title(f"1D Binary State & Micromirror Tilt Cross-Section [±{tilt_deg:.1f}°]", fontsize=9, color=fg_text, fontweight='bold')
            ax_prof.set_xlabel("X (mm)", fontsize=8, color=fg_subtext)
            ax_prof.set_ylabel("Binary State (0 / 1)", fontsize=8, color=fg_subtext)
            ax_prof.set_ylim(-0.15, 1.15)
            ax_prof.set_yticks([0, 1])
            ax_prof.set_yticklabels(["0 (OFF)", "1 (ON)"])

            # Twin axis for tilt angle in degrees
            ax_prof_twin = ax_prof.twinx()
            twin_col = '#38bdf8' if is_dark else '#0284c7'
            tilt_1d = np.where(bin_1d > 0, tilt_deg, -tilt_deg)
            ax_prof_twin.step(x_mm, tilt_1d, where='mid', color=twin_col, linewidth=1.5, linestyle=':')
            ax_prof_twin.set_ylabel("Mirror Tilt Angle [°]", color=twin_col, fontsize=8)
            ax_prof_twin.set_ylim(-1.3 * tilt_deg, 1.3 * tilt_deg)
            ax_prof_twin.set_yticks([-tilt_deg, 0.0, tilt_deg])
            ax_prof_twin.set_yticklabels([f"-{tilt_deg:.0f}° (OFF)", "0° (FLAT)", f"+{tilt_deg:.0f}° (ON)"])
            ax_prof_twin.tick_params(colors=twin_col, labelsize=7)
            for sp in ax_prof_twin.spines.values():
                sp.set_color(spine_col)

            leg_bg = '#1e293b' if is_dark else '#f8fafc'
            leg_ec = '#334155' if is_dark else '#cbd5e1'
            ax_prof.legend(loc="upper right", fontsize=7, facecolor=leg_bg, edgecolor=leg_ec, labelcolor=fg_text)

            eff_total = float(np.mean(maps["diffraction_efficiency"])) * 100.0
            eff_grat = float(np.mean(maps["grating_efficiency_only"])) * 100.0
            self.lbl_status.setText(
                f"TI {plm_cfg.model} 1-Bit DMD [{cur_st}] | Tilt=±{tilt_deg:.1f}° | "
                f"Array: {plm_cfg.array_cols} × {plm_cfg.array_rows} ({plm_cfg.pixel_pitch_um:.2f} µm pitch) | "
                f"Grating η={eff_grat:.1f}% | Far-Field Transmission={eff_total:.1f}%"
            )

        # --------------------------------------------------------------------
        # BRANCH B: TI-PLM (4-Bit Phase Light Modulator)
        # --------------------------------------------------------------------
        else:
            unit_mode = self.combo_unit.currentText() if hasattr(self, 'combo_unit') else "Phase OPD (µm)"
            if "Piston" in unit_mode:
                data_map = maps["piston_nm"] / 1000.0  # physical piston stroke in um
                unit_label = "Piston Stroke [µm]"
                vmax = wl_um / 2.0
            elif "Angle" in unit_mode:
                data_map = maps["quantized_waves"] * 360.0
                unit_label = "Phase [°]"
                vmax = 360.0
            elif "Waves" in unit_mode:
                data_map = maps["quantized_waves"]
                unit_label = "Phase [Waves λ₀]"
                vmax = 1.0
            else:
                data_map = maps["quantized_waves"] * wl_um
                unit_label = "Phase OPD [µm]"
                vmax = wl_um

            # 1. 2D 4-Bit Quantized Phase Map in µm
            cmap_16 = matplotlib.colormaps.get_cmap("viridis").resampled(16)
            im_map = ax_map.imshow(
                data_map,
                cmap=cmap_16,
                vmin=0.0,
                vmax=vmax,
                extent=[-0.5 * plm_cfg.active_width_mm, 0.5 * plm_cfg.active_width_mm,
                        -0.5 * plm_cfg.active_height_mm, 0.5 * plm_cfg.active_height_mm],
                origin='lower',
                aspect='auto'
            )
            ax_map.set_title(f"TI-PLM 2D Phase Map [{cur_st}] ({unit_label})", fontsize=9, color=fg_text, fontweight='bold')
            ax_map.set_xlabel("X (mm)", fontsize=8, color=fg_subtext)
            ax_map.set_ylabel("Y (mm)", fontsize=8, color=fg_subtext)
            cbar_map = self.fig.colorbar(im_map, ax=ax_map, fraction=0.046, pad=0.04)
            cbar_map.ax.tick_params(colors=fg_subtext, labelsize=7)
            cbar_map.set_label(unit_label, color=fg_subtext, fontsize=8)
            if hasattr(cbar_map, 'outline') and cbar_map.outline is not None:
                cbar_map.outline.set_edgecolor(spine_col)

            if cur_st.upper() in ("OFF", "STATE_0", "FLAT") or np.all(data_map == 0.0):
                banner_fc = "#1e293b" if is_dark else "#fffbeb"
                banner_ec = "#f59e0b" if is_dark else "#b45309"
                banner_txt = "#fbbf24" if is_dark else "#92400e"
                ax_map.text(
                    0.5, 0.5,
                    f"[{cur_st}] UNPOWERED / SPECULAR FLAT STATE\n(0 µm Phase OPD · Actuators at Rest)\nSelect 'ON' or adjust Zernike/Gratings to synthesize phase",
                    ha='center', va='center', transform=ax_map.transAxes,
                    color=banner_txt, fontsize=8.0, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.5", fc=banner_fc, ec=banner_ec, lw=1.2, alpha=0.92)
                )

            # 2. Simulated Far-Field PSF / Diffraction Pattern (2D FFT)
            psf = maps["psf_intensity"]
            ny_p, nx_p = psf.shape
            cy, cx = ny_p // 2, nx_p // 2

            zoom_idx = self.combo_psf_zoom.currentIndex() if hasattr(self, 'combo_psf_zoom') and self.combo_psf_zoom else 0
            ext_full = maps.get("psf_extent_mrad", None)
            if zoom_idx == 1:
                # Wide (Zoom 2×)
                ry = max(10, ny_p // 4)
                rx = max(10, nx_p // 4)
                psf_crop = psf[cy - ry: cy + ry, cx - rx: cx + rx]
                ext = [ext_full[0]/2.0, ext_full[1]/2.0, ext_full[2]/2.0, ext_full[3]/2.0] if ext_full else None
                zoom_tag = " [Wide 2×]"
            elif zoom_idx == 2:
                # Medium (Zoom 4×)
                ry = max(10, ny_p // 8)
                rx = max(10, nx_p // 8)
                psf_crop = psf[cy - ry: cy + ry, cx - rx: cx + rx]
                ext = [ext_full[0]/4.0, ext_full[1]/4.0, ext_full[2]/4.0, ext_full[3]/4.0] if ext_full else None
                zoom_tag = " [Medium 4×]"
            elif zoom_idx == 3:
                # Close-up (Zoom 8×)
                ry = max(10, ny_p // 16)
                rx = max(10, nx_p // 16)
                psf_crop = psf[cy - ry: cy + ry, cx - rx: cx + rx]
                ext = [ext_full[0]/8.0, ext_full[1]/8.0, ext_full[2]/8.0, ext_full[3]/8.0] if ext_full else None
                zoom_tag = " [Close-up 8×]"
            else:
                # Full Field (Zoom Out 1×) - completely uncropped!
                psf_crop = psf
                ext = ext_full
                zoom_tag = " [Zoom Out 1×]"

            use_log = self.chk_psf_log.isChecked() if hasattr(self, 'chk_psf_log') and self.chk_psf_log else True
            if use_log:
                psf_disp = 10.0 * np.log10(np.maximum(psf_crop, 3e-4))
                vmin_psf, vmax_psf = -35.0, 0.0
                cbar_title = "Intensity [dB]"
            else:
                psf_disp = psf_crop
                vmin_psf, vmax_psf = 0.0, 1.0
                cbar_title = "Linear [0..1]"

            if ext is not None:
                im_psf = ax_psf.imshow(psf_disp, cmap="inferno", origin='lower', aspect='auto', extent=ext, vmin=vmin_psf, vmax=vmax_psf)
                ax_psf.set_xlabel("θx (mrad)", fontsize=8, color=fg_subtext)
                ax_psf.set_ylabel("θy (mrad)", fontsize=8, color=fg_subtext)
                ax_psf.tick_params(colors=fg_subtext, labelsize=7)
                for sp in ax_psf.spines.values():
                    sp.set_color(spine_col)
            else:
                im_psf = ax_psf.imshow(psf_disp, cmap="inferno", origin='lower', aspect='auto', vmin=vmin_psf, vmax=vmax_psf)
                ax_psf.set_xticks([])
                ax_psf.set_yticks([])

            ax_psf.set_title(f"Simulated Far-Field Diffraction / PSF{zoom_tag}", fontsize=8.5, color=fg_text, fontweight='bold')
            cbar_psf = self.fig.colorbar(im_psf, ax=ax_psf, fraction=0.046, pad=0.04)
            cbar_psf.set_label(cbar_title, color=fg_subtext, fontsize=7.5)
            cbar_psf.ax.tick_params(colors=fg_subtext, labelsize=7)
            if hasattr(cbar_psf, 'outline') and cbar_psf.outline is not None:
                cbar_psf.outline.set_edgecolor(spine_col)
            self._last_ax_psf = ax_psf

            # 3. 1D Cross-Section (Continuous vs 4-Bit Staircase in µm)
            mid_row = maps["state_4bit"].shape[0] // 2
            x_mm = maps["x_mm"]
            cont_opd_um = maps["continuous_waves"][mid_row, :] * wl_um
            q_opd_um = maps["quantized_waves"][mid_row, :] * wl_um

            line1_col = '#38bdf8' if is_dark else '#0284c7'
            line2_col = '#fbbf24' if is_dark else '#d97706'
            ax_prof.plot(x_mm, cont_opd_um, label="Continuous Phase OPD (µm)", color=line1_col, linestyle='--', linewidth=1.2)
            ax_prof.plot(x_mm, q_opd_um, label="4-Bit Staircase OPD (µm)", color=line2_col, linewidth=2.0)
            ax_prof.set_title("1D Phase Map & 4-Bit Staircase Cross-Section [µm]", fontsize=9, color=fg_text, fontweight='bold')
            ax_prof.set_xlabel("X (mm)", fontsize=8, color=fg_subtext)
            ax_prof.set_ylabel("Phase OPD [µm]", fontsize=8, color=fg_subtext)
            ax_prof.set_ylim(-0.04 * wl_um, 1.08 * wl_um)

            # Twin axis on right for physical piston stroke in um
            ax_prof_twin = ax_prof.twinx()
            twin_col = '#f472b6' if is_dark else '#db2777'
            ax_prof_twin.set_ylabel("Physical Stroke [µm]", color=twin_col, fontsize=8)
            ax_prof_twin.set_ylim(-0.02 * wl_um, 0.54 * wl_um)
            ax_prof_twin.tick_params(colors=twin_col, labelsize=7)
            for sp in ax_prof_twin.spines.values():
                sp.set_color(spine_col)

            leg_bg = '#1e293b' if is_dark else '#f8fafc'
            leg_ec = '#334155' if is_dark else '#cbd5e1'
            ax_prof.legend(loc="upper right", fontsize=7, facecolor=leg_bg, edgecolor=leg_ec, labelcolor=fg_text)

            eff_total = float(np.mean(maps["diffraction_efficiency"])) * 100.0
            eff_grat = float(np.mean(maps["grating_efficiency_only"])) * 100.0
            self.lbl_status.setText(
                f"TI {plm_cfg.model} 4-Bit PLM [{cur_st}] | λ₀={wl_um:.3f} µm | "
                f"1-LSB Phase={wl_um/16.0:.4f} µm (Stroke={wl_um/32.0:.4f} µm) | "
                f"Grating η={eff_grat:.1f}% | Total Throughput={eff_total:.1f}%"
            )

        self.fig.tight_layout()
        if hasattr(self, 'canvas') and self.canvas is not None:
            self.canvas.draw_idle()

    def set_dark_mode(self, enabled: bool) -> None:
        """Switch between dark and light background themes."""
        self.dark_mode = bool(enabled)
        self._update_theme_styles()
        self.update_live_plots()

    def _update_theme_styles(self) -> None:
        """Update styled elements to match the current theme."""
        if not QT_AVAILABLE:
            return
        is_dark = getattr(self, 'dark_mode', False)
        if hasattr(self, 'lbl_status') and self.lbl_status:
            col = '#38bdf8' if is_dark else '#0284c7'
            self.lbl_status.setStyleSheet(f"font-weight: bold; color: {col};")
        if hasattr(self, 'fig') and self.fig:
            self.fig.patch.set_facecolor('#0f172a' if is_dark else '#ffffff')


class PLMStudioDock(QDockWidget if QT_AVAILABLE else object):
    """
    Independent Dockable Tool Window for Texas Instruments Modulator Studio
    (Phase Light Modulator PLM 4-Bit & Digital Micromirror Device DMD 1-Bit).
    Operates independently from the Optical Analysis & Diagnostics window.
    """
    def __init__(self, main_window=None, surface_stack=None, parent=None):
        if QT_AVAILABLE:
            super().__init__("🔲 TI Modulator Studio (PLM / DMD)", parent or main_window)
            self.setObjectName("PLMStudioDock")
            self.setAllowedAreas(Qt.AllDockWidgetAreas)
            self.main_window = main_window or parent
            self.studio = PLMStudioWidget(surface_stack=surface_stack, parent=self)
            self.setWidget(self.studio)
            self.visibilityChanged.connect(self._on_visibility_changed)
        else:
            self.studio = None
            self.main_window = None

    def set_surface_stack(self, stack, force: bool = False):
        if self.studio is not None:
            self.studio.set_surface_stack(stack, force=force)

    def _on_visibility_changed(self, visible: bool):
        if visible and self.studio is not None:
            if self.main_window and hasattr(self.main_window, 'get_current_surface_stack'):
                stack = self.main_window.get_current_surface_stack()
                self.studio.set_surface_stack(stack)
            if hasattr(self.studio, 'canvas') and self.studio.canvas is not None:
                self.studio.canvas.draw_idle()

    def showEvent(self, event):
        super().showEvent(event)
        if self.studio is not None:
            if self.main_window and hasattr(self.main_window, 'get_current_surface_stack'):
                stack = self.main_window.get_current_surface_stack()
                self.studio.set_surface_stack(stack)
            if hasattr(self.studio, 'canvas') and self.studio.canvas is not None:
                self.studio.canvas.draw_idle()

    def set_dark_mode(self, enabled: bool) -> None:
        """Propagate theme switch to inner studio widget."""
        if self.studio is not None and hasattr(self.studio, 'set_dark_mode'):
            self.studio.set_dark_mode(enabled)
