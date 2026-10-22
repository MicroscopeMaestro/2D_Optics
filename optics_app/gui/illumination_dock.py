"""
Zemax OpticStudio Illumination, Mask & Projection System Dock Widget for 2D Optics GUI.
======================================================================================
Interactive GUI dock for designing and analyzing high-throughput energy carrying
optical trains from source/lamp/laser to mask/reticle and image plane.

Theoretical Foundations:
- Herbert Gross (Editor), "Handbook of Optical Systems", Wiley-VCH (Vols 2, 4, 6)
- Coursera / University of Colorado Boulder "Optical Engineering" (Paraxial Layout, Etendue, Kohler Illumination)
- Zemax OpticStudio Sequential Modeling, Pupil Matching Solves, and Radiometric Analysis
"""

import os
import math
import traceback
from typing import Optional, List, Dict, Any, Union
import numpy as np

try:
    from optics_app.gui.qt_compat import (
        QDockWidget, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
        QComboBox, QDoubleSpinBox, QSpinBox, QTableWidget, QTableWidgetItem, QHeaderView,
        Qt, QT_AVAILABLE, QThread, pyqtSignal, QScrollArea, QAbstractItemView, QtWidgets,
        QCheckBox, QFileDialog, QMessageBox, QGroupBox, QTabWidget, QSplitter
    )
    from optics_app.gui.theme import get_theme_colors, apply_axes_theme
    from optics_app.engine.illumination import (
        IlluminationMaskProjectionTrain,
        IlluminationSource,
        SourceType,
        IlluminationMode,
        HomogenizerType,
        HomogenizerConfig,
        MaskType,
        MaskSubsystemConfig,
        CollectorType,
        ProjectionType,
        ProjectionLensConfig,
        EnergyThroughputMetrics,
    )
    from optics_app.engine.lde import SurfaceStack
except ImportError:
    from v2.gui.qt_compat import (
        QDockWidget, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
        QComboBox, QDoubleSpinBox, QSpinBox, QTableWidget, QTableWidgetItem, QHeaderView,
        Qt, QT_AVAILABLE, QThread, pyqtSignal, QScrollArea, QAbstractItemView, QtWidgets,
        QCheckBox, QFileDialog, QMessageBox, QGroupBox, QTabWidget, QSplitter
    )
    from v2.gui.theme import get_theme_colors, apply_axes_theme
    from v2.engine.illumination import (
        IlluminationMaskProjectionTrain,
        IlluminationSource,
        SourceType,
        IlluminationMode,
        HomogenizerType,
        HomogenizerConfig,
        MaskType,
        MaskSubsystemConfig,
        CollectorType,
        ProjectionType,
        ProjectionLensConfig,
        EnergyThroughputMetrics,
    )
    from v2.engine.lde import SurfaceStack

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    MPL_AVAILABLE = True
except Exception:
    MPL_AVAILABLE = False


class IlluminationDockWidget(QDockWidget if QT_AVAILABLE else object):
    """
    Zemax-Style Illumination, Mask & Projection Design & Throughput Dock.
    """

    def __init__(self, model: Optional[Any] = None, parent: Optional[QWidget] = None):
        if not QT_AVAILABLE:
            return
        super().__init__("💡 Illumination & Mask Projection System", parent)
        self.setObjectName("IlluminationDockWidget")
        self.model = model
        self.dark_mode = False
        self._is_updating_ui = False

        # Current optical train
        self.train: IlluminationMaskProjectionTrain = IlluminationMaskProjectionTrain.create_photolithography_stepper_4x()

        self._build_ui()
        self.refresh_ui_from_train()
        self.run_evaluation()

    def _build_ui(self) -> None:
        """Construct the interactive dock interface."""
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(6)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(4, 4, 4, 4)
        scroll_layout.setSpacing(8)

        # ---------------------------------------------------------------------
        # 1. Preset Selector Group
        # ---------------------------------------------------------------------
        preset_group = QGroupBox("System Archetype Preset")
        preset_layout = QHBoxLayout(preset_group)
        preset_layout.setContentsMargins(4, 6, 4, 6)

        self.combo_presets = QComboBox()
        self.combo_presets.addItem("Deep-UV / i-Line 365nm Köhler Stepper (4x)", "stepper_4x")
        self.combo_presets.addItem("Laser 532nm Mask Projection System", "laser_proj")
        self.combo_presets.addItem("High-Power Xenon Arc Lamp & DMD Projector", "lamp_dmd")
        self.combo_presets.addItem("Double-Telecentric 1x Metrology Train", "metrology_1x")
        self.combo_presets.currentIndexChanged.connect(self._on_preset_changed)
        preset_layout.addWidget(self.combo_presets, stretch=3)

        btn_apply_preset = QPushButton("Load Preset")
        btn_apply_preset.clicked.connect(self._on_preset_changed)
        preset_layout.addWidget(btn_apply_preset, stretch=1)
        scroll_layout.addWidget(preset_group)

        # ---------------------------------------------------------------------
        # 2. Source & Collector Configuration
        # ---------------------------------------------------------------------
        source_group = QGroupBox("1. Source & Collector (Lamp / Laser)")
        source_layout = QVBoxLayout(source_group)
        source_layout.setSpacing(4)

        row_s1 = QHBoxLayout()
        row_s1.addWidget(QLabel("Type:"))
        self.combo_source_type = QComboBox()
        for st in SourceType:
            self.combo_source_type.addItem(st.value, st)
        self.combo_source_type.currentIndexChanged.connect(self._update_train_from_ui)
        row_s1.addWidget(self.combo_source_type)

        row_s1.addWidget(QLabel("Power (W):"))
        self.spin_power = QDoubleSpinBox()
        self.spin_power.setRange(0.01, 10000.0)
        self.spin_power.setValue(350.0)
        self.spin_power.valueChanged.connect(self._update_train_from_ui)
        row_s1.addWidget(self.spin_power)
        source_layout.addLayout(row_s1)

        row_s2 = QHBoxLayout()
        row_s2.addWidget(QLabel("Wavelength (µm):"))
        self.spin_wavelength = QDoubleSpinBox()
        self.spin_wavelength.setRange(0.150, 2.500)
        self.spin_wavelength.setDecimals(4)
        self.spin_wavelength.setSingleStep(0.010)
        self.spin_wavelength.setValue(0.365)
        self.spin_wavelength.valueChanged.connect(self._update_train_from_ui)
        row_s2.addWidget(self.spin_wavelength)

        row_s2.addWidget(QLabel("NA / Divergence:"))
        self.spin_source_na = QDoubleSpinBox()
        self.spin_source_na.setRange(0.001, 0.950)
        self.spin_source_na.setDecimals(3)
        self.spin_source_na.setValue(0.650)
        self.spin_source_na.valueChanged.connect(self._update_train_from_ui)
        row_s2.addWidget(self.spin_source_na)
        source_layout.addLayout(row_s2)

        row_s3 = QHBoxLayout()
        row_s3.addWidget(QLabel("Emitter / Waist (mm):"))
        self.spin_emitter_size = QDoubleSpinBox()
        self.spin_emitter_size.setRange(0.01, 50.0)
        self.spin_emitter_size.setValue(1.50)
        self.spin_emitter_size.valueChanged.connect(self._update_train_from_ui)
        row_s3.addWidget(self.spin_emitter_size)

        row_s3.addWidget(QLabel("Collector:"))
        self.combo_collector = QComboBox()
        for ct in CollectorType:
            self.combo_collector.addItem(ct.value, ct)
        self.combo_collector.currentIndexChanged.connect(self._update_train_from_ui)
        row_s3.addWidget(self.combo_collector)
        source_layout.addLayout(row_s3)

        scroll_layout.addWidget(source_group)

        # ---------------------------------------------------------------------
        # 3. Illumination & Homogenizer Configuration
        # ---------------------------------------------------------------------
        illum_group = QGroupBox("2. Illumination & Spatial Homogenizer")
        illum_layout = QVBoxLayout(illum_group)
        illum_layout.setSpacing(4)

        row_i1 = QHBoxLayout()
        row_i1.addWidget(QLabel("Mode:"))
        self.combo_illum_mode = QComboBox()
        for im in IlluminationMode:
            self.combo_illum_mode.addItem(im.value, im)
        self.combo_illum_mode.currentIndexChanged.connect(self._update_train_from_ui)
        row_i1.addWidget(self.combo_illum_mode)

        row_i1.addWidget(QLabel("Homogenizer:"))
        self.combo_homogenizer = QComboBox()
        for ht in HomogenizerType:
            self.combo_homogenizer.addItem(ht.value, ht)
        self.combo_homogenizer.currentIndexChanged.connect(self._update_train_from_ui)
        row_i1.addWidget(self.combo_homogenizer)
        illum_layout.addLayout(row_i1)

        row_i2 = QHBoxLayout()
        row_i2.addWidget(QLabel("Condenser FL (mm):"))
        self.spin_cond_fl = QDoubleSpinBox()
        self.spin_cond_fl.setRange(5.0, 500.0)
        self.spin_cond_fl.setValue(55.0)
        self.spin_cond_fl.valueChanged.connect(self._update_train_from_ui)
        row_i2.addWidget(self.spin_cond_fl)

        row_i2.addWidget(QLabel("Condenser NA:"))
        self.spin_cond_na = QDoubleSpinBox()
        self.spin_cond_na.setRange(0.01, 0.90)
        self.spin_cond_na.setDecimals(4)
        self.spin_cond_na.setValue(0.1125)
        self.spin_cond_na.valueChanged.connect(self._update_train_from_ui)
        row_i2.addWidget(self.spin_cond_na)
        illum_layout.addLayout(row_i2)

        scroll_layout.addWidget(illum_group)

        # ---------------------------------------------------------------------
        # 4. Mask / Reticle Subsystem
        # ---------------------------------------------------------------------
        mask_group = QGroupBox("3. Mask / Reticle Subsystem")
        mask_layout = QVBoxLayout(mask_group)
        mask_layout.setSpacing(4)

        row_m1 = QHBoxLayout()
        row_m1.addWidget(QLabel("Mask Type:"))
        self.combo_mask_type = QComboBox()
        for mt in MaskType:
            self.combo_mask_type.addItem(mt.value, mt)
        self.combo_mask_type.currentIndexChanged.connect(self._update_train_from_ui)
        row_m1.addWidget(self.combo_mask_type)

        row_m1.addWidget(QLabel("Field Size (mm):"))
        self.spin_mask_size = QDoubleSpinBox()
        self.spin_mask_size.setRange(1.0, 200.0)
        self.spin_mask_size.setValue(20.0)
        self.spin_mask_size.valueChanged.connect(self._update_train_from_ui)
        row_m1.addWidget(self.spin_mask_size)
        mask_layout.addLayout(row_m1)

        row_m2 = QHBoxLayout()
        row_m2.addWidget(QLabel("Transmission Factor:"))
        self.spin_mask_trans = QDoubleSpinBox()
        self.spin_mask_trans.setRange(0.01, 1.00)
        self.spin_mask_trans.setSingleStep(0.05)
        self.spin_mask_trans.setValue(0.50)
        self.spin_mask_trans.valueChanged.connect(self._update_train_from_ui)
        row_m2.addWidget(self.spin_mask_trans)
        mask_layout.addLayout(row_m2)

        scroll_layout.addWidget(mask_group)

        # ---------------------------------------------------------------------
        # 5. Projection Lens Subsystem
        # ---------------------------------------------------------------------
        proj_group = QGroupBox("4. Projection Lens Objective")
        proj_layout = QVBoxLayout(proj_group)
        proj_layout.setSpacing(4)

        row_p1 = QHBoxLayout()
        row_p1.addWidget(QLabel("Type:"))
        self.combo_proj_type = QComboBox()
        for pt in ProjectionType:
            self.combo_proj_type.addItem(pt.value, pt)
        self.combo_proj_type.currentIndexChanged.connect(self._update_train_from_ui)
        row_p1.addWidget(self.combo_proj_type)

        row_p1.addWidget(QLabel("Mag (M):"))
        self.spin_proj_mag = QDoubleSpinBox()
        self.spin_proj_mag.setRange(-100.0, 100.0)
        self.spin_proj_mag.setDecimals(3)
        self.spin_proj_mag.setValue(-0.250)
        self.spin_proj_mag.valueChanged.connect(self._update_train_from_ui)
        row_p1.addWidget(self.spin_proj_mag)
        proj_layout.addLayout(row_p1)

        row_p2 = QHBoxLayout()
        row_p2.addWidget(QLabel("Image NA:"))
        self.spin_proj_na = QDoubleSpinBox()
        self.spin_proj_na.setRange(0.01, 0.95)
        self.spin_proj_na.setDecimals(3)
        self.spin_proj_na.setValue(0.450)
        self.spin_proj_na.valueChanged.connect(self._update_train_from_ui)
        row_p2.addWidget(self.spin_proj_na)

        self.chk_telecentric = QCheckBox("Double-Telecentric")
        self.chk_telecentric.setChecked(True)
        self.chk_telecentric.toggled.connect(self._update_train_from_ui)
        row_p2.addWidget(self.chk_telecentric)
        proj_layout.addLayout(row_p2)

        scroll_layout.addWidget(proj_group)

        # ---------------------------------------------------------------------
        # 6. Action Buttons
        # ---------------------------------------------------------------------
        btn_layout = QHBoxLayout()
        self.btn_solve_pupil = QPushButton("⚡ Solve & Conjugate Pupils")
        self.btn_solve_pupil.setStyleSheet("font-weight: bold; background-color: #0284c7; color: white;")
        self.btn_solve_pupil.clicked.connect(self.solve_and_update_pupil)
        btn_layout.addWidget(self.btn_solve_pupil)

        self.btn_eval_throughput = QPushButton("📊 Evaluate Throughput & Uniformity")
        self.btn_eval_throughput.setStyleSheet("font-weight: bold; background-color: #059669; color: white;")
        self.btn_eval_throughput.clicked.connect(self.run_evaluation)
        btn_layout.addWidget(self.btn_eval_throughput)
        scroll_layout.addLayout(btn_layout)

        btn_lde_layout = QHBoxLayout()
        self.btn_push_lde = QPushButton("📥 Push to Active LDE & 2D/3D Tracing")
        self.btn_push_lde.setStyleSheet("font-weight: bold; background-color: #4f46e5; color: white;")
        self.btn_push_lde.clicked.connect(self.push_to_active_lde)
        btn_lde_layout.addWidget(self.btn_push_lde)

        self.btn_export_report = QPushButton("📄 Export Zemax Report (.txt)")
        self.btn_export_report.clicked.connect(self.export_report)
        btn_lde_layout.addWidget(self.btn_export_report)

        self.btn_export_zmx = QPushButton("💾 Export to .ZMX")
        self.btn_export_zmx.clicked.connect(self.export_zmx)
        btn_lde_layout.addWidget(self.btn_export_zmx)
        scroll_layout.addLayout(btn_lde_layout)

        # ---------------------------------------------------------------------
        # 7. Live Performance Metrics Dashboard
        # ---------------------------------------------------------------------
        dash_group = QGroupBox("5. Live Energy Throughput & Radiometric Audit")
        dash_layout = QVBoxLayout(dash_group)

        self.table_metrics = QTableWidget()
        self.table_metrics.setColumnCount(2)
        self.table_metrics.setHorizontalHeaderLabels(["Performance Parameter", "Value & Diagnostics"])
        self.table_metrics.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table_metrics.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table_metrics.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table_metrics.setMinimumHeight(240)
        dash_layout.addWidget(self.table_metrics)

        scroll_layout.addWidget(dash_group)

        # ---------------------------------------------------------------------
        # 8. Visual Plots (Matplotlib Canvas)
        # ---------------------------------------------------------------------
        plot_group = QGroupBox("6. Irradiance Profiles & Spatial Uniformity")
        plot_layout = QVBoxLayout(plot_group)

        if MPL_AVAILABLE:
            self.figure = Figure(figsize=(5, 3.5), dpi=90)
            self.canvas = FigureCanvas(self.figure)
            plot_layout.addWidget(self.canvas)
        else:
            plot_layout.addWidget(QLabel("Matplotlib not available."))

        scroll_layout.addWidget(plot_group)

        scroll.setWidget(scroll_content)
        main_layout.addWidget(scroll)
        self.setWidget(main_widget)

    # -------------------------------------------------------------------------
    # UI <-> Model Synchronization
    # -------------------------------------------------------------------------

    def refresh_ui_from_train(self) -> None:
        """Populate UI widgets from internal train state."""
        self._is_updating_ui = True
        try:
            st_val = getattr(self.train.source.source_type, "value", str(self.train.source.source_type))
            idx = self.combo_source_type.findText(st_val)
            if idx >= 0:
                self.combo_source_type.setCurrentIndex(idx)
            self.spin_power.setValue(self.train.source.optical_power_w)
            self.spin_wavelength.setValue(self.train.source.wavelength_um)
            self.spin_source_na.setValue(self.train.source.numerical_aperture)
            self.spin_emitter_size.setValue(self.train.source.emitter_width_mm)

            ct_val = getattr(self.train.collector_type, "value", str(self.train.collector_type))
            idx = self.combo_collector.findText(ct_val)
            if idx >= 0:
                self.combo_collector.setCurrentIndex(idx)

            im_val = getattr(self.train.illumination_mode, "value", str(self.train.illumination_mode))
            idx = self.combo_illum_mode.findText(im_val)
            if idx >= 0:
                self.combo_illum_mode.setCurrentIndex(idx)

            ht_val = getattr(self.train.homogenizer.homogenizer_type, "value", str(self.train.homogenizer.homogenizer_type))
            idx = self.combo_homogenizer.findText(ht_val)
            if idx >= 0:
                self.combo_homogenizer.setCurrentIndex(idx)

            self.spin_cond_fl.setValue(self.train.condenser_focal_length_mm)
            self.spin_cond_na.setValue(self.train.condenser_numerical_aperture)

            mt_val = getattr(self.train.mask.mask_type, "value", str(self.train.mask.mask_type))
            idx = self.combo_mask_type.findText(mt_val)
            if idx >= 0:
                self.combo_mask_type.setCurrentIndex(idx)
            self.spin_mask_size.setValue(self.train.mask.active_width_mm)
            self.spin_mask_trans.setValue(self.train.mask.transmission_or_reflectivity)

            pt_val = getattr(self.train.projection.projection_type, "value", str(self.train.projection.projection_type))
            idx = self.combo_proj_type.findText(pt_val)
            if idx >= 0:
                self.combo_proj_type.setCurrentIndex(idx)
            self.spin_proj_mag.setValue(self.train.projection.magnification)
            self.spin_proj_na.setValue(self.train.projection.image_numerical_aperture)
            self.chk_telecentric.setChecked(self.train.projection.is_double_telecentric)
        finally:
            self._is_updating_ui = False

    def _update_train_from_ui(self) -> None:
        """Read UI widgets and apply to internal train state."""
        if getattr(self, "_is_updating_ui", False):
            return

        st_data = self.combo_source_type.currentData()
        if st_data:
            try:
                self.train.source.source_type = SourceType(st_data)
            except ValueError:
                self.train.source.source_type = st_data
        self.train.source.optical_power_w = float(self.spin_power.value())
        self.train.source.wavelength_um = float(self.spin_wavelength.value())
        self.train.source.numerical_aperture = float(self.spin_source_na.value())
        self.train.source.emitter_width_mm = float(self.spin_emitter_size.value())
        self.train.source.emitter_height_mm = float(self.spin_emitter_size.value())

        ct_data = self.combo_collector.currentData()
        if ct_data:
            try:
                self.train.collector_type = CollectorType(ct_data)
            except ValueError:
                self.train.collector_type = ct_data

        im_data = self.combo_illum_mode.currentData()
        if im_data:
            try:
                self.train.illumination_mode = IlluminationMode(im_data)
            except ValueError:
                self.train.illumination_mode = im_data

        ht_data = self.combo_homogenizer.currentData()
        if ht_data:
            try:
                self.train.homogenizer.homogenizer_type = HomogenizerType(ht_data)
            except ValueError:
                self.train.homogenizer.homogenizer_type = ht_data

        self.train.condenser_focal_length_mm = float(self.spin_cond_fl.value())
        self.train.condenser_numerical_aperture = float(self.spin_cond_na.value())

        mt_data = self.combo_mask_type.currentData()
        if mt_data:
            try:
                self.train.mask.mask_type = MaskType(mt_data)
            except ValueError:
                self.train.mask.mask_type = mt_data
        self.train.mask.active_width_mm = float(self.spin_mask_size.value())
        self.train.mask.active_height_mm = float(self.spin_mask_size.value())
        self.train.mask.transmission_or_reflectivity = float(self.spin_mask_trans.value())

        pt_data = self.combo_proj_type.currentData()
        if pt_data:
            try:
                self.train.projection.projection_type = ProjectionType(pt_data)
            except ValueError:
                self.train.projection.projection_type = pt_data
        self.train.projection.magnification = float(self.spin_proj_mag.value())
        self.train.projection.image_numerical_aperture = float(self.spin_proj_na.value())
        self.train.projection.is_double_telecentric = bool(self.chk_telecentric.isChecked())
        self.train.projection.design_wavelength_um = float(self.spin_wavelength.value())

    def _on_preset_changed(self) -> None:
        """Handle preset selection change."""
        idx = self.combo_presets.currentIndex()
        if idx == 0:
            self.train = IlluminationMaskProjectionTrain.create_photolithography_stepper_4x()
        elif idx == 1:
            self.train = IlluminationMaskProjectionTrain.create_laser_mask_projection_system()
        elif idx == 2:
            self.train = IlluminationMaskProjectionTrain.create_high_power_lamp_projector()
        elif idx == 3:
            self.train = IlluminationMaskProjectionTrain.create_telecentric_metrology_inspection_system()

        self.refresh_ui_from_train()
        self.run_evaluation()

    # -------------------------------------------------------------------------
    # Actions & Solves
    # -------------------------------------------------------------------------

    def solve_and_update_pupil(self) -> None:
        """Solve pupil matching and update metrics."""
        self._update_train_from_ui()
        res = self.train.solve_pupil_matching()
        self.run_evaluation()
        msg = (
            f"Pupil Matching Solve Complete:\n"
            f"- Pupil Status: {'Conjugated (Zero Vignetting)' if res['is_matched'] else 'Approximated'}\n"
            f"- Axial Mismatch: {res['axial_error_mm']:.2f} mm\n"
            f"- Pupil Diameter Ratio: {res['pupil_diameter_ratio']:.3f}x\n"
            f"- Partial Coherence (sigma): {res['partial_coherence_sigma']:.3f}"
        )
        if QT_AVAILABLE:
            QMessageBox.information(self, "Pupil Matching Solve", msg)

    def run_evaluation(self) -> None:
        """Run radiometric throughput calculation and update dashboard & plots."""
        self._update_train_from_ui()
        m = self.train.evaluate_throughput_and_uniformity()
        self._populate_metrics_table(m)
        self._update_plots()

    def _populate_metrics_table(self, m: EnergyThroughputMetrics) -> None:
        """Populate the live metrics audit table."""
        rows = [
            ("Total Source Power", f"{m.source_power_w:.2f} W"),
            ("Image Delivered Power", f"{m.image_delivered_power_w:.2f} W"),
            ("Total Optical Throughput", f"{m.total_optical_throughput_pct:.2f} %"),
            ("Collection Efficiency", f"{m.collection_efficiency_pct:.1f} %"),
            ("Homogenizer Efficiency", f"{m.homogenizer_efficiency_pct:.1f} %"),
            ("Pupil Matching Transmission", f"{m.pupil_matching_efficiency_pct:.1f} %"),
            ("Limiting Étendue", f"{m.limiting_etendue_mm2_sr:.4e} mm²·sr"),
            ("Helmholtz-Lagrange Invariant", f"{m.helmholtz_lagrange_invariant_mm_rad:.4e} mm·rad"),
            ("Mask Peak Irradiance", f"{m.mask_peak_irradiance_w_per_cm2:.2f} W/cm²"),
            ("Mask RMS Uniformity", f"{m.mask_uniformity_rms_pct:.1f} % (Flat-Top)"),
            ("Image Peak Irradiance", f"{m.image_peak_irradiance_w_per_cm2:.2f} W/cm²"),
            ("Image Exposure Field", f"{m.image_field_width_mm:.2f} x {m.image_field_height_mm:.2f} mm"),
            ("Partial Coherence (σ)", f"{m.partial_coherence_sigma:.3f}"),
            ("Pupil Conjugation", "CONJUGATED (No Vignetting)" if m.is_pupil_matched else "MISMATCHED"),
            ("Rayleigh Resolution (Airy)", f"{m.rayleigh_resolution_um:.3f} µm"),
            ("Lithography Half-Pitch", f"{m.lithography_resolution_um:.3f} µm"),
            ("Depth of Focus (±DOF)", f"{m.depth_of_focus_um:.3f} µm"),
        ]

        self.table_metrics.setRowCount(len(rows))
        for row_idx, (param, val) in enumerate(rows):
            it0 = QTableWidgetItem(param)
            it1 = QTableWidgetItem(val)
            if "CONJUGATED" in val or "Throughput" in param:
                it1.setForeground(Qt.darkGreen if not self.dark_mode else Qt.green)
            self.table_metrics.setItem(row_idx, 0, it0)
            self.table_metrics.setItem(row_idx, 1, it1)

    def _update_plots(self) -> None:
        """Render 1D irradiance cross sections in matplotlib canvas."""
        if not MPL_AVAILABLE or not hasattr(self, "figure"):
            return

        profiles = self.train.generate_irradiance_profiles(grid_size=51)
        colors = get_theme_colors(self.dark_mode)

        self.figure.clear()
        ax1 = self.figure.add_subplot(121)
        ax2 = self.figure.add_subplot(122)

        # Plot 1: Mask Plane Irradiance
        x_m = profiles["mask"]["x_mm"]
        y_m = profiles["mask"]["slice_1d"]
        ax1.plot(x_m, y_m, color="#0284c7", lw=2, label="Mask Profile")
        ax1.axhline(profiles["mask"]["mean_irradiance"], color="#f59e0b", ls="--", label="Mean")
        ax1.set_title(f"Mask Irradiance (U={profiles['mask']['uniformity_pct']:.1f}%)", fontsize=9, color=colors["text"])
        ax1.set_xlabel("Field Coordinate (mm)", fontsize=8, color=colors["text"])
        ax1.set_ylabel("Irradiance (W/cm²)", fontsize=8, color=colors["text"])
        ax1.grid(True, ls=":", alpha=0.5, color=colors.get("grid_col", colors.get("grid", "#334155")))
        ax1.tick_params(colors=colors["text"], labelsize=7)

        # Plot 2: Image Plane Irradiance
        x_i = profiles["image"]["x_mm"]
        y_i = profiles["image"]["slice_1d"]
        ax2.plot(x_i, y_i, color="#10b981", lw=2, label="Image Profile")
        ax2.axhline(profiles["image"]["mean_irradiance"], color="#ef4444", ls="--", label="Mean")
        ax2.set_title(f"Image Irradiance (U={profiles['image']['uniformity_pct']:.1f}%)", fontsize=9, color=colors["text"])
        ax2.set_xlabel("Wafer/Sensor (mm)", fontsize=8, color=colors["text"])
        ax2.set_ylabel("Irradiance (W/cm²)", fontsize=8, color=colors["text"])
        ax2.grid(True, ls=":", alpha=0.5, color=colors.get("grid_col", colors.get("grid", "#334155")))
        ax2.tick_params(colors=colors["text"], labelsize=7)

        apply_axes_theme(ax1, self.dark_mode)
        apply_axes_theme(ax2, self.dark_mode)
        self.figure.tight_layout()
        self.canvas.draw_idle()

    # -------------------------------------------------------------------------
    # Application Integration & Export
    # -------------------------------------------------------------------------

    def push_to_active_lde(self) -> None:
        """
        Convert complete multi-lens illumination, mask, and projection train
        into SurfaceStack and inject into the active Lens Data Editor.
        """
        self._update_train_from_ui()
        stack = self.train.to_surface_stack()

        parent_win = self.parent()
        target_model = None
        if hasattr(parent_win, "model"):
            target_model = parent_win.model
        elif self.model is not None:
            target_model = self.model

        if target_model is not None:
            if hasattr(target_model, "set_surface_stack"):
                target_model.set_surface_stack(stack)
            elif hasattr(target_model, "set_system"):
                target_model.set_system(stack)

        if hasattr(parent_win, "statusBar") and parent_win.statusBar():
            parent_win.statusBar().showMessage(
                f"Successfully loaded '{self.train.name}' ({len(stack.surfaces)} surfaces) into LDE & Visualizers.", 4000
            )

        if QT_AVAILABLE and self.isVisible():
            QMessageBox.information(
                self,
                "Illumination Train Loaded",
                f"Successfully injected complete optical train into LDE:\n"
                f"- Name: {self.train.name}\n"
                f"- Total Surfaces: {len(stack.surfaces)}\n"
                f"- Primary Wavelength: {self.train.source.wavelength_um * 1000:.1f} nm\n"
                f"- Active Tracing: 2D Dual View and 3D Visualizer updated!"
            )

    def export_report(self) -> None:
        """Export formatted Zemax ASCII report to file."""
        report_text = self.train.generate_radiometric_report()
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export Radiometric Report", f"{self.train.name}_report.txt", "Text Files (*.txt);;All Files (*)"
        )
        if filepath:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(report_text)
            QMessageBox.information(self, "Report Exported", f"Saved report to:\n{filepath}")

    def export_zmx(self) -> None:
        """Export system to Zemax .ZMX file."""
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export to Zemax .ZMX", f"{self.train.name}.zmx", "Zemax Files (*.zmx);;All Files (*)"
        )
        if filepath:
            self.train.export_to_zmx(filepath)
            QMessageBox.information(self, "ZMX Exported", f"Saved Zemax prescription to:\n{filepath}")

    def set_dark_mode(self, enabled: bool) -> None:
        """Update theme styling for dark mode."""
        self.dark_mode = enabled
        self._update_plots()
