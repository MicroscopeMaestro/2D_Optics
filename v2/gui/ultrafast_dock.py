import os
import sys
import math
from typing import Optional, Dict, Any, List

import numpy as np

from optics_app.gui.qt_compat import (
    QWidget, QVBoxLayout, QHBoxLayout, QDockWidget, QTableWidget, QTableWidgetItem,
    QPushButton, QLabel, QHeaderView, QAbstractItemView, QTabWidget, QComboBox,
    QDoubleSpinBox, QFormLayout, QGroupBox, QGridLayout, QScrollArea, QCheckBox,
    QFileDialog, QT_AVAILABLE
)
from optics_app.analysis.pulse_width import PulseWidthAnalysis, PulseAnalysisConfig
from optics_app.analysis.ultrafast_nonlinear import (
    AOMMaterial,
    AcoustoOpticModulator,
    EOMMaterial,
    ElectroOpticModulator,
    NonlinearModality,
    LaserSourceParams,
    NonlinearSampleParams,
    ObjectiveFocusParams,
    NonlinearModalityResult,
    UltrafastNonlinearSimulator,
    DispersiveGlassMaterial,
    glass_gvd_fs2_per_mm,
    SpectralFocusingConfig,
    SpectralFocusingResult,
    SpectralFocusingSRS,
    NonlinearSignalTemporalProbe,
    SpatialBeamDefects,
    InteractionPlaneConfig,
    THORLABS_ULTRAFAST_CATALOG,
    delay_from_stage_displacement_mm,
    stage_displacement_from_delay_ps,
    OilSampleType,
    OilRamanStandard,
    SRSPhotodiodeModel,
    SRSPhotodiodeSignal,
    SRSScanFieldResult,
    PostSampleSRSProbeResult,
    ThorlabsSRSMicroscopeConfig,
    ThorlabsSRSMicroscopeSimulator,
    build_thorlabs_lsm_template,
    build_thorlabs_lsm_system,
    save_thorlabs_lsm_json,
    load_thorlabs_lsm_json,
    build_srs_microscope_template,
    build_srs_microscope_relayed_template,
    build_edmund_optics_srs_template,
    build_olympus_high_na_srs_template,
    build_srs_microscope_system,
    save_srs_microscope_json,
    load_srs_microscope_json,
    SRSSetupBuilderTool,
    SRSGalvoFieldDiagnosticTool,
    SRSSpectralFocusingTool,
    SRSSignalLockInTool,
    SRSBenchAlignmentTool,
    SRSSimulationTools,
    NonSequentialSRSMicroscopeBench,
    build_non_sequential_srs_system,
)
from optics_app.engine.tracer import OpticalSystem


class UltrafastAnalysisDock(QDockWidget):
    """
    Comprehensive Dockable Panel for Ultrafast Optics & Multimodal Nonlinear Imaging.
    
    Includes:
    1. Surface-by-surface dispersion tracking & pulse broadening (GD, GDD, TOD, FWHM).
    2. Multimodal Nonlinear Imaging (2P, 3P, SHG, CARS, SRS) with focal field & SNR metrics.
    3. Acousto-Optic (AOM) and Electro-Optic (EOM / Pockels cell) device modeling.
    4. Spectral Focusing SRS (Chirped pulse hyperspectral Raman spectroscopy).
    5. Virtual Detector Temporal Signal Probe (Femtosecond pulse overlap & nanosecond pulse trains).
    """
    def __init__(self, main_window=None, parent=None):
        super().__init__("⚡ Ultrafast & Nonlinear Analysis", parent)
        self.setObjectName("UltrafastAnalysisDock")
        self.main_window = main_window
        self.system: Optional[OpticalSystem] = None
        self._last_pulse_result = None
        self._last_nonlinear_result = None
        self._last_spectral_focusing_result = None
        self._last_probe_result = None
        self._last_srs_microscope_result = None
        self.srs_tools = SRSSimulationTools
        self.setup_builder_tool = SRSSetupBuilderTool
        self.galvo_diagnostic_tool = SRSGalvoFieldDiagnosticTool
        self.spectral_focusing_tool = SRSSpectralFocusingTool
        self.signal_lockin_tool = SRSSignalLockInTool
        self.bench_alignment_tool = SRSBenchAlignmentTool
        self.non_sequential_bench = NonSequentialSRSMicroscopeBench
        self._last_non_sequential_srs_result = None
        self.dark_mode: bool = False
        self._init_ui()

    def _init_ui(self):
        container = QWidget(self)
        root_layout = QVBoxLayout(container)
        root_layout.setContentsMargins(4, 4, 4, 4)
        root_layout.setSpacing(4)

        # Quick Module Navigator Header
        self.nav_container = QWidget(self)
        self.nav_container.setStyleSheet("background-color: #ECEFF1; border-radius: 4px; padding: 3px;")
        nav_container = self.nav_container
        nav_layout = QHBoxLayout(nav_container)
        nav_layout.setContentsMargins(6, 4, 6, 4)
        nav_layout.setSpacing(6)

        lbl_module = QLabel("<b>📑 Tab / Module:</b>", nav_container)
        nav_layout.addWidget(lbl_module)

        self.combo_module_nav = QComboBox(nav_container)
        self.combo_module_nav.addItems([
            "Tab 1: ⚡ Dispersion & Pulse Broadening",
            "Tab 2: 🔬 Multimodal Nonlinear Imaging",
            "Tab 3: 🎛️ AOM & EOM Modulators",
            "Tab 4: 🌈 Spectral Focusing SRS",
            "Tab 5: ⏱️ Temporal Signal Probe",
            "Tab 6: 🔬 Thorlabs SRS Microscope",
        ])
        self.combo_module_nav.currentIndexChanged.connect(self.switch_to_tab)
        nav_layout.addWidget(self.combo_module_nav, stretch=1)

        self.btn_go_tab6 = QPushButton("🔬 Go to Tab 6 (Thorlabs SRS)", nav_container)
        self.btn_go_tab6.setToolTip("Immediately jump to Tab 6: Thorlabs SRS Microscope Simulation & Preset Loader")
        self.btn_go_tab6.setStyleSheet(
            "background-color: #2E7D32; color: white; font-weight: bold; "
            "padding: 5px 12px; border-radius: 4px;"
        )
        self.btn_go_tab6.clicked.connect(lambda: self.switch_to_tab(5))
        nav_layout.addWidget(self.btn_go_tab6)

        root_layout.addWidget(nav_container)

        # Tab Widget
        self.tabs = QTabWidget(self)
        if QT_AVAILABLE:
            self.tabs.setUsesScrollButtons(True)
        self.tabs.currentChanged.connect(self._sync_module_nav_combo)
        root_layout.addWidget(self.tabs)

        # ----------------------------------------------------------------------
        # Tab 1: Surface Dispersion & Pulse Broadening
        # ----------------------------------------------------------------------
        tab_dispersion = QWidget()
        disp_layout = QVBoxLayout(tab_dispersion)
        disp_layout.setContentsMargins(6, 6, 6, 6)
        disp_layout.setSpacing(6)

        self.lbl_status = QLabel("Ready: Run analysis to trace pulse evolution through surfaces.")
        if QT_AVAILABLE:
            self.lbl_status.setStyleSheet(
                "background-color: #F3E5F5; color: #4A148C; font-weight: bold; "
                "padding: 5px 8px; border-radius: 4px; border: 1px solid #CE93D8;"
            )
        self.lbl_status.setWordWrap(True)
        disp_layout.addWidget(self.lbl_status)

        # Toolbar
        toolbar = QHBoxLayout()
        self.btn_analyze = QPushButton("🔄 Run Ultrafast Trace")
        self.btn_analyze.setToolTip("Trace femtosecond pulse and compute dispersion / pulse duration at each surface")
        self.btn_analyze.clicked.connect(self.run_analysis)
        toolbar.addWidget(self.btn_analyze)

        self.lbl_prechirp = QLabel("Pre-Chirp Needed: 0 fs²")
        self.lbl_prechirp.setStyleSheet("font-weight: bold; color: #1E88E5;")
        toolbar.addWidget(self.lbl_prechirp)
        toolbar.addStretch()
        disp_layout.addLayout(toolbar)

        # Dispersion Table
        self.table = QTableWidget(self)
        self.table.setColumnCount(8)
        self.table.setHorizontalHeaderLabels([
            "Surf #", "Name", "GD (fs)", "Mat GDD (fs²)", "Geo GDD (fs²)", 
            "Tot GDD (fs²)", "FWHM (fs)", "Spatial Chirp (mm)"
        ])
        if QT_AVAILABLE:
            self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
            self.table.setSelectionMode(QAbstractItemView.SingleSelection)
            self.table.horizontalHeader().setStretchLastSection(True)
            self.table.verticalHeader().setVisible(False)
        disp_layout.addWidget(self.table)
        self.tabs.addTab(tab_dispersion, "Tab 1: ⚡ Dispersion")
        if QT_AVAILABLE:
            self.tabs.setTabToolTip(0, "Surface Dispersion & Pulse Broadening Analysis")

        # ----------------------------------------------------------------------
        # Tab 2: Multimodal Nonlinear Imaging (2P, 3P, SHG, CARS, SRS)
        # ----------------------------------------------------------------------
        tab_nonlinear = QWidget()
        nl_scroll = QScrollArea()
        nl_scroll.setWidgetResizable(True)
        nl_inner = QWidget()
        nl_layout = QVBoxLayout(nl_inner)
        nl_layout.setContentsMargins(6, 6, 6, 6)
        nl_layout.setSpacing(6)

        # Modality selector
        grp_modality = QGroupBox("Modality & Excitation Parameters")
        form_mod = QFormLayout(grp_modality)

        self.combo_modality = QComboBox()
        self.combo_modality.addItems([
            "2P — Two-Photon Fluorescence",
            "3P — Three-Photon Fluorescence",
            "SHG — Second Harmonic Generation",
            "CARS — Coherent Anti-Stokes Raman",
            "SRS — Stimulated Raman Scattering"
        ])
        self.combo_modality.currentIndexChanged.connect(self._on_modality_changed)
        form_mod.addRow("Imaging Modality:", self.combo_modality)

        self.spin_laser_power_mw = QDoubleSpinBox()
        self.spin_laser_power_mw.setRange(0.1, 5000.0)
        self.spin_laser_power_mw.setValue(20.0)
        self.spin_laser_power_mw.setSuffix(" mW")
        form_mod.addRow("Average Power:", self.spin_laser_power_mw)

        self.spin_laser_wl_nm = QDoubleSpinBox()
        self.spin_laser_wl_nm.setRange(300.0, 2500.0)
        self.spin_laser_wl_nm.setValue(800.0)
        self.spin_laser_wl_nm.setSuffix(" nm")
        form_mod.addRow("Center Wavelength:", self.spin_laser_wl_nm)

        self.spin_pulse_duration_fs = QDoubleSpinBox()
        self.spin_pulse_duration_fs.setRange(5.0, 10000.0)
        self.spin_pulse_duration_fs.setValue(100.0)
        self.spin_pulse_duration_fs.setSuffix(" fs")
        form_mod.addRow("Pulse Duration (FWHM):", self.spin_pulse_duration_fs)

        self.spin_rep_rate_mhz = QDoubleSpinBox()
        self.spin_rep_rate_mhz.setRange(0.01, 1000.0)
        self.spin_rep_rate_mhz.setValue(80.0)
        self.spin_rep_rate_mhz.setSuffix(" MHz")
        form_mod.addRow("Repetition Rate:", self.spin_rep_rate_mhz)

        self.spin_na = QDoubleSpinBox()
        self.spin_na.setRange(0.05, 1.49)
        self.spin_na.setValue(0.80)
        self.spin_na.setSingleStep(0.05)
        form_mod.addRow("Objective NA:", self.spin_na)

        self.spin_stokes_wl_nm = QDoubleSpinBox()
        self.spin_stokes_wl_nm.setRange(400.0, 2500.0)
        self.spin_stokes_wl_nm.setValue(1040.0)
        self.spin_stokes_wl_nm.setSuffix(" nm")
        self.spin_stokes_wl_nm.setEnabled(False)
        form_mod.addRow("Stokes Wavelength (CARS/SRS):", self.spin_stokes_wl_nm)
        nl_layout.addWidget(grp_modality)

        # Sample and Modulator Coupling
        grp_sample = QGroupBox("Sample & Interaction Properties")
        form_sample = QFormLayout(grp_sample)

        self.spin_conc_um = QDoubleSpinBox()
        self.spin_conc_um.setRange(0.001, 1000.0)
        self.spin_conc_um.setValue(10.0)
        self.spin_conc_um.setSuffix(" µM")
        form_sample.addRow("Fluorophore Concentration:", self.spin_conc_um)

        self.spin_sigma_2p = QDoubleSpinBox()
        self.spin_sigma_2p.setRange(0.01, 10000.0)
        self.spin_sigma_2p.setValue(36.0)
        self.spin_sigma_2p.setSuffix(" GM")
        form_sample.addRow("2P Cross Section:", self.spin_sigma_2p)

        self.spin_tissue_depth_um = QDoubleSpinBox()
        self.spin_tissue_depth_um.setRange(0.0, 2000.0)
        self.spin_tissue_depth_um.setValue(0.0)
        self.spin_tissue_depth_um.setSuffix(" µm")
        form_sample.addRow("Tissue Depth (3P attenuation):", self.spin_tissue_depth_um)

        self.spin_raman_shift = QDoubleSpinBox()
        self.spin_raman_shift.setRange(100.0, 4000.0)
        self.spin_raman_shift.setValue(2845.0)
        self.spin_raman_shift.setSuffix(" cm⁻¹")
        form_sample.addRow("Raman Mode (CH₂ stretch):", self.spin_raman_shift)

        self.cb_apply_aom = QCheckBox("Pass beam through AOM (power attenuation & crystal GDD)")
        self.cb_apply_eom = QCheckBox("Pass beam through EOM (polarization modulation & GDD)")
        form_sample.addRow(self.cb_apply_aom)
        form_sample.addRow(self.cb_apply_eom)
        nl_layout.addWidget(grp_sample)

        # Interaction Plane Flexibility (Image Plane or User-defined Intermediate Plane)
        grp_plane = QGroupBox("Sample Interaction Plane & Axial Defocus")
        form_plane = QFormLayout(grp_plane)

        self.combo_interaction_plane = QComboBox()
        self.combo_interaction_plane.addItems([
            "Image Plane (Nominal Focus, z = 0 µm)",
            "User-Defined Intermediate Plane (Defocus / Surface)",
        ])
        form_plane.addRow("Interaction Plane:", self.combo_interaction_plane)

        self.spin_defocus_um = QDoubleSpinBox()
        self.spin_defocus_um.setRange(-2000.0, 2000.0)
        self.spin_defocus_um.setValue(0.0)
        self.spin_defocus_um.setSingleStep(1.0)
        self.spin_defocus_um.setSuffix(" µm")
        form_plane.addRow("Axial Defocus (Δz):", self.spin_defocus_um)
        nl_layout.addWidget(grp_plane)

        # Spatial Defects & Beam Distortions
        grp_defects = QGroupBox("Spatial Defects & Beam Distortions")
        form_defects = QFormLayout(grp_defects)

        self.cb_enable_defects = QCheckBox("Model Spatiotemporal Couplings & Wavefront Aberrations")
        form_defects.addRow(self.cb_enable_defects)

        self.spin_spatial_chirp = QDoubleSpinBox()
        self.spin_spatial_chirp.setRange(0.0, 2.0)
        self.spin_spatial_chirp.setValue(0.0)
        self.spin_spatial_chirp.setSingleStep(0.01)
        self.spin_spatial_chirp.setSuffix(" mm/nm")
        form_defects.addRow("Spatial Chirp (dx/dλ):", self.spin_spatial_chirp)

        self.spin_angular_disp = QDoubleSpinBox()
        self.spin_angular_disp.setRange(0.0, 10.0)
        self.spin_angular_disp.setValue(0.0)
        self.spin_angular_disp.setSingleStep(0.05)
        self.spin_angular_disp.setSuffix(" mrad/nm")
        form_defects.addRow("Angular Dispersion (dθ/dλ):", self.spin_angular_disp)

        self.spin_wavefront_rms = QDoubleSpinBox()
        self.spin_wavefront_rms.setRange(0.0, 2.0)
        self.spin_wavefront_rms.setValue(0.0)
        self.spin_wavefront_rms.setSingleStep(0.02)
        self.spin_wavefront_rms.setSuffix(" waves (λ)")
        form_defects.addRow("Wavefront RMS Error (σ):", self.spin_wavefront_rms)

        self.spin_ellipticity = QDoubleSpinBox()
        self.spin_ellipticity.setRange(0.2, 5.0)
        self.spin_ellipticity.setValue(1.0)
        self.spin_ellipticity.setSingleStep(0.05)
        form_defects.addRow("Beam Ellipticity (w_y / w_x):", self.spin_ellipticity)

        self.lbl_defects_status = QLabel("Strehl Ratio: 1.000 | Pulse Front Tilt: 0.0 mrad | Intensity Factor: 1.000")
        self.lbl_defects_status.setStyleSheet("font-weight: bold; color: #7B1FA2;")
        form_defects.addRow("Defects Impact:", self.lbl_defects_status)
        nl_layout.addWidget(grp_defects)

        # Simulation Button & Results Table
        self.btn_run_nonlinear = QPushButton("🔬 Simulate Nonlinear Modality")
        self.btn_run_nonlinear.setStyleSheet("background-color: #00796B; color: white; font-weight: bold; padding: 6px;")
        self.btn_run_nonlinear.clicked.connect(self.run_nonlinear_simulation)
        nl_layout.addWidget(self.btn_run_nonlinear)

        grp_results = QGroupBox("Focal Diagnostics & Signal-to-Noise Ratio (SNR)")
        res_layout = QVBoxLayout(grp_results)

        self.tbl_nonlinear_results = QTableWidget()
        self.tbl_nonlinear_results.setColumnCount(2)
        self.tbl_nonlinear_results.setHorizontalHeaderLabels(["Metric", "Calculated Value"])
        if QT_AVAILABLE:
            self.tbl_nonlinear_results.horizontalHeader().setStretchLastSection(True)
            self.tbl_nonlinear_results.verticalHeader().setVisible(False)
        res_layout.addWidget(self.tbl_nonlinear_results)
        nl_layout.addWidget(grp_results)

        nl_scroll.setWidget(nl_inner)
        tab_nonlinear_layout = QVBoxLayout(tab_nonlinear)
        tab_nonlinear_layout.setContentsMargins(0, 0, 0, 0)
        tab_nonlinear_layout.addWidget(nl_scroll)
        self.tabs.addTab(tab_nonlinear, "Tab 2: 🔬 Nonlinear")
        if QT_AVAILABLE:
            self.tabs.setTabToolTip(1, "Multimodal Nonlinear Imaging (2P, 3P, SHG, CARS, SRS)")

        # ----------------------------------------------------------------------
        # Tab 3: Modulators (AOM & EOM)
        # ----------------------------------------------------------------------
        tab_modulators = QWidget()
        mod_scroll = QScrollArea()
        mod_scroll.setWidgetResizable(True)
        mod_inner = QWidget()
        mod_layout = QVBoxLayout(mod_inner)
        mod_layout.setContentsMargins(6, 6, 6, 6)
        mod_layout.setSpacing(6)

        # AOM Group
        grp_aom = QGroupBox("Acousto-Optic Modulator (AOM / Bragg Cell)")
        form_aom = QFormLayout(grp_aom)

        self.combo_aom_mat = QComboBox()
        self.combo_aom_mat.addItems(["TeO2 Slow Shear (617 m/s)", "TeO2 Longitudinal (4200 m/s)", "Fused Silica (5960 m/s)", "Quartz (5740 m/s)"])
        form_aom.addRow("AOM Crystal:", self.combo_aom_mat)

        self.spin_aom_rf_freq = QDoubleSpinBox()
        self.spin_aom_rf_freq.setRange(10.0, 500.0)
        self.spin_aom_rf_freq.setValue(80.0)
        self.spin_aom_rf_freq.setSuffix(" MHz")
        form_aom.addRow("RF Carrier Frequency:", self.spin_aom_rf_freq)

        self.spin_aom_rf_pwr = QDoubleSpinBox()
        self.spin_aom_rf_pwr.setRange(0.01, 10.0)
        self.spin_aom_rf_pwr.setValue(1.0)
        self.spin_aom_rf_pwr.setSuffix(" W")
        form_aom.addRow("RF Acoustic Power:", self.spin_aom_rf_pwr)

        self.spin_aom_waist = QDoubleSpinBox()
        self.spin_aom_waist.setRange(0.05, 5.0)
        self.spin_aom_waist.setValue(0.5)
        self.spin_aom_waist.setSuffix(" mm")
        form_aom.addRow("Beam Waist (w₀):", self.spin_aom_waist)

        self.lbl_aom_results = QLabel("Bragg Angle: -- | 1st Order Eff: -- | Rise Time: -- | GDD: --")
        self.lbl_aom_results.setStyleSheet("font-weight: bold; color: #2E7D32;")
        self.lbl_aom_results.setWordWrap(True)
        form_aom.addRow("AOM Outputs:", self.lbl_aom_results)
        mod_layout.addWidget(grp_aom)

        # EOM Group
        grp_eom = QGroupBox("Electro-Optic Modulator (EOM / Pockels Cell)")
        form_eom = QFormLayout(grp_eom)

        self.combo_eom_mat = QComboBox()
        self.combo_eom_mat.addItems(["RTP (Rubidium Titanyl Phosphate)", "KD*P (Potassium Dideuterium Phosphate)", "LiNbO3 (Lithium Niobate)"])
        form_eom.addRow("EOM Crystal:", self.combo_eom_mat)

        self.spin_eom_vpi = QDoubleSpinBox()
        self.spin_eom_vpi.setRange(10.0, 5000.0)
        self.spin_eom_vpi.setValue(240.0)
        self.spin_eom_vpi.setSuffix(" V")
        form_eom.addRow("Half-Wave Voltage (V_π):", self.spin_eom_vpi)

        self.spin_eom_applied_v = QDoubleSpinBox()
        self.spin_eom_applied_v.setRange(0.0, 5000.0)
        self.spin_eom_applied_v.setValue(120.0)
        self.spin_eom_applied_v.setSuffix(" V")
        form_eom.addRow("Applied / Bias Voltage:", self.spin_eom_applied_v)

        self.spin_eom_mod_freq = QDoubleSpinBox()
        self.spin_eom_mod_freq.setRange(0.1, 100.0)
        self.spin_eom_mod_freq.setValue(10.0)
        self.spin_eom_mod_freq.setSuffix(" MHz")
        form_eom.addRow("Lock-In Modulation Freq:", self.spin_eom_mod_freq)

        self.lbl_eom_results = QLabel("Retardation: -- | PBS Transmission: -- | GDD: --")
        self.lbl_eom_results.setStyleSheet("font-weight: bold; color: #1565C0;")
        self.lbl_eom_results.setWordWrap(True)
        form_eom.addRow("EOM Outputs:", self.lbl_eom_results)
        mod_layout.addWidget(grp_eom)

        # Modulator Refresh Button
        self.btn_calc_modulators = QPushButton("⚙ Recalculate Modulators")
        self.btn_calc_modulators.clicked.connect(self.update_modulator_metrics)
        mod_layout.addWidget(self.btn_calc_modulators)

        mod_scroll.setWidget(mod_inner)
        tab_mod_layout = QVBoxLayout(tab_modulators)
        tab_mod_layout.setContentsMargins(0, 0, 0, 0)
        tab_mod_layout.addWidget(mod_scroll)
        self.tabs.addTab(tab_modulators, "Tab 3: 🎛️ Modulators")
        if QT_AVAILABLE:
            self.tabs.setTabToolTip(2, "AOM & EOM Modulators (Depth, Extinction, Demodulation)")

        # ----------------------------------------------------------------------
        # Tab 4: Spectral Focusing SRS (Chirped Pulses & Delay Scanning)
        # ----------------------------------------------------------------------
        tab_sf = QWidget()
        sf_scroll = QScrollArea()
        sf_scroll.setWidgetResizable(True)
        sf_inner = QWidget()
        sf_layout = QVBoxLayout(sf_inner)
        sf_layout.setContentsMargins(6, 6, 6, 6)
        sf_layout.setSpacing(6)

        grp_sf_glass = QGroupBox("Dispersive Glass Rods (Pulse Stretching & Linear Chirping)")
        form_sf_glass = QFormLayout(grp_sf_glass)

        self.combo_pump_glass = QComboBox()
        self.combo_pump_glass.addItems(["SF57 (Thorlabs WG11100-SF57 / WG11150-SF57)", "SF11 (Thorlabs WG11100-SF11)", "Fused Silica", "N-BK7"])
        form_sf_glass.addRow("Pump Glass Rod Material:", self.combo_pump_glass)

        self.spin_pump_rod_len = QDoubleSpinBox()
        self.spin_pump_rod_len.setRange(10.0, 1000.0)
        self.spin_pump_rod_len.setValue(150.0)
        self.spin_pump_rod_len.setSuffix(" mm")
        form_sf_glass.addRow("Pump Glass Rod Length:", self.spin_pump_rod_len)

        self.combo_stokes_glass = QComboBox()
        self.combo_stokes_glass.addItems(["SF57 (Thorlabs WG11100-SF57 / WG11150-SF57)", "SF11 (Thorlabs WG11100-SF11)", "Fused Silica", "N-BK7"])
        form_sf_glass.addRow("Stokes Glass Rod Material:", self.combo_stokes_glass)

        self.spin_stokes_rod_len = QDoubleSpinBox()
        self.spin_stokes_rod_len.setRange(10.0, 1000.0)
        self.spin_stokes_rod_len.setValue(280.0)
        self.spin_stokes_rod_len.setSuffix(" mm")
        form_sf_glass.addRow("Stokes Glass Rod Length:", self.spin_stokes_rod_len)
        sf_layout.addWidget(grp_sf_glass)

        grp_sf_delay = QGroupBox("Thorlabs Optical Delay Line (Inter-pulse Delay Scanning)")
        form_sf_delay = QFormLayout(grp_sf_delay)

        self.combo_delay_stage = QComboBox()
        self.combo_delay_stage.addItems([
            "ODL220 (220 mm Travel / 1467.6 ps Delay, Double-Pass)",
            "ODL100 (100 mm Travel / 667.1 ps Delay, Double-Pass)",
            "DLM100 (100 mm Direct-Drive Linear Motor / 667.1 ps)",
        ])
        form_sf_delay.addRow("Thorlabs Delay Stage:", self.combo_delay_stage)

        self.spin_delay_scan_ps = QDoubleSpinBox()
        self.spin_delay_scan_ps.setRange(0.1, 50.0)
        self.spin_delay_scan_ps.setValue(3.0)
        self.spin_delay_scan_ps.setSuffix(" ps (±)")
        form_sf_delay.addRow("Delay Scan Range (±τ):", self.spin_delay_scan_ps)

        self.lbl_stage_displacement = QLabel("Stage Travel Required: ±0.450 mm")
        self.lbl_stage_displacement.setStyleSheet("font-weight: bold; color: #E65100;")
        form_sf_delay.addRow("Physical Travel (Δx):", self.lbl_stage_displacement)
        sf_layout.addWidget(grp_sf_delay)

        self.btn_run_spectral_focusing = QPushButton("🚀 Simulate Spectral Focusing SRS Spectrum")
        self.btn_run_spectral_focusing.setStyleSheet("background-color: #E65100; color: white; font-weight: bold; padding: 6px;")
        self.btn_run_spectral_focusing.clicked.connect(self.run_spectral_focusing_simulation)
        sf_layout.addWidget(self.btn_run_spectral_focusing)

        grp_sf_results = QGroupBox("Spectral Focusing Diagnostics & Reconstructed Raman Spectrum")
        v_sf_res = QVBoxLayout(grp_sf_results)

        self.lbl_sf_summary = QLabel("Chirp Match Ratio: -- | Resolution: -- cm⁻¹ | Pump FWHM: -- ps | Stokes FWHM: -- ps")
        self.lbl_sf_summary.setStyleSheet("font-weight: bold; color: #BF360C;")
        v_sf_res.addWidget(self.lbl_sf_summary)

        self.tbl_sf_spectrum = QTableWidget()
        self.tbl_sf_spectrum.setColumnCount(4)
        self.tbl_sf_spectrum.setHorizontalHeaderLabels(["Delay (ps)", "Stage Pos (mm)", "Raman Shift (cm⁻¹)", "SRS Gain"])
        if QT_AVAILABLE:
            self.tbl_sf_spectrum.horizontalHeader().setStretchLastSection(True)
            self.tbl_sf_spectrum.verticalHeader().setVisible(False)
        v_sf_res.addWidget(self.tbl_sf_spectrum)
        sf_layout.addWidget(grp_sf_results)

        sf_scroll.setWidget(sf_inner)
        tab_sf_layout = QVBoxLayout(tab_sf)
        tab_sf_layout.setContentsMargins(0, 0, 0, 0)
        tab_sf_layout.addWidget(sf_scroll)
        self.tabs.addTab(tab_sf, "Tab 4: 🌈 Spectral Focusing")
        if QT_AVAILABLE:
            self.tabs.setTabToolTip(3, "Spectral Focusing SRS Chirp Matching & Resolution")

        # ----------------------------------------------------------------------
        # Tab 5: Temporal Signal Probe (Ultrafast Overlap & Modulated Pulse Train)
        # ----------------------------------------------------------------------
        tab_probe = QWidget()
        probe_scroll = QScrollArea()
        probe_scroll.setWidgetResizable(True)
        probe_inner = QWidget()
        probe_layout = QVBoxLayout(probe_inner)
        probe_layout.setContentsMargins(6, 6, 6, 6)
        probe_layout.setSpacing(6)

        grp_overlap = QGroupBox("Sub-Picosecond Ultrafast Overlap Probe")
        form_overlap = QFormLayout(grp_overlap)

        self.spin_probe_delay_fs = QDoubleSpinBox()
        self.spin_probe_delay_fs.setRange(-2000.0, 2000.0)
        self.spin_probe_delay_fs.setValue(0.0)
        self.spin_probe_delay_fs.setSuffix(" fs")
        form_overlap.addRow("Inter-Pulse Delay (τ):", self.spin_probe_delay_fs)

        self.btn_run_overlap_probe = QPushButton("🔍 Probe Cross-Correlation Overlap")
        self.btn_run_overlap_probe.clicked.connect(self.run_overlap_probe)
        form_overlap.addRow(self.btn_run_overlap_probe)

        self.lbl_overlap_metrics = QLabel("2P Overlap: -- | 3P Overlap: -- | SRS Cross-Corr: --")
        self.lbl_overlap_metrics.setStyleSheet("font-weight: bold; color: #0277BD;")
        form_overlap.addRow("Instantaneous Overlap:", self.lbl_overlap_metrics)
        probe_layout.addWidget(grp_overlap)

        grp_train = QGroupBox("Nanosecond Periodic Pulse Train & Lock-In Modulation Probe")
        form_train = QFormLayout(grp_train)

        self.spin_probe_duration_ns = QDoubleSpinBox()
        self.spin_probe_duration_ns.setRange(10.0, 2000.0)
        self.spin_probe_duration_ns.setValue(200.0)
        self.spin_probe_duration_ns.setSuffix(" ns")
        form_train.addRow("Observation Duration:", self.spin_probe_duration_ns)

        self.btn_run_train_probe = QPushButton("📡 Probe Modulated Pulse Train Waveform")
        self.btn_run_train_probe.clicked.connect(self.run_pulse_train_probe)
        form_train.addRow(self.btn_run_train_probe)

        self.lbl_train_metrics = QLabel("Pulses in Window: -- | Modulation Envelope: -- | Demodulated Lock-In: --")
        self.lbl_train_metrics.setStyleSheet("font-weight: bold; color: #2E7D32;")
        form_train.addRow("Pulse Train Signal:", self.lbl_train_metrics)
        probe_layout.addWidget(grp_train)

        # Thorlabs Delay Stage Converter
        grp_odl_calc = QGroupBox("Thorlabs Optical Delay Stage Travel ↔ Delay Calculator")
        form_odl_calc = QFormLayout(grp_odl_calc)

        self.spin_calc_disp_mm = QDoubleSpinBox()
        self.spin_calc_disp_mm.setRange(0.001, 500.0)
        self.spin_calc_disp_mm.setValue(1.0)
        self.spin_calc_disp_mm.setSuffix(" mm")
        self.spin_calc_disp_mm.valueChanged.connect(self._on_calc_disp_changed)
        form_odl_calc.addRow("Stage Displacement (Δx):", self.spin_calc_disp_mm)

        self.lbl_calc_delay = QLabel("Optical Delay (Double-Pass): 6.671 ps")
        self.lbl_calc_delay.setStyleSheet("font-weight: bold; color: #D84315;")
        form_odl_calc.addRow("Generated Delay (Δt):", self.lbl_calc_delay)
        probe_layout.addWidget(grp_odl_calc)

        probe_scroll.setWidget(probe_inner)
        tab_probe_layout = QVBoxLayout(tab_probe)
        tab_probe_layout.setContentsMargins(0, 0, 0, 0)
        tab_probe_layout.addWidget(probe_scroll)
        self.tabs.addTab(tab_probe, "Tab 5: ⏱️ Signal Probe")
        if QT_AVAILABLE:
            self.tabs.setTabToolTip(4, "Temporal Signal Probe & Pulse Train Waveform")

        # ----------------------------------------------------------------------
        # Tab 6: Thorlabs SRS Microscope (Oil Spectrum, Scanning Effects & PD)
        # ----------------------------------------------------------------------
        tab_srs = QWidget()
        srs_scroll = QScrollArea()
        srs_scroll.setWidgetResizable(True)
        srs_inner = QWidget()
        srs_layout = QVBoxLayout(srs_inner)
        srs_layout.setContentsMargins(6, 6, 6, 6)
        srs_layout.setSpacing(6)

        # 1. Thorlabs Optical Train & Hardware Preset
        grp_srs_hw = QGroupBox("🏗 Real-World SRS Microscope (Folded 4f-Relayed Architecture)")
        v_srs_hw = QVBoxLayout(grp_srs_hw)

        # Primary prominent buttons to load the SRS microscope
        self.btn_load_non_sequential_srs = QPushButton("🔬 Load Non-Sequential Dual-Laser SRS Microscope (Pump + Stokes EOM + Delay Stage + Dichroic Combiner + Galvos + PD)")
        self.btn_load_non_sequential_srs.setStyleSheet("background-color: #004D40; color: white; font-weight: bold; padding: 8px; font-size: 11px; border-radius: 4px;")
        self.btn_load_non_sequential_srs.clicked.connect(self.load_non_sequential_srs_into_project)
        v_srs_hw.addWidget(self.btn_load_non_sequential_srs)

        self.btn_load_real_srs = QPushButton("🔬 Load Real Folded SRS Microscope (Power Controls + EOM + Dichroic + 4f-Relayed Galvos + Olympus NA 1.05 W + χ⁽³⁾ + PD)")
        self.btn_load_real_srs.setStyleSheet("background-color: #1A237E; color: white; font-weight: bold; padding: 8px; font-size: 11px; border-radius: 4px;")
        self.btn_load_real_srs.clicked.connect(self.load_folded_real_world_srs_template_into_project)
        v_srs_hw.addWidget(self.btn_load_real_srs)
        self.btn_load_folded_srs = self.btn_load_real_srs

        h_srs_json = QHBoxLayout()
        self.btn_save_srs_json = QPushButton("💾 Save Preset JSON")
        self.btn_save_srs_json.clicked.connect(self.save_srs_microscope_preset_dialog)
        h_srs_json.addWidget(self.btn_save_srs_json)

        self.btn_load_srs_json = QPushButton("📂 Load Preset JSON")
        self.btn_load_srs_json.clicked.connect(self.load_srs_microscope_preset_dialog)
        h_srs_json.addWidget(self.btn_load_srs_json)
        v_srs_hw.addLayout(h_srs_json)

        # Retain backward-compatible button attributes for test / automation access
        self.btn_load_lsm_template = QPushButton("Load LSM Core", self)
        self.btn_load_lsm_template.clicked.connect(self.load_thorlabs_lsm_template_into_project)
        self.btn_load_srs_template = QPushButton("Load Close-Coupled SRS", self)
        self.btn_load_srs_template.clicked.connect(self.load_srs_microscope_template_into_project)
        self.btn_load_relayed_srs = QPushButton("Load 4f-Relayed SRS", self)
        self.btn_load_relayed_srs.clicked.connect(self.load_relayed_srs_template_into_project)
        self.btn_load_olympus_srs = QPushButton("Load Olympus High-NA SRS", self)
        self.btn_load_olympus_srs.clicked.connect(self.load_olympus_high_na_template_into_project)
        self.btn_load_edmund_srs = QPushButton("Load Edmund Optics SRS", self)
        self.btn_load_edmund_srs.clicked.connect(self.load_edmund_optics_srs_template_into_project)

        self.lbl_srs_hw_summary = QLabel(
            "Laser Scanning Train: GVS002 Galvos (Close-coupled 9.5 mm, 1:1 4f-Relayed, or 90° Folded Mirror Reflections) → "
            "Dichroic Pulse Combiner (DMLP900 45° joins Pump 797 nm & EOM-Modulated Stokes 1040 nm) → 4f Telecentric Relay (SL50-2P2 + TTL200MP) → "
            "Objective BFP Pupil Conjugate → High-NA Objective (Thorlabs TL15X-2P NA 0.70 / Olympus XLPLN25XWMP2 NA 1.05 W / Nikon CFI75 NA 1.10 W) → "
            "Oil Sample Interaction Plane → ACL25416U-B Condenser (NA 0.79) → FESH0850 Stokes Dump → AC254-050-B Pupil Descan Collector → DET10N2 Photodiode."
        )
        self.lbl_srs_hw_summary.setWordWrap(True)
        self.lbl_srs_hw_summary.setStyleSheet("color: #37474F; font-size: 11px;")
        v_srs_hw.addWidget(self.lbl_srs_hw_summary)
        srs_layout.addWidget(grp_srs_hw)

        # 2. Modulator Selection (AOM or EOM)
        grp_srs_mod = QGroupBox("📻 Beam Modulation (AOM / EOM & Beam Dump)")
        form_srs_mod = QFormLayout(grp_srs_mod)

        self.combo_srs_modulator = QComboBox()
        self.combo_srs_modulator.addItems([
            "EOM (Thorlabs EO-AM-NR-C1 + PBS252 + BT610 Dump)",
            "AOM (TeO2 Slow Shear Bragg Deflector + BT610 Dump)",
        ])
        form_srs_mod.addRow("Modulator Device:", self.combo_srs_modulator)

        self.spin_srs_mod_freq_mhz = QDoubleSpinBox()
        self.spin_srs_mod_freq_mhz.setRange(0.1, 50.0)
        self.spin_srs_mod_freq_mhz.setValue(10.0)
        self.spin_srs_mod_freq_mhz.setSuffix(" MHz")
        form_srs_mod.addRow("Modulation Frequency:", self.spin_srs_mod_freq_mhz)

        self.spin_srs_eom_vpi = QDoubleSpinBox()
        self.spin_srs_eom_vpi.setRange(10.0, 1000.0)
        self.spin_srs_eom_vpi.setValue(180.0)
        self.spin_srs_eom_vpi.setSuffix(" V")
        form_srs_mod.addRow("EOM V_pi / Applied Voltage:", self.spin_srs_eom_vpi)

        self.spin_srs_aom_power_w = QDoubleSpinBox()
        self.spin_srs_aom_power_w.setRange(0.05, 10.0)
        self.spin_srs_aom_power_w.setValue(1.0)
        self.spin_srs_aom_power_w.setSuffix(" W")
        form_srs_mod.addRow("AOM RF Drive Power:", self.spin_srs_aom_power_w)
        srs_layout.addWidget(grp_srs_mod)

        # 3. Oil Sample & Raman Resonances
        grp_srs_oil = QGroupBox("🧪 Sample & Vibrational Raman Fingerprint (Oil)")
        form_srs_oil = QFormLayout(grp_srs_oil)

        self.combo_srs_oil = QComboBox()
        self.combo_srs_oil.addItems([
            "Mineral Oil (Paraffin CH2: 2850, 2885 cm⁻¹; CH3: 2935 cm⁻¹)",
            "Olive Oil (Extra Virgin: CH2/CH3 + cis-alkene =C-H 3005 cm⁻¹)",
            "Immersion Oil (Type A/F: C-H + aromatic 3050 cm⁻¹)",
        ])
        form_srs_oil.addRow("Oil Sample Standard:", self.combo_srs_oil)

        self.combo_srs_pattern = QComboBox()
        self.combo_srs_pattern.addItems([
            "Uniform Oil Film",
            "Oil Droplets in Water (Emulsion)",
            "Periodic Grid Target",
        ])
        form_srs_oil.addRow("Spatial Target Pattern:", self.combo_srs_pattern)

        self.spin_srs_sample_thick_um = QDoubleSpinBox()
        self.spin_srs_sample_thick_um.setRange(1.0, 500.0)
        self.spin_srs_sample_thick_um.setValue(50.0)
        self.spin_srs_sample_thick_um.setSuffix(" µm")
        form_srs_oil.addRow("Sample Thickness:", self.spin_srs_sample_thick_um)
        srs_layout.addWidget(grp_srs_oil)

        # 4. Galvo Scanning Unit & Aberration Effects
        grp_srs_scan = QGroupBox("📐 Galvo Scanner & Back Pupil Conjugation")
        form_srs_scan = QFormLayout(grp_srs_scan)

        self.combo_srs_scan_arch = QComboBox()
        self.combo_srs_scan_arch.addItems([
            "Close-Coupled Galvo (Thorlabs GVS002, 9.5 mm separation)",
            "4f-Relayed Galvo (Zero Pupil Walk-Off, Strictly Conjugated)",
            "Edmund Optics Telecentric (#64-422 + #58-520)",
            "Folded Real-World SRS (Dichroic + Reflected Galvos + Steering Mirrors)",
        ])
        self.combo_srs_scan_arch.currentIndexChanged.connect(self._on_srs_scan_changed)
        form_srs_scan.addRow("Scanning Architecture:", self.combo_srs_scan_arch)

        self.chk_srs_galvo_fold = QCheckBox("Break Optical Axis at Galvos (90° Specular Reflection Fold)")
        self.chk_srs_galvo_fold.setChecked(False)
        self.chk_srs_galvo_fold.stateChanged.connect(self._on_srs_scan_changed)
        form_srs_scan.addRow("Galvo Axis Reflection:", self.chk_srs_galvo_fold)

        self.combo_srs_objective = QComboBox()
        self.combo_srs_objective.addItems([
            "Thorlabs TL15X-2P (15x NA 0.70 Air)",
            "Olympus XLPLN25XWMP2 (25x NA 1.05 W Multiphoton)",
            "Olympus XLUMPLFLN20XW (20x NA 1.00 W)",
            "Olympus UPLSAPO60XW (60x NA 1.20 W High-NA)",
            "Nikon CFI75 25XC W (25x NA 1.10 W)",
        ])
        self.combo_srs_objective.currentIndexChanged.connect(self._on_srs_scan_changed)
        form_srs_scan.addRow("Objective Lens:", self.combo_srs_objective)

        h_angles = QHBoxLayout()
        self.spin_srs_galvo_x = QDoubleSpinBox()
        self.spin_srs_galvo_x.setRange(-5.0, 5.0)
        self.spin_srs_galvo_x.setValue(0.0)
        self.spin_srs_galvo_x.setSuffix(" °")
        self.spin_srs_galvo_x.valueChanged.connect(self._on_srs_scan_changed)
        h_angles.addWidget(QLabel("θx:"))
        h_angles.addWidget(self.spin_srs_galvo_x)

        self.spin_srs_galvo_y = QDoubleSpinBox()
        self.spin_srs_galvo_y.setRange(-5.0, 5.0)
        self.spin_srs_galvo_y.setValue(0.0)
        self.spin_srs_galvo_y.setSuffix(" °")
        self.spin_srs_galvo_y.valueChanged.connect(self._on_srs_scan_changed)
        h_angles.addWidget(QLabel("θy:"))
        h_angles.addWidget(self.spin_srs_galvo_y)
        form_srs_scan.addRow("Mechanical Galvo Angles:", h_angles)

        self.spin_srs_chrom_walk = QDoubleSpinBox()
        self.spin_srs_chrom_walk.setRange(0.0, 0.05)
        self.spin_srs_chrom_walk.setValue(0.004)
        self.spin_srs_chrom_walk.setSingleStep(0.001)
        self.spin_srs_chrom_walk.setSuffix(" (0.4%)")
        self.spin_srs_chrom_walk.valueChanged.connect(self._on_srs_scan_changed)
        form_srs_scan.addRow("Transverse Chromatic Walk-Off (C_LCA):", self.spin_srs_chrom_walk)

        self.spin_srs_petzval_mm = QDoubleSpinBox()
        self.spin_srs_petzval_mm.setRange(5.0, 200.0)
        self.spin_srs_petzval_mm.setValue(25.0)
        self.spin_srs_petzval_mm.setSuffix(" mm")
        self.spin_srs_petzval_mm.valueChanged.connect(self._on_srs_scan_changed)
        form_srs_scan.addRow("Petzval Field Curvature Radius:", self.spin_srs_petzval_mm)

        self.lbl_srs_pupil_metrics = QLabel("Pupil Mag: 4.0× | BFP Beam: ∅ 18.0 mm | Obj Pupil: ∅ 18.7 mm (Fill: 96.4%) | Walk-off: 0.00 µm | Status: Conjugated")
        self.lbl_srs_pupil_metrics.setStyleSheet("font-weight: bold; color: #1565C0;")
        form_srs_scan.addRow("Back Pupil Conjugation:", self.lbl_srs_pupil_metrics)

        self.lbl_srs_scan_metrics = QLabel("Sample Pos: (0.0, 0.0) µm | Chromatic Walk-off: 0.00 µm | Defocus: 0.00 µm | Scan Eff: 100.0%")
        self.lbl_srs_scan_metrics.setStyleSheet("font-weight: bold; color: #00695C;")
        form_srs_scan.addRow("Scan Field Diagnostics:", self.lbl_srs_scan_metrics)
        srs_layout.addWidget(grp_srs_scan)

        # 5. Photodiode Detector & Lock-In Demodulation
        grp_srs_pd = QGroupBox("⚡ Photodiode Detector & Lock-In Demodulation (DET10N2)")
        form_srs_pd = QFormLayout(grp_srs_pd)

        self.spin_srs_lockin_tc_us = QDoubleSpinBox()
        self.spin_srs_lockin_tc_us.setRange(0.5, 1000.0)
        self.spin_srs_lockin_tc_us.setValue(20.0)
        self.spin_srs_lockin_tc_us.setSuffix(" µs")
        form_srs_pd.addRow("Lock-In Time Constant (τ):", self.spin_srs_lockin_tc_us)

        self.spin_srs_lockin_phase_deg = QDoubleSpinBox()
        self.spin_srs_lockin_phase_deg.setRange(-180.0, 180.0)
        self.spin_srs_lockin_phase_deg.setValue(0.0)
        self.spin_srs_lockin_phase_deg.setSuffix(" °")
        form_srs_pd.addRow("Lock-In Reference Phase:", self.spin_srs_lockin_phase_deg)

        self.lbl_srs_pd_metrics = QLabel("DC: -- mA (-- V) | Lock-in R: -- µV | Noise Floor: -- µV | SNR: -- dB")
        self.lbl_srs_pd_metrics.setStyleSheet("font-weight: bold; color: #2E7D32;")
        form_srs_pd.addRow("Detector Output Signal:", self.lbl_srs_pd_metrics)
        srs_layout.addWidget(grp_srs_pd)

        # 6. Post-Sample SRS Optical Probe (Spectral Focusing Specific Delay & Intensity)
        grp_srs_probe = QGroupBox("🔬 Post-Sample SRS Probe & Spectral Focusing Delay Control")
        form_srs_probe = QFormLayout(grp_srs_probe)

        h_probe_ctrl = QHBoxLayout()
        self.spin_srs_probe_delay_ps = QDoubleSpinBox()
        self.spin_srs_probe_delay_ps.setRange(-15.0, 15.0)
        self.spin_srs_probe_delay_ps.setValue(0.0)
        self.spin_srs_probe_delay_ps.setSingleStep(0.05)
        self.spin_srs_probe_delay_ps.setSuffix(" ps")
        self.spin_srs_probe_delay_ps.valueChanged.connect(self._on_srs_probe_delay_changed)
        h_probe_ctrl.addWidget(QLabel("Inter-Pulse Delay (τ_d):"))
        h_probe_ctrl.addWidget(self.spin_srs_probe_delay_ps)

        self.spin_srs_probe_stage_mm = QDoubleSpinBox()
        self.spin_srs_probe_stage_mm.setRange(-2.25, 2.25)
        self.spin_srs_probe_stage_mm.setValue(0.0)
        self.spin_srs_probe_stage_mm.setSingleStep(0.01)
        self.spin_srs_probe_stage_mm.setSuffix(" mm")
        self.spin_srs_probe_stage_mm.valueChanged.connect(self._on_srs_probe_stage_changed)
        h_probe_ctrl.addWidget(QLabel("Stage Pos (Δx):"))
        h_probe_ctrl.addWidget(self.spin_srs_probe_stage_mm)
        form_srs_probe.addRow("Probe Optical Delay:", h_probe_ctrl)

        # Quick Jump Buttons for Target Vibrational Resonances
        h_quick_peaks = QHBoxLayout()
        self.btn_probe_ch2_sym = QPushButton("CH₂ Sym (2850)")
        self.btn_probe_ch2_sym.clicked.connect(lambda: self._set_probe_target_cm1(2850.0))
        h_quick_peaks.addWidget(self.btn_probe_ch2_sym)

        self.btn_probe_ch2_asym = QPushButton("CH₂ Asym (2885)")
        self.btn_probe_ch2_asym.clicked.connect(lambda: self._set_probe_target_cm1(2885.0))
        h_quick_peaks.addWidget(self.btn_probe_ch2_asym)

        self.btn_probe_ch3 = QPushButton("CH₃ (2935)")
        self.btn_probe_ch3.clicked.connect(lambda: self._set_probe_target_cm1(2935.0))
        h_quick_peaks.addWidget(self.btn_probe_ch3)

        self.btn_probe_cis_alkene = QPushButton("cis =C-H (3005)")
        self.btn_probe_cis_alkene.clicked.connect(lambda: self._set_probe_target_cm1(3005.0))
        h_quick_peaks.addWidget(self.btn_probe_cis_alkene)

        self.btn_probe_off_res = QPushButton("Off-Res (2750)")
        self.btn_probe_off_res.clicked.connect(lambda: self._set_probe_target_cm1(2750.0))
        h_quick_peaks.addWidget(self.btn_probe_off_res)
        form_srs_probe.addRow("Target Resonance Presets:", h_quick_peaks)

        self.lbl_srs_probe_wavenumber = QLabel("Probed Raman Shift: 2928.3 cm⁻¹ | Mode: CH3 mode (ON-Resonance)")
        self.lbl_srs_probe_wavenumber.setStyleSheet("font-weight: bold; color: #1565C0;")
        form_srs_probe.addRow("Instantaneous Beat (Ω):", self.lbl_srs_probe_wavenumber)

        self.btn_run_post_sample_probe = QPushButton("🔍 Probe Generated Intensity After Sample Plane")
        self.btn_run_post_sample_probe.setStyleSheet("background-color: #00838F; color: white; font-weight: bold; padding: 6px;")
        self.btn_run_post_sample_probe.clicked.connect(self.run_post_sample_probe)
        form_srs_probe.addRow(self.btn_run_post_sample_probe)

        self.lbl_srs_probe_intensity_res = QLabel("Post-Sample SRL: -- | Modulated AC Power: -- µW | Lock-In: -- µV | SNR: -- dB")
        self.lbl_srs_probe_intensity_res.setStyleSheet("font-weight: bold; color: #2E7D32;")
        form_srs_probe.addRow("Post-Sample Signal:", self.lbl_srs_probe_intensity_res)
        srs_layout.addWidget(grp_srs_probe)

        # 7. Run Button & Reconstructed Spectrum Results
        self.btn_run_srs_microscope = QPushButton("🚀 Simulate Thorlabs SRS Microscope (Oil Spectrum + Scan Effects + PD Signal)")
        self.btn_run_srs_microscope.setStyleSheet("background-color: #2E7D32; color: white; font-weight: bold; padding: 7px; font-size: 13px;")
        self.btn_run_srs_microscope.clicked.connect(self.run_srs_microscope_simulation)
        srs_layout.addWidget(self.btn_run_srs_microscope)

        grp_srs_res = QGroupBox("Reconstructed Vibrational Raman Spectrum & Scanning Roll-Off")
        v_srs_res = QVBoxLayout(grp_srs_res)

        self.lbl_srs_summary = QLabel("Peak Band: -- cm⁻¹ | On-Axis: -- µV | Off-Axis: -- µV (Roll-off: --%) | Lock-in SNR: -- dB")
        self.lbl_srs_summary.setStyleSheet("font-weight: bold; color: #1B5E20;")
        v_srs_res.addWidget(self.lbl_srs_summary)

        self.tbl_srs_spectrum = QTableWidget()
        self.tbl_srs_spectrum.setColumnCount(7)
        self.tbl_srs_spectrum.setHorizontalHeaderLabels([
            "Delay (ps)", "Stage (mm)", "Raman Shift (cm⁻¹)", "On-Axis (µV)", "Off-Axis (µV)", "Scan Eff (%)", "SNR (dB)"
        ])
        if QT_AVAILABLE:
            self.tbl_srs_spectrum.horizontalHeader().setStretchLastSection(True)
            self.tbl_srs_spectrum.verticalHeader().setVisible(False)
        v_srs_res.addWidget(self.tbl_srs_spectrum)
        srs_layout.addWidget(grp_srs_res)

        srs_scroll.setWidget(srs_inner)
        tab_srs_layout = QVBoxLayout(tab_srs)
        tab_srs_layout.setContentsMargins(0, 0, 0, 0)
        tab_srs_layout.addWidget(srs_scroll)
        self.tabs.addTab(tab_srs, "Tab 6: 🔬 Thorlabs SRS Microscope")
        if QT_AVAILABLE:
            self.tabs.setTabToolTip(5, "Complete 18-Component Thorlabs SRS Microscope Simulation & Preset Loader")

        self.setWidget(container)
        self.update_modulator_metrics()
        self.switch_to_tab(5)
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
        if is_dark:
            if hasattr(self, 'nav_container') and self.nav_container:
                self.nav_container.setStyleSheet("background-color: #1e293b; border-radius: 4px; padding: 3px; border: 1px solid #334155;")
            if hasattr(self, 'lbl_status') and self.lbl_status:
                self.lbl_status.setStyleSheet(
                    "background-color: #3b0764; color: #e9d5ff; font-weight: bold; "
                    "padding: 5px 8px; border-radius: 4px; border: 1px solid #7e22ce;"
                )
            if hasattr(self, 'lbl_prechirp') and self.lbl_prechirp:
                self.lbl_prechirp.setStyleSheet("font-weight: bold; color: #38bdf8;")
            if hasattr(self, 'lbl_defects_status') and self.lbl_defects_status:
                self.lbl_defects_status.setStyleSheet("font-weight: bold; color: #c084fc;")
            if hasattr(self, 'lbl_aom_results') and self.lbl_aom_results:
                self.lbl_aom_results.setStyleSheet("font-weight: bold; color: #4ade80;")
            if hasattr(self, 'lbl_eom_results') and self.lbl_eom_results:
                self.lbl_eom_results.setStyleSheet("font-weight: bold; color: #60a5fa;")
            if hasattr(self, 'lbl_stage_displacement') and self.lbl_stage_displacement:
                self.lbl_stage_displacement.setStyleSheet("font-weight: bold; color: #fb923c;")
            if hasattr(self, 'lbl_sf_summary') and self.lbl_sf_summary:
                self.lbl_sf_summary.setStyleSheet("font-weight: bold; color: #f87171;")
            if hasattr(self, 'lbl_overlap_metrics') and self.lbl_overlap_metrics:
                self.lbl_overlap_metrics.setStyleSheet("font-weight: bold; color: #38bdf8;")
            if hasattr(self, 'lbl_train_metrics') and self.lbl_train_metrics:
                self.lbl_train_metrics.setStyleSheet("font-weight: bold; color: #4ade80;")
            if hasattr(self, 'lbl_calc_delay') and self.lbl_calc_delay:
                self.lbl_calc_delay.setStyleSheet("font-weight: bold; color: #fb923c;")
            if hasattr(self, 'lbl_srs_hw_summary') and self.lbl_srs_hw_summary:
                self.lbl_srs_hw_summary.setStyleSheet("color: #cbd5e1; font-size: 11px;")
            if hasattr(self, 'lbl_srs_pupil_metrics') and self.lbl_srs_pupil_metrics:
                self.lbl_srs_pupil_metrics.setStyleSheet("font-weight: bold; color: #60a5fa;")
            if hasattr(self, 'lbl_srs_scan_metrics') and self.lbl_srs_scan_metrics:
                self.lbl_srs_scan_metrics.setStyleSheet("font-weight: bold; color: #2dd4bf;")
            if hasattr(self, 'lbl_srs_pd_metrics') and self.lbl_srs_pd_metrics:
                self.lbl_srs_pd_metrics.setStyleSheet("font-weight: bold; color: #4ade80;")
            if hasattr(self, 'lbl_srs_probe_wavenumber') and self.lbl_srs_probe_wavenumber:
                self.lbl_srs_probe_wavenumber.setStyleSheet("font-weight: bold; color: #60a5fa;")
            if hasattr(self, 'lbl_srs_probe_intensity_res') and self.lbl_srs_probe_intensity_res:
                self.lbl_srs_probe_intensity_res.setStyleSheet("font-weight: bold; color: #4ade80;")
            if hasattr(self, 'lbl_srs_summary') and self.lbl_srs_summary:
                self.lbl_srs_summary.setStyleSheet("font-weight: bold; color: #4ade80;")
        else:
            if hasattr(self, 'nav_container') and self.nav_container:
                self.nav_container.setStyleSheet("background-color: #ECEFF1; border-radius: 4px; padding: 3px;")
            if hasattr(self, 'lbl_status') and self.lbl_status:
                self.lbl_status.setStyleSheet(
                    "background-color: #F3E5F5; color: #4A148C; font-weight: bold; "
                    "padding: 5px 8px; border-radius: 4px; border: 1px solid #CE93D8;"
                )
            if hasattr(self, 'lbl_prechirp') and self.lbl_prechirp:
                self.lbl_prechirp.setStyleSheet("font-weight: bold; color: #1E88E5;")
            if hasattr(self, 'lbl_defects_status') and self.lbl_defects_status:
                self.lbl_defects_status.setStyleSheet("font-weight: bold; color: #7B1FA2;")
            if hasattr(self, 'lbl_aom_results') and self.lbl_aom_results:
                self.lbl_aom_results.setStyleSheet("font-weight: bold; color: #2E7D32;")
            if hasattr(self, 'lbl_eom_results') and self.lbl_eom_results:
                self.lbl_eom_results.setStyleSheet("font-weight: bold; color: #1565C0;")
            if hasattr(self, 'lbl_stage_displacement') and self.lbl_stage_displacement:
                self.lbl_stage_displacement.setStyleSheet("font-weight: bold; color: #E65100;")
            if hasattr(self, 'lbl_sf_summary') and self.lbl_sf_summary:
                self.lbl_sf_summary.setStyleSheet("font-weight: bold; color: #BF360C;")
            if hasattr(self, 'lbl_overlap_metrics') and self.lbl_overlap_metrics:
                self.lbl_overlap_metrics.setStyleSheet("font-weight: bold; color: #0277BD;")
            if hasattr(self, 'lbl_train_metrics') and self.lbl_train_metrics:
                self.lbl_train_metrics.setStyleSheet("font-weight: bold; color: #2E7D32;")
            if hasattr(self, 'lbl_calc_delay') and self.lbl_calc_delay:
                self.lbl_calc_delay.setStyleSheet("font-weight: bold; color: #D84315;")
            if hasattr(self, 'lbl_srs_hw_summary') and self.lbl_srs_hw_summary:
                self.lbl_srs_hw_summary.setStyleSheet("color: #37474F; font-size: 11px;")
            if hasattr(self, 'lbl_srs_pupil_metrics') and self.lbl_srs_pupil_metrics:
                self.lbl_srs_pupil_metrics.setStyleSheet("font-weight: bold; color: #1565C0;")
            if hasattr(self, 'lbl_srs_scan_metrics') and self.lbl_srs_scan_metrics:
                self.lbl_srs_scan_metrics.setStyleSheet("font-weight: bold; color: #00695C;")
            if hasattr(self, 'lbl_srs_pd_metrics') and self.lbl_srs_pd_metrics:
                self.lbl_srs_pd_metrics.setStyleSheet("font-weight: bold; color: #2E7D32;")
            if hasattr(self, 'lbl_srs_probe_wavenumber') and self.lbl_srs_probe_wavenumber:
                self.lbl_srs_probe_wavenumber.setStyleSheet("font-weight: bold; color: #1565C0;")
            if hasattr(self, 'lbl_srs_probe_intensity_res') and self.lbl_srs_probe_intensity_res:
                self.lbl_srs_probe_intensity_res.setStyleSheet("font-weight: bold; color: #2E7D32;")
            if hasattr(self, 'lbl_srs_summary') and self.lbl_srs_summary:
                self.lbl_srs_summary.setStyleSheet("font-weight: bold; color: #1B5E20;")

    def switch_to_tab(self, idx: int):
        """Programmatically switch to a tab and sync the module navigation combo box."""
        if hasattr(self, 'tabs') and 0 <= idx < self.tabs.count():
            self.tabs.setCurrentIndex(idx)
        if hasattr(self, 'combo_module_nav'):
            self.combo_module_nav.blockSignals(True)
            self.combo_module_nav.setCurrentIndex(idx)
            self.combo_module_nav.blockSignals(False)

    def _sync_module_nav_combo(self, idx: int):
        """Syncs the top combo box when user clicks on a tab directly."""
        if hasattr(self, 'combo_module_nav') and 0 <= idx < self.combo_module_nav.count():
            self.combo_module_nav.blockSignals(True)
            self.combo_module_nav.setCurrentIndex(idx)
            self.combo_module_nav.blockSignals(False)

    def _on_calc_disp_changed(self, val: float):
        delay_ps = delay_from_stage_displacement_mm(val, passes=2)
        self.lbl_calc_delay.setText(f"Optical Delay (Double-Pass): {delay_ps:.3f} ps")

    def _on_modality_changed(self, index: int):
        is_cars_or_srs = (index in (3, 4))
        self.spin_stokes_wl_nm.setEnabled(is_cars_or_srs)
        if index == 1:  # 3P
            self.spin_laser_wl_nm.setValue(1300.0)
            self.spin_rep_rate_mhz.setValue(4.0)
        elif index in (0, 2):  # 2P, SHG
            self.spin_laser_wl_nm.setValue(800.0)
            self.spin_rep_rate_mhz.setValue(80.0)

    def set_system(self, system: OpticalSystem):
        self.system = system

    def update_ui(self):
        """Automatically refresh if system has been traced or source changed."""
        if getattr(self.main_window, 'source_config', None):
            mode = getattr(self.main_window.source_config, 'laser_mode', 'CW')
            if mode == "Ultrafast":
                self.run_analysis()
            else:
                self.table.setRowCount(0)
                self.lbl_status.setText("Source is not in 'Ultrafast' mode.")

    def run_analysis(self):
        """Run pulse trace along sequential surfaces and compute pre-chirp."""
        if self.system is None and self.main_window is not None:
            if hasattr(self.main_window, 'get_optical_system'):
                self.system = self.main_window.get_optical_system()
                
        if self.system is None:
            self.lbl_status.setText("No optical system available to analyze.")
            return

        source = getattr(self.main_window, 'source_config', None)
        if not source:
            self.lbl_status.setText("No source configuration found.")
            return

        mode = getattr(source, 'laser_mode', 'CW')
        if mode != "Ultrafast":
            self.lbl_status.setText("Source is not in 'Ultrafast' mode. Change it in the Source Editor.")
            self.table.setRowCount(0)
            return

        config = PulseAnalysisConfig()
        wls = [float(x.strip()) for x in str(source.wavelengths).split(',') if x.strip()]
        if wls:
            config.center_wavelength_nm = wls[len(wls)//2] * 1000.0
            
        config.transform_limit_fwhm_fs = getattr(source, 'pulse_duration_fs', 100.0)
        config.spectral_span_fwhm = getattr(source, 'bandwidth_nm', 10.0) / 4.0
        
        try:
            result = PulseWidthAnalysis.analyze(self.system, config)
            self._last_pulse_result = result
            
            self.table.setRowCount(len(result.probes))
            accumulated_gdd = 0.0
            for i, p in enumerate(result.probes):
                self.table.setItem(i, 0, QTableWidgetItem(str(p.surface_index)))
                self.table.setItem(i, 1, QTableWidgetItem(p.surface_name))
                self.table.setItem(i, 2, QTableWidgetItem(f"{p.group_delay_fs:.1f}"))
                self.table.setItem(i, 3, QTableWidgetItem(f"{p.material_gdd_fs2:.1f}"))
                self.table.setItem(i, 4, QTableWidgetItem(f"{p.geometric_gdd_fs2:.1f}"))
                self.table.setItem(i, 5, QTableWidgetItem(f"{p.total_gdd_fs2:.1f}"))
                self.table.setItem(i, 6, QTableWidgetItem(f"{p.pulse_fwhm_fs:.1f}"))
                self.table.setItem(i, 7, QTableWidgetItem(f"{p.spatial_chirp_mm:.4f}"))
                accumulated_gdd = p.total_gdd_fs2

            prechirp = -accumulated_gdd
            self.lbl_prechirp.setText(f"Pre-Chirp Needed: {prechirp:.1f} fs² (to restore {config.transform_limit_fwhm_fs:.0f} fs at focus)")
                
            if result.warnings:
                self.lbl_status.setText(f"Analyzed {len(result.probes)} surfaces. Warning: {result.warnings[0]}")
            else:
                self.lbl_status.setText(f"Analyzed {len(result.probes)} surfaces successfully. Total GDD: {accumulated_gdd:.1f} fs²")
                
        except Exception as e:
            self.lbl_status.setText(f"Error during analysis: {e}")

    def update_modulator_metrics(self):
        """Recalculate AOM and EOM physical parameters."""
        mat_map = {
            0: AOMMaterial.TEO2_SLOW_SHEAR,
            1: AOMMaterial.TEO2_LONGITUDINAL,
            2: AOMMaterial.FUSED_SILICA,
            3: AOMMaterial.QUARTZ,
        }
        aom = AcoustoOpticModulator(
            material=mat_map.get(self.combo_aom_mat.currentIndex(), AOMMaterial.TEO2_SLOW_SHEAR),
            rf_frequency_mhz=self.spin_aom_rf_freq.value(),
            rf_power_w=self.spin_aom_rf_pwr.value(),
        )
        wl = self.spin_laser_wl_nm.value()
        bragg_mrad = aom.bragg_angle_rad(wl) * 1e3
        eta = aom.diffraction_efficiency(wavelength_nm=wl) * 100.0
        rise_ns = aom.acoustic_rise_time_ns(self.spin_aom_waist.value())
        bw_mhz = aom.modulation_bandwidth_mhz(self.spin_aom_waist.value())
        aom_gdd = aom.crystal_gdd_fs2(wl)
        self.lbl_aom_results.setText(
            f"Bragg Angle: {bragg_mrad:.2f} mrad | 1st Order Eff: {eta:.1f}% | "
            f"Rise Time: {rise_ns:.1f} ns (BW: {bw_mhz:.1f} MHz) | GDD: +{aom_gdd:.0f} fs²"
        )

        eom_mat_map = {
            0: EOMMaterial.RTP,
            1: EOMMaterial.KDP,
            2: EOMMaterial.LINBO3,
        }
        eom = ElectroOpticModulator(
            material=eom_mat_map.get(self.combo_eom_mat.currentIndex(), EOMMaterial.RTP),
            half_wave_voltage_v=self.spin_eom_vpi.value(),
            applied_voltage_v=self.spin_eom_applied_v.value(),
            modulation_frequency_mhz=self.spin_eom_mod_freq.value(),
        )
        ret_rad = eom.phase_retardation_rad()
        t_pct = eom.transmission() * 100.0
        eom_gdd = eom.crystal_gdd_fs2(wl)
        self.lbl_eom_results.setText(
            f"Retardation: {ret_rad:.2f} rad ({math.degrees(ret_rad):.1f}°) | "
            f"PBS Transmission: {t_pct:.2f}% | GDD: +{eom_gdd:.0f} fs²"
        )
        return aom, eom

    def run_nonlinear_simulation(self):
        """Simulate selected nonlinear imaging modality with interaction plane and defects."""
        aom, eom = self.update_modulator_metrics()
        use_aom = self.cb_apply_aom.isChecked()
        use_eom = self.cb_apply_eom.isChecked()

        laser = LaserSourceParams(
            center_wavelength_nm=self.spin_laser_wl_nm.value(),
            average_power_w=self.spin_laser_power_mw.value() * 1e-3,
            repetition_rate_hz=self.spin_rep_rate_mhz.value() * 1e6,
            pulse_duration_fs=self.spin_pulse_duration_fs.value(),
        )
        sample = NonlinearSampleParams(
            fluorophore_concentration_molar=self.spin_conc_um.value() * 1e-6,
            sigma_2p_gm=self.spin_sigma_2p.value(),
            raman_shift_cm1=self.spin_raman_shift.value(),
        )
        obj = ObjectiveFocusParams(
            numerical_aperture=self.spin_na.value(),
        )

        # Interaction plane configuration
        interaction_plane = InteractionPlaneConfig(
            surface_name=self.combo_interaction_plane.currentText(),
            axial_defocus_um=self.spin_defocus_um.value(),
        )

        # Spatial defects configuration
        spatial_defects = None
        if self.cb_enable_defects.isChecked():
            spatial_defects = SpatialBeamDefects(
                spatial_chirp_mm_per_nm=self.spin_spatial_chirp.value(),
                angular_dispersion_mrad_per_nm=self.spin_angular_disp.value(),
                wavefront_error_rms_waves=self.spin_wavefront_rms.value(),
                beam_ellipticity=self.spin_ellipticity.value(),
            )
            w0_um = obj.beam_waist_radius_m(laser.center_wavelength_nm) * 1e6
            strehl = spatial_defects.strehl_ratio
            pft_mrad = spatial_defects.pulse_front_tilt_rad(laser.center_wavelength_nm) * 1e3
            deg = spatial_defects.intensity_degradation_factor(w0_um, laser.pulse_duration_fs, laser.center_wavelength_nm)
            self.lbl_defects_status.setText(
                f"Strehl Ratio: {strehl:.3f} | Pulse Front Tilt: {pft_mrad:.2f} mrad | Intensity Factor: {deg:.3f}"
            )

        idx = self.combo_modality.currentIndex()
        mod_aom = aom if use_aom else None
        mod_eom = eom if use_eom else None

        if idx == 0:  # 2P
            res = UltrafastNonlinearSimulator.simulate_two_photon(
                laser, sample, obj, aom=mod_aom, eom=mod_eom,
                interaction_plane=interaction_plane, spatial_defects=spatial_defects
            )
        elif idx == 1:  # 3P
            res = UltrafastNonlinearSimulator.simulate_three_photon(
                laser, sample, obj, tissue_depth_um=self.spin_tissue_depth_um.value(),
                aom=mod_aom, eom=mod_eom,
                interaction_plane=interaction_plane, spatial_defects=spatial_defects
            )
        elif idx == 2:  # SHG
            res = UltrafastNonlinearSimulator.simulate_shg(
                laser, sample, obj, aom=mod_aom, eom=mod_eom,
                interaction_plane=interaction_plane, spatial_defects=spatial_defects
            )
        elif idx == 3:  # CARS
            stokes = LaserSourceParams(
                center_wavelength_nm=self.spin_stokes_wl_nm.value(),
                average_power_w=self.spin_laser_power_mw.value() * 1e-3 * 0.5,
                repetition_rate_hz=self.spin_rep_rate_mhz.value() * 1e6,
                pulse_duration_fs=self.spin_pulse_duration_fs.value(),
            )
            res = UltrafastNonlinearSimulator.simulate_cars(
                laser, stokes, sample, obj,
                interaction_plane=interaction_plane, spatial_defects=spatial_defects
            )
        else:  # SRS
            stokes = LaserSourceParams(
                center_wavelength_nm=self.spin_stokes_wl_nm.value(),
                average_power_w=self.spin_laser_power_mw.value() * 1e-3 * 0.5,
                repetition_rate_hz=self.spin_rep_rate_mhz.value() * 1e6,
                pulse_duration_fs=self.spin_pulse_duration_fs.value(),
            )
            res = UltrafastNonlinearSimulator.simulate_srs(
                laser, stokes, sample, obj, eom_modulator=mod_eom,
                interaction_plane=interaction_plane, spatial_defects=spatial_defects
            )

        self._last_nonlinear_result = res
        self._display_nonlinear_results(res)

    def _display_nonlinear_results(self, res: NonlinearModalityResult):
        metrics = [
            ("Modality", res.modality),
            ("Emission Wavelength", f"{res.emission_wavelength_nm:.1f} nm"),
            ("Focal Spot Diameter (2w)", f"{res.focal_spot_diameter_um:.2f} µm"),
            ("Focal Peak Intensity", f"{res.focal_peak_intensity_w_cm2:.2e} W/cm²"),
            ("Focal Peak Fluence", f"{res.focal_peak_fluence_j_cm2:.2e} J/cm²"),
            ("Excitation Volume", f"{res.excitation_volume_fl:.3f} fL"),
            ("Generated Photons / sec", f"{res.generated_photons_per_second:.2e}"),
            ("Detected Photons / pixel dwell", f"{res.detected_photons_per_dwell:.2f}"),
            ("Shot-Noise Limited SNR", f"{res.shot_noise_snr:.1f}"),
            ("Power Scaling Exponent", f"P^{res.scaling_law_power_exponent:.0f}"),
        ]

        for k, v in res.details.items():
            if isinstance(v, float):
                metrics.append((k.replace('_', ' ').title(), f"{v:.3e}" if abs(v) < 1e-2 or abs(v) > 1e4 else f"{v:.3f}"))
            elif isinstance(v, (int, str)):
                metrics.append((k.replace('_', ' ').title(), str(v)))

        self.tbl_nonlinear_results.setRowCount(len(metrics))
        for row, (k, val) in enumerate(metrics):
            self.tbl_nonlinear_results.setItem(row, 0, QTableWidgetItem(k))
            self.tbl_nonlinear_results.setItem(row, 1, QTableWidgetItem(val))

    def run_spectral_focusing_simulation(self):
        """Simulate Spectral Focusing SRS with chirped broadband pulses and delay scan."""
        pump = LaserSourceParams(
            center_wavelength_nm=self.spin_laser_wl_nm.value(),
            pulse_duration_fs=self.spin_pulse_duration_fs.value(),
            average_power_w=self.spin_laser_power_mw.value() * 1e-3,
            repetition_rate_hz=self.spin_rep_rate_mhz.value() * 1e6,
        )
        stokes = LaserSourceParams(
            center_wavelength_nm=self.spin_stokes_wl_nm.value(),
            pulse_duration_fs=self.spin_pulse_duration_fs.value(),
            average_power_w=self.spin_laser_power_mw.value() * 1e-3 * 0.5,
            repetition_rate_hz=self.spin_rep_rate_mhz.value() * 1e6,
        )

        glass_map = {
            0: DispersiveGlassMaterial.SF57,
            1: DispersiveGlassMaterial.SF11,
            2: DispersiveGlassMaterial.FUSED_SILICA,
            3: DispersiveGlassMaterial.N_BK7,
        }
        pump_mat = glass_map.get(self.combo_pump_glass.currentIndex(), DispersiveGlassMaterial.SF57)
        stokes_mat = glass_map.get(self.combo_stokes_glass.currentIndex(), DispersiveGlassMaterial.SF57)

        cfg = SpectralFocusingConfig(
            pump_glass_material=pump_mat,
            pump_glass_length_mm=self.spin_pump_rod_len.value(),
            stokes_glass_material=stokes_mat,
            stokes_glass_length_mm=self.spin_stokes_rod_len.value(),
            delay_scan_range_ps=self.spin_delay_scan_ps.value(),
            delay_samples=101,
        )
        sample = NonlinearSampleParams(
            raman_gain_m_per_w=1.2e-13,
        )

        res = SpectralFocusingSRS.simulate(pump, stokes, cfg, sample)
        self._last_spectral_focusing_result = res

        # Update stage travel label
        disp_mm = stage_displacement_from_delay_ps(self.spin_delay_scan_ps.value(), passes=2)
        self.lbl_stage_displacement.setText(f"Stage Travel Required: ±{disp_mm:.3f} mm")

        self.lbl_sf_summary.setText(
            f"Chirp Match: {res.chirp_matching_ratio:.2f} | Resolution: {res.spectral_resolution_cm1:.1f} cm⁻¹ | "
            f"Pump: {res.chirped_pump_fwhm_ps:.2f} ps (+{res.pump_gdd_fs2:.0f} fs²) | "
            f"Stokes: {res.chirped_stokes_fwhm_ps:.2f} ps (+{res.stokes_gdd_fs2:.0f} fs²)"
        )

        # Populate spectrum table
        n_pts = min(100, len(res.delays_ps))
        step = max(1, len(res.delays_ps) // n_pts)
        sample_indices = range(0, len(res.delays_ps), step)

        self.tbl_sf_spectrum.setRowCount(len(sample_indices))
        for row, idx in enumerate(sample_indices):
            d_ps = res.delays_ps[idx]
            pos_mm = stage_displacement_from_delay_ps(d_ps, passes=2)
            shift_cm1 = res.instantaneous_frequency_difference_cm1[idx]
            gain = res.reconstructed_raman_spectrum[idx]

            self.tbl_sf_spectrum.setItem(row, 0, QTableWidgetItem(f"{d_ps:+.2f}"))
            self.tbl_sf_spectrum.setItem(row, 1, QTableWidgetItem(f"{pos_mm:+.3f}"))
            self.tbl_sf_spectrum.setItem(row, 2, QTableWidgetItem(f"{shift_cm1:.1f}"))
            self.tbl_sf_spectrum.setItem(row, 3, QTableWidgetItem(f"{gain:.2e}"))

    def run_overlap_probe(self):
        """Simulate sub-picosecond pulse cross-correlation and nonlinear overlap."""
        pump_tau = self.spin_pulse_duration_fs.value()
        stokes_tau = self.spin_pulse_duration_fs.value()
        delay_fs = self.spin_probe_delay_fs.value()

        res = NonlinearSignalTemporalProbe.probe_ultrafast_overlap(
            pump_tau_fs=pump_tau,
            stokes_tau_fs=stokes_tau,
            delay_fs=delay_fs,
            window_fs=max(200.0, 3.0 * pump_tau),
        )
        self._last_probe_result = res

        # Overlap peak values at tau
        idx_zero = len(res["time_fs"]) // 2
        val_2p = float(res["2p_overlap"][idx_zero])
        val_3p = float(res["3p_overlap"][idx_zero])
        val_srs = float(res["srs_overlap"][idx_zero])

        self.lbl_overlap_metrics.setText(
            f"2P Overlap: {val_2p:.3f} | 3P Overlap: {val_3p:.3f} | SRS Cross-Corr: {val_srs:.3f}"
        )

    def run_pulse_train_probe(self):
        """Simulate nanosecond periodic pulse train modulated by AOM/EOM."""
        rep_mhz = self.spin_rep_rate_mhz.value()
        mod_mhz = self.spin_eom_mod_freq.value()
        dur_ns = self.spin_probe_duration_ns.value()

        res = NonlinearSignalTemporalProbe.probe_modulated_pulse_train(
            rep_rate_mhz=rep_mhz,
            modulation_freq_mhz=mod_mhz,
            duration_ns=dur_ns,
        )
        self._last_probe_result = res

        n_pulses = int(dur_ns / (1000.0 / rep_mhz))
        env_min = float(np.min(res["modulation_envelope"]))
        env_max = float(np.max(res["modulation_envelope"]))
        sig_peak = float(np.max(res["modulated_signal"]))

        self.lbl_train_metrics.setText(
            f"Pulses: {n_pulses} | Envelope: [{env_min:.2f}, {env_max:.2f}] | Peak Signal: {sig_peak:.2f}"
        )

    def _on_srs_scan_changed(self):
        """Update live scan field metrics when galvo angles or parameters change."""
        if hasattr(self, "combo_srs_scan_arch") and hasattr(self, "chk_srs_galvo_fold"):
            if self.combo_srs_scan_arch.currentIndex() == 3 and not self.chk_srs_galvo_fold.isChecked():
                self.chk_srs_galvo_fold.blockSignals(True)
                self.chk_srs_galvo_fold.setChecked(True)
                self.chk_srs_galvo_fold.blockSignals(False)
        cfg = self._get_srs_config_from_gui()
        res = ThorlabsSRSMicroscopeSimulator.simulate_scanning_point(
            cfg, self.spin_srs_galvo_x.value(), self.spin_srs_galvo_y.value()
        )
        conj_str = "Strictly Conjugated (Zero Walk-Off)" if res.is_strictly_conjugated else f"Walk-Off ({res.back_pupil_walk_off_total_um:.1f} µm)"
        fold_str = " | 🪞 90° Reflected Galvo Axis" if getattr(cfg, "enable_galvo_reflection_fold", False) else ""
        self.lbl_srs_pupil_metrics.setText(
            f"Pupil Mag: {res.pupil_magnification:.1f}× | BFP Beam: ∅ {res.back_pupil_beam_diameter_mm:.1f} mm | "
            f"Obj Pupil: ∅ {res.objective_pupil_diameter_mm:.1f} mm (Fill: {res.pupil_fill_factor*100.0:.1f}%) | "
            f"Walk-Off: {res.back_pupil_walk_off_total_um:.1f} µm | Status: {conj_str}{fold_str}"
        )
        self.lbl_srs_scan_metrics.setText(
            f"Sample Pos: ({res.sample_x_um:+.1f}, {res.sample_y_um:+.1f}) µm | "
            f"Chromatic Walk-off: {res.chromatic_walk_off_um:.2f} µm | "
            f"Defocus: {res.field_curvature_defocus_um:.2f} µm | "
            f"Scan Eff: {res.total_scan_efficiency * 100.0:.1f}%"
        )

    def _get_srs_config_from_gui(self) -> ThorlabsSRSMicroscopeConfig:
        oil_map = {
            0: OilSampleType.MINERAL_OIL,
            1: OilSampleType.OLIVE_OIL,
            2: OilSampleType.IMMERSION_OIL,
        }
        oil_type = oil_map.get(self.combo_srs_oil.currentIndex(), OilSampleType.MINERAL_OIL)

        pattern_map = {
            0: "Uniform Film",
            1: "Oil Droplets",
            2: "Grid",
        }
        pattern = pattern_map.get(self.combo_srs_pattern.currentIndex(), "Uniform Film")
        mod_type = "AOM" if self.combo_srs_modulator.currentIndex() == 1 else "EOM"

        arch_map = {
            0: "Close-Coupled (Thorlabs GVS002)",
            1: "4f-Relayed (Zero Walk-Off)",
            2: "Edmund Optics Telecentric",
            3: "Folded Real-World SRS (Dichroic + Reflected Galvos + Steering Mirrors)",
        }
        arch = arch_map.get(self.combo_srs_scan_arch.currentIndex() if hasattr(self, "combo_srs_scan_arch") else 0, "Close-Coupled (Thorlabs GVS002)")
        fold_enabled = self.chk_srs_galvo_fold.isChecked() if hasattr(self, "chk_srs_galvo_fold") else False
        if "fold" in arch.lower():
            fold_enabled = True

        obj_map = {
            0: "Thorlabs TL15X-2P (15x NA 0.70)",
            1: "Olympus XLPLN25XWMP2 (25x NA 1.05 W Multiphoton)",
            2: "Olympus XLUMPLFLN20XW (20x NA 1.00 W)",
            3: "Olympus UPLSAPO60XW (60x NA 1.20 W High-NA)",
            4: "Nikon CFI75 25XC W (25x NA 1.10 W)",
        }
        obj_name = obj_map.get(self.combo_srs_objective.currentIndex() if hasattr(self, "combo_srs_objective") else 0, "Thorlabs TL15X-2P (15x NA 0.70)")

        cfg = ThorlabsSRSMicroscopeConfig(
            pump_laser_wavelength_nm=self.spin_laser_wl_nm.value(),
            pump_laser_power_mw=self.spin_laser_power_mw.value(),
            stokes_laser_wavelength_nm=self.spin_stokes_wl_nm.value(),
            stokes_laser_power_mw=self.spin_laser_power_mw.value() * 0.5,
            modulator_type=mod_type,
            modulator_frequency_mhz=self.spin_srs_mod_freq_mhz.value(),
            modulator_v_pi_v=self.spin_srs_eom_vpi.value(),
            aom_rf_power_w=self.spin_srs_aom_power_w.value(),
            sample_type=oil_type,
            sample_pattern=pattern,
            sample_thickness_um=self.spin_srs_sample_thick_um.value(),
            current_scan_angle_x_deg=self.spin_srs_galvo_x.value(),
            current_scan_angle_y_deg=self.spin_srs_galvo_y.value(),
            lateral_chromatic_coeff=self.spin_srs_chrom_walk.value(),
            petzval_radius_mm=self.spin_srs_petzval_mm.value(),
            lockin_time_constant_us=self.spin_srs_lockin_tc_us.value(),
            lockin_phase_deg=self.spin_srs_lockin_phase_deg.value(),
            current_probe_delay_ps=self.spin_srs_probe_delay_ps.value() if hasattr(self, "spin_srs_probe_delay_ps") else 0.0,
            scanning_architecture=arch,
            enable_galvo_reflection_fold=fold_enabled,
        )
        cfg.apply_objective_preset(obj_name)
        return cfg

    def _on_srs_probe_delay_changed(self):
        """Callback when probe inter-pulse delay is adjusted."""
        d_ps = self.spin_srs_probe_delay_ps.value()
        stage_mm = stage_displacement_from_delay_ps(d_ps, passes=2)
        if hasattr(self, "spin_srs_probe_stage_mm"):
            self.spin_srs_probe_stage_mm.blockSignals(True)
            self.spin_srs_probe_stage_mm.setValue(stage_mm)
            self.spin_srs_probe_stage_mm.blockSignals(False)
        self._update_probe_wavenumber_label(d_ps)

    def _on_srs_probe_stage_changed(self):
        """Callback when probe stage displacement is adjusted."""
        stage_mm = self.spin_srs_probe_stage_mm.value()
        d_ps = delay_from_stage_displacement_mm(stage_mm, passes=2)
        if hasattr(self, "spin_srs_probe_delay_ps"):
            self.spin_srs_probe_delay_ps.blockSignals(True)
            self.spin_srs_probe_delay_ps.setValue(d_ps)
            self.spin_srs_probe_delay_ps.blockSignals(False)
        self._update_probe_wavenumber_label(d_ps)

    def _update_probe_wavenumber_label(self, delay_ps: float):
        cfg = self._get_srs_config_from_gui()
        oil = OilRamanStandard.create(cfg.sample_type)
        gvd_p = glass_gvd_fs2_per_mm(cfg.pump_glass_material, cfg.pump_laser_wavelength_nm)
        gvd_s = glass_gvd_fs2_per_mm(cfg.stokes_glass_material, cfg.stokes_laser_wavelength_nm)
        gdd_p = gvd_p * cfg.pump_glass_length_mm
        gdd_s = gvd_s * cfg.stokes_glass_length_mm
        tau_p0 = cfg.pump_pulse_duration_fs
        tau_s0 = cfg.stokes_pulse_duration_fs
        alpha_p = (4.0 * math.log(2.0) * gdd_p) / ((tau_p0 ** 4) + ((4.0 * math.log(2.0) * gdd_p) ** 2))
        alpha_s = (4.0 * math.log(2.0) * gdd_s) / ((tau_s0 ** 4) + ((4.0 * math.log(2.0) * gdd_s) ** 2))
        avg_alpha = 0.5 * (alpha_p + alpha_s)
        base_omega_cm1 = 1e7 * (1.0 / cfg.pump_laser_wavelength_nm - 1.0 / cfg.stokes_laser_wavelength_nm)
        c_cm_per_fs = 299792458.0 * 100.0 * 1e-15
        delta_nu_shift_cm1 = (2.0 * avg_alpha * (delay_ps * 1e3)) / (2.0 * math.pi * c_cm_per_fs)
        freq_diff_cm1 = base_omega_cm1 + delta_nu_shift_cm1

        min_dist = float("inf")
        nearest_mode = "Off-resonance"
        for p_info in oil.vibrational_peaks:
            center_cm1 = p_info[0]
            desc = p_info[3] if len(p_info) > 3 else f"{center_cm1:.0f} cm⁻¹ mode"
            dist = abs(freq_diff_cm1 - center_cm1)
            if dist < min_dist:
                min_dist = dist
                nearest_mode = desc
        status = "ON-Resonance" if min_dist <= 25.0 else "OFF-Resonance"
        is_dark = getattr(self, 'dark_mode', False)
        on_col = "#60a5fa" if is_dark else "#1565C0"
        off_col = "#94a3b8" if is_dark else "#757575"
        color = on_col if min_dist <= 25.0 else off_col
        self.lbl_srs_probe_wavenumber.setText(
            f"Probed Raman Shift: {freq_diff_cm1:.1f} cm⁻¹ (Δν: {delta_nu_shift_cm1:+.1f} cm⁻¹) | Mode: {nearest_mode} ({status})"
        )
        self.lbl_srs_probe_wavenumber.setStyleSheet(f"font-weight: bold; color: {color};")

    def _set_probe_target_cm1(self, target_cm1: float):
        cfg = self._get_srs_config_from_gui()
        gvd_p = glass_gvd_fs2_per_mm(cfg.pump_glass_material, cfg.pump_laser_wavelength_nm)
        gvd_s = glass_gvd_fs2_per_mm(cfg.stokes_glass_material, cfg.stokes_laser_wavelength_nm)
        gdd_p = gvd_p * cfg.pump_glass_length_mm
        gdd_s = gvd_s * cfg.stokes_glass_length_mm
        tau_p0 = cfg.pump_pulse_duration_fs
        tau_s0 = cfg.stokes_pulse_duration_fs
        alpha_p = (4.0 * math.log(2.0) * gdd_p) / ((tau_p0 ** 4) + ((4.0 * math.log(2.0) * gdd_p) ** 2))
        alpha_s = (4.0 * math.log(2.0) * gdd_s) / ((tau_s0 ** 4) + ((4.0 * math.log(2.0) * gdd_s) ** 2))
        avg_alpha = 0.5 * (alpha_p + alpha_s)
        base_omega_cm1 = 1e7 * (1.0 / cfg.pump_laser_wavelength_nm - 1.0 / cfg.stokes_laser_wavelength_nm)
        c_cm_per_fs = 299792458.0 * 100.0 * 1e-15
        needed_shift = target_cm1 - base_omega_cm1
        needed_delay_fs = (needed_shift * 2.0 * math.pi * c_cm_per_fs) / max(1e-30, 2.0 * avg_alpha)
        needed_delay_ps = float(needed_delay_fs * 1e-3)
        self.spin_srs_probe_delay_ps.setValue(needed_delay_ps)
        self.run_post_sample_probe()

    def run_post_sample_probe(self):
        """Execute post-sample probe observation at the current specific delay and galvo angles."""
        cfg = self._get_srs_config_from_gui()
        delay_ps = self.spin_srs_probe_delay_ps.value()
        angle_x = self.spin_srs_galvo_x.value()
        angle_y = self.spin_srs_galvo_y.value()
        res = ThorlabsSRSMicroscopeSimulator.probe_post_sample_intensity(cfg, delay_ps, angle_x, angle_y)
        self._last_post_sample_probe_result = res

        pd = res.photodiode_signal
        srl_pct = res.fractional_srl * 100.0
        eff_pct = res.scan_field.total_scan_efficiency * 100.0
        res_tag = "ON-Resonance" if res.is_on_resonance else "OFF-Resonance"

        self.lbl_srs_probe_intensity_res.setText(
            f"Post-Sample SRL: {res.fractional_srl:.2e} ({srl_pct:.4f}%) | "
            f"Modulated AC: {res.generated_signal_power_uw:.3f} µW | "
            f"Lock-In: {pd.lockin_magnitude_r_v * 1e6:.2f} µV | "
            f"SNR: {pd.snr_db:.1f} dB ({res_tag}, Scan Eff: {eff_pct:.1f}%)"
        )

    def run_srs_microscope_simulation(self):
        """Simulate SRS Microscope: Oil spectrum, scanning effects, and photodiode signal."""
        cfg = self._get_srs_config_from_gui()
        res = ThorlabsSRSMicroscopeSimulator.simulate_oil_spectrum(
            cfg,
            angle_x_deg=self.spin_srs_galvo_x.value(),
            angle_y_deg=self.spin_srs_galvo_y.value(),
        )
        self._last_srs_microscope_result = res

        sig_on = res["peak_sig_on_axis"]
        sig_off = res["peak_sig_off_axis"]
        scan_off = res["scan_off_axis"]

        # Update photodiode diagnostics
        self.lbl_srs_pd_metrics.setText(
            f"DC: {sig_off.dc_photocurrent_a * 1e3:.2f} mA ({sig_off.dc_output_voltage_v:.1f} V) | "
            f"Lock-in R: {sig_off.lockin_magnitude_r_v * 1e6:.2f} µV | "
            f"Noise Floor: {sig_off.output_noise_voltage_v * 1e6:.3f} µV | "
            f"SNR: {sig_off.snr_db:.1f} dB"
        )

        # Update summary label
        idx_max = int(np.argmax(res["raw_srl_spectrum"]))
        peak_wavenumber = float(res["wavenumbers_cm1"][idx_max])
        rolloff_pct = (1.0 - scan_off.total_scan_efficiency) * 100.0

        self.lbl_srs_summary.setText(
            f"Peak Band: {peak_wavenumber:.1f} cm⁻¹ ({res['sample_name']}) | "
            f"On-Axis: {sig_on.lockin_magnitude_r_v * 1e6:.2f} µV | "
            f"Off-Axis ({scan_off.field_radius_um:.1f} µm): {sig_off.lockin_magnitude_r_v * 1e6:.2f} µV "
            f"(Roll-off: -{rolloff_pct:.1f}%) | "
            f"Lock-in SNR: {sig_off.snr_db:.1f} dB"
        )

        # Populate table
        n_pts = min(100, len(res["delays_ps"]))
        step = max(1, len(res["delays_ps"]) // n_pts)
        indices = list(range(0, len(res["delays_ps"]), step))

        self.tbl_srs_spectrum.setRowCount(len(indices))
        for row, idx in enumerate(indices):
            d_ps = res["delays_ps"][idx]
            pos_mm = res["stage_positions_mm"][idx]
            cm1 = res["wavenumbers_cm1"][idx]
            v_on_uv = res["on_axis_lockin_voltage_v"][idx] * 1e6
            v_off_uv = res["off_axis_lockin_voltage_v"][idx] * 1e6
            eff_pct = scan_off.total_scan_efficiency * 100.0
            snr_val = res["off_axis_snr_db"][idx]

            self.tbl_srs_spectrum.setItem(row, 0, QTableWidgetItem(f"{d_ps:+.2f}"))
            self.tbl_srs_spectrum.setItem(row, 1, QTableWidgetItem(f"{pos_mm:+.3f}"))
            self.tbl_srs_spectrum.setItem(row, 2, QTableWidgetItem(f"{cm1:.1f}"))
            self.tbl_srs_spectrum.setItem(row, 3, QTableWidgetItem(f"{v_on_uv:.2f}"))
            self.tbl_srs_spectrum.setItem(row, 4, QTableWidgetItem(f"{v_off_uv:.2f}"))
            self.tbl_srs_spectrum.setItem(row, 5, QTableWidgetItem(f"{eff_pct:.1f}%"))
            self.tbl_srs_spectrum.setItem(row, 6, QTableWidgetItem(f"{snr_val:.1f}"))

    def _load_stack_into_project(self, stack: Any, log_msg: str, undo_label: str = "Load Optical Layout"):
        """Helper to cleanly load any generated optical stack into the active project."""
        cfg = getattr(stack, "srs_microscope_config", None) or self._get_srs_config_from_gui()
        if self.main_window:
            if hasattr(self.main_window, "undo_manager") and self.main_window.undo_manager:
                self.main_window.undo_manager.push_state(stack, getattr(self.main_window, "system_config", None), undo_label)
            if hasattr(self.main_window, "model") and self.main_window.model:
                if hasattr(self.main_window.model, "set_surface_stack"):
                    self.main_window.model.set_surface_stack(stack)
                self.main_window.model.surface_stack = stack
            if hasattr(self.main_window, "system_config") and self.main_window.system_config:
                self.main_window.system_config.wavelength_list_um = [cfg.pump_laser_wavelength_nm * 1e-3, cfg.stokes_laser_wavelength_nm * 1e-3]
                self.main_window.system_config.primary_wavelength_um = cfg.pump_laser_wavelength_nm * 1e-3
                if hasattr(self.main_window, "system_config_dock") and self.main_window.system_config_dock:
                    self.main_window.system_config_dock.update_ui()
            if hasattr(self.main_window, "component_editor_dock") and self.main_window.component_editor_dock:
                self.main_window.component_editor_dock.update_ui()
            if hasattr(self.main_window, "lde_table") and self.main_window.lde_table:
                if hasattr(self.main_window.lde_table, "rebuild_from_stack"):
                    self.main_window.lde_table.rebuild_from_stack()
            # Reset 2D layout zoom to auto-fit the full folded microscope beam path
            if hasattr(self.main_window, "dual_view") and self.main_window.dual_view:
                if hasattr(self.main_window.dual_view, "reset_view_limits"):
                    self.main_window.dual_view.reset_view_limits()
                elif hasattr(self.main_window.dual_view, "saved_limits"):
                    self.main_window.dual_view.saved_limits.clear()
                    self.main_window.dual_view._force_reset_view = True
            # Immediately trigger 2D ray tracing through all surfaces
            if hasattr(self.main_window, "on_trace_rays"):
                self.main_window.on_trace_rays()
            # Switch to 2D Projections layout viewer tab if viewer_tabs exists
            if hasattr(self.main_window, "viewer_tabs") and hasattr(self.main_window, "dual_view"):
                self.main_window.viewer_tabs.setCurrentWidget(self.main_window.dual_view)
            elif hasattr(self.main_window, "visualizer_3d") and self.main_window.visualizer_3d:
                self.main_window.visualizer_3d.update_visualization()
            if hasattr(self.main_window, "statusBar") and self.main_window.statusBar():
                self.main_window.statusBar().showMessage(log_msg, 5000)

    def load_thorlabs_lsm_template_into_project(self):
        """Loads the pristine 17-surface Thorlabs 4f Laser Scanning Microscope layout into the active project."""
        cfg = self._get_srs_config_from_gui()
        stack = build_thorlabs_lsm_template(cfg)
        self._load_stack_into_project(
            stack,
            "Loaded Thorlabs Laser Scanning Microscope (17 Surfaces, 12 Components) into 2D Optics layout & traced rays",
            "Load Thorlabs 4f LSM Template"
        )

    def load_srs_microscope_template_into_project(self):
        """Loads the full Thorlabs SRS Laser Scanning Microscope layout into the active project."""
        cfg = self._get_srs_config_from_gui()
        cfg.scanning_architecture = "Close-Coupled (Thorlabs GVS002)"
        stack = build_srs_microscope_template(cfg)
        self._load_stack_into_project(
            stack,
            f"Loaded Thorlabs Close-Coupled SRS Microscope ({len(stack.surfaces)} Surfaces, {len(stack.components)} Components) into 2D Optics layout & traced rays",
            "Load Thorlabs Close-Coupled SRS Template"
        )

    def load_relayed_srs_template_into_project(self):
        """Loads the 4f-Relayed SRS Laser Scanning Microscope layout with strict zero pupil walk-off."""
        cfg = self._get_srs_config_from_gui()
        cfg.scanning_architecture = "4f-Relayed (Zero Walk-Off)"
        stack = build_srs_microscope_relayed_template(cfg)
        self._load_stack_into_project(
            stack,
            f"Loaded 4f-Relayed SRS Microscope ({len(stack.surfaces)} Surfaces, Zero Pupil Walk-Off) into 2D Optics layout & traced rays",
            "Load 4f-Relayed SRS Template"
        )

    def load_olympus_high_na_template_into_project(self):
        """Loads the SRS Microscope layout featuring Olympus High Numerical Aperture immersion optics."""
        cfg = self._get_srs_config_from_gui()
        if "olympus" not in cfg.objective_model.lower():
            cfg.apply_objective_preset("Olympus XLPLN25XWMP2 (25x NA 1.05 W Multiphoton)")
        stack = build_olympus_high_na_srs_template(cfg)
        self._load_stack_into_project(
            stack,
            f"Loaded Olympus High-NA SRS Microscope ({cfg.objective_model}, {len(stack.surfaces)} Surfaces) into 2D Optics layout & traced rays",
            "Load Olympus High-NA SRS Template"
        )

    def load_edmund_optics_srs_template_into_project(self):
        """Loads the Edmund Optics telecentric SRS Laser Scanning Microscope layout."""
        cfg = self._get_srs_config_from_gui()
        cfg.scanning_architecture = "Edmund Optics Telecentric"
        stack = build_edmund_optics_srs_template(cfg)
        self._load_stack_into_project(
            stack,
            f"Loaded Edmund Optics Telecentric SRS Microscope ({len(stack.surfaces)} Surfaces) into 2D Optics layout & traced rays",
            "Load Edmund Optics SRS Template"
        )

    def load_non_sequential_srs_into_project(self):
        """Loads the true Non-Sequential Dual-Beam SRS Microscope layout (Pump + Stokes EOM + Delay Stage + Dichroic Combiner + Galvos + PD)."""
        cfg = self._get_srs_config_from_gui()
        cfg.scanning_architecture = "Non-Sequential Dual-Laser SRS Microscope"
        res = NonSequentialSRSMicroscopeBench.simulate_non_sequential_srs(
            config=cfg,
            scan_angle_x_deg=self.spin_srs_galvo_x.value(),
            scan_angle_y_deg=self.spin_srs_galvo_y.value(),
            delay_ps=self.spin_srs_probe_delay_ps.value() if hasattr(self, "spin_srs_probe_delay_ps") else 0.0,
            num_rays_per_beam=7,
        )
        self._last_non_sequential_srs_result = res
        stack = res.surface_stack
        self._load_stack_into_project(
            stack,
            f"Loaded Non-Sequential Dual-Laser SRS Microscope ({len(res.non_sequential_system.elements)} Elements, Dual Laser Inputs, Dichroic Combiner, EOM, Delay Stage) into 2D Optics layout & traced {len(res.all_traced_rays)} rays",
            "Load Non-Sequential SRS Microscope"
        )

    def load_folded_real_world_srs_template_into_project(self):
        """Loads the realistic Folded Real-World SRS Microscope layout with 90° reflected galvos, steering mirrors, EOM, and dichroic fs combiner."""
        cfg = self._get_srs_config_from_gui()
        cfg.scanning_architecture = "Folded Real-World SRS (Dichroic + Reflected Galvos + Steering Mirrors)"
        cfg.enable_galvo_reflection_fold = True
        try:
            from optics_app.analysis.ultrafast_nonlinear import build_folded_real_world_srs_template
        except ImportError:
            from v2.analysis.ultrafast_nonlinear import build_folded_real_world_srs_template
        stack = build_folded_real_world_srs_template(cfg)
        self._load_stack_into_project(
            stack,
            f"Loaded Folded Real-World SRS Microscope (Power Control + EOM + Dichroic + 4f-Relayed Galvos + Olympus NA 1.05 W + χ⁽³⁾, {len(stack.surfaces)} Surfaces) into 2D Optics layout & traced rays",
            "Load Folded Real-World SRS Template"
        )

    load_real_srs_microscope_template_into_project = load_folded_real_world_srs_template_into_project

    def save_srs_microscope_preset_dialog(self):
        """Saves current SRS microscope configuration to a JSON file."""
        cfg = self._get_srs_config_from_gui()
        default_path = os.path.join(os.getcwd(), "thorlabs_srs_microscope.json")
        filepath = default_path
        if QT_AVAILABLE:
            try:
                fn, _ = QFileDialog.getSaveFileName(self, "Save SRS Microscope Preset JSON", default_path, "JSON Files (*.json)")
                if fn:
                    filepath = fn
            except Exception:
                pass
        save_srs_microscope_json(filepath, cfg)
        if self.main_window and hasattr(self.main_window, "statusBar") and self.main_window.statusBar():
            self.main_window.statusBar().showMessage(f"Saved SRS Microscope preset to {os.path.basename(filepath)}", 5000)

    def load_srs_microscope_preset_dialog(self):
        """Loads an SRS microscope prescription from a JSON file."""
        default_path = os.path.join(os.getcwd(), "thorlabs_srs_microscope.json")
        filepath = default_path
        if QT_AVAILABLE:
            try:
                fn, _ = QFileDialog.getOpenFileName(self, "Load SRS Microscope Preset JSON", default_path, "JSON Files (*.json)")
                if fn:
                    filepath = fn
            except Exception:
                pass
        if os.path.exists(filepath):
            if self.main_window and hasattr(self.main_window, "on_load_project"):
                self.main_window.on_load_project(filepath)
            else:
                sys = load_srs_microscope_json(filepath)
                if self.main_window and hasattr(self.main_window, "model"):
                    self.main_window.model.set_surface_stack(sys.surface_stack)
                if self.main_window and hasattr(self.main_window, "on_trace_rays"):
                    self.main_window.on_trace_rays()
            stack = self.main_window.model.get_surface_stack() if (self.main_window and hasattr(self.main_window, "model")) else None
            cfg = getattr(stack, "srs_microscope_config", None) if stack else None
            if cfg is not None:
                self.spin_srs_galvo_x.setValue(cfg.current_scan_angle_x_deg)
                self.spin_srs_galvo_y.setValue(cfg.current_scan_angle_y_deg)
                self.spin_srs_mod_freq_mhz.setValue(cfg.modulator_frequency_mhz)
                self.spin_srs_sample_thick_um.setValue(cfg.sample_thickness_um)
                self.run_srs_microscope_simulation()
            if self.main_window and hasattr(self.main_window, "statusBar") and self.main_window.statusBar():
                self.main_window.statusBar().showMessage(f"Loaded SRS Microscope preset from {os.path.basename(filepath)}", 5000)
