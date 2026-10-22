"""
Zemax-Style Tolerance Analysis Dock Widget for 2D Optics GUI.
============================================================
Provides an interactive GUI interface for:
- Defining Zemax tolerance operands (TRAD, TCUR, TTHI, TSDX, TSDY, TSTX, TSTY, TEDX, TEDY, TIND, TABB, TCON, TIRR).
- Applying standard industry presets (Commercial, Precision, High-Precision).
- Automatic compensator optimization (refocusing image plane / BFL).
- Running univariate Sensitivity Analysis with RSS & Worst-Case estimations (in µm and mm).
- Running Zemax Inverse Sensitivity Analysis to solve for maximum allowable tolerances given a target degradation (µm) and applying solved limits back to the TDE table.
- Running multivariate Monte Carlo simulations on a background QThread.
- Interactive visual charts: Worst-Offenders Pareto Bar Chart (µm) and Monte Carlo Histogram + Cumulative Yield Curve (CDF in µm).
- Interactive Yield Rate analysis for user-specified performance thresholds (µm).
- Exporting Zemax OpticStudio ASCII reports (.txt) and .zmx prescriptions.
"""

import os
import traceback
from typing import Optional, List, Dict, Any, Union
import numpy as np

try:
    from v2.gui.qt_compat import (
        QDockWidget, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
        QComboBox, QDoubleSpinBox, QSpinBox, QTableWidget, QTableWidgetItem, QHeaderView,
        Qt, QT_AVAILABLE, QThread, pyqtSignal, QScrollArea, QAbstractItemView, QtWidgets,
        QCheckBox, QFileDialog, QMessageBox, QGroupBox, QTabWidget
    )
    from v2.gui.lde_table import LensDataTableModel
    from v2.engine.tolerancing import (
        TolerancingEngine,
        PerturbationSpec,
        ZemaxToleranceOperand,
        ToleranceOperandType,
        ToleranceDistribution,
        TolerancingCriterion,
        SensitivityReport,
        InverseSensitivityReport,
        MonteCarloReport,
        create_commercial_tolerances,
        create_precision_tolerances,
        create_high_precision_tolerances,
        export_zemax_tolerance_report,
        export_zmx_with_tolerances,
    )
except ImportError:
    from optics_app.gui.qt_compat import (
        QDockWidget, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
        QComboBox, QDoubleSpinBox, QSpinBox, QTableWidget, QTableWidgetItem, QHeaderView,
        Qt, QT_AVAILABLE, QThread, pyqtSignal, QScrollArea, QAbstractItemView, QtWidgets,
        QCheckBox, QFileDialog, QMessageBox, QGroupBox, QTabWidget
    )
    from optics_app.gui.lde_table import LensDataTableModel
    from optics_app.engine.tolerancing import (
        TolerancingEngine,
        PerturbationSpec,
        ZemaxToleranceOperand,
        ToleranceOperandType,
        ToleranceDistribution,
        TolerancingCriterion,
        SensitivityReport,
        InverseSensitivityReport,
        MonteCarloReport,
        create_commercial_tolerances,
        create_precision_tolerances,
        create_high_precision_tolerances,
        export_zemax_tolerance_report,
        export_zmx_with_tolerances,
    )

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    MPL_AVAILABLE = True
except Exception:
    MPL_AVAILABLE = False

try:
    from optics_app.gui.theme import get_theme_colors, apply_axes_theme
except ImportError:
    try:
        from v2.gui.theme import get_theme_colors, apply_axes_theme
    except ImportError:
        def get_theme_colors(dark_mode=False):
            return {
                "bg": "#0f172a" if dark_mode else "#ffffff",
                "text": "#f1f5f9" if dark_mode else "#0f172a",
                "text_dim": "#94a3b8" if dark_mode else "#64748b",
                "axis_line": "#334155" if dark_mode else "#cbd5e1",
                "grid": "#334155" if dark_mode else "#e2e8f0",
            }
        def apply_axes_theme(ax, dark_mode=False):
            pass


class TolerancingWorker(QThread if QT_AVAILABLE else object):
    """Background worker thread to execute tolerancing without blocking the GUI."""
    if QT_AVAILABLE:
        result_ready = pyqtSignal(object, str)  # (report_obj, report_text)
        error_occurred = pyqtSignal(str)

    def __init__(
        self,
        system: Any,
        tolerances: List[PerturbationSpec],
        criterion: str = "rms_spot_radius",
        is_monte_carlo: bool = False,
        mode: Optional[str] = None,
        allowable_degradation_um: float = 5.0,
        trials: int = 500,
        seed: Optional[int] = None,
        compensator_enabled: bool = True,
        compensator_surface_idx: int = -2,
        compensator_range_mm: float = 2.0,
        parent=None
    ):
        if QT_AVAILABLE:
            super().__init__(parent)
        self.system = system
        self.tolerances = tolerances
        self.criterion = criterion
        self.is_monte_carlo = is_monte_carlo
        self.mode = mode if mode is not None else ("monte_carlo" if is_monte_carlo else "sensitivity")
        self.allowable_degradation_um = float(allowable_degradation_um)
        self.trials = trials
        self.seed = seed
        self.compensator_enabled = compensator_enabled
        self.compensator_surface_idx = compensator_surface_idx
        self.compensator_range_mm = compensator_range_mm

    def run(self):
        try:
            engine = TolerancingEngine(
                system=self.system,
                criterion=self.criterion,
                compensator_enabled=self.compensator_enabled,
                compensator_surface_idx=self.compensator_surface_idx,
                compensator_range_mm=self.compensator_range_mm,
                seed=self.seed,
            )

            if self.mode == "monte_carlo" or self.is_monte_carlo:
                report = engine.monte_carlo_simulation(
                    tolerances=self.tolerances,
                    num_trials=self.trials,
                )
                text = report.format_report()
            elif self.mode == "inverse":
                report = engine.inverse_sensitivity_analysis(
                    tolerances=self.tolerances,
                    allowable_degradation_um=self.allowable_degradation_um,
                )
                text = report.format_table()
            else:
                report = engine.sensitivity_analysis(
                    tolerances=self.tolerances,
                )
                text = report.format_table()

            if hasattr(self, "result_ready"):
                self.result_ready.emit(report, text)

        except Exception:
            if hasattr(self, "error_occurred"):
                self.error_occurred.emit(f"Error during tolerancing:\n{traceback.format_exc()}")


class TolerancingDockWidget(QDockWidget if QT_AVAILABLE else object):
    """Interactive Zemax-Style Tolerancing Dock for 2D Optics."""

    def __init__(self, model: Optional[LensDataTableModel] = None, parent=None):
        if QT_AVAILABLE:
            super().__init__("Zemax Tolerance Analysis", parent)
            self.setObjectName("TolerancingDockWidget")
            self.setAllowedAreas(Qt.AllDockWidgetAreas)

        self.model = model
        self.dark_mode: bool = False
        self.last_sensitivity_report: Optional[SensitivityReport] = None
        self.last_inverse_report: Optional[InverseSensitivityReport] = None
        self.last_monte_carlo_report: Optional[MonteCarloReport] = None
        self.worker: Optional[TolerancingWorker] = None

        self._setup_ui()

    def set_dark_mode(self, enabled: bool) -> None:
        """Update tolerancing dock styling between dark and light themes."""
        self.dark_mode = bool(enabled)
        self._update_theme_styles()
        if hasattr(self, "figure") and self.figure:
            theme = get_theme_colors(self.dark_mode)
            self.figure.patch.set_facecolor(theme["bg"])
            if self.last_monte_carlo_report:
                self._plot_monte_carlo_chart(self.last_monte_carlo_report)
            elif self.last_inverse_report:
                self._plot_inverse_sensitivity_chart(self.last_inverse_report)
            elif self.last_sensitivity_report:
                self._plot_sensitivity_chart(self.last_sensitivity_report)
            else:
                self._draw_placeholder_chart()

    def _update_theme_styles(self) -> None:
        if not QT_AVAILABLE:
            return
        is_dark = getattr(self, "dark_mode", False)
        if hasattr(self, "output_text") and self.output_text:
            if is_dark:
                self.output_text.setStyleSheet(
                    "font-family: Consolas, 'Courier New', monospace; font-size: 11px; "
                    "background-color: #0b0f19; color: #f1f5f9; border: 1px solid #1e293b;"
                )
            else:
                self.output_text.setStyleSheet(
                    "font-family: Consolas, 'Courier New', monospace; font-size: 11px; "
                    "background-color: #fafafa; color: #222222; border: 1px solid #cbd5e1;"
                )
        if hasattr(self, "btn_sensitivity") and self.btn_sensitivity:
            if is_dark:
                self.btn_sensitivity.setStyleSheet("font-weight: bold; background-color: #0c4a6e; color: #38bdf8; padding: 6px; border-radius: 3px;")
            else:
                self.btn_sensitivity.setStyleSheet("font-weight: bold; background-color: #e3f2fd; color: #0f172a; padding: 6px; border-radius: 3px;")
        if hasattr(self, "btn_inverse") and self.btn_inverse:
            if is_dark:
                self.btn_inverse.setStyleSheet("font-weight: bold; background-color: #451a03; color: #fb923c; padding: 6px; border-radius: 3px;")
            else:
                self.btn_inverse.setStyleSheet("font-weight: bold; background-color: #fff3e0; color: #0f172a; padding: 6px; border-radius: 3px;")
        if hasattr(self, "btn_monte_carlo") and self.btn_monte_carlo:
            if is_dark:
                self.btn_monte_carlo.setStyleSheet("font-weight: bold; background-color: #064e3b; color: #4ade80; padding: 6px; border-radius: 3px;")
            else:
                self.btn_monte_carlo.setStyleSheet("font-weight: bold; background-color: #e8f5e9; color: #0f172a; padding: 6px; border-radius: 3px;")

    def _get_active_system(self) -> Any:
        """Resolve active optical system from model."""
        if self.model is not None:
            if hasattr(self.model, "get_surface_stack") and callable(self.model.get_surface_stack):
                stack = self.model.get_surface_stack()
                if stack is not None and hasattr(stack, "surfaces") and len(stack.surfaces) > 0:
                    return stack
            if hasattr(self.model, "_stack") and self.model._stack is not None:
                return self.model._stack
            if hasattr(self.model, "system") and self.model.system is not None:
                return self.model.system
        return None

    def _setup_ui(self):
        if not QT_AVAILABLE:
            return

        container = QWidget(self)
        main_layout = QVBoxLayout(container)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # ---------------------------------------------------------------------
        # 1. Preset & Quick Configuration Toolbar
        # ---------------------------------------------------------------------
        preset_group = QGroupBox("Tolerance Quality Presets (Zemax / ISO 10110 Standards)", container)
        preset_layout = QHBoxLayout(preset_group)

        self.preset_combo = QComboBox(preset_group)
        self.preset_combo.addItems([
            "Select Quality Preset...",
            "Commercial Grade (Thorlabs/Edmund Std: +/-0.2% R, +/-0.05mm t)",
            "Precision Grade (Research: +/-0.1% R, +/-0.02mm t)",
            "High-Precision Grade (Laser: +/-0.05% R, +/-0.01mm t)",
        ])
        preset_layout.addWidget(self.preset_combo, stretch=2)

        self.btn_apply_preset = QPushButton("Auto-Populate System", preset_group)
        self.btn_apply_preset.clicked.connect(self._apply_preset_clicked)
        preset_layout.addWidget(self.btn_apply_preset, stretch=1)

        self.btn_clear_table = QPushButton("Clear", preset_group)
        self.btn_clear_table.clicked.connect(self._clear_table)
        preset_layout.addWidget(self.btn_clear_table)

        main_layout.addWidget(preset_group)

        # ---------------------------------------------------------------------
        # 2. Tolerancing Table (Tolerance Data Editor)
        # ---------------------------------------------------------------------
        table_group = QGroupBox("Tolerance Data Editor (TDE Operands)", container)
        table_layout = QVBoxLayout(table_group)

        self.table = QTableWidget(0, 6, table_group)
        self.table.setHorizontalHeaderLabels([
            "Operand", "Surf / Elem", "Delta (+/-)", "Distribution", "Min Delta", "Comment"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(5, QHeaderView.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setMinimumHeight(140)
        table_layout.addWidget(self.table)

        btn_row_layout = QHBoxLayout()
        self.btn_add = QPushButton("+ Add Operand", table_group)
        self.btn_add.clicked.connect(self._add_row)
        self.btn_remove = QPushButton("- Remove Selected", table_group)
        self.btn_remove.clicked.connect(self._remove_row)
        self.btn_apply_solved = QPushButton("Apply Solved Inverse Limits to TDE", table_group)
        self.btn_apply_solved.setToolTip(
            "Overwrite TDE Delta (+/-) values with the Tightest Solved Limits from the last Inverse Sensitivity run."
        )
        self.btn_apply_solved.setEnabled(False)
        self.btn_apply_solved.clicked.connect(self._apply_solved_limits_to_table)
        btn_row_layout.addWidget(self.btn_add)
        btn_row_layout.addWidget(self.btn_remove)
        btn_row_layout.addWidget(self.btn_apply_solved)
        btn_row_layout.addStretch()
        table_layout.addLayout(btn_row_layout)

        main_layout.addWidget(table_group)

        # ---------------------------------------------------------------------
        # 3. Analysis Settings & Compensator Options
        # ---------------------------------------------------------------------
        settings_group = QGroupBox("Simulation Criteria, Inverse Target & Compensators", container)
        settings_layout = QHBoxLayout(settings_group)

        # Criterion Selector
        crit_vbox = QVBoxLayout()
        crit_vbox.addWidget(QLabel("<b>Performance Criterion:</b>"))
        self.crit_combo = QComboBox(settings_group)
        self.crit_combo.addItems([
            "RMS Spot Radius (TRAC / RSCE) [µm]",
            "Geometric Spot Radius (GEOR) [µm]",
            "Effective Focal Length (EFFL) [mm]",
            "Back Focal Length (BFL) [mm]",
            "Merit Function",
        ])
        crit_vbox.addWidget(self.crit_combo)

        inv_h = QHBoxLayout()
        inv_h.addWidget(QLabel("Inverse Allow. \u0394 (\u03bcm):"))
        self.inv_deg_spin = QDoubleSpinBox(settings_group)
        self.inv_deg_spin.setDecimals(2)
        self.inv_deg_spin.setRange(0.05, 5000.0)
        self.inv_deg_spin.setValue(5.00)
        self.inv_deg_spin.setSingleStep(1.0)
        self.inv_deg_spin.setToolTip(
            "Target maximum criterion degradation (\u03bcm for spot size) allowed per operand in Inverse Sensitivity mode."
        )
        inv_h.addWidget(self.inv_deg_spin)
        crit_vbox.addLayout(inv_h)
        settings_layout.addLayout(crit_vbox, stretch=1)

        # Compensator (Auto-Refocus)
        comp_vbox = QVBoxLayout()
        self.chk_compensator = QCheckBox("Auto-Refocus (Zemax COMP)", settings_group)
        self.chk_compensator.setChecked(True)
        comp_vbox.addWidget(self.chk_compensator)

        comp_h = QHBoxLayout()
        comp_h.addWidget(QLabel("Range (+/- mm):"))
        self.comp_range_spin = QDoubleSpinBox(settings_group)
        self.comp_range_spin.setRange(0.1, 50.0)
        self.comp_range_spin.setValue(2.0)
        self.comp_range_spin.setSingleStep(0.5)
        comp_h.addWidget(self.comp_range_spin)
        comp_vbox.addLayout(comp_h)
        settings_layout.addLayout(comp_vbox, stretch=1)

        # Monte Carlo Settings
        mc_vbox = QVBoxLayout()
        mc_h1 = QHBoxLayout()
        mc_h1.addWidget(QLabel("MC Trials:"))
        self.trials_spin = QSpinBox(settings_group)
        self.trials_spin.setRange(10, 50000)
        self.trials_spin.setValue(200)
        self.trials_spin.setSingleStep(50)
        mc_h1.addWidget(self.trials_spin)
        mc_vbox.addLayout(mc_h1)

        mc_h2 = QHBoxLayout()
        mc_h2.addWidget(QLabel("Seed:"))
        self.seed_spin = QSpinBox(settings_group)
        self.seed_spin.setRange(0, 999999)
        self.seed_spin.setValue(42)
        mc_h2.addWidget(self.seed_spin)
        mc_vbox.addLayout(mc_h2)
        settings_layout.addLayout(mc_vbox, stretch=1)

        main_layout.addWidget(settings_group)

        # ---------------------------------------------------------------------
        # 4. Action Buttons (Run & Export)
        # ---------------------------------------------------------------------
        action_layout = QHBoxLayout()

        self.btn_sensitivity = QPushButton("Run Sensitivity (RSS)", container)
        self.btn_sensitivity.setStyleSheet("font-weight: bold; background-color: #e3f2fd; padding: 6px;")
        self.btn_sensitivity.clicked.connect(self._run_sensitivity)
        action_layout.addWidget(self.btn_sensitivity)

        self.btn_inverse = QPushButton("Run Inverse Sensitivity", container)
        self.btn_inverse.setStyleSheet("font-weight: bold; background-color: #fff3e0; padding: 6px;")
        self.btn_inverse.clicked.connect(self._run_inverse_sensitivity)
        action_layout.addWidget(self.btn_inverse)

        self.btn_monte_carlo = QPushButton("Run Monte Carlo", container)
        self.btn_monte_carlo.setStyleSheet("font-weight: bold; background-color: #e8f5e9; padding: 6px;")
        self.btn_monte_carlo.clicked.connect(self._run_monte_carlo)
        action_layout.addWidget(self.btn_monte_carlo)

        self.btn_export_report = QPushButton("Export Report (.txt)", container)
        self.btn_export_report.clicked.connect(self._export_report_clicked)
        action_layout.addWidget(self.btn_export_report)

        self.btn_export_zmx = QPushButton("Export .ZMX", container)
        self.btn_export_zmx.clicked.connect(self._export_zmx_clicked)
        action_layout.addWidget(self.btn_export_zmx)

        main_layout.addLayout(action_layout)

        # ---------------------------------------------------------------------
        # 5. Yield Calculator Bar (in µm for spot criteria)
        # ---------------------------------------------------------------------
        yield_box = QHBoxLayout()
        yield_box.addWidget(QLabel("<b>Yield Spec Target (\u03bcm for Spot / mm for EFL):</b>"))
        self.yield_thresh_spin = QDoubleSpinBox(container)
        self.yield_thresh_spin.setDecimals(2)
        self.yield_thresh_spin.setRange(0.01, 100000.0)
        self.yield_thresh_spin.setValue(50.00)
        self.yield_thresh_spin.setSingleStep(5.0)
        yield_box.addWidget(self.yield_thresh_spin)

        self.btn_calc_yield = QPushButton("Calculate Yield %", container)
        self.btn_calc_yield.clicked.connect(self._calculate_yield_clicked)
        yield_box.addWidget(self.btn_calc_yield)

        self.lbl_yield_result = QLabel("Yield: --- %")
        self.lbl_yield_result.setStyleSheet("font-weight: bold; color: #1565c0; font-size: 13px;")
        yield_box.addWidget(self.lbl_yield_result)
        yield_box.addStretch()

        main_layout.addLayout(yield_box)

        # ---------------------------------------------------------------------
        # 6. Results Display Tabs (Visual Charts + ASCII Report)
        # ---------------------------------------------------------------------
        self.results_tabs = QTabWidget(container)

        # Tab 1: Visual Charts (Pareto & Monte Carlo Yield CDF)
        self.chart_container = QWidget(self.results_tabs)
        chart_layout = QVBoxLayout(self.chart_container)
        chart_layout.setContentsMargins(2, 2, 2, 2)
        if MPL_AVAILABLE:
            self.figure = Figure(figsize=(7.5, 3.4), dpi=100)
            self.figure.patch.set_facecolor("#fafafa")
            self.canvas = FigureCanvas(self.figure)
            self.canvas.setMinimumHeight(240)
            chart_layout.addWidget(self.canvas)
            self._draw_placeholder_chart()
        else:
            chart_layout.addWidget(QLabel("Matplotlib not available for visual charts."))
        self.results_tabs.addTab(self.chart_container, "Visual Charts (Pareto \u03bcm & Yield CDF)")

        # Tab 2: Zemax ASCII Report
        self.output_text = QtWidgets.QTextEdit(self.results_tabs)
        self.output_text.setReadOnly(True)
        self.output_text.setLineWrapMode(QtWidgets.QTextEdit.NoWrap)
        self.output_text.setMinimumHeight(240)
        self.results_tabs.addTab(self.output_text, "Zemax OpticStudio ASCII Report")

        main_layout.addWidget(self.results_tabs)
        self._update_theme_styles()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(container)
        self.setWidget(scroll)

        # Default sample row
        self._add_row(operand="TRAD", surf_idx=1, delta=0.1, dist="gaussian", comment="Radius 0.2%")

    def _draw_placeholder_chart(self):
        if not MPL_AVAILABLE:
            return
        theme = get_theme_colors(self.dark_mode)
        self.figure.clear()
        self.figure.patch.set_facecolor(theme["bg"])
        ax = self.figure.add_subplot(111)
        apply_axes_theme(ax, self.dark_mode)
        ax.text(
            0.5, 0.5,
            "Run Sensitivity, Inverse Sensitivity, or Monte Carlo\nto view Worst-Offenders Pareto (\u03bcm) & Yield CDF Charts",
            ha="center", va="center", fontsize=10, color=theme["text_dim"]
        )
        ax.set_xticks([])
        ax.set_yticks([])
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _plot_sensitivity_chart(self, report: SensitivityReport):
        if not MPL_AVAILABLE:
            return
        theme = get_theme_colors(self.dark_mode)
        self.figure.clear()
        self.figure.patch.set_facecolor(theme["bg"])
        ax = self.figure.add_subplot(111)

        worst = report.worst_offenders[:10]
        if not worst:
            self._draw_placeholder_chart()
            return

        is_spot = report.is_spot_criterion
        scale = 1000.0 if is_spot else 1.0
        unit = "\u03bcm" if is_spot else "mm"

        labels = [
            f"S{e.surface_index if e.surface_index is not None else e.element_index} {e.operand_type} (\u00b1{e.delta:.3g})"
            for e in reversed(worst)
        ]
        vals = [e.max_degradation * scale for e in reversed(worst)]

        bars = ax.barh(labels, vals, color="#38bdf8" if self.dark_mode else "#1976d2", alpha=0.85, edgecolor="#0284c7" if self.dark_mode else "#0d47a1")
        ax.set_xlabel(f"Criterion Degradation \u0394 ({unit})", fontsize=9, fontweight="bold")
        nom_val = report.nominal_merit * scale
        rss_val = report.estimated_rss * scale
        wc_val = report.estimated_worst_case * scale
        ax.set_title(
            f"Zemax Top Worst Offenders ({unit}) | Nominal: {nom_val:.2f} {unit} \u2192 Est. RSS: {rss_val:.2f} {unit} (WC: {wc_val:.2f} {unit})",
            fontsize=9.5, fontweight="bold"
        )
        ax.grid(axis="x", linestyle="--", alpha=0.4)
        ax.tick_params(axis="both", labelsize=8)

        max_v = max(vals) if vals else 1.0
        if max_v > 0:
            ax.set_xlim(0, max_v * 1.18)
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_width() + max_v * 0.015,
                bar.get_y() + bar.get_height() / 2.0,
                f"+{v:.2f} {unit}",
                va="center", ha="left", fontsize=8, color=theme["text"]
            )

        apply_axes_theme(ax, self.dark_mode)
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _plot_inverse_sensitivity_chart(self, report: InverseSensitivityReport):
        if not MPL_AVAILABLE:
            return
        theme = get_theme_colors(self.dark_mode)
        self.figure.clear()
        self.figure.patch.set_facecolor(theme["bg"])
        ax = self.figure.add_subplot(111)

        entries = sorted(
            report.entries,
            key=lambda e: (e.solved_symmetric_delta / max(1e-9, e.initial_delta))
        )[:10]
        if not entries:
            self._draw_placeholder_chart()
            return

        is_spot = report.is_spot_criterion
        target_disp = f"{report.target_degradation * 1000.0:.2f} \u03bcm" if is_spot else f"{report.target_degradation:.5f}"

        labels = [
            f"S{e.surface_index if e.surface_index is not None else e.element_index} {e.operand_type}"
            for e in reversed(entries)
        ]
        ratios = [
            (e.solved_symmetric_delta / max(1e-9, e.initial_delta)) * 100.0
            for e in reversed(entries)
        ]
        solved_vals = [e.solved_symmetric_delta for e in reversed(entries)]
        init_vals = [e.initial_delta for e in reversed(entries)]

        colors = ["#ef4444" if r < 80.0 else ("#f59e0b" if r < 120.0 else "#22c55e") for r in ratios]
        bars = ax.barh(labels, ratios, color=colors, alpha=0.85)
        ax.axvline(100.0, color="#94a3b8" if self.dark_mode else "#333333", linestyle="--", linewidth=1.0, label="Initial Spec (100%)")
        ax.set_xlabel("Solved Limit as % of Initial Tolerance (<100% = Must Tighten)", fontsize=8.5, fontweight="bold")
        ax.set_title(
            f"Zemax Inverse Sensitivity Solved Limits (Target Allow. Degradation: +{target_disp})",
            fontsize=9.5, fontweight="bold"
        )
        ax.grid(axis="x", linestyle="--", alpha=0.4)
        ax.tick_params(axis="both", labelsize=8)

        max_r = max(ratios + [110.0])
        ax.set_xlim(0, max_r * 1.25)
        for bar, s_val, i_val in zip(bars, solved_vals, init_vals):
            ax.text(
                bar.get_width() + max_r * 0.015,
                bar.get_y() + bar.get_height() / 2.0,
                f"\u00b1{s_val:.4g} (init \u00b1{i_val:.4g})",
                va="center", ha="left", fontsize=7.5, color=theme["text"]
            )

        apply_axes_theme(ax, self.dark_mode)
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _plot_monte_carlo_chart(self, report: MonteCarloReport):
        if not MPL_AVAILABLE:
            return
        theme = get_theme_colors(self.dark_mode)
        self.figure.clear()
        self.figure.patch.set_facecolor(theme["bg"])
        if not report.trials:
            self._draw_placeholder_chart()
            return

        is_spot = report.is_spot_criterion
        scale = 1000.0 if is_spot else 1.0
        unit = "\u03bcm" if is_spot else "mm"

        merits_scaled = np.array([t.merit * scale for t in report.trials], dtype=float)
        nom_scaled = report.nominal_merit * scale
        p50_scaled = report.p50 * scale
        p90_scaled = report.p90 * scale
        p99_scaled = report.p99 * scale

        ax_hist = self.figure.add_subplot(1, 2, 1)
        ax_cdf = self.figure.add_subplot(1, 2, 2)

        # Left Panel: Histogram
        bins = min(25, max(8, int(np.sqrt(len(merits_scaled)) * 1.5)))
        ax_hist.hist(merits_scaled, bins=bins, color="#38bdf8" if self.dark_mode else "#42a5f5", edgecolor="#0284c7" if self.dark_mode else "#1565c0", alpha=0.8)
        ax_hist.axvline(nom_scaled, color="#4ade80" if self.dark_mode else "#2e7d32", linestyle="--", linewidth=1.4, label=f"Nom: {nom_scaled:.1f} {unit}")
        ax_hist.axvline(p50_scaled, color="#60a5fa" if self.dark_mode else "#1565c0", linestyle="-", linewidth=1.4, label=f"P50: {p50_scaled:.1f} {unit}")
        ax_hist.axvline(p90_scaled, color="#fb923c" if self.dark_mode else "#ef6c00", linestyle="-.", linewidth=1.4, label=f"P90: {p90_scaled:.1f} {unit}")
        ax_hist.set_xlabel(f"{report.criterion_name} ({unit})", fontsize=8.5, fontweight="bold")
        ax_hist.set_ylabel("Trial Count", fontsize=8.5)
        ax_hist.set_title(f"Monte Carlo Distribution (N={report.num_trials})", fontsize=9, fontweight="bold")
        ax_hist.legend(fontsize=7.5, loc="upper right")
        ax_hist.grid(linestyle=":", alpha=0.4)
        ax_hist.tick_params(axis="both", labelsize=8)

        # Right Panel: Cumulative Yield Curve (CDF)
        sorted_m = np.sort(merits_scaled)
        cdf_pct = np.arange(1, len(sorted_m) + 1, dtype=float) / len(sorted_m) * 100.0
        ax_cdf.plot(sorted_m, cdf_pct, color="#4ade80" if self.dark_mode else "#1b5e20", linewidth=2.0, label="Cumulative Yield %")
        ax_cdf.axhline(90.0, color="#fb923c" if self.dark_mode else "#ef6c00", linestyle=":", linewidth=1.0)
        ax_cdf.axvline(p90_scaled, color="#fb923c" if self.dark_mode else "#ef6c00", linestyle="-.", linewidth=1.2, label=f"90% <= {p90_scaled:.1f} {unit}")
        ax_cdf.axvline(p99_scaled, color="#f87171" if self.dark_mode else "#c62828", linestyle="--", linewidth=1.2, label=f"99% <= {p99_scaled:.1f} {unit}")

        thresh_disp = self.yield_thresh_spin.value()
        ax_cdf.axvline(thresh_disp, color="#c084fc" if self.dark_mode else "#6a1b9a", linestyle="-", linewidth=1.3, label=f"Spec: {thresh_disp:.1f} {unit}")

        ax_cdf.set_xlabel(f"Performance Threshold ({unit})", fontsize=8.5, fontweight="bold")
        ax_cdf.set_ylabel("Manufacturing Yield (%)", fontsize=8.5)
        ax_cdf.set_ylim(0, 105)
        ax_cdf.set_title("Cumulative Manufacturing Yield (CDF)", fontsize=9, fontweight="bold")
        ax_cdf.legend(fontsize=7.5, loc="lower right")
        ax_cdf.grid(linestyle=":", alpha=0.4)
        ax_cdf.tick_params(axis="both", labelsize=8)

        apply_axes_theme(ax_hist, self.dark_mode)
        apply_axes_theme(ax_cdf, self.dark_mode)
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _add_row(
        self,
        operand: str = "TRAD",
        surf_idx: int = 1,
        delta: float = 0.1,
        dist: str = "uniform",
        min_delta: Optional[float] = None,
        comment: str = ""
    ):
        row = self.table.rowCount()
        self.table.insertRow(row)

        op_combo = QComboBox()
        op_combo.addItems([
            "TRAD", "TCUR", "TTHI", "TSDX", "TSDY", "TSTX", "TSTY",
            "TEDX", "TEDY", "TETX", "TETY", "TIND", "TABB", "TCON", "TIRR"
        ])
        idx = op_combo.findText(str(operand).upper())
        if idx >= 0:
            op_combo.setCurrentIndex(idx)
        self.table.setCellWidget(row, 0, op_combo)

        self.table.setItem(row, 1, QTableWidgetItem(str(surf_idx)))
        self.table.setItem(row, 2, QTableWidgetItem(f"{delta:.4g}"))

        dist_combo = QComboBox()
        dist_combo.addItems(["uniform", "gaussian", "parabolic"])
        d_idx = dist_combo.findText(str(dist).lower())
        if d_idx >= 0:
            dist_combo.setCurrentIndex(d_idx)
        self.table.setCellWidget(row, 3, dist_combo)

        m_del = min_delta if min_delta is not None else -abs(delta)
        self.table.setItem(row, 4, QTableWidgetItem(f"{m_del:.4g}"))
        self.table.setItem(row, 5, QTableWidgetItem(comment))

    def _remove_row(self):
        row = self.table.currentRow()
        if row >= 0:
            self.table.removeRow(row)

    def _clear_table(self):
        self.table.setRowCount(0)

    def _apply_preset_clicked(self):
        system = self._get_active_system()
        if system is None:
            QMessageBox.warning(self, "No System", "Please load or build an optical system before generating tolerances.")
            return

        idx = self.preset_combo.currentIndex()
        if idx == 1:
            tols = create_commercial_tolerances(system)
        elif idx == 2:
            tols = create_precision_tolerances(system)
        elif idx == 3:
            tols = create_high_precision_tolerances(system)
        else:
            return

        self._clear_table()
        for t in tols:
            s_idx = t.surface_index if t.surface_index is not None else t.element_index
            self._add_row(
                operand=t.parameter_name.upper(),
                surf_idx=s_idx,
                delta=t.delta,
                dist=t.distribution,
                min_delta=t.min_delta,
                comment=t.notes
            )
        self.output_text.setText(f"Loaded {len(tols)} standard tolerances from quality preset.")

    def _apply_solved_limits_to_table(self):
        """Apply solved symmetric limits from the last Inverse Sensitivity analysis back into the TDE table."""
        if self.last_inverse_report is None or not self.last_inverse_report.entries:
            QMessageBox.information(
                self, "No Inverse Sensitivity Data", "Run Inverse Sensitivity first to solve for tolerance limits."
            )
            return

        entries = self.last_inverse_report.entries
        row_count = self.table.rowCount()
        updated = 0
        for row in range(min(row_count, len(entries))):
            entry = entries[row]
            sym_val = entry.solved_symmetric_delta
            self.table.setItem(row, 2, QTableWidgetItem(f"{sym_val:.4g}"))
            self.table.setItem(row, 4, QTableWidgetItem(f"{-sym_val:.4g}"))
            updated += 1

        self.output_text.append(
            f"\n[Applied {updated} solved Inverse Sensitivity tolerance limits to the TDE table. Ready for Monte Carlo verification.]"
        )

    def _get_tolerances(self) -> List[PerturbationSpec]:
        tolerances = []
        for row in range(self.table.rowCount()):
            op_widget = self.table.cellWidget(row, 0)
            op = op_widget.currentText() if op_widget else "TRAD"

            idx_item = self.table.item(row, 1)
            surf_idx = int(idx_item.text()) if idx_item and idx_item.text() else 1

            delta_item = self.table.item(row, 2)
            try:
                delta = float(delta_item.text()) if delta_item and delta_item.text() else 0.05
            except ValueError:
                delta = 0.05

            dist_widget = self.table.cellWidget(row, 3)
            dist = dist_widget.currentText() if dist_widget else "uniform"

            min_item = self.table.item(row, 4)
            try:
                min_del = float(min_item.text()) if min_item and min_item.text() else -abs(delta)
            except ValueError:
                min_del = -abs(delta)

            comm_item = self.table.item(row, 5)
            notes = comm_item.text() if comm_item else ""

            tolerances.append(PerturbationSpec(
                parameter_name=op.lower(),
                element_index=surf_idx,
                surface_index=surf_idx,
                delta=delta,
                min_delta=min_del,
                max_delta=abs(delta),
                distribution=dist,
                notes=notes
            ))
        return tolerances

    def _get_criterion(self) -> str:
        idx = self.crit_combo.currentIndex()
        if idx == 0: return "rms_spot_radius"
        elif idx == 1: return "geo_spot_radius"
        elif idx == 2: return "effl"
        elif idx == 3: return "bfl"
        return "merit_function"

    def _run_sensitivity(self):
        self._run_analysis(mode="sensitivity")

    def _run_inverse_sensitivity(self):
        self._run_analysis(mode="inverse")

    def _run_monte_carlo(self):
        self._run_analysis(mode="monte_carlo")

    def _run_analysis(self, is_monte_carlo: bool = False, mode: Optional[str] = None):
        active_mode = mode if mode is not None else ("monte_carlo" if is_monte_carlo else "sensitivity")
        system = self._get_active_system()
        if system is None:
            self.output_text.setText("Error: No active optical system available in Lens Data Editor.")
            return

        tolerances = self._get_tolerances()
        if not tolerances:
            self.output_text.setText("Please define at least one tolerance operand or select a quality preset.")
            return

        criterion = self._get_criterion()
        trials = self.trials_spin.value()
        seed = self.seed_spin.value() if self.seed_spin.value() > 0 else None
        comp_enabled = self.chk_compensator.isChecked()
        comp_range = self.comp_range_spin.value()
        allow_deg_um = self.inv_deg_spin.value()

        self.output_text.clear()
        self.output_text.setText(f"Executing Zemax OpticStudio Tolerancing ({active_mode.upper()})...")
        self.btn_sensitivity.setEnabled(False)
        self.btn_inverse.setEnabled(False)
        self.btn_monte_carlo.setEnabled(False)

        self.worker = TolerancingWorker(
            system=system,
            tolerances=tolerances,
            criterion=criterion,
            is_monte_carlo=(active_mode == "monte_carlo"),
            mode=active_mode,
            allowable_degradation_um=allow_deg_um,
            trials=trials,
            seed=seed,
            compensator_enabled=comp_enabled,
            compensator_surface_idx=-2,
            compensator_range_mm=comp_range,
            parent=self
        )
        self.worker.result_ready.connect(self._on_result)
        self.worker.error_occurred.connect(self._on_error)
        self.worker.finished.connect(self._on_finished)
        self.worker.start()

    def _on_result(self, report_obj: Any, text: str):
        if isinstance(report_obj, MonteCarloReport):
            self.last_monte_carlo_report = report_obj
            scale = 1000.0 if report_obj.is_spot_criterion else 1.0
            self.yield_thresh_spin.setValue(float(report_obj.p90 * scale))
            self._update_yield_label()
            self._plot_monte_carlo_chart(report_obj)
        elif isinstance(report_obj, InverseSensitivityReport):
            self.last_inverse_report = report_obj
            self.btn_apply_solved.setEnabled(True)
            self._plot_inverse_sensitivity_chart(report_obj)
        elif isinstance(report_obj, SensitivityReport):
            self.last_sensitivity_report = report_obj
            self._plot_sensitivity_chart(report_obj)

        self.output_text.setText(text)

    def _on_error(self, err: str):
        self.output_text.setText(err)

    def _on_finished(self):
        self.btn_sensitivity.setEnabled(True)
        self.btn_inverse.setEnabled(True)
        self.btn_monte_carlo.setEnabled(True)

    def _calculate_yield_clicked(self):
        if self.last_monte_carlo_report is None:
            QMessageBox.information(self, "No Monte Carlo Data", "Please run a Monte Carlo simulation first to calculate yield.")
            return
        self._update_yield_label()
        self._plot_monte_carlo_chart(self.last_monte_carlo_report)

    def _update_yield_label(self):
        if self.last_monte_carlo_report is not None:
            is_spot = self.last_monte_carlo_report.is_spot_criterion
            thresh_ui = self.yield_thresh_spin.value()
            # Convert UI threshold (µm for spot, mm for EFL/BFL) to internal criterion units (mm)
            thresh_internal = (thresh_ui / 1000.0) if is_spot else thresh_ui
            unit = "\u03bcm" if is_spot else "mm"
            rate = self.last_monte_carlo_report.yield_percentage(thresh_internal)
            self.lbl_yield_result.setText(f"Yield: {rate:.1f}% (<= {thresh_ui:.2f} {unit})")

    def _export_report_clicked(self):
        if not self.last_sensitivity_report and not self.last_monte_carlo_report and not self.last_inverse_report:
            QMessageBox.warning(self, "No Results", "Run an analysis before exporting a report.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Export Zemax Tolerance Report", "zemax_tolerance_report.txt", "Text Files (*.txt);;All Files (*)"
        )
        if path:
            export_zemax_tolerance_report(
                sensitivity_report=self.last_sensitivity_report,
                monte_carlo_report=self.last_monte_carlo_report,
                inverse_sensitivity_report=self.last_inverse_report,
                system_title=getattr(self._get_active_system(), "name", "2D Optics System"),
                filepath=path
            )
            QMessageBox.information(self, "Report Exported", f"Successfully saved Zemax tolerance report to:\n{path}")

    def _export_zmx_clicked(self):
        system = self._get_active_system()
        if system is None or not hasattr(system, "surfaces"):
            QMessageBox.warning(self, "Invalid System", "Active system must be a SurfaceStack to export to .ZMX.")
            return

        tols = self._get_tolerances()
        if not tols:
            QMessageBox.warning(self, "No Tolerances", "Please define at least one tolerance operand.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Export Toleranced .ZMX Prescription", "toleranced_prescription.zmx", "Zemax Files (*.zmx);;All Files (*)"
        )
        if path:
            export_zmx_with_tolerances(system, tols, path, title=getattr(system, "name", "Toleranced System"))
            QMessageBox.information(self, "ZMX Exported", f"Successfully exported toleranced Zemax file to:\n{path}")
