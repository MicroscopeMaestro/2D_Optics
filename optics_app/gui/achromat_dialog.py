"""
Interactive Achromatic Doublet Lens Designer & Optimization Dialog.
===================================================================
Provides a graphical workspace for designing, optimizing, and visualizing
achromatic doublet lenses using real optical glass catalogs (Crown + Flint):
- Select target focal length (EFL), clear aperture diameter, and spectral band.
- Explore the Abbe Diagram (nd vs Vd) and live refractive index dispersion curves n(lambda).
- Filter and choose recommended Crown/Flint glass pairs (e.g. N-BK7 + N-SF2, CaF2 + N-SF11).
- Automatically solve initial curvatures and run non-linear optimization on surface radii
  (R1, R2, R3) and thicknesses (t1, t2) to nullify chromatic aberration (C_L -> 0) and
  drastically minimize spherical aberration (S_I -> 0).
- Inspect diagnostic plots:
  * Refractive Index Dispersion Curves & Abbe Diagram
  * Surface Aberration Reductions (Before vs After Optimization)
  * Chromatic Focal Shift Curve Delta f(lambda)
  * Transverse Ray Fan Curves Ey(Py) for F, d, C Fraunhofer lines
- Direct one-click insertion of designed doublet into the active 2D Optics system.
"""

from __future__ import annotations
import os
import json
import warnings
from typing import Optional, List, Dict, Any, Tuple
import numpy as np

from optics_app.gui.qt_compat import (
    QT_AVAILABLE, QtWidgets, QtCore,
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QPushButton, QComboBox, QLineEdit, QTableWidget, QTableWidgetItem,
    QHeaderView, QGroupBox, QRadioButton, QCheckBox, QDoubleSpinBox,
    QAbstractItemView, QMessageBox, Qt, QTabWidget, QWidget, QFileDialog
)

# Matplotlib Qt Canvas
try:
    import matplotlib
    from matplotlib.figure import Figure
    if QT_AVAILABLE:
        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
            from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
        except ImportError:
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
    else:
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        NavigationToolbar = None
except ImportError:
    FigureCanvas = None
    NavigationToolbar = None

try:
    from v2.engine.achromat_designer import (
        AchromatDesignSpec,
        AchromatDesignResult,
        GlassInfo,
        GlassPairRecommendation,
        get_all_catalog_glasses,
        rank_achromatic_pairs,
        optimize_achromatic_doublet,
        get_dispersion_curve_points,
        FRAUNHOFER_D_UM,
        FRAUNHOFER_F_UM,
        FRAUNHOFER_C_UM,
    )
    from v2.engine.surface import Surface, SurfaceType
except ImportError:
    from optics_app.engine.achromat_designer import (
        AchromatDesignSpec,
        AchromatDesignResult,
        GlassInfo,
        GlassPairRecommendation,
        get_all_catalog_glasses,
        rank_achromatic_pairs,
        optimize_achromatic_doublet,
        get_dispersion_curve_points,
        FRAUNHOFER_D_UM,
        FRAUNHOFER_F_UM,
        FRAUNHOFER_C_UM,
    )
    from optics_app.engine.surface import Surface, SurfaceType

try:
    from optics_app.gui.theme import apply_axes_theme, get_theme_colors
except ImportError:
    from v2.gui.theme import apply_axes_theme, get_theme_colors


class AchromatDesignerDialog(QDialog if QT_AVAILABLE else object):
    """
    Interactive Dialog for designing, optimizing, and visualizing Achromatic Doublet lenses.
    """

    def __init__(self, model: Optional[Any] = None, parent: Optional[Any] = None):
        if QT_AVAILABLE:
            super().__init__(parent)
            self.setWindowTitle("🎨 Achromatic Doublet Lens Designer — Glass Catalog & Surface Optimization")
            self.resize(1150, 780)

        self.model = model
        dm = getattr(parent, "dark_mode", None)
        if dm is None and parent is not None and hasattr(parent, "window"):
            w = parent.window()
            if w is not None:
                dm = getattr(w, "dark_mode", None)
        self.dark_mode = bool(dm) if dm is not None else False

        self.catalog_glasses = get_all_catalog_glasses()
        self.ranked_pairs = rank_achromatic_pairs(self.catalog_glasses, min_delta_v=18.0)
        self.current_result: Optional[AchromatDesignResult] = None

        if QT_AVAILABLE:
            self._setup_ui()
            self._update_theme_styles()
            self._populate_glass_combos()
            self._populate_pairs_table()
            # Run initial optimization with default N-BK7 + N-SF2
            self._run_optimization()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(8)

        # ---------------------------------------------------------------------
        # 1. Top Control Bar: Specifications & Optimization Triggers
        # ---------------------------------------------------------------------
        top_group = QGroupBox("Achromatic Lens Design Specifications", self)
        top_layout = QGridLayout(top_group)
        top_layout.setContentsMargins(8, 8, 8, 8)
        top_layout.setSpacing(8)

        # Target Focal Length (EFL)
        top_layout.addWidget(QLabel("Target EFL (mm):"), 0, 0)
        self.spin_efl = QDoubleSpinBox(top_group)
        self.spin_efl.setRange(10.0, 5000.0)
        self.spin_efl.setValue(100.0)
        self.spin_efl.setSingleStep(10.0)
        self.spin_efl.setDecimals(1)
        top_layout.addWidget(self.spin_efl, 0, 1)

        # Lens Diameter
        top_layout.addWidget(QLabel("Diameter (mm):"), 0, 2)
        self.spin_diam = QDoubleSpinBox(top_group)
        self.spin_diam.setRange(5.0, 200.0)
        self.spin_diam.setValue(25.4)
        self.spin_diam.setSingleStep(2.54)
        self.spin_diam.setDecimals(2)
        top_layout.addWidget(self.spin_diam, 0, 3)

        # Doublet Construction Type
        top_layout.addWidget(QLabel("Doublet Type:"), 0, 4)
        self.combo_type = QComboBox(top_group)
        self.combo_type.addItems(["Cemented Doublet", "Air-Spaced Doublet"])
        top_layout.addWidget(self.combo_type, 0, 5)

        # Flat Interface Checkbox
        self.chk_flat_interface = QCheckBox("Flat Interface (R2 = Flat)", top_group)
        self.chk_flat_interface.setChecked(False)
        self.chk_flat_interface.setToolTip("Constrain the cemented contact interface between crown and flint to be planar (infinite radius)")
        self.chk_flat_interface.toggled.connect(self._run_optimization)
        top_layout.addWidget(self.chk_flat_interface, 2, 0, 1, 3)

        # Spectral Band Presets
        top_layout.addWidget(QLabel("Spectral Band:"), 1, 0)
        self.combo_band = QComboBox(top_group)
        self.combo_band.addItems([
            "Visible (F, d, C: 486.1, 587.6, 656.3 nm)",
            "NIR (800, 1064, 1300 nm)",
            "UV-Visible (365, 486, 588 nm)"
        ])
        top_layout.addWidget(self.combo_band, 1, 1)

        # Crown Glass Combo
        top_layout.addWidget(QLabel("Crown Glass (Low Disp.):"), 1, 2)
        self.combo_crown = QComboBox(top_group)
        self.combo_crown.currentIndexChanged.connect(self._on_glass_combo_changed)
        top_layout.addWidget(self.combo_crown, 1, 3)

        # Flint Glass Combo
        top_layout.addWidget(QLabel("Flint Glass (High Disp.):"), 1, 4)
        self.combo_flint = QComboBox(top_group)
        self.combo_flint.currentIndexChanged.connect(self._on_glass_combo_changed)
        top_layout.addWidget(self.combo_flint, 1, 5)

        # Optimize Button
        self.btn_optimize = QPushButton("⚡ Optimize Surfaces (Reduce S_I & C_L)", top_group)
        self.btn_optimize.setStyleSheet(
            "QPushButton { background-color: #0284c7; color: white; font-weight: bold; padding: 6px 14px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #0369a1; }"
        )
        self.btn_optimize.clicked.connect(self._run_optimization)
        top_layout.addWidget(self.btn_optimize, 0, 6, 3, 1)

        main_layout.addWidget(top_group)

        # ---------------------------------------------------------------------
        # 2. Main Content Splitter: Left Recommended Pairs & Prescription, Right Plots
        # ---------------------------------------------------------------------
        content_layout = QHBoxLayout()
        content_layout.setSpacing(8)

        # --- Left Side: Catalog Glass Pairs & Prescription Summary ---
        left_panel = QWidget(self)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)

        # Catalog Pairing Table
        pairs_group = QGroupBox("Recommended Catalog Crown-Flint Pairs", left_panel)
        pairs_layout = QVBoxLayout(pairs_group)
        pairs_layout.setContentsMargins(4, 4, 4, 4)

        self.table_pairs = QTableWidget(pairs_group)
        self.table_pairs.setColumnCount(5)
        self.table_pairs.setHorizontalHeaderLabels(["Crown", "Flint", "ΔV", "Sec. Spec (ppm)", "Type"])
        self.table_pairs.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table_pairs.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table_pairs.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table_pairs.setStyleSheet("font-size: 11px;")
        self.table_pairs.itemSelectionChanged.connect(self._on_pair_table_selected)
        pairs_layout.addWidget(self.table_pairs)
        left_layout.addWidget(pairs_group, stretch=3)

        # Prescription & Aberration Cards
        presc_group = QGroupBox("Optimized Lens Prescription & Aberration Reduction", left_panel)
        presc_layout = QVBoxLayout(presc_group)
        presc_layout.setContentsMargins(6, 6, 6, 6)

        self.lbl_prescription = QLabel("Computing optimization...", presc_group)
        self.lbl_prescription.setStyleSheet("font-family: monospace; font-size: 11px; background-color: #f8fafc; padding: 6px; border: 1px solid #e2e8f0; border-radius: 4px;")
        self.lbl_prescription.setWordWrap(True)
        presc_layout.addWidget(self.lbl_prescription)

        self.lbl_aberrations = QLabel("", presc_group)
        self.lbl_aberrations.setStyleSheet("font-family: monospace; font-size: 11px; background-color: #f0fdf4; padding: 6px; border: 1px solid #bbf7d0; border-radius: 4px; color: #166534;")
        self.lbl_aberrations.setWordWrap(True)
        presc_layout.addWidget(self.lbl_aberrations)

        left_layout.addWidget(presc_group, stretch=2)
        content_layout.addWidget(left_panel, stretch=4)

        # --- Right Side: Visual Diagnostics Tabs (Matplotlib) ---
        self.tabs_plot = QTabWidget(self)

        # Tab 1: Refractive Index Curves & Abbe Diagram
        self.tab_dispersion = self._create_canvas_tab()
        self.tabs_plot.addTab(self.tab_dispersion, "📈 Dispersion Curves & Abbe Diagram")

        # Tab 2: Aberration Reductions Comparison
        self.tab_aberration = self._create_canvas_tab()
        self.tabs_plot.addTab(self.tab_aberration, "📊 Aberration Reduction (S_I & C_L)")

        # Tab 3: Chromatic Focal Shift Curve
        self.tab_focal_shift = self._create_canvas_tab()
        self.tabs_plot.addTab(self.tab_focal_shift, "🌈 Chromatic Focal Shift Δf(λ)")

        # Tab 4: Transverse Ray Fan
        self.tab_ray_fan = self._create_canvas_tab()
        self.tabs_plot.addTab(self.tab_ray_fan, "🔍 Transverse Ray Fan (F, d, C)")

        content_layout.addWidget(self.tabs_plot, stretch=6)
        main_layout.addLayout(content_layout, stretch=1)

        # ---------------------------------------------------------------------
        # 3. Bottom Button Bar: Actions & System Insertion
        # ---------------------------------------------------------------------
        bottom_layout = QHBoxLayout()
        bottom_layout.setContentsMargins(0, 4, 0, 0)

        self.lbl_status = QLabel("Ready.", self)
        self.lbl_status.setStyleSheet("color: #64748b; font-style: italic;")
        bottom_layout.addWidget(self.lbl_status)
        bottom_layout.addStretch(1)

        self.btn_export = QPushButton("💾 Export Prescription (.json)", self)
        self.btn_export.clicked.connect(self._export_prescription)
        bottom_layout.addWidget(self.btn_export)

        self.btn_insert = QPushButton("➕ Insert Doublet into Active Project", self)
        self.btn_insert.setStyleSheet(
            "QPushButton { background-color: #16a34a; color: white; font-weight: bold; padding: 6px 14px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #15803d; }"
        )
        self.btn_insert.clicked.connect(self._insert_into_project)
        bottom_layout.addWidget(self.btn_insert)

        self.btn_close = QPushButton("Close", self)
        self.btn_close.clicked.connect(self.accept)
        bottom_layout.addWidget(self.btn_close)

        main_layout.addLayout(bottom_layout)

    def _create_canvas_tab(self) -> QWidget:
        widget = QWidget(self)
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(2, 2, 2, 2)
        fig = Figure(figsize=(6, 4.5), dpi=100)
        colors = get_theme_colors(getattr(self, "dark_mode", False))
        fig.patch.set_facecolor(colors["bg_card"])
        canvas = FigureCanvas(fig)
        toolbar = NavigationToolbar(canvas, widget) if NavigationToolbar else None
        if toolbar:
            layout.addWidget(toolbar)
        layout.addWidget(canvas)
        widget.fig = fig
        widget.canvas = canvas
        return widget

    def _populate_glass_combos(self):
        # Allow all glasses that can serve as crown (Vd >= 38.0 or Crown type), sorted by Vd descending
        crowns = [g for g in self.catalog_glasses if g.glass_type == "Crown" or g.vd >= 38.0]
        crowns.sort(key=lambda g: g.vd, reverse=True)
        # Allow all glasses that can serve as flint (Vd <= 55.0 or Flint type), sorted by Vd ascending
        flints = [g for g in self.catalog_glasses if g.glass_type == "Flint" or g.vd <= 55.0]
        flints.sort(key=lambda g: g.vd)

        self.combo_crown.blockSignals(True)
        self.combo_flint.blockSignals(True)
        self.combo_crown.clear()
        self.combo_flint.clear()

        for c in crowns:
            self.combo_crown.addItem(f"{c.name} (nd={c.nd:.3f}, Vd={c.vd:.1f})", c.name)
        for f in flints:
            self.combo_flint.addItem(f"{f.name} (nd={f.nd:.3f}, Vd={f.vd:.1f})", f.name)

        # Set default to N-BK7 and N-SF2
        idx_bk7 = self.combo_crown.findData("N-BK7")
        if idx_bk7 >= 0:
            self.combo_crown.setCurrentIndex(idx_bk7)
        idx_sf2 = self.combo_flint.findData("N-SF2")
        if idx_sf2 >= 0:
            self.combo_flint.setCurrentIndex(idx_sf2)

        self.combo_crown.blockSignals(False)
        self.combo_flint.blockSignals(False)

    def _populate_pairs_table(self):
        self.table_pairs.setRowCount(len(self.ranked_pairs))
        for row, pair in enumerate(self.ranked_pairs):
            crown_item = QTableWidgetItem(pair.crown.name)
            flint_item = QTableWidgetItem(pair.flint.name)
            delta_v_item = QTableWidgetItem(f"{pair.delta_v:.1f}")
            sec_spec_item = QTableWidgetItem(f"{pair.secondary_spectrum_ppm:.0f}")
            tag = "⭐ Standard" if pair.is_commercial_favorite else "Custom"
            if "Apo" in pair.notes:
                tag = "✨ Apo"
            tag_item = QTableWidgetItem(tag)

            crown_item.setTextAlignment(Qt.AlignCenter)
            flint_item.setTextAlignment(Qt.AlignCenter)
            delta_v_item.setTextAlignment(Qt.AlignCenter)
            sec_spec_item.setTextAlignment(Qt.AlignCenter)
            tag_item.setTextAlignment(Qt.AlignCenter)

            self.table_pairs.setItem(row, 0, crown_item)
            self.table_pairs.setItem(row, 1, flint_item)
            self.table_pairs.setItem(row, 2, delta_v_item)
            self.table_pairs.setItem(row, 3, sec_spec_item)
            self.table_pairs.setItem(row, 4, tag_item)

        if len(self.ranked_pairs) > 0:
            self.table_pairs.selectRow(0)

    def _on_pair_table_selected(self):
        sel = self.table_pairs.selectedItems()
        if not sel:
            return
        row = sel[0].row()
        if 0 <= row < len(self.ranked_pairs):
            pair = self.ranked_pairs[row]
            c_idx = self.combo_crown.findData(pair.crown.name)
            f_idx = self.combo_flint.findData(pair.flint.name)
            if c_idx >= 0 and f_idx >= 0:
                self.combo_crown.blockSignals(True)
                self.combo_flint.blockSignals(True)
                self.combo_crown.setCurrentIndex(c_idx)
                self.combo_flint.setCurrentIndex(f_idx)
                self.combo_crown.blockSignals(False)
                self.combo_flint.blockSignals(False)
                self._run_optimization()

    def _on_glass_combo_changed(self):
        self._run_optimization()

    def _run_optimization(self):
        crown_glass = self.combo_crown.currentData() or "N-BK7"
        flint_glass = self.combo_flint.currentData() or "N-SF2"
        target_efl = float(self.spin_efl.value())
        diameter = float(self.spin_diam.value())
        doublet_type = "cemented" if self.combo_type.currentIndex() == 0 else "air_spaced"

        # Spectral band handling
        band_idx = self.combo_band.currentIndex()
        if band_idx == 1:  # NIR
            wl_d, wl_F, wl_C = 1.064, 0.800, 1.300
        elif band_idx == 2:  # UV
            wl_d, wl_F, wl_C = 0.486, 0.365, 0.588
        else:  # Visible
            wl_d, wl_F, wl_C = FRAUNHOFER_D_UM, FRAUNHOFER_F_UM, FRAUNHOFER_C_UM

        flat_interface = self.chk_flat_interface.isChecked() if hasattr(self, 'chk_flat_interface') else False

        spec = AchromatDesignSpec(
            target_efl=target_efl,
            diameter=diameter,
            crown_glass=crown_glass,
            flint_glass=flint_glass,
            doublet_type=doublet_type,
            flat_interface=flat_interface,
            wavelength_d=wl_d,
            wavelength_F=wl_F,
            wavelength_C=wl_C,
        )

        try:
            self.lbl_status.setText("Optimizing surface curvatures and aberration balance...")
            QtWidgets.QApplication.processEvents()
            result = optimize_achromatic_doublet(spec)
            self.current_result = result
            self._update_prescription_labels(result)
            self._render_all_plots(result)
            self.lbl_status.setText(f"Optimized {crown_glass} + {flint_glass} doublet successfully (EFL: {result.efl:.2f} mm).")
        except Exception as e:
            self.lbl_status.setText(f"Optimization error: {str(e)}")
            self.lbl_prescription.setText(f"Error during optimization: {str(e)}")

    def _update_prescription_labels(self, res: AchromatDesignResult):
        if res.spec.doublet_type == "cemented":
            r2_str = "Flat (Infinity)" if (abs(res.radii[1]) > 1e7 or getattr(res.spec, 'flat_interface', False)) else f"{res.radii[1]:.2f} mm"
            presc_text = (
                f"<b>Prescription (Cemented):</b><br>"
                f"• R1 (Crown Front): <b>{res.radii[0]:.2f} mm</b> | t1: {res.thicknesses[0]:.2f} mm ({res.crown.name})<br>"
                f"• R2 (Cemented Contact): <b>{r2_str}</b> | t2: {res.thicknesses[1]:.2f} mm ({res.flint.name})<br>"
                f"• R3 (Flint Rear): <b>{res.radii[2]:.2f} mm</b> | Air<br>"
                f"• Paraxial EFL: <b>{res.efl:.3f} mm</b> | Paraxial BFL: <b>{res.bfl:.3f} mm</b><br>"
                f"• Clear Aperture: <b>{res.spec.diameter:.1f} mm</b> (f/{res.efl/res.spec.diameter:.2f})"
            )
        else:
            presc_text = (
                f"<b>Prescription (Air-Spaced):</b><br>"
                f"• R1: <b>{res.radii[0]:.2f} mm</b> | t1: {res.thicknesses[0]:.2f} mm ({res.crown.name})<br>"
                f"• R2: <b>{res.radii[1]:.2f} mm</b> | Air gap: {res.thicknesses[1]:.2f} mm<br>"
                f"• R3: <b>{res.radii[2]:.2f} mm</b> | t2: {res.thicknesses[2]:.2f} mm ({res.flint.name})<br>"
                f"• R4: <b>{res.radii[3]:.2f} mm</b> | BFL: {res.bfl:.2f} mm<br>"
                f"• Paraxial EFL: <b>{res.efl:.3f} mm</b> (f/{res.efl/res.spec.diameter:.2f})"
            )
        self.lbl_prescription.setText(presc_text)

        # Aberration reductions
        s_i_drop = abs(res.initial_s_I / max(1e-12, abs(res.optimized_s_I))) if abs(res.optimized_s_I) > 1e-12 else float("inf")
        c_l_drop = abs(res.initial_c_L / max(1e-12, abs(res.optimized_c_L))) if abs(res.optimized_c_L) > 1e-12 else float("inf")

        aber_text = (
            f"<b>Seidel Aberration Reduction:</b><br>"
            f"• Spherical Aberration S_I: {res.initial_s_I:.3e} mm ➔ <b>{res.optimized_s_I:.3e} mm</b> "
            f"(<b>{s_i_drop:.1f}×</b> reduction!)<br>"
            f"• Longitudinal Color C_L: {res.initial_c_L:.3e} mm ➔ <b>{res.optimized_c_L:.3e} mm</b> "
            f"(<b>{c_l_drop:.1f}×</b> reduction!)<br>"
            f"• Coma S_II: {res.optimized_seidel.get('S_II', 0.0):.3e} mm | Astigmatism S_III: {res.optimized_seidel.get('S_III', 0.0):.3e} mm"
        )
        self.lbl_aberrations.setText(aber_text)

    def set_dark_mode(self, enabled: bool) -> None:
        """Switch between dark and light background themes."""
        self.dark_mode = bool(enabled)
        self._update_theme_styles()
        if self.current_result:
            self._render_all_plots(self.current_result)

    def _update_theme_styles(self) -> None:
        """Update styled elements to match the current theme."""
        if not QT_AVAILABLE:
            return
        is_dark = getattr(self, "dark_mode", False)

        if hasattr(self, "lbl_prescription") and self.lbl_prescription:
            if is_dark:
                self.lbl_prescription.setStyleSheet(
                    "font-family: monospace; font-size: 11px; background-color: #1e293b; color: #f1f5f9; padding: 6px; border: 1px solid #334155; border-radius: 4px;"
                )
            else:
                self.lbl_prescription.setStyleSheet(
                    "font-family: monospace; font-size: 11px; background-color: #f8fafc; color: #0f172a; padding: 6px; border: 1px solid #e2e8f0; border-radius: 4px;"
                )

        if hasattr(self, "lbl_aberrations") and self.lbl_aberrations:
            if is_dark:
                self.lbl_aberrations.setStyleSheet(
                    "font-family: monospace; font-size: 11px; background-color: #064e3b; color: #6ee7b7; padding: 6px; border: 1px solid #047857; border-radius: 4px;"
                )
            else:
                self.lbl_aberrations.setStyleSheet(
                    "font-family: monospace; font-size: 11px; background-color: #f0fdf4; color: #166534; padding: 6px; border: 1px solid #bbf7d0; border-radius: 4px;"
                )

        if hasattr(self, "lbl_status") and self.lbl_status:
            stat_col = "#94a3b8" if is_dark else "#64748b"
            self.lbl_status.setStyleSheet(f"color: {stat_col}; font-style: italic;")

    def _render_all_plots(self, res: AchromatDesignResult):
        colors = get_theme_colors(self.dark_mode)

        # 1. Tab 1: Refractive Index Dispersion & Abbe Diagram
        fig1 = self.tab_dispersion.fig
        fig1.clear()
        fig1.patch.set_facecolor(colors["bg_card"])
        ax1 = fig1.add_subplot(1, 2, 1)
        ax2 = fig1.add_subplot(1, 2, 2)

        # Left: n(lambda) curves
        for gname, data in res.dispersion_curves.items():
            wl = data["wl_nm"]
            n = data["n"]
            label = f"{gname} ({'Crown' if gname == res.crown.name else 'Flint'})"
            color = "#38bdf8" if (self.dark_mode and gname == res.crown.name) else ("#0284c7" if gname == res.crown.name else "#ea580c")
            ax1.plot(wl, n, label=label, color=color, linewidth=2.0)

        # Mark Fraunhofer lines
        ax1.axvline(486.1, color="#3b82f6", linestyle="--", alpha=0.7, label="F (486 nm)")
        ax1.axvline(587.6, color="#eab308", linestyle="--", alpha=0.7, label="d (588 nm)")
        ax1.axvline(656.3, color="#ef4444", linestyle="--", alpha=0.7, label="C (656 nm)")
        ax1.set_xlabel("Wavelength (nm)", fontsize=10)
        ax1.set_ylabel("Refractive Index n(λ)", fontsize=10)
        ax1.set_title("Glass Dispersion Curves n(λ)", fontsize=11, fontweight="bold")
        ax1.legend(fontsize=8, loc="best")
        apply_axes_theme(ax1, self.dark_mode)

        # Right: Abbe Diagram
        all_glasses = self.catalog_glasses
        crowns = [g for g in all_glasses if g.glass_type == "Crown"]
        flints = [g for g in all_glasses if g.glass_type == "Flint"]

        crown_scatter_col = "#34d399" if self.dark_mode else "#10b981"
        flint_scatter_col = "#fb923c" if self.dark_mode else "#f97316"
        ax2.scatter([c.vd for c in crowns], [c.nd for c in crowns],
                    color=crown_scatter_col, s=40, label="Crowns (Vd ≥ 50)", alpha=0.8, edgecolors="none")
        ax2.scatter([f.vd for f in flints], [f.nd for f in flints],
                    color=flint_scatter_col, s=40, label="Flints (Vd < 50)", alpha=0.8, edgecolors="none")

        # Highlight chosen pair
        c_pt = (res.crown.vd, res.crown.nd)
        f_pt = (res.flint.vd, res.flint.nd)
        edge_col = "#ffffff" if self.dark_mode else "#000000"
        ax2.scatter([c_pt[0]], [c_pt[1]], color="#059669" if self.dark_mode else "#047857", s=110, edgecolors=edge_col, linewidth=1.5, zorder=5)
        ax2.scatter([f_pt[0]], [f_pt[1]], color="#ea580c" if self.dark_mode else "#c2410c", s=110, edgecolors=edge_col, linewidth=1.5, zorder=5)
        ax2.plot([c_pt[0], f_pt[0]], [c_pt[1], f_pt[1]], color="#818cf8" if self.dark_mode else "#6366f1", linestyle="-", linewidth=2.0,
                 label=f"Selected Pair (ΔV={res.delta_v:.1f})")

        ax2.annotate(res.crown.name, c_pt, textcoords="offset points", xytext=(-10, 8),
                     fontweight="bold", fontsize=8, color="#34d399" if self.dark_mode else "#047857")
        ax2.annotate(res.flint.name, f_pt, textcoords="offset points", xytext=(8, -10),
                     fontweight="bold", fontsize=8, color="#fb923c" if self.dark_mode else "#c2410c")

        # Invert x-axis to follow optical convention (high Abbe on left)
        ax2.set_xlim(max(95.0, res.crown.vd + 10), min(18.0, res.flint.vd - 5))
        ax2.set_xlabel("Abbe Number Vd (Inverted)", fontsize=10)
        ax2.set_ylabel("Refractive Index nd", fontsize=10)
        ax2.set_title("Abbe Diagram (nd vs Vd)", fontsize=11, fontweight="bold")
        ax2.legend(fontsize=8, loc="upper right")
        apply_axes_theme(ax2, self.dark_mode)

        fig1.tight_layout()
        self.tab_dispersion.canvas.draw()

        # 2. Tab 2: Aberration Reduction Comparison
        fig2 = self.tab_aberration.fig
        fig2.clear()
        fig2.patch.set_facecolor(colors["bg_card"])
        ax_bar = fig2.add_subplot(1, 1, 1)

        labels = ["Spherical S_I (mm)", "Longitudinal Color C_L (mm)", "Coma S_II (mm)"]
        init_vals = [abs(res.initial_s_I), abs(res.initial_c_L), abs(res.initial_seidel.get("S_II", 0.0))]
        opt_vals = [abs(res.optimized_s_I), abs(res.optimized_c_L), abs(res.optimized_seidel.get("S_II", 0.0))]

        x = np.arange(len(labels))
        width = 0.35

        bar_before_col = "#64748b" if self.dark_mode else "#94a3b8"
        bar_after_col = "#22c55e" if self.dark_mode else "#16a34a"
        ax_bar.bar(x - width/2, init_vals, width, label="Before Surface Optimization", color=bar_before_col)
        ax_bar.bar(x + width/2, opt_vals, width, label="After Surface Optimization", color=bar_after_col)

        ax_bar.set_yscale("log")
        ax_bar.set_ylabel("Absolute Aberration Magnitude (mm, log scale)", fontsize=10)
        ax_bar.set_title("Aberration Reduction via Optimal Doublet Bending", fontsize=12, fontweight="bold")
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(labels, fontsize=10, fontweight="bold")
        ax_bar.legend(fontsize=9)
        apply_axes_theme(ax_bar, self.dark_mode)

        fig2.tight_layout()
        self.tab_aberration.canvas.draw()

        # 3. Tab 3: Chromatic Focal Shift Curve
        fig3 = self.tab_focal_shift.fig
        fig3.clear()
        fig3.patch.set_facecolor(colors["bg_card"])
        ax_shift = fig3.add_subplot(1, 1, 1)

        wls = [pt[0] for pt in res.chromatic_focal_shift]
        shifts_um = [pt[1] * 1000.0 for pt in res.chromatic_focal_shift]  # convert to um

        shift_col = "#a78bfa" if self.dark_mode else "#8b5cf6"
        ax_shift.plot(wls, shifts_um, color=shift_col, linewidth=2.5, label="Chromatic Focal Shift Δf(λ)")
        ax_shift.axhline(0.0, color="#94a3b8" if self.dark_mode else "#64748b", linestyle="--", alpha=0.7)
        ax_shift.axvline(486.1, color="#3b82f6", linestyle=":", label="F line (486 nm)")
        ax_shift.axvline(587.6, color="#eab308", linestyle=":", label="d line (588 nm)")
        ax_shift.axvline(656.3, color="#ef4444", linestyle=":", label="C line (656 nm)")

        ax_shift.set_xlabel("Wavelength (nm)", fontsize=10)
        ax_shift.set_ylabel("Focal Shift ΔBFL (μm)", fontsize=10)
        ax_shift.set_title(f"Achromatic Secondary Spectrum Curve (f = {res.efl:.1f} mm)", fontsize=11, fontweight="bold")
        ax_shift.legend(fontsize=8, loc="best")
        apply_axes_theme(ax_shift, self.dark_mode)

        fig3.tight_layout()
        self.tab_focal_shift.canvas.draw()

        # 4. Tab 4: Transverse Ray Fans
        fig4 = self.tab_ray_fan.fig
        fig4.clear()
        fig4.patch.set_facecolor(colors["bg_card"])
        ax_fan = fig4.add_subplot(1, 1, 1)

        fan_colors = {
            "F (486.1 nm)": "#60a5fa" if self.dark_mode else "#2563eb",
            "d (587.6 nm)": "#4ade80" if self.dark_mode else "#16a34a",
            "C (656.3 nm)": "#f87171" if self.dark_mode else "#dc2626"
        }
        for flabel, fdata in res.ray_fans.items():
            py = fdata["py"]
            ey_um = [e * 1000.0 for e in fdata["ey"]]  # convert mm to um
            ax_fan.plot(py, ey_um, label=flabel, color=fan_colors.get(flabel, "#38bdf8" if self.dark_mode else "#000000"), linewidth=2.0)

        ax_fan.set_xlabel("Normalized Pupil Height Py", fontsize=10)
        ax_fan.set_ylabel("Transverse Ray Intercept Ey (μm)", fontsize=10)
        ax_fan.set_title("Transverse Ray Aberration Fans Ey(Py)", fontsize=11, fontweight="bold")
        ax_fan.axhline(0.0, color="#64748b" if self.dark_mode else "#94a3b8", linestyle="-", alpha=0.5)
        ax_fan.axvline(0.0, color="#64748b" if self.dark_mode else "#94a3b8", linestyle="-", alpha=0.5)
        ax_fan.legend(fontsize=9, loc="best")
        apply_axes_theme(ax_fan, self.dark_mode)

        fig4.tight_layout()
        self.tab_ray_fan.canvas.draw()

    def _export_prescription(self):
        if not self.current_result:
            return
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Achromatic Doublet Prescription", "achromatic_doublet.json", "JSON Files (*.json)"
        )
        if filename:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(self.current_result.to_dict(), f, indent=2)
            self.lbl_status.setText(f"Exported prescription to {os.path.basename(filename)}")

    def _insert_into_project(self):
        if not self.current_result or not self.model:
            import sys
            is_headless = os.environ.get("QT_QPA_PLATFORM") == "offscreen" or "pytest" in sys.modules or "unittest" in sys.modules
            if not is_headless and QT_AVAILABLE and hasattr(QMessageBox, 'information'):
                QMessageBox.information(
                    self, "Insert Doublet",
                    "No active project or model available to insert surfaces into."
                )
            return

        res = self.current_result
        semi_diam = float(res.spec.diameter) / 2.0
        try:
            try:
                from v2.engine.lde import SurfaceStack
                from v2.engine.surface import Surface, SurfaceType
            except ImportError:
                from optics_app.engine.lde import SurfaceStack
                from optics_app.engine.surface import Surface, SurfaceType

            new_stack = SurfaceStack()
            
            # S0: Object
            s0 = Surface(
                name="S0 (OBJ)",
                surface_type=SurfaceType.OBJECT,
                radius=0.0,
                thickness=1000.0,
                glass="AIR",
                semi_diameter=semi_diam
            )
            new_stack.add_surface(s0)

            if res.spec.doublet_type.lower() == "cemented":
                s1 = Surface(
                    name="S1 (STO)",
                    surface_type=SurfaceType.STOP,
                    radius=float(res.radii[0]),
                    thickness=float(res.thicknesses[0]),
                    glass=res.crown.name,
                    semi_diameter=semi_diam,
                    radius_variable=True,
                    thickness_variable=True
                )
                r2 = float(res.radii[1])
                if abs(r2) > 1e7 or getattr(res.spec, 'flat_interface', False):
                    r2 = 0.0
                s2 = Surface(
                    name="S2 (Contact)",
                    surface_type=SurfaceType.STANDARD,
                    radius=r2,
                    thickness=float(res.thicknesses[1]),
                    glass=res.flint.name,
                    semi_diameter=semi_diam,
                    radius_variable=True,
                    thickness_variable=True
                )
                s3 = Surface(
                    name="S3 (Rear)",
                    surface_type=SurfaceType.STANDARD,
                    radius=float(res.radii[2]),
                    thickness=max(1.0, float(res.bfl)),
                    glass="AIR",
                    semi_diameter=semi_diam,
                    radius_variable=True,
                    thickness_variable=True
                )
                new_stack.add_surface(s1)
                new_stack.add_surface(s2)
                new_stack.add_surface(s3)
            else:
                s1 = Surface(
                    name="S1 (STO)",
                    surface_type=SurfaceType.STOP,
                    radius=float(res.radii[0]),
                    thickness=float(res.thicknesses[0]),
                    glass=res.crown.name,
                    semi_diameter=semi_diam,
                    radius_variable=True,
                    thickness_variable=True
                )
                s2 = Surface(
                    name="S2 (Crown Rear)",
                    surface_type=SurfaceType.STANDARD,
                    radius=float(res.radii[1]),
                    thickness=float(res.thicknesses[1]),
                    glass="AIR",
                    semi_diameter=semi_diam,
                    radius_variable=True,
                    thickness_variable=True
                )
                s3 = Surface(
                    name="S3 (Flint Front)",
                    surface_type=SurfaceType.STANDARD,
                    radius=float(res.radii[2]),
                    thickness=float(res.thicknesses[2]),
                    glass=res.flint.name,
                    semi_diameter=semi_diam,
                    radius_variable=True,
                    thickness_variable=True
                )
                s4 = Surface(
                    name="S4 (Flint Rear)",
                    surface_type=SurfaceType.STANDARD,
                    radius=float(res.radii[3]),
                    thickness=max(1.0, float(res.bfl)),
                    glass="AIR",
                    semi_diameter=semi_diam,
                    radius_variable=True,
                    thickness_variable=True
                )
                new_stack.add_surface(s1)
                new_stack.add_surface(s2)
                new_stack.add_surface(s3)
                new_stack.add_surface(s4)

            # S_ima: Detector
            s_det = Surface(
                name=f"S{len(new_stack.surfaces)} (IMA)",
                surface_type=SurfaceType.DETECTOR,
                radius=0.0,
                thickness=0.0,
                glass="AIR",
                semi_diameter=max(5.0, semi_diam * 0.8)
            )
            new_stack.add_surface(s_det)

            # Push undo and deliver to model
            if hasattr(self.model, '_push_undo'):
                self.model._push_undo("Insert Achromatic Doublet")
            if hasattr(self.model, "set_surface_stack"):
                self.model.set_surface_stack(new_stack)
            elif hasattr(self.model, "surfaces"):
                self.model.surfaces = list(new_stack.surfaces)

            # Notify parent window
            parent = self.parent() or getattr(self, '_parent', None)
            if parent:
                if hasattr(parent, 'on_trace_rays'):
                    parent.on_trace_rays()
                if hasattr(parent, 'optimizer_dock') and parent.optimizer_dock:
                    if hasattr(parent.optimizer_dock, '_rebuild_surface_checkboxes'):
                        parent.optimizer_dock._rebuild_surface_checkboxes()
                    if hasattr(parent.optimizer_dock, 'update_system_info'):
                        parent.optimizer_dock.update_system_info()
                if hasattr(parent, 'analysis_tabs') and parent.analysis_tabs:
                    if hasattr(parent.analysis_tabs, 'refresh_analysis'):
                        parent.analysis_tabs.refresh_analysis()
                if hasattr(parent, 'table_view') and parent.table_view:
                    try:
                        parent.table_view.viewport().update()
                    except Exception:
                        pass

            import sys
            is_headless = os.environ.get("QT_QPA_PLATFORM") == "offscreen" or "pytest" in sys.modules or "unittest" in sys.modules
            if not is_headless and QT_AVAILABLE and hasattr(QMessageBox, 'information'):
                QMessageBox.information(
                    self, "Doublet Inserted",
                    f"✅ Achromatic Doublet Successfully Delivered to Application!\n\n"
                    f"• Glasses: {res.crown.name} (Crown) + {res.flint.name} (Flint)\n"
                    f"• Target EFL: {res.spec.target_efl:.1f} mm | Achieved EFL: {res.efl:.2f} mm\n"
                    f"• Back Focal Length (BFL): {res.bfl:.2f} mm\n"
                    f"• Clear Aperture: Ø {res.spec.diameter:.1f} mm\n"
                    f"• Radii: R1={res.radii[0]:.2f}, R2={res.radii[1]:.2f}, R3={res.radii[2]:.2f} mm\n\n"
                    f"The Lens Data Editor (LDE), 2D/3D CAD visualizers, and ray tracer have been fully updated."
                )
            self.lbl_status.setText("Achromatic doublet prescription delivered to project.")
            if hasattr(self, 'accept'):
                self.accept()
        except Exception as e:
            import sys
            is_headless = os.environ.get("QT_QPA_PLATFORM") == "offscreen" or "pytest" in sys.modules or "unittest" in sys.modules
            if not is_headless and QT_AVAILABLE and hasattr(QMessageBox, 'warning'):
                QMessageBox.warning(self, "Insertion Error", f"Could not insert doublet surfaces: {str(e)}")
