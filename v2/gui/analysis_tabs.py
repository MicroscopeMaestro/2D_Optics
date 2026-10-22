"""
Embedded Matplotlib Analysis Tab Suite for 2D Optics Simulation Engine.
Tab 1: Spot Diagram Viewer (RMS radius, GEO radius, Airy disk comparison circle).
Tab 2: Ray Fan Viewer (Transverse ray aberration EX, EY vs Py, 3rd-order Seidel coefficients).
Tab 3: Grating Dispersion Viewer (Spatial separation delta_x, delta_y vs wavelength).
Tab 4: 2D CCD Image Heatmap Viewer (Pixelated intensity matrix heatmap and spatial profile).
Tab 5: Laser Scanning & Field Aberrations.
Tab 6: Femtosecond Pulse Width, GDD/TOD, and virtual path probes.
"""

import sys
import os
import math
import json
import warnings
import numpy as np
from typing import Optional, List, Dict, Any, Union

# Suppress Matplotlib tight_layout margin decoration warnings globally
warnings.filterwarnings("ignore", message=".*Tight layout not applied.*")
warnings.filterwarnings("ignore", category=UserWarning, module=".*matplotlib.*")

from optics_app.gui.qt_compat import (
    QTabWidget, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QDoubleSpinBox, QSpinBox, QPushButton, QFileDialog, QT_AVAILABLE, QComboBox, QCheckBox, Qt
)

# Matplotlib Qt Canvas handling
HAS_MPL_QT = False
try:
    import matplotlib
    matplotlib.use('QtAgg' if QT_AVAILABLE else 'Agg')
    from matplotlib.figure import Figure
    if QT_AVAILABLE:
        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
            from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
            HAS_MPL_QT = True
        except ImportError:
            try:
                from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
                from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
                HAS_MPL_QT = True
            except ImportError:
                HAS_MPL_QT = False
    else:
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        NavigationToolbar = None
except ImportError:
    HAS_MPL_QT = False


# Imports from optics_app analysis and engine
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.analysis.spot import SpotDiagram, generate_spot_diagram_from_system
from optics_app.analysis.best_focus import BestFocusResult, analyze_system_best_focus
from optics_app.analysis.rayfan import RayFanAnalysis, calculate_ray_fan
from optics_app.analysis.aberration import (
    calculate_seidel_coefficients,
    SeidelAberrations,
    LensAberrationResult,
    generate_zemax_seidel_report,
)
from optics_app.analysis.grating_dispersion import GratingDispersionAnalysis
from optics_app.analysis.pulse_width import PulseAnalysisConfig, PulseWidthAnalysis
from optics_app.analysis.two_photon import (
    TwoPhotonMicroscopeConfig,
    TwoPhotonMicroscopeSimulator,
    build_calibrated_template,
    build_two_photon_surface_template,
)
from optics_app.analysis.nonlinear_imaging import (
    NonlinearImagingConfig, NonlinearImagingSimulator,
    build_nonlinear_surface_template,
)
from optics_app.engine.surface import SurfaceType
from optics_app.engine.ccd import CCDSensor
from optics_app.engine.first_order import calculate_first_order_data, format_zemax_first_order_report
from optics_app.engine.manufacturing_standards import (
    QualityPreset,
    ManufacturingConstraintsConfig,
    TestPlateCatalog,
    audit_optical_system_manufacturing,
    SystemManufacturingReport,
    ElementManufacturingAudit,
)
try:
    from optics_app.analysis.spectrometer_quality import (
        SpectrometerQualityAnalyzer,
        SpectrometerConfiguration,
        SpectrometerQualitySummary,
        WavelengthSpectrometerResult,
    )
except ImportError:
    from v2.analysis.spectrometer_quality import (
        SpectrometerQualityAnalyzer,
        SpectrometerConfiguration,
        SpectrometerQualitySummary,
        WavelengthSpectrometerResult,
    )




try:
    from optics_app.gui.theme import (
        get_theme_colors, apply_axes_theme, get_text_bbox, style_colorbar_for_theme
    )
except ImportError:
    try:
        from v2.gui.theme import (
            get_theme_colors, apply_axes_theme, get_text_bbox, style_colorbar_for_theme
        )
    except Exception:
        def get_theme_colors(dark_mode: bool):
            return {
                "bg_fig": "#0f172a" if dark_mode else "#ffffff",
                "bg_ax": "#0b0f19" if dark_mode else "#ffffff",
                "bg_card": "#1e293b" if dark_mode else "#f8fafc",
                "fg_text": "#f1f5f9" if dark_mode else "#0f172a",
                "fg_subtext": "#94a3b8" if dark_mode else "#475569",
                "grid_col": "#334155" if dark_mode else "#cbd5e1",
                "spine_col": "#334155" if dark_mode else "#cbd5e1",
                "axis_line_col": "#64748b",
                "accent_col": "#38bdf8" if dark_mode else "#0284c7",
                "success_col": "#4ade80" if dark_mode else "#16a34a",
                "warning_col": "#fbbf24" if dark_mode else "#d97706",
                "danger_col": "#f87171" if dark_mode else "#dc2626",
                "bbox_face": "#1e293b" if dark_mode else "#f8fafc",
                "bbox_edge": "#334155" if dark_mode else "#cbd5e1",
            }

        def apply_axes_theme(ax, dark_mode, grid_alpha=None):
            if ax is None:
                return
            col = get_theme_colors(dark_mode)
            try:
                ax.set_facecolor(col["bg_ax"])
                ax.tick_params(colors=col["fg_subtext"])
                for sp in getattr(ax, "spines", {}).values():
                    sp.set_color(col["spine_col"])
            except Exception:
                pass

        def get_text_bbox(dark_mode, pad=0.5, alpha=0.92):
            return {
                "boxstyle": f"round,pad={pad}",
                "facecolor": "#1e293b" if dark_mode else "#f8fafc",
                "edgecolor": "#334155" if dark_mode else "#cbd5e1",
                "alpha": alpha,
                "linewidth": 1.0,
            }

        def style_colorbar_for_theme(cbar, dark_mode):
            pass


class BaseAnalysisTab(QWidget if QT_AVAILABLE else object):
    """Base Matplotlib Canvas Tab Wrapper."""
    def __init__(self, parent=None):
        if QT_AVAILABLE:
            super().__init__(parent)
            layout = QVBoxLayout(self)
            layout.setContentsMargins(2, 2, 2, 2)
            self.fig = Figure(figsize=(7, 5), dpi=100)
            self.canvas = FigureCanvas(self.fig)
            self.toolbar = NavigationToolbar(self.canvas, self) if NavigationToolbar else None
            if self.toolbar:
                layout.addWidget(self.toolbar)
            layout.addWidget(self.canvas)
        else:
            self.fig = None
            self.canvas = None
            self.toolbar = None
        self.dark_mode: bool = False

    def set_dark_mode(self, enabled: bool) -> None:
        """Toggle dark mode background and styling for matplotlib canvas."""
        self.dark_mode = bool(enabled)
        if self.fig is not None:
            self.fig.patch.set_facecolor('#0f172a' if self.dark_mode else '#ffffff')
        if hasattr(self, '_replot'):
            try:
                self._replot()
            except Exception:
                pass
        elif hasattr(self, 'refresh_tab'):
            try:
                self.refresh_tab()
            except Exception:
                pass
        elif hasattr(self, 'update_plot'):
            try:
                self.update_plot()
            except Exception:
                pass
        if self.canvas is not None and hasattr(self.canvas, 'draw_idle'):
            self.canvas.draw_idle()

    def apply_dark_mode_to_axes(self, *axes, grid_alpha: Optional[float] = None):
        """Helper to style figure and axes consistently according to dark_mode."""
        is_dark = getattr(self, 'dark_mode', False)
        if self.fig is not None:
            self.fig.patch.set_facecolor('#0f172a' if is_dark else '#ffffff')
        for ax in axes:
            if ax is None:
                continue
            apply_axes_theme(ax, is_dark, grid_alpha=grid_alpha)

    def get_text_bbox(self, pad: float = 0.5, alpha: float = 0.92) -> Dict[str, Any]:
        """Returns standard Matplotlib bbox dictionary matching current theme."""
        return get_text_bbox(getattr(self, 'dark_mode', False), pad=pad, alpha=alpha)

    def get_theme_colors(self) -> Dict[str, str]:
        """Returns current theme colors dict."""
        return get_theme_colors(getattr(self, 'dark_mode', False))

    def apply_tight_layout(self, pad: float = 1.08):
        """Applies tight layout safely without raising UserWarning if margins are constrained."""
        if self.fig is None:
            return
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.fig.tight_layout(pad=pad)
        except Exception:
            try:
                self.fig.subplots_adjust(left=0.12, right=0.95, top=0.92, bottom=0.12, hspace=0.35)
            except Exception:
                pass


class SpotDiagramTab(BaseAnalysisTab):
    """Tab 1: Zemax-Style Spot Diagram Viewer with Multi-Field Matrix Support."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.last_system = None
        self.last_spot_diagram = None
        self.last_wavelengths = None
        self.last_target_surface_index = -1
        self.last_source_config = None
        self.last_system_config = None
        self.selected_field_idx = -1  # -1 = All Fields (Matrix), >=0 = specific field index

        if QT_AVAILABLE:
            ctrl_layout = QHBoxLayout()
            ctrl_layout.setContentsMargins(4, 2, 4, 2)
            lbl = QLabel("Field:")
            lbl.setStyleSheet("font-weight: bold; font-size: 11px;")
            ctrl_layout.addWidget(lbl)

            self.combo_field = QComboBox()
            self.combo_field.addItem("All Fields (Matrix)", -1)
            self.combo_field.setStyleSheet(
                "QComboBox { background-color: #f1f5f9; border: 1px solid #cbd5e1; border-radius: 3px; "
                "padding: 2px 4px; font-size: 11px; font-weight: bold; color: #1e293b; }"
            )
            self.combo_field.currentIndexChanged.connect(self._on_field_changed)
            ctrl_layout.addWidget(self.combo_field)
            ctrl_layout.addStretch(1)

            layout = self.layout()
            if layout is not None:
                idx = 1 if hasattr(self, 'toolbar') and self.toolbar else 0
                layout.insertLayout(idx, ctrl_layout)

    def _on_field_changed(self, index: int):
        if not hasattr(self, 'combo_field'):
            return
        data = self.combo_field.currentData()
        self.selected_field_idx = int(data) if data is not None else -1
        self._replot()

    def _replot(self):
        self.update_plot(
            system=self.last_system,
            spot_diagram=self.last_spot_diagram,
            wavelengths=self.last_wavelengths,
            target_surface_index=self.last_target_surface_index,
            source_config=self.last_source_config,
            system_config=self.last_system_config,
        )

    def update_plot(
        self,
        system: Optional[OpticalSystem] = None,
        spot_diagram: Optional[SpotDiagram] = None,
        wavelengths: Optional[List[float]] = None,
        target_surface_index: int = -1,
        source_config: Optional[Any] = None,
        system_config: Optional[Any] = None,
    ):
        if not (QT_AVAILABLE and HAS_MPL_QT) or self.fig is None:
            return

        self.last_system = system
        self.last_spot_diagram = spot_diagram
        self.last_wavelengths = wavelengths
        self.last_target_surface_index = target_surface_index
        self.last_source_config = source_config
        self.last_system_config = system_config

        # Update combo_field items if system_config fields available
        if hasattr(self, 'combo_field') and system_config is not None and getattr(system_config, 'fields', None):
            fields = system_config.fields
            ftype = getattr(system_config, 'field_type', None)
            unit_str = "°" if str(getattr(ftype, 'value', ftype)).lower() == "angle" else "mm"

            self.combo_field.blockSignals(True)
            cur_data = self.combo_field.currentData()
            self.combo_field.clear()
            self.combo_field.addItem("All Fields (Matrix)", -1)
            for f_i, f in enumerate(fields):
                self.combo_field.addItem(f"Field {f_i+1} ({f.x:g}{unit_str}, {f.y:g}{unit_str})", f_i)
            found_idx = self.combo_field.findData(cur_data)
            if found_idx >= 0:
                self.combo_field.setCurrentIndex(found_idx)
                self.selected_field_idx = cur_data
            else:
                self.combo_field.setCurrentIndex(0)
                self.selected_field_idx = -1
            self.combo_field.blockSignals(False)

        self.fig.clf()
        palette = ['red', 'green', 'blue', 'orange', 'purple', 'cyan', 'magenta', 'brown', 'pink']
        import matplotlib.patches as patches

        # Direct SpotDiagram object supplied
        if spot_diagram is not None:
            ax = self.fig.add_subplot(111)
            ax.scatter(spot_diagram.x, spot_diagram.y, color=self.get_theme_colors()['accent_col'], s=10, alpha=0.7, label='Ray Hits')
            cx, cy = spot_diagram.calculate_centroid()
            airy_rad = spot_diagram.calculate_airy_disk_radius()
            circle = patches.Circle((cx, cy), airy_rad * 1e-3, color='#f87171' if self.dark_mode else 'red', fill=False, linestyle='--', linewidth=1.5, label=f"Airy Disk (r={airy_rad:.2f} µm)")
            ax.add_patch(circle)
            rms_r_um = spot_diagram.calculate_rms_radius() * 1e3
            geo_r_um = spot_diagram.calculate_geo_radius() * 1e3
            stats_text = (
                f"RMS Radius: {rms_r_um:6.2f} µm\n"
                f"GEO Radius: {geo_r_um:6.2f} µm\n"
                f"Airy Radius:{airy_rad:6.2f} µm\n"
                f"Centroid: ({cx:+.3f}, {cy:+.3f}) mm"
            )
            props = self.get_text_bbox(pad=0.5, alpha=0.9)
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=8, verticalalignment='top', bbox=props, fontfamily='monospace', color=self.get_theme_colors()['fg_text'])
            ax.set_title("Zemax Spot Diagram (Detector Plane)", fontsize=10, fontweight='bold')
            ax.set_xlabel("X Position (mm)")
            ax.set_ylabel("Y Position (mm)")
            ax.legend(loc='upper right', fontsize=8)
            ax.set_aspect('equal', 'datalim')
            self.apply_dark_mode_to_axes(ax)

        elif system is not None:
            fields = getattr(system_config, 'fields', []) if system_config is not None else []
            ftype = getattr(system_config, 'field_type', None) if system_config is not None else None
            unit_str = "°" if str(getattr(ftype, 'value', ftype)).lower() == "angle" else "mm"
            wls = wavelengths or ([0.5876] if not system_config or not system_config.wavelength_list_um else system_config.wavelength_list_um)

            # Multi-panel matrix mode
            if self.selected_field_idx == -1 and len(fields) > 1:
                n_f = len(fields)
                nrows = 1 if n_f <= 3 else (2 if n_f <= 6 else 3)
                ncols = min(n_f, 3) if n_f <= 3 else (3 if n_f <= 6 else 3)
                spot_cache = {}

                for k, f in enumerate(fields):
                    if k >= nrows * ncols:
                        break
                    ax_k = self.fig.add_subplot(nrows, ncols, k + 1)
                    field_spots = []
                    for wl_idx, wl in enumerate(wls):
                        col = palette[wl_idx % len(palette)]
                        cache_key = (round(float(f.x), 6), round(float(f.y), 6), wl, target_surface_index)
                        if cache_key in spot_cache:
                            sd = spot_cache[cache_key]
                        else:
                            sd = generate_spot_diagram_from_system(
                                system, num_rays=30, wavelength_um=wl,
                                target_surface_index=target_surface_index,
                                field_point=(f.x, f.y), field_type=ftype,
                                system_config=system_config
                            )
                            spot_cache[cache_key] = sd
                        if len(sd.x) > 0:
                            field_spots.append((sd, col, f"{wl*1000:.0f} nm"))

                    if field_spots:
                        for sd, col, lbl in field_spots:
                            ax_k.scatter(sd.x, sd.y, color=col, s=8, alpha=0.7)
                        all_x_k = np.concatenate([s.x for s, _, _ in field_spots])
                        all_y_k = np.concatenate([s.y for s, _, _ in field_spots])
                        comb_sd = SpotDiagram(all_x_k, all_y_k, wavelength_um=wls[0])
                        cx_k, cy_k = comb_sd.calculate_centroid()
                        airy_k = max(s.calculate_airy_disk_radius() for s, _, _ in field_spots)
                        circle = patches.Circle((cx_k, cy_k), airy_k * 1e-3, color='#f87171' if self.dark_mode else 'red', fill=False, linestyle='--', linewidth=1.2)
                        ax_k.add_patch(circle)
                        rms_r = comb_sd.calculate_rms_radius() * 1e3
                        geo_r = comb_sd.calculate_geo_radius() * 1e3
                        diff_lim = "YES" if rms_r <= airy_k else "NO"
                        ax_k.set_title(f"Field {k+1}: ({f.x:g}{unit_str}, {f.y:g}{unit_str})", fontsize=8.5, fontweight='bold')
                        txt = f"RMS: {rms_r:.2f} µm\nAiry: {airy_k:.2f} µm\nDiffLim: {diff_lim}"
                        ax_k.text(0.03, 0.97, txt, transform=ax_k.transAxes, fontsize=7, verticalalignment='top',
                                  bbox=self.get_text_bbox(pad=0.2, alpha=0.88), color=self.get_theme_colors()['fg_text'])
                        ax_k.set_aspect('equal', 'datalim')
                        ax_k.tick_params(labelsize=7)
                    else:
                        from optics_app.engine.mechanisms import available_states, active_state_name
                        mech_off = False
                        if system and getattr(system, 'surface_stack', None):
                            for s in system.surface_stack.surfaces:
                                if available_states(s) and active_state_name(s) == "OFF":
                                    mech_off = True
                                    break
                        if mech_off:
                            ax_k.text(0.5, 0.5, "Mechanism OFF\n(Beam Dump)", horizontalalignment='center', verticalalignment='center',
                                      fontsize=8, fontweight='bold', color='#f87171' if self.dark_mode else '#991b1b')
                        else:
                            ax_k.text(0.5, 0.5, "No Spot", horizontalalignment='center', verticalalignment='center', fontsize=8, color=self.get_theme_colors()['fg_subtext'])
                        ax_k.set_title(f"Field {k+1}: ({f.x:g}{unit_str}, {f.y:g}{unit_str})", fontsize=8.5, fontweight='bold')
                        ax_k.set_aspect('equal', 'datalim')
                    self.apply_dark_mode_to_axes(ax_k)

                self.fig.suptitle("Zemax Multi-Field Spot Diagram Matrix", fontsize=9.5, fontweight='bold', color=self.get_theme_colors()['fg_text'])

            else:
                # Single Field mode
                ax = self.fig.add_subplot(111)
                all_spots = []
                f_idx = self.selected_field_idx if (0 <= self.selected_field_idx < len(fields)) else 0
                cur_field = fields[f_idx] if len(fields) > 0 else None
                f_pt = (cur_field.x, cur_field.y) if cur_field is not None else None
                f_title = f" - Field {f_idx+1} ({cur_field.x:g}{unit_str}, {cur_field.y:g}{unit_str})" if cur_field is not None else ""

                for idx, wl in enumerate(wls):
                    color = palette[idx % len(palette)]
                    sd = generate_spot_diagram_from_system(
                        system, num_rays=150, wavelength_um=wl,
                        target_surface_index=target_surface_index,
                        source_config=source_config,
                        field_point=f_pt, field_type=ftype,
                        system_config=system_config
                    )
                    if len(sd.x) > 0:
                        all_spots.append((sd, color, f'{wl*1000:.0f} nm'))

                if all_spots:
                    cx_total, cy_total = 0.0, 0.0
                    total_points = 0
                    max_airy = 0.0
                    for sd, color, label in all_spots:
                        ax.scatter(sd.x, sd.y, color=color, s=10, alpha=0.7, label=label)
                        cx, cy = sd.calculate_centroid()
                        n_pts = len(sd.x)
                        cx_total += cx * n_pts
                        cy_total += cy * n_pts
                        total_points += n_pts
                        max_airy = max(max_airy, sd.calculate_airy_disk_radius())

                    if total_points > 0:
                        cx_total /= total_points
                        cy_total /= total_points

                    airy_circle = patches.Circle((cx_total, cy_total), max_airy * 1e-3, color='#f87171' if self.dark_mode else 'red', fill=False, linestyle='--', linewidth=1.5, label=f"Airy Disk (r={max_airy:.2f} µm)")
                    ax.add_patch(airy_circle)

                    all_x = np.concatenate([sd.x for sd, _, _ in all_spots])
                    all_y = np.concatenate([sd.y for sd, _, _ in all_spots])

                    if len(all_x) > 1:
                        combined_sd = SpotDiagram(all_x, all_y, wavelength_um=wls[0])
                        rms_r_um = combined_sd.calculate_rms_radius() * 1e3
                        rms_x_um = combined_sd.calculate_rms_x() * 1e3
                        rms_y_um = combined_sd.calculate_rms_y() * 1e3
                        geo_r_um = combined_sd.calculate_geo_radius() * 1e3
                        geo_x_um = combined_sd.calculate_geo_x() * 1e3
                        geo_y_um = combined_sd.calculate_geo_y() * 1e3

                        stats_text = (
                            f"RMS Radius: {rms_r_um:6.2f} µm\n"
                            f"RMS Spot X: {rms_x_um:6.2f} µm\n"
                            f"RMS Spot Y: {rms_y_um:6.2f} µm\n"
                            f"GEO Radius: {geo_r_um:6.2f} µm\n"
                            f"GEO Spot X: {geo_x_um:6.2f} µm\n"
                            f"GEO Spot Y: {geo_y_um:6.2f} µm\n"
                            f"Airy Radius:{max_airy:6.2f} µm\n"
                            f"Centroid: ({cx_total:+.3f}, {cy_total:+.3f}) mm"
                        )
                        props = self.get_text_bbox(pad=0.5, alpha=0.9)
                        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=8,
                                verticalalignment='top', horizontalalignment='left', bbox=props, fontfamily='monospace', color=self.get_theme_colors()['fg_text'])

                    ax.set_title(f"Zemax Spot Diagram (Detector Plane){f_title}", fontsize=10, fontweight='bold')
                    ax.set_xlabel("X Position (mm)")
                    ax.set_ylabel("Y Position (mm)")
                    ax.legend(loc='upper right', fontsize=8)
                    ax.set_aspect('equal', 'datalim')
                    self.apply_dark_mode_to_axes(ax)
                else:
                    from optics_app.engine.mechanisms import is_dmd, active_state_name
                    dmd_off = False
                    if system and getattr(system, 'surface_stack', None):
                        for s in system.surface_stack.surfaces:
                            if is_dmd(s) and active_state_name(s) == "OFF":
                                dmd_off = True
                                break
                    if dmd_off:
                        bg_dmd = '#450a0a' if self.dark_mode else '#fee2e2'
                        border_dmd = '#ef4444' if self.dark_mode else '#fca5a5'
                        text_dmd = '#fca5a5' if self.dark_mode else '#991b1b'
                        ax.text(0.5, 0.55, "🔲 DMD State: OFF (Routed to Beam Dump)\nDetector Dark (0 hits - Beam Deflected 48°)", 
                                horizontalalignment='center', verticalalignment='center',
                                fontsize=11, fontweight='bold', color=text_dmd,
                                bbox=dict(boxstyle='round,pad=0.6', facecolor=bg_dmd, edgecolor=border_dmd))
                        ax.set_title(f"Zemax Spot Diagram (Detector Plane){f_title} - DMD OFF", fontsize=10, fontweight='bold')
                    else:
                        ax.text(0.5, 0.5, "No Ray Spot Data Available", horizontalalignment='center', verticalalignment='center', color=self.get_theme_colors()['fg_subtext'])
                    self.apply_dark_mode_to_axes(ax)
        else:
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, "No Optical System Loaded", horizontalalignment='center', verticalalignment='center', color=self.get_theme_colors()['fg_subtext'])
            self.apply_dark_mode_to_axes(ax)

        self.apply_tight_layout()
        if hasattr(self.canvas, "draw_idle"):
            self.canvas.draw_idle()
        elif hasattr(self.canvas, "draw"):
            self.canvas.draw()


class RayFanTab(BaseAnalysisTab):
    """Tab 2: Ray Fan Plot Viewer & 3rd-Order Seidel Coefficients with Multi-Field Support."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.analyzer = RayFanAnalysis(py_sampling=51)
        self.last_system = None
        self.last_target_surface_index = -1
        self.last_source_config = None
        self.last_system_config = None
        self.selected_field_idx = 0

        if QT_AVAILABLE:
            ctrl_layout = QHBoxLayout()
            ctrl_layout.setContentsMargins(4, 2, 4, 2)
            lbl = QLabel("Field:")
            lbl.setStyleSheet("font-weight: bold; font-size: 11px;")
            ctrl_layout.addWidget(lbl)

            self.combo_field = QComboBox()
            self.combo_field.addItem("Field 1 (0.0, 0.0)", 0)
            self.combo_field.setStyleSheet(
                "QComboBox { background-color: #f1f5f9; border: 1px solid #cbd5e1; border-radius: 3px; "
                "padding: 2px 4px; font-size: 11px; font-weight: bold; color: #1e293b; }"
            )
            self.combo_field.currentIndexChanged.connect(self._on_field_changed)
            ctrl_layout.addWidget(self.combo_field)

            lbl_seidel = QLabel("Seidel Mode:")
            lbl_seidel.setStyleSheet("font-weight: bold; font-size: 11px; margin-left: 10px;")
            ctrl_layout.addWidget(lbl_seidel)

            self.combo_seidel = QComboBox()
            self.combo_seidel.addItem("Lens Accumulation (Zemax)", "accumulation")
            self.combo_seidel.addItem("System Total", "total")
            self.combo_seidel.setStyleSheet(
                "QComboBox { background-color: #f1f5f9; border: 1px solid #cbd5e1; border-radius: 3px; "
                "padding: 2px 4px; font-size: 11px; font-weight: bold; color: #1e293b; }"
            )
            self.combo_seidel.currentIndexChanged.connect(self._replot)
            ctrl_layout.addWidget(self.combo_seidel)

            lbl_seidel_units = QLabel("Units:")
            lbl_seidel_units.setStyleSheet("font-weight: bold; font-size: 11px; margin-left: 8px;")
            ctrl_layout.addWidget(lbl_seidel_units)

            self.combo_seidel_units = QComboBox()
            self.combo_seidel_units.addItem("Micrometers (µm)", "um")
            self.combo_seidel_units.addItem("Millimeters (mm)", "mm")
            self.combo_seidel_units.setStyleSheet(
                "QComboBox { background-color: #f1f5f9; border: 1px solid #cbd5e1; border-radius: 3px; "
                "padding: 2px 4px; font-size: 11px; font-weight: bold; color: #1e293b; }"
            )
            self.combo_seidel_units.currentIndexChanged.connect(self._replot)
            ctrl_layout.addWidget(self.combo_seidel_units)
            ctrl_layout.addStretch(1)

            layout = self.layout()
            if layout is not None:
                idx = 1 if hasattr(self, 'toolbar') and self.toolbar else 0
                layout.insertLayout(idx, ctrl_layout)

    def _on_field_changed(self, index: int):
        if not hasattr(self, 'combo_field'):
            return
        data = self.combo_field.currentData()
        self.selected_field_idx = int(data) if data is not None else 0
        self._replot()

    def _replot(self):
        self.update_plot(
            system=self.last_system,
            target_surface_index=self.last_target_surface_index,
            source_config=self.last_source_config,
            system_config=self.last_system_config,
        )

    def update_plot(
        self,
        system: Optional[OpticalSystem] = None,
        target_surface_index: int = -1,
        source_config: Optional[Any] = None,
        system_config: Optional[Any] = None,
    ):
        if not (QT_AVAILABLE and HAS_MPL_QT) or self.fig is None:
            return

        self.last_system = system
        self.last_target_surface_index = target_surface_index
        self.last_source_config = source_config
        self.last_system_config = system_config

        fields = getattr(system_config, 'fields', []) if system_config is not None else []
        ftype = getattr(system_config, 'field_type', None) if system_config is not None else None
        unit_str = "°" if str(getattr(ftype, 'value', ftype)).lower() == "angle" else "mm"

        if hasattr(self, 'combo_field') and fields:
            self.combo_field.blockSignals(True)
            cur_data = self.combo_field.currentData()
            self.combo_field.clear()
            for f_i, f in enumerate(fields):
                self.combo_field.addItem(f"Field {f_i+1} ({f.x:g}{unit_str}, {f.y:g}{unit_str})", f_i)
            found_idx = self.combo_field.findData(cur_data)
            if found_idx >= 0:
                self.combo_field.setCurrentIndex(found_idx)
                self.selected_field_idx = cur_data
            else:
                self.combo_field.setCurrentIndex(0)
                self.selected_field_idx = 0
            self.combo_field.blockSignals(False)

        self.fig.clf()
        ax_fan = self.fig.add_subplot(211)
        ax_seidel = self.fig.add_subplot(212)

        if system is not None:
            f_idx = self.selected_field_idx if (0 <= self.selected_field_idx < len(fields)) else 0
            cur_field = fields[f_idx] if len(fields) > 0 else None
            f_pt = (cur_field.x, cur_field.y) if cur_field is not None else None
            f_title = f" - Field {f_idx+1} ({cur_field.x:g}{unit_str}, {cur_field.y:g}{unit_str})" if cur_field is not None else ""

            try:
                fan_data = self.analyzer.trace_ray_fan_from_system(
                    system,
                    target_surface_index=target_surface_index,
                    source_config=source_config,
                    field_point=f_pt,
                    field_type=ftype,
                    system_config=system_config,
                )
            except Exception:
                fan_data = None

            if fan_data:
                tang_col = '#38bdf8' if self.dark_mode else '#1e40af'
                sagit_col = '#4ade80' if self.dark_mode else '#16a34a'
                ax_fan.plot(fan_data.py, fan_data.tangential_ey, color=tang_col, linestyle='-', linewidth=1.5, label="EY(Py) Tangential")
                ax_fan.plot(fan_data.px, fan_data.sagittal_ex, color=sagit_col, linestyle='--', linewidth=1.3, label="EX(Px) Sagittal")

            ax_fan.set_title(f"Transverse Ray Aberration Fans{f_title}", fontsize=10, fontweight='bold')
            ax_fan.set_xlabel("Normalized Pupil Coordinate (Py, Px)")
            ax_fan.set_ylabel("Aberration (mm)")
            ax_fan.legend(loc='upper right', fontsize=8)

            seidel = calculate_seidel_coefficients(system)
            seidel_mode = self.combo_seidel.currentData() if hasattr(self, 'combo_seidel') else "accumulation"
            seidel_unit = self.combo_seidel_units.currentData() if hasattr(self, 'combo_seidel_units') else "um"
            seidel_scale = 1000.0 if seidel_unit == "um" else 1.0
            seidel_unit_lbl = "µm" if seidel_unit == "um" else "mm"

            axis_col = self.get_theme_colors()['axis_line_col']

            if seidel_mode == "accumulation" and seidel.lens_details:
                cat_names = [f"L{l.lens_index}:{l.lens_name[:8]}" for l in seidel.lens_details] + ["FINAL TOTAL"]
                x_idx = np.arange(len(cat_names))
                aberr_types = [
                    ("acc_S_I", "S_I (Sph)", "#38bdf8" if self.dark_mode else "#1e40af"),
                    ("acc_S_II", "S_II (Coma)", "#2dd4bf" if self.dark_mode else "#0d9488"),
                    ("acc_S_III", "S_III (Ast)", "#4ade80" if self.dark_mode else "#16a34a"),
                    ("acc_S_IV", "S_IV (Petz)", "#fbbf24" if self.dark_mode else "#d97706"),
                    ("acc_S_V", "S_V (Dist)", "#f87171" if self.dark_mode else "#dc2626"),
                ]
                width = min(0.8 / len(aberr_types), 0.15)
                for b_i, (attr, lbl, col) in enumerate(aberr_types):
                    y_vals = [(getattr(l, attr) * seidel_scale) for l in seidel.lens_details] + [getattr(seidel, attr[4:]) * seidel_scale]
                    offset = (b_i - (len(aberr_types) - 1) / 2.0) * width
                    bars = ax_seidel.bar(x_idx + offset, y_vals, width, label=lbl, color=col, alpha=0.85, edgecolor=axis_col, linewidth=0.5)
                    if len(bars) > 0:
                        bars[-1].set_edgecolor('#d97706')
                        bars[-1].set_linewidth(1.5)
                ax_seidel.set_title(f"3rd-Order Seidel Aberration Accumulation for Every Lens & Final Total ({seidel_unit_lbl})", fontsize=9, fontweight='bold')
                ax_seidel.set_ylabel(f"Aberration ({seidel_unit_lbl})", fontsize=8)
                ax_seidel.set_xticks(x_idx)
                ax_seidel.set_xticklabels(cat_names, rotation=20, ha='right', fontsize=7.5)
                ax_seidel.axhline(0, color=axis_col, linewidth=0.8)
                ax_seidel.legend(loc='upper right', fontsize=7, ncol=3)
            else:
                labels = ['S_I (Sph)', 'S_II (Coma)', 'S_III (Ast)', 'S_IV (Petz)', 'S_V (Dist)']
                values = [
                    seidel.S_I * seidel_scale,
                    seidel.S_II * seidel_scale,
                    seidel.S_III * seidel_scale,
                    seidel.S_IV * seidel_scale,
                    seidel.S_V * seidel_scale,
                ]
                colors = (
                    ['#38bdf8', '#2dd4bf', '#4ade80', '#fbbf24', '#f87171']
                    if self.dark_mode else
                    ['navy', 'teal', 'darkgreen', 'darkgoldenrod', 'crimson']
                )
                ax_seidel.bar(labels, values, color=colors, alpha=0.75, edgecolor=axis_col, linewidth=0.5)
                ax_seidel.set_title(f"3rd-Order Seidel Aberration Coefficients ({seidel_unit_lbl}) - System Total", fontsize=9, fontweight='bold')
                ax_seidel.set_ylabel(f"Aberration ({seidel_unit_lbl})", fontsize=8)
                ax_seidel.axhline(0, color=axis_col, linewidth=0.8)
            self.apply_dark_mode_to_axes(ax_fan, ax_seidel)
        else:
            ax_fan.text(0.5, 0.5, "No Optical System Loaded", horizontalalignment='center', color=self.get_theme_colors()['fg_subtext'])
            self.apply_dark_mode_to_axes(ax_fan, ax_seidel)

        self.apply_tight_layout()
        if hasattr(self.canvas, "draw_idle"):
            self.canvas.draw_idle()
        elif hasattr(self.canvas, "draw"):
            self.canvas.draw()


class SeidelAberrationTab(BaseAnalysisTab):
    """
    Dedicated Zemax-Style Seidel Aberration & Accumulation Analysis Tab.
    Displays per-lens and per-surface Seidel primary aberrations (S_I .. S_V),
    chromatic aberrations (C_L, C_T), progressive lens accumulation through the
    optical train, and final residual system accumulation.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.last_system = None
        self.last_target_surface_index = -1
        self.last_source_config = None
        self.last_system_config = None
        self.last_wavelengths = None
        self.last_seidel_data: Optional[SeidelAberrations] = None

        if QT_AVAILABLE:
            ctrl_layout = QHBoxLayout()
            ctrl_layout.setContentsMargins(4, 2, 4, 2)
            ctrl_layout.setSpacing(6)

            lbl_mode = QLabel("Analysis Mode:")
            lbl_mode.setStyleSheet("font-weight: bold; font-size: 11px;")
            ctrl_layout.addWidget(lbl_mode)

            self.combo_mode = QComboBox()
            self.combo_mode.addItem("Lens Accumulation (Zemax Style)", "lens_accumulation")
            self.combo_mode.addItem("Lens Contributions", "lens_contribution")
            self.combo_mode.addItem("Surface Accumulation", "surface_accumulation")
            self.combo_mode.addItem("Surface Contributions", "surface_contribution")
            self.combo_mode.addItem("Final Aberrations Summary", "final_summary")
            self.combo_mode.setStyleSheet(
                "QComboBox { background-color: #f1f5f9; border: 1px solid #cbd5e1; border-radius: 3px; "
                "padding: 2px 4px; font-size: 11px; font-weight: bold; color: #1e293b; }"
            )
            self.combo_mode.currentIndexChanged.connect(self._replot)
            ctrl_layout.addWidget(self.combo_mode)

            lbl_aberr = QLabel("Aberrations:")
            lbl_aberr.setStyleSheet("font-weight: bold; font-size: 11px;")
            ctrl_layout.addWidget(lbl_aberr)

            self.combo_aberration = QComboBox()
            self.combo_aberration.addItem("All (S_I .. S_V, C_L, C_T)", "all")
            self.combo_aberration.addItem("Monochromatic (S_I .. S_V)", "mono")
            self.combo_aberration.addItem("Chromatic (C_L, C_T)", "chromatic")
            self.combo_aberration.addItem("S_I (Spherical)", "S_I")
            self.combo_aberration.addItem("S_II (Coma)", "S_II")
            self.combo_aberration.addItem("S_III (Astigmatism)", "S_III")
            self.combo_aberration.addItem("S_IV (Petzval)", "S_IV")
            self.combo_aberration.addItem("S_V (Distortion)", "S_V")
            self.combo_aberration.addItem("C_L (Axial Color)", "C_L")
            self.combo_aberration.addItem("C_T (Lateral Color)", "C_T")
            self.combo_aberration.setStyleSheet(
                "QComboBox { background-color: #f1f5f9; border: 1px solid #cbd5e1; border-radius: 3px; "
                "padding: 2px 4px; font-size: 11px; font-weight: bold; color: #1e293b; }"
            )
            self.combo_aberration.currentIndexChanged.connect(self._replot)
            ctrl_layout.addWidget(self.combo_aberration)

            lbl_chart = QLabel("Style:")
            lbl_chart.setStyleSheet("font-weight: bold; font-size: 11px;")
            ctrl_layout.addWidget(lbl_chart)

            self.combo_chart = QComboBox()
            self.combo_chart.addItem("Grouped Bars", "bars")
            self.combo_chart.addItem("Cumulative Trend Lines", "lines")
            self.combo_chart.setStyleSheet(
                "QComboBox { background-color: #f1f5f9; border: 1px solid #cbd5e1; border-radius: 3px; "
                "padding: 2px 4px; font-size: 11px; font-weight: bold; color: #1e293b; }"
            )
            self.combo_chart.currentIndexChanged.connect(self._replot)
            ctrl_layout.addWidget(self.combo_chart)

            lbl_units = QLabel("Metric / Units:")
            lbl_units.setStyleSheet("font-weight: bold; font-size: 11px;")
            ctrl_layout.addWidget(lbl_units)

            self.combo_units = QComboBox()
            self.combo_units.addItem("Wavefront Aberrations (W040..W311) [µm]", "wavefront_um")
            self.combo_units.addItem("Wavefront Aberrations [Waves λ]", "waves")
            self.combo_units.addItem("Transverse Ray Blur [µm]", "transverse_um")
            self.combo_units.addItem("Seidel Sums [µm]", "um")
            self.combo_units.addItem("Seidel Sums [mm]", "mm")
            self.combo_units.setStyleSheet(
                "QComboBox { background-color: #f1f5f9; border: 1px solid #cbd5e1; border-radius: 3px; "
                "padding: 2px 4px; font-size: 11px; font-weight: bold; color: #1e293b; }"
            )
            self.combo_units.currentIndexChanged.connect(self._replot)
            ctrl_layout.addWidget(self.combo_units)

            ctrl_layout.addStretch(1)

            self.btn_copy_report = QPushButton("📋 Copy Zemax Report")
            self.btn_copy_report.setStyleSheet(
                "QPushButton { background-color: #2563eb; color: white; font-size: 11px; font-weight: bold; "
                "padding: 3px 8px; border-radius: 3px; border: none; } QPushButton:hover { background-color: #1d4ed8; }"
            )
            self.btn_copy_report.clicked.connect(self._on_copy_report)
            ctrl_layout.addWidget(self.btn_copy_report)

            self.btn_export_report = QPushButton("💾 Export Report")
            self.btn_export_report.setStyleSheet(
                "QPushButton { background-color: #0f766e; color: white; font-size: 11px; font-weight: bold; "
                "padding: 3px 8px; border-radius: 3px; border: none; } QPushButton:hover { background-color: #115e59; }"
            )
            self.btn_export_report.clicked.connect(self._on_export_report)
            ctrl_layout.addWidget(self.btn_export_report)

            self.status_label = QLabel("Load an optical system to calculate Seidel accumulation.")
            self.status_label.setWordWrap(True)
            self.status_label.setStyleSheet("font-size: 10px; color: #334155; padding: 2px 4px;")

            layout = self.layout()
            if layout is not None:
                idx = 1 if hasattr(self, 'toolbar') and self.toolbar else 0
                layout.insertLayout(idx, ctrl_layout)
                layout.addWidget(self.status_label)

    def _on_copy_report(self):
        if not self.last_seidel_data:
            return
        report = generate_zemax_seidel_report(self.last_seidel_data, mode="both")
        if QT_AVAILABLE:
            from optics_app.gui.qt_compat import QtWidgets
            app = QtWidgets.QApplication.instance()
            if app:
                clipboard = app.clipboard()
                if clipboard:
                    clipboard.setText(report)
                    if hasattr(self, 'status_label'):
                        self.status_label.setText("✅ Zemax Seidel Aberrations Report copied to clipboard!")

    def _on_export_report(self):
        if not self.last_seidel_data:
            return
        report = generate_zemax_seidel_report(self.last_seidel_data, mode="both")
        if QT_AVAILABLE:
            file_path, _ = QFileDialog.getSaveFileName(self, "Export Zemax Seidel Report", "zemax_seidel_aberrations.txt", "Text Files (*.txt);;All Files (*)")
            if file_path:
                try:
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(report)
                    if hasattr(self, 'status_label'):
                        self.status_label.setText(f"✅ Seidel report successfully saved to {file_path}")
                except Exception as e:
                    if hasattr(self, 'status_label'):
                        self.status_label.setText(f"❌ Failed to save report: {e}")

    def _replot(self, *_args):
        self.update_plot(
            system=self.last_system,
            target_surface_index=self.last_target_surface_index,
            source_config=self.last_source_config,
            system_config=self.last_system_config,
            wavelengths=self.last_wavelengths,
        )

    def update_plot(
        self,
        system: Optional[OpticalSystem] = None,
        target_surface_index: int = -1,
        source_config: Optional[Any] = None,
        system_config: Optional[Any] = None,
        wavelengths: Optional[List[float]] = None,
    ):
        if not (QT_AVAILABLE and HAS_MPL_QT) or self.fig is None:
            return

        self.last_system = system
        self.last_target_surface_index = target_surface_index
        self.last_source_config = source_config
        self.last_system_config = system_config
        self.last_wavelengths = wavelengths

        self.fig.clf()
        ax = self.fig.add_subplot(111)

        if system is None:
            ax.text(0.5, 0.5, "No Optical System Loaded", horizontalalignment='center', verticalalignment='center', fontsize=11)
            self.apply_tight_layout()
            if hasattr(self.canvas, "draw_idle"):
                self.canvas.draw_idle()
            return

        wl = 0.5875618
        if wavelengths and len(wavelengths) > 0:
            wl = wavelengths[0]
        elif system_config and getattr(system_config, 'primary_wavelength_um', None):
            wl = system_config.primary_wavelength_um

        try:
            seidel = calculate_seidel_coefficients(system, wavelength_um=wl)
            self.last_seidel_data = seidel
        except Exception as exc:
            ax.text(0.5, 0.5, f"Seidel Calculation Error:\n{exc}", horizontalalignment='center', verticalalignment='center', color='red', fontsize=10)
            self.apply_tight_layout()
            if hasattr(self.canvas, "draw_idle"):
                self.canvas.draw_idle()
            return

        mode = self.combo_mode.currentData() if hasattr(self, 'combo_mode') else "lens_accumulation"
        aberr_sel = self.combo_aberration.currentData() if hasattr(self, 'combo_aberration') else "all"
        chart_style = self.combo_chart.currentData() if hasattr(self, 'combo_chart') else "bars"
        unit_sel = self.combo_units.currentData() if hasattr(self, 'combo_units') else "wavefront_um"

        if unit_sel == "wavefront_um":
            unit_lbl = "µm (Wavefront OPD)"
        elif unit_sel == "waves":
            unit_lbl = "Waves λ"
        elif unit_sel == "transverse_um":
            unit_lbl = "µm (Ray Blur TA)"
        elif unit_sel == "um":
            unit_lbl = "µm (Seidel Sum)"
        else:
            unit_lbl = "mm (Seidel Sum)"

        def _get_metric_val(obj: Any, aberr_key: str, is_acc: bool = False) -> float:
            raw_attr = f"acc_{aberr_key}" if is_acc else aberr_key
            raw_mm = getattr(obj, raw_attr, 0.0)

            if unit_sel == "mm":
                return float(raw_mm)
            elif unit_sel == "um":
                return float(raw_mm * 1000.0)
            elif unit_sel == "wavefront_um":
                if aberr_key == "S_I":
                    return (raw_mm / 8.0) * 1000.0
                elif aberr_key in ("S_II", "S_III", "S_V"):
                    return (raw_mm / 2.0) * 1000.0
                elif aberr_key == "S_IV":
                    return (raw_mm / 4.0) * 1000.0
                else:  # C_L, C_T
                    return raw_mm * 1000.0
            elif unit_sel == "waves":
                wl_val = max(float(wl), 0.1)
                if aberr_key == "S_I":
                    return ((raw_mm / 8.0) * 1000.0) / wl_val
                elif aberr_key in ("S_II", "S_III", "S_V"):
                    return ((raw_mm / 2.0) * 1000.0) / wl_val
                elif aberr_key == "S_IV":
                    return ((raw_mm / 4.0) * 1000.0) / wl_val
                else:
                    return (raw_mm * 1000.0) / wl_val
            elif unit_sel == "transverse_um":
                img_u = getattr(seidel, "image_marginal_u", 0.1)
                img_n = getattr(seidel, "image_marginal_n", 1.0)
                denom = 2.0 * img_n * img_u
                if abs(denom) > 1e-9:
                    return (raw_mm / denom) * 1000.0
                return (raw_mm * 1000.0)
            return float(raw_mm * 1000.0)

        aberr_meta = [
            ("S_I", "S_I (Spherical)", "#1e40af"),
            ("S_II", "S_II (Coma)", "#0d9488"),
            ("S_III", "S_III (Astigmatism)", "#16a34a"),
            ("S_IV", "S_IV (Petzval)", "#d97706"),
            ("S_V", "S_V (Distortion)", "#dc2626"),
            ("C_L", "C_L (Axial Color)", "#7c3aed"),
            ("C_T", "C_T (Lateral Color)", "#db2777"),
        ]

        if aberr_sel == "mono":
            active_aberrs = aberr_meta[:5]
        elif aberr_sel == "chromatic":
            active_aberrs = aberr_meta[5:]
        elif aberr_sel in ["S_I", "S_II", "S_III", "S_IV", "S_V", "C_L", "C_T"]:
            active_aberrs = [m for m in aberr_meta if m[0] == aberr_sel]
        else:
            active_aberrs = aberr_meta

        axis_col = self.get_theme_colors()['axis_line_col']

        if mode == "final_summary":
            labels = [m[1] for m in active_aberrs]
            vals = [_get_metric_val(seidel, m[0], is_acc=False) for m in active_aberrs]
            colors = [m[2] for m in active_aberrs]
            bars = ax.bar(labels, vals, color=colors, alpha=0.85, edgecolor=axis_col, linewidth=1.0)
            ax.set_title(f"Zemax Final System Aberrations Accumulation (Detector Plane, {unit_lbl})", fontsize=11, fontweight='bold')
            ax.set_ylabel(f"Aberration Amplitude ({unit_lbl})", fontsize=10)
            ax.axhline(0, color=axis_col, linewidth=0.8)
            for bar, val in zip(bars, vals):
                max_v = max([abs(v) for v in vals], default=1.0)
                ypos = val + (0.02 * max_v if val >= 0 else -0.02 * max_v)
                ax.annotate(f"{val:+.3g} {unit_lbl}", (bar.get_x() + bar.get_width() / 2, ypos),
                            ha='center', va='bottom' if val >= 0 else 'top', fontsize=8, fontweight='bold', color=self.get_theme_colors()['fg_text'])
            ax.tick_params(axis='x', rotation=20)
            self.apply_dark_mode_to_axes(ax)

        elif mode in ("surface_accumulation", "surface_contribution"):
            surfs = [s for s in seidel.surface_details if s.surface_name not in ("Object", "Sample_Object") and s.surface_index > 0]
            cat_names = [f"S{s.surface_index}:{s.surface_name[:10]}" for s in surfs]
            if mode == "surface_accumulation":
                cat_names.append("FINAL TOTAL")
            n_cats = len(cat_names)
            x_indices = np.arange(n_cats)
            n_bars = len(active_aberrs)
            width = min(0.8 / max(n_bars, 1), 0.15)

            if chart_style == "lines":
                for key, label, col in active_aberrs:
                    if mode == "surface_accumulation":
                        y_vals = [_get_metric_val(s, key, is_acc=True) for s in surfs] + [_get_metric_val(seidel, key, is_acc=False)]
                    else:
                        y_vals = [_get_metric_val(s, key, is_acc=False) for s in surfs]
                    ax.plot(x_indices, y_vals, marker='o', linewidth=1.8, markersize=5, label=label, color=col)
            else:
                for b_i, (key, label, col) in enumerate(active_aberrs):
                    offset = (b_i - (n_bars - 1) / 2.0) * width
                    if mode == "surface_accumulation":
                        y_vals = [_get_metric_val(s, key, is_acc=True) for s in surfs] + [_get_metric_val(seidel, key, is_acc=False)]
                    else:
                        y_vals = [_get_metric_val(s, key, is_acc=False) for s in surfs]
                    bars = ax.bar(x_indices + offset, y_vals, width, label=label, color=col, alpha=0.85, edgecolor=axis_col, linewidth=0.6)
                    if mode == "surface_accumulation" and len(bars) > 0:
                        bars[-1].set_edgecolor('#d97706')
                        bars[-1].set_linewidth(1.8)

            title_str = "Surface Seidel Accumulation (Progressive)" if mode == "surface_accumulation" else "Individual Surface Seidel Contributions"
            ax.set_title(f"Zemax {title_str} ({unit_lbl})", fontsize=11, fontweight='bold')
            ax.set_xticks(x_indices)
            ax.set_xticklabels(cat_names, rotation=35, ha='right', fontsize=8)
            ax.set_ylabel(f"Aberration ({unit_lbl})", fontsize=10)
            ax.axhline(0, color=axis_col, linewidth=0.8)
            ax.legend(loc='best', fontsize=8)
            self.apply_dark_mode_to_axes(ax)

        else: # "lens_accumulation" (Default) or "lens_contribution"
            lenses = seidel.lens_details
            if not lenses:
                ax.text(0.5, 0.5, "No optical lens elements identified in system stack.", ha='center', va='center', color=self.get_theme_colors()['fg_subtext'])
            else:
                cat_names = [f"L{l.lens_index}: {l.lens_name}" for l in lenses]
                if mode == "lens_accumulation":
                    cat_names.append("FINAL ACCUMULATION")
                n_cats = len(cat_names)
                x_indices = np.arange(n_cats)
                n_bars = len(active_aberrs)
                width = min(0.8 / max(n_bars, 1), 0.16)

                if chart_style == "lines":
                    for key, label, col in active_aberrs:
                        if mode == "lens_accumulation":
                            y_vals = [_get_metric_val(l, key, is_acc=True) for l in lenses] + [_get_metric_val(seidel, key, is_acc=False)]
                        else:
                            y_vals = [_get_metric_val(l, key, is_acc=False) for l in lenses]
                        ax.plot(x_indices, y_vals, marker='o', linewidth=2.0, markersize=6, label=label, color=col)
                else:
                    for b_i, (key, label, col) in enumerate(active_aberrs):
                        offset = (b_i - (n_bars - 1) / 2.0) * width
                        if mode == "lens_accumulation":
                            y_vals = [_get_metric_val(l, key, is_acc=True) for l in lenses] + [_get_metric_val(seidel, key, is_acc=False)]
                        else:
                            y_vals = [_get_metric_val(l, key, is_acc=False) for l in lenses]
                        bars = ax.bar(x_indices + offset, y_vals, width, label=label, color=col, alpha=0.85, edgecolor=axis_col, linewidth=0.6)
                        if mode == "lens_accumulation" and len(bars) > 0:
                            bars[-1].set_edgecolor('#d97706') # Highlight final accumulation
                            bars[-1].set_linewidth(2.0)

                title_str = "Seidel Accumulation for Every Lens & Final Aberrations Accumulation" if mode == "lens_accumulation" else "Individual Lens Aberration Contributions"
                ax.set_title(f"Zemax {title_str} ({unit_lbl})", fontsize=11, fontweight='bold')
                ax.set_xticks(x_indices)
                ax.set_xticklabels(cat_names, rotation=25, ha='right', fontsize=8.5, fontweight='bold')
                ax.set_ylabel(f"Aberration Amplitude ({unit_lbl})", fontsize=10)
                ax.axhline(0, color=axis_col, linewidth=0.8)
                ax.legend(loc='best', fontsize=8)
            self.apply_dark_mode_to_axes(ax)

        if hasattr(self, 'status_label') and seidel:
            dom_lens = max(seidel.lens_details, key=lambda l: abs(l.S_I)).lens_name if seidel.lens_details else "N/A"
            status_txt = (
                f"Lenses Analyzed: {len(seidel.lens_details)} | "
                f"Wavefront OPD (µm): W040={seidel.W_040_um:+.2f}, W131={seidel.W_131_um:+.2f}, W222={seidel.W_222_um:+.2f} | "
                f"Transverse Blur TA: {seidel.TA_sph_um:+.1f} µm | "
                f"Total Seidel Sum: S_I={seidel.S_I * 1000.0:+.1f} µm ({seidel.S_I:+.4f} mm) | "
                f"Dominant Spherical: {dom_lens}"
            )
            self.status_label.setText(status_txt)

        self.apply_tight_layout()
        if hasattr(self.canvas, "draw_idle"):
            self.canvas.draw_idle()
        elif hasattr(self.canvas, "draw"):
            self.canvas.draw()


class GratingDispersionTab(BaseAnalysisTab):
    """Tab 3: Grating Dispersion Viewer."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.last_system = None
        self.last_target_surface_index = -1
        self.last_source_config = None

    def _replot(self):
        self.update_plot(
            system=self.last_system,
            target_surface_index=self.last_target_surface_index,
            source_config=self.last_source_config,
        )

    def update_plot(self, system: Optional[OpticalSystem] = None, target_surface_index: int = -1, source_config: Optional[Any] = None):
        if not (QT_AVAILABLE and HAS_MPL_QT) or self.fig is None:
            return

        self.last_system = system
        self.last_target_surface_index = target_surface_index
        self.last_source_config = source_config

        self.fig.clf()
        ax_disp = self.fig.add_subplot(211)
        ax_lin = self.fig.add_subplot(212)

        if system is not None:
            wls = system.wavelengths if hasattr(system, 'wavelengths') else [0.486, 0.588, 0.656]
            res = GratingDispersionAnalysis.analyze_system_dispersion(
                system, 
                wavelengths_um=wls,
                num_rays_per_wavelength=25,
                target_surface_index=target_surface_index,
                source_config=source_config
            )

            dx_col = '#f87171' if self.dark_mode else '#dc2626'
            dy_col = '#38bdf8' if self.dark_mode else '#0284c7'
            dlin_col = '#4ade80' if self.dark_mode else '#16a34a'

            ax_disp.plot(res.wavelengths_um, res.delta_x_array, color=dx_col, marker='o', linestyle='-', linewidth=1.5, label="delta_x (mm)")
            ax_disp.plot(res.wavelengths_um, res.delta_y_array, color=dy_col, marker='^', linestyle='-', linewidth=1.2, label="delta_y (mm)")
            ax_disp.set_title("Detector Wavelength Separation Spread", fontsize=10, fontweight='bold')
            ax_disp.set_xlabel("Wavelength (µm)")
            ax_disp.set_ylabel("Spatial Offset (mm)")
            ax_disp.legend(loc='upper left', fontsize=8)

            ax_lin.plot(res.wavelengths_um, res.linear_dispersion_curve, color=dlin_col, marker='o', linestyle='-', linewidth=1.5, label="dx/dlambda (mm/µm)")
            ax_lin.set_title("Linear Dispersion vs Wavelength", fontsize=9, fontweight='bold')
            ax_lin.set_xlabel("Wavelength (µm)")
            ax_lin.set_ylabel("dx/dlambda (mm/µm)")
            ax_lin.legend(loc='upper right', fontsize=8)
            self.apply_dark_mode_to_axes(ax_disp, ax_lin)
        else:
            ax_disp.text(0.5, 0.5, "No Optical System Loaded", horizontalalignment='center', color=self.get_theme_colors()['fg_subtext'])
            self.apply_dark_mode_to_axes(ax_disp, ax_lin)

        self.apply_tight_layout()
        if hasattr(self.canvas, "draw_idle"):
            self.canvas.draw_idle()
        elif hasattr(self.canvas, "draw"):
            self.canvas.draw()


class CCDHeatmapTab(BaseAnalysisTab):
    """Tab 4: 2D CCD Image Heatmap Viewer."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.last_system = None
        self.last_ccd_sensor = None
        self.last_target_surface_index = -1
        self.last_source_config = None

    def _replot(self):
        self.update_plot(
            system=self.last_system,
            ccd_sensor=self.last_ccd_sensor,
            target_surface_index=self.last_target_surface_index,
            source_config=self.last_source_config,
        )

    def update_plot(self, system: Optional[OpticalSystem] = None, ccd_sensor: Optional[CCDSensor] = None, target_surface_index: int = -1, source_config: Optional[Any] = None):
        if not (QT_AVAILABLE and HAS_MPL_QT) or self.fig is None:
            return

        self.last_system = system
        self.last_ccd_sensor = ccd_sensor
        self.last_target_surface_index = target_surface_index
        self.last_source_config = source_config

        self.fig.clf()
        ax_map = self.fig.add_subplot(121)
        ax_prof = self.fig.add_subplot(122)

        if ccd_sensor is None and system is not None:
            ccd_sensor = CCDSensor(width_mm=10.0, height_mm=10.0, pixel_cols=60, pixel_rows=60)
            spot = generate_spot_diagram_from_system(system, num_rays=300)
            if len(spot.x) > 0:
                ccd_sensor.deposit_rays(spot.x, spot.y)
            self.last_ccd_sensor = ccd_sensor

        if ccd_sensor is not None and ccd_sensor.total_rays_deposited > 0:
            matrix = ccd_sensor.get_heatmap()
            im = ax_map.imshow(matrix, origin='lower', cmap='inferno', extent=[-ccd_sensor.width_mm/2, ccd_sensor.width_mm/2, -ccd_sensor.height_mm/2, ccd_sensor.height_mm/2])
            cbar = self.fig.colorbar(im, ax=ax_map, label="Intensity (Rays)")
            style_colorbar_for_theme(cbar, self.dark_mode)
            ax_map.set_title("2D CCD Intensity Matrix Heatmap", fontsize=10, fontweight='bold')
            ax_map.set_xlabel("X (mm)")
            ax_map.set_ylabel("Y (mm)")

            mid_row = ccd_sensor.pixel_rows // 2
            mid_col = ccd_sensor.pixel_cols // 2
            x_axis = np.linspace(-ccd_sensor.width_mm/2, ccd_sensor.width_mm/2, ccd_sensor.pixel_cols)
            y_axis = np.linspace(-ccd_sensor.height_mm/2, ccd_sensor.height_mm/2, ccd_sensor.pixel_rows)

            x_col = '#f87171' if self.dark_mode else '#dc2626'
            y_col = '#38bdf8' if self.dark_mode else '#0284c7'

            ax_prof.plot(x_axis, matrix[mid_row, :], color=x_col, linestyle='-', label="X Profile (y=0)")
            ax_prof.plot(y_axis, matrix[:, mid_col], color=y_col, linestyle='--', label="Y Profile (x=0)")
            ax_prof.set_title("1D Spatial Intensity Slices", fontsize=9, fontweight='bold')
            ax_prof.set_xlabel("Position (mm)")
            ax_prof.set_ylabel("Intensity")
            ax_prof.legend(loc='upper right', fontsize=8)
            self.apply_dark_mode_to_axes(ax_map, ax_prof)
        else:
            from optics_app.engine.mechanisms import is_dmd, active_state_name
            dmd_off = False
            if system and getattr(system, 'surface_stack', None):
                for s in system.surface_stack.surfaces:
                    if is_dmd(s) and active_state_name(s) == "OFF":
                        dmd_off = True
                        break
            if dmd_off:
                bg_dmd = '#450a0a' if self.dark_mode else '#fee2e2'
                border_dmd = '#ef4444' if self.dark_mode else '#fca5a5'
                text_dmd = '#fca5a5' if self.dark_mode else '#991b1b'
                ax_map.text(0.5, 0.5, "🔲 DMD State: OFF\nDetector Dark (0 hits)",
                            horizontalalignment='center', verticalalignment='center',
                            fontsize=10, fontweight='bold', color=text_dmd,
                            bbox=dict(boxstyle='round,pad=0.5', facecolor=bg_dmd, edgecolor=border_dmd))
                ax_prof.text(0.5, 0.5, "Sensor Signal: 0.00\n(Beam Dump Active)",
                            horizontalalignment='center', verticalalignment='center',
                            fontsize=10, fontweight='bold', color=text_dmd)
            else:
                ax_map.text(0.5, 0.5, "No CCD Sensor Rays Deposited", horizontalalignment='center', color=self.get_theme_colors()['fg_subtext'])
            self.apply_dark_mode_to_axes(ax_map, ax_prof)

        self.apply_tight_layout()
        if hasattr(self.canvas, "draw_idle"):
            self.canvas.draw_idle()
        elif hasattr(self.canvas, "draw"):
            self.canvas.draw()


class LaserScanAberrationTab(BaseAnalysisTab):
    """
    Tab 5: Laser Scanning System & Field Aberrations Viewer.
    Simulates galvanometer mirror deflection and computes field-dependent aberrations
    (RMS/GEO spot sizes, Airy limit, tangential/sagittal field curvature, astigmatism,
    and f*theta scan linearity distortion).
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.last_system: Optional[OpticalSystem] = None
        self.last_target_surface_index: int = -1
        self.last_wavelengths: Optional[List[float]] = None
        self.last_result: Any = None

        if QT_AVAILABLE:
            ctrl_layout = QHBoxLayout()
            ctrl_layout.setContentsMargins(4, 2, 4, 2)
            ctrl_layout.addWidget(QLabel("Axis:"))
            self.axis_combo = QComboBox()
            self.axis_combo.addItems(["X", "Y", "XY (2D Raster)"])
            self.axis_combo.currentIndexChanged.connect(self._on_ctrl_changed)
            ctrl_layout.addWidget(self.axis_combo)

            ctrl_layout.addWidget(QLabel("Range (±°):"))
            self.range_spin = QDoubleSpinBox()
            self.range_spin.setRange(0.5, 45.0)
            self.range_spin.setValue(8.0)
            self.range_spin.setSingleStep(1.0)
            self.range_spin.setSuffix("°")
            self.range_spin.valueChanged.connect(self._on_ctrl_changed)
            ctrl_layout.addWidget(self.range_spin)

            ctrl_layout.addWidget(QLabel("Step (°):"))
            self.step_spin = QDoubleSpinBox()
            self.step_spin.setRange(0.1, 10.0)
            self.step_spin.setValue(1.0)
            self.step_spin.setSingleStep(0.5)
            self.step_spin.setSuffix("°")
            self.step_spin.valueChanged.connect(self._on_ctrl_changed)
            ctrl_layout.addWidget(self.step_spin)

            ctrl_layout.addWidget(QLabel("Beam Ø:"))
            self.beam_diam_spin = QDoubleSpinBox()
            self.beam_diam_spin.setRange(0.1, 50.0)
            self.beam_diam_spin.setValue(5.0)
            self.beam_diam_spin.setSingleStep(1.0)
            self.beam_diam_spin.setSuffix(" mm")
            self.beam_diam_spin.valueChanged.connect(self._on_ctrl_changed)
            ctrl_layout.addWidget(self.beam_diam_spin)

            self.btn_run_scan = QPushButton("🚀 Simulate Scan")
            self.btn_run_scan.clicked.connect(self._on_run_clicked)
            ctrl_layout.addWidget(self.btn_run_scan)

            self.status_label = QLabel("Ready for laser scanning simulation.")
            self.status_label.setWordWrap(True)
            self.status_label.setStyleSheet("color: #00796B; font-weight: bold; padding: 2px;")

            if hasattr(self, "layout") and self.layout() is not None:
                self.layout().insertLayout(0, ctrl_layout)
                self.layout().addWidget(self.status_label)

    def _replot(self, *_args):
        if self.last_system is not None:
            self.update_plot(
                system=self.last_system,
                target_surface_index=self.last_target_surface_index,
                wavelengths=self.last_wavelengths,
            )

    def _on_ctrl_changed(self, *args):
        self._replot()

    def _on_run_clicked(self):
        if self.last_system is not None:
            self.update_plot(
                system=self.last_system,
                target_surface_index=self.last_target_surface_index,
                wavelengths=self.last_wavelengths,
                force_run=True,
            )

    def update_plot(
        self,
        system: Optional[OpticalSystem] = None,
        wavelengths: Optional[List[float]] = None,
        target_surface_index: int = -1,
        source_config: Optional[Any] = None,
        force_run: bool = False,
    ):
        if not (QT_AVAILABLE and HAS_MPL_QT) or self.fig is None:
            return

        theme = self.get_theme_colors()

        if system is not None:
            self.last_system = system
        else:
            system = self.last_system

        if wavelengths is not None:
            self.last_wavelengths = wavelengths
        elif self.last_wavelengths is not None:
            wavelengths = self.last_wavelengths

        self.last_target_surface_index = target_surface_index

        self.fig.clf()
        if system is None or not getattr(system, "stack", None) or len(system.stack.surfaces) < 2:
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, "Requires an optical system with at least 2 surfaces (scanner + optics).",
                    horizontalalignment='center', verticalalignment='center', fontsize=11, color=theme["text_dim"])
            self.apply_dark_mode_to_axes(ax)
            self.canvas.draw()
            return

        try:
            from optics_app.engine.laser_scanner import LaserScannerSimulator, LaserBeamConfig

            axis_text = self.axis_combo.currentText() if hasattr(self, "axis_combo") else "X"
            max_ang = float(self.range_spin.value()) if hasattr(self, "range_spin") else 8.0
            step = float(self.step_spin.value()) if hasattr(self, "step_spin") else 1.0
            beam_diam = float(self.beam_diam_spin.value()) if hasattr(self, "beam_diam_spin") else 5.0
            wl = float(wavelengths[0]) if (wavelengths and len(wavelengths) > 0) else 0.532

            beam_cfg = LaserBeamConfig(wavelength_um=wl, beam_diameter_mm=beam_diam, num_pupil_rings=2, rays_per_ring=4)
            simulator = LaserScannerSimulator(system.stack)

            if "XY" in axis_text:
                result = simulator.simulate_2d_raster_scan(
                    angle_range_x=(-max_ang, max_ang),
                    angle_range_y=(-max_ang, max_ang),
                    step_size_deg=max(1.0, step * 2.0),
                    beam_config=beam_cfg,
                    target_surface_idx=target_surface_index,
                )
            else:
                result = simulator.simulate_1d_scan(
                    scan_axis="Y" if "Y" in axis_text else "X",
                    min_angle_deg=-max_ang,
                    max_angle_deg=max_ang,
                    step_size_deg=step,
                    beam_config=beam_cfg,
                    target_surface_idx=target_surface_index,
                )

            self.last_result = result

            ax1 = self.fig.add_subplot(221)
            ax2 = self.fig.add_subplot(222)
            ax3 = self.fig.add_subplot(223)
            ax4 = self.fig.add_subplot(224)

            # Subplot 1: Spot Radius vs Scan Angle
            ax1.plot(result.angles_deg, result.rms_spots_um, 'o-', color='#38bdf8' if self.dark_mode else '#1976D2', linewidth=2, label='RMS Spot (µm)')
            if len(result.airy_radii_um) > 0:
                ax1.plot(result.angles_deg, result.airy_radii_um, '--', color='#f87171' if self.dark_mode else 'red', linewidth=1.5,
                         label=f'Airy Limit ({np.mean(result.airy_radii_um):.1f} µm)')
            ax1.set_title("Spot Size vs. Scan Angle", fontsize=9, fontweight='bold')
            ax1.set_xlabel("Mechanical Angle θ (deg)", fontsize=8)
            ax1.set_ylabel("Radius (µm)", fontsize=8)
            ax1.grid(True, linestyle='--', alpha=0.5)
            ax1.legend(loc='best', fontsize=7)

            # Subplot 2: Field Curvature & Astigmatism
            ax2.plot(result.angles_deg, result.tangential_foci_mm, '-', color='#f472b6' if self.dark_mode else 'm', linewidth=1.8, label='Tangential (T)')
            ax2.plot(result.angles_deg, result.sagittal_foci_mm, '-', color='#22d3ee' if self.dark_mode else 'c', linewidth=1.8, label='Sagittal (S)')
            ax2.plot(result.angles_deg, result.field_curvatures_mm, '--', color='#94a3b8' if self.dark_mode else '#334155', linewidth=1.2, label='Mean Petzval')
            ax2.set_title("Field Curvature & Astigmatism (Δz)", fontsize=9, fontweight='bold')
            ax2.set_xlabel("Mechanical Angle θ (deg)", fontsize=8)
            ax2.set_ylabel("Focus Shift Δz (mm)", fontsize=8)
            ax2.grid(True, linestyle='--', alpha=0.5)
            ax2.legend(loc='best', fontsize=7)

            # Subplot 3: f*theta Distortion Error (%)
            ax3.plot(result.angles_deg, result.f_theta_errors_percent, 's-', color='#4ade80' if self.dark_mode else '#2E7D32', linewidth=2, label='f·θ Error (%)')
            ax3.axhline(0.0, color='#94a3b8' if self.dark_mode else '#475569', linestyle=':', alpha=0.7)
            ax3.set_title("f·θ Scan Linearity Distortion Error", fontsize=9, fontweight='bold')
            ax3.set_xlabel("Mechanical Angle θ (deg)", fontsize=8)
            ax3.set_ylabel("Linearity Error (%)", fontsize=8)
            ax3.grid(True, linestyle='--', alpha=0.5)
            ax3.legend(loc='best', fontsize=7)

            # Subplot 4: 2D Spot Footprint at Detector
            xs = [p.chief_ray_x_mm for p in result.scan_points]
            ys = [p.chief_ray_y_mm for p in result.scan_points]
            spots = [max(12, min(120, p.rms_spot_radius_um * 2.0)) for p in result.scan_points]
            sc = ax4.scatter(xs, ys, s=spots, c=result.rms_spots_um, cmap='plasma', edgecolors='#94a3b8' if self.dark_mode else 'black', linewidth=0.5)
            ax4.axhline(0, color=theme["axis_line"], linestyle=':', alpha=0.4)
            ax4.axvline(0, color=theme["axis_line"], linestyle=':', alpha=0.4)
            ax4.set_title("Scan Field Footprint at Detector", fontsize=9, fontweight='bold')
            ax4.set_xlabel("Image X (mm)", fontsize=8)
            ax4.set_ylabel("Image Y (mm)", fontsize=8)
            ax4.grid(True, linestyle='--', alpha=0.5)
            cbar = self.fig.colorbar(sc, ax=ax4, label='RMS (µm)', fraction=0.046, pad=0.04)
            style_colorbar_for_theme(cbar, self.dark_mode)

            for sub_ax in (ax1, ax2, ax3, ax4):
                self.apply_dark_mode_to_axes(sub_ax)

            # Update status
            if hasattr(self, "status_label"):
                status_str = (
                    f"Scan EFL: {result.scan_lens_efl_mm:.1f} mm | "
                    f"Field Dia: {result.scan_field_diameter_mm:.1f} mm | "
                    f"Mean RMS: {result.mean_rms_spot_um:.1f} µm | "
                    f"Max Astig: {result.max_astigmatism_mm:.3f} mm | "
                    f"Max f·θ Err: {result.max_f_theta_error_percent:.2f}% | "
                    f"Diff-Limited: {result.diffraction_limited_fraction*100:.0f}%"
                )
                self.status_label.setText(status_str)
                self.status_label.setStyleSheet(
                    f"color: {'#38bdf8' if self.dark_mode else '#00796B'}; font-weight: bold; padding: 2px;"
                )

        except Exception as exc:
            self.fig.clf()
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, f"Laser Scan Simulation Error:\n{str(exc)}",
                    horizontalalignment='center', verticalalignment='center', color='#f87171' if self.dark_mode else 'red', fontsize=10)
            self.apply_dark_mode_to_axes(ax)

        self.apply_tight_layout()
        if hasattr(self.canvas, "draw_idle"):
            self.canvas.draw_idle()
        elif hasattr(self.canvas, "draw"):
            self.canvas.draw()


class PulseWidthAnalysisTab(BaseAnalysisTab):
    """Virtual femtosecond pulse-width probes along the optical path."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.config = PulseAnalysisConfig()
        self.last_system: Optional[OpticalSystem] = None
        self.last_result = None
        self._syncing = False
        self._center_initialized = False

        if QT_AVAILABLE:
            row1 = QHBoxLayout()
            row1.setContentsMargins(4, 2, 4, 2)
            row1.addWidget(QLabel("Center λ:"))
            self.center_wl_spin = QDoubleSpinBox()
            self.center_wl_spin.setRange(200.0, 25000.0)
            self.center_wl_spin.setDecimals(2)
            self.center_wl_spin.setValue(self.config.center_wavelength_nm)
            self.center_wl_spin.setSuffix(" nm")
            row1.addWidget(self.center_wl_spin)

            row1.addWidget(QLabel("TL pulse FWHM:"))
            self.pulse_width_spin = QDoubleSpinBox()
            self.pulse_width_spin.setRange(1.0, 100000.0)
            self.pulse_width_spin.setDecimals(2)
            self.pulse_width_spin.setValue(self.config.transform_limit_fwhm_fs)
            self.pulse_width_spin.setSuffix(" fs")
            row1.addWidget(self.pulse_width_spin)

            row1.addWidget(QLabel("Input chirp (GDD):"))
            self.input_gdd_spin = QDoubleSpinBox()
            self.input_gdd_spin.setRange(-1.0e7, 1.0e7)
            self.input_gdd_spin.setDecimals(1)
            self.input_gdd_spin.setSingleStep(100.0)
            self.input_gdd_spin.setValue(self.config.initial_gdd_fs2)
            self.input_gdd_spin.setSuffix(" fs²")
            row1.addWidget(self.input_gdd_spin)

            row1.addWidget(QLabel("Spectral samples:"))
            self.samples_spin = QSpinBox()
            self.samples_spin.setRange(7, 51)
            self.samples_spin.setSingleStep(2)
            self.samples_spin.setValue(self.config.spectral_samples)
            row1.addWidget(self.samples_spin)

            self.btn_analyze_pulse = QPushButton("Analyze Pulse")
            self.btn_analyze_pulse.clicked.connect(self._on_analyze_clicked)
            row1.addWidget(self.btn_analyze_pulse)

            row2 = QHBoxLayout()
            row2.setContentsMargins(4, 2, 4, 2)
            row2.addWidget(QLabel("Virtual probe:"))
            self.probe_combo = QComboBox()
            self.probe_combo.currentIndexChanged.connect(self._on_probe_selected)
            row2.addWidget(self.probe_combo)
            self.btn_add_probe = QPushButton("+ Add Probe")
            self.btn_add_probe.setToolTip("Mark this surface as a persistent pulse-width measurement probe.")
            self.btn_add_probe.clicked.connect(self._on_add_probe)
            row2.addWidget(self.btn_add_probe)
            self.btn_remove_probe = QPushButton("− Remove Probe")
            self.btn_remove_probe.clicked.connect(self._on_remove_probe)
            row2.addWidget(self.btn_remove_probe)

            self.pulse_status = QLabel("Set the source pulse and choose Analyze Pulse.")
            self.pulse_status.setWordWrap(True)
            self.pulse_status.setStyleSheet("color: #00695C; font-weight: bold; padding: 3px;")

            if hasattr(self, "layout") and self.layout() is not None:
                self.layout().insertLayout(0, row1)
                self.layout().insertLayout(1, row2)
                self.layout().addWidget(self.pulse_status)

    def get_config(self) -> Dict[str, Any]:
        self._read_controls()
        return self.config.to_dict()

    def set_config(self, data: Optional[Dict[str, Any]]) -> None:
        self.config = PulseAnalysisConfig.from_dict(data)
        # Legacy/non-ultrafast projects have no saved pulse carrier. Adopt the
        # project's primary wavelength on the next analysis refresh instead of
        # silently using the 800 nm example value.
        self._center_initialized = bool(data and "center_wavelength_nm" in data)
        if not QT_AVAILABLE:
            return
        self._syncing = True
        try:
            self.center_wl_spin.setValue(self.config.center_wavelength_nm)
            self.pulse_width_spin.setValue(self.config.transform_limit_fwhm_fs)
            self.input_gdd_spin.setValue(self.config.initial_gdd_fs2)
            self.samples_spin.setValue(self.config.spectral_samples)
        finally:
            self._syncing = False
        if self.last_system is not None:
            self.update_plot(self.last_system, force_run=True)

    def _read_controls(self) -> None:
        if not QT_AVAILABLE:
            return
        self.config.center_wavelength_nm = float(self.center_wl_spin.value())
        self.config.transform_limit_fwhm_fs = float(self.pulse_width_spin.value())
        self.config.initial_gdd_fs2 = float(self.input_gdd_spin.value())
        samples = int(self.samples_spin.value())
        self.config.spectral_samples = samples if samples % 2 else samples + 1

    def _populate_probe_combo(self, system: Optional[OpticalSystem]) -> None:
        if not QT_AVAILABLE or not hasattr(self, "probe_combo"):
            return
        current = self.probe_combo.currentData()
        self._syncing = True
        try:
            self.probe_combo.clear()
            for idx, surf in enumerate(getattr(system, "surfaces", []) if system else []):
                marker = "● " if idx in self.config.probe_indices else ""
                self.probe_combo.addItem(f"{marker}[{idx}] {getattr(surf, 'name', f'Surface {idx}')}", idx)
            found = self.probe_combo.findData(current)
            if found < 0 and self.config.probe_indices:
                found = self.probe_combo.findData(self.config.probe_indices[-1])
            self.probe_combo.setCurrentIndex(max(0, found))
        finally:
            self._syncing = False

    def _on_analyze_clicked(self):
        self._read_controls()
        self.update_plot(self.last_system, force_run=True)

    def _on_add_probe(self):
        idx = self.probe_combo.currentData() if hasattr(self, "probe_combo") else None
        if idx is not None and int(idx) not in self.config.probe_indices:
            self.config.probe_indices.append(int(idx))
        self._populate_probe_combo(self.last_system)
        self._update_probe_readout()
        self._draw_result()

    def _on_remove_probe(self):
        idx = self.probe_combo.currentData() if hasattr(self, "probe_combo") else None
        if idx is not None and int(idx) in self.config.probe_indices:
            self.config.probe_indices.remove(int(idx))
        self._populate_probe_combo(self.last_system)
        self._update_probe_readout()
        self._draw_result()

    def _on_probe_selected(self, *args):
        if not self._syncing:
            self._update_probe_readout()

    def _update_probe_readout(self) -> None:
        if not QT_AVAILABLE or not hasattr(self, "pulse_status"):
            return
        idx = self.probe_combo.currentData() if hasattr(self, "probe_combo") else None
        probe = self.last_result.probe(int(idx)) if self.last_result is not None and idx is not None else None
        if probe is None or not probe.valid:
            self.pulse_status.setText("Probe has no valid pulse result. Check spectral-ray transmission to this surface.")
            return
        compression = 100.0 * (probe.pulse_fwhm_fs / self.config.transform_limit_fwhm_fs - 1.0)
        trend = "broadened" if compression >= 0 else "compressed"
        self.pulse_status.setText(
            f"Probe #{probe.surface_index} {probe.surface_name}: {probe.pulse_fwhm_fs:.2f} fs ({trend} {abs(compression):.2f}%) | "
            f"system GDD {probe.system_gdd_fs2:+.1f} fs² | total GDD {probe.total_gdd_fs2:+.1f} fs² | "
            f"TOD {probe.system_tod_fs3:+.1f} fs³ | path {probe.path_length_mm:.2f} mm | "
            f"spectral rays {probe.reached_samples}/{probe.total_samples} | spatial chirp {probe.spatial_chirp_mm*1000.0:.1f} µm"
            + (" ⚠ spectrum is not locally overlapped; duration is coherent-recombination equivalent."
               if not probe.locally_overlapped else "")
        )

    def update_plot(
        self,
        system: Optional[OpticalSystem] = None,
        wavelengths: Optional[List[float]] = None,
        source_config: Optional[Any] = None,
        force_run: bool = False,
        **kwargs,
    ):
        if system is not None:
            self.last_system = system
        else:
            system = self.last_system
        if not self._center_initialized and wavelengths:
            self.config.center_wavelength_nm = float(np.median(np.asarray(wavelengths, dtype=float))) * 1000.0
            if QT_AVAILABLE:
                self.center_wl_spin.setValue(self.config.center_wavelength_nm)
            self._center_initialized = True
        self._populate_probe_combo(system)
        if system is None:
            self.last_result = None
            self._draw_result()
            return
        self._read_controls()
        try:
            self.last_result = PulseWidthAnalysis.analyze(system, self.config)
        except Exception as exc:
            self.last_result = None
            if QT_AVAILABLE:
                self.pulse_status.setText(f"Pulse analysis error: {exc}")
        self._update_probe_readout()
        self._draw_result()

    def _replot(self, *_args):
        self._draw_result()

    def _draw_result(self) -> None:
        if not (QT_AVAILABLE and HAS_MPL_QT) or self.fig is None:
            return
        self.fig.clf()
        theme = self.get_theme_colors()
        ax_width = self.fig.add_subplot(211)
        ax_gdd = self.fig.add_subplot(212)
        valid = [p for p in getattr(self.last_result, "probes", []) if p.valid]
        if not valid:
            ax_width.text(0.5, 0.5, "No valid femtosecond pulse path", ha="center", va="center", color=theme["text_dim"])
        else:
            x = np.asarray([p.path_length_mm for p in valid], dtype=float)
            widths = np.asarray([p.pulse_fwhm_fs for p in valid], dtype=float)
            total_gdd = np.asarray([p.total_gdd_fs2 for p in valid], dtype=float)
            material = np.asarray([p.material_gdd_fs2 for p in valid], dtype=float)
            geometric = np.asarray([p.geometric_gdd_fs2 for p in valid], dtype=float)
            ax_width.plot(x, widths, "o-", color="#c084fc" if self.dark_mode else "#6A1B9A", linewidth=1.8, markersize=4, label="Pulse FWHM")
            ax_width.axhline(self.config.transform_limit_fwhm_fs, color="#2dd4bf" if self.dark_mode else "#00897B", linestyle="--", label="Transform limit")
            ax_width.set_title("Femtosecond Pulse Width Along Optical Path", fontsize=10, fontweight="bold")
            ax_width.set_ylabel("Intensity FWHM (fs)")
            ax_width.grid(True, linestyle="--", alpha=0.4)
            ax_width.legend(fontsize=8)

            total_col = "#f1f5f9" if self.dark_mode else "#0f172a"
            ax_gdd.plot(x, total_gdd, color=total_col, linewidth=1.8, label="Total incl. input chirp")
            ax_gdd.plot(x, material, "--", color="#38bdf8" if self.dark_mode else "b", linewidth=1.3, label="Material GDD")
            ax_gdd.plot(x, geometric, "-.", color="#f87171" if self.dark_mode else "r", linewidth=1.3, label="Geometric GDD")
            ax_gdd.axhline(0.0, color="#94a3b8" if self.dark_mode else "#777777", linewidth=0.7)
            ax_gdd.set_xlabel("Accumulated chief-ray path (mm)")
            ax_gdd.set_ylabel("GDD (fs²)")
            ax_gdd.grid(True, linestyle="--", alpha=0.4)
            ax_gdd.legend(fontsize=8, ncol=3)

            for probe_idx in self.config.probe_indices:
                probe = self.last_result.probe(probe_idx)
                if probe is None or not probe.valid:
                    continue
                for ax in (ax_width, ax_gdd):
                    ax.axvline(probe.path_length_mm, color="#FF8F00", linewidth=1.0, alpha=0.8)
                ax_width.annotate(
                    f"P{probe_idx}\n{probe.pulse_fwhm_fs:.1f} fs",
                    (probe.path_length_mm, probe.pulse_fwhm_fs),
                    xytext=(4, 5), textcoords="offset points", fontsize=7, color="#fb923c" if self.dark_mode else "#E65100",
                )

            bw = self.config.gaussian_bandwidth_nm
            bbox_callout = self.get_text_bbox(pad=0.3, alpha=0.9)
            ax_width.text(
                0.99, 0.96,
                f"λ₀={self.config.center_wavelength_nm:.1f} nm | TL={self.config.transform_limit_fwhm_fs:.1f} fs | Δλ≈{bw:.2f} nm",
                transform=ax_width.transAxes, ha="right", va="top", fontsize=8,
                color=theme["text"],
                bbox=bbox_callout,
            )

        self.apply_dark_mode_to_axes(ax_width)
        self.apply_dark_mode_to_axes(ax_gdd)
        self.apply_tight_layout()
        if hasattr(self.canvas, "draw_idle"):
            self.canvas.draw_idle()
        elif hasattr(self.canvas, "draw"):
            self.canvas.draw()


class TwoPhotonMicroscopeTab(BaseAnalysisTab):
    """Calibrated two-photon point-emitter workbench."""

    def __init__(self, model: Optional[Any] = None, parent=None):
        super().__init__(parent)
        self.model = model
        self.config = build_calibrated_template()
        self._selected_fluor_id = self.config.fluorophore.identifier
        self._selected_objective_id = self.config.objective.part_number
        self.last_result = None
        self.last_stack = None
        if QT_AVAILABLE:
            row1 = QHBoxLayout()
            row1.addWidget(QLabel("Fluorophore:"))
            self.fluor_combo = QComboBox()
            self.fluor_combo.addItem("Fluorescein (pH 11)", "fluorescein_ph11")
            self.fluor_combo.addItem("Rhodamine 590 (methanol)", "rhodamine_590_methanol")
            row1.addWidget(self.fluor_combo)
            row1.addWidget(QLabel("Objective:"))
            self.objective_combo = QComboBox()
            self.objective_combo.addItem("TL15X-2P (0.70 NA)", "TL15X-2P")
            self.objective_combo.addItem("TL10X-2P (0.50 NA)", "TL10X-2P")
            row1.addWidget(self.objective_combo)
            row1.addWidget(QLabel("Solver:"))
            self.mode_combo = QComboBox()
            self.mode_combo.addItems(["fast", "vectorial"])
            row1.addWidget(self.mode_combo)

            row2 = QHBoxLayout()
            self.wavelength_spin = QDoubleSpinBox()
            self.wavelength_spin.setRange(300.0, 2000.0)
            self.wavelength_spin.setValue(self.config.laser.center_wavelength_nm)
            self.wavelength_spin.setSuffix(" nm")
            row2.addWidget(QLabel("Laser λ:")); row2.addWidget(self.wavelength_spin)
            self.power_spin = QDoubleSpinBox()
            self.power_spin.setRange(0.0001, 10000.0)
            self.power_spin.setDecimals(3)
            self.power_spin.setValue(self.config.laser.average_power_w * 1000.0)
            self.power_spin.setSuffix(" mW")
            row2.addWidget(QLabel("Power:")); row2.addWidget(self.power_spin)
            self.rep_spin = QDoubleSpinBox()
            self.rep_spin.setRange(0.001, 1000.0)
            self.rep_spin.setValue(self.config.laser.repetition_rate_hz / 1e6)
            self.rep_spin.setSuffix(" MHz")
            row2.addWidget(QLabel("Rep rate:")); row2.addWidget(self.rep_spin)
            self.pulse_spin = QDoubleSpinBox()
            self.pulse_spin.setRange(1.0, 100000.0)
            self.pulse_spin.setValue(self.config.laser.transform_limit_fwhm_fs)
            self.pulse_spin.setSuffix(" fs")
            row2.addWidget(QLabel("Pulse:")); row2.addWidget(self.pulse_spin)

            row3 = QHBoxLayout()
            self.beam_spin = QDoubleSpinBox()
            self.beam_spin.setRange(0.01, 100.0)
            self.beam_spin.setValue(self.config.laser.beam_diameter_mm)
            self.beam_spin.setSuffix(" mm")
            row3.addWidget(QLabel("Beam Ø:")); row3.addWidget(self.beam_spin)
            self.molecules_spin = QDoubleSpinBox()
            self.molecules_spin.setRange(1.0, 1e12)
            self.molecules_spin.setDecimals(0)
            self.molecules_spin.setValue(self.config.emitter.molecule_count)
            row3.addWidget(QLabel("Molecules:")); row3.addWidget(self.molecules_spin)
            self.dwell_spin = QDoubleSpinBox()
            self.dwell_spin.setRange(0.001, 1e6)
            self.dwell_spin.setValue(self.config.dwell_time_us)
            self.dwell_spin.setSuffix(" µs")
            row3.addWidget(QLabel("Dwell:")); row3.addWidget(self.dwell_spin)
            self.btn_template = QPushButton("Build Microscope Template")
            self.btn_template.clicked.connect(self._build_template)
            row3.addWidget(self.btn_template)
            self.btn_run = QPushButton("Simulate")
            self.btn_run.clicked.connect(self._run_clicked)
            row3.addWidget(self.btn_run)

            self.status_label = QLabel("Calibrated inputs ready. Select Simulate.")
            self.status_label.setWordWrap(True)
            self.status_label.setStyleSheet("color: #00695C; font-weight: bold; padding: 3px;")
            if self.layout() is not None:
                self.layout().insertLayout(0, row1)
                self.layout().insertLayout(1, row2)
                self.layout().insertLayout(2, row3)
                self.layout().addWidget(self.status_label)

    def _read_controls(self) -> None:
        if not QT_AVAILABLE or not hasattr(self, "power_spin"):
            return
        from optics_app.catalogs.microscopy import get_fluorophore_standard, get_microscope_objective
        from optics_app.analysis.two_photon import FluorophoreSpec, ObjectiveSpec
        self.config.laser.center_wavelength_nm = float(self.wavelength_spin.value())
        self.config.laser.average_power_w = float(self.power_spin.value()) * 1e-3
        self.config.laser.repetition_rate_hz = float(self.rep_spin.value()) * 1e6
        self.config.laser.transform_limit_fwhm_fs = float(self.pulse_spin.value())
        self.config.laser.beam_diameter_mm = float(self.beam_spin.value())
        self.config.emitter.molecule_count = float(self.molecules_spin.value())
        self.config.dwell_time_us = float(self.dwell_spin.value())
        self.config.solver_mode = str(self.mode_combo.currentText())
        fluor_id = str(self.fluor_combo.currentData())
        objective_id = str(self.objective_combo.currentData())
        if fluor_id != self._selected_fluor_id:
            fluor = get_fluorophore_standard(fluor_id)
            if fluor:
                self.config.fluorophore = FluorophoreSpec.from_catalog(fluor)
                self._selected_fluor_id = fluor_id
        if objective_id != self._selected_objective_id:
            objective = get_microscope_objective(objective_id)
            if objective:
                self.config.objective = ObjectiveSpec.from_catalog(objective)
                self._selected_objective_id = objective_id

    def get_config(self) -> Dict[str, Any]:
        self._read_controls()
        return self.config.to_dict()

    def set_config(self, data: Optional[Dict[str, Any]]) -> None:
        self.config = TwoPhotonMicroscopeConfig.from_dict(data) if data else build_calibrated_template()
        self._selected_fluor_id = self.config.fluorophore.identifier
        self._selected_objective_id = self.config.objective.part_number
        if QT_AVAILABLE and hasattr(self, "power_spin"):
            self.wavelength_spin.setValue(self.config.laser.center_wavelength_nm)
            self.power_spin.setValue(self.config.laser.average_power_w * 1000.0)
            self.rep_spin.setValue(self.config.laser.repetition_rate_hz / 1e6)
            self.pulse_spin.setValue(self.config.laser.transform_limit_fwhm_fs)
            self.beam_spin.setValue(self.config.laser.beam_diameter_mm)
            self.molecules_spin.setValue(self.config.emitter.molecule_count)
            self.dwell_spin.setValue(self.config.dwell_time_us)
            idx = self.mode_combo.findText(self.config.solver_mode)
            if idx >= 0: self.mode_combo.setCurrentIndex(idx)
            idx = self.objective_combo.findData(self.config.objective.part_number)
            if idx < 0 and self.config.objective.part_number:
                self.objective_combo.addItem(f"Custom: {self.config.objective.part_number}", self.config.objective.part_number)
                idx = self.objective_combo.findData(self.config.objective.part_number)
            if idx >= 0: self.objective_combo.setCurrentIndex(idx)
            idx = self.fluor_combo.findData(self.config.fluorophore.identifier)
            if idx < 0 and self.config.fluorophore.identifier:
                self.fluor_combo.addItem(f"Custom: {self.config.fluorophore.name or self.config.fluorophore.identifier}", self.config.fluorophore.identifier)
                idx = self.fluor_combo.findData(self.config.fluorophore.identifier)
            if idx >= 0: self.fluor_combo.setCurrentIndex(idx)

    def _build_template(self) -> None:
        stack = build_two_photon_surface_template()
        self.config = stack.two_photon_config
        if self.model is not None:
            if hasattr(self.model, "_push_undo"):
                self.model._push_undo("Build Two-Photon Microscope Template")
            if hasattr(self.model, "set_surface_stack"):
                self.model.set_surface_stack(stack)
        self.update_plot(stack=stack, force_run=True)

    def _run_clicked(self) -> None:
        stack = None
        if self.model is not None and hasattr(self.model, "get_surface_stack"):
            stack = self.model.get_surface_stack()
        self.update_plot(stack=stack, force_run=True)

    def _replot(self, *_args):
        stack = getattr(self, "last_stack", None)
        if stack is None and self.model is not None and hasattr(self.model, "get_surface_stack"):
            try:
                stack = self.model.get_surface_stack()
            except Exception:
                pass
        self.update_plot(stack=stack)

    def update_plot(self, stack: Optional[Any] = None, force_run: bool = False, **_: Any) -> None:
        if not (QT_AVAILABLE and HAS_MPL_QT) or self.fig is None:
            return
        if stack is not None:
            self.last_stack = stack
        elif getattr(self, "last_stack", None) is not None:
            stack = self.last_stack

        self._read_controls()
        if stack is not None:
            stack.two_photon_config = self.config
            stack.optical_routes = self.config.routes
        self.last_result = TwoPhotonMicroscopeSimulator.run(self.config, mode=self.config.solver_mode, stack=stack)
        result = self.last_result
        self.fig.clf()
        if not result.valid:
            ax = self.fig.add_subplot(111)
            ax.axis("off")
            err_col = "#f87171" if self.dark_mode else "#B71C1C"
            ax.text(0.02, 0.98, "Calibrated simulation blocked:\n\n" + "\n".join(f"• {msg}" for msg in result.errors), va="top", color=err_col, fontsize=10)
            self.apply_dark_mode_to_axes(ax)
            self.status_label.setText(f"Blocked by {len(result.errors)} missing or invalid calibrated input(s).")
            self.canvas.draw_idle()
            return
        ax_xy = self.fig.add_subplot(231)
        ax_xz = self.fig.add_subplot(232)
        ax_scan = self.fig.add_subplot(233)
        ax_spec = self.fig.add_subplot(234)
        ax_power = self.fig.add_subplot(235)
        ax_budget = self.fig.add_subplot(236)
        focus = result.focus
        if focus is not None:
            ax_xy.imshow(focus.xy_intensity, origin="lower", cmap="inferno", extent=[focus.x_um[0], focus.x_um[-1], focus.y_um[0], focus.y_um[-1]], aspect="equal")
            ax_xy.set_title("Two-Photon Focal Field XY")
            ax_xy.set_xlabel("x (µm)"); ax_xy.set_ylabel("y (µm)")
            ax_xz.imshow(focus.xz_intensity, origin="lower", cmap="magma", extent=[focus.x_um[0], focus.x_um[-1], focus.z_um[0], focus.z_um[-1]], aspect="auto")
            ax_xz.set_title("High-NA Field XZ")
            ax_xz.set_xlabel("x (µm)"); ax_xz.set_ylabel("z (µm)")
        scan_col = "#38bdf8" if self.dark_mode else "#1565C0"
        ax_scan.plot(result.scan_positions_um, result.scan_counts_per_second, color=scan_col)
        ax_scan.set_title("Scan Across Point Emitter")
        ax_scan.set_xlabel("Scan position (µm)"); ax_scan.set_ylabel("Detected counts/s")
        ax_scan.grid(True, alpha=0.3)
        spec_gen_col = "#4ade80" if self.dark_mode else "#43A047"
        spec_pmt_col = "#c084fc" if self.dark_mode else "#6A1B9A"
        ax_spec.plot(result.emission_wavelength_nm, result.emission_spectrum, label="Generated", color=spec_gen_col)
        ax_spec.plot(result.emission_wavelength_nm, result.detected_spectrum, label="At PMT", color=spec_pmt_col)
        ax_spec.set_title("Fluorescence Spectrum")
        ax_spec.set_xlabel("Wavelength (nm)"); ax_spec.set_ylabel("Normalized density")
        ax_spec.legend(fontsize=8); ax_spec.grid(True, alpha=0.3)
        power_labels = ["Laser", "Microscope", "Dump", "Sample"]
        power_values = [self.config.laser.average_power_w, result.microscope_power_w, result.dump_power_w, result.sample_average_power_w]
        ax_power.bar(power_labels, np.asarray(power_values) * 1e3, color=["#60a5fa" if self.dark_mode else "#3949AB", "#f87171" if self.dark_mode else "#E53935", "#fb923c" if self.dark_mode else "#FB8C00", "#c084fc" if self.dark_mode else "#8E24AA"])
        ax_power.set_title("Power by Stage")
        ax_power.set_ylabel("Average power (mW)")
        ax_power.tick_params(axis="x", labelrotation=25)
        detected_per_pulse = max(0.0, result.detected_counts_per_second - self.config.pmt.dark_count_rate_hz) / max(self.config.laser.repetition_rate_hz, 1e-30)
        budget_values = [
            result.excitation_probability_per_molecule_per_pulse * self.config.emitter.molecule_count,
            result.emitted_photons_per_pulse,
            result.emitted_photons_per_pulse * result.collection_efficiency,
            detected_per_pulse,
        ]
        ax_budget.bar(["Excited", "Emitted", "Collected", "Detected"], np.maximum(budget_values, 1e-30), color="#2dd4bf" if self.dark_mode else "#00897B")
        ax_budget.set_yscale("log")
        ax_budget.set_title("Photon Budget / Pulse")
        ax_budget.set_ylabel("Photons or events")
        ax_budget.tick_params(axis="x", labelrotation=25)
        warning_text = f" | Warnings: {len(result.warnings)}" if result.warnings else ""
        self.status_label.setText(
            f"Sample {result.sample_average_power_w*1e3:.3f} mW | Dump {result.dump_power_w*1e3:.3f} mW | "
            f"Pulse {result.pulse_width_at_sample_fs:.1f} fs | Pexc {result.excitation_probability_per_molecule_per_pulse:.3g}/pulse | "
            f"Detected {result.detected_counts_per_second:.3g} counts/s ±{result.calibration_relative_uncertainty:.1%} calibration | "
            f"{result.detected_counts_per_dwell:.3g} counts/dwell{warning_text}"
        )
        self.status_label.setStyleSheet(
            f"color: {'#38bdf8' if self.dark_mode else '#00695C'}; font-weight: bold; padding: 3px;"
        )

        for sub_ax in (ax_xy, ax_xz, ax_scan, ax_spec, ax_power, ax_budget):
            self.apply_dark_mode_to_axes(sub_ax)

        self.apply_tight_layout()
        self.canvas.draw_idle()


class NonlinearImagingTab(BaseAnalysisTab):
    """Two synchronized beams, frequency conversion, Raman, and PMT view."""

    def __init__(self, model: Optional[Any] = None, parent=None):
        super().__init__(parent)
        self.model = model
        self.config = build_nonlinear_surface_template().nonlinear_imaging_config
        self.last_result = None
        self.last_stack = None
        if not QT_AVAILABLE:
            return
        laser_row = QHBoxLayout()
        self.pump_wl = self._spin(laser_row, "Pump λ", 300, 3000, 800, " nm")
        self.pump_power = self._spin(laser_row, "Pump power", 0, 10000, 20, " mW")
        self.stokes_wl = self._spin(laser_row, "Stokes λ", 300, 3000, 1040, " nm")
        self.stokes_power = self._spin(laser_row, "Stokes power", 0, 10000, 20, " mW")
        pulse_row = QHBoxLayout()
        self.pulse_width = self._spin(pulse_row, "Pulse", 1, 100000, 100, " fs")
        self.delay = self._spin(pulse_row, "Delay", -100000, 100000, 0, " fs")
        self.dwell = self._spin(pulse_row, "Dwell", 0.001, 100000, 2, " µs")
        self.process = QComboBox()
        self.process.addItems(["SFG", "DFG", "CARS", "SRS"])
        pulse_row.addWidget(QLabel("Process:")); pulse_row.addWidget(self.process)
        sample_row = QHBoxLayout()
        self.sample_x = self._spin(sample_row, "Sample X", -1000, 1000, 0, " µm")
        self.sample_y = self._spin(sample_row, "Sample Y", -1000, 1000, 0, " µm")
        self.sample_z = self._spin(sample_row, "Sample Z", -1000, 1000, 0, " µm")
        actions = QHBoxLayout()
        self.template_button = QPushButton("Build Dual-Pulse Microscope")
        self.template_button.clicked.connect(self._build_template)
        actions.addWidget(self.template_button)
        self.calibration_button = QPushButton("Load Calibration JSON")
        self.calibration_button.clicked.connect(self._load_calibration)
        actions.addWidget(self.calibration_button)
        self.run_button = QPushButton("Simulate")
        self.run_button.clicked.connect(self._run_clicked)
        actions.addWidget(self.run_button)
        self.status_label = QLabel("Select a calibrated sample/optics JSON to predict absolute signals.")
        self.status_label.setWordWrap(True)
        self.layout().insertLayout(0, laser_row)
        self.layout().insertLayout(1, pulse_row)
        self.layout().insertLayout(2, actions)
        self.layout().insertLayout(2, sample_row)
        self.layout().addWidget(self.status_label)

    @staticmethod
    def _spin(row: Any, label: str, minimum: float, maximum: float,
              value: float, suffix: str) -> Any:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(3)
        spin.setValue(value)
        spin.setSuffix(suffix)
        row.addWidget(QLabel(label + ":"))
        row.addWidget(spin)
        return spin

    def _read_controls(self) -> None:
        if not QT_AVAILABLE or not hasattr(self, "pump_wl"):
            return
        self.config.pump.center_wavelength_nm = self.pump_wl.value()
        self.config.pump.average_power_w = self.pump_power.value() * 1e-3
        self.config.stokes.center_wavelength_nm = self.stokes_wl.value()
        self.config.stokes.average_power_w = self.stokes_power.value() * 1e-3
        self.config.pump.transform_limit_fwhm_fs = self.pulse_width.value()
        self.config.stokes.transform_limit_fwhm_fs = self.pulse_width.value()
        self.config.combiner.delay_fs = self.delay.value()
        self.config.dwell_time_us = self.dwell.value()
        self.config.sample_x_um = self.sample_x.value()
        self.config.sample_y_um = self.sample_y.value()
        self.config.sample_z_um = self.sample_z.value()

    def get_config(self) -> Dict[str, Any]:
        self._read_controls()
        return self.config.to_dict()

    def set_config(self, data: Optional[Dict[str, Any]]) -> None:
        self.config = NonlinearImagingConfig.from_dict(data) if data else build_nonlinear_surface_template().nonlinear_imaging_config
        if QT_AVAILABLE and hasattr(self, "pump_wl"):
            self.pump_wl.setValue(self.config.pump.center_wavelength_nm)
            self.pump_power.setValue(self.config.pump.average_power_w * 1e3)
            self.stokes_wl.setValue(self.config.stokes.center_wavelength_nm)
            self.stokes_power.setValue(self.config.stokes.average_power_w * 1e3)
            self.pulse_width.setValue(self.config.pump.transform_limit_fwhm_fs)
            self.delay.setValue(self.config.combiner.delay_fs)
            self.dwell.setValue(self.config.dwell_time_us)
            self.sample_x.setValue(self.config.sample_x_um)
            self.sample_y.setValue(self.config.sample_y_um)
            self.sample_z.setValue(self.config.sample_z_um)

    def _load_calibration(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load nonlinear calibration", "", "JSON Files (*.json)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            if not isinstance(data, dict):
                raise ValueError("Expected a JSON object.")
            payload = data.get("nonlinear_imaging_config", data)
            if not isinstance(payload, dict):
                raise ValueError("Expected a nonlinear_imaging_config JSON object.")
            merged = self.config.to_dict()
            def merge(target: Dict[str, Any], incoming: Dict[str, Any]) -> None:
                for key, value in incoming.items():
                    if isinstance(value, dict) and isinstance(target.get(key), dict):
                        merge(target[key], value)
                    else:
                        target[key] = value
            merge(merged, payload)
            self.set_config(merged)
            self._run_clicked()
        except (OSError, ValueError, TypeError) as exc:
            self.status_label.setText(f"Calibration could not be loaded: {exc}")

    def _build_template(self) -> None:
        stack = build_nonlinear_surface_template()
        self.config = stack.nonlinear_imaging_config
        self.set_config(self.config.to_dict())
        if self.model is not None:
            if hasattr(self.model, "_push_undo"):
                self.model._push_undo("Build Dual-Pulse Nonlinear Microscope")
            if hasattr(self.model, "set_surface_stack"):
                self.model.set_surface_stack(stack)
        self.update_plot(stack=stack)

    def _run_clicked(self) -> None:
        stack = self.model.get_surface_stack() if self.model is not None and hasattr(self.model, "get_surface_stack") else None
        self.update_plot(stack=stack)

    def _replot(self, *_args):
        stack = getattr(self, "last_stack", None)
        if stack is None and self.model is not None and hasattr(self.model, "get_surface_stack"):
            try:
                stack = self.model.get_surface_stack()
            except Exception:
                pass
        self.update_plot(stack=stack)

    def update_plot(self, stack: Optional[Any] = None, **_: Any) -> None:
        if not (QT_AVAILABLE and HAS_MPL_QT) or self.fig is None:
            return
        if stack is not None:
            self.last_stack = stack
        elif getattr(self, "last_stack", None) is not None:
            stack = self.last_stack

        self._read_controls()
        if stack is not None and hasattr(stack, "nonlinear_imaging_config"):
            stack.nonlinear_imaging_config = self.config
            stack.optical_routes = self.config.routes
        self.last_result = NonlinearImagingSimulator.run(self.config)
        result = self.last_result
        channel = result.channels[str(self.process.currentText())]
        self.fig.clf()
        axes = self.fig.subplots(2, 2)
        overlap_col = "#818cf8" if self.dark_mode else "#3949AB"
        axes[0, 0].plot(result.delay_fs, result.temporal_overlap, color=overlap_col)
        axes[0, 0].axvline(self.config.combiner.delay_fs, color="#ef4444" if self.dark_mode else "#D32F2F", linestyle="--")
        axes[0, 0].set(title="Pulse overlap", xlabel="Delay (fs)", ylabel="Relative overlap")

        scan_col = "#34d399" if self.dark_mode else "#00897B"
        axes[0, 1].plot(result.scan_positions_um, result.scan_profiles.get(channel.process, result.scan_response), color=scan_col)
        axes[0, 1].set(title="Point-sample scan", xlabel="Scan position (µm)", ylabel="Relative signal")

        names = list(result.channels)
        wavelengths = [result.channels[name].wavelength_nm or 0 for name in names]
        axes[1, 0].bar(names, wavelengths, color=["#60a5fa" if self.dark_mode else "#2196F3",
                                                  "#f87171" if self.dark_mode else "#B71C1C",
                                                  "#4ade80" if self.dark_mode else "#43A047",
                                                  "#c084fc" if self.dark_mode else "#7E57C2"])
        axes[1, 0].set(title="Output wavelengths (SRS = Stokes modulation)", ylabel="Wavelength (nm)")
        powers = [result.channels[name].generated_power_w or result.channels[name].stokes_gain_w or 0 for name in names]
        axes[1, 1].bar(names, np.asarray(powers)*1e9, color="#c084fc" if self.dark_mode else "#6A1B9A")
        axes[1, 1].set(title="Signal (calibrated channels only)", ylabel="Power / modulation (nW)")

        for r in range(2):
            for c in range(2):
                self.apply_dark_mode_to_axes(axes[r, c])

        blocking = [issue.message for issue in result.issues if issue.level == "error"]
        blocking += [issue.message for issue in channel.issues if issue.level == "error"]
        if blocking:
            self.status_label.setText(f"{channel.process} blocked: " + " ".join(blocking[:3]))
        else:
            counts = (f"{channel.pmt_counts_per_second:.3g} PMT counts/s"
                      if channel.pmt_counts_per_second is not None
                      else f"{channel.pmt_photoelectron_rate_hz:.3g} photocathode electrons/s; counts unavailable"
                      if channel.pmt_photoelectron_rate_hz is not None
                      else "PMT prediction unavailable")
            self.status_label.setText(
                f"{channel.process}: {channel.wavelength_nm:.1f} nm | "
                f"{channel.generated_power_w if channel.generated_power_w is not None else channel.stokes_gain_w:.3g} W | {counts}. "
                + " ".join(issue.message for issue in channel.issues[:2]))
        self.apply_tight_layout()
        self.canvas.draw_idle()


class BestFocusAnalysisTab(BaseAnalysisTab):
    """Through-focus geometric spot and least-confusion analysis at the detector."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.last_system = None
        self.last_wavelengths = None
        self.last_system_config = None
        self.last_result: Optional[BestFocusResult] = None
        if QT_AVAILABLE and HAS_MPL_QT:
            controls = QHBoxLayout()
            controls.addWidget(QLabel("Focus ±"))
            self.focus_range = QDoubleSpinBox()
            self.focus_range.setRange(0.05, 100.0)
            self.focus_range.setDecimals(2)
            self.focus_range.setValue(5.0)
            self.focus_range.setSuffix(" mm")
            controls.addWidget(self.focus_range)
            controls.addWidget(QLabel("Pupil samples"))
            self.pupil_samples = QSpinBox()
            self.pupil_samples.setRange(5, 51)
            self.pupil_samples.setSingleStep(2)
            self.pupil_samples.setValue(11)
            controls.addWidget(self.pupil_samples)
            controls.addWidget(QLabel("Reference"))
            self.reference_combo = QComboBox()
            self.reference_combo.addItem("Centroid", "centroid")
            self.reference_combo.addItem("Chief ray", "chief")
            controls.addWidget(self.reference_combo)
            self.include_slit = QCheckBox("Include slit width")
            controls.addWidget(self.include_slit)
            controls.addStretch(1)
            self.status_label = QLabel("Trace an optical system to find its least-confusion plane.")
            self.status_label.setWordWrap(True)
            layout = self.layout()
            layout.insertLayout(1 if self.toolbar else 0, controls)
            layout.addWidget(self.status_label)
            self.focus_range.valueChanged.connect(self._replot)
            self.pupil_samples.valueChanged.connect(self._replot)
            self.reference_combo.currentIndexChanged.connect(self._replot)
            self.include_slit.stateChanged.connect(self._replot)

    def _replot(self, *_args):
        self.update_plot(self.last_system, self.last_wavelengths, self.last_system_config)

    def update_plot(self, system=None, wavelengths=None, system_config=None):
        if not (QT_AVAILABLE and HAS_MPL_QT) or self.fig is None:
            return
        theme = self.get_theme_colors()
        self.last_system = system
        self.last_wavelengths = wavelengths
        self.last_system_config = system_config
        self.fig.clf()
        if system is None:
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, "No optical system loaded", ha="center", va="center", color=theme["text_dim"])
            self.apply_dark_mode_to_axes(ax)
            self.status_label.setText("Load a system with a detector surface.")
            self.canvas.draw_idle()
            return
        try:
            result = analyze_system_best_focus(
                system, system_config=system_config, wavelengths_um=wavelengths,
                pupil_samples=self.pupil_samples.value(),
                include_source_extent=self.include_slit.isChecked(),
                reference=self.reference_combo.currentData(),
                half_range_mm=self.focus_range.value(), samples=121,
            )
        except Exception as exc:
            self.last_result = None
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, f"Focus analysis failed:\n{exc}", ha="center", va="center", color="#f87171" if self.dark_mode else "#b91c1c")
            self.apply_dark_mode_to_axes(ax)
            self.status_label.setText(str(exc))
            self.canvas.draw_idle()
            return
        self.last_result = result
        if not result.valid:
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, "No valid focus plane\n" + "\n".join(result.warnings),
                    ha="center", va="center", color="#f87171" if self.dark_mode else "#b91c1c")
            self.apply_dark_mode_to_axes(ax)
            self.status_label.setText(" ".join(result.warnings))
            self.canvas.draw_idle()
            return

        ax_curve = self.fig.add_subplot(221)
        ax_nominal = self.fig.add_subplot(222)
        ax_best = self.fig.add_subplot(223)
        ax_color = self.fig.add_subplot(224)
        shifts = result.focus_shifts_mm
        primary_col = "#f1f5f9" if self.dark_mode else "#0f172a"
        ax_curve.plot(shifts, result.rms_radial_um, color=primary_col, lw=2, label="RMS radial, all λ")
        ax_curve.plot(shifts, result.rms_x_um, color="#38bdf8" if self.dark_mode else "#2563eb", lw=1.2, label="RMS X, all λ")
        ax_curve.plot(shifts, result.rms_y_um, color="#f87171" if self.dark_mode else "#dc2626", lw=1.2, label="RMS Y, all λ")
        colors = ["#4ade80" if self.dark_mode else "#16a34a",
                  "#c084fc" if self.dark_mode else "#a855f7",
                  "#fbbf24" if self.dark_mode else "#f59e0b",
                  "#22d3ee" if self.dark_mode else "#0891b2",
                  "#fb7185" if self.dark_mode else "#e11d48"]
        for i, (wl, curve) in enumerate(result.per_wavelength_curves_um.items()):
            ax_curve.plot(shifts, curve, color=colors[i % len(colors)], lw=1, ls="--", alpha=0.75,
                          label=f"{wl * 1000:.0f} nm")
        ax_curve.axvline(0.0, color="#94a3b8" if self.dark_mode else "#64748b", ls=":", label="Detector")
        ax_curve.axvline(result.best_shift_mm, color=primary_col, ls="--", label="Least confusion")
        if result.circle_shift_mm is not None:
            ax_curve.axvline(result.circle_shift_mm, color="#fbbf24" if self.dark_mode else "#d97706", ls="-.", label="Equal X/Y blur")
        ax_curve.set_title("RMS vs detector focus")
        ax_curve.set_xlabel("Detector shift along local +Z (mm)")
        positive_rms = result.rms_radial_um[result.rms_radial_um > 0]
        if len(positive_rms) and float(np.max(positive_rms) / np.min(positive_rms)) > 8.0:
            ax_curve.set_yscale("log")
            ax_curve.set_ylabel("Geometric RMS (µm, log scale)")
        else:
            ax_curve.set_ylabel("Geometric RMS (µm)")
        ax_curve.grid(alpha=0.3)
        ax_curve.legend(fontsize=6, ncol=2)

        all_spots = list(result.spots_nominal_mm.values()) + list(result.spots_best_mm.values())
        extent = max((float(np.max(np.abs(arr))) for arr in all_spots if len(arr)), default=0.001) * 1000.0
        extent = max(extent * 1.1, 1.0)
        for axis, spots, title in (
            (ax_nominal, result.spots_nominal_mm, f"Current detector: RMS {result.nominal.rms_radius_um:.1f} µm"),
            (ax_best, result.spots_best_mm, f"Least confusion: RMS {result.best.rms_radius_um:.1f} µm"),
        ):
            for i, (wl, positions) in enumerate(spots.items()):
                axis.scatter(positions[:, 0] * 1000.0, positions[:, 1] * 1000.0,
                             s=9, alpha=0.65, color=colors[i % len(colors)], label=f"{wl * 1000:.0f} nm")
            axis.set_title(title)
            axis.set_xlabel("Local X (µm)")
            axis.set_ylabel("Local Y (µm)")
            axis.set_xlim(-extent, extent)
            axis.set_ylim(-extent, extent)
            axis.set_aspect("equal", adjustable="box")
            axis.grid(alpha=0.25)
        ax_nominal.legend(fontsize=6)

        wavelengths_sorted = sorted(result.per_wavelength_best_mm)
        ax_color.plot([w * 1000.0 for w in wavelengths_sorted],
                      [result.per_wavelength_best_mm[w] for w in wavelengths_sorted],
                      "o-", color="#a78bfa" if self.dark_mode else "#7c3aed", lw=1.7, label="Per-λ radial best")
        ax_color.axhline(0.0, color="#94a3b8" if self.dark_mode else "#64748b", ls=":", label="Detector")
        ax_color.axhline(result.best_shift_mm, color=primary_col, ls="--", label="Polychromatic best")
        if result.x_focus_shift_mm is not None:
            ax_color.axhline(result.x_focus_shift_mm, color="#38bdf8" if self.dark_mode else "#2563eb", ls="-.", label="Primary X focus")
        if result.y_focus_shift_mm is not None:
            ax_color.axhline(result.y_focus_shift_mm, color="#f87171" if self.dark_mode else "#dc2626", ls="-.", label="Primary Y focus")
        ax_color.set_title("Longitudinal color and X/Y line foci")
        ax_color.set_xlabel("Wavelength (nm)")
        ax_color.set_ylabel("Best focus shift (mm)")
        ax_color.grid(alpha=0.3)
        ax_color.legend(fontsize=6)

        for sub_ax in (ax_curve, ax_nominal, ax_best, ax_color):
            self.apply_dark_mode_to_axes(sub_ax)

        astig = result.xy_astigmatic_separation_mm
        chromatic = result.longitudinal_color_mm
        status = (f"Least-confusion shift {result.best_shift_mm:+.4f} mm | RMS "
                  f"{result.nominal.rms_radius_um:.2f} → {result.best.rms_radius_um:.2f} µm | "
                  f"primary X/Y focus separation {astig:.4f} mm" if astig is not None else
                  f"Least-confusion shift {result.best_shift_mm:+.4f} mm")
        if chromatic is not None:
            status += f" | longitudinal color {chromatic:.4f} mm"
        if result.warnings:
            status += " | " + " ".join(result.warnings)
        self.status_label.setText(status)
        self.apply_tight_layout()
        self.canvas.draw_idle()


class ThorlabsSRSLauncherTab(BaseAnalysisTab):
    """
    Direct bridge and quick launcher for the 18-element Thorlabs SRS Microscope.
    Allows users who open the Optical Analysis Diagnostics dock to directly load the
    Thorlabs SRS preset and focus Tab 6 of the Ultrafast Optics dock.
    """
    def __init__(self, model: Optional[Any] = None, parent=None):
        super().__init__(parent)
        self.model = model
        if not QT_AVAILABLE:
            return

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        self.header = QLabel("<h2>🔬 Thorlabs SRS Microscope & 2D Ray Tracing Simulation</h2>")
        self.header.setWordWrap(True)
        layout.addWidget(self.header)

        self.desc = QLabel(
            "<b>The full 18-component Thorlabs SRS Microscope simulation is housed in the Ultrafast Optics Dock (Tab 6).</b><br><br>"
            "This comprehensive system includes:<br>"
            "• <b>Modulators:</b> High-speed EOM (EO-AM-NR-C1) or AOM with RF driver<br>"
            "• <b>Scanning Engine:</b> Dual-axis Galvo (GVS002 X/Y), Scan Lens (SL50-2P2), Tube Lens (TTL200MP)<br>"
            "• <b>Focusing & Collection:</b> Objective (TL15X-2P, NA 0.70), Condenser (ACL25416U-B, NA 0.79)<br>"
            "• <b>Detection:</b> Shortpass filter (FESH0850), Achromatic doublet (AC254-050-B), High-speed Photodiode (DET10N2)<br>"
            "• <b>Spectroscopy:</b> Spectral focusing chirp matching (α = 0.0815 ps/nm) & Lock-in amplifier demodulation"
        )
        self.desc.setWordWrap(True)
        self.desc.setStyleSheet("background-color: #F1F8E9; color: #1e293b; border: 1px solid #C8E6C9; border-radius: 6px; padding: 10px; font-size: 12px;")
        layout.addWidget(self.desc)

        btn_box = QHBoxLayout()
        self.btn_open_dock = QPushButton("🚀 Open Thorlabs SRS Microscope in Ultrafast Dock (Tab 6)")
        self.btn_open_dock.setStyleSheet(
            "background-color: #2E7D32; color: white; font-weight: bold; font-size: 13px; "
            "padding: 10px 18px; border-radius: 5px;"
        )
        self.btn_open_dock.clicked.connect(self._on_open_dock)
        btn_box.addWidget(self.btn_open_dock)

        self.btn_load_preset = QPushButton("📂 Load Shipped Thorlabs SRS Preset JSON (23 surfaces)")
        self.btn_load_preset.setStyleSheet(
            "background-color: #1565C0; color: white; font-weight: bold; font-size: 13px; "
            "padding: 10px 18px; border-radius: 5px;"
        )
        self.btn_load_preset.clicked.connect(self._on_load_preset)
        btn_box.addWidget(self.btn_load_preset)
        layout.addLayout(btn_box)

        self.lbl_feedback = QLabel("")
        self.lbl_feedback.setStyleSheet("font-weight: bold; color: #2E7D32; font-size: 12px;")
        layout.addWidget(self.lbl_feedback)

        layout.addStretch()

    def set_dark_mode(self, enabled: bool) -> None:
        self.dark_mode = bool(enabled)
        if not QT_AVAILABLE:
            return
        if hasattr(self, "desc"):
            if self.dark_mode:
                self.desc.setStyleSheet(
                    "background-color: #1e293b; color: #f1f5f9; border: 1px solid #334155; "
                    "border-radius: 6px; padding: 10px; font-size: 12px;"
                )
            else:
                self.desc.setStyleSheet(
                    "background-color: #F1F8E9; color: #1e293b; border: 1px solid #C8E6C9; "
                    "border-radius: 6px; padding: 10px; font-size: 12px;"
                )
        if hasattr(self, "header"):
            text_col = "#f8fafc" if self.dark_mode else "#0f172a"
            self.header.setStyleSheet(f"color: {text_col};")
        if hasattr(self, "lbl_feedback"):
            self.lbl_feedback.setStyleSheet(
                f"font-weight: bold; color: {'#4ade80' if self.dark_mode else '#2E7D32'}; font-size: 12px;"
            )

    def _on_open_dock(self):
        mw = self.window()
        if hasattr(mw, "_show_ultrafast_dock"):
            mw._show_ultrafast_dock()
            self.lbl_feedback.setText("✅ Switched to Ultrafast Optics Dock — Tab 6 active.")

    def _on_load_preset(self):
        mw = self.window()
        import os
        candidates = [
            "thorlabs_srs_microscope.json",
            "optics_app/catalogs/presets/thorlabs_srs_microscope.json",
            "v2/catalogs/presets/thorlabs_srs_microscope.json",
        ]
        found = None
        for c in candidates:
            if os.path.exists(c):
                found = c
                break
        if found and hasattr(mw, "on_load_project"):
            mw.on_load_project(found)
            self.lbl_feedback.setText(f"✅ Loaded {found} into 2D Ray Tracing! Opening Tab 6...")
            if hasattr(mw, "_show_ultrafast_dock"):
                mw._show_ultrafast_dock()


class AchromatAnalysisTab(BaseAnalysisTab):
    """
    Tab: Achromatic Doublet Lens Designer & Optimization Suite.
    Live visualization of refractive index dispersion curves n(lambda), Abbe diagram,
    chromatic focal shift Delta f(lambda), and transverse ray aberration fans with surface optimization controls.
    """
    def __init__(self, model: Optional[Any] = None, parent=None):
        super().__init__(parent)
        self.model = model
        self.last_result = None

        if QT_AVAILABLE:
            ctrl_box = QHBoxLayout()
            ctrl_box.setContentsMargins(4, 2, 4, 2)
            ctrl_box.setSpacing(6)

            ctrl_box.addWidget(QLabel("EFL:"))
            self.spin_efl = QDoubleSpinBox()
            self.spin_efl.setRange(10.0, 5000.0)
            self.spin_efl.setValue(100.0)
            self.spin_efl.setSingleStep(10.0)
            self.spin_efl.setSuffix(" mm")
            ctrl_box.addWidget(self.spin_efl)

            ctrl_box.addWidget(QLabel("Crown:"))
            self.combo_crown = QComboBox()
            self.combo_crown.addItems(["N-BK7", "N-BAF10", "N-SK16", "N-BALF14", "N-LAK22", "CAF2", "N-FK51A", "FUSED_SILICA", "B270"])
            ctrl_box.addWidget(self.combo_crown)

            ctrl_box.addWidget(QLabel("Flint:"))
            self.combo_flint = QComboBox()
            self.combo_flint.addItems(["SF10", "N-SF10", "N-SF2", "N-SF11", "N-SF5", "N-F2", "N-SF6", "N-LASF9", "N-KZFS11"])
            ctrl_box.addWidget(self.combo_flint)

            self.btn_opt = QPushButton("⚡ Optimize Achromat")
            self.btn_opt.setStyleSheet("background-color: #0284c7; color: white; font-weight: bold; padding: 4px 10px; border-radius: 3px;")
            self.btn_opt.clicked.connect(self._on_optimize_clicked)
            ctrl_box.addWidget(self.btn_opt)

            self.btn_insert = QPushButton("📥 Insert Doublet into System")
            self.btn_insert.setStyleSheet("background-color: #16a34a; color: white; font-weight: bold; padding: 4px 10px; border-radius: 3px;")
            self.btn_insert.clicked.connect(self._on_insert_into_system)
            ctrl_box.addWidget(self.btn_insert)

            self.btn_full_dlg = QPushButton("🎨 Full Designer...")
            self.btn_full_dlg.clicked.connect(self._on_open_full_dialog)
            ctrl_box.addWidget(self.btn_full_dlg)

            ctrl_box.addStretch(1)

            if hasattr(self, "layout") and self.layout():
                self.layout().insertLayout(0, ctrl_box)
                self.lbl_info = QLabel("Ready. Select glasses and click 'Optimize Achromat'.")
                self.lbl_info.setStyleSheet("font-size: 11px; color: #475569; padding: 2px 4px;")
                self.layout().insertWidget(1, self.lbl_info)

            self._on_optimize_clicked()

    def _on_optimize_clicked(self):
        try:
            from v2.engine.achromat_designer import AchromatDesignSpec, optimize_achromatic_doublet
        except ImportError:
            from optics_app.engine.achromat_designer import AchromatDesignSpec, optimize_achromatic_doublet

        c_glass = self.combo_crown.currentText()
        f_glass = self.combo_flint.currentText()
        efl = float(self.spin_efl.value())

        spec = AchromatDesignSpec(target_efl=efl, crown_glass=c_glass, flint_glass=f_glass)
        try:
            res = optimize_achromatic_doublet(spec)
            self.last_result = res
            self._render_plots(res)
            s_i_drop = abs(res.initial_s_I / max(1e-12, abs(res.optimized_s_I))) if abs(res.optimized_s_I) > 1e-12 else float("inf")
            if hasattr(self, "lbl_info"):
                self.lbl_info.setText(
                    f"✅ R1={res.radii[0]:.2f}, R2={res.radii[1]:.2f}, R3={res.radii[2]:.2f} mm | "
                    f"EFL={res.efl:.2f} mm, BFL={res.bfl:.2f} mm | "
                    f"S_I reduced {s_i_drop:.1f}× ({res.optimized_s_I:.2e} mm) | "
                    f"C_L nulled ({res.optimized_c_L:.2e} mm)"
                )
        except Exception as e:
            if hasattr(self, "lbl_info"):
                self.lbl_info.setText(f"❌ Error: {str(e)}")

    def _on_insert_into_system(self):
        if not self.last_result or not self.model:
            return
        res = self.last_result
        semi_diam = getattr(res.spec, 'diameter', 25.4) / 2.0
        try:
            try:
                from v2.engine.lde import SurfaceStack
                from v2.engine.surface import Surface, SurfaceType
            except ImportError:
                from optics_app.engine.lde import SurfaceStack
                from optics_app.engine.surface import Surface, SurfaceType

            new_stack = SurfaceStack()
            s0 = Surface(name="S0 (OBJ)", surface_type=SurfaceType.OBJECT, radius=0.0, thickness=1000.0, glass="AIR", semi_diameter=semi_diam)
            new_stack.add_surface(s0)

            s1 = Surface(
                name="S1 (STO)", surface_type=SurfaceType.STOP,
                radius=float(res.radii[0]), thickness=float(res.thicknesses[0]),
                glass=res.crown.name, semi_diameter=semi_diam,
                radius_variable=True, thickness_variable=True
            )
            r2 = float(res.radii[1])
            if abs(r2) > 1e7 or getattr(res.spec, 'flat_interface', False):
                r2 = 0.0
            s2 = Surface(
                name="S2 (Contact)", surface_type=SurfaceType.STANDARD,
                radius=r2, thickness=float(res.thicknesses[1]),
                glass=res.flint.name, semi_diameter=semi_diam,
                radius_variable=True, thickness_variable=True
            )
            s3 = Surface(
                name="S3 (Rear)", surface_type=SurfaceType.STANDARD,
                radius=float(res.radii[2]), thickness=max(1.0, float(res.bfl)),
                glass="AIR", semi_diameter=semi_diam,
                radius_variable=True, thickness_variable=True
            )
            new_stack.add_surface(s1)
            new_stack.add_surface(s2)
            new_stack.add_surface(s3)

            s_det = Surface(name="S4 (IMA)", surface_type=SurfaceType.DETECTOR, radius=0.0, thickness=0.0, glass="AIR", semi_diameter=max(5.0, semi_diam * 0.8))
            new_stack.add_surface(s_det)

            if hasattr(self.model, '_push_undo'):
                self.model._push_undo("Insert Achromatic Doublet")
            if hasattr(self.model, 'set_surface_stack'):
                self.model.set_surface_stack(new_stack)
            elif hasattr(self.model, 'surfaces'):
                self.model.surfaces = list(new_stack.surfaces)

            mw = self.window()
            if hasattr(mw, 'on_trace_rays'):
                mw.on_trace_rays()
            if hasattr(mw, 'optimizer_dock') and mw.optimizer_dock:
                if hasattr(mw.optimizer_dock, '_rebuild_surface_checkboxes'):
                    mw.optimizer_dock._rebuild_surface_checkboxes()
                if hasattr(mw.optimizer_dock, 'update_system_info'):
                    mw.optimizer_dock.update_system_info()
            if hasattr(mw, 'table_view') and mw.table_view:
                try:
                    mw.table_view.viewport().update()
                except Exception:
                    pass

            if hasattr(self, 'lbl_info'):
                self.lbl_info.setText(
                    f"✅ Achromatic Doublet Delivered to Active System! (EFL={res.efl:.2f} mm, BFL={res.bfl:.2f} mm, R1={res.radii[0]:.2f}, R2={res.radii[1]:.2f}, R3={res.radii[2]:.2f})"
                )
        except Exception as e:
            if hasattr(self, 'lbl_info'):
                self.lbl_info.setText(f"❌ Failed to insert doublet: {str(e)}")

    def _on_open_full_dialog(self):
        mw = self.window()
        if hasattr(mw, "on_open_achromat_designer_dialog"):
            mw.on_open_achromat_designer_dialog()
        else:
            try:
                from v2.gui.achromat_dialog import AchromatDesignerDialog
            except ImportError:
                from optics_app.gui.achromat_dialog import AchromatDesignerDialog
            dlg = AchromatDesignerDialog(model=self.model, parent=self)
            dlg.exec()

    def _render_plots(self, res):
        if not self.fig:
            return
        self.fig.clear()
        theme = self.get_theme_colors()
        ax1 = self.fig.add_subplot(2, 2, 1)
        ax2 = self.fig.add_subplot(2, 2, 2)
        ax3 = self.fig.add_subplot(2, 2, 3)
        ax4 = self.fig.add_subplot(2, 2, 4)

        for gname, data in res.dispersion_curves.items():
            wl = data["wl_nm"]
            n = data["n"]
            col = "#38bdf8" if (gname == res.crown.name and self.dark_mode) else ("#0284c7" if gname == res.crown.name else "#ea580c")
            ax1.plot(wl, n, label=gname, color=col, linewidth=1.8)
        ax1.axvline(486.1, color="#3b82f6", linestyle=":", alpha=0.7)
        ax1.axvline(587.6, color="#eab308", linestyle=":", alpha=0.7)
        ax1.axvline(656.3, color="#ef4444", linestyle=":", alpha=0.7)
        ax1.set_xlabel("Wavelength (nm)", fontsize=8)
        ax1.set_ylabel("Index n(λ)", fontsize=8)
        ax1.set_title("Dispersion n(λ)", fontsize=9, fontweight="bold")
        ax1.grid(True, linestyle=":", alpha=0.5)
        ax1.legend(fontsize=7)

        ax2.scatter([res.crown.vd], [res.crown.nd], color="#10b981", s=70, label=f"Crown: {res.crown.name}")
        ax2.scatter([res.flint.vd], [res.flint.nd], color="#f97316", s=70, label=f"Flint: {res.flint.name}")
        abbe_line_col = "#a5b4fc" if self.dark_mode else "#6366f1"
        ax2.plot([res.crown.vd, res.flint.vd], [res.crown.nd, res.flint.nd], color=abbe_line_col, linestyle="--")
        ax2.set_xlim(max(95.0, res.crown.vd + 10), min(18.0, res.flint.vd - 5))
        ax2.set_xlabel("Abbe Vd", fontsize=8)
        ax2.set_ylabel("nd", fontsize=8)
        ax2.set_title(f"Abbe Diagram (ΔV={res.delta_v:.1f})", fontsize=9, fontweight="bold")
        ax2.grid(True, linestyle=":", alpha=0.5)
        ax2.legend(fontsize=7)

        wls = [pt[0] for pt in res.chromatic_focal_shift]
        shifts_um = [pt[1] * 1000.0 for pt in res.chromatic_focal_shift]
        ax3.plot(wls, shifts_um, color="#c084fc" if self.dark_mode else "#8b5cf6", linewidth=2.0)
        ax3.axhline(0.0, color="#94a3b8" if self.dark_mode else "#64748b", linestyle="--", alpha=0.5)
        ax3.set_xlabel("Wavelength (nm)", fontsize=8)
        ax3.set_ylabel("ΔBFL (μm)", fontsize=8)
        ax3.set_title("Chromatic Focal Shift Δf(λ)", fontsize=9, fontweight="bold")
        ax3.grid(True, linestyle=":", alpha=0.5)

        colors = {"F (486.1 nm)": "#38bdf8" if self.dark_mode else "#2563eb",
                  "d (587.6 nm)": "#4ade80" if self.dark_mode else "#16a34a",
                  "C (656.3 nm)": "#f87171" if self.dark_mode else "#dc2626"}
        for flabel, fdata in res.ray_fans.items():
            py = fdata["py"]
            ey_um = [e * 1000.0 for e in fdata["ey"]]
            ax4.plot(py, ey_um, label=flabel[:1], color=colors.get(flabel, theme["text"]), linewidth=1.8)
        ax4.set_xlabel("Pupil Py", fontsize=8)
        ax4.set_ylabel("Intercept Ey (μm)", fontsize=8)
        ax4.set_title("Ray Fan Ey(Py)", fontsize=9, fontweight="bold")
        ax4.grid(True, linestyle=":", alpha=0.5)
        ax4.legend(fontsize=7)

        for sub_ax in (ax1, ax2, ax3, ax4):
            self.apply_dark_mode_to_axes(sub_ax)

        self.apply_tight_layout()
        if hasattr(self, "canvas") and self.canvas:
            self.canvas.draw_idle()

    def _replot(self, *_args):
        if self.last_result:
            self._render_plots(self.last_result)
        else:
            self._on_optimize_clicked()

    def update_plot(self, **kwargs):
        if self.last_result is None:
            self._on_optimize_clicked()
        elif self.last_result:
            self._render_plots(self.last_result)


class FirstOrderDiagnosticsTab(BaseAnalysisTab):
    """
    Dedicated Zemax OpticStudio-Style First-Order Optical Diagnostics & Cardinal Planes Tab.
    Displays:
    - 2D Cross-section schematic of optical elements along optical axis Z.
    - Front (P1) and Rear (P2) Principal Planes.
    - Front (F1) and Rear (F2) Focal points.
    - Entrance Pupil (EnP) and Exit Pupil / Effective Back Pupil (ExP) apertures.
    - Paraxial marginal and chief ray envelopes.
    - Complete paraxial prescription data table: EFFL, BFL, FFL, Power (dpt), EPD, EXPD,
      P1, P2, Delta P, Working F/#, NA, TOTR, Glass/Air thickness, and ABCD matrix.
    """
    def __init__(self, model=None, parent=None):
        super().__init__(parent)
        self.model = model
        self.last_system = None
        self.last_wavelengths = None
        self.last_system_config = None
        self.cached_report = ""

        if QT_AVAILABLE:
            ctrl_layout = QHBoxLayout()
            ctrl_layout.setContentsMargins(6, 2, 6, 2)
            ctrl_layout.setSpacing(8)

            lbl_title = QLabel("📑 <b>Paraxial Cardinal Diagnostics</b>")
            lbl_title.setStyleSheet("color: #1e293b; font-size: 11px;")
            ctrl_layout.addWidget(lbl_title)

            self.lbl_badge = QLabel("EFFL: -- mm | BFL: -- mm | EXPD: -- mm | P1/P2: -- mm")
            self.lbl_badge.setStyleSheet(
                "background-color: #f1f5f9; color: #0284c7; font-weight: bold; font-size: 11px; "
                "border: 1px solid #cbd5e1; border-radius: 4px; padding: 2px 8px;"
            )
            ctrl_layout.addWidget(self.lbl_badge)

            self.combo_mode = QComboBox()
            self.combo_mode.addItems([
                "📐 Cardinal Planes, Stops & Windows",
                "⚖️ Helmholtz-Lagrange Invariant & Delano"
            ])
            self.combo_mode.setStyleSheet(
                "QComboBox { background-color: #f8fafc; border: 1px solid #cbd5e1; border-radius: 3px; "
                "padding: 2px 6px; font-size: 11px; font-weight: bold; color: #1e293b; }"
            )
            self.combo_mode.currentIndexChanged.connect(self._replot)
            ctrl_layout.addWidget(self.combo_mode)

            self.combo_wl = QComboBox()
            self.combo_wl.setToolTip("Select evaluation wavelength for paraxial cardinal points, stops, and Helmholtz-Lagrange invariant")
            self.combo_wl.setStyleSheet(
                "QComboBox { background-color: #f8fafc; border: 1px solid #cbd5e1; border-radius: 3px; "
                "padding: 2px 6px; font-size: 11px; font-weight: bold; color: #1e293b; }"
            )
            self.combo_wl.currentIndexChanged.connect(self._replot)
            ctrl_layout.addWidget(self.combo_wl)

            ctrl_layout.addStretch(1)

            btn_copy = QPushButton("📋 Copy Report")
            btn_copy.setToolTip("Copy complete Zemax First Order prescription report to clipboard")
            btn_copy.setStyleSheet(
                "QPushButton { background-color: #f8fafc; border: 1px solid #cbd5e1; border-radius: 3px; "
                "padding: 2px 8px; font-size: 11px; font-weight: bold; } QPushButton:hover { background-color: #e2e8f0; }"
            )
            btn_copy.clicked.connect(self._copy_to_clipboard)
            ctrl_layout.addWidget(btn_copy)

            btn_refresh = QPushButton("🔄 Refresh")
            btn_refresh.setToolTip("Recalculate first-order optical diagnostics")
            btn_refresh.setStyleSheet(
                "QPushButton { background-color: #f8fafc; border: 1px solid #cbd5e1; border-radius: 3px; "
                "padding: 2px 8px; font-size: 11px; font-weight: bold; } QPushButton:hover { background-color: #e2e8f0; }"
            )
            btn_refresh.clicked.connect(self._replot)
            ctrl_layout.addWidget(btn_refresh)

            layout = self.layout()
            if layout is not None:
                idx = 1 if hasattr(self, 'toolbar') and self.toolbar else 0
                layout.insertLayout(idx, ctrl_layout)

    def _copy_to_clipboard(self):
        if not QT_AVAILABLE or not self.cached_report:
            return
        cb = QtWidgets.QApplication.clipboard()
        if cb:
            cb.setText(self.cached_report)

    def _replot(self):
        self.update_plot(
            system=self.last_system,
            wavelengths=self.last_wavelengths,
            system_config=self.last_system_config
        )

    def set_dark_mode(self, enabled: bool) -> None:
        self.dark_mode = bool(enabled)
        if QT_AVAILABLE and hasattr(self, 'lbl_badge') and self.lbl_badge:
            if self.dark_mode:
                self.lbl_badge.setStyleSheet(
                    "background-color: #1e293b; color: #38bdf8; font-weight: bold; font-size: 11px; "
                    "border: 1px solid #334155; border-radius: 4px; padding: 2px 8px;"
                )
            else:
                self.lbl_badge.setStyleSheet(
                    "background-color: #f1f5f9; color: #0284c7; font-weight: bold; font-size: 11px; "
                    "border: 1px solid #cbd5e1; border-radius: 4px; padding: 2px 8px;"
                )
        if QT_AVAILABLE and hasattr(self, 'combo_mode') and self.combo_mode:
            if self.dark_mode:
                self.combo_mode.setStyleSheet(
                    "QComboBox { background-color: #1e293b; border: 1px solid #334155; border-radius: 3px; "
                    "padding: 2px 6px; font-size: 11px; font-weight: bold; color: #f1f5f9; }"
                )
            else:
                self.combo_mode.setStyleSheet(
                    "QComboBox { background-color: #f8fafc; border: 1px solid #cbd5e1; border-radius: 3px; "
                    "padding: 2px 6px; font-size: 11px; font-weight: bold; color: #1e293b; }"
                )
        if QT_AVAILABLE and hasattr(self, 'combo_wl') and self.combo_wl:
            if self.dark_mode:
                self.combo_wl.setStyleSheet(
                    "QComboBox { background-color: #1e293b; border: 1px solid #334155; border-radius: 3px; "
                    "padding: 2px 6px; font-size: 11px; font-weight: bold; color: #f1f5f9; }"
                )
            else:
                self.combo_wl.setStyleSheet(
                    "QComboBox { background-color: #f8fafc; border: 1px solid #cbd5e1; border-radius: 3px; "
                    "padding: 2px 6px; font-size: 11px; font-weight: bold; color: #1e293b; }"
                )
        self._replot()

    def update_plot(
        self,
        system=None,
        wavelengths=None,
        system_config=None,
        **kwargs
    ):
        if not hasattr(self, 'fig') or self.fig is None:
            return
    
        theme = self.get_theme_colors()
        self.last_system = system
        self.last_wavelengths = wavelengths
        self.last_system_config = system_config
    
        stack = None
        if system is not None and hasattr(system, "stack"):
            stack = system.stack
        elif self.model is not None and hasattr(self.model, "get_surface_stack"):
            try:
                stack = self.model.get_surface_stack()
            except Exception:
                pass
    
        if stack is None or not getattr(stack, "surfaces", []):
            self.fig.clf()
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, "No optical surfaces defined in system", ha="center", va="center", color=theme["text_dim"], fontsize=11)
            ax.axis("off")
            self.apply_dark_mode_to_axes(ax)
            self.canvas.draw_idle()
            return
    
        # Available wavelengths detection
        available_wls = []
        primary_wl = None
        if system_config and hasattr(system_config, "wavelengths") and system_config.wavelengths:
            for w in system_config.wavelengths:
                w_val = float(getattr(w, "wavelength_um", 0.5876))
                if w_val not in available_wls:
                    available_wls.append(w_val)
                if getattr(w, "is_primary", False):
                    primary_wl = w_val
        elif hasattr(stack, "wavelengths") and stack.wavelengths:
            for w in stack.wavelengths:
                w_val = float(w)
                if w_val not in available_wls:
                    available_wls.append(w_val)
            if hasattr(stack, "primary_wavelength") and stack.primary_wavelength:
                primary_wl = float(stack.primary_wavelength)
        elif wavelengths and len(wavelengths) > 0:
            available_wls = [float(w) for w in wavelengths]

        if not available_wls:
            available_wls = [0.5876]
        if primary_wl is None:
            primary_wl = available_wls[0]

        # Populate / sync combo_wl
        if QT_AVAILABLE and hasattr(self, 'combo_wl') and self.combo_wl:
            current_chosen = self.combo_wl.currentData()
            existing_data = [self.combo_wl.itemData(i) for i in range(self.combo_wl.count())]
            if existing_data != available_wls:
                self.combo_wl.blockSignals(True)
                self.combo_wl.clear()
                sel_idx = 0
                for idx, w_val in enumerate(available_wls):
                    is_p = abs(w_val - primary_wl) < 1e-5
                    lbl = f"λ = {w_val:.3f} µm (Primary)" if is_p else f"λ = {w_val:.3f} µm"
                    self.combo_wl.addItem(lbl, userData=w_val)
                    if current_chosen is not None and abs(float(current_chosen) - w_val) < 1e-5:
                        sel_idx = idx
                    elif current_chosen is None and is_p:
                        sel_idx = idx
                self.combo_wl.setCurrentIndex(sel_idx)
                self.combo_wl.blockSignals(False)

        wl = primary_wl
        if QT_AVAILABLE and hasattr(self, 'combo_wl') and self.combo_wl and self.combo_wl.count() > 0:
            chosen_data = self.combo_wl.currentData()
            if chosen_data is not None:
                wl = float(chosen_data)
        elif wavelengths and len(wavelengths) > 0:
            wl = float(wavelengths[0])

        data = calculate_first_order_data(stack.surfaces, wavelength_um=wl, config=system_config)

        # Precompute multi-wavelength data if multiple wavelengths exist
        data_by_wl = {}
        if len(available_wls) > 1:
            for w_item in available_wls:
                if abs(w_item - wl) < 1e-5:
                    data_by_wl[w_item] = data
                else:
                    try:
                        data_by_wl[w_item] = calculate_first_order_data(stack.surfaces, wavelength_um=w_item, config=system_config)
                    except Exception:
                        pass
        from optics_app.engine.matrix_tracer import RayMatrix
        self.cached_report = format_zemax_first_order_report(data, title=getattr(stack, "name", "Optical System"))
    
        effl = data.get("effl", float("inf"))
        bfl = data.get("bfl", float("inf"))
        ffl = data.get("ffl", float("inf"))
        p1 = data.get("p1", 0.0)
        p2 = data.get("p2", 0.0)
        inter_p = data.get("inter_p", 0.0)
        n1 = data.get("n1", 0.0)
        n2 = data.get("n2", 0.0)
        epd = data.get("epd", 0.0)
        enp_pos = data.get("enp_pos", 0.0)
        expd = data.get("expd", 0.0)
        exp_pos = data.get("exp_pos", 0.0)
        exp_to_img = data.get("exp_to_image", 0.0)
        pupil_mag = data.get("pupil_magnification", 1.0)
        pwr_dpt = data.get("optical_power_diopters", 0.0)
        f_num = data.get("f_number", float("inf"))
        wf_num = data.get("working_f_number", float("inf"))
        na = data.get("na", 0.0)
        totr = data.get("totr", 0.0)
        t_glass = data.get("lens_thickness", 0.0)
        t_air = data.get("air_thickness", 0.0)
        is_afocal = data.get("is_afocal", False)
        gp = data.get("global_positions", {})
        mat = data.get("matrix", RayMatrix())
        diagnostics = data.get("paraxial_diagnostics", [])
        sw = data.get("stops_and_windows", {})
        hl = data.get("helmholtz_lagrange", {})
    
        as_info = sw.get("aperture_stop", {})
        fs_info = sw.get("field_stop", {})
        enp_info = sw.get("entrance_pupil", {})
        exp_info = sw.get("exit_pupil", {})
        enw_info = sw.get("entrance_window", {})
        exw_info = sw.get("exit_window", {})
    
        H_nom = hl.get("lagrange_invariant_H", 0.0)
        max_err = hl.get("max_conservation_error", 0.0)
        is_conserved = hl.get("is_conserved", True)
        etendue_G = hl.get("etendue_G", 0.0)
        m_trans = hl.get("transverse_magnification", 0.0)
        m_ang = hl.get("angular_magnification", float("inf"))
        sm_prod = hl.get("smith_helmholtz_product", 1.0)
        sm_target = hl.get("smith_helmholtz_target", 1.0)
        surface_steps = hl.get("surface_steps", [])
        delano = hl.get("delano_diagram", {})
        cardinal_inv = hl.get("cardinal_invariants", {})
        abbe_comp = hl.get("non_paraxial_abbe_comparison", [])
    
        mode = self.combo_mode.currentIndex() if hasattr(self, 'combo_mode') and self.combo_mode else 0
        vig_str = f" | Vig: {data.get('system_vignetting_factor', 0.0)*100:.1f}%" if data.get("is_vignetted") else " | Vig: 0%"
        is_p_active = abs(wl - primary_wl) < 1e-5
        wl_tag = f"[λ={wl:.3f} µm{'*' if is_p_active else ''}] "

        if hasattr(self, 'lbl_badge'):
            if mode == 0:
                if is_afocal:
                    self.lbl_badge.setText(f"{wl_tag}EFFL: Afocal | EPD: {epd:.1f} mm | EXPD: {expd:.1f} mm | TOTR: {totr:.1f} mm{vig_str}")
                else:
                    self.lbl_badge.setText(f"{wl_tag}EFFL: {effl:.2f} mm | BFL: {bfl:.2f} mm | EXPD: {expd:.1f} mm | P1: {p1:+.2f} mm | P2: {p2:+.2f} mm | Power: {pwr_dpt:+.2f} dpt{vig_str}")
            else:
                self.lbl_badge.setText(
                    f"{wl_tag}Lagrange H: {H_nom:+.4e} mm·rad | Etendue G: {etendue_G:.4e} mm²·sr | ΔH/H: {max_err:.1e} (Conserved: {is_conserved}) | Delano Area: {delano.get('polygon_area', 0.0):.2f} mm²"
                )
    
        self.fig.clf()
    
        if mode == 0:
            # =========================================================================
            # Mode 0: 📐 Cardinal Planes, Stops & Windows
            # =========================================================================
            ax_schematic = self.fig.add_subplot(2, 1, 1)
            ax_table = self.fig.add_subplot(2, 1, 2)
    
            z_v1 = gp.get("v1_z", 0.0)
            z_v2 = gp.get("v2_z", z_v1 + totr)
            z_img = gp.get("image_z", z_v2)
            z_f2 = gp.get("f2_z", z_v2 + bfl) if not is_afocal else z_img + 20.0
            z_f1 = gp.get("f1_z", z_v1 - ffl) if not is_afocal else z_v1 - 20.0
            z_p1 = gp.get("p1_z", z_v1 + p1)
            z_p2 = gp.get("p2_z", z_v2 + p2)
            z_enp = enp_info.get("z_global", z_v1 + enp_pos)
            z_exp = exp_info.get("z_global", z_v2 + exp_pos)
            z_as = as_info.get("z_global", z_v1)
            z_fs = fs_info.get("z_global", z_img)
            z_enw = enw_info.get("z_global", z_v1)
            z_exw = exw_info.get("z_global", z_v2)
    
            z_all = [z_v1, z_v2, z_img, z_p1, z_p2, z_enp, z_exp, z_as, z_fs]
            if not is_afocal:
                z_all.extend([z_f1, z_f2])
            for zw in [z_enw, z_exw]:
                if abs(zw) < 1e4 and not math.isinf(zw):
                    z_all.append(zw)
            z_min = min(z_all) - 8.0
            z_max = max(z_all) + 8.0
    
            ax_schematic.axhline(0.0, color="#94a3b8", linestyle="-", linewidth=0.8, alpha=0.7)
    
            # Draw surfaces and glass elements
            z_curr = z_v1
            semi_max = max(10.0, epd / 2.0, expd / 2.0)
            opt_surfs = [s for s in stack.surfaces if s.surface_type != SurfaceType.OBJECT or (s.glass and str(s.glass).upper() not in ("AIR", ""))]
            if not opt_surfs:
                opt_surfs = stack.surfaces
    
            for i, s in enumerate(opt_surfs):
                sd = float(getattr(s, "semi_diameter", 12.5))
                if sd <= 0:
                    sd = 12.5
                semi_max = max(semi_max, sd)
                r_surf = float(getattr(s, "radius", 0.0))
                t_surf = max(0.0, float(getattr(s, "thickness", 0.0)))
                g_surf = str(getattr(s, "glass", "AIR") or "AIR").upper()
    
                y_pts = np.linspace(-sd, sd, 41)
                if abs(r_surf) > 1e-4:
                    rad_sq = r_surf**2
                    valid_mask = rad_sq >= y_pts**2
                    sag = np.zeros_like(y_pts)
                    sag[valid_mask] = r_surf - np.sign(r_surf) * np.sqrt(rad_sq - y_pts[valid_mask]**2)
                else:
                    sag = np.zeros_like(y_pts)
    
                z_profile = z_curr + sag
                is_mirror = g_surf in ("MIRROR", "MIRR")
                s_color = ("#94a3b8" if self.dark_mode else "#334155") if not is_mirror else "#e11d48"
                ax_schematic.plot(z_profile, y_pts, color=s_color, linewidth=1.5)
    
                if g_surf not in ("AIR", "NONE", "MIRROR", "MIRR", "") and t_surf > 0 and i < len(opt_surfs) - 1:
                    next_s = opt_surfs[i + 1]
                    next_r = float(getattr(next_s, "radius", 0.0))
                    next_sd = float(getattr(next_s, "semi_diameter", sd))
                    y_sh = np.linspace(-min(sd, next_sd), min(sd, next_sd), 31)
                    sag_front = (r_surf - np.sign(r_surf) * np.sqrt(np.maximum(0.0, r_surf**2 - y_sh**2))) if abs(r_surf) > 1e-4 else np.zeros_like(y_sh)
                    sag_rear = (next_r - np.sign(next_r) * np.sqrt(np.maximum(0.0, next_r**2 - y_sh**2))) if abs(next_r) > 1e-4 else np.zeros_like(y_sh)
                    z_front = z_curr + sag_front
                    z_back = z_curr + t_surf + sag_rear
                    ax_schematic.fill_betweenx(y_sh, z_front, z_back, color="#38bdf8", alpha=0.25 if self.dark_mode else 0.18)
    
                z_curr += t_surf
    
            # Cardinal Lines
            ax_schematic.axvline(z_p1, color="#38bdf8" if self.dark_mode else "#2563eb", linestyle="--", linewidth=1.6, label=f"P1 (Front Principal): {p1:+.2f} mm")
            ax_schematic.axvline(z_p2, color="#f87171" if self.dark_mode else "#dc2626", linestyle="--", linewidth=1.6, label=f"P2 (Rear Principal): {p2:+.2f} mm")
            if not is_afocal and abs(z_f2) < 1e5:
                ax_schematic.axvline(z_f2, color="#4ade80" if self.dark_mode else "#16a34a", linestyle=":", linewidth=1.6, label=f"F2 (Back Focus): {bfl:.2f} mm")
            if not is_afocal and abs(z_f1) < 1e5:
                ax_schematic.axvline(z_f1, color="#2dd4bf" if self.dark_mode else "#0d9488", linestyle=":", linewidth=1.4, label=f"F1 (Front Focus): {ffl:.2f} mm")
    
            # Aperture Stop (AS)
            as_sd = float(as_info.get("semi_diameter", 8.0))
            as_idx = as_info.get("surface_index", 0)
            as_col = "#ef4444" if self.dark_mode else "#b91c1c"
            ax_schematic.plot([z_as, z_as], [as_sd, as_sd * 1.3], color=as_col, linewidth=3.5)
            ax_schematic.plot([z_as, z_as], [-as_sd * 1.3, -as_sd], color=as_col, linewidth=3.5)
            ax_schematic.plot([z_as, z_as], [-as_sd, as_sd], color=as_col, linestyle=":", linewidth=1.2, label=f"AS (Aperture Stop S{as_idx}): Ø {2*as_sd:.1f} mm")
    
            # Field Stop (FS)
            fs_sd = float(fs_info.get("semi_diameter", 10.0))
            fs_idx = fs_info.get("surface_index", 0)
            fs_col = "#f472b6" if self.dark_mode else "#c026d3"
            ax_schematic.plot([z_fs, z_fs], [fs_sd, fs_sd * 1.3], color=fs_col, linewidth=3.5)
            ax_schematic.plot([z_fs, z_fs], [-fs_sd * 1.3, -fs_sd], color=fs_col, linewidth=3.5)
            ax_schematic.plot([z_fs, z_fs], [-fs_sd, fs_sd], color=fs_col, linestyle=":", linewidth=1.2, label=f"FS (Field Stop S{fs_idx}): Ø {2*fs_sd:.1f} mm")
    
            # Pupils (Vertical aperture discs)
            enp_half = max(1.0, float(enp_info.get("semi_diameter", epd / 2.0)))
            exp_half = max(1.0, float(exp_info.get("semi_diameter", expd / 2.0)))
            ax_schematic.plot([z_enp, z_enp], [enp_half * 0.9, enp_half * 1.1], color="#eab308", linewidth=3.0)
            ax_schematic.plot([z_enp, z_enp], [-enp_half * 1.1, -enp_half * 0.9], color="#eab308", linewidth=3.0)
            ax_schematic.plot([z_enp, z_enp], [-enp_half, enp_half], color="#eab308", linestyle="-.", linewidth=1.3, label=f"EnP (Entrance Pupil): Ø {2*enp_half:.1f} mm")
    
            ax_schematic.plot([z_exp, z_exp], [exp_half * 0.9, exp_half * 1.1], color="#c084fc" if self.dark_mode else "#9333ea", linewidth=3.0)
            ax_schematic.plot([z_exp, z_exp], [-exp_half * 1.1, -exp_half * 0.9], color="#c084fc" if self.dark_mode else "#9333ea", linewidth=3.0)
            ax_schematic.plot([z_exp, z_exp], [-exp_half, exp_half], color="#c084fc" if self.dark_mode else "#9333ea", linestyle="-.", linewidth=1.3, label=f"ExP (Exit Pupil): Ø {2*exp_half:.1f} mm")
    
            # Windows (Vertical aperture discs)
            enw_d = float(enw_info.get("diameter", 20.0))
            exw_d = float(exw_info.get("diameter", 20.0))
            if abs(z_enw) < 1e3 and not math.isinf(z_enw):
                enw_half = enw_d / 2.0
                ax_schematic.plot([z_enw, z_enw], [-enw_half, enw_half], color="#06b6d4", linestyle="--", linewidth=1.2, alpha=0.9, label=f"EnW (Entrance Window): Ø {enw_d:.1f} mm")
            if abs(z_exw) < 1e3 and not math.isinf(z_exw):
                exw_half = exw_d / 2.0
                ax_schematic.plot([z_exw, z_exw], [-exw_half, exw_half], color="#10b981", linestyle="--", linewidth=1.2, alpha=0.9, label=f"ExW (Exit Window): Ø {exw_d:.1f} mm")
    
            # Paraxial Marginal & Chief Rays
            paraxial_rays = data.get("paraxial_rays", {})
            traj_m = paraxial_rays.get("marginal_trajectory", {})
            traj_c = paraxial_rays.get("chief_trajectory", {})
    
            if traj_m and traj_m.get("z") and len(traj_m["z"]) > 1:
                z_pts = traj_m["z"]
                ym_pts = traj_m["y"]
                yc_pts = traj_c.get("y", [0.0] * len(z_pts))
    
                m_col = "#fb923c" if self.dark_mode else "#ea580c"
                c_col = "#22d3ee" if self.dark_mode else "#0891b2"
                ax_schematic.plot(z_pts, ym_pts, color=m_col, linewidth=1.6, label="Marginal Ray (+ym)")
                ax_schematic.plot(z_pts, [-y for y in ym_pts], color=m_col, linewidth=1.2, linestyle="--", alpha=0.7, label="Marginal Ray (-ym)")
                ax_schematic.plot(z_pts, yc_pts, color=c_col, linewidth=1.6, label="Chief Ray (+yc)")
                ax_schematic.plot(z_pts, [-y for y in yc_pts], color=c_col, linewidth=1.2, linestyle="--", alpha=0.7, label="Chief Ray (-yc)")
    
                # Beam Envelope Fill
                env_up = [yc + abs(ym) for ym, yc in zip(ym_pts, yc_pts)]
                env_dn = [yc - abs(ym) for ym, yc in zip(ym_pts, yc_pts)]
                ax_schematic.fill_between(z_pts, env_dn, env_up, color=m_col, alpha=0.10)
    
            ax_schematic.set_xlim(z_min, z_max)
            ax_schematic.set_ylim(-semi_max * 1.45, semi_max * 1.45)
            ax_schematic.set_title(f"Paraxial Cardinal Planes, Pupils, Windows & Stops (OpticStudio Standard) [λ = {wl:.3f} µm]", fontsize=10, fontweight="bold")
            ax_schematic.set_xlabel("Optical Axis Z (mm)", fontsize=8)
            ax_schematic.set_ylabel("Height Y (mm)", fontsize=8)
            ax_schematic.grid(True, linestyle=":", alpha=0.5)
            ax_schematic.legend(loc="upper right", fontsize=7, ncol=3, framealpha=0.92)
            self.apply_dark_mode_to_axes(ax_schematic)
    
            # Subplot 2: Prescription Table & Diagnostic Cards (Mode 0)
            ax_table.axis("off")
            effl_txt = "Afocal" if is_afocal else f"{effl:.3f} mm"
            bfl_txt = "Afocal" if is_afocal else f"{bfl:.3f} mm"
            ffl_txt = "Afocal" if is_afocal else f"{ffl:.3f} mm"
            fnum_txt = "Afocal" if is_afocal else f"{f_num:.2f}"
            wfnum_txt = "Afocal" if is_afocal else f"{wf_num:.2f}"
            pwr_txt = "0.00 dpt" if is_afocal else f"{pwr_dpt:+.2f} dpt"
    
            col1_title = "FOCAL & POWER PROPERTIES"
            col1_lines = [
                f"Effective Focal (EFFL):  {effl_txt}",
                f"Back Focal Length (BFL): {bfl_txt}",
                f"Front Focal Length (FFL):{ffl_txt}",
                f"Optical Power (Phi):     {pwr_txt}",
                f"Image F-Number (F/#):    {fnum_txt}",
                f"Working F-Number (W F/#):{wfnum_txt}",
                f"Numerical Aperture (NA): {na:.4f}"
            ]
    
            col2_title = "PRINCIPAL & NODAL PLANES"
            col2_lines = [
                f"Front Principal (P1):  {p1:+.3f} mm (Z={z_p1:.2f} mm)",
                f"Rear Principal (P2):   {p2:+.3f} mm (Z={z_p2:.2f} mm)",
                f"Principal Separation:  Delta P = {inter_p:.3f} mm",
                f"Front Nodal Point (N1):{n1:+.3f} mm",
                f"Rear Nodal Point (N2): {n2:+.3f} mm",
                f"Front Focus (F1):      Z = {z_f1:.2f} mm",
                f"Rear Focus (F2):       Z = {z_f2:.2f} mm"
            ]
    
            col3_title = "PUPILS & APERTURE STOP"
            col3_lines = [
                f"Aperture Stop (AS):    S{as_idx} ({as_info.get('name','Stop')}) dia={2*as_sd:.2f}mm",
                f"AS Position:           Z = {z_as:.2f} mm",
                f"Entrance Pupil (EnP):  dia {2*enp_half:.2f} mm @ Z={z_enp:.2f} mm",
                f"Exit Pupil (ExP):      dia {2*exp_half:.2f} mm @ Z={z_exp:.2f} mm",
                f"ExP to Image Distance: {exp_to_img:.3f} mm",
                f"Pupil Magnification:   mp = {pupil_mag:.3f}",
                f"Total Track (TOTR):    {totr:.2f} mm"
            ]
    
            col4_title = "WINDOWS & FIELD STOP"
            col4_lines = [
                f"Field Stop (FS):       S{fs_idx} ({fs_info.get('name','FS')}) dia={2*fs_sd:.2f}mm",
                f"FS Position:           Z = {z_fs:.2f} mm",
                f"Entrance Window (EnW): dia {enw_d:.2f} mm @ Z={z_enw:.2f} mm",
                f"Exit Window (ExW):     dia {exw_d:.2f} mm @ Z={z_exw:.2f} mm",
                f"Max Unvignetted FOV:   {fs_info.get('max_unvignetted_field_deg', 0.0):.2f} deg (semi)",
                f"Subtended EnW FOV:     {enw_info.get('angular_fov_deg', 0.0):.2f} deg (full)",
                f"Lagrange Invariant H:  {H_nom:.4e} mm*rad"
            ]
    
            text_block = []
            text_block.append(f"{col1_title:<32} {col2_title:<34} {col3_title:<34} {col4_title}")
            text_block.append("-" * 138)
            for r in range(max(len(col1_lines), len(col2_lines), len(col3_lines), len(col4_lines))):
                c1 = col1_lines[r] if r < len(col1_lines) else ""
                c2 = col2_lines[r] if r < len(col2_lines) else ""
                c3 = col3_lines[r] if r < len(col3_lines) else ""
                c4 = col4_lines[r] if r < len(col4_lines) else ""
                text_block.append(f"{c1:<32} {c2:<34} {c3:<34} {c4}")
            text_block.append("-" * 138)
            text_block.append(
                f"ABCD Matrix: [A={mat.A:9.5f}, B={mat.B:9.5f}, C={mat.C:10.6f}, D={mat.D:9.5f}] | Det={mat.determinant():.6f} | Etendue G = {etendue_G:.4e} mm^2*sr"
            )
            if diagnostics:
                diag_str = " | ".join(diagnostics[:2])
                text_block.append(f"Diagnostics: {diag_str}")
    
            full_table_text = "\n".join(text_block)
            bbox_props = self.get_text_bbox(pad=0.5, alpha=0.92)
            ax_table.text(
                0.5, 0.5, full_table_text,
                transform=ax_table.transAxes,
                ha="center", va="center",
                family="monospace", fontsize=7.2,
                color=theme["text"],
                bbox=bbox_props
            )
            self.apply_dark_mode_to_axes(ax_table)
    
        else:
            # =========================================================================
            # Mode 1: ⚖️ Helmholtz-Lagrange Invariant & Delano
            # =========================================================================
            ax_hl = self.fig.add_subplot(2, 2, 1)
            ax_delano = self.fig.add_subplot(2, 2, 2)
            ax_table = self.fig.add_subplot(2, 1, 2)
    
            # Subplot 1 (Top Left): Invariant H(z) along Optical Axis
            z_steps = [s.get("z_global", 0.0) for s in surface_steps]
            Hin_vals = [s.get("H_in", H_nom) for s in surface_steps]
            Hout_vals = [s.get("H_out", H_nom) for s in surface_steps]
    
            if z_steps:
                z_plot_min = min(z_steps) - 5.0
                z_plot_max = max(z_steps) + 5.0
            else:
                z_plot_min, z_plot_max = -10.0, 50.0
    
            # Nominal H line
            ax_hl.axhline(H_nom, color="#10b981" if not self.dark_mode else "#34d399", linewidth=2.0, linestyle="-", label=f"Nominal H = {H_nom:+.4e} mm·rad")
            ax_hl.fill_between([z_plot_min, z_plot_max], [H_nom * 0.9999, H_nom * 0.9999], [H_nom * 1.0001, H_nom * 1.0001], color="#10b981", alpha=0.15)
    
            # Surface invariant points
            if z_steps:
                ax_hl.plot(z_steps, Hin_vals, marker="o", markersize=5, linestyle=":", color="#38bdf8", label="Surface Hin")
                ax_hl.plot(z_steps, Hout_vals, marker="x", markersize=5, linestyle="--", color="#f59e0b", label="Surface Hout")
    
            # Cardinal plane points
            c_markers = {
                "object_plane": ("Object", "o", "#38bdf8"),
                "entrance_pupil": ("EnP", "s", "#eab308"),
                "aperture_stop": ("Stop", "D", "#ef4444"),
                "exit_pupil": ("ExP", "*", "#c084fc"),
                "field_stop": ("FS", "^", "#ec4899"),
                "image_plane": ("Image", "P", "#10b981")
            }
            for c_key, c_val in cardinal_inv.items():
                cz = c_val.get("z", 0.0)
                ch = c_val.get("H", H_nom)
                if not math.isinf(cz) and abs(cz) < 1e5:
                    c_lbl, c_mk, c_cl = c_markers.get(c_key, (c_key, "o", "#94a3b8"))
                    ax_hl.plot([cz], [ch if abs(ch) > 1e-12 else H_nom], marker=c_mk, markersize=7, color=c_cl, markeredgecolor="white", markeredgewidth=1.0, zorder=8, label=f"Cardinal {c_lbl}")
    
            ax_hl.set_title(f"Helmholtz-Lagrange Invariant H(z) [λ={wl:.3f} µm, ΔH/H={max_err:.1e}]", fontsize=9.5, fontweight="bold")
            ax_hl.set_xlabel("Optical Axis Z (mm)", fontsize=8)
            ax_hl.set_ylabel("Invariant H = n(yc·um - ym·uc) (mm·rad)", fontsize=8)
            ax_hl.grid(True, linestyle=":", alpha=0.5)
            ax_hl.legend(loc="best", fontsize=6.8, ncol=2, framealpha=0.90)
            self.apply_dark_mode_to_axes(ax_hl)
    
            # Subplot 2 (Top Right): Delano y-ȳ Diagram (ym vs yc)
            del_ym = delano.get("ym", [])
            del_yc = delano.get("yc", [])
            del_lbls = delano.get("labels", [])
            poly_area = delano.get("polygon_area", 0.0)
    
            if del_ym and del_yc:
                ax_delano.plot(del_ym, del_yc, marker="o", markersize=5, color="#0284c7" if not self.dark_mode else "#38bdf8", linewidth=2.0, label=f"Delano (λ={wl:.3f} µm)")
                ax_delano.fill(del_ym, del_yc, color="#38bdf8", alpha=0.20)

                # Overlap other wavelengths if available to visualize chromatic shift
                chromatic_cols = ["#ec4899", "#10b981", "#f59e0b", "#a855f7", "#06b6d4"]
                col_i = 0
                for w_item, d_item in data_by_wl.items():
                    if abs(w_item - wl) > 1e-5:
                        del_other = d_item.get("helmholtz_lagrange", {}).get("delano_diagram", {})
                        oym = del_other.get("ym", [])
                        oyc = del_other.get("yc", [])
                        if oym and oyc:
                            c_col = chromatic_cols[col_i % len(chromatic_cols)]
                            col_i += 1
                            ax_delano.plot(oym, oyc, linestyle="--", linewidth=1.3, color=c_col, alpha=0.85, label=f"λ={w_item:.3f} µm")
    
                for x_pt, y_pt, lbl_pt in zip(del_ym, del_yc, del_lbls):
                    ax_delano.annotate(
                        f" {lbl_pt}",
                        xy=(x_pt, y_pt),
                        xytext=(4, 4),
                        textcoords="offset points",
                        fontsize=7.5,
                        fontweight="bold",
                        color=theme["text"]
                    )
    
            # Conjugate reference axes
            ax_delano.axvline(0.0, color="#eab308", linestyle="--", linewidth=1.2, label="Pupil Axis (ym=0)")
            ax_delano.axhline(0.0, color="#10b981", linestyle="--", linewidth=1.2, label="Image/Obj Axis (yc=0)")
    
            ax_delano.set_title(f"Delano y-ȳ Diagram [Area={poly_area:.3f} mm² = 0.5·|H|/n]", fontsize=9.5, fontweight="bold")
            ax_delano.set_xlabel("Marginal Ray Height ym (mm)", fontsize=8)
            ax_delano.set_ylabel("Chief Ray Height yc (mm)", fontsize=8)
            ax_delano.grid(True, linestyle=":", alpha=0.5)
            ax_delano.legend(loc="upper right", fontsize=7, framealpha=0.90)
            self.apply_dark_mode_to_axes(ax_delano)
    
            # Subplot 3 (Bottom): Surface Invariant Table & Abbe Comparison Card
            ax_table.axis("off")
            lines_tbl = []
            lines_tbl.append("SURFACE-BY-SURFACE HELMHOLTZ-LAGRANGE OPTICAL INVARIANT CONSERVATION TABLE:")
            lines_tbl.append(f"{'Surf':<5} {'Name':<12} {'Z (mm)':<10} {'y_m (mm)':<11} {'u_m (rad)':<12} {'y_c (mm)':<11} {'u_c (rad)':<12} {'n_in':<8} {'H_in (mm·rad)':<16} {'H_out':<16} {'ΔH':<10}")
            lines_tbl.append("-" * 138)
    
            for step in surface_steps:
                s_i = step.get("surface_index", 0)
                s_n = step.get("name", f"S{s_i}")[:11]
                s_z = step.get("z_global", 0.0)
                s_ym = step.get("y_marginal", 0.0)
                s_um = step.get("u_marginal_in", 0.0)
                s_yc = step.get("y_chief", 0.0)
                s_uc = step.get("u_chief_in", 0.0)
                s_nin = step.get("n_in", 1.0)
                s_hin = step.get("H_in", 0.0)
                s_hout = step.get("H_out", 0.0)
                s_dh = step.get("delta_H", 0.0)
                lines_tbl.append(
                    f"S{s_i:<4} {s_n:<12} {s_z:<10.2f} {s_ym:<11.4f} {s_um:<12.5f} {s_yc:<11.4f} {s_uc:<12.5f} {s_nin:<8.4f} {s_hin:<16.6e} {s_hout:<16.6e} {s_dh:<10.2e}"
                )
            lines_tbl.append("-" * 138)

            if len(available_wls) > 1 and data_by_wl:
                lines_tbl.append("HELMHOLTZ-LAGRANGE CHROMATIC CONSERVATION SUMMARY ACROSS WAVELENGTHS:")
                lines_tbl.append(f"{'Wavelength':<22} {'Lagrange H (mm·rad)':<24} {'EnP Dia (mm)':<16} {'ExP Dia (mm)':<16} {'Max |ΔH/H|':<16} {'Conservation'}")
                lines_tbl.append("-" * 138)
                for w_val in available_wls:
                    d_w = data_by_wl.get(w_val)
                    if d_w:
                        hl_w = d_w.get("helmholtz_lagrange", {})
                        h_val = hl_w.get("lagrange_invariant_H", 0.0)
                        err_w = hl_w.get("max_conservation_error", 0.0)
                        sw_w = d_w.get("stops_and_windows", {})
                        enp_d_w = sw_w.get("entrance_pupil", {}).get("diameter", 0.0)
                        exp_d_w = sw_w.get("exit_pupil", {}).get("diameter", 0.0)
                        is_p = abs(w_val - primary_wl) < 1e-5
                        w_tag = f"λ = {w_val:.3f} µm {'(Primary)' if is_p else '         '}"
                        lines_tbl.append(f"{w_tag:<22} {h_val:<24.6e} {enp_d_w:<16.3f} {exp_d_w:<16.3f} {err_w:<16.2e} {'EXACT' if err_w < 1e-4 else 'APPROX'}")
                lines_tbl.append("-" * 138)
    
            # Smith-Helmholtz Identity & Etendue Summary
            lines_tbl.append(
                f"SMITH-HELMHOLTZ THEOREM: Transverse Mag m = {m_trans:+.4f} | Angular Mag m_α = {m_ang:+.4f} | Product m·m_α = {sm_prod:.6f} (Target n_obj/n_img = {sm_target:.6f}) -> EXACT"
            )
            lines_tbl.append(
                f"OPTICAL ETENDUE: G = π² · H² = {etendue_G:.6e} mm²·sr | Paraxial Conservation Residual: Max |ΔH/H| = {max_err:.2e} (100% Invariance)"
            )
            lines_tbl.append("-" * 138)
    
            # Abbe Sine Condition Zone Comparison
            if abbe_comp:
                lines_tbl.append("NON-PARAXIAL ABBE SINE INVARIANT COMPARISON [H_paraxial = n·y·u vs Abbe = n·y·sin(u)]:")
                comp_strs = []
                for z_item in abbe_comp:
                    zn = z_item.get("pupil_zone", 0.0)
                    ph = z_item.get("paraxial_H", 0.0)
                    ah = z_item.get("abbe_sine_invariant", 0.0)
                    disc = z_item.get("discrepancy_pct", 0.0)
                    comp_strs.append(f"Zone {zn:.2f} (NA={zn*na:.3f}): Paraxial={ph:.5f} | Abbe={ah:.5f} | Diff={disc:.2f}%")
                lines_tbl.append("  |  ".join(comp_strs))
    
            full_hl_text = "\n".join(lines_tbl)
            bbox_props = self.get_text_bbox(pad=0.5, alpha=0.92)
            ax_table.text(
                0.5, 0.5, full_hl_text,
                transform=ax_table.transAxes,
                ha="center", va="center",
                family="monospace", fontsize=7.2,
                color=theme["text"],
                bbox=bbox_props
            )
            self.apply_dark_mode_to_axes(ax_table)
    
        self.apply_tight_layout()
        if hasattr(self, "canvas") and self.canvas:
            self.canvas.draw_idle()
    


class ManufacturingAuditTab(BaseAnalysisTab):
    """
    Dedicated ISO 10110 / Zemax Manufacturing & Mounting Audit Tab.
    Displays:
    - 2D Cross-section of lens elements with optical clear apertures, mechanical mounting rims, and chamfers.
    - Color-coded element compliance highlighting (Blue = compliant, Amber = warning, Red = violation).
    - Element-by-element fabrication table:
      - Center thickness tc (MNCG / MXCG)
      - Clear & mechanical edge thickness tedge (MNEG / MNEG_MECH)
      - Aspect ratio D / tc (ISO 10110 limit <= 10)
      - Max ray incidence angle theta_inc (RAID <= 55 deg)
      - Standard tooling test plate radius R1 / R2 and fringe power error (TFLT)
    - Air gap spacer ring seating clearances (MNCA / MNEA)
    - Overall system compliance score and critical violations / advisories.
    - Interactive "Fit to Tooling Test Plates" and "Copy Report" actions.
    """
    def __init__(self, model=None, parent=None):
        super().__init__(parent)
        self.model = model
        self.last_system = None
        self.last_wavelengths = None
        self.last_system_config = None
        self.cached_report = ""
        self.current_preset = QualityPreset.COMMERCIAL

        if QT_AVAILABLE:
            ctrl_layout = QHBoxLayout()
            ctrl_layout.setContentsMargins(6, 2, 6, 2)
            ctrl_layout.setSpacing(8)

            lbl_title = QLabel("🏭 <b>ISO 10110 / Zemax Manufacturing Audit</b>")
            lbl_title.setStyleSheet("color: #1e293b; font-size: 11px;")
            ctrl_layout.addWidget(lbl_title)

            lbl_preset = QLabel("Preset:")
            lbl_preset.setStyleSheet("font-size: 11px; font-weight: bold; color: #475569;")
            ctrl_layout.addWidget(lbl_preset)

            self.combo_preset = QComboBox()
            self.combo_preset.addItems([
                "Commercial (ISO 10110 Standard)",
                "Precision (Research / Laser)",
                "High-Precision (Space / Litho)"
            ])
            self.combo_preset.setStyleSheet(
                "QComboBox { background-color: #f1f5f9; border: 1px solid #cbd5e1; border-radius: 3px; "
                "padding: 2px 6px; font-size: 11px; font-weight: bold; color: #1e293b; }"
            )
            self.combo_preset.currentTextChanged.connect(self._on_preset_changed)
            ctrl_layout.addWidget(self.combo_preset)

            self.lbl_badge = QLabel("Compliance: --% | Status: --")
            self.lbl_badge.setStyleSheet(
                "background-color: #f1f5f9; color: #0284c7; font-weight: bold; font-size: 11px; "
                "border: 1px solid #cbd5e1; border-radius: 4px; padding: 2px 8px;"
            )
            ctrl_layout.addWidget(self.lbl_badge)

            ctrl_layout.addStretch(1)

            btn_fit = QPushButton("Tooling Test Plates (TFLT)")
            btn_fit.setToolTip("Snap lens surface radii to standard ISO/Thorlabs/Edmund tooling test plates and re-optimize air spaces")
            btn_fit.setStyleSheet(
                "QPushButton { background-color: #eff6ff; border: 1px solid #bfdbfe; color: #1d4ed8; border-radius: 3px; "
                "padding: 2px 8px; font-size: 11px; font-weight: bold; } QPushButton:hover { background-color: #dbeafe; }"
            )
            btn_fit.clicked.connect(self._fit_test_plates)
            ctrl_layout.addWidget(btn_fit)

            btn_copy = QPushButton("📋 Copy Report")
            btn_copy.setToolTip("Copy complete ISO 10110 manufacturing and mounting audit report to clipboard")
            btn_copy.setStyleSheet(
                "QPushButton { background-color: #f8fafc; border: 1px solid #cbd5e1; border-radius: 3px; "
                "padding: 2px 8px; font-size: 11px; font-weight: bold; } QPushButton:hover { background-color: #e2e8f0; }"
            )
            btn_copy.clicked.connect(self._copy_to_clipboard)
            ctrl_layout.addWidget(btn_copy)

            btn_refresh = QPushButton("🔄 Refresh")
            btn_refresh.setToolTip("Recalculate manufacturing and mounting audit")
            btn_refresh.setStyleSheet(
                "QPushButton { background-color: #f8fafc; border: 1px solid #cbd5e1; border-radius: 3px; "
                "padding: 2px 8px; font-size: 11px; font-weight: bold; } QPushButton:hover { background-color: #e2e8f0; }"
            )
            btn_refresh.clicked.connect(self._replot)
            ctrl_layout.addWidget(btn_refresh)

            layout = self.layout()
            if layout is not None:
                idx = 1 if hasattr(self, 'toolbar') and self.toolbar else 0
                layout.insertLayout(idx, ctrl_layout)

    def _on_preset_changed(self, text: str):
        if "High-Precision" in text:
            self.current_preset = QualityPreset.HIGH_PRECISION
        elif "Precision" in text:
            self.current_preset = QualityPreset.PRECISION
        else:
            self.current_preset = QualityPreset.COMMERCIAL
        self._replot()

    def _copy_to_clipboard(self):
        if not QT_AVAILABLE or not self.cached_report:
            return
        try:
            from optics_app.gui.qt_compat import QtWidgets
            cb = QtWidgets.QApplication.clipboard()
            if cb:
                cb.setText(self.cached_report)
        except Exception:
            pass

    def _fit_test_plates(self):
        stack = None
        if self.model is not None and hasattr(self.model, "get_surface_stack"):
            stack = self.model.get_surface_stack()
        elif self.last_system is not None and hasattr(self.last_system, "stack"):
            stack = self.last_system.stack
        if not stack or not getattr(stack, "surfaces", []):
            return

        from optics_app.engine.optimizer import OpticalOptimizer
        opt = OpticalOptimizer(OpticalSystem(stack), wavelengths=self.last_wavelengths or [0.5876])
        res = opt.fit_to_test_plates(re_optimize_bfl=True)
        if res and res.optimized_stack:
            if self.model is not None and hasattr(self.model, "set_surface_stack"):
                if hasattr(self.model, "_push_undo"):
                    self.model._push_undo("Fit to Test Plates")
                self.model.set_surface_stack(res.optimized_stack)
                if hasattr(self.model, "system_updated"):
                    self.model.system_updated.emit()
            elif self.last_system is not None:
                self.last_system.stack = res.optimized_stack
            self.update_plot(system=OpticalSystem(res.optimized_stack), wavelengths=self.last_wavelengths, system_config=self.last_system_config)

    def set_dark_mode(self, enabled: bool) -> None:
        self.dark_mode = bool(enabled)
        if QT_AVAILABLE:
            if hasattr(self, "combo_preset") and self.combo_preset:
                if self.dark_mode:
                    self.combo_preset.setStyleSheet(
                        "QComboBox { background-color: #1e293b; border: 1px solid #334155; border-radius: 3px; "
                        "padding: 2px 6px; font-size: 11px; font-weight: bold; color: #f1f5f9; }"
                    )
                else:
                    self.combo_preset.setStyleSheet(
                        "QComboBox { background-color: #f1f5f9; border: 1px solid #cbd5e1; border-radius: 3px; "
                        "padding: 2px 6px; font-size: 11px; font-weight: bold; color: #1e293b; }"
                    )
        self._replot()

    def _replot(self):
        self.update_plot(
            system=self.last_system,
            wavelengths=self.last_wavelengths,
            system_config=self.last_system_config
        )

    def update_plot(
        self,
        system: Optional[OpticalSystem] = None,
        wavelengths: Optional[List[float]] = None,
        system_config: Optional[Any] = None,
        **kwargs
    ):
        if not (QT_AVAILABLE and HAS_MPL_QT) or self.fig is None:
            return

        theme = self.get_theme_colors()
        self.last_system = system
        self.last_wavelengths = wavelengths
        self.last_system_config = system_config

        stack = None
        if system is not None and hasattr(system, "stack"):
            stack = system.stack
        elif self.model is not None and hasattr(self.model, "get_surface_stack"):
            try:
                stack = self.model.get_surface_stack()
            except Exception:
                pass

        if stack is None or not getattr(stack, "surfaces", []):
            self.fig.clf()
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, "No optical surfaces defined in system", ha="center", va="center", color=theme["text_dim"], fontsize=11)
            ax.axis("off")
            self.apply_dark_mode_to_axes(ax)
            self.canvas.draw_idle()
            return

        surfaces = stack.surfaces
        num_surfs = len(surfaces)

        # 1. Collect ray incidence angles if system has them
        ray_angles = {}
        if system is not None:
            try:
                p_wl = wavelengths[0] if (wavelengths and len(wavelengths) > 0) else 0.5876
                traced = system.trace_marginal_and_chief_rays(wl=p_wl)
                for r in traced:
                    for h in getattr(r, 'history', []):
                        s_idx = h.get("surface_index")
                        th = h.get("theta_inc", 0.0)
                        if s_idx is not None:
                            deg = math.degrees(th) if th < 3.14159 else th
                            ray_angles[s_idx] = max(ray_angles.get(s_idx, 0.0), deg)
            except Exception:
                pass

        # 2. Run Audit
        cfg = ManufacturingConstraintsConfig.from_preset(self.current_preset)
        report = audit_optical_system_manufacturing(surfaces, config=cfg, ray_incidence_angles=ray_angles)
        self.cached_report = report.format_markdown()

        # Update Badge
        if hasattr(self, 'lbl_badge') and self.lbl_badge:
            score = report.compliance_score
            if report.is_compliant:
                self.lbl_badge.setText(f"Status: PASS ({score:.0f}%) | TOTR: {report.total_track_mm:.1f} mm")
                if self.dark_mode:
                    self.lbl_badge.setStyleSheet(
                        "background-color: #064e3b; color: #34d399; font-weight: bold; font-size: 11px; "
                        "border: 1px solid #059669; border-radius: 4px; padding: 2px 8px;"
                    )
                else:
                    self.lbl_badge.setStyleSheet(
                        "background-color: #ecfdf5; color: #059669; font-weight: bold; font-size: 11px; "
                        "border: 1px solid #a7f3d0; border-radius: 4px; padding: 2px 8px;"
                    )
            else:
                self.lbl_badge.setText(f"Status: NON-COMPLIANT ({score:.0f}%) | TOTR: {report.total_track_mm:.1f} mm")
                if self.dark_mode:
                    self.lbl_badge.setStyleSheet(
                        "background-color: #450a0a; color: #f87171; font-weight: bold; font-size: 11px; "
                        "border: 1px solid #dc2626; border-radius: 4px; padding: 2px 8px;"
                    )
                else:
                    self.lbl_badge.setStyleSheet(
                        "background-color: #fef2f2; color: #dc2626; font-weight: bold; font-size: 11px; "
                        "border: 1px solid #fecaca; border-radius: 4px; padding: 2px 8px;"
                    )

        # 3. Plotting
        self.fig.clf()
        from matplotlib.patches import Polygon as MplPolygon
        import math
        gs = self.fig.add_gridspec(2, 1, height_ratios=[1.1, 1.0], hspace=0.35)
        ax_cross = self.fig.add_subplot(gs[0])
        ax_table = self.fig.add_subplot(gs[1])

        # Global Z positions
        z_pos = [0.0]
        for s in surfaces[:-1]:
            z_pos.append(z_pos[-1] + float(getattr(s, "thickness", 0.0)))

        max_sd = 10.0
        # Draw Optical Axis
        z_min = min(z_pos) - 5.0
        z_max = max(z_pos) + 10.0
        ax_cross.plot([z_min, z_max], [0, 0], color="#94a3b8", linestyle="--", linewidth=1.0, zorder=1)

        # Draw Elements
        for el in report.elements:
            i_f = el.surface_front_index
            i_r = el.surface_rear_index
            s_f = surfaces[i_f]
            s_r = surfaces[i_r]
            zf0 = z_pos[i_f]
            zr0 = z_pos[i_r]

            sd_clr = el.diameter_clear_mm / 2.0
            sd_mch = el.diameter_mechanical_mm / 2.0
            max_sd = max(max_sd, sd_mch)

            # Sample profile
            y_pts = np.linspace(-sd_clr, sd_clr, 40)
            zf_pts = [zf0 + s_f.sag(y) for y in y_pts]
            zr_pts = [zr0 + s_r.sag(y) for y in y_pts]

            # Construct closed contour: Front -> Lower Rim -> Rear -> Upper Rim
            poly_pts = []
            # 1. Front surface top to bottom
            for y, z in zip(reversed(y_pts), reversed(zf_pts)):
                poly_pts.append((z, y))
            # 2. Lower mechanical rim flat
            zf_bot_clr = zf0 + s_f.sag(sd_clr)
            zr_bot_clr = zr0 + s_r.sag(sd_clr)
            poly_pts.append((zf_bot_clr, -sd_mch))
            poly_pts.append((zr_bot_clr, -sd_mch))
            # 3. Rear surface bottom to top
            for y, z in zip(y_pts, zr_pts):
                poly_pts.append((z, y))
            # 4. Upper mechanical rim flat
            poly_pts.append((zr_bot_clr, sd_mch))
            poly_pts.append((zf_bot_clr, sd_mch))

            # Color styling based on audit status
            if self.dark_mode:
                if el.errors:
                    fc, ec, lw = "#450a0a", "#ef4444", 1.6
                elif el.warnings:
                    fc, ec, lw = "#451a03", "#f59e0b", 1.3
                else:
                    fc, ec, lw = "#0c4a6e", "#38bdf8", 1.3
            else:
                if el.errors:
                    fc, ec, lw = "#fee2e2", "#dc2626", 1.6
                elif el.warnings:
                    fc, ec, lw = "#fef3c7", "#d97706", 1.3
                else:
                    fc, ec, lw = "#e0f2fe", "#0284c7", 1.3

            poly = MplPolygon(poly_pts, closed=True, facecolor=fc, edgecolor=ec, linewidth=lw, alpha=0.85, zorder=2)
            ax_cross.add_patch(poly)

            # Optical clear aperture boundary line (dashed)
            ax_cross.plot([zf_bot_clr, zr_bot_clr], [sd_clr, sd_clr], color="#38bdf8" if self.dark_mode else "#0284c7", linestyle=":", linewidth=0.8, zorder=3)
            ax_cross.plot([zf_bot_clr, zr_bot_clr], [-sd_clr, -sd_clr], color="#38bdf8" if self.dark_mode else "#0284c7", linestyle=":", linewidth=0.8, zorder=3)

            # Element Label
            z_mid = 0.5 * (zf0 + zr0)
            status_symbol = "OK" if el.is_compliant else ("WARN" if el.warnings and not el.errors else "FAIL")
            ax_cross.text(
                z_mid, sd_mch + 1.2,
                f"E{el.element_index}: {el.glass}\n[{status_symbol}]",
                ha="center", va="bottom", fontsize=7.5, fontweight="bold",
                color="#ef4444" if el.errors else ("#f59e0b" if el.warnings else theme["text"])
            )

            # Thickness annotations
            ax_cross.text(z_mid, 0.2, f"tc={el.center_thickness_mm:.1f}", ha="center", va="bottom", fontsize=7, color=theme["text"])
            ax_cross.text(zr_bot_clr + 0.3, sd_clr, f"te={el.edge_thickness_clear_mm:.1f}", ha="left", va="center", fontsize=6.5, color=theme["text_dim"])

        # Detector Plane
        z_det = z_pos[-1]
        det_sd = max_sd * 0.9
        ax_cross.plot([z_det, z_det], [-det_sd, det_sd], color="#f43f5e" if self.dark_mode else "#e11d48", linewidth=2.0, linestyle="-", zorder=3)
        ax_cross.text(z_det, det_sd + 1.0, "Detector", ha="center", va="bottom", fontsize=8, color="#f43f5e" if self.dark_mode else "#e11d48", fontweight="bold")

        ax_cross.set_title(f"Optical & Mechanical Cross-Section (Clear Apertures, Mechanical Rims & Spacers) — {self.current_preset.value}", fontsize=9, fontweight="bold", pad=8)
        ax_cross.set_xlabel("Z Position (mm)", fontsize=8)
        ax_cross.set_ylabel("Height Y (mm)", fontsize=8)
        ax_cross.grid(True, linestyle=":", alpha=0.5)
        ax_cross.set_aspect("equal", adjustable="datalim")
        ax_cross.set_ylim(-max_sd * 1.5, max_sd * 1.6)
        self.apply_dark_mode_to_axes(ax_cross)

        # 4. Monospace Audit Table in ax_table
        ax_table.axis("off")
        lines = []
        hdr = f"{'Elem':<5} | {'Glass':<8} | {'Diam(Clr/Mch)':<15} | {'tc (mm)':<8} | {'tedge(Clr/Mch)':<17} | {'D/tc':<6} | {'Max RAID':<9} | {'Tooling R1 / R2':<22} | {'Status':<6}"
        sep = "-" * len(hdr)
        lines.append(hdr)
        lines.append(sep)

        for el in report.elements:
            stat = "PASS" if el.is_compliant else ("WARN" if el.warnings and not el.errors else "FAIL")
            diam_str = f"{el.diameter_clear_mm:.1f} / {el.diameter_mechanical_mm:.1f}"
            tedge_str = f"{el.edge_thickness_clear_mm:.2f} / {el.edge_thickness_mechanical_mm:.2f}"
            r1_fr = f"{el.r1_test_plate_mm:.1f}({el.r1_fringes:.1f}fr)" if abs(el.r1_test_plate_mm) > 0 else "PLANO"
            r2_fr = f"{el.r2_test_plate_mm:.1f}({el.r2_fringes:.1f}fr)" if abs(el.r2_test_plate_mm) > 0 else "PLANO"
            tool_str = f"{r1_fr} / {r2_fr}"
            lines.append(
                f"#{el.element_index:<4} | {el.glass:<8} | {diam_str:<15} | {el.center_thickness_mm:<8.2f} | "
                f"{tedge_str:<17} | {el.aspect_ratio:<6.1f} | {el.max_incidence_angle_deg:<8.1f}° | "
                f"{tool_str:<22} | [{stat}]"
            )

        if report.air_spacings:
            lines.append(sep)
            for air in report.air_spacings:
                lines.append(f"Air Gap #{air['gap_index']}: Center tc = {air['center_thickness']:.2f} mm, Edge Clearance = {air['edge_clearance']:.2f} mm -> [{air['status']}]")

        summary_footer = f"Preset: {report.preset_name} | Overall Compliance: {report.compliance_score:.1f}% | Total Track: {report.total_track_mm:.2f} mm"
        lines.append(sep)
        lines.append(summary_footer)

        if report.overall_errors:
            lines.append("CRITICAL MANUFACTURING VIOLATIONS:")
            for err in report.overall_errors[:4]:
                lines.append(f" ❌ {err}")

        if report.overall_warnings:
            lines.append("MANUFACTURING ADVISORIES:")
            for w in report.overall_warnings[:3]:
                lines.append(f" ⚠️ {w}")

        bbox_props = self.get_text_bbox(pad=0.5, alpha=0.92)
        ax_table.text(
            0.5, 0.5, "\n".join(lines),
            transform=ax_table.transAxes,
            ha="center", va="center",
            family="monospace", fontsize=7.2,
            color=theme["text"],
            bbox=bbox_props
        )
        self.apply_dark_mode_to_axes(ax_table)

        self.apply_tight_layout()
        if hasattr(self, "canvas") and self.canvas:
            self.canvas.draw_idle()


class SpectrometerQualityTab(BaseAnalysisTab):
    """
    Dedicated Zemax OpticStudio-Style Spectrometer Quality & Resolution Analysis Tab.
    ================================================================================
    Provides multi-mode interactive physical analysis:
    - Mode 0: 📈 Spectral Resolution & Resolving Power Breakdown (Slit, Diff, Aber, Pixel limits)
    - Mode 1: 💡 Light Throughput & Transmission Components (Geometric, Grating, NA, Det collection)
    - Mode 2: 🌈 Slit Footprint, Smile & Keystone Distortion (2D detector plane footprint & curvature)
    - Mode 3: 📊 Line Spread Function (LSF) & Synthetic Spectrum (Instrument-broadened doublet test)
    - Mode 4: 📑 Zemax Spectrometer ASCII Prescription Report (Complete OpticStudio text report)
    """

    def __init__(self, model=None, parent=None):
        super().__init__(parent)
        self.model = model
        self.last_system = None
        self.last_wavelengths = None
        self.last_system_config = None
        self.cached_summary: Optional[SpectrometerQualitySummary] = None
        self.cached_report: str = ""

        if QT_AVAILABLE:
            # Control layout 1: Optical parameters
            ctrl1 = QHBoxLayout()
            ctrl1.setContentsMargins(6, 2, 6, 2)
            ctrl1.setSpacing(6)

            lbl_title = QLabel("🔬 <b>Spectrometer Quality (Zemax)</b>")
            lbl_title.setStyleSheet("color: #1e293b; font-size: 11px;")
            ctrl1.addWidget(lbl_title)

            ctrl1.addWidget(QLabel("Slit W:"))
            self.spin_slit_w = QDoubleSpinBox()
            self.spin_slit_w.setRange(1.0, 2000.0)
            self.spin_slit_w.setValue(25.0)
            self.spin_slit_w.setSingleStep(5.0)
            self.spin_slit_w.setSuffix(" µm")
            self.spin_slit_w.setToolTip("Entrance slit physical width in micrometers")
            self.spin_slit_w.valueChanged.connect(self._replot)
            ctrl1.addWidget(self.spin_slit_w)

            ctrl1.addWidget(QLabel("Slit H:"))
            self.spin_slit_h = QDoubleSpinBox()
            self.spin_slit_h.setRange(0.1, 20.0)
            self.spin_slit_h.setValue(1.0)
            self.spin_slit_h.setSingleStep(0.5)
            self.spin_slit_h.setSuffix(" mm")
            self.spin_slit_h.setToolTip("Entrance slit height in millimeters")
            self.spin_slit_h.valueChanged.connect(self._replot)
            ctrl1.addWidget(self.spin_slit_h)

            ctrl1.addWidget(QLabel("NA:"))
            self.spin_na = QDoubleSpinBox()
            self.spin_na.setRange(0.01, 0.95)
            self.spin_na.setValue(0.12)
            self.spin_na.setSingleStep(0.01)
            self.spin_na.setDecimals(3)
            self.spin_na.setToolTip("Source / Input beam numerical aperture (NA)")
            self.spin_na.valueChanged.connect(self._replot)
            ctrl1.addWidget(self.spin_na)

            ctrl1.addWidget(QLabel("Pixel:"))
            self.spin_pixel = QDoubleSpinBox()
            self.spin_pixel.setRange(1.0, 500.0)
            self.spin_pixel.setValue(14.0)
            self.spin_pixel.setSingleStep(1.0)
            self.spin_pixel.setSuffix(" µm")
            self.spin_pixel.setToolTip("Detector pixel pitch along dispersion axis")
            self.spin_pixel.valueChanged.connect(self._replot)
            ctrl1.addWidget(self.spin_pixel)

            ctrl1.addWidget(QLabel("Laser λ₀:"))
            self.spin_laser = QDoubleSpinBox()
            self.spin_laser.setRange(0.0, 3.0)
            self.spin_laser.setValue(0.0)
            self.spin_laser.setSingleStep(0.01)
            self.spin_laser.setDecimals(3)
            self.spin_laser.setSuffix(" µm")
            self.spin_laser.setToolTip("Excitation laser wavelength for Raman shift conversion (0.0 = Disabled)")
            self.spin_laser.valueChanged.connect(self._replot)
            ctrl1.addWidget(self.spin_laser)

            ctrl1.addStretch(1)

            # Control layout 2: Modes & Action buttons
            ctrl2 = QHBoxLayout()
            ctrl2.setContentsMargins(6, 2, 6, 2)
            ctrl2.setSpacing(6)

            self.combo_mode = QComboBox()
            self.combo_mode.addItems([
                "📈 Resolution & Resolving Power Breakdown",
                "💡 Light Throughput & Efficiency Components",
                "🌈 Slit Footprint, Smile & Keystone Distortion",
                "📊 Line Spread Function (LSF) & Synthetic Spectrum",
                "📑 Zemax Spectrometer ASCII Prescription Report"
            ])
            self.combo_mode.setStyleSheet(
                "QComboBox { background-color: #f8fafc; border: 1px solid #cbd5e1; border-radius: 3px; "
                "padding: 2px 6px; font-size: 11px; font-weight: bold; color: #1e293b; }"
            )
            self.combo_mode.currentIndexChanged.connect(self._replot)
            ctrl2.addWidget(self.combo_mode)

            self.lbl_badge = QLabel("Res: -- nm | R: -- | Recip Disp: -- nm/mm | TP: --% | Nyquist: --")
            self.lbl_badge.setStyleSheet(
                "background-color: #f1f5f9; color: #0284c7; font-weight: bold; font-size: 11px; "
                "border: 1px solid #cbd5e1; border-radius: 4px; padding: 2px 8px;"
            )
            ctrl2.addWidget(self.lbl_badge)

            ctrl2.addStretch(1)

            btn_copy = QPushButton("📋 Copy Report")
            btn_copy.setToolTip("Copy complete Zemax Spectrometer ASCII report to clipboard")
            btn_copy.setStyleSheet(
                "QPushButton { background-color: #f8fafc; border: 1px solid #cbd5e1; border-radius: 3px; "
                "padding: 2px 8px; font-size: 11px; font-weight: bold; } QPushButton:hover { background-color: #e2e8f0; }"
            )
            btn_copy.clicked.connect(self._copy_to_clipboard)
            ctrl2.addWidget(btn_copy)

            btn_export = QPushButton("💾 Export Report (.txt)")
            btn_export.setToolTip("Export formatted Zemax Spectrometer report to text file")
            btn_export.setStyleSheet(
                "QPushButton { background-color: #f8fafc; border: 1px solid #cbd5e1; border-radius: 3px; "
                "padding: 2px 8px; font-size: 11px; font-weight: bold; } QPushButton:hover { background-color: #e2e8f0; }"
            )
            btn_export.clicked.connect(self._export_report)
            ctrl2.addWidget(btn_export)

            btn_refresh = QPushButton("🔄 Refresh")
            btn_refresh.setToolTip("Recalculate 3D ray-traced spectrometer performance")
            btn_refresh.setStyleSheet(
                "QPushButton { background-color: #f8fafc; border: 1px solid #cbd5e1; border-radius: 3px; "
                "padding: 2px 8px; font-size: 11px; font-weight: bold; } QPushButton:hover { background-color: #e2e8f0; }"
            )
            btn_refresh.clicked.connect(self._replot)
            ctrl2.addWidget(btn_refresh)

            layout = self.layout()
            if layout is not None:
                idx = 1 if hasattr(self, 'toolbar') and self.toolbar else 0
                layout.insertLayout(idx, ctrl1)
                layout.insertLayout(idx + 1, ctrl2)

    def _copy_to_clipboard(self):
        if not QT_AVAILABLE or not self.cached_report:
            return
        from optics_app.gui.qt_compat import QtWidgets
        cb = QtWidgets.QApplication.clipboard()
        if cb:
            cb.setText(self.cached_report)

    def _export_report(self):
        if not QT_AVAILABLE or not self.cached_report:
            return
        fn, _ = QFileDialog.getSaveFileName(self, "Export Zemax Spectrometer Report", "spectrometer_quality_report.txt", "Text Files (*.txt);;All Files (*)")
        if fn:
            try:
                with open(fn, "w", encoding="utf-8") as f:
                    f.write(self.cached_report)
            except Exception:
                pass

    def _replot(self, *_args):
        self.update_plot(
            system=self.last_system,
            wavelengths=self.last_wavelengths,
            system_config=self.last_system_config
        )

    def set_dark_mode(self, enabled: bool) -> None:
        self.dark_mode = bool(enabled)
        theme = self.get_theme_colors()
        if QT_AVAILABLE and hasattr(self, 'lbl_badge') and self.lbl_badge:
            if self.dark_mode:
                self.lbl_badge.setStyleSheet(
                    "background-color: #1e293b; color: #38bdf8; font-weight: bold; font-size: 11px; "
                    "border: 1px solid #334155; border-radius: 4px; padding: 2px 8px;"
                )
            else:
                self.lbl_badge.setStyleSheet(
                    "background-color: #f1f5f9; color: #0284c7; font-weight: bold; font-size: 11px; "
                    "border: 1px solid #cbd5e1; border-radius: 4px; padding: 2px 8px;"
                )
        if QT_AVAILABLE and hasattr(self, 'combo_mode') and self.combo_mode:
            if self.dark_mode:
                self.combo_mode.setStyleSheet(
                    "QComboBox { background-color: #1e293b; border: 1px solid #334155; border-radius: 3px; "
                    "padding: 2px 6px; font-size: 11px; font-weight: bold; color: #f1f5f9; }"
                )
            else:
                self.combo_mode.setStyleSheet(
                    "QComboBox { background-color: #f8fafc; border: 1px solid #cbd5e1; border-radius: 3px; "
                    "padding: 2px 6px; font-size: 11px; font-weight: bold; color: #1e293b; }"
                )
        self._replot()

    def update_plot(self, system: Optional[OpticalSystem] = None, wavelengths: Optional[List[float]] = None, system_config: Optional[Any] = None):
        if not (QT_AVAILABLE and HAS_MPL_QT) or self.fig is None:
            return

        self.last_system = system
        self.last_wavelengths = wavelengths
        self.last_system_config = system_config
        theme = self.get_theme_colors()
        self.fig.clf()

        if system is None:
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, "No Optical System Loaded", ha="center", va="center", color=theme.get("fg_subtext", "#64748b"), fontsize=12)
            self.apply_dark_mode_to_axes(ax)
            if hasattr(self, "canvas") and self.canvas:
                self.canvas.draw_idle()
            return

        # Prepare Spectrometer Configuration
        slit_w = self.spin_slit_w.value() if hasattr(self, 'spin_slit_w') else 25.0
        slit_h = self.spin_slit_h.value() if hasattr(self, 'spin_slit_h') else 1.0
        na_val = self.spin_na.value() if hasattr(self, 'spin_na') else 0.12
        pix_val = self.spin_pixel.value() if hasattr(self, 'spin_pixel') else 14.0
        laser_val = self.spin_laser.value() if hasattr(self, 'spin_laser') else 0.0

        cfg = SpectrometerConfiguration(
            slit_width_um=float(slit_w),
            slit_height_mm=float(slit_h),
            input_na=float(na_val),
            pixel_pitch_um=float(pix_val),
            raman_laser_wavelength_um=float(laser_val) if laser_val > 0.0 else None
        )

        try:
            summary = SpectrometerQualityAnalyzer.analyze_system(system, config=cfg, wavelengths_um=wavelengths)
            self.cached_summary = summary
            self.cached_report = summary.zemax_report_text
        except Exception as e:
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, f"Analysis Error: {str(e)}", ha="center", va="center", color="#ef4444", fontsize=11)
            self.apply_dark_mode_to_axes(ax)
            if hasattr(self, "canvas") and self.canvas:
                self.canvas.draw_idle()
            return

        # Update status badge
        if hasattr(self, 'lbl_badge'):
            res_txt = f"Res: {summary.avg_resolution_nm:.2f} nm | R: {summary.avg_resolving_power:.0f} | Recip Disp: {summary.avg_reciprocal_dispersion_nm_per_mm:.1f} nm/mm | TP: {summary.avg_throughput_pct:.1f}% | Smile: {summary.max_smile_um:.1f} µm"
            nyq_badge = " | Nyquist: PASS" if summary.nyquist_compliant else " | Nyquist: WARN"
            self.lbl_badge.setText(res_txt + nyq_badge)

        mode = self.combo_mode.currentIndex() if hasattr(self, 'combo_mode') and self.combo_mode else 0
        wls_nm = [r.wavelength_nm for r in summary.wavelength_results]

        if mode == 0:
            # =========================================================================
            # Mode 0: 📈 Spectral Resolution & Resolving Power Breakdown
            # =========================================================================
            ax1 = self.fig.add_subplot(2, 1, 1)
            ax2 = self.fig.add_subplot(2, 1, 2)

            res_tot = [r.res_total_nm for r in summary.wavelength_results]
            res_slit = [r.res_slit_nm for r in summary.wavelength_results]
            res_diff = [r.res_diff_nm for r in summary.wavelength_results]
            res_aber = [r.res_aber_nm for r in summary.wavelength_results]
            res_pix = [r.res_pixel_nm for r in summary.wavelength_results]

            c_tot = '#38bdf8' if self.dark_mode else '#0284c7'
            c_slit = '#f97316' if self.dark_mode else '#ea580c'
            c_diff = '#a855f7' if self.dark_mode else '#7e22ce'
            c_aber = '#ef4444' if self.dark_mode else '#dc2626'
            c_pix = '#10b981' if self.dark_mode else '#059669'

            ax1.plot(wls_nm, res_tot, color=c_tot, marker='o', linewidth=2.2, label=f"Total Res Δλ (Mean: {summary.avg_resolution_nm:.2f} nm)")
            ax1.plot(wls_nm, res_slit, color=c_slit, linestyle='--', linewidth=1.4, label="Slit Image Limit Δλ_slit")
            ax1.plot(wls_nm, res_diff, color=c_diff, linestyle=':', linewidth=1.4, label="Diffraction Limit Δλ_diff")
            ax1.plot(wls_nm, res_aber, color=c_aber, linestyle='-.', linewidth=1.4, label="Aberration Limit Δλ_aber")
            ax1.plot(wls_nm, res_pix, color=c_pix, linestyle='-', linewidth=1.2, alpha=0.8, label="Pixel Sampling Limit Δλ_pixel")

            ax1.set_title("Spectral Resolution Components vs Wavelength (Zemax Methodology)", fontsize=10, fontweight='bold')
            ax1.set_ylabel("Spectral Bandpass Δλ (nm)", fontsize=9)
            ax1.legend(loc='best', fontsize=8)
            ax1.grid(True, alpha=0.25)

            # Subplot 2: Resolving Power R and Reciprocal Dispersion
            r_powers = [r.resolving_power for r in summary.wavelength_results]
            recip_disp = [r.reciprocal_dispersion_nm_per_mm for r in summary.wavelength_results]

            ax2.plot(wls_nm, r_powers, color='#4ade80' if self.dark_mode else '#16a34a', marker='s', linewidth=1.8, label=f"Resolving Power R = λ/Δλ (Mean: {summary.avg_resolving_power:.0f})")
            ax2.set_xlabel("Wavelength (nm)", fontsize=9)
            ax2.set_ylabel("Resolving Power (R)", fontsize=9, color='#16a34a')
            ax2.tick_params(axis='y', labelcolor='#16a34a')
            ax2.grid(True, alpha=0.25)

            ax2_twin = ax2.twinx()
            ax2_twin.plot(wls_nm, recip_disp, color='#fbbf24' if self.dark_mode else '#d97706', marker='^', linestyle='--', linewidth=1.5, label="Reciprocal Dispersion dλ/ds (nm/mm)")
            ax2_twin.set_ylabel("Reciprocal Dispersion (nm/mm)", fontsize=9, color='#d97706')
            ax2_twin.tick_params(axis='y', labelcolor='#d97706')

            ax2.set_title(f"Resolving Power & Reciprocal Linear Dispersion (Span: {summary.total_detector_span_mm:.2f} mm)", fontsize=9, fontweight='bold')
            self.apply_dark_mode_to_axes(ax1, ax2)

        elif mode == 1:
            # =========================================================================
            # Mode 1: 💡 Light Throughput & Efficiency Components
            # =========================================================================
            ax1 = self.fig.add_subplot(2, 1, 1)
            ax2 = self.fig.add_subplot(2, 1, 2)

            tp_tot = [r.total_throughput * 100.0 for r in summary.wavelength_results]
            tp_geom = [r.geometric_transmission * 100.0 for r in summary.wavelength_results]
            tp_grat = [r.grating_efficiency * 100.0 for r in summary.wavelength_results]
            tp_na = [r.na_acceptance_eff * 100.0 for r in summary.wavelength_results]
            tp_det = [r.detector_collection_eff * 100.0 for r in summary.wavelength_results]

            c_tot = '#38bdf8' if self.dark_mode else '#0284c7'
            ax1.plot(wls_nm, tp_tot, color=c_tot, marker='o', linewidth=2.5, label=f"Total Throughput (Peak: {summary.peak_throughput_pct:.1f}%)")
            ax1.fill_between(wls_nm, 0, tp_tot, color=c_tot, alpha=0.15)
            ax1.plot(wls_nm, tp_geom, color='#4ade80', linestyle='--', linewidth=1.5, label="Geometric Ray Transmission")
            ax1.plot(wls_nm, tp_grat, color='#fbbf24', linestyle=':', linewidth=1.5, label="Grating Blaze Efficiency")
            ax1.plot(wls_nm, tp_na, color='#a855f7', linestyle='-.', linewidth=1.2, label="NA / F# Matching Factor")
            ax1.plot(wls_nm, tp_det, color='#f87171', linestyle='-', linewidth=1.2, label="Detector Chip Collection")

            ax1.set_title(f"Spectrometer Light Throughput Breakdown (Étendue: {summary.optical_etendue_mm2_sr:.4e} mm²·sr)", fontsize=10, fontweight='bold')
            ax1.set_ylabel("Efficiency / Transmission (%)", fontsize=9)
            ax1.set_ylim(-2, 105)
            ax1.legend(loc='best', fontsize=8)
            ax1.grid(True, alpha=0.25)

            # Subplot 2: Relative Illumination across detector plane
            pos_mm = [r.dispersion_pos_mm for r in summary.wavelength_results]
            rel_illum = [t / max(1e-4, summary.peak_throughput_pct) * 100.0 for t in tp_tot]
            ax2.bar(pos_mm, rel_illum, width=max(0.02, summary.total_detector_span_mm / (len(pos_mm) * 2.0)), color='#38bdf8', alpha=0.7, edgecolor='#0284c7')
            ax2.set_xlabel("Detector Dispersion Coordinate s (mm)", fontsize=9)
            ax2.set_ylabel("Relative Illumination (%)", fontsize=9)
            ax2.set_title("Relative Illumination Profile Across Active Detector Span", fontsize=9, fontweight='bold')
            ax2.set_ylim(0, 110)
            ax2.grid(True, alpha=0.25)

            self.apply_dark_mode_to_axes(ax1, ax2)

        elif mode == 2:
            # =========================================================================
            # Mode 2: 🌈 Slit Footprint, Smile & Keystone Distortion
            # =========================================================================
            ax1 = self.fig.add_subplot(2, 1, 1)
            ax2 = self.fig.add_subplot(2, 1, 2)

            # Draw active detector boundaries
            hw = summary.config.detector_width_mm / 2.0
            hh = summary.config.detector_height_mm / 2.0
            rect_col = '#ef4444' if self.dark_mode else '#dc2626'
            ax1.plot([-hw, hw, hw, -hw, -hw], [-hh, -hh, hh, hh, -hh], color=rect_col, linestyle='--', linewidth=1.2, label=f"Active Detector ({hw*2:.1f}x{hh*2:.1f} mm)")

            # Plot centroids and slit footprints
            for r in summary.wavelength_results:
                ax1.scatter([r.centroid_x_mm], [r.centroid_y_mm], s=35, label=f"{r.wavelength_nm:.0f} nm")
                # Draw smile arc vector if present
                if r.smile_um > 0.1:
                    ax1.plot([r.centroid_x_mm, r.centroid_x_mm + r.smile_um / 1000.0], [r.centroid_y_mm, r.centroid_y_mm], color='#fbbf24', linewidth=1.5)

            ax1.set_title("2D Detector Slit Footprints & Geometric Smile Curvature", fontsize=10, fontweight='bold')
            ax1.set_xlabel("Detector X (mm)", fontsize=9)
            ax1.set_ylabel("Detector Y (mm)", fontsize=9)
            ax1.legend(loc='upper right', fontsize=7, ncol=2)
            ax1.grid(True, alpha=0.25)

            # Subplot 2: Smile Curvature and RMS Spot Size vs Wavelength
            smiles = [r.smile_um for r in summary.wavelength_results]
            rms_spots = [r.spot_rms_radius_um for r in summary.wavelength_results]
            airys = [r.airy_radius_um for r in summary.wavelength_results]

            ax2.plot(wls_nm, smiles, color='#fbbf24', marker='o', linewidth=1.8, label=f"Smile Distortion (Max: {summary.max_smile_um:.2f} µm)")
            ax2.plot(wls_nm, rms_spots, color='#38bdf8', marker='^', linestyle='--', linewidth=1.5, label="Centroid RMS Spot Radius (µm)")
            ax2.plot(wls_nm, airys, color='#a855f7', linestyle=':', linewidth=1.2, label="Diffraction Airy Radius (µm)")
            ax2.set_xlabel("Wavelength (nm)", fontsize=9)
            ax2.set_ylabel("Dimension (µm)", fontsize=9)
            ax2.set_title(f"Slit Smile & Spot Quality (Keystone: {summary.max_keystone_um:.2f} µm, Plane Tilt: {summary.focal_plane_tilt_deg:.1f}°)", fontsize=9, fontweight='bold')
            ax2.legend(loc='best', fontsize=8)
            ax2.grid(True, alpha=0.25)

            self.apply_dark_mode_to_axes(ax1, ax2)

        elif mode == 3:
            # =========================================================================
            # Mode 3: 📊 Line Spread Function (LSF) & Synthetic Spectrum
            # =========================================================================
            ax1 = self.fig.add_subplot(2, 1, 1)
            ax2 = self.fig.add_subplot(2, 1, 2)

            synth = summary.synthetic_spectrum
            if synth is not None and len(synth.wavelength_axis_nm) > 0:
                grid = synth.wavelength_axis_nm
                ax1.plot(grid, synth.lsf_profile, color='#38bdf8', linewidth=2.0, label="Normalized Instrument LSF")
                ax1.fill_between(grid, 0, synth.lsf_profile, color='#38bdf8', alpha=0.2)
                ax1.set_title(f"Instrument Response Function / Line Spread Function (FWHM ≈ {summary.avg_resolution_nm:.2f} nm)", fontsize=10, fontweight='bold')
                ax1.set_ylabel("Normalized Intensity", fontsize=9)
                ax1.set_ylim(-0.02, 1.08)
                ax1.legend(loc='upper right', fontsize=8)
                ax1.grid(True, alpha=0.25)

                # Synthetic Spectrum
                ax2.plot(grid, synth.synthetic_spectrum, color='#4ade80' if self.dark_mode else '#16a34a', linewidth=2.0, label="Simulated Recorded Spectrum")
                for p_wl, p_lbl in zip(synth.peak_wavelengths_nm, synth.peak_labels):
                    ax2.axvline(p_wl, color='#ef4444', linestyle=':', alpha=0.7)
                    ax2.text(p_wl, 0.95, f"{p_wl:.1f} nm", rotation=90, verticalalignment='top', fontsize=7, color='#ef4444')

                ax2.set_title("Simulated Emission Doublet & Calibration Spectrum (Resolving Quality Test)", fontsize=9, fontweight='bold')
                ax2.set_xlabel("Wavelength (nm)", fontsize=9)
                ax2.set_ylabel("Detected Irradiance (a.u.)", fontsize=9)
                ax2.legend(loc='upper right', fontsize=8)
                ax2.grid(True, alpha=0.25)
            else:
                ax1.text(0.5, 0.5, "No Spectrum Synthesized", ha="center", va="center")

            self.apply_dark_mode_to_axes(ax1, ax2)

        elif mode == 4:
            # =========================================================================
            # Mode 4: 📑 Zemax Spectrometer ASCII Prescription Report
            # =========================================================================
            ax_rep = self.fig.add_subplot(1, 1, 1)
            ax_rep.axis('off')
            bbox_props = self.get_text_bbox(pad=0.6, alpha=0.92)
            ax_rep.text(
                0.5, 0.5, summary.zemax_report_text,
                transform=ax_rep.transAxes,
                ha="center", va="center",
                family="monospace", fontsize=7.2,
                color=theme.get("fg_text", "#1e293b"),
                bbox=bbox_props
            )
            self.apply_dark_mode_to_axes(ax_rep)

        self.apply_tight_layout()
        if hasattr(self, "canvas") and self.canvas:
            self.canvas.draw_idle()


class AnalysisTabsWidget(QTabWidget if QT_AVAILABLE else object):
    """
    Main Analysis Tab Suite Container.
    Integrates Tab 1 (Spot Diagram), Tab 2 (Ray Fan), Tab 3 (Grating Dispersion),
    Tab 4 (CCD Heatmap), Tab 5 (Laser Scanning & Field Aberrations), and
    Tab 6 (Femtosecond Pulse Width & Dispersion).
    """

    def __init__(self, model: Optional[Any] = None, parent=None, auto_refresh: bool = True):
        if QT_AVAILABLE:
            super().__init__(parent)
        self.model = model
        self.tabs: List[str] = [
            "Spot Diagram",
            "Optical Diagnostics & First Order",
            "Manufacturing & Mounting Audit",
            "Best Focus & Aberrations",
            "Seidel Aberrations",
            "Ray Fan",
            "Grating Dispersion",
            "CCD Heatmap",
            "Laser Scanning & Field Aberrations",
            "Femtosecond Pulse",
            "Two-Photon Microscope",
            "Nonlinear Imaging",
            "Thorlabs SRS",
            "Achromatic Doublet Designer",
            "Spectrometer Quality & Resolution (Zemax)",
        ]
        self.active_tab: str = "Spot Diagram"
        self.last_wavelengths: Optional[List[float]] = None
        self.last_source_config: Optional[Any] = None
        self.two_photon_config = build_calibrated_template()
        self.nonlinear_imaging_config = build_nonlinear_surface_template().nonlinear_imaging_config
        self._two_photon_explicit = False
        self._nonlinear_explicit = False
        self.dark_mode: bool = False

        if QT_AVAILABLE and HAS_MPL_QT:
            self._init_tabs()
            self._init_surface_selector()

        if auto_refresh and self.model is not None and hasattr(self.model, "system_updated"):
            try:
                self.model.system_updated.connect(self.refresh_analysis)
            except Exception:
                pass

    def count(self) -> int:
        return len(self.tabs)

    def widget(self, index: int) -> Optional[Any]:
        if hasattr(self, '_all_widgets') and 0 <= index < len(self._all_widgets):
            return self._all_widgets[index]
        return None

    def tabText(self, index: int) -> str:
        if 0 <= index < len(self.tabs):
            return self.tabs[index]
        return ""

    def currentIndex(self) -> int:
        if hasattr(self, 'active_tab') and hasattr(self, 'tabs') and self.active_tab in self.tabs:
            return self.tabs.index(self.active_tab)
        return 0

    def setCurrentIndex(self, index: int) -> None:
        if hasattr(self, 'tabs') and 0 <= index < len(self.tabs):
            self.set_active_tab(self.tabs[index])
        elif hasattr(super(), 'setCurrentIndex'):
            super().setCurrentIndex(index)

    def indexOf(self, widget: Any) -> int:
        if hasattr(self, '_all_widgets') and widget in self._all_widgets:
            return self._all_widgets.index(widget)
        if hasattr(super(), 'indexOf'):
            return super().indexOf(widget)
        return -1

    def set_dark_mode(self, enabled: bool) -> None:
        """Propagate dark mode toggle to all analysis diagnostic tabs."""
        self.dark_mode = bool(enabled)
        for w in getattr(self, '_all_widgets', []):
            if w is not None and hasattr(w, "set_dark_mode"):
                try:
                    w.set_dark_mode(self.dark_mode)
                except Exception:
                    pass
        is_dark = self.dark_mode
        sub_style = (
            "QTabWidget::pane { border: 1px solid #334155; background: #0f172a; border-radius: 4px; }"
            "QTabBar::tab { background: #1e293b; color: #94a3b8; padding: 4px 10px; font-weight: bold; font-size: 11px; margin-right: 2px; border-top-left-radius: 3px; border-top-right-radius: 3px; }"
            "QTabBar::tab:selected { background: #0f172a; color: #38bdf8; border-bottom: 2px solid #38bdf8; }"
        ) if is_dark else (
            "QTabWidget::pane { border: 1px solid #cbd5e1; background: #ffffff; border-radius: 4px; }"
            "QTabBar::tab { background: #f1f5f9; color: #475569; padding: 4px 10px; font-weight: bold; font-size: 11px; margin-right: 2px; border-top-left-radius: 3px; border-top-right-radius: 3px; }"
            "QTabBar::tab:selected { background: #ffffff; color: #0284c7; border-bottom: 2px solid #0284c7; }"
        )
        if hasattr(self, '_sub_tab_widgets'):
            for sub in self._sub_tab_widgets.values():
                if hasattr(sub, 'setStyleSheet'):
                    try:
                        sub.setStyleSheet(sub_style)
                    except Exception:
                        pass

    def _init_tabs(self):
        self.tab_spot = SpotDiagramTab()
        self.tab_first_order = FirstOrderDiagnosticsTab(model=self.model)
        self.tab_mfg = ManufacturingAuditTab(model=self.model)
        self.tab_best_focus = BestFocusAnalysisTab()
        self.tab_seidel = SeidelAberrationTab()
        self.tab_rayfan = RayFanTab()
        self.tab_dispersion = GratingDispersionTab()
        self.tab_ccd = CCDHeatmapTab()
        self.tab_laser_scan = LaserScanAberrationTab()
        self.tab_pulse = PulseWidthAnalysisTab()
        self.tab_two_photon = TwoPhotonMicroscopeTab(model=self.model)
        self.tab_nonlinear = NonlinearImagingTab(model=self.model)
        self.tab_launcher = ThorlabsSRSLauncherTab(model=self.model)
        self.tab_achromat = AchromatAnalysisTab(model=self.model)
        self.tab_spectrometer = SpectrometerQualityTab(model=self.model)

        self._all_widgets = [
            self.tab_spot,
            self.tab_first_order,
            self.tab_mfg,
            self.tab_best_focus,
            self.tab_seidel,
            self.tab_rayfan,
            self.tab_dispersion,
            self.tab_ccd,
            self.tab_laser_scan,
            self.tab_pulse,
            self.tab_two_photon,
            self.tab_nonlinear,
            self.tab_launcher,
            self.tab_achromat,
            self.tab_spectrometer,
        ]

        # Categorized 3-tier structure requested by user
        self.category_names = [
            "🎯 Aberrations & Focus",
            "📊 Diagnostics & Standards",
            "🔬 Advanced Physics"
        ]
        self.categories = {
            "🎯 Aberrations & Focus": [
                ("Spot Diagram", self.tab_spot),
                ("Ray Fan", self.tab_rayfan),
                ("Seidel Aberrations", self.tab_seidel),
                ("Best Focus & Aberrations", self.tab_best_focus),
                ("Achromatic Doublet Designer", self.tab_achromat),
            ],
            "📊 Diagnostics & Standards": [
                ("Optical Diagnostics & First Order", self.tab_first_order),
                ("Manufacturing & Mounting Audit", self.tab_mfg),
                ("Spectrometer Quality & Resolution (Zemax)", self.tab_spectrometer),
                ("CCD Heatmap", self.tab_ccd),
                ("Grating Dispersion", self.tab_dispersion),
                ("Laser Scanning & Field Aberrations", self.tab_laser_scan),
            ],
            "🔬 Advanced Physics": [
                ("Femtosecond Pulse", self.tab_pulse),
                ("Two-Photon Microscope", self.tab_two_photon),
                ("Nonlinear Imaging", self.tab_nonlinear),
                ("Thorlabs SRS", self.tab_launcher),
            ],
        }

        self._sub_tab_widgets = {}
        for cat_name in self.category_names:
            sub_tab = QTabWidget()
            for tab_label, tab_widget in self.categories[cat_name]:
                sub_tab.addTab(tab_widget, tab_label)
            sub_tab.currentChanged.connect(lambda idx, st=sub_tab: self._on_sub_tab_changed(st, idx))
            self._sub_tab_widgets[cat_name] = sub_tab
            super().addTab(sub_tab, cat_name)

        if hasattr(self, "currentChanged"):
            self.currentChanged.connect(self._on_category_tab_changed)

    def _init_surface_selector(self):
        self.surface_combo = QComboBox(self)
        self.surface_combo.addItem("Detector (Final)", -1)
        self.surface_combo.currentIndexChanged.connect(self._on_surface_changed)
        self.setCornerWidget(self.surface_combo, Qt.TopRightCorner)

    def _on_surface_changed(self, index: int):
        self.refresh_analysis()

    def update_surface_combo(self, stack: Any):
        if not hasattr(self, 'surface_combo'):
            return
        self.surface_combo.blockSignals(True)
        current_data = self.surface_combo.currentData()
        self.surface_combo.clear()
        
        self.surface_combo.addItem("Detector (Final)", -1)
        
        if stack is not None and len(stack.surfaces) > 0:
            for i, surf in enumerate(stack.surfaces):
                name = getattr(surf, 'name', '') or f"Surface {i+1}"
                self.surface_combo.addItem(f"[{i}] {name}", i)
                
        # Try to restore selection
        idx = self.surface_combo.findData(current_data)
        if idx >= 0:
            self.surface_combo.setCurrentIndex(idx)
        else:
            self.surface_combo.setCurrentIndex(0)
            
        self.surface_combo.blockSignals(False)

    def _switch_to_tab_name(self, tab_name: str):
        if tab_name in self.tabs:
            self.active_tab = tab_name
            if self.active_tab == "Two-Photon Microscope":
                self._two_photon_explicit = True
            elif self.active_tab == "Nonlinear Imaging":
                self._nonlinear_explicit = True
            if hasattr(self, 'surface_combo'):
                self.surface_combo.setEnabled(self.active_tab != "Best Focus & Aberrations")
            self.refresh_analysis()

    def _on_category_tab_changed(self, cat_idx: int):
        if hasattr(self, 'category_names') and 0 <= cat_idx < len(self.category_names):
            cat_name = self.category_names[cat_idx]
            sub_tab = self._sub_tab_widgets.get(cat_name) if hasattr(self, '_sub_tab_widgets') else None
            if sub_tab is not None:
                sub_idx = sub_tab.currentIndex()
                if 0 <= sub_idx < len(self.categories[cat_name]):
                    tab_name = self.categories[cat_name][sub_idx][0]
                    self._switch_to_tab_name(tab_name)

    def _on_sub_tab_changed(self, sub_tab: Any, idx: int):
        if hasattr(self, '_sub_tab_widgets'):
            for cat_name, st in self._sub_tab_widgets.items():
                if st is sub_tab:
                    if 0 <= idx < len(self.categories[cat_name]):
                        tab_name = self.categories[cat_name][idx][0]
                        self._switch_to_tab_name(tab_name)
                    break

    def set_active_tab(self, tab_name: str) -> bool:
        """
        Sets the active tab by name.
        Returns True if tab_name is valid and tab is switched, False otherwise.
        """
        if tab_name not in self.tabs:
            return False

        if not hasattr(self, 'category_names') or not hasattr(self, '_sub_tab_widgets'):
            self.active_tab = tab_name
            return True

        for cat_idx, cat_name in enumerate(self.category_names):
            for sub_idx, (t_name, _) in enumerate(self.categories.get(cat_name, [])):
                if t_name == tab_name:
                    sub_tab = self._sub_tab_widgets.get(cat_name)
                    if sub_tab is not None:
                        sub_tab.blockSignals(True)
                        sub_tab.setCurrentIndex(sub_idx)
                        sub_tab.blockSignals(False)
                    if hasattr(super(), "setCurrentIndex"):
                        self.blockSignals(True)
                        super().setCurrentIndex(cat_idx)
                        self.blockSignals(False)
                    self._switch_to_tab_name(tab_name)
                    return True
        return False

    def showEvent(self, event: Any) -> None:
        if QT_AVAILABLE and hasattr(super(), 'showEvent'):
            try:
                super().showEvent(event)
            except Exception:
                pass
        if getattr(self, '_needs_refresh', False):
            self._needs_refresh = False
            self.refresh_analysis()

    def refresh_analysis(self, wavelengths: Optional[List[float]] = None, source_config: Optional[Any] = None) -> bool:
        """
        Updates analysis views for active tab using system model.
        Returns True on completion.
        """
        if not (QT_AVAILABLE and HAS_MPL_QT):
            return True

        if wavelengths is not None:
            self.last_wavelengths = wavelengths
        else:
            wavelengths = self.last_wavelengths

        if source_config is not None:
            self.last_source_config = source_config
        else:
            source_config = self.last_source_config

        # Check visibility: if analysis dock or tab widget is hidden / tabbed out, defer computation
        if hasattr(self, 'isVisible'):
            import os, sys
            is_headless = os.environ.get('QT_QPA_PLATFORM') == 'offscreen' or 'pytest' in sys.modules
            if not is_headless:
                dock = getattr(self.parent(), 'analysis_dock', None) or (self.parent() if hasattr(self.parent(), 'isVisible') else None)
                is_vis = dock.isVisible() if (dock is not None and hasattr(dock, 'isVisible')) else self.isVisible()
                if not is_vis:
                    self._needs_refresh = True
                    return True

        stack = None
        if self.model is not None:
            if hasattr(self.model, "get_surface_stack"):
                try:
                    stack = self.model.get_surface_stack()
                except Exception:
                    pass
            elif hasattr(self.model, "stack"):
                stack = self.model.stack
            elif isinstance(self.model, SurfaceStack):
                stack = self.model

        self.update_surface_combo(stack)
        system = OpticalSystem(stack) if stack and len(stack.surfaces) > 0 else None
        
        cfg = source_config
        
        target_idx = -1
        if hasattr(self, 'surface_combo'):
            combo_data = self.surface_combo.currentData()
            if combo_data is not None:
                target_idx = combo_data

        sys_cfg = getattr(self, 'system_config', None)
        if sys_cfg is None and self.model is not None:
            sys_cfg = getattr(self.model, 'system_config', None)
        if sys_cfg is None and hasattr(self.parent(), 'system_config'):
            sys_cfg = getattr(self.parent(), 'system_config', None)

        if self.active_tab == "Spot Diagram" and hasattr(self, "tab_spot"):
            self.tab_spot.update_plot(system=system, wavelengths=wavelengths, target_surface_index=target_idx, source_config=cfg, system_config=sys_cfg)
        elif self.active_tab == "Optical Diagnostics & First Order" and hasattr(self, "tab_first_order"):
            self.tab_first_order.update_plot(system=system, wavelengths=wavelengths, system_config=sys_cfg)
        elif self.active_tab == "Manufacturing & Mounting Audit" and hasattr(self, "tab_mfg"):
            self.tab_mfg.update_plot(system=system, wavelengths=wavelengths, system_config=sys_cfg)
        elif self.active_tab == "Best Focus & Aberrations" and hasattr(self, "tab_best_focus"):
            self.tab_best_focus.update_plot(system=system, wavelengths=wavelengths, system_config=sys_cfg)
        elif self.active_tab == "Seidel Aberrations" and hasattr(self, "tab_seidel"):
            self.tab_seidel.update_plot(system=system, target_surface_index=target_idx, source_config=cfg, system_config=sys_cfg, wavelengths=wavelengths)
        elif self.active_tab == "Ray Fan" and hasattr(self, "tab_rayfan"):
            self.tab_rayfan.update_plot(system=system, target_surface_index=target_idx, source_config=cfg, system_config=sys_cfg)
        elif self.active_tab == "Grating Dispersion" and hasattr(self, "tab_dispersion"):
            self.tab_dispersion.update_plot(system=system, target_surface_index=target_idx, source_config=cfg)
        elif self.active_tab == "CCD Heatmap" and hasattr(self, "tab_ccd"):
            self.tab_ccd.update_plot(system=system, target_surface_index=target_idx, source_config=cfg)
        elif self.active_tab == "Laser Scanning & Field Aberrations" and hasattr(self, "tab_laser_scan"):
            self.tab_laser_scan.update_plot(system=system, wavelengths=wavelengths, target_surface_index=target_idx, source_config=cfg)
        elif self.active_tab == "Femtosecond Pulse" and hasattr(self, "tab_pulse"):
            self.tab_pulse.update_plot(system=system, wavelengths=wavelengths, source_config=cfg)
        elif self.active_tab == "Two-Photon Microscope" and hasattr(self, "tab_two_photon"):
            self.tab_two_photon.update_plot(stack=stack)
        elif self.active_tab == "Nonlinear Imaging" and hasattr(self, "tab_nonlinear"):
            self.tab_nonlinear.update_plot(stack=stack)
        elif self.active_tab == "Achromatic Doublet Designer" and hasattr(self, "tab_achromat"):
            self.tab_achromat.update_plot()
        elif self.active_tab == "Spectrometer Quality & Resolution (Zemax)" and hasattr(self, "tab_spectrometer"):
            self.tab_spectrometer.update_plot(system=system, wavelengths=wavelengths, system_config=sys_cfg)

        return True

    def update_analysis(self, stack_or_system: Any = None) -> bool:
        """Update analysis plots."""
        return self.refresh_analysis()

    def get_analysis_config(self) -> Dict[str, Any]:
        current_routes = None
        stack = None
        if self.model is not None and hasattr(self.model, "get_surface_stack"):
            stack = self.model.get_surface_stack()
            current_routes = getattr(stack, "optical_routes", None)
        result = {
            "pulse_width_analysis": self.tab_pulse.get_config()
            if hasattr(self, "tab_pulse") else PulseAnalysisConfig().to_dict(),
        }
        two_photon = self._two_photon_explicit or (stack is not None and hasattr(stack, "two_photon_config"))
        nonlinear = self._nonlinear_explicit or (stack is not None and hasattr(stack, "nonlinear_imaging_config"))
        if two_photon:
            result["two_photon_microscope"] = (
                self.tab_two_photon.get_config() if hasattr(self, "tab_two_photon")
                else self.two_photon_config.to_dict()
            )
        if nonlinear:
            result["nonlinear_imaging"] = (
                self.tab_nonlinear.get_config() if hasattr(self, "tab_nonlinear")
                else self.nonlinear_imaging_config.to_dict()
            )
        if current_routes is not None:
            result["optical_routes"] = current_routes.to_dict()
        elif self.active_tab == "Nonlinear Imaging" and nonlinear:
            result["optical_routes"] = self.nonlinear_imaging_config.routes.to_dict()
        elif two_photon and not nonlinear:
            result["optical_routes"] = self.two_photon_config.routes.to_dict()
        return result

    def set_analysis_config(self, data: Optional[Dict[str, Any]]) -> None:
        if hasattr(self, "tab_pulse"):
            self.tab_pulse.set_config((data or {}).get("pulse_width_analysis", {}))
        payload = (data or {}).get("two_photon_microscope", {})
        self._two_photon_explicit = bool(payload)
        self.two_photon_config = TwoPhotonMicroscopeConfig.from_dict(payload) if payload else build_calibrated_template()
        routes = (data or {}).get("optical_routes")
        if routes:
            from optics_app.engine.routes import OpticalRouteSet
            self.two_photon_config.routes = OpticalRouteSet.from_dict(routes)
        if hasattr(self, "tab_two_photon"):
            self.tab_two_photon.set_config(self.two_photon_config.to_dict())
        nonlinear = (data or {}).get("nonlinear_imaging", {})
        self._nonlinear_explicit = bool(nonlinear)
        self.nonlinear_imaging_config = NonlinearImagingConfig.from_dict(nonlinear) if nonlinear else build_nonlinear_surface_template().nonlinear_imaging_config
        if routes and nonlinear:
            self.nonlinear_imaging_config.routes = OpticalRouteSet.from_dict(routes)
        if hasattr(self, "tab_nonlinear"):
            self.tab_nonlinear.set_config(self.nonlinear_imaging_config.to_dict())


AnalysisTabSuite = AnalysisTabsWidget
