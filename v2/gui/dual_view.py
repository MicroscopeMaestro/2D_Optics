"""
Synchronized Dual-View System Layout Window for 2D Optics Simulation Engine.
Provides View 1: Longitudinal Optical Axis Plane (R-Z / Y-Z) showing 2D surface cross-sections z(r) & rays.
Provides View 2: Transverse Layout Plane (X-Y) showing semi-diameters, decenters, tilts, grating lines & 2D aperture geometries.
"""

import sys
import os
import warnings
import numpy as np
from typing import Optional, List, Dict, Any, Union, Tuple

warnings.filterwarnings("ignore", message=".*Tight layout not applied.*")
warnings.filterwarnings("ignore", category=UserWarning, module=".*matplotlib.*")

from optics_app.gui.qt_compat import (
    QWidget, QVBoxLayout, QHBoxLayout, QT_AVAILABLE, QCheckBox, QPushButton, QLabel, pyqtSignal, DummySignal,
    QDialog, QGridLayout, QGroupBox, QDoubleSpinBox, Qt
)
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.ray import Ray
from optics_app.engine.tracer import OpticalSystem
try:
    from optics_app.engine.mechanisms import available_states, active_state_name, get_effective_pose, is_dmd
    from optics_app.engine.plm import is_plm, generate_plm_2d_maps
except ImportError:
    from v2.engine.mechanisms import available_states, active_state_name, get_effective_pose, is_dmd
    from v2.engine.plm import is_plm, generate_plm_2d_maps

FIELD_COLORS = ['#2563eb', '#16a34a', '#dc2626', '#9333ea', '#ea580c', '#0891b2', '#ca8a04', '#e11d48']

# Matplotlib Qt Canvas handling
HAS_MPL_QT = False
try:
    import matplotlib
    matplotlib.use('QtAgg' if QT_AVAILABLE else 'Agg')
    import matplotlib.patches as patches
    from matplotlib.figure import Figure
    from matplotlib.collections import LineCollection
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


def _surface_sag_values(surf: Surface, radii: np.ndarray) -> np.ndarray:
    """Vectorized surface sag calculation."""
    radius = float(getattr(surf, "radius", 0.0) or 0.0)
    conic = float(getattr(surf, "conic", 0.0) or 0.0)
    r2 = radii ** 2
    if radius == 0.0 or np.isinf(radius):
        z = np.zeros_like(radii, dtype=float)
    else:
        curvature = 1.0 / radius
        argument = 1.0 - (1.0 + conic) * (curvature ** 2) * r2
        z = curvature * r2 / (1.0 + np.sqrt(np.maximum(0.0, argument)))
    for index, coefficient in enumerate(getattr(surf, "asphere_coeffs", []) or []):
        z = z + float(coefficient) * radii ** (2 * index + 4)
    return z


def draw_aperture_2d_front_view(ax, surface: Surface):
    """
    Renders 2D Front-View (Y vs X) clear aperture bounds and spider legs
    in Matplotlib axes using geometry patches.
    """
    import matplotlib.patches as patches

    ax.clear()
    ax.set_title(f"Aperture View: S ({surface.name}) - {surface.aperture_type}", fontsize=10, fontweight='bold')
    ax.set_xlabel("X Position (mm)")
    ax.set_ylabel("Y Position (mm)")
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.axhline(0, color='black', linestyle=':', alpha=0.5)
    ax.axvline(0, color='black', linestyle=':', alpha=0.5)

    atype = str(surface.aperture_type).capitalize() if surface.aperture_type else "Circular"
    rout = surface.aperture_r_outer if surface.aperture_r_outer > 0 else surface.semi_diameter

    if atype == "Circular":
        circle = patches.Circle((0, 0), rout, fill=False, edgecolor='blue', linewidth=1.8, label='Clear Aperture')
        ax.add_patch(circle)
        limit = rout * 1.2

    elif atype == "Rectangular":
        xw = surface.aperture_x_half if getattr(surface, 'aperture_x_half', 0) > 0 else (surface.semi_diameter if surface.semi_diameter > 0 else 10.0)
        yw = surface.aperture_y_half if getattr(surface, 'aperture_y_half', 0) > 0 else (surface.semi_diameter if surface.semi_diameter > 0 else 10.0)
        rect = patches.Rectangle((-xw, -yw), 2*xw, 2*yw, fill=False, edgecolor='green', linewidth=1.8, label='Rectangular Aperture')
        ax.add_patch(rect)
        limit = max(xw, yw) * 1.2

    elif atype == "Elliptical":
        a, b = surface.aperture_a_semi, surface.aperture_b_semi
        ellipse = patches.Ellipse((0, 0), 2*a, 2*b, fill=False, edgecolor='cyan', linewidth=1.8, label='Elliptical Aperture')
        ax.add_patch(ellipse)
        limit = max(a, b) * 1.2

    elif atype == "Annular":
        rin = surface.aperture_r_inner
        outer = patches.Circle((0, 0), rout, fill=False, edgecolor='blue', linewidth=1.8)
        inner = patches.Circle((0, 0), rin, fill=True, facecolor='gray', alpha=0.5, edgecolor='black', label='Central Obscuration')
        ax.add_patch(outer)
        ax.add_patch(inner)
        limit = rout * 1.2

    elif atype == "Spider":
        rcnt = surface.aperture_center_radius
        w = surface.aperture_spider_width
        n_legs = surface.aperture_spider_count
        phi0 = np.radians(surface.aperture_spider_angle)

        outer = patches.Circle((0, 0), rout, fill=False, edgecolor='blue', linewidth=1.8)
        ax.add_patch(outer)

        hub = patches.Circle((0, 0), rcnt, fill=True, facecolor='darkgray', edgecolor='black', zorder=3, label='Central Hub')
        ax.add_patch(hub)

        half_w = w / 2.0
        for k in range(n_legs):
            angle = phi0 + 2.0 * np.pi * k / n_legs
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            corners_local = np.array([
                [rcnt, -half_w],
                [rout, -half_w],
                [rout,  half_w],
                [rcnt,  half_w]
            ])
            R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            corners_global = corners_local @ R.T
            leg_poly = patches.Polygon(corners_global, closed=True, facecolor='black', edgecolor='black', alpha=0.85, zorder=2)
            ax.add_patch(leg_poly)

        limit = rout * 1.2
    else:
        limit = rout * 1.2

    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect('equal')



class RescaleViewDialog(QDialog if QT_AVAILABLE else object):
    """
    Dialog to precisely rescale the optical system view, setting Z and Transverse limits
    as well as transverse aspect ratio scaling factor.
    """
    def __init__(self, dual_view_widget, parent=None):
        if QT_AVAILABLE:
            super().__init__(parent)
            self.setWindowTitle("Rescale Optical System View")
            self.resize(380, 360)
        self.dual_view = dual_view_widget
        self._init_ui()

    def _init_ui(self):
        if not QT_AVAILABLE:
            return
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # 1. Transverse Aspect Scaling (Exaggeration factor)
        grp_aspect = QGroupBox("Transverse Scale (Y/X Exaggeration)")
        layout_aspect = QVBoxLayout(grp_aspect)

        lbl_desc = QLabel("Adjust aspect ratio to make thin lens elements clearer or view 1:1 true scale:")
        lbl_desc.setWordWrap(True)
        lbl_desc.setStyleSheet("color: #475569; font-size: 11px;")
        layout_aspect.addWidget(lbl_desc)

        row_scale = QHBoxLayout()
        row_scale.addWidget(QLabel("Transverse ×:"))
        self.spin_scale = QDoubleSpinBox()
        self.spin_scale.setRange(1.0, 100.0)
        self.spin_scale.setSingleStep(0.5)
        self.spin_scale.setDecimals(1)
        cur_scale = float(getattr(self.dual_view, "transverse_scale", 1.0) or 1.0)
        self.spin_scale.setValue(cur_scale)
        row_scale.addWidget(self.spin_scale)
        layout_aspect.addLayout(row_scale)

        # Preset buttons row
        row_presets = QHBoxLayout()
        presets = [("1:1 True", 1.0), ("2×", 2.0), ("5×", 5.0), ("10×", 10.0)]
        for label, val in presets:
            btn = QPushButton(label)
            btn.clicked.connect(lambda checked=False, v=val: self.spin_scale.setValue(v))
            row_presets.addWidget(btn)
        layout_aspect.addLayout(row_presets)
        layout.addWidget(grp_aspect)

        # 2. Coordinate View Limits (mm)
        grp_limits = QGroupBox("Coordinate View Limits (mm)")
        grid_limits = QGridLayout(grp_limits)

        cur_z_min, cur_z_max, cur_y_min, cur_y_max = self._get_current_limits()

        grid_limits.addWidget(QLabel("Z Min (mm):"), 0, 0)
        self.spin_z_min = QDoubleSpinBox()
        self.spin_z_min.setRange(-100000.0, 100000.0)
        self.spin_z_min.setDecimals(2)
        self.spin_z_min.setValue(cur_z_min)
        grid_limits.addWidget(self.spin_z_min, 0, 1)

        grid_limits.addWidget(QLabel("Z Max (mm):"), 0, 2)
        self.spin_z_max = QDoubleSpinBox()
        self.spin_z_max.setRange(-100000.0, 100000.0)
        self.spin_z_max.setDecimals(2)
        self.spin_z_max.setValue(cur_z_max)
        grid_limits.addWidget(self.spin_z_max, 0, 3)

        grid_limits.addWidget(QLabel("Transverse Min:"), 1, 0)
        self.spin_y_min = QDoubleSpinBox()
        self.spin_y_min.setRange(-100000.0, 100000.0)
        self.spin_y_min.setDecimals(2)
        self.spin_y_min.setValue(cur_y_min)
        grid_limits.addWidget(self.spin_y_min, 1, 1)

        grid_limits.addWidget(QLabel("Transverse Max:"), 1, 2)
        self.spin_y_max = QDoubleSpinBox()
        self.spin_y_max.setRange(-100000.0, 100000.0)
        self.spin_y_max.setDecimals(2)
        self.spin_y_max.setValue(cur_y_max)
        grid_limits.addWidget(self.spin_y_max, 1, 3)

        btn_auto_fit = QPushButton("🔍 Auto-Calculate from System Extents")
        btn_auto_fit.setToolTip("Set limits to encompass all optical surfaces and ray paths with 10% margin")
        btn_auto_fit.clicked.connect(self._auto_calculate_extents)
        grid_limits.addWidget(btn_auto_fit, 2, 0, 1, 4)

        layout.addWidget(grp_limits)

        # 3. Action Buttons
        btn_row = QHBoxLayout()
        btn_apply = QPushButton("Apply")
        btn_apply.setToolTip("Apply limits immediately to the viewer")
        btn_apply.clicked.connect(self.apply_changes)
        btn_row.addWidget(btn_apply)

        btn_ok = QPushButton("OK")
        btn_ok.setDefault(True)
        btn_ok.clicked.connect(self._on_ok)
        btn_row.addWidget(btn_ok)

        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(btn_cancel)

        layout.addLayout(btn_row)

    def _get_current_limits(self) -> Tuple[float, float, float, float]:
        if self.dual_view is not None and getattr(self.dual_view, "ax_rz", None) is not None:
            xlim = self.dual_view.ax_rz.get_xlim()
            ylim = self.dual_view.ax_rz.get_ylim()
            return float(xlim[0]), float(xlim[1]), float(ylim[0]), float(ylim[1])
        if self.dual_view is not None and hasattr(self.dual_view, "get_system_extents"):
            return self.dual_view.get_system_extents()
        return -10.0, 110.0, -30.0, 30.0

    def _auto_calculate_extents(self):
        if self.dual_view is not None and hasattr(self.dual_view, "get_system_extents"):
            zm, zx, ym, yx = self.dual_view.get_system_extents()
            self.spin_z_min.setValue(zm)
            self.spin_z_max.setValue(zx)
            self.spin_y_min.setValue(ym)
            self.spin_y_max.setValue(yx)

    def apply_changes(self):
        if self.dual_view is None:
            return
        scale = float(self.spin_scale.value())
        self.dual_view.transverse_scale = max(1.0, scale)
        if hasattr(self.dual_view, 'spin_transverse_scale') and self.dual_view.spin_transverse_scale:
            self.dual_view.spin_transverse_scale.blockSignals(True)
            self.dual_view.spin_transverse_scale.setValue(scale)
            self.dual_view.spin_transverse_scale.blockSignals(False)

        zm = float(self.spin_z_min.value())
        zx = float(self.spin_z_max.value())
        ym = float(self.spin_y_min.value())
        yx = float(self.spin_y_max.value())
        self.dual_view.set_view_limits(zm, zx, ym, yx)

    def _on_ok(self):
        self.apply_changes()
        self.accept()


class DualViewWidget(QWidget if QT_AVAILABLE else object):
    """
    Synchronized Dual-View Layout Widget.
    View 1 (R-Z): Longitudinal Z-axis optical cross-section (surface sags z(r) and ray trajectories).
    View 2 (X-Y): Transverse layout plane (clear apertures, decenters, tilts, grating lines, 2D Front-View apertures).
    """
    measurements_changed = pyqtSignal(list, str) if QT_AVAILABLE else DummySignal()
    angles_visibility_changed = pyqtSignal(bool) if QT_AVAILABLE else DummySignal()
    labels_visibility_changed = pyqtSignal(bool) if QT_AVAILABLE else DummySignal()
    dark_mode_changed = pyqtSignal(bool) if QT_AVAILABLE else DummySignal()
    marginal_chief_visibility_changed = pyqtSignal(bool) if QT_AVAILABLE else DummySignal()

    def __init__(self, model: Optional[Any] = None, parent=None, auto_refresh: bool = True):
        if QT_AVAILABLE:
            super().__init__(parent)
        self.model = model
        self.surface_stack: Optional[SurfaceStack] = None
        self.rays: Optional[List[Ray]] = None

        self.rz_view_active: bool = True
        self.xz_view_active: bool = False
        self.xy_front_view_active: bool = False
        self.cb_view_rz: Optional[Any] = None
        self.cb_view_xz: Optional[Any] = None
        self.cb_view_front: Optional[Any] = None
        self.measure_mode: Optional[str] = None # None, "distance", "angle"
        self.measure_points: List[Dict[str, Any]] = []
        self.measurement_history: List[Dict[str, Any]] = []
        self.active_drag_handle: Optional[Dict[str, Any]] = None
        self.hover_drag_handle: Optional[Dict[str, Any]] = None
        self.hover_snap_target: Optional[Dict[str, Any]] = None
        self.selected_measurement_idx: Optional[int] = None
        self.mouse_cursor_pos: Optional[Tuple[float, float]] = None
        self.saved_limits: Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]] = {}
        self._force_reset_view: bool = False
        self._pan_start: Optional[Dict[str, Any]] = None
        self.snap_enabled: bool = True
        self.show_incidence_angles: bool = True
        self.show_labels: bool = True
        self.cb_show_labels: Optional[Any] = None
        self.dark_mode: bool = False
        self.cb_dark_mode: Optional[Any] = None
        self.show_marginal_chief: bool = False
        self.marginal_chief_only: bool = False
        self.cb_marginal_chief: Optional[Any] = None
        self.show_pupils_windows: bool = False
        self.cb_pupils_windows: Optional[Any] = None
        self.color_mode: str = "wavelength"
        self._ray_artists_rz: List[Any] = []
        self._ray_artists_xz: List[Any] = []
        self._ray_artists_xy: List[Any] = []
        self._last_geom_fingerprint: Optional[Any] = None
        self._surf_sag_cache: Dict[Any, Any] = {}

        if QT_AVAILABLE and HAS_MPL_QT:
            self._init_ui()
        else:
            self.fig = None
            self.canvas = None

        if auto_refresh and self.model is not None and hasattr(self.model, "system_updated"):
            try:
                self.model.system_updated.connect(self.refresh_layout)
            except Exception:
                pass

    def set_color_mode(self, mode: str) -> None:
        """Set ray coloring mode: 'wavelength' or 'field'."""
        self.color_mode = mode
        self.refresh_layout()

    def toggle_color_mode(self) -> None:
        """Toggle ray coloring between 'wavelength' and 'field'."""
        self.color_mode = "field" if getattr(self, "color_mode", "wavelength") == "wavelength" else "wavelength"
        if hasattr(self, 'btn_color_mode') and self.btn_color_mode:
            self.btn_color_mode.setText("🎨 Field" if self.color_mode == "field" else "🎨 Wavelength")
        self.refresh_layout()

    def get_mech_surface_info(self) -> Optional[Tuple[int, Surface, str]]:
        """Return (index, surface, current_state) for the first DMD in the stack, or None."""
        if not self.surface_stack or not self.surface_stack.surfaces:
            return None
        for idx, s in enumerate(self.surface_stack.surfaces):
            if available_states(s):
                return idx, s, active_state_name(s)
        return None

    get_dmd_surface_info = get_mech_surface_info

    def toggle_mech_state(self) -> Optional[str]:
        """
        Toggle DMD mechanical state between ON and OFF, update the model,
        and re-trace rays to visualize the diverted beam without moving optical elements.
        """
        info = self.get_mech_surface_info()
        if not info:
            if hasattr(self, "btn_mech_toggle") and self.btn_mech_toggle:
                self.btn_mech_toggle.setVisible(False)
            return None

        mech_idx, mech_surf, cur_state = info
        from optics_app.engine.mechanisms import available_states
        states = list(available_states(mech_surf))
        if not states:
            return None
        if "ON" in states and "OFF" in states:
            new_state = "OFF" if cur_state == "ON" else "ON"
        else:
            new_state = states[(states.index(cur_state) + 1) % len(states)] if cur_state in states else states[0]

        # Apply state through model if available
        if hasattr(self, "model") and self.model is not None and hasattr(self.model, "set_mechanical_state"):
            self.model.set_mechanical_state(mech_idx, new_state)
            if hasattr(self.model, "get_surface_stack"):
                self.surface_stack = self.model.get_surface_stack()
        else:
            from optics_app.engine.mechanisms import MechanicalStateController
            ctrl = MechanicalStateController(self.surface_stack)
            ctrl.set_state(mech_idx, new_state)
            self.surface_stack._update_accumulated_positions()

        # Update button text & style
        self._update_mech_button_ui(new_state)

        # Notify parent or window if available to trigger debounced trace
        p = self.parent()
        traced = False
        while p is not None:
            if hasattr(p, "request_debounced_trace"):
                p.request_debounced_trace()
                traced = True
                break
            elif hasattr(p, "on_trace_rays"):
                p.on_trace_rays()
                traced = True
                break
            p = p.parent() if hasattr(p, "parent") else None

        if not traced and self.surface_stack:
            sys = OpticalSystem(self.surface_stack)
            if self.rays:
                new_rays = [sys.trace_ray(Ray(pos=r.pos.copy(), dir=r.dir.copy(), wavelength=r.wavelength)) for r in self.rays]
                self.update_layout(self.surface_stack, new_rays)
            else:
                self.refresh_layout()

        return new_state

    toggle_dmd_state = toggle_mech_state

    def _sync_mech_button_ui(self) -> None:
        if not hasattr(self, "btn_mech_toggle") or not self.btn_mech_toggle:
            return
        info = self.get_mech_surface_info()
        if not info:
            self.btn_mech_toggle.setVisible(False)
            return
        surface = info[1]
        try:
            from optics_app.engine.mechanisms import is_dmd
            from optics_app.engine.plm import is_plm
        except ImportError:
            from v2.engine.mechanisms import is_dmd
            from v2.engine.plm import is_plm
        if surface and (is_dmd(surface) or is_plm(surface) or getattr(surface, 'is_plm', False)):
            # TI devices (PLM / DMD) are controlled exclusively via TI Modulator Studio
            self.btn_mech_toggle.setVisible(False)
            self._update_mech_button_ui(info[2])
            return
        self.btn_mech_toggle.setVisible(True)
        self._update_mech_button_ui(info[2])

    def _update_mech_button_ui(self, state: str) -> None:
        if not hasattr(self, "btn_mech_toggle") or not self.btn_mech_toggle:
            return
        from optics_app.engine.mechanisms import available_states
        info = self.get_mech_surface_info()
        surface = info[1] if info else None
        spec = available_states(surface).get(state, {}) if surface else {}
        route = str(spec.get("route", "") or "")

        try:
            from optics_app.engine.mechanisms import is_dmd
            from optics_app.engine.plm import is_plm
        except ImportError:
            from v2.engine.mechanisms import is_dmd
            from v2.engine.plm import is_plm

        if surface and is_dmd(surface):
            prefix = "DMD"
            if route == "detector_path" or state == "ON":
                route_desc = "Detector"
            elif route == "beam_dump" or state == "OFF":
                route_desc = "Dump"
            else:
                route_desc = route
            suffix = f" ({route_desc})" if route_desc else ""
            self.btn_mech_toggle.setText(f"🔲 {prefix}: {state}{suffix}")
            # TI devices (PLM / DMD) are controlled exclusively via TI Modulator Studio
            self.btn_mech_toggle.setVisible(False)
            return
        elif surface and (is_plm(surface) or getattr(surface, 'is_plm', False)):
            prefix = "PLM"
            suffix = f" ({route})" if route else ""
            self.btn_mech_toggle.setText(f"🔲 {prefix}: {state}{suffix}")
            # TI devices (PLM / DMD) are controlled exclusively via TI Modulator Studio
            self.btn_mech_toggle.setVisible(False)
            return
        else:
            prefix = "MECH"
            route_desc = route
            self.btn_mech_toggle.setToolTip(f"Mechanism state: {state}. Click to toggle active state.")
            suffix = f" ({route_desc})" if route_desc else ""
            self.btn_mech_toggle.setText(f"🔲 {prefix}: {state}{suffix}")
        self.btn_mech_toggle.setEnabled(bool(info and len(available_states(surface)) > 1))
        if state == "ON":
            self.btn_mech_toggle.setStyleSheet(
                "QPushButton { background: #dcfce7; color: #166534; border: 1px solid #86efac; border-radius: 4px; padding: 4px 10px; font-weight: bold; font-size: 11px; }"
                "QPushButton:hover { background: #bbf7d0; }"
            )
        elif state == "OFF":
            self.btn_mech_toggle.setStyleSheet(
                "QPushButton { background: #fee2e2; color: #991b1b; border: 1px solid #fca5a5; border-radius: 4px; padding: 4px 10px; font-weight: bold; font-size: 11px; }"
                "QPushButton:hover { background: #fecaca; }"
            )
        else:
            self.btn_mech_toggle.setStyleSheet(
                "QPushButton { background: #e0f2fe; color: #075985; border: 1px solid #7dd3fc; border-radius: 4px; padding: 4px 10px; font-weight: bold; font-size: 11px; }"
            )

    def apply_tight_layout(self, pad: float = 1.08):
        """Safely applies tight layout without raising UserWarnings if margins are constrained."""
        if self.fig is None:
            return
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.fig.tight_layout(pad=pad)
        except Exception:
            try:
                self.fig.subplots_adjust(left=0.08, right=0.96, top=0.94, bottom=0.08, hspace=0.25)
            except Exception:
                pass

    def set_show_incidence_angles(self, enabled: bool) -> None:
        """Enables or disables on-screen visualization of incidence and reflection/output angles for the main wavelength."""
        self.show_incidence_angles = bool(enabled)
        if hasattr(self, 'angles_visibility_changed'):
            try:
                self.angles_visibility_changed.emit(self.show_incidence_angles)
            except Exception:
                pass
        self.refresh_layout()

    def set_show_labels(self, show: bool) -> None:
        """Toggle optical element, stop, and callout labels in 2D projections."""
        self.show_labels = bool(show)
        if self.cb_show_labels and hasattr(self.cb_show_labels, "isChecked") and self.cb_show_labels.isChecked() != self.show_labels:
            self.cb_show_labels.blockSignals(True)
            self.cb_show_labels.setChecked(self.show_labels)
            self.cb_show_labels.blockSignals(False)
        if hasattr(self, 'labels_visibility_changed'):
            try:
                self.labels_visibility_changed.emit(self.show_labels)
            except Exception:
                pass
        self.refresh_layout()

    def set_dark_mode(self, enabled: bool) -> None:
        """Toggle dark background design in 2D projections."""
        self.dark_mode = bool(enabled)
        if self.cb_dark_mode and hasattr(self.cb_dark_mode, "isChecked") and self.cb_dark_mode.isChecked() != self.dark_mode:
            self.cb_dark_mode.blockSignals(True)
            self.cb_dark_mode.setChecked(self.dark_mode)
            self.cb_dark_mode.blockSignals(False)
        cb_col = '#cbd5e1' if self.dark_mode else '#475569'
        hover_col = '#f1f5f9' if self.dark_mode else '#0f172a'
        cb_style = (
            f"QCheckBox {{ font-size: 11px; font-weight: 500; color: {cb_col}; spacing: 4px; }} "
            f"QCheckBox:hover {{ color: {hover_col}; }}"
        )
        if hasattr(self, 'cb_show_labels') and self.cb_show_labels:
            self.cb_show_labels.setStyleSheet(cb_style)
        if hasattr(self, 'cb_dark_mode') and self.cb_dark_mode:
            self.cb_dark_mode.setStyleSheet(cb_style)
        if hasattr(self, 'cb_marginal_chief') and self.cb_marginal_chief:
            self.cb_marginal_chief.setStyleSheet(cb_style)
        if hasattr(self, 'cb_pupils_windows') and self.cb_pupils_windows:
            self.cb_pupils_windows.setStyleSheet(cb_style)
        if hasattr(self, 'dark_mode_changed'):
            try:
                self.dark_mode_changed.emit(self.dark_mode)
            except Exception:
                pass
        self.refresh_layout()

    def _classify_ray(self, ray: Any) -> Tuple[bool, bool]:
        """
        Determine whether a ray is a Chief Ray or a Marginal Ray.
        Returns (is_chief, is_marginal).

        Rigorous Optical Definitions:
        - Marginal Ray: origin from on-axis object point (y=0), passes through aperture stop / entrance pupil rim (Py = ±1.0 or Px = ±1.0).
        - Chief Ray: origin from off-axis field point (or max field), passes through aperture stop / entrance pupil center (Px = 0, Py = 0).
        """
        is_chief = bool(getattr(ray, "is_chief", False))
        is_marginal = bool(getattr(ray, "is_marginal", False))
        f_idx = getattr(ray, "field_idx", 0)
        fc = getattr(ray, "field_coord", None)
        is_on_axis = (f_idx == 0 and (fc is None or (abs(fc[0]) < 1e-6 and abs(fc[1]) < 1e-6)))

        if hasattr(ray, "pupil_coord"):
            p = ray.pupil_coord
            if p is not None:
                if not is_marginal and is_on_axis:
                    if (abs(p[0]) < 1e-5 and abs(abs(p[1]) - 1.0) < 1e-4) or (abs(p[1]) < 1e-5 and abs(abs(p[0]) - 1.0) < 1e-4):
                        is_marginal = True
                if not is_chief and not is_on_axis:
                    if abs(p[0]) < 1e-5 and abs(p[1]) < 1e-5:
                        is_chief = True

        return is_chief, is_marginal

    def set_show_marginal_chief(self, enabled: bool) -> None:
        """Toggle highlighting of marginal and chief rays in optical projections."""
        self.show_marginal_chief = bool(enabled)
        if self.cb_marginal_chief and hasattr(self.cb_marginal_chief, "isChecked") and self.cb_marginal_chief.isChecked() != self.show_marginal_chief:
            self.cb_marginal_chief.blockSignals(True)
            self.cb_marginal_chief.setChecked(self.show_marginal_chief)
            self.cb_marginal_chief.blockSignals(False)
        if hasattr(self, 'marginal_chief_visibility_changed'):
            try:
                self.marginal_chief_visibility_changed.emit(self.show_marginal_chief)
            except Exception:
                pass
        self.refresh_layout()

    def set_marginal_chief_only(self, enabled: bool) -> None:
        """Filter layout to show exclusively marginal and chief rays."""
        self.marginal_chief_only = bool(enabled)
        self.refresh_layout()

    def set_show_pupils_windows(self, enabled: bool) -> None:
        """Toggle visual overlay of Entrance/Exit Pupils, Entrance/Exit Windows, and Field/Aperture Stops."""
        self.show_pupils_windows = bool(enabled)
        if self.cb_pupils_windows and hasattr(self.cb_pupils_windows, "isChecked") and self.cb_pupils_windows.isChecked() != self.show_pupils_windows:
            self.cb_pupils_windows.blockSignals(True)
            self.cb_pupils_windows.setChecked(self.show_pupils_windows)
            self.cb_pupils_windows.blockSignals(False)
        self.refresh_layout()

    def delete_measurement(self, index: int):
        """Deletes a measurement by index and refreshes layout & inspector."""
        if 0 <= index < len(self.measurement_history):
            self.measurement_history.pop(index)
            self.selected_measurement_idx = None
            self._emit_measurements_changed("Measurement deleted.")
            self.refresh_layout()

    def select_measurement(self, index: Optional[int]):
        """Selects and highlights a measurement in the 2D view."""
        self.selected_measurement_idx = index
        self.refresh_layout()

    def _emit_measurements_changed(self, msg: str = ""):
        """Broadcast updated measurement list to MeasurementToolsDockWidget."""
        if hasattr(self, 'measurements_changed') and self.measurements_changed is not None:
            try:
                self.measurements_changed.emit(self.measurement_history, msg)
            except Exception:
                pass

    def set_selected_surface(self, index: Optional[int]) -> None:
        """Update selected surface index and re-render highlight on 2D longitudinal axis."""
        if getattr(self, 'selected_surface_index', None) == index:
            return
        self.selected_surface_index = index
        self.refresh_layout()

    @property
    def system(self):
        if self.surface_stack is not None:
            return OpticalSystem(self.surface_stack)
        return None

    def update_views(self) -> None:
        self.refresh_layout()

    def redraw(self) -> None:
        self.refresh_layout()

    transverse_scale: float = 1.0

    def set_transverse_scale(self, value: float) -> None:
        """Exaggerate transverse axis vs optical (Z) axis; 1.0 is true scale."""
        self.transverse_scale = max(1.0, float(value))
        self.reset_view_limits()

    def _apply_aspect(self, ax) -> None:
        scale = float(getattr(self, "transverse_scale", 1.0) or 1.0)
        ax.set_aspect(scale if scale > 1.0 else 'equal', 'datalim')

    def reset_view_limits(self) -> None:
        """Reset zoom and pan limits to fit the complete optical system within view."""
        self.saved_limits.clear()
        self._force_reset_view = True
        if hasattr(self, 'lbl_measure_readout') and self.lbl_measure_readout:
            self.lbl_measure_readout.setText("View reset to fit entire optical system.")
        self.refresh_layout()

    def get_system_extents(self) -> Tuple[float, float, float, float]:
        """Compute (z_min, z_max, y_min, y_max) covering all optical surfaces and ray paths with 10% margin."""
        z_coords: List[float] = []
        y_coords: List[float] = []
        stack = self.surface_stack
        if stack and stack.surfaces:
            surfaces = stack.get_active_sequence() if hasattr(stack, 'get_active_sequence') else stack.surfaces
            for surf in surfaces:
                if not hasattr(surf, 'global_origin'):
                    stack._update_accumulated_positions()
                vz = float(surf.global_origin[2])
                vy = float(surf.global_origin[1])
                sd = float(getattr(surf, 'semi_diameter', 0.0) or 15.0)
                z_coords.extend([vz - 5.0, vz + 5.0])
                y_coords.extend([vy - sd, vy + sd])
        if self.rays:
            for r in self.rays[:100]:
                if hasattr(r, 'history'):
                    for h in r.history:
                        z_coords.append(float(h.get('z_global', 0.0)))
                        y_coords.append(float(h.get('y_global', 0.0)))
        if not z_coords:
            z_coords = [0.0, 100.0]
        if not y_coords:
            y_coords = [-20.0, 20.0]
        z_min, z_max = min(z_coords), max(z_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        dz = max(1.0, z_max - z_min)
        dy = max(1.0, y_max - y_min)
        return (z_min - 0.1 * dz, z_max + 0.1 * dz, y_min - 0.1 * dy, y_max + 0.1 * dy)

    def set_view_limits(self, z_min: float, z_max: float, y_min: Optional[float] = None, y_max: Optional[float] = None) -> None:
        """Explicitly set view coordinates for longitudinal and transverse axes."""
        if z_min >= z_max:
            z_max = z_min + 1.0
        if y_min is not None and y_max is not None and y_min >= y_max:
            y_max = y_min + 1.0

        self._force_reset_view = False
        if self.ax_rz is not None:
            self.ax_rz.set_xlim(z_min, z_max)
            if y_min is not None and y_max is not None:
                self.ax_rz.set_ylim(y_min, y_max)
            self.saved_limits["rz"] = (self.ax_rz.get_xlim(), self.ax_rz.get_ylim())

        if self.ax_xz is not None:
            self.ax_xz.set_xlim(z_min, z_max)
            if y_min is not None and y_max is not None:
                self.ax_xz.set_ylim(y_min, y_max)
            self.saved_limits["xz"] = (self.ax_xz.get_xlim(), self.ax_xz.get_ylim())

        if hasattr(self, 'lbl_measure_readout') and self.lbl_measure_readout:
            self.lbl_measure_readout.setText(f"View rescaled: Z=[{z_min:.2f}, {z_max:.2f}], Transverse=[{y_min:.2f}, {y_max:.2f}]")

        if hasattr(self.canvas, "draw_idle"):
            self.canvas.draw_idle()
        elif hasattr(self.canvas, "draw"):
            self.canvas.draw()

    def zoom_by(self, scale_factor: float, center: Optional[Tuple[float, float]] = None) -> None:
        """Zoom view in/out smoothly by scale_factor (<1 zooms in, >1 zooms out)."""
        self._force_reset_view = False
        for ax_name, ax in [("rz", self.ax_rz), ("xz", self.ax_xz)]:
            if ax is None:
                continue
            cur_xlim = ax.get_xlim()
            cur_ylim = ax.get_ylim()
            cx = (cur_xlim[0] + cur_xlim[1]) * 0.5 if center is None else center[0]
            cy = (cur_ylim[0] + cur_ylim[1]) * 0.5 if center is None else center[1]

            hw = (cur_xlim[1] - cur_xlim[0]) * 0.5 * scale_factor
            hh = (cur_ylim[1] - cur_ylim[0]) * 0.5 * scale_factor

            new_xlim = (cx - hw, cx + hw)
            new_ylim = (cy - hh, cy + hh)
            ax.set_xlim(new_xlim)
            ax.set_ylim(new_ylim)
            self.saved_limits[ax_name] = (new_xlim, new_ylim)

        if hasattr(self.canvas, "draw_idle"):
            self.canvas.draw_idle()
        elif hasattr(self.canvas, "draw"):
            self.canvas.draw()

    def zoom_in(self, factor: float = 1.25) -> None:
        """Zoom in optical system view."""
        self.zoom_by(1.0 / factor)

    def zoom_out(self, factor: float = 1.25) -> None:
        """Zoom out optical system view."""
        self.zoom_by(factor)

    def open_rescale_dialog(self) -> None:
        """Open interactive modal dialog to rescale limits and aspect ratio."""
        if not QT_AVAILABLE:
            return
        dlg = RescaleViewDialog(self, parent=self)
        dlg.exec_() if hasattr(dlg, "exec_") else dlg.show()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)

        self.fig = Figure(figsize=(10, 5), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self) if (NavigationToolbar and QT_AVAILABLE) else None
        top_bar = QHBoxLayout()
        top_bar.setContentsMargins(4, 2, 4, 2)
        top_bar.setSpacing(6)

        btn_style = (
            "QPushButton {"
            "  background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ffffff, stop:1 #f8fafc);"
            "  border: 1px solid #cbd5e1;"
            "  border-radius: 4px;"
            "  padding: 3px 9px;"
            "  font-size: 11px;"
            "  font-weight: bold;"
            "  color: #1e293b;"
            "}"
            "QPushButton:hover { background: #f1f5f9; border-color: #94a3b8; }"
            "QPushButton:pressed { background: #e2e8f0; }"
        )

        self.btn_mech_toggle = QPushButton("🔲 MECH: ON (Detector)")
        self.btn_mech_toggle.setToolTip("Toggle mechanism between ON (+12° → Detector) and OFF (-12° → Beam Dump)")
        self._update_mech_button_ui("ON")
        self.btn_mech_toggle.clicked.connect(self.toggle_mech_state)
        self.btn_dmd_toggle = self.btn_mech_toggle
        top_bar.addWidget(self.btn_mech_toggle)

        self.btn_color_mode = QPushButton("🎨 Wavelength")
        self.btn_color_mode.setStyleSheet(btn_style)
        self.btn_color_mode.setToolTip("Toggle ray coloring: by Wavelength (dispersion) or Field Index (multi-field)")
        self.btn_color_mode.clicked.connect(self.toggle_color_mode)
        top_bar.addWidget(self.btn_color_mode)

        self.btn_fit_system = QPushButton("🔍 Fit System")
        self.btn_fit_system.setStyleSheet(btn_style)
        self.btn_fit_system.setToolTip("Reset view limits to fit entire optical system (Ctrl+0 / F)")
        self.btn_fit_system.clicked.connect(self.reset_view_limits)
        top_bar.addWidget(self.btn_fit_system)

        self.btn_zoom_in = None
        self.btn_zoom_out = None
        self.btn_rescale_dialog = None

        from optics_app.gui.qt_compat import QDoubleSpinBox
        top_bar.addWidget(QLabel("Transverse ×:"))
        self.spin_transverse_scale = QDoubleSpinBox()
        self.spin_transverse_scale.setRange(1.0, 50.0)
        self.spin_transverse_scale.setSingleStep(0.5)
        self.spin_transverse_scale.setDecimals(1)
        self.spin_transverse_scale.setValue(self.transverse_scale)
        self.spin_transverse_scale.setToolTip(
            "Exaggerate the transverse (Y/X) axis relative to the optical (Z) axis so "
            "element diameters appear larger. 1.0 = true-to-scale."
        )
        self.spin_transverse_scale.valueChanged.connect(self.set_transverse_scale)
        top_bar.addWidget(self.spin_transverse_scale)

        cb_style = (
            "QCheckBox { font-size: 11px; font-weight: 500; color: #475569; spacing: 4px; } "
            "QCheckBox:hover { color: #0f172a; }"
        )
        self.cb_show_labels = QCheckBox("Labels")
        self.cb_show_labels.setStyleSheet(cb_style)
        self.cb_show_labels.setChecked(self.show_labels)
        self.cb_show_labels.setToolTip("Toggle optical element, stop, and callout labels in 2D projections.")
        self.cb_show_labels.toggled.connect(self.set_show_labels)
        top_bar.addWidget(self.cb_show_labels)

        self.cb_dark_mode = QCheckBox("Dark BG")
        self.cb_dark_mode.setStyleSheet(cb_style)
        self.cb_dark_mode.setChecked(self.dark_mode)
        self.cb_dark_mode.setToolTip("Toggle dark background design for optical layout views.")
        self.cb_dark_mode.toggled.connect(self.set_dark_mode)
        top_bar.addWidget(self.cb_dark_mode)

        self.cb_marginal_chief = QCheckBox("Marg/Chief")
        self.cb_marginal_chief.setStyleSheet(cb_style)
        self.cb_marginal_chief.setChecked(self.show_marginal_chief)
        self.cb_marginal_chief.setToolTip("Highlight Marginal (pupil edge Py=±1) and Chief (pupil center Py=0) rays (Ctrl+M)")
        self.cb_marginal_chief.toggled.connect(self.set_show_marginal_chief)
        top_bar.addWidget(self.cb_marginal_chief)

        self.cb_pupils_windows = QCheckBox("Pupils/Windows")
        self.cb_pupils_windows.setStyleSheet(cb_style)
        self.cb_pupils_windows.setChecked(self.show_pupils_windows)
        self.cb_pupils_windows.setToolTip("Overlay Entrance/Exit Pupils, Windows, and Stops on 2D layout")
        self.cb_pupils_windows.toggled.connect(self.set_show_pupils_windows)
        top_bar.addWidget(self.cb_pupils_windows)

        # 2D Projection View Toggles
        self.cb_view_rz = QCheckBox("R-Z")
        self.cb_view_rz.setStyleSheet(cb_style)
        self.cb_view_rz.setChecked(self.rz_view_active)
        self.cb_view_rz.setToolTip("Toggle Longitudinal (R-Z / Y-Z) 2D projection")
        self.cb_view_rz.toggled.connect(self.set_rz_view_active)
        top_bar.addWidget(self.cb_view_rz)

        self.cb_view_xz = QCheckBox("X-Z")
        self.cb_view_xz.setStyleSheet(cb_style)
        self.cb_view_xz.setChecked(self.xz_view_active)
        self.cb_view_xz.setToolTip("Toggle Transverse (X-Z) 2D projection")
        self.cb_view_xz.toggled.connect(self.set_xz_view_active)
        top_bar.addWidget(self.cb_view_xz)

        self.cb_view_front = QCheckBox("Front (X-Y)")
        self.cb_view_front.setStyleSheet(cb_style)
        self.cb_view_front.setChecked(self.xy_front_view_active)
        self.cb_view_front.setToolTip("Toggle Front View (X-Y) projected aperture plane (renders 2D pattern masks & beam footprints)")
        self.cb_view_front.toggled.connect(self.set_xy_front_view_active)
        top_bar.addWidget(self.cb_view_front)

        top_bar.addStretch()
        layout.addLayout(top_bar)

        if self.toolbar:
            layout.addWidget(self.toolbar)

        # View controls
        self.rz_view_active = getattr(self, "rz_view_active", True)
        self.xz_view_active = getattr(self, "xz_view_active", False)
        self.xy_front_view_active = getattr(self, "xy_front_view_active", False)

        layout.addWidget(self.canvas)

        self.ax_rz = None
        self.ax_xz = None
        self.ax_xy_front = None

        self.apply_tight_layout()

        self.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.canvas.mpl_connect('button_release_event', self.on_mouse_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.refresh_layout()

    def set_measure_mode(self, mode: Optional[str]):
        """Set active measurement mode: None (Pan/Zoom), 'distance', or 'angle'."""
        self.measure_mode = mode
        self.measure_points = []
        self.active_drag_handle = None

        if hasattr(self, 'btn_mode_navigate') and self.btn_mode_navigate:
            self.btn_mode_navigate.setChecked(mode is None)
        if hasattr(self, 'btn_mode_distance') and self.btn_mode_distance:
            self.btn_mode_distance.setChecked(mode == "distance")
        if hasattr(self, 'btn_mode_angle') and self.btn_mode_angle:
            self.btn_mode_angle.setChecked(mode == "angle")

        if mode == "distance":
            msg = "Distance Line Tool: Click Point 1 (or snap to surface), then click Point 2. Drag handles anytime to edit!"
        elif mode == "angle":
            msg = "Angle Tool (3 Pts / 2 Lines): Click Point 1 -> Click Point 2 (Apex) -> Click Point 3. Drag handles anytime to edit!"
        else:
            msg = "Pan / Zoom Mode: Left-drag to pan, scroll to zoom. Click any measurement handle to drag and edit."

        if hasattr(self, 'lbl_measure_readout') and self.lbl_measure_readout:
            self.lbl_measure_readout.setText(msg)

        self._emit_measurements_changed(msg)
        self.refresh_layout()

    def clear_measurements(self):
        """Clear all active measurement lines, angles, and overlays."""
        self.measure_points = []
        self.measurement_history = []
        self.active_drag_handle = None
        self.selected_measurement_idx = None
        msg = "All measurements cleared. Ready."
        if hasattr(self, 'lbl_measure_readout') and self.lbl_measure_readout:
            self.lbl_measure_readout.setText(msg)
        self._emit_measurements_changed(msg)
        self.refresh_layout()

    def _find_nearest_surface(self, z_click: float, y_click: float, max_dist: float = 3.0, event: Any = None, snap_pix: float = 14.0) -> Optional[Dict[str, Any]]:
        """Find the nearest surface feature (vertex or aperture rim) to mouse coordinates in R-Z longitudinal view."""
        if not getattr(self, "snap_enabled", True):
            return None
        stack = self.surface_stack
        if not stack or not stack.surfaces:
            return None

        surfaces = stack.get_active_sequence() if hasattr(stack, 'get_active_sequence') else stack.surfaces
        best_item = None
        best_score = float('inf')
        ax = getattr(event, 'inaxes', None) if event is not None else getattr(self, 'ax_rz', None)

        for i, surf in enumerate(surfaces):
            if not hasattr(surf, 'global_origin'):
                stack._update_accumulated_positions()

            eff_orig, eff_R = get_effective_pose(surf)
            vz = float(eff_orig[2])
            vy = float(eff_orig[1])
            features = [("Vertex", (vz, vy))]

            sd = float(getattr(surf, 'semi_diameter', 0.0))
            if sd > 0:
                sag_val = 0.0
                if hasattr(surf, 'sag') and surf.surface_type != SurfaceType.CAD:
                    try:
                        sag_val = float(surf.sag(sd))
                    except Exception:
                        pass
                top_glob = eff_orig + eff_R @ np.array([0.0, sd, sag_val])
                bot_glob = eff_orig + eff_R @ np.array([0.0, -sd, sag_val])
                features.append(("Top Rim", (float(top_glob[2]), float(top_glob[1]))))
                features.append(("Bottom Rim", (float(bot_glob[2]), float(bot_glob[1]))))

            for feat_name, (fz, fy) in features:
                d_data = float(np.sqrt((z_click - fz)**2 + (y_click - fy)**2))
                is_hit = False
                hit_score = d_data

                if ax is not None and hasattr(ax, 'transData') and getattr(event, 'x', None) is not None:
                    try:
                        feat_pix = ax.transData.transform((fz, fy))
                        p_dist = float(np.hypot(event.x - feat_pix[0], event.y - feat_pix[1]))
                        if p_dist <= snap_pix:
                            is_hit = True
                            hit_score = p_dist
                    except Exception:
                        pass
                elif d_data <= max_dist:
                    is_hit = True
                    hit_score = d_data * 10.0

                if is_hit and hit_score < best_score:
                    best_score = hit_score
                    best_item = {
                        "surface_index": i,
                        "surface": surf,
                        "name": surf.name,
                        "vertex": (fz, fy),
                        "feature": feat_name,
                        "normal": eff_R[:, 2]
                    }

        return best_item

    def _find_handle_near(self, z_mouse: float, y_mouse: float, max_dist: float = 8.0, event: Any = None) -> Optional[Dict[str, Any]]:
        """Find if mouse is clicking/hovering near an existing measurement control handle in screen pixel space."""
        ax = getattr(event, 'inaxes', None) if event is not None else getattr(self, 'ax_rz', None)
        for i, m in enumerate(self.measurement_history):
            # Check individual points
            for key in ["p1", "p2", "apex"]:
                if key in m:
                    pt = m[key]
                    d_data = float(np.sqrt((z_mouse - pt[0])**2 + (y_mouse - pt[1])**2))
                    is_hit = False
                    if ax is not None and hasattr(ax, 'transData') and getattr(event, 'x', None) is not None:
                        try:
                            pt_pix = ax.transData.transform(pt)
                            pix_dist = float(np.hypot(event.x - pt_pix[0], event.y - pt_pix[1]))
                            if pix_dist <= 20.0:
                                is_hit = True
                        except Exception:
                            pass
                    elif d_data <= max_dist:
                        is_hit = True
                    if is_hit:
                        return {"idx": i, "handle": key, "coord": pt, "type": m.get("type"), "pix_dist": d_data}
                        
            # Check body dragging (clicking near lines)
            if m.get("type") in ("angle", "3point_angle") and "apex" in m and "p1" in m and "p2" in m:
                # Distance to segment apex->p1 and apex->p2
                p_mouse = np.array([z_mouse, y_mouse])
                for p_end in [m["p1"], m["p2"]]:
                    a = np.array(m["apex"])
                    b = np.array(p_end)
                    ab = b - a
                    l2 = np.dot(ab, ab)
                    if l2 > 1e-6:
                        t = np.clip(np.dot(p_mouse - a, ab) / l2, 0.0, 1.0)
                        proj = a + t * ab
                        d_line = float(np.linalg.norm(p_mouse - proj))
                        if d_line <= 3.0:
                            return {"idx": i, "handle": "body", "start_mouse": [z_mouse, y_mouse], "type": m.get("type")}
                            
            elif m.get("type") == "distance" and "p1" in m and "p2" in m:
                p_mouse = np.array([z_mouse, y_mouse])
                a = np.array(m["p1"])
                b = np.array(m["p2"])
                ab = b - a
                l2 = np.dot(ab, ab)
                if l2 > 1e-6:
                    t = np.clip(np.dot(p_mouse - a, ab) / l2, 0.0, 1.0)
                    proj = a + t * ab
                    d_line = float(np.linalg.norm(p_mouse - proj))
                    if d_line <= 3.0:
                        return {"idx": i, "handle": "body", "start_mouse": [z_mouse, y_mouse], "type": m.get("type")}
        return None

    def on_mouse_press(self, event):
        """Handle mouse clicks on matplotlib canvas for drawing, editing, and panning."""
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return

        # 1. Middle-click or Right-click to pan freely at any time
        if getattr(event, 'button', 1) in (2, 3):
            ax = event.inaxes
            self._pan_start = {
                "ax": ax,
                "x_start": event.x,
                "y_start": event.y,
                "xlim_start": ax.get_xlim(),
                "ylim_start": ax.get_ylim()
            }
            return

        z_pt = float(event.xdata)
        y_pt = float(event.ydata)

        # 2. Check if user clicked an existing edit handle to drag/adjust
        handle_hit = self._find_handle_near(z_pt, y_pt, event=event)
        if handle_hit is not None:
            self.active_drag_handle = handle_hit
            self.selected_measurement_idx = handle_hit["idx"]
            m = self.measurement_history[handle_hit["idx"]]
            h_name = handle_hit["handle"].upper()
            if hasattr(self, 'lbl_measure_readout') and self.lbl_measure_readout:
                if m.get("type") == "distance":
                    self.lbl_measure_readout.setText(f"Editing Distance Handle ({h_name}): Drag mouse to reposition point")
                else:
                    self.lbl_measure_readout.setText(f"Editing Angle Handle ({h_name}): Drag mouse to reposition point")
            self.refresh_layout()
            return

        if self.measure_mode is None:
            # Check if user clicked near any measurement to select it
            handle_hit = self._find_handle_near(z_pt, y_pt, max_dist=20.0, event=event)
            if handle_hit is not None:
                self.selected_measurement_idx = handle_hit["idx"]
                self.refresh_layout()
            return

        near_surf = self._find_nearest_surface(z_pt, y_pt, max_dist=8.0, event=event)

        # If snapped to surface, use surface feature coordinates
        if near_surf:
            pt_coord = [float(near_surf["vertex"][0]), float(near_surf["vertex"][1])]
            feat_tag = f" [{near_surf['feature']}]" if near_surf.get('feature') != 'Vertex' else ""
            label = f"S{near_surf['surface_index']} ({near_surf['name']}){feat_tag}"
        else:
            pt_coord = [z_pt, y_pt]
            label = f"({z_pt:.2f}, {y_pt:.2f})"

        pt_data = {
            "coord": pt_coord,
            "label": label,
            "surface_info": near_surf
        }
        self.measure_points.append(pt_data)

        if self.measure_mode == "distance":
            if len(self.measure_points) == 1:
                if hasattr(self, 'lbl_measure_readout') and self.lbl_measure_readout:
                    self.lbl_measure_readout.setText(f"Point 1 set at {label}. Click Point 2 to draw distance line...")
                self.refresh_layout()
            elif len(self.measure_points) >= 2:
                p1 = self.measure_points[0]["coord"]
                p2 = self.measure_points[1]["coord"]
                dz = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                dist = float(np.sqrt(dz**2 + dy**2))
                angle_deg = float(np.degrees(np.arctan2(dy, dz)))

                measurement = {
                    "type": "distance",
                    "p1": [p1[0], p1[1]],
                    "p2": [p2[0], p2[1]],
                    "label1": self.measure_points[0]["label"],
                    "label2": self.measure_points[1]["label"],
                    "distance": dist,
                    "dz": abs(dz),
                    "dy": abs(dy),
                    "angle": angle_deg
                }
                self.measurement_history.append(measurement)
                self.selected_measurement_idx = len(self.measurement_history) - 1
                self.measure_points = []
                msg = f"📏 Distance = {dist:.3f} mm (ΔZ={abs(dz):.3f}, ΔY={abs(dy):.3f}, θ={angle_deg:.1f}°) - Drag handles to edit"
                if hasattr(self, 'lbl_measure_readout') and self.lbl_measure_readout:
                    self.lbl_measure_readout.setText(msg)
                self._emit_measurements_changed(msg)
                self.refresh_layout()

        elif self.measure_mode == "angle":
            if len(self.measure_points) == 1:
                msg = f"Point 1 set at {label}. Click Point 2 (Apex / Vertex intersection)..."
                if hasattr(self, 'lbl_measure_readout') and self.lbl_measure_readout:
                    self.lbl_measure_readout.setText(msg)
                self._emit_measurements_changed(msg)
                self.refresh_layout()
            elif len(self.measure_points) == 2:
                msg = f"Apex set at {label}. Click Point 3 (End of second line) to complete angle..."
                if hasattr(self, 'lbl_measure_readout') and self.lbl_measure_readout:
                    self.lbl_measure_readout.setText(msg)
                self._emit_measurements_changed(msg)
                self.refresh_layout()
            elif len(self.measure_points) >= 3:
                p1 = self.measure_points[0]["coord"]
                apex = self.measure_points[1]["coord"]
                p2 = self.measure_points[2]["coord"]

                v1 = np.array(p1) - np.array(apex)
                v2 = np.array(p2) - np.array(apex)
                len1 = np.linalg.norm(v1)
                len2 = np.linalg.norm(v2)
                if len1 > 1e-9 and len2 > 1e-9:
                    dot = float(np.clip(np.dot(v1, v2) / (len1 * len2), -1.0, 1.0))
                    angle_deg = float(np.degrees(np.arccos(dot)))
                else:
                    angle_deg = 0.0

                measurement = {
                    "type": "angle",
                    "p1": [p1[0], p1[1]],
                    "apex": [apex[0], apex[1]],
                    "p2": [p2[0], p2[1]],
                    "label1": self.measure_points[0]["label"],
                    "label_apex": self.measure_points[1]["label"],
                    "label2": self.measure_points[2]["label"],
                    "angle": angle_deg
                }
                self.measurement_history.append(measurement)
                self.selected_measurement_idx = len(self.measurement_history) - 1
                self.measure_points = []
                msg = f"📐 Angle = {angle_deg:.2f}° (2 lines, 3 points) - Drag handles to edit"
                if hasattr(self, 'lbl_measure_readout') and self.lbl_measure_readout:
                    self.lbl_measure_readout.setText(msg)
                self._emit_measurements_changed(msg)
                self.refresh_layout()

    def on_mouse_move(self, event):
        """Handle mouse movement for interactive panning, dragging handles, and live snap reticles."""
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return

        # 1. Interactive right-click / middle-click panning
        if getattr(self, '_pan_start', None) is not None:
            p = self._pan_start
            ax = p["ax"]
            dx_pix = event.x - p["x_start"]
            dy_pix = event.y - p["y_start"]
            try:
                inv = ax.transData.inverted()
                p0 = inv.transform((0, 0))
                p_disp = inv.transform((dx_pix, dy_pix))
                dx_data = p_disp[0] - p0[0]
                dy_data = p_disp[1] - p0[1]
                new_xlim = (p["xlim_start"][0] - dx_data, p["xlim_start"][1] - dx_data)
                new_ylim = (p["ylim_start"][0] - dy_data, p["ylim_start"][1] - dy_data)
                ax.set_xlim(new_xlim)
                ax.set_ylim(new_ylim)
                if ax == self.ax_rz:
                    self.saved_limits["rz"] = (new_xlim, new_ylim)
                elif ax == self.ax_xz:
                    self.saved_limits["xz"] = (new_xlim, new_ylim)
                if hasattr(self.canvas, "draw_idle"):
                    self.canvas.draw_idle()
            except Exception:
                pass
            return

        z_pt = float(event.xdata)
        y_pt = float(event.ydata)
        self.mouse_cursor_pos = (z_pt, y_pt)

        # 2. If dragging an existing handle or the entire measurement body
        if self.active_drag_handle is not None:
            idx = self.active_drag_handle["idx"]
            h_key = self.active_drag_handle["handle"]
            if 0 <= idx < len(self.measurement_history):
                m = self.measurement_history[idx]
                
                # Check for surface snap ONLY when within tight pixel threshold (10px)
                near_surf = self._find_nearest_surface(z_pt, y_pt, max_dist=2.0, event=event, snap_pix=10.0)
                if near_surf:
                    target_pt = [float(near_surf["vertex"][0]), float(near_surf["vertex"][1])]
                else:
                    target_pt = [z_pt, y_pt]

                if h_key == "body":
                    # Drag the entire measurement as a rigid body across the plot
                    start_m = self.active_drag_handle.get("start_mouse", [z_pt, y_pt])
                    dz = z_pt - start_m[0]
                    dy = y_pt - start_m[1]
                    self.active_drag_handle["start_mouse"] = [z_pt, y_pt]
                    for k in ["p1", "p2", "apex"]:
                        if k in m:
                            m[k] = [float(m[k][0] + dz), float(m[k][1] + dy)]
                else:
                    # Drag specific point handle freely to any position on the plot
                    m[h_key] = target_pt

                # Recompute metrics dynamically
                if m.get("type") == "distance":
                    p1 = m["p1"]
                    p2 = m["p2"]
                    dz = p2[0] - p1[0]
                    dy = p2[1] - p1[1]
                    dist = float(np.sqrt(dz**2 + dy**2))
                    ang = float(np.degrees(np.arctan2(dy, dz)))
                    m["distance"] = dist
                    m["dz"] = abs(dz)
                    m["dy"] = abs(dy)
                    m["angle"] = ang
                    msg = f"📏 Distance = {dist:.3f} mm (ΔZ={abs(dz):.3f}, ΔY={abs(dy):.3f}, θ={ang:.1f}°)"
                    if hasattr(self, 'lbl_measure_readout') and self.lbl_measure_readout:
                        self.lbl_measure_readout.setText(msg)
                    self._emit_measurements_changed(msg)
                elif m.get("type") in ("angle", "3point_angle", "surface_angle"):
                    p1 = m["p1"]
                    apex = m.get("apex", m["p1"])
                    p2 = m["p2"]
                    v1 = np.array(p1) - np.array(apex)
                    v2 = np.array(p2) - np.array(apex)
                    l1 = float(np.linalg.norm(v1))
                    l2 = float(np.linalg.norm(v2))
                    if l1 > 1e-9 and l2 > 1e-9:
                        dot = float(np.clip(np.dot(v1, v2) / (l1 * l2), -1.0, 1.0))
                        ang = float(np.degrees(np.arccos(dot)))
                        theta1 = float(np.degrees(np.arctan2(v1[1], v1[0])))
                        theta2 = float(np.degrees(np.arctan2(v2[1], v2[0])))
                    else:
                        ang = 0.0
                        theta1 = 0.0
                        theta2 = 0.0
                    m["angle"] = ang
                    msg = f"📐 Angle = {ang:.2f}° | Ray 1 ({l1:.2f} mm, θ1={theta1:.1f}°) | Ray 2 ({l2:.2f} mm, θ2={theta2:.1f}°) | Apex = ({apex[0]:.2f}, {apex[1]:.2f})"
                    if hasattr(self, 'lbl_measure_readout') and self.lbl_measure_readout:
                        self.lbl_measure_readout.setText(msg)
                    self._emit_measurements_changed(msg)

                self.refresh_layout()
            return

        # 3. Check handle hover state
        prev_hover = self.hover_drag_handle
        self.hover_drag_handle = self._find_handle_near(z_pt, y_pt, max_dist=8.0, event=event)

        # 4. Check surface snap target hover
        prev_snap = self.hover_snap_target
        if self.measure_mode in ("distance", "angle"):
            self.hover_snap_target = self._find_nearest_surface(z_pt, y_pt, max_dist=2.0, event=event, snap_pix=10.0)
        else:
            self.hover_snap_target = None

        if prev_hover != self.hover_drag_handle or prev_snap != self.hover_snap_target:
            self.refresh_layout()

    def on_mouse_release(self, event):
        """Finish dragging measurement handle or panning."""
        if getattr(self, '_pan_start', None) is not None:
            self._pan_start = None
        if self.active_drag_handle is not None:
            self.active_drag_handle = None
            self.refresh_layout()

    def on_key_press(self, event):
        """Handle keyboard shortcuts (Delete/Backspace to remove measurement, Escape to cancel, F/Home to fit, +/- to zoom)."""
        key = str(getattr(event, 'key', '') or '').lower()
        if key in ("delete", "backspace"):
            if self.selected_measurement_idx is not None and 0 <= self.selected_measurement_idx < len(self.measurement_history):
                self.measurement_history.pop(self.selected_measurement_idx)
                self.selected_measurement_idx = None
                if hasattr(self, 'lbl_measure_readout') and self.lbl_measure_readout:
                    self.lbl_measure_readout.setText("Selected measurement deleted.")
                self.refresh_layout()
        elif key == "escape":
            self.measure_points = []
            self.active_drag_handle = None
            if hasattr(self, 'lbl_measure_readout') and self.lbl_measure_readout:
                self.lbl_measure_readout.setText("Cancelled drawing.")
            self.refresh_layout()
        elif key in ("f", "home", "."):
            self.reset_view_limits()
        elif key in ("+", "=", "plus", "kp_add"):
            self.zoom_in()
        elif key in ("-", "_", "minus", "kp_subtract"):
            self.zoom_out()
        elif key in ("1", "kp_1"):
            self.set_transverse_scale(1.0)
            if hasattr(self, 'spin_transverse_scale') and self.spin_transverse_scale:
                self.spin_transverse_scale.setValue(1.0)
        elif key in ("r", "ctrl+r"):
            self.open_rescale_dialog()

    def keyPressEvent(self, event) -> None:
        """Handle Qt keyboard events for optical system view navigation."""
        if not QT_AVAILABLE or event is None:
            return
        k = event.key() if hasattr(event, "key") else 0
        modifiers = event.modifiers() if hasattr(event, "modifiers") else None
        is_ctrl = bool(modifiers & Qt.ControlModifier) if (modifiers is not None and Qt is not None) else False

        if k in (getattr(Qt, 'Key_F', 70), getattr(Qt, 'Key_Home', 0x01000010)):
            self.reset_view_limits()
            event.accept()
            return
        elif k in (getattr(Qt, 'Key_Plus', 43), getattr(Qt, 'Key_Equal', 61)):
            self.zoom_in()
            event.accept()
            return
        elif k in (getattr(Qt, 'Key_Minus', 45), getattr(Qt, 'Key_Underscore', 95)):
            self.zoom_out()
            event.accept()
            return
        elif k == getattr(Qt, 'Key_1', 49):
            self.set_transverse_scale(1.0)
            if hasattr(self, 'spin_transverse_scale') and self.spin_transverse_scale:
                self.spin_transverse_scale.setValue(1.0)
            event.accept()
            return
        elif k == getattr(Qt, 'Key_R', 82) and is_ctrl:
            self.open_rescale_dialog()
            event.accept()
            return

        super_kpe = getattr(super(), 'keyPressEvent', None)
        if callable(super_kpe):
            super_kpe(event)

    def resizeEvent(self, event) -> None:
        """Dynamically rescale and adapt optical view layout to the window/dock geometry."""
        super_re = getattr(super(), 'resizeEvent', None)
        if callable(super_re):
            super_re(event)
        self._on_widget_resized()

    def _on_widget_resized(self) -> None:
        """Handle window/widget resize: adapt layout, refresh tight bounds, and scale view."""
        if not hasattr(self, 'fig') or self.fig is None:
            return
        try:
            self.apply_tight_layout()
            if not self.saved_limits or getattr(self, '_force_reset_view', False):
                if self.ax_rz is not None:
                    self._apply_aspect(self.ax_rz)
                if self.ax_xz is not None:
                    self._apply_aspect(self.ax_xz)
            if hasattr(self, 'canvas') and self.canvas is not None:
                if hasattr(self.canvas, "draw_idle"):
                    self.canvas.draw_idle()
                elif hasattr(self.canvas, "draw"):
                    self.canvas.draw()
        except Exception:
            pass

    def set_rz_view_active(self, active: bool) -> None:
        if not active and not self.xz_view_active and not self.xy_front_view_active:
            if hasattr(self, 'cb_view_rz') and self.cb_view_rz:
                self.cb_view_rz.blockSignals(True)
                self.cb_view_rz.setChecked(True)
                self.cb_view_rz.blockSignals(False)
            return
        self.rz_view_active = active
        if hasattr(self, 'cb_view_rz') and self.cb_view_rz and self.cb_view_rz.isChecked() != active:
            self.cb_view_rz.blockSignals(True)
            self.cb_view_rz.setChecked(active)
            self.cb_view_rz.blockSignals(False)
        self.refresh_layout()

    def set_xz_view_active(self, active: bool) -> None:
        if not active and not self.rz_view_active and not self.xy_front_view_active:
            if hasattr(self, 'cb_view_xz') and self.cb_view_xz:
                self.cb_view_xz.blockSignals(True)
                self.cb_view_xz.setChecked(True)
                self.cb_view_xz.blockSignals(False)
            return
        self.xz_view_active = active
        if hasattr(self, 'cb_view_xz') and self.cb_view_xz and self.cb_view_xz.isChecked() != active:
            self.cb_view_xz.blockSignals(True)
            self.cb_view_xz.setChecked(active)
            self.cb_view_xz.blockSignals(False)
        self.refresh_layout()

    def set_xy_front_view_active(self, active: bool) -> None:
        if not active and not self.rz_view_active and not self.xz_view_active:
            if hasattr(self, 'cb_view_front') and self.cb_view_front:
                self.cb_view_front.blockSignals(True)
                self.cb_view_front.setChecked(True)
                self.cb_view_front.blockSignals(False)
            return
        self.xy_front_view_active = active
        self.xy_view_active = active
        if hasattr(self, 'cb_view_front') and self.cb_view_front and self.cb_view_front.isChecked() != active:
            self.cb_view_front.blockSignals(True)
            self.cb_view_front.setChecked(active)
            self.cb_view_front.blockSignals(False)
        self.refresh_layout()

    set_xy_view_active = set_xy_front_view_active

    def on_view_toggled(self):
        pass

    def on_scroll(self, event):
        if event.inaxes is None:
            return
        ax = event.inaxes
        base_scale = 1.2
        if event.button == 'up':
            scale_factor = 1 / base_scale
        elif event.button == 'down':
            scale_factor = base_scale
        else:
            scale_factor = 1

        xdata, ydata = event.xdata, event.ydata
        if xdata is None or ydata is None:
            return

        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()

        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

        relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
        rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])

        new_xlim = (xdata - new_width * (1 - relx), xdata + new_width * relx)
        new_ylim = (ydata - new_height * (1 - rely), ydata + new_height * rely)

        ax.set_xlim(new_xlim)
        ax.set_ylim(new_ylim)
        if ax == self.ax_rz:
            self.saved_limits["rz"] = (new_xlim, new_ylim)
        elif ax == self.ax_xz:
            self.saved_limits["xz"] = (new_xlim, new_ylim)

        if hasattr(self.canvas, "draw_idle"):
            self.canvas.draw_idle()
        else:
            self.canvas.draw()

    def refresh_layout(self) -> bool:
        """
        Refreshes the dual-view plots from stored model or surface stack.
        Returns True on successful refresh.
        """
        stack = None
        if self.model is not None:
            if hasattr(self.model, "get_surface_stack"):
                try:
                    stack = self.model.get_surface_stack()
                except Exception:
                    pass
            elif hasattr(self.model, "stack"):
                stack = self.model.stack
            elif hasattr(self.model, "_data"):
                stack = SurfaceStack()
                for row in getattr(self.model, "_data", []):
                    try:
                        surf = Surface(
                            name=str(row[0]),
                            surface_type=str(row[1]),
                            radius=float(row[2]),
                            thickness=float(row[3]),
                            glass=str(row[4]),
                            semi_diameter=float(row[5]),
                            conic=float(row[6]),
                            decenter_x=float(row[7]),
                            decenter_y=float(row[8]),
                            tilt_x=float(row[9]),
                            tilt_y=float(row[10]),
                            tilt_z=float(row[11])
                        )
                        stack.add_surface(surf)
                    except (ValueError, IndexError):
                        continue
            elif isinstance(self.model, SurfaceStack):
                stack = self.model

        if stack is None:
            stack = self.surface_stack

        self.update_layout(surface_stack=stack, rays=self.rays)
        return True

    def _render_rays_and_annotations_rz(self, stack, surfaces) -> None:
        if hasattr(self, "_ray_artists_rz"):
            for art in self._ray_artists_rz:
                try:
                    art.remove()
                except Exception:
                    pass
            self._ray_artists_rz.clear()
        else:
            self._ray_artists_rz = []

        if not self.rays or self.ax_rz is None:
            return

        is_dark = getattr(self, "dark_mode", False)
        show_mc = getattr(self, "show_marginal_chief", False)
        mc_only = getattr(self, "marginal_chief_only", False)
        has_marginal_drawn = False
        has_chief_drawn = False
        marginal_col = '#38bdf8' if is_dark else '#0284c7'
        chief_col = '#fb7185' if is_dark else '#dc2626'

        unique_wls = sorted(list(set(round(getattr(r, "wavelength", 0.5876), 4) for r in self.rays)))
        palette = (
            ['#38bdf8', '#4ade80', '#f87171', '#fbbf24', '#c084fc', '#fb923c', '#f472b6']
            if is_dark else
            ['#1f77b4', '#2ca02c', '#d62728', '#ff7f0e', '#9467bd', '#8c564b', '#e377c2']
        )
        field_palette = (
            ['#60a5fa', '#4ade80', '#f87171', '#c084fc', '#fb923c', '#22d3ee', '#facc15', '#fb7185']
            if is_dark else FIELD_COLORS
        )
        segments_by_style = {}

        for ray in self.rays:
            if hasattr(ray, "history") and len(ray.history) > 0:
                rz_z = [h.get("z_global", h.get("z", 0.0)) for h in ray.history]
                rz_y = [h.get("y_global", h.get("y", 0.0)) for h in ray.history]

                if len(rz_z) < 2:
                    continue

                is_chief, is_marginal = self._classify_ray(ray)
                if mc_only and not (is_chief or is_marginal):
                    continue

                is_vig = getattr(ray, "vignetted", False)

                if (show_mc or mc_only) and is_marginal:
                    ray_color = marginal_col
                    alpha = 0.95 if not is_vig else 0.50
                    ls = '-' if not is_vig else '--'
                    lw = 2.4 if not is_vig else 1.4
                    zorder = 8
                    has_marginal_drawn = True
                elif (show_mc or mc_only) and is_chief:
                    ray_color = chief_col
                    alpha = 0.95 if not is_vig else 0.50
                    ls = '-' if not is_vig else '--'
                    lw = 2.4 if not is_vig else 1.4
                    zorder = 9
                    has_chief_drawn = True
                else:
                    if getattr(self, "color_mode", "wavelength") == "field":
                        f_idx = int(getattr(ray, "field_idx", 0))
                        ray_color = field_palette[f_idx % len(field_palette)]
                    else:
                        wl = round(getattr(ray, "wavelength", 0.5876), 4)
                        wl_idx = unique_wls.index(wl) if wl in unique_wls else 0
                        ray_color = palette[wl_idx % len(palette)]

                    if show_mc:
                        alpha = 0.28 if not is_vig else 0.15
                        ls = '-' if not is_vig else '--'
                        lw = 0.9 if not is_vig else 0.6
                        zorder = 5
                    else:
                        alpha = 0.85 if not is_vig else 0.4
                        ls = '-' if not is_vig else '--'
                        lw = 1.3 if not is_vig else 0.9
                        zorder = 5

                style_key = (ray_color, alpha, ls, lw, zorder)
                if style_key not in segments_by_style:
                    segments_by_style[style_key] = []
                segments_by_style[style_key].append(np.column_stack([rz_z, rz_y]))

        for (color, alpha, ls, lw, zorder), segs in segments_by_style.items():
            lc = LineCollection(segs, colors=color, alpha=alpha, linestyles=ls, linewidths=lw, zorder=zorder)
            self.ax_rz.add_collection(lc)
            self._ray_artists_rz.append(lc)

        if (show_mc or mc_only) and (has_marginal_drawn or has_chief_drawn) and getattr(self, "show_labels", True):
            from matplotlib.lines import Line2D
            handles = []
            if has_marginal_drawn:
                handles.append(Line2D([0], [0], color=marginal_col, lw=2.4, label='Marginal Ray (Py=±1.0)'))
            if has_chief_drawn:
                handles.append(Line2D([0], [0], color=chief_col, lw=2.4, label='Chief Ray (Py=0.0)'))
            if handles:
                leg = self.ax_rz.legend(
                    handles=handles,
                    loc='upper right',
                    fontsize=8.5,
                    framealpha=0.88,
                    facecolor='#1e293b' if is_dark else '#ffffff',
                    edgecolor='#475569' if is_dark else '#cbd5e1',
                    labelcolor='#f1f5f9' if is_dark else '#0f172a'
                )
                leg.set_zorder(15)
                self._ray_artists_rz.append(leg)

        # Render Ray Incidence & Output Angles for Primary Wavelength
        if getattr(self, 'show_incidence_angles', True) and self.rays and stack is not None:
            main_wl = stack.primary_wavelength if hasattr(stack, 'primary_wavelength') and stack.primary_wavelength else getattr(self.rays[0], 'wavelength', 0.5876)
            cand_rays = [r for r in self.rays if abs(getattr(r, 'wavelength', 0.0) - main_wl) < 1e-4 and not getattr(r, 'vignetted', False)]
            main_ray = cand_rays[len(cand_rays)//2] if cand_rays else ([r for r in self.rays if not getattr(r, 'vignetted', False)] or self.rays)[0]

            if hasattr(main_ray, 'history') and len(main_ray.history) >= 3:
                for h_idx in range(1, len(main_ray.history) - 1):
                    h_cur = main_ray.history[h_idx]
                    s_idx = h_cur.get("surface_index", h_idx)
                    if s_idx < len(surfaces):
                        surf = surfaces[s_idx]
                        if bool(getattr(surf, "is_air_space", False)):
                            continue
                        is_refl = (hasattr(surf, 'is_reflective') and surf.is_reflective()) or \
                                  surf.surface_type in (SurfaceType.MIRROR, SurfaceType.GRATING) or \
                                  "DMD" in (surf.name or "").upper() or \
                                  str(surf.glass).upper() in ('MIRROR', 'REFL') or \
                                  getattr(surf, 'grating_lines_per_mm', 0) > 0

                        if is_refl:
                            h_prev = main_ray.history[h_idx - 1]

                            p_hit = np.array([h_cur.get("z_global", 0.0), h_cur.get("y_global", 0.0)])
                            v_in = np.array([h_prev.get("dir_z", 0.0), h_prev.get("dir_y", 0.0)])
                            v_out = np.array([h_cur.get("dir_z", 0.0), h_cur.get("dir_y", 0.0)])
                            len_in = np.linalg.norm(v_in)
                            len_out = np.linalg.norm(v_out)

                            if len_in > 1e-6 and len_out > 1e-6:
                                u_in = v_in / len_in
                                u_out = v_out / len_out

                                # Surface Normal vector in 2D (Z, Y)
                                try:
                                    eff_orig, eff_R = get_effective_pose(surf)
                                    norm_3d = eff_R[:, 2]
                                except Exception:
                                    norm_3d = surf.global_R[:, 2]
                                n_vec = np.array([float(norm_3d[2]), float(norm_3d[1])])
                                n_len = np.linalg.norm(n_vec)
                                if n_len > 1e-6:
                                    n_vec = n_vec / n_len
                                else:
                                    n_vec = np.array([1.0, 0.0])

                                if np.dot(n_vec, -u_in) < 0:
                                    n_vec = -n_vec

                                cos_in = np.clip(abs(np.dot(u_in, n_vec)), 0.0, 1.0)
                                cos_out = np.clip(abs(np.dot(u_out, n_vec)), 0.0, 1.0)
                                ang_in = np.degrees(np.arccos(cos_in))
                                ang_out = np.degrees(np.arccos(cos_out))
                                fold_ang = np.degrees(np.arccos(np.clip(np.dot(u_in, u_out), -1.0, 1.0)))

                                norm_len = 14.0
                                p_norm_end = p_hit + n_vec * norm_len
                                l_norm = self.ax_rz.plot([p_hit[0], p_norm_end[0]], [p_hit[1], p_norm_end[1]], color='#00ACC1', linestyle='--', linewidth=1.5, zorder=9)
                                p_norm = self.ax_rz.plot([p_hit[0]], [p_hit[1]], marker='o', markersize=5, color='#00ACC1', zorder=10)
                                self._ray_artists_rz.extend(l_norm)
                                self._ray_artists_rz.extend(p_norm)

                                is_grating = (surf.surface_type == SurfaceType.GRATING) or (getattr(surf, 'grating_lines_per_mm', 0) > 0)
                                is_mech = bool(available_states(surf))

                                if is_grating:
                                    badge_txt = f"S{s_idx} ({surf.name}): α={ang_in:.1f}°, β={ang_out:.1f}°"
                                elif is_mech:
                                    from optics_app.engine.mechanisms import active_state_name
                                    mech_st = active_state_name(surf)
                                    badge_txt = f"S{s_idx} (MECH [{mech_st}]): θ_in={ang_in:.1f}°, θ_out={ang_out:.1f}°"
                                else:
                                    badge_txt = f"S{s_idx} ({surf.name}): θ_in={ang_in:.1f}°, θ_out={ang_out:.1f}° (Fold={fold_ang:.1f}°)"

                                if getattr(self, "show_labels", True):
                                    offset_y = 18 if n_vec[1] >= 0 else -24
                                    badge_fc = '#1e293b' if is_dark else '#FFF9C4'
                                    badge_ec = '#38bdf8' if is_dark else '#FFB300'
                                    badge_tc = '#e2e8f0' if is_dark else '#0D47A1'
                                    ann = self.ax_rz.annotate(
                                        badge_txt,
                                        xy=(p_hit[0], p_hit[1]),
                                        xytext=(0, offset_y),
                                        textcoords="offset points",
                                        ha='center',
                                        va='bottom' if offset_y > 0 else 'top',
                                        fontsize=8.5,
                                        fontweight='bold',
                                        color=badge_tc,
                                        bbox=dict(boxstyle='round,pad=0.28', facecolor=badge_fc, edgecolor=badge_ec, alpha=0.95, linewidth=1.3),
                                        arrowprops=dict(arrowstyle="->", color=badge_ec, lw=1.2),
                                        zorder=12
                                    )
                                    self._ray_artists_rz.append(ann)

        mech_info = self.get_mech_surface_info()
        if mech_info and self.rays and getattr(self, "show_labels", True):
            from optics_app.engine.mechanisms import active_state
            mech_surface = mech_info[1]
            route = str(active_state(mech_surface).get("route", "") or "")
            active_surfaces = self.surface_stack.get_active_sequence() if self.surface_stack else []
            active_index = next((i for i, surface in enumerate(active_surfaces)
                                 if surface is mech_surface), None)
            if route and active_index is not None:
                hit = next((entry for ray in self.rays for entry in ray.history
                            if entry.get("surface_index") == active_index), None)
                if hit is not None:
                    tz = float(hit.get("z_global", 0.0))
                    ty = float(hit.get("y_global", 0.0))
                    txt = self.ax_rz.text(
                        tz, ty, f" MECH {mech_info[2]}: configured route → {route} ",
                        color="#38bdf8" if is_dark else "#155e75", fontsize=8.5, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.25", facecolor="#1e293b" if is_dark else "#ecfeff",
                                  edgecolor="#0891b2", alpha=0.92, linewidth=1.1),
                        ha="left", va="bottom", zorder=15,
                    )
                    self._ray_artists_rz.append(txt)

        # Render Pupils and Windows Overlays
        if getattr(self, "show_pupils_windows", False) and stack is not None and surfaces:
            self._render_pupils_and_windows_rz(stack, surfaces, is_dark)

    def _render_pupils_and_windows_rz(self, stack: Any, surfaces: List[Surface], is_dark: bool) -> None:
        """
        Overlay Entrance/Exit Pupils, Entrance/Exit Windows, and Field/Aperture Stops on 2D R-Z layout.
        """
        if not surfaces or self.ax_rz is None:
            return

        from optics_app.engine.first_order import calculate_first_order_data
        try:
            wl = 0.5876
            if hasattr(stack, "primary_wavelength") and stack.primary_wavelength:
                wl = float(stack.primary_wavelength)
            fod = calculate_first_order_data(surfaces, wavelength_um=wl)
        except Exception:
            return

        sw = fod.get("stops_and_windows", {})
        if not sw:
            return

        as_info = sw.get("aperture_stop", {})
        fs_info = sw.get("field_stop", {})
        enp_info = sw.get("entrance_pupil", {})
        exp_info = sw.get("exit_pupil", {})
        enw_info = sw.get("entrance_window", {})
        exw_info = sw.get("exit_window", {})

        show_lbl = getattr(self, "show_labels", True)

        # 1. Aperture Stop (AS)
        z_as = as_info.get("z_global", 0.0)
        sd_as = as_info.get("semi_diameter", 10.0)
        as_idx = as_info.get("surface_index", 0)
        as_color = "#ef4444" if is_dark else "#dc2626"
        l1, = self.ax_rz.plot([z_as, z_as], [sd_as, sd_as * 1.35], color=as_color, linewidth=3.2, zorder=11)
        l2, = self.ax_rz.plot([z_as, z_as], [-sd_as * 1.35, -sd_as], color=as_color, linewidth=3.2, zorder=11)
        l3, = self.ax_rz.plot([z_as, z_as], [-sd_as, sd_as], color=as_color, linestyle=":", linewidth=1.2, zorder=11)
        self._ray_artists_rz.extend([l1, l2, l3])
        if show_lbl:
            ann_as = self.ax_rz.annotate(
                f" AS: S{as_idx} (Ø={2*sd_as:.1f}mm) ",
                xy=(z_as, sd_as), xytext=(0, 10), textcoords="offset points",
                ha="center", va="bottom", fontsize=7.5, fontweight="bold",
                color="#fecaca" if is_dark else "#7f1d1d",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#450a0a" if is_dark else "#fee2e2", edgecolor=as_color, alpha=0.92, linewidth=1.0),
                zorder=12
            )
            self._ray_artists_rz.append(ann_as)

        # 2. Field Stop (FS)
        z_fs = fs_info.get("z_global", 0.0)
        sd_fs = fs_info.get("semi_diameter", 10.0)
        fs_idx = fs_info.get("surface_index", 0)
        fs_color = "#f472b6" if is_dark else "#c026d3"
        l4, = self.ax_rz.plot([z_fs, z_fs], [sd_fs, sd_fs * 1.35], color=fs_color, linewidth=3.2, zorder=11)
        l5, = self.ax_rz.plot([z_fs, z_fs], [-sd_fs * 1.35, -sd_fs], color=fs_color, linewidth=3.2, zorder=11)
        l6, = self.ax_rz.plot([z_fs, z_fs], [-sd_fs, sd_fs], color=fs_color, linestyle=":", linewidth=1.2, zorder=11)
        self._ray_artists_rz.extend([l4, l5, l6])
        if show_lbl:
            ann_fs = self.ax_rz.annotate(
                f" FS: S{fs_idx} (Ø={2*sd_fs:.1f}mm) ",
                xy=(z_fs, sd_fs), xytext=(0, 10), textcoords="offset points",
                ha="center", va="bottom", fontsize=7.5, fontweight="bold",
                color="#fbcfe8" if is_dark else "#701a75",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#500724" if is_dark else "#fdf2f8", edgecolor=fs_color, alpha=0.92, linewidth=1.0),
                zorder=12
            )
            self._ray_artists_rz.append(ann_fs)

        # 3. Entrance Pupil (EnP)
        z_enp = enp_info.get("z_global", 0.0)
        sd_enp = enp_info.get("semi_diameter", 10.0)
        enp_color = "#eab308"
        l7, = self.ax_rz.plot([z_enp, z_enp], [-sd_enp, sd_enp], color=enp_color, linestyle="-.", linewidth=1.4, zorder=11)
        l8, = self.ax_rz.plot([z_enp, z_enp], [sd_enp * 0.9, sd_enp * 1.1], color=enp_color, linewidth=2.8, zorder=11)
        l9, = self.ax_rz.plot([z_enp, z_enp], [-sd_enp * 1.1, -sd_enp * 0.9], color=enp_color, linewidth=2.8, zorder=11)
        self._ray_artists_rz.extend([l7, l8, l9])
        if show_lbl:
            ann_enp = self.ax_rz.annotate(
                f" EnP: Ø={2*sd_enp:.1f}mm ",
                xy=(z_enp, -sd_enp), xytext=(0, -12), textcoords="offset points",
                ha="center", va="top", fontsize=7.5, fontweight="bold",
                color="#fef08a" if is_dark else "#713f12",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#422006" if is_dark else "#fefce8", edgecolor=enp_color, alpha=0.92, linewidth=1.0),
                zorder=12
            )
            self._ray_artists_rz.append(ann_enp)

        # 4. Exit Pupil (ExP)
        z_exp = exp_info.get("z_global", 0.0)
        sd_exp = exp_info.get("semi_diameter", 10.0)
        exp_color = "#c084fc" if is_dark else "#9333ea"
        l10, = self.ax_rz.plot([z_exp, z_exp], [-sd_exp, sd_exp], color=exp_color, linestyle="-.", linewidth=1.4, zorder=11)
        l11, = self.ax_rz.plot([z_exp, z_exp], [sd_exp * 0.9, sd_exp * 1.1], color=exp_color, linewidth=2.8, zorder=11)
        l12, = self.ax_rz.plot([z_exp, z_exp], [-sd_exp * 1.1, -sd_exp * 0.9], color=exp_color, linewidth=2.8, zorder=11)
        self._ray_artists_rz.extend([l10, l11, l12])
        if show_lbl:
            ann_exp = self.ax_rz.annotate(
                f" ExP: Ø={2*sd_exp:.1f}mm ",
                xy=(z_exp, -sd_exp), xytext=(0, -12), textcoords="offset points",
                ha="center", va="top", fontsize=7.5, fontweight="bold",
                color="#e9d5ff" if is_dark else "#581c87",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#3b0764" if is_dark else "#faf5ff", edgecolor=exp_color, alpha=0.92, linewidth=1.0),
                zorder=12
            )
            self._ray_artists_rz.append(ann_exp)

        # 5. Entrance Window (EnW)
        z_enw = enw_info.get("z_global", 0.0)
        sd_enw = enw_info.get("semi_diameter", 10.0)
        if abs(z_enw) < 1500.0 and not math.isinf(z_enw):
            enw_color = "#06b6d4"
            l13, = self.ax_rz.plot([z_enw, z_enw], [-sd_enw, sd_enw], color=enw_color, linestyle="--", linewidth=1.2, alpha=0.9, zorder=11)
            self._ray_artists_rz.append(l13)
            if show_lbl:
                ann_enw = self.ax_rz.annotate(
                    f" EnW: Ø={2*sd_enw:.1f}mm ",
                    xy=(z_enw, sd_enw), xytext=(0, 8), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7.2, fontweight="bold",
                    color="#cffafe" if is_dark else "#155e75",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#083344" if is_dark else "#ecfeff", edgecolor=enw_color, alpha=0.92, linewidth=1.0),
                    zorder=12
                )
                self._ray_artists_rz.append(ann_enw)

        # 6. Exit Window (ExW)
        z_exw = exw_info.get("z_global", 0.0)
        sd_exw = exw_info.get("semi_diameter", 10.0)
        if abs(z_exw) < 1500.0 and not math.isinf(z_exw):
            exw_color = "#10b981"
            l14, = self.ax_rz.plot([z_exw, z_exw], [-sd_exw, sd_exw], color=exw_color, linestyle="--", linewidth=1.2, alpha=0.9, zorder=11)
            self._ray_artists_rz.append(l14)
            if show_lbl:
                ann_exw = self.ax_rz.annotate(
                    f" ExW: Ø={2*sd_exw:.1f}mm ",
                    xy=(z_exw, sd_exw), xytext=(0, 8), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7.2, fontweight="bold",
                    color="#d1fae5" if is_dark else "#065f46",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#064e3b" if is_dark else "#ecfdf5", edgecolor=exw_color, alpha=0.92, linewidth=1.0),
                    zorder=12
                )
                self._ray_artists_rz.append(ann_exw)

    def _render_rays_and_annotations_xz(self, stack, surfaces) -> None:
        if hasattr(self, "_ray_artists_xz"):
            for art in self._ray_artists_xz:
                try:
                    art.remove()
                except Exception:
                    pass
            self._ray_artists_xz.clear()
        else:
            self._ray_artists_xz = []

        if not self.rays or self.ax_xz is None:
            return

        is_dark = getattr(self, "dark_mode", False)
        show_mc = getattr(self, "show_marginal_chief", False)
        mc_only = getattr(self, "marginal_chief_only", False)
        has_marginal_drawn = False
        has_chief_drawn = False
        marginal_col = '#38bdf8' if is_dark else '#0284c7'
        chief_col = '#fb7185' if is_dark else '#dc2626'

        unique_wls = sorted(list(set(round(getattr(r, "wavelength", 0.5876), 4) for r in self.rays)))
        palette = (
            ['#38bdf8', '#4ade80', '#f87171', '#fbbf24', '#c084fc', '#fb923c', '#f472b6']
            if is_dark else
            ['#1f77b4', '#2ca02c', '#d62728', '#ff7f0e', '#9467bd', '#8c564b', '#e377c2']
        )
        field_palette = (
            ['#60a5fa', '#4ade80', '#f87171', '#c084fc', '#fb923c', '#22d3ee', '#facc15', '#fb7185']
            if is_dark else FIELD_COLORS
        )
        segments_by_style = {}
        for ray in self.rays:
            if hasattr(ray, "history") and len(ray.history) > 0:
                rz_z = [h.get("z_global", h.get("z", 0.0)) for h in ray.history]
                rz_x = [h.get("x_global", h.get("x", 0.0)) for h in ray.history]
                if len(rz_z) < 2:
                    continue

                is_chief, is_marginal = self._classify_ray(ray)
                if mc_only and not (is_chief or is_marginal):
                    continue

                is_vig = getattr(ray, "vignetted", False)

                if (show_mc or mc_only) and is_marginal:
                    ray_color = marginal_col
                    alpha = 0.95 if not is_vig else 0.50
                    ls = '-' if not is_vig else '--'
                    lw = 2.4 if not is_vig else 1.4
                    zorder = 8
                    has_marginal_drawn = True
                elif (show_mc or mc_only) and is_chief:
                    ray_color = chief_col
                    alpha = 0.95 if not is_vig else 0.50
                    ls = '-' if not is_vig else '--'
                    lw = 2.4 if not is_vig else 1.4
                    zorder = 9
                    has_chief_drawn = True
                else:
                    if getattr(self, "color_mode", "wavelength") == "field":
                        f_idx = int(getattr(ray, "field_idx", 0))
                        ray_color = field_palette[f_idx % len(field_palette)]
                    else:
                        wl = round(getattr(ray, "wavelength", 0.5876), 4)
                        wl_idx = unique_wls.index(wl) if wl in unique_wls else 0
                        ray_color = palette[wl_idx % len(palette)]

                    if show_mc:
                        alpha = 0.28 if not is_vig else 0.15
                        ls = '-' if not is_vig else '--'
                        lw = 0.9 if not is_vig else 0.6
                        zorder = 5
                    else:
                        alpha = 0.85 if not is_vig else 0.4
                        ls = '-' if not is_vig else '--'
                        lw = 1.3 if not is_vig else 0.9
                        zorder = 5

                style_key = (ray_color, alpha, ls, lw, zorder)
                if style_key not in segments_by_style:
                    segments_by_style[style_key] = []
                segments_by_style[style_key].append(np.column_stack([rz_z, rz_x]))

        for (color, alpha, ls, lw, zorder), segs in segments_by_style.items():
            lc = LineCollection(segs, colors=color, alpha=alpha, linestyles=ls, linewidths=lw, zorder=zorder)
            self.ax_xz.add_collection(lc)
            self._ray_artists_xz.append(lc)

        if (show_mc or mc_only) and (has_marginal_drawn or has_chief_drawn) and getattr(self, "show_labels", True):
            from matplotlib.lines import Line2D
            handles = []
            if has_marginal_drawn:
                handles.append(Line2D([0], [0], color=marginal_col, lw=2.4, label='Marginal Ray (Px=±1.0)'))
            if has_chief_drawn:
                handles.append(Line2D([0], [0], color=chief_col, lw=2.4, label='Chief Ray (Px=0.0)'))
            if handles:
                leg = self.ax_xz.legend(
                    handles=handles,
                    loc='upper right',
                    fontsize=8.5,
                    framealpha=0.88,
                    facecolor='#1e293b' if is_dark else '#ffffff',
                    edgecolor='#475569' if is_dark else '#cbd5e1',
                    labelcolor='#f1f5f9' if is_dark else '#0f172a'
                )
                leg.set_zorder(15)
                self._ray_artists_xz.append(leg)

        mech_info = self.get_mech_surface_info()
        if mech_info and mech_info[2] == "OFF" and self.rays and getattr(self, "show_labels", True):
            r0 = self.rays[0]
            if r0.history and len(r0.history) > mech_info[0]:
                term_h = r0.history[-1]
                tz = float(term_h.get("z_global", 0.0))
                tx = float(term_h.get("x_global", 0.0))
                txt = self.ax_xz.text(
                    tz, tx, " ⏹ Beam Dump (MECH OFF) ",
                    color="#f87171" if is_dark else "#991b1b", fontsize=8.5, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="#450a0a" if is_dark else "#fee2e2", edgecolor="#ef4444", alpha=0.92, linewidth=1.1),
                    ha="left", va="bottom", zorder=15
                )
                self._ray_artists_xz.append(txt)

        # Render Pupils and Windows Overlays
        if getattr(self, "show_pupils_windows", False) and stack is not None and surfaces:
            self._render_pupils_and_windows_xz(stack, surfaces, is_dark)

    def _render_pupils_and_windows_xz(self, stack: Any, surfaces: List[Surface], is_dark: bool) -> None:
        """
        Overlay Entrance/Exit Pupils, Entrance/Exit Windows, and Field/Aperture Stops on 2D X-Z layout.
        """
        if not surfaces or self.ax_xz is None:
            return

        from optics_app.engine.first_order import calculate_first_order_data
        try:
            wl = 0.5876
            if hasattr(stack, "primary_wavelength") and stack.primary_wavelength:
                wl = float(stack.primary_wavelength)
            fod = calculate_first_order_data(surfaces, wavelength_um=wl)
        except Exception:
            return

        sw = fod.get("stops_and_windows", {})
        if not sw:
            return

        as_info = sw.get("aperture_stop", {})
        fs_info = sw.get("field_stop", {})
        enp_info = sw.get("entrance_pupil", {})
        exp_info = sw.get("exit_pupil", {})
        enw_info = sw.get("entrance_window", {})
        exw_info = sw.get("exit_window", {})

        show_lbl = getattr(self, "show_labels", True)

        # 1. Aperture Stop (AS)
        z_as = as_info.get("z_global", 0.0)
        sd_as = as_info.get("semi_diameter", 10.0)
        as_idx = as_info.get("surface_index", 0)
        as_color = "#ef4444" if is_dark else "#dc2626"
        l1, = self.ax_xz.plot([z_as, z_as], [sd_as, sd_as * 1.35], color=as_color, linewidth=3.2, zorder=11)
        l2, = self.ax_xz.plot([z_as, z_as], [-sd_as * 1.35, -sd_as], color=as_color, linewidth=3.2, zorder=11)
        l3, = self.ax_xz.plot([z_as, z_as], [-sd_as, sd_as], color=as_color, linestyle=":", linewidth=1.2, zorder=11)
        self._ray_artists_xz.extend([l1, l2, l3])
        if show_lbl:
            ann_as = self.ax_xz.annotate(
                f" AS: S{as_idx} (Ø={2*sd_as:.1f}mm) ",
                xy=(z_as, sd_as), xytext=(0, 10), textcoords="offset points",
                ha="center", va="bottom", fontsize=7.5, fontweight="bold",
                color="#fecaca" if is_dark else "#7f1d1d",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#450a0a" if is_dark else "#fee2e2", edgecolor=as_color, alpha=0.92, linewidth=1.0),
                zorder=12
            )
            self._ray_artists_xz.append(ann_as)

        # 2. Field Stop (FS)
        z_fs = fs_info.get("z_global", 0.0)
        sd_fs = fs_info.get("semi_diameter", 10.0)
        fs_idx = fs_info.get("surface_index", 0)
        fs_color = "#f472b6" if is_dark else "#c026d3"
        l4, = self.ax_xz.plot([z_fs, z_fs], [sd_fs, sd_fs * 1.35], color=fs_color, linewidth=3.2, zorder=11)
        l5, = self.ax_xz.plot([z_fs, z_fs], [-sd_fs * 1.35, -sd_fs], color=fs_color, linewidth=3.2, zorder=11)
        l6, = self.ax_xz.plot([z_fs, z_fs], [-sd_fs, sd_fs], color=fs_color, linestyle=":", linewidth=1.2, zorder=11)
        self._ray_artists_xz.extend([l4, l5, l6])
        if show_lbl:
            ann_fs = self.ax_xz.annotate(
                f" FS: S{fs_idx} (Ø={2*sd_fs:.1f}mm) ",
                xy=(z_fs, sd_fs), xytext=(0, 10), textcoords="offset points",
                ha="center", va="bottom", fontsize=7.5, fontweight="bold",
                color="#fbcfe8" if is_dark else "#701a75",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#500724" if is_dark else "#fdf2f8", edgecolor=fs_color, alpha=0.92, linewidth=1.0),
                zorder=12
            )
            self._ray_artists_xz.append(ann_fs)

        # 3. Entrance Pupil (EnP)
        z_enp = enp_info.get("z_global", 0.0)
        sd_enp = enp_info.get("semi_diameter", 10.0)
        enp_color = "#eab308"
        l7, = self.ax_xz.plot([z_enp, z_enp], [-sd_enp, sd_enp], color=enp_color, linestyle="-.", linewidth=1.4, zorder=11)
        l8, = self.ax_xz.plot([z_enp, z_enp], [sd_enp * 0.9, sd_enp * 1.1], color=enp_color, linewidth=2.8, zorder=11)
        l9, = self.ax_xz.plot([z_enp, z_enp], [-sd_enp * 1.1, -sd_enp * 0.9], color=enp_color, linewidth=2.8, zorder=11)
        self._ray_artists_xz.extend([l7, l8, l9])
        if show_lbl:
            ann_enp = self.ax_xz.annotate(
                f" EnP: Ø={2*sd_enp:.1f}mm ",
                xy=(z_enp, -sd_enp), xytext=(0, -12), textcoords="offset points",
                ha="center", va="top", fontsize=7.5, fontweight="bold",
                color="#fef08a" if is_dark else "#713f12",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#422006" if is_dark else "#fefce8", edgecolor=enp_color, alpha=0.92, linewidth=1.0),
                zorder=12
            )
            self._ray_artists_xz.append(ann_enp)

        # 4. Exit Pupil (ExP)
        z_exp = exp_info.get("z_global", 0.0)
        sd_exp = exp_info.get("semi_diameter", 10.0)
        exp_color = "#c084fc" if is_dark else "#9333ea"
        l10, = self.ax_xz.plot([z_exp, z_exp], [-sd_exp, sd_exp], color=exp_color, linestyle="-.", linewidth=1.4, zorder=11)
        l11, = self.ax_xz.plot([z_exp, z_exp], [sd_exp * 0.9, sd_exp * 1.1], color=exp_color, linewidth=2.8, zorder=11)
        l12, = self.ax_xz.plot([z_exp, z_exp], [-sd_exp * 1.1, -sd_exp * 0.9], color=exp_color, linewidth=2.8, zorder=11)
        self._ray_artists_xz.extend([l10, l11, l12])
        if show_lbl:
            ann_exp = self.ax_xz.annotate(
                f" ExP: Ø={2*sd_exp:.1f}mm ",
                xy=(z_exp, -sd_exp), xytext=(0, -12), textcoords="offset points",
                ha="center", va="top", fontsize=7.5, fontweight="bold",
                color="#e9d5ff" if is_dark else "#581c87",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#3b0764" if is_dark else "#faf5ff", edgecolor=exp_color, alpha=0.92, linewidth=1.0),
                zorder=12
            )
            self._ray_artists_xz.append(ann_exp)

        # 5. Entrance Window (EnW)
        z_enw = enw_info.get("z_global", 0.0)
        sd_enw = enw_info.get("semi_diameter", 10.0)
        if abs(z_enw) < 1500.0 and not math.isinf(z_enw):
            enw_color = "#06b6d4"
            l13, = self.ax_xz.plot([z_enw, z_enw], [-sd_enw, sd_enw], color=enw_color, linestyle="--", linewidth=1.2, alpha=0.9, zorder=11)
            self._ray_artists_xz.append(l13)
            if show_lbl:
                ann_enw = self.ax_xz.annotate(
                    f" EnW: Ø={2*sd_enw:.1f}mm ",
                    xy=(z_enw, sd_enw), xytext=(0, 8), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7.2, fontweight="bold",
                    color="#cffafe" if is_dark else "#155e75",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#083344" if is_dark else "#ecfeff", edgecolor=enw_color, alpha=0.92, linewidth=1.0),
                    zorder=12
                )
                self._ray_artists_xz.append(ann_enw)

        # 6. Exit Window (ExW)
        z_exw = exw_info.get("z_global", 0.0)
        sd_exw = exw_info.get("semi_diameter", 10.0)
        if abs(z_exw) < 1500.0 and not math.isinf(z_exw):
            exw_color = "#10b981"
            l14, = self.ax_xz.plot([z_exw, z_exw], [-sd_exw, sd_exw], color=exw_color, linestyle="--", linewidth=1.2, alpha=0.9, zorder=11)
            self._ray_artists_xz.append(l14)
            if show_lbl:
                ann_exw = self.ax_xz.annotate(
                    f" ExW: Ø={2*sd_exw:.1f}mm ",
                    xy=(z_exw, sd_exw), xytext=(0, 8), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7.2, fontweight="bold",
                    color="#d1fae5" if is_dark else "#065f46",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#064e3b" if is_dark else "#ecfdf5", edgecolor=exw_color, alpha=0.92, linewidth=1.0),
                    zorder=12
                )
                self._ray_artists_xz.append(ann_exw)

    def _render_rays_and_annotations_xy(self, stack, surfaces) -> None:
        if hasattr(self, "_ray_artists_xy"):
            for art in self._ray_artists_xy:
                try:
                    art.remove()
                except Exception:
                    pass
            self._ray_artists_xy.clear()
        else:
            self._ray_artists_xy = []

        if not self.rays or self.ax_xy_front is None:
            return

        is_dark = getattr(self, "dark_mode", False)
        unique_wls = sorted(list(set(round(getattr(r, "wavelength", 0.5876), 4) for r in self.rays)))
        palette = (
            ['#38bdf8', '#4ade80', '#f87171', '#fbbf24', '#c084fc', '#fb923c', '#f472b6']
            if is_dark else
            ['#1f77b4', '#2ca02c', '#d62728', '#ff7f0e', '#9467bd', '#8c564b', '#e377c2']
        )
        field_palette = (
            ['#60a5fa', '#4ade80', '#f87171', '#c084fc', '#fb923c', '#22d3ee', '#facc15', '#fb7185']
            if is_dark else FIELD_COLORS
        )
        show_mc = getattr(self, "show_marginal_chief", False)
        mc_only = getattr(self, "marginal_chief_only", False)
        marginal_col = '#38bdf8' if is_dark else '#0284c7'
        chief_col = '#fb7185' if is_dark else '#dc2626'

        segments_by_style = {}
        for ray in self.rays:
            if hasattr(ray, "history") and len(ray.history) > 0:
                rz_x = [h.get("x_global", h.get("x", 0.0)) for h in ray.history]
                rz_y = [h.get("y_global", h.get("y", 0.0)) for h in ray.history]
                if len(rz_x) < 2:
                    continue

                is_chief, is_marginal = self._classify_ray(ray)
                if mc_only and not (is_chief or is_marginal):
                    continue

                if (show_mc or mc_only) and is_marginal:
                    ray_color = marginal_col
                    alpha = 0.85
                    lw = 1.8
                elif (show_mc or mc_only) and is_chief:
                    ray_color = chief_col
                    alpha = 0.85
                    lw = 1.8
                else:
                    if getattr(self, "color_mode", "wavelength") == "field":
                        f_idx = int(getattr(ray, "field_idx", 0))
                        ray_color = field_palette[f_idx % len(field_palette)]
                    else:
                        wl = round(getattr(ray, "wavelength", 0.5876), 4)
                        wl_idx = unique_wls.index(wl) if wl in unique_wls else 0
                        ray_color = palette[wl_idx % len(palette)]
                    alpha = 0.20 if (show_mc and not getattr(ray, "vignetted", False)) else (0.4 if not getattr(ray, "vignetted", False) else 0.15)
                    lw = 0.8

                style_key = (ray_color, alpha, lw)
                if style_key not in segments_by_style:
                    segments_by_style[style_key] = []
                segments_by_style[style_key].append(np.column_stack([rz_x, rz_y]))

        for (color, alpha, lw), segs in segments_by_style.items():
            lc = LineCollection(segs, colors=color, alpha=alpha, linewidths=lw, zorder=5)
            self.ax_xy_front.add_collection(lc)
            self._ray_artists_xy.append(lc)

        # Scatter ray footprint spots on the active front aperture
        spots_x = []
        spots_y = []
        spots_col = []
        for ray in self.rays:
            if hasattr(ray, "history") and len(ray.history) > 1:
                for h in ray.history:
                    s_idx = h.get("surface_index", -1)
                    if 0 <= s_idx < len(surfaces):
                        s = surfaces[s_idx]
                        if is_plm(s) or is_dmd(s) or getattr(s, "plm_config", None) is not None:
                            spots_x.append(float(h.get("x_global", h.get("x", 0.0))))
                            spots_y.append(float(h.get("y_global", h.get("y", 0.0))))
                            is_vig = getattr(ray, "vignetted", False)
                            spots_col.append('#ef4444' if is_vig else ('#38bdf8' if is_dark else '#0284c7'))
                            break
        if spots_x:
            sc = self.ax_xy_front.scatter(spots_x, spots_y, c=spots_col, s=22, marker='o', edgecolors='white', linewidths=0.6, zorder=6, alpha=0.9)
            self._ray_artists_xy.append(sc)

    def update_layout(self, surface_stack: Optional[SurfaceStack] = None, rays: Optional[List[Ray]] = None, force_rebuild: bool = False) -> None:
        """
        Renders the R-Z longitudinal profile and X-Y transverse layout views with zoom state preservation.
        """
        if surface_stack is not None:
            self.surface_stack = surface_stack
        if rays is not None:
            self.rays = rays
        if force_rebuild:
            self._last_geom_fingerprint = None

        self._sync_mech_button_ui()

        if not (QT_AVAILABLE and HAS_MPL_QT) or self.fig is None:
            return

        # 1. Save current zoom / pan limits before redrawing
        if not getattr(self, '_force_reset_view', False):
            if self.ax_rz is not None:
                self.saved_limits["rz"] = (self.ax_rz.get_xlim(), self.ax_rz.get_ylim())
            if self.ax_xz is not None:
                self.saved_limits["xz"] = (self.ax_xz.get_xlim(), self.ax_xz.get_ylim())

        active_views = sum([self.rz_view_active, self.xz_view_active, self.xy_front_view_active])
        active_tuple = (self.rz_view_active, self.xz_view_active, self.xy_front_view_active)

        if active_views == 0:
            if hasattr(self.canvas, "draw_idle"):
                self.canvas.draw_idle()
            return

        stack = self.surface_stack
        surfaces = (stack.get_active_sequence() if hasattr(stack, 'get_active_sequence') else stack.surfaces) if stack is not None else []

        curr_geom_fingerprint = None
        if stack is not None:
            surf_tokens = []
            for s in surfaces:
                orig = getattr(s, 'global_origin', None)
                orig_t = tuple(np.round(np.asarray(orig, dtype=float), 4)) if orig is not None else ()
                R_mat = getattr(s, 'global_R', None)
                r_t = tuple(np.round(np.asarray(R_mat, dtype=float).ravel(), 4)) if R_mat is not None else ()
                try:
                    eff_orig, eff_R = get_effective_pose(s)
                    eff_orig_t = tuple(np.round(np.asarray(eff_orig, dtype=float), 4))
                    eff_r_t = tuple(np.round(np.asarray(eff_R, dtype=float).ravel(), 4))
                except Exception:
                    eff_orig_t = orig_t
                    eff_r_t = r_t
                mech_st = str(getattr(s, 'mechanical_state', '') or '')
                plm_c = getattr(s, 'plm_config', None)
                grat_sig = (
                    bool(plm_c.grating.enabled),
                    round(float(plm_c.grating.lines_per_mm), 2),
                    int(plm_c.grating.diffraction_order),
                    round(float(plm_c.grating.orientation_deg), 1),
                ) if plm_c is not None and getattr(plm_c, 'grating', None) is not None else ()
                pat_sig = (
                    bool(plm_c.pattern.enabled),
                    str(plm_c.pattern.pattern_type),
                    int(getattr(plm_c.pattern, 'hadamard_order', 0)),
                    int(getattr(plm_c.pattern, 'hadamard_row_index', 0)),
                    str(getattr(plm_c.pattern, 'hadamard_axis', 'X')),
                    int(getattr(plm_c.pattern, 'column_start', 0)),
                    int(getattr(plm_c.pattern, 'column_end', 0)),
                    bool(getattr(plm_c.pattern, 'column_invert', False)),
                    float(getattr(plm_c.pattern, 'pattern_angle_deg', 0.0)),
                    int(getattr(plm_c.pattern, 'column_state', 1)),
                    int(getattr(plm_c.pattern, 'column_background_state', 0)),
                    tuple((int(r.get('start', 0)), int(r.get('end', 0)), int(r.get('state', 1))) for r in getattr(plm_c.pattern, 'column_ranges', []) if isinstance(r, dict)),
                    int(getattr(plm_c.pattern, 'checkerboard_period_pixels', 8)),
                    float(getattr(plm_c.pattern, 'phase_scale', 1.0)),
                    float(getattr(plm_c.pattern, 'fresnel_focal_length_mm', 100.0)),
                    int(getattr(plm_c.pattern, 'vortex_charge', 1)),
                    str(getattr(plm_c.pattern, 'file_path', '')),
                ) if plm_c is not None and getattr(plm_c, 'pattern', None) is not None else ()
                surf_tokens.append((
                    id(s),
                    str(getattr(s, 'name', '')),
                    str(getattr(s, 'surface_type', '')),
                    round(float(getattr(s, 'radius', 0.0)), 4),
                    round(float(getattr(s, 'thickness', 0.0)), 4),
                    str(getattr(s, 'glass', '')),
                    round(float(getattr(s, 'semi_diameter', 0.0)), 4),
                    round(float(getattr(s, 'conic', 0.0)), 4),
                    str(getattr(s, 'aperture_type', '')),
                    round(float(getattr(s, 'aperture_x_half', 0.0)), 4),
                    round(float(getattr(s, 'aperture_y_half', 0.0)), 4),
                    orig_t,
                    r_t,
                    eff_orig_t,
                    eff_r_t,
                    mech_st,
                    grat_sig,
                    pat_sig,
                ))
            curr_geom_fingerprint = (
                len(surfaces),
                active_tuple,
                getattr(self, 'selected_surface_index', None),
                len(getattr(self, 'measurement_history', [])),
                getattr(self, 'selected_measurement_idx', None),
                getattr(self, 'show_labels', True),
                getattr(self, 'dark_mode', False),
                getattr(self, 'show_marginal_chief', False),
                getattr(self, 'marginal_chief_only', False),
                tuple(surf_tokens),
            )

        can_reuse = (
            getattr(self, '_last_active_views_tuple', None) == active_tuple
            and (not self.rz_view_active or self.ax_rz is not None)
            and (not self.xz_view_active or self.ax_xz is not None)
            and (not self.xy_front_view_active or self.ax_xy_front is not None)
        )

        # Fast incremental ray path update if geometry has not changed
        if (
            can_reuse
            and not force_rebuild
            and not getattr(self, '_force_reset_view', False)
            and self._last_geom_fingerprint is not None
            and self._last_geom_fingerprint == curr_geom_fingerprint
        ):
            if self.ax_rz is not None:
                self._render_rays_and_annotations_rz(stack, surfaces)
            if self.ax_xz is not None:
                self._render_rays_and_annotations_xz(stack, surfaces)
            if self.ax_xy_front is not None:
                self._render_rays_and_annotations_xy(stack, surfaces)

            if hasattr(self.canvas, "draw_idle"):
                self.canvas.draw_idle()
            elif hasattr(self.canvas, "draw"):
                self.canvas.draw()
            return

        is_dark = getattr(self, 'dark_mode', False)
        bg_fig = '#0f172a' if is_dark else '#ffffff'
        bg_ax = '#0b0f19' if is_dark else '#ffffff'
        fg_text = '#f1f5f9' if is_dark else '#0f172a'
        fg_subtext = '#94a3b8' if is_dark else '#475569'
        grid_col = '#334155' if is_dark else '#cbd5e1'
        spine_col = '#334155' if is_dark else '#cbd5e1'
        axis_line_col = '#64748b' if is_dark else 'black'

        self.fig.patch.set_facecolor(bg_fig)

        if can_reuse:
            if self.ax_rz is not None:
                self.ax_rz.cla()
            if self.ax_xz is not None:
                self.ax_xz.cla()
            if self.ax_xy_front is not None:
                self.ax_xy_front.cla()
        else:
            self.fig.clf()
            self._last_active_views_tuple = active_tuple
            self.ax_rz = None
            self.ax_xz = None
            self.ax_xy_front = None

            cols = active_views
            idx = 1
            if self.rz_view_active:
                self.ax_rz = self.fig.add_subplot(1, cols, idx)
                idx += 1
            if self.xz_view_active:
                self.ax_xz = self.fig.add_subplot(1, cols, idx)
                idx += 1
            if self.xy_front_view_active:
                self.ax_xy_front = self.fig.add_subplot(1, cols, idx)
                idx += 1

        # ----------------------------------------------------
        # 1. Render View 1: Longitudinal R-Z Plane
        # ----------------------------------------------------
        if self.ax_rz is not None:
            self.ax_rz.set_facecolor(bg_ax)
            self.ax_rz.set_title("Longitudinal Axis (R-Z / Y-Z)", fontsize=10, fontweight='bold', color=fg_text)
            self.ax_rz.set_xlabel("Z Position (mm)", color=fg_subtext)
            self.ax_rz.set_ylabel("Y Coordinate (mm)", color=fg_subtext)
            self.ax_rz.grid(True, linestyle='--', alpha=0.35 if is_dark else 0.5, color=grid_col)
            self.ax_rz.tick_params(colors=fg_subtext)
            for spine in self.ax_rz.spines.values():
                spine.set_color(spine_col)
            self.ax_rz.axhline(0, color=axis_line_col, linestyle=':', alpha=0.6)
            if len(surfaces) > 0:
                # Keep geometry keyed by prescription index: spacing-only
                # rows are not drawn and must not shift later lens-face keys.
                surface_points = {}

                for i, surf in enumerate(surfaces):
                    if bool(getattr(surf, "is_air_space", False)):
                        continue
                    sd = surf.semi_diameter if surf.semi_diameter > 0 else 15.0
                    if getattr(surf, 'aperture_type', 'Circular') == 'Rectangular':
                        if getattr(surf, 'aperture_y_half', 0.0) > 0:
                            sd = surf.aperture_y_half
                    if surf.radius != 0.0 and not np.isinf(surf.radius):
                        sd = min(sd, abs(surf.radius) * 0.99)

                    r_vals = np.linspace(-sd, sd, 100)
                    z_sag = _surface_sag_values(surf, np.abs(r_vals))
                    z_sag = np.nan_to_num(z_sag, nan=0.0)

                    local_pts = np.vstack([np.zeros_like(r_vals), r_vals, z_sag])

                    if not hasattr(surf, 'global_origin'):
                        stack._update_accumulated_positions()

                    try:
                        eff_orig, eff_R = get_effective_pose(surf)
                        global_pts = eff_orig[:, None] + eff_R @ local_pts
                    except Exception:
                        global_pts = surf.global_origin[:, None] + surf.global_R @ local_pts

                    z_glob = global_pts[2, :]
                    y_glob = global_pts[1, :]

                    stype_up = str(surf.surface_type).upper()
                    is_selected = (getattr(self, 'selected_surface_index', None) == i)
                    surf_is_mod = is_plm(surf) or is_dmd(surf) or (getattr(surf, 'plm_config', None) is not None)
                    plm_c = getattr(surf, 'plm_config', None)
                    if surf_is_mod and plm_c is not None and getattr(plm_c, 'pattern', None) is not None and plm_c.pattern.enabled:
                        self.ax_rz.plot(z_glob, y_glob, color='#38bdf8' if is_dark else '#0284c7', linestyle='-', linewidth=2.6, zorder=4)
                        if getattr(self, "show_labels", True) and not is_selected:
                            dev_tag = "TI-DMD" if plm_c.is_dmd else "TI-PLM"
                            pat_name = plm_c.pattern.pattern_type.replace('_', ' ').title()
                            if plm_c.pattern.pattern_type.lower() == "hadamard":
                                pat_name += f" (H{plm_c.pattern.hadamard_order}, r{plm_c.pattern.hadamard_row_index}, {getattr(plm_c.pattern, 'hadamard_axis', 'X')})"
                            elif "column" in plm_c.pattern.pattern_type.lower() or plm_c.pattern.pattern_type.lower() == "column_range":
                                c_ranges = getattr(plm_c.pattern, 'column_ranges', None)
                                inv = getattr(plm_c.pattern, 'column_invert', False)
                                ang = getattr(plm_c.pattern, 'pattern_angle_deg', 0.0)
                                ang_str = f" @ {ang:+.0f}°" if abs(ang) > 0.01 else ""
                                if c_ranges and len(c_ranges) > 1:
                                    pat_name += f" ({len(c_ranges)} Ranges{ang_str})"
                                elif c_ranges and len(c_ranges) == 1:
                                    r0 = c_ranges[0]
                                    st_val = r0.get("state", 1)
                                    pat_name += f" (Cols {r0.get('start', 0)}..{r0.get('end', 0)} [S{st_val}]{ang_str}{' Inv' if inv else ''})"
                                else:
                                    c_s = getattr(plm_c.pattern, 'column_start', 0)
                                    c_e = getattr(plm_c.pattern, 'column_end', 0)
                                    st_val = getattr(plm_c.pattern, 'column_state', 1)
                                    st_str = f" [S{st_val}]" if not plm_c.is_dmd or st_val != 1 else ""
                                    pat_name += f" (Cols {c_s}..{c_e}{st_str}{ang_str}{' Inv' if inv else ''})"
                            mid_idx = len(z_glob) // 2
                            self.ax_rz.annotate(
                                f" {dev_tag} Mask: {pat_name} ",
                                xy=(float(z_glob[mid_idx]), float(y_glob[mid_idx])),
                                xytext=(0, 16),
                                textcoords="offset points",
                                ha='center',
                                va='bottom',
                                fontsize=8,
                                fontweight='bold',
                                color='#38bdf8' if is_dark else '#0369a1',
                                bbox=dict(boxstyle='round,pad=0.2', facecolor='#0f172a' if is_dark else '#f0f9ff', edgecolor='#38bdf8' if is_dark else '#0284c7', alpha=0.92, linewidth=1.0),
                                zorder=9
                            )
                    elif "STOP" in stype_up or getattr(surf, 'is_stop', False) or getattr(surf, 'is_aperture_stop', False):
                        self.ax_rz.plot(z_glob, y_glob, color='#f87171' if is_dark else '#d62728', linestyle='--', linewidth=2.0, zorder=3)
                        top_z = float(z_glob[-1])
                        top_y = float(y_glob[-1])
                        if getattr(self, "show_labels", True):
                            self.ax_rz.annotate(
                                f" Stop: Semi-D R={sd:.2f} mm (Ø={2*sd:.2f} mm) ",
                                xy=(top_z, top_y), xytext=(5, 6), textcoords="offset points",
                                fontsize=8, fontweight='bold', color='#fca5a5' if is_dark else '#b71c1c',
                                bbox=dict(boxstyle='round,pad=0.2', facecolor='#450a0a' if is_dark else '#ffebee', edgecolor='#ef4444', alpha=0.92, linewidth=1.0),
                                zorder=10
                            )
                    elif "DETECTOR" in stype_up or "IMAGE" in stype_up:
                        self.ax_rz.plot(z_glob, y_glob, color='#c084fc' if is_dark else '#9467bd', linestyle='-', linewidth=2.5, zorder=3)
                    elif "MIRROR" in str(surf.glass).upper() or "MIRROR" in stype_up:
                        self.ax_rz.plot(z_glob, y_glob, color='#fbbf24' if is_dark else '#ff7f0e', linestyle='-', linewidth=2.0, zorder=3)
                    elif "GRATING" in stype_up or getattr(surf, 'grating_lines_per_mm', 0) > 0:
                        self.ax_rz.plot(z_glob, y_glob, color='#c084fc' if is_dark else '#9467bd', linestyle='-.', linewidth=2.0, zorder=3)
                    else:
                        self.ax_rz.plot(z_glob, y_glob, color='#38bdf8' if is_dark else 'blue', linestyle='-', linewidth=1.5, zorder=3, label=f"S{i}: {surf.name}" if i < 5 else "")

                    # Highlight selected surface on 2D Longitudinal Axis (R-Z / Y-Z)
                    is_selected = (getattr(self, 'selected_surface_index', None) == i)
                    if is_selected:
                        self.ax_rz.plot(z_glob, y_glob, color='#FF9100', linewidth=6.5, alpha=0.5, zorder=6)
                        self.ax_rz.plot(z_glob, y_glob, color='#FFD700', linestyle='-', linewidth=3.2, zorder=7)
                        mid_idx = len(z_glob) // 2
                        vz, vy = z_glob[mid_idx], y_glob[mid_idx]
                        self.ax_rz.plot([vz], [vy], marker='o', markersize=8, color='#FF3D00', markeredgecolor='white', markeredgewidth=1.8, zorder=8)
                        if getattr(self, "show_labels", True):
                            self.ax_rz.annotate(
                                f" S{i}: {surf.name} ",
                                xy=(vz, vy),
                                xytext=(0, 24),
                                textcoords="offset points",
                                ha='center',
                                va='bottom',
                                fontsize=9,
                                fontweight='bold',
                                color='#fed7aa' if is_dark else '#BF360C',
                                bbox=dict(boxstyle='round,pad=0.25', facecolor='#7c2d12' if is_dark else '#FFF9C4', edgecolor='#f97316' if is_dark else '#FF6D00', alpha=0.95, linewidth=1.4),
                                arrowprops=dict(arrowstyle="->", color='#f97316' if is_dark else '#FF6D00', lw=1.5),
                                zorder=9
                            )

                    surface_points[i] = (z_glob, y_glob, surf)

                    if i > 0:
                        prev_surf = surfaces[i-1]
                        prev_glass = str(prev_surf.glass or "").upper().strip()
                        if i - 1 in surface_points and prev_glass and prev_glass not in ["AIR", "VACUUM"] and "MIRROR" not in prev_glass and str(prev_surf.surface_type).upper() != "MIRROR":
                            prev_z, prev_y, _ = surface_points[i-1]
                            edge_col = '#0284c7' if is_dark else 'darkblue'
                            self.ax_rz.plot([prev_z[0], z_glob[0]], [prev_y[0], y_glob[0]], color=edge_col, linewidth=1.2, zorder=3)
                            self.ax_rz.plot([prev_z[-1], z_glob[-1]], [prev_y[-1], y_glob[-1]], color=edge_col, linewidth=1.2, zorder=3)

                self._render_rays_and_annotations_rz(stack, surfaces)

                # Render active measurements (Distance: 1 line 2 points; Angle: 2 lines 3 points)
                for idx_m, m in enumerate(getattr(self, 'measurement_history', [])):
                    mtype = m.get("type")
                    is_sel = (getattr(self, 'selected_measurement_idx', None) == idx_m)
                    d_color = '#FFD700' if is_sel else '#D81B60'
                    a_color = '#FFD700' if is_sel else '#00897B'

                    if mtype == "distance":
                        p1 = m["p1"]
                        p2 = m["p2"]
                        dist = m["distance"]
                        dz = m["dz"]
                        dy = m["dy"]
                        ang = m["angle"]

                        # Draw dashed dimension line
                        self.ax_rz.plot([p1[0], p2[0]], [p1[1], p2[1]], color=d_color, linestyle='--', linewidth=2.4, zorder=10)

                        # Draw interactive edit handles at P1 and P2
                        self.ax_rz.plot([p1[0]], [p1[1]], marker='o', markersize=9, color=d_color, markeredgecolor='black', markeredgewidth=1.4, zorder=12)
                        self.ax_rz.plot([p2[0]], [p2[1]], marker='o', markersize=9, color=d_color, markeredgecolor='black', markeredgewidth=1.4, zorder=12)

                        mid_z = (p1[0] + p2[0]) / 2.0
                        mid_y = (p1[1] + p2[1]) / 2.0
                        if getattr(self, "show_labels", True):
                            self.ax_rz.annotate(
                                f" Dist: {dist:.2f} mm (ΔZ={dz:.2f}, ΔY={dy:.2f}, θ={ang:.1f}°) ",
                                xy=(mid_z, mid_y),
                                xytext=(0, 12),
                                textcoords="offset points",
                                ha='center',
                                va='bottom',
                                fontsize=9,
                                fontweight='bold',
                                color='#f472b6' if is_dark else '#880E4F',
                                bbox=dict(boxstyle='round,pad=0.35', facecolor='#831843' if is_dark else '#FCE4EC', edgecolor=d_color, alpha=0.95, linewidth=1.5),
                                zorder=13
                            )

                    elif mtype in ("angle", "3point_angle", "surface_angle"):
                        p1 = m["p1"]
                        apex = m.get("apex", m.get("p1"))
                        p2 = m["p2"]
                        ang = m["angle"]

                        # Draw the 2 lines: Apex -> P1 and Apex -> P2
                        self.ax_rz.plot([p1[0], apex[0], p2[0]], [p1[1], apex[1], p2[1]], color=a_color, linestyle='-', linewidth=2.4, zorder=10)

                        # Draw interactive edit handles at P1, Apex, and P2
                        self.ax_rz.plot([p1[0]], [p1[1]], marker='s', markersize=8, color=a_color, markeredgecolor='black', markeredgewidth=1.4, zorder=12)
                        self.ax_rz.plot([apex[0]], [apex[1]], marker='o', markersize=10, color='#E91E63', markeredgecolor='black', markeredgewidth=1.8, zorder=12)
                        self.ax_rz.plot([p2[0]], [p2[1]], marker='s', markersize=8, color=a_color, markeredgecolor='black', markeredgewidth=1.4, zorder=12)

                        # Draw angle arc at Apex
                        v1 = np.array([p1[0] - apex[0], p1[1] - apex[1]])
                        v2 = np.array([p2[0] - apex[0], p2[1] - apex[1]])
                        if np.linalg.norm(v1) > 1e-3 and np.linalg.norm(v2) > 1e-3:
                            theta1 = np.degrees(np.arctan2(v1[1], v1[0]))
                            theta2 = np.degrees(np.arctan2(v2[1], v2[0]))
                            r_arc = min(15.0, float(np.linalg.norm(v1) * 0.4), float(np.linalg.norm(v2) * 0.4))
                            if r_arc > 0.5:
                                t_span = np.linspace(np.radians(theta1), np.radians(theta2), 30)
                                arc_z = apex[0] + r_arc * np.cos(t_span)
                                arc_y = apex[1] + r_arc * np.sin(t_span)
                                self.ax_rz.plot(arc_z, arc_y, color='#E91E63', linestyle=':', linewidth=2.0, zorder=11)

                        if getattr(self, "show_labels", True):
                            self.ax_rz.annotate(
                                f" Angle: {ang:.2f}° (2 lines, 3 pts) ",
                                xy=(apex[0], apex[1]),
                                xytext=(0, 14),
                                textcoords="offset points",
                                ha='center',
                                va='bottom',
                                fontsize=9,
                                fontweight='bold',
                                color='#2dd4bf' if is_dark else '#004D40',
                                bbox=dict(boxstyle='round,pad=0.35', facecolor='#134e4a' if is_dark else '#E0F2F1', edgecolor=a_color, alpha=0.95, linewidth=1.5),
                                zorder=13
                            )

                # Render in-progress drawing preview (rubberband line and control points)
                if len(self.measure_points) > 0:
                    pts_z = [p["coord"][0] for p in self.measure_points]
                    pts_y = [p["coord"][1] for p in self.measure_points]
                    self.ax_rz.plot(pts_z, pts_y, color='#FF6D00', marker='o', markersize=8, linestyle=':', linewidth=1.8, zorder=15)
                    if getattr(self, "show_labels", True):
                        for k, pt in enumerate(self.measure_points):
                            self.ax_rz.annotate(
                                f" P{k+1} ",
                                xy=(pt["coord"][0], pt["coord"][1]),
                                xytext=(8, 8),
                                textcoords="offset points",
                                fontsize=8,
                                fontweight='bold',
                                color='#fdba74' if is_dark else '#E65100',
                                bbox=dict(boxstyle='round,pad=0.2', facecolor='#7c2d12' if is_dark else '#FFE0B2', edgecolor='#FF6D00', alpha=0.9),
                                zorder=16
                            )

                # Render glowing snap reticle on hover target
                if getattr(self, 'hover_snap_target', None) is not None:
                    ht = self.hover_snap_target
                    hz, hy = ht["vertex"]
                    self.ax_rz.plot([hz], [hy], marker='+', markersize=14, color='#00E676', markeredgewidth=2.2, zorder=20)
                    self.ax_rz.plot([hz], [hy], marker='o', markersize=12, fillstyle='none', color='#00E676', markeredgewidth=2.0, zorder=20)

            # Restore zoom / pan limits or apply equal aspect
            if not getattr(self, '_force_reset_view', False) and "rz" in self.saved_limits:
                x_lims, y_lims = self.saved_limits["rz"]
                if np.isfinite(x_lims).all() and np.isfinite(y_lims).all() and x_lims[0] != x_lims[1] and y_lims[0] != y_lims[1]:
                    self.ax_rz.set_xlim(x_lims)
                    self.ax_rz.set_ylim(y_lims)
                else:
                    self._apply_aspect(self.ax_rz)
            else:
                self._apply_aspect(self.ax_rz)

        # ----------------------------------------------------
        # 2. Render View 2: Transverse Layout Plane (X-Z)
        # ----------------------------------------------------
        surfaces = (stack.get_active_sequence() if hasattr(stack, 'get_active_sequence') else stack.surfaces) if stack is not None else []
        
        if self.ax_xz is not None:
            self.ax_xz.set_facecolor(bg_ax)
            self.ax_xz.set_title("Transverse Layout (X-Z) [Top View]", fontsize=10, fontweight='bold', color=fg_text)
            self.ax_xz.set_xlabel("Z Position (mm) [Optical Axis]", color=fg_subtext)
            self.ax_xz.set_ylabel("X Position (mm)", color=fg_subtext)
            self.ax_xz.grid(True, linestyle='--', alpha=0.35 if is_dark else 0.5, color=grid_col)
            self.ax_xz.tick_params(colors=fg_subtext)
            for spine in self.ax_xz.spines.values():
                spine.set_color(spine_col)
            self.ax_xz.axhline(0, color=axis_line_col, linestyle=':', alpha=0.6)
            self.ax_xz.axvline(0, color=axis_line_col, linestyle=':', alpha=0.6)

            if stack and len(surfaces) > 0:
                surface_points = {}
                for i, surf in enumerate(surfaces):
                    if bool(getattr(surf, "is_air_space", False)):
                        continue
                    sd = surf.semi_diameter if surf.semi_diameter > 0 else 15.0
                    if getattr(surf, 'aperture_type', 'Circular') == 'Rectangular':
                        if getattr(surf, 'aperture_x_half', 0.0) > 0:
                            sd = surf.aperture_x_half
                    y_local = np.linspace(-sd, sd, 50)
                    z_local = np.zeros_like(y_local)
                    if hasattr(surf, 'sag') and surf.surface_type != SurfaceType.CAD:
                        try:
                            z_local = _surface_sag_values(surf, np.abs(y_local))
                        except Exception:
                            pass

                    local_pts = np.vstack([y_local, np.zeros_like(y_local), z_local])

                    if not hasattr(surf, 'global_origin'):
                        stack._update_accumulated_positions()

                    try:
                        eff_orig, eff_R = get_effective_pose(surf)
                        global_pts = eff_orig[:, None] + eff_R @ local_pts
                    except Exception:
                        global_pts = surf.global_origin[:, None] + surf.global_R @ local_pts

                    z_glob = global_pts[2, :]
                    x_glob = global_pts[0, :]

                    stype_up = str(surf.surface_type).upper()
                    is_selected = (getattr(self, 'selected_surface_index', None) == i)
                    surf_is_mod = is_plm(surf) or is_dmd(surf) or (getattr(surf, 'plm_config', None) is not None)
                    plm_c = getattr(surf, 'plm_config', None)
                    if surf_is_mod and plm_c is not None and getattr(plm_c, 'pattern', None) is not None and plm_c.pattern.enabled:
                        self.ax_xz.plot(z_glob, x_glob, color='#38bdf8' if is_dark else '#0284c7', linestyle='-', linewidth=2.6, zorder=4)
                        if getattr(self, "show_labels", True) and not is_selected:
                            dev_tag = "TI-DMD" if plm_c.is_dmd else "TI-PLM"
                            pat_name = plm_c.pattern.pattern_type.replace('_', ' ').title()
                            if plm_c.pattern.pattern_type.lower() == "hadamard":
                                pat_name += f" (H{plm_c.pattern.hadamard_order}, r{plm_c.pattern.hadamard_row_index}, {getattr(plm_c.pattern, 'hadamard_axis', 'X')})"
                            elif "column" in plm_c.pattern.pattern_type.lower() or plm_c.pattern.pattern_type.lower() == "column_range":
                                c_ranges = getattr(plm_c.pattern, 'column_ranges', None)
                                inv = getattr(plm_c.pattern, 'column_invert', False)
                                ang = getattr(plm_c.pattern, 'pattern_angle_deg', 0.0)
                                ang_str = f" @ {ang:+.0f}°" if abs(ang) > 0.01 else ""
                                if c_ranges and len(c_ranges) > 1:
                                    pat_name += f" ({len(c_ranges)} Ranges{ang_str})"
                                elif c_ranges and len(c_ranges) == 1:
                                    r0 = c_ranges[0]
                                    st_val = r0.get("state", 1)
                                    pat_name += f" (Cols {r0.get('start', 0)}..{r0.get('end', 0)} [S{st_val}]{ang_str}{' Inv' if inv else ''})"
                                else:
                                    c_s = getattr(plm_c.pattern, 'column_start', 0)
                                    c_e = getattr(plm_c.pattern, 'column_end', 0)
                                    st_val = getattr(plm_c.pattern, 'column_state', 1)
                                    st_str = f" [S{st_val}]" if not plm_c.is_dmd or st_val != 1 else ""
                                    pat_name += f" (Cols {c_s}..{c_e}{st_str}{ang_str}{' Inv' if inv else ''})"
                            mid_idx = len(z_glob) // 2
                            self.ax_xz.annotate(
                                f" {dev_tag} Mask: {pat_name} ",
                                xy=(float(z_glob[mid_idx]), float(x_glob[mid_idx])),
                                xytext=(0, 16),
                                textcoords="offset points",
                                ha='center',
                                va='bottom',
                                fontsize=8,
                                fontweight='bold',
                                color='#38bdf8' if is_dark else '#0369a1',
                                bbox=dict(boxstyle='round,pad=0.2', facecolor='#0f172a' if is_dark else '#f0f9ff', edgecolor='#38bdf8' if is_dark else '#0284c7', alpha=0.92, linewidth=1.0),
                                zorder=9
                            )
                    elif "STOP" in stype_up or getattr(surf, 'is_stop', False) or getattr(surf, 'is_aperture_stop', False):
                        self.ax_xz.plot(z_glob, x_glob, color='#f87171' if is_dark else '#d62728', linestyle='--', linewidth=2.0, zorder=3)
                        top_z = float(z_glob[-1])
                        top_x = float(x_glob[-1])
                        if getattr(self, "show_labels", True):
                            self.ax_xz.annotate(
                                f" Stop: Semi-D R={sd:.2f} mm (Ø={2*sd:.2f} mm) ",
                                xy=(top_z, top_x), xytext=(5, 6), textcoords="offset points",
                                fontsize=8, fontweight='bold', color='#fca5a5' if is_dark else '#b71c1c',
                                bbox=dict(boxstyle='round,pad=0.2', facecolor='#450a0a' if is_dark else '#ffebee', edgecolor='#ef4444', alpha=0.92, linewidth=1.0),
                                zorder=10
                            )
                    elif "DETECTOR" in stype_up or "IMAGE" in stype_up:
                        self.ax_xz.plot(z_glob, x_glob, color='#c084fc' if is_dark else '#9467bd', linestyle='-', linewidth=2.5, zorder=3)
                    elif "MIRROR" in str(surf.glass).upper() or "MIRROR" in stype_up:
                        self.ax_xz.plot(z_glob, x_glob, color='#fbbf24' if is_dark else '#ff7f0e', linestyle='-', linewidth=2.0, zorder=3)
                    elif "GRATING" in stype_up or getattr(surf, 'grating_lines_per_mm', 0) > 0:
                        self.ax_xz.plot(z_glob, x_glob, color='#c084fc' if is_dark else '#9467bd', linestyle='-.', linewidth=2.0, zorder=3)
                    else:
                        self.ax_xz.plot(z_glob, x_glob, color='#4ade80' if is_dark else 'green', linestyle='-', linewidth=1.5, zorder=3)

                    # Highlight selected surface on Transverse (X-Z)
                    is_selected = (getattr(self, 'selected_surface_index', None) == i)
                    if is_selected:
                        self.ax_xz.plot(z_glob, x_glob, color='#FF9100', linewidth=6.5, alpha=0.5, zorder=6)
                        self.ax_xz.plot(z_glob, x_glob, color='#FFD700', linestyle='-', linewidth=3.2, zorder=7)
                        mid_idx = len(z_glob) // 2
                        vz, vx = z_glob[mid_idx], x_glob[mid_idx]
                        self.ax_xz.plot([vz], [vx], marker='o', markersize=8, color='#FF3D00', markeredgecolor='white', markeredgewidth=1.8, zorder=8)

                    surface_points[i] = (z_glob, x_glob, surf)

                    if i > 0:
                        prev_surf = surfaces[i-1]
                        prev_glass = str(prev_surf.glass or "").upper().strip()
                        if i - 1 in surface_points and prev_glass and prev_glass not in ["AIR", "VACUUM"] and "MIRROR" not in prev_glass and str(prev_surf.surface_type).upper() != "MIRROR":
                            prev_z, prev_x, _ = surface_points[i-1]
                            edge_col_xz = '#16a34a' if is_dark else 'darkgreen'
                            self.ax_xz.plot([prev_z[0], z_glob[0]], [prev_x[0], x_glob[0]], color=edge_col_xz, linewidth=1.2, zorder=3)
                            self.ax_xz.plot([prev_z[-1], z_glob[-1]], [prev_x[-1], x_glob[-1]], color=edge_col_xz, linewidth=1.2, zorder=3)

                self._render_rays_and_annotations_xz(stack, surfaces)

            if not getattr(self, '_force_reset_view', False) and "xz" in self.saved_limits:
                x_lims, y_lims = self.saved_limits["xz"]
                if np.isfinite(x_lims).all() and np.isfinite(y_lims).all() and x_lims[0] != x_lims[1] and y_lims[0] != y_lims[1]:
                    self.ax_xz.set_xlim(x_lims)
                    self.ax_xz.set_ylim(y_lims)
                else:
                    self._apply_aspect(self.ax_xz)
            else:
                self._apply_aspect(self.ax_xz)

        # 3. Render View 3: Front View Plane (X-Y)
        # ----------------------------------------------------
        if self.ax_xy_front is not None:
            self.ax_xy_front.set_facecolor(bg_ax)
            self.ax_xy_front.set_title("Front View (X-Y) [Projected]", fontsize=10, fontweight='bold', color=fg_text)
            self.ax_xy_front.set_xlabel("X Position (mm)", color=fg_subtext)
            self.ax_xy_front.set_ylabel("Y Position (mm)", color=fg_subtext)
            self.ax_xy_front.grid(True, linestyle='--', alpha=0.35 if is_dark else 0.5, color=grid_col)
            self.ax_xy_front.tick_params(colors=fg_subtext)
            for spine in self.ax_xy_front.spines.values():
                spine.set_color(spine_col)
            self.ax_xy_front.axhline(0, color=axis_line_col, linestyle=':', alpha=0.6)
            self.ax_xy_front.axvline(0, color=axis_line_col, linestyle=':', alpha=0.6)

            if stack and len(surfaces) > 0:
                for i, surf in enumerate(surfaces):
                    if bool(getattr(surf, "is_air_space", False)):
                        continue
                    sd = surf.semi_diameter if surf.semi_diameter > 0 else 15.0
                    
                    if getattr(surf, 'aperture_type', 'Circular') == 'Rectangular':
                        xw = surf.aperture_x_half if getattr(surf, 'aperture_x_half', 0) > 0 else sd
                        yw = surf.aperture_y_half if getattr(surf, 'aperture_y_half', 0) > 0 else sd
                        x_local = np.array([-xw, xw, xw, -xw, -xw])
                        y_local = np.array([-yw, -yw, yw, yw, -yw])
                        z_local = np.zeros_like(x_local)
                    else:
                        # Create a 2D circle in local coordinates (X, Y)
                        theta = np.linspace(0, 2*np.pi, 100)
                        x_local = sd * np.cos(theta)
                        y_local = sd * np.sin(theta)
                        z_local = np.zeros_like(theta)
                    
                    if hasattr(surf, 'sag') and surf.surface_type != SurfaceType.CAD:
                        try:
                            # sag takes r = sqrt(x^2 + y^2) which is just sd
                            r_vals = np.sqrt(x_local**2 + y_local**2)
                            z_local = np.array([surf.sag(r) for r in r_vals])
                        except Exception:
                            pass
                            
                    local_pts = np.vstack([x_local, y_local, z_local])
                    
                    if not hasattr(surf, 'global_origin'):
                        stack._update_accumulated_positions()
                        
                    try:
                        eff_orig, eff_R = get_effective_pose(surf)
                        global_pts = eff_orig[:, None] + eff_R @ local_pts
                    except Exception:
                        global_pts = surf.global_origin[:, None] + surf.global_R @ local_pts
                    
                    x_glob = global_pts[0, :]
                    y_glob = global_pts[1, :]
                    
                    color = ('#fb923c' if is_dark else 'orange') if surf.surface_type not in (SurfaceType.STOP, SurfaceType.DETECTOR) else (('#f87171' if is_dark else 'red') if surf.surface_type == SurfaceType.STOP else ('#c084fc' if is_dark else 'purple'))
                    if "MIRROR" in (surf.glass.upper() if surf.glass else "") or str(surf.surface_type).upper() == "MIRROR":
                        color = '#94a3b8' if is_dark else 'gray'
                    linestyle = '-' if surf.surface_type not in (SurfaceType.STOP, SurfaceType.DETECTOR) else '--'
                    self.ax_xy_front.plot(x_glob, y_glob, color=color, linestyle=linestyle, linewidth=1.5)

                    surf_is_mod = is_plm(surf) or is_dmd(surf) or (getattr(surf, 'plm_config', None) is not None)
                    plm_c = getattr(surf, 'plm_config', None)
                    if surf_is_mod and plm_c is not None and getattr(plm_c, 'pattern', None) is not None and plm_c.pattern.enabled:
                        try:
                            from matplotlib.colors import ListedColormap
                            cur_st = active_state_name(surf) or "ON"
                            maps = generate_plm_2d_maps(plm_c, mechanical_state=cur_st, nx=120, ny=80)
                            if plm_c.is_dmd:
                                p_data = maps["state_4bit"]
                                cmap_m = ListedColormap(['#1e293b', '#38bdf8']) if is_dark else ListedColormap(['#cbd5e1', '#0284c7'])
                            else:
                                p_data = maps["phase_rad"]
                                cmap_m = 'twilight'

                            grid_local = np.vstack([maps["X"].ravel(), maps["Y"].ravel(), np.zeros_like(maps["X"].ravel())])
                            grid_glob = eff_orig[:, None] + eff_R @ grid_local
                            X_g = grid_glob[0, :].reshape(maps["X"].shape)
                            Y_g = grid_glob[1, :].reshape(maps["Y"].shape)
                            self.ax_xy_front.pcolormesh(X_g, Y_g, p_data, shading='auto', cmap=cmap_m, alpha=0.72, zorder=2)

                            if getattr(self, "show_labels", True):
                                dev_tag = "TI-DMD" if plm_c.is_dmd else "TI-PLM"
                                pat_name = plm_c.pattern.pattern_type.replace('_', ' ').title()
                                if plm_c.pattern.pattern_type.lower() == "hadamard":
                                    pat_name += f" (H{plm_c.pattern.hadamard_order}, r{plm_c.pattern.hadamard_row_index}, {getattr(plm_c.pattern, 'hadamard_axis', 'X')})"
                                elif "column" in plm_c.pattern.pattern_type.lower() or plm_c.pattern.pattern_type.lower() == "column_range":
                                    c_ranges = getattr(plm_c.pattern, 'column_ranges', None)
                                    inv = getattr(plm_c.pattern, 'column_invert', False)
                                    ang = getattr(plm_c.pattern, 'pattern_angle_deg', 0.0)
                                    ang_str = f" @ {ang:+.0f}°" if abs(ang) > 0.01 else ""
                                    if c_ranges and len(c_ranges) > 1:
                                        pat_name += f" ({len(c_ranges)} Ranges{ang_str})"
                                    elif c_ranges and len(c_ranges) == 1:
                                        r0 = c_ranges[0]
                                        st_val = r0.get("state", 1)
                                        pat_name += f" (Cols {r0.get('start', 0)}..{r0.get('end', 0)} [S{st_val}]{ang_str}{' Inv' if inv else ''})"
                                    else:
                                        c_s = getattr(plm_c.pattern, 'column_start', 0)
                                        c_e = getattr(plm_c.pattern, 'column_end', 0)
                                        st_val = getattr(plm_c.pattern, 'column_state', 1)
                                        st_str = f" [S{st_val}]" if not plm_c.is_dmd or st_val != 1 else ""
                                        pat_name += f" (Cols {c_s}..{c_e}{st_str}{ang_str}{' Inv' if inv else ''})"
                                self.ax_xy_front.annotate(
                                    f" {dev_tag} Mask: {pat_name} ",
                                    xy=(float(np.mean(x_glob)), float(np.max(y_glob))),
                                    xytext=(0, 6),
                                    textcoords="offset points",
                                    ha='center',
                                    va='bottom',
                                    fontsize=8,
                                    fontweight='bold',
                                    color='#38bdf8' if is_dark else '#0369a1',
                                    bbox=dict(
                                        boxstyle='round,pad=0.25',
                                        facecolor='#0f172a' if is_dark else '#e0f2fe',
                                        edgecolor='#38bdf8' if is_dark else '#0284c7',
                                        alpha=0.92,
                                        linewidth=1.0
                                    ),
                                    zorder=10
                                )
                        except Exception:
                            pass

                    stype_up = str(surf.surface_type).upper()
                    if "STOP" in stype_up or getattr(surf, 'is_stop', False) or getattr(surf, 'is_aperture_stop', False):
                        if getattr(self, "show_labels", True):
                            self.ax_xy_front.annotate(
                                f" Stop: Semi-D={surf.semi_diameter:.2f} mm ",
                                xy=(float(x_glob[-1]), float(y_glob[-1])), xytext=(4, 4), textcoords="offset points",
                                fontsize=8, fontweight='bold', color='#fca5a5' if is_dark else '#b71c1c',
                                bbox=dict(boxstyle='round,pad=0.2', facecolor='#450a0a' if is_dark else '#ffebee', edgecolor='#ef4444', alpha=0.9, linewidth=0.9),
                                zorder=10
                            )
                    
                self._render_rays_and_annotations_xy(stack, surfaces)

            self.ax_xy_front.set_aspect('equal', 'datalim')

        self._last_geom_fingerprint = curr_geom_fingerprint
        self._force_reset_view = False
        if not can_reuse:
            self.apply_tight_layout()
        if hasattr(self.canvas, "draw_idle"):
            self.canvas.draw_idle()
        elif hasattr(self.canvas, "draw"):
            self.canvas.draw()

    def update_system(self, stack_or_system: Any = None) -> bool:
        """Update system layout view."""
        if stack_or_system is not None:
            if isinstance(stack_or_system, SurfaceStack):
                self.surface_stack = stack_or_system
            elif hasattr(stack_or_system, "stack"):
                self.surface_stack = stack_or_system.stack
        return self.refresh_layout()


DualViewWindow = DualViewWidget
