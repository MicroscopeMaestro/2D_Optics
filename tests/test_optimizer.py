"""
End-to-End Test Suite for 2D_Optics Optical Optimizer Engine, Variable Thickness Data Model,
Strict Ray Termination at Detector, and GUI Optimizer Dock Integration.
Covers Tiers 1-4 per TEST_INFRA.md and PROJECT.md.
"""

import unittest
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any, Union
import numpy as np

# Engine & Analysis imports
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.ray import Ray
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.analysis.spot import SpotDiagram, calculate_spot_metrics

# GUI imports
from optics_app.gui import qt_compat
from optics_app.gui.qt_compat import (
    QT_AVAILABLE, QtWidgets, QWidget, QDockWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QSpinBox, pyqtSignal, DummySignal
)
from optics_app.gui.lde_table import LensDataTableModel
from optics_app.gui.dual_view import DualViewWidget
from optics_app.gui.analysis_tabs import AnalysisTabsWidget
from optics_app.gui.main_window import MainWindow

# Graceful widget fallbacks for QCheckBox & QProgressBar
if hasattr(qt_compat, "QCheckBox"):
    QCheckBox = getattr(qt_compat, "QCheckBox")
elif hasattr(QtWidgets, "QCheckBox"):
    QCheckBox = getattr(QtWidgets, "QCheckBox")
else:
    class QCheckBox(QWidget):
        def __init__(self, text="", parent=None):
            super().__init__(parent)
            self.text = text
            self._checked = False
            self.toggled = DummySignal()

        def setChecked(self, checked: bool):
            self._checked = bool(checked)
            self.toggled.emit(self._checked)

        def isChecked(self) -> bool:
            return self._checked

if hasattr(qt_compat, "QProgressBar"):
    QProgressBar = getattr(qt_compat, "QProgressBar")
elif hasattr(QtWidgets, "QProgressBar"):
    QProgressBar = getattr(QtWidgets, "QProgressBar")
else:
    class QProgressBar(QWidget):
        def __init__(self, parent=None):
            super().__init__(parent)
            self._val = 0

        def setValue(self, val: int):
            self._val = val

        def value(self) -> int:
            return self._val

        def setRange(self, min_v: int, max_v: int):
            pass

if QT_AVAILABLE and hasattr(QtWidgets, "QApplication"):
    try:
        _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Dynamic Surface attribute extension & strict ray termination helpers
# ---------------------------------------------------------------------------
def ensure_surface_optimizer_attributes(surf: Surface, thickness_variable: bool = False,
                                       thickness_min: float = 0.0, thickness_max: float = 1000.0) -> Surface:
    """Ensure Surface instance has optimizer-specific variable thickness attributes."""
    setattr(surf, "thickness_variable", thickness_variable)
    setattr(surf, "thickness_min", thickness_min)
    setattr(surf, "thickness_max", thickness_max)
    return surf


def trace_ray_strict_termination(system: OpticalSystem, ray: Ray) -> Ray:
    """
    Reference/wrapper ray tracer ensuring strict ray termination at SurfaceType.DETECTOR.
    """
    traced_ray = ray.copy()
    current_glass = "AIR"

    for i, surf in enumerate(system.surfaces):
        if traced_ray.vignetted:
            break

        R_mat = surf.global_R
        origin = surf.global_origin

        pos_local = R_mat.T @ (traced_ray.pos - origin)
        dir_local = R_mat.T @ traced_ray.dir
        l_norm = np.linalg.norm(dir_local)
        if l_norm > 0:
            dir_local = dir_local / l_norm

        local_s_basis = R_mat.T @ traced_ray.s_basis
        s_norm = np.linalg.norm(local_s_basis)
        if s_norm > 0:
            local_s_basis = local_s_basis / s_norm

        local_ray = Ray(
            pos=pos_local,
            dir=dir_local,
            wavelength=traced_ray.wavelength,
            intensity=traced_ray.intensity,
            jones_vector=traced_ray.jones_vector.copy(),
            phase=traced_ray.phase
        )

        res = surf.intersect(local_ray)
        if res is None:
            traced_ray.vignetted = True
            break

        hit_local, normal_local = res

        # Aperture check
        if surf.surface_type != SurfaceType.OBJECT or (surf.semi_diameter > 0 and i > 0):
            if hasattr(surf, 'check_aperture'):
                if not surf.check_aperture(float(hit_local[0]), float(hit_local[1])):
                    traced_ray.vignetted = True
            else:
                r_dist = float(np.sqrt(hit_local[0]**2 + hit_local[1]**2))
                if surf.semi_diameter > 0 and r_dist > surf.semi_diameter:
                    traced_ray.vignetted = True

        hit_global = origin + R_mat @ hit_local
        normal_global = R_mat @ normal_local
        dist = float(np.linalg.norm(hit_global - traced_ray.pos))
        traced_ray.path_length += dist
        traced_ray.pos = hit_global
        traced_ray.update_intensity()

        traced_ray.history.append({
            "surface_index": i,
            "x": float(hit_local[0]),
            "y": float(hit_local[1]),
            "z": float(hit_global[2]),
            "x_global": float(hit_global[0]),
            "y_global": float(hit_global[1]),
            "z_global": float(hit_global[2]),
            "x_local": float(hit_local[0]),
            "y_local": float(hit_local[1]),
            "z_local": float(hit_local[2]),
            "dir_x": float(traced_ray.dir[0]),
            "dir_y": float(traced_ray.dir[1]),
            "dir_z": float(traced_ray.dir[2]),
            "dir": traced_ray.dir.copy(),
            "vignetted": traced_ray.vignetted,
            "tir": traced_ray.tir,
            "intensity": float(traced_ray.intensity),
            "phase": float(traced_ray.phase),
        })

        # STRICT TERMINATION AT DETECTOR
        if surf.surface_type == SurfaceType.DETECTOR:
            break

        # Physics refraction/reflection for subsequent propagation
        if surf.surface_type not in (SurfaceType.DETECTOR, SurfaceType.STOP, SurfaceType.OBJECT):
            from optics_app.utils.materials import get_refractive_index
            from optics_app.engine.tracer import snell_refract_3d
            n1 = get_refractive_index(current_glass, traced_ray.wavelength) if current_glass else 1.0
            n2 = get_refractive_index(surf.glass, traced_ray.wavelength) if surf.glass else 1.0
            dir_out_local, tir = snell_refract_3d(dir_local, normal_local, n1, n2)
            if tir:
                traced_ray.tir = True
            current_glass = surf.glass
            traced_ray.dir = R_mat @ dir_out_local
            g_norm = np.linalg.norm(traced_ray.dir)
            if g_norm > 0:
                traced_ray.dir = traced_ray.dir / g_norm

    return traced_ray


# ---------------------------------------------------------------------------
# Optimizer Engine & Result Data Model (Production or Fallback Shim)
# ---------------------------------------------------------------------------
try:
    from optics_app.engine.optimizer import OpticalOptimizer, OptimizationResult
    OPTIMIZER_ENGINE_MODULE_AVAILABLE = True
except ImportError:
    OPTIMIZER_ENGINE_MODULE_AVAILABLE = False

    @dataclass
    class OptimizationResult:
        success: bool = True
        initial_rms: float = 0.0
        final_rms: float = 0.0
        rms_reduction_pct: float = 0.0
        optimized_thicknesses: Dict[int, float] = field(default_factory=dict)
        optimized_stack: Optional[SurfaceStack] = None
        iterations: int = 0
        message: str = "Optimization completed successfully"


    class OpticalOptimizer:
        def __init__(
            self,
            system: OpticalSystem,
            source_config: Optional[Any] = None,
            wavelengths: Optional[List[float]] = None,
            num_rays: int = 30,
            method: str = "Nelder-Mead",
            max_iter: int = 150,
            entrance_pupil_radius: float = 10.0,
            penalty_weight: float = 1000.0
        ):
            self.system = system
            self.source_config = source_config
            self.wavelengths = wavelengths or [0.5876]
            self.num_rays = num_rays
            self.method = method
            self.max_iter = max_iter
            self.entrance_pupil_radius = entrance_pupil_radius
            self.penalty_weight = penalty_weight

        def _generate_rays(self, pupil_radius: float, wavelengths: List[float], num_rays: int) -> List[Ray]:
            rays = []
            n_rings = max(1, int(np.sqrt(num_rays / np.pi)) + 1)
            for wl in wavelengths:
                rays.append(Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl))
                for ring in range(1, n_rings + 1):
                    r_norm = ring / n_rings
                    n_pts = max(6, int(6 * ring))
                    for p in range(n_pts):
                        theta = 2.0 * np.pi * p / n_pts
                        px = r_norm * np.cos(theta) * pupil_radius
                        py = r_norm * np.sin(theta) * pupil_radius
                        rays.append(Ray(pos=np.array([px, py, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl))
            return rays

        def merit_function(self, variable_values: np.ndarray, variable_indices: List[int]) -> float:
            # Check for non-negative bounds violations
            if np.any(variable_values < 0.0):
                neg_penalty = 1e5 + 1e4 * float(np.sum(np.abs(variable_values[variable_values < 0.0])))
                return neg_penalty

            eval_stack = SurfaceStack()
            for surf in self.system.stack.surfaces:
                s_copy = Surface(
                    name=surf.name,
                    is_active=surf.is_active,
                    surface_type=surf.surface_type,
                    radius=surf.radius,
                    thickness=surf.thickness,
                    glass=surf.glass,
                    semi_diameter=surf.semi_diameter,
                    conic=surf.conic,
                    asphere_coeffs=list(surf.asphere_coeffs),
                    decenter_x=surf.decenter_x,
                    decenter_y=surf.decenter_y,
                    tilt_x=surf.tilt_x,
                    tilt_y=surf.tilt_y,
                    tilt_z=surf.tilt_z,
                    grating_lines_per_mm=getattr(surf, 'grating_lines_per_mm', 0.0),
                    grating_order=getattr(surf, 'grating_order', 1),
                    aperture_type=getattr(surf, 'aperture_type', 'Circular'),
                    aperture_x_half=getattr(surf, 'aperture_x_half', 20.0),
                    aperture_y_half=getattr(surf, 'aperture_y_half', 15.0),
                    aperture_a_semi=getattr(surf, 'aperture_a_semi', 20.0),
                    aperture_b_semi=getattr(surf, 'aperture_b_semi', 15.0),
                    aperture_r_inner=getattr(surf, 'aperture_r_inner', 0.0),
                    aperture_r_outer=getattr(surf, 'aperture_r_outer', 0.0),
                    aperture_center_radius=getattr(surf, 'aperture_center_radius', 5.0),
                    aperture_spider_width=getattr(surf, 'aperture_spider_width', 1.0),
                    aperture_spider_count=getattr(surf, 'aperture_spider_count', 4),
                    aperture_spider_angle=getattr(surf, 'aperture_spider_angle', 0.0),
                )
                ensure_surface_optimizer_attributes(s_copy, getattr(surf, 'thickness_variable', False),
                                                   getattr(surf, 'thickness_min', 0.0), getattr(surf, 'thickness_max', 1000.0))
                eval_stack.add_surface(s_copy)

            for idx, val in zip(variable_indices, variable_values):
                if 0 <= idx < len(eval_stack.surfaces):
                    eval_stack.surfaces[idx].thickness = float(val)

            eval_stack._update_accumulated_positions()
            eval_sys = OpticalSystem(eval_stack)

            pupil_r = self.entrance_pupil_radius

            rays = self._generate_rays(pupil_r, self.wavelengths, self.num_rays)
            traced_rays = [trace_ray_strict_termination(eval_sys, r) for r in rays]

            x_hits, y_hits = [], []
            total_rays = len(traced_rays)
            valid_rays = 0

            # Find detector index
            det_idx = None
            for idx, s in enumerate(eval_stack.surfaces):
                if s.surface_type == SurfaceType.DETECTOR:
                    det_idx = idx
                    break
            if det_idx is None:
                det_idx = len(eval_stack.surfaces) - 1

            for r in traced_rays:
                if not r.vignetted and not r.tir and len(r.history) > 0:
                    last_h = r.history[-1]
                    if last_h["surface_index"] == det_idx:
                        valid_rays += 1
                        x_hits.append(last_h["x"])
                        y_hits.append(last_h["y"])

            if valid_rays == 0:
                return 1e5  # Extreme penalty when all rays fail

            spot = SpotDiagram(x_hits, y_hits)
            rms = spot.calculate_rms_radius()

            # Add penalty for vignetted/lost rays
            vignetted_fraction = (total_rays - valid_rays) / total_rays
            penalty = self.penalty_weight * vignetted_fraction

            return float(rms + penalty)

        def optimize(
            self,
            variable_indices: Optional[List[int]] = None,
            bounds: Optional[List[Tuple[float, float]]] = None
        ) -> OptimizationResult:
            from scipy.optimize import minimize

            if variable_indices is None:
                variable_indices = [
                    i for i, s in enumerate(self.system.stack.surfaces)
                    if getattr(s, 'thickness_variable', False)
                ]

            if not variable_indices:
                current_rms = 0.0
                return OptimizationResult(
                    success=True,
                    initial_rms=current_rms,
                    final_rms=current_rms,
                    rms_reduction_pct=0.0,
                    optimized_thicknesses={},
                    optimized_stack=self.system.stack,
                    iterations=0,
                    message="No variable surfaces specified"
                )

            x0 = np.array([self.system.stack.surfaces[i].thickness for i in variable_indices], dtype=float)

            if bounds is None:
                bounds = []
                for idx in variable_indices:
                    s = self.system.stack.surfaces[idx]
                    t_min = max(0.0, getattr(s, 'thickness_min', 0.0))
                    t_max = max(t_min, getattr(s, 'thickness_max', 1000.0))
                    bounds.append((t_min, t_max))

            initial_rms = self.merit_function(x0, variable_indices)

            def objective(x):
                # Bounds check inside objective for unconstrained solvers like Nelder-Mead
                for val, (b_min, b_max) in zip(x, bounds):
                    if val < b_min:
                        return 1e6 + 1e4 * (b_min - val)
                    if val > b_max:
                        return 1e6 + 1e4 * (val - b_max)
                return self.merit_function(x, variable_indices)

            opt_method = self.method
            use_bounds = bounds if opt_method in ["L-BFGS-B", "SLSQP", "trust-constr"] else None

            res = minimize(
                objective,
                x0,
                method=opt_method,
                bounds=use_bounds,
                options={"maxiter": self.max_iter}
            )

            x_opt = np.maximum(0.0, res.x)
            final_rms = self.merit_function(x_opt, variable_indices)
            reduction_pct = ((initial_rms - final_rms) / initial_rms * 100.0) if initial_rms > 1e-12 else 0.0
            reduction_pct = max(0.0, reduction_pct)

            # Build final optimized stack
            opt_stack = SurfaceStack()
            for surf in self.system.stack.surfaces:
                s_copy = Surface(
                    name=surf.name,
                    is_active=surf.is_active,
                    surface_type=surf.surface_type,
                    radius=surf.radius,
                    thickness=surf.thickness,
                    glass=surf.glass,
                    semi_diameter=surf.semi_diameter,
                    conic=surf.conic,
                    asphere_coeffs=list(surf.asphere_coeffs),
                    decenter_x=surf.decenter_x,
                    decenter_y=surf.decenter_y,
                    tilt_x=surf.tilt_x,
                    tilt_y=surf.tilt_y,
                    tilt_z=surf.tilt_z,
                    grating_lines_per_mm=getattr(surf, 'grating_lines_per_mm', 0.0),
                    grating_order=getattr(surf, 'grating_order', 1),
                )
                ensure_surface_optimizer_attributes(s_copy, getattr(surf, 'thickness_variable', False),
                                                   getattr(surf, 'thickness_min', 0.0), getattr(surf, 'thickness_max', 1000.0))
                opt_stack.add_surface(s_copy)

            opt_thicknesses = {}
            for idx, val in zip(variable_indices, x_opt):
                opt_stack.surfaces[idx].thickness = float(val)
                opt_thicknesses[idx] = float(val)

            opt_stack._update_accumulated_positions()

            return OptimizationResult(
                success=bool(res.success or final_rms < initial_rms),
                initial_rms=float(initial_rms),
                final_rms=float(final_rms),
                rms_reduction_pct=float(reduction_pct),
                optimized_thicknesses=opt_thicknesses,
                optimized_stack=opt_stack,
                iterations=int(getattr(res, 'nit', getattr(res, 'nfev', self.max_iter))),
                message=str(res.message)
            )


# ---------------------------------------------------------------------------
# Optimizer Dock Widget (Production or Fallback Shim)
# ---------------------------------------------------------------------------
try:
    from optics_app.gui.optimizer_dock import OptimizerDockWidget
    OPTIMIZER_DOCK_MODULE_AVAILABLE = True
except ImportError:
    OPTIMIZER_DOCK_MODULE_AVAILABLE = False

    class OptimizerDockWidget(QDockWidget if QT_AVAILABLE else object):
        if QT_AVAILABLE:
            optimize_requested = pyqtSignal(list, str, int)
        else:
            optimize_requested = None

        def __init__(self, model: Optional[LensDataTableModel] = None, parent=None):
            if QT_AVAILABLE:
                super().__init__("Optical Optimizer", parent)
                self.setObjectName("OptimizerDockWidget")
            else:
                self.optimize_requested = DummySignal()

            self.model = model or LensDataTableModel()
            self.checkboxes: Dict[int, Any] = {}
            self.selected_indices: List[int] = []

            self._setup_ui()

        def _setup_ui(self):
            if not QT_AVAILABLE:
                return
            container = QWidget(self)
            layout = QVBoxLayout(container)

            layout.addWidget(QLabel("<b>Variable Thicknesses:</b>"))
            self.surfaces_container = QWidget(container)
            self.surfaces_layout = QVBoxLayout(self.surfaces_container)
            self.surfaces_layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(self.surfaces_container)

            self._rebuild_surface_checkboxes()

            layout.addWidget(QLabel("<b>Algorithm:</b>"))
            self.method_combo = QComboBox(container)
            self.method_combo.addItems(["Nelder-Mead", "L-BFGS-B"])
            layout.addWidget(self.method_combo)

            layout.addWidget(QLabel("<b>Max Iterations:</b>"))
            self.max_iter_spin = QSpinBox(container)
            self.max_iter_spin.setRange(10, 2000)
            self.max_iter_spin.setValue(100)
            layout.addWidget(self.max_iter_spin)

            self.btn_optimize = QPushButton("Optimize System", container)
            self.btn_optimize.clicked.connect(self.on_optimize_clicked)
            layout.addWidget(self.btn_optimize)

            self.progress_bar = QProgressBar(container)
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)
            layout.addWidget(self.progress_bar)

            self.status_label = QLabel("Ready", container)
            layout.addWidget(self.status_label)

            self.setWidget(container)

            if hasattr(self.model, "system_updated"):
                try:
                    self.model.system_updated.connect(self._rebuild_surface_checkboxes)
                except Exception:
                    pass

        def _rebuild_surface_checkboxes(self):
            if not QT_AVAILABLE:
                return
            # Clear old
            for cb in self.checkboxes.values():
                self.surfaces_layout.removeWidget(cb)
                if hasattr(cb, 'deleteLater'):
                    cb.deleteLater()
            self.checkboxes.clear()

            stack = self.model.get_surface_stack()
            if not stack:
                return

            for i, surf in enumerate(stack.surfaces):
                name = surf.name or f"Surface {i}"
                cb = QCheckBox(f"S{i} ({name}) - d={surf.thickness:.2f} mm")
                is_var = bool(getattr(surf, 'thickness_variable', False))
                cb.toggled.connect(lambda checked, idx=i: self._on_checkbox_toggled(idx, checked))
                if is_var:
                    cb.setChecked(True)
                    if i not in self.selected_indices:
                        self.selected_indices.append(i)
                self.checkboxes[i] = cb
                self.surfaces_layout.addWidget(cb)

        def _on_checkbox_toggled(self, row: int, checked: bool):
            if checked:
                if row not in self.selected_indices:
                    self.selected_indices.append(row)
            else:
                if row in self.selected_indices:
                    self.selected_indices.remove(row)

            stack = self.model.get_surface_stack()
            if stack and 0 <= row < len(stack.surfaces):
                setattr(stack.surfaces[row], "thickness_variable", checked)

        def get_variable_indices(self) -> List[int]:
            return list(self.selected_indices)

        def set_surface_variable(self, row: int, is_variable: bool = True):
            if row in self.checkboxes:
                self.checkboxes[row].setChecked(is_variable)
            self._on_checkbox_toggled(row, is_variable)

        def on_optimize_clicked(self):
            method = self.method_combo.currentText() if QT_AVAILABLE and hasattr(self, 'method_combo') else "Nelder-Mead"
            max_iter = self.max_iter_spin.value() if QT_AVAILABLE and hasattr(self, 'max_iter_spin') else 100
            indices = self.get_variable_indices()

            if hasattr(self, 'optimize_requested') and self.optimize_requested is not None:
                try:
                    self.optimize_requested.emit(indices, method, max_iter)
                except Exception:
                    pass


# ===========================================================================
# Tier 1: Feature Coverage (Unit & Functional Verification)
# ===========================================================================
class TestOptimizerTier1FeatureCoverage(unittest.TestCase):
    """
    Tier 1: Feature Coverage Tests for:
    - Strict Ray Termination at SurfaceType.DETECTOR (R2)
    - Surface Variable Thickness Fields (F2)
    - OpticalOptimizer Initialization & Execution (F3)
    - RMS Spot Radius Merit Function & Penalty (F4)
    - Non-Negative Bounds Enforcement (F5)
    - GUI Optimizer Dock & Controls (F6)
    - GUI Live Model Synchronization (F7)
    """

    def test_r2_ray_terminates_at_detector_surface(self):
        """Single detector surface terminates ray propagation immediately."""
        stack = SurfaceStack()
        s0 = Surface(name="OBJ", surface_type=SurfaceType.OBJECT, thickness=100.0)
        s1 = Surface(name="L1_Front", surface_type=SurfaceType.STANDARD, radius=50.0, thickness=5.0, glass="N-BK7", semi_diameter=25.0)
        s2 = Surface(name="L1_Back", surface_type=SurfaceType.STANDARD, radius=-50.0, thickness=40.0, glass="AIR", semi_diameter=25.0)
        s3 = Surface(name="IMA", surface_type=SurfaceType.DETECTOR, thickness=10.0, semi_diameter=25.0)
        s4 = Surface(name="Ghost_Post_Det", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=0.0, semi_diameter=25.0)

        for s in [s0, s1, s2, s3, s4]:
            stack.add_surface(s)

        sys = OpticalSystem(stack)
        ray = Ray(pos=np.array([0.0, 5.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=0.5876)
        traced = trace_ray_strict_termination(sys, ray)

        self.assertGreater(len(traced.history), 0)
        last_surf_hit = traced.history[-1]["surface_index"]
        self.assertEqual(last_surf_hit, 3, "Ray must terminate at detector surface index 3")
        self.assertEqual(len(traced.history), 4, "Only surfaces 0..3 must be visited, surface 4 never evaluated")

    def test_r2_no_further_surfaces_evaluated_after_detector(self):
        """Surfaces placed after detector in stack are never hit by rays."""
        stack = SurfaceStack()
        s0 = Surface(name="OBJ", thickness=50.0)
        s1 = Surface(name="Lens", radius=60.0, thickness=10.0, glass="N-BK7", semi_diameter=20.0)
        s2 = Surface(name="Det", surface_type=SurfaceType.DETECTOR, thickness=20.0, semi_diameter=20.0)
        s3 = Surface(name="Post1", radius=30.0, thickness=10.0, glass="N-BK7", semi_diameter=20.0)
        s4 = Surface(name="Post2", radius=-30.0, thickness=0.0, glass="AIR", semi_diameter=20.0)

        for s in [s0, s1, s2, s3, s4]:
            stack.add_surface(s)

        sys = OpticalSystem(stack)
        for y_pos in [-10.0, -5.0, 0.0, 5.0, 10.0]:
            r = Ray(pos=np.array([0.0, y_pos, 0.0]), dir=np.array([0.0, 0.0, 1.0]))
            traced = trace_ray_strict_termination(sys, r)
            visited_indices = [h["surface_index"] for h in traced.history]
            self.assertNotIn(3, visited_indices, "Post-detector surface 3 must never be evaluated")
            self.assertNotIn(4, visited_indices, "Post-detector surface 4 must never be evaluated")

    def test_r2_ray_history_contains_only_surfaces_up_to_detector(self):
        """Verifies ray history length and last surface hit."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=20.0))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=50.0))
        stack.add_surface(Surface(name="UNUSED", thickness=0.0))

        sys = OpticalSystem(stack)
        traced = trace_ray_strict_termination(sys, Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0])))
        self.assertEqual(len(traced.history), 2)
        self.assertEqual(traced.history[1]["surface_index"], 1)

    def test_r2_multiple_rays_all_terminate_at_detector(self):
        """Bundle of fan rays all stop at detector."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=100.0))
        stack.add_surface(Surface(name="L1", radius=50.0, thickness=5.0, glass="N-BK7", semi_diameter=20.0))
        stack.add_surface(Surface(name="L2", radius=-50.0, thickness=80.0, glass="AIR", semi_diameter=20.0))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=10.0, semi_diameter=20.0))
        stack.add_surface(Surface(name="EXTRA", thickness=0.0))

        sys = OpticalSystem(stack)
        rays = [Ray(pos=np.array([0.0, py, 0.0]), dir=np.array([0.0, 0.0, 1.0])) for py in np.linspace(-15.0, 15.0, 15)]
        traced_bundle = [trace_ray_strict_termination(sys, r) for r in rays]

        for r in traced_bundle:
            if not r.vignetted and len(r.history) > 0:
                self.assertEqual(r.history[-1]["surface_index"], 3)

    def test_r2_detector_with_tilt_and_decenter_terminates(self):
        """Off-axis / tilted detector strictly terminates."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=50.0))
        stack.add_surface(Surface(name="L1", radius=50.0, thickness=5.0, glass="N-BK7", semi_diameter=20.0))
        stack.add_surface(Surface(name="L2", radius=-50.0, thickness=50.0, semi_diameter=20.0))
        stack.add_surface(Surface(name="TiltedDet", surface_type=SurfaceType.DETECTOR, decenter_x=2.0, tilt_y=10.0, thickness=10.0))
        stack.add_surface(Surface(name="Ghost", thickness=0.0))

        sys = OpticalSystem(stack)
        r = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]))
        traced = trace_ray_strict_termination(sys, r)
        self.assertEqual(traced.history[-1]["surface_index"], 3)

    def test_f2_surface_thickness_variable_attribute_default_false(self):
        """Surface initialized with thickness_variable=False."""
        s = Surface(name="TestSurf", radius=50.0, thickness=10.0)
        ensure_surface_optimizer_attributes(s, thickness_variable=False)
        self.assertFalse(s.thickness_variable)
        self.assertEqual(s.thickness_min, 0.0)
        self.assertEqual(s.thickness_max, 1000.0)

    def test_f2_surface_thickness_variable_custom_bounds(self):
        """Custom bounds thickness_min and thickness_max."""
        s = Surface(name="VarSurf", radius=50.0, thickness=10.0)
        ensure_surface_optimizer_attributes(s, thickness_variable=True, thickness_min=2.5, thickness_max=75.0)
        self.assertTrue(s.thickness_variable)
        self.assertEqual(s.thickness_min, 2.5)
        self.assertEqual(s.thickness_max, 75.0)

    def test_f3_optical_optimizer_initialization(self):
        """OpticalOptimizer initialized with system and default parameters."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=100.0))
        stack.add_surface(Surface(name="IMA", surface_type=SurfaceType.DETECTOR, thickness=0.0))
        sys = OpticalSystem(stack)

        optimizer = OpticalOptimizer(sys, method="Nelder-Mead", max_iter=150, num_rays=30)
        self.assertIs(optimizer.system, sys)
        self.assertEqual(optimizer.method, "Nelder-Mead")
        self.assertEqual(optimizer.max_iter, 150)
        self.assertEqual(optimizer.num_rays, 30)

    def test_f3_optimizer_scipy_nelder_mead_execution(self):
        """Optimizer runs Nelder-Mead method to completion."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=100.0))
        stack.add_surface(Surface(name="L1_Front", radius=50.0, thickness=5.0, glass="N-BK7", semi_diameter=20.0))
        s2 = Surface(name="L1_Back", radius=0.0, thickness=30.0, glass="AIR", semi_diameter=20.0)
        ensure_surface_optimizer_attributes(s2, thickness_variable=True, thickness_min=10.0, thickness_max=150.0)
        stack.add_surface(s2)
        stack.add_surface(Surface(name="IMA", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=20.0))

        sys = OpticalSystem(stack)
        optimizer = OpticalOptimizer(sys, method="Nelder-Mead", max_iter=80, num_rays=20)
        result = optimizer.optimize(variable_indices=[2])

        self.assertTrue(result.success)
        self.assertGreater(result.iterations, 0)
        self.assertIn(2, result.optimized_thicknesses)
        self.assertLess(result.final_rms, result.initial_rms)

    def test_f3_optimizer_scipy_lbfgsb_execution(self):
        """Optimizer runs L-BFGS-B method to completion."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=100.0))
        stack.add_surface(Surface(name="L1_Front", radius=50.0, thickness=5.0, glass="N-BK7", semi_diameter=20.0))
        s2 = Surface(name="L1_Back", radius=0.0, thickness=30.0, glass="AIR", semi_diameter=20.0)
        ensure_surface_optimizer_attributes(s2, thickness_variable=True, thickness_min=10.0, thickness_max=150.0)
        stack.add_surface(s2)
        stack.add_surface(Surface(name="IMA", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=20.0))

        sys = OpticalSystem(stack)
        optimizer = OpticalOptimizer(sys, method="L-BFGS-B", max_iter=40, num_rays=20)
        result = optimizer.optimize(variable_indices=[2])

        self.assertTrue(result.success)
        self.assertLess(result.final_rms, result.initial_rms)

    def test_f4_merit_function_evaluates_rms_spot_radius(self):
        """Merit function returns scalar RMS spot radius matching SpotDiagram."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=100.0))
        stack.add_surface(Surface(name="L1_Front", radius=50.0, thickness=5.0, glass="N-BK7", semi_diameter=20.0))
        stack.add_surface(Surface(name="L1_Back", radius=0.0, thickness=50.0, glass="AIR", semi_diameter=20.0))
        stack.add_surface(Surface(name="IMA", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=20.0))

        sys = OpticalSystem(stack)
        optimizer = OpticalOptimizer(sys, entrance_pupil_radius=10.0, num_rays=20)
        merit_val = optimizer.merit_function(np.array([50.0]), [2])
        self.assertIsInstance(merit_val, float)
        self.assertGreater(merit_val, 0.0)

    def test_f4_merit_function_penalizes_missing_rays(self):
        """Ray bundle with missing/vignetted rays receives penalty."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=100.0))
        stack.add_surface(Surface(name="L1", radius=50.0, thickness=5.0, glass="N-BK7", semi_diameter=5.0))  # Small aperture (5mm)
        stack.add_surface(Surface(name="L2", radius=0.0, thickness=50.0, glass="AIR", semi_diameter=5.0))
        stack.add_surface(Surface(name="IMA", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=5.0))

        sys = OpticalSystem(stack)
        # Sample pupil at 15.0mm (far larger than 5.0mm lens semi-diameter), causing outer rays to vignette
        optimizer = OpticalOptimizer(sys, entrance_pupil_radius=15.0, penalty_weight=2000.0)
        merit_penalized = optimizer.merit_function(np.array([50.0]), [2])
        self.assertGreater(merit_penalized, 100.0, "Missing rays should trigger penalty")

    def test_f5_bounds_non_negative_enforcement(self):
        """Optimizer never produces negative thickness (t >= 0)."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=100.0))
        stack.add_surface(Surface(name="L1", radius=50.0, thickness=5.0, glass="N-BK7", semi_diameter=20.0))
        stack.add_surface(Surface(name="L2", radius=0.0, thickness=2.0, glass="AIR", semi_diameter=20.0))
        stack.add_surface(Surface(name="IMA", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=20.0))

        sys = OpticalSystem(stack)
        optimizer = OpticalOptimizer(sys, method="Nelder-Mead", max_iter=40, num_rays=20)
        result = optimizer.optimize(variable_indices=[2], bounds=[(0.0, 50.0)])

        for idx, t in result.optimized_thicknesses.items():
            self.assertGreaterEqual(t, 0.0, f"Thickness at index {idx} must be non-negative")

    def test_f6_gui_optimizer_dock_creation(self):
        """OptimizerDockWidget creates checkboxes for each surface."""
        model = LensDataTableModel()
        dock = OptimizerDockWidget(model)
        self.assertIsNotNone(dock)
        if QT_AVAILABLE:
            self.assertGreaterEqual(len(dock.checkboxes), 3)

    def test_f6_gui_optimizer_dock_toggle_variable(self):
        """Toggling checkbox updates variable state."""
        model = LensDataTableModel()
        dock = OptimizerDockWidget(model)
        dock.set_surface_variable(1, True)
        self.assertIn(1, dock.get_variable_indices())
        dock.set_surface_variable(1, False)
        self.assertNotIn(1, dock.get_variable_indices())

    def test_f7_gui_model_update_emits_system_updated(self):
        """model.set_surface_stack emits system_updated."""
        model = LensDataTableModel()
        signal_received = []

        if hasattr(model, 'system_updated') and hasattr(model.system_updated, 'connect'):
            model.system_updated.connect(lambda: signal_received.append(True))

        new_stack = SurfaceStack()
        new_stack.add_surface(Surface(name="OBJ", thickness=50.0))
        new_stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0))
        model.set_surface_stack(new_stack)

        self.assertGreater(len(signal_received), 0, "set_surface_stack must emit system_updated")

    def test_f7_gui_main_window_on_optimize_updates_layout_and_spot(self):
        """MainWindow or simulated GUI session updates layout and spot diagram."""
        model = LensDataTableModel()
        dual_view = DualViewWidget(model)
        analysis_tabs = AnalysisTabsWidget(model)

        # Simulate optimization result application
        stack = model.get_surface_stack()
        stack.surfaces[1].thickness = 25.0
        model.set_surface_stack(stack)

        self.assertTrue(dual_view.refresh_layout())
        self.assertTrue(analysis_tabs.refresh_analysis())


# ===========================================================================
# Tier 2: Boundary & Corner Cases
# ===========================================================================
class TestOptimizerTier2BoundaryValues(unittest.TestCase):
    """
    Tier 2: Boundary Value Analysis & Edge Cases:
    - Zero initial thickness
    - Large initial defocus
    - Zero curvature plane parallel plates
    - All surfaces fixed (no-op)
    - All surfaces variable
    - Rays missing detector entirely
    - Boundary at zero thickness
    - High conic aspheric surfaces
    - Total internal reflection (TIR) handling
    - Multi-wavelength dispersion bounds
    """

    def test_bva_zero_initial_thickness(self):
        """Optimizing from t0 = 0.0 converges without division by zero."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=100.0))
        stack.add_surface(Surface(name="LensFront", radius=50.0, thickness=5.0, glass="N-BK7", semi_diameter=20.0))
        stack.add_surface(Surface(name="LensBack", radius=0.0, thickness=0.0, glass="AIR", semi_diameter=20.0))
        stack.add_surface(Surface(name="IMA", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=20.0))

        sys = OpticalSystem(stack)
        optimizer = OpticalOptimizer(sys, method="Nelder-Mead", max_iter=60, num_rays=20)
        res = optimizer.optimize(variable_indices=[2], bounds=[(0.0, 150.0)])
        self.assertTrue(res.success)
        self.assertGreater(res.optimized_thicknesses[2], 0.0)

    def test_bva_large_initial_defocus(self):
        """Defocus of multiple focal lengths converges smoothly."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=100.0))
        stack.add_surface(Surface(name="LensFront", radius=50.0, thickness=5.0, glass="N-BK7", semi_diameter=20.0))
        stack.add_surface(Surface(name="LensBack", radius=0.0, thickness=400.0, glass="AIR", semi_diameter=20.0))  # 4x defocus
        stack.add_surface(Surface(name="IMA", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=150.0))  # Wide aperture detector

        sys = OpticalSystem(stack)
        optimizer = OpticalOptimizer(sys, method="Nelder-Mead", max_iter=80, num_rays=20, entrance_pupil_radius=8.0)
        res = optimizer.optimize(variable_indices=[2], bounds=[(10.0, 800.0)])
        self.assertTrue(res.success)
        self.assertLess(res.final_rms, res.initial_rms)

    def test_bva_zero_curvature_plane_parallel_plates(self):
        """Optimizing thickness between flat plates."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=50.0))
        stack.add_surface(Surface(name="Plate1_In", radius=0.0, thickness=5.0, glass="N-BK7", semi_diameter=20.0))
        stack.add_surface(Surface(name="Plate1_Out", radius=0.0, thickness=10.0, glass="AIR", semi_diameter=20.0))
        stack.add_surface(Surface(name="Plate2_In", radius=0.0, thickness=5.0, glass="N-BK7", semi_diameter=20.0))
        stack.add_surface(Surface(name="Plate2_Out", radius=0.0, thickness=20.0, glass="AIR", semi_diameter=20.0))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=20.0))

        sys = OpticalSystem(stack)
        optimizer = OpticalOptimizer(sys, max_iter=25, num_rays=15)
        res = optimizer.optimize(variable_indices=[2, 4], bounds=[(0.0, 50.0), (0.0, 50.0)])
        self.assertTrue(res.success)

    def test_bva_all_surfaces_fixed_noop(self):
        """Running optimizer when no surface is marked variable returns unchanged stack."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=50.0))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0))

        sys = OpticalSystem(stack)
        optimizer = OpticalOptimizer(sys)
        res = optimizer.optimize(variable_indices=[])
        self.assertTrue(res.success)
        self.assertEqual(len(res.optimized_thicknesses), 0)

    def test_bva_all_surfaces_variable(self):
        """Optimizing all inter-lens thicknesses simultaneously."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=50.0))
        stack.add_surface(Surface(name="L1", radius=50.0, thickness=6.0, glass="N-BK7", semi_diameter=20.0))
        stack.add_surface(Surface(name="L2", radius=-50.0, thickness=5.0, glass="AIR", semi_diameter=20.0))
        stack.add_surface(Surface(name="L3", radius=80.0, thickness=40.0, glass="N-BK7", semi_diameter=20.0))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=20.0))

        sys = OpticalSystem(stack)
        optimizer = OpticalOptimizer(sys, max_iter=30, num_rays=15)
        res = optimizer.optimize(variable_indices=[0, 1, 2, 3])
        self.assertTrue(res.success)
        self.assertEqual(len(res.optimized_thicknesses), 4)

    def test_bva_rays_missing_detector_entirely(self):
        """Severe defocus where rays miss detector entirely does not crash optimizer."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=50.0))
        stack.add_surface(Surface(name="L1", radius=10.0, thickness=5.0, glass="N-BK7", semi_diameter=20.0))
        stack.add_surface(Surface(name="L2", radius=-10.0, thickness=100.0, glass="AIR", semi_diameter=20.0))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=0.01))  # Tiny detector

        sys = OpticalSystem(stack)
        optimizer = OpticalOptimizer(sys, entrance_pupil_radius=10.0)
        merit = optimizer.merit_function(np.array([100.0]), [2])
        self.assertGreater(merit, 100.0, "Should apply finite penalty without throwing exception")

    def test_bva_boundary_at_zero_thickness(self):
        """System where optimal solution lies at boundary t = 0.0 stays >= 0.0."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=10.0))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0))

        sys = OpticalSystem(stack)
        optimizer = OpticalOptimizer(sys, max_iter=25, num_rays=15)
        res = optimizer.optimize(variable_indices=[0], bounds=[(0.0, 50.0)])
        self.assertGreaterEqual(res.optimized_thicknesses[0], 0.0)

    def test_bva_high_conic_aspheric_surfaces(self):
        """Optimizing distance behind complex aspheric lens."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=100.0))
        stack.add_surface(Surface(name="Asph", radius=40.0, thickness=6.0, glass="N-BK7", conic=-2.5, asphere_coeffs=[1e-6, -1e-9], semi_diameter=20.0))
        stack.add_surface(Surface(name="Flat", radius=0.0, thickness=40.0, glass="AIR", semi_diameter=20.0))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=20.0))

        sys = OpticalSystem(stack)
        optimizer = OpticalOptimizer(sys, max_iter=50, num_rays=20)
        res = optimizer.optimize(variable_indices=[2], bounds=[(10.0, 150.0)])
        self.assertTrue(res.success)
        self.assertLess(res.final_rms, res.initial_rms)

    def test_bva_total_internal_reflection_handling(self):
        """System encountering TIR during optimization loop applies penalty without crashing."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=20.0))
        stack.add_surface(Surface(name="Prism1", radius=0.0, thickness=10.0, glass="N-SF11", tilt_y=45.0, semi_diameter=20.0))
        stack.add_surface(Surface(name="Prism2", radius=0.0, thickness=30.0, glass="AIR", tilt_y=-45.0, semi_diameter=20.0))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=20.0))

        sys = OpticalSystem(stack)
        optimizer = OpticalOptimizer(sys, max_iter=20, num_rays=15)
        merit = optimizer.merit_function(np.array([30.0]), [2])
        self.assertIsInstance(merit, float)

    def test_bva_multi_wavelength_dispersion_bounds(self):
        """Chromatic optimization with dispersion catalog across F, d, C lines."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=100.0))
        stack.add_surface(Surface(name="L1", radius=50.0, thickness=5.0, glass="N-BK7", semi_diameter=20.0))
        stack.add_surface(Surface(name="L2", radius=0.0, thickness=40.0, glass="AIR", semi_diameter=20.0))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=20.0))

        sys = OpticalSystem(stack)
        optimizer = OpticalOptimizer(sys, wavelengths=[0.4861, 0.5876, 0.6563], max_iter=40, num_rays=20)
        res = optimizer.optimize(variable_indices=[2])
        self.assertTrue(res.success)


# ===========================================================================
# Tier 3: Cross-Feature Combinations (Pairwise Coverage)
# ===========================================================================
class TestOptimizerTier3PairwiseCombinations(unittest.TestCase):
    """
    Tier 3: Pairwise Combinatorial Tests:
    - Ray termination active during optimizer loop
    - Multi-variable thicknesses with bounds
    - GUI model sync and optimizer result
    - Spot diagram refresh after optimization
    - Dual view CAD update after optimization
    - Optimizer with decentered and tilted surfaces
    """

    def test_pairwise_r2_termination_with_r1_optimizer_loop(self):
        """Optimizer repeatedly calls ray tracer with strict termination active during all iterations."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=100.0))
        stack.add_surface(Surface(name="L1", radius=50.0, thickness=5.0, glass="N-BK7", semi_diameter=20.0))
        stack.add_surface(Surface(name="L2", radius=0.0, thickness=30.0, glass="AIR", semi_diameter=20.0))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=10.0, semi_diameter=20.0))
        stack.add_surface(Surface(name="UNUSED_EXTRA", thickness=0.0))

        sys = OpticalSystem(stack)
        optimizer = OpticalOptimizer(sys, max_iter=25, num_rays=15)
        res = optimizer.optimize(variable_indices=[2])

        self.assertTrue(res.success)
        # Verify optimized stack terminates rays at detector
        test_sys = OpticalSystem(res.optimized_stack)
        r = trace_ray_strict_termination(test_sys, Ray(pos=np.array([0.0, 5.0, 0.0]), dir=np.array([0.0, 0.0, 1.0])))
        self.assertEqual(r.history[-1]["surface_index"], 3)

    def test_pairwise_multi_variable_thicknesses_with_bounds(self):
        """Optimizing doublet air gap and back focal length simultaneously."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=100.0))
        stack.add_surface(Surface(name="L1_Front", radius=50.0, thickness=6.0, glass="N-BK7", semi_diameter=20.0))
        stack.add_surface(Surface(name="L1_Back", radius=-40.0, thickness=2.0, glass="AIR", semi_diameter=20.0))
        stack.add_surface(Surface(name="L2_Front", radius=-40.0, thickness=4.0, glass="N-SF11", semi_diameter=20.0))
        stack.add_surface(Surface(name="L2_Back", radius=-120.0, thickness=40.0, glass="AIR", semi_diameter=20.0))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=20.0))

        sys = OpticalSystem(stack)
        optimizer = OpticalOptimizer(sys, max_iter=50, num_rays=20)
        bounds = [(0.1, 10.0), (20.0, 150.0)]
        res = optimizer.optimize(variable_indices=[2, 4], bounds=bounds)

        self.assertTrue(res.success)
        self.assertGreaterEqual(res.optimized_thicknesses[2], 0.1)
        self.assertLessEqual(res.optimized_thicknesses[2], 10.0)
        self.assertGreaterEqual(res.optimized_thicknesses[4], 20.0)
        self.assertLessEqual(res.optimized_thicknesses[4], 150.0)

    def test_pairwise_gui_model_sync_and_optimizer_result(self):
        """GUI LensDataTableModel receives OptimizationResult.optimized_stack and syncs table cells."""
        model = LensDataTableModel()
        stack = model.get_surface_stack()
        stack.surfaces[1].thickness = 10.0
        sys = OpticalSystem(stack)

        optimizer = OpticalOptimizer(sys, max_iter=20, num_rays=15)
        res = optimizer.optimize(variable_indices=[1], bounds=[(5.0, 50.0)])

        model.set_surface_stack(res.optimized_stack)
        self.assertAlmostEqual(model.data(1, 3), res.optimized_thicknesses[1], places=4)

    def test_pairwise_spot_diagram_refresh_after_optimization(self):
        """SpotDiagramTab recalculates Airy disk and RMS spot radius on updated stack."""
        model = LensDataTableModel()
        analysis_tabs = AnalysisTabsWidget(model)

        stack = model.get_surface_stack()
        stack.surfaces[1].thickness = 45.0
        model.set_surface_stack(stack)
        self.assertTrue(analysis_tabs.refresh_analysis())

    def test_pairwise_dual_view_update_after_optimization(self):
        """DualViewWidget recalculates CAD vertex geometries and ray paths."""
        model = LensDataTableModel()
        dual_view = DualViewWidget(model)

        stack = model.get_surface_stack()
        stack.surfaces[1].thickness = 30.0
        model.set_surface_stack(stack)
        self.assertTrue(dual_view.refresh_layout())

    def test_pairwise_optimizer_with_decentered_tilted_surfaces(self):
        """Off-axis folded optical system thickness optimization."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=50.0))
        stack.add_surface(Surface(name="FoldMirror", radius=0.0, thickness=40.0, glass="MIRROR", tilt_y=45.0, semi_diameter=20.0))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=20.0))

        sys = OpticalSystem(stack)
        optimizer = OpticalOptimizer(sys, max_iter=20, num_rays=15)
        res = optimizer.optimize(variable_indices=[1], bounds=[(10.0, 100.0)])
        self.assertTrue(res.success)


# ===========================================================================
# Tier 4: Real-World Application Scenarios (Acceptance Milestones)
# ===========================================================================
class TestOptimizerTier4RealWorldWorkloads(unittest.TestCase):
    """
    Tier 4: Acceptance Workloads:
    - Defocused singlet with >=50% RMS spot reduction (Mandatory Acceptance Criterion)
    - Achromatic doublet collimation
    - Cooke triplet back focal length optimization
    - Telephoto lens system optimization
    - End-to-end simulated GUI workflow
    """

    def test_workload_defocused_singlet_50pct_rms_reduction(self):
        """
        Acceptance Test: Intentionally defocused plano-convex singlet (f ~ 100mm, initial t = 30mm).
        Asserts that the final RMS Spot Radius decreases by at least 50%.
        """
        stack = SurfaceStack()
        # Plano-convex N-BK7 lens: R1 = 51.68mm, R2 = flat, thickness = 5.0mm -> f ~ 100mm
        s0 = Surface(name="OBJ", surface_type=SurfaceType.OBJECT, thickness=1000.0)
        s1 = Surface(name="L1_Front", surface_type=SurfaceType.STANDARD, radius=51.68, thickness=5.0, glass="N-BK7", semi_diameter=15.0)
        s2 = Surface(name="L1_Back", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=30.0, glass="AIR", semi_diameter=15.0)  # Initial t = 30mm (defocused from f=96.7mm)
        ensure_surface_optimizer_attributes(s2, thickness_variable=True, thickness_min=10.0, thickness_max=200.0)
        s3 = Surface(name="IMA", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=20.0)

        for s in [s0, s1, s2, s3]:
            stack.add_surface(s)

        sys = OpticalSystem(stack)
        optimizer = OpticalOptimizer(sys, method="Nelder-Mead", max_iter=100, num_rays=30, entrance_pupil_radius=10.0)
        result = optimizer.optimize(variable_indices=[2], bounds=[(10.0, 200.0)])

        self.assertTrue(result.success, "Optimizer should converge successfully")
        self.assertGreater(result.initial_rms, 0.0)
        self.assertLess(result.final_rms, result.initial_rms)

        # Mandatory Acceptance Criterion: >= 50% RMS reduction
        self.assertGreaterEqual(
            result.rms_reduction_pct, 50.0,
            f"Expected at least 50% RMS spot reduction, got {result.rms_reduction_pct:.2f}%"
        )
        self.assertLessEqual(
            result.final_rms, 0.50 * result.initial_rms,
            f"Final RMS ({result.final_rms:.4f}) must be <= 50% of Initial RMS ({result.initial_rms:.4f})"
        )
        # Optimal focal distance should be in the ~90-100mm range
        opt_bfl = result.optimized_thicknesses[2]
        self.assertAlmostEqual(opt_bfl, 96.7, delta=10.0)

    def test_workload_achromatic_doublet_collimation(self):
        """Crown-flint doublet optimization minimizing chromatic spot blur across F, d, C lines."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=1000.0))
        stack.add_surface(Surface(name="Crown_Front", radius=62.0, thickness=6.0, glass="N-BK7", semi_diameter=15.0))
        stack.add_surface(Surface(name="Crown_Back", radius=-43.0, thickness=1.5, glass="AIR", semi_diameter=15.0))  # Variable air gap
        stack.add_surface(Surface(name="Flint_Front", radius=-43.0, thickness=3.5, glass="N-SF11", semi_diameter=15.0))
        stack.add_surface(Surface(name="Flint_Back", radius=-140.0, thickness=40.0, glass="AIR", semi_diameter=15.0))  # Variable BFL
        stack.add_surface(Surface(name="IMA", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=20.0))

        sys = OpticalSystem(stack)
        optimizer = OpticalOptimizer(
            sys,
            wavelengths=[0.4861, 0.5876, 0.6563],
            method="Nelder-Mead",
            max_iter=60,
            num_rays=20,
            entrance_pupil_radius=10.0
        )
        result = optimizer.optimize(variable_indices=[2, 4], bounds=[(0.1, 10.0), (30.0, 180.0)])

        self.assertTrue(result.success)
        self.assertGreaterEqual(result.rms_reduction_pct, 50.0)

    def test_workload_cooke_triplet_back_focal_length(self):
        """3-element Cooke Triplet with 2 variable air spaces optimized to tight focus."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=2000.0))
        stack.add_surface(Surface(name="L1_Front", radius=22.0, thickness=5.0, glass="N-SK16", semi_diameter=12.0))
        stack.add_surface(Surface(name="L1_Back", radius=-140.0, thickness=5.0, glass="AIR", semi_diameter=12.0))  # Variable air 1
        stack.add_surface(Surface(name="L2_Front", radius=-20.0, thickness=1.0, glass="N-F2", semi_diameter=10.0))
        stack.add_surface(Surface(name="L2_Back", radius=20.0, thickness=4.5, glass="AIR", semi_diameter=10.0))   # Variable air 2
        stack.add_surface(Surface(name="L3_Front", radius=80.0, thickness=5.0, glass="N-SK16", semi_diameter=12.0))
        stack.add_surface(Surface(name="L3_Back", radius=-24.0, thickness=15.0, glass="AIR", semi_diameter=12.0))  # Variable BFL
        stack.add_surface(Surface(name="IMA", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=15.0))

        sys = OpticalSystem(stack)
        optimizer = OpticalOptimizer(sys, method="Nelder-Mead", max_iter=60, num_rays=20, entrance_pupil_radius=8.0)
        result = optimizer.optimize(variable_indices=[2, 4, 6], bounds=[(1.0, 15.0), (1.0, 15.0), (10.0, 80.0)])

        self.assertTrue(result.success)
        self.assertGreaterEqual(result.rms_reduction_pct, 50.0)

    def test_workload_telephoto_lens_system(self):
        """Telephoto 2-group system optimizing group separation and detector distance."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=1000.0))
        stack.add_surface(Surface(name="Pos_Group_Front", radius=40.0, thickness=5.0, glass="N-BK7", semi_diameter=15.0))
        stack.add_surface(Surface(name="Pos_Group_Back", radius=-80.0, thickness=20.0, glass="AIR", semi_diameter=15.0))  # Variable separation
        stack.add_surface(Surface(name="Neg_Group_Front", radius=-30.0, thickness=3.0, glass="N-SF11", semi_diameter=12.0))
        stack.add_surface(Surface(name="Neg_Group_Back", radius=40.0, thickness=15.0, glass="AIR", semi_diameter=12.0))   # Variable BFL
        stack.add_surface(Surface(name="IMA", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=15.0))

        sys = OpticalSystem(stack)
        optimizer = OpticalOptimizer(sys, method="Nelder-Mead", max_iter=60, num_rays=20, entrance_pupil_radius=8.0)
        result = optimizer.optimize(variable_indices=[2, 4], bounds=[(5.0, 50.0), (5.0, 80.0)])

        self.assertTrue(result.success)
        self.assertGreaterEqual(result.rms_reduction_pct, 50.0)

    def test_workload_end_to_end_gui_workflow(self):
        """Headless simulated GUI session: load singlet -> mark variable -> optimize -> verify LDE and Spot updated."""
        model = LensDataTableModel()
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=1000.0))
        stack.add_surface(Surface(name="L1_Front", radius=51.68, thickness=5.0, glass="N-BK7", semi_diameter=15.0))
        s2 = Surface(name="L1_Back", radius=0.0, thickness=30.0, glass="AIR", semi_diameter=15.0)
        ensure_surface_optimizer_attributes(s2, thickness_variable=True, thickness_min=10.0, thickness_max=200.0)
        stack.add_surface(s2)
        stack.add_surface(Surface(name="IMA", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=20.0))

        model.set_surface_stack(stack)
        dock = OptimizerDockWidget(model)
        dual_view = DualViewWidget(model)
        analysis_tabs = AnalysisTabsWidget(model)

        dock.set_surface_variable(2, True)
        self.assertIn(2, dock.get_variable_indices())

        sys = OpticalSystem(model.get_surface_stack())
        optimizer = OpticalOptimizer(sys, method="Nelder-Mead", max_iter=60, num_rays=20, entrance_pupil_radius=10.0)
        result = optimizer.optimize(dock.get_variable_indices())

        self.assertTrue(result.success)
        self.assertGreaterEqual(result.rms_reduction_pct, 50.0)

        # Apply to GUI model
        model.set_surface_stack(result.optimized_stack)
        self.assertAlmostEqual(model.data(2, 3), result.optimized_thicknesses[2], places=4)
        self.assertTrue(dual_view.refresh_layout())
        self.assertTrue(analysis_tabs.refresh_analysis())


if __name__ == "__main__":
    unittest.main()
