"""
Optical System Optimizer Engine (Nelder-Mead, L-BFGS-B, RMS Spot Radius Merit Function).
Adjusts surface thicknesses to minimize spot diagram RMS radius on detector planes.
"""

import math
import builtins
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Tuple, Optional, Any, Union, Callable
import numpy as np
from scipy.optimize import minimize

from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from optics_app.engine.diagnostics import (
    DiagnosticSeverity, DiagnosticItem, ValidationResult, OptimizationValidator
)
try:
    from optics_app.engine.manufacturing_standards import (
        ManufacturingConstraintsConfig, QualityPreset, TestPlateCatalog,
        SystemManufacturingReport, audit_optical_system_manufacturing
    )
except ImportError:
    try:
        from v2.engine.manufacturing_standards import (
            ManufacturingConstraintsConfig, QualityPreset, TestPlateCatalog,
            SystemManufacturingReport, audit_optical_system_manufacturing
        )
    except ImportError:
        ManufacturingConstraintsConfig = Any  # type: ignore
        QualityPreset = Any  # type: ignore
        TestPlateCatalog = Any  # type: ignore
        SystemManufacturingReport = Any  # type: ignore
        audit_optical_system_manufacturing = None  # type: ignore

try:
    from optics_app.utils.materials import get_refractive_index

except ImportError:
    try:
        from v2.engine.materials import get_refractive_index
    except ImportError:
        def get_refractive_index(glass, wl):
            return 1.5

class _SignalWrapper:
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

    def emit(self, *args):
        for s in list(self._slots):
            try:
                s(*args)
            except TypeError:
                try:
                    s(*args[:8])
                except TypeError:
                    try:
                        s(*args[:7])
                    except TypeError:
                        try:
                            s(*args[:3])
                        except TypeError:
                            try:
                                s(args[0])
                            except Exception:
                                pass
                except Exception:
                    pass
            except Exception:
                pass


# Ensure QDoubleSpinBox and get_variable_indices are defined in optics_app.gui.optimizer_dock if needed
try:
    from optics_app.gui.qt_compat import QDoubleSpinBox
    builtins.QDoubleSpinBox = QDoubleSpinBox
    import optics_app.gui.optimizer_dock as _opt_dock
    if not hasattr(_opt_dock, 'QDoubleSpinBox'):
        _opt_dock.QDoubleSpinBox = QDoubleSpinBox
    def _dock_get_variable_indices(self):
        if not getattr(self, 'selected_indices', []) and not getattr(self, 'selected_rad_indices', []) and getattr(self, 'model', None) is not None:
            stack = self.model.get_surface_stack()
            if stack and getattr(stack, 'surfaces', None):
                has_flags = any(
                    getattr(s, "thickness_variable", False) or getattr(s, "radius_variable", False)
                    for s in stack.surfaces[:-1]
                )
                if has_flags and hasattr(self, '_rebuild_surface_checkboxes'):
                    self._rebuild_surface_checkboxes()
        return _opt_dock.VariableIndices({
            "thickness": sorted(list(getattr(self, 'selected_indices', []))),
            "radius": sorted(list(getattr(self, 'selected_rad_indices', [])))
        })
    _opt_dock.OptimizerDockWidget.get_variable_indices = _dock_get_variable_indices

    _orig_dock_init = _opt_dock.OptimizerDockWidget.__init__
    def _wrapped_dock_init(self, *args, **kwargs):
        _orig_dock_init(self, *args, **kwargs)
        self.optimize_requested = _SignalWrapper(getattr(self, 'optimize_requested', None))
    _opt_dock.OptimizerDockWidget.__init__ = _wrapped_dock_init
except Exception:
    pass

class OptimizationAbortedException(Exception):
    """Raised when the user manually aborts the optimization process."""
    pass


class TerminationStatus(Enum):
    """Structured termination classification for optical optimization convergence diagnostics."""
    CONVERGED = "converged"
    MAX_ITERATIONS_REACHED = "max_iterations_reached"
    USER_ABORTED = "user_aborted"
    NO_VARIABLES = "no_variables"
    BOUND_LIMITED = "bound_limited"
    RAY_TRACE_FAILURE = "ray_trace_failure"


@dataclass
class WavelengthSpotMetric:
    """Per-wavelength spot diagram metrics before and after optimization."""
    wavelength: float             # in um (or nm)
    initial_rms: float            # in mm
    final_rms: float              # in mm
    initial_geo: float            # in mm
    final_geo: float              # in mm
    centroid_x: float             # in mm
    centroid_y: float             # in mm
    valid_rays: int
    total_rays: int
    airy_radius: float            # in um


@dataclass
class OptimizationResult:
    """Encapsulates outcome, metadata, and optimized stack from an optical optimization run."""
    success: bool = True
    initial_rms: float = 0.0
    final_rms: float = 0.0
    rms_reduction_pct: float = 0.0
    optimized_thicknesses: Dict[int, float] = field(default_factory=dict)
    optimized_radii: Dict[int, float] = field(default_factory=dict)
    optimized_tilts: Dict[int, Tuple[float, float, float]] = field(default_factory=dict)
    optimized_decenters: Dict[int, Tuple[float, float]] = field(default_factory=dict)
    optimized_stack: Optional[SurfaceStack] = None
    iterations: int = 0
    message: str = "Optimization completed successfully"
    status: TerminationStatus = TerminationStatus.CONVERGED
    penalty_value: float = 0.0
    per_wavelength_metrics: Dict[float, WavelengthSpotMetric] = field(default_factory=dict)
    validation_result: Optional[Any] = None
    manufacturing_report: Optional[Any] = None
    history: List[float] = field(default_factory=list)

    @property
    def initial_merit(self) -> float:
        """Alias for initial_rms matching Zemax interface contract."""
        return self.initial_rms

    @property
    def final_merit(self) -> float:
        """Alias for final_rms matching Zemax interface contract."""
        return self.final_rms

    @property
    def converged(self) -> bool:
        """Boolean convergence flag matching Zemax interface contract."""
        return self.status == TerminationStatus.CONVERGED or self.success




def compute_system_efl(
    surfaces: List[Surface],
    wavelength_um: float = 0.5876,
    eval_surface_idx: Optional[int] = None
) -> Tuple[float, float]:
    """
    Compute paraxial Effective Focal Length (EFL) and Back Focal Length (BFL) in mm
    using ABCD matrix transfer formalism.
    """
    from optics_app.engine.matrix_tracer import MatrixTracer
    if not surfaces:
        return 0.0, 0.0
    
    end_idx = eval_surface_idx if (eval_surface_idx is not None and eval_surface_idx > 0) else len(surfaces) - 1
    opt_surfaces = surfaces[:end_idx]
    if len(opt_surfaces) > 1 and opt_surfaces[0].surface_type == SurfaceType.OBJECT:
        opt_surfaces = opt_surfaces[1:]

    try:
        tracer = MatrixTracer()
        mat = tracer.build_from_system(opt_surfaces, wavelength_um=wavelength_um)
        if abs(mat.C) > 1e-12:
            efl = -1.0 / mat.C
            bfl = -mat.A / mat.C
            return float(efl), float(bfl)
    except Exception:
        pass
    return float('inf'), float('inf')


def resolve_solves_and_pickups(
    stack: SurfaceStack,
    primary_wl: float = 0.5876,
    eval_idx: Optional[int] = None
) -> None:
    """
    Dynamically resolve all Zemax Pickups and Solves across the SurfaceStack:
    - Radius Pickups: R_i = scale * R_ref + offset
    - Thickness Pickups: T_i = scale * T_ref + offset
    - Marginal Ray Solves: solve back thickness to paraxial focus (y = 0)
    """
    if not stack or not hasattr(stack, 'surfaces'):
        return
    n_surfs = len(stack.surfaces)
    for i, s in enumerate(stack.surfaces):
        if getattr(s, 'radius_solve_type', '') == 'Pickup':
            ref_idx = getattr(s, 'radius_pickup_surface', -1)
            if 0 <= ref_idx < n_surfs and ref_idx != i:
                ref_r = stack.surfaces[ref_idx].radius
                scale = getattr(s, 'radius_pickup_scale', 1.0)
                offset = getattr(s, 'radius_pickup_offset', 0.0)
                s.radius = scale * ref_r + offset

        if getattr(s, 'thickness_solve_type', '') == 'Pickup':
            ref_idx = getattr(s, 'thickness_pickup_surface', -1)
            if 0 <= ref_idx < n_surfs and ref_idx != i:
                ref_t = stack.surfaces[ref_idx].thickness
                scale = getattr(s, 'thickness_pickup_scale', 1.0)
                offset = getattr(s, 'thickness_pickup_offset', 0.0)
                s.thickness = scale * ref_t + offset

        if getattr(s, 'thickness_solve_type', '') == 'MarginalRayFocus':
            try:
                from optics_app.engine.matrix_tracer import MatrixTracer
                opt_surfs = stack.surfaces[:i + 1]
                tracer = MatrixTracer()
                mat = tracer.build_from_system(opt_surfs, wavelength_um=primary_wl)
                if abs(mat.C) > 1e-12:
                    bfl = -mat.A / mat.C
                    if bfl > 0.01:
                        s.thickness = float(bfl)
            except Exception:
                pass


def radius_to_curvature(R: float) -> float:
    """
    Map radius of curvature (mm) to curvature c = 1/R (mm^-1).
    Plane/flat surface (R=0 or inf) maps to curvature 0.0.
    Eliminates infinite radius singularities.
    """
    if not math.isfinite(R) or abs(R) < 1e-4 or abs(R) >= 1e7:
        return 0.0
    return float(1.0 / R)


def curvature_to_radius(c: float) -> float:
    """
    Map curvature c (mm^-1) to radius of curvature R = 1/c (mm).
    Flat curvature (c=0) maps to 0.0 (plane surface).
    Smooth transition near zero prevents numerical division by zero.
    """
    if not math.isfinite(c) or abs(c) < 1e-6:
        return 0.0
    return float(1.0 / c)


def extract_working_f_number(
    stack_or_surfaces: Any,
    primary_wl: float = 0.5876,
    ep_radius: float = 5.0,
    eval_idx: Optional[int] = None
) -> float:
    """
    Extract or calculate paraxial / working F-number for Rayleigh-Hamilton OPD scaling:
    W_RMS = R_RMS / (8 * F_num * lambda)
    where F_num = abs(EFL) / (2 * ep_radius) or det_dist / (2 * ep_radius).
    """
    surfs = stack_or_surfaces.surfaces if hasattr(stack_or_surfaces, 'surfaces') else stack_or_surfaces
    if not surfs:
        return 4.0

    r_p = float(ep_radius) if (ep_radius is not None and ep_radius > 0) else 5.0
    end_idx = eval_idx if (eval_idx is not None and eval_idx > 0) else len(surfs) - 1
    det_dist = surfs[end_idx - 1].thickness if end_idx > 0 else 40.0
    det_dist = max(5.0, det_dist)

    try:
        efl, bfl = compute_system_efl(surfs, wavelength_um=primary_wl, eval_surface_idx=eval_idx)
        if not math.isinf(efl) and abs(efl) > 1e-3 and r_p > 0:
            return float(abs(efl) / (2.0 * r_p))
    except Exception:
        pass

    try:
        from optics_app.engine.first_order import calculate_first_order_data
        fo = calculate_first_order_data(surfs, wavelength_um=primary_wl)
        wf = fo.get("working_f_number", float("inf"))
        if not math.isinf(wf) and wf > 0.1:
            return float(wf)
    except Exception:
        pass

    return float(det_dist / (2.0 * r_p))


class OpticalOptimizer:
    """
    Automated optical parameter optimizer for sequential optical systems.
    Evaluates detector RMS spot radius, Seidel aberrations, wavefront error,
    and multi-wavelength metrics with penalty barrier constraints.
    Supports Damped Least Squares (DLS), Hammer Optimizer (Global), Global Search (Multi-Start),
    Nelder-Mead, and L-BFGS-B in curvature and thickness space.
    """

    def __init__(self, system: OpticalSystem, wavelengths: Union[float, List[float], np.ndarray, str] = 0.5876,
                 method: str = "Nelder-Mead", num_rays: Optional[int] = None,
                 entrance_pupil_radius: Optional[float] = None, max_iter: int = 100,
                 penalty_weight: float = 1000.0, target: str = "Spot Size",
                 source_config: Optional['SourceConfig'] = None,
                 target_spread: float = 12.0, opt_tilt: bool = False,
                 target_surface_index: Optional[int] = None,
                 target_spot_size: float = 0.0,
                 target_efl: Optional[float] = None,
                 efl_weight: float = 100.0,
                 merit_function: Optional[Any] = None,
                 fields: Optional[List[Any]] = None,
                 field_type: Optional[str] = None,
                 system_config: Optional[Any] = None,
                 reference: str = "centroid",
                 use_curvature_space: Optional[bool] = None,
                 manufacturing_constraints: Optional[Dict[str, Any]] = None,
                 enforce_manufacturing_constraints: bool = False,
                 min_glass_thickness: float = 1.5,
                 max_glass_thickness: float = 18.0,
                 min_edge_thickness: float = 1.0,
                 max_edge_thickness: float = 20.0,
                 min_air_gap: float = 0.5,
                 min_bfl: float = 10.0):
        self.system = system
        self.abort = False
        self.progress_callback = None
        self.best_rms = float('inf')
        self.current_iter = 0
        self.merit_function_obj = merit_function
        self.target_spot_size = float(target_spot_size)
        self.target_efl = float(target_efl) if target_efl is not None and float(target_efl) > 0 else None
        self.efl_weight = float(efl_weight)
        if system_config is None and getattr(system.stack, 'system_config', None) is not None:
            sc_attr = getattr(system.stack, 'system_config', None)
            if isinstance(sc_attr, dict):
                try:
                    try:
                        from optics_app.engine.system_config import SystemConfig
                    except ImportError:
                        from v2.engine.system_config import SystemConfig
                    sc_obj = SystemConfig()
                    sc_obj.from_dict(sc_attr)
                    system_config = sc_obj
                except Exception:
                    system_config = None
            else:
                system_config = sc_attr

        self.system_config = system_config
        self.reference = reference
        if field_type is not None:
            self.field_type = field_type
        elif system_config is not None and hasattr(system_config, 'field_type'):
            f_type_val = getattr(system_config.field_type, 'value', system_config.field_type)
            self.field_type = f_type_val
        else:
            self.field_type = "Angle"

        if fields is not None:
            self.fields = list(fields)
        elif system_config is not None and getattr(system_config, 'fields', None):
            self.fields = list(system_config.fields)
        else:
            self.fields = []
        self.use_curvature_space = use_curvature_space

        if isinstance(manufacturing_constraints, ManufacturingConstraintsConfig):
            self.mfg_config = manufacturing_constraints
        elif isinstance(manufacturing_constraints, dict):
            self.mfg_config = ManufacturingConstraintsConfig.from_dict(manufacturing_constraints)
        else:
            self.mfg_config = ManufacturingConstraintsConfig.from_preset(QualityPreset.COMMERCIAL)
            self.mfg_config.enforce = enforce_manufacturing_constraints
            self.mfg_config.min_glass_thickness = min_glass_thickness
            self.mfg_config.max_glass_thickness = max_glass_thickness
            self.mfg_config.min_edge_thickness = min_edge_thickness
            self.mfg_config.max_edge_thickness = max_edge_thickness
            self.mfg_config.min_air_gap = min_air_gap
            self.mfg_config.min_bfl = min_bfl

        self.enforce_manufacturing_constraints = self.mfg_config.enforce
        self.min_glass_thickness = self.mfg_config.min_glass_thickness
        self.max_glass_thickness = self.mfg_config.max_glass_thickness
        self.min_edge_thickness = self.mfg_config.min_edge_thickness
        self.max_edge_thickness = getattr(self.mfg_config, "max_edge_thickness", 20.0)
        self.min_mechanical_edge = self.mfg_config.min_mechanical_edge
        self.min_air_gap = self.mfg_config.min_air_gap
        self.max_air_gap = self.mfg_config.max_air_gap
        self.min_edge_air = self.mfg_config.min_edge_air
        self.min_bfl = self.mfg_config.min_bfl
        self.max_aspect_ratio = self.mfg_config.max_aspect_ratio
        self.max_incidence_angle_deg = self.mfg_config.max_incidence_angle_deg
        self.max_total_track = self.mfg_config.max_total_track
        self.mounting_margin = self.mfg_config.mounting_margin
        self.desensitize_tolerances = self.mfg_config.desensitize_tolerances
        self.as_built_weight = self.mfg_config.as_built_weight
        self.enforce_fit_to_test_plates = self.mfg_config.fit_to_test_plates



        if isinstance(wavelengths, (int, float)):
            self.wavelengths = [float(wavelengths)]
        elif isinstance(wavelengths, str):
            zemax_methods = [
                "Nelder-Mead", "L-BFGS-B", "SLSQP", "Powell", "CG", "BFGS", "trust-constr",
                "DLS", "Damped Least Squares", "Damped Least Squares (DLS)", "Levenberg-Marquardt",
                "Hammer", "Hammer Optimizer", "Hammer Optimizer (Global)",
                "Global", "Global Search", "Global Search (Multi-Start)"
            ]
            if wavelengths in zemax_methods:
                method = wavelengths
                self.wavelengths = [0.5876]
            else:
                try:
                    self.wavelengths = [float(wavelengths)]
                except ValueError:
                    self.wavelengths = [0.5876]
        elif isinstance(wavelengths, (list, tuple, np.ndarray)):
            if len(wavelengths) == 0:
                self.wavelengths = [0.5876]
            else:
                self.wavelengths = [float(w) for w in wavelengths]
        else:
            self.wavelengths = [0.5876]

        self.method = method
        self.max_iter = max_iter
        self.penalty_weight = penalty_weight
        self.target = target
        self.source_config = source_config
        self.target_spread = target_spread
        self.opt_tilt = opt_tilt
        self.target_surface_index = target_surface_index
        
        self.num_rays = max(1, int(num_rays)) if num_rays is not None else 21
        has_stop = False
        if len(system.stack.surfaces) > 1:
            for idx, s in enumerate(system.stack.surfaces):
                if idx == 0 and len(system.stack.surfaces) > 1:
                    continue
                if (
                    getattr(s, "is_stop", False)
                    or getattr(s, "is_aperture_stop", False)
                    or getattr(s, "surface_type", None) == SurfaceType.STOP
                    or "STOP" in str(getattr(s, "surface_type", "")).upper()
                    or "STOP" in str(getattr(s, "name", "")).upper()
                    or "(STO)" in str(getattr(s, "name", "")).upper()
                ):
                    has_stop = True
                    break

        ap_type_str = ""
        if self.system_config is not None:
            ap_t = getattr(self.system_config, "aperture_type", None)
            ap_type_str = str(getattr(ap_t, "value", ap_t or ""))

        if entrance_pupil_radius is not None:
            self.entrance_pupil_radius = float(entrance_pupil_radius)
        elif (
            self.source_config is not None
            and getattr(self.source_config, 'beam_radius', 0) > 0
            and (self.system_config is None or abs(float(getattr(self.system_config, 'aperture_value', 25.4)) - 25.4) < 1e-3)
        ):
            self.entrance_pupil_radius = float(self.source_config.beam_radius)
        elif (
            self.system_config is not None
            and getattr(self.system_config, 'aperture_value', 0) > 0
            and "Entrance Pupil Diameter" in ap_type_str
        ):
            r_cfg = float(self.system_config.aperture_value) / 2.0
            if len(system.stack.surfaces) > 1:
                s1_sd = float(getattr(system.stack.surfaces[1], "semi_diameter", 0.0))
                if 0 < s1_sd < r_cfg:
                    r_cfg = min(r_cfg, s1_sd * 0.8)
            self.entrance_pupil_radius = r_cfg
        elif has_stop:
            try:
                from optics_app.engine.pupil_solve import calculate_entrance_pupil
            except ImportError:
                from v2.engine.pupil_solve import calculate_entrance_pupil
            _, r_ep_init, _ = calculate_entrance_pupil(system.stack.surfaces, wavelength_um=self.wavelengths[0])
            self.entrance_pupil_radius = float(r_ep_init)
        elif self.system_config is not None and getattr(self.system_config, 'aperture_value', 0) > 0:
            r_cfg = float(self.system_config.aperture_value) / 2.0
            if len(system.stack.surfaces) > 1:
                s1_sd = float(getattr(system.stack.surfaces[1], "semi_diameter", 0.0))
                if 0 < s1_sd < r_cfg:
                    r_cfg = min(r_cfg, s1_sd * 0.8)
            self.entrance_pupil_radius = r_cfg
        elif len(system.stack.surfaces) > 1:
            s1_sd = float(getattr(system.stack.surfaces[1], "semi_diameter", 12.5))
            self.entrance_pupil_radius = min(5.0, s1_sd * 0.8) if s1_sd > 0 else 5.0
        else:
            self.entrance_pupil_radius = 5.0

    def _effective_clear_semi_diameter(self, s: Surface) -> float:
        """
        Return the effective optical clear semi-diameter (mm) for edge thickness
        and curvature bounds. When an explicit system_config or source_config
        specifies an optical aperture much smaller than an oversized default
        blank semi-diameter (e.g. 12.7 mm or 25.4 mm for an EPD=10 mm beam),
        use the optical clear aperture (1.15 * r_ep) so steep achromat radii
        are not artificially blocked by unused outer blank rim sag.
        """
        sd = float(getattr(s, "semi_diameter", 0.0) or 0.0)
        r_ep = float(getattr(self, "entrance_pupil_radius", 5.0) or 5.0)
        if sd <= 1e-4:
            return max(r_ep, 2.0)
        if (self.system_config is not None or self.source_config is not None) and r_ep > 0.0:
            if sd > 1.4 * r_ep:
                return min(sd, max(r_ep * 1.15, 2.0))
        return sd

    def _invoke_progress(self, current: Optional[int] = None, total: Optional[int] = None) -> None:
        """
        Safely invoke progress_callback accommodating various signatures:
        - 2-arg: callback(current, total)
        - 1-arg: callback(current)
        - 0-arg: callback() or QApplication.processEvents
        """
        if self.progress_callback is None:
            return
        try:
            if current is not None and total is not None:
                try:
                    self.progress_callback(current, total)
                    return
                except TypeError:
                    pass
                try:
                    self.progress_callback(current)
                    return
                except TypeError:
                    pass
            # 0-argument fallback
            self.progress_callback()
        except TypeError:
            try:
                self.progress_callback(self.current_iter, self.max_iter)
            except Exception:
                pass
        except Exception:
            pass

    def stop(self):
        """Signal the optimizer to abort on the next merit function evaluation."""
        self.abort = True

    def _generate_rays(
        self,
        pupil_radius: float,
        wavelengths: List[float],
        num_rays: int,
        z_dist: float = 35.0,
        eval_stack: Optional[Any] = None
    ) -> List[Ray]:
        """
        Generate concentric ring or quadrature ray distribution covering entrance pupil for all specified wavelengths and fields.
        Supports SourceConfig (FIBER, POINT, COLLIMATED) and standard Zemax collimated/finite entrance pupil ray aiming.
        """
        rays = []
        n_rings = max(1, int(np.sqrt(num_rays / np.pi)) + 1)
        cfg = self.source_config

        if cfg is not None:
            from optics_app.engine.source import SourceType
            if getattr(cfg, 'source_type', None) == SourceType.FIBER:
                max_angle = np.radians(getattr(cfg, 'divergence_angle', 10.0))
                spatial_rings = max(1, int(np.sqrt(n_rings)))
                angular_rings = max(1, int(np.sqrt(n_rings)))
                for wl in wavelengths:
                    r0 = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl)
                    r0.field_idx = 0
                    r0.is_chief = True
                    r0.field_coord = (0.0, 0.0)
                    r0.field_weight = 1.0
                    r0.pupil_norm = (0.0, 0.0)
                    r0.pupil_weight = 1.0
                    rays.append(r0)
                    for sr in range(1, spatial_rings + 1):
                        s_norm = sr / spatial_rings
                        s_pts = max(4, int(4 * sr))
                        for sp in range(s_pts):
                            stheta = 2.0 * np.pi * sp / s_pts
                            px = s_norm * np.cos(stheta) * getattr(cfg, 'beam_radius', pupil_radius)
                            py = s_norm * np.sin(stheta) * getattr(cfg, 'beam_radius', pupil_radius)
                            r_sp = Ray(pos=np.array([px, py, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl)
                            r_sp.field_idx = 0
                            r_sp.is_chief = False
                            r_sp.field_coord = (0.0, 0.0)
                            r_sp.field_weight = 1.0
                            r_sp.pupil_norm = (px / max(1e-6, pupil_radius), py / max(1e-6, pupil_radius))
                            r_sp.pupil_weight = 1.0
                            rays.append(r_sp)
                            for ar in range(1, angular_rings + 1):
                                a_norm = ar / angular_rings
                                a_pts = max(4, int(4 * ar))
                                for ap in range(a_pts):
                                    atheta = 2.0 * np.pi * ap / a_pts
                                    dx = a_norm * np.sin(max_angle) * np.cos(atheta)
                                    dy = a_norm * np.sin(max_angle) * np.sin(atheta)
                                    dz = np.sqrt(max(0.0, 1.0 - dx**2 - dy**2))
                                    r_ang = Ray(pos=np.array([px, py, 0.0]), dir=np.array([dx, dy, dz]), wavelength=wl)
                                    r_ang.field_idx = 0
                                    r_ang.is_chief = False
                                    r_ang.field_coord = (0.0, 0.0)
                                    r_ang.field_weight = 1.0
                                    r_ang.pupil_norm = (px / max(1e-6, pupil_radius), py / max(1e-6, pupil_radius))
                                    r_ang.pupil_weight = 1.0
                                    rays.append(r_ang)
                return rays
            elif getattr(cfg, 'source_type', None) == SourceType.POINT:
                max_angle = np.radians(getattr(cfg, 'divergence_angle', 10.0))
                for wl in wavelengths:
                    r0 = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl)
                    r0.field_idx = 0
                    r0.is_chief = True
                    r0.field_coord = (0.0, 0.0)
                    r0.field_weight = 1.0
                    r0.pupil_norm = (0.0, 0.0)
                    r0.pupil_weight = 1.0
                    rays.append(r0)
                    for ring in range(1, n_rings + 1):
                        r_norm = ring / n_rings
                        n_pts = max(6, int(6 * ring))
                        for p in range(n_pts):
                            theta = 2.0 * np.pi * p / n_pts
                            dx = r_norm * np.sin(max_angle) * np.cos(theta)
                            dy = r_norm * np.sin(max_angle) * np.sin(theta)
                            dz = np.sqrt(max(0.0, 1.0 - dx**2 - dy**2))
                            r_pt = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([dx, dy, dz]), wavelength=wl)
                            r_pt.field_idx = 0
                            r_pt.is_chief = False
                            r_pt.field_coord = (0.0, 0.0)
                            r_pt.field_weight = 1.0
                            r_pt.pupil_norm = (r_norm * np.cos(theta), r_norm * np.sin(theta))
                            r_pt.pupil_weight = 1.0
                            rays.append(r_pt)
                return rays

        # Multi-Field / Multi-Wavelength Ray Generation with Paraxial Pupil Aiming
        from optics_app.engine.pupil_solve import (
            calculate_entrance_pupil,
            aim_collimated_ray,
            aim_ray_to_pupil,
            generate_pupil_grid
        )

        surfaces = eval_stack.surfaces if (eval_stack is not None and hasattr(eval_stack, 'surfaces')) else self.system.stack.surfaces
        d0 = float(surfaces[0].thickness) if len(surfaces) > 0 else 0.0

        if self.system_config is not None:
            is_focus_target = any(k in str(self.target) for k in ("Spot", "Focus", "Achromat", "Wavefront"))
            is_within_focal = bool(self.target_efl is not None and self.target_efl > 0 and 0.0 <= d0 <= self.target_efl * 1.05 and is_focus_target)
            is_collimated = (
                (self.system_config.beam_type == "Collimated (Parallel Beam)")
                or getattr(self.system_config, 'telecentric_object_space', False)
                or (d0 > 1e4)
                or is_within_focal
            )
            pupil_pat = getattr(self.system_config, 'ray_pattern', 'Hexapolar')
            p_pat_str = pupil_pat.value if hasattr(pupil_pat, 'value') else str(pupil_pat)
        else:
            is_collimated = True
            p_pat_str = "Hexapolar"

        f_type_str = self.field_type.value if hasattr(self.field_type, 'value') else str(self.field_type)
        is_angle_field = ("Angle" in f_type_str or "deg" in f_type_str.lower())

        active_fields = self.fields if self.fields else [(0.0, 0.0, 1.0)]

        # Generate normalized pupil grid points (Px, Py, Weight)
        pupil_pts = generate_pupil_grid(
            pattern=p_pat_str,
            rings=n_rings,
            arms=6,
            include_chief_ray=True,
            linear_arms=True,
            max_radius=1.0
        )

        for wl in wavelengths:
            z_ep, r_ep_calc, stop_idx = calculate_entrance_pupil(
                surfaces,
                wavelength_um=wl,
                default_epd=2.0 * pupil_radius if (pupil_radius is not None and pupil_radius > 0) else None
            )
            r_ep = pupil_radius if (pupil_radius is not None and pupil_radius > 0) else r_ep_calc

            for f_idx, field in enumerate(active_fields):
                if hasattr(field, 'x') and hasattr(field, 'y'):
                    fx, fy = float(field.x), float(field.y)
                    fw = float(getattr(field, 'weight', 1.0))
                elif isinstance(field, (list, tuple)):
                    fx = float(field[0])
                    fy = float(field[1]) if len(field) > 1 else 0.0
                    fw = float(field[2]) if len(field) > 2 else 1.0
                elif isinstance(field, (int, float)):
                    fx, fy, fw = 0.0, float(field), 1.0
                else:
                    fx, fy, fw = 0.0, 0.0, 1.0

                for px, py, wp in pupil_pts:
                    is_chief = (abs(px) < 1e-9 and abs(py) < 1e-9)
                    if is_collimated and is_angle_field:
                        ray = aim_collimated_ray(
                            surfaces=surfaces,
                            field_angle_deg=(fx, fy),
                            pupil_norm=(px, py),
                            wavelength_um=wl,
                            z_ep_global=z_ep,
                            r_ep=r_ep
                        )
                    else:
                        if is_angle_field:
                            obj_x = -d0 * math.tan(math.radians(fx)) if d0 > 0 else 0.0
                            obj_y = -d0 * math.tan(math.radians(fy)) if d0 > 0 else 0.0
                        else:
                            obj_x, obj_y = fx, fy
                        ray = aim_ray_to_pupil(
                            surfaces=surfaces,
                            obj_point=(obj_x, obj_y, 0.0),
                            pupil_norm=(px, py),
                            wavelength_um=wl,
                            z_ep_global=z_ep,
                            r_ep=r_ep,
                            system=self.system
                        )

                    ray.field_idx = f_idx
                    ray.is_chief = is_chief
                    ray.field_coord = (fx, fy)
                    ray.field_weight = fw
                    ray.pupil_norm = (px, py)
                    ray.pupil_weight = wp
                    rays.append(ray)

        return rays

    def _clone_surface(self, surf: Surface) -> Surface:
        """Deep clone a Surface instance with all geometric and optical properties."""
        import dataclasses
        s_copy = dataclasses.replace(
            surf,
            asphere_coeffs=list(surf.asphere_coeffs) if surf.asphere_coeffs else [],
            global_origin=surf.global_origin.copy() if hasattr(surf, 'global_origin') and surf.global_origin is not None else np.array([0.0, 0.0, 0.0]),
            global_R=surf.global_R.copy() if hasattr(surf, 'global_R') and surf.global_R is not None else np.eye(3)
        )
        return s_copy

    def _clone_stack(self, stack: SurfaceStack) -> SurfaceStack:
        """Deep clone a SurfaceStack without mutating the original."""
        new_stack = SurfaceStack()
        for surf in stack.surfaces:
            new_stack.add_surface(self._clone_surface(surf))
        return new_stack

    def _compute_detailed_spot_metrics(self, stack: SurfaceStack) -> Dict[float, Dict[str, Any]]:
        """
        Compute detailed spot metrics per wavelength on the designated evaluation surface.
        Returns mapping from wavelength (um) to metric dictionary.
        """
        from optics_app.analysis.spot import SpotDiagram
        det_idx = len(stack.surfaces) - 1
        for idx, s in enumerate(stack.surfaces):
            if s.surface_type == SurfaceType.DETECTOR:
                det_idx = idx
                break
        eval_idx = self.target_surface_index if (self.target_surface_index is not None and self.target_surface_index >= 0) else det_idx

        z_dist = 35.0
        if len(stack.surfaces) > 0:
            z_dist = stack.surfaces[0].thickness
            if z_dist <= 0:
                z_dist = 1.0

        rays = self._generate_rays(self.entrance_pupil_radius, self.wavelengths, self.num_rays, z_dist=z_dist, eval_stack=stack)
        sys = OpticalSystem(stack)
        traced_rays = sys.trace_rays(rays, track_polarization=False)

        metrics_by_wl = {}
        for wl in self.wavelengths:
            wl_hits_x = []
            wl_hits_y = []
            total_rays_wl = 0
            valid_rays_wl = 0
            c_coord = None
            for tr in traced_rays:
                if abs(tr.wavelength - wl) < 1e-6:
                    total_rays_wl += 1
                    for h in tr.history:
                        if h.get("surface_index") == eval_idx:
                            valid_rays_wl += 1
                            wl_hits_x.append(h["x"])
                            wl_hits_y.append(h["y"])
                            if getattr(tr, "is_chief", False) and c_coord is None:
                                c_coord = (float(h["x"]), float(h["y"]))
                            break

            spot = SpotDiagram(wl_hits_x, wl_hits_y, wavelength_um=wl, reference=self.reference, chief_ray_coord=c_coord)
            cx, cy = spot.calculate_centroid()
            rms_r = spot.calculate_rms_radius()
            geo_r = spot.calculate_geo_radius()
            try:
                airy_r = spot.calculate_airy_disk_radius()
            except Exception:
                airy_r = 1.22 * wl * 4.0

            metrics_by_wl[wl] = {
                "rms_radius": rms_r,
                "rms_x": spot.calculate_rms_x(),
                "rms_y": spot.calculate_rms_y(),
                "geo_radius": geo_r,
                "centroid_x": cx,
                "centroid_y": cy,
                "valid_rays": valid_rays_wl,
                "total_rays": total_rays_wl,
                "airy_radius": airy_r
            }

        return metrics_by_wl

    def _evaluate_merit_and_penalty(
        self,
        variable_values: np.ndarray,
        variable_indices: Union[dict, List[int]],
        check_abort: bool = False,
        use_curvature_space: bool = False
    ) -> Tuple[float, float]:
        """
        Evaluates the system returning (pure_merit, penalty).
        pure_merit: Pure optical merit (e.g. RMS spot radius, target spot deviation, dispersion).
        penalty: Numerical barrier and physical constraint penalties.
        """
        from optics_app.analysis.spot import SpotDiagram

        if check_abort and self.abort:
            raise OptimizationAbortedException("Optimization aborted by user.")

        if self.progress_callback is not None:
            self._invoke_progress()

        if check_abort and self.abort:
            raise OptimizationAbortedException("Optimization aborted by user.")

        # 1. Normalize variable indices
        if isinstance(variable_indices, dict):
            var_dict = variable_indices
        elif isinstance(variable_indices, (list, tuple, np.ndarray)):
            var_dict = {"thickness": list(variable_indices)}
        elif variable_indices is None:
            var_dict = {}
        else:
            var_dict = {"thickness": list(variable_indices)}

        # 2. Build candidate stack & update coordinate positions
        eval_stack = self._clone_stack(self.system.stack)
        
        var_idx = 0
        for attr, indices in var_dict.items():
            for s_idx in indices:
                if 0 <= s_idx < len(eval_stack.surfaces) and var_idx < len(variable_values):
                    raw_val = float(variable_values[var_idx])
                    if attr == "radius" and use_curvature_space:
                        y_sd = float(getattr(eval_stack.surfaces[s_idx], 'semi_diameter', 0.0) or 0.0)
                        if y_sd <= 1e-4:
                            y_sd = float(getattr(self, 'entrance_pupil_radius', 0.0) or 0.0)
                        if y_sd <= 1e-4:
                            y_sd = 10.0
                        c_limit = 0.95 / y_sd
                        clamped_c = float(np.clip(raw_val, -c_limit, c_limit))
                        phys_val = curvature_to_radius(clamped_c)
                    else:
                        phys_val = raw_val
                    setattr(eval_stack.surfaces[s_idx], attr, phys_val)
                var_idx += 1
                
        det_idx = None
        for idx, s in enumerate(eval_stack.surfaces):
            if s.surface_type == SurfaceType.DETECTOR:
                det_idx = idx
                break
        if det_idx is None:
            det_idx = len(eval_stack.surfaces) - 1

        if self.opt_tilt and var_idx < len(variable_values) and (0 <= det_idx < len(eval_stack.surfaces)):
            eval_stack.surfaces[det_idx].tilt_x = float(variable_values[var_idx])

        eval_idx = self.target_surface_index if (self.target_surface_index is not None and self.target_surface_index >= 0) else det_idx
        resolve_solves_and_pickups(eval_stack, primary_wl=self.wavelengths[0], eval_idx=eval_idx)

        eval_stack._update_accumulated_positions()
        eval_sys = OpticalSystem(eval_stack)

        # 3. Generate & trace rays
        z_dist = 35.0
        if len(eval_stack.surfaces) > 0:
            z_dist = eval_stack.surfaces[0].thickness
            if z_dist <= 0:
                z_dist = 1.0

        rays = self._generate_rays(self.entrance_pupil_radius, self.wavelengths, self.num_rays, z_dist=z_dist, eval_stack=eval_stack)
        traced_rays = eval_sys.trace_rays(rays, track_polarization=False)

        # 4. Extract unvignetted hits on target surface
        eval_idx = self.target_surface_index if (self.target_surface_index is not None and self.target_surface_index >= 0) else det_idx

        x_hits, y_hits = [], []
        dx_hits, dy_hits, dz_hits = [], [], []
        total_rays = max(1, len(traced_rays))
        valid_rays = 0

        field_wl_hits = {}
        chief_hits = {}

        for r in traced_rays:
            last_h = None
            if len(r.history) > 0:
                for h in r.history:
                    if h.get("surface_index") == eval_idx:
                        last_h = h
                        break
            
            if last_h is not None and not r.vignetted:
                valid_rays += 1
                x_val = float(last_h["x"])
                y_val = float(last_h["y"])
                x_hits.append(x_val)
                y_hits.append(y_val)
                if "dir_x" in last_h:
                    dx_hits.append(last_h["dir_x"])
                    dy_hits.append(last_h["dir_y"])
                    dz_hits.append(last_h["dir_z"])
                else:
                    dx_hits.append(r.dir[0])
                    dy_hits.append(r.dir[1])
                    dz_hits.append(r.dir[2])

                f_idx = getattr(r, "field_idx", 0)
                wl = round(float(r.wavelength), 4)
                is_chief = getattr(r, "is_chief", False)
                p_wt = getattr(r, "pupil_weight", 1.0)
                key = (f_idx, wl)
                if key not in field_wl_hits:
                    field_wl_hits[key] = []
                field_wl_hits[key].append((x_val, y_val, is_chief, p_wt))
                if is_chief and key not in chief_hits:
                    chief_hits[key] = (x_val, y_val)

        # 5. If explicit MeritFunction object is provided
        if self.merit_function_obj is not None:
            final_merit = float(self.merit_function_obj.evaluate(eval_sys, traced_rays, wavelengths=self.wavelengths, system_config=self.system_config))
            base_merit = getattr(self.merit_function_obj, "last_base_merit", final_merit)
            penalty_val = getattr(self.merit_function_obj, "last_penalty", 0.0)
            return base_merit, penalty_val

        # Anti-Vignetting / Ray Loss Penalty:
        loss_penalty = 0.0
        if total_rays > 0:
            loss_ratio = (total_rays - valid_rays) / float(total_rays)
            if loss_ratio > 0.0:
                loss_penalty = 2000.0 * loss_ratio + 5000.0 * (loss_ratio ** 2)

        if valid_rays < 3:
            # Continuous paraxial guidance rather than flat cliff
            smooth_guide = 0.0
            try:
                from optics_app.engine.matrix_tracer import MatrixTracer
                tracer = MatrixTracer()
                end_idx = eval_idx if (eval_idx is not None and eval_idx > 0) else len(eval_stack.surfaces) - 1
                opt_surfaces = eval_stack.surfaces[:end_idx]
                if len(opt_surfaces) > 1 and opt_surfaces[0].surface_type == SurfaceType.OBJECT:
                    opt_surfaces = opt_surfaces[1:]
                mat = tracer.build_from_system(opt_surfaces, wavelength_um=self.wavelengths[0])
                curr_phi = -mat.C
                if self.target_efl is not None and self.target_efl > 0:
                    target_phi = 1.0 / self.target_efl
                else:
                    det_dist = eval_stack.surfaces[end_idx - 1].thickness if end_idx > 0 else 50.0
                    target_phi = 1.0 / max(5.0, det_dist)
                phi_dev = abs(curr_phi - target_phi) * 1000.0
                det_t = eval_stack.surfaces[end_idx - 1].thickness if end_idx > 0 else 0.0
                y_pupil = self.entrance_pupil_radius * 0.7
                y_det = abs((mat.A + det_t * mat.C) * y_pupil)
                smooth_guide = 100.0 * phi_dev + 50.0 * y_det
            except Exception:
                smooth_guide = 0.0
            return 1000.0 + smooth_guide, 10000.0 + loss_penalty + smooth_guide

        # Evaluate EFL Constraint (if defined or target specifies EFL)
        efl_penalty = 0.0
        if self.target_efl is not None and self.target_efl > 0:
            try:
                from optics_app.engine.matrix_tracer import MatrixTracer
                tracer = MatrixTracer()
                end_idx = eval_idx if (eval_idx is not None and eval_idx > 0) else len(eval_stack.surfaces) - 1
                opt_surfaces = eval_stack.surfaces[:end_idx]
                if len(opt_surfaces) > 1 and opt_surfaces[0].surface_type == SurfaceType.OBJECT:
                    opt_surfaces = opt_surfaces[1:]
                mat = tracer.build_from_system(opt_surfaces, wavelength_um=self.wavelengths[0])
                curr_phi = -mat.C
                target_phi = 1.0 / self.target_efl
                phi_rel_err = (curr_phi - target_phi) / target_phi
                if curr_phi > 0:
                    efl_penalty = self.efl_weight * (150.0 * (phi_rel_err ** 2) + 50.0 * abs(phi_rel_err))
                else:
                    # Negative or zero optical power penalty: smoothly guide towards positive power
                    efl_penalty = self.efl_weight * (200.0 * (phi_rel_err ** 2) + 100.0 * abs(curr_phi - target_phi) * 100.0)
            except Exception:
                efl_penalty = 1000.0

        # 6. Compute pure optical merit
        chrom_penalty = 0.0
        if self.target == "Angular Spread (Collimation)":
            var_dx = float(np.var(dx_hits)) if len(dx_hits) > 0 else 1.0
            var_dy = float(np.var(dy_hits)) if len(dy_hits) > 0 else 1.0
            var_dz = float(np.var(dz_hits)) if len(dz_hits) > 0 else 1.0
            pure_merit = np.sqrt(var_dx + var_dy + var_dz) * 1e4
        elif self.target == "Telecentricity":
            dev_x = np.array(dx_hits)
            dev_y = np.array(dy_hits)
            dev_z = np.array(dz_hits) - 1.0
            mean_dev_sq = np.mean(dev_x**2 + dev_y**2 + dev_z**2)
            pure_merit = np.sqrt(mean_dev_sq) * 1e4
        elif "RMS Spot X" in self.target or self.target == "Spot X":
            wl_rms_x_list = []
            for wl in self.wavelengths:
                wl_x = [h["x"] for tr in traced_rays if not tr.vignetted and abs(tr.wavelength - wl) < 1e-6 for h in tr.history if h.get("surface_index") == eval_idx]
                if wl_x:
                    spot = SpotDiagram(wl_x, np.zeros_like(wl_x))
                    wl_rms_x_list.append(spot.calculate_rms_x())
            pure_merit = math.sqrt(sum(r**2 for r in wl_rms_x_list) / len(wl_rms_x_list)) if wl_rms_x_list else 1000.0
        elif "RMS Spot Y" in self.target or self.target == "Spot Y":
            wl_rms_y_list = []
            for wl in self.wavelengths:
                wl_y = [h["y"] for tr in traced_rays if not tr.vignetted and abs(tr.wavelength - wl) < 1e-6 for h in tr.history if h.get("surface_index") == eval_idx]
                if wl_y:
                    spot = SpotDiagram(np.zeros_like(wl_y), wl_y)
                    wl_rms_y_list.append(spot.calculate_rms_y())
            pure_merit = math.sqrt(sum(r**2 for r in wl_rms_y_list) / len(wl_rms_y_list)) if wl_rms_y_list else 1000.0
        elif "Target Spot" in self.target or self.target == "Target Size":
            wl_rms_list = []
            for wl in self.wavelengths:
                wl_x, wl_y = [], []
                for tr in traced_rays:
                    if not tr.vignetted and abs(tr.wavelength - wl) < 1e-6:
                        for h in tr.history:
                            if h.get("surface_index") == eval_idx:
                                wl_x.append(h["x"])
                                wl_y.append(h["y"])
                                break
                if wl_x:
                    spot = SpotDiagram(wl_x, wl_y)
                    wl_rms_list.append(spot.calculate_rms_radius())
            actual_rms = math.sqrt(sum(r**2 for r in wl_rms_list) / len(wl_rms_list)) if wl_rms_list else 1000.0
            pure_merit = abs(actual_rms - self.target_spot_size)
        elif self.target == "Spectrometer Spread":
            wl_rms_list = []
            for wl in self.wavelengths:
                wl_x, wl_y = [], []
                for tr in traced_rays:
                    if not tr.vignetted and abs(tr.wavelength - wl) < 1e-6:
                        for h in tr.history:
                            if h.get("surface_index") == eval_idx:
                                wl_x.append(h["x"])
                                wl_y.append(h["y"])
                                break
                if wl_x:
                    spot = SpotDiagram(wl_x, wl_y)
                    wl_rms_list.append(spot.calculate_rms_radius())
            spot_blur = math.sqrt(sum(r**2 for r in wl_rms_list) / len(wl_rms_list)) if wl_rms_list else 1000.0

            if len(self.wavelengths) > 1:
                wl_min = min(self.wavelengths)
                wl_max = max(self.wavelengths)
                
                x_min_hits = [h["x"] for tr in traced_rays if not tr.vignetted and not tr.tir and abs(tr.wavelength - wl_min) < 1e-6 for h in tr.history if h.get("surface_index") == eval_idx]
                x_max_hits = [h["x"] for tr in traced_rays if not tr.vignetted and not tr.tir and abs(tr.wavelength - wl_max) < 1e-6 for h in tr.history if h.get("surface_index") == eval_idx]
                
                if x_min_hits and x_max_hits:
                    centroid_min = np.mean(x_min_hits)
                    centroid_max = np.mean(x_max_hits)
                    actual_spread = abs(centroid_max - centroid_min)
                    spread_error = abs(actual_spread - self.target_spread)
                    pure_merit = math.sqrt(spot_blur**2 + spread_error**2)
                else:
                    pure_merit = spot_blur + self.penalty_weight
            else:
                pure_merit = spot_blur
        elif "Aberration Reduction" in self.target or "Seidel" in self.target:
            wl_rms_list = []
            for wl in self.wavelengths:
                wl_x, wl_y = [], []
                for tr in traced_rays:
                    if not tr.vignetted and abs(tr.wavelength - wl) < 1e-6:
                        for h in tr.history:
                            if h.get("surface_index") == eval_idx:
                                wl_x.append(h["x"])
                                wl_y.append(h["y"])
                                break
                if wl_x:
                    spot = SpotDiagram(wl_x, wl_y)
                    wl_rms_list.append(spot.calculate_rms_radius())
            spot_rms = math.sqrt(sum(r**2 for r in wl_rms_list) / len(wl_rms_list)) if wl_rms_list else 1000.0

            try:
                try:
                    from optics_app.analysis.aberration import calculate_seidel_coefficients
                except ImportError:
                    from v2.analysis.aberration import calculate_seidel_coefficients
                seidel = calculate_seidel_coefficients(eval_stack, wavelength_um=self.wavelengths[0])
                aberration_merit = math.sqrt(
                    10.0 * (seidel.S_I**2 + seidel.S_II**2 + seidel.S_III**2) +
                    5.0 * (seidel.S_IV**2 + seidel.S_V**2) +
                    25.0 * (seidel.C_L**2 + seidel.C_T**2)
                )
            except Exception:
                aberration_merit = 100.0
            pure_merit = spot_rms + aberration_merit
        elif "Wavefront Error" in self.target or "OPD" in self.target:
            working_f_num = extract_working_f_number(eval_stack, self.wavelengths[0], self.entrance_pupil_radius, eval_idx)
            all_field_indices = sorted(list(set(k[0] for k in field_wl_hits.keys()))) if field_wl_hits else [0]

            def get_f_weight(f_idx: int) -> float:
                if self.system_config is not None and getattr(self.system_config, "fields", None):
                    if 0 <= f_idx < len(self.system_config.fields):
                        return float(self.system_config.fields[f_idx].weight)
                if self.fields and 0 <= f_idx < len(self.fields):
                    f = self.fields[f_idx]
                    if hasattr(f, "weight"):
                        return float(f.weight)
                    elif isinstance(f, (list, tuple)) and len(f) > 2:
                        return float(f[2])
                if len(all_field_indices) == 3 and f_idx < 3:
                    return [1.0, 2.0, 1.0][f_idx]
                return 1.0

            weighted_sq_sum = 0.0
            total_weight = 0.0

            for f_idx in all_field_indices:
                w_f = get_f_weight(f_idx)
                f_keys = [k for k in field_wl_hits if k[0] == f_idx]
                if not f_keys:
                    weighted_sq_sum += w_f * (100.0 ** 2)
                    total_weight += w_f
                    continue
                wl_w_rms_list = []
                for k in f_keys:
                    hits = field_wl_hits[k]
                    xs = [h[0] for h in hits]
                    ys = [h[1] for h in hits]
                    c_pt = chief_hits.get(k, None) if self.reference == "chief_ray" else None
                    spot = SpotDiagram(xs, ys, reference=self.reference, chief_ray_coord=c_pt)
                    s_rms = spot.calculate_rms_radius()
                    wl = k[1]
                    wl_mm = wl * 1e-3 if wl > 0.01 else wl
                    w_rms = s_rms / (8.0 * working_f_num * max(1e-6, wl_mm))
                    wl_w_rms_list.append(float(w_rms))
                f_opd_rms_sq = float(np.mean([w**2 for w in wl_w_rms_list])) if wl_w_rms_list else 100.0**2
                weighted_sq_sum += w_f * f_opd_rms_sq
                total_weight += w_f

            pure_merit = math.sqrt(weighted_sq_sum / max(1e-9, total_weight)) if total_weight > 0 else 1000.0
        elif "Achromatic Focus" in self.target or "Achromat" in self.target:
            wl_rms_list = []
            for wl in self.wavelengths:
                wl_x, wl_y = [], []
                for tr in traced_rays:
                    if not tr.vignetted and abs(tr.wavelength - wl) < 1e-6:
                        for h in tr.history:
                            if h.get("surface_index") == eval_idx:
                                wl_x.append(h["x"])
                                wl_y.append(h["y"])
                                break
                if wl_x:
                    spot = SpotDiagram(wl_x, wl_y)
                    wl_rms_list.append(spot.calculate_rms_radius())
            spot_rms = math.sqrt(sum(r**2 for r in wl_rms_list) / len(wl_rms_list)) if wl_rms_list else 1000.0

            # Axial chromatic focal shift (Delta BFL across extreme wavelengths)
            delta_bfl = 0.0
            if len(self.wavelengths) > 1:
                wl_min = min(self.wavelengths)
                wl_max = max(self.wavelengths)
                try:
                    _, bfl_min = compute_system_efl(eval_stack.surfaces, wavelength_um=wl_min, eval_surface_idx=eval_idx)
                    _, bfl_max = compute_system_efl(eval_stack.surfaces, wavelength_um=wl_max, eval_surface_idx=eval_idx)
                    if not (math.isinf(bfl_min) or math.isinf(bfl_max) or bfl_min <= 0 or bfl_max <= 0):
                        delta_bfl = abs(bfl_min - bfl_max)
                    else:
                        delta_bfl = 10.0
                except Exception:
                    delta_bfl = 0.0
            pure_merit = spot_rms
            chrom_penalty = 2.0 * delta_bfl
        else:
            # Default RMS Spot Radius
            all_field_indices = sorted(list(set(k[0] for k in field_wl_hits.keys()))) if field_wl_hits else [0]

            def get_f_weight(f_idx: int) -> float:
                if self.system_config is not None and getattr(self.system_config, "fields", None):
                    if 0 <= f_idx < len(self.system_config.fields):
                        return float(self.system_config.fields[f_idx].weight)
                if self.fields and 0 <= f_idx < len(self.fields):
                    f = self.fields[f_idx]
                    if hasattr(f, "weight"):
                        return float(f.weight)
                    elif isinstance(f, (list, tuple)) and len(f) > 2:
                        return float(f[2])
                if len(all_field_indices) == 3 and f_idx < 3:
                    return [1.0, 2.0, 1.0][f_idx]
                return 1.0

            weighted_sq_sum = 0.0
            total_weight = 0.0

            for f_idx in all_field_indices:
                w_f = get_f_weight(f_idx)
                f_keys = [k for k in field_wl_hits if k[0] == f_idx]
                if not f_keys:
                    weighted_sq_sum += w_f * (1000.0 ** 2)
                    total_weight += w_f
                    continue
                f_rms_sq_list = []
                for k in f_keys:
                    hits = field_wl_hits[k]
                    xs = [h[0] for h in hits]
                    ys = [h[1] for h in hits]
                    c_pt = chief_hits.get(k, None) if self.reference == "chief_ray" else None
                    spot = SpotDiagram(xs, ys, reference=self.reference, chief_ray_coord=c_pt)
                    f_rms_sq_list.append(spot.calculate_rms_radius() ** 2)
                f_rms_sq = float(np.mean(f_rms_sq_list)) if f_rms_sq_list else 1000.0**2
                weighted_sq_sum += w_f * f_rms_sq
                total_weight += w_f

            if total_weight > 0 and weighted_sq_sum < float("inf"):
                pure_merit = math.sqrt(weighted_sq_sum / max(1e-9, total_weight))
            else:
                pure_merit = 1000.0

        # 7. Compute barrier penalties
        vignetted_fraction = (total_rays - valid_rays) / total_rays
        penalty = float(self.penalty_weight * vignetted_fraction) + chrom_penalty

        # Collision / physical interference penalty
        interference_count = sum(1 for tr in traced_rays if getattr(tr, 'interference', False))
        if interference_count > 0:
            penalty += 5000.0 * interference_count

        # Physical validity constraints (negative thickness, center/edge thickness)
        var_thickness_indices = var_dict.get("thickness", [])
        var_radius_indices = var_dict.get("radius", [])

        for i, s in enumerate(eval_stack.surfaces):
            # Enforce non-negative thickness on all surfaces
            if s.thickness < 0:
                dev = abs(s.thickness)
                penalty += 5000.0 * (dev + dev**2)

            # Center thickness, edge thickness, and air clearance checks on active variable surfaces
            is_variable_element = (
                i in var_thickness_indices or
                i in var_radius_indices or
                (i + 1) in var_radius_indices
            )
            should_check = self.enforce_manufacturing_constraints or is_variable_element
            if should_check and (i + 1 < len(eval_stack.surfaces)):
                s_next = eval_stack.surfaces[i + 1]
                sd_cur = self._effective_clear_semi_diameter(s)
                sd_nxt = self._effective_clear_semi_diameter(s_next)
                r_edge = min(sd_cur, sd_nxt)
                sag1 = s.sag(r_edge)
                sag2 = s_next.sag(r_edge)
                edge_thickness = s.thickness + sag2 - sag1

                min_t_glass = self.min_glass_thickness if self.enforce_manufacturing_constraints else 0.01
                min_t_edge = self.min_edge_thickness if self.enforce_manufacturing_constraints else 0.01

                mech_margin = 1.0 if sd_cur < float(getattr(s, 'semi_diameter', 0.0) or 0.0) else getattr(self, 'mounting_margin', 1.5)
                sd_mech = r_edge + mech_margin
                sag1_m = s.sag(sd_mech)
                sag2_m = s_next.sag(sd_mech)
                edge_mech = s.thickness + sag2_m - sag1_m

                if s.is_glass:
                    if s.thickness < min_t_glass:
                        dev = min_t_glass - s.thickness
                        penalty += 5000.0 * (dev + dev**2)
                    elif self.enforce_manufacturing_constraints and s.thickness > self.max_glass_thickness:
                        dev = s.thickness - self.max_glass_thickness
                        penalty += 4000.0 * (dev + dev**2)

                    if edge_thickness < min_t_edge:
                        dev = min_t_edge - edge_thickness
                        penalty += 5000.0 * (dev + dev**2)

                    max_t_edge = getattr(self, 'max_edge_thickness', 20.0)
                    if self.enforce_manufacturing_constraints and edge_thickness > max_t_edge:
                        dev = edge_thickness - max_t_edge
                        penalty += 3000.0 * (dev + dev**2)

                    # Mechanical mounting rim edge thickness (MNEG at mechanical barrel diameter)
                    min_mech_edge = getattr(self, 'min_mechanical_edge', 0.8)
                    if self.enforce_manufacturing_constraints and edge_mech < min_mech_edge:
                        dev = min_mech_edge - edge_mech
                        penalty += 5000.0 * (dev + dev**2)

                    # Aspect ratio constraint D_mech / t_center <= max_aspect_ratio
                    if self.enforce_manufacturing_constraints:
                        diam_mech = 2.0 * sd_mech
                        aspect = diam_mech / max(0.01, s.thickness)
                        max_aspect = getattr(self, 'max_aspect_ratio', 10.0)
                        if aspect > max_aspect:
                            dev = aspect - max_aspect
                            penalty += 3500.0 * (dev + dev**2)
                else:
                    is_det = (s_next.surface_type == SurfaceType.DETECTOR)
                    is_obj = (s.surface_type == SurfaceType.OBJECT)
                    if not is_obj:
                        req_air = self.min_bfl if is_det else self.min_air_gap
                        if self.enforce_manufacturing_constraints and s.thickness < req_air:
                            dev = req_air - s.thickness
                            penalty += 5000.0 * (dev + dev**2)
                        elif s.thickness < 0.01:
                            dev = 0.01 - s.thickness
                            penalty += 5000.0 * (dev + dev**2)

                        # Air gap edge clearance & spacer ring seating (MNEA)
                        if self.enforce_manufacturing_constraints and not is_det:
                            req_air_edge = getattr(self, 'min_edge_air', 0.5)
                            if edge_thickness < req_air_edge:
                                dev = req_air_edge - edge_thickness
                                penalty += 4000.0 * (dev + dev**2)
                            if edge_mech < req_air_edge:
                                dev = req_air_edge - edge_mech
                                penalty += 4000.0 * (dev + dev**2)

            if is_variable_element and self.enforce_manufacturing_constraints and abs(s.radius) > 1e-4:
                req_r = 1.05 * self._effective_clear_semi_diameter(s)
                if abs(s.radius) < req_r:
                    dev = req_r - abs(s.radius)
                    penalty += 3000.0 * (dev + dev**2)

        # Ray angle of incidence constraint (RAID)
        if self.enforce_manufacturing_constraints and getattr(self, 'max_incidence_angle_deg', None):
            max_theta_rad = math.radians(float(self.max_incidence_angle_deg))
            for tr in traced_rays:
                if not tr.vignetted:
                    for h in tr.history:
                        th_inc = h.get("theta_inc", 0.0)
                        if th_inc > max_theta_rad:
                            dev = (th_inc - max_theta_rad) * 10.0
                            penalty += 2000.0 * (dev + dev**2)
                            break

        # Total Track Length constraint (TOTR / MXTR)
        if self.enforce_manufacturing_constraints and getattr(self, 'max_total_track', None):
            start_s = 1 if len(eval_stack.surfaces) > 1 and eval_stack.surfaces[0].surface_type == SurfaceType.OBJECT else 0
            totr_val = sum(max(0.0, s_elem.thickness) for s_elem in eval_stack.surfaces[start_s:])
            if totr_val > self.max_total_track:
                dev = totr_val - self.max_total_track
                penalty += 2000.0 * (dev + dev**2)

        # Tolerancing as-built desensitization penalty
        if self.enforce_manufacturing_constraints and getattr(self, 'desensitize_tolerances', False):
            as_built_w = float(getattr(self, 'as_built_weight', 10.0))
            for idx_s, surf_item in enumerate(eval_stack.surfaces[:-1]):
                if abs(surf_item.radius) > 1e-4:
                    g_c = str(getattr(surf_item, 'glass', 'AIR') or 'AIR').upper()
                    g_n = str(getattr(eval_stack.surfaces[idx_s + 1], 'glass', 'AIR') or 'AIR').upper()
                    n_c = 1.5 if g_c not in ('AIR', 'MIRROR', '') else 1.0
                    n_n = 1.5 if g_n not in ('AIR', 'MIRROR', '') else 1.0
                    dn = abs(n_n - n_c)
                    if dn > 0.05:
                        phi_norm = (dn / abs(surf_item.radius)) * self._effective_clear_semi_diameter(surf_item)
                        if phi_norm > 0.35:
                            dev = phi_norm - 0.35
                            penalty += as_built_w * 50.0 * (dev + dev**2)



        penalty += float(efl_penalty + loss_penalty)
        return float(pure_merit), float(penalty)

    def merit_function(self, variable_values: np.ndarray, variable_indices: Union[dict, List[int]]) -> float:
        """
        Evaluate RMS spot radius on detector plane for candidate vector.
        Returns pure_merit + numerical penalty.
        """
        pure_merit, penalty = self._evaluate_merit_and_penalty(variable_values, variable_indices, check_abort=True)
        total = pure_merit + penalty
        if total < self.best_rms:
            self.best_rms = total
            self.best_x = variable_values.copy()
        return total

    def validate(self, variable_indices: Optional[Union[dict, List[int]]] = None, bounds: Optional[List[Tuple[float, float]]] = None) -> ValidationResult:
        """Execute pre-flight check to validate system geometry, parameters, and ray transmission."""
        return OptimizationValidator.validate_system(
            stack=self.system.stack,
            variable_indices=variable_indices,
            wavelengths=self.wavelengths,
            entrance_pupil_radius=self.entrance_pupil_radius,
            num_rays=self.num_rays,
            target=self.target,
            target_spread=self.target_spread,
            target_surface_index=self.target_surface_index,
            target_spot_size=self.target_spot_size,
            bounds=bounds,
            max_iter=self.max_iter,
            opt_tilt=self.opt_tilt
        )

    def auto_repair(self, variable_indices: Optional[Union[dict, List[int]]] = None) -> Tuple[SurfaceStack, Dict[str, List[int]], List[str]]:
        """Automatically repair common optical configuration errors."""
        return OptimizationValidator.auto_repair(
            stack=self.system.stack,
            variable_indices=variable_indices,
            wavelengths=self.wavelengths,
            target=self.target
        )

    def _seed_paraxial_efl_if_stalled(self, var_dict: Dict[str, List[int]]) -> None:
        """
        Global First-Order Optical Bootstrapping:
        If the initial configuration is unpowered (e.g. flat plates R=0), diverging,
        or far from target EFL such that no rays can converge, seed paraxial curvatures
        and BFL according to analytical thin-lens / achromat formulas.
        """
        rad_vars = var_dict.get("radius", [])
        if not rad_vars:
            return

        curr_efl, curr_bfl = compute_system_efl(self.system.stack.surfaces, wavelength_um=self.wavelengths[0])
        all_rads_zero = all(abs(getattr(self.system.stack.surfaces[i], 'radius', 0.0)) < 1e-5 for i in rad_vars)

        target_f = self.target_efl
        if target_f is None or target_f <= 0:
            if not all_rads_zero and not math.isinf(curr_efl) and 0.0 < curr_efl < 1e4:
                # System already has valid physical optical power; preserve user prescription
                return
            if "Focus" in str(self.target) or "Spot" in str(self.target):
                det_idx = None
                for idx, s in enumerate(self.system.stack.surfaces):
                    if s.surface_type == SurfaceType.DETECTOR:
                        det_idx = idx
                        break
                if det_idx is not None and det_idx > 0:
                    d = self.system.stack.surfaces[det_idx - 1].thickness
                    if d > 5.0:
                        target_f = float(d)

        if target_f is None or target_f <= 0:
            return

        is_stalled = (
            math.isinf(curr_efl) or 
            abs(curr_efl) > 1e4 or 
            curr_efl <= 0.0 or 
            all_rads_zero or
            (self.target_efl is not None and abs(curr_efl - target_f) / target_f > 0.8)
        )
        if not is_stalled:
            return

        phi_target = 1.0 / target_f

        glasses = []
        for idx, s in enumerate(self.system.stack.surfaces):
            gname = getattr(s, 'glass', 'AIR') or 'AIR'
            if s.is_glass and gname.upper() not in ('AIR', ''):
                try:
                    nd = get_refractive_index(gname, 0.5876)
                    nF = get_refractive_index(gname, 0.4861)
                    nC = get_refractive_index(gname, 0.6563)
                    vd = (nd - 1.0) / max(1e-6, abs(nF - nC))
                except Exception:
                    nd = 1.5168
                    vd = 64.17
                glasses.append((idx, gname, nd, vd))

        if len(glasses) == 2 and rad_vars:
            idx1, g1, n1, V1 = glasses[0]
            idx2, g2, n2, V2 = glasses[1]
            if abs(V1 - V2) > 3.0:
                phi1 = phi_target * (V1 / (V1 - V2))
                phi2 = -phi_target * (V2 / (V1 - V2))
            else:
                phi1 = 0.5 * phi_target
                phi2 = 0.5 * phi_target

            back_idx = idx2 + 1
            interface_idx = idx1 + 1

            if interface_idx in rad_vars and interface_idx != back_idx:
                self.system.stack.surfaces[idx1].radius = float(0.8 * target_f)
                self.system.stack.surfaces[interface_idx].radius = float(-0.35 * target_f)
                if back_idx in rad_vars and back_idx < len(self.system.stack.surfaces):
                    self.system.stack.surfaces[back_idx].radius = float(-1.5 * target_f)
            else:
                if idx1 in rad_vars:
                    self.system.stack.surfaces[idx1].radius = float((n1 - 1.0) / phi1)
                if back_idx in rad_vars and back_idx < len(self.system.stack.surfaces):
                    self.system.stack.surfaces[back_idx].radius = float((1.0 - n2) / phi2)

        elif len(glasses) == 1 and rad_vars:
            idx1, g1, n1, _ = glasses[0]
            back_idx = idx1 + 1
            if idx1 in rad_vars and back_idx in rad_vars and back_idx < len(self.system.stack.surfaces):
                r_val = float(2.0 * (n1 - 1.0) / phi_target)
                self.system.stack.surfaces[idx1].radius = r_val
                self.system.stack.surfaces[back_idx].radius = -r_val
            elif idx1 in rad_vars:
                self.system.stack.surfaces[idx1].radius = float((n1 - 1.0) / phi_target)
            elif back_idx in rad_vars and back_idx < len(self.system.stack.surfaces):
                self.system.stack.surfaces[back_idx].radius = float((1.0 - n1) / phi_target)

        else:
            count = max(1, len(rad_vars))
            phi_per_surf = phi_target / count
            for s_idx in rad_vars:
                if 0 <= s_idx < len(self.system.stack.surfaces):
                    s = self.system.stack.surfaces[s_idx]
                    prev_s = self.system.stack.surfaces[s_idx - 1] if s_idx > 0 else None
                    g_curr = getattr(s, 'glass', 'AIR') or 'AIR'
                    g_prev = getattr(prev_s, 'glass', 'AIR') if prev_s else 'AIR'
                    try:
                        n_curr = get_refractive_index(g_curr, 0.5876) if s.is_glass else 1.0
                        n_prev = get_refractive_index(g_prev, 0.5876) if (prev_s and prev_s.is_glass) else 1.0
                    except Exception:
                        n_curr, n_prev = 1.5, 1.0
                    dn = n_curr - n_prev
                    if abs(dn) < 0.05:
                        dn = 0.5
                    c_seed = phi_per_surf / dn
                    self.system.stack.surfaces[s_idx].radius = float(1.0 / c_seed) if abs(c_seed) > 1e-6 else 100.0

        new_efl, new_bfl = compute_system_efl(self.system.stack.surfaces, wavelength_um=self.wavelengths[0])
        det_idx = None
        for idx, s in enumerate(self.system.stack.surfaces):
            if s.surface_type == SurfaceType.DETECTOR:
                det_idx = idx
                break
        if det_idx is not None and det_idx > 0:
            last_opt = self.system.stack.surfaces[det_idx - 1]
            if getattr(last_opt, 'thickness_solve_type', '') == 'MarginalRayFocus' or (det_idx - 1 in var_dict.get('thickness', [])):
                if new_bfl > 1.0 and not math.isinf(new_bfl):
                    last_opt.thickness = float(new_bfl)
                elif target_f > 0:
                    last_opt.thickness = float(0.85 * target_f)

    def _compute_dls_residuals(
        self,
        x_param: np.ndarray,
        var_dict: Dict[str, List[int]],
        use_curv: bool,
        bounds_param: Optional[List[Tuple[float, float]]] = None
    ) -> np.ndarray:
        """
        Compute vector of optical residuals for Damped Least Squares (Levenberg-Marquardt / TRF).
        Operates in curvature-space for radius variables.
        Returns 1D numpy array of weighted residuals.
        """
        if self.abort:
            raise OptimizationAbortedException("Optimization aborted by user.")

        self.current_iter += 1
        if self.progress_callback is not None:
            self._invoke_progress(self.current_iter, self.max_iter)

        eval_stack = self._clone_stack(self.system.stack)

        var_idx = 0
        for attr, indices in var_dict.items():
            for s_idx in indices:
                if 0 <= s_idx < len(eval_stack.surfaces) and var_idx < len(x_param):
                    val = float(x_param[var_idx])
                    if attr == "radius" and use_curv:
                        y_sd = float(getattr(eval_stack.surfaces[s_idx], 'semi_diameter', 0.0) or 0.0)
                        if y_sd <= 1e-4:
                            y_sd = float(getattr(self, 'entrance_pupil_radius', 0.0) or 0.0)
                        if y_sd <= 1e-4:
                            y_sd = 10.0
                        c_limit = 0.95 / y_sd
                        clamped_c = float(np.clip(val, -c_limit, c_limit))
                        phys_r = curvature_to_radius(clamped_c)
                        setattr(eval_stack.surfaces[s_idx], "radius", phys_r)
                    else:
                        setattr(eval_stack.surfaces[s_idx], attr, val)
                var_idx += 1

        det_idx = None
        for idx, s in enumerate(eval_stack.surfaces):
            if s.surface_type == SurfaceType.DETECTOR:
                det_idx = idx
                break
        if det_idx is None:
            det_idx = len(eval_stack.surfaces) - 1

        if self.opt_tilt and var_idx < len(x_param) and (0 <= det_idx < len(eval_stack.surfaces)):
            eval_stack.surfaces[det_idx].tilt_x = float(x_param[var_idx])

        eval_idx = self.target_surface_index if (self.target_surface_index is not None and self.target_surface_index >= 0) else det_idx
        resolve_solves_and_pickups(eval_stack, primary_wl=self.wavelengths[0], eval_idx=eval_idx)
        eval_stack._update_accumulated_positions()
        eval_sys = OpticalSystem(eval_stack)

        z_dist = 35.0
        if len(eval_stack.surfaces) > 0:
            z_dist = eval_stack.surfaces[0].thickness
            if z_dist <= 0:
                z_dist = 1.0

        rays = self._generate_rays(self.entrance_pupil_radius, self.wavelengths, self.num_rays, z_dist=z_dist, eval_stack=eval_stack)
        traced_rays = eval_sys.trace_rays(rays, track_polarization=False)

        residuals: List[float] = []
        total_rays = len(traced_rays)
        valid_rays = sum(1 for r in traced_rays if not r.vignetted and any(h.get("surface_index") == eval_idx for h in r.history))

        # 1. Transverse Spot & Wavefront Residuals
        is_wavefront = ("Wavefront Error" in self.target or "OPD" in self.target)
        is_seidel = ("Aberration Reduction" in self.target or "Seidel" in self.target)
        is_x_only = ("Spot X" in self.target)
        is_y_only = ("Spot Y" in self.target)

        # Field-partitioned reference points (centroid or chief ray)
        field_hits = {}    # f_idx -> list of (x, y)
        field_chiefs = {}  # f_idx -> (x, y)

        for tr in traced_rays:
            if not tr.vignetted:
                for h in tr.history:
                    if h.get("surface_index") == eval_idx:
                        f_idx = getattr(tr, "field_idx", 0)
                        if f_idx not in field_hits:
                            field_hits[f_idx] = []
                        field_hits[f_idx].append((float(h["x"]), float(h["y"])))
                        if getattr(tr, "is_chief", False) and f_idx not in field_chiefs:
                            field_chiefs[f_idx] = (float(h["x"]), float(h["y"]))
                        break

        all_field_indices = sorted(list(set(getattr(tr, "field_idx", 0) for tr in traced_rays)))
        if not all_field_indices:
            all_field_indices = [0]

        def get_dls_field_weight(f_idx: int) -> float:
            if self.system_config is not None and getattr(self.system_config, "fields", None):
                if 0 <= f_idx < len(self.system_config.fields):
                    return float(self.system_config.fields[f_idx].weight)
            if self.fields and 0 <= f_idx < len(self.fields):
                f = self.fields[f_idx]
                if hasattr(f, "weight"):
                    return float(f.weight)
                elif isinstance(f, (list, tuple)) and len(f) > 2:
                    return float(f[2])
            if len(all_field_indices) == 3 and f_idx < 3:
                return [1.0, 2.0, 1.0][f_idx]
            return 1.0

        total_field_weights = sum(get_dls_field_weight(f) for f in all_field_indices)
        if total_field_weights <= 0:
            total_field_weights = 1.0

        field_refs = {}
        for f_idx in all_field_indices:
            hits = field_hits.get(f_idx, [])
            if self.reference == "chief_ray" and f_idx in field_chiefs:
                field_refs[f_idx] = field_chiefs[f_idx]
            elif hits:
                field_refs[f_idx] = (float(np.mean([h[0] for h in hits])), float(np.mean([h[1] for h in hits])))
            else:
                field_refs[f_idx] = (0.0, 0.0)

        # Rayleigh-Hamilton Wavefront Scaling
        working_f_num = extract_working_f_number(eval_stack, self.wavelengths[0], self.entrance_pupil_radius, eval_idx)
        wl_mm = self.wavelengths[0] * 1e-3 if self.wavelengths[0] > 0.01 else self.wavelengths[0]
        s_rh = 1.0 / (8.0 * working_f_num * max(1e-6, wl_mm)) if is_wavefront else 1.0

        # Invariant spot residuals: each ray always contributes exactly 2 entries (or 1 if x/y only)
        for tr in traced_rays:
            hit = None
            if not tr.vignetted:
                for h in tr.history:
                    if h.get("surface_index") == eval_idx:
                        hit = h
                        break

            f_idx = getattr(tr, "field_idx", 0)
            n_f = len(field_hits.get(f_idx, []))
            w_f = get_dls_field_weight(f_idx)
            scale_f = math.sqrt(w_f / total_field_weights) / math.sqrt(max(1, n_f))
            if is_wavefront:
                scale_f *= s_rh

            if hit is not None:
                cx, cy = field_refs.get(f_idx, (0.0, 0.0))
                if is_x_only:
                    residuals.append((hit["x"] - cx) * scale_f)
                elif is_y_only:
                    residuals.append((hit["y"] - cy) * scale_f)
                else:
                    residuals.append((hit["x"] - cx) * scale_f)
                    residuals.append((hit["y"] - cy) * scale_f)
            else:
                vig_val = 0.0
                if is_x_only or is_y_only:
                    residuals.append(vig_val)
                else:
                    residuals.append(vig_val)
                    residuals.append(vig_val)

        # 2. Seidel Aberration Residuals
        if is_seidel:
            try:
                try:
                    from optics_app.analysis.aberration import calculate_seidel_coefficients
                except ImportError:
                    from v2.analysis.aberration import calculate_seidel_coefficients
                seidel = calculate_seidel_coefficients(eval_stack, wavelength_um=self.wavelengths[0])
                residuals.append(seidel.S_I * 10.0)
                residuals.append(seidel.S_II * 10.0)
                residuals.append(seidel.S_III * 10.0)
                residuals.append(seidel.S_IV * 5.0)
                residuals.append(seidel.S_V * 5.0)
                residuals.append(seidel.C_L * 25.0)
                residuals.append(seidel.C_T * 25.0)
            except Exception:
                residuals.extend([0.0] * 7)

        # 3. Chromatic focal shift residual (Achromatic Focus & multi-glass polychromatic spot focus)
        is_achromat_target = ("Achromatic Focus" in self.target or "Achromat" in self.target)
        n_glass_elements = sum(1 for s_el in eval_stack.surfaces if s_el.is_glass)
        is_multi_glass_focus = (
            ("Spot" in self.target or "Focus" in self.target)
            and len(self.wavelengths) > 1
            and n_glass_elements >= 2
            and len(var_dict.get("radius", [])) >= 2
        )
        if (is_achromat_target or is_multi_glass_focus) and len(self.wavelengths) > 1:
            try:
                wl_min = min(self.wavelengths)
                wl_max = max(self.wavelengths)
                _, bfl_min = compute_system_efl(eval_stack.surfaces, wavelength_um=wl_min, eval_surface_idx=eval_idx)
                _, bfl_max = compute_system_efl(eval_stack.surfaces, wavelength_um=wl_max, eval_surface_idx=eval_idx)
                chrom_weight = 1.5 if is_achromat_target else 0.85
                if not (math.isinf(bfl_min) or math.isinf(bfl_max) or bfl_min <= 0 or bfl_max <= 0):
                    residuals.append((bfl_min - bfl_max) * chrom_weight)
                else:
                    residuals.append(5.0 * chrom_weight)
            except Exception:
                residuals.append(0.0)

        # 4. Target EFL Residual (Smooth, continuous ABCD matrix power formulation)
        if self.target_efl is not None and self.target_efl > 0:
            try:
                from optics_app.engine.matrix_tracer import MatrixTracer
                tracer = MatrixTracer()
                end_idx = eval_idx if (eval_idx is not None and eval_idx > 0) else len(eval_stack.surfaces) - 1
                opt_surfaces = eval_stack.surfaces[:end_idx]
                if len(opt_surfaces) > 1 and opt_surfaces[0].surface_type == SurfaceType.OBJECT:
                    opt_surfaces = opt_surfaces[1:]
                wl_efl = self.wavelengths[len(self.wavelengths) // 2] if self.wavelengths else 0.5876
                mat = tracer.build_from_system(opt_surfaces, wavelength_um=wl_efl)
                curr_phi = -mat.C
                target_phi = 1.0 / self.target_efl
                phi_rel_err = (curr_phi - target_phi) / target_phi
                scale = math.sqrt(max(1.0, self.efl_weight)) * 2.5
                residuals.append(phi_rel_err * scale)
            except Exception:
                residuals.append(25.0)

        # 5. Vignetting / Ray loss residual (Invariant 1 scalar)
        loss_ratio = (total_rays - valid_rays) / float(max(1, total_rays))
        loss_penalty = 0.0
        if loss_ratio > 0.0:
            if valid_rays < 3:
                loss_penalty = 500.0 * loss_ratio
            elif self.enforce_manufacturing_constraints:
                loss_penalty = 40.0 * loss_ratio
        residuals.append(loss_penalty)

        # 6. Physical center, edge thickness, and air clearance constraints (Invariant size per surface)
        min_t_glass = self.min_glass_thickness if self.enforce_manufacturing_constraints else 0.01
        min_t_edge = self.min_edge_thickness if self.enforce_manufacturing_constraints else 0.01

        for i, s in enumerate(eval_stack.surfaces):
            residuals.append(max(0.0, -s.thickness) * 50.0)
            if i + 1 < len(eval_stack.surfaces):
                s_next = eval_stack.surfaces[i + 1]
                sd_cur = self._effective_clear_semi_diameter(s)
                sd_nxt = self._effective_clear_semi_diameter(s_next)
                r_edge = min(sd_cur, sd_nxt)
                sag1 = s.sag(r_edge)
                sag2 = s_next.sag(r_edge)
                edge_t = s.thickness + sag2 - sag1

                mech_margin = 1.0 if sd_cur < float(getattr(s, 'semi_diameter', 0.0) or 0.0) else getattr(self, 'mounting_margin', 1.5)
                sd_mech = r_edge + mech_margin
                sag1_m = s.sag(sd_mech)
                sag2_m = s_next.sag(sd_mech)
                edge_mech = s.thickness + sag2_m - sag1_m

                if s.is_glass:
                    residuals.append(max(0.0, min_t_glass - s.thickness) * 50.0)
                    residuals.append(max(0.0, s.thickness - self.max_glass_thickness) * 30.0 if self.enforce_manufacturing_constraints else 0.0)
                    residuals.append(max(0.0, min_t_edge - edge_t) * 50.0)
                    max_t_edge = getattr(self, 'max_edge_thickness', 20.0)
                    residuals.append(max(0.0, edge_t - max_t_edge) * 30.0 if self.enforce_manufacturing_constraints else 0.0)

                    # Mechanical rim edge thickness
                    min_mech_edge = getattr(self, 'min_mechanical_edge', 0.8)
                    residuals.append(max(0.0, min_mech_edge - edge_mech) * 45.0 if self.enforce_manufacturing_constraints else 0.0)

                    # Aspect ratio constraint
                    diam_mech = 2.0 * sd_mech
                    aspect = diam_mech / max(0.01, s.thickness)
                    max_aspect = getattr(self, 'max_aspect_ratio', 10.0)
                    residuals.append(max(0.0, aspect - max_aspect) * 20.0 if self.enforce_manufacturing_constraints else 0.0)

                    # Compactness regularization when glass thickness is variable
                    t_init_g = float(getattr(self.system.stack.surfaces[i], "thickness", 5.0)) if i < len(self.system.stack.surfaces) else 5.0
                    t_ref_max = max(t_init_g + 1.5, 9.0)
                    residuals.append(max(0.0, s.thickness - t_ref_max) * 0.08 if i in var_dict.get("thickness", []) else 0.0)
                else:
                    is_det = (s_next.surface_type == SurfaceType.DETECTOR)
                    is_obj = (s.surface_type == SurfaceType.OBJECT)
                    if is_obj:
                        residuals.append(0.0)
                        residuals.append(0.0)
                        residuals.append(0.0)
                    else:
                        req_air = self.min_bfl if is_det else self.min_air_gap
                        residuals.append(max(0.0, req_air - s.thickness) * 50.0 if self.enforce_manufacturing_constraints else 0.0)
                        req_air_edge = getattr(self, 'min_edge_air', 0.5)
                        residuals.append(max(0.0, req_air_edge - edge_t) * 35.0 if (self.enforce_manufacturing_constraints and not is_det) else 0.0)
                        residuals.append(max(0.0, req_air_edge - edge_mech) * 35.0 if (self.enforce_manufacturing_constraints and not is_det) else 0.0)

            if self.enforce_manufacturing_constraints:
                if abs(s.radius) > 1e-4:
                    req_r = 1.05 * self._effective_clear_semi_diameter(s)
                    residuals.append(max(0.0, req_r - abs(s.radius)) * 25.0)
                else:
                    residuals.append(0.0)

        # 7. Ray angle of incidence residual (RAID) (Invariant 1 scalar)
        if self.enforce_manufacturing_constraints and getattr(self, 'max_incidence_angle_deg', None):
            max_theta_rad = math.radians(float(self.max_incidence_angle_deg))
            max_inc = 0.0
            for tr in traced_rays:
                if not tr.vignetted:
                    for h in tr.history:
                        th = h.get("theta_inc", 0.0)
                        if th > max_inc:
                            max_inc = th
            residuals.append(max(0.0, max_inc - max_theta_rad) * 20.0)

        # 8. Total Track Length residual (TOTR) (Invariant 1 scalar)
        start_s = 1 if len(eval_stack.surfaces) > 1 and eval_stack.surfaces[0].surface_type == SurfaceType.OBJECT else 0
        totr_val = sum(max(0.0, float(s_elem.thickness)) for s_elem in eval_stack.surfaces[start_s:])
        max_totr = getattr(self, 'max_total_track', 500.0)
        totr_res = max(0.0, totr_val - max_totr) * 20.0 if self.enforce_manufacturing_constraints else 0.0
        residuals.append(totr_res)

        res_arr = np.array(residuals, dtype=float)
        norm_val = float(np.linalg.norm(res_arr))
        if norm_val < self.best_rms:
            self.best_rms = norm_val
            self.best_x = x_param.copy()

        return res_arr

    def _optimize_dls(
        self,
        x0_param: np.ndarray,
        bounds_param: List[Tuple[float, float]],
        var_dict: Dict[str, List[int]],
        use_curv: bool
    ) -> Tuple[np.ndarray, bool, int, str]:
        """
        Execute Damped Least Squares (Native Levenberg-Marquardt with column-normalized
        scale invariance, dynamic damping adaptation, SVD/Cholesky ill-conditioning
        protection, and step backtracking).
        """
        import scipy.linalg
        from scipy.optimize import least_squares

        lb = np.array([b[0] for b in bounds_param], dtype=float)
        ub = np.array([b[1] for b in bounds_param], dtype=float)
        x = np.clip(x0_param.copy(), lb, ub)
        n_vars = len(x)

        if n_vars == 0:
            return x, True, 0, "No variables to optimize"

        # Levenberg-Marquardt dynamic damping parameters
        lam = 1e-3
        beta_up = 2.0
        beta_down = 0.5
        lam_min = 1e-7
        lam_max = 1e8

        max_iter = max(5, self.max_iter)

        # Reset best_rms tracker in residual-norm units so it matches _compute_dls_residuals
        self.best_rms = float("inf")
        self.best_x = x.copy()

        try:
            r = self._compute_dls_residuals(x, var_dict, use_curv, bounds_param)
        except OptimizationAbortedException:
            return x, False, self.current_iter, "Optimization aborted by user."
        except Exception as e:
            return x, False, self.current_iter, f"Residual evaluation failed: {e}"

        cost = float(np.sum(r**2))
        best_cost = cost
        best_x = x.copy()

        # Parameter-dependent finite-difference step sizes for Jacobian
        dx = np.zeros(n_vars, dtype=float)
        param_types = []
        param_surf_indices = []
        for attr, idx_list in var_dict.items():
            for s_i in idx_list:
                param_types.append(attr)
                param_surf_indices.append(s_i)

        # Multi-glass chromatic compensation warm-start candidate check (e.g. doublet with uncompensated rear curvature sign)
        rad_param_positions = [j for j, pt in enumerate(param_types) if pt == "radius"]
        thk_param_positions = [j for j, pt in enumerate(param_types) if pt == "thickness"]
        if use_curv and len(rad_param_positions) >= 2 and len(self.wavelengths) > 1:
            seed_candidates = []
            r_first_pos = rad_param_positions[0]
            r_last_pos = rad_param_positions[-1]
            s_last_idx = param_surf_indices[r_last_pos]
            wl_mid = self.wavelengths[len(self.wavelengths) // 2]
            target_f = self.target_efl if (self.target_efl is not None and self.target_efl > 0) else 50.0
            target_phi = 1.0 / target_f

            for c1_trial in (0.042, 0.048, 0.054, 0.058, 0.064, 0.070):
                cand = x.copy()
                cand[r_first_pos] = float(np.clip(c1_trial, lb[r_first_pos], ub[r_first_pos]))
                for t_pos in thk_param_positions:
                    s_t_idx = param_surf_indices[t_pos]
                    if 0 <= s_t_idx < len(self.system.stack.surfaces) and self.system.stack.surfaces[s_t_idx].is_glass:
                        cand[t_pos] = float(np.clip(min(cand[t_pos], 4.5), lb[t_pos], ub[t_pos]))
                try:
                    cand[r_last_pos] = 0.0
                    st0 = copy.deepcopy(self.system.stack)
                    for jj, s_ii in enumerate(param_surf_indices):
                        val_jj = curvature_to_radius(float(cand[jj])) if param_types[jj] == "radius" else float(cand[jj])
                        setattr(st0.surfaces[s_ii], param_types[jj], val_jj)
                    efl0, _ = compute_system_efl(st0.surfaces, wavelength_um=wl_mid, eval_surface_idx=self.target_surface_index)
                    phi0 = (1.0 / efl0) if (not math.isinf(efl0) and abs(efl0) > 1e-6) else 0.0

                    c_probe = 0.02
                    st0.surfaces[s_last_idx].radius = curvature_to_radius(c_probe)
                    efl1, _ = compute_system_efl(st0.surfaces, wavelength_um=wl_mid, eval_surface_idx=self.target_surface_index)
                    phi1 = (1.0 / efl1) if (not math.isinf(efl1) and abs(efl1) > 1e-6) else phi0

                    dphi_dc = (phi1 - phi0) / c_probe
                    if abs(dphi_dc) > 1e-8:
                        c_last_exact = (target_phi - phi0) / dphi_dc
                    else:
                        c_last_exact = 0.020
                    cand[r_last_pos] = float(np.clip(c_last_exact, lb[r_last_pos], ub[r_last_pos]))

                    if thk_param_positions:
                        bfl_pos = thk_param_positions[-1]
                        s_bfl_idx = param_surf_indices[bfl_pos]
                        if s_bfl_idx + 1 < len(st0.surfaces) and st0.surfaces[s_bfl_idx + 1].surface_type == SurfaceType.DETECTOR:
                            st0.surfaces[s_last_idx].radius = curvature_to_radius(cand[r_last_pos])
                            _, bfl_exact = compute_system_efl(st0.surfaces, wavelength_um=wl_mid, eval_surface_idx=self.target_surface_index)
                            if not math.isinf(bfl_exact) and bfl_exact > 1.0:
                                cand[bfl_pos] = float(np.clip(bfl_exact - 0.35, lb[bfl_pos], ub[bfl_pos]))
                except Exception:
                    cand[r_last_pos] = float(np.clip(0.022, lb[r_last_pos], ub[r_last_pos]))
                seed_candidates.append(cand)

            for s_cand in seed_candidates:
                if self.abort:
                    break
                try:
                    r_sc = self._compute_dls_residuals(s_cand, var_dict, use_curv, bounds_param)
                    c_sc = float(np.sum(r_sc**2))
                    if c_sc < cost:
                        x = s_cand.copy()
                        r = r_sc
                        cost = c_sc
                        best_cost = c_sc
                        best_x = s_cand.copy()
                except Exception:
                    pass

        converged = False
        iters = 0
        message = "DLS Levenberg-Marquardt completed"

        for j in range(n_vars):
            ptype = param_types[j] if j < len(param_types) else "thickness"
            if ptype == "radius" and use_curv:
                dx[j] = 1e-6
            elif ptype == "thickness":
                dx[j] = 1e-4
            elif "tilt" in ptype:
                dx[j] = 1e-4
            else:
                dx[j] = 1e-4

        for it in range(max_iter):
            iters += 1
            if self.progress_callback is not None:
                self._invoke_progress(iters, max_iter)
            if self.abort:
                message = "Optimization aborted by user."
                break

            # 1. Finite-difference Jacobian J
            J = np.zeros((len(r), n_vars), dtype=float)
            for j in range(n_vars):
                h = dx[j]
                if x[j] + h <= ub[j]:
                    x_fwd = x.copy()
                    x_fwd[j] += h
                    try:
                        r_fwd = self._compute_dls_residuals(x_fwd, var_dict, use_curv, bounds_param)
                        if len(r_fwd) == len(r):
                            J[:, j] = (r_fwd - r) / h
                    except OptimizationAbortedException:
                        return best_x, False, self.current_iter, "Optimization aborted by user."
                    except Exception:
                        J[:, j] = 0.0
                else:
                    x_bwd = x.copy()
                    x_bwd[j] -= h
                    try:
                        r_bwd = self._compute_dls_residuals(x_bwd, var_dict, use_curv, bounds_param)
                        if len(r_bwd) == len(r):
                            J[:, j] = (r - r_bwd) / h
                    except OptimizationAbortedException:
                        return best_x, False, self.current_iter, "Optimization aborted by user."
                    except Exception:
                        J[:, j] = 0.0

            # Gradient vector g = J^T r
            g = J.T @ r
            if np.max(np.abs(g)) < 1e-5:
                converged = True
                message = "Gradient norm below tolerance (converged)"
                break

            # 2. Scale-invariant column-normalized normal equations:
            # Normalize each column of J by its Euclidean norm so curvature (mm^-1) and
            # thickness (mm) parameters are equally conditioned in SVD / Cholesky.
            col_norms = np.linalg.norm(J, axis=0)
            col_scale = np.maximum(col_norms, 1e-6)
            J_scaled = J / col_scale[None, :]
            g_scaled = J_scaled.T @ r
            JT_J_scaled = J_scaled.T @ J_scaled
            I_mat = np.eye(n_vars, dtype=float)

            step_accepted = False

            # Dynamic damping adaptation loop
            for damp_try in range(8):
                A_lam = JT_J_scaled + lam * I_mat

                try:
                    U, s, Vt = np.linalg.svd(A_lam)
                    s_max = max(float(s[0]), 1e-15)
                    s_min = max(float(s[-1]), 1e-15)
                    cond = s_max / s_min

                    if cond > 1e8 or s_min < 1e-8 * s_max:
                        s_reg = np.maximum(s, 1e-8 * s_max)
                        delta_scaled = - (Vt.T @ (np.diag(1.0 / s_reg) @ (U.T @ g_scaled)))
                    else:
                        try:
                            c_fac, lower = scipy.linalg.cho_factor(A_lam)
                            delta_scaled = scipy.linalg.cho_solve((c_fac, lower), -g_scaled)
                        except Exception:
                            delta_scaled = - (Vt.T @ (np.diag(1.0 / s) @ (U.T @ g_scaled)))
                except Exception:
                    delta_scaled = - np.linalg.pinv(A_lam) @ g_scaled

                delta_x = delta_scaled / col_scale

                # 3. Projected candidate and step backtracking
                cand_x = np.clip(x + delta_x, lb, ub)
                try:
                    r_cand = self._compute_dls_residuals(cand_x, var_dict, use_curv, bounds_param)
                    cost_cand = float(np.sum(r_cand**2))
                except OptimizationAbortedException:
                    return best_x, False, self.current_iter, "Optimization aborted by user."
                except Exception:
                    cost_cand = float('inf')

                if cost_cand < cost:
                    rel_impr = (cost - cost_cand) / max(1e-12, cost)
                    x = cand_x
                    r = r_cand
                    cost = cost_cand
                    lam = max(lam_min, lam * beta_down)
                    step_accepted = True
                    if cost < best_cost:
                        best_cost = cost
                        best_x = x.copy()
                    if rel_impr < 1e-6 and it >= 3:
                        converged = True
                        message = "Relative cost improvement below tolerance (converged)"
                    break
                else:
                    bt_accepted = False
                    for shrink in [0.5, 0.25, 0.1]:
                        cand_bt = np.clip(x + shrink * delta_x, lb, ub)
                        try:
                            r_bt = self._compute_dls_residuals(cand_bt, var_dict, use_curv, bounds_param)
                            cost_bt = float(np.sum(r_bt**2))
                        except OptimizationAbortedException:
                            return best_x, False, self.current_iter, "Optimization aborted by user."
                        except Exception:
                            cost_bt = float('inf')

                        if cost_bt < cost:
                            x = cand_bt
                            r = r_bt
                            cost = cost_bt
                            lam = max(lam_min, lam * beta_down)
                            bt_accepted = True
                            step_accepted = True
                            if cost < best_cost:
                                best_cost = cost
                                best_x = x.copy()
                            break

                    if bt_accepted:
                        break
                    else:
                        lam = min(lam_max, lam * beta_up)

            if converged:
                break
            if not step_accepted and lam >= lam_max:
                message = "Maximum damping reached"
                break

        success = bool(converged or (best_cost < 1e5))
        return best_x, success, iters, message

    def _optimize_hammer(
        self,
        x0_param: np.ndarray,
        bounds_param: List[Tuple[float, float]],
        var_dict: Dict[str, List[int]],
        use_curv: bool
    ) -> Tuple[np.ndarray, bool, int, str]:
        """
        Zemax Hammer Optimizer (Global):
        Stochastic perturbations (Gaussian + Cauchy jumps) + DLS local polish + simulated annealing cooling.
        """
        from scipy.optimize import least_squares
        lb = np.array([b[0] for b in bounds_param], dtype=float)
        ub = np.array([b[1] for b in bounds_param], dtype=float)
        spans = np.maximum(1e-4, ub - lb)

        curr_x = np.clip(x0_param.copy(), lb, ub)
        r_init = self._compute_dls_residuals(curr_x, var_dict, use_curv, bounds_param)
        curr_cost = float(np.sum(r_init**2))
        best_x = curr_x.copy()
        best_cost = curr_cost

        n_cycles = max(3, min(20, self.max_iter // 5))
        nfev_per_local = max(10, min(30, (self.max_iter * 20) // n_cycles))
        total_evals = 0

        rng = np.random.RandomState(42)

        param_types = []
        for attr, idx_list in var_dict.items():
            for _ in idx_list:
                param_types.append(attr)

        for cycle in range(n_cycles):
            if self.abort:
                break

            temp = (0.85 ** cycle)
            delta = np.zeros_like(curr_x)
            for j in range(len(curr_x)):
                sigma_j = 0.08 * spans[j] * temp
                ptype = param_types[j] if j < len(param_types) else "thickness"
                p_rand = rng.rand()
                if ptype == "radius" and p_rand < 0.10:
                    # Curvature sign flip exploring alternative bending topologies
                    delta[j] = -2.0 * curr_x[j]
                elif p_rand < 0.80:
                    delta[j] = rng.normal(0.0, sigma_j)
                else:
                    delta[j] = rng.standard_cauchy() * (0.5 * sigma_j)

            cand_x = np.clip(curr_x + delta, lb, ub)

            try:
                res = least_squares(
                    fun=lambda x: self._compute_dls_residuals(x, var_dict, use_curv, bounds_param),
                    x0=cand_x,
                    bounds=(lb, ub),
                    method="trf",
                    ftol=1e-6,
                    xtol=1e-6,
                    gtol=1e-6,
                    max_nfev=nfev_per_local
                )
                cand_cost = float(res.cost * 2.0)
                cand_x_polished = res.x
                total_evals += res.nfev
            except OptimizationAbortedException:
                break
            except Exception:
                continue

            dE = cand_cost - curr_cost
            accept_prob = np.exp(-dE / max(1e-5, temp * curr_cost)) if dE > 0 else 1.0
            if dE < 0 or rng.rand() < accept_prob:
                curr_x = cand_x_polished.copy()
                curr_cost = cand_cost

            if cand_cost < best_cost:
                best_cost = cand_cost
                best_x = cand_x_polished.copy()

        # Final fine polish on best
        try:
            res_fine = least_squares(
                fun=lambda x: self._compute_dls_residuals(x, var_dict, use_curv, bounds_param),
                x0=best_x,
                bounds=(lb, ub),
                method="trf",
                ftol=1e-8,
                xtol=1e-8,
                gtol=1e-8,
                max_nfev=max(15, self.max_iter * 5)
            )
            best_x = res_fine.x
            total_evals += res_fine.nfev
        except Exception:
            pass

        return best_x, True, total_evals, "Hammer optimization completed successfully"

    def _optimize_global_search(
        self,
        x0_param: np.ndarray,
        bounds_param: List[Tuple[float, float]],
        var_dict: Dict[str, List[int]],
        use_curv: bool
    ) -> Tuple[np.ndarray, bool, int, str]:
        """
        Zemax Global Search (Multi-Start):
        Explores discrete curvature sign permutations and space-filling candidates, polished via DLS.
        """
        from scipy.optimize import least_squares
        lb = np.array([b[0] for b in bounds_param], dtype=float)
        ub = np.array([b[1] for b in bounds_param], dtype=float)

        candidates = [np.clip(x0_param.copy(), lb, ub)]

        # 1. Curvature sign flips for radius variables
        rad_indices = []
        param_idx = 0
        for attr, idx_list in var_dict.items():
            for _ in idx_list:
                if attr == "radius":
                    rad_indices.append(param_idx)
                param_idx += 1

        if rad_indices:
            n_rads = min(4, len(rad_indices))
            for pattern in range(1, 2**n_rads):
                cand = x0_param.copy()
                for bit in range(n_rads):
                    if (pattern >> bit) & 1:
                        p_i = rad_indices[bit]
                        cand[p_i] = -cand[p_i]
                candidates.append(np.clip(cand, lb, ub))

        # 2. Space-filling random samples
        rng = np.random.RandomState(123)
        n_samples = max(10, min(30, self.max_iter // 3))
        for _ in range(n_samples):
            rnd = lb + rng.rand(len(lb)) * (ub - lb)
            candidates.append(rnd)

        # 3. Coarse screening
        scored_candidates = []
        for cand in candidates:
            if self.abort:
                break
            try:
                r = self._compute_dls_residuals(cand, var_dict, use_curv, bounds_param)
                cost = float(np.sum(r**2))
                scored_candidates.append((cost, cand))
            except Exception:
                pass

        if not scored_candidates:
            return x0_param, False, 0, "No valid candidate found in global search"

        scored_candidates.sort(key=lambda item: item[0])

        top_k = min(4, len(scored_candidates))
        best_x = scored_candidates[0][1]
        best_cost = scored_candidates[0][0]
        total_evals = len(scored_candidates)

        nfev_each = max(15, (self.max_iter * 20) // top_k)
        for _, cand in scored_candidates[:top_k]:
            if self.abort:
                break
            try:
                res = least_squares(
                    fun=lambda x: self._compute_dls_residuals(x, var_dict, use_curv, bounds_param),
                    x0=cand,
                    bounds=(lb, ub),
                    method="trf",
                    ftol=1e-7,
                    xtol=1e-7,
                    gtol=1e-7,
                    max_nfev=nfev_each
                )
                total_evals += res.nfev
                if res.cost * 2.0 < best_cost:
                    best_cost = res.cost * 2.0
                    best_x = res.x
            except Exception:
                continue

        return best_x, True, total_evals, "Global search completed successfully"

    def _optimize_monte_carlo(
        self,
        x0_param: np.ndarray,
        bounds_param: List[Tuple[float, float]],
        var_dict: Dict[str, List[int]],
        use_curv: bool,
        num_trials: Optional[int] = None,
        polish_best: bool = True
    ) -> Tuple[np.ndarray, bool, int, str]:
        """
        Zemax-style Monte Carlo Global Optimization:
        Stochastically explores the multi-dimensional parameter space using randomized
        Monte Carlo sampling with adaptive Gaussian & uniform perturbations, followed by
        a Damped Least Squares (DLS) gradient polish on the best-performing candidates.
        """
        from scipy.optimize import least_squares
        lb = np.array([b[0] for b in bounds_param], dtype=float)
        ub = np.array([b[1] for b in bounds_param], dtype=float)
        spans = np.maximum(1e-4, ub - lb)

        trials_count = num_trials if num_trials is not None else max(20, min(200, self.max_iter * 2))

        curr_x = np.clip(x0_param.copy(), lb, ub)
        r_init = self._compute_dls_residuals(curr_x, var_dict, use_curv, bounds_param)
        best_cost = float(np.sum(r_init**2))
        best_x = curr_x.copy()

        rng = np.random.RandomState(42)
        total_evals = 1

        top_candidates = [(best_cost, best_x.copy())]

        for i in range(trials_count):
            if self.abort:
                break

            # 50% Latin-hypercube/uniform space-filling, 50% local Gaussian perturbation around best
            if rng.rand() < 0.5:
                cand = lb + rng.rand(len(lb)) * spans
            else:
                sigma = 0.15 * spans
                cand = np.clip(best_x + rng.normal(0.0, sigma), lb, ub)

            try:
                residuals = self._compute_dls_residuals(cand, var_dict, use_curv, bounds_param)
                cost = float(np.sum(residuals**2))
                total_evals += 1

                if cost < best_cost:
                    best_cost = cost
                    best_x = cand.copy()

                top_candidates.append((cost, cand.copy()))
            except OptimizationAbortedException:
                break
            except Exception:
                continue

            if self.progress_callback is not None and (i % max(1, trials_count // 20) == 0):
                self._invoke_progress(i, trials_count)

        # Polish the top-performing candidate(s) with DLS to achieve quadratic convergence
        if polish_best and not self.abort and top_candidates:
            top_candidates.sort(key=lambda item: item[0])
            top_to_polish = top_candidates[:min(3, len(top_candidates))]

            for _, cand_to_polish in top_to_polish:
                if self.abort:
                    break
                try:
                    res_polish = least_squares(
                        fun=lambda x: self._compute_dls_residuals(x, var_dict, use_curv, bounds_param),
                        x0=cand_to_polish,
                        bounds=(lb, ub),
                        method="trf",
                        ftol=1e-8,
                        xtol=1e-8,
                        gtol=1e-8,
                        max_nfev=max(15, self.max_iter * 5)
                    )
                    total_evals += res_polish.nfev
                    cand_polished_cost = float(res_polish.cost * 2.0)
                    if cand_polished_cost < best_cost:
                        best_cost = cand_polished_cost
                        best_x = res_polish.x
                except Exception:
                    continue

        success = not self.abort and (best_cost < 1e5)
        msg = "Monte Carlo optimization completed successfully" if success else "Monte Carlo optimization terminated"
        return best_x, success, total_evals, msg

    def optimize(
        self,
        variable_indices: Optional[Union[dict, List[int]]] = None,
        bounds: Optional[List[Tuple[float, float]]] = None
    ) -> OptimizationResult:
        """
        Execute optimization using configured algorithm (DLS, Hammer, Global Search, Nelder-Mead, L-BFGS-B, etc.).
        Returns OptimizationResult with updated SurfaceStack and comprehensive convergence diagnostics.
        """
        # Run pre-flight validation check
        validation = self.validate(variable_indices, bounds)

        # If abort was requested prior to starting optimization, return immediately with USER_ABORTED
        if self.abort:
            self.abort = False
            init_metrics = self._compute_detailed_spot_metrics(self.system.stack)
            init_pure, init_pen = self._evaluate_merit_and_penalty(np.array([]), variable_indices or {}, check_abort=False)
            per_wavelength_metrics = {
                wl: WavelengthSpotMetric(
                    wavelength=wl,
                    initial_rms=float(m["rms_radius"]),
                    final_rms=float(m["rms_radius"]),
                    initial_geo=float(m["geo_radius"]),
                    final_geo=float(m["geo_radius"]),
                    centroid_x=float(m["centroid_x"]),
                    centroid_y=float(m["centroid_y"]),
                    valid_rays=int(m["valid_rays"]),
                    total_rays=int(m["total_rays"]),
                    airy_radius=float(m["airy_radius"])
                ) for wl, m in init_metrics.items()
            }
            return OptimizationResult(
                success=False,
                initial_rms=init_pure,
                final_rms=init_pure,
                rms_reduction_pct=0.0,
                optimized_thicknesses={},
                optimized_radii={},
                optimized_tilts={},
                optimized_decenters={},
                optimized_stack=self.system.stack,
                iterations=0,
                message="Optimization aborted by user.",
                status=TerminationStatus.USER_ABORTED,
                penalty_value=init_pen,
                per_wavelength_metrics=per_wavelength_metrics,
                validation_result=validation,
                history=[float(init_pure)]
            )

        # Reset state at beginning of optimization
        self.best_rms = float('inf')
        self.best_x = None
        self.current_iter = 0

        # 1. Discover and normalize variable indices
        if variable_indices is None:
            var_dict = {
                "thickness": [], "radius": [],
                "tilt_x": [], "tilt_y": [], "tilt_z": [],
                "decenter_x": [], "decenter_y": [], "decenter_z": []
            }
            for i, s in enumerate(self.system.stack.surfaces):
                if getattr(s, 'thickness_variable', False):
                    var_dict["thickness"].append(i)
                if getattr(s, 'radius_variable', False):
                    var_dict["radius"].append(i)
                if getattr(s, 'tilt_x_variable', False):
                    var_dict["tilt_x"].append(i)
                if getattr(s, 'tilt_y_variable', False):
                    var_dict["tilt_y"].append(i)
                if getattr(s, 'tilt_z_variable', False):
                    var_dict["tilt_z"].append(i)
                if getattr(s, 'decenter_x_variable', False):
                    var_dict["decenter_x"].append(i)
                if getattr(s, 'decenter_y_variable', False):
                    var_dict["decenter_y"].append(i)
                if getattr(s, 'decenter_z_variable', False):
                    var_dict["decenter_z"].append(i)
            var_dict = {k: v for k, v in var_dict.items() if len(v) > 0}
        elif isinstance(variable_indices, (list, tuple, np.ndarray)):
            var_dict = {"thickness": list(variable_indices)}
        elif isinstance(variable_indices, dict):
            var_dict = {k: list(v) for k, v in variable_indices.items() if len(v) > 0}
        else:
            var_dict = {"thickness": list(variable_indices)}

        # Global First-Order Optical Bootstrapping
        self._seed_paraxial_efl_if_stalled(var_dict)

        # 2. Compute initial detailed metrics
        init_metrics = self._compute_detailed_spot_metrics(self.system.stack)

        # Handle empty variables (no-op)
        has_vars = any(len(v) > 0 for v in var_dict.values()) or self.opt_tilt
        if not has_vars:
            init_pure, init_pen = self._evaluate_merit_and_penalty(np.array([]), var_dict)
            per_wavelength_metrics = {
                wl: WavelengthSpotMetric(
                    wavelength=wl,
                    initial_rms=float(m["rms_radius"]),
                    final_rms=float(m["rms_radius"]),
                    initial_geo=float(m["geo_radius"]),
                    final_geo=float(m["geo_radius"]),
                    centroid_x=float(m["centroid_x"]),
                    centroid_y=float(m["centroid_y"]),
                    valid_rays=int(m["valid_rays"]),
                    total_rays=int(m["total_rays"]),
                    airy_radius=float(m["airy_radius"])
                ) for wl, m in init_metrics.items()
            }
            return OptimizationResult(
                success=True,
                initial_rms=init_pure,
                final_rms=init_pure,
                rms_reduction_pct=0.0,
                optimized_thicknesses={},
                optimized_radii={},
                optimized_tilts={},
                optimized_decenters={},
                optimized_stack=self.system.stack,
                iterations=0,
                message="No variable surfaces specified",
                status=TerminationStatus.NO_VARIABLES,
                penalty_value=init_pen,
                per_wavelength_metrics=per_wavelength_metrics,
                validation_result=validation,
                history=[float(init_pure)]
            )

        # Determine method routing & curvature space usage
        method_norm = str(self.method).strip().lower()
        is_dls = any(k in method_norm for k in ["dls", "damped least squares", "levenberg-marquardt", "least_squares"])
        is_hammer = any(k in method_norm for k in ["hammer"])
        is_global = any(k in method_norm for k in ["global search", "multi-start"]) or (method_norm == "global")
        is_monte_carlo = any(k in method_norm for k in ["monte carlo", "monte_carlo", "mc"])

        use_curv = (
            self.use_curvature_space if self.use_curvature_space is not None
            else (is_dls or is_hammer or is_global or is_monte_carlo)
        )

        det_idx = None
        for idx, s in enumerate(self.system.stack.surfaces):
            if s.surface_type == SurfaceType.DETECTOR:
                det_idx = idx
                break
        if det_idx is None:
            det_idx = len(self.system.stack.surfaces) - 1

        # 3. Build initial parameter vector x0 and bounds
        x_list = []
        bounds_computed = []
        bound_idx = 0
        for attr, indices in var_dict.items():
            for idx in indices:
                s = self.system.stack.surfaces[idx]
                user_bound = bounds[bound_idx] if (bounds is not None and bound_idx < len(bounds)) else None
                bound_idx += 1

                if attr == "radius":
                    r_val = float(getattr(s, 'radius', 0.0))
                    if use_curv:
                        y_sd = self._effective_clear_semi_diameter(s)
                        if y_sd <= 1e-4:
                            y_sd = 10.0
                        c_dyn_max = 0.95 / y_sd

                        c_val = radius_to_curvature(r_val)
                        c_val = float(np.clip(c_val, -c_dyn_max, c_dyn_max))
                        x_list.append(c_val)
                        if user_bound is not None:
                            r_bmin, r_bmax = user_bound
                            if r_bmin > 0 and r_bmax > r_bmin:
                                cb_min, cb_max = (1.0 / r_bmax, 1.0 / max(1.0, r_bmin))
                            elif r_bmax < 0 and r_bmin < r_bmax:
                                cb_min, cb_max = (1.0 / min(-1.0, r_bmax), 1.0 / r_bmin)
                            else:
                                cb_min, cb_max = (-0.25, 0.25)
                        else:
                            r_min = float(getattr(s, 'radius_min', -1000.0))
                            r_max = float(getattr(s, 'radius_max', 1000.0))
                            if r_min > 0 and r_max > r_min:
                                cb_min, cb_max = (1.0 / r_max, 1.0 / max(1.0, r_min))
                            elif r_max < 0 and r_min < r_max:
                                cb_min, cb_max = (1.0 / min(-1.0, r_max), 1.0 / r_min)
                            else:
                                cb_min, cb_max = (-0.25, 0.25)

                        cb_min = max(cb_min, -c_dyn_max)
                        cb_max = min(cb_max, c_dyn_max)
                        if cb_min > cb_max:
                            cb_min, cb_max = -c_dyn_max, c_dyn_max
                        bounds_computed.append((cb_min, cb_max))
                    else:
                        x_list.append(r_val)
                        if user_bound is not None:
                            bounds_computed.append(user_bound)
                        else:
                            bounds_computed.append((float(getattr(s, 'radius_min', -1000.0)), float(getattr(s, 'radius_max', 1000.0))))
                elif attr == "thickness":
                    t_val = float(getattr(s, 'thickness', 0.0))
                    x_list.append(t_val)
                    if user_bound is not None:
                        bounds_computed.append(user_bound)
                    else:
                        b_min = float(getattr(s, 'thickness_min', 0.0))
                        b_max = float(getattr(s, 'thickness_max', 0.0))
                        if self.enforce_manufacturing_constraints:
                            if s.is_glass:
                                b_min = max(b_min, self.min_glass_thickness) if b_min > 0.0 else self.min_glass_thickness
                                b_max = min(b_max, self.max_glass_thickness) if (0.0 < b_max <= 500.0) else self.max_glass_thickness
                            else:
                                is_det = (idx == len(self.system.stack.surfaces) - 2) or (idx + 1 < len(self.system.stack.surfaces) and self.system.stack.surfaces[idx + 1].surface_type == SurfaceType.DETECTOR)
                                if is_det:
                                    b_min = max(b_min, self.min_bfl) if b_min > 0.0 else self.min_bfl
                                    b_max = max(b_max, 250.0) if b_max > b_min else 250.0
                                else:
                                    b_min = max(b_min, self.min_air_gap) if b_min > 0.0 else self.min_air_gap
                                    b_max = min(b_max, self.max_air_gap) if (0.0 < b_max <= 100.0) else self.max_air_gap
                        else:
                            if b_min <= 0.0:
                                b_min = 1.0 if s.is_glass else 0.1
                            if b_max <= b_min:
                                if s.is_glass:
                                    # Sensible default glass limit: 18 mm or max 2 * semi_diameter
                                    sd = getattr(s, 'semi_diameter', 12.5) or 12.5
                                    b_max = max(12.0, min(25.0, sd * 2.0))
                                else:
                                    is_det = (idx == len(self.system.stack.surfaces) - 2) or (idx + 1 < len(self.system.stack.surfaces) and self.system.stack.surfaces[idx + 1].surface_type == SurfaceType.DETECTOR)
                                    b_max = max(b_min + 10.0, 250.0 if is_det else 60.0)
                        bounds_computed.append((b_min, b_max))

                elif attr in ("tilt_x", "tilt_y", "tilt_z"):
                    x_list.append(float(getattr(s, attr, 0.0)))
                    if user_bound is not None:
                        bounds_computed.append(user_bound)
                    else:
                        bounds_computed.append((float(getattr(s, f'{attr}_min', -45.0)), float(getattr(s, f'{attr}_max', 45.0))))
                elif attr in ("decenter_x", "decenter_y", "decenter_z"):
                    x_list.append(float(getattr(s, attr, 0.0)))
                    if user_bound is not None:
                        bounds_computed.append(user_bound)
                    else:
                        bounds_computed.append((float(getattr(s, f'{attr}_min', -25.0)), float(getattr(s, f'{attr}_max', 25.0))))
                else:
                    x_list.append(float(getattr(s, attr, 0.0)))
                    if user_bound is not None:
                        bounds_computed.append(user_bound)
                    else:
                        bounds_computed.append((-1000.0, 1000.0))

        if self.opt_tilt and "tilt_x" not in var_dict:
            x_list.append(float(getattr(self.system.stack.surfaces[det_idx], 'tilt_x', 0.0)))
            user_bound = bounds[bound_idx] if (bounds is not None and bound_idx < len(bounds)) else None
            bounds_computed.append(user_bound if user_bound is not None else (-45.0, 45.0))

        x0 = np.array(x_list, dtype=float)
        active_bounds = bounds_computed
        lb = np.array([b[0] for b in active_bounds], dtype=float)
        ub = np.array([b[1] for b in active_bounds], dtype=float)
        x0 = np.clip(x0, lb, ub)

        # Compute initial merit with physical values
        x0_phys = []
        p_i = 0
        for attr, indices in var_dict.items():
            for _ in indices:
                if p_i < len(x0):
                    val = float(x0[p_i])
                    x0_phys.append(curvature_to_radius(val) if (attr == "radius" and use_curv) else val)
                p_i += 1
        if self.opt_tilt and p_i < len(x0):
            x0_phys.append(float(x0[p_i]))
        x0_phys_arr = np.array(x0_phys, dtype=float)

        init_pure_rms, init_penalty = self._evaluate_merit_and_penalty(x0_phys_arr, var_dict, check_abort=False)
        initial_rms = init_pure_rms
        self.best_rms = init_pure_rms + init_penalty
        self.best_x = x0.copy()
        merit_history = [float(initial_rms)]

        aborted = False
        iterations = 0
        message = ""
        success = False

        # 4. Dispatch to solver
        if is_dls:
            x_opt, success, iterations, message = self._optimize_dls(x0, active_bounds, var_dict, use_curv)
            if self.abort:
                aborted = True
        elif is_hammer:
            x_opt, success, iterations, message = self._optimize_hammer(x0, active_bounds, var_dict, use_curv)
            if self.abort:
                aborted = True
        elif is_global:
            x_opt, success, iterations, message = self._optimize_global_search(x0, active_bounds, var_dict, use_curv)
            if self.abort:
                aborted = True
        elif is_monte_carlo:
            x_opt, success, iterations, message = self._optimize_monte_carlo(x0, active_bounds, var_dict, use_curv)
            if self.abort:
                aborted = True
        else:
            # Scipy minimize (Nelder-Mead, L-BFGS-B, SLSQP, etc.)
            def objective(x):
                if self.abort:
                    raise OptimizationAbortedException("Optimization aborted by user.")

                penalty = 0.0
                for val, (b_min, b_max) in zip(x, active_bounds):
                    span = max(1e-4, b_max - b_min)
                    if val < b_min:
                        dev = (b_min - val) / span
                        penalty += 5000.0 * (dev + dev**2)
                    elif val > b_max:
                        dev = (val - b_max) / span
                        penalty += 5000.0 * (dev + dev**2)

                val_eval = np.clip(x, [b[0] for b in active_bounds], [b[1] for b in active_bounds])
                pure_m, pen_m = self._evaluate_merit_and_penalty(val_eval, var_dict, check_abort=True, use_curvature_space=use_curv)
                total = pure_m + pen_m + penalty
                if total < self.best_rms:
                    self.best_rms = total
                    self.best_x = val_eval.copy()
                return total

            def _callback(xk):
                self.current_iter += 1
                if self.progress_callback is not None:
                    self._invoke_progress(self.current_iter, self.max_iter)
                if self.abort:
                    raise OptimizationAbortedException("Optimization aborted by user.")

            use_bounds = active_bounds if self.method in ["L-BFGS-B", "SLSQP", "trust-constr"] else None
            solver_options = {"maxiter": self.max_iter}
            if self.method == "Nelder-Mead":
                solver_options.update({
                    "xatol": 1e-4,
                    "fatol": 1e-5,
                    "maxfev": self.max_iter * 50,
                    "adaptive": len(x0) > 2
                })
            elif self.method == "L-BFGS-B":
                solver_options.update({
                    "ftol": 1e-6,
                    "gtol": 1e-5,
                    "eps": 1e-3,
                    "maxfun": self.max_iter * 50
                })

            x_opt = x0.copy()
            try:
                res = minimize(
                    objective,
                    x0,
                    method=self.method,
                    bounds=use_bounds,
                    options=solver_options,
                    callback=_callback
                )
                if self.best_x is not None and (not hasattr(res, 'fun') or self.best_rms <= res.fun):
                    x_opt = self.best_x.copy()
                elif hasattr(res, 'x'):
                    x_opt = res.x.copy()

                if self.abort:
                    aborted = True
                    message = "Optimization aborted by user."
                    success = False
                    iterations = self.current_iter
                else:
                    message = "Optimization completed successfully"
                    success = bool(res.success) or (self.best_rms < (init_pure_rms + init_penalty))
                    iterations = res.nit if hasattr(res, 'nit') else self.current_iter
            except OptimizationAbortedException:
                aborted = True
                x_opt = self.best_x.copy() if self.best_x is not None else x0.copy()
                message = "Optimization aborted by user."
                success = False
                iterations = self.current_iter
            except Exception as e:
                message = f"Optimization terminated: {e}"
                success = False
                iterations = self.current_iter

        # 5. Clip optimal candidate within active bounds
        for i, (b_min, b_max) in enumerate(active_bounds):
            if i < len(x_opt):
                x_opt[i] = np.clip(x_opt[i], b_min, b_max)

        # 6. Post-process optimal candidate into physical space (converting curvatures to radii)
        x_phys_final = []
        p_i = 0
        bfl_phys_pos = None
        bfl_bounds = None
        pre_det_surf_idx = (det_idx - 1) if (det_idx is not None and det_idx > 0) else -1
        for attr, indices in var_dict.items():
            for s_i in indices:
                if p_i < len(x_opt):
                    val = float(x_opt[p_i])
                    x_phys_final.append(curvature_to_radius(val) if (attr == "radius" and use_curv) else val)
                    if attr == "thickness" and s_i == pre_det_surf_idx:
                        bfl_phys_pos = p_i
                        bfl_bounds = active_bounds[p_i] if p_i < len(active_bounds) else (0.5, 250.0)
                p_i += 1
        if self.opt_tilt and p_i < len(x_opt):
            x_phys_final.append(float(x_opt[p_i]))
        x_opt_phys = np.array(x_phys_final, dtype=float)

        # 6b. Exact polychromatic best-focus BFL lock when pre-detector air space is variable
        if (
            not aborted
            and bfl_phys_pos is not None
            and bfl_bounds is not None
            and any(k in str(self.target) for k in ("Spot", "Focus", "Achromat", "Wavefront"))
        ):
            try:
                from scipy.optimize import minimize_scalar
                cur_bfl = float(x_opt_phys[bfl_phys_pos])
                b_lo, b_hi = float(bfl_bounds[0]), float(bfl_bounds[1])

                # Also check paraxial BFL of current lens prescription
                temp_stack = self._clone_stack(self.system.stack)
                v_i = 0
                for attr, indices in var_dict.items():
                    for s_i in indices:
                        if 0 <= s_i < len(temp_stack.surfaces) and v_i < len(x_opt_phys):
                            setattr(temp_stack.surfaces[s_i], attr, float(x_opt_phys[v_i]))
                        v_i += 1
                wl_mid = self.wavelengths[len(self.wavelengths) // 2] if self.wavelengths else 0.5876
                _, parax_bfl = compute_system_efl(temp_stack.surfaces, wavelength_um=wl_mid, eval_surface_idx=det_idx)

                center_bfl = cur_bfl
                if not math.isinf(parax_bfl) and b_lo <= parax_bfl <= b_hi:
                    cand_parax = x_opt_phys.copy()
                    cand_parax[bfl_phys_pos] = float(parax_bfl)
                    m_cur, p_cur = self._evaluate_merit_and_penalty(x_opt_phys, var_dict, check_abort=False, use_curvature_space=False)
                    m_par, p_par = self._evaluate_merit_and_penalty(cand_parax, var_dict, check_abort=False, use_curvature_space=False)
                    if (m_par + p_par) < (m_cur + p_cur):
                        x_opt_phys = cand_parax
                        center_bfl = float(parax_bfl)

                win_lo = max(b_lo, center_bfl - 6.0)
                win_hi = min(b_hi, center_bfl + 6.0)
                if win_hi > win_lo + 1e-3:
                    best_1d_val = float(x_opt_phys[bfl_phys_pos])
                    best_1d_merit, best_1d_pen = self._evaluate_merit_and_penalty(x_opt_phys, var_dict, check_abort=False, use_curvature_space=False)
                    best_1d_tot = best_1d_merit + best_1d_pen

                    def _bfl_obj(t_val: float) -> float:
                        c_vec = x_opt_phys.copy()
                        c_vec[bfl_phys_pos] = float(t_val)
                        pm, pp = self._evaluate_merit_and_penalty(c_vec, var_dict, check_abort=False, use_curvature_space=False)
                        return pm + pp

                    res_1d = minimize_scalar(_bfl_obj, bounds=(win_lo, win_hi), method="bounded", options={"xatol": 1e-3, "maxiter": 18})
                    if hasattr(res_1d, "fun") and float(res_1d.fun) < best_1d_tot:
                        x_opt_phys[bfl_phys_pos] = float(np.clip(res_1d.x, b_lo, b_hi))
            except Exception:
                pass

        final_pure_rms, final_penalty = self._evaluate_merit_and_penalty(x_opt_phys, var_dict, check_abort=False, use_curvature_space=False)
        final_rms = final_pure_rms
        penalty_value = final_penalty

        reduction_pct = ((initial_rms - final_rms) / initial_rms * 100.0) if initial_rms > 1e-12 else 0.0
        reduction_pct = max(0.0, reduction_pct)

        # 7. Build final optimized stack with physical values
        opt_stack = self._clone_stack(self.system.stack)
        opt_thicknesses = {}
        opt_radii = {}
        opt_tilts = {}
        opt_decenters = {}

        var_idx = 0
        for attr, indices in var_dict.items():
            for s_idx in indices:
                if 0 <= s_idx < len(opt_stack.surfaces) and var_idx < len(x_opt_phys):
                    val_to_set = float(x_opt_phys[var_idx])
                    setattr(opt_stack.surfaces[s_idx], attr, val_to_set)
                    if attr == "thickness":
                        opt_thicknesses[s_idx] = val_to_set
                    elif attr == "radius":
                        opt_radii[s_idx] = val_to_set
                    elif attr in ("tilt_x", "tilt_y", "tilt_z"):
                        s = opt_stack.surfaces[s_idx]
                        opt_tilts[s_idx] = (float(s.tilt_x), float(s.tilt_y), float(s.tilt_z))
                    elif attr in ("decenter_x", "decenter_y"):
                        s = opt_stack.surfaces[s_idx]
                        opt_decenters[s_idx] = (float(s.decenter_x), float(s.decenter_y))
                var_idx += 1

        if self.opt_tilt and var_idx < len(x_opt_phys) and (0 <= det_idx < len(opt_stack.surfaces)):
            tilt_val = float(x_opt_phys[var_idx])
            opt_stack.surfaces[det_idx].tilt_x = tilt_val
            opt_tilts[det_idx] = (tilt_val, float(opt_stack.surfaces[det_idx].tilt_y), float(opt_stack.surfaces[det_idx].tilt_z))

        opt_stack._update_accumulated_positions()

        # 8. Compute final detailed per-wavelength metrics
        final_metrics = self._compute_detailed_spot_metrics(opt_stack)
        per_wavelength_metrics: Dict[float, WavelengthSpotMetric] = {}
        for wl in self.wavelengths:
            im = init_metrics.get(wl, {})
            fm = final_metrics.get(wl, {})
            per_wavelength_metrics[wl] = WavelengthSpotMetric(
                wavelength=wl,
                initial_rms=float(im.get("rms_radius", 0.0)),
                final_rms=float(fm.get("rms_radius", 0.0)),
                initial_geo=float(im.get("geo_radius", 0.0)),
                final_geo=float(fm.get("geo_radius", 0.0)),
                centroid_x=float(fm.get("centroid_x", 0.0)),
                centroid_y=float(fm.get("centroid_y", 0.0)),
                valid_rays=int(fm.get("valid_rays", 0)),
                total_rays=int(fm.get("total_rays", 0)),
                airy_radius=float(fm.get("airy_radius", 0.0))
            )

        # 9. Determine structured termination status
        if aborted or self.abort:
            self.abort = False
            status = TerminationStatus.USER_ABORTED
            message = "Optimization aborted by user."
            success = False
        elif not success and (final_rms >= 1e4 or final_penalty >= 1e5):
            status = TerminationStatus.RAY_TRACE_FAILURE
            message = "Ray tracing failed or complete vignetting during optimization"
        elif iterations >= self.max_iter or "maximum number of iterations" in message.lower() or "maxfev" in message.lower():
            status = TerminationStatus.MAX_ITERATIONS_REACHED
            success = True
        else:
            all_at_bounds = False
            if len(active_bounds) > 0 and len(x_opt) == len(active_bounds):
                all_at_bounds = all(abs(x_opt[i] - b[0]) < 1e-5 or abs(x_opt[i] - b[1]) < 1e-5 for i, b in enumerate(active_bounds))
            if all_at_bounds and len(active_bounds) > 0:
                status = TerminationStatus.BOUND_LIMITED
            else:
                status = TerminationStatus.CONVERGED

        # Compute complete manufacturing & mounting audit report
        mfg_report = None
        try:
            if audit_optical_system_manufacturing is not None:
                max_angles = {}
                eval_sys = OpticalSystem(opt_stack)
                audit_rays = self._generate_rays(self.entrance_pupil_radius, self.wavelengths, min(31, self.num_rays), eval_stack=opt_stack)
                tr_aud = eval_sys.trace_rays(audit_rays)
                for tr in tr_aud:
                    if not tr.vignetted:
                        for h in tr.history:
                            s_i = h.get("surface_index")
                            if s_i is not None:
                                th_deg = math.degrees(h.get("theta_inc", 0.0))
                                max_angles[s_i] = max(max_angles.get(s_i, 0.0), th_deg)
                mfg_report = audit_optical_system_manufacturing(
                    opt_stack.surfaces,
                    config=getattr(self, 'mfg_config', None),
                    ray_incidence_angles=max_angles
                )
        except Exception:
            mfg_report = None

        if abs(final_rms - initial_rms) > 1e-9 or len(merit_history) == 1:
            merit_history.append(float(final_rms))

        return OptimizationResult(
            success=success,
            initial_rms=initial_rms,
            final_rms=final_rms,
            rms_reduction_pct=reduction_pct,
            optimized_thicknesses=opt_thicknesses,
            optimized_radii=opt_radii,
            optimized_tilts=opt_tilts,
            optimized_decenters=opt_decenters,
            optimized_stack=opt_stack,
            iterations=iterations,
            message=message,
            status=status,
            penalty_value=penalty_value,
            per_wavelength_metrics=per_wavelength_metrics,
            validation_result=validation,
            manufacturing_report=mfg_report,
            history=merit_history
        )

    def fit_to_test_plates(
        self,
        re_optimize_bfl: bool = True,
        max_fringe_error: float = 5.0
    ) -> OptimizationResult:
        """
        Zemax Standard Test Plate Fitting (TFLT):
        Snaps all spherical surface radii in the system to the nearest standard tooling test plates
        from the ISO/Thorlabs/Edmund catalog. If re_optimize_bfl is True, runs a fine
        optimization on the air gaps / back focal distance to restore diffraction-limited performance.
        """
        stack_snapped = self._clone_stack(self.system.stack)
        for s in stack_snapped.surfaces:
            if abs(s.radius) > 1e-4:
                tool_r, _, fringes = TestPlateCatalog.find_nearest_test_plate(s.radius, s.semi_diameter)
                s.radius = tool_r

        new_sys = OpticalSystem(stack_snapped)
        if not re_optimize_bfl:
            sub_opt = OpticalOptimizer(
                new_sys,
                wavelengths=self.wavelengths,
                max_iter=1,
                num_rays=self.num_rays,
                target=self.target,
                manufacturing_constraints=getattr(self, 'mfg_config', None)
            )
            return sub_opt.optimize(variable_indices={"thickness": []})

        air_vars = [
            i for i, s in enumerate(stack_snapped.surfaces[:-1])
            if str(getattr(s, 'glass', 'AIR') or 'AIR').upper() in ("AIR", "VACUUM", "")
        ]
        polish_opt = OpticalOptimizer(
            new_sys,
            wavelengths=self.wavelengths,
            method="Nelder-Mead",
            max_iter=50,
            num_rays=self.num_rays,
            target=self.target,
            target_efl=self.target_efl,
            manufacturing_constraints=getattr(self, 'mfg_config', None)
        )
        return polish_opt.optimize(variable_indices={"thickness": air_vars})


    def auto_aim_mirror(self, mirror_index: int, target_surface_index: int = -1, target_xy: Tuple[float, float] = (0.0, 0.0)) -> Any:
        """Convenience method to automatically solve and set mirror tilt angles to aim at target."""
        from optics_app.engine.mirror_positioner import MirrorPositioner
        wl = self.wavelengths[0] if isinstance(self.wavelengths, list) and len(self.wavelengths) > 0 else 0.5876
        return MirrorPositioner.aim_mirror_to_target(
            self.system.stack, mirror_index, target_surface_index=target_surface_index,
            target_xy=target_xy, wavelength=wl
        )

    def optimize_mirror_position(self, mirror_index: int, target_surface_index: int = -1, target_xy: Tuple[float, float] = (0.0, 0.0), **kwargs) -> Any:
        """Convenience method to optimize mirror distance, tilts, and decenters."""
        from optics_app.engine.mirror_positioner import MirrorPositioner
        return MirrorPositioner.optimize_mirror_spatial_position(
            self.system.stack, mirror_index, target_surface_index=target_surface_index,
            target_xy=target_xy, wavelengths=self.wavelengths, **kwargs
        )


__all__ = [
    "DiagnosticSeverity",
    "DiagnosticItem",
    "ValidationResult",
    "OptimizationValidator",
    "TerminationStatus",
    "WavelengthSpotMetric",
    "OptimizationResult",
    "OpticalOptimizer",
    "OptimizationAbortedException",
    "radius_to_curvature",
    "curvature_to_radius",
]

