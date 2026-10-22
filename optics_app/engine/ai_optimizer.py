"""
AI Optimization Loop for Optical System Design Parameters.

Implements an optimization routine where AI adjusts system design parameters
(such as wavelengths, fields, and apertures) to evaluate and minimize
a Zemax-style merit function.
"""

from typing import List, Dict, Any, Optional, Union, Tuple, Callable
import copy
import math
import numpy as np

try:
    from optics_app.engine.merit_function import MeritFunction, MeritOperand, OperandType
    from optics_app.engine.surface import Surface, SurfaceType
    from optics_app.engine.ray import Ray
except ImportError:
    from v2.engine.merit_function import MeritFunction, MeritOperand, OperandType
    from v2.engine.surface import Surface, SurfaceType
    from v2.engine.ray import Ray


def trace_system_for_merit(system: Any, wavelengths: Optional[List[float]] = None, num_rays: int = 11) -> List[Ray]:
    """
    Traces a representative paraxial/geometric ray fan through system to evaluate optical merit.
    """
    if hasattr(system, "stack") and hasattr(system.stack, "_update_accumulated_positions"):
        system.stack._update_accumulated_positions()
    elif hasattr(system, "surface_stack") and hasattr(system.surface_stack, "_update_accumulated_positions"):
        system.surface_stack._update_accumulated_positions()
    wls = (
        wavelengths
        or getattr(system, "wavelengths", None)
        or (getattr(system.stack, "wavelengths", None) if hasattr(system, "stack") else None)
        or [1.064]
    )

    beam_radius = 5.0
    if hasattr(system, "source_config") and system.source_config is not None:
        cfg = system.source_config
        beam_radius = float(cfg.get("beam_radius", 5.0)) if isinstance(cfg, dict) else float(getattr(cfg, "beam_radius", 5.0))
    elif hasattr(system, "system_config") and system.system_config is not None:
        cfg = system.system_config
        ap_val = float(cfg.get("aperture_value", 10.0)) if isinstance(cfg, dict) else float(getattr(cfg, "aperture_value", 10.0))
        beam_radius = ap_val / 2.0
    elif hasattr(system, "beam_radius") and system.beam_radius is not None:
        beam_radius = float(system.beam_radius)
    elif hasattr(system, "entrance_pupil_radius") and system.entrance_pupil_radius is not None:
        beam_radius = float(system.entrance_pupil_radius)


    field_angle_deg = 0.0
    if hasattr(system, "field_angle") and system.field_angle is not None:
        field_angle_deg = float(system.field_angle)
    elif hasattr(system, "system_config") and system.system_config is not None:
        cfg = system.system_config
        if isinstance(cfg, dict):
            field_angle_deg = float(cfg.get("field_angle", 0.0))
        elif hasattr(cfg, "fields") and cfg.fields:
            field_angle_deg = float(getattr(cfg.fields[0], "angle_y_deg", getattr(cfg.fields[0], "y", 0.0)))

    rays = []
    for wl in wls:
        for py in np.linspace(-1.0, 1.0, num_rays):
            y_pos = py * beam_radius
            ang_rad = math.radians(field_angle_deg)
            r = Ray(
                pos=np.array([0.0, y_pos, 0.0], dtype=float),
                dir=np.array([0.0, math.sin(ang_rad), math.cos(ang_rad)], dtype=float),
                wavelength=wl
            )
            rays.append(r)

    # Perform ray trace
    if hasattr(system, "trace_ray"):
        traced = [system.trace_ray(r) for r in rays]
    elif hasattr(system, "trace_rays"):
        traced = system.trace_rays(rays)
    else:
        try:
            from optics_app.engine.tracer import OpticalSystem
            opt_sys = OpticalSystem(system.surface_stack if hasattr(system, "surface_stack") else system)
            traced = [opt_sys.trace_ray(r) for r in rays]
        except Exception:
            traced = rays

    return traced


class AiOptimizationLoop:
    """
    AI Optimization Routine for system parameters (apertures, wavelengths, fields, curvatures).
    """

    def __init__(
        self,
        system: Any,
        merit_function: Optional[MeritFunction] = None,
        max_iterations: int = 20,
        tolerance: float = 1e-4
    ):
        self.system = system
        self.merit_function = merit_function or MeritFunction()
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.history: List[Dict[str, Any]] = []

    def evaluate_merit(self) -> float:
        """Trace rays and evaluate current scalar merit function value."""
        wls = getattr(self.system, "wavelengths", [1.064])
        traced_rays = trace_system_for_merit(self.system, wavelengths=wls)
        score = self.merit_function.evaluate(self.system, traced_rays, wavelengths=wls)
        return float(score)

    def get_parameter(self, param_name: str) -> float:
        """Read current value of a system parameter."""
        p_low = param_name.lower().strip()

        # Apertures
        if "aperture" in p_low or "beam_radius" in p_low or "epd" in p_low:
            if hasattr(self.system, "beam_radius") and self.system.beam_radius is not None:
                return float(self.system.beam_radius)
            if hasattr(self.system, "entrance_pupil_radius") and self.system.entrance_pupil_radius is not None:
                return float(self.system.entrance_pupil_radius)
            if hasattr(self.system, "source_config") and self.system.source_config is not None:
                cfg = self.system.source_config
                if isinstance(cfg, dict):
                    return float(cfg.get("beam_radius", 5.0))
                return float(getattr(cfg, "beam_radius", 5.0))
            if hasattr(self.system, "system_config") and self.system.system_config is not None:
                cfg = self.system.system_config
                if isinstance(cfg, dict):
                    return float(cfg.get("aperture_value", 10.0)) / 2.0
                return float(getattr(cfg, "aperture_value", 10.0)) / 2.0
            if hasattr(self.system, "surfaces") and len(self.system.surfaces) > 1:
                return float(self.system.surfaces[1].semi_diameter)
            return 5.0

        # Wavelengths
        if "wavelength" in p_low or "wl" in p_low:
            if hasattr(self.system, "primary_wavelength"):
                return float(self.system.primary_wavelength)
            if hasattr(self.system, "wavelengths") and self.system.wavelengths:
                return float(self.system.wavelengths[0])
            return 1.064

        # Field
        if "field" in p_low or "angle" in p_low:
            if hasattr(self.system, "field_angle") and self.system.field_angle is not None:
                return float(self.system.field_angle)
            if hasattr(self.system, "system_config") and self.system.system_config is not None:
                cfg = self.system.system_config
                if isinstance(cfg, dict):
                    return float(cfg.get("field_angle", 0.0))
                elif hasattr(cfg, "fields") and cfg.fields:
                    return float(getattr(cfg.fields[0], "angle_y_deg", getattr(cfg.fields[0], "y", 0.0)))
            return 0.0

        # Surface thickness / curvature e.g. "thickness_1" or "radius_2"
        if "_" in p_low:
            prop, idx_str = p_low.rsplit("_", 1)
            if idx_str.isdigit():
                idx = int(idx_str)
                surfs = getattr(self.system, "surfaces", [])
                if 0 <= idx < len(surfs):
                    return float(getattr(surfs[idx], prop, 0.0))

        return 1.0

    def set_parameter(self, param_name: str, value: float) -> None:
        """Set new value for a system parameter."""
        p_low = param_name.lower().strip()

        # Apertures
        if "aperture" in p_low or "beam_radius" in p_low or "epd" in p_low:
            val = max(0.1, float(value))
            self.system.beam_radius = val
            if hasattr(self.system, "entrance_pupil_radius"):
                self.system.entrance_pupil_radius = val
            if hasattr(self.system, "source_config") and self.system.source_config is not None:
                if isinstance(self.system.source_config, dict):
                    self.system.source_config["beam_radius"] = val
                else:
                    self.system.source_config.beam_radius = val
            if hasattr(self.system, "system_config") and self.system.system_config is not None:
                if isinstance(self.system.system_config, dict):
                    self.system.system_config["aperture_value"] = val * 2.0
                else:
                    self.system.system_config.aperture_value = val * 2.0
            if hasattr(self.system, "surfaces"):
                for s in self.system.surfaces[:3]:
                    if s.surface_type == SurfaceType.STOP:
                        s.semi_diameter = val
            return

        # Wavelengths
        if "wavelength" in p_low or "wl" in p_low:
            val = max(0.2, float(value))
            if hasattr(self.system, "primary_wavelength"):
                self.system.primary_wavelength = val
            if hasattr(self.system, "surface_stack") and hasattr(self.system.surface_stack, "primary_wavelength"):
                self.system.surface_stack.primary_wavelength = val
            if hasattr(self.system, "wavelengths") and self.system.wavelengths:
                self.system.wavelengths[0] = val
            if hasattr(self.system, "surface_stack") and hasattr(self.system.surface_stack, "wavelengths") and self.system.surface_stack.wavelengths:
                self.system.surface_stack.wavelengths[0] = val
            return

        # Field
        if "field" in p_low or "angle" in p_low:
            val = float(value)
            self.system.field_angle = val
            if hasattr(self.system, "system_config") and self.system.system_config is not None:
                if isinstance(self.system.system_config, dict):
                    self.system.system_config["field_angle"] = val
                elif hasattr(self.system.system_config, "fields") and self.system.system_config.fields:
                    f0 = self.system.system_config.fields[0]
                    f0.angle_y_deg = val
                    if hasattr(f0, "y"):
                        f0.y = val
            else:
                self.system.system_config = {"field_angle": val}
            return

        # Specific surface property e.g. "thickness_2" or "radius_1"
        if "_" in p_low:
            prop, idx_str = p_low.rsplit("_", 1)
            if idx_str.isdigit():
                idx = int(idx_str)
                surfs = getattr(self.system, "surfaces", [])
                if 0 <= idx < len(surfs):
                    setattr(surfs[idx], prop, float(value))
                    if hasattr(self.system, "surface_stack"):
                        self.system.surface_stack._update_accumulated_positions()
                    elif hasattr(self.system, "stack"):
                        self.system.stack._update_accumulated_positions()

    def optimize(
        self,
        parameters_to_adjust: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Execute iterative AI optimization loop to minimize merit function.
        Returns detailed summary of initial merit, final merit, iterations, and parameter adjustments.
        """
        params = parameters_to_adjust or ["aperture"]
        self.history = []

        initial_merit = self.evaluate_merit()
        current_merit = initial_merit
        best_merit = initial_merit
        best_params = {p: self.get_parameter(p) for p in params}

        self.history.append({
            "iteration": 0,
            "merit": current_merit,
            "parameters": dict(best_params)
        })

        for it in range(1, self.max_iterations + 1):
            improved_in_iteration = False

            for p_name in params:
                curr_val = best_params[p_name]
                step = max(abs(curr_val) * 0.05, 0.1)

                candidates = [
                    curr_val - step,
                    curr_val + step,
                    curr_val - 2.0 * step,
                    curr_val + 2.0 * step,
                ]

                for cand in candidates:
                    self.set_parameter(p_name, cand)
                    trial_merit = self.evaluate_merit()

                    if trial_merit < best_merit - self.tolerance:
                        best_merit = trial_merit
                        best_params[p_name] = cand
                        improved_in_iteration = True

                # Restore best candidate
                self.set_parameter(p_name, best_params[p_name])

            self.history.append({
                "iteration": it,
                "merit": best_merit,
                "parameters": dict(best_params)
            })

            # Check convergence
            if not improved_in_iteration:
                break

        final_merit = best_merit
        reduction = initial_merit - final_merit

        return {
            "success": bool(final_merit <= initial_merit),
            "initial_merit": float(initial_merit),
            "final_merit": float(final_merit),
            "merit_reduction": float(reduction),
            "percent_improvement": float((reduction / initial_merit * 100.0) if initial_merit > 0 else 0.0),
            "iterations": len(self.history) - 1,
            "adjusted_parameters": best_params,
            "history": self.history
        }


def run_ai_optimization_loop(
    system: Any,
    merit_function: Optional[MeritFunction] = None,
    parameters_to_adjust: Optional[List[str]] = None,
    max_iterations: int = 20
) -> Dict[str, Any]:
    """
    Convenience functional API to execute the AI optimization loop.
    """
    opt = AiOptimizationLoop(
        system=system,
        merit_function=merit_function,
        max_iterations=max_iterations
    )
    return opt.optimize(parameters_to_adjust=parameters_to_adjust)
