"""
Extended Optical Merit Operands and Glatzel Adaptive DLS Optimization.
=====================================================================
Implements standardized optical design operands extracted from classical optical
engineering literature (Kidger, Smith, Kingslake, Fischer, ISO 10110-12):
- TransverseRayAberrationOperand (TRAR): Measures geometric ray intercept error (y - y_chief).
- SpotSizeRMSOperand (RSCE): Evaluates Root-Mean-Square (RMS) spot radius on focal plane.
- SpotSizeGEOOperand (SPOT): Evaluates 100% geometric encircled spot radius.
- EdgeThicknessConstraintOperand (MNET / ETMN): Enforces minimum edge thickness constraint (te >= min_et)
  via smooth one-sided quadratic penalty barriers.

Also implements Erhard Glatzel's Adaptive Damped Least Squares Optimizer (1968):
- GlatzelAdaptiveDLSOptimizer: Parameter-specific adaptive damping where:
    (J^T W^2 J + D) Delta x = - J^T W^2 r
    D_jj = p^2 * (J^T W^2 J)_jj * gamma_j
  Dynamically steers parameter-specific damping factors gamma_j based on linear prediction
  discrepancy vs actual non-linear ray evaluation.
"""

from __future__ import annotations
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

from optics_app.engine.lens_merit import (
    Operand,
    PhysicalMeritFunction,
    TargetEFLOperand,
    ThicknessConstraintOperand,
    DiameterConstraintOperand,
)
from optics_app.engine.lens_optimizer import (
    BaseOptimizer,
    OptimizationResult,
    OptimizationStepRecord,
)
from optics_app.engine.thick_lens import ThickLens, OpticalElement
from optics_app.engine.aspheric import AsphericLens, AsphericSurfaceModel
from optics_app.engine.ray import Ray
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.surface import Surface, SurfaceType


# =============================================================================
# Helper Utilities for Ray Tracing and Geometry Evaluation
# =============================================================================

def build_evaluation_optical_system(
    system: Any,
    back_focal_distance: Optional[float] = None,
    detector_semi_diameter: Optional[float] = None,
) -> Tuple[OpticalSystem, float]:
    """
    Construct a ray-traceable OpticalSystem with a detector placed at the focal plane.

    Args:
        system: OpticalElement (ThickLens, AsphericLens) or existing OpticalSystem.
        back_focal_distance: Distance from rear lens vertex to detector.
            If None, extracted paraxially from system.cardinal_points().bfl or system.bfl.
        detector_semi_diameter: Clear aperture radius for detector surface.

    Returns:
        Tuple of (OpticalSystem, effective_bfl).
    """
    if isinstance(system, OpticalSystem):
        # Already an OpticalSystem
        bfl = 0.0
        if len(system.surfaces) > 1:
            bfl = float(system.surfaces[-2].thickness)
        return system, bfl

    # Extract paraxial BFL if not provided
    if back_focal_distance is not None:
        bfl = float(back_focal_distance)
    elif hasattr(system, "bfl") and not math.isnan(system.bfl) and not math.isinf(system.bfl):
        bfl = float(system.bfl)
    elif hasattr(system, "cardinal_points"):
        cp = system.cardinal_points()
        bfl = float(cp.bfl)
    elif hasattr(system, "efl"):
        bfl = float(system.efl)
    else:
        bfl = 20.0

    if bfl <= 0.5 or math.isinf(bfl) or math.isnan(bfl):
        bfl = max(1.0, float(getattr(system, "efl", 20.0)))

    # Convert element to surfaces
    if hasattr(system, "to_surfaces") and callable(system.to_surfaces):
        surfaces = system.to_surfaces()
    else:
        raise TypeError(f"Cannot construct OpticalSystem from object of type {type(system)}")

    if len(surfaces) < 2:
        raise ValueError("Expected at least 2 surfaces (front, rear) from OpticalElement.")

    # Rear surface thickness is the distance to detector (BFL)
    surfaces[-1].thickness = bfl
    surfaces[-1].glass = "AIR"

    diam = float(getattr(system, "diameter", 25.4))
    det_semi_diam = float(detector_semi_diameter if detector_semi_diameter is not None else max(diam, 25.4))

    detector = Surface(
        name="Evaluation_Detector",
        surface_type=SurfaceType.DETECTOR,
        radius=float("inf"),
        thickness=0.0,
        glass="AIR",
        semi_diameter=det_semi_diam,
    )

    stack = SurfaceStack()
    for s in surfaces:
        stack.add_surface(s)
    stack.add_surface(detector)

    opt_sys = OpticalSystem(stack)
    return opt_sys, bfl


def trace_pupil_rays(
    system: Any,
    num_rays: int = 21,
    wavelength_nm: float = 587.6,
    field_angle_deg: float = 0.0,
    pupil_coverage: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Trace a 1D meridional fan of parallel rays across the entrance pupil.

    Args:
        system: OpticalElement or OpticalSystem to trace through.
        num_rays: Number of rays in the pupil fan.
        wavelength_nm: Optical wavelength in nm.
        field_angle_deg: Field angle in degrees (default 0.0 on-axis).
        pupil_coverage: Fractional pupil radius to sample (0.95 prevents rim clipping).

    Returns:
        Tuple of (x_hits, y_hits, valid_count) on the detector plane.
    """
    opt_sys, _ = build_evaluation_optical_system(system)

    diam = float(getattr(system, "diameter", 25.4))
    semi_diam = diam / 2.0
    r_max = semi_diam * pupil_coverage

    rho_vals = np.linspace(-1.0, 1.0, num_rays)
    wl_um = wavelength_nm / 1000.0

    # Field angle direction vector
    angle_rad = math.radians(field_angle_deg)
    dir_x = 0.0
    dir_y = math.sin(angle_rad)
    dir_z = math.cos(angle_rad)
    dir_vec = np.array([dir_x, dir_y, dir_z], dtype=float)

    x_hits: List[float] = []
    y_hits: List[float] = []

    for rho in rho_vals:
        y0 = rho * r_max
        x0 = 0.0
        pos0 = np.array([x0, y0, -1.0], dtype=float)

        ray = Ray(pos=pos0, dir=dir_vec.copy(), wavelength=wl_um)
        traced = opt_sys.trace_ray(ray)

        if not traced.vignetted and not traced.tir and len(traced.history) > 0:
            last_hit = traced.history[-1]
            x_h = float(last_hit.get("x_local", last_hit.get("x", 0.0)))
            y_h = float(last_hit.get("y_local", last_hit.get("y", 0.0)))
            x_hits.append(x_h)
            y_hits.append(y_h)

    valid_count = len(y_hits)
    return np.array(x_hits, dtype=float), np.array(y_hits, dtype=float), valid_count


# =============================================================================
# 1. Standard Optical Design Operands
# =============================================================================

class TransverseRayAberrationOperand(Operand):
    """
    Transverse Ray Aberration Operand (TRAR).

    Evaluates geometric transverse ray deviation:
        delta_y = y(rho_y) - y_chief
    at the paraxial image / detector plane for a prescribed normalized pupil coordinate
    rho_y in [-1.0, +1.0].
    """

    def __init__(
        self,
        ray_height_fraction: float = 1.0,
        target: float = 0.0,
        weight: float = 1.0,
        wavelength_nm: float = 587.6,
        field_angle_deg: float = 0.0,
        name: str = "TRAR",
    ) -> None:
        super().__init__(name=name, target=target, weight=weight)
        self.ray_height_fraction = float(ray_height_fraction)
        self.wavelength_nm = float(wavelength_nm)
        self.field_angle_deg = float(field_angle_deg)

    def evaluate(self, system: Any, context: Optional[Dict[str, Any]] = None) -> float:
        """
        Evaluate transverse ray intercept error y(rho_y) - y_chief (mm).
        """
        try:
            opt_sys, _ = build_evaluation_optical_system(system)
            diam = float(getattr(system, "diameter", 25.4))
            semi_diam = diam / 2.0
            wl_um = self.wavelength_nm / 1000.0

            angle_rad = math.radians(self.field_angle_deg)
            dir_vec = np.array([0.0, math.sin(angle_rad), math.cos(angle_rad)], dtype=float)

            # 1. Chief ray at rho_y = 0.0
            ray_chief = Ray(pos=np.array([0.0, 0.0, -1.0]), dir=dir_vec.copy(), wavelength=wl_um)
            traced_chief = opt_sys.trace_ray(ray_chief)
            y_chief = 0.0
            if not traced_chief.vignetted and len(traced_chief.history) > 0:
                y_chief = float(traced_chief.history[-1].get("y_local", traced_chief.history[-1].get("y", 0.0)))

            # 2. Marginal / Zonal ray at rho_y
            y0 = self.ray_height_fraction * semi_diam * 0.95
            ray_zone = Ray(pos=np.array([0.0, y0, -1.0]), dir=dir_vec.copy(), wavelength=wl_um)
            traced_zone = opt_sys.trace_ray(ray_zone)

            if traced_zone.vignetted or traced_zone.tir or len(traced_zone.history) == 0:
                # Heavy penalty if ray is clipped
                return 10.0

            y_zone = float(traced_zone.history[-1].get("y_local", traced_zone.history[-1].get("y", 0.0)))
            delta_y = y_zone - y_chief
            return float(delta_y)
        except Exception:
            return 10.0

    def compute_residual(self, system: Any, context: Optional[Dict[str, Any]] = None) -> float:
        actual = self.evaluate(system, context)
        return float(actual - self.target)


class SpotSizeRMSOperand(Operand):
    """
    RMS Spot Radius Operand (RSCE / TRAC).

    Evaluates Root-Mean-Square (RMS) spot radius relative to spot centroid:
        R_RMS = sqrt( (1 / N) * sum_i ( (x_i - x_bar)^2 + (y_i - y_bar)^2 ) )
    Matching standard Zemax RSCE conventions.
    """

    def __init__(
        self,
        num_rays: int = 21,
        target: float = 0.0,
        weight: float = 1.0,
        wavelength_nm: float = 587.6,
        field_angle_deg: float = 0.0,
        name: str = "RSCE",
    ) -> None:
        super().__init__(name=name, target=target, weight=weight)
        self.num_rays = max(3, int(num_rays))
        self.wavelength_nm = float(wavelength_nm)
        self.field_angle_deg = float(field_angle_deg)

    def evaluate(self, system: Any, context: Optional[Dict[str, Any]] = None) -> float:
        """
        Evaluate RMS spot radius R_RMS in mm.
        """
        try:
            x_hits, y_hits, valid_count = trace_pupil_rays(
                system,
                num_rays=self.num_rays,
                wavelength_nm=self.wavelength_nm,
                field_angle_deg=self.field_angle_deg,
            )
            if valid_count < 2:
                return 10.0

            x_bar = np.mean(x_hits)
            y_bar = np.mean(y_hits)
            r2 = (x_hits - x_bar) ** 2 + (y_hits - y_bar) ** 2
            rms = math.sqrt(float(np.mean(r2)))

            # Penalize vignetting fraction if some rays clipped
            if valid_count < self.num_rays:
                penalty = 1.0 + (self.num_rays - valid_count) / float(self.num_rays)
                rms *= penalty

            return float(rms)
        except Exception:
            return 10.0

    def compute_residual(self, system: Any, context: Optional[Dict[str, Any]] = None) -> float:
        actual = self.evaluate(system, context)
        return float(actual - self.target)


class SpotSizeGEOOperand(Operand):
    """
    100% Geometric Spot Size Operand (SPOT / GEOR).

    Evaluates maximum geometric blur radius encircling all valid rays:
        R_GEO = max_i sqrt( (x_i - x_bar)^2 + (y_i - y_bar)^2 )
    Matching standard Zemax SPOT conventions.
    """

    def __init__(
        self,
        num_rays: int = 21,
        target: float = 0.0,
        weight: float = 1.0,
        wavelength_nm: float = 587.6,
        field_angle_deg: float = 0.0,
        name: str = "SPOT",
    ) -> None:
        super().__init__(name=name, target=target, weight=weight)
        self.num_rays = max(3, int(num_rays))
        self.wavelength_nm = float(wavelength_nm)
        self.field_angle_deg = float(field_angle_deg)

    def evaluate(self, system: Any, context: Optional[Dict[str, Any]] = None) -> float:
        """
        Evaluate 100% geometric spot radius R_GEO in mm.
        """
        try:
            x_hits, y_hits, valid_count = trace_pupil_rays(
                system,
                num_rays=self.num_rays,
                wavelength_nm=self.wavelength_nm,
                field_angle_deg=self.field_angle_deg,
            )
            if valid_count < 2:
                return 10.0

            x_bar = np.mean(x_hits)
            y_bar = np.mean(y_hits)
            r = np.sqrt((x_hits - x_bar) ** 2 + (y_hits - y_bar) ** 2)
            geo = float(np.max(r))

            if valid_count < self.num_rays:
                geo += (self.num_rays - valid_count) * 0.5

            return float(geo)
        except Exception:
            return 10.0

    def compute_residual(self, system: Any, context: Optional[Dict[str, Any]] = None) -> float:
        actual = self.evaluate(system, context)
        return float(actual - self.target)


class EdgeThicknessConstraintOperand(Operand):
    """
    Minimum Edge Thickness Constraint Operand (MNET / ETMN).

    Enforces minimum edge thickness:
        t_e >= min_edge_thickness (default 1.5 mm)
    at the clear aperture rim (h = diameter / 2).

    Uses a smooth one-sided quadratic penalty barrier:
        r = max(0, min_edge_thickness - t_e)
    When t_e >= min_edge_thickness, the residual is exactly 0.0 and contributes zero penalty.
    """

    def __init__(
        self,
        min_edge_thickness: float = 1.5,
        weight: float = 10.0,
        name: str = "MNET",
        target: Optional[float] = None,
    ) -> None:
        tgt = float(target if target is not None else min_edge_thickness)
        super().__init__(name=name, target=tgt, weight=weight)
        self.min_edge_thickness = float(min_edge_thickness)

    def evaluate(self, system: Any, context: Optional[Dict[str, Any]] = None) -> float:
        """
        Compute actual physical edge thickness t_e in mm at clear aperture rim.
        """
        try:
            diam = float(getattr(system, "diameter", 25.4))
            h = diam / 2.0
            t = float(getattr(system, "thickness", 5.0))

            # Check if AsphericLens with surface models
            if hasattr(system, "surface1") and hasattr(system, "surface2"):
                try:
                    z1 = system.surface1.sag(h)
                except Exception:
                    z1 = 0.0
                try:
                    z2 = system.surface2.sag(h)
                except Exception:
                    z2 = 0.0
                te = t + z2 - z1
                return float(te)

            # Standard ThickLens or radius-based element
            r1 = float(getattr(system, "r1", 50.0))
            r2 = float(getattr(system, "r2", -50.0))

            c1 = 1.0 / r1 if abs(r1) > 1e-12 and not math.isinf(r1) else 0.0
            c2 = 1.0 / r2 if abs(r2) > 1e-12 and not math.isinf(r2) else 0.0

            # Front sag: convex front curves into positive z
            z1 = 0.0
            if abs(c1) > 1e-15:
                arg1 = 1.0 - (c1 * h) ** 2
                if arg1 >= 0.0:
                    z1 = (c1 * h * h) / (1.0 + math.sqrt(arg1))
                else:
                    z1 = (c1 * h * h)  # Fallback approximation beyond hemisphere

            # Rear sag: convex rear curves into negative z relative to rear vertex
            z2 = 0.0
            if abs(c2) > 1e-15:
                arg2 = 1.0 - (c2 * h) ** 2
                if arg2 >= 0.0:
                    z2 = (c2 * h * h) / (1.0 + math.sqrt(arg2))
                else:
                    z2 = (c2 * h * h)

            te = t + z2 - z1
            return float(te)
        except Exception:
            return 0.0

    def compute_residual(self, system: Any, context: Optional[Dict[str, Any]] = None) -> float:
        """
        One-sided penalty: returns max(0, min_edge_thickness - t_e).
        """
        te = self.evaluate(system, context)
        if te < self.min_edge_thickness:
            return float(self.min_edge_thickness - te)
        return 0.0


# =============================================================================
# Zemax Illumination, Étendue & Throughput Operands (Herbert Gross Vol 6)
# =============================================================================

class EtendueMatchOperand(Operand):
    """
    Étendue Matching Operand (ETEN).
    Evaluates system geometrical optical étendue G = A * Omega = pi * A * NA^2 (mm^2 * sr)
    and penalizes mismatch from target étendue G_target.
    """

    def __init__(self, target_etendue: float = 1.0, weight: float = 1.0, name: str = "ETEN") -> None:
        super().__init__(name=name, target=target_etendue, weight=weight)

    def evaluate(self, system: Any, context: Optional[Dict[str, Any]] = None) -> float:
        if hasattr(system, "evaluate_throughput_and_uniformity"):
            m = system.evaluate_throughput_and_uniformity()
            return float(m.limiting_etendue_mm2_sr)
        diam = float(getattr(system, "diameter", 25.4))
        na = float(getattr(system, "na", 0.25))
        area = math.pi * ((diam / 2.0) ** 2)
        return float(math.pi * area * (na ** 2))

    def compute_residual(self, system: Any, context: Optional[Dict[str, Any]] = None) -> float:
        return float(self.evaluate(system, context) - self.target)


class FluxThroughputOperand(Operand):
    """
    Radiant Flux / Energy Throughput Operand (FLUX).
    Evaluates total delivered optical power (Watts) or percentage (%) reaching the image/detector.
    Maximizes throughput towards target (e.g. 100.0%).
    """

    def __init__(self, target_throughput_pct: float = 85.0, weight: float = 2.0, name: str = "FLUX") -> None:
        super().__init__(name=name, target=target_throughput_pct, weight=weight)

    def evaluate(self, system: Any, context: Optional[Dict[str, Any]] = None) -> float:
        if hasattr(system, "evaluate_throughput_and_uniformity"):
            m = system.evaluate_throughput_and_uniformity()
            return float(m.total_optical_throughput_pct)
        return 80.0

    def compute_residual(self, system: Any, context: Optional[Dict[str, Any]] = None) -> float:
        actual = self.evaluate(system, context)
        if actual < self.target:
            return float(self.target - actual)
        return 0.0


class IrradianceUniformityOperand(Operand):
    """
    Spatial Irradiance Uniformity Operand (UNIF).
    Evaluates spatial flat-top uniformity percentage (100% = ideal flat top).
    Penalizes deviations below prescribed target (e.g. 95%).
    """

    def __init__(self, target_uniformity_pct: float = 95.0, weight: float = 3.0, name: str = "UNIF") -> None:
        super().__init__(name=name, target=target_uniformity_pct, weight=weight)

    def evaluate(self, system: Any, context: Optional[Dict[str, Any]] = None) -> float:
        if hasattr(system, "evaluate_throughput_and_uniformity"):
            m = system.evaluate_throughput_and_uniformity()
            return float(m.mask_uniformity_rms_pct)
        return 90.0

    def compute_residual(self, system: Any, context: Optional[Dict[str, Any]] = None) -> float:
        actual = self.evaluate(system, context)
        if actual < self.target:
            return float(self.target - actual)
        return 0.0


class TelecentricityOperand(Operand):
    """
    Telecentricity Angle Operand (TELE).
    Evaluates chief ray angle of incidence (degrees) at mask or image plane.
    Target is strictly 0.0 deg for telecentric lithography and metrology.
    """

    def __init__(self, target_deg: float = 0.0, weight: float = 5.0, name: str = "TELE") -> None:
        super().__init__(name=name, target=target_deg, weight=weight)

    def evaluate(self, system: Any, context: Optional[Dict[str, Any]] = None) -> float:
        if hasattr(system, "evaluate_throughput_and_uniformity"):
            m = system.evaluate_throughput_and_uniformity()
            return float(m.chief_ray_telecentricity_error_deg)
        return 0.0

    def compute_residual(self, system: Any, context: Optional[Dict[str, Any]] = None) -> float:
        return float(abs(self.evaluate(system, context) - self.target))


class PupilMatchingOperand(Operand):
    """
    Pupil Matching Error Operand (PUMA).
    Evaluates axial distance mismatch (mm) between Illumination Exit Pupil and Projection Entrance Pupil.
    Target is strictly 0.0 mm for zero vignetting and maximum throughput.
    """

    def __init__(self, target_mismatch_mm: float = 0.0, weight: float = 4.0, name: str = "PUMA") -> None:
        super().__init__(name=name, target=target_mismatch_mm, weight=weight)

    def evaluate(self, system: Any, context: Optional[Dict[str, Any]] = None) -> float:
        if hasattr(system, "solve_pupil_matching"):
            res = system.solve_pupil_matching()
            return float(res.get("axial_error_mm", 0.0))
        return 0.0

    def compute_residual(self, system: Any, context: Optional[Dict[str, Any]] = None) -> float:
        return float(abs(self.evaluate(system, context) - self.target))


class ExtendedMeritFunction(PhysicalMeritFunction):
    """
    Extended Composite Optical Merit Function.

    Subclasses PhysicalMeritFunction to provide standardized optical operands:
    - TargetEFLOperand (EFFL)
    - SpotSizeRMSOperand (RSCE)
    - SpotSizeGEOOperand (SPOT)
    - TransverseRayAberrationOperand (TRAR)
    - EdgeThicknessConstraintOperand (MNET)
    - ThicknessConstraintOperand (MNCT)
    """

    def __init__(
        self,
        operands: Optional[Sequence[Operand]] = None,
        target_efl: Optional[float] = None,
        min_edge_thickness: float = 1.5,
        weight_efl: float = 10.0,
        weight_spot: float = 1.0,
        weight_edge: float = 20.0,
        num_rays: int = 21,
    ) -> None:
        super().__init__(operands=operands)
        if target_efl is not None:
            self.add_operand(TargetEFLOperand(target_efl=target_efl, weight=weight_efl))
            self.add_operand(SpotSizeRMSOperand(num_rays=num_rays, weight=weight_spot))
            self.add_operand(EdgeThicknessConstraintOperand(min_edge_thickness=min_edge_thickness, weight=weight_edge))


# =============================================================================
# 2. Glatzel Adaptive Damped Least Squares (DLS) Optimizer
# =============================================================================

class GlatzelAdaptiveDLSOptimizer(BaseOptimizer):
    """
    Erhard Glatzel Adaptive Damped Least Squares Optimizer (1968).

    Addresses ill-conditioned optical optimization landscapes with extreme eigenvalue
    spreads (kappa > 10^8) by replacing scalar Levenberg-Marquardt damping lambda * I
    with parameter-specific diagonal damping:
        (J^T W^2 J + D) Delta x = - J^T W^2 r
    where:
        D_jj = p^2 * (J^T W^2 J)_jj * gamma_j

    The parameter-specific steering factors gamma_j are updated adaptively each iteration
    by assessing the discrepancy between linear model predictions and actual non-linear
    ray trace evaluations:
        rho_j = |r_actual,j - r_predicted,j| / max(|r_predicted,j|, 1e-6)
    - If rho_j > 0.5: parameter j is non-linear/runaway -> gamma_j <- gamma_j * 2.0 (heavier damping)
    - If rho_j < 0.1: parameter j is well-behaved/linear  -> gamma_j <- max(0.01, gamma_j * 0.7) (acceleration)
    """

    def __init__(
        self,
        merit_function: Optional[PhysicalMeritFunction] = None,
        max_iterations: int = 50,
        tolerance: float = 1e-5,
        initial_damping: float = 0.01,
        variable_names: Optional[List[str]] = None,
    ) -> None:
        super().__init__(merit_function=merit_function, max_iterations=max_iterations, tolerance=tolerance)
        self.initial_damping = float(initial_damping)
        self._current_damping = float(initial_damping)
        self.variable_names = list(variable_names) if variable_names is not None else None
        self.gamma: Optional[np.ndarray] = None  # Parameter steering factors

    def reset(self) -> None:
        super().reset()
        self._current_damping = self.initial_damping
        self.gamma = None

    def _get_active_variables(self, system: OpticalElement) -> Tuple[np.ndarray, List[str]]:
        """Extract active optimization variables matching configured variable_names or defaults."""
        if self.variable_names is not None:
            names = self.variable_names
            vals = []
            for name in names:
                n_lower = name.lower()
                if n_lower == "c1":
                    c1 = getattr(system, "c1", 1.0 / system.r1 if abs(system.r1) > 1e-12 else 0.0)
                    vals.append(c1)
                elif n_lower == "c2":
                    c2 = getattr(system, "c2", 1.0 / system.r2 if abs(system.r2) > 1e-12 else 0.0)
                    vals.append(c2)
                elif n_lower == "r1":
                    vals.append(float(system.r1))
                elif n_lower == "r2":
                    vals.append(float(system.r2))
                elif n_lower in ("thickness", "t"):
                    vals.append(float(system.thickness))
                elif n_lower in ("diameter", "d"):
                    vals.append(float(system.diameter))
                elif n_lower in ("k1", "conic1", "conic"):
                    vals.append(float(getattr(system, "k1", 0.0)))
                elif n_lower in ("k2", "conic2"):
                    vals.append(float(getattr(system, "k2", 0.0)))
                elif n_lower in ("alpha4", "a4"):
                    coeffs = getattr(system.surface1, "aspheric_coeffs", []) if hasattr(system, "surface1") else []
                    vals.append(float(coeffs[0]) if len(coeffs) > 0 else 0.0)
                elif n_lower in ("alpha6", "a6"):
                    coeffs = getattr(system.surface1, "aspheric_coeffs", []) if hasattr(system, "surface1") else []
                    vals.append(float(coeffs[1]) if len(coeffs) > 1 else 0.0)
                else:
                    vals.append(float(system.get_parameters().get(name, 0.0)))
            return np.array(vals, dtype=float), names

        # Default extraction
        if isinstance(system, AsphericLens):
            # Aspheric lens default: c1, c2, thickness, k1
            names = ["c1", "c2", "thickness", "k1"]
            vals = [system.c1, system.c2, system.thickness, system.k1]
            return np.array(vals, dtype=float), names
        else:
            # ThickLens default: c1, c2, thickness
            r1 = system.r1
            r2 = system.r2
            c1 = 1.0 / r1 if abs(r1) > 1e-12 and not math.isinf(r1) else 0.0
            c2 = 1.0 / r2 if abs(r2) > 1e-12 and not math.isinf(r2) else 0.0
            names = ["c1", "c2", "thickness"]
            vals = [c1, c2, system.thickness]
            return np.array(vals, dtype=float), names

    def _apply_variables(self, system: OpticalElement, p: np.ndarray, var_names: List[str]) -> None:
        """Apply variable updates back to the optical element."""
        for name, val in zip(var_names, p):
            n_lower = name.lower()
            if n_lower == "c1":
                c_val = float(val)
                r_val = 1.0 / c_val if abs(c_val) > 1e-15 else float("inf")
                system.r1 = r_val
            elif n_lower == "c2":
                c_val = float(val)
                r_val = 1.0 / c_val if abs(c_val) > 1e-15 else float("inf")
                system.r2 = r_val
            elif n_lower == "r1":
                system.r1 = float(val)
            elif n_lower == "r2":
                system.r2 = float(val)
            elif n_lower in ("thickness", "t"):
                system.thickness = max(0.5, float(val))
            elif n_lower in ("diameter", "d"):
                system.diameter = max(1.0, float(val))
            elif n_lower in ("k1", "conic1", "conic"):
                if hasattr(system, "k1"):
                    system.k1 = float(val)
                elif hasattr(system, "surface1"):
                    setattr(system.surface1, "conic", float(val))
            elif n_lower in ("k2", "conic2"):
                if hasattr(system, "k2"):
                    system.k2 = float(val)
                elif hasattr(system, "surface2"):
                    setattr(system.surface2, "conic", float(val))
            elif n_lower in ("alpha4", "a4"):
                if hasattr(system, "surface1") and hasattr(system.surface1, "aspheric_coeffs"):
                    coeffs = list(system.surface1.aspheric_coeffs)
                    while len(coeffs) < 1:
                        coeffs.append(0.0)
                    coeffs[0] = float(val)
                    system.surface1.aspheric_coeffs = coeffs
            elif n_lower in ("alpha6", "a6"):
                if hasattr(system, "surface1") and hasattr(system.surface1, "aspheric_coeffs"):
                    coeffs = list(system.surface1.aspheric_coeffs)
                    while len(coeffs) < 2:
                        coeffs.append(0.0)
                    coeffs[1] = float(val)
                    system.surface1.aspheric_coeffs = coeffs

    def _compute_bounds(self, system: OpticalElement, var_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Compute physical box bounds for active variables."""
        curr_d = float(getattr(system, "diameter", 25.4))
        semi_d = max(curr_d / 2.0, 5.0)
        c_limit = 0.999 / semi_d  # Hemispherical curvature limit

        lb_list: List[float] = []
        ub_list: List[float] = []

        for name in var_names:
            n_lower = name.lower()
            if n_lower in ("c1", "c2"):
                lb_list.append(-c_limit)
                ub_list.append(c_limit)
            elif n_lower in ("r1", "r2"):
                lb_list.append(-1000.0)
                ub_list.append(1000.0)
            elif n_lower in ("thickness", "t"):
                lb_list.append(1.0)
                ub_list.append(50.0)
            elif n_lower in ("diameter", "d"):
                lb_list.append(5.0)
                ub_list.append(200.0)
            elif n_lower in ("k1", "k2", "conic1", "conic2", "conic"):
                lb_list.append(-20.0)
                ub_list.append(10.0)
            elif n_lower in ("alpha4", "a4"):
                lb_list.append(-1e-3)
                ub_list.append(1e-3)
            elif n_lower in ("alpha6", "a6"):
                lb_list.append(-1e-6)
                ub_list.append(1e-6)
            else:
                lb_list.append(-100.0)
                ub_list.append(100.0)

        return np.array(lb_list, dtype=float), np.array(ub_list, dtype=float)

    def step(
        self,
        system: OpticalElement,
        merit_function: Optional[PhysicalMeritFunction] = None,
    ) -> OptimizationStepRecord:
        """
        Execute one Glatzel adaptive DLS optimization step.
        """
        mf = self._get_merit_function(merit_function)
        self._step_count += 1

        p, var_names = self._get_active_variables(system)
        num_vars = len(p)

        if self.gamma is None or len(self.gamma) != num_vars:
            self.gamma = np.ones(num_vars, dtype=float)

        lb, ub = self._compute_bounds(system, var_names)

        # Baseline evaluation
        r0 = mf.residuals(system)
        merit0 = mf.evaluate(system)

        # Finite difference step sizes per parameter
        dp = np.zeros(num_vars, dtype=float)
        for j, name in enumerate(var_names):
            n_lower = name.lower()
            if n_lower in ("c1", "c2"):
                dp[j] = 1e-6
            elif n_lower in ("thickness", "t"):
                dp[j] = 1e-5
            elif n_lower in ("k1", "k2", "conic"):
                dp[j] = 1e-4
            elif n_lower in ("alpha4", "a4"):
                dp[j] = 1e-8
            elif n_lower in ("alpha6", "a6"):
                dp[j] = 1e-11
            else:
                dp[j] = 1e-5

        # Numerical Jacobian calculation: J_ij = d r_i / d x_j
        J = np.zeros((len(r0), num_vars), dtype=float)
        for j in range(num_vars):
            p_trial = p.copy()
            p_trial[j] += dp[j]
            self._apply_variables(system, p_trial, var_names)
            r_trial = mf.residuals(system)
            J[:, j] = (r_trial - r0) / dp[j]

        # Restore nominal state
        self._apply_variables(system, p, var_names)

        # Weighting matrix W^2
        weights = np.array([op.weight for op in mf.operands], dtype=float)
        W2 = np.diag(weights) if len(weights) == len(r0) else np.eye(len(r0))

        # Normal equations: (J^T W^2 J + D) Delta x = - J^T W^2 r
        JT_W2 = J.T @ W2
        JT_W2_J = JT_W2 @ J
        JT_W2_r = JT_W2 @ r0

        # Glatzel diagonal damping: D_jj = p^2 * (JT_W2_J_jj + eps) * gamma_j
        eps = 1e-7
        diag_A = np.diag(JT_W2_J)
        D_diag = self._current_damping * (np.maximum(diag_A, 0.0) + eps) * self.gamma
        D = np.diag(D_diag)

        A_system = JT_W2_J + D
        b_system = -JT_W2_r

        try:
            delta_p = np.linalg.solve(A_system, b_system)
        except np.linalg.LinAlgError:
            delta_p = np.linalg.lstsq(A_system, b_system, rcond=None)[0]

        # Apply bounds clamping
        p_new = np.clip(p + delta_p, lb, ub)
        delta_p_eff = p_new - p

        # Linear model prediction for new residuals
        r_pred = r0 + J @ delta_p_eff

        # Actual non-linear evaluation
        self._apply_variables(system, p_new, var_names)
        r_actual = mf.residuals(system)
        new_merit = mf.evaluate(system)

        # Glatzel Dynamic Steering Factor Update:
        # Measure per-parameter discrepancy between actual non-linear ray evaluation
        # and linear prediction
        for j in range(num_vars):
            # Projection of non-linear discrepancy onto parameter gradient
            nonlin_diff = np.abs(r_actual - r_pred)
            pred_scale = np.maximum(np.abs(r_pred), 1e-6)
            rel_err_vector = nonlin_diff / pred_scale

            # Weight by parameter sensitivity
            param_weight = np.abs(J[:, j])
            weight_sum = float(np.sum(param_weight))
            if weight_sum > 1e-12:
                rho_j = float(np.sum(param_weight * rel_err_vector) / weight_sum)
            else:
                rho_j = float(np.mean(rel_err_vector))

            # Glatzel adaptation rules:
            if rho_j > 0.5:
                # Highly non-linear / runaway: damp more heavily
                self.gamma[j] = min(1e5, self.gamma[j] * 2.0)
            elif rho_j < 0.1:
                # Well-behaved linear region: accelerate step
                self.gamma[j] = max(0.01, self.gamma[j] * 0.7)

        # Damping acceptance / backtracking
        if new_merit <= merit0:
            # Step accepted: reduce base damping for faster convergence
            self._current_damping = max(1e-7, self._current_damping * 0.8)
        else:
            # Step increased merit: increase base damping
            self._current_damping = min(1e4, self._current_damping * 2.0)

        params_dict = {name: float(val) for name, val in zip(var_names, p_new)}
        residuals_dict = {op.name: float(op.compute_residual(system)) for op in mf.operands}

        return OptimizationStepRecord(
            iteration=self._step_count,
            parameters=params_dict,
            merit_value=new_merit,
            operand_residuals=residuals_dict,
            step_norm=float(np.linalg.norm(delta_p_eff)),
            damping=self._current_damping,
            jacobian=J,
            notes=(
                f"Glatzel DLS step {self._step_count} (damping={self._current_damping:.2e}, "
                f"gamma_mean={np.mean(self.gamma):.2f})"
            ),
        )

    def optimize(
        self,
        system: OpticalElement,
        merit_function: Optional[PhysicalMeritFunction] = None,
    ) -> OptimizationResult:
        """
        Execute iterative Glatzel adaptive DLS optimization.
        """
        mf = self._get_merit_function(merit_function)
        self.reset()

        p0, var_names = self._get_active_variables(system)
        lb, ub = self._compute_bounds(system, var_names)
        p0 = np.clip(p0, lb, ub)
        self._apply_variables(system, p0, var_names)

        initial_merit = mf.evaluate(system)
        initial_params = {name: float(val) for name, val in zip(var_names, p0)}
        initial_residuals = {op.name: float(op.compute_residual(system)) for op in mf.operands}

        history: List[OptimizationStepRecord] = [
            OptimizationStepRecord(
                iteration=0,
                parameters=initial_params.copy(),
                merit_value=initial_merit,
                operand_residuals=initial_residuals,
                step_norm=0.0,
                damping=self.initial_damping,
                notes="Initial state before Glatzel DLS optimization",
            )
        ]

        converged = False
        prev_merit = initial_merit
        best_merit = initial_merit
        best_p = p0.copy()

        for _ in range(self.max_iterations):
            step_rec = self.step(system, mf)
            history.append(step_rec)

            cur_merit = step_rec.merit_value
            if cur_merit < best_merit:
                best_merit = cur_merit
                p_curr, _ = self._get_active_variables(system)
                best_p = p_curr.copy()

            if cur_merit < self.tolerance:
                converged = True
                break

            if abs(prev_merit - cur_merit) < self.tolerance * 0.1 and cur_merit < 1e-3:
                converged = True
                break

            prev_merit = cur_merit

        # Always restore best parameter state encountered
        self._apply_variables(system, best_p, var_names)
        p_final, _ = self._get_active_variables(system)
        final_params = {name: float(val) for name, val in zip(var_names, p_final)}
        final_merit = best_merit

        return OptimizationResult(
            success=(converged or final_merit < initial_merit),
            initial_merit=initial_merit,
            final_merit=final_merit,
            iterations=len(history) - 1,
            parameters=final_params,
            history=history,
            message=(
                f"Glatzel Adaptive DLS finished in {len(history)-1} iterations "
                f"(initial merit={initial_merit:.3e}, final merit={final_merit:.3e})"
            ),
            optimized_system=system,
        )
