"""
Optical profile data model and interface contract for 2D Optics Simulation.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple, Dict, Any, Union
import numpy as np
from scipy.interpolate import UnivariateSpline


class FitType(Enum):
    SPHERICAL = "Spherical"
    CONIC = "Conic"
    ASPHERIC = "Aspheric"
    CUBIC_SPLINE = "CubicSpline"
    SAMPLED = "SampledPoints"


# Alias for backward/forward compatibility
ProfileFitType = FitType


@dataclass
class OpticalProfile:
    """
    Extracted and fitted 2D optical surface profile z(r).
    Serves as the data contract between optics_app.cad and optics_app.engine.
    """
    surface_name: str = "Optical Surface"
    fit_type: FitType = FitType.CONIC
    radius_of_curvature: float = float('inf')  # R in mm (1/c)
    curvature: float = 0.0                     # c = 1/R in mm^-1
    conic_constant: float = 0.0                # k
    aspheric_coeffs: List[float] = field(default_factory=list)  # [a4, a6, a8, a10, ...]
    semi_diameter: float = 0.0                 # Clear aperture radius R_semi in mm
    vertex_offset_z: float = 0.0               # Axial shift applied to align vertex z(0) = 0
    rms_residual_um: float = 0.0               # RMS fit residual error in micrometers
    max_residual_um: float = 0.0               # Max absolute fit residual error in micrometers
    spline_object: Optional[UnivariateSpline] = None
    r_sampled: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    z_sampled: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))

    @property
    def asphere_coeffs(self) -> List[float]:
        """Property alias for aspheric_coeffs for test compatibility."""
        return self.aspheric_coeffs

    @asphere_coeffs.setter
    def asphere_coeffs(self, val: List[float]):
        self.aspheric_coeffs = val

    @property
    def clear_aperture(self) -> float:
        """Property alias for semi_diameter clear aperture."""
        return self.semi_diameter

    @clear_aperture.setter
    def clear_aperture(self, val: float):
        self.semi_diameter = float(val)

    def sag(self, r: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Calculates surface sag z(r) for given radial coordinate(s) r >= 0."""
        r_arr = np.asarray(r, dtype=np.float64)
        r_abs = np.abs(r_arr)

        if self.fit_type == FitType.CUBIC_SPLINE and self.spline_object is not None:
            z_val = self.spline_object(r_abs)
            return z_val.item() if np.ndim(r) == 0 else z_val

        c = self.curvature
        k = self.conic_constant

        if abs(c) < 1e-12:
            z_conic = np.zeros_like(r_abs)
        else:
            arg = 1.0 - (1.0 + k) * (c ** 2) * (r_abs ** 2)
            arg = np.maximum(arg, 0.0)  # Guard against negative values under sqrt
            denom = 1.0 + np.sqrt(arg)
            # Guard against zero division
            denom = np.where(abs(denom) < 1e-15, 1e-15, denom)
            z_conic = (c * (r_abs ** 2)) / denom

        z_asph = np.zeros_like(r_abs)
        if self.aspheric_coeffs:
            for idx, a_coeff in enumerate(self.aspheric_coeffs):
                power = 2 * (idx + 2)  # a4 -> r^4, a6 -> r^6, a8 -> r^8, a10 -> r^10
                z_asph += a_coeff * (r_abs ** power)

        total_z = z_conic + z_asph
        return total_z.item() if np.ndim(r) == 0 else total_z

    def sag_function(self, r: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Alias for sag(r) matching legacy/test interfaces."""
        return self.sag(r)

    def dz_dr(self, r: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Calculates radial slope derivative dz/dr at radial position r."""
        r_arr = np.asarray(r, dtype=np.float64)
        r_abs = np.abs(r_arr)

        if self.fit_type == FitType.CUBIC_SPLINE and self.spline_object is not None:
            deriv_spline = self.spline_object.derivative(n=1)
            dz = deriv_spline(r_abs)
            return dz.item() if np.ndim(r) == 0 else dz

        c = self.curvature
        k = self.conic_constant

        if abs(c) < 1e-12:
            dz_conic = np.zeros_like(r_abs)
        else:
            arg = 1.0 - (1.0 + k) * (c ** 2) * (r_abs ** 2)
            arg = np.maximum(arg, 1e-12)
            denom = 1.0 + np.sqrt(arg)
            dz_conic = (c * r_abs) / np.sqrt(arg)

        dz_asph = np.zeros_like(r_abs)
        if self.aspheric_coeffs:
            for idx, a_coeff in enumerate(self.aspheric_coeffs):
                power = 2 * (idx + 2)
                dz_asph += power * a_coeff * (r_abs ** (power - 1))

        total_dz = dz_conic + dz_asph
        return total_dz.item() if np.ndim(r) == 0 else total_dz

    def normal_vector(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        """
        Computes 3D normalized unit surface normal vector n = (nx, ny, nz) at (x, y, z).
        Points towards positive Z direction.
        """
        r = float(np.sqrt(x**2 + y**2))
        if r < 1e-12:
            return (0.0, 0.0, 1.0)

        slope = float(self.dz_dr(r))
        nx_unnorm = -slope * (x / r)
        ny_unnorm = -slope * (y / r)
        nz_unnorm = 1.0

        norm = float(np.sqrt(nx_unnorm**2 + ny_unnorm**2 + nz_unnorm**2))
        if norm < 1e-15:
            return (0.0, 0.0, 1.0)
        return (nx_unnorm / norm, ny_unnorm / norm, nz_unnorm / norm)

    def normal_function(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        """Alias for normal_vector(x, y, z) matching test interfaces."""
        return self.normal_vector(x, y, z)

    def intersect_ray(
        self,
        ray_origin: Union[Tuple[float, float, float], np.ndarray],
        ray_direction: Union[Tuple[float, float, float], np.ndarray],
        max_iters: int = 20,
        tol: float = 1e-10
    ) -> Tuple[float, np.ndarray, np.ndarray, bool]:
        """
        Newton-Raphson 3D Ray-Surface Intersection Solver.
        Ray equation: P(t) = P0 + t * v_hat

        Returns:
            t: Ray distance to intersection
            P_intersect: 3D point (x, y, z)
            n_hat: 3D unit normal vector at intersection point
            is_valid: bool indicating if ray distance t > 0, converged, and hit within clear aperture (r <= semi_diameter)
        """
        P0 = np.asarray(ray_origin, dtype=np.float64)
        v = np.asarray(ray_direction, dtype=np.float64)
        norm_v = np.linalg.norm(v)
        v = v / norm_v if norm_v > 1e-15 else np.array([0.0, 0.0, 1.0])

        # Initial estimate for t: intersect with vertex plane z = 0
        if abs(v[2]) > 1e-12:
            t = -P0[2] / v[2]
        else:
            t = 0.0

        converged = False
        for _ in range(max_iters):
            P = P0 + t * v
            r = float(np.sqrt(P[0]**2 + P[1]**2))
            z_surf = float(self.sag(r))

            g = P[2] - z_surf  # Objective function g(t) = 0
            if abs(g) < tol:
                converged = True
                break

            slope = float(self.dz_dr(r))
            dr_dt = (P[0] * v[0] + P[1] * v[1]) / r if r > 1e-12 else 0.0

            g_prime = v[2] - slope * dr_dt
            if abs(g_prime) < 1e-14:
                break

            t = t - g / g_prime

        P_intersect = P0 + t * v
        r_final = float(np.sqrt(P_intersect[0]**2 + P_intersect[1]**2))
        n_hat = np.array(self.normal_vector(P_intersect[0], P_intersect[1], P_intersect[2]))

        # Forward distance guard: t > 0
        forward_guard = (t > 0.0)

        # Clear aperture bound check: r <= semi_diameter
        within_aperture = (r_final <= self.semi_diameter) if self.semi_diameter > 0.0 else True

        is_valid = (converged or abs(P_intersect[2] - self.sag(r_final)) < 1e-4) and forward_guard and within_aperture

        return float(t), P_intersect, n_hat, is_valid

    def to_lde_dict(self) -> Dict[str, Any]:
        """Formats extracted optical profile into LDE spreadsheet table row fields."""
        return {
            "radius": self.radius_of_curvature,
            "conic": self.conic_constant,
            "asph_a4": self.aspheric_coeffs[0] if len(self.aspheric_coeffs) > 0 else 0.0,
            "asph_a6": self.aspheric_coeffs[1] if len(self.aspheric_coeffs) > 1 else 0.0,
            "asph_a8": self.aspheric_coeffs[2] if len(self.aspheric_coeffs) > 2 else 0.0,
            "asph_a10": self.aspheric_coeffs[3] if len(self.aspheric_coeffs) > 3 else 0.0,
            "semi_diameter": self.semi_diameter,
            "fit_type": self.fit_type.value,
            "rms_residual_um": self.rms_residual_um,
        }
