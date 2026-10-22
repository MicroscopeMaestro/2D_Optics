"""
3rd-Order Seidel Aberration Coefficients (S_I ... S_V) and Primary Chromatic Aberrations.
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Any, Dict
import math
import re
import numpy as np
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.utils.materials import get_refractive_index
from optics_app.engine.plm import is_plm, ensure_plm_config


@dataclass
class ParaxialSurfaceResult:
    """Paraxial ray trace and aberration contributions for a single surface."""
    surface_index: int
    surface_name: str
    radius: float
    thickness: float
    glass: str
    n: float           # Refractive index in space following surface (n')

    # Paraxial Marginal Ray parameters
    y: float           # Ray height at surface vertex
    u: float           # Angle before refraction
    u_prime: float     # Angle after refraction
    A: float           # Refraction invariant n(y c + u)

    # Paraxial Chief Ray parameters
    y_bar: float       # Chief ray height at surface vertex
    u_bar: float       # Angle before refraction
    u_bar_prime: float # Angle after refraction
    A_bar: float       # Refraction invariant n(y_bar c + u_bar)

    # Surface Aberration Contributions
    S_I: float = 0.0   # Spherical aberration
    S_II: float = 0.0  # Coma
    S_III: float = 0.0 # Astigmatism
    S_IV: float = 0.0  # Petzval curvature
    S_V: float = 0.0   # Distortion
    C_L: float = 0.0   # Longitudinal chromatic
    C_T: float = 0.0   # Transverse chromatic

    # Surface Cumulative Accumulations (running sum up to this surface)
    acc_S_I: float = 0.0
    acc_S_II: float = 0.0
    acc_S_III: float = 0.0
    acc_S_IV: float = 0.0
    acc_S_V: float = 0.0
    acc_C_L: float = 0.0
    acc_C_T: float = 0.0

    @property
    def W_040_um(self) -> float:
        """Wavefront Spherical Aberration OPD W040 in micrometers (µm)."""
        return (self.S_I / 8.0) * 1000.0

    @property
    def W_131_um(self) -> float:
        """Wavefront Coma OPD W131 in micrometers (µm)."""
        return (self.S_II / 2.0) * 1000.0

    @property
    def W_222_um(self) -> float:
        """Wavefront Astigmatism OPD W222 in micrometers (µm)."""
        return (self.S_III / 2.0) * 1000.0

    @property
    def W_220_um(self) -> float:
        """Wavefront Field Curvature OPD W220 in micrometers (µm)."""
        return ((self.S_III + self.S_IV) / 4.0) * 1000.0

    @property
    def W_311_um(self) -> float:
        """Wavefront Distortion OPD W311 in micrometers (µm)."""
        return (self.S_V / 2.0) * 1000.0

    @property
    def acc_W_040_um(self) -> float:
        return (self.acc_S_I / 8.0) * 1000.0

    @property
    def acc_W_131_um(self) -> float:
        return (self.acc_S_II / 2.0) * 1000.0

    @property
    def acc_W_222_um(self) -> float:
        return (self.acc_S_III / 2.0) * 1000.0

    @property
    def acc_W_220_um(self) -> float:
        return ((self.acc_S_III + self.acc_S_IV) / 4.0) * 1000.0

    @property
    def acc_W_311_um(self) -> float:
        return (self.acc_S_V / 2.0) * 1000.0


@dataclass
class LensAberrationResult:
    """
    3rd-Order Seidel Aberration and Chromatic contributions & cumulative accumulation for a single lens element.
    Modeled directly on Zemax OpticStudio Seidel Aberrations and Seidel Accumulation analysis.
    """
    lens_index: int                       # 1-based index (1, 2, ...)
    lens_name: str                        # Descriptive name (e.g. "Collect_ACL25416U", "Focus_AC254-040-C")
    surface_indices: List[int]            # Optical surfaces belonging to this lens [start, ..., end]
    lens_type: str                        # "Singlet", "Doublet", "Triplet", "Mirror", "Grating", "Filter/Plate", "Stop", "Element"
    glass_types: List[str]                # Glasses present in this lens element
    total_thickness: float                # Center / glass thickness of element (mm)

    # Individual Lens Contributions (Sum of member surface aberrations)
    S_I: float = 0.0                      # Spherical aberration contribution (mm)
    S_II: float = 0.0                     # Coma contribution (mm)
    S_III: float = 0.0                    # Astigmatism contribution (mm)
    S_IV: float = 0.0                     # Petzval field curvature contribution (mm)
    S_V: float = 0.0                      # Distortion contribution (mm)
    C_L: float = 0.0                      # Longitudinal chromatic aberration contribution (mm)
    C_T: float = 0.0                      # Transverse chromatic aberration contribution (mm)

    # Cumulative / Accumulated Aberrations (running sum up to and including this lens)
    acc_S_I: float = 0.0                  # Accumulated Spherical Aberration (mm)
    acc_S_II: float = 0.0                 # Accumulated Coma (mm)
    acc_S_III: float = 0.0                # Accumulated Astigmatism (mm)
    acc_S_IV: float = 0.0                 # Accumulated Petzval (mm)
    acc_S_V: float = 0.0                  # Accumulated Distortion (mm)
    acc_C_L: float = 0.0                  # Accumulated Longitudinal Color (mm)
    acc_C_T: float = 0.0                  # Accumulated Transverse Color (mm)

    @property
    def W_040_um(self) -> float:
        """Wavefront Spherical Aberration OPD W040 in micrometers (µm)."""
        return (self.S_I / 8.0) * 1000.0

    @property
    def W_131_um(self) -> float:
        """Wavefront Coma OPD W131 in micrometers (µm)."""
        return (self.S_II / 2.0) * 1000.0

    @property
    def W_222_um(self) -> float:
        """Wavefront Astigmatism OPD W222 in micrometers (µm)."""
        return (self.S_III / 2.0) * 1000.0

    @property
    def W_220_um(self) -> float:
        """Wavefront Field Curvature OPD W220 in micrometers (µm)."""
        return ((self.S_III + self.S_IV) / 4.0) * 1000.0

    @property
    def W_311_um(self) -> float:
        """Wavefront Distortion OPD W311 in micrometers (µm)."""
        return (self.S_V / 2.0) * 1000.0

    @property
    def acc_W_040_um(self) -> float:
        return (self.acc_S_I / 8.0) * 1000.0

    @property
    def acc_W_131_um(self) -> float:
        return (self.acc_S_II / 2.0) * 1000.0

    @property
    def acc_W_222_um(self) -> float:
        return (self.acc_S_III / 2.0) * 1000.0

    @property
    def acc_W_220_um(self) -> float:
        return ((self.acc_S_III + self.acc_S_IV) / 4.0) * 1000.0

    @property
    def acc_W_311_um(self) -> float:
        return (self.acc_S_V / 2.0) * 1000.0

    @property
    def surface_span_str(self) -> str:
        if not self.surface_indices:
            return ""
        if len(self.surface_indices) == 1:
            return f"S{self.surface_indices[0]}"
        return f"S{self.surface_indices[0]}-S{self.surface_indices[-1]}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lens_index": self.lens_index,
            "lens_name": self.lens_name,
            "surface_indices": list(self.surface_indices),
            "lens_type": self.lens_type,
            "glass_types": list(self.glass_types),
            "total_thickness": self.total_thickness,
            "S_I": self.S_I, "S_II": self.S_II, "S_III": self.S_III,
            "S_IV": self.S_IV, "S_V": self.S_V, "C_L": self.C_L, "C_T": self.C_T,
            "acc_S_I": self.acc_S_I, "acc_S_II": self.acc_S_II, "acc_S_III": self.acc_S_III,
            "acc_S_IV": self.acc_S_IV, "acc_S_V": self.acc_S_V,
            "acc_C_L": self.acc_C_L, "acc_C_T": self.acc_C_T,
            "W_040_um": self.W_040_um, "W_131_um": self.W_131_um, "W_222_um": self.W_222_um,
            "W_220_um": self.W_220_um, "W_311_um": self.W_311_um,
            "acc_W_040_um": self.acc_W_040_um, "acc_W_131_um": self.acc_W_131_um,
            "acc_W_222_um": self.acc_W_222_um, "acc_W_220_um": self.acc_W_220_um, "acc_W_311_um": self.acc_W_311_um,
        }


@dataclass
class SeidelAberrations:
    """3rd-Order Seidel Aberrations and Chromatic Aberrations summary container."""
    S_I: float = 0.0    # Total Spherical Aberration (mm)
    S_II: float = 0.0   # Total Coma (mm)
    S_III: float = 0.0  # Total Astigmatism (mm)
    S_IV: float = 0.0   # Total Petzval Field Curvature (mm)
    S_V: float = 0.0    # Total Distortion (mm)
    C_L: float = 0.0    # Total Longitudinal Chromatic Aberration (mm)
    C_T: float = 0.0    # Total Transverse Chromatic Aberration (mm)

    image_marginal_u: float = 0.0
    image_marginal_n: float = 1.0

    lagrange_invariant_H: float = 0.0
    surface_details: List[ParaxialSurfaceResult] = field(default_factory=list)
    lens_details: List[LensAberrationResult] = field(default_factory=list)

    @property
    def W_040_um(self) -> float:
        """Wavefront Spherical Aberration OPD W040 in micrometers (µm)."""
        return (self.S_I / 8.0) * 1000.0

    @property
    def W_131_um(self) -> float:
        """Wavefront Coma OPD W131 in micrometers (µm)."""
        return (self.S_II / 2.0) * 1000.0

    @property
    def W_222_um(self) -> float:
        """Wavefront Astigmatism OPD W222 in micrometers (µm)."""
        return (self.S_III / 2.0) * 1000.0

    @property
    def W_220_um(self) -> float:
        """Wavefront Field Curvature OPD W220 in micrometers (µm)."""
        return ((self.S_III + self.S_IV) / 4.0) * 1000.0

    @property
    def W_311_um(self) -> float:
        """Wavefront Distortion OPD W311 in micrometers (µm)."""
        return (self.S_V / 2.0) * 1000.0

    @property
    def W_CL_um(self) -> float:
        """Longitudinal chromatic aberration in micrometers (µm)."""
        return self.C_L * 1000.0

    @property
    def W_CT_um(self) -> float:
        """Transverse chromatic aberration in micrometers (µm)."""
        return self.C_T * 1000.0

    @property
    def TA_sph_um(self) -> float:
        """Transverse spherical aberration blur at paraxial focus: TA = S_I / (2 * n_k' * u_k') in µm."""
        denom = 2.0 * self.image_marginal_n * self.image_marginal_u
        return ((self.S_I / denom) * 1000.0) if abs(denom) > 1e-9 else 0.0

    @property
    def TA_coma_um(self) -> float:
        """Transverse coma blur at paraxial focus: TA = S_II / (2 * n_k' * u_k') in µm."""
        denom = 2.0 * self.image_marginal_n * self.image_marginal_u
        return ((self.S_II / denom) * 1000.0) if abs(denom) > 1e-9 else 0.0

    @property
    def TA_astig_um(self) -> float:
        """Transverse astigmatic blur at paraxial focus: TA = S_III / (2 * n_k' * u_k') in µm."""
        denom = 2.0 * self.image_marginal_n * self.image_marginal_u
        return ((self.S_III / denom) * 1000.0) if abs(denom) > 1e-9 else 0.0

    @property
    def final_accumulation(self) -> Dict[str, float]:
        """Final total aberration accumulation (mm) across all optical elements."""
        return {
            "S_I": self.S_I,
            "S_II": self.S_II,
            "S_III": self.S_III,
            "S_IV": self.S_IV,
            "S_V": self.S_V,
            "C_L": self.C_L,
            "C_T": self.C_T,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "S_I": self.S_I, "S_II": self.S_II, "S_III": self.S_III,
            "S_IV": self.S_IV, "S_V": self.S_V, "C_L": self.C_L, "C_T": self.C_T,
            "W_040_um": self.W_040_um, "W_131_um": self.W_131_um, "W_222_um": self.W_222_um,
            "W_220_um": self.W_220_um, "W_311_um": self.W_311_um,
            "TA_sph_um": self.TA_sph_um, "TA_coma_um": self.TA_coma_um, "TA_astig_um": self.TA_astig_um,
            "lagrange_invariant_H": self.lagrange_invariant_H,
            "surface_details": [s.to_dict() if hasattr(s, "to_dict") else str(s) for s in self.surface_details],
            "lens_details": [l.to_dict() for l in self.lens_details],
        }

    def generate_zemax_report(self, mode: str = "both") -> str:
        """Generate a complete text report formatted like Zemax OpticStudio."""
        return generate_zemax_seidel_report(self, mode=mode)

    @classmethod
    def fit_from_ray_fan(cls, py_coords: np.ndarray, ey_coords: np.ndarray) -> "SeidelAberrations":
        """
        Fit cubic polynomial EY(Py) = a3 * Py^3 + a2 * Py^2 + a1 * Py + a0
        a3 -> 4 * W040 -> S_I
        a2 -> 3 * W131 -> S_II
        a1 -> 2 * W222 -> S_III / S_IV
        a0 -> W311     -> S_V
        """
        if len(py_coords) < 4:
            return cls()
        coeffs = np.polyfit(py_coords, ey_coords, 3)  # [a3, a2, a1, a0]
        a3, a2, a1, a0 = coeffs
        return cls(
            S_I=float(a3),
            S_II=float(a2),
            S_III=float(a1),
            S_IV=float(0.5 * a1),
            S_V=float(a0)
        )


def _extract_surfaces(system: Any) -> List[Surface]:
    """Helper to extract list of surfaces from system, stack, or list."""
    if hasattr(system, "surfaces"):
        return list(system.surfaces)
    elif hasattr(system, "stack") and hasattr(system.stack, "surfaces"):
        return list(system.stack.surfaces)
    elif isinstance(system, list):
        return list(system)
    return []


def _get_effective_curvature(surf: Any, wavelength_um: float = 0.5875618, curr_n: float = 1.0) -> float:
    """
    Return the surface paraxial curvature c = 1 / R, taking into account
    any active phase optical power on a Texas Instruments PLM surface
    (e.g., Zernike defocus Z4, synthetic Fresnel lens, or chirped grating).
    """
    c = (1.0 / surf.radius) if (surf.radius != 0.0 and not np.isinf(surf.radius)) else 0.0
    if is_plm(surf):
        plm_cfg = ensure_plm_config(surf)
        st_up = str(getattr(surf, "mechanical_state", "ON") or "ON").upper()
        if st_up not in ("OFF", "STATE_0"):
            wl_mm = max(float(wavelength_um), 0.1) * 1e-3
            r_norm = max(float(plm_cfg.zernike.normalization_radius_mm), 1e-3)
            # 1. Zernike Defocus Z4 power: W = c4 * sqrt(3) * (2 r^2 - 1) -> dW/dr = 4*sqrt(3)*c4/R^2 * r
            c4 = float(plm_cfg.zernike.coeffs.get("z4_defocus", 0.0)) if plm_cfg.zernike.enabled else 0.0
            phi_z4 = - (4.0 * math.sqrt(3.0) * c4 * wl_mm) / (r_norm * r_norm)
            # 2. Fresnel lens power: 1 / f
            phi_fresnel = (1.0 / float(plm_cfg.pattern.fresnel_focal_length_mm)) if (
                plm_cfg.pattern.enabled
                and plm_cfg.pattern.pattern_type.upper() == "FRESNEL_LENS"
                and abs(float(plm_cfg.pattern.fresnel_focal_length_mm)) > 1e-3
            ) else 0.0
            # 3. Grating linear frequency chirp d(lines_per_mm)/du
            phi_chirp = - float(plm_cfg.grating.chirp_lines_per_mm2) * wl_mm if plm_cfg.grating.enabled else 0.0
            phi_total = phi_z4 + phi_fresnel + phi_chirp
            # In reflection (delta_n = -2 * curr_n), c_equiv = -phi / delta_n = phi / (2 * curr_n)
            c = c + phi_total / (2.0 * max(abs(curr_n), 1e-6))
    return c


def _find_stop_index(surfaces: List[Surface]) -> int:
    """
    Identifies the aperture stop surface index according to ISO/Zemax conventions:
    1. Check if any surface has is_stop == True or surface_type == SurfaceType.STOP.
    2. Check if any surface name contains 'STOP' or 'APERTURE' (case-insensitive).
    3. Fallback: If no explicit stop is defined, choose the first interior optical surface
       with a non-zero semi-diameter (or surface 1 if valid, else 0).
    """
    for idx, s in enumerate(surfaces):
        if getattr(s, "is_stop", False) or s.surface_type == SurfaceType.STOP:
            return idx
    for idx, s in enumerate(surfaces):
        name_up = (s.name or "").upper()
        if "STOP" in name_up or "APERTURE" in name_up:
            return idx
    return 1 if len(surfaces) > 1 else 0


def trace_paraxial_marginal_ray(
    system: Any,
    wavelength_um: float = 0.5875618
) -> List[Tuple[float, float, float, float]]:
    """
    Traces paraxial marginal ray across all surfaces.
    Returns list of tuples: (y_i, u_i, u_prime_i, A_i).
    """
    surfaces = _extract_surfaces(system)
    if not surfaces:
        return []

    # Identify object thickness (d0 == 0 or >= 1e8 or inf means object at infinity)
    d0 = surfaces[0].thickness if len(surfaces) > 0 else 0.0
    is_infinite_obj = (d0 == 0.0 or d0 >= 1e8 or np.isinf(d0))

    # 1. Trial Marginal Ray Trace
    y_trial = [0.0] * len(surfaces)
    u_trial = [0.0] * len(surfaces)
    u_prime_trial = [0.0] * len(surfaces)

    if is_infinite_obj:
        y_trial[0] = 1.0
        u_trial[0] = 0.0
    else:
        y_trial[0] = 0.0
        u_trial[0] = 0.1

    curr_n = 1.0
    for i in range(len(surfaces)):
        surf = surfaces[i]
        if surf.glass and surf.glass.upper() in ["MIRROR", "REFLECTOR", "MIRR"]:
            next_n = -curr_n
        elif surf.glass:
            next_n = get_refractive_index(surf.glass, wavelength_um)
        else:
            next_n = curr_n

        c = _get_effective_curvature(surf, wavelength_um, curr_n)

        if i == 0:
            y_i = y_trial[0]
            u_i = u_trial[0]
        else:
            u_i = u_prime_trial[i - 1]
            y_i = y_trial[i - 1] + surfaces[i - 1].thickness * u_i
            u_trial[i] = u_i

        y_trial[i] = y_i

        if surf.surface_type in (SurfaceType.DETECTOR, SurfaceType.STOP) or next_n == curr_n:
            u_p = u_i
        else:
            u_p = (curr_n * u_i - y_i * c * (next_n - curr_n)) / next_n

        u_prime_trial[i] = u_p
        curr_n = next_n

    # 2. Stop Normalization
    stop_idx = _find_stop_index(surfaces)
    target_semi = surfaces[stop_idx].semi_diameter if surfaces[stop_idx].semi_diameter > 0 else 12.5
    trial_stop_y = y_trial[stop_idx]
    scale = (target_semi / trial_stop_y) if abs(trial_stop_y) > 1e-12 else 1.0

    # Physical conjugate bounding check:
    # If finite object distance d0 > 0, ensure scaling does not overfill entrance element semi-diameter
    if not is_infinite_obj and len(surfaces) > 1 and surfaces[1].semi_diameter > 0:
        max_y1 = surfaces[1].semi_diameter
        y1_scaled = abs(y_trial[1] * scale)
        if y1_scaled > max_y1:
            scale *= (max_y1 / y1_scaled)

    # Check system_config or source_config for aperture limits
    sys_cfg = getattr(system, "system_config", None)
    if sys_cfg is not None:
        ap_val = getattr(sys_cfg, "aperture_value", 0.0)
        ap_type = getattr(sys_cfg, "aperture_type", None)
        ap_type_str = getattr(ap_type, "value", str(ap_type)) if ap_type else ""
        if "NA" in ap_type_str.upper() and ap_val > 0 and not is_infinite_obj:
            max_u0 = float(ap_val)
            if abs(u_trial[0] * scale) > max_u0:
                scale = max_u0 / abs(u_trial[0])
        elif ("ENTRANCE" in ap_type_str.upper() or "EPD" in ap_type_str.upper()) and ap_val > 0 and is_infinite_obj:
            target_epd_semi = float(ap_val) / 2.0
            if abs(y_trial[0] * scale) > target_epd_semi:
                scale = target_epd_semi / abs(y_trial[0])

    src_cfg = getattr(system, "source_config", None)
    if src_cfg is not None:
        na = getattr(src_cfg, "numerical_aperture", None) if not isinstance(src_cfg, dict) else src_cfg.get("numerical_aperture")
        beam_r = getattr(src_cfg, "beam_radius", None) if not isinstance(src_cfg, dict) else src_cfg.get("beam_radius")
        if na is not None and float(na) > 0 and not is_infinite_obj:
            if abs(u_trial[0] * scale) > float(na):
                scale = float(na) / abs(u_trial[0])
        if beam_r is not None and float(beam_r) > 0 and is_infinite_obj:
            if abs(y_trial[0] * scale) > float(beam_r):
                scale = float(beam_r) / abs(y_trial[0])

    # 3. Final Marginal Ray parameters
    results = []
    curr_n = 1.0
    for i in range(len(surfaces)):
        surf = surfaces[i]
        if surf.glass and surf.glass.upper() in ["MIRROR", "REFLECTOR", "MIRR"]:
            next_n = -curr_n
        elif surf.glass:
            next_n = get_refractive_index(surf.glass, wavelength_um)
        else:
            next_n = curr_n

        c = _get_effective_curvature(surf, wavelength_um, curr_n)
        y_i = y_trial[i] * scale
        u_i = u_trial[i] * scale
        u_p = u_prime_trial[i] * scale
        A_i = curr_n * (y_i * c + u_i)
        results.append((float(y_i), float(u_i), float(u_p), float(A_i)))
        curr_n = next_n

    return results


def trace_paraxial_chief_ray(
    system: Any,
    field_angle_deg: float = 1.0,
    object_height: float = 1.0,
    wavelength_um: float = 0.5875618
) -> List[Tuple[float, float, float, float]]:
    """
    Traces paraxial chief ray across all surfaces.
    Returns list of tuples: (y_bar_i, u_bar_i, u_bar_prime_i, A_bar_i).
    """
    surfaces = _extract_surfaces(system)
    if not surfaces:
        return []

    # Get marginal ray to find stop ratio
    marginal_trace = trace_paraxial_marginal_ray(system, wavelength_um)

    stop_idx = _find_stop_index(surfaces)
    y_stop_marginal = marginal_trace[stop_idx][0] if len(marginal_trace) > stop_idx else 1.0

    d0 = surfaces[0].thickness if len(surfaces) > 0 else 0.0
    is_infinite_obj = (d0 == 0.0 or d0 >= 1e8 or np.isinf(d0))

    # 1. Trace trial chief ray
    u_bar_0 = np.tan(np.radians(field_angle_deg)) if is_infinite_obj else 0.0
    y_bar_0 = 0.0 if is_infinite_obj else object_height

    y_bar_trial = [0.0] * len(surfaces)
    u_bar_trial = [0.0] * len(surfaces)
    u_bar_prime_trial = [0.0] * len(surfaces)

    y_bar_trial[0] = y_bar_0
    u_bar_trial[0] = u_bar_0

    curr_n = 1.0
    for i in range(len(surfaces)):
        surf = surfaces[i]
        if surf.glass and surf.glass.upper() in ["MIRROR", "REFLECTOR", "MIRR"]:
            next_n = -curr_n
        elif surf.glass:
            next_n = get_refractive_index(surf.glass, wavelength_um)
        else:
            next_n = curr_n

        c = _get_effective_curvature(surf, wavelength_um, curr_n)

        if i == 0:
            yb_i = y_bar_trial[0]
            ub_i = u_bar_trial[0]
        else:
            ub_i = u_bar_prime_trial[i - 1]
            yb_i = y_bar_trial[i - 1] + surfaces[i - 1].thickness * ub_i
            u_bar_trial[i] = ub_i

        y_bar_trial[i] = yb_i

        if surf.surface_type in (SurfaceType.DETECTOR, SurfaceType.STOP) or next_n == curr_n:
            ub_p = ub_i
        else:
            ub_p = (curr_n * ub_i - yb_i * c * (next_n - curr_n)) / next_n

        u_bar_prime_trial[i] = ub_p
        curr_n = next_n

    # Stop adjustment linear scaling
    trial_stop_yb = y_bar_trial[stop_idx]

    if is_infinite_obj:
        # True y_bar_1 = - (trial_stop_yb / y_stop_marginal) * y_marginal_1
        # For chief ray: true y_bar_i = y_bar_trial[i] - (trial_stop_yb / y_stop_marginal) * y_marginal_i
        shift_ratio = (trial_stop_yb / y_stop_marginal) if abs(y_stop_marginal) > 1e-12 else 0.0
        results = []
        curr_n = 1.0
        for i in range(len(surfaces)):
            surf = surfaces[i]
            if surf.glass and surf.glass.upper() in ["MIRROR", "REFLECTOR", "MIRR"]:
                next_n = -curr_n
            elif surf.glass:
                next_n = get_refractive_index(surf.glass, wavelength_um)
            else:
                next_n = curr_n
            c = _get_effective_curvature(surf, wavelength_um, curr_n)

            ym_i, um_i, um_p, _ = marginal_trace[i]
            yb_i = y_bar_trial[i] - shift_ratio * ym_i
            ub_i = u_bar_trial[i] - shift_ratio * um_i
            ub_p = u_bar_prime_trial[i] - shift_ratio * um_p
            A_bar_i = curr_n * (yb_i * c + ub_i)

            results.append((float(yb_i), float(ub_i), float(ub_p), float(A_bar_i)))
            curr_n = next_n
        return results
    else:
        # Finite object distance
        corr_ub0 = -(trial_stop_yb / y_stop_marginal) * marginal_trace[0][1] if abs(y_stop_marginal) > 1e-12 else 0.0
        u_bar_trial[0] = corr_ub0

        results = []
        curr_n = 1.0
        for i in range(len(surfaces)):
            surf = surfaces[i]
            if surf.glass and surf.glass.upper() in ["MIRROR", "REFLECTOR", "MIRR"]:
                next_n = -curr_n
            elif surf.glass:
                next_n = get_refractive_index(surf.glass, wavelength_um)
            else:
                next_n = curr_n
            c = _get_effective_curvature(surf, wavelength_um, curr_n)

            if i == 0:
                yb_i = y_bar_0
                ub_i = u_bar_trial[0]
            else:
                ub_i = u_bar_prime_trial[i - 1]
                yb_i = y_bar_trial[i - 1] + surfaces[i - 1].thickness * ub_i
                u_bar_trial[i] = ub_i

            y_bar_trial[i] = yb_i

            if surf.surface_type in (SurfaceType.DETECTOR, SurfaceType.STOP) or next_n == curr_n:
                ub_p = ub_i
            else:
                ub_p = (curr_n * ub_i - yb_i * c * (next_n - curr_n)) / next_n

            u_bar_prime_trial[i] = ub_p
            A_bar_i = curr_n * (yb_i * c + ub_i)
            results.append((float(yb_i), float(ub_i), float(ub_p), float(A_bar_i)))
            curr_n = next_n
        return results


def calculate_seidel_coefficients(
    system: Any,
    wavelength_um: float = 0.5875618,
    field_angle_deg: float = 1.0,
    object_height: float = 1.0
) -> SeidelAberrations:
    """
    Computes 3rd-order Seidel aberration sums (S_I ... S_V) and primary
    chromatic aberrations (C_L, C_T) for the given OpticalSystem.
    """
    surfaces = _extract_surfaces(system)
    if not surfaces:
        return SeidelAberrations()

    marginal_trace = trace_paraxial_marginal_ray(system, wavelength_um)
    chief_trace = trace_paraxial_chief_ray(system, field_angle_deg, object_height, wavelength_um)

    # Compute Lagrange Invariant H at surface 0
    y0, u0, _, _ = marginal_trace[0]
    yb0, ub0, _, _ = chief_trace[0]
    H_val = float(1.0 * (y0 * ub0 - yb0 * u0))

    tot_S_I = 0.0
    tot_S_II = 0.0
    tot_S_III = 0.0
    tot_S_IV = 0.0
    tot_S_V = 0.0
    tot_C_L = 0.0
    tot_C_T = 0.0

    surface_details = []

    wl_F = 0.4861327  # F-line (blue)
    wl_C = 0.6562725  # C-line (red)

    curr_n = 1.0
    curr_dn = 0.0

    for i in range(len(surfaces)):
        surf = surfaces[i]
        glass_name = surf.glass or "AIR"
        upper_glass = glass_name.upper()
        if surf.glass and upper_glass in ["MIRROR", "REFLECTOR", "MIRR"]:
            next_n = -curr_n
            n_F = next_n
            n_C = next_n
            next_dn = 0.0
        elif not surf.glass:
            next_n = curr_n
            n_F = next_n
            n_C = next_n
            next_dn = 0.0
        elif upper_glass in ["AIR", "VACUUM"]:
            next_n = get_refractive_index(surf.glass, wavelength_um)
            n_F = next_n
            n_C = next_n
            next_dn = 0.0
        else:
            next_n = get_refractive_index(surf.glass, wavelength_um)
            n_F = get_refractive_index(surf.glass, wl_F)
            n_C = get_refractive_index(surf.glass, wl_C)
            next_dn = n_F - n_C

        c = _get_effective_curvature(surf, wavelength_um, curr_n)

        y_i, u_i, u_p_i, A_i = marginal_trace[i]
        yb_i, ub_i, ub_p_i, A_bar_i = chief_trace[i]

        delta_u_over_n = (u_p_i / next_n) - (u_i / curr_n) if (curr_n != 0 and next_n != 0) else 0.0
        delta_one_over_n = (1.0 / next_n) - (1.0 / curr_n) if (curr_n != 0 and next_n != 0) else 0.0

        # Spherical contribution
        s_I_sph = - (A_i**2) * y_i * delta_u_over_n
        s_II_sph = - A_i * A_bar_i * y_i * delta_u_over_n
        s_III_sph = - (A_bar_i**2) * y_i * delta_u_over_n
        s_IV_sph = - (H_val**2) * c * delta_one_over_n

        if abs(A_i) > 1e-12:
            s_V_sph = - (((A_bar_i**3) / A_i) * y_i * delta_u_over_n + (A_bar_i / A_i) * H_val * c * delta_one_over_n)
        else:
            s_V_sph = 0.0

        # Aspheric contribution
        a4 = surf.asphere_coeffs[0] if (surf.asphere_coeffs and len(surf.asphere_coeffs) > 0) else 0.0
        kappa = (1.0 + surf.conic) * (c**3) + 8.0 * a4
        delta_n = next_n - curr_n

        s_I_asph = - kappa * (y_i**4) * delta_n
        s_II_asph = - kappa * (y_i**3) * yb_i * delta_n
        s_III_asph = - kappa * (y_i**2) * (yb_i**2) * delta_n
        s_IV_asph = 0.0
        s_V_asph = - kappa * y_i * (yb_i**3) * delta_n

        s_I = s_I_sph + s_I_asph
        s_II = s_II_sph + s_II_asph
        s_III = s_III_sph + s_III_asph
        s_IV = s_IV_sph + s_IV_asph
        s_V = s_V_sph + s_V_asph

        # Check for active Phase Light Modulator (PLM) wavefront phase contribution
        if is_plm(surf):
            plm_cfg = ensure_plm_config(surf)
            st_up = str(getattr(surf, "mechanical_state", "ON") or "ON").upper()
            if st_up not in ("OFF", "STATE_0") and plm_cfg.zernike.enabled:
                wl_mm = max(float(wavelength_um), 0.1) * 1e-3
                r_norm = max(float(plm_cfg.zernike.normalization_radius_mm), 1e-3)
                inv_r4 = 1.0 / (r_norm**4)
                zcoeffs = plm_cfg.zernike.coeffs or {}

                # Spherical aberration Z11 = sqrt(5) * (6 r^4 - 6 r^2 + 1) -> S_I = 8 * W040
                c11 = float(zcoeffs.get("z11_spherical", 0.0))
                s_I_plm = 48.0 * math.sqrt(5.0) * c11 * wl_mm * inv_r4 * (y_i**4)

                # Coma Z7, Z8: S_II = 2 * W131
                c7 = float(zcoeffs.get("z7_coma_vertical", 0.0))
                c8 = float(zcoeffs.get("z8_coma_horizontal", 0.0))
                c_coma = math.sqrt(c7**2 + c8**2) * (1.0 if (c8 >= 0 and c7 >= 0) else -1.0)
                s_II_plm = 6.0 * math.sqrt(8.0) * c_coma * wl_mm * inv_r4 * (y_i**3) * yb_i

                # Astigmatism Z5, Z6: S_III = 2 * W222
                c5 = float(zcoeffs.get("z5_astig_oblique", 0.0))
                c6 = float(zcoeffs.get("z6_astig_vertical", 0.0))
                c_astig = math.sqrt(c5**2 + c6**2) * (1.0 if c6 >= 0 else -1.0)
                s_III_plm = 2.0 * math.sqrt(6.0) * c_astig * wl_mm * inv_r4 * (y_i**2) * (yb_i**2)

                s_I += s_I_plm
                s_II += s_II_plm
                s_III += s_III_plm

        # Chromatic Aberration contribution
        disp_term = (next_dn / next_n) - (curr_dn / curr_n) if (curr_n != 0 and next_n != 0) else 0.0
        c_L = A_i * y_i * disp_term
        c_T = A_bar_i * y_i * disp_term

        # Only sum up inner optical surfaces (not object or detector)
        if surf.surface_type not in (SurfaceType.OBJECT, SurfaceType.DETECTOR):
            tot_S_I += s_I
            tot_S_II += s_II
            tot_S_III += s_III
            tot_S_IV += s_IV
            tot_S_V += s_V
            tot_C_L += c_L
            tot_C_T += c_T

        surface_details.append(ParaxialSurfaceResult(
            surface_index=i,
            surface_name=surf.name,
            radius=surf.radius,
            thickness=surf.thickness,
            glass=glass_name,
            n=float(next_n),
            y=float(y_i),
            u=float(u_i),
            u_prime=float(u_p_i),
            A=float(A_i),
            y_bar=float(yb_i),
            u_bar=float(ub_i),
            u_bar_prime=float(ub_p_i),
            A_bar=float(A_bar_i),
            S_I=float(s_I),
            S_II=float(s_II),
            S_III=float(s_III),
            S_IV=float(s_IV),
            S_V=float(s_V),
            C_L=float(c_L),
            C_T=float(c_T)
        ))

        curr_n = next_n
        curr_dn = next_dn

    # 1. Compute surface-level running accumulations
    cum_S_I = 0.0
    cum_S_II = 0.0
    cum_S_III = 0.0
    cum_S_IV = 0.0
    cum_S_V = 0.0
    cum_C_L = 0.0
    cum_C_T = 0.0
    for surf_res in surface_details:
        surf_type = surfaces[surf_res.surface_index].surface_type
        if surf_type not in (SurfaceType.OBJECT, SurfaceType.DETECTOR):
            cum_S_I += surf_res.S_I
            cum_S_II += surf_res.S_II
            cum_S_III += surf_res.S_III
            cum_S_IV += surf_res.S_IV
            cum_S_V += surf_res.S_V
            cum_C_L += surf_res.C_L
            cum_C_T += surf_res.C_T
        surf_res.acc_S_I = float(cum_S_I)
        surf_res.acc_S_II = float(cum_S_II)
        surf_res.acc_S_III = float(cum_S_III)
        surf_res.acc_S_IV = float(cum_S_IV)
        surf_res.acc_S_V = float(cum_S_V)
        surf_res.acc_C_L = float(cum_C_L)
        surf_res.acc_C_T = float(cum_C_T)

    # 2. Group surfaces into physical lenses & optical elements
    lens_groups = group_surfaces_into_lenses(surfaces)
    surf_map = {res.surface_index: res for res in surface_details}

    lens_details: List[LensAberrationResult] = []
    lens_acc_S_I = 0.0
    lens_acc_S_II = 0.0
    lens_acc_S_III = 0.0
    lens_acc_S_IV = 0.0
    lens_acc_S_V = 0.0
    lens_acc_C_L = 0.0
    lens_acc_C_T = 0.0

    for idx, grp in enumerate(lens_groups, start=1):
        idxs = grp["indices"]
        l_s_I = sum(surf_map[k].S_I for k in idxs if k in surf_map)
        l_s_II = sum(surf_map[k].S_II for k in idxs if k in surf_map)
        l_s_III = sum(surf_map[k].S_III for k in idxs if k in surf_map)
        l_s_IV = sum(surf_map[k].S_IV for k in idxs if k in surf_map)
        l_s_V = sum(surf_map[k].S_V for k in idxs if k in surf_map)
        l_c_L = sum(surf_map[k].C_L for k in idxs if k in surf_map)
        l_c_T = sum(surf_map[k].C_T for k in idxs if k in surf_map)

        lens_acc_S_I += l_s_I
        lens_acc_S_II += l_s_II
        lens_acc_S_III += l_s_III
        lens_acc_S_IV += l_s_IV
        lens_acc_S_V += l_s_V
        lens_acc_C_L += l_c_L
        lens_acc_C_T += l_c_T

        lens_details.append(LensAberrationResult(
            lens_index=idx,
            lens_name=grp["name"],
            surface_indices=idxs,
            lens_type=grp["type"],
            glass_types=grp["glasses"],
            total_thickness=float(grp["thickness"]),
            S_I=float(l_s_I),
            S_II=float(l_s_II),
            S_III=float(l_s_III),
            S_IV=float(l_s_IV),
            S_V=float(l_s_V),
            C_L=float(l_c_L),
            C_T=float(l_c_T),
            acc_S_I=float(lens_acc_S_I),
            acc_S_II=float(lens_acc_S_II),
            acc_S_III=float(lens_acc_S_III),
            acc_S_IV=float(lens_acc_S_IV),
            acc_S_V=float(lens_acc_S_V),
            acc_C_L=float(lens_acc_C_L),
            acc_C_T=float(lens_acc_C_T),
        ))

    img_u = marginal_trace[-1][2] if marginal_trace else 0.1
    img_n = curr_n

    return SeidelAberrations(
        S_I=float(tot_S_I),
        S_II=float(tot_S_II),
        S_III=float(tot_S_III),
        S_IV=float(tot_S_IV),
        S_V=float(tot_S_V),
        C_L=float(tot_C_L),
        C_T=float(tot_C_T),
        image_marginal_u=float(img_u),
        image_marginal_n=float(img_n),
        lagrange_invariant_H=float(H_val),
        surface_details=surface_details,
        lens_details=lens_details
    )


def group_surfaces_into_lenses(surfaces: List[Surface]) -> List[Dict[str, Any]]:
    """
    Groups sequential optical surfaces into distinct physical lenses and optical elements
    (Singlets, Doublets, Triplets, Mirrors, Gratings, Stops, Filters/Plates).

    Identifies elements using:
    1. Explicit group_name or element_id if assigned.
    2. Naming patterns (e.g. prefixes before _S1, _S2, _Surf1, etc.).
    3. Sequential physical boundaries (air -> glass -> ... -> air transitions).
    4. Special optical components (reflective mirrors, diffraction gratings, DMDs, stops).
    """
    groups: List[Dict[str, Any]] = []
    i = 0
    n = len(surfaces)
    while i < n:
        s = surfaces[i]
        # Skip Object (surface 0) and Detector/Image (last surface)
        if s.surface_type in (SurfaceType.OBJECT, SurfaceType.DETECTOR) or i == 0 or i == n - 1:
            i += 1
            continue

        glass_u = (s.glass or "").strip().upper()
        is_mirror = glass_u in ["MIRROR", "REFLECTOR", "MIRR"] or s.surface_type == SurfaceType.MIRROR
        is_grating = s.surface_type == SurfaceType.GRATING or getattr(s, "grating_lines_per_mm", 0) > 0

        # 1. Reflective Elements / Mirrors / Gratings / DMDs
        if is_mirror or is_grating:
            if is_grating:
                elem_type = "Grating"
            elif "DMD" in s.name.upper():
                elem_type = "DMD"
            else:
                elem_type = "Mirror"
            groups.append({
                "name": s.name or f"{elem_type} {len(groups) + 1}",
                "type": elem_type,
                "indices": [i],
                "glasses": [s.glass],
                "thickness": float(s.thickness)
            })
            i += 1
            continue

        # 2. Pure Air Spaces / Coordinate Breaks / Stops
        is_air = glass_u in ["", "AIR", "VACUUM", "NONE"]
        if is_air and (s.radius == 0.0 or abs(s.radius) > 1e10 or np.isinf(s.radius)):
            if s.surface_type == SurfaceType.STOP or "STOP" in s.name.upper():
                groups.append({
                    "name": s.name or "Aperture Stop",
                    "type": "Stop",
                    "indices": [i],
                    "glasses": ["AIR"],
                    "thickness": float(s.thickness)
                })
            i += 1
            continue

        # 3. Refractive Lens Element (starts when entering solid glass medium)
        cur_indices = [i]
        cur_glasses = [s.glass]
        tot_glass_th = float(s.thickness) if not is_air else 0.0
        j = i + 1

        # Check explicit group or element_id
        explicit_group = s.group_name or (str(s.element_id) if s.element_id is not None else "")
        if explicit_group:
            while j < n - 1:
                next_s = surfaces[j]
                next_grp = next_s.group_name or (str(next_s.element_id) if next_s.element_id is not None else "")
                if next_grp == explicit_group:
                    cur_indices.append(j)
                    cur_glasses.append(next_s.glass)
                    tot_glass_th += float(next_s.thickness)
                    j += 1
                else:
                    break
        else:
            # Physical boundary tracing: continue while intermediate surfaces are within glass
            while j < n - 1:
                next_s = surfaces[j]
                next_g = (next_s.glass or "").strip().upper()
                if next_g in ["MIRROR", "REFLECTOR", "MIRR"] or next_s.surface_type in (
                    SurfaceType.MIRROR, SurfaceType.GRATING, SurfaceType.OBJECT, SurfaceType.DETECTOR
                ):
                    break
                cur_indices.append(j)
                cur_glasses.append(next_s.glass)
                tot_glass_th += float(next_s.thickness)
                j += 1
                if next_g in ["", "AIR", "VACUUM", "NONE"]:
                    # Outgoing medium is AIR -> lens rear surface reached!
                    break

        # Base lens name resolution
        surfs_in_lens = [surfaces[idx] for idx in cur_indices]
        base_name = ""
        for pattern in [
            r"^(.*?)(?:_rev)?(?:_S\d+|_Surf\d+|_Front|_Back|_Rear|_s\d+|_r\d+| #\d+)$",
            r"^(.*?)(?:_rev)?$"
        ]:
            m = re.match(pattern, surfs_in_lens[0].name, re.IGNORECASE)
            if m and m.group(1):
                base_name = m.group(1).strip()
                break
        if not base_name:
            base_name = surfs_in_lens[0].group_name or f"Lens {len(groups) + 1}"

        # Lens classification
        num_faces = len(cur_indices)
        is_flat = all(
            (surfaces[idx].radius == 0.0 or abs(surfaces[idx].radius) > 1e10 or np.isinf(surfaces[idx].radius))
            for idx in cur_indices
        )
        has_asphere = any(
            (surfaces[idx].conic != 0.0 or bool(surfaces[idx].asphere_coeffs))
            for idx in cur_indices
        )

        if is_flat:
            ltype = "Filter/Plate"
        elif num_faces == 2:
            ltype = "Aspheric Singlet" if has_asphere else "Singlet"
        elif num_faces == 3:
            ltype = "Aspheric Doublet" if has_asphere else "Doublet"
        elif num_faces == 4:
            ltype = "Triplet"
        else:
            ltype = f"Lens ({num_faces} surfs)"

        groups.append({
            "name": base_name,
            "type": ltype,
            "indices": cur_indices,
            "glasses": [g for g in cur_glasses if g.upper() not in ["", "AIR", "VACUUM", "NONE"]],
            "thickness": tot_glass_th
        })
        i = j

    return groups


def generate_zemax_seidel_report(seidel: SeidelAberrations, mode: str = "both") -> str:
    """
    Format a complete text report following standard Zemax OpticStudio conventions.
    Shows surface contributions, lens contributions, and progressive accumulation
    up to the final aberrations accumulation.
    """
    lines: List[str] = []
    lines.append("=" * 105)
    lines.append("                             ZEMAX OPTICSTUDIO SEIDEL ABERRATION ANALYSIS")
    lines.append("=" * 105)
    lines.append(f"Lagrange Invariant (H):            {seidel.lagrange_invariant_H:+.6e} mm-rad")
    lines.append(f"Total Optical Surfaces Analyzed:   {len(seidel.surface_details)}")
    lines.append(f"Total Optical Lenses / Components: {len(seidel.lens_details)}")
    lines.append("")

    if mode in ("both", "surface") and seidel.surface_details:
        lines.append("-" * 105)
        lines.append("SURFACE-BY-SURFACE SEIDEL CONTRIBUTIONS & RUNNING ACCUMULATION (mm & µm)")
        lines.append("-" * 105)
        lines.append(f"{'Surf':<5} {'Name':<22} {'Glass':<10} {'Radius':>9} {'Thick':>8} {'S_I(Sph)':>10} {'S_II(Coma)':>10} {'S_III(Ast)':>10} {'S_IV(Petz)':>10} {'S_V(Dist)':>10}")
        lines.append("-" * 105)
        for s in seidel.surface_details:
            r_str = f"{s.radius:9.2f}" if (abs(s.radius) < 1e8 and s.radius != 0.0) else "   Infinity"
            lines.append(f"{s.surface_index:<5d} {s.surface_name[:21]:<22} {s.glass[:9]:<10} {r_str:>9} {s.thickness:8.3f} {s.S_I:10.4f} {s.S_II:10.4f} {s.S_III:10.4f} {s.S_IV:10.4f} {s.S_V:10.4f}")
            lines.append(f"      {'[Accumulated to Surface ' + str(s.surface_index) + ']':<31} {'':>18} {s.acc_S_I:10.4f} {s.acc_S_II:10.4f} {s.acc_S_III:10.4f} {s.acc_S_IV:10.4f} {s.acc_S_V:10.4f}")
            lines.append(
                f"      Chromatic: C_L={s.C_L:+.5f} mm ({s.C_L*1000.0:+.2f} µm) [acc: {s.acc_C_L*1000.0:+.2f} µm] | "
                f"C_T={s.C_T:+.5f} mm ({s.C_T*1000.0:+.2f} µm) [acc: {s.acc_C_T*1000.0:+.2f} µm]"
            )
        lines.append("")

    if mode in ("both", "lens") and seidel.lens_details:
        lines.append("-" * 105)
        lines.append("LENS-BY-LENS SEIDEL ABERRATIONS & PROGRESSIVE ACCUMULATION (mm & µm) - ZEMAX STYLE")
        lines.append("-" * 105)
        lines.append(f"{'Lens':<5} {'Lens Name':<22} {'Type':<12} {'Surfaces':<9} {'Thickness':>9} {'S_I(Sph)':>10} {'S_II(Coma)':>10} {'S_III(Ast)':>10} {'S_IV(Petz)':>10} {'S_V(Dist)':>10}")
        lines.append("-" * 105)
        for l in seidel.lens_details:
            lines.append(f"{l.lens_index:<5d} {l.lens_name[:21]:<22} {l.lens_type[:11]:<12} {l.surface_span_str:<9} {l.total_thickness:9.3f} {l.S_I:10.4f} {l.S_II:10.4f} {l.S_III:10.4f} {l.S_IV:10.4f} {l.S_V:10.4f}")
            lines.append(f"      [ACCUMULATED after {l.lens_name[:18]}]{'':>8} {l.acc_S_I:10.4f} {l.acc_S_II:10.4f} {l.acc_S_III:10.4f} {l.acc_S_IV:10.4f} {l.acc_S_V:10.4f}")
            lines.append(
                f"      Chromatic: C_L (Axial)={l.C_L:+.4f} mm ({l.C_L*1000.0:+.2f} µm, acc: {l.acc_C_L*1000.0:+.2f} µm) | "
                f"C_T (Lateral)={l.C_T:+.4f} mm ({l.C_T*1000.0:+.2f} µm, acc: {l.acc_C_T*1000.0:+.2f} µm)"
            )
            lines.append("." * 105)
        lines.append("")

    lines.append("=" * 105)
    lines.append("FINAL SYSTEM ABERRATIONS ACCUMULATION (mm & µm) [IMAGE / DETECTOR PLANE]:")
    lines.append("=" * 105)
    lines.append(f"  Spherical Aberration (S_I):        {seidel.S_I:+14.6f} mm ({seidel.S_I*1000.0:+10.2f} µm)")
    lines.append(f"  Coma (S_II):                       {seidel.S_II:+14.6f} mm ({seidel.S_II*1000.0:+10.2f} µm)")
    lines.append(f"  Astigmatism (S_III):               {seidel.S_III:+14.6f} mm ({seidel.S_III*1000.0:+10.2f} µm)")
    lines.append(f"  Petzval Field Curvature (S_IV):    {seidel.S_IV:+14.6f} mm ({seidel.S_IV*1000.0:+10.2f} µm)")
    lines.append(f"  Distortion (S_V):                  {seidel.S_V:+14.6f} mm ({seidel.S_V*1000.0:+10.2f} µm)")
    lines.append(f"  Longitudinal Chromatic (C_L):      {seidel.C_L:+14.6f} mm ({seidel.C_L*1000.0:+10.2f} µm)")
    lines.append(f"  Transverse Chromatic (C_T):        {seidel.C_T:+14.6f} mm ({seidel.C_T*1000.0:+10.2f} µm)")
    lines.append("-" * 105)
    lines.append("WAVEFRONT ABERRATION OPD (PUPIL EDGE ρ=1.0) & TRANSVERSE RAY BLUR (PARAXIAL FOCUS):")
    lines.append("-" * 105)
    lines.append(f"  Wavefront Spherical (W040 = S_I / 8):           {seidel.W_040_um:+10.4f} µm  ({seidel.W_040_um / 0.5875618:+8.3f} waves λ)")
    lines.append(f"  Wavefront Coma (W131 = S_II / 2):               {seidel.W_131_um:+10.4f} µm  ({seidel.W_131_um / 0.5875618:+8.3f} waves λ)")
    lines.append(f"  Wavefront Astigmatism (W222 = S_III / 2):       {seidel.W_222_um:+10.4f} µm  ({seidel.W_222_um / 0.5875618:+8.3f} waves λ)")
    lines.append(f"  Wavefront Field Curvature (W220 = (III+IV)/4):  {seidel.W_220_um:+10.4f} µm  ({seidel.W_220_um / 0.5875618:+8.3f} waves λ)")
    lines.append(f"  Wavefront Distortion (W311 = S_V / 2):          {seidel.W_311_um:+10.4f} µm  ({seidel.W_311_um / 0.5875618:+8.3f} waves λ)")
    lines.append(f"  Transverse Spherical Blur (TA = S_I/(2 n' u')): {seidel.TA_sph_um:+10.3f} µm")
    lines.append(f"  Transverse Coma Blur (TA = S_II/(2 n' u')):     {seidel.TA_coma_um:+10.3f} µm")
    lines.append(f"  Transverse Astigmatic Blur (TA = S_III/(2 n' u')):{seidel.TA_astig_um:+10.3f} µm")
    lines.append("=" * 105)
    return "\n".join(lines)
