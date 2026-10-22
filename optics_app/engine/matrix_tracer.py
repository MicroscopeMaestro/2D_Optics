"""
Matrix Propagation Ray Tracing & Optical Q-Factor Calculation Engine.

Implements paraxial 2x2 ABCD ray transfer matrix analysis, cardinal point extraction,
multi-surface sequential matrix ray tracing, and optical cavity/spectrometer Q-factor calculations.
"""

import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np

# Physical Constants
SPEED_OF_LIGHT = 299792458.0  # m/s


@dataclass
class RayMatrix:
    """2x2 Optical Ray Transfer Matrix (ABCD Matrix)."""
    A: float = 1.0
    B: float = 0.0
    C: float = 0.0
    D: float = 1.0

    def to_array(self) -> np.ndarray:
        return np.array([[self.A, self.B], [self.C, self.D]], dtype=float)

    @classmethod
    def from_array(cls, mat: np.ndarray) -> "RayMatrix":
        return cls(A=float(mat[0, 0]), B=float(mat[0, 1]), C=float(mat[1, 0]), D=float(mat[1, 1]))

    def __matmul__(self, other: "RayMatrix") -> "RayMatrix":
        """Matrix multiplication: M_new = self @ other."""
        res = self.to_array() @ other.to_array()
        return RayMatrix.from_array(res)

    def determinant(self) -> float:
        """Determinant AD - BC."""
        return float(self.A * self.D - self.B * self.C)

    def inverse(self) -> "RayMatrix":
        """Matrix inverse."""
        det = self.determinant()
        if abs(det) < 1e-15:
            raise ValueError("Ray matrix is singular, cannot invert.")
        return RayMatrix(
            A=self.D / det,
            B=-self.B / det,
            C=-self.C / det,
            D=self.A / det
        )

    def propagate_ray(self, y: float, u: float) -> Tuple[float, float]:
        """
        Propagate paraxial ray vector [y, u]^T through this matrix.
        y: Ray height from optical axis (mm)
        u: Ray slope/angle with respect to optical axis (rad)
        Returns: (y_out, u_out)
        """
        y_out = self.A * y + self.B * u
        u_out = self.C * y + self.D * u
        return float(y_out), float(u_out)

    def effective_focal_length(self) -> float:
        """
        Effective Focal Length (EFL) of system:
        f = -1 / C (if optical power C != 0).
        """
        if abs(self.C) < 1e-12:
            return float("inf")
        return -1.0 / self.C

    def back_focal_length(self) -> float:
        """
        Back Focal Length (BFL) measured from rear vertex to focal point:
        BFL = -A / C (for incoming parallel rays where u_in = 0).
        """
        if abs(self.C) < 1e-12:
            return float("inf")
        return -self.A / self.C

    def front_focal_length(self) -> float:
        """
        Front Focal Length (FFL) measured from front vertex to front focal point:
        FFL = -D / C.
        """
        if abs(self.C) < 1e-12:
            return float("inf")
        return -self.D / self.C

    def principal_planes(self) -> Tuple[float, float]:
        """
        Principal plane locations P1 (from front vertex) and P2 (from rear vertex):
        P1 = (D - 1) / C
        P2 = (1 - A) / C
        """
        if abs(self.C) < 1e-12:
            return 0.0, 0.0
        p1 = (self.D - 1.0) / self.C
        p2 = (1.0 - self.A) / self.C
        return float(p1), float(p2)

    def stability_parameter(self) -> float:
        """
        Resonator half-trace stability parameter:
        m = (A + D) / 2
        A cavity is stable if -1 <= m <= 1.
        """
        return float((self.A + self.D) / 2.0)

    def is_stable_resonator(self) -> bool:
        """Returns True if optical resonator satisfies |(A+D)/2| <= 1."""
        return abs(self.stability_parameter()) <= 1.0 + 1e-9

    def to_dict(self) -> Dict[str, Any]:
        return {
            "A": self.A,
            "B": self.B,
            "C": self.C,
            "D": self.D,
            "determinant": self.determinant(),
            "efl_mm": None if math.isinf(self.effective_focal_length()) else self.effective_focal_length(),
            "bfl_mm": None if math.isinf(self.back_focal_length()) else self.back_focal_length(),
            "ffl_mm": None if math.isinf(self.front_focal_length()) else self.front_focal_length(),
            "stability_parameter": self.stability_parameter(),
            "is_stable_resonator": self.is_stable_resonator()
        }


# Canonical Optical Component Matrices
def space_matrix(d: float, n: float = 1.0) -> RayMatrix:
    """
    Free space or homogeneous medium translation by distance d (mm).
    [1  d]
    [0  1]
    """
    return RayMatrix(A=1.0, B=float(d), C=0.0, D=1.0)


def thin_lens_matrix(f: float) -> RayMatrix:
    """
    Thin lens of focal length f (mm).
    [1     0]
    [-1/f  1]
    """
    if abs(f) < 1e-12:
        raise ValueError("Focal length cannot be zero.")
    return RayMatrix(A=1.0, B=0.0, C=-1.0 / float(f), D=1.0)


def refraction_matrix(R: float, n1: float = 1.0, n2: float = 1.5) -> RayMatrix:
    """
    Refraction at spherical curved dielectric interface of radius R (mm).
    R > 0 is convex to incident ray.
    [1                      0   ]
    [-(n2 - n1) / (n2 * R)  n1/n2]
    """
    if abs(R) < 1e-9 or math.isinf(R):
        # Planar interface
        return RayMatrix(A=1.0, B=0.0, C=0.0, D=float(n1 / n2))
    p = (n2 - n1) / (n2 * R)
    return RayMatrix(A=1.0, B=0.0, C=-float(p), D=float(n1 / n2))


def reflection_matrix(R: float) -> RayMatrix:
    """
    Reflection at spherical curved mirror of radius of curvature R (mm).
    For concave mirror (R < 0 in standard Zemax convention), optical power P = -2/R > 0,
    focal length f = -R/2 > 0, and transfer matrix C = -P = 2/R.
    [1    0]
    [2/R  1]
    """
    if abs(R) < 1e-9 or math.isinf(R):
        # Flat mirror
        return RayMatrix(A=1.0, B=0.0, C=0.0, D=1.0)
    return RayMatrix(A=1.0, B=0.0, C=2.0 / float(R), D=1.0)


def thick_lens_matrix(
    R1: float,
    R2: float,
    thickness: float,
    n_lens: float,
    n_ambient: float = 1.0
) -> RayMatrix:
    """
    Thick lens matrix: front refraction -> propagation through glass -> rear refraction.
    """
    m_front = refraction_matrix(R1, n_ambient, n_lens)
    m_glass = space_matrix(thickness, n_lens)
    m_rear = refraction_matrix(R2, n_lens, n_ambient)
    return m_rear @ m_glass @ m_front


class MatrixTracer:
    """
    Paraxial Matrix Propagation Ray Tracer.
    Traces rays through multi-surface sequential optical systems using ABCD matrices.
    """

    def __init__(self, surfaces_or_system: Any = None):
        self.system = surfaces_or_system
        self.accumulated_matrix = RayMatrix()
        self.step_matrices: List[Tuple[str, RayMatrix]] = []

    def build_from_system(self, system: Any, wavelength_um: float = 1.064) -> RayMatrix:
        """
        Build overall system transfer matrix from OpticalSystem or SurfaceStack.
        """
        surfaces = []
        if hasattr(system, "surfaces"):
            surfaces = system.surfaces
        elif hasattr(system, "stack") and hasattr(system.stack, "surfaces"):
            surfaces = system.stack.surfaces
        elif isinstance(system, list):
            surfaces = system

        if not surfaces:
            self.accumulated_matrix = RayMatrix()
            return self.accumulated_matrix

        try:
            from optics_app.utils.materials import get_refractive_index
        except ImportError:
            try:
                from optics_app.engine.dispersion import get_refractive_index
            except ImportError:
                from v2.engine.materials import get_refractive_index

        current_matrix = RayMatrix()
        self.step_matrices = []

        n_prev = 1.0

        for i, s in enumerate(surfaces):
            s_name = getattr(s, "name", f"Surface_{i}")
            r = getattr(s, "radius", 0.0)
            t = getattr(s, "thickness", 0.0)
            glass = str(getattr(s, "glass", "AIR")).upper()
            stype = getattr(s, "surface_type", None)
            stype_str = str(stype.name) if hasattr(stype, "name") else str(stype)

            # Determine refractive index after surface
            if "MIRROR" in glass or "REFL" in glass or "MIRROR" in stype_str:
                n_curr = 1.0
                m_surf = reflection_matrix(r)
            elif glass == "AIR" or not glass:
                n_curr = 1.0
                m_surf = refraction_matrix(r, n_prev, n_curr)
            else:
                try:
                    n_curr = float(get_refractive_index(glass, wavelength_um))
                except Exception:
                    n_curr = 1.5168  # Fallback standard N-BK7
                m_surf = refraction_matrix(r, n_prev, n_curr)

            # Accumulate surface refraction/reflection
            current_matrix = m_surf @ current_matrix
            self.step_matrices.append((f"{s_name}_surface", m_surf))

            # Accumulate space propagation to next surface (if thickness > 0 and not last)
            if t > 0.0 and i < len(surfaces) - 1:
                m_thick = space_matrix(t, n_curr)
                current_matrix = m_thick @ current_matrix
                self.step_matrices.append((f"{s_name}_thickness", m_thick))

            n_prev = n_curr

        self.accumulated_matrix = current_matrix
        return current_matrix

    def trace_ray(self, y0: float, u0: float) -> List[Dict[str, Any]]:
        """
        Trace a single paraxial ray (y0 [mm], u0 [rad]) step-by-step through the system.
        Returns a list of records with (step_name, y, u).
        """
        trajectory = [{"step": "input", "y": float(y0), "u": float(u0)}]
        y_curr, u_curr = float(y0), float(u0)

        for step_name, mat in self.step_matrices:
            y_curr, u_curr = mat.propagate_ray(y_curr, u_curr)
            trajectory.append({
                "step": step_name,
                "y": y_curr,
                "u": u_curr
            })

        return trajectory

    def trace_ray_fan(
        self,
        heights: List[float],
        angles: List[float]
    ) -> List[Dict[str, Any]]:
        """
        Trace a fan of paraxial rays and return initial vs final ray coordinates.
        """
        results = []
        for y in heights:
            for u in angles:
                traj = self.trace_ray(y, u)
                results.append({
                    "y_in": y,
                    "u_in": u,
                    "y_out": traj[-1]["y"],
                    "u_out": traj[-1]["u"],
                    "trajectory": traj
                })
        return results


# Optical Q-Factor Calculations
def calculate_cavity_q_factor(
    length_mm: float,
    wavelength_um: float,
    r1: float = 0.99,
    r2: float = 0.99,
    internal_loss: float = 0.0,
    refractive_index: float = 1.0
) -> Dict[str, float]:
    """
    Calculate the Quality Factor (Q-factor) and spectral metrics of an optical resonator / cavity.

    Parameters:
    - length_mm: Physical cavity length L (mm)
    - wavelength_um: Center resonance wavelength lambda_0 (um)
    - r1: Power reflectivity of mirror 1 (0 < r1 < 1)
    - r2: Power reflectivity of mirror 2 (0 < r2 < 1)
    - internal_loss: Internal round-trip power loss fraction delta_loss (>= 0)
    - refractive_index: Intracavity medium refractive index n (>= 1.0)

    Returns:
    Dict containing:
    - 'q_factor': Dimensionless optical cavity quality factor Q = omega_0 * tau_p
    - 'finesse': Cavity finesse F
    - 'photon_lifetime_ns': Photon storage time in nanoseconds
    - 'linewidth_hz': Full-width at half-maximum spectral linewidth Delta_nu in Hz
    - 'fsr_hz': Free Spectral Range in Hz
    - 'wavelength_fwhm_pm': Spectral resonance width in picometers
    """
    L = float(length_mm) * 1e-3  # meters
    wl = float(wavelength_um) * 1e-6  # meters
    n = float(refractive_index)

    # Round-trip optical path length
    optical_length = n * L
    t_roundtrip = (2.0 * optical_length) / SPEED_OF_LIGHT  # seconds

    # Round-trip fractional power loss delta
    r_prod = max(1e-12, min(1.0 - 1e-12, float(r1) * float(r2)))
    delta = -math.log(r_prod) + float(internal_loss)
    delta = max(delta, 1e-12)

    # Photon lifetime tau_p = t_rt / delta
    tau_p = t_roundtrip / delta

    # Resonance frequencies
    nu_0 = SPEED_OF_LIGHT / wl
    omega_0 = 2.0 * math.pi * nu_0

    # Quality factor Q = omega_0 * tau_p
    q_factor = omega_0 * tau_p

    # Linewidth Delta_nu (FWHM)
    linewidth_hz = 1.0 / (2.0 * math.pi * tau_p)
    wavelength_fwhm_m = (wl**2 / SPEED_OF_LIGHT) * linewidth_hz
    wavelength_fwhm_pm = wavelength_fwhm_m * 1e12

    # Free Spectral Range
    fsr_hz = SPEED_OF_LIGHT / (2.0 * optical_length)

    # Finesse F = FSR / linewidth
    finesse = fsr_hz / linewidth_hz if linewidth_hz > 0 else float("inf")

    return {
        "q_factor": float(q_factor),
        "finesse": float(finesse),
        "photon_lifetime_ns": float(tau_p * 1e9),
        "linewidth_hz": float(linewidth_hz),
        "fsr_hz": float(fsr_hz),
        "wavelength_fwhm_pm": float(wavelength_fwhm_pm),
        "roundtrip_time_ps": float(t_roundtrip * 1e12),
        "center_frequency_thz": float(nu_0 * 1e-12)
    }


def calculate_spectrometer_q_factor(
    wavelength_um: float,
    lines_per_mm: float,
    beam_width_mm: float,
    order: int = 1
) -> Dict[str, float]:
    """
    Calculate the theoretical resolving power Q-factor for a grating spectrometer.
    Resolving Power R = lambda / Delta_lambda = |m| * N = Q.

    Parameters:
    - wavelength_um: Operating wavelength (um)
    - lines_per_mm: Grating groove density (lines/mm)
    - beam_width_mm: Illuminated beam width on grating (mm)
    - order: Diffraction order m (typically 1)
    """
    N = float(lines_per_mm) * float(beam_width_mm)  # Total illuminated grooves
    resolving_power = abs(int(order)) * N
    d_lambda_nm = (float(wavelength_um) * 1e3) / resolving_power if resolving_power > 0 else 0.0

    return {
        "q_factor": float(resolving_power),
        "resolving_power": float(resolving_power),
        "illuminated_grooves": float(N),
        "spectral_resolution_nm": float(d_lambda_nm),
        "diffraction_order": int(order)
    }


def calculate_system_q_factor(
    system: Any,
    mode: str = "auto",
    r1: float = 0.99,
    r2: float = 0.99,
    internal_loss: float = 0.0
) -> Dict[str, float]:
    """
    High-level Q-factor evaluation for an optical system.
    Extracts system length and primary wavelength automatically.
    """
    surfaces = []
    if hasattr(system, "surfaces"):
        surfaces = system.surfaces
    elif hasattr(system, "stack") and hasattr(system.stack, "surfaces"):
        surfaces = system.stack.surfaces

    total_track_mm = sum(getattr(s, "thickness", 0.0) for s in surfaces) if surfaces else 100.0
    if total_track_mm <= 0.0:
        total_track_mm = 100.0

    wl_um = 1.064
    if hasattr(system, "primary_wavelength"):
        wl_um = system.primary_wavelength
    elif hasattr(system, "wavelengths") and system.wavelengths:
        wl_um = system.wavelengths[0]

    # Check for grating surface in spectrometer mode
    grating_surf = None
    for s in surfaces:
        st_str = str(getattr(s, "surface_type", "")).upper()
        if "GRATING" in st_str or getattr(s, "grating_lines_per_mm", 0.0) > 0 or "GRATING" in getattr(s, "name", "").upper():
            grating_surf = s
            break

    if mode == "spectrometer" or (mode == "auto" and grating_surf is not None):
        g_lines = getattr(grating_surf, "grating_lines_per_mm", 600.0) if grating_surf else 600.0
        g_order = getattr(grating_surf, "grating_order", 1) if grating_surf else 1
        semi_d = getattr(grating_surf, "semi_diameter", 12.7) if grating_surf else 12.7
        return calculate_spectrometer_q_factor(
            wavelength_um=wl_um,
            lines_per_mm=g_lines,
            beam_width_mm=semi_d * 2.0,
            order=g_order
        )
    else:
        return calculate_cavity_q_factor(
            length_mm=total_track_mm,
            wavelength_um=wl_um,
            r1=r1,
            r2=r2,
            internal_loss=internal_loss
        )
