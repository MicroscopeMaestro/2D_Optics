"""
Diffraction Grating Physics & 3D Vector Diffraction Equations.
"""
from typing import Tuple, Optional, Union, List
import numpy as np

def grating_diffract_3d(ray_dir: Union[Tuple[float, float, float], np.ndarray],
                       grating_normal: Union[Tuple[float, float, float], np.ndarray],
                       line_vector: Union[Tuple[float, float, float], np.ndarray],
                       line_density_lines_per_mm: float,
                       order: int,
                       wavelength_um: float,
                       is_reflection: bool = False) -> Tuple[Optional[np.ndarray], bool]:
    """
    3D Vector Grating Diffraction Solver using Orthonormal Surface Basis (e_disp, e_groove, N).
    Supports transmission and reflection diffraction directly.
    Returns (v_out, evanescent_flag).
    """
    v_in = np.array(ray_dir, dtype=float)
    v_norm = np.linalg.norm(v_in)
    v_in = v_in / v_norm if v_norm > 0 else np.array([0.0, 0.0, 1.0])

    N = np.array(grating_normal, dtype=float)
    n_norm = np.linalg.norm(N)
    N = N / n_norm if n_norm > 0 else np.array([0.0, 0.0, 1.0])

    L = np.array(line_vector, dtype=float)
    l_norm = np.linalg.norm(L)
    L = L / l_norm if l_norm > 0 else np.array([1.0, 0.0, 0.0])

    # Line vector L represents the grating periodicity / dispersion vector in the surface plane
    L_proj = L - np.dot(L, N) * N
    l_proj_norm = np.linalg.norm(L_proj)

    if l_proj_norm > 1e-12:
        e_disp = L_proj / l_proj_norm
    else:
        v_temp = np.array([1.0, 0.0, 0.0]) if abs(N[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        v_proj = v_temp - np.dot(v_temp, N) * N
        e_disp = v_proj / np.linalg.norm(v_proj)

    # Groove direction e_groove is in the grating plane perpendicular to dispersion vector (N x e_disp)
    e_groove = np.cross(N, e_disp)
    e_groove_norm = np.linalg.norm(e_groove)
    if e_groove_norm > 0:
        e_groove = e_groove / e_groove_norm

    v_in_disp = np.dot(v_in, e_disp)
    v_in_groove = np.dot(v_in, e_groove)
    v_in_N = np.dot(v_in, N)

    d_um = 1000.0 / line_density_lines_per_mm if line_density_lines_per_mm > 0 else 1e9

    v_out_disp = v_in_disp + (order * wavelength_um / d_um)
    v_out_groove = v_in_groove

    sum_sq = v_out_disp**2 + v_out_groove**2
    if sum_sq > 1.0:
        return None, True  # Evanescent wave

    # Reflection flips normal component back to incident half-space
    if is_reflection:
        v_out_N = -np.sign(v_in_N) * np.sqrt(np.maximum(0.0, 1.0 - sum_sq)) if v_in_N != 0 else -np.sqrt(np.maximum(0.0, 1.0 - sum_sq))
    else:
        v_out_N = np.sign(v_in_N) * np.sqrt(np.maximum(0.0, 1.0 - sum_sq)) if v_in_N != 0 else np.sqrt(np.maximum(0.0, 1.0 - sum_sq))

    v_out = v_out_disp * e_disp + v_out_groove * e_groove + v_out_N * N
    v_out_norm = np.linalg.norm(v_out)
    if v_out_norm > 0:
        v_out = v_out / v_out_norm
    return v_out, False

class DiffractionGrating:
    def __init__(self, lines_per_mm: float = 600.0, order: int = 1, is_reflection: bool = False):
        self.lines_per_mm = float(lines_per_mm)
        self.order = int(order)
        self.is_reflection = bool(is_reflection)
        self.pitch_um = 1000.0 / self.lines_per_mm if self.lines_per_mm > 0 else 1e9

    def calculate_diffraction_angle(self, incident_angle_rad: float, wavelength_um: float) -> float:
        sin_i = np.sin(incident_angle_rad)
        sin_m = sin_i + (self.order * wavelength_um / self.pitch_um)
        if abs(sin_m) > 1.0:
            raise ValueError(f"Evanescent order: diffraction angle undefined for lambda={wavelength_um} um.")
        return float(np.arcsin(sin_m))

    def angular_dispersion(self, diffracted_angle_rad: float) -> float:
        d_um = self.pitch_um
        return float(self.order / (d_um * np.cos(diffracted_angle_rad)))
