"""
Polarization Physics, Jones Calculus, and Fresnel Boundary Solver.
"""
from dataclasses import dataclass
from typing import Tuple, Optional, Union
import numpy as np


@dataclass
class FresnelResult:
    r_s: complex
    r_p: complex
    t_s: complex
    t_p: complex
    tir: bool
    theta_inc: float
    theta_trans: complex


def compute_fresnel_coefficients(
    n1: Union[float, complex],
    n2: Union[float, complex],
    cos_theta1: float
) -> FresnelResult:
    """
    Calculate complex Fresnel reflection and transmission coefficients for s and p polarizations.
    Handles dielectric boundaries, metallic boundaries (n + ik), and Total Internal Reflection (TIR).
    """
    cos_theta1 = float(np.clip(cos_theta1, -1.0, 1.0))
    sin_theta1 = np.sqrt(1.0 - cos_theta1**2)

    c_n1 = complex(n1)
    c_n2 = complex(n2)

    sin_theta2 = (c_n1 / c_n2) * sin_theta1

    # Complex square root for cos(theta2) with Re(cos_theta2) >= 0
    cos_theta2 = np.sqrt(1.0 - sin_theta2**2 + 0j)
    if cos_theta2.real < 0:
        cos_theta2 = -cos_theta2

    is_tir = False
    if abs(c_n1.imag) < 1e-12 and abs(c_n2.imag) < 1e-12 and c_n1.real > c_n2.real:
        if sin_theta1 > (c_n2.real / c_n1.real):
            is_tir = True

    r_s = (c_n1 * cos_theta1 - c_n2 * cos_theta2) / (c_n1 * cos_theta1 + c_n2 * cos_theta2)
    r_p = (c_n2 * cos_theta1 - c_n1 * cos_theta2) / (c_n2 * cos_theta1 + c_n1 * cos_theta2)
    t_s = (2.0 * c_n1 * cos_theta1) / (c_n1 * cos_theta1 + c_n2 * cos_theta2)
    t_p = (2.0 * c_n1 * cos_theta1) / (c_n2 * cos_theta1 + c_n1 * cos_theta2)

    theta_inc = float(np.arccos(cos_theta1))
    theta_trans = complex(np.arcsin(np.clip(sin_theta2, -1.0+0j, 1.0+0j))) if abs(sin_theta2.imag) < 1e-12 and abs(sin_theta2.real) <= 1.0 else complex(np.arcsin(sin_theta2))

    return FresnelResult(
        r_s=complex(r_s),
        r_p=complex(r_p),
        t_s=complex(t_s),
        t_p=complex(t_p),
        tir=is_tir,
        theta_inc=theta_inc,
        theta_trans=theta_trans
    )


def _cross3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.array([
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    ], dtype=float)


def _norm3(v: np.ndarray) -> float:
    return float(np.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]))


def _default_orthogonal(v: np.ndarray) -> np.ndarray:
    """Find a unit vector orthogonal to v."""
    ortho = np.array([1.0, 0.0, 0.0]) if abs(v[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    s = _cross3(v, ortho)
    norm = _norm3(s)
    return s / norm if norm > 0 else np.array([1.0, 0.0, 0.0])


def compute_sp_basis(
    k_dir: np.ndarray,
    normal: np.ndarray,
    s_in_ray: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute local S and P basis vectors at surface intersection.
    Returns (s_basis, p_basis, normal_aligned).
    """
    k = np.asarray(k_dir, dtype=float)
    k_norm = _norm3(k)
    k = k / k_norm if k_norm > 0 else np.array([0.0, 0.0, 1.0])

    N = np.asarray(normal, dtype=float)
    n_norm = _norm3(N)
    N = N / n_norm if n_norm > 0 else np.array([0.0, 0.0, 1.0])

    if (k[0] * N[0] + k[1] * N[1] + k[2] * N[2]) > 0:
        N = -N

    s_raw = _cross3(k, N)
    s_norm = _norm3(s_raw)

    if s_norm > 1e-10:
        s_basis = s_raw / s_norm
    else:
        # Normal incidence fallback (k || N)
        if s_in_ray is not None and _norm3(s_in_ray) > 0:
            s_in = np.asarray(s_in_ray, dtype=float)
            dot_val = s_in[0] * k[0] + s_in[1] * k[1] + s_in[2] * k[2]
            s_proj = s_in - dot_val * k
            s_proj_norm = _norm3(s_proj)
            s_basis = s_proj / s_proj_norm if s_proj_norm > 1e-10 else _default_orthogonal(k)
        else:
            s_basis = _default_orthogonal(k)

    p_basis = _cross3(s_basis, k)
    p_norm = _norm3(p_basis)
    p_basis = p_basis / p_norm if p_norm > 0 else p_basis

    return s_basis, p_basis, N


def rotate_jones_vector(
    jones_vec: np.ndarray,
    s_in: np.ndarray,
    s_target: np.ndarray,
    k_dir: np.ndarray
) -> np.ndarray:
    """
    Rotate 2D Jones vector [E_s, E_p] from s_in basis to s_target basis around propagation direction k_dir.
    """
    k = np.asarray(k_dir, dtype=float)
    k_norm = _norm3(k)
    if k_norm > 0:
        k = k / k_norm

    dot_st = float(s_in[0] * s_target[0] + s_in[1] * s_target[1] + s_in[2] * s_target[2])
    cos_rot = float(np.clip(dot_st, -1.0, 1.0))
    cross_st = _cross3(s_in, s_target)
    sin_rot = float(cross_st[0] * k[0] + cross_st[1] * k[1] + cross_st[2] * k[2])
    R = np.array([[cos_rot, sin_rot], [-sin_rot, cos_rot]], dtype=float)
    return R @ np.asarray(jones_vec, dtype=complex)
