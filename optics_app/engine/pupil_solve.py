"""
Zemax-Style Paraxial Entrance Pupil Calculation & Iterative Ray Aiming Engine.

Computes the paraxial entrance pupil location (Z_ep) and radius (R_ep) by imaging
the physical aperture stop backwards into object space through all preceding surfaces.
Provides robust off-axis ray aiming from finite object points (Object Height mm)
and infinite conjugate field angles (Angle deg).
"""

from __future__ import annotations
import math
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union

from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.ray import Ray
from optics_app.engine.matrix_optics import refraction_matrix, space_matrix, TransferMatrix

try:
    from optics_app.utils.materials import get_refractive_index
except ImportError:
    try:
        from v2.engine.materials import get_refractive_index
    except ImportError:
        try:
            from optics_app.engine.dispersion import get_refractive_index
        except ImportError:
            def get_refractive_index(glass, wl):
                return 1.5168


def _get_surf_refractive_index(glass_val: Any, wavelength_um: float = 0.5876) -> float:
    """Safely obtain refractive index for a glass name or value at wavelength (um)."""
    if not glass_val:
        return 1.0
    g_str = str(glass_val).strip()
    if not g_str or g_str.upper() in ("AIR", "VACUUM", "NONE"):
        return 1.0
    try:
        return float(get_refractive_index(g_str, wavelength_um))
    except Exception:
        return 1.0


FIELD_COLORS = ['#2563eb', '#16a34a', '#dc2626', '#9333ea', '#ea580c', '#0891b2', '#ca8a04', '#e11d48']


def find_stop_surface_index(surfaces: List[Surface]) -> int:
    """
    Locate the 0-based index of the aperture stop surface in the active sequence.
    If no explicit stop is defined, defaults to the first optical element (index 1),
    or index 0 if the sequence has fewer than 2 surfaces.
    """
    if not surfaces:
        return 0

    for idx, surf in enumerate(surfaces):
        if idx == 0 and len(surfaces) > 1:
            # Object surface is typically not the stop unless single-surface system
            continue

        is_stop = (
            getattr(surf, "is_stop", False)
            or getattr(surf, "is_aperture_stop", False)
            or surf.surface_type == SurfaceType.STOP
            or "STOP" in str(getattr(surf, "surface_type", "")).upper()
            or "STOP" in str(getattr(surf, "name", "")).upper()
            or "(STO)" in str(getattr(surf, "name", "")).upper()
        )
        if is_stop:
            return idx

    # Fallback to first physical optical surface
    return 1 if len(surfaces) > 1 else 0


def get_effective_clear_radius(surf: Surface, default_val: float = 12.7) -> float:
    """
    Determine the effective physical clear aperture radius of an optical surface.
    Respects explicit aperture definitions (aperture_r_outer, rectangular apertures),
    falling back to mechanical semi_diameter.
    """
    outer_r = float(getattr(surf, "aperture_r_outer", 0.0) or 0.0)
    if outer_r > 0.0:
        return outer_r
    atype = str(getattr(surf, "aperture_type", "Circular") or "Circular").capitalize()
    if atype == "Rectangular":
        hx = float(getattr(surf, "aperture_x_half", 0.0) or 0.0)
        hy = float(getattr(surf, "aperture_y_half", 0.0) or 0.0)
        if hx > 0.0 and hy > 0.0:
            return min(hx, hy)
        if hx > 0.0 or hy > 0.0:
            return max(hx, hy)
    elif atype == "Elliptical":
        a = float(getattr(surf, "aperture_a_semi", 0.0) or 0.0)
        b = float(getattr(surf, "aperture_b_semi", 0.0) or 0.0)
        if a > 0.0 and b > 0.0:
            return min(a, b)
        if a > 0.0 or b > 0.0:
            return max(a, b)
    sd = float(getattr(surf, "semi_diameter", 0.0) or 0.0)
    return sd if sd > 0.0 else default_val


def calculate_entrance_pupil(
    surfaces: List[Surface],
    wavelength_um: float = 0.5876,
    default_epd: Optional[float] = None,
) -> Tuple[float, float, int]:
    """
    Calculate the paraxial Entrance Pupil position and radius.

    The entrance pupil is the paraxial image of the aperture stop in object space.
    Its axial position Z_ep_global is computed by transferring from Surface 1 vertex
    to the Aperture Stop vertex via ABCD ray transfer matrix:
        Z_ep = B / A  (measured from Surface 1 vertex)
        R_ep = R_stop / |A|

    Args:
        surfaces: Active sequence of Surface objects (S0 = Object, S1 = Front element, ...).
        wavelength_um: Evaluation wavelength for refractive index calculation.
        default_epd: Fallback entrance pupil diameter if stop semi-diameter is not set.

    Returns:
        Tuple of (z_ep_global, r_ep, stop_index):
            z_ep_global: Global Z position of the entrance pupil (mm).
            r_ep: Semi-diameter (radius) of the entrance pupil (mm).
            stop_index: 0-based surface index of the physical aperture stop.
    """
    if not surfaces or len(surfaces) < 2:
        z0 = float(getattr(surfaces[0], "thickness", 0.0)) if surfaces else 0.0
        r0 = get_effective_clear_radius(surfaces[0], default_val=10.0) if surfaces else 10.0
        return z0, max(r0, 1.0), 0

    stop_idx = find_stop_surface_index(surfaces)
    stop_surf = surfaces[stop_idx]

    r_stop = get_effective_clear_radius(stop_surf, default_val=0.0)
    if r_stop <= 0.0:
        if default_epd is not None and default_epd > 0.0:
            r_stop = default_epd / 2.0
        else:
            r_stop = get_effective_clear_radius(surfaces[1], default_val=12.7)
            if r_stop <= 0.0:
                r_stop = 12.7
    elif default_epd is not None and default_epd > 0.0 and stop_idx <= 1:
        r_stop = min(r_stop, default_epd / 2.0)

    d0 = float(getattr(surfaces[0], "thickness", 0.0))
    z_s1_global = d0  # Surface 1 vertex global Z position

    # Case 1: Stop is at Surface 1
    if stop_idx <= 1:
        return z_s1_global, r_stop, stop_idx

    # Case 2: Stop is embedded inside or behind optical surfaces (stop_idx > 1)
    # Compute ABCD transfer matrix from Surface 1 vertex to Stop vertex:
    # M_total = M_space(t_{k-1}) @ M_refr(S_{k-1}) @ ... @ M_space(t_1) @ M_refr(S_1)
    M_front = TransferMatrix(A=1.0, B=0.0, C=0.0, D=1.0, n_in=1.0, n_out=1.0)

    for i in range(1, stop_idx):
        cur_surf = surfaces[i]

        # Determine refractive indices
        # Medium preceding surface i (medium between surface i-1 and surface i):
        if i == 1:
            g_prev = getattr(surfaces[0], "glass", "AIR") if surfaces else "AIR"
        else:
            g_prev = getattr(surfaces[i - 1], "glass", "AIR")

        g_curr = getattr(cur_surf, "glass", "AIR")

        n_in = _get_surf_refractive_index(g_prev, wavelength_um)
        n_out = _get_surf_refractive_index(g_curr, wavelength_um)

        # Refraction at surface i
        rad = float(cur_surf.radius)
        is_flat = (rad == 0.0 or math.isinf(rad) or abs(rad) > 1e12)
        refr_rad = float("inf") if is_flat else rad

        # Handle mirrors
        is_mirror = (
            getattr(cur_surf, "is_reflective", lambda: False)()
            or str(getattr(cur_surf, "glass", "")).upper() in ["MIRROR", "MIRR", "REFL"]
            or cur_surf.surface_type == SurfaceType.MIRROR
        )
        if is_mirror:
            C_refl = 0.0 if is_flat else (2.0 / rad)
            M_step = TransferMatrix(A=1.0, B=0.0, C=C_refl, D=1.0, n_in=n_in, n_out=n_in)
            n_out = n_in
        else:
            M_step = refraction_matrix(R=refr_rad, n1=n_in, n2=n_out)

        # Space propagation across thickness t_i of current surface
        thick = float(getattr(cur_surf, "thickness", 0.0))
        M_trans = space_matrix(d=thick, n=n_out)

        M_front = M_trans @ M_step @ M_front

    A = M_front.A
    B = M_front.B

    # Paraxial entrance pupil position from Surface 1 vertex: Z_ep = B / A
    if abs(A) > 1e-7:
        z_ep_from_s1 = B / A
        r_ep = r_stop / abs(A)
    else:
        # Afocal front group fallback
        z_ep_from_s1 = 0.0
        r_ep = r_stop

    # Clamp pupil radius to sensible physical boundaries
    if r_ep <= 0.0 or math.isinf(r_ep) or math.isnan(r_ep) or r_ep > 500.0:
        r_ep = r_stop

    z_ep_global = z_s1_global + z_ep_from_s1
    return float(z_ep_global), float(r_ep), stop_idx


def aim_ray_to_pupil(
    surfaces: List[Surface],
    obj_point: Tuple[float, float, float],
    pupil_norm: Tuple[float, float],
    wavelength_um: float = 0.5876,
    z_ep_global: Optional[float] = None,
    r_ep: Optional[float] = None,
    iterative: Optional[bool] = None,
    max_iter: int = 4,
    system: Optional[Any] = None,
) -> Ray:
    """
    Launch a ray from an off-axis object point aiming at a normalized entrance pupil coordinate.

    Args:
        surfaces: Active sequence of optical surfaces.
        obj_point: Object coordinates (x_obj, y_obj, z_obj) in millimeters.
        pupil_norm: Normalized entrance pupil coordinates (Px, Py) in [-1, 1].
        wavelength_um: Ray wavelength in micrometers.
        z_ep_global: Precomputed entrance pupil Z position (computed if None).
        r_ep: Precomputed entrance pupil radius (computed if None).
        iterative: If True (or None when system is provided), performs Newton-Raphson refinement to hit exact stop coordinates.
        max_iter: Maximum iterations for real ray aiming refinement.
        system: OpticalSystem instance for tracing during iterative refinement.

    Returns:
        Ray object positioned at obj_point with unit direction aiming at the pupil.
    """
    px, py = pupil_norm
    x_obj, y_obj, z_obj = obj_point

    if z_ep_global is None or r_ep is None:
        z_ep, r_pupil, stop_idx = calculate_entrance_pupil(surfaces, wavelength_um=wavelength_um)
    else:
        z_ep = z_ep_global
        r_pupil = r_ep
        stop_idx = find_stop_surface_index(surfaces)

    # Initial paraxial pupil aim point
    target_x = px * r_pupil
    target_y = py * r_pupil
    target_z = z_ep

    # Direction vector from object point to entrance pupil
    dx = target_x - x_obj
    dy = target_y - y_obj
    dz = target_z - z_obj
    norm = math.sqrt(dx * dx + dy * dy + dz * dz)
    if norm < 1e-12:
        norm = 1.0
        dz = 1.0

    start_pos = np.array([x_obj, y_obj, z_obj], dtype=float)
    start_dir = np.array([dx / norm, dy / norm, dz / norm], dtype=float)

    # Iterative ray aiming refinement (if requested or system provided and stop is downstream)
    do_iter = (system is not None) if iterative is None else bool(iterative)
    if do_iter and stop_idx > 1 and system is not None:
        stop_surf = surfaces[stop_idx]
        r_stop = float(getattr(stop_surf, "semi_diameter", 6.0))
        r_stop_eff = r_stop * (1.0 - 1e-5)
        target_stop_x = px * r_stop_eff
        target_stop_y = py * r_stop_eff

        cur_target_x = target_x
        cur_target_y = target_y

        for _ in range(max_iter):
            test_ray = Ray(pos=start_pos.copy(), dir=start_dir.copy(), wavelength=wavelength_um)
            traced = system.trace_ray(test_ray, max_surface_index=stop_idx)

            if len(traced.history) <= stop_idx:
                break

            hit = traced.history[stop_idx]
            err_x = hit.get("x", 0.0) - target_stop_x
            err_y = hit.get("y", 0.0) - target_stop_y

            if math.sqrt(err_x * err_x + err_y * err_y) < 1e-4:
                break

            # Newton-Raphson pupil correction step
            mag = r_stop / max(r_pupil, 0.01)
            cur_target_x -= err_x / max(mag, 0.1)
            cur_target_y -= err_y / max(mag, 0.1)

            dx = cur_target_x - x_obj
            dy = cur_target_y - y_obj
            dz = target_z - z_obj
            norm = math.sqrt(dx * dx + dy * dy + dz * dz)
            if norm > 1e-12:
                start_dir = np.array([dx / norm, dy / norm, dz / norm], dtype=float)

    return Ray(pos=start_pos, dir=start_dir, wavelength=wavelength_um)


def aim_collimated_ray(
    surfaces: List[Surface],
    field_angle_deg: Tuple[float, float],
    pupil_norm: Tuple[float, float],
    wavelength_um: float = 0.5876,
    z_ep_global: Optional[float] = None,
    r_ep: Optional[float] = None,
) -> Ray:
    """
    Launch a collimated (infinite conjugate) ray with field angle (ang_x_deg, ang_y_deg).

    The ray passes through normalized pupil coordinate (Px, Py) at the entrance pupil
    and propagates along the field direction cosines.

    Args:
        surfaces: Active optical surfaces.
        field_angle_deg: (ang_x_deg, ang_y_deg) field angles in degrees.
        pupil_norm: (Px, Py) normalized pupil coordinates in [-1, 1].
        wavelength_um: Ray wavelength in micrometers.
        z_ep_global: Global Z position of the entrance pupil.
        r_ep: Entrance pupil radius.
    """
    ang_x_deg, ang_y_deg = field_angle_deg
    px, py = pupil_norm

    if z_ep_global is None or r_ep is None:
        z_ep, r_pupil, _ = calculate_entrance_pupil(surfaces, wavelength_um=wavelength_um)
    else:
        z_ep = z_ep_global
        r_pupil = r_ep

    # Direction cosines from field angles
    ax_rad = math.radians(ang_x_deg)
    ay_rad = math.radians(ang_y_deg)

    kx = math.sin(ax_rad)
    ky = math.sin(ay_rad)
    kz_sq = 1.0 - kx * kx - ky * ky
    kz = math.sqrt(max(kz_sq, 1e-9))

    direction = np.array([kx, ky, kz], dtype=float)
    direction = direction / np.linalg.norm(direction)

    # Launch position in object plane Z=0 projected backwards from entrance pupil
    # Ray passing through (px*r_pupil, py*r_pupil, z_ep) intersects Z=0 at:
    # x(0) = x_ep - (z_ep / kz) * kx
    # y(0) = y_ep - (z_ep / kz) * ky
    x_ep = px * r_pupil
    y_ep = py * r_pupil

    t_back = z_ep / max(kz, 1e-6)
    x_start = x_ep - t_back * kx
    y_start = y_ep - t_back * ky
    z_start = 0.0

    return Ray(
        pos=np.array([x_start, y_start, z_start], dtype=float),
        dir=direction,
        wavelength=wavelength_um,
    )


def generate_pupil_grid(
    pattern: Union[str, Any] = "Hexapolar",
    rings: int = 4,
    arms: int = 6,
    include_chief_ray: bool = True,
    linear_arms: bool = True,
    max_radius: float = 1.0,
) -> List[Tuple[float, float, float]]:
    """
    Generate normalized entrance pupil coordinates (Px, Py) and quadrature weights (W).

    Supports:
      1. Hexapolar: Concentric rings (1 to N_R) with arms*r rays (linear_arms=True) or constant
         arms rays per ring, up to max_radius (default 1.0), plus central chief ray (0, 0).
      2. Gaussian Quadrature: Radial roots rho_r = sqrt((1 + t_r) / 2) from Legendre polynomial
         roots t_r in (-1, 1), weights w_r = omega_r / 2, and N_theta azimuthal arms.
      3. Square Grid: Cartesian grid bounded by Px^2 + Py^2 <= max_radius^2.

    Args:
        pattern: RayPatternType enum, "Hexapolar", "Gaussian Quadrature", or "Square Grid".
        rings: Number of concentric rings (Hexapolar) or Gauss-Legendre polynomial order.
        arms: Number of azimuthal arms / rays on the first ring (default 6).
        include_chief_ray: If True, prepends chief ray (0.0, 0.0).
        linear_arms: For Hexapolar, if True ring r has arms*r rays; if False has arms rays.
        max_radius: Maximum normalized pupil semi-diameter (default 1.0 for full pupil).

    Returns:
        List of (Px, Py, Weight) tuples where Px^2 + Py^2 <= max_radius^2 and sum(Weight) == 1.0.
    """
    p_name = pattern.value if hasattr(pattern, "value") else str(pattern)
    p_lower = p_name.lower()
    rings = max(1, int(rings))
    arms = max(1, int(arms))
    max_radius = float(max_radius)

    pts: List[Tuple[float, float, float]] = []

    if "hex" in p_lower:
        raw_pts: List[Tuple[float, float, float]] = []
        if include_chief_ray:
            w0 = 1.0 if linear_arms else (0.25 / (rings ** 2))
            raw_pts.append((0.0, 0.0, w0))

        for r in range(1, rings + 1):
            rho = (r / rings) * max_radius
            n_pts = arms * r if linear_arms else arms
            w_r = 1.0 if linear_arms else (2.0 * r) / (rings ** 2 * arms)
            for k in range(n_pts):
                theta = 2.0 * math.pi * k / n_pts
                px = float(rho * math.cos(theta))
                py = float(rho * math.sin(theta))
                raw_pts.append((px, py, w_r))

        total_w = sum(w for _, _, w in raw_pts)
        pts = [(px, py, w / total_w) for px, py, w in raw_pts]

    elif "gauss" in p_lower:
        roots, weights = np.polynomial.legendre.leggauss(rings)
        u_roots = 0.5 * (roots + 1.0)
        rho_roots = np.sqrt(u_roots) * max_radius
        radial_weights = 0.5 * weights

        if include_chief_ray:
            pts.append((0.0, 0.0, 0.0))

        for rho_r, w_r in zip(rho_roots, radial_weights):
            w_pt = float(w_r / arms)
            for a in range(arms):
                theta = 2.0 * math.pi * a / arms
                px = float(rho_r * math.cos(theta))
                py = float(rho_r * math.sin(theta))
                pts.append((px, py, w_pt))

    elif "square" in p_lower:
        n_side = 2 * rings + 1
        coords = np.linspace(-max_radius, max_radius, n_side)
        raw_pts = []
        for y in coords:
            for x in coords:
                if (x * x + y * y) <= (max_radius * max_radius + 1e-12):
                    raw_pts.append((float(x), float(y), 1.0))
        total_w = sum(w for _, _, w in raw_pts)
        pts = [(px, py, w / total_w) for px, py, w in raw_pts]

    else:
        # Fallback to standard Hexapolar
        return generate_pupil_grid(
            pattern="Hexapolar",
            rings=rings,
            arms=arms,
            include_chief_ray=include_chief_ray,
            linear_arms=linear_arms,
            max_radius=max_radius,
        )

    return pts

