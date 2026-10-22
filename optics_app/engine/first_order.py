"""
Zemax OpticStudio-Compatible First Order Optical Data Engine.
=============================================================
Calculates paraxial first order optical properties using ABCD transfer matrix analysis:
- Effective Focal Length (EFFL)
- Back Focal Length (BFL)
- Front Focal Length (FFL)
- Front & Rear Principal Planes (P1, P2) in relative and absolute Global Z
- Front & Rear Nodal Points (N1, N2)
- Entrance Pupil Diameter (EPD) and Position (relative and Global Z)
- Exit Pupil / Effective Back Pupil Diameter (EXPD) and Position
- Exit Pupil to Image Plane Distance
- Pupil Magnification (EXPD / EPD)
- Optical Power (Diopters [m^-1] and [mm^-1])
- Paraxial Image Space F/# & Working F/#
- Paraxial Image Space Numerical Aperture (NA)
- Total Track (TOTR), Optical Glass Track, and Air Space Track
- Full System Transfer Matrix (ABCD) & Determinant
- Paraxial Diagnostic Status Flags
"""

from typing import List, Dict, Any, Optional, Tuple
import math
import numpy as np

from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.matrix_tracer import MatrixTracer, RayMatrix, space_matrix


def get_surface_clear_radius(surf: Surface, default_val: float = 12.5) -> float:
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


def compute_paraxial_marginal_and_principal_rays(
    surfaces: List[Surface],
    wavelength_um: float = 0.5876,
    epd: Optional[float] = None,
    field_angle_deg: float = 0.0,
    object_height: Optional[float] = None,
    stop_idx: Optional[int] = None,
    eval_surface_idx: Optional[int] = None,
    config: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Compute paraxial marginal and principal (chief) ray propagation and surface-by-surface vignetting.

    Follows Zemax OpticStudio first-order paraxial ray tracing equations:
        Refraction: n' * u' = n * u - y * Phi
        Transfer:   y_{i+1} = y_i + t_i * u'_i
    where Phi = (n' - n) / R (or 2*n / R for mirrors, 0.0 for plano).
    """
    if not surfaces or len(surfaces) < 2:
        return {
            "surfaces_data": [],
            "system_vignetting_factor": 0.0,
            "is_vignetted": False,
            "limiting_surface_index": 0,
            "limiting_surface_name": "None",
            "marginal_trajectory": {"z": [], "y": []},
            "chief_trajectory": {"z": [], "y": []},
            "envelope_upper": {"z": [], "y": []},
            "envelope_lower": {"z": [], "y": []},
            "semi_diameters": [],
            "stop_index": 0,
            "enp_pos": 0.0,
            "epd": 0.0
        }

    try:
        from optics_app.utils.materials import get_refractive_index
    except ImportError:
        try:
            from optics_app.engine.dispersion import get_refractive_index
        except ImportError:
            try:
                from v2.engine.materials import get_refractive_index
            except ImportError:
                def get_refractive_index(glass, wl):
                    return 1.0

    def _safe_n(glass_val: Any) -> float:
        if not glass_val:
            return 1.0
        g = str(glass_val).strip().upper()
        if not g or g in ("AIR", "NONE", "VACUUM", ""):
            return 1.0
        try:
            return float(get_refractive_index(g, wavelength_um))
        except Exception:
            return 1.5168

    end_idx = eval_surface_idx if (eval_surface_idx is not None and eval_surface_idx > 0) else len(surfaces) - 1
    end_idx = min(end_idx, len(surfaces) - 1)
    start_idx = 0
    s0 = float(getattr(surfaces[0], "thickness", 0.0))
    if len(surfaces) > 1 and surfaces[0].surface_type == SurfaceType.OBJECT:
        g = str(getattr(surfaces[0], "glass", "") or "").upper()
        if not g or g in ("AIR", "NONE"):
            start_idx = 1

    opt_surfaces = surfaces[start_idx:end_idx + 1]
    num_opt = len(opt_surfaces)

    # 1. Stop Identification
    solved_stop_idx = None
    if stop_idx is not None:
        if stop_idx >= start_idx and stop_idx <= end_idx:
            solved_stop_idx = stop_idx - start_idx
        elif 0 <= stop_idx < len(opt_surfaces):
            solved_stop_idx = stop_idx

    if solved_stop_idx is None:
        for idx, s in enumerate(opt_surfaces):
            s_name = str(getattr(s, "name", "")).upper()
            stype_str = str(getattr(s, "surface_type", "")).upper()
            if (getattr(s, "is_stop", False) or getattr(s, "is_aperture_stop", False)
                or s.surface_type == SurfaceType.STOP or "STOP" in stype_str or "STOP" in s_name or "(STO)" in s_name):
                solved_stop_idx = idx
                break
        if solved_stop_idx is None:
            solved_stop_idx = 0

    # 2. Entrance Pupil Calculation
    tracer = MatrixTracer()
    z_enp = 0.0
    stop_dia = 2.0 * get_surface_clear_radius(opt_surfaces[solved_stop_idx], default_val=12.5)
    if stop_dia <= 0.0:
        stop_dia = 20.0

    if solved_stop_idx > 0:
        front_elements = opt_surfaces[:solved_stop_idx]
        M_front = tracer.build_from_system(front_elements, wavelength_um=wavelength_um)
        t_gap = max(0.0, float(getattr(front_elements[-1], "thickness", 0.0)))
        g_last = str(getattr(front_elements[-1], "glass", "AIR") or "AIR").upper()
        n_gap = _safe_n(g_last)
        M_to_stop = space_matrix(t_gap, n_gap) @ M_front
        if abs(M_to_stop.A) > 1e-9:
            z_enp = M_to_stop.B / M_to_stop.A
            auto_epd = stop_dia / abs(M_to_stop.A)
        else:
            auto_epd = stop_dia
    else:
        auto_epd = stop_dia

    if epd is not None and epd > 0.0:
        eff_epd = float(epd)
    else:
        eff_epd = auto_epd
    r_ep = max(0.1, eff_epd / 2.0)

    # 3. Object Conjugate & Initial Conditions
    is_inf = (start_idx == 0 or s0 <= 0.0 or math.isinf(s0) or s0 > 1e7)
    eff_field_deg = float(field_angle_deg)
    eff_obj_h = float(object_height) if object_height is not None else None
    if eff_field_deg == 0.0 and eff_obj_h is None:
        if config is not None and hasattr(config, "fields") and config.fields:
            eff_field_deg = max(float(getattr(f, "y", 0.0)) for f in config.fields)
    if eff_field_deg == 0.0 and (eff_obj_h is None or eff_obj_h == 0.0):
        if is_inf:
            eff_field_deg = 1.0  # 1.0 deg reference field for canonical chief ray
        else:
            eff_obj_h = max(1.0, r_ep * 0.25)

    if is_inf:
        y_m = r_ep
        u_m = 0.0
        theta_rad = math.radians(eff_field_deg)
        u_c = math.tan(theta_rad)
        y_c = -z_enp * u_c
    else:
        denom = s0 + z_enp if abs(s0 + z_enp) > 1e-6 else s0
        u_m = r_ep / denom
        y_m = u_m * s0
        theta_rad = math.radians(eff_field_deg)
        h_obj = float(eff_obj_h) if eff_obj_h is not None else s0 * math.tan(theta_rad)
        u_c = -h_obj / denom
        y_c = h_obj + s0 * u_c

    # 4. Paraxial Sequential Trace
    z_curr = 0.0 if is_inf else (s0 if start_idx == 1 else 0.0)
    n_prev = 1.0

    surfaces_data = []
    z_list = []
    ym_list = []
    yc_list = []
    env_up_list = []
    env_dn_list = []
    sd_list = []
    vig_factors = []

    for i, s in enumerate(opt_surfaces):
        z_list.append(float(z_curr))
        ym_list.append(float(y_m))
        yc_list.append(float(y_c))
        env_up = float(y_c + abs(y_m))
        env_dn = float(y_c - abs(y_m))
        env_up_list.append(env_up)
        env_dn_list.append(env_dn)

        sd = get_surface_clear_radius(s, default_val=0.0)
        if sd <= 0.0:
            sd = max(12.5, abs(y_m) * 1.2)
        sd_list.append(float(sd))

        beam_radius = abs(y_c) + abs(y_m)
        clearance = sd - beam_radius
        if abs(y_m) > 1e-9:
            vig_ratio = max(0.0, min(1.0, (sd - abs(y_c)) / abs(y_m)))
        else:
            vig_ratio = 1.0 if abs(y_c) <= sd else 0.0
        vig_factor = 1.0 - vig_ratio
        vig_factors.append(float(vig_factor))

        surfaces_data.append({
            "surface_index": start_idx + i,
            "name": getattr(s, "name", f"Surface_{start_idx + i}"),
            "z_global": float(z_curr),
            "y_marginal": float(y_m),
            "u_marginal": float(u_m),
            "y_chief": float(y_c),
            "u_chief": float(u_c),
            "semi_diameter": float(sd),
            "beam_radius": float(beam_radius),
            "clearance": float(clearance),
            "vignetting_factor": float(vig_factor),
            "is_vignetted": bool(vig_factor > 1e-6)
        })

        if i < num_opt - 1:
            g_curr = str(getattr(s, "glass", "AIR") or "AIR").upper()
            is_mirror = ("MIRROR" in g_curr or "REFL" in g_curr or s.surface_type == SurfaceType.MIRROR)
            n_curr = 1.0 if is_mirror else _safe_n(g_curr)
            R = float(getattr(s, "radius", 0.0))
            if is_mirror:
                phi = (2.0 * n_prev) / R if abs(R) > 1e-9 else 0.0
                n_out = n_prev
            else:
                phi = (n_curr - n_prev) / R if abs(R) > 1e-9 else 0.0
                n_out = n_curr

            u_m_prime = (n_prev * u_m - y_m * phi) / n_out
            u_c_prime = (n_prev * u_c - y_c * phi) / n_out
            t = float(getattr(s, "thickness", 0.0))

            y_m = y_m + t * u_m_prime
            y_c = y_c + t * u_c_prime
            u_m = u_m_prime
            u_c = u_c_prime
            n_prev = n_out
            z_curr += t

    sys_vig = max(vig_factors) if vig_factors else 0.0
    lim_idx = int(np.argmax(vig_factors)) if vig_factors else 0
    lim_name = opt_surfaces[lim_idx].name if lim_idx < len(opt_surfaces) else "None"

    return {
        "surfaces_data": surfaces_data,
        "system_vignetting_factor": float(sys_vig),
        "is_vignetted": bool(sys_vig > 1e-6),
        "limiting_surface_index": start_idx + lim_idx,
        "limiting_surface_name": lim_name,
        "marginal_trajectory": {"z": z_list, "y": ym_list},
        "chief_trajectory": {"z": z_list, "y": yc_list},
        "envelope_upper": {"z": z_list, "y": env_up_list},
        "envelope_lower": {"z": z_list, "y": env_dn_list},
        "semi_diameters": sd_list,
        "stop_index": start_idx + solved_stop_idx,
        "enp_pos": float(z_enp),
        "epd": float(2.0 * r_ep)
    }


def compute_stops_windows_and_lagrange(
    surfaces: List[Surface],
    wavelength_um: float = 0.5876,
    epd: Optional[float] = None,
    field_angle_deg: float = 0.0,
    object_height: Optional[float] = None,
    config: Optional[Any] = None,
    eval_surface_idx: Optional[int] = None,
    first_order_data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Rigorously compute Aperture Stop, Field Stop, Entrance & Exit Pupils,
    Entrance & Exit Windows, and Helmholtz-Lagrange Optical Invariant conservation.

    Follows classical optical design definitions (Zemax / Born & Wolf / Kidger):
    - Aperture Stop (AS): Physical element limiting the axial ray cone.
    - Field Stop (FS): Physical element limiting the chief ray / field of view angle (arg min sd_i / |yc_i|).
    - Entrance Pupil (EnP): Paraxial image of Aperture Stop in object space.
    - Exit Pupil (ExP): Paraxial image of Aperture Stop in image space.
    - Entrance Window (EnW): Paraxial image of Field Stop in object space (subtends object FOV).
    - Exit Window (ExW): Paraxial image of Field Stop in image space (subtends image FOV).
    - Helmholtz-Lagrange Invariant: H = n * (yc * um - ym * uc) = const throughout all media.
    """
    if not surfaces or len(surfaces) < 2:
        return {}

    if first_order_data is not None:
        fod = first_order_data
    else:
        fod = calculate_first_order_data(
            surfaces,
            wavelength_um=wavelength_um,
            epd=epd,
            config=config,
            eval_surface_idx=eval_surface_idx,
            skip_stops_and_lagrange=True
        )
    tracer = MatrixTracer()

    end_idx = eval_surface_idx if (eval_surface_idx is not None and eval_surface_idx > 0) else len(surfaces) - 1
    end_idx = min(end_idx, len(surfaces) - 1)

    start_idx = 0
    s0 = float(getattr(surfaces[0], "thickness", 0.0))
    if len(surfaces) > 1 and surfaces[0].surface_type == SurfaceType.OBJECT:
        g = str(getattr(surfaces[0], "glass", "") or "").upper()
        if not g or g in ("AIR", "NONE"):
            start_idx = 1

    opt_surfaces = surfaces[start_idx:end_idx + 1]
    num_opt = len(opt_surfaces)

    try:
        from optics_app.utils.materials import get_refractive_index
    except ImportError:
        try:
            from optics_app.engine.dispersion import get_refractive_index
        except ImportError:
            def get_refractive_index(glass, wl):
                return 1.0

    def _safe_idx(glass_val: Any) -> float:
        if not glass_val:
            return 1.0
        g_str = str(glass_val).strip().upper()
        if not g_str or g_str in ("AIR", "NONE", "VACUUM", ""):
            return 1.0
        try:
            return float(get_refractive_index(g_str, wavelength_um))
        except Exception:
            return 1.5168

    # 1. Identify Stop
    solved_stop_idx = None
    for idx, s in enumerate(opt_surfaces):
        s_name = str(getattr(s, "name", "")).upper()
        stype_str = str(getattr(s, "surface_type", "")).upper()
        if (getattr(s, "is_stop", False) or getattr(s, "is_aperture_stop", False)
            or s.surface_type == SurfaceType.STOP or "STOP" in stype_str or "STOP" in s_name or "(STO)" in s_name):
            solved_stop_idx = idx
            break
    if solved_stop_idx is None:
        solved_stop_idx = 0

    stop_surf = opt_surfaces[solved_stop_idx]
    r_stop = get_surface_clear_radius(stop_surf, default_val=10.0)
    if r_stop <= 0.0:
        r_stop = 10.0

    # 2. Entrance & Exit Pupil calculation
    eff_epd = fod.get("epd", 2.0 * r_stop)
    r_ep = max(0.1, eff_epd / 2.0)
    z_enp = fod.get("enp_pos", 0.0)
    enp_z_global = fod.get("global_positions", {}).get("enp_z", z_enp)
    exp_pos = fod.get("exp_pos", 0.0)
    exp_z_global = fod.get("global_positions", {}).get("exp_z", 0.0)
    expd = fod.get("expd", 2.0 * r_stop)
    r_exp = expd / 2.0

    # 3. Determine Field Conditions
    eff_field_deg = float(field_angle_deg)
    eff_obj_h = float(object_height) if object_height is not None else None
    if eff_field_deg == 0.0 and eff_obj_h is None:
        if config is not None and hasattr(config, "fields") and config.fields:
            eff_field_deg = max(float(getattr(f, "y", 0.0)) for f in config.fields)
    is_inf = (start_idx == 0 or s0 <= 0.0 or math.isinf(s0) or s0 > 1e7)
    if eff_field_deg == 0.0 and (eff_obj_h is None or eff_obj_h == 0.0):
        if is_inf:
            eff_field_deg = 1.0  # 1.0 deg reference field for canonical chief ray
        else:
            eff_obj_h = max(1.0, r_ep * 0.25)

    # 4. Paraxial Marginal & Chief Ray Trace
    curr_n = 1.0
    u_m = 0.0 if is_inf else (r_ep / (s0 + z_enp if abs(s0 + z_enp) > 1e-6 else s0))
    y_m = r_ep if is_inf else (u_m * s0)

    theta_rad = math.radians(eff_field_deg)
    if is_inf:
        u_c = math.tan(theta_rad)
        y_c = -z_enp * u_c
    else:
        h_val = float(eff_obj_h) if eff_obj_h is not None else s0 * math.tan(theta_rad)
        denom = s0 + z_enp if abs(s0 + z_enp) > 1e-6 else s0
        u_c = -h_val / denom
        y_c = h_val + s0 * u_c

    # Surface-by-surface propagation
    z_curr = 0.0 if is_inf else (s0 if start_idx == 1 else 0.0)
    ym_list = []
    um_list = []
    ump_list = []
    yc_list = []
    uc_list = []
    ucp_list = []
    z_list = []
    n_in_list = []
    n_out_list = []
    Hin_list = []
    Hout_list = []
    delano_coords = []

    curr_n = 1.0
    for i, s in enumerate(opt_surfaces):
        z_list.append(float(z_curr))
        ym_list.append(float(y_m))
        yc_list.append(float(y_c))
        um_list.append(float(u_m))
        uc_list.append(float(u_c))
        n_in_list.append(float(curr_n))
        delano_coords.append((float(y_m), float(y_c)))

        H_in = curr_n * (y_c * u_m - y_m * u_c)
        Hin_list.append(float(H_in))

        # Refraction/Reflection
        g_curr = str(getattr(s, "glass", "AIR") or "AIR").upper()
        is_mirror = ("MIRROR" in g_curr or "REFL" in g_curr or s.surface_type == SurfaceType.MIRROR)
        next_n = 1.0 if is_mirror else _safe_idx(g_curr)
        R = float(getattr(s, "radius", 0.0))
        if is_mirror:
            phi = (2.0 * curr_n) / R if abs(R) > 1e-9 else 0.0
            n_out = curr_n
        else:
            phi = (next_n - curr_n) / R if abs(R) > 1e-9 else 0.0
            n_out = next_n

        u_m_prime = (curr_n * u_m - y_m * phi) / n_out
        u_c_prime = (curr_n * u_c - y_c * phi) / n_out
        ump_list.append(float(u_m_prime))
        ucp_list.append(float(u_c_prime))
        n_out_list.append(float(n_out))

        H_out = n_out * (y_c * u_m_prime - y_m * u_c_prime)
        Hout_list.append(float(H_out))

        t = float(getattr(s, "thickness", 0.0))
        y_m = y_m + t * u_m_prime
        y_c = y_c + t * u_c_prime
        u_m = u_m_prime
        u_c = u_c_prime
        curr_n = n_out
        z_curr += t

    # 5. Field Stop Identification
    # FS is the surface i != solved_stop_idx with minimum semi_diameter / |y_c|
    fs_idx_opt = None
    min_ratio = float("inf")
    for i, s in enumerate(opt_surfaces):
        if i == solved_stop_idx:
            continue
        sd = get_surface_clear_radius(s, default_val=12.5)
        if sd <= 0.0:
            sd = 12.5
        yc_val = abs(yc_list[i])
        if yc_val > 1e-7:
            ratio = sd / yc_val
            if ratio < min_ratio:
                min_ratio = ratio
                fs_idx_opt = i
    if fs_idx_opt is None:
        fs_idx_opt = num_opt - 1  # Fallback to image plane

    field_stop_surf = opt_surfaces[fs_idx_opt]
    r_fs = get_surface_clear_radius(field_stop_surf, default_val=10.0)
    if r_fs <= 0.0:
        r_fs = 10.0
    fs_global_idx = start_idx + fs_idx_opt
    fs_z_global = z_list[fs_idx_opt]

    # Calculate Maximum Unvignetted FOV from Field Stop
    if is_inf:
        max_fov_deg = eff_field_deg * min_ratio if not math.isinf(min_ratio) else 5.0
        max_obj_h = None
    else:
        h_ref = float(eff_obj_h) if eff_obj_h is not None else s0 * math.tan(theta_rad)
        max_obj_h = h_ref * min_ratio if not math.isinf(min_ratio) else 10.0
        max_fov_deg = math.degrees(math.atan(max_obj_h / s0)) if s0 > 0 else 5.0

    # 6. Entrance Window (EnW) & Exit Window (ExW)
    v1_z = fod.get("global_positions", {}).get("v1_z", 0.0)
    v2_z = fod.get("global_positions", {}).get("v2_z", v1_z + fod.get("totr", 0.0))

    if fs_idx_opt > 0:
        front_fs = opt_surfaces[:fs_idx_opt]
        M_front_fs = tracer.build_from_system(front_fs, wavelength_um=wavelength_um)
        t_gap_fs = max(0.0, float(getattr(front_fs[-1], "thickness", 0.0)))
        g_gap_fs = str(getattr(front_fs[-1], "glass", "AIR") or "AIR").upper()
        n_gap_fs = _safe_idx(g_gap_fs)
        M_to_fs = space_matrix(t_gap_fs, n_gap_fs) @ M_front_fs
        if abs(M_to_fs.A) > 1e-9:
            enw_pos = M_to_fs.B / M_to_fs.A
            enw_d = 2.0 * r_fs / abs(M_to_fs.A)
        else:
            enw_pos = 0.0
            enw_d = 2.0 * r_fs
    else:
        enw_pos = 0.0
        enw_d = 2.0 * r_fs
    enw_z_global = v1_z + enw_pos

    # ExW = image of Field Stop formed by elements following FS into image space
    rear_fs = opt_surfaces[fs_idx_opt + 1:]
    if rear_fs:
        t_fs = max(0.0, float(getattr(field_stop_surf, "thickness", 0.0)))
        g_fs = str(getattr(field_stop_surf, "glass", "AIR") or "AIR").upper()
        n_fs = _safe_idx(g_fs)
        T_fs = space_matrix(t_fs, n_fs)
        M_rear_fs = tracer.build_from_system(rear_fs, wavelength_um=wavelength_um)
        M_from_fs = M_rear_fs @ T_fs
        if abs(M_from_fs.D) > 1e-9:
            exw_pos = -M_from_fs.B / M_from_fs.D
            exw_d = 2.0 * r_fs / abs(M_from_fs.D)
        else:
            exw_pos = 0.0
            exw_d = 2.0 * r_fs
    else:
        exw_pos = 0.0
        exw_d = 2.0 * r_fs
    exw_z_global = v2_z + exw_pos

    # Angular subtense of Windows
    dist_enw_enp = abs(enw_z_global - enp_z_global)
    enw_fov_deg = 2.0 * math.degrees(math.atan((enw_d / 2.0) / dist_enw_enp)) if dist_enw_enp > 1e-3 else 2.0 * max_fov_deg

    dist_exw_exp = abs(exw_z_global - exp_z_global)
    exw_fov_deg = 2.0 * math.degrees(math.atan((exw_d / 2.0) / dist_exw_exp)) if dist_exw_exp > 1e-3 else 2.0 * max_fov_deg

    # 7. Helmholtz-Lagrange Conservation Analysis
    H_nominal = Hin_list[0]
    errors = [abs(H - H_nominal) / max(1e-12, abs(H_nominal)) for H in Hout_list]
    max_err = max(errors) if errors else 0.0
    is_conserved = bool(max_err < 1e-9)

    etendue_G = (math.pi**2) * (H_nominal**2)

    u_m_obj = um_list[0]
    u_m_img = ump_list[-1]
    y_c_obj = yc_list[0]
    y_c_img = yc_list[-1]
    n_obj = n_in_list[0]
    n_img = n_out_list[-1]

    if abs(u_m_img) > 1e-12:
        m_transverse = (n_obj * u_m_obj) / (n_img * u_m_img)
        m_angular = u_m_img / u_m_obj if abs(u_m_obj) > 1e-12 else float("inf")
    else:
        m_transverse = float("inf")
        m_angular = 0.0
    pupil_mag = expd / eff_epd if eff_epd > 1e-9 else 1.0

    poly_area = 0.0
    for i in range(len(delano_coords) - 1):
        y1, yb1 = delano_coords[i]
        y2, yb2 = delano_coords[i + 1]
        poly_area += 0.5 * abs(y1 * yb2 - yb1 * y2)

    surface_steps = []
    for i in range(num_opt):
        surface_steps.append({
            "surface_index": start_idx + i,
            "name": getattr(opt_surfaces[i], "name", f"Surface_{start_idx+i}"),
            "z_global": float(z_list[i]),
            "y_marginal": float(ym_list[i]),
            "u_marginal_in": float(um_list[i]),
            "u_marginal_out": float(ump_list[i]),
            "y_chief": float(yc_list[i]),
            "u_chief_in": float(uc_list[i]),
            "u_chief_out": float(ucp_list[i]),
            "n_in": float(n_in_list[i]),
            "n_out": float(n_out_list[i]),
            "H_in": float(Hin_list[i]),
            "H_out": float(Hout_list[i]),
            "delta_H": float(Hout_list[i] - Hin_list[i]),
            "relative_error": float(errors[i]),
            "delano_coord": (float(delano_coords[i][0]), float(delano_coords[i][1])),
        })

    cardinal_invariants = {
        "object_plane": {
            "z": float(z_list[0]),
            "H": float(Hin_list[0]),
            "formula": "H = n_obj * (y_c * u_m - y_m * u_c) = n_obj * h_obj * NA_obj",
            "description": f"Object plane: y_m={ym_list[0]:.2f} mm, y_c={yc_list[0]:.2f} mm"
        },
        "entrance_pupil": {
            "z": float(enp_z_global),
            "H": float(-n_obj * r_ep * um_list[0] if is_inf else -n_obj * r_ep * uc_list[0]),
            "formula": "H = -n_obj * r_EnP * u_c_obj (at pupil y_c = 0)",
            "description": f"EnP: Z={enp_z_global:.2f} mm, dia={eff_epd:.2f} mm"
        },
        "aperture_stop": {
            "z": float(z_list[solved_stop_idx]),
            "H": float(-n_in_list[solved_stop_idx] * r_stop * uc_list[solved_stop_idx]),
            "formula": "H = -n_stop * r_stop * u_c_stop (at stop y_c = 0)",
            "description": f"Stop ({stop_surf.name}): Z={z_list[solved_stop_idx]:.2f} mm, dia={2.0*r_stop:.2f} mm"
        },
        "exit_pupil": {
            "z": float(exp_z_global),
            "H": float(-n_img * r_exp * ucp_list[-1]),
            "formula": "H = -n_img * r_ExP * u_c_img (at exit pupil y_c = 0)",
            "description": f"ExP: Z={exp_z_global:.2f} mm, dia={expd:.2f} mm"
        },
        "field_stop": {
            "z": float(fs_z_global),
            "H": float(Hin_list[fs_idx_opt]),
            "formula": "H = n_FS * (r_FS * u_m - y_m * u_c) (at field stop y_c = r_FS)",
            "description": f"Field Stop ({field_stop_surf.name}): Z={fs_z_global:.2f} mm, dia={2.0*r_fs:.2f} mm"
        },
        "image_plane": {
            "z": float(z_list[-1]),
            "H": float(Hout_list[-1]),
            "formula": "H = n_img * h_img * u'_m_img = n_img * h_img * NA_img",
            "description": f"Image plane: y_m={ym_list[-1]:.3f} mm, y_c={yc_list[-1]:.3f} mm"
        }
    }

    abbe_comparison = []
    effl_val = abs(fod.get("effl", 100.0))
    for zone in [0.25, 0.50, 0.707, 1.00]:
        y_zone = r_ep * zone
        u_zone = math.asin(min(1.0, y_zone / effl_val)) if effl_val > 0 else 0.0
        abbe_val = n_obj * y_c_obj * math.sin(u_zone) if not is_inf else n_obj * y_zone * math.sin(theta_rad)
        disc = abs(abbe_val - abs(H_nominal) * zone) / max(1e-12, abs(H_nominal) * zone) * 100.0 if H_nominal != 0 else 0.0
        note = "Excellent paraxial/Abbe agreement" if disc < 1.0 else f"High-NA non-paraxial spherical departure ({disc:.1f}%)"
        abbe_comparison.append({
            "pupil_zone": zone,
            "paraxial_H": float(abs(H_nominal) * zone),
            "abbe_sine_invariant": float(abbe_val),
            "discrepancy_pct": float(disc),
            "optical_aberration_note": note
        })

    return {
        "aperture_stop": {
            "surface_index": start_idx + solved_stop_idx,
            "name": stop_surf.name,
            "z_global": float(z_list[solved_stop_idx]),
            "semi_diameter": float(r_stop),
            "diameter": float(2.0 * r_stop),
            "is_explicit": bool(stop_surf.surface_type == SurfaceType.STOP or "STOP" in stop_surf.name.upper()),
        },
        "field_stop": {
            "surface_index": fs_global_idx,
            "name": field_stop_surf.name,
            "z_global": float(fs_z_global),
            "semi_diameter": float(r_fs),
            "diameter": float(2.0 * r_fs),
            "limiting_ratio": float(min_ratio),
            "max_unvignetted_field_deg": float(max_fov_deg),
            "max_unvignetted_object_height": float(max_obj_h) if max_obj_h is not None else 0.0,
        },
        "entrance_pupil": {
            "z_global": float(enp_z_global),
            "z_from_s1": float(z_enp),
            "diameter": float(eff_epd),
            "semi_diameter": float(r_ep),
        },
        "exit_pupil": {
            "z_global": float(exp_z_global),
            "z_from_rear": float(exp_pos),
            "diameter": float(expd),
            "semi_diameter": float(r_exp),
            "distance_to_image": float(fod.get("exp_to_image", 0.0)),
            "pupil_magnification": float(pupil_mag),
        },
        "entrance_window": {
            "z_global": float(enw_z_global),
            "z_from_s1": float(enw_pos),
            "diameter": float(enw_d),
            "semi_diameter": float(enw_d / 2.0),
            "angular_fov_deg": float(enw_fov_deg),
        },
        "exit_window": {
            "z_global": float(exw_z_global),
            "z_from_rear": float(exw_pos),
            "diameter": float(exw_d),
            "semi_diameter": float(exw_d / 2.0),
            "image_fov_deg": float(exw_fov_deg),
        },
        "helmholtz_lagrange": {
            "lagrange_invariant_H": float(H_nominal),
            "mean_H": float(np.mean(Hin_list)),
            "max_conservation_error": float(max_err),
            "is_conserved": bool(is_conserved),
            "etendue_G": float(etendue_G),
            "transverse_magnification": float(m_transverse),
            "angular_magnification": float(m_angular),
            "pupil_magnification": float(pupil_mag),
            "smith_helmholtz_product": float(m_transverse * m_angular) if abs(m_angular) < 1e9 else 1.0,
            "smith_helmholtz_target": float(n_obj / n_img),
            "cardinal_invariants": cardinal_invariants,
            "surface_steps": surface_steps,
            "delano_diagram": {
                "ym": ym_list,
                "yc": yc_list,
                "labels": [s.name for s in opt_surfaces],
                "polygon_area": float(poly_area),
            },
            "non_paraxial_abbe_comparison": abbe_comparison,
        },
        "paraxial_marginal_ray": {
            "z": z_list,
            "y": ym_list,
            "u": um_list,
            "u_prime": ump_list,
        },
        "paraxial_chief_ray": {
            "z": z_list,
            "y": yc_list,
            "u": uc_list,
            "u_prime": ucp_list,
        }
    }


def calculate_first_order_data(
    surfaces: List[Surface],
    wavelength_um: float = 0.5876,
    epd: Optional[float] = None,
    config: Optional[Any] = None,
    eval_surface_idx: Optional[int] = None,
    skip_stops_and_lagrange: bool = False
) -> Dict[str, Any]:
    """
    Calculate paraxial first-order properties following Zemax OpticStudio standards.

    Parameters:
    - surfaces: Optical surface stack.
    - wavelength_um: Evaluation wavelength in micrometers (default 0.5876 d-line).
    - epd: Optional Entrance Pupil Diameter in mm.
    - config: Optional SystemConfig object.
    - eval_surface_idx: Index of image/detector surface (None = last surface).

    Returns:
    Dictionary containing all first-order parameters, cardinal points, pupil data, and diagnostics.
    """
    if not surfaces:
        return {
            "effl": float("inf"),
            "bfl": float("inf"),
            "ffl": float("inf"),
            "p1": 0.0,
            "p2": 0.0,
            "inter_p": 0.0,
            "n1": 0.0,
            "n2": 0.0,
            "f1_dist": 0.0,
            "f2_dist": 0.0,
            "epd": 0.0,
            "enp_pos": 0.0,
            "expd": 0.0,
            "exp_pos": 0.0,
            "exp_to_image": 0.0,
            "pupil_magnification": 1.0,
            "optical_power_diopters": 0.0,
            "optical_power_inv_mm": 0.0,
            "f_number": float("inf"),
            "working_f_number": float("inf"),
            "na": 0.0,
            "totr": 0.0,
            "lens_thickness": 0.0,
            "air_thickness": 0.0,
            "global_positions": {},
            "paraxial_diagnostics": ["No surfaces in system"],
            "matrix": RayMatrix(),
            "wavelength_um": wavelength_um,
            "is_afocal": True,
            "error": "No surfaces in system"
        }

    # Determine image plane index
    end_idx = eval_surface_idx if (eval_surface_idx is not None and eval_surface_idx > 0) else len(surfaces) - 1
    end_idx = min(end_idx, len(surfaces) - 1)

    # Determine first optical surface (skip object surface if dummy plane with zero thickness or air)
    start_idx = 0
    if len(surfaces) > 1 and surfaces[0].surface_type == SurfaceType.OBJECT:
        g = str(getattr(surfaces[0], "glass", "") or "").upper()
        if not g or g in ("AIR", "NONE"):
            start_idx = 1

    opt_surfaces = surfaces[start_idx:end_idx]

    # Calculate Total Track (TOTR) and Glass vs Air breakdown
    totr = 0.0
    lens_thickness = 0.0
    for s in surfaces[start_idx:end_idx]:
        t = max(0.0, float(getattr(s, "thickness", 0.0)))
        totr += t
        g = str(getattr(s, "glass", "AIR") or "AIR").upper()
        if g not in ("AIR", "NONE", "MIRROR", "MIRR", ""):
            lens_thickness += t
    air_thickness = max(0.0, totr - lens_thickness)

    # Track Global Z coordinates along the optical train
    z_curr = 0.0
    for i in range(start_idx):
        t_val = float(getattr(surfaces[i], "thickness", 0.0))
        if math.isinf(t_val) or t_val > 1e7:
            continue
        z_curr += max(0.0, t_val)
    v1_z = z_curr

    # Collect surface vertex Z positions for optical elements
    surf_vertex_z = []
    accum_z = v1_z
    for s in opt_surfaces:
        surf_vertex_z.append(accum_z)
        accum_z += max(0.0, float(getattr(s, "thickness", 0.0)))
    image_z = accum_z
    v2_z = surf_vertex_z[-1] if surf_vertex_z else v1_z

    # Build ABCD Matrix through optical elements
    tracer = MatrixTracer()
    mat = tracer.build_from_system(opt_surfaces, wavelength_um=wavelength_um)

    c_val = mat.C
    is_afocal = abs(c_val) < 1e-12
    effl_tmp = (-1.0 / c_val) if not is_afocal else float("inf")

    # Determine Entrance Pupil Diameter (EPD) following the Zemax System Explorer aperture type
    eff_epd = None
    if epd is not None and epd > 0.0:
        eff_epd = float(epd)
    elif config is not None and float(getattr(config, "aperture_value", 0.0) or 0.0) > 0.0:
        ap_type = getattr(config, "aperture_type", None)
        ap_name = getattr(ap_type, "name", str(ap_type))
        ap_val = float(config.aperture_value)
        if ap_name == "ENTRANCE_PUPIL_DIAMETER":
            eff_epd = ap_val
        elif ap_name in ("IMAGE_SPACE_F_NUM", "PARAXIAL_WORKING_F_NUM") and not is_afocal:
            eff_epd = abs(effl_tmp) / ap_val
    if eff_epd is None:
        # Float by Stop Size / fallback: stop surface or first optical surface clear aperture
        stop_surfs = [s for s in surfaces if s.surface_type == SurfaceType.STOP]
        if stop_surfs:
            eff_epd = 2.0 * get_surface_clear_radius(stop_surfs[0], default_val=12.5)
        elif opt_surfaces:
            eff_epd = 2.0 * get_surface_clear_radius(opt_surfaces[0], default_val=12.5)
        else:
            eff_epd = 20.0

    if not is_afocal:
        effl = -1.0 / c_val
        bfl = -mat.A / c_val
        ffl = -mat.D / c_val
        p1 = (mat.D - 1.0) / c_val
        p2 = (1.0 - mat.A) / c_val
        n1 = p1
        n2 = p2

        f_number = abs(effl) / eff_epd if eff_epd > 0.0 else float("inf")
        theta_m = math.atan(eff_epd / (2.0 * abs(effl))) if effl != 0 else 0.0
        na = math.sin(theta_m)
        working_f_number = 1.0 / (2.0 * na) if na > 1e-9 else float("inf")
        optical_power_dpt = 1000.0 / effl
        optical_power_inv_mm = 1.0 / effl
    else:
        effl = float("inf")
        bfl = float("inf")
        ffl = float("inf")
        p1 = 0.0
        p2 = 0.0
        n1 = 0.0
        n2 = 0.0
        f_number = float("inf")
        working_f_number = float("inf")
        na = 0.0
        optical_power_dpt = 0.0
        optical_power_inv_mm = 0.0

    # -------------------------------------------------------------
    # Pupil Imaging: Entrance Pupil and Exit Pupil (Effective Back Pupil)
    # -------------------------------------------------------------
    stop_idx = None
    for idx, s in enumerate(opt_surfaces):
        s_name = getattr(s, "name", "").upper()
        if s.surface_type == SurfaceType.STOP or s_name == "STO" or "STOP" in s_name:
            stop_idx = idx
            break

    try:
        from optics_app.utils.materials import get_refractive_index
    except ImportError:
        try:
            from optics_app.engine.dispersion import get_refractive_index
        except ImportError:
            def get_refractive_index(glass, wl):
                return 1.0

    if stop_idx is not None:
        stop_surf = opt_surfaces[stop_idx]
        stop_dia = 2.0 * get_surface_clear_radius(stop_surf, default_val=10.0)
        if eff_epd is None or eff_epd <= 0.0:
            eff_epd = stop_dia

        # Entrance Pupil: Image of Stop through elements preceding the stop
        front_elements = opt_surfaces[:stop_idx]
        if front_elements:
            M_front = tracer.build_from_system(front_elements, wavelength_um=wavelength_um)
            # Add gap propagation from last front element to stop plane
            t_gap = max(0.0, float(getattr(front_elements[-1], "thickness", 0.0)))
            g_last = str(getattr(front_elements[-1], "glass", "AIR") or "AIR").upper()
            n_gap = float(get_refractive_index(g_last, wavelength_um)) if g_last not in ("AIR", "NONE", "") else 1.0
            M_to_stop = space_matrix(t_gap, n_gap) @ M_front

            if abs(M_to_stop.A) > 1e-9:
                enp_pos = M_to_stop.B / M_to_stop.A
                enp_d = stop_dia / abs(M_to_stop.A)
            else:
                enp_pos = 0.0
                enp_d = stop_dia
        else:
            enp_pos = 0.0
            enp_d = stop_dia

        # Exit Pupil: Image of Stop through elements following the stop
        rear_elements = opt_surfaces[stop_idx + 1:]
        if rear_elements:
            t_stop = max(0.0, float(getattr(stop_surf, "thickness", 0.0)))
            g_stop = str(getattr(stop_surf, "glass", "AIR") or "AIR").upper()
            n_stop = float(get_refractive_index(g_stop, wavelength_um)) if g_stop not in ("AIR", "NONE", "") else 1.0
            T_stop = space_matrix(t_stop, n_stop)

            M_rear = tracer.build_from_system(rear_elements, wavelength_um=wavelength_um)
            M_from_stop = M_rear @ T_stop

            if abs(M_from_stop.D) > 1e-9:
                exp_pos = -M_from_stop.B / M_from_stop.D
                expd = stop_dia / abs(M_from_stop.D)
            else:
                exp_pos = 0.0
                expd = stop_dia
        else:
            exp_pos = 0.0
            expd = stop_dia
    else:
        # Standard system: Entrance pupil at Surface 1 vertex
        enp_pos = 0.0
        enp_d = eff_epd
        if abs(mat.D) > 1e-9:
            exp_pos = -mat.B / mat.D
            expd = eff_epd / abs(mat.D)
        else:
            expd = eff_epd
            exp_pos = 0.0

    # Global positions of cardinal points
    p1_z = v1_z + p1
    p2_z = v2_z + p2
    inter_p = p2_z - p1_z
    f1_z = v1_z - ffl if not is_afocal else float("-inf")
    f2_z = v2_z + bfl if not is_afocal else float("inf")
    n1_z = v1_z + n1
    n2_z = v2_z + n2
    enp_z = v1_z + enp_pos
    exp_z = v2_z + exp_pos
    exp_to_image = image_z - exp_z
    pupil_magnification = (expd / enp_d) if enp_d > 1e-9 else 1.0

    # Paraxial diagnostic status flags
    diagnostics = []
    if is_afocal:
        diagnostics.append("Afocal / Infinite EFL (Parallel collimated beam)")
    elif effl > 0:
        diagnostics.append(f"Converging System (+{optical_power_dpt:.2f} dpt)")
    else:
        diagnostics.append(f"Diverging System ({optical_power_dpt:.2f} dpt)")

    if not is_afocal:
        if bfl > 0:
            diagnostics.append(f"Real Image Space Focus (BFL = {bfl:.2f} mm)")
        else:
            diagnostics.append(f"Virtual Image Space Focus (BFL = {bfl:.2f} mm < 0)")

        if abs(exp_pos) > 1000.0 or abs(expd) > 2000.0:
            diagnostics.append("Image-Space Telecentric (Near-Infinite Exit Pupil)")
        else:
            diagnostics.append(f"Effective Back Pupil: dia {expd:.2f} mm @ {exp_pos:+.2f} mm from rear vertex")

        if p2_z < p1_z:
            diagnostics.append("Inverted Principal Planes (P2 < P1: Telephoto / Retrofocus configuration)")
        else:
            diagnostics.append(f"Normal Principal Planes (Separation Delta P = {inter_p:.2f} mm)")

        if f_number < 1.0:
            diagnostics.append(f"Fast Optical Aperture (F/{f_number:.2f})")
        elif f_number > 20.0:
            diagnostics.append(f"Slow Optical Aperture (F/{f_number:.2f})")

    # Paraxial Marginal & Chief Ray Tracing and Vignetting Analysis
    field_deg = 0.0
    if config is not None and hasattr(config, "fields") and config.fields:
        field_deg = max(float(getattr(f, "y", 0.0)) for f in config.fields)

    paraxial_ray_data = compute_paraxial_marginal_and_principal_rays(
        surfaces=surfaces,
        wavelength_um=wavelength_um,
        epd=eff_epd,
        field_angle_deg=field_deg,
        stop_idx=stop_idx,
        eval_surface_idx=eval_surface_idx,
        config=config
    )

    if paraxial_ray_data.get("is_vignetted", False):
        vig_pct = paraxial_ray_data["system_vignetting_factor"] * 100.0
        lim_name = paraxial_ray_data.get("limiting_surface_name", "Unknown")
        lim_idx = paraxial_ray_data.get("limiting_surface_index", 0)
        diagnostics.append(f"Paraxial Aperture Vignetting: {vig_pct:.1f}% beam clipping at Surface #{lim_idx} ('{lim_name}')")

    # Stops, Windows, and Helmholtz-Lagrange Optical Invariant Analysis
    stops_windows_lagrange = {}
    if not skip_stops_and_lagrange:
        partial_fod = {
            "effl": float(effl),
            "epd": float(enp_d),
            "enp_pos": float(enp_pos),
            "expd": float(expd),
            "exp_pos": float(exp_pos),
            "totr": float(totr),
            "exp_to_image": float(exp_to_image),
            "global_positions": {
                "v1_z": float(v1_z),
                "v2_z": float(v2_z),
                "image_z": float(image_z),
                "enp_z": float(enp_z),
                "exp_z": float(exp_z)
            }
        }
        stops_windows_lagrange = compute_stops_windows_and_lagrange(
            surfaces=surfaces,
            wavelength_um=wavelength_um,
            epd=eff_epd,
            field_angle_deg=field_deg,
            config=config,
            eval_surface_idx=eval_surface_idx,
            first_order_data=partial_fod
        )

        if stops_windows_lagrange:
            if "paraxial_marginal_ray" in stops_windows_lagrange and "paraxial_chief_ray" in stops_windows_lagrange:
                paraxial_ray_data["marginal_trajectory"] = stops_windows_lagrange["paraxial_marginal_ray"]
                paraxial_ray_data["chief_trajectory"] = stops_windows_lagrange["paraxial_chief_ray"]

    return {
        "effl": float(effl),
        "bfl": float(bfl),
        "ffl": float(ffl),
        "p1": float(p1),
        "p2": float(p2),
        "inter_p": float(inter_p),
        "n1": float(n1),
        "n2": float(n2),
        "f1_dist": float(ffl),
        "f2_dist": float(bfl),
        "epd": float(enp_d),
        "enp_pos": float(enp_pos),
        "expd": float(expd),
        "exp_pos": float(exp_pos),
        "exp_to_image": float(exp_to_image),
        "pupil_magnification": float(pupil_magnification),
        "optical_power_diopters": float(optical_power_dpt),
        "optical_power_inv_mm": float(optical_power_inv_mm),
        "f_number": float(f_number),
        "working_f_number": float(working_f_number),
        "na": float(na),
        "totr": float(totr),
        "lens_thickness": float(lens_thickness),
        "air_thickness": float(air_thickness),
        "global_positions": {
            "v1_z": float(v1_z),
            "v2_z": float(v2_z),
            "image_z": float(image_z),
            "p1_z": float(p1_z),
            "p2_z": float(p2_z),
            "f1_z": float(f1_z),
            "f2_z": float(f2_z),
            "n1_z": float(n1_z),
            "n2_z": float(n2_z),
            "enp_z": float(enp_z),
            "exp_z": float(exp_z)
        },
        "paraxial_diagnostics": diagnostics,
        "matrix": mat,
        "wavelength_um": float(wavelength_um),
        "is_afocal": bool(is_afocal),
        "paraxial_rays": paraxial_ray_data,
        "stops_and_windows": stops_windows_lagrange,
        "aperture_stop": stops_windows_lagrange.get("aperture_stop", {}),
        "field_stop": stops_windows_lagrange.get("field_stop", {}),
        "entrance_pupil": stops_windows_lagrange.get("entrance_pupil", {}),
        "exit_pupil": stops_windows_lagrange.get("exit_pupil", {}),
        "entrance_window": stops_windows_lagrange.get("entrance_window", {}),
        "exit_window": stops_windows_lagrange.get("exit_window", {}),
        "helmholtz_lagrange": stops_windows_lagrange.get("helmholtz_lagrange", {}),
        "system_vignetting_factor": float(paraxial_ray_data.get("system_vignetting_factor", 0.0)),
        "is_vignetted": bool(paraxial_ray_data.get("is_vignetted", False)),
        "limiting_surface_index": int(paraxial_ray_data.get("limiting_surface_index", 0)),
        "limiting_surface_name": str(paraxial_ray_data.get("limiting_surface_name", "None")),
        "error": None
    }


def format_zemax_first_order_report(data: Dict[str, Any], title: str = "Optical System") -> str:
    """
    Format first-order data as a Zemax OpticStudio Prescription / First Order Report.
    """
    mat = data.get("matrix", RayMatrix())
    effl = data.get("effl", float("inf"))
    bfl = data.get("bfl", float("inf"))
    ffl = data.get("ffl", float("inf"))
    p1 = data.get("p1", 0.0)
    p2 = data.get("p2", 0.0)
    inter_p = data.get("inter_p", 0.0)
    n1 = data.get("n1", 0.0)
    n2 = data.get("n2", 0.0)
    epd = data.get("epd", 0.0)
    enp_pos = data.get("enp_pos", 0.0)
    expd = data.get("expd", 0.0)
    exp_pos = data.get("exp_pos", 0.0)
    exp_to_img = data.get("exp_to_image", 0.0)
    pupil_mag = data.get("pupil_magnification", 1.0)
    pwr_dpt = data.get("optical_power_diopters", 0.0)
    f_num = data.get("f_number", float("inf"))
    wf_num = data.get("working_f_number", float("inf"))
    na = data.get("na", 0.0)
    totr = data.get("totr", 0.0)
    t_glass = data.get("lens_thickness", 0.0)
    t_air = data.get("air_thickness", 0.0)
    wl = data.get("wavelength_um", 0.5876)
    is_afocal = data.get("is_afocal", False)
    gp = data.get("global_positions", {})
    diagnostics = data.get("paraxial_diagnostics", [])

    effl_str = "Afocal / Infinite" if math.isinf(effl) or is_afocal else f"{effl:12.6f} mm"
    bfl_str = "Afocal / Infinite" if math.isinf(bfl) or is_afocal else f"{bfl:12.6f} mm"
    ffl_str = "Afocal / Infinite" if math.isinf(ffl) or is_afocal else f"{ffl:12.6f} mm"
    pwr_str = "0.000000 dpt" if is_afocal else f"{pwr_dpt:+12.6f} dpt"
    f_num_str = "Afocal (Infinite)" if math.isinf(f_num) or is_afocal else f"{f_num:12.6f}"
    wf_num_str = "Afocal (Infinite)" if math.isinf(wf_num) or is_afocal else f"{wf_num:12.6f}"

    as_data = data.get("aperture_stop", {})
    fs_data = data.get("field_stop", {})
    enw_data = data.get("entrance_window", {})
    exw_data = data.get("exit_window", {})
    hl = data.get("helmholtz_lagrange", {})

    lines = [
        "*" * 79,
        f"                   ZEMAX OPTICSTUDIO FIRST ORDER DATA",
        f"                   Configuration: {title}",
        "*" * 79,
        "",
        f"Primary Wavelength (um)           :    {wl:10.6f} ({wl*1000.0:6.1f} nm)",
        f"Effective Focal Length (EFFL)     :    {effl_str}",
        f"Back Focal Length (BFL)           :    {bfl_str}",
        f"Front Focal Length (FFL)          :    {ffl_str}",
        f"Optical Power (Diopters)          :    {pwr_str}",
        f"Total Track Length (TOTR)         :    {totr:12.6f} mm  (Glass: {t_glass:.3f} mm, Air: {t_air:.3f} mm)",
        f"Paraxial Beam Vignetting         :    {data.get('system_vignetting_factor', 0.0)*100.0:12.1f}% beam clipping at Surface #{data.get('limiting_surface_index', 0)} ('{data.get('limiting_surface_name', 'None')}')" if data.get("is_vignetted") else "Paraxial Beam Vignetting         :    None (100.0% aperture transmission)",
        "",
        "STOPS AND WINDOWS (PUPILS & DELANO CONJUGATES):",
        f"   Aperture Stop (AS)             :    Surface #{as_data.get('surface_index', 0)} ('{as_data.get('name', 'STO')}'), dia = {as_data.get('diameter', 0.0):10.4f} mm @ Global Z = {as_data.get('z_global', 0.0):.3f} mm",
        f"   Field Stop (FS)                :    Surface #{fs_data.get('surface_index', 0)} ('{fs_data.get('name', 'FS')}'), dia = {fs_data.get('diameter', 0.0):10.4f} mm @ Global Z = {fs_data.get('z_global', 0.0):.3f} mm",
        f"   Max Unvignetted FOV (Half/Full):    {fs_data.get('max_unvignetted_field_deg', 0.0):10.4f} deg  /  {2.0*fs_data.get('max_unvignetted_field_deg', 0.0):.4f} deg",
        "",
        f"   Entrance Pupil Diameter (EPD)  :    {epd:12.6f} mm",
        f"   Entrance Pupil Position (EnP)  :    {enp_pos:12.6f} mm (from S1 vertex, Global Z = {gp.get('enp_z', 0.0):.3f} mm)",
        f"   Exit Pupil Diameter (EXPD)     :    {expd:12.6f} mm (Effective Back Pupil)",
        f"   Exit Pupil Position (ExP)      :    {exp_pos:12.6f} mm (from rear vertex, Global Z = {gp.get('exp_z', 0.0):.3f} mm)",
        f"   Exit Pupil to Image Distance   :    {exp_to_img:12.6f} mm",
        f"   Pupil Magnification (m_p)      :    {pupil_mag:12.6f}",
        "",
        f"   Entrance Window Diameter (EnW) :    {enw_data.get('diameter', 0.0):12.6f} mm @ Global Z = {enw_data.get('z_global', 0.0):.3f} mm",
        f"   Entrance Window Position       :    {enw_data.get('z_from_s1', 0.0):12.6f} mm from Surface 1 vertex",
        f"   Entrance Window Subtended FOV  :    {enw_data.get('angular_fov_deg', 0.0):12.6f} deg (seen from Entrance Pupil)",
        f"   Exit Window Diameter (ExW)     :    {exw_data.get('diameter', 0.0):12.6f} mm @ Global Z = {exw_data.get('z_global', 0.0):.3f} mm",
        f"   Exit Window Subtended Image FOV:    {exw_data.get('image_fov_deg', 0.0):12.6f} deg (seen from Exit Pupil)",
        "",
        "HELMHOLTZ-LAGRANGE OPTICAL INVARIANT CONSERVATION:",
        f"   Optical Invariant (H)          :    {hl.get('lagrange_invariant_H', 0.0):+12.6e} mm-rad",
        f"   Optical Etendue / Throughput(G):    {hl.get('etendue_G', 0.0):12.6e} mm^2-sr  (G = pi^2 * H^2)",
        f"   Surface Invariant Invariance   :    {'CONSERVED (Exact paraxial conservation)' if hl.get('is_conserved') else 'Approximated'}",
        f"   Max Conservation Delta Error   :    {hl.get('max_conservation_error', 0.0):12.2e} (|Delta H / H_0|)",
        f"   Transverse Paraxial Mag (m)    :    {hl.get('transverse_magnification', 0.0):12.6f}",
        f"   Angular Paraxial Mag (m_alpha) :    {hl.get('angular_magnification', 0.0):12.6f}",
        f"   Smith-Helmholtz Product(m*m_a) :    {hl.get('smith_helmholtz_product', 0.0):12.6f} (Target: n_obj/n_img = {hl.get('smith_helmholtz_target', 1.0):.6f})",
        f"   Delano Polygon Area (ym vs yc) :    {hl.get('delano_diagram', {}).get('polygon_area', 0.0):12.6f} mm^2",
        "",
        f"Image Space F-Number (F/#)        :    {f_num_str}",
        f"Working F-Number (W F/#)          :    {wf_num_str}",
        f"Paraxial Numerical Aperture (NA)  :    {na:12.6f}",
        "",
        f"Front Principal Plane (P1)        :    {p1:12.6f} mm (from S1 vertex, Global Z = {gp.get('p1_z', 0.0):.3f} mm)",
        f"Rear Principal Plane (P2)         :    {p2:12.6f} mm (from rear vertex, Global Z = {gp.get('p2_z', 0.0):.3f} mm)",
        f"Inter-Principal Separation (Delta P):  {inter_p:12.6f} mm",
        f"Front Nodal Plane (N1)            :    {n1:12.6f} mm (Global Z = {gp.get('n1_z', 0.0):.3f} mm)",
        f"Rear Nodal Plane (N2)             :    {n2:12.6f} mm (Global Z = {gp.get('n2_z', 0.0):.3f} mm)",
        f"Front Focal Point (F1)            :    Global Z = {gp.get('f1_z', 0.0):.3f} mm",
        f"Rear Focal Point (F2)             :    Global Z = {gp.get('f2_z', 0.0):.3f} mm",
        "",
        "Paraxial ABCD Ray Transfer Matrix:",
        f"   A = {mat.A:12.6f}       B = {mat.B:12.6f}",
        f"   C = {mat.C:12.6f}       D = {mat.D:12.6f}",
        f"   Determinant (AD - BC) = {mat.determinant():12.6f}",
        "",
        "Paraxial Diagnostics & Classification:"
    ]
    for d in diagnostics:
        lines.append(f"   * {d}")
    lines.append("*" * 79)
    return "\n".join(lines)


__all__ = [
    "calculate_first_order_data",
    "format_zemax_first_order_report",
    "compute_paraxial_marginal_and_principal_rays",
    "compute_stops_windows_and_lagrange"
]
