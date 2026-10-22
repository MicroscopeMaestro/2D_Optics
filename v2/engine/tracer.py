"""
3D Vector Sequential Ray Tracer, Polarization Physics & Gaussian Beam Engine.
"""
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Any, Optional, Union, Sequence
import math
import numpy as np

try:
    from optics_app.engine.surface import Surface, SurfaceType, SphericalSurface, AsphericSurface
    from optics_app.engine.ray import Ray, RayBundle
    from optics_app.engine.lde import SurfaceStack
    from optics_app.utils.materials import get_refractive_index, get_complex_refractive_index
    from optics_app.engine.polarization import (
        compute_fresnel_coefficients, compute_sp_basis, rotate_jones_vector, FresnelResult
    )
    from optics_app.engine.mechanisms import (
        get_effective_pose, interaction_rotation, active_state_name, state_transmission, is_dmd
    )
    from optics_app.engine.plm import (
        is_plm, ensure_plm_config, deflect_ray_on_plm_3d, evaluate_plm_phase_and_gradient
    )
except ImportError:
    from v2.engine.surface import Surface, SurfaceType, SphericalSurface, AsphericSurface
    from v2.engine.ray import Ray, RayBundle
    from v2.engine.lde import SurfaceStack
    from v2.engine.materials import get_refractive_index, get_complex_refractive_index
    from v2.engine.polarization import (
        compute_fresnel_coefficients, compute_sp_basis, rotate_jones_vector, FresnelResult
    )
    from v2.engine.mechanisms import (
        get_effective_pose, interaction_rotation, active_state_name, state_transmission, is_dmd
    )
    from v2.engine.plm import (
        is_plm, ensure_plm_config, deflect_ray_on_plm_3d, evaluate_plm_phase_and_gradient
    )


class _LocalRay:
    __slots__ = ('pos', 'dir', 'wavelength')
    def __init__(self, pos: np.ndarray, dir: np.ndarray, wavelength: float):
        self.pos = pos
        self.dir = dir
        self.wavelength = wavelength


def check_segment_surface_interference(p1_global: np.ndarray, p2_global: np.ndarray, surface: Surface) -> bool:
    """Checks if the line segment from P1 to P2 physically intersects the 3D solid volume of the given surface."""
    if not hasattr(surface, 'global_R') or not hasattr(surface, 'global_origin'):
        return False
    
    # Virtual planes, apertures, stops, and detectors do not physically block crossed beams
    stype = getattr(surface, 'surface_type', None)
    if stype in (SurfaceType.OBJECT, SurfaceType.STOP, SurfaceType.DETECTOR):
        return False
    glass_name = getattr(surface, 'glass', 'AIR') or 'AIR'
    if glass_name.upper() in ('AIR', ''):
        return False
        
    R_mat = surface.global_R
    origin = surface.global_origin
    
    # Transform to local frame
    p1_local = R_mat.T @ (p1_global - origin)
    p2_local = R_mat.T @ (p2_global - origin)
    
    z1 = float(p1_local[2])
    z2 = float(p2_local[2])
    
    # Determine the physical Z bounds of the solid element
    is_mirror = (glass_name.upper() in ('MIRROR', 'MIRR')) or (stype == SurfaceType.GRATING)
    if is_mirror:
        # Mirror/grating substrate is BEHIND the optical surface: z in [-sub_thick, 0]
        sub_thick = float(getattr(surface, 'substrate_thickness', 6.0))
        z_min = -sub_thick
        z_max = 0.0
    else:
        # Solid lens element spans from z = 0 to z = thickness
        z_min = 0.0
        z_max = float(getattr(surface, 'thickness', 3.0))

    # Check if segment passes through the solid Z span
    seg_z_min = min(z1, z2)
    seg_z_max = max(z1, z2)
    if seg_z_max < z_min or seg_z_min > z_max:
        return False
        
    # Check intersection with the boundary planes z_min and z_max
    sd = float(getattr(surface, 'semi_diameter', 12.5))
    if sd <= 0:
        sd = 12.5
        
    dz = z2 - z1
    if abs(dz) > 1e-6:
        for z_plane in (z_min, z_max):
            if (z1 <= z_plane <= z2) or (z2 <= z_plane <= z1):
                t = (z_plane - z1) / dz
                x_int = p1_local[0] + t * (p2_local[0] - p1_local[0])
                y_int = p1_local[1] + t * (p2_local[1] - p1_local[1])
                if (x_int**2 + y_int**2) <= sd**2:
                    return True
                    
    return False


def build_rotation_matrix(tilt_x_deg: float, tilt_y_deg: float, tilt_z_deg: float) -> np.ndarray:
    """Compute composite 3D Euler rotation matrix R = Rz(rz) @ Ry(ry) @ Rx(rx)."""
    rx = np.radians(tilt_x_deg)
    ry = np.radians(tilt_y_deg)
    rz = np.radians(tilt_z_deg)

    Rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(rx), -np.sin(rx)],
        [0.0, np.sin(rx), np.cos(rx)]
    ], dtype=float)

    Ry = np.array([
        [np.cos(ry), 0.0, np.sin(ry)],
        [0.0, 1.0, 0.0],
        [-np.sin(ry), 0.0, np.cos(ry)]
    ], dtype=float)

    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0.0],
        [np.sin(rz), np.cos(rz), 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=float)

    return Rz @ Ry @ Rx


get_rotation_matrix = build_rotation_matrix

from optics_app.engine.grating import grating_diffract_3d

def _cross3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.array([
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    ], dtype=float)


def _norm3(v: np.ndarray) -> float:
    return float(np.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]))


def snell_refract_3d(ray_dir: np.ndarray, normal: np.ndarray, n1: float, n2: float) -> Tuple[np.ndarray, bool]:
    """
    3D Vector Snell's Law Refraction:
    v' = (n1/n2)*v + ((n1/n2)*cos1 - sqrt(1 - (n1/n2)^2 * (1 - cos1^2))) * N
    Returns (v_out, tir_boolean).
    """
    v = np.asarray(ray_dir, dtype=float)
    v_norm = _norm3(v)
    v = v / v_norm if v_norm > 0 else np.array([0.0, 0.0, 1.0])

    N = np.asarray(normal, dtype=float)
    n_norm = _norm3(N)
    N = N / n_norm if n_norm > 0 else np.array([0.0, 0.0, 1.0])

    # Ensure normal opposes incoming ray
    dot_vn = float(v[0] * N[0] + v[1] * N[1] + v[2] * N[2])
    if dot_vn > 0:
        N = -N
        dot_vn = -dot_vn

    cos1 = -dot_vn
    mu = n1 / n2
    sin2_t2 = mu**2 * (1.0 - cos1**2)

    if sin2_t2 > 1.0:
        # Total Internal Reflection (TIR)
        v_ref = v + (2.0 * cos1) * N
        ref_norm = _norm3(v_ref)
        if ref_norm > 0:
            v_ref = v_ref / ref_norm
        return v_ref, True

    cos2 = float(np.sqrt(max(0.0, 1.0 - sin2_t2)))
    v_prime = mu * v + (mu * cos1 - cos2) * N
    vp_norm = _norm3(v_prime)
    if vp_norm > 0:
        v_prime = v_prime / vp_norm
    return v_prime, False


def ray_surface_intersect(surface: Any, ray: Ray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Generic ray-surface intersection solver."""
    if hasattr(surface, 'intersect'):
        return surface.intersect(ray)
    stype = getattr(surface, 'surface_type', SurfaceType.STANDARD)
    has_asphere = (getattr(surface, 'conic', 0.0) != 0.0) or (len(getattr(surface, 'asphere_coeffs', [])) > 0)
    if stype == SurfaceType.ASPHERIC or has_asphere:
        asph = AsphericSurface(
            radius=getattr(surface, 'radius', 0.0),
            conic=getattr(surface, 'conic', 0.0),
            asphere_coeffs=getattr(surface, 'asphere_coeffs', [])
        )
        return asph.intersect(ray)
    sph = SphericalSurface(radius=getattr(surface, 'radius', 0.0), semi_diameter=getattr(surface, 'semi_diameter', 12.7))
    return sph.intersect(ray)


class OpticalSystem:
    def __init__(self, surface_stack: SurfaceStack, verbose: bool = False):
        self.stack = surface_stack
        self.verbose = verbose
        if hasattr(self.stack, "_update_accumulated_positions"):
            self.stack._update_accumulated_positions()

    @classmethod
    def from_json(cls, filepath: str) -> "OpticalSystem":
        """Load and initialize an OpticalSystem directly from a JSON file."""
        loaded = SurfaceStack.load_from_json(filepath)
        stack = loaded["stack"] if isinstance(loaded, dict) else loaded
        return cls(stack)

    @property
    def surfaces(self) -> List[Surface]:
        if hasattr(self.stack, 'get_active_sequence'):
            return self.stack.get_active_sequence()
        return self.stack.surfaces

    def get_surface_stack(self) -> SurfaceStack:
        return self.stack

    def rotate_element(self, element_idx: Union[int, str], tilt_x: float = 0.0, tilt_y: float = 0.0, tilt_z: float = 0.0, pivot: Any = "front_vertex") -> None:
        if hasattr(self.stack, "rotate_element"):
            self.stack.rotate_element(element_idx, tilt_x=tilt_x, tilt_y=tilt_y, tilt_z=tilt_z, pivot=pivot)

    def flip_element(self, element_idx: Union[int, str]) -> List[int]:
        if hasattr(self.stack, "flip_element"):
            return self.stack.flip_element(element_idx)
        return []

    def displace_element(self, element_idx: Union[int, str], decenter_x: float = 0.0, decenter_y: float = 0.0, decenter_z: float = 0.0, **kwargs) -> None:
        if hasattr(self.stack, "displace_element"):
            self.stack.displace_element(element_idx, decenter_x=decenter_x, decenter_y=decenter_y, decenter_z=decenter_z, **kwargs)

    def tilt_element(self, element_idx: Union[int, str], angle_deg: float = 0.0, axis: str = "x", **kwargs) -> None:
        if hasattr(self.stack, "tilt_element"):
            self.stack.tilt_element(element_idx, angle_deg=angle_deg, axis=axis, **kwargs)

    def offset_element(self, element_idx: Union[int, str], reference_surface_idx: Union[int, str], offset_bias_mm: float, axis: str = "z") -> None:
        if hasattr(self.stack, "offset_element"):
            self.stack.offset_element(element_idx, reference_surface_idx=reference_surface_idx, offset_bias_mm=offset_bias_mm, axis=axis)

    def insert_element(self, after_surface_idx: int, element: Any) -> List[int]:
        if hasattr(self.stack, "insert_element"):
            return self.stack.insert_element(after_surface_idx, element)
        return []

    def delete_element(self, element_idx: Union[int, str]) -> List[int]:
        if hasattr(self.stack, "delete_element"):
            return self.stack.delete_element(element_idx)
        return []

    def replace_element(self, element_idx: Union[int, str], new_element: Any) -> List[int]:
        if hasattr(self.stack, "replace_element"):
            return self.stack.replace_element(element_idx, new_element)
        return []

    def replace_element_with_catalog(self, element_idx: Union[int, str], part_number: str) -> List[int]:
        if hasattr(self.stack, "replace_element_with_catalog"):
            return self.stack.replace_element_with_catalog(element_idx, part_number)
        return []

    def _update_accumulated_positions(self) -> None:
        if hasattr(self.stack, '_update_accumulated_positions'):
            self.stack._update_accumulated_positions()


    def trace_ray(
        self,
        ray: Ray,
        _interaction_rotations: Optional[List[np.ndarray]] = None,
        _interaction_origins: Optional[List[np.ndarray]] = None,
        _interaction_active_states: Optional[List[str]] = None,
        verbose: Optional[bool] = None,
        record_history: bool = True,
        max_surface_index: Optional[int] = None,
        track_polarization: bool = True,
    ) -> Ray:
        """Sequential ray trace through all surfaces in the stack with optional polarization tracking."""
        traced_ray = ray.copy()
        # Preserve the launch point for layout renderers.  ``pos`` is advanced
        # to the final hit during tracing and is not the start of the path.
        traced_ray.launch_pos = traced_ray.pos.copy()
        current_glass = "AIR"
        if track_polarization:
            current_n_complex = get_complex_refractive_index("AIR", ray.wavelength, 0.0)
        else:
            current_n_real = 1.0
        is_verbose = self.verbose if verbose is None else verbose

        for i, surf in enumerate(self.surfaces):
            if max_surface_index is not None and i > max_surface_index:
                break
            if traced_ray.vignetted:
                break

            # 1. Transform ray from Global frame to Surface Local frame
            # Resolve effective pose incorporating nominal geometry,
            # continuous galvo deflections with pivot offset kinematics,
            # and discrete/arbitrary active mechanisms.
            if _interaction_rotations is not None and _interaction_origins is not None:
                R_mat = _interaction_rotations[i]
                origin = _interaction_origins[i]
            elif _interaction_rotations is not None:
                R_mat = _interaction_rotations[i]
                origin = get_effective_pose(surf)[0]
            else:
                origin, R_mat = get_effective_pose(surf)

            pos_local = R_mat.T @ (traced_ray.pos - origin)
            dir_local = R_mat.T @ traced_ray.dir
            l_norm = _norm3(dir_local)
            if l_norm > 0:
                dir_local = dir_local / l_norm

            if track_polarization:
                local_s_basis = R_mat.T @ traced_ray.s_basis
                s_norm = _norm3(local_s_basis)
                if s_norm > 0:
                    local_s_basis = local_s_basis / s_norm

            local_ray = _LocalRay(
                pos=pos_local,
                dir=dir_local,
                wavelength=traced_ray.wavelength,
            )

            if bool(getattr(surf, "is_air_space", False)):
                # An air-space row only locates downstream components through
                # its thickness. Keep this ray at the actual preceding hit;
                # intersecting a dummy plane here can backtrack or reflect it
                # a second time at a folded mirror's vertex.
                if record_history:
                    traced_ray.history.append({
                        "surface_index": i,
                        "x": float(pos_local[0]), "y": float(pos_local[1]),
                        "z": float(traced_ray.pos[2]),
                        "x_global": float(traced_ray.pos[0]),
                        "y_global": float(traced_ray.pos[1]),
                        "z_global": float(traced_ray.pos[2]),
                        "x_local": float(pos_local[0]),
                        "y_local": float(pos_local[1]),
                        "z_local": float(pos_local[2]),
                        "dir_x": float(traced_ray.dir[0]),
                        "dir_y": float(traced_ray.dir[1]),
                        "dir_z": float(traced_ray.dir[2]),
                        "dir": traced_ray.dir.copy(),
                        "vignetted": False, "tir": traced_ray.tir,
                        "jones_vector": traced_ray.jones_vector.copy(),
                        "s_basis": traced_ray.s_basis.copy(),
                        "p_basis": traced_ray.p_basis.copy(),
                        "intensity": float(traced_ray.intensity),
                        "phase": float(traced_ray.phase),
                        "fresnel_rs": 0j, "fresnel_rp": 0j,
                        "fresnel_ts": 1+0j, "fresnel_tp": 1+0j,
                        "mechanical_state": "",
                        "interaction": "air_space", "reference_only": True,
                    })
                continue

            # 2. Ray-surface intersection in local frame
            is_cb = getattr(surf, 'is_coord_break', False) or (getattr(surf, 'surface_type', None) == getattr(SurfaceType, 'COORDINATE_BREAK', None))
            if is_cb:
                res = (pos_local.copy(), np.array([0.0, 0.0, 1.0]))
            else:
                res = ray_surface_intersect(surf, local_ray)
                if res is not None:
                    travel = float(np.dot(np.asarray(res[0]) - pos_local, dir_local))
                    if not np.isfinite(travel) or travel < -1e-9:
                        res = None
            if res is None:
                traced_ray.vignetted = True
                if i > 0 and len(traced_ray.history) > 0:
                    prev_s = self.surfaces[i - 1]
                    proj_dist = float(getattr(prev_s, "thickness", 50.0))
                    if proj_dist <= 0:
                        proj_dist = 50.0
                    term_pos = traced_ray.pos + traced_ray.dir * proj_dist
                    traced_ray.history.append({
                        "surface_index": i,
                        "x": float(term_pos[0]),
                        "y": float(term_pos[1]),
                        "z": float(term_pos[2]),
                        "x_global": float(term_pos[0]),
                        "y_global": float(term_pos[1]),
                        "z_global": float(term_pos[2]),
                        "x_local": 0.0,
                        "y_local": 0.0,
                        "z_local": 0.0,
                        "dir_x": float(traced_ray.dir[0]),
                        "dir_y": float(traced_ray.dir[1]),
                        "dir_z": float(traced_ray.dir[2]),
                        "dir": traced_ray.dir.copy(),
                        "vignetted": True,
                        "tir": traced_ray.tir,
                        "jones_vector": getattr(traced_ray, "jones_vector", np.array([0.0, 0.0], dtype=complex)).copy(),
                        "s_basis": getattr(traced_ray, "s_basis", np.array([1.0, 0.0, 0.0])).copy(),
                        "p_basis": getattr(traced_ray, "p_basis", np.array([0.0, 1.0, 0.0])).copy(),
                        "intensity": 0.0,
                        "phase": float(traced_ray.phase),
                        "fresnel_rs": complex(0.0),
                        "fresnel_rp": complex(0.0),
                        "fresnel_ts": complex(0.0),
                        "fresnel_tp": complex(0.0),
                        "power": 0.0,
                        "failure_reason": "no_forward_intersection",
                    })
                break

            hit_local, normal_local = res

            # 3. Complex Aperture vignetting check
            is_cb = getattr(surf, 'is_coord_break', False) or (getattr(surf, 'surface_type', None) == getattr(SurfaceType, 'COORDINATE_BREAK', None))
            if not is_cb and (surf.surface_type != SurfaceType.OBJECT or (surf.semi_diameter > 0 and i > 0)):
                if hasattr(surf, 'check_aperture'):
                    if not surf.check_aperture(float(hit_local[0]), float(hit_local[1])):
                        if is_verbose: print('VIGNETTED AT', i, surf.name)
                        traced_ray.vignetted = True
                else:
                    r_dist = float(np.sqrt(hit_local[0]**2 + hit_local[1]**2))
                    if surf.semi_diameter > 0 and r_dist > surf.semi_diameter:
                        if is_verbose: print('VIGNETTED AT', i, surf.name)
                        traced_ray.vignetted = True

            # 4. Refractive index & polarization basis calculation
            ext_k = getattr(surf, 'extinction_coeff', 0.0)
            is_mirror = surf.is_mirror()

            surf_is_modulator = is_plm(surf) or (is_dmd(surf) and (getattr(surf, "plm_config", None) is not None or getattr(surf, "plm_enabled", False) or getattr(surf, "is_dmd", False)))
            plm_phase_rad = 0.0
            plm_state_4bit = -1
            if _interaction_active_states is not None:
                current_active_state = _interaction_active_states[i]
            else:
                has_mech = bool(getattr(surf, "mechanical_states", None) or getattr(surf, "mechanical_state", None) or is_dmd(surf) or surf_is_modulator)
                current_active_state = active_state_name(surf) if has_mech else ""

            if track_polarization:
                next_n_complex = get_complex_refractive_index(surf.glass, traced_ray.wavelength, ext_k) if surf.glass else get_complex_refractive_index("AIR", traced_ray.wavelength, 0.0)

                s_surf, p_surf_inc, normal_aligned = compute_sp_basis(dir_local, normal_local, local_s_basis)

                # Rotate incoming Jones vector to local surface s/p basis
                J_inc = rotate_jones_vector(traced_ray.jones_vector, local_s_basis, s_surf, dir_local)

                cos_inc = float(np.clip(-float(dir_local[0]*normal_aligned[0] + dir_local[1]*normal_aligned[1] + dir_local[2]*normal_aligned[2]), -1.0, 1.0))
                if is_mirror and ext_k == 0.0:
                    fresnel = FresnelResult(
                        r_s=complex(-1.0, 0.0),
                        r_p=complex(1.0, 0.0),
                        t_s=complex(0.0, 0.0),
                        t_p=complex(0.0, 0.0),
                        tir=False,
                        theta_inc=float(np.arccos(cos_inc)),
                        theta_trans=complex(0.0, 0.0)
                    )
                else:
                    fresnel = compute_fresnel_coefficients(current_n_complex, next_n_complex, cos_inc)

                # 5. Physics interaction & Jones state transformation
                is_bs_surf = bool(getattr(surf, "is_beam_splitter", False)) or (getattr(surf, "beamsplitter_config", None) is not None)
                if is_bs_surf:
                    bs_cfg = getattr(surf, "beamsplitter_config", None)
                    branch = str(getattr(surf, "active_branch", "transmitted")).lower()
                    if branch == "reflected":
                        dot_dn = float(dir_local[0]*normal_aligned[0] + dir_local[1]*normal_aligned[1] + dir_local[2]*normal_aligned[2])
                        dir_out_local = dir_local - (2.0 * dot_dn) * normal_aligned
                        r_norm = _norm3(dir_out_local)
                        if r_norm > 0: dir_out_local /= r_norm
                        p_surf_out = _cross3(s_surf, dir_out_local)
                        p_norm = _norm3(p_surf_out)
                        p_surf_out = p_surf_out / p_norm if p_norm > 0 else p_surf_inc
                        if bs_cfg and hasattr(bs_cfg, "evaluate_jones"):
                            _, _, J_out, I_refl = bs_cfg.evaluate_jones(J_inc, traced_ray.wavelength, traced_ray.intensity)
                            traced_ray.intensity = float(I_refl)
                        else:
                            R_val = float(getattr(bs_cfg, "split_ratio_R", 0.5)) if bs_cfg else 0.5
                            J_out = math.sqrt(max(0.0, R_val)) * np.array([J_inc[0], -J_inc[1]], dtype=complex)
                            traced_ray.intensity *= R_val
                    else:
                        if surf.glass and surf.glass.upper() not in ("AIR", "MIRROR"):
                            dir_out_local, tir = snell_refract_3d(dir_local, normal_aligned, current_n_complex.real, next_n_complex.real)
                        else:
                            dir_out_local = dir_local
                            tir = False
                        p_surf_out = _cross3(s_surf, dir_out_local)
                        p_norm = _norm3(p_surf_out)
                        p_surf_out = p_surf_out / p_norm if p_norm > 0 else p_surf_inc
                        if bs_cfg and hasattr(bs_cfg, "evaluate_jones"):
                            J_out, I_trans, _, _ = bs_cfg.evaluate_jones(J_inc, traced_ray.wavelength, traced_ray.intensity)
                            traced_ray.intensity = float(I_trans)
                        else:
                            T_val = float(getattr(bs_cfg, "split_ratio_T", 0.5)) if bs_cfg else 0.5
                            J_out = math.sqrt(max(0.0, T_val)) * J_inc
                            traced_ray.intensity *= T_val
                        if surf.glass and surf.glass.upper() not in ("AIR", "MIRROR"):
                            current_glass = surf.glass
                            current_n_complex = next_n_complex

                elif surf_is_modulator:
                    plm_cfg = ensure_plm_config(surf)
                    dir_out_global_plm, plm_phase_rad, plm_eff, plm_state_4bit = deflect_ray_on_plm_3d(
                        ray_dir_in=traced_ray.dir,
                        surf_normal=R_mat @ normal_local,
                        surf_global_R=R_mat,
                        local_x_mm=float(hit_local[0]),
                        local_y_mm=float(hit_local[1]),
                        plm_cfg=plm_cfg,
                        mechanical_state=current_active_state or "ON",
                        wavelength_um=traced_ray.wavelength,
                    )
                    dir_out_local = R_mat.T @ dir_out_global_plm
                    p_surf_out = _cross3(s_surf, dir_out_local)
                    p_norm = _norm3(p_surf_out)
                    p_surf_out = p_surf_out / p_norm if p_norm > 0 else p_surf_inc
                    phase_fac = np.exp(1j * plm_phase_rad)
                    eff_amp = math.sqrt(max(0.0, float(plm_eff))) if plm_eff is not None else 1.0
                    J_out = np.array([fresnel.r_s * J_inc[0], fresnel.r_p * J_inc[1]], dtype=complex) * (phase_fac * eff_amp)
                    if (plm_cfg.is_dmd and plm_state_4bit == 0 and (current_active_state or "ON") == "ON" and plm_cfg.pattern.enabled) or (plm_eff is not None and float(plm_eff) <= 1e-9):
                        traced_ray.vignetted = True
                        traced_ray.intensity = 0.0
                        J_out = np.zeros(2, dtype=complex)

                elif surf.surface_type == SurfaceType.GRATING and surf.grating_lines_per_mm > 0:
                    # Compute grating diffraction in GLOBAL frame to match _update_accumulated_positions.
                    global_grating_normal = surf.global_R[:, 2].copy()
                    if (traced_ray.dir[0]*global_grating_normal[0] + traced_ray.dir[1]*global_grating_normal[1] + traced_ray.dir[2]*global_grating_normal[2]) > 0:
                        global_grating_normal = -global_grating_normal
                    local_disp = np.array([0.0, 1.0, 0.0]) if abs(getattr(surf, 'tilt_x', 0.0)) > abs(getattr(surf, 'tilt_y', 0.0)) else np.array([1.0, 0.0, 0.0])
                    global_line_vector = surf.global_R @ local_disp

                    dir_out_global_grating, evan = grating_diffract_3d(
                        ray_dir=traced_ray.dir,
                        grating_normal=global_grating_normal,
                        line_vector=global_line_vector,
                        line_density_lines_per_mm=surf.grating_lines_per_mm,
                        order=surf.grating_order,
                        wavelength_um=traced_ray.wavelength,
                        is_reflection=is_mirror
                    )

                    if evan or dir_out_global_grating is None:
                        if is_verbose: print('VIGNETTED AT', i, surf.name)
                        traced_ray.vignetted = True
                        dir_out_local = dir_local
                    else:
                        dir_out_local = R_mat.T @ dir_out_global_grating

                    p_surf_out = _cross3(s_surf, dir_out_local)
                    p_norm = _norm3(p_surf_out)
                    p_surf_out = p_surf_out / p_norm if p_norm > 0 else p_surf_inc
                    J_out = np.array([fresnel.r_s * J_inc[0], fresnel.r_p * J_inc[1]], dtype=complex) if is_mirror else np.array([fresnel.t_s * J_inc[0], fresnel.t_p * J_inc[1]], dtype=complex)

                elif is_mirror:
                    dot_dn = float(dir_local[0]*normal_aligned[0] + dir_local[1]*normal_aligned[1] + dir_local[2]*normal_aligned[2])
                    dir_out_local = dir_local - (2.0 * dot_dn) * normal_aligned
                    r_norm = _norm3(dir_out_local)
                    if r_norm > 0:
                        dir_out_local = dir_out_local / r_norm
                    p_surf_out = _cross3(s_surf, dir_out_local)
                    p_norm = _norm3(p_surf_out)
                    p_surf_out = p_surf_out / p_norm if p_norm > 0 else p_surf_inc
                    J_out = np.array([fresnel.r_s * J_inc[0], fresnel.r_p * J_inc[1]], dtype=complex)

                elif surf.surface_type not in (SurfaceType.DETECTOR, SurfaceType.STOP, SurfaceType.OBJECT, getattr(SurfaceType, 'COORDINATE_BREAK', None)) and not is_cb:
                    dir_out_local, tir = snell_refract_3d(dir_local, normal_aligned, current_n_complex.real, next_n_complex.real)
                    if tir or fresnel.tir:
                        traced_ray.tir = True
                        p_surf_out = _cross3(s_surf, dir_out_local)
                        p_norm = _norm3(p_surf_out)
                        p_surf_out = p_surf_out / p_norm if p_norm > 0 else p_surf_inc
                        J_out = np.array([fresnel.r_s * J_inc[0], fresnel.r_p * J_inc[1]], dtype=complex)
                    else:
                        p_surf_out = _cross3(s_surf, dir_out_local)
                        p_norm = _norm3(p_surf_out)
                        p_surf_out = p_surf_out / p_norm if p_norm > 0 else p_surf_inc
                        J_out = np.array([fresnel.t_s * J_inc[0], fresnel.t_p * J_inc[1]], dtype=complex)
                    current_glass = surf.glass
                    current_n_complex = next_n_complex
                else:
                    dir_out_local = dir_local
                    p_surf_out = p_surf_inc
                    J_out = J_inc

                # Apply custom surface Jones matrix if defined (e.g. polarizer or waveplate)
                if hasattr(surf, 'jones_matrix') and surf.jones_matrix is not None:
                    J_out = np.array(surf.jones_matrix, dtype=complex) @ J_out

                traced_ray.jones_vector = J_out

                if current_active_state:
                    state_T = state_transmission(surf, traced_ray.wavelength)
                    if state_T < 1.0:
                        traced_ray.jones_vector *= np.sqrt(state_T)
            else:
                # Fast geometric ray tracing without polarization calculations
                next_n_real = get_refractive_index(surf.glass, traced_ray.wavelength) if surf.glass else 1.0
                normal_aligned = normal_local
                if (dir_local[0]*normal_aligned[0] + dir_local[1]*normal_aligned[1] + dir_local[2]*normal_aligned[2]) > 0:
                    normal_aligned = -normal_aligned
                fresnel = None

                if surf_is_modulator:
                    plm_cfg = ensure_plm_config(surf)
                    dir_out_global_plm, plm_phase_rad, plm_eff, plm_state_4bit = deflect_ray_on_plm_3d(
                        ray_dir_in=traced_ray.dir,
                        surf_normal=R_mat @ normal_local,
                        surf_global_R=R_mat,
                        local_x_mm=float(hit_local[0]),
                        local_y_mm=float(hit_local[1]),
                        plm_cfg=plm_cfg,
                        mechanical_state=current_active_state or "ON",
                        wavelength_um=traced_ray.wavelength,
                    )
                    dir_out_local = R_mat.T @ dir_out_global_plm
                    if (plm_cfg.is_dmd and plm_state_4bit == 0 and (current_active_state or "ON") == "ON" and plm_cfg.pattern.enabled) or (plm_eff is not None and float(plm_eff) <= 1e-9):
                        traced_ray.vignetted = True
                        traced_ray.intensity = 0.0
                    elif plm_eff is not None:
                        traced_ray.intensity *= float(plm_eff)

                elif surf.surface_type == SurfaceType.GRATING and surf.grating_lines_per_mm > 0:
                    global_grating_normal = surf.global_R[:, 2].copy()
                    if (traced_ray.dir[0]*global_grating_normal[0] + traced_ray.dir[1]*global_grating_normal[1] + traced_ray.dir[2]*global_grating_normal[2]) > 0:
                        global_grating_normal = -global_grating_normal
                    local_disp = np.array([0.0, 1.0, 0.0]) if abs(getattr(surf, 'tilt_x', 0.0)) > abs(getattr(surf, 'tilt_y', 0.0)) else np.array([1.0, 0.0, 0.0])
                    global_line_vector = surf.global_R @ local_disp

                    dir_out_global_grating, evan = grating_diffract_3d(
                        ray_dir=traced_ray.dir,
                        grating_normal=global_grating_normal,
                        line_vector=global_line_vector,
                        line_density_lines_per_mm=surf.grating_lines_per_mm,
                        order=surf.grating_order,
                        wavelength_um=traced_ray.wavelength,
                        is_reflection=is_mirror
                    )
                    if evan or dir_out_global_grating is None:
                        if is_verbose: print('VIGNETTED AT', i, surf.name)
                        traced_ray.vignetted = True
                        dir_out_local = dir_local
                    else:
                        dir_out_local = R_mat.T @ dir_out_global_grating

                elif is_mirror:
                    dot_dn = float(dir_local[0]*normal_aligned[0] + dir_local[1]*normal_aligned[1] + dir_local[2]*normal_aligned[2])
                    dir_out_local = dir_local - (2.0 * dot_dn) * normal_aligned
                    r_norm = _norm3(dir_out_local)
                    if r_norm > 0:
                        dir_out_local = dir_out_local / r_norm

                elif surf.surface_type not in (SurfaceType.DETECTOR, SurfaceType.STOP, SurfaceType.OBJECT, getattr(SurfaceType, 'COORDINATE_BREAK', None)) and not is_cb:
                    dir_out_local, tir = snell_refract_3d(dir_local, normal_aligned, current_n_real, next_n_real)
                    if tir:
                        traced_ray.tir = True
                    current_glass = surf.glass
                    current_n_real = next_n_real
                else:
                    dir_out_local = dir_local

            # 6. Transform hit point, normal, basis vectors, and direction back to Global frame
            hit_global = origin + R_mat @ hit_local
            normal_global = R_mat @ normal_local
            dir_out_global = R_mat @ dir_out_local

            g_norm = _norm3(dir_out_global)
            if g_norm > 0:
                dir_out_global = dir_out_global / g_norm

            if track_polarization:
                traced_ray.s_basis = R_mat @ s_surf
                traced_ray.p_basis = R_mat @ p_surf_out

            dist = _norm3(hit_global - traced_ray.pos)
            traced_ray.path_length += dist

            # Phase accumulation delta_phi = (2 * pi * n_real * dist) / lambda + PLM 4-bit piston phase
            wl_mm = traced_ray.wavelength * 1e-3 if traced_ray.wavelength > 0.01 else traced_ray.wavelength
            med_n = current_n_complex.real if track_polarization else current_n_real
            delta_phi = (2.0 * np.pi * med_n * dist) / wl_mm + plm_phase_rad
            traced_ray.phase += float(delta_phi)

            # Spatial interference check against all other surfaces
            if getattr(self.stack, 'enable_interference_check', False) and i > 0:
                p1 = traced_ray.pos
                p2 = hit_global
                for k, check_surf in enumerate(self.surfaces):
                    if k != i and k != i - 1:
                        if check_segment_surface_interference(p1, p2, check_surf):
                            traced_ray.interference = True
                            if is_verbose: print('VIGNETTED AT', i, surf.name)
                            traced_ray.vignetted = True
                            break

            traced_ray.pos = hit_global
            traced_ray.dir = dir_out_global
            traced_ray.update_intensity()

            if record_history:
                hist_entry = {
                    "surface_index": i,
                    "x": float(hit_local[0]),
                    "y": float(hit_local[1]),
                    "z": float(hit_global[2]),
                    "x_global": float(hit_global[0]),
                    "y_global": float(hit_global[1]),
                    "z_global": float(hit_global[2]),
                    "x_local": float(hit_local[0]),
                    "y_local": float(hit_local[1]),
                    "z_local": float(hit_local[2]),
                    "dir_x": float(dir_out_global[0]),
                    "dir_y": float(dir_out_global[1]),
                    "dir_z": float(dir_out_global[2]),
                    "dir": dir_out_global.copy(),
                    "vignetted": traced_ray.vignetted,
                    "tir": traced_ray.tir,
                    "jones_vector": traced_ray.jones_vector.copy() if track_polarization else None,
                    "s_basis": traced_ray.s_basis.copy() if track_polarization else None,
                    "p_basis": traced_ray.p_basis.copy() if track_polarization else None,
                    "intensity": float(traced_ray.intensity),
                    "phase": float(traced_ray.phase),
                    "fresnel_rs": fresnel.r_s if (track_polarization and fresnel is not None) else 0.0,
                    "fresnel_rp": fresnel.r_p if (track_polarization and fresnel is not None) else 0.0,
                    "fresnel_ts": fresnel.t_s if (track_polarization and fresnel is not None) else 0.0,
                    "fresnel_tp": fresnel.t_p if (track_polarization and fresnel is not None) else 0.0,
                    "theta_inc": float(getattr(fresnel, "theta_inc", 0.0)) if (track_polarization and fresnel is not None) else float(np.arccos(np.clip(abs(float(dir_local[0]*normal_aligned[0] + dir_local[1]*normal_aligned[1] + dir_local[2]*normal_aligned[2])), 0.0, 1.0))),
                    "mechanical_state": current_active_state,
                }
                if surf_is_modulator:
                    hist_entry["plm_state_4bit"] = int(plm_state_4bit)
                    hist_entry["plm_phase_rad"] = float(plm_phase_rad)
                traced_ray.history.append(hist_entry)


            # Strict ray termination at detector surface (Requirement R2 / Milestone M1)
            if surf.surface_type == SurfaceType.DETECTOR:
                break

        return traced_ray

    def trace_rays(
        self,
        rays: List[Ray],
        verbose: Optional[bool] = None,
        record_history: bool = True,
        track_polarization: bool = True,
    ) -> List[Ray]:
        if not rays:
            return []
        # Mechanical state and surface orientation/origin are invariant across one
        # synchronous ray batch. Compute their transforms once, not per ray.
        poses = [get_effective_pose(surf) for surf in self.surfaces]
        origins = [p[0] for p in poses]
        rotations = [p[1] for p in poses]
        active_states = [
            active_state_name(surf)
            if (getattr(surf, "mechanical_states", None) or getattr(surf, "mechanical_state", None) or is_dmd(surf) or is_plm(surf))
            else ""
            for surf in self.surfaces
        ]
        is_verbose = self.verbose if verbose is None else verbose
        return [
            self.trace_ray(
                r,
                _interaction_rotations=rotations,
                _interaction_origins=origins,
                _interaction_active_states=active_states,
                verbose=is_verbose,
                record_history=record_history,
                track_polarization=track_polarization,
            )
            for r in rays
        ]

    def trace_bundle(
        self,
        bundle: RayBundle,
        record_history: bool = True,
        verbose: Optional[bool] = None,
    ) -> RayBundle:
        """
        High-throughput vectorized ray tracing leveraging (N, 3) NumPy arrays.
        Performs batch coordinate transforms, surface intersections, and
        refraction/reflection for planar, spherical, and conical surfaces.
        """
        if len(bundle) == 0:
            return bundle

        traced = bundle.copy()
        N = len(traced)
        is_verbose = self.verbose if verbose is None else verbose

        # Preserve launch positions
        traced.launch_pos = traced.pos.copy()

        current_glass = "AIR"
        current_n_complex = get_complex_refractive_index("AIR", float(traced.wavelength[0]) if len(traced.wavelength) else 0.5876, 0.0)

        # Global ray positions and directions
        P_glob = traced.pos.copy()
        D_glob = traced.dir.copy()
        active_mask = ~traced.vignetted

        step_history = [] if record_history else None

        for i, surf in enumerate(self.surfaces):
            if not np.any(active_mask):
                break

            origin, R_mat = get_effective_pose(surf)
            origin = np.asarray(origin, dtype=float)
            R_mat = np.asarray(R_mat, dtype=float)

            # 1. Batch transform to surface local frame:
            # P_local = (P_glob - origin) @ R_mat
            # D_local = D_glob @ R_mat
            P_local = (P_glob - origin) @ R_mat
            D_local = D_glob @ R_mat
            d_norm = np.linalg.norm(D_local, axis=1, keepdims=True)
            d_norm[d_norm == 0.0] = 1.0
            D_local = D_local / d_norm

            # 2. Batch intersection solver
            stype = getattr(surf, 'surface_type', SurfaceType.STANDARD)
            is_cb = getattr(surf, 'is_coord_break', False) or (stype == getattr(SurfaceType, 'COORDINATE_BREAK', None))
            radius = float(getattr(surf, 'radius', 0.0))
            conic = float(getattr(surf, 'conic', 0.0))
            asph_coeffs = getattr(surf, 'asphere_coeffs', []) or []
            has_poly_asph = len(asph_coeffs) > 0
            semi_d = float(getattr(surf, 'semi_diameter', 0.0))
            is_mirror = surf.is_mirror()

            hit_local = np.zeros_like(P_local)
            normal_local = np.zeros_like(D_local)
            normal_local[:, 2] = 1.0
            intersect_success = active_mask.copy()

            if is_cb:
                hit_local = P_local.copy()
                normal_local = np.tile([0.0, 0.0, 1.0], (N, 1))

            elif has_poly_asph or (hasattr(surf, 'intersect') and type(surf).__name__ not in ('Surface', 'SphericalSurface', 'AsphericSurface')):
                # Specialized custom surface or high-order polynomial asphere: evaluate active rays
                for k in range(N):
                    if not active_mask[k]:
                        continue
                    r_temp = Ray(pos=P_local[k], dir=D_local[k], wavelength=float(traced.wavelength[k]))
                    res = ray_surface_intersect(surf, r_temp)
                    if res is None:
                        intersect_success[k] = False
                    else:
                        hit_local[k], normal_local[k] = res

            elif radius == 0.0 or np.isinf(radius):
                # Flat plane at z = 0
                dz = D_local[:, 2]
                valid_dz = np.abs(dz) > 1e-12
                t = np.zeros(N, dtype=float)
                t[valid_dz] = -P_local[valid_dz, 2] / dz[valid_dz]
                valid_t = valid_dz & (t >= -1e-9) & active_mask
                intersect_success = valid_t
                t[~valid_t] = 0.0
                t = np.maximum(0.0, t)
                hit_local = P_local + t[:, None] * D_local
                normal_local = np.tile([0.0, 0.0, 1.0], (N, 1))

            elif conic == 0.0:
                # Spherical surface (conic = 0)
                R = radius
                c_center = np.array([0.0, 0.0, R], dtype=float)
                oc = P_local - c_center
                b = np.sum(D_local * oc, axis=1)
                c_val = np.sum(oc * oc, axis=1) - R**2
                disc = b**2 - c_val
                pos_disc = (disc >= 0.0) & active_mask
                sq_disc = np.zeros(N, dtype=float)
                sq_disc[pos_disc] = np.sqrt(disc[pos_disc])
                t1 = -b - sq_disc
                t2 = -b + sq_disc

                sd_bound = (semi_d * 2.0)**2 if semi_d > 0 else (abs(R) * 2.0)**2
                h1 = P_local + t1[:, None] * D_local
                h2 = P_local + t2[:, None] * D_local
                cand1 = (t1 >= -1e-12) & (h1[:, 0]**2 + h1[:, 1]**2 <= sd_bound) & (np.abs(h1[:, 2]) <= abs(R))
                cand2 = (t2 >= -1e-12) & (h2[:, 0]**2 + h2[:, 1]**2 <= sd_bound) & (np.abs(h2[:, 2]) <= abs(R))
                t_chosen = np.where(cand1, t1, np.where(cand2, t2, np.where(t1 >= -1e-12, t1, np.where(t2 >= -1e-12, t2, -1.0))))

                hit_mask = (t_chosen >= -1e-12) & pos_disc
                intersect_success = hit_mask
                t_chosen[~hit_mask] = 0.0
                hit_local = P_local + t_chosen[:, None] * D_local

                r_dist = np.sqrt(hit_local[:, 0]**2 + hit_local[:, 1]**2)
                curv = 1.0 / R
                rad = np.sqrt(np.maximum(1e-12, 1.0 - curv**2 * r_dist**2))
                dz_dr = curv * r_dist / rad
                safe_r = np.where(r_dist > 0, r_dist, 1.0)
                dz_dx = np.where(r_dist > 0, dz_dr * (hit_local[:, 0] / safe_r), 0.0)
                dz_dy = np.where(r_dist > 0, dz_dr * (hit_local[:, 1] / safe_r), 0.0)
                raw_norm = np.column_stack([-dz_dx, -dz_dy, np.ones(N)])
                norm_vals = np.linalg.norm(raw_norm, axis=1, keepdims=True)
                norm_vals[norm_vals == 0.0] = 1.0
                normal_local = raw_norm / norm_vals

            else:
                # Conical surface (conic != 0, quadric)
                c_curv = 1.0 / radius
                k_con = conic
                A_q = (1.0 + k_con) * c_curv * D_local[:, 2]**2 + c_curv * (D_local[:, 0]**2 + D_local[:, 1]**2)
                B_q = (2.0 * (1.0 + k_con) * c_curv * P_local[:, 2] * D_local[:, 2] -
                       2.0 * D_local[:, 2] +
                       2.0 * c_curv * (P_local[:, 0] * D_local[:, 0] + P_local[:, 1] * D_local[:, 1]))
                C_q = ((1.0 + k_con) * c_curv * P_local[:, 2]**2 -
                       2.0 * P_local[:, 2] +
                       c_curv * (P_local[:, 0]**2 + P_local[:, 1]**2))
                is_quad = np.abs(A_q) > 1e-12
                is_lin = (~is_quad) & (np.abs(B_q) > 1e-12)

                disc_q = B_q**2 - 4.0 * np.where(is_quad, A_q, 1.0) * C_q
                pos_disc_q = (disc_q >= 0.0) & is_quad & active_mask
                sq_d = np.zeros(N, dtype=float)
                sq_d[pos_disc_q] = np.sqrt(disc_q[pos_disc_q])
                denom = 2.0 * np.where(is_quad, A_q, 1.0)
                t_c1 = (-B_q - sq_d) / denom
                t_c2 = (-B_q + sq_d) / denom
                t_quad = np.where(t_c1 > 1e-9, t_c1, np.where(t_c2 > 1e-9, t_c2, -1.0))

                t_lin = -C_q / np.where(is_lin, B_q, 1.0)
                valid_lin = (t_lin > 1e-9) & is_lin & active_mask
                t_lin = np.where(valid_lin, t_lin, -1.0)

                t_chosen = np.where(is_quad, t_quad, t_lin)
                hit_mask = (t_chosen > 1e-9) & (pos_disc_q | valid_lin)
                intersect_success = hit_mask
                t_chosen[~hit_mask] = 0.0
                hit_local = P_local + t_chosen[:, None] * D_local

                r_dist = np.sqrt(hit_local[:, 0]**2 + hit_local[:, 1]**2)
                arg_con = 1.0 - (1.0 + k_con) * c_curv**2 * r_dist**2
                dz_dr = np.where(arg_con > 0.0, (c_curv * r_dist) / np.sqrt(np.maximum(1e-12, arg_con)), 0.0)
                safe_r = np.where(r_dist > 0, r_dist, 1.0)
                dz_dx = np.where(r_dist > 0, dz_dr * (hit_local[:, 0] / safe_r), 0.0)
                dz_dy = np.where(r_dist > 0, dz_dr * (hit_local[:, 1] / safe_r), 0.0)
                raw_norm = np.column_stack([-dz_dx, -dz_dy, np.ones(N)])
                norm_vals = np.linalg.norm(raw_norm, axis=1, keepdims=True)
                norm_vals[norm_vals == 0.0] = 1.0
                normal_local = raw_norm / norm_vals

            # Handle missed intersections
            missed = active_mask & (~intersect_success)
            if np.any(missed):
                if is_verbose:
                    for k in np.where(missed)[0]:
                        print('VIGNETTED AT', i, surf.name)
                traced.vignetted |= missed
                active_mask &= ~missed

            # 3. Aperture vignetting check
            if not is_cb and (stype != SurfaceType.OBJECT or (semi_d > 0 and i > 0)):
                ap_type = getattr(surf, 'aperture_type', 'Circular')
                if ap_type == 'Rectangular' and hasattr(surf, 'aperture_x_half') and hasattr(surf, 'aperture_y_half'):
                    hx = float(surf.aperture_x_half)
                    hy = float(surf.aperture_y_half)
                    if hx > 0 and hy > 0:
                        out_ap = (np.abs(hit_local[:, 0]) > hx) | (np.abs(hit_local[:, 1]) > hy)
                        vignette_now = active_mask & out_ap
                        if np.any(vignette_now):
                            if is_verbose:
                                for k in np.where(vignette_now)[0]:
                                    print('VIGNETTED AT', i, surf.name)
                            traced.vignetted |= vignette_now
                            active_mask &= ~vignette_now
                elif semi_d > 0:
                    r_hit = np.sqrt(hit_local[:, 0]**2 + hit_local[:, 1]**2)
                    out_ap = r_hit > semi_d
                    vignette_now = active_mask & out_ap
                    if np.any(vignette_now):
                        if is_verbose:
                            for k in np.where(vignette_now)[0]:
                                print('VIGNETTED AT', i, surf.name)
                        traced.vignetted |= vignette_now
                        active_mask &= ~vignette_now

            # 4. Refraction / Reflection physics
            mean_wl = float(traced.wavelength[0]) if len(traced.wavelength) else 0.5876
            next_n_complex = get_complex_refractive_index(surf.glass, mean_wl, 0.0) if surf.glass else get_complex_refractive_index("AIR", mean_wl, 0.0)

            dot_dn = np.sum(D_local * normal_local, axis=1, keepdims=True)
            flip_norm = dot_dn > 0
            normal_aligned = np.where(flip_norm, -normal_local, normal_local)
            dot_dn = np.where(flip_norm, -dot_dn, dot_dn)
            cos1 = -dot_dn

            surf_is_modulator = is_plm(surf) or (is_dmd(surf) and (getattr(surf, "plm_config", None) is not None or getattr(surf, "plm_enabled", False) or getattr(surf, "is_dmd", False)))
            plm_phase_vec = np.zeros(N, dtype=float)

            if surf_is_modulator:
                plm_cfg = ensure_plm_config(surf)
                mech_st = active_state_name(surf) or "ON"
                eval_b = evaluate_plm_phase_and_gradient(
                    hit_local[:, 0],
                    hit_local[:, 1],
                    plm_cfg,
                    mechanical_state=mech_st,
                    wavelength_um=mean_wl,
                )
                plm_phase_vec = np.asarray(eval_b["phase_rad"], dtype=float)
                gx_w_mm = np.asarray(eval_b["grad_x_waves_per_mm"], dtype=float)
                gy_w_mm = np.asarray(eval_b["grad_y_waves_per_mm"], dtype=float)
                wl_mm_vec = np.maximum(float(mean_wl), 0.1) * 1e-3 if len(traced.wavelength) == 0 else np.maximum(np.asarray(traced.wavelength, dtype=float), 0.1) * 1e-3

                # In local surface frame, normal is normal_aligned (pointing into reflected space, i.e., -normal_aligned opposes D_local)
                n_refl = -normal_aligned
                d_in_tan_x = D_local[:, 0] + wl_mm_vec * gx_w_mm
                d_in_tan_y = D_local[:, 1] + wl_mm_vec * gy_w_mm
                tan_sq = d_in_tan_x**2 + d_in_tan_y**2
                rem_z = np.sqrt(np.maximum(0.0, 1.0 - tan_sq)) * np.sign(n_refl[:, 2])
                D_plm = np.column_stack([d_in_tan_x, d_in_tan_y, rem_z])
                d_spec = D_local - 2.0 * dot_dn * normal_aligned
                D_out_local = np.where((tan_sq < 1.0)[:, None], D_plm, d_spec)
                d_out_norm = np.linalg.norm(D_out_local, axis=1, keepdims=True)
                d_out_norm[d_out_norm == 0.0] = 1.0
                D_out_local = D_out_local / d_out_norm

                # Handle DMD OFF micro-mirror deflection (-48° dump) and mask attenuation
                b_st = np.asarray(eval_b.get("state_4bit", np.ones(N, dtype=int)), dtype=int)
                diff_eff = np.asarray(eval_b.get("diffraction_efficiency", np.ones(N, dtype=float)), dtype=float)
                is_dmd_off = (plm_cfg.is_dmd) & (b_st == 0) & (str(mech_st).upper() in ("ON", "HADAMARD_A", "HADAMARD_B")) & bool(plm_cfg.pattern.enabled)
                if np.any(is_dmd_off):
                    is_y_hinge = abs(float(surf.global_R[0, 2])) > abs(float(surf.global_R[1, 2]))
                    rad_off = math.radians(-24.0)
                    c, s = math.cos(rad_off), math.sin(rad_off)
                    if is_y_hinge:
                        R_off = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=float)
                    else:
                        R_off = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=float)
                    n_dump_local = R_off @ np.array([0.0, 0.0, 1.0])
                    dot_d_nd = np.sum(D_local * n_dump_local, axis=1, keepdims=True)
                    n_dump_aligned = np.where(dot_d_nd > 0, -n_dump_local, n_dump_local)
                    dot_d_nd = np.where(dot_d_nd > 0, -dot_d_nd, dot_d_nd)
                    d_dump_local = D_local - 2.0 * dot_d_nd * n_dump_aligned
                    d_dump_norm = np.linalg.norm(d_dump_local, axis=1, keepdims=True)
                    d_dump_norm[d_dump_norm == 0.0] = 1.0
                    d_dump_local = d_dump_local / d_dump_norm
                    D_out_local = np.where(is_dmd_off[:, None], d_dump_local, D_out_local)
                    traced.vignetted |= is_dmd_off
                    traced.intensity[is_dmd_off] = 0.0

                is_zero_eff = (diff_eff <= 1e-9) & active_mask
                if np.any(is_zero_eff):
                    traced.vignetted |= is_zero_eff
                    traced.intensity[is_zero_eff] = 0.0
                traced.intensity *= np.clip(diff_eff, 0.0, 1.0)

            elif stype == SurfaceType.GRATING and getattr(surf, 'grating_lines_per_mm', 0) > 0:
                global_grating_normal = surf.global_R[:, 2].copy()
                local_disp = np.array([0.0, 1.0, 0.0]) if abs(getattr(surf, 'tilt_x', 0.0)) > abs(getattr(surf, 'tilt_y', 0.0)) else np.array([1.0, 0.0, 0.0])
                global_line_vector = surf.global_R @ local_disp

                # High-speed vectorized 3D diffraction across all active rays
                l_dens = float(surf.grating_lines_per_mm)
                d_um = 1000.0 / l_dens
                order = int(surf.grating_order)

                dot_dn_g = D_glob @ global_grating_normal
                flip_g = dot_dn_g > 0
                N_rays = np.where(flip_g[:, None], -global_grating_normal, global_grating_normal)

                L = global_line_vector
                dot_LN = N_rays @ L
                L_proj = L - dot_LN[:, None] * N_rays
                l_norm = np.linalg.norm(L_proj, axis=1, keepdims=True)
                e_disp = L_proj / np.where(l_norm > 1e-12, l_norm, 1.0)
                e_groove = np.cross(N_rays, e_disp)
                eg_norm = np.linalg.norm(e_groove, axis=1, keepdims=True)
                e_groove /= np.where(eg_norm > 0, eg_norm, 1.0)

                v_in_disp = np.sum(D_glob * e_disp, axis=1)
                v_in_groove = np.sum(D_glob * e_groove, axis=1)
                v_in_N = np.sum(D_glob * N_rays, axis=1)

                wl_arr = np.asarray(traced.wavelength, dtype=float)
                v_out_disp = v_in_disp + (order * wl_arr / d_um)
                v_out_groove = v_in_groove
                sum_sq = v_out_disp**2 + v_out_groove**2

                evan_mask = (sum_sq > 1.0) & active_mask
                rem = np.sqrt(np.maximum(0.0, 1.0 - sum_sq))
                sign_vin = np.where(v_in_N != 0, np.sign(v_in_N), 1.0)
                v_out_N = (-sign_vin * rem) if is_mirror else (sign_vin * rem)

                v_out_glob = v_out_disp[:, None] * e_disp + v_out_groove[:, None] * e_groove + v_out_N[:, None] * N_rays
                vo_norm = np.linalg.norm(v_out_glob, axis=1, keepdims=True)
                v_out_glob /= np.where(vo_norm > 0, vo_norm, 1.0)

                if np.any(evan_mask):
                    if is_verbose:
                        for k in np.where(evan_mask)[0]:
                            print('VIGNETTED AT', i, surf.name)
                    traced.vignetted |= evan_mask
                    active_mask &= ~evan_mask

                D_out_local = v_out_glob @ R_mat

            elif is_mirror:
                D_out_local = D_local - 2.0 * dot_dn * normal_aligned
                d_out_norm = np.linalg.norm(D_out_local, axis=1, keepdims=True)
                d_out_norm[d_out_norm == 0.0] = 1.0
                D_out_local = D_out_local / d_out_norm
            elif stype not in (SurfaceType.DETECTOR, SurfaceType.STOP, SurfaceType.OBJECT, getattr(SurfaceType, 'COORDINATE_BREAK', None)) and not is_cb:
                n1 = float(current_n_complex.real)
                n2 = float(next_n_complex.real)
                mu = n1 / n2
                sin2_t2 = mu**2 * (1.0 - cos1**2)
                tir_mask = (sin2_t2 > 1.0).squeeze(-1) & active_mask

                D_tir = D_local + (2.0 * cos1) * normal_aligned
                d_tir_norm = np.linalg.norm(D_tir, axis=1, keepdims=True)
                d_tir_norm[d_tir_norm == 0.0] = 1.0
                D_tir = D_tir / d_tir_norm

                cos2 = np.sqrt(np.maximum(0.0, 1.0 - sin2_t2))
                D_refr = mu * D_local + (mu * cos1 - cos2) * normal_aligned
                d_refr_norm = np.linalg.norm(D_refr, axis=1, keepdims=True)
                d_refr_norm[d_refr_norm == 0.0] = 1.0
                D_refr = D_refr / d_refr_norm

                D_out_local = np.where(tir_mask[:, None], D_tir, D_refr)
                traced.tir |= tir_mask
                current_glass = surf.glass
                current_n_complex = next_n_complex
            else:
                D_out_local = D_local.copy()

            # 5. Transform hit point and direction back to Global frame
            hit_global = origin + hit_local @ R_mat.T
            dir_out_global = D_out_local @ R_mat.T
            d_out_g_norm = np.linalg.norm(dir_out_global, axis=1, keepdims=True)
            d_out_g_norm[d_out_g_norm == 0.0] = 1.0
            dir_out_global = dir_out_global / d_out_g_norm

            dist = np.linalg.norm(hit_global - P_glob, axis=1)
            traced.path_length += dist

            wl_mm = np.where(traced.wavelength > 0.01, traced.wavelength * 1e-3, traced.wavelength)
            delta_phi = (2.0 * np.pi * current_n_complex.real * dist) / np.maximum(1e-12, wl_mm) + plm_phase_vec
            traced.phase += delta_phi

            P_glob = hit_global
            D_glob = dir_out_global

            if record_history:
                step_entry = {
                    "surface_index": i,
                    "x": hit_local[:, 0].copy(),
                    "y": hit_local[:, 1].copy(),
                    "z": hit_global[:, 2].copy(),
                    "x_global": hit_global[:, 0].copy(),
                    "y_global": hit_global[:, 1].copy(),
                    "z_global": hit_global[:, 2].copy(),
                    "x_local": hit_local[:, 0].copy(),
                    "y_local": hit_local[:, 1].copy(),
                    "z_local": hit_local[:, 2].copy(),
                    "dir_x": dir_out_global[:, 0].copy(),
                    "dir_y": dir_out_global[:, 1].copy(),
                    "dir_z": dir_out_global[:, 2].copy(),
                    "dir": dir_out_global.copy(),
                    "vignetted": traced.vignetted.copy(),
                    "tir": traced.tir.copy(),
                    "intensity": traced.intensity.copy(),
                    "phase": traced.phase.copy(),
                    "mechanical_state": active_state_name(surf),
                }
                step_history.append(step_entry)

            if stype == SurfaceType.DETECTOR:
                break

        traced.pos = P_glob
        traced.dir = D_glob
        if record_history and step_history is not None:
            traced.history = step_history

        return traced

    def trace_rays_batch(
        self,
        rays: Union[List[Ray], RayBundle],
        record_history: bool = True,
        verbose: Optional[bool] = None,
    ) -> Union[List[Ray], RayBundle]:
        """
        Batch ray trace that accepts either a RayBundle or List[Ray].
        When given List[Ray], converts to RayBundle, traces vectorized, and returns List[Ray].
        """
        is_list = isinstance(rays, list)
        bundle = RayBundle.from_rays(rays) if is_list else rays
        traced_bundle = self.trace_bundle(bundle, record_history=record_history, verbose=verbose)
        if is_list:
            return traced_bundle.to_rays()
        return traced_bundle

    def trace_gaussian_beam(self, w0: float, wavelength: float = 0.5876, source_z: float = 0.0) -> Any:
        """Trace Gaussian beam waist and curvature through the optical system stack."""
        from optics_app.engine.gaussian import trace_system_gaussian_beam
        return trace_system_gaussian_beam(self.stack, w0=w0, wavelength=wavelength, source_z=source_z)


class DualSpeedTracer:
    """Dual-speed ray tracing engine providing fast interactive updates (<50ms) and full analysis."""
    def __init__(self, system_or_stack: Any):
        if isinstance(system_or_stack, OpticalSystem):
            self.opt_sys = system_or_stack
            self.system = system_or_stack
            self.stack = system_or_stack.stack
        elif hasattr(system_or_stack, "surface_stack"):
            self.system = system_or_stack
            self.stack = system_or_stack.surface_stack
            self.opt_sys = OpticalSystem(self.stack)
        elif hasattr(system_or_stack, "surfaces"):
            self.system = system_or_stack
            self.stack = system_or_stack
            self.opt_sys = OpticalSystem(self.stack)
        else:
            self.system = system_or_stack
            self.stack = getattr(system_or_stack, "stack", system_or_stack)
            self.opt_sys = OpticalSystem(self.stack)

    def trace_ray(self, ray: Ray, *args, **kwargs) -> Ray:
        return self.opt_sys.trace_ray(ray, *args, **kwargs)

    def trace_fast(
        self,
        wavelengths: Optional[List[float]] = None,
        num_rays: int = 5,
        beam_radius: float = 5.0,
        y_fan: bool = True,
        rays_per_wl: Optional[int] = None,
        **kwargs
    ) -> Any:
        import time
        try:
            from optics_app.engine.models import FastTraceResult
        except ImportError:
            from v2.engine.models import FastTraceResult
        if rays_per_wl is not None:
            num_rays = rays_per_wl
        t0 = time.perf_counter()
        wls = wavelengths or getattr(self.system, "wavelengths", [1.100, 1.225, 1.350])
        ray_paths = []
        spot_coords = {}
        vignetted = 0

        y_vals = np.linspace(-beam_radius, beam_radius, num_rays) if num_rays > 1 else [0.0]
        all_rays = []
        ray_wl_list = []
        for wl in wls:
            spot_coords[wl] = []
            for y in y_vals:
                all_rays.append(Ray(pos=np.array([0.0, float(y), 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=float(wl)))
                ray_wl_list.append(wl)

        bundle = RayBundle.from_rays(all_rays)
        traced_rays = self.opt_sys.trace_bundle(bundle, record_history=True).to_rays()
        total = len(traced_rays)
        for wl, traced in zip(ray_wl_list, traced_rays):
            if traced.vignetted:
                vignetted += 1
            pts = [np.array([h["x_global"], h["y_global"], h["z_global"]]) for h in traced.history]
            if pts:
                ray_paths.append(np.array(pts))
                spot_coords[wl].append([traced.history[-1]["x"], traced.history[-1]["y"]])

        for wl in wls:
            spot_coords[wl] = np.array(spot_coords[wl]) if spot_coords[wl] else np.empty((0, 2))

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return FastTraceResult(
            ray_paths=ray_paths,
            spot_coordinates=spot_coords,
            elapsed_ms=elapsed_ms,
            wavelengths=list(wls),
            vignetted_count=vignetted,
            total_rays=total
        )

    def trace_full(
        self,
        wavelengths: Optional[List[float]] = None,
        rings: int = 4,
        beam_radius: float = 5.0,
        arm_rays: int = 6,
        **kwargs
    ) -> Any:
        import time
        try:
            from optics_app.engine.models import FullTraceResult
        except ImportError:
            from v2.engine.models import FullTraceResult
        t0 = time.perf_counter()
        wls = wavelengths or getattr(self.system, "wavelengths", [1.100, 1.225, 1.350])
        ray_paths = []
        spot_coords = {}
        vignetted = 0
        all_spots = []

        pupil_pts = [(0.0, 0.0)]
        for r_idx in range(1, rings + 1):
            r_norm = r_idx / rings
            n_pts = r_idx * arm_rays
            for p_i in range(n_pts):
                th = 2.0 * np.pi * p_i / n_pts
                pupil_pts.append((beam_radius * r_norm * np.cos(th), beam_radius * r_norm * np.sin(th)))

        all_rays = []
        ray_wl_list = []
        for wl in wls:
            spot_coords[wl] = []
            for px, py in pupil_pts:
                all_rays.append(Ray(pos=np.array([float(px), float(py), 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=float(wl)))
                ray_wl_list.append(wl)

        bundle = RayBundle.from_rays(all_rays)
        traced_rays = self.opt_sys.trace_bundle(bundle, record_history=True).to_rays()
        total = len(traced_rays)
        for wl, traced in zip(ray_wl_list, traced_rays):
            if traced.vignetted:
                vignetted += 1
            else:
                if traced.history:
                    sx = traced.history[-1]["x"]
                    sy = traced.history[-1]["y"]
                    spot_coords[wl].append([sx, sy])
                    all_spots.append([sx, sy])
                    pts = [np.array([h["x_global"], h["y_global"], h["z_global"]]) for h in traced.history]
                    ray_paths.append(np.array(pts))

        spot_diagrams_by_wl = {}
        for wl in wls:
            arr_wl = np.array(spot_coords[wl]) if spot_coords[wl] else np.empty((0, 2))
            spot_coords[wl] = arr_wl
            if len(arr_wl) > 0:
                cx = float(np.mean(arr_wl[:, 0]))
                cy = float(np.mean(arr_wl[:, 1]))
                r_sq = (arr_wl[:, 0] - cx)**2 + (arr_wl[:, 1] - cy)**2
                rms_wl = float(np.sqrt(np.mean(r_sq)))
                geo_wl = float(np.max(np.sqrt(r_sq)))
            else:
                cx, cy, rms_wl, geo_wl = 0.0, 0.0, 0.0, 0.0
            spot_diagrams_by_wl[wl] = {
                "centroid": (cx, cy),
                "rms_radius": rms_wl,
                "geo_radius": geo_wl,
                "spot_count": len(arr_wl)
            }

        all_arr = np.array(all_spots) if all_spots else np.empty((0, 2))
        if len(all_arr) > 0:
            c_x = float(np.mean(all_arr[:, 0]))
            c_y = float(np.mean(all_arr[:, 1]))
            r_sq = (all_arr[:, 0] - c_x)**2 + (all_arr[:, 1] - c_y)**2
            rms_radius = float(np.sqrt(np.mean(r_sq)))
            geo_radius = float(np.max(np.sqrt(r_sq)))
            disp_spread = float(np.max(all_arr[:, 1]) - np.min(all_arr[:, 1]))
        else:
            c_x, c_y, rms_radius, geo_radius, disp_spread = 0.0, 0.0, 0.0, 0.0, 0.0

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return FullTraceResult(
            ray_paths=ray_paths,
            spot_coordinates=spot_coords,
            rms_radius=rms_radius,
            geo_radius=geo_radius,
            airy_radius=0.015,
            dispersion_spread=disp_spread,
            centroid_x=c_x,
            centroid_y=c_y,
            elapsed_ms=elapsed_ms,
            wavelengths=list(wls),
            vignetted_count=vignetted,
            total_rays=total,
            spot_diagrams_by_wl=spot_diagrams_by_wl
        )


# =========================================================================
# Generic Ray Tree & Branched Tracing Engine (Requirement R2 / Milestone M1)
# =========================================================================

@dataclass
class RayNode:
    """
    A single node/segment in a branched ray trace tree.
    Represents an incident or spawned child branch (reflected, transmitted, diffracted).
    """
    node_id: int
    parent_id: Optional[int]
    branch_type: str        # "incident", "transmitted", "reflected", "diffracted"
    generation: int         # 0 = root ray from source
    ray: Ray                # Ray instance for this branch segment
    children_ids: List[int] = field(default_factory=list)
    origin_surface_idx: int = 0
    terminal_surface_idx: int = -1


@dataclass
class RayTree:
    """
    Directed Acyclic Graph (DAG) representing the complete tree of ray branches
    spawned by beam splitters, multi-facet prisms, and active mechanisms.
    """
    root_node_id: int = 0
    nodes: Dict[int, RayNode] = field(default_factory=dict)

    def all_terminal_rays(self) -> List[Ray]:
        """Return all leaf rays that reached the end of their branch or terminated."""
        return [n.ray for n in self.nodes.values() if not n.children_ids]

    def all_rays(self) -> List[Ray]:
        """Return all ray segments across all tree nodes."""
        return [n.ray for n in self.nodes.values()]

    def total_energy(self) -> float:
        """Sum of optical intensities across all terminal leaf rays."""
        return float(sum(n.ray.intensity for n in self.nodes.values() if not n.children_ids))

    def flatten_paths(self) -> List[List[np.ndarray]]:
        """Return array of 3D polyline paths for visualization."""
        paths: List[List[np.ndarray]] = []
        for node in self.nodes.values():
            pts: List[np.ndarray] = []
            if hasattr(node.ray, "launch_pos") and node.ray.launch_pos is not None:
                pts.append(np.asarray(node.ray.launch_pos, dtype=float))
            for hit in getattr(node.ray, "history", []):
                pts.append(np.array([
                    float(hit.get("x_global", hit.get("x", 0.0))),
                    float(hit.get("y_global", hit.get("y", 0.0))),
                    float(hit.get("z_global", hit.get("z", 0.0)))
                ], dtype=float))
            if len(pts) >= 2:
                paths.append(pts)
        return paths


class BranchedTracer:
    """
    Queue-based branched ray tracer supporting:
    - Specular beam splitter division into simultaneous transmitted and reflected ray trees
    - Polarization-dependent (PBS) and dichroic wavelength-dependent split ratios
    - Continuous galvo scanner deflection with off-vertex pivot offset kinematics
    - Multi-interface prism refraction and chromatic angular dispersion
    - Configurable tree depth limit (max_depth) and intensity cutoff (min_intensity)
    """

    def __init__(
        self,
        system: Union[OpticalSystem, SurfaceStack, Any],
        max_depth: int = 5,
        min_intensity: float = 1e-4,
    ):
        if isinstance(system, OpticalSystem):
            self.opt_sys = system
        elif hasattr(system, "surfaces"):
            self.opt_sys = OpticalSystem(system)
        else:
            raise TypeError("system must be an OpticalSystem or SurfaceStack")
        self.max_depth = max_depth
        self.min_intensity = min_intensity

    def trace_bundle_tree(self, initial_rays: Sequence[Ray]) -> List[RayTree]:
        """Trace a collection of rays into individual ray trees."""
        return [self.trace_ray_tree(r) for r in initial_rays]

    def trace_ray_tree(self, initial_ray: Ray) -> RayTree:
        """
        Trace a single ray through the optical system, spawning child branches
        whenever light encounters a beam splitter surface.
        """
        tree = RayTree(root_node_id=0)
        root_ray = initial_ray.copy()
        root_ray.launch_pos = root_ray.pos.copy()
        root_node = RayNode(
            node_id=0,
            parent_id=None,
            branch_type="incident",
            generation=0,
            ray=root_ray,
            origin_surface_idx=0
        )
        tree.nodes[0] = root_node

        # Queue contains tuples of (node_id, ray_segment, starting_surface_index)
        queue: List[Tuple[int, Ray, int]] = [(0, root_ray, 0)]
        node_counter = 0

        surfaces = self.opt_sys.surfaces
        num_surfaces = len(surfaces)

        while queue:
            parent_id, curr_ray, start_idx = queue.pop(0)
            curr_node = tree.nodes[parent_id]

            current_glass = "AIR"
            current_n_complex = get_complex_refractive_index("AIR", curr_ray.wavelength, 0.0)

            for i in range(start_idx, num_surfaces):
                surf = surfaces[i]

                if curr_ray.vignetted:
                    curr_node.terminal_surface_idx = i
                    break

                # 1. Resolve unified effective pose (galvo deflection, pivot offset, DMD tilts)
                origin, R_mat = get_effective_pose(surf)

                pos_local = R_mat.T @ (curr_ray.pos - origin)
                dir_local = R_mat.T @ curr_ray.dir
                l_norm = _norm3(dir_local)
                if l_norm > 0:
                    dir_local = dir_local / l_norm

                local_s_basis = R_mat.T @ curr_ray.s_basis
                s_norm = _norm3(local_s_basis)
                if s_norm > 0:
                    local_s_basis = local_s_basis / s_norm

                local_ray = Ray(
                    pos=pos_local,
                    dir=dir_local,
                    wavelength=curr_ray.wavelength,
                    intensity=curr_ray.intensity,
                    jones_vector=curr_ray.jones_vector.copy(),
                    phase=curr_ray.phase
                )

                if bool(getattr(surf, "is_air_space", False)):
                    curr_ray.history.append({
                        "surface_index": i,
                        "x": float(pos_local[0]), "y": float(pos_local[1]), "z": float(curr_ray.pos[2]),
                        "x_global": float(curr_ray.pos[0]), "y_global": float(curr_ray.pos[1]), "z_global": float(curr_ray.pos[2]),
                        "x_local": float(pos_local[0]), "y_local": float(pos_local[1]), "z_local": float(pos_local[2]),
                        "dir_x": float(curr_ray.dir[0]), "dir_y": float(curr_ray.dir[1]), "dir_z": float(curr_ray.dir[2]),
                        "dir": curr_ray.dir.copy(),
                        "vignetted": False, "tir": curr_ray.tir,
                        "intensity": float(curr_ray.intensity),
                        "phase": float(curr_ray.phase),
                        "interaction": "air_space", "reference_only": True,
                    })
                    continue

                # 2. Ray-surface intersection solver in local frame
                res = ray_surface_intersect(surf, local_ray)
                if res is not None:
                    travel = float(np.dot(np.asarray(res[0]) - pos_local, dir_local))
                    if not np.isfinite(travel) or travel < -1e-9:
                        res = None

                if res is None:
                    curr_ray.vignetted = True
                    curr_node.terminal_surface_idx = i
                    if i > 0 and len(curr_ray.history) > 0:
                        prev_s = surfaces[i - 1]
                        proj_dist = float(getattr(prev_s, "thickness", 50.0))
                        if proj_dist <= 0: proj_dist = 50.0
                        term_pos = curr_ray.pos + curr_ray.dir * proj_dist
                        curr_ray.history.append({
                            "surface_index": i,
                            "x": float(term_pos[0]), "y": float(term_pos[1]), "z": float(term_pos[2]),
                            "x_global": float(term_pos[0]), "y_global": float(term_pos[1]), "z_global": float(term_pos[2]),
                            "x_local": 0.0, "y_local": 0.0, "z_local": 0.0,
                            "dir_x": float(curr_ray.dir[0]), "dir_y": float(curr_ray.dir[1]), "dir_z": float(curr_ray.dir[2]),
                            "dir": curr_ray.dir.copy(),
                            "vignetted": True, "tir": curr_ray.tir,
                            "intensity": 0.0, "failure_reason": "no_forward_intersection",
                        })
                    break

                hit_local, normal_local = res

                # 3. Aperture vignetting check
                is_cb = getattr(surf, 'is_coord_break', False) or (getattr(surf, 'surface_type', None) == getattr(SurfaceType, 'COORDINATE_BREAK', None))
                if not is_cb and (surf.surface_type != SurfaceType.OBJECT or (surf.semi_diameter > 0 and i > 0)):
                    if hasattr(surf, 'check_aperture'):
                        if not surf.check_aperture(float(hit_local[0]), float(hit_local[1])):
                            curr_ray.vignetted = True
                    else:
                        r_dist = float(np.sqrt(hit_local[0]**2 + hit_local[1]**2))
                        if surf.semi_diameter > 0 and r_dist > surf.semi_diameter:
                            curr_ray.vignetted = True

                if curr_ray.vignetted:
                    curr_node.terminal_surface_idx = i
                    break

                # 4. Refractive index & polarization basis
                ext_k = getattr(surf, 'extinction_coeff', 0.0)
                next_n_complex = get_complex_refractive_index(surf.glass, curr_ray.wavelength, ext_k) if surf.glass else get_complex_refractive_index("AIR", curr_ray.wavelength, 0.0)
                is_mirror = surf.is_mirror()

                s_surf, p_surf_inc, normal_aligned = compute_sp_basis(dir_local, normal_local, local_s_basis)
                J_inc = rotate_jones_vector(curr_ray.jones_vector, local_s_basis, s_surf, dir_local)
                cos_inc = float(np.clip(-float(dir_local[0]*normal_aligned[0] + dir_local[1]*normal_aligned[1] + dir_local[2]*normal_aligned[2]), -1.0, 1.0))

                if is_mirror and ext_k == 0.0:
                    fresnel = FresnelResult(
                        r_s=complex(-1.0, 0.0), r_p=complex(1.0, 0.0),
                        t_s=complex(0.0, 0.0), t_p=complex(0.0, 0.0),
                        tir=False, theta_inc=float(np.arccos(cos_inc)),
                        theta_trans=complex(0.0, 0.0)
                    )
                else:
                    fresnel = compute_fresnel_coefficients(current_n_complex, next_n_complex, cos_inc)

                hit_global = origin + R_mat @ hit_local
                dist = _norm3(hit_global - curr_ray.pos)
                curr_ray.path_length += dist

                wl_mm = curr_ray.wavelength * 1e-3 if curr_ray.wavelength > 0.01 else curr_ray.wavelength
                delta_phi = (2.0 * np.pi * current_n_complex.real * dist) / wl_mm
                curr_ray.phase += float(delta_phi)

                # 5. Check Beam Splitter Branching
                is_bs = bool(getattr(surf, "is_beam_splitter", False)) or (getattr(surf, "beamsplitter_config", None) is not None)
                if is_bs and curr_node.generation < self.max_depth:
                    bs_cfg = getattr(surf, "beamsplitter_config", None)
                    if bs_cfg is not None and hasattr(bs_cfg, "evaluate_jones"):
                        j_trans_loc, I_trans, j_refl_loc, I_refl = bs_cfg.evaluate_jones(
                            J_inc, curr_ray.wavelength, curr_ray.intensity
                        )
                        T_coeff = I_trans / max(1e-12, curr_ray.intensity)
                        R_coeff = I_refl / max(1e-12, curr_ray.intensity)
                    elif bs_cfg is not None and hasattr(bs_cfg, "evaluate_split"):
                        R_coeff, T_coeff = bs_cfg.evaluate_split(curr_ray.wavelength)
                        I_trans = float(curr_ray.intensity * T_coeff)
                        I_refl = float(curr_ray.intensity * R_coeff)
                        j_trans_loc = math.sqrt(max(0.0, T_coeff)) * J_inc
                        j_refl_loc = math.sqrt(max(0.0, R_coeff)) * np.array([J_inc[0], -J_inc[1]], dtype=complex)
                    elif bs_cfg is not None:
                        R_coeff = float(getattr(bs_cfg, "split_ratio_R", 0.5))
                        T_coeff = float(getattr(bs_cfg, "split_ratio_T", 0.5))
                        I_trans = float(curr_ray.intensity * T_coeff)
                        I_refl = float(curr_ray.intensity * R_coeff)
                        j_trans_loc = math.sqrt(max(0.0, T_coeff)) * J_inc
                        j_refl_loc = math.sqrt(max(0.0, R_coeff)) * np.array([J_inc[0], -J_inc[1]], dtype=complex)
                    else:
                        R_coeff, T_coeff = 0.5, 0.5
                        I_trans = float(curr_ray.intensity * 0.5)
                        I_refl = float(curr_ray.intensity * 0.5)
                        j_trans_loc = math.sqrt(0.5) * J_inc
                        j_refl_loc = math.sqrt(0.5) * np.array([J_inc[0], -J_inc[1]], dtype=complex)

                    # Terminate parent branch at the splitter interface
                    curr_node.terminal_surface_idx = i
                    curr_ray.history.append({
                        "surface_index": i,
                        "x": float(hit_local[0]), "y": float(hit_local[1]), "z": float(hit_global[2]),
                        "x_global": float(hit_global[0]), "y_global": float(hit_global[1]), "z_global": float(hit_global[2]),
                        "x_local": float(hit_local[0]), "y_local": float(hit_local[1]), "z_local": float(hit_local[2]),
                        "dir_x": float(curr_ray.dir[0]), "dir_y": float(curr_ray.dir[1]), "dir_z": float(curr_ray.dir[2]),
                        "dir": curr_ray.dir.copy(),
                        "vignetted": False, "tir": False,
                        "intensity": float(curr_ray.intensity),
                        "phase": float(curr_ray.phase),
                        "interaction": "beam_splitter_split"
                    })

                    # Calculate reflected direction
                    dot_dn = float(dir_local[0]*normal_aligned[0] + dir_local[1]*normal_aligned[1] + dir_local[2]*normal_aligned[2])
                    dir_refl_local = dir_local - (2.0 * dot_dn) * normal_aligned
                    r_norm = _norm3(dir_refl_local)
                    if r_norm > 0:
                        dir_refl_local /= r_norm
                    dir_refl_global = R_mat @ dir_refl_local
                    g_norm = _norm3(dir_refl_global)
                    if g_norm > 0:
                        dir_refl_global /= g_norm

                    # Calculate transmitted direction
                    if surf.glass and surf.glass.upper() not in ("AIR", "MIRROR"):
                        dir_trans_local, _ = snell_refract_3d(dir_local, normal_aligned, current_n_complex.real, next_n_complex.real)
                    else:
                        dir_trans_local = dir_local
                    dir_trans_global = R_mat @ dir_trans_local
                    g_norm = _norm3(dir_trans_global)
                    if g_norm > 0:
                        dir_trans_global /= g_norm

                    # 5a. Spawn Transmitted Child
                    if I_trans >= self.min_intensity:
                        node_counter += 1
                        t_ray = curr_ray.copy()
                        t_ray.intensity = float(I_trans)
                        t_ray.jones_vector = j_trans_loc.copy()
                        t_ray.pos = hit_global.copy()
                        t_ray.dir = dir_trans_global.copy()
                        t_ray.launch_pos = hit_global.copy()
                        t_ray.s_basis = R_mat @ s_surf
                        p_surf_trans = _cross3(s_surf, dir_trans_local)
                        p_norm_t = _norm3(p_surf_trans)
                        t_ray.p_basis = R_mat @ (p_surf_trans / p_norm_t if p_norm_t > 0 else p_surf_inc)
                        t_ray.history = [dict(h) for h in curr_ray.history]
                        t_node = RayNode(
                            node_id=node_counter,
                            parent_id=parent_id,
                            branch_type="transmitted",
                            generation=curr_node.generation + 1,
                            ray=t_ray,
                            origin_surface_idx=i
                        )
                        tree.nodes[node_counter] = t_node
                        curr_node.children_ids.append(node_counter)
                        queue.append((node_counter, t_ray, i + 1))

                    # 5b. Spawn Reflected Child
                    if I_refl >= self.min_intensity:
                        node_counter += 1
                        r_ray = curr_ray.copy()
                        r_ray.intensity = float(I_refl)
                        r_ray.jones_vector = j_refl_loc.copy()
                        r_ray.pos = hit_global.copy()
                        r_ray.dir = dir_refl_global.copy()
                        r_ray.launch_pos = hit_global.copy()
                        r_ray.s_basis = R_mat @ s_surf
                        p_surf_refl = _cross3(s_surf, dir_refl_local)
                        p_norm_r = _norm3(p_surf_refl)
                        r_ray.p_basis = R_mat @ (p_surf_refl / p_norm_r if p_norm_r > 0 else p_surf_inc)
                        r_ray.history = [dict(h) for h in curr_ray.history]
                        r_node = RayNode(
                            node_id=node_counter,
                            parent_id=parent_id,
                            branch_type="reflected",
                            generation=curr_node.generation + 1,
                            ray=r_ray,
                            origin_surface_idx=i
                        )
                        tree.nodes[node_counter] = r_node
                        curr_node.children_ids.append(node_counter)

                        refl_start = getattr(surf, "reflected_branch_start_idx", None)
                        if refl_start is None:
                            for idx_s, s_cand in enumerate(surfaces):
                                if getattr(s_cand, "is_reflected_arm", False) or getattr(s_cand, "branch", "").lower() == "reflected":
                                    refl_start = idx_s
                                    break
                        if refl_start is not None and 0 <= refl_start < num_surfaces:
                            queue.append((node_counter, r_ray, refl_start))
                        else:
                            r_term_dist = float(getattr(surf, "thickness", 50.0))
                            if r_term_dist <= 0:
                                r_term_dist = 50.0
                            r_term_pos = r_ray.pos + r_ray.dir * r_term_dist
                            r_ray.history.append({
                                "surface_index": i,
                                "x": float(r_term_pos[0]), "y": float(r_term_pos[1]), "z": float(r_term_pos[2]),
                                "x_global": float(r_term_pos[0]), "y_global": float(r_term_pos[1]), "z_global": float(r_term_pos[2]),
                                "x_local": 0.0, "y_local": 0.0, "z_local": 0.0,
                                "dir_x": float(r_ray.dir[0]), "dir_y": float(r_ray.dir[1]), "dir_z": float(r_ray.dir[2]),
                                "dir": r_ray.dir.copy(),
                                "vignetted": False, "tir": False,
                                "intensity": float(r_ray.intensity),
                                "phase": float(r_ray.phase),
                                "interaction": "reflected_free_space"
                            })

                    # This branch finished splitting, break from sequential loop
                    break

                # 6. Standard Physical Interactions
                surf_is_modulator = is_plm(surf) or (is_dmd(surf) and (getattr(surf, "plm_config", None) is not None or getattr(surf, "plm_enabled", False) or getattr(surf, "is_dmd", False)))
                if surf_is_modulator:
                    plm_cfg = ensure_plm_config(surf)
                    plm_st = active_state_name(surf) or "ON"
                    dir_out_global_plm, plm_phase_rad, plm_eff, plm_state_4bit = deflect_ray_on_plm_3d(
                        ray_dir_in=curr_ray.dir,
                        surf_normal=R_mat @ normal_local,
                        surf_global_R=R_mat,
                        local_x_mm=float(hit_local[0]),
                        local_y_mm=float(hit_local[1]),
                        plm_cfg=plm_cfg,
                        mechanical_state=plm_st,
                        wavelength_um=curr_ray.wavelength,
                    )
                    dir_out_local = R_mat.T @ dir_out_global_plm
                    p_surf_out = _cross3(s_surf, dir_out_local)
                    p_norm = _norm3(p_surf_out)
                    p_surf_out = p_surf_out / p_norm if p_norm > 0 else p_surf_inc
                    phase_fac = np.exp(1j * plm_phase_rad)
                    eff_amp = math.sqrt(max(0.0, float(plm_eff))) if plm_eff is not None else 1.0
                    J_out = np.array([fresnel.r_s * J_inc[0], fresnel.r_p * J_inc[1]], dtype=complex) * (phase_fac * eff_amp)
                    if (plm_cfg.is_dmd and plm_state_4bit == 0 and (current_active_state or "ON") == "ON" and plm_cfg.pattern.enabled) or (plm_eff is not None and float(plm_eff) <= 1e-9):
                        traced_ray.vignetted = True
                        traced_ray.intensity = 0.0
                        J_out = np.zeros(2, dtype=complex)
                    curr_ray.phase += plm_phase_rad
                    if plm_eff is not None:
                        curr_ray.intensity *= float(plm_eff)

                elif surf.surface_type == SurfaceType.GRATING and surf.grating_lines_per_mm > 0:
                    try:
                        from optics_app.engine.grating import grating_diffract_3d
                    except ImportError:
                        from v2.engine.grating import grating_diffract_3d
                    global_grating_normal = surf.global_R[:, 2].copy()
                    if (curr_ray.dir[0]*global_grating_normal[0] + curr_ray.dir[1]*global_grating_normal[1] + curr_ray.dir[2]*global_grating_normal[2]) > 0:
                        global_grating_normal = -global_grating_normal
                    local_disp = np.array([0.0, 1.0, 0.0]) if abs(getattr(surf, 'tilt_x', 0.0)) > abs(getattr(surf, 'tilt_y', 0.0)) else np.array([1.0, 0.0, 0.0])
                    global_line_vector = surf.global_R @ local_disp

                    dir_out_global_grating, evan = grating_diffract_3d(
                        ray_dir=curr_ray.dir,
                        grating_normal=global_grating_normal,
                        line_vector=global_line_vector,
                        line_density_lines_per_mm=surf.grating_lines_per_mm,
                        order=surf.grating_order,
                        wavelength_um=curr_ray.wavelength,
                        is_reflection=is_mirror
                    )
                    if evan or dir_out_global_grating is None:
                        curr_ray.vignetted = True
                        dir_out_local = dir_local
                    else:
                        dir_out_local = R_mat.T @ dir_out_global_grating
                    p_surf_out = _cross3(s_surf, dir_out_local)
                    p_norm = _norm3(p_surf_out)
                    p_surf_out = p_surf_out / p_norm if p_norm > 0 else p_surf_inc
                    J_out = np.array([fresnel.r_s * J_inc[0], fresnel.r_p * J_inc[1]], dtype=complex) if is_mirror else np.array([fresnel.t_s * J_inc[0], fresnel.t_p * J_inc[1]], dtype=complex)

                elif is_mirror:
                    dot_dn = float(dir_local[0]*normal_aligned[0] + dir_local[1]*normal_aligned[1] + dir_local[2]*normal_aligned[2])
                    dir_out_local = dir_local - (2.0 * dot_dn) * normal_aligned
                    r_norm = _norm3(dir_out_local)
                    if r_norm > 0:
                        dir_out_local = dir_out_local / r_norm
                    p_surf_out = _cross3(s_surf, dir_out_local)
                    p_norm = _norm3(p_surf_out)
                    p_surf_out = p_surf_out / p_norm if p_norm > 0 else p_surf_inc
                    J_out = np.array([fresnel.r_s * J_inc[0], fresnel.r_p * J_inc[1]], dtype=complex)

                elif surf.surface_type not in (SurfaceType.DETECTOR, SurfaceType.STOP, SurfaceType.OBJECT, getattr(SurfaceType, 'COORDINATE_BREAK', None)) and not is_cb:
                    dir_out_local, tir = snell_refract_3d(dir_local, normal_aligned, current_n_complex.real, next_n_complex.real)
                    if tir or fresnel.tir:
                        curr_ray.tir = True
                        p_surf_out = _cross3(s_surf, dir_out_local)
                        p_norm = _norm3(p_surf_out)
                        p_surf_out = p_surf_out / p_norm if p_norm > 0 else p_surf_inc
                        J_out = np.array([fresnel.r_s * J_inc[0], fresnel.r_p * J_inc[1]], dtype=complex)
                    else:
                        p_surf_out = _cross3(s_surf, dir_out_local)
                        p_norm = _norm3(p_surf_out)
                        p_surf_out = p_surf_out / p_norm if p_norm > 0 else p_surf_inc
                        J_out = np.array([fresnel.t_s * J_inc[0], fresnel.t_p * J_inc[1]], dtype=complex)
                    current_glass = surf.glass
                    current_n_complex = next_n_complex
                else:
                    dir_out_local = dir_local
                    p_surf_out = p_surf_inc
                    J_out = J_inc

                if hasattr(surf, 'jones_matrix') and surf.jones_matrix is not None:
                    J_out = np.array(surf.jones_matrix, dtype=complex) @ J_out

                curr_ray.jones_vector = J_out
                state_T = state_transmission(surf, curr_ray.wavelength)
                if state_T < 1.0:
                    curr_ray.jones_vector *= np.sqrt(state_T)

                dir_out_global = R_mat @ dir_out_local
                g_norm = _norm3(dir_out_global)
                if g_norm > 0:
                    dir_out_global = dir_out_global / g_norm

                curr_ray.s_basis = R_mat @ s_surf
                curr_ray.p_basis = R_mat @ p_surf_out
                curr_ray.pos = hit_global
                curr_ray.dir = dir_out_global
                curr_ray.update_intensity()

                curr_ray.history.append({
                    "surface_index": i,
                    "x": float(hit_local[0]), "y": float(hit_local[1]), "z": float(hit_global[2]),
                    "x_global": float(hit_global[0]), "y_global": float(hit_global[1]), "z_global": float(hit_global[2]),
                    "x_local": float(hit_local[0]), "y_local": float(hit_local[1]), "z_local": float(hit_local[2]),
                    "dir_x": float(dir_out_global[0]), "dir_y": float(dir_out_global[1]), "dir_z": float(dir_out_global[2]),
                    "dir": curr_ray.dir.copy(),
                    "vignetted": curr_ray.vignetted, "tir": curr_ray.tir,
                    "intensity": float(curr_ray.intensity),
                    "phase": float(curr_ray.phase),
                    "theta_inc": float(getattr(fresnel, "theta_inc", 0.0)),
                    "mechanical_state": active_state_name(surf),
                })


                curr_node.terminal_surface_idx = i
                if surf.surface_type == SurfaceType.DETECTOR:
                    break

        return tree
