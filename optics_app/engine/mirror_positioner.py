"""
Automated Mirror Spatial Positioning & Ray Aiming Optimization Engine.
Provides tools for:
1. aim_mirror_to_global_point: Exact analytical law-of-reflection ray aiming to a 3D target in space.
2. set_mirror_fold_angle: Automatic 90-degree, 45-degree, or arbitrary-angle beam folding with coordinate alignment.
3. auto_position_mirror_distance: Solves axial distance/thickness for optimal beam centering on mirror aperture.
4. optimize_mirror_spatial_position: Multi-DOF optimizer for mirror position, tilts, decenters, and air gaps.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, Union
import math
import numpy as np
from scipy.optimize import minimize

from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem, build_rotation_matrix
from optics_app.engine.ray import Ray


@dataclass
class MirrorAimResult:
    """Outcome of an analytical mirror ray-aim solve."""
    success: bool
    surface_index: int
    target_global_point: Tuple[float, float, float]
    solved_tilt_x: float
    solved_tilt_y: float
    solved_tilt_z: float
    target_error_mm: float
    message: str = "Mirror aiming solve completed"


@dataclass
class MirrorPositionResult:
    """Outcome of a multi-variable mirror spatial positioning optimization."""
    success: bool
    surface_index: int
    initial_error_mm: float
    final_error_mm: float
    initial_rms_um: float
    final_rms_um: float
    optimized_params: Dict[str, float] = field(default_factory=dict)
    iterations: int = 0
    message: str = "Mirror spatial optimization completed"


class MirrorPositioner:
    """
    Automated tools to calculate and optimize the spatial position and orientation of mirrors.
    """

    @staticmethod
    def set_mirror_fold_angle(
        stack: SurfaceStack,
        mirror_index: int,
        fold_angle_deg: float = 90.0,
        fold_plane: str = "YZ"
    ) -> Tuple[float, float]:
        """
        Sets the mirror tilt to achieve an exact beam deviation (fold angle).
        For a fold angle theta, the required mirror tilt is theta / 2.0.

        Parameters:
            stack: SurfaceStack containing the mirror.
            mirror_index: Index of the mirror surface.
            fold_angle_deg: Desired beam turn angle in degrees (e.g. 90.0 for right-angle fold).
            fold_plane: "YZ" (bends beam in Y-direction via tilt_x) or "XZ" (bends beam in X-direction via tilt_y).
        Returns:
            Tuple of (tilt_x, tilt_y) applied to the mirror.
        """
        if mirror_index < 0 or mirror_index >= len(stack.surfaces):
            raise ValueError(f"Invalid mirror index {mirror_index}")

        mirror_surf = stack.surfaces[mirror_index]
        is_reflective = (
            getattr(mirror_surf, "surface_type", None) == SurfaceType.GRATING
            or (hasattr(mirror_surf, "is_mirror") and mirror_surf.is_mirror())
            or (hasattr(mirror_surf, "is_reflective") and mirror_surf.is_reflective())
        )
        if not is_reflective:
            raise ValueError(
                f"Mirror fold presets can only be applied to reflective surfaces; "
                f"Surface #{mirror_index} ('{mirror_surf.name}') is refractive/non-reflective."
            )
        half_angle = fold_angle_deg / 2.0

        if fold_plane.upper() == "YZ":
            mirror_surf.tilt_x = half_angle
            mirror_surf.tilt_y = 0.0
        elif fold_plane.upper() == "XZ":
            mirror_surf.tilt_x = 0.0
            mirror_surf.tilt_y = half_angle
        else:
            mirror_surf.tilt_x = half_angle
            mirror_surf.tilt_y = 0.0

        stack._update_accumulated_positions()
        return (mirror_surf.tilt_x, mirror_surf.tilt_y)

    @staticmethod
    def aim_mirror_to_global_point(
        stack: SurfaceStack,
        mirror_index: int,
        target_point: Tuple[float, float, float],
        wavelength: float = 0.5876
    ) -> MirrorAimResult:
        """
        Analytically computes and sets the exact mirror tilt angles (tilt_x, tilt_y)
        so that the reflected chief ray points directly at target_point (X, Y, Z) in global coordinates.

        Parameters:
            stack: SurfaceStack containing the optical system.
            mirror_index: Index of the mirror surface in the stack.
            target_point: 3D point (X, Y, Z) in global coordinates to aim at.
            wavelength: Ray tracing wavelength (um).
        """
        if mirror_index < 0 or mirror_index >= len(stack.surfaces):
            return MirrorAimResult(
                success=False, surface_index=mirror_index,
                target_global_point=target_point,
                solved_tilt_x=0.0, solved_tilt_y=0.0, solved_tilt_z=0.0,
                target_error_mm=float('inf'),
                message=f"Invalid mirror index {mirror_index}"
            )

        # 1. Update geometry with zero tilt to determine incoming ray vector at mirror
        mirror_surf = stack.surfaces[mirror_index]
        orig_tilt_x = mirror_surf.tilt_x
        orig_tilt_y = mirror_surf.tilt_y
        orig_tilt_z = mirror_surf.tilt_z

        mirror_surf.tilt_x = 0.0
        mirror_surf.tilt_y = 0.0
        mirror_surf.tilt_z = 0.0
        stack._update_accumulated_positions()

        # Trace chief ray up to mirror
        opt_sys = OpticalSystem(stack)
        chief_ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wavelength)
        tr = opt_sys.trace_ray(chief_ray)

        if not tr or len(tr.history) <= mirror_index:
            # Restore tilts
            mirror_surf.tilt_x = orig_tilt_x
            mirror_surf.tilt_y = orig_tilt_y
            stack._update_accumulated_positions()
            return MirrorAimResult(
                success=False, surface_index=mirror_index, target_global_point=target_point,
                solved_tilt_x=orig_tilt_x, solved_tilt_y=orig_tilt_y, solved_tilt_z=orig_tilt_z,
                target_error_mm=float('inf'), message="Chief ray could not reach mirror"
            )

        hit = tr.history[mirror_index]
        p_mirror_global = np.array([hit.get("x_global", 0.0), hit.get("y_global", 0.0), hit.get("z_global", 0.0)])
        if mirror_index > 0:
            prev_hit = tr.history[mirror_index - 1]
            v_inc_global = np.array([prev_hit.get("dir_x", 0.0), prev_hit.get("dir_y", 0.0), prev_hit.get("dir_z", 1.0)])
        else:
            v_inc_global = chief_ray.dir.copy()
        v_inc_global /= np.linalg.norm(v_inc_global)

        # Desired outgoing direction vector in global space
        target_vec = np.array(target_point, dtype=float) - p_mirror_global
        dist_to_target = np.linalg.norm(target_vec)
        if dist_to_target < 1e-6:
            v_out_global = v_inc_global
        else:
            v_out_global = target_vec / dist_to_target

        # Law of reflection: v_out = v_inc - 2 (v_inc . N) N
        # N_ideal = (v_out - v_inc) / ||v_out - v_inc||
        delta_v = v_out_global - v_inc_global
        norm_delta = np.linalg.norm(delta_v)
        if norm_delta < 1e-6:
            # No deviation required
            solved_tilt_x, solved_tilt_y = 0.0, 0.0
        else:
            n_global = delta_v / norm_delta

            # Convert global normal into mirror's incoming reference frame
            frame_R = mirror_surf.global_R  # Reference frame without tilt
            n_local = frame_R.T @ n_global

            # Normal vector with Euler angles R = Ry(tilt_y) @ Rx(tilt_x):
            # N_local = [sin(tilt_y)*cos(tilt_x), -sin(tilt_x), cos(tilt_y)*cos(tilt_x)]
            # Align normal vector to point towards positive half-space (n_local[2] >= 0)
            if n_local[2] < 0:
                n_local = -n_local

            solved_tilt_x = math.degrees(math.asin(np.clip(-n_local[1], -1.0, 1.0)))
            denom = math.cos(math.radians(solved_tilt_x))
            if abs(denom) > 1e-4:
                solved_tilt_y = math.degrees(math.atan2(n_local[0], n_local[2]))
            else:
                solved_tilt_y = 0.0

        mirror_surf.tilt_x = float(solved_tilt_x)
        mirror_surf.tilt_y = float(solved_tilt_y)
        stack._update_accumulated_positions()

        # Measure pointing error
        opt_sys_new = OpticalSystem(stack)
        tr_new = opt_sys_new.trace_ray(chief_ray)
        final_err = 0.0
        if tr_new and len(tr_new.history) > mirror_index:
            h_mirror = tr_new.history[mirror_index]
            p_refl = np.array([h_mirror.get("x_global", 0.0), h_mirror.get("y_global", 0.0), h_mirror.get("z_global", 0.0)])
            d_refl = np.array([h_mirror.get("dir_x", 0.0), h_mirror.get("dir_y", 0.0), h_mirror.get("dir_z", 1.0)])
            d_refl /= np.linalg.norm(d_refl)
            
            # Distance from target_point to ray line
            v_to_target = np.array(target_point, dtype=float) - p_refl
            proj = np.dot(v_to_target, d_refl)
            closest_pt = p_refl + proj * d_refl
            final_err = float(np.linalg.norm(np.array(target_point, dtype=float) - closest_pt))

        return MirrorAimResult(
            success=final_err < 0.01,
            surface_index=mirror_index,
            target_global_point=target_point,
            solved_tilt_x=float(solved_tilt_x),
            solved_tilt_y=float(solved_tilt_y),
            solved_tilt_z=0.0,
            target_error_mm=final_err,
            message=f"Mirror aimed at target {target_point} with {final_err*1000.0:.2f} um error"
        )

    @staticmethod
    def aim_mirror_to_target(
        stack: SurfaceStack,
        mirror_index: int,
        target_surface_index: int = -1,
        target_xy: Tuple[float, float] = (0.0, 0.0),
        wavelength: float = 0.5876,
        fold_angle_deg: Optional[float] = None
    ) -> MirrorAimResult:
        """
        Aims the mirror at a target surface or sets the standard fold angle.
        If fold_angle_deg is provided, directly sets the fold angle.
        Otherwise, aims the chief ray so the downstream optical axis centers on the target.
        """
        if fold_angle_deg is not None:
            tx, ty = MirrorPositioner.set_mirror_fold_angle(stack, mirror_index, fold_angle_deg=fold_angle_deg)
            return MirrorAimResult(
                success=True, surface_index=mirror_index,
                target_global_point=(0.0, 0.0, 0.0),
                solved_tilt_x=tx, solved_tilt_y=ty, solved_tilt_z=0.0,
                target_error_mm=0.0,
                message=f"Mirror fold angle set to {fold_angle_deg:.1f} deg"
            )

        if target_surface_index == -1 or target_surface_index >= len(stack.surfaces):
            target_surface_index = len(stack.surfaces) - 1

        target_surf = stack.surfaces[target_surface_index]
        target_pt = tuple(target_surf.global_origin)

        return MirrorPositioner.aim_mirror_to_global_point(
            stack=stack,
            mirror_index=mirror_index,
            target_point=target_pt,
            wavelength=wavelength
        )

    @staticmethod
    def auto_position_mirror_distance(
        stack: SurfaceStack,
        mirror_index: int,
        target_beam_diameter_mm: Optional[float] = None,
        min_clearance_mm: float = 5.0,
        wavelength: float = 0.5876
    ) -> float:
        """
        Finds the optimal axial distance (thickness of the preceding surface) so that
        the expanding/converging beam fits comfortably on the mirror clear aperture.
        """
        if mirror_index <= 0 or mirror_index >= len(stack.surfaces):
            return 0.0

        prev_surf = stack.surfaces[mirror_index - 1]
        mirror_surf = stack.surfaces[mirror_index]
        clear_radius = mirror_surf.semi_diameter

        if target_beam_diameter_mm is not None:
            desired_beam_r = target_beam_diameter_mm / 2.0
        else:
            desired_beam_r = max(1.0, clear_radius - min_clearance_mm)

        def eval_dist(thick):
            prev_surf.thickness = float(thick[0])
            stack._update_accumulated_positions()
            opt_sys = OpticalSystem(stack)

            rays = [
                Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([math.sin(0.1), 0.0, math.cos(0.1)]), wavelength=wavelength),
                Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([-math.sin(0.1), 0.0, math.cos(0.1)]), wavelength=wavelength),
            ]
            r_hits = []
            for r in rays:
                tr = opt_sys.trace_ray(r)
                if tr and len(tr.history) > mirror_index:
                    h = tr.history[mirror_index]
                    r_hits.append(abs(h.get("x_local", 0.0)))

            if not r_hits:
                return 1e6

            beam_r = max(r_hits)
            return (beam_r - desired_beam_r)**2

        res = minimize(eval_dist, x0=[prev_surf.thickness], bounds=[(1.0, 500.0)], method="L-BFGS-B")
        opt_thick = float(res.x[0])
        prev_surf.thickness = opt_thick
        stack._update_accumulated_positions()
        return opt_thick

    @staticmethod
    def optimize_mirror_spatial_position(
        stack: SurfaceStack,
        mirror_index: int,
        target_surface_index: int = -1,
        target_xy: Tuple[float, float] = (0.0, 0.0),
        optimize_tilt: bool = True,
        optimize_distance: bool = True,
        optimize_decenter: bool = False,
        target_spot_rms_weight: float = 1.0,
        centroid_weight: float = 10.0,
        max_iter: int = 50,
        wavelengths: Union[float, List[float]] = 0.5876
    ) -> MirrorPositionResult:
        """
        Multi-degree-of-freedom spatial optimizer for folding mirrors.
        Optimizes distance (pre/post thickness), tilt angles (tilt_x, tilt_y),
        and transverse decenters (decenter_x, decenter_y).
        """
        if mirror_index < 0 or mirror_index >= len(stack.surfaces):
            return MirrorPositionResult(
                success=False, surface_index=mirror_index,
                initial_error_mm=0.0, final_error_mm=0.0,
                initial_rms_um=0.0, final_rms_um=0.0,
                message=f"Invalid mirror index {mirror_index}"
            )

        if target_surface_index == -1 or target_surface_index >= len(stack.surfaces):
            target_surface_index = len(stack.surfaces) - 1

        wls = [wavelengths] if isinstance(wavelengths, (int, float)) else list(wavelengths)
        mirror_surf = stack.surfaces[mirror_index]
        prev_surf = stack.surfaces[mirror_index - 1] if mirror_index > 0 else None

        var_names = []
        x0 = []
        bounds = []

        if optimize_tilt:
            var_names.extend(["tilt_x", "tilt_y"])
            x0.extend([mirror_surf.tilt_x, mirror_surf.tilt_y])
            bounds.extend([(-85.0, 85.0), (-85.0, 85.0)])

        if optimize_distance and prev_surf is not None:
            var_names.append("pre_thickness")
            x0.append(prev_surf.thickness)
            bounds.append((1.0, 500.0))

        if optimize_decenter:
            var_names.extend(["decenter_x", "decenter_y"])
            x0.extend([mirror_surf.decenter_x, mirror_surf.decenter_y])
            bounds.extend([(-50.0, 50.0), (-50.0, 50.0)])

        if not var_names:
            return MirrorPositionResult(
                success=True, surface_index=mirror_index,
                initial_error_mm=0.0, final_error_mm=0.0,
                initial_rms_um=0.0, final_rms_um=0.0,
                message="No parameters selected for optimization"
            )

        def apply_vars(vec):
            idx = 0
            for name in var_names:
                val = float(vec[idx])
                if name == "tilt_x":
                    mirror_surf.tilt_x = val
                elif name == "tilt_y":
                    mirror_surf.tilt_y = val
                elif name == "pre_thickness" and prev_surf is not None:
                    prev_surf.thickness = val
                elif name == "decenter_x":
                    mirror_surf.decenter_x = val
                elif name == "decenter_y":
                    mirror_surf.decenter_y = val
                idx += 1
            stack._update_accumulated_positions()

        def evaluate_metrics():
            opt_sys = OpticalSystem(stack)
            all_x, all_y = [], []
            vign_count = 0
            total_count = 0

            for wl in wls:
                for py in np.linspace(-0.8, 0.8, 5):
                    for px in np.linspace(-0.8, 0.8, 5):
                        if px**2 + py**2 <= 1.0:
                            pos = np.array([px * 0.025, py * 0.025, 0.0])
                            ang = math.asin(0.12)
                            tx = px * ang
                            ty = py * ang
                            tz = math.cos(math.sqrt(tx**2 + ty**2))
                            dvec = np.array([math.sin(tx), math.sin(ty), tz])
                            dvec /= np.linalg.norm(dvec)

                            r = Ray(pos=pos, dir=dvec, wavelength=wl)
                            tr = opt_sys.trace_ray(r)
                            total_count += 1
                            if tr and len(tr.history) > target_surface_index and not tr.vignetted:
                                h = tr.history[target_surface_index]
                                all_x.append(h.get("x_local", h.get("x", 0.0)))
                                all_y.append(h.get("y_local", h.get("y", 0.0)))
                            else:
                                vign_count += 1

            if len(all_x) < 3:
                return float('inf'), float('inf'), 1.0

            mean_x = float(np.mean(all_x))
            mean_y = float(np.mean(all_y))
            cent_err = math.sqrt((mean_x - target_xy[0])**2 + (mean_y - target_xy[1])**2)

            dx = np.array(all_x) - mean_x
            dy = np.array(all_y) - mean_y
            rms_rad = math.sqrt(np.mean(dx**2 + dy**2)) * 1000.0  # um
            vign_ratio = vign_count / total_count if total_count > 0 else 1.0

            return cent_err, rms_rad, vign_ratio

        apply_vars(x0)
        init_cent_err, init_rms_um, _ = evaluate_metrics()

        def objective(vec):
            apply_vars(vec)
            cent_err, rms_um, vign_ratio = evaluate_metrics()
            if math.isinf(cent_err):
                return 1e6
            score = (centroid_weight * cent_err * 1000.0) + (target_spot_rms_weight * rms_um) + (vign_ratio * 1e4)
            return score

        res = minimize(
            objective,
            x0=x0,
            bounds=bounds,
            method="Nelder-Mead",
            options={"maxiter": max_iter * 20, "xatol": 1e-4, "fatol": 1e-4}
        )

        apply_vars(res.x)
        final_cent_err, final_rms_um, _ = evaluate_metrics()

        opt_dict = {}
        for idx, name in enumerate(var_names):
            opt_dict[name] = float(res.x[idx])

        return MirrorPositionResult(
            success=res.success or final_cent_err <= init_cent_err,
            surface_index=mirror_index,
            initial_error_mm=init_cent_err if not math.isinf(init_cent_err) else 999.0,
            final_error_mm=final_cent_err if not math.isinf(final_cent_err) else 999.0,
            initial_rms_um=init_rms_um if not math.isinf(init_rms_um) else 999.0,
            final_rms_um=final_rms_um if not math.isinf(final_rms_um) else 999.0,
            optimized_params=opt_dict,
            iterations=res.nit if hasattr(res, 'nit') else max_iter,
            message=f"Mirror positioned: Centroid error {final_cent_err*1000.0:.2f} um, RMS spot {final_rms_um:.2f} um"
        )
