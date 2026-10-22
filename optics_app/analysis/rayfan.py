"""
Zemax-style Ray Fan Plot Analysis (EX, EY vs Py, Px) and Focal Shift Evaluation.
"""
from dataclasses import dataclass
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
from optics_app.engine.ray import Ray
from optics_app.engine.tracer import OpticalSystem


@dataclass
class RayFanData:
    """Dataclass holding ray fan curves and pupil coordinates."""
    py: np.ndarray                  # Normalized tangential pupil coords P_y [-1, 1]
    tangential_ex: np.ndarray       # EX(P_y) mm
    tangential_ey: np.ndarray       # EY(P_y) mm
    px: np.ndarray                  # Normalized sagittal pupil coords P_x [-1, 1]
    sagittal_ex: np.ndarray         # EX(P_x) mm
    sagittal_ey: np.ndarray         # EY(P_x) mm
    vignetted_tangential: np.ndarray# Boolean mask for vignetted rays
    vignetted_sagittal: np.ndarray  # Boolean mask for vignetted rays
    focal_shift_mm: float = 0.0     # Applied Z-defocus shift


class RayFanAnalysis:
    """
    Ray Fan Plot Analyzer for synthetic wavefronts and real OpticalSystem ray tracing.
    """
    def __init__(self, py_sampling: int = 51, px_sampling: int = 51):
        if py_sampling <= 1:
            self.py = np.array([0.0], dtype=float)
        else:
            self.py = np.linspace(-1.0, 1.0, py_sampling, dtype=float)

        if px_sampling <= 1:
            self.px = np.array([0.0], dtype=float)
        else:
            self.px = np.linspace(-1.0, 1.0, px_sampling, dtype=float)

    def generate_tangential_fan(
        self,
        W040: float = 0.0,
        W131: float = 0.0,
        W222: float = 0.0,
        W220: float = 0.0,
        W311: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate synthetic tangential ray fan EY(Py):
        EY(Py) = 4 * W040 * Py^3 + 3 * W131 * Py^2 + 2 * (W222 + W220) * Py + W311
        Returns: (py, ex, ey)
        """
        ey = (4.0 * W040 * self.py**3 +
              3.0 * W131 * self.py**2 +
              2.0 * (W222 + W220) * self.py +
              W311)
        ex = np.zeros_like(self.py)
        return self.py, ex, ey

    def generate_sagittal_fan(
        self,
        W040: float = 0.0,
        W222: float = 0.0,
        W220: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate synthetic sagittal ray fan EX(Px):
        EX(Px) = 4 * W040 * Px^3 + 2 * W220 * Px
        Returns: (px, ex, ey)
        """
        ex = 4.0 * W040 * self.px**3 + 2.0 * W220 * self.px
        ey = np.zeros_like(self.px)
        return self.px, ex, ey

    def trace_ray_fan_from_system(
        self,
        system: OpticalSystem,
        entrance_pupil_radius: float = 12.5,
        wavelength_um: float = 0.5876,
        focal_shift_mm: float = 0.0,
        target_surface_index: int = -1,
        source_config: Optional[Any] = None,
        field_point: Optional[Tuple[float, float]] = None,
        field_type: Optional[Any] = None,
        system_config: Optional[Any] = None,
    ) -> RayFanData:
        """
        Trace actual rays through OpticalSystem for tangential and sagittal fans.
        Supports Zemax-style off-axis multi-field ray fans relative to chief ray.
        """
        if field_point is not None or system_config is not None:
            from optics_app.engine.pupil_solve import calculate_entrance_pupil, aim_ray_to_pupil, aim_collimated_ray
            from optics_app.engine.system_config import FieldType

            surfaces = system.surfaces
            d0 = float(getattr(surfaces[0], "thickness", 0.0)) if len(surfaces) > 0 else 0.0

            is_collimated = False
            if system_config is not None:
                btype = getattr(system_config, "beam_type", "")
                is_collimated = (btype == "Collimated (Parallel Beam)") or getattr(system_config, "telecentric_object_space", False) or (d0 > 1e4)
                if field_type is None:
                    field_type = getattr(system_config, "field_type", FieldType.ANGLE_DEG)
            else:
                is_collimated = (d0 > 1e4 or d0 <= 0.0)

            if field_type is None:
                field_type = FieldType.ANGLE_DEG if is_collimated else FieldType.OBJECT_HEIGHT_MM

            fx, fy = field_point if field_point is not None else (0.0, 0.0)

            z_ep, r_ep, _ = calculate_entrance_pupil(
                surfaces,
                wavelength_um=wavelength_um,
                default_epd=2.0 * entrance_pupil_radius,
            )

            if field_type == FieldType.OBJECT_HEIGHT_MM:
                obj_x, obj_y = float(fx), float(fy)
            else:
                obj_x = -d0 * np.tan(np.radians(float(fx))) if d0 > 0 else 0.0
                obj_y = -d0 * np.tan(np.radians(float(fy))) if d0 > 0 else 0.0

            # 1. Chief Ray (P_x = 0, P_y = 0)
            if is_collimated and field_type == FieldType.ANGLE_DEG:
                chief_ray = aim_collimated_ray(
                    surfaces=surfaces,
                    field_angle_deg=(float(fx), float(fy)),
                    pupil_norm=(0.0, 0.0),
                    wavelength_um=wavelength_um,
                    z_ep_global=z_ep,
                    r_ep=r_ep,
                )
            else:
                chief_ray = aim_ray_to_pupil(
                    surfaces=surfaces,
                    obj_point=(obj_x, obj_y, 0.0),
                    pupil_norm=(0.0, 0.0),
                    wavelength_um=wavelength_um,
                    z_ep_global=z_ep,
                    r_ep=r_ep,
                    system=system,
                )
            traced_chief = system.trace_ray(chief_ray)
            x_chief, y_chief = 0.0, 0.0
            if len(traced_chief.history) > 0:
                idx = target_surface_index if target_surface_index != -1 else len(traced_chief.history) - 1
                if idx < len(traced_chief.history) and not (traced_chief.vignetted and idx >= len(traced_chief.history) - 1):
                    last_hit = traced_chief.history[idx]
                    x_chief_raw = last_hit["x"]
                    y_chief_raw = last_hit["y"]
                    dir_chief = traced_chief.dir
                    dz_ratio_x = (dir_chief[0] / dir_chief[2]) if abs(dir_chief[2]) > 1e-12 else 0.0
                    dz_ratio_y = (dir_chief[1] / dir_chief[2]) if abs(dir_chief[2]) > 1e-12 else 0.0
                    x_chief = x_chief_raw + focal_shift_mm * dz_ratio_x
                    y_chief = y_chief_raw + focal_shift_mm * dz_ratio_y

            # 2. Tangential Fan Rays (P_x = 0, P_y in py)
            tang_ex = np.zeros_like(self.py)
            tang_ey = np.zeros_like(self.py)
            vign_tang = np.zeros_like(self.py, dtype=bool)

            tang_rays = []
            for py_val in self.py:
                if is_collimated and field_type == FieldType.ANGLE_DEG:
                    r = aim_collimated_ray(
                        surfaces=surfaces,
                        field_angle_deg=(float(fx), float(fy)),
                        pupil_norm=(0.0, float(py_val)),
                        wavelength_um=wavelength_um,
                        z_ep_global=z_ep,
                        r_ep=r_ep,
                    )
                else:
                    r = aim_ray_to_pupil(
                        surfaces=surfaces,
                        obj_point=(obj_x, obj_y, 0.0),
                        pupil_norm=(0.0, float(py_val)),
                        wavelength_um=wavelength_um,
                        z_ep_global=z_ep,
                        r_ep=r_ep,
                        system=system,
                    )
                tang_rays.append(r)

            traced_tang = system.trace_rays(tang_rays)
            for i, r in enumerate(traced_tang):
                if len(r.history) == 0:
                    vign_tang[i] = True
                    tang_ex[i] = np.nan
                    tang_ey[i] = np.nan
                    continue

                idx = target_surface_index if target_surface_index != -1 else len(r.history) - 1
                if idx >= len(r.history) or (r.vignetted and idx >= len(r.history) - 1):
                    vign_tang[i] = True
                    tang_ex[i] = np.nan
                    tang_ey[i] = np.nan
                else:
                    last_hit = r.history[idx]
                    x_raw, y_raw = last_hit["x"], last_hit["y"]
                    rd = r.dir
                    dz_ratio_x = (rd[0] / rd[2]) if abs(rd[2]) > 1e-12 else 0.0
                    dz_ratio_y = (rd[1] / rd[2]) if abs(rd[2]) > 1e-12 else 0.0
                    xf = x_raw + focal_shift_mm * dz_ratio_x
                    yf = y_raw + focal_shift_mm * dz_ratio_y
                    tang_ex[i] = xf - x_chief
                    tang_ey[i] = yf - y_chief

            # 3. Sagittal Fan Rays (P_y = 0, P_x in px)
            sag_ex = np.zeros_like(self.px)
            sag_ey = np.zeros_like(self.px)
            vign_sag = np.zeros_like(self.px, dtype=bool)

            sag_rays = []
            for px_val in self.px:
                if is_collimated and field_type == FieldType.ANGLE_DEG:
                    r = aim_collimated_ray(
                        surfaces=surfaces,
                        field_angle_deg=(float(fx), float(fy)),
                        pupil_norm=(float(px_val), 0.0),
                        wavelength_um=wavelength_um,
                        z_ep_global=z_ep,
                        r_ep=r_ep,
                    )
                else:
                    r = aim_ray_to_pupil(
                        surfaces=surfaces,
                        obj_point=(obj_x, obj_y, 0.0),
                        pupil_norm=(float(px_val), 0.0),
                        wavelength_um=wavelength_um,
                        z_ep_global=z_ep,
                        r_ep=r_ep,
                        system=system,
                    )
                sag_rays.append(r)

            traced_sag = system.trace_rays(sag_rays)
            for i, r in enumerate(traced_sag):
                if len(r.history) == 0:
                    vign_sag[i] = True
                    sag_ex[i] = np.nan
                    sag_ey[i] = np.nan
                    continue

                idx = target_surface_index if target_surface_index != -1 else len(r.history) - 1
                if idx >= len(r.history) or (r.vignetted and idx >= len(r.history) - 1):
                    vign_sag[i] = True
                    sag_ex[i] = np.nan
                    sag_ey[i] = np.nan
                else:
                    last_hit = r.history[idx]
                    v_dir = r.dir
                    dz_rx = (v_dir[0] / v_dir[2]) if abs(v_dir[2]) > 1e-12 else 0.0
                    dz_ry = (v_dir[1] / v_dir[2]) if abs(v_dir[2]) > 1e-12 else 0.0
                    dx = focal_shift_mm * dz_rx
                    dy = focal_shift_mm * dz_ry
                    sag_ex[i] = (last_hit["x"] + dx) - x_chief
                    sag_ey[i] = (last_hit["y"] + dy) - y_chief

            return RayFanData(
                py=self.py.copy(),
                tangential_ex=tang_ex,
                tangential_ey=tang_ey,
                px=self.px.copy(),
                sagittal_ex=sag_ex,
                sagittal_ey=sag_ey,
                vignetted_tangential=vign_tang,
                vignetted_sagittal=vign_sag,
                focal_shift_mm=focal_shift_mm
            )

        cfg = source_config
        from optics_app.engine.source import SourceType
        
        is_angle_based = cfg is not None and getattr(cfg, 'source_type', None) in (SourceType.FIBER, SourceType.POINT, SourceType.GAUSSIAN)
        max_angle = np.radians(getattr(cfg, 'divergence_angle', 0.0)) if is_angle_based else 0.0
        rad = getattr(cfg, 'beam_radius', entrance_pupil_radius) if cfg else entrance_pupil_radius

        # 1. Chief Ray (P_x = 0, P_y = 0)
        chief_ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wavelength_um)
        traced_chief = system.trace_ray(chief_ray)
        x_chief, y_chief = 0.0, 0.0
        if len(traced_chief.history) > 0:
            idx = target_surface_index if target_surface_index != -1 else len(traced_chief.history) - 1
            if idx < len(traced_chief.history) and not (traced_chief.vignetted and idx >= len(traced_chief.history) - 1):
                last_hit = traced_chief.history[idx]
                x_chief_raw = last_hit["x"]
                y_chief_raw = last_hit["y"]
                dir_chief = traced_chief.dir
                dz_ratio_x = (dir_chief[0] / dir_chief[2]) if abs(dir_chief[2]) > 1e-12 else 0.0
                dz_ratio_y = (dir_chief[1] / dir_chief[2]) if abs(dir_chief[2]) > 1e-12 else 0.0
                x_chief = x_chief_raw + focal_shift_mm * dz_ratio_x
                y_chief = y_chief_raw + focal_shift_mm * dz_ratio_y

        # 2. Tangential Fan Rays (P_x = 0, P_y in py)
        tang_ex = np.zeros_like(self.py)
        tang_ey = np.zeros_like(self.py)
        vign_tang = np.zeros_like(self.py, dtype=bool)

        tang_rays = []
        for py_val in self.py:
            if is_angle_based:
                ang = py_val * max_angle
                tang_rays.append(Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, np.sin(ang), np.cos(ang)]), wavelength=wavelength_um))
            else:
                tang_rays.append(Ray(pos=np.array([0.0, py_val * rad, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wavelength_um))
        traced_tang = system.trace_rays(tang_rays)
        for i, r in enumerate(traced_tang):
            if len(r.history) == 0:
                vign_tang[i] = True
                tang_ex[i] = np.nan
                tang_ey[i] = np.nan
                continue
                
            idx = target_surface_index if target_surface_index != -1 else len(r.history) - 1
            if idx >= len(r.history) or (r.vignetted and idx >= len(r.history) - 1):
                vign_tang[i] = True
                tang_ex[i] = np.nan
                tang_ey[i] = np.nan
            else:
                last_hit = r.history[idx]
                x_raw, y_raw = last_hit["x"], last_hit["y"]
                rd = r.dir
                dz_ratio_x = (rd[0] / rd[2]) if abs(rd[2]) > 1e-12 else 0.0
                dz_ratio_y = (rd[1] / rd[2]) if abs(rd[2]) > 1e-12 else 0.0
                xf = x_raw + focal_shift_mm * dz_ratio_x
                yf = y_raw + focal_shift_mm * dz_ratio_y
                tang_ex[i] = xf - x_chief
                tang_ey[i] = yf - y_chief

        # 3. Sagittal Fan Rays (P_y = 0, P_x in px)
        sag_ex = np.zeros_like(self.px)
        sag_ey = np.zeros_like(self.px)
        vign_sag = np.zeros_like(self.px, dtype=bool)

        sag_rays = []
        for px_val in self.px:
            if is_angle_based:
                ang = px_val * max_angle
                sag_rays.append(Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([np.sin(ang), 0.0, np.cos(ang)]), wavelength=wavelength_um))
            else:
                sag_rays.append(Ray(pos=np.array([px_val * rad, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wavelength_um))
        traced_sag = system.trace_rays(sag_rays)
        for i, r in enumerate(traced_sag):
            if len(r.history) == 0:
                vign_sag[i] = True
                sag_ex[i] = np.nan
                sag_ey[i] = np.nan
                continue
                
            idx = target_surface_index if target_surface_index != -1 else len(r.history) - 1
            if idx >= len(r.history) or (r.vignetted and idx >= len(r.history) - 1):
                vign_sag[i] = True
                sag_ex[i] = np.nan
                sag_ey[i] = np.nan
            else:
                last_hit = r.history[idx]
                v_dir = r.dir
                dz_rx = (v_dir[0] / v_dir[2]) if abs(v_dir[2]) > 1e-12 else 0.0
                dz_ry = (v_dir[1] / v_dir[2]) if abs(v_dir[2]) > 1e-12 else 0.0
                dx = focal_shift_mm * dz_rx
                dy = focal_shift_mm * dz_ry
                sag_ex[i] = (last_hit["x"] + dx) - x_chief
                sag_ey[i] = (last_hit["y"] + dy) - y_chief

        return RayFanData(
            py=self.py.copy(),
            tangential_ex=tang_ex,
            tangential_ey=tang_ey,
            px=self.px.copy(),
            sagittal_ex=sag_ex,
            sagittal_ey=sag_ey,
            vignetted_tangential=vign_tang,
            vignetted_sagittal=vign_sag,
            focal_shift_mm=focal_shift_mm
        )


def calculate_ray_fan(system: Any, num_rays: int = 51) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Helper function for backwards compatibility with tests and GUI tabs."""
    analyzer = RayFanAnalysis(py_sampling=num_rays)
    if hasattr(system, "stack") or hasattr(system, "trace_ray") or hasattr(system, "surfaces"):
        # If it's a SurfaceStack, wrap in OpticalSystem
        from optics_app.engine.lde import SurfaceStack
        if isinstance(system, SurfaceStack):
            system = OpticalSystem(system)
        fan_data = analyzer.trace_ray_fan_from_system(system)
        return fan_data.py, fan_data.tangential_ex, fan_data.tangential_ey
    return analyzer.generate_tangential_fan()
