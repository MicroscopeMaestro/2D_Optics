"""
Zemax-style Spot Diagram Diagnostics (Centroid, RMS Radius, GEO Radius, Airy Disk Circle).
"""
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union
from optics_app.engine.ray import Ray
from optics_app.engine.tracer import OpticalSystem


class SpotDiagram:
    """
    Spot Diagram analyzer for ray hit arrays on detector plane.
    Supports Centroid and Chief-Ray referenced RMS and Geometric spot blurs.
    """
    def __init__(
        self,
        x_coords: Union[List[float], np.ndarray],
        y_coords: Union[List[float], np.ndarray],
        wavelength_um: float = 0.5876,
        f_number: float = 4.0,
        reference: str = "centroid",
        chief_ray_coord: Optional[Tuple[float, float]] = None
    ):
        self.x = np.array(x_coords, dtype=float)
        self.y = np.array(y_coords, dtype=float)
        self.wavelength_um = float(wavelength_um)
        self.f_number = float(f_number)
        self.reference = str(reference).lower()
        self.chief_ray_coord = (
            (float(chief_ray_coord[0]), float(chief_ray_coord[1]))
            if chief_ray_coord is not None else None
        )

    def calculate_centroid(self) -> Tuple[float, float]:
        """Compute centroid (x_bar, y_bar) of spot ray hits."""
        if len(self.x) == 0:
            return 0.0, 0.0
        return float(np.mean(self.x)), float(np.mean(self.y))

    def get_reference_point(self, reference: Optional[str] = None) -> Tuple[float, float]:
        """
        Return the evaluation reference origin (x_ref, y_ref).
        Modes supported:
        - "chief_ray": returns self.chief_ray_coord, falling back to centroid if None.
        - "centroid": returns self.calculate_centroid().
        """
        ref_mode = (reference or self.reference).lower()
        if ref_mode in ("chief_ray", "chiefray", "chief"):
            if self.chief_ray_coord is not None:
                return float(self.chief_ray_coord[0]), float(self.chief_ray_coord[1])
        return self.calculate_centroid()

    def calculate_rms_radius(self, reference: Optional[str] = None) -> float:
        """Compute Total RMS spot radius R_RMS relative to reference point: R_RMS = sqrt(RMS_x^2 + RMS_y^2)."""
        if len(self.x) <= 1:
            return 0.0
        rx, ry = self.get_reference_point(reference)
        r2 = (self.x - rx)**2 + (self.y - ry)**2
        return float(np.sqrt(np.mean(r2)))

    def calculate_rms_x(self, reference: Optional[str] = None) -> float:
        """Compute RMS Spot X size relative to reference point: RMS_x = sqrt(mean((x - x_bar)^2))."""
        if len(self.x) <= 1:
            return 0.0
        rx, _ = self.get_reference_point(reference)
        return float(np.sqrt(np.mean((self.x - rx)**2)))

    def calculate_rms_y(self, reference: Optional[str] = None) -> float:
        """Compute RMS Spot Y size relative to reference point: RMS_y = sqrt(mean((y - y_bar)^2))."""
        if len(self.y) <= 1:
            return 0.0
        _, ry = self.get_reference_point(reference)
        return float(np.sqrt(np.mean((self.y - ry)**2)))

    def calculate_geo_radius(self, reference: Optional[str] = None) -> float:
        """Compute maximum 100% geometric spot radius R_GEO relative to reference point."""
        if len(self.x) == 0:
            return 0.0
        rx, ry = self.get_reference_point(reference)
        r2 = (self.x - rx)**2 + (self.y - ry)**2
        return float(np.max(np.sqrt(r2)))

    def calculate_geo_x(self, reference: Optional[str] = None) -> float:
        """Compute maximum 100% geometric spot X half-width: GEO_x = max(|x - x_bar|)."""
        if len(self.x) == 0:
            return 0.0
        rx, _ = self.get_reference_point(reference)
        return float(np.max(np.abs(self.x - rx)))

    def calculate_geo_y(self, reference: Optional[str] = None) -> float:
        """Compute maximum 100% geometric spot Y half-width: GEO_y = max(|y - y_bar|)."""
        if len(self.y) == 0:
            return 0.0
        _, ry = self.get_reference_point(reference)
        return float(np.max(np.abs(self.y - ry)))

    def calculate_airy_disk_radius(self) -> float:
        """Compute theoretical Airy disk radius r_airy = 1.22 * lambda * F/#."""
        if self.wavelength_um <= 0.0 or self.f_number <= 0.0:
            raise ValueError("Wavelength and F/# must be positive.")
        return 1.22 * self.wavelength_um * self.f_number

    def is_diffraction_limited(self) -> bool:
        """Return True if R_RMS <= r_Airy."""
        return self.calculate_rms_radius() <= self.calculate_airy_disk_radius()


def calculate_spot_metrics(
    x_coords: Union[List[float], np.ndarray],
    y_coords: Union[List[float], np.ndarray],
    wavelength_um: float = 0.5876,
    f_number: float = 4.0,
    reference: str = "centroid",
    chief_ray_coord: Optional[Tuple[float, float]] = None
) -> Dict[str, Any]:
    """Helper function producing comprehensive dictionary of Zemax spot metrics."""
    spot = SpotDiagram(
        x_coords, y_coords,
        wavelength_um=wavelength_um,
        f_number=f_number,
        reference=reference,
        chief_ray_coord=chief_ray_coord
    )
    cx, cy = spot.calculate_centroid()
    rx, ry = spot.get_reference_point()
    return {
        "centroid_x": cx,
        "centroid_y": cy,
        "reference_x": rx,
        "reference_y": ry,
        "reference_type": reference,
        "rms_radius": spot.calculate_rms_radius(),
        "rms_x": spot.calculate_rms_x(),
        "rms_y": spot.calculate_rms_y(),
        "geo_radius": spot.calculate_geo_radius(),
        "geo_x": spot.calculate_geo_x(),
        "geo_y": spot.calculate_geo_y(),
        "airy_radius": spot.calculate_airy_disk_radius(),
        "diffraction_limited": spot.is_diffraction_limited()
    }


def generate_spot_diagram_from_system(
    system: OpticalSystem,
    num_rays: int = 100,
    wavelength_um: float = 0.5876,
    entrance_pupil_radius: float = 12.5,
    f_number: float = 4.0,
    target_surface_index: int = -1,
    source_config: Optional[Any] = None,
    field_point: Optional[Tuple[float, float]] = None,
    field_type: Optional[Any] = None,
    system_config: Optional[Any] = None,
    reference: str = "centroid",
) -> SpotDiagram:
    """
    Generate SpotDiagram object by sampling entrance pupil grid and tracing rays through OpticalSystem.
    Supports Zemax-style multi-field simulation (Object Height mm or Angle deg) with entrance pupil aiming.
    """
    x_hits = []
    y_hits = []

    # If field_point or system_config is provided, use Zemax paraxial entrance pupil solve & aiming
    if field_point is not None or system_config is not None:
        from optics_app.engine.pupil_solve import (
            calculate_entrance_pupil,
            aim_ray_to_pupil,
            aim_collimated_ray,
            generate_pupil_grid,
        )
        from optics_app.engine.system_config import FieldType, RayPatternType

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

        pattern = getattr(system_config, "ray_pattern", RayPatternType.HEXAPOLAR) if system_config is not None else RayPatternType.HEXAPOLAR
        n_rings = max(1, int(round(np.sqrt(num_rays / 3.0))))
        pupil_grid = generate_pupil_grid(
            pattern=pattern,
            rings=n_rings,
            arms=6,
            include_chief_ray=True,
            linear_arms=True,
            max_radius=1.0,
        )
        pupil_pts = [(px, py) for px, py, _ in pupil_grid]

        rays_to_trace = []
        for px, py in pupil_pts:
            is_chief = (abs(px) < 1e-9 and abs(py) < 1e-9)
            if is_collimated and field_type == FieldType.ANGLE_DEG:
                ray = aim_collimated_ray(
                    surfaces=surfaces,
                    field_angle_deg=(float(fx), float(fy)),
                    pupil_norm=(float(px), float(py)),
                    wavelength_um=wavelength_um,
                    z_ep_global=z_ep,
                    r_ep=r_ep,
                )
            else:
                if field_type == FieldType.OBJECT_HEIGHT_MM:
                    obj_x, obj_y = float(fx), float(fy)
                else:
                    obj_x = -d0 * np.tan(np.radians(float(fx))) if d0 > 0 else 0.0
                    obj_y = -d0 * np.tan(np.radians(float(fy))) if d0 > 0 else 0.0

                ray = aim_ray_to_pupil(
                    surfaces=surfaces,
                    obj_point=(obj_x, obj_y, 0.0),
                    pupil_norm=(float(px), float(py)),
                    wavelength_um=wavelength_um,
                    z_ep_global=z_ep,
                    r_ep=r_ep,
                    system=system,
                )
            ray.is_chief = is_chief
            rays_to_trace.append(ray)

    else:
        n_rings = max(1, int(np.sqrt(num_rays / np.pi)) + 1)
        rays_to_trace = []
        
        # Try using SourceConfig
        cfg = source_config
        from optics_app.engine.source import SourceType
        
        if cfg is not None and getattr(cfg, 'source_type', None) == SourceType.FIBER:
            max_angle = np.radians(cfg.divergence_angle)
            chief_r = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wavelength_um)
            chief_r.is_chief = True
            rays_to_trace.append(chief_r)
            spatial_rings = max(1, int(np.sqrt(n_rings)))
            angular_rings = max(1, int(np.sqrt(n_rings)))
            
            for sr in range(1, spatial_rings + 1):
                s_norm = sr / spatial_rings
                s_pts = max(4, int(4 * sr))
                for sp in range(s_pts):
                    stheta = 2.0 * np.pi * sp / s_pts
                    px = s_norm * np.cos(stheta) * cfg.beam_radius
                    py = s_norm * np.sin(stheta) * cfg.beam_radius
                    r_base = Ray(pos=np.array([px, py, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wavelength_um)
                    r_base.is_chief = False
                    rays_to_trace.append(r_base)
                    
                    for ar in range(1, angular_rings + 1):
                        a_norm = ar / angular_rings
                        a_pts = max(4, int(4 * ar))
                        for ap in range(a_pts):
                            atheta = 2.0 * np.pi * ap / a_pts
                            dx = a_norm * np.sin(max_angle) * np.cos(atheta)
                            dy = a_norm * np.sin(max_angle) * np.sin(atheta)
                            dz = np.sqrt(max(0.0, 1.0 - dx**2 - dy**2))
                            r_ang = Ray(pos=np.array([px, py, 0.0]), dir=np.array([dx, dy, dz]), wavelength=wavelength_um)
                            r_ang.is_chief = False
                            rays_to_trace.append(r_ang)
        elif cfg is not None and getattr(cfg, 'source_type', None) == SourceType.POINT:
            max_angle = np.radians(cfg.divergence_angle)
            chief_r = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wavelength_um)
            chief_r.is_chief = True
            rays_to_trace.append(chief_r)
            for ring in range(1, n_rings + 1):
                r_norm = ring / n_rings
                n_pts = max(6, int(6 * ring))
                for p in range(n_pts):
                    theta = 2.0 * np.pi * p / n_pts
                    dx = r_norm * np.sin(max_angle) * np.cos(theta)
                    dy = r_norm * np.sin(max_angle) * np.sin(theta)
                    dz = np.sqrt(max(0.0, 1.0 - dx**2 - dy**2))
                    r_pt = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([dx, dy, dz]), wavelength=wavelength_um)
                    r_pt.is_chief = False
                    rays_to_trace.append(r_pt)
        else:
            # Collimated fallback
            chief_r = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wavelength_um)
            chief_r.is_chief = True
            rays_to_trace.append(chief_r)
            for ring in range(1, n_rings + 1):
                r_norm = ring / n_rings
                n_pts = max(6, int(6 * ring))
                for p in range(n_pts):
                    theta = 2.0 * np.pi * p / n_pts
                    px = r_norm * np.cos(theta)
                    py = r_norm * np.sin(theta)
                    rad = cfg.beam_radius if (cfg is not None and getattr(cfg, 'source_type', None) == SourceType.COLLIMATED) else entrance_pupil_radius
                    ray_x = px * rad
                    ray_y = py * rad
                    r_col = Ray(pos=np.array([ray_x, ray_y, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wavelength_um)
                    r_col.is_chief = False
                    rays_to_trace.append(r_col)

    traced_rays = system.trace_rays(rays_to_trace)
    chief_coord = None
    for r in traced_rays:
        if len(r.history) > 0:
            idx = target_surface_index if target_surface_index != -1 else len(r.history) - 1
            if idx < len(r.history):
                if r.vignetted and idx >= len(r.history) - 1:
                    continue
                hit = r.history[idx]
                x_hits.append(hit["x"])
                y_hits.append(hit["y"])
                if getattr(r, "is_chief", False) and chief_coord is None:
                    chief_coord = (float(hit["x"]), float(hit["y"]))

    return SpotDiagram(
        x_hits,
        y_hits,
        wavelength_um=wavelength_um,
        f_number=f_number,
        reference=reference,
        chief_ray_coord=chief_coord
    )
