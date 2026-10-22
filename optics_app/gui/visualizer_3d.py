import sys
import numpy as np
from typing import Optional, List, Any, Tuple, Dict, Union

from optics_app.gui.qt_compat import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QCheckBox, QLabel,
    QComboBox, QDoubleSpinBox, QFrame, QtWidgets, QT_AVAILABLE, Qt, pyqtSignal, DummySignal, QTimer
)
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.ray import Ray
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.element_colors import (
    compute_surface_groups, is_object_plane, is_detector_plane, is_stop_plane,
    LENS_ELEMENT_PALETTES
)
try:
    from optics_app.engine.laser_scanner import is_galvo
except ImportError:
    from v2.engine.laser_scanner import is_galvo

try:
    from optics_app.engine.mechanisms import (
        is_dmd, available_states, active_state_name, active_state, get_effective_pose
    )
except ImportError:
    from v2.engine.mechanisms import (
        is_dmd, available_states, active_state_name, active_state, get_effective_pose
    )

HAS_MPL_QT = False
try:
    import matplotlib
    matplotlib.use('QtAgg' if QT_AVAILABLE else 'Agg')
    from matplotlib.figure import Figure
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
    if QT_AVAILABLE:
        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
            from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
            HAS_MPL_QT = True
        except ImportError:
            try:
                from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
                from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
                HAS_MPL_QT = True
            except ImportError:
                HAS_MPL_QT = False
    else:
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        NavigationToolbar = None
except ImportError:
    HAS_MPL_QT = False


def wavelength_to_color(wavelength_um: float) -> str:
    """Map optical wavelength in microns to realistic hex color code."""
    wl = float(wavelength_um)
    if wl < 0.440:
        return "#3f51b5"  # Violet-Blue (e.g. 405 nm laser)
    elif wl < 0.490:
        return "#1e88e5"  # Royal Blue (e.g. 450 nm / 486.1 nm F-line)
    elif wl < 0.520:
        return "#00acc1"  # Cyan (e.g. 488 nm / 514 nm Ar-ion)
    elif wl < 0.565:
        return "#43a047"  # Green (e.g. 532 nm Nd:YAG / 546.1 nm e-line)
    elif wl < 0.595:
        return "#f57c00"  # Amber / Yellow (e.g. 587.6 nm d-line / 589 nm Na)
    elif wl < 0.690:
        return "#e53935"  # Bright Red (e.g. 632.8 nm He-Ne / 656.3 nm C-line)
    else:
        return "#8e24aa"  # Deep Crimson / NIR (785 nm, 830 nm, 1064 nm)


def diffraction_wavelength_colors(wavelengths: List[float]) -> Dict[float, str]:
    """Distinct, explicitly false colors for any multi-wavelength ray bundle."""
    values = sorted({float(w) for w in wavelengths if np.isfinite(w) and w > 0})
    if len(values) < 2:
        return {w: wavelength_to_color(w) for w in values}
    stops = ((37, 99, 235), (245, 158, 11), (220, 38, 38))
    result = {}
    span = values[-1] - values[0]
    for wavelength in values:
        position = (wavelength - values[0]) / span
        section = min(int(position * 2), 1)
        fraction = position * 2 - section
        rgb = tuple(round(a + fraction * (b - a)) for a, b in zip(stops[section], stops[section + 1]))
        result[wavelength] = "#{:02x}{:02x}{:02x}".format(*rgb)
    return result


def collect_grating_dispersion_samples(surfaces: List[Surface], rays: List[Ray]) -> List[Dict[str, Any]]:
    """Select a near-chief traced ray per wavelength and grating for 3D fan annotations."""
    if hasattr(rays, "all_rays"):
        rays = rays.all_rays()
    grating_indices = {i for i, s in enumerate(surfaces) if getattr(s, 'surface_type', None) == SurfaceType.GRATING and getattr(s, 'grating_lines_per_mm', 0) > 0}
    if not grating_indices:
        return []
    samples: Dict[Tuple[int, float], Tuple[float, Dict[str, Any]]] = {}
    for ray in rays:
        wavelength = float(getattr(ray, "wavelength", 0.0))
        if not np.isfinite(wavelength) or wavelength <= 0:
            continue
        history = getattr(ray, "history", [])
        for hit_index, hit in enumerate(history):
            index = int(hit.get("surface_index", -1))
            if index not in grating_indices or hit.get("vignetted"):
                continue
            surface = surfaces[index]
            point = np.array([hit.get("x_global", 0.0), hit.get("y_global", 0.0), hit.get("z_global", 0.0)], dtype=float)
            direction = np.asarray(hit.get("dir", [0.0, 0.0, 0.0]), dtype=float)
            norm = float(np.linalg.norm(direction))
            if not np.all(np.isfinite(point)) or not np.all(np.isfinite(direction)) or norm <= 0:
                continue
            direction = direction / norm
            rotation = np.asarray(surface.global_R, dtype=float)
            normal = rotation[:, 2].copy()
            if np.dot(direction, normal) < 0:
                normal = -normal
            local_disp = np.array([0.0, 1.0, 0.0]) if abs(surface.tilt_x) > abs(surface.tilt_y) else np.array([1.0, 0.0, 0.0])
            dispersion_axis = rotation @ local_disp
            dispersion_axis -= np.dot(dispersion_axis, normal) * normal
            dispersion_axis /= max(float(np.linalg.norm(dispersion_axis)), 1e-12)
            angle_deg = float(np.degrees(np.arctan2(np.dot(direction, dispersion_axis), np.dot(direction, normal))))
            incident_angle_deg = None
            if hit_index > 0:
                incident = np.asarray(history[hit_index - 1].get("dir", [0.0, 0.0, 0.0]), dtype=float)
                incident_norm = float(np.linalg.norm(incident))
                if np.all(np.isfinite(incident)) and incident_norm > 0:
                    incident /= incident_norm
                    incident_angle_deg = float(np.degrees(np.arctan2(
                        -np.dot(incident, dispersion_axis), -np.dot(incident, normal))))
            next_distance = 8.0
            if hit_index + 1 < len(history):
                next_hit = history[hit_index + 1]
                next_point = np.array([next_hit.get("x_global", 0.0), next_hit.get("y_global", 0.0), next_hit.get("z_global", 0.0)], dtype=float)
                next_distance = float(np.linalg.norm(next_point - point))
            sample = {
                "surface_index": index, "wavelength_um": wavelength,
                "point": point, "direction": direction, "angle_deg": angle_deg,
                "incident_angle_deg": incident_angle_deg,
                "arrow_length_mm": min(20.0, max(0.5, next_distance * 0.6)),
                "next_distance_mm": next_distance,
                "order": int(surface.grating_order),
                "lines_per_mm": float(surface.grating_lines_per_mm),
            }
            score = float(hit.get("x_local", 0.0)) ** 2 + float(hit.get("y_local", 0.0)) ** 2
            key = (index, wavelength)
            if key not in samples or score < samples[key][0]:
                samples[key] = (score, sample)
    return [samples[key][1] for key in sorted(samples)]


FIELD_COLORS = ['#2563eb', '#16a34a', '#dc2626', '#9333ea', '#ea580c', '#0891b2', '#ca8a04', '#e11d48']


def classify_surface_for_3d(surface: Surface, index: int, total: int) -> Dict[str, bool]:
    """Assign visual roles from component data, never from first/last position alone."""
    name = str(getattr(surface, "name", "") or "").upper()
    kind = getattr(surface, "surface_type", None)
    grating = kind == SurfaceType.GRATING or float(getattr(surface, "grating_lines_per_mm", 0.0) or 0.0) > 0
    dmd = is_dmd(surface)
    stop = is_stop_plane(surface)
    reflective = getattr(surface, "is_reflective", False)
    mirror = (kind == SurfaceType.MIRROR
              or bool(reflective() if callable(reflective) else reflective)
              or str(getattr(surface, "glass", "")).upper() in ("MIRROR", "MIRR", "REFL"))
    detector = (kind == SurfaceType.DETECTOR
                or (not (grating or dmd or mirror or stop)
                    and (is_detector_plane(surface, index, total)
                         or any(token in name for token in ("CCD", "PMT", "PHOTODETECTOR", "SENSOR")))))
    obj = (kind == SurfaceType.OBJECT or is_object_plane(surface, index)
           or (index == 0 and (name == "SAMPLE" or name.startswith("SAMPLE_"))))
    obj = obj and not (grating or dmd or detector or stop or mirror)
    return {"object": obj, "stop": stop, "detector": detector,
            "mirror": mirror, "grating": grating, "dmd": dmd}


def _surface_sag_values(surf: Surface, radii: np.ndarray) -> np.ndarray:
    """Evaluate the standard surface prescription on a grid without Python loops."""
    if type(surf).sag is not Surface.sag or "sag" in getattr(surf, "__dict__", {}):
        return np.vectorize(surf.sag, otypes=[float])(radii)
    radius = float(surf.radius)
    r2 = radii * radii
    if radius == 0.0 or np.isinf(radius):
        z = np.zeros_like(radii, dtype=float)
    else:
        curvature = 1.0 / radius
        argument = 1.0 - (1.0 + float(surf.conic)) * curvature**2 * r2
        z = curvature * r2 / (1.0 + np.sqrt(np.maximum(0.0, argument)))
    for index, coefficient in enumerate(surf.asphere_coeffs):
        z = z + float(coefficient) * radii ** (2 * index + 4)
    return z


class MeshCache:
    """
    Cache for 3D surface geometry meshes keyed by surface geometry hashes
    (radius, conic, aperture, type, origin, R_mat, samples).
    Reuses cached 3D surface meshes instead of recreating polygon vertices from scratch.
    """
    def __init__(self, maxsize: int = 512):
        self._cache: Dict[tuple, Any] = {}
        self.maxsize = maxsize
        self.hits: int = 0
        self.misses: int = 0

    @staticmethod
    def compute_key(
        surf: Surface,
        radial_samples: int,
        angular_samples: int,
        origin: Optional[np.ndarray] = None,
        R_mat: Optional[np.ndarray] = None,
        extra: str = "mesh",
    ) -> tuple:
        if origin is None or R_mat is None:
            eff_origin, eff_R = get_effective_pose(surf)
            origin = origin if origin is not None else eff_origin
            R_mat = R_mat if R_mat is not None else eff_R

        orig_t = tuple(np.round(np.asarray(origin, dtype=float).ravel(), decimals=6))
        r_mat_t = tuple(np.round(np.asarray(R_mat, dtype=float).ravel(), decimals=6))
        asph_t = tuple(np.round(getattr(surf, 'asphere_coeffs', []) or [], decimals=6))
        ap_type = str(getattr(surf, 'aperture_type', '')).lower()
        sub_shape = str(getattr(surf, 'substrate_shape', '')).lower()
        stype = str(getattr(surf, 'surface_type', ''))
        radius = round(float(getattr(surf, 'radius', 0.0)), 6)
        conic = round(float(getattr(surf, 'conic', 0.0)), 6)
        semi_d = round(float(getattr(surf, 'semi_diameter', 0.0)), 6)
        hx = round(float(getattr(surf, 'semi_diameter_x', 0.0)), 6)
        hy = round(float(getattr(surf, 'semi_diameter_y', 0.0)), 6)

        return (
            extra, radius, conic, asph_t, ap_type, sub_shape, semi_d, hx, hy,
            stype, radial_samples, angular_samples, orig_t, r_mat_t
        )

    def get(self, key: tuple) -> Optional[Any]:
        if key in self._cache:
            self.hits += 1
            return self._cache[key]
        self.misses += 1
        return None

    def set(self, key: tuple, value: Any) -> None:
        if len(self._cache) >= self.maxsize:
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = value

    def clear(self) -> None:
        self._cache.clear()
        self.hits = 0
        self.misses = 0

    def get_stats(self) -> Dict[str, int]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "entries": len(self._cache),
            "maxsize": self.maxsize,
        }

    put = set
    make_key = compute_key

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, key: tuple) -> bool:
        return key in self._cache


GLOBAL_MESH_CACHE = MeshCache()


def _surface_mesh(surf: Surface, radial_samples: int = 6, angular_samples: int = 32, cache: Optional[MeshCache] = None):
    """Return a transformed optical face mesh in the application's XYZ frame with MeshCache acceleration."""
    mesh_cache = cache or GLOBAL_MESH_CACHE
    cache_key = mesh_cache.compute_key(surf, radial_samples, angular_samples, extra="mesh")
    cached = mesh_cache.get(cache_key)
    if cached is not None:
        return cached

    ap_type = str(getattr(surf, 'aperture_type', '')).lower()
    sub_shape = str(getattr(surf, 'substrate_shape', '')).lower()
    
    # If sub_shape is auto, infer it from the aperture type (if it's rectangular, make the substrate rectangular)
    if sub_shape == 'auto':
        sub_shape = 'rectangular' if ap_type in ('rectangular', 'slit') else 'circular'

    if sub_shape == 'rectangular':
        hx = float(getattr(surf, "aperture_x_half", 0.0) or getattr(surf, "semi_diameter_x", 0.0) or getattr(surf, "semi_diameter", 0.0) or 0.1)
        hy = float(getattr(surf, "aperture_y_half", 0.0) or getattr(surf, "semi_diameter_y", 0.0) or getattr(surf, "semi_diameter", 0.0) or 0.1)
        if hx <= 0.0:
            sd = max(float(getattr(surf, "semi_diameter", 0.0)), 0.1)
            hx = sd
        if hy <= 0.0:
            sd = max(float(getattr(surf, "semi_diameter", 0.0)), 0.1)
            hy = sd
        x_lin = np.linspace(-hx, hx, angular_samples)
        y_lin = np.linspace(-hy, hy, radial_samples)
        x_local, y_local = np.meshgrid(x_lin, y_lin)
        radius_grid = np.sqrt(x_local**2 + y_local**2)
        z_local = _surface_sag_values(surf, radius_grid)
    else:
        sd = max(float(getattr(surf, "semi_diameter", 0.0)), 0.1)
        theta = np.linspace(0.0, 2.0 * np.pi, angular_samples)
        radii = np.linspace(0.0, sd, radial_samples)
        theta_grid, radius_grid = np.meshgrid(theta, radii)
        x_local = radius_grid * np.cos(theta_grid)
        y_local = radius_grid * np.sin(theta_grid)
        z_local = _surface_sag_values(surf, radius_grid)

    center, rotation = get_effective_pose(surf)
    shape = x_local.shape
    local = np.vstack((x_local.ravel(), y_local.ravel(), z_local.ravel()))
    world = rotation @ local + center.reshape(3, 1)
    res = tuple(world[i].reshape(shape) for i in range(3))
    mesh_cache.set(cache_key, res)
    return res


def _edge_ring(surf: Surface, angular_samples: int = 32, cache: Optional[MeshCache] = None):
    """Return the transformed clear-aperture edge of an optical face with MeshCache acceleration."""
    mesh_cache = cache or GLOBAL_MESH_CACHE
    cache_key = mesh_cache.compute_key(surf, 0, angular_samples, extra="ring")
    cached = mesh_cache.get(cache_key)
    if cached is not None:
        return cached

    ap_type = str(getattr(surf, 'aperture_type', '')).lower()
    sub_shape = str(getattr(surf, 'substrate_shape', '')).lower()
    
    if sub_shape == 'auto':
        sub_shape = 'rectangular' if ap_type in ('rectangular', 'slit') else 'circular'

    if sub_shape == 'rectangular':
        hx = float(getattr(surf, "aperture_x_half", 0.0) or getattr(surf, "semi_diameter_x", 0.0) or getattr(surf, "semi_diameter", 0.0) or 0.1)
        hy = float(getattr(surf, "aperture_y_half", 0.0) or getattr(surf, "semi_diameter_y", 0.0) or getattr(surf, "semi_diameter", 0.0) or 0.1)
        if hx <= 0.0:
            sd = max(float(getattr(surf, "semi_diameter", 0.0)), 0.1)
            hx = sd
        if hy <= 0.0:
            sd = max(float(getattr(surf, "semi_diameter", 0.0)), 0.1)
            hy = sd
        # Rectangular perimeter path
        n_side = max(angular_samples // 4, 2)
        x_top = np.linspace(-hx, hx, n_side)
        y_top = np.full_like(x_top, hy)
        y_right = np.linspace(hy, -hy, n_side)
        x_right = np.full_like(y_right, hx)
        x_bot = np.linspace(hx, -hx, n_side)
        y_bot = np.full_like(x_bot, -hy)
        y_left = np.linspace(-hy, hy, n_side)
        x_left = np.full_like(y_left, -hx)
        x_local = np.concatenate([x_top, x_right[1:], x_bot[1:], x_left[1:]])
        y_local = np.concatenate([y_top, y_right[1:], y_bot[1:], y_left[1:]])
        # If needed, resample to match angular_samples exactly to avoid mismatches
        # But for now let's ensure we return the perimeter.
        if len(x_local) != angular_samples:
            # Interpolate to exactly match angular_samples
            t_orig = np.linspace(0, 1, len(x_local))
            t_new = np.linspace(0, 1, angular_samples)
            x_local = np.interp(t_new, t_orig, x_local)
            y_local = np.interp(t_new, t_orig, y_local)
            
        r_local = np.sqrt(x_local**2 + y_local**2)
        z_local = _surface_sag_values(surf, r_local)
        local = np.vstack((x_local, y_local, z_local))
    else:
        sd = max(float(getattr(surf, "semi_diameter", 0.0)), 0.1)
        theta = np.linspace(0.0, 2.0 * np.pi, angular_samples)
        local = np.vstack((
            sd * np.cos(theta),
            sd * np.sin(theta),
            np.full_like(theta, float(surf.sag(sd))),
        ))
    center, rotation = get_effective_pose(surf)
    world = rotation @ local + center.reshape(3, 1)
    res = (world[0], world[1], world[2])
    mesh_cache.set(cache_key, res)
    return res



def build_cylinder_geometry(
    surf: Surface,
    radius: float,
    thickness: float,
    radial_samples: int = 12,
    angular_samples: int = 32,
    rear_surf: Optional[Surface] = None,
):
    import numpy as np
    theta = np.linspace(0, 2 * np.pi, angular_samples)
    r = np.linspace(0, radius, radial_samples)
    T, R = np.meshgrid(theta, r)
    X = R * np.cos(T)
    Y = R * np.sin(T)
    
    # Local Z coordinates:
    # Front face (z=0, or surf.sag(r) if curved)
    has_sag_front = (
        hasattr(surf, 'sag') and callable(surf.sag)
        and getattr(surf, 'radius', 0.0) != 0.0
        and not np.isinf(getattr(surf, 'radius', 0.0))
    )
    if has_sag_front:
        z_f_1d = np.array([float(surf.sag(rad)) for rad in r])
        Z_front = np.tile(z_f_1d.reshape(-1, 1), (1, angular_samples))
        z_f_edge = float(z_f_1d[-1])
    else:
        Z_front = np.zeros_like(X)
        z_f_edge = 0.0

    # Rear face (z=thickness, or thickness + rear_surf.sag(r) if curved, e.g. aspheric lens front face)
    has_sag_rear = (
        rear_surf is not None
        and hasattr(rear_surf, 'sag') and callable(rear_surf.sag)
        and getattr(rear_surf, 'radius', 0.0) != 0.0
        and not np.isinf(getattr(rear_surf, 'radius', 0.0))
    )
    if has_sag_rear:
        z_r_1d = thickness + np.array([float(rear_surf.sag(rad)) for rad in r])
        Z_rear = np.tile(z_r_1d.reshape(-1, 1), (1, angular_samples))
        z_r_edge = float(z_r_1d[-1])
    else:
        Z_rear = np.full_like(X, thickness)
        z_r_edge = thickness
    
    # Side cylinder barrel wall at constant radius
    Z_side = np.array([np.full_like(theta, z_f_edge), np.full_like(theta, z_r_edge)])
    X_side = np.array([radius * np.cos(theta), radius * np.cos(theta)])
    Y_side = np.array([radius * np.sin(theta), radius * np.sin(theta)])
    
    center, rotation = get_effective_pose(surf)
    
    def transform_mesh(x_m, y_m, z_m):
        orig_shape = x_m.shape
        local_pts = np.vstack((x_m.ravel(), y_m.ravel(), z_m.ravel()))
        world_pts = rotation @ local_pts + center.reshape(3, 1)
        return tuple(world_pts[k].reshape(orig_shape) for k in range(3))
        
    world_front = transform_mesh(X, Y, Z_front)
    world_rear = transform_mesh(X, Y, Z_rear)
    world_side = transform_mesh(X_side, Y_side, Z_side)
    
    return world_front, world_rear, world_side


def build_mount_collar_geometry(
    surf: Surface,
    inner_radius: float,
    outer_radius: float,
    thickness: float,
    z_offset: float = 0.0,
    radial_samples: int = 4,
    angular_samples: int = 32,
):
    """
    Build 3D mesh geometry for an optomechanical mount cell / sleeve collar
    around an optical element with specified inner radius, outer diameter,
    and axial Z thickness.
    """
    import numpy as np
    theta = np.linspace(0, 2 * np.pi, angular_samples)
    z0 = float(z_offset)
    z1 = float(z_offset + thickness)

    # Outer barrel wall at constant radius outer_radius
    Z_outer = np.array([np.full_like(theta, z0), np.full_like(theta, z1)])
    X_outer = np.array([outer_radius * np.cos(theta), outer_radius * np.cos(theta)])
    Y_outer = np.array([outer_radius * np.sin(theta), outer_radius * np.sin(theta)])

    # Annular front and rear rim faces (from inner_radius to outer_radius)
    r_ann = np.linspace(inner_radius, outer_radius, max(radial_samples, 2))
    T_ann, R_ann = np.meshgrid(theta, r_ann)
    X_rim = R_ann * np.cos(T_ann)
    Y_rim = R_ann * np.sin(T_ann)
    Z_front_rim = np.full_like(X_rim, z0)
    Z_rear_rim = np.full_like(X_rim, z1)

    center, rotation = get_effective_pose(surf)

    def transform_mesh(x_m, y_m, z_m):
        orig_shape = x_m.shape
        local_pts = np.vstack((x_m.ravel(), y_m.ravel(), z_m.ravel()))
        world_pts = rotation @ local_pts + center.reshape(3, 1)
        return tuple(world_pts[k].reshape(orig_shape) for k in range(3))

    world_outer = transform_mesh(X_outer, Y_outer, Z_outer)
    world_front_rim = transform_mesh(X_rim, Y_rim, Z_front_rim)
    world_rear_rim = transform_mesh(X_rim, Y_rim, Z_rear_rim)

    # Perimeter edge rings
    ring_x = outer_radius * np.cos(theta)
    ring_y = outer_radius * np.sin(theta)
    world_front_ring = transform_mesh(ring_x, ring_y, np.full_like(theta, z0))
    world_rear_ring = transform_mesh(ring_x, ring_y, np.full_like(theta, z1))

    return world_outer, world_front_rim, world_rear_rim, world_front_ring, world_rear_ring


def build_tolerance_envelope_geometry(
    surf: Surface,
    radius: float,
    thickness: float,
    tolerance: float = 1.0,
    z_offset: float = 0.0,
    angular_samples: int = 32,
):
    """
    Build 3D mesh geometry for the mounting tolerance pocket / travel envelope.
    Extends by +/- tolerance along the local optical axis Z to show allowable
    physical mounting displacement and pocket depth in the 3D CAD system.
    """
    import numpy as np
    theta = np.linspace(0, 2 * np.pi, angular_samples)
    tol = max(float(tolerance), 0.0)
    z_min = float(z_offset - tol)
    z_max = float(z_offset + thickness + tol)
    r_env = float(radius)

    # Outer cylindrical boundary wall of the tolerance pocket
    Z_cyl = np.array([np.full_like(theta, z_min), np.full_like(theta, z_max)])
    X_cyl = np.array([r_env * np.cos(theta), r_env * np.cos(theta)])
    Y_cyl = np.array([r_env * np.sin(theta), r_env * np.sin(theta)])

    center, rotation = get_effective_pose(surf)

    def transform_mesh(x_m, y_m, z_m):
        orig_shape = x_m.shape
        local_pts = np.vstack((x_m.ravel(), y_m.ravel(), z_m.ravel()))
        world_pts = rotation @ local_pts + center.reshape(3, 1)
        return tuple(world_pts[k].reshape(orig_shape) for k in range(3))

    world_cyl = transform_mesh(X_cyl, Y_cyl, Z_cyl)

    # Limit rings at min and max tolerance travel
    ring_x = r_env * np.cos(theta)
    ring_y = r_env * np.sin(theta)
    min_ring = transform_mesh(ring_x, ring_y, np.full_like(theta, z_min))
    max_ring = transform_mesh(ring_x, ring_y, np.full_like(theta, z_max))

    return world_cyl, min_ring, max_ring


def build_lens_solid_geometry(
    front: Surface, rear: Surface, radial_samples: int = 6, angular_samples: int = 32,
    cache: Optional[MeshCache] = None,
):
    """Build the two optical faces and closed barrel wall for one glass region."""
    front_mesh = _surface_mesh(front, radial_samples, angular_samples, cache=cache)
    rear_mesh = _surface_mesh(rear, radial_samples, angular_samples, cache=cache)
    front_ring = _edge_ring(front, angular_samples, cache=cache)
    rear_ring = _edge_ring(rear, angular_samples, cache=cache)
    side_mesh = tuple(np.vstack((front_ring[i], rear_ring[i])) for i in range(3))
    return front_mesh, rear_mesh, side_mesh


def build_prism_geometry(
    front: Any,
    rear: Optional[Any] = None,
    config: Optional[Any] = None,
    **kwargs
) -> Tuple[List[List[Tuple[float, float, float]]], np.ndarray]:
    """
    Build 3D polyhedral surface mesh for an optical prism (Equilateral, Right-Angle, Wedge, etc.).
    Returns:
        plot_faces: list of face polygons in visualizer coordinate frame (Z, X, Y)
        world_vertices: (N, 3) numpy array of 3D vertices in global coordinates (X, Y, Z)
    """
    import math
    if isinstance(front, dict):
        surfs = front.get("surfaces", [])
        s1 = surfs[0] if surfs else None
        s2 = surfs[1] if len(surfs) > 1 else None
        cfg = config or front.get("config") or (getattr(s1, "prism_config", None) if s1 else None)
    else:
        s1 = front
        s2 = rear
        cfg = config or getattr(s1, "prism_config", None) or (getattr(s2, "prism_config", None) if s2 else None)

    if s1 is None:
        return [], np.zeros((0, 3))

    apex_deg = float(kwargs.get("apex_angle_deg", getattr(cfg, "apex_angle_deg", 60.0) if cfg else 60.0))
    geom_type = str(kwargs.get("geometry_type", getattr(cfg, "geometry_type", "equilateral") if cfg else "equilateral")).lower()

    w_default = float(getattr(s1, "aperture_y_half", 0.0) * 2.0 or getattr(s1, "semi_diameter", 12.7) * 2.0 or 25.0)
    w = float(kwargs.get("length_mm", kwargs.get("width", getattr(cfg, "length_mm", 0.0) if cfg else 0.0)) or w_default)

    base_default = float(getattr(s1, "aperture_x_half", 0.0) * 2.0 or getattr(s1, "semi_diameter", 12.7) * 2.0 or 25.0)
    base_w = float(kwargs.get("base_width_mm", getattr(cfg, "base_width_mm", 0.0) if cfg else 0.0) or base_default)

    h_default = float(s1.thickness if s1.thickness > 0 else (s1.semi_diameter * 2.0 or 25.0))
    height = float(kwargs.get("height_mm", kwargs.get("height", getattr(cfg, "height_mm", 0.0) if cfg else 0.0)) or h_default)

    if cfg is None and s2 is not None and "apex_angle_deg" not in kwargs:
        tilt1 = float(getattr(s1, "tilt_x", 0.0))
        tilt2 = float(getattr(s2, "tilt_x", 0.0))
        diff = abs(tilt1 - tilt2)
        if diff > 0.01:
            apex_deg = diff

    hw = w / 2.0
    hb = base_w / 2.0

    if "right_angle" in geom_type or "rightangle" in geom_type or abs(apex_deg - 90.0) < 1.0:
        local_verts = np.array([
            [-hb, -hw, 0.0],
            [ hb, -hw, 0.0],
            [-hb, -hw, height],
            [-hb,  hw, 0.0],
            [ hb,  hw, 0.0],
            [-hb,  hw, height],
        ], dtype=float)
        face_indices = [
            (0, 1, 2),        # Front triangle (Y = -hw)
            (3, 5, 4),        # Back triangle (Y = +hw)
            (0, 3, 4, 1),     # Base face (Z = 0)
            (0, 2, 5, 3),     # Leg face (X = -hb)
            (1, 4, 5, 2),     # Hypotenuse face
        ]
    elif "wedge" in geom_type or apex_deg <= 15.0:
        z_wedge = base_w * math.tan(math.radians(apex_deg))
        local_verts = np.array([
            [-hb, -hw, 0.0],
            [ hb, -hw, 0.0],
            [ hb, -hw, height + z_wedge],
            [-hb, -hw, height],
            [-hb,  hw, 0.0],
            [ hb,  hw, 0.0],
            [ hb,  hw, height + z_wedge],
            [-hb,  hw, height],
        ], dtype=float)
        face_indices = [
            (0, 1, 2, 3),
            (4, 7, 6, 5),
            (0, 4, 5, 1),
            (3, 2, 6, 7),
            (0, 3, 7, 4),
            (1, 5, 6, 2),
        ]
    else:
        if 0.0 < apex_deg < 179.0:
            ht_apex = hb / math.tan(math.radians(apex_deg / 2.0))
        else:
            ht_apex = height
        if getattr(cfg, "height_mm", 0.0) > 0 and "height_mm" not in kwargs:
            ht_apex = float(cfg.height_mm)
        local_verts = np.array([
            [-hb, -hw, 0.0],
            [ hb, -hw, 0.0],
            [0.0, -hw, ht_apex],
            [-hb,  hw, 0.0],
            [ hb,  hw, 0.0],
            [0.0,  hw, ht_apex],
        ], dtype=float)
        face_indices = [
            (0, 1, 2),
            (3, 5, 4),
            (0, 3, 4, 1),
            (0, 2, 5, 3),
            (1, 4, 5, 2),
        ]

    eff_origin, eff_R = get_effective_pose(s1)
    world_verts = (eff_R @ local_verts.T).T + eff_origin
    plot_faces = [
        [(world_verts[idx, 2], world_verts[idx, 0], world_verts[idx, 1]) for idx in face]
        for face in face_indices
    ]
    return plot_faces, world_verts


def build_beamsplitter_geometry(
    front: Any,
    rear: Optional[Any] = None,
    config: Optional[Any] = None,
    **kwargs
) -> Tuple[List[List[Tuple[float, float, float]]], List[List[Tuple[float, float, float]]], np.ndarray]:
    """
    Build 3D surface mesh for cube or plate beam splitter with internal splitting interface.
    Returns:
        outer_faces: list of outer boundary faces in visualizer coords (Z, X, Y)
        interface_faces: list of internal splitting interface faces in visualizer coords (Z, X, Y)
        world_vertices: (N, 3) numpy array of 3D vertices in global coordinates (X, Y, Z)
    """
    if isinstance(front, dict):
        surfs = front.get("surfaces", [])
        s1 = surfs[0] if surfs else None
        s2 = surfs[1] if len(surfs) > 1 else None
        cfg = config or front.get("config") or (getattr(s1, "beamsplitter_config", None) if s1 else None)
    else:
        s1 = front
        s2 = rear
        cfg = config or getattr(s1, "beamsplitter_config", None) or (getattr(s2, "beamsplitter_config", None) if s2 else None)

    if s1 is None:
        return [], [], np.zeros((0, 3))

    bs_type = str(kwargs.get("splitter_type", kwargs.get("bs_type", getattr(cfg, "splitter_type", "Plate") if cfg else "Plate"))).lower()
    if "cube" in (s1.name or "").lower():
        bs_type = "cube"

    sd = float(getattr(s1, "semi_diameter", 12.7) or 12.7)

    if "cube" in bs_type:
        L_default = getattr(cfg, "cube_side_length_mm", 0.0) if cfg else 0.0
        if L_default <= 0.0:
            L_default = getattr(cfg, "clear_aperture_mm", 0.0) if cfg else 0.0
        if L_default <= 0.0:
            L_default = sd * 2.0 or 25.4
        L = float(kwargs.get("cube_side_length_mm", kwargs.get("size", L_default)))
        hL = L / 2.0

        local_verts = np.array([
            [-hL, -hL, 0.0],
            [ hL, -hL, 0.0],
            [ hL,  hL, 0.0],
            [-hL,  hL, 0.0],
            [-hL, -hL, L],
            [ hL, -hL, L],
            [ hL,  hL, L],
            [-hL,  hL, L],
        ], dtype=float)

        outer_indices = [
            (0, 1, 2, 3),  # Front face (Z = 0)
            (4, 7, 6, 5),  # Rear face (Z = L)
            (0, 4, 5, 1),  # Bottom
            (2, 6, 7, 3),  # Top
            (0, 3, 7, 4),  # Left
            (1, 5, 6, 2),  # Right
        ]
        interface_indices = [
            (0, 2, 6, 4),  # 45-degree diagonal splitting plane
        ]
    else:
        t_default = float(getattr(cfg, "substrate_thickness_mm", 0.0) if cfg else 0.0)
        if t_default <= 0.0:
            t_default = float(getattr(s1, "substrate_thickness", 0.0) or getattr(s1, "thickness", 0.0) or 2.0)
        t = max(float(kwargs.get("substrate_thickness_mm", kwargs.get("thickness", t_default))), 0.5)

        wx_default = float(getattr(s1, "aperture_x_half", 0.0) * 2.0 or sd * 2.0 or 25.4)
        wy_default = float(getattr(s1, "aperture_y_half", 0.0) * 2.0 or sd * 2.0 or 25.4)
        wx = float(kwargs.get("clear_aperture_x", kwargs.get("width", wx_default)))
        wy = float(kwargs.get("clear_aperture_y", kwargs.get("height", wy_default)))
        hwx = wx / 2.0
        hwy = wy / 2.0

        local_verts = np.array([
            [-hwx, -hwy, 0.0],
            [ hwx, -hwy, 0.0],
            [ hwx,  hwy, 0.0],
            [-hwx,  hwy, 0.0],
            [-hwx, -hwy, -t],
            [ hwx, -hwy, -t],
            [ hwx,  hwy, -t],
            [-hwx,  hwy, -t],
        ], dtype=float)

        outer_indices = [
            (4, 7, 6, 5),
            (0, 4, 5, 1),
            (2, 6, 7, 3),
            (0, 3, 7, 4),
            (1, 5, 6, 2),
        ]
        interface_indices = [
            (0, 1, 2, 3),
        ]

    eff_origin, eff_R = get_effective_pose(s1)
    world_verts = (eff_R @ local_verts.T).T + eff_origin
    outer_faces = [
        [(world_verts[idx, 2], world_verts[idx, 0], world_verts[idx, 1]) for idx in face]
        for face in outer_indices
    ]
    interface_faces = [
        [(world_verts[idx, 2], world_verts[idx, 0], world_verts[idx, 1]) for idx in face]
        for face in interface_indices
    ]
    return outer_faces, interface_faces, world_verts


def build_aperture_stop_geometry(
    surf: Surface,
    thickness: float = 1.5,
    outer_scale: float = 1.45,
    angular_samples: int = 32,
    surrounding_max_sd: float = 0.0,
    z_offset_mode: str = "auto",
):
    """
    Build 3D closed solid annular diaphragm baffle plate for an aperture stop.
    Returns:
        stop_meshes: tuple of 4 meshes (front_mesh, back_mesh, outer_wall_mesh, inner_wall_mesh)
                     forming the complete closed 3D solid diaphragm plate
        inner_rim: (x_ring, y_ring, z_ring) for the clear aperture inner opening at active plane
        outer_rim: (x_ring, y_ring, z_ring) for the outer mounting diaphragm edge at active plane
    """
    # The outer metal plate size
    sd = max(float(getattr(surf, "semi_diameter", 0.0)), 0.1)
    
    # The clear hole size (where light passes through)
    hole_radius = float(getattr(surf, "aperture_r_outer", 0.0))
    if hole_radius <= 0:
        hole_radius = sd
        
    # The inside edge of the drawn wall is the clear hole
    r_inner = hole_radius
    
    # The outside edge of the drawn wall is the semi_diameter (mount size)
    theta = np.linspace(0.0, 2.0 * np.pi, angular_samples)
    r_inner = np.full_like(theta, hole_radius)

    sub_shape = str(getattr(surf, 'substrate_shape', '')).lower()
    if sub_shape == 'rectangular':
        outer_hx = float(getattr(surf, "semi_diameter_x", 0.0))
        outer_hy = float(getattr(surf, "semi_diameter_y", 0.0))
        if outer_hx <= 0.0: outer_hx = max(sd, hole_radius + 0.1)
        if outer_hy <= 0.0: outer_hy = max(sd, hole_radius + 0.1)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        cos_t[np.abs(cos_t) < 1e-9] = 1e-9 * np.sign(cos_t[np.abs(cos_t) < 1e-9] + 1e-12)
        sin_t[np.abs(sin_t) < 1e-9] = 1e-9 * np.sign(sin_t[np.abs(sin_t) < 1e-9] + 1e-12)
        r_out_x = np.abs(outer_hx / cos_t)
        r_out_y = np.abs(outer_hy / sin_t)
        r_outer = np.minimum(r_out_x, r_out_y)
        r_outer = np.maximum(r_outer, r_inner + 0.001)
    else:
        r_outer = np.full_like(theta, max(sd, hole_radius))
        r_outer[np.abs(r_outer - r_inner) < 1e-4] += 0.001
        
    t = float(getattr(surf, "aperture_thickness", 0.0) or getattr(surf, "stop_thickness", 0.0) or thickness)
    t = max(t, 0.4)

    radii_frac = np.linspace(0.0, 1.0, 4)
    radius_grid = r_inner[np.newaxis, :] + radii_frac[:, np.newaxis] * (r_outer - r_inner)[np.newaxis, :]
    theta_grid = np.broadcast_to(theta, radius_grid.shape)

    x_local = radius_grid * np.cos(theta_grid)
    y_local = radius_grid * np.sin(theta_grid)
    if surf.radius != 0.0 and not np.isinf(surf.radius):
        z_sag_base = _surface_sag_values(surf, radius_grid)
        outer_sag = _surface_sag_values(surf, r_outer)
        inner_sag = _surface_sag_values(surf, r_inner)
    else:
        z_sag_base = np.zeros_like(radius_grid)
        outer_sag = np.zeros_like(r_outer)
        inner_sag = np.zeros_like(r_inner)

    if z_offset_mode == "backward":
        z_front_local = z_sag_base - t
        z_back_local = z_sag_base
        z_outer_0, z_outer_1 = outer_sag - t, outer_sag
        z_inner_0, z_inner_1 = inner_sag - t, inner_sag
        active_sag_inner = inner_sag
        active_sag_outer = outer_sag
    elif z_offset_mode == "centered":
        z_front_local = z_sag_base - t * 0.5
        z_back_local = z_sag_base + t * 0.5
        z_outer_0, z_outer_1 = outer_sag - t * 0.5, outer_sag + t * 0.5
        z_inner_0, z_inner_1 = inner_sag - t * 0.5, inner_sag + t * 0.5
        active_sag_inner = inner_sag
        active_sag_outer = outer_sag
    else:
        # Default "forward" or "auto" with positive thickness
        z_front_local = z_sag_base
        z_back_local = z_sag_base + t
        z_outer_0, z_outer_1 = outer_sag, outer_sag + t
        z_inner_0, z_inner_1 = inner_sag, inner_sag + t
        active_sag_inner = inner_sag
        active_sag_outer = outer_sag

    center = np.asarray(getattr(surf, "global_origin", [0.0, 0.0, 0.0]), dtype=float)
    rotation = np.asarray(getattr(surf, "global_R", np.eye(3)), dtype=float)
    shape = x_local.shape

    # 1. Front annular face
    local_front = np.vstack((x_local.ravel(), y_local.ravel(), z_front_local.ravel()))
    world_front = rotation @ local_front + center.reshape(3, 1)
    front_mesh = tuple(world_front[i].reshape(shape) for i in range(3))

    # 2. Back annular face
    local_back = np.vstack((x_local.ravel(), y_local.ravel(), z_back_local.ravel()))
    world_back = rotation @ local_back + center.reshape(3, 1)
    back_mesh = tuple(world_back[i].reshape(shape) for i in range(3))

    # 3. Outer cylindrical barrel wall
    x_outer_wall = np.vstack((r_outer * np.cos(theta), r_outer * np.cos(theta)))
    y_outer_wall = np.vstack((r_outer * np.sin(theta), r_outer * np.sin(theta)))
    z_outer_wall = np.vstack((np.full_like(theta, z_outer_0), np.full_like(theta, z_outer_1)))
    local_outer = np.vstack((x_outer_wall.ravel(), y_outer_wall.ravel(), z_outer_wall.ravel()))
    world_outer = rotation @ local_outer + center.reshape(3, 1)
    outer_wall_mesh = tuple(world_outer[i].reshape((2, angular_samples)) for i in range(3))

    # 4. Inner cylindrical bore wall
    x_inner_wall = np.vstack((r_inner * np.cos(theta), r_inner * np.cos(theta)))
    y_inner_wall = np.vstack((r_inner * np.sin(theta), r_inner * np.sin(theta)))
    z_inner_wall = np.vstack((np.full_like(theta, z_inner_0), np.full_like(theta, z_inner_1)))
    local_inner = np.vstack((x_inner_wall.ravel(), y_inner_wall.ravel(), z_inner_wall.ravel()))
    world_inner = rotation @ local_inner + center.reshape(3, 1)
    inner_wall_mesh = tuple(world_inner[i].reshape((2, angular_samples)) for i in range(3))

    # Rims at nominal active aperture plane
    inner_local = np.vstack((r_inner * np.cos(theta), r_inner * np.sin(theta), np.full_like(theta, active_sag_inner)))
    inner_world = rotation @ inner_local + center.reshape(3, 1)
    inner_rim = (inner_world[0], inner_world[1], inner_world[2])

    outer_local = np.vstack((r_outer * np.cos(theta), r_outer * np.sin(theta), np.full_like(theta, active_sag_outer)))
    outer_world = rotation @ outer_local + center.reshape(3, 1)
    outer_rim = (outer_world[0], outer_world[1], outer_world[2])

    stop_meshes = (front_mesh, back_mesh, outer_wall_mesh, inner_wall_mesh)
    return stop_meshes, inner_rim, outer_rim


def build_rectangular_aperture_geometry(
    surf: Surface,
    thickness: float = 1.5,
    outer_scale: float = 1.0,
    surrounding_max_sd: float = 0.0,
    z_offset_mode: str = "auto",
):
    """
    Build 3D closed solid plate for a rectangular aperture.
    Draws a circular outer disk (semi_diameter) with a rectangular inner hole (aperture_x_half, aperture_y_half).
    """
    hx = max(float(getattr(surf, "aperture_x_half", 0.0)), 0.001)
    hy = max(float(getattr(surf, "aperture_y_half", 0.0)), 0.001)
    sd = max(float(getattr(surf, "semi_diameter", 0.0)), 0.1)

    t = float(getattr(surf, "aperture_thickness", 0.0) or getattr(surf, "stop_thickness", 0.0) or thickness)
    t = max(t, 0.4)

    if z_offset_mode == "backward":
        z_front_local = -t
        z_back_local = 0.0
    elif z_offset_mode == "centered":
        z_front_local = -t * 0.5
        z_back_local = t * 0.5
    else:
        z_front_local = 0.0
        z_back_local = t

    center = np.asarray(getattr(surf, "global_origin", [0.0, 0.0, 0.0]), dtype=float)
    rotation = np.asarray(getattr(surf, "global_R", np.eye(3)), dtype=float)

    corners = np.array([
        np.arctan2(hy, hx),
        np.arctan2(hy, -hx),
        np.arctan2(-hy, -hx),
        np.arctan2(-hy, hx)
    ])
    corners = np.mod(corners, 2 * np.pi)
    axes_ang = np.array([0.0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    theta = np.sort(np.unique(np.concatenate([np.linspace(0, 2*np.pi, 64), corners, axes_ang])))
    angular_samples = len(theta)
    
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    cos_t[np.abs(cos_t) < 1e-9] = 1e-9 * np.sign(cos_t[np.abs(cos_t) < 1e-9] + 1e-12)
    sin_t[np.abs(sin_t) < 1e-9] = 1e-9 * np.sign(sin_t[np.abs(sin_t) < 1e-9] + 1e-12)
    
    r_x = np.abs(hx / cos_t)
    r_y = np.abs(hy / sin_t)
    r_in = np.minimum(r_x, r_y)

    sub_shape = str(getattr(surf, 'substrate_shape', '')).lower()
    if sub_shape == 'rectangular':
        outer_hx = float(getattr(surf, "semi_diameter_x", 0.0))
        outer_hy = float(getattr(surf, "semi_diameter_y", 0.0))
        if outer_hx <= 0.0: outer_hx = max(sd, hx + 0.1)
        if outer_hy <= 0.0: outer_hy = max(sd, hy + 0.1)
        r_out_x = np.abs(outer_hx / cos_t)
        r_out_y = np.abs(outer_hy / sin_t)
        r_out = np.minimum(r_out_x, r_out_y)
        r_out = np.maximum(r_out, r_in + 0.001)
    else:
        r_out = np.full_like(theta, max(sd, np.max(r_in) + 0.001))
    
    r_in = np.minimum(r_in, r_out - 0.001)

    n_rad = 4
    radii_frac = np.linspace(0.0, 1.0, n_rad)
    radius_grid = r_in[np.newaxis, :] + radii_frac[:, np.newaxis] * (r_out - r_in)[np.newaxis, :]
    
    x_local = radius_grid * np.cos(theta)[np.newaxis, :]
    y_local = radius_grid * np.sin(theta)[np.newaxis, :]
    
    z_front_grid = np.full_like(x_local, z_front_local)
    z_back_grid = np.full_like(x_local, z_back_local)
    
    def transform_grid(x_g, y_g, z_g):
        pts = np.column_stack([x_g.ravel(), y_g.ravel(), z_g.ravel()])
        world = (rotation @ pts.T).T + center
        return world[:, 0].reshape(x_g.shape), world[:, 1].reshape(y_g.shape), world[:, 2].reshape(z_g.shape)

    xf, yf, zf = transform_grid(x_local, y_local, z_front_grid)
    xb, yb, zb = transform_grid(x_local, y_local, z_back_grid)

    world_quads = []
    
    for i in range(n_rad - 1):
        for j in range(angular_samples - 1):
            world_quads.append([
                (zf[i, j], xf[i, j], yf[i, j]),
                (zf[i, j+1], xf[i, j+1], yf[i, j+1]),
                (zf[i+1, j+1], xf[i+1, j+1], yf[i+1, j+1]),
                (zf[i+1, j], xf[i+1, j], yf[i+1, j])
            ])
            world_quads.append([
                (zb[i, j], xb[i, j], yb[i, j]),
                (zb[i, j+1], xb[i, j+1], yb[i, j+1]),
                (zb[i+1, j+1], xb[i+1, j+1], yb[i+1, j+1]),
                (zb[i+1, j], xb[i+1, j], yb[i+1, j])
            ])

    for j in range(angular_samples - 1):
        world_quads.append([
            (zf[-1, j], xf[-1, j], yf[-1, j]),
            (zf[-1, j+1], xf[-1, j+1], yf[-1, j+1]),
            (zb[-1, j+1], xb[-1, j+1], yb[-1, j+1]),
            (zb[-1, j], xb[-1, j], yb[-1, j])
        ])
        world_quads.append([
            (zf[0, j], xf[0, j], yf[0, j]),
            (zf[0, j+1], xf[0, j+1], yf[0, j+1]),
            (zb[0, j+1], xb[0, j+1], yb[0, j+1]),
            (zb[0, j], xb[0, j], yb[0, j])
        ])

    inner_rim = (xf[0, :], yf[0, :], zf[0, :])
    outer_rim = (xf[-1, :], yf[-1, :], zf[-1, :])

    all_verts = np.column_stack([xf.ravel(), yf.ravel(), zf.ravel()])
    return world_quads, inner_rim, outer_rim, all_verts


def generate_3d_rays(
    surface_stack: SurfaceStack,
    wavelengths: Optional[List[float]] = None,
    num_rings: int = 2,
    rays_per_ring: int = 6,
    system_config: Optional[Any] = None,
) -> List[Ray]:
    """Generate and trace a 3D volumetric ray bundle through the surface stack with multi-field pupil aiming."""
    if not surface_stack or not surface_stack.surfaces:
        return []

    surfaces = surface_stack.get_active_sequence() if hasattr(surface_stack, 'get_active_sequence') else surface_stack.surfaces
    if not surfaces:
        return []

    if hasattr(surface_stack, '_update_accumulated_positions'):
        surface_stack._update_accumulated_positions()

    if system_config is None:
        saved_config = getattr(surface_stack, "system_config", None)
        if isinstance(saved_config, dict):
            from optics_app.engine.system_config import SystemConfig
            system_config = SystemConfig()
            system_config.from_dict(saved_config)
        elif saved_config is not None:
            system_config = saved_config

    system = OpticalSystem(surface_stack)
    wls = wavelengths
    if not wls and system_config is not None and getattr(system_config, 'wavelength_list_um', None):
        wls = system_config.wavelength_list_um
    # A standalone viewer must not invent visible wavelengths for an NIR or
    # UV design. Use the project's reference wavelength when no list exists.
    wls = wls or [float(getattr(surface_stack, "primary_wavelength", 0.5876))]

    d0 = float(getattr(surfaces[0], "thickness", 0.0))
    r1 = float(getattr(surfaces[1] if len(surfaces) > 1 else surfaces[0], "semi_diameter", 10.0))
    if r1 <= 0.0:
        r1 = 12.7

    is_collimated = False
    if system_config is not None:
        beam_type = getattr(system_config, "beam_type", "")
        telecentric = getattr(system_config, "telecentric_object_space", False)
        is_collimated = (beam_type == "Collimated (Parallel Beam)") or telecentric or (d0 > 1e4)
    else:
        is_collimated = (d0 > 1e4 or d0 <= 0.0)

    # Zemax-style paraxial entrance pupil solve
    from optics_app.engine.pupil_solve import calculate_entrance_pupil, aim_ray_to_pupil, aim_collimated_ray
    from optics_app.engine.system_config import FieldType, FieldPoint

    default_epd = 2.0 * r1
    if system_config is not None:
        from optics_app.engine.system_config import ApertureType
        if getattr(system_config, "aperture_type", None) == ApertureType.ENTRANCE_PUPIL_DIAMETER:
            default_epd = float(getattr(system_config, "aperture_value", 2.0 * r1))

    if len(surfaces) > 1:
        z_ep, r_ep, stop_idx = calculate_entrance_pupil(
            surfaces,
            wavelength_um=wls[0] if wls else 0.5876,
            default_epd=default_epd,
        )
    else:
        z_ep = float(surfaces[0].global_origin[2])
        r_ep = max(r1, 1.0)
        stop_idx = 0

    from optics_app.engine.pupil_solve import generate_pupil_grid
    from optics_app.engine.system_config import RayPatternType

    pattern = getattr(system_config, "ray_pattern", RayPatternType.HEXAPOLAR) if system_config is not None else RayPatternType.HEXAPOLAR
    pupil_grid = generate_pupil_grid(
        pattern=pattern,
        rings=num_rings,
        arms=rays_per_ring,
        include_chief_ray=True,
        linear_arms=True,
        max_radius=1.0,
    )
    pupil_coords = [(px, py) for px, py, _ in pupil_grid]

    # Full pupil cross-hair fan rays extending to the outer aperture rim
    for frac in [-1.0, -0.707, -0.45, 0.45, 0.707, 1.0]:
        pupil_coords.append((0.0, frac))
        pupil_coords.append((frac, 0.0))

    unique_pupil = []
    seen = set()
    for px, py in pupil_coords:
        key = (round(px, 4), round(py, 4))
        if key not in seen:
            seen.add(key)
            unique_pupil.append((px, py))

    # Multi-field configuration
    fields = [FieldPoint(x=0.0, y=0.0, weight=1.0)]
    field_type = FieldType.ANGLE_DEG if is_collimated else FieldType.OBJECT_HEIGHT_MM
    if system_config is not None:
        if getattr(system_config, "fields", None) and len(system_config.fields) > 0:
            fields = system_config.fields
        field_type = getattr(system_config, "field_type", field_type)

    has_off_axis = any(abs(float(getattr(f, "x", 0.0))) > 1e-5 or abs(float(getattr(f, "y", 0.0))) > 1e-5 for f in fields)
    synthetic_field_idx = None
    if not has_off_axis:
        nom_y = 1.0 if field_type == FieldType.ANGLE_DEG else max(1.0, r_ep * 0.25)
        syn_field = FieldPoint(x=0.0, y=nom_y, weight=1.0)
        synthetic_field_idx = len(fields)
        fields = list(fields) + [syn_field]

    input_rays = []
    source_is_modeled = classify_surface_for_3d(surfaces[0], 0, len(surfaces))["object"]
    first_origin = np.asarray(surfaces[0].global_origin, dtype=float)
    first_rotation = np.asarray(surfaces[0].global_R, dtype=float)
    preview_launch_distance = max(10.0, 2.0 * max(r_ep, float(surfaces[0].semi_diameter)))
    for wl in wls:
        for f_idx, field in enumerate(fields):
            fx = float(field.x)
            fy = float(field.y)
            is_synthetic = (synthetic_field_idx is not None and f_idx == synthetic_field_idx)
            is_on_axis = (abs(fx) < 1e-6 and abs(fy) < 1e-6)

            for px, py in unique_pupil:
                # If synthetic canonical field, only trace its chief ray (px=0, py=0)
                if is_synthetic and not (abs(px) < 1e-7 and abs(py) < 1e-7):
                    continue
                if not source_is_modeled:
                    # No source plane is present in this design. Supply a
                    # clearly marked preview bundle incident on its first
                    # element instead of pretending the element emits rays.
                    target = (first_origin + first_rotation[:, 0] * px * r_ep
                              + first_rotation[:, 1] * py * r_ep)
                    if field_type == FieldType.ANGLE_DEG:
                        direction = np.array([np.tan(np.radians(fx)),
                                              np.tan(np.radians(fy)), 1.0], dtype=float)
                    else:
                        target = target + first_rotation[:, 0] * fx + first_rotation[:, 1] * fy
                        direction = np.array([0.0, 0.0, 1.0], dtype=float)
                    direction /= np.linalg.norm(direction)
                    ray = Ray(pos=target - direction * preview_launch_distance,
                              dir=direction, wavelength=wl)
                    ray.preview_source_assumed = True
                elif is_collimated and field_type == FieldType.ANGLE_DEG:
                    ray = aim_collimated_ray(
                        surfaces=surfaces,
                        field_angle_deg=(fx, fy),
                        pupil_norm=(px, py),
                        wavelength_um=wl,
                        z_ep_global=z_ep,
                        r_ep=r_ep,
                    )
                else:
                    if field_type == FieldType.OBJECT_HEIGHT_MM:
                        obj_x = fx
                        obj_y = fy
                    else:
                        obj_x = -d0 * np.tan(np.radians(fx)) if d0 > 0 else 0.0
                        obj_y = -d0 * np.tan(np.radians(fy)) if d0 > 0 else 0.0

                    ray = aim_ray_to_pupil(
                        surfaces=surfaces,
                        obj_point=(obj_x, obj_y, 0.0),
                        pupil_norm=(px, py),
                        wavelength_um=wl,
                        z_ep_global=z_ep,
                        r_ep=r_ep,
                        system=system,
                    )

                is_marginal = is_on_axis and (
                    (abs(px) < 1e-5 and abs(abs(py) - 1.0) < 1e-4) or
                    (abs(py) < 1e-5 and abs(abs(px) - 1.0) < 1e-4)
                )

                ray.field_idx = f_idx
                ray.is_chief = (not is_on_axis) and (abs(px) < 1e-9 and abs(py) < 1e-9)
                ray.is_axial = is_on_axis and (abs(px) < 1e-9 and abs(py) < 1e-9)
                ray.is_marginal = is_marginal
                ray.is_upper_marginal = bool(is_marginal and py > 0)
                ray.is_lower_marginal = bool(is_marginal and py < 0)
                ray.field_coord = (fx, fy)
                ray.field_weight = float(getattr(field, "weight", 1.0))
                ray.pupil_coord = (float(px), float(py))
                input_rays.append(ray)

    return system.trace_rays(input_rays)


class Visualizer3DWidget(QWidget if QT_AVAILABLE else object):
    """
    Interactive 3D Optical Layout, Ray Trace, and Aperture Visualizer.
    Provides volumetric 3D solid rendering for lenses, mirrors, gratings, DMDs,
    annular diaphragms and clear apertures for stops, 3D ray trajectories,
    and optical component callout labels.
    """
    transparency_changed = pyqtSignal(bool) if QT_AVAILABLE else DummySignal()
    surface_selected = pyqtSignal(int) if QT_AVAILABLE else DummySignal()
    dark_mode_changed = pyqtSignal(bool) if QT_AVAILABLE else DummySignal()

    def __init__(self, model: Optional[Any] = None, parent=None, auto_refresh: bool = True):
        if QT_AVAILABLE:
            super().__init__(parent)
        self.model = model
        self.surface_stack = None
        self.rays: List[Ray] = []
        self.show_rays: bool = True
        self.show_marginal_chief: bool = False
        self.marginal_chief_only: bool = False
        self.show_labels: bool = True
        self.show_apertures: bool = True
        self.show_grid: bool = True
        self.dark_mode: bool = False
        self.cb_dark_mode: Optional[Any] = None
        self.show_air_gaps: bool = True
        self.air_gap_constant_diameter: float = 25.4
        self.show_mounts: bool = False
        self.mount_diameter: float = 30.5
        self.mount_z_thickness: float = 0.0
        self.tolerance_thickness: float = 1.0
        self.show_tolerance_envelope: bool = False
        self.element_displacements: Dict[int, float] = {}
        self.global_displacement_z: float = 0.0
        self.no_transparency: bool = False
        self.aspect_mode: str = "wide"  # "wide", "physical", "cubic"
        self.mesh_quality: str = "interactive"  # interactive or detailed
        self.projection_mode: str = "persp"  # "persp", "ortho"

        self.fig = None
        self.canvas = None
        self.ax = None
        self.toolbar = None
        self.cb_show_rays = None
        self.cb_show_labels = None
        self.cb_show_air_gaps = None
        self.lbl_air_gap_diam = None
        self.combo_air_gap_diam = None
        self.cb_show_mounts = None
        self.lbl_mount_diam = None
        self.combo_mount_diam = None
        self.lbl_mount_z = None
        self.combo_mount_z = None
        self.lbl_tol_z = None
        self.combo_tol_z = None
        self.cb_show_tol = None
        self.lbl_displace = None
        self.combo_displace_elem = None
        self.spn_displace_z = None
        self.btn_displace_reset = None
        self.cb_show_apertures = None
        self.cb_show_grid = None
        self.cb_no_transparency = None
        self.btn_trace_3d = None
        self.btn_color_mode = None
        self.btn_grating_fan = None
        self.btn_mech_state = None
        self.btn_dmd_state = None
        self.btn_dmd = None
        self.color_mode: str = "wavelength"
        self.system_config: Optional[Any] = None
        self.btn_view_iso = None
        self.btn_reset_view = None
        self.btn_view_top = None
        self.btn_view_bottom = None
        self.btn_view_side = None
        self.btn_view_side_opposite = None
        self.btn_view_front = None
        self.btn_view_back = None
        self.combo_views = None
        self.btn_orbit_left = None
        self.btn_orbit_right = None
        self.btn_flip_view = None
        self.btn_fit_view = None
        self.combo_aspect = None
        self.combo_detail = None
        self.btn_toggle_proj = None
        self.btn_toggle_cad = None
        self.strip_cad_widget = None
        self.btn_export_step = None
        self.lbl_status = None
        self._view_anim_timer = None
        self._anim_step: int = 0
        self._anim_total_steps: int = 5

        self._is_panning: bool = False
        self._is_rotating: bool = False
        self._is_zooming: bool = False
        self._mouse_press_pos = None
        self._mouse_last_pos = None
        self._base_xlim = None
        self._base_ylim = None
        self._base_zlim = None
        self._cam_dist: float = 7.5
        self._last_route_count: int = 0
        self._focused_grating_index: Optional[int] = None
        self._ray_artists: List[Any] = []
        self._ray_line_colls: Dict[Any, Any] = {}
        self._ray_text_artists: List[Any] = []
        self._grating_artists: List[Any] = []
        self._last_ray_annot_key: Any = None
        self._last_grating_key: Any = None
        self._last_geom_fingerprint: Optional[tuple] = None
        self._mesh_cache: MeshCache = GLOBAL_MESH_CACHE

        if QT_AVAILABLE and HAS_MPL_QT:
            self._init_ui()

        if auto_refresh and self.model is not None and hasattr(self.model, "system_updated"):
            try:
                self.model.system_updated.connect(self.refresh_layout)
            except Exception:
                pass

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(1)

        if QT_AVAILABLE and hasattr(self, 'setSizePolicy'):
            try:
                expanding = getattr(getattr(QtWidgets, 'QSizePolicy', None), 'Expanding', None)
                if expanding is not None:
                    self.setSizePolicy(QtWidgets.QSizePolicy(expanding, expanding))
            except Exception:
                pass

        btn_style = (
            "QPushButton { background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 4px; "
            "padding: 2px 7px; font-size: 11px; font-weight: 500; color: #334155; min-height: 20px; } "
            "QPushButton:hover { background-color: #f1f5f9; border-color: #cbd5e1; color: #0f172a; } "
            "QPushButton:checked, QPushButton:pressed { background-color: #e2e8f0; border-color: #94a3b8; color: #0f172a; }"
        )
        accent_btn_style = (
            "QPushButton { background-color: #eff6ff; border: 1px solid #bfdbfe; border-radius: 4px; "
            "padding: 2px 8px; font-size: 11px; font-weight: 600; color: #1d4ed8; min-height: 20px; } "
            "QPushButton:hover { background-color: #dbeafe; border-color: #93c5fd; } "
            "QPushButton:pressed { background-color: #bfdbfe; }"
        )
        step_btn_style = (
            "QPushButton { background-color: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 4px; "
            "padding: 2px 8px; font-size: 11px; font-weight: 600; color: #15803d; min-height: 20px; } "
            "QPushButton:hover { background-color: #dcfce7; border-color: #86efac; } "
            "QPushButton:pressed { background-color: #bbf7d0; }"
        )
        toggle_cad_style = (
            "QPushButton { background-color: #faf5ff; border: 1px solid #e9d5ff; border-radius: 4px; "
            "padding: 2px 8px; font-size: 11px; font-weight: 600; color: #7e22ce; min-height: 20px; } "
            "QPushButton:hover { background-color: #f3e8ff; border-color: #d8b4fe; } "
            "QPushButton:checked { background-color: #ede9fe; border-color: #c084fc; color: #6b21a8; }"
        )
        combo_style = (
            "QComboBox { background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 4px; "
            "padding: 1px 6px; font-size: 11px; font-weight: 500; color: #334155; min-height: 20px; } "
            "QComboBox:hover { border-color: #cbd5e1; } "
            "QComboBox:focus { border-color: #3b82f6; }"
        )
        cb_style = (
            "QCheckBox { font-size: 11px; font-weight: 500; color: #475569; spacing: 4px; } "
            "QCheckBox:hover { color: #0f172a; }"
        )
        lbl_style = "font-size: 11px; font-weight: 600; color: #475569;"

        def create_sep():
            sep = QFrame()
            sep.setFrameShape(QFrame.VLine)
            sep.setFrameShadow(QFrame.Sunken)
            sep.setStyleSheet("color: #e2e8f0; margin: 2px 3px;")
            return sep

        # Row 1: Streamlined Primary Viewport Bar
        container_main = QWidget()
        self.container_main = container_main
        container_main.setStyleSheet("background-color: #0f172a; border-bottom: 1px solid #334155;" if self.dark_mode else "background-color: #ffffff; border-bottom: 1px solid #e2e8f0;")
        strip_main = QHBoxLayout(container_main)
        strip_main.setContentsMargins(6, 4, 6, 4)
        strip_main.setSpacing(4)

        # Camera & Standard Views
        self.btn_view_iso = QPushButton("Iso")
        self.btn_view_iso.setStyleSheet(btn_style)
        self.btn_view_iso.setToolTip("Isometric 3D perspective view (elev=20°, azim=-60°). Hotkey: Num 0.")
        self.btn_view_iso.clicked.connect(lambda: self.set_view_isometric(animate=True))
        self.btn_reset_view = self.btn_view_iso
        strip_main.addWidget(self.btn_view_iso)

        self.btn_view_front = QPushButton("Front")
        self.btn_view_front.setStyleSheet(btn_style)
        self.btn_view_front.setToolTip("Front View XY (elev=0°, azim=0°). Hotkey: Num 1.")
        self.btn_view_front.clicked.connect(lambda: self.set_view_front(animate=True))
        strip_main.addWidget(self.btn_view_front)

        self.btn_view_side = QPushButton("Side")
        self.btn_view_side.setStyleSheet(btn_style)
        self.btn_view_side.setToolTip("Side View ZY (elev=0°, azim=-90°). Hotkey: Num 3.")
        self.btn_view_side.clicked.connect(lambda: self.set_view_side(animate=True))
        strip_main.addWidget(self.btn_view_side)

        self.btn_view_top = QPushButton("Top")
        self.btn_view_top.setStyleSheet(btn_style)
        self.btn_view_top.setToolTip("Top View XZ (elev=90°, azim=-90°). Hotkey: Num 7.")
        self.btn_view_top.clicked.connect(lambda: self.set_view_top(animate=True))
        strip_main.addWidget(self.btn_view_top)

        self.btn_fit_view = QPushButton("Fit")
        self.btn_fit_view.setStyleSheet(btn_style)
        self.btn_fit_view.setToolTip("Fit camera to frame all optical elements (Hotkey: Num .).")
        self.btn_fit_view.clicked.connect(self.fit_view)
        strip_main.addWidget(self.btn_fit_view)

        self.combo_views = QComboBox()
        self.combo_views.setStyleSheet(combo_style)
        self.combo_views.addItems([
            "Views ▾",
            "Isometric View [0]",
            "Front (XY) [1]",
            "Back (-XY) [Ctrl+1]",
            "Side (ZY) [3]",
            "Opposite Side (-ZY) [Ctrl+3]",
            "Top (XZ) [7]",
            "Bottom (-XZ) [Ctrl+7]",
            "Diagonal: Front-Side (45°)",
            "Diagonal: Side-Back (135°)",
            "Diagonal: Back-Opposite (225°)",
            "Diagonal: Opposite-Front (315°)",
            "Diagonal: Top-Front (45° Elev)",
            "Diagonal: Top-Side (45° Elev)",
            "Iso NE (30°, -45°)",
            "Iso NW (30°, -135°)",
            "Iso SE (30°, 45°)",
            "Iso SW (30°, 135°)",
        ])
        self.combo_views.setToolTip("Switch directly to standard orthographic planes, inverted planes, or diagonal angles.")
        self.combo_views.currentIndexChanged.connect(self._on_view_combo_changed)
        strip_main.addWidget(self.combo_views)

        self.btn_toggle_proj = QPushButton("Persp" if getattr(self, "projection_mode", "persp") == "persp" else "Ortho")
        self.btn_toggle_proj.setStyleSheet(btn_style)
        self.btn_toggle_proj.setToolTip("Toggle Perspective / Orthographic projection (Hotkey: Num 5).")
        self.btn_toggle_proj.clicked.connect(self.toggle_projection)
        strip_main.addWidget(self.btn_toggle_proj)

        strip_main.addWidget(create_sep())

        # Display Toggles
        self.cb_show_rays = QCheckBox("Rays")
        self.cb_show_rays.setStyleSheet(cb_style)
        self.cb_show_rays.setChecked(self.show_rays)
        self.cb_show_rays.setToolTip("Toggle 3D ray propagation trajectories.")
        self.cb_show_rays.toggled.connect(self.set_show_rays)
        strip_main.addWidget(self.cb_show_rays)

        self.cb_show_labels = QCheckBox("Labels")
        self.cb_show_labels.setStyleSheet(cb_style)
        self.cb_show_labels.setChecked(self.show_labels)
        self.cb_show_labels.setToolTip("Toggle optical element callout labels.")
        self.cb_show_labels.toggled.connect(self.set_show_labels)
        strip_main.addWidget(self.cb_show_labels)

        self.cb_show_grid = QCheckBox("Grid")
        self.cb_show_grid.setStyleSheet(cb_style)
        self.cb_show_grid.setChecked(self.show_grid)
        self.cb_show_grid.setToolTip("Toggle 3D coordinate grid planes.")
        self.cb_show_grid.toggled.connect(self.set_show_grid)
        strip_main.addWidget(self.cb_show_grid)

        self.cb_show_air_gaps = QCheckBox("Air Gaps")
        self.cb_show_air_gaps.setStyleSheet(cb_style)
        self.cb_show_air_gaps.setChecked(self.show_air_gaps)
        self.cb_show_air_gaps.setToolTip("Toggle air gap path cylinders between lenses.")
        self.cb_show_air_gaps.toggled.connect(self.set_show_air_gaps)
        strip_main.addWidget(self.cb_show_air_gaps)

        self.cb_show_mounts = QCheckBox("Mounts")
        self.cb_show_mounts.setStyleSheet(cb_style)
        self.cb_show_mounts.setChecked(self.show_mounts)
        self.cb_show_mounts.setToolTip("Toggle mechanical mounts / cells around optical elements (optional).")
        self.cb_show_mounts.toggled.connect(self.set_show_mounts)
        strip_main.addWidget(self.cb_show_mounts)

        self.cb_dark_mode = QCheckBox("Dark BG")
        self.cb_dark_mode.setStyleSheet(cb_style)
        self.cb_dark_mode.setChecked(self.dark_mode)
        self.cb_dark_mode.setToolTip("Toggle dark background design in 3D visualizer.")
        self.cb_dark_mode.toggled.connect(self.set_dark_mode)
        strip_main.addWidget(self.cb_dark_mode)

        strip_main.addWidget(create_sep())

        # Simulation & Tools
        self.btn_trace_3d = QPushButton("Trace 3D")
        self.btn_trace_3d.setStyleSheet(accent_btn_style)
        self.btn_trace_3d.setToolTip("Generate and trace a 3D volumetric ray bundle through the system.")
        self.btn_trace_3d.clicked.connect(self.trace_3d_rays)
        strip_main.addWidget(self.btn_trace_3d)

        self.btn_color_mode = QPushButton("Color")
        self.btn_color_mode.setStyleSheet(btn_style)
        self.btn_color_mode.setToolTip("Toggle ray coloring: Wavelength (dispersion) or Field Index.")
        self.btn_color_mode.clicked.connect(self.toggle_color_mode)
        strip_main.addWidget(self.btn_color_mode)

        self.btn_mech_state = QPushButton("MECH: ON")
        self.btn_mech_state.setStyleSheet(
            "QPushButton { background: #dcfce7; color: #166534; border: 1px solid #86efac; border-radius: 4px; padding: 2px 7px; font-weight: bold; font-size: 11px; min-height: 20px; } "
            "QPushButton:hover { background: #bbf7d0; }"
        )
        self.btn_mech_state.setToolTip("Toggle mechanism between active states (e.g. DMD, galvos, filters).")
        self.btn_mech_state.clicked.connect(self.toggle_mech_state)
        strip_main.addWidget(self.btn_mech_state)
        self.btn_dmd_state = self.btn_mech_state
        self.btn_dmd = self.btn_mech_state

        self.btn_grating_fan = QPushButton("Grating Fan")
        self.btn_grating_fan.setStyleSheet(btn_style)
        self.btn_grating_fan.setToolTip("Zoom onto the traced wavelength-dependent diffraction fan at the grating.")
        self.btn_grating_fan.clicked.connect(self.focus_grating_diffraction)
        strip_main.addWidget(self.btn_grating_fan)

        strip_main.addStretch(1)

        self.btn_toggle_cad = QPushButton("⚙ CAD & Tolerances ▾")
        self.btn_toggle_cad.setCheckable(True)
        self.btn_toggle_cad.setChecked(True)
        self.btn_toggle_cad.setStyleSheet(toggle_cad_style)
        self.btn_toggle_cad.setToolTip("Show/hide CAD mounting parameters, pocket tolerances, and element displacement controls.")
        strip_main.addWidget(self.btn_toggle_cad)

        self.btn_export_step = QPushButton("Export STEP...")
        self.btn_export_step.setStyleSheet(step_btn_style)
        self.btn_export_step.setToolTip("Export chosen optical elements into a 3D STEP CAD assembly (.step).")
        self.btn_export_step.clicked.connect(lambda: self.export_step_dialog())
        strip_main.addWidget(self.btn_export_step)

        layout.addWidget(container_main)

        # Row 2: Secondary Collapsible Strip for CAD, Mounting, Tolerances & Alignment
        self.strip_cad_widget = QWidget()
        self.strip_cad_widget.setStyleSheet("background-color: #f8fafc; border-bottom: 1px solid #e2e8f0;")
        strip_cad = QHBoxLayout(self.strip_cad_widget)
        strip_cad.setContentsMargins(6, 3, 6, 3)
        strip_cad.setSpacing(5)

        # 1. Air Gap Cylinder Diameter
        self.lbl_air_gap_diam = QLabel("Air Gap Ø:")
        self.lbl_air_gap_diam.setStyleSheet(lbl_style)
        self.lbl_air_gap_diam.setToolTip("Fixed diameter for all air gap cylinders throughout the optical path (e.g. 25.4 mm / 1 inch).")
        strip_cad.addWidget(self.lbl_air_gap_diam)

        self.combo_air_gap_diam = QComboBox()
        self.combo_air_gap_diam.setEditable(True)
        self.combo_air_gap_diam.setStyleSheet(combo_style)
        self.combo_air_gap_diam.addItems([
            "25.4 mm (1\")",
            "12.7 mm (1/2\")",
            "20.0 mm",
            "25.0 mm",
            "30.0 mm (Cage)",
            "38.1 mm (1.5\")",
            "50.0 mm",
            "50.8 mm (2\")",
        ])
        self.combo_air_gap_diam.setToolTip("Cylinder diameter for air gaps throughout the optical path.")
        self.combo_air_gap_diam.currentTextChanged.connect(self._on_air_gap_diam_changed)
        strip_cad.addWidget(self.combo_air_gap_diam)

        strip_cad.addWidget(create_sep())

        # 2. Mounts Outer Diameter & Z Thickness
        self.lbl_mount_diam = QLabel("Mount Ø:")
        self.lbl_mount_diam.setStyleSheet(lbl_style)
        self.lbl_mount_diam.setToolTip("Outer diameter of optical mounts (e.g. 30.5 mm for SM1, or Auto).")
        self.lbl_mount_diam.setEnabled(self.show_mounts)
        strip_cad.addWidget(self.lbl_mount_diam)

        self.combo_mount_diam = QComboBox()
        self.combo_mount_diam.setEditable(True)
        self.combo_mount_diam.setStyleSheet(combo_style)
        self.combo_mount_diam.addItems([
            "30.5 mm (SM1)",
            "Auto (+5 mm)",
            "25.4 mm (1\")",
            "28.0 mm",
            "30.0 mm",
            "32.0 mm",
            "35.0 mm",
            "50.0 mm",
            "56.0 mm (SM2)",
        ])
        self.combo_mount_diam.setEnabled(self.show_mounts)
        self.combo_mount_diam.currentTextChanged.connect(self._on_mount_diam_changed)
        strip_cad.addWidget(self.combo_mount_diam)

        self.lbl_mount_z = QLabel("Z:")
        self.lbl_mount_z.setStyleSheet(lbl_style)
        self.lbl_mount_z.setToolTip("Axial thickness / Z size of optical mounts.")
        self.lbl_mount_z.setEnabled(self.show_mounts)
        strip_cad.addWidget(self.lbl_mount_z)

        self.combo_mount_z = QComboBox()
        self.combo_mount_z.setEditable(True)
        self.combo_mount_z.setStyleSheet(combo_style)
        self.combo_mount_z.addItems([
            "Auto (Lens Z)",
            "3.0 mm",
            "5.0 mm",
            "8.0 mm",
            "10.0 mm",
            "12.0 mm",
            "15.0 mm",
            "20.0 mm",
        ])
        self.combo_mount_z.setEnabled(self.show_mounts)
        self.combo_mount_z.currentTextChanged.connect(self._on_mount_z_changed)
        strip_cad.addWidget(self.combo_mount_z)

        strip_cad.addWidget(create_sep())

        # 3. Pocket Tolerance Thickness
        self.lbl_tol_z = QLabel("Tol Z:")
        self.lbl_tol_z.setStyleSheet("font-size: 11px; font-weight: 600; color: #0284c7;")
        self.lbl_tol_z.setToolTip("Axial pocket tolerance thickness (+/- mm) for element pockets in CAD / 3D printing.")
        strip_cad.addWidget(self.lbl_tol_z)

        self.combo_tol_z = QComboBox()
        self.combo_tol_z.setEditable(True)
        self.combo_tol_z.setStyleSheet(combo_style)
        self.combo_tol_z.addItems([
            "±1.0 mm",
            "±0.0 mm (Exact)",
            "±0.2 mm",
            "±0.5 mm",
            "±1.5 mm",
            "±2.0 mm",
            "±3.0 mm",
            "±5.0 mm",
        ])
        self.combo_tol_z.setToolTip("Axial tolerance thickness (+/- mm) for 3D mounting and CAD pocket clearance.")
        self.combo_tol_z.currentTextChanged.connect(self._on_tol_z_changed)
        strip_cad.addWidget(self.combo_tol_z)

        self.cb_show_tol = QCheckBox("Tol Bounds")
        self.cb_show_tol.setStyleSheet("font-size: 11px; font-weight: 500; color: #0284c7;")
        self.cb_show_tol.setChecked(self.show_tolerance_envelope)
        self.cb_show_tol.setToolTip("Toggle visual 3D display of +/- tolerance pocket bounds around optical elements.")
        self.cb_show_tol.toggled.connect(self.set_show_tolerance_envelope)
        strip_cad.addWidget(self.cb_show_tol)

        strip_cad.addWidget(create_sep())

        # 4. Element Axial Displacement (+/- mm)
        self.lbl_displace = QLabel("Displace:")
        self.lbl_displace.setStyleSheet("font-size: 11px; font-weight: 600; color: #166534;")
        self.lbl_displace.setToolTip("Select element to displace along its local optical axis for alignment.")
        strip_cad.addWidget(self.lbl_displace)

        self.combo_displace_elem = QComboBox()
        self.combo_displace_elem.setStyleSheet(combo_style)
        self.combo_displace_elem.addItem("All Elements", -1)
        self.combo_displace_elem.currentIndexChanged.connect(self._on_displace_elem_changed)
        strip_cad.addWidget(self.combo_displace_elem)

        self.spn_displace_z = QDoubleSpinBox()
        self.spn_displace_z.setRange(-25.0, 25.0)
        self.spn_displace_z.setSingleStep(0.5)
        self.spn_displace_z.setDecimals(2)
        self.spn_displace_z.setPrefix("ΔZ: ")
        self.spn_displace_z.setSuffix(" mm")
        self.spn_displace_z.setValue(0.0)
        self.spn_displace_z.setStyleSheet(
            "QDoubleSpinBox { background-color: #ffffff; border: 1px solid #bbf7d0; border-radius: 4px; "
            "padding: 1px 4px; font-size: 11px; font-weight: 600; color: #166534; min-height: 20px; }"
        )
        self.spn_displace_z.setToolTip("Displace selected element along its optical axis by +/- mm.")
        self.spn_displace_z.valueChanged.connect(self._on_displace_z_changed)
        strip_cad.addWidget(self.spn_displace_z)

        self.btn_displace_reset = QPushButton("↺ 0")
        self.btn_displace_reset.setStyleSheet(btn_style)
        self.btn_displace_reset.setToolTip("Reset displacement to 0.0 mm.")
        self.btn_displace_reset.clicked.connect(self._on_displace_reset_clicked)
        strip_cad.addWidget(self.btn_displace_reset)

        # 5. Advanced rendering options in CAD strip
        strip_cad.addWidget(create_sep())

        self.combo_aspect = QComboBox()
        self.combo_aspect.setStyleSheet(combo_style)
        self.combo_aspect.addItems(["Wide Box", "Physical 1:1", "Cubic Box"])
        self.combo_aspect.currentIndexChanged.connect(self._on_aspect_combo_changed)
        strip_cad.addWidget(self.combo_aspect)

        self.combo_detail = QComboBox()
        self.combo_detail.setStyleSheet(combo_style)
        self.combo_detail.addItems(["Fast 3D", "Detailed 3D"])
        self.combo_detail.currentIndexChanged.connect(self._on_detail_combo_changed)
        strip_cad.addWidget(self.combo_detail)

        self.cb_show_apertures = QCheckBox("Stops")
        self.cb_show_apertures.setStyleSheet(cb_style)
        self.cb_show_apertures.setChecked(self.show_apertures)
        self.cb_show_apertures.setToolTip("Toggle 3D aperture stop diaphragms and clear aperture rims.")
        self.cb_show_apertures.toggled.connect(self.set_show_apertures)
        strip_cad.addWidget(self.cb_show_apertures)

        self.cb_no_transparency = QCheckBox("Opaque")
        self.cb_no_transparency.setStyleSheet(cb_style)
        self.cb_no_transparency.setChecked(self.no_transparency)
        self.cb_no_transparency.setToolTip("Toggle solid opaque rendering (alpha=1.0, high-contrast edges).")
        self.cb_no_transparency.toggled.connect(self.set_no_transparency)
        strip_cad.addWidget(self.cb_no_transparency)

        strip_cad.addStretch(1)

        layout.addWidget(self.strip_cad_widget)

        def _on_toggle_cad(checked):
            self.strip_cad_widget.setVisible(checked)
            self.btn_toggle_cad.setText("⚙ CAD & Tolerances ▾" if checked else "⚙ CAD & Tolerances ▸")

        self.btn_toggle_cad.toggled.connect(_on_toggle_cad)

        # Backward compatibility aliases & orbit buttons
        self.btn_orbit_left = QPushButton("⟲")
        self.btn_orbit_left.clicked.connect(lambda: self.orbit_horizontal(-15.0, animate=True))
        self.btn_orbit_right = QPushButton("⟳")
        self.btn_orbit_right.clicked.connect(lambda: self.orbit_horizontal(15.0, animate=True))
        self.btn_flip_view = QPushButton("Flip")
        self.btn_flip_view.clicked.connect(lambda: self.flip_view(animate=True))
        self.lbl_status = QLabel("")

        # Clean 3D Canvas
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.fig.patch.set_facecolor('#ffffff')
        self.fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
        self.canvas = FigureCanvas(self.fig)
        if QT_AVAILABLE and hasattr(self.canvas, 'setSizePolicy'):
            try:
                expanding = getattr(getattr(QtWidgets, 'QSizePolicy', None), 'Expanding', None)
                if expanding is not None:
                    self.canvas.setSizePolicy(QtWidgets.QSizePolicy(expanding, expanding))
            except Exception:
                pass

        # NavigationToolbar omitted for clean minimalist CAD viewport
        self.toolbar = None

        layout.addWidget(self.canvas, 1)

        self.ax = self.fig.add_subplot(111, projection='3d')
        if hasattr(self.ax, 'mouse_init'):
            try:
                self.ax.mouse_init(rotate_btn=1, pan_btn=2, zoom_btn=3)
            except Exception:
                pass
        if hasattr(self.ax, '_dist'):
            self.ax._dist = self._cam_dist

        self.set_view_isometric()
        self.refresh_layout()

        # Keyboard Focus Policies for Blender Hotkeys
        if QT_AVAILABLE and hasattr(self, 'setFocusPolicy') and Qt is not None:
            try:
                self.setFocusPolicy(getattr(Qt, 'StrongFocus', 11))
            except Exception:
                pass
        if self.canvas is not None and hasattr(self.canvas, 'setFocusPolicy') and Qt is not None:
            try:
                self.canvas.setFocusPolicy(getattr(Qt, 'StrongFocus', 11))
            except Exception:
                pass

        self.canvas.mpl_connect('button_press_event', self.on_button_press)
        self.canvas.mpl_connect('pick_event', self.on_pick)
        self.canvas.mpl_connect('button_release_event', self.on_button_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.canvas.mpl_connect('key_press_event', self.on_mpl_key_press)

    def set_show_rays(self, show: bool) -> None:
        self.show_rays = bool(show)
        if self.cb_show_rays and hasattr(self.cb_show_rays, "isChecked") and self.cb_show_rays.isChecked() != self.show_rays:
            self.cb_show_rays.blockSignals(True)
            self.cb_show_rays.setChecked(self.show_rays)
            self.cb_show_rays.blockSignals(False)
        self.update_layout()

    def export_step_dialog(self, file_path: Optional[str] = None) -> bool:
        """
        Open the interactive STEP Export pop-up window to choose which elements to export together.
        """
        stack = None
        if self.model is not None:
            if hasattr(self.model, "get_surface_stack"):
                stack = self.model.get_surface_stack()
            elif hasattr(self.model, "stack"):
                stack = self.model.stack
        if stack is None:
            stack = self.surface_stack

        if stack is None or not getattr(stack, "surfaces", None):
            if QT_AVAILABLE and hasattr(self, "isVisible") and self.isVisible():
                try:
                    from optics_app.gui.qt_compat import QMessageBox
                    QMessageBox.warning(self, "Export STEP", "No optical surfaces available in 3D scene.")
                except Exception:
                    pass
            return False

        from optics_app.gui.step_export_dialog import STEPExportDialog
        dialog = STEPExportDialog(
            stack, parent=self, default_file_path=file_path,
            default_air_gap_diam=getattr(self, "air_gap_constant_diameter", 0.0),
            default_mount_diam=getattr(self, "mount_diameter", 30.5),
            default_mount_z=getattr(self, "mount_z_thickness", 0.0),
            default_export_mounts=getattr(self, "show_mounts", False),
            default_tol_z=getattr(self, "tolerance_thickness", 1.0),
        )
        res = dialog.exec_() if hasattr(dialog, "exec_") else dialog.exec()
        if not res:
            return False

        out_path = dialog.get_file_path()
        selected_indices = dialog.get_selected_element_indices()
        engine = dialog.get_engine()
        export_enclosure = dialog.get_export_enclosure()
        air_gap_diam = dialog.get_air_gap_constant_diameter()
        mount_diam = dialog.get_mount_diameter()
        mount_z = dialog.get_mount_z_thickness()
        export_mounts = dialog.get_export_mounts()
        tol_z = dialog.get_tolerance_thickness() if hasattr(dialog, "get_tolerance_thickness") else getattr(self, "tolerance_thickness", 1.0)
        if abs(getattr(self, "air_gap_constant_diameter", 0.0) - air_gap_diam) > 1e-4:
            self.set_air_gap_constant_diameter(air_gap_diam)
        if abs(getattr(self, "mount_diameter", 30.5) - mount_diam) > 1e-4:
            self.set_mount_diameter(mount_diam)
        if abs(getattr(self, "mount_z_thickness", 0.0) - mount_z) > 1e-4:
            self.set_mount_z_thickness(mount_z)
        if abs(getattr(self, "tolerance_thickness", 1.0) - tol_z) > 1e-4:
            self.set_tolerance_thickness(tol_z)

        from optics_app.cad.exporter import CADExporter, STEPExportOptions, STEPExportEngine
        opt_engine = STEPExportEngine(engine) if isinstance(engine, str) and engine in ("auto", "build123d", "pure_python") else STEPExportEngine.AUTO
        options = STEPExportOptions(
            engine=opt_engine,
            selected_element_indices=selected_indices,
            export_3d_printable_enclosure=export_enclosure,
            enclosure_wall_margin_mm=10.0,
            air_gap_cylinder_diameter=air_gap_diam,
            mount_diameter=mount_diam,
            mount_z_thickness=mount_z,
            export_mechanical_mounts=export_mounts,
            tolerance_thickness_mm=tol_z,
            element_displacements=dict(self.element_displacements),
        )

        export_result = CADExporter.export_system_step(
            stack,
            out_path,
            selected_element_indices=selected_indices,
            engine=engine,
            options=options,
        )

        if export_result.is_success:
            msg = f"Exported {export_result.num_components_exported} components to {os.path.basename(out_path)}"
            if self.lbl_status:
                self.lbl_status.setText(f"📦 {msg}")
            if QT_AVAILABLE and hasattr(self, "isVisible") and self.isVisible():
                try:
                    from optics_app.gui.qt_compat import QMessageBox
                    QMessageBox.information(
                        self,
                        "STEP Export Complete",
                        f"Successfully exported 3D assembly:\n{out_path}\n({export_result.num_components_exported} components)"
                    )
                except Exception:
                    pass
            return True
        else:
            err = export_result.error_message or "Unknown export error"
            if self.lbl_status:
                self.lbl_status.setText(f"❌ STEP Export Error: {err}")
            if QT_AVAILABLE and hasattr(self, "isVisible") and self.isVisible():
                try:
                    from optics_app.gui.qt_compat import QMessageBox
                    QMessageBox.critical(self, "STEP Export Error", f"Failed to export STEP file:\n{err}")
                except Exception:
                    pass
            return False

    def get_view_settings(self) -> Dict[str, Any]:
        """Serializable, instrument-independent display settings for a project."""
        return {
            "show_rays": self.show_rays,
            "show_labels": self.show_labels,
            "show_apertures": self.show_apertures,
            "show_grid": self.show_grid,
            "show_air_gaps": self.show_air_gaps,
            "air_gap_constant_diameter": self.air_gap_constant_diameter,
            "show_mounts": self.show_mounts,
            "mount_diameter": self.mount_diameter,
            "mount_z_thickness": self.mount_z_thickness,
            "tolerance_thickness": self.tolerance_thickness,
            "show_tolerance_envelope": self.show_tolerance_envelope,
            "global_displacement_z": self.global_displacement_z,
            "element_displacements": dict(self.element_displacements),
            "opaque": self.no_transparency,
            "dark_mode": self.dark_mode,
            "aspect_mode": self.aspect_mode,
            "mesh_quality": self.mesh_quality,
            "projection_mode": self.projection_mode,
            "color_mode": self.color_mode,
            "camera_elevation_deg": float(getattr(self.ax, "elev", 20.0)),
            "camera_azimuth_deg": float(getattr(self.ax, "azim", -60.0)),
        }

    def set_view_settings(self, settings: Optional[Dict[str, Any]]) -> None:
        """Restore valid saved choices, ignoring unknown keys from newer projects."""
        data = settings if isinstance(settings, dict) else {}
        defaults = {
            "show_rays": True, "show_labels": True, "show_apertures": True,
            "show_grid": True, "opaque": False, "dark_mode": False, "aspect_mode": "wide",
            "mesh_quality": "interactive", "projection_mode": "persp",
            "color_mode": "wavelength",
        }
        for key, default in defaults.items():
            value = data.get(key, default)
            if isinstance(default, bool):
                value = value if isinstance(value, bool) else default
            elif key == "aspect_mode" and value not in ("wide", "physical", "cubic"):
                value = default
            elif key == "mesh_quality" and value not in ("interactive", "detailed"):
                value = default
            elif key == "projection_mode" and value not in ("persp", "ortho"):
                value = default
            elif key == "color_mode" and value not in ("wavelength", "field"):
                value = default
            if key == "opaque":
                self.no_transparency = value
            elif key == "dark_mode":
                self.dark_mode = value
            else:
                setattr(self, key, value)

        if "tolerance_thickness" in data:
            self.set_tolerance_thickness(float(data["tolerance_thickness"]))
        if "show_tolerance_envelope" in data:
            self.set_show_tolerance_envelope(bool(data["show_tolerance_envelope"]))
        if "global_displacement_z" in data:
            self.global_displacement_z = float(data["global_displacement_z"])
        if "element_displacements" in data and isinstance(data["element_displacements"], dict):
            self.element_displacements = {int(k): float(v) for k, v in data["element_displacements"].items()}

        for checkbox, value in (
            (self.cb_show_rays, self.show_rays), (self.cb_show_labels, self.show_labels),
            (self.cb_show_apertures, self.show_apertures), (self.cb_show_grid, self.show_grid),
            (self.cb_no_transparency, self.no_transparency), (self.cb_dark_mode, self.dark_mode),
            (self.cb_show_mounts, self.show_mounts),
            (self.cb_show_tol, self.show_tolerance_envelope),
        ):
            if checkbox is not None:
                checkbox.blockSignals(True)
                checkbox.setChecked(value)
                checkbox.blockSignals(False)
        for combo, index in (
            (self.combo_aspect, ("wide", "physical", "cubic").index(self.aspect_mode)),
            (self.combo_detail, ("interactive", "detailed").index(self.mesh_quality)),
        ):
            if combo is not None:
                combo.blockSignals(True)
                combo.setCurrentIndex(index)
                combo.blockSignals(False)
        if self.btn_color_mode is not None:
            self.btn_color_mode.setText("Field" if self.color_mode == "field" else "Wavelength")
        if self.btn_toggle_proj is not None:
            self.btn_toggle_proj.setText("Ortho" if self.projection_mode == "ortho" else "Persp")
        if self.ax is not None:
            if hasattr(self.ax, "set_proj_type"):
                self.ax.set_proj_type(self.projection_mode)
            try:
                elev = float(data.get("camera_elevation_deg", 20.0))
                azim = float(data.get("camera_azimuth_deg", -60.0))
                if np.isfinite(elev) and np.isfinite(azim):
                    self.ax.view_init(elev=elev, azim=azim)
            except (TypeError, ValueError):
                pass
        bg_fig = '#0f172a' if self.dark_mode else '#ffffff'
        if self.fig is not None:
            self.fig.patch.set_facecolor(bg_fig)
        self.transparency_changed.emit(self.no_transparency)
        self.update_layout(force_rebuild=True)

    def set_mesh_quality(self, quality: str) -> None:
        selected = str(quality).lower()
        if selected not in ("interactive", "detailed"):
            raise ValueError("3D mesh quality must be interactive or detailed.")
        if self.mesh_quality == selected:
            return
        self.mesh_quality = selected
        if self.combo_detail is not None:
            wanted = 0 if selected == "interactive" else 1
            if self.combo_detail.currentIndex() != wanted:
                self.combo_detail.blockSignals(True)
                self.combo_detail.setCurrentIndex(wanted)
                self.combo_detail.blockSignals(False)
        self.update_layout()

    def _on_detail_combo_changed(self, index: int) -> None:
        self.set_mesh_quality("interactive" if index == 0 else "detailed")

    def set_dark_mode(self, enabled: bool) -> None:
        """Toggle dark background design in 3D visualizer."""
        self.dark_mode = bool(enabled)
        if self.cb_dark_mode and hasattr(self.cb_dark_mode, "isChecked") and self.cb_dark_mode.isChecked() != self.dark_mode:
            self.cb_dark_mode.blockSignals(True)
            self.cb_dark_mode.setChecked(self.dark_mode)
            self.cb_dark_mode.blockSignals(False)
        bg_fig = '#0f172a' if self.dark_mode else '#ffffff'
        if self.fig is not None:
            self.fig.patch.set_facecolor(bg_fig)
        if hasattr(self, 'container_main') and self.container_main is not None:
            self.container_main.setStyleSheet(
                "background-color: #0f172a; border-bottom: 1px solid #334155;" if self.dark_mode
                else "background-color: #ffffff; border-bottom: 1px solid #e2e8f0;"
            )
        if hasattr(self, 'strip_cad_widget') and self.strip_cad_widget is not None:
            self.strip_cad_widget.setStyleSheet(
                "background-color: #1e293b; border-bottom: 1px solid #334155;" if self.dark_mode
                else "background-color: #f8fafc; border-bottom: 1px solid #e2e8f0;"
            )
        cb_col = '#cbd5e1' if self.dark_mode else '#475569'
        hover_col = '#f1f5f9' if self.dark_mode else '#0f172a'
        cb_style = (
            f"QCheckBox {{ font-size: 11px; font-weight: 500; color: {cb_col}; spacing: 4px; }} "
            f"QCheckBox:hover {{ color: {hover_col}; }}"
        )
        for cb in [
            getattr(self, 'cb_show_rays', None),
            getattr(self, 'cb_show_labels', None),
            getattr(self, 'cb_show_grid', None),
            getattr(self, 'cb_show_air_gaps', None),
            getattr(self, 'cb_show_mounts', None),
            getattr(self, 'cb_dark_mode', None),
            getattr(self, 'cb_show_apertures', None),
            getattr(self, 'cb_no_transparency', None),
        ]:
            if cb is not None:
                cb.setStyleSheet(cb_style)
        if hasattr(self, 'dark_mode_changed'):
            try:
                self.dark_mode_changed.emit(self.dark_mode)
            except Exception:
                pass
        self.update_layout(force_rebuild=True)

    def set_show_marginal_chief(self, enabled: bool) -> None:
        """Toggle highlighting of marginal and chief rays in 3D visualizer."""
        self.show_marginal_chief = bool(enabled)
        self.update_layout()

    def set_marginal_chief_only(self, enabled: bool) -> None:
        """Filter 3D visualizer to show exclusively marginal and chief rays."""
        self.marginal_chief_only = bool(enabled)
        self.update_layout()

    def set_show_labels(self, show: bool) -> None:
        self.show_labels = bool(show)
        if self.cb_show_labels and hasattr(self.cb_show_labels, "isChecked") and self.cb_show_labels.isChecked() != self.show_labels:
            self.cb_show_labels.blockSignals(True)
            self.cb_show_labels.setChecked(self.show_labels)
            self.cb_show_labels.blockSignals(False)
        self.update_layout()

    def set_show_apertures(self, show: bool) -> None:
        self.show_apertures = bool(show)
        if self.cb_show_apertures and hasattr(self.cb_show_apertures, "isChecked") and self.cb_show_apertures.isChecked() != self.show_apertures:
            self.cb_show_apertures.blockSignals(True)
            self.cb_show_apertures.setChecked(self.show_apertures)
            self.cb_show_apertures.blockSignals(False)
        self.update_layout()

    def set_show_air_gaps(self, show: bool) -> None:
        self.show_air_gaps = bool(show)
        if self.cb_show_air_gaps and hasattr(self.cb_show_air_gaps, "isChecked") and self.cb_show_air_gaps.isChecked() != self.show_air_gaps:
            self.cb_show_air_gaps.blockSignals(True)
            self.cb_show_air_gaps.setChecked(self.show_air_gaps)
            self.cb_show_air_gaps.blockSignals(False)
        if self.combo_air_gap_diam and hasattr(self.combo_air_gap_diam, "setEnabled"):
            self.combo_air_gap_diam.setEnabled(self.show_air_gaps)
        if self.lbl_air_gap_diam and hasattr(self.lbl_air_gap_diam, "setEnabled"):
            self.lbl_air_gap_diam.setEnabled(self.show_air_gaps)
        self.update_layout()

    def _on_air_gap_diam_changed(self, text: str) -> None:
        """Parse combo box text and dynamically update air gap constant cylinder diameter."""
        import re
        if not text:
            return
        match = re.search(r"[-+]?\d*\.?\d+", text)
        if match:
            try:
                val = float(match.group(0))
                if val <= 0.0:
                    val = 25.4
            except ValueError:
                val = 25.4
        else:
            val = 25.4
        if abs(getattr(self, "air_gap_constant_diameter", 25.4) - val) > 1e-4:
            self.air_gap_constant_diameter = val
            self.update_layout()

    def set_air_gap_constant_diameter(self, diameter_mm: float) -> None:
        """Set constant fixed diameter (in mm) for all air gap cylinders throughout the optical path."""
        val = float(diameter_mm) if float(diameter_mm) > 0.0 else 25.4
        if abs(getattr(self, "air_gap_constant_diameter", 25.4) - val) < 1e-4:
            return
        self.air_gap_constant_diameter = val
        if hasattr(self, "combo_air_gap_diam") and self.combo_air_gap_diam:
            self.combo_air_gap_diam.blockSignals(True)
            found = False
            import re
            for idx in range(self.combo_air_gap_diam.count()):
                item_text = self.combo_air_gap_diam.itemText(idx)
                m = re.search(r"[-+]?\d*\.?\d+", item_text)
                if m and abs(float(m.group(0)) - val) < 1e-3:
                    self.combo_air_gap_diam.setCurrentIndex(idx)
                    found = True
                    break
            if not found:
                self.combo_air_gap_diam.setEditText(f"{val:.1f} mm")
            self.combo_air_gap_diam.blockSignals(False)
        self.update_layout()

    def set_show_mounts(self, show: bool) -> None:
        """Toggle visualization of mechanical mounts / cells around optical elements."""
        self.show_mounts = bool(show)
        if self.cb_show_mounts and hasattr(self.cb_show_mounts, "isChecked") and self.cb_show_mounts.isChecked() != self.show_mounts:
            self.cb_show_mounts.blockSignals(True)
            self.cb_show_mounts.setChecked(self.show_mounts)
            self.cb_show_mounts.blockSignals(False)
        if self.combo_mount_diam and hasattr(self.combo_mount_diam, "setEnabled"):
            self.combo_mount_diam.setEnabled(self.show_mounts)
        if self.lbl_mount_diam and hasattr(self.lbl_mount_diam, "setEnabled"):
            self.lbl_mount_diam.setEnabled(self.show_mounts)
        if self.combo_mount_z and hasattr(self.combo_mount_z, "setEnabled"):
            self.combo_mount_z.setEnabled(self.show_mounts)
        if self.lbl_mount_z and hasattr(self.lbl_mount_z, "setEnabled"):
            self.lbl_mount_z.setEnabled(self.show_mounts)
        self.update_layout()

    def _on_mount_diam_changed(self, text: str) -> None:
        """Parse combo box text and dynamically update mount outer diameter."""
        import re
        if not text:
            return
        if "auto" in text.lower():
            val = 0.0
        else:
            match = re.search(r"[-+]?\d*\.?\d+", text)
            if match:
                try:
                    val = float(match.group(0))
                    if val <= 0.0:
                        val = 30.5
                except ValueError:
                    val = 30.5
            else:
                val = 30.5
        if abs(getattr(self, "mount_diameter", 30.5) - val) > 1e-4:
            self.mount_diameter = val
            self.update_layout()

    def set_mount_diameter(self, diameter_mm: float) -> None:
        """Set outer diameter (in mm) for optical element mounts/cells (0.0 = Auto)."""
        val = float(diameter_mm) if float(diameter_mm) >= 0.0 else 30.5
        if abs(getattr(self, "mount_diameter", 30.5) - val) < 1e-4:
            return
        self.mount_diameter = val
        if hasattr(self, "combo_mount_diam") and self.combo_mount_diam:
            self.combo_mount_diam.blockSignals(True)
            found = False
            import re
            for idx in range(self.combo_mount_diam.count()):
                item_text = self.combo_mount_diam.itemText(idx)
                if val == 0.0 and "auto" in item_text.lower():
                    self.combo_mount_diam.setCurrentIndex(idx)
                    found = True
                    break
                m = re.search(r"[-+]?\d*\.?\d+", item_text)
                if m and abs(float(m.group(0)) - val) < 1e-3:
                    self.combo_mount_diam.setCurrentIndex(idx)
                    found = True
                    break
            if not found:
                self.combo_mount_diam.setEditText(f"{val:.1f} mm")
            self.combo_mount_diam.blockSignals(False)
        self.update_layout()

    def _on_mount_z_changed(self, text: str) -> None:
        """Parse combo box text and dynamically update mount Z axial thickness."""
        import re
        if not text:
            return
        if "auto" in text.lower():
            val = 0.0
        else:
            match = re.search(r"[-+]?\d*\.?\d+", text)
            if match:
                try:
                    val = float(match.group(0))
                    if val < 0.0:
                        val = 0.0
                except ValueError:
                    val = 0.0
            else:
                val = 0.0
        if abs(getattr(self, "mount_z_thickness", 0.0) - val) > 1e-4:
            self.mount_z_thickness = val
            self.update_layout()

    def set_mount_z_thickness(self, thickness_mm: float) -> None:
        """Set axial Z thickness (in mm) for optical element mounts (0.0 = Auto)."""
        val = float(thickness_mm) if float(thickness_mm) >= 0.0 else 0.0
        if abs(getattr(self, "mount_z_thickness", 0.0) - val) < 1e-4:
            return
        self.mount_z_thickness = val
        if hasattr(self, "combo_mount_z") and self.combo_mount_z:
            self.combo_mount_z.blockSignals(True)
            found = False
            import re
            for idx in range(self.combo_mount_z.count()):
                item_text = self.combo_mount_z.itemText(idx)
                if val == 0.0 and "auto" in item_text.lower():
                    self.combo_mount_z.setCurrentIndex(idx)
                    found = True
                    break
                m = re.search(r"[-+]?\d*\.?\d+", item_text)
                if m and abs(float(m.group(0)) - val) < 1e-3:
                    self.combo_mount_z.setCurrentIndex(idx)
                    found = True
                    break
            if not found:
                self.combo_mount_z.setEditText(f"{val:.1f} mm")
            self.combo_mount_z.blockSignals(False)
        self.update_layout()

    def set_tolerance_thickness(self, tol_mm: float) -> None:
        """Set axial tolerance thickness (+/- mm) for element pockets and mounting adjustment range."""
        val = max(0.0, float(tol_mm))
        if abs(getattr(self, "tolerance_thickness", 1.0) - val) < 1e-4:
            return
        self.tolerance_thickness = val
        if hasattr(self, "combo_tol_z") and self.combo_tol_z:
            self.combo_tol_z.blockSignals(True)
            found = False
            import re
            for idx in range(self.combo_tol_z.count()):
                item_text = self.combo_tol_z.itemText(idx)
                m = re.search(r"[-+]?\d*\.?\d+", item_text)
                if m and abs(float(m.group(0)) - val) < 1e-3:
                    self.combo_tol_z.setCurrentIndex(idx)
                    found = True
                    break
            if not found:
                self.combo_tol_z.setEditText(f"±{val:.1f} mm")
            self.combo_tol_z.blockSignals(False)
        self.update_layout()

    def _on_tol_z_changed(self, text: str) -> None:
        """Parse combo box text and dynamically update axial tolerance thickness."""
        import re
        if not text:
            return
        match = re.search(r"[-+]?\d*\.?\d+", text)
        if match:
            try:
                val = max(0.0, float(match.group(0)))
            except ValueError:
                val = 1.0
        else:
            val = 1.0
        if abs(getattr(self, "tolerance_thickness", 1.0) - val) > 1e-4:
            self.tolerance_thickness = val
            self.update_layout()

    def set_show_tolerance_envelope(self, show: bool) -> None:
        """Toggle visual 3D display of +/- tolerance pocket bounds around optical elements."""
        self.show_tolerance_envelope = bool(show)
        if self.cb_show_tol and hasattr(self.cb_show_tol, "isChecked") and self.cb_show_tol.isChecked() != self.show_tolerance_envelope:
            self.cb_show_tol.blockSignals(True)
            self.cb_show_tol.setChecked(self.show_tolerance_envelope)
            self.cb_show_tol.blockSignals(False)
        self.update_layout()

    def get_element_displacement(self, elem_idx: int) -> float:
        """Get axial displacement (+/- mm) for element by index, or global if -1."""
        if elem_idx == -1:
            return getattr(self, "global_displacement_z", 0.0)
        return self.element_displacements.get(int(elem_idx), 0.0)

    def set_element_displacement(self, elem_idx: int, dz_mm: float) -> None:
        """Set axial displacement (+/- mm) for element by index, or global if -1."""
        val = float(dz_mm)
        if elem_idx == -1:
            self.global_displacement_z = val
        else:
            self.element_displacements[int(elem_idx)] = val
        if hasattr(self, "combo_displace_elem") and self.combo_displace_elem:
            curr_sel = self.combo_displace_elem.currentData()
            if curr_sel == elem_idx and hasattr(self, "spn_displace_z") and self.spn_displace_z:
                self.spn_displace_z.blockSignals(True)
                self.spn_displace_z.setValue(val)
                self.spn_displace_z.blockSignals(False)
        self.update_layout()

    def _on_displace_elem_changed(self, combo_idx: int) -> None:
        """Called when user changes the target element in combo_displace_elem."""
        if not hasattr(self, "combo_displace_elem") or not self.combo_displace_elem:
            return
        elem_idx = self.combo_displace_elem.currentData()
        if elem_idx is None:
            elem_idx = -1
        current_val = self.get_element_displacement(elem_idx)
        if hasattr(self, "spn_displace_z") and self.spn_displace_z:
            self.spn_displace_z.blockSignals(True)
            self.spn_displace_z.setValue(current_val)
            self.spn_displace_z.blockSignals(False)

    def _on_displace_z_changed(self, val: float) -> None:
        """Called when user edits the spinbox value."""
        elem_idx = -1
        if hasattr(self, "combo_displace_elem") and self.combo_displace_elem:
            elem_idx = self.combo_displace_elem.currentData()
            if elem_idx is None:
                elem_idx = -1
        if elem_idx == -1:
            self.global_displacement_z = float(val)
        else:
            self.element_displacements[int(elem_idx)] = float(val)
        self.update_layout()

    def _on_displace_reset_clicked(self) -> None:
        """Reset displacement of current selection (or all) to 0.0 mm."""
        elem_idx = -1
        if hasattr(self, "combo_displace_elem") and self.combo_displace_elem:
            elem_idx = self.combo_displace_elem.currentData()
            if elem_idx is None:
                elem_idx = -1
        if elem_idx == -1:
            self.global_displacement_z = 0.0
            self.element_displacements.clear()
        else:
            self.element_displacements[int(elem_idx)] = 0.0
        if hasattr(self, "spn_displace_z") and self.spn_displace_z:
            self.spn_displace_z.blockSignals(True)
            self.spn_displace_z.setValue(0.0)
            self.spn_displace_z.blockSignals(False)
        self.update_layout()

    def _update_element_displace_combo(self, elements: Optional[List[Dict[str, Any]]] = None) -> None:
        """Populate the element displacement dropdown with available optical elements."""
        if not hasattr(self, "combo_displace_elem") or self.combo_displace_elem is None:
            return
        if elements is None and self.surface_stack is not None:
            from optics_app.cad.exporter import get_exportable_elements
            try:
                elements = get_exportable_elements(self.surface_stack)
            except Exception:
                elements = []
        if elements is None:
            elements = []

        curr_data = self.combo_displace_elem.currentData()
        if curr_data is None:
            curr_data = -1

        real_elements = [e for e in elements if e.get("type") != "Air Gap"]
        new_items = [("All Elements", -1)]
        for elem in real_elements:
            e_idx = elem.get("index", 0)
            e_name = elem.get("name", f"Element {e_idx}")
            lbl = f"#{e_idx + 1}: {e_name}"
            new_items.append((lbl, e_idx))

        current_items = [(self.combo_displace_elem.itemText(i), self.combo_displace_elem.itemData(i))
                         for i in range(self.combo_displace_elem.count())]
        if current_items == new_items:
            return

        self.combo_displace_elem.blockSignals(True)
        self.combo_displace_elem.clear()
        target_idx = 0
        for i, (text, data) in enumerate(new_items):
            self.combo_displace_elem.addItem(text, data)
            if data == curr_data:
                target_idx = i
        self.combo_displace_elem.setCurrentIndex(target_idx)
        self.combo_displace_elem.blockSignals(False)


    def set_show_grid(self, show: bool) -> None:
        """Toggle 3D coordinate grid and pane visibility."""
        self.show_grid = bool(show)
        if self.cb_show_grid and hasattr(self.cb_show_grid, "isChecked") and self.cb_show_grid.isChecked() != self.show_grid:
            self.cb_show_grid.blockSignals(True)
            self.cb_show_grid.setChecked(self.show_grid)
            self.cb_show_grid.blockSignals(False)
        if self.ax is not None:
            self.ax.grid(self.show_grid, linestyle=':', alpha=0.45, color='#cbd5e1')
            for axis in (self.ax.xaxis, self.ax.yaxis, self.ax.zaxis):
                if hasattr(axis, 'set_pane_color'):
                    pane_alpha = 0.45 if self.show_grid else 0.0
                    axis.set_pane_color((0.97, 0.98, 0.99, pane_alpha))
                if hasattr(axis, 'line'):
                    axis.line.set_color('#cbd5e1' if self.show_grid else (0, 0, 0, 0))
        if hasattr(self.canvas, "draw_idle"):
            self.canvas.draw_idle()

    def toggle_color_mode(self) -> None:
        """Toggle 3D ray coloring between Wavelength and Field Index."""
        if getattr(self, "color_mode", "wavelength") == "wavelength":
            self.color_mode = "field"
            if hasattr(self, "btn_color_mode") and self.btn_color_mode:
                self.btn_color_mode.setText("Field")
                self.btn_color_mode.setToolTip("Currently coloring rays by Field Index. Click to color by Wavelength.")
        else:
            self.color_mode = "wavelength"
            if hasattr(self, "btn_color_mode") and self.btn_color_mode:
                self.btn_color_mode.setText("Wavelength")
                self.btn_color_mode.setToolTip("Currently coloring rays by Wavelength. Click to color by Field Index.")
        self.update_layout()

    def trace_3d_rays(self) -> List[Ray]:
        """Trace a volumetric 3D bundle and update the visualizer."""
        stack = self.surface_stack
        if stack is None and self.model is not None and hasattr(self.model, "get_surface_stack"):
            stack = self.model.get_surface_stack()
        if stack is not None:
            self.surface_stack = stack
            sys_cfg = getattr(self, "system_config", None)
            if sys_cfg is None and self.model is not None:
                sys_cfg = getattr(self.model, "system_config", None)
            if sys_cfg is None and hasattr(self.parent(), "system_config"):
                sys_cfg = getattr(self.parent(), "system_config", None)
            self.rays = generate_3d_rays(stack, system_config=sys_cfg)
            if self.model is not None and hasattr(self.model, "set_rays"):
                self.model.set_rays(self.rays)
            self.update_layout(surface_stack=self.surface_stack, rays=self.rays, system_config=sys_cfg)
        return self.rays

    def get_mech_surface_info(self) -> Optional[Tuple[int, Surface, str]]:
        stack = self.surface_stack
        if stack is None and self.model is not None and hasattr(self.model, "get_surface_stack"):
            stack = self.model.get_surface_stack()
        if not stack or not getattr(stack, "surfaces", None):
            return None
        from optics_app.engine.mechanisms import available_states, active_state_name
        for idx, s in enumerate(stack.surfaces):
            if available_states(s):
                return idx, s, active_state_name(s)
            if getattr(s, "galvo_config", None) is not None:
                ang = getattr(s.galvo_config, "current_angle_deg", 0.0)
                return idx, s, f"{ang:+.1f}°"
            mech = getattr(s, "mechanism", None)
            if mech is not None and hasattr(mech, "channels") and mech.channels:
                return idx, s, "ACTIVE"
        return None

    get_dmd_surface_info = get_mech_surface_info

    def toggle_dmd_state(self) -> Optional[str]:
        """Backward-compatible alias for toggle_mech_state."""
        return self.toggle_mech_state()

    def toggle_mech_state(self) -> Optional[str]:
        """Advance the selected mechanism state and re-trace its actual optical path."""
        info = self.get_mech_surface_info()
        if not info:
            if hasattr(self, "btn_mech_state") and self.btn_mech_state:
                self.btn_mech_state.setVisible(False)
            return None

        mech_idx, mech_surf, cur_state = info
        galvo_cfg = getattr(mech_surf, "galvo_config", None)
        if galvo_cfg is not None:
            scan_min = float(getattr(galvo_cfg, "scan_angle_min_deg", getattr(galvo_cfg, "min_angle_deg", -10.0)))
            scan_max = float(getattr(galvo_cfg, "scan_angle_max_deg", getattr(galvo_cfg, "max_angle_deg", 10.0)))
            cur_ang = float(getattr(galvo_cfg, "current_angle_deg", 0.0))
            step = 2.5
            next_ang = cur_ang + step
            if next_ang > scan_max + 1e-4:
                next_ang = scan_min
            galvo_cfg.current_angle_deg = round(next_ang, 2)
            if self.surface_stack:
                self.surface_stack._update_accumulated_positions()
            new_state = f"{next_ang:+.1f}°"
            self._update_mech_button_ui(new_state)
            self._trigger_retrace()
            return new_state

        states = list(available_states(mech_surf))
        if not states:
            return None
        if "ON" in states and "OFF" in states:
            new_state = "OFF" if cur_state == "ON" else "ON"
        else:
            new_state = states[(states.index(cur_state) + 1) % len(states)] if cur_state in states else states[0]

        if hasattr(self, "model") and self.model is not None and hasattr(self.model, "set_mechanical_state"):
            self.model.set_mechanical_state(mech_idx, new_state)
            if hasattr(self.model, "get_surface_stack"):
                self.surface_stack = self.model.get_surface_stack()
        elif self.surface_stack:
            from optics_app.engine.mechanisms import MechanicalStateController
            ctrl = MechanicalStateController(self.surface_stack)
            ctrl.set_state(mech_idx, new_state)
            self.surface_stack._update_accumulated_positions()

        self._update_mech_button_ui(new_state)
        self._trigger_retrace()
        return new_state

    def _trigger_retrace(self) -> None:
        p = self.parent()
        traced = False
        while p is not None:
            if hasattr(p, "request_debounced_trace"):
                p.request_debounced_trace()
                traced = True
                break
            elif hasattr(p, "on_trace_rays"):
                p.on_trace_rays()
                traced = True
                break
            p = p.parent() if hasattr(p, "parent") else None

        if not traced:
            self.trace_3d_rays()

    def _sync_mech_button_ui(self) -> None:
        if not hasattr(self, "btn_mech_state") or not self.btn_mech_state:
            return
        info = self.get_mech_surface_info()
        if not info:
            self.btn_mech_state.setVisible(False)
            return
        surface = info[1]
        try:
            from optics_app.engine.mechanisms import is_dmd
            from optics_app.engine.plm import is_plm
        except ImportError:
            from v2.engine.mechanisms import is_dmd
            from v2.engine.plm import is_plm
        if surface and (is_dmd(surface) or is_plm(surface) or getattr(surface, 'is_plm', False)):
            # TI devices (PLM / DMD) are controlled exclusively via TI Modulator Studio
            self.btn_mech_state.setVisible(False)
            self._update_mech_button_ui(info[2])
            return
        self.btn_mech_state.setVisible(True)
        self._update_mech_button_ui(info[2])

    def _update_mech_button_ui(self, state: str) -> None:
        if not hasattr(self, "btn_mech_state") or not self.btn_mech_state:
            return
        info = self.get_mech_surface_info()
        surface = info[1] if info else None
        options = available_states(surface) if surface is not None else {}
        is_dmd_surf = is_dmd(surface) if surface is not None else False
        is_galvo_surf = getattr(surface, "galvo_config", None) is not None if surface is not None else False
        try:
            from optics_app.engine.plm import is_plm
        except ImportError:
            from v2.engine.plm import is_plm
        is_plm_surf = (is_plm(surface) or getattr(surface, 'is_plm', False)) if surface is not None else False

        if is_dmd_surf:
            prefix = "DMD:"
            self.btn_mech_state.setText(f"{prefix} {state or 'UNCONFIGURED'}")
            # TI devices (PLM / DMD) are controlled exclusively via TI Modulator Studio
            self.btn_mech_state.setVisible(False)
            return
        elif is_plm_surf:
            prefix = "PLM:"
            self.btn_mech_state.setText(f"{prefix} {state or 'UNCONFIGURED'}")
            # TI devices (PLM / DMD) are controlled exclusively via TI Modulator Studio
            self.btn_mech_state.setVisible(False)
            return
        elif is_galvo_surf:
            prefix = "GALVO:"
        else:
            prefix = "MECH:"

        self.btn_mech_state.setEnabled(len(options) > 1 or is_galvo_surf)
        self.btn_mech_state.setText(f"{prefix} {state or 'UNCONFIGURED'}")
        state_data = options.get(state, {})
        description = str(state_data.get("description") or state_data.get("label") or "")
        if is_galvo_surf:
            description = f"Galvo optical scanner angle: {state}. Click to step deflection angle."
        self.btn_mech_state.setToolTip(description or "Cycle the configured mechanism states.")
        if state == "ON":
            self.btn_mech_state.setStyleSheet(
                "QPushButton { background: #dcfce7; color: #166534; border: 1px solid #86efac; border-radius: 4px; padding: 4px 10px; font-weight: bold; font-size: 11px; }"
                "QPushButton:hover { background: #bbf7d0; }"
            )
        elif state == "OFF":
            self.btn_mech_state.setStyleSheet(
                "QPushButton { background: #fee2e2; color: #991b1b; border: 1px solid #fca5a5; border-radius: 4px; padding: 4px 10px; font-weight: bold; font-size: 11px; }"
                "QPushButton:hover { background: #fecaca; }"
            )
        else:
            self.btn_mech_state.setStyleSheet(
                "QPushButton { background: #e0f2fe; color: #075985; border: 1px solid #7dd3fc; border-radius: 4px; padding: 4px 10px; font-weight: bold; font-size: 11px; }"
            )

    def set_no_transparency(self, no_transp: bool) -> None:
        """Toggle opaque rendering for 3D elements (alpha=1.0)."""
        self.no_transparency = bool(no_transp)
        if self.cb_no_transparency and hasattr(self.cb_no_transparency, "isChecked"):
            if self.cb_no_transparency.isChecked() != self.no_transparency:
                self.cb_no_transparency.blockSignals(True)
                self.cb_no_transparency.setChecked(self.no_transparency)
                self.cb_no_transparency.blockSignals(False)
        self.transparency_changed.emit(self.no_transparency)
        self.update_layout()

    def toggle_no_transparency(self) -> bool:
        """Toggle opaque rendering on/off (Hotkey: T)."""
        self.set_no_transparency(not self.no_transparency)
        return self.no_transparency

    def _set_view_direct(self, elev: float, azim: float) -> None:
        """Directly set camera elevation and azimuth without animation."""
        if self.ax is not None and hasattr(self.ax, "view_init"):
            self.ax.view_init(elev=float(elev), azim=float(azim))
            if hasattr(self.ax, '_dist'):
                self.ax._dist = self._cam_dist
            self._update_status_text(elev, azim)
            if hasattr(self.canvas, "draw_idle"):
                self.canvas.draw_idle()
            elif hasattr(self.canvas, "draw"):
                self.canvas.draw()

    def animate_to_view(self, target_elev: float, target_azim: float, steps: int = 5, interval_ms: int = 20) -> None:
        """Smoothly interpolate camera to target elevation and azimuth (Blender-style cubic ease-out transition)."""
        if self.ax is None or not hasattr(self.ax, "view_init"):
            return

        # If Qt timer is not available or steps <= 1, jump directly
        if not QT_AVAILABLE or QTimer is None or steps <= 1:
            self._set_view_direct(target_elev, target_azim)
            return

        # Stop any active animation timer
        if self._view_anim_timer is not None:
            try:
                self._view_anim_timer.stop()
            except Exception:
                pass
            self._view_anim_timer = None

        start_elev = float(getattr(self.ax, 'elev', 20.0))
        start_azim = float(getattr(self.ax, 'azim', -60.0))

        # Shortest angular distance around 360-degree turntable for azimuth
        d_azim = ((float(target_azim) - start_azim + 180.0) % 360.0) - 180.0
        d_elev = float(target_elev) - start_elev

        if abs(d_azim) < 0.2 and abs(d_elev) < 0.2:
            self._set_view_direct(target_elev, target_azim)
            return

        self._anim_step = 0
        self._anim_total_steps = max(steps, 2)

        def _step():
            self._anim_step += 1
            t = self._anim_step / self._anim_total_steps
            # Cubic ease-out curve
            ease = 1.0 - (1.0 - t) ** 3
            cur_e = start_elev + d_elev * ease
            cur_a = (start_azim + d_azim * ease) % 360.0
            if cur_a > 180.0:
                cur_a -= 360.0

            if self._anim_step >= self._anim_total_steps:
                if self._view_anim_timer is not None:
                    self._view_anim_timer.stop()
                    self._view_anim_timer = None
                self._set_view_direct(target_elev, target_azim)
            else:
                self.ax.view_init(elev=cur_e, azim=cur_a)
                if hasattr(self.ax, '_dist'):
                    self.ax._dist = self._cam_dist
                self._update_status_text(cur_e, cur_a)
                if hasattr(self.canvas, "draw_idle"):
                    self.canvas.draw_idle()

        try:
            self._view_anim_timer = QTimer(self)
            self._view_anim_timer.setInterval(interval_ms)
            self._view_anim_timer.timeout.connect(_step)
            self._view_anim_timer.start()
        except Exception:
            self._set_view_direct(target_elev, target_azim)

    def set_view_isometric(self, animate: bool = False) -> None:
        """Reset camera orientation to standard isometric angle (elev=20°, azim=-60°). Hotkey: Num 0."""
        if animate:
            self.animate_to_view(20.0, -60.0)
        else:
            self._set_view_direct(20.0, -60.0)

    def reset_view(self) -> None:
        """Restore the default camera orientation (legacy public API)."""
        self.set_view_isometric()

    def set_view_front(self, animate: bool = False) -> None:
        """Front View (X-Y aperture plane): looking along optical axis Z (elev=0°, azim=0°). Hotkey: Num 1."""
        if animate:
            self.animate_to_view(0.0, 0.0)
        else:
            self._set_view_direct(0.0, 0.0)

    def set_view_back(self, animate: bool = False) -> None:
        """Back View (-X-Y plane): looking along optical axis from reverse (elev=0°, azim=180°). Hotkey: Ctrl+Num 1."""
        if animate:
            self.animate_to_view(0.0, 180.0)
        else:
            self._set_view_direct(0.0, 180.0)

    def set_view_side(self, animate: bool = False) -> None:
        """Side View (Z-Y plane): optical axis Z horizontal, transverse Y vertical (elev=0°, azim=-90°). Hotkey: Num 3."""
        if animate:
            self.animate_to_view(0.0, -90.0)
        else:
            self._set_view_direct(0.0, -90.0)

    def set_view_side_opposite(self, animate: bool = False) -> None:
        """Opposite Side View (-Z-Y plane): looking from reverse side (elev=0°, azim=90°). Hotkey: Ctrl+Num 3."""
        if animate:
            self.animate_to_view(0.0, 90.0)
        else:
            self._set_view_direct(0.0, 90.0)

    def set_view_top(self, animate: bool = False) -> None:
        """Top View (X-Z plane): optical axis Z horizontal, transverse X vertical (elev=90°, azim=-90°). Hotkey: Num 7."""
        if animate:
            self.animate_to_view(90.0, -90.0)
        else:
            self._set_view_direct(90.0, -90.0)

    def set_view_bottom(self, animate: bool = False) -> None:
        """Bottom View (-X-Z plane): looking up from underside (elev=-90°, azim=-90°). Hotkey: Ctrl+Num 7."""
        if animate:
            self.animate_to_view(-90.0, -90.0)
        else:
            self._set_view_direct(-90.0, -90.0)

    def set_view_in_between(self, mode: str, animate: bool = False) -> None:
        """Set camera to in-between diagonal angles between principal planes."""
        mapping = {
            "front_side": (0.0, -45.0),
            "side_back": (0.0, -135.0),
            "back_opposite": (0.0, 135.0),
            "opposite_front": (0.0, 45.0),
            "top_front": (45.0, 0.0),
            "top_side": (45.0, -90.0),
            "top_back": (45.0, 180.0),
            "top_opposite": (45.0, 90.0),
            "iso_ne": (30.0, -45.0),
            "iso_nw": (30.0, -135.0),
            "iso_se": (30.0, 45.0),
            "iso_sw": (30.0, 135.0),
        }
        key = str(mode).lower().replace(" ", "_").replace("-", "_")
        if key in mapping:
            elev, azim = mapping[key]
            if animate:
                self.animate_to_view(elev, azim)
            else:
                self._set_view_direct(elev, azim)

    def orbit_horizontal(self, delta_deg: float = 15.0, animate: bool = False) -> None:
        """Orbit camera horizontally by delta_deg (Blender Num 4 / Num 6)."""
        if self.ax is None:
            return
        cur_azim = float(getattr(self.ax, 'azim', -60.0))
        cur_elev = float(getattr(self.ax, 'elev', 20.0))
        target_azim = (cur_azim + delta_deg) % 360.0
        if target_azim > 180.0:
            target_azim -= 360.0
        if animate:
            self.animate_to_view(cur_elev, target_azim, steps=3, interval_ms=15)
        else:
            self._set_view_direct(cur_elev, target_azim)

    def orbit_vertical(self, delta_deg: float = 15.0, animate: bool = False) -> None:
        """Orbit camera vertically by delta_deg (Blender Num 8 / Num 2)."""
        if self.ax is None:
            return
        cur_azim = float(getattr(self.ax, 'azim', -60.0))
        cur_elev = float(getattr(self.ax, 'elev', 20.0))
        target_elev = max(-90.0, min(90.0, cur_elev + delta_deg))
        if animate:
            self.animate_to_view(target_elev, cur_azim, steps=3, interval_ms=15)
        else:
            self._set_view_direct(target_elev, cur_azim)

    def flip_view(self, animate: bool = False) -> None:
        """Flip camera 180° to look from opposite direction (Blender Num 9)."""
        if self.ax is None:
            return
        cur_azim = float(getattr(self.ax, 'azim', -60.0))
        cur_elev = float(getattr(self.ax, 'elev', 20.0))
        target_azim = (cur_azim + 180.0) % 360.0
        if target_azim > 180.0:
            target_azim -= 360.0
        if animate:
            self.animate_to_view(cur_elev, target_azim, steps=4, interval_ms=15)
        else:
            self._set_view_direct(cur_elev, target_azim)

    def _on_view_combo_changed(self, index: int) -> None:
        """Handle selection in the Planes & Diagonals combo box."""
        if index == 0:
            return
        combo_map = {
            1: (20.0, -60.0),    # Isometric
            2: (0.0, 0.0),       # Front (X-Y)
            3: (0.0, 180.0),     # Back (-X-Y)
            4: (0.0, -90.0),     # Side (Z-Y)
            5: (0.0, 90.0),      # Opposite Side (-Z-Y)
            6: (90.0, -90.0),    # Top (X-Z)
            7: (-90.0, -90.0),   # Bottom (-X-Z)
            8: (0.0, -45.0),     # In-Between: Front-Side
            9: (0.0, -135.0),    # In-Between: Side-Back
            10: (0.0, 135.0),    # In-Between: Back-Opposite
            11: (0.0, 45.0),     # In-Between: Opposite-Front
            12: (45.0, 0.0),     # In-Between: Top-Front
            13: (45.0, -90.0),   # In-Between: Top-Side
            14: (30.0, -45.0),   # Iso NE
            15: (30.0, -135.0),  # Iso NW
            16: (30.0, 45.0),    # Iso SE
            17: (30.0, 135.0),   # Iso SW
        }
        if index in combo_map:
            elev, azim = combo_map[index]
            self.animate_to_view(elev, azim)

    def _handle_navigation_key(self, key_text: str, is_ctrl: bool = False, animate: bool = False) -> bool:
        """Core keyboard navigation logic following Blender numpad conventions."""
        kt = str(key_text).strip().lower()
        if kt in ("1", "kp_1", "numpad1", "key_1"):
            if is_ctrl:
                self.set_view_back(animate=animate)
            else:
                self.set_view_front(animate=animate)
            return True
        elif kt in ("3", "kp_3", "numpad3", "key_3"):
            if is_ctrl:
                self.set_view_side_opposite(animate=animate)
            else:
                self.set_view_side(animate=animate)
            return True
        elif kt in ("7", "kp_7", "numpad7", "key_7"):
            if is_ctrl:
                self.set_view_bottom(animate=animate)
            else:
                self.set_view_top(animate=animate)
            return True
        elif kt in ("5", "kp_5", "numpad5", "key_5"):
            self.toggle_projection()
            return True
        elif kt in ("9", "kp_9", "numpad9", "key_9"):
            self.flip_view(animate=animate)
            return True
        elif kt in ("4", "kp_4", "numpad4", "key_4"):
            self.orbit_horizontal(-15.0, animate=animate)
            return True
        elif kt in ("6", "kp_6", "numpad6", "key_6"):
            self.orbit_horizontal(15.0, animate=animate)
            return True
        elif kt in ("8", "kp_8", "numpad8", "key_8"):
            self.orbit_vertical(15.0, animate=animate)
            return True
        elif kt in ("2", "kp_2", "numpad2", "key_2"):
            self.orbit_vertical(-15.0, animate=animate)
            return True
        elif kt in ("0", "kp_0", "numpad0", "key_0"):
            self.set_view_isometric(animate=animate)
            return True
        elif kt in (".", "kp_decimal", "period", "del", "f", "home"):
            self.fit_view()
            return True
        elif kt in ("+", "=", "plus", "kp_add"):
            self.zoom_view(0.85)
            return True
        elif kt in ("-", "_", "minus", "kp_subtract"):
            self.zoom_view(1.15)
            return True
        elif kt in ("t",):
            self.toggle_no_transparency()
            return True
        return False

    def keyPressEvent(self, event) -> None:
        """Handle Qt keyboard events for Blender navigation."""
        if not QT_AVAILABLE or event is None:
            return
        modifiers = event.modifiers() if hasattr(event, "modifiers") else None
        is_ctrl = bool(modifiers & Qt.ControlModifier) if (modifiers is not None and Qt is not None) else False

        txt = str(event.text() or '')
        if txt and self._handle_navigation_key(txt, is_ctrl):
            event.accept()
            return

        k = event.key() if hasattr(event, "key") else 0
        key_names = {
            getattr(Qt, 'Key_1', 49): '1',
            getattr(Qt, 'Key_2', 50): '2',
            getattr(Qt, 'Key_3', 51): '3',
            getattr(Qt, 'Key_4', 52): '4',
            getattr(Qt, 'Key_5', 53): '5',
            getattr(Qt, 'Key_6', 54): '6',
            getattr(Qt, 'Key_7', 55): '7',
            getattr(Qt, 'Key_8', 56): '8',
            getattr(Qt, 'Key_9', 57): '9',
            getattr(Qt, 'Key_0', 48): '0',
            getattr(Qt, 'Key_Period', 46): '.',
            getattr(Qt, 'Key_T', 84): 't',
            getattr(Qt, 'Key_F', 70): 'f',
            getattr(Qt, 'Key_Home', 0x01000010): 'home',
            getattr(Qt, 'Key_Plus', 43): '+',
            getattr(Qt, 'Key_Equal', 61): '=',
            getattr(Qt, 'Key_Minus', 45): '-',
            getattr(Qt, 'Key_Underscore', 95): '_',
        }
        if k in key_names and self._handle_navigation_key(key_names[k], is_ctrl):
            event.accept()
            return

        super_kpe = getattr(super(), 'keyPressEvent', None)
        if callable(super_kpe):
            super_kpe(event)

    def on_mpl_key_press(self, event) -> None:
        """Handle Matplotlib canvas key press events for Blender navigation."""
        if event is None or not getattr(event, 'key', None):
            return
        raw = str(event.key).lower()
        is_ctrl = 'ctrl' in raw or 'control' in raw
        clean = raw.replace('ctrl+', '').replace('control+', '').replace('ctrl', '').strip()
        self._handle_navigation_key(clean, is_ctrl)

    def resizeEvent(self, event) -> None:
        """Dynamically adapt 3D camera viewport when window is resized."""
        super_re = getattr(super(), 'resizeEvent', None)
        if callable(super_re):
            super_re(event)
        if hasattr(self, 'canvas') and self.canvas is not None:
            if hasattr(self.canvas, "draw_idle"):
                self.canvas.draw_idle()
            elif hasattr(self.canvas, "draw"):
                self.canvas.draw()

    def fit_view(self) -> None:
        """Fit camera to cleanly frame the entire optical system from first surface to end."""
        if self.ax is None:
            return
        if self._base_xlim is not None:
            self.ax.set_xlim3d(self._base_xlim)
            self.ax.set_ylim3d(self._base_ylim)
            self.ax.set_zlim3d(self._base_zlim)
        self._cam_dist = 7.5
        if hasattr(self.ax, '_dist'):
            self.ax._dist = 7.5
        self._apply_box_aspect()
        if hasattr(self.canvas, "draw_idle"):
            self.canvas.draw_idle()

    def focus_grating_diffraction(self) -> bool:
        """Frame a traced grating fan; repeated clicks cycle through gratings."""
        if self.ax is None or self.surface_stack is None or not self.rays:
            return False
        surfaces = self.surface_stack.get_active_sequence()
        samples = collect_grating_dispersion_samples(surfaces, self.rays)
        if not samples:
            return False
        grating_indices = sorted({sample["surface_index"] for sample in samples})
        if self._focused_grating_index in grating_indices:
            current = grating_indices.index(self._focused_grating_index)
            self._focused_grating_index = grating_indices[(current + 1) % len(grating_indices)]
        else:
            self._focused_grating_index = grating_indices[0]
        fan = [sample for sample in samples
               if sample["surface_index"] == self._focused_grating_index]
        points = []
        for sample in fan:
            point = sample["point"]
            distance = min(35.0, max(8.0, sample["next_distance_mm"] * 0.85))
            points.extend((point, point + sample["direction"] * distance))
        points = np.asarray(points, dtype=float)
        low = np.min(points, axis=0)
        high = np.max(points, axis=0)
        padding = np.maximum((high - low) * 0.18, [3.0, 3.0, 3.0])
        low -= padding
        high += padding
        self.ax.set_xlim3d(float(low[2]), float(high[2]))
        self.ax.set_ylim3d(float(low[0]), float(high[0]))
        self.ax.set_zlim3d(float(low[1]), float(high[1]))
        spans = high - low
        self.ax.set_box_aspect((float(spans[2]), float(spans[0]), float(spans[1])))
        self._set_view_direct(12.0, -88.0)
        if hasattr(self.canvas, "draw_idle"):
            self.canvas.draw_idle()
        return True

    def toggle_projection(self) -> None:
        """Toggle between Perspective and Orthographic camera projection."""
        if self.ax is None or not hasattr(self.ax, "set_proj_type"):
            return
        if self.projection_mode == "persp":
            self.projection_mode = "ortho"
            self.ax.set_proj_type('ortho')
            if self.btn_toggle_proj:
                self.btn_toggle_proj.setText("Ortho")
                self.btn_toggle_proj.setToolTip("Currently Orthographic projection. Click for Perspective.")
        else:
            self.projection_mode = "persp"
            self.ax.set_proj_type('persp')
            if self.btn_toggle_proj:
                self.btn_toggle_proj.setText("Persp")
                self.btn_toggle_proj.setToolTip("Currently Perspective projection. Click for Orthographic.")
        if hasattr(self.canvas, "draw_idle"):
            self.canvas.draw_idle()

    def set_aspect_mode(self, mode: str) -> None:
        """Set 3D box aspect mode ('wide', 'physical', 'cubic')."""
        self.aspect_mode = mode
        self._apply_box_aspect()
        if hasattr(self.canvas, "draw_idle"):
            self.canvas.draw_idle()

    def _on_aspect_combo_changed(self, index: int) -> None:
        modes = ["wide", "physical", "cubic"]
        if 0 <= index < len(modes):
            self.set_aspect_mode(modes[index])

    def on_pick(self, event):
        if not hasattr(event, "artist"): return
        idx = getattr(event.artist, '_surface_index', None)
        if idx is not None:
            self.surface_selected.emit(idx)
            
    def on_button_press(self, event):
        if event.inaxes != self.ax:
            return
        self._mouse_press_pos = (event.x, event.y)
        self._mouse_last_pos = (event.x, event.y)
        key = str(getattr(event, 'key', '') or '').lower()
        if event.button == 2 or (event.button == 1 and 'shift' in key):
            self._is_panning = True
            if QT_AVAILABLE and hasattr(self, 'setCursor') and Qt is not None:
                try:
                    self.setCursor(getattr(Qt, 'ClosedHandCursor', 0))
                except Exception:
                    pass
        elif event.button == 3:
            self._is_zooming = True
            if QT_AVAILABLE and hasattr(self, 'setCursor') and Qt is not None:
                try:
                    self.setCursor(getattr(Qt, 'SizeVerCursor', 0))
                except Exception:
                    pass
        elif event.button == 1:
            self._is_rotating = True
            if QT_AVAILABLE and hasattr(self, 'setCursor') and Qt is not None:
                try:
                    self.setCursor(getattr(Qt, 'SizeAllCursor', 0))
                except Exception:
                    pass

    def on_mouse_move(self, event):
        if self._mouse_last_pos is None or event.x is None or event.y is None:
            return
        dx = event.x - self._mouse_last_pos[0]
        dy = event.y - self._mouse_last_pos[1]
        self._mouse_last_pos = (event.x, event.y)

        if self._is_panning:
            self.pan_view(dx, dy)
        elif self._is_zooming:
            scale_factor = 1.0 - (dy / 250.0)
            scale_factor = max(0.80, min(1.25, scale_factor))
            self.zoom_view(scale_factor)
        elif self._is_rotating:
            if hasattr(self.ax, '_dist'):
                self.ax._dist = self._cam_dist
            self._update_status_text()

    def on_button_release(self, event):
        self._is_panning = False
        self._is_zooming = False
        self._is_rotating = False
        if hasattr(self.ax, '_dist'):
            self.ax._dist = self._cam_dist
        if QT_AVAILABLE and hasattr(self, 'setCursor') and Qt is not None:
            try:
                self.setCursor(getattr(Qt, 'ArrowCursor', 0))
            except Exception:
                pass
        self._update_status_text()

    def on_scroll(self, event):
        if event.inaxes != self.ax:
            return
        if event.button == 'up':
            scale_factor = 0.88
        elif event.button == 'down':
            scale_factor = 1.14
        else:
            scale_factor = 1.0
        self.zoom_view(scale_factor)

    def pan_view(self, dx_pixels: float, dy_pixels: float) -> None:
        """Pan 3D camera smoothly along screen projection vectors."""
        if self.ax is None:
            return
        try:
            M = self.ax.get_proj()[:2, :3]
            norm_u = float(np.linalg.norm(M[0]))
            norm_v = float(np.linalg.norm(M[1]))
            if norm_u < 1e-9 or norm_v < 1e-9:
                return
            u = M[0] / norm_u
            v = M[1] / norm_v

            xlim = self.ax.get_xlim3d()
            ylim = self.ax.get_ylim3d()
            zlim = self.ax.get_zlim3d()

            span_x = xlim[1] - xlim[0]
            span_y = ylim[1] - ylim[0]
            span_z = zlim[1] - zlim[0]

            w = float(self.canvas.width()) if hasattr(self.canvas, 'width') and self.canvas.width() > 50 else 800.0
            h = float(self.canvas.height()) if hasattr(self.canvas, 'height') and self.canvas.height() > 50 else 600.0

            du = float(dx_pixels) / w
            dv = float(dy_pixels) / h

            shift_x = -(du * u[0] + dv * v[0]) * span_x
            shift_y = -(du * u[1] + dv * v[1]) * span_y
            shift_z = -(du * u[2] + dv * v[2]) * span_z

            self.ax.set_xlim3d([xlim[0] + shift_x, xlim[1] + shift_x])
            self.ax.set_ylim3d([ylim[0] + shift_y, ylim[1] + shift_y])
            self.ax.set_zlim3d([zlim[0] + shift_z, zlim[1] + shift_z])

            if hasattr(self.canvas, "draw_idle"):
                self.canvas.draw_idle()
            elif hasattr(self.canvas, "draw"):
                self.canvas.draw()
        except Exception:
            pass

    def zoom_view(self, scale_factor: float) -> None:
        """Zoom in/out smoothly scaling 3D limits symmetrically around center."""
        if self.ax is None:
            return
        try:
            xlim = self.ax.get_xlim3d()
            ylim = self.ax.get_ylim3d()
            zlim = self.ax.get_zlim3d()

            xc = (xlim[0] + xlim[1]) * 0.5
            yc = (ylim[0] + ylim[1]) * 0.5
            zc = (zlim[0] + zlim[1]) * 0.5

            xw = (xlim[1] - xlim[0]) * 0.5 * scale_factor
            yw = (ylim[1] - ylim[0]) * 0.5 * scale_factor
            zw = (zlim[1] - zlim[0]) * 0.5 * scale_factor

            self.ax.set_xlim3d([xc - xw, xc + xw])
            self.ax.set_ylim3d([yc - yw, yc + yw])
            self.ax.set_zlim3d([zc - zw, zc + zw])

            if hasattr(self.canvas, "draw_idle"):
                self.canvas.draw_idle()
            elif hasattr(self.canvas, "draw"):
                self.canvas.draw()
        except Exception:
            pass

    def _apply_box_aspect(self) -> None:
        """Configure 3D box aspect ratio so optics are prominent and fill the window."""
        if self.ax is None or not hasattr(self.ax, 'set_box_aspect'):
            return
        if self._base_xlim is None:
            return

        span_z = max(abs(self._base_xlim[1] - self._base_xlim[0]), 1.0)
        span_x = max(abs(self._base_ylim[1] - self._base_ylim[0]) if self._base_ylim else 1.0, 1.0)
        span_y = max(abs(self._base_zlim[1] - self._base_zlim[0]) if self._base_zlim else 1.0, 1.0)

        if self.aspect_mode == "wide":
            # Elongate propagation axis Z horizontally across wide window
            aspect_z = 2.4
            aspect_x = max(float(span_x / max(span_y, 0.01)), 1.0)
            aspect_y = 1.0
            self.ax.set_box_aspect((aspect_z, aspect_x, aspect_y))
        elif self.aspect_mode == "physical":
            self.ax.set_box_aspect((span_z, span_x, span_y))
        elif self.aspect_mode == "cubic":
            self.ax.set_box_aspect((1.0, 1.0, 1.0))

    def _update_status_text(self, elev: Optional[float] = None, azim: Optional[float] = None) -> None:
        """Update live status readout with surface count, ray count, camera angles, and opaque state."""
        if not self.lbl_status or not hasattr(self.lbl_status, "setText"):
            return
        n_surfs = len(self.surface_stack.surfaces) if self.surface_stack else 0
        if hasattr(self.rays, "all_rays"):
            n_rays = len(self.rays.all_rays())
        else:
            n_rays = len(self.rays) if self.rays else 0
        route_text = f" | {self._last_route_count} routes" if self._last_route_count else ""
        if elev is None and self.ax is not None and hasattr(self.ax, 'elev'):
            elev = self.ax.elev
        if azim is None and self.ax is not None and hasattr(self.ax, 'azim'):
            azim = self.ax.azim
        angle_str = f" | Elev: {elev:.0f}° Azim: {azim:.0f}°" if (elev is not None and azim is not None) else ""
        opaque_badge = " | 🛡️ Opaque" if self.no_transparency else ""
        self.lbl_status.setText(f"3D: {n_surfs} surfs | {n_rays} rays{route_text}{angle_str}{opaque_badge}")

    def refresh_layout(self) -> bool:
        stack = None
        if self.model is not None:
            if hasattr(self.model, "get_surface_stack"):
                try:
                    stack = self.model.get_surface_stack()
                except Exception:
                    pass
            if hasattr(self.model, "get_rays"):
                try:
                    cached_rays = self.model.get_rays()
                    if cached_rays:
                        self.rays = cached_rays
                except Exception:
                    pass

        if stack is not None:
            self.surface_stack = stack

        rays_empty = (
            not self.rays
            or (len(self.rays.all_rays()) == 0 if hasattr(self.rays, "all_rays") else len(self.rays) == 0)
        )
        if rays_empty and self.surface_stack is not None and len(self.surface_stack.surfaces) > 1:
            try:
                self.rays = generate_3d_rays(self.surface_stack)
            except Exception:
                pass

        self.update_layout(surface_stack=self.surface_stack, rays=self.rays)
        return True

    def _render_element_labels(self, surfaces: List[Surface], all_x: list, all_y: list, all_z: list) -> None:
        """Draw 3D callout labels and anchor stems for optical components."""
        if not surfaces:
            return

        groups = compute_surface_groups(surfaces)
        n = len(surfaces)
        i = 0
        while i < n:
            grp = groups[i] if i < len(groups) else {}
            gtype = grp.get("group_type", "")
            gname = grp.get("group_name", "")
            eid = grp.get("element_id", -1)

            if gtype == "LENS":
                start_idx = i
                end_idx = i
                while end_idx + 1 < n and groups[end_idx + 1].get("element_id") == eid and groups[end_idx + 1].get("group_type") == "LENS":
                    end_idx += 1
                if end_idx + 1 < n and surfaces[end_idx].is_glass and not surfaces[end_idx + 1].is_glass:
                    end_idx += 1

                front = surfaces[start_idx]
                rear = surfaces[min(end_idx, n - 1)]

                c_front = np.asarray(getattr(front, "global_origin", [0.0, 0.0, 0.0]), dtype=float)
                c_rear = np.asarray(getattr(rear, "global_origin", [0.0, 0.0, 0.0]), dtype=float)
                mid_c = (c_front + c_rear) * 0.5
                R = np.asarray(getattr(front, "global_R", np.eye(3)), dtype=float)

                dz_val = self.element_displacements.get(eid, 0.0) + self.global_displacement_z
                if abs(dz_val) > 1e-5:
                    mid_c = mid_c + R[:, 2] * dz_val

                max_sd = max(
                    float(getattr(front, "semi_diameter", 10.0)),
                    float(getattr(rear, "semi_diameter", 10.0)),
                    2.0
                )

                n_surfs = end_idx - start_idx + 1
                surf_tag = f"S{start_idx}-S{end_idx}" if n_surfs > 1 else f"S{start_idx}"
                disp_name = gname or f"Lens {eid + 1}"
                display_label = f"{disp_name} ({surf_tag}) · Semi-D: {max_sd:.2f} mm (Ø = {2*max_sd:.2f} mm)"

                anchor_pt = mid_c + R @ np.array([0.0, max_sd, 0.0])
                label_pt = mid_c + R @ np.array([0.0, max_sd * 1.25 + 3.0, 0.0])

                pal = grp.get("palette") or (LENS_ELEMENT_PALETTES[eid % len(LENS_ELEMENT_PALETTES)] if eid >= 0 else LENS_ELEMENT_PALETTES[0])
                badge_col = pal.get("badge_hex", "#0d47a1")
                border_col = pal.get("border_hex", "#1976d2")
                bg_col = pal.get("bg_hex", "#e3f2fd")

                self.ax.plot(
                    [anchor_pt[2], label_pt[2]],
                    [anchor_pt[0], label_pt[0]],
                    [anchor_pt[1], label_pt[1]],
                    color=border_col, linestyle=':', linewidth=0.9, alpha=0.75
                )

                self.ax.text(
                    label_pt[2], label_pt[0], label_pt[1],
                    f" {display_label} ",
                    color=badge_col,
                    fontsize=8.5,
                    fontweight='bold',
                    ha='center',
                    va='bottom',
                    bbox=dict(
                        boxstyle='round,pad=0.25',
                        facecolor=bg_col,
                        edgecolor=border_col,
                        alpha=0.92,
                        linewidth=0.9
                    ),
                    zorder=15
                )
                all_x.extend([anchor_pt[0], label_pt[0]])
                all_y.extend([anchor_pt[1], label_pt[1]])
                all_z.extend([anchor_pt[2], label_pt[2]])
                i = end_idx + 1

            else:
                surf = surfaces[i]
                role = classify_surface_for_3d(surf, i, n)
                c, R = get_effective_pose(surf)
                dz_surf = self.element_displacements.get(i, 0.0) + self.global_displacement_z
                if abs(dz_surf) > 1e-5:
                    c = c + R[:, 2] * dz_surf
                sd_val = float(getattr(surf, "semi_diameter", 0.0))
                ap_type_str = str(getattr(surf, "aperture_type", "")).lower()
                is_pt = sd_val <= 0.0 or ap_type_str in ("point", "none")

                if role["object"]:
                    if is_pt:
                        label_txt = f"S{i}: Point Object"
                    else:
                        hx_o = float(getattr(surf, "aperture_x_half", 0.0) or sd_val)
                        hy_o = float(getattr(surf, "aperture_y_half", 0.0) or sd_val)
                        if ap_type_str in ("rectangular", "slit"):
                            label_txt = f"S{i}: {surf.name or 'Object'} [Semi-D: X={hx_o:.2f}, Y={hy_o:.2f} mm | Size: {2*hx_o:.2f} × {2*hy_o:.2f} mm]"
                        else:
                            label_txt = f"S{i}: {surf.name or 'Object'} [Semi-D: {sd_val:.2f} mm | Ø = {2*sd_val:.2f} mm]"
                    theme = ('#0369a1', '#e0f2fe', '#0284c7') if is_pt else ('#424242', '#f5f5f5', '#9e9e9e')
                elif role["stop"]:
                    ap_type_stop = str(getattr(surf, "aperture_type", "Circular")).lower()
                    if ap_type_stop in ("rectangular", "slit"):
                        hx = float(getattr(surf, "aperture_x_half", 0.0) or sd_val)
                        hy = float(getattr(surf, "aperture_y_half", 0.0) or sd_val)
                        label_txt = f"S{i}: Stop (Semi-D X={hx:.2f}, Y={hy:.2f} mm | Size: {2*hx:.2f} × {2*hy:.2f} mm | Ø={2*sd_val:.2f} mm)"
                    elif ap_type_stop == "annular":
                        rin = float(getattr(surf, "aperture_r_inner", 0.0))
                        label_txt = f"S{i}: Stop Annular (Inner R={rin:.2f} mm | Outer Semi-D={sd_val:.2f} mm, Ø={2*sd_val:.2f} mm)"
                    else:
                        label_txt = f"S{i}: Stop (Semi-D R={sd_val:.2f} mm, Ø={2*sd_val:.2f} mm)"
                    theme = ('#b71c1c', '#ffebee', '#e53935')
                elif role["detector"]:
                    if is_pt:
                        label_txt = f"S{i}: Point Image"
                    else:
                        hx_d = float(getattr(surf, "aperture_x_half", 0.0) or sd_val)
                        hy_d = float(getattr(surf, "aperture_y_half", 0.0) or sd_val)
                        if ap_type_str in ("rectangular", "slit"):
                            label_txt = f"S{i}: Detector ({surf.name or 'Image'}) [Semi-D: X={hx_d:.2f}, Y={hy_d:.2f} mm | Size: {2*hx_d:.2f} × {2*hy_d:.2f} mm]"
                        else:
                            label_txt = f"S{i}: Detector ({surf.name or 'Image'}) [Semi-D: {sd_val:.2f} mm | Ø = {2*sd_val:.2f} mm]"
                    theme = ('#991b1b', '#fee2e2', '#dc2626') if is_pt else ('#880e4f', '#fce4ec', '#c2185b')
                elif is_galvo(surf):
                    axis = getattr(getattr(surf, "galvo_config", None), "scan_axis", "X")
                    cur_ang = getattr(getattr(surf, "galvo_config", None), "current_angle_deg", 0.0)
                    label_txt = f"S{i}: Galvo {axis} [{cur_ang:+.1f}°] ({surf.name or 'Mirror'}) [Semi-D: {sd_val:.2f} mm | Ø = {2*sd_val:.2f} mm]"
                    theme = ('#004d40', '#e0f2f1', '#00897b')
                elif role["grating"]:
                    lines = getattr(surf, "grating_lines_per_mm", 0)
                    blaze_um = getattr(surf, "grating_blaze_wavelength_um", None) or getattr(surf, "grating_primary_wavelength_um", None)
                    blaze_ang = getattr(surf, "grating_blaze_angle_deg", None)
                    if blaze_ang is None and blaze_um and lines > 0:
                        sin_b = (lines * blaze_um * abs(getattr(surf, "grating_order", 1))) / 2000.0
                        if 0 < sin_b < 1.0:
                            blaze_ang = float(np.degrees(np.arcsin(sin_b)))
                    
                    blaze_parts = []
                    if blaze_um:
                        b_val = blaze_um * 1000.0 if blaze_um < 5.0 else blaze_um
                        unit = "nm" if blaze_um < 5.0 else "µm"
                        blaze_parts.append(f"Blazed @ {b_val:g} {unit}")
                    if blaze_ang is not None and blaze_ang > 0:
                        blaze_parts.append(f"θb={blaze_ang:.1f}°")
                    
                    extra = f" ({', '.join(blaze_parts)})" if blaze_parts else ""
                    dim_str = f" [Semi-D: {sd_val:.2f} mm | Ø = {2*sd_val:.2f} mm]" if sd_val > 0 else ""
                    label_txt = f"S{i}: Grating {lines:g} l/mm{extra}{dim_str} [Top Ruled Face]" if lines > 0 else f"S{i}: Grating{dim_str} [Top Ruled Face]"
                    theme = ('#78350f', '#fef3c7', '#d97706')
                elif role["dmd"]:
                    dmd_hx = float(getattr(surf, 'aperture_x_half', 0.0) or sd_val or 4.93)
                    dmd_hy = float(getattr(surf, 'aperture_y_half', 0.0) or sd_val or 3.08)
                    label_txt = f"S{i}: {surf.name or 'DMD'} [{active_state_name(surf)}] [Semi-D: X={dmd_hx:.2f}, Y={dmd_hy:.2f} mm | Size: {2*dmd_hx:.2f} × {2*dmd_hy:.2f} mm]"
                    theme = ('#155e75', '#ecfeff', '#0891b2')
                elif role["mirror"]:
                    hx_m = float(getattr(surf, "aperture_x_half", 0.0) or sd_val)
                    hy_m = float(getattr(surf, "aperture_y_half", 0.0) or sd_val)
                    if ap_type_str in ("rectangular", "slit"):
                        label_txt = f"S{i}: {surf.name or 'Mirror'} [Semi-D: X={hx_m:.2f}, Y={hy_m:.2f} mm | Size: {2*hx_m:.2f} × {2*hy_m:.2f} mm]"
                    else:
                        label_txt = f"S{i}: {surf.name or 'Mirror'} [Semi-D: {sd_val:.2f} mm | Ø = {2*sd_val:.2f} mm]"
                    theme = ('#e65100', '#fff8e1', '#ffa000')
                else:
                    dim_str = f" [Semi-D: {sd_val:.2f} mm | Ø = {2*sd_val:.2f} mm]" if sd_val > 0 else ""
                    label_txt = f"S{i}: {surf.name or 'Surface'}{dim_str}"
                    theme = ('#37474f', '#eceff1', '#78909c')

                if is_pt and (role["object"] or role["detector"]):
                    anchor_pt = c + R @ np.array([0.0, 0.5, 0.0])
                    label_pt = c + R @ np.array([0.0, 2.5, 0.0])
                else:
                    sd = max(sd_val, 2.0)
                    anchor_pt = c + R @ np.array([0.0, sd, 0.0])
                    label_pt = c + R @ np.array([0.0, sd * 1.25 + 3.0, 0.0])

                text_color, bg_color, border_color = theme
                self.ax.plot(
                    [anchor_pt[2], label_pt[2]],
                    [anchor_pt[0], label_pt[0]],
                    [anchor_pt[1], label_pt[1]],
                    color=border_color, linestyle=':', linewidth=0.9, alpha=0.75
                )
                self.ax.text(
                    label_pt[2], label_pt[0], label_pt[1],
                    f" {label_txt} ",
                    color=text_color,
                    fontsize=8.5,
                    fontweight='bold',
                    ha='center',
                    va='bottom',
                    bbox=dict(
                        boxstyle='round,pad=0.25',
                        facecolor=bg_color,
                        edgecolor=border_color,
                        alpha=0.92,
                        linewidth=0.9
                    ),
                    zorder=15
                )
                all_x.extend([anchor_pt[0], label_pt[0]])
                all_y.extend([anchor_pt[1], label_pt[1]])
                all_z.extend([anchor_pt[2], label_pt[2]])
                i += 1

    def _render_field_labels(self, surfaces: List[Surface], all_x: list, all_y: list, all_z: list) -> None:
        """Render 3D markers and callout tags for multi-field object points."""
        sys_cfg = getattr(self, "system_config", None)
        if sys_cfg is None and self.model is not None:
            sys_cfg = getattr(self.model, "system_config", None)
        if sys_cfg is None and hasattr(self.parent(), "system_config"):
            sys_cfg = getattr(self.parent(), "system_config", None)
        if sys_cfg is None:
            return

        fields = getattr(sys_cfg, "fields", [])
        if not fields:
            return
        if len(fields) == 1 and abs(getattr(fields[0], "x", 0.0)) < 1e-6 and abs(getattr(fields[0], "y", 0.0)) < 1e-6:
            return

        import math
        d0 = float(getattr(surfaces[0], "thickness", 0.0)) if surfaces else 0.0
        ftype = getattr(sys_cfg, "field_type", None)
        is_angle = str(getattr(ftype, "value", ftype)).lower() == "angle"

        for f_idx, field in enumerate(fields):
            fx = float(field.x)
            fy = float(field.y)
            if is_angle:
                obj_x = -d0 * math.tan(math.radians(fx)) if d0 > 0 else 0.0
                obj_y = -d0 * math.tan(math.radians(fy)) if d0 > 0 else 0.0
                tag = f"Field {f_idx + 1}: ({fx:g}°, {fy:g}°)"
            else:
                obj_x = fx
                obj_y = fy
                tag = f"Field {f_idx + 1}: ({fx:g}mm, {fy:g}mm)"

            col = FIELD_COLORS[f_idx % len(FIELD_COLORS)]
            self.ax.scatter([0.0], [obj_x], [obj_y], color=col, s=36, marker='o', edgecolors='#0f172a' if self.dark_mode else '#ffffff', linewidths=1.0, zorder=16)
            self.ax.text(
                0.0, obj_x, obj_y + 1.2,
                f" {tag} ",
                color=col,
                fontsize=8,
                fontweight='bold',
                ha='center',
                va='bottom',
                bbox=dict(
                    boxstyle='round,pad=0.2',
                    facecolor='#1e293b' if self.dark_mode else '#ffffff',
                    edgecolor=col,
                    alpha=0.92,
                    linewidth=0.9
                ),
                zorder=16
            )
            all_x.append(obj_x)
            all_y.append(obj_y)
            all_z.append(0.0)

    def _render_rays(self, rays: List[Ray], all_x: Optional[list] = None, all_y: Optional[list] = None, all_z: Optional[list] = None) -> None:
        """Render 3D ray trajectories through the optical system using batched Line3DCollection."""
        if hasattr(rays, "all_rays"):
            rays = rays.all_rays()
        if not rays:
            return

        surfaces = self.surface_stack.get_active_sequence() if self.surface_stack is not None else []
        spectral_colors = diffraction_wavelength_colors([r.wavelength for r in rays])

        # Group line segments by visual styling to batch with Line3DCollection
        segments_by_style: Dict[Tuple[str, float, str, float, int], List[np.ndarray]] = {}

        for ray in rays:
            if not hasattr(ray, "history") or len(ray.history) == 0:
                continue

            pts = []
            if getattr(ray, "launch_pos", None) is not None and len(ray.history) > 0:
                p0 = np.asarray(ray.launch_pos, dtype=float)
                first_h = ray.history[0]
                h0 = np.array([
                    float(first_h.get("x_global", 0.0)),
                    float(first_h.get("y_global", 0.0)),
                    float(first_h.get("z_global", 0.0)),
                ])
                if np.linalg.norm(p0 - h0) > 1e-4:
                    pts.append((p0[2], p0[0], p0[1]))
                    if all_x is not None:
                        all_x.append(p0[0])
                        all_y.append(p0[1])
                        all_z.append(p0[2])

            for h in ray.history:
                gz = float(h.get("z_global", h.get("z", 0.0)))
                gx = float(h.get("x_global", h.get("x", 0.0)))
                gy = float(h.get("y_global", h.get("y", 0.0)))
                pts.append((gz, gx, gy))
                if all_x is not None:
                    all_x.append(gx)
                    all_y.append(gy)
                    all_z.append(gz)

            if len(pts) < 2:
                continue

            seg = np.asarray(pts, dtype=float)

            show_mc = getattr(self, "show_marginal_chief", False)
            mc_only = getattr(self, "marginal_chief_only", False)

            is_chief = bool(getattr(ray, "is_chief", False))
            is_marginal = bool(getattr(ray, "is_marginal", False))
            f_idx = getattr(ray, "field_idx", 0)
            fc = getattr(ray, "field_coord", None)
            is_on_axis = (f_idx == 0 and (fc is None or (abs(fc[0]) < 1e-6 and abs(fc[1]) < 1e-6)))

            if hasattr(ray, "pupil_coord"):
                p = ray.pupil_coord
                if p is not None:
                    if not is_marginal and is_on_axis:
                        if (abs(p[0]) < 1e-5 and abs(abs(p[1]) - 1.0) < 1e-4) or (abs(p[1]) < 1e-5 and abs(abs(p[0]) - 1.0) < 1e-4):
                            is_marginal = True
                    if not is_chief and not is_on_axis:
                        if abs(p[0]) < 1e-5 and abs(p[1]) < 1e-5:
                            is_chief = True

            if mc_only and not (is_chief or is_marginal):
                continue

            is_vig = getattr(ray, "vignetted", False)

            if (show_mc or mc_only) and is_marginal:
                ray_color = '#38bdf8' if self.dark_mode else '#0284c7'
                alpha = 1.0 if not is_vig else 0.50
                linestyle = '-' if not is_vig else '--'
                linewidth = 2.6 if not is_vig else 1.5
                ray_zorder = 30
            elif (show_mc or mc_only) and is_chief:
                ray_color = '#fb7185' if self.dark_mode else '#dc2626'
                alpha = 1.0 if not is_vig else 0.50
                linestyle = '-' if not is_vig else '--'
                linewidth = 2.6 if not is_vig else 1.5
                ray_zorder = 31
            else:
                if getattr(self, "color_mode", "wavelength") == "field":
                    f_idx = int(getattr(ray, "field_idx", 0))
                    ray_color = FIELD_COLORS[f_idx % len(FIELD_COLORS)]
                else:
                    wl = float(getattr(ray, "wavelength", 0.5876))
                    ray_color = spectral_colors.get(wl, wavelength_to_color(wl))

                if show_mc:
                    alpha = 0.22 if not is_vig else 0.10
                    linestyle = '-' if not is_vig else '--'
                    linewidth = 0.9 if not is_vig else 0.6
                    ray_zorder = 5
                else:
                    alpha = (1.0 if self.no_transparency else 0.85) if not is_vig else 0.30
                    linestyle = '-' if not is_vig else '--'
                    linewidth = (1.5 if self.no_transparency else 1.3) if not is_vig else 0.85
                    ray_zorder = 25 if self.no_transparency else 6

            style_key = (ray_color, alpha, linestyle, linewidth, ray_zorder)
            if style_key not in segments_by_style:
                segments_by_style[style_key] = []
            segments_by_style[style_key].append(seg)

        can_reuse_colls = (
            hasattr(self, "_ray_line_colls")
            and len(self._ray_line_colls) > 0
            and set(self._ray_line_colls.keys()) == set(segments_by_style.keys())
        )
        if can_reuse_colls:
            for style_key, seg_list in segments_by_style.items():
                self._ray_line_colls[style_key].set_segments(seg_list)
        else:
            if hasattr(self, "_ray_line_colls"):
                for coll in self._ray_line_colls.values():
                    try:
                        coll.remove()
                    except Exception:
                        pass
                self._ray_line_colls.clear()
            else:
                self._ray_line_colls = {}

            for (color, alpha, ls, lw, zo), seg_list in segments_by_style.items():
                coll = Line3DCollection(
                    seg_list,
                    colors=color,
                    alpha=alpha,
                    linestyles=ls,
                    linewidths=lw,
                    zorder=zo
                )
                self.ax.add_collection3d(coll, autolim=False)
                self._ray_line_colls[(color, alpha, ls, lw, zo)] = coll
                self._ray_artists.append(coll)

        mech_info = self.get_mech_surface_info()
        mech_surface = mech_info[1] if mech_info else None
        mech_state = mech_info[2] if mech_info else ""
        annot_key = (
            getattr(self, "color_mode", "wavelength"),
            tuple(sorted(spectral_colors.keys())),
            any(getattr(ray, "preview_source_assumed", False) for ray in rays),
            getattr(mech_surface, "name", ""),
            mech_state,
        )

        if getattr(self, "_last_ray_annot_key", None) != annot_key:
            if hasattr(self, "_ray_text_artists"):
                for art in self._ray_text_artists:
                    try:
                        art.remove()
                    except Exception:
                        pass
                self._ray_text_artists.clear()
            else:
                self._ray_text_artists = []
            self._last_ray_annot_key = annot_key

            if getattr(self, "color_mode", "wavelength") == "wavelength" and len(spectral_colors) > 1:
                t_hdr = self.ax.text2D(
                    0.02, 0.98, "Ray wavelengths (false color)",
                    transform=self.ax.transAxes, fontsize=7.5, va="top", color="#334155",
                    bbox=dict(facecolor="#f8fafc", edgecolor="#cbd5e1", alpha=0.94,
                              boxstyle="round,pad=0.25"), zorder=100,
                )
                self._ray_text_artists.append(t_hdr)
                self._ray_artists.append(t_hdr)
                for index, (wavelength, color) in enumerate(list(spectral_colors.items())[:8]):
                    t_lbl = self.ax.text2D(
                        0.025, 0.93 - index * 0.035, f"{wavelength * 1000:.0f} nm",
                        transform=self.ax.transAxes, fontsize=7.5, va="top",
                        color=color, fontweight="bold", zorder=101,
                    )
                    self._ray_text_artists.append(t_lbl)
                    self._ray_artists.append(t_lbl)

            if any(getattr(ray, "preview_source_assumed", False) for ray in rays):
                t_prev = self.ax.text2D(
                    0.02, 0.08, "Preview source: incident bundle assumed before first element",
                    transform=self.ax.transAxes, fontsize=7.5, color="#92400e",
                    bbox=dict(facecolor="#fffbeb", edgecolor="#fbbf24", alpha=0.95,
                              boxstyle="round,pad=0.25"), zorder=100,
                )
                self._ray_text_artists.append(t_prev)
                self._ray_artists.append(t_prev)

            if mech_info and rays:
                route = str(active_state(mech_surface).get("route", "") or "")
                active_index = next((i for i, surface in enumerate(surfaces)
                                     if surface is mech_surface), None)
                if route and active_index is not None:
                    hit = next((entry for ray in rays for entry in ray.history
                                if entry.get("surface_index") == active_index), None)
                    if hit is not None:
                        tz = float(hit.get("z_global", 0.0))
                        tx = float(hit.get("x_global", 0.0))
                        ty = float(hit.get("y_global", 0.0))
                        t_mech = self.ax.text(
                            tz, tx, ty + 2.0,
                            f" Mech {mech_state}: route → {route} ",
                            color="#155e75", fontsize=8, fontweight='bold',
                            ha='center', va='bottom',
                            bbox=dict(boxstyle='round,pad=0.25', facecolor='#ecfeff',
                                      edgecolor='#0891b2', alpha=0.92, linewidth=1.0),
                            zorder=16,
                        )
                        self._ray_text_artists.append(t_mech)
                        self._ray_artists.append(t_mech)

    def _render_grating_diffraction(self, surfaces: List[Surface], rays: List[Ray],
                                    all_x: Optional[list] = None, all_y: Optional[list] = None, all_z: Optional[list] = None) -> None:
        """Highlight actual wavelength-dependent outgoing grating directions."""
        if not any(getattr(surf, "surface_type", None) == SurfaceType.GRATING for surf in surfaces):
            return
        samples = getattr(self, "_cached_grating_samples", None)
        if samples is None:
            samples = collect_grating_dispersion_samples(surfaces, rays)
        if not samples:
            if hasattr(self, "_grating_artists"):
                for art in self._grating_artists:
                    try:
                        art.remove()
                    except Exception:
                        pass
                self._grating_artists.clear()
            self._last_grating_key = None
            return

        grating_key = tuple(
            (float(s["wavelength_um"]), int(s["order"]), int(s["surface_index"]), round(float(s["angle_deg"]), 2))
            for s in samples[:12]
        )
        if getattr(self, "_last_grating_key", None) == grating_key:
            return

        if hasattr(self, "_grating_artists"):
            for art in self._grating_artists:
                try:
                    art.remove()
                except Exception:
                    pass
            self._grating_artists.clear()
        else:
            self._grating_artists = []
        self._last_grating_key = grating_key

        colors = diffraction_wavelength_colors([s["wavelength_um"] for s in samples])
        shown = samples[:12]
        for sample in shown:
            point = sample["point"]
            direction = sample["direction"]
            length = sample["arrow_length_mm"]
            color = colors[sample["wavelength_um"]]
            qv = self.ax.quiver(
                point[2], point[0], point[1], direction[2], direction[0], direction[1],
                length=length, normalize=True, color=color, linewidth=2.1,
                arrow_length_ratio=0.24, zorder=27,
            )
            self._grating_artists.append(qv)
            self._ray_artists.append(qv)
            if all_x is not None:
                tip = point + direction * length
                all_x.append(float(tip[0])); all_y.append(float(tip[1])); all_z.append(float(tip[2]))

        grating_count = len({sample["surface_index"] for sample in samples})
        t_g_hdr = self.ax.text2D(
            0.63, 0.98,
            f"Traced grating diffraction: {grating_count} grating(s)\n"
            "NIR false color · angles only; efficiency uncalibrated",
            transform=self.ax.transAxes, fontsize=7.3, va="top",
            color="#93c5fd" if self.dark_mode else "#172554", zorder=100,
            bbox=dict(facecolor="#1e293b" if self.dark_mode else "#eff6ff", edgecolor="#3b82f6" if self.dark_mode else "#93c5fd", alpha=0.98, boxstyle="round,pad=0.35"),
        )
        self._grating_artists.append(t_g_hdr)
        self._ray_artists.append(t_g_hdr)
        for index, sample in enumerate(shown):
            t_g_item = self.ax.text2D(
                0.635, 0.90 - index * 0.035,
                f"S{sample['surface_index']} m={sample['order']:+d} {sample['lines_per_mm']:g} l/mm · "
                f"{sample['wavelength_um'] * 1000:.0f} nm · "
                + (f"α={sample['incident_angle_deg']:+.1f}°, "
                   if sample['incident_angle_deg'] is not None else "")
                + f"β={sample['angle_deg']:+.2f}°",
                transform=self.ax.transAxes, fontsize=7.5, va="top",
                color=colors[sample["wavelength_um"]], fontweight="bold", zorder=101,
                bbox=dict(facecolor="#1e293b" if self.dark_mode else "#ffffff", edgecolor="none", alpha=0.98, pad=1.0),
            )
            self._grating_artists.append(t_g_item)
            self._ray_artists.append(t_g_item)

    def _render_optical_routes(self, stack: Any, surfaces: List[Surface], all_x: list, all_y: list, all_z: list) -> int:
        """Overlay routed excitation, dump, and incoherent fluorescence paths."""
        routes = getattr(stack, "optical_routes", None)
        if routes is None:
            config = getattr(stack, "two_photon_config", None)
            routes = getattr(config, "routes", None)
        if routes is None or not hasattr(routes, "all_routes"):
            return 0

        centers = {}
        for surface in surfaces:
            uid = str(getattr(surface, "component_uid", "") or "")
            if uid and uid not in centers:
                centers[uid] = np.asarray(getattr(surface, "global_origin", [0.0, 0.0, 0.0]), dtype=float)

        drawn = 0
        emitter_center = None
        objective_center = None
        for route in routes.all_routes():
            if not getattr(route, "enabled", True):
                continue
            pts = [centers[step.component_uid] for step in route.steps if step.component_uid in centers]
            if len(pts) >= 2:
                arr = np.asarray(pts)
                self.ax.plot(
                    arr[:, 2], arr[:, 0], arr[:, 1],
                    color=getattr(route, "color", "#FFFFFF"), linewidth=2.6,
                    alpha=0.88, linestyle="-", label=f"Route: {route.name}", zorder=12,
                )
                all_x.extend(arr[:, 0]); all_y.extend(arr[:, 1]); all_z.extend(arr[:, 2])
                drawn += 1
            if getattr(route, "name", "") == "epi_detection":
                for step in route.steps:
                    if step.port == "emit" and step.component_uid in centers:
                        emitter_center = centers[step.component_uid]
                    if step.port == "collect" and step.component_uid in centers:
                        objective_center = centers[step.component_uid]

        # Fluorescence is spontaneous and incoherent.  These seeded rays are a
        # visual sample only; the photon budget is calculated analytically.
        if emitter_center is not None:
            rng = np.random.default_rng(4500)
            for idx in range(24):
                if objective_center is not None and idx < 12:
                    target = objective_center + rng.normal(0.0, 0.7, 3)
                else:
                    direction = rng.normal(size=3)
                    direction /= max(np.linalg.norm(direction), 1e-12)
                    target = emitter_center + direction * 8.0
                wl_um = 0.50 + 0.10 * (idx / 23.0)
                color = wavelength_to_color(wl_um)
                self.ax.plot(
                    [emitter_center[2], target[2]], [emitter_center[0], target[0]], [emitter_center[1], target[1]],
                    color=color, linewidth=0.8, alpha=0.42, linestyle=":" if idx >= 12 else "-", zorder=8,
                )
                all_x.extend([emitter_center[0], target[0]])
                all_y.extend([emitter_center[1], target[1]])
                all_z.extend([emitter_center[2], target[2]])
        return drawn

    def update_layout(self, surface_stack=None, rays=None, system_config=None, force_rebuild: bool = False) -> None:
        if surface_stack is not None:
            self.surface_stack = surface_stack
        if rays is not None:
            self.rays = rays
        if system_config is not None:
            self.system_config = system_config
        if force_rebuild:
            self._last_geom_fingerprint = None

        self._sync_mech_button_ui()
        if self.btn_grating_fan is not None:
            active = (self.surface_stack.get_active_sequence()
                      if self.surface_stack is not None else [])
            has_grating = any(getattr(surf, 'surface_type', None) == SurfaceType.GRATING for surf in active)
            grating_samples = collect_grating_dispersion_samples(active, self.rays) if has_grating else []
            self._cached_grating_samples = grating_samples
            grating_count = len({sample["surface_index"] for sample in grating_samples})
            self.btn_grating_fan.setVisible(has_grating)
            self.btn_grating_fan.setEnabled(bool(grating_samples))
            self.btn_grating_fan.setToolTip(
                "Cycle through traced grating diffraction fans."
                if grating_count > 1 else "Zoom onto the traced wavelength-dependent diffraction fan."
            )

        if not (QT_AVAILABLE and HAS_MPL_QT) or self.fig is None or self.ax is None:
            return

        stack = self.surface_stack
        surfaces = (stack.get_active_sequence() if hasattr(stack, 'get_active_sequence') else stack.surfaces) if stack is not None else []

        curr_fingerprint = None
        if stack is not None:
            surf_tokens = []
            for s in surfaces:
                orig = getattr(s, 'global_origin', None)
                orig_t = tuple(np.round(np.asarray(orig, dtype=float), 4)) if orig is not None else ()
                R_mat = getattr(s, 'global_R', None)
                r_t = tuple(np.round(np.asarray(R_mat, dtype=float).ravel(), 4)) if R_mat is not None else ()
                surf_tokens.append((
                    id(s),
                    str(getattr(s, 'name', '')),
                    str(getattr(s, 'surface_type', '')),
                    round(float(getattr(s, 'radius', 0.0)), 4),
                    round(float(getattr(s, 'thickness', 0.0)), 4),
                    str(getattr(s, 'glass', '')),
                    round(float(getattr(s, 'semi_diameter', 0.0)), 4),
                    round(float(getattr(s, 'conic', 0.0)), 4),
                    str(getattr(s, 'aperture_type', '')),
                    orig_t,
                    r_t,
                ))
            curr_fingerprint = (
                len(surfaces),
                self.mesh_quality,
                self.no_transparency,
                self.show_air_gaps,
                getattr(self, "air_gap_constant_diameter", 25.4),
                self.show_mounts,
                self.mount_diameter,
                self.mount_z_thickness,
                self.tolerance_thickness,
                self.show_tolerance_envelope,
                self.global_displacement_z,
                tuple(sorted(self.element_displacements.items())),
                self.show_labels,
                self.dark_mode,
                tuple(surf_tokens),
            )

        # Fast incremental ray path update if geometry has not changed
        if (
            not force_rebuild
            and self._last_geom_fingerprint is not None
            and self._last_geom_fingerprint == curr_fingerprint
            and hasattr(self, '_base_all_x')
        ):
            if self.show_rays and self.rays:
                self._render_rays(self.rays, None, None, None)
                self._render_grating_diffraction(surfaces, self.rays, None, None, None)
            else:
                if hasattr(self, "_ray_line_colls"):
                    for coll in self._ray_line_colls.values():
                        coll.set_segments([])

            if hasattr(self.canvas, "draw_idle"):
                self.canvas.draw_idle()
            elif hasattr(self.canvas, "draw"):
                self.canvas.draw()
            return

        self.ax.clear()
        self._ray_artists.clear()
        if hasattr(self, "_ray_line_colls"):
            self._ray_line_colls.clear()
        if hasattr(self, "_ray_text_artists"):
            self._ray_text_artists.clear()
        if hasattr(self, "_grating_artists"):
            self._grating_artists.clear()
        self._last_ray_annot_key = None
        self._last_grating_key = None

        bg_fig = '#0f172a' if self.dark_mode else '#ffffff'
        self.fig.patch.set_facecolor(bg_fig)
        if hasattr(self.ax, 'set_facecolor'):
            self.ax.set_facecolor(bg_fig)

        self.ax.set_title("")
        axis_lbl_col = '#94a3b8' if self.dark_mode else '#64748b'
        self.ax.set_xlabel("Z (mm)", fontsize=8, color=axis_lbl_col, labelpad=2)
        self.ax.set_ylabel("X (mm)", fontsize=8, color=axis_lbl_col, labelpad=2)
        self.ax.set_zlabel("Y (mm)", fontsize=8, color=axis_lbl_col, labelpad=2)

        if hasattr(self.ax, 'xaxis') and hasattr(self.ax.xaxis, 'set_pane_color'):
            pane_alpha = (0.35 if self.dark_mode else 0.45) if self.show_grid else 0.0
            pane_color = (0.07, 0.10, 0.16, pane_alpha) if self.dark_mode else (0.97, 0.98, 0.99, pane_alpha)
            self.ax.xaxis.set_pane_color(pane_color)
            self.ax.yaxis.set_pane_color(pane_color)
            self.ax.zaxis.set_pane_color(pane_color)
            spine_color = '#334155' if self.dark_mode else '#cbd5e1'
            for axis in (self.ax.xaxis, self.ax.yaxis, self.ax.zaxis):
                if hasattr(axis, 'line'):
                    axis.line.set_color(spine_color if self.show_grid else (0, 0, 0, 0))
        grid_color = '#334155' if self.dark_mode else '#cbd5e1'
        self.ax.grid(self.show_grid, linestyle=':', alpha=0.45, color=grid_color)
        if hasattr(self.ax, 'tick_params'):
            tick_col = '#94a3b8' if self.dark_mode else '#64748b'
            self.ax.tick_params(labelsize=7.5, colors=tick_col, pad=1)

        all_x, all_y, all_z = [], [], []

        if stack is not None:
            solid_face_ids = set()
            prism_bs_face_ids = set()
            mesh_radial, mesh_angular = (12, 64) if self.no_transparency else ((6, 32) if self.mesh_quality == "interactive" else (9, 48))

            def add_mesh_bounds(mesh):
                x_mesh, y_mesh, z_mesh = mesh
                all_x.extend(np.asarray(x_mesh).ravel())
                all_y.extend(np.asarray(y_mesh).ravel())
                all_z.extend(np.asarray(z_mesh).ravel())

            # Seed all surface origins and clear aperture extents to guarantee full coverage from first surface to end
            for s in surfaces:
                if bool(getattr(s, "is_air_space", False)):
                    continue
                pos = getattr(s, 'global_origin', None)
                if pos is not None:
                    pos = np.asarray(pos, dtype=float)
                    sd_val = float(getattr(s, 'semi_diameter', 0.0))
                    if sd_val <= 0.0 or str(getattr(s, 'aperture_type', '')).lower() in ('point', 'none'):
                        all_x.extend([pos[0] - 0.5, pos[0] + 0.5])
                        all_y.extend([pos[1] - 0.5, pos[1] + 0.5])
                    else:
                        all_x.extend([pos[0] - sd_val, pos[0] + sd_val])
                        all_y.extend([pos[1] - sd_val, pos[1] + sd_val])
                    all_z.append(pos[2])
            if not all_x:
                all_x, all_y, all_z = [0.0], [0.0], [0.0]

            # 0. Specialized 3D solids for generic components: Prisms and Beam Splitters
            from optics_app.cad.exporter import get_exportable_elements
            try:
                exportable_elements = get_exportable_elements(self.surface_stack) if self.surface_stack else []
            except Exception:
                exportable_elements = []
            self._update_element_displace_combo(exportable_elements)

            surf_to_elem: Dict[int, Dict[str, Any]] = {}
            for elem in exportable_elements:
                for s in elem.get("surfaces", []):
                    surf_to_elem[id(s)] = elem

            def get_surf_displacement(surf_obj: Any) -> np.ndarray:
                elem_info = surf_to_elem.get(id(surf_obj))
                e_i = elem_info.get("index", -1) if elem_info else -1
                dz_val = self.element_displacements.get(e_i, 0.0) + self.global_displacement_z
                if abs(dz_val) < 1e-5:
                    return np.zeros(3)
                _, R_s = get_effective_pose(surf_obj)
                return R_s[:, 2] * dz_val

            def displace_mesh(m_tuple, d_vec):
                if not np.any(d_vec != 0.0):
                    return m_tuple
                xm, ym, zm = m_tuple
                return (xm + d_vec[0], ym + d_vec[1], zm + d_vec[2])

            def displace_ring(r_tuple, d_vec):
                if not np.any(d_vec != 0.0):
                    return r_tuple
                rx, ry, rz = r_tuple
                return (rx + d_vec[0], ry + d_vec[1], rz + d_vec[2])

            for elem in exportable_elements:
                elem_type = elem.get("type", "")
                elem_surfs = elem.get("surfaces", [])
                if not elem_surfs:
                    continue
                if elem_type == "Prism":
                    s_front = elem_surfs[0]
                    s_rear = elem_surfs[1] if len(elem_surfs) > 1 else None
                    plot_faces, world_verts = build_prism_geometry(
                        s_front, s_rear, config=elem.get("config")
                    )
                    d_prism = get_surf_displacement(s_front)
                    if np.any(d_prism != 0.0):
                        world_verts = world_verts + d_prism
                        plot_faces = [
                            [(pt[0] + d_prism[2], pt[1] + d_prism[0], pt[2] + d_prism[1]) for pt in f]
                            for f in plot_faces
                        ]
                    if plot_faces:
                        prism_alpha = 1.0 if self.no_transparency else 0.55
                        prism_color = getattr(s_front, 'color', None) or '#93c5fd'
                        prism_edge = 'none' if self.no_transparency else '#1d4ed8'
                        prism_body = Poly3DCollection(
                            plot_faces, facecolors=prism_color, edgecolors=prism_edge,
                            linewidths=0.0 if self.no_transparency else 0.75,
                            alpha=prism_alpha, shade=True, zorder=8
                        )
                        self.ax.add_collection3d(prism_body)
                        for s in elem_surfs:
                            solid_face_ids.add(id(s))
                            prism_bs_face_ids.add(id(s))
                        all_x.extend(world_verts[:, 0])
                        all_y.extend(world_verts[:, 1])
                        all_z.extend(world_verts[:, 2])

                elif elem_type == "Beam Splitter":
                    s_front = elem_surfs[0]
                    s_rear = elem_surfs[1] if len(elem_surfs) > 1 else None
                    outer_faces, iface_faces, world_verts = build_beamsplitter_geometry(
                        s_front, s_rear, config=elem.get("config")
                    )
                    d_bs = get_surf_displacement(s_front)
                    if np.any(d_bs != 0.0):
                        world_verts = world_verts + d_bs
                        if outer_faces:
                            outer_faces = [
                                [(pt[0] + d_bs[2], pt[1] + d_bs[0], pt[2] + d_bs[1]) for pt in f]
                                for f in outer_faces
                            ]
                        if iface_faces:
                            iface_faces = [
                                [(pt[0] + d_bs[2], pt[1] + d_bs[0], pt[2] + d_bs[1]) for pt in f]
                                for f in iface_faces
                            ]
                    if outer_faces:
                        bs_alpha = 1.0 if self.no_transparency else 0.45
                        bs_color = getattr(s_front, 'color', None) or '#bae6fd'
                        bs_edge = 'none' if self.no_transparency else '#0284c7'
                        bs_body = Poly3DCollection(
                            outer_faces, facecolors=bs_color, edgecolors=bs_edge,
                            linewidths=0.0 if self.no_transparency else 0.6,
                            alpha=bs_alpha, shade=True, zorder=8
                        )
                        self.ax.add_collection3d(bs_body)
                    if iface_faces:
                        iface_alpha = 1.0 if self.no_transparency else 0.85
                        iface_body = Poly3DCollection(
                            iface_faces, facecolors='#f59e0b', edgecolors='#d97706',
                            linewidths=1.0, alpha=iface_alpha, shade=True, zorder=12
                        )
                        self.ax.add_collection3d(iface_body)
                    if outer_faces or iface_faces:
                        for s in elem_surfs:
                            solid_face_ids.add(id(s))
                            prism_bs_face_ids.add(id(s))
                        all_x.extend(world_verts[:, 0])
                        all_y.extend(world_verts[:, 1])
                        all_z.extend(world_verts[:, 2])

            # 1. Closed solid bodies for optical glass elements
            groups = compute_surface_groups(surfaces)
            lens_solid_idx = 0
            for index, front in enumerate(surfaces[:-1]):
                if id(front) in prism_bs_face_ids:
                    continue
                front_role = classify_surface_for_3d(front, index, len(surfaces))
                is_glass_region = (
                    not any(front_role.values())
                    and front.surface_type in (SurfaceType.STANDARD, SurfaceType.ASPHERIC, SurfaceType.CAD)
                    and getattr(front, 'glass', '').upper() not in ('AIR', '', 'NONE', 'MIRROR', 'MIRR', 'REFL')
                    and float(getattr(front, 'thickness', 0.0)) > 0.0
                )
                is_air_region = (
                    self.show_air_gaps
                    and (
                        getattr(front, 'glass', '').upper() in ('AIR', '', 'NONE')
                        or front_role.get("mirror") or front_role.get("grating") or front_role.get("dmd")
                    )
                    and 0.0 < float(getattr(front, 'thickness', 0.0)) < 1000.0
                )
                if not (is_glass_region or is_air_region):
                    continue
                rear = surfaces[index + 1]
                grp = groups[index] if index < len(groups) and groups[index] is not None else {}

                # Deterministic element ID and palette for the same lens assembly:
                eid = None
                if getattr(front, 'element_id', None) is not None and front.element_id >= 0:
                    eid = int(front.element_id)
                elif grp.get("element_id") is not None and grp["element_id"] >= 0:
                    eid = int(grp["element_id"])
                elif getattr(rear, 'element_id', None) is not None and rear.element_id >= 0:
                    eid = int(rear.element_id)
                else:
                    eid = lens_solid_idx

                pal = grp.get("palette") or LENS_ELEMENT_PALETTES[eid % len(LENS_ELEMENT_PALETTES)]

                custom_color = (
                    getattr(front, 'color', None)
                    or getattr(front, 'element_color', None)
                    or getattr(rear, 'color', None)
                    or getattr(rear, 'element_color', None)
                )
                if is_air_region:
                    color = "#4fc3f7"  # Light transparent blue for air gaps
                    rim_color = "#29b6f6"
                    glass_alpha = 0.10 if not self.no_transparency else 1.0
                    edge_width = 0.5
                    edge_color = "#0288d1"
                    
                    cyl_diam = float(getattr(self, "air_gap_constant_diameter", 25.4) or 25.4)
                    if cyl_diam <= 0.0:
                        cyl_diam = 25.4
                    cyl_rad = cyl_diam / 2.0
                    front_mesh, rear_mesh, side_mesh = build_cylinder_geometry(
                        surf=front,
                        radius=cyl_rad,
                        thickness=float(getattr(front, 'thickness', 10.0)),
                        radial_samples=mesh_radial, angular_samples=mesh_angular,
                        rear_surf=rear,
                    )
                else:
                    front_mesh, rear_mesh, side_mesh = build_lens_solid_geometry(
                        front, rear, mesh_radial, mesh_angular, cache=self._mesh_cache,
                    )
                    color = custom_color or pal.get("fill_hex", "#4FC3F7")
                    rim_color = custom_color or pal.get("border_hex", "#0288D1")
                    glass_alpha = 1.0 if self.no_transparency else 0.50
                    edge_color = 'none' if self.no_transparency else rim_color
                    edge_width = 0.0 if self.no_transparency else 0.25
                d_front = get_surf_displacement(front)
                front_mesh = displace_mesh(front_mesh, d_front)
                rear_mesh = displace_mesh(rear_mesh, d_front)
                side_mesh = displace_mesh(side_mesh, d_front)

                for mesh in (front_mesh, rear_mesh, side_mesh):
                    x_mesh, y_mesh, z_mesh = mesh
                    self.ax.plot_surface(
                        z_mesh, x_mesh, y_mesh,
                        color=color, alpha=glass_alpha, shade=True,
                        edgecolor=edge_color, linewidth=edge_width,
                        antialiased=True,
                    )
                    add_mesh_bounds(mesh)
                # Draw smooth perimeter silhouette rings for the front and rear
                if is_air_region:
                    front_ring = (side_mesh[0][0], side_mesh[1][0], side_mesh[2][0])
                    rear_ring = (side_mesh[0][1], side_mesh[1][1], side_mesh[2][1])
                else:
                    front_ring = displace_ring(_edge_ring(front, mesh_angular, cache=self._mesh_cache), d_front)
                    rear_ring = displace_ring(_edge_ring(rear, mesh_angular, cache=self._mesh_cache), d_front)
                self.ax.plot(front_ring[2], front_ring[0], front_ring[1], color=rim_color, linewidth=1.2, zorder=9)
                self.ax.plot(rear_ring[2], rear_ring[0], rear_ring[1], color=rim_color, linewidth=1.2, zorder=9)
                if is_glass_region:
                    solid_face_ids.update((id(front), id(rear)))
                    is_prev_glass = (
                        index > 0
                        and getattr(surfaces[index - 1], 'glass', '').upper() not in ('AIR', '', 'NONE', 'MIRROR', 'MIRR', 'REFL')
                        and float(getattr(surfaces[index - 1], 'thickness', 0.0)) > 0.0
                        and groups[index - 1] is not None
                        and groups[index - 1].get('element_id') == grp.get('element_id')
                    )
                    if not is_prev_glass:
                        k = index
                        elem_tot_t = 0.0
                        elem_max_sd = float(getattr(front, 'semi_diameter', 0.0) or 0.0)
                        while k < len(surfaces) - 1:
                            sk = surfaces[k]
                            sk_glass = getattr(sk, 'glass', '').upper()
                            if sk_glass in ('AIR', '', 'NONE', 'MIRROR', 'MIRR', 'REFL') or float(getattr(sk, 'thickness', 0.0)) <= 0.0:
                                break
                            if k > index and groups[k] is not None and groups[k].get('element_id') != grp.get('element_id'):
                                break
                            elem_tot_t += float(getattr(sk, 'thickness', 0.0))
                            elem_max_sd = max(elem_max_sd, float(getattr(sk, 'semi_diameter', 0.0) or 0.0), float(getattr(surfaces[k + 1], 'semi_diameter', 0.0) or 0.0))
                            k += 1

                        inner_r = max(elem_max_sd, 1.0)

                        # Tolerance Bounds Envelope Visualization (+/- tol mm)
                        if self.show_tolerance_envelope and self.tolerance_thickness > 0:
                            t_cyl, t_min_r, t_max_r = build_tolerance_envelope_geometry(
                                surf=front,
                                radius=inner_r + 0.35,
                                thickness=elem_tot_t,
                                tolerance=self.tolerance_thickness,
                                z_offset=0.0,
                                angular_samples=mesh_angular,
                            )
                            t_cyl = displace_mesh(t_cyl, d_front)
                            t_min_r = displace_ring(t_min_r, d_front)
                            t_max_r = displace_ring(t_max_r, d_front)
                            tcx, tcy, tcz = t_cyl
                            self.ax.plot_surface(
                                tcz, tcx, tcy,
                                color="#f59e0b", alpha=0.18, shade=True,
                                edgecolor="#d97706", linewidth=0.3, antialiased=True,
                            )
                            add_mesh_bounds(t_cyl)
                            self.ax.plot(t_min_r[2], t_min_r[0], t_min_r[1], color="#d97706", linestyle="--", linewidth=1.2, zorder=8)
                            self.ax.plot(t_max_r[2], t_max_r[0], t_max_r[1], color="#d97706", linestyle="--", linewidth=1.2, zorder=8)

                        # Mechanical Mount Collar / Cell (Optional)
                        if self.show_mounts:
                            user_diam = float(getattr(self, "mount_diameter", 30.5) or 30.5)
                            if user_diam <= 0.0:
                                outer_r = inner_r + 2.5
                            else:
                                outer_r = user_diam / 2.0
                                if outer_r <= inner_r:
                                    outer_r = inner_r + 2.5

                            user_z = float(getattr(self, "mount_z_thickness", 0.0) or 0.0)
                            if user_z <= 0.0:
                                mount_thick = max(elem_tot_t, 2.0)
                                mount_z_off = 0.0
                            else:
                                mount_thick = user_z
                                mount_z_off = (elem_tot_t - mount_thick) / 2.0

                            m_outer, m_f_rim, m_r_rim, m_f_ring, m_r_ring = build_mount_collar_geometry(
                                surf=front,
                                inner_radius=inner_r,
                                outer_radius=outer_r,
                                thickness=mount_thick,
                                z_offset=mount_z_off,
                                radial_samples=mesh_radial,
                                angular_samples=mesh_angular,
                            )
                            m_outer = displace_mesh(m_outer, d_front)
                            m_f_rim = displace_mesh(m_f_rim, d_front)
                            m_r_rim = displace_mesh(m_r_rim, d_front)
                            m_f_ring = displace_ring(m_f_ring, d_front)
                            m_r_ring = displace_ring(m_r_ring, d_front)

                            m_color = "#1e293b" if self.no_transparency else "#334155"
                            m_alpha = 1.0 if self.no_transparency else 0.88
                            m_rim_color = "#64748b"

                            for m_mesh in (m_outer, m_f_rim, m_r_rim):
                                mx, my, mz = m_mesh
                                self.ax.plot_surface(
                                    mz, mx, my,
                                    color=m_color, alpha=m_alpha, shade=True,
                                    edgecolor='none' if self.no_transparency else '#1e293b',
                                    linewidth=0.2, antialiased=True,
                                )
                                add_mesh_bounds(m_mesh)

                            self.ax.plot(m_f_ring[2], m_f_ring[0], m_f_ring[1], color=m_rim_color, linewidth=1.2, zorder=8)
                            self.ax.plot(m_r_ring[2], m_r_ring[0], m_r_ring[1], color=m_rim_color, linewidth=1.2, zorder=8)

                # Advance lens_solid_idx only when moving to next lens assembly (air boundary or distinct element_id)
                is_next_glass = (
                    index + 1 < len(surfaces) - 1
                    and getattr(rear, 'glass', '').upper() not in ('AIR', '', 'NONE', 'MIRROR', 'MIRR', 'REFL')
                    and float(getattr(rear, 'thickness', 0.0)) > 0.0
                )
                next_grp = groups[index + 1] if index + 1 < len(groups) and groups[index + 1] is not None else {}
                if not is_next_glass or next_grp.get("element_id") != grp.get("element_id"):
                    lens_solid_idx += 1

            # 2. Reflective, diffraction, detector, stop, aperture, object, and curved surfaces
            for surf_idx, surf in enumerate(surfaces):
                if bool(getattr(surf, "is_air_space", False)):
                    continue
                role = classify_surface_for_3d(surf, surf_idx, len(surfaces))
                is_mirror = role["mirror"]
                is_grating = role["grating"]
                is_detector = role["detector"]
                is_dmd = role["dmd"]
                is_stop = role["stop"]
                is_object = role["object"]
                ap_type = str(getattr(surf, 'aperture_type', 'Circular')).lower()
                has_complex_aperture = (
                    ap_type in ('rectangular', 'slit', 'elliptical', 'spider')
                    or getattr(surf, 'aperture_r_inner', 0.0) > 0.0
                    or getattr(surf, 'aperture_r_outer', 0.0) > 0.0
                )
                is_curved = (surf.radius != 0.0 and not np.isinf(surf.radius) and getattr(surf, 'glass', '').upper() not in ('AIR', '', 'NONE'))

                # Render Aperture Stops and Custom Surface Apertures
                if self.show_apertures and not (is_mirror or is_grating or is_dmd or is_detector) and (is_stop or has_complex_aperture):
                    d_next = float(getattr(surf, 'thickness', 0.0))
                    stop_z_mode = "backward" if (d_next == 0.0 and surf_idx < len(surfaces) - 1) else "forward"
                    max_sys_sd = max([float(getattr(s, 'semi_diameter', 0.0)) for s in surfaces] or [0.0])

                    if ap_type in ('rectangular', 'slit'):
                        if id(surf) not in solid_face_ids:
                            plot_faces, inner_rim, outer_rim, all_verts = build_rectangular_aperture_geometry(
                                surf, surrounding_max_sd=max_sys_sd, z_offset_mode=stop_z_mode,
                            )
                            stop_alpha = 1.0 if self.no_transparency else 0.82
                            body = Poly3DCollection(
                                plot_faces, facecolors='#1e293b' if self.no_transparency else '#37474f',
                                edgecolors='none' if self.no_transparency else '#263238',
                                linewidths=0.0 if self.no_transparency else 0.5,
                                alpha=stop_alpha
                            )
                            if hasattr(body, 'set_picker'): body.set_picker(5); body._surface_index = surf_idx

                            self.ax.add_collection3d(body)
                            # Inner aperture rim in stop red
                            self.ax.plot(inner_rim[2], inner_rim[0], inner_rim[1], color='#ef4444', linewidth=2.5, zorder=12)
                            self.ax.plot(outer_rim[2], outer_rim[0], outer_rim[1], color='#475569' if self.no_transparency else '#212121', linewidth=1.2, zorder=9)
                            all_x.extend(all_verts[:, 0])
                            all_y.extend(all_verts[:, 1])
                            all_z.extend(all_verts[:, 2])

                        c_slit = np.asarray(getattr(surf, 'global_origin', [0.0, 0.0, 0.0]), dtype=float)
                        R_slit = np.asarray(getattr(surf, 'global_R', np.eye(3)), dtype=float)
                        hx_s = float(getattr(surf, 'aperture_x_half', 0.0) or getattr(surf, 'semi_diameter', 0.0))
                        hy_s = float(getattr(surf, 'aperture_y_half', 0.0) or getattr(surf, 'semi_diameter', 0.0))
                        if hx_s > 0 and hy_s > 0:
                            # Full X caliper line from -hx_s to +hx_s with perpendicular end ticks
                            p_hx_neg = c_slit + R_slit @ np.array([-hx_s, 0.0, 0.001])
                            p_hx_pos = c_slit + R_slit @ np.array([ hx_s, 0.0, 0.001])
                            self.ax.plot([p_hx_neg[2], p_hx_pos[2]], [p_hx_neg[0], p_hx_pos[0]], [p_hx_neg[1], p_hx_pos[1]],
                                         color='#ef4444', linestyle='--', linewidth=1.5, zorder=13)
                            tick_y = max(hy_s * 0.12, 0.6)
                            t_x1_a = c_slit + R_slit @ np.array([-hx_s, -tick_y, 0.001])
                            t_x1_b = c_slit + R_slit @ np.array([-hx_s,  tick_y, 0.001])
                            t_x2_a = c_slit + R_slit @ np.array([ hx_s, -tick_y, 0.001])
                            t_x2_b = c_slit + R_slit @ np.array([ hx_s,  tick_y, 0.001])
                            self.ax.plot([t_x1_a[2], t_x1_b[2]], [t_x1_a[0], t_x1_b[0]], [t_x1_a[1], t_x1_b[1]],
                                         color='#ef4444', linewidth=1.8, zorder=13)
                            self.ax.plot([t_x2_a[2], t_x2_b[2]], [t_x2_a[0], t_x2_b[0]], [t_x2_a[1], t_x2_b[1]],
                                         color='#ef4444', linewidth=1.8, zorder=13)

                            # Full Y caliper line from -hy_s to +hy_s with perpendicular end ticks
                            p_hy_neg = c_slit + R_slit @ np.array([0.0, -hy_s, 0.001])
                            p_hy_pos = c_slit + R_slit @ np.array([0.0,  hy_s, 0.001])
                            self.ax.plot([p_hy_neg[2], p_hy_pos[2]], [p_hy_neg[0], p_hy_pos[0]], [p_hy_neg[1], p_hy_pos[1]],
                                         color='#ef4444', linestyle='--', linewidth=1.5, zorder=13)
                            tick_x = max(hx_s * 0.12, 0.6)
                            t_y1_a = c_slit + R_slit @ np.array([-tick_x, -hy_s, 0.001])
                            t_y1_b = c_slit + R_slit @ np.array([ tick_x, -hy_s, 0.001])
                            t_y2_a = c_slit + R_slit @ np.array([-tick_x,  hy_s, 0.001])
                            t_y2_b = c_slit + R_slit @ np.array([ tick_x,  hy_s, 0.001])
                            self.ax.plot([t_y1_a[2], t_y1_b[2]], [t_y1_a[0], t_y1_b[0]], [t_y1_a[1], t_y1_b[1]],
                                         color='#ef4444', linewidth=1.8, zorder=13)
                            self.ax.plot([t_y2_a[2], t_y2_b[2]], [t_y2_a[0], t_y2_b[0]], [t_y2_a[1], t_y2_b[1]],
                                         color='#ef4444', linewidth=1.8, zorder=13)

                            self.ax.scatter([c_slit[2]], [c_slit[0]], [c_slit[1]],
                                            color='#ef4444', s=24, zorder=14)

                            # Slit caliper readout badge
                            if self.show_labels:
                                r_badge = max(max(hx_s, hy_s) * 1.5, max_sys_sd * 1.08, max(hx_s, hy_s) + 4.0)
                                p_badge = c_slit + R_slit @ np.array([0.0, r_badge * 0.95, 0.002])
                                self.ax.text(p_badge[2], p_badge[0], p_badge[1],
                                             f"Stop [{hx_s:.2f}×{hy_s:.2f} mm] (Semi-D X={hx_s:.2f}, Y={hy_s:.2f} mm | Size: {2*hx_s:.2f} × {2*hy_s:.2f} mm)",
                                             color='#b71c1c', fontsize=8, fontweight='bold',
                                             bbox=dict(boxstyle='round,pad=0.25', facecolor='#ffebee', edgecolor='#ef4444', alpha=0.92, linewidth=0.9),
                                             zorder=15)
                        if id(surf) not in solid_face_ids:
                            continue
                    else:
                        c_stop = np.asarray(getattr(surf, 'global_origin', [0.0, 0.0, 0.0]), dtype=float)
                        R_stop = np.asarray(getattr(surf, 'global_R', np.eye(3)), dtype=float)
                        sd_stop = float(getattr(surf, 'semi_diameter', 0.0))
                        rin = float(getattr(surf, 'aperture_r_inner', 0.0))
                        is_annular = ap_type == 'annular' or rin > 0.0

                        if id(surf) not in solid_face_ids:
                            stop_meshes, inner_rim, outer_rim = build_aperture_stop_geometry(
                                surf, angular_samples=mesh_angular, surrounding_max_sd=max_sys_sd,
                                z_offset_mode=stop_z_mode,
                            )
                            annular_alpha = 1.0 if self.no_transparency else 0.80
                            stop_color = '#1e293b' if self.no_transparency else '#37474f'
                            for x_mesh, y_mesh, z_mesh in stop_meshes:
                                self.ax.plot_surface(
                                    z_mesh, x_mesh, y_mesh,
                                    color=stop_color, alpha=annular_alpha, shade=True,
                                    edgecolor='none' if self.no_transparency else '#263238',
                                    linewidth=0.0 if self.no_transparency else 0.35,
                                    antialiased=True
                                )
                            # Clear aperture inner hole in Stop Red
                            self.ax.plot(inner_rim[2], inner_rim[0], inner_rim[1], color='#ef4444', linewidth=2.5, zorder=12)
                            # Outer diaphragm rim in dark metal
                            self.ax.plot(outer_rim[2], outer_rim[0], outer_rim[1], color='#475569' if self.no_transparency else '#212121', linewidth=1.2, zorder=9)
                            for mesh in stop_meshes:
                                add_mesh_bounds(mesh)

                        # Full diameter caliper line across clear aperture opening along X
                        if sd_stop > 0:
                            p_d_neg = c_stop + R_stop @ np.array([-sd_stop, 0.0, 0.001])
                            p_d_pos = c_stop + R_stop @ np.array([ sd_stop, 0.0, 0.001])
                            self.ax.plot([p_d_neg[2], p_d_pos[2]], [p_d_neg[0], p_d_pos[0]], [p_d_neg[1], p_d_pos[1]],
                                         color='#ef4444', linestyle='-', linewidth=1.6, alpha=0.9, zorder=13)
                            tick_len = max(sd_stop * 0.12, 0.6)
                            t_neg1 = c_stop + R_stop @ np.array([-sd_stop, -tick_len, 0.001])
                            t_neg2 = c_stop + R_stop @ np.array([-sd_stop,  tick_len, 0.001])
                            t_pos1 = c_stop + R_stop @ np.array([ sd_stop, -tick_len, 0.001])
                            t_pos2 = c_stop + R_stop @ np.array([ sd_stop,  tick_len, 0.001])
                            self.ax.plot([t_neg1[2], t_neg2[2]], [t_neg1[0], t_neg2[0]], [t_neg1[1], t_neg2[1]],
                                         color='#ef4444', linewidth=2.0, zorder=13)
                            self.ax.plot([t_pos1[2], t_pos2[2]], [t_pos1[0], t_pos2[0]], [t_pos1[1], t_pos2[1]],
                                         color='#ef4444', linewidth=2.0, zorder=13)

                            # Radial semi-diameter measurement indicator along Y from center to rim
                            p_rim_pt = c_stop + R_stop @ np.array([0.0, sd_stop, 0.001])
                            self.ax.plot([c_stop[2], p_rim_pt[2]], [c_stop[0], p_rim_pt[0]], [c_stop[1], p_rim_pt[1]],
                                         color='#ef4444', linestyle='--', linewidth=1.8, zorder=13)
                            self.ax.scatter([c_stop[2]], [c_stop[0]], [c_stop[1]],
                                            color='#ef4444', s=28, marker='o', zorder=14)
                            self.ax.scatter([p_rim_pt[2]], [p_rim_pt[0]], [p_rim_pt[1]],
                                            color='#ef4444', s=22, marker='s', zorder=14)

                            # In-situ semi-diameter measurement text on the radial arm
                            if self.show_labels:
                                p_arm_mid = c_stop + R_stop @ np.array([0.0, sd_stop * 0.50, 0.002])
                                self.ax.text(
                                    p_arm_mid[2], p_arm_mid[0], p_arm_mid[1],
                                    f" Semi-D: R = {sd_stop:.2f} mm ",
                                    color='#b71c1c', fontsize=7.5, fontweight='bold',
                                    ha='left', va='center',
                                    bbox=dict(boxstyle='round,pad=0.15', facecolor='#fee2e2', edgecolor='#ef4444', alpha=0.92, linewidth=0.8),
                                    zorder=15
                                )

                                # Stop dimension readout badge on outer diaphragm collar
                                r_badge = max(sd_stop * 1.5, max_sys_sd * 1.08, sd_stop + 4.0)
                                p_badge = c_stop + R_stop @ np.array([0.0, r_badge * 0.95, 0.002])
                                if is_annular:
                                    badge_txt = f"Stop Annular: Inner R = {rin:.2f} mm | Outer Semi-D = {sd_stop:.2f} mm (Ø = {2*sd_stop:.2f} mm) | R={sd_stop:.2f} mm"
                                else:
                                    badge_txt = f"Stop Clear Aperture: Semi-D = {sd_stop:.2f} mm | Ø = {2*sd_stop:.2f} mm (R={sd_stop:.2f} mm)"
                                self.ax.text(
                                    p_badge[2], p_badge[0], p_badge[1],
                                    badge_txt, color='#b71c1c', fontsize=8.2, fontweight='bold',
                                    ha='center', va='bottom',
                                    bbox=dict(boxstyle='round,pad=0.25', facecolor='#ffebee', edgecolor='#ef4444', alpha=0.95, linewidth=1.0),
                                    zorder=15
                                )
                        if id(surf) not in solid_face_ids:
                            continue

                # Render an explicitly declared object or sample plane.
                if is_object:
                    center = np.asarray(getattr(surf, 'global_origin', [0.0, 0.0, 0.0]), dtype=float)
                    rotation = np.asarray(getattr(surf, 'global_R', np.eye(3)), dtype=float)
                    sd = float(getattr(surf, 'semi_diameter', 0.0))
                    ap_type_str = str(getattr(surf, 'aperture_type', 'Circular')).lower()
                    is_rect = ap_type_str in ('rectangular', 'slit')
                    hx_raw = float(getattr(surf, 'aperture_x_half', 0.0)) if is_rect else 0.0
                    hy_raw = float(getattr(surf, 'aperture_y_half', 0.0)) if is_rect else 0.0

                    is_point_object = (
                        ap_type_str in ('point', 'point object', 'none')
                        or (is_rect and hx_raw <= 0.0 and hy_raw <= 0.0 and sd <= 0.0)
                        or (not is_rect and sd <= 0.0)
                    )

                    if is_point_object:
                        # 1. Point Object: compact luminous emitter marker & crosshair reticle
                        self.ax.scatter([center[2]], [center[0]], [center[1]],
                                        color='#0284c7', s=70, edgecolors='#0369a1',
                                        linewidths=1.5, zorder=16)
                        reticle_r = 0.8
                        p_cx1 = rotation @ np.array([-reticle_r, 0.0, 0.0]) + center
                        p_cx2 = rotation @ np.array([ reticle_r, 0.0, 0.0]) + center
                        p_cy1 = rotation @ np.array([0.0, -reticle_r, 0.0]) + center
                        p_cy2 = rotation @ np.array([0.0,  reticle_r, 0.0]) + center
                        self.ax.plot([p_cx1[2], p_cx2[2]], [p_cx1[0], p_cx2[0]], [p_cx1[1], p_cx2[1]],
                                     color='#0284c7', linewidth=1.5, zorder=15)
                        self.ax.plot([p_cy1[2], p_cy2[2]], [p_cy1[0], p_cy2[0]], [p_cy1[1], p_cy2[1]],
                                     color='#0284c7', linewidth=1.5, zorder=15)
                        all_x.extend([center[0] - reticle_r, center[0] + reticle_r])
                        all_y.extend([center[1] - reticle_r, center[1] + reticle_r])
                        all_z.append(center[2])
                        continue

                    # 2. Extended Sample Stage Mount
                    hx = hx_raw if hx_raw > 0.0 else sd
                    hy = hy_raw if hy_raw > 0.0 else sd
                    t_stage = float(getattr(surf, 'substrate_thickness', 0.0) or 1.8)

                    local_vertices = np.array([
                        [-hx, -hy, 0.0], [hx, -hy, 0.0], [hx, hy, 0.0], [-hx, hy, 0.0],
                        [-hx, -hy, -t_stage], [hx, -hy, -t_stage], [hx, hy, -t_stage], [-hx, hy, -t_stage],
                    ], dtype=float)
                    world_vertices = (rotation @ local_vertices.T).T + center
                    face_indices = (
                        (0, 1, 2, 3), (4, 7, 6, 5),
                        (0, 4, 5, 1), (1, 5, 6, 2),
                        (2, 6, 7, 3), (3, 7, 4, 0),
                    )

                    # Front sample / target face at z = 0
                    front_face = [[(world_vertices[i, 2], world_vertices[i, 0], world_vertices[i, 1]) for i in (0, 1, 2, 3)]]
                    front_body = Poly3DCollection(
                        front_face, facecolors='#38bdf8' if not self.no_transparency else '#0284c7',
                        edgecolors='#0369a1',
                        linewidths=1.2,
                        alpha=0.85 if not self.no_transparency else 1.0,
                        zorder=10,
                    )
                    if hasattr(front_body, 'set_picker'): front_body.set_picker(5); front_body._surface_index = surf_idx

                    self.ax.add_collection3d(front_body)

                    # Solid mount substrate (sides and back)
                    sub_faces = [
                        [(world_vertices[i, 2], world_vertices[i, 0], world_vertices[i, 1]) for i in face]
                        for face in face_indices[1:]
                    ]
                    sub_body = Poly3DCollection(
                        sub_faces, facecolors='#334155',
                        edgecolors='#1e293b',
                        linewidths=0.6 if self.no_transparency else 0.5,
                        alpha=0.90 if not self.no_transparency else 1.0,
                        zorder=8,
                    )
                    if hasattr(sub_body, 'set_picker'): sub_body.set_picker(5); sub_body._surface_index = surf_idx

                    self.ax.add_collection3d(sub_body)

                    # Reticle crosshair on front face
                    p_cx1 = rotation @ np.array([-hx * 0.7, 0.0, 0.001]) + center
                    p_cx2 = rotation @ np.array([ hx * 0.7, 0.0, 0.001]) + center
                    p_cy1 = rotation @ np.array([0.0, -hy * 0.7, 0.001]) + center
                    p_cy2 = rotation @ np.array([0.0,  hy * 0.7, 0.001]) + center
                    self.ax.plot([p_cx1[2], p_cx2[2]], [p_cx1[0], p_cx2[0]], [p_cx1[1], p_cx2[1]],
                                 color='#0c4a6e', linewidth=1.0, zorder=12)
                    self.ax.plot([p_cy1[2], p_cy2[2]], [p_cy1[0], p_cy2[0]], [p_cy1[1], p_cy2[1]],
                                 color='#0c4a6e', linewidth=1.0, zorder=12)

                    all_x.extend(world_vertices[:, 0])
                    all_y.extend(world_vertices[:, 1])
                    all_z.extend(world_vertices[:, 2])
                    continue

                # Render an explicitly declared detector or image plane.
                if is_detector:
                    center = np.asarray(getattr(surf, 'global_origin', [0.0, 0.0, 0.0]), dtype=float)
                    rotation = np.asarray(getattr(surf, 'global_R', np.eye(3)), dtype=float)
                    sd = float(getattr(surf, 'semi_diameter', 0.0))
                    ap_type_str = str(getattr(surf, 'aperture_type', 'Circular')).lower()
                    is_rect = ap_type_str in ('rectangular', 'slit')
                    hx_raw = float(getattr(surf, 'aperture_x_half', 0.0)) if is_rect else 0.0
                    hy_raw = float(getattr(surf, 'aperture_y_half', 0.0)) if is_rect else 0.0

                    is_point_image = (
                        ap_type_str in ('point', 'point image', 'focal point', 'none')
                        or (is_rect and hx_raw <= 0.0 and hy_raw <= 0.0 and sd <= 0.0)
                        or (not is_rect and sd <= 0.0)
                    )

                    if is_point_image:
                        # 1. Point Image: sharp focal point marker & crosshair reticle
                        self.ax.scatter([center[2]], [center[0]], [center[1]],
                                        color='#dc2626', s=70, edgecolors='#991b1b',
                                        linewidths=1.5, zorder=16)
                        reticle_r = 0.8
                        p_cx1 = rotation @ np.array([-reticle_r, 0.0, 0.0]) + center
                        p_cx2 = rotation @ np.array([ reticle_r, 0.0, 0.0]) + center
                        p_cy1 = rotation @ np.array([0.0, -reticle_r, 0.0]) + center
                        p_cy2 = rotation @ np.array([0.0,  reticle_r, 0.0]) + center
                        self.ax.plot([p_cx1[2], p_cx2[2]], [p_cx1[0], p_cx2[0]], [p_cx1[1], p_cx2[1]],
                                     color='#dc2626', linewidth=1.5, zorder=15)
                        self.ax.plot([p_cy1[2], p_cy2[2]], [p_cy1[0], p_cy2[0]], [p_cy1[1], p_cy2[1]],
                                     color='#dc2626', linewidth=1.5, zorder=15)
                        all_x.extend([center[0] - reticle_r, center[0] + reticle_r])
                        all_y.extend([center[1] - reticle_r, center[1] + reticle_r])
                        all_z.append(center[2])
                        continue

                    # 2. Extended Physical Sensor Package
                    hx = hx_raw if hx_raw > 0.0 else sd
                    hy = hy_raw if hy_raw > 0.0 else sd
                    t_det = float(getattr(surf, 'substrate_thickness', 0.0) or 2.5)

                    prev_origin = None
                    for p in reversed(surfaces[:surf_idx]):
                        po = getattr(p, 'global_origin', None)
                        if po is not None and np.linalg.norm(center - po) > 1e-4:
                            prev_origin = np.asarray(po, dtype=float)
                            break
                    if prev_origin is not None:
                        d_in = center - prev_origin
                        d_in_norm = np.linalg.norm(d_in)
                        d_in = d_in / d_in_norm if d_in_norm > 1e-6 else rotation[:, 2]
                    else:
                        d_in = rotation[:, 2]
                    sub_sign = 1.0 if np.dot(d_in, rotation[:, 2]) >= 0.0 else -1.0
                    sub_thickness = sub_sign * t_det

                    pkg_hx = hx * 1.35 + 0.5
                    pkg_hy = hy * 1.35 + 0.5

                    pkg_local = np.array([
                        [-pkg_hx, -pkg_hy, 0.0], [pkg_hx, -pkg_hy, 0.0],
                        [pkg_hx, pkg_hy, 0.0], [-pkg_hx, pkg_hy, 0.0],
                        [-pkg_hx, -pkg_hy, sub_thickness], [pkg_hx, -pkg_hy, sub_thickness],
                        [pkg_hx, pkg_hy, sub_thickness], [-pkg_hx, pkg_hy, sub_thickness],
                    ], dtype=float)
                    pkg_world = (rotation @ pkg_local.T).T + center
                    face_indices = (
                        (0, 1, 2, 3), (4, 7, 6, 5),
                        (0, 4, 5, 1), (1, 5, 6, 2),
                        (2, 6, 7, 3), (3, 7, 4, 0),
                    )

                    # 1. Package housing (sides & rear): Dark ceramic/aluminum enclosure
                    housing_faces = [
                        [(pkg_world[i, 2], pkg_world[i, 0], pkg_world[i, 1]) for i in face]
                        for face in face_indices[1:]
                    ]
                    housing_body = Poly3DCollection(
                        housing_faces, facecolors='#1e293b',
                        edgecolors='#0f172a',
                        linewidths=0.6 if self.no_transparency else 0.5,
                        alpha=0.95 if not self.no_transparency else 1.0,
                        zorder=8,
                    )
                    if hasattr(housing_body, 'set_picker'): housing_body.set_picker(5); housing_body._surface_index = surf_idx

                    self.ax.add_collection3d(housing_body)

                    # 2. Package Bezel (front face surrounding active area)
                    bezel_face = [[(pkg_world[i, 2], pkg_world[i, 0], pkg_world[i, 1]) for i in (0, 1, 2, 3)]]
                    bezel_body = Poly3DCollection(
                        bezel_face, facecolors='#0f172a',
                        edgecolors='#475569',
                        linewidths=1.0,
                        alpha=0.95 if not self.no_transparency else 1.0,
                        zorder=9,
                    )
                    if hasattr(bezel_body, 'set_picker'): bezel_body.set_picker(5); bezel_body._surface_index = surf_idx

                    self.ax.add_collection3d(bezel_body)

                    # 3. Active Photosensitive Sensor Chip (front face at z = 0, offset slightly in air)
                    z_chip = -sub_sign * 0.002
                    chip_local = np.array([
                        [-hx, -hy, z_chip], [hx, -hy, z_chip],
                        [hx, hy, z_chip], [-hx, hy, z_chip],
                    ], dtype=float)
                    chip_world = (rotation @ chip_local.T).T + center
                    chip_face = [[(chip_world[i, 2], chip_world[i, 0], chip_world[i, 1]) for i in (0, 1, 2, 3)]]
                    chip_body = Poly3DCollection(
                        chip_face, facecolors='#dc2626' if not self.no_transparency else '#b91c1c',
                        edgecolors='#ef4444',
                        linewidths=1.2,
                        alpha=0.95 if not self.no_transparency else 1.0,
                        zorder=12,
                    )
                    if hasattr(chip_body, 'set_picker'): chip_body.set_picker(5); chip_body._surface_index = surf_idx

                    self.ax.add_collection3d(chip_body)

                    # Sensor reticle line
                    p1 = rotation @ np.array([-hx * 0.8, 0, z_chip - sub_sign * 0.001]) + center
                    p2 = rotation @ np.array([ hx * 0.8, 0, z_chip - sub_sign * 0.001]) + center
                    self.ax.plot([p1[2], p2[2]], [p1[0], p2[0]], [p1[1], p2[1]], color='#fca5a5', linewidth=0.8, zorder=13)

                    all_x.extend(pkg_world[:, 0])
                    all_y.extend(pkg_world[:, 1])
                    all_z.extend(pkg_world[:, 2])
                    continue

                if id(surf) in solid_face_ids and not (is_mirror or is_grating or is_dmd or is_stop or has_complex_aperture):
                    continue
                grp = groups[surf_idx] if surf_idx < len(groups) and groups[surf_idx] is not None else {}
                custom_color = getattr(surf, 'color', None) or getattr(surf, 'element_color', None)
                is_generic_surface = (
                    surf.surface_type in (SurfaceType.STANDARD, SurfaceType.ASPHERIC, SurfaceType.CAD)
                    and not any(role.values())
                )
                d_surf = get_surf_displacement(surf)
                if is_generic_surface:
                    t_sub = max(float(getattr(surf, 'substrate_thickness', 0.0) or getattr(surf, 'thickness', 0.0)), 1.5)
                    front_mesh = displace_mesh(_surface_mesh(surf, mesh_radial, mesh_angular, cache=self._mesh_cache), d_surf)
                    front_ring = displace_ring(_edge_ring(surf, mesh_angular, cache=self._mesh_cache), d_surf)
                    center = np.asarray(getattr(surf, 'global_origin', [0.0, 0.0, 0.0]), dtype=float) + d_surf
                    rotation = np.asarray(getattr(surf, 'global_R', np.eye(3)), dtype=float)

                    # Back face and cylindrical barrel
                    back_center = center + rotation[:, 2] * t_sub
                    radius = max(float(getattr(surf, 'semi_diameter', 0.0)), 0.1)
                    theta = np.linspace(0.0, 2.0 * np.pi, front_ring[0].size)
                    back_local = np.vstack((
                        radius * np.cos(theta), radius * np.sin(theta), np.full_like(theta, float(surf.sag(radius)) + t_sub)
                    ))
                    back_world = rotation @ back_local + center.reshape(3, 1)
                    back_ring = (back_world[0], back_world[1], back_world[2])
                    side_mesh = tuple(np.vstack((front_ring[i], back_ring[i])) for i in range(3))

                    radial = np.linspace(0.0, 1.0, mesh_radial).reshape(-1, 1)
                    back_mesh = tuple(
                        back_center[i] + radial * (back_ring[i] - back_center[i])
                        for i in range(3)
                    )

                    gen_color = custom_color or (grp.get("palette", {}).get("fill_hex", "#94a3b8") if grp.get("palette") else '#94a3b8')
                    rim_color = custom_color or (grp.get("palette", {}).get("border_hex", "#64748b") if grp.get("palette") else '#64748b')
                    sub_alpha = 1.0 if self.no_transparency else 0.50
                    edge_color = 'none' if self.no_transparency else rim_color
                    edge_width = 0.0 if self.no_transparency else 0.25

                    for mesh in (front_mesh, back_mesh, side_mesh):
                        self.ax.plot_surface(
                            mesh[2], mesh[0], mesh[1],
                            color=gen_color, alpha=sub_alpha, shade=True,
                            edgecolor=edge_color, linewidth=edge_width,
                            antialiased=True,
                        )
                        add_mesh_bounds(mesh)
                    self.ax.plot(front_ring[2], front_ring[0], front_ring[1], color=rim_color, linewidth=1.1, zorder=9)
                    self.ax.plot(back_ring[2], back_ring[0], back_ring[1], color=rim_color, linewidth=1.1, zorder=9)
                    continue
                if not (is_mirror or is_grating or is_dmd or is_curved):
                    continue

                if custom_color:
                    color = custom_color
                elif is_galvo(surf):
                    color = '#80cbc4'
                elif is_mirror:
                    color = '#e2e8f0'
                elif is_grating:
                    color = '#fbbf24'
                elif is_detector:
                    color = '#dc2626'
                elif is_dmd:
                    color = '#56c8d8'
                elif grp.get("palette"):
                    color = grp["palette"].get("fill_hex", "#4FC3F7")
                else:
                    color = '#8a959f'

                front_mesh = displace_mesh(_surface_mesh(surf, mesh_radial, mesh_angular, cache=self._mesh_cache), d_surf)
                front_ring = displace_ring(_edge_ring(surf, mesh_angular, cache=self._mesh_cache), d_surf)
                center, rotation = get_effective_pose(surf)
                center = center + d_surf
                thickness = max(float(getattr(surf, 'substrate_thickness', 0.0)), 0.8)

                # Determine incident beam direction to position substrate behind active reflective face
                prev_origin = None
                for p in reversed(surfaces[:surf_idx]):
                    po = getattr(p, 'global_origin', None)
                    if po is not None and np.linalg.norm(center - po) > 1e-4:
                        prev_origin = np.asarray(po, dtype=float)
                        break
                if prev_origin is not None:
                    d_in = center - prev_origin
                    d_in_norm = np.linalg.norm(d_in)
                    d_in = d_in / d_in_norm if d_in_norm > 1e-6 else rotation[:, 2]
                else:
                    d_in = rotation[:, 2]

                sub_sign = 1.0 if np.dot(d_in, rotation[:, 2]) >= 0.0 else -1.0
                sub_thickness = sub_sign * thickness
                back_center = center + rotation[:, 2] * sub_thickness
                is_rectangular = (
                    ap_type == 'rectangular'
                    or str(getattr(surf, 'substrate_shape', '')).lower() == 'rectangular'
                    or is_dmd
                )
                if is_rectangular:
                    half_x = float(getattr(surf, 'aperture_x_half', 0.0))
                    if half_x <= 0:
                        half_x = float(getattr(surf, 'semi_diameter', 0.0))
                    if half_x <= 0 and is_dmd:
                        half_x = 4.93
                    half_x = max(half_x, 0.1)

                    half_y = float(getattr(surf, 'aperture_y_half', 0.0))
                    if half_y <= 0:
                        half_y = float(getattr(surf, 'semi_diameter', 0.0))
                    if half_y <= 0 and is_dmd:
                        half_y = 3.08
                    half_y = max(half_y, 0.1)

                    # The DMD clear aperture is the active array, not the
                    # outer ceramic package.  A cataloged package can have a
                    # separate plane and size while ray interactions remain
                    # on the ON-state micromirror plane.
                    body_half_x, body_half_y = half_x, half_y
                    body_rotation = rotation
                    dmd_meta = getattr(surf, 'catalog_metadata', {}) if is_dmd else {}
                    dmd_meta = dmd_meta if isinstance(dmd_meta, dict) else {}
                    if is_dmd and isinstance(dmd_meta, dict):
                        package_size = dmd_meta.get('package_body_size_mm')
                        if isinstance(package_size, (list, tuple)) and len(package_size) == 2:
                            body_half_x = max(float(package_size[0]) / 2.0, half_x)
                            body_half_y = max(float(package_size[1]) / 2.0, half_y)
                        package_offset = dmd_meta.get('package_plane_offset_from_on_deg')
                        if isinstance(package_offset, (list, tuple)) and len(package_offset) == 3:
                            from optics_app.engine.mechanisms import _rotation_matrix_xyz
                            body_rotation = rotation @ _rotation_matrix_xyz(package_offset)

                    local_vertices = np.array([
                        [-body_half_x, -body_half_y, 0.0], [body_half_x, -body_half_y, 0.0],
                        [body_half_x, body_half_y, 0.0], [-body_half_x, body_half_y, 0.0],
                        [-body_half_x, -body_half_y, sub_thickness], [body_half_x, -body_half_y, sub_thickness],
                        [body_half_x, body_half_y, sub_thickness], [-body_half_x, body_half_y, sub_thickness],
                    ], dtype=float)
                    world_vertices = (body_rotation @ local_vertices.T).T + center
                    face_indices = (
                        (0, 1, 2, 3), (4, 7, 6, 5),
                        (0, 4, 5, 1), (1, 5, 6, 2),
                        (2, 6, 7, 3), (3, 7, 4, 0),
                    )
                    plot_faces = [
                        [(world_vertices[i, 2], world_vertices[i, 0], world_vertices[i, 1]) for i in face]
                        for face in face_indices
                    ]
                    if is_grating:
                        # Dedicated Grating Solid Rendering:
                        # 1. Top Reflective Ruled Face (z = 0): Metallic Gold with prominent border
                        top_face = [[(world_vertices[i, 2], world_vertices[i, 0], world_vertices[i, 1]) for i in (0, 1, 2, 3)]]
                        top_alpha = 1.0 if self.no_transparency else 0.95
                        top_body = Poly3DCollection(
                            top_face, facecolors='#fbbf24',
                            edgecolors='#b45309',
                            linewidths=1.2,
                            alpha=top_alpha,
                            zorder=12,
                        )
                        if hasattr(top_body, 'set_picker'): top_body.set_picker(5); top_body._surface_index = surf_idx

                        self.ax.add_collection3d(top_body)

                        # 2. Substrate Body (Sides and Rear Face at z = sub_thickness): Dark optical blank (Pyrex/Zerodur)
                        sub_faces = [
                            [(world_vertices[i, 2], world_vertices[i, 0], world_vertices[i, 1]) for i in face]
                            for face in face_indices[1:]
                        ]
                        sub_alpha = 1.0 if self.no_transparency else 0.75
                        sub_body = Poly3DCollection(
                            sub_faces, facecolors='#334155',
                            edgecolors='#1e293b',
                            linewidths=0.55 if self.no_transparency else 0.45,
                            alpha=sub_alpha,
                            zorder=8,
                        )
                        if hasattr(sub_body, 'set_picker'): sub_body.set_picker(5); sub_body._surface_index = surf_idx

                        self.ax.add_collection3d(sub_body)

                        # 3. Parallel Ruled Grating Grooves across the active top face (offset in air toward source)
                        disp_in_y = abs(getattr(surf, 'tilt_x', 0.0)) >= abs(getattr(surf, 'tilt_y', 0.0))
                        n_rulings = 13
                        z_groove = -sub_sign * 0.002
                        if disp_in_y:
                            # Tilt around X -> Dispersion along Y -> Ruling grooves run parallel along X
                            for y_val in np.linspace(-0.85 * half_y, 0.85 * half_y, n_rulings):
                                p1_loc = np.array([-0.90 * half_x, y_val, z_groove], dtype=float)
                                p2_loc = np.array([ 0.90 * half_x, y_val, z_groove], dtype=float)
                                w1 = rotation @ p1_loc + center
                                w2 = rotation @ p2_loc + center
                                self.ax.plot([w1[2], w2[2]], [w1[0], w2[0]], [w1[1], w2[1]],
                                             color='#78350f', linewidth=1.1, alpha=0.92, zorder=14)
                        else:
                            # Tilt around Y -> Dispersion along X -> Ruling grooves run parallel along Y
                            for x_val in np.linspace(-0.85 * half_x, 0.85 * half_x, n_rulings):
                                p1_loc = np.array([x_val, -0.90 * half_y, z_groove], dtype=float)
                                p2_loc = np.array([x_val,  0.90 * half_y, z_groove], dtype=float)
                                w1 = rotation @ p1_loc + center
                                w2 = rotation @ p2_loc + center
                                self.ax.plot([w1[2], w2[2]], [w1[0], w2[0]], [w1[1], w2[1]],
                                             color='#78350f', linewidth=1.1, alpha=0.92, zorder=14)

                        # 4. Outward normal vector from top reflective face pointing into air
                        norm_len = min(half_x, half_y) * 0.75
                        n_end = center - (sub_sign * rotation[:, 2]) * norm_len
                        self.ax.plot([center[2], n_end[2]], [center[0], n_end[0]], [center[1], n_end[1]],
                                     color='#b45309', linewidth=1.8, zorder=15)

                    elif is_dmd:
                        # Generic DMD package and active micromirror face. Catalog
                        # metadata supplies package dimensions when available.
                        # 1. Package Substrate (Sides and Rear at z in substrate direction): Dark ceramic chip package
                        sub_faces = [
                            [(world_vertices[i, 2], world_vertices[i, 0], world_vertices[i, 1]) for i in face]
                            for face in face_indices[1:]
                        ]
                        sub_alpha = 1.0 if self.no_transparency else 0.85
                        sub_body = Poly3DCollection(
                            sub_faces, facecolors='#1e293b',
                            edgecolors='#0f172a',
                            linewidths=0.6 if self.no_transparency else 0.5,
                            alpha=sub_alpha,
                            zorder=8,
                        )
                        if hasattr(sub_body, 'set_picker'): sub_body.set_picker(5); sub_body._surface_index = surf_idx

                        self.ax.add_collection3d(sub_body)

                        # 2. Package Bezel / Aperture Window Frame (at z = 0 local on the package face):
                        bezel_face = [[(world_vertices[i, 2], world_vertices[i, 0], world_vertices[i, 1]) for i in (0, 1, 2, 3)]]
                        bezel_body = Poly3DCollection(
                            bezel_face, facecolors='#0f172a',
                            edgecolors='#d97706',
                            linewidths=1.2,
                            alpha=1.0 if self.no_transparency else 0.90,
                            zorder=9,
                        )
                        if hasattr(bezel_body, 'set_picker'): bezel_body.set_picker(5); bezel_body._surface_index = surf_idx

                        self.ax.add_collection3d(bezel_body)

                        # 3. Active Micromirror Array (Top Reflective Face):
                        from optics_app.engine.mechanisms import interaction_rotation
                        curr_state = active_state_name(surf)
                        R_active = interaction_rotation(surf)

                        act_scale_x = 1.0 if dmd_meta.get('active_array_size_mm') else 0.85
                        act_scale_y = 1.0 if dmd_meta.get('active_array_size_mm') else 0.85
                        act_hx = half_x * act_scale_x
                        act_hy = half_y * act_scale_y

                        # Vertices of the active micromirror facet array (tilted with active state, offset in air)
                        z_micro = -sub_sign * 0.002
                        micro_loc = np.array([
                            [-act_hx, -act_hy, z_micro],
                            [ act_hx, -act_hy, z_micro],
                            [ act_hx,  act_hy, z_micro],
                            [-act_hx,  act_hy, z_micro]
                        ], dtype=float)
                        micro_world = (R_active @ micro_loc.T).T + center

                        micro_face = [[(micro_world[k, 2], micro_world[k, 0], micro_world[k, 1]) for k in (0, 1, 2, 3)]]
                        face_col = '#f8fafc' if curr_state == 'ON' else ('#cbd5e1' if curr_state == 'FLAT' else '#94a3b8')
                        edge_col = '#16a34a' if curr_state == 'ON' else ('#2563eb' if curr_state == 'FLAT' else '#dc2626')
                        micro_body = Poly3DCollection(
                            micro_face, facecolors=face_col,
                            edgecolors=edge_col,
                            linewidths=1.4,
                            alpha=1.0 if self.no_transparency else 0.95,
                            zorder=13,
                        )
                        if hasattr(micro_body, 'set_picker'): micro_body.set_picker(5); micro_body._surface_index = surf_idx

                        self.ax.add_collection3d(micro_body)

                        # 4. Diamond / Pixel Array Grid Texture on the active face:
                        n_grid = 7
                        z_grid = -sub_sign * 0.003
                        for gx in np.linspace(-0.85 * act_hx, 0.85 * act_hx, n_grid):
                            p1 = R_active @ np.array([gx, -act_hy * 0.9, z_grid]) + center
                            p2 = R_active @ np.array([gx,  act_hy * 0.9, z_grid]) + center
                            self.ax.plot([p1[2], p2[2]], [p1[0], p2[0]], [p1[1], p2[1]],
                                         color='#94a3b8', linewidth=0.6, alpha=0.7, zorder=14)
                        for gy in np.linspace(-0.85 * act_hy, 0.85 * act_hy, n_grid):
                            p1 = R_active @ np.array([-act_hx * 0.9, gy, z_grid]) + center
                            p2 = R_active @ np.array([ act_hx * 0.9, gy, z_grid]) + center
                            self.ax.plot([p1[2], p2[2]], [p1[0], p2[0]], [p1[1], p2[1]],
                                         color='#94a3b8', linewidth=0.6, alpha=0.7, zorder=14)

                        # 5. Facet Normal Arrow (showing physical micromirror tilt pointing into air):
                        normal_len = min(half_x, half_y) * 0.8
                        facet_normal = -sub_sign * R_active[:, 2]
                        n_end = center + facet_normal * normal_len
                        norm_color = '#16a34a' if curr_state == 'ON' else ('#2563eb' if curr_state == 'FLAT' else '#dc2626')
                        self.ax.plot([center[2], n_end[2]], [center[0], n_end[0]], [center[1], n_end[1]],
                                     color=norm_color, linewidth=2.0, zorder=15)

                        # 6. 3D Text Badge indicating State and Angle:
                        part_name = str(dmd_meta.get('part_number') or surf.name or 'DMD')
                        state_label = f"{part_name} [{curr_state}]"
                        state_spec = active_state(surf)
                        tilt = state_spec.get('micromirror_tilt_deg', state_spec.get('tilt_angle_deg'))
                        if tilt is not None:
                            state_label += f" ({float(tilt):+.1f}°)"
                        badge_pos = center + body_rotation[:, 1] * (body_half_y * 1.15)
                        self.ax.text(badge_pos[2], badge_pos[0], badge_pos[1],
                                     state_label, color=norm_color, fontsize=8, fontweight='bold', zorder=16)

                    elif is_mirror:
                        # Dedicated Rectangular Mirror / Galvo Solid Rendering
                        is_g = is_galvo(surf)
                        top_face = [[(world_vertices[i, 2], world_vertices[i, 0], world_vertices[i, 1]) for i in (0, 1, 2, 3)]]
                        top_alpha = 1.0 if self.no_transparency else 0.92
                        top_color = '#80cbc4' if is_g else '#e2e8f0'
                        top_edge = '#004d40' if is_g else ('none' if self.no_transparency else '#64748b')
                        top_body = Poly3DCollection(
                            top_face, facecolors=top_color,
                            edgecolors=top_edge,
                            linewidths=1.2 if is_g else (0.0 if self.no_transparency else 1.0),
                            alpha=top_alpha,
                            zorder=11,
                        )
                        if hasattr(top_body, 'set_picker'): top_body.set_picker(5); top_body._surface_index = surf_idx

                        self.ax.add_collection3d(top_body)

                        # 2. Substrate Body (Sides and Rear at z = sub_thickness): Dark Optical Blank
                        sub_faces = [
                            [(world_vertices[i, 2], world_vertices[i, 0], world_vertices[i, 1]) for i in face]
                            for face in face_indices[1:]
                        ]
                        sub_alpha = 1.0 if self.no_transparency else 0.72
                        sub_body = Poly3DCollection(
                            sub_faces, facecolors='#334155',
                            edgecolors='#1e293b',
                            linewidths=0.55 if self.no_transparency else 0.45,
                            alpha=sub_alpha,
                            zorder=8,
                        )
                        if hasattr(sub_body, 'set_picker'): sub_body.set_picker(5); sub_body._surface_index = surf_idx

                        self.ax.add_collection3d(sub_body)

                        # 3. Outward normal vector from top reflective face pointing into air
                        norm_len = min(half_x, half_y) * 0.75
                        n_end = center - (sub_sign * rotation[:, 2]) * norm_len
                        norm_color = '#004d40' if is_g else '#64748b'
                        self.ax.plot([center[2], n_end[2]], [center[0], n_end[0]], [center[1], n_end[1]],
                                     color=norm_color, linewidth=1.8, zorder=15)
                        if is_g:
                            ang_val = float(getattr(getattr(surf, "galvo_config", None), "current_angle_deg", 0.0))
                            galvo_label = f"GALVO [{ang_val:+.1f}°]"
                            badge_pos = center + rotation[:, 1] * (body_half_y * 1.15)
                            self.ax.text(badge_pos[2], badge_pos[0], badge_pos[1],
                                         galvo_label, color='#004d40', fontsize=8, fontweight='bold', zorder=16)

                    else:
                        rect_alpha = 1.0 if self.no_transparency else 0.78
                        body = Poly3DCollection(
                            plot_faces, facecolors=color,
                            edgecolors='#263238' if self.no_transparency else '#4c5359',
                            linewidths=0.55 if self.no_transparency else 0.45,
                            alpha=rect_alpha,
                        )
                        if hasattr(body, 'set_picker'): body.set_picker(5); body._surface_index = surf_idx

                        self.ax.add_collection3d(body)

                    if self.show_tolerance_envelope and self.tolerance_thickness > 0:
                        t_cyl, t_min_r, t_max_r = build_tolerance_envelope_geometry(
                            surf=surf,
                            radius=max(half_x, half_y) + 0.35,
                            thickness=thickness,
                            tolerance=self.tolerance_thickness,
                            z_offset=0.0,
                            angular_samples=mesh_angular,
                        )
                        t_cyl = displace_mesh(t_cyl, d_surf)
                        t_min_r = displace_ring(t_min_r, d_surf)
                        t_max_r = displace_ring(t_max_r, d_surf)
                        tcx, tcy, tcz = t_cyl
                        self.ax.plot_surface(
                            tcz, tcx, tcy,
                            color="#f59e0b", alpha=0.18, shade=True,
                            edgecolor="#d97706", linewidth=0.3, antialiased=True,
                        )
                        add_mesh_bounds(t_cyl)
                        self.ax.plot(t_min_r[2], t_min_r[0], t_min_r[1], color="#d97706", linestyle="--", linewidth=1.2, zorder=8)
                        self.ax.plot(t_max_r[2], t_max_r[0], t_max_r[1], color="#d97706", linestyle="--", linewidth=1.2, zorder=8)

                    all_x.extend(world_vertices[:, 0])
                    all_y.extend(world_vertices[:, 1])
                    all_z.extend(world_vertices[:, 2])
                    continue

                theta = np.linspace(0.0, 2.0 * np.pi, front_ring[0].size)
                radius = max(float(getattr(surf, 'semi_diameter', 0.0)), 0.1)
                back_local = np.vstack((
                    radius * np.cos(theta), radius * np.sin(theta), np.zeros_like(theta)
                ))
                back_world = rotation @ back_local + back_center.reshape(3, 1)
                back_ring = (back_world[0], back_world[1], back_world[2])
                side_mesh = tuple(np.vstack((front_ring[i], back_ring[i])) for i in range(3))

                radial = np.linspace(0.0, 1.0, mesh_radial).reshape(-1, 1)
                back_mesh = tuple(
                    back_center[i] + radial * (back_ring[i] - back_center[i])
                    for i in range(3)
                )

                if is_grating:
                    # Circular Grating Solid Rendering:
                    # 1. Front Active Reflective Ruled Face (z = 0) in Metallic Gold
                    front_alpha = 1.0 if self.no_transparency else 0.95
                    self.ax.plot_surface(
                        front_mesh[2], front_mesh[0], front_mesh[1],
                        color='#fbbf24', alpha=front_alpha, shade=True,
                        edgecolor='none' if self.no_transparency else '#b45309',
                        linewidth=0.0 if self.no_transparency else 0.5,
                        antialiased=True,
                    )
                    self.ax.plot(front_ring[2], front_ring[0], front_ring[1], color='#b45309', linewidth=1.2, zorder=10)
                    # 2. Cylindrical Substrate Body and Rear Face in Dark Substrate Blank
                    sub_alpha = 1.0 if self.no_transparency else 0.72
                    for mesh in (back_mesh, side_mesh):
                        self.ax.plot_surface(
                            mesh[2], mesh[0], mesh[1],
                            color='#334155', alpha=sub_alpha, shade=True,
                            edgecolor='none' if self.no_transparency else '#1e293b',
                            linewidth=0.0 if self.no_transparency else 0.35,
                            antialiased=True,
                        )
                    self.ax.plot(back_ring[2], back_ring[0], back_ring[1], color='#1e293b', linewidth=1.0, zorder=9)
                    # 3. Ruling Grooves across the circular aperture offset into air
                    z_groove = -sub_sign * 0.002
                    disp_in_y = abs(getattr(surf, 'tilt_x', 0.0)) >= abs(getattr(surf, 'tilt_y', 0.0))
                    n_rulings = 13
                    if disp_in_y:
                        for y_val in np.linspace(-0.82 * radius, 0.82 * radius, n_rulings):
                            x_span = np.sqrt(max(0.0, radius**2 - y_val**2)) * 0.90
                            p1_loc = np.array([-x_span, y_val, z_groove], dtype=float)
                            p2_loc = np.array([ x_span, y_val, z_groove], dtype=float)
                            w1 = rotation @ p1_loc + center
                            w2 = rotation @ p2_loc + center
                            self.ax.plot([w1[2], w2[2]], [w1[0], w2[0]], [w1[1], w2[1]],
                                         color='#78350f', linewidth=1.1, alpha=0.92, zorder=14)
                    else:
                        for x_val in np.linspace(-0.82 * radius, 0.82 * radius, n_rulings):
                            y_span = np.sqrt(max(0.0, radius**2 - x_val**2)) * 0.90
                            p1_loc = np.array([x_val, -y_span, z_groove], dtype=float)
                            p2_loc = np.array([x_val,  y_span, z_groove], dtype=float)
                            w1 = rotation @ p1_loc + center
                            w2 = rotation @ p2_loc + center
                            self.ax.plot([w1[2], w2[2]], [w1[0], w2[0]], [w1[1], w2[1]],
                                         color='#78350f', linewidth=1.1, alpha=0.92, zorder=14)

                    # 4. Outward normal vector from top reflective face pointing into air
                    norm_len = radius * 0.75
                    n_end = center - (sub_sign * rotation[:, 2]) * norm_len
                    self.ax.plot([center[2], n_end[2]], [center[0], n_end[0]], [center[1], n_end[1]],
                                 color='#b45309', linewidth=1.8, zorder=15)

                    add_mesh_bounds(front_mesh)
                    add_mesh_bounds(back_mesh)
                    add_mesh_bounds(side_mesh)
                    continue

                if is_mirror:
                    # Circular Mirror / Galvo Solid Rendering:
                    # 1. Front Active Reflective Coating (z = 0) in Bright Silver, or Scanning Teal for galvos
                    is_g = is_galvo(surf)
                    front_alpha = 1.0 if self.no_transparency else 0.92
                    m_front_color = '#80cbc4' if is_g else '#e2e8f0'
                    m_edge_color = '#004d40' if is_g else ('none' if self.no_transparency else '#94a3b8')
                    m_rim_color = '#004d40' if is_g else '#94a3b8'
                    self.ax.plot_surface(
                        front_mesh[2], front_mesh[0], front_mesh[1],
                        color=m_front_color, alpha=front_alpha, shade=True,
                        edgecolor=m_edge_color,
                        linewidth=0.0 if self.no_transparency and not is_g else 0.5,
                        antialiased=True,
                    )
                    self.ax.plot(front_ring[2], front_ring[0], front_ring[1], color=m_rim_color, linewidth=1.2, zorder=10)
                    # 2. Cylindrical Substrate Body and Rear Face in Dark Substrate Blank
                    sub_alpha = 1.0 if self.no_transparency else 0.72
                    for mesh in (back_mesh, side_mesh):
                        self.ax.plot_surface(
                            mesh[2], mesh[0], mesh[1],
                            color='#334155', alpha=sub_alpha, shade=True,
                            edgecolor='none' if self.no_transparency else '#1e293b',
                            linewidth=0.0 if self.no_transparency else 0.35,
                            antialiased=True,
                        )
                    self.ax.plot(back_ring[2], back_ring[0], back_ring[1], color='#1e293b', linewidth=1.0, zorder=9)
                    # 3. Outward normal vector from top reflective face pointing into air
                    norm_len = radius * 0.75
                    n_end = center - (sub_sign * rotation[:, 2]) * norm_len
                    norm_color = '#004d40' if is_g else '#64748b'
                    self.ax.plot([center[2], n_end[2]], [center[0], n_end[0]], [center[1], n_end[1]],
                                 color=norm_color, linewidth=1.8, zorder=15)
                    if is_g:
                        ang_val = float(getattr(getattr(surf, "galvo_config", None), "current_angle_deg", 0.0))
                        galvo_label = f"GALVO [{ang_val:+.1f}°]"
                        badge_pos = center + rotation[:, 1] * (radius * 1.15)
                        self.ax.text(badge_pos[2], badge_pos[0], badge_pos[1],
                                     galvo_label, color='#004d40', fontsize=8, fontweight='bold', zorder=16)
                    add_mesh_bounds(front_mesh)
                    add_mesh_bounds(back_mesh)
                    add_mesh_bounds(side_mesh)
                    continue

                circ_alpha = 1.0 if self.no_transparency else 0.72
                edge_color = 'none' if self.no_transparency else '#4c5359'
                edge_width = 0.0 if self.no_transparency else 0.3
                for mesh in (front_mesh, back_mesh, side_mesh):
                    x_mesh, y_mesh, z_mesh = mesh
                    self.ax.plot_surface(
                        z_mesh, x_mesh, y_mesh,
                        color=color, alpha=circ_alpha, shade=True,
                        edgecolor=edge_color, linewidth=edge_width,
                        antialiased=True,
                    )
                    add_mesh_bounds(mesh)
                self.ax.plot(front_ring[2], front_ring[0], front_ring[1], color='#263238', linewidth=1.0, zorder=9)

                if self.show_tolerance_envelope and self.tolerance_thickness > 0:
                    t_cyl, t_min_r, t_max_r = build_tolerance_envelope_geometry(
                        surf=surf,
                        radius=radius + 0.35,
                        thickness=thickness,
                        tolerance=self.tolerance_thickness,
                        z_offset=0.0,
                        angular_samples=mesh_angular,
                    )
                    t_cyl = displace_mesh(t_cyl, d_surf)
                    t_min_r = displace_ring(t_min_r, d_surf)
                    t_max_r = displace_ring(t_max_r, d_surf)
                    tcx, tcy, tcz = t_cyl
                    self.ax.plot_surface(
                        tcz, tcx, tcy,
                        color="#f59e0b", alpha=0.18, shade=True,
                        edgecolor="#d97706", linewidth=0.3, antialiased=True,
                    )
                    add_mesh_bounds(t_cyl)
                    self.ax.plot(t_min_r[2], t_min_r[0], t_min_r[1], color="#d97706", linestyle="--", linewidth=1.2, zorder=8)
                    self.ax.plot(t_max_r[2], t_max_r[0], t_max_r[1], color="#d97706", linestyle="--", linewidth=1.2, zorder=8)

            # 3. Render 3D Element Labels and Field Badges
            if self.show_labels:
                self._render_element_labels(surfaces, all_x, all_y, all_z)
                self._render_field_labels(surfaces, all_x, all_y, all_z)

            self._base_all_x = list(all_x)
            self._base_all_y = list(all_y)
            self._base_all_z = list(all_z)
            self._last_geom_fingerprint = curr_fingerprint

        # 4. Render 3D Rays
        if self.show_rays and self.rays:
            self._render_rays(self.rays, all_x, all_y, all_z)
            self._render_grating_diffraction(surfaces, self.rays, all_x, all_y, all_z)

        route_count = self._render_optical_routes(stack, surfaces, all_x, all_y, all_z) if stack is not None else 0
        self._last_route_count = route_count

        # 5. Aspect Ratio & Bound Auto-scaling (Tight Bounds with Proportional Padding & Prominent Sizing)
        if len(all_x) > 1:
            min_z, max_z = float(min(all_z)), float(max(all_z))
            min_x, max_x = float(min(all_x)), float(max(all_x))
            min_y, max_y = float(min(all_y)), float(max(all_y))

            span_z = max(max_z - min_z, 1.0)
            span_x = max(max_x - min_x, 1.0)
            span_y = max(max_y - min_y, 1.0)

            pad_z = max(span_z * 0.18, 5.0)
            pad_x = max(span_x * 0.15, 3.5)
            pad_y = max(span_y * 0.15, 3.5)

            self._base_xlim = (min_z - pad_z, max_z + pad_z)
            self._base_ylim = (min_x - pad_x, max_x + pad_x)
            self._base_zlim = (min_y - pad_y, max_y + pad_y)

            self.ax.set_xlim(self._base_xlim)
            self.ax.set_ylim(self._base_ylim)
            self.ax.set_zlim(self._base_zlim)

            self._apply_box_aspect()

            self._cam_dist = 7.5
            if hasattr(self.ax, '_dist'):
                self.ax._dist = 7.5

        # 6. Status Readout with camera angles
        self._update_status_text()

        if hasattr(self.canvas, "draw_idle"):
            self.canvas.draw_idle()
        elif hasattr(self.canvas, "draw"):
            self.canvas.draw()
