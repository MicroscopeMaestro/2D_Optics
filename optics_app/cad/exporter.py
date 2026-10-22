import numpy as np
import math
import os
from typing import Optional, Any, List, Dict, Collection, Sequence, Tuple
from dataclasses import dataclass, field
from enum import Enum

try:
    from optics_app.engine.lde import SurfaceStack
    from optics_app.engine.surface import SurfaceType
except ImportError:
    from v2.engine.lde import SurfaceStack
    from v2.engine.surface import SurfaceType

try:
    try:
        from optics_app.engine.laser_scanner import is_galvo
    except ImportError:
        from v2.engine.laser_scanner import is_galvo
except Exception:
    def is_galvo(s):
        return bool(getattr(s, "galvo_config", None) is not None or str(getattr(s, "mechanism_type", "")).upper() == "GALVO")

try:
    try:
        from optics_app.engine.mechanisms import is_dmd, available_states, get_effective_pose
    except ImportError:
        from v2.engine.mechanisms import is_dmd, available_states, get_effective_pose
except Exception:
    def is_dmd(s):
        return "DMD" in str(getattr(s, "name", "")).upper() or str(getattr(s, "mechanism_type", "")).upper() == "DMD"
    def available_states(s):
        return getattr(s, "mechanical_states", {}) or {}
    def get_effective_pose(s):
        return np.asarray(getattr(s, "global_origin", [0.0, 0.0, 0.0]), dtype=float), np.asarray(getattr(s, "global_R", np.eye(3)), dtype=float)

class STEPExportEngine(Enum):
    AUTO = "auto"
    BUILD123D = "build123d"
    PURE_PYTHON = "pure_python"

@dataclass
class STEPExportOptions:
    engine: STEPExportEngine = STEPExportEngine.AUTO
    selected_element_indices: Optional[Collection[int]] = None
    selected_surface_indices: Optional[Collection[int]] = None
    export_3d_printable_enclosure: bool = False
    enclosure_wall_margin_mm: float = 10.0
    export_mated_air_cavity: bool = False
    export_optics_with_enclosure: bool = False
    air_gap_cylinder_diameter: float = 25.4
    mount_diameter: float = 30.5
    mount_z_thickness: float = 0.0
    export_mechanical_mounts: bool = False
    tolerance_thickness_mm: float = 1.0
    element_displacements: Dict[int, float] = field(default_factory=dict)

@dataclass
class STEPExportResult:
    file_path: str
    is_success: bool
    num_components_exported: int
    engine_used: STEPExportEngine
    file_size_bytes: int = 0
    error_message: str = ""
    warnings: list = field(default_factory=list)

    def __eq__(self, other):
        if isinstance(other, str):
            return self.file_path == other or os.path.abspath(self.file_path) == os.path.abspath(other)
        if isinstance(other, STEPExportResult):
            return (self.file_path == other.file_path and
                    self.is_success == other.is_success and
                    self.num_components_exported == other.num_components_exported and
                    self.engine_used == other.engine_used)
        return False

    def __str__(self):
        return self.file_path

    def __fspath__(self):
        return self.file_path

HAS_BUILD123D = True

def _get_build123d():
    try:
        import build123d as b3d
        return b3d
    except Exception:
        return None

def _extract_surface_stack(system_or_stack: Any) -> Optional[SurfaceStack]:
    """Safely extract SurfaceStack from OpticalSystem, LensDataEditor, LensDataTableModel, list of surfaces, or SurfaceStack."""
    if system_or_stack is None:
        return None
    if isinstance(system_or_stack, SurfaceStack):
        return system_or_stack
    if isinstance(system_or_stack, list):
        st = SurfaceStack()
        for s in system_or_stack:
            st.add_surface(s)
        return st
    if hasattr(system_or_stack, 'surface_stack') and isinstance(system_or_stack.surface_stack, SurfaceStack):
        return system_or_stack.surface_stack
    if hasattr(system_or_stack, 'get_surface_stack') and callable(system_or_stack.get_surface_stack):
        try:
            return system_or_stack.get_surface_stack()
        except Exception:
            pass
    if hasattr(system_or_stack, 'stack'):
        st = system_or_stack.stack
        if isinstance(st, SurfaceStack):
            return st
        if hasattr(st, 'get_surface_stack') and callable(st.get_surface_stack):
            try:
                return st.get_surface_stack()
            except Exception:
                pass
        if hasattr(st, 'surfaces'):
            return st
    if hasattr(system_or_stack, 'surfaces'):
        if isinstance(system_or_stack.surfaces, list):
            st = SurfaceStack()
            for s in system_or_stack.surfaces:
                st.add_surface(s)
            return st
        return system_or_stack
    return None

def R_mat_to_euler_degrees(R_mat: np.ndarray):
    sy = math.sqrt(R_mat[0,0] * R_mat[0,0] + R_mat[1,0] * R_mat[1,0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R_mat[2,1], R_mat[2,2])
        y = math.atan2(-R_mat[2,0], sy)
        z = math.atan2(R_mat[1,0], R_mat[0,0])
    else:
        x = math.atan2(-R_mat[1,2], R_mat[1,1])
        y = math.atan2(-R_mat[2,0], sy)
        z = 0
    return (math.degrees(x), math.degrees(y), math.degrees(z))

def create_solid_lens(surf1, surf2):
    b3d = _get_build123d()
    if b3d is None:
        return None
    raw_pts = []
    
    sd1 = surf1.semi_diameter
    sd2 = surf2.semi_diameter
    sd = max(sd1, sd2)
    
    r_vals_front = np.linspace(0, sd, 30)
    for r in r_vals_front:
        z = surf1.sag(r) if r <= sd1 else surf1.sag(sd1)
        raw_pts.append((float(r), float(z)))
        
    z_back_edge = surf1.thickness + (surf2.sag(sd2) if sd > sd2 else surf2.sag(sd))
    raw_pts.append((float(sd), float(z_back_edge)))
    
    r_vals_back = np.linspace(sd, 0, 30)
    for r in r_vals_back:
        z = surf1.thickness + (surf2.sag(r) if r <= sd2 else surf2.sag(sd2))
        raw_pts.append((float(r), float(z)))
        
    # Deduplicate adjacent points to guarantee valid manifold polygon for revolve
    pts = [raw_pts[0]]
    for p in raw_pts[1:]:
        if np.hypot(p[0] - pts[-1][0], p[1] - pts[-1][1]) > 1e-5:
            pts.append(p)

    with b3d.BuildPart() as part:
        with b3d.BuildSketch(b3d.Plane.XZ):
            b3d.Polygon(*pts)
        b3d.revolve(axis=b3d.Axis.Z)
        
    return part.part

def create_mirror_or_planar_solid(surf, thickness=None):
    """
    Generate accurate 3D solid body for mirrors, gratings, DMDs, and planar optical components.
    Extrudes the physical substrate volume backwards (-Z) so the front optical/reflective
    face remains at Z = 0 in the surface coordinate system.
    """
    b3d = _get_build123d()
    if b3d is None:
        return None
        
    t = float(surf.substrate_thickness if getattr(surf, 'substrate_thickness', 0) > 0 else (thickness if thickness is not None else 6.0))
    if surf.surface_type == SurfaceType.OBJECT:
        t = 1.8
    elif surf.surface_type == SurfaceType.DETECTOR:
        t = float(surf.substrate_thickness if getattr(surf, 'substrate_thickness', 0) > 0 else 3.0)
    elif surf.surface_type == SurfaceType.STOP:
        t = float(surf.substrate_thickness if getattr(surf, 'substrate_thickness', 0) > 0 else (surf.thickness if surf.thickness > 0 else 1.5))

    is_rect = getattr(surf, 'aperture_type', '') in ["Rectangular", "Slit"] or getattr(surf, 'substrate_shape', '') == "Rectangular" or "DMD" in (surf.name or "").upper()

    try:
        if surf.radius != 0.0 and not np.isinf(surf.radius) and not is_rect:
            # Revolve curved mirror substrate with real optical sag face
            sd = float(surf.semi_diameter)
            raw_pts = []
            r_vals = np.linspace(0, sd, 25)
            for r in r_vals:
                raw_pts.append((float(r), float(surf.sag(r))))
            raw_pts.append((float(sd), float(-t)))
            raw_pts.append((0.0, float(-t)))
            pts = [raw_pts[0]]
            for p in raw_pts[1:]:
                if np.hypot(p[0] - pts[-1][0], p[1] - pts[-1][1]) > 1e-5:
                    pts.append(p)
            with b3d.BuildPart() as part:
                with b3d.BuildSketch(b3d.Plane.XZ):
                    b3d.Polygon(*pts)
                b3d.revolve(axis=b3d.Axis.Z)
            return part.part

        with b3d.BuildPart() as part:
            if is_rect:
                hx = float(surf.aperture_x_half if getattr(surf, 'aperture_x_half', 0) > 0 else surf.semi_diameter)
                hy = float(surf.aperture_y_half if getattr(surf, 'aperture_y_half', 0) > 0 else surf.semi_diameter)
                if hx <= 0 and "DMD" in (surf.name or "").upper():
                    hx = 4.93
                if hy <= 0 and "DMD" in (surf.name or "").upper():
                    hy = 3.08
                hx = max(hx, 2.0)
                hy = max(hy, 2.0)
                wx = 2.0 * hx
                wy = 2.0 * hy
                with b3d.BuildSketch(b3d.Plane.XY):
                    b3d.Rectangle(wx, wy)
                b3d.extrude(amount=-t)
            else:
                r = max(float(surf.semi_diameter), 2.0)
                with b3d.BuildSketch(b3d.Plane.XY):
                    b3d.Circle(radius=r)
                b3d.extrude(amount=-t)
        return part.part
    except Exception:
        # Fallback to simple cylinder
        with b3d.BuildPart() as part:
            b3d.Cylinder(radius=max(float(surf.semi_diameter), 2.0), height=t)
        return part.part

def get_surface_clear_radius(surf: Any) -> float:
    """Return physical clear aperture radius, properly accounting for rectangular apertures."""
    ap_type = str(getattr(surf, 'aperture_type', 'Circular')).lower()
    if ap_type in ('rectangular', 'slit') or 'DMD' in (getattr(surf, 'name', '') or '').upper():
        hx = float(getattr(surf, 'aperture_x_half', 0.0) or getattr(surf, 'semi_diameter_x', 0.0) or getattr(surf, 'semi_diameter', 0.0) or 0.0)
        hy = float(getattr(surf, 'aperture_y_half', 0.0) or getattr(surf, 'semi_diameter_y', 0.0) or getattr(surf, 'semi_diameter', 0.0) or 0.0)
        if hx <= 0.0 and 'DMD' in (getattr(surf, 'name', '') or '').upper():
            hx = 4.93
        if hy <= 0.0 and 'DMD' in (getattr(surf, 'name', '') or '').upper():
            hy = 3.08
        if hx > 0.0 and hy > 0.0:
            return float(np.hypot(hx, hy))
        return max(float(getattr(surf, 'semi_diameter', 0.0) or 0.0), hx, hy, 2.0)
    return max(float(getattr(surf, 'semi_diameter', 0.0) or 0.0), 2.0)

def create_air_gap_solid(s1: Any, s2: Any, constant_diameter: float = 25.4) -> Optional[Any]:
    """
    Generate 3D solid air volume bridging optical elements from s1 to s2.
    Starts at Z = 0 (on s1 face) and extends by s1.thickness to s2 face,
    as a clean, constant-diameter cylinder throughout the entire optical path
    with optical sag matching on element faces to eliminate space gaps.
    """
    b3d = _get_build123d()
    if b3d is None:
        return None
    t = max(float(getattr(s1, 'thickness', 0.0)), 0.001)

    diam = float(constant_diameter or 0.0)
    if diam <= 0.0:
        diam = 25.4
    rad = max(diam / 2.0, 0.5)

    has_sag1 = hasattr(s1, 'sag') and callable(s1.sag) and getattr(s1, 'radius', 0.0) != 0.0 and not np.isinf(getattr(s1, 'radius', 0.0))
    has_sag2 = hasattr(s2, 'sag') and callable(s2.sag) and getattr(s2, 'radius', 0.0) != 0.0 and not np.isinf(getattr(s2, 'radius', 0.0))

    if not has_sag1 and not has_sag2:
        try:
            with b3d.BuildPart() as p:
                with b3d.BuildSketch(b3d.Plane.XY):
                    b3d.Circle(radius=rad)
                b3d.extrude(amount=t)
            return p.part
        except Exception:
            return None

    # Revolve constant-diameter cylinder with exact optical sag matching on front and rear faces
    try:
        raw_pts = []
        r_vals_front = np.linspace(0, rad, 30)
        for r in r_vals_front:
            z = float(s1.sag(r)) if has_sag1 else 0.0
            raw_pts.append((float(r), float(z)))

        z_back_edge = t + (float(s2.sag(rad)) if has_sag2 else 0.0)
        raw_pts.append((float(rad), float(z_back_edge)))

        r_vals_back = np.linspace(rad, 0, 30)
        for r in r_vals_back:
            z = t + (float(s2.sag(r)) if has_sag2 else 0.0)
            raw_pts.append((float(r), float(z)))

        pts = [raw_pts[0]]
        for p in raw_pts[1:]:
            if np.hypot(p[0] - pts[-1][0], p[1] - pts[-1][1]) > 1e-5:
                pts.append(p)

        with b3d.BuildPart() as p:
            with b3d.BuildSketch(b3d.Plane.XZ):
                b3d.Polygon(*pts)
            b3d.revolve(axis=b3d.Axis.Z)
        return p.part
    except Exception:
        try:
            with b3d.BuildPart() as p:
                with b3d.BuildSketch(b3d.Plane.XY):
                    b3d.Circle(radius=rad)
                b3d.extrude(amount=t)
            return p.part
        except Exception:
            return None

def create_lens_mount_solid(inner_radius: float, outer_radius: float, thickness: float, z_offset: float = 0.0) -> Optional[Any]:
    """
    Generate 3D solid annular collar / sleeve for optomechanical component mount.
    Inner radius fits the optic; outer radius provides the mounting barrel wall.
    """
    b3d = _get_build123d()
    if b3d is None:
        return None
    r_in = max(float(inner_radius), 0.5)
    r_out = max(float(outer_radius), r_in + 0.5)
    t = max(float(thickness), 0.5)
    try:
        with b3d.BuildPart() as part:
            with b3d.BuildSketch(b3d.Plane.XY):
                b3d.Circle(radius=r_out)
                b3d.Circle(radius=r_in, mode=b3d.Mode.SUBTRACT)
            b3d.extrude(amount=t)
        solid = part.part
        if abs(z_offset) > 1e-5:
            solid = solid.locate(b3d.Location((0, 0, z_offset)))
        return solid
    except Exception:
        return None

def create_tolerance_pocket_solid(radius: float, thickness: float, tolerance: float = 1.0, z_offset: float = 0.0) -> Optional[Any]:
    """
    Generate 3D solid cylinder pocket with +/- tolerance axial clearance for enclosure boolean subtraction.
    Extends from z_offset - tolerance to z_offset + thickness + tolerance (total length: thickness + 2*tolerance).
    """
    b3d = _get_build123d()
    if b3d is None:
        return None
    r = max(float(radius), 0.5)
    tol = max(float(tolerance), 0.0)
    t = max(float(thickness), 0.1)
    total_len = t + 2.0 * tol
    z_start = float(z_offset - tol)
    try:
        with b3d.BuildPart() as part:
            with b3d.BuildSketch(b3d.Plane.XY):
                b3d.Circle(radius=r)
            b3d.extrude(amount=total_len)
        solid = part.part
        if abs(z_start) > 1e-5:
            solid = solid.locate(b3d.Location((0, 0, z_start)))
        return solid
    except Exception:
        return None

def create_aperture_stop_solid(surf, thickness=1.5):
    """
    Generate 3D solid diaphragm / baffle plate with central clear aperture hole for stops and apertures.
    """
    b3d = _get_build123d()
    if b3d is None:
        return None
    t = float(surf.substrate_thickness if getattr(surf, 'substrate_thickness', 0) > 0 else (surf.thickness if surf.thickness > 0 else thickness))
    t = max(t, 1.0)
    is_rect = getattr(surf, 'aperture_type', '') in ["Rectangular", "Slit"] or getattr(surf, 'substrate_shape', '') == "Rectangular"
    try:
        sd = max(float(getattr(surf, 'semi_diameter', 12.7)), 0.1)
        with b3d.BuildPart() as part:
            if is_rect:
                hx = max(float(getattr(surf, 'aperture_x_half', 0.0)), 0.001)
                hy = max(float(getattr(surf, 'aperture_y_half', 0.0)), 0.001)
                if hx == 0.001 and hy == 0.001:
                    hx, hy = sd, sd
                with b3d.BuildSketch(b3d.Plane.XY):
                    b3d.Circle(radius=sd)
                    b3d.Rectangle(2.0 * hx, 2.0 * hy, mode=b3d.Mode.SUBTRACT)
                b3d.extrude(amount=-t)
            else:
                r_in = float(getattr(surf, 'aperture_r_outer', 0.0))
                if r_in <= 0.0:
                    r_in = sd
                r_out = max(sd, r_in + 0.001)
                with b3d.BuildSketch(b3d.Plane.XY):
                    b3d.Circle(radius=r_out)
                    if r_in > 0:
                        b3d.Circle(radius=r_in, mode=b3d.Mode.SUBTRACT)
                b3d.extrude(amount=-t)
        return part.part
    except Exception:
        return None

def create_prism_solid(surfaces: Sequence[Any], config: Optional[Any] = None, **kwargs) -> Optional[Any]:
    """
    Create non-axisymmetric polyhedral B-Rep solid for optical prisms
    (Equilateral, Right-Angle, Wedge, Dove, etc.) using build123d sketches
    and extrusions/lofts bounded by prism facet planes and clear aperture.
    """
    b3d = _get_build123d()
    if b3d is None:
        return None
    if not surfaces:
        return None
    s1 = surfaces[0]
    s2 = surfaces[1] if len(surfaces) > 1 else None

    cfg = config or getattr(s1, "prism_config", None) or (getattr(s2, "prism_config", None) if s2 else None)
    
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

    try:
        with b3d.BuildPart() as part:
            with b3d.BuildSketch(b3d.Plane.XZ):
                if "right_angle" in geom_type or "rightangle" in geom_type or abs(apex_deg - 90.0) < 1.0:
                    b3d.Polygon((-base_w / 2.0, 0.0), (base_w / 2.0, 0.0), (-base_w / 2.0, height))
                elif "wedge" in geom_type or apex_deg <= 15.0:
                    z_wedge = base_w * math.tan(math.radians(apex_deg))
                    b3d.Polygon(
                        (-base_w / 2.0, 0.0),
                        (base_w / 2.0, 0.0),
                        (base_w / 2.0, height + z_wedge),
                        (-base_w / 2.0, height),
                    )
                else:
                    if 0.0 < apex_deg < 179.0:
                        ht_apex = (base_w / 2.0) / math.tan(math.radians(apex_deg / 2.0))
                    else:
                        ht_apex = height
                    if getattr(cfg, "height_mm", 0.0) > 0 and "height_mm" not in kwargs:
                        ht_apex = float(cfg.height_mm)
                    b3d.Polygon((-base_w / 2.0, 0.0), (base_w / 2.0, 0.0), (0.0, ht_apex))
            b3d.extrude(amount=w / 2.0, both=True)
        return part.part
    except Exception:
        try:
            with b3d.BuildPart() as part:
                b3d.Box(base_w, w, height)
            return part.part
        except Exception:
            return None

def create_beamsplitter_solid(surfaces: Sequence[Any], config: Optional[Any] = None, **kwargs) -> Optional[Any]:
    """
    Create cube or plate B-Rep solid for beam splitters with internal splitting interface.
    For cubes, creates two cemented right-angle prism solids sharing a diagonal splitting interface.
    For plates, creates planar optical substrate with splitting interface face.
    """
    b3d = _get_build123d()
    if b3d is None:
        return None
    if not surfaces:
        return None
    s1 = surfaces[0]
    s2 = surfaces[1] if len(surfaces) > 1 else None

    cfg = config or getattr(s1, "beamsplitter_config", None) or (getattr(s2, "beamsplitter_config", None) if s2 else None)
    
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
        try:
            with b3d.BuildPart() as p1:
                with b3d.BuildSketch(b3d.Plane.XZ):
                    b3d.Polygon((-L / 2.0, 0.0), (L / 2.0, 0.0), (L / 2.0, L))
                b3d.extrude(amount=L / 2.0, both=True)
            with b3d.BuildPart() as p2:
                with b3d.BuildSketch(b3d.Plane.XZ):
                    b3d.Polygon((-L / 2.0, 0.0), (L / 2.0, L), (-L / 2.0, L))
                b3d.extrude(amount=L / 2.0, both=True)
            p1_solid = p1.part
            p2_solid = p2.part
            p1_solid.label = "Prism_Half_1"
            p2_solid.label = "Prism_Half_2"
            cube_compound = b3d.Compound(children=[p1_solid, p2_solid])
            return cube_compound
        except Exception:
            try:
                with b3d.BuildPart() as part:
                    b3d.Box(L, L, L)
                return part.part
            except Exception:
                return None
    else:
        t_default = float(getattr(cfg, "substrate_thickness_mm", 0.0) if cfg else 0.0)
        if t_default <= 0.0:
            t_default = float(getattr(s1, "substrate_thickness", 0.0) or getattr(s1, "thickness", 0.0) or 2.0)
        t = max(float(kwargs.get("substrate_thickness_mm", kwargs.get("thickness", t_default))), 0.5)

        wx_default = float(getattr(s1, "aperture_x_half", 0.0) * 2.0 or sd * 2.0 or 25.4)
        wy_default = float(getattr(s1, "aperture_y_half", 0.0) * 2.0 or sd * 2.0 or 25.4)
        wx = float(kwargs.get("clear_aperture_x", kwargs.get("width", wx_default)))
        wy = float(kwargs.get("clear_aperture_y", kwargs.get("height", wy_default)))

        is_rect = (
            getattr(s1, 'aperture_type', '') in ["Rectangular", "Slit"]
            or getattr(s1, 'substrate_shape', '') == "Rectangular"
            or "width" in kwargs
            or "clear_aperture_x" in kwargs
            or getattr(cfg, "clear_aperture_x_mm", 0.0) > 0.0
        )
        try:
            with b3d.BuildPart() as part:
                if is_rect:
                    with b3d.BuildSketch(b3d.Plane.XY):
                        b3d.Rectangle(wx, wy)
                    b3d.extrude(amount=-t)
                else:
                    with b3d.BuildSketch(b3d.Plane.XY):
                        b3d.Circle(radius=sd)
                    b3d.extrude(amount=-t)
            return part.part
        except Exception:
            try:
                with b3d.BuildPart() as part:
                    b3d.Box(wx, wy, t)
                return part.part
            except Exception:
                return None

def create_flat_element(surf, thickness=None):
    """Backward-compatible wrapper for create_mirror_or_planar_solid."""
    return create_mirror_or_planar_solid(surf, thickness=thickness)

def get_exportable_elements(system_or_stack: Any) -> List[Dict[str, Any]]:
    """
    Parse optical system or SurfaceStack into discrete exportable optical components
    (Prisms, Beam Splitters, Galvo Scanners, Active Mechanisms, Singlet lenses,
    Cemented Doublets, Mirrors, Gratings, Stops/Diaphragms, Detectors, Sample Stages, Air Gaps).
    Returns list of component metadata dictionaries for UI selection and selective CAD export.
    """
    stack = _extract_surface_stack(system_or_stack)
    if stack is None or not getattr(stack, "surfaces", None):
        return []

    try:
        stack._update_accumulated_positions()
    except Exception:
        pass

    elements: List[Dict[str, Any]] = []
    i = 0
    elem_idx = 0

    def is_refractive(s):
        return (
            (s.glass.upper() not in ["AIR", "", "NONE", "MIRROR", "MIRR", "REFL"])
            and (not s.is_reflective())
            and (s.surface_type not in [SurfaceType.MIRROR, SurfaceType.GRATING])
        )

    while i < len(stack.surfaces):
        surf = stack.surfaces[i]
        if bool(getattr(surf, "is_air_space", False)) and float(getattr(surf, "thickness", 0.0)) <= 0.0:
            i += 1
            continue

        # Check if surface belongs to an OpticalComponent in stack.components
        c_uid = getattr(surf, "component_uid", "")
        comp_match = None
        if c_uid and hasattr(stack, "components") and isinstance(stack.components, list):
            for comp in stack.components:
                if getattr(comp, "component_uid", "") == c_uid:
                    comp_match = comp
                    break
        c_type = str(getattr(comp_match, "component_type", "")).lower() if comp_match else ""
        if c_type == "prism" and not getattr(surf, "prism_config", None):
            surf.prism_config = getattr(comp_match, "prism_config", None)
        elif c_type in ("beamsplitter", "beam_splitter") and not getattr(surf, "beamsplitter_config", None):
            surf.beamsplitter_config = getattr(comp_match, "beamsplitter_config", None)
        elif c_type == "galvo" and not getattr(surf, "galvo_config", None):
            surf.galvo_config = getattr(comp_match, "galvo_config", None)

        # 1. Optical Prism
        has_prism_config = getattr(surf, "prism_config", None) is not None
        is_prism_name = "PRISM" in (surf.name or "").upper() or "PRISM" in (getattr(surf, "group_name", "") or "").upper()
        is_prism_type = c_type == "prism" or getattr(surf, "mechanism_type", "").upper() == "PRISM" or getattr(surf, "component_type", "").lower() == "prism" or getattr(surf, "group_type", "").upper() == "PRISM"
        if has_prism_config or is_prism_name or is_prism_type:
            c_uid = getattr(surf, "component_uid", "")
            p_surfs = [surf]
            p_indices = [i]
            next_idx = i + 1
            while next_idx < len(stack.surfaces):
                s_next = stack.surfaces[next_idx]
                if bool(getattr(s_next, "is_air_space", False)):
                    next_idx += 1
                    continue
                same_comp = c_uid and getattr(s_next, "component_uid", "") == c_uid
                has_cfg_next = getattr(s_next, "prism_config", None) is not None
                name_match = "PRISM" in (s_next.name or "").upper()
                if same_comp or has_cfg_next or (name_match and len(p_surfs) == 1):
                    p_surfs.append(s_next)
                    p_indices.append(next_idx)
                    next_idx += 1
                    break
                elif len(p_surfs) == 1 and surf.thickness > 0 and s_next.glass.upper() in ["AIR", "", "NONE"]:
                    p_surfs.append(s_next)
                    p_indices.append(next_idx)
                    next_idx += 1
                    break
                else:
                    break
            cfg = getattr(surf, "prism_config", None) or (getattr(p_surfs[-1], "prism_config", None) if len(p_surfs) > 1 else None)
            name = surf.group_name or surf.name or f"Prism_{elem_idx + 1}"
            if not name or name == f"S{i}":
                name = f"Prism ({surf.glass or 'Glass'})"
            max_sd = max(float(getattr(s, "semi_diameter", 12.7)) for s in p_surfs)
            elements.append({
                "index": elem_idx,
                "name": name,
                "type": "Prism",
                "surface_indices": p_indices,
                "surfaces": p_surfs,
                "material": surf.glass or "N-SF11",
                "config": cfg,
                "semi_diameter": max_sd,
                "clear_aperture_diam": 2.0 * max_sd,
                "z_position": float(surf.global_origin[2]),
                "origin": surf.global_origin,
                "is_default_selected": True,
            })
            elem_idx += 1
            i = next_idx
            continue

        # 2. Beam Splitter (Cube or Plate)
        has_bs_config = getattr(surf, "beamsplitter_config", None) is not None or bool(getattr(surf, "is_beam_splitter", False))
        is_bs_name = any(k in (surf.name or "").upper() for k in ["BEAMSPLITTER", "BEAM_SPLITTER", "BEAM SPLITTER", "SPLITTER"]) or any(k in (getattr(surf, "group_name", "") or "").upper() for k in ["BEAMSPLITTER", "BEAM_SPLITTER", "BEAM SPLITTER", "SPLITTER"])
        is_bs_type = c_type in ("beamsplitter", "beam_splitter") or getattr(surf, "mechanism_type", "").upper() in ["BEAM_SPLITTER", "BEAMSPLITTER", "CUBE_BEAMSPLITTER", "PLATE_BEAMSPLITTER"] or getattr(surf, "component_type", "").lower() in ["beamsplitter", "beam_splitter"]
        if has_bs_config or is_bs_name or is_bs_type:
            c_uid = getattr(surf, "component_uid", "")
            bs_surfs = [surf]
            bs_indices = [i]
            next_idx = i + 1
            while next_idx < len(stack.surfaces):
                s_next = stack.surfaces[next_idx]
                if bool(getattr(s_next, "is_air_space", False)):
                    next_idx += 1
                    continue
                same_comp = c_uid and getattr(s_next, "component_uid", "") == c_uid
                has_cfg_next = getattr(s_next, "beamsplitter_config", None) is not None
                name_match = any(k in (s_next.name or "").upper() for k in ["BEAMSPLITTER", "BEAM_SPLITTER", "BEAM SPLITTER", "SPLITTER", "REAR"])
                if same_comp or has_cfg_next or (name_match and len(bs_surfs) == 1):
                    bs_surfs.append(s_next)
                    bs_indices.append(next_idx)
                    next_idx += 1
                    break
                elif len(bs_surfs) == 1 and surf.thickness > 0 and s_next.glass.upper() in ["AIR", "", "NONE"]:
                    bs_surfs.append(s_next)
                    bs_indices.append(next_idx)
                    next_idx += 1
                    break
                else:
                    break
            cfg = getattr(surf, "beamsplitter_config", None) or (getattr(bs_surfs[-1], "beamsplitter_config", None) if len(bs_surfs) > 1 else None)
            bs_type_label = getattr(cfg, "splitter_type", "Beam Splitter") if cfg else "Beam Splitter"
            name = surf.group_name or surf.name or f"BeamSplitter_{elem_idx + 1}"
            if not name or name == f"S{i}":
                name = f"Beam Splitter ({bs_type_label})"
            mat = getattr(cfg, "substrate_glass", surf.glass or "N-BK7") if cfg else (surf.glass or "N-BK7")
            max_sd = max(float(getattr(s, "semi_diameter", 12.7)) for s in bs_surfs)
            elements.append({
                "index": elem_idx,
                "name": name,
                "type": "Beam Splitter",
                "surface_indices": bs_indices,
                "surfaces": bs_surfs,
                "material": mat,
                "config": cfg,
                "semi_diameter": max_sd,
                "clear_aperture_diam": 2.0 * max_sd,
                "z_position": float(surf.global_origin[2]),
                "origin": surf.global_origin,
                "is_default_selected": True,
            })
            elem_idx += 1
            i = next_idx
            continue

        # 3. Galvo Scanning Mirror
        is_g = c_type == "galvo" or getattr(surf, "galvo_config", None) is not None or getattr(surf, "mechanism_type", "").upper() == "GALVO" or getattr(surf, "component_type", "").lower() == "galvo" or is_galvo(surf)
        if is_g:
            name = surf.name or f"GalvoScanner_{elem_idx + 1}"
            sd = float(surf.semi_diameter)
            elements.append({
                "index": elem_idx,
                "name": name,
                "type": "Galvo Scanner",
                "surface_indices": [i],
                "surfaces": [surf],
                "material": "Galvo Scanning Mirror",
                "config": getattr(surf, "galvo_config", None),
                "semi_diameter": sd,
                "clear_aperture_diam": 2.0 * sd,
                "z_position": float(surf.global_origin[2]),
                "origin": surf.global_origin,
                "is_default_selected": True,
            })
            elem_idx += 1
            i += 1
            continue

        # 4. Active Mechanism (Discrete states, flip mounts, arbitrary actuators)
        has_mech = (
            c_type in ("mechanism", "active_mechanism")
            or (getattr(surf, "mechanism", None) is not None)
            or (bool(getattr(surf, "mechanical_states", None)) and not is_dmd(surf))
            or (getattr(surf, "mechanism_type", "") and getattr(surf, "mechanism_type", "").upper() not in ["", "NONE", "STANDARD", "DMD"])
        )
        if has_mech:
            name = surf.name or f"ActiveMechanism_{elem_idx + 1}"
            sd = float(surf.semi_diameter)
            elements.append({
                "index": elem_idx,
                "name": name,
                "type": "Active Mechanism",
                "surface_indices": [i],
                "surfaces": [surf],
                "material": "Actuated Optical Mechanism",
                "config": getattr(surf, "mechanism", None),
                "semi_diameter": sd,
                "clear_aperture_diam": 2.0 * sd,
                "z_position": float(surf.global_origin[2]),
                "origin": surf.global_origin,
                "is_default_selected": True,
            })
            elem_idx += 1
            i += 1
            continue

        is_refr_1 = is_refractive(surf)
        is_refr_2 = is_refractive(stack.surfaces[i + 1]) if (i + 1 < len(stack.surfaces)) else False

        # 5. Cemented Refractive Doublet (e.g. glass1 -> glass2 -> air)
        if (
            i + 2 < len(stack.surfaces)
            and is_refr_1
            and surf.thickness > 0
            and is_refr_2
            and stack.surfaces[i + 1].thickness > 0
        ):
            s1 = surf
            s2 = stack.surfaces[i + 1]
            s3 = stack.surfaces[i + 2]
            name = s1.group_name or s1.name or f"Doublet_{elem_idx + 1}"
            if not name or name == f"S{i}":
                name = f"Achromatic Doublet ({s1.glass} / {s2.glass})"
            max_sd = max(float(s1.semi_diameter), float(s2.semi_diameter), float(s3.semi_diameter))
            elements.append({
                "index": elem_idx,
                "name": name,
                "type": "Doublet",
                "surface_indices": [i, i + 1, i + 2],
                "surfaces": [s1, s2, s3],
                "material": f"{s1.glass} / {s2.glass}",
                "semi_diameter": max_sd,
                "clear_aperture_diam": 2.0 * max_sd,
                "z_position": float(s1.global_origin[2]),
                "origin": s1.global_origin,
                "is_default_selected": True,
            })
            elem_idx += 1
            i += 2
            continue

        # 6. Singlet Lens
        elif surf.thickness > 0 and i < len(stack.surfaces) - 1 and is_refr_1:
            s1 = surf
            s2 = stack.surfaces[i + 1]
            name = s1.name or f"Lens_{elem_idx + 1}"
            if not name or name == f"S{i}":
                name = f"Singlet Lens ({s1.glass})"
            max_sd = max(float(s1.semi_diameter), float(s2.semi_diameter))
            elements.append({
                "index": elem_idx,
                "name": name,
                "type": "Singlet",
                "surface_indices": [i, i + 1],
                "surfaces": [s1, s2],
                "material": s1.glass,
                "semi_diameter": max_sd,
                "clear_aperture_diam": 2.0 * max_sd,
                "z_position": float(s1.global_origin[2]),
                "origin": s1.global_origin,
                "is_default_selected": True,
            })
            elem_idx += 1
            i += 1
            continue



        # 8. Mirror, Grating, DMD, Detector
        elif (
            surf.is_reflective()
            or getattr(surf, "glass", "").upper() in ["MIRROR", "MIRR", "REFL"]
            or surf.surface_type in [SurfaceType.MIRROR, SurfaceType.GRATING, SurfaceType.DETECTOR]
            or "DMD" in (surf.name or "").upper()
            or "MIRROR" in (surf.name or "").upper()
            or "GRATING" in (surf.name or "").upper()
            or "DETECTOR" in (surf.name or "").upper()
            or "CCD" in (surf.name or "").upper()
            or "SENSOR" in (surf.name or "").upper()
        ):
            s_name_up = (surf.name or "").upper()
            if surf.surface_type == SurfaceType.GRATING or "GRATING" in s_name_up:
                etype = "Grating"
                lines = getattr(surf, "grating_lines_per_mm", 0)
                mat = f"Diffraction Grating ({lines:.0f} lp/mm)" if lines > 0 else "Diffraction Grating"
            elif "DMD" in s_name_up:
                etype = "DMD"
                mat = "Micromirror Array"
            elif surf.surface_type == SurfaceType.DETECTOR or any(k in s_name_up for k in ["DETECTOR", "CCD", "SENSOR", "IMAGE"]):
                etype = "Detector"
                mat = "Si Sensor"
            else:
                etype = "Mirror"
                mat = "Mirror Substrate"

            name = surf.name or f"{etype}_{elem_idx + 1}"
            sd = float(surf.semi_diameter)
            elements.append({
                "index": elem_idx,
                "name": name,
                "type": etype,
                "surface_indices": [i],
                "surfaces": [surf],
                "material": mat,
                "semi_diameter": sd,
                "clear_aperture_diam": 2.0 * sd,
                "z_position": float(surf.global_origin[2]),
                "origin": surf.global_origin,
                "is_default_selected": True,
            })
            elem_idx += 1

            # If this reflective element carries a propagation distance to the next surface,
            # export that space as an explicit Air Gap cylinder without altering the layout stack!
            if surf.thickness > 0.0 and i + 1 < len(stack.surfaces):
                s_next = stack.surfaces[i + 1]
                gap_sd = max(sd, float(getattr(s_next, "semi_diameter", 12.7)), 2.0)
                elements.append({
                    "index": elem_idx,
                    "name": f"Air Gap ({surf.name} -> {s_next.name})",
                    "type": "Air Gap",
                    "surface_indices": [i, i + 1],
                    "surfaces": [surf, s_next],
                    "material": "Air",
                    "semi_diameter": gap_sd,
                    "clear_aperture_diam": 2.0 * gap_sd,
                    "z_position": float(surf.global_origin[2]),
                    "origin": surf.global_origin,
                    "is_default_selected": False,
                })
                elem_idx += 1

            i += 1
            continue


        # 8. Aperture Stop / Slit / Baffle
        elif (
            surf.surface_type == SurfaceType.STOP
            or getattr(surf, "is_stop", False)
            or getattr(surf, "is_aperture_stop", False)
            or "STOP" in (surf.name or "").upper()
            or "(STO)" in (surf.name or "").upper()
            or getattr(surf, "aperture_type", "") in ["Rectangular", "Slit"]
        ):
            etype = "Aperture Stop"
            name = surf.name or f"ApertureStop_{elem_idx + 1}"
            sd = float(surf.semi_diameter)
            elements.append({
                "index": elem_idx,
                "name": name,
                "type": etype,
                "surface_indices": [i],
                "surfaces": [surf],
                "material": "Aperture Diaphragm",
                "semi_diameter": sd,
                "clear_aperture_diam": 2.0 * sd,
                "z_position": float(surf.global_origin[2]),
                "origin": surf.global_origin,
                "is_default_selected": True,
            })
            elem_idx += 1
            i += 1
            continue

        # 9. Object / Sample Stage
        elif (
            surf.surface_type == SurfaceType.OBJECT
            or (i == 0 and surf.name and any(k in surf.name.upper() for k in ["SAMPLE", "OBJECT", "SPECIMEN"]))
        ):
            etype = "Sample Stage"
            name = surf.name or "Object / Sample Stage"
            sd = float(surf.semi_diameter)
            elements.append({
                "index": elem_idx,
                "name": name,
                "type": etype,
                "surface_indices": [i],
                "surfaces": [surf],
                "material": "Stage / Mount",
                "semi_diameter": sd,
                "clear_aperture_diam": 2.0 * sd,
                "z_position": float(surf.global_origin[2]),
                "origin": surf.global_origin,
                "is_default_selected": True,
            })
            elem_idx += 1
            i += 1
            continue

        # 10. Air Gap
        elif surf.thickness > 0 and i < len(stack.surfaces) - 1 and (surf.glass.upper() in ["AIR", "", "NONE"]):
            s1 = surf
            s2 = stack.surfaces[i + 1]
            name = surf.name or f"Air_Gap_{elem_idx + 1}"
            if not name or name == f"S{i}":
                name = f"Air Gap (S{i}-S{i+1})"
            max_sd = max(get_surface_clear_radius(s1), get_surface_clear_radius(s2))
            elements.append({
                "index": elem_idx,
                "name": name,
                "type": "Air Gap",
                "surface_indices": [i, i + 1],
                "surfaces": [s1, s2],
                "material": "Air",
                "semi_diameter": max_sd,
                "clear_aperture_diam": 2.0 * max_sd,
                "z_position": float(s1.global_origin[2]),
                "origin": s1.global_origin,
                "is_default_selected": False,
            })
            elem_idx += 1
            i += 1
            continue

        else:
            i += 1
            continue

    return elements

class STEPExporter:
    @classmethod
    def export(
        cls,
        system_or_stack: Any,
        file_path: str,
        options: Optional[STEPExportOptions] = None,
        engine: Any = "auto",
        selected_element_indices: Optional[Collection[int]] = None,
        selected_surface_indices: Optional[Collection[int]] = None,
    ) -> STEPExportResult:
        engine_enum = STEPExportEngine.AUTO
        if options is not None:
            if isinstance(options.engine, STEPExportEngine):
                engine_enum = options.engine
            elif str(options.engine).lower() in ["build123d", "cadquery", "occ"]:
                engine_enum = STEPExportEngine.BUILD123D
            elif str(options.engine).lower() in ["pure_python", "python"]:
                engine_enum = STEPExportEngine.PURE_PYTHON
            if selected_element_indices is None and getattr(options, "selected_element_indices", None) is not None:
                selected_element_indices = options.selected_element_indices
            if selected_surface_indices is None and getattr(options, "selected_surface_indices", None) is not None:
                selected_surface_indices = options.selected_surface_indices
        elif isinstance(engine, str):
            if engine.lower() in ["build123d", "cadquery", "occ"]:
                engine_enum = STEPExportEngine.BUILD123D
            elif engine.lower() in ["pure_python", "python"]:
                engine_enum = STEPExportEngine.PURE_PYTHON
            elif engine.lower() == "auto":
                engine_enum = STEPExportEngine.BUILD123D if _get_build123d() is not None else STEPExportEngine.PURE_PYTHON
        elif isinstance(engine, STEPExportEngine):
            engine_enum = engine

        if engine_enum == STEPExportEngine.AUTO:
            engine_enum = STEPExportEngine.BUILD123D if _get_build123d() is not None else STEPExportEngine.PURE_PYTHON

        stack = _extract_surface_stack(system_or_stack)
        if stack is None:
            return STEPExportResult(file_path, False, 0, engine_enum, error_message="Unable to extract SurfaceStack from input.")

        try:
            dir_name = os.path.dirname(os.path.abspath(file_path))
            if dir_name and not os.path.exists(dir_name):
                os.makedirs(dir_name, exist_ok=True)
        except Exception as e:
            return STEPExportResult(file_path, False, 0, engine_enum, error_message=str(e))

        all_elements = get_exportable_elements(stack)
        selected_elem_set = set(selected_element_indices) if selected_element_indices is not None else None
        selected_surf_set = set(selected_surface_indices) if selected_surface_indices is not None else None

        if engine_enum == STEPExportEngine.PURE_PYTHON:
            try:
                stack._update_accumulated_positions()
                step_content = [
                    "ISO-10303-21;",
                    "HEADER;",
                    "FILE_DESCRIPTION(('2D_Optics Pure Python STEP Export (AP203/AUTOMOTIVE_DESIGN)'),'2;1');",
                    f"FILE_NAME('{os.path.basename(file_path)}','2026-08-20T12:00:00',('OpticsUser'),('2D_Optics'),'2D_Optics CAD Exporter','2D_Optics','');",
                    "FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));",
                    "ENDSEC;",
                    "DATA;",
                    "#1 = APPLICATION_CONTEXT('optical configuration design');",
                    "#2 = APPLICATION_PROTOCOL_DEFINITION('international standard','automotive_design',2000,#1);",
                    "#3 = PRODUCT_CONTEXT('',#1,'mechanical');",
                    "#4 = PRODUCT('OPTICAL_ASSEMBLY','Optical Assembly System','',(#3));",
                    "#5 = PRODUCT_DEFINITION_FORMATION('1.0','Initial Release',#4);",
                    "#6 = PRODUCT_DEFINITION('design','Optical System Assembly Definition',#5,#3);",
                    "#7 = PRODUCT_DEFINITION_SHAPE('Assembly Shape','',#6);",
                    "#8 = ( LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.MILLI.,.METRE.) );",
                    "#9 = ( NAMED_UNIT(*) PLANE_ANGLE_UNIT() SI_UNIT($,.RADIAN.) );",
                    "#10 = ( NAMED_UNIT(*) SI_UNIT($,.STERADIAN.) SOLID_ANGLE_UNIT() );",
                    "#11 = UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE(1.0E-07),#8,'closure',.UNCERTAINTY.);",
                    "#12 = ( GEOMETRIC_REPRESENTATION_CONTEXT(3) GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT((#11)) GLOBAL_UNIT_ASSIGNED_CONTEXT((#8,#9,#10)) REPRESENTATION_CONTEXT('Context #1','3D') );",
                    "#13 = CARTESIAN_POINT('',(0.0,0.0,0.0));",
                    "#14 = DIRECTION('',(0.0,0.0,1.0));",
                    "#15 = DIRECTION('',(1.0,0.0,0.0));",
                    "#16 = AXIS2_PLACEMENT_3D('',#13,#14,#15);",
                ]
                
                entity_id = 20
                comp_idx = 1
                for elem in all_elements:
                    elem_idx = elem["index"]
                    if selected_elem_set is not None:
                        if elem_idx not in selected_elem_set:
                            continue
                    elif selected_surf_set is not None:
                        if not any(s_idx in selected_surf_set for s_idx in elem["surface_indices"]):
                            continue
                    else:
                        if not elem.get("is_default_selected", True):
                            continue

                    first_surf = elem["surfaces"][0]
                    clean_name = elem['name'].replace(' ', '_').replace('-', '_')
                    p_name = f"Element_{elem_idx + 1}_{clean_name}"
                    eff_origin, eff_R = get_effective_pose(first_surf)
                    ox, oy, oz = float(eff_origin[0]), float(eff_origin[1]), float(eff_origin[2])
                    step_content.append(f"#{entity_id} = PRODUCT('{p_name}','{p_name}','',(#3));")
                    step_content.append(f"#{entity_id+1} = PRODUCT_DEFINITION_FORMATION('1.0','',#{entity_id});")
                    step_content.append(f"#{entity_id+2} = PRODUCT_DEFINITION('design','',#{entity_id+1},#3);")
                    step_content.append(f"#{entity_id+3} = PRODUCT_DEFINITION_SHAPE('','',#{entity_id+2});")
                    step_content.append(f"#{entity_id+4} = CARTESIAN_POINT('',({ox:.6f},{oy:.6f},{oz:.6f}));")
                    step_content.append(f"#{entity_id+5} = AXIS2_PLACEMENT_3D('Placement',#{entity_id+4},#14,#15);")
                    step_content.append(f"#{entity_id+6} = MANIFOLD_SOLID_BREP('{p_name}_Solid',#{entity_id+7});")
                    step_content.append(f"#{entity_id+7} = CLOSED_SHELL('{p_name}_Shell',(#{entity_id+8},#{entity_id+9}));")
                    step_content.append(f"#{entity_id+8} = ADVANCED_FACE('{p_name}_Face1',(),#{entity_id+5},.T.);")
                    step_content.append(f"#{entity_id+9} = ADVANCED_FACE('{p_name}_Face2',(),#{entity_id+5},.T.);")
                    step_content.append(f"#{entity_id+10} = SHAPE_REPRESENTATION('{p_name}_Shape',(#16,#{entity_id+5},#{entity_id+6}),#12);")
                    step_content.append(f"#{entity_id+11} = SHAPE_DEFINITION_REPRESENTATION(#{entity_id+3},#{entity_id+10});")
                    entity_id += 15
                    comp_idx += 1
                step_content.append("ENDSEC;")
                step_content.append("END-ISO-10303-21;")

                with open(file_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(step_content))
                size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
                return STEPExportResult(file_path, True, max(0, comp_idx - 1), STEPExportEngine.PURE_PYTHON, size)
            except Exception as e:
                return STEPExportResult(file_path, False, 0, STEPExportEngine.PURE_PYTHON, error_message=str(e))

        b3d = _get_build123d()
        if b3d is None:
            return STEPExportResult(file_path, False, 0, STEPExportEngine.BUILD123D, error_message="build123d not available.")

        try:
            stack._update_accumulated_positions()
            parts = []
            pocket_parts = []

            export_enclosure = options.export_3d_printable_enclosure if options else False
            export_air_only = options.export_mated_air_cavity if options else False

            for elem in all_elements:
                elem_idx = elem["index"]
                if export_enclosure or export_air_only:
                    # When generating an enclosure or air cavity, include all elements and air volumes
                    pass
                elif selected_elem_set is not None:
                    if elem_idx not in selected_elem_set:
                        continue
                elif selected_surf_set is not None:
                    if not any(s_idx in selected_surf_set for s_idx in elem["surface_indices"]):
                        continue
                else:
                    if not elem.get("is_default_selected", True):
                        continue

                part_3d = None
                clean_name = elem["name"].replace(' ', '_').replace('-', '_')
                label_name = f"Element_{elem_idx + 1}_{clean_name}"
                etype = elem["type"]
                surfs = elem["surfaces"]

                if etype == "Doublet":
                    s1, s2, s3 = surfs[0], surfs[1], surfs[2]
                    e1 = create_solid_lens(s1, s2)
                    e2_local = create_solid_lens(s2, s3)
                    if e1 is not None and e2_local is not None:
                        e2 = e2_local.locate(b3d.Location((0, 0, s1.thickness)))
                        try:
                            doublet = e1 + e2
                        except Exception:
                            doublet = b3d.Compound(children=[e1, e2])
                        part_3d = doublet
                    elif e1 is not None:
                        part_3d = e1
                elif etype == "Prism":
                    part_3d = create_prism_solid(surfs, config=elem.get("config"))
                elif etype == "Beam Splitter":
                    part_3d = create_beamsplitter_solid(surfs, config=elem.get("config"))
                elif etype == "Singlet":
                    s1, s2 = surfs[0], surfs[1]
                    part_3d = create_solid_lens(s1, s2)
                elif etype in ["Air Volume", "Air Gap"]:
                    s1, s2 = surfs[0], surfs[1]
                    const_diam = float(getattr(options, "air_gap_cylinder_diameter", 0.0) if options else 0.0)
                    part_3d = create_air_gap_solid(s1, s2, constant_diameter=const_diam)
                elif etype in ["Mirror", "DMD", "Detector", "Galvo Scanner", "Active Mechanism"]:
                    part_3d = create_mirror_or_planar_solid(surfs[0])
                elif etype == "Grating":
                    part_3d = create_mirror_or_planar_solid(surfs[0])
                    if part_3d is not None:
                        # Optionally add visual differentiation for grating in CAD if needed
                        part_3d.label = f"{label_name}_Grating_Element"
                elif etype == "Aperture Stop":
                    part_3d = create_aperture_stop_solid(surfs[0])
                elif etype == "Sample Stage":
                    part_3d = create_mirror_or_planar_solid(surfs[0], thickness=1.8)

                if part_3d is not None:
                    first_surf = surfs[0]
                    eff_origin, eff_R = get_effective_pose(first_surf)

                    # Apply axial displacement (+/- dz mm) if configured for this element
                    elem_disps = getattr(options, "element_displacements", {}) if options else {}
                    dz = float(elem_disps.get(elem_idx, 0.0))
                    if abs(dz) > 1e-5:
                        eff_origin = eff_origin + eff_R[:, 2] * dz

                    loc = tuple(eff_origin)
                    rot_deg = R_mat_to_euler_degrees(eff_R)
                    placed_part = part_3d.locate(b3d.Location(loc, rot_deg))
                    placed_part.label = label_name
                    parts.append(placed_part)

                    # Generate mechanical mount collar / cell for optical components if enabled
                    if options and getattr(options, "export_mechanical_mounts", False) and etype in ["Singlet", "Doublet", "Mirror"]:
                        m_diam = float(getattr(options, "mount_diameter", 30.5) or 30.5)
                        m_z = float(getattr(options, "mount_z_thickness", 0.0) or 0.0)
                        sd_elem = float(elem.get("semi_diameter", 12.7))
                        r_out = m_diam / 2.0 if m_diam > 0 else (sd_elem + 2.5)
                        if r_out <= sd_elem:
                            r_out = sd_elem + 2.5

                        elem_t = sum(float(getattr(s, "thickness", 0.0)) for s in surfs[:-1]) if len(surfs) > 1 else float(getattr(first_surf, "thickness", 3.0))
                        if elem_t <= 0.0:
                            elem_t = float(getattr(first_surf, "substrate_thickness", 6.0) or 6.0)

                        if m_z > 0:
                            collar_t = m_z
                            z_off = (elem_t - collar_t) / 2.0
                        else:
                            collar_t = max(elem_t, 2.0)
                            z_off = 0.0

                        mount_solid = create_lens_mount_solid(inner_radius=sd_elem, outer_radius=r_out, thickness=collar_t, z_offset=z_off)
                        if mount_solid is not None:
                            placed_mount = mount_solid.locate(b3d.Location(loc, rot_deg))
                            placed_mount.label = f"{label_name}_Mount"
                            parts.append(placed_mount)

                    # Generate mounting pocket solid with +/- tolerance clearance for enclosure boolean cut
                    tol_mm = float(getattr(options, "tolerance_thickness_mm", 1.0) if options else 1.0)
                    if export_enclosure and tol_mm > 0 and etype not in ["Air Volume", "Air Gap"]:
                        elem_sd = float(elem.get("semi_diameter", 12.7))
                        elem_t = sum(float(getattr(s, "thickness", 0.0)) for s in surfs[:-1]) if len(surfs) > 1 else float(getattr(first_surf, "thickness", 3.0))
                        if elem_t <= 0.0:
                            elem_t = float(getattr(first_surf, "substrate_thickness", 6.0) or 6.0)
                        pocket_solid = create_tolerance_pocket_solid(
                            radius=elem_sd + 0.25,
                            thickness=elem_t,
                            tolerance=tol_mm,
                            z_offset=0.0
                        )
                        if pocket_solid is not None:
                            nom_origin, nom_R = get_effective_pose(first_surf)
                            placed_pocket = pocket_solid.locate(b3d.Location(tuple(nom_origin), R_mat_to_euler_degrees(nom_R)))
                            placed_pocket.label = f"{label_name}_TolPocket"
                            pocket_parts.append(placed_pocket)

            if parts:
                if export_enclosure and b3d is not None:
                    try:
                        # 3D Printable Bench Enclosure Generation via Smart Subtraction
                        all_to_subtract = parts + pocket_parts
                        all_solid = b3d.Compound(children=all_to_subtract)
                        bbox = all_solid.bounding_box()
                        wall_margin = float(options.enclosure_wall_margin_mm if options and getattr(options, "enclosure_wall_margin_mm", 0) > 0 else 10.0)
                        
                        box_dx = bbox.size.X + 2.0 * wall_margin
                        box_dy = bbox.size.Y + 2.0 * wall_margin
                        box_dz = bbox.size.Z + 2.0 * wall_margin
                        c_pt = bbox.center()
                        
                        with b3d.BuildPart() as p_enc:
                            b3d.Box(box_dx, box_dy, box_dz)
                        solid_blank = p_enc.part.locate(b3d.Location((c_pt.X, c_pt.Y, c_pt.Z)))
                        
                        # Subtract optical element mounting pockets (with +/- tol clearance) and air tunnel paths
                        hollow_enclosure = solid_blank - all_solid
                        hollow_enclosure.label = "3D_Printable_Optomechanical_Enclosure"
                        
                        # In the final 3D printable export, output the hollow enclosure
                        # (optionally with real physical optics, but NEVER with solid air tunnels inside!)
                        optic_parts = [p for p in parts if "Air" not in getattr(p, "label", "")]
                        if options and getattr(options, "export_optics_with_enclosure", False):
                            final_export_solid = b3d.Compound(children=[hollow_enclosure] + optic_parts)
                        else:
                            final_export_solid = hollow_enclosure
                        compound = final_export_solid
                    except Exception as e_enc:
                        # Fallback to standard assembly compound if boolean union/subtraction fails
                        compound = b3d.Compound(children=parts)
                elif export_air_only and b3d is not None:
                    air_parts = [p for p in parts if "Air" in getattr(p, "label", "")]
                    compound = b3d.Compound(children=air_parts if air_parts else parts)
                else:
                    compound = b3d.Compound(children=parts)

                b3d.export_step(compound, file_path)
                if os.path.exists(file_path):
                    try:
                        import re
                        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                            txt = f.read()
                        if "FILE_SCHEMA" in txt and "AUTOMOTIVE_DESIGN" in txt:
                            txt = re.sub(r"FILE_SCHEMA\s*\(\s*\(\s*['\"]AUTOMOTIVE_DESIGN.*?['\"]\s*\)\s*\)\s*;", "FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));", txt)
                            with open(file_path, "w", encoding="utf-8") as f:
                                f.write(txt)
                    except Exception:
                        pass
                size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
                return STEPExportResult(file_path, True, len(parts), STEPExportEngine.BUILD123D, size)
            else:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(f"ISO-10303-21;\nHEADER;\nFILE_DESCRIPTION(('2D_Optics STEP Export (Empty/Degenerate)'),'2;1');\nFILE_NAME('{os.path.basename(file_path)}','2026-08-20T12:00:00',('OpticsUser'),('2D_Optics'),'2D_Optics CAD Exporter','2D_Optics','');\nFILE_SCHEMA(('AUTOMOTIVE_DESIGN'));\nENDSEC;\nDATA;\nENDSEC;\nEND-ISO-10303-21;\n")
                size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
                return STEPExportResult(file_path, True, 0, STEPExportEngine.BUILD123D, size)
        except Exception as e:
            return STEPExportResult(file_path, False, 0, STEPExportEngine.BUILD123D, error_message=str(e))

    @classmethod
    def export_assembly_step(
        cls,
        system_or_stack: Any,
        file_path: str,
        options: Optional[STEPExportOptions] = None,
        engine: Any = "auto",
        selected_element_indices: Optional[Collection[int]] = None,
        selected_surface_indices: Optional[Collection[int]] = None,
    ) -> STEPExportResult:
        return cls.export(
            system_or_stack,
            file_path,
            options=options,
            engine=engine,
            selected_element_indices=selected_element_indices,
            selected_surface_indices=selected_surface_indices,
        )

    @classmethod
    def export_step_assembly(
        cls,
        system_or_stack: Any,
        file_path: str,
        options: Optional[STEPExportOptions] = None,
        engine: Any = "auto",
        selected_element_indices: Optional[Collection[int]] = None,
        selected_surface_indices: Optional[Collection[int]] = None,
    ) -> STEPExportResult:
        return cls.export(
            system_or_stack,
            file_path,
            options=options,
            engine=engine,
            selected_element_indices=selected_element_indices,
            selected_surface_indices=selected_surface_indices,
        )

class CADExporter:
    def __init__(self, system_or_stack: Optional[Any] = None):
        self.system_or_stack = system_or_stack

    @classmethod
    def export_system_step(
        cls,
        stack: Any,
        file_path: str = "",
        selected_element_indices: Optional[Collection[int]] = None,
        selected_surface_indices: Optional[Collection[int]] = None,
        engine: Any = "auto",
        options: Optional[Any] = None,
    ) -> STEPExportResult:
        if options is None:
            options = STEPExportOptions(
                engine=STEPExportEngine(engine) if isinstance(engine, str) and engine in ("auto", "build123d", "pure_python") else STEPExportEngine.AUTO,
                selected_element_indices=selected_element_indices,
                selected_surface_indices=selected_surface_indices,
            )
        return STEPExporter.export(
            stack,
            file_path,
            engine=engine,
            options=options,
            selected_element_indices=selected_element_indices,
            selected_surface_indices=selected_surface_indices,
        )
    
    def export_to_step(
        self_or_cls,
        system_or_path: Any,
        file_path: Optional[str] = None,
        engine: Any = "auto",
        selected_element_indices: Optional[Collection[int]] = None,
        selected_surface_indices: Optional[Collection[int]] = None,
    ) -> STEPExportResult:
        if isinstance(self_or_cls, CADExporter):
            if file_path is None:
                target = self_or_cls.system_or_stack
                path = system_or_path
            else:
                target = system_or_path
                path = file_path
        else:
            target = self_or_cls
            path = system_or_path
        return STEPExporter.export(
            target,
            path,
            engine=engine,
            selected_element_indices=selected_element_indices,
            selected_surface_indices=selected_surface_indices,
        )

    def export_assembly_step(
        self_or_cls,
        system_or_path: Any,
        file_path: Optional[str] = None,
        engine: Any = "auto",
        selected_element_indices: Optional[Collection[int]] = None,
        selected_surface_indices: Optional[Collection[int]] = None,
    ) -> STEPExportResult:
        return CADExporter.export_to_step(
            self_or_cls,
            system_or_path,
            file_path,
            engine=engine,
            selected_element_indices=selected_element_indices,
            selected_surface_indices=selected_surface_indices,
        )

    def export_step_assembly(
        self_or_cls,
        system_or_path: Any,
        file_path: Optional[str] = None,
        engine: Any = "auto",
        selected_element_indices: Optional[Collection[int]] = None,
        selected_surface_indices: Optional[Collection[int]] = None,
    ) -> STEPExportResult:
        return CADExporter.export_to_step(
            self_or_cls,
            system_or_path,
            file_path,
            engine=engine,
            selected_element_indices=selected_element_indices,
            selected_surface_indices=selected_surface_indices,
        )

    def export_optical_system_step(
        self_or_cls,
        system_or_path: Any,
        file_path: Optional[str] = None,
        engine: Any = "auto",
        selected_element_indices: Optional[Collection[int]] = None,
        selected_surface_indices: Optional[Collection[int]] = None,
    ) -> STEPExportResult:
        return CADExporter.export_to_step(
            self_or_cls,
            system_or_path,
            file_path,
            engine=engine,
            selected_element_indices=selected_element_indices,
            selected_surface_indices=selected_surface_indices,
        )

    def export_to_dxf_2d(
        self_or_cls,
        system_or_path: Any,
        file_path: Optional[str] = None,
        include_dimensions: bool = True,
        include_rays: bool = True,
        include_hatches: bool = True,
        include_table: bool = True,
        folded: bool = True,
        wavelengths: Optional[list] = None,
        plane: str = "DUAL"
    ) -> str:
        if isinstance(self_or_cls, CADExporter):
            if file_path is None:
                target = self_or_cls.system_or_stack
                path = system_or_path
            else:
                target = system_or_path
                path = file_path
        else:
            target = self_or_cls
            path = system_or_path
        try:
            from optics_app.cad.dxf_exporter import DXF2DExporter
        except ImportError:
            from v2.cad.dxf_exporter import DXF2DExporter
        exporter = DXF2DExporter(target, folded=folded)
        return exporter.export(
            file_path=path,
            include_dimensions=include_dimensions,
            include_rays=include_rays,
            include_hatches=include_hatches,
            include_table=include_table,
            wavelengths=wavelengths,
            plane=plane
        )

    def export_to_svg_2d(
        self_or_cls,
        system_or_path: Any,
        file_path: Optional[str] = None,
        include_dimensions: bool = True,
        include_rays: bool = True,
        include_table: bool = True,
        folded: bool = True,
        wavelengths: Optional[list] = None
    ) -> str:
        if isinstance(self_or_cls, CADExporter):
            if file_path is None:
                target = self_or_cls.system_or_stack
                path = system_or_path
            else:
                target = system_or_path
                path = file_path
        else:
            target = self_or_cls
            path = system_or_path
        try:
            from optics_app.cad.svg_exporter import SVG2DExporter
        except ImportError:
            from v2.cad.svg_exporter import SVG2DExporter
        exporter = SVG2DExporter(target, folded=folded)
        return exporter.export(
            file_path=path,
            include_dimensions=include_dimensions,
            include_rays=include_rays,
            include_table=include_table,
            wavelengths=wavelengths
        )

def export_to_step(
    system_or_stack: Any,
    file_path: str,
    engine: Any = "auto",
    selected_element_indices: Optional[Collection[int]] = None,
    selected_surface_indices: Optional[Collection[int]] = None,
) -> STEPExportResult:
    return CADExporter.export_to_step(
        system_or_stack,
        file_path,
        engine=engine,
        selected_element_indices=selected_element_indices,
        selected_surface_indices=selected_surface_indices,
    )

def export_assembly_step(
    system_or_stack: Any,
    file_path: str,
    engine: Any = "auto",
    selected_element_indices: Optional[Collection[int]] = None,
    selected_surface_indices: Optional[Collection[int]] = None,
) -> STEPExportResult:
    return CADExporter.export_assembly_step(
        system_or_stack,
        file_path,
        engine=engine,
        selected_element_indices=selected_element_indices,
        selected_surface_indices=selected_surface_indices,
    )

def export_to_dxf_2d(system_or_stack, file_path, include_dimensions=True, include_rays=True, include_hatches=True, include_table=True, folded=True, wavelengths=None, plane="DUAL"):
    return CADExporter.export_to_dxf_2d(system_or_stack, file_path, include_dimensions=include_dimensions, include_rays=include_rays, include_hatches=include_hatches, include_table=include_table, folded=folded, wavelengths=wavelengths, plane=plane)

def export_to_svg_2d(system_or_stack, file_path, include_dimensions=True, include_rays=True, include_table=True, folded=True, wavelengths=None):
    return CADExporter.export_to_svg_2d(system_or_stack, file_path, include_dimensions=include_dimensions, include_rays=include_rays, include_table=include_table, folded=folded, wavelengths=wavelengths)
