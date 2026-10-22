import numpy as np
import math
from typing import Optional

try:
    from build123d import *
    HAS_BUILD123D = True
except ImportError:
    HAS_BUILD123D = False

from optics_app.engine.lde import SurfaceStack
from optics_app.engine.surface import SurfaceType

def R_mat_to_euler_degrees(R_mat: np.ndarray):
    """Convert a rotation matrix to XYZ Euler angles in degrees."""
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
    return math.degrees(x), math.degrees(y), math.degrees(z)

def create_solid_lens(surf1, surf2):
    """Creates a solid piece of glass bounded by surf1 and surf2 using build123d revolved solid."""
    pts = []
    sd1 = surf1.semi_diameter
    sd2 = surf2.semi_diameter
    sd = max(sd1, sd2)
    
    r_vals_front = np.linspace(0, sd, 50)
    for r in r_vals_front:
        z = surf1.sag(r) if r <= sd1 else surf1.sag(sd1)
        pts.append((r, z))
        
    pts.append((sd, surf1.sag(sd1) if sd > sd1 else surf1.sag(sd)))
    
    z_back_edge = surf1.thickness + (surf2.sag(sd2) if sd > sd2 else surf2.sag(sd))
    pts.append((sd, z_back_edge))
    
    r_vals_back = np.linspace(sd, 0, 50)
    for r in r_vals_back:
        z = surf1.thickness + (surf2.sag(r) if r <= sd2 else surf2.sag(sd2))
        pts.append((r, z))
        
    with BuildPart() as part:
        with BuildSketch(Plane.XZ):
            Polygon(*pts)
        revolve(axis=Axis.Z)
        
    return part.part

def create_flat_element(surf, thickness):
    """Creates a simple cylindrical element for gratings, DMDs, or detectors."""
    with BuildPart() as part:
        # We model it sitting on the XY plane with thickness along Z
        Cylinder(radius=surf.semi_diameter, height=thickness)
    return part.part

def export_system_to_step(stack: SurfaceStack, filename: str) -> bool:
    if not HAS_BUILD123D:
        return False
        
    stack._update_accumulated_positions()
    parts = []
    
    # Simple heuristic to find lenses: surface sequence without thickness jumping to next component
    i = 0
    while i < len(stack.surfaces):
        surf = stack.surfaces[i]
        
        # Identify if this is a solid element (e.g. glass lens)
        is_solid = surf.thickness > 0 and i < len(stack.surfaces) - 1 and (surf.glass.upper() != "AIR")
        if is_solid:
            # It's a lens element
            next_surf = stack.surfaces[i+1]
            part_3d = create_solid_lens(surf, next_surf)
            
            loc = tuple(surf.global_origin)
            rot_deg = R_mat_to_euler_degrees(surf.global_R)
            placed_part = part_3d.locate(Location(loc, rot_deg))
            parts.append(placed_part)
            # if the next surface also has a material, it might be a cemented doublet
            # we iterate normally to catch it (e.g. L1 middle to L1 back)
            
        elif surf.surface_type in [SurfaceType.GRATING, SurfaceType.DETECTOR] or "DMD" in surf.name:
            part_3d = create_flat_element(surf, 2.0)
            loc = tuple(surf.global_origin)
            rot_deg = R_mat_to_euler_degrees(surf.global_R)
            placed_part = part_3d.locate(Location(loc, rot_deg))
            parts.append(placed_part)
            
        i += 1

    if parts:
        export_step(Compound(children=parts), filename)
        return True
    
    return False
