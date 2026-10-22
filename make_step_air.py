
import sys, os, json, numpy as np
import build123d as bd
from scipy.spatial.transform import Rotation
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.tracer import OpticalSystem

sys_opt = OpticalSystem.from_json('spectrometer_yz_40mm_100mm_optimized.json')
sys_opt._update_accumulated_positions()

def make_solid_body(s1, s2, is_air=False):
    d = max(s1.semi_diameter, 5.0)
    # If it's an air gap, we might want to make it slightly smaller diameter or keep it same
    # For now, keep it same
    r1 = s1.radius
    r2 = s2.radius
    t = s1.thickness
    
    cyl = bd.Solid.make_cylinder(d, t)
    
    # Bottom surface (s1)
    if r1 != 0:
        sph1 = bd.Solid.make_sphere(abs(r1))
        sph1 = sph1.move(bd.Location((0, 0, r1)))
        if r1 > 0: cyl = cyl.intersect(sph1)
        else: cyl = cyl.cut(sph1)
        if hasattr(cyl, 'solids'): cyl = cyl.solids()[0]
            
    # Top surface (s2)
    # If s2 is a coordinate break, it has radius 0, so it's flat.
    if r2 != 0:
        sph2 = bd.Solid.make_sphere(abs(r2))
        sph2 = sph2.move(bd.Location((0, 0, t + r2)))
        if r2 < 0: cyl = cyl.intersect(sph2)
        else: cyl = cyl.cut(sph2)
        if hasattr(cyl, 'solids'): cyl = cyl.solids()[0]
        
    return cyl

glass_parts = []
air_parts = []

physical_indices = [i for i, s in enumerate(sys_opt.surfaces) if not s.is_coord_break]
for i in range(len(physical_indices)-1):
    idx1, idx2 = physical_indices[i], physical_indices[i+1]
    s1, s2 = sys_opt.surfaces[idx1], sys_opt.surfaces[idx2]
    
    # Glass Lenses
    if s1.glass.upper() not in ['AIR', 'MIRROR'] and s1.thickness > 0:
        part = make_solid_body(s1, s2, is_air=False)
        pos = s1.global_origin
        rot = Rotation.from_matrix(s1.global_R).as_euler('XYZ', degrees=True)
        loc = bd.Location((pos[0], pos[1], pos[2]), (rot[0], rot[1], rot[2]))
        glass_parts.append(part.move(loc))

# Air Gaps
for i in range(len(sys_opt.surfaces)-1):
    s1 = sys_opt.surfaces[i]
    s2 = sys_opt.surfaces[i+1]
    
    if s1.thickness > 0 and s1.glass.upper() == 'AIR':
        # Skip the huge initial air gap before the slit
        if 'OBJ' in s1.name: continue
        
        # Don't try to build the 1e6 air gaps
        if s1.thickness > 1000: continue
        
        try:
            part = make_solid_body(s1, s2, is_air=True)
            pos = s1.global_origin
            rot = Rotation.from_matrix(s1.global_R).as_euler('XYZ', degrees=True)
            loc = bd.Location((pos[0], pos[1], pos[2]), (rot[0], rot[1], rot[2]))
            air_parts.append(part.move(loc))
        except Exception as e:
            print(f'Failed to make air body for {s1.name}: {e}')

for s in sys_opt.surfaces:
    if 'Slit' in s.name or 'Grating' in s.name or 'DMD' in s.name or 'IMA' in s.name:
        w = 10.0 if 'IMA' not in s.name else 2.0
        box = bd.Solid.make_box(w, w, 1.0)
        box = box.move(bd.Location((-w/2, -w/2, -0.5)))
        pos = s.global_origin
        rot = Rotation.from_matrix(s.global_R).as_euler('XYZ', degrees=True)
        loc = bd.Location((pos[0], pos[1], pos[2]), (rot[0], rot[1], rot[2]))
        glass_parts.append(box.move(loc))

all_parts = glass_parts + air_parts
bd.export_step(bd.Compound(children=all_parts), 'C:/Users/Juan/Documents/GitHub/2D_Optics/Final_Assembly_40mm_100mm_With_Air.step')
print('DONE!')

