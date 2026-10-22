
import sys, os, json, numpy as np
import build123d as bd
from scipy.spatial.transform import Rotation
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.tracer import OpticalSystem

sys_opt = OpticalSystem.from_json('spectrometer_yz_40mm_100mm_optimized.json')
sys_opt._update_accumulated_positions()

def make_solid_body(s_first, s_last, total_thickness):
    d = max(s_first.semi_diameter, 5.0)
    r1 = s_first.radius
    r2 = s_last.radius
    
    cyl = bd.Solid.make_cylinder(d, total_thickness)
    
    if r1 != 0:
        sph1 = bd.Solid.make_sphere(abs(r1))
        sph1 = sph1.move(bd.Location((0, 0, r1)))
        if r1 > 0: cyl = cyl.intersect(sph1)
        else: cyl = cyl.cut(sph1)
        if hasattr(cyl, 'solids'): cyl = cyl.solids()[0]
            
    if r2 != 0:
        sph2 = bd.Solid.make_sphere(abs(r2))
        sph2 = sph2.move(bd.Location((0, 0, total_thickness + r2)))
        if r2 < 0: cyl = cyl.intersect(sph2)
        else: cyl = cyl.cut(sph2)
        if hasattr(cyl, 'solids'): cyl = cyl.solids()[0]
        
    return cyl

glass_parts = []
air_parts = []

physical_indices = [i for i, s in enumerate(sys_opt.surfaces) if not s.is_coord_break]

i = 0
while i < len(physical_indices) - 1:
    idx_start = physical_indices[i]
    s_start = sys_opt.surfaces[idx_start]
    
    # If it is glass, trace forward until we hit AIR to find the end of the doublet/triplet
    if s_start.glass.upper() not in ['AIR', 'MIRROR'] and s_start.thickness > 0:
        idx_end = idx_start
        total_thickness = 0
        
        while idx_end < len(sys_opt.surfaces) - 1:
            curr = sys_opt.surfaces[idx_end]
            if curr.glass.upper() in ['AIR', 'MIRROR'] or curr.thickness == 0:
                break
            total_thickness += curr.thickness
            idx_end += 1
            
        s_end = sys_opt.surfaces[idx_end]
        
        part = make_solid_body(s_start, s_end, total_thickness)
        pos = s_start.global_origin
        rot = Rotation.from_matrix(s_start.global_R).as_euler('XYZ', degrees=True)
        loc = bd.Location((pos[0], pos[1], pos[2]), (rot[0], rot[1], rot[2]))
        glass_parts.append(part.move(loc))
        
        # Advance 'i' past the cemented elements
        # Find which physical_index corresponds to idx_end
        next_i = i
        while next_i < len(physical_indices) and physical_indices[next_i] < idx_end:
            next_i += 1
        i = next_i
    else:
        i += 1


# Air Gaps (Keep them independent, from physical surface to physical surface)
# Wait, if we treat the doublet as a single element, the air gaps shouldn't change, 
# because air gaps are only between the OUTSIDE of the glass!
for i in range(len(sys_opt.surfaces)-1):
    s1 = sys_opt.surfaces[i]
    s2 = sys_opt.surfaces[i+1]
    
    if s1.thickness > 0 and s1.glass.upper() == 'AIR':
        if 'OBJ' in s1.name: continue
        if s1.thickness > 1000: continue
        
        try:
            part = make_solid_body(s1, s2, s1.thickness)
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
bd.export_step(bd.Compound(children=all_parts), 'C:/Users/Juan/Documents/GitHub/2D_Optics/Final_Assembly_40mm_100mm_Merged_Glass.step')
print('DONE!')

