
import sys, os, json, numpy as np
import build123d as bd
from scipy.spatial.transform import Rotation
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.tracer import OpticalSystem

sys_opt = OpticalSystem.from_json('spectrometer_yz_40mm_100mm_optimized.json')
sys_opt._update_accumulated_positions()

parts = []
def make_lens_body(s1, s2):
    d = s1.semi_diameter
    r1 = s1.radius
    r2 = s2.radius
    t = s1.thickness
    
    cyl = bd.Solid.make_cylinder(d, t)
    
    if r1 != 0:
        sph1 = bd.Solid.make_sphere(abs(r1))
        sph1 = sph1.move(bd.Location((0, 0, r1)))
        if r1 > 0: cyl = cyl.intersect(sph1)
        else: cyl = cyl.cut(sph1)
        if hasattr(cyl, 'solids'): cyl = cyl.solids()[0]
            
    if r2 != 0:
        sph2 = bd.Solid.make_sphere(abs(r2))
        sph2 = sph2.move(bd.Location((0, 0, t + r2)))
        if r2 < 0: cyl = cyl.intersect(sph2)
        else: cyl = cyl.cut(sph2)
        if hasattr(cyl, 'solids'): cyl = cyl.solids()[0]
        
    return cyl

physical_indices = [i for i, s in enumerate(sys_opt.surfaces) if not s.is_coord_break]
for i in range(len(physical_indices)-1):
    idx1, idx2 = physical_indices[i], physical_indices[i+1]
    s1, s2 = sys_opt.surfaces[idx1], sys_opt.surfaces[idx2]
    if s1.glass.upper() not in ['AIR', 'MIRROR'] and s1.thickness > 0:
        part = make_lens_body(s1, s2)
        pos = s1.global_origin
        rot = Rotation.from_matrix(s1.global_R).as_euler('XYZ', degrees=True)
        loc = bd.Location((pos[0], pos[1], pos[2]), (rot[0], rot[1], rot[2]))
        parts.append(part.move(loc))

for s in sys_opt.surfaces:
    if 'Slit' in s.name or 'Grating' in s.name or 'DMD' in s.name or 'IMA' in s.name:
        w = 10.0 if 'IMA' not in s.name else 2.0
        box = bd.Solid.make_box(w, w, 1.0)
        box = box.move(bd.Location((-w/2, -w/2, -0.5)))
        pos = s.global_origin
        rot = Rotation.from_matrix(s.global_R).as_euler('XYZ', degrees=True)
        loc = bd.Location((pos[0], pos[1], pos[2]), (rot[0], rot[1], rot[2]))
        parts.append(box.move(loc))

bd.export_step(bd.Compound(children=parts), 'C:/Users/Juan/.gemini/antigravity/brain/49fd158d-b0e1-421c-b15b-b237b2e3ad7f/Final_Assembly_40mm_100mm.step')
print('DONE!')

