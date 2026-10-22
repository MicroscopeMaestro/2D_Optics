
import sys, os, json, numpy as np
import build123d as bd
from scipy.spatial.transform import Rotation
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.tracer import OpticalSystem

sys_opt = OpticalSystem.from_json('spectrometer_yz_40mm_100mm_optimized.json')
sys_opt._update_accumulated_positions()

def make_lens_body(s1, s2):
    d = s1.semi_diameter
    r1 = s1.radius
    r2 = s2.radius
    t = s1.thickness
    
    with bd.BuildPart() as p:
        bd.Cylinder(radius=d, height=t, align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN))
        
        if r1 != 0:
            sphere1 = bd.Sphere(radius=abs(r1))
            sphere1 = sphere1.locate(bd.Location((0, 0, r1)))
            if r1 > 0: p.part = p.part.intersect(sphere1)
            else: p.part = p.part.cut(sphere1)
                
        if r2 != 0:
            sphere2 = bd.Sphere(radius=abs(r2))
            sphere2 = sphere2.locate(bd.Location((0, 0, t + r2)))
            if r2 < 0: p.part = p.part.intersect(sphere2)
            else: p.part = p.part.cut(sphere2)
    return p.part

assembly = None
physical_indices = [i for i, s in enumerate(sys_opt.surfaces) if not s.is_coord_break]

for i in range(len(physical_indices)-1):
    idx1, idx2 = physical_indices[i], physical_indices[i+1]
    s1, s2 = sys_opt.surfaces[idx1], sys_opt.surfaces[idx2]
    if s1.glass.upper() not in ['AIR', 'MIRROR'] and s1.thickness > 0:
        part = make_lens_body(s1, s2)
        pos = s1.global_origin
        rot = Rotation.from_matrix(s1.global_R).as_euler('XYZ', degrees=True)
        loc = bd.Location((pos[0], pos[1], pos[2]), (rot[0], rot[1], rot[2]))
        part = part.locate(loc)
        
        if assembly is None:
            assembly = part
        else:
            assembly = assembly.fuse(part)

for s in sys_opt.surfaces:
    if 'Slit' in s.name or 'Grating' in s.name or 'DMD' in s.name or 'IMA' in s.name:
        with bd.BuildPart() as p:
            w = 10.0 if 'IMA' not in s.name else 2.0
            bd.Box(length=w, width=w, height=1.0, align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.CENTER))
        pos = s.global_origin
        rot = Rotation.from_matrix(s.global_R).as_euler('XYZ', degrees=True)
        loc = bd.Location((pos[0], pos[1], pos[2]), (rot[0], rot[1], rot[2]))
        part = p.part.locate(loc)
        if assembly is None:
            assembly = part
        else:
            assembly = assembly.fuse(part)

assembly.export_step('C:/Users/Juan/.gemini/antigravity/brain/49fd158d-b0e1-421c-b15b-b237b2e3ad7f/Final_Assembly_40mm_100mm.step')
print('DONE!')

