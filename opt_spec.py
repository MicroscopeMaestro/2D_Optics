
import sys, os, json, copy, numpy as np
from scipy.optimize import minimize
sys.path.insert(0, os.path.dirname(os.path.abspath('')))
from v2.engine.surface import Surface, SurfaceType
from v2.engine.lde import SurfaceStack
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

with open('C:/Users/Juan/Desktop/Spectrometer_V2_Full_System.json', 'r') as f: data = json.load(f)

def build_sys(t_gap, t_align):
    stack = SurfaceStack()
    for sd in data['surfaces']:
        s = Surface(name=sd.get('name', ''))
        for k,v in sd.items():
            if k == 'surface_type': s.surface_type = SurfaceType.GRATING if v.lower() == 'grating' else (SurfaceType.ASPHERIC if v.lower() == 'aspheric' else SurfaceType.STANDARD)
            elif hasattr(s, k): setattr(s, k, v)
        if 'Field_B' in s.name: s.thickness = t_gap[0]
        if 'Cond_Asphere_B' in s.name: s.thickness = t_gap[1]
        if 'Cond_Asphere_F' in s.name: 
            s.radius = t_gap[2]
            s.conic = t_gap[3]
        if 'DMD4500' in s.name: s.tilt_x = t_align[0]
        if 'Field' in s.name or 'Cond' in s.name or 'Single' in s.name:
            s.decenter_y = t_align[1]
            s.tilt_x = t_align[2]
        s.aperture_type = 'Circular'
        s.semi_diameter = float('inf')
        stack.add_surface(s)
    sys_opt = OpticalSystem(stack)
    sys_opt.stack._update_accumulated_positions()
    return sys_opt

res_align = minimize(lambda x: 0, [30.19, 0.0, 0.0], method='Nelder-Mead')

def merit_gap(t_gap):
    sys_opt = build_sys(t_gap, res_align.x)
    rays = []
    # Start rays AT THE DMD instead of from Object to avoid Fore-optics TIR during extreme guesses
    # But wait, starting at DMD is hard because of the Z-fold angles!
    # Instead, start at the SLIT (Surface 4)!
    # But Ray() pos/dir is global!
    pass

sys_opt = build_sys([216.89, 57.62, 28.02, -0.80], [30.19, 0.0, 0.0])
r = Ray(pos=np.array([0.,0.,0.]), dir=np.array([0.,0.,1.]), wavelength=1.225)
tr = sys_opt.trace_ray(r)
for h in tr.history:
    if h['surface_index'] >= 10:
        print(h['surface_index'], sys_opt.surfaces[h['surface_index']].name, 'y=', h['y_local'])

