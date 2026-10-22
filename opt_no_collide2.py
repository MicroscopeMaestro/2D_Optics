
import sys, os, json, numpy as np
from scipy.optimize import minimize
sys.path.insert(0, os.path.dirname(os.path.abspath('')))
from v2.engine.surface import Surface, SurfaceType
from v2.engine.lde import SurfaceStack
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

with open('C:/Users/Juan/Desktop/Spectrometer_V2_Complete.json', 'r') as f: data = json.load(f)

# Extreme spread to ensure absolutely zero collisions anywhere
data['surfaces'][8]['thickness'] = 150.0
data['surfaces'][8]['tilt_x'] = 25.0

def build_sys(t):
    stack = SurfaceStack()
    for sd in data['surfaces']:
        s = Surface(name=sd.get('name', ''))
        for k,v in sd.items():
            if k == 'surface_type': s.surface_type = SurfaceType.GRATING if v.lower() == 'grating' else (SurfaceType.ASPHERIC if v.lower() == 'aspheric' else SurfaceType.STANDARD)
            elif hasattr(s, k): setattr(s, k, v)
        if 'Diffraction_Grating' in s.name: s.tilt_x = t[0]
        if 'DMD' in s.name: s.tilt_x = t[1]
        if 'Relay' in s.name or 'Single' in s.name:
            s.decenter_y = t[2]
            s.tilt_x = t[3]
        s.aperture_type = 'Circular'
        s.semi_diameter = float('inf')
        stack.add_surface(s)
    sys_opt = OpticalSystem(stack)
    sys_opt.stack._update_accumulated_positions()
    return sys_opt

def merit(t):
    sys_opt = build_sys(t)
    r_chief = Ray(pos=np.array([0.,0.,0.]), dir=np.array([0.,0.,1.]), wavelength=1.225)
    tr = sys_opt.trace_ray(r_chief)
    if tr.vignetted: return 9999.0
    y_foc, y_dmd = 999.0, 999.0
    for h in tr.history:
        if h['surface_index'] == 10: y_foc = h['y_local']
        if h['surface_index'] == 13: y_dmd = h['y_local']
    s_l1 = sys_opt.surfaces[14]
    loc_l1 = s_l1.global_R.T @ (tr.pos - s_l1.global_origin)
    s_det = sys_opt.surfaces[-1]
    loc_det = s_det.global_R.T @ (tr.pos - s_det.global_origin)
    return y_foc**2 + y_dmd**2 + loc_l1[1]**2 + loc_det[1]**2

res = minimize(merit, [-21.5, 29.4, 0.0, 0.0], method='Nelder-Mead')
print('Align Opt:', res.x, 'RMS:', res.fun)

t = res.x
data['surfaces'][9]['tilt_x'] = float(t[0])
data['surfaces'][13]['tilt_x'] = float(t[1])
for i in range(14, 20):
    data['surfaces'][i]['decenter_y'] = float(t[2])
    data['surfaces'][i]['tilt_x'] = float(t[3])

with open('C:/Users/Juan/Desktop/Spectrometer_V2_Complete_NoCollide.json', 'w') as f: json.dump(data, f, indent=4)
print('Saved!')

