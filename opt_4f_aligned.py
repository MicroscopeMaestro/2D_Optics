
import sys, os, json, numpy as np
from scipy.optimize import minimize
sys.path.insert(0, os.path.dirname(os.path.abspath('')))
from v2.engine.surface import Surface, SurfaceType
from v2.engine.lde import SurfaceStack
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray
from scipy.spatial.transform import Rotation

with open('C:/Users/Juan/Desktop/Spectrometer_V2_Optimized.json', 'r') as f: data = json.load(f)

# AC254-100-C
data['surfaces'][7].update({'name': 'Relay_L1_F', 'radius': 64.9, 'thickness': 5.0, 'glass': 'N-BK7', 'semi_diameter': 12.7})
data['surfaces'][8].update({'name': 'Relay_L1_M', 'radius': -47.1, 'thickness': 2.0, 'glass': 'N-SF5', 'semi_diameter': 12.7})
data['surfaces'][9].update({'name': 'Relay_L1_B', 'radius': -147.2, 'thickness': 116.0, 'glass': 'AIR', 'semi_diameter': 12.7})

# ACL25416U
data['surfaces'][10].update({'name': 'Relay_L2_F', 'radius': 12.0, 'thickness': 12.0, 'glass': 'B270', 'surface_type': 'Aspheric', 'conic': -1.0, 'semi_diameter': 12.7})
data['surfaces'][11].update({'name': 'Relay_L2_B', 'radius': 0.0, 'thickness': 16.0, 'glass': 'AIR', 'surface_type': 'Standard', 'semi_diameter': 12.7})
if 'conic' in data['surfaces'][11]: del data['surfaces'][11]['conic']

data['surfaces'] = data['surfaces'][:13]
data['surfaces'][12].update({'name': 'Single_Detector_0.8mm', 'surface_type': 'Detector'})

def build_sys(t):
    stack = SurfaceStack()
    for sd in data['surfaces']:
        s = Surface(name=sd.get('name', ''))
        for k,v in sd.items():
            if k == 'surface_type': s.surface_type = SurfaceType.GRATING if v.lower() == 'grating' else (SurfaceType.ASPHERIC if v.lower() == 'aspheric' else SurfaceType.STANDARD)
            elif hasattr(s, k): setattr(s, k, v)
        if 'DMD' in s.name: 
            s.thickness = t[0]
            s.tilt_x = t[2]
        if 'Relay_L2_B' in s.name: s.thickness = t[1]
        if 'Relay' in s.name or 'Single' in s.name:
            s.decenter_y = t[3]
            s.tilt_x = t[4]
        s.aperture_type = 'Circular'
        s.semi_diameter = float('inf')
        stack.add_surface(s)
    sys_opt = OpticalSystem(stack)
    sys_opt.stack._update_accumulated_positions()
    return sys_opt

def merit_align(t_align):
    sys_opt = build_sys([94.0, 16.0, t_align[0], t_align[1], t_align[2]])
    r_chief = Ray(pos=np.array([0.,0.,0.]), dir=np.array([0.,0.,1.]), wavelength=1.225)
    tr = sys_opt.trace_ray(r_chief)
    if tr.vignetted: return 9999.0
    loc_l1 = sys_opt.surfaces[7].global_R.T @ (tr.pos - sys_opt.surfaces[7].global_origin)
    loc_det = sys_opt.surfaces[-1].global_R.T @ (tr.pos - sys_opt.surfaces[-1].global_origin)
    return loc_l1[1]**2 + loc_det[1]**2

res_align = minimize(merit_align, [30.19, 0.0, 0.0], method='Nelder-Mead')
print('Align Opt:', res_align.x)

def merit_focus(t_foc):
    sys_opt = build_sys([t_foc[0], t_foc[1], res_align.x[0], res_align.x[1], res_align.x[2]])
    rays = []
    theta = np.arcsin(0.12)
    for phi in [0, np.pi/2, np.pi, 3*np.pi/2]:
        dx, dy, dz = np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)
        rays.append(Ray(pos=np.array([0.,0.,0.]), dir=np.array([dx,dy,dz]), wavelength=1.225))
    pts = []
    for r in rays:
        tr = sys_opt.trace_ray(r)
        if tr.vignetted: return 9999.0
        loc = sys_opt.surfaces[-1].global_R.T @ (tr.pos - sys_opt.surfaces[-1].global_origin)
        pts.append([loc[0], loc[1]])
    pts = np.array(pts)
    spot = np.sqrt(np.mean(pts[:,0]**2 + pts[:,1]**2))
    bfl_pen = (t_foc[0] - 94.0)**2 * 0.1
    return spot + bfl_pen

res_foc = minimize(merit_focus, [94.0, 16.0], method='Nelder-Mead')
print('Focus Opt:', res_foc.x, 'RMS:', res_foc.fun)

data['surfaces'][6]['thickness'] = float(res_foc.x[0])
data['surfaces'][6]['tilt_x'] = float(res_align.x[0])
data['surfaces'][11]['thickness'] = float(res_foc.x[1])
for i in range(7, 13):
    data['surfaces'][i]['decenter_y'] = float(res_align.x[1])
    data['surfaces'][i]['tilt_x'] = float(res_align.x[2])
    data['surfaces'][i]['semi_diameter'] = 12.7

with open('C:/Users/Juan/Desktop/Spectrometer_V2_4f_1inch_Aligned.json', 'w') as f: json.dump(data, f, indent=4)
print('Saved!')

