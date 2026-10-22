
import sys, os, json, copy, numpy as np
from scipy.optimize import minimize
sys.path.insert(0, os.path.dirname(os.path.abspath('')))
from v2.engine.surface import Surface, SurfaceType
from v2.engine.lde import SurfaceStack
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

with open('C:/Users/Juan/Desktop/Spectrometer_V2_Optimized.json', 'r') as f: data = json.load(f)

# Need 4 lenses (Field1, Relay1, Field2, Condenser). Each doublet is 3 surfaces, Condenser is 2 surfaces.
# Field1 (AC254-050): 7,8,9
# Relay1 (AC254-030): 10,11,12
# Field2 (AC254-050): 13,14,15
# Condenser (ACL25416U): 16,17
# Detector: 18

while len(data['surfaces']) < 19:
    data['surfaces'].append(copy.deepcopy(data['surfaces'][-1]))
    
# Field1: AC254-050-C
data['surfaces'][7].update({'name': 'Field1_F', 'radius': 33.3, 'thickness': 6.5, 'glass': 'N-BAK4', 'surface_type': 'Standard'})
data['surfaces'][8].update({'name': 'Field1_M', 'radius': -22.3, 'thickness': 2.0, 'glass': 'N-SF5', 'surface_type': 'Standard'})
data['surfaces'][9].update({'name': 'Field1_B', 'radius': -83.0, 'thickness': 50.0, 'glass': 'AIR', 'surface_type': 'Standard'})

# Relay1: AC254-030-C
data['surfaces'][10].update({'name': 'Relay1_F', 'radius': 19.9, 'thickness': 12.0, 'glass': 'N-BAF10', 'surface_type': 'Standard'})
data['surfaces'][11].update({'name': 'Relay1_M', 'radius': -15.3, 'thickness': 3.0, 'glass': 'N-SF6A', 'surface_type': 'Standard'})
data['surfaces'][12].update({'name': 'Relay1_B', 'radius': -60.0, 'thickness': 50.0, 'glass': 'AIR', 'surface_type': 'Standard'})

# Field2: AC254-050-C
data['surfaces'][13].update({'name': 'Field2_F', 'radius': 33.3, 'thickness': 6.5, 'glass': 'N-BAK4', 'surface_type': 'Standard'})
data['surfaces'][14].update({'name': 'Field2_M', 'radius': -22.3, 'thickness': 2.0, 'glass': 'N-SF5', 'surface_type': 'Standard'})
data['surfaces'][15].update({'name': 'Field2_B', 'radius': -83.0, 'thickness': 50.0, 'glass': 'AIR', 'surface_type': 'Standard'})

# Condenser: ACL25416U
data['surfaces'][16].update({'name': 'Cond_Asphere_F', 'radius': 12.0, 'thickness': 12.0, 'glass': 'B270', 'surface_type': 'Aspheric', 'conic': -1.0})
data['surfaces'][17].update({'name': 'Cond_Asphere_B', 'radius': 0.0, 'thickness': 10.0, 'glass': 'AIR', 'surface_type': 'Standard'})
if 'conic' in data['surfaces'][17]: del data['surfaces'][17]['conic']

data['surfaces'][18].update({'name': 'Single_Detector_0.8mm', 'surface_type': 'Detector'})

for i in range(7, 18):
    data['surfaces'][i]['semi_diameter'] = 12.7

def build_sys(t_gap, t_align):
    stack = SurfaceStack()
    for sd in data['surfaces']:
        s = Surface(name=sd.get('name', ''))
        for k,v in sd.items():
            if k == 'surface_type': s.surface_type = SurfaceType.GRATING if v.lower() == 'grating' else (SurfaceType.ASPHERIC if v.lower() == 'aspheric' else SurfaceType.STANDARD)
            elif hasattr(s, k): setattr(s, k, v)
        if 'Field1_B' in s.name: s.thickness = t_gap[0]
        if 'Relay1_B' in s.name: s.thickness = t_gap[1]
        if 'Field2_B' in s.name: s.thickness = t_gap[2]
        if 'Cond_Asphere_B' in s.name: s.thickness = t_gap[3]
        if 'Cond_Asphere_F' in s.name: s.conic = t_gap[4]
        
        if 'DMD4500' in s.name: s.tilt_x = t_align[0]
        if s.name.startswith('Field') or s.name.startswith('Relay') or s.name.startswith('Cond') or s.name.startswith('Single'):
            s.decenter_y = t_align[1]
            s.tilt_x = t_align[2]
        s.semi_diameter = float('inf')
        stack.add_surface(s)
    sys_opt = OpticalSystem(stack)
    sys_opt.stack._update_accumulated_positions()
    return sys_opt

def merit_align(t_align):
    sys_opt = build_sys([60, 60, 60, 10, -1.0], t_align)
    tr = sys_opt.trace_ray(Ray(pos=np.array([0.,0.,1e-3]), dir=np.array([0.,0.,1.]), wavelength=1.225))
    if tr.vignetted: return 9999.0
    s_f = sys_opt.surfaces[7]
    loc_f = s_f.global_R.T @ (tr.pos - s_f.global_origin)
    s_l = sys_opt.surfaces[-1]
    loc_l = s_l.global_R.T @ (tr.pos - s_l.global_origin)
    return loc_f[1]**2 + loc_l[1]**2

res_align = minimize(merit_align, [30.0, 0.0, 0.0], method='Nelder-Mead')
print('Align Opt:', res_align.x)

def merit_gap(t_gap):
    sys_opt = build_sys(t_gap, res_align.x)
    rays = []
    theta = np.arcsin(0.12)
    for wl in [1.111, 1.225, 1.389]:
        for phi in [0, np.pi/2, np.pi, 3*np.pi/2]:
            dx, dy, dz = np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)
            rays.append(Ray(pos=np.array([0.,0.,1e-3]), dir=np.array([dx,dy,dz]), wavelength=wl))
    pts = []
    for r in rays:
        tr2 = sys_opt.trace_ray(r)
        if tr2.vignetted: return 999.0
        
        # Check hard apertures (1-inch = 12.7 mm radius)
        for i in [7, 10, 13, 16]:
            h = [x for x in tr2.history if x['surface_index'] == i]
            if h and np.sqrt(h[0]['x_local']**2 + h[0]['y_local']**2) > 12.7: return 500.0
            
        s = sys_opt.surfaces[-1]
        loc = s.global_R.T @ (tr2.pos - s.global_origin)
        pts.append([loc[0], loc[1]])
    pts = np.array(pts)
    return np.sqrt(np.mean(pts[:,0]**2 + pts[:,1]**2))

res_gap = minimize(merit_gap, [60, 60, 60, 10, -1.0], method='Nelder-Mead', options={'maxiter': 2000})
print('Gap Opt:', res_gap.x, 'RMS:', res_gap.fun)

for s in data['surfaces']:
    name = s.get('name', '')
    if 'Field1_B' in name: s['thickness'] = float(res_gap.x[0])
    if 'Relay1_B' in name: s['thickness'] = float(res_gap.x[1])
    if 'Field2_B' in name: s['thickness'] = float(res_gap.x[2])
    if 'Cond_Asphere_B' in name: s['thickness'] = float(res_gap.x[3])
    if 'Cond_Asphere_F' in name: s['conic'] = float(res_gap.x[4])
    if 'DMD4500' in name: s['tilt_x'] = float(res_align.x[0])
    if name.startswith('Field') or name.startswith('Relay') or name.startswith('Cond') or name.startswith('Single'):
        s['decenter_y'] = float(res_align.x[1])
        s['tilt_x'] = float(res_align.x[2])

with open('C:/Users/Juan/Desktop/Spectrometer_V2_4Lens_Optimized.json', 'w') as f: json.dump(data, f, indent=4)
print('Saved!')

