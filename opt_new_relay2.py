
import sys, os, json, numpy as np
from scipy.optimize import minimize
sys.path.insert(0, os.path.dirname(os.path.abspath('')))
from v2.engine.surface import Surface, SurfaceType
from v2.engine.lde import SurfaceStack
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

with open('C:/Users/Juan/Desktop/Spectrometer_V2_Zemax.json', 'r') as f: data = json.load(f)

for i, s in enumerate(data['surfaces']):
    if 'DMD' in s.get('name', ''):
        dmd_idx = i
        break

data['surfaces'] = data['surfaces'][:dmd_idx+1]

# Add new relay L1 (AC254-200-C)
data['surfaces'].append({'name': 'AC254-200-C_Relay_L1_F', 'surface_type': 'Standard', 'radius': 129.8, 'thickness': 3.6, 'glass': 'N-BK7', 'semi_diameter': 12.7})
data['surfaces'].append({'name': 'AC254-200-C_Relay_L1_M', 'surface_type': 'Standard', 'radius': -93.8, 'thickness': 2.0, 'glass': 'N-SF11', 'semi_diameter': 12.7})
data['surfaces'].append({'name': 'AC254-200-C_Relay_L1_B', 'surface_type': 'Standard', 'radius': -342.3, 'thickness': 230.0, 'glass': 'AIR', 'semi_diameter': 12.7})

# Add new relay L2 (AC254-030-C)
data['surfaces'].append({'name': 'AC254-030-C_Relay_L2_F', 'surface_type': 'Standard', 'radius': 19.9, 'thickness': 9.0, 'glass': 'N-BK7', 'semi_diameter': 12.7})
data['surfaces'].append({'name': 'AC254-030-C_Relay_L2_M', 'surface_type': 'Standard', 'radius': -15.3, 'thickness': 2.0, 'glass': 'N-SF11', 'semi_diameter': 12.7})
data['surfaces'].append({'name': 'AC254-030-C_Relay_L2_B', 'surface_type': 'Standard', 'radius': -55.7, 'thickness': 25.0, 'glass': 'AIR', 'semi_diameter': 12.7})

# Add detector
data['surfaces'].append({'name': 'Image_Detector', 'surface_type': 'Detector', 'radius': 0.0, 'thickness': 0.0, 'glass': 'AIR', 'semi_diameter': 15.0})

def merit(t):
    stack = SurfaceStack()
    dummy = {'name': 'Dummy_DMD_Obj', 'surface_type': 'Object', 'thickness': t[0], 'glass': 'AIR'}
    stack.add_surface(Surface(**dummy))

    for sd in data['surfaces'][dmd_idx+1:]:
        s = Surface(name=sd.get('name', ''))
        for k,v in sd.items():
            if k == 'surface_type': s.surface_type = SurfaceType.DETECTOR if v.lower() == 'detector' else SurfaceType.STANDARD
            elif hasattr(s, k): setattr(s, k, v)
        stack.add_surface(s)
    
    stack.surfaces[3].thickness = t[1]
    stack.surfaces[6].thickness = t[2]
    
    sys_opt = OpticalSystem(stack)

    theta_max = np.arcsin(0.06)
    det_hits_x = []
    det_hits_y = []

    for oy in np.linspace(-0.5, 0.5, 3):
        for ox in np.linspace(-0.015, 0.015, 3):
            for theta in np.linspace(0.001, theta_max, 3):
                for phi in np.linspace(0, 2*np.pi, 4, endpoint=False):
                    dx, dy, dz = np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)
                    r = Ray(pos=np.array([ox, oy, 0.]), dir=np.array([dx, dy, dz]), wavelength=1.225)
                    tr = sys_opt.trace_ray(r)
                    if not tr.vignetted:
                        det_hits_x.append(tr.history[-1]['x_local'])
                        det_hits_y.append(tr.history[-1]['y_local'])

    if len(det_hits_x) == 0: return 999.0
    r_hits = np.sqrt(np.array(det_hits_x)**2 + np.array(det_hits_y)**2)
    return np.max(r_hits)

res = minimize(merit, [200.0, 230.0, 25.0], method='Nelder-Mead')
print('Opt:', res.x, 'Max Radius:', res.fun)

t = res.x
data['surfaces'][dmd_idx]['thickness'] = float(t[0])
data['surfaces'][dmd_idx+3]['thickness'] = float(t[1])
data['surfaces'][dmd_idx+6]['thickness'] = float(t[2])
data['surfaces'][-1]['semi_diameter'] = 0.4 

with open('C:/Users/Juan/Desktop/Spectrometer_V2_Zemax_Opt.json', 'w') as f: json.dump(data, f, indent=4)

