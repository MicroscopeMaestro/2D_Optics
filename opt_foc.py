
import sys, os, json, numpy as np
from scipy.optimize import minimize
sys.path.insert(0, os.path.dirname(os.path.abspath('')))
from v2.engine.surface import Surface, SurfaceType
from v2.engine.lde import SurfaceStack
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

with open('C:/Users/Juan/Desktop/Spectrometer_V2_Zemax.json', 'r') as f: data = json.load(f)

for i, s in enumerate(data['surfaces']):
    if 'Grating' in s.get('name', ''):
        grat_idx = i
    if 'Foc_S1' in s.get('name', ''):
        foc_idx = i
    if 'DMD' in s.get('name', ''):
        dmd_idx = i
        break

def build_sys(t):
    # t = [foc_distance, foc_tilt, dmd_distance]
    stack = SurfaceStack()
    for sd in data['surfaces'][:dmd_idx+1]:
        s = Surface(name=sd.get('name', ''))
        for k,v in sd.items():
            if k == 'surface_type': s.surface_type = SurfaceType.GRATING if v.lower() == 'grating' else (SurfaceType.STOP if v.lower() == 'stop' else (SurfaceType.OBJECT if v.lower() == 'object' else SurfaceType.STANDARD))
            elif hasattr(s, k): setattr(s, k, v)
        stack.add_surface(s)
    
    stack.surfaces[grat_idx].thickness = t[0]
    stack.surfaces[foc_idx].tilt_x = t[1]
    stack.surfaces[foc_idx+2].thickness = t[2]
    
    return OpticalSystem(stack)

def merit(t):
    sys_opt = build_sys(t)
    
    # We want to minimize the spot size on the DMD!
    # Trace rays from the center of the slit (ox=0, oy=0)
    theta_max = np.arcsin(0.06)
    hits_x = []
    hits_y = []

    for theta in np.linspace(0.001, theta_max, 5):
        for phi in np.linspace(0, 2*np.pi, 8, endpoint=False):
            dx, dy, dz = np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)
            r = Ray(pos=np.array([0., 0., 0.]), dir=np.array([dx, dy, dz]), wavelength=1.225)
            tr = sys_opt.trace_ray(r)
            if not tr.vignetted:
                h = tr.history[-1]
                hits_x.append(h['x_local'])
                hits_y.append(h['y_local'])
    
    if len(hits_x) == 0: return 999.0
    # Calculate RMS spot size
    mx = np.mean(hits_x)
    my = np.mean(hits_y)
    r_sq = (np.array(hits_x) - mx)**2 + (np.array(hits_y) - my)**2
    rms = np.sqrt(np.mean(r_sq))
    
    # Add a penalty if the chief ray (theta=0.0001) hits far from center of DMD
    # We want the chief ray to hit near (0,0) on the DMD
    r = Ray(pos=np.array([0., 0., 0.]), dir=np.array([0., 0., 1.]), wavelength=1.225)
    tr = sys_opt.trace_ray(r)
    if tr.vignetted: return 999.0
    cy = tr.history[-1]['y_local']
    
    return rms + abs(cy)*0.01

# Initial guess: current values
sys_opt0 = build_sys([50.0, data['surfaces'][foc_idx]['tilt_x'], 46.8])
t0 = [50.0, data['surfaces'][foc_idx]['tilt_x'], 46.8]
print('Initial merit:', merit(t0))

res = minimize(merit, t0, method='Nelder-Mead')
print('Opt:', res.x, 'Merit:', res.fun)

