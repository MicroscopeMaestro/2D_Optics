import sys, os, json, numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from optics_app.engine.surface import Surface
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray

with open('C:/Users/Juan/Desktop/Spectrometer_V2_Optimized.json', 'r') as f:
    data = json.load(f)

for s in data['surfaces']:
    if 'Diffraction_Grating' in s['name']:
        s['grating_lines_per_mm'] = 300.0
        s['name'] = 'Diffraction_Grating_300lpm'
        
def get_dmd_hits(tilt_x):
    stack = SurfaceStack()
    for sd in data['surfaces'][:8]:
        s = Surface(name=sd.get('name', ''))
        for k,v in sd.items():
            if hasattr(s, k): setattr(s, k, v)
        if s.name == 'Diffraction_Grating_300lpm':
            s.tilt_x = tilt_x
        s.semi_diameter = float('inf')
        stack.add_surface(s)
        
    sys_opt = OpticalSystem(stack)
    sys_opt.stack._update_accumulated_positions()
    
    hits = []
    for wl in [1.111, 1.225, 1.389]:
        curr = Ray(pos=np.array([0.,0.,1e-3]), dir=np.array([0.,0.,1.]), wavelength=wl)
        tr = sys_opt.trace_ray(curr)
        R_mat = sys_opt.surfaces[-1].global_R
        orig = sys_opt.surfaces[-1].global_origin
        loc = R_mat.T @ (tr.pos - orig)
        hits.append(loc[1])
    return hits

def merit(tilt_x):
    hits = get_dmd_hits(tilt_x[0])
    # We want to center 1.250 at 0. But let's just center the average of the min and max
    return (hits[0] + hits[2])**2

res = minimize(merit, [-12.0], method='Nelder-Mead')
print(f"Optimal Grating Tilt: {res.x[0]}")
hits = get_dmd_hits(res.x[0])
print(f"DMD Hits: 1.111={hits[0]:.2f}, 1.225={hits[1]:.2f}, 1.389={hits[2]:.2f}")
