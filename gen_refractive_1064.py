
import sys, os, json, numpy as np
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.surface import Surface, SurfaceType
from v2.engine.lde import SurfaceStack
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

with open('spectrometer_refractive_1064.json', 'r') as f:
    data = json.load(f)

def build_opt(t_coll, t_foc, t_rel, t_cond):
    data['surfaces'][0]['thickness'] = t_coll
    data['surfaces'][7]['thickness'] = t_foc
    data['surfaces'][8]['thickness'] = t_rel
    data['surfaces'][12]['thickness'] = t_cond
    
    stack = SurfaceStack()
    for sd in data['surfaces']:
        s = Surface(name=sd.get('name', ''))
        for k,v in sd.items():
            if k == 'surface_type': s.surface_type = SurfaceType.GRATING if v.lower() == 'grating' else (SurfaceType.STOP if v.lower() == 'stop' else (SurfaceType.OBJECT if v.lower() == 'object' else (SurfaceType.DETECTOR if v.lower() == 'detector' else SurfaceType.STANDARD)))
            elif hasattr(s, k): setattr(s, k, v)
        if 'Aspheric' in s.name or s.surface_type == SurfaceType.ASPHERIC:
            s.is_aspheric = True
            s.conic = sd.get('conic', 0.0)
        stack.add_surface(s)
    sys_opt = OpticalSystem(stack)
    sys_opt._update_accumulated_positions()
    return sys_opt

def evaluate(sys_opt):
    r = Ray(pos=np.array([0., 0., 0.]), dir=np.array([0., 0., 1.]), wavelength=1.225)
    tr = sys_opt.trace_ray(r)
    if tr.vignetted: return False, 999
    
    hits_x, hits_y = [], []
    for wl in [1.087, 1.225, 1.350]:
        for theta in np.linspace(0.001, np.arcsin(0.04), 3):
            for phi in np.linspace(0, 2*np.pi, 4, endpoint=False):
                dx, dy, dz = np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)
                r = Ray(pos=np.array([0., 0., 0.]), dir=np.array([dx, dy, dz]), wavelength=wl)
                tr = sys_opt.trace_ray(r)
                if not tr.vignetted:
                    h = tr.history[-1]
                    hits_x.append(h['x_local'])
                    hits_y.append(h['y_local'])
    if len(hits_x) == 0: return False, 999
    
    mx, my = np.mean(hits_x), np.mean(hits_y)
    r_sq = (np.array(hits_x) - mx)**2 + (np.array(hits_y) - my)**2
    max_r = np.max(np.sqrt(r_sq))
    return True, max_r

best_r = 999
best_params = None
for t_coll in np.linspace(20, 30, 5):
    for t_foc in np.linspace(20, 30, 5):
        for t_rel in np.linspace(80, 100, 5):
            for t_cond in np.linspace(5, 15, 5):
                sys_opt = build_opt(t_coll, t_foc, t_rel, t_cond)
                ok, max_r = evaluate(sys_opt)
                if ok and max_r < best_r:
                    best_r = max_r
                    best_params = (t_coll, t_foc, t_rel, t_cond)
                    print(f'New best: {max_r:.3f} at {best_params}')

if best_params:
    build_opt(*best_params)
    with open('spectrometer_refractive_1064.json', 'w') as f:
        json.dump(data, f, indent=4)
    print('Saved fine sweep!')
else:
    print('Failed!')

