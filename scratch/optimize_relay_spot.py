import sys, os, json, numpy as np
from scipy.optimize import minimize
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.surface import Surface, SurfaceType
from v2.engine.lde import SurfaceStack
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

with open('spectrometer_refractive_1064.json', 'r') as f:
    base_data = json.load(f)

def build_system(t_dmd_relay, t_relay_cond, t_cond_det, t_coll=25.0, t_foc=45.0):
    data = json.loads(json.dumps(base_data))
    data['surfaces'][0]['thickness'] = float(t_coll)
    data['surfaces'][10]['thickness'] = float(t_foc)
    data['surfaces'][12]['thickness'] = float(t_dmd_relay)
    data['surfaces'][15]['thickness'] = float(t_relay_cond)
    data['surfaces'][17]['thickness'] = float(t_cond_det)
    
    stack = SurfaceStack()
    for sd in data['surfaces']:
        s = Surface(name=sd.get('name', ''))
        for k, v in sd.items():
            if k == 'surface_type':
                st = v.lower()
                s.surface_type = (
                    SurfaceType.GRATING if st == 'grating' else
                    SurfaceType.STOP if st == 'stop' else
                    SurfaceType.OBJECT if st == 'object' else
                    SurfaceType.DETECTOR if st == 'detector' else
                    SurfaceType.ASPHERIC if st == 'aspheric' else
                    SurfaceType.STANDARD
                )
            elif hasattr(s, k):
                setattr(s, k, v)
        if 'Aspheric' in s.name or s.surface_type == SurfaceType.ASPHERIC or sd.get('conic', 0) != 0:
            s.is_aspheric = True
            s.conic = sd.get('conic', 0.0)
            s.asphere_coeffs = sd.get('asphere_coeffs', [])
        # Set detector semi-diameter large during tracing to catch all hits without early clipping
        if s.surface_type == SurfaceType.DETECTOR:
            s.semi_diameter = 5.0
        stack.add_surface(s)
    
    sys_opt = OpticalSystem(stack)
    sys_opt._update_accumulated_positions()
    return sys_opt, data

def evaluate(params):
    t_dmd_relay, t_relay_cond, t_cond_det = params
    
    # Boundary checks
    if t_dmd_relay < 50 or t_dmd_relay > 300: return 1e6
    if t_relay_cond < 20 or t_relay_cond > 800: return 1e6
    if t_cond_det < 5 or t_cond_det > 30: return 1e6
    
    sys_opt, _ = build_system(t_dmd_relay, t_relay_cond, t_cond_det)
    
    wls = [1.087, 1.225, 1.350]
    hits_x, hits_y = [], []
    vignetted = 0
    total = 0
    
    for wl in wls:
        # Chief ray
        r0 = Ray(pos=np.array([0., 0., 0.]), dir=np.array([0., 0., 1.]), wavelength=wl)
        tr0 = sys_opt.trace_ray(r0)
        total += 1
        if tr0.vignetted:
            vignetted += 1
        else:
            h = tr0.history[-1]
            hits_x.append(h['x_local'])
            hits_y.append(h['y_local'])
            
        for theta in np.linspace(0.005, np.arcsin(0.04), 3):
            for phi in np.linspace(0, 2*np.pi, 4, endpoint=False):
                dx = np.sin(theta)*np.cos(phi)
                dy = np.sin(theta)*np.sin(phi)
                dz = np.cos(theta)
                r = Ray(pos=np.array([0., 0., 0.]), dir=np.array([dx, dy, dz]), wavelength=wl)
                tr = sys_opt.trace_ray(r)
                total += 1
                if tr.vignetted:
                    vignetted += 1
                else:
                    h = tr.history[-1]
                    hits_x.append(h['x_local'])
                    hits_y.append(h['y_local'])
    
    if len(hits_x) == 0:
        return 1e6
    
    mx, my = np.mean(hits_x), np.mean(hits_y)
    r_sq = (np.array(hits_x) - mx)**2 + (np.array(hits_y) - my)**2
    max_r = np.max(np.sqrt(r_sq))
    rms_r = np.sqrt(np.mean(r_sq))
    
    # Loss: minimize max spot radius and penalty for any internal vignetting
    loss = max_r + 10.0 * vignetted + 0.1 * rms_r
    return loss

# Current baseline
init_p = [150.0, 531.09, 15.094]
print(f"Initial loss: {evaluate(init_p):.4f}")

res = minimize(evaluate, init_p, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-3, 'fatol': 1e-4})
print("Optimization finished:")
print("Success:", res.success)
print("Best params (t_dmd_relay, t_relay_cond, t_cond_det):", res.x)
print("Best loss:", res.fun)

# Evaluate best system
best_sys, best_data = build_system(*res.x)
wls = [1.087, 1.225, 1.350]
hits_x, hits_y = [], []
for wl in wls:
    for theta in np.linspace(0.001, np.arcsin(0.04), 3):
        for phi in np.linspace(0, 2*np.pi, 4, endpoint=False):
            dx = np.sin(theta)*np.cos(phi)
            dy = np.sin(theta)*np.sin(phi)
            dz = np.cos(theta)
            r = Ray(pos=np.array([0., 0., 0.]), dir=np.array([dx, dy, dz]), wavelength=wl)
            tr = best_sys.trace_ray(r)
            if not tr.vignetted:
                h = tr.history[-1]
                hits_x.append(h['x_local'])
                hits_y.append(h['y_local'])

if hits_x:
    mx, my = np.mean(hits_x), np.mean(hits_y)
    r_sq = (np.array(hits_x) - mx)**2 + (np.array(hits_y) - my)**2
    max_r = np.max(np.sqrt(r_sq))
    print(f"Optimized Max Spot Radius (non-vignetted): {max_r:.4f} mm across {len(hits_x)} rays")
