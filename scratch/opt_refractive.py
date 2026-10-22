import sys, os, json, numpy as np
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.surface import Surface, SurfaceType
from v2.engine.lde import SurfaceStack
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

with open('spectrometer_refractive_1064.json', 'r') as f:
    base_data = json.load(f)

def build_system(params):
    # params: t_coll, t_foc_in, t_foc_out, t_relay_in, t_relay_mid, t_cond_out
    t_coll, t_foc_in, t_foc_out, t_relay_in, t_relay_mid, t_cond_out = params
    
    data = json.loads(json.dumps(base_data))
    
    # 0: Slit thickness (distance to Collimator)
    data['surfaces'][0]['thickness'] = float(t_coll)
    # 7: CB_After_Grating thickness (distance to Focusing)
    data['surfaces'][7]['thickness'] = float(t_foc_in)
    # 10: Focusing_S3 thickness (distance to DMD)
    data['surfaces'][10]['thickness'] = float(t_foc_out)
    # 12: CB_After_DMD thickness (distance to Relay)
    data['surfaces'][12]['thickness'] = float(t_relay_in)
    # 15: Imaging_Relay_S3 thickness (distance between Relay and Condenser)
    data['surfaces'][15]['thickness'] = float(t_relay_mid)
    # 17: Imaging_Cond_S2 thickness (distance to Detector)
    data['surfaces'][17]['thickness'] = float(t_cond_out)
    
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
        stack.add_surface(s)
    
    sys_opt = OpticalSystem(stack)
    sys_opt._update_accumulated_positions()
    return sys_opt, data

def evaluate(sys_opt):
    wls = [1.087, 1.225, 1.350]
    hits_x, hits_y = [], []
    vignettes = 0
    total_rays = 0
    
    theta_max = np.arcsin(0.04)
    for wl in wls:
        # Chief ray
        r0 = Ray(pos=np.array([0., 0., 0.]), dir=np.array([0., 0., 1.]), wavelength=wl)
        tr0 = sys_opt.trace_ray(r0)
        total_rays += 1
        if tr0.vignetted:
            vignettes += 1
        else:
            h = tr0.history[-1]
            hits_x.append(h['x_local'])
            hits_y.append(h['y_local'])
            
        for theta in np.linspace(0.005, theta_max, 3):
            for phi in np.linspace(0, 2*np.pi, 6, endpoint=False):
                dx = np.sin(theta)*np.cos(phi)
                dy = np.sin(theta)*np.sin(phi)
                dz = np.cos(theta)
                r = Ray(pos=np.array([0., 0., 0.]), dir=np.array([dx, dy, dz]), wavelength=wl)
                tr = sys_opt.trace_ray(r)
                total_rays += 1
                if tr.vignetted:
                    vignettes += 1
                else:
                    h = tr.history[-1]
                    hits_x.append(h['x_local'])
                    hits_y.append(h['y_local'])
    
    if len(hits_x) == 0:
        return 999.0, vignettes, total_rays
    
    mx, my = np.mean(hits_x), np.mean(hits_y)
    r_sq = (np.array(hits_x) - mx)**2 + (np.array(hits_y) - my)**2
    max_r = np.max(np.sqrt(r_sq))
    return max_r, vignettes, total_rays

# Initial evaluation
p0 = [25.0, 45.0, 45.0, 150.0, 531.09, 15.094]
sys_opt, _ = build_system(p0)
max_r, vig, tot = evaluate(sys_opt)
print(f"Initial: Max spot radius = {max_r:.4f} mm, Vignettes = {vig}/{tot}")
