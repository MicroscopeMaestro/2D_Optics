import sys, os, json, numpy as np
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.surface import Surface, SurfaceType
from v2.engine.lde import SurfaceStack
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

with open('spectrometer_refractive_1064.json', 'r') as f:
    data = json.load(f)

# Let's test a telecentric relay:
# Distance DMD -> Relay: ~95 mm (BFL of 100mm doublet)
# Distance Relay -> Condenser: ~40 mm
# Distance Condenser -> Detector: ~12-15 mm

def build_test(d_dmd_relay, d_relay_cond, d_cond_det, use_30mm_focus=False):
    d = json.loads(json.dumps(data))
    
    if use_30mm_focus:
        # Replace focusing lens with AC254-030-C
        # S1: 21.09, t=13.0, N-BAF10
        # S2: -15.24, t=1.8, N-SF6HT
        # S3: -71.12, t=25.0, AIR
        d['surfaces'][8]['radius'] = 21.09
        d['surfaces'][8]['thickness'] = 13.0
        d['surfaces'][8]['glass'] = "N-BAF10"
        d['surfaces'][9]['radius'] = -15.24
        d['surfaces'][9]['thickness'] = 1.8
        d['surfaces'][9]['glass'] = "N-SF6HT"
        d['surfaces'][10]['radius'] = -71.12
        d['surfaces'][10]['thickness'] = 24.5 # Back focal length ~ 24.5 mm
        d['surfaces'][10]['glass'] = "AIR"
    
    d['surfaces'][12]['thickness'] = float(d_dmd_relay)
    d['surfaces'][15]['thickness'] = float(d_relay_cond)
    d['surfaces'][17]['thickness'] = float(d_cond_det)
    
    stack = SurfaceStack()
    for sd in d['surfaces']:
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
    return sys_opt, d

print("Testing Telecentric Relay configurations...")
for d_dmd in np.linspace(85, 105, 5):
    for d_mid in [30.0, 50.0, 70.0]:
        for d_det in np.linspace(10.0, 16.0, 7):
            sys_opt, _ = build_test(d_dmd, d_mid, d_det, use_30mm_focus=False)
            
            # Trace chief rays
            wls = [1.087, 1.225, 1.350]
            chief_hits = []
            vig = 0
            for wl in wls:
                r = Ray(pos=np.array([0., 0., 0.]), dir=np.array([0., 0., 1.]), wavelength=wl)
                tr = sys_opt.trace_ray(r)
                if tr.vignetted:
                    vig += 1
                else:
                    h = tr.history[-1]
                    chief_hits.append((h['x_local'], h['y_local']))
            
            if vig == 0 and len(chief_hits) == 3:
                cx = [pt[0] for pt in chief_hits]
                cy = [pt[1] for pt in chief_hits]
                span_chief = np.sqrt((max(cx)-min(cx))**2 + (max(cy)-min(cy))**2)
                max_rad_chief = max(np.sqrt(x**2 + y**2) for x, y in chief_hits)
                if max_rad_chief < 0.2:
                    print(f"d_dmd={d_dmd:.1f}, d_mid={d_mid:.1f}, d_det={d_det:.2f} -> Chief span={span_chief:.4f} mm, Max radius={max_rad_chief:.4f} mm")
