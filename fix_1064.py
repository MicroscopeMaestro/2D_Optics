
import sys, os, json, numpy as np
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.surface import Surface, SurfaceType
from v2.engine.lde import SurfaceStack
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

with open('spectrometer_relay_redesign/Spectrometer_Hybrid_1064_Repaired.json', 'r') as f: data = json.load(f)

# Find M1 and Grating indices
m1_idx = next(i for i,s in enumerate(data['surfaces']) if 'M1' in s['name'])
grat_idx = next(i for i,s in enumerate(data['surfaces']) if 'Grating' in s['name'])

def test_tilt(m1_tilt):
    data['surfaces'][m1_idx]['tilt_x'] = m1_tilt
    # Rebuild stack
    stack = SurfaceStack()
    for sd in data['surfaces']:
        s = Surface(name=sd.get('name', ''))
        for k,v in sd.items():
            if k == 'surface_type': s.surface_type = SurfaceType.GRATING if v.lower() == 'grating' else (SurfaceType.STOP if v.lower() == 'stop' else (SurfaceType.OBJECT if v.lower() == 'object' else SurfaceType.STANDARD))
            elif hasattr(s, k): setattr(s, k, v)
        if 'Aspheric' in s.name:
            s.is_aspheric = True
            s.conic = sd.get('conic', 0.0)
        stack.add_surface(s)
    sys_opt = OpticalSystem(stack)
    sys_opt._update_accumulated_positions()
    
    m2 = sys_opt.surfaces[4]
    t = np.dot(m2.global_origin - np.array([0,0,0]), m2.global_R[:, 2]) / np.dot(np.array([0,0,1]), m2.global_R[:, 2])
    p_int = np.array([0,0,t])
    dist = np.linalg.norm(p_int - m2.global_origin)
    
    # Check vignetting at detector for 1.225
    r = Ray(pos=np.array([0., 0., 0.]), dir=np.array([0., 0., 1.]), wavelength=1.225)
    tr = sys_opt.trace_ray(r)
    vig = tr.vignetted
    
    return dist, vig

for tilt in np.arange(-8.0, -25.0, -1.0):
    try:
        dist, vig = test_tilt(tilt)
        print(f'M1 Tilt: {tilt}, Distance: {dist:.2f}, Vig: {vig}')
    except Exception as e:
        pass

