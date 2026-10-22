
import sys, os, json, numpy as np
from scipy.optimize import minimize
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.surface import Surface, SurfaceType
from v2.engine.lde import SurfaceStack
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

with open('teamwork_refractive_1064/spectrometer_refractive_1064.json', 'r') as f:
    teamwork_data = json.load(f)

# The original slit was 25 mm from Collimator, but the teamwork agent used 30mm symmetric lenses!
# Let's check what thickness the teamwork agent used for the Slit Object.
team_slit_thick = teamwork_data['surfaces'][0]['thickness']

def build_collection(t_obj, t_slit):
    surfaces = [
        {'name': 'Sample_Object', 'surface_type': 'Object', 'thickness': t_obj},
        {'name': 'Collect_L1_S1', 'surface_type': 'Aspheric', 'radius': 8.32, 'conic': -1.0, 'thickness': 8.0, 'glass': 'B270', 'semi_diameter': 12.7},
        {'name': 'Collect_L1_S2', 'surface_type': 'Standard', 'radius': 0.0, 'thickness': 30.0, 'glass': 'AIR', 'semi_diameter': 12.7},
        {'name': 'Collect_L2_S1', 'surface_type': 'Standard', 'radius': 19.4, 'thickness': 7.0, 'glass': 'N-BAF10', 'semi_diameter': 12.7},
        {'name': 'Collect_L2_S2', 'surface_type': 'Standard', 'radius': -14.2, 'thickness': 2.0, 'glass': 'N-SF6HT', 'semi_diameter': 12.7},
        {'name': 'Collect_L2_S3', 'surface_type': 'Standard', 'radius': -64.3, 'thickness': t_slit, 'glass': 'AIR', 'semi_diameter': 12.7},
        {'name': 'Spectrometer_Slit', 'surface_type': 'Standard', 'thickness': team_slit_thick, 'semi_diameter': 1.0, 'glass': 'AIR'}
    ]
    
    stack = SurfaceStack()
    for sd in surfaces:
        s = Surface(name=sd['name'])
        s.surface_type = SurfaceType.OBJECT if 'Object' in s.name else (SurfaceType.ASPHERIC if 'Aspheric' in s.name else SurfaceType.STANDARD)
        if s.surface_type == SurfaceType.ASPHERIC: s.is_aspheric = True
        for k, v in sd.items():
            if hasattr(s, k): setattr(s, k, v)
        stack.add_surface(s)
    
    sys_opt = OpticalSystem(stack)
    sys_opt._update_accumulated_positions()
    return sys_opt

def obj_func(x):
    t_obj, t_slit = x
    sys_opt = build_collection(t_obj, t_slit)
    
    # 1. Check collimation in the middle (Collect_L1_S2)
    r1 = Ray(pos=np.array([0., 0., 0.]), dir=np.array([0., np.sin(0.1), np.cos(0.1)]), wavelength=1.225)
    tr1 = sys_opt.trace_ray(r1)
    if tr1.vignetted: return 1e6
    h_mid = tr1.history[2] # after Collect_L1_S2
    dir_mid = np.array([h_mid['dir_x'], h_mid['dir_y'], h_mid['dir_z']])
    collimation_error = np.abs(dir_mid[1]) # should be 0 if perfectly collimated
    
    # 2. Check focus on the slit
    h_end = tr1.history[-2] # at Slit
    focus_error = np.abs(h_end['y_local'])
    
    return collimation_error*1000 + focus_error

res = minimize(obj_func, [11.4, 23.5], method='Nelder-Mead')
print(f'Optimized t_obj: {res.x[0]:.3f}, t_slit: {res.x[1]:.3f}')

# Now combine them!
final_sys = build_collection(res.x[0], res.x[1])

# Append teamwork surfaces
for i in range(1, len(teamwork_data['surfaces'])):
    sd = teamwork_data['surfaces'][i]
    s = Surface(name=sd.get('name', ''))
    for k,v in sd.items():
        if k == 'surface_type':
            if 'grating' in v.lower(): s.surface_type = SurfaceType.GRATING
            elif 'object' in v.lower(): s.surface_type = SurfaceType.OBJECT
            elif 'detector' in v.lower(): s.surface_type = SurfaceType.DETECTOR
            elif 'aspheric' in v.lower(): 
                s.surface_type = SurfaceType.ASPHERIC
                s.is_aspheric = True
            elif 'stop' in v.lower():
                s.surface_type = SurfaceType.STOP
            elif 'coordinate' in v.lower():
                s.surface_type = SurfaceType.COORDINATE_BREAK
                s.is_coord_break = True
            else: s.surface_type = SurfaceType.STANDARD
        elif hasattr(s, k): setattr(s, k, v)
    final_sys.stack.add_surface(s)

final_sys.stack.save_to_json('spectrometer_with_collection.json', wavelengths=[1.087, 1.225, 1.350])
print('Generated complete system JSON!')

