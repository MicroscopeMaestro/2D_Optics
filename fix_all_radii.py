
import sys, os, json, numpy as np
from scipy.optimize import minimize, minimize_scalar
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

with open('teamwork_yz_spectrometer/spectrometer_yz_1064_final.json', 'r') as f: data = json.load(f)

# True AC254-030-C radii (from Collect lens)
r1, r2, r3 = 19.4, -14.2, -64.3
t1, t2 = 7.0, 2.0

# Collimator: light from slit (point source). 
# Infinite conjugate: the most curved surface (r1=19.4) should face the collimated beam (towards Grating).
# So slit -> R3 -> R2 -> R1
collimator_indices = [i for i, s in enumerate(data['surfaces']) if 'Collimator' in s.get('name', '')]
data['surfaces'][collimator_indices[0]]['radius'] = -r3
data['surfaces'][collimator_indices[0]]['thickness'] = t2
data['surfaces'][collimator_indices[0]]['glass'] = 'F2'
data['surfaces'][collimator_indices[1]]['radius'] = -r2
data['surfaces'][collimator_indices[1]]['thickness'] = t1
data['surfaces'][collimator_indices[1]]['glass'] = 'N-BAF10'
data['surfaces'][collimator_indices[2]]['radius'] = -r1

# Focusing: light from Grating (collimated).
# Infinite conjugate: most curved surface (r1=19.4) should face the collimated beam (towards Grating).
# So Grating -> R1 -> R2 -> R3
focus_indices = [i for i, s in enumerate(data['surfaces']) if 'Focusing' in s.get('name', '')]
data['surfaces'][focus_indices[0]]['radius'] = r1
data['surfaces'][focus_indices[0]]['thickness'] = t1
data['surfaces'][focus_indices[0]]['glass'] = 'N-BAF10'
data['surfaces'][focus_indices[1]]['radius'] = r2
data['surfaces'][focus_indices[1]]['thickness'] = t2
data['surfaces'][focus_indices[1]]['glass'] = 'F2'
data['surfaces'][focus_indices[2]]['radius'] = r3

data['surfaces'][6]['semi_diameter'] = 5.0 # widen slit

# Optimize Collimator distance (t_slit_collimator) to achieve perfect collimation!
# Wait, distance from slit to collimator is just the thickness of the slit surface (index 6).
def collimate_collimator(t):
    data['surfaces'][6]['thickness'] = t
    with open('temp_fix.json', 'w') as f: json.dump(data, f)
    sys_opt = OpticalSystem.from_json('temp_fix.json')
    sys_opt._update_accumulated_positions()
    r1_ray = Ray(pos=sys_opt.surfaces[6].global_origin, dir=np.array([0., 0.01, 0.99995]), wavelength=1.225)
    r2_ray = Ray(pos=sys_opt.surfaces[6].global_origin, dir=np.array([0., -0.01, 0.99995]), wavelength=1.225)
    tr1 = sys_opt.trace_ray(r1_ray)
    tr2 = sys_opt.trace_ray(r2_ray)
    c_out = collimator_indices[2]
    if tr1.vignetted or tr2.vignetted or len(tr1.history) <= c_out: return 1e6
    return np.abs(tr1.history[c_out]['dir_y'] - tr2.history[c_out]['dir_y'])

res_col = minimize_scalar(collimate_collimator, bounds=(20.0, 40.0), method='bounded')
print('t_slit_collimator =', res_col.x)
data['surfaces'][6]['thickness'] = res_col.x

# Optimize Focusing distance to DMD
dmd_idx = next(i for i, s in enumerate(data['surfaces']) if 'DMD' in s.get('name',''))
def focus_dmd(t):
    data['surfaces'][focus_indices[2]]['thickness'] = t
    with open('temp_fix.json', 'w') as f: json.dump(data, f)
    sys_opt = OpticalSystem.from_json('temp_fix.json')
    sys_opt._update_accumulated_positions()
    r1_ray = Ray(pos=sys_opt.surfaces[6].global_origin, dir=np.array([0., 0.01, 0.99995]), wavelength=1.225)
    r2_ray = Ray(pos=sys_opt.surfaces[6].global_origin, dir=np.array([0., -0.01, 0.99995]), wavelength=1.225)
    tr1 = sys_opt.trace_ray(r1_ray)
    tr2 = sys_opt.trace_ray(r2_ray)
    if tr1.vignetted or tr2.vignetted or len(tr1.history) <= dmd_idx: return 1e6
    return np.abs(tr1.history[dmd_idx]['y_local'] - tr2.history[dmd_idx]['y_local'])

res_foc = minimize_scalar(focus_dmd, bounds=(10.0, 40.0), method='bounded')
print('t_focus =', res_foc.x)
data['surfaces'][focus_indices[2]]['thickness'] = res_foc.x

# Now focus detector!
cond_idx = next(i for i, s in enumerate(data['surfaces']) if 'Cond' in s.get('name','') and 'S1' in s.get('name',''))
det_idx = len(data['surfaces']) - 1
def focus_det(t):
    data['surfaces'][cond_idx+1]['thickness'] = t
    with open('temp_fix.json', 'w') as f: json.dump(data, f)
    sys_opt = OpticalSystem.from_json('temp_fix.json')
    sys_opt._update_accumulated_positions()
    r1_ray = Ray(pos=sys_opt.surfaces[6].global_origin, dir=np.array([0., 0.01, 0.99995]), wavelength=1.225)
    r2_ray = Ray(pos=sys_opt.surfaces[6].global_origin, dir=np.array([0., -0.01, 0.99995]), wavelength=1.225)
    tr1 = sys_opt.trace_ray(r1_ray)
    tr2 = sys_opt.trace_ray(r2_ray)
    if tr1.vignetted or tr2.vignetted or len(tr1.history) <= det_idx: return 1e6
    return np.abs(tr1.history[det_idx]['y_local'] - tr2.history[det_idx]['y_local'])

res_det = minimize_scalar(focus_det, bounds=(5.0, 15.0), method='bounded')
print('t_cond_det =', res_det.x)
data['surfaces'][cond_idx+1]['thickness'] = res_det.x

data['surfaces'][6]['semi_diameter'] = 0.1
with open('teamwork_yz_spectrometer/spectrometer_yz_1064_final.json', 'w') as f:
    json.dump(data, f, indent=4)

