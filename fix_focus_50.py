
import sys, os, json, numpy as np
from scipy.optimize import minimize_scalar
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

with open('teamwork_yz_spectrometer/spectrometer_yz_1064_final.json', 'r') as f: data = json.load(f)
data['surfaces'][6]['semi_diameter'] = 5.0
data['surfaces'][-1]['semi_diameter'] = 5.0 # widen detector so it doesn't fail

# True AC254-050-C radii 
r1, r2, r3 = 34.5, -23.1, -135.2
t1, t2 = 12.0, 2.0
glass1, glass2 = 'N-BAF10', 'F2' # Wait, originally it was N-SF6HT, let's use F2 or N-SF6HT, N-SF6HT is better.
glass2 = 'N-SF6HT'

focus_indices = [i for i, s in enumerate(data['surfaces']) if 'Focusing' in s.get('name', '')]
data['surfaces'][focus_indices[0]]['radius'] = r1
data['surfaces'][focus_indices[0]]['thickness'] = t1
data['surfaces'][focus_indices[0]]['glass'] = glass1
data['surfaces'][focus_indices[1]]['radius'] = r2
data['surfaces'][focus_indices[1]]['thickness'] = t2
data['surfaces'][focus_indices[1]]['glass'] = glass2
data['surfaces'][focus_indices[2]]['radius'] = r3
for i in range(3):
    data['surfaces'][focus_indices[i]]['name'] = f'Focusing_AC254-050-C_S{i+1}'

dmd_idx = next(i for i, s in enumerate(data['surfaces']) if 'DMD' in s.get('name',''))
def focus_dmd(t):
    data['surfaces'][focus_indices[2]]['thickness'] = t
    with open('temp_50.json', 'w') as f: json.dump(data, f)
    sys_opt = OpticalSystem.from_json('temp_50.json')
    sys_opt._update_accumulated_positions()
    r1_ray = Ray(pos=sys_opt.surfaces[6].global_origin, dir=np.array([0., 0.01, 0.99995]), wavelength=1.225)
    r2_ray = Ray(pos=sys_opt.surfaces[6].global_origin, dir=np.array([0., -0.01, 0.99995]), wavelength=1.225)
    tr1 = sys_opt.trace_ray(r1_ray)
    tr2 = sys_opt.trace_ray(r2_ray)
    if tr1.vignetted or tr2.vignetted or len(tr1.history) <= dmd_idx: return 1e6
    return np.abs(tr1.history[dmd_idx]['y_local'] - tr2.history[dmd_idx]['y_local'])

res_foc = minimize_scalar(focus_dmd, bounds=(30.0, 60.0), method='bounded')
print('t_focus =', res_foc.x)
data['surfaces'][focus_indices[2]]['thickness'] = res_foc.x
data['surfaces'][6]['semi_diameter'] = 0.1
data['surfaces'][-1]['semi_diameter'] = 0.4

with open('teamwork_yz_spectrometer/spectrometer_yz_1064_final.json', 'w') as f:
    json.dump(data, f, indent=4)

