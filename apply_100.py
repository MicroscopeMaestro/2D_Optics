
import sys, os, json, numpy as np
from scipy.optimize import minimize_scalar
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

data = json.load(open('teamwork_yz_spectrometer/spectrometer_yz_1064_final.json'))

# Update Relay lens to 100mm reversed doublet
# Index 20, 21, 22
data['surfaces'][20]['name'] = 'Relay_AC254-100-C_S1'
data['surfaces'][20]['radius'] = 96.16646564
data['surfaces'][20]['thickness'] = 2.0
data['surfaces'][20]['glass'] = 'N-SF6HT'

data['surfaces'][21]['name'] = 'Relay_AC254-100-C_S2'
data['surfaces'][21]['radius'] = 17.19369913
data['surfaces'][21]['thickness'] = 4.0
data['surfaces'][21]['glass'] = 'N-BAF10'

data['surfaces'][22]['name'] = 'Relay_AC254-100-C_S3'
data['surfaces'][22]['radius'] = -63.46078084
# thickness is distance to Condenser, keep it arbitrary 20.0 or whatever, wait, the distance is Condenser_S1 - Relay_S3.
# The JSON has thickness=20.0 for Relay_S3. Let's keep it.

# Now optimize t_dmd_relay!
# That is the distance from CB_After_DMD (index 19) to Relay_S1 (index 20).
def opt_relay(t):
    data['surfaces'][19]['thickness'] = t
    json.dump(data, open('t_temp.json', 'w'))
    sys_opt = OpticalSystem.from_json('t_temp.json')
    sys_opt._update_accumulated_positions()
    
    # We want perfectly collimated light AFTER Relay_S3 (index 22).
    # We trace NA rays from the Slit... actually from the Object!
    r0 = Ray(pos=sys_opt.surfaces[0].global_origin, dir=np.array([0., 0.0, 1.0]), wavelength=1.225)
    r1 = Ray(pos=sys_opt.surfaces[0].global_origin, dir=np.array([0., 0.01, 0.99995]), wavelength=1.225)
    tr0 = sys_opt.trace_ray(r0)
    tr1 = sys_opt.trace_ray(r1)
    
    if len(tr1.history) <= 23: return 1e6
    
    # Check if rays are parallel after Relay_S3 (i.e. at Condenser_S1, index 23)
    dir_y_0 = tr0.history[23]['dir_y']
    dir_y_1 = tr1.history[23]['dir_y']
    return np.abs(dir_y_1 - dir_y_0)

res = minimize_scalar(opt_relay, bounds=(90.0, 110.0), method='bounded')
print('t_dmd_relay =', res.x)
data['surfaces'][19]['thickness'] = res.x

json.dump(data, open('teamwork_yz_spectrometer/spectrometer_yz_1064_final.json', 'w'), indent=4)

