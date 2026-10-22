
import sys, os, json, numpy as np
from scipy.optimize import minimize
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

# Load the current layout
data = json.load(open('teamwork_yz_spectrometer/spectrometer_yz_1064_final.json'))

# We need to insert a lens. The current Focusing lens is at index 15, 16, 17.
# We will replace it with TWO lenses (6 surfaces + air gap).
# ACN254-030-C (-30mm): R1 = -16.9 (N-SF6HT), R2 = 19.3 (N-BAF10), R3 = 86.8. Wait, is it this?
# Thorlabs ACN254-030-C radii: -16.9, 19.3, 86.8. (Light hits -16.9 first)
# AC254-030-C (+30mm): R1 = 19.4 (N-BAF10), R2 = -14.2 (F2), R3 = -64.3.

surfs_before = data['surfaces'][:15]
surfs_after = data['surfaces'][18:]

retro_surfs = [
    # ACN254-030-C
    {'name': 'Retro_L1_ACN254-030-C_S1', 'radius': -16.9, 'thickness': 2.0, 'glass': 'N-SF6HT', 'surface_type': 'Standard', 'semi_diameter': 12.7},
    {'name': 'Retro_L1_ACN254-030-C_S2', 'radius': 19.3, 'thickness': 4.0, 'glass': 'N-BAF10', 'surface_type': 'Standard', 'semi_diameter': 12.7},
    {'name': 'Retro_L1_ACN254-030-C_S3', 'radius': 86.8, 'thickness': 45.0, 'glass': 'AIR', 'surface_type': 'Standard', 'semi_diameter': 12.7},
    
    # AC254-030-C
    {'name': 'Retro_L2_AC254-030-C_S1', 'radius': 19.4, 'thickness': 7.0, 'glass': 'N-BAF10', 'surface_type': 'Standard', 'semi_diameter': 12.7},
    {'name': 'Retro_L2_AC254-030-C_S2', 'radius': -14.2, 'thickness': 2.0, 'glass': 'F2', 'surface_type': 'Standard', 'semi_diameter': 12.7},
    {'name': 'Retro_L2_AC254-030-C_S3', 'radius': -64.3, 'thickness': 50.0, 'glass': 'AIR', 'surface_type': 'Standard', 'semi_diameter': 12.7},
]

data['surfaces'] = surfs_before + retro_surfs + surfs_after
json.dump(data, open('spectrometer_yz_retrofocus.json', 'w'), indent=4)

def opt(x):
    d, bfl = x
    data['surfaces'][17]['thickness'] = d
    data['surfaces'][20]['thickness'] = bfl
    json.dump(data, open('t_temp.json', 'w'))
    try:
        sys_opt = OpticalSystem.from_json('t_temp.json')
        sys_opt._update_accumulated_positions()
        
        # We want the NA ray from the object to focus perfectly on the DMD (index 21)
        r0 = Ray(pos=sys_opt.surfaces[0].global_origin, dir=np.array([0., 0.0, 1.0]), wavelength=1.225)
        r1 = Ray(pos=sys_opt.surfaces[0].global_origin, dir=np.array([0., 0.01, 0.99995]), wavelength=1.225)
        tr0 = sys_opt.trace_ray(r0)
        tr1 = sys_opt.trace_ray(r1)
        if len(tr1.history) <= 21: return 1e6
        
        spot_err = np.abs(tr1.history[21]['y_local'] - tr0.history[21]['y_local'])
        
        # And we want EFL to be approx 20.0 (or just whatever it naturally becomes with d~45)
        # Let's just minimize the spot size, and heavily constrain d to stay near 45 so we get ~50mm BFL.
        # Actually, let's just fix d = 45.0, and only optimize BFL!
        return spot_err
    except Exception as e:
        return 1e6

from scipy.optimize import minimize_scalar
res = minimize_scalar(lambda bfl: opt([45.0, bfl]), bounds=(40.0, 60.0), method='bounded')
print('Optimal BFL =', res.x)
data['surfaces'][17]['thickness'] = 45.0
data['surfaces'][20]['thickness'] = res.x
json.dump(data, open('spectrometer_yz_retrofocus.json', 'w'), indent=4)

