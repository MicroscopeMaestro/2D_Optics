import sys
import os
import json
import numpy as np
from scipy.optimize import minimize

sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/..'))
from optics_app.engine.tracer import OpticalSystem, Ray
import tempfile

with open('spectrometer_yz_40mm_100mm_slit.json') as f:
    config = json.load(f)

# Ensure optimal focusing on DMD
config['surfaces'][17]['thickness'] = 43.80

def evaluate(x):
    dmd_to_relay, relay_to_condenser, condenser_to_det = x
    
    # Assign thicknesses
    config['surfaces'][18]['thickness'] = dmd_to_relay
    config['surfaces'][22]['thickness'] = relay_to_condenser
    config['surfaces'][24]['thickness'] = condenser_to_det
    
    with tempfile.NamedTemporaryFile('w', delete=False) as tf:
        json.dump(config, tf)
        tmp = tf.name
    
    sys_temp = OpticalSystem.from_json(tmp)
    
    # Trace the 3 chief rays
    y_chiefs = []
    for wl in [1.087, 1.225, 1.35]:
        r = Ray(pos=(0,0,0), dir=(0, 0, 1.0), wavelength=wl)
        traced = sys_temp.trace_ray(r)
        if len(traced.history) < 26:
            os.remove(tmp)
            return 1000 # Vignetted
        y_chiefs.append(traced.history[-1]['y_global'])
    
    spread = max(y_chiefs) - min(y_chiefs)
    
    # Trace a marginal ray to check focus (spot size for center wavelength)
    # The slit has y_half = 0.0125. Let's trace from the slit.
    # Actually, we can just trace a slightly off-axis ray from the object.
    r_marg = Ray(pos=(0, 0.005, 0), dir=(0, 0, 1.0), wavelength=1.225)
    traced_marg = sys_temp.trace_ray(r_marg)
    if len(traced_marg.history) < 26:
        os.remove(tmp)
        return 1000
    
    spot_defocus = abs(traced_marg.history[-1]['y_global'] - y_chiefs[1])
    
    os.remove(tmp)
    
    # We want spread + spot_size to be < 0.8
    # Let's heavily penalize spot_defocus (we want sharp focus)
    # And we want spread < 0.7
    penalty = spot_defocus * 100 + max(0, spread - 0.7) * 50
    return penalty

# Initial guess: current values
x0 = [74.76, 28.0, 10.13]
res = minimize(evaluate, x0, method='Nelder-Mead', bounds=[(30, 150), (5, 100), (2, 30)])
print("Optimization Result:")
print("DMD to Relay:", res.x[0])
print("Relay to Condenser:", res.x[1])
print("Condenser to Detector:", res.x[2])
print("Penalty:", res.fun)
