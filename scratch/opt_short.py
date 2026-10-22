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

# Widen all apertures to avoid vignetting during optimization
for s in config['surfaces']:
    if 'aperture_y_half' in s:
        s['aperture_y_half'] = 1000
    if 'aperture_x_half' in s:
        s['aperture_x_half'] = 1000
    if 'semi_diameter' in s:
        s['semi_diameter'] = 1000

def evaluate(x):
    dmd_to_relay, relay_to_condenser, condenser_to_det = x
    
    config['surfaces'][18]['thickness'] = dmd_to_relay
    config['surfaces'][22]['thickness'] = relay_to_condenser
    config['surfaces'][24]['thickness'] = condenser_to_det
    
    with tempfile.NamedTemporaryFile('w', delete=False) as tf:
        json.dump(config, tf)
        tmp = tf.name
    
    sys_temp = OpticalSystem.from_json(tmp)
    
    y_chiefs = []
    for wl in [1.087, 1.225, 1.35]:
        r = Ray(pos=(0,0,0), dir=(0, 0, 1.0), wavelength=wl)
        traced = sys_temp.trace_ray(r)
        if len(traced.history) < 26:
            os.remove(tmp)
            return 1000000 
        y_chiefs.append(traced.history[-1]['y_global'])
    
    spread = max(y_chiefs) - min(y_chiefs)
    
    r_marg = Ray(pos=(0, 0.005, 0), dir=(0, 0, 1.0), wavelength=1.225)
    traced_marg = sys_temp.trace_ray(r_marg)
    if len(traced_marg.history) < 26:
        os.remove(tmp)
        return 1000000
    
    spot_defocus = abs(traced_marg.history[-1]['y_global'] - y_chiefs[1])
    
    os.remove(tmp)
    
    total_length = dmd_to_relay + relay_to_condenser + condenser_to_det
    
    penalty_spread = max(0, spread - 0.75) * 500
    penalty_focus = spot_defocus * 500
    penalty_length = total_length * 0.5
    
    return penalty_spread + penalty_focus + penalty_length

print("Running optimizations...")
best_res = None
best_score = float('inf')

for x0 in [
    [75, 28, 10],
    [97, 20, 8.5],
    [50, 40, 15]
]:
    res = minimize(evaluate, x0, method='Nelder-Mead', bounds=[(20, 150), (2, 100), (2, 30)])
    if res.fun < best_score:
        best_score = res.fun
        best_res = res

print("Best Optimization Result:")
print("DMD to Relay:", best_res.x[0])
print("Relay to Condenser:", best_res.x[1])
print("Condenser to Detector:", best_res.x[2])
print("Total length:", sum(best_res.x))
print("Penalty:", best_res.fun)
