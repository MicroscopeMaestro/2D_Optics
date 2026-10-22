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

for s in config['surfaces']:
    if 'aperture_y_half' in s:
        s['aperture_y_half'] = 1000
    if 'aperture_x_half' in s:
        s['aperture_x_half'] = 1000
    if 'semi_diameter' in s:
        s['semi_diameter'] = 1000

def eval_inner(x, dmd_dist):
    relay_to_condenser, condenser_to_det = x
    config['surfaces'][18]['thickness'] = dmd_dist
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
            return 1e6
        y_chiefs.append(traced.history[-1]['y_global'])
    
    spread = max(y_chiefs) - min(y_chiefs)
    
    r_marg = Ray(pos=(0, 0.005, 0), dir=(0, 0, 1.0), wavelength=1.225)
    traced_marg = sys_temp.trace_ray(r_marg)
    if len(traced_marg.history) < 26:
        os.remove(tmp)
        return 1e6
        
    spot_defocus = abs(traced_marg.history[-1]['y_global'] - y_chiefs[1])
    os.remove(tmp)
    
    # We want perfect focus, so we heavily penalize defocus
    # We want spread < 0.78
    return spot_defocus * 1000 + max(0, spread - 0.78) * 100

print("Scanning DMD distance...")
best_res = None
best_score = float('inf')
best_dmd = None

for dmd in range(30, 110, 5):
    res = minimize(eval_inner, [20, 10], args=(dmd,), method='Nelder-Mead', bounds=[(2, 100), (2, 30)])
    if res.fun < 1e5:
        # Check actual spread and defocus
        config['surfaces'][18]['thickness'] = dmd
        config['surfaces'][22]['thickness'] = res.x[0]
        config['surfaces'][24]['thickness'] = res.x[1]
        with tempfile.NamedTemporaryFile('w', delete=False) as tf:
            json.dump(config, tf)
            tmp = tf.name
        sys_temp = OpticalSystem.from_json(tmp)
        ys = []
        for wl in [1.087, 1.225, 1.35]:
            r = Ray(pos=(0,0,0), dir=(0, 0, 1.0), wavelength=wl)
            ys.append(sys_temp.trace_ray(r).history[-1]['y_global'])
        spread = max(ys) - min(ys)
        rm = Ray(pos=(0, 0.005, 0), dir=(0, 0, 1.0), wavelength=1.225)
        defocus = abs(sys_temp.trace_ray(rm).history[-1]['y_global'] - ys[1])
        os.remove(tmp)
        
        print(f"DMD={dmd:3d} | Relay={res.x[0]:5.2f} | Cond={res.x[1]:5.2f} | Spread={spread:.3f} | Defocus={defocus:.4f} | Pen={res.fun:.2f}")

