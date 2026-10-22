import sys
import os
import json
from scipy.optimize import minimize

sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/..'))
from optics_app.engine.tracer import OpticalSystem, Ray
import tempfile

with open('spectrometer_yz_40mm_100mm_slit.json') as f:
    config = json.load(f)

# Restore AC254-050-C
config['surfaces'][15]['radius'] = 34.5
config['surfaces'][15]['thickness'] = 12.0
config['surfaces'][15]['glass'] = 'N-BAK4'
config['surfaces'][16]['radius'] = -25.2
config['surfaces'][16]['thickness'] = 2.0
config['surfaces'][16]['glass'] = 'N-SF6HT'
config['surfaces'][17]['radius'] = -135.2
config['surfaces'][17]['glass'] = 'AIR'
config['surfaces'][17]['thickness'] = 43.80

# The CB_After_DMD must be aligned to the central ray (1.227)
# Let's set its tilt_x to -0.50 (found earlier)
config['surfaces'][19]['tilt_x'] = -0.50

# We'll use a short distance to prevent clipping
config['surfaces'][18]['thickness'] = 40.0

def evaluate(x):
    # x = [dmd_to_relay, relay_to_condenser, condenser_to_det]
    config['surfaces'][18]['thickness'] = x[0]
    config['surfaces'][22]['thickness'] = x[1]
    config['surfaces'][24]['thickness'] = x[2]
    
    with tempfile.NamedTemporaryFile('w', delete=False) as tf:
        json.dump(config, tf)
        tmp = tf.name
    sys_temp = OpticalSystem.from_json(tmp)
    
    ys = []
    for wl in [1.137, 1.227, 1.316]:
        tr = sys_temp.trace_ray(Ray(pos=(0,0,0), dir=(0,0,1), wavelength=wl))
        if len(tr.history) < 26:
            os.remove(tmp)
            return 1e6
        ys.append(tr.history[-1]['y_global'])
    
    spread = max(ys) - min(ys)
    
    tr_marg = sys_temp.trace_ray(Ray(pos=(0, 0.005, 0), dir=(0,0,1), wavelength=1.225))
    if len(tr_marg.history) < 26:
        os.remove(tmp)
        return 1e6
    defocus = abs(tr_marg.history[-1]['y_global'] - ys[1])
    
    length = sum(x)
    os.remove(tmp)
    
    # We want perfect focus, and minimal spread.
    return defocus * 10000 + max(0, spread - 0.75) * 100 + length * 0.1

print("Optimizing...")
res = minimize(evaluate, [40, 20, 15], method='Nelder-Mead', bounds=[(20, 65), (2, 50), (2, 30)])
print("Best:", res.x)
print("Length:", sum(res.x))
print("Penalty:", res.fun)

if res.fun < 1e5:
    config['surfaces'][18]['thickness'] = res.x[0]
    config['surfaces'][22]['thickness'] = res.x[1]
    config['surfaces'][24]['thickness'] = res.x[2]
    with tempfile.NamedTemporaryFile('w', delete=False) as tf:
        json.dump(config, tf)
        tmp = tf.name
    sys_temp = OpticalSystem.from_json(tmp)
    ys = []
    for wl in [1.137, 1.227, 1.316]:
        ys.append(sys_temp.trace_ray(Ray(pos=(0,0,0), dir=(0,0,1), wavelength=wl)).history[-1]['y_global'])
    print("Spread:", max(ys)-min(ys))
