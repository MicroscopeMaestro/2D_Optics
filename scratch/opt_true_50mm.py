import sys
import os
import json
from scipy.optimize import minimize

sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/..'))
from optics_app.engine.tracer import OpticalSystem, Ray
import tempfile

with open('spectrometer_yz_40mm_100mm_slit.json') as f:
    config = json.load(f)

# True AC254-050-C
config['surfaces'][15]['radius'] = 22.9
config['surfaces'][15]['thickness'] = 9.0
config['surfaces'][15]['glass'] = 'N-LAK22'
config['surfaces'][16]['radius'] = -25.9
config['surfaces'][16]['thickness'] = 1.8
config['surfaces'][16]['glass'] = 'N-SF6'
config['surfaces'][17]['radius'] = 194.5
config['surfaces'][17]['glass'] = 'AIR'
config['surfaces'][17]['thickness'] = 41.2
config['surfaces'][18]['semi_diameter'] = 25.0

for s in config['surfaces'][19:]:
    if 'aperture_y_half' in s: s['aperture_y_half'] = 1000
    if 'aperture_x_half' in s: s['aperture_x_half'] = 1000

def evaluate(x):
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
    
    os.remove(tmp)
    # Give it freedom, just optimize for focus and spread
    return defocus * 10000 + max(0, spread - 0.79) * 500

# Start with a very short DMD distance (e.g. 50 mm)
res = minimize(evaluate, [50, 20, 15], method='Nelder-Mead', bounds=[(10, 60), (2, 50), (2, 50)])
print("Best:", res.x)
print("Penalty:", res.fun)

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

tr_marg = sys_temp.trace_ray(Ray(pos=(0, 0.005, 0), dir=(0,0,1), wavelength=1.225))
defocus = abs(tr_marg.history[-1]['y_global'] - ys[1])
print("Defocus:", defocus)
os.remove(tmp)
