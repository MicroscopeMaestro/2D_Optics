import sys
import os
import json
from scipy.optimize import minimize

sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/..'))
from optics_app.engine.tracer import OpticalSystem, Ray
import tempfile

with open('spectrometer_yz_40mm_100mm_slit.json') as f:
    config = json.load(f)

# Use original optics
config['surfaces'][15]['radius'] = 34.5
config['surfaces'][15]['thickness'] = 12.0
config['surfaces'][15]['glass'] = 'N-BAK4'
config['surfaces'][16]['radius'] = -25.2
config['surfaces'][16]['thickness'] = 2.0
config['surfaces'][16]['glass'] = 'N-SF6HT'
config['surfaces'][17]['radius'] = -135.2
config['surfaces'][17]['glass'] = 'AIR'
config['surfaces'][17]['thickness'] = 43.80

for s in config['surfaces']:
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
    y_chiefs = []
    for wl in [1.087, 1.225, 1.35]:
        r = Ray(pos=(0,0,0), dir=(0, 0, 1.0), wavelength=wl)
        y_chiefs.append(sys_temp.trace_ray(r).history[-1]['y_global'])
    spread = max(y_chiefs) - min(y_chiefs)
    r_marg = Ray(pos=(0, 0.005, 0), dir=(0, 0, 1.0), wavelength=1.225)
    spot_defocus = abs(sys_temp.trace_ray(r_marg).history[-1]['y_global'] - y_chiefs[1])
    os.remove(tmp)
    length = sum(x)
    
    # Strictly enforce length <= 100 (which is shorter than original 102.7)
    penalty_length = 0 if length <= 100 else (length - 100) * 1000
    
    # Try to minimize spread and focus
    return spot_defocus * 10000 + penalty_length

# We don't penalize spread here. We just want to see if ANY focused solution exists with length <= 100
# that has spread < 0.8.
# Actually, let's just optimize for spread while keeping defocus = 0 and length <= 100.
def eval2(x):
    config['surfaces'][18]['thickness'] = x[0]
    config['surfaces'][22]['thickness'] = x[1]
    config['surfaces'][24]['thickness'] = x[2]
    with tempfile.NamedTemporaryFile('w', delete=False) as tf:
        json.dump(config, tf)
        tmp = tf.name
    sys_temp = OpticalSystem.from_json(tmp)
    ys = [sys_temp.trace_ray(Ray(pos=(0,0,0), dir=(0,0,1), wavelength=wl)).history[-1]['y_global'] for wl in [1.087, 1.225, 1.35]]
    spread = max(ys) - min(ys)
    defocus = abs(sys_temp.trace_ray(Ray(pos=(0, 0.005, 0), dir=(0,0,1), wavelength=1.225)).history[-1]['y_global'] - ys[1])
    os.remove(tmp)
    length = sum(x)
    
    return max(0, spread - 0.75)*100 + defocus * 10000 + (0 if length <= 100 else (length - 100)*1000)

res = minimize(eval2, [75, 15, 10], method='Nelder-Mead', bounds=[(10, 100), (1, 50), (2, 30)])
print("Best:", res.x)
print("Length:", sum(res.x))
print("Penalty:", res.fun)

# What was the spread?
config['surfaces'][18]['thickness'] = res.x[0]
config['surfaces'][22]['thickness'] = res.x[1]
config['surfaces'][24]['thickness'] = res.x[2]
with tempfile.NamedTemporaryFile('w', delete=False) as tf:
    json.dump(config, tf)
    tmp = tf.name
sys_temp = OpticalSystem.from_json(tmp)
ys = [sys_temp.trace_ray(Ray(pos=(0,0,0), dir=(0,0,1), wavelength=wl)).history[-1]['y_global'] for wl in [1.087, 1.225, 1.35]]
spread = max(ys) - min(ys)
print("Actual spread:", spread)
os.remove(tmp)
