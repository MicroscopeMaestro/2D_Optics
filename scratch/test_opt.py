import sys
import os
import json

sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/..'))
from optics_app.engine.tracer import OpticalSystem, Ray
import tempfile

with open('spectrometer_yz_40mm_100mm_slit.json') as f:
    config = json.load(f)

# Update distances
config['surfaces'][17]['thickness'] = 43.80
config['surfaces'][18]['thickness'] = 97.03
config['surfaces'][22]['thickness'] = 19.88
config['surfaces'][24]['thickness'] = 8.525

with tempfile.NamedTemporaryFile('w', delete=False) as tf:
    json.dump(config, tf)
    tmp = tf.name

sys_temp = OpticalSystem.from_json(tmp)

print("Tracing chief rays...")
y_chiefs = []
for wl in [1.087, 1.225, 1.35]:
    r = Ray(pos=(0,0,0), dir=(0, 0, 1.0), wavelength=wl)
    traced = sys_temp.trace_ray(r)
    y = traced.history[-1]['y_global']
    y_chiefs.append(y)
    print(f"WL {wl} hit at {y:.4f}")

spread = max(y_chiefs) - min(y_chiefs)
print(f"Total Spread: {spread:.4f} mm")

print("Tracing marginal ray...")
r_marg = Ray(pos=(0,0.005,0), dir=(0, 0, 1.0), wavelength=1.225)
traced_marg = sys_temp.trace_ray(r_marg)
y_marg = traced_marg.history[-1]['y_global']
print(f"Marginal ray diff: {abs(y_marg - y_chiefs[1]):.4f} mm")

os.remove(tmp)
