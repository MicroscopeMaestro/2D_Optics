import sys
import os
import json

sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/..'))
from optics_app.engine.tracer import OpticalSystem, Ray
import tempfile

with open('spectrometer_yz_40mm_100mm_slit.json') as f:
    config = json.load(f)

# Restore standard infinite conjugate optics
config['surfaces'][15]['radius'] = 34.5
config['surfaces'][15]['thickness'] = 12.0
config['surfaces'][15]['glass'] = 'N-BAK4'
config['surfaces'][15]['name'] = 'Focusing_AC254-050-C_S1'

config['surfaces'][16]['radius'] = -25.2
config['surfaces'][16]['thickness'] = 2.0
config['surfaces'][16]['glass'] = 'N-SF6HT'
config['surfaces'][16]['name'] = 'Focusing_AC254-050-C_S2'

config['surfaces'][17]['radius'] = -135.2
config['surfaces'][17]['glass'] = 'AIR'
config['surfaces'][17]['thickness'] = 43.80
config['surfaces'][17]['name'] = 'Focusing_AC254-050-C_S3'

config['surfaces'][18]['thickness'] = 74.76
config['surfaces'][22]['thickness'] = 28.0
config['surfaces'][24]['thickness'] = 10.13

with tempfile.NamedTemporaryFile('w', delete=False) as tf:
    json.dump(config, tf)
    tmp = tf.name

sys_temp = OpticalSystem.from_json(tmp)

# Trace the new spectral range
wavelengths = [1.137, (1.137+1.316)/2, 1.316]
y_chiefs = []

print("Tracing new spectral range: 600 - 1800 cm-1")
for wl in wavelengths:
    r = Ray(pos=(0,0,0), dir=(0, 0, 1.0), wavelength=wl)
    traced = sys_temp.trace_ray(r)
    dmd_y = traced.history[18]['y_global']
    det_y = traced.history[-1]['y_global']
    y_chiefs.append(det_y)
    print(f"WL: {wl:.3f} um | DMD y: {dmd_y:.3f} mm | Det y: {det_y:.3f} mm")

spread = max(y_chiefs) - min(y_chiefs)
print(f"\nFinal Detector Footprint: {spread:.3f} mm")

# Verify focus
r_marg = Ray(pos=(0, 0.005, 0), dir=(0, 0, 1.0), wavelength=wavelengths[1])
tr_marg = sys_temp.trace_ray(r_marg)
defocus = abs(tr_marg.history[-1]['y_global'] - y_chiefs[1])
print(f"Defocus (Marginal ray diff): {defocus:.4f} mm")

os.remove(tmp)
