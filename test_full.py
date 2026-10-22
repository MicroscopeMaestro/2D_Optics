
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath('')))
from v2.engine.lde import SurfaceStack
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

stack = SurfaceStack.load_from_json('C:/Users/Juan/Desktop/Spectrometer_V2_Full_System.json')
if isinstance(stack, dict): stack = stack['stack']
sys_opt = OpticalSystem(stack)

r = Ray(pos=np.array([0.,0.,0.]), dir=np.array([0.,0.,1.]), wavelength=1.225)
tr = sys_opt.trace_ray(r)
print('Chief Ray Vignetted:', tr.vignetted)

print('Check Ray off-axis (x=0, y=0.5) to see if it images to the slit properly:')
theta = np.arcsin(0.01) # Small NA for object emission
r2 = Ray(pos=np.array([0.,0.5,0.]), dir=np.array([0., np.sin(theta), np.cos(theta)]), wavelength=1.225)
tr2 = sys_opt.trace_ray(r2)
print('Ray 2 vignetted:', tr2.vignetted)

for h in tr2.history:
    sname = sys_opt.surfaces[h['surface_index']].name
    print(str(h['surface_index']) + ' ' + sname + ': Y=' + str(h['y_local']))

