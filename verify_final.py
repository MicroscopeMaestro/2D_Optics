
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath('')))
from v2.engine.lde import SurfaceStack
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray
stack = SurfaceStack.load_from_json('C:/Users/Juan/Desktop/Spectrometer_V2_Full_System.json')
if isinstance(stack, dict): stack = stack['stack']
sys_opt = OpticalSystem(stack)
theta = np.arcsin(0.05)
for phi in [0, np.pi/2, np.pi, 3*np.pi/2]:
    dx, dy, dz = np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)
    tr = sys_opt.trace_ray(Ray(pos=np.array([0.,0.,0.]), dir=np.array([dx,dy,dz]), wavelength=1.225))
    if tr.vignetted:
        print('Vignetted at ' + str(tr.history[-1]['surface_index']))
    else:
        print('Hit detector at X=' + str(tr.history[-1]['x_local']) + ' Y=' + str(tr.history[-1]['y_local']))

