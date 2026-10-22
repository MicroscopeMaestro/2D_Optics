
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath('')))
from v2.engine.lde import SurfaceStack
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

stack = SurfaceStack.load_from_json('C:/Users/Juan/Desktop/Spectrometer_V2_2inch_Optimized.json')
if isinstance(stack, dict): stack = stack['stack']
sys_opt = OpticalSystem(stack)
for s in sys_opt.surfaces: s.semi_diameter = float('inf')

theta = np.arcsin(0.12)
dx, dy, dz = np.sin(theta)*np.cos(0), np.sin(theta)*np.sin(0), np.cos(theta)
r = Ray(pos=np.array([0.,0.,1e-3]), dir=np.array([dx,dy,dz]), wavelength=1.225)
tr = sys_opt.trace_ray(r)
last_dir = tr.history[-1]['dir']
s = sys_opt.surfaces[-1]
local_dir = s.global_R.T @ last_dir
print('local_dir:', local_dir)
angle = np.arccos(abs(local_dir[2])) * 180 / np.pi
print(f'Angle of ray at detector: {angle:.1f} degrees (NA={np.sin(angle*np.pi/180):.3f})')

