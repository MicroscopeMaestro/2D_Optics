
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath('')))
from v2.engine.surface import Surface, SurfaceType
from v2.engine.lde import SurfaceStack
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

stack = SurfaceStack()
s1 = Surface(name='Air')
s2 = Surface(name='Glass', radius=33.3, thickness=10.0, glass='N-BAK4')
stack.add_surface(s1)
stack.add_surface(s2)
sys_opt = OpticalSystem(stack)

r = Ray(pos=np.array([0, 5.0, -10.0]), dir=np.array([0, 0, 1.0]), wavelength=1.225)
tr = sys_opt.trace_ray(r)
for h in tr.history:
    print(h['surface_index'], h['dir'])

