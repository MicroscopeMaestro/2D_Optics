
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath('')))
from v2.engine.surface import SphericalSurface
from v2.engine.ray import Ray

sph = SphericalSurface(radius=33.3)
r = Ray(pos=np.array([0, 5.0, -10.0]), dir=np.array([0, 0, 1.0]), wavelength=1.0)
hit, normal = sph.intersect(r)
print('Hit:', hit)
print('Normal:', normal)

