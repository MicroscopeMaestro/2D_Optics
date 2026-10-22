import sys, os, numpy as np
sys.path.insert(0, os.path.abspath('.'))
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem, Ray

json_path = 'spectrometer_raman_50mm_100mm_manual.json'
stack = SurfaceStack.load_from_json(json_path)['stack']
for s in stack.surfaces:
    s.semi_diameter = 100.0
    s.aperture_type = 'None'
sys_opt = OpticalSystem(stack)

ray = Ray(pos=np.array([0.,0.,0.]), dir=np.array([0.,0.,1.]), wavelength=1.087)
sys_opt.trace_rays([ray])
