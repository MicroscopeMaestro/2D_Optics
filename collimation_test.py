import json
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
import numpy as np

stack = SurfaceStack.load_from_json('fiber_spectrometer_setup.json')['stack']

surf1 = stack.surfaces[1]
surf2 = stack.surfaces[2]
surf3 = stack.surfaces[3]

print(f'L1 Front: r={surf1.radius}, t={surf1.thickness}, glass={surf1.glass}')
print(f'L1 Middle: r={surf2.radius}, t={surf2.thickness}, glass={surf2.glass}')
print(f'L1 Back: r={surf3.radius}, t={surf3.thickness}, glass={surf3.glass}')
print(f'Distance from source to L1 Front: {stack.surfaces[0].thickness}')

ang = np.radians(1.0)
ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, np.sin(ang), np.cos(ang)]), wavelength=0.7)
sys = OpticalSystem(stack)
traced = sys.trace_ray(ray)
dir_after_l1 = traced.history[3] # Leaving L1 Back
print(f'Ray direction after L1: dir_y = {dir_after_l1["dir_y"]:.6f}')
