import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from verify_layout import RAMAN_WAVELENGTHS_UM, generate_fiber_phase_space_rays

stack = SurfaceStack.load_from_json('hybrid_raman_1064_full_range.json')['stack']
stack.surfaces[-1].semi_diameter = 25.0

# Trace 1.3892 um center wavelength
rays_center = generate_fiber_phase_space_rays(fiber_radius_mm=0.0125, numerical_aperture=0.12, wavelengths=[1.38919])
opt = OpticalSystem(stack)
traced = opt.trace_rays(rays_center)

det_hits = [h for r in traced for h in r.history if h['surface_index'] == 14]
rs = [np.sqrt(h['x']**2 + h['y']**2) for h in det_hits]
xs = [h['x'] for h in det_hits]
ys = [h['y'] for h in det_hits]

print(f"Center Wavelength 1.38919 um (45 rays):")
print(f"  Detector x in [{min(xs):.4f}, {max(xs):.4f}] mm")
print(f"  Detector y in [{min(ys):.4f}, {max(ys):.4f}] mm")
print(f"  Max spot radius = {max(rs):.4f} mm (Diameter: {2*max(rs):.4f} mm)")
print(f"  RMS spot radius = {np.sqrt(np.mean(np.array(rs)**2)):.4f} mm")
