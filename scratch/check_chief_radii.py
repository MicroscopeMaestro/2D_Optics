import sys, os
sys.path.insert(0, os.path.abspath('.'))
import numpy as np
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from verify_layout import generate_fiber_phase_space_rays, RAMAN_WAVELENGTHS_UM

stack = SurfaceStack.load_from_json("hybrid_raman_1064.json")["stack"]
opt = OpticalSystem(stack)
rays = generate_fiber_phase_space_rays(fiber_radius_mm=0.0125, numerical_aperture=0.12, wavelengths=RAMAN_WAVELENGTHS_UM)
traced = opt.trace_rays(rays)
det_idx = len(stack.surfaces) - 1

chief_radii = []
for r in traced:
    if len(r.history) > 1 and (r.history[1]['x']**2 + r.history[1]['y']**2) < 0.01:
        for h in r.history:
            if h['surface_index'] == det_idx:
                rad = np.sqrt(h['x']**2 + h['y']**2)
                chief_radii.append((r.wavelength, h['x'], h['y'], rad))

print(f"Found {len(chief_radii)} chief rays:")
for wl, x, y, rad in chief_radii:
    print(f"  wl={wl:.5f} um: x={x:+.4f} mm, y={y:+.4f} mm -> radius = {rad:.4f} mm (dia = {rad*2:.4f} mm)")

max_chief_r = max(r[3] for r in chief_radii)
print(f"\nMAX CHIEF RAY RADIUS AT DETECTOR: {max_chief_r:.4f} mm (MAX DIAMETER: {max_chief_r*2:.4f} mm)")
print(f"PASS <= 0.500 mm? {max_chief_r <= 0.500}")
