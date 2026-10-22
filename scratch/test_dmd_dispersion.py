"""
Verify DMD dispersion and focus across all 7 wavelengths.
"""
import sys
import os
import math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray

WAVELENGTHS = [1.1113, 1.2000, 1.3000, 1.3892, 1.5000, 1.6000, 1.6671]

stack = SurfaceStack()
stack.primary_wavelength = 1.3892
stack.add_surface(Surface(name="Fiber", surface_type=SurfaceType.OBJECT, thickness=64.57, glass="AIR"))
stack.add_surface(Surface(name="Coll1", radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
stack.add_surface(Surface(name="Coll2", radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
stack.add_surface(Surface(name="Coll3", radius=-115.0, thickness=15.0, glass="AIR", semi_diameter=12.7))
stack.add_surface(Surface(name="Grating", surface_type=SurfaceType.GRATING, radius=0.0, thickness=15.0, glass="MIRROR", semi_diameter=12.7, tilt_y=12.0, grating_lines_per_mm=300.0, grating_order=1))
stack.add_surface(Surface(name="Foc1", radius=25.1, thickness=6.8, glass="N-BK7", semi_diameter=12.7))
stack.add_surface(Surface(name="Foc2", radius=-20.2, thickness=2.0, glass="SF5", semi_diameter=12.7))
stack.add_surface(Surface(name="Foc3", radius=-84.0, thickness=46.08, glass="AIR", semi_diameter=12.7))
stack.add_surface(Surface(name="DMD", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=0.0, glass="MIRROR", semi_diameter=12.7, tilt_y=12.0, aperture_type="Rectangular", aperture_x_half=4.332, aperture_y_half=2.4365))
stack._update_accumulated_positions()

opt = OpticalSystem(stack)
na = 0.12
max_theta = math.asin(na)
fiber_r = 0.0125

print("=== SPECTRAL DISPERSION ON DMD PLANE (SURFACE 8) ===")
dmd_xs_all = []
for wl in WAVELENGTHS:
    rays = [
        Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl),
        Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([math.sin(max_theta), 0.0, math.cos(max_theta)]), wavelength=wl),
        Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([-math.sin(max_theta), 0.0, math.cos(max_theta)]), wavelength=wl),
        Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, math.sin(max_theta), math.cos(max_theta)]), wavelength=wl),
        Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, -math.sin(max_theta), math.cos(max_theta)]), wavelength=wl),
        Ray(pos=np.array([fiber_r, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl),
        Ray(pos=np.array([-fiber_r, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl),
    ]
    tr = opt.trace_rays(rays)
    hits = [h for r in tr for h in r.history if h['surface_index'] == 8]
    xs = [h['x'] for h in hits]
    ys = [h['y'] for h in hits]
    dmd_xs_all.extend(xs)
    cx = np.mean(xs)
    spot_rms = np.std(xs)
    print(f"WL {wl:.4f} um: Centroid x = {cx:+.3f} mm, Span x in [{min(xs):+.3f}, {max(xs):+.3f}], RMS = {spot_rms*1000:.1f} um, y max = {max(map(abs, ys)):.3f} mm")

print(f"\nTotal Spectral Dispersion Span: {max(dmd_xs_all) - min(dmd_xs_all):.3f} mm (TI DMD4500 Active Width: 8.664 mm)")
print(f"DMD Active Width Coverage: {(max(dmd_xs_all) - min(dmd_xs_all)) / 8.664 * 100:.1f}%")
