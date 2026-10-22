import sys, os, math
import numpy as np
from collections import Counter

sys.path.insert(0, r"c:\Users\Juan\Documents\GitHub\2D_Optics")
from scratch.test_wavelength_band import test_band
from v2.engine.lde import SurfaceStack
from v2.engine.tracer import OpticalSystem, Ray
from v2.engine.surface import Surface, SurfaceType

# Build Band A stack
wls = [0.800, 0.815, 0.830, 0.845, 0.860]
stack = SurfaceStack()
stack.primary_wavelength = 0.830
stack.wavelengths = wls

stack.add_surface(Surface(
    name="Input_Slit_30um",
    surface_type=SurfaceType.OBJECT,
    thickness=48.5,
    glass="AIR",
    semi_diameter=12.7,
    aperture_type="Rectangular",
    aperture_x_half=0.250,
    aperture_y_half=0.015
))

stack.add_surface(Surface(
    name="Thorlabs_SM1D12_Iris_Stop",
    surface_type=SurfaceType.STOP,
    thickness=1.5,
    glass="AIR",
    semi_diameter=5.0,
    aperture_type="Circular",
    aperture_r_outer=5.0
))

stack.add_surface(Surface(
    name="M1_Collimator_CM254-050-P01",
    surface_type=SurfaceType.STANDARD,
    radius=-100.0,
    thickness=50.0,
    glass="MIRROR",
    semi_diameter=12.7,
    tilt_x=8.5,
    aperture_type="Circular",
    aperture_r_outer=12.0
))

stack.add_surface(Surface(
    name="Grating_GR25-0608_600lpmm",
    surface_type=SurfaceType.GRATING,
    radius=0.0,
    thickness=60.0,
    glass="MIRROR",
    semi_diameter=12.5,
    tilt_x=18.2,
    grating_lines_per_mm=600.0,
    grating_order=1,
    aperture_type="Rectangular",
    aperture_x_half=12.5,
    aperture_y_half=12.5
))

stack.add_surface(Surface(
    name="M2_Camera_CM254-075-P01",
    surface_type=SurfaceType.STANDARD,
    radius=-150.0,
    thickness=73.5,
    glass="MIRROR",
    semi_diameter=12.7,
    tilt_x=-8.5,
    aperture_type="Circular",
    aperture_r_outer=12.0
))

stack.add_surface(Surface(
    name="DMD4500_Spatial_Modulator",
    surface_type=SurfaceType.STANDARD,
    radius=0.0,
    thickness=150.0,
    glass="MIRROR",
    semi_diameter=6.0,
    tilt_x=12.0,
    aperture_type="Rectangular",
    aperture_x_half=2.5,
    aperture_y_half=2.5
))

stack.add_surface(Surface(
    name="M3_Relay_Collector_CM254-150-P01",
    surface_type=SurfaceType.STANDARD,
    radius=-300.0,
    thickness=50.0,
    glass="MIRROR",
    semi_diameter=12.7,
    tilt_x=8.0,
    aperture_type="Circular",
    aperture_r_outer=12.0
))

stack.add_surface(Surface(
    name="M4_Relay_Condenser_CM254-019-P01",
    surface_type=SurfaceType.STANDARD,
    radius=-38.0,
    thickness=18.8,
    glass="MIRROR",
    semi_diameter=12.7,
    tilt_x=-8.0,
    aperture_type="Circular",
    aperture_r_outer=12.0
))

stack.add_surface(Surface(
    name="Detector_0.8mm_Photodiode",
    surface_type=SurfaceType.DETECTOR,
    thickness=0.0,
    glass="AIR",
    semi_diameter=0.400,
    aperture_type="Circular",
    aperture_r_outer=0.400
))

stack._update_accumulated_positions()
opt = OpticalSystem(stack)

theta_max = math.asin(0.10)
rays = []
for wl in wls:
    for x in [-0.25, 0.0, 0.25]:
        for y in [-0.015, 0.0, 0.015]:
            rays.append(Ray(pos=np.array([x, y, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl))
            for ring in [0.4, 0.7, 1.0]:
                for p in range(8):
                    phi = 2 * math.pi * p / 8
                    kx = math.sin(ring * theta_max) * math.cos(phi)
                    ky = math.sin(ring * theta_max) * math.sin(phi)
                    kz = math.cos(ring * theta_max)
                    rays.append(Ray(pos=np.array([x, y, 0.0]), dir=np.array([kx, ky, kz]), wavelength=wl))

traced = [opt.trace_ray(r) for r in rays]
vig_surfs = []
for tr in traced:
    if tr.vignetted:
        for h in tr.history:
            s = stack.surfaces[h["surface_index"]]
            if not s.check_aperture(h["x_local"], h["y_local"]):
                vig_surfs.append((h["surface_index"], s.name, h["x_local"], h["y_local"]))
                break

c = Counter([v[0] for v in vig_surfs])
print("Vignetted surfaces distribution in Band A:")
for s_idx, cnt in c.items():
    print(f"  Surface {s_idx} ({stack.surfaces[s_idx].name}): {cnt} rays")

print("\nSample fails:")
for v in vig_surfs[:10]:
    print(f"  S{v[0]} ({v[1]}): local=({v[2]:.3f}, {v[3]:.3f})")
