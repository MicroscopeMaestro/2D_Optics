import sys, os, math
import numpy as np

sys.path.insert(0, r"c:\Users\Juan\Documents\GitHub\2D_Optics")
from v2.engine.lde import SurfaceStack
from v2.engine.tracer import OpticalSystem, Ray

loaded = SurfaceStack.load_from_json("mirror_spectrometer/all_mirror_spectrometer.json")
stack = loaded["stack"]

print("Surface frames and origins:")
for i, s in enumerate(stack.surfaces):
    print(f"S{i:02d} ({s.name:<32}): origin = {s.global_origin}, z_axis = {s.global_R[:, 2]}")

opt = OpticalSystem(stack)
r_830 = Ray(pos=np.array([0., 0., 0.]), dir=np.array([0., 0., 1.]), wavelength=0.830)
r_785 = Ray(pos=np.array([0., 0., 0.]), dir=np.array([0., 0., 1.]), wavelength=0.785)
r_880 = Ray(pos=np.array([0., 0., 0.]), dir=np.array([0., 0., 1.]), wavelength=0.880)

t_830 = opt.trace_ray(r_830)
t_785 = opt.trace_ray(r_785)
t_880 = opt.trace_ray(r_880)

print("\n--- Ray Directions after Grating (S3) in Global Frame ---")
h3_830 = [h for h in t_830.history if h["surface_index"] == 3][0]
h3_785 = [h for h in t_785.history if h["surface_index"] == 3][0]
h3_880 = [h for h in t_880.history if h["surface_index"] == 3][0]

d_830 = h3_830["dir"]
d_785 = h3_785["dir"]
d_880 = h3_880["dir"]

print("d_830:", d_830)
print("d_785:", d_785)
print("d_880:", d_880)

ang_785 = math.degrees(math.acos(np.dot(d_830, d_785)))
ang_880 = math.degrees(math.acos(np.dot(d_830, d_880)))
print(f"Angle 785 vs 830: {ang_785:.3f} deg")
print(f"Angle 880 vs 830: {ang_880:.3f} deg")
print(f"Total angular spread: {ang_785 + ang_880:.3f} deg")
