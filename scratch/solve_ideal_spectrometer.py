import sys
import os
import math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from verify_layout import RAMAN_WAVELENGTHS_UM, generate_fiber_phase_space_rays

wl_ref = 1.38919 # Mid-band Raman wavelength

# 1. Solve Collimator Front Focal Length (Fiber -> AC254-050-C)
print("--- 1. SOLVING COLLIMATOR FRONT FOCAL DISTANCE (d0) ---")
# Light enters AC254-050-C from front (R1=34, R2=-27.8, R3=-115).
# Parallel beam in reverse: trace parallel ray from back into lens and find front focal point.
stack_rev = SurfaceStack()
stack_rev.add_surface(Surface(name="Obj", surface_type=SurfaceType.OBJECT, thickness=20.0, semi_diameter=25.0))
# Reversed AC254-050-C (back surface first)
stack_rev.add_surface(Surface(name="S3", radius=115.0, thickness=2.0, glass="SF5", semi_diameter=12.7))
stack_rev.add_surface(Surface(name="S2", radius=27.8, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
stack_rev.add_surface(Surface(name="S1", radius=-34.0, thickness=100.0, glass="AIR", semi_diameter=12.7))
stack_rev.add_surface(Surface(name="Det", surface_type=SurfaceType.DETECTOR, radius=0.0, thickness=0.0, glass="AIR", semi_diameter=25.0))
stack_rev._update_accumulated_positions()

opt_rev = OpticalSystem(stack_rev)
ray = Ray(pos=[3.0, 0.0, 0.0], dir=[0.0, 0.0, 1.0], wavelength=wl_ref)
tr = opt_rev.trace_rays([ray])[0]
h_out = tr.history[3]
# Find where ray hits x=0: z = -x / dir_x * dir_z
d0_ideal = -h_out['x'] / h_out['dir_x'] * h_out['dir_z']
print(f"Ideal Fiber-to-Collimator Distance (d0): {d0_ideal:.4f} mm")

# 2. Solve Camera Back Focal Length (Camera -> DMD)
print("\n--- 2. SOLVING CAMERA BACK FOCAL DISTANCE (d_cam_dmd) ---")
stack_cam = SurfaceStack()
stack_cam.add_surface(Surface(name="Obj", surface_type=SurfaceType.OBJECT, thickness=20.0, semi_diameter=25.0))
stack_cam.add_surface(Surface(name="S1", radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
stack_cam.add_surface(Surface(name="S2", radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
stack_cam.add_surface(Surface(name="S3", radius=-115.0, thickness=100.0, glass="AIR", semi_diameter=12.7))
stack_cam.add_surface(Surface(name="Det", surface_type=SurfaceType.DETECTOR, radius=0.0, thickness=0.0, glass="AIR", semi_diameter=25.0))
stack_cam._update_accumulated_positions()

opt_cam = OpticalSystem(stack_cam)
ray_cam = Ray(pos=[3.0, 0.0, 0.0], dir=[0.0, 0.0, 1.0], wavelength=wl_ref)
tr_cam = opt_cam.trace_rays([ray_cam])[0]
h_cam_out = tr_cam.history[3]
d_cam_dmd_ideal = -h_cam_out['x'] / h_cam_out['dir_x'] * h_cam_out['dir_z']
print(f"Ideal Camera-to-DMD Distance (d_cam_dmd): {d_cam_dmd_ideal:.4f} mm")

# 3. Solve Condenser Back Focal Length (ACL12708U-C -> Detector)
print("\n--- 3. SOLVING CONDENSER BACK FOCAL DISTANCE (d_cond_det) ---")
stack_cond = SurfaceStack()
stack_cond.add_surface(Surface(name="Obj", surface_type=SurfaceType.OBJECT, thickness=20.0, semi_diameter=25.0))
stack_cond.add_surface(Surface(name="C1", surface_type=SurfaceType.ASPHERIC, radius=4.16, conic=-0.6265, asphere_coeffs=[1.5e-5], thickness=4.5, glass="B270", semi_diameter=6.35))
stack_cond.add_surface(Surface(name="C2", radius=0.0, thickness=20.0, glass="AIR", semi_diameter=6.35))
stack_cond.add_surface(Surface(name="Det", surface_type=SurfaceType.DETECTOR, radius=0.0, thickness=0.0, glass="AIR", semi_diameter=25.0))
stack_cond._update_accumulated_positions()

opt_cond = OpticalSystem(stack_cond)
ray_cond = Ray(pos=[2.0, 0.0, 0.0], dir=[0.0, 0.0, 1.0], wavelength=wl_ref)
tr_cond = opt_cond.trace_rays([ray_cond])[0]
h_cond_out = tr_cond.history[2]
d_cond_det_ideal = -h_cond_out['x'] / h_cond_out['dir_x'] * h_cond_out['dir_z']
print(f"Ideal Condenser-to-Detector Distance (d_cond_det): {d_cond_det_ideal:.4f} mm")

