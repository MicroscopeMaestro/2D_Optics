
import sys, os, json, numpy as np
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

sys_opt = OpticalSystem.from_json('teamwork_yz_spectrometer/spectrometer_yz_1064.json')
sys_opt._update_accumulated_positions()

# Trace a bundle of rays for 1.225 um
rays = []
for theta in np.linspace(0.001, 0.04, 3):
    for phi in np.linspace(0, 2*np.pi, 6, endpoint=False):
        dx, dy, dz = np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)
        r = Ray(pos=np.array([0., 0., 0.]), dir=np.array([dx, dy, dz]), wavelength=1.225)
        tr = sys_opt.trace_ray(r)
        rays.append(tr)

# The focusing lens S3 is at index 15. The DMD is at index 16.
z_start = sys_opt.surfaces[15].global_origin[2]
z_end = sys_opt.surfaces[16].global_origin[2]

print(f'Focusing_S3 is at Z = {z_start:.2f}')
print(f'DMD is at Z = {z_end:.2f}')

# Sample planes between S3 and DMD
best_z = None
min_rms = 999.0

for test_z in np.linspace(z_start + 10, z_end + 20, 100):
    # Calculate ray intersections with the plane z = test_z
    pts = []
    for tr in rays:
        h1 = tr.history[15] # after S3
        # h1 has pos (x,y,z) and dir (dx,dy,dz)
        # ray eq: P = p0 + t * dir
        # z = p0.z + t * dz => t = (test_z - p0.z) / dz
        t = (test_z - h1['z_global']) / h1['dir_z']
        x = h1['x_global'] + t * h1['dir_x']
        y = h1['y_global'] + t * h1['dir_y']
        pts.append([x, y])
    
    pts = np.array(pts)
    centroid = np.mean(pts, axis=0)
    rms = np.sqrt(np.mean(np.sum((pts - centroid)**2, axis=1)))
    if rms < min_rms:
        min_rms = rms
        best_z = test_z

print(f'Best focus found at Z = {best_z:.2f} (RMS = {min_rms:.4f})')
print(f'Distance from S3 to best focus = {best_z - z_start:.2f}')
print(f'Distance from S3 to DMD in JSON = {z_end - z_start:.2f}')

