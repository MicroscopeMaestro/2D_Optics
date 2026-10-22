
import sys, os, json, numpy as np
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

sys_opt = OpticalSystem.from_json('spectrometer_yz_75mm_stop4.json')
sys_opt._update_accumulated_positions()

# F/7.5 implies NA = 0.066
# Wait, let's just trace from the origin out to the edge of the stop (4mm diam = 2mm radius).
# Stop is at index 10. Distance to stop is approx 30mm.
wls = [1.087, 1.225, 1.350]
all_hits_x, all_hits_y = [], []
vignettes = 0

for wl in wls:
    # Use NA corresponding to the stop!
    # Let's just use theta = 0.066
    for theta in np.linspace(0.001, np.arcsin(0.066), 5):
        for phi in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            dx = np.sin(theta) * np.cos(phi)
            dy = np.sin(theta) * np.sin(phi)
            dz = np.cos(theta)
            r = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([dx, dy, dz]), wavelength=wl)
            tr = sys_opt.trace_ray(r)
            if tr.vignetted: vignettes += 1
            else:
                h = tr.history[-1]
                all_hits_x.append(h['x_local'])
                all_hits_y.append(h['y_local'])

if len(all_hits_x) > 0:
    all_r_from_origin = np.sqrt(np.array(all_hits_x)**2 + np.array(all_hits_y)**2)
    max_r_origin = float(np.max(all_r_from_origin))
    print(f'Max spot radius = {max_r_origin:.4f} mm')
print(f'Vignetted rays: {vignettes} out of 120')

