
import sys, os, json, numpy as np
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

sys_opt = OpticalSystem.from_json('spectrometer_yz_75mm_halfinch.json')
sys_opt._update_accumulated_positions()

# Generate rays that perfectly pass through the slit!
# A ray passing through the slit has y_local < 0.1 at the slit plane.
# Since we just want to test the spectrometer, let's start the trace from the slit!
slit_idx = 6
slit_pos = sys_opt.surfaces[slit_idx].global_origin
slit_R = sys_opt.surfaces[slit_idx].global_R

wls = [1.087, 1.225, 1.350]
vignettes = 0
total = 0

for wl in wls:
    # 4mm stop at 30mm distance -> NA = 4/60 = 0.066
    for theta in np.linspace(0.001, 0.065, 5):
        for phi in np.linspace(0, 2*np.pi, 8, endpoint=False):
            # Start slightly inside the slit (y = 0.0)
            # Just test center of the slit
            dx = np.sin(theta) * np.cos(phi)
            dy = np.sin(theta) * np.sin(phi)
            dz = np.cos(theta)
            
            # Since trace_ray only works from the beginning, we can just trace from 0
            # BUT we need to find the correct initial ray angle that lands at y=0 at the slit!
            # Since the sample is at 0, a ray starting at origin with angle theta_0 
            # crosses the slit at some Y. 
            # If we just set the origin to [0,0,0], it's perfectly on-axis, so it hits the center of the slit!
            # Let's trace it, and just temporarily widen the slit so it doesn't block!
            pass

data = json.load(open('spectrometer_yz_75mm_halfinch.json'))
data['surfaces'][6]['semi_diameter'] = 5.0 # Widen slit
with open('temp_wide_slit.json', 'w') as f: json.dump(data, f)

sys_opt = OpticalSystem.from_json('temp_wide_slit.json')
sys_opt._update_accumulated_positions()

for wl in wls:
    for theta in np.linspace(0.001, 0.065, 5):
        for phi in np.linspace(0, 2*np.pi, 8, endpoint=False):
            dx = np.sin(theta) * np.cos(phi)
            dy = np.sin(theta) * np.sin(phi)
            dz = np.cos(theta)
            r = Ray(pos=np.array([0., 0., 0.]), dir=np.array([dx, dy, dz]), wavelength=wl)
            tr = sys_opt.trace_ray(r)
            total += 1
            if len(tr.history) < len(sys_opt.surfaces):
                vignettes += 1
                clip_surf = sys_opt.surfaces[len(tr.history)-1].name
                print(f'wl={wl} clipped at {clip_surf} (r={sys_opt.surfaces[len(tr.history)-1].semi_diameter})')

print(f'Total clipped: {vignettes} / {total}')

