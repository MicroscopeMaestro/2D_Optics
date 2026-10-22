
import sys, os, json, numpy as np
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

import json, os

with open('test_50_200.json', 'r') as f:
    data = json.load(f)

# Ensure proper group names, labels, aperture settings
for s in data['surfaces']:
    if 'Relay' in s['name']:
        s['group_name'] = 'Relay_AC508_200_C'
    if 'DMD' in s['name']:
        s['aperture_type'] = 'Rectangular'
        s['aperture_x_half'] = 4.95
        s['aperture_y_half'] = 4.95
        s['aperture_a_semi'] = 4.95
        s['aperture_b_semi'] = 4.95
        s['aperture_r_outer'] = 5.0
        s['semi_diameter'] = 5.0

target_file = 'teamwork_refractive_1064/spectrometer_refractive_1064.json'
with open(target_file, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=4)

print(f"Updated {target_file} successfully.")


mx, my = np.mean(all_hits_x), np.mean(all_hits_y)
r_sq = (np.array(all_hits_x) - mx)**2 + (np.array(all_hits_y) - my)**2
max_r = float(np.max(np.sqrt(r_sq)))
rms_r = float(np.sqrt(np.mean(r_sq)))
max_r_origin = float(np.max(np.sqrt(np.array(all_hits_x)**2 + np.array(all_hits_y)**2)))

print(f"Total Traced Rays across 5 slit positions: {total}")
print(f"Non-Vignetted Rays: {len(all_hits_x)}/{total}")
print(f"Centroid: ({mx:.4f}, {my:.4f}) mm")
print(f"Max Spot Radius (from centroid): {max_r:.4f} mm (Threshold: <= 0.4000 mm)")
print(f"Max Spot Radius (from origin):   {max_r_origin:.4f} mm")
print(f"RMS Spot Radius:                 {rms_r:.4f} mm")
print(f"PASSED: {max_r <= 0.400}")

print("\n=== ALIGNMENT TOLERANCE (+/- 0.5 deg DMD tilt) ===")
dmd_idx = next(i for i, s in enumerate(sys_opt.surfaces) if 'DMD' in s.name)
for delta_tilt in [-0.5, -0.2, 0.0, +0.2, +0.5]:
    sys_opt.surfaces[dmd_idx].tilt_y = 12.0 + delta_tilt
    sys_opt._update_accumulated_positions()
    hits = []
    for wl in wls:
        r = Ray(pos=np.array([0., 0., 0.]), dir=np.array([0., 0., 1.]), wavelength=wl)
        tr = sys_opt.trace_ray(r)
        if not tr.vignetted:
            h = tr.history[-1]
            hits.append((h['x_local'], h['y_local']))
    print(f"DMD tilt {12.0+delta_tilt:+.2f} deg: {len(hits)}/3 chief rays captured, detector y coords: {[round(p[1], 3) for p in hits]}")













