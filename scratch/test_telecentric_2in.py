import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from verify_layout import RAMAN_WAVELENGTHS_UM, generate_fiber_phase_space_rays

stack = SurfaceStack.load_from_json('hybrid_raman_1064_full_range.json')['stack']
# Surfaces 9-11 are AC508-075-C with semi_diameter = 25.4 mm (2 inch)
# Surface 12-13 is ACL12708U-C with semi_diameter = 12.7 mm
# Surface 14 is Detector with semi_diameter = 0.50 mm (1.0 mm diameter)

rays = generate_fiber_phase_space_rays(fiber_radius_mm=0.0125, numerical_aperture=0.12, wavelengths=RAMAN_WAVELENGTHS_UM)
print(f"Total test rays: {len(rays)}")

print("\nScanning d_dmd_relay (S8 T) around 68-76mm and d_cond_det (S13 T) around 2.0-5.0mm:")
for d1 in [68.0, 70.0, 72.0, 73.5, 75.0, 76.0]:
    for d2 in [5.0, 10.0, 15.0]:
        for d3 in np.linspace(2.0, 5.0, 16):
            stack.surfaces[8].thickness = float(d1)
            stack.surfaces[11].thickness = float(d2)
            stack.surfaces[13].thickness = float(d3)
            stack.surfaces[14].semi_diameter = 25.0
            stack._update_accumulated_positions()
            
            opt = OpticalSystem(stack)
            traced = opt.trace_rays(rays)
            
            vig_optics = [r for r in traced if r.vignetted and (not r.history or r.history[-1]['surface_index'] < 14)]
            det_hits = [h for r in traced for h in r.history if h['surface_index'] == 14]
            
            if len(vig_optics) == 0 and len(det_hits) == len(rays):
                rs = [np.sqrt(h['x']**2 + h['y']**2) for h in det_hits]
                max_r = max(rs)
                rms_r = np.sqrt(np.mean(np.array(rs)**2))
                if max_r <= 0.500:
                    print(f"PERFECT PASS: d1={d1:4.1f}, d2={d2:4.1f}, d3={d3:4.2f} -> vig=0, max_r={max_r:.4f} mm (diam={2*max_r:.4f} mm <= 1.0mm), rms_r={rms_r:.4f} mm")
