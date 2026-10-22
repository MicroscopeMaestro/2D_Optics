import sys, os, json, numpy as np
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

sys_opt = OpticalSystem.from_json('spectrometer_refractive_1064.json')
sys_opt._update_accumulated_positions()

wls = [1.087, 1.225, 1.350]
theta_max = np.arcsin(0.04)

for wl in wls:
    print(f"\n--- Wavelength {wl} um ---")
    for theta in [0.0, theta_max]:
        for phi in ([0.0] if theta == 0.0 else [0.0, np.pi/2, np.pi, 3*np.pi/2]):
            dx = np.sin(theta) * np.cos(phi)
            dy = np.sin(theta) * np.sin(phi)
            dz = np.cos(theta)
            r = Ray(pos=np.array([0., 0., 0.]), dir=np.array([dx, dy, dz]), wavelength=wl)
            tr = sys_opt.trace_ray(r)
            
            print(f"Ray theta={np.degrees(theta):.1f} deg, phi={np.degrees(phi):.1f} deg:")
            for h in tr.history:
                s = sys_opt.surfaces[h['surface_index']]
                xl, yl = h['x_local'], h['y_local']
                r_hit = np.sqrt(xl**2 + yl**2)
                sd = s.semi_diameter
                vig_str = " [VIGNETTED]" if h['vignetted'] else ""
                print(f"   surf {h['surface_index']:2d} ({s.name:22s}): r={r_hit:6.2f} mm (sd={sd:5.2f}){vig_str}")
            if tr.vignetted:
                print(f"   --> VIGNETTED at surface {len(tr.history)-1}: {sys_opt.surfaces[len(tr.history)-1].name}")
            else:
                h_det = tr.history[-1]
                print(f"   --> DETECTOR HIT: ({h_det['x_local']:.4f}, {h_det['y_local']:.4f})")
