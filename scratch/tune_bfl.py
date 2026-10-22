import sys, os, math, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scratch.design_1inch_optics_spectrometer import build_1inch_raman_spectrometer
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray

for bfl in np.linspace(7.0, 9.0, 21):
    stack = build_1inch_raman_spectrometer()
    stack.surfaces[14].thickness = bfl
    stack._update_accumulated_positions()
    sys_opt = OpticalSystem(stack)
    
    det_hits = []
    wls = [0.785, 0.850, 0.935]
    max_half_angle = math.asin(0.12)
    for wl in wls:
        for py in np.linspace(-1.0, 1.0, 7):
            for px in np.linspace(-1.0, 1.0, 7):
                if px**2 + py**2 <= 1.0:
                    pos = np.array([px * 0.025, py * 0.025, 0.0])
                    theta_x = px * max_half_angle
                    theta_y = py * max_half_angle
                    dir_z = math.cos(math.sqrt(theta_x**2 + theta_y**2))
                    dir_vec = np.array([math.sin(theta_x), math.sin(theta_y), dir_z])
                    dir_vec /= np.linalg.norm(dir_vec)

                    r = Ray(pos=pos, dir=dir_vec, wavelength=wl)
                    tr = sys_opt.trace_ray(r)
                    if tr and len(tr.history) > 1 and not tr.vignetted:
                        for h in tr.history:
                            if h["surface_index"] == 15:
                                det_hits.append((h["x_local"], h["y_local"]))
    if det_hits:
        radii = [math.hypot(x, y) for x, y in det_hits]
        print(f"BFL = {bfl:.2f} mm -> r_max = {max(radii)*1000:.1f} um, r_rms = {math.sqrt(np.mean(np.array(radii)**2))*1000:.1f} um, count = {len(det_hits)}")
