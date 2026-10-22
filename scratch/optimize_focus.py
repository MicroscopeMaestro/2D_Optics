import os
import sys
import math
import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.abspath("."))

from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray

WAVELENGTHS = [1.11130, 1.20000, 1.30000, 1.40000, 1.50000, 1.61330, 1.66708]
PRIMARY_WL = 1.36200

def create_system(t_cam_dmd=44.0, t_dmd_relay=20.0, t_relay_cond=5.0, t_cond_det=4.95):
    stack = SurfaceStack()
    stack.primary_wavelength = PRIMARY_WL
    
    # 0. Slit
    stack.add_surface(Surface(name="Slit_Source", surface_type=SurfaceType.OBJECT, thickness=47.5, glass="AIR", semi_diameter=25.0))
    
    # 1-3. Collimator AC254-050-C
    stack.add_surface(Surface(name="AC254-050-C_Coll_Front", radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Coll_Mid", radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Coll_Back", radius=-115.0, thickness=20.0, glass="AIR", semi_diameter=12.7))
    
    # 4. Grating GR25-0316
    stack.add_surface(Surface(name="Grating_GR25-0316", surface_type=SurfaceType.GRATING, radius=0.0, thickness=20.0, glass="MIRROR", semi_diameter=12.7, tilt_y=12.0, grating_lines_per_mm=300.0, grating_order=1))
    
    # 5-7. Camera AC254-050-C
    stack.add_surface(Surface(name="AC254-050-C_Focus_Front", radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Focus_Mid", radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Focus_Back", radius=-115.0, thickness=t_cam_dmd, glass="AIR", semi_diameter=12.7))
    
    # 8. DMD4500
    stack.add_surface(Surface(name="TI_DMD4500", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=t_dmd_relay, glass="MIRROR", semi_diameter=12.7, tilt_y=12.0))
    
    # 9-11. Relay AC508-075-C / AC254-075-C
    stack.add_surface(Surface(name="AC254-075-C_Relay_Front", radius=49.3, thickness=8.0, glass="N-BK7", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-075-C_Relay_Mid", radius=-39.0, thickness=3.0, glass="SF5", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-075-C_Relay_Back", radius=-170.0, thickness=t_relay_cond, glass="AIR", semi_diameter=12.7))
    
    # 12-13. Condenser ACL12708U-C
    stack.add_surface(Surface(name="ACL12708U-C_Cond_Front", surface_type=SurfaceType.ASPHERIC, radius=4.16, thickness=4.5, glass="B270", semi_diameter=12.7, conic=-0.6265, asphere_coeffs=[1.5e-5]))
    stack.add_surface(Surface(name="ACL12708U-C_Cond_Back", radius=0.0, thickness=t_cond_det, glass="AIR", semi_diameter=12.7))
    
    # 14. Detector
    stack.add_surface(Surface(name="Detector_InGaAs_1mm", surface_type=SurfaceType.DETECTOR, radius=0.0, thickness=0.0, glass="AIR", semi_diameter=1.00)) # temporarily large semi_diameter to catch all hit positions
    
    stack._update_accumulated_positions()
    return stack

def eval_params(params):
    t_cam_dmd, t_dmd_relay, t_relay_cond, t_cond_det = params
    stack = create_system(t_cam_dmd, t_dmd_relay, t_relay_cond, t_cond_det)
    opt = OpticalSystem(stack)
    
    # Chief rays
    chief_wls = [1.11130, 1.36200, 1.61330]
    chief_rays = [Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=w) for w in chief_wls]
    chief_traced = opt.trace_rays(chief_rays)
    
    dmd_hits = {}
    for r in chief_traced:
        hit = next(h for h in r.history if h["surface_index"] == 8)
        dmd_hits[r.wavelength] = hit["x"]
    span_1613 = abs(dmd_hits[1.61330] - dmd_hits[1.11130])
    
    # Sample rays across all 7 wavelengths
    na = 0.10
    max_theta = math.asin(na)
    slit_r = 0.0125
    rays = []
    for wl in WAVELENGTHS:
        for px in [-slit_r, 0.0, slit_r]:
            for ar in [0.0, 1.0]:
                if ar == 0.0:
                    rays.append(Ray(pos=np.array([px, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl))
                else:
                    for phi in [0, np.pi/2, np.pi, 3*np.pi/2]:
                        theta = ar * max_theta
                        vx = math.sin(theta) * math.cos(phi)
                        vy = math.sin(theta) * math.sin(phi)
                        vz = math.cos(theta)
                        rays.append(Ray(pos=np.array([px, 0.0, 0.0]), dir=np.array([vx, vy, vz]), wavelength=wl))
                        
    traced = opt.trace_rays(rays)
    det_hits = []
    for r in traced:
        dh = next((h for h in r.history if h["surface_index"] == 14), None)
        if dh:
            det_hits.append((dh["x"], dh["y"]))
            
    if len(det_hits) < len(rays) * 0.8:
        return 999.0
        
    radii = [math.hypot(x, y) for x, y in det_hits]
    max_r = max(radii)
    
    # Penalty for span outside [8.00, 8.664]
    span_penalty = 0.0
    if span_1613 < 8.00:
        span_penalty += (8.00 - span_1613) * 100.0
    elif span_1613 > 8.664:
        span_penalty += (span_1613 - 8.664) * 100.0
        
    return max_r + span_penalty

if __name__ == "__main__":
    p0 = [44.0, 20.0, 5.0, 4.95]
    print(f"Initial max_r = {eval_params(p0):.4f}")
    
    # Grid search around p0
    best_p = p0
    best_val = eval_params(p0)
    
    for t_cam in np.linspace(43.0, 45.0, 5):
        for t_relay in np.linspace(15.0, 30.0, 4):
            for t_rc in np.linspace(2.0, 10.0, 5):
                for t_cd in np.linspace(4.5, 5.5, 11):
                    val = eval_params([t_cam, t_relay, t_rc, t_cd])
                    if val < best_val:
                        best_val = val
                        best_p = [t_cam, t_relay, t_rc, t_cd]
                        print(f"New best: max_r = {best_val:.4f} | params = {best_p}")
                        
    print("\nFINAL OPTIMIZED PARAMETERS:", best_p)
    print(f"Optimized metric (max spot radius): {best_val:.4f} mm ({best_val*1000:.1f} um)")
