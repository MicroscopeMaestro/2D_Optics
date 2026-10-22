import os
import sys
import math
import numpy as np

sys.path.insert(0, os.path.abspath("."))

from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray

def generate_rays(wls, na=0.10, slit_r=0.0125, num_spatial=5, num_angular=9):
    max_theta = math.asin(na)
    rays = []
    # Spatial points on slit
    sp_pts = [(0.0, 0.0)]
    for r in [slit_r]:
        for phi in np.linspace(0, 2*np.pi, num_spatial - 1, endpoint=False):
            sp_pts.append((r * math.cos(phi), r * math.sin(phi)))
            
    # Angular directions
    ang_dirs = [(0.0, 0.0, 1.0)]
    for ar in [0.5, 1.0]:
        theta = ar * max_theta
        for phi in np.linspace(0, 2*np.pi, (num_angular - 1)//2, endpoint=False):
            ang_dirs.append((math.sin(theta)*math.cos(phi), math.sin(theta)*math.sin(phi), math.cos(theta)))
            
    for wl in wls:
        for px, py in sp_pts:
            for vx, vy, vz in ang_dirs:
                rays.append(Ray(pos=np.array([px, py, 0.0]), dir=np.array([vx, vy, vz]), wavelength=wl))
    return rays

def evaluate_system(stack, wls, na=0.10, slit_r=0.0125):
    stack._update_accumulated_positions()
    opt = OpticalSystem(stack)
    
    # 1. Chief rays
    chief_rays = [Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=w) for w in wls]
    chief_traced = opt.trace_rays(chief_rays)
    dmd_idx = next(i for i, s in enumerate(stack.surfaces) if "DMD" in s.name.upper())
    det_idx = len(stack.surfaces) - 1
    
    dmd_hits = {}
    for r in chief_traced:
        hit = next(h for h in r.history if h["surface_index"] == dmd_idx)
        dmd_hits[r.wavelength] = hit["x"]
        
    span_1613 = abs(dmd_hits[1.61330] - dmd_hits[1.11130])
    span_1667 = abs(dmd_hits[1.66708] - dmd_hits[1.11130])
    
    # 2. Phase-space rays
    rays = generate_rays(wls, na=na, slit_r=slit_r, num_spatial=5, num_angular=9)
    traced = opt.trace_rays(rays)
    
    det_hits = []
    vignetted_count = 0
    for r in traced:
        if not r.vignetted:
            dh = next((h for h in r.history if h["surface_index"] == det_idx), None)
            if dh:
                det_hits.append((dh["x"], dh["y"]))
            else:
                vignetted_count += 1
        else:
            vignetted_count += 1
            
    if det_hits:
        radii = [math.hypot(x, y) for x, y in det_hits]
        max_r = max(radii)
        rms_r = math.sqrt(np.mean(np.array(radii)**2))
        max_d = 2.0 * max_r
        return {
            "dmd_span_1613": span_1613,
            "dmd_span_1667": span_1667,
            "max_r": max_r,
            "max_d": max_d,
            "rms_r": rms_r,
            "vignetted": vignetted_count,
            "total_rays": len(rays),
            "dmd_hits": dmd_hits,
            "det_hits": det_hits
        }
    return None

wls = [1.11130, 1.20000, 1.30000, 1.40000, 1.50000, 1.61330, 1.66708]

# Let's test varying the condenser gap and camera focal position
print("Scanning condenser back focal distance...")
for gap_det in np.linspace(4.0, 6.0, 21):
    stack = SurfaceStack()
    stack.primary_wavelength = 1.36200
    stack.add_surface(Surface(name="Slit_Source", surface_type=SurfaceType.OBJECT, thickness=47.5, glass="AIR", semi_diameter=25.0))
    stack.add_surface(Surface(name="AC254-050-C_Coll_Front", radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Coll_Mid", radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Coll_Back", radius=-115.0, thickness=20.0, glass="AIR", semi_diameter=12.7))
    stack.add_surface(Surface(name="Grating_GR25-0316", surface_type=SurfaceType.GRATING, radius=0.0, thickness=20.0, glass="MIRROR", semi_diameter=12.7, tilt_y=12.0, grating_lines_per_mm=300.0, grating_order=1))
    stack.add_surface(Surface(name="AC254-050-C_Focus_Front", radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Focus_Mid", radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Focus_Back", radius=-115.0, thickness=44.0, glass="AIR", semi_diameter=12.7))
    stack.add_surface(Surface(name="TI_DMD4500", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=20.0, glass="MIRROR", semi_diameter=12.7, tilt_y=12.0))
    stack.add_surface(Surface(name="AC508-075-C_Relay_Front", radius=49.3, thickness=8.0, glass="N-BK7", semi_diameter=25.4))
    stack.add_surface(Surface(name="AC508-075-C_Relay_Mid", radius=-39.0, thickness=3.0, glass="SF5", semi_diameter=25.4))
    stack.add_surface(Surface(name="AC508-075-C_Relay_Back", radius=-170.0, thickness=5.0, glass="AIR", semi_diameter=25.4))
    stack.add_surface(Surface(name="ACL12708U-C_Cond_Front", surface_type=SurfaceType.ASPHERIC, radius=4.16, thickness=4.5, glass="B270", semi_diameter=12.7, conic=-0.6265, asphere_coeffs=[1.5e-5]))
    stack.add_surface(Surface(name="ACL12708U-C_Cond_Back", radius=0.0, thickness=gap_det, glass="AIR", semi_diameter=12.7))
    stack.add_surface(Surface(name="Detector_InGaAs_1mm", surface_type=SurfaceType.DETECTOR, radius=0.0, thickness=0.0, glass="AIR", semi_diameter=0.50))
    
    res = evaluate_system(stack, wls, na=0.10)
    if res:
        print(f"GapDet = {gap_det:5.2f} mm -> DMD Span: {res['dmd_span_1613']:.3f} mm | Spot Max D: {res['max_d']:.4f} mm | Spot RMS: {res['rms_r']:.4f} mm | Vignetted: {res['vignetted']}/{res['total_rays']}")
