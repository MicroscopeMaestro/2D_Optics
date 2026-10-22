import os
import sys
import math
import numpy as np
import json

sys.path.insert(0, os.path.abspath("."))

from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from optics_app.cad.exporter import CADExporter

WAVELENGTHS = [1.11130, 1.20000, 1.30000, 1.40000, 1.50000, 1.61330, 1.66708]
PRIMARY_WL = 1.36200

def test_single_design():
    stack = SurfaceStack()
    stack.primary_wavelength = PRIMARY_WL
    
    # 0. Slit
    stack.add_surface(Surface(
        name="Slit_Source",
        surface_type=SurfaceType.OBJECT,
        thickness=45.0,
        glass="AIR",
        semi_diameter=12.7,
        aperture_type="Circular",
        aperture_r_outer=0.0125
    ))
    
    # 1-3. Collimator: Thorlabs AC254-050-C
    stack.add_surface(Surface(name="AC254-050-C_Coll_Front", surface_type=SurfaceType.STANDARD, radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Coll_Mid", surface_type=SurfaceType.STANDARD, radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Coll_Back", surface_type=SurfaceType.STANDARD, radius=-115.0, thickness=15.0, glass="AIR", semi_diameter=12.7))
    
    # 4. Grating: Thorlabs GR25-0316 (300 l/mm, 1600nm blaze)
    stack.add_surface(Surface(name="Grating_GR25-0316", surface_type=SurfaceType.GRATING, radius=0.0, thickness=15.0, glass="MIRROR", semi_diameter=12.7, tilt_y=12.0, grating_lines_per_mm=300.0, grating_order=1))
    
    # 5-7. Camera: Thorlabs AC254-050-C
    stack.add_surface(Surface(name="AC254-050-C_Focus_Front", surface_type=SurfaceType.STANDARD, radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Focus_Mid", surface_type=SurfaceType.STANDARD, radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Focus_Back", surface_type=SurfaceType.STANDARD, radius=-115.0, thickness=45.0, glass="AIR", semi_diameter=12.7))
    
    # 8. DMD4500 (8.664 x 4.873 mm, +12 deg mechanical tilt -> +24 deg reflection)
    stack.add_surface(Surface(name="TI_DMD4500", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=45.0, glass="MIRROR", semi_diameter=12.7, tilt_y=12.0))
    
    # 9-11. Collection Relay: Thorlabs AC254-075-C
    stack.add_surface(Surface(name="AC254-075-C_Relay_Front", surface_type=SurfaceType.STANDARD, radius=49.3, thickness=8.0, glass="N-BK7", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-075-C_Relay_Mid", surface_type=SurfaceType.STANDARD, radius=-39.0, thickness=3.0, glass="SF5", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-075-C_Relay_Back", surface_type=SurfaceType.STANDARD, radius=-170.0, thickness=10.0, glass="AIR", semi_diameter=12.7))
    
    # 12-13. Condenser: Thorlabs ACL12708U-C (f=8mm, NA=0.78, B270)
    stack.add_surface(Surface(name="ACL12708U-C_Cond_Front", surface_type=SurfaceType.ASPHERIC, radius=4.16, thickness=4.5, glass="B270", semi_diameter=12.7, conic=-0.6265, asphere_coeffs=[1.5e-5]))
    stack.add_surface(Surface(name="ACL12708U-C_Cond_Back", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=4.85, glass="AIR", semi_diameter=12.7))
    
    # 14. Single-pixel Detector (1.0 mm InGaAs photodiode)
    stack.add_surface(Surface(name="Detector_InGaAs_1mm", surface_type=SurfaceType.DETECTOR, radius=0.0, thickness=0.0, glass="AIR", semi_diameter=0.50))
    
    stack._update_accumulated_positions()
    opt = OpticalSystem(stack)
    
    # Trace Chief Rays
    chief_wls = [1.11130, 1.20000, 1.30000, 1.36200, 1.40000, 1.50000, 1.61330, 1.66708]
    chief_rays = [Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=w) for w in chief_wls]
    chief_traced = opt.trace_rays(chief_rays)
    
    dmd_hits = {}
    for r in chief_traced:
        hit = next(h for h in r.history if h["surface_index"] == 8)
        dmd_hits[r.wavelength] = hit["x"]
        print(f"Chief ray @ {r.wavelength:.5f} um -> DMD X = {hit['x']:7.3f} mm")
        
    span_1613 = abs(dmd_hits[1.61330] - dmd_hits[1.11130])
    span_1667 = abs(dmd_hits[1.66708] - dmd_hits[1.11130])
    print(f"\nDMD Chief Ray Span (1.11130 to 1.61330 um): {span_1613:.4f} mm")
    print(f"DMD Chief Ray Span (1.11130 to 1.66708 um): {span_1667:.4f} mm")
    print(f"Span in [8.00 mm, 8.664 mm]: {8.00 <= span_1613 <= 8.664}")
    
    # Trace Polychromatic Phase-Space Bundle (NA = 0.10, 25 um slit)
    na = 0.10
    max_theta = math.asin(na)
    slit_r = 0.0125
    rays = []
    for wl in chief_wls:
        for pr in [0.0, slit_r]:
            for p_phi in np.linspace(0, 2*np.pi, 4, endpoint=False):
                px = pr * math.cos(p_phi)
                py = pr * math.sin(p_phi)
                for ar in [0.0, 0.5, 1.0]:
                    if ar == 0.0:
                        rays.append(Ray(pos=np.array([px, py, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl))
                    else:
                        for a_phi in np.linspace(0, 2*np.pi, 4, endpoint=False):
                            theta = ar * max_theta
                            vx = math.sin(theta) * math.cos(a_phi)
                            vy = math.sin(theta) * math.sin(a_phi)
                            vz = math.cos(theta)
                            rays.append(Ray(pos=np.array([px, py, 0.0]), dir=np.array([vx, vy, vz]), wavelength=wl))
                            
    print(f"\nTracing {len(rays)} rays...")
    traced = opt.trace_rays(rays)
    vignetted = sum(1 for r in traced if r.vignetted)
    print(f"Vignetted rays: {vignetted} / {len(rays)}")
    
    det_hits = []
    for r in traced:
        if not r.vignetted:
            dh = next((h for h in r.history if h["surface_index"] == 14), None)
            if dh:
                det_hits.append((dh["x"], dh["y"]))
                
    radii = [math.hypot(x, y) for x, y in det_hits]
    max_r = max(radii)
    rms_r = math.sqrt(np.mean(np.array(radii)**2))
    max_d = 2.0 * max_r
    print(f"Detector Max Spot Radius:   {max_r:.4f} mm ({max_r*1000.0:.1f} um)")
    print(f"Detector RMS Spot Radius:   {rms_r:.4f} mm ({rms_r*1000.0:.1f} um)")
    print(f"Detector Max Spot Diameter: {max_d:.4f} mm ({max_d*1000.0:.1f} um)")
    print(f"Spot <= 1.0 mm Spec satisfied: {max_d <= 1.000}")
    
    return stack

if __name__ == "__main__":
    test_single_design()
