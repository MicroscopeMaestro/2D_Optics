import os
import sys
import math
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath("."))

from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray

def test_stack(stack, name="Test Stack"):
    print("=" * 70)
    print(f"EVALUATING CONFIGURATION: {name}")
    print("=" * 70)
    stack._update_accumulated_positions()
    opt = OpticalSystem(stack)

    # 1. Chief rays at key wavelengths
    wls = [1.11130, 1.20000, 1.30000, 1.36200, 1.40000, 1.50000, 1.61330, 1.66708]
    chief_rays = [Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=w) for w in wls]
    chief_traced = opt.trace_rays(chief_rays)

    # Find DMD surface index
    dmd_idx = next(i for i, s in enumerate(stack.surfaces) if "DMD" in s.name.upper())
    det_idx = next(i for i, s in enumerate(stack.surfaces) if s.surface_type == SurfaceType.DETECTOR or "DETECTOR" in s.name.upper())

    dmd_hits = {}
    for r in chief_traced:
        hit = next(h for h in r.history if h["surface_index"] == dmd_idx)
        dmd_hits[r.wavelength] = hit["x"]
        print(f"Chief ray @ {r.wavelength:.5f} um -> DMD X = {hit['x']:7.3f} mm, Y = {hit['y']:7.3f} mm")

    dmd_span_1613 = abs(dmd_hits[1.61330] - dmd_hits[1.11130])
    dmd_span_1667 = abs(dmd_hits[1.66708] - dmd_hits[1.11130])
    print(f"\nDMD Chief ray span (1.11130 to 1.61330 um): {dmd_span_1613:.4f} mm")
    print(f"DMD Chief ray span (1.11130 to 1.66708 um): {dmd_span_1667:.4f} mm")
    print(f"DMD Spec [8.00 mm, 8.664 mm] satisfied: {8.00 <= dmd_span_1613 <= 8.664}")

    # 2. Phase-space rays from 25 um slit (NA = 0.10)
    na = 0.10
    max_theta = math.asin(na)
    slit_r = 0.0125  # 25 um slit width / 2

    test_rays = []
    for wl in wls:
        # Spatial positions (center and edges of 25 um slit)
        for px in [-slit_r, 0.0, slit_r]:
            for py in [-slit_r, 0.0, slit_r]:
                # Angular cone
                for ar in [0.0, 0.5, 1.0]:
                    if ar == 0.0:
                        test_rays.append(Ray(pos=np.array([px, py, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl))
                    else:
                        for phi in np.linspace(0, 2*np.pi, 8, endpoint=False):
                            theta = ar * max_theta
                            vx = math.sin(theta) * math.cos(phi)
                            vy = math.sin(theta) * math.sin(phi)
                            vz = math.cos(theta)
                            test_rays.append(Ray(pos=np.array([px, py, 0.0]), dir=np.array([vx, vy, vz]), wavelength=wl))

    print(f"\nTracing {len(test_rays)} rays across all wavelengths and field angles...")
    traced = opt.trace_rays(test_rays)
    vignetted = sum(1 for r in traced if r.vignetted)
    print(f"Vignetted rays: {vignetted} / {len(test_rays)}")

    det_hits = []
    for r in traced:
        if not r.vignetted:
            dh = next((h for h in r.history if h["surface_index"] == det_idx), None)
            if dh:
                det_hits.append((dh["x"], dh["y"]))

    if det_hits:
        radii = [math.hypot(x, y) for x, y in det_hits]
        max_r = max(radii)
        rms_r = math.sqrt(np.mean(np.array(radii)**2))
        max_d = 2.0 * max_r
        print(f"Detector Max Spot Radius:   {max_r:.4f} mm ({max_r*1000.0:.1f} um)")
        print(f"Detector RMS Spot Radius:   {rms_r:.4f} mm ({rms_r*1000.0:.1f} um)")
        print(f"Detector Max Spot Diameter: {max_d:.4f} mm ({max_d*1000.0:.1f} um)")
        print(f"Detector Spot <= 1.0 mm Spec satisfied: {max_d <= 1.000}")
    else:
        print("ERROR: No rays reached the detector!")

if __name__ == "__main__":
    # Test Config 1: Asphere Condenser (ACL12708U-C)
    stack1 = SurfaceStack()
    stack1.primary_wavelength = 1.36200
    stack1.add_surface(Surface(name="Slit_Source", surface_type=SurfaceType.OBJECT, thickness=47.5, glass="AIR", semi_diameter=12.7))
    stack1.add_surface(Surface(name="AC254-050-C_Coll_Front", radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
    stack1.add_surface(Surface(name="AC254-050-C_Coll_Mid", radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
    stack1.add_surface(Surface(name="AC254-050-C_Coll_Back", radius=-115.0, thickness=20.0, glass="AIR", semi_diameter=12.7))
    stack1.add_surface(Surface(name="Grating_GR25-0316", surface_type=SurfaceType.GRATING, radius=0.0, thickness=20.0, glass="MIRROR", semi_diameter=12.7, tilt_y=12.0, grating_lines_per_mm=300.0, grating_order=1))
    stack1.add_surface(Surface(name="AC254-050-C_Focus_Front", radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
    stack1.add_surface(Surface(name="AC254-050-C_Focus_Mid", radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
    stack1.add_surface(Surface(name="AC254-050-C_Focus_Back", radius=-115.0, thickness=44.0, glass="AIR", semi_diameter=12.7))
    stack1.add_surface(Surface(name="TI_DMD4500", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=20.0, glass="MIRROR", semi_diameter=12.7, tilt_y=12.0))
    stack1.add_surface(Surface(name="AC508-075-C_Relay_Front", radius=49.3, thickness=8.0, glass="N-BK7", semi_diameter=25.4))
    stack1.add_surface(Surface(name="AC508-075-C_Relay_Mid", radius=-39.0, thickness=3.0, glass="SF5", semi_diameter=25.4))
    stack1.add_surface(Surface(name="AC508-075-C_Relay_Back", radius=-170.0, thickness=5.0, glass="AIR", semi_diameter=25.4))
    stack1.add_surface(Surface(name="ACL12708U-C_Cond_Front", surface_type=SurfaceType.ASPHERIC, radius=4.16, thickness=4.5, glass="B270", semi_diameter=12.7, conic=-0.6265, asphere_coeffs=[1.5e-5]))
    stack1.add_surface(Surface(name="ACL12708U-C_Cond_Back", radius=0.0, thickness=5.2, glass="AIR", semi_diameter=12.7))
    stack1.add_surface(Surface(name="Detector_InGaAs_1mm", surface_type=SurfaceType.DETECTOR, radius=0.0, thickness=0.0, glass="AIR", semi_diameter=0.50))
    test_stack(stack1, "Config 1: 75mm Relay + ACL12708U-C Aspheric Condenser")
