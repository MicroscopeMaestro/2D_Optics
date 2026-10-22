"""
Test & simulate 15-surface Hybrid Raman 1064nm Spectrometer in 2D_Optics.
"""
import sys
import os
import json
import math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from optics_app.engine.source import SourceConfig, SourceType
from optics_app.analysis.spot import SpotDiagram

def test_prescription():
    stack = SurfaceStack()
    stack.primary_wavelength = 1.3892

    # Surface 0: Fiber Input (Object)
    # 25 um core -> semi_diameter = 0.0125 mm
    s0 = Surface(
        name="Fiber_FG025LJA",
        surface_type=SurfaceType.OBJECT,
        radius=0.0,
        thickness=45.0, # distance to collimator
        glass="AIR",
        semi_diameter=25.0
    )
    stack.add_surface(s0)

    # Surfaces 1-3: Thorlabs AC254-050-C (f = 50.0 mm Collimator)
    s1 = Surface(
        name="AC254-050-C_Coll_Front",
        surface_type=SurfaceType.STANDARD,
        radius=34.0,
        thickness=6.0,
        glass="N-BK7",
        semi_diameter=12.7
    )
    s2 = Surface(
        name="AC254-050-C_Coll_Mid",
        surface_type=SurfaceType.STANDARD,
        radius=-27.8,
        thickness=2.0,
        glass="SF5",
        semi_diameter=12.7
    )
    s3 = Surface(
        name="AC254-050-C_Coll_Back",
        surface_type=SurfaceType.STANDARD,
        radius=-115.0,
        thickness=15.0, # distance to grating
        glass="AIR",
        semi_diameter=12.7
    )
    stack.add_surface(s1)
    stack.add_surface(s2)
    stack.add_surface(s3)

    # Surface 4: Thorlabs GR25-0316 (300 l/mm Grating)
    # Blaze 1600nm, gold coated, alpha = 12 deg tilt
    s4 = Surface(
        name="Grating_GR25-0316",
        surface_type=SurfaceType.GRATING,
        radius=0.0,
        thickness=15.0, # distance to focusing lens
        glass="MIRROR",
        semi_diameter=12.7,
        tilt_y=12.0, # or tilt_x
        grating_lines_per_mm=300.0,
        grating_order=1
    )
    stack.add_surface(s4)

    # Surfaces 5-7: Thorlabs AC254-040-C (f = 40.0 mm Focusing Doublet)
    s5 = Surface(
        name="AC254-040-C_Focus_Front",
        surface_type=SurfaceType.STANDARD,
        radius=25.1,
        thickness=6.8,
        glass="N-BK7",
        semi_diameter=12.7
    )
    s6 = Surface(
        name="AC254-040-C_Focus_Mid",
        surface_type=SurfaceType.STANDARD,
        radius=-20.2,
        thickness=2.0,
        glass="SF5",
        semi_diameter=12.7
    )
    s7 = Surface(
        name="AC254-040-C_Focus_Back",
        surface_type=SurfaceType.STANDARD,
        radius=-84.0,
        thickness=36.5, # back focal distance to DMD
        glass="AIR",
        semi_diameter=12.7
    )
    stack.add_surface(s5)
    stack.add_surface(s6)
    stack.add_surface(s7)

    # Surface 8: Texas Instruments DLP4500 (DMD)
    s8 = Surface(
        name="TI_DMD4500",
        surface_type=SurfaceType.STANDARD,
        radius=0.0,
        thickness=45.0, # distance to relay lens
        glass="MIRROR",
        semi_diameter=12.7,
        tilt_y=12.0, # or tilt_x
        aperture_type="Rectangular",
        aperture_x_half=4.332,
        aperture_y_half=2.705
    )
    stack.add_surface(s8)

    # Surfaces 9-11: Thorlabs AC508-075-C (f = 75.0 mm 2" Relay Doublet)
    s9 = Surface(
        name="AC508-075-C_Relay_Front",
        surface_type=SurfaceType.STANDARD,
        radius=49.3,
        thickness=8.0,
        glass="N-BK7",
        semi_diameter=25.4 # 2" lens
    )
    s10 = Surface(
        name="AC508-075-C_Relay_Mid",
        surface_type=SurfaceType.STANDARD,
        radius=-39.0,
        thickness=3.0,
        glass="SF5",
        semi_diameter=25.4
    )
    s11 = Surface(
        name="AC508-075-C_Relay_Back",
        surface_type=SurfaceType.STANDARD,
        radius=-170.0,
        thickness=10.0, # distance to condenser
        glass="AIR",
        semi_diameter=25.4
    )
    stack.add_surface(s9)
    stack.add_surface(s10)
    stack.add_surface(s11)

    # Surfaces 12-13: Thorlabs ACL12708U-C (f = 8.0 mm Aspheric Condenser)
    s12 = Surface(
        name="ACL12708U-C_Cond_Front",
        surface_type=SurfaceType.ASPHERIC,
        radius=4.16,
        thickness=4.5,
        glass="B270",
        semi_diameter=12.7,
        conic=-0.6265,
        asphere_coeffs=[1.5e-5]
    )
    s13 = Surface(
        name="ACL12708U-C_Cond_Back",
        surface_type=SurfaceType.STANDARD,
        radius=0.0,
        thickness=4.85, # back focal distance to detector
        glass="AIR",
        semi_diameter=12.7
    )
    stack.add_surface(s12)
    stack.add_surface(s13)

    # Surface 14: Single-element InGaAs Detector (1.0 mm dia -> semi_diameter = 0.5 mm)
    s14 = Surface(
        name="Detector_InGaAs_1mm",
        surface_type=SurfaceType.DETECTOR,
        radius=0.0,
        thickness=0.0,
        glass="AIR",
        semi_diameter=0.5
    )
    stack.add_surface(s14)

    print(f"Total surfaces created: {len(stack.surfaces)}")
    stack._update_accumulated_positions()

    wavelengths = [1.1113, 1.2000, 1.3000, 1.3892, 1.5000, 1.6000, 1.6671]
    
    # Generate fiber rays
    na = 0.12
    max_theta = math.asin(na)
    fiber_r = 0.0125
    rays = []
    for wl in wavelengths:
        # Chief ray
        rays.append(Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl))
        # Marginal rays
        for angle in [max_theta, -max_theta]:
            rays.append(Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([math.sin(angle), 0.0, math.cos(angle)]), wavelength=wl))
            rays.append(Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, math.sin(angle), math.cos(angle)]), wavelength=wl))
        for pos_r in [fiber_r, -fiber_r]:
            rays.append(Ray(pos=np.array([pos_r, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl))
            rays.append(Ray(pos=np.array([0.0, pos_r, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl))

    opt = OpticalSystem(stack)
    traced = opt.trace_rays(rays)

    vignetted = [r for r in traced if r.vignetted]
    print(f"Total rays: {len(rays)}, Vignetted: {len(vignetted)}, Transmission: {(len(rays)-len(vignetted))/len(rays)*100:.2f}%")

    # Analyze DMD (Surface 8)
    dmd_hits = {}
    for r in traced:
        for h in r.history:
            if h["surface_index"] == 8:
                dmd_hits.setdefault(r.wavelength, []).append((h["x"], h["y"]))

    print("\nDMD Hits by wavelength:")
    all_dmd_x = []
    for wl, pts in dmd_hits.items():
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        all_dmd_x.extend(xs)
        print(f"  WL {wl:.4f} um: x in [{min(xs):.3f}, {max(xs):.3f}], y in [{min(ys):.3f}, {max(ys):.3f}]")

    if all_dmd_x:
        print(f"DMD Total X Span: {max(all_dmd_x) - min(all_dmd_x):.3f} mm (Active width: 8.664 mm)")

    # Analyze Detector (Surface 14)
    det_hits = {}
    for r in traced:
        for h in r.history:
            if h["surface_index"] == 14:
                det_hits.setdefault(r.wavelength, []).append((h["x"], h["y"]))

    print("\nDetector Hits by wavelength:")
    all_det_r = []
    for wl, pts in det_hits.items():
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        rs = [math.sqrt(x**2 + y**2) for x, y in zip(xs, ys)]
        all_det_r.extend(rs)
        print(f"  WL {wl:.4f} um: x in [{min(xs):.4f}, {max(xs):.4f}], max radius = {max(rs):.4f} mm")

    if all_det_r:
        print(f"\nDetector Max Radius across all rays: {max(all_det_r):.4f} mm (Target <= 0.500 mm / Diameter <= 1.000 mm)")

if __name__ == "__main__":
    test_prescription()
