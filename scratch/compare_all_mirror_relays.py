import sys, os, math, json
import numpy as np

sys.path.insert(0, r"c:\Users\Juan\Documents\GitHub\2D_Optics")
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray

def test_full_system(relay_type="2-mirror", f1=50.0, f2=75.0, f3=150.0, f4=15.0, f_single=25.0, m_single=0.15):
    print("=" * 80)
    print(f"ANALYZING ALL-MIRROR SPECTROMETER WITH RELAY TYPE: {relay_type.upper()}")
    print("=" * 80)

    stack = SurfaceStack()
    wls = [0.785, 0.808, 0.830, 0.850, 0.880]
    stack.primary_wavelength = 0.830
    stack.wavelengths = wls

    # S0: Slit
    stack.add_surface(Surface(
        name="Input_Slit_30um",
        surface_type=SurfaceType.OBJECT,
        thickness=48.5,
        glass="AIR",
        semi_diameter=1.0,
        aperture_type="Rectangular",
        aperture_x_half=0.015,
        aperture_y_half=0.500
    ))

    # S1: Aperture Stop
    stack.add_surface(Surface(
        name="Thorlabs_SM1D12_Stop",
        surface_type=SurfaceType.STOP,
        thickness=1.5,
        glass="AIR",
        semi_diameter=5.0,
        aperture_type="Circular",
        aperture_r_outer=5.0
    ))

    # S2: Collimator M1 (f1=50mm, ROC=-100mm, Dia=25.4mm)
    stack.add_surface(Surface(
        name="M1_CM254-050_Collimator",
        surface_type=SurfaceType.STANDARD,
        radius=-2.0 * f1,
        thickness=50.0,
        glass="MIRROR",
        semi_diameter=12.7,
        tilt_x=8.5,
        substrate_thickness=6.0,
        aperture_type="Circular",
        aperture_r_outer=12.0
    ))

    # S3: Grating (600 l/mm, 25x25mm)
    stack.add_surface(Surface(
        name="Grating_GR25-0608_600lpmm",
        surface_type=SurfaceType.GRATING,
        radius=0.0,
        thickness=60.0,
        glass="MIRROR",
        semi_diameter=12.5,
        tilt_x=18.2,
        grating_lines_per_mm=600.0,
        grating_order=1,
        substrate_thickness=6.0,
        aperture_type="Rectangular",
        aperture_x_half=12.5,
        aperture_y_half=12.5
    ))

    # S4: Focusing Mirror M2 (f2=75mm, ROC=-150mm, Dia=25.4mm)
    stack.add_surface(Surface(
        name="M2_CM254-075_Focusing",
        surface_type=SurfaceType.STANDARD,
        radius=-2.0 * f2,
        thickness=73.5, # Fine tuned focal position
        glass="MIRROR",
        semi_diameter=12.7,
        tilt_x=-8.5, # Coma compensation
        substrate_thickness=6.0,
        aperture_type="Circular",
        aperture_r_outer=12.0
    ))

    # S5: DMD4500 (5 mm active area)
    if relay_type == "2-mirror":
        d_dmd_to_m3 = f3 # 150.0 mm
        stack.add_surface(Surface(
            name="DMD4500_Micromirrors",
            surface_type=SurfaceType.STANDARD,
            radius=0.0,
            thickness=d_dmd_to_m3,
            glass="MIRROR",
            semi_diameter=2.5,
            tilt_x=12.0, # ON-state tilt
            substrate_thickness=3.0,
            aperture_type="Rectangular",
            aperture_x_half=2.5,
            aperture_y_half=2.5
        ))

        # S6: M3 Relay Collimator (f3=150mm, ROC=-300mm)
        stack.add_surface(Surface(
            name="M3_CM254-150_Relay_Collector",
            surface_type=SurfaceType.STANDARD,
            radius=-2.0 * f3,
            thickness=50.0, # Compact folded distance to M4
            glass="MIRROR",
            semi_diameter=12.7,
            tilt_x=10.0,
            substrate_thickness=6.0,
            aperture_type="Circular",
            aperture_r_outer=12.0
        ))

        # S7: M4 Relay Condenser (f4=15mm, ROC=-30mm)
        stack.add_surface(Surface(
            name="M4_CM254-015_Relay_Condenser",
            surface_type=SurfaceType.STANDARD,
            radius=-2.0 * f4,
            thickness=14.8, # Distance to detector
            glass="MIRROR",
            semi_diameter=12.7,
            tilt_x=-10.0,
            substrate_thickness=6.0,
            aperture_type="Circular",
            aperture_r_outer=12.0
        ))
    else:
        # Single Mirror Relay
        # s1 = f_single * (1 + 1/m_single), s2 = f_single * (1 + m_single)
        s1 = f_single * (1.0 + 1.0 / m_single) # e.g. 25 * (1 + 1/0.15) = 191.67 mm
        s2 = f_single * (1.0 + m_single) # 25 * 1.15 = 28.75 mm
        stack.add_surface(Surface(
            name="DMD4500_Micromirrors",
            surface_type=SurfaceType.STANDARD,
            radius=0.0,
            thickness=s1,
            glass="MIRROR",
            semi_diameter=2.5,
            tilt_x=12.0,
            substrate_thickness=3.0,
            aperture_type="Rectangular",
            aperture_x_half=2.5,
            aperture_y_half=2.5
        ))

        stack.add_surface(Surface(
            name="M3_CM254-025_Single_Relay",
            surface_type=SurfaceType.STANDARD,
            radius=-2.0 * f_single, # -50.0 mm
            thickness=s2,
            glass="MIRROR",
            semi_diameter=12.7,
            tilt_x=8.0,
            substrate_thickness=6.0,
            aperture_type="Circular",
            aperture_r_outer=12.0
        ))

    # Detector (0.8 mm diameter, semi_diameter = 0.40 mm)
    stack.add_surface(Surface(
        name="Detector_0.8mm_IMA",
        surface_type=SurfaceType.DETECTOR,
        thickness=0.0,
        glass="AIR",
        semi_diameter=0.40,
        aperture_type="Circular",
        aperture_r_outer=0.40
    ))

    stack._update_accumulated_positions()
    opt_sys = OpticalSystem(stack)

    # Ray trace: generate ray grid
    rays = []
    theta_max = math.asin(5.0 / 50.0) # NA defined by aperture stop (r_stop = 5.0 mm at f = 50 mm -> NA = 0.10)
    for wl in wls:
        for y_obj in [-0.015, 0.0, 0.015]: # Slit width
            for x_obj in [-0.25, 0.0, 0.25]: # Slit height
                # Chief ray
                rays.append(Ray(pos=np.array([x_obj, y_obj, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl))
                # Pupil rings
                for ring in [0.4, 0.7, 0.95]:
                    for phi in np.linspace(0, 2*np.pi, 8, endpoint=False):
                        th = ring * theta_max
                        kx = math.sin(th) * math.cos(phi)
                        ky = math.sin(th) * math.sin(phi)
                        kz = math.cos(th)
                        rays.append(Ray(pos=np.array([x_obj, y_obj, 0.0]), dir=np.array([kx, ky, kz]), wavelength=wl))

    print(f"Tracing {len(rays)} rays...")
    traced = [opt_sys.trace_ray(r) for r in rays]
    vignetted_count = sum(1 for tr in traced if tr.vignetted)
    valid = [tr for tr in traced if not tr.vignetted]
    print(f"Traced: {len(valid)}/{len(rays)} transmitted ({len(valid)/len(rays)*100:.1f}%)")

    # Audit DMD
    dmd_idx = 5
    dmd_hits_x = [h["x_local"] for tr in valid for h in tr.history if h["surface_index"] == dmd_idx]
    dmd_hits_y = [h["y_local"] for tr in valid for h in tr.history if h["surface_index"] == dmd_idx]
    if dmd_hits_x:
        print(f"DMD Footprint:")
        print(f"  X span: [{min(dmd_hits_x):.3f}, {max(dmd_hits_x):.3f}] mm (Width = {max(dmd_hits_x)-min(dmd_hits_x):.3f} mm)")
        print(f"  Y span: [{min(dmd_hits_y):.3f}, {max(dmd_hits_y):.3f}] mm (Height = {max(dmd_hits_y)-min(dmd_hits_y):.3f} mm)")

    # Audit Detector
    det_idx = len(stack.surfaces) - 1
    det_hits_x = [h["x_local"] for tr in valid for h in tr.history if h["surface_index"] == det_idx]
    det_hits_y = [h["y_local"] for tr in valid for h in tr.history if h["surface_index"] == det_idx]
    if det_hits_x:
        r_hits = np.sqrt(np.array(det_hits_x)**2 + np.array(det_hits_y)**2)
        print(f"Detector Footprint (Surface {det_idx}):")
        print(f"  Max Radius: {np.max(r_hits):.4f} mm (Detector radius limit: 0.4000 mm)")
        print(f"  RMS Radius: {np.sqrt(np.mean(r_hits**2)):.4f} mm")
        print(f"  100% rays collected within Ø0.8 mm detector: {np.max(r_hits) <= 0.4000}")

    # Clearance audit between origins
    print("\nGlobal Component Positions & Clearances:")
    for i, s in enumerate(stack.surfaces):
        pos = s.global_origin
        print(f"  S{i:02d} ({s.name:<32}): Origin = [{pos[0]:7.2f}, {pos[1]:7.2f}, {pos[2]:7.2f}] mm")

    return stack

print("Evaluating 2-Mirror Relay...")
test_full_system(relay_type="2-mirror")

print("\n\nEvaluating Single-Mirror Relay...")
test_full_system(relay_type="single", f_single=25.0, m_single=0.15)
