import sys, os, math, json
import numpy as np

sys.path.insert(0, r"c:\Users\Juan\Documents\GitHub\2D_Optics")
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray

def build_all_mirror_czerny_turner_spectrometer(
    f1=50.0, f2=75.0, f3=150.0, f4=15.0,
    theta1=8.0, alpha=18.2, theta2=-8.0,
    theta_dmd=12.0, theta3=8.0, theta4=-8.0,
    wls=[0.785, 0.808, 0.830, 0.850, 0.880],
    primary_wl=0.830
):
    """
    Builds a 100% All-Mirror Spectrometer Layout using standard Thorlabs spherical mirrors.
    
    Sequence of Optical Elements:
    S0: Input Slit (30 um width, NA = 0.10)
    S1: Input Aperture Stop (Iris SM1D12, semi_diameter = 5.5 mm)
    S2: Collimating Mirror M1 (Thorlabs CM254-050, ROC = -100 mm, f1 = 50 mm, tilt = theta1)
    S3: Diffraction Grating (Thorlabs GR25-0608, 600 l/mm, tilt = alpha)
    S4: Focusing Mirror M2 (Thorlabs CM254-075, ROC = -150 mm, f2 = 75 mm, tilt = theta2)
    S5: DMD4500 Micromirror Array (5 mm active area, tilt = theta_dmd = +12 deg ON state)
    S6: Relay Collimating Mirror M3 (Thorlabs CM254-150, ROC = -300 mm, f3 = 150 mm, tilt = theta3)
    S7: Relay Condensing Mirror M4 (Thorlabs CM254-015, ROC = -30 mm, f4 = 15 mm, tilt = theta4)
    S8: Single-Element Detector (0.8 mm diameter, semi_diameter = 0.40 mm)
    """
    stack = SurfaceStack()
    stack.name = "Thorlabs_All_Mirror_Spectrometer_0.8mm_Detector"
    stack.primary_wavelength = primary_wl
    stack.wavelengths = wls

    # S0: Object Slit
    stack.add_surface(Surface(
        name="Input_Slit_30um",
        surface_type=SurfaceType.OBJECT,
        thickness=f1 - 2.0, # Distance to aperture stop
        glass="AIR",
        semi_diameter=0.015, # 30 um / 2
        aperture_type="Rectangular",
        aperture_x_half=0.015,
        aperture_y_half=1.000
    ))

    # S1: Aperture Stop
    stack.add_surface(Surface(
        name="Thorlabs_SM1D12_Aperture_Stop",
        surface_type=SurfaceType.STOP,
        thickness=2.0, # Distance to M1
        glass="AIR",
        semi_diameter=5.5, # Controls beam diameter to 11.0 mm
        aperture_type="Circular",
        aperture_r_outer=5.5
    ))

    # S2: Collimator M1 (Thorlabs CM254-050-E03, ROC = -100 mm, f = 50 mm)
    stack.add_surface(Surface(
        name="M1_Thorlabs_CM254-050_Collimator",
        surface_type=SurfaceType.STANDARD,
        radius=-2.0 * f1, # -100.0 mm
        thickness=50.0, # Distance M1 -> Grating
        glass="MIRROR",
        semi_diameter=12.7, # 1-inch optic
        tilt_x=theta1,
        substrate_thickness=6.0,
        substrate_shape="Circular",
        aperture_type="Circular",
        aperture_r_outer=12.0
    ))

    # S3: Diffraction Grating (Thorlabs GR25-0608, 600 l/mm)
    stack.add_surface(Surface(
        name="Grating_Thorlabs_GR25-0608_600lpmm",
        surface_type=SurfaceType.GRATING,
        radius=0.0,
        thickness=55.0, # Distance Grating -> M2
        glass="MIRROR",
        semi_diameter=12.5,
        tilt_x=alpha,
        grating_lines_per_mm=600.0,
        grating_order=1,
        substrate_thickness=6.0,
        substrate_shape="Rectangular",
        aperture_type="Rectangular",
        aperture_x_half=12.5,
        aperture_y_half=12.5
    ))

    # S4: Focusing Mirror M2 (Thorlabs CM254-075-E03, ROC = -150 mm, f = 75 mm)
    stack.add_surface(Surface(
        name="M2_Thorlabs_CM254-075_Focusing",
        surface_type=SurfaceType.STANDARD,
        radius=-2.0 * f2, # -150.0 mm
        thickness=f2, # 75.0 mm to DMD focal plane
        glass="MIRROR",
        semi_diameter=12.7,
        tilt_x=theta2,
        substrate_thickness=6.0,
        substrate_shape="Circular",
        aperture_type="Circular",
        aperture_r_outer=12.0
    ))

    # S5: DMD4500 Active Area (5 mm, ON state tilt = 12 deg)
    stack.add_surface(Surface(
        name="DMD4500_Micromirror_Array_5mm",
        surface_type=SurfaceType.STANDARD,
        radius=0.0,
        thickness=f3, # 150.0 mm to M3
        glass="MIRROR",
        semi_diameter=2.5, # 5.0 mm active width / 2
        tilt_x=theta_dmd, # ON state tilt
        substrate_thickness=3.0,
        substrate_shape="Rectangular",
        aperture_type="Rectangular",
        aperture_x_half=2.5,
        aperture_y_half=2.5
    ))

    # S6: Relay Mirror M3 (Thorlabs CM254-150-E03, ROC = -300 mm, f = 150 mm)
    stack.add_surface(Surface(
        name="M3_Thorlabs_CM254-150_Relay_Collector",
        surface_type=SurfaceType.STANDARD,
        radius=-2.0 * f3, # -300.0 mm
        thickness=f3 + f4, # Distance M3 -> M4 = 165.0 mm (or compact folded spacing)
        glass="MIRROR",
        semi_diameter=12.7,
        tilt_x=theta3,
        substrate_thickness=6.0,
        substrate_shape="Circular",
        aperture_type="Circular",
        aperture_r_outer=12.0
    ))

    # S7: Relay Mirror M4 (Thorlabs CM254-015-E03, ROC = -30 mm, f = 15 mm)
    stack.add_surface(Surface(
        name="M4_Thorlabs_CM254-015_Relay_Condenser",
        surface_type=SurfaceType.STANDARD,
        radius=-2.0 * f4, # -30.0 mm
        thickness=f4, # 15.0 mm to Detector
        glass="MIRROR",
        semi_diameter=12.7,
        tilt_x=theta4,
        substrate_thickness=6.0,
        substrate_shape="Circular",
        aperture_type="Circular",
        aperture_r_outer=12.0
    ))

    # S8: Detector (0.8 mm diameter, semi_diameter = 0.40 mm)
    stack.add_surface(Surface(
        name="Detector_0.8mm_IMA",
        surface_type=SurfaceType.DETECTOR,
        thickness=0.0,
        glass="AIR",
        semi_diameter=0.40, # 0.8 mm diameter / 2
        aperture_type="Circular",
        aperture_r_outer=0.40
    ))

    stack._update_accumulated_positions()
    return stack

def test_ray_tracing():
    wls = [0.785, 0.808, 0.830, 0.850, 0.880]
    stack = build_all_mirror_czerny_turner_spectrometer(
        f1=50.0, f2=75.0, f3=150.0, f4=15.0,
        theta1=8.0, alpha=18.2, theta2=-8.0,
        theta_dmd=12.0, theta3=8.0, theta4=-8.0,
        wls=wls
    )
    opt_sys = OpticalSystem(stack)

    # Generate ray fan from 30 um slit with NA = 0.10
    theta_max = math.asin(0.10)
    rays = []
    for wl in wls:
        for y_obj in [-0.015, 0.0, 0.015]: # Slit spatial width
            for x_obj in [-0.25, 0.0, 0.25]: # Slit height
                # Chief ray
                rays.append(Ray(pos=np.array([x_obj, y_obj, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl))
                # Pupil rays
                for ring in [0.5, 1.0]:
                    for phi in np.linspace(0, 2*np.pi, 8, endpoint=False):
                        th = ring * theta_max
                        kx = math.sin(th) * math.cos(phi)
                        ky = math.sin(th) * math.sin(phi)
                        kz = math.cos(th)
                        rays.append(Ray(pos=np.array([x_obj, y_obj, 0.0]), dir=np.array([kx, ky, kz]), wavelength=wl))

    print(f"Tracing {len(rays)} rays through 8-surface all-mirror system...")
    traced = opt_sys.trace_rays(rays)
    vignetted_count = sum(1 for tr in traced if tr.vignetted)
    valid_count = len(traced) - vignetted_count
    print(f"Ray Trace Results: {valid_count}/{len(traced)} valid ({valid_count/len(traced)*100:.1f}%)")

    # Evaluate DMD footprint
    dmd_idx = 5
    dmd_hits_x = [h["x_local"] for tr in traced for h in tr.history if h["surface_index"] == dmd_idx and not tr.vignetted]
    dmd_hits_y = [h["y_local"] for tr in traced for h in tr.history if h["surface_index"] == dmd_idx and not tr.vignetted]
    if dmd_hits_x:
        print(f"\nDMD Footprint (Surface {dmd_idx}):")
        print(f"  X span: [{min(dmd_hits_x):.3f}, {max(dmd_hits_x):.3f}] mm (Total = {max(dmd_hits_x)-min(dmd_hits_x):.3f} mm)")
        print(f"  Y span: [{min(dmd_hits_y):.3f}, {max(dmd_hits_y):.3f}] mm (Total = {max(dmd_hits_y)-min(dmd_hits_y):.3f} mm)")

    # Evaluate Detector footprint
    det_idx = 8
    det_hits_x = [h["x_local"] for tr in traced for h in tr.history if h["surface_index"] == det_idx and not tr.vignetted]
    det_hits_y = [h["y_local"] for tr in traced for h in tr.history if h["surface_index"] == det_idx and not tr.vignetted]
    if det_hits_x:
        r_hits = np.sqrt(np.array(det_hits_x)**2 + np.array(det_hits_y)**2)
        print(f"\nDetector Footprint (Surface {det_idx}):")
        print(f"  Max Radius: {np.max(r_hits):.4f} mm (Limit: <= 0.4000 mm)")
        print(f"  RMS Radius: {np.sqrt(np.mean(r_hits**2)):.4f} mm")
        print(f"  100% light inside 0.8 mm detector: {np.max(r_hits) <= 0.4000}")

test_ray_tracing()
