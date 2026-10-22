import sys, os, math, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray

def build_1inch_raman_spectrometer(input_core_dia_um=50.0, input_na=0.12):
    stack = SurfaceStack()
    stack.primary_wavelength = 0.850
    core_radius_mm = (input_core_dia_um / 2000.0)

    # Surface 0: Fiber Output (Object)
    s0 = Surface(
        name="Fiber_Core_Input",
        surface_type=SurfaceType.OBJECT,
        thickness=15.00,
        glass="AIR",
        semi_diameter=core_radius_mm,
        aperture_type="Circular",
        aperture_x_half=core_radius_mm,
        aperture_y_half=core_radius_mm
    )
    stack.add_surface(s0)

    # Surface 1: 1-Inch Input Iris Aperture Stop (SM1D12, Ø 7.2 mm clear aperture)
    stop_semi_diam = 28.50 * math.tan(math.asin(input_na))
    s1 = Surface(
        name="1in_Input_Aperture_Stop",
        surface_type=SurfaceType.STANDARD,
        radius=0.0,
        thickness=13.50,
        glass="AIR",
        semi_diameter=12.70,
        aperture_type="Circular",
        aperture_r_outer=stop_semi_diam
    )
    stack.add_surface(s1)

    # Surfaces 2-4: Thorlabs AC254-030-B (1" NIR Achromat, Dia 25.4 mm, f = 30.0 mm)
    s2 = Surface(name="Thorlabs_AC254-030-B_Front", radius=18.50, thickness=7.50, glass="N-BK7", semi_diameter=12.70, aperture_type="Circular", aperture_r_outer=11.43)
    s3 = Surface(name="Thorlabs_AC254-030-B_Mid", radius=-14.80, thickness=2.00, glass="SF5", semi_diameter=12.70, aperture_type="Circular", aperture_r_outer=11.43)
    s4 = Surface(name="Thorlabs_AC254-030-B_Back", radius=-57.20, thickness=25.00, glass="AIR", semi_diameter=12.70, aperture_type="Circular", aperture_r_outer=11.43)
    stack.add_surface(s2)
    stack.add_surface(s3)
    stack.add_surface(s4)

    # Surface 5: Thorlabs GR25-0608 (1" NIR Reflective Grating, 25x25 mm, 600 l/mm)
    s5 = Surface(name="Thorlabs_GR25-0608_Grating", surface_type=SurfaceType.GRATING, radius=0.0, thickness=35.00, glass="MIRROR", semi_diameter=12.70, tilt_x=30.00, grating_lines_per_mm=600.0, grating_order=1, aperture_type="Rectangular", aperture_x_half=12.50, aperture_y_half=12.50)
    stack.add_surface(s5)

    # Surfaces 6-8: Thorlabs AC254-050-B (1" NIR Achromat, Dia 25.4 mm, f = 50.0 mm)
    s6 = Surface(name="Thorlabs_AC254-050-B_Front", radius=34.00, thickness=6.00, glass="N-BK7", semi_diameter=12.70, aperture_type="Circular", aperture_r_outer=11.43)
    s7 = Surface(name="Thorlabs_AC254-050-B_Mid", radius=-27.80, thickness=2.00, glass="SF5", semi_diameter=12.70, aperture_type="Circular", aperture_r_outer=11.43)
    s8 = Surface(name="Thorlabs_AC254-050-B_Back", radius=-115.00, thickness=47.50, glass="AIR", semi_diameter=12.70, aperture_type="Circular", aperture_r_outer=11.43)
    stack.add_surface(s6)
    stack.add_surface(s7)
    stack.add_surface(s8)

    # Surface 9: TI DLP4500 DMD (Active Area 9.855 x 6.16 mm)
    s9 = Surface(name="TI_DLP4500_DMD_Array", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=100.00, glass="MIRROR", semi_diameter=12.70, tilt_x=30.00, aperture_type="Rectangular", aperture_x_half=4.9275, aperture_y_half=3.0800)
    stack.add_surface(s9)

    # Surfaces 10-12: Thorlabs AC254-100-B (Reversed 1" NIR Achromat, Dia 25.4 mm, f = 100.0 mm)
    s10 = Surface(name="Thorlabs_AC254-100-B_Relay_Front", radius=198.00, thickness=2.00, glass="SF5", semi_diameter=12.70, aperture_type="Circular", aperture_r_outer=11.43)
    s11 = Surface(name="Thorlabs_AC254-100-B_Relay_Mid", radius=44.50, thickness=5.00, glass="N-BK7", semi_diameter=12.70, aperture_type="Circular", aperture_r_outer=11.43)
    s12 = Surface(name="Thorlabs_AC254-100-B_Relay_Back", radius=-61.20, thickness=10.00, glass="AIR", semi_diameter=12.70, aperture_type="Circular", aperture_r_outer=11.43)
    stack.add_surface(s10)
    stack.add_surface(s11)
    stack.add_surface(s12)

    # Surfaces 13-14: Thorlabs ACL25416U-B (1" Precision CNC Aspheric Condenser, Dia 25.4 mm, f = 16.0 mm, NA = 0.68)
    s13 = Surface(name="Thorlabs_ACL25416U-B_Condenser_Front", surface_type=SurfaceType.ASPHERIC, radius=8.32, thickness=12.00, glass="B270", semi_diameter=12.70, conic=-0.6265, asphere_coeffs=[1.8e-6], aperture_type="Circular", aperture_r_outer=11.00)
    s14 = Surface(name="Thorlabs_ACL25416U-B_Condenser_Back", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=8.45, glass="AIR", semi_diameter=12.70, aperture_type="Circular", aperture_r_outer=11.00)
    stack.add_surface(s13)
    stack.add_surface(s14)

    # Surface 15: Detector 1mm^2
    s15 = Surface(name="Detector_1mm2", surface_type=SurfaceType.DETECTOR, radius=0.0, thickness=0.00, glass="AIR", semi_diameter=0.50, aperture_type="Rectangular", aperture_x_half=0.50, aperture_y_half=0.50)
    stack.add_surface(s15)

    stack._update_accumulated_positions()
    return stack

stack = build_1inch_raman_spectrometer()
opt_sys = OpticalSystem(stack)
ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=0.785)
res = opt_sys.trace_ray(ray)
print("Ray vignetted:", res.vignetted)
for i, h in enumerate(res.history):
    print(f"Step {i}: Surf {h.get('surface_index')}, name={stack.surfaces[h.get('surface_index')].name}, pos=({h.get('x',0):.3f}, {h.get('y',0):.3f}, {h.get('z',0):.3f}), x_loc={h.get('x_local',0):.3f}, y_loc={h.get('y_local',0):.3f}, vignetted={h.get('vignetted', False)}")
