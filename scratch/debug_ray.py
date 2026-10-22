def build_thorlabs_raman_spectrometer(input_aperture_type="Circular", input_core_dia_um=50.0, input_na=0.12):
    stack = SurfaceStack()
    stack.primary_wavelength = 0.850
    core_radius_mm = (input_core_dia_um / 2000.0)

    # Surface 0: Fiber Output / Entrance Aperture (Object)
    s0 = Surface(
        name="Fiber_Core_Input",
        surface_type=SurfaceType.OBJECT,
        thickness=10.00,
        glass="AIR",
        semi_diameter=core_radius_mm,
        aperture_type=input_aperture_type,
        aperture_x_half=core_radius_mm,
        aperture_y_half=core_radius_mm if input_aperture_type == "Circular" else 0.50
    )
    stack.add_surface(s0)

    # Surface 1: Input Iris Stop (at z = 10.0 mm)
    stop_semi_diam = 18.50 * math.tan(math.asin(input_na))
    s1 = Surface(
        name="Input_Aperture_Stop",
        surface_type=SurfaceType.STANDARD,
        radius=0.0,
        thickness=8.50,
        glass="AIR",
        semi_diameter=stop_semi_diam,
        aperture_type="Circular",
        aperture_r_outer=stop_semi_diam
    )
    stack.add_surface(s1)
