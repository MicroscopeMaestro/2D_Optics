"""
Empirical challenger test script to stress-test M2 optics engine implementation.
"""
import numpy as np
from optics_app.engine.surface import AsphericSurface, Surface, SurfaceType
from optics_app.engine.ray import Ray
from optics_app.engine.tracer import snell_refract_3d, OpticalSystem
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.grating import grating_diffract_3d, DiffractionGrating
from optics_app.engine.dispersion import SellmeierDispersion, CauchyDispersion, get_refractive_index
from optics_app.io.zmx_parser import parse_zmx_string

def test_aspheric_normal_and_intersection():
    print("--- Testing AsphericSurface Normal & Intersection ---")
    # Parabolic surface sag z(r) = r^2 / (2 * R) where R = 50mm, conic = -1
    # At r = 10mm, z = 100 / 100 = 1.0mm
    # Surface gradient dz/dr = r / R = 10 / 50 = 0.2
    # The surface normal vector at (10, 0, 1.0) should be normalized (-dz/dr, 0, 1) = (-0.2, 0, 1) / sqrt(1.04) = (-0.196, 0, 0.980)
    asph = AsphericSurface(radius=50.0, conic=-1.0)
    ray = Ray(pos=(10.0, 0.0, -10.0), dir=(0.0, 0.0, 1.0))
    hit_pos, normal = asph.intersect(ray)
    print(f"Ray hit_pos: {hit_pos}")
    print(f"Returned normal: {normal}")
    
    # Expected normal z component should NOT be 1.0 for a curved surface off-axis!
    expected_normal_x = -0.2 / np.sqrt(1 + 0.2**2)
    expected_normal_z = 1.0 / np.sqrt(1 + 0.2**2)
    print(f"True physical normal at r=10: ({expected_normal_x:.4f}, 0, {expected_normal_z:.4f})")
    is_facade = np.allclose(normal, [0.0, 0.0, 1.0])
    print(f"Is normal hardcoded facade [0, 0, 1]? {is_facade}")
    return is_facade

def test_grating_3d_line_vector_and_normal():
    print("\n--- Testing 3D Grating line_vector & normal usage ---")
    # Test case A: Grating with lines along Y (line_vector = [0, 1, 0]).
    # Periodicity is along X. Ray parallel to Z [0, 0, 1].
    # Diffraction should change v_x!
    v_in = [0.0, 0.0, 1.0]
    N = [0.0, 0.0, 1.0]
    
    # Lines along Y axis -> line_vector = [0, 1, 0]
    L_along_y = [0.0, 1.0, 0.0]
    v_out_y, evan_y = grating_diffract_3d(v_in, N, L_along_y, line_density_lines_per_mm=600.0, order=1, wavelength_um=0.5)
    print(f"With line_vector=[0, 1, 0] (grooves along Y): v_out = {v_out_y}")

    # Lines along X axis -> line_vector = [1, 0, 0]
    # Periodicity is along Y. Diffraction SHOULD change v_y, NOT v_x!
    L_along_x = [1.0, 0.0, 0.0]
    v_out_x, evan_x = grating_diffract_3d(v_in, N, L_along_x, line_density_lines_per_mm=600.0, order=1, wavelength_um=0.5)
    print(f"With line_vector=[1, 0, 0] (grooves along X): v_out = {v_out_x}")
    
    ignores_line_vector = np.allclose(v_out_y, v_out_x)
    print(f"Does grating_diffract_3d give identical output regardless of line_vector? {ignores_line_vector}")
    return ignores_line_vector

def test_surface_tilt():
    print("\n--- Testing Surface Tilt ---")
    # Create surface with tilt_x = 45 degrees
    stack = SurfaceStack()
    stack.add_surface(Surface("S1", radius=0.0, thickness=10.0, tilt_x=45.0))
    opt_sys = OpticalSystem(stack)
    ray = Ray(pos=(0.0, 0.0, -10.0), dir=(0.0, 0.0, 1.0))
    traced = opt_sys.trace_ray(ray)
    # If surface is tilted 45 deg around X, normal should be (0, -sin(45), cos(45)), ray should refract/deflect
    # But because tracer ignores tilt_x, ray direction stays [0, 0, 1]
    print(f"Traced ray dir after 45 deg tilted surface: {traced.dir}")
    ignores_tilt = np.allclose(traced.dir, [0.0, 0.0, 1.0])
    print(f"Is surface tilt completely ignored by ray tracer? {ignores_tilt}")
    return ignores_tilt

def test_grating_sign_convention_mismatch():
    print("\n--- Testing Grating Sign Convention Mismatch ---")
    g = DiffractionGrating(lines_per_mm=600.0, order=1)
    wl = 0.5 # 500 nm
    theta_i = np.radians(10.0) # +10 deg incidence angle
    # calculate_diffraction_angle: sin_m = (m*lambda/d) - sin_i
    ang1 = g.calculate_diffraction_angle(theta_i, wl)
    # grating_diffract_3d: sin_m = sin_i + (m*lambda/d)
    v_in = [np.sin(theta_i), 0.0, np.cos(theta_i)]
    v_out, _ = grating_diffract_3d(v_in, [0,0,1], [1,0,0], 600.0, 1, wl)
    ang2 = np.arcsin(v_out[0])
    print(f"DiffractionGrating.calculate_diffraction_angle for theta_i=+10 deg: {np.degrees(ang1):.4f} deg (sin_m = {np.sin(ang1):.4f})")
    print(f"grating_diffract_3d angle for theta_i=+10 deg: {np.degrees(ang2):.4f} deg (sin_m = {np.sin(ang2):.4f})")
    mismatch = not np.isclose(np.sin(ang1), np.sin(ang2))
    print(f"Is there a sign mismatch for incident angle between the two grating functions? {mismatch}")
    return mismatch

if __name__ == "__main__":
    t1 = test_aspheric_normal_and_intersection()
    t2 = test_grating_3d_line_vector_and_normal()
    t3 = test_surface_tilt()
    t4 = test_grating_sign_convention_mismatch()
