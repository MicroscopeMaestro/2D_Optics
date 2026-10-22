import sys, os
sys.path.insert(0, os.path.abspath('.'))
import numpy as np
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from verify_layout import generate_fiber_phase_space_rays, RAMAN_WAVELENGTHS_UM

# Let's test a design where:
# 1. Primary wavelength is 1.38919 um (symmetric full band 1.1113 to 1.667 um)
# 2. Camera lens / Collimator are tuned so 1.1113 to 1.667 um fits within [-4.332, +4.332] mm on DMD
# 3. Relay + Condenser collection optics compress the spot to <= 0.500 mm radius on the detector

def build_symmetric_spectrometer(
    f_coll,
    f_cam_r1, f_cam_t1, f_cam_g1, f_cam_r2, f_cam_t2, f_cam_g2, f_cam_r3, f_dmd,
    dmd_r,
    relay_r1, relay_t1, relay_g1, relay_r2, relay_t2, relay_g2, relay_r3, r_c,
    c_d,
    pri_wl=1.38919
):
    stack = SurfaceStack()
    # 0. Fiber
    stack.add_surface(Surface(name="Fiber_FG025LJA", surface_type=SurfaceType.OBJECT, thickness=f_coll, semi_diameter=25.0))
    # 1-3. Collimator: AC254-050-C
    stack.add_surface(Surface(name="AC254-050-C_Coll_Front", radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Coll_Mid", radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Coll_Back", radius=-115.0, thickness=20.0, glass="AIR", semi_diameter=12.7))
    # 4. Grating: GR25-0316
    stack.add_surface(Surface(name="Grating_GR25-0316", surface_type=SurfaceType.GRATING, radius=0.0, thickness=20.0, glass="MIRROR", semi_diameter=12.7, tilt_y=12.0, grating_lines_per_mm=300.0, grating_order=1))
    # 5-7. Camera Focus
    stack.add_surface(Surface(name="Camera_Focus_Front", radius=f_cam_r1, thickness=f_cam_t1, glass=f_cam_g1, semi_diameter=12.7))
    stack.add_surface(Surface(name="Camera_Focus_Mid", radius=f_cam_r2, thickness=f_cam_t2, glass=f_cam_g2, semi_diameter=12.7))
    stack.add_surface(Surface(name="Camera_Focus_Back", radius=f_cam_r3, thickness=f_dmd, glass="AIR", semi_diameter=12.7))
    # 8. DMD: TI DMD4500
    stack.add_surface(Surface(name="TI_DMD4500", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=dmd_r, glass="MIRROR", semi_diameter=12.7, tilt_y=12.0))
    # 9-11. Relay Lens
    stack.add_surface(Surface(name="Relay_Front", radius=relay_r1, thickness=relay_t1, glass=relay_g1, semi_diameter=12.7))
    stack.add_surface(Surface(name="Relay_Mid", radius=relay_r2, thickness=relay_t2, glass=relay_g2, semi_diameter=12.7))
    stack.add_surface(Surface(name="Relay_Back", radius=relay_r3, thickness=r_c, glass="AIR", semi_diameter=12.7))
    # 12-13. Condenser: ACL12708U-C (semi_diameter=6.35 mm clear aperture)
    stack.add_surface(Surface(name="ACL12708U-C_Cond_Front", surface_type=SurfaceType.ASPHERIC, radius=4.16, thickness=4.5, glass="B270", semi_diameter=6.35, conic=-0.6265, asphere_coeffs=[1.5e-5]))
    stack.add_surface(Surface(name="ACL12708U-C_Cond_Back", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=c_d, glass="AIR", semi_diameter=6.35))
    # 14. Detector: InGaAs 1mm (semi_diameter=0.50 mm)
    stack.add_surface(Surface(name="Detector_InGaAs_1mm", surface_type=SurfaceType.DETECTOR, radius=0.0, thickness=0.0, glass="AIR", semi_diameter=0.50))
    
    stack.primary_wavelength = pri_wl
    stack._update_accumulated_positions()
    return stack

# Test AC254-050-C and AC254-040-C as camera lens
# Test AC254-075-C and AC254-050-C as relay lens
rays = generate_fiber_phase_space_rays(fiber_radius_mm=0.0125, numerical_aperture=0.12, wavelengths=RAMAN_WAVELENGTHS_UM)

# 1. With AC254-050-C as camera and AC254-075-C as relay
for pri_wl in [1.3800, 1.3850, 1.38919]:
    for f_dmd in [40.0, 42.0, 44.0]:
        for dmd_r in [15.0, 18.0, 20.0, 22.0]:
            for r_c in [3.0, 5.0, 7.0]:
                for c_d in [5.0, 5.2, 5.4]:
                    stack = build_symmetric_spectrometer(
                        f_coll=47.5,
                        f_cam_r1=34.0, f_cam_t1=6.0, f_cam_g1='N-BK7', f_cam_r2=-27.8, f_cam_t2=2.0, f_cam_g2='SF5', f_cam_r3=-115.0, f_dmd=f_dmd,
                        dmd_r=dmd_r,
                        relay_r1=49.3, relay_t1=5.2, relay_g1='N-BK7', relay_r2=-39.0, relay_t2=2.0, relay_g2='SF5', relay_r3=-170.0, r_c=r_c,
                        c_d=c_d,
                        pri_wl=pri_wl
                    )
                    opt = OpticalSystem(stack)
                    r1 = opt.trace_ray(Ray(pos=np.array([0., 0., 0.]), dir=np.array([0., 0., 1.]), wavelength=1.1113))
                    r2 = opt.trace_ray(Ray(pos=np.array([0., 0., 0.]), dir=np.array([0., 0., 1.]), wavelength=1.6133))
                    r3 = opt.trace_ray(Ray(pos=np.array([0., 0., 0.]), dir=np.array([0., 0., 1.]), wavelength=1.66708))
                    x1 = r1.history[8]['x']
                    x2 = r2.history[8]['x']
                    x3 = r3.history[8]['x']
                    span = abs(x2 - x1)
                    
                    traced = opt.trace_rays(rays)
                    det_idx = len(stack.surfaces) - 1
                    vig_optics = sum(1 for r in traced if r.vignetted and (not r.history or r.history[-1]['surface_index'] < det_idx))
                    det_hits = [(h['x'], h['y']) for r in traced for h in r.history if h['surface_index'] == det_idx and not (r.vignetted and r.history[-1]['surface_index'] < det_idx)]
                    
                    if vig_optics == 0 and len(det_hits) == len(rays):
                        rads = [np.sqrt(x**2 + y**2) for x, y in det_hits]
                        max_r = max(rads)
                        rms_r = np.sqrt(np.mean(np.array(rads)**2))
                        print(f"PASS: pri_wl={pri_wl:.4f}, f_dmd={f_dmd:.1f}, dmd_r={dmd_r:.1f}, r_c={r_c:.1f}, c_d={c_d:.1f} | span={span:.3f} mm [x1={x1:+.3f}, x2={x2:+.3f}, x3={x3:+.3f}] | max_r={max_r:.4f} mm (dia={max_r*2:.4f} mm), rms_r={rms_r:.4f} mm")
                        sys.stdout.flush()
