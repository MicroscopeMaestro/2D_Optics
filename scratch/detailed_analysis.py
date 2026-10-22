import sys, os
sys.path.insert(0, os.path.abspath('.'))
import numpy as np
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from verify_layout import generate_fiber_phase_space_rays, RAMAN_WAVELENGTHS_UM

def analyze_system(
    f_coll=47.5,
    c_g=20.0,
    g_f=20.0,
    f_dmd=44.0,
    dmd_relay=20.0,
    relay_type='AC254-075-C',
    relay_rev=False,
    relay_cond=5.0,
    cond_det=5.2,
    primary_wl=1.3620,
    grating_tilt=12.0,
    dmd_tilt=12.0,
    dense=False
):
    stack = SurfaceStack()
    stack.add_surface(Surface(name="Fiber_FG025LJA", surface_type=SurfaceType.OBJECT, thickness=f_coll, semi_diameter=25.0))
    stack.add_surface(Surface(name="AC254-050-C_Coll_Front", radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Coll_Mid", radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Coll_Back", radius=-115.0, thickness=c_g, glass="AIR", semi_diameter=12.7))
    stack.add_surface(Surface(name="Grating_GR25-0316", surface_type=SurfaceType.GRATING, radius=0.0, thickness=g_f, glass="MIRROR", semi_diameter=12.7, tilt_y=grating_tilt, grating_lines_per_mm=300.0, grating_order=1))
    stack.add_surface(Surface(name="AC254-050-C_Focus_Front", radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Focus_Mid", radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Focus_Back", radius=-115.0, thickness=f_dmd, glass="AIR", semi_diameter=12.7))
    stack.add_surface(Surface(name="TI_DMD4500", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=dmd_relay, glass="MIRROR", semi_diameter=12.7, tilt_y=dmd_tilt))
    
    # Relay Lens
    if relay_type == 'AC254-075-C':
        if not relay_rev:
            stack.add_surface(Surface(name="AC254-075-C_Relay_Front", radius=49.3, thickness=5.2, glass="N-BK7", semi_diameter=12.7))
            stack.add_surface(Surface(name="AC254-075-C_Relay_Mid", radius=-39.0, thickness=2.0, glass="SF5", semi_diameter=12.7))
            stack.add_surface(Surface(name="AC254-075-C_Relay_Back", radius=-170.0, thickness=relay_cond, glass="AIR", semi_diameter=12.7))
        else:
            stack.add_surface(Surface(name="AC254-075-C_Relay_Front", radius=170.0, thickness=2.0, glass="SF5", semi_diameter=12.7))
            stack.add_surface(Surface(name="AC254-075-C_Relay_Mid", radius=39.0, thickness=5.2, glass="N-BK7", semi_diameter=12.7))
            stack.add_surface(Surface(name="AC254-075-C_Relay_Back", radius=-49.3, thickness=relay_cond, glass="AIR", semi_diameter=12.7))
    elif relay_type == 'AC254-050-C':
        if not relay_rev:
            stack.add_surface(Surface(name="AC254-050-C_Relay_Front", radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
            stack.add_surface(Surface(name="AC254-050-C_Relay_Mid", radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
            stack.add_surface(Surface(name="AC254-050-C_Relay_Back", radius=-115.0, thickness=relay_cond, glass="AIR", semi_diameter=12.7))
        else:
            stack.add_surface(Surface(name="AC254-050-C_Relay_Front", radius=115.0, thickness=2.0, glass="SF5", semi_diameter=12.7))
            stack.add_surface(Surface(name="AC254-050-C_Relay_Mid", radius=27.8, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
            stack.add_surface(Surface(name="AC254-050-C_Relay_Back", radius=-34.0, thickness=relay_cond, glass="AIR", semi_diameter=12.7))
            
    # Condenser (ACL12708U-C)
    stack.add_surface(Surface(name="ACL12708U-C_Cond_Front", surface_type=SurfaceType.ASPHERIC, radius=4.16, thickness=4.5, glass="B270", semi_diameter=12.7, conic=-0.6265, asphere_coeffs=[1.5e-5]))
    stack.add_surface(Surface(name="ACL12708U-C_Cond_Back", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=cond_det, glass="AIR", semi_diameter=12.7))
    # Detector
    stack.add_surface(Surface(name="Detector_InGaAs_1mm", surface_type=SurfaceType.DETECTOR, radius=0.0, thickness=0.0, glass="AIR", semi_diameter=0.50))

    stack.primary_wavelength = primary_wl
    stack._update_accumulated_positions()
    
    opt = OpticalSystem(stack)
    rays = generate_fiber_phase_space_rays(fiber_radius_mm=0.0125, numerical_aperture=0.12, wavelengths=RAMAN_WAVELENGTHS_UM, dense_mode=dense)
    traced = opt.trace_rays(rays)
    det_idx = len(stack.surfaces) - 1
    
    print(f"\n=======================================================")
    print(f"Detailed Analysis: Relay={relay_type} (rev={relay_rev}), dmd_r={dmd_relay}, r_c={relay_cond}, c_d={cond_det}")
    print(f"=======================================================")
    
    det_hits_per_wl = {}
    vig_per_surf = {i: 0 for i in range(len(stack.surfaces))}
    
    for wl in RAMAN_WAVELENGTHS_UM:
        det_hits_per_wl[wl] = []
        
    for r in traced:
        wl = r.wavelength
        if r.vignetted:
            last_idx = r.history[-1]['surface_index'] if r.history else 0
            vig_per_surf[last_idx] += 1
        else:
            h = next((h for h in r.history if h['surface_index'] == det_idx), None)
            if h:
                det_hits_per_wl[wl].append((h['x'], h['y']))
                
    total_det_hits = sum(len(v) for v in det_hits_per_wl.values())
    print(f"Total Rays: {len(rays)}, Received at Detector: {total_det_hits} ({total_det_hits/len(rays)*100:.2f}%)")
    print(f"Vignetting per surface:")
    for s_i, count in vig_per_surf.items():
        if count > 0:
            print(f"  Surf {s_i:2d} ({stack.surfaces[s_i].name}): {count} rays blocked")
            
    print("\nPer-wavelength metrics at Detector:")
    all_radii = []
    for wl in RAMAN_WAVELENGTHS_UM:
        hits = det_hits_per_wl[wl]
        if hits:
            radii = [np.sqrt(x**2 + y**2) for x, y in hits]
            all_radii.extend(radii)
            mean_x = np.mean([x for x, y in hits])
            mean_y = np.mean([y for x, y in hits])
            max_r = np.max(radii)
            rms_r = np.sqrt(np.mean(np.array(radii)**2))
            print(f"  wl={wl:.5f} um: count={len(hits):3d} | centroid=({mean_x:+.4f}, {mean_y:+.4f}) mm | rms_r={rms_r:.4f} mm | max_r={max_r:.4f} mm (dia={max_r*2:.4f} mm)")
        else:
            print(f"  wl={wl:.5f} um: 0 rays at detector!")
            
    if all_radii:
        print(f"\nOverall Max Spot Radius: {max(all_radii):.4f} mm (Diameter: {max(all_radii)*2:.4f} mm)")
        print(f"Overall RMS Spot Radius: {np.sqrt(np.mean(np.array(all_radii)**2)):.4f} mm")
    sys.stdout.flush()

if __name__ == '__main__':
    # Test a few hand-tuned designs
    analyze_system(dmd_relay=20.0, relay_type='AC254-075-C', relay_rev=False, relay_cond=5.0, cond_det=5.2)
    analyze_system(dmd_relay=20.0, relay_type='AC254-050-C', relay_rev=False, relay_cond=5.0, cond_det=5.2)
