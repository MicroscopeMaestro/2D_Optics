import sys, os
sys.path.insert(0, os.path.abspath('.'))
import numpy as np
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from verify_layout import generate_fiber_phase_space_rays, RAMAN_WAVELENGTHS_UM

# Let's test the collection optics standalone starting from DMD
# At DMD, we have rays across x in [-4.3, +4.3] mm with NA=0.12 divergence
def test_collection_standalone(
    dmd_relay,
    relay_type,
    relay_rev,
    relay_cond,
    cond_det,
    primary_wl=1.38919
):
    stack = SurfaceStack()
    # Front end
    stack.add_surface(Surface(name="Fiber_FG025LJA", surface_type=SurfaceType.OBJECT, thickness=47.5, semi_diameter=25.0))
    stack.add_surface(Surface(name="AC254-050-C_Coll_Front", radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Coll_Mid", radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Coll_Back", radius=-115.0, thickness=20.0, glass="AIR", semi_diameter=12.7))
    stack.add_surface(Surface(name="Grating_GR25-0316", surface_type=SurfaceType.GRATING, radius=0.0, thickness=20.0, glass="MIRROR", semi_diameter=12.7, tilt_y=12.0, grating_lines_per_mm=300.0, grating_order=1))
    stack.add_surface(Surface(name="AC254-050-C_Focus_Front", radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Focus_Mid", radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Focus_Back", radius=-115.0, thickness=44.0, glass="AIR", semi_diameter=12.7))
    stack.add_surface(Surface(name="TI_DMD4500", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=dmd_relay, glass="MIRROR", semi_diameter=12.7, tilt_y=12.0))
    
    # Relay Lens options
    if relay_type == 'AC254-075-C':
        if not relay_rev:
            stack.add_surface(Surface(name="Relay_F", radius=49.3, thickness=5.2, glass="N-BK7", semi_diameter=12.7))
            stack.add_surface(Surface(name="Relay_M", radius=-39.0, thickness=2.0, glass="SF5", semi_diameter=12.7))
            stack.add_surface(Surface(name="Relay_B", radius=-170.0, thickness=relay_cond, glass="AIR", semi_diameter=12.7))
        else:
            stack.add_surface(Surface(name="Relay_F", radius=170.0, thickness=2.0, glass="SF5", semi_diameter=12.7))
            stack.add_surface(Surface(name="Relay_M", radius=39.0, thickness=5.2, glass="N-BK7", semi_diameter=12.7))
            stack.add_surface(Surface(name="Relay_B", radius=-49.3, thickness=relay_cond, glass="AIR", semi_diameter=12.7))
    elif relay_type == 'AC254-050-C':
        if not relay_rev:
            stack.add_surface(Surface(name="Relay_F", radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
            stack.add_surface(Surface(name="Relay_M", radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
            stack.add_surface(Surface(name="Relay_B", radius=-115.0, thickness=relay_cond, glass="AIR", semi_diameter=12.7))
        else:
            stack.add_surface(Surface(name="Relay_F", radius=115.0, thickness=2.0, glass="SF5", semi_diameter=12.7))
            stack.add_surface(Surface(name="Relay_M", radius=27.8, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
            stack.add_surface(Surface(name="Relay_B", radius=-34.0, thickness=relay_cond, glass="AIR", semi_diameter=12.7))
    elif relay_type == 'AC254-030-C':
        if not relay_rev:
            stack.add_surface(Surface(name="Relay_F", radius=18.5, thickness=7.5, glass="N-BK7", semi_diameter=12.7))
            stack.add_surface(Surface(name="Relay_M", radius=-14.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
            stack.add_surface(Surface(name="Relay_B", radius=-57.2, thickness=relay_cond, glass="AIR", semi_diameter=12.7))
        else:
            stack.add_surface(Surface(name="Relay_F", radius=57.2, thickness=2.0, glass="SF5", semi_diameter=12.7))
            stack.add_surface(Surface(name="Relay_M", radius=14.8, thickness=7.5, glass="N-BK7", semi_diameter=12.7))
            stack.add_surface(Surface(name="Relay_B", radius=-18.5, thickness=relay_cond, glass="AIR", semi_diameter=12.7))

    # Condenser (ACL12708U-C)
    stack.add_surface(Surface(name="ACL12708U-C_Cond_Front", surface_type=SurfaceType.ASPHERIC, radius=4.16, thickness=4.5, glass="B270", semi_diameter=12.7, conic=-0.6265, asphere_coeffs=[1.5e-5]))
    stack.add_surface(Surface(name="ACL12708U-C_Cond_Back", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=cond_det, glass="AIR", semi_diameter=12.7))
    # Detector (semi_diameter=25.0)
    stack.add_surface(Surface(name="Detector_InGaAs_1mm", surface_type=SurfaceType.DETECTOR, radius=0.0, thickness=0.0, glass="AIR", semi_diameter=25.0))

    stack.primary_wavelength = primary_wl
    stack._update_accumulated_positions()
    return stack

if __name__ == '__main__':
    rays = generate_fiber_phase_space_rays(fiber_radius_mm=0.0125, numerical_aperture=0.12, wavelengths=RAMAN_WAVELENGTHS_UM)
    print(f"Loaded {len(rays)} rays.")
    
    best_configs = []
    
    # We want to find dmd_relay, relay_type, relay_rev, relay_cond, cond_det that minimizes max spot radius
    for relay_type in ['AC254-075-C', 'AC254-050-C', 'AC254-030-C']:
        for relay_rev in [False, True]:
            for dmd_r in np.arange(10.0, 45.0, 5.0):
                for r_c in np.arange(1.0, 20.0, 3.0):
                    for c_d in np.arange(4.0, 7.0, 0.5):
                        stack = test_collection_standalone(dmd_r, relay_type, relay_rev, r_c, c_d, primary_wl=1.3620)
                        opt = OpticalSystem(stack)
                        traced = opt.trace_rays(rays)
                        det_idx = len(stack.surfaces) - 1
                        
                        vig_optics = sum(1 for r in traced if r.vignetted and (not r.history or r.history[-1]['surface_index'] < det_idx))
                        if vig_optics > 0:
                            continue
                            
                        det_hits = [(h['x'], h['y']) for r in traced for h in r.history if h['surface_index'] == det_idx and not (r.vignetted and r.history[-1]['surface_index'] < det_idx)]
                        if len(det_hits) == len(rays):
                            radii = [np.sqrt(x**2 + y**2) for x, y in det_hits]
                            max_r = max(radii)
                            rms_r = np.sqrt(np.mean(np.array(radii)**2))
                            best_configs.append((max_r, rms_r, relay_type, relay_rev, dmd_r, r_c, c_d))
                            if max_r <= 0.60:
                                print(f"  Good: max_r={max_r:.4f} mm | Relay: {relay_type} (rev={relay_rev}), dmd_r={dmd_r:.1f}, r_c={r_c:.1f}, c_d={c_d:.1f}")
                                sys.stdout.flush()

    best_configs.sort(key=lambda x: x[0])
    print(f"\nTop 10 configurations:")
    for max_r, rms_r, rtype, rrev, dmd_r, r_c, c_d in best_configs[:10]:
        print(f"  max_r = {max_r:.4f} mm (dia = {max_r*2:.4f} mm), rms_r = {rms_r:.4f} mm | Relay: {rtype} (rev={rrev}) | dmd_r={dmd_r:.1f}, r_c={r_c:.1f}, c_d={c_d:.1f}")
    sys.stdout.flush()
