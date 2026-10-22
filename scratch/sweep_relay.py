import sys, os, json, numpy as np
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.surface import Surface, SurfaceType
from v2.engine.lde import SurfaceStack
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

def test_hybrid_relay(d_dmd_relay=45.0, d_relay_cond=10.0, d_cond_det=4.85, use_30mm_focus=True):
    stack = SurfaceStack()
    stack.primary_wavelength = 1.225
    stack.wavelengths = [1.087, 1.225, 1.350]
    
    # 0. Slit
    t0 = 24.5 if use_30mm_focus else 45.0
    stack.add_surface(Surface(name="Input_Slit_30um", surface_type=SurfaceType.OBJECT, thickness=t0, glass="AIR", semi_diameter=12.7))
    
    # Collimator: AC254-030-C (f=30mm)
    stack.add_surface(Surface(name="Collimator_S1", surface_type=SurfaceType.STANDARD, radius=71.12, thickness=1.8, glass="N-SF6HT", semi_diameter=12.7))
    stack.add_surface(Surface(name="Collimator_S2", surface_type=SurfaceType.STANDARD, radius=15.24, thickness=13.0, glass="N-BAF10", semi_diameter=12.7))
    stack.add_surface(Surface(name="Collimator_S3", surface_type=SurfaceType.STANDARD, radius=-21.09, thickness=25.0, glass="AIR", semi_diameter=12.7))
    
    # Aperture Stop
    stack.add_surface(Surface(name="Aperture_Stop", surface_type=SurfaceType.STOP, thickness=25.0, glass="AIR", semi_diameter=5.0))
    
    # Out of plane fold: 90 deg Z rotation
    stack.add_surface(Surface(name="CB_Before_Grating", surface_type=SurfaceType.STANDARD, is_coord_break=True, thickness=0.0, glass="AIR", semi_diameter=12.7, tilt_z=90.0))
    
    # Grating 600 l/mm, tilt_x = -45 deg
    stack.add_surface(Surface(name="Grating_600lpmm", surface_type=SurfaceType.GRATING, radius=0.0, thickness=0.0, glass="MIRROR", semi_diameter=12.7, tilt_x=-45.0, grating_lines_per_mm=600.0, grating_order=1, grating_primary_wavelength_um=1.225))
    
    # Coordinate break after grating for 1.225 um central wavelength
    stack.add_surface(Surface(name="CB_After_Grating", surface_type=SurfaceType.STANDARD, is_coord_break=True, thickness=45.0, glass="AIR", semi_diameter=12.7, tilt_x=46.598371024629536))
    
    if use_30mm_focus:
        # Focusing Lens AC254-030-C (f=30mm)
        stack.add_surface(Surface(name="Focusing_S1", surface_type=SurfaceType.STANDARD, radius=21.09, thickness=13.0, glass="N-BAF10", semi_diameter=12.7))
        stack.add_surface(Surface(name="Focusing_S2", surface_type=SurfaceType.STANDARD, radius=-15.24, thickness=1.8, glass="N-SF6HT", semi_diameter=12.7))
        stack.add_surface(Surface(name="Focusing_S3", surface_type=SurfaceType.STANDARD, radius=-71.12, thickness=24.5, glass="AIR", semi_diameter=12.7))
    else:
        # Focusing Lens AC254-050-C (f=50mm)
        stack.add_surface(Surface(name="Focusing_S1", surface_type=SurfaceType.STANDARD, radius=34.5, thickness=12.0, glass="N-BAF10", semi_diameter=12.7))
        stack.add_surface(Surface(name="Focusing_S2", surface_type=SurfaceType.STANDARD, radius=-23.1, thickness=2.0, glass="N-SF6HT", semi_diameter=12.7))
        stack.add_surface(Surface(name="Focusing_S3", surface_type=SurfaceType.STANDARD, radius=-135.2, thickness=45.0, glass="AIR", semi_diameter=12.7))
        
    # DMD4500 (tilted -12 deg)
    stack.add_surface(Surface(name="DMD4500_Micromirrors", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=0.0, glass="MIRROR", semi_diameter=5.0, tilt_y=-12.0))
    
    # CB after DMD
    stack.add_surface(Surface(name="CB_After_DMD", surface_type=SurfaceType.STANDARD, is_coord_break=True, thickness=float(d_dmd_relay), glass="AIR", semi_diameter=12.7))
    
    # Relay Lens AC254-075-C (f=75mm)
    stack.add_surface(Surface(name="Relay_S1", surface_type=SurfaceType.STANDARD, radius=49.3, thickness=8.0, glass="N-BK7", semi_diameter=12.7))
    stack.add_surface(Surface(name="Relay_S2", surface_type=SurfaceType.STANDARD, radius=-39.0, thickness=3.0, glass="SF5", semi_diameter=12.7))
    stack.add_surface(Surface(name="Relay_S3", surface_type=SurfaceType.STANDARD, radius=-170.0, thickness=float(d_relay_cond), glass="AIR", semi_diameter=12.7))
    
    # Condenser ACL12708U-C (f=8mm)
    stack.add_surface(Surface(name="Cond_S1", surface_type=SurfaceType.ASPHERIC, radius=4.16, thickness=4.5, glass="B270", semi_diameter=6.35, conic=-0.6265, asphere_coeffs=[1.5e-5]))
    stack.add_surface(Surface(name="Cond_S2", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=float(d_cond_det), glass="AIR", semi_diameter=6.35))
    
    # Detector (0.8mm diameter -> semi_diameter 0.40mm)
    stack.add_surface(Surface(name="Detector", surface_type=SurfaceType.DETECTOR, radius=0.0, thickness=0.0, glass="AIR", semi_diameter=0.40))
    
    sys_opt = OpticalSystem(stack)
    sys_opt._update_accumulated_positions()
    return sys_opt

# Test sweep
print("Testing relay sweep...")
best_max_r = 999.0
best_p = None

for d_dmd in np.linspace(35, 60, 6):
    for d_mid in np.linspace(5, 25, 5):
        for d_det in np.linspace(3.5, 6.5, 7):
            sys_opt = test_hybrid_relay(d_dmd, d_mid, d_det, use_30mm_focus=True)
            
            wls = [1.087, 1.225, 1.350]
            hits_x, hits_y = [], []
            vignettes = 0
            total_rays = 0
            
            theta_max = np.arcsin(0.04)
            for wl in wls:
                # Chief ray
                r0 = Ray(pos=np.array([0., 0., 0.]), dir=np.array([0., 0., 1.]), wavelength=wl)
                tr0 = sys_opt.trace_ray(r0)
                total_rays += 1
                if tr0.vignetted:
                    vignettes += 1
                else:
                    h = tr0.history[-1]
                    hits_x.append(h['x_local'])
                    hits_y.append(h['y_local'])
                    
                for theta in np.linspace(0.01, theta_max, 2):
                    for phi in [0, np.pi/2, np.pi, 3*np.pi/2]:
                        dx = np.sin(theta)*np.cos(phi)
                        dy = np.sin(theta)*np.sin(phi)
                        dz = np.cos(theta)
                        r = Ray(pos=np.array([0., 0., 0.]), dir=np.array([dx, dy, dz]), wavelength=wl)
                        tr = sys_opt.trace_ray(r)
                        total_rays += 1
                        if tr.vignetted:
                            vignettes += 1
                        else:
                            h = tr.history[-1]
                            hits_x.append(h['x_local'])
                            hits_y.append(h['y_local'])
                            
            if vignettes == 0 and len(hits_x) == total_rays:
                mx, my = np.mean(hits_x), np.mean(hits_y)
                r_sq = (np.array(hits_x) - mx)**2 + (np.array(hits_y) - my)**2
                max_r = np.max(np.sqrt(r_sq))
                r_origin = np.max(np.sqrt(np.array(hits_x)**2 + np.array(hits_y)**2))
                if max_r < best_max_r:
                    best_max_r = max_r
                    best_p = (d_dmd, d_mid, d_det, max_r, r_origin)
                    print(f"NEW BEST (0 vig!): d_dmd={d_dmd:.1f}, d_mid={d_mid:.1f}, d_det={d_det:.2f} -> Max spot radius (centroid)={max_r:.4f} mm, Max radius (origin)={r_origin:.4f} mm")

if best_p:
    print(f"\nFinal Best: {best_p}")
else:
    print("No zero-vignetting solution in coarse sweep, let's refine.")
