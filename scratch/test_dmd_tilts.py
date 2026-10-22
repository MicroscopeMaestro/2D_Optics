import sys, os, json, numpy as np
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.surface import Surface, SurfaceType
from v2.engine.lde import SurfaceStack
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

def test_config(dmd_tilt_y=12.0, d_dmd_relay=70.0, d_relay_cond=20.0, d_cond_det=5.5, use_30mm_focus=True, f_relay=75.0):
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
        
    # DMD4500 (tilted +/-12 deg)
    stack.add_surface(Surface(name="DMD4500_Micromirrors", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=0.0, glass="MIRROR", semi_diameter=5.0, tilt_y=float(dmd_tilt_y)))
    
    # CB after DMD
    stack.add_surface(Surface(name="CB_After_DMD", surface_type=SurfaceType.STANDARD, is_coord_break=True, thickness=float(d_dmd_relay), glass="AIR", semi_diameter=12.7))
    
    if f_relay == 75.0:
        # Relay Lens AC254-075-C (f=75mm)
        stack.add_surface(Surface(name="Relay_S1", surface_type=SurfaceType.STANDARD, radius=49.3, thickness=8.0, glass="N-BK7", semi_diameter=12.7))
        stack.add_surface(Surface(name="Relay_S2", surface_type=SurfaceType.STANDARD, radius=-39.0, thickness=3.0, glass="SF5", semi_diameter=12.7))
        stack.add_surface(Surface(name="Relay_S3", surface_type=SurfaceType.STANDARD, radius=-170.0, thickness=float(d_relay_cond), glass="AIR", semi_diameter=12.7))
    elif f_relay == 100.0:
        # Relay Lens AC508-100-C / AC254-100-C (f=100mm)
        stack.add_surface(Surface(name="Relay_S1", surface_type=SurfaceType.STANDARD, radius=61.8, thickness=7.5, glass="N-BK7", semi_diameter=25.4))
        stack.add_surface(Surface(name="Relay_S2", surface_type=SurfaceType.STANDARD, radius=-49.2, thickness=3.5, glass="SF5", semi_diameter=25.4))
        stack.add_surface(Surface(name="Relay_S3", surface_type=SurfaceType.STANDARD, radius=-250.0, thickness=float(d_relay_cond), glass="AIR", semi_diameter=25.4))
        
    # Condenser ACL12708U-C (f=8mm)
    stack.add_surface(Surface(name="Cond_S1", surface_type=SurfaceType.ASPHERIC, radius=4.16, thickness=4.5, glass="B270", semi_diameter=6.35, conic=-0.6265, asphere_coeffs=[1.5e-5]))
    stack.add_surface(Surface(name="Cond_S2", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=float(d_cond_det), glass="AIR", semi_diameter=6.35))
    
    # Detector (0.8mm diameter -> semi_diameter 0.40mm)
    stack.add_surface(Surface(name="Detector", surface_type=SurfaceType.DETECTOR, radius=0.0, thickness=0.0, glass="AIR", semi_diameter=0.40))
    
    sys_opt = OpticalSystem(stack)
    sys_opt._update_accumulated_positions()
    return sys_opt

# Test both DMD tilt directions
for dmd_tilt in [12.0, -12.0]:
    for f_rel in [75.0, 100.0]:
        for d_dmd in [50.0, 75.0, 100.0]:
            sys_opt = test_config(dmd_tilt_y=dmd_tilt, d_dmd_relay=d_dmd, d_relay_cond=20.0, d_cond_det=5.5, use_30mm_focus=True, f_relay=f_rel)
            
            # Check collisions
            components = {
                'Collimator': {'surfs': [1, 2, 3], 'radius': 12.7},
                'Grating': {'surfs': [6], 'radius': 12.7},
                'Focusing': {'surfs': [8, 9, 10], 'radius': 12.7},
                'DMD4500': {'surfs': [11], 'radius': 12.7},
                'Relay': {'surfs': [13, 14, 15], 'radius': 12.7 if f_rel==75.0 else 25.4},
                'Condenser': {'surfs': [16, 17], 'radius': 6.35},
            }
            for comp_name, comp_info in components.items():
                centers = [sys_opt.surfaces[idx].global_origin for idx in comp_info['surfs']]
                comp_info['center'] = np.mean(centers, axis=0)

            # Trace chief and marginal rays
            wls = [1.087, 1.225, 1.350]
            rays = []
            for wl in wls:
                rays.append(Ray(pos=np.array([0., 0., 0.]), dir=np.array([0., 0., 1.]), wavelength=wl))
                theta_max = np.arcsin(0.04)
                for theta in [theta_max]:
                    for phi in [0, np.pi/2, np.pi, 3*np.pi/2]:
                        dx = np.sin(theta)*np.cos(phi)
                        dy = np.sin(theta)*np.sin(phi)
                        dz = np.cos(theta)
                        rays.append(Ray(pos=np.array([0., 0., 0.]), dir=np.array([dx, dy, dz]), wavelength=wl))

            traced = [sys_opt.trace_ray(r) for r in rays]
            vignetted_count = sum(1 for tr in traced if tr.vignetted)
            
            # Check clearance
            min_margin = 999.0
            min_center_dist = 999.0
            for tr in traced:
                for seg_idx in range(len(tr.history) - 1):
                    h1 = tr.history[seg_idx]
                    h2 = tr.history[seg_idx+1]
                    p1 = np.array([h1['x_global'], h1['y_global'], h1['z_global']])
                    p2 = np.array([h2['x_global'], h2['y_global'], h2['z_global']])
                    s1_idx = h1['surface_index']
                    s2_idx = h2['surface_index']
                    
                    seg_vec = p2 - p1
                    seg_len = np.linalg.norm(seg_vec)
                    if seg_len < 1e-6: continue
                    seg_dir = seg_vec / seg_len
                    
                    for comp_name, comp_info in components.items():
                        comp_surfs = set(comp_info['surfs'])
                        if comp_name == 'Grating': comp_surfs.update([5, 7])
                        elif comp_name == 'DMD4500': comp_surfs.update([12])
                        if s1_idx in comp_surfs or s2_idx in comp_surfs: continue
                        
                        comp_center = comp_info['center']
                        req_dist = comp_info['radius'] + 5.0
                        
                        v = comp_center - p1
                        t = np.clip(np.dot(v, seg_dir), 0.0, seg_len)
                        closest_pt = p1 + t * seg_dir
                        dist = np.linalg.norm(comp_center - closest_pt)
                        
                        margin_clr = dist - req_dist
                        if dist < min_center_dist: min_center_dist = dist
                        if margin_clr < min_margin: min_margin = margin_clr

            # Spot size
            hits_x = [tr.history[-1]['x_local'] for tr in traced if not tr.vignetted]
            hits_y = [tr.history[-1]['y_local'] for tr in traced if not tr.vignetted]
            if hits_x:
                mx, my = np.mean(hits_x), np.mean(hits_y)
                max_r = np.max(np.sqrt((np.array(hits_x)-mx)**2 + (np.array(hits_y)-my)**2))
                r_origin = np.max(np.sqrt(np.array(hits_x)**2 + np.array(hits_y)**2))
            else:
                max_r = 999.0
                r_origin = 999.0
                
            print(f"tilt={dmd_tilt:5.1f}, f_rel={f_rel:5.1f}, d_dmd={d_dmd:5.1f} -> Vig={vignetted_count:2d}/{len(rays)}, Max_R={max_r:.4f} mm, R_origin={r_origin:.4f} mm, MinMargin={min_margin:.2f} mm, MinCenterDist={min_center_dist:.2f} mm")
