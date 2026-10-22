import sys, os, json, numpy as np
from scipy.optimize import minimize
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray

def generate_na_rays(pos, dir_z, na, wavelengths):
    rays = []
    # Divergence angle
    theta = np.arcsin(na)
    for wl in wavelengths:
        # Central ray
        rays.append(Ray(pos=np.array(pos), dir=np.array(dir_z), wavelength=wl))
        # 4 marginal rays
        for phi in [0, np.pi/2, np.pi, 3*np.pi/2]:
            dx = np.sin(theta) * np.cos(phi)
            dy = np.sin(theta) * np.sin(phi)
            dz = np.cos(theta)
            rays.append(Ray(pos=np.array(pos), dir=np.array([dx, dy, dz]), wavelength=wl))
    return rays

def run_relay_optimization():
    print("Loading base layout...")
    with open('dmd4500_nir_fingerprint_0p8mm_spectrometer.json', 'r') as f:
        data = json.load(f)
        
    stack_base = SurfaceStack()
    # Load up to DMD (index 7)
    for sd in data['surfaces'][:8]:
        s = Surface(name=sd.get('name', ''))
        for k,v in sd.items():
            if hasattr(s, k): setattr(s, k, v)
        stack_base.add_surface(s)
        
    sys_base = OpticalSystem(stack_base)
    
    # Generate rays at slit
    input_rays = generate_na_rays([0.,0.,0.], [0.,0.,1.], 0.12, [1.111, 1.225, 1.389])
    
    print("Tracing rays to DMD...")
    dmd_rays = []
    for r in input_rays:
        tr = sys_base.trace_ray(r)
        # trace_ray returns the ray at the last valid surface.
        # But wait, it traces through all surfaces.
        # If we only have up to DMD, the ray reflects off DMD and exits into space.
        # We need the ray state EXACTLY after bouncing off the DMD.
        # trace_ray does exactly this, the returned ray has pos and dir after the last surface!
        if not tr.vignetted:
            dmd_rays.append(tr)
            
    print(f"Captured {len(dmd_rays)} rays after DMD.")
    if len(dmd_rays) == 0:
        print("ERROR: All rays vignetted before DMD!")
        return
        
    # Now build the dynamic relay stack
    def build_relay(vars_array):
        # variables: 
        # 0: DMD to L1_front thickness
        # 1: L1 radius (front)
        # 2: L1 radius (back)
        # 3: L1 thickness
        # 4: L1 to L2 thickness
        # 5: L2 radius (front)
        # 6: L2 radius (back)
        # 7: L2 thickness
        # 8: L2 to Detector thickness
        d_dmd_l1, r1f, r1b, t1, d_l1_l2, r2f, r2b, t2, d_l2_det = vars_array
        
        rs = SurfaceStack()
        # Add a dummy surface at DMD position so the first thickness acts as distance from DMD
        rs.add_surface(Surface(name="Relay_Start", surface_type=SurfaceType.STANDARD, radius=0, thickness=d_dmd_l1))
        
        # L1 (Field Lens) - semi_diameter 25.4 (2-inch)
        rs.add_surface(Surface(name="L1_Front", surface_type=SurfaceType.ASPHERIC, radius=r1f, conic=-1.0, thickness=t1, glass="N-SF6", semi_diameter=25.4))
        rs.add_surface(Surface(name="L1_Back", surface_type=SurfaceType.ASPHERIC, radius=r1b, conic=-1.0, thickness=d_l1_l2, glass="AIR", semi_diameter=25.4))
        
        # L2 (Condenser) - semi_diameter 25.4
        rs.add_surface(Surface(name="L2_Front", surface_type=SurfaceType.ASPHERIC, radius=r2f, conic=-1.0, thickness=t2, glass="N-SF6", semi_diameter=25.4))
        rs.add_surface(Surface(name="L2_Back", surface_type=SurfaceType.ASPHERIC, radius=r2b, conic=-1.0, thickness=d_l2_det, glass="AIR", semi_diameter=25.4))
        
        # Detector
        rs.add_surface(Surface(name="Detector_0.8mm", surface_type=SurfaceType.DETECTOR, radius=0, thickness=0, semi_diameter=0.4))
        return rs

    def merit(vars_array):
        # Penalize negative thicknesses or crazy radii
        d_dmd_l1, r1f, r1b, t1, d_l1_l2, r2f, r2b, t2, d_l2_det = vars_array
        if d_dmd_l1 < 5 or d_l1_l2 < 2 or d_l2_det < 5 or t1 < 2 or t2 < 2:
            return 1e9
        if abs(r1f) < 5 or abs(r1b) < 5 or abs(r2f) < 5 or abs(r2b) < 5:
            return 1e9
            
        relay_stack = build_relay(vars_array)
        
        # We need to set the initial frame of the relay to MATCH the frame AFTER the DMD.
        # But our `dmd_rays` are already in GLOBAL coordinates!
        # If we just place the relay starting at the DMD's global origin and orientation.
        # Wait, the relay's `_update_accumulated_positions` starts from origin=[0,0,0], frame=I.
        # We must seed it with the DMD's final frame!
        dmd_surf = stack_base.surfaces[7]
        # The frame after DMD reflection:
        z_axis_in = dmd_surf.global_R[:, 2]
        normal = dmd_surf.global_R[:, 2] # actually mirror normal
        z_axis_out = z_axis_in - 2 * np.dot(z_axis_in, normal) * normal
        # We need a rotation matrix whose Z is z_axis_out.
        # The engine normally does this in `_update_accumulated_positions`.
        # To make it simple, we just append the relay surfaces to the BASE stack and trace!
        
        full_stack = SurfaceStack()
        for s in stack_base.surfaces: full_stack.add_surface(s)
        
        # Modify DMD thickness
        full_stack.surfaces[7].thickness = d_dmd_l1
        
        # Add Relay
        full_stack.add_surface(Surface(name="L1_Front", surface_type=SurfaceType.ASPHERIC, radius=r1f, conic=-1.0, thickness=t1, glass="N-SF6", semi_diameter=25.4))
        full_stack.add_surface(Surface(name="L1_Back", surface_type=SurfaceType.ASPHERIC, radius=r1b, conic=-1.0, thickness=d_l1_l2, glass="AIR", semi_diameter=25.4))
        full_stack.add_surface(Surface(name="L2_Front", surface_type=SurfaceType.ASPHERIC, radius=r2f, conic=-1.0, thickness=t2, glass="N-SF6", semi_diameter=25.4))
        full_stack.add_surface(Surface(name="L2_Back", surface_type=SurfaceType.ASPHERIC, radius=r2b, conic=-1.0, thickness=d_l2_det, glass="AIR", semi_diameter=25.4))
        full_stack.add_surface(Surface(name="Detector", surface_type=SurfaceType.DETECTOR, radius=0, thickness=0, semi_diameter=25.4)) # Large semi for score
        
        sys_full = OpticalSystem(full_stack)
        
        score = 0.0
        # Re-trace from slit to evaluate full system
        for r in input_rays:
            tr = sys_full.trace_ray(r)
            if tr.vignetted:
                score += 1000.0
            else:
                # Local hit position on detector
                R_mat = full_stack.surfaces[-1].global_R
                origin = full_stack.surfaces[-1].global_origin
                local_pos = R_mat.T @ (tr.pos - origin)
                
                # We want all rays to land within 0.4mm (squared error)
                dist2 = local_pos[0]**2 + local_pos[1]**2
                score += dist2 * 10.0 # Heavy weight on spot size
                
        # Optional: Add small penalty to keep lengths reasonable
        score += 0.01 * (d_dmd_l1 + d_l1_l2 + d_l2_det)
        
        return score

    print("Starting optimization (this may take 1-2 minutes)...")
    # Initial guess: 
    # d_dmd_l1=30, r1f=50, r1b=-50, t1=5, d_l1_l2=50, r2f=30, r2b=-30, t2=5, d_l2_det=30
    x0 = [30.0, 50.0, -50.0, 5.0, 50.0, 30.0, -30.0, 5.0, 30.0]
    
    res = minimize(merit, x0, method='Nelder-Mead', options={'maxiter': 500})
    print(res.message)
    print("Final Score:", res.fun)
    print("Optimized Vars:", res.x)
    
    # Save the perfected JSON
    vars_array = res.x
    d_dmd_l1, r1f, r1b, t1, d_l1_l2, r2f, r2b, t2, d_l2_det = vars_array
    
    stack_base.surfaces[7].thickness = d_dmd_l1
    stack_base.add_surface(Surface(name="L1_Front", surface_type=SurfaceType.ASPHERIC, radius=r1f, conic=-1.0, thickness=t1, glass="N-SF6", semi_diameter=25.4))
    stack_base.add_surface(Surface(name="L1_Back", surface_type=SurfaceType.ASPHERIC, radius=r1b, conic=-1.0, thickness=d_l1_l2, glass="AIR", semi_diameter=25.4))
    stack_base.add_surface(Surface(name="L2_Front", surface_type=SurfaceType.ASPHERIC, radius=r2f, conic=-1.0, thickness=t2, glass="N-SF6", semi_diameter=25.4))
    stack_base.add_surface(Surface(name="L2_Back", surface_type=SurfaceType.ASPHERIC, radius=r2b, conic=-1.0, thickness=d_l2_det, glass="AIR", semi_diameter=25.4))
    stack_base.add_surface(Surface(name="Detector_0.8mm", surface_type=SurfaceType.DETECTOR, radius=0, thickness=0, semi_diameter=0.4))
    
    final_dict = data
    # Convert back to dicts manually
    new_surfaces = []
    for s in stack_base.surfaces:
        sd = {
            "name": s.name,
            "surface_type": s.surface_type,
            "radius": s.radius,
            "thickness": s.thickness,
            "glass": s.glass,
            "semi_diameter": s.semi_diameter,
            "conic": getattr(s, 'conic', 0.0),
            "is_active": True
        }
        if hasattr(s, 'tilt_x') and s.tilt_x != 0: sd['tilt_x'] = s.tilt_x
        if hasattr(s, 'tilt_y') and s.tilt_y != 0: sd['tilt_y'] = s.tilt_y
        if hasattr(s, 'decenter_y') and s.decenter_y != 0: sd['decenter_y'] = s.decenter_y
        if hasattr(s, 'diffraction_order') and s.diffraction_order is not None: sd['diffraction_order'] = s.diffraction_order
        if hasattr(s, 'lines_per_um') and s.lines_per_um is not None: sd['lines_per_um'] = s.lines_per_um
        new_surfaces.append(sd)
        
    final_dict['surfaces'] = new_surfaces
    
    with open('C:/Users/Juan/Desktop/Spectrometer_V2_Perfected.json', 'w') as f:
        json.dump(final_dict, f, indent=4)
        
    print("Saved custom imaging system to Spectrometer_V2_Perfected.json")

if __name__ == '__main__':
    run_relay_optimization()
