import os, sys, json, numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray

def generate_dmd_rays():
    rays = []
    theta = np.arcsin(0.12)
    # 3 object points on DMD
    for y_obj, y_target in [(4.0, -0.4), (0.0, 0.0), (-4.0, 0.4)]:
        # Central ray
        rays.append((Ray(pos=np.array([0., y_obj, 1e-3]), dir=np.array([0.,0.,1.]), wavelength=1.0), y_target))
        # 4 marginal rays
        for phi in [0, np.pi/2, np.pi, 3*np.pi/2]:
            dx = np.sin(theta)*np.cos(phi)
            dy = np.sin(theta)*np.sin(phi)
            dz = np.cos(theta)
            rays.append((Ray(pos=np.array([0., y_obj, 1e-3]), dir=np.array([dx, dy, dz]), wavelength=1.0), y_target))
    return rays

def design_relay():
    input_data = generate_dmd_rays()
    
    def build_relay(vars_array):
        # vars: d1, r1f, r1b, t1, d2, r2f, r2b, t2, d3
        d1, r1f, r1b, t1, d2, r2f, r2b, t2, d3 = vars_array
        
        stack = SurfaceStack()
        # Origin = DMD
        stack.add_surface(Surface(name="DMD_Plane", surface_type=SurfaceType.STANDARD, radius=0, thickness=d1, semi_diameter=float('inf')))
        
        # Field Lens
        stack.add_surface(Surface(name="Field_F", surface_type=SurfaceType.ASPHERIC, radius=r1f, conic=-1.0, thickness=t1, glass="N-SF6", semi_diameter=float('inf')))
        stack.add_surface(Surface(name="Field_B", surface_type=SurfaceType.ASPHERIC, radius=r1b, conic=-1.0, thickness=d2, glass="AIR", semi_diameter=float('inf')))
        
        # Condenser
        stack.add_surface(Surface(name="Cond_F", surface_type=SurfaceType.ASPHERIC, radius=r2f, conic=-1.0, thickness=t2, glass="N-SF6", semi_diameter=float('inf')))
        stack.add_surface(Surface(name="Cond_B", surface_type=SurfaceType.ASPHERIC, radius=r2b, conic=-1.0, thickness=d3, glass="AIR", semi_diameter=float('inf')))
        
        # Detector
        stack.add_surface(Surface(name="Detector", surface_type=SurfaceType.DETECTOR, radius=0, thickness=0, semi_diameter=float('inf')))
        
        return OpticalSystem(stack)

    def merit(vars_array):
        d1, r1f, r1b, t1, d2, r2f, r2b, t2, d3 = vars_array
        # Constraints
        if d1 < 2 or d2 < 2 or d3 < 2 or t1 < 2 or t2 < 2: return 1e9
        if d1 > 100 or d2 > 100 or d3 > 50 or t1 > 20 or t2 > 20: return 1e9
        if abs(r1f) < 26 or abs(r1b) < 26 or abs(r2f) < 13 or abs(r2b) < 13: return 1e9
        
        sys_opt = build_relay(vars_array)
        score = 0.0
        
        for ray, y_target in input_data:
            tr = sys_opt.trace_ray(ray)
            # Dist to target
            R_mat = sys_opt.surfaces[-1].global_R
            orig = sys_opt.surfaces[-1].global_origin
            loc = R_mat.T @ (tr.pos - orig)
            
            score += loc[0]**2 + (loc[1] - y_target)**2
                
        # small penalty to keep lenses reasonable
        score += 0.01 * (d1 + d2 + d3 + t1 + t2)
        score += 0.01 * (1.0/abs(r1f) + 1.0/abs(r1b) + 1.0/abs(r2f) + 1.0/abs(r2b))
        
        return score
        
    x0 = [15.0, 50.0, -50.0, 5.0, 40.0, 20.0, -20.0, 5.0, 20.0]
    print("Optimizing Custom Relay System (0.1x Magnification)...")
    res = minimize(merit, x0, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-4, 'fatol': 1e-4})
    
    print("Success:", res.success)
    print("Score:", res.fun)
    print("Vars:", res.x)
    
    return res.x

def apply_relay_to_json(vars_array):
    d1, r1f, r1b, t1, d2, r2f, r2b, t2, d3 = vars_array
    with open('dmd4500_nir_fingerprint_0p8mm_spectrometer.json', 'r') as f:
        data = json.load(f)
        
    # keep only up to DMD
    surfaces = data['surfaces'][:8]
    surfaces[7]['thickness'] = d1
    
    # Add new relay
    surfaces.append({
        "name": "Field_F_Custom", "surface_type": "Aspheric", "radius": r1f, "conic": -1.0,
        "thickness": t1, "glass": "N-SF6", "semi_diameter": 25.4, "is_active": True
    })
    surfaces.append({
        "name": "Field_B_Custom", "surface_type": "Aspheric", "radius": r1b, "conic": -1.0,
        "thickness": d2, "glass": "AIR", "semi_diameter": 25.4, "is_active": True
    })
    surfaces.append({
        "name": "Cond_F_Custom", "surface_type": "Aspheric", "radius": r2f, "conic": -1.0,
        "thickness": t2, "glass": "N-SF6", "semi_diameter": 12.7, "is_active": True
    })
    surfaces.append({
        "name": "Cond_B_Custom", "surface_type": "Aspheric", "radius": r2b, "conic": -1.0,
        "thickness": d3, "glass": "AIR", "semi_diameter": 12.7, "is_active": True
    })
    surfaces.append({
        "name": "Detector_0.8mm", "surface_type": "Detector", "radius": 0.0, "thickness": 0.0,
        "glass": "AIR", "semi_diameter": 0.4, "is_active": True
    })
    
    data['surfaces'] = surfaces
    with open('C:/Users/Juan/Desktop/Spectrometer_V2_Optimized.json', 'w') as f:
        json.dump(data, f, indent=4)
    print("Saved optimized layout to Spectrometer_V2_Optimized.json")

if __name__ == '__main__':
    best_x = design_relay()
    apply_relay_to_json(best_x)
