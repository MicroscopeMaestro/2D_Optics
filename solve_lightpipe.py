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
    # 3 active patches on DMD
    for y_obj in [4.0, 0.0, -4.0]:
        # Central ray
        rays.append(Ray(pos=np.array([0., y_obj, 1e-3]), dir=np.array([0.,0.,1.]), wavelength=1.0))
        # Marginal rays
        for phi in [0, np.pi/2, np.pi, 3*np.pi/2]:
            dx = np.sin(theta)*np.cos(phi)
            dy = np.sin(theta)*np.sin(phi)
            dz = np.cos(theta)
            rays.append(Ray(pos=np.array([0., y_obj, 1e-3]), dir=np.array([dx, dy, dz]), wavelength=1.0))
    return rays

def design_relay():
    input_rays = generate_dmd_rays()
    
    def build_relay(vars_array):
        d1, d2, d3 = vars_array
        
        stack = SurfaceStack()
        # Origin = DMD
        stack.add_surface(Surface(name="DMD_Plane", surface_type=SurfaceType.STANDARD, radius=0, thickness=d1, semi_diameter=float('inf')))
        
        # Field Lens: AC254-030-C
        stack.add_surface(Surface(name="Field_F", surface_type=SurfaceType.STANDARD, radius=19.9, thickness=5.0, glass="N-BAF10", semi_diameter=12.7))
        stack.add_surface(Surface(name="Field_M", surface_type=SurfaceType.STANDARD, radius=-15.3, thickness=2.0, glass="N-SF6", semi_diameter=12.7))
        stack.add_surface(Surface(name="Field_B", surface_type=SurfaceType.STANDARD, radius=-60.0, thickness=d2, glass="AIR", semi_diameter=12.7))
        
        # Condenser: AC254-030-C (Reversed)
        stack.add_surface(Surface(name="Cond_F", surface_type=SurfaceType.STANDARD, radius=60.0, thickness=2.0, glass="N-SF6", semi_diameter=12.7))
        stack.add_surface(Surface(name="Cond_M", surface_type=SurfaceType.STANDARD, radius=15.3, thickness=5.0, glass="N-BAF10", semi_diameter=12.7))
        stack.add_surface(Surface(name="Cond_B", surface_type=SurfaceType.STANDARD, radius=-19.9, thickness=d3, glass="AIR", semi_diameter=12.7))
        
        # Light Pipe Entrance (Detector)
        stack.add_surface(Surface(name="LightPipe_Entrance", surface_type=SurfaceType.DETECTOR, radius=0, thickness=0, semi_diameter=float('inf')))
        
        return OpticalSystem(stack)

    def merit(vars_array):
        d1, d2, d3 = vars_array
        if d1 < 2 or d2 < 2 or d3 < 2: return 1e9
        if d1 > 100 or d2 > 150 or d3 > 100: return 1e9
        
        sys_opt = build_relay(vars_array)
        score = 0.0
        
        for ray in input_rays:
            tr = sys_opt.trace_ray(ray)
            # Dist to target (0,0) - pupil imaging means all patches overlap at the center of the light pipe
            R_mat = sys_opt.surfaces[-1].global_R
            orig = sys_opt.surfaces[-1].global_origin
            loc = R_mat.T @ (tr.pos - orig)
            
            score += loc[0]**2 + loc[1]**2
                
        # soft penalty to prefer reasonable lengths
        score += 0.001 * (d1 + d2 + d3)
        return score
        
    x0 = [15.0, 50.0, 15.0]
    print("Optimizing Light-Pipe Relay...")
    res = minimize(merit, x0, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-3, 'fatol': 1e-3})
    
    print("Success:", res.success)
    print("Score:", res.fun)
    print("Distances:", res.x)
    
    return res.x

def apply_relay_to_json(vars_array):
    d1, d2, d3 = vars_array
    with open('dmd4500_nir_fingerprint_0p8mm_spectrometer.json', 'r') as f:
        data = json.load(f)
        
    surfaces = data['surfaces'][:8]
    surfaces[7]['thickness'] = d1
    
    # Add Thorlabs AC254-030-C
    surfaces.append({"name": "Field_F", "surface_type": "Standard", "radius": 19.9, "thickness": 5.0, "glass": "N-BAF10", "semi_diameter": 12.7, "is_active": True})
    surfaces.append({"name": "Field_M", "surface_type": "Standard", "radius": -15.3, "thickness": 2.0, "glass": "N-SF6", "semi_diameter": 12.7, "is_active": True})
    surfaces.append({"name": "Field_B", "surface_type": "Standard", "radius": -60.0, "thickness": d2, "glass": "AIR", "semi_diameter": 12.7, "is_active": True})
    
    # Add Reversed AC254-030-C
    surfaces.append({"name": "Cond_F", "surface_type": "Standard", "radius": 60.0, "thickness": 2.0, "glass": "N-SF6", "semi_diameter": 12.7, "is_active": True})
    surfaces.append({"name": "Cond_M", "surface_type": "Standard", "radius": 15.3, "thickness": 5.0, "glass": "N-BAF10", "semi_diameter": 12.7, "is_active": True})
    surfaces.append({"name": "Cond_B", "surface_type": "Standard", "radius": -19.9, "thickness": d3, "glass": "AIR", "semi_diameter": 12.7, "is_active": True})
    
    # Light Pipe Entrance
    surfaces.append({"name": "LightPipe_Entrance", "surface_type": "Detector", "radius": 0.0, "thickness": 0.0, "glass": "AIR", "semi_diameter": 2.5, "is_active": True})
    
    data['surfaces'] = surfaces
    with open('C:/Users/Juan/Desktop/Spectrometer_V2_Optimized.json', 'w') as f:
        json.dump(data, f, indent=4)
    print("Saved optimized layout to Spectrometer_V2_Optimized.json")

if __name__ == '__main__':
    best_x = design_relay()
    apply_relay_to_json(best_x)
