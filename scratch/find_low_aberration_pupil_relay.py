import sys
import os
import math
import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray

# Field Lenses
FIELD_LENSES = [
    {"name": "AC254-030-C", "f": 30.0, "roc": 15.0, "sd": 12.7},
    {"name": "AC254-050-C", "f": 50.0, "roc": 25.0, "sd": 12.7}
]

# Dual-Condenser combinations to reduce spherical aberration
# We use two identical achromats face-to-face (Plössl style)
DUAL_CONDENSERS = [
    {"name": "Dual_AC127-019-C (f~9.5)", "f": 19.0, "roc": 11.0, "sd": 6.35},
    {"name": "Dual_AC050-015-C (f~7.5)", "f": 15.0, "roc": 8.0, "sd": 2.5},
    {"name": "Dual_AC254-030-C (f~15)", "f": 30.0, "roc": 15.0, "sd": 12.7},
    {"name": "Dual_AC254-040-C (f~20)", "f": 40.0, "roc": 20.0, "sd": 12.7}
]

def build_dual_condenser_relay(field, cond, d_dmd_field, d_field_c1, d_c1_c2, d_c2_det):
    stack = SurfaceStack()
    
    # 0. DMD Object
    stack.add_surface(Surface(
        name="DMD_Source", surface_type=SurfaceType.OBJECT,
        radius=0.0, thickness=d_dmd_field, glass="AIR", semi_diameter=6.0
    ))
    
    # 1. Field Lens
    stack.add_surface(Surface(
        name=field["name"]+"_Front", surface_type=SurfaceType.STANDARD,
        radius=field["roc"], thickness=3.0, glass="N-BK7", semi_diameter=field["sd"]
    ))
    stack.add_surface(Surface(
        name=field["name"]+"_Back", surface_type=SurfaceType.STANDARD,
        radius=0.0, thickness=d_field_c1, glass="AIR", semi_diameter=field["sd"]
    ))
    
    # 2. Condenser 1 (convex side facing field lens)
    stack.add_surface(Surface(
        name=cond["name"]+"_C1_Front", surface_type=SurfaceType.STANDARD,
        radius=cond["roc"], thickness=3.0, glass="N-BK7", semi_diameter=cond["sd"]
    ))
    stack.add_surface(Surface(
        name=cond["name"]+"_C1_Back", surface_type=SurfaceType.STANDARD,
        radius=0.0, thickness=d_c1_c2, glass="AIR", semi_diameter=cond["sd"]
    ))
    
    # 3. Condenser 2 (convex side facing detector -> flipped radius)
    stack.add_surface(Surface(
        name=cond["name"]+"_C2_Front", surface_type=SurfaceType.STANDARD,
        radius=0.0, thickness=3.0, glass="N-BK7", semi_diameter=cond["sd"]
    ))
    stack.add_surface(Surface(
        name=cond["name"]+"_C2_Back", surface_type=SurfaceType.STANDARD,
        radius=-cond["roc"], thickness=d_c2_det, glass="AIR", semi_diameter=cond["sd"]
    ))
    
    # 4. Detector
    stack.add_surface(Surface(
        name="Detector_0.8mm", surface_type=SurfaceType.DETECTOR,
        radius=0.0, thickness=0.0, glass="AIR", semi_diameter=0.4
    ))
    return OpticalSystem(stack)

def evaluate_pair(field, cond):
    def objective(params):
        d_dmd_field, d_field_c1, d_c1_c2, d_c2_det = params
        if d_dmd_field < 5 or d_field_c1 < 10 or d_c1_c2 < 0.1 or d_c2_det < 1:
            return 1e6
            
        sys_eval = build_dual_condenser_relay(field, cond, d_dmd_field, d_field_c1, d_c1_c2, d_c2_det)
        
        rays = []
        # Center point
        for angle in [-4, 0, 4]:
            dy = math.sin(math.radians(angle))
            dz = math.cos(math.radians(angle))
            rays.append(Ray(pos=(0.0, 0.0, 0.0), dir=(0.0, dy, dz), wavelength=1.2))
        # Edge points
        for angle in [-4, 0, 4]:
            dy = math.sin(math.radians(angle))
            dz = math.cos(math.radians(angle))
            rays.append(Ray(pos=(0.0, 4.4, 0.0), dir=(0.0, dy, dz), wavelength=1.2))
            rays.append(Ray(pos=(0.0, -4.4, 0.0), dir=(0.0, dy, dz), wavelength=1.2))
            
        traced_rays = sys_eval.trace_rays(rays)
        
        loss_penalty = 0
        spot_rms = 0
        hits = 0
        for r in traced_rays:
            if not r.vignetted:
                hits += 1
                spot_rms += r.pos[1]**2
            else:
                loss_penalty += 1000
                
        if hits > 0:
            spot_rms = math.sqrt(spot_rms / hits)
        else:
            spot_rms = 1000
            
        return spot_rms * 100 + loss_penalty
        
    # Pupil image: M2 is 50mm before DMD. 
    # Field lens at 10mm from DMD. Total u = 60mm.
    # To image pupil, v = 1 / (1/f - 1/60).
    # For f=30, v=60. For f=50, v=300.
    v_pupil = 1.0 / (1.0/field["f"] - 1.0/60.0) if field["f"] != 60 else 100.0
    if v_pupil < 0: v_pupil = 50.0 # Virtual image fallback
    
    x0 = [10.0, v_pupil, 1.0, cond["f"]/2.0]
    res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 300})
    return res.x, res.fun

best_val = float('inf')

for field in FIELD_LENSES:
    for cond in DUAL_CONDENSERS:
        params, val = evaluate_pair(field, cond)
        
        test_sys = build_dual_condenser_relay(field, cond, params[0], params[1], params[2], params[3])
        rays = []
        for a in [-4, 0, 4]:
            dy = math.sin(math.radians(a))
            dz = math.cos(math.radians(a))
            rays.append(Ray(pos=(0.0, 4.4, 0.0), dir=(0.0, dy, dz), wavelength=1.2))
            rays.append(Ray(pos=(0.0, -4.4, 0.0), dir=(0.0, dy, dz), wavelength=1.2))
        traced_rays = test_sys.trace_rays(rays)
        vignetted = sum([1 for r in traced_rays if r.vignetted])
        throughput = (len(rays) - vignetted) / len(rays)
        
        print(f"Field: {field['name']:<15} Cond: {cond['name']:<30} Spot: {val/100:.3f} mm, Thru: {throughput*100:.0f}%")
