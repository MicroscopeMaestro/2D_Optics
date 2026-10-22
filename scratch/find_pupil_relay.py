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

# Field Lenses (Large CA, placed near DMD)
FIELD_LENSES = [
    {"name": "AC254-030-C", "f": 30.0, "roc": 15.0, "sd": 12.7},
    {"name": "AC254-040-C", "f": 40.0, "roc": 20.0, "sd": 12.7},
    {"name": "AC254-050-C", "f": 50.0, "roc": -100.0, "sd": 12.7},
    {"name": "AC254-075-C", "f": 75.0, "roc": -150.0, "sd": 12.7},
    {"name": "LA1131 (f=50)", "f": 50.0, "roc": 25.0, "sd": 12.7}
]

# Condensers (Short focal length, placed near detector)
CONDENSERS = [
    {"name": "A240 (f=8.0)", "f": 8.0, "roc": 4.5, "sd": 4.0},
    {"name": "A230 (f=7.5)", "f": 7.5, "roc": 4.2, "sd": 3.17},
    {"name": "A220 (f=11.0)", "f": 11.0, "roc": 6.0, "sd": 5.5},
    {"name": "AC050-010 (f=10.0)", "f": 10.0, "roc": 6.0, "sd": 2.5}
]

def build_pupil_relay(field, cond, d_dmd_field, d_field_cond, d_cond_det):
    stack = SurfaceStack()
    
    # 0. DMD Object (simulating the dispersed beam)
    stack.add_surface(Surface(
        name="DMD_Source", surface_type=SurfaceType.OBJECT,
        radius=0.0, thickness=d_dmd_field, glass="AIR", semi_diameter=6.0
    ))
    
    # 1. Field Lens (Plano-Convex or Biconvex simplified)
    stack.add_surface(Surface(
        name=field["name"]+"_Front", surface_type=SurfaceType.STANDARD,
        radius=field["roc"], thickness=3.0, glass="BK7",
        semi_diameter=field["sd"]
    ))
    stack.add_surface(Surface(
        name=field["name"]+"_Back", surface_type=SurfaceType.STANDARD,
        radius=0.0, thickness=d_field_cond, glass="AIR",
        semi_diameter=field["sd"]
    ))
    
    # 2. Condenser Lens
    stack.add_surface(Surface(
        name=cond["name"]+"_Front", surface_type=SurfaceType.STANDARD,
        radius=cond["roc"], thickness=3.0, glass="BK7",
        semi_diameter=cond["sd"]
    ))
    stack.add_surface(Surface(
        name=cond["name"]+"_Back", surface_type=SurfaceType.STANDARD,
        radius=0.0, thickness=d_cond_det, glass="AIR",
        semi_diameter=cond["sd"]
    ))
    
    # 3. Detector
    stack.add_surface(Surface(
        name="Detector_0.8mm", surface_type=SurfaceType.DETECTOR,
        radius=0.0, thickness=0.0, glass="AIR", semi_diameter=0.4
    ))
    return OpticalSystem(stack)

def evaluate_pair(field, cond):
    def objective(params):
        d_dmd_field, d_field_cond, d_cond_det = params
        if d_dmd_field < 2 or d_field_cond < 2 or d_cond_det < 1:
            return 1e6
            
        sys_eval = build_pupil_relay(field, cond, d_dmd_field, d_field_cond, d_cond_det)
        
        rays = []
        # Center point
        for angle in [-4, 0, 4]:
            dy = math.sin(math.radians(angle))
            dz = math.cos(math.radians(angle))
            rays.append(Ray(pos=(0.0, 0.0, 0.0), dir=(0.0, dy, dz), wavelength=1.2))
        # Edge points (representing extreme wavelengths at +/- 4.4mm on DMD)
        # Note: the chief ray for these edge points actually comes in at an angle in a real spectrometer!
        # But for conservative pupil imaging, we'll assume they diverge from the DMD normal.
        for angle in [-4, 0, 4]:
            dy = math.sin(math.radians(angle))
            dz = math.cos(math.radians(angle))
            rays.append(Ray(pos=(0.0, 4.4, 0.0), dir=(0.0, dy, dz), wavelength=1.2))
            rays.append(Ray(pos=(0.0, -4.4, 0.0), dir=(0.0, dy, dz), wavelength=1.2))
            
        sys_eval.trace_rays(rays)
        
        loss_penalty = 0
        spot_rms = 0
        hits = 0
        for r in rays:
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
        
    x0 = [10.0, field["f"] + cond["f"], cond["f"]]
    res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 200})
    return res.x, res.fun

best_val = float('inf')
best_pair = None
best_params = None

for field in FIELD_LENSES:
    for cond in CONDENSERS:
        params, val = evaluate_pair(field, cond)
        
        test_sys = build_pupil_relay(field, cond, params[0], params[1], params[2])
        rays = []
        for a in [-4, 0, 4]:
            dy = math.sin(math.radians(a))
            dz = math.cos(math.radians(a))
            rays.append(Ray(pos=(0.0, 4.4, 0.0), dir=(0.0, dy, dz), wavelength=1.2))
            rays.append(Ray(pos=(0.0, -4.4, 0.0), dir=(0.0, dy, dz), wavelength=1.2))
        test_sys.trace_rays(rays)
        vignetted = sum([1 for r in rays if r.vignetted])
        throughput = (len(rays) - vignetted) / len(rays)
        
        if val < best_val and throughput == 1.0:
            spot_rms = val / 100
            if spot_rms <= 0.4:
                best_val = val
                best_pair = (field, cond)
                best_params = params
        
        print(f"Field: {field['name']:<15} Cond: {cond['name']:<15} Spot: {val/100:.3f} mm, Throughput: {throughput*100:.0f}%")

if best_pair:
    print(f"SUCCESS! Found a Pupil-Imaging setup with 100% throughput!")
    print(f"Field Lens: {best_pair[0]['name']}")
    print(f"Condenser Lens: {best_pair[1]['name']}")
    print(f"DMD to Field Lens: {best_params[0]:.2f} mm")
    print(f"Field to Condenser: {best_params[1]:.2f} mm")
    print(f"Condenser to Detector: {best_params[2]:.2f} mm")
    print(f"Spot RMS: {best_val/100:.3f} mm")
else:
    print("Could not find a 100% throughput pupil relay with spot <= 0.4mm")
