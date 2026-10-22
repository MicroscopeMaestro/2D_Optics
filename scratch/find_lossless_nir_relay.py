import sys
import os
import math
import numpy as np
from scipy.optimize import minimize
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray

# Standard Thorlabs mirrors for Collector (M3 or M2 in relay)
# 1-inch (Dia=25.4mm, Semi-Dia=12.7) or 2-inch (Dia=50.8, Semi-Dia=25.4)
COLLECTORS = [
    {"name": "CM254-050", "f": 50.0, "roc": -100.0, "sd": 12.7},
    {"name": "CM254-075", "f": 75.0, "roc": -150.0, "sd": 12.7},
    {"name": "CM254-100", "f": 100.0, "roc": -200.0, "sd": 12.7},
    {"name": "CM254-150", "f": 150.0, "roc": -300.0, "sd": 12.7},
    {"name": "CM254-200", "f": 200.0, "roc": -400.0, "sd": 12.7},
    {"name": "CM508-100", "f": 100.0, "roc": -200.0, "sd": 25.4},
    {"name": "CM508-150", "f": 150.0, "roc": -300.0, "sd": 25.4},
    {"name": "CM508-200", "f": 200.0, "roc": -400.0, "sd": 25.4}
]

# Standard Thorlabs aspheres / short focal length lenses for Condenser
CONDENSERS = [
    {"name": "C330TMD (f=3.1)", "f": 3.1, "roc": 1.6, "sd": 2.5, "glass": "D-ZK3"},
    {"name": "C392TMD (f=4.6)", "f": 4.6, "roc": 2.5, "sd": 2.5, "glass": "ECO550"},
    {"name": "A240 (f=8.0)", "f": 8.0, "roc": 4.5, "sd": 4.0, "glass": "N-BK7"},
    {"name": "A230 (f=7.5)", "f": 7.5, "roc": 4.2, "sd": 3.17, "glass": "N-BK7"},
    {"name": "A220 (f=11.0)", "f": 11.0, "roc": 6.0, "sd": 5.5, "glass": "N-BK7"},
    {"name": "A280 (f=18.4)", "f": 18.4, "roc": 10.0, "sd": 5.5, "glass": "N-BK7"},
    {"name": "AC050-010 (f=10.0)", "f": 10.0, "roc": 6.0, "sd": 2.5, "glass": "N-BK7"},
    {"name": "AC050-015 (f=15.0)", "f": 15.0, "roc": 8.0, "sd": 2.5, "glass": "N-BK7"},
    {"name": "AC127-019 (f=19.0)", "f": 19.0, "roc": 11.0, "sd": 6.35, "glass": "N-BK7"}
]

# We need a 0.09x magnification. So f_cond / f_coll <= 0.091.
# Example: 150mm collector -> 150 * 0.091 = 13.6mm max condenser focal length.

def build_relay(coll, cond, d_dmd_coll, d_coll_cond, d_cond_det):
    stack = SurfaceStack()
    
    # 0. DMD Object (simulating the dispersed beam)
    # The DMD width is 8.8mm, so semi-diameter is 4.4mm.
    stack.add_surface(Surface(
        name="DMD_Source", surface_type=SurfaceType.OBJECT,
        radius=0.0, thickness=d_dmd_coll, glass="AIR", semi_diameter=4.4
    ))
    
    # 1. Collector Mirror
    stack.add_surface(Surface(
        name=coll["name"], surface_type=SurfaceType.STANDARD,
        radius=coll["roc"], thickness=d_coll_cond, glass="MIRROR",
        semi_diameter=coll["sd"]
    ))
    
    # 2. Condenser Lens (Simplified as thin lens equivalent with 2 surfaces, or just 1 surface with glass for tracing)
    # We will use a proxy paraxial lens surface if possible, or build a plano-convex lens.
    # To use real tracing, we build a plano-convex lens.
    stack.add_surface(Surface(
        name=cond["name"]+"_S1", surface_type=SurfaceType.STANDARD,
        radius=cond["roc"], thickness=3.0, glass="BK7",
        semi_diameter=cond["sd"]
    ))
    stack.add_surface(Surface(
        name=cond["name"]+"_S2", surface_type=SurfaceType.STANDARD,
        radius=0.0, thickness=d_cond_det, glass="AIR",
        semi_diameter=cond["sd"]
    ))
    
    # 3. Detector
    stack.add_surface(Surface(
        name="Detector_0.8mm", surface_type=SurfaceType.DETECTOR,
        radius=0.0, thickness=0.0, glass="AIR", semi_diameter=0.4
    ))
    return OpticalSystem(stack)

def evaluate_pair(coll, cond):
    # Base constraints
    mag = cond["f"] / coll["f"]
    if mag > 0.12:
        return None, "Magnification too large"
        
    def objective(params):
        d_dmd_coll, d_coll_cond, d_cond_det = params
        if d_dmd_coll < 10 or d_coll_cond < 10 or d_cond_det < 1:
            return 1e6
            
        sys_eval = build_relay(coll, cond, d_dmd_coll, d_coll_cond, d_cond_det)
        
        # Trace rays from center and edges of DMD
        # Wavelength is ~1200nm (fingerprint center)
        wl = 1.2
        
        rays = []
        # Center point
        for angle in [-2, 0, 2]:
            dy = math.sin(math.radians(angle))
            dz = math.cos(math.radians(angle))
            rays.append(Ray(pos=(0.0, 0.0, 0.0), dir=(0.0, dy, dz), wavelength=wl))
        # Edge point
        for angle in [-2, 0, 2]:
            dy = math.sin(math.radians(angle))
            dz = math.cos(math.radians(angle))
            rays.append(Ray(pos=(0.0, 4.4, 0.0), dir=(0.0, dy, dz), wavelength=wl))
            rays.append(Ray(pos=(0.0, -4.4, 0.0), dir=(0.0, dy, dz), wavelength=wl))
            
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
            
        return spot_rms * 100 + loss_penalty + abs(d_dmd_coll - coll["f"]) + abs(d_cond_det - cond["f"])
        
    # Start with ideal focal distances
    x0 = [coll["f"], coll["f"] + cond["f"], cond["f"]]
    res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 100})
    
    val = res.fun
    return res.x, val

best_val = float('inf')
best_pair = None
best_params = None

print(f"{'Collector':<15} {'Condenser':<15} {'Mag':<10} {'Score':<10} {'Throughput'}")
for coll in COLLECTORS:
    for cond in CONDENSERS:
        mag = cond["f"] / coll["f"]
        if mag > 0.12:
            continue
        params, val = evaluate_pair(coll, cond)
        if params is not None:
            # Re-evaluate throughput strictly
            test_sys = build_relay(coll, cond, params[0], params[1], params[2])
            rays = []
            for a in [-2, 0, 2]:
                dy = math.sin(math.radians(a))
                dz = math.cos(math.radians(a))
                rays.append(Ray(pos=(0.0, 4.4, 0.0), dir=(0.0, dy, dz), wavelength=1.2))
                rays.append(Ray(pos=(0.0, -4.4, 0.0), dir=(0.0, dy, dz), wavelength=1.2))
            test_sys.trace_rays(rays)
            vignetted = sum([1 for r in rays if r.vignetted])
            throughput = (len(rays) - vignetted) / len(rays)
            
            print(f"{coll['name']:<15} {cond['name']:<15} {mag:<10.3f} {val:<10.2f} {throughput*100:.0f}%")
            if val < best_val and throughput == 1.0:
                best_val = val
                best_pair = (coll, cond)
                best_params = params

if best_pair:
    print("\nBest Pair with 100% throughput:")
    print("Collector:", best_pair[0]["name"])
    print("Condenser:", best_pair[1]["name"])
    print("Distances:", best_params)
else:
    print("\nNo pair achieved 100% throughput with the desired spot size.")
