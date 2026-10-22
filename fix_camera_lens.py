import sys, os, json, numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray

def run_fix():
    print("Loading base layout...")
    with open('dmd4500_nir_fingerprint_0p8mm_spectrometer.json', 'r') as f:
        data = json.load(f)
        
    # Fix the camera lens semi_diameter to 25.4 (2-inch lens instead of 1-inch)
    for s in data['surfaces']:
        if 'AC254-050-C_Foc' in s['name']:
            s['semi_diameter'] = 25.4
            
    # Save it temporarily
    with open('Spectrometer_V2_Perfected.json', 'w') as f:
        json.dump(data, f, indent=4)
        
    stack_base = SurfaceStack()
    for sd in data['surfaces'][:8]:
        s = Surface(name=sd.get('name', ''))
        for k,v in sd.items():
            if hasattr(s, k): setattr(s, k, v)
        stack_base.add_surface(s)
        
    sys_base = OpticalSystem(stack_base)
    
    # 1.111, 1.225, 1.389
    rays = []
    theta = np.arcsin(0.12)
    for wl in [1.111, 1.225, 1.389]:
        rays.append(Ray(pos=np.array([0.,0.,1e-6]), dir=np.array([0.,0.,1.]), wavelength=wl))
        for phi in [0, np.pi/2, np.pi, 3*np.pi/2]:
            rays.append(Ray(pos=np.array([0.,0.,1e-6]), dir=np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)]), wavelength=wl))
            
    print("Tracing rays to DMD...")
    dmd_rays = []
    for idx, r in enumerate(rays):
        curr = r.copy()
        vig_surf = -1
        for i, s in enumerate(sys_base.surfaces):
            curr = sys_base.trace_ray(curr)
            if curr.vignetted:
                vig_surf = i
                break
        if vig_surf == -1: 
            dmd_rays.append(r)
        else:
            print(f"Ray {idx} wl={r.wavelength} vignetted at surface {vig_surf} ({sys_base.surfaces[vig_surf].name})")
    print(f"Captured {len(dmd_rays)} / {len(rays)} rays after DMD.")
    
run_fix()
