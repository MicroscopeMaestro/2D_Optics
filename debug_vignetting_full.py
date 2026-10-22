import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from optics_app.engine.surface import Surface
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray

with open('C:/Users/Juan/Desktop/Spectrometer_V2_Optimized.json', 'r') as f:
    data = json.load(f)

stack = SurfaceStack()
for sd in data['surfaces']:
    s = Surface(name=sd.get('name', ''))
    for k,v in sd.items():
        if hasattr(s, k): setattr(s, k, v)
    stack.add_surface(s)
sys_opt = OpticalSystem(stack)
sys_opt.stack._update_accumulated_positions()

theta = np.arcsin(0.12)
for wl in [1.111, 1.225, 1.389]:
    rays = []
    # Generate 17 rays per wavelength
    for r_frac in [0, 0.5, 1.0]:
        th = np.arcsin(r_frac * 0.12)
        if r_frac == 0:
            rays.append(Ray(pos=np.array([0.,0.,1e-3]), dir=np.array([0.,0.,1.]), wavelength=wl))
        else:
            for phi in np.linspace(0, 2*np.pi, 8, endpoint=False):
                dx = np.sin(th)*np.cos(phi)
                dy = np.sin(th)*np.sin(phi)
                dz = np.cos(th)
                rays.append(Ray(pos=np.array([0.,0.,1e-3]), dir=np.array([dx,dy,dz]), wavelength=wl))
    
    # Trace step-by-step
    active_rays = rays
    print(f"\n--- Wavelength {wl} um ---")
    for i, s in enumerate(sys_opt.surfaces):
        if i == 0: continue # Skip slit
        survivors = []
        for r in active_rays:
            # We trace one surface at a time.
            # But trace_ray expects to trace all surfaces. We can just use s.intersect
            # Wait, easier to just run trace_ray and look at tr.vignetted
            pass
            
    # Let's just do full trace and find where it died
    vig_counts = {}
    for r in rays:
        curr = r.copy()
        dead_at = -1
        for i, s in enumerate(sys_opt.surfaces):
            # To do step by step, we need the local coordinate system
            # Let's just use sys_opt.trace_ray but truncate the stack
            temp_stack = SurfaceStack()
            for j in range(i+1):
                temp_stack.add_surface(sys_opt.surfaces[j])
            temp_sys = OpticalSystem(temp_stack)
            
            tr = temp_sys.trace_ray(curr.copy())
            if tr.vignetted:
                dead_at = i
                break
        if dead_at != -1:
            name = sys_opt.surfaces[dead_at].name
            vig_counts[name] = vig_counts.get(name, 0) + 1
            
    print(f"Total starting rays: {len(rays)}")
    if not vig_counts:
        print("All rays survived!")
    else:
        for name, count in vig_counts.items():
            print(f"  Vignetted at {name}: {count} rays")
