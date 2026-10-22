import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from optics_app.engine.surface import Surface
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray

def test_fix():
    with open('C:/Users/Juan/Desktop/Spectrometer_V2_Optimized.json', 'r') as f:
        data = json.load(f)

    # 1. Change Slit thickness to 30.0
    data['surfaces'][0]['thickness'] = 30.0
    
    # 2. Change M1 Collimator radius to -60.0 (f=30mm)
    for s in data['surfaces']:
        if s['name'] == 'M1_Collimator_50mm':
            s['radius'] = -60.0
            s['name'] = 'M1_Collimator_30mm'
            
        # Also let's fix Input_Aperture which was EXACTLY 6.0 and clipping marginal rays
        if s['name'] == 'Input_Aperture':
            s['semi_diameter'] = 10.0  # Make it larger so it doesn't artificially clip

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
        
        vig_counts = {}
        survivors = 0
        for r in rays:
            curr = r.copy()
            dead_at = -1
            for i, s in enumerate(sys_opt.surfaces):
                temp_stack = SurfaceStack()
                for j in range(i+1): temp_stack.add_surface(sys_opt.surfaces[j])
                temp_sys = OpticalSystem(temp_stack)
                tr = temp_sys.trace_ray(curr.copy())
                if tr.vignetted:
                    dead_at = i
                    break
            if dead_at != -1:
                name = sys_opt.surfaces[dead_at].name
                vig_counts[name] = vig_counts.get(name, 0) + 1
            else:
                survivors += 1
                
        print(f"WL {wl} um: {survivors}/{len(rays)} survived.")
        for name, count in vig_counts.items():
            print(f"  Vignetted at {name}: {count}")

test_fix()
