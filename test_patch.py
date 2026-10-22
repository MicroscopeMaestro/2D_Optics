
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath('')))
from v2.engine.surface import Surface, SurfaceType
from v2.engine.lde import SurfaceStack
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

with open('spectrometer_relay_redesign/spectrometer_primary_relay_redesign.json', 'r') as f: data_prim = json.load(f)
with open('spectrometer_relay_redesign/spectrometer_nir_900_1700nm.json', 'r') as f: data_nir = json.load(f)

# Extract relay lenses from NIR
dmd_idx_nir = next(i for i,s in enumerate(data_nir['surfaces']) if 'DMD' in s['name'])
relay_surfaces = data_nir['surfaces'][dmd_idx_nir+1 : -1]

# Replace relay in Primary
dmd_idx_prim = next(i for i,s in enumerate(data_prim['surfaces']) if 'DMD' in s['name'])
new_surfaces = data_prim['surfaces'][:dmd_idx_prim+1] + relay_surfaces + [data_prim['surfaces'][-1]]
data_prim['surfaces'] = new_surfaces

# M2 in Primary is a 50mm doublet. Let's change it to a 30mm doublet like NIR to shrink DMD spread
m2_idx_nir = next(i for i,s in enumerate(data_nir['surfaces']) if 'Foc' in s['name'])
m2_idx_prim = next(i for i,s in enumerate(data_prim['surfaces']) if 'Foc' in s['name'])

# Wait, Foc has 3 surfaces (doublet). Let's just replace all surfaces between Grating and DMD!
grat_idx_prim = next(i for i,s in enumerate(data_prim['surfaces']) if 'Grating' in s['name'])
grat_idx_nir = next(i for i,s in enumerate(data_nir['surfaces']) if 'Grating' in s['name'])
dmd_idx_prim = next(i for i,s in enumerate(data_prim['surfaces']) if 'DMD' in s['name'])
dmd_idx_nir = next(i for i,s in enumerate(data_nir['surfaces']) if 'DMD' in s['name'])

new_surfaces = data_prim['surfaces'][:grat_idx_prim+1] + data_nir['surfaces'][grat_idx_nir+1 : dmd_idx_nir+1] + relay_surfaces + [data_prim['surfaces'][-1]]

# Keep the Grating for primary as 600 lpm, 1.225 wl
new_surfaces[grat_idx_prim]['grating_primary_wavelength_um'] = 1.225
new_surfaces[grat_idx_prim]['tilt_x'] = -28.1 # Optimal for 1.225

data_prim['surfaces'] = new_surfaces

with open('C:/Users/Juan/.gemini/antigravity/brain/49fd158d-b0e1-421c-b15b-b237b2e3ad7f/Spectrometer_Hybrid_1064_Repaired.json', 'w') as f: json.dump(data_prim, f, indent=4)

stack = SurfaceStack()
for sd in data_prim['surfaces']:
    s = Surface(name=sd.get('name', ''))
    for k,v in sd.items():
        if k == 'surface_type': s.surface_type = SurfaceType.GRATING if v.lower() == 'grating' else (SurfaceType.STOP if v.lower() == 'stop' else (SurfaceType.OBJECT if v.lower() == 'object' else SurfaceType.STANDARD))
        elif hasattr(s, k): setattr(s, k, v)
    if 'Aspheric' in s.name:
        s.is_aspheric = True
        s.conic = sd.get('conic', 0.0)
    stack.add_surface(s)
sys_opt = OpticalSystem(stack)

wls = [1.087, 1.150, 1.225, 1.290, 1.350]
print('--- Primary Chief Ray Pass ---')
for wl in wls:
    r_chief = Ray(pos=np.array([0., 0., 0.]), dir=np.array([0., 0., 1.]), wavelength=wl)
    tr = sys_opt.trace_ray(r_chief)
    if tr.vignetted:
        print(f'{wl} Vignetted at:', sys_opt.surfaces[tr.history[-1]['surface_index']].name)
    else:
        print(f'{wl} passed. Final y_local:', tr.history[-1]['y_local'])

