
import sys, os, json, numpy as np
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.tracer import OpticalSystem
from v2.engine.surface import Surface, SurfaceType
from v2.engine.lde import SurfaceStack
from v2.engine.ray import Ray

data = {
    'surfaces': [
        {'name': 'Input_Slit', 'surface_type': 'Object', 'thickness': 25.0},
        {'name': 'Collimator', 'surface_type': 'Standard', 'thickness': 50.0},
        {'name': 'CB_Before_Grating', 'surface_type': 'Coordinate Break', 'is_coord_break': True, 'tilt_z': 90.0, 'thickness': 0.0},
        {'name': 'Grating_600lpmm', 'surface_type': 'Grating', 'thickness': 0.0, 'grating_lines_per_mm': 600.0, 'grating_order': 1, 'grating_primary_wavelength_um': 1.225, 'tilt_x': -45.0, 'glass': 'MIRROR'},
        {'name': 'CB_After_Grating', 'surface_type': 'Coordinate Break', 'is_coord_break': True, 'thickness': 45.0}
    ]
}
stack = SurfaceStack()
for sd in data['surfaces']:
    s = Surface(name=sd['name'])
    s.surface_type = SurfaceType.GRATING if 'Grating' in s.name else (SurfaceType.COORDINATE_BREAK if 'CB_' in s.name else SurfaceType.STANDARD)
    s.is_coord_break = ('CB_' in s.name)
    for k, v in sd.items():
        if hasattr(s, k): setattr(s, k, v)
    stack.add_surface(s)
sys_opt = OpticalSystem(stack)
sys_opt._update_accumulated_positions()

r = Ray(pos=np.array([0., 0., 0.]), dir=np.array([0., 0., 1.]), wavelength=1.225)
tr = sys_opt.trace_ray(r)
dir_after = tr.dir

specular_frame = sys_opt.surfaces[-1].global_R
local_dir = specular_frame.T @ dir_after
print('Local dir in specular frame:', local_dir)

# Since we want to align Z to local_dir, we compute tilt_x required
tilt_x_req = -np.degrees(np.arctan2(local_dir[1], local_dir[2]))
print(f'Required CB_After_Grating tilt_x = {tilt_x_req}')

