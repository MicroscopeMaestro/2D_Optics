import sys, os, json, numpy as np
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.surface import Surface, SurfaceType
from v2.engine.lde import SurfaceStack
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

def build_mirror_system():
    # We will build it surface by surface with explicit coordinate breaks
    # Light travels +Z.
    
    surfaces = []
    surfaces.append(Surface(name='OBJ', surface_type=SurfaceType.OBJECT, thickness=10.0, glass='AIR'))
    surfaces.append(Surface(name='Input_Slit_200um', surface_type=SurfaceType.STOP, thickness=50.0, semi_diameter=0.1, glass='AIR'))
    
    # M1 Collimator (EFL 50.0 -> Radius -100.0)
    # Fold angle: let's deflect by 30 degrees (tilt 15 deg)
    surfaces.append(Surface(name='CB_Before_M1', surface_type=SurfaceType.COORDINATE_BREAK, tilt_x=15.0))
    surfaces.append(Surface(name='M1_Collimator', surface_type=SurfaceType.MIRROR, radius=-100.0, thickness=0.0, semi_diameter=12.7, glass='MIRROR'))
    surfaces.append(Surface(name='CB_After_M1', surface_type=SurfaceType.COORDINATE_BREAK, tilt_x=15.0, thickness=-50.0))
    # Note: after mirror, thickness is negative. 

    # Grating (150 l/mm)
    # We need to deflect back.
    surfaces.append(Surface(name='CB_Before_Grating', surface_type=SurfaceType.COORDINATE_BREAK, tilt_x=-25.0))
    surfaces.append(Surface(name='Grating_150lpmm', surface_type=SurfaceType.GRATING, grating_lines_per_mm=150.0, grating_order=1, thickness=0.0, semi_diameter=12.7, glass='MIRROR'))
    surfaces.append(Surface(name='CB_After_Grating', surface_type=SurfaceType.COORDINATE_BREAK, tilt_x=-25.0, thickness=25.0)) # Light travels +Z again!
    
    # M2 Focusing (EFL 25.0 -> Radius -50.0)
    surfaces.append(Surface(name='CB_Before_M2', surface_type=SurfaceType.COORDINATE_BREAK, tilt_x=15.0))
    surfaces.append(Surface(name='M2_Focus', surface_type=SurfaceType.MIRROR, radius=-50.0, thickness=0.0, semi_diameter=12.7, glass='MIRROR'))
    surfaces.append(Surface(name='CB_After_M2', surface_type=SurfaceType.COORDINATE_BREAK, tilt_x=15.0, thickness=-25.0))

    # DMD (tilted 12 deg locally)
    # Since light is travelling -Z, we want to reflect it back to +Z.
    surfaces.append(Surface(name='CB_Before_DMD', surface_type=SurfaceType.COORDINATE_BREAK, tilt_x=-12.0))
    surfaces.append(Surface(name='DMD', surface_type=SurfaceType.MIRROR, thickness=0.0, semi_diameter=5.0, glass='MIRROR'))
    surfaces.append(Surface(name='CB_After_DMD', surface_type=SurfaceType.COORDINATE_BREAK, tilt_x=-12.0, thickness=50.0))
    
    # M3 Relay (EFL 50.0 -> Radius -100.0)
    surfaces.append(Surface(name='CB_Before_M3', surface_type=SurfaceType.COORDINATE_BREAK, tilt_x=15.0))
    surfaces.append(Surface(name='M3_Relay', surface_type=SurfaceType.MIRROR, radius=-100.0, thickness=0.0, semi_diameter=12.7, glass='MIRROR'))
    surfaces.append(Surface(name='CB_After_M3', surface_type=SurfaceType.COORDINATE_BREAK, tilt_x=15.0, thickness=-75.0))

    # M4 Condenser (EFL 25.0 -> Radius -50.0)
    surfaces.append(Surface(name='CB_Before_M4', surface_type=SurfaceType.COORDINATE_BREAK, tilt_x=-15.0))
    surfaces.append(Surface(name='M4_Condenser', surface_type=SurfaceType.MIRROR, radius=-50.0, thickness=0.0, semi_diameter=12.7, glass='MIRROR'))
    surfaces.append(Surface(name='CB_After_M4', surface_type=SurfaceType.COORDINATE_BREAK, tilt_x=-15.0, thickness=25.0))

    # Detector
    surfaces.append(Surface(name='Detector', surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=0.4, glass='AIR'))

    stack = SurfaceStack(); stack.surfaces = surfaces
    return OpticalSystem(stack)

sys_opt = build_mirror_system()


from v2.engine.tracer import DualSpeedTracer

res_rays = []
for wl in [0.8, 1.75, 2.7]:
    for oy in np.linspace(-0.1, 0.1, 5):
        for phi in np.linspace(-0.1, 0.1, 5):
            r = Ray(pos=np.array([0., oy, 0.]), dir=np.array([0., np.sin(phi), np.cos(phi)]), wavelength=wl)
            tr = sys_opt.trace_ray(r)
            if not tr.vignetted:
                res_rays.append(tr)

class MockRes:
    def __init__(self, rays):
        self.ray_paths = [[[h['x_global'], h['y_global'], h['z_global']] for h in r.history] for r in rays]

res = MockRes(res_rays)


print("Trace completed.")



# Save to JSON
sys_opt.stack.save_to_json('spectrometer_nir_mirror_v2.json')
sys_opt.export_step('Final_Assembly_NIR_Mirror_V2.step', include_air=True, merge_glass=True)
print("Saved JSON and STEP files!")

# Save plot
fig, ax = plt.subplots(figsize=(10, 8))
sys_opt._update_accumulated_positions()
for i, s in enumerate(sys_opt.surfaces):
    if not getattr(s, 'is_coord_break', False):
        ox, oy, oz = s.global_origin
        ax.plot(oz, oy, 'k|', markersize=20)
        ax.text(oz, oy, s.name, fontsize=8)

for ray_path in res.ray_paths:
    z_pts = [p[2] for p in ray_path]
    y_pts = [p[1] for p in ray_path]
    ax.plot(z_pts, y_pts, 'b-', alpha=0.3)

ax.set_aspect('equal')
plt.savefig('nir_mirror_layout.png')
