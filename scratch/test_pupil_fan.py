import sys, os
sys.path.insert(0, os.path.abspath('.'))

from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray

res = SurfaceStack.load_from_json(r'C:\Users\Juan\Desktop\spectrometer_raman_50mm_100mm_manual_optimized.json')
stack = res['stack']
wls = res.get('wavelengths', [1.064])

sys_opt = OpticalSystem(stack)

# Generate pupil fan rays
pupil_rays = sys_opt.generate_pupil_rays(num_rays=9, wavelength=wls[0])
print(f"Generated {len(pupil_rays)} pupil rays")
for i, r in enumerate(pupil_rays):
    tr = sys_opt.trace_ray(r)
    hist = getattr(tr, 'history', [])
    valid = getattr(tr, 'valid', True)
    err = getattr(tr, 'error_message', '')
    last_z = hist[-1].get('z_global', 0.0) if hist else 0.0
    print(f"Ray {i:02d} (py={r.pos[1]:.2f}): valid={valid} steps={len(hist)} last_z={last_z:.2f} err='{err}'")
