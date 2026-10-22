import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from optics_app.utils.materials import get_refractive_index

stack = SurfaceStack.load_from_json("hybrid_raman_1064.json")["stack"]

# Check glass indices across NIR range
wls = [1.064, 1.1113, 1.362, 1.6133, 1.6671]
glasses = ["N-BK7", "SF5", "B270"]
print("Glass Refractive Indices across NIR:")
for g in glasses:
    indices = [get_refractive_index(g, wl) for wl in wls]
    ind_str = ", ".join([f"{wl:.4f}µm: {n:.5f}" for wl, n in zip(wls, indices)])
    print(f"  {g:<8s}: {ind_str}")

# Check tolerance sensitivity (Monte Carlo perturbing tilt and decenter)
print("\nAlignment Tolerance & Sensitivity Stress (100 perturbations):")
np.random.seed(42)
dmd_idx = next(i for i, s in enumerate(stack.surfaces) if "DMD" in s.name.upper())
det_idx = len(stack.surfaces) - 1

all_passed_tolerance = True
max_drift_det = 0.0

for trial in range(100):
    trial_stack = stack.clone() if hasattr(stack, 'clone') else SurfaceStack.load_from_json("hybrid_raman_1064.json")["stack"]
    # Perturb surfaces with random machining tolerances (+-25 um decenter, +-0.05 deg tilt)
    for s in trial_stack.surfaces:
        s.decenter_x += np.random.uniform(-0.025, 0.025)
        s.decenter_y += np.random.uniform(-0.025, 0.025)
        s.tilt_x += np.random.uniform(-0.05, 0.05)
        s.tilt_y += np.random.uniform(-0.05, 0.05)
    
    trial_opt = OpticalSystem(trial_stack)
    chief_ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=1.362)
    traced = trial_opt.trace_rays([chief_ray])[0]
    
    det_hit = next((h for h in traced.history if h["surface_index"] == det_idx), None)
    if det_hit is None or traced.vignetted:
        all_passed_tolerance = False
        print(f"Trial {trial+1}: FAILED (Vignetted)")
    else:
        r_det = np.sqrt(det_hit["x"]**2 + det_hit["y"]**2)
        if r_det > max_drift_det:
            max_drift_det = r_det
        if r_det > 0.500:
            all_passed_tolerance = False
            print(f"Trial {trial+1}: EXCEEDED 1mm detector (r = {r_det:.4f} mm)")

print(f"100 Monte Carlo alignment perturbations result: {'PASS' if all_passed_tolerance else 'FAIL'}")
print(f"Maximum chief ray displacement under +-25µm / +-0.05° tolerance: {max_drift_det:.4f} mm (Limit: 0.500 mm)")
