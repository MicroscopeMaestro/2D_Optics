import json
import shutil
import sys
import numpy as np

sys.path.insert(0, ".")
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray

files_to_update = [
    r"C:\Users\Juan\Desktop\spectrometer_raman_50mm_100mm_manual.json",
    r"C:\Users\Juan\Desktop\spectrometer_raman_50mm_100mm.json"
]

for filepath in files_to_update:
    print(f"\nProcessing: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    surfaces = data.get("surfaces", [])
    grating_idx = None
    cb_idx = None

    for i, s in enumerate(surfaces):
        name = s.get("name", "").lower()
        stype = s.get("surface_type", "")
        if "cb_after" in name or ("cb" in name and s.get("is_coord_break")):
            cb_idx = i
        elif stype == "Grating" or ("grating" in name and not s.get("is_coord_break")):
            grating_idx = i

    print(f"  Found grating at {grating_idx} ({surfaces[grating_idx]['name']}), CB at {cb_idx} ({surfaces[cb_idx]['name'] if cb_idx is not None else 'None'})")
    if grating_idx is not None and cb_idx is not None and cb_idx == grating_idx + 1:
        cb_surf = surfaces[cb_idx]
        grating_surf = surfaces[grating_idx]

        # Transfer thickness
        grating_surf["thickness"] = float(cb_surf.get("thickness", 20.0))
        grating_surf["grating_lines_per_mm"] = float(grating_surf.get("grating_lines_per_mm", 600.0))
        grating_surf["grating_order"] = int(grating_surf.get("grating_order", 1))
        grating_surf["grating_primary_wavelength_um"] = 1.064
        grating_surf["grating_blaze_wavelength_um"] = 1.064
        grating_surf["grating_blaze_angle_deg"] = 18.61
        grating_surf["grating_type"] = "Blazed Reflective"
        grating_surf["is_transmission_grating"] = False
        grating_surf["substrate_thickness"] = float(grating_surf.get("substrate_thickness", 6.0))
        grating_surf["substrate_shape"] = "Rectangular"
        grating_surf["aperture_type"] = "Rectangular"

        # Remove redundant CB surface
        surfaces.pop(cb_idx)
        print(f"  Transferred thickness {grating_surf['thickness']} mm to Grating and removed CB surface")

    # Save updated file
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    print(f"  Saved updated JSON with {len(surfaces)} surfaces")

    # Verify loading and ray tracing
    loaded = SurfaceStack.load_from_json(filepath)
    stack = loaded["stack"]
    sys_opt = OpticalSystem(stack)
    ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=1.064)
    res = sys_opt.trace_ray(ray)
    last_hit = res.history[-1]
    last_surf = stack.surfaces[-1]
    print(f"  Verification Ray Trace -> Detector ({last_surf.name}): hit={np.round([last_hit['x_global'], last_hit['y_global'], last_hit['z_global']], 2)}, vig={last_hit['vignetted']}")
    assert not last_hit['vignetted'], "Ray was vignetted!"
    print("  SUCCESS: Clean single-surface grating trace confirmed!")
