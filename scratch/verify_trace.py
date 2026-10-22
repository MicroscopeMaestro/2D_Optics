import sys
import os
import json

sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/..'))
from optics_app.engine.tracer import OpticalSystem, Ray
import math

with open('spectrometer_yz_40mm_100mm_slit.json') as f:
    config = json.load(f)

system = OpticalSystem.from_json('spectrometer_yz_40mm_100mm_slit.json')

for wl_info in config["system_config"].get("wavelengths", []):
    wl = wl_info["wavelength_um"]
    r = Ray(pos=(0, 0, 0), dir=(0, 0.05, 1.0), wavelength=wl)
    traced_ray = system.trace_ray(r)
    
    final_seg = traced_ray.history[-1]
    last_surf = final_seg['surface_index']
    y = final_seg['y_global']
    z = final_seg['z_global']

    if last_surf is not None and last_surf == len(system.surfaces) - 1:
        print(f"Wavelength {wl} um: Reached Image plane successfully. Y={y:.3f}, Z={z:.3f}")
    else:
        print(f"Wavelength {wl} um: FAILED. Stopped at surface {last_surf}")
