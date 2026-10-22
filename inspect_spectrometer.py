import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from optics_app.engine.surface import SurfaceType

def inspect_json(filepath):
    print("=" * 70)
    print(f"INSPECTING: {filepath}")
    print("=" * 70)
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Version: {data.get('version')}")
    print(f"Wavelengths: {data.get('wavelengths')}")
    print(f"Primary WL: {data.get('primary_wavelength')}")
    print(f"Source config: {data.get('source_config')}")
    
    surfaces = data.get('surfaces', [])
    print(f"\nSurfaces ({len(surfaces)} total):")
    for i, s in enumerate(surfaces):
        print(f"[{i:2d}] {s.get('name'):<25} type={s.get('surface_type'):<10} r={s.get('radius'):<8.2f} t={s.get('thickness'):<6.2f} g={s.get('glass'):<8} sd={s.get('semi_diameter'):<5.1f} tilt_y={s.get('tilt_y'):<5.1f} grating={s.get('grating_lines_per_mm'):<5.1f}")
        if s.get('conic', 0.0) != 0.0 or s.get('asphere_coeffs'):
            print(f"     asphere: conic={s.get('conic')}, coeffs={s.get('asphere_coeffs')}")
        if s.get('aperture_type'):
            print(f"     aperture: type={s.get('aperture_type')}, x_half={s.get('aperture_x_half')}, y_half={s.get('aperture_y_half')}, r_outer={s.get('aperture_r_outer')}")

    # Now load with SurfaceStack
    loaded = SurfaceStack.load_from_json(filepath)
    stack = loaded["stack"] if isinstance(loaded, dict) else loaded
    system = OpticalSystem(stack)
    print("\nOpticalSystem created successfully. Active surfaces:", len(system.surfaces))
    return data, stack, system

if __name__ == "__main__":
    inspect_json("hybrid_raman_spectrometer.json")
