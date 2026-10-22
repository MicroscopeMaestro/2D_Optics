import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from optics_app.engine.lde import SurfaceStack
from optics_app.cad.exporter import is_refractive

stack = SurfaceStack.load_from_json(r'C:\Users\Juan\Desktop\spectrometer_raman_50mm_100mm_manual_optimized.json')
print(f"Stack surfaces count: {len(stack.surfaces)}")
for i, s in enumerate(stack.surfaces):
    print(f"i={i}: name={s.name}, is_refractive={is_refractive(s)}, glass={s.glass}, t={s.thickness}")
