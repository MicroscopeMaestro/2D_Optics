import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from optics_app.engine.lde import SurfaceStack
from optics_app.cad.exporter import _extract_surface_stack

stack = SurfaceStack.load_from_json(r'C:\Users\Juan\Desktop\spectrometer_raman_50mm_100mm_manual_optimized.json')
extracted = _extract_surface_stack(stack)
print("Extracted:", extracted)
if extracted:
    print("Surfaces in extracted:", len(getattr(extracted, 'surfaces', [])))
