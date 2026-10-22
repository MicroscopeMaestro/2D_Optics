import sys, os
sys.path.insert(0, os.path.abspath('.'))
from optics_app.engine.lde import SurfaceStack
from optics_app.cad.exporter import get_exportable_elements

res = SurfaceStack.load_from_json(r'C:\Users\Juan\Desktop\spectrometer_raman_50mm_100mm_manual_optimized.json')
stack = res['stack']
print(f"Total surfaces in stack: {len(stack.surfaces)}")

elems = get_exportable_elements(stack)
print(f"Total exportable elements: {len(elems)}")
for el in elems:
    print(f"[{el['index']:02d}] {el['name']:<35} | Type: {el['type']:<15} | surfs: {el['surface_indices']} | default_sel: {el['is_default_selected']}")
