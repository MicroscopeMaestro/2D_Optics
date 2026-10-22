import sys, os
sys.path.insert(0, os.path.abspath('.'))
from optics_app.engine.lde import SurfaceStack
from optics_app.cad.exporter import get_exportable_elements

st = SurfaceStack()
st.load_dict(SurfaceStack.load_from_json(r'C:\Users\Juan\Desktop\spectrometer_raman_50mm_100mm_manual_optimized.json'))
print("Surfaces loaded:", len(st.surfaces))
elems = get_exportable_elements(st)
print(f"Total exportable elements: {len(elems)}")
for el in elems:
    print(f"[{el['index']:02d}] {el['name']:<35} | Type: {el['type']:<15} | surfs: {el['surface_indices']} | default_sel: {el['is_default_selected']}")
