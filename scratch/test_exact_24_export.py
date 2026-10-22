import sys, os
sys.path.insert(0, os.path.abspath('.'))
from optics_app.engine.lde import SurfaceStack
from optics_app.gui.visualizer_3d import Visualizer3DWidget
from optics_app.cad.exporter import STEPExporter

res = SurfaceStack.load_from_json(r'C:\Users\Juan\Desktop\spectrometer_raman_50mm_100mm_manual_optimized.json')
stack = res['stack']
print(f"Verified surfaces count: {len(stack.surfaces)}")

# Test STEP export with exact layout
exporter = STEPExporter()
success = exporter.export(stack, 'scratch/test_spectrometer_exact_24.step')
print('Export success:', success)
if os.path.exists('scratch/test_spectrometer_exact_24.step'):
    sz = os.path.getsize('scratch/test_spectrometer_exact_24.step')
    print(f'STEP file size: {sz:,} bytes')
