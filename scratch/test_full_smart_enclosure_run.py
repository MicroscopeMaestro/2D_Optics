import sys, os
sys.path.insert(0, os.path.abspath('.'))

from optics_app.engine.lde import SurfaceStack
from optics_app.cad.exporter import CADExporter, STEPExportOptions, STEPExportEngine

res = SurfaceStack.load_from_json(r'C:\Users\Juan\Desktop\spectrometer_raman_50mm_100mm_manual_optimized.json')
stack = res['stack']

opts = STEPExportOptions(
    engine=STEPExportEngine.BUILD123D,
    export_3d_printable_enclosure=True,
    enclosure_wall_margin_mm=10.0
)

out_path = 'scratch/Smart_Air_Gap_Enclosure_Benchmark.step'
result = CADExporter.export_system_step(stack, out_path, options=opts)
print('Export success:', result.is_success)
print('Num components in result:', result.num_components_exported)
if os.path.exists(out_path):
    print(f'File exists! Size: {os.path.getsize(out_path):,} bytes')
