import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optics_app.engine.lde import SurfaceStack
from optics_app.cad.exporter import STEPExporter, STEPExportOptions, export_to_step
from optics_app.cad.dxf_exporter import DXF2DExporter

print("Loading hybrid_raman_1064.json...")
stack_data = SurfaceStack.load_from_json('hybrid_raman_1064.json')
stack = stack_data['stack']

step_path = 'hybrid_raman_1064.step'
dxf_path = 'hybrid_raman_1064.dxf'

print(f"Exporting 3D STEP to {step_path}...")
step_res = STEPExporter.export(stack, step_path)
print(f"  STEP export result: success={step_res.is_success}, path={step_res.file_path}, components={step_res.num_components_exported}")

print(f"Exporting 2D DXF to {dxf_path}...")
dxf_exporter = DXF2DExporter(stack)
dxf_out = dxf_exporter.export(dxf_path, include_dimensions=True, include_rays=True, include_hatches=True, include_table=True, plane="DUAL")
print(f"  DXF export result: path={dxf_out}")

# Verify file sizes
step_size = os.path.getsize(step_path)
dxf_size = os.path.getsize(dxf_path)

print(f"\nDeliverables verification:")
print(f"  {step_path}: {step_size:,} bytes ({step_size/1024:.2f} KB)")
print(f"  {dxf_path}: {dxf_size:,} bytes ({dxf_size/1024:.2f} KB)")

assert step_size > 5000, "STEP file size too small!"
assert dxf_size > 10000, "DXF file size too small!"
print("\n[SUCCESS] CAD deliverables generated and verified successfully!")
