import os
import sys
from optics_app.engine.lde import SurfaceStack
from optics_app.cad.exporter import STEPExporter, STEPExportEngine
import build123d as b3d

loaded = SurfaceStack.load_from_json("hybrid_raman_spectrometer.json")
stack = loaded["stack"]

print("--- Exporting with build123d engine ---")
res = STEPExporter.export(stack, "test_b3d_raman.step", engine="build123d")
print("Export result:", res)

if os.path.exists("test_b3d_raman.step"):
    size = os.path.getsize("test_b3d_raman.step")
    print(f"File size: {size} bytes")
    imported = b3d.import_step("test_b3d_raman.step")
    print("Imported type:", type(imported))
    if hasattr(imported, "solids"):
        solids = imported.solids()
        print(f"Number of 3D solid bodies: {len(solids)}")
        for i, s in enumerate(solids):
            print(f"  Solid {i}: volume = {s.volume:.2f} mm^3, is_valid = {s.is_valid()}")

