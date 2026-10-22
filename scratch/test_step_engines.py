import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from optics_app.engine.lde import SurfaceStack
from optics_app.cad.exporter import STEPExporter, STEPExportEngine

stack = SurfaceStack.load_from_json("hybrid_raman_1064.json")["stack"]

# Test pure python export
res_py = STEPExporter.export(stack, "scratch/test_python_engine.step", engine=STEPExportEngine.PURE_PYTHON)
print("Pure Python Engine export:", res_py)

# Test build123d export
res_b3d = STEPExporter.export(stack, "scratch/test_b3d_engine.step", engine=STEPExportEngine.BUILD123D)
print("Build123d Engine export:", res_b3d)

if os.path.exists("scratch/test_b3d_engine.step"):
    try:
        import build123d as b3d
        imp = b3d.import_step("scratch/test_b3d_engine.step")
        print("Imported b3d STEP:", type(imp))
        solids = imp.solids() if hasattr(imp, "solids") else []
        print(f"Solids count in b3d STEP: {len(solids)}")
        for i, s in enumerate(solids):
            print(f"  Solid {i+1}: vol={s.volume:.2f} mm^3, is_valid={s.is_valid()}")
    except Exception as e:
        print("Error inspecting b3d step:", e)
