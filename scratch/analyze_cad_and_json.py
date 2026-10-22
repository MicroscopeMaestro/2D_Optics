import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.cad.exporter import CADExporter, STEPExporter

def analyze():
    print("=== DETAILED CAD & JSON ANALYSIS ===")
    
    files = ["hybrid_raman_spectrometer.json", "hybrid_raman_1064_full_range.json", "hybrid_raman_1064.json"]
    
    for f in files:
        fpath = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", f))
        if os.path.exists(fpath):
            print(f"\n--- Analyzing {f} ---")
            with open(fpath, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            print(f"Version: {data.get('version')}")
            surfaces = data.get("surfaces", [])
            print(f"Surface count: {len(surfaces)}")
            for i, s in enumerate(surfaces):
                print(f"  [{i:2d}] {s.get('name'):<25} type={s.get('surface_type'):<10} R={s.get('radius'):<7} t={s.get('thickness'):<6} glass={s.get('glass'):<8} semi={s.get('semi_diameter'):<5} tilt_y={s.get('tilt_y')}")
            
            # Test STEP and DXF export
            step_out = fpath.replace(".json", ".step")
            dxf_out = fpath.replace(".json", ".dxf")
            
            loaded = SurfaceStack.load_from_json(fpath)
            stack = loaded["stack"] if isinstance(loaded, dict) else loaded
            
            res_step = CADExporter.export_to_step(stack, step_out)
            res_dxf = CADExporter.export_to_dxf_2d(stack, dxf_out)
            
            step_size = os.path.getsize(step_out) if os.path.exists(step_out) else 0
            dxf_size = os.path.getsize(dxf_out) if os.path.exists(dxf_out) else 0
            
            print(f"STEP Export: success={res_step.is_success}, size={step_size} bytes, components={res_step.num_components_exported}")
            print(f"DXF Export: path={res_dxf}, size={dxf_size} bytes")

if __name__ == "__main__":
    analyze()
