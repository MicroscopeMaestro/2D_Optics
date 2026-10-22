import os
import sys

def verify_step_and_dxf():
    print("=== STEP & DXF DEEP INSPECTION ===")
    step_file = "hybrid_raman_spectrometer.step"
    dxf_file = "hybrid_raman_spectrometer.dxf"

    # 1. Inspect STEP file
    print(f"\n1. Inspecting {step_file}...")
    assert os.path.exists(step_file), f"{step_file} does not exist"
    with open(step_file, "r", encoding="utf-8", errors="ignore") as f:
        step_lines = f.readlines()
    
    print(f"Total lines: {len(step_lines)}")
    header_found = False
    data_found = False
    products = []
    manifolds = []
    
    for line in step_lines:
        if "HEADER;" in line:
            header_found = True
        if "DATA;" in line:
            data_found = True
        if "PRODUCT(" in line:
            products.append(line.strip())
        if "MANIFOLD_SOLID_BREP" in line:
            manifolds.append(line.strip())

    print(f"Header found: {header_found}")
    print(f"Data found: {data_found}")
    print(f"Number of PRODUCT entities: {len(products)}")
    for p in products:
        print(f"  {p}")
    print(f"Number of MANIFOLD_SOLID_BREP entities: {len(manifolds)}")

    # 2. Inspect DXF file
    print(f"\n2. Inspecting {dxf_file}...")
    assert os.path.exists(dxf_file), f"{dxf_file} does not exist"
    with open(dxf_file, "r", encoding="utf-8", errors="ignore") as f:
        dxf_lines = f.readlines()
    
    print(f"Total lines: {len(dxf_lines)}")
    layers = set()
    entity_counts = {}
    
    current_entity = None
    for i, line in enumerate(dxf_lines):
        line_s = line.strip()
        if line_s in ["LINE", "CIRCLE", "ARC", "TEXT", "MTEXT", "HATCH", "LWPOLYLINE", "SOLID", "POLYLINE"]:
            entity_counts[line_s] = entity_counts.get(line_s, 0) + 1
        if line_s == "8" and i + 1 < len(dxf_lines): # Layer name in DXF group code 8
            layers.add(dxf_lines[i+1].strip())

    print("Found DXF Layers:", sorted(list(layers)))
    print("Found DXF Entities:", entity_counts)

if __name__ == "__main__":
    verify_step_and_dxf()
