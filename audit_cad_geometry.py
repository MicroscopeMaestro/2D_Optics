import os
import sys

print("--- Testing ezdxf DXF import ---")
try:
    import ezdxf
    doc = ezdxf.readfile("hybrid_raman_spectrometer.dxf")
    msp = doc.modelspace()
    print("Imported DXF successfully with ezdxf!")
    print("DXF Version:", doc.dxfversion)
    print("Total entities in modelspace:", len(msp))
    layers = [l.dxf.name for l in doc.layers]
    print("Layers defined:", layers)
    
    entity_types = {}
    for ent in msp:
        dxftype = ent.dxftype()
        entity_types[dxftype] = entity_types.get(dxftype, 0) + 1
    print("Entity breakdown:", entity_types)
except Exception as e:
    print("ezdxf error:", e)

print("\n--- Testing build123d STEP import ---")
try:
    import build123d as b3d
    imported = b3d.import_step("hybrid_raman_spectrometer.step")
    print("Imported STEP successfully with build123d!")
    print("Type:", type(imported))
    if hasattr(imported, "solids"):
        solids = imported.solids()
        print("Number of solids:", len(solids))
        for i, sol in enumerate(solids):
            print(f"Solid {i}: volume = {sol.volume:.2f} mm^3, is_valid = {sol.is_valid()}")
except Exception as e:
    print("build123d error:", e)

print("\n--- STEP Entity Parser Audit ---")
step_path = "hybrid_raman_spectrometer.step"
with open(step_path, "r", encoding="utf-8", errors="ignore") as f:
    lines = f.readlines()

entity_counts = {}
for line in lines:
    line = line.strip()
    if "=" in line:
        parts = line.split("=", 1)
        ent = parts[1].split("(")[0].strip()
        entity_counts[ent] = entity_counts.get(ent, 0) + 1

for k, v in sorted(entity_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"  {k}: {v}")

