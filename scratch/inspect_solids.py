import os, sys
import build123d as b3d

imp = b3d.import_step("scratch/test_b3d_engine.step")
solids = imp.solids()
print(f"Total solids in B3D STEP: {len(solids)}")
for i, s in enumerate(solids):
    valid = s.is_valid
    bbox = s.bounding_box()
    print(f"Solid {i+1:2d}: Vol={s.volume:9.2f} mm^3 | Valid={valid} | BoundingBox=X[{bbox.min.X:.1f}, {bbox.max.X:.1f}], Y[{bbox.min.Y:.1f}, {bbox.max.Y:.1f}], Z[{bbox.min.Z:.1f}, {bbox.max.Z:.1f}]")
