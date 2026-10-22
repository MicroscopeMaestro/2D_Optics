import sys
sys.path.append(r'C:\Program Files\FreeCAD 1.1\bin')
import FreeCAD, Part

# === REVISED SPLIT ===
# All 4 M4 holes are at Z > 94. The user said "keep 2 M4 holes in the DMD region".
# This means we should split BETWEEN the two pairs of M4 holes:
#   Near pair: Z=99.26, Y=-52.64  -> keep in OPTICS region
#   Far  pair: Z=107.23, Y=-81.56 -> keep in DMD region
#
# So Z_SPLIT should be between 99.26 and 107.23.
# We'll use Z = 103.0 as the split plane.
#
# At Z=103: the DMD arm already exists (started at Z=95), so this is safe.
# The optics region (Z<103) will have the near M4 holes.
# The DMD region (Z>=103) will have the far 2 M4 holes + the full DMD aperture.

Z_SPLIT = 103.0

print("Loading support_top_half.step ...")
shape = Part.Shape()
shape.read('support_top_half.step')
bb = shape.BoundBox
print(f"Shape volume: {shape.Volume:.2f}")

# Probe at split zone
for z_probe in [95, 98, 99, 100, 101, 102, 103, 104, 105, 107, 110]:
    slab = shape.common(Part.makeBox(100, 300, 1, FreeCAD.Vector(-50, -200, z_probe)))
    sb = slab.BoundBox
    if slab.Volume > 0:
        print(f"  Z={z_probe}: Y=[{sb.YMin:.1f},{sb.YMax:.1f}] vol={slab.Volume:.1f}")
    else:
        print(f"  Z={z_probe}: no material")

print(f"\nSplitting at Z = {Z_SPLIT} ...")

# DMD region: Z >= Z_SPLIT (contains far M4 holes at Z=107.23)
Z_dmd_size = bb.ZMax - Z_SPLIT + 10
dmd_keeper = Part.makeBox(200, 300, Z_dmd_size, FreeCAD.Vector(-100, -200, Z_SPLIT))
dmd_piece = shape.common(dmd_keeper)

# Optics region: Z < Z_SPLIT (contains near M4 holes at Z=99.26)
Z_opt_size = Z_SPLIT - bb.ZMin + 10
optics_keeper = Part.makeBox(200, 300, Z_opt_size, FreeCAD.Vector(-100, -200, bb.ZMin - 10))
optics_piece = shape.common(optics_keeper)

print(f"  DMD piece    volume: {dmd_piece.Volume:.2f}")
print(f"  Optics piece volume: {optics_piece.Volume:.2f}")
print(f"  Sum: {dmd_piece.Volume + optics_piece.Volume:.2f}  (original: {shape.Volume:.2f})")

if dmd_piece.isValid() and optics_piece.isValid() and dmd_piece.Volume > 100 and optics_piece.Volume > 100:
    dmd_piece.exportStep('top_dmd_region.step')
    optics_piece.exportStep('top_optics_region.step')
    print(f"\nSUCCESS: Exported top_dmd_region.step and top_optics_region.step")
    print(f"  DMD region   Z: [{dmd_piece.BoundBox.ZMin:.2f}, {dmd_piece.BoundBox.ZMax:.2f}]")
    print(f"  Optics region Z: [{optics_piece.BoundBox.ZMin:.2f}, {optics_piece.BoundBox.ZMax:.2f}]")
    print(f"\nM4 hole distribution:")
    print(f"  Optics piece: near pair at Z=99.26, Y=-52.64 (2 holes)")
    print(f"  DMD piece:    far pair  at Z=107.23, Y=-81.56 (2 holes, deep in DMD arm)")
else:
    print("ERROR: Invalid pieces!")
