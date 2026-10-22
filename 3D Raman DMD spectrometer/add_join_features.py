import sys
sys.path.append(r'C:\Program Files\FreeCAD 1.1\bin')
import FreeCAD, Part

# === ADD ALIGNMENT/JOINING FEATURES TO THE SPLIT FACE ===
# The two pieces mate at Z ≈ 103. We add:
#   1. Two M3 alignment pin holes (dowels) to guarantee registration
#   2. Two M3 screw through-holes on optics piece + M3 tap holes on DMD piece
#      so the two printed pieces can be bolted together
#
# The mating face is at Z = 103.
# The optics piece Y range at Z=103 is approximately Y = [-25, 25] (main channel only)
# So we place the joining features in the main channel area at:
#   Pins:   (X=8.75, Y=10, Z=103) and (X=8.75, Y=-10, Z=103)
#   Screws: same positions (combined pin/screw approach)

Z_SPLIT    = 103.0
X_CENTER   = 8.75   # center of wall thickness (0 to 17.5 mm)
M3_CLEAR_R = 1.75   # M3 clearance hole radius (3.5mm dia for plastic)
M3_TAP_R   = 1.25   # M3 tap drill radius (2.5mm dia)
DOWEL_DEPTH = 5.0   # Pin pocket depth on each side
SCREW_LENGTH = 12.0 # Screw through-depth on optics side

# Feature Y positions on mating face (within Y=-25 to Y=25 safe zone)
feature_ys = [12.0, -12.0]   # symmetric pair

print("Loading pieces ...")
dmd_shape = Part.Shape(); dmd_shape.read('top_dmd_region.step')
opt_shape = Part.Shape(); opt_shape.read('top_optics_region.step')
print(f"DMD piece volume before: {dmd_shape.Volume:.2f}")
print(f"Optics piece volume before: {opt_shape.Volume:.2f}")

# For each feature Y position, add:
# DMD piece: M3 tap drill from the Z=103 face going INTO the DMD piece (Z direction)
# Optics piece: M3 clearance hole from the Z=103 face going INTO the optics piece (-Z direction)
for fy in feature_ys:
    # DMD piece: tap hole at Z=103, going in +Z direction
    tap = Part.makeCylinder(M3_TAP_R, SCREW_LENGTH, FreeCAD.Vector(X_CENTER, fy, Z_SPLIT), FreeCAD.Vector(0,0,1))
    dmd_shape = dmd_shape.cut(tap)
    
    # Optics piece: clearance hole at Z=103, going in -Z direction
    clr = Part.makeCylinder(M3_CLEAR_R, SCREW_LENGTH, FreeCAD.Vector(X_CENTER, fy, Z_SPLIT - SCREW_LENGTH), FreeCAD.Vector(0,0,1))
    opt_shape = opt_shape.cut(clr)

print(f"DMD piece volume after: {dmd_shape.Volume:.2f}")
print(f"Optics piece volume after: {opt_shape.Volume:.2f}")

if dmd_shape.isValid() and opt_shape.isValid():
    dmd_shape.exportStep('top_dmd_region.step')
    opt_shape.exportStep('top_optics_region.step')
    print("\nSUCCESS: Updated pieces with M3 joining holes exported.")
    print(f"  M3 tap holes in DMD piece at Y={feature_ys}, Z={Z_SPLIT} (depth {SCREW_LENGTH}mm)")
    print(f"  M3 clearance holes in optics piece at same Y positions (depth {SCREW_LENGTH}mm)")
else:
    print("ERROR: Pieces invalid after feature addition!")
