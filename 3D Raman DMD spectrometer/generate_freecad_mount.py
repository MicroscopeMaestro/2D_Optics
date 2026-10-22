import sys
sys.path.append(r'C:\Program Files\FreeCAD 1.1\bin')
import FreeCAD, Part

# --- PARAMETERS ---
m4_tilt_clearance = 2.6 # radius for 5.2 diameter clearance
hole_spacing = 30.0

pocket_y_offset = 0.9 

heatsink_length = 43.5
heatsink_width = 14.5

dmd_length = 20.9
dmd_width = 9.3

# LOWER M4 HOLES:
# User requested lower M4 holes to lift the entire mount + heatsink to avoid crashing into the spectrometer floor.
# By lowering the holes on the block, the block moves up when mounted.
hole_y_shift = -3.0

def make_efficient_robust_base(thickness):
    # Base box, 45x45xThickness
    # We shift it vertically by hole_y_shift so the M4 holes remain roughly centered on the physical block.
    # The optical center (0,0) is now inside the upper half of the block.
    box = Part.makeBox(45.0, 45.0, thickness, FreeCAD.Vector(-22.5, -22.5 + hole_y_shift, 0))
    
    # Weight Relief edge scoops
    scoop1 = Part.makeBox(18.0, 6.0, thickness, FreeCAD.Vector(-9.0, 19.5 + hole_y_shift, 0))
    scoop2 = Part.makeBox(18.0, 6.0, thickness, FreeCAD.Vector(-9.0, -25.5 + hole_y_shift, 0))
    box = box.cut(scoop1).cut(scoop2)
    return box

# --- ALIGNMENT FRAME ---
frame = make_efficient_robust_base(3.8)

# DMD Pocket
# 1.0mm offset from optical plane (+Z face) means it cuts down to Z=2.8
dmd_pocket = Part.makeBox(dmd_length, dmd_width, 3.8, FreeCAD.Vector(-dmd_length/2, pocket_y_offset - dmd_width/2, 1.0))
frame = frame.cut(dmd_pocket)

# Heatsink Pocket (cut from the back, Z=0 to Z=1.0)
hs_pocket = Part.makeBox(heatsink_length, heatsink_width, 1.0, FreeCAD.Vector(-heatsink_length/2, pocket_y_offset - heatsink_width/2, 0))
frame = frame.cut(hs_pocket)

# 4x Mounting Holes (Shifted by hole_y_shift)
for x in [-hole_spacing/2, hole_spacing/2]:
    for y in [-hole_spacing/2 + hole_y_shift, hole_spacing/2 + hole_y_shift]:
        h = Part.makeCylinder(m4_tilt_clearance, 3.8, FreeCAD.Vector(x, y, 0), FreeCAD.Vector(0,0,1))
        frame = frame.cut(h)

frame.exportStep("DMD4500_Alignment_Frame_Final.step")


# --- CLAMPING LID ---
lid = make_efficient_robust_base(3.0)

# Cooling Fin Aperture
fin_aperture = Part.makeBox(40.0, 12.0, 3.0, FreeCAD.Vector(-20.0, pocket_y_offset - 6.0, 0))
lid = lid.cut(fin_aperture)

# 4x Mounting Holes
for x in [-hole_spacing/2, hole_spacing/2]:
    for y in [-hole_spacing/2 + hole_y_shift, hole_spacing/2 + hole_y_shift]:
        h = Part.makeCylinder(m4_tilt_clearance, 3.0, FreeCAD.Vector(x, y, 0), FreeCAD.Vector(0,0,1))
        lid = lid.cut(h)

lid.exportStep("DMD4500_Clamping_Lid_Final.step")
print("Done generating mounts in FreeCAD!")
