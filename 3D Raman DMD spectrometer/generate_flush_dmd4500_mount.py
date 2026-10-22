import sys
sys.path.append(r'C:\Program Files\FreeCAD 1.1\bin')
import FreeCAD, Part

# --- PARAMETERS ---
m4_tilt_clearance = 2.15 # 4.3 diameter for Close Fit M4 clearance holes
hole_spacing = 30.0

pocket_y_offset = 0.9 

heatsink_length = 43.5 # Reverted to original length
heatsink_width = 15.5  # Increased by 1mm as requested

dmd_length = 20.9
dmd_width = 9.3

# The spectrometer casing floor is at Y = -19.88 relative to the DMD optical center.
# We truncate the bottom edge of the mount to Y = -18.0 to leave a guaranteed air gap
# above the floor, reducing the spacing below the holes to 3.0 mm.
y_min = -18.0
y_max = 22.5
actual_height = y_max - y_min

def make_efficient_robust_base(thickness, top_y):
    # Base box truncated at the bottom (y_min) and top (top_y) to be uniform and constant
    actual_height = top_y - y_min
    box = Part.makeBox(45.0, actual_height, thickness, FreeCAD.Vector(-22.5, y_min, 0))
    return box

# --- ALIGNMENT FRAME ---
# Frame truncated uniformly across the top at Y = 18.5
frame = make_efficient_robust_base(3.8, top_y=18.5)

# DMD Pocket
# 1.0mm offset from optical plane (+Z face) means it cuts down to Z=2.8
dmd_pocket = Part.makeBox(dmd_length, dmd_width, 3.8, FreeCAD.Vector(-dmd_length/2, pocket_y_offset - dmd_width/2, 1.0))
frame = frame.cut(dmd_pocket)

# Heatsink Pocket (cut from the back, Z=0 to Z=1.0)
hs_pocket = Part.makeBox(heatsink_length, heatsink_width, 1.0, FreeCAD.Vector(-heatsink_length/2, pocket_y_offset - heatsink_width/2, 0))
frame = frame.cut(hs_pocket)

# 4x Mounting Holes 
for x in [-hole_spacing/2, hole_spacing/2]:
    for y in [-hole_spacing/2, hole_spacing/2]:
        h = Part.makeCylinder(m4_tilt_clearance, 3.8, FreeCAD.Vector(x, y, 0), FreeCAD.Vector(0,0,1))
        frame = frame.cut(h)

frame.exportStep("DMD4500_Alignment_Frame_Final.step")


# --- CLAMPING LID ---
# Lid truncated uniformly across the top even shorter, at Y = 17.5
lid = make_efficient_robust_base(3.0, top_y=17.5)

# Cooling Fin Aperture
fin_aperture = Part.makeBox(40.0, 12.0, 3.0, FreeCAD.Vector(-20.0, pocket_y_offset - 6.0, 0))
lid = lid.cut(fin_aperture)

# 4x Mounting Holes
for x in [-hole_spacing/2, hole_spacing/2]:
    for y in [-hole_spacing/2, hole_spacing/2]:
        h = Part.makeCylinder(m4_tilt_clearance, 3.0, FreeCAD.Vector(x, y, 0), FreeCAD.Vector(0,0,1))
        lid = lid.cut(h)

lid.exportStep("DMD4500_Clamping_Lid_Final.step")
print("Done generating corrected flush mounts in FreeCAD!")
