import cadquery as cq

print("Building Precision-Aligned Spectrometer Mount (0.9mm Offset)...")

# --- DIMENSIONS ---
m4_tilt_clearance = 5.2
hole_spacing = 30.0

# 1. OPTICAL ALIGNMENT FIX: 
# The active micromirror array is exactly 0.9 mm off-center on the physical 20.7mm ceramic package.
# Since the spectrometer holes are centered on the optical beam (the array), 
# we must shift the ceramic pocket by +0.9 mm so the array lands exactly at Y=0.
pocket_y_offset = 0.9 

# 2. Heatsink pocket length is 43.5 mm
heatsink_length = 43.5
heatsink_width = 14.5

dmd_length = 20.9
dmd_width = 9.3

# 3. VERTICAL SHIFT FIX:
# The EVM heatsink extends to -20.85 mm, but the spectrometer floor is at -19.88 mm.
# To prevent the heatsink from crashing into the floor, we must lower the M4 holes on the mount by 3.0 mm.
# When these lowered holes align with the casing, the entire mount is lifted UP by 3.0 mm, giving the heatsink clearance.
hole_y_shift = -3.0

def get_efficient_robust_base(thickness):
    # To keep the holes safely inside the block, we also shift the outer block down by hole_y_shift 
    # (which effectively centers the block on the lowered holes)
    base = cq.Workplane("XY").box(45.0, 45.0, thickness, centered=(True, True, False)).translate((0, hole_y_shift, 0))
    base = base.edges("|Z").chamfer(6.0)
    # Weight-Relief Edge Scoops
    base = base.faces(">Z").workplane().center(0, 22.5 + hole_y_shift).rect(18.0, 6.0).cutThruAll()
    base = base.faces(">Z").workplane().center(0, -22.5 + hole_y_shift).rect(18.0, 6.0).cutThruAll()
    return base

# --- PART 1: THE ALIGNMENT FRAME ---
frame = get_efficient_robust_base(3.8)

# Cut the heatsink pocket, shifted by exactly +0.9 mm
frame = frame.faces(">Z").workplane().center(0, pocket_y_offset).rect(heatsink_length, heatsink_width).cutBlind(-1.0)

# Cut the DMD pocket, shifted by exactly +0.9 mm
frame = frame.faces(">Z").workplane(offset=-1.0).center(0, pocket_y_offset).rect(dmd_length, dmd_width).cutThruAll()

# Add the 4x Mounting Holes (These are lowered by 3.0mm to lift the mount up)
frame = (frame.faces(">Z").workplane().center(0, hole_y_shift)
         .rect(hole_spacing, hole_spacing, forConstruction=True)
         .vertices()
         .hole(m4_tilt_clearance))
frame = frame.edges("<Z").chamfer(0.5)

# --- PART 2: THE CLAMPING LID ---
lid_thickness = 3.0
lid = get_efficient_robust_base(lid_thickness)

# Shift the cooling fin aperture to match the heatsink
lid = lid.faces(">Z").workplane().center(0, pocket_y_offset).rect(40.0, 12.0).cutThruAll()

# Add the 4x Mounting Holes
lid = (lid.faces(">Z").workplane().center(0, hole_y_shift)
       .rect(hole_spacing, hole_spacing, forConstruction=True)
       .vertices()
       .hole(m4_tilt_clearance))
lid = lid.edges("<Z").chamfer(0.5)

# --- EXPORT ---
cq.exporters.export(frame, "DMD4500_Alignment_Frame_Final.step")
cq.exporters.export(frame, "DMD4500_Alignment_Frame_Final.stl")
cq.exporters.export(lid, "DMD4500_Clamping_Lid_Final.step")
cq.exporters.export(lid, "DMD4500_Clamping_Lid_Final.stl")

print("Successfully generated Precision-Aligned Frame and Lid files!")
