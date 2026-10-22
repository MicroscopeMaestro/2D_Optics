import re

with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Remove the old Breadboard slab block
bb_code = '''    # 2. Add Breadboard Slab
    if verbose: print(f"[STEP 2/7] Adding {breadboard_thickness}mm breadboard slab...")
    bb_x_min = -(CLAMSHELL_HALF_THICKNESS + breadboard_thickness)
    bb_slab = Part.makeBox(breadboard_thickness, bbox.YMax - bbox.YMin, bbox.ZMax - bbox.ZMin, FreeCAD.Vector(bb_x_min, bbox.YMin, bbox.ZMin))
    bottom_half = bottom_half.fuse(bb_slab)'''
content = content.replace(bb_code, '')

# 2. Add it back right before STEP 3
step_3 = '    # 3. Add External Walls (Optional)\n'
new_bb = '''    # 2.9 Add Breadboard Slab matching the FINAL bounding box!
    if verbose: print(f"[STEP 2.9/7] Adding {breadboard_thickness}mm breadboard slab matching final footprint...")
    bbox = top_half.BoundBox
    bb_x_min = -(CLAMSHELL_HALF_THICKNESS + breadboard_thickness)
    bb_slab = Part.makeBox(breadboard_thickness, bbox.YMax - bbox.YMin, bbox.ZMax - bbox.ZMin, FreeCAD.Vector(bb_x_min, bbox.YMin, bbox.ZMin))
    bottom_half = bottom_half.fuse(bb_slab)

'''
content = content.replace(step_3, new_bb + step_3)

with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Moved bb_slab properly!")
