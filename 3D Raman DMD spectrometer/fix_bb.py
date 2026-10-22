with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

new_bb = '''    # 2.95 Add Breadboard Slab matching the FINAL bounding box!
    if verbose: print(f"[STEP 2.95/7] Adding {breadboard_thickness}mm breadboard slab matching final footprint...")
    bbox = top_half.BoundBox
    bb_x_min = -(CLAMSHELL_HALF_THICKNESS + breadboard_thickness)
    bb_slab = Part.makeBox(breadboard_thickness, bbox.YMax - bbox.YMin, bbox.ZMax - bbox.ZMin, FreeCAD.Vector(bb_x_min, bbox.YMin, bbox.ZMin))
    bottom_half = bottom_half.fuse(bb_slab)

'''

step_4 = '    # 4. Optical Subtractions\n'
if step_4 in content:
    content = content.replace(step_4, new_bb + step_4)
    with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("Injected Breadboard Slab before Step 4")
else:
    print("Could not find Step 4")
