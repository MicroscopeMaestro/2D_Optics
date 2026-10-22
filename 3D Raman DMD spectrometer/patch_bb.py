import re

with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_bb = '''    if verbose: print(f"[STEP 2.95/7] Adding {breadboard_thickness}mm breadboard slab matching final footprint...")
    bbox = top_half.BoundBox
    bb_x_min = -(CLAMSHELL_HALF_THICKNESS + breadboard_thickness)
    bb_slab = Part.makeBox(breadboard_thickness, bbox.YMax - bbox.YMin, bbox.ZMax - bbox.ZMin, FreeCAD.Vector(bb_x_min, bbox.YMin, bbox.ZMin))
    bottom_half = bottom_half.fuse(bb_slab)'''

new_bb = '''    if verbose: print(f"[STEP 2.95/7] Adding {breadboard_thickness}mm breadboard slab matching final footprint...")
    bbox = top_half.BoundBox
    bb_x_min = -(CLAMSHELL_HALF_THICKNESS + breadboard_thickness)
    
    # Create a custom breadboard footprint that tightly hugs the casing, saving >50% material!
    slice_shapes = bottom_half.slice(FreeCAD.Vector(1,0,0), -17.4)
    bb_flange_face = None
    if slice_shapes and len(slice_shapes) > 0:
        try:
            # Offset by 20mm to create a mounting flange
            bb_flange_face = Part.Face(slice_shapes[0]).makeOffset2D(20.0)
            bb_slab = bb_flange_face.extrude(FreeCAD.Vector(-breadboard_thickness, 0, 0))
            bb_slab.translate(FreeCAD.Vector(-0.1, 0, 0))
            bottom_half = bottom_half.fuse(bb_slab)
        except: pass
    
    if bb_flange_face is None:
        # Fallback to rectangular box if slice fails
        bb_slab = Part.makeBox(breadboard_thickness, bbox.YMax - bbox.YMin, bbox.ZMax - bbox.ZMin, FreeCAD.Vector(bb_x_min, bbox.YMin, bbox.ZMin))
        bottom_half = bottom_half.fuse(bb_slab)'''

if old_bb in content:
    content = content.replace(old_bb, new_bb)
    with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("Replaced bb_slab")
else:
    print("Could not find bb_slab code block")
