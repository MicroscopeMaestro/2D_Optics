import re

with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Remove [STEP 2.95/7] completely
step_295 = '''    if verbose: print(f"[STEP 2.95/7] Adding {breadboard_thickness}mm breadboard slab matching final footprint...")
    bbox = top_half.BoundBox
    bb_x_min = -(CLAMSHELL_HALF_THICKNESS + breadboard_thickness)
    
    # Create a custom breadboard footprint that tightly hugs the casing, saving >50% material!
    slice_shapes = bottom_half.slice(FreeCAD.Vector(1,0,0), -17.4)
    bb_flange_face = None
    if slice_shapes and len(slice_shapes) > 0:
        try:
            # Offset by 12.0mm to create a mounting flange
            bb_flange_face = Part.Face(slice_shapes[0]).makeOffset2D(12.0)
            bb_slab = bb_flange_face.extrude(FreeCAD.Vector(-breadboard_thickness, 0, 0))
            bb_slab.translate(FreeCAD.Vector(-0.1, 0, 0))
            bottom_half = bottom_half.fuse(bb_slab)
        except: pass
    
    if bb_flange_face is None:
        # Fallback to rectangular box if slice fails
        bb_slab = Part.makeBox(breadboard_thickness, bbox.YMax - bbox.YMin, bbox.ZMax - bbox.ZMin, FreeCAD.Vector(bb_x_min, bbox.YMin, bbox.ZMin))
        bottom_half = bottom_half.fuse(bb_slab)'''

content = content.replace(step_295, '''    # Breadboard removed per user request
    bb_x_min = -CLAMSHELL_HALF_THICKNESS''')

# 2. Modify [STEP 6/7] to remove M6 breadboard tap holes
step_6 = '''    # 6. Breadboard
    if verbose: print("[STEP 6/7] Breadboard Tap Holes...")
    safe_bb = list(manual_bb) if manual_bb else compute_breadboard_grid(bbox, screws, pegs, breadboard_left_extension, breadboard_right_extension, breadboard_spacing, BREADBOARD_EDGE_MARGIN, bb_flange_face if 'bb_flange_face' in locals() else None)
    bb_cyls = [Part.makeCylinder(SCREW_TAP_RADIUS, breadboard_thickness + 0.5, FreeCAD.Vector(bb_x_min - 0.2, y, z), FreeCAD.Vector(1, 0, 0)) for (y, z) in safe_bb]
    if bb_cyls:
        try:
            bottom_half = bottom_half.cut(Part.makeCompound(bb_cyls))
        except:
            all_bb = bb_cyls[0]
            for c in bb_cyls[1:]: all_bb = all_bb.fuse(c)
            bottom_half = bottom_half.cut(all_bb)
    
    if enable_threads:
        rot_x = FreeCAD.Rotation(FreeCAD.Vector(0, 1, 0), 90)
        for (y, z) in screws:
            try:
                th = Part.makeThread(THREAD_PITCH, THREAD_DEPTH, 20.0, THREAD_TAP_RADIUS)
                th.transformShape(FreeCAD.Placement(FreeCAD.Vector(bb_x_min, y, z), rot_x).Matrix)
                bottom_half = bottom_half.cut(th)
            except: pass'''

new_step_6 = '''    # 6. Fastener Threads (Breadboard holes removed)
    if enable_threads:
        if verbose: print("[STEP 6/7] Threading clamshell fastener holes...")
        rot_x = FreeCAD.Rotation(FreeCAD.Vector(0, 1, 0), 90)
        for (y, z) in screws:
            try:
                th = Part.makeThread(THREAD_PITCH, THREAD_DEPTH, 20.0, THREAD_TAP_RADIUS)
                th.transformShape(FreeCAD.Placement(FreeCAD.Vector(bb_x_min, y, z), rot_x).Matrix)
                bottom_half = bottom_half.cut(th)
            except: pass'''

content = content.replace(step_6, new_step_6)

with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Breadboard removed")
