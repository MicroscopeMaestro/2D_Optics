import re

with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_step5 = '''    # 5. Fasteners
    if verbose: print("[STEP 5/7] Fasteners & Pegs...")
    screws, pegs = compute_safe_fastener_locations(face, wire, bbox, optics_bboxes, corner_size, manual_screws, manual_pegs)
    for (y, z) in pegs:
        bottom_half = bottom_half.fuse(Part.makeCylinder(PEG_RADIUS, PEG_HEIGHT, FreeCAD.Vector(0, y, z), FreeCAD.Vector(1, 0, 0)))
        top_half = top_half.cut(Part.makeCylinder(PEG_HOLE_RADIUS, PEG_HOLE_DEPTH, FreeCAD.Vector(0, y, z), FreeCAD.Vector(1, 0, 0)))
    for (y, z) in screws:'''

new_step5 = '''    # 5. Fasteners
    if verbose: print("[STEP 5/7] Fasteners (Pegs removed per user request)...")
    screws, pegs = compute_safe_fastener_locations(face, wire, bbox, optics_bboxes, corner_size, manual_screws, manual_pegs)
    for (y, z) in screws:'''

content = content.replace(old_step5, new_step5)

with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Removed pegs")
