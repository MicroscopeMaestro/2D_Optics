import re

with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_grid_def = '''def compute_breadboard_grid(bbox, screws, pegs, left_ext, right_ext, spacing, edge_margin):
    safe_bb = []
    y_start = (bbox.YMin - left_ext) + edge_margin
    y_end = (bbox.YMax + right_ext) - edge_margin'''

new_grid_def = '''def compute_breadboard_grid(bbox, screws, pegs, left_ext, right_ext, spacing, edge_margin, bb_flange_face=None):
    safe_bb = []
    y_start = (bbox.YMin - left_ext) + edge_margin
    y_end = (bbox.YMax + right_ext) - edge_margin'''

content = content.replace(old_grid_def, new_grid_def)

old_grid_loop = '''            if not any(math.hypot(y - sy, z - sz) < 5.95 for sy, sz in screws + pegs):
                safe_bb.append((round(y, 1), round(z, 1)))'''

new_grid_loop = '''            if not any(math.hypot(y - sy, z - sz) < 5.95 for sy, sz in screws + pegs):
                if bb_flange_face is None or bb_flange_face.distToShape(Part.Vertex(FreeCAD.Vector(-17.4, y, z)))[0] < 1e-3:
                    safe_bb.append((round(y, 1), round(z, 1)))'''

content = content.replace(old_grid_loop, new_grid_loop)

old_call = '''    safe_bb = list(manual_bb) if manual_bb else compute_breadboard_grid(bbox, screws, pegs, breadboard_left_extension, breadboard_right_extension, breadboard_spacing, BREADBOARD_EDGE_MARGIN)'''
new_call = '''    safe_bb = list(manual_bb) if manual_bb else compute_breadboard_grid(bbox, screws, pegs, breadboard_left_extension, breadboard_right_extension, breadboard_spacing, BREADBOARD_EDGE_MARGIN, bb_flange_face if 'bb_flange_face' in locals() else None)'''

content = content.replace(old_call, new_call)

with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Updated grid computation")
