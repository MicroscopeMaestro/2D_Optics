with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Move the entrance screws closer to Z=18.0
old_screws = '''MANUAL_SCREW_POSITIONS = [
    ( 16.5, 21.0),   # ENTRANCE RIGHT
    (-16.5, 21.0),   # ENTRANCE LEFT'''
new_screws = '''MANUAL_SCREW_POSITIONS = [
    ( 16.5, 18.0),   # ENTRANCE RIGHT
    (-16.5, 18.0),   # ENTRANCE LEFT'''
content = content.replace(old_screws, new_screws)

# 2. Thicken the DMD lateral wall
old_dmd_housing = '''    dmd_housing_top = Part.makeBox(CLAMSHELL_HALF_THICKNESS, 65.0, 140.0, FreeCAD.Vector(0, -82.0, 95.0))
    dmd_housing_bot = Part.makeBox(CLAMSHELL_HALF_THICKNESS, 65.0, 140.0, FreeCAD.Vector(-CLAMSHELL_HALF_THICKNESS, -82.0, 95.0))'''
new_dmd_housing = '''    dmd_housing_top = Part.makeBox(CLAMSHELL_HALF_THICKNESS, 83.0, 140.0, FreeCAD.Vector(0, -100.0, 95.0))
    dmd_housing_bot = Part.makeBox(CLAMSHELL_HALF_THICKNESS, 83.0, 140.0, FreeCAD.Vector(-CLAMSHELL_HALF_THICKNESS, -100.0, 95.0))'''
content = content.replace(old_dmd_housing, new_dmd_housing)

old_poly_tl = '''    poly_tl = Part.makePolygon([
        FreeCAD.Vector(0, -90, 120), FreeCAD.Vector(0, -60, 200), FreeCAD.Vector(0, -58.5, 240),
        FreeCAD.Vector(0, -150, 240), FreeCAD.Vector(0, -150, 120), FreeCAD.Vector(0, -90, 120)
    ])'''
new_poly_tl = '''    poly_tl = Part.makePolygon([
        FreeCAD.Vector(0, -110, 120), FreeCAD.Vector(0, -80, 200), FreeCAD.Vector(0, -58.5, 240),
        FreeCAD.Vector(0, -150, 240), FreeCAD.Vector(0, -150, 120), FreeCAD.Vector(0, -110, 120)
    ])'''
content = content.replace(old_poly_tl, new_poly_tl)

# 3. Change apply_mount_holes to a 6-hole grid (3 columns x 2 rows)
old_apply_holes = '''    def apply_mount_holes(obj, spacing, depth, z_offset, target_z, reverse_dir=False):
        nonlocal top_half, bottom_half
        if not obj: return
        half = spacing / 2.0
        z_vec = obj.Placement.Rotation.multVec(vZ)
        for dx, dy in [(-half, -half), (-half, half), (half, -half), (half, half)]:'''
new_apply_holes = '''    def apply_mount_holes(obj, spacing, depth, z_offset, target_z, reverse_dir=False):
        nonlocal top_half, bottom_half
        if not obj: return
        half = spacing / 2.0
        z_vec = obj.Placement.Rotation.multVec(vZ)
        # 6-hole grid (3x2)
        for dx, dy in [(-half, -half), (0, -half), (half, -half), (-half, half), (0, half), (half, half)]:'''
content = content.replace(old_apply_holes, new_apply_holes)

with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Applied all 3 user requests!")
