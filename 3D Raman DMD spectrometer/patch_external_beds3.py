with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Add USE_EXTERNAL_ASPHERIC_MOUNTS parameter
if 'USE_EXTERNAL_ASPHERIC_MOUNTS' not in content:
    old_param = 'SLIDING_TOLERANCE           = 5.0'
    new_param = 'SLIDING_TOLERANCE           = 5.0\nUSE_EXTERNAL_ASPHERIC_MOUNTS= True # True to cut 40.4x40.4mm square drop-in pockets for external beds'
    content = content.replace(old_param, new_param)

# Modify make_aspheric_cradle to return a box if USE_EXTERNAL_ASPHERIC_MOUNTS is True
# I'll just find the start of the function and the end of the docstring
import re
func_pattern = r'def make_aspheric_cradle\([^)]+\):.*?\"\"\"'
match = re.search(func_pattern, content, flags=re.DOTALL)
if match and 'USE_EXTERNAL_ASPHERIC_MOUNTS' not in match.group(0):
    insertion_point = match.end()
    
    new_logic = '''
    local_s = obj.Shape.copy().transformShape(obj.Placement.Matrix.inverse())
    bb = local_s.BoundBox
    
    if USE_EXTERNAL_ASPHERIC_MOUNTS:
        z_min = bb.ZMin - 0.1
        z_max = bb.ZMax + 5.0
        p_z_min = z_min - sliding_tolerance - 0.2
        p_z_max = z_max + sliding_tolerance + 0.2
        p_thickness = p_z_max - p_z_min
        p_width = 40.4
        p_height = 40.4
        
        box = Part.makeBox(p_width, p_height, p_thickness, FreeCAD.Vector(-p_width/2, -p_height/2, p_z_min))
        if axial_dz != 0.0:
            box.translate(FreeCAD.Vector(0, 0, axial_dz))
        box.transformShape(obj.Placement.Matrix)
        return box
'''
    # We replace everything from insertion point down to the 'local_s' line
    # Actually, simpler: replace the whole function
    old_func = '''def make_aspheric_cradle(obj, pocket_radius=ASPHERIC_POCKET_RADIUS, sliding_tolerance=0.0, axial_dz=0.0, clearance=0.25):
    """
    Construct a specialized mounting bed (cradle) specifically conforming to the 16mm
    aspheric lens (Thorlabs ACL25416U, f=16mm, nominal OD=25.5mm).
    The bed closely hugs the lens barrel (pocket radius 13.05mm providing 0.60mm
    diametric clearance) and the convex aspheric surface with a controlled clearance
    (default 0.25mm), securely capturing the lens against translation and tipping
    while strictly preserving the 18mm clear aperture (minimum radius 9.0mm).
    """
    local_s = obj.Shape.copy().transformShape(obj.Placement.Matrix.inverse())
    bb = local_s.BoundBox
    z_min = bb.ZMin - 0.1 - sliding_tolerance
    z_max = bb.ZMax + 1.0 + sliding_tolerance
    
    pts = [FreeCAD.Vector(0, 0, z_min), FreeCAD.Vector(9.0, 0, z_min)]
    z_curr = bb.ZMin
    while z_curr <= bb.ZMax:
        sl = local_s.slice(FreeCAD.Vector(0, 0, 1), min(z_curr + 0.05, bb.ZMax - 0.01))
        r = sl[0].BoundBox.XMax if sl else 0.0
        r_cleared = pocket_radius if r > 12.0 else max(r + clearance, 9.0)
        pts.append(FreeCAD.Vector(r_cleared, 0, z_curr))
        z_curr += 0.2
    pts.append(FreeCAD.Vector(pocket_radius, 0, bb.ZMax))
    pts.append(FreeCAD.Vector(pocket_radius, 0, z_max))
    pts.append(FreeCAD.Vector(0, 0, z_max))
    pts.append(FreeCAD.Vector(0, 0, z_min))
    
    wire = Part.makePolygon(pts)
    face = Part.Face(wire)
    revolved = face.revolve(FreeCAD.Vector(0, 0, 0), FreeCAD.Vector(0, 0, 1), 360)
    if axial_dz != 0.0:
        revolved.translate(FreeCAD.Vector(0, 0, axial_dz))
    revolved.transformShape(obj.Placement.Matrix)
    return revolved'''

    new_func = '''def make_aspheric_cradle(obj, pocket_radius=ASPHERIC_POCKET_RADIUS, sliding_tolerance=0.0, axial_dz=0.0, clearance=0.25):
    """
    Construct a specialized mounting bed (cradle) or drop-in pockets for external mounts.
    """
    local_s = obj.Shape.copy().transformShape(obj.Placement.Matrix.inverse())
    bb = local_s.BoundBox
    
    if USE_EXTERNAL_ASPHERIC_MOUNTS:
        z_min = bb.ZMin - 0.1
        z_max = bb.ZMax + 5.0
        p_z_min = z_min - sliding_tolerance - 0.2
        p_z_max = z_max + sliding_tolerance + 0.2
        p_thickness = p_z_max - p_z_min
        p_width = 40.4
        p_height = 40.4
        
        box = Part.makeBox(p_width, p_height, p_thickness, FreeCAD.Vector(-p_width/2, -p_height/2, p_z_min))
        if axial_dz != 0.0:
            box.translate(FreeCAD.Vector(0, 0, axial_dz))
        box.transformShape(obj.Placement.Matrix)
        return box

    z_min = bb.ZMin - 0.1 - sliding_tolerance
    z_max = bb.ZMax + 1.0 + sliding_tolerance
    
    pts = [FreeCAD.Vector(0, 0, z_min), FreeCAD.Vector(9.0, 0, z_min)]
    z_curr = bb.ZMin
    while z_curr <= bb.ZMax:
        sl = local_s.slice(FreeCAD.Vector(0, 0, 1), min(z_curr + 0.05, bb.ZMax - 0.01))
        r = sl[0].BoundBox.XMax if sl else 0.0
        r_cleared = pocket_radius if r > 12.0 else max(r + clearance, 9.0)
        pts.append(FreeCAD.Vector(r_cleared, 0, z_curr))
        z_curr += 0.2
    pts.append(FreeCAD.Vector(pocket_radius, 0, bb.ZMax))
    pts.append(FreeCAD.Vector(pocket_radius, 0, z_max))
    pts.append(FreeCAD.Vector(0, 0, z_max))
    pts.append(FreeCAD.Vector(0, 0, z_min))
    
    wire = Part.makePolygon(pts)
    face = Part.Face(wire)
    revolved = face.revolve(FreeCAD.Vector(0, 0, 0), FreeCAD.Vector(0, 0, 1), 360)
    if axial_dz != 0.0:
        revolved.translate(FreeCAD.Vector(0, 0, axial_dz))
    revolved.transformShape(obj.Placement.Matrix)
    return revolved'''

    content = content.replace(old_func, new_func)

with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Re-applied USE_EXTERNAL_ASPHERIC_MOUNTS safely!")
