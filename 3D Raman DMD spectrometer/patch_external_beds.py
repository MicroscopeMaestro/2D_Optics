with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Add USE_EXTERNAL_ASPHERIC_MOUNTS parameter
old_param = 'SLIDING_TOLERANCE           = 5.0  # Adds +/- 5mm slotting for fine axial tuning'
new_param = 'SLIDING_TOLERANCE           = 5.0  # Adds +/- 5mm slotting for fine axial tuning\nUSE_EXTERNAL_ASPHERIC_MOUNTS= True # True to cut 40.4x40.4mm square drop-in pockets for external beds'
content = content.replace(old_param, new_param)

# 2. Modify make_aspheric_cradle to return a box if USE_EXTERNAL_ASPHERIC_MOUNTS is True
old_func_start = '''def make_aspheric_cradle(obj, pocket_radius=ASPHERIC_POCKET_RADIUS, sliding_tolerance=0.0, axial_dz=0.0):
    \"\"\"
    Construct a slotted conical mounting bed for the aspheric lens.
    Allows sliding by +/- sliding_tolerance while maintaining perfect conical seating.
    \"\"\"
    local_s = obj.Shape.copy().transformShape(obj.Placement.Matrix.inverse())
    bb = local_s.BoundBox'''

new_func_start = '''def make_aspheric_cradle(obj, pocket_radius=ASPHERIC_POCKET_RADIUS, sliding_tolerance=0.0, axial_dz=0.0):
    \"\"\"
    Construct a slotted conical mounting bed for the aspheric lens.
    Allows sliding by +/- sliding_tolerance while maintaining perfect conical seating.
    \"\"\"
    local_s = obj.Shape.copy().transformShape(obj.Placement.Matrix.inverse())
    bb = local_s.BoundBox
    
    if USE_EXTERNAL_ASPHERIC_MOUNTS:
        # Instead of a tight cone, create a 40.4 x 40.4mm drop-in pocket for the standalone external mount!
        # The external mount was exactly z_max - z_min thick (where z_min=bb.ZMin-0.1, z_max=bb.ZMax+5.0).
        z_min = bb.ZMin - 0.1
        z_max = bb.ZMax + 5.0
        
        # We also need to add sliding tolerance to the pocket so the 40x40 block can slide!
        p_z_min = z_min - sliding_tolerance - 0.2
        p_z_max = z_max + sliding_tolerance + 0.2
        p_thickness = p_z_max - p_z_min
        
        p_width = 40.4
        p_height = 40.4
        
        box = Part.makeBox(p_width, p_height, p_thickness, FreeCAD.Vector(-p_width/2, -p_height/2, p_z_min))
        if axial_dz != 0.0:
            box.translate(FreeCAD.Vector(0, 0, axial_dz))
        box.transformShape(obj.Placement.Matrix)
        return box'''

content = content.replace(old_func_start, new_func_start)

with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Updated build_spectrometer.py to cut 40x40 external mount pockets!")
