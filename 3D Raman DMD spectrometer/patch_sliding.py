with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Add SLIDING_TOLERANCE parameter
old_param = 'SLIT_IS_CIRCULAR_1_INCH     = True # True for 1-inch round disk'
new_param = 'SLIT_IS_CIRCULAR_1_INCH     = True # True for 1-inch round disk\nSLIDING_TOLERANCE           = 5.0  # Adds +/- 5mm slotting for fine axial tuning'
content = content.replace(old_param, new_param)

# 2. Update standard lenses
old_cyl = 'cyl = Part.makeCylinder(OPTICS_POCKET_RADIUS, length + extra_z, FreeCAD.Vector(0, 0, bb.ZMin - extra_z / 2 + axial_dz), FreeCAD.Vector(0, 0, 1))'
new_cyl = 'cyl = Part.makeCylinder(OPTICS_POCKET_RADIUS, length + extra_z + 2 * SLIDING_TOLERANCE, FreeCAD.Vector(0, 0, bb.ZMin - extra_z / 2 - SLIDING_TOLERANCE + axial_dz), FreeCAD.Vector(0, 0, 1))'
content = content.replace(old_cyl, new_cyl)

# 3. Update AirGaps
old_air = 'cyl, _, end_pt, axis = build_airgap_cylinder(obj.Shape, radius_reduction=airgap_radius_reduction, extension=1.0, max_t=max_t)'
new_air = 'cyl, _, end_pt, axis = build_airgap_cylinder(obj.Shape, radius_reduction=airgap_radius_reduction, extension=1.0 + SLIDING_TOLERANCE, max_t=max_t)'
content = content.replace(old_air, new_air)

# 4. Rewrite make_aspheric_cradle for sliding tolerance
old_func = '''def make_aspheric_cradle(obj, pocket_radius=ASPHERIC_POCKET_RADIUS, clearance=0.25, axial_dz=0.0):
    \"\"\"
    Construct a conical mounting bed for the aspheric lens.
    Starts at pocket_radius and tapers continuously to 10.0mm clear aperture.
    \"\"\"
    local_s = obj.Shape.copy().transformShape(obj.Placement.Matrix.inverse())
    bb = local_s.BoundBox
    z_min = bb.ZMin - 0.1
    z_max = bb.ZMax + 1.0
    
    # 1. Find the end of the straight barrel
    z_curr = bb.ZMin
    z_barrel = bb.ZMin
    while z_curr <= bb.ZMax:
        sl = local_s.slice(FreeCAD.Vector(0, 0, 1), min(z_curr + 0.05, bb.ZMax - 0.01))
        r = sl[0].BoundBox.XMax if sl else 0.0
        if r > 12.0:
            z_barrel = z_curr
        z_curr += 0.3
        
    # 2. Build the continuous cone profile
    pts = [FreeCAD.Vector(0, 0, z_min)]
    z_curr = bb.ZMin
    while z_curr <= bb.ZMax:
        if z_curr <= z_barrel:
            r_cleared = pocket_radius
        else:
            fraction = (z_curr - z_barrel) / (bb.ZMax - z_barrel) if bb.ZMax > z_barrel else 1.0
            r_cleared = pocket_radius - fraction * (pocket_radius - 10.0)
        pts.append(FreeCAD.Vector(r_cleared, 0, z_curr))
        z_curr += 0.3
        
    pts.append(FreeCAD.Vector(10.0, 0, z_max))
    pts.append(FreeCAD.Vector(0, 0, z_max))
    pts.append(FreeCAD.Vector(0, 0, z_min))
    
    wire = Part.makePolygon(pts)
    face = Part.Face(wire)
    revolved = face.revolve(FreeCAD.Vector(0, 0, 0), FreeCAD.Vector(0, 0, 1), 360)
    if axial_dz != 0.0:
        revolved.translate(FreeCAD.Vector(0, 0, axial_dz))
    revolved.transformShape(obj.Placement.Matrix)
    return revolved'''

new_func = '''def make_aspheric_cradle(obj, pocket_radius=ASPHERIC_POCKET_RADIUS, sliding_tolerance=0.0, axial_dz=0.0):
    \"\"\"
    Construct a slotted conical mounting bed for the aspheric lens.
    Allows sliding by +/- sliding_tolerance while maintaining perfect conical seating.
    \"\"\"
    local_s = obj.Shape.copy().transformShape(obj.Placement.Matrix.inverse())
    bb = local_s.BoundBox
    
    # 1. Find the end of the straight barrel
    z_curr = bb.ZMin
    z_barrel = bb.ZMin
    while z_curr <= bb.ZMax:
        sl = local_s.slice(FreeCAD.Vector(0, 0, 1), min(z_curr + 0.05, bb.ZMax - 0.01))
        r = sl[0].BoundBox.XMax if sl else 0.0
        if r > 12.0:
            z_barrel = z_curr
        z_curr += 0.3
        
    # 2. Build the continuous cone profile shifted for sliding tolerance
    z_min = bb.ZMin - 0.1 - sliding_tolerance
    z_max = bb.ZMax + 1.0 + sliding_tolerance
    
    actual_z_barrel = z_barrel + sliding_tolerance
    actual_z_apex = bb.ZMax + sliding_tolerance
    
    pts = [
        FreeCAD.Vector(0, 0, z_min),
        FreeCAD.Vector(pocket_radius, 0, z_min),
        FreeCAD.Vector(pocket_radius, 0, actual_z_barrel),
        FreeCAD.Vector(10.0, 0, actual_z_apex),
        FreeCAD.Vector(10.0, 0, z_max),
        FreeCAD.Vector(0, 0, z_max),
        FreeCAD.Vector(0, 0, z_min)
    ]
    
    wire = Part.makePolygon(pts)
    face = Part.Face(wire)
    revolved = face.revolve(FreeCAD.Vector(0, 0, 0), FreeCAD.Vector(0, 0, 1), 360)
    if axial_dz != 0.0:
        revolved.translate(FreeCAD.Vector(0, 0, axial_dz))
    revolved.transformShape(obj.Placement.Matrix)
    return revolved'''

content = content.replace(old_func, new_func)

# 5. Update call to make_aspheric_cradle
old_call = 'final_cut = make_aspheric_cradle(obj, pocket_radius=ASPHERIC_POCKET_RADIUS, clearance=0.50, axial_dz=axial_dz)'
new_call = 'final_cut = make_aspheric_cradle(obj, pocket_radius=ASPHERIC_POCKET_RADIUS, sliding_tolerance=SLIDING_TOLERANCE, axial_dz=axial_dz)'
content = content.replace(old_call, new_call)

with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Updated CAD script to include +/- 5mm sliding pockets!")
