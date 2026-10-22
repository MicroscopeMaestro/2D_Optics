with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_func = '''def make_aspheric_cradle(obj, pocket_radius=ASPHERIC_POCKET_RADIUS, clearance=0.25):
    \"\"\"
    Construct a specialized mounting bed (cradle) specifically conforming to the 16mm
    aspheric lens (Thorlabs ACL25416U, f=16mm, nominal OD=25.5mm).
    The bed closely hugs the lens barrel (pocket radius 13.05mm providing 0.60mm
    diametric clearance) and the convex aspheric surface with a controlled clearance
    (default 0.25mm), securely capturing the lens against translation and tipping
    while strictly preserving the 20mm clear aperture (minimum radius 10.0mm).
    \"\"\"
    local_s = obj.Shape.copy().transformShape(obj.Placement.Matrix.inverse())
    bb = local_s.BoundBox
    z_min = bb.ZMin - 0.1
    z_max = bb.ZMax + 1.0
    pts = [FreeCAD.Vector(0, 0, z_min)]
    z_curr = bb.ZMin
    while z_curr <= bb.ZMax:
        sl = local_s.slice(FreeCAD.Vector(0, 0, 1), min(z_curr + 0.05, bb.ZMax - 0.01))
        r = sl[0].BoundBox.XMax if sl else 0.0
        if r > 12.0:
            r_cleared = pocket_radius
        else:
            r_cleared = min(max(r + clearance, 10.0), pocket_radius)
        pts.append(FreeCAD.Vector(r_cleared, 0, z_curr))
        z_curr += 0.3
    pts.append(FreeCAD.Vector(pocket_radius, 0, z_max))
    pts.append(FreeCAD.Vector(0, 0, z_max))
    pts.append(FreeCAD.Vector(0, 0, z_min))
    wire = Part.makePolygon(pts)
    face = Part.Face(wire)
    revolved = face.revolve(FreeCAD.Vector(0, 0, 0), FreeCAD.Vector(0, 0, 1), 360)
    revolved.transformShape(obj.Placement.Matrix)
    return revolved'''

new_func = '''def make_aspheric_cradle(obj, pocket_radius=ASPHERIC_POCKET_RADIUS, clearance=0.25):
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
    revolved.transformShape(obj.Placement.Matrix)
    return revolved'''

content = content.replace(old_func, new_func)

with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Updated aspheric cradle to a continuous cone!")
