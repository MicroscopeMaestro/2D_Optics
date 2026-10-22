with open('build_external_bed.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Box dimensions: Width (X) becomes 36. Height (Y) becomes 28.
content = content.replace('width = 28.0', 'width = 36.0')
content = content.replace('height = 36.0', 'height = 28.0')

# Box position: X from -14 to +22, Y from -14 to +14
content = content.replace('box = Part.makeBox(width, height, thickness, FreeCAD.Vector(-width/2, -14.0, z_min))',
                          'box = Part.makeBox(width, height, thickness, FreeCAD.Vector(-14.0, -14.0, z_min))')

# 2. Update M4 screw logic (it now drills along -X instead of -Y)
# Counterbore at X=22, going down along -X
content = content.replace('cb = Part.makeCylinder(cb_radius, cb_depth, FreeCAD.Vector(0, 22.0, bb.ZMax - 1.5), FreeCAD.Vector(0, -1, 0))',
                          'cb = Part.makeCylinder(cb_radius, cb_depth, FreeCAD.Vector(22.0, 0, bb.ZMax - 1.5), FreeCAD.Vector(-1, 0, 0))')

# M4 core hole at X = 22 - cb_depth
content = content.replace('m4_hole = Part.makeCylinder(th_radius, th_depth + 2.0, FreeCAD.Vector(0, 22.0 - cb_depth, bb.ZMax - 1.5), FreeCAD.Vector(0, -1, 0))',
                          'm4_hole = Part.makeCylinder(th_radius, th_depth + 2.0, FreeCAD.Vector(22.0 - cb_depth, 0, bb.ZMax - 1.5), FreeCAD.Vector(-1, 0, 0))')

# Rotate the physical thread to align with X-axis instead of Y-axis
# Z-axis (0,0,1) rotated by 90 around Y (0,1,0) becomes X-axis. 
# Wait, rotating Z by +90 around Y: X becomes Z, Z becomes -X.
# So we want it to point along -X. Rot around Y by +90 points to X. Wait: (0,0,1) -> (sin(90), 0, cos(90)) = (1, 0, 0).
# If we rotate by -90 around Y: (0,0,1) -> (-1, 0, 0).
content = content.replace('th.rotate(FreeCAD.Vector(0,0,0), FreeCAD.Vector(1,0,0), 90)',
                          'th.rotate(FreeCAD.Vector(0,0,0), FreeCAD.Vector(0,1,0), -90)')

content = content.replace('th.translate(FreeCAD.Vector(0, 22.0 - cb_depth, bb.ZMax - 1.5))',
                          'th.translate(FreeCAD.Vector(22.0 - cb_depth, 0, bb.ZMax - 1.5))')

# Remove the 90 degree rotation at the end! The user wants it vertical in the spectrometer (which it now is, along X)
content = content.replace('standalone_bed.rotate(FreeCAD.Vector(0,0,0), FreeCAD.Vector(1,0,0), 90)\n', '')

with open('build_external_bed.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Updated external bed to stand vertically along the X axis!")
