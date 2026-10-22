with open('build_external_bed.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Increase pocket radius and clearance
content = content.replace('pocket_radius = 13.05', 'pocket_radius = 13.3  # Generous radial tolerance for standard FDM')
content = content.replace('clearance = 0.25', 'clearance = 0.5  # Generous curve tolerance')

# Ease the snap-fit lip so it isn't too brutal
content = content.replace('pts.append(FreeCAD.Vector(12.4, 0, bb.ZMax + 0.3))', 'pts.append(FreeCAD.Vector(12.55, 0, bb.ZMax + 0.3))')
content = content.replace('pts.append(FreeCAD.Vector(12.4, 0, z_max))', 'pts.append(FreeCAD.Vector(12.55, 0, z_max))')

with open('build_external_bed.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Added extra spacing tolerance to the aspheric lens pocket!")
