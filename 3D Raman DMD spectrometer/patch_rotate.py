with open('build_external_bed.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_export = 'standalone_bed.exportStep("external_aspheric_bed.step")'
new_export = '''
# Rotate 90 degrees around X so the part stands vertically!
# This aligns the long dimension and the M4 threaded hole with the Z-axis,
# which is vastly superior for 3D printing the physical threads without sagging.
standalone_bed.rotate(FreeCAD.Vector(0,0,0), FreeCAD.Vector(1,0,0), 90)

standalone_bed.exportStep("external_aspheric_bed.step")
'''

content = content.replace(old_export, new_export)

with open('build_external_bed.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Added 90-degree rotation to stand the mount vertically!")
