with open('build_external_bed.py', 'r', encoding='utf-8') as f:
    content = f.read()

# The current cutout radius is 10.0mm. We will increase it to 12.0mm.
content = content.replace('pts = [FreeCAD.Vector(0, 0, z_min), FreeCAD.Vector(10.0, 0, z_min)]', 
                          'pts = [FreeCAD.Vector(0, 0, z_min), FreeCAD.Vector(12.0, 0, z_min)]')

with open('build_external_bed.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Increased internal cutout radius of the extended bed to 12.0mm (24.0mm diameter)!")
