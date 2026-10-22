with open('build_external_bed.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('pts = [FreeCAD.Vector(0, 0, z_min), FreeCAD.Vector(12.0, 0, z_min)]', 
                          'pts = [FreeCAD.Vector(0, 0, z_min), FreeCAD.Vector(p_radius, 0, z_min)]')

with open('build_external_bed.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Updated extended bed depth to use the full pocket radius!")
