with open('build_external_bed.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Y-axis (width) from 28.0 to 32.0
content = content.replace('height = 28.0', 'height = 32.0')
content = content.replace('FreeCAD.Vector(-14.0, -14.0, z_min)', 'FreeCAD.Vector(-14.0, -16.0, z_min)')

with open('build_external_bed.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Updated external bed script for 32.0mm Y-width!")
