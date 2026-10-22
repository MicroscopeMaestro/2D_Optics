with open('build_external_bed.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Make clear aperture smaller (from 10.0 radius to 9.0 radius)
content = content.replace('FreeCAD.Vector(10.0, 0, z_min)', 'FreeCAD.Vector(9.0, 0, z_min)')
content = content.replace('max(r + clearance, 10.0)', 'max(r + clearance, 9.0)')

# Tighten the conic slope clearance
content = content.replace('clearance = 0.5  # Generous curve tolerance', 'clearance = 0.15  # Tighter conic slope')

with open('build_external_bed.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Updated clear aperture and conic slope clearance!")
