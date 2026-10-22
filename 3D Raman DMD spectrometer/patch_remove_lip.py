with open('build_external_bed.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Remove the snap-fit lip so it's a straight 13.3mm radius cylinder opening (26.6mm diameter)
old_lip = '''pts.append(FreeCAD.Vector(pocket_radius, 0, bb.ZMax))
# Create a small retaining "snap-fit" lip at the front opening
# The lens radius is 12.7mm. We ramp down to 12.4mm to create a 0.3mm overlap lock.
pts.append(FreeCAD.Vector(12.55, 0, bb.ZMax + 0.3)) 
pts.append(FreeCAD.Vector(12.55, 0, z_max))'''

new_lip = '''pts.append(FreeCAD.Vector(pocket_radius, 0, bb.ZMax))
pts.append(FreeCAD.Vector(pocket_radius, 0, z_max))'''

content = content.replace(old_lip, new_lip)

with open('build_external_bed.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Removed front snap-fit lip!")
