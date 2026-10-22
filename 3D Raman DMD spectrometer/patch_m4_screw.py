with open('build_external_bed.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Add the M4 screw hole logic right before the export
old_export = 'standalone_bed.exportStep("external_aspheric_bed.step")'

new_export = '''
# Add M4 set screw hole from the top down to the lens barrel
# The lens edge is 3.0mm thick, starting at ZMax and going to ZMax - 3.0.
# We place the screw exactly in the middle of the barrel at ZMax - 1.5.
m4_radius = 1.65  # Tap drill size for M4
screw_depth = 8.0 # From Y=14 down to Y=6, which breaks perfectly into the 13.3mm radius pocket
m4_hole = Part.makeCylinder(m4_radius, screw_depth, FreeCAD.Vector(0, height/2, bb.ZMax - 1.5), FreeCAD.Vector(0, -1, 0))

standalone_bed = standalone_bed.cut(m4_hole)

standalone_bed.exportStep("external_aspheric_bed.step")
'''

content = content.replace(old_export, new_export)

with open('build_external_bed.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Added M4 set screw hole to external bed!")
