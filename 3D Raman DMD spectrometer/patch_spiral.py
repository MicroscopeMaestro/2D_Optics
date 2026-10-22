with open('build_external_bed.py', 'r', encoding='utf-8') as f:
    content = f.read()

import re

# Update Box dimensions
content = content.replace('width = 28.0', 'width = 28.0')
content = content.replace('height = 28.0', 'height = 36.0')
content = content.replace('box = Part.makeBox(width, height, thickness, FreeCAD.Vector(-width/2, -height/2, z_min))', 
                          'box = Part.makeBox(width, height, thickness, FreeCAD.Vector(-width/2, -14.0, z_min))')

# Update the M4 hole logic
old_m4_logic = '''m4_radius = 1.65  # Tap drill size for M4
screw_depth = 8.0 # From Y=14 down to Y=6, which breaks perfectly into the 13.3mm radius pocket
m4_hole = Part.makeCylinder(m4_radius, screw_depth, FreeCAD.Vector(0, height/2, bb.ZMax - 1.5), FreeCAD.Vector(0, -1, 0))

standalone_bed = standalone_bed.cut(m4_hole)'''

new_m4_logic = '''
# Create the Counterbore (radius 3.6 for M4 head, depth 4.5)
cb_depth = 4.5
cb_radius = 3.6
cb = Part.makeCylinder(cb_radius, cb_depth, FreeCAD.Vector(0, 22.0, bb.ZMax - 1.5), FreeCAD.Vector(0, -1, 0))
standalone_bed = standalone_bed.cut(cb)

# Create the Physical Threads
th_depth = 5.0
th_radius = 1.65
th_pitch = 0.7
th_thread_depth = 0.35

# Central hole
m4_hole = Part.makeCylinder(th_radius, th_depth + 2.0, FreeCAD.Vector(0, 22.0 - cb_depth, bb.ZMax - 1.5), FreeCAD.Vector(0, -1, 0))

# Spiral threads
try:
    th = Part.makeThread(th_pitch, th_thread_depth, th_depth, th_radius)
    # Rotate Z axis to -Y axis
    th.rotate(FreeCAD.Vector(0,0,0), FreeCAD.Vector(1,0,0), 90)
    # Translate to start of thread
    th.translate(FreeCAD.Vector(0, 22.0 - cb_depth, bb.ZMax - 1.5))
    m4_hole = m4_hole.fuse(th)
except Exception as e:
    print("Thread error:", e)

standalone_bed = standalone_bed.cut(m4_hole)
'''

content = content.replace(old_m4_logic, new_m4_logic)

with open('build_external_bed.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Updated external bed to be 36mm tall, with counterbore and physical M4 threads!")
