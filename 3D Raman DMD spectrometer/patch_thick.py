import re

with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_interlock = '''                # Move the tongue further inwards from the outer edge (-1.6mm) so it's not too close to the surface
                t_outer = outer_face.makeOffset2D(-1.6)
                t_inner = outer_face.makeOffset2D(-3.2)
                if t_outer.Area > 1e-4:
                    tongue = t_outer.cut(t_inner)
                    if tongue_shape is None: tongue_shape = tongue
                    else: tongue_shape = tongue_shape.fuse(tongue)

                # Groove is 0.2mm wider on both sides for 3D printing tolerance
                g_outer = outer_face.makeOffset2D(-1.4)
                g_inner = outer_face.makeOffset2D(-3.4)'''

new_interlock = '''                # Use a MUCH THICKER tongue (2.4mm instead of 1.6mm) to prevent 3D printing snapping issues!
                t_outer = outer_face.makeOffset2D(-1.2)
                t_inner = outer_face.makeOffset2D(-3.6)
                if t_outer.Area > 1e-4:
                    tongue = t_outer.cut(t_inner)
                    if tongue_shape is None: tongue_shape = tongue
                    else: tongue_shape = tongue_shape.fuse(tongue)

                # Groove is 0.2mm wider on both sides for 3D printing tolerance
                g_outer = outer_face.makeOffset2D(-1.0)
                g_inner = outer_face.makeOffset2D(-3.8)'''

content = content.replace(old_interlock, new_interlock)

with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Updated interlock to 2.4mm width")
