import re

with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_interlock = '''        for f in x0_faces:
            try:
                t_outer = f.makeOffset2D(-0.8)
                t_inner = f.makeOffset2D(-2.4)
                if t_outer.Area > 1e-4:
                    tongue = t_outer.cut(t_inner)
                    if tongue_shape is None: tongue_shape = tongue
                    else: tongue_shape = tongue_shape.fuse(tongue)

                g_outer = f.makeOffset2D(-0.6)
                g_inner = f.makeOffset2D(-2.6)
                if g_outer.Area > 1e-4:
                    groove = g_outer.cut(g_inner)
                    if groove_shape is None: groove_shape = groove
                    else: groove_shape = groove_shape.fuse(groove)
            except Exception:'''

new_interlock = '''        for f in x0_faces:
            try:
                # Use ONLY the outer perimeter to prevent locking grooves from being placed near the optical paths!
                outer_face = Part.Face(f.OuterWire)
                
                # Move the tongue further inwards from the outer edge (-1.6mm) so it's not too close to the surface
                t_outer = outer_face.makeOffset2D(-1.6)
                t_inner = outer_face.makeOffset2D(-3.2)
                if t_outer.Area > 1e-4:
                    tongue = t_outer.cut(t_inner)
                    if tongue_shape is None: tongue_shape = tongue
                    else: tongue_shape = tongue_shape.fuse(tongue)

                # Groove is 0.2mm wider on both sides for 3D printing tolerance
                g_outer = outer_face.makeOffset2D(-1.4)
                g_inner = outer_face.makeOffset2D(-3.4)
                if g_outer.Area > 1e-4:
                    groove = g_outer.cut(g_inner)
                    if groove_shape is None: groove_shape = groove
                    else: groove_shape = groove_shape.fuse(groove)
            except Exception:'''

if old_interlock in content:
    content = content.replace(old_interlock, new_interlock)
    with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("Updated interlock spacing")
else:
    print("Could not find old interlock block")
