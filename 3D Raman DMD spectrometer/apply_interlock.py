with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_step_3_5 = '''    # 3.5 Interlocking Perimeter
    if ENABLE_PERIMETER_INTERLOCK:
        if verbose: print("[STEP 3.5/7] Constructing perimeter interlock...")
        x0_faces = [f for f in top_half.Faces if abs(f.BoundBox.XMin) < 1e-4 and abs(f.BoundBox.XMax) < 1e-4]
        if x0_faces:
            x0_face = x0_faces[0]
            for f in x0_faces[1:]: x0_face = x0_face.fuse(f)
            outer_wire = sorted(x0_face.Wires, key=lambda w: w.Length, reverse=True)[0]
            try:
                tongue_face = Part.Face(outer_wire.makeOffset2D(-0.8).Wires[0]).cut(Part.Face(outer_wire.makeOffset2D(-2.4).Wires[0]))
                bottom_half = bottom_half.fuse(tongue_face.extrude(FreeCAD.Vector(3.0, 0, 0)))
                groove_face = Part.Face(outer_wire.makeOffset2D(-0.6).Wires[0]).cut(Part.Face(outer_wire.makeOffset2D(-2.6).Wires[0]))
                top_half = top_half.cut(groove_face.extrude(FreeCAD.Vector(3.2, 0, 0)))
            except Exception as e:
                if verbose: print(f"  [WARNING] Perimeter interlock failed: {e}")'''

if old_step_3_5 in content:
    content = content.replace(old_step_3_5, '')
    print("Removed Step 3.5")
else:
    print("Could not find Step 3.5!")

new_step_4_5 = '''    # 4.5 Interlocking Perimeter (Extended around optics)
    if ENABLE_PERIMETER_INTERLOCK:
        if verbose: print("[STEP 4.5/7] Constructing extended perimeter interlock...")
        x0_faces = [f for f in top_half.Faces if abs(f.BoundBox.XMin) < 1e-4 and abs(f.BoundBox.XMax) < 1e-4]
        
        tongue_shape = None
        groove_shape = None
        for f in x0_faces:
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
            except Exception:
                pass
                
        if tongue_shape:
            bottom_half = bottom_half.fuse(tongue_shape.extrude(FreeCAD.Vector(3.0, 0, 0)))
        if groove_shape:
            top_half = top_half.cut(groove_shape.extrude(FreeCAD.Vector(3.2, 0, 0)))'''

old_step_5 = '''    # 5. Fasteners & Pegs'''
content = content.replace(old_step_5, new_step_4_5 + '\n\n' + old_step_5)

with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
    f.write(content)
