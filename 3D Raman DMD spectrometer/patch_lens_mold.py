import re

with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_logic = '''        if "glass" in low:
            extra_z = lens_extra_thickness_30_100 if any(k in low for k in ['030', '100']) else 0.0
            local_s = obj.Shape.copy().transformShape(obj.Placement.Matrix.inverse())
            bb = local_s.BoundBox
            length = (bb.ZMax - bb.ZMin) + extra_z
            cyl = Part.makeCylinder(OPTICS_POCKET_RADIUS, length, FreeCAD.Vector(0, 0, bb.ZMin), FreeCAD.Vector(0, 0, 1))
            cyl.transformShape(obj.Placement.Matrix)
            shapes_to_cut.append((obj.Label, cyl))
            if "01_collect" in low: first_placement = obj.Placement'''

new_logic = '''        if "glass" in low:
            # Preserve the true conic B-spline surface to make a perfect molded bed
            local_s = obj.Shape.copy()
            local_s.transformShape(obj.Placement.Matrix.inverse())
            
            # Uniform 1.01x scale for exact conforming clearance
            mat = FreeCAD.Matrix()
            mat.scale(1.01, 1.01, 1.01)
            scaled_s = local_s.copy()
            scaled_s.transformShape(mat)
            scaled_s.transformShape(obj.Placement.Matrix)
            
            # Apply +0.5mm Z pocket thickness if requested by sweeping along optical axis
            extra_z = lens_extra_thickness_30_100 if any(k in low for k in ['030', '100']) else 0.0
            if extra_z > 0:
                z_vec = obj.Placement.Rotation.multVec(FreeCAD.Vector(0,0,1))
                shifted = scaled_s.copy()
                shifted.translate(z_vec * extra_z)
                final_cut = scaled_s.fuse(shifted)
            else:
                final_cut = scaled_s
                
            shapes_to_cut.append((obj.Label, final_cut))
            if "01_collect" in low: first_placement = obj.Placement'''

if old_logic in content:
    content = content.replace(old_logic, new_logic)
    print("Replaced lens molding successfully!")
else:
    print("Failed to replace lens molding!")

with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
    f.write(content)
