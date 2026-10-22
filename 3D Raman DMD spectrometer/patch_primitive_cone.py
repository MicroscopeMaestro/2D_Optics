import re

with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

# I will replace the "glass" parsing logic entirely.
pattern = r'(if "glass" in low:).*?(shapes_to_cut\.append\(\(obj\.Label, final_cut\)\)\n\s+if "01_collect" in low: first_placement = obj\.Placement)'

new_logic = '''if "glass" in low:
            # Use stable parametric primitives to construct a conforming conical bed
            # This completely avoids B-Spline mesh shattering while perfectly hugging the aspheric lens
            local_s = obj.Shape.copy()
            local_s.transformShape(obj.Placement.Matrix.inverse())
            bb = local_s.BoundBox
            
            slice_min = local_s.slice(FreeCAD.Vector(0,0,1), bb.ZMin + 0.5)
            r_min = slice_min[0].BoundBox.XMax if slice_min and len(slice_min)>0 else OPTICS_POCKET_RADIUS
            
            slice_max = local_s.slice(FreeCAD.Vector(0,0,1), bb.ZMax - 0.5)
            r_max = slice_max[0].BoundBox.XMax if slice_max and len(slice_max)>0 else OPTICS_POCKET_RADIUS
            
            length = (bb.ZMax - bb.ZMin)
            extra_z = lens_extra_thickness_30_100 if any(k in low for k in ['030', '100']) else 0.0
            
            if r_min < 12.0:
                cone_len = length - 3.0
                if cone_len < 0: cone_len = length * 0.5
                cone = Part.makeCone(max(2.0, r_min), OPTICS_POCKET_RADIUS, cone_len, FreeCAD.Vector(0,0,bb.ZMin), FreeCAD.Vector(0,0,1))
                cyl = Part.makeCylinder(OPTICS_POCKET_RADIUS, length - cone_len + extra_z, FreeCAD.Vector(0,0,bb.ZMin + cone_len), FreeCAD.Vector(0,0,1))
                final_cut = cone.fuse(cyl)
            elif r_max < 12.0:
                cone_len = length - 3.0
                if cone_len < 0: cone_len = length * 0.5
                cyl = Part.makeCylinder(OPTICS_POCKET_RADIUS, length - cone_len + extra_z, FreeCAD.Vector(0,0,bb.ZMin), FreeCAD.Vector(0,0,1))
                cone = Part.makeCone(OPTICS_POCKET_RADIUS, max(2.0, r_max), cone_len, FreeCAD.Vector(0,0,bb.ZMin + length - cone_len + extra_z), FreeCAD.Vector(0,0,1))
                final_cut = cyl.fuse(cone)
            else:
                final_cut = Part.makeCylinder(OPTICS_POCKET_RADIUS, length + extra_z, FreeCAD.Vector(0,0,bb.ZMin), FreeCAD.Vector(0,0,1))
            
            final_cut.transformShape(obj.Placement.Matrix)
            shapes_to_cut.append((obj.Label, final_cut))
            if "01_collect" in low: first_placement = obj.Placement'''

content, count = re.subn(pattern, new_logic, content, flags=re.DOTALL)
print(f"Replaced {count} times!")

with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
    f.write(content)
