import re

with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

step4_code = '''    # 4. Optical Subtractions (Zero-Intersection Proof & Extra 0.5mm Thickness for 30/100mm)
    if verbose: print("[STEP 4/7] Optical subtractions...")
    shapes_to_cut, first_placement, last_airgap_axis, last_airgap_end = [], None, None, None
    for obj in spectrometer.Group:
        if not hasattr(obj, 'Shape') or obj.Shape.isNull(): continue
        low = obj.Label.lower()
        if "glass" in low:
            # Use stable parametric primitives to construct a conforming conical bed
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
            if "01_collect" in low: first_placement = obj.Placement
        elif "airgap" in low:
            cyl, _, end_pt, axis = build_airgap_cylinder(obj.Shape, radius_reduction=airgap_radius_reduction, extension=1.0)
            shapes_to_cut.append((obj.Label, cyl if cyl else obj.Shape))
            if "18_condenser" in low: last_airgap_axis, last_airgap_end = axis, end_pt
        elif "hardware" in low:
            if "19_input_slit" in low:
                shapes_to_cut.append((obj.Label, Part.makeBox(SLIT_SLOT_WIDTH, SLIT_SLOT_HEIGHT, SLIT_SLOT_THICKNESS, FreeCAD.Vector(-SLIT_SLOT_WIDTH/2, -SLIT_SLOT_HEIGHT/2, obj.Placement.Base.z - SLIT_SLOT_THICKNESS/2))))
            elif "20_grating" in low:
                gr_box = Part.makeBox(26.0, 26.0, 7.0, FreeCAD.Vector(-13.0, -13.0, -3.5))
                gr_box.translate(FreeCAD.Vector(0, 5.0, 0))
                gr_box.transformShape(obj.Placement.Matrix)
                shapes_to_cut.append((obj.Label, gr_box.fuse(obj.Shape)))
            elif "22_cb_after_dmd" in low:
                shapes_to_cut.append((obj.Label, obj.Shape))
            elif "23_sensor" in low:
                shapes_to_cut.append((obj.Label, obj.Shape))
                
    success_cuts = 0
    for label, shape in shapes_to_cut:
        try:
            top_half = top_half.cut(shape)
            bottom_half = bottom_half.cut(shape)
            success_cuts += 1
        except Exception as e:
            if verbose: print(f"  [ERROR] Subtraction {label} failed: {e}")
            
    if first_placement:
        front_drill = Part.makeCylinder(DRILL_APERTURE_RADIUS, DRILL_APERTURE_DEPTH, FreeCAD.Vector(0, 0, -DRILL_APERTURE_DEPTH), FreeCAD.Vector(0, 0, 1))
        front_drill.transformShape(first_placement.Matrix)
        top_half, bottom_half = top_half.cut(front_drill), bottom_half.cut(front_drill)
        
    if last_airgap_axis and last_airgap_end:
        sensor_tunnel = Part.makeCylinder(SENSOR_TUNNEL_RADIUS, 150.0, last_airgap_end, last_airgap_axis)
        top_half, bottom_half = top_half.cut(sensor_tunnel), bottom_half.cut(sensor_tunnel)
        
    if verbose: print(f"  Applied {success_cuts}/{len(shapes_to_cut)} subtractions.")

'''

pattern_insert = r'(\s*# 5\. Fasteners)'
content = re.sub(pattern_insert, '\n' + step4_code + r'\1', content)

with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Inserted STEP 4 successfully!")
