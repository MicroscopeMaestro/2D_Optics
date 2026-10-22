with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_chamfer_step = '''    # 2. Exterior Edge Smoothing (Chamfers)
    if enable_fillets:
        if verbose: print("[STEP 2/7] Applying massive 2mm chamfer to external surfaces...")'''

new_corner_step = '''    # 1.9 Vertical Corner Fillets
    if enable_fillets:
        if verbose: print("  -> Rounding sharp vertical corners...")
        def make_vertical_fillet(y_val, r_fillet, dir_y):
            # Cutter that spans X from -50 to +50
            # If dir_y == -1, it cuts the max Y boundary (e.g. Y=24)
            # If dir_y == 1, it cuts the min Y boundary (e.g. Y=-24)
            y_origin = y_val - r_fillet if dir_y == -1 else y_val
            box = Part.makeBox(100.0, r_fillet, 300.0, FreeCAD.Vector(-50.0, y_origin, -10.0))
            center_y = y_val - r_fillet if dir_y == -1 else y_val + r_fillet
            cyl = Part.makeCylinder(r_fillet, 300.0, FreeCAD.Vector(0, center_y, -10.0), FreeCAD.Vector(0, 0, 1))
            # Wait, cyl is just a cylinder at X=0, but we need it to span X.
            # Part.makeCylinder doesn't span X if direction is Z.
            # To cut a 2D profile across X, we sweep a face or make a big cylinder.
            pass

        # Easier way: create a cutter block that is a box with a rounded corner subtracted, then extrude in X
        # For Y=24 (Right side):
        r = 5.0
        # Right front corner
        box_rf = Part.makeBox(100, r, 300, FreeCAD.Vector(-50, 24 - r, -10))
        cyl_rf = Part.makeCylinder(r, 300, FreeCAD.Vector(0, 24 - r, -10), FreeCAD.Vector(0,0,1))
        # Wait, cyl_rf at X=0 will only have a circular profile in XY if it's pointing in Z.
        # But we need the profile to be constant across X!
        # So we create a 2D face in YZ and extrude it along X.
        
        # Let's just use the bounding corners: Y=24 and Y=-24 at Z=15.36 (Front)
        front_right_cut = Part.makeBox(100, r, r, FreeCAD.Vector(-50, 24 - r, 15.36))
        cyl_fr = Part.makeCylinder(r, 100, FreeCAD.Vector(0, 24 - r, 15.36 + r), FreeCAD.Vector(1,0,0))
        cyl_fr.translate(FreeCAD.Vector(-50, 0, 0))
        front_right_cut = front_right_cut.cut(cyl_fr)
        
        front_left_cut = Part.makeBox(100, r, r, FreeCAD.Vector(-50, -24, 15.36))
        cyl_fl = Part.makeCylinder(r, 100, FreeCAD.Vector(0, -24 + r, 15.36 + r), FreeCAD.Vector(1,0,0))
        cyl_fl.translate(FreeCAD.Vector(-50, 0, 0))
        front_left_cut = front_left_cut.cut(cyl_fl)
        
        # Apply cuts
        top_half = top_half.cut(front_right_cut).cut(front_left_cut)
        bottom_half = bottom_half.cut(front_right_cut).cut(front_left_cut)

    # 2. Exterior Edge Smoothing (Chamfers)
    if enable_fillets:
        if verbose: print("[STEP 2/7] Applying massive 2mm chamfer to external surfaces...")'''

content = content.replace(old_chamfer_step, new_corner_step)

with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Injected corner rounding logic!")
