with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_logic = '''        top_half = top_half.cut(front_right_cut).cut(front_left_cut)
        bottom_half = bottom_half.cut(front_right_cut).cut(front_left_cut)'''

new_logic = '''        # Rear corners (Z=240)
        rear_right_cut = Part.makeBox(100, r, r, FreeCAD.Vector(-50, 50 - r, 240 - r))
        cyl_rr = Part.makeCylinder(r, 100, FreeCAD.Vector(0, 50 - r, 240 - r), FreeCAD.Vector(1,0,0))
        cyl_rr.translate(FreeCAD.Vector(-50, 0, 0))
        rear_right_cut = rear_right_cut.cut(cyl_rr)
        
        rear_left_cut = Part.makeBox(100, r, r, FreeCAD.Vector(-50, -58.5, 240 - r))
        cyl_rl = Part.makeCylinder(r, 100, FreeCAD.Vector(0, -58.5 + r, 240 - r), FreeCAD.Vector(1,0,0))
        cyl_rl.translate(FreeCAD.Vector(-50, 0, 0))
        rear_left_cut = rear_left_cut.cut(cyl_rl)
        
        top_half = top_half.cut(front_right_cut).cut(front_left_cut).cut(rear_right_cut).cut(rear_left_cut)
        bottom_half = bottom_half.cut(front_right_cut).cut(front_left_cut).cut(rear_right_cut).cut(rear_left_cut)'''

content = content.replace(old_logic, new_logic)

with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Injected rear corner rounding logic!")
