with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if 'if "19_input_slit" in low:' in line:
        # Align this line and the following to 12 spaces
        lines[i] = '            if "19_input_slit" in low:\n'
    elif 'p_width = 33.4' in line:
        lines[i] = '                p_width = 33.4\n'
    elif 'p_height = 23.4' in line:
        lines[i] = '                p_height = 23.4\n'
    elif 'p_thickness = 4.4' in line:
        lines[i] = '                p_thickness = 4.4\n'
    elif 'p_z = 81.67081153406' in line:
        lines[i] = '                p_z = 81.67081153406\n'
    elif '# We place the pocket' in line:
        lines[i] = '                # We place the pocket slightly higher so it doesn\'t protrude the floor (XMin = -13.7)\n'
    elif 'slit_pocket = Part.makeBox' in line:
        lines[i] = '                slit_pocket = Part.makeBox(p_width, p_height, p_thickness, FreeCAD.Vector(-13.7, -11.7, p_z - 2.2))\n'
    elif 'FreeCAD.Vector(-13.7, -11.7, p_z - 2.2))' in line and 'slit_pocket =' not in line:
        lines[i] = '\n'
    elif '# Add lateral M4' in line:
        lines[i] = '                # Add lateral M4 locking set screw for the slit cartridge\n'
    elif 'slit_screw_r = ' in line:
        lines[i] = '                slit_screw_r = Part.makeCylinder(1.65, 15.0, FreeCAD.Vector(4.0, 20.0, p_z), FreeCAD.Vector(0, -1, 0))\n'
    elif 'slit_screw_l = ' in line:
        lines[i] = '                slit_screw_l = Part.makeCylinder(1.65, 15.0, FreeCAD.Vector(4.0, -20.0, p_z), FreeCAD.Vector(0, 1, 0))\n'
    elif 'slit_pocket = slit_pocket.fuse' in line:
        lines[i] = '                slit_pocket = slit_pocket.fuse(slit_screw_r).fuse(slit_screw_l)\n'
    elif 'shapes_to_cut.append((obj.Label, slit_pocket))' in line:
        lines[i] = '                shapes_to_cut.append((obj.Label, slit_pocket))\n'
    elif 'elif "20_grating" in low:' in line:
        lines[i] = '            elif "20_grating" in low:\n'

with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)
print("Fixed indentation!")
