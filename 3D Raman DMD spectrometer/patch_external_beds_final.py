with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
in_func = False
for line in lines:
    new_lines.append(line)
    if 'bb = local_s.BoundBox' in line and in_func:
        # inject the logic right here
        injection = '''
    if USE_EXTERNAL_ASPHERIC_MOUNTS:
        z_min = bb.ZMin - 0.1
        z_max = bb.ZMax + 5.0
        p_z_min = z_min - sliding_tolerance - 0.2
        p_z_max = z_max + sliding_tolerance + 0.2
        p_thickness = p_z_max - p_z_min
        p_width = 40.4
        p_height = 40.4
        box = Part.makeBox(p_width, p_height, p_thickness, FreeCAD.Vector(-p_width/2, -p_height/2, p_z_min))
        if axial_dz != 0.0:
            box.translate(FreeCAD.Vector(0, 0, axial_dz))
        box.transformShape(obj.Placement.Matrix)
        return box
'''
        new_lines.append(injection)
        in_func = False
    if 'def make_aspheric_cradle(' in line:
        in_func = True

with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)
print("Injected logic reliably!")
