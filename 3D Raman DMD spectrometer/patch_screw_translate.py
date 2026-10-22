with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_block = '''        if axial_dz != 0.0:
            box.translate(FreeCAD.Vector(0, 0, axial_dz))
        
        # Inject set screw cylinders locally before transform
        r_screw = Part.makeCylinder(cartridge_screw_radius, 15.0, FreeCAD.Vector(cartridge_screw_x, 26.0, cartridge_screw_z_local), FreeCAD.Vector(0, -1, 0))
        l_screw = Part.makeCylinder(cartridge_screw_radius, 15.0, FreeCAD.Vector(cartridge_screw_x, -26.0, cartridge_screw_z_local), FreeCAD.Vector(0, 1, 0))
        box = box.fuse(r_screw).fuse(l_screw)'''

new_block = '''        # Inject set screw cylinders locally before transform
        r_screw = Part.makeCylinder(cartridge_screw_radius, 15.0, FreeCAD.Vector(cartridge_screw_x, 26.0, cartridge_screw_z_local), FreeCAD.Vector(0, -1, 0))
        l_screw = Part.makeCylinder(cartridge_screw_radius, 15.0, FreeCAD.Vector(cartridge_screw_x, -26.0, cartridge_screw_z_local), FreeCAD.Vector(0, 1, 0))
        box = box.fuse(r_screw).fuse(l_screw)

        if axial_dz != 0.0:
            box.translate(FreeCAD.Vector(0, 0, axial_dz))'''

if old_block in content:
    content = content.replace(old_block, new_block)
    with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("Updated make_aspheric_cradle: screws now translate together with box on axial_dz!")
else:
    print("Pattern not found, checking...")
