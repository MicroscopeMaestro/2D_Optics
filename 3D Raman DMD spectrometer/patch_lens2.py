import re

with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

pattern = r'(\s+if r_min < 12\.0:.*?\n\s+else:\n\s+final_cut = Part\.makeCylinder\(OPTICS_POCKET_RADIUS, length \+ extra_z, FreeCAD\.Vector\(0,0,bb\.ZMin\), FreeCAD\.Vector\(0,0,1\)\))'

new_lens_logic = '''
            if r_min < 12.0:
                # Dome at ZMin: 18mm clear aperture hole, 2mm chamfer bed, full barrel pocket
                cyl_hole = Part.makeCylinder(9.0, bb.ZMin + 5.0 + 10.0, FreeCAD.Vector(0,0,bb.ZMin - 10.0), FreeCAD.Vector(0,0,1))
                cone = Part.makeCone(9.0, OPTICS_POCKET_RADIUS, 2.0, FreeCAD.Vector(0,0,bb.ZMin + 5.0), FreeCAD.Vector(0,0,1))
                cyl_barrel = Part.makeCylinder(OPTICS_POCKET_RADIUS, length - 7.0 + extra_z + 5.0, FreeCAD.Vector(0,0,bb.ZMin + 7.0), FreeCAD.Vector(0,0,1))
                final_cut = cyl_hole.fuse(cone).fuse(cyl_barrel)
            elif r_max < 12.0:
                # Dome at ZMax
                cyl_barrel = Part.makeCylinder(OPTICS_POCKET_RADIUS, length - 7.0 + 5.0, FreeCAD.Vector(0,0,bb.ZMin - 5.0), FreeCAD.Vector(0,0,1))
                cone = Part.makeCone(OPTICS_POCKET_RADIUS, 9.0, 2.0, FreeCAD.Vector(0,0,bb.ZMax - 7.0), FreeCAD.Vector(0,0,1))
                cyl_hole = Part.makeCylinder(9.0, 10.0, FreeCAD.Vector(0,0,bb.ZMax - 5.0), FreeCAD.Vector(0,0,1))
                final_cut = cyl_barrel.fuse(cone).fuse(cyl_hole)
            else:
                final_cut = Part.makeCylinder(OPTICS_POCKET_RADIUS, length + extra_z, FreeCAD.Vector(0,0,bb.ZMin), FreeCAD.Vector(0,0,1))'''

content, count = re.subn(pattern, new_lens_logic, content, flags=re.DOTALL)
print(f"Replaced {count} times")

with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
    f.write(content)

