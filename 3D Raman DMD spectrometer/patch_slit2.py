with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

import re

new_slit_logic = '''            if "19_input_slit" in low:
                # New: Drop-in pocket for removable 3D-printed slit cartridge
                p_width = 33.4
                p_height = 23.4
                p_thickness = 4.4
                p_z = 81.67081153406
                
                # We place the pocket slightly higher so it doesn't protrude the floor (XMin = -13.7)
                slit_pocket = Part.makeBox(p_width, p_height, p_thickness, 
                                           FreeCAD.Vector(-13.7, -11.7, p_z - 2.2))
                                           
                # Add lateral M4 locking set screw for the slit cartridge
                slit_screw_r = Part.makeCylinder(1.65, 15.0, FreeCAD.Vector(4.0, 20.0, p_z), FreeCAD.Vector(0, -1, 0))
                slit_screw_l = Part.makeCylinder(1.65, 15.0, FreeCAD.Vector(4.0, -20.0, p_z), FreeCAD.Vector(0, 1, 0))
                
                slit_pocket = slit_pocket.fuse(slit_screw_r).fuse(slit_screw_l)
                shapes_to_cut.append((obj.Label, slit_pocket))'''

content = re.sub(r'if "19_input_slit" in low:.*?elif "20_grating" in low:', 
                 new_slit_logic + '\n            elif "20_grating" in low:', content, flags=re.DOTALL)

with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Injected removable slit pocket into spectrometer!")
