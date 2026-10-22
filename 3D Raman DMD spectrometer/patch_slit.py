with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

import re

# Find the old slit_cut logic
old_slit_logic = '''            if "19_input_slit" in low:
                # R4: 3D-printed 200 µm rectangular optical slit cut through solid barrier wall
                SLIT_WIDTH = 0.2
                SLIT_HEIGHT = 8.0
                SLIT_THICKNESS = 2.2
                slit_cut = Part.makeBox(SLIT_HEIGHT, SLIT_WIDTH, SLIT_THICKNESS,
                                        FreeCAD.Vector(-SLIT_HEIGHT/2.0, -SLIT_WIDTH/2.0, 80.67081153406 - 0.1))
                shapes_to_cut.append((obj.Label, slit_cut))'''

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

# Wait, the R4 comment might have a weird character 'µ'. Let's use regex to replace everything under '19_input_slit'
content = re.sub(r'if "19_input_slit" in low:.*?elif "20_grating" in low:', 
                 new_slit_logic + '\n            elif "20_grating" in low:', content, flags=re.DOTALL)

with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Injected removable slit pocket into spectrometer!")
