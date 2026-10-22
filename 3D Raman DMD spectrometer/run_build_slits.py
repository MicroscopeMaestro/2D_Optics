import os
import sys

FREECAD_BIN = r'C:\Program Files\FreeCAD 1.1\bin'
if FREECAD_BIN not in sys.path:
    sys.path.append(FREECAD_BIN)

import FreeCAD
import Part

def create_slit_cartridge(slit_width_mm, output_step):
    width = 32.0       # X dimension (-14.0 to 18.0)
    height = 22.0      # Y dimension (-11.0 to 11.0)
    thickness = 3.0    # Z dimension (-1.5 to 1.5)
    
    # Base solid block
    box = Part.makeBox(width, height, thickness, FreeCAD.Vector(-14.0, -11.0, -thickness/2.0))
    
    # Optical slit cut (8.0 mm vertical height in X, slit_width in Y, completely through Z)
    slit_height = 8.0
    slit_cut = Part.makeBox(slit_height, slit_width_mm, thickness + 0.4,
                            FreeCAD.Vector(-slit_height/2.0, -slit_width_mm/2.0, -thickness/2.0 - 0.2))
    cartridge = box.cut(slit_cut)
    
    # Lateral guide channels for casing M4 locking set screws (aligned at X = 4.0)
    channel_depth = 1.2
    channel_width = 3.0
    channel_x = 4.0
    right_channel = Part.makeBox(channel_width, channel_depth + 0.1, thickness + 0.4,
                                 FreeCAD.Vector(channel_x - channel_width/2.0, height/2.0 - channel_depth, -thickness/2.0 - 0.2))
    left_channel = Part.makeBox(channel_width, channel_depth + 0.1, thickness + 0.4,
                                FreeCAD.Vector(channel_x - channel_width/2.0, -height/2.0 - 0.1, -thickness/2.0 - 0.2))
    cartridge = cartridge.cut(right_channel).cut(left_channel)
    
    # Top M4 pull-handle hole for easy extraction
    pull_hole = Part.makeCylinder(1.65, 6.0, FreeCAD.Vector(18.0, 0, 0), FreeCAD.Vector(-1, 0, 0))
    cartridge = cartridge.cut(pull_hole)
    
    cartridge.exportStep(output_step)
    print(f"Exported: {output_step} (Slit: {slit_width_mm*1000:.0f} um, BoundBox: {cartridge.BoundBox}, Vol: {cartridge.Volume:.2f} mm^3)")
    return cartridge

c150 = create_slit_cartridge(0.150, "external_slit_cartridge_150um.step")
c200 = create_slit_cartridge(0.200, "external_slit_cartridge_200um.step")
# Also update the default external_slit_cartridge.step with the 200um standard
create_slit_cartridge(0.200, "external_slit_cartridge.step")
