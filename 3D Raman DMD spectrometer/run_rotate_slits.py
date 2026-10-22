import os
import sys

FREECAD_BIN = r'C:\Program Files\FreeCAD 1.1\bin'
if FREECAD_BIN not in sys.path:
    sys.path.append(FREECAD_BIN)

import FreeCAD
import Part

def build_slit_cartridge(slit_width_um=200, output_step=None, verbose=True):
    slit_width_mm = slit_width_um / 1000.0
    if output_step is None:
        output_step = f"external_slit_cartridge_{slit_width_um}um.step"

    width = 32.0       # X dimension (-14.0 to 18.0 mm)
    height = 22.0      # Y dimension (-11.0 to 11.0 mm)
    thickness = 3.0    # Z dimension (-1.5 to 1.5 mm)

    # Base solid block
    box = Part.makeBox(width, height, thickness, FreeCAD.Vector(-14.0, -11.0, -thickness / 2.0))

    # Optical slit cut ROTATED 90 DEGREES:
    # Long direction is along Y (8.0 mm, from -4.0 to +4.0)
    # Narrow slit width is along X (slit_width_mm, from -slit_width_mm/2 to +slit_width_mm/2)
    slit_length = 8.0
    slit_cut = Part.makeBox(slit_width_mm, slit_length, thickness + 0.4,
                            FreeCAD.Vector(-slit_width_mm / 2.0, -slit_length / 2.0, -thickness / 2.0 - 0.2))
    cartridge = box.cut(slit_cut)

    # Lateral guide channels for casing M4 locking set screws (aligned at X = 4.0 mm)
    channel_depth = 1.2
    channel_width = 3.0
    channel_x = 4.0
    right_channel = Part.makeBox(channel_width, channel_depth + 0.1, thickness + 0.4,
                                 FreeCAD.Vector(channel_x - channel_width / 2.0, height / 2.0 - channel_depth, -thickness / 2.0 - 0.2))
    left_channel = Part.makeBox(channel_width, channel_depth + 0.1, thickness + 0.4,
                                FreeCAD.Vector(channel_x - channel_width / 2.0, -height / 2.0 - 0.1, -thickness / 2.0 - 0.2))
    cartridge = cartridge.cut(right_channel).cut(left_channel)

    # Top M4 pull-handle hole for easy extraction
    pull_hole = Part.makeCylinder(1.65, 6.0, FreeCAD.Vector(18.0, 0, 0), FreeCAD.Vector(-1, 0, 0))
    cartridge = cartridge.cut(pull_hole)

    cartridge.exportStep(output_step)
    if verbose:
        print(f"Exported: {output_step} (Slit: {slit_width_um} um, oriented 90 deg rotated: length 8mm along Y, width along X)")

    return cartridge

build_slit_cartridge(150, "external_slit_cartridge_150um.step")
build_slit_cartridge(200, "external_slit_cartridge_200um.step")
build_slit_cartridge(200, "external_slit_cartridge.step")
