import os
import sys

FREECAD_BIN = r'C:\Program Files\FreeCAD 1.1\bin'
if FREECAD_BIN not in sys.path:
    sys.path.append(FREECAD_BIN)

import FreeCAD
import Part

def build_slit_cartridge(slit_width_um=200, output_step=None, web_thickness=0.50, verbose=True, thickness=2.0):
    """
    Builds a drop-in optical slit cartridge with single-sided knife-edge relief:
    - The front face at Z = -thickness/2 is flat, with a web_thickness membrane.
    - The single-sided relief expands towards the rear face at Z = +thickness/2.
    - Overall thickness defaults to 2.0 mm and can be set independently of the web.
    - Optical slit: 8.0 mm length along Y, slit_width_um along X (90-degree rotated).
    - Lateral M4 guide channels and top M4 extraction pull-handle hole.
    """
    slit_width_mm = slit_width_um / 1000.0
    if not 0 < web_thickness < thickness:
        raise ValueError('Require 0 < web_thickness < thickness for the relief taper')
    if output_step is None:
        output_step = f"external_slit_cartridge_{slit_width_um}um.step"

    width = 32.0       # X dimension (-14.0 to 18.0 mm)
    height = 22.0      # Y dimension (-11.0 to 11.0 mm)

    # Base solid block
    box = Part.makeBox(width, height, thickness, FreeCAD.Vector(-14.0, -11.0, -thickness / 2.0))

    # Single-sided relief parameters:
    # Keep the membrane thickness while varying the overall structural thickness.
    z_web_inner = -thickness / 2.0 + web_thickness
    z_back_face = thickness / 2.0

    slit_length = 8.00
    slit_margin_y = 0.50

    x_inner = 0.50                                  # Half-width at web (1.0 mm total)
    x_outer = 3.50                                  # Half-width at rear face (7.0 mm wide opening)
    y_inner = slit_length / 2.0 + slit_margin_y      # 4.5 mm (9.0 mm total)
    y_outer = y_inner + 3.0                         # 7.5 mm (15.0 mm total)

    w_in = Part.makePolygon([
        FreeCAD.Vector(-x_inner, -y_inner, z_web_inner),
        FreeCAD.Vector( x_inner, -y_inner, z_web_inner),
        FreeCAD.Vector( x_inner,  y_inner, z_web_inner),
        FreeCAD.Vector(-x_inner,  y_inner, z_web_inner),
        FreeCAD.Vector(-x_inner, -y_inner, z_web_inner)
    ])
    w_out = Part.makePolygon([
        FreeCAD.Vector(-x_outer, -y_outer, z_back_face),
        FreeCAD.Vector( x_outer, -y_outer, z_back_face),
        FreeCAD.Vector( x_outer,  y_outer, z_back_face),
        FreeCAD.Vector(-x_outer,  y_outer, z_back_face),
        FreeCAD.Vector(-x_outer, -y_outer, z_back_face)
    ])
    loft_relief = Part.makeLoft([w_in, w_out], True)
    tapered_box = box.cut(loft_relief)

    # Cut the optical slit through the 0.50 mm front membrane
    slit_cut = Part.makeBox(slit_width_mm, slit_length, web_thickness + 0.3,
                            FreeCAD.Vector(-slit_width_mm / 2.0, -slit_length / 2.0, -thickness / 2.0 - 0.15))
    cartridge = tapered_box.cut(slit_cut)

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
        print(f"Exported: {output_step} (Slit: {slit_width_um} um, Web: {web_thickness} mm on front face, BBox: {cartridge.BoundBox}, Vol: {cartridge.Volume:.2f} mm^3)")

    return cartridge


if __name__ == "__main__":
    build_slit_cartridge(150, "external_slit_cartridge_150um.step")
    build_slit_cartridge(200, "external_slit_cartridge_200um.step")
    build_slit_cartridge(200, "external_slit_cartridge.step")
