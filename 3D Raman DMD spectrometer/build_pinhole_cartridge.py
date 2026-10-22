import os
import sys

FREECAD_BIN = r'C:\Program Files\FreeCAD 1.1\bin'
if FREECAD_BIN not in sys.path:
    sys.path.append(FREECAD_BIN)

import FreeCAD
import Part

def build_pinhole_cartridge(pinhole_diameter_um=100, output_step=None, web_thickness=0.50, verbose=True):
    """
    Builds a drop-in optical pinhole cartridge for alignment:
    - One face (front, Z = -1.5 mm) is completely flat with the thin 0.50 mm pinhole membrane.
    - The relief chamfer/taper expands continuously outward towards the other face (Z = +1.5 mm).
    - Circular pinhole of specified diameter.
    - Lateral M4 guide channels and top M4 extraction pull-handle hole.
    """
    pinhole_diameter_mm = pinhole_diameter_um / 1000.0
    pinhole_radius_mm = pinhole_diameter_mm / 2.0
    
    if output_step is None:
        output_step = f"external_pinhole_cartridge_{pinhole_diameter_um}um.step"

    width = 32.0       # X dimension (-14.0 to 18.0 mm)
    height = 22.0      # Y dimension (-11.0 to 11.0 mm)
    thickness = 3.0    # Z dimension (-1.5 to 1.5 mm)

    # Base solid block
    box = Part.makeBox(width, height, thickness, FreeCAD.Vector(-14.0, -11.0, -thickness / 2.0))

    # Single-sided relief parameters:
    # Front face at Z = -1.5 mm is flat.
    # The thin membrane extends from Z = -1.50 to Z = -1.50 + web_thickness (-1.00 mm).
    # Relief taper expands from Z = -1.00 mm to the rear face at Z = +1.50 mm.
    z_web_inner = -thickness / 2.0 + web_thickness  # -1.00 mm
    z_back_face = thickness / 2.0                   # +1.50 mm

    # Taper size - we'll make it square for the pinhole instead of rectangular
    x_inner = 0.50
    x_outer = 3.50
    y_inner = 0.50
    y_outer = 3.50

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

    # Cut the optical pinhole through the 0.50 mm front membrane
    pinhole_cut = Part.makeCylinder(pinhole_radius_mm, web_thickness + 0.3,
                                    FreeCAD.Vector(0, 0, -thickness / 2.0 - 0.15),
                                    FreeCAD.Vector(0, 0, 1))
    cartridge = tapered_box.cut(pinhole_cut)

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

    # Use robust path resolution
    base_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    full_output_step = os.path.abspath(output_step if os.path.isabs(output_step) else os.path.join(base_dir, output_step))

    os.makedirs(os.path.dirname(full_output_step) or '.', exist_ok=True)
    cartridge.exportStep(full_output_step)
    
    if verbose:
        print(f"Exported: {output_step} (Pinhole: {pinhole_diameter_um} um, Web: {web_thickness} mm on front face, BBox: {cartridge.BoundBox}, Vol: {cartridge.Volume:.2f} mm^3)")

    return cartridge


build_pinhole_cartridge(100, "external_pinhole_cartridge_100um.step")
build_pinhole_cartridge(200, "external_pinhole_cartridge_200um.step")
