import os
import sys

FREECAD_BIN = r'C:\Program Files\FreeCAD 1.1\bin'
if FREECAD_BIN not in sys.path:
    sys.path.append(FREECAD_BIN)

import FreeCAD
import Part

# Default module-level precision parameters
pocket_radius = 13.40      # Generous radial tolerance for standard 25.4mm barrel
shelf_radius = 13.00       # Intermediate shelf radius
aperture_radius = 10.00    # Minimum clear aperture radius
taper_depth = 4.50         # Conical taper thickness/depth along Z

CARTRIDGE_CHANNEL_DEPTH = 1.2
CARTRIDGE_CHANNEL_WIDTH = 5.0
CARTRIDGE_CHANNEL_X_CENTER = 4.0
CARTRIDGE_CHANNEL_Z_MIN = 2.0
CARTRIDGE_CHANNEL_Z_MAX = 16.0


def build_external_bed(doc_path="model1.FCStd",
                       output_step="external_aspheric_bed_4_5mm_taper.step",
                       p_radius=pocket_radius,
                       s_radius=shelf_radius,
                       a_radius=aperture_radius,
                       t_depth=taper_depth,
                       channel_depth=CARTRIDGE_CHANNEL_DEPTH,
                       channel_width=CARTRIDGE_CHANNEL_WIDTH,
                       channel_x_center=CARTRIDGE_CHANNEL_X_CENTER,
                       channel_z_min=CARTRIDGE_CHANNEL_Z_MIN,
                       channel_z_max=CARTRIDGE_CHANNEL_Z_MAX,
                       verbose=True):
    base_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    full_doc_path = os.path.abspath(doc_path if os.path.isabs(doc_path) else os.path.join(base_dir, doc_path))
    full_output_step = os.path.abspath(output_step if os.path.isabs(output_step) else os.path.join(base_dir, output_step))

    if not os.path.exists(full_doc_path):
        raise FileNotFoundError(f"CAD document not found: {full_doc_path}")

    doc = FreeCAD.open(full_doc_path)
    asm = doc.getObject('Optical_System_Assembly')
    if asm is None:
        raise RuntimeError("Optical_System_Assembly not found in document")

    lens_obj = None
    for obj in asm.Group:
        if "01_collect_acl25416u" in obj.Label.lower() and "glass" in obj.Label.lower():
            lens_obj = obj
            break
    if lens_obj is None:
        raise RuntimeError("01_collect_acl25416u glass lens object not found")

    local_s = lens_obj.Shape.copy().transformShape(lens_obj.Placement.Matrix.inverse())
    bb = local_s.BoundBox

    z_min = bb.ZMin - 0.1
    z_max = bb.ZMax + 1.0

    z_shelf = 8.60
    z_taper_end = z_shelf - t_depth  # 4.10 mm for 4.5mm taper

    pts = [
        FreeCAD.Vector(0, 0, z_min),
        FreeCAD.Vector(a_radius, 0, z_min),
        FreeCAD.Vector(a_radius, 0, z_taper_end),
        FreeCAD.Vector(s_radius, 0, z_shelf),
        FreeCAD.Vector(p_radius, 0, z_shelf),
        FreeCAD.Vector(p_radius, 0, z_max),
        FreeCAD.Vector(0, 0, z_max),
        FreeCAD.Vector(0, 0, z_min)
    ]

    wire = Part.makePolygon(pts)
    face = Part.Face(wire)
    cutout = face.revolve(FreeCAD.Vector(0, 0, 0), FreeCAD.Vector(0, 0, 1), 360)

    # Standard compact 34.0 x 32.0 mm footprint (fits 35.4 x 33.4 mm casing pocket)
    width = 34.0
    height = 32.0
    thickness = z_max - z_min

    box = Part.makeBox(width, height, thickness, FreeCAD.Vector(-13.0, -16.0, z_min))
    standalone_bed = box.cut(cutout)

    # Top M4 clamping set screw
    cb_depth = 4.5
    cb_radius = 3.6
    cb = Part.makeCylinder(cb_radius, cb_depth, FreeCAD.Vector(21.0, 0, bb.ZMax - 1.5), FreeCAD.Vector(-1, 0, 0))
    standalone_bed = standalone_bed.cut(cb)

    th_depth = 5.0
    th_radius = 1.65
    m4_hole = Part.makeCylinder(th_radius, th_depth + 2.0, FreeCAD.Vector(21.0 - cb_depth, 0, bb.ZMax - 1.5), FreeCAD.Vector(-1, 0, 0))
    standalone_bed = standalone_bed.cut(m4_hole)

    # Lateral locking channels
    right_channel = Part.makeBox(channel_width, channel_depth + 0.1, channel_z_max - channel_z_min,
                                 FreeCAD.Vector(channel_x_center - channel_width / 2.0, 16.0 - channel_depth, channel_z_min))
    left_channel = Part.makeBox(channel_width, channel_depth + 0.1, channel_z_max - channel_z_min,
                                FreeCAD.Vector(channel_x_center - channel_width / 2.0, -16.0 - 0.1, channel_z_min))
    standalone_bed = standalone_bed.cut(right_channel).cut(left_channel)

    os.makedirs(os.path.dirname(full_output_step) or '.', exist_ok=True)
    standalone_bed.exportStep(full_output_step)
    if verbose:
        print(f"Exported standalone {output_step}!")

    return standalone_bed


if __name__ == "__main__":
    build_external_bed()
