import os
import sys

FREECAD_BIN = r'C:\Program Files\FreeCAD 1.1\bin'
if FREECAD_BIN not in sys.path:
    sys.path.append(FREECAD_BIN)

import FreeCAD
import Part

# Default module-level precision parameters
pocket_radius = 13.40      # Generous radial tolerance for standard 25.4mm barrel
shelf_radius = 12.50       # Intermediate shelf radius
aperture_radius = 10.50    # Minimum clear aperture radius (10.5mm radius = 21mm clear aperture > 18mm required)

CARTRIDGE_CHANNEL_DEPTH = 1.2
CARTRIDGE_CHANNEL_WIDTH = 5.0
CARTRIDGE_CHANNEL_X_CENTER = 4.0
CARTRIDGE_CHANNEL_Z_MIN = 2.0
CARTRIDGE_CHANNEL_Z_MAX = 10.0


def build_external_bed(doc_path="model1.FCStd",
                       output_step="external_aspheric_bed.step",
                       width_x=34.00,
                       height_y=32.50,
                       p_radius=pocket_radius,
                       s_radius=shelf_radius,
                       a_radius=aperture_radius,
                       z_shelf=8.60,
                       z_taper_start=0.40,
                       lip_thickness=0.60,
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

    z_min = -0.10
    z_max = 10.693
    thickness = z_max - z_min  # 10.793 mm
    x_min = -13.00             # X in [-13.00, +21.00], width = 34.00 mm

    pts = [
        FreeCAD.Vector(0, 0, z_min),
        FreeCAD.Vector(a_radius, 0, z_min),
        FreeCAD.Vector(a_radius, 0, z_taper_start),
        FreeCAD.Vector(s_radius, 0, z_shelf),
        FreeCAD.Vector(p_radius, 0, z_shelf),
        FreeCAD.Vector(p_radius, 0, z_max),
        FreeCAD.Vector(0, 0, z_max),
        FreeCAD.Vector(0, 0, z_min)
    ]

    wire = Part.makePolygon(pts)
    face = Part.Face(wire)
    cutout = face.revolve(FreeCAD.Vector(0, 0, 0), FreeCAD.Vector(0, 0, 1), 360)

    box = Part.makeBox(width_x, height_y, thickness, FreeCAD.Vector(x_min, -height_y / 2.0, z_min))
    mount = box.cut(cutout)

    # Retaining lip removed per user request

    # Top M4 set screw (XMax = 21.0, enters from +21.0)
    cb_depth = 4.5
    cb_radius = 3.6
    screw_z = 8.193
    cb = Part.makeCylinder(cb_radius, cb_depth, FreeCAD.Vector(x_min + width_x, 0, screw_z), FreeCAD.Vector(-1, 0, 0))
    mount = mount.cut(cb)

    th_depth = 5.0
    th_radius = 1.65
    m4_hole = Part.makeCylinder(th_radius, th_depth + 2.0, FreeCAD.Vector(x_min + width_x - cb_depth, 0, screw_z), FreeCAD.Vector(-1, 0, 0))
    mount = mount.cut(m4_hole)

    # Lateral guide channels
    right_channel = Part.makeBox(channel_width, channel_depth + 0.1, channel_z_max - channel_z_min,
                                 FreeCAD.Vector(channel_x_center - channel_width / 2.0, height_y / 2.0 - channel_depth, channel_z_min))
    left_channel = Part.makeBox(channel_width, channel_depth + 0.1, channel_z_max - channel_z_min,
                                FreeCAD.Vector(channel_x_center - channel_width / 2.0, -height_y / 2.0 - 0.1, channel_z_min))
    mount = mount.cut(right_channel).cut(left_channel)

    os.makedirs(os.path.dirname(full_output_step) or '.', exist_ok=True)
    mount.exportStep(full_output_step)
    collision = mount.common(local_s).Volume
    if verbose:
        print(f"Exported standalone {output_step}!")
        print(f"  BoundBox: {mount.BoundBox}")
        print(f"  Width (X): {mount.BoundBox.XLength:.3f} mm, Height (Y): {mount.BoundBox.YLength:.3f} mm, Thickness (Z): {mount.BoundBox.ZLength:.3f} mm")
        print(f"  Collision with lens: {collision:.6f} mm^3")

    return mount


def is_main_entry(script_name="build_external_bed.py"):
    if __name__ == "__main__":
        return True
    this_file = os.path.abspath(__file__) if '__file__' in globals() else script_name
    for arg in sys.argv[1:]:
        if arg in ("--", "--pass"): break
        if arg.startswith("-"): continue
        try:
            if os.path.samefile(arg, this_file) or os.path.basename(arg) == os.path.basename(this_file): return True
        except:
            if os.path.basename(arg) == os.path.basename(this_file): return True
        if arg.endswith(".py"): return False
    return False


if __name__ == "__main__" or is_main_entry("build_external_bed.py"):
    # 1. Normal Case: 34.0 x 32.5 x 10.79 mm
    build_external_bed(width_x=34.00, height_y=32.50, output_step="external_aspheric_bed.step")
    # 2. Special Case (entrance mount): 34.0 x 33.5 x 10.79 mm
    build_external_bed(width_x=34.00, height_y=33.50, output_step="external_aspheric_bed_33_5mm_width.step")
