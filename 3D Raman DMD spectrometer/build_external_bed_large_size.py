import os
import sys

FREECAD_BIN = r'C:\Program Files\FreeCAD 1.1\bin'
if FREECAD_BIN not in sys.path:
    sys.path.append(FREECAD_BIN)

import FreeCAD
import Part

# Parameters for the large-size external aspheric mount
width = 36.0               # Outer width along X (X from -14.0 to +22.0 mm)
height = 32.0              # Outer height along Y (Y from -16.0 to +16.0 mm)
forward_extension = 8.0    # 8.0 mm extension into the optical tunnel (ZMin = -8.0 mm)
total_thickness = 18.693   # Preserved total length matching previous casing pocket

pocket_radius = 13.40      # Barrel pocket radius (26.8 mm diameter) on large aperture side
shelf_radius = 13.00       # Intermediate shelf radius
aperture_radius = 10.50    # Clear aperture radius (21.0 mm diameter) on small aperture side

# Reduced large aperture depth by 2 mm fewer (delta_z = 1.0 mm instead of 3.0 mm)
# Large aperture depth becomes 3.09 mm (was 5.09 mm), giving 0.5 mm clearance behind lens to the 1.5mm lip
delta_z = 1.00
lip_height = 3.0           # 3.0 mm high retaining lip on the large aperture surface
lip_thickness_z = 1.50     # 1.5 mm axial thickness of the retaining lip on the back surface

CARTRIDGE_CHANNEL_DEPTH = 1.2
CARTRIDGE_CHANNEL_WIDTH = 5.0
CARTRIDGE_CHANNEL_X_CENTER = 4.0
CARTRIDGE_CHANNEL_Z_MIN = 2.0
CARTRIDGE_CHANNEL_Z_MAX = 16.0


def build_large_external_bed(doc_path="model1.FCStd",
                             output_step="external_aspheric_bed_large_size.step",
                             p_radius=pocket_radius,
                             s_radius=shelf_radius,
                             a_radius=aperture_radius,
                             shift_dz=delta_z,
                             lip_h=lip_height,
                             lip_tz=lip_thickness_z,
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

    lens_obj = next((o for o in asm.Group if "01_collect_acl25416u" in o.Label.lower() and "glass" in o.Label.lower()), None)
    if lens_obj is None:
        raise RuntimeError("01_collect_acl25416u glass lens object not found")

    local_s = lens_obj.Shape.copy().transformShape(lens_obj.Placement.Matrix.inverse())
    bb = local_s.BoundBox

    z_min = -forward_extension  # -8.00 mm (preserved)
    z_max = bb.ZMax + 1.0       # 10.693 mm (preserved)
    thickness = z_max - z_min   # 18.693 mm (constant length!)

    # Internal conical and barrel stations:
    z_shelf = 8.60 - shift_dz          # 7.60 mm (shelf position)
    z_taper_start = 1.40 - shift_dz    # 0.40 mm (conical taper start)
    # Constant small aperture depth = z_taper_start - z_min = 0.40 - (-8.00) = 8.40 mm
    # Large aperture pocket depth = z_max - z_shelf = 10.693 - 7.60 = 3.093 mm (2 mm fewer than 5.093 mm!)

    # Base block for large mount
    box = Part.makeBox(width, height, thickness, FreeCAD.Vector(-14.0, -16.0, z_min))

    # Revolved internal bore
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
    mount = box.cut(cutout)

    # 3mm high retaining lip on the surface of the large aperture side (at Z = z_max):
    # Bottom of barrel bore is at X = -p_radius (-13.40 mm).
    # Lip rises 3.0 mm from X = -14.0 (mount bottom) to X = -10.40 mm.
    # Along Z: flush with the back surface, thickness = lip_tz (1.5 mm)
    lip_x_max = -p_radius + lip_h  # -10.40 mm
    lip_thick_x = lip_x_max - (-14.0)  # 3.60 mm
    lip_z_start = z_max - lip_tz
    lip_box = Part.makeBox(lip_thick_x, 20.0, lip_tz, FreeCAD.Vector(-14.0, -10.0, lip_z_start))
    mount = mount.fuse(lip_box)

    # Top M4 clamping set screw positioned over the shifted barrel seat (centered at Z = 8.0 mm)
    screw_z = 8.00  # Centered on the lens barrel (Z in [7.60, 8.69])
    cb_depth = 4.5
    cb_radius = 3.6
    cb = Part.makeCylinder(cb_radius, cb_depth, FreeCAD.Vector(22.0, 0, screw_z), FreeCAD.Vector(-1, 0, 0))
    mount = mount.cut(cb)

    th_depth = 5.0
    th_radius = 1.65
    m4_hole = Part.makeCylinder(th_radius, th_depth + 2.0, FreeCAD.Vector(22.0 - cb_depth, 0, screw_z), FreeCAD.Vector(-1, 0, 0))
    mount = mount.cut(m4_hole)

    # Lateral locking channels
    right_channel = Part.makeBox(CARTRIDGE_CHANNEL_WIDTH, CARTRIDGE_CHANNEL_DEPTH + 0.1, CARTRIDGE_CHANNEL_Z_MAX - CARTRIDGE_CHANNEL_Z_MIN,
                                 FreeCAD.Vector(CARTRIDGE_CHANNEL_X_CENTER - CARTRIDGE_CHANNEL_WIDTH / 2.0, 16.0 - CARTRIDGE_CHANNEL_DEPTH, CARTRIDGE_CHANNEL_Z_MIN))
    left_channel = Part.makeBox(CARTRIDGE_CHANNEL_WIDTH, CARTRIDGE_CHANNEL_DEPTH + 0.1, CARTRIDGE_CHANNEL_Z_MAX - CARTRIDGE_CHANNEL_Z_MIN,
                                FreeCAD.Vector(CARTRIDGE_CHANNEL_X_CENTER - CARTRIDGE_CHANNEL_WIDTH / 2.0, -16.0 - 0.1, CARTRIDGE_CHANNEL_Z_MIN))
    mount = mount.cut(right_channel).cut(left_channel)

    os.makedirs(os.path.dirname(full_output_step) or '.', exist_ok=True)
    mount.exportStep(full_output_step)

    if verbose:
        print(f"Exported large mount to {output_step}!")
        print(f"Bounding box: {mount.BoundBox}")
        print(f"Total mount length: {thickness:.3f} mm (constant preserved)")
        print(f"Large aperture barrel pocket depth: {z_max - z_shelf:.3f} mm (reduced by 2mm from 5.093mm)")
        print(f"Constant small aperture tunnel depth: {z_taper_start - z_min:.3f} mm")
        print(f"Volume: {mount.Volume:.2f} mm^3")
        shifted_lens = local_s.copy()
        shifted_lens.translate(FreeCAD.Vector(0, 0, -shift_dz))
        print(f"Intersection with lens: {mount.common(shifted_lens).Volume:.6f} mm^3")
        beam = Part.makeCylinder(9.0, 25.0, FreeCAD.Vector(0, 0, -8.5), FreeCAD.Vector(0, 0, 1))
        print(f"Intersection with 18mm optical beam: {mount.common(beam).Volume:.6f} mm^3")

    return mount


if __name__ == "__main__":
    build_large_external_bed()
