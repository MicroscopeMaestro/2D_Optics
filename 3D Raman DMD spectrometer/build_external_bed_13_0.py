import os
import sys

FREECAD_BIN = r'C:\Program Files\FreeCAD 1.1\bin'
if FREECAD_BIN not in sys.path:
    sys.path.append(FREECAD_BIN)

import FreeCAD
import Part

# Default module-level precision parameters
pocket_radius = 13.4       # Generous radial tolerance for standard FDM (26.8mm diameter)
clearance = 0.15           # Conic slope clearance
CARTRIDGE_CHANNEL_DEPTH = 1.2
CARTRIDGE_CHANNEL_WIDTH = 5.0
CARTRIDGE_CHANNEL_X_CENTER = 4.0
CARTRIDGE_CHANNEL_Z_MIN = 2.0
CARTRIDGE_CHANNEL_Z_MAX = 16.0


def build_external_bed(doc_path="model1.FCStd",
                       output_step="external_aspheric_bed_13mm.step",
                       p_radius=pocket_radius,
                       c_clearance=clearance,
                       channel_depth=CARTRIDGE_CHANNEL_DEPTH,
                       channel_width=CARTRIDGE_CHANNEL_WIDTH,
                       channel_x_center=CARTRIDGE_CHANNEL_X_CENTER,
                       channel_z_min=CARTRIDGE_CHANNEL_Z_MIN,
                       channel_z_max=CARTRIDGE_CHANNEL_Z_MAX,
                       verbose=True):
    """
    Constructs the standalone external aspheric lens bed (cartridge) with:
    - 26.8mm lens pocket radius (r=13.4mm)
    - 0.15mm conic slope clearance
    - >= 18.0mm clear aperture (designed 20.0mm aperture)
    - Top M4 lens-clamping set screw hole with counterbore
    - Dual parameterized lateral locking guide channels for mechanical casing retention
    """
    base_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    full_doc_path = os.path.abspath(doc_path if os.path.isabs(doc_path) else os.path.join(base_dir, doc_path))
    full_output_step = os.path.abspath(output_step if os.path.isabs(output_step) else os.path.join(base_dir, output_step))

    if not os.path.exists(full_doc_path):
        raise FileNotFoundError(f"CAD document not found: {full_doc_path}")

    doc = FreeCAD.open(full_doc_path)
    asm = doc.getObject('Optical_System_Assembly')
    if asm is None:
        raise RuntimeError("Optical_System_Assembly not found in document")

    # Find the ACL25416U lens
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

    pts = [FreeCAD.Vector(0, 0, z_min), FreeCAD.Vector(p_radius, 0, z_min)]
    z_curr = bb.ZMin
    while z_curr <= bb.ZMax:
        sl = local_s.slice(FreeCAD.Vector(0, 0, 1), min(z_curr + 0.05, bb.ZMax - 0.01))
        r = sl[0].BoundBox.XMax if sl else 0.0
        if r > 12.0:
            r_cleared = p_radius
        else:
            r_cleared = max(r + 1.0, 10.0)
        pts.append(FreeCAD.Vector(r_cleared, 0, z_curr))
        z_curr += 0.2
    pts.append(FreeCAD.Vector(p_radius, 0, bb.ZMax))
    pts.append(FreeCAD.Vector(p_radius, 0, z_max))
    pts.append(FreeCAD.Vector(0, 0, z_max))
    pts.append(FreeCAD.Vector(0, 0, z_min))

    wire = Part.makePolygon(pts)
    face = Part.Face(wire)
    cutout = face.revolve(FreeCAD.Vector(0, 0, 0), FreeCAD.Vector(0, 0, 1), 360)

    # 34.0 x 32.0 mm block footprint (matching 35.4 x 33.4 mm casing pocket with 0.70mm slip-fit clearance)
    width = 34.0
    height = 32.0
    thickness = z_max - z_min

    box = Part.makeBox(width, height, thickness, FreeCAD.Vector(-13.0, -16.0, z_min))

    # Subtract the conic cutout
    standalone_bed = box.cut(cutout)

    # Top M4 set screw hole down to lens barrel
    cb_depth = 4.5
    cb_radius = 3.6
    cb = Part.makeCylinder(cb_radius, cb_depth, FreeCAD.Vector(21.0, 0, bb.ZMax - 1.5), FreeCAD.Vector(-1, 0, 0))
    standalone_bed = standalone_bed.cut(cb)

    th_depth = 5.0
    th_radius = 1.65
    m4_hole = Part.makeCylinder(th_radius, th_depth + 2.0, FreeCAD.Vector(21.0 - cb_depth, 0, bb.ZMax - 1.5), FreeCAD.Vector(-1, 0, 0))
    standalone_bed = standalone_bed.cut(m4_hole)

    # Cut matching axial guide channels on lateral walls (Y = +16.0 and Y = -16.0)
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
