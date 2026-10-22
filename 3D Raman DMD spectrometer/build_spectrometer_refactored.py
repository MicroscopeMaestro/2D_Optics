"""
build_spectrometer.py

Consolidated, production-ready FreeCAD CAD generation script for the 2D Raman DMD
Spectrometer clamshell enclosure.
"""

import os
import sys
import math
import argparse
import FreeCAD
import Part

# ==============================================================================
# CONFIGURATION CONSTANTS (Easily customized for different CAD documents)
# ==============================================================================
DEFAULT_INPUT_DOCUMENT      = "model1.FCStd"
DEFAULT_SUPPORT_OBJECT      = "Sketch"
DEFAULT_OPTICS_ASSEMBLY     = "Optical_System_Assembly"
DEFAULT_THORLABS_STEP       = "ACL25416U-Step.step"

MANUAL_SCREW_POSITIONS = [
    (16.0, 181.0), (-55.0, 227.0), (19.0, 115.0), 
    (-21.0, 115.0), (20.0, 8.0), (-16.5, 30.0)
]
MANUAL_PEG_POSITIONS = [
    (19.0, 170.0), (-21.0, 196.0), (19.0, 65.0), (19.0, 35.0)
]
MANUAL_BREADBOARD_POSITIONS = None

LATERAL_HOLES_RIGHT = [
    (8.75, 25.0, 50.0), (8.75, 25.0, 70.0), (8.75, 25.0, 90.0),
    (8.75, 25.0, 110.0), (8.75, 25.0, 130.0), (8.75, 25.0, 150.0), (8.75, 25.0, 170.0),
    (-8.75, 25.0, 50.0), (-8.75, 25.0, 70.0), (-8.75, 25.0, 90.0),
    (-8.75, 25.0, 110.0), (-8.75, 25.0, 130.0), (-8.75, 25.0, 150.0), (-8.75, 25.0, 170.0)
]
LATERAL_HOLES_LEFT_1 = [
    (8.75, -21.0, 50.0), (8.75, -21.0, 70.0), (8.75, -21.0, 90.0),
    (-8.75, -21.0, 50.0), (-8.75, -21.0, 70.0), (-8.75, -21.0, 90.0)
]
LATERAL_HOLES_LEFT_2 = [
    (8.75, -61.0, 150.0), (8.75, -61.0, 170.0), (8.75, -61.0, 190.0), (8.75, -61.0, 210.0),
    (-8.75, -61.0, 150.0), (-8.75, -61.0, 170.0), (-8.75, -61.0, 190.0), (-8.75, -61.0, 210.0)
]
LATERAL_HOLES_DIAGONAL = [
    (8.75, -31.0, 114.5), (8.75, -41.0, 124.0), (8.75, -51.0, 133.5),
    (-8.75, -31.0, 114.5), (-8.75, -41.0, 124.0), (-8.75, -51.0, 133.5)
]
LATERAL_HOLES_RIGHT_TAPER = [(8.75, 21.0, 27.0), (-8.75, 21.0, 27.0)]
LATERAL_HOLES_LEFT_TAPER = [(8.75, -19.0, 27.0), (-8.75, -19.0, 27.0)]
LATERAL_HOLES_TOP = [
    (8.75, 10.0, 233.0), (8.75, -10.0, 233.0), (8.75, -30.0, 233.0), (8.75, -50.0, 233.0),
    (-8.75, 10.0, 233.0), (-8.75, -10.0, 233.0), (-8.75, -30.0, 233.0), (-8.75, -50.0, 233.0)
]
LATERAL_HOLE_DEPTH = 5.0

CLAMSHELL_HALF_THICKNESS    = 17.5       
BREADBOARD_THICKNESS        = 3.0        
BREADBOARD_LEFT_EXTENSION   = 60.0       
BREADBOARD_RIGHT_EXTENSION  = 0.0        
BREADBOARD_GRID_SPACING     = 10.0       
BREADBOARD_TAP_DIAMETER     = 3.3        
BREADBOARD_HOLE_DEPTH       = 3.5        
BREADBOARD_EDGE_MARGIN      = 4.0        

SENSOR_M4_HOLES_SPACING     = 30.0       
SENSOR_M4_HOLE_DEPTH        = 15.0       
DMD_M4_HOLES_SPACING        = 30.0       
DMD_M4_HOLE_DEPTH           = 10.0       
BREADBOARD_OPTICS_CLEARANCE = 6.0        

ENABLE_PERIMETER_INTERLOCK  = True       
ENABLE_PHYSICAL_THREADS     = True       
THREAD_PITCH                = 0.7        
THREAD_DEPTH                = 0.4        
THREAD_TAP_RADIUS           = 1.65       

SLIT_SLOT_WIDTH             = 25.4       
SLIT_SLOT_HEIGHT            = 25.4       
SLIT_SLOT_THICKNESS         = 2.0        

CORNER_CUTOUT_SIZE          = 40.0       
OPTICS_POCKET_RADIUS        = 12.7       
AIRGAP_RADIUS_REDUCTION     = 1.0        
DRILL_APERTURE_RADIUS       = 11.7       
DRILL_EXIT_LENGTH           = 30.0       

SCREW_CLEARANCE_RADIUS      = 2.25       
SCREW_TAP_RADIUS            = 1.65       
SCREW_HOLE_LENGTH           = 25.0       

PEG_RADIUS                  = 2.5        
PEG_HEIGHT                  = 3.0        
PEG_HOLE_RADIUS             = 2.7        
PEG_HOLE_DEPTH              = 3.5        

DEFAULT_TOP_STEP_OUTPUT     = "support_top_half.step"
DEFAULT_BOTTOM_STEP_OUTPUT  = "support_bottom_half.step"


def get_base_dir():
    if '__file__' in globals(): return os.path.dirname(os.path.abspath(__file__))
    return os.getcwd()


def get_bounding_cylinder(shape, placement, radius=12.7):
    bbox = shape.BoundBox
    length = bbox.ZMax - bbox.ZMin
    cyl = Part.makeCylinder(radius, length, FreeCAD.Vector(0, 0, bbox.ZMin), FreeCAD.Vector(0, 0, 1))
    cyl.transformShape(placement.Matrix)
    return cyl


def build_airgap_cylinder(shape, radius_reduction=1.0, extension=1.0):
    axis, center, radius = None, None, None
    for face in shape.Faces:
        if type(face.Surface) == Part.Cylinder:
            axis = face.Surface.Axis
            center = face.Surface.Center
            radius = face.Surface.Radius
            break
    if not (axis and center and radius):
        return None, None, None, None

    t_min, t_max = float('inf'), float('-inf')
    for v in shape.Vertexes:
        t = (v.Point - center).dot(axis)
        t_min, t_max = min(t_min, t), max(t_max, t)

    start_pt = center + axis * t_min
    length = t_max - t_min
    ext_start = start_pt - axis * (extension / 2.0)
    ext_length = length + extension
    new_radius = radius - radius_reduction
    cyl = Part.makeCylinder(new_radius, ext_length, ext_start, axis)
    return cyl, new_radius, center + axis * t_max, axis


def load_support_geometry(doc, support_name):
    obj = doc.getObject(support_name)
    if not obj and support_name == DEFAULT_SUPPORT_OBJECT:
        for name in ("Sketch", "Body"):
            obj = doc.getObject(name)
            if obj: break
    if not obj: raise RuntimeError(f"Support object '{support_name}' not found in document.")

    if hasattr(obj, 'Shape') and not obj.Shape.isNull():
        if len(obj.Shape.Wires) > 0:
            return obj.Shape.Wires[0], Part.Face(obj.Shape.Wires[0])
        elif len(obj.Shape.Faces) > 0:
            for f in obj.Shape.Faces:
                if abs(f.BoundBox.XMin) < 1e-3 and abs(f.BoundBox.XMax) < 1e-3:
                    return f.Wires[0], f
            return obj.Shape.Faces[0].Wires[0], obj.Shape.Faces[0]
    raise RuntimeError("Failed to extract valid planar wire/face from support object.")


def auto_compute_fasteners(face, wire, bbox, optics_bboxes, corner_size=40.0):
    safe_points = []
    y_min, y_max = int(bbox.YMin), int(bbox.YMax)
    z_min, z_max = int(bbox.ZMin), int(bbox.ZMax)

    for y in range(y_min + 3, y_max - 3, 2):
        for z in range(z_min + 5, z_max - 5, 2):
            if y >= (bbox.YMax - corner_size - 1.0) and z >= (bbox.ZMax - corner_size - 1.0): continue
            v = Part.Vertex(FreeCAD.Vector(0, y, z))
            if face.distToShape(v)[0] > 0.01 or wire.distToShape(v)[0] < 3.0: continue
            safe = True
            for ob in optics_bboxes:
                dy = max(0.0, ob.YMin - y, y - ob.YMax)
                dz = max(0.0, ob.ZMin - z, z - ob.ZMax)
                if math.hypot(dy, dz) < 1.5:
                    safe = False; break
            if safe: safe_points.append((y, z))

    def pick_best(targets, avail):
        chosen = []
        for ty, tz in targets:
            best, min_d = None, float('inf')
            for p in avail:
                d = math.hypot(p[0] - ty, p[1] - tz)
                if d < min_d: min_d, best = d, p
            if best:
                chosen.append(best)
                avail = [p for p in avail if math.hypot(p[0] - best[0], p[1] - best[1]) > 10.0]
        return chosen, avail

    screw_targets = [(16, z_max - 52), (-55, z_max - 6), (19, 115), (-21, 115), (20, 8), (-16, 30)]
    chosen_screws, avail = pick_best(screw_targets, safe_points)
    peg_targets = [(19, 170), (-21, 196), (19, 65), (19, 35)]
    chosen_pegs, avail = pick_best(peg_targets, avail)
    return chosen_screws, chosen_pegs


def compute_safe_fastener_locations(face, wire, bbox, optics_bboxes, corner_size=40.0, manual_screws=None, manual_pegs=None):
    screws = manual_screws if manual_screws is not None else MANUAL_SCREW_POSITIONS
    pegs = manual_pegs if manual_pegs is not None else MANUAL_PEG_POSITIONS

    def verify(pts, radius):
        for y, z in pts:
            v = Part.Vertex(FreeCAD.Vector(0, y, z))
            if face.distToShape(v)[0] > 0.01 or wire.distToShape(v)[0] < (radius + 0.5) or \
               ((y + radius >= bbox.YMax - corner_size) and (z + radius >= bbox.ZMax - corner_size)): return False
            for ob in optics_bboxes:
                dy = max(0.0, ob.YMin - y, y - ob.YMax)
                dz = max(0.0, ob.ZMin - z, z - ob.ZMax)
                if math.hypot(dy, dz) < radius: return False
        return True

    if not verify(screws, SCREW_CLEARANCE_RADIUS) or not verify(pegs, PEG_HOLE_RADIUS):
        return auto_compute_fasteners(face, wire, bbox, optics_bboxes, corner_size)
    return list(screws), list(pegs)


def compute_breadboard_grid(bbox, screws, pegs, left_ext, right_ext, spacing, edge_margin):
    safe_bb = []
    y_start = (bbox.YMin - left_ext) + edge_margin
    y_end = (bbox.YMax + right_ext) - edge_margin
    z_start = max(14.36 + edge_margin, bbox.ZMin + edge_margin)
    z_end = bbox.ZMax - edge_margin

    y = y_start
    while y <= y_end + 1e-4:
        z = z_start
        while z <= z_end + 1e-4:
            if not any(math.hypot(y - sy, z - sz) < 5.95 for sy, sz in screws + pegs):
                safe_bb.append((round(y, 1), round(z, 1)))
            z += spacing
        y += spacing
    return safe_bb


def drill_lateral_holes(hole_list, dir_vec, rot_matrix, part_obj, optics_solid, verbose):
    for (x, y, z) in hole_list:
        chk_cyl = Part.makeCylinder(THREAD_TAP_RADIUS + 3.0, LATERAL_HOLE_DEPTH + 3.0, FreeCAD.Vector(x, y, z), dir_vec)
        if chk_cyl.common(optics_solid).Volume > 1e-4:
            if verbose: print(f"  [SKIPPED] Lateral hole ({x}, {y}, {z}) intersects optical path.")
            continue
        try:
            cyl = Part.makeCylinder(THREAD_TAP_RADIUS, LATERAL_HOLE_DEPTH, FreeCAD.Vector(x, y, z), dir_vec)
            part_obj = part_obj.cut(cyl)
            if ENABLE_PHYSICAL_THREADS:
                th = Part.makeThread(THREAD_PITCH, THREAD_DEPTH, LATERAL_HOLE_DEPTH, THREAD_TAP_RADIUS)
                th.transformShape(FreeCAD.Placement(FreeCAD.Vector(x, y, z), rot_matrix).Matrix)
                part_obj = part_obj.cut(th)
        except Exception as e:
            if verbose: print(f"  [THREAD WARNING] Lateral hole at ({x}, {y}, {z}) failed: {e}")
    return part_obj


def build_spectrometer(doc_path=None, support_name=DEFAULT_SUPPORT_OBJECT, optics_name=DEFAULT_OPTICS_ASSEMBLY,
                       thorlabs_step=DEFAULT_THORLABS_STEP, top_output=DEFAULT_TOP_STEP_OUTPUT, bottom_output=DEFAULT_BOTTOM_STEP_OUTPUT,
                       breadboard_thickness=BREADBOARD_THICKNESS, breadboard_left_extension=BREADBOARD_LEFT_EXTENSION,
                       breadboard_right_extension=BREADBOARD_RIGHT_EXTENSION, breadboard_spacing=BREADBOARD_GRID_SPACING,
                       corner_size=CORNER_CUTOUT_SIZE, enable_threads=ENABLE_PHYSICAL_THREADS,
                       manual_screws=None, manual_pegs=None, manual_bb=None, auto_bb=False, verbose=True):
                       
    base_dir = get_base_dir()
    doc_path = os.path.abspath(doc_path or os.path.join(base_dir, DEFAULT_INPUT_DOCUMENT))
    thorlabs_step = os.path.abspath(thorlabs_step or os.path.join(base_dir, DEFAULT_THORLABS_STEP))
    top_output = os.path.abspath(top_output or os.path.join(base_dir, DEFAULT_TOP_STEP_OUTPUT))
    bottom_output = os.path.abspath(bottom_output or os.path.join(base_dir, DEFAULT_BOTTOM_STEP_OUTPUT))

    if verbose:
        print("=" * 90 + "\n        CONSOLIDATED FREECAD SPECTROMETER CAD GENERATOR\n" + "=" * 90)
        print(f"Source Document       : {doc_path}\nSupport Object        : {support_name}\nOptics Assembly       : {optics_name}")
        print(f"Thorlabs Lens Step    : {thorlabs_step}\nTop Output STEP       : {top_output}\nBottom Output STEP    : {bottom_output}")
        print(f"Breadboard Thickness  : {breadboard_thickness} mm\nBreadboard Flange Ext : {breadboard_left_extension} mm")
        print(f"Breadboard Spacing    : {breadboard_spacing} mm\nCorner Cutout Size    : {corner_size} x {corner_size} mm")
        print(f"Physical Threads      : {'ENABLED' if enable_threads else 'DISABLED'}\n" + "=" * 90)
        sys.stdout.flush()

    if not os.path.exists(doc_path): raise FileNotFoundError(f"CAD document not found: {doc_path}")
    if not os.path.exists(thorlabs_step): raise FileNotFoundError(f"Thorlabs CAD file not found: {thorlabs_step}")

    doc = FreeCAD.open(doc_path)
    wire, face = load_support_geometry(doc, support_name)
    bbox = face.BoundBox
    spectrometer = doc.getObject(optics_name)
    if not spectrometer: raise RuntimeError(f"Optical assembly '{optics_name}' not found.")

    optics_bboxes = [o.Shape.BoundBox for o in spectrometer.Group if hasattr(o, 'Shape') and not o.Shape.isNull()]

    # 1. Base Clamshell Extrusions
    if verbose: print("\n[STEP 1/7] Extruding clamshell halves...")
    top_half = face.extrude(FreeCAD.Vector(CLAMSHELL_HALF_THICKNESS, 0, 0))
    bottom_half = face.extrude(FreeCAD.Vector(-CLAMSHELL_HALF_THICKNESS, 0, 0))

    # 1.5 Enclose DMD optical path
    if verbose: print("  -> Creating targeted housing for the DMD optical path...")
    dmd_housing_top = Part.makeBox(CLAMSHELL_HALF_THICKNESS, 65.0, 140.0, FreeCAD.Vector(0, -82.0, 95.0))
    dmd_housing_bot = Part.makeBox(CLAMSHELL_HALF_THICKNESS, 65.0, 140.0, FreeCAD.Vector(-CLAMSHELL_HALF_THICKNESS, -82.0, 95.0))
    top_half, bottom_half = top_half.fuse(dmd_housing_top), bottom_half.fuse(dmd_housing_bot)

    # Trim extreme left overhang
    left_trim_box = Part.makeBox(200, 100, 300, FreeCAD.Vector(-100, -180.0, -50))
    top_half, bottom_half = top_half.cut(left_trim_box), bottom_half.cut(left_trim_box)
    bbox = top_half.BoundBox

    if breadboard_right_extension > 0:
        top_shifted, bot_shifted = top_half.copy(), bottom_half.copy()
        top_shifted.translate(FreeCAD.Vector(0, breadboard_right_extension, 0))
        bot_shifted.translate(FreeCAD.Vector(0, breadboard_right_extension, 0))
        top_half, bottom_half = top_half.fuse(top_shifted), bottom_half.fuse(bot_shifted)
        bbox = top_half.BoundBox

    # 2. Add Breadboard Slab
    if verbose: print(f"[STEP 2/7] Adding {breadboard_thickness}mm breadboard slab...")
    bb_x_min = -(CLAMSHELL_HALF_THICKNESS + breadboard_thickness)
    bb_slab = Part.makeBox(breadboard_thickness, bbox.YMax - bbox.YMin, bbox.ZMax - bbox.ZMin, FreeCAD.Vector(bb_x_min, bbox.YMin, bbox.ZMin))
    bottom_half = bottom_half.fuse(bb_slab)

    # 2.5 Corners & Entrance
    if verbose: print("[STEP 2.5/7] Rounding corners and opening entrance tunnel...")
    for x in [CLAMSHELL_HALF_THICKNESS, -CLAMSHELL_HALF_THICKNESS]:
        for y in [25.0, -80.0]:
            sign_x, sign_y = (-1 if x > 0 else 1), (-1 if y > 0 else 1)
            box = Part.makeBox(5.0, 5.0, 300.0, FreeCAD.Vector(x + (0 if sign_x==1 else -5.0), y + (0 if sign_y==1 else -5.0), -10))
            cyl = Part.makeCylinder(5.0, 300.0, FreeCAD.Vector(x + sign_x*5.0, y + sign_y*5.0, -10), FreeCAD.Vector(0,0,1))
            fillet_tool = box.cut(cyl)
            if x > 0: top_half = top_half.cut(fillet_tool)
            else: bottom_half = bottom_half.cut(fillet_tool)

    entrance_tunnel = Part.makeCylinder(13.0, 20.0, FreeCAD.Vector(0, 0, -2.0), FreeCAD.Vector(0, 0, 1))
    top_half, bottom_half = top_half.cut(entrance_tunnel), bottom_half.cut(entrance_tunnel)

    # 2.8 Sensor Trim & Taper
    sensor = next((obj for obj in doc.getObject(optics_name).Group if hasattr(obj, 'Shape') and not obj.Shape.isNull() and "23_S23" in obj.Label), None)
    if sensor:
        if verbose: print("[STEP 2.8/7] Trimming top exit sensor area...")
        top_pad = Part.makeBox(CLAMSHELL_HALF_THICKNESS, 85.0, 15.0, FreeCAD.Vector(0, -100.0, 233.0))
        bot_pad = Part.makeBox(CLAMSHELL_HALF_THICKNESS, 85.0, 15.0, FreeCAD.Vector(-CLAMSHELL_HALF_THICKNESS, -100.0, 233.0))
        top_half, bottom_half = top_half.fuse(top_pad), bottom_half.fuse(bot_pad)
        
        sensor_z = sensor.Placement.Rotation.multVec(FreeCAD.Vector(0,0,1))
        sensor_cut_box = Part.makeBox(200, 200, 200, FreeCAD.Vector(-100, -100, 0))
        sensor_cut_box.transformShape(FreeCAD.Placement(sensor.Placement.Base + sensor_z * 2.0, sensor.Placement.Rotation).Matrix)
        top_half, bottom_half = top_half.cut(sensor_cut_box), bottom_half.cut(sensor_cut_box)

    poly_tl = Part.makePolygon([
        FreeCAD.Vector(0, -125, 140), FreeCAD.Vector(0, -85, 210), FreeCAD.Vector(0, -35, 245),
        FreeCAD.Vector(0, -35, 300), FreeCAD.Vector(0, -125, 300), FreeCAD.Vector(0, -125, 140)
    ])
    taper_tl = Part.Face(poly_tl).extrude(FreeCAD.Vector(100, 0, 0))
    taper_tl.translate(FreeCAD.Vector(-50, 0, 0))
    top_half, bottom_half = top_half.cut(taper_tl), bottom_half.cut(taper_tl)
    bbox = top_half.BoundBox

    # 2.9 DMD Window
    dmd = next((obj for obj in doc.getObject(optics_name).Group if hasattr(obj, 'Shape') and not obj.Shape.isNull() and "22_CB_After_DMD_Hardware" in obj.Label), None)
    if dmd:
        if verbose: print("[STEP 2.9/7] Exposing DMD window...")
        dmd_z = dmd.Placement.Rotation.multVec(FreeCAD.Vector(0,0,1))
        dmd_window_cut = Part.makeBox(45.0, 45.0, 100.0, FreeCAD.Vector(-22.5, -22.5, -100.0))
        dmd_window_cut.transformShape(FreeCAD.Placement(dmd.Placement.Base - dmd_z, dmd.Placement.Rotation).Matrix)
        top_half, bottom_half = top_half.cut(dmd_window_cut), bottom_half.cut(dmd_window_cut)
        bbox = top_half.BoundBox

    # 3.5 Interlocking Perimeter
    if ENABLE_PERIMETER_INTERLOCK:
        if verbose: print("[STEP 3.5/7] Constructing perimeter interlock...")
        x0_faces = [f for f in top_half.Faces if abs(f.BoundBox.XMin) < 1e-4 and abs(f.BoundBox.XMax) < 1e-4]
        if x0_faces:
            x0_face = x0_faces[0]
            for f in x0_faces[1:]: x0_face = x0_face.fuse(f)
            outer_wire = sorted(x0_face.Wires, key=lambda w: w.Length, reverse=True)[0]
            try:
                tongue_face = Part.Face(outer_wire.makeOffset2D(-0.8).Wires[0]).cut(Part.Face(outer_wire.makeOffset2D(-2.4).Wires[0]))
                bottom_half = bottom_half.fuse(tongue_face.extrude(FreeCAD.Vector(3.0, 0, 0)))
                groove_face = Part.Face(outer_wire.makeOffset2D(-0.6).Wires[0]).cut(Part.Face(outer_wire.makeOffset2D(-2.6).Wires[0]))
                top_half = top_half.cut(groove_face.extrude(FreeCAD.Vector(3.2, 0, 0)))
            except Exception as e:
                if verbose: print(f"  [WARNING] Perimeter interlock failed: {e}")

    # 4. Optical Subtractions
    if verbose: print("[STEP 4/7] Optical subtractions...")
    thorlabs_lens = Part.Shape()
    thorlabs_lens.read(thorlabs_step)
    thorlabs_lens = thorlabs_lens.Solids[0]
    thorlabs_lens.transformShape(FreeCAD.Placement(FreeCAD.Vector(), FreeCAD.Rotation(FreeCAD.Vector(1, 0, 0), 90)).Matrix)

    shapes_to_cut, first_placement, last_airgap_axis, last_airgap_end = [], None, None, None
    for obj in spectrometer.Group:
        if not hasattr(obj, 'Shape') or obj.Shape.isNull(): continue
        low = obj.Label.lower()
        if "glass" in low:
            cyl = get_bounding_cylinder(thorlabs_lens if "acl25416u" in low else obj.Shape.copy().transformShape(obj.Placement.Matrix.inverse()), obj.Placement, OPTICS_POCKET_RADIUS)
            shapes_to_cut.append((obj.Label, cyl))
            if "01_collect_acl25416u" in low: first_placement = obj.Placement
        elif "airgap" in low:
            cyl, _, end_pt, axis = build_airgap_cylinder(obj.Shape, radius_reduction=AIRGAP_RADIUS_REDUCTION, extension=1.0)
            shapes_to_cut.append((obj.Label, cyl if cyl else obj.Shape))
            if "18_condenser" in low: last_airgap_axis, last_airgap_end = axis, end_pt
        elif "hardware" in low:
            if "19_input_slit" in low:
                shapes_to_cut.append((obj.Label, Part.makeBox(SLIT_SLOT_WIDTH, SLIT_SLOT_HEIGHT, SLIT_SLOT_THICKNESS, FreeCAD.Vector(-SLIT_SLOT_WIDTH/2, -SLIT_SLOT_HEIGHT/2, obj.Placement.Base.z - SLIT_SLOT_THICKNESS/2))))
            elif "20_grating" in low:
                gr_box = Part.makeBox(25.0, 25.0, 6.0, FreeCAD.Vector(-12.5, -12.5, -3.0))
                gr_box.translate(FreeCAD.Vector(0, 5.0, 0))
                gr_box.transformShape(obj.Placement.Matrix)
                shapes_to_cut.append((obj.Label, gr_box))
            else:
                shapes_to_cut.append((obj.Label, obj.Shape))
        else:
            shapes_to_cut.append((obj.Label, obj.Shape))

    if first_placement:
        shapes_to_cut.append(("Front Drill", Part.makeCylinder(DRILL_APERTURE_RADIUS, first_placement.Base.z, FreeCAD.Vector(), FreeCAD.Vector(0, 0, 1))))
    if last_airgap_axis and last_airgap_end:
        shapes_to_cut.append(("Back Drill", Part.makeCylinder(DRILL_APERTURE_RADIUS, DRILL_EXIT_LENGTH, last_airgap_end, last_airgap_axis)))

    all_optics_solid = shapes_to_cut[0][1]
    success_cuts = 0
    for i, (name, s) in enumerate(shapes_to_cut):
        try:
            if i > 0: all_optics_solid = all_optics_solid.fuse(s)
            t, b = top_half.cut(s), bottom_half.cut(s)
            if not t.isNull() and t.Volume > 0: top_half = t
            if not b.isNull() and b.Volume > 0: bottom_half = b
            success_cuts += 1
        except Exception: pass
    if verbose: print(f"  Applied {success_cuts}/{len(shapes_to_cut)} subtractions.")

    # 5. Fasteners
    if verbose: print("[STEP 5/7] Fasteners & Pegs...")
    screws, pegs = compute_safe_fastener_locations(face, wire, bbox, optics_bboxes, corner_size, manual_screws, manual_pegs)
    for (y, z) in pegs:
        bottom_half = bottom_half.fuse(Part.makeCylinder(PEG_RADIUS, PEG_HEIGHT, FreeCAD.Vector(0, y, z), FreeCAD.Vector(1, 0, 0)))
        top_half = top_half.cut(Part.makeCylinder(PEG_HOLE_RADIUS, PEG_HOLE_DEPTH, FreeCAD.Vector(0, y, z), FreeCAD.Vector(1, 0, 0)))
    for (y, z) in screws:
        top_half = top_half.cut(Part.makeCylinder(SCREW_CLEARANCE_RADIUS, SCREW_HOLE_LENGTH, FreeCAD.Vector(0, y, z), FreeCAD.Vector(1, 0, 0)))
        bottom_half = bottom_half.cut(Part.makeCylinder(SCREW_TAP_RADIUS, SCREW_HOLE_LENGTH, FreeCAD.Vector(bb_x_min - 1.0, y, z), FreeCAD.Vector(1, 0, 0)))

    # 6. Breadboard
    if verbose: print("[STEP 6/7] Breadboard Tap Holes...")
    safe_bb = list(manual_bb) if manual_bb else compute_breadboard_grid(bbox, screws, pegs, breadboard_left_extension, breadboard_right_extension, breadboard_spacing, BREADBOARD_EDGE_MARGIN)
    bb_cyls = [Part.makeCylinder(SCREW_TAP_RADIUS, breadboard_thickness + 0.5, FreeCAD.Vector(bb_x_min - 0.2, y, z), FreeCAD.Vector(1, 0, 0)) for (y, z) in safe_bb]
    if bb_cyls:
        try:
            bottom_half = bottom_half.cut(Part.makeCompound(bb_cyls))
        except:
            all_bb = bb_cyls[0]
            for c in bb_cyls[1:]: all_bb = all_bb.fuse(c)
            bottom_half = bottom_half.cut(all_bb)
    
    if enable_threads:
        rot_x = FreeCAD.Rotation(FreeCAD.Vector(0, 1, 0), 90)
        for (y, z) in screws:
            try:
                th = Part.makeThread(THREAD_PITCH, THREAD_DEPTH, 20.0, THREAD_TAP_RADIUS)
                th.transformShape(FreeCAD.Placement(FreeCAD.Vector(bb_x_min, y, z), rot_x).Matrix)
                bottom_half = bottom_half.cut(th)
            except: pass

    # 6.5 Lateral M4 Threads
    if verbose: print("[STEP 6.5/7] Lateral M4 Threads...")
    vX, vY, vZ = FreeCAD.Vector(1,0,0), FreeCAD.Vector(0,1,0), FreeCAD.Vector(0,0,1)
    rot_pos_y, rot_neg_y, rot_diag, rot_r_taper, rot_l_taper, rot_top = [FreeCAD.Rotation(vX, a) for a in [90, -90, -43.53, 68.7, -76.8, 180]]
    
    top_half = drill_lateral_holes([h for h in LATERAL_HOLES_RIGHT if h[0]>0], -vY, rot_pos_y, top_half, all_optics_solid, verbose)
    bottom_half = drill_lateral_holes([h for h in LATERAL_HOLES_RIGHT if h[0]<0], -vY, rot_pos_y, bottom_half, all_optics_solid, verbose)
    for h_list in [LATERAL_HOLES_LEFT_1, LATERAL_HOLES_LEFT_2]:
        top_half = drill_lateral_holes([h for h in h_list if h[0]>0], vY, rot_neg_y, top_half, all_optics_solid, verbose)
        bottom_half = drill_lateral_holes([h for h in h_list if h[0]<0], vY, rot_neg_y, bottom_half, all_optics_solid, verbose)
    
    diag_dir, r_taper_dir, l_taper_dir = FreeCAD.Vector(0, 0.689, 0.725), FreeCAD.Vector(0, -0.932, 0.363), FreeCAD.Vector(0, 0.974, 0.228)
    top_half = drill_lateral_holes([h for h in LATERAL_HOLES_DIAGONAL if h[0]>0], diag_dir, rot_diag, top_half, all_optics_solid, verbose)
    bottom_half = drill_lateral_holes([h for h in LATERAL_HOLES_DIAGONAL if h[0]<0], diag_dir, rot_diag, bottom_half, all_optics_solid, verbose)
    top_half = drill_lateral_holes([h for h in LATERAL_HOLES_RIGHT_TAPER if h[0]>0], r_taper_dir, rot_r_taper, top_half, all_optics_solid, verbose)
    bottom_half = drill_lateral_holes([h for h in LATERAL_HOLES_RIGHT_TAPER if h[0]<0], r_taper_dir, rot_r_taper, bottom_half, all_optics_solid, verbose)
    top_half = drill_lateral_holes([h for h in LATERAL_HOLES_LEFT_TAPER if h[0]>0], l_taper_dir, rot_l_taper, top_half, all_optics_solid, verbose)
    bottom_half = drill_lateral_holes([h for h in LATERAL_HOLES_LEFT_TAPER if h[0]<0], l_taper_dir, rot_l_taper, bottom_half, all_optics_solid, verbose)
    top_half = drill_lateral_holes([h for h in LATERAL_HOLES_TOP if h[0]>0], -vZ, rot_top, top_half, all_optics_solid, verbose)
    bottom_half = drill_lateral_holes([h for h in LATERAL_HOLES_TOP if h[0]<0], -vZ, rot_top, bottom_half, all_optics_solid, verbose)

    # 6.8 & 6.9 Sensor & DMD Mount Holes
    def apply_mount_holes(obj, spacing, depth, z_offset, target_z, reverse_dir=False):
        if not obj: return
        half = spacing / 2.0
        z_vec = obj.Placement.Rotation.multVec(vZ)
        for dx, dy in [(-half, -half), (-half, half), (half, -half), (half, half)]:
            pos = obj.Placement.Base + z_vec * z_offset + obj.Placement.Rotation.multVec(FreeCAD.Vector(dx, dy, 0))
            cyl = Part.makeCylinder(SCREW_TAP_RADIUS, depth, pos, target_z * (-1.0 if reverse_dir else 1.0))
            if pos.x > 0: nonlocal top_half; top_half = top_half.cut(cyl)
            else: nonlocal bottom_half; bottom_half = bottom_half.cut(cyl)
            
    if sensor: apply_mount_holes(sensor, SENSOR_M4_HOLES_SPACING, SENSOR_M4_HOLE_DEPTH, 3.0, sensor.Placement.Rotation.multVec(vZ), True)
    if dmd: apply_mount_holes(dmd, DMD_M4_HOLES_SPACING, DMD_M4_HOLE_DEPTH, -2.0, dmd.Placement.Rotation.multVec(vZ))

    # 7. Validate & Export
    if verbose: print(f"[STEP 7/7] Validating & Exporting...\n  Top Vol: {top_half.Volume:.1f}  Bot Vol: {bottom_half.Volume:.1f}")
    if top_half.isNull() or not top_half.isValid() or len(top_half.Solids) != 1: raise RuntimeError(f"Top half CAD failed (solids count: {len(top_half.Solids) if not top_half.isNull() else 0})")
    if bottom_half.isNull() or not bottom_half.isValid() or len(bottom_half.Solids) != 1: raise RuntimeError(f"Bottom half CAD failed")

    for p in [top_output, bottom_output]: os.makedirs(os.path.dirname(p) or '.', exist_ok=True)
    top_half.exportStep(top_output); bottom_half.exportStep(bottom_output)
    if verbose: print("=" * 90 + "\n                 BUILD SPECTROMETER COMPLETE: SUCCESS\n" + "=" * 90)

    return {"top_volume": top_half.Volume, "bottom_volume": bottom_half.Volume}


def is_main_entry(script_name="build_spectrometer_refactored.py"):
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

if is_main_entry("build_spectrometer_refactored.py"):
    import shlex
    argv = sys.argv
    clean_args = []
    for arg in argv:
        if arg in ("--", "--pass"): continue
        if arg.lower().endswith("exe") or arg.endswith(".py"): continue
        if " " in arg and not arg.startswith('"'):
            try: clean_args.extend(shlex.split(arg))
            except: clean_args.append(arg)
        else:
            clean_args.append(arg)

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-doc", "-i", default=DEFAULT_INPUT_DOCUMENT)
    parser.add_argument("--support-obj", default=DEFAULT_SUPPORT_OBJECT)
    parser.add_argument("--optics-asm", default=DEFAULT_OPTICS_ASSEMBLY)
    parser.add_argument("--thorlabs-step", default=DEFAULT_THORLABS_STEP)
    parser.add_argument("--top-output", "-t", default=DEFAULT_TOP_STEP_OUTPUT)
    parser.add_argument("--bottom-output", "-b", default=DEFAULT_BOTTOM_STEP_OUTPUT)
    parser.add_argument("--breadboard-thickness", type=float, default=BREADBOARD_THICKNESS)
    parser.add_argument("--breadboard-left-ext", type=float, default=BREADBOARD_LEFT_EXTENSION)
    parser.add_argument("--breadboard-right-ext", type=float, default=BREADBOARD_RIGHT_EXTENSION)
    parser.add_argument("--breadboard-spacing", type=float, default=BREADBOARD_GRID_SPACING)
    parser.add_argument("--corner-size", type=float, default=CORNER_CUTOUT_SIZE)
    parser.add_argument("--threads", action="store_true", default=ENABLE_PHYSICAL_THREADS)
    parser.add_argument("--no-threads", action="store_false", dest="threads")
    parser.add_argument("--auto-bb", action="store_true", default=False)
    parser.add_argument("-q", "--quiet", action="store_true")
    
    args = parser.parse_args(clean_args)
    try:
        
        kwargs = vars(args).copy()
        kwargs['doc_path'] = kwargs.pop('input_doc'); kwargs['support_name'] = kwargs.pop('support_obj'); kwargs['optics_name'] = kwargs.pop('optics_asm')
        kwargs['breadboard_left_extension'] = kwargs.pop('breadboard_left_ext')
        kwargs['breadboard_right_extension'] = kwargs.pop('breadboard_right_ext')
        kwargs['enable_threads'] = kwargs.pop('threads')
        kwargs['verbose'] = not kwargs.pop('quiet')
        build_spectrometer(**kwargs)
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}", file=sys.stderr)
        import traceback; traceback.print_exc()
        sys.exit(1)


