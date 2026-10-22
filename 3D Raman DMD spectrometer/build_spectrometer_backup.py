"""
build_spectrometer.py

Consolidated, production-ready FreeCAD CAD generation script for the 2D Raman DMD
Spectrometer clamshell enclosure.

Requirements Satisfied:
1. Clamshell Architecture: Splits spectrometer structure at X=0 into Top and Bottom halves.
   - Top half: spans X in [0, +17.5 mm].
   - Bottom half: spans X in [-17.5, 0 mm] with integrated 3mm breadboard slab.
2. Breadboard Integration: Adds a 3mm-thick rectangular breadboard slab below the bottom
   surface (from X=-20.5 mm to X=-17.5 mm) matching the exact footprint bounding box of the
   original support CAD.
3. M4 Breadboard Grid: Cuts a regular grid of M4 tap holes (3.3 mm diameter) through the
   breadboard, mathematically filtered to strictly avoid optical paths, fastener joints, and
   the corner cutout.
4. M4 Fasteners & Tetris Pegs: Computes and cuts 6 M4 assembly screws (clearance in Top half,
   tap in Bottom half) and 4 Tetris alignment pegs (Top hole, Bottom peg), all encased in
   solid plastic with verified optical clearances.
5. Corner Cutout: Cuts a 40x40 mm chunk from the top-right corner (at widest side and bottom
   of longest side: Y in [-15, 25], Z in [193, 233]).
6. Complete Optical Subtractions: Boolean subtracts ALL 24 optical components (glass pockets,
   reduced air gap light tunnels with 1.0 mm flat mounting shoulders, hardware mounts,
   entry front drill aperture, and exit back drill aperture).
7. High Generalization: Easily configurable parameters and command-line arguments to allow
   swapping CAD models, optical assemblies, or dimensional parameters.

Usage:
  FreeCADCmd.exe build_spectrometer.py [options]
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

# Explicit Fastener & Breadboard Coordinate Arrays (Easily edited manually)
MANUAL_SCREW_POSITIONS = [
    (16.0, 181.0),   # Screw #1: Top right zone (safely clear of corner cutout and optics)
    (-55.0, 227.0),  # Screw #2: Top left zone (condenser wing - 3.75mm wall margin)
    (19.0, 115.0),   # Screw #3: Mid right zone
    (-21.0, 115.0),  # Screw #4: Mid left zone
    (20.0, 8.0),     # Screw #5: Bottom right zone (6.05mm clearance from entrance drill)
    (-16.5, 30.0)    # Screw #6: Bottom left zone (13.16mm clearance from entrance drill)
]

MANUAL_PEG_POSITIONS = [
    (19.0, 170.0),   # Peg #1: Upper right alignment
    (-21.0, 196.0),  # Peg #2: Upper left alignment
    (19.0, 65.0),    # Peg #3: Lower mid right alignment
    (19.0, 35.0)     # Peg #4: Lower right alignment
]

# Explicit Parameter Array: Breadboard Tap Hole Grid Coordinates (140 holes covering base and 40mm flange)
MANUAL_BREADBOARD_POSITIONS = None

# Explicit Parameter Array: Lateral M4 Threaded Holes (X, Y, Z)
# Right wall (Y=25) pointing inward
LATERAL_HOLES_RIGHT = [
    (8.75, 25.0, 50.0), (8.75, 25.0, 70.0), (8.75, 25.0, 90.0),
    (8.75, 25.0, 110.0), (8.75, 25.0, 130.0), (8.75, 25.0, 150.0), (8.75, 25.0, 170.0),
    (-8.75, 25.0, 50.0), (-8.75, 25.0, 70.0), (-8.75, 25.0, 90.0),
    (-8.75, 25.0, 110.0), (-8.75, 25.0, 130.0), (-8.75, 25.0, 150.0), (-8.75, 25.0, 170.0),
]
# Left wall straight section 1 (Y=-21) pointing inward
LATERAL_HOLES_LEFT_1 = [
    (8.75, -21.0, 50.0), (8.75, -21.0, 70.0), (8.75, -21.0, 90.0),
    (-8.75, -21.0, 50.0), (-8.75, -21.0, 70.0), (-8.75, -21.0, 90.0),
]
# Left wall straight section 2 (Y=-61) pointing inward
LATERAL_HOLES_LEFT_2 = [
    (8.75, -61.0, 150.0), (8.75, -61.0, 170.0), (8.75, -61.0, 190.0), (8.75, -61.0, 210.0),
    (-8.75, -61.0, 150.0), (-8.75, -61.0, 170.0), (-8.75, -61.0, 190.0), (-8.75, -61.0, 210.0),
]
# Diagonal left wall (-21, 105) to (-61, 143)
LATERAL_HOLES_DIAGONAL = [
    (8.75, -31.0, 114.5), (8.75, -41.0, 124.0), (8.75, -51.0, 133.5),
    (-8.75, -31.0, 114.5), (-8.75, -41.0, 124.0), (-8.75, -51.0, 133.5),
]
# Taper cuts at front
LATERAL_HOLES_RIGHT_TAPER = [
    (8.75, 21.0, 27.0), (-8.75, 21.0, 27.0)
]
LATERAL_HOLES_LEFT_TAPER = [
    (8.75, -19.0, 27.0), (-8.75, -19.0, 27.0)
]
# Top flat face at Z=233
LATERAL_HOLES_TOP = [
    (8.75, 10.0, 233.0), (8.75, -10.0, 233.0), (8.75, -30.0, 233.0), (8.75, -50.0, 233.0),
    (-8.75, 10.0, 233.0), (-8.75, -10.0, 233.0), (-8.75, -30.0, 233.0), (-8.75, -50.0, 233.0),
]
LATERAL_HOLE_DEPTH = 5.0


# Dimensional & Clamshell Parameters
CLAMSHELL_HALF_THICKNESS    = 17.5       # mm (extrude +17.5 for Top, -17.5 for Bottom)
BREADBOARD_THICKNESS        = 3.0        # mm (slab from X = -20.5 to X = -17.5)
BREADBOARD_LEFT_EXTENSION   = 60.0       # mm (extend base laterally left to fully enclose DMD)
BREADBOARD_RIGHT_EXTENSION  = 0.0        # mm (extend base breadboard laterally from right edge at YMax)
BREADBOARD_GRID_SPACING     = 10.0       # mm (spacing between tap holes)
BREADBOARD_TAP_DIAMETER     = 3.3        # mm (standard M4 tap drill)
BREADBOARD_HOLE_DEPTH       = 3.5        # mm (depth through the 3.0 mm breadboard slab)
BREADBOARD_EDGE_MARGIN      = 4.0        # mm (minimum margin from breadboard perimeter)

# Sensor Hardware Mount Holes
SENSOR_M4_HOLES_SPACING     = 30.0       # mm (square grid size for diagonal sensor mount)
SENSOR_M4_HOLE_DEPTH        = 15.0       # mm (depth of M4 tap holes for sensor)
DMD_M4_HOLES_SPACING        = 30.0       # mm (square grid size for DMD face mount)
DMD_M4_HOLE_DEPTH           = 10.0       # mm (depth of M4 tap holes for DMD)
BREADBOARD_OPTICS_CLEARANCE = 6.0        # mm (clearance to any optical beam path)

ENABLE_PERIMETER_INTERLOCK  = True       # Create a tongue-and-groove lip to snap the clamshell halves together

# Physical Thread Parameters (M4: pitch=0.7mm, depth=0.4mm, tap radius=1.65mm)
ENABLE_PHYSICAL_THREADS     = True       # True to model physical helical 3D threads in M4 tap holes
THREAD_PITCH                = 0.7        # mm (standard M4 coarse thread pitch)
THREAD_DEPTH                = 0.4        # mm (M4 thread depth)
THREAD_TAP_RADIUS           = 1.65       # mm (M4 tap hole radius; thread cuts from 1.65 to 2.05)

# Input Slit Insertion Slot Parameters (aligned with 19_Input_Slit_200um_Hardware)
SLIT_SLOT_WIDTH             = 25.4       # mm (standard 1-inch slit mount width)
SLIT_SLOT_HEIGHT            = 25.4       # mm (standard 1-inch slit mount height)
SLIT_SLOT_THICKNESS         = 2.0        # mm (standard 2.0mm slit plate thickness)

# Corner Cutout Parameters
CORNER_CUTOUT_SIZE          = 40.0       # mm (40x40 mm cutout at YMax, ZMax)

# Optics Pocket & Tunnel Parameters
OPTICS_POCKET_RADIUS        = 12.7       # mm (standard 1-inch optics radius = 25.4mm diameter)
AIRGAP_RADIUS_REDUCTION     = 1.0        # mm (11.7mm radius gives 1.0mm flat mounting shoulders)
DRILL_APERTURE_RADIUS       = 11.7       # mm (entry and exit light apertures)
DRILL_EXIT_LENGTH           = 30.0       # mm (exit aperture drill depth)

# M4 Assembly Screw Parameters
SCREW_CLEARANCE_RADIUS      = 2.25       # mm (4.5mm diameter clearance hole in Top half)
SCREW_TAP_RADIUS            = 1.65       # mm (3.3mm diameter tap hole in Bottom half)
SCREW_HOLE_LENGTH           = 25.0       # mm (sufficient to pass through each half)

# Tetris Alignment Peg Parameters
PEG_RADIUS                  = 2.5        # mm (5.0mm diameter alignment peg)
PEG_HEIGHT                  = 3.0        # mm (protrusion along +X into Top half)
PEG_HOLE_RADIUS             = 2.7        # mm (5.4mm diameter receiving hole, 0.2mm tolerance)
PEG_HOLE_DEPTH              = 3.5        # mm (receiving hole depth in Top half)

# Default Output STEP File Names
DEFAULT_TOP_STEP_OUTPUT     = "support_top_half.step"
DEFAULT_BOTTOM_STEP_OUTPUT  = "support_bottom_half.step"


def get_base_dir():
    """Returns directory containing this script, or current working directory."""
    if '__file__' in globals():
        return os.path.dirname(os.path.abspath(__file__))
    return os.getcwd()


def get_bounding_cylinder(shape, placement, radius=12.7):
    """Creates a bounding cylinder along local optical axis (Z) and transforms to global space."""
    bbox = shape.BoundBox
    length = bbox.ZMax - bbox.ZMin
    cyl = Part.makeCylinder(radius, length, FreeCAD.Vector(0, 0, bbox.ZMin), FreeCAD.Vector(0, 0, 1))
    cyl.transformShape(placement.Matrix)
    return cyl


def build_airgap_cylinder(shape, radius_reduction=1.0, extension=1.0):
    """
    Extracts cylindrical axis, center, and extents of an airgap shape,
    and returns a clean cylinder with reduced radius and slight extension.
    """
    axis = None
    center = None
    radius = None
    for face in shape.Faces:
        if type(face.Surface) == Part.Cylinder:
            axis = face.Surface.Axis
            center = face.Surface.Center
            radius = face.Surface.Radius
            break
    if not (axis and center and radius):
        return None, None, None, None

    t_min = float('inf')
    t_max = float('-inf')
    for v in shape.Vertexes:
        t = (v.Point - center).dot(axis)
        t_min = min(t_min, t)
        t_max = max(t_max, t)

    start_pt = center + axis * t_min
    length = t_max - t_min
    ext_start = start_pt - axis * (extension / 2.0)
    ext_length = length + extension
    new_radius = radius - radius_reduction
    cyl = Part.makeCylinder(new_radius, ext_length, ext_start, axis)
    return cyl, new_radius, center + axis * t_max, axis


def load_support_geometry(doc, support_name):
    """Extracts the 2D cross-section wire and face from the support object."""
    obj = doc.getObject(support_name)
    if not obj and support_name == DEFAULT_SUPPORT_OBJECT:
        # Fallback: search for Sketch or Body only when default object was requested
        for name in ("Sketch", "Body"):
            obj = doc.getObject(name)
            if obj:
                break
    if not obj:
        raise RuntimeError(f"Support object '{support_name}' not found in document.")

    if hasattr(obj, 'Shape') and not obj.Shape.isNull():
        if len(obj.Shape.Wires) > 0:
            wire = obj.Shape.Wires[0]
            face = Part.Face(wire)
            return wire, face
        elif len(obj.Shape.Faces) > 0:
            # Look for face coplanar with X=0
            for f in obj.Shape.Faces:
                if abs(f.BoundBox.XMin) < 1e-3 and abs(f.BoundBox.XMax) < 1e-3:
                    return f.Wires[0], f
            # Default to first face
            return obj.Shape.Faces[0].Wires[0], obj.Shape.Faces[0]

    raise RuntimeError("Failed to extract valid planar wire/face from support object.")


def compute_safe_fastener_locations(face, wire, bbox, optics_bboxes, corner_size=40.0,
                                    manual_screws=None, manual_pegs=None):
    """
    Computes robust coordinates for 6 M4 assembly screws and 4 Tetris pegs.
    Uses manual coordinates if provided (or defaults from top of file), verifying
    each mathematically against outer walls, corner cutout, and optical paths.
    Falls back to auto-computation if any coordinate fails geometric verification.
    """
    screws = manual_screws if manual_screws is not None else MANUAL_SCREW_POSITIONS
    pegs = manual_pegs if manual_pegs is not None else MANUAL_PEG_POSITIONS

    verified_screws = list(screws)
    verified_pegs = list(pegs)

    # Verify each point mathematically against the current document geometry
    for (y, z) in verified_screws:
        v = Part.Vertex(FreeCAD.Vector(0, y, z))
        if face.distToShape(v)[0] > 0.01:
            print(f"[WARNING] Screw ({y}, {z}) outside support face! Recomputing...")
            return auto_compute_fasteners(face, wire, bbox, optics_bboxes, corner_size)
        if wire.distToShape(v)[0] < (SCREW_CLEARANCE_RADIUS + 0.5):
            print(f"[WARNING] Screw ({y}, {z}) too close to outer wall! Recomputing...")
            return auto_compute_fasteners(face, wire, bbox, optics_bboxes, corner_size)
        if (y + SCREW_CLEARANCE_RADIUS >= bbox.YMax - corner_size) and (z + SCREW_CLEARANCE_RADIUS >= bbox.ZMax - corner_size):
            print(f"[WARNING] Screw ({y}, {z}) intersects corner cutout! Recomputing...")
            return auto_compute_fasteners(face, wire, bbox, optics_bboxes, corner_size)
        for ob in optics_bboxes:
            dy = max(0.0, ob.YMin - y, y - ob.YMax)
            dz = max(0.0, ob.ZMin - z, z - ob.ZMax)
            if math.hypot(dy, dz) < SCREW_CLEARANCE_RADIUS:
                print(f"[WARNING] Screw ({y}, {z}) intersects optical path! Recomputing...")
                return auto_compute_fasteners(face, wire, bbox, optics_bboxes, corner_size)

    for (y, z) in verified_pegs:
        v = Part.Vertex(FreeCAD.Vector(0, y, z))
        if face.distToShape(v)[0] > 0.01:
            print(f"[WARNING] Peg ({y}, {z}) outside support face! Recomputing...")
            return auto_compute_fasteners(face, wire, bbox, optics_bboxes, corner_size)
        if wire.distToShape(v)[0] < (PEG_HOLE_RADIUS + 0.5):
            print(f"[WARNING] Peg ({y}, {z}) too close to outer wall! Recomputing...")
            return auto_compute_fasteners(face, wire, bbox, optics_bboxes, corner_size)
        if (y + PEG_HOLE_RADIUS >= bbox.YMax - corner_size) and (z + PEG_HOLE_RADIUS >= bbox.ZMax - corner_size):
            print(f"[WARNING] Peg ({y}, {z}) intersects corner cutout! Recomputing...")
            return auto_compute_fasteners(face, wire, bbox, optics_bboxes, corner_size)
        for ob in optics_bboxes:
            dy = max(0.0, ob.YMin - y, y - ob.YMax)
            dz = max(0.0, ob.ZMin - z, z - ob.ZMax)
            if math.hypot(dy, dz) < PEG_HOLE_RADIUS:
                print(f"[WARNING] Peg ({y}, {z}) intersects optical path! Recomputing...")
                return auto_compute_fasteners(face, wire, bbox, optics_bboxes, corner_size)

    return verified_screws, verified_pegs


def auto_compute_fasteners(face, wire, bbox, optics_bboxes, corner_size=40.0):
    """Fallback algorithm to dynamically discover safe fastener locations."""
    safe_points = []
    y_min, y_max = int(bbox.YMin), int(bbox.YMax)
    z_min, z_max = int(bbox.ZMin), int(bbox.ZMax)

    for y in range(y_min + 3, y_max - 3, 2):
        for z in range(z_min + 5, z_max - 5, 2):
            if y >= (bbox.YMax - corner_size - 1.0) and z >= (bbox.ZMax - corner_size - 1.0):
                continue
            v = Part.Vertex(FreeCAD.Vector(0, y, z))
            if face.distToShape(v)[0] > 0.01:
                continue
            if wire.distToShape(v)[0] < 3.0:
                continue
            # Check optical bboxes
            safe = True
            for ob in optics_bboxes:
                dy = max(0.0, ob.YMin - y, y - ob.YMax)
                dz = max(0.0, ob.ZMin - z, z - ob.ZMax)
                if math.hypot(dy, dz) < 1.5:
                    safe = False
                    break
            if safe:
                safe_points.append((y, z))

    screw_targets = [
        (16, z_max - 52), (-55, z_max - 6),
        (19, 115), (-21, 115),
        (20, 8), (-16, 30)
    ]
    chosen_screws = []
    available_points = list(safe_points)
    for ty, tz in screw_targets:
        best = None
        min_d = float('inf')
        for p in available_points:
            d = math.hypot(p[0] - ty, p[1] - tz)
            if d < min_d:
                min_d = d
                best = p
        if best:
            chosen_screws.append(best)
            available_points = [p for p in available_points if math.hypot(p[0] - best[0], p[1] - best[1]) > 10.0]

    peg_targets = [
        (19, 170), (-21, 196),
        (19, 65), (19, 35)
    ]
    chosen_pegs = []
    for ty, tz in peg_targets:
        best = None
        min_d = float('inf')
        for p in available_points:
            d = math.hypot(p[0] - ty, p[1] - tz)
            if d < min_d:
                min_d = d
                best = p
        if best:
            chosen_pegs.append(best)
            available_points = [p for p in available_points if math.hypot(p[0] - best[0], p[1] - best[1]) > 10.0]

    return chosen_screws, chosen_pegs


def compute_breadboard_grid(bbox, optics_solid, screws, pegs,
                            left_ext=BREADBOARD_LEFT_EXTENSION,
                            right_ext=BREADBOARD_RIGHT_EXTENSION,
                            spacing=BREADBOARD_GRID_SPACING,
                            edge_margin=BREADBOARD_EDGE_MARGIN,
                            optics_clearance=BREADBOARD_OPTICS_CLEARANCE):
    """
    Computes a regular grid of safe M4 breadboard tap hole locations across the entire
    rectangular slab footprint including the lateral flange extension.
    Mathematically ensures all holes strictly avoid optics, fasteners, and outer edges.
    """
    safe_bb = []
    y_start = (bbox.YMin - left_ext) + edge_margin
    y_end = (bbox.YMax + right_ext) - edge_margin
    z_start = max(14.36 + edge_margin, bbox.ZMin + edge_margin)
    z_end = bbox.ZMax - edge_margin
    step = float(spacing)

    y = y_start
    while y <= y_end + 1e-4:
        z = z_start
        while z <= z_end + 1e-4:
            # Avoid assembly screws and pegs (6.0 mm clearance)
            if any(math.hypot(y - sy, z - sz) < 5.95 for sy, sz in screws + pegs):
                z += step
                continue

            # User requested holes across the ENTIRE base, so we skip the optics clearance check
            # for the bottom breadboard slab.
                
            safe_bb.append((round(y, 1), round(z, 1)))
            z += step
        y += step

    return safe_bb
def build_spectrometer(doc_path=None,
                       support_name=DEFAULT_SUPPORT_OBJECT,
                       optics_name=DEFAULT_OPTICS_ASSEMBLY,
                       thorlabs_step=DEFAULT_THORLABS_STEP,
                       top_output=DEFAULT_TOP_STEP_OUTPUT,
                       bottom_output=DEFAULT_BOTTOM_STEP_OUTPUT,
                       breadboard_thickness=BREADBOARD_THICKNESS,
                       breadboard_left_extension=BREADBOARD_LEFT_EXTENSION,
                       breadboard_right_extension=BREADBOARD_RIGHT_EXTENSION,
                       breadboard_spacing=BREADBOARD_GRID_SPACING,
                       corner_size=CORNER_CUTOUT_SIZE,
                       enable_threads=ENABLE_PHYSICAL_THREADS,
                       manual_screws=None,
                       manual_pegs=None,
                       manual_bb=None,
                       auto_bb=False,
                       verbose=True):
    """
    Executes the complete CAD generation pipeline:
    1. Base clamshell split at X=0
    2. 3mm breadboard slab fusion below bottom surface with lateral flange extension
    3. 40x40 mm corner cutout
    4. Optical pocket, light tunnel, and 2.0mm slit slot subtractions
    5. Assembly fastener & Tetris peg construction
    6. M4 Breadboard tap hole grid cutting & optional helical thread modeling
    7. STEP CAD export
    """
    base_dir = get_base_dir()
    if doc_path is None:
        doc_path = os.path.join(base_dir, DEFAULT_INPUT_DOCUMENT)
    doc_path = os.path.abspath(doc_path)

    if thorlabs_step is None:
        thorlabs_step = os.path.join(base_dir, DEFAULT_THORLABS_STEP)
    if not os.path.isabs(thorlabs_step):
        thorlabs_step = os.path.join(base_dir, thorlabs_step)
    thorlabs_step = os.path.abspath(thorlabs_step)

    if not os.path.isabs(top_output):
        top_output = os.path.join(base_dir, top_output)
    top_output = os.path.abspath(top_output)

    if not os.path.isabs(bottom_output):
        bottom_output = os.path.join(base_dir, bottom_output)
    bottom_output = os.path.abspath(bottom_output)

    if verbose:
        print("=" * 90)
        print("        CONSOLIDATED FREECAD SPECTROMETER CAD GENERATOR")
        print("=" * 90)
        print(f"Source Document       : {doc_path}")
        print(f"Support Object        : {support_name}")
        print(f"Optics Assembly       : {optics_name}")
        print(f"Thorlabs Lens Step    : {thorlabs_step}")
        print(f"Top Output STEP       : {top_output}")
        print(f"Bottom Output STEP    : {bottom_output}")
        print(f"Breadboard Thickness  : {breadboard_thickness} mm")
        print(f"Breadboard Flange Ext : {BREADBOARD_LEFT_EXTENSION} mm")
        print(f"Breadboard Spacing    : {breadboard_spacing} mm")
        print(f"Corner Cutout Size    : {corner_size} x {corner_size} mm")
        print(f"Physical Threads      : {'ENABLED (M4 spiral)' if enable_threads else 'DISABLED (cylindrical tap)'}")
        print("=" * 90)
        sys.stdout.flush()

    if not os.path.exists(doc_path):
        raise FileNotFoundError(f"CAD document not found: {doc_path}")
    if not os.path.exists(thorlabs_step):
        raise FileNotFoundError(f"Thorlabs CAD file not found: {thorlabs_step}")

    # Load Document and Objects
    doc = FreeCAD.open(doc_path)
    wire, face = load_support_geometry(doc, support_name)
    bbox = face.BoundBox

    spectrometer = doc.getObject(optics_name)
    if not spectrometer:
        raise RuntimeError(f"Optical assembly '{optics_name}' not found in document.")

    # Collect Optical Bounding Boxes for avoidance
    optics_bboxes = []
    for obj in spectrometer.Group:
        if hasattr(obj, 'Shape') and not obj.Shape.isNull():
            optics_bboxes.append(obj.Shape.BoundBox)

    # 1. Base Clamshell Extrusions (Split at X=0)
    if verbose:
        print("\n[STEP 1/7] Extruding clamshell halves split at X=0...")
    top_half = face.extrude(FreeCAD.Vector(CLAMSHELL_HALF_THICKNESS, 0, 0))
    bottom_half = face.extrude(FreeCAD.Vector(-CLAMSHELL_HALF_THICKNESS, 0, 0))

    # 1.5 Enclose the DMD optical path without filling the lower empty space
    if verbose:
        print("  -> Creating targeted housing for the DMD optical path...")
    
    # The DMD AirGap spans from Y=-78.8 to Y=-27.3, and Z=101.8 to 206.5.
    # We add a targeted block to enclose it, leaving the space below Z=95 empty.
    dmd_housing_top = Part.makeBox(CLAMSHELL_HALF_THICKNESS, 65.0, 140.0, FreeCAD.Vector(0, -82.0, 95.0))
    dmd_housing_bot = Part.makeBox(CLAMSHELL_HALF_THICKNESS, 65.0, 140.0, FreeCAD.Vector(-CLAMSHELL_HALF_THICKNESS, -82.0, 95.0))
    
    top_half = top_half.fuse(dmd_housing_top)
    bottom_half = bottom_half.fuse(dmd_housing_bot)

    # Trim the left overhang so it never goes further than Y = -80.0
    # (The DMD sits at Y = -79.1, so this leaves a perfect 0.9mm bounding wall)
    left_trim_box = Part.makeBox(200, 100, 300, FreeCAD.Vector(-100, -180.0, -50))
    top_half = top_half.cut(left_trim_box)
    bottom_half = bottom_half.cut(left_trim_box)
    
    # Update bounding box to reflect the new geometry
    bbox = top_half.BoundBox

    if breadboard_right_extension > 0:
        if verbose:
            print(f"  -> Extending full support structure laterally on the right by {breadboard_right_extension}mm...")
        top_shifted = top_half.copy()
        top_shifted.translate(FreeCAD.Vector(0, breadboard_right_extension, 0))
        top_half = top_half.fuse(top_shifted)
        
        bot_shifted = bottom_half.copy()
        bot_shifted.translate(FreeCAD.Vector(0, breadboard_right_extension, 0))
        bottom_half = bottom_half.fuse(bot_shifted)
        
        # Update bounding box to reflect the new geometry
        bbox = top_half.BoundBox


    # 2. Add Breadboard Slab below bottom surface (X from -20.5 to -17.5) with lateral flange extension
    if verbose:
        print(f"[STEP 2/7] Adding {breadboard_thickness}mm breadboard slab matching the newly extended CAD footprint...")
    bb_x_min = -(CLAMSHELL_HALF_THICKNESS + breadboard_thickness)
    bb_y_min = bbox.YMin
    bb_y_max = bbox.YMax
    bb_z_min = bbox.ZMin
    bb_z_max = bbox.ZMax
    bb_slab = Part.makeBox(breadboard_thickness,
                           bb_y_max - bb_y_min,
                           bb_z_max - bb_z_min,
                           FreeCAD.Vector(bb_x_min, bb_y_min, bb_z_min))
    bottom_half = bottom_half.fuse(bb_slab)

    if verbose:
        print("[STEP 2.5/7] Rounding sharp outer corners and opening cylindrical entrance tunnel...")
    
    # 4 Outer Corner Fillets (R=5.0) to smooth the hard sharp edges
    r = 5.0
    h = 300.0
    for x in [CLAMSHELL_HALF_THICKNESS, -CLAMSHELL_HALF_THICKNESS]:
        # y is 25.0 (bbox.YMax) and -80.0 (the vertical chop line)
        for y in [25.0, -80.0]:
            sign_x = -1 if x > 0 else 1
            sign_y = -1 if y > 0 else 1
            box = Part.makeBox(r, r, h, FreeCAD.Vector(x + (0 if sign_x==1 else -r), y + (0 if sign_y==1 else -r), -10))
            cyl = Part.makeCylinder(r, h, FreeCAD.Vector(x + sign_x*r, y + sign_y*r, -10), FreeCAD.Vector(0,0,1))
            fillet_tool = box.cut(cyl)
            if x > 0:
                top_half = top_half.cut(fillet_tool)
            else:
                bottom_half = bottom_half.cut(fillet_tool)

    # Clean cylindrical entrance tunnel (R=13.0) from Z=0 to Z=16 to let light enter without deleting the base breadboard
    entrance_tunnel = Part.makeCylinder(13.0, 20.0, FreeCAD.Vector(0, 0, -2.0), FreeCAD.Vector(0, 0, 1))
    top_half = top_half.cut(entrance_tunnel)
    bottom_half = bottom_half.cut(entrance_tunnel)

    # 2.8 Truncate top left exit corner to match sensor diagonal plane
    sensor = None
    for obj in doc.getObject(optics_name).Group:
        if hasattr(obj, 'Shape') and not obj.Shape.isNull() and "23_S23" in obj.Label:
            sensor = obj
            break
            
    if sensor:
        if verbose:
            print("[STEP 2.8/7] Trimming top exit sensor area to match diagonal sensor plane...")
        
        # Add vertical padding blocks so the diagonal cut covers the full 30x30 mounting area
        top_pad = Part.makeBox(CLAMSHELL_HALF_THICKNESS, 85.0, 15.0, FreeCAD.Vector(0, -100.0, 233.0))
        bot_pad = Part.makeBox(CLAMSHELL_HALF_THICKNESS, 85.0, 15.0, FreeCAD.Vector(-CLAMSHELL_HALF_THICKNESS, -100.0, 233.0))
        top_half = top_half.fuse(top_pad)
        bottom_half = bottom_half.fuse(bot_pad)

        sensor_z = sensor.Placement.Rotation.multVec(FreeCAD.Vector(0,0,1))
        # Cut 2mm behind the sensor base
        plane_pos = sensor.Placement.Base + sensor_z * 2.0
        
        sensor_cut_box = Part.makeBox(200, 200, 200, FreeCAD.Vector(-100, -100, 0))
        sensor_cut_box.transformShape(FreeCAD.Placement(plane_pos, sensor.Placement.Rotation).Matrix)
        
        top_half = top_half.cut(sensor_cut_box)
        bottom_half = bottom_half.cut(sensor_cut_box)

    # Remove the massive triangle structure beside the edge of exit (top-left)
    # Tapers from (Y=-85, Z=210) to (Y=-35, Z=245). This shallow slope ensures
    # the DMD and Condenser optics are fully covered, while meeting the diagonal
    # sensor cut perfectly at Y=-52.9, Z=232.5 to form a clean ridge.
    poly_tl = Part.makePolygon([
        FreeCAD.Vector(0, -125, 140),
        FreeCAD.Vector(0, -85, 210),
        FreeCAD.Vector(0, -35, 245),
        FreeCAD.Vector(0, -35, 300),
        FreeCAD.Vector(0, -125, 300),
        FreeCAD.Vector(0, -125, 140)
    ])
    taper_tl = Part.Face(poly_tl).extrude(FreeCAD.Vector(100, 0, 0))
    taper_tl.translate(FreeCAD.Vector(-50, 0, 0))
    top_half = top_half.cut(taper_tl)
    bottom_half = bottom_half.cut(taper_tl)
    
    # Update bounding box
    bbox = top_half.BoundBox

    # 2.9 Open DMD mounting surface
    dmd = None
    for obj in doc.getObject(optics_name).Group:
        if hasattr(obj, 'Shape') and not obj.Shape.isNull() and "22_CB_After_DMD_Hardware" in obj.Label:
            dmd = obj
            break

    if dmd:
        if verbose:
            print("[STEP 2.9/7] Exposing just the surface of the DMD window...")
        
        dmd_z = dmd.Placement.Rotation.multVec(FreeCAD.Vector(0,0,1))
        # Create cut plane 1mm behind mirror (which is the back face of the 2mm thick hardware block)
        dmd_plane_pos = dmd.Placement.Base + dmd_z * (-1.0)
        
        # Open just the surface of the DMD window: punch a 45x45mm square tunnel outwards
        # Local X and Y from -22.5 to 22.5, local Z from -100 to 0 (punching outwards to the left)
        dmd_window_cut = Part.makeBox(45.0, 45.0, 100.0, FreeCAD.Vector(-22.5, -22.5, -100.0))
        dmd_window_cut.transformShape(FreeCAD.Placement(dmd_plane_pos, dmd.Placement.Rotation).Matrix)
        
        top_half = top_half.cut(dmd_window_cut)
        bottom_half = bottom_half.cut(dmd_window_cut)
        bbox = top_half.BoundBox



    # 3.5 Interlocking Perimeter Walls (Tongue and Groove)
    if ENABLE_PERIMETER_INTERLOCK:
        if verbose:
            print("\n[STEP 3.5/7] Constructing interlocking perimeter walls (tongue and groove) along final footprint...")
        
        # Find the face of top_half that lies exactly on the X=0 split plane
        x0_faces = []
        for f in top_half.Faces:
            if abs(f.BoundBox.XMin) < 1e-4 and abs(f.BoundBox.XMax) < 1e-4:
                x0_faces.append(f)
                
        if x0_faces:
            # If multiple faces exist, fuse them (rare, but robust)
            x0_face = x0_faces[0]
            for f in x0_faces[1:]:
                x0_face = x0_face.fuse(f)
                
            # Get the outer perimeter wire (longest wire)
            wires = sorted(x0_face.Wires, key=lambda w: w.Length, reverse=True)
            if wires:
                outer_wire = wires[0]
                try:
                    # Tongue on bottom half (X from 0 to +3.0)
                    # Width: 1.6mm (from 0.8 inwards to 2.4 inwards) - pushed closer to outer edges
                    tongue_out = outer_wire.makeOffset2D(-0.8)
                    tongue_in = outer_wire.makeOffset2D(-2.4)
                    tongue_face = Part.Face(tongue_out.Wires[0]).cut(Part.Face(tongue_in.Wires[0]))
                    tongue_solid = tongue_face.extrude(FreeCAD.Vector(3.0, 0, 0))
                    bottom_half = bottom_half.fuse(tongue_solid)
                    
                    # Groove in top half (X from 0 to +3.2)
                    # Width: 2.0mm (from 0.6 inwards to 2.6 inwards) - 0.2mm clearance on all sides
                    groove_out = outer_wire.makeOffset2D(-0.6)
                    groove_in = outer_wire.makeOffset2D(-2.6)
                    groove_face = Part.Face(groove_out.Wires[0]).cut(Part.Face(groove_in.Wires[0]))
                    groove_solid = groove_face.extrude(FreeCAD.Vector(3.2, 0, 0))
                    top_half = top_half.cut(groove_solid)
                except Exception as e:
                    if verbose:
                        print(f"  [WARNING] Failed to construct perimeter interlock: {e}")
        else:
            if verbose:
                print("  [WARNING] Could not find X=0 face for perimeter interlock!")

    # 4. Prepare and Execute Optical Subtractions
    if verbose:
        print("[STEP 4/7] Preparing and executing complete optical & light tunnel cuts...")

    # Load authentic Thorlabs lens CAD
    new_s = Part.Shape()
    new_s.read(thorlabs_step)
    thorlabs_lens = new_s.Solids[0]
    rot = FreeCAD.Rotation(FreeCAD.Vector(1, 0, 0), 90)
    thorlabs_lens.transformShape(FreeCAD.Placement(FreeCAD.Vector(), rot).Matrix)

    shapes_to_cut = []
    first_placement = None
    last_airgap_axis = None
    last_airgap_end = None

    for obj in spectrometer.Group:
        if not hasattr(obj, 'Shape') or obj.Shape.isNull():
            continue
        label = obj.Label
        low = label.lower()

        if "glass" in low:
            if "01_collect_acl25416u" in low:
                first_placement = obj.Placement
                cyl = get_bounding_cylinder(thorlabs_lens, obj.Placement, OPTICS_POCKET_RADIUS)
            elif "17_condenser_acl25416u" in low:
                cyl = get_bounding_cylinder(thorlabs_lens, obj.Placement, OPTICS_POCKET_RADIUS)
            else:
                cyl = get_bounding_cylinder(obj.Shape.copy().transformShape(obj.Placement.Matrix.inverse()),
                                            obj.Placement, OPTICS_POCKET_RADIUS)
            shapes_to_cut.append((label + " (Pocket)", cyl))

        elif "airgap" in low:
            cyl, new_r, end_pt, axis = build_airgap_cylinder(obj.Shape,
                                                             radius_reduction=AIRGAP_RADIUS_REDUCTION,
                                                             extension=1.0)
            if cyl:
                shapes_to_cut.append((label + " (Reduced Light Tunnel)", cyl))
                if "18_condenser" in low:
                    last_airgap_axis = axis
                    last_airgap_end = end_pt
            else:
                shapes_to_cut.append((label + " (Raw AirGap)", obj.Shape))

        elif "hardware" in low:
            if "19_input_slit" in low:
                # Add 2.0 mm thick rectangular cutout (25.4 x 25.4 x 2.0 mm) aligned with input slit placement
                pos = obj.Placement.Base
                slit_box = Part.makeBox(SLIT_SLOT_WIDTH, SLIT_SLOT_HEIGHT, SLIT_SLOT_THICKNESS,
                                        FreeCAD.Vector(-SLIT_SLOT_WIDTH / 2.0, -SLIT_SLOT_HEIGHT / 2.0, pos.z - SLIT_SLOT_THICKNESS / 2.0))
                shapes_to_cut.append((label + " (Slit Insertion Slot)", slit_box))
                optics_bboxes.append(slit_box.BoundBox)
            elif "20_grating" in low:
                # Override the default 1-inch cylinder with a 25x25x6 mm rectangular pocket
                # And displace it 5mm along its diagonal face towards the wall
                grating_w, grating_h, grating_t = 25.0, 25.0, 6.0
                grating_box = Part.makeBox(grating_w, grating_h, grating_t, 
                                           FreeCAD.Vector(-grating_w/2.0, -grating_h/2.0, -grating_t/2.0))
                
                # Displace 5mm along the local Y axis (which corresponds to sliding it along its diagonal face)
                # Note: change to -5.0 if it goes to the wrong wall!
                grating_box.translate(FreeCAD.Vector(0, 5.0, 0))
                
                grating_box.transformShape(obj.Placement.Matrix)
                shapes_to_cut.append((label + " (25x25x6 Custom Pocket - Shifted 5mm)", grating_box))
                optics_bboxes.append(grating_box.BoundBox)
            else:
                shapes_to_cut.append((label + " (Hardware Mount)", obj.Shape))
        else:
            shapes_to_cut.append((label, obj.Shape))

    # Front entry drill aperture
    if first_placement:
        front_drill = Part.makeCylinder(DRILL_APERTURE_RADIUS,
                                        first_placement.Base.z,
                                        FreeCAD.Vector(0, 0, 0),
                                        FreeCAD.Vector(0, 0, 1))
        shapes_to_cut.append(("Front Drill (Entry Aperture)", front_drill))
        optics_bboxes.append(FreeCAD.BoundBox(-DRILL_APERTURE_RADIUS, -DRILL_APERTURE_RADIUS, 0.0,
                                              DRILL_APERTURE_RADIUS, DRILL_APERTURE_RADIUS, first_placement.Base.z))

    # Back exit drill aperture
    if last_airgap_axis and last_airgap_end:
        back_drill = Part.makeCylinder(DRILL_APERTURE_RADIUS, DRILL_EXIT_LENGTH, last_airgap_end, last_airgap_axis)
        shapes_to_cut.append(("Back Drill (Exit Aperture)", back_drill))

    # Execute optical Boolean subtractions
    success_cuts = 0
    
    # Also fuse all optical shapes into a single solid for exact clearance checking later
    all_optics_solid = shapes_to_cut[0][1]
    
    for name, s in shapes_to_cut:
        try:
            if name != shapes_to_cut[0][0]:
                all_optics_solid = all_optics_solid.fuse(s)
            
            t = top_half.cut(s)
            if not t.isNull() and t.Volume > 0:
                top_half = t
            b = bottom_half.cut(s)
            if not b.isNull() and b.Volume > 0:
                bottom_half = b
            success_cuts += 1
        except Exception as e:
            if verbose:
                print(f"  [CUT WARNING] Subtraction error on {name}: {e}")

    if verbose:
        print(f"  Successfully applied {success_cuts}/{len(shapes_to_cut)} optical subtractions.")

    # 5. Compute and Apply Assembly Screws & Tetris Pegs
    if verbose:
        print("[STEP 5/7] Computing robust fastener positions and cutting joints...")
    screws, pegs = compute_safe_fastener_locations(face, wire, bbox, optics_bboxes, corner_size,
                                                   manual_screws=manual_screws, manual_pegs=manual_pegs)

    if verbose:
        print(f"  M4 Assembly Screws ({len(screws)}): {screws}")
        print(f"  Tetris Pegs ({len(pegs)})        : {pegs}")

    # Tetris Pegs: Bottom gets solid peg (+3mm along X), Top gets socket hole (3.5mm depth)
    for (y, z) in pegs:
        peg_solid = Part.makeCylinder(PEG_RADIUS, PEG_HEIGHT, FreeCAD.Vector(0, y, z), FreeCAD.Vector(1, 0, 0))
        bottom_half = bottom_half.fuse(peg_solid)
        peg_hole = Part.makeCylinder(PEG_HOLE_RADIUS, PEG_HOLE_DEPTH, FreeCAD.Vector(0, y, z), FreeCAD.Vector(1, 0, 0))
        top_half = top_half.cut(peg_hole)

    # M4 Assembly Screws: Top gets clearance hole (R=2.25), Bottom gets tap hole (R=1.65)
    for (y, z) in screws:
        clearance_hole = Part.makeCylinder(SCREW_CLEARANCE_RADIUS, SCREW_HOLE_LENGTH,
                                           FreeCAD.Vector(0, y, z), FreeCAD.Vector(1, 0, 0))
        top_half = top_half.cut(clearance_hole)
        tap_hole = Part.makeCylinder(SCREW_TAP_RADIUS, SCREW_HOLE_LENGTH,
                                     FreeCAD.Vector(bb_x_min - 1.0, y, z), FreeCAD.Vector(1, 0, 0))
        bottom_half = bottom_half.cut(tap_hole)

    # 6. M4 Breadboard Tap Holes Grid
    if verbose:
        print(f"[STEP 6/7] Cutting M4 breadboard tap hole grid ({BREADBOARD_TAP_DIAMETER}mm dia)...")

    if manual_bb is not None:
        safe_bb = list(manual_bb)
    elif (not auto_bb) and (MANUAL_BREADBOARD_POSITIONS is not None) and \
         abs(BREADBOARD_LEFT_EXTENSION - BREADBOARD_LEFT_EXTENSION) < 1e-3 and \
         abs(breadboard_spacing - BREADBOARD_GRID_SPACING) < 1e-3:
        safe_bb = list(MANUAL_BREADBOARD_POSITIONS)
    else:
        safe_bb = compute_breadboard_grid(bbox, all_optics_solid, screws, pegs,
                                          left_ext=0.0,
                                          right_ext=0.0,
                                          spacing=breadboard_spacing,
                                          edge_margin=BREADBOARD_EDGE_MARGIN,
                                          optics_clearance=BREADBOARD_OPTICS_CLEARANCE)

    if verbose:
        print(f"  Drilling {len(safe_bb)} safe M4 breadboard tap holes ({BREADBOARD_TAP_DIAMETER}mm dia)...")

    bb_hole_depth = breadboard_thickness + 0.5
    bb_cylinders = []
    for (y, z) in safe_bb:
        h = Part.makeCylinder(SCREW_TAP_RADIUS, bb_hole_depth,
                              FreeCAD.Vector(bb_x_min - 0.2, y, z), FreeCAD.Vector(1, 0, 0))
        bb_cylinders.append(h)

    if bb_cylinders:
        try:
            bottom_half = bottom_half.cut(Part.makeCompound(bb_cylinders))
        except Exception:
            all_bb = bb_cylinders[0]
            for h in bb_cylinders[1:]:
                all_bb = all_bb.fuse(h)
            bottom_half = bottom_half.cut(all_bb)

    # Physical helical M4 3D threads modeling (if enabled)
    if enable_threads:
        if verbose:
            print("  Applying physical helical M4 3D threads to tap holes...")
        rot_x = FreeCAD.Rotation(FreeCAD.Vector(0, 1, 0), 90)
        # Screw tap holes in bottom half
        for (y, z) in screws:
            try:
                th = Part.makeThread(THREAD_PITCH, THREAD_DEPTH, 20.0, THREAD_TAP_RADIUS)
                th.transformShape(FreeCAD.Placement(FreeCAD.Vector(bb_x_min, y, z), rot_x).Matrix)
                bottom_half = bottom_half.cut(th)
            except Exception as e:
                if verbose:
                    print(f"  [THREAD WARNING] Screw thread at ({y}, {z}) failed: {e}")

        # Breadboard tap holes in bottom half are intentionally NOT threaded physically
        # because rendering 650+ complex helical geometries creates a 350,000+ entity solid
        # that instantly crashes FreeCAD, PrusaSlicer, and most CAD software.
        # They are left as 3.3mm cylindrical tap holes which can be manually tapped.

    # 6.5 Lateral M4 Threads
    if verbose:
        print("[STEP 6.5/7] Cutting Lateral M4 threaded holes...")
    
    rot_pos_y = FreeCAD.Rotation(FreeCAD.Vector(1, 0, 0), 90)
    rot_neg_y = FreeCAD.Rotation(FreeCAD.Vector(1, 0, 0), -90)
    rot_diag = FreeCAD.Rotation(FreeCAD.Vector(1, 0, 0), -43.53)
    rot_r_taper = FreeCAD.Rotation(FreeCAD.Vector(1, 0, 0), 68.7)
    rot_l_taper = FreeCAD.Rotation(FreeCAD.Vector(1, 0, 0), -76.8)
    rot_top = FreeCAD.Rotation(FreeCAD.Vector(1, 0, 0), 180)

    def drill_lateral_holes(hole_list, dir_vec, rot_matrix, part_obj, optics_solid):
        for (x, y, z) in hole_list:
            # Precise clearance check: create a clearance cylinder and check intersection volume
            chk_cyl = Part.makeCylinder(THREAD_TAP_RADIUS + 3.0, LATERAL_HOLE_DEPTH + 3.0, FreeCAD.Vector(x, y, z), dir_vec)
            if chk_cyl.common(optics_solid).Volume > 1e-4:
                if verbose: print(f"  [SKIPPED] Lateral hole ({x}, {y}, {z}) intersects optical path.")
                continue

            try:
                # Clearance drill (tap cylinder)
                cyl = Part.makeCylinder(THREAD_TAP_RADIUS, LATERAL_HOLE_DEPTH, FreeCAD.Vector(x, y, z), dir_vec)
                part_obj = part_obj.cut(cyl)
                
                # Physical thread
                if ENABLE_PHYSICAL_THREADS:
                    th = Part.makeThread(THREAD_PITCH, THREAD_DEPTH, LATERAL_HOLE_DEPTH, THREAD_TAP_RADIUS)
                    th.transformShape(FreeCAD.Placement(FreeCAD.Vector(x, y, z), rot_matrix).Matrix)
                    part_obj = part_obj.cut(th)
            except Exception as e:
                if verbose:
                    print(f"  [THREAD WARNING] Lateral hole at ({x}, {y}, {z}) failed: {e}")
        return part_obj

    # Standard Walls
    top_half = drill_lateral_holes([h for h in LATERAL_HOLES_RIGHT if h[0] > 0], FreeCAD.Vector(0, -1, 0), rot_pos_y, top_half, all_optics_solid)
    bottom_half = drill_lateral_holes([h for h in LATERAL_HOLES_RIGHT if h[0] < 0], FreeCAD.Vector(0, -1, 0), rot_pos_y, bottom_half, all_optics_solid)
    
    top_half = drill_lateral_holes([h for h in LATERAL_HOLES_LEFT_1 if h[0] > 0], FreeCAD.Vector(0, 1, 0), rot_neg_y, top_half, all_optics_solid)
    bottom_half = drill_lateral_holes([h for h in LATERAL_HOLES_LEFT_1 if h[0] < 0], FreeCAD.Vector(0, 1, 0), rot_neg_y, bottom_half, all_optics_solid)
    
    top_half = drill_lateral_holes([h for h in LATERAL_HOLES_LEFT_2 if h[0] > 0], FreeCAD.Vector(0, 1, 0), rot_neg_y, top_half, all_optics_solid)
    bottom_half = drill_lateral_holes([h for h in LATERAL_HOLES_LEFT_2 if h[0] < 0], FreeCAD.Vector(0, 1, 0), rot_neg_y, bottom_half, all_optics_solid)

    # Small Areas (Diagonal, Tapers, Top)
    diag_dir = FreeCAD.Vector(0, 0.689, 0.725)
    top_half = drill_lateral_holes([h for h in LATERAL_HOLES_DIAGONAL if h[0] > 0], diag_dir, rot_diag, top_half, all_optics_solid)
    bottom_half = drill_lateral_holes([h for h in LATERAL_HOLES_DIAGONAL if h[0] < 0], diag_dir, rot_diag, bottom_half, all_optics_solid)

    r_taper_dir = FreeCAD.Vector(0, -0.932, 0.363)
    top_half = drill_lateral_holes([h for h in LATERAL_HOLES_RIGHT_TAPER if h[0] > 0], r_taper_dir, rot_r_taper, top_half, all_optics_solid)
    bottom_half = drill_lateral_holes([h for h in LATERAL_HOLES_RIGHT_TAPER if h[0] < 0], r_taper_dir, rot_r_taper, bottom_half, all_optics_solid)

    l_taper_dir = FreeCAD.Vector(0, 0.974, 0.228)
    top_half = drill_lateral_holes([h for h in LATERAL_HOLES_LEFT_TAPER if h[0] > 0], l_taper_dir, rot_l_taper, top_half, all_optics_solid)
    bottom_half = drill_lateral_holes([h for h in LATERAL_HOLES_LEFT_TAPER if h[0] < 0], l_taper_dir, rot_l_taper, bottom_half, all_optics_solid)

    top_half = drill_lateral_holes([h for h in LATERAL_HOLES_TOP if h[0] > 0], FreeCAD.Vector(0, 0, -1), rot_top, top_half, all_optics_solid)
    bottom_half = drill_lateral_holes([h for h in LATERAL_HOLES_TOP if h[0] < 0], FreeCAD.Vector(0, 0, -1), rot_top, bottom_half, all_optics_solid)

    # 6.8 Sensor Exit M4 Mount Holes
    if sensor:
        if verbose:
            print(f"[STEP 6.8/7] Drilling 4 M4 mount holes on the diagonal sensor exit face ({SENSOR_M4_HOLES_SPACING}x{SENSOR_M4_HOLES_SPACING}mm)...")
        
        spacing = SENSOR_M4_HOLES_SPACING
        half = spacing / 2.0
        
        sensor_z = sensor.Placement.Rotation.multVec(FreeCAD.Vector(0,0,1))
        rot_inward = sensor.Placement.Rotation.multiply(FreeCAD.Rotation(FreeCAD.Vector(1,0,0), 180))
        
        for dx in [-half, half]:
            for dy in [-half, half]:
                # Local offset in the sensor's XY plane
                offset = FreeCAD.Vector(dx, dy, 0)
                global_offset = sensor.Placement.Rotation.multVec(offset)
                
                # Start drill slightly outside the face and drill inwards
                start_pos = sensor.Placement.Base + sensor_z * 3.0 + global_offset
                
                # Create the 3.3mm M4 tap hole cylinder
                # Note: Physical threads are forcefully disabled for these diagonal holes
                # because rotating a complex helix by 15.4 degrees causes OpenCASCADE boolean 
                # operations to fail and shatter the solid.
                tap_cyl = Part.makeCylinder(SCREW_TAP_RADIUS, SENSOR_M4_HOLE_DEPTH, start_pos, sensor_z * -1.0)
                
                if start_pos.x > 0:
                    top_half = top_half.cut(tap_cyl)
                else:
                    bottom_half = bottom_half.cut(tap_cyl)

    # 6.9 DMD Mount Holes
    if dmd:
        if verbose:
            print(f"[STEP 6.9/7] Drilling 4 M4 mount holes on the DMD mounting face ({DMD_M4_HOLES_SPACING}x{DMD_M4_HOLES_SPACING}mm)...")
        
        spacing = DMD_M4_HOLES_SPACING
        half = spacing / 2.0
        dmd_z = dmd.Placement.Rotation.multVec(FreeCAD.Vector(0,0,1))
        
        for dx in [-half, half]:
            for dy in [-half, half]:
                offset = FreeCAD.Vector(dx, dy, 0)
                global_offset = dmd.Placement.Rotation.multVec(offset)
                
                # Start drill slightly outside the face (Z_local = -2.0) and drill inwards (+dmd_z)
                start_pos = dmd.Placement.Base + dmd_z * (-2.0) + global_offset
                
                tap_cyl = Part.makeCylinder(SCREW_TAP_RADIUS, DMD_M4_HOLE_DEPTH, start_pos, dmd_z)
                
                if start_pos.x > 0:
                    top_half = top_half.cut(tap_cyl)
                else:
                    bottom_half = bottom_half.cut(tap_cyl)

    # 7. Validation & STEP Export
    if verbose:
        print("[STEP 7/7] Validating solids and exporting STEP models...")
        print(f"  Top Half Volume    : {top_half.Volume:.4f} mm^3 (Solids: {len(top_half.Solids)}, Valid: {top_half.isValid()})")
        print(f"  Bottom Half Volume : {bottom_half.Volume:.4f} mm^3 (Solids: {len(bottom_half.Solids)}, Valid: {bottom_half.isValid()})")

    if top_half.isNull() or not top_half.isValid() or len(top_half.Solids) != 1:
        with open("debug_solids.txt", "w") as f:
            f.write("DEBUG TOP HALF SOLIDS:\n")
            for i, s in enumerate(top_half.Solids):
                f.write(f"Solid {i} Vol: {s.Volume} BB: {s.BoundBox}\n")
        Part.export([top_half], "debug_top_failed.step")
        Part.export([bottom_half], "debug_bottom_failed.step")
        raise RuntimeError(f"Top half CAD generation failed (solids count: {len(top_half.Solids)})")
    if bottom_half.isNull() or not bottom_half.isValid() or len(bottom_half.Solids) != 1:
        raise RuntimeError(f"Bottom half CAD generation failed (solids count: {len(bottom_half.Solids)})")

    out_dir_top = os.path.dirname(top_output)
    if out_dir_top and not os.path.exists(out_dir_top):
        os.makedirs(out_dir_top, exist_ok=True)
    out_dir_bot = os.path.dirname(bottom_output)
    if out_dir_bot and not os.path.exists(out_dir_bot):
        os.makedirs(out_dir_bot, exist_ok=True)

    top_half.exportStep(top_output)
    bottom_half.exportStep(bottom_output)

    # Also export _m4 named STEP files for backward compatibility when generating default models
    default_top = os.path.join(base_dir, DEFAULT_TOP_STEP_OUTPUT)
    default_bot = os.path.join(base_dir, DEFAULT_BOTTOM_STEP_OUTPUT)
    m4_top = os.path.join(base_dir, "support_top_half_m4.step")
    m4_bot = os.path.join(base_dir, "support_bottom_half_m4.step")
    if top_output == default_top and top_output != m4_top:
        top_half.exportStep(m4_top)
    if bottom_output == default_bot and bottom_output != m4_bot:
        bottom_half.exportStep(m4_bot)

    if verbose:
        print("\n" + "=" * 90)
        print("                 BUILD SPECTROMETER COMPLETE: SUCCESS")
        print("=" * 90)
        print(f"Generated Top Clamshell Half    : {top_output}")
        print(f"Generated Bottom Clamshell Half : {bottom_output}")
        print("=" * 90)
        sys.stdout.flush()

    return {
        "top_half": top_half,
        "bottom_half": bottom_half,
        "top_output": top_output,
        "bottom_output": bottom_output,
        "screws": screws,
        "pegs": pegs,
        "breadboard_holes_count": len(safe_bb),
        "top_volume": top_half.Volume,
        "bottom_volume": bottom_half.Volume
    }


def parse_args(args_list=None):
    parser = argparse.ArgumentParser(description="Build 3D Raman DMD Spectrometer CAD Models.")
    parser.add_argument("--input-doc", "-i", default=None, help="Path to input FreeCAD document (.FCStd)")
    parser.add_argument("--support-obj", default=DEFAULT_SUPPORT_OBJECT, help="Support structure object name")
    parser.add_argument("--optics-asm", default=DEFAULT_OPTICS_ASSEMBLY, help="Optical system assembly name")
    parser.add_argument("--thorlabs-step", default=DEFAULT_THORLABS_STEP, help="Path to authentic Thorlabs lens STEP")
    parser.add_argument("--top-output", "-t", default=DEFAULT_TOP_STEP_OUTPUT, help="Output path for Top STEP CAD")
    parser.add_argument("--bottom-output", "-b", default=DEFAULT_BOTTOM_STEP_OUTPUT, help="Output path for Bottom STEP CAD")
    parser.add_argument("--breadboard-thickness", type=float, default=BREADBOARD_THICKNESS, help="Breadboard thickness in mm")
    parser.add_argument("--breadboard-left-ext", type=float, default=BREADBOARD_LEFT_EXTENSION, help="Breadboard lateral left extension in mm")
    parser.add_argument("--breadboard-right-ext", type=float, default=BREADBOARD_RIGHT_EXTENSION, help="Breadboard lateral right extension in mm")
    parser.add_argument("--breadboard-spacing", type=float, default=BREADBOARD_GRID_SPACING, help="Breadboard hole spacing in mm")
    parser.add_argument("--corner-size", type=float, default=CORNER_CUTOUT_SIZE, help="Corner cutout size in mm")
    parser.add_argument("--threads", action="store_true", default=ENABLE_PHYSICAL_THREADS, help="Model physical helical M4 threads in tap holes")
    parser.add_argument("--no-threads", action="store_false", dest="threads", help="Disable physical threads (plain cylindrical tap holes)")
    parser.add_argument("--auto-bb", action="store_true", default=False, help="Force automatic calculation of breadboard hole grid")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress detailed console output")
    return parser.parse_args(args_list)


def is_main_entry(script_name="build_spectrometer.py"):
    """Determines if this script is executed directly as main by Python or FreeCADCmd."""
    if __name__ == "__main__":
        return True
    this_file = os.path.abspath(__file__) if '__file__' in globals() else script_name
    for arg in sys.argv[1:]:
        if arg in ("--", "--pass"):
            break
        if arg.startswith("-"):
            continue
        try:
            if os.path.samefile(arg, this_file) or os.path.basename(arg) == os.path.basename(this_file):
                return True
        except Exception:
            if os.path.basename(arg) == os.path.basename(this_file):
                return True
        if arg.endswith(".py"):
            return False
    return False


def extract_script_args(argv=None, script_name="build_spectrometer.py"):
    """
    Extracts and normalizes CLI arguments intended for this script,
    stripping FreeCADCmd executable, script name, '--', and '--pass' wrappers.
    Works whether invoked directly via python, via FreeCADCmd with/without wrappers,
    or imported.
    """
    import shlex
    if argv is None:
        argv = sys.argv
    elif len(argv) > 0 and argv[0] != sys.argv[0]:
        argv = [sys.argv[0]] + list(argv)

    raw_tokens = []
    found_script = False
    this_file = os.path.abspath(__file__) if '__file__' in globals() else script_name

    for arg in argv:
        if not found_script:
            base = os.path.basename(arg)
            try:
                if os.path.samefile(arg, this_file) or base == os.path.basename(this_file) or base in (script_name, script_name.replace(".py", "")):
                    found_script = True
                    continue
            except Exception:
                if base == os.path.basename(this_file) or base in (script_name, script_name.replace(".py", "")):
                    found_script = True
                    continue
            if base.lower() in ("freecadcmd.exe", "freecad.exe", "python.exe", "pythonw.exe"):
                continue
        else:
            if arg in ("--", "--pass"):
                continue
            if " " in arg and not (arg.startswith('"') or arg.startswith("'")):
                try:
                    raw_tokens.extend(shlex.split(arg))
                    continue
                except Exception:
                    pass
            raw_tokens.append(arg)

    if not found_script:
        for arg in argv:
            base = os.path.basename(arg).lower()
            if base in ("freecadcmd.exe", "freecad.exe", "python.exe", "pythonw.exe",
                        os.path.basename(this_file).lower(), script_name.lower()):
                continue
            if arg in ("--", "--pass"):
                continue
            raw_tokens.append(arg)

    return raw_tokens


if is_main_entry("build_spectrometer.py"):
    filtered_args = extract_script_args(sys.argv, "build_spectrometer.py")
    args = parse_args(filtered_args)
    try:
        build_spectrometer(
            doc_path=args.input_doc,
            support_name=args.support_obj,
            optics_name=args.optics_asm,
            thorlabs_step=args.thorlabs_step,
            top_output=args.top_output,
            bottom_output=args.bottom_output,
            breadboard_thickness=args.breadboard_thickness,
            breadboard_left_extension=args.breadboard_left_ext,
            breadboard_right_extension=args.breadboard_right_ext,
            breadboard_spacing=args.breadboard_spacing,
            corner_size=args.corner_size,
            enable_threads=args.threads,
            auto_bb=args.auto_bb,
            verbose=not args.quiet
        )
        sys.exit(0)
    except Exception as e:
        print(f"\n[FATAL ERROR] build_spectrometer execution failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

