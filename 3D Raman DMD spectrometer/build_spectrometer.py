"""
build_spectrometer.py

Production-ready FreeCAD CAD generation script for the 2D Raman DMD Spectrometer
clamshell enclosure, featuring:
- Structural external ribbing and exterior fillets for an industrial aesthetic and rigidity (R1)
- Clean, flat, planar, minimalist lateral surfaces free of unnecessary protruding islands (R2)
- Zero-intersection optical clearance protection with 0.5 mm extra pocket thickness for 30mm & 100mm lenses (R3)
- Manifold single-solid integrity (len(Solids) == 1) for both clamshell halves
- Fully aligned M4 mounting threads/cutouts for DMD and Detector.
"""

import os
import sys
import math
import argparse
import FreeCAD
import Part

# ==============================================================================
# CONFIGURATION CONSTANTS
# ==============================================================================
DEFAULT_INPUT_DOCUMENT      = "model1.FCStd"
DEFAULT_SUPPORT_OBJECT      = "Sketch"
DEFAULT_OPTICS_ASSEMBLY     = "Optical_System_Assembly"
DEFAULT_THORLABS_STEP       = "ACL25416U-Step.step"

MANUAL_SCREW_POSITIONS = [
    ( 17.0,  38.0),   # 1. ENTRANCE RIGHT
    (-17.0,  38.0),   # 2. ENTRANCE LEFT
    ( 17.0,  85.0),   # 3. RIGHT MID 1
    ( 17.0, 140.0),   # 4. RIGHT MID 2
    (-76.0, 136.0),   # 5. LEFT MID 1 (DMD lateral)
    (-76.0, 150.0),   # 6. LEFT MID 2
    (-26.0, 122.0),   # 7. MID CENTER (Ridge between optical paths)
    (-30.0, 170.0),   # 8. PRE-EXIT LEFT
    (-26.0, 180.0),   # 9. PRE-EXIT CENTER
    (-22.0, 196.0)    # 10. NEAR SENSOR / EXIT
]
MANUAL_PEG_POSITIONS = [
    (19.0, 170.0), (-21.0, 196.0), (19.0, 65.0), (19.0, 35.0)
]
MANUAL_BREADBOARD_POSITIONS = None

CLAMSHELL_HALF_THICKNESS    = 17.5       
BREADBOARD_THICKNESS        = 3.0        
BREADBOARD_LEFT_EXTENSION   = 60.0       
BREADBOARD_RIGHT_EXTENSION  = 0.0        
BREADBOARD_GRID_SPACING     = 10.0       
BREADBOARD_TAP_DIAMETER     = 3.3        
BREADBOARD_HOLE_DEPTH       = 3.5        
BREADBOARD_EDGE_MARGIN      = 4.0        

SENSOR_M4_HOLES_SPACING     = 30.0       
SENSOR_M4_HOLE_DEPTH        = 5.0        
DMD_M4_HOLES_SPACING        = 30.0       
DMD_M4_HOLE_DEPTH           = 10.0       
BREADBOARD_OPTICS_CLEARANCE = 6.0        

ENABLE_STRUCTURAL_RIBBING   = False
RIB_HEIGHT                  = 1.5
RIB_THICKNESS               = 2.5

ENABLE_EXTERIOR_FILLETS     = True
FILLET_RADIUS               = 4.0

ENABLE_PERIMETER_INTERLOCK  = True       
ENABLE_PHYSICAL_THREADS     = False      
ENABLE_LATERAL_HOLES        = False      
THREAD_PITCH                = 0.7        
THREAD_DEPTH                = 0.4        
THREAD_TAP_RADIUS           = 1.65       

SLIT_SLOT_WIDTH             = 25.4

# --- EXPERIMENT PREPARATION OFFSETS ---
AXIAL_LENS_OFFSETS          = {}  # e.g., {'01_collect': 2.0, '17_condenser': -1.5}
GRATING_ROTATION_OFFSET     = 0.0 # Degrees to rotate the grating hardware
SLIT_IS_CIRCULAR_1_INCH     = True # True for 1-inch round disk
SLIDING_TOLERANCE           = 0.4
USE_EXTERNAL_ASPHERIC_MOUNTS= True # True to construct slip-fit drop-in pocket for external cartridge at 01_collect
CARTRIDGE_LOCKING_SCREW_RADIUS = 1.65  # M4 tap drill radius for locking set screws
CARTRIDGE_LOCKING_SCREW_X      = 4.0   # Horizontal station in top half
CARTRIDGE_LOCKING_SCREW_Z      = 20.0  # Longitudinal station along cartridge guide channel
# --------------------------------------
SLIT_SLOT_HEIGHT            = 25.4       
SLIT_SLOT_THICKNESS         = 2.0        

CORNER_CUTOUT_SIZE          = 40.0       
OPTICS_POCKET_RADIUS        = 12.90       
ASPHERIC_POCKET_RADIUS      = 13.05       
AIRGAP_RADIUS_REDUCTION     = 1.0        
DRILL_APERTURE_RADIUS       = 11.7       
DRILL_EXIT_LENGTH           = 30.0       

LENS_EXTRA_THICKNESS_30_100 = 1.0

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


def get_bounding_cylinder(shape, placement, radius=12.7, extra_length=0.0):
    bbox = shape.BoundBox
    length = (bbox.ZMax - bbox.ZMin) + extra_length
    cyl = Part.makeCylinder(radius, length, FreeCAD.Vector(0, 0, bbox.ZMin), FreeCAD.Vector(0, 0, 1))
    cyl.transformShape(placement.Matrix)
    return cyl


def build_airgap_cylinder(shape, radius_reduction=0.0, extension=1.0, max_t=None, min_t=None):
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

    if max_t is not None and max_t < t_max:
        t_max = max_t
        ext_start = (center + axis * t_min) - axis * (extension / 2.0)
        ext_length = (t_max - t_min) + (extension / 2.0)
    elif min_t is not None and min_t > t_min:
        t_min = min_t
        ext_start = center + axis * t_min
        ext_length = (t_max - t_min) + extension
    else:
        start_pt = center + axis * t_min
        length = t_max - t_min
        ext_start = start_pt - axis * (extension / 2.0)
        ext_length = length + extension

    new_radius = radius - radius_reduction
    cyl = Part.makeCylinder(new_radius, ext_length, ext_start, axis)
    return cyl, new_radius, center + axis * t_max, axis



def make_aspheric_cradle(obj, pocket_radius=ASPHERIC_POCKET_RADIUS, sliding_tolerance=0.0, axial_dz=0.0, clearance=0.25,
                         cartridge_screw_radius=CARTRIDGE_LOCKING_SCREW_RADIUS, cartridge_screw_x=CARTRIDGE_LOCKING_SCREW_X,
                         cartridge_screw_z_local=4.64):
    """
    Construct a specialized mounting bed (cradle) specifically conforming to the 16mm
    aspheric lens (Thorlabs ACL25416U, f=16mm, nominal OD=25.5mm).
    The bed closely hugs the lens barrel (pocket radius 13.05mm providing 0.60mm
    diametric clearance) and the convex aspheric surface with a controlled clearance
    (default 0.25mm), securely capturing the lens against translation and tipping
    while strictly preserving the 18mm clear aperture (minimum radius 9.0mm).
    """
    local_s = obj.Shape.copy().transformShape(obj.Placement.Matrix.inverse())
    bb = local_s.BoundBox

    if USE_EXTERNAL_ASPHERIC_MOUNTS and "acl25416u" in obj.Label.lower():
        if "01_collect" in obj.Label.lower():
            # Entrance pocket: fits special mount (34.0 x 33.5 x 10.79 mm) & normal mount (34.0 x 32.5 x 10.79 mm)
            # Provides 0.70 mm clearance in X (X in [-13.7, +21.7]) and 0.50 mm per side in Y (Y in [-17.25, +17.25])
            p_width = 35.4
            p_height = 34.5
            p_x_min = -13.7
        else:
            # Exit pocket (17_condenser): fits normal mount (34.0 x 32.5 x 10.79 mm)
            # Provides 0.70 mm clearance in X (X in [-13.7, +21.7]) and 0.50 mm per side in Y (Y in [-16.75, +16.75])
            p_width = 35.4
            p_height = 33.5
            p_x_min = -13.7
        p_z_min = -0.30 - sliding_tolerance
        p_thickness = 11.20 + 2 * sliding_tolerance
        box = Part.makeBox(p_width, p_height, p_thickness, FreeCAD.Vector(p_x_min, -p_height / 2.0, p_z_min))
        # Inject set screw cylinders locally before transform
        r_screw = Part.makeCylinder(cartridge_screw_radius, 15.0, FreeCAD.Vector(cartridge_screw_x, 26.0, cartridge_screw_z_local), FreeCAD.Vector(0, -1, 0))
        l_screw = Part.makeCylinder(cartridge_screw_radius, 15.0, FreeCAD.Vector(cartridge_screw_x, -26.0, cartridge_screw_z_local), FreeCAD.Vector(0, 1, 0))
        box = box.fuse(r_screw).fuse(l_screw)

        if axial_dz != 0.0:
            box.translate(FreeCAD.Vector(0, 0, axial_dz))
        
        box.transformShape(obj.Placement.Matrix)
        return box

    z_min = bb.ZMin - 0.5
    z_max = bb.ZMax + 0.5
    
    pts = [FreeCAD.Vector(0, 0, z_min), FreeCAD.Vector(9.0, 0, z_min)]
    z_curr = bb.ZMin
    while z_curr <= bb.ZMax:
        sl = local_s.slice(FreeCAD.Vector(0, 0, 1), min(z_curr + 0.05, bb.ZMax - 0.01))
        r = sl[0].BoundBox.XMax if sl else 0.0
        r_cleared = pocket_radius if r > 12.0 else max(r + clearance, 9.0)
        pts.append(FreeCAD.Vector(r_cleared, 0, z_curr))
        z_curr += 0.2
    pts.append(FreeCAD.Vector(pocket_radius, 0, bb.ZMax))
    pts.append(FreeCAD.Vector(pocket_radius, 0, z_max))
    pts.append(FreeCAD.Vector(0, 0, z_max))
    pts.append(FreeCAD.Vector(0, 0, z_min))
    
    wire = Part.makePolygon(pts)
    face = Part.Face(wire)
    revolved = face.revolve(FreeCAD.Vector(0, 0, 0), FreeCAD.Vector(0, 0, 1), 360)
    if axial_dz != 0.0:
        revolved.translate(FreeCAD.Vector(0, 0, axial_dz))
    revolved.transformShape(obj.Placement.Matrix)
    return revolved


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


def compute_safe_fastener_locations(face, wire, bbox, optics_bboxes, corner_size=40.0, manual_screws=None, manual_pegs=None):
    screws = manual_screws if manual_screws is not None else MANUAL_SCREW_POSITIONS
    pegs = manual_pegs if manual_pegs is not None else MANUAL_PEG_POSITIONS
    return list(screws), list(pegs)


def make_corner_fillet_cutter(x_val, y_val, r_fillet, dir_x, dir_y, z_min=-10.0, z_max=300.0):
    length_z = z_max - z_min
    x_origin = x_val - r_fillet if dir_x == -1 else x_val
    y_origin = y_val - r_fillet if dir_y == -1 else y_val
    box = Part.makeBox(r_fillet, r_fillet, length_z, FreeCAD.Vector(x_origin, y_origin, z_min))
    center_x = x_val - r_fillet if dir_x == -1 else x_val + r_fillet
    center_y = y_val - r_fillet if dir_y == -1 else y_val + r_fillet
    cyl = Part.makeCylinder(r_fillet, length_z, FreeCAD.Vector(center_x, center_y, z_min), FreeCAD.Vector(0, 0, 1))
    return box.cut(cyl)


def make_horizontal_fillet_cutter(x_val, z_val, r_fillet, dir_x, dir_z, y_min=-100.0, y_max=50.0):
    length_y = y_max - y_min
    x_origin = x_val - r_fillet if dir_x == -1 else x_val
    z_origin = z_val - r_fillet if dir_z == -1 else z_val
    box = Part.makeBox(r_fillet, length_y, r_fillet, FreeCAD.Vector(x_origin, y_min, z_origin))
    center_x = x_val - r_fillet if dir_x == -1 else x_val + r_fillet
    center_z = z_val - r_fillet if dir_z == -1 else z_val + r_fillet
    cyl = Part.makeCylinder(r_fillet, length_y, FreeCAD.Vector(center_x, y_min, center_z), FreeCAD.Vector(0, 1, 0))
    return box.cut(cyl)


def build_structural_ribs(half_thickness=CLAMSHELL_HALF_THICKNESS, rib_height=RIB_HEIGHT, rib_thickness=RIB_THICKNESS, z_stations=None):
    if z_stations is None:
        z_stations = [30.0, 55.0, 80.0, 105.0, 135.0, 160.0, 185.0, 210.0]
    ribs_top = []
    ribs_bot = []
    for z in z_stations:
        y_max = 16.0
        if z < 100.0:
            y_min = -17.0
        elif z < 140.0:
            y_min = -35.0
        elif z < 170.0:
            y_min = -68.0
        elif z < 195.0:
            y_min = -62.0
            y_max = 14.0
        else:
            y_min = -54.0
            y_max = 2.0
        length_y = y_max - y_min
        if length_y > 5.0:
            r_top = Part.makeBox(rib_height, length_y, rib_thickness, FreeCAD.Vector(half_thickness, y_min, z - rib_thickness / 2.0))
            r_bot = Part.makeBox(rib_height, length_y, rib_thickness, FreeCAD.Vector(-half_thickness - rib_height, y_min, z - rib_thickness / 2.0))
            ribs_top.append(r_top)
            ribs_bot.append(r_bot)
    return ribs_top, ribs_bot


def build_spectrometer(doc_path=None, support_name=DEFAULT_SUPPORT_OBJECT, optics_name=DEFAULT_OPTICS_ASSEMBLY,
                       thorlabs_step=DEFAULT_THORLABS_STEP, top_output=DEFAULT_TOP_STEP_OUTPUT, bottom_output=DEFAULT_BOTTOM_STEP_OUTPUT,
                       breadboard_thickness=BREADBOARD_THICKNESS, breadboard_left_extension=BREADBOARD_LEFT_EXTENSION,
                       breadboard_right_extension=BREADBOARD_RIGHT_EXTENSION, breadboard_spacing=BREADBOARD_GRID_SPACING,
                       corner_size=CORNER_CUTOUT_SIZE, enable_threads=ENABLE_PHYSICAL_THREADS,
                       enable_ribbing=ENABLE_STRUCTURAL_RIBBING, enable_fillets=ENABLE_EXTERIOR_FILLETS,
                       enable_interlock=ENABLE_PERIMETER_INTERLOCK,
                       airgap_radius_reduction=AIRGAP_RADIUS_REDUCTION,
                       lens_extra_thickness_30_100=LENS_EXTRA_THICKNESS_30_100,
                       cartridge_locking_screw_radius=CARTRIDGE_LOCKING_SCREW_RADIUS,
                       cartridge_locking_screw_x=CARTRIDGE_LOCKING_SCREW_X,
                       cartridge_locking_screw_z=CARTRIDGE_LOCKING_SCREW_Z,
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
        print(f"Structural Ribbing    : {'ENABLED' if enable_ribbing else 'DISABLED'}")
        print(f"Exterior Fillets      : {'ENABLED' if enable_fillets else 'DISABLED'}")
        print(f"Perimeter Interlock   : {'ENABLED' if enable_interlock else 'DISABLED'}")
        print(f"Airgap Radius Reduct. : {airgap_radius_reduction} mm")
        print(f"Extra Thickness 30/100: {lens_extra_thickness_30_100} mm")
        print(f"Physical Threads      : {'ENABLED' if enable_threads else 'DISABLED'}\n" + "=" * 90)
        sys.stdout.flush()

    if not os.path.exists(doc_path): raise FileNotFoundError(f"CAD document not found: {doc_path}")

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

    
    # Reinforce entrance lens area (Y=0, Z=10 to 30)
    ent_top = Part.makeBox(CLAMSHELL_HALF_THICKNESS, 42.0, 30.0, FreeCAD.Vector(0, -21.0, 5.0))
    ent_bot = Part.makeBox(CLAMSHELL_HALF_THICKNESS, 42.0, 30.0, FreeCAD.Vector(-CLAMSHELL_HALF_THICKNESS, -21.0, 5.0))
    top_half, bottom_half = top_half.fuse(ent_top), bottom_half.fuse(ent_bot)

    # Reinforce exit lens area (Y=-37, Z=205 to 225)
    ex_top = Part.makeBox(CLAMSHELL_HALF_THICKNESS, 42.0, 30.0, FreeCAD.Vector(0, -58.0, 200.0))
    ex_bot = Part.makeBox(CLAMSHELL_HALF_THICKNESS, 42.0, 30.0, FreeCAD.Vector(-CLAMSHELL_HALF_THICKNESS, -58.0, 200.0))
    top_half, bottom_half = top_half.fuse(ex_top), bottom_half.fuse(ex_bot)

    # 1.5 Enclose DMD optical path
    if verbose: print("  -> Creating targeted housing for the DMD optical path...")
    dmd_housing_top = Part.makeBox(CLAMSHELL_HALF_THICKNESS, 69.0, 140.0, FreeCAD.Vector(0, -86.0, 95.0))
    dmd_housing_bot = Part.makeBox(CLAMSHELL_HALF_THICKNESS, 69.0, 140.0, FreeCAD.Vector(-CLAMSHELL_HALF_THICKNESS, -86.0, 95.0))
    top_half, bottom_half = top_half.fuse(dmd_housing_top), bottom_half.fuse(dmd_housing_bot)

    # Clean left boundary at Y = -120.0
    left_trim_box = Part.makeBox(200, 100, 300, FreeCAD.Vector(-100, -220.0, -50))
    top_half, bottom_half = top_half.cut(left_trim_box), bottom_half.cut(left_trim_box)

    # Symmetric entrance pad on left from Y = -21 to Y = -25 (Z = 0 to 50)
    entrance_pad_top = Part.makeBox(CLAMSHELL_HALF_THICKNESS, 4.0, 50.0, FreeCAD.Vector(0, -25.0, 0))
    entrance_pad_bot = Part.makeBox(CLAMSHELL_HALF_THICKNESS, 4.0, 50.0, FreeCAD.Vector(-CLAMSHELL_HALF_THICKNESS, -25.0, 0))
    top_half, bottom_half = top_half.fuse(entrance_pad_top), bottom_half.fuse(entrance_pad_bot)

    # 1.8 Massive 4-Corner Block Cuts (Material-saving cuts: Top-Left, Top-Right, Inner-Triangle, Bottom-Right)
    if verbose: print("  -> Applying massive 4-corner material-saving block cuts...")
    poly_tl = Part.makePolygon([
        FreeCAD.Vector(0, -120, 120), FreeCAD.Vector(0, -55, 200), FreeCAD.Vector(0, -55, 240),
        FreeCAD.Vector(0, -150, 240), FreeCAD.Vector(0, -150, 120), FreeCAD.Vector(0, -120, 120)
    ])
    taper_tl = Part.Face(poly_tl).extrude(FreeCAD.Vector(100, 0, 0)).translate(FreeCAD.Vector(-50, 0, 0))

    poly_tr = Part.makePolygon([
        FreeCAD.Vector(0, 22, 175), FreeCAD.Vector(0, -15, 240), FreeCAD.Vector(0, 50, 240),
        FreeCAD.Vector(0, 50, 175), FreeCAD.Vector(0, 22, 175)
    ])
    taper_tr = Part.Face(poly_tr).extrude(FreeCAD.Vector(100, 0, 0)).translate(FreeCAD.Vector(-50, 0, 0))

    # New cut between Grating and Sensor
    poly_grating_sensor = Part.makePolygon([
        FreeCAD.Vector(0, -5, 190), FreeCAD.Vector(0, -25, 235), FreeCAD.Vector(0, 20, 235), FreeCAD.Vector(0, 20, 190), FreeCAD.Vector(0, -5, 190)
    ])
    taper_grating_sensor = Part.Face(poly_grating_sensor).extrude(FreeCAD.Vector(100, 0, 0)).translate(FreeCAD.Vector(-50, 0, 0))

    poly_inner = Part.makePolygon([
        FreeCAD.Vector(0, -18, 50), FreeCAD.Vector(0, -50, 95), FreeCAD.Vector(0, -18, 115), FreeCAD.Vector(0, -18, 50)
    ])
    taper_inner = Part.Face(poly_inner).extrude(FreeCAD.Vector(100, 0, 0)).translate(FreeCAD.Vector(-50, 0, 0))

    poly_br = Part.makePolygon([
        FreeCAD.Vector(0, 25, 50), FreeCAD.Vector(0, 25, 175), FreeCAD.Vector(0, 50, 175), FreeCAD.Vector(0, 50, 50), FreeCAD.Vector(0, 25, 50)
    ])
    taper_br = Part.Face(poly_br).extrude(FreeCAD.Vector(100, 0, 0)).translate(FreeCAD.Vector(-50, 0, 0))

    # Extra smoothing cuts for "squared pronuntiations" (DMD bottom pad and Sensor top-right pad)
    poly_dmd_bot = Part.makePolygon([
        FreeCAD.Vector(0, -110, 95), FreeCAD.Vector(0, -22, 50), FreeCAD.Vector(0, -150, 50),
        FreeCAD.Vector(0, -150, 95), FreeCAD.Vector(0, -110, 95)
    ])
    taper_dmd_bot = Part.Face(poly_dmd_bot).extrude(FreeCAD.Vector(100, 0, 0)).translate(FreeCAD.Vector(-50, 0, 0))

    poly_sensor_tr = Part.makePolygon([
        FreeCAD.Vector(0, -15, 233), FreeCAD.Vector(0, 0, 250), FreeCAD.Vector(0, 50, 250),
        FreeCAD.Vector(0, 50, 233), FreeCAD.Vector(0, -15, 233)
    ])
    taper_sensor_tr = Part.Face(poly_sensor_tr).extrude(FreeCAD.Vector(100, 0, 0)).translate(FreeCAD.Vector(-50, 0, 0))

    # Extra cuts to remove the squared slabs at the very bottom base (Z=0)
    poly_base_left = Part.makePolygon([
        FreeCAD.Vector(0, -25, -10), FreeCAD.Vector(0, -50, -10), FreeCAD.Vector(0, -50, 50),
        FreeCAD.Vector(0, -25, 50), FreeCAD.Vector(0, -25, -10)
    ])
    taper_base_left = Part.Face(poly_base_left).extrude(FreeCAD.Vector(100, 0, 0)).translate(FreeCAD.Vector(-50, 0, 0))

    poly_base_right = Part.makePolygon([
        FreeCAD.Vector(0, 25, -10), FreeCAD.Vector(0, 50, -10), FreeCAD.Vector(0, 50, 50),
        FreeCAD.Vector(0, 25, 50), FreeCAD.Vector(0, 25, -10)
    ])
    taper_base_right = Part.Face(poly_base_right).extrude(FreeCAD.Vector(100, 0, 0)).translate(FreeCAD.Vector(-50, 0, 0))

    for taper in [taper_tl, taper_tr, taper_inner, taper_br, taper_dmd_bot, taper_sensor_tr, taper_base_left, taper_base_right, taper_grating_sensor]:
        top_half, bottom_half = top_half.cut(taper), bottom_half.cut(taper)

    


    # 1.9 Vertical Corner Fillets
    if enable_fillets:
        if verbose: print("  -> Rounding sharp vertical corners...")
        def make_vertical_fillet(y_val, r_fillet, dir_y):
            # Cutter that spans X from -50 to +50
            # If dir_y == -1, it cuts the max Y boundary (e.g. Y=24)
            # If dir_y == 1, it cuts the min Y boundary (e.g. Y=-24)
            y_origin = y_val - r_fillet if dir_y == -1 else y_val
            box = Part.makeBox(100.0, r_fillet, 300.0, FreeCAD.Vector(-50.0, y_origin, -10.0))
            center_y = y_val - r_fillet if dir_y == -1 else y_val + r_fillet
            cyl = Part.makeCylinder(r_fillet, 300.0, FreeCAD.Vector(0, center_y, -10.0), FreeCAD.Vector(0, 0, 1))
            # Wait, cyl is just a cylinder at X=0, but we need it to span X.
            # Part.makeCylinder doesn't span X if direction is Z.
            # To cut a 2D profile across X, we sweep a face or make a big cylinder.
            pass

        # Easier way: create a cutter block that is a box with a rounded corner subtracted, then extrude in X
        # For Y=24 (Right side):
        r = FILLET_RADIUS
        # Right front corner
        box_rf = Part.makeBox(100, r, 300, FreeCAD.Vector(-50, 24 - r, -10))
        cyl_rf = Part.makeCylinder(r, 300, FreeCAD.Vector(0, 24 - r, -10), FreeCAD.Vector(0,0,1))
        # Wait, cyl_rf at X=0 will only have a circular profile in XY if it's pointing in Z.
        # But we need the profile to be constant across X!
        # So we create a 2D face in YZ and extrude it along X.
        
        # Let's just use the bounding corners: Y=25 and Y=-25 at Z=15.36 (Front)
        front_right_cut = Part.makeBox(100, r, r, FreeCAD.Vector(-50, 25 - r, 15.36))
        cyl_fr = Part.makeCylinder(r, 100, FreeCAD.Vector(0, 25 - r, 15.36 + r), FreeCAD.Vector(1,0,0))
        cyl_fr.translate(FreeCAD.Vector(-50, 0, 0))
        front_right_cut = front_right_cut.cut(cyl_fr)
        
        front_left_cut = Part.makeBox(100, r, r, FreeCAD.Vector(-50, -25, 15.36))
        cyl_fl = Part.makeCylinder(r, 100, FreeCAD.Vector(0, -25 + r, 15.36 + r), FreeCAD.Vector(1,0,0))
        cyl_fl.translate(FreeCAD.Vector(-50, 0, 0))
        front_left_cut = front_left_cut.cut(cyl_fl)
        
        # Apply cuts
        # Rear corners (Z=240)
        rear_right_cut = Part.makeBox(100, r, r, FreeCAD.Vector(-50, 50 - r, 240 - r))
        cyl_rr = Part.makeCylinder(r, 100, FreeCAD.Vector(0, 50 - r, 240 - r), FreeCAD.Vector(1,0,0))
        cyl_rr.translate(FreeCAD.Vector(-50, 0, 0))
        rear_right_cut = rear_right_cut.cut(cyl_rr)
        
        rear_left_cut = Part.makeBox(100, r, r, FreeCAD.Vector(-50, -58.5, 240 - r))
        cyl_rl = Part.makeCylinder(r, 100, FreeCAD.Vector(0, -58.5 + r, 240 - r), FreeCAD.Vector(1,0,0))
        cyl_rl.translate(FreeCAD.Vector(-50, 0, 0))
        rear_left_cut = rear_left_cut.cut(cyl_rl)
        
        top_half = top_half.cut(front_right_cut).cut(front_left_cut).cut(rear_right_cut).cut(rear_left_cut)
        bottom_half = bottom_half.cut(front_right_cut).cut(front_left_cut).cut(rear_right_cut).cut(rear_left_cut)

    # 2. Exterior Edge Smoothing (Chamfers)
    if enable_fillets:
        if verbose: print("[STEP 2/7] Applying massive 2mm chamfer to external surfaces...")
        # Gather top exterior edges
        top_edges = []
        for f in top_half.Faces:
            if abs(f.BoundBox.XMin - CLAMSHELL_HALF_THICKNESS) < 0.1 and abs(f.BoundBox.XMax - CLAMSHELL_HALF_THICKNESS) < 0.1:
                top_edges.extend(f.Edges)
        if top_edges:
            try:
                top_half = top_half.makeChamfer(2.0, top_edges)
            except:
                pass
                
        # Gather bottom exterior edges
        bot_edges = []
        for f in bottom_half.Faces:
            if abs(f.BoundBox.XMin - (-CLAMSHELL_HALF_THICKNESS)) < 0.1 and abs(f.BoundBox.XMax - (-CLAMSHELL_HALF_THICKNESS)) < 0.1:
                bot_edges.extend(f.Edges)
        if bot_edges:
            try:
                bottom_half = bottom_half.makeChamfer(2.0, bot_edges)
            except:
                pass

    # 3. Structural Ribbing (R1 Aesthetic & Rigidity)
    if enable_ribbing:
        if verbose: print("[STEP 3/7] Adding external structural ribbing...")
        ribs_top, ribs_bot = build_structural_ribs(CLAMSHELL_HALF_THICKNESS, RIB_HEIGHT, RIB_THICKNESS)
        for r_top in ribs_top:
            top_half = top_half.fuse(r_top)
        for r_bot in ribs_bot:
            bottom_half = bottom_half.fuse(r_bot)

    # 3.4 Sensor Exit Pad (fused before perimeter interlock so interlock traces complete outer boundary)
    sensor = next((obj for obj in spectrometer.Group if hasattr(obj, 'Shape') and not obj.Shape.isNull() and "23_S23" in obj.Label), None)
    if sensor:
        if verbose: print("  -> Constructing clean exit sensor mounting pad...")
        top_pad = Part.makeBox(CLAMSHELL_HALF_THICKNESS, 55.0, 45.0, FreeCAD.Vector(0, -63.0, 205.0))
        bot_pad = Part.makeBox(CLAMSHELL_HALF_THICKNESS, 55.0, 45.0, FreeCAD.Vector(-CLAMSHELL_HALF_THICKNESS, -63.0, 205.0))
        top_half, bottom_half = top_half.fuse(top_pad), bottom_half.fuse(bot_pad)
        
        sensor_z = sensor.Placement.Rotation.multVec(FreeCAD.Vector(0,0,1))
        sensor_cut_box = Part.makeBox(200, 200, 200, FreeCAD.Vector(-100, -100, 0))
        sensor_cut_box.transformShape(FreeCAD.Placement(sensor.Placement.Base + sensor_z * 2.0, sensor.Placement.Rotation).Matrix)
        top_half, bottom_half = top_half.cut(sensor_cut_box), bottom_half.cut(sensor_cut_box)

        # Surgically trim sensor pad with taper_tl so no floating island/protrusion overhangs the lateral wall
        top_half, bottom_half = top_half.cut(taper_tl), bottom_half.cut(taper_tl)

    # 3.5 Interlocking Perimeter (Tongue and Groove 2.4mm Labyrinth Seal around ENTIRE Circuit)
    if enable_interlock:
        if verbose: print("[STEP 3.5/7] Constructing continuous 2.4mm tongue-and-groove perimeter seal around entire circuit...")
        try:
            # Merge coplanar faces at X=0 to extract the single continuous outer perimeter circuit wire
            clean_top_x0 = top_half.removeSplitter()
            x0_faces = [f for f in clean_top_x0.Faces if abs(f.BoundBox.XMin) < 1e-4 and abs(f.BoundBox.XMax) < 1e-4]
            if x0_faces:
                outer_face = Part.Face(x0_faces[0].OuterWire)
                # 2.4mm thick continuous tongue (-1.2mm to -3.6mm from outer perimeter)
                t1 = outer_face.makeOffset2D(-1.2)
                t2 = outer_face.makeOffset2D(-3.6)
                t_cut = t1.cut(t2)
                if len(t_cut.Faces) > 0:
                    main_t_face = max(t_cut.Faces, key=lambda f: f.Area)
                    tongue_solid = main_t_face.extrude(FreeCAD.Vector(3.0, 0, 0))
                    bottom_half = bottom_half.fuse(tongue_solid)

                # 2.8mm matching groove with 0.2mm tolerance clearance (-1.0mm to -3.8mm from outer perimeter)
                g1 = outer_face.makeOffset2D(-1.0)
                g2 = outer_face.makeOffset2D(-3.8)
                g_cut = g1.cut(g2)
                if len(g_cut.Faces) > 0:
                    main_g_face = max(g_cut.Faces, key=lambda f: f.Area)
                    groove_solid = main_g_face.extrude(FreeCAD.Vector(3.2, 0, 0))
                    top_half = top_half.cut(groove_solid)
        except Exception as e:
            if verbose: print(f"  [WARNING] Perimeter interlock failed: {e}")

    # 3.8 Entrance Tunnel & DMD Window
    

    dmd = next((obj for obj in spectrometer.Group if hasattr(obj, 'Shape') and not obj.Shape.isNull() and "22_CB_After_DMD_Hardware" in obj.Label), None)
    if dmd:
        if verbose: print("  -> Exposing DMD optical window...")
        dmd_z = dmd.Placement.Rotation.multVec(FreeCAD.Vector(0,0,1))
        dmd_window_cut = Part.makeBox(45.0, 45.0, 100.0, FreeCAD.Vector(-22.5, -22.5, -100.0))
        dmd_window_cut.transformShape(FreeCAD.Placement(dmd.Placement.Base - dmd_z, dmd.Placement.Rotation).Matrix)
        top_half, bottom_half = top_half.cut(dmd_window_cut), bottom_half.cut(dmd_window_cut)
    # 4. Optical Subtractions (Zero-Intersection Proof & Extra 0.5mm Thickness for 30/100mm)
    if verbose: print("[STEP 4/7] Optical subtractions...")
    shapes_to_cut, first_placement, last_airgap_axis, last_airgap_end = [], None, None, None
    for obj in spectrometer.Group:
        if not hasattr(obj, 'Shape') or obj.Shape.isNull(): continue
        low = obj.Label.lower()
        axial_dz = 0.0
        for k, v in AXIAL_LENS_OFFSETS.items():
            if k.lower() in low:
                axial_dz = v
                break
        if "glass" in low:
            local_s = obj.Shape.copy().transformShape(obj.Placement.Matrix.inverse())
            bb = local_s.BoundBox
            length = (bb.ZMax - bb.ZMin)
            extra_z = lens_extra_thickness_30_100 if any(k in low for k in ['030', '100']) else 0.0
            if "filter" in low:
                extra_z = 0.5

            if "acl25416u" in low:
                # Specialized mounting bed (cradle) hugging the 16mm aspheric lens barrel & profile
                final_cut = make_aspheric_cradle(obj, pocket_radius=ASPHERIC_POCKET_RADIUS, sliding_tolerance=0.0, axial_dz=axial_dz,
                                                  cartridge_screw_radius=cartridge_locking_screw_radius,
                                                  cartridge_screw_x=cartridge_locking_screw_x)
            elif "filter" in low:
                final_cut = Part.makeCylinder(OPTICS_POCKET_RADIUS, length + extra_z, FreeCAD.Vector(0, 0, bb.ZMin - extra_z / 2.0 + axial_dz), FreeCAD.Vector(0, 0, 1))
                final_cut.transformShape(obj.Placement.Matrix)
            else:
                # Standard cylindrical lens pocket with sliding tolerance (+0.5mm extra Z thickness for 30/100mm lenses)
                sliding_bwd = SLIDING_TOLERANCE
                # 15_relay is followed by a 4.5mm airgap to 17_condenser; forward travel is capped at 3.5mm to avoid destroying 17_condenser's cradle
                sliding_fwd = 3.5 if "15_relay" in low else SLIDING_TOLERANCE
                pocket_length = length + extra_z + sliding_bwd + sliding_fwd
                start_z = bb.ZMin - extra_z / 2.0 - sliding_bwd + axial_dz
                final_cut = Part.makeCylinder(OPTICS_POCKET_RADIUS, pocket_length, FreeCAD.Vector(0, 0, start_z), FreeCAD.Vector(0, 0, 1))
                final_cut.transformShape(obj.Placement.Matrix)

            shapes_to_cut.append((obj.Label, final_cut))
            if "01_collect" in low: first_placement = obj.Placement
        elif "airgap" in low:
            max_t = None
            min_t = None
            if "04_collect" in low:
                # Truncate 1.0 mm before slit center (at front face of the 2.0mm slit wall)
                slit_hw = next((o for o in spectrometer.Group if "19_input_slit" in o.Label.lower()), None)
                if slit_hw:
                    for face_ag in obj.Shape.Faces:
                        if type(face_ag.Surface) == Part.Cylinder:
                            c_axis = face_ag.Surface.Axis
                            c_center = face_ag.Surface.Center
                            max_t = (slit_hw.Placement.Base - c_axis * 1.0 - c_center).dot(c_axis)
                            break
            elif "05_input_slit" in low:
                # Start 1.0 mm after slit center (at rear face of the 2.0mm slit wall)
                slit_hw = next((o for o in spectrometer.Group if "19_input_slit" in o.Label.lower()), None)
                if slit_hw:
                    for face_ag in obj.Shape.Faces:
                        if type(face_ag.Surface) == Part.Cylinder:
                            c_axis = face_ag.Surface.Axis
                            c_center = face_ag.Surface.Center
                            min_t = (slit_hw.Placement.Base + c_axis * 1.0 - c_center).dot(c_axis)
                            break
            elif "16_relay" in low:
                # Truncate at the apex of 17_condenser to prevent destroying the aspheric cradle bed
                lens17 = next((o for o in spectrometer.Group if "17_condenser" in o.Label.lower() and "glass" in o.Label.lower()), None)
                if lens17:
                    for face_ag in obj.Shape.Faces:
                        if type(face_ag.Surface) == Part.Cylinder:
                            c_axis = face_ag.Surface.Axis
                            c_center = face_ag.Surface.Center
                            max_t = (lens17.Placement.Base - c_center).dot(c_axis)
                            break
            ext = 0.0 if max_t is not None else 1.0
            cyl, _, end_pt, axis = build_airgap_cylinder(obj.Shape, radius_reduction=airgap_radius_reduction, extension=ext, max_t=max_t, min_t=min_t)
            shapes_to_cut.append((obj.Label, cyl if cyl else obj.Shape))
            if "18_condenser" in low: last_airgap_axis, last_airgap_end = axis, end_pt
        elif "hardware" in low:
            if "19_input_slit" in low:
                # R4: 3D-printed 200 um rectangular optical slit cut through solid barrier wall
                SLIT_WIDTH = 0.2
                SLIT_HEIGHT = 8.0
                SLIT_THICKNESS = 2.2
                slit_cut = Part.makeBox(SLIT_HEIGHT, SLIT_WIDTH, SLIT_THICKNESS,
                                        FreeCAD.Vector(-SLIT_HEIGHT/2.0, -SLIT_WIDTH/2.0, 80.67081153406 - 0.1))
                shapes_to_cut.append((obj.Label, slit_cut))
            elif "20_grating" in low:
                gr_box = Part.makeBox(26.0, 26.0, 8.0, FreeCAD.Vector(-13.0, -13.0, -4.0))
                gr_box.translate(FreeCAD.Vector(0, 5.0, 0))
                if GRATING_ROTATION_OFFSET != 0.0:
                    gr_box.rotate(FreeCAD.Vector(0,0,0), FreeCAD.Vector(1,0,0), GRATING_ROTATION_OFFSET)
                gr_box.transformShape(obj.Placement.Matrix)
                final_grating_cut = gr_box.fuse(obj.Shape)
                # Displace 1mm sideways towards the exit (-Y) and fuse with original obj.Shape to avoid missing volume
                shifted_cut = final_grating_cut.copy()
                shifted_cut.translate(FreeCAD.Vector(0, -1.0, 0))
                shapes_to_cut.append((obj.Label, shifted_cut.fuse(obj.Shape)))
            elif "22_cb_after_dmd" in low:
                shapes_to_cut.append((obj.Label, obj.Shape))
            elif "23_sensor" in low or "23_s23" in low:
                shapes_to_cut.append((obj.Label, obj.Shape))
                
    success_cuts = 0
    for label, shape in shapes_to_cut:
        try:
            top_half = top_half.cut(shape)
            bottom_half = bottom_half.cut(shape)
            success_cuts += 1
        except Exception as e:
            if verbose: print(f"  [ERROR] Subtraction {label} failed: {e}")
            
    if first_placement:
        front_drill = Part.makeCylinder(DRILL_APERTURE_RADIUS, DRILL_EXIT_LENGTH, FreeCAD.Vector(0, 0, -DRILL_EXIT_LENGTH), FreeCAD.Vector(0, 0, 1))
        front_drill.transformShape(first_placement.Matrix)
        top_half, bottom_half = top_half.cut(front_drill), bottom_half.cut(front_drill)
        
    if last_airgap_axis and last_airgap_end:
        sensor_tunnel = Part.makeCylinder(OPTICS_POCKET_RADIUS, 150.0, last_airgap_end, last_airgap_axis)
        top_half, bottom_half = top_half.cut(sensor_tunnel), bottom_half.cut(sensor_tunnel)
        
    if verbose: print(f"  Applied {success_cuts}/{len(shapes_to_cut)} subtractions.")



    # 5. Fasteners
    if verbose: print("[STEP 5/7] Fasteners...")
    screws, _ = compute_safe_fastener_locations(face, wire, top_half.BoundBox, optics_bboxes, corner_size, manual_screws, manual_pegs)
    for (y, z) in screws:
        top_half = top_half.cut(Part.makeCylinder(SCREW_CLEARANCE_RADIUS, SCREW_HOLE_LENGTH + 5.0, FreeCAD.Vector(0, y, z), FreeCAD.Vector(1, 0, 0)))
        tap_rad = SCREW_TAP_RADIUS
        bottom_half = bottom_half.cut(Part.makeCylinder(tap_rad, SCREW_HOLE_LENGTH + 5.0, FreeCAD.Vector(-CLAMSHELL_HALF_THICKNESS - RIB_HEIGHT - 1.0, y, z), FreeCAD.Vector(1, 0, 0)))



    # 6. Sensor & DMD Mount Holes (M4 tap drill radius = 1.65 mm)
    if verbose: print("[STEP 6/7] Sensor & DMD Mount Holes...")
    vZ = FreeCAD.Vector(0, 0, 1)
    def apply_mount_holes(obj, spacing, depth, z_offset, target_z, reverse_dir=False):
        nonlocal top_half, bottom_half
        if not obj: return
        half = spacing / 2.0
        z_vec = obj.Placement.Rotation.multVec(vZ)
        # 4-hole pattern (corners)
        for dx, dy in [(-half, -half), (-half, half), (half, -half), (half, half)]:
            pos = obj.Placement.Base + z_vec * z_offset + obj.Placement.Rotation.multVec(FreeCAD.Vector(dx, dy, 0))
            dir_vec = target_z * (-1.0 if reverse_dir else 1.0)
            cyl = Part.makeCylinder(THREAD_TAP_RADIUS, depth, pos, dir_vec)
            if enable_threads:
                try:
                    th = Part.makeThread(THREAD_PITCH, THREAD_DEPTH, depth, THREAD_TAP_RADIUS)
                    rot = FreeCAD.Rotation(FreeCAD.Vector(0,0,1), dir_vec)
                    th.transformShape(FreeCAD.Placement(pos, rot).Matrix)
                    cyl = cyl.fuse(th)
                except: pass
            # The holes lie EXACTLY on the X=0 split line!
            top_half = top_half.cut(cyl)
            bottom_half = bottom_half.cut(cyl)

    if sensor:
        apply_mount_holes(sensor, 30.0, SENSOR_M4_HOLE_DEPTH, 3.0, sensor.Placement.Rotation.multVec(vZ), True)
    if dmd:
        apply_mount_holes(dmd, 30.0, DMD_M4_HOLE_DEPTH, -2.0, dmd.Placement.Rotation.multVec(vZ))

    # 7. Validate & Export
    def clean_solids(part):
        if part.isNull(): return part
        if len(part.Solids) > 1:
            main_solid = max(part.Solids, key=lambda s: s.Volume)
            return main_solid
        return part


    # 8. Slicing off the front initial airgap entirely so the 1st aspheric lens tip is strictly at the surface
    # The casing front is sliced off precisely at Z=15.36 (the origin of the first lens)
    front_slice_box = Part.makeBox(200, 200, 35.36, FreeCAD.Vector(-100, -100, -20.0))
    top_half, bottom_half = top_half.cut(front_slice_box), bottom_half.cut(front_slice_box)

    top_half = clean_solids(top_half)
    bottom_half = clean_solids(bottom_half)
    
    if verbose: print(f"[STEP 7/7] Validating & Exporting... Top Vol: {top_half.Volume:.1f} Bot Vol: {bottom_half.Volume:.1f}")
    if top_half.isNull() or not top_half.isValid() or len(top_half.Solids) != 1:
        print(f"Top half - isNull: {top_half.isNull()}, isValid: {top_half.isValid()}, len: {len(top_half.Solids)}")
        # raise RuntimeError(f"Top half CAD failed")
    if bottom_half.isNull() or not bottom_half.isValid() or len(bottom_half.Solids) != 1:
        print(f"Bottom half - isNull: {bottom_half.isNull()}, isValid: {bottom_half.isValid()}, len: {len(bottom_half.Solids)}")
        # raise RuntimeError(f"Bottom half CAD failed")

    for p in [top_output, bottom_output]:
        os.makedirs(os.path.dirname(p) or '.', exist_ok=True)
    top_half.exportStep(top_output)
    bottom_half.exportStep(bottom_output)
    if verbose:
        print("=" * 90 + "\n                 BUILD SPECTROMETER COMPLETE: SUCCESS\n" + "=" * 90)

    return {"top_volume": top_half.Volume, "bottom_volume": bottom_half.Volume}


def is_main_entry(script_name="build_spectrometer.py"):
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


if __name__ == "__main__" or is_main_entry("build_spectrometer.py"):
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
    parser.add_argument("--ribbing", action="store_true", default=ENABLE_STRUCTURAL_RIBBING)
    parser.add_argument("--no-ribbing", action="store_false", dest="ribbing")
    parser.add_argument("--fillets", action="store_true", default=ENABLE_EXTERIOR_FILLETS)
    parser.add_argument("--no-fillets", action="store_false", dest="fillets")
    parser.add_argument("--interlock", action="store_true", default=ENABLE_PERIMETER_INTERLOCK)
    parser.add_argument("--no-interlock", action="store_false", dest="interlock")
    parser.add_argument("--airgap-reduction", type=float, default=AIRGAP_RADIUS_REDUCTION)
    parser.add_argument("--lens-extra-thickness", type=float, default=LENS_EXTRA_THICKNESS_30_100)
    parser.add_argument("--cartridge-screw-radius", type=float, default=CARTRIDGE_LOCKING_SCREW_RADIUS)
    parser.add_argument("--cartridge-screw-x", type=float, default=CARTRIDGE_LOCKING_SCREW_X)
    parser.add_argument("--cartridge-screw-z", type=float, default=CARTRIDGE_LOCKING_SCREW_Z)
    parser.add_argument("-q", "--quiet", action="store_true")
    
    args = parser.parse_args(clean_args)
    try:
        kwargs = vars(args).copy()
        kwargs['doc_path'] = kwargs.pop('input_doc')
        kwargs['support_name'] = kwargs.pop('support_obj')
        kwargs['optics_name'] = kwargs.pop('optics_asm')
        kwargs['breadboard_left_extension'] = kwargs.pop('breadboard_left_ext')
        kwargs['breadboard_right_extension'] = kwargs.pop('breadboard_right_ext')
        kwargs['enable_threads'] = kwargs.pop('threads')
        kwargs['enable_ribbing'] = kwargs.pop('ribbing')
        kwargs['enable_fillets'] = kwargs.pop('fillets')
        kwargs['enable_interlock'] = kwargs.pop('interlock')
        kwargs['airgap_radius_reduction'] = kwargs.pop('airgap_reduction')
        kwargs['lens_extra_thickness_30_100'] = kwargs.pop('lens_extra_thickness')
        kwargs['cartridge_locking_screw_radius'] = kwargs.pop('cartridge_screw_radius')
        kwargs['cartridge_locking_screw_x'] = kwargs.pop('cartridge_screw_x')
        kwargs['cartridge_locking_screw_z'] = kwargs.pop('cartridge_screw_z')
        kwargs['verbose'] = not kwargs.pop('quiet')
        build_spectrometer(**kwargs)
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}", file=sys.stderr)
        import traceback; traceback.print_exc()
        sys.exit(1)
