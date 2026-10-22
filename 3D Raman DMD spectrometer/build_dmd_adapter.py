# -*- coding: utf-8 -*-
import sys
sys.path.append(r'C:\Program Files\FreeCAD 1.1\bin')
import FreeCAD, Part

# --- PARAMETERS ---
width = 44.6  # slightly undersized from 45.0 for a slip fit into the 45.0 aperture
height = 44.6
base_thickness = 6.0  # increased to allow M4 counterbores
standoff_height = 4.0
total_thickness = base_thickness + standoff_height

# M2 Threads for the tiny IPD1231G DMD board
M2_PITCH = 0.4
M2_DEPTH = 0.2
M2_TAP_RADIUS = 0.8

# Instead of symmetric height 44.6, we trim the bottom to exactly match the casing floor (Y=-86.0)
# which projects to y_local = -19.88 on the adapter plane
y_min = -19.88
y_max = height / 2.0  # 22.3
actual_height = y_max - y_min

# The adapter is built in the -Z direction so Z=0 is the optical interface (flush with spectrometer pocket ledge)
base = Part.makeBox(width, actual_height, base_thickness, FreeCAD.Vector(-width/2, y_min, -base_thickness))

# --- DLP4500 BOARD DIMENSIONS ---
# Please measure your DLP4500 board and update these hole coordinates (relative to the DMD center at 0,0).
# These represent the locations of the standoffs that will hold the DMD board.
DLP4500_HOLE_X = 15.0  # Half-width spacing
DLP4500_HOLE_Y = 15.0  # Half-height spacing
DMD_TAP_RADIUS = 1.25  # 1.25mm radius for M2.5 screws (update if using M3 or M2)
DMD_STANDOFF_RADIUS = 2.5 # Radius of the plastic standoff post

dmd_holes = [
    ( DLP4500_HOLE_X,  DLP4500_HOLE_Y), 
    (-DLP4500_HOLE_X,  DLP4500_HOLE_Y), 
    ( DLP4500_HOLE_X, -DLP4500_HOLE_Y), 
    (-DLP4500_HOLE_X, -DLP4500_HOLE_Y)
]

standoffs = []
for dx, dy in dmd_holes:
    # Standoffs hold the board AWAY from the optical interface. 
    # If Z=0 is the optical plane (inside the spectrometer), the board is outside.
    # So the board is at Z = -base_thickness - standoff_height.
    # Therefore standoffs go from -base_thickness down to -total_thickness.
    cyl = Part.makeCylinder(DMD_STANDOFF_RADIUS, standoff_height, FreeCAD.Vector(dx, dy, -total_thickness), FreeCAD.Vector(0,0,1))
    standoffs.append(cyl)

adapter = base
for s in standoffs:
    adapter = adapter.fuse(s)

# Central optical window for the DMD
opt_hole = Part.makeBox(20.0, 15.0, total_thickness, FreeCAD.Vector(-10.0, -7.5, -total_thickness))
adapter = adapter.cut(opt_hole)

# M4 clearance holes and counterbores to mount adapter to spectrometer (+-15, +-15)
for dx in [-15.0, 15.0]:
    for dy in [-15.0, 15.0]:
        # Close Fit M4 clearance (r=2.15, diameter=4.3mm)
        h = Part.makeCylinder(2.15, base_thickness, FreeCAD.Vector(dx, dy, -base_thickness), FreeCAD.Vector(0,0,1))
        adapter = adapter.cut(h)
        
        # Standard M4 Socket Head Counterbore (r=3.75, depth=4.2)
        # Counterbore is on the exterior (most negative Z) side, so the screw head goes in.
        cb = Part.makeCylinder(3.75, 4.2, FreeCAD.Vector(dx, dy, -base_thickness), FreeCAD.Vector(0,0,1))
        adapter = adapter.cut(cb)

# Threaded holes in the standoffs
for dx, dy in dmd_holes:
    pos = FreeCAD.Vector(dx, dy, -total_thickness)
    h = Part.makeCylinder(DMD_TAP_RADIUS, total_thickness, pos, FreeCAD.Vector(0,0,1))
    adapter = adapter.cut(h)

adapter.exportStep("DMD_Adapter.step")
print("Exported DMD_Adapter.step!")
