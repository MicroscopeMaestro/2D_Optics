# -*- coding: utf-8 -*-
import sys
sys.path.append(r'C:\Program Files\FreeCAD 1.1\bin')
import FreeCAD, Part

width = 50.0
height = 50.0
base_thickness = 5.0
standoff_height = 5.0
total_thickness = base_thickness + standoff_height

THREAD_PITCH = 0.7
THREAD_DEPTH = 0.35
THREAD_TAP_RADIUS = 1.65

base = Part.makeBox(width, height, base_thickness, FreeCAD.Vector(-width/2, -height/2, 0))

standoffs = []
for dx in [-15.0, 15.0]:
    for dy in [-15.0, 15.0]:
        cyl = Part.makeCylinder(3.5, standoff_height, FreeCAD.Vector(dx, dy, base_thickness), FreeCAD.Vector(0,0,1))
        standoffs.append(cyl)

adapter = base
for s in standoffs:
    adapter = adapter.fuse(s)

opt_hole = Part.makeCylinder(10.0, total_thickness, FreeCAD.Vector(0,0,0), FreeCAD.Vector(0,0,1))
adapter = adapter.cut(opt_hole)

def make_threaded_hole(pos, depth):
    cyl = Part.makeCylinder(THREAD_TAP_RADIUS, depth, pos, FreeCAD.Vector(0,0,1))
    try:
        th = Part.makeThread(THREAD_PITCH, THREAD_DEPTH, depth, THREAD_TAP_RADIUS)
        th.translate(pos)
        cyl = cyl.fuse(th)
    except: pass
    return cyl

# Clearance radius 2.2mm for M4
for dx in [-15.0, 15.0]:
    for dy in [-15.0, 15.0]:
        h = Part.makeCylinder(2.2, total_thickness, FreeCAD.Vector(dx, dy, 0), FreeCAD.Vector(0,0,1))
        adapter = adapter.cut(h)

for x in [-20, 0, 20]:
    for y in [-20, 0, 20]:
        if abs(x) < 16 and abs(y) < 16: continue
        th_hole = make_threaded_hole(FreeCAD.Vector(x, y, 0), base_thickness)
        adapter = adapter.cut(th_hole)

adapter.exportStep("PDAPC3_Adapter.step")
print("Exported PDAPC3_Adapter.step with clearance holes!")
