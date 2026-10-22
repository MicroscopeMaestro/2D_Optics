import sys
sys.path.append(r'C:\Program Files\FreeCAD 1.1\bin')
import FreeCAD, Part

shape = Part.Shape()
shape.read('support_top_half.step')

# Let's test a range of cut parameters:
# Y_cut from -45 to -30
# Z_cut from 118 to 130

for y_cut in [-45.0, -42.0, -40.0, -38.0, -35.0]:
    for z_cut in [118.0, 120.0, 122.0, 125.0]:
        box = Part.makeBox(25.0, (y_cut - (-110.0)), (z_cut - 94.0), FreeCAD.Vector(-2, -110, 94))
        dmd = shape.common(box)
        rest = shape.cut(box)
        
        # Check M4 holes in dmd:
        holes = []
        for f in dmd.Faces:
            if type(f.Surface) == Part.Cylinder and 1.5 <= f.Surface.Radius <= 1.8:
                holes.append(f.Surface.Center)
                
        # Check solids
        valid = (len(dmd.Solids) == 1 and len(rest.Solids) == 1 and len(holes) == 2)
        if valid:
            print(f"PASS: y_cut={y_cut:5.1f}, z_cut={z_cut:5.1f} | dmd_vol={dmd.Volume:8.1f}, rest_vol={rest.Volume:8.1f} | holes={len(holes)}")
        else:
            print(f"FAIL: y_cut={y_cut:5.1f}, z_cut={z_cut:5.1f} | dmd_solids={len(dmd.Solids)}, rest_solids={len(rest.Solids)}, holes={len(holes)}")
