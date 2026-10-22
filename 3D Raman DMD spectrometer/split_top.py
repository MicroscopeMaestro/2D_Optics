import sys, math
sys.path.append(r'C:\Program Files\FreeCAD 1.1\bin')
import FreeCAD, Part

def split_diagonally(input_file, out1, out2):
    shape = Part.Shape()
    shape.read(input_file)
    bbox = shape.BoundBox
    
    cx = (bbox.XMin + bbox.XMax) / 2.0
    cy = (bbox.YMin + bbox.YMax) / 2.0
    cz = (bbox.ZMin + bbox.ZMax) / 2.0
    
    # We use a 45 degree angle to split it diagonally
    angle_deg = 45.0
    
    plc = FreeCAD.Placement(FreeCAD.Vector(cx, cy, cz), FreeCAD.Rotation(FreeCAD.Vector(1,0,0), angle_deg))
    
    # Box 1 for part 1
    box1 = Part.makeBox(1000, 1000, 1000, FreeCAD.Vector(-500, -500, 0))
    box1 = box1.transformShape(plc.Matrix)
    
    # Box 2 for part 2
    box2 = Part.makeBox(1000, 1000, 1000, FreeCAD.Vector(-500, -500, -1000))
    box2 = box2.transformShape(plc.Matrix)
    
    part1 = shape.common(box1)
    part2 = shape.common(box2)
    
    print(f"Shape Volume: {shape.Volume:.2f}")
    print(f"Part 1 Volume: {part1.Volume:.2f}")
    print(f"Part 2 Volume: {part2.Volume:.2f}")
    
    part1.exportStep(out1)
    part2.exportStep(out2)
    print(f"Exported {out1} and {out2}")

split_diagonally('support_top_half_m4.step', 'support_top_half_m4_part1.step', 'support_top_half_m4_part2.step')
