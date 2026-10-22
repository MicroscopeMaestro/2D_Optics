import sys
sys.path.append(r'C:\Program Files\FreeCAD 1.1\bin')
import FreeCAD, Part

top = Part.Shape()
top.read('temp_top_no_screws.step')
bot = Part.Shape()
bot.read('temp_bot_no_screws.step')

print(f"Top bounds: {top.BoundBox}")
print(f"Bot bounds: {bot.BoundBox}")

# Pre-filter candidate points
candidates = []
for z in range(18, 233, 2):
    for y in range(-85, 25, 2):
        pt_top = FreeCAD.Vector(8.75, float(y), float(z))
        pt_bot = FreeCAD.Vector(-8.75, float(y), float(z))
        if top.isInside(pt_top, 1e-3, False) and bot.isInside(pt_bot, 1e-3, False):
            candidates.append((float(y), float(z)))

print(f"Pre-filter candidate points: {len(candidates)}")

safe_points = []
for y, z in candidates:
    cyl_top = Part.makeCylinder(3.75, 17.5, FreeCAD.Vector(0, y, z), FreeCAD.Vector(1, 0, 0))
    top_diff = cyl_top.Volume - top.common(cyl_top).Volume
    if top_diff > 1e-3:
        continue
    
    # In bottom half, tap hole radius is 1.65 mm, sleeve radius 3.15 mm
    cyl_bot = Part.makeCylinder(3.15, 17.5, FreeCAD.Vector(-17.5, y, z), FreeCAD.Vector(1, 0, 0))
    bot_diff = cyl_bot.Volume - bot.common(cyl_bot).Volume
    if bot_diff > 1e-3:
        continue
    
    safe_points.append((y, z))

print(f"Total fully safe points with >= 1.5mm wall margin in BOTH halves: {len(safe_points)}")

# Group safe points into zones:
zones = {
    'entrance': [p for p in safe_points if p[1] < 45],
    'mid_right': [p for p in safe_points if 45 <= p[1] < 170 and p[0] > 0],
    'mid_left': [p for p in safe_points if 45 <= p[1] < 170 and p[0] < -40],
    'mid_center': [p for p in safe_points if 45 <= p[1] < 170 and -40 <= p[0] <= 0],
    'pre_exit': [p for p in safe_points if 170 <= p[1] < 215],
    'exit': [p for p in safe_points if p[1] >= 215]
}

for zname, pts in zones.items():
    print(f"Zone '{zname}': {len(pts)} points. Sample: {pts[:10]}")
