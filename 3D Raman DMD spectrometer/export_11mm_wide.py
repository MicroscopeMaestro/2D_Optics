import sys
sys.path.append(r'C:\Program Files\FreeCAD 1.1\bin')
import FreeCAD, Part

doc = FreeCAD.open('model1.FCStd')
asm = doc.getObject('Optical_System_Assembly')
lens_obj = next(o for o in asm.Group if "01_collect_acl25416u" in o.Label.lower() and "glass" in o.Label.lower())
local_s = lens_obj.Shape.copy().transformShape(lens_obj.Placement.Matrix.inverse())
bb = local_s.BoundBox

width = 36.0
height = 32.0
z_min = -0.10
z_max = 10.693
thickness = z_max - z_min

p_radius = 13.40
shelf_radius = 13.00
aperture_radius = 10.50

pts = [
    FreeCAD.Vector(0, 0, z_min),
    FreeCAD.Vector(aperture_radius, 0, z_min),
    FreeCAD.Vector(aperture_radius, 0, 1.40),
    FreeCAD.Vector(shelf_radius, 0, 8.60),
    FreeCAD.Vector(p_radius, 0, 8.60),
    FreeCAD.Vector(p_radius, 0, z_max),
    FreeCAD.Vector(0, 0, z_max),
    FreeCAD.Vector(0, 0, z_min)
]
wire = Part.makePolygon(pts)
face = Part.Face(wire)
cutout = face.revolve(FreeCAD.Vector(0, 0, 0), FreeCAD.Vector(0, 0, 1), 360)

box = Part.makeBox(width, height, thickness, FreeCAD.Vector(-14.0, -16.0, z_min))
mount = box.cut(cutout)

# 3mm lip on the large aperture surface
lip_h = 3.0
lip_x_max = -p_radius + lip_h  # -10.40 mm
lip_thick_x = lip_x_max - (-14.0)  # 3.60 mm
lip_tz = 1.00  # 1.0 mm thick lip on the surface
lip_z_start = z_max - lip_tz
lip_box = Part.makeBox(lip_thick_x, 20.0, lip_tz, FreeCAD.Vector(-14.0, -10.0, lip_z_start))
mount = mount.fuse(lip_box)

# Top M4 set screw
cb_depth = 4.5
cb_radius = 3.6
cb = Part.makeCylinder(cb_radius, cb_depth, FreeCAD.Vector(22.0, 0, bb.ZMax - 1.5), FreeCAD.Vector(-1, 0, 0))
mount = mount.cut(cb)

th_depth = 5.0
th_radius = 1.65
m4_hole = Part.makeCylinder(th_radius, th_depth + 2.0, FreeCAD.Vector(22.0 - cb_depth, 0, bb.ZMax - 1.5), FreeCAD.Vector(-1, 0, 0))
mount = mount.cut(m4_hole)

# Lateral guide channels
channel_depth = 1.2
channel_width = 5.0
channel_x_center = 4.0
channel_z_min = 2.0
channel_z_max = 10.0

right_channel = Part.makeBox(channel_width, channel_depth + 0.1, channel_z_max - channel_z_min,
                             FreeCAD.Vector(channel_x_center - channel_width / 2.0, 16.0 - channel_depth, channel_z_min))
left_channel = Part.makeBox(channel_width, channel_depth + 0.1, channel_z_max - channel_z_min,
                            FreeCAD.Vector(channel_x_center - channel_width / 2.0, -16.0 - 0.1, channel_z_min))
mount = mount.cut(right_channel).cut(left_channel)

output_step = "external_aspheric_bed_11mm_thick_wide.step"
mount.exportStep(output_step)
print(f"Exported: {output_step}")
