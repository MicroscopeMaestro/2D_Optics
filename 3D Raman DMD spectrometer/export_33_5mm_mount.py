import sys
sys.path.append(r'C:\Program Files\FreeCAD 1.1\bin')
import FreeCAD, Part

doc = FreeCAD.open('model1.FCStd')
asm = doc.getObject('Optical_System_Assembly')
lens_obj = next(o for o in asm.Group if "01_collect_acl25416u" in o.Label.lower() and "glass" in o.Label.lower())
local_s = lens_obj.Shape.copy().transformShape(lens_obj.Placement.Matrix.inverse())
bb = local_s.BoundBox

# User requested: change the 32 mm dimension to 33.5 mm!
width = 36.00   # X dimension (standard wide mount width from -14.0 to +22.0 mm)
height = 33.50  # Y dimension (changed from 32.0 mm to 33.5 mm!)
z_min = -0.10
z_max = 10.693
thickness = z_max - z_min  # 10.793 mm (~11 mm)

p_radius = 13.40
shelf_radius = 13.00
aperture_radius = 10.50

# Conical structure inside:
z_shelf = 7.60          # Shifted forward by 1.0 mm to add 1 mm more spacing between shelf and slit
z_taper_start = 0.40    # Shifted forward by 1.0 mm to preserve 7.20 mm conical taper

pts = [
    FreeCAD.Vector(0, 0, z_min),
    FreeCAD.Vector(aperture_radius, 0, z_min),
    FreeCAD.Vector(aperture_radius, 0, z_taper_start),
    FreeCAD.Vector(shelf_radius, 0, z_shelf),
    FreeCAD.Vector(p_radius, 0, z_shelf),
    FreeCAD.Vector(p_radius, 0, z_max),
    FreeCAD.Vector(0, 0, z_max),
    FreeCAD.Vector(0, 0, z_min)
]
wire = Part.makePolygon(pts)
face = Part.Face(wire)
cutout = face.revolve(FreeCAD.Vector(0, 0, 0), FreeCAD.Vector(0, 0, 1), 360)

# Box width = 36.0 mm, height = 33.5 mm:
x_min = -14.00
box = Part.makeBox(width, height, thickness, FreeCAD.Vector(x_min, -height/2.0, z_min))
mount = box.cut(cutout)

# 3mm lip on the large aperture rear surface (at z_max = 10.693):
lip_h = 3.0
lip_x_max = -p_radius + lip_h  # -10.40 mm
lip_thick_x = lip_x_max - x_min  # -10.40 - (-14.00) = 3.60 mm
lip_tz = 0.60                    # Thinner retaining slit (0.60 mm thick along Z)
lip_z_start = z_max - lip_tz     # 10.093 mm (strictly inside the box, flush with rear surface)
lip_box = Part.makeBox(lip_thick_x, 20.0, lip_tz, FreeCAD.Vector(x_min, -10.0, lip_z_start))
mount = mount.fuse(lip_box)

# Top M4 set screw (XMax = 22.0, screw enters from X=22.0)
cb_depth = 4.5
cb_radius = 3.6
screw_z = 8.193
cb = Part.makeCylinder(cb_radius, cb_depth, FreeCAD.Vector(22.0, 0, screw_z), FreeCAD.Vector(-1, 0, 0))
mount = mount.cut(cb)

th_depth = 5.0
th_radius = 1.65
m4_hole = Part.makeCylinder(th_radius, th_depth + 2.0, FreeCAD.Vector(22.0 - cb_depth, 0, screw_z), FreeCAD.Vector(-1, 0, 0))
mount = mount.cut(m4_hole)

# Lateral guide channels
channel_depth = 1.2
channel_width = 5.0
channel_x_center = 4.0
channel_z_min = 2.0
channel_z_max = 10.0

right_channel = Part.makeBox(channel_width, channel_depth + 0.1, channel_z_max - channel_z_min,
                             FreeCAD.Vector(channel_x_center - channel_width / 2.0, height/2.0 - channel_depth, channel_z_min))
left_channel = Part.makeBox(channel_width, channel_depth + 0.1, channel_z_max - channel_z_min,
                            FreeCAD.Vector(channel_x_center - channel_width / 2.0, -height/2.0 - 0.1, channel_z_min))
mount = mount.cut(right_channel).cut(left_channel)

output_step = "external_aspheric_bed_33_5mm_width.step"
mount.exportStep(output_step)

print(f"Exported: {output_step}")
print(f"Bounding Box: {mount.BoundBox}")
print(f"Width (X): {mount.BoundBox.XLength:.3f} mm")
print(f"Height (Y): {mount.BoundBox.YLength:.3f} mm")
print(f"Thickness (Z): {mount.BoundBox.ZLength:.3f} mm")
print(f"Collision with lens: {mount.common(local_s).Volume:.6f} mm^3")
print(f"Large aperture shelf Z: {z_shelf:.2f} mm")
print(f"Large aperture pocket depth: {z_max - z_shelf:.3f} mm")
print(f"Spacing between shelf and slit: {lip_z_start - z_shelf:.3f} mm")
