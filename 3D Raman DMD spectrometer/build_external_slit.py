import sys
sys.path.append(r'C:\Program Files\FreeCAD 1.1\bin')
import FreeCAD, Part

# Create the main block: 40x40mm square, 10mm thick
width = 40.0
height = 40.0
thickness = 10.0
block = Part.makeBox(width, height, thickness, FreeCAD.Vector(-width/2, -height/2, 0))

# Create the pocket for the 1-inch slit disc (Diameter 25.5mm, depth 2.5mm)
slit_radius = 25.5 / 2.0
slit_depth = 2.5
pocket = Part.makeCylinder(slit_radius, slit_depth, FreeCAD.Vector(0, 0, thickness - slit_depth), FreeCAD.Vector(0, 0, 1))

# Create the clear aperture light tunnel (Diameter 20mm) so it doesn't clip the beam
clear_aperture_radius = 10.0
tunnel = Part.makeCylinder(clear_aperture_radius, thickness, FreeCAD.Vector(0, 0, 0), FreeCAD.Vector(0, 0, 1))

# Cut the pocket and the tunnel out of the block
mount = block.cut(pocket).cut(tunnel)

# Export to STEP
mount.exportStep("external_slit_mount.step")
print("Exported standalone external_slit_mount.step!")
