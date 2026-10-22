import sys
sys.path.append(r'C:\Program Files\FreeCAD 1.1\bin')
import FreeCAD, Part

# We want to trace the EXACT bounding boxes of the optics and just keep a uniform wall thickness!
# Since the script applies a fillet, if we just cut away massive polygons, the fillet will smooth it out perfectly.

print("Let's define the cutting polygons.")
