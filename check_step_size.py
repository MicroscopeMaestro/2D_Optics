import build123d as bd
from build123d import import_step

assembly = import_step('C:/Users/Juan/Documents/GitHub/2D_Optics/Final_Assembly_40mm_100mm_Labeled.step')
for child in assembly.children:
    bbox = child.bounding_box()
    dx = bbox.max.X - bbox.min.X
    dy = bbox.max.Y - bbox.min.Y
    print(f"{child.label:40} Diameter: {max(dx, dy):.2f}")
