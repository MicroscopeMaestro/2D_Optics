import sys
FREECAD_BIN = r"C:\Program Files\FreeCAD 1.1\bin"
if FREECAD_BIN not in sys.path:
    sys.path.append(FREECAD_BIN)
import FreeCAD
import Part

bot = Part.Shape()
bot.read("support_bottom_half.step")

print("Bot BB:", bot.BoundBox)

# Check all cylindrical faces in bot
for i, f in enumerate(bot.Faces):
    if type(f.Surface) == Part.Cylinder:
        print(f"Face {i}: R={f.Surface.Radius:.2f}, Center={f.Surface.Center}, Axis={f.Surface.Axis}, BB={f.BoundBox}")
