import re

with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_fillet = '''          for x in [10.0, -10.0]:
              if x > 0:
                  top_half = top_half.cut(fillet_tool)
              else:
                  # Restrict the fillet tool so it ONLY cuts the casing, not the bb_slab at X < -17.5
                  casing_region = Part.makeBox(CLAMSHELL_HALF_THICKNESS, 20.0, 300.0, FreeCAD.Vector(-CLAMSHELL_HALF_THICKNESS, y-10.0, -10))
                  fillet_tool = fillet_tool.common(casing_region)
                  bottom_half = bottom_half.cut(fillet_tool)'''

new_fillet = '''          top_half = top_half.cut(fillet_tool)
          # Mirror the fillet tool to cut the bottom half symmetrically since the breadboard slab was removed!
          mirror_mat = FreeCAD.Matrix()
          mirror_mat.scale(-1, 1, 1) # Mirror across X=0
          fillet_bot = fillet_tool.copy().transformShape(mirror_mat)
          bottom_half = bottom_half.cut(fillet_bot)'''

content = content.replace(old_fillet, new_fillet)

with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Fixed fillet tool symmetry")
