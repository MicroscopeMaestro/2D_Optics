import re

with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace the corner fillet block completely
pattern = re.compile(r'# 2\.5 Corners & Entrance.*?entrance_tunnel = Part\.makeCylinder', re.DOTALL)

new_fillet = '''# 2.5 Corners & Entrance
      if verbose: print("[STEP 2.5/7] Rounding corners and opening entrance tunnel...")
      
      # We want the breadboard slab to be perfectly rectangular, so we ONLY round the casing!
      for x in [CLAMSHELL_HALF_THICKNESS, -CLAMSHELL_HALF_THICKNESS]:
          for y in [25.0, -80.0]:
              sign_x, sign_y = (-1 if x > 0 else 1), (-1 if y > 0 else 1)
              box = Part.makeBox(5.0, 5.0, 300.0, FreeCAD.Vector(x + (0 if sign_x==1 else -5.0), y + (0 if sign_y==1 else -5.0), -10))
              cyl = Part.makeCylinder(5.0, 300.0, FreeCAD.Vector(x + sign_x*5.0, y + sign_y*5.0, -10), FreeCAD.Vector(0,0,1))
              fillet_tool = box.cut(cyl)
              if x > 0:
                  top_half = top_half.cut(fillet_tool)
              else:
                  # Restrict the fillet tool so it ONLY cuts the casing, not the bb_slab at X < -17.5
                  casing_region = Part.makeBox(CLAMSHELL_HALF_THICKNESS, 20.0, 300.0, FreeCAD.Vector(-CLAMSHELL_HALF_THICKNESS, y-10.0, -10))
                  fillet_tool = fillet_tool.common(casing_region)
                  bottom_half = bottom_half.cut(fillet_tool)

      # Also, to ensure NO islands stick out at Y < -80, we must remove anything below Y=-80 that isn't the bb_slab!
      bot_trim = Part.makeBox(200.0, 200.0, 300.0, FreeCAD.Vector(-CLAMSHELL_HALF_THICKNESS, -280.0, -10))
      # but wait, the DMD housing is there!
      
      entrance_tunnel = Part.makeCylinder'''

content = pattern.sub(new_fillet, content)

with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Patched with regex")
