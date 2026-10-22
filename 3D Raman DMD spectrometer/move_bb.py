with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

import re

# We will remove the old bb_slab creation from Step 2, and add it AFTER Step 2.8!
# Find bb_slab creation
bb_pattern = re.compile(r'      # 2\. Add Breadboard Slab\n.*?      bottom_half = bottom_half\.fuse\(bb_slab\)\n', re.DOTALL)
content = bb_pattern.sub('', content)

# Now find the end of Step 2.8
sensor_trim_pattern = re.compile(r'(          top_half, bottom_half = top_half\.cut\(trim_box\), bottom_half\.cut\(trim_box\)\n)', re.DOTALL)

new_bb_code = '''\1
      # 2.9 (Moved) Add Breadboard Slab matching the FINAL bounding box!
      if verbose: print(f"[STEP 2.9/7] Adding {breadboard_thickness}mm breadboard slab matching final footprint...")
      bbox = top_half.BoundBox
      bb_x_min = -(CLAMSHELL_HALF_THICKNESS + breadboard_thickness)
      # Make it perfectly rectangular by taking the bounding box, and DO NOT apply corner fillets to it.
      bb_slab = Part.makeBox(breadboard_thickness, bbox.YMax - bbox.YMin, bbox.ZMax - bbox.ZMin, FreeCAD.Vector(bb_x_min, bbox.YMin, bbox.ZMin))
      bottom_half = bottom_half.fuse(bb_slab)
'''

content = sensor_trim_pattern.sub(new_bb_code, content)

with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Moved bb_slab!")
