with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

# We need to add reinforcement blocks in step 1.
add_blocks = '''
    # Reinforce entrance lens area (Y=0, Z=10 to 30)
    ent_top = Part.makeBox(CLAMSHELL_HALF_THICKNESS, 42.0, 30.0, FreeCAD.Vector(0, -21.0, 5.0))
    ent_bot = Part.makeBox(CLAMSHELL_HALF_THICKNESS, 42.0, 30.0, FreeCAD.Vector(-CLAMSHELL_HALF_THICKNESS, -21.0, 5.0))
    top_half, bottom_half = top_half.fuse(ent_top), bottom_half.fuse(ent_bot)

    # Reinforce exit lens area (Y=-37, Z=205 to 225)
    ex_top = Part.makeBox(CLAMSHELL_HALF_THICKNESS, 42.0, 30.0, FreeCAD.Vector(0, -58.0, 200.0))
    ex_bot = Part.makeBox(CLAMSHELL_HALF_THICKNESS, 42.0, 30.0, FreeCAD.Vector(-CLAMSHELL_HALF_THICKNESS, -58.0, 200.0))
    top_half, bottom_half = top_half.fuse(ex_top), bottom_half.fuse(ex_bot)
'''

content = content.replace('# 1.5 Enclose DMD optical path', add_blocks + '\n    # 1.5 Enclose DMD optical path')

with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Added reinforcement blocks!")
