with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

chamfer_cuts = '''
    # --- MANUAL CORNER CHAMFERS TO REMOVE SHARP EDGES ---
    if verbose: print("  -> Slicing off sharp exterior corner edges...")
    
    def make_chamfer_cut(y1, z1, y2, z2, y_out, z_out1, z_out2):
        poly = Part.makePolygon([
            FreeCAD.Vector(0, y1, z1), FreeCAD.Vector(0, y2, z2),
            FreeCAD.Vector(0, y_out, z_out2), FreeCAD.Vector(0, y_out, z_out1), FreeCAD.Vector(0, y1, z1)
        ])
        return Part.Face(poly).extrude(FreeCAD.Vector(100, 0, 0)).translate(FreeCAD.Vector(-50, 0, 0))

    # 1. Right side taper corner at Y=22, Z=175
    cc1 = make_chamfer_cut(22, 168, 17, 178, 30, 165, 185)
    
    # 2. Left entrance corner at Y=-22, Z=50
    cc2 = make_chamfer_cut(-22, 45, -27, 53, -10, 40, 60)
    
    # 3. Left DMD bottom corner at Y=-110, Z=95
    cc3 = make_chamfer_cut(-105, 92, -110, 100, -125, 85, 105)
    
    # 4. Left DMD top corner at Y=-110, Z=132
    cc4 = make_chamfer_cut(-110, 127, -105, 137, -125, 120, 145)

    top_half = top_half.cut(cc1).cut(cc2).cut(cc3).cut(cc4)
    bottom_half = bottom_half.cut(cc1).cut(cc2).cut(cc3).cut(cc4)
'''

# Insert this right before "# 2. Applying massive 2mm chamfer"
anchor = "    # 2. Applying massive 2mm chamfer"
content = content.replace(anchor, chamfer_cuts + "\n" + anchor)

with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Injected corner chamfers!")
