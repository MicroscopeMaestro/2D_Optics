with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_grating = '''            elif "20_grating" in low:
                gr_box = Part.makeBox(26.0, 26.0, 6.5, FreeCAD.Vector(-13.0, -13.0, -3.25))
                gr_box.translate(FreeCAD.Vector(0, 5.0, 0))
                gr_box.transformShape(obj.Placement.Matrix)
                shapes_to_cut.append((obj.Label, gr_box.fuse(obj.Shape)))'''

new_grating = '''            elif "20_grating" in low:
                gr_box = Part.makeBox(26.0, 26.0, 6.5, FreeCAD.Vector(-13.0, -13.0, -3.25))
                gr_box.translate(FreeCAD.Vector(0, 5.0, 0))
                gr_box.transformShape(obj.Placement.Matrix)
                final_grating_cut = gr_box.fuse(obj.Shape)
                # Displace 1mm sideways towards the exit (-Y)
                final_grating_cut.translate(FreeCAD.Vector(0, -1.0, 0))
                shapes_to_cut.append((obj.Label, final_grating_cut))'''

content = content.replace(old_grating, new_grating)

with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Displaced grating by 1mm towards exit!")
