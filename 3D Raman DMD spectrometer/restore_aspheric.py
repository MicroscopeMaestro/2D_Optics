import re

with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_aspheric = '''def get_aspheric_pocket(lens_shape, placement, clearance_scale=1.01, aperture_radius=11.0):
    # Standard optical "ring contact" mount for aspheric lenses!
    # Instead of molding a matching curve (which causes 3D printing tolerance issues),
    # we trap the convex lens against a sharp circular ledge at Z=8.4 (where R=11.0).
    
    # Outer pocket for the lens body (Radius 12.8mm, from Z=8.4 to Z=14.5)
    lens_body = Part.makeCylinder(12.8, 14.5 - 8.4, FreeCAD.Vector(0, 0, 8.4))
    
    # Airgaps for the light path (Radius 11.0mm)
    front_airgap = Part.makeCylinder(aperture_radius, 8.4 + 15.0, FreeCAD.Vector(0, 0, -15.0))
    back_airgap = Part.makeCylinder(aperture_radius, 20.0, FreeCAD.Vector(0, 0, 14.0))
    
    pocket = lens_body.fuse([front_airgap, back_airgap])
    pocket.transformShape(placement.Matrix)
    return pocket'''

new_aspheric = '''def get_aspheric_pocket(lens_shape, placement, clearance_scale=1.01, aperture_radius=11.0):
    # User requested a smooth conic surface to hold the lenses in the bottom section.
    # We will use the exact Thorlabs STEP geometry scaled by 1.01x to carve a perfect negative mold!
    
    # PRE-ROTATE the Thorlabs STEP file so its optical axis (Y) aligns with local Z!
    rot = FreeCAD.Rotation(FreeCAD.Vector(1,0,0), 90)
    rotated_lens = lens_shape.copy().transformShape(FreeCAD.Placement(FreeCAD.Vector(), rot).Matrix)

    mat = FreeCAD.Matrix()
    mat.scale(clearance_scale, clearance_scale, clearance_scale)
    scaled = rotated_lens.transformShape(mat)
    z_offset = -14.0 * (clearance_scale - 1.0) / 2.0
    scaled.translate(FreeCAD.Vector(0, 0, z_offset))
    
    front_airgap = Part.makeCylinder(aperture_radius, 27.0, FreeCAD.Vector(0, 0, -15.0))
    z_back = 14.0 * clearance_scale + z_offset
    back_airgap = Part.makeCylinder(aperture_radius, 20.0, FreeCAD.Vector(0, 0, z_back - 2.0))
    
    try:
        pocket = scaled.fuse([front_airgap, back_airgap])
    except:
        pocket = scaled.fuse(front_airgap).fuse(back_airgap)
        
    pocket.transformShape(placement.Matrix)
    return pocket'''

if old_aspheric in content:
    content = content.replace(old_aspheric, new_aspheric)
    print("Replaced successfully!")
else:
    print("Failed to replace!")

with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
    f.write(content)
