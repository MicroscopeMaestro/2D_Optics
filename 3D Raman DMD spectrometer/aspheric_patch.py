def get_aspheric_pocket(lens_shape, placement, clearance_scale=1.01, aperture_radius=11.0):
    mat = FreeCAD.Matrix()
    mat.scale(clearance_scale, clearance_scale, clearance_scale)
    scaled = lens_shape.copy().transformShape(mat)
    # Re-center Z after scaling
    scaled.translate(FreeCAD.Vector(0, 0, -14.0 * (clearance_scale - 1.0) / 2.0))
    # Light aperture
    aperture = Part.makeCylinder(aperture_radius, 40.0, FreeCAD.Vector(0, 0, -15.0))
    pocket = scaled.fuse(aperture)
    pocket.transformShape(placement.Matrix)
    return pocket
