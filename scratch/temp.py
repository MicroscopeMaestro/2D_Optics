def create_cylinder_air_gap(surf1, surf2):
    b3d = _get_build123d()
    if b3d is None:
        return None
    
    sd = max(surf1.semi_diameter, surf2.semi_diameter)
    t = max(surf1.thickness, 0.001)
    
    with b3d.BuildPart() as part:
        b3d.Cylinder(radius=max(sd, 1.0), height=t)
    return part.part
