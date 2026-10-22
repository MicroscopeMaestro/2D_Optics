import re

path = 'optics_app/cad/exporter.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

old_logic = '''def create_aperture_stop_solid(surf, thickness=1.5):
    """
    Generate 3D solid diaphragm / baffle plate with central clear aperture hole for stops and apertures.
    """
    b3d = _get_build123d()
    if b3d is None:
        return None
    t = float(surf.substrate_thickness if getattr(surf, 'substrate_thickness', 0) > 0 else (surf.thickness if surf.thickness > 0 else thickness))
    t = max(t, 1.0)
    is_rect = getattr(surf, 'aperture_type', '') in ["Rectangular", "Slit"] or getattr(surf, 'substrate_shape', '') == "Rectangular"
    try:
        with b3d.BuildPart() as part:
            if is_rect:
                hx = max(float(surf.aperture_x_half if getattr(surf, 'aperture_x_half', 0) > 0 else surf.semi_diameter), 0.5)
                hy = max(float(surf.aperture_y_half if getattr(surf, 'aperture_y_half', 0) > 0 else surf.semi_diameter), 0.5)
                ox = max(hx * 1.5, hx + 5.0)
                oy = max(hy * 1.5, hy + 5.0)
                with b3d.BuildSketch(b3d.Plane.XY):
                    b3d.Rectangle(2.0 * ox, 2.0 * oy)
                    b3d.Rectangle(2.0 * hx, 2.0 * hy, mode=b3d.Mode.SUBTRACT)
                b3d.extrude(amount=-t)
            else:
                r_in = max(float(surf.semi_diameter), 0.5)
                r_out = max(r_in * 1.5, r_in + 5.0)
                with b3d.BuildSketch(b3d.Plane.XY):
                    b3d.Circle(radius=r_out)
                    b3d.Circle(radius=r_in, mode=b3d.Mode.SUBTRACT)
                b3d.extrude(amount=-t)
        return part.part
    except Exception:'''

new_logic = '''def create_aperture_stop_solid(surf, thickness=1.5):
    """
    Generate 3D solid diaphragm / baffle plate with central clear aperture hole for stops and apertures.
    """
    b3d = _get_build123d()
    if b3d is None:
        return None
    t = float(surf.substrate_thickness if getattr(surf, 'substrate_thickness', 0) > 0 else (surf.thickness if surf.thickness > 0 else thickness))
    t = max(t, 1.0)
    is_rect = getattr(surf, 'aperture_type', '') in ["Rectangular", "Slit"] or getattr(surf, 'substrate_shape', '') == "Rectangular"
    try:
        sd = max(float(getattr(surf, 'semi_diameter', 12.7)), 0.1)
        with b3d.BuildPart() as part:
            if is_rect:
                hx = max(float(getattr(surf, 'aperture_x_half', 0.0)), 0.001)
                hy = max(float(getattr(surf, 'aperture_y_half', 0.0)), 0.001)
                if hx == 0.001 and hy == 0.001:
                    hx, hy = sd, sd
                with b3d.BuildSketch(b3d.Plane.XY):
                    b3d.Circle(radius=sd)
                    b3d.Rectangle(2.0 * hx, 2.0 * hy, mode=b3d.Mode.SUBTRACT)
                b3d.extrude(amount=-t)
            else:
                r_in = float(getattr(surf, 'aperture_r_outer', 0.0))
                if r_in <= 0.0:
                    r_in = sd
                r_out = max(sd, r_in + 0.001)
                with b3d.BuildSketch(b3d.Plane.XY):
                    b3d.Circle(radius=r_out)
                    if r_in > 0:
                        b3d.Circle(radius=r_in, mode=b3d.Mode.SUBTRACT)
                b3d.extrude(amount=-t)
        return part.part
    except Exception:'''

content = content.replace(old_logic, new_logic)
with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
print('Replaced')
