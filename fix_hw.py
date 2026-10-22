with open('v2/engine/cad_export.py', 'r') as f:
    content = f.read()

old_if = '        if surf.surface_type in [SurfaceType.GRATING, SurfaceType.DETECTOR, SurfaceType.MIRROR] or "DMD" in surf.name.upper() or "SLIT" in surf.name.upper():'
new_if = '        if not getattr(surf, "is_coord_break", False) and (surf.surface_type in [SurfaceType.GRATING, SurfaceType.DETECTOR, SurfaceType.MIRROR] or "DMD" in surf.name.upper() or "SLIT" in surf.name.upper()):'

if old_if in content:
    content = content.replace(old_if, new_if)
    with open('v2/engine/cad_export.py', 'w') as f:
        f.write(content)
