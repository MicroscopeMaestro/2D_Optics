import sys, os
sys.path.insert(0, os.path.abspath('.'))

import build123d as b3d
from optics_app.engine.lde import SurfaceStack
from optics_app.cad.exporter import (
    get_exportable_elements, create_solid_lens, create_mirror_or_planar_solid,
    create_aperture_stop_solid, get_effective_pose, R_mat_to_euler_degrees
)

res = SurfaceStack.load_from_json(r'C:\Users\Juan\Desktop\spectrometer_raman_50mm_100mm_manual_optimized.json')
stack = res['stack']
stack._update_accumulated_positions()

elems = get_exportable_elements(stack)
print(f"Total elements: {len(elems)}")

optical_solids = []
air_solids = []

for elem in elems:
    etype = elem['type']
    surfs = elem['surfaces']
    part_3d = None
    if etype == 'Doublet':
        s1, s2, s3 = surfs[0], surfs[1], surfs[2]
        e1 = create_solid_lens(s1, s2)
        e2_local = create_solid_lens(s2, s3)
        if e1 is not None and e2_local is not None:
            part_3d = e1 + e2_local.locate(b3d.Location((0, 0, s1.thickness)))
    elif etype == 'Singlet':
        part_3d = create_solid_lens(surfs[0], surfs[1])
    elif etype in ['Mirror', 'DMD', 'Detector', 'Grating']:
        part_3d = create_mirror_or_planar_solid(surfs[0])
    elif etype == 'Aperture Stop':
        part_3d = create_aperture_stop_solid(surfs[0])
    elif etype in ['Air Gap', 'Air Volume']:
        s1, s2 = surfs[0], surfs[1]
        sd = max(s1.semi_diameter, s2.semi_diameter, 2.0)
        t = max(s1.thickness, 0.001)
        with b3d.BuildPart() as p:
            b3d.Cylinder(radius=sd, height=t)
        part_3d = p.part
        
    if part_3d is not None:
        eff_origin, eff_R = get_effective_pose(surfs[0])
        loc = tuple(eff_origin)
        rot_deg = R_mat_to_euler_degrees(eff_R)
        placed = part_3d.locate(b3d.Location(loc, rot_deg))
        placed.label = elem['name']
        if etype in ['Air Gap', 'Air Volume']:
            air_solids.append(placed)
        else:
            optical_solids.append(placed)

print(f"Optical solids created: {len(optical_solids)}")
print(f"Air gap solids created: {len(air_solids)}")

# Test fusing all air solids into a continuous tunnel
total_air = air_solids[0]
for a in air_solids[1:]:
    try:
        total_air = total_air + a
    except Exception as e:
        print(f"Fuse join note on {a.label}: {e}")

print("Fused air tunnel volume:", getattr(total_air, 'volume', 0.0))

# Export both as STEP to check enclosure subtraction readiness
b3d.export_step(b3d.Compound(children=optical_solids + air_solids), 'scratch/test_smart_air_path.step')
sz = os.path.getsize('scratch/test_smart_air_path.step')
print(f"Exported combined STEP file size: {sz:,} bytes")
