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

# Compound of all optics
optics_compound = b3d.Compound(children=optical_solids)

# Compound of all air volumes
air_compound = b3d.Compound(children=air_solids)

# Bounding box of the entire optical layout
bbox = (optics_compound + air_compound).bounding_box()
print(f"Assembly Bounding Box:")
print(f"  X: [{bbox.min.X:.1f}, {bbox.max.X:.1f}] (span={bbox.size.X:.1f} mm)")
print(f"  Y: [{bbox.min.Y:.1f}, {bbox.max.Y:.1f}] (span={bbox.size.Y:.1f} mm)")
print(f"  Z: [{bbox.min.Z:.1f}, {bbox.max.Z:.1f}] (span={bbox.size.Z:.1f} mm)")

# Create an enclosure blank around the system with 10mm wall margin
margin = 10.0
box_dx = bbox.size.X + 2 * margin
box_dy = bbox.size.Y + 2 * margin
box_dz = bbox.size.Z + 2 * margin
center_pt = bbox.center()

with b3d.BuildPart() as p_bench:
    b3d.Box(box_dx, box_dy, box_dz)
enclosure_blank = p_bench.part.locate(b3d.Location((center_pt.X, center_pt.Y, center_pt.Z)))

# Perform smart 3D boolean subtraction:
# Subtraction of continuous Air Cavity + Lens Pockets!
print("Subtracting air tunnels and optical mounting pockets from solid enclosure block...")
hollow_bench = enclosure_blank - (air_compound + optics_compound)
print("Hollow enclosure created successfully! Volume:", hollow_bench.volume)
print("Euler characteristic chi:", hollow_bench.wrapped.Closed())

b3d.export_step(hollow_bench, 'scratch/test_3d_printable_enclosure.step')
sz = os.path.getsize('scratch/test_3d_printable_enclosure.step')
print(f"Exported 3D Printable Enclosure STEP file size: {sz:,} bytes")
