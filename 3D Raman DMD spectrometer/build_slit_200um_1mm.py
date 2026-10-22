"""Export and verify the 200 um slit with a 1 mm maximum overall thickness."""
from pathlib import Path
import json
import FreeCAD as App
import Part
import MeshPart
from build_slit_cartridge import build_slit_cartridge

root = Path(__file__).resolve().parent
step_path = root / 'external_slit_cartridge_200um_1mm.step'
shape = build_slit_cartridge(200, str(step_path), web_thickness=0.5, thickness=1.0)
assert shape.isValid() and shape.isClosed() and len(shape.Solids) == 1
bounds = shape.BoundBox
assert abs(bounds.ZLength - 1.0) < 1e-7
assert abs(bounds.XLength - 32.0) < 1e-7
assert abs(bounds.YLength - 22.0) < 1e-7
# Measure the opening in a section through the membrane, not the wider relief.
section = shape.section(Part.makePlane(60, 60, App.Vector(-30, -30, -0.25)))
section_wires = [Part.Wire(edges) for edges in Part.sortEdges(section.Edges)]
opening = [wire for wire in section_wires
           if abs(wire.BoundBox.XLength - 0.2) < 1e-6
           and abs(wire.BoundBox.YLength - 8.0) < 1e-6]
assert len(opening) == 1, 'Expected one 0.2 x 8 mm optical slit'
slit = Part.makeBox(0.2, 8, 2, App.Vector(-0.1, -4, -1))
assert shape.common(slit).Volume < 1e-8, 'Optical opening is obstructed'
reread = Part.read(str(step_path))
assert reread.isValid() and len(reread.Solids) == 1
assert abs(reread.Volume - shape.Volume) < 1e-7
mesh = MeshPart.meshFromShape(Shape=shape, LinearDeflection=0.01,
                              AngularDeflection=0.1, Relative=False)
assert mesh.isSolid()
mesh.write(str(root / 'external_slit_200um_1mm.stl'))
report = {'overall_dimensions_mm': [bounds.XLength, bounds.YLength, bounds.ZLength],
          'slit_width_um': 200, 'slit_length_mm': 8,
          'membrane_thickness_mm': 0.5, 'volume_mm3': shape.Volume,
          'checks': ['Valid single closed solid', 'Maximum thickness 1 mm',
                     'Measured 0.2 x 8 mm membrane opening', 'Unobstructed optical opening',
                     'STEP round-trip volume match', 'Closed STL mesh']}
(root / 'slit_200um_1mm_validation.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
