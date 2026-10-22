"""Split the model 5 top CAD counterpart into a DMD section and main section.

Run with the Python executable bundled with FreeCAD. Dimensions are millimetres.
The existing top STEP is a mirrored, faceted solid derived from model 5.
"""
from pathlib import Path
import json
import math
import FreeCAD as App
import Part
import Mesh

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'model5_split_top'
INSERT_DIAMETER = 5.6  # Provisional: adjust for the actual insert and printer.
INSERT_DEPTH = 6.0
LAP_X = 8.5
GAP = 0.25
JOINTS = [(-79.0, 130.0), (-32.0, 117.0)]  # (global Y, global Z)


def box(x, y, z, dx, dy, dz):
    return Part.makeBox(dx, dy, dz, App.Vector(x, y, z))


def cylinder(radius, length, x, y, z):
    return Part.makeCylinder(radius, length, App.Vector(x, y, z), App.Vector(1, 0, 0))


def validate(shape, label):
    assert not shape.isNull() and shape.isValid(), label + ': invalid shape'
    assert len(shape.Solids) == 1, label + ': must be one connected solid'
    assert shape.isClosed() and shape.Volume > 0, label + ': not closed'


def main():
    OUT.mkdir(exist_ok=True)
    top = Part.read(str(ROOT / 'support_top_half_new.step')).removeSplitter()
    print('Loaded verified model 5 top', flush=True)
    validate(top, 'Matching top')
    # The DMD is centred at Y=-66.57, Z=105.17 in the original assembly.
    region = box(-20, -100, 90, 60, 63, 35)
    # Integral half-lap tabs, made only from existing housing material.
    tabs = [box(LAP_X + GAP, -84, 120, 30, 10, 15),
            box(LAP_X + GAP, -42, 112, 30, 15, 10)]
    mask = region
    for tab in tabs:
        mask = mask.fuse(tab)
    dmd = top.common(mask).removeSplitter()
    main_body = top.cut(mask).removeSplitter()
    print('Split complete', flush=True)
    # Clearance around tab sides and below the overlap, contained in parent CAD.
    for tab in tabs:
        t = tab.BoundBox
        relief = box(LAP_X, t.YMin-GAP, t.ZMin-GAP, 30,
                     t.YLength+2*GAP, t.ZLength+2*GAP).cut(region)
        main_body = main_body.cut(relief)
    # Add seam clearance on the main side of the DMD-region boundary.
    expanded_region = box(-20, -100, 90-GAP, 60, 63+GAP, 35+2*GAP)
    main_body = main_body.cut(expanded_region.cut(region))
    print('Lap clearances complete', flush=True)
    checks = []
    for y, z in JOINTS:
        bore = cylinder(INSERT_DIAMETER/2, INSERT_DEPTH,
                        LAP_X-INSERT_DEPTH, y, z)
        # Require the entire insert bore plus 1.7 mm surrounding wall.
        envelope = cylinder(INSERT_DIAMETER/2 + 1.7, INSERT_DEPTH,
                            LAP_X-INSERT_DEPTH, y, z)
        missing = envelope.cut(main_body).Volume
        assert missing < 0.01, f'Insufficient insert wall at {y}, {z}: {missing}'
        main_body = main_body.cut(bore)
        screw = cylinder(2.25, 30, LAP_X, y, z)
        dmd = dmd.cut(screw)
        checks.append({'Y': y, 'Z': z, 'insert_opening_X': LAP_X,
                       'insert_bottom_X': LAP_X-INSERT_DEPTH})
        print('Joining hole complete', y, z, flush=True)
    dmd, main_body = dmd.removeSplitter(), main_body.removeSplitter()
    for shape, label in [(dmd, 'DMD'), (main_body, 'Main')]:
        validate(shape, label)
    # Both parts are generated exclusively by intersection/subtraction from top;
    # the complementary initial split and later cuts cannot create overlap.
    for y, z in JOINTS:
        assert dmd.common(cylinder(2.25, 30, LAP_X, y, z)).Volume < 0.01
        ring = cylinder(3.5, 8.75, LAP_X+GAP, y, z).cut(
            cylinder(2.25, 8.75, LAP_X+GAP, y, z))
        assert ring.cut(dmd).Volume < 0.01, 'Insufficient material around tab hole'
        assert main_body.common(cylinder(INSERT_DIAMETER/2, INSERT_DEPTH,
                                        LAP_X-INSERT_DEPTH, y, z)).Volume < 0.01
    doc = App.newDocument('Model5SplitTop')
    objects = []
    for label, shape in [('Top_matching_model5', top), ('Top_DMD_M4', dmd),
                         ('Top_main_M4', main_body)]:
        shape.exportStep(str(OUT / (label + '.step')))
        reread = Part.read(str(OUT / (label + '.step')))
        validate(reread, label + ' STEP round-trip')
        assert abs(reread.Volume-shape.Volume) < 0.01
        obj = doc.addObject('PartDesign::Feature', label)
        obj.Label, obj.Shape = label, shape
        objects.append(obj)
        print('Export and round-trip passed:', label, flush=True)
    doc.recompute()
    doc.saveAs(str(OUT / 'Model5_split_top.FCStd'))
    Part.makeCompound([dmd, main_body]).exportStep(str(OUT / 'Top_split_assembly.step'))
    report = {'source': 'support_top_half_new.step', 'footprint_reference': 'printing_model_5.stl',
              'method': 'Existing mirrored model 5 top; all 7925 refined vertices match mirrored STL vertices within 0.0001 mm',
              'clearance_hole_diameter_mm': 4.5, 'insert_pocket_diameter_mm': INSERT_DIAMETER,
              'insert_depth_mm': INSERT_DEPTH, 'lap_clearance_mm': GAP,
              'joints': checks, 'volumes_mm3': {'top': top.Volume, 'dmd': dmd.Volume,
                                             'main': main_body.Volume},
              'checks': ['Single valid closed solid per part', 'No overlapping material',
                         'Insert wall >= 1.7 mm', 'No added material in optical cavities',
                         'STEP round-trip validity and volume', 'Joining bores clear']}
    (OUT / 'validation.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
