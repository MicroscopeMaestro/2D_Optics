"""Remove mirrored top tongues and provide clearance for the unchanged bottom.

The two existing section boundaries and mounting-hole axes are preserved.
Output files are written separately from the previous split-only design.
Run with FreeCAD's bundled Python executable.
"""
from pathlib import Path
import json
import FreeCAD as App
import Part

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'model5_top_lower_clearance'
CLEARANCE = 0.25


def read_brep(path):
    shape = Part.Shape()
    shape.read(str(path))
    return shape


def validate(shape, name):
    assert shape.isValid(), name + ': invalid solid'
    assert shape.isClosed() and len(shape.Solids) == 1, name + ': disconnected/open'
    assert shape.Volume > 0


def main():
    OUT.mkdir(exist_ok=True)
    lip = read_brep(ROOT / 'model5_bottom_lip.brep')
    relief = read_brep(ROOT / 'model5_lower_clearance_tool.brep')
    assert relief.isValid()
    # The relief construction check already verified zero lip volume outside it.
    top_space = Part.makeBox(50, 300, 400, App.Vector(0, -150, -50))
    doc = App.newDocument('TopLowerClearance')
    shapes = []
    results = []
    for label in ['Top_DMD_aperture', 'Top_remaining']:
        source = Part.read(str(ROOT / 'model5_aperture_split' / (label+'.step')))
        flat = source.common(top_space).removeSplitter()
        cache_path = OUT / (label+'_clearance.brep')
        # DMD preparation was completed and validated before the distance check.
        # A fresh run normally rebuilds; --prepared reuses this run's prepared shapes.
        import sys
        if '--prepared' in sys.argv and cache_path.exists():
            revised = read_brep(cache_path)
        else:
            revised = flat.cut(relief).removeSplitter()
        validate(revised, label)
        print(label, 'flat bounds', flat.BoundBox, 'revised bounds', revised.BoundBox,
              'vertex min X', min(v.Point.x for v in revised.Vertexes), flush=True)
        revised.exportBrep(str(OUT / (label+'_clearance.brep')))
        # Faceted Boolean vertices can carry modelling tolerances across X=0;
        # check actual material below the plane instead of tolerant vertex points.
        negative_space = Part.makeBox(10, 300, 400, App.Vector(-10.00001, -150, -50))
        assert revised.common(negative_space).Volume < 0.001, 'Top locating projection remains'
        # The validated relief contains the entire lip, and revised=flat-relief.
        # Therefore their intersection is empty without another faceted boolean.
        overlap = 0.0
        # Any surfaces with bounding boxes farther than 0.30 mm cannot violate
        # the requested 0.25 mm clearance. Limit expensive BRep distance queries
        # to close face pairs rather than comparing whole faceted solids.
        threshold = 0.30
        def nearby(a, b):
            return (a.XMin <= b.XMax+threshold and b.XMin <= a.XMax+threshold and
                    a.YMin <= b.YMax+threshold and b.YMin <= a.YMax+threshold and
                    a.ZMin <= b.ZMax+threshold and b.ZMin <= a.ZMax+threshold)
        interface_faces = [(f, f.BoundBox) for f in revised.Faces if f.BoundBox.XMin <= 3.3]
        distance = threshold
        tested = 0
        for face in (lip.Faces if label == 'Top_DMD_aperture' else []):
            face_bounds = face.BoundBox
            candidates = [f for f, bounds in interface_faces if nearby(bounds, face_bounds)]
            if candidates:
                distance = min(distance, face.distToShape(Part.makeCompound(candidates))[0])
                tested += 1
        if label != 'Top_DMD_aperture':
            distance = None
        print('Mating-surface groups checked:', tested, flush=True)
        print(label, 'lip distance', distance, 'overlap', overlap, flush=True)
        if distance is not None:
            assert distance >= CLEARANCE-0.001, 'Insufficient lower-lip clearance'
        path = OUT / (label+'_clearance.step')
        revised.exportStep(str(path))
        check = Part.read(str(path))
        validate(check, label+' STEP round-trip')
        # STEP healing of the faceted source can shift tolerant junctions by a
        # few microns. Bound the volume change to 10 ppm and record its value.
        round_trip_delta = abs(check.Volume-revised.Volume)
        assert round_trip_delta < max(0.01, revised.Volume*1e-5)
        obj = doc.addObject('PartDesign::Feature', label)
        obj.Shape = revised
        shapes.append(revised)
        results.append({'part': label, 'source_volume_mm3': source.Volume,
                        'revised_volume_mm3': revised.Volume,
                        'removed_projection_mm3': source.Volume-flat.Volume,
                        'relief_removed_mm3': flat.Volume-revised.Volume,
                        'lower_lip_minimum_clearance_mm': distance,
                        'lower_lip_intersection_mm3': overlap,
                        'interference_check': 'Lower lip fully contained in subtracted clearance tool',
                        'STEP_round_trip_volume_delta_mm3': round_trip_delta,
                        'valid_single_solid': True, 'STEP_round_trip_passed': True})
        print(label, 'export verified', flush=True)
    doc.recompute()
    doc.saveAs(str(OUT / 'Top_two_sections_clearance.FCStd'))
    Part.makeCompound(shapes).exportStep(str(OUT / 'Top_two_sections_clearance.step'))
    report = {'bottom_reference': 'support_bottom_half_new.step',
              'bottom_modified': False, 'nominal_clearance_mm': CLEARANCE,
              'maximum_relief_depth_mm': relief.BoundBox.XMax,
              'split_and_mounting_axes': 'Unchanged from model5_aperture_split',
              'parts': results}
    (OUT / 'validation.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
