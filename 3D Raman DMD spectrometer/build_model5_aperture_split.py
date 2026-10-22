"""Partition the existing model 5 top without changing its design or holes."""
from pathlib import Path
import json
import FreeCAD as App
import Part

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'model5_aperture_split'


def box(y, z, dy, dz):
    return Part.makeBox(60, dy, dz, App.Vector(-20, y, z))


def validate(shape):
    assert shape.isValid() and shape.isClosed() and len(shape.Solids) == 1
    assert shape.Volume > 0


def main():
    OUT.mkdir(exist_ok=True)
    top = Part.read(str(ROOT / 'support_top_half_new.step')).removeSplitter()
    # Compact aperture region plus narrow extensions retaining the nearest
    # existing housing mounts at (Y,Z)=(-76,136) and (-26,122).
    # The DMD's two existing mount axes are at local X=+15, Y=+/-15.
    mask = box(-100, 90, 58, 32).fuse(box(-82, 120, 12, 21)).fuse(
        box(-43, 116, 22, 12))
    dmd = top.common(mask).removeSplitter()
    main_top = top.cut(mask).removeSplitter()
    validate(top)
    validate(dmd)
    validate(main_top)
    volume_error = abs(dmd.Volume + main_top.Volume - top.Volume)
    assert volume_error < 0.01
    # Since the two pieces are complementary common/cut operations on the same
    # source and tool, no material is added, removed, or overlapped by the split.
    print('Split validated; volume error:', volume_error, flush=True)
    doc = App.newDocument('Model5ApertureSplit')
    for name, shape in [('Top_DMD_aperture', dmd), ('Top_remaining', main_top)]:
        path = OUT / (name + '.step')
        shape.exportStep(str(path))
        check = Part.read(str(path))
        validate(check)
        assert abs(shape.Volume - check.Volume) < 0.01
        obj = doc.addObject('PartDesign::Feature', name)
        obj.Shape = shape
        print(name, shape.Volume, 'STEP round-trip passed', flush=True)
    doc.recompute()
    doc.saveAs(str(OUT / 'Top_two_sections.FCStd'))
    Part.makeCompound([dmd, main_top]).exportStep(str(OUT / 'Top_two_sections.step'))
    report = {
        'source': 'support_top_half_new.step',
        'operation': 'Split only; no changes to existing geometry or holes; zero seam gap',
        'source_volume_mm3': top.Volume,
        'dmd_volume_mm3': dmd.Volume,
        'remaining_volume_mm3': main_top.Volume,
        'volume_error_mm3': volume_error,
        'dmd_share_percent': dmd.Volume / top.Volume * 100,
        'retained_housing_mount_axes_YZ_mm': [[-76, 136], [-26, 122]],
        'retained_DMD_mounts_local_XY_mm': [[15, -15], [15, 15]],
        'checks': ['One valid closed solid per section', 'Source volume preserved',
                   'Both STEP exports reimport as valid solids with matching volume'],
    }
    (OUT / 'validation.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
