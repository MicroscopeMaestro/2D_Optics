"""Package and verify the already-built clearance STEP parts."""
from pathlib import Path
import json
import FreeCAD as App
import Part

root = Path(__file__).resolve().parent
out = root/'model5_top_lower_clearance'
doc = App.newDocument('TopTwoSectionsClearance')
parts=[]
checks=[]
for label in ['Top_DMD_aperture','Top_remaining']:
    brep=Part.Shape(); brep.read(str(out/(label+'_clearance.brep')))
    exported=Part.read(str(out/(label+'_clearance.step')))
    assert exported.isValid() and exported.isClosed() and len(exported.Solids)==1
    delta=abs(exported.Volume-brep.Volume)
    assert delta < max(0.01,brep.Volume*1e-5)
    obj=doc.addObject('PartDesign::Feature',label)
    obj.Shape=exported
    parts.append(exported)
    checks.append({'part':label,'valid_single_closed_solid':True,
                   'volume_mm3':exported.Volume,'STEP_volume_delta_mm3':delta,
                   'STEP_volume_delta_ppm':delta/brep.Volume*1e6,
                   'minimum_X_mm':exported.BoundBox.XMin})
    print(label,'verified',checks[-1],flush=True)
doc.recompute()
doc.saveAs(str(out/'Top_two_sections_clearance.FCStd'))
Part.makeCompound(parts).exportStep(str(out/'Top_two_sections_clearance.step'))
report={'source':'model5_aperture_split',
        'lower_reference':'support_bottom_half_new.step','lower_modified':False,
        'clearance_offset_mm':0.25,'clearance_recess_depth_mm':3.25,
        'DMD_minimum_clearance_before_STEP_mm':0.2499999999838018,
        'main_minimum_clearance':'Nominal construction offset; full distance calculation not completed',
        'interference_verification':'Before STEP export, the lower lip had zero volume outside the subtracted clearance cutter',
        'original_projection_depth_mm':3.0,
        'mounting_hole_axes_and_split':'Unchanged',
        'STEP_precision_note':'Main-part STEP healing changes volume by about 2.5 ppm and leaves a tolerant junction 0.0061 mm below the nominal mating plane; the original 3 mm projecting ribs are removed.',
        'parts':checks}
(out/'validation.json').write_text(json.dumps(report,indent=2))
print('Clearance assembly and validation saved.',flush=True)
