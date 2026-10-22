# Model 5 split top

Source: `support_top_half_new.step`, the existing mirrored counterpart of
`printing_model_5.stl`. All 7,925 vertices remaining after coplanar refinement
match the mirrored STL vertices to 0.0001 mm. The source is a faceted solid;
its curved surfaces remain faceted in these STEP exports.

## Files

- `Top_matching_model5.step`: the complete source top, with coplanar faces refined.
- `Top_DMD_M4.step`: the removable DMD section, including two integral lap tabs.
- `Top_main_M4.step`: the rest of the top, with insert pockets under the tabs.
- `Top_split_assembly.step`: both new sections in their original coordinates.
- `Model5_split_top.FCStd`: the complete reference top and the two new sections.
  Hide the reference top when viewing the split parts to avoid superposition.
- `validation.json`: dimensions and completed geometry checks.
- `split_top_preview.png`: assembled and exploded views.

## Joint design

Two M4 screws run along global X, through Ø4.5 mm clearance holes in the
DMD-section tabs. Their Y/Z centres are (-79, 130) and (-32, 117) mm.
The main section contains provisional Ø5.6 mm × 6 mm insert pockets,
opening at X = 8.5 mm and ending at X = 2.5 mm. Install the inserts from
the exposed lap faces before fitting the DMD section. The lap clearance
is 0.25 mm. Screw heads bear on the exterior surface; no counterbores
were added. The nominal tab thickness is 8.75 mm, so select screw length
for that thickness, any washer, and the actual insert's usable thread depth.

The DMD section occupies the housing within Y ≤ -37 mm and Z = 90–125 mm,
plus its two joining tabs. Both sections keep the source coordinate system.
Material is only removed from the existing top; optical cavities and the
source's existing clamshell features are retained except where intersected
by the new split and joint cuts.

The insert pockets are provisional, as requested. Check their fit against
the purchased inserts and a small printed test coupon before printing the
whole housing. This work verifies the two new top sections; it does not
certify the original mirrored top's fit to an already printed bottom.

Rebuild with the FreeCAD-bundled Python executable and
`build_model5_split_top.py` in the parent directory. Render the preview with
`render_model5_split_top.py` after the build finishes.
