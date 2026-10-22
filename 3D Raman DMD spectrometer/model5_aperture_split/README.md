# Model 5 top: split only

These files partition `support_top_half_new.step` into two sections. The existing
design is unchanged: no drilling, insert pockets, lap tabs, or clearance cuts.
Existing holes retain their original dimensions. The bottom is unchanged.

- `Top_DMD_aperture.step`: small aperture section retaining the two DMD mounts
  and the two nearest existing top-to-bottom mounts.
- `Top_remaining.step`: the complementary remaining section.
- `Top_two_sections.step`: both sections assembled in the original coordinates.
- `Top_two_sections.FCStd`: native FreeCAD document containing exactly two parts.
- `validation.json`: topology, STEP reimport, and volume-preservation results.

The aperture section includes narrow extensions to retain the existing housing
holes at global (Y,Z) = (-76,136) and (-26,122) mm. It also retains the original
DMD holes at DMD-local (X,Y) = (15,-15) and (15,15) mm.

The two new interface faces have zero gap: they reproduce the original top
exactly. Faceted source surfaces and all original mating features are retained.
No correction of the source top-to-bottom fit is included in this split-only
operation. These outputs supersede the earlier lap-joint proposal for this task.

Rebuild using FreeCAD's bundled Python and `build_model5_aperture_split.py`
in the parent folder.
