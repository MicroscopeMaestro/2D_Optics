# Top sections with clearance for the lower housing

These are the revised versions of the two parts in `model5_aperture_split`.

- `Top_DMD_aperture_clearance.step`: small DMD section.
- `Top_remaining_clearance.step`: remaining top section.
- `Top_two_sections_clearance.step`: both sections in assembly coordinates.
- `Top_two_sections_clearance.FCStd`: editable FreeCAD assembly of the two solids.
- `validation.json`: measured lower-lip clearance and solid/export checks.

The top's mirrored projecting locating ribs below the X=0 mating plane are
removed. Clearance recesses above X=0 accommodate the raised lip of the unchanged
`support_bottom_half_new.step`. The recess outline includes both the base and
tip outlines of the lower lip, with a nominal 0.25 mm lateral offset; recess
depth is 3.25 mm for the 3 mm lip. The DMD section's minimum clearance was
measured against the actual lower lip. For the main section, 0.25 mm is the
nominal construction offset; its full minimum-distance calculation was not
completed. Interference exclusion follows from the validated clearance cutter
containing the entire lower lip, with that cutter subtracted from each top part.

The existing two-section split, mounting-hole axes, optical apertures and outer
dimensions are preserved outside this mating-interface edit. No insert holes or
new fastener holes were added. The lower housing and earlier files are unchanged.

Source surfaces remain faceted as in the original model 5 STEP files.
