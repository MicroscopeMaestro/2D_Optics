# Original User Request

## Initial Request — 2026-09-07T13:33:03Z

This is a single self-contained design task; keep it small and focused. 

Design a mechanical fastening system to securely attach and lock the external slip-fit lens cartridges into the main 3D spectrometer body. Perform a comprehensive final geometric audit of the entire CAD assembly (including all adapter mounting holes, optical paths, and clamshell closures) to guarantee everything is ready for 3D printing.

Working directory: C:\Users\Juan\Documents\GitHub\2D_Optics\3D Raman DMD spectrometer
Integrity mode: demo

## Requirements

### R1. Mechanical Attachment
Modify the CAD generation scripts (`build_external_bed.py` and `build_spectrometer.py`) to add a robust mechanical fastening mechanism (e.g., set screws, interlocking tabs, or flanges) that locks the external lens cartridge into the spectrometer casing. The team has full autonomy to choose the best mechanical approach. **CRITICAL:** Exercise extreme care with the external aspheric mounts; the new fasteners must not warp, break, or compromise the delicate sliding tolerances of the cartridges.

### R2. Pre-Print Geometric Audit
Perform a comprehensive script-based geometric audit of the final generated STEP files. Verify that the lens cartridges drop into the casing without unwanted collisions, and that the generic 30x30mm M4 mounting patterns align properly across the newly generated PDAPC3 and DMD adapters.

### R3. Strict Optical Path & Element Constraints
The geometric audit must rigorously verify that no new fasteners or structural changes clip the main optical path. Furthermore, the team must preserve all previously established precision constraints: the 18mm clear aperture, 26.8mm lens pocket radius, 0.15mm conic slope clearance, the top M4 lens-clamping set screw, and the 8.0mm centered grating slot.

### R4. 3D-Printed 200 µm Optical Slit
The `19_input_slit` feature in `build_spectrometer.py` must be completely rewritten. Instead of a slot for a 1-inch metal disk, it must generate a solid plastic wall blocking the optical path, with an exact `0.2 mm` (200 µm) wide rectangular slit physically cut through the center of it.

## Acceptance Criteria

### Script-Based Verification
- [ ] An automated FreeCAD headless script must be executed to verify that the external cartridge securely locks into the casing without violating any tolerances.
- [ ] An optical path validation script must output exactly `0.0 mm³` of intersection between the simulated light beams and the final assembled casing (guaranteeing zero clipping).
- [ ] The validation script must confirm that the generic 30x30mm M4 mounting patterns on the main casing perfectly align with the clearance holes on the adapters.
- [ ] A script must confirm that the physical 3D-printed input slit has an exact geometric width of 0.2 mm.
- [ ] The chosen locking mechanism must be fully parameterized in the Python code, without breaking any existing optical constraints.

## Follow-up Directive — 2026-09-08T07:09:58Z

CRITICAL DIRECTIVE FROM MAIN ORCHESTRATOR:

Do NOT segregate the external slip-fit cartridge geometry to only `01_collect`! 
The user EXPLICITLY requested that the external cartridge drop-in pockets be applied to BOTH the entrance (`01_collect`) AND the exit (`17_condenser`) aspheric lenses!

You MUST change the line:
`if USE_EXTERNAL_ASPHERIC_MOUNTS and "01_collect" in obj.Label.lower():`
back to:
`if USE_EXTERNAL_ASPHERIC_MOUNTS and "acl25416u" in obj.Label.lower():`

Also, ensure that the lateral M4 locking set screws are dynamically generated for ALL lenses, not just hardcoded to `Z = 20.0`!

Implement these changes immediately and do not revert them!

## Follow-up Directive — 2026-09-08T07:40:43Z

CRITICAL DIRECTIVE FROM MAIN ORCHESTRATOR:

Do NOT "restore concentric positioning" to XMin = -14.7 in `build_spectrometer.py`!
The coordinates `XMin = -13.0` for the cartridge (width 34.0) and `XMin = -13.7` for the pocket (width 35.4) are INTENTIONAL and correct!

The reason they were changed is because at XMin = -14.7, the 15.4-degree rotation of the exit lens caused the corner of the pocket to rotate down to `X_global = -18.33`, which punctured completely through the `-17.5 mm` floor of the spectrometer!

By raising them to `-13.0` and `-13.7` respectively, the lowest rotated corner mathematically stops at `X_global = -17.23`, keeping the pocket safely inside the casing and leaving a solid floor!

The 0.7 mm symmetric slip-fit clearance on all sides is already perfect (`-13.0 - (-13.7) = +0.7` and `21.7 - 21.0 = +0.7`).

LEAVE the origins at `-13.0` and `-13.7`! Do not restore them!

## Follow-up Directive (R4 Override) — 2026-09-08T08:01:56Z

CRITICAL OVERRIDE DIRECTIVE FROM MAIN ORCHESTRATOR / USER:

Requirement R4 HAS BEEN OFFICIALLY OVERRIDDEN BY THE USER.

The user explicitly requested: "the slit must be external extenstion that i can remove and load".

Therefore:
1. DO NOT reconstruct the solid plastic wall!
2. DO NOT cut a 0.2 mm slit through the casing!
3. The massive 33.4 x 23.4 x 4.4 mm hollow drop-in pocket at `19_input_slit` is the CORRECT and FINAL implementation. The 0.2mm slit is now located in a standalone cartridge (`build_slit_cartridge.py`), which drops into this pocket.

The Victory Auditor is working off OUTDATED user requirements for R4. You must FORCE the auditor to accept the hollow drop-in pocket as the correct implementation of the slit.

Additionally:
- The IndentationError on line 649 was already fixed in the main thread. Pull the latest `build_spectrometer.py`.
- Please DO fix the Top screw #3 at (17.0, 85.0) wall thickness!

Do not revert the drop-in pocket!
