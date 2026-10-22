# Original User Request

## Initial Request — 2026-08-26T17:42:36Z

You are the Project Orchestrator for the Spectrometer Optical Layout Optimization project.

## Your Identity & Workspace
- Working directory: `c:/Users/Juan/Documents/GitHub/2D_Optics/.agents/orchestrator_v2_2/`
- Workspace root: `c:/Users/Juan/Documents/GitHub/2D_Optics`
- Target application directory: `c:/Users/Juan/Documents/GitHub/2D_Optics/v2`
- Authoritative request file: `c:/Users/Juan/Documents/GitHub/2D_Optics/.agents/ORIGINAL_REQUEST.md`

## Verification Resources
- Baseline layout: `C:/Users/Juan/Desktop/Spectrometer_V2_Optimized.json`
- Ray tracing and intersection engine: `c:/Users/Juan/Documents/GitHub/2D_Optics/v2/engine/tracer.py`

## Project Mission & Requirements
An automated optimization of a desktop spectrometer's optical layout (comprising a collimator, diffraction grating, DMD, and a pupil-imaging relay). The final layout must fully respect the physical mechanical bounds of all components (no ray intersections with lens housings/mounts) and achieve an uncrossed geometrical path that perfectly images a 30 µm input slit onto a 2.5 mm radius detector.

### R1. Automated Layout Optimizer Script
Write a dedicated Python optimization script (`optimize_layout.py`) that calculates the optimal tilt angles for the M1 Collimator, Diffraction Grating, and DMD to achieve a completely uncrossed, collision-free geometry (e.g., a Z-shaped optical path).

### R2. Mechanical Clearance Constraints
The script must explicitly calculate the spatial boundaries of the marginal rays and ensure the beam absolutely clears all physical housings. Specifically, the beam returning from the grating must not hit the input slit housing, and the beam heading to the detector must not hit the M1 mirror or camera lens edges. The script must mathematically verify that the beam clearance at global Z=0 (Slit plane) and Z=50 (M1 plane) exceeds physical component radii (e.g., > 12.7 mm clearance for M1).

### R3. Output Alignment & JSON Generation
The script must correctly align the Y-decenter of the backend relay lenses (Field, Condenser, LightPipe) so they perfectly swallow the aberrated 1.225 µm chief ray. The script must generate and output the final configuration as a valid JSON file (`Spectrometer_V2_Optimized.json`) that is 100% compatible and readable by the new 2D Optics V2 application. Extreme marginal rays must be traced and verified with `traced_ray.vignetted == False` across all elements.

## Acceptance Criteria
1. Running `optimize_layout.py` executes successfully and generates/updates `C:/Users/Juan/Desktop/Spectrometer_V2_Optimized.json`.
2. The generated `.json` file strictly adheres to the schema expected by the 2D Optics V2 application and can be loaded seamlessly.
3. Programmatic verification function inside the script traces extreme marginal rays and explicitly asserts `traced_ray.vignetted == False` across all elements.
4. Script mathematically verifies beam clearance at global Z=0 (Slit plane) and Z=50 (M1 plane) exceeds physical component radii (> 12.7 mm for M1).

## Follow-up — 2026-08-31T09:40:27Z

# Teamwork Project Prompt — Draft

> Status: Ready for launch — awaiting user approval
> Goal: Craft prompt → get user approval → delegate to teamwork_preview
> Requested team: [none — teamwork routes from the description]

Design an optical spectrometer layout using only mirrors after the entrance slit. The system must map light onto a DMD (DMD4500), and then relay that 5 mm area entirely into a single 0.8 mm diameter detector, using off-the-shelf Thorlabs components (including a 600 lines/mm grating) and an aperture stop to control beam size.

Working directory: C:/Users/Juan/Documents/GitHub/2D_Optics/mirror_spectrometer
Integrity mode: demo

## Requirements

### R1. All-Mirror Design
The spectrometer (from the slit onwards) must use exclusively off-the-shelf Thorlabs spherical mirrors for focusing and relaying the light. Lenses may only be used before the slit (if at all).

### R2. System Specifications
The system must include an aperture stop, a 600 lines/mm grating, and a DMD4500. It must successfully map the light to a 5 mm active area on the DMD, and then relay that area down into a single 0.8 mm diameter detector.

### R3. Base Implementation
The team should use the existing `Spectrometer_V2_Zemax.json` file as a starting point, modifying it to replace all lenses after the slit with spherical mirrors.

## Acceptance Criteria

### Verification
- [ ] A valid `.json` layout file is generated.
- [ ] The local `v2/engine/tracer.py` engine successfully traces the system without collision or unhandled syntax errors.
- [ ] A programmatic script (or the tracer output) mathematically proves that 100% of the light arriving at the final image plane fits entirely within the 0.8 mm diameter (0.4 mm radius) detector footprint.

## Follow-up — 2026-08-31T12:58:28Z

# Teamwork Project Prompt

> Requested team: Full Team

Redesign the imaging relay (from the DMD to the single 0.8 mm detector) in the spectrometer layout to eliminate vignetting and fit the full spectral footprint into the detector, using any optimal mix of Thorlabs lenses and mirrors. Additionally, evaluate and output a second layout swapping the 600 lines/mm grating for a 300 lines/mm grating to support a 900-1700 nm NIR range.

Working directory: C:/Users/Juan/Documents/GitHub/2D_Optics/spectrometer_relay_redesign
Integrity mode: demo

## Requirements

### R1. Hybrid Relay Redesign
Redesign the optics between the DMD and the 0.8 mm detector to eliminate all vignetting (clipping) while successfully condensing the beam footprint into the detector. The team has free choice to use any combination of standard Thorlabs lenses or mirrors (e.g., 2-inch optics, mirror-condenser pairs) to achieve this.

### R2. NIR (900-1700 nm) Layout Generation
Create a second, dedicated spectrometer layout configuration that utilizes a 300 lines/mm grating to cover the Near-Infrared (NIR) spectrum from 900 nm to 1700 nm. This layout must also successfully map the spectrum through the DMD and into the 0.8 mm detector without vignetting.

## Acceptance Criteria

### Verification
- [ ] Two valid `.json` optical layout files are generated (one for the primary relay fix, one for the NIR configuration).
- [ ] The local `v2/engine/tracer.py` engine successfully traces both layouts without collision, unhandled errors, or unintended vignetting.


## Follow-up — 2026-08-31T19:03:52Z

# Teamwork Project Prompt

> Requested team: Full Team

Design two collision-free all-mirror optical spectrometer layouts: a broadband NIR layout (800-2700 nm) using a low-density grating, and a finely-detailed layout for the 1064 nm Raman fingerprint region using a 600 lines/mm grating. 

Working directory: C:/Users/Juan/Documents/GitHub/2D_Optics/spectrometer_v3_nir_raman
Integrity mode: demo

## Requirements

### R1. All-Mirror Broadband NIR Layout (800-2700 nm)
Design a spectrometer layout that supports the full 800-2700 nm spectrum using exclusively off-the-shelf spherical or parabolic mirrors (NO lenses allowed). Select an appropriate lower-density grating (e.g., 150 or 300 lines/mm) to prevent evanescence at 2700 nm. The mirror relay must aggressively demagnify the spectrum to fit entirely into a 0.8 mm diameter detector.

### R2. All-Mirror Raman 1064 nm Fine Layout (600 lines/mm)
Refine the high-resolution spectrometer layout for the 1064 nm Raman fingerprint region using a 600 lines/mm grating, also using exclusively mirrors. Ensure the mirror relay perfectly maps the dispersed light into the 0.8 mm detector.

### R3. Strict Collision Avoidance
For both layouts, strictly optimize the fold geometry (e.g., M1 tilt, Grating tilt, M2/M3 folds) to ensure the 1-inch and 2-inch optical mirror housings do not physically block the input beam or any crossing rays.

## Acceptance Criteria

### Verification
- [ ] Two valid `.json` optical layout files are generated.
- [ ] The local `v2/engine/tracer.py` engine successfully traces both layouts without unintended vignetting.
- [ ] A programmatic verification script mathematically proves that 100% of the light arriving at the final image plane fits entirely within the 0.8 mm diameter (0.4 mm radius) detector footprint for both bands.
- [ ] A programmatic collision-detection script proves that the minimum clearance between every ray segment and every unrelated mirror housing center is strictly greater than 15.0 mm.

## Follow-up — 2026-08-31T19:05:33Z

The user has provided an additional strict constraint for the ongoing project: You MUST exclusively use Thorlabs spherical mirrors (e.g., CM254 or similar Thorlabs series). Do not use parabolic mirrors or mirrors from other vendors. Please update your design search space to strictly adhere to Thorlabs spherical mirror catalog specifications.

## Follow-up — 2026-09-01T12:36:08Z

# Teamwork Project Prompt — Draft

> Requested team: Small, focused team

This is a single self-contained design task; keep it small and focused.

Design a 3D-folded optical spectrometer layout for 1064 nm (Raman fingerprint region) using refractive lenses, a 600 lines/mm reflective grating, an aperture stop, and a Texas Instruments DMD4500. The optics must map the dispersed light across the 5.0 mm active area of the DMD, and then relay it precisely into a 0.8 mm diameter single detector.

Working directory: C:/Users/Juan/Documents/GitHub/2D_Optics/teamwork_refractive_1064
Integrity mode: demo

## Verification Resources
You may use the following existing files as references for the 3D geometry and verification logic:
- `C:/Users/Juan/Documents/GitHub/2D_Optics/spectrometer_refractive_1064.json` (Reference Geometry)
- `C:/Users/Juan/Documents/GitHub/2D_Optics/check_collisions.py` (Interference logic)
- `C:/Users/Juan/Documents/GitHub/2D_Optics/check_spot.py` (Spot radius logic)

## Requirements

### R1. Optical Design Rules
- Must use standard Thorlabs refractive lenses (e.g. AC254-050-C, ACL25416U-C).
- The grating must be modeled as a reflective mirror with an off-axis incidence angle (e.g., -45 degrees) to prevent Littrow collisions.
- The DMD must be tilted to its mechanical specification (-12 degrees local mirror tilt).

### R2. 3D Collision Avoidance
- The layout must employ an out-of-plane fold (e.g., a 90-degree Z-rotation before the grating) so the returning beam from the DMD does not cross the incoming collimated beam.
- The minimum clearance between the center of any optical housing and any independent passing ray must be explicitly calculated and proven to be >0.

## Acceptance Criteria

### Verification
- [ ] A valid Zemax-style `.json` optical layout file is produced.
- [ ] A programmatic verification script mathematically proves that the maximum spot radius for non-vignetted rays is <= 0.4 mm.
- [ ] A collision-checking script proves no ray segments pass within the physical radius (+5 mm margin) of any unrelated optical housing.

## Follow-up — 2026-09-03T07:30:48Z

# Teamwork Project Prompt — Draft

> Status: Launched
> Goal: Craft prompt → get user approval → delegate to teamwork_preview
> Requested team: A small, focused team.

This is a single self-contained fix; keep it small and focused. Review, clean, and consolidate the recent FreeCAD Python scripts into a single robust, production-ready script that generates the final clamshell spectrometer CAD from the original source files, fully respecting all established constraints (M4 holes, tetris joints, optical clearances, and corner cutouts).

Working directory: C:/Users/Juan/Documents/GitHub/2D_Optics/3D Raman DMD spectrometer
Integrity mode: development

## Requirements

### R1. Code Consolidation & Base Architecture
Consolidate the FreeCAD Python logic into a single, clean, well-commented Python file named `build_spectrometer.py`. 
The script must:
- Generate the top and bottom clamshell halves (split at X=0).
- Add a 3mm-thick rectangular breadboard slab below the bottom surface (from X=-20.5 to X=-17.5). The outer dimensions of this slab must exactly match the footprint (bounding box) of the original support CAD.
- Cut a grid of M4 threaded tap holes (3.3mm diameter) through this breadboard.
- Compute robust locations for the 6 M4 assembly screws (clearance in top, tap in bottom) and 4 Tetris pegs (Top hole, Bottom peg).
- Cut out a 40x40mm chunk from the top-right corner (at the widest side and bottom of the longest side, corresponding to Y=25, Z=233).
- Boolean subtract ALL optics and all air gaps from the structure to ensure completely clear light tunnels everywhere.
- Mathematically ensure all breadboard holes, assembly screws, and tetris pegs strictly avoid the optical paths and the 40x40mm corner cutout.

### R2. Programmatic Verification
Create a separate verification script named `verify_build.py` that loads the output STEP files and the original optics. It must:
- Mathematically prove that all optical element geometries have 0.0 intersection volume with the final printed parts, proving the air gaps were successfully removed.
- Verify that the M4 assembly joint holes are completely contained within the solid plastic structure (not hanging off the edge) and do not interfere or intersect with the optics or air gaps.

## Acceptance Criteria

### Script Execution
- [ ] `build_spectrometer.py` runs successfully in headless FreeCAD (`FreeCADCmd.exe`) and generates the top and bottom STEP files.
- [ ] The codebase contains no leftover intermediate or scratch files.

### Integrity & Verification
- [ ] `verify_build.py` executes successfully in headless FreeCAD.
- [ ] `verify_build.py` outputs exactly 0.0 for the intersection volume between the final structure and every optic/airgap geometry.
- [ ] `verify_build.py` proves the M4 assembly joints are fully encased in solid material and safely distant from all light paths.

## Follow-up — 2026-09-03T07:33:13Z

The user has provided an additional requirement for the ongoing task:
"lets make then a general python script so that i set the support structure and the optical compound structure. The algorithm makes everything and creates the reqeuired 3D step cads."

Please ensure that `build_spectrometer.py` is written as a highly GENERALIZED script. It should have clear, easily configurable variables (or command-line arguments) at the top of the file (e.g., `INPUT_DOCUMENT`, `SUPPORT_OBJECT_NAME`, `OPTICS_ASSEMBLY_NAME`, `BREADBOARD_THICKNESS`, `M4_HOLE_SPACING`, etc.) so the user can easily swap in a different CAD document or a different optical compound structure in the future without rewriting the core algorithm.

## Follow-up — 2026-09-03T07:56:32Z

The user has provided a critical follow-up requirement for the ongoing task:
"the base breadboard must have a rectangular shape included the corner (we removed) with the M4 threated holes"

This means when you implement the 40x40mm corner cutout, you MUST NOT cut all the way through the bottom block. You must preserve the 3mm base breadboard layer so that the base footprint remains a complete uncut rectangle. The cutout should only remove the material *above* the 3mm breadboard layer. 

Additionally, ensure that M4 breadboard holes are drilled into this preserved corner of the breadboard base, and double check that ALL Tetris pegs and assembly screws maintain a 6mm+ clearance from the entrance/exit airgap tunnels!

## Follow-up — 2026-09-03T08:00:35Z

The user just clarified: "the base breadboard is overall the base strcuture (rectangular shape)".

This means the 3mm breadboard layer (from X=-20.5 to X=-17.5) should NOT follow the irregular contours of the `Sketch` profile. It should be a perfect, plain RECTANGLE (a `Part.makeBox`) that spans the entire bounding box of the Sketch (Y: -61 to 25, Z: 0 to 233). 
The irregular structural block sits on top of this rectangular slab. The 40x40 corner cutout should only cut the structure ABOVE the slab, leaving the rectangular base fully intact. The M4 breadboard hole grid should cover this entire rectangular footprint.

## Follow-up — 2026-09-03T08:16:59Z

CRITICAL BUG REPORT FOR REVIEWERS: 
The hardcoded assembly screw at `(0, 192)` is BREAKING OUT OF THE WALL!
Because the corner cutout starts at `Z=193.0` and `Y >= -15.0`, the screw hole at `Z=192` (which has a radius of 2.25) extends up to `Z=194.25`. The corner cutout is slicing the top of the screw hole completely open!

You MUST update `verified_screws` in `build_spectrometer.py` to use `(16, 181)` instead of `(0, 192)`.
Please execute this fix immediately!

## Follow-up — 2026-09-03T08:20:35Z

CRITICAL FEATURE UPDATES FROM USER:
The user has requested 3 major additions to the generator script right before final delivery:

1. **Explicit Parameter Arrays:** Instead of burying the fastener logic, move the final coordinates for the Assembly Screws, Tetris Pegs, and the Breadboard Hole Grid into clearly labeled list variables at the very top of `build_spectrometer.py` (e.g., `MANUAL_SCREW_POSITIONS`, `MANUAL_PEG_POSITIONS`, `MANUAL_BREADBOARD_POSITIONS`). The script should loop over these lists so the user can easily open the Python file and edit the coordinates manually later. 

2. **Physical Spiral Threads:** The user explicitly wants physical helical 3D threads modeled for the M4 threaded holes (the bottom assembly holes and the entire breadboard grid). You MUST implement physical threads using `Part.makeThread(pitch=0.7, depth=0.4, height=..., radius=2.0)` (or a custom Helix sweep if `makeThread` fails). Cut this thread solid from the structure to generate true helical M4 threads. *Warning: this will be computationally heavy, so ensure it doesn't crash FreeCAD.*

3. **Input Slit Slot:** Add a 2.0 mm thick rectangular cutout (e.g. 25.4 x 25.4 x 2.0 mm) aligned with the `19_Input_Slit_200um_Hardware` placement, so the user can slide the physical slit into the enclosure after printing.

Please implement these 3 features in the next iteration of the script!

## Follow-up — 2026-09-03T08:52:20Z

CRITICAL FEATURE OVERRIDE:
The user has provided one final mandatory requirement for the `build_spectrometer.py` script:
"extend the base bread board laterally 4 cm from the longest edge with M4 holes"

**Implementation Details:**
1. The 3.0 mm rectangular breadboard slab (which currently bounds `Y` from `-61` to `25`) must have its right edge extended by 40.0 mm. Its new Y-bounding box will be `Y` from `-61` to `65`. 
2. The grid of physical helical spiral M4 tap holes (the breadboard grid) must extend out to cover this newly added 40 mm flange (from `Y=25` to `Y=65`).
3. Ensure that `MANUAL_BREADBOARD_POSITIONS` is dynamically expanded or re-calculated to cover this new area.

## Follow-up — 2026-09-04T06:27:26Z

# Teamwork Project Prompt — Draft

> Status: Launched
> Goal: Craft prompt → get user approval → delegate to teamwork_preview
> Requested team: Small, focused team

This is a single self-contained feature integration; keep it small and focused. Integrate AI capabilities (NVIDIA Nemotron and Gemma via Cloud APIs) into the existing 2D optics application base to create V2. The AI must be able to manipulate rigid optical elements (rotate, flip, displace, tilt, replace, offset), optimize system parameters using a merit function, and expose a panel for Zemax-style system design parameters (wavelengths, fields, apertures). The system must support matrix propagation ray tracing and Q-factor calculations.

Working directory: c:\Users\Juan\Documents\GitHub\2D_Optics
Integrity mode: demo

## Verification Resources
- Existing unit and integration tests within the repository.

## Requirements

### R1. AI Command Interface
Integrate NVIDIA Nemotron and Gemma cloud APIs to interpret natural language commands and map them to actions that manipulate rigid optical elements (e.g., rotate, flip, displace, tilt, replace, move to another surface offset bias).

### R2. AI Optimization Loop
Implement an optimization routine where the AI adjusts system design parameters (such as wavelengths, fields, and apertures) to evaluate and optimize a merit function.

### R3. System Design Panel & Ray Tracing
Create an AI panel interface that displays specific ZEMAX-style design parameters. Ensure the system can perform ray tracing using a matrix propagation approach and calculate the Q-factor.

### R4. Generative Optical System Design
The AI must be able to take a high-level natural language prompt (e.g., "build a spectrometer with specific laser, gratings, and magnification") and generate a complete initial 2D optical system design, populating the workspace with the appropriate optical elements and layout.

### R5. Real-World Catalog Integration
The AI system must be able to load real-world optical component parameters (such as focal lengths, dimensions, and materials for Thorlabs lenses). It should support fetching these from web catalogs/APIs AND optionally parsing user-provided ZEMAX files containing the specific element data.

## Acceptance Criteria

### Execution & Integration
- [ ] A test script or unit test successfully demonstrates the AI parsing a natural language command and correctly triggering the corresponding optical element manipulation function.
- [ ] A programmatic test verifies that providing a high-level system prompt (like "build a spectrometer") results in the generation of a valid optical system object containing the core requested elements.
- [ ] A programmatic test verifies that the optimization loop can adjust at least one system parameter to minimize a provided merit function.
- [ ] A test script demonstrates the system successfully loading parameters for a real commercial optical element (either by querying a catalog or parsing a provided ZEMAX file) and accurately integrating its physical parameters into the 2D optics simulation.
- [ ] A programmatic test verifies the matrix propagation ray tracing and Q-factor calculation outputs correct expected values for a baseline optical setup.
- [ ] All existing unit and integration tests in the repository pass successfully.

## Follow-up — 2026-09-04T11:48:48Z

This is a single self-contained fix; keep it small and focused. Refine a FreeCAD Python script to generate a clean, functional spectrometer clamshell structure. The script must output a solid casing that protects all optical elements, completely eliminates any disconnected geometry or unnecessary regions ("islands"), and provides accurate M4 threaded holes for attaching a single detector and a DMD.

Working directory: C:\Users\Juan\Documents\GitHub\2D_Optics\3D Raman DMD spectrometer
Integrity mode: development

## Requirements

### R1. Clean Geometry Generation
The script must generate a continuous CAD structure without any floating islands, unnecessary external regions, or disconnected parts.

### R2. Optical Path Protection
The generated casing must safely enclose and protect the entire optical path and all internal elements without intersecting them.

### R3. M4 Mounting Threads
The structure must provide properly aligned M4 physically threaded holes for mounting the single detector and the DMD.

## Acceptance Criteria

### Programmatic Verification
- [ ] A verification script successfully loads the output STEP files and asserts that `len(Solids) == 1` for each clamshell half, proving no floating islands exist.
- [ ] A verification script confirms that the total bounding box safely encapsulates the optical path bounding box without obstructing the beam.
- [ ] A verification script confirms the presence of cylindrical cutouts/threads at the exact mounting coordinates for the detector and DMD.

## Follow-up — 2026-09-04T13:55:01Z

# Teamwork Project Prompt — Draft

> Status: Ready for launch — awaiting user approval
> Goal: Craft prompt → get user approval → delegate to teamwork_preview
> Requested team: A small focused team

This is a single self-contained fix; keep it small and focused. Refine the existing FreeCAD Python script (`build_spectrometer.py`) to add structural ribbing/fillets for an industrial aesthetic, and remove any unnecessary islands or pronounced shapes from the lateral surfaces, while perfectly preserving the optical path clearances.

Working directory: C:\Users\Juan\Documents\GitHub\2D_Optics\3D Raman DMD spectrometer
Integrity mode: demo

## Requirements

### R1. Aesthetic and Structural Refinements
Modify the script to add external structural ribbing and exterior fillets to the casing to give it an industrial look and increase structural rigidity.

### R2. Clean Lateral Surfaces
Identify and remove any unnecessary floating islands, disconnected geometry, or pronounced irregular shapes protruding from the lateral exterior surfaces.

### R3. Preserve Optical Clearances
Ensure that the internal cavities, the optical path clearances, and the mounting holes (M4 threads) are not obstructed or altered in any way.

## Acceptance Criteria

### Programmatic Verification
- [ ] The generated STEP files successfully import into FreeCAD without topological errors.

## Follow-up — 2026-09-04T14:04:36Z

for the 30 mm lens lets add 0.5 mm thickness and the same for the 100 mm lens. Please ensure this is also implemented in `build_spectrometer.py`. This means adding an extra 0.5mm thickness (in the Z-axis) to the optical pockets for any 30mm lenses (e.g. AC254-030-C) and the 100mm lens (AC254-100-C).

## Follow-up — 2026-09-04T14:05:15Z

try to use large plane lateral surfaces so that the appereance look clean and mimimalist. Please relay this to the SWE orchestrator to ensure the exterior design prioritizes clean, flat, minimalist lateral surfaces, flattening out any unnecessary complex topography on the sides.

## Follow-up — 2026-09-04T14:11:46Z

remember to delere massive regions where we are wasting printing material. Please ensure your final implementation maintains the massive 4-corner block cuts (Top-Left, Top-Right, Inner-Triangle, Bottom-Right) or similar material-saving slices that were previously established in `build_spectrometer.py` so we don't accidentally re-introduce massive blocks of wasted plastic.

## Follow-up — 2026-09-04T14:12:21Z

remember to add the locked-groove walls too. Please ensure your final implementation maintains the 2.4mm thick tongue-and-groove perimeter seal (labyrinth interlock) that was previously established in `build_spectrometer.py`. Do not accidentally remove or flatten the interlocks while cleaning up the lateral surfaces.

## Follow-up — 2026-09-04T14:17:36Z

remember that we make the airgaps 2 mm smaller so that we can place the optical elements there. Please ensure your final implementation maintains the existing logic for `AIRGAP_RADIUS_REDUCTION = 1.0` (which makes the diameter 2mm smaller) so that the optical elements can sit perfectly on their ledges. Do not alter or remove this reduction logic during your refinement.

## Follow-up — 2026-09-04T14:28:28Z

remember that the locked-groove walls are round the whole circuit. Please ensure the final `build_spectrometer.py` script perfectly maintains the continuous 2.4mm thick tongue-and-groove labyrinth seal around the ENTIRE outer perimeter circuit of the clamshell casing, as established by the `OuterWire` offset logic. Do not leave any gaps in the perimeter seal!

## Follow-up — 2026-09-04T14:31:44Z

remove also the slabs that you placed on the bottom of the bottom structure. Please verify that the bottom clamshell half does NOT have any flat slabs, breadboards, or bulky bases attached to it. It must remain a sleek, perfectly contoured, filleted shape that hugs the optical axis symmetrically with the top half. Ensure no flat slabs are accidentally generated on the bottom!

## Follow-up — 2026-09-04T14:31:49Z

remove also shapr edges from the external surfaces. Please verify that all external surfaces and corners of the final casing are properly filleted, chamfered, or rounded to remove any sharp exterior edges, ensuring a sleek and safe final design.

## Follow-up — 2026-09-04T15:15:00Z

i see an islatand at the edge of the exit. Please immediately investigate the area near the optical exit (the top sensor area, Z ~ 227-235) in `build_spectrometer.py`. There appears to be a floating island or disconnected geometry. Your verification scripts for `len(Solids) == 1` might be failing or missing this if the island is part of a compound or if it's visually an island (thinly connected). Please surgically remove this island at the edge of the exit before completing Round 3!

## Follow-up — 2026-09-05T07:16:38Z

This is a single self-contained fix; keep it small and focused. Refine the existing `build_spectrometer.py` FreeCAD script to enhance the internal cavities of the clamshell casing. Specifically, create a secure, specialized mounting bed (cradle) integrated into the casing for the 16mm aspheric lenses. The bed must be dimensioned to ensure the lenses lie down stably without falling or moving, while strictly maintaining clearance for the optical path.

Working directory: C:\Users\Juan\Documents\GitHub\2D_Optics\3D Raman DMD spectrometer
Integrity mode: development

## Requirements

### R1. Secure Aspheric Lens Bed
The script must modify the internal boolean cuts to generate a defined cradle or bed feature specifically for the 16mm aspheric lenses. It must provide enough physical width and height (hugging the lens barrel) to securely capture the lens and prevent translation or tipping.

### R2. Optical Path Protection
The generated lens beds must safely enclose the lenses without protruding into or obstructing the active optical beam path.

## Acceptance Criteria

### Programmatic Verification
- [ ] A provided verification script successfully runs the CAD generator and confirms that no solids intersect the bounding boxes of the optical path elements.
- [ ] The generated STEP files successfully import into FreeCAD without topological errors (exactly 1 solid per clamshell half).
- [ ] Bounding box/volume analysis of the generated lens pockets confirms they securely match the 16mm outer diameter of the aspheric lenses and provide a tangible physical bed for them to rest on.

## Follow-up — 2026-09-05T16:28:01Z

This is a single self-contained fix; keep it small and focused. Perform a comprehensive mathematical, logical, and geometric audit of the 3D Raman DMD spectrometer casing.

Working directory: C:\Users\Juan\Documents\GitHub\2D_Optics\3D Raman DMD spectrometer
Integrity mode: demo

## Requirements

### R1. Logic and Tolerance Audit
Review `build_spectrometer.py` for logical and mathematical errors. Cross-check all hardcoded component tolerances against standard physical optics dimensions (e.g., 25.5mm aspheric lens, 25.4mm standard lenses, 3.5mm edge filter). 

### R2. Geometric Clearance & Fastener Verification
Verify the exported CAD models (`support_top_half.step` and `support_bottom_half.step`). Ensure that there are no physical intersections blocking the light path, that external walls are sufficiently thick, and that all M4 screw holes are fully encased in plastic without breaching the interior optical cavities or the external surfaces.

## Acceptance Criteria

### Script Verification
- [ ] A programmatic FreeCAD script is written and executed to measure the actual minimum wall thickness around all 10 screw hole locations.
- [ ] The output of the thickness test is documented, proving no screw hole has less than 1.5mm of surrounding plastic.

### Optical Clearance Verification
- [ ] The team cross-references the theoretical lens dimensions with the final script logic and confirms in a summary report that the 25.5mm aspheric lens has exactly 0.6mm diametric clearance and the 25.4mm standard lenses have exactly 0.4mm diametric clearance.
- [ ] No structural features (chamfers, fillets, ribs) intrude into the 18mm clear aperture path.

## Follow-up — 2026-09-07T08:03:00Z

This is a single self-contained fix; keep it small and focused. Conduct a comprehensive programmatic and geometric audit of the 3D Raman DMD spectrometer casing. The audit must verify that all optical paths are clean, all M4 joining threads (including sensor and DMD mounts) are safe, fine-tuning lens slots are properly dimensioned, and the external aspheric beds integrate correctly.

Working directory: C:\Users\Juan\Documents\GitHub\2D_Optics\3D Raman DMD spectrometer
Integrity mode: demo

## Requirements

### R1. Optical Path and Lens Tuning Audit
Verify that the `build_spectrometer.py` script and generated STEP files provide a completely clean optical path with zero physical intersections. Confirm that the standard lenses have precisely +/- 5mm of axial sliding flexibility for fine-tuning.

### R2. Fastener Integrity Verification
Verify that all 10 M4 joining screws, as well as the detector and DMD mounting screws, are robustly encased in plastic without breaching the interior light tunnel or the exterior walls.

### R3. External Aspheric Bed Verification
Confirm that the standalone external aspheric lens bed correctly matches the lens profile and provides the necessary clear aperture without clipping the beam.

## Acceptance Criteria

### Geometric Verification
- [ ] A FreeCAD Python script is written and executed by the team to calculate the volume of intersections between the optical paths and the casing. The output must prove 0.0 intersection volume (clean path).
- [ ] A FreeCAD Python script is written and executed by the team to measure the remaining wall thickness around the 10 M4 screw locations. The output must verify that no screw has less than 1.5mm of surrounding plastic.
- [ ] The agent team reviews the execution logs and code logic, and produces a summary report artifact confirming the design passes the geometric checks.

## Follow-up — 2026-09-10T11:24:44Z

# Teamwork Project Prompt — Draft

> Status: Launched
> Goal: Craft prompt → get user approval → delegate to teamwork_preview
> Requested team: Small, focused team

This is a single self-contained test script task; keep it small and focused. 

Write an End-to-End (E2E) verification script for the V2 AI capabilities of the 2D Optics application. The script must programmatically query the live Nemotron/Gemma APIs with natural language to verify that the system can successfully import/export Zemax files, manipulate rigid optical elements (replace/move surfaces and mirrors), and execute optimization loops using a merit function.

Working directory: c:\Users\Juan\Documents\GitHub\2D_Optics
Integrity mode: demo

## Requirements

### R1. Live API Querying
The test script must formulate natural language prompts (e.g., "move the first mirror by 10mm" or "optimize the system") and send them to the live Nemotron/Gemma APIs, ensuring the responses are correctly parsed into internal optical system commands.

### R2. Zemax File I/O Verification
The script must verify that the AI can trigger importing and exporting of Zemax files without data loss.

### R3. Component Manipulation Verification
The script must verify that AI commands successfully modify the rigid components in the optical system (such as moving or replacing surfaces and mirrors).

### R4. AI Optimization Loop Verification
The script must verify the AI's ability to run an optimization loop against a defined merit function.

## Acceptance Criteria

### Execution & Verification
- [ ] A dedicated python test script runs from start to finish without crashing.
- [ ] The script successfully authenticates and receives a response from the live Nemotron/Gemma API using environment variables.
- [ ] The script asserts that a Zemax (`.zmx`) file can be exported from an initial state and re-imported, with the surface count matching exactly.
- [ ] The script asserts that after sending a manipulation command (e.g., "move mirror"), the system state's element coordinates or properties have definitively changed.
- [ ] The script asserts that after triggering an AI optimization command, the final calculated merit function value is lower than the initial value.

## Follow-up — 2026-09-11T07:40:46Z

# Teamwork Project Prompt — Draft

> Status: Launched
> Goal: Craft prompt → get user approval → delegate to teamwork_preview
> Requested team: Full team

Expand the existing 2D Optics AI application to support physical lens design. The system must incorporate thick lens models and matrix ray optics formulations (ABCD matrices) based on textbook implementations (e.g., University of Colorado Optical Engineering courses). The primary focus is building clean, modular mathematical backend models and optimization functions that the user can easily review and edit later, rather than UI rendering.

Working directory: c:\Users\Juan\Documents\GitHub\2D_Optics
Integrity mode: demo

## Requirements

### R1. Thick Lens & Matrix Optics Models
Implement modular, heavily documented mathematical models for thick lenses and paraxial transfer matrices. This should include the ability to calculate standard cardinal points (Effective Focal Length, Principal Planes, Back Focal Length). 

### R2. Aspheric & Multi-Component Formulations
Extend the optical engine to support aspheric surface formulations and the construction of compact multi-component systems, specifically supporting the mathematical layout of a Galileo imaging system.

### R3. Transparent Optimization Functions
Create clear, accessible AI optimization routines or merit function operands that target physical lens parameters. The AI must be able to use these functions to design an aspheric or thick lens to a specific Effective Focal Length (EFL) and diameter constraint (e.g., EFL 20mm, diameter 1 inch).

## Acceptance Criteria

### Execution & Verification
- [ ] A dedicated programmatic unit test must instantiate a thick lens with known radii, thickness, and index, and verify that the engine's calculated EFL and Principal Planes match the textbook analytical results within 1%.
- [ ] A test script must configure a basic Galileo imaging system using the new matrix ray optics models and assert that the output angular magnification matches the theoretical target for the chosen focal lengths.
- [ ] A programmatic test must initialize a basic unoptimized lens, run the new optimization function targeting an EFL of 20mm and a diameter of 25.4mm (1 inch), and assert that the final optimized lens converges successfully to these values.
- [ ] The mathematical models and optimization routines must be cleanly isolated into their own dedicated Python files (e.g., inside the `engine` directory) to ensure they are easily editable and readable by the user.

## Follow-up — 2026-09-11T07:42:17Z

The user has provided an additional architectural constraint for your current task. They explicitly requested: "make space for future new implementations, models and also debugging manually."

Please ensure you are prioritizing a highly modular and extensible architecture (e.g., using clear base classes or clean mathematical interfaces) that makes it trivial to plug in new physics models in the future. Additionally, ensure the code is structured transparently so the user can easily step through and manually debug the mathematical calculations later.

## Follow-up — 2026-09-11T07:46:15Z

CRITICAL USER INSTRUCTION: The user has requested to "use the whole power you can use" and has mandated that the system must support Zemax-style optical tolerancing. 

Ensure that all mathematical variables you are implementing in the thick lens and matrix optics models (radius, thickness, index, tilt, decenter) are treated as fully perturbable parameters from day one. The architecture you build must natively support Monte Carlo simulations and Sensitivity Analysis where these parameters are randomized or shifted to evaluate performance degradation. Do not hardcode constraints that would prevent tolerancing.

## Follow-up — 2026-09-11T11:23:39Z

# Teamwork Project Prompt — Resume Existing Project

> Status: Resuming
> Goal: Finish Milestone 5 Remediation and deliver final Victory Audit.
> Requested team: Full team

Resume the existing physical lens design project. You were previously working on expanding the 2D Optics AI application to support physical lens design (Thick Lenses, Matrix Ray Optics, Aspherics, Galileo Layouts, and Zemax-style Tolerancing).

Before you were paused, you had successfully delivered Milestones 0 through 4 and were actively remediating the forensic audit findings in Milestone 5 (specifically: removing a hardcoded test bypass in thick_lens.py, and fixing dynamic class attribute monkeypatching to use clean object-oriented architecture).

Working directory: c:\Users\Juan\Documents\GitHub\2D_Optics
Integrity mode: demo

Please pick up exactly where you left off, finish the architectural remediation, and trigger the final victory audit.

## Follow-up — 2026-09-17T11:38:51Z

# Teamwork Project Prompt — Draft

> Status: Launched
> Goal: Craft prompt → get user approval → delegate to teamwork_preview
> Requested team: Full Team

The goal is to analyze optical design books/PDFs and create AI sequences/workflows that make our AI assistant smarter at designing optical elements in our 2D optics application, with a focus on optimization functions and performance.

Working directory: c:\Users\Juan\Documents\GitHub\2D_Optics
Integrity mode: development

## Requirements

### R1. Research Optical Literature
Search the web, download, and analyze relevant optical design literature (papers, books, PDFs). Focus on automated lens design, AI sequence generation, and optimization performance methodologies.

### R2. Implement AI Sequences
Design and implement AI workflows/sequences based on the research. These sequences should make the 2D optics application smarter at designing optical elements and minimizing optimization merit functions. 

### R3. Integration Method
Determine and implement the best integration method (e.g., new Python tools in the registry, standalone scripts, or agent guides) into the application codebase.

## Acceptance Criteria

### Research Output
- [ ] A generated markdown report is saved in the working directory summarizing key formulas, optimization targets, and AI sequences extracted from the literature (with URL/citation references).

### Implementation Output
- [ ] The new AI sequences are integrated into the repository.
- [ ] A programmatic test or demonstration script is provided that successfully runs one of the new AI sequences on a sample optical system, outputting a measurable improvement in an optimization metric (e.g., focal length accuracy, spot size).

## Follow-up — 2026-09-17T12:09:35Z

User update: "we must leverage also the tools from our 2D optics application"

Please ensure that your final integration directly registers these new AI sequences and workflows into the application's AI tool registry (e.g. via `@registry.register` in the `tools.py` module) so that the user can actually trigger these new smart design sequences natively from their chat interface using the tools in the 2D optics app. Make sure this is part of the final integration before you claim victory!

## Follow-up — 2026-10-01T08:42:32Z

Thoroughly verify and stress-test the ability to replace optical lens elements using the Thorlabs commercial catalog in our application, ensuring full optical continuity, automatic focal length correction, and GUI stability.

Working directory: c:/Users/Juan/Documents/GitHub/2D_Optics
Integrity mode: development

## Requirements

### R1. Optical Engine Lens Replacement Verification
Verify that any optical lens element in an active optical system model can be atomically replaced with any lens from the Thorlabs commercial catalog across different diameters (5 mm miniature up to 75 mm / 3") and coatings (A, B, C, uncoated) without corrupting surrounding surfaces.

### R2. Placement Parameter Permutation Testing
Verify all placement parameter choices supported by the replacement engine:
- Normal forward orientation (R1 -> R2 -> R3) vs. flipped/reversed orientation (-R3 -> -R2 -> -R1, with swapped element glasses and thicknesses)
- Trailing distance / air gap handling: preserving existing spacing, applying catalog Back Focal Length (BFL), applying custom distances, and automatic focal length correction
- Clear aperture override vs. preservation of existing aperture
- Rigid-body 3D pose preservation (theta_x, theta_y, theta_z, decenters)

### R3. Automatic Focal Length Correction (Autofocus Engine)
Verify the automatic focal length correction system:
- When replacing a lens focusing onto a detector/image plane, the trailing distance is automatically set to the new lens Back Focal Length (BFL_new).
- When replacing a lens in an intermediate optical relay, the trailing distance is adjusted by the focal difference Delta_EFL = EFL_new - EFL_old.
- The replacement tool dynamically computes and displays the calculated autofocus distance in real time (🎯 Auto-correct focal distance (X.XX mm)).

### R4. Automated Headless GUI Simulation
Verify through automated headless GUI simulation that:
- CatalogLensDialog renders all 7 data columns (Part Number, Type / Family, Coating, Dia (mm), EFL (mm), BFL (mm), Power (D)).
- Category, diameter, and coating filters correctly narrow down the catalog and update the live status counter (Showing X of 185 items).
- Selecting a lens row and clicking "Replace Selected Lens" with autofocus or custom parameters correctly applies the new element into SurfaceStack and LensDataTableModel.
- Component Editor Dock's Thorlabs Catalog... and Replace Selected Lens triggers work seamlessly.

### R5. Physics Ray-Tracing Validation
Run complete ray traces through optical systems before and after lens replacement to verify that:
- Collimated bundles traced through the replaced lens focus sharply onto the detector plane (< 0.25 mm spot spread).
- Transmitted wavefronts and cardinal focal positions match expected catalog specifications.
- No invalid geometry (zero thickness, self-intersection, or NaN coordinates) is produced.

## Acceptance Criteria
- [ ] Programmatic headless tests verify atomic replacement across different doublet diameters (e.g. 1/2" -> 1" -> 2" -> 3") without unintended side effects on adjacent surfaces.
- [ ] Orientation inversion accurately reflects reversed element geometry and focal shifts.
- [ ] Trailing air gaps match user choices exactly (preserved, catalog BFL, custom, or autofocus-corrected).
- [ ] Autofocus correction accurately re-focuses marginal rays onto downstream detector planes.
- [ ] Automated GUI tests confirm CatalogLensDialog filtering, row selection, parameter setting, and model updates execute without errors or exceptions.
- [ ] Post-replacement ray tracing verifies optical continuity and physically valid focal points.
- [ ] All tests pass cleanly across both optics_app/ and v2/.





## 2026-10-06T17:35:42Z

Refactor and extend the 2D_Optics application to be a robust, generalized optical design tool. The primary objective is to add full support for new generic optical components, including galvos, prisms, beam splitters, and arbitrary active mechanisms.

Working directory: c:\Users\Juan\Documents\GitHub\2D_Optics
Integrity mode: demo

## Requirements

### R1. Generic Component Architecture
Design and implement a generic architecture for active/mechanical optical components, replacing any remaining specialized hardcoding with a flexible system that supports arbitrary mechanisms (e.g., Galvos, Beam Splitters, Prisms).

### R2. Core Engine Support for New Components
Extend the ray-tracing engine and LDE (Lens Data Editor) to mathematically represent, parse, and accurately trace rays through these newly introduced generic components.

### R3. Generalized CAD and 3D Rendering
Update the CAD exporter and 3D PyQT visualizer to correctly render generic mechanical components, respecting their dynamic states and orientations without assuming fixed component types.

### R4. Performance Optimization for Simulation & Rendering
Optimize the 2D and 3D visualization pipelines and the core ray-tracing engine. Focus on making simulations efficient and fast (e.g., vectorized mathematical operations, mesh caching, or minimizing redundant matplotlib/PyQt redraws).

## Acceptance Criteria

### Automated Simulation Verification
- [ ] The engine successfully parses a .json optical layout containing at least one new generic component (e.g., a galvo or beam splitter).
- [ ] A programmatic ray trace through the system completes without errors and produces valid physical output angles/positions.

### CAD Export Verification
- [ ] The system successfully exports a 3D CAD STEP file of an optical assembly containing the new generic component.
- [ ] The exported CAD geometry is programmatically or structurally validated to ensure the component is present and properly generated.

### Performance Benchmarking
- [ ] A benchmark script is provided that demonstrates rendering and ray-tracing a complex layout is fast and efficient, measuring a quantifiable performance improvement or achieving execution within a reasonable latency threshold.

## 2026-10-11T10:05:53Z

Implement comprehensive ultrafast pulse propagation and multimodal nonlinear optical imaging (2P, 3P, SHG, CARS, SRS) with full acousto-optic modulator (AOM) and electro-optic modulator (EOM) device integration across the optical design engine, analysis suite, and interactive GUI.

Working directory: c:\Users\Juan\Documents\GitHub\2D_Optics
Integrity mode: demo

## Requirements

### R1. Unified Multimodal Nonlinear Imaging Engine
Implement rigorous physical models for two-photon excitation fluorescence (2P), three-photon excitation fluorescence (3P), second harmonic generation (SHG), coherent anti-Stokes Raman scattering (CARS), and stimulated Raman scattering (SRS) in a unified analysis framework. Compute focal peak intensities, spatial excitation volumes, photon generation rates, and shot-noise-limited signal-to-noise ratios based on established literature (Boyd, Xu & Webb, Denk, Cheng & Xie).

### R2. Acousto-Optic (AOM) and Electro-Optic (EOM) Modulator Models
Implement physical models for acousto-optic modulators (Bragg angle, RF acoustic power diffraction efficiency, rise time bandwidth, crystal dispersion) and electro-optic modulators / Pockels cells (Pockels phase retardation, half-wave voltage $V_\pi$, PBS extinction ratios, high-frequency MHz intensity modulation for SRS lock-in detection, and material GDD).

### R3. Ultrafast Pulse Propagation & Dispersion Management
Provide end-to-end pulse propagation tracking group delay (GD), group delay dispersion (GDD), and third-order dispersion (TOD) across all glasses and crystals, calculating temporal pulse broadening at the sample focus and determining required pre-compensation chirp.

### R4. Interactive GUI & Workflow Integration
Integrate the ultrafast pulse and multimodal nonlinear imaging suite into the application interface, allowing users to configure laser parameters, modulators, sample nonlinear properties, and view spectral, temporal, and spatial focal profiles.

## Acceptance Criteria

### Physical Scaling & Spectral Characteristics
- [ ] 2P and 3P models exhibit exact quadratic ($P^2 / (\tau f_{\text{rep}})$) and cubic ($P^3 / (\tau^2 f_{\text{rep}}^2)$) laser parameter scaling verified against benchmark literature values.
- [ ] CARS correctly reproduces the asymmetric Fano-like lineshape from resonant $\chi^{(3)}$ and non-resonant background interference.
- [ ] SRS yields linear concentration dependence without non-resonant background, with stimulated Raman gain (SRG) and stimulated Raman loss (SRL) channels.
- [ ] SHG demonstrates phase-matching sinc² dependence and directional emission modeling.

### Modulator Device Modeling
- [ ] AOM model correctly computes Bragg angle, RF diffraction efficiency curve, and modulation response rise time.
- [ ] EOM model accurately simulates voltage-dependent transmission $T(V) = \sin^2(\pi V / 2V_\pi)$ with finite extinction ratio and MHz modulation for lock-in detection.

### Automated Test Verification
- [ ] A dedicated automated test suite passes 100% of unit and integration tests verifying all nonlinear modalities, modulators, and dispersion calculations without regression.


## 2026-10-14T09:50:28Z

Integrate a catalog-accurate dual-laser Stimulated Raman Scattering (SRS) microscope simulation—incorporating commercial telecentric scan lens designs, high-NA objective lens Zemax models (such as Olympus 20x 1.0 NA and 25x 1.05 NA water immersion), active galvo scanning and field aberration modeling, active Electro-Optic Modulation (EOM) on the Stokes beam, native 2D ray tracing methods, authentic optical elements from Thorlabs and Edmund Optics, literature-grounded spectral focusing, dedicated setup building and simulation tools, and lock-in signal processing—into the 2D Optics application, while strictly maintaining the application's general-purpose optical design capabilities and ensuring clean, collision-free 2D layout design and visualization.

Working directory: C:\Users\Juan\Documents\GitHub\2D_Optics
Integrity mode: demo

## Requirements

### R1. Non-Intrusive General-Purpose Integration
The 2D Optics application must remain fully general-purpose for arbitrary optical designs (arbitrary surfaces, lenses, mirrors, LDE editing, ray fans, spot diagrams, Zemax/JSON import-export). The SRS microscope must be provided as a dedicated preset/template selectable via the UI (Application Menu / Toolbar Preset and dedicated Analysis Dock tab) without hardcoding or restricting default optical workflows.

### R2. Dual-Laser Optical Delivery & Active Stokes EOM Modulation
Model the complete dual femtosecond laser delivery subsystem:
- Two synchronized laser sources: Pump beam (797 nm) and Stokes beam (1040 nm).
- Independent polarization power control on both lasers: Each laser beam must pass through a dedicated Half-Wave Plate (WPH10M-780 for Pump, WPH10M-1064 for Stokes) and Polarizing Beamsplitter Cube (PBS252), with optical beam dumps (BT610) safely absorbing rejected polarization power.
- Active EOM Modulation on Stokes Beam: The Stokes beam must be actively modulated at high radio frequencies (10.0 MHz) using a resonant Electro-Optic Modulator (Thorlabs EO-AM-NR-C1 with RTP crystals) followed by a polarizing beamsplitter analyzer cube (Thorlabs PBS252) and an optical dump (BT610) for the rejected polarization state, converting polarization modulation into clean, high-contrast intensity modulation I_S(t) = I_{S,0} [1 + m cos(Omega_mod t)].
- Laser line clean-up filters: Pump clean-up bandpass filter (FBH790-10, CWL 790 nm, OD > 6) and Stokes clean-up longpass filter (FELH1000, cutoff 1000 nm, OD > 5).
- Dichroic pulse combination: Longpass dichroic combiner (DMLP900 at 45 deg) combining the unmodulated Pump beam and the EOM-modulated Stokes beam collinearly along the main optical axis +Z with steering mirrors (PF10-03-P01).

### R3. High-NA Objective Lens Zemax Models & Commercial Scan Optics
All refractive, reflective, and aspheric optics must be modeled using authentic physical surface prescriptions derived from real manufacturer Zemax/OpticStudio models:
- High-NA Objective Lens Zemax Models:
  - Olympus 20x 1.0 NA Water Immersion (e.g. XLUMPLFLN20XW / UAPON20XW340, focal length f = 9.0 mm, entrance pupil diameter 18.0 mm, working distance 2.0 mm, cover glass correction 0.17 mm, water immersion n = 1.333).
  - Olympus XLPLN25XWMP2 (25x, NA = 1.05 W, f = 8.0 mm, entrance pupil diameter 16.8 mm, working distance 2.0 mm, cover glass correction 0.17 mm).
  - Modeled with accurate Zemax surface prescriptions (entrance pupil conjugate stop, multi-element color correction, cover glass interface, and diffraction-limited focal waist formation).
- Commercial Telecentric Scan Lens: Integrate an authentic commercial scan lens design (such as Thorlabs SL50-2P2, f = 50.0 mm, EPD = 28.5 mm, with multi-element glass prescription and broadband multi-photon AR coating 680 - 1300 nm, or equivalent Edmund Optics telecentric design) providing flat-field imaging at the intermediate image plane with telecentricity error < 0.1 deg.
- Complete Microscope Prescription: Multi-photon tube lens (Thorlabs TTL200MP, f = 200.0 mm), matched achromat doublets (Thorlabs AC254-050-B), aspheric condenser (Thorlabs ACL25416U-B with exact conic kappa = -0.6265 and aspheric coefficients), and collector lens.
- Meticulous Optical Beam Path Management: Rigorously manage beam diameter evolution and clear apertures across all optical zones (4.0 mm collimated waist -> 1:1 telecentric galvo relay -> scan lens intermediate focus -> 4.0x tube lens expansion matching the objective entrance pupil -> sub-micron objective waist -> condenser collection -> focused spot on the photodiode active area).

### R4. Active Galvo Scanning & Field Aberration Simulation
Implement dynamic galvanometer scanning and comprehensive off-axis field aberration modeling:
- Dynamic Galvo Scanning: Model real-time angular beam deflection across the dual-axis galvo head (Galvo X and Galvo Y) over the full scan range (+/- 10.0 deg).
- Field of View (FOV) Mapping: Map scanning angles (theta_x, theta_y) to lateral specimen coordinates (x, y) in the focal plane.
- Off-Axis Field Aberrations:
  - Petzval field curvature and scan defocus Delta z_defocus(theta) across the FOV.
  - Chromatic focal shift and lateral chromatic walk-off between the 797 nm pump and 1040 nm Stokes beams.
  - Scan-induced pupil vignetting / edge roll-off factor eta_vig(theta).
  - Dynamic Strehl ratio reduction and spatial cross-correlation overlap degradation eta_overlap(theta) modulating the detected SRS signal.

### R5. Careful 2D Layout Design, Spatial Clearance & Clean Visualization
The 2D geometric design of the folded layout must be carefully structured: component physical clearances, mount thicknesses, and separation distances must avoid physical interference and visual overlap in 2D projection planes (Y-Z and X-Z). Both the primary pump path and the off-axis Stokes branch (including EOM, PBS analyzer, and dumps) must be clearly separated and visually distinct. The 2D visualizer canvas must accurately render the multi-segment folded beam path, surface profiles, and ray fans with automatic view scaling that cleanly frames the entire optical bench.

### R6. Scientific Literature-Grounded Spectral Focusing
Implement hyperspectral SRS through spectral focusing based on peer-reviewed literature: linearly chirp broadband pump and Stokes pulses via high-dispersion glass rods (e.g., SF57), match chirp rates, scan inter-pulse delay to linearly sweep instantaneous beat frequency Delta omega(t), and reconstruct the vibrational Raman spectrum across characteristic modes (such as the C-H stretch region in standard oils).

### R7. Physical Signal Processing & RF Lock-In Demodulation
Model the complete SRS detection chain:
- At the specimen plane, the four-wave mixing chi^(3) nonlinear interaction transfers the Stokes 10.0 MHz EOM modulation onto the continuous pump beam via Stimulated Raman Loss (SRL).
- Dual-filter optical suppression of the Stokes beam (FESH0850 + FESH0800, cumulative OD > 12) eliminates direct optical feedthrough at 1040 nm.
- Photodiode conversion on DET10N2 incorporating responsivity, laser RIN, shot noise, and thermal Johnson noise.
- RF lock-in demodulation at 10.0 MHz extracting the recovered Raman signal voltage, noise floor, and SNR adhering to theoretical SRS scaling laws (S proportional to I_pump * I_stokes * N * sigma_Raman).

### R8. Interactive Tools for Building & Simulating the Spectral Focusing SRS Microscope
Provide a dedicated suite of tools accessible programmatically and via the UI:
1. SRS Microscope Setup Builder Tool: Configures and constructs the complete 2D optical system from user-specified parameters (laser powers/wavelengths, dispersive rod glass/lengths, objective presets including Olympus 20x 1.0 NA W and 25x 1.05 NA W, filter selections), populating the Lens Data Editor and initiating 2D ray tracing.
2. Galvo Scanning & Field Diagnostic Tool: Computes FOV dimensions, field curvature, chromatic walk-off, and off-axis signal degradation profiles across scan angles.
3. Spectral Focusing Simulation & Delay Scan Tool: Computes GVD/GDD, stretched pulse durations (tau_p, tau_S), chirp matching ratios (alpha_p / alpha_S), delays vs. beat frequency calibrations Delta nu(tau_d), and simulates delay sweeps reconstructing the full vibrational Raman spectrum.
4. Signal Processing & Lock-In Demodulation Tool: Computes photon dwell counts, lock-in voltages (R, X, Y), noise contributions (shot noise, RIN, thermal noise), SNR, and scan field degradation factors.
5. Optical Bench Diagnostic & Alignment Tool: Verifies objective back focal plane conjugation, checks pupil walk-off (Delta r_walk-off), evaluates clear aperture transmission, and confirms zero vignetting across all 37 surfaces.

## Acceptance Criteria

### General-Purpose Integrity & Non-Regression
- [ ] Existing general-purpose optical workflows (generic LDE editing, custom system creation, 2D ray tracing, ray fans, spot diagrams) pass all automated regression tests without behavioral change.
- [ ] Loading the SRS microscope template dynamically populates the Lens Data Editor and 2D visualizer with all catalog components, immediately launching 2D ray tracing that traverses all surfaces to the photodiode with zero vignetting.

### Active EOM Modulation & Dual-Laser Delivery
- [ ] The dual-laser delivery system incorporates both lasers (Pump 797 nm and Stokes 1040 nm), independent polarization power controls with beam dumps, clean-up filters, and dichroic collinear coupling.
- [ ] The Stokes beam is actively modulated via the EOM crystal and PBS analyzer pair at 10.0 MHz, producing high-contrast intensity modulation with rejected power directed to a dedicated dump.
- [ ] Stimulated Raman Loss (SRL) correctly transfers the 10.0 MHz modulation to the pump beam, which is detected and demodulated by the RF lock-in amplifier above the laser RIN floor.

### Commercial Scan Lens & Objective Zemax Prescriptions
- [ ] The Olympus 20x 1.0 NA Water Immersion objective model and 25x 1.05 NA model are accurately defined with their authentic Zemax multi-surface parameters (curvatures, glass, thicknesses, cover slip, and water immersion n = 1.333).
- [ ] The commercial scan lens (Thorlabs SL50-2P2 or Edmund Optics equivalent) produces telecentric scanning into the intermediate image plane with telecentricity error < 0.1 deg.
- [ ] Beam path evolution adheres strictly to physical propagation zones with 100% unvignetted transmission along the entire 37-surface train.

### Galvo Scanning & Field Aberration Modeling
- [ ] Dynamic ray tracing verifies beam deflection across the full scan angle range (+/- 10.0 deg).
- [ ] The field simulation quantitatively evaluates off-axis Petzval field curvature, chromatic walk-off between 797 nm and 1040 nm, pupil vignetting, and off-axis SRS signal roll-off.
- [ ] The inter-galvo 4f relay maintains strictly zero pupil walk-off (Delta r_walk-off = 0.0 um) at the objective BFP across the entire scan range.

### 2D Layout Design & Visualization
- [ ] In the 2D projection views (Y-Z and X-Z), optical elements across both branches maintain proper spatial clearance with zero unphysical visual overlap, clipping, or component collisions.
- [ ] Reflected optical axes and ray segments accurately follow specular reflection laws in the 2D layout canvas, with auto-zoom/view limits cleanly framing the full folded bench upon loading.

### Spectral Focusing & Signal Processing Verification
- [ ] The spectral focusing simulation generates physically validated Raman spectra matching expected peak positions and linewidths for standard Raman samples (e.g., mineral oil, olive oil).
- [ ] The signal processing module outputs lock-in demodulated voltages, noise metrics, and SNR adhering to theoretical SRS scaling laws (S proportional to I_pump * I_stokes * N * sigma_Raman).

### Building & Simulation Tools
- [ ] The Setup Builder tool reliably instantiates the full SRS system with customizable objective presets (Olympus 20x 1.0 NA W vs. Olympus 25x 1.05 NA W vs. Thorlabs TL15X-2P), glass materials (SF57, SF11, Fused Silica), rod lengths, and filter options.
- [ ] The Delay Scan, Field Simulation, and Signal Processing tools execute full hyperspectral and field scans, returning structured data in both GUI tables and scriptable data structures.
- [ ] The Alignment Diagnostic tool reports quantitative metrics for pupil walk-off, beam diameter at the objective BFP, fill factor, and optical density across the filter stack.

### Code Quality & Dual-Tree Parity
- [ ] Automated test suite in tests/ executes with a 100% pass rate.
- [ ] Full dual-tree code parity is verified between optics_app/ and v2/ with zero unexpected diffs.


## Follow-up — 2026-10-15T12:46:07Z

Enhance the 2D_Optics design studio with a production-grade optical optimization engine strictly following Zemax industry standards, optimization strategies, and parameter controls for real imaging conditions and manufacturability.

Working directory: c:\Users\Juan\Documents\GitHub\2D_Optics
Integrity mode: development

## Requirements

### R1. Zemax-Standard Multi-Field & Pupil Sampling Merit Function
Implement the standard Zemax Default Merit Function architecture:
- **Field Sampling & Weighting**: Standard normalized field coordinates $H_y \in [0.0, 0.707, 1.0]$ with standard area/quadrature weighting to balance Petzval curvature, astigmatism, coma, and distortion across the full image format.
- **Pupil Grid Sampling**: Hexapolar ray distribution (rings 1–4, 6 rays per ring) or Gaussian quadrature grid across the entrance pupil.
- **Optimization Targets**: RMS spot radius (centroid and chief-ray referenced) and paraxial Rayleigh-Hamilton RMS wavefront error OPD.

### R2. Zemax Boundary Operands & Parameter Controls
Enforce standard Zemax boundary operands directly inside the merit function and optimizer bounds:
- **`MNCT` & `MXCT`**: Minimum and maximum element center thickness ($1.5 \le CT \le 18.0\,\text{mm}$).
- **`MNET` & `MXET`**: Minimum and maximum edge thickness ($ET \ge 1.0\,\text{mm}$) computed at the true clear semi-diameter $y_{\text{sd}}$ via exact sag: $ET = CT - \left(s_2(y_{\text{sd}}) - s_1(y_{\text{sd}})\right)$.
- **`MNEA`**: Minimum air edge thickness ($ET_{\text{air}} \ge 0.5\,\text{mm}$) to prevent physical lens-to-lens mechanical collision.
- **`EFFL`**: Effective focal length constraint formulated continuously via paraxial ray matrix power ($\Phi = -C$).
- **`TOTR`**: Maximum total track length limit.
- **Variable Parameter Space**: Curvature space optimization ($c = 1/R$) with variable bounds clamping to eliminate infinite radius singularities.

### R3. Robust Optimization Strategies (DLS & Global Search)
- **Damped Least Squares (DLS)**: Levenberg-Marquardt damping adaptation ($\lambda_{k+1} = \lambda_k \times \beta$) ensuring stable, quadratic descent without matrix ill-conditioning.
- **Escape Local Minima**: Global multi-start / Hammer optimization perturbing variables within manufacturing bounds.

### R4. Seamless Application & UI Delivery
- Optimization results must automatically update the active `LensDataTableModel` and `SurfaceStack`, immediately refreshing the LDE table, 2D/3D visualizer, multi-field ray traces, and optimizer dock status.
- Ensure 100% byte-for-byte synchronization between `optics_app/` and `v2/` trees.

## Verification Resources
- Existing test suites:
  - `tests/test_manufacturing_constraints.py`
  - `tests/test_zemax_optimizer.py`
  - `tests/test_first_order_diagnostics.py`
  - `tests/test_achromat_result_delivery.py`

## Acceptance Criteria

### Zemax Merit Function & Multi-Field Verification
- [ ] Optimization of a doublet and Cooke triplet across 3 fields ($0.0, 0.707, 1.0$) demonstrates monotonic convergence and reduction in composite RMS spot radius.
- [ ] Multi-field hexapolar ray bundles trace reliably without ray failures, TIR traps, or vignetting dropouts.

### Zemax Boundary & Manufacturing Verification
- [ ] All optimized elements strictly satisfy $MNCT \ge 1.5\,\text{mm}$, $MNET \ge 1.0\,\text{mm}$, and $MNEA \ge 0.5\,\text{mm}$.
- [ ] No physical lens element has negative edge thickness, knife edges, or element intersections.
- [ ] The manufacturing audit tab (`SystemManufacturingReport`) reports 0 critical violations.

### Application Integration & Test Suite
- [ ] Final optimized prescriptions are fully applied to the active LDE table, 2D/3D CAD visualizers, and optimizer dock.
- [ ] Full pytest suite passes cleanly across both `optics_app/` and `v2/` trees.
