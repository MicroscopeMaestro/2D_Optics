# 2D Raman DMD Spectrometer — Full Project Documentation

> **Purpose:** This document is a complete handoff guide for any AI (or human) continuing the CAD development of this project. It covers what was built, why decisions were made, what files exist, and what remains to be done.

---

## 1. Project Overview

This is a **3D-printable 2D Raman DMD Spectrometer** designed and iterated in FreeCAD using headless Python scripting. The spectrometer uses a **DLP4500 EVM (TI LightCrafter)** as its spatial light modulator (DMD), and the casing is printed in PLA on a Prusa printer.

The system consists of:
- **Bottom Casing Half:** \support_bottom_half.step\ (single continuous structure holding the lower optical channels and M4 threaded holes).
- **Top Casing Structure (Model 5 base, 2-piece split):**
  - \	op_dmd_region.step\: Removable DMD access piece containing the DMD aperture, 2 DMD adapter mount holes, and 3 vertical M4 clamshell clamping holes to bolt into the bottom half.
  - \	op_optics_region.step\: The main top body containing all other optical stations (slit, collimator, filter, grating, lenses, sensor).
- **2-part DMD Mount Adapter:** \DMD4500_Alignment_Frame_Final.step\ + \DMD4500_Clamping_Lid_Final.step\ (attaches the DLP4500 EVM to the DMD aperture).

**Working directory:** `C:\Users\Juan\Documents\GitHub\2D_Optics\3D Raman DMD spectrometer\`
**FreeCAD executable:** `C:\Program Files\FreeCAD 1.1\bin\FreeCADCmd.exe`
**Output directory for printing:** `C:\Users\Juan\Desktop\Spectrometer_Print_Files\`

> **CRITICAL CONSTRAINT:** The spectrometer casing halves (`support_top_half.step`, `support_bottom_half.step`) have already been physically 3D printed and must **NOT** be modified or regenerated. All future changes must be limited to the DMD mount adapter scripts only.

---

## 2. How to Run Scripts (Headless FreeCAD)

All scripts must be executed via FreeCAD's headless interpreter. **Never use system Python** — it will fail with `ImportError: Module use of python311.dll conflicts`.

```powershell
# From the working directory:
& "C:\Program Files\FreeCAD 1.1\bin\FreeCADCmd.exe" script_name.py
```

FreeCAD outputs verbose STEP statistics to stdout on success. If a script outputs `Part 1 Volume: 0.00`, the boolean operation silently failed (see Section 7 for known issues).

---

## 3. File Map

### 3.1 Core CAD Scripts

| Script | Purpose |
|---|---|
| `build_spectrometer.py` | Main clamshell generator. Reads `model1.FCStd`, extrudes the sketch profile, cuts all optical pockets, grating slot, DMD window, M4 holes, interlock grooves. **DO NOT re-run** (casing already printed). |
| `generate_flush_dmd4500_mount.py` | **Primary active script.** Generates the 2-part DLP4500 DMD mount (Alignment Frame + Clamping Lid). Pure FreeCAD Part API, no external dependencies. Run this to iterate on the mount. |

### 3.2 Output STEP Files

| File | Description | Status |
|---|---|---|
| `top_dmd_region.step` | **Active Top Part 1:** Removable DMD region (with 3 bottom-clamping M4 holes + 2 DMD mount M4 holes) | **ACTIVE** |
| `top_optics_region.step` | **Active Top Part 2:** Main optics top body (Model 5 base geometry) | **ACTIVE** |
| `support_bottom_half.step` | Bottom clamshell half (printed base reference) | **FROZEN** |
| `support_top_half_model5.step` | Monolithic Model 5 reference top half | Reference |
| `DMD4500_Alignment_Frame_Final.step` | DMD mount frame (3.8mm thick, has DMD pocket + heatsink pocket) | ACTIVE |
| `DMD4500_Clamping_Lid_Final.step` | DMD clamping lid (3.0mm thick, has fin aperture, no pockets) | ACTIVE |
| `external_slit_cartridge_200um_height_2mm.step` | 200 µm optical slit cartridge (2.0mm thick, 200µm opening along Y / height, beveled knife-edge) | **ACTIVE** |
| `external_slit_cartridge_200um_2mm.step` | 200 µm optical slit cartridge (2.0mm thick, 200µm opening along X / width, beveled knife-edge) | **ACTIVE** |
| `support_top_half.step` | Old monolithic top half (superseded by Model 5 split) | Obsolete |
| `support_top_half_m4.step` | Older revision with baked M4 holes | Archive |
| `support_bottom_half_m4.step` | Older revision with baked M4 holes | Archive |
| `DMD_Adapter.step` | Legacy single-piece adapter (superseded) | Legacy |
| `PDAPC3_Adapter.step` / `.stl` | Active flat adapter for Thorlabs PDAPC3 photodetector (M4 pass-through + accessory grid) | **ACTIVE** |
| `PDAPC3_XYZ_Stage1_Base.step` / `.stl` | Stage 1 (Base Plate): M4 counterbored mount to casing + trapped M4 hex nuts | **ACTIVE** |
| `PDAPC3_XYZ_Stage2_XY_Slider.step` / `.stl` | Stage 2 (XY Slider): ±2.75 mm cross-slots + split-collar pinch clamp | **ACTIVE** |
| `PDAPC3_XYZ_Stage3_Z_Carrier.step` / `.stl` | Stage 3 (Z Carrier): 8 mm axial focus barrel + 37.8mm PDAPC3 pocket | **ACTIVE** |
| `PDAPC3_Light_Shield_Hood.step` / `.stl` | Light-tight stray-light outer enclosure hood with labyrinth seam & cable slot | **ACTIVE** |
| `PDAPC3_Light_Shield_Cable_Cap.step` / `.stl` | Baffled light-trap cable clamp cap | **ACTIVE** |
| `PDAPC3_Full_System_Assembly.step` | Complete assembled XYZ mount & light-tight enclosure system | **ACTIVE** |
| `PDAPC3_Slim_Stage1_Base.step` / `.stl` | **Slim-Profile Stage 1 Base:** 40x54mm footprint (only 2.5mm table drop) | **ACTIVE** |
| `PDAPC3_Slim_Stage2_XY_Slider.step` / `.stl` | **Slim-Profile Stage 2 XY Slider:** ±2.0mm XY travel + vertical pinch clamp | **ACTIVE** |
| `PDAPC3_Slim_Stage3_Z_Carrier.step` / `.stl` | **Slim-Profile Stage 3 Z Carrier:** 8mm axial focus + PDAPC3 form pocket | **ACTIVE** |
| `PDAPC3_Slim_Light_Shield_Hood.step` / `.stl` | **Slim-Profile Light Shield Hood:** 40x54x68mm, clears 3.5-5mm rubber feet | **ACTIVE** |
| `PDAPC3_Slim_Light_Shield_Cable_Cap.step` / `.stl` | **Slim-Profile Cable Cap:** Baffled cable clamp | **ACTIVE** |
| `Spectrometer_Leveling_Feet_5mm.stl` | Set of 4x 5mm round leveling riser feet for spectrometer bottom | **ACTIVE** |
| `PDAPC3_Slim_Full_Assembly.step` | Complete assembled Slim XYZ mount & light-tight enclosure system | **ACTIVE** |

### 3.3 Helper / Analysis Scripts

| Script | Purpose |
|---|---|
| `check_bbox.py` | Prints bounding box of any STEP file |
| `split_top.py` | WIP: Attempts to split `support_top_half_m4.step` diagonally — see Section 8 |
| `build_external_bed.py` | Generates the external aspheric lens cartridge slip-fit mount |
| `build_pdapc3_adapter.py` | Generates PDAPC3 sensor adapter |

---

## 4. Spectrometer Casing — Geometry Reference

### 4.1 Casing Coordinate System

The casing sketch lives in `model1.FCStd` (FreeCAD document) and is extruded along the **X axis**:
- `top_half` is extruded in `+X` direction (thickness = `CLAMSHELL_HALF_THICKNESS = 17.5 mm`)
- `bottom_half` is extruded in `-X` direction (same thickness)
- The split plane is at `X = 0`

The optical axis runs along **Z**. **Y** is the lateral/transverse axis.

### 4.2 Key Z Positions (optical stations)

| Z (mm) | Description |
|---|---|
| ~15.36 | Front face of casing / entrance aperture (first aspheric lens) |
| ~80.67 | 200 µm input slit |
| ~95–234 | DMD housing region |
| ~233 | Sensor / detector exit |

### 4.3 DMD Region

The DMD optical element in the FreeCAD assembly is labelled `22_CB_After_DMD_Hardware`. Key facts:
- **M4 mounting holes** in the printed casing are at local `X = ±15.0`, `Y = ±15.0` (30 mm spacing pattern)
- The **casing floor** is at global `Y = -86.0`, which in the DMD's local frame sits at `Y = -19.88` relative to the DMD optical center
- The mount bottom edge was truncated to `Y = -18.0` to leave a **1.88 mm guaranteed air gap** above the floor

### 4.4 Casing Bounding Box (top half)

From `support_top_half_m4.step`:
```
X:  0.00 → 17.50   (casing wall thickness)
Y: -82.07 → 25.00  (lateral extent)
Z:  0.00 → 233.98  (optical axis length)
```

---

## 5. DMD Mount — Design Details

### 5.1 DLP4500 EVM Physical Dimensions

| Parameter | Value |
|---|---|
| DMD ceramic package length | 20.9 mm |
| DMD ceramic package width | 9.3 mm |
| Heatsink length | 43.5 mm |
| Heatsink width | 15.5 mm (widened by 1 mm from original 14.5 mm) |
| Active micromirror offset from package center | +0.9 mm in Y (`pocket_y_offset`) |
| M4 hole pattern spacing | 30 mm × 30 mm |
| M4 clearance hole diameter | 4.3 mm (Close Fit for PLA) |
| M4 clearance hole radius in script | `m4_tilt_clearance = 2.15` |

### 5.2 Mount Geometry

**Alignment Frame** (`DMD4500_Alignment_Frame_Final.step`):
- 45 mm wide × 36.5 mm tall (Y: -18.0 → +18.5) × 3.8 mm thick
- Has a **DMD pocket** (20.9 × 9.3 mm) recessed 1.0 mm from the front face (Z=1.0 to Z=3.8)
- Has a **heatsink pocket** (43.5 × 15.5 mm) cut from the back face (Z=0 to Z=1.0)
- Has 4× M4 clearance holes at ±15 mm corners
- Top edge uniformly truncated to Y = +18.5 mm

**Clamping Lid** (`DMD4500_Clamping_Lid_Final.step`):
- 45 mm wide × 35.5 mm tall (Y: -18.0 → +17.5) × 3.0 mm thick
- Has a **fin aperture** (40 × 12 mm) for heatsink cooling fin access
- Has 4× M4 clearance holes at ±15 mm corners
- Top edge uniformly truncated to Y = +17.5 mm (shorter than frame)

### 5.3 Why the bottom is truncated to Y = -18.0

The printed spectrometer casing floor is at `Y = -19.88` (in DMD local frame). The original 45×45 mm square mount extended to `Y = -22.5`, causing it to collide with the floor before the M4 holes (`Y = -15.0`) could engage. Truncating to `Y = -18.0` leaves a 1.88 mm air gap.

### 5.4 Why the top is truncated

The top "ears" of the mount (at Y = +22.5) were colliding with cabling inside the spectrometer housing. After trying an asymmetric top-right corner cut (rejected as non-uniform), the final approach is:
- Frame: uniform flat top at Y = +18.5 mm
- Lid: uniform flat top at Y = +17.5 mm (even shorter for extra clearance)

---

## 6. generate_flush_dmd4500_mount.py — Annotated Reference

```python
# Key parameters:
m4_tilt_clearance = 2.15   # Radius of M4 clearance hole (4.3mm diameter = Close Fit for PLA)
hole_spacing = 30.0         # M4 bolt pattern: ±15 mm in X and Y

pocket_y_offset = 0.9       # DMD active array is 0.9mm off-center of ceramic package
                            # This offsets the DMD/heatsink pockets so optics are at Y=0

heatsink_length = 43.5      # mm (original TI spec)
heatsink_width = 15.5       # mm (original was 14.5, widened by 1mm for clearance)

dmd_length = 20.9           # DLP4500 FQE package
dmd_width = 9.3

y_min = -18.0               # Truncated bottom (originally -22.5); leaves 1.88mm gap above floor
# top_y passed per-piece: Frame=18.5, Lid=17.5

def make_efficient_robust_base(thickness, top_y):
    # Creates the rectangular base. top_y controls where the top is uniformly cut.
    actual_height = top_y - y_min  # = 36.5mm for frame, 35.5mm for lid
    box = Part.makeBox(45.0, actual_height, thickness, FreeCAD.Vector(-22.5, y_min, 0))
    return box
```

**To regenerate and deploy:**
```powershell
cd "C:\Users\Juan\Documents\GitHub\2D_Optics\3D Raman DMD spectrometer"
& "C:\Program Files\FreeCAD 1.1\bin\FreeCADCmd.exe" generate_flush_dmd4500_mount.py
cp DMD4500_Alignment_Frame_Final.step "C:\Users\Juan\Desktop\Spectrometer_Print_Files\"
cp DMD4500_Clamping_Lid_Final.step "C:\Users\Juan\Desktop\Spectrometer_Print_Files\"
```

---

## 7. FreeCAD API — Critical Pitfalls

### 7.1 The transformShape + boolean = Volume 0 bug

When a `Part.Shape` returned by `Part.makeBox(w, h, d, FreeCAD.Vector(x,y,z))` is passed through `transformShape(matrix)` and then used in a boolean `.common()` or `.cut()` operation, the result silently returns **Volume = 0**.

**Root cause:** After `makeBox` with a position vector, calling `transformShape` on the result corrupts the topology for subsequent boolean ops.

**Workaround:** Create the box at origin, call `transformShape`, then use for booleans:
```python
# WORKS:
box = Part.makeBox(w, h, d)           # at origin
plc = FreeCAD.Placement(pos, rot)
box = box.transformShape(plc.Matrix)  # transform in place
result = shape.cut(box)               # boolean works

# ALSO WORKS (simple axis-aligned, no transform):
box = Part.makeBox(500, 500, 500, FreeCAD.Vector(-250, -250, 95))
result = shape.common(box)  # works correctly
```

### 7.2 Always verify boolean success
```python
result = shape.common(tool)
print(f"Volume: {result.Volume:.2f}")  # Must be > 0
```

### 7.3 CenterOfGravity vs CenterOfMass
```python
# WRONG - AttributeError:
shape.CenterOfMass

# CORRECT:
shape.Solids[0].CenterOfGravity  # Returns FreeCAD.Vector
```

### 7.4 Check a STEP file from command line
```powershell
& "C:\Program Files\FreeCAD 1.1\bin\FreeCADCmd.exe" -c "
import sys; sys.path.append(r'C:\Program Files\FreeCAD 1.1\bin')
import FreeCAD, Part
s = Part.Shape(); s.read('your_file.step')
print('Type:', s.ShapeType, '| Solids:', len(s.Solids), '| Volume:', round(s.Volume,2))
print('BBox X:', round(s.BoundBox.XMin,2), 'to', round(s.BoundBox.XMax,2))
print('BBox Y:', round(s.BoundBox.YMin,2), 'to', round(s.BoundBox.YMax,2))
print('BBox Z:', round(s.BoundBox.ZMin,2), 'to', round(s.BoundBox.ZMax,2))
"
```

---

## 8. Outstanding Work

### 8.1 Diagonal / Regional Split of top casing half — BLOCKED

**User request:** Split `support_top_half.step` into two pieces:
- One piece: the **DMD region** (Z ≈ 95 → 234 mm, Y ≈ -86 to 0)
- One piece: the **other optical elements** (Z ≈ 0 → 95 mm)

This would let each region be printed/modified independently. The clamshell split is along X; this new cut would be along Z (or diagonally in YZ).

**Status:** Blocked by the `transformShape` bug (Section 7.1). All approaches using rotated boxes returned Volume = 0.

**Recommended next approach (simple, working):**
```python
shape = Part.Shape()
shape.read('support_top_half.step')

# Axis-aligned cut at Z = 95 (boundary of DMD housing region)
dmd_region_box  = Part.makeBox(500, 500, 200, FreeCAD.Vector(-250, -250, 95))
optics_region_box = Part.makeBox(500, 500, 95,  FreeCAD.Vector(-250, -250, 0))

dmd_part    = shape.common(dmd_region_box)
optics_part = shape.common(optics_region_box)

print("DMD part volume:", dmd_part.Volume)      # Should be > 0
print("Optics part volume:", optics_part.Volume) # Should be > 0

dmd_part.exportStep("top_dmd_region.step")
optics_part.exportStep("top_optics_region.step")
```

If a **diagonal** cut is required, construct the cutting solid from a polygon face extruded along X (do NOT use `makeBox` + `transformShape`):
```python
import math
# Define diagonal cutting plane in YZ at X=0, then extrude along X
y0, z0 = -82.07, 0      # Bottom-left corner of casing
y1, z1 = 25.0, 233.98   # Top-right corner of casing

# Split diagonal: from (y0, z0+95) to (y1, z0)
# or however the user wants the diagonal orientation
pts = [
    FreeCAD.Vector(0, y0,  95),
    FreeCAD.Vector(0, y1,   0),
    FreeCAD.Vector(0, y1, 300),
    FreeCAD.Vector(0, y0, 300),
    FreeCAD.Vector(0, y0,  95),
]
wire = Part.makePolygon(pts)
face = Part.Face(wire)
cutter = face.extrude(FreeCAD.Vector(100, 0, 0))
cutter.translate(FreeCAD.Vector(-50, 0, 0))

part1 = shape.common(cutter)
part2 = shape.cut(cutter)
```

### 8.2 DMD Mount — Possible Improvements

- **Alignment pins:** Add 2 mm dowel pin holes at 2 diagonal corners of the mount to guarantee repeatable assembly
- **Cable relief notch:** If the Y = +17.5 / +18.5 truncation is still insufficient, carve a small notch into the top edge
- **Heatsink fin check:** Verify the `40 × 12 mm` fin aperture aligns with the actual heatsink fin position on the DLP4500 EVM board

---

## 9. Print Settings & Hardware Notes

| Parameter | Value |
|---|---|
| Printer | Prusa (MK series) |
| Material | PLA |
| M4 hole diameter for PLA | **4.3 mm** (Close Fit, accounts for plastic shrinkage) |
| M4 hole radius in scripts | `m4_tilt_clearance = 2.15` |
| Layer height | Standard (0.2 mm recommended) |

---

## 10. build_spectrometer.py — Key Constants Reference

For context only (casing is printed; do not re-run):

```python
CLAMSHELL_HALF_THICKNESS = 17.5    # mm — wall thickness each side
OPTICS_POCKET_RADIUS     = 12.90   # mm — standard lens pocket radius
ASPHERIC_POCKET_RADIUS   = 13.05   # mm — aspheric lens (ACL25416U) pocket
SLIDING_TOLERANCE        = 0.4     # mm — axial play for lens positioning
DMD_M4_HOLE_DEPTH        = 10.0    # mm — depth of M4 tap holes for DMD mount
THREAD_TAP_RADIUS        = 1.65    # mm — M4 tap drill size
ENABLE_PERIMETER_INTERLOCK = True  # Tongue-and-groove between top/bottom halves

DEFAULT_TOP_STEP_OUTPUT    = "support_top_half.step"
DEFAULT_BOTTOM_STEP_OUTPUT = "support_bottom_half.step"
```

---

## 11. Top Half Split — Full Groove Coverage for Printed Bottom with Tongue (Updated 2026-09-24)

### Printed Bottom Half Interlock Compatibility
The physical printed bottom half (\support_bottom_half.step\) features 3.0 mm high perimeter tongues extending along the split plane into the top half ( > 0$), including the critical outer lateral tongue along the DMD housing from  = -103.8$ to $-82.9$ mm ( = 96.2$ to .0$ mm).

To ensure complete, flush seating with zero collision:
- **Base model:** \support_top_half.step\ ( \in [-105.0, 25.0	ext{ mm}]$,  \in [0.0, 17.5	ext{ mm}]$).
- **Mating face:** Flat at  = 0.0$ (no extruding surfaces sticking out into  < 0$).
- **Matching grooves:** 3.2 mm deep matching groove channels ( \in [0.0, 3.2	ext{ mm}]$) with 0.2 mm lateral clearance, achieving **.0000	ext{ mm}^3$ collision** with all bottom tongues.
  - Outer DMD groove: .48	ext{ mm}^2$ ( \in [-104.0, -85.9]$,  \in [96.0, 156.0]$)
  - Inner ridge groove: .14	ext{ mm}^2$ ( \in [-49.9, -35.0]$,  \in [98.5, 108.9]$)
  - Central ridge groove: .35	ext{ mm}^2$ ( \in [-25.2, -20.0]$,  \in [115.0, 118.2]$)
  - Total DMD groove area: **.97	ext{ mm}^2$**.

### Holes Captured in \	op_dmd_region.step\ (Volume: 42,717.04 mm³, ~23.3%):
1. **3× Vertical Clamshell M4 Holes (along global X):**
   - **Screw 5 (DMD lateral 1):**  = -76.0, Z = 136.0$
   - **Screw 6 (DMD lateral 2):**  = -76.0, Z = 150.0$
   - **Screw 7 (MID CENTER ridge):**  = -26.0, Z = 122.0$
   *Forms a rigid triangular clamp directly bolting into the bottom half.*
2. **2× DMD Adapter Mounting M4 Holes (along DMD normal):**
   -  = 15.0, Y = -81.56, Z = 107.23$
   -  = 15.0, Y = -52.64, Z = 99.26$

### The Rest (\	op_optics_region.step\, Volume: 140,650.14 mm³, ~76.7%):
- Single continuous solid holding all other optics (slit, grating, collimator, lenses, sensor) and remaining clamping holes.
- Exact volume conservation: ,717.04 + 140,650.14 = 183,367.18	ext{ mm}^3$ (.0001	ext{ mm}^3$ diff).
- 100% collision-free (.0000	ext{ mm}^3$) with bottom half perimeter tongues.

Generator script: \uild_true_matching_top.pyFiles deployed to: \C:\Users\Juan\Desktop\Spectrometer_Print_Files\