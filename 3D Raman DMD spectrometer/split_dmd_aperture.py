import sys
import os
sys.path.append(r'C:\Program Files\FreeCAD 1.1\bin')
import FreeCAD
import Part

# ==============================================================================
# CONFIGURATION
# ==============================================================================
INPUT_FILE = r"C:\Users\Juan\Documents\GitHub\2D_Optics\3D Raman DMD spectrometer\support_top_half.step"
OUTPUT_DIR = r"C:\Users\Juan\Documents\GitHub\2D_Optics\3D Raman DMD spectrometer"
OUTPUT_PREFIX = "top"
TRIM_Y_TO_86 = True

# DMD Aperture piece components (enable/disable as needed)
DMD_CONFIG = {
    "aperture": {
        "enabled": True,
        "bounds": {"x_min": 0, "x_max": 17.5, "y_min": -86, "y_max": -20, "z_min": 95, "z_max": 125},
    },
    "lens_section": {
        "enabled": True,
        "bounds": {"x_min": 0, "x_max": 17.5, "y_min": -86, "y_max": -20, "z_min": 125, "z_max": 147},
    },
    "upper_housing": {
        "enabled": True,
        "bounds": {"x_min": 0, "x_max": 17.5, "y_min": -86, "y_max": -20, "z_min": 147, "z_max": 166},
    },
    "side_extension": {
        "enabled": True,
        "bounds": {"x_min": 0, "x_max": 17.5, "y_min": -40, "y_max": -20, "z_min": 95, "z_max": 166},
    },
}

DMD_NAME = "dmd_aperture_focused"
REST_NAME = "rest"

# ==============================================================================
# EXECUTION
# ==============================================================================

print("=" * 70)
print("SPLIT TOP - DMD Aperture Focused Split")
print("=" * 70)

# Load top half
top = Part.Shape()
top.read(INPUT_FILE)
print(f"Loaded: {top.Volume:.2f} mm³")

# Trim Y if needed
if TRIM_Y_TO_86:
    trim_box = Part.makeBox(500, 500, 500, FreeCAD.Vector(-250, -86, -50))
    top = top.common(trim_box)
    print(f"Trimmed to Y=-86: {top.Volume:.2f} mm³")

# Build DMD piece
def make_box(bounds):
    return Part.makeBox(
        bounds["x_max"] - bounds["x_min"],
        bounds["y_max"] - bounds["y_min"],
        bounds["z_max"] - bounds["z_min"],
        FreeCAD.Vector(bounds["x_min"], bounds["y_min"], bounds["z_min"])
    )

dmd_piece = None
for name, component in DMD_CONFIG.items():
    if not component["enabled"]:
        continue
    box = make_box(component["bounds"])
    piece = top.common(box)
    if dmd_piece is None:
        dmd_piece = piece
    else:
        dmd_piece = dmd_piece.fuse(piece)
    print(f"  {name}: {piece.Volume:.2f} mm³")

print(f"Total DMD piece volume: {dmd_piece.Volume:.2f} mm³")

# Rest piece
cut_box = None
for name, component in DMD_CONFIG.items():
    if not component["enabled"]:
        continue
    box = make_box(component["bounds"])
    if cut_box is None:
        cut_box = box
    else:
        cut_box = cut_box.fuse(box)

rest_piece = top.cut(cut_box)

# Print M4 holes
def print_m4_holes(shape, label):
    print(f"\nM4 holes in {label}:")
    found = False
    for i, face in enumerate(shape.Faces):
        surf = face.Surface
        if hasattr(surf, 'Radius') and surf.Radius > 1.5 and surf.Radius < 3.0:
            com = face.CenterOfMass
            print(f"  Face {i}: Radius={surf.Radius:.2f} at X={com.x:.2f}, Y={com.y:.2f}, Z={com.z:.2f}")
            found = True
    if not found:
        print("  (none found)")

print_m4_holes(dmd_piece, "DMD Aperture Focused")
print_m4_holes(rest_piece, "Rest")

total_volume = dmd_piece.Volume + rest_piece.Volume
print(f"\nTotal volume: {total_volume:.2f} mm³")

# Export
OUTPUT_DIR = r"C:\Users\Juan\Documents\GitHub\2D_Optics\3D Raman DMD spectrometer"
OUTPUT_PREFIX = "top"
dmd_piece.exportStep(os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_dmd_aperture_focused.step"))
rest_piece.exportStep(os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_rest.step"))
print(f"\nExported: top_dmd_aperture_focused.step ({dmd_piece.Volume:.2f} mm³)")
print(f"Exported: top_rest.step ({rest_piece.Volume:.2f} mm³)")
print(f"Total volume: {total_volume:.2f} mm³")
print("Done!")