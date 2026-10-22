#!/usr/bin/env python3
"""
================================================================================
SPECTROMETER TOP HALF - Complete Section Control System
================================================================================

A comprehensive Python script for splitting the spectrometer top half
(support_top_half.step) into multiple precisely controlled sections.

FEATURES:
- Define unlimited sections with exact boundaries
- Each section has independent bounds, exclusions, inclusions
- Automatic Y-trim to match printing_model_5.stl (Y=-86)
- Volume preservation verification
- M4 hole detection and reporting per section
- Clean STEP file export with custom naming
- Section overlap detection and validation
- Visual summary of all sections

USAGE:
    & "C:\Program Files\FreeCAD 1.1\bin\FreeCADCmd.exe" split_top_half_complete.py

CONFIGURATION:
    Edit the SECTIONS list below to define your sections.
    Each section is a dictionary with full control over boundaries.

================================================================================
"""

import sys
import os

# Add FreeCAD to path
sys.path.append(r'C:\Program Files\FreeCAD 1.1\bin')
import FreeCAD
import Part


# ==============================================================================
# USER CONFIGURATION - DEFINE ALL YOUR SECTIONS HERE
# ==============================================================================

# -----------------------------------------------------------------------------
# INPUT/OUTPUT SETTINGS
# -----------------------------------------------------------------------------
INPUT_FILE = r"C:\Users\Juan\Documents\GitHub\2D_Optics\3D Raman DMD spectrometer\support_top_half.step"
OUTPUT_DIR = r"C:\Users\Juan\Documents\GitHub\2D_Optics\3D Raman DMD spectrometer"
OUTPUT_PREFIX = "top"

# -----------------------------------------------------------------------------
# GLOBAL SETTINGS
# -----------------------------------------------------------------------------
TRIM_Y_TO_86 = True          # Trim to Y=-86 (matches printing_model_5.stl)
VALIDATE_NO_OVERLAP = True   # Check for section overlaps
VALIDATE_COMPLETE = True     # Check if sections cover the trimmed model

# -----------------------------------------------------------------------------
# SECTION DEFINITIONS
# Each section is a dictionary with the following keys:
#   name:           Output filename suffix (e.g., "dmd" -> top_dmd.step)
#   bounds:         Main bounding box {x_min, x_max, y_min, y_max, z_min, z_max}
#   exclude:        List of boxes to REMOVE from this section
#   include:        List of boxes to ADD to this section
#   color:          Optional color for visualization (not used in STEP)
#   description:    Human-readable description
# -----------------------------------------------------------------------------

SECTIONS = [
    # ==========================================================================
    # SECTION 1: DMD Aperture Region (Z=95-125)
    # Contains: DMD aperture, DMD mount M4 holes (tap radius 1.65mm)
    # ==========================================================================
    {
        "name": "dmd_aperture",
        "description": "DMD aperture region (Z=95-125) with DMD mount holes",
        "bounds": {
            "x_min": 0.0, "x_max": 17.5,
            "y_min": -86.0, "y_max": -20.0,
            "z_min": 95.0, "z_max": 125.0,
        },
        "exclude": [],
        "include": [],
    },
    
    # ==========================================================================
    # SECTION 2: Lens Section (Z=125-147) - REMOVED from DMD, goes to rest
    # Contains: Focus lens pocket, joining M4 hole at Z=136
    # ==========================================================================
    {
        "name": "lens_section",
        "description": "Lens section (Z=125-147) with focus lens and joining M4 hole",
        "bounds": {
            "x_min": 0.0, "x_max": 17.5,
            "y_min": -86.0, "y_max": -20.0,
            "z_min": 125.0, "z_max": 147.0,
        },
        "exclude": [],
        "include": [],
    },
    
    # ==========================================================================
    # SECTION 3: Upper DMD Housing (Z=147-166)
    # Contains: Upper housing, joining M4 hole at Z=150
    # ==========================================================================
    {
        "name": "dmd_upper",
        "description": "Upper DMD housing (Z=147-166) with joining M4 hole",
        "bounds": {
            "x_min": 0.0, "x_max": 17.5,
            "y_min": -86.0, "y_max": -20.0,
            "z_min": 147.0, "z_max": 166.0,
        },
        "exclude": [],
        "include": [],
    },
    
    # ==========================================================================
    # SECTION 4: Side Extension (Y=-40 to -20, Z=95-166)
    # The "island" on the side of the DMD region
    # ==========================================================================
    {
        "name": "side_extension",
        "description": "Side extension (Y=-40 to -20, Z=95-166)",
        "bounds": {
            "x_min": 0.0, "x_max": 17.5,
            "y_min": -40.0, "y_max": -20.0,
            "z_min": 95.0, "z_max": 166.0,
        },
        "exclude": [],
        "include": [],
    },
    
    # ==========================================================================
    # SECTION 5: Optics Region (Z=15-95)
    # Contains: Entrance, slit, collimator, grating, etc.
    # ==========================================================================
    {
        "name": "optics",
        "description": "Optics region (Z=15-95) - entrance, slit, collimator, grating",
        "bounds": {
            "x_min": -50.0, "x_max": 50.0,
            "y_min": -105.0, "y_max": 25.0,
            "z_min": 15.0, "z_max": 95.0,
        },
        "exclude": [],
        "include": [],
    },
    
    # ==========================================================================
    # SECTION 6: Upper Housing (Z=166-235)
    # Contains: Relay lens, condenser, sensor area
    # ==========================================================================
    {
        "name": "upper_housing",
        "description": "Upper housing (Z=166-235) - relay, condenser, sensor",
        "bounds": {
            "x_min": -50.0, "x_max": 50.0,
            "y_min": -105.0, "y_max": 25.0,
            "z_min": 166.0, "z_max": 235.0,
        },
        "exclude": [],
        "include": [],
    },
]

# -----------------------------------------------------------------------------
# OUTPUT SETTINGS
# -----------------------------------------------------------------------------
OUTPUT_PREFIX = "top"


# ==============================================================================
# CORE FUNCTIONS
# ==============================================================================

def load_top_half(input_file):
    """Load the top half STEP file."""
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    print(f"Loading: {input_file}")
    top = Part.Shape()
    top.read(input_file)
    print(f"  Volume: {top.Volume:.2f} mm³")
    bb = top.BoundBox
    print(f"  BBox: X={bb.XMin:.1f}-{bb.XMax:.1f}, Y={bb.YMin:.1f}-{bb.YMax:.1f}, Z={bb.ZMin:.1f}-{bb.ZMax:.1f}")
    return top


def trim_y(top, y_limit=-86.0):
    """Trim model to Y limit (default -86 to match printing_model_5.stl)."""
    print(f"Trimming to Y={y_limit}...")
    trim_box = Part.makeBox(500, 500, 500, FreeCAD.Vector(-250, y_limit, -50))
    return top.common(trim_box)


def make_box(bounds):
    """Create a box from bounds dictionary."""
    return Part.makeBox(
        bounds["x_max"] - bounds["x_min"],
        bounds["y_max"] - bounds["y_min"],
        bounds["z_max"] - bounds["z_min"],
        FreeCAD.Vector(bounds["x_min"], bounds["y_min"], bounds["z_min"])
    )


def build_section(top, section_def):
    """Build a section from its definition."""
    print(f"\nBuilding section: {section_def['name']}")
    print(f"  {section_def['description']}")
    print(f"  Bounds: X={section_def['bounds']['x_min']}-{section_def['bounds']['x_max']}, "
          f"Y={section_def['bounds']['y_min']}-{section_def['bounds']['y_max']}, "
          f"Z={section_def['bounds']['z_min']}-{section_def['bounds']['z_max']}")
    
    # Main box
    main_box = make_box(section_def["bounds"])
    section_piece = top.common(main_box)
    
    # Apply exclusions
    for excl in section_def.get("exclude", []):
        excl_box = make_box(excl)
        section_piece = section_piece.cut(excl_box)
        print(f"  Excluded: {excl}")
    
    # Apply inclusions
    for incl in section_def.get("include", []):
        incl_box = make_box(incl)
        section_piece = section_piece.fuse(incl_box)
        print(f"  Included: {incl}")
    
    print(f"  Volume: {section_piece.Volume:.2f} mm³")
    return section_piece


def print_m4_holes(shape, label):
    """Print M4 holes in a shape."""
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


def check_overlaps(sections):
    """Check for overlaps between sections."""
    print("\n" + "=" * 50)
    print("OVERLAP CHECK")
    print("=" * 50)
    
    section_shapes = {s["name"]: s["shape"] for s in sections}
    names = list(section_shapes.keys())
    
    has_overlap = False
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            name1, name2 = names[i], names[j]
            shape1, shape2 = section_shapes[name1], section_shapes[name2]
            
            try:
                intersection = shape1.common(shape2)
                if intersection.Volume > 0.01:  # Small tolerance
                    print(f"  OVERLAP: {name1} ∩ {name2} = {intersection.Volume:.2f} mm³")
                    has_overlap = True
            except:
                pass
    
    if not has_overlap:
        print("  No significant overlaps detected.")
    return has_overlap


def check_coverage(sections, trimmed_top):
    """Check if sections cover the trimmed model."""
    print("\n" + "=" * 50)
    print("COVERAGE CHECK")
    print("=" * 50)
    
    # Fuse all sections
    all_sections = None
    for s in sections:
        if all_sections is None:
            all_sections = s["shape"]
        else:
            all_sections = all_sections.fuse(s["shape"])
    
    if all_sections is None:
        print("  No sections to check.")
        return
    
    # Check difference
    missing = trimmed_top.cut(all_sections)
    extra = all_sections.cut(trimmed_top)
    
    print(f"  Trimmed model volume: {trimmed_top.Volume:.2f} mm³")
    print(f"  Sections fused volume: {all_sections.Volume:.2f} mm³")
    print(f"  Missing volume: {missing.Volume:.2f} mm³")
    print(f"  Extra volume: {extra.Volume:.2f} mm³")
    
    if missing.Volume < 1.0 and extra.Volume < 1.0:
        print("  ✓ Complete coverage (within tolerance)")
    else:
        print("  ⚠ Coverage gaps or excess detected")


def export_section(section, output_dir, prefix):
    """Export a section to STEP file."""
    if section["shape"] is None:
        return
    filename = f"{prefix}_{section['name']}.step"
    filepath = os.path.join(OUTPUT_DIR, filename)
    section["shape"].exportStep(filepath)
    print(f"Exported: {filename} ({section['shape'].Volume:.2f} mm³)")


def print_summary(sections):
    """Print a summary table of all sections."""
    print("\n" + "=" * 80)
    print("SECTION SUMMARY")
    print("=" * 80)
    print(f"{'Name':<20} {'Volume (mm³)':>15} {'X Range':<20} {'Y Range':<20} {'Z Range':<20}")
    print("-" * 80)
    
    total = 0
    for s in sections:
        if s["shape"] is None:
            continue
        vol = s["shape"].Volume
        bb = s["shape"].BoundBox
        total += vol
        x_range = f"{bb.XMin:.1f}-{bb.XMax:.1f}"
        y_range = f"{bb.YMin:.1f}-{bb.YMax:.1f}"
        z_range = f"{bb.ZMin:.1f}-{bb.ZMax:.1f}"
        print(f"{s['name']:<20} {vol:>15.2f} {x_range:<20} {y_range:<20} {z_range:<20}")
    
    print("-" * 80)
    print(f"{'TOTAL':<20} {total:>15.2f}")


def main():
    print("=" * 70)
    print("SPECTROMETER TOP HALF - Complete Section Control System")
    print("=" * 70)
    print(f"Sections defined: {len(SECTIONS)}")
    print()
    
    # Load top half
    top = load_top_half(INPUT_FILE)
    
    # Trim Y if needed
    if TRIM_Y_TO_86:
        top = trim_y(top)
    
    # Build all sections
    for section in SECTIONS:
        section["shape"] = build_section(top, section)
    
    # Print M4 holes for each section
    for section in SECTIONS:
        if section["shape"] is not None:
            print_m4_holes(section["shape"], section["name"])
    
    # Check overlaps
    if VALIDATE_NO_OVERLAP:
        check_overlaps(SECTIONS)
    
    # Check coverage
    if VALIDATE_COMPLETE:
        check_coverage(SECTIONS, top)
    
    # Print summary
    print_summary(SECTIONS)
    
    # Export all sections
    print("\n" + "=" * 50)
    print("EXPORTING SECTIONS")
    print("=" * 50)
    for section in SECTIONS:
        export_section(section, OUTPUT_DIR, OUTPUT_PREFIX)
    
    print("\nDone!")


def load_top_half(input_file):
    """Load the top half STEP file."""
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    print(f"Loading: {input_file}")
    top = Part.Shape()
    top.read(input_file)
    print(f"  Volume: {top.Volume:.2f} mm³")
    bb = top.BoundBox
    print(f"  BBox: X={bb.XMin:.1f}-{bb.XMax:.1f}, Y={bb.YMin:.1f}-{bb.YMax:.1f}, Z={bb.ZMin:.1f}-{bb.ZMax:.1f}")
    return top


def trim_y(top, y_limit=-86.0):
    """Trim model to Y limit (default -86 to match printing_model_5.stl)."""
    print(f"Trimming to Y={y_limit}...")
    trim_box = Part.makeBox(500, 500, 500, FreeCAD.Vector(-250, y_limit, -50))
    return top.common(trim_box)


def make_box(bounds):
    """Create a box from bounds dictionary."""
    return Part.makeBox(
        bounds["x_max"] - bounds["x_min"],
        bounds["y_max"] - bounds["y_min"],
        bounds["z_max"] - bounds["z_min"],
        FreeCAD.Vector(bounds["x_min"], bounds["y_min"], bounds["z_min"])
    )


# Call main directly (FreeCADCmd doesn't set __name__ == "__main__")
try:
    main()
except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)