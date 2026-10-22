#!/usr/bin/env python3
"""
================================================================================
SPLIT TOP HALF - Main Control Script for Spectrometer Top Half Sectioning
================================================================================

A flexible, well-documented script for splitting the spectrometer top half
(support_top_half.step) into configurable sections.

FEATURES:
- Multiple predefined splitting strategies
- Custom cut boundaries with full control
- Automatic Y-trim to match printing_model_5.stl (Y=-86)
- Volume preservation verification
- M4 hole detection and reporting
- Clean STEP file export

USAGE:
    # Run with default strategy (no_lens)
    & "C:\Program Files\FreeCAD 1.1\bin\FreeCADCmd.exe" split_top_half_main.py

    # To change strategy, edit the STRATEGY variable in the CONFIG section below
    # Strategies: "full_housing", "no_lens", "aperture_only", "focused", "three_way", "custom"

================================================================================
"""

import sys
import os

# Add FreeCAD to path
sys.path.append(r'C:\Program Files\FreeCAD 1.1\bin')
import FreeCAD
import Part


# ==============================================================================
# USER CONFIGURATION - EDIT THESE VALUES TO CONTROL THE SPLIT
# ==============================================================================

# -----------------------------------------------------------------------------
# INPUT/OUTPUT PATHS
# -----------------------------------------------------------------------------
INPUT_FILE = r"C:\Users\Juan\Documents\GitHub\2D_Optics\3D Raman DMD spectrometer\support_top_half.step"
OUTPUT_DIR = r"C:\Users\Juan\Documents\GitHub\2D_Optics\3D Raman DMD spectrometer"
OUTPUT_PREFIX = "top"

# -----------------------------------------------------------------------------
# SPLITTING STRATEGY
# Choose one: "full_housing", "no_lens", "aperture_only", "focused", "three_way", "custom"
# -----------------------------------------------------------------------------
STRATEGY = "no_lens"

# -----------------------------------------------------------------------------
# Y-TRIM: Trim model to Y=-86 (matches printing_model_5.stl, removes material below Y=-86)
# -----------------------------------------------------------------------------
TRIM_Y_TO_86 = True

# -----------------------------------------------------------------------------
# CUSTOM BOUNDARIES (only used when STRATEGY = "custom")
# All coordinates in mm, global coordinate system
# -----------------------------------------------------------------------------
CUSTOM_CONFIG = {
    # Main DMD piece bounding box
    "dmd_box": {
        "x_min": 0.0,
        "x_max": 17.5,
        "y_min": -86.0,
        "y_max": -20.0,
        "z_min": 95.0,
        "z_max": 166.0,
    },
    
    # Regions to REMOVE from DMD piece (list of boxes)
    # Example: remove lens section
    "exclude_regions": [
        # {"x_min": 0, "x_max": 17.5, "y_min": -86, "y_max": -20, "z_min": 125, "z_max": 147}
    ],
    
    # Additional regions to ADD to DMD piece (list of boxes)
    # Example: add side material
    "include_regions": [
        # {"x_min": 0, "x_max": 17.5, "y_min": -40, "y_max": -20, "z_min": 95, "z_max": 166}
    ],
}

# -----------------------------------------------------------------------------
# PREDEFINED STRATEGY DEFINITIONS
# Each strategy defines a base box and optional exclusions/inclusions
# -----------------------------------------------------------------------------
STRATEGIES = {
    "full_housing": {
        "description": "Full DMD housing (Y=-86 to -20, Z=95-166)",
        "dmd_box": {"x_min": 0, "x_max": 17.5, "y_min": -86, "y_max": -20, "z_min": 95, "z_max": 166},
        "exclude": [],
        "include": [],
    },
    
    "no_lens": {
        "description": "DMD housing WITHOUT lens section (removes Z=125-147)",
        "dmd_box": {"x_min": 0, "x_max": 17.5, "y_min": -86, "y_max": -20, "z_min": 95, "z_max": 166},
        "exclude": [
            {"x_min": 0, "x_max": 17.5, "y_min": -86, "y_max": -20, "z_min": 125, "z_max": 147}
        ],
        "include": [],
    },
    
    "aperture_only": {
        "description": "Only aperture area (Z=95-125)",
        "dmd_box": {"x_min": 0, "x_max": 17.5, "y_min": -86, "y_max": -20, "z_min": 95, "z_max": 125},
        "exclude": [],
        "include": [],
    },
    
    "focused": {
        "description": "Focused DMD region (Y=-86 to -40, Z=95-160)",
        "dmd_box": {"x_min": 0, "x_max": 17.5, "y_min": -86, "y_max": -40, "z_min": 95, "z_max": 160},
        "exclude": [],
        "include": [],
    },
    
    "three_way": {
        "description": "Three pieces: optics (Z=15-95), DMD (Z=95-160), upper (Z=160-235)",
        "special": "three_way",
    },
}


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


def apply_strategy(top, strategy_name, custom_config=None):
    """Apply the selected splitting strategy."""
    
    if strategy_name == "custom":
        if custom_config is None:
            raise ValueError("custom_config required for 'custom' strategy")
        
        dmd_box = make_box(custom_config["dmd_box"])
        dmd_piece = top.common(dmd_box)
        
        # Apply exclusions
        for excl in custom_config.get("exclude_regions", []):
            excl_box = make_box(excl)
            dmd_piece = dmd_piece.cut(excl_box)
        
        # Apply inclusions
        for incl in custom_config.get("include_regions", []):
            incl_box = make_box(incl)
            dmd_piece = dmd_piece.fuse(incl_box)
        
        # Return the cut box for rest piece calculation
        cut_box = dmd_box
        for excl in custom_config.get("exclude_regions", []):
            excl_box = make_box(excl)
            cut_box = cut_box.cut(excl_box)
        
        return dmd_piece, cut_box
    
    elif strategy_name == "three_way":
        return None, None  # Handled separately
    
    else:
        if strategy_name not in STRATEGIES:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        strat = STRATEGIES[strategy_name]
        print(f"Strategy: {strategy_name} - {strat['description']}")
        
        dmd_box = make_box(strat["dmd_box"])
        dmd_piece = top.common(dmd_box)
        
        # Apply exclusions
        for excl in strat.get("exclude", []):
            excl_box = make_box(excl)
            dmd_piece = dmd_piece.cut(excl_box)
        
        # Apply inclusions
        for incl in strat.get("include", []):
            incl_box = make_box(incl)
            dmd_piece = dmd_piece.fuse(incl_box)
        
        # Build cut box for rest piece (same as dmd_box minus exclusions)
        cut_box = dmd_box
        for excl in strat.get("exclude", []):
            excl_box = make_box(excl)
            cut_box = cut_box.cut(excl_box)
        
        return dmd_piece, cut_box


def three_way_split(top):
    """Split into three pieces: optics, DMD, upper housing."""
    print("Performing three-way split...")
    
    # Optics: Z=15-95
    optics_box = Part.makeBox(500, 500, 95, FreeCAD.Vector(-250, -250, 0))
    optics = top.common(optics_box)
    
    # DMD: Z=95-160
    dmd_box = Part.makeBox(500, 500, 65, FreeCAD.Vector(-250, -250, 95))
    dmd = top.common(dmd_box)
    
    # Upper: Z=160-235
    upper_box = Part.makeBox(500, 500, 75, FreeCAD.Vector(-250, -250, 160))
    upper = top.common(upper_box)
    
    return optics, dmd, upper


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


def export_piece(piece, name, output_dir, prefix):
    """Export a piece to STEP file."""
    if piece is None:
        return
    filename = f"{prefix}_{name}.step"
    filepath = os.path.join(output_dir, filename)
    piece.exportStep(filepath)
    print(f"Exported: {filename} ({piece.Volume:.2f} mm³)")


def main():
    print("=" * 70)
    print("SPLIT TOP HALF - Spectrometer Top Half Sectioning")
    print("=" * 70)
    print(f"Strategy: {STRATEGY}")
    if STRATEGY in STRATEGIES:
        print(f"Description: {STRATEGIES[STRATEGY]['description']}")
    print()
    
    # Load top half
    top = load_top_half(INPUT_FILE)
    
    # Trim Y if needed
    if TRIM_Y_TO_86:
        top = trim_y(top)
    
    # Apply strategy
    if STRATEGY == "three_way":
        optics, dmd, upper = three_way_split(top)
        
        print_m4_holes(optics, "Optics")
        print_m4_holes(dmd, "DMD")
        print_m4_holes(upper, "Upper")
        
        pieces = {
            "optics": optics,
            "dmd": dmd,
            "upper": upper,
        }
    else:
        dmd_piece, cut_box = apply_strategy(top, STRATEGY, CUSTOM_CONFIG)
        
        # Rest piece
        rest_piece = top.cut(cut_box)
        
        print_m4_holes(dmd_piece, "DMD")
        print_m4_holes(rest_piece, "Rest")
        
        pieces = {
            "dmd": dmd_piece,
            "rest": rest_piece,
        }
    
    # Verify volume
    total_volume = sum(p.Volume for p in pieces.values() if p is not None)
    print(f"\nTotal volume: {total_volume:.2f} mm³")
    
    # Export
    for name, piece in pieces.items():
        export_piece(piece, name, OUTPUT_DIR, OUTPUT_PREFIX)
    
    print("\nDone!")


# Call main directly (FreeCADCmd doesn't set __name__ == "__main__")
try:
    main()
except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)