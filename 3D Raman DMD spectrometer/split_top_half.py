#!/usr/bin/env python3
"""
================================================================================
SPLIT TOP HALF - Flexible Spectrometer Top Half Sectioning
================================================================================

This script splits the spectrometer top half (support_top_half.step) into
configurable sections. It provides multiple splitting strategies and allows
fine-tuning of cut boundaries.

USAGE:
    & "C:\Program Files\FreeCAD 1.1\bin\FreeCADCmd.exe" split_top_half.py

CONFIGURATION:
    Modify the CONFIG dictionary below to choose splitting strategy and parameters.

OUTPUT:
    Creates STEP files for each section in the same directory.

================================================================================
"""

import sys
import os

# Add FreeCAD to path
sys.path.append(r'C:\Program Files\FreeCAD 1.1\bin')
import FreeCAD
import Part


# ==============================================================================
# CONFIGURATION - Modify these parameters to control the split
# ==============================================================================

CONFIG = {
    # Input/Output
    "input_file": r"C:\Users\Juan\Documents\GitHub\2D_Optics\3D Raman DMD spectrometer\support_top_half.step",
    "output_dir": r"C:\Users\Juan\Documents\GitHub\2D_Optics\3D Raman DMD spectrometer",
    
    # -------------------------------------------------------------------------
    # SPLITTING STRATEGY - Choose one:
    # -------------------------------------------------------------------------
    # "full_housing"      - Full DMD housing (Y=-86 to -20, Z=95-166) as one piece
    # "no_lens"           - DMD housing WITHOUT lens section (Z=125-147 removed)
    # "aperture_only"     - Only aperture area (Z=95-125)
    # "focused"           - Focused DMD region (Y=-86 to -40, Z=95-160)
    # "three_way"         - Three pieces: optics, DMD, upper housing
    # "custom"            - Use custom cut boundaries below
    # -------------------------------------------------------------------------
    "strategy": "no_lens",
    
    # -------------------------------------------------------------------------
    # CUSTOM CUT BOUNDARIES (used when strategy="custom")
    # All coordinates in mm, relative to global coordinate system
    # -------------------------------------------------------------------------
    "custom": {
        # DMD piece boundaries
        "dmd_x_min": 0.0,
        "dmd_x_max": 17.5,
        "dmd_y_min": -86.0,
        "dmd_y_max": -20.0,
        "dmd_z_min": 95.0,
        "dmd_z_max": 166.0,
        
        # Regions to REMOVE from DMD piece (list of boxes)
        "exclude_regions": [
            # Example: {"x_min": 0, "x_max": 17.5, "y_min": -86, "y_max": -20, "z_min": 125, "z_max": 147}
        ],
        
        # Additional regions to ADD to DMD piece (list of boxes)
        "include_regions": [
            # Example: {"x_min": 0, "x_max": 17.5, "y_min": -40, "y_max": -20, "z_min": 95, "z_max": 166}
        ],
    },
    
    # -------------------------------------------------------------------------
    # GROOVE/EDGE SUPPORTS
    # Add vertical edge supports at DMD housing boundaries
    # -------------------------------------------------------------------------
    "add_edge_supports": True,
    "support_width": 5.0,  # Width of edge support in Y direction (mm)
    
    # -------------------------------------------------------------------------
    # Y-TRIM
    # Trim the model to match printing_model_5.stl (Y=-86 instead of Y=-105)
    # -------------------------------------------------------------------------
    "trim_y_to_86": True,
    
    # -------------------------------------------------------------------------
    # OUTPUT FILENAMES
    # -------------------------------------------------------------------------
    "output_prefix": "top",
    "dmd_suffix": "_dmd",
    "rest_suffix": "_rest",
}

# ==============================================================================
# PREDEFINED STRATEGIES
# ==============================================================================

STRATEGIES = {
    "full_housing": {
        "description": "Full DMD housing as one piece (Y=-86 to -20, Z=95-166)",
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
# MAIN FUNCTIONS
# ==============================================================================

def load_top_half(config):
    """Load the top half STEP file."""
    input_file = config["input_file"]
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    print(f"Loading: {input_file}")
    top = Part.Shape()
    top.read(input_file)
    print(f"  Volume: {top.Volume:.2f} mm³")
    bb = top.BoundBox
    print(f"  BBox: X={bb.XMin:.1f}-{bb.XMax:.1f}, Y={bb.YMin:.1f}-{bb.YMax:.1f}, Z={bb.ZMin:.1f}-{bb.ZMax:.1f}")
    return top


def trim_y(top, config):
    """Trim model to Y=-86 (match printing_model_5.stl)."""
    if not config.get("trim_y_to_86", True):
        return top
    
    print("Trimming to Y=-86...")
    trim_box = Part.makeBox(500, 200, 500, FreeCAD.Vector(-250, -86, -50))
    return top.common(trim_box)


def make_box(bounds):
    """Create a box from bounds dictionary."""
    return Part.makeBox(
        bounds["x_max"] - bounds["x_min"],
        bounds["y_max"] - bounds["y_min"],
        bounds["z_max"] - bounds["z_min"],
        FreeCAD.Vector(bounds["x_min"], bounds["y_min"], bounds["z_min"])
    )


def apply_strategy(top, config):
    """Apply the selected splitting strategy."""
    strategy_name = config["strategy"]
    
    if strategy_name == "custom":
        # Use custom boundaries
        dmd_box = make_box(config["custom"]["dmd_box"])
        dmd_piece = top.common(dmd_box)
        
        # Apply exclusions
        for excl in config["custom"].get("exclude_regions", []):
            excl_box = make_box(excl)
            dmd_piece = dmd_piece.cut(excl_box)
        
        # Apply inclusions
        for incl in config["custom"].get("include_regions", []):
            incl_box = make_box(incl)
            dmd_piece = dmd_piece.fuse(incl_box)
        
        return dmd_piece
    
    elif strategy_name == "three_way":
        # Special three-way split
        return None  # Handled separately
    
    else:
        # Use predefined strategy
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
        
        return dmd_piece


def add_edge_supports(dmd_piece, top, config):
    """Edge supports are already naturally part of the DMD housing geometry.
    This function is kept for compatibility but does nothing.
    """
    return dmd_piece


def three_way_split(top, config):
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
    for i, face in enumerate(shape.Faces):
        surf = face.Surface
        if hasattr(surf, 'Radius') and surf.Radius > 1.5 and surf.Radius < 3.0:
            com = face.CenterOfMass
            print(f"  Face {i}: Radius={surf.Radius:.2f} at X={com.x:.2f}, Y={com.y:.2f}, Z={com.z:.2f}")


def export_pieces(pieces, config):
    """Export pieces to STEP files."""
    output_dir = config["output_dir"]
    prefix = config["output_prefix"]
    
    for name, piece in pieces.items():
        if piece is None:
            continue
        filename = f"{prefix}_{name}{config.get('dmd_suffix', '')}.step"
        filepath = os.path.join(output_dir, filename)
        piece.exportStep(filepath)
        print(f"Exported: {filename} ({piece.Volume:.2f} mm³)")


def main():
    print("=" * 70)
    print("SPLIT TOP HALF - Spectrometer Top Half Sectioning")
    print("=" * 70)
    print(f"Strategy: {CONFIG['strategy']}")
    if CONFIG['strategy'] in STRATEGIES:
        print(f"Description: {STRATEGIES[CONFIG['strategy']]['description']}")
    print()
    
    # Load top half
    top = load_top_half(CONFIG)
    
    # Trim Y if needed
    top = trim_y(top, CONFIG)
    
    # Apply strategy
    if CONFIG["strategy"] == "three_way":
        optics, dmd, upper = three_way_split(top, CONFIG)
        
        print_m4_holes(optics, "Optics")
        print_m4_holes(dmd, "DMD")
        print_m4_holes(upper, "Upper")
        
        pieces = {
            "optics": optics,
            "dmd": dmd,
            "upper": upper,
        }
    else:
        # Get DMD piece
        dmd_piece = apply_strategy(top, CONFIG)
        
        # Add edge supports
        dmd_piece = add_edge_supports(dmd_piece, top, CONFIG)
        
        # Rest piece - cut with the same box used for DMD (before exclusions)
        if CONFIG["strategy"] == "custom":
            cut_box = make_box(CONFIG["custom"]["dmd_box"])
        else:
            cut_box = make_box(STRATEGIES[CONFIG["strategy"]]["dmd_box"])
        
        # Apply same exclusions to cut_box for rest piece
        rest_cut_box = cut_box
        if CONFIG["strategy"] == "custom":
            for excl in CONFIG["custom"].get("exclude_regions", []):
                excl_box = make_box(excl)
                rest_cut_box = rest_cut_box.cut(excl_box)
        else:
            for excl in STRATEGIES[CONFIG["strategy"]].get("exclude", []):
                excl_box = make_box(excl)
                rest_cut_box = rest_cut_box.cut(excl_box)
        
        rest_piece = top.cut(rest_cut_box)
        
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
    export_pieces(pieces, CONFIG)
    
    print("\nDone!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)