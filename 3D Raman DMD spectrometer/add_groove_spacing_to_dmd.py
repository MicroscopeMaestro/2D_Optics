#!/usr/bin/env python3
"""
================================================================================
DMD PIECE - Add Groove Spacing to DMD Top Support
================================================================================

This script takes the DMD aperture piece and adds "spacing for grooves"
at the Y edges (Y=-86 and Y=-20). This creates tongue-and-groove joints
where the DMD piece has tongues that fit into grooves on the rest piece.

The groove spacing is added as cutouts/recesses on the DMD piece edges.
================================================================================
"""

import sys
import os

# Add FreeCAD to path
sys.path.append(r'C:\Program Files\FreeCAD 1.1\bin')
import FreeCAD
import Part


# ==============================================================================
# CONFIGURATION
# ==============================================================================

INPUT_DMD = r"C:\Users\Juan\Documents\GitHub\2D_Optics\3D Raman DMD spectrometer\top_dmd_aperture.step"
OUTPUT_DMD = r"C:\Users\Juan\Documents\GitHub\2D_Optics\3D Raman DMD spectrometer\top_dmd_aperture_grooved.step"

# Groove spacing parameters
GROOVE_WIDTH = 3.0      # Width of each groove spacing (mm)
GROOVE_DEPTH = 2.0      # Depth of groove spacing into DMD piece (mm)
GROOVE_SPACING = 10.0   # Center-to-center spacing between grooves (mm)
GROOVE_HEIGHT = 5.0     # Height of each groove along Z (mm)

# Which edges to add groove spacing
ADD_AT_Y_MINUS_86 = True   # Bottom edge (Y=-86)
ADD_AT_Y_MINUS_20 = True   # Top edge (Y=-20)

# Z range for grooves (within DMD piece)
GROOVE_Z_START = 98.0
GROOVE_Z_END = 136.0


# ==============================================================================
# CORE FUNCTIONS
# ==============================================================================

def load_shape(filepath):
    """Load a STEP file."""
    shape = Part.Shape()
    shape.read(filepath)
    return shape


def create_groove_spacing_boxes(dmd_piece):
    """Create boxes for groove spacing cutouts at Y edges."""
    bb = dmd_piece.BoundBox
    x_min, x_max = bb.XMin, bb.XMax
    z_min = max(GROOVE_Z_START, bb.ZMin)
    z_max = min(GROOVE_Z_END, bb.ZMax)
    
    groove_boxes = []
    
    # Calculate number of grooves
    z_range = z_max - z_min
    num_grooves = int(z_range / GROOVE_SPACING) + 1
    
    for i in range(num_grooves):
        z_center = z_min + i * GROOVE_SPACING
        if z_center + GROOVE_HEIGHT/2 > z_max:
            break
            
        z_start = z_center - GROOVE_HEIGHT/2
        z_end = z_center + GROOVE_HEIGHT/2
        
        # Groove at Y=-86 (bottom edge)
        if ADD_AT_Y_MINUS_86:
            box = Part.makeBox(
                GROOVE_DEPTH,           # X depth into piece
                GROOVE_WIDTH,           # Y width of groove
                GROOVE_HEIGHT,          # Z height
                FreeCAD.Vector(0, -86 - GROOVE_WIDTH/2, z_start)
            )
            groove_boxes.append(box)
        
        # Groove at Y=-20 (top edge)
        if ADD_AT_Y_MINUS_20:
            box = Part.makeBox(
                GROOVE_DEPTH,           # X depth into piece
                GROOVE_WIDTH,           # Y width of groove
                GROOVE_HEIGHT,          # Z height
                FreeCAD.Vector(0, 20 - GROOVE_DEPTH, z_start)
            )
            groove_boxes.append(box)
    
    return groove_boxes


def add_groove_spacing(dmd_piece):
    """Add groove spacing cutouts to DMD piece."""
    print(f"Adding groove spacing to DMD piece...")
    print(f"  Groove width: {GROOVE_WIDTH}mm, depth: {GROOVE_DEPTH}mm")
    print(f"  Groove height: {GROOVE_HEIGHT}mm, spacing: {GROOVE_SPACING}mm")
    print(f"  Z range: {GROOVE_Z_START}-{GROOVE_Z_END}")
    
    groove_boxes = create_groove_spacing_boxes(dmd_piece)
    print(f"  Created {len(groove_boxes)} groove spacing boxes")
    
    # Apply all groove cutouts
    grooved_piece = dmd_piece
    for i, box in enumerate(groove_boxes):
        grooved_piece = grooved_piece.cut(box)
        if i < 3:  # Print first few
            bb = box.BoundBox
            print(f"  Groove {i}: X={bb.XMin:.1f}-{bb.XMax:.1f}, Y={bb.YMin:.1f}-{bb.YMax:.1f}, Z={bb.ZMin:.1f}-{bb.ZMax:.1f}")
    
    return grooved_piece


def main():
    print("=" * 70)
    print("DMD PIECE - Add Groove Spacing")
    print("=" * 70)
    
    # Load DMD piece
    print(f"Loading: {INPUT_DMD}")
    dmd = load_shape(INPUT_DMD)
    print(f"  Volume: {dmd.Volume:.2f} mm³")
    bb = dmd.BoundBox
    print(f"  BBox: X={bb.XMin:.1f}-{bb.XMax:.1f}, Y={bb.YMin:.1f}-{bb.YMax:.1f}, Z={bb.ZMin:.1f}-{bb.ZMax:.1f}")
    
    # Add groove spacing
    grooved_dmd = add_groove_spacing(dmd)
    
    print(f"\nOriginal volume: {dmd.Volume:.2f} mm³")
    print(f"Grooved volume: {grooved_dmd.Volume:.2f} mm³")
    print(f"Material removed: {dmd.Volume - grooved_dmd.Volume:.2f} mm³")
    
    # Export
    print(f"\nExporting to: {OUTPUT_DMD}")
    grooved_dmd.exportStep(OUTPUT_DMD)
    print("Done!")


def load_shape(filepath):
    """Load a STEP file."""
    shape = Part.Shape()
    shape.read(filepath)
    return shape


# Call main directly (FreeCADCmd doesn't set __name__ == "__main__")
try:
    main()
except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)