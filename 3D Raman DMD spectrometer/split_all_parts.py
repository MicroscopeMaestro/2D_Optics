#!/usr/bin/env python3
"""
================================================================================
SPLIT ALL PARTS - Export Each Component as Separate STEP File
================================================================================

This script loads the original STEP file and exports each component
as a separate STEP file.

USAGE:
    & "C:\Program Files\FreeCAD 1.1\bin\FreeCADCmd.exe" split_all_parts.py

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
INPUT_FILE = r"C:\Users\Juan\Documents\GitHub\2D_Optics\3D Raman DMD spectrometer\support_top_half.step"
OUTPUT_DIR = r"C:\Users\Juan\Documents\GitHub\2D_Optics\3D Raman DMD spectrometer\split_parts"
TRIM_Y_TO_86 = True

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("=" * 70)
    print("SPLIT ALL PARTS - Export Each Component as STEP")
    print("=" * 70)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load the original STEP file
    print(f"Loading: {INPUT_FILE}")
    top = Part.Shape()
    top.read(INPUT_FILE)
    print(f"  Loaded: {top.Volume:.2f} mm³")
    print(f"  ShapeType: {top.ShapeType}")
    print(f"  Solids: {len(top.Solids)}")
    
    # Trim Y if needed
    if TRIM_Y_TO_86:
        trim_box = Part.makeBox(500, 500, 500, FreeCAD.Vector(-250, -86, -50))
        top = top.common(trim_box)
        print(f"Trimmed to Y=-86: {top.Volume:.2f} mm³")
    
    # The top shape is a Compound containing multiple solids
    # Let's extract each solid as a separate part
    print(f"\nExtracting {len(top.Solids)} solids...")
    
    for i, solid in enumerate(top.Solids):
        # Create a compound with just this solid
        piece = Part.Compound([solid])
        
        # Export
        filename = f"part_{i:03d}.step"
        filepath = os.path.join(OUTPUT_DIR, filename)
        piece.exportStep(filepath)
        print(f"  Part {i:03d}: {solid.Volume:.2f} mm³ -> {filename}")
    
    # Also try to split by the original component structure
    # The original file might have named components
    print("\nDone!")
    print(f"All parts exported to: {OUTPUT_DIR}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)